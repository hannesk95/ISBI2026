from model import DenseNet3DMonai, EfficientNet3DMonai, ResNet3DMonai
import torch
from dataset import RnCDataset
from torch.utils.data import DataLoader
from utils import seed_everything, worker_init_fn
from glob import glob
import mlflow
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import time
import uuid
from scipy.special import softmax
from losses import RnCLoss
from monai.data.utils import pad_list_data_collate

EPOCHS = 400
BATCH_SIZE = 32
WARMUP_EPOCHS = 50
INITIAL_LR = 0.0
TARGET_LR = 0.001
SEED = 42

def main(dataset: str, backbone: str):    

    identifier = str(uuid.uuid4())
    seed_everything(SEED)
    mlflow.log_param("train_strategy", "rank-n-contrastive")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("warmup_epochs", WARMUP_EPOCHS)
    mlflow.log_param("initial_lr", INITIAL_LR)
    mlflow.log_param("target_lr", TARGET_LR)
    mlflow.log_param("seed", SEED)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("backbone", backbone)   

    match dataset:
        case "soft_tissue_tumors":
            data = glob("./data/OrdinalClassificationSarcoma/dataset_final/*.pt")
            labels = [int(path.split("/")[-1].split("_")[-3][1]) for path in data]
            n_classes = 4
        case "lung_nodules":
            data = glob("./data/OrdinalClassificationLung/dataset_final/*.pt")
            labels = [int(path.split("/")[-1].split("_")[-1][0]) for path in data] 
            labels = np.array(labels) - np.min(labels)  # Ensure labels start from 0
            labels = labels.tolist()  # Convert back to list for compatibility
            n_classes = 5

        case "adni":
            # CN → SMC → EMCI → LMCI → AD
            data = glob("./data/OrdinalClassificationADNI/dataset_final/*2mm*.pt")
            patient_ids = [f.split("/")[-1][:10] for f in data]
            patient_ids = list(set(patient_ids))            

            train_ids, val_ids = train_test_split(patient_ids, test_size=0.3, random_state=SEED)
            val_ids, test_ids = train_test_split(val_ids, test_size=0.5, random_state=SEED)

            assert set(train_ids).isdisjoint(val_ids)
            assert set(train_ids).isdisjoint(test_ids)
            assert set(val_ids).isdisjoint(test_ids)

            train_data = []
            val_data = []
            test_data = []
            for f in data:
                pid = f.split("/")[-1][:10]
                if pid in train_ids:
                    train_data.append(f)
                elif pid in val_ids:
                    val_data.append(f)
                elif pid in test_ids:
                    test_data.append(f)
            
            train_labels = [f.split("/")[-1].split("_")[-3].replace(".nii.gz", "") for f in train_data]
            val_labels = [f.split("/")[-1].split("_")[-3].replace(".nii.gz", "") for f in val_data]
            test_labels = [f.split("/")[-1].split("_")[-3].replace(".nii.gz", "") for f in test_data]
            label_mapping = {"CN": 0, "SMC": 1, "EMCI": 2, "LMCI": 3, "AD": 4}
            train_labels = [label_mapping[label] for label in train_labels]
            val_labels = [label_mapping[label] for label in val_labels]
            test_labels = [label_mapping[label] for label in test_labels]
            n_classes = 5

        case _:
            raise ValueError("Dataset not implemented!")
            
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=SEED)
    val_data, test_data, val_labels, test_labels = train_test_split(val_data, val_labels, test_size=0.5, stratify=val_labels, random_state=SEED)

    train_dataset = RnCDataset(train_data, train_labels, training=True)
    val_dataset = RnCDataset(val_data, val_labels, training=False)
    test_dataset = RnCDataset(test_data, test_labels, training=False)

    print(f"Number of train samples: {str(len(train_dataset)).zfill(3)}")
    print(f"Number of val samples:   {str(len(val_dataset)).zfill(3)}")
    print(f"Number of test samples:  {str(len(test_dataset)).zfill(3)}")

    # --- Step 1: Compute class counts ---
    class_counts = np.bincount(train_labels)
    print("Class counts:", class_counts)  # e.g. [4, 2]

    # --- Step 2: Compute weights for each class ---
    class_weights = 1.0 / class_counts
    print("Class weights:", class_weights)  # e.g. [0.25, 0.5]

    # --- Step 3: Assign a weight to each sample based on its label ---
    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    # --- Step 4: Define the sampler ---
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights), # draw same size as dataset
                                    replacement=True,
                                    )
    
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=sampler, generator=generator, drop_last=True, worker_init_fn=worker_init_fn, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, generator=generator, drop_last=True, worker_init_fn=worker_init_fn, collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, generator=generator, drop_last=True, worker_init_fn=worker_init_fn, collate_fn=pad_list_data_collate)

    match backbone:
        case "resnet10":
            model = ResNet3DMonai(depth=10, n_classes=n_classes, features_only=True).cuda()        
        case "resnet18":
            model = ResNet3DMonai(depth=18, n_classes=n_classes, features_only=True).cuda()
        case "densenet121":
            model = DenseNet3DMonai(depth=121, n_classes=n_classes, features_only=True).cuda()             
        case _:
            raise ValueError("Backbone not implemented!")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in ResNet: {model_params}")
    mlflow.log_param("num_params", model_params)
    
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = RnCLoss(temperature=2.0, label_diff='l1', feature_sim='l2') 
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-3)

    scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='min',           # Minimize the monitored metric (e.g., validation loss)
                                                            factor=0.95,           # Reduce LR by a factor of 0.1
                                                            patience=5,           # Wait for n epochs without improvement
                                                            threshold=1e-4)       # Minimum change to qualify as an improvement                                    

    # Variables to track the best model
    best_val_loss = float("inf")
    best_val_loss_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_list = []
        for batch_data in train_loader:
            
            X = batch_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
            X = torch.concat([X[:, 0], X[:, 1]], dim=0)  # Create a dummy second channel by duplicating the first channel         
            y = batch_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
            y = y.view(-1, 1)

            optimizer.zero_grad()            

            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                output = model(X)
                output = output.view(BATCH_SIZE, -1, output.shape[-1])  # Split the output back into two parts
                loss = loss_fn(output, y)

            scaler.scale(loss).backward()            
            scaler.step(optimizer)            
            scaler.update()

            train_loss_list.append(loss.item())                

        train_loss = sum(train_loss_list) / len(train_loader)

        print(f"[TRAINING]   Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}")          

        model.eval()
        val_loss_list = []        
        with torch.no_grad():           
            for val_data in val_loader:
                
                X_val = val_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
                X_val = torch.concat([X_val[:, 0], X_val[:, 1]], dim=0)
                y_val = val_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
                y_val = y_val.view(-1, 1)

                val_output = model(X_val)
                val_output = val_output.view(BATCH_SIZE, -1, val_output.shape[-1])
                loss = loss_fn(val_output, y_val)

                val_output = val_output.detach().cpu()

                val_loss_list.append(loss.item())
                
        val_loss = sum(val_loss_list) / len(val_loader)             

        print(f"[VALIDATION] Epoch [{epoch+1}/{EPOCHS}], Loss: {val_loss:.4f}")

        # Warm-up phase
        if epoch < WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / WARMUP_EPOCHS  # Linear warm-up
            lr = INITIAL_LR + warmup_factor * (TARGET_LR - INITIAL_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Step the ReduceLROnPlateau scheduler after the warm-up phase
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']

        if epoch == 0:
            torch.save(model.state_dict(), f"model_loss_{identifier}.pth")
            mlflow.log_artifact(f"model_loss_{identifier}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), f"model_loss_{identifier}.pth")
            mlflow.log_artifact(f"model_loss_{identifier}.pth")
            print(f"Best model saved at epoch {epoch+1} with loss {best_val_loss:.4f}")
            
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("learning_rate", lr, step=epoch)
        torch.cuda.empty_cache()            

    # Save last epoch model
    torch.save(model.state_dict(), f"model_last_{identifier}.pth")
    mlflow.log_artifact(f"model_last_{identifier}.pth")

    # Log the best model parameters (only once)    
    mlflow.log_param("best_val_loss", best_val_loss)
    mlflow.log_param("best_val_loss_epoch", best_val_loss_epoch)           

    # Load the best model for evaluation  
    model.load_state_dict(torch.load(f"model_loss_{identifier}.pth"))

    model.eval()
    test_loss_list = []
    test_true_list = []
    test_pred_list = []
    test_score_list = []
    with torch.no_grad():           
        for test_data in test_loader:
                         
            X_test = test_data[0].to(torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")    
            X_test = torch.concat([X_test[:, 0], X_test[:, 1]], dim=0)        
            y_test = test_data[1].to(torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
            y_test = y_test.view(-1, 1)

            test_output = model(X_test)
            test_output = test_output.view(BATCH_SIZE, -1, test_output.shape[-1])
            loss = loss_fn(test_output, y_test)

            test_loss_list.append(loss.item())
            test_pred = torch.argmax(test_output, dim=1)
            test_true_list.append(y_test.cpu().numpy())
            test_pred_list.append(test_pred.cpu().numpy())
            test_score_list.append(F.softmax(test_output, dim=1)[:,:].detach().cpu().numpy())

    test_loss = sum(test_loss_list) / len(test_loader)

    print(f"[TESTING]   Loss: {test_loss:.4f}")

    mlflow.log_metric("test_loss", test_loss)

    artifact_uri = mlflow.get_artifact_uri()
    mlflow.log_param("loss_model_path", f"{artifact_uri}/model_loss_{identifier}.pth")
    os.remove(f"model_loss_{identifier}.pth")

if __name__ == "__main__":

    for dataset in ["lung_nodules", "soft_tissue_tumors", "adni"]:
        # for backbone in ["resnet10", "resnet18", "densenet121"]:            
        for backbone in ["resnet10"]:            

            mlflow.set_experiment(experiment_name=f"{dataset}")
            mlflow.start_run() 
            main(dataset, backbone)
            mlflow.end_run()
