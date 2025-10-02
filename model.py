import torch
import torchvision
# from inflated_convnets_pytorch.src.i3res import I3ResNet
# from inflated_convnets_pytorch.src.i3dense import I3DenseNet
# import copy
# from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, DenseNet121_Weights
import monai
import torch.nn as nn

# class ResNet3DImageNetInflated(torch.nn.Module):
#     def __init__(self, depth: int = 50, n_classes: int = 2, pretrained: bool = True):
#         super(ResNet3DImageNetInflated, self).__init__()
        
#         match depth:
#             case 18:
#                 weights = ResNet18_Weights.DEFAULT if pretrained else None
#                 resnet = torchvision.models.resnet18(weights=weights)
#             case 34:
#                 weights = ResNet34_Weights.DEFAULT if pretrained else None
#                 resnet = torchvision.models.resnet34(weights=weights)
#             case 50:
#                 weights = ResNet50_Weights.DEFAULT if pretrained else None
#                 resnet = torchvision.models.resnet50(weights=weights)
#             case _:
#                 raise ValueError(f"Unsupported ResNet depth: {depth}")

#         self.model = I3ResNet(copy.deepcopy(resnet), frame_nb=16)

#         if self.model.fc.out_features != n_classes:
#             self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes, bias=True)

#     def forward(self, x):
#         # Assuming x is of shape (batch_size, channels, depth, height, width)
#         return self.model(x)
    
    
# class DenseNet3DImageNetInflated(torch.nn.Module):
#     def __init__(self, depth: int = 121, n_classes: int = 2, pretrained: bool = True):
#         super(DenseNet3DImageNetInflated, self).__init__()

#         match depth:
#             case 121:
#                 weights = DenseNet121_Weights.DEFAULT if pretrained else None
#                 densenet = torchvision.models.densenet121(weights=weights)
#             case _:
#                 raise ValueError(f"Unsupported DenseNet depth: {depth}")

#         self.model = I3DenseNet(copy.deepcopy(densenet), frame_nb=128, inflate_block_convs=True)

#         if self.model.classifier.out_features != n_classes:
#             self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, n_classes, bias=True)

#     def forward(self, x):
#         # Assuming x is of shape (batch_size, channels, depth, height, width)
#         return self.model(x)


class DenseNet3DMonai(torch.nn.Module):
    def __init__(self, depth: int = 121, n_classes: int = 5, features_only: bool = False):
        super(DenseNet3DMonai, self).__init__()

        match depth:
            case 121:
                self.model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes)
                if features_only:
                    self.model = self.model.features.append(nn.AdaptiveAvgPool3d(1)).append(nn.Flatten(start_dim=1, end_dim=-1))

            case _:
                raise ValueError(f"Unsupported DenseNet depth: {depth}")

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, depth, height, width)
        return self.model(x)
    
class EfficientNet3DMonai(torch.nn.Module):
    def __init__(self, model_name: str = "efficientnet-b0", n_classes: int = 5):
        super(EfficientNet3DMonai, self).__init__()

        # b0:  4694785 params
        # b1:  7452901 params
        # b2:  8721991 params
        # b3: 12066157 params

        self.model = monai.networks.nets.EfficientNetBN(spatial_dims=3, in_channels=1, num_classes=n_classes, model_name=model_name)

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, depth, height, width)
        return self.model(x)
    
class ResNet3DMonai(torch.nn.Module):
    def __init__(self, depth: int = 18, n_classes: int = 5, features_only: bool = False):
        super(ResNet3DMonai, self).__init__()

        match depth:
            case 10:
                self.model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=n_classes)

                if features_only:
                    self.model = monai.networks.nets.ResNetFeatures("resnet10", pretrained=True, spatial_dims=3, in_channels=1)
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                    self.model.append(nn.AdaptiveAvgPool3d(1)).append(nn.Flatten(start_dim=1, end_dim=-1))

            case 18:
                self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=n_classes)

                if features_only:
                    self.model = monai.networks.nets.ResNetFeatures("resnet18", pretrained=True, spatial_dims=3, in_channels=1)
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                    self.model.append(nn.AdaptiveAvgPool3d(1)).append(nn.Flatten(start_dim=1, end_dim=-1))

            case 34:
                self.model = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=1, num_classes=n_classes)

                if features_only:
                    self.model = monai.networks.nets.ResNetFeatures("resnet34", pretrained=True, spatial_dims=3, in_channels=1)
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                    self.model.append(nn.AdaptiveAvgPool3d(1)).append(nn.Flatten(start_dim=1, end_dim=-1))
            
            case 50:
                self.model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=n_classes)

                if features_only:
                    self.model = monai.networks.nets.ResNetFeatures("resnet50", pretrained=True, spatial_dims=3, in_channels=1)
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                    self.model.append(nn.AdaptiveAvgPool3d(1)).append(nn.Flatten(start_dim=1, end_dim=-1))
            
            case _:
                raise ValueError(f"Unsupported ResNet depth: {depth}")

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, depth, height, width)
        return self.model(x)