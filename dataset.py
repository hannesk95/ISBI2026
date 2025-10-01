# Soft-Tissue Tumors
# G0:     88 -->  44
# G1:    164 -->  82
# G2:    262 --> 141
# G3:    438 --> 219
# Total: 952 --> 486

import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import Compose


class OrdinalClassificationDataset(Dataset):
    def __init__(self, data: list, labels: list, training: bool):
        
        self.data = data
        self.labels = labels
        self.training = training

        rotate = monai.transforms.RandRotate(prob=0.2, range_x=10, range_y=10, range_z=10)
        scale = monai.transforms.RandZoom(prob=0.2, min_zoom=0.7, max_zoom=1.4)
        gaussian_noise = monai.transforms.RandGaussianNoise()
        gaussian_blur = monai.transforms.RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0))
        contrast = monai.transforms.RandAdjustContrast()
        intensity = monai.transforms.RandScaleIntensity(factors=(2, 10))
        histogram_shift = monai.transforms.RandHistogramShift()
        self.transforms = Compose([rotate, scale, gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = torch.load(image_path)
        label = self.labels[idx]

        if self.training:
            image = self.transforms(image)

        # if "resnet" in self.backbone:
        #     image = image.repeat(3, 1, 1, 1)  # Convert to 3 channels if input is single-channel

        return image, label

class RnCDataset(Dataset):
    def __init__(self, data: list, labels: list, training: bool):
        self.data = data
        self.labels = labels
        self.training = training

        rotate = monai.transforms.RandRotate(prob=0.2, range_x=10, range_y=10, range_z=10)
        scale = monai.transforms.RandZoom(prob=0.2, min_zoom=0.7, max_zoom=1.4)
        gaussian_noise = monai.transforms.RandGaussianNoise()
        gaussian_blur = monai.transforms.RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0))
        contrast = monai.transforms.RandAdjustContrast()
        intensity = monai.transforms.RandScaleIntensity(factors=(2, 10))
        histogram_shift = monai.transforms.RandHistogramShift()
        self.transforms = Compose([rotate, scale, gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = torch.load(image_path)
        label = self.labels[idx]

        # if self.training:
        image1 = self.transforms(image)
        image2 = self.transforms(image)

        return torch.stack([image1, image2], dim=0), label