from model_wrappers.image_model_wrapper import ImageModelWrapper
import torchvision.transforms as transforms

from torchvision.models import resnet50, ResNet50_Weights
import torch
import os
from torchvision.utils import save_image
import numpy as np
import torchvision.datasets as datasets
import random
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.nn as nn
from datasets import load_dataset


class ResNetWrapper(ImageModelWrapper):
    def __init__(self, model_path=None, device=None, **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)
        if self.model_path:
            self.model = torch.load(self.model_path, map_location=device)
        else:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
        self.imagenet_normalizer = transforms.Compose([
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, input):
        input = self.imagenet_normalizer(input)
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        return input.to(device=self.device, dtype=torch.float32)

    
    def get_transform(self, preprocessed=True):
        transforms_to_apply = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()),
        ]
        if preprocessed:
            transforms_to_apply.extend(self.imagenet_normalizer.transforms)
        return transforms.Compose(transforms_to_apply)

    def get_seeds(self, dataset_name, count=-1, split="test", preprocessed=True, data_dir=None):
        return super().get_seeds(dataset_name, count, split, transform=self.get_transform(preprocessed), data_dir=data_dir)
