from model_wrappers import ModelWrapper
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import random
from torch.utils.data import Subset
from collections import defaultdict
from PIL import Image
from io import BytesIO
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from model_wrappers.image_model_wrapper import ImageModelWrapper

class MobileViTFeatureTransform:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, img):
        tensor = self.processor(images=img, do_resize=False, do_center_crop=False, return_tensors="pt")['pixel_values'][0]
        return tensor

class MobileViTWrapper(ImageModelWrapper):
    def __init__(self, model_path=None, device=None, **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)
        if self.model_path:
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small")
            self.processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
        self.model.to(self.device)
        self.model.eval()


    def preprocess(self, input):
        if isinstance(input, torch.Tensor) and input.dim() == 4:
            # input: [B, 3, H, W]
            images = [img for img in input] 
            inputs = self.processor(images=images, do_resize=False, do_center_crop=False, return_tensors="pt")['pixel_values']
            return inputs.to(self.device, dtype=torch.float32)
        elif isinstance(input, torch.Tensor) and input.dim() == 3:
            # input: [3, H, W]
            inputs = self.processor(images=input, do_resize=False, do_center_crop=False, return_tensors="pt")['pixel_values'][0]
            return inputs[0].to(self.device, dtype=torch.float32)
        else:
            raise ValueError("Input must be a torch tensor of shape [3, H, W] or [B, 3, H, W]")


    def get_transform(self, preprocessed=True):
        transforms_to_apply = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()),
        ]

        if preprocessed:
            transforms_to_apply.append(MobileViTFeatureTransform(self.processor))
        return transforms.Compose(transforms_to_apply)

    def get_seeds(self, dataset_name, count=-1, split="test", preprocessed=True, data_dir=None):
        return super().get_seeds(dataset_name, count, split, transform=self.get_transform(preprocessed), data_dir=data_dir)

    def predict_logits(self, preprocessed_inputs):
        with torch.no_grad():
            outputs = self.model(pixel_values=preprocessed_inputs)
            logits = outputs.logits
        return logits
