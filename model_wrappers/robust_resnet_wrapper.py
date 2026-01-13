# robust_resnet_wrapper.py
from model_wrappers.image_model_wrapper import ImageModelWrapper
from robustbench.utils import load_model
import torch
import numpy as np
import torchvision.transforms as transforms
class RobustResNetWrapper(ImageModelWrapper):
    """
    Target model wrapper for RobustBench ImageNet L-inf-robust ResNet-50.
    Defaults to Salman2020Do_R50 (eps = 4/255). Inputs must be in [0,1].
    """
    def __init__(self,
                 model_path=None,
                 device=None,
                 **kwargs):
        super().__init__(model_path=None, device=device, **kwargs)
        # Load the robust model (RB handles download & internal preprocessing)
        model_name = "Salman2020Do_R50"

        self.model = load_model(
            model_name=model_name,
            dataset="imagenet",
            threat_model="Linf"
        ).to(self.device).eval()

        # IMPORTANT: RobustBench ImageNet models expect [0,1] inputs (no external Normalize)
        self.to01 = transforms.Lambda(lambda x: x / 255.0)

    def preprocess(self, input):
        input = self.to01(input)
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
            transforms_to_apply.append(self.to01)
        return transforms.Compose(transforms_to_apply)
    
    def get_seeds(self, dataset_name, count=-1, split="test", preprocessed=True, data_dir=None):
        # Reuse your base helper but DO NOT append normalization
        return super().get_seeds(dataset_name, count, split,
                                 transform=self.get_transform(preprocessed),
                                 data_dir=data_dir)
