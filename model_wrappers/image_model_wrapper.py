from model_wrappers import ModelWrapper
import torchvision.transforms as transforms

import torch
import os
import numpy as np
import torchvision.datasets as datasets
import random
from torch.utils.data import Subset
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F


class ImageSeedsDatasetLazy(Dataset):
    """
    Lazy dataset: stores (path, orig_label, adv_label) metadata only.
    Loads and transforms images on-the-fly in __getitem__.
    """
    def __init__(self, meta_list, transform=None):
        """
        meta_list: list of tuples (img_path, original_label, adversarial_label_or_None)
        transform: torchvision transform that consumes PIL.Image and returns tensor.
        """
        self.meta = list(meta_list)
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        path, orig_label, adv_label = self.meta[idx]
        # lazy load and decode
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # adv_label may be None or an int
        return img, orig_label, adv_label

class UnsafeBenchTorchDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 1 if item["safety_label"] == "Unsafe" else 0
        return image, label

    def __len__(self):
        return len(self.dataset)


class ImageModelWrapper(ModelWrapper):
    def __init__(self, model_path=None, device=None, **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)

    def naturalness_loss(self, inputs, noise):
        return torch.norm(noise, p=float("inf"))


    def clamp_noise(self, noise, inputs):
        min_inputs = inputs.min(dim=0)[0]  # Shape: (3, 224, 224)
        max_inputs = inputs.max(dim=0)[0]  # Shape: (3, 224, 224)

        max_noise = 255 - max_inputs
        min_noise = -min_inputs

        noise = torch.floor(noise)
        clamped_noise = noise.clamp(min=min_noise, max=max_noise)
        return clamped_noise  # Shape: (3, 224, 224)

    def clamp(self, mutated_input):
        return torch.clamp(mutated_input, 0, 255).byte().float()


    def save_adversarial_example(self, id, input, mutated_input, mutated_label, original_prediction, ground_truth, target_label, ae_dir):
      
        os.makedirs(f"{ae_dir}/aes/{ground_truth}/", exist_ok=True)
        os.makedirs(f"{ae_dir}/orig/{ground_truth}/", exist_ok=True)
        
        input = input.byte()
        mutated_input = mutated_input.byte()
        to_pil = transforms.ToPILImage()
        
        input_img = to_pil(input)
        mutated_img = to_pil(mutated_input)
        input_img.save( f"{ae_dir}/orig/{ground_truth}/{id}_orig_{original_prediction}.png", format='PNG')
        mutated_img.save(f"{ae_dir}/aes/{ground_truth}/{id}_ae_{original_prediction}_{mutated_label}_{target_label}.png", format='PNG')

    def save_poison(self, id, poison, ae_dir):
        d = f"{ae_dir}/poisons/"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/{id}-poison.npy", poison)

    def get_seeds(self, dataset_name, count=-1, split="test", transform=None, data_dir=None):

        if dataset_name == "Adversarial":
            assert data_dir is not None, "data_dir must be specified for Adversarial dataset"
            samples = []
            for root, _, files in os.walk(data_dir):
                for filename in files:
                    if "_ae_" in filename:
                        # print(filename)
                        original_label = int(root.split("/")[-1])
                        parts = filename.split('_')
                    
                        adversarial_label = int(parts[4].split(".")[0])
                    
                        img_path = os.path.join(root, filename)
                        samples.append((img_path, original_label, adversarial_label))
            
            seed_dataset = ImageSeedsDatasetLazy(samples, transform=transform)
            if count!=-1:
                indices = random.sample(range(len(seed_dataset)), min(count, len(seed_dataset)))
                seed_dataset = Subset(seed_dataset, indices)
            return seed_dataset
        
        elif dataset_name == "Original":
            assert data_dir is not None, "data_dir must be specified for Adversarial dataset"
            samples = []
            for root, _, files in os.walk(data_dir):
                for filename in files:
                 
                    if "_orig_" in filename and filename.endswith(".png"):
                        original_label = int(root.split("/")[-1])
                        parts = filename.split('_')
        
                        img_path = os.path.join(root, filename)
                        samples.append((img_path, original_label, original_label ))
            
            seed_dataset = ImageSeedsDatasetLazy(samples, transform=transform)
            if count!=-1:
                indices = random.sample(range(len(seed_dataset)), min(count, len(seed_dataset)))
                seed_dataset = Subset(seed_dataset, indices)
            return seed_dataset

        if dataset_name == "ImageNet":

            imagenet_seed_path = f"seeds/imagenet-mini/{split}"
            seed_dataset = datasets.ImageFolder(root=imagenet_seed_path, transform=transform)

            if count!=-1:
                indices = random.sample(range(len(seed_dataset)), count)
                seed_dataset = Subset(seed_dataset, indices)
            
            return seed_dataset
        if dataset_name == "UnsafeBench":
            dataset = load_dataset("yiting/UnsafeBench")
            seed_dataset = UnsafeBenchTorchDataset(dataset[split], transform=transform)
            if count!=-1:
                indices = random.sample(range(len(seed_dataset)), count)
                seed_dataset = Subset(seed_dataset, indices)
            
            return seed_dataset

    def predict_logits(self, preprocessed_inputs):
        logits = self.model(preprocessed_inputs)
        return logits

    def predict_outputs(self, preprocessed_inputs):
        logits = self.predict_logits(preprocessed_inputs)
        preds = torch.argmax(logits, dim=1)
        return preds.to('cpu')

    def predict_probs(self, preprocessed_inputs):
        logits = self.predict_logits(preprocessed_inputs)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def is_correct(self, inputs, outputs, preprocessed=False):
        inputs = inputs.to(self.device)

        if not preprocessed:
            inputs = self.preprocess(inputs)  # must handle batched input

        preds = self.predict_outputs(inputs)  # tensor of shape (B,)

        return preds.eq(outputs)  # Boolean mask (B,)

    def compute_loss_cw(self, preprocessed_inputs, ground_labels, target_label):
        logits = self.predict_logits(preprocessed_inputs)

        batch_size = logits.shape[0]

        # True class logits
        true_logits = logits[torch.arange(batch_size), ground_labels]

        if target_label is None:
            # Untargeted attack: max other logit - true logit
            # Mask out true class
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size), ground_labels] = False
            other_logits = logits.masked_fill(~mask, float('-inf')).max(dim=1).values

            margin = other_logits - true_logits
        else:
            # Targeted attack: target logit - true logit
            target_logits = logits[:, target_label]
            margin = target_logits - true_logits

        # CW loss: clamp with -kappa to enforce margin
        loss = torch.clamp(margin, min=-5).mean()

        return -loss

    def compute_loss_ce(self, preprocessed_inputs, ground_labels, target_label=None):
        logits = self.predict_logits(preprocessed_inputs)
        if target_label is None:
            return -F.cross_entropy(logits, ground_labels)
        else:
            target = torch.full((logits.size(0),), target_label, device=logits.device, dtype=torch.long)
            return F.cross_entropy(logits, target)
    
    def compute_loss(self, preprocessed_inputs, ground_labels, target_label):
        ground_labels = ground_labels.to(self.device)
        return self.compute_loss_ce(preprocessed_inputs, ground_labels, target_label)
    
    def targeted_adversarial_count(self, predictions, ground_truths, target_label):
        if target_label is None:
            return 0

        targeted_success = ((predictions == target_label) & (predictions != ground_truths)).sum().item()
        return targeted_success
    
    def adversarial_mask(self, predictions, original_predictions, ground_truths, target_label):
        if target_label is None:
            adversarial_mask = (predictions != ground_truths)
            return adversarial_mask
        adversarial_mask = (predictions == target_label) & (predictions != ground_truths)
        return adversarial_mask

    @staticmethod
    def collate_fn(batch):
        if len(batch[0]) == 3:
            images, labels, adversarial_labels = zip(*batch)
            adversarial_labels = torch.tensor(adversarial_labels)
        else:
            images, labels = zip(*batch)
            adversarial_labels = None
            
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels, adversarial_labels
