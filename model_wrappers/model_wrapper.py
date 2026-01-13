import torch
import os
import numpy as np

class ModelWrapper:
    def __init__(self, model_path=None, device=None, **kwargs):
        self.model_path = model_path
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def get_seeds(self, dataset_name, count=-1, split="test", preprocessed=True, **kwargs):
        raise NotImplemented
    
    def get_mutators(self):
        raise NotImplemented
    
    def preprocess(self, input):
        return input
    
    def save_adversarial_example(self, id, input, mutated_input, label, mutated_label, original_prediction, ground_truth, ae_dir):
        raise NotImplemented
    
    def save_poison(self, id, poison, label, ae_dir):
        os.makedirs(f"{ae_dir}/poisons/", exist_ok=True)
        d = f"{ae_dir}/poisons/{label}" if label is not None else f"{ae_dir}/poisons"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/{id}-poison.npy", poison)

    def predict_outputs(self, preprocessed_inputs):
        raise NotImplemented

    def compute_loss(self, preprocessed_inputs, ground_labels, target_label):
        raise NotImplemented
    
    def targeted_adversarial_count(self, predictions, ground_truths, target_label):
        raise NotImplemented

    def adversarial_mask(self, predictions, ground_truths):
        raise NotImplemented
    
    @staticmethod
    def collate_fn(batch):
        return batch

