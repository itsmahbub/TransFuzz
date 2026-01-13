
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model_wrappers.resnet_wrapper import ResNetWrapper
from utils import AdversarialImageDataset
from tqdm import tqdm
from collections import Counter
import json

def calculate_stability(model_wrapper, dataset):
   
    # Stats
    preserved_adversarial = 0
    adversarial = 0

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=model_wrapper.collate_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, orig_labels, adv_labels = batch

            inputs = inputs.to(model_wrapper.device)
            # Count how many of each adversarial label
        
            predicted_labels = model_wrapper.predict_outputs(inputs)
           
            # Exclusive classification
            preserved_mask = predicted_labels == adv_labels
            adv_mask = predicted_labels != orig_labels

            preserved_adversarial += preserved_mask.sum().item()
            adversarial += adv_mask.sum().item()

    result = {
        "adversarial": adversarial,
        "preservation": preserved_adversarial,
        "adversarial_rate": 100 * adversarial / len(dataset),
        "preserved_adversarial_rate": 100 * preserved_adversarial / len(dataset)
    }
    return result



def calculate_adv_class_dist(model_wrapper, dataset):
   
    # Stats
    adv_label_counts = Counter()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=model_wrapper.collate_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            _, _, adv_labels = batch

            # Count how many of each adversarial label
            adv_label_counts.update(adv_labels.tolist())
        
     
    adversarial_label_distribution = dict(adv_label_counts)
    adversarial_label_distribution = dict(sorted(adversarial_label_distribution.items()))
    result = {
        "aversarial_classes": adversarial_label_distribution
    }
    return result


