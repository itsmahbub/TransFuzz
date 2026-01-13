import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_wrappers.mobile_vit_wrapper import MobileViTWrapper
from analysis.utils import AdversarialImageDataset
import json



def model_transfer_attack(model_wrapper, dataset):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=model_wrapper.collate_fn)

    correct = 0
    total = 0
    preserved_adversarial = 0
    adversarial = 0

    with torch.no_grad():
        for inputs, labels, adversarial_label in tqdm(data_loader):
            # inputs, labels, adversarial_label = inputs.to(model_wrapper.device), labels.to(model_wrapper.device), adversarial_label.to(model_wrapper.device)
            inputs = inputs.to(model_wrapper.device)
            preds = model_wrapper.predict_outputs(inputs)
         
            correct += (preds == labels).sum().item()
            preserved_adversarial += ((preds != labels) & (preds == adversarial_label)).sum().item()
            adversarial += (preds != labels).sum().item()

            total += inputs.size(0)

    preserved_adversarial_rate = 100 * preserved_adversarial / total
    adversarial_rate = 100 * adversarial / total
   
    result = {
    
        "preservation": preserved_adversarial,
        "adversarial": adversarial,
        "success_rate": adversarial_rate,
        "preserved_success_rate": preserved_adversarial_rate,
    
    }

    return result

