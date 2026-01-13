
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_wrappers.resnet_wrapper import ResNetWrapper
from model_wrappers.mobile_vit_wrapper import MobileViTWrapper
import torch
import tool
import coverage_metrics
from torch.utils.data import DataLoader
import argparse
from model_wrappers.distil_hubert_wrapper import DistilHuBERTWrapper
from model_wrappers.whisper_wrapper import WhisperTinyWrapper
from model_wrappers.blip_wrapper import BLIPWrapper
from utils import AdversarialImageDataset
from coverage_metrics import NLC
import argparse


def calculate_coverage(model_wrapper, random_input, dataset, batch_size=32):
    # random_input = random_input.to(model_wrapper.device)
    # random_input = model_wrapper.preprocess(random_input)
    layer_size_dict = tool.get_layer_output_sizes(model_wrapper.model, random_input, inference_func=model_wrapper.predict_outputs)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=model_wrapper.collate_fn)
    coverage_metric_instance = NLC(model_wrapper.model, layer_size_dict, device=model_wrapper.device, inference_func=model_wrapper.predict_outputs)
    coverage_metric_instance.build(dataset_loader)
    coverage_metric_instance.assess(dataset_loader)
    return coverage_metric_instance.current.item() if isinstance(coverage_metric_instance.current, torch.Tensor) else coverage_metric_instance.current

