from model_wrappers import ModelWrapper
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torchaudio
import numpy as np
import os
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from model_wrappers.kws_wrapper import KWSWrapper

import torch

class DiffWav2Vec2Norm(torch.nn.Module):
    """
    Differentiable normalization for raw waveforms, 
    equivalent to Wav2Vec2FeatureExtractor default.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, waveforms: torch.Tensor):
        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms).float()
            
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        mean = waveforms.mean(dim=1, keepdim=True)
        std = waveforms.std(dim=1, keepdim=True) + self.eps
        return (waveforms - mean) / std

class Wav2Vec2KWSWrapper(KWSWrapper):

    def __init__(self, model_path=None, device="cpu", **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)
        model_id = "Amirhossein75/Keyword-Spotting"
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
        self.feature_extractor = DiffWav2Vec2Norm()

        self.model.eval()

        # Audio sample rate
        self.sample_rate = 16000

    def num_classes(self):
        return self.model.config.num_labels
    
    def preprocess(self, waveforms):
        features = self.feature_extractor(waveforms)
        return features.to(self.device)
