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

class AudioSeedsDataset(Dataset):
    def __init__(self, seeds):
        self.seeds = seeds
    def __len__(self):
        return len(self.seeds)
    def __getitem__(self, idx):
        return self.seeds[idx]

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np

class DiffMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=128, target_frames=128,
                 n_fft=400, hop_length=160, win_length=400,
                 device=None, return_permuted=False):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1e-6
        self.return_permuted = return_permuted

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
        ).to(self.device)

        # Normalization constants from AST pretraining
        self._mean_shift = 4.26
        self._std_scale = 9.14  # 4.57 * 2

    def _to_tensor(self, waveforms):
        if isinstance(waveforms, np.ndarray):
            if waveforms.ndim == 1:
                waveforms = torch.from_numpy(waveforms).float().unsqueeze(0)
            else:
                waveforms = torch.from_numpy(waveforms).float()
        elif isinstance(waveforms, torch.Tensor):
            if waveforms.ndim == 1:
                waveforms = waveforms.float().unsqueeze(0)
            else:
                waveforms = waveforms.float()
        else:
            raise ValueError("Unsupported waveform type")
        return waveforms.to(self.device)

    def __call__(self, waveforms):
        waveforms = self._to_tensor(waveforms)  # (B, T)

        mel = self.mel_spec(waveforms)  # (B, n_mels, T_frames)
        log_mel = torch.log(mel + self.eps)

        # pad/truncate to exactly target_frames
        B, n_mels, T = log_mel.shape
        if T < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - T))
        elif T > self.target_frames:
            log_mel = log_mel[:, :, :self.target_frames]

        # Normalize
        log_mel = (log_mel + self._mean_shift) / self._std_scale

        # Match HF extractor output shape: (B, frames, n_mels)
        if self.return_permuted:
            return log_mel.permute(0, 2, 1).contiguous()  # (B, frames, n_mels)
        else:
            return log_mel  # (B, n_mels, frames)


class MITASTWrapper(KWSWrapper):

    def __init__(self, model_path=None, device="cpu", **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)
        model_id = "MIT/ast-finetuned-speech-commands-v2"
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
        self.feature_extractor = DiffMelExtractor(device=device, return_permuted=True)

        self.model.eval()

        # Audio sample rate
        self.sample_rate = 16000

    def preprocess(self, waveforms):
        features = self.feature_extractor(waveforms)
        return features.to(self.device)
