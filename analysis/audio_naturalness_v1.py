import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalDistortionRatio,
)
from pystoi import stoi

import csv
import math

SAMPLE_RATE = 16000
PESQ_MODE = "wb"  # "wb"=wideband (16k), "nb"=narrowband (8k)


def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return torch.tensor(y, dtype=torch.float32)


def align_length(x, y):
    L = min(len(x), len(y))
    return x[:L], y[:L]


def perturbation_metrics(clean_wav: torch.Tensor, adv_wav: torch.Tensor, eps=1e-9):
    """
    Returns: snr_db, l2, linf
    clean_wav, adv_wav: 1D torch.Tensor
    """
    perturb = adv_wav - clean_wav
    norm_clean = torch.norm(clean_wav) + eps
    norm_pert = torch.norm(perturb) + eps
    snr_db = 20.0 * math.log10(norm_clean.item() / norm_pert.item())
    l2 = norm_pert.item()
    linf = torch.max(torch.abs(perturb)).item()
    return snr_db, l2, linf


def compute_stoi(clean_wav: torch.Tensor, adv_wav: torch.Tensor, extended=True, sr=SAMPLE_RATE):
    clean_np = clean_wav.detach().cpu().numpy().astype(np.float32)
    adv_np   = adv_wav.detach().cpu().numpy().astype(np.float32)
    return float(stoi(clean_np, adv_np, sr, extended=extended))

def calculate_audio_naturalness(orig_root, aes_root):
    # Torchmetrics objects
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode=PESQ_MODE)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=False)
    estoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=True)
    sisdr_metric = ScaleInvariantSignalDistortionRatio()

    pesq_vals, stoi_vals, sisdr_vals = [], [], []
    snr_vals, l2_vals, linf_vals = [], [], []
    estoi_vals = []
    high_dist_samples = []

    # Traverse subfolders (classes)
    for cls in sorted(os.listdir(orig_root)):
        orig_cls_dir = os.path.join(orig_root, cls)
        aes_cls_dir = os.path.join(aes_root, cls)
        if not os.path.isdir(orig_cls_dir) or not os.path.isdir(aes_cls_dir):
            continue

        orig_files = os.listdir(orig_cls_dir)
        aes_files = os.listdir(aes_cls_dir)

        # Match files by id (e.g., '0_1.wav' => '0-1')
        orig_dict = {f"{f.split('_')[0]}-{f.split('_')[1].split('.')[0]}": f for f in orig_files}
        aes_dict = {f"{f.split('_')[0]}-{f.split('_')[1].split('.')[0]}": f for f in aes_files}
        common_ids = set(orig_dict.keys()) & set(aes_dict.keys())

        for file_id in tqdm(common_ids, desc=f"Class {cls}"):
            orig_path = os.path.join(orig_cls_dir, orig_dict[file_id])
            aes_path = os.path.join(aes_cls_dir, aes_dict[file_id])

            y_ref = load_audio(orig_path)
            y_hat = load_audio(aes_path)
            y_ref, y_hat = align_length(y_ref, y_hat)

            # PESQ can fail on very short clips
            try:
                pesq = pesq_metric(y_hat, y_ref).item()
                pesq_vals.append(pesq)
            except Exception:
                pesq = None  # or np.nan

            sisdr = sisdr_metric(y_hat.unsqueeze(0), y_ref.unsqueeze(0)).item()

            # stoi_vals.append(stoi)
            sisdr_vals.append(sisdr)

            # Perturbation metrics
            snr_db, l2, linf = perturbation_metrics(y_ref, y_hat)
            snr_vals.append(snr_db)
            l2_vals.append(l2)
            linf_vals.append(linf)

    
            stoi = stoi_metric(y_hat, y_ref).item()
            stoi_vals.append(stoi)
            estoi = estoi_metric(y_hat, y_ref).item()
            estoi_vals.append(estoi)


    summary = {
        "mean_PESQ": round(np.mean(pesq_vals), 4) if pesq_vals else None,
        "mean_STOI": round(np.mean(stoi_vals), 4) if stoi_vals else None,
        "mean_ESTOI": round(np.mean(estoi_vals), 4) if estoi_vals else None,
        "mean_SI-SDR": round(np.mean(sisdr_vals), 4) if sisdr_vals else None,
        "low_PESQ_count(<2.0)": len(high_dist_samples),
        "mean_SNR_dB": round(np.mean(snr_vals), 4) if snr_vals else None,
        "mean_L2": round(np.mean(l2_vals), 6) if l2_vals else None,
        "mean_Linf": round(np.mean(linf_vals), 6) if linf_vals else None,
        "num_samples": sum([len(os.listdir(os.path.join(orig_root, cls)))
                            for cls in os.listdir(orig_root) if os.path.isdir(os.path.join(orig_root, cls))]),
    }
    return summary

if __name__ == "__main__":
    orig_root = "adversarial-examples/distilhubert/speech_commands/NLC/None/32/orig"
    aes_root = "adversarial-examples/distilhubert/speech_commands/NLC/None/32/aes"
    out_csv = "adversarial-examples/distilhubert/speech_commands/NLC/None/32/audio_naturalness_per_sample.csv"

    results = calculate_audio_naturalness(orig_root, aes_root)
    print("Audio Naturalness Results:", results)
