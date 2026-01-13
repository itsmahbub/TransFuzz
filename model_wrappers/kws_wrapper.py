from model_wrappers import ModelWrapper
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torchaudio
import numpy as np
import os
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np

class AudioSeedsDataset(Dataset):
    def __init__(self, seeds):
        self.seeds = seeds
    def __len__(self):
        return len(self.seeds)
    def __getitem__(self, idx):
        return self.seeds[idx]

class KWSWrapper(ModelWrapper):
    """
    Wrapper for any HuggingFace keyword-spotting model.
    """
    def __init__(self, model_path=None, device="cpu", **kwargs):
        super().__init__(model_path=model_path, device=device, **kwargs)
       
    def num_classes(self):
        return self.model.config.num_labels
    
    def clamp(self, mutated_inputs):
        clamped = torch.clamp(mutated_inputs, -1.0, 1.0)
        return clamped

    def naturalness_loss(self, inputs, noise, sample_rate=16000,
                        n_fft=512, hop_length=160, win_length=400, n_mels=40):

        # Ensure batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)

        # --- Mel spectrograms ---
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate / 2.0,
            power=2.0,
        )

        mel_in = mel_transform(inputs) + 1e-8   # [B, n_mels, frames]
        mel_no = mel_transform(noise)  + 1e-12

        # --- Frame loudness ---
        frame_loud = mel_in.mean(dim=1, keepdim=True)  # [B,1,F]
        frame_w = 1.0 / (frame_loud / (frame_loud.mean() + 1e-8) + 1e-6)
        frame_w = frame_w.clamp(0.5, 3.0)  # avoid extreme scaling

        # --- Band sensitivity weighting ---
        fb = mel_transform.mel_scale.fb.to(mel_in.device)  # shape [n_freqs, n_mels]
        freqs = torch.linspace(0, sample_rate/2, steps=(n_fft//2 + 1), device=mel_in.device)  # [n_freqs]

        centers = (fb.T @ freqs) / (fb.T.sum(-1) + 1e-12)  # [n_mels]

        w = torch.ones_like(centers)
        w = torch.where((centers >= 2000) & (centers <= 4000), w * 1.8, w)  # boost 2–4kHz
        w = torch.where(centers >= 6000, w * 0.6, w)                        # relax >6kHz
        w = w.view(1, -1, 1)

        # --- Perceptual ratio ---
        perceptual_ratio = (w * frame_w * (mel_no / mel_in)).mean()

        # === Spectral flatness (from earlier impl) ===
        spectrum = torch.stft(noise, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, return_complex=True).abs() + 1e-8
        geo_mean = torch.exp(torch.mean(torch.log(spectrum), dim=-1))
        arith_mean = torch.mean(spectrum, dim=-1)
        flatness = (geo_mean / arith_mean).mean()
        flatness_penalty = 1.0 - flatness

        return perceptual_ratio + 0.1 * flatness_penalty


    def clamp_noise(self, noise, inputs, min_snr_db=25.0, linf_cap=0.005,
                  frame_len=320, hop_len=160, reduction="mean", q=0.3):
        device = noise.device

        # Ensure [B, T]
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        B, T = inputs.shape
        assert noise.dim() == 1 and noise.size(0) == T, "noise must be 1-D of length T"

        # Frame both signal and noise
        frames_in = inputs.unfold(-1, frame_len, hop_len)  # [B, n_frames, frame_len]
        frames_no = noise.unfold(-1, frame_len, hop_len)   # [n_frames, frame_len]

        sig_p   = frames_in.pow(2).mean(-1)                # [B, n_frames]
        noise_p = frames_no.pow(2).mean(-1) + 1e-12        # [n_frames]

        # Broadcast noise_p -> [B, n_frames]
        target_noise_p = sig_p / (10.0 ** (min_snr_db / 10.0))
        scale_mat = torch.sqrt(target_noise_p / noise_p)   # [B, n_frames]

        # Reduce across frames and batch
        if reduction == "min":
            scale = scale_mat.min()
        elif reduction == "mean":
            scale = scale_mat.mean()
        elif reduction == "percentile":
            scale = torch.quantile(scale_mat.flatten(), q)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        # Do not increase noise above original
        scale = torch.clamp(scale, max=1.0)

        # Apply single global scale
        noise = noise * scale.to(device)

        # L-inf clamp
        noise = torch.clamp(noise, -linf_cap, linf_cap)
        return noise


    def resample(self, waveform, orig_sr):
        if orig_sr != self.sample_rate:
            waveform = torch.tensor(waveform) if not torch.is_tensor(waveform) else waveform
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            waveform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)(waveform)
            return waveform.numpy(), self.sample_rate
        return waveform, orig_sr

    def predict_logits(self, preprocessed_inputs):
        return self.model(preprocessed_inputs).logits

    def predict_outputs(self, preprocessed_inputs):
        logits = self.predict_logits(preprocessed_inputs)
        pred_ids = torch.argmax(logits, dim=-1)
        pred_ids = pred_ids.to('cpu')
        return pred_ids

    def predict_probs(self, preprocessed_inputs):
        logits = self.predict_logits(preprocessed_inputs)
        probs = torch.softmax(logits, dim=-1)
        return probs
    

    def get_seeds(self, dataset_name, split="test", count=-1, preprocessed=True, data_dir=None, source_label2id=None):

        if dataset_name == "speech_commands":
            ds = load_dataset("google/speech_commands", "v0.02", split=split)

            # Model's label mapping
            ds_label_names = ds.features["label"].names
            label_remap = {}
            for idx, label in enumerate(ds_label_names):
                if label in self.model.config.label2id:
                    label_remap[idx] = self.model.config.label2id[label]
            
            seeds = []
            for i, ex in enumerate(ds):
                if count != -1 and len(seeds) >= count:
                    break

                wav = ex["audio"]["array"]
                sr = ex["audio"]["sampling_rate"]
                if ex["label"] not in label_remap:
                    continue  # skip unknown labels
                label = int(label_remap[ex["label"]])
                wav, sr = self.resample(wav, sr)
                if preprocessed:
                    processed = self.preprocess(wav)
                    seeds.append((processed, label))
                else:
                    seeds.append((wav, label))
            return AudioSeedsDataset(seeds)
        elif dataset_name == "Adversarial":
            assert data_dir is not None, "data_dir must be specified for Adversarial dataset"
            assert source_label2id is not None, "ds_label_names must be specified for Adversarial dataset"

            id_to_label = {v: k for k, v in source_label2id.items()}
            samples = []
            for root, _, files in os.walk(data_dir):
                for filename in files:
                    parts = filename.split('_')
                    original_label = int(root.split("/")[-1])
                    original_label = id_to_label[original_label]
                    original_label = int(self.model.config.label2id[original_label])
                    adversarial_label = int(parts[4].split(".")[0])
                    adversarial_label = id_to_label[adversarial_label]
                    adversarial_label = int(self.model.config.label2id[adversarial_label])
                    audio_path = os.path.join(root, filename)

                    waveform, sample_rate = torchaudio.load(audio_path)
                    waveform, sample_rate = self.resample(waveform, sample_rate)

                    if preprocessed:
                        waveform = self.preprocess(waveform)
                    samples.append((waveform, original_label, adversarial_label))
            return AudioSeedsDataset(samples)

        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' not supported in get_seeds.")
    
    def compute_loss_ce(self, preprocessed_inputs, ground_labels, target_label=None):
        logits = self.predict_logits(preprocessed_inputs)
        if target_label is None:
            return -F.cross_entropy(logits, ground_labels)
        else:
            target = torch.full((logits.size(0),), target_label, device=logits.device, dtype=torch.long)
            return F.cross_entropy(logits, target)

    def compute_loss(self, preprocessed_inputs, ground_labels, target_label=None):
        ground_labels = ground_labels.to(self.device)
        return self.compute_loss_ce(preprocessed_inputs, ground_labels, target_label)

    @staticmethod
    def collate_fn(batch):

        if len(batch[0]) == 3:
            waveforms, labels, adversarial_labels = zip(*batch)
            adversarial_labels = torch.tensor(adversarial_labels)
        else:
            waveforms, labels = zip(*batch)
            adversarial_labels = None
        # Convert all to 1D numpy arrays (CPU)
        waveforms = [w.cpu().numpy() if torch.is_tensor(w) else w for w in waveforms]
        waveforms = [w.squeeze() for w in waveforms]  # remove any extra dims
        lengths = [len(w) for w in waveforms]
        max_len = max(lengths)
        # Pad each waveform to max_len
        padded_waveforms = [np.pad(w, (0, max_len - len(w)), mode='constant') for w in waveforms]
        # Stack into [batch, max_len]
        batch_waveforms = torch.tensor(np.stack(padded_waveforms), dtype=torch.float32)
        return batch_waveforms, torch.tensor(labels), adversarial_labels

    def is_correct(self, inputs, outputs, preprocessed=False):

        if not preprocessed:
            inputs = self.preprocess(inputs)  # should handle batching
        else:
            inputs = inputs.to(self.device)

        preds = self.predict_outputs(inputs)
        mask = preds.eq(outputs)

        return mask

    def adversarial_mask(self, predictions, original_predictions, ground_truths, target_label):
        if target_label is None:
            adversarial_mask = (predictions != ground_truths)
            return adversarial_mask
        adversarial_mask = (predictions == target_label) & (predictions != ground_truths)
        return adversarial_mask

    def save_adversarial_example(self, id, orig_waveform, mutated_waveform, mutated_label, original_prediction, ground_truth, target_label, ae_dir):
        os.makedirs(f"{ae_dir}/orig/{ground_truth}", exist_ok=True)
        os.makedirs(f"{ae_dir}/aes/{ground_truth}", exist_ok=True)
        torchaudio.save(f"{ae_dir}/orig/{ground_truth}/{id}_orig_{original_prediction}.wav", orig_waveform.unsqueeze(0).detach(), self.sample_rate)
        torchaudio.save(f"{ae_dir}/aes/{ground_truth}/{id}_ae_{original_prediction}_{mutated_label}_{target_label}.wav", mutated_waveform.unsqueeze(0).detach(), self.sample_rate)

    def save_poison(self, id, poison_waveform, ae_dir):
        poison_dir = f"{ae_dir}/poisons"
        os.makedirs(poison_dir, exist_ok=True)
        np.save(f"{poison_dir}/{id}-poison.npy", poison_waveform)
