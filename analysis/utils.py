import os
from PIL import Image
from torch.utils.data import Dataset
import torchaudio

class AdversarialImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        for root, _, files in os.walk(root_dir):
            for filename in files:
                # print(filename)
                original_label = int(root.split("/")[-1])
                parts = filename.split('_')
                adversarial_label = int(parts[4].split(".")[0])
                full_path = os.path.join(root, filename)
                self.samples.append((full_path, original_label, adversarial_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, original_label, adversarial_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, original_label, adversarial_label
    

class AdversarialAudioDataset(Dataset):
    def __init__(self, root_dir, model_wrapper):
        self.samples = []
        self.model_wrapper = model_wrapper

        for root, _, files in os.walk(root_dir):
            for filename in files:
                parts = filename.split('_')
                original_label = int(parts[0])
                adversarial_label = int(parts[4].split(".")[0])
                full_path = os.path.join(root, filename)
                self.samples.append((full_path, original_label, adversarial_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, original_label, adversarial_label = self.samples[idx]
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform, sample_rate = self.model_wrapper.resample(waveform, sample_rate)
        processed = self.model_wrapper.preprocess(waveform)
        return processed, original_label, adversarial_label