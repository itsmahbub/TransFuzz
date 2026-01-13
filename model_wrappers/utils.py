from torch.utils.data import Dataset
import torch

from torch.utils.data import DataLoader, Dataset

class FilteredDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, label


def filter_incorrect(dataset, model_wrapper, batch_size=64, preprocessed=False):
    """
    Filters out incorrectly classified inputs from the dataset.
    Returns a new Dataset containing only correctly classified inputs.
    """
    print("Dataset size before filtering:", len(dataset))
    filtered_samples = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=model_wrapper.collate_fn)

    with torch.no_grad():
        for imgs, labels, *_ in loader:
            
            correct_mask = model_wrapper.is_correct(imgs, labels, preprocessed=preprocessed)

            # Keep only correct samples
            for img, label, correct in zip(imgs, labels, correct_mask):
                if correct.item():
                    filtered_samples.append((img, label))

    print("Dataset size after filtering:", len(filtered_samples))
    return FilteredDataset(filtered_samples)
