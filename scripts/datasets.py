import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    PyTorch Dataset wrapping a .npy latent code file.

    Args:
        path:  path to .npy file of shape (N, D), float32
        label: integer label for all samples (0 = ID, 1 = OOD).
               If None, __getitem__ returns a tuple (z,) for unsupervised use
               (compatible with `for (z,) in loader` in train_energy.py).
    """

    def __init__(self, path: str, label: int | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Latent file not found: {path}")
        data = np.load(path).astype(np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array (N, D), got shape {data.shape}")
        self.data = torch.from_numpy(data)
        self.label = label
        self.dim = data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        z = self.data[idx]
        if self.label is None:
            return (z,)
        return z, torch.tensor(self.label, dtype=torch.float32)


def load_latents(path: str) -> np.ndarray:
    """Load a .npy latent file as a float32 numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing latent file: {path}")
    return np.load(path).astype(np.float32)
