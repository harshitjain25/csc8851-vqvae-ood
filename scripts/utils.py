import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0, 1].
    Returns zeros if all values are identical (avoids division by zero).
    """
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def combine_scores(
    energy_scores: np.ndarray,
    mlp_scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine energy-based and MLP-based OOD scores via weighted alpha blend.

    Both arrays are independently min-max normalized to [0, 1] before blending
    so neither dominates due to scale differences.

    Higher combined score = more likely OOD (consistent with both inputs).

    Args:
        energy_scores: raw energy outputs from EnergyMLP, shape (N,)
        mlp_scores:    sigmoid probabilities from OODMLP, shape (N,)
        alpha:         weight on energy score (default 0.5 = equal blend)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * normalize_scores(energy_scores) + (1.0 - alpha) * normalize_scores(mlp_scores)


def load_model_checkpoint(model_class, checkpoint_path: str, device: str):
    """
    Load a model from a checkpoint saved with format {"dim": int, "state_dict": ...}.
    Returns the model in eval mode on the specified device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = model_class(dim=ckpt["dim"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
