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


def combine_scores_best_alpha(
    energy_scores: np.ndarray,
    mlp_scores: np.ndarray,
    y_true: np.ndarray,
    alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
):
    """
    Grid-search alpha to maximise AUROC, then return the combined scores
    and the winning alpha. Uses the *first half* of each class (ID / OOD)
    as a val split to pick alpha, so the reported test AUROC is not the
    same data used to select the weight.

    Returns:
        combined: np.ndarray of combined test-split scores, shape (N_test,)
        best_alpha: float
        y_test: np.ndarray of aligned labels for `combined`
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    e_n = normalize_scores(energy_scores)
    m_n = normalize_scores(mlp_scores)

    id_idx  = np.where(y_true == 0)[0]
    ood_idx = np.where(y_true == 1)[0]
    id_val,  id_test  = np.array_split(id_idx, 2)
    ood_val, ood_test = np.array_split(ood_idx, 2)

    val_sel  = np.concatenate([id_val,  ood_val])
    test_sel = np.concatenate([id_test, ood_test])
    y_val  = y_true[val_sel]
    y_test = y_true[test_sel]

    best_alpha, best_auroc = 0.5, -1.0
    for a in alphas:
        s_val = a * e_n[val_sel] + (1.0 - a) * m_n[val_sel]
        try:
            au = roc_auc_score(y_val, s_val)
        except ValueError:
            continue
        if au > best_auroc:
            best_auroc, best_alpha = au, a

    combined_test = best_alpha * e_n[test_sel] + (1.0 - best_alpha) * m_n[test_sel]
    return combined_test, best_alpha, y_test


def combine_scores_max(energy_scores: np.ndarray, mlp_scores: np.ndarray) -> np.ndarray:
    """Element-wise max of the two normalised scores — OR-gate: flag OOD if either detector is confident."""
    return np.maximum(normalize_scores(energy_scores), normalize_scores(mlp_scores))


def load_model_checkpoint(model_class, checkpoint_path: str, device: str):
    """
    Load a model from a checkpoint saved with format {"dim": int, "state_dict": ...}.
    Returns the model in eval mode on the specified device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = model_class(dim=ckpt["dim"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
