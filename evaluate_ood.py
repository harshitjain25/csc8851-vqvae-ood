import os
import numpy as np
import torch
from scripts.metrics import compute_all
from scripts.utils import combine_scores, load_model_checkpoint
from models.energy_model import EnergyMLP
from models.mlp import OODMLP

DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
ENERGY_CKPT = "checkpoints/energy_mlp.pth"
MLP_CKPT    = "checkpoints/ood_mlp.pth"


def safe_load(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    arr = np.load(path).astype(np.float32)
    print(f"[OK] {path} shape={arr.shape}")
    return arr


def score_energy(model, latents):
    z = torch.from_numpy(latents).to(DEVICE)
    with torch.no_grad():
        return model(z).squeeze(1).cpu().numpy()


def score_mlp(model, latents):
    z = torch.from_numpy(latents).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(z).squeeze(1)).cpu().numpy()


def eval_pair(energy_model, mlp_model, id_lat, ood_lat, name, alpha=0.5):
    y_true = np.concatenate([np.zeros(len(id_lat)), np.ones(len(ood_lat))])

    e_scores = np.concatenate([score_energy(energy_model, id_lat),
                                score_energy(energy_model, ood_lat)])
    m_scores = np.concatenate([score_mlp(mlp_model, id_lat),
                                score_mlp(mlp_model, ood_lat)])
    c_scores = combine_scores(e_scores, m_scores, alpha=alpha)

    print(f"\n=== {name} ===")
    print(f"{'Method':<14} {'AUROC':>8} {'AUPR':>8} {'FPR@95TPR':>12}")
    print("-" * 46)

    results = {}
    for method_name, scores in [("Energy", e_scores), ("MLP", m_scores), ("Energy+MLP", c_scores)]:
        m = compute_all(y_true, scores)
        print(f"{method_name:<14} {m['AUROC']:>8.4f} {m['AUPR']:>8.4f} {m['FPR@95TPR']:>12.4f}")
        results[method_name] = m

    return results


def main():
    id_lat   = safe_load("latent_codes/cifar10.npy")
    near_lat = safe_load("latent_codes/cifar100.npy")
    far_lat  = safe_load("latent_codes/svhn.npy")

    if any(x is None for x in [id_lat, near_lat, far_lat]):
        print("\nLatent files missing — run extract_latents.py first.")
        return

    for ckpt in (ENERGY_CKPT, MLP_CKPT):
        if not os.path.exists(ckpt):
            print(f"[MISSING] {ckpt} — run the corresponding training script first.")
            return

    energy_model = load_model_checkpoint(EnergyMLP, ENERGY_CKPT, DEVICE)
    mlp_model    = load_model_checkpoint(OODMLP,    MLP_CKPT,    DEVICE)
    print(f"\nDevice: {DEVICE} | Latent D={id_lat.shape[1]}")

    near_res = eval_pair(energy_model, mlp_model, id_lat, near_lat,
                         "CIFAR-10 (ID) vs CIFAR-100 (Near-OOD)")
    far_res  = eval_pair(energy_model, mlp_model, id_lat, far_lat,
                         "CIFAR-10 (ID) vs SVHN (Far-OOD)")

    # Save results table
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.txt", "w") as f:
        for label, res in [("CIFAR-10 vs CIFAR-100 (Near-OOD)", near_res),
                            ("CIFAR-10 vs SVHN (Far-OOD)", far_res)]:
            f.write(f"{label}\n")
            f.write(f"{'Method':<14} {'AUROC':>8} {'AUPR':>8} {'FPR@95TPR':>12}\n")
            f.write("-" * 46 + "\n")
            for method, m in res.items():
                f.write(f"{method:<14} {m['AUROC']:>8.4f} {m['AUPR']:>8.4f} {m['FPR@95TPR']:>12.4f}\n")
            f.write("\n")

    print("\nSaved: outputs/results.txt")


if __name__ == "__main__":
    main()
