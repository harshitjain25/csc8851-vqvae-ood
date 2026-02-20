import os
import numpy as np
from scripts.metrics import compute_all

def safe_load(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    return np.load(path)

def eval_pair(id_lat, ood_lat, name):
    y = np.concatenate([np.zeros(len(id_lat)), np.ones(len(ood_lat))])

    # dummy score just to test pipeline (replace later with energy/mlp score)
    scores = np.concatenate([
        np.linalg.norm(id_lat, axis=1),
        np.linalg.norm(ood_lat, axis=1)
    ])

    m = compute_all(y, scores)
    print(f"\n=== {name} ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

def main():
    id_lat   = safe_load("latent_codes/cifar10.npy")
    near_lat = safe_load("latent_codes/cifar100.npy")
    far_lat  = safe_load("latent_codes/svhn.npy")

    if id_lat is None or near_lat is None or far_lat is None:
        print("\nLatent files not available yet. Waiting for Parsh.")
        print("Expected files:")
        print("  latent_codes/cifar10.npy")
        print("  latent_codes/cifar100.npy")
        print("  latent_codes/svhn.npy")
        return

    eval_pair(id_lat, near_lat, "CIFAR-10 (ID) vs CIFAR-100 (Near-OOD)")
    eval_pair(id_lat, far_lat,  "CIFAR-10 (ID) vs SVHN (Far-OOD)")

if __name__ == "__main__":
    main()