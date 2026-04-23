"""
Evaluates energy-based OOD detection (Liu et al. 2020) on multiple OOD datasets.

Tests two models:
  1. Pre-trained WRN-40-2 (inference-time energy score, no fine-tuning)
  2. Fine-tuned WRN-40-2 (energy-bounded learning objective)

OOD datasets evaluated (subset of paper's 6):
  - SVHN          (torchvision, auto-download)
  - Textures/DTD  (torchvision, auto-download)
  - CIFAR-100     (torchvision, auto-download — bonus near-OOD)

Energy score: E(x; f) = -T * log sum_i exp(f_i(x)/T), T=1  [Eq. 4]
Higher energy → more OOD.

Metrics: FPR@95TPR ↓, AUROC ↑, AUPR ↑
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scripts.metrics import compute_all
from models.wideresnet import WideResNet

DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps"  if torch.backends.mps.is_available() else "cpu"

PRETRAINED = "checkpoints/wrn40_cifar10_best.pth"
FINETUNED  = "checkpoints/wrn40_energy_ft.pth"

# CIFAR-10 normalization (same as training)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# All OOD images resized to 32x32 to match CIFAR-10
ood_tf = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
id_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_energy_scores(model, loader):
    """Returns energy scores E(x) = -logsumexp(logits) for all samples in loader."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(DEVICE)
            logits = model(x)
            energy = -torch.logsumexp(logits, dim=1)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores)


def load_model(path):
    m = WideResNet(depth=40, widen_factor=2, num_classes=10).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval()


def print_table(results, title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  {'Dataset':<18} {'FPR@95':>8} {'AUROC':>8} {'AUPR':>8}")
    print(f"  {'-'*54}")
    fprs, aurocs, auprs = [], [], []
    for name, m in results.items():
        print(f"  {name:<18} {m['FPR@95TPR']:>8.4f} {m['AUROC']:>8.4f} {m['AUPR']:>8.4f}")
        fprs.append(m['FPR@95TPR']); aurocs.append(m['AUROC']); auprs.append(m['AUPR'])
    print(f"  {'-'*54}")
    print(f"  {'Average':<18} {np.mean(fprs):>8.4f} {np.mean(aurocs):>8.4f} {np.mean(auprs):>8.4f}")
    print(f"{'='*62}")


# ── Load datasets ────────────────────────────────────────────────────────────
print("Loading datasets...", flush=True)

cifar10 = datasets.CIFAR10("./data", train=False, download=True, transform=id_tf)
id_loader = DataLoader(cifar10, batch_size=256, num_workers=2)

ood_datasets = {}

# SVHN
ood_datasets["SVHN"] = DataLoader(
    datasets.SVHN("./data", split="test", download=True, transform=ood_tf),
    batch_size=256, num_workers=2
)

# Textures / DTD
ood_datasets["Textures"] = DataLoader(
    datasets.DTD("./data", split="test", download=True, transform=ood_tf),
    batch_size=256, num_workers=2
)

# CIFAR-100 (near-OOD, not in paper's 6 but useful for our project)
ood_datasets["CIFAR-100"] = DataLoader(
    datasets.CIFAR100("./data", train=False, download=True, transform=id_tf),
    batch_size=256, num_workers=2
)

print("All datasets loaded.", flush=True)

# ── Get ID scores (shared for both models) ───────────────────────────────────
# We load each model separately and compute ID scores each time so we can
# print a clean per-model table.

os.makedirs("outputs", exist_ok=True)
results_all = {}

for label, ckpt_path in [("Pre-trained (no fine-tune)", PRETRAINED),
                          ("Energy fine-tuned",          FINETUNED)]:
    if not os.path.exists(ckpt_path):
        print(f"\n[SKIP] {ckpt_path} not found — run the corresponding training script first.")
        continue

    print(f"\nEvaluating: {label} ({ckpt_path})", flush=True)
    model = load_model(ckpt_path)

    id_scores = get_energy_scores(model, id_loader)

    res = {}
    for name, ood_loader in ood_datasets.items():
        ood_scores = get_energy_scores(model, ood_loader)
        y_true     = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores     = np.concatenate([id_scores, ood_scores])
        res[name]  = compute_all(y_true, scores)
        print(f"  {name}: AUROC={res[name]['AUROC']:.4f}  "
              f"FPR95={res[name]['FPR@95TPR']:.4f}", flush=True)

    results_all[label] = res
    print_table(res, label)

# ── Save to file ─────────────────────────────────────────────────────────────
with open("outputs/energy_results.txt", "w") as f:
    for label, res in results_all.items():
        f.write(f"{label}\n")
        f.write(f"{'Dataset':<18} {'FPR@95':>8} {'AUROC':>8} {'AUPR':>8}\n")
        f.write("-" * 44 + "\n")
        fprs, aurocs, auprs = [], [], []
        for name, m in res.items():
            f.write(f"{name:<18} {m['FPR@95TPR']:>8.4f} {m['AUROC']:>8.4f} {m['AUPR']:>8.4f}\n")
            fprs.append(m['FPR@95TPR']); aurocs.append(m['AUROC']); auprs.append(m['AUPR'])
        f.write(f"{'Average':<18} {np.mean(fprs):>8.4f} {np.mean(aurocs):>8.4f} {np.mean(auprs):>8.4f}\n\n")

print("\nSaved → outputs/energy_results.txt")
