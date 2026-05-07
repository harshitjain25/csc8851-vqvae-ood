"""
compare_all.py — Unified comparison: Liu et al. 2020 (WRN-40-2) vs VQ-VAE extension.

Runs all detectors on the same test sets and produces:
  outputs/comparison_table.txt  — full numeric results
  outputs/roc_curves.png        — ROC curves per dataset   (poster figure)
  outputs/auroc_bars.png        — grouped AUROC bar chart  (poster figure)

Requirements (all local, no Colab needed):
  checkpoints/wrn40_cifar10_best.pth
  checkpoints/wrn40_energy_ft.pth
  checkpoints/energy_mlp.pth
  checkpoints/ood_mlp.pth
  latent_codes/cifar10.npy
  latent_codes/cifar100.npy
  latent_codes/svhn.npy
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.wideresnet import WideResNet
from models.energy_model import EnergyMLP
from models.mlp import OODMLP
from scripts.metrics import compute_all
from scripts.utils import normalize_scores, load_model_checkpoint

DEVICE = "mps"  if torch.backends.mps.is_available()  else \
         "cuda" if torch.cuda.is_available()           else "cpu"
print(f"Device: {DEVICE}")

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# ── Image loaders for WRN ────────────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize(32), transforms.CenterCrop(32),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
id_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

print("Loading datasets...", flush=True)
id_loader   = DataLoader(datasets.CIFAR10("./data",  train=False, download=True, transform=id_tf),
                         batch_size=256, num_workers=0)
near_loader = DataLoader(datasets.CIFAR100("./data", train=False, download=True, transform=img_tf),
                         batch_size=256, num_workers=0)
far_loader  = DataLoader(datasets.SVHN("./data", split="test", download=True, transform=img_tf),
                         batch_size=256, num_workers=0)

# ── Latent codes for VQ-VAE ───────────────────────────────────────────────────
print("Loading latent codes...", flush=True)
lat_id   = np.load("latent_codes/cifar10.npy").astype(np.float32)
lat_near = np.load("latent_codes/cifar100.npy").astype(np.float32)
lat_far  = np.load("latent_codes/svhn.npy").astype(np.float32)
print(f"  ID={lat_id.shape}  Near={lat_near.shape}  Far={lat_far.shape}")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...", flush=True)

wrn_pre = WideResNet(depth=40, widen_factor=2, num_classes=10).to(DEVICE)
wrn_pre.load_state_dict(torch.load("checkpoints/wrn40_cifar10_best.pth", map_location=DEVICE))
wrn_pre.eval()

wrn_ft = WideResNet(depth=40, widen_factor=2, num_classes=10).to(DEVICE)
wrn_ft.load_state_dict(torch.load("checkpoints/wrn40_energy_ft.pth", map_location=DEVICE))
wrn_ft.eval()

energy_mlp = load_model_checkpoint(EnergyMLP, "checkpoints/energy_mlp.pth", DEVICE)
ood_mlp    = load_model_checkpoint(OODMLP,    "checkpoints/ood_mlp.pth",    DEVICE)

# ── Score functions ───────────────────────────────────────────────────────────
def wrn_energy(model, loader):
    out = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            out.append(-torch.logsumexp(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)

def lat_energy(latents):
    with torch.no_grad():
        return energy_mlp(torch.from_numpy(latents).to(DEVICE)).squeeze(1).cpu().numpy()

def lat_mlp(latents):
    with torch.no_grad():
        return torch.sigmoid(ood_mlp(torch.from_numpy(latents).to(DEVICE)).squeeze(1)).cpu().numpy()

def best_alpha_combine(e_id, e_ood, m_id, m_ood):
    """Grid-search alpha on a val split, return scores+labels on held-out test half."""
    e_all = np.concatenate([e_id, e_ood])
    m_all = np.concatenate([m_id, m_ood])
    y_all = np.concatenate([np.zeros(len(e_id)), np.ones(len(e_ood))])

    e_n, m_n = normalize_scores(e_all), normalize_scores(m_all)

    id_idx  = np.where(y_all == 0)[0]
    ood_idx = np.where(y_all == 1)[0]
    id_val,  id_test  = np.array_split(id_idx, 2)
    ood_val, ood_test = np.array_split(ood_idx, 2)
    val_sel  = np.concatenate([id_val,  ood_val])
    test_sel = np.concatenate([id_test, ood_test])

    from sklearn.metrics import roc_auc_score
    best_a, best_au = 0.0, -1.0
    for a in np.linspace(0, 1, 11):
        s = a * e_n[val_sel] + (1 - a) * m_n[val_sel]
        try:
            au = roc_auc_score(y_all[val_sel], s)
        except ValueError:
            continue
        if au > best_au:
            best_au, best_a = au, a

    s_test = best_a * e_n[test_sel] + (1 - best_a) * m_n[test_sel]
    return s_test, y_all[test_sel], best_a

# ── Compute all scores ────────────────────────────────────────────────────────
print("Computing WRN scores (this takes ~1 min)...", flush=True)
w_id_pre   = wrn_energy(wrn_pre, id_loader)
w_id_ft    = wrn_energy(wrn_ft,  id_loader)
w_near_pre = wrn_energy(wrn_pre, near_loader)
w_near_ft  = wrn_energy(wrn_ft,  near_loader)
w_far_pre  = wrn_energy(wrn_pre, far_loader)
w_far_ft   = wrn_energy(wrn_ft,  far_loader)

print("Computing VQ-VAE scores...", flush=True)
e_id, e_near, e_far = lat_energy(lat_id), lat_energy(lat_near), lat_energy(lat_far)
m_id, m_near, m_far = lat_mlp(lat_id),    lat_mlp(lat_near),    lat_mlp(lat_far)

near_comb, y_near_test, near_alpha = best_alpha_combine(e_id, e_near, m_id, m_near)
far_comb,  y_far_test,  far_alpha  = best_alpha_combine(e_id, e_far,  m_id, m_far)
print(f"  α* Near-OOD={near_alpha:.1f}   α* Far-OOD={far_alpha:.1f}")

# ── Build methods dict ────────────────────────────────────────────────────────
# Each entry: (near_scores, near_y, far_scores, far_y)
def cat(id_s, ood_s):
    return (np.concatenate([id_s, ood_s]),
            np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))]))

METHODS = {
    "WRN-40-2 Energy [1]":         (*cat(w_id_pre, w_near_pre), *cat(w_id_pre, w_far_pre)),
    "WRN-40-2 Fine-tuned [1]":     (*cat(w_id_ft,  w_near_ft),  *cat(w_id_ft,  w_far_ft)),
    "VQ-VAE Latent Energy (ours)": (*cat(e_id, e_near),          *cat(e_id, e_far)),
    "VQ-VAE Latent MLP (ours)":    (*cat(m_id, m_near),          *cat(m_id, m_far)),
    f"VQ-VAE E+MLP α* (ours)":     (near_comb, y_near_test,      far_comb, y_far_test),
}

COLORS = {
    "WRN-40-2 Energy [1]":         "#2C6FAC",
    "WRN-40-2 Fine-tuned [1]":     "#7BAFD4",
    "VQ-VAE Latent Energy (ours)": "#E07B39",
    "VQ-VAE Latent MLP (ours)":    "#2E8B57",
    f"VQ-VAE E+MLP α* (ours)":     "#1A5C37",
}

# ── Results table ─────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
lines = []

for ds_name, sidx in [("Near-OOD · CIFAR-10 vs CIFAR-100", 0),
                       ("Far-OOD  · CIFAR-10 vs SVHN",      2)]:
    header = f"  {'Method':<32} {'AUROC':>8} {'AUPR':>8} {'FPR@95':>8}"
    sep    = "  " + "-" * 60
    print(f"\n{'='*64}\n  {ds_name}\n{'='*64}")
    print(header); print(sep)
    lines += [ds_name, header.strip(), sep.strip()]
    for name, vals in METHODS.items():
        m = compute_all(vals[sidx+1], vals[sidx])
        row = f"  {name:<32} {m['AUROC']:>8.4f} {m['AUPR']:>8.4f} {m['FPR@95TPR']:>8.4f}"
        print(row)
        lines.append(row.strip())
    lines.append("")

with open("outputs/comparison_table.txt", "w") as f:
    f.write("\n".join(lines))
print("\nSaved: outputs/comparison_table.txt")

# ── Figure 1: ROC Curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("ROC Curves: Liu et al. 2020 Baseline vs VQ-VAE Extension",
             fontsize=14, fontweight="bold", y=1.01)

DS_TITLES = ["Near-OOD: CIFAR-10 vs CIFAR-100", "Far-OOD: CIFAR-10 vs SVHN"]

for ax, ds_title, sidx in zip(axes, DS_TITLES, [0, 2]):
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.35, label="Random (AUC=0.500)")
    for name, vals in METHODS.items():
        scores, y = vals[sidx], vals[sidx + 1]
        fpr, tpr, _ = roc_curve(y, scores)
        auroc = sk_auc(fpr, tpr)
        is_ours = "ours" in name
        ls = "-" if is_ours else "--"
        lw = 2.8 if is_ours else 1.8
        ax.plot(fpr, tpr, color=COLORS[name], lw=lw, ls=ls,
                label=f"{name}  (AUC = {auroc:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(ds_title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("outputs/roc_curves.png", dpi=200, bbox_inches="tight")
print("Saved: outputs/roc_curves.png")
plt.close(fig)

# ── Figure 2: Grouped AUROC Bar Chart ─────────────────────────────────────────
short_names = [
    "WRN-40-2\nEnergy [1]",
    "WRN-40-2\nFine-tuned [1]",
    "VQ-VAE\nLatent Energy",
    "VQ-VAE\nLatent MLP",
    "VQ-VAE\nE+MLP α*",
]
near_aurocs, far_aurocs = [], []
for vals in METHODS.values():
    near_aurocs.append(compute_all(vals[1], vals[0])["AUROC"])
    far_aurocs.append(compute_all(vals[3], vals[2])["AUROC"])

x     = np.arange(len(short_names))
width = 0.38
colors = list(COLORS.values())

fig, ax = plt.subplots(figsize=(13, 6))

bars_near = ax.bar(x - width / 2, near_aurocs, width,
                   label="Near-OOD (CIFAR-100)",
                   color=colors, alpha=0.65, edgecolor="white", linewidth=1.2,
                   hatch="//")
bars_far  = ax.bar(x + width / 2, far_aurocs,  width,
                   label="Far-OOD (SVHN)",
                   color=colors, alpha=1.00, edgecolor="white", linewidth=1.2)

for bar, v in zip(bars_near, near_aurocs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.009,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars_far, far_aurocs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.009,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(0.5, color="gray", lw=0.9, ls=":", alpha=0.5)
ax.set_ylim([0.4, 1.10])
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10.5)
ax.set_ylabel("AUROC ↑", fontsize=12)
ax.set_title("AUROC Comparison: Liu et al. 2020 vs VQ-VAE Extension\n"
             "(CIFAR-10 In-Distribution · Solid = Far-OOD · Hatched = Near-OOD)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Color legend patches (one per method)
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=c, label=n)
                  for n, c in COLORS.items()]
ax.legend(handles=legend_patches + [
              Patch(facecolor="gray", alpha=0.65, hatch="//", label="Near-OOD (CIFAR-100)"),
              Patch(facecolor="gray", alpha=1.00, label="Far-OOD (SVHN)"),
          ], fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)

plt.tight_layout()
fig.savefig("outputs/auroc_bars.png", dpi=200, bbox_inches="tight")
print("Saved: outputs/auroc_bars.png")
plt.close(fig)

print("\nAll done. Outputs in outputs/")
