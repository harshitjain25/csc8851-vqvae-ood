"""
Evaluates the baseline CNN energy score on CIFAR-100 (near-OOD) and SVHN (far-OOD).
Energy formula: E(x) = -log sum_i exp(f_i(x))   (Liu et al. Eq. 2)
Reports AUROC, AUPR, and FPR@95TPR to match the latent model evaluation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from scripts.metrics import compute_all

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


model = Net().to(device)
model.load_state_dict(torch.load("checkpoints/baseline_cnn.pth", map_location=device))
model.eval()

transform = transforms.ToTensor()
cifar10  = datasets.CIFAR10("./data",  train=False, download=True, transform=transform)
cifar100 = datasets.CIFAR100("./data", train=False, download=True, transform=transform)
svhn     = datasets.SVHN("./data", split="test",   download=True, transform=transform)


def get_energy_scores(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            energy = -torch.logsumexp(logits, dim=1)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores)


print("Computing baseline energy scores...")
id_scores   = get_energy_scores(cifar10)
ood1_scores = get_energy_scores(cifar100)
ood2_scores = get_energy_scores(svhn)


def report(name, id_s, ood_s):
    y_true = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    scores = np.concatenate([id_s, ood_s])
    m = compute_all(y_true, scores)
    print(f"\n=== Baseline Energy — {name} ===")
    print(f"  AUROC    : {m['AUROC']:.4f}")
    print(f"  AUPR     : {m['AUPR']:.4f}")
    print(f"  FPR@95TPR: {m['FPR@95TPR']:.4f}")
    return m


m_near = report("CIFAR-10 vs CIFAR-100 (Near-OOD)", id_scores, ood1_scores)
m_far  = report("CIFAR-10 vs SVHN (Far-OOD)",       id_scores, ood2_scores)

# Save to outputs/
os.makedirs("outputs", exist_ok=True)
with open("outputs/baseline_results.txt", "w") as f:
    f.write("=== Baseline CNN Energy — Full Metrics ===\n\n")
    f.write("CIFAR-10 vs CIFAR-100 (Near-OOD)\n")
    for k, v in m_near.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\nCIFAR-10 vs SVHN (Far-OOD)\n")
    for k, v in m_far.items():
        f.write(f"  {k}: {v:.4f}\n")

print("\nSaved: outputs/baseline_results.txt")
