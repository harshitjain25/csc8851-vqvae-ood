"""
Fine-tunes WRN-40-2 with energy-bounded learning objective (Liu et al. 2020, Eq. 8-10).

Training objective:
    L = L_CE  +  lambda * [E[max(0, E(x_in) - m_in)^2]
                         +  E[max(0, m_out - E(x_out))^2]]

Hyperparameters (Appendix B of paper):
    lambda = 0.1,  m_in = -23,  m_out = -5

Requires:
    checkpoints/wrn40_cifar10_best.pth   (from train_wrn.py)
    data/300K_random_images.npy          (downloaded in Colab notebook)

Expected runtime: ~10-15 min on Colab T4 GPU.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from models.wideresnet import WideResNet

# ── Config (exact values from paper Appendix B) ─────────────────────────────
EPOCHS    = 10
BATCH_IN  = 128
BATCH_OUT = 256
LR        = 0.001
M_IN      = -23.0   # energy margin for in-distribution (Appendix B, CIFAR-10)
M_OUT     = -5.0    # energy margin for OOD (Appendix B)
LAMBDA    = 0.1     # weight for energy regularisation term (Section 3.3)
PRETRAINED = "checkpoints/wrn40_cifar10_best.pth"
FINETUNED  = "checkpoints/wrn40_energy_ft.pth"
OOD_NPY    = "data/300K_random_images.npy"
DEVICE     = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}", flush=True)

# CIFAR-10 channel statistics (same normalisation as pre-training)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# In-distribution: CIFAR-10 training set
cifar10_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
id_loader = DataLoader(cifar10_train, batch_size=BATCH_IN, shuffle=True,
                       num_workers=2, pin_memory=True)

# Auxiliary OOD: 300K Random Images
print("Loading 300K Random Images...", flush=True)
if not os.path.exists(OOD_NPY):
    raise FileNotFoundError(
        f"{OOD_NPY} not found.\n"
        "Download with:\n"
        "  mkdir -p data && wget -P data/ "
        "https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy"
    )
raw = np.load(OOD_NPY)             # (300000, 32, 32, 3) uint8
print(f"  Loaded: {raw.shape}", flush=True)

# Apply same normalisation as CIFAR-10
t_mean = torch.tensor(MEAN).view(3, 1, 1)
t_std  = torch.tensor(STD).view(3, 1, 1)
ood_t  = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div(255.0)
ood_t  = (ood_t - t_mean) / t_std

ood_loader = DataLoader(TensorDataset(ood_t), batch_size=BATCH_OUT, shuffle=True,
                        num_workers=0, pin_memory=True)

# Load pre-trained model
model = WideResNet(depth=40, widen_factor=2, num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(PRETRAINED, map_location=DEVICE))
print(f"Loaded pre-trained weights from {PRETRAINED}", flush=True)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
ce_loss   = nn.CrossEntropyLoss()

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    ood_iter   = iter(ood_loader)

    for x_in, y_in in id_loader:
        x_in = x_in.to(DEVICE)
        y_in = y_in.to(DEVICE)

        # Grab a batch of auxiliary OOD data
        try:
            (x_out,) = next(ood_iter)
        except StopIteration:
            ood_iter = iter(ood_loader)
            (x_out,) = next(ood_iter)
        x_out = x_out.to(DEVICE)

        logits_in  = model(x_in)
        logits_out = model(x_out)

        # Standard cross-entropy on in-distribution
        l_ce = ce_loss(logits_in, y_in)

        # Energy scores: E(x) = -logsumexp(f(x))  [Eq. 4]
        e_in  = -torch.logsumexp(logits_in,  dim=1)
        e_out = -torch.logsumexp(logits_out, dim=1)

        # Energy margin loss [Eq. 9-10]: penalise ID above m_in, OOD below m_out
        l_energy = (torch.clamp(e_in  - M_IN,  min=0) ** 2).mean() + \
                   (torch.clamp(M_OUT - e_out, min=0) ** 2).mean()

        loss = l_ce + LAMBDA * l_energy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"loss={total_loss/len(id_loader):.4f} | "
          f"lr={scheduler.get_last_lr()[0]:.5f}", flush=True)

torch.save(model.state_dict(), FINETUNED)
print(f"\nSaved fine-tuned model → {FINETUNED}")
