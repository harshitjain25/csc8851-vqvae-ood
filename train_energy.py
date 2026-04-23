"""
Trains the latent energy model using a contrastive hinge loss.

- ID data : CIFAR-10 latents (real in-distribution)
- OOD data: *latent-perturbation* pseudo-OOD (near-manifold negatives)
            half the batch is spatial-shuffled CIFAR-10 codes — uses
            real codebook entries in structurally wrong spatial arrangements;
            half remains Gaussian noise as a far-manifold anchor.
- No OOD labels are used, matching the paper's unsupervised framing.

Loss (Eqs. 6-8 in paper):
  L_in  = E[max(0, E(z) - m_in )^2]   m_in  = -10
  L_out = E[max(0, m_out - E(z))^2]   m_out =  -5
  L     = L_in + L_out
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.energy_model import EnergyMLP

print("STARTING ENERGY TRAINING...", flush=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
M_IN   = -10.0   # in-distribution energy target (push below this)
M_OUT  =  -5.0   # OOD energy target             (push above this)
LR     =  1e-3
EPOCHS = 10
BATCH  = 512
C, H, W = 64, 8, 8   # VQ-VAE latent shape: channels × spatial (flattens to 4096)
# ──────────────────────────────────────────────────────────────────────────


def spatial_shuffle(z_flat: torch.Tensor) -> torch.Tensor:
    """
    Permute the H*W spatial positions within each sample, independently.

    Uses real codebook entries in structurally impossible arrangements —
    e.g. 'sky' codes where 'ground' codes should be. This is a much stronger
    pseudo-OOD signal than Gaussian noise because it stays on the data
    manifold (same codes, same per-feature statistics) while destroying the
    spatial coherence the VQ-VAE encoder learned.
    """
    B = z_flat.size(0)
    z = z_flat.view(B, C, H * W)                         # (B, 64, 64)
    idx = torch.argsort(torch.rand(B, H * W, device=z.device), dim=1)  # (B, 64)
    z = torch.gather(z, 2, idx.unsqueeze(1).expand(B, C, H * W))
    return z.view(B, -1)


def make_pseudo_ood(z_id: torch.Tensor) -> torch.Tensor:
    """Half-batch spatial-shuffle + half-batch Gaussian noise."""
    B = z_id.size(0)
    half = B // 2
    z_near = spatial_shuffle(z_id[:half])                # near-manifold
    z_far  = torch.randn_like(z_id[half:])               # far-manifold
    return torch.cat([z_near, z_far], dim=0)

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# 1. Load only CIFAR-10 latents (no OOD labels needed)
cifar10 = np.load("latent_codes/cifar10.npy").astype(np.float32)
D = cifar10.shape[1]
print(f"CIFAR-10 latents: {cifar10.shape}", flush=True)

# 2. Normalize using CIFAR-10 statistics
mean = cifar10.mean(axis=0)
std  = cifar10.std(axis=0) + 1e-8
cifar10_norm = (cifar10 - mean) / std

X_id = torch.from_numpy(cifar10_norm)
loader = DataLoader(TensorDataset(X_id), batch_size=BATCH, shuffle=True)

# 3. Model + optimiser
model = EnergyMLP(dim=D).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for (z_id,) in loader:
        z_id = z_id.to(device)

        # Mixed pseudo-OOD: half spatial-shuffle (near) + half Gaussian (far)
        z_neg = make_pseudo_ood(z_id)

        e_in  = model(z_id)     # (B, 1)
        e_out = model(z_neg)    # (B, 1)

        # Hinge loss: push ID energy below M_IN, noise energy above M_OUT
        L_in  = (torch.clamp(e_in  - M_IN,  min=0) ** 2).mean()
        L_out = (torch.clamp(M_OUT - e_out, min=0) ** 2).mean()
        loss  = L_in + L_out

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * z_id.size(0)

    print(f"Epoch {epoch:02d} | loss={total_loss / len(X_id):.6f}", flush=True)

# 5. Save – use {"dim": D, "state_dict": ...} format for load_model_checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save({"dim": D, "state_dict": model.state_dict()}, "checkpoints/energy_mlp.pth")
print("Saved: checkpoints/energy_mlp.pth", flush=True)
