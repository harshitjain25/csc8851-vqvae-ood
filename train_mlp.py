import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import OODMLP

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_latents(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing latent file: {path}")
    return np.load(path).astype(np.float32)

def main():
    id_lat   = load_latents("latent_codes/cifar10.npy")
    near_lat = load_latents("latent_codes/cifar100.npy")
    far_lat  = load_latents("latent_codes/svhn.npy")

    X = np.concatenate([id_lat, near_lat, far_lat], axis=0)
    y = np.concatenate([
        np.zeros(len(id_lat), dtype=np.float32),
        np.ones(len(near_lat), dtype=np.float32),
        np.ones(len(far_lat), dtype=np.float32),
    ], axis=0)

    D = X.shape[1]
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = OODMLP(dim=D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 10
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for z, t in loader:
            z = z.to(DEVICE)
            t = t.to(DEVICE)

            logits = model(z)
            loss = loss_fn(logits, t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * z.size(0)

        print(f"Epoch {ep:02d} | loss={total/len(ds):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"dim": D, "state_dict": model.state_dict()}, "checkpoints/ood_mlp.pth")
    print("Saved: checkpoints/ood_mlp.pth")

if __name__ == "__main__":
    main()