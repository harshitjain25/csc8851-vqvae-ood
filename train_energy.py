import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.energy_model import EnergyMLP

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_latents(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing latent file: {path}")
    return np.load(path).astype(np.float32)

def main():
    # Train only on CIFAR-10 latents
    X = load_latents("latent_codes/cifar10.npy")
    D = X.shape[1]

    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = EnergyMLP(dim=D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Margin hyperparameters: push ID energy below m_in, pseudo-OOD above m_out
    m_in  = -10.0
    m_out =  -5.0

    epochs = 10
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (z,) in loader:
            z = z.to(DEVICE)

            # Gaussian noise as pseudo-OOD (same shape as real latents)
            z_noise = torch.randn_like(z)

            energy_id  = model(z)        # (B,1)
            energy_ood = model(z_noise)  # (B,1)

            # Hinge loss: penalise ID energy above m_in, OOD energy below m_out
            loss_in  = torch.clamp(energy_id  - m_in,  min=0).pow(2).mean()
            loss_out = torch.clamp(m_out - energy_ood,  min=0).pow(2).mean()
            loss = loss_in + loss_out

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * z.size(0)

        print(f"Epoch {ep:02d} | loss={total/len(ds):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"dim": D, "state_dict": model.state_dict()}, "checkpoints/energy_mlp.pth")
    print("Saved: checkpoints/energy_mlp.pth")

if __name__ == "__main__":
    main()