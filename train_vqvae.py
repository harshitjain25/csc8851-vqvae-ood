import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vqvae import VQVAE
import os

print("STARTING TRAINING...", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, flush=True)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

os.makedirs("checkpoints", exist_ok=True)

transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

model = VQVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0

    for i, (x, _) in enumerate(loader):
        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}", flush=True)

        x = x.to(device)

        recon, vq_loss = model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}", flush=True)

    torch.save(model.state_dict(), f"checkpoints/vqvae_epoch_{epoch+1}.pth")
    print(f"Saved checkpoint for epoch {epoch+1}", flush=True)

torch.save(model.state_dict(), "checkpoints/vqvae.pth")
print("Model saved!", flush=True)