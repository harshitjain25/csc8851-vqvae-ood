import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vqvae import VQVAE
import os   

os.makedirs("latent_codes", exist_ok=True)  
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = VQVAE().to(device)
model.load_state_dict(torch.load("checkpoints/vqvae.pth", map_location=device))
model.eval()

transform = transforms.ToTensor()


def extract(dataset, name):
    loader = DataLoader(dataset, batch_size=256)
    latents = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            z = z.view(z.size(0), -1)
            latents.append(z.cpu().numpy())

    latents = np.concatenate(latents)
    np.save(f"latent_codes/{name}.npy", latents)
    print(f"Saved {name}")


cifar10 = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
cifar100 = datasets.CIFAR100("./data", train=False, download=True, transform=transform)
svhn = datasets.SVHN("./data", split="test", download=True, transform=transform)

extract(cifar10, "cifar10")
extract(cifar100, "cifar100")
extract(svhn, "svhn")