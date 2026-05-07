import torch
import torch.nn as nn

class EnergyMLP(nn.Module):
    """
    Input: latent vector z of shape (B, D)
    Output: energy of shape (B, 1)
    Higher energy => more OOD
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z)