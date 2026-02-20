import torch
import torch.nn as nn

class OODMLP(nn.Module):
    """
    Binary classifier in latent space.
    Output: logits (B,1). Use sigmoid for probability.
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