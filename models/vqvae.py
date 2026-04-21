import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Discretizes encoder output into codebook vectors.
    Uses straight-through estimator so gradients flow back to encoder.
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W)  where C == embedding_dim
        z_perm = z.permute(0, 2, 3, 1).contiguous()          # (B, H, W, C)
        z_flat = z_perm.view(-1, self.embedding_dim)          # (B*H*W, C)

        # Squared L2 distances to every codebook entry
        d = (
            (z_flat ** 2).sum(1, keepdim=True)
            + (self.embedding.weight ** 2).sum(1)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )  # (B*H*W, K)

        indices = d.argmin(1)                                  # (B*H*W,)
        z_q = self.embedding(indices).view(z_perm.shape)      # (B, H, W, C)

        # Codebook loss + commitment loss (Eq. 4 from paper)
        loss = (
            F.mse_loss(z_q.detach(), z_perm)           # codebook term
            + self.beta * F.mse_loss(z_q, z_perm.detach())  # commitment term
        )

        # Straight-through: copy gradients from z_q to z
        z_q_st = z_perm + (z_q - z_perm).detach()
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()    # (B, C, H, W)

        return z_q_st, loss, indices


class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, beta: float = 0.25):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 4, 2, 1),
            nn.ReLU(),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ze = self.encoder(x)
        zq, vq_loss, _ = self.vq(ze)
        x_recon = self.decoder(zq)
        return x_recon, vq_loss

    def encode(self, x):
        """Returns quantized latent vectors, flattened to (B, D)."""
        ze = self.encoder(x)
        zq, _, _ = self.vq(ze)
        return zq.view(zq.size(0), -1)   # (B, embedding_dim * H * W)
