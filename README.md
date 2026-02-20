# csc8851-vqvae-ood
Latent Energy-Based OOD Detection using VQ-VAE (CSC 8851 Project)

## Team Split
- Parsh: VQ-VAE training + latent extraction
  - files: models/vqvae.py, train_vqvae.py, extract_latents.py
  - outputs (shared via Drive): latent_codes/cifar10.npy, latent_codes/cifar100.npy, latent_codes/svhn.npy

- Harshit: Energy model + MLP + evaluation + report
  - files: models/energy_model.py, models/mlp.py, train_energy.py, train_mlp.py, evaluate_ood.py, scripts/*