#!/usr/bin/env bash
# Full pipeline — run from the project root.
# Each step must succeed before the next begins.
set -e

echo "=== Step 1: Train VQ-VAE ==="
python train_vqvae.py

echo "=== Step 2: Train baseline CNN ==="
python train_baseline.py

echo "=== Step 3: Extract latent codes ==="
python extract_latents.py

echo "=== Step 4: Train latent energy model (hinge loss) ==="
python train_energy.py

echo "=== Step 5: Train latent MLP classifier ==="
python train_mlp.py

echo "=== Step 6: Evaluate baseline CNN ==="
python evaluate_baseline.py

echo "=== Step 7: Evaluate latent models ==="
python evaluate_ood.py

echo ""
echo "All done. Results written to outputs/"
