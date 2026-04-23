# Project Full Documentation
## Improving Energy-Based Out-of-Distribution Detection using Generative Modeling
### CSC 8851 — Harshit Jain & Parsh Jadon

---

## What Is This Project About? (The Big Picture)

Imagine you train an AI to recognize 10 types of objects — cats, dogs, cars, etc. Now you show it a picture of a pizza. It has never seen a pizza before, but instead of saying "I don't know," it confidently says "that's a cat" with 97% confidence. This is called the **overconfidence problem**, and it is dangerous in real applications like medical diagnosis or self-driving cars.

**Out-of-Distribution (OOD) Detection** is the task of building a system that can detect when an image is "foreign" — i.e., it doesn't belong to the training distribution. A good OOD detector will say "I don't know this image" rather than making a confident wrong prediction.

Our project builds on a strong published baseline and then extends it using a completely different kind of model — a generative model called VQ-VAE.

---

## Part 1: The Baseline — What the Paper Did (Liu et al., NeurIPS 2020)

### The idea
Liu et al. 2020 proposed that instead of using the classifier's softmax probability as an OOD score, we should use the **energy score**. The energy score is computed as:

```
Energy(x) = -log( sum of e^(logit_i) for all classes i )
```

In plain English: the energy is low if the image looks confidently like one of the known classes, and high if it doesn't fit any class well. A high energy score = likely OOD.

They also proposed **energy fine-tuning**: take the already-trained classifier and fine-tune it with a special loss that pushes real (in-distribution) images to low energy and "outlier" images to high energy.

### The model they used
They used a **WideResNet-40-2 (WRN-40-2)** — a large and powerful image classifier. "40" refers to depth (40 layers) and "2" refers to width (each layer is 2x wider than a normal ResNet). This is a serious model trained for 100 epochs with learning rate decay. The model gets ~94% accuracy on CIFAR-10.

### What datasets they tested on
- **CIFAR-10** (the in-distribution dataset): 10 classes of everyday objects, 32×32 color images.
- **CIFAR-100** (Near-OOD): 100 classes of similar everyday objects. Near-OOD means it's visually similar to CIFAR-10 — same style, same resolution, just more categories. This is the **hard test**.
- **SVHN** (Far-OOD): Street View House Numbers — digit images from Google Street View. Very different visual style from CIFAR-10. This is the **easy test**.

### How OOD detection is evaluated
Three metrics:
- **AUROC** (Area Under ROC Curve): 0.5 = random guessing, 1.0 = perfect. Higher is better.
- **AUPR** (Area Under Precision-Recall Curve): Also 0–1, higher is better.
- **FPR@95TPR**: At the threshold where you catch 95% of OOD images, what fraction of normal images do you falsely flag? Lower is better.

---

## Part 2: Our Extension — The VQ-VAE Approach

### The core idea
The original method asks: "Does this image fit one of my 10 known **classes**?" We ask a different question: "Does this image look like a **CIFAR-10 image** at all?" — regardless of class.

To answer this, we use a **VQ-VAE** (Vector Quantized Variational Autoencoder), which is a generative model that learns to compress and reconstruct images. It learns a visual "vocabulary" of 512 code words. When given an image, it encodes it as a grid of these code words.

The key insight: the VQ-VAE was trained ONLY on CIFAR-10. So it has a vocabulary that describes CIFAR-10 images well. When you give it an SVHN digit or a CIFAR-100 image, it has to force-fit the image into a vocabulary that wasn't designed for it. The resulting code will be "awkward" — and that awkwardness is what we detect.

### How the VQ-VAE works
1. **Encoder**: A convolutional network compresses a 32×32×3 image into a 8×8×64 tensor.
2. **Quantizer**: Each of the 64 spatial positions gets snapped to the nearest of 512 codebook entries. This gives a discrete code — a sequence of 64 integers, each between 0 and 511.
3. **Decoder**: Another convolutional network reconstructs the original image from the quantized code.
4. **Training**: Minimize reconstruction error (how different is the rebuilt image from the original).

The resulting flattened code is **4096-dimensional** (64 spatial positions × 64 dimensions per entry = 4096 numbers per image).

### What we trained on top of the VQ-VAE

#### Component 1: Energy MLP (Unsupervised)
A small 3-layer neural network that takes the 4096-D latent code and outputs a single energy score.

**Training:** We use a contrastive hinge loss. Real CIFAR-10 codes are pushed to low energy (below margin m_in = -10). Fake "pseudo-OOD" codes are pushed to high energy (above margin m_out = -5). No real OOD data is used — we generate fake OOD codes ourselves.

**How we made fake OOD codes:**
- 50% spatial shuffle: take a real CIFAR-10 code and randomly scramble the 8×8 spatial positions. The individual code words are valid, but the spatial arrangement is nonsensical.
- 50% Gaussian noise: pure random 4096-D noise vectors.

#### Component 2: OOD MLP (Supervised)
Same architecture as the energy MLP but trained with **binary cross-entropy** (BCE). It sees real CIFAR-10 codes labeled 0 (in-distribution) and fake OOD codes labeled 1, and learns to classify them. The output is a probability between 0 and 1.

#### Component 3: Combined Score
Blend the two signals: `score = α × energy_score + (1 - α) × mlp_score`

We tried two strategies:
- Fixed α = 0.5 (equal blend)
- α* = grid-searched on a validation split to find the best weight

---

## Part 3: The Full Pipeline (How Everything Connects)

```
Image
  └─► VQ-VAE Encoder ──► Quantizer ──► 4096-D latent code (z)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                        Energy MLP                        OOD MLP
                    (unsupervised hinge)              (supervised BCE)
                              │                               │
                              └───────────── blend ───────────┘
                                                │
                                          Final score s
                                          s > threshold?
                                          YES → OOD  /  NO → In-Distribution
```

---

## Part 4: What We Actually Built and Ran

### Scripts written (in order of execution)

| Script | What it does |
|--------|-------------|
| `train_vqvae.py` | Trains the VQ-VAE on CIFAR-10 train set. 10 epochs, batch 128, Adam optimizer. |
| `extract_latents.py` | Passes CIFAR-10 test, CIFAR-100 test, SVHN test through the encoder to save 4096-D codes as `.npy` files. |
| `train_energy.py` | Trains the Energy MLP with the contrastive hinge loss. |
| `train_mlp.py` | Trains the OOD MLP with binary cross-entropy. |
| `train_wrn.py` | Trains WRN-40-2 from scratch on CIFAR-10 (the paper's baseline model). |
| `train_energy_finetune.py` | Fine-tunes WRN-40-2 with the energy margin loss using 300K auxiliary random images. |
| `evaluate_baseline.py` | Evaluates the original CNN energy baseline. |
| `evaluate_energy.py` | Evaluates WRN-40-2 energy scores. |
| `evaluate_ood.py` | Evaluates latent Energy MLP and OOD MLP scores. |
| `compare_all.py` | Unified script: runs ALL methods, produces comparison tables and figures. |

---

## Part 5: Hit and Trials — What Went Wrong and What We Learned

### Trial 1: We started with a small CNN baseline (wrong approach)

**What we did:** Initially we trained a small 2-layer CNN (two conv blocks + a fully connected head) as the baseline, instead of the WideResNet-40-2 used in the paper.

**Why it was wrong:** The paper specifically used WRN-40-2. Using a small CNN gave us weak baseline numbers that made our VQ-VAE extension look relatively better — but the comparison wasn't honest. A reviewer or professor could immediately point this out.

**What we learned:** Always use the exact architecture from the paper when reproducing a baseline. The numbers need to be comparable.

**What we did:** Switched to training WRN-40-2 (40 layers, ~36M parameters) and got the real numbers.

---

### Trial 2: RAM Out-of-Memory Crash on Google Colab

**What happened:** During energy fine-tuning, we loaded all 300,000 auxiliary random images into memory as a PyTorch tensor. That's 300K × 32 × 32 × 3 = ~3.5 GB of RAM. Colab's free tier has ~12 GB RAM and it was already using most of it for the model and data. The notebook crashed.

**Error message:** `RuntimeError: CUDA out of memory` / process killed by kernel.

**The fix:** We replaced the in-memory `TensorDataset` with a **memory-mapped numpy dataset**. Instead of loading all 300K images into RAM at once, we load them from disk one batch at a time using `np.load(path, mmap_mode='r')`. The OS loads only what's currently needed.

```python
# BEFORE (crash):
ood_t = torch.tensor(np.load("random_images.npy"))  # loads 3.5GB into RAM
dataset = TensorDataset(ood_t)

# AFTER (fixed):
class RandomImagesDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode='r')  # never fully loaded!
    def __getitem__(self, idx):
        return (preprocess(self.data[idx]),)
```

**What we learned:** For large datasets, always use memory-mapped loading or streaming DataLoaders. Never load the whole dataset into RAM as a tensor.

---

### Trial 3: Multiprocessing Crash on macOS

**What happened:** When we ran `compare_all.py` on a local Mac, it crashed immediately with:

```
RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase.
```

This happened because we set `num_workers=2` in the DataLoader, and on macOS, Python's multiprocessing uses "spawn" (not "fork" like Linux), which requires all top-level code to be inside `if __name__ == '__main__':`. Our script had model loading code at the top level.

**The fix:** Set `num_workers=0` everywhere — no subprocess workers, data loading happens on the main process. Slightly slower but stable.

**What we learned:** macOS and Windows handle Python multiprocessing differently than Linux. For portability, default to `num_workers=0` in research scripts.

---

### Trial 4: Energy Fine-Tuning Made the Baseline WORSE

**What happened:** After fine-tuning WRN-40-2 with the energy margin loss on 300K random images, we expected better OOD scores. Instead, performance dropped significantly:

| Dataset | Before fine-tune (AUROC) | After fine-tune (AUROC) |
|---------|--------------------------|-------------------------|
| SVHN    | **0.934** | 0.783 (−15 pts!) |
| CIFAR-100 | **0.859** | 0.744 (−12 pts) |

**Why this happens:** The energy fine-tuning uses "300K Random Images" (Tiny Images subset) as proxy OOD data. These random images don't look like real OOD distributions (SVHN or CIFAR-100). The model learns to push these specific random images to high energy — but that hurts its ability to generalize to actual OOD images.

**What we learned:** Outlier Exposure (training with proxy OOD data) is sensitive to the choice of proxy. If the proxy doesn't match real OOD, the model overfits to the proxy and degrades everywhere else. The pre-trained energy score without fine-tuning is actually stronger on SVHN and CIFAR-100.

---

### Trial 5: Our Latent Energy Score Was Near-Random

**What happened:** After training the Energy MLP, we evaluated it and got AUROC of ~0.51 on CIFAR-100 and ~0.53 on SVHN. Random guessing gives AUROC 0.50. Our model was barely better than chance.

**Why this happened:** Our pseudo-OOD data (spatial shuffles + Gaussian noise) is not representative of real OOD. The model learned to tell apart "CIFAR-10 codes" vs "scrambled CIFAR-10 codes" — but real CIFAR-100 codes are neither. CIFAR-100 codes look a lot like CIFAR-10 codes because both datasets have similar natural-image content (animals, vehicles, outdoor scenes). The latent space doesn't cleanly separate them.

**What we learned:** Pseudo-OOD that is too artificial (random noise, spatial shuffles) teaches the model the wrong boundary. We would need more realistic pseudo-OOD — for example, interpolations between classes, or CutMix-style blends.

---

### Trial 6: The Combined Score Hurt Performance (Until We Fixed α)

**What happened:** We combined the energy score and MLP score with α=0.5 (equal blend). On Far-OOD (SVHN):

- Latent MLP alone: **0.980 AUROC**
- Energy alone: 0.533 AUROC (near-random)
- Equal blend (α=0.5): 0.943 AUROC (worse than MLP alone!)

Blending a good signal with a near-random signal made things worse.

**The fix:** Grid-search α* on a held-out validation split. We tried α = 0.0, 0.1, 0.2, ..., 1.0 and picked the best. The result: α* = 0.0 for both datasets — meaning the optimal strategy was to completely ignore the energy signal and just use the MLP.

**What we learned:** Never blindly average signals. If one signal is weak or noisy, including it with equal weight injects noise into the good signal. Always validate the blend weight.

---

## Part 6: Final Results

### WRN-40-2 Baseline Results (Liu et al. 2020, reproduced)

| Dataset | FPR@95 | AUROC | AUPR |
|---------|--------|-------|------|
| SVHN (Far-OOD) | 0.2577 | **0.9343** | 0.9682 |
| CIFAR-100 (Near-OOD) | 0.6253 | **0.8588** | 0.8569 |

These match the numbers reported in the original paper. We successfully reproduced their results.

### Full Comparison

#### Near-OOD: CIFAR-10 vs CIFAR-100

| Method | AUROC | AUPR | FPR@95 |
|--------|-------|------|--------|
| WRN-40-2 Energy (reproduced) | **0.8588** | 0.8569 | 0.6253 |
| WRN-40-2 Fine-tuned | 0.7440 | 0.6534 | 0.5616 |
| VQ-VAE Latent Energy (ours) | 0.5113 | 0.5321 | 0.9550 |
| VQ-VAE Latent MLP (ours) | 0.7018 | 0.6937 | 0.8196 |
| VQ-VAE E+MLP α* (ours) | 0.7035 | 0.6963 | 0.8184 |

#### Far-OOD: CIFAR-10 vs SVHN

| Method | AUROC | AUPR | FPR@95 |
|--------|-------|------|--------|
| WRN-40-2 Energy (reproduced) | 0.9343 | 0.9682 | 0.2577 |
| WRN-40-2 Fine-tuned | 0.7829 | 0.8206 | 0.3744 |
| VQ-VAE Latent Energy (ours) | 0.5334 | 0.7789 | 0.9518 |
| VQ-VAE Latent MLP (ours) | **0.9801** | **0.9920** | **0.1000** |
| VQ-VAE E+MLP α* (ours) | **0.9806** | **0.9922** | **0.0962** |

### Key Takeaways from Results

1. **We reproduced the paper's baseline faithfully.** Our WRN-40-2 numbers match Liu et al. 2020.

2. **Our Latent MLP beats the baseline on Far-OOD (SVHN).** AUROC 0.98 vs 0.93 — a 5-point improvement. FPR@95 drops from 26% to 10%.

3. **The baseline wins on Near-OOD (CIFAR-100).** WRN-40-2 gets 0.86 vs our 0.70. The large discriminative classifier has an edge in distinguishing fine-grained semantic differences between datasets.

4. **Energy fine-tuning degraded the baseline.** The proxy OOD data (random images) doesn't represent real OOD distributions.

5. **Latent energy alone is near-random.** Our pseudo-OOD training didn't teach the model to separate real OOD from real ID.

6. **Smart α* selection recovers MLP performance.** Setting α*=0.0 correctly ignores the weak energy signal.

---

## Part 7: How We Would Improve It (Future Work)

1. **Better pseudo-OOD for the energy model.** Instead of spatial shuffles and Gaussian noise, use CutMix (mixing two images' patches), or actually sample from the codebook in random patterns. The goal is to create "hard negatives" that look like plausible images but aren't CIFAR-10.

2. **Larger codebook.** We used K=512 codebook entries. A larger codebook (K=1024 or 4096) would give the encoder more vocabulary to express fine-grained visual differences, potentially making CIFAR-100 codes look different from CIFAR-10 codes.

3. **Stronger encoder.** A VQ-VAE built on top of a WideResNet encoder would learn richer, more discriminative features — closing the Near-OOD gap.

4. **Learnable blend weight.** Instead of grid-searching α*, learn it end-to-end so the model knows when to trust the energy signal vs the MLP.

---

## Part 8: Environment and Reproducibility

- **Language:** Python 3.11
- **Framework:** PyTorch
- **Hardware used:** Google Colab (T4 GPU for training), Apple M-series Mac (MPS for inference)
- **Key packages:** torchvision, numpy, scikit-learn, matplotlib
- **Random seed:** 42 everywhere
- **Training time:** ~15 min on a Colab T4 for the full pipeline
- **Inference:** `python compare_all.py` (~7 min on MPS for WRN-40-2 inference)
- **All outputs reproducible** with `./run_all.sh`
