# Complete Project Summary
## Improving Energy-Based Out-of-Distribution Detection Using Generative Modeling
### CSC 8851 — Harshit Jain & Parsh Jadon

---

# PART 1: WHAT IS THE PROBLEM WE ARE SOLVING?

## The Big Idea — In Plain Words

Imagine you train an AI to recognize cats and dogs. It learns very well.
Now you show it a picture of a pizza.

A smart AI should say: **"I don't know what this is."**
But most AIs will confidently say: **"That's a dog — 97% sure."**

This is the **overconfidence problem**, and it is very dangerous in real life:
- A self-driving car that confidently misclassifies a child as a tree
- A medical AI that confidently diagnoses a healthy person as sick
- An industrial inspection system that misses defects

**Out-of-Distribution (OOD) Detection** is the field that tries to solve this.
The goal is to build a system that can say: *"I have never seen anything like this before — I am not sure."*

---

# PART 2: WHAT HAS BEEN DONE BEFORE? (The Baseline Paper)

## The Paper We Reproduced

**Title:** Energy-based Out-of-Distribution Detection
**Authors:** Liu, Wang, Owens, Li
**Published:** NeurIPS 2020 (one of the top AI conferences in the world)

## What They Proposed

Most people used **softmax probabilities** to detect OOD images. The problem with this is that softmax scores are always forced to sum to 1, making the model always sound confident even when it shouldn't be.

Liu et al. (2020) proposed using the **Energy Score** instead.

### What is the Energy Score?

The energy score is a single number calculated from the raw output of the classifier:

```
Energy(x) = -log( sum of e^(logit_i) for all classes i )
```

**In plain words:**
- Low energy = the image looks very much like one of the known classes
- High energy = the image doesn't fit any known class well

So: **high energy = likely OOD image**.

### What Model Did They Use?

They used **WideResNet-40-2 (WRN-40-2)**:
- A deep neural network with 40 layers
- "40" = 40 layers deep, "2" = 2× wider than standard
- Trained on CIFAR-10 (10 categories of everyday images)
- Gets ~94% accuracy on CIFAR-10

### Did They Also Fine-Tune the Model?

Yes. They also fine-tuned WRN-40-2 with a special training trick:
- Fed the model "fake OOD" training images (300,000 random images from the internet)
- Forced real images to have low energy, fake OOD to have high energy
- The hope: the model learns to better separate real from fake

---

# PART 3: WHAT DID WE PROPOSE? (Our Extension)

## Our Key Idea

The baseline asks: *"Does this image look like one of my 10 known classes?"*

We asked a different question: **"Does this image look like a CIFAR-10 image at all?"**

These are very different questions:
- The first relies on what the classifier learned about class labels
- The second relies on what the image actually *looks like* — its visual structure

To answer our question, we used a **VQ-VAE (Vector Quantized Variational Autoencoder)**.

## What is a VQ-VAE?

A VQ-VAE is a generative model — it learns to compress images into a small "code" and then reconstruct them back.

**Think of it like a visual dictionary:**
- The VQ-VAE learned from CIFAR-10 images only
- It built a vocabulary of 512 "visual words" (codebook entries)
- When you give it any image, it translates it into a sequence of these visual words

**Why does this help with OOD detection?**
- The vocabulary was built ONLY from CIFAR-10 images
- When you show it an SVHN digit or CIFAR-100 image, it forces the image into a vocabulary it was never designed for
- The resulting "translation" looks awkward and unusual
- Our detectors learned to recognize this awkwardness

## Our Three Detectors

We built three separate detectors that all take the VQ-VAE's code as input:

### Detector 1: Energy MLP (Unsupervised)
- A small neural network (3 layers, 256 units each)
- Takes the 4096-dimensional VQ-VAE code
- Outputs a single energy score (lower = more normal, higher = more OOD)
- Trained with a **contrastive hinge loss**:
  - Real CIFAR-10 codes → push energy LOW
  - Fake "pseudo-OOD" codes → push energy HIGH
  - **No real OOD data was used during training!**

### Detector 2: OOD MLP (Supervised)
- Same structure as Energy MLP
- But trained with binary classification (0 = normal, 1 = OOD)
- Uses standard binary cross-entropy loss
- Trained on CIFAR-10 codes (labeled 0) and fake OOD codes (labeled 1)

### Detector 3: Combined Score
- Blends Energy MLP and OOD MLP scores together
- Formula: `score = α × energy_score + (1 − α) × mlp_score`
- Tried two approaches:
  - Fixed α = 0.5 (equal blend)
  - Learned α* (grid search to find the best value)

## How Did We Make Fake OOD Data?

Since we didn't want to use real OOD data during training (that would be cheating), we created **pseudo-OOD** from CIFAR-10 codes:

1. **Spatial Shuffle (50%)**: Take a real CIFAR-10 code and randomly scramble the spatial positions. Like taking a sentence and randomly rearranging the words — all valid words, but meaningless order.

2. **Gaussian Noise (50%)**: Pure random 4096-dimensional noise vectors.

The hope was these would teach the model what "normal" vs "abnormal" codes look like.

---

# PART 4: HOW WE BUILT EVERYTHING

## The Full Technical Pipeline

```
Image (32×32×3 pixels)
    │
    ▼
VQ-VAE Encoder
    │
    ▼
Quantizer (snaps to nearest of 512 codebook entries)
    │
    ▼
4096-D Latent Code (64 spatial positions × 64 dimensions)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
Energy MLP                             OOD MLP
(unsupervised hinge loss)         (supervised BCE loss)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
           Combined Score s
                   │
                   ▼
           s > threshold?
           YES → OOD image
           NO  → Normal image
```

## What Scripts We Built (In Order)

| Script | What It Does |
|--------|-------------|
| `train_vqvae.py` | Trains the VQ-VAE on 50,000 CIFAR-10 training images |
| `extract_latents.py` | Passes all test images through VQ-VAE, saves the 4096-D codes |
| `train_energy.py` | Trains the Energy MLP with hinge loss |
| `train_mlp.py` | Trains the OOD MLP with binary cross-entropy |
| `train_wrn.py` | Trains the WideResNet-40-2 classifier from scratch |
| `train_energy_finetune.py` | Fine-tunes WRN-40-2 with energy margin loss |
| `evaluate_energy.py` | Evaluates WRN-40-2 energy scores |
| `evaluate_ood.py` | Evaluates latent Energy MLP and OOD MLP scores |
| `compare_all.py` | Unified script: runs all 5 methods, produces comparison tables and graphs |

## What Datasets We Used

| Dataset | Role | Size | Description |
|---------|------|------|-------------|
| CIFAR-10 | In-distribution (normal) | 10,000 test images | 10 classes: cats, dogs, cars, etc. |
| CIFAR-100 | Near-OOD (hard test) | 10,000 test images | 100 classes, similar visual style |
| SVHN | Far-OOD (easy test) | 26,032 test images | Street View House Numbers, very different from CIFAR-10 |

### Why Two OOD Test Sets?
- **CIFAR-100 (Near-OOD)** is the hard test. Same type of images (natural photos), just more categories. The model has to detect subtle differences.
- **SVHN (Far-OOD)** is the easy test. Digit images look completely different from natural objects. Any decent detector should catch these.

## How We Measured Performance

Three metrics, all standard in OOD detection research:

| Metric | Meaning | Better when |
|--------|---------|-------------|
| **AUROC** | How well the score separates normal from OOD images | Higher (closer to 1.0) |
| **AUPR** | Precision-Recall trade-off | Higher (closer to 1.0) |
| **FPR@95TPR** | At 95% OOD detection rate, what % of normal images get falsely flagged? | Lower (closer to 0) |

---

# PART 5: WHAT PROBLEMS DID WE RUN INTO?

## Problem 1: Wrong Baseline Model

**What happened:**
We initially trained a small 2-layer CNN as the baseline instead of WideResNet-40-2 (what the paper actually used).

**Why it was wrong:**
Our small CNN gave us weak numbers. Comparing our VQ-VAE against a weak CNN would make our results look artificially good and wouldn't be a fair comparison to the paper.

**How we fixed it:**
We trained the actual WideResNet-40-2 (40 layers, ~36 million parameters) from scratch, which got ~94% accuracy on CIFAR-10 and matched the paper's reported numbers.

**What we learned:**
Always use the exact model from the paper you're reproducing. Numbers need to be comparable.

---

## Problem 2: RAM Out-of-Memory Crash

**What happened:**
During energy fine-tuning on Google Colab, the training script loaded all 300,000 Random Images into RAM as a single tensor. This took ~3.5 GB of RAM. Colab's free tier only has ~12 GB, and the model + data was already using most of it. The notebook crashed.

**Error message:** `RuntimeError: CUDA out of memory` / process killed by kernel

**How we fixed it:**
We replaced the in-memory approach with **memory-mapped loading**:

```python
# BEFORE (crash):
ood_tensor = torch.tensor(np.load("random_images.npy"))  # loads 3.5GB at once
dataset = TensorDataset(ood_tensor)

# AFTER (fixed):
class RandomImagesDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode='r')  # never fully loaded!
    def __getitem__(self, idx):
        return (preprocess(self.data[idx]),)
```

**What we learned:**
For large datasets, always use memory-mapped loading. Never load the entire dataset into RAM as a tensor.

---

## Problem 3: macOS Multiprocessing Crash

**What happened:**
When we ran `compare_all.py` on a local Mac, it crashed immediately with:
```
RuntimeError: An attempt has been made to start a new process
before the current process has finished its bootstrapping phase.
```

**Why it happened:**
We had `num_workers=2` in the DataLoader. On macOS, Python uses "spawn" for multiprocessing (not "fork" like Linux). This requires all top-level code to be inside `if __name__ == '__main__':`. Our script had model loading code at the top level.

**How we fixed it:**
Changed `num_workers=2` to `num_workers=0` everywhere. No worker subprocesses — data loading happens on the main process. Slightly slower but stable.

**What we learned:**
macOS handles Python multiprocessing differently than Linux. Use `num_workers=0` for portability in research scripts.

---

## Problem 4: Energy Fine-Tuning Made the Baseline WORSE

**What happened:**
This was a surprising result. After fine-tuning WRN-40-2 with the energy margin loss (using 300K random images), we expected better OOD detection. Instead:

| Dataset | Before Fine-Tuning | After Fine-Tuning | Change |
|---------|-------------------|-------------------|--------|
| SVHN (Far-OOD) | **0.9343** AUROC | 0.7829 AUROC | **−15 points!** |
| CIFAR-100 (Near-OOD) | **0.8588** AUROC | 0.7440 AUROC | **−12 points!** |

Fine-tuning made the model significantly worse.

**Why did this happen?**
The energy fine-tuning method uses "300K Random Images" as proxy OOD data. These images:
- Are random crops from the internet
- Don't look like actual test OOD data (SVHN or CIFAR-100)
- So the model learned to flag these specific random textures, NOT real OOD distributions

This is a known problem with "Outlier Exposure" methods — they are sensitive to the quality of the proxy OOD data.

**What we learned:**
The choice of proxy OOD data matters enormously. If the proxy doesn't match real OOD distributions, the model overfits to the proxy and performance degrades everywhere else. The pre-trained (no fine-tuning) energy score was actually better.

---

## Problem 5: Our Latent Energy Score Was Near-Random

**What happened:**
After training the Energy MLP, we evaluated it:
- SVHN (Far-OOD): AUROC = **0.5334** (random = 0.5)
- CIFAR-100 (Near-OOD): AUROC = **0.5113** (random = 0.5)

Our energy model was barely better than flipping a coin.

**Why did this happen?**
The root cause: our **pseudo-OOD training data was too artificial**.

We trained the model to distinguish:
- Real CIFAR-10 codes (normal)
- Spatially-shuffled CIFAR-10 codes + Gaussian noise (fake OOD)

But real OOD datasets (SVHN, CIFAR-100) don't look like random noise or spatially-shuffled codes. Their codes actually look reasonably similar to CIFAR-10 codes in the VQ-VAE's latent space (because both use the same 512-word vocabulary).

The model learned: *"shuffled/noisy codes = OOD"*
But the actual test: *"SVHN/CIFAR-100 codes = OOD"*

These are very different boundaries, so the model failed.

**What we learned:**
Fake OOD that is too artificial teaches the model the wrong decision boundary. To improve this, we would need more realistic "hard negatives" — fake OOD codes that look similar to CIFAR-10 codes but aren't.

---

## Problem 6: Combining Signals Made Things Worse

**What happened:**
We tried combining the Energy MLP score with the OOD MLP score using a 50/50 blend (α = 0.5):

On Far-OOD (SVHN):
- OOD MLP alone: **0.9801 AUROC** (great!)
- Energy MLP alone: **0.5334 AUROC** (near-random)
- 50/50 blend: **0.9430 AUROC** (worse than MLP alone!)

Blending a good signal with a near-random signal made performance drop from 0.98 to 0.94.

**How we fixed it:**
We grid-searched α on a held-out validation split, testing α = 0.0, 0.1, 0.2, ..., 1.0. The optimal α* turned out to be **0.0** for both datasets — meaning: **ignore the energy signal entirely and just use the OOD MLP**.

The combined score with α* = 0.0 recovered almost all the OOD MLP's performance:
- α* combination: **0.9806 AUROC** (essentially the same as MLP alone)

**What we learned:**
Never blindly average two signals. If one signal is weak or noisy, including it with equal weight injects noise into the good signal. Always validate the blend weight.

---

# PART 6: THE FINAL RESULTS

## Complete Results Table

### Near-OOD: CIFAR-10 vs CIFAR-100 (The Hard Test)

| Method | AUROC ↑ | AUPR ↑ | FPR@95 ↓ |
|--------|---------|--------|----------|
| WRN-40-2 Energy (paper baseline) | **0.8588** | 0.8569 | 0.6253 |
| WRN-40-2 Fine-tuned (paper) | 0.7440 | 0.6534 | 0.5616 |
| VQ-VAE Latent Energy (ours) | 0.5113 | 0.5321 | 0.9550 |
| VQ-VAE Latent MLP (ours) | 0.7018 | 0.6937 | 0.8196 |
| VQ-VAE E+MLP α* (ours) | 0.7035 | 0.6963 | 0.8184 |

### Far-OOD: CIFAR-10 vs SVHN (The Easy Test)

| Method | AUROC ↑ | AUPR ↑ | FPR@95 ↓ |
|--------|---------|--------|----------|
| WRN-40-2 Energy (paper baseline) | 0.9343 | 0.9682 | 0.2577 |
| WRN-40-2 Fine-tuned (paper) | 0.7829 | 0.8206 | 0.3744 |
| VQ-VAE Latent Energy (ours) | 0.5334 | 0.7789 | 0.9518 |
| **VQ-VAE Latent MLP (ours)** | **0.9801** | **0.9920** | **0.1000** |
| VQ-VAE E+MLP α* (ours) | **0.9806** | **0.9922** | **0.0962** |

---

# PART 7: HEAD-TO-HEAD COMPARISON

## Our Best Method vs The Paper's Best Method

| Comparison | Paper's Best | Our Best | Winner |
|------------|-------------|----------|--------|
| Far-OOD AUROC (SVHN) | 0.9343 | **0.9806** | **Ours (+4.6%)** |
| Far-OOD AUPR (SVHN) | 0.9682 | **0.9922** | **Ours (+2.4%)** |
| Far-OOD FPR@95 (SVHN) | 0.2577 | **0.0962** | **Ours (−16%)** |
| Near-OOD AUROC (CIFAR-100) | **0.8588** | 0.7035 | **Paper (−15.5%)** |
| Near-OOD AUPR (CIFAR-100) | **0.8569** | 0.6963 | **Paper (−16%)** |
| Near-OOD FPR@95 (CIFAR-100) | **0.6253** | 0.8184 | **Paper (worse)** |

## Simple Takeaway

```
╔══════════════════════════════════════════════════════════════════╗
║  Far-OOD (SVHN):   OUR METHOD WINS   (+5% AUROC, FPR cut in half)  ║
║  Near-OOD (CIFAR-100):   PAPER WINS   (paper +15% AUROC)          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

# PART 8: WHY DID WE WIN ON FAR-OOD AND LOSE ON NEAR-OOD?

## Why We Won on Far-OOD (SVHN)

SVHN images are street view digit images. They look very different from CIFAR-10 natural images (animals, vehicles, etc.). When SVHN images go through the VQ-VAE (which was trained only on CIFAR-10), the encoder is forced to represent digits using a vocabulary built for natural objects. The result is a clearly "awkward" code pattern.

Our OOD MLP, trained to recognize CIFAR-10 codes as normal, could easily identify these awkward patterns as OOD.

**Bottom line:** The VQ-VAE latent space provides strong signal for visually very different OOD images.

## Why We Lost on Near-OOD (CIFAR-100)

CIFAR-100 images are natural photos just like CIFAR-10 — animals, plants, people, vehicles, landscapes. They are very similar in visual style to CIFAR-10 images. When CIFAR-100 images go through the VQ-VAE, the resulting codes look almost identical to CIFAR-10 codes because both datasets use the same types of natural image textures.

The WRN-40-2 classifier, with its 40 layers of discriminative features, can detect subtle semantic differences between CIFAR-10 and CIFAR-100 classes. Our VQ-VAE focuses on visual reconstruction, not discrimination.

**Bottom line:** The VQ-VAE latent space cannot distinguish between CIFAR-10 and CIFAR-100 because they look too similar. The discriminative classifier has an advantage here.

---

# PART 9: WHAT DID WE SUCCESSFULLY REPRODUCE?

## Reproducing the Paper's Baseline

We faithfully reproduced the Liu et al. (2020) paper's main results:

| Paper's Reported Number | Our Reproduced Number | Match? |
|------------------------|----------------------|--------|
| WRN-40-2 AUROC on SVHN ≈ 0.93 | 0.9343 | ✅ |
| WRN-40-2 AUROC on CIFAR-100 ≈ 0.86 | 0.8588 | ✅ |

**This is a significant achievement.** Many papers are difficult to reproduce because:
- Training details are not fully specified
- Different random seeds produce different results
- Hardware differences affect floating-point precision

We matched the paper's numbers, confirming our implementation is correct.

---

# PART 10: WHAT WOULD WE DO DIFFERENTLY?

## Future Improvements

### 1. Better Pseudo-OOD Training Data
Instead of random noise and spatial shuffles, use:
- **CutMix-style latent perturbations**: Mix regions of two different CIFAR-10 codes
- **Codebook index shuffling**: Shuffle only certain spatial positions
- **Cross-class mixing**: Mix codes from different CIFAR-10 classes
The goal: create "hard negatives" that are close to real CIFAR-10 codes but slightly off, forcing the model to learn a tighter decision boundary.

### 2. Larger Codebook
We used K = 512 codebook entries. A larger codebook (K = 1024 or 4096) would give the VQ-VAE more vocabulary to express fine-grained visual differences. CIFAR-100 codes might look more distinct from CIFAR-10 codes.

### 3. Stronger Encoder
Our VQ-VAE used a simple 2-layer convolutional encoder. A WideResNet-based encoder would learn richer, more discriminative features. This could close the Near-OOD gap by making the codes from different datasets more distinguishable.

### 4. Learnable Blend Weight
Instead of grid-searching α on a validation set, train α as a learned parameter end-to-end. This way the model automatically learns to weight the energy and MLP signals based on which is more informative for the current type of OOD data.

---

# PART 11: FINAL CONCLUSIONS

## What We Achieved

1. **We reproduced the paper.** Liu et al. 2020's WRN-40-2 energy score works exactly as reported. AUROC 0.93 on SVHN, 0.86 on CIFAR-100 — verified.

2. **We built a complete alternative OOD detection pipeline** using VQ-VAE latent codes. This includes training scripts, evaluation scripts, and a unified comparison tool.

3. **We beat the baseline on Far-OOD.** Our Latent MLP achieves 0.98 AUROC on SVHN vs. the baseline's 0.93 — a 5 percentage point improvement. FPR@95 dropped from 26% to 10%, meaning we catch the same percentage of OOD images with half the false alarms.

4. **We identified exactly why our method fails on Near-OOD.** CIFAR-100 codes look too similar to CIFAR-10 codes in our 512-entry codebook. This motivates a better encoder.

5. **We found that fine-tuning with proxy OOD data can hurt.** Energy fine-tuning degraded performance by ~15% on SVHN, a finding that aligns with known limitations of Outlier Exposure methods.

6. **We proved the "combining two signals" problem.** Averaging a strong signal (OOD MLP) with a weak one (Energy MLP) degrades performance. The learned α* correctly set itself to 0.0, ignoring the bad signal.

## The One-Sentence Summary

> Our VQ-VAE latent space approach is **5% better** than the paper's baseline on easy OOD images (SVHN) but **15% worse** on hard OOD images (CIFAR-100), revealing that generative model codes are powerful for visual outliers but not yet strong enough for semantically subtle ones.

---

# APPENDIX: KEY NUMBERS TO REMEMBER FOR PRESENTATION

| Fact | Number |
|------|--------|
| Baseline AUROC on SVHN | 0.93 |
| Our best AUROC on SVHN | **0.98 (+5%)** |
| Baseline FPR@95 on SVHN | 26% |
| Our best FPR@95 on SVHN | **10%** |
| Baseline AUROC on CIFAR-100 | **0.86** |
| Our best AUROC on CIFAR-100 | 0.70 (−16%) |
| α* optimal value | **0.0** (ignore energy signal) |
| Energy MLP AUROC (both datasets) | ~0.51–0.53 (near-random) |
| VQ-VAE codebook size | 512 entries |
| VQ-VAE latent code dimension | **4096-D** (64 × 64) |
| WRN-40-2 training accuracy | ~94% on CIFAR-10 |
| Fine-tuning effect on SVHN | 0.93 → 0.78 (WORSE) |
