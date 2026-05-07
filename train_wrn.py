"""
Pre-trains WideResNet-40-2 on CIFAR-10.
  - 100 epochs, SGD + cosine LR decay (lr=0.1 → 0)
  - Standard CIFAR-10 normalization + random crop/flip augmentation
  - Saves best checkpoint by validation accuracy
  - Resumes automatically from latest checkpoint if one exists

Expected runtime: ~90 min on Colab T4 GPU.
Target: ~94% CIFAR-10 test accuracy.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.wideresnet import WideResNet

# ── Config ──────────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 128
LR         = 0.1
MOMENTUM   = 0.9
WD         = 5e-4
CKPT_BEST  = "checkpoints/wrn40_cifar10_best.pth"
CKPT_LAST  = "checkpoints/wrn40_cifar10_last.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}", flush=True)
os.makedirs("checkpoints", exist_ok=True)

# CIFAR-10 channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=True)

model     = WideResNet(depth=40, widen_factor=2, num_classes=10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# Resume from last checkpoint if available
start_epoch = 1
best_acc    = 0.0
if os.path.exists(CKPT_LAST):
    ckpt = torch.load(CKPT_LAST, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_acc    = ckpt.get("best_acc", 0.0)
    print(f"Resumed from epoch {ckpt['epoch']} (best_acc={best_acc:.4f})", flush=True)

for epoch in range(start_epoch, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    scheduler.step()

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_correct += model(x).argmax(1).eq(y).sum().item()
            val_total   += x.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch:03d}/{EPOCHS} | "
          f"loss={total_loss/total:.4f} | "
          f"train_acc={correct/total:.4f} | "
          f"val_acc={val_acc:.4f} | "
          f"lr={scheduler.get_last_lr()[0]:.5f}", flush=True)

    # Save last checkpoint (for resuming)
    torch.save({
        "epoch": epoch, "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, CKPT_LAST)

    # Save best checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"  → New best: {best_acc:.4f} — saved to {CKPT_BEST}", flush=True)

print(f"\nDone. Best val_acc={best_acc:.4f}")
