import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.energy_model import EnergyMLP

print("STARTING ENERGY TRAINING...", flush=True)

# ----------------------------
# 1. Load latent data
# ----------------------------
cifar10 = np.load("latent_codes/cifar10.npy")
cifar100 = np.load("latent_codes/cifar100.npy")
svhn = np.load("latent_codes/svhn.npy")

# ----------------------------
# 2. Create dataset
# ----------------------------
X = np.concatenate([cifar10, cifar100, svhn], axis=0)

y = np.concatenate([
    np.zeros(len(cifar10)),                     # ID = 0
    np.ones(len(cifar100) + len(svhn))          # OOD = 1
], axis=0)

# ----------------------------
# 3. Normalize (IMPORTANT FIX)
# ----------------------------
X = (X - X.mean()) / (X.std() + 1e-8)

# ----------------------------
# 4. Shuffle
# ----------------------------
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# ----------------------------
# 5. Convert to torch
# ----------------------------
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ----------------------------
# 6. Model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EnergyMLP(dim=X.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X = X.to(device)
y = y.to(device)

# ----------------------------
# 7. Training loop
# ----------------------------
epochs = 10
batch_size = 512

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | loss={total_loss:.4f}", flush=True)

# ----------------------------
# 8. Save model
# ----------------------------
torch.save(model.state_dict(), "checkpoints/energy_mlp.pth")
print("Saved: checkpoints/energy_mlp.pth")