import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# same model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = Net().to(device)
model.load_state_dict(torch.load("checkpoints/baseline_cnn.pth"))
model.eval()

transform = transforms.ToTensor()

cifar10 = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
cifar100 = datasets.CIFAR100("./data", train=False, download=True, transform=transform)
svhn = datasets.SVHN("./data", split="test", download=True, transform=transform)

def get_energy(loader):
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)

            # ENERGY FORMULA
            energy = -torch.logsumexp(logits, dim=1)

            scores.extend(energy.cpu().numpy())
    return np.array(scores)

loader_id = torch.utils.data.DataLoader(cifar10, batch_size=128)
loader_ood1 = torch.utils.data.DataLoader(cifar100, batch_size=128)
loader_ood2 = torch.utils.data.DataLoader(svhn, batch_size=128)

print("Computing energy scores...")

id_scores = get_energy(loader_id)
ood1_scores = get_energy(loader_ood1)
ood2_scores = get_energy(loader_ood2)

def compute_auroc(id_s, ood_s):
    y = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    scores = np.concatenate([id_s, ood_s])
    return roc_auc_score(y, scores)

print("\n=== BASELINE RESULTS ===")
print("CIFAR-10 vs CIFAR-100:", compute_auroc(id_scores, ood1_scores))
print("CIFAR-10 vs SVHN:", compute_auroc(id_scores, ood2_scores))