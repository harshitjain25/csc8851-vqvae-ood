import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
os.makedirs("checkpoints", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple CNN
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

transform = transforms.ToTensor()
trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training baseline classifier...")

for epoch in range(5):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/baseline_cnn.pth")
print("Saved baseline model")