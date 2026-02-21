import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ChangeDataset(Dataset):
    def __init__(self, root, limit=50):
        self.root = root
        self.files = os.listdir(os.path.join(root, "A"))[:limit]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        imgA = cv2.imread(os.path.join(self.root, "A", name))
        imgB = cv2.imread(os.path.join(self.root, "B", name))
        label = cv2.imread(os.path.join(self.root, "label", name), 0)

        imgA = cv2.resize(imgA, (128, 128))
        imgB = cv2.resize(imgB, (128, 128))
        label = cv2.resize(label, (128, 128))

        imgA = self.transform(imgA)
        imgB = self.transform(imgB)

        label = torch.tensor(label / 255.0).unsqueeze(0).float()

        x = torch.cat([imgA, imgB], dim=0)

        return x, label

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")

dataset = ChangeDataset("dataset/train", limit=50)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleUNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")

for epoch in range(1):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch complete | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")

print("Training finished and model saved!")
