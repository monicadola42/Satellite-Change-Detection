import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

# ======================
# Model (same as training)
# ======================

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ======================
# Load Model
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleUNet().to(device)

# ⚠️ Make sure model.pth exists in same folder
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.ToTensor()

# ======================
# Load One Test Image
# ======================

root = "dataset/test"

files = os.listdir(os.path.join(root, "A"))

if len(files) == 0:
    print("❌ No test images found!")
    exit()

name = files[0]
print("Testing on image:", name)

imgA = cv2.imread(os.path.join(root, "A", name))
imgB = cv2.imread(os.path.join(root, "B", name))
label = cv2.imread(os.path.join(root, "label", name), 0)

imgA_rgb = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB_rgb = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

imgA_res = cv2.resize(imgA_rgb, (256, 256))
imgB_res = cv2.resize(imgB_rgb, (256, 256))
label_res = cv2.resize(label, (256, 256))

imgA_tensor = transform(imgA_res)
imgB_tensor = transform(imgB_res)

x = torch.cat([imgA_tensor, imgB_tensor], dim=0).unsqueeze(0).to(device)

# ======================
# Predict
# ======================

with torch.no_grad():
    pred = model(x)
    pred = pred.squeeze().cpu().numpy()

# ✅ Print prediction range
print("Min prediction value:", pred.min())
print("Max prediction value:", pred.max())

# ✅ LOWER threshold (important fix)
threshold = 0.15
pred_binary = (pred > threshold).astype(np.uint8)

# ======================
# Show Raw Heatmap
# ======================

plt.figure(figsize=(5,4))
plt.imshow(pred, cmap='gray')
plt.title("Raw Prediction Heatmap")
plt.colorbar()
plt.show()

# ======================
# Display Results
# ======================

plt.figure(figsize=(14,4))

plt.subplot(1,4,1)
plt.imshow(imgA_res)
plt.title("Before")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(imgB_res)
plt.title("After")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(label_res, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(pred_binary, cmap="gray")
plt.title(f"Predicted (threshold={threshold})")
plt.axis("off")

plt.show()