import os
import cv2
import matplotlib.pyplot as plt

root = "dataset/test"

files = os.listdir(os.path.join(root, "A"))
name = files[0]

imgA = cv2.imread(os.path.join(root, "A", name))
imgB = cv2.imread(os.path.join(root, "B", name))
label = cv2.imread(os.path.join(root, "label", name), 0)

imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

imgA = cv2.resize(imgA, (256,256))
imgB = cv2.resize(imgB, (256,256))
label = cv2.resize(label, (256,256))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgA)
plt.title("Before Image")

plt.subplot(1,3,2)
plt.imshow(imgB)
plt.title("After Image")

plt.subplot(1,3,3)
plt.imshow(label, cmap="gray")
plt.title("Ground Truth Change Map")

plt.show()
