# Satellite Change Detection using Deep Learning

## Project Overview

This project focuses on detecting changes in satellite imagery using deep learning techniques. By comparing paired “before” and “after” remote sensing images, the system identifies regions where significant changes have occurred. The project demonstrates a complete machine learning workflow, including dataset handling, preprocessing, model training, and visualization.

It is designed as an introductory computer vision project for remote sensing applications and showcases practical implementation using Python and PyTorch.

---

## Objectives

* Load and preprocess paired satellite image datasets
* Train a convolutional neural network for change detection
* Visualize before and after images with ground truth masks
* Build a structured and reproducible ML pipeline

---

## Dataset Structure

The dataset is organized into training, validation, and testing folders. Each subset contains paired images and corresponding ground truth labels.


* **A**: Image captured at time T1
* **B**: Image captured at time T2
* **label**: Binary mask representing changed regions

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/satellite-change-detection.git
cd satellite-change-detection
```

Install required packages:

```
pip install torch torchvision opencv-python matplotlib numpy
```

---

## Usage

Train the model:

```
python train.py
```

Run visualization or evaluation:

```
python predict.py
```

The output displays the before image, after image, and corresponding ground truth mask for comparison.


## Results

The trained model learns to identify structural and environmental changes between two time periods. The system provides visual comparison between input images and labeled change regions, demonstrating the effectiveness of deep learning in satellite image analysis.


This project is created for academic and educational purposes.
