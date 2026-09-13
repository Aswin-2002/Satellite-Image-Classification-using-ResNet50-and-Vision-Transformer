# Satellite Image Classification using CNN and Vision Transformer

## Project Overview
This project implements a comparative study between a Convolutional Neural Network (ResNet50) and a Vision Transformer (ViT) for satellite image classification. The models are trained and evaluated on the EuroSAT dataset.

The objective is to classify satellite images into 10 categories and compare the performance of CNN-based and transformer-based architectures.

---
## 📂 Dataset

The dataset used is the EuroSAT RGB dataset.

🔗 Download link:
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

- Use the **EuroSAT (RGB)** folder (NOT EuroSATallBands)
- The dataset contains ~27,000 images across 10 classes

---
##  Requirements

Install the following Python libraries before running the code:
pip install torch torchvision timm matplotlib seaborn scikit-learn

---
##  How to Run the Code

### Step 1: Download Dataset
- Download the dataset from the Kaggle link above
- Extract it
- Use RGB images folder only
- Keep the folder structure as:
EuroSAT/
AnnualCrop/
Forest/
...

---

### Step 2: Update Dataset Path

Upload the the dataset folder in Google Colab
Update the path in code:
```python
root = "PATH_TO_EuroSAT"

### Step 3: Run the Notebook

-Ensure GPU is selected in runtime (Colab)
-Ignore the first 2 cells for mounting (Google Drive) and uploading dataset if you already    uploaded the dataset to colab.
-Run the cells in order.

----------------------------------------------------
### Models Used:
ResNet50 (CNN-based model using transfer learning)
Vision Transformer (ViT) using pretrained weights

Both models were fine-tuned for 10-class classification.

### Output:
The code produces:

-Training and validation loss curves
-Test accuracy for both models
-Confusion matrices
-Classification Report

### Hardware Used:
Google Colab (NVIDIA T4 GPU)

### Reproducibility:
Same dataset split (70/15/15)
Same preprocessing (resize to 224×224)
Same training parameters (batch size = 32, learning rate = 0.0001)

--------------------------------
35055474_ITNPAI1 (2025/2026)
