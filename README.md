# Satellite Image Classification: ResNet50 vs Vision Transformer

## Project Overview

This project presents a comparative study of two deep learning architectures for satellite image classification: **ResNet50**, a Convolutional Neural Network (CNN), and a **Vision Transformer (ViT)**.

Both models were fine-tuned using pretrained weights on the **EuroSAT RGB dataset** to classify satellite images into 10 land-use and land-cover categories. Their performance was evaluated and compared using test accuracy, confusion matrices, classification reports, and training/validation loss.

## Objectives

The main objectives of this project were to:

- Develop an image classification pipeline for satellite imagery.
- Implement a CNN-based approach using ResNet50.
- Implement a Transformer-based approach using Vision Transformer (ViT).
- Fine-tune pretrained models for 10-class satellite image classification.
- Evaluate both models using test accuracy, confusion matrices, and classification reports.
- Compare the performance and training behaviour of the two architectures.

---

## Dataset

The project uses the **EuroSAT RGB dataset**, containing approximately 27,000 satellite images across 10 land-use and land-cover classes.

The RGB version of EuroSAT was used rather than the multispectral **EuroSATallBands** version.

### Classes

The dataset contains the following 10 classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

### Dataset Source

[EuroSAT RGB Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

> Download the RGB dataset and keep the original class-folder structure required by `ImageFolder`.

---

## Methodology

The project follows this general pipeline:

```text
EuroSAT RGB Images
        ↓
Image Preprocessing
        ↓
70% Training / 15% Validation / 15% Testing
        ↓
 ┌───────────────────────┐
 │                       │
ResNet50                ViT
 │                       │
 └───────────┬───────────┘
             ↓
       Model Evaluation
             ↓
     Performance Comparison
```

### Image Preprocessing

Images were resized to **224 × 224 pixels** to match the input requirements of the pretrained models and were converted into PyTorch tensors.

No additional data augmentation or image normalization was applied in the final implementation.

### Dataset Split

The dataset was divided into:

| Dataset | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The split was generated using PyTorch's `random_split`. Because no explicit random seed was specified, the exact samples assigned to each subset may differ between separate runs.

---

## Model 1: ResNet50

**ResNet50** is a 50-layer Convolutional Neural Network based on residual learning and skip connections.

A pretrained ResNet50 model was adapted for the EuroSAT classification task by replacing its original classification layer with a layer producing **10 output classes**.

### Configuration

- Architecture: ResNet50
- Initialisation: Pretrained weights
- Number of classes: 10
- Optimiser: Adam
- Learning rate: 0.0001
- Loss function: Cross Entropy Loss
- Epochs: 10
- Batch size: 32
- Input size: 224 × 224

---

## Model 2: Vision Transformer (ViT)

The second approach uses a **Vision Transformer (ViT)**, a Transformer-based architecture that represents an image as a sequence of patches and uses self-attention to model relationships between them.

The `vit_base_patch16_224` model from the `timm` library was initialised with pretrained weights and configured for the 10 EuroSAT classes.

### Configuration

- Architecture: ViT Base Patch16 224
- Initialisation: Pretrained weights
- Number of classes: 10
- Optimiser: Adam
- Learning rate: 0.0001
- Loss function: Cross Entropy Loss
- Epochs: 10
- Batch size: 32
- Input size: 224 × 224

---

## Transfer Learning and Fine-Tuning

Both models were initialised with pretrained weights rather than trained entirely from random initialisation.

The pretrained models were then **fine-tuned** for the EuroSAT satellite image classification task. Model parameters were optimised during training using the Adam optimiser.

---

## Training

Both architectures were trained using the same main training configuration to provide a consistent basis for comparison.

The training process involved:

1. Forward propagation through the model.
2. Calculation of cross-entropy loss.
3. Backpropagation to calculate gradients.
4. Parameter updates using Adam.
5. Evaluation on the validation set after each epoch.

Training and validation losses were recorded throughout the training process.

### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 0.0001 |
| Optimiser | Adam |
| Loss function | CrossEntropyLoss |
| Epochs | 10 |
| Input resolution | 224 × 224 |
| Training split | 70% |
| Validation split | 15% |
| Test split | 15% |

---

## Evaluation

The models were evaluated on the held-out test set using:

### Test Accuracy

Measures the proportion of test images that were correctly classified.

### Confusion Matrix

Shows correct and incorrect predictions for each of the 10 classes and helps identify which categories are most frequently confused.

### Classification Report

Provides the following metrics for each class:

- Precision
- Recall
- F1-score
- Support

### Training and Validation Loss

Training and validation loss curves were used to examine model learning behaviour across the 10 training epochs.

---

## Results

The notebook generates the following outputs for comparing ResNet50 and ViT:

- Test accuracy
- Training loss curves
- Validation loss curves
- Confusion matrices
- Classification reports
- Model performance comparison

The numerical results and plots are available in the Jupyter notebook.

---

## ResNet50 vs Vision Transformer

The project compares two fundamentally different approaches to visual recognition.

| Aspect | ResNet50 | Vision Transformer |
|---|---|---|
| Architecture | CNN | Transformer |
| Main mechanism | Convolution | Self-attention |
| Image representation | Feature maps | Image patches |
| Initialisation | Pretrained weights | Pretrained weights |
| Input size | 224 × 224 | 224 × 224 |
| Classification task | 10 EuroSAT classes | 10 EuroSAT classes |

This comparison provides an opportunity to evaluate a conventional CNN architecture against a Transformer-based vision architecture on the same satellite image classification problem.

---

## Limitations

The following limitations should be considered when interpreting the results:

- Only the RGB version of EuroSAT was used rather than the multispectral bands.
- Images were resized to 224 × 224 pixels.
- No data augmentation was used in the final preprocessing pipeline.
- Both models were trained for 10 epochs.
- The same main training configuration was used for both architectures.
- The dataset split was not controlled with an explicit random seed, so exact train/validation/test samples may vary between runs.
- The experiment was conducted on a single dataset, so the results may not generalise to other satellite image datasets.

---

## Future Improvements

Potential improvements include:

- Apply data augmentation such as rotations, flips, and crops.
- Experiment with different learning rates and optimisers.
- Use learning-rate scheduling.
- Compare additional CNN architectures.
- Compare additional Vision Transformer variants.
- Investigate the multispectral EuroSAT data.
- Increase the number of training epochs.
- Use cross-validation for more robust evaluation.
- Compare computational requirements such as training time and GPU memory.
- Investigate explainability methods such as Grad-CAM for CNNs and attention visualisation for ViT.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- timm
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab

---

## Repository Contents

| File | Description |
|---|---|
| `3504474_ITNPAI1.ipynb` | Complete implementation, training, evaluation, and comparison |
| `README.md` | Project documentation |

---

## Requirements

Install the required Python libraries using:

```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn
```

---

## How to Run

### 1. Download the Dataset

Download the EuroSAT RGB dataset from Kaggle and extract it locally or upload it to Google Colab.

### 2. Set the Dataset Path

Update the dataset path in the notebook:

```python
root = "PATH_TO_EuroSAT"
```

### 3. Run the Notebook

Open `3504474_ITNPAI1.ipynb` in Jupyter Notebook/JupyterLab or Google Colab and run the cells in order.

For the original experiment, GPU acceleration was used through Google Colab with an NVIDIA T4 GPU.

---

## Reproducibility

The experiment configuration is documented above, including the dataset proportions, input resolution, batch size, learning rate, optimiser, loss function, and number of epochs.

However, because the dataset was split using `random_split` without a fixed random seed, the exact train/validation/test split is **not guaranteed to be identical between runs**. Consequently, the exact numerical results may vary slightly when the notebook is executed again.

---

## Author

**Aswin**

MSc Artificial Intelligence  
University of Stirling

