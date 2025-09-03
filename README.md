
# Waste Image Classification with TensorFlow/Keras

## Overview

This repository provides an end-to-end deep learning pipeline for image classification of waste items. You can use this notebook-based workflow to train a convolutional neural network (CNN) that distinguishes between classes such as cardboard, glass, metal, paper, plastic, and trash. The code is designed for ease of use within Google Colab, supports GPU acceleration, and includes robust data augmentation and model evaluation routines.

***

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Description](#model-description)
- [Training \& Evaluation](#training--evaluation)
- [Results](#results)
- [How to Improve This Project](#how-to-improve-this-project)
- [Acknowledgments](#acknowledgments)
- [Data Augmentation](#data-augmentation)


***

## Dataset

- The dataset must have one subdirectory per class, for example:

```
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

- Each subdirectory should contain images (supported: `.jpg`, `.jpeg`, `.png`).
- Place your dataset in Google Drive at `/MyDrive/dataset/dataset/` for easy access in Colab.

***

## Requirements

- Python 3.8+
- TensorFlow 2.x
- numpy
- matplotlib
- Google Colab (recommended for GPU and Drive integration)

**Colab users:** Required libraries are pre-installed, but you may install extras with:

```python
!pip install tensorflow matplotlib
```


***

## Setup Instructions

1. **Mount Google Drive in Colab**
Use the provided notebook cell to mount your Google Drive.
2. **Copy Dataset to Local Runtime**
The provided script copies your dataset from Drive to Colab’s `/content/` folder for faster access.
3. **Adjust Parameters**
Edit class names, paths, and hyperparameters as needed in the notebook.
4. **Run Cells in Order**
The notebook is sectioned by: environment setup, dataset copy, data pipeline, model definition, training, evaluation, and visualization.

***

## Usage

1. Place your dataset in your Google Drive at `/MyDrive/dataset/dataset/`.
2. Open the provided notebook in Google Colab (or Jupyter with Drive API access).
3. Run through each notebook cell—following cells for setup, data splitting, preprocessing, augmentation, model building, training, and evaluation.

***

## Model Description

- **Type:** Custom CNN (Convolutional Neural Network)
- **Input Size:** 128x128 RGB images
- **Architecture:**
    - Three blocks of Conv2D → ReLU → MaxPooling → Dropout
    - Global Average Pooling
    - Dense layer
    - Softmax output (6 classes)
- **Loss:** Categorical Cross-Entropy (with optional label smoothing)
- **Optimizer:** Adam

***

## Training \& Evaluation

- **Split:** 60% train, 20% validation, 20% test (randomized).
- **Batch Size:** 16 .
- **Epochs:** 100.
- **Metrics:** Training/Validation/Test accuracy and loss.
- **Augmentation:** Image flips, rotations, brightness/contrast jitter, etc.
- **Visualization:** Plots accuracy and loss curves after training.

***

## Data Augmentation
To improve the generalization and accuracy of the model, data augmentation techniques were applied to the training images. Data augmentation artificially creates new, diverse training samples by transforming existing images, which helps prevent overfitting and enables the model to better handle variability in real-world data.
Without augmentation, the model achieved a test accuracy of 0.76. After adding augmentation, accuracy increased to approximately 0.80. 

*** 

## Results

- **Example test accuracy:** ~0.80 
- Example output:
    - Training/validation/test accuracy and loss
    - Loss and accuracy visualization plots after training

***

## How to Improve This Project

- Add more data and/or use more aggressive augmentation techniques.
- Tune learning rate, batch size, and model architecture parameters.
- Switch to a pre-trained model (transfer learning) for better performance with fewer data.
- Add batch normalization layers in the model.
- Apply regularization (higher dropout/L2-regularization).
- Implement early stopping on validation loss to prevent overfitting.

***


***

## Acknowledgments

- Built with TensorFlow and Keras.
- Inspired by classic image classification benchmarks and open-source tutorials.
- Thanks to dataset providers and open-source contributors.

