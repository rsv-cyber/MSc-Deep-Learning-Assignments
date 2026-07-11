# Homework 2 - Question 2
## Effect of Data Augmentation on Fine-Tuned CNN Models

---

## Abstract

In this section, we investigate the effect of data augmentation on the performance of pre-trained convolutional neural networks using the Fine-Tuning method. Two well-known architectures, VGG16 and ResNet50, are evaluated on the Cats vs Dogs dataset using various data augmentation techniques including rotation, horizontal flipping, and zooming.

---

## 1. Paper Introduction

The paper **"Effect of data-augmentation on fine-tuned CNN model performance"** by Ramaprasad Poojary, Roma Raina, and Amit Kumar Mondal examines the impact of data augmentation on pre-trained CNN models using Fine-Tuning.

### Key Objectives:
- Building CNN models using Fine-Tuning on VGG16 and ResNet50
- Using data augmentation to prevent overfitting
- Improving model performance by increasing training data diversity

### Dataset:
- 700 images of cats and dogs from Kaggle CAT vs DOG database
- Split ratio: 70% training, 30% validation

---

## 2. Data Preprocessing and Augmentation

### 2.1. Data Loading

The dataset contains cat and dog images in Train and Test folders:

| Dataset | Number of Samples |
|---------|-------------------|
| Training (before augmentation) | 491 |
| Validation | 211 |
| Test | 100 |

### 2.2. Augmentation Methods

Three augmentation methods were implemented following the paper:

1. **Horizontal Flip**: Random horizontal flipping of images
2. **Rotation**: Random rotation up to 30 degrees
3. **Zooming**: Random scaling between 0.75 and 1.25

### 2.3. Augmented Dataset

For each training image, three augmented versions were created (flip, rotation, zoom), plus the original:

| Dataset | Number of Samples |
|---------|-------------------|
| Training (after augmentation) | 491 × 4 = 1964 |
| Validation | 211 |
| Test | 100 |

### 2.4. Image Preprocessing

All images were preprocessed using standard ImageNet normalization:

```python
preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
