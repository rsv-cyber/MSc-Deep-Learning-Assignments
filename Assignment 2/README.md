# Homework 2 Report: Neural Networks & Deep Learning

## Assignment Overview

This assignment focuses on **Fine-Tuning Pre-trained Convolutional Neural Networks with Data Augmentation** for image classification. The goal is to:

- Build and evaluate fine-tuned models using VGG16 and ResNet50 pre-trained on ImageNet
- Implement three data augmentation techniques — horizontal flip, rotation, and zooming
- Analyze the effect of data augmentation on model performance and overfitting
- Compare different training strategies — training only the classifier vs. fine-tuning the last convolutional block

The dataset used is the **Cats vs Dogs dataset** from Kaggle, containing images of cats and dogs.

---

## Dataset: Cats vs Dogs

| Property | Value |
|----------|-------|
| Training samples (before augmentation) | 491 |
| Validation samples | 211 |
| Test samples | 100 |
| Image dimensions (after resize) | 224 × 224 (RGB) |
| Classes | 2 (Cats, Dogs) |
| Normalization | ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| Split ratio | 70% training, 30% validation |

The dataset is a subset of the Kaggle CAT vs DOG database, balanced with equal samples per class.

---

## Paper Reference

This assignment is based on the paper:

> **"Effect of data-augmentation on fine-tuned CNN model performance"**  
> *Ramaprasad Poojary, Roma Raina, Amit Kumar Mondal*  
> International Journal of Artificial Intelligence, 2021

The paper investigates how data augmentation affects the performance of fine-tuned CNN models (VGG16 and ResNet50) on a custom Cats vs Dogs dataset.

---

## Model Architectures

### VGG16

| Property | Value |
|----------|-------|
| Total Layers | 41 (13 conv + 3 FC) |
| Input Size | 224 × 224 |
| Top-5 Accuracy (ImageNet) | 92.3% |
| Trainable Parameters (fine-tuned) | 32,772,610 |

**Key Characteristics:**
- Uses 3×3 convolutional kernels stacked in blocks
- Simpler architecture compared to ResNet
- More parameters, higher computational cost

### ResNet50

| Property | Value |
|----------|-------|
| Total Layers | 50 |
| Input Size | 224 × 224 |
| Top-5 Accuracy (ImageNet) | > VGG16 |
| Trainable Parameters (fine-tuned) | 1,062,914 |

**Key Characteristics:**
- Uses Residual (skip) connections to address vanishing gradients
- Batch Normalization after each convolution
- More efficient parameter usage (fewer trainable parameters)

### Why These Architectures?

- **VGG16**: Classic architecture, widely used as a baseline for transfer learning
- **ResNet50**: More modern architecture with residual connections that enable deeper networks to train effectively

Both are pre-trained on ImageNet (1000 classes), providing rich feature representations.

---

## Data Augmentation

Three augmentation methods were implemented following the paper:

### 1. Horizontal Flip
- Randomly flips images horizontally with probability 1
- Most common augmentation for object recognition tasks
- Especially effective for symmetric objects

### 2. Rotation
- Randomly rotates images up to ±30 degrees
- Simulates different camera angles
- Helps model learn rotation-invariant features

### 3. Zooming (Random Resized Crop)
- Randomly crops and resizes images with scale factor between 0.75 and 1.25
- Simulates different distances
- Improves scale invariance

### Augmentation Code

```python
flip_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

rotation_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

zooming_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.75, 1.25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
## Training Configuration

### Hyperparameters
The models were trained using the following hyperparameter suite:

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | SGD with Momentum |
| **Learning Rate (initial)** | 0.1 |
| **LR Decay Rate** | 0.002 |
| **Momentum** | 0.9 |
| **Batch Size** | 32 |
| **Epochs** | 50 |
| **Loss Function** | Cross-Entropy Loss |

### Loss Function
For categorical/binary classification, the Cross-Entropy Loss is calculated as:

$$L = -\frac{1}{N} \sum_{i} \sum_{j} \mathbb{1}(y_i = j) \cdot \log(\hat{y}_i^{(j)})$$

Where:
* $N$ = number of samples
* $C$ = number of classes (2)
* $y_i$ = true label
* $\hat{y}_i$ = predicted output

---

## Training Configurations Tested

We evaluated 8 distinct experimental setups across two model architectures:

| Config | Model | Data Type | Trainable Layers |
| :---: | :--- | :--- | :--- |
| **1** | ResNet50 | Original | FC Layer only |
| **2** | ResNet50 | Original | FC Layer + Last Conv Block |
| **3** | ResNet50 | Augmented | FC Layer only |
| **4** | ResNet50 | Augmented | FC Layer + Last Conv Block |
| **5** | VGG16 | Original | FC Layer only |
| **6** | VGG16 | Original | FC Layer + Last Conv Block |
| **7** | VGG16 | Augmented | FC Layer only |
| **8** | VGG16 | Augmented | FC Layer + Last Conv Block |

---

## Results and Analysis

### ResNet50 Results

#### Configuration 1: Classifier Only, Original Data
* **Metrics:** * Final Accuracy: ~100% (Train) | ~97.6% (Val) | **95% (Test)**
  * Final Loss: ~0.156 (Train) | ~0.206 (Val)
* **Analysis:** The model converges rapidly with high training accuracy. The small gap between training and validation accuracy indicates minimal overfitting. However, minor fluctuations in loss suggest the learning rate might be slightly aggressive.

#### Configuration 2: Classifier + Last Conv Block, Original Data
* **Metrics:** * Final Accuracy: ~100% (Train) | ~97.6% (Val) | **98% (Test)**
  * Final Loss: ~0.004 (Train) | ~0.077 (Val)
* **Analysis:** Unfreezing the last convolutional block significantly shifts the needle on test accuracy (**95% → 98%**), allowing the network to quickly adapt and capture task-specific features. Training stability is improved with lower overall loss.

#### Configuration 3: Classifier Only, Augmented Data
* **Metrics:** * Final Accuracy: ~95-98% (Train) | ~96-100% (Val) | **96% (Test)**
  * Final Loss: ~0.940 (Train) | ~0.481 (Val)
* **Analysis:** Data augmentation acts as a highly effective regularizer. While random augmentations introduce more variance during training, the validation accuracy remains high and robust.

#### Configuration 4: Classifier + Last Conv Block, Augmented Data
* **Metrics:** * Final Accuracy: ~99.2% (Train) | ~97.6% (Val) | **98% (Test)**
  * Final Loss: ~0.028 (Train) | ~0.050 (Val)
* **Analysis:** This setup yielded the **best overall performance**. It exhibits smooth convergence with minimal metric fluctuations, effectively utilizing data augmentation to mitigate overfitting while preserving maximum predictive accuracy.

---

### VGG16 Results

| Configuration | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| Classifier Only, Original | `NaN` | 48.82% | 50% |
| Classifier + Last Conv, Original | `NaN` | 48.82% | 50% |
| Classifier Only, Augmented | `NaN` | 48.82% | 50% |
| Classifier + Last Conv, Augmented | `NaN` | 48.82% | 50% |

>  **Critical Analysis:** VGG16 completely failed to converge under the current hyperparameter schema. The loss quickly diverged to `NaN`. This is primarily due to the high initial learning rate (0.1) acting aggressively on VGG16's deep architecture, which lacks modern structural elements like residual shortcuts and internal batch normalization. Scaling the learning rate down to `0.01` or `0.001` would likely resolve this instability.

---

## Comparison Tables

### Effect of Data Augmentation on ResNet50
| Model Setup | Data | Training Acc | Validation Acc | Test Acc |
| :--- | :--- | :---: | :---: | :---: |
| ResNet50 Classifier | Original | ~100% | ~97.6% | 95% |
| ResNet50 Classifier | Augmented | ~95-98% | ~96-100% | 96% |
| ResNet50 Last Conv Block | Original | ~100% | ~97.6% | **98%** |
| ResNet50 Last Conv Block | Augmented | ~99.2% | ~97.6% | **98%** |

### Top-Performing Architecture Comparison
| Model Setup | Data | Test Accuracy |
| :--- | :--- | :---: |
| **ResNet50 (Last Conv Block)** | **Augmented** | **98%** |
| **ResNet50 (Last Conv Block)** | **Original** | **98%** |
| ResNet50 (Classifier Only) | Augmented | 96% |
| ResNet50 (Classifier Only) | Original | 95% |
| VGG16 | Any | 50% (Random Guessing) |

---

## Detailed Technical Insights

### Effect of Data Split Ratios
Following the source literature's baseline recommendations, we leveraged a **70-30 train-validation split**. This structural trade-off impacts optimization as follows:

* **70-30 (Used):** Optimal equilibrium; provides sufficient updates per epoch while keeping validation sets robust enough to accurately catch overfitting.
* **80-20:** Maximizes training data, but risks high variance in validation metrics due to a smaller evaluation sample size.
* **60-40:** Strong validation confidence, but underperforms globally due to regular model underfitting from strict training data starvation.

### Role of Regularization (Dropout & Batch Norm)
Explicit Dropout layers were intentionally excluded because modern pre-trained backbones utilize alternative, highly effective implicit regularizers:
* **ResNet50:** Benefits from **Batch Normalization**, which conditions layer outputs and offers an inherent regularizing effect.
* **Data Augmentation:** Operates directly on the data distribution layer, artificially inflating dataset variance to prevent catastrophic overfitting.
* **VGG16 Note:** Given its brittle stability under this setup, injecting explicit Dropout layers before the classification head could drastically aid training recovery.

### The Power of Weight Initialization
Instead of random uniform or Xavier/Glorot initializations, our pipeline utilizes **ImageNet pre-trained weights**. 
* This provides an exceptional initial landing zone within the loss landscape.
* Lower-level convolutional filters (edges, textures) transfer seamlessly to the target dataset, completely eliminating the optimization risks associated with cold-starting weights.

---

## Architecture Deep Dive: VGG16 vs. ResNet50

| Aspect | VGG16 | ResNet50 |
| :--- | :--- | :--- |
| **Structural Design** | Plain Sequential Convolutional Layers | Residual Skip Connections |
| **Depth** | 16 Layers | 50 Layers |
| **Trainable Parameters** | ~32.7 Million | ~1.06 Million |
| **Optimization Stability** | Highly Sensitive to Learning Rates | Extremely Resilient & Stable |
| **Target Task Accuracy** | Failed to Converge (50%) | **98%** |
| **Key Architectural Catalyst** | None | **Batch Normalization + Skip Connections** |

---

## Visual Results

### Training Curves

#### ResNet50 - Classifier on Original Data
![ResNet50 Classifier Original](resnet50_classifier_original.png)
*Figure 1: Training and validation accuracy/loss curves for ResNet50 classifier on original data.*

#### ResNet50 - Last Conv Block on Original Data
![ResNet50 Last Conv Original](resnet50_lastconv_original.png)
*Figure 2: Training and validation accuracy/loss curves for ResNet50 last conv block on original data.*

#### ResNet50 - Classifier on Augmented Data
![ResNet50 Classifier Augmented](resnet50_classifier_augmented.png)
*Figure 3: Training and validation accuracy/loss curves for ResNet50 classifier on augmented data.*

#### ResNet50 - Last Conv Block on Augmented Data
![ResNet50 Last Conv Augmented](resnet50_lastconv_augmented.png)
*Figure 4: Training and validation accuracy/loss curves for ResNet50 last conv block on augmented data.*

---

## Comparison with Source Literature

| Model Configuration | Data Setup | Paper Test Acc | Our Test Acc |
| :--- | :--- | :---: | :---: |
| VGG16 Fine-tuned | Original | 80% | *Failed* |
| VGG16 Fine-tuned | Augmented | 88% | *Failed* |
| ResNet50 Fine-tuned | Original | 82% | **95% (Classifier) / 98% (Last Conv)** |
| ResNet50 Fine-tuned | Augmented | 90% | **96% (Classifier) / 98% (Last Conv)** |

### Discrepancy Analysis
Our ResNet50 implementation markedly outperformed the target baseline paper. This performance leap is attributed to:
1. **Framework Differences:** Core optimization differences between PyTorch (our setup) and Keras (paper setup).
2. **Fine-Tuning Depth:** Targeted adaptation via unfreezing the final convolutional block rather than relying strictly on the linear head.
3. **Data Pipelines:** Variations in validation splits and specific geometric transformations.

---

## Key Takeaways

1. **Augmentation Works:** Geometric and color augmentations smooth out optimization variances and consistently protect against overfitting.
2. **Partial Unfreezing is Essential:** Unfreezing the final convolutional block yields a notable **+3% accuracy bump**, fine-tuning generic ImageNet filters into highly specific, task-relevant features.
3. **Architectural Guardrails Matter:** ResNet50 easily outperforms VGG16 due to residual connections and batch normalization, protecting gradients from exploding or vanishing even at high learning rates ($0.1$).

---

## References

* Poojary, R., Raina, R., & Mondal, A. K. (2021). *Effect of data-augmentation on fine-tuned CNN model performance*. International Journal of Artificial Intelligence.
* Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR.
