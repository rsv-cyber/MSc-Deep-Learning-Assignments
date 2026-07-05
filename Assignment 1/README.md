# Homework 1 Report: Neural Networks & Deep Learning

## Assignment Overview 

This assignment focuses on **Adversarial Attacks on Neural Networks** using the MNIST dataset. The goal is to:

1. **Build and train a Multi-Layer Perceptron (MLP)** for handwritten digit classification on MNIST
2. **Implement and analyze two adversarial attack methods**:
   - **FGSM (Fast Gradient Sign Method)** — a single-step white-box attack
   - **PGD (Projected Gradient Descent)** — an iterative white-box attack
3. **Evaluate model robustness** by measuring accuracy drop and visualizing misclassified examples

---

## Dataset: MNIST

| Property | Value |
|----------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image dimensions | 28 × 28 (grayscale) |
| Classes | 0–9 (digits) |
| Normalization | Min-max scaling to [0, 1] |

The dataset is well-balanced, with approximately equal samples per digit in both training and test sets.

---

## Model Architecture

The implemented MLP has the following structure:

| Layer | Input → Output | Activation |
|-------|---------------|------------|
| FC1 | 784 → 512 | ReLU |
| FC2 | 512 → 128 | ReLU |
| FC3 | 128 → 32 | ReLU |
| FC4 | 32 → 10 | Softmax |

**Why 784 inputs?**  
MNIST images are 28×28 = 784 pixels, flattened into a 1D vector for the fully connected network.

**Why Softmax at the output?**  
Softmax converts logits into probability distribution over 10 classes, enabling probabilistic interpretation of predictions.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 6×10⁻⁵ |
| Epochs | 25 |
| Batch Size | 64 |
| Loss Function | CrossEntropyLoss |
| Device | GPU (CUDA) |

### Training Results

| Metric | Value |
|--------|-------|
| Final Training Loss | ~2.17 |
| Final Training Accuracy | ~83.5% |
| **Test Accuracy (Clean)** | **84.93%** |

The model shows steady improvement over 25 epochs, though accuracy could be further improved with hyperparameter tuning or deeper architectures.

---

## Adversarial Attacks

### FGSM (Fast Gradient Sign Method)
**Concept:**  
FGSM generates adversarial examples by adding a small perturbation in the direction of the gradient of the loss with respect to the input:

**Concept:**  
FGSM generates adversarial examples by adding a small perturbation in the direction of the gradient of the loss with respect to the input:
x_adv = x + ε · sign(∇x L(f(x), y))

| Parameter | Value |
|-----------|-------|
| ε (Epsilon) | 0.1 |
| Attack Type | White-box, single-step |

### PGD (Projected Gradient Descent)

**Concept:**  
PGD extends FGSM by applying the attack iteratively with a small step size, projecting the perturbation back into the ε-ball after each step:
x_adv⁽ᵗ⁺¹⁾ = Π ( x_adv⁽ᵗ⁾ + α · sign(∇x L(f(x_adv⁽ᵗ⁾), y)) )

| Parameter | Value |
|-----------|-------|
| ε (Epsilon) | 0.1 |
| α (Step size) | 0.01 |
| Iterations | 10 |
| Projection | Clamp to [x−ε, x+ε] and [0,1] |

**Advantages of PGD over FGSM:**
- **Stronger attack** — multiple iterations allow more precise perturbation
- **Better optimization** — step-by-step adjustment finds more effective adversarial directions
- **More realistic threat model** — stronger adversary, better for evaluating robustness

---

## Attack Results Comparison

| Metric | Clean | FGSM | PGD |
|--------|-------|------|-----|
| **Accuracy** | 84.93% | ~23.8% | ~12.9% |
| **Attack Success Rate** | — | ~61.1% | ~72.0% |

### Key Observations

- Both attacks severely degrade model performance
- **PGD is significantly more effective** than FGSM, reducing accuracy by ~72% compared to ~61% for FGSM
- The iterative nature of PGD produces stronger perturbations within the same ε constraint

---

## Visual Results

For each attack, at least one misclassified example per digit class is displayed:

- **Top row:** Original images with true labels
- **Bottom row:** Adversarial images with predicted labels (misclassified)

The perturbations are imperceptible to the human eye but cause the model to make incorrect predictions with high confidence.

---

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.
3. MNIST Dataset: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*.

---

## Conclusions

1. The trained MLP achieves **84.93% accuracy** on clean MNIST test data.
2. Both **FGSM and PGD** successfully fool the model, with **PGD being more powerful**.
3. Adversarial perturbations are **visually imperceptible** but cause significant misclassification.
4. These results highlight the **vulnerability of deep learning models** to adversarial attacks and the need for robust training techniques.

---


