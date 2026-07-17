# Intel Image Classification with a CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch to classify natural scene images into 6 categories: buildings, forest, glacier, mountain, sea, and street.

## Problem

Given a photo of a natural or urban scene, predict which of 6 categories it belongs to. This is a multi-class image classification task using the [Intel Image Classification dataset](https://www.kaggle.com/puneet6060/intel-image-classification) (~14,000 training images, ~3,000 test images).

## Approach

**Architecture:** A CNN with 2 convolutional layers followed by 3 fully connected layers:
- Conv2d(3→6, kernel=5) → ReLU → MaxPool
- Conv2d(6→16, kernel=5) → ReLU → MaxPool
- Flatten → FC(2704→1000) → ReLU → FC(1000→100) → ReLU → FC(100→6)

**Preprocessing:** Images resized to 64×64, converted to tensors, and normalized.

**Training:** CrossEntropyLoss, SGD optimizer (lr=0.001, momentum=0.9), 20 epochs, batch size 32.

## Key finding: diagnosing and fixing overfitting

An initial run without data augmentation showed a revealing pattern:

| Epochs | Training Loss | Test Accuracy |
|--------|---------------|----------------|
| 10 | 0.70 | 73.10% |
| 20 | 0.36 | 73.93% |

Training loss kept falling steadily, but test accuracy barely moved — a classic sign of overfitting: the model was memorizing training images rather than learning generalizable features.

**Fix:** Added data augmentation (random horizontal flips and rotations) to the training pipeline only, leaving the test set unchanged.

**Result after augmentation:**

| Epochs | Training Loss | Test Accuracy |
|--------|---------------|----------------|
| 20 | 0.57 | **78.53%** |

Training loss was *higher* than before (as expected — augmentation makes training intentionally harder), but test accuracy improved by nearly 5 points, confirming the model was generalizing better rather than memorizing.

## Results

**Final test accuracy: 78.53%**

### Training loss curve
![Loss curve](loss_curve_augmented.png)

### Confusion matrix
![Confusion matrix](confusion_matrix_augmented.png)

The confusion matrix reveals which classes the model confuses most:
- **Glacier ↔ Mountain** is the largest source of error (51 glaciers predicted as mountain, 64 mountains predicted as glacier) — visually, snowy peaks and glaciers share many features.
- **Buildings ↔ Street** is the second largest confusion (47 buildings predicted as street, 62 streets predicted as buildings) — street-level photos often contain buildings in frame.
- **Forest** was the most reliably classified category, with the fewest misclassifications in either direction, likely due to its visually distinct texture.

## What I'd try next

- **Transfer learning:** fine-tune a pretrained ResNet on this dataset and compare against the from-scratch CNN — likely to significantly boost accuracy given the small dataset size.
- **Deeper architecture / batch normalization:** could improve feature extraction without needing a pretrained model.
- **Track test accuracy at multiple checkpoints during training** (not just the final epoch) to pinpoint exactly when overfitting begins.

## Tech stack

Python, PyTorch, torchvision, scikit-learn (confusion matrix), matplotlib & seaborn (visualization)

## How to run

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
python3 model.py
```

Dataset should be placed in `seg_train/seg_train/` and `seg_test/seg_test/`, with each subfolder representing a class, as provided by the Kaggle download.
