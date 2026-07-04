# NN Tumor Identification

A convolutional neural network for classifying medical images as tumor or normal. The pipeline covers data loading, training, and evaluation, and is set up to run directly in Google Colab against a Drive-hosted dataset.

## Approach

Images are loaded with `torchvision.datasets.ImageFolder` and split into stratified train/validation sets so the validation split preserves the tumor/normal class ratio. The model itself is a 4-block CNN (conv → batch norm → ReLU → max-pool, channels doubling from 32 to 256) followed by a fully connected classifier with dropout, trained with `BCEWithLogitsLoss` using a `pos_weight` term to counteract class imbalance. An optional `TransferNet` variant swaps in a frozen, ImageNet-pretrained ResNet18 with a new classification head, toggled with a single config flag when the from-scratch CNN needs a stronger feature extractor.

Training uses SGD with momentum, a `ReduceLROnPlateau` scheduler, and early stopping on validation loss, checkpointing the best model to `best_model.pt`. Final evaluation reports a confusion matrix, precision/recall/F1, and ROC-AUC on the held-out test set, plus a training/validation loss curve, an ROC curve, and a sample grid of predictions vs. ground truth.

## Data

The dataset is not included in this repo. `NN_tumor_ID.py` expects Google Drive paths (mounted via `google.colab.drive`) of the form:

```
train_images/
├── normal/
└── tumor/
test_images/
├── normal/
└── tumor/
```

Images are resized to 200x200 and normalized on load; training images are additionally augmented with random horizontal flips and small rotations.

## Model architecture

| Stage | Layers |
|---|---|
| Feature extractor | 4x [Conv2d → BatchNorm2d → ReLU → MaxPool2d], channels 3→32→64→128→256, then AdaptiveAvgPool2d(6x6) |
| Classifier | Dropout → Linear(9216, 512) → ReLU → Dropout → Linear(512, 128) → ReLU → Linear(128, 1) |

Outputs are raw logits (no sigmoid), consumed by `BCEWithLogitsLoss`; probabilities are recovered with `torch.sigmoid` at inference/evaluation time.

## Running

This script is written to run as a Colab notebook cell (it mounts Google Drive directly). To run:

1. Upload `train_images/` and `test_images/` to Google Drive in the structure above.
2. Open `NN_tumor_ID.py` in Colab, update `train_dir`/`test_dir` if needed, and run.
3. Set `USE_TRANSFER_LEARNING = True` in the config block to fine-tune ResNet18 instead of training the CNN from scratch.

## Dependencies

```
numpy
Pillow
matplotlib
scikit-learn
torch
torchvision
```

(`google-colab` is provided automatically in the Colab runtime.)
