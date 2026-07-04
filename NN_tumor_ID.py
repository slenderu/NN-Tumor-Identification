# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 12:41:13 2025

@author: Tony Huang
"""

import numpy as np
import os
import random
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_score, recall_score, f1_score)

##########################
#reproducibility + device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

##########################
#config
IMG_SIZE = 200
BATCH_SIZE = 16
NUM_WORKERS = 2
VAL_FRACTION = 0.15
NUM_EPOCHS = 60
PATIENCE = 8 # epochs of no val improvement before early stopping
LR = 1e-4
WEIGHT_DECAY = 1e-4
USE_TRANSFER_LEARNING = False  # flip to True to fine-tune a pretrained ResNet18 instead

#path to image dirs
train_dir = '/content/drive/My Drive/train_images/'
test_dir = '/content/drive/My Drive/test_images/'

#augmentation on train only
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

##########################
#datasets use stratified train/val split so val isn't accidentally imbalanced
base_dataset = datasets.ImageFolder(root=train_dir)
targets = np.array(base_dataset.targets)
classes = tuple(base_dataset.classes)

train_idx, val_idx = train_test_split(
    np.arange(len(base_dataset)),
    test_size=VAL_FRACTION,
    stratify=targets,
    random_state=SEED
)

train_dataset_full = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset_full = datasets.ImageFolder(root=train_dir, transform=eval_transform)

trainset = Subset(train_dataset_full, train_idx)
valset = Subset(val_dataset_full, val_idx)
testset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def get_class_counts(subset):
    """Count samples per class inside a Subset (indices into an ImageFolder)."""
    sub_targets = np.array(subset.dataset.targets)[subset.indices]
    return np.bincount(sub_targets)


class NeuralNet(nn.Module):
    def __init__(self, input_channels=3):
        super(NeuralNet, self).__init__()

        # Convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((6, 6))  # pins flatten size regardless of input resolution
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # no Sigmoid here -- BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TransferNet(nn.Module):
    """ResNet18 pretrained on ImageNet, fine-tuned for tumor/normal classification.
    Swaps in for NeuralNet when USE_TRANSFER_LEARNING = True. Worth trying if the
    from-scratch CNN struggles -- the pretrained filters already encode general-purpose
    edge/texture detectors, which matters a lot when the dataset is small."""
    def __init__(self, freeze_backbone=True):
        super(TransferNet, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels_dev = labels.float().unsqueeze(1).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels_dev)
        running_loss += loss.item()

        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_labels), np.array(all_probs)


##########################
#model/loss/optimizer

if USE_TRANSFER_LEARNING:
    net = TransferNet(freeze_backbone=True).to(device)
else:
    net = NeuralNet().to(device)

print(net)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Trainable parameters: {num_params:,}')

# 0 normal 1 tumor (order confirmed via classes = base_dataset.classes above)
class_counts = get_class_counts(trainset)
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                       lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

##########################
#training loop + best-checkpoint saving

best_val_loss = float('inf')
epochs_no_improve = 0
train_loss_history = []
val_loss_history = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(net, trainloader, criterion, optimizer)
    val_loss, _, _ = evaluate(net, valloader, criterion)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    scheduler.step(val_loss)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(net.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

plt.figure(1)
plt.title("Training vs Validation Loss")
plt.plot(train_loss_history, label='train')
plt.plot(val_loss_history, label='val')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

print('Train Finished')

#reload the best checkpoint before final evaluation
net.load_state_dict(torch.load('best_model.pt'))

##########################
#test evalutaion
test_loss, test_labels, test_probs = evaluate(net, testloader, criterion)
test_preds = (test_probs >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
false_positive = fp / (fp + tn) * 100
false_negative = fn / (fn + tp) * 100
precision = precision_score(test_labels, test_preds, zero_division=0)
recall = recall_score(test_labels, test_preds, zero_division=0)
f1 = f1_score(test_labels, test_preds, zero_division=0)

fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

print('Test Finished')
print(f'Test loss: {test_loss:.4f}')
print(f'{accuracy:.2f} {false_positive:.2f} {false_negative:.2f}')
print(f'Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  ROC-AUC: {roc_auc:.3f}')

plt.figure(2)
plt.title("ROC Curve")
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

##########################
#sanity check
import torchvision


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

net.eval()
with torch.no_grad():
    outputs = net(images.to(device))
    predicted = (torch.sigmoid(outputs) >= 0.5).int().squeeze().cpu()

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(labels))))
