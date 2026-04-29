"""
Run this ONCE before the demo to generate the confusion matrix.

Usage:
    python generate_confusion_matrix.py

It assumes:
    - best_model.pth is in the current directory
    - The PlantVillage/ dataset folder is in the current directory (same layout as training)

It produces:
    - confusion_matrix.npy   (raw matrix, loaded by the app)
    - confusion_matrix.png   (pre-rendered figure, used as fallback)
    - class_names.json       (ordered class list)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# ---------- Re-declare everything the model needs ----------

class PlantVillage(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = Path(root)
        self.transforms = transforms
        self.classes = sorted(c.name for c in self.root.iterdir() if c.is_dir())
        self.IDXclasses = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            class_path = self.root / cls
            for f in sorted(class_path.iterdir()):
                if f.suffix.lower() == ".jpg":
                    self.images.append((f, self.IDXclasses[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18Scratch(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._makeLayer(64, 2, 1)
        self.layer2 = self._makeLayer(128, 2, 2)
        self.layer3 = self._makeLayer(256, 2, 2)
        self.layer4 = self._makeLayer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _makeLayer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


# ---------- Recreate the test split with the SAME seed ----------

def main():
    print("Building dataset...")
    full_data = PlantVillage("PlantVillage")
    n = len(full_data)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    _, _, test_subset = random_split(
        full_data,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(123),
    )

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_data = TransformSubset(test_subset, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=128)

    classes = full_data.classes
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ResNet18Scratch(num_classes=n_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print("Running inference on test set...")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Overall accuracy: {100*correct/total:.2f}% ({correct}/{total})")

    # Save raw artifacts
    np.save("confusion_matrix.npy", cm)
    with open("class_names.json", "w") as f:
        json.dump(classes, f, indent=2)

    # Pre-render a fallback PNG in case Streamlit's matplotlib has issues
    fig, ax = plt.subplots(figsize=(11, 9))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    short_names = [c.replace("_", " ").replace("  ", " ") for c in classes]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")
    # Annotate cells with raw counts
    for i in range(n_classes):
        for j in range(n_classes):
            v = cm[i, j]
            if v > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Wrote: confusion_matrix.npy, confusion_matrix.png, class_names.json")


if __name__ == "__main__":
    main()
