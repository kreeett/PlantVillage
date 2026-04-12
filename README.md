# PlantVillage Disease Classification

A deep learning project for classifying plant diseases using the PlantVillage dataset. This repository implements a custom ResNet-18 architecture trained from scratch to identify various diseases affecting tomatoes, potatoes, and peppers.

## 🌱 Project Overview

This project addresses the critical agricultural challenge of plant disease identification by leveraging computer vision and deep learning. The model can classify 15 different categories of plant health conditions, helping farmers and agricultural professionals quickly diagnose and treat plant diseases.

## 📊 Dataset

The project uses the **PlantVillage Dataset**, which contains images of healthy and diseased plant leaves across three plant species:

### Classes (15 total)

**Pepper (Bell)**
- Bacterial spot
- Healthy

**Potato**
- Early blight
- Late blight
- Healthy

**Tomato**
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted spider mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- Healthy

### Data Split
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

## 🏗️ Model Architecture

The project implements a **ResNet-18** architecture built from scratch with the following specifications:

- **Input**: 224×224×3 RGB images
- **Architecture**: Custom ResNet-18 with residual blocks
- **Output**: 15 classes (plant disease categories)
- **Activation**: ReLU
- **Pooling**: Adaptive Average Pooling
- **Regularization**: Batch Normalization

### Key Components

```
Conv1 (7×7, stride=2) → BatchNorm → ReLU → MaxPool
↓
Layer 1: 2 Residual Blocks (64 channels)
Layer 2: 2 Residual Blocks (128 channels, stride=2)
Layer 3: 2 Residual Blocks (256 channels, stride=2)
Layer 4: 2 Residual Blocks (512 channels, stride=2)
↓
AdaptiveAvgPool → Fully Connected (512 → 15)
```

## 🔧 Implementation Details

### Data Augmentation

**Training Transforms:**
- Resize to 224×224
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation ±30%)
- Normalization to tensor

**Validation/Test Transforms:**
- Resize to 224×224
- Normalization to tensor

### Training Configuration

- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate Scheduler**: Cosine Annealing (T_max=75)
- **Device**: CUDA (GPU) if available, else CPU
- **Batch Size**: Configurable via DataLoader

## 📁 Repository Structure

```
PlantVillage/
├── main.ipynb              # Main Jupyter notebook with complete pipeline
├── best_model.pth          # Trained model weights (43MB)
├── Model_stats.jpg         # Model performance visualization
├── PlantLine.pdf           # Additional documentation
└── PlantVillage/           # Dataset directory
    ├── Pepper__bell___Bacterial_spot/
    ├── Pepper__bell___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Tomato_Bacterial_spot/
    ├── Tomato_Early_blight/
    ├── Tomato_Late_blight/
    ├── Tomato_Leaf_Mold/
    ├── Tomato_Septoria_leaf_spot/
    ├── Tomato_Spider_mites_Two_spotted_spider_mite/
    ├── Tomato__Target_Spot/
    ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
    ├── Tomato__Tomato_mosaic_virus/
    └── Tomato_healthy/
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib Pillow tqdm
```

### Required Libraries

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```

### Usage

1. **Clone the repository:**
```bash
git clone https://github.com/kreeett/PlantVillage.git
cd PlantVillage
```

2. **Open and run the notebook:**
```bash
jupyter notebook main.ipynb
```

3. **Load the pre-trained model:**
```python
import torch
from model import ResNet18Scratch

# Initialize model
model = ResNet18Scratch(num_classes=15)

# Load trained weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## 📈 Training Process

The training pipeline includes:

1. **Data Loading**: Custom `PlantVillage` dataset class
2. **Data Splitting**: 70-15-15 train-val-test split with fixed seed
3. **Transform Application**: Different augmentations for train/test
4. **Model Training**: Training loop with progress tracking
5. **Validation**: Evaluation on validation set each epoch
6. **Model Saving**: Best model checkpointing based on validation performance

### Training Functions

- `train_one()`: Trains model for one epoch
- `evaluate()`: Evaluates model on validation/test set
- Both functions track loss and accuracy metrics

## 🎯 Model Performance

Performance metrics are visualized in `Model_stats.jpg`. The model achieves competitive accuracy on the test set after training.

## 💡 Key Features

- ✅ Custom ResNet-18 implementation from scratch
- ✅ Comprehensive data augmentation pipeline
- ✅ Support for both CPU and GPU training
- ✅ Progress tracking with tqdm
- ✅ Learning rate scheduling with Cosine Annealing
- ✅ Model checkpointing for best performance
- ✅ Reproducible results with fixed random seeds

## 🔬 Applications

This model can be used for:
- Real-time plant disease detection in agricultural settings
- Mobile applications for farmers
- Automated greenhouse monitoring systems
- Educational tools for agricultural students
- Research in plant pathology

## 📚 Dataset Citation

If you use this dataset or code, please cite:

```
PlantVillage Dataset
Source: [PlantVillage Project](https://plantvillage.psu.edu/)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- [ ] Add more plant species
- [ ] Implement transfer learning comparisons
- [ ] Add gradCAM visualizations
- [ ] Create a web interface for predictions
- [ ] Implement model quantization for mobile deployment

## 📝 License

Please refer to the original PlantVillage dataset license for usage terms.

## 👤 Author

**kreeett**
- GitHub: [@kreeett](https://github.com/kreeett)

## 🙏 Acknowledgments

- PlantVillage project for providing the dataset
- PyTorch team for the excellent deep learning framework
- The agricultural research community for making this work possible

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: The pre-trained model (`best_model.pth`) is included in this repository for immediate use. The model size is approximately 43MB.
