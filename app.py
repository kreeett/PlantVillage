"""
PlantVillage Demo — Streamlit app.

Run with:
    streamlit run app.py

Expected files in the same directory:
    - app.py                  (this file)
    - best_model.pth          (your trained weights)
    - confusion_matrix.npy    (from generate_confusion_matrix.py)
    - confusion_matrix.png    (fallback figure)
    - class_names.json        (from generate_confusion_matrix.py)
    - PlantVillage/           (the dataset folder, used to grab sample images)
    - Model_stats.jpg         (your training-history figure)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="PlantVillage — ResNet-18 Demo",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# Model definition (must match training code exactly)
# ============================================================
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


# ============================================================
# Cached resources
# ============================================================
@st.cache_resource
def load_model_and_classes():
    with open("class_names.json") as f:
        classes = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18Scratch(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model, classes, device


@st.cache_data
def load_confusion_matrix():
    return np.load("confusion_matrix.npy")


@st.cache_data
def get_sample_images_per_class(n_per_class=1):
    """Pick one example image from each class folder for quick demo gallery."""
    root = Path("PlantVillage")
    samples = {}
    if not root.exists():
        return samples
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        jpgs = sorted(cls_dir.glob("*.jpg"))
        if jpgs:
            samples[cls_dir.name] = [str(p) for p in jpgs[:n_per_class]]
    return samples


@st.cache_data
def get_class_distribution():
    """Count images per class — used in dataset section."""
    root = Path("PlantVillage")
    counts = {}
    if not root.exists():
        return counts
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name] = len(list(cls_dir.glob("*.jpg")))
    return counts


# ============================================================
# Inference helpers
# ============================================================
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_pil, model, classes, device):
    x = infer_transform(image_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = probs.argsort()[::-1]
    return [(classes[i], float(probs[i])) for i in top_idx]


def pretty(name):
    """Human-friendly class name."""
    return name.replace("_", " ").replace("  ", " ").strip()


# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("🌿 PlantVillage Demo")
st.sidebar.caption("Special Topics in AI (1) — PSUT")
st.sidebar.caption("Saif Al-Dein Shoujen · Amer Mansour")
st.sidebar.markdown("---")

SECTIONS = [
    "1. Overview",
    "2. The Problem",
    "3. Dataset",
    "4. Architecture",
    "5. Training Process",
    "6. Results",
    "7. Live Inference",
    "8. Challenges & Conclusion",
]
section = st.sidebar.radio("Sections", SECTIONS, label_visibility="collapsed")

# Show device in sidebar
device_info = "CUDA" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown("---")
st.sidebar.caption(f"Compute: **{device_info}**")


# ============================================================
# Section 1: Overview
# ============================================================
if section == SECTIONS[0]:
    st.title("Plant Disease Classification")
    st.subheader("ResNet-18 trained from scratch on PlantVillage")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", "99.03%")
    col2.metric("Classes", "15")
    col3.metric("Test Images", "3,096")
    col4.metric("Epochs", "75")

    st.markdown("---")
    st.markdown("""
    ### What we built
    A ResNet-18 convolutional neural network, implemented from first principles in PyTorch
    (no `torchvision.models`, no pretrained weights), trained end-to-end to classify leaf
    images of pepper, potato, and tomato plants into healthy and diseased categories.

    ### Why it matters
    Plant disease is a leading cause of global crop loss. A reliable image classifier
    deployable on a phone could let farmers diagnose problems early without specialist help.

    ### Course context
    - **Course:** Special Topics in Artificial Intelligence (1)
    - **Instructor:** Dr. Tariq Bdair
    - **Institution:** Princess Sumaya University for Technology
    - **Authors:** Saif Al-Dein Shoujen, Amer Mansour
    - **Repository:** [github.com/kreeett/PlantVillage](https://github.com/kreeett/PlantVillage)
    """)


# ============================================================
# Section 2: The Problem
# ============================================================
elif section == SECTIONS[1]:
    st.title("The Problem")
    st.markdown("""
    Identifying plant disease from a leaf image is a fine-grained visual task.
    The same crop can show many different diseases, and the visual differences
    between them are sometimes subtle — small spots, slight discolouration, leaf
    curling. The model needs to distinguish **15 categories** across 3 crops.
    """)

    samples = get_sample_images_per_class(n_per_class=1)
    if samples:
        st.markdown("### One sample from each class")
        items = list(samples.items())
        cols_per_row = 5
        for row_start in range(0, len(items), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, (cls, paths) in zip(cols, items[row_start:row_start + cols_per_row]):
                with col:
                    img = Image.open(paths[0]).convert("RGB")
                    st.image(img, caption=pretty(cls), width="stretch")
    else:
        st.warning("PlantVillage/ folder not found in working directory. "
                   "Sample gallery is unavailable but the rest of the demo still works.")


# ============================================================
# Section 3: Dataset
# ============================================================
elif section == SECTIONS[2]:
    st.title("Dataset")

    st.markdown("""
    **Source:** PlantVillage dataset (Kaggle), ~20,000 RGB leaf images on uniform backgrounds.

    **Splits:** 70% train · 15% validation · 15% test, using `random_split` with seed 123 for reproducibility.

    **Augmentation (training only):** resize to 224×224, random horizontal flip, random rotation, colour jitter.
    Validation and test see only resize + tensor conversion.
    """)

    counts = get_class_distribution()
    if counts:
        st.markdown("### Class distribution")
        df = pd.DataFrame({
            "Class": [pretty(k) for k in counts.keys()],
            "Image count": list(counts.values()),
        }).sort_values("Image count", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["Class"], df["Image count"], color="#3a6ea5")
        ax.set_xlabel("Number of images")
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(df["Image count"]):
            ax.text(v + 20, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total images", f"{sum(counts.values()):,}")
        col2.metric("Largest class", f"{max(counts.values()):,}")
        col3.metric("Smallest class", f"{min(counts.values()):,}")

        st.info(
            "**Note the imbalance:** the largest class has roughly "
            f"{max(counts.values()) // max(min(counts.values()), 1)}× more images than the smallest. "
            "We addressed this through augmentation rather than weighted loss — "
            "see the Challenges section."
        )
    else:
        st.warning("PlantVillage/ folder not found — class distribution chart unavailable.")


# ============================================================
# Section 4: Architecture
# ============================================================
elif section == SECTIONS[3]:
    st.title("Architecture: ResNet-18 from scratch")

    st.markdown("""
    We implemented the network using only `torch.nn` primitives. No pretrained weights.
    The structure follows He et al. (2016).
    """)

    st.markdown("### Top-level flow")
    st.code("""
Input (3 × 224 × 224)
        │
        ▼
   Conv 7×7, stride 2, 64 filters
        │
   BatchNorm → ReLU
        │
   MaxPool 3×3, stride 2
        │
        ▼
   Stage 1:  2 residual blocks,  64 ch,  stride 1
   Stage 2:  2 residual blocks, 128 ch,  stride 2
   Stage 3:  2 residual blocks, 256 ch,  stride 2
   Stage 4:  2 residual blocks, 512 ch,  stride 2
        │
        ▼
   Adaptive Average Pool (1×1)
        │
   Flatten → 512-dim vector
        │
   Fully Connected → 15 logits
""", language="text")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What's in a residual block?")
        st.code("""
def forward(self, x):
    identity = x
    out = relu(bn1(conv1(x)))   # 3×3 conv
    out = bn2(conv2(out))        # 3×3 conv
    if self.downsample:          # 1×1 projection
        identity = downsample(x) # if shape changes
    out += identity              # SKIP CONNECTION
    return relu(out)
""", language="python")

    with col2:
        st.markdown("### Why skip connections matter")
        st.markdown("""
        In a deep network without skip connections, gradients have to pass through
        every layer during backprop and tend to **vanish** — the early layers stop learning.

        The skip connection adds the input directly to the output of the block.
        This gives the gradient a **direct path** back to earlier layers, which makes
        deep networks trainable in practice.

        Each block effectively learns a **residual** — what to *add* to the
        input — rather than a full transformation. If the right thing to do
        is "leave the input alone," learning zero is easier than learning identity.
        """)

    st.markdown("### Parameter count")
    model, classes, device = load_model_and_classes()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total parameters", f"{n_params:,}")
    col2.metric("Trainable parameters", f"{n_trainable:,}")
    col3.metric("Output classes", len(classes))


# ============================================================
# Section 5: Training Process
# ============================================================
elif section == SECTIONS[4]:
    st.title("Training Process")

    st.markdown("### Hyperparameters")
    hp_data = {
        "Hyperparameter": [
            "Optimizer", "Initial learning rate", "LR scheduler", "Loss",
            "Batch size", "Epochs", "Input size", "Weight init"
        ],
        "Value": [
            "Adam", "1e-3", "Cosine annealing (T_max = 75)", "Cross-entropy",
            "128", "75", "224 × 224", "Kaiming normal (fan_out, ReLU)"
        ],
    }
    st.dataframe(pd.DataFrame(hp_data), hide_index=True, width="stretch")

    st.markdown("### Training history")
    if Path("Model_stats.jpg").exists():
        st.image("Model_stats.jpg", caption="Training and validation curves over 75 epochs",
                 width="stretch")
    else:
        st.warning("Model_stats.jpg not found in working directory.")

    st.markdown("### What we observe")
    st.markdown("""
    - **Training loss** decreases smoothly and monotonically from ≈1.3 to near zero.
    - **Validation loss** is volatile in the first ~35 epochs, with sharp spikes,
      then settles into a stable band around 0.02–0.05 from epoch 40 onwards.
    - **Train and validation accuracy** converge tightly at the end of training,
      with no widening gap → augmentation prevented severe overfitting.
    - The early volatility is partly an artifact of small classes:
      misclassifying just 2–3 images in *Potato healthy* swings the average loss noticeably.
      As the cosine schedule shrinks the learning rate, updates get smaller and the curve smooths out.
    """)

    st.markdown("### Hardware")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU", "Ryzen 9 7950HX")
    col2.metric("GPU", "RTX 4070 Laptop")
    col3.metric("RAM", "32 GB DDR5")


# ============================================================
# Section 6: Results
# ============================================================
elif section == SECTIONS[5]:
    st.title("Results")

    st.markdown("### Overall test performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "99.03%")
    col2.metric("Correct", "3,066")
    col3.metric("Total", "3,096")

    st.markdown("### Per-class accuracy")
    per_class = [
        ("Pepper bell — Bacterial spot", 158, 158),
        ("Pepper bell — healthy", 219, 219),
        ("Potato — Early blight", 139, 139),
        ("Potato — Late blight", 157, 160),
        ("Potato — healthy", 21, 22),
        ("Tomato — Bacterial spot", 341, 344),
        ("Tomato — Early blight", 165, 172),
        ("Tomato — Late blight", 293, 296),
        ("Tomato — Leaf Mold", 144, 144),
        ("Tomato — Septoria leaf spot", 262, 265),
        ("Tomato — Spider mites", 232, 235),
        ("Tomato — Target Spot", 204, 209),
        ("Tomato — Yellow Leaf Curl Virus", 446, 447),
        ("Tomato — Mosaic virus", 54, 55),
        ("Tomato — healthy", 231, 231),
    ]
    df = pd.DataFrame(per_class, columns=["Class", "Correct", "Total"])
    df["Accuracy"] = (df["Correct"] / df["Total"] * 100).round(2)
    df["Accuracy %"] = df["Accuracy"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(
        df[["Class", "Correct", "Total", "Accuracy %"]].sort_values("Accuracy %", ascending=False).reset_index(drop=True),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Confusion matrix")
    try:
        cm = load_confusion_matrix()
        with open("class_names.json") as f:
            classes = json.load(f)
        n_classes = len(classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        short_names = [pretty(c) for c in classes]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm[i, j]
                if v > 0:
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Cells are coloured by row-normalized accuracy. The diagonal dominates — "
            "off-diagonal cells reveal where the model gets confused."
        )

        # Find the most-confused pair
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        if cm_off.max() > 0:
            i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
            st.info(
                f"**Most common confusion:** the model misclassified "
                f"`{pretty(classes[i])}` as `{pretty(classes[j])}` "
                f"**{cm_off[i, j]} time(s)** in the test set."
            )

    except FileNotFoundError:
        st.warning(
            "Confusion matrix files not found. Run "
            "`python generate_confusion_matrix.py` first to produce them."
        )
        if Path("confusion_matrix.png").exists():
            st.image("confusion_matrix.png")


# ============================================================
# Section 7: Live Inference
# ============================================================
elif section == SECTIONS[6]:
    st.title("Live Inference")
    st.markdown("Pick a sample leaf or upload your own image — the model classifies it in real time.")

    model, classes, device = load_model_and_classes()
    samples = get_sample_images_per_class(n_per_class=1)

    tab_sample, tab_upload = st.tabs(["📁 Sample gallery", "⬆️ Upload your own"])

    image_to_classify = None
    true_label = None

    with tab_sample:
        if samples:
            cls_choice = st.selectbox(
                "Pick a class to test",
                list(samples.keys()),
                format_func=pretty,
            )
            if cls_choice:
                image_to_classify = Image.open(samples[cls_choice][0]).convert("RGB")
                true_label = cls_choice
        else:
            st.warning("PlantVillage/ folder not found, sample gallery unavailable.")

    with tab_upload:
        uploaded = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_to_classify = Image.open(uploaded).convert("RGB")
            true_label = None  # unknown

    if image_to_classify is not None:
        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(image_to_classify, caption="Input", width="stretch")
            if true_label:
                st.caption(f"**Ground truth:** {pretty(true_label)}")

        with col_pred:
            preds = predict(image_to_classify, model, classes, device)
            top_label, top_prob = preds[0]

            st.markdown("### Prediction")
            if true_label:
                if top_label == true_label:
                    st.success(f"✅ **{pretty(top_label)}** — {top_prob*100:.2f}%")
                else:
                    st.error(f"❌ Predicted **{pretty(top_label)}** — {top_prob*100:.2f}%")
            else:
                st.info(f"**{pretty(top_label)}** — {top_prob*100:.2f}%")

            st.markdown("#### Top 5 probabilities")
            top5 = preds[:5]
            df_top = pd.DataFrame({
                "Class": [pretty(c) for c, _ in top5],
                "Probability": [p for _, p in top5],
            })
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(df_top["Class"][::-1], df_top["Probability"][::-1], color="#3a6ea5")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for i, p in enumerate(df_top["Probability"][::-1]):
                ax.text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)


# ============================================================
# Section 8: Challenges & Conclusion
# ============================================================
elif section == SECTIONS[7]:
    st.title("Challenges & Conclusion")

    st.markdown("### Challenges we faced")

    with st.expander("1. Folder-based labelling", expanded=True):
        st.markdown("""
        The PlantVillage archive is organized as one folder per class, with no
        manifest file. We wrote a custom `Dataset` class that walks the directory
        tree, reads the class names off the folder names, and assigns integer
        indices alphabetically. The folder names also use inconsistent
        underscore conventions (`Pepper__bell___...` vs `Tomato_...`),
        which the parser had to tolerate.
        """)

    with st.expander("2. Class imbalance", expanded=True):
        st.markdown("""
        The largest class has ~20× more images than the smallest. We addressed
        this **indirectly** through data augmentation rather than explicit
        class-weighted loss or oversampling. Augmentation exposes the smaller
        classes to a wider variety of viewpoints during training, partially
        compensating for the limited number of unique source images.

        With more time, we would compare this approach against
        `nn.CrossEntropyLoss(weight=...)` with inverse-frequency weights.
        """)

    with st.expander("3. Validation curve volatility", expanded=True):
        st.markdown("""
        The validation loss spikes visibly during the first 35 epochs.
        We initially thought this was instability in the model, but we
        confirmed it was an artifact of small validation classes:
        misclassifying 2–3 images in *Potato healthy* (only ~22 validation samples)
        produces a visible jump in the average loss.
        The cosine schedule eventually shrinks the learning rate enough
        that the curve smooths out.
        """)

    st.markdown("### Conclusion")
    st.markdown("""
    A ResNet-18 trained from scratch reaches **99.03% test accuracy** on PlantVillage
    when paired with sensible augmentation, Adam, and cosine learning-rate annealing.
    The model performs uniformly well across classes, with the only sub-98% scores
    appearing in the smallest classes where a single misclassification dominates the metric.

    **The honest caveat:** PlantVillage images have uniform backgrounds and centered single leaves.
    Real-world field photos contain soil, multiple leaves, varied lighting, and partial occlusion.
    Models that hit 99% on PlantVillage typically lose substantial accuracy on field-captured images —
    that's a known limitation in the literature, and would be the next thing to study.

    ### Future work
    - Class-weighted loss to close the small remaining gap on under-represented classes
    - Evaluation on field-captured images to measure the domain-shift drop
    - Mobile deployment (ONNX or CoreML export) for in-field inference
    """)

    st.markdown("---")
    st.markdown("**Repository:** [github.com/kreeett/PlantVillage](https://github.com/kreeett/PlantVillage)")
