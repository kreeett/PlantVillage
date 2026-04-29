# PlantVillage Demo — Setup & Run Guide

A Streamlit demo of the ResNet-18 PlantVillage disease classifier, designed for live presentation to Dr. Tariq Bdair.

## What's in this folder

```
demo/
├── app.py                          # Main Streamlit app (run this)
├── generate_confusion_matrix.py    # Run ONCE before the demo
├── README.md                       # This file
└── presentation_script.md          # What to say during the demo
```

## What you need to add

Put these files into the same folder as `app.py` before running:

```
best_model.pth           # your trained weights (from training)
Model_stats.jpg          # the training-history figure
PlantVillage/            # the dataset folder (same layout as training)
    Pepper__bell___Bacterial_spot/
    Pepper__bell___healthy/
    Potato___Early_blight/
    ...etc
```

## Setup (one-time, ~5 minutes)

### 1. Create a virtual environment

```bash
cd path/to/demo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install streamlit torch torchvision pillow matplotlib pandas numpy
```

If you already have PyTorch with CUDA installed elsewhere on the laptop, install Streamlit into that environment instead so inference uses your GPU.

### 3. Generate the confusion matrix (once)

This runs the test set through your saved model and produces three artifact files. Takes 1–2 minutes on the RTX 4070.

```bash
python generate_confusion_matrix.py
```

You should see:

```
Building dataset...
Classes (15): [...]
Loading model...
Using device: cuda
Running inference on test set...
Overall accuracy: 99.03% (3066/3096)
Wrote: confusion_matrix.npy, confusion_matrix.png, class_names.json
```

If overall accuracy doesn't match 99.03% exactly, that's fine — small differences from re-splitting are normal even with the same seed across machines, but it should be very close.

## Run the demo

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Use the sidebar on the left to navigate between the 8 sections.

To stop: `Ctrl+C` in the terminal.

## Pre-demo checklist (do this 30 minutes before class)

1. ✅ Plug in the laptop (don't run on battery — GPU throttles)
2. ✅ Connect to projector, set display to **mirror** (not extend)
3. ✅ Open a fresh terminal in the demo folder
4. ✅ Activate the virtual environment
5. ✅ Run `python generate_confusion_matrix.py` once to verify the pipeline still works
6. ✅ Run `streamlit run app.py` and click through all 8 sections to confirm each loads
7. ✅ In Section 7 (Live Inference), pick 2–3 sample classes and verify predictions appear
8. ✅ Open the Streamlit settings (top-right `⋮` menu) → switch to **Wide mode** and **Light theme** for projector visibility
9. ✅ Increase browser zoom to 110–125% so people in the back row can read

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Wrong virtualenv active. Re-run `source .venv/bin/activate` |
| App loads but Section 4 (Architecture) fails | `best_model.pth` missing or corrupted, or model class definition doesn't match the saved weights |
| Section 7 (Live Inference) is slow | `device` showed as CPU in the sidebar. Check CUDA install with `python -c "import torch; print(torch.cuda.is_available())"` |
| Sample gallery is empty in Sections 2 / 7 | `PlantVillage/` folder is not in the same directory as `app.py` |
| Confusion matrix shows "files not found" | You haven't run `generate_confusion_matrix.py` yet |
| Charts look tiny on the projector | Browser zoom Ctrl/Cmd + several times. Streamlit also has Settings → Wide mode |

## After the demo

To shut down cleanly:

```bash
# Ctrl+C in the terminal
deactivate    # exit the virtualenv
```
