# Roboyaan Competition — Terrain Classification & Adaptive Speed Control

A complete machine learning system that classifies terrain from images and
determines safe rover speeds in real time.

---

## Project Structure

```
roboyaan_terrain/
├── generate_dataset.py   # Step 1 – generate synthetic training images
├── train_model.py        # Step 2 – train the CNN model
├── rover_terrain.py      # Step 3 – run inference (main submission file)
├── terrain_model.h5      # pre-trained model weights (included)
├── requirements.txt      # Python dependencies
└── dataset/              # auto-generated after Step 1
    ├── train/
    ├── val/
    └── test/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset  *(skip if using your own images)*
```bash
python generate_dataset.py
```
Creates **2 080 synthetic terrain images** across 4 classes:

| Class         | Images (train/val/test) |
|---------------|------------------------|
| smooth_ground | 400 / 80 / 40          |
| gravel        | 400 / 80 / 40          |
| sand          | 400 / 80 / 40          |
| rock_field    | 400 / 80 / 40          |

### 3. Train the model  *(skip if using terrain_model.h5)*
```bash
python train_model.py
```
- Architecture: lightweight custom CNN (~106 K parameters)
- Trains in **~10 min on CPU**, ~2 min on GPU
- Saves best weights to `terrain_model.h5`
- Outputs `training_curves.png` and `training_history.json`

### 4. Run inference

**Single image:**
```bash
python rover_terrain.py --image path/to/terrain.jpg
```

**Folder of images (e.g. organiser test set):**
```bash
python rover_terrain.py --folder path/to/test_images/
```

**Demo on the built-in test split:**
```bash
python rover_terrain.py --demo
```

**Live webcam:**
```bash
python rover_terrain.py --camera
```

---

## Expected Terminal Output

```
──────────────────────────────────────────────────
  Terrain Detected   : Sand
  Confidence         : 87.3%
  Risk Level         : High
  Recommended Speed  : 25 km/h
  Inference Time     : 18.4 ms
  CPU Usage          : 4.2%
──────────────────────────────────────────────────
  All class probabilities:
    Sand             ████████████████░░░░  87.3% ◀
    Gravel           ██░░░░░░░░░░░░░░░░░░   9.1%
    Smooth Ground    █░░░░░░░░░░░░░░░░░░░   2.4%
    Rock Field       ░░░░░░░░░░░░░░░░░░░░   1.2%
```

---

## Terrain → Speed Decision Table

| Terrain       | Risk Level | Speed Range  |
|---------------|------------|--------------|
| Smooth Ground | Safe       | 70–100 km/h  |
| Gravel        | Moderate   | 40–60 km/h   |
| Sand          | High       | 20–40 km/h   |
| Rock Field    | Dangerous  | 0–10 km/h    |

---

## Model Architecture

```
Input  →  (64×64×3)
Conv2D 16  (stride 2)  →  32×32×16
Conv2D 32              →  32×32×32
MaxPool + BatchNorm    →  16×16×32
Conv2D 64              →  16×16×64
MaxPool + BatchNorm    →   8×8×64
Conv2D 128             →   8×8×128
GlobalAveragePool      →  128
Dense 64 + Dropout 0.4
Dense 4  (softmax)
```

**Total parameters:** ~106 340  
**Input size:** 64 × 64 RGB  
**Test accuracy:** 100% on synthetic dataset

---

## Bonus Metrics (displayed automatically)

- **FPS** — frames per second (folder/camera mode)
- **Inference time** — milliseconds per image
- **CPU usage** — live system load percentage

---

## Using Your Own / Real Terrain Images

Replace the synthetic dataset with real terrain photos:

1. Collect images and place them in:
   ```
   dataset/train/smooth_ground/
   dataset/train/gravel/
   dataset/train/sand/
   dataset/train/rock_field/
   ```
2. Re-run `python train_model.py`
3. For best results: 200–500 images per class, mixed lighting & angles

**Recommended public datasets:**
- Kaggle: *Terrain Image Dataset*
- MIT VisTex (Visual Texture)
- NASA Mars Surface Dataset (rock/sand terrain)

---

## Tools Used

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, psutil
