"""
Roboyaan Competition – Terrain Classification & Adaptive Speed Control
======================================================================
Usage:
  python rover_terrain.py --image path/to/image.jpg
  python rover_terrain.py --folder path/to/folder/
  python rover_terrain.py --camera          (live webcam demo)
  python rover_terrain.py --demo            (run on test dataset)

Output (terminal):
  ─────────────────────────────────────────────
  Terrain Detected   : Sand
  Confidence         : 87.3%
  Risk Level         : High
  Recommended Speed  : 25 km/h
  Inference Time     : 14.2 ms
  ─────────────────────────────────────────────
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import psutil

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "terrain_model.h5"
IMG_SIZE    = 64          # must match training resolution
CLASSES     = ["smooth_ground", "gravel", "sand", "rock_field"]

# Speed & risk tables (as per task spec)
SPEED_TABLE = {
    "smooth_ground": (70, 100),
    "gravel":        (40, 60),
    "sand":          (20, 40),
    "rock_field":    (0,  10),
}
RISK_TABLE = {
    "smooth_ground": "Safe",
    "gravel":        "Moderate",
    "sand":          "High",
    "rock_field":    "Dangerous",
}
# Friendly display names
DISPLAY_NAMES = {
    "smooth_ground": "Smooth Ground",
    "gravel":        "Gravel",
    "sand":          "Sand",
    "rock_field":    "Rock Field",
}
# ANSI colours
C = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "orange": "\033[33m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}
RISK_COLOUR = {
    "Safe":      C["green"],
    "Moderate":  C["yellow"],
    "High":      C["orange"],
    "Dangerous": C["red"],
}
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Load the trained Keras model."""
    if not Path(MODEL_PATH).exists():
        print(f"\n{C['red']}ERROR:{C['reset']} Model file '{MODEL_PATH}' not found.")
        print("  Run  python train_model.py  first.\n")
        sys.exit(1)
    import tensorflow as tf
    print(f"{C['dim']}Loading model …{C['reset']}", end="\r")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"{C['green']}✔  Model loaded{C['reset']}           ")
    return model


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalise, and add batch dimension."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def pick_speed(terrain: str) -> int:
    lo, hi = SPEED_TABLE[terrain]
    if lo == hi:
        return lo
    # Pick middle of range, jitter slightly for realism
    mid = (lo + hi) // 2
    jitter = np.random.randint(-3, 4)
    return int(np.clip(mid + jitter, lo, hi))


def classify(model, img_bgr: np.ndarray) -> dict:
    """Run inference and return structured result dict."""
    t0         = time.perf_counter()
    tensor     = preprocess(img_bgr)
    probs      = model.predict(tensor, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    idx        = int(np.argmax(probs))
    terrain    = CLASSES[idx]
    confidence = float(probs[idx]) * 100

    return {
        "terrain"    : terrain,
        "display"    : DISPLAY_NAMES[terrain],
        "confidence" : confidence,
        "risk"       : RISK_TABLE[terrain],
        "speed"      : pick_speed(terrain),
        "speed_range": SPEED_TABLE[terrain],
        "time_ms"    : elapsed_ms,
        "all_probs"  : {CLASSES[i]: float(probs[i]) * 100 for i in range(len(CLASSES))},
    }


def print_result(r: dict, bonus: bool = True):
    risk_col = RISK_COLOUR.get(r["risk"], C["reset"])
    sep = f"{C['dim']}{'─' * 50}{C['reset']}"
    print(sep)
    print(f"  {C['bold']}Terrain Detected{C['reset']}   : "
          f"{C['cyan']}{r['display']}{C['reset']}")
    print(f"  {C['bold']}Confidence{C['reset']}         : "
          f"{r['confidence']:.1f}%")
    print(f"  {C['bold']}Risk Level{C['reset']}         : "
          f"{risk_col}{r['risk']}{C['reset']}")
    print(f"  {C['bold']}Recommended Speed{C['reset']}  : "
          f"{C['bold']}{r['speed']} km/h{C['reset']}  "
          f"{C['dim']}(range {r['speed_range'][0]}–{r['speed_range'][1]} km/h){C['reset']}")
    if bonus:
        print(f"  {C['dim']}Inference Time     : {r['time_ms']:.1f} ms{C['reset']}")
        cpu = psutil.cpu_percent(interval=None)
        print(f"  {C['dim']}CPU Usage          : {cpu:.1f}%{C['reset']}")
    print(sep)
    # Probability bar chart
    print(f"  {C['dim']}All class probabilities:{C['reset']}")
    for cls, prob in sorted(r["all_probs"].items(), key=lambda x: -x[1]):
        bar_len = int(prob / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        marker  = " ◀" if cls == r["terrain"] else ""
        print(f"    {DISPLAY_NAMES[cls]:<16} {bar} {prob:5.1f}%{marker}")
    print()


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_image(model, path: str):
    img = cv2.imread(path)
    if img is None:
        print(f"{C['red']}Cannot read image: {path}{C['reset']}")
        return
    print(f"\n  Image: {path}")
    r = classify(model, img)
    print_result(r)


def run_folder(model, folder: str):
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"No images found in {folder}")
        return
    print(f"\n  Found {len(files)} image(s) in '{folder}'\n")
    correct = 0
    for fp in sorted(files):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        r = classify(model, img)
        # If filename matches a class name, check accuracy
        for cls in CLASSES:
            if cls in fp.stem.lower() or cls in str(fp.parent).lower():
                if r["terrain"] == cls:
                    correct += 1
                break
        print(f"  {fp.name}")
        print_result(r, bonus=False)
    fps_avg = 1000 / np.mean([classify(model, cv2.imread(str(f)))["time_ms"]
                               for f in files[:5]])
    print(f"  Average FPS: {fps_avg:.1f}")


def run_demo(model):
    """Run inference on the test split and print summary stats."""
    test_dir = Path("dataset/test")
    if not test_dir.exists():
        print("Test dataset not found. Run generate_dataset.py first.")
        return
    run_folder(model, str(test_dir))


def run_camera(model):
    """Live webcam loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{C['red']}Cannot open camera.{C['reset']}")
        return
    print(f"\n{C['bold']}Live camera mode{C['reset']} – press Q to quit\n")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = classify(model, frame)

        # Overlay on frame
        risk_bgr = {
            "Safe":      (0, 200, 0),
            "Moderate":  (0, 200, 255),
            "High":      (0, 140, 255),
            "Dangerous": (0, 0, 220),
        }
        col = risk_bgr.get(r["risk"], (255, 255, 255))
        cv2.rectangle(frame, (0, 0), (340, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Terrain: {r['display']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"Confidence: {r['confidence']:.1f}%",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Risk: {r['risk']}  Speed: {r['speed']} km/h",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Roboyaan – Terrain Classifier", frame)
        # Also print to terminal
        print_result(r)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Roboyaan Terrain Classifier & Adaptive Speed Control"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  metavar="PATH", help="Single image file")
    group.add_argument("--folder", metavar="PATH", help="Folder of images")
    group.add_argument("--camera", action="store_true", help="Live webcam")
    group.add_argument("--demo",   action="store_true", help="Run on test split")

    args   = parser.parse_args()
    model  = load_model()

    print("\n" + "=" * 50)
    print("  Roboyaan Terrain Classification System")
    print("=" * 50)

    if args.image:
        run_image(model, args.image)
    elif args.folder:
        run_folder(model, args.folder)
    elif args.camera:
        run_camera(model)
    elif args.demo:
        run_demo(model)


if __name__ == "__main__":
    main()
