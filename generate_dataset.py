"""
Roboyaan Competition – Terrain Dataset Generator
Generates synthetic but realistic terrain images for training.
"""

import os
import numpy as np
import cv2
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
N_TRAIN     = 400   # images per class for training
N_VAL       = 80    # images per class for validation
N_TEST      = 40    # images per class for testing
SEED        = 42
CLASSES     = ["smooth_ground", "gravel", "sand", "rock_field"]

BASE_DIR    = Path("dataset")
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)


def add_noise(img, intensity=15):
    noise = rng.integers(-intensity, intensity, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_lighting(img):
    """Random lighting gradient to simulate real-world illumination."""
    h, w = img.shape[:2]
    cx = rng.integers(0, w)
    cy = rng.integers(0, h)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    dist = (dist / dist.max() * 60).astype(np.int16)
    bias = rng.choice([-1, 1])
    out = np.clip(img.astype(np.int16) + bias * dist[:, :, None], 0, 255)
    return out.astype(np.uint8)


# ── Per-class generators ──────────────────────────────────────────────────────

def gen_smooth_ground(i):
    """Flat asphalt / packed dirt – even gray-brown tones."""
    base_color = rng.integers([60, 50, 40], [110, 90, 75], 3)
    img = np.full((IMG_SIZE, IMG_SIZE, 3), base_color, dtype=np.uint8)
    # Subtle horizontal streaks
    for _ in range(rng.integers(3, 8)):
        y  = rng.integers(0, IMG_SIZE)
        c  = base_color + rng.integers(-15, 15, 3)
        c  = np.clip(c, 0, 255).astype(np.uint8)
        cv2.line(img, (0, y), (IMG_SIZE, y + rng.integers(-5, 5)),
                 c.tolist(), rng.integers(1, 4))
    img = add_noise(img, 10)
    img = add_lighting(img)
    return img


def gen_gravel(i):
    """Small pebbles – mottled gray with circular blobs."""
    base = rng.integers(80, 130)
    img  = np.full((IMG_SIZE, IMG_SIZE, 3), [base]*3, dtype=np.uint8)
    n_stones = rng.integers(40, 90)
    for _ in range(n_stones):
        cx   = rng.integers(0, IMG_SIZE)
        cy   = rng.integers(0, IMG_SIZE)
        rad  = rng.integers(2, 8)
        col  = int(rng.integers(50, 180))
        cv2.circle(img, (cx, cy), rad, (col, col-10, col-20), -1)
        # Edge highlight
        cv2.circle(img, (cx-1, cy-1), rad, (min(col+30,255),)*3, 1)
    img = add_noise(img, 18)
    img = add_lighting(img)
    return img


def gen_sand(i):
    """Sandy – warm beige/yellow with ripple patterns."""
    r = int(rng.integers(180, 220))
    g = int(rng.integers(160, 200))
    b = int(rng.integers(80, 130))
    img = np.full((IMG_SIZE, IMG_SIZE, 3), [b, g, r], dtype=np.uint8)
    # Ripple lines
    freq  = rng.uniform(0.05, 0.15)
    phase = rng.uniform(0, np.pi * 2)
    for y in range(IMG_SIZE):
        offset = int(6 * np.sin(freq * y + phase))
        amp    = int(rng.integers(5, 15))
        x_start = max(0, offset)
        x_end   = min(IMG_SIZE, IMG_SIZE + offset)
        img[y, x_start:x_end] = np.clip(
            img[y, x_start:x_end].astype(np.int16) + amp, 0, 255
        ).astype(np.uint8)
    # Small dune bumps
    for _ in range(rng.integers(2, 6)):
        cx  = rng.integers(0, IMG_SIZE)
        cy  = rng.integers(0, IMG_SIZE)
        rad = rng.integers(10, 30)
        col = (b-10, g+20, r+10)
        cv2.ellipse(img, (cx, cy), (rad, rad//2),
                    rng.integers(0, 180), 0, 360, col, -1)
    img = add_noise(img, 12)
    img = add_lighting(img)
    return img


def gen_rock_field(i):
    """Rocky terrain – dark irregular boulders & crevices."""
    base = int(rng.integers(30, 70))
    img  = np.full((IMG_SIZE, IMG_SIZE, 3), [base, base-5, base-10], dtype=np.uint8)
    n_rocks = rng.integers(8, 20)
    for _ in range(n_rocks):
        pts = rng.integers(0, IMG_SIZE, (rng.integers(4, 8), 1, 2))
        col = int(rng.integers(50, 140))
        cv2.fillPoly(img, [pts], (col-10, col, col+5))
        cv2.polylines(img, [pts], True, (max(0,col-40),)*3, 1)
    # Deep cracks
    for _ in range(rng.integers(3, 8)):
        x1, y1 = rng.integers(0, IMG_SIZE, 2)
        x2, y2 = rng.integers(0, IMG_SIZE, 2)
        cv2.line(img, (x1, y1), (x2, y2), (10, 10, 10), rng.integers(1, 3))
    img = add_noise(img, 20)
    img = add_lighting(img)
    return img


GENERATORS = {
    "smooth_ground": gen_smooth_ground,
    "gravel":        gen_gravel,
    "sand":          gen_sand,
    "rock_field":    gen_rock_field,
}


def create_split(split_name, n_per_class):
    for cls in CLASSES:
        folder = BASE_DIR / split_name / cls
        folder.mkdir(parents=True, exist_ok=True)
        gen_fn = GENERATORS[cls]
        for i in range(n_per_class):
            img  = gen_fn(i)
            path = folder / f"{cls}_{i:04d}.jpg"
            cv2.imwrite(str(path), img)
    print(f"  [{split_name}]  {n_per_class} images × {len(CLASSES)} classes = "
          f"{n_per_class * len(CLASSES)} total")


if __name__ == "__main__":
    print("=" * 55)
    print("  Roboyaan – Terrain Dataset Generator")
    print("=" * 55)
    create_split("train", N_TRAIN)
    create_split("val",   N_VAL)
    create_split("test",  N_TEST)
    total = (N_TRAIN + N_VAL + N_TEST) * len(CLASSES)
    print(f"\n✅  Dataset ready  →  {BASE_DIR}/  ({total} images total)")
