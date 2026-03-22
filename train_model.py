"""
Roboyaan Competition – Terrain Classifier  ·  train_model.py
=============================================================
Trains a lightweight custom CNN on the generated terrain dataset.

Usage:
    python generate_dataset.py   # create dataset first (if not already done)
    python train_model.py        # train the model

Output:
    terrain_model.h5        – trained Keras model
    training_history.json   – epoch-by-epoch metrics
    training_curves.png     – accuracy / loss plot
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 64          # input resolution  (height × width)
BATCH_SIZE = 32
EPOCHS     = 50          # EarlyStopping will stop sooner if converged
LR         = 8e-4
CLASSES    = ["smooth_ground", "gravel", "sand", "rock_field"]
DATA_DIR   = Path("dataset")
MODEL_PATH = "terrain_model.h5"
# ──────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 4) -> Model:
    """
    Lightweight CNN  ~106 K parameters.
    Runs on CPU in under 10 minutes for the default dataset size.
    """
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inp)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inp, outputs, name="TerrainClassifier")


def get_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.20,
        brightness_range=[0.75, 1.25],
    )
    plain = ImageDataGenerator(rescale=1.0 / 255)

    kw = dict(target_size=(IMG_SIZE, IMG_SIZE),
              batch_size=BATCH_SIZE, class_mode="categorical", classes=CLASSES)

    train_gen = train_aug.flow_from_directory(DATA_DIR / "train", shuffle=True,  **kw)
    val_gen   = plain.flow_from_directory(DATA_DIR / "val",   shuffle=False, **kw)
    test_gen  = plain.flow_from_directory(DATA_DIR / "test",  shuffle=False, **kw)
    return train_gen, val_gen, test_gen


def plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["accuracy"],     label="Train")
    ax1.plot(history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True)

    ax2.plot(history["loss"],     label="Train")
    ax2.plot(history["val_loss"], label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.close()
    print("  Curves saved  →  training_curves.png")


def main():
    print("=" * 60)
    print("  Roboyaan – Terrain Classifier Training")
    print("=" * 60)

    if not (DATA_DIR / "train").exists():
        print("\nERROR: dataset/ not found.")
        print("  Run:  python generate_dataset.py  first.\n")
        return

    train_gen, val_gen, test_gen = get_generators()
    print(f"\n  Class indices  : {train_gen.class_indices}")
    print(f"  Train          : {train_gen.samples} images")
    print(f"  Val            : {val_gen.samples} images")
    print(f"  Test           : {test_gen.samples} images\n")

    model = build_model(len(CLASSES))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                        callbacks=callbacks, verbose=1)
    print(f"\n  Training completed in {time.time() - t0:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n── Test-set Evaluation ──────────────────────────────────────────")
    best = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = best.evaluate(test_gen, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%\n")

    preds  = np.argmax(best.predict(test_gen, verbose=0), axis=1)
    y_true = test_gen.classes
    print("  Per-class accuracy:")
    for idx, cls in enumerate(CLASSES):
        mask = y_true == idx
        print(f"    {cls:<16} : {(preds[mask] == idx).mean() * 100:.1f}%")

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open("training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs]
                   for k, vs in history.history.items()}, f, indent=2)
    plot_history(history.history)

    print(f"\n  Model saved   →  {MODEL_PATH}")
    print(f"  History saved →  training_history.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
