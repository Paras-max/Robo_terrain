"""
Roboyaan Competition – Web Server  ·  app.py
=============================================
Serves the terrain_ui.html frontend and exposes a /predict API
that the browser calls when the user uploads an image.

Usage:
    pip install flask
    python app.py
    → open http://localhost:5000 in your browser
"""


import os, io, base64, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load model once at startup ─────────────────────────────────────────────
import tensorflow as tf
print("Loading terrain model …", end=" ", flush=True)
MODEL      = tf.keras.models.load_model("terrain_model.h5")
IMG_SIZE   = 64
CLASSES    = ["smooth_ground", "gravel", "sand", "rock_field"]
DISPLAY    = {"smooth_ground": "Smooth Ground", "gravel": "Gravel",
              "sand": "Sand",                   "rock_field": "Rock Field"}
RISK       = {"smooth_ground": "Safe", "gravel": "Moderate",
              "sand": "High",          "rock_field": "Dangerous"}
SPEED      = {"smooth_ground": (70,100), "gravel": (40,60),
              "sand": (20,40),           "rock_field": (0,10)}
print("done ✓")


def preprocess(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "terrain_ui.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with field 'image'  (file upload)
          OR application/json with field 'image_b64' (base64 string)
    Returns: JSON with terrain, confidence, risk, speed, all_scores
    """
    try:
        # ── get raw image bytes ──────────────────────────────────────────
        if request.content_type and "multipart" in request.content_type:
            file = request.files.get("image")
            if not file:
                return jsonify({"error": "No image field in form data"}), 400
            img_bytes = file.read()
        else:
            data = request.get_json(force=True)
            b64  = data.get("image_b64", "")
            # strip data-URL prefix if present
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)

        # ── run inference ────────────────────────────────────────────────
        t0     = time.perf_counter()
        tensor = np.expand_dims(preprocess(img_bytes), axis=0)
        probs  = MODEL.predict(tensor, verbose=0)[0]
        ms     = (time.perf_counter() - t0) * 1000

        idx        = int(np.argmax(probs))
        terrain    = CLASSES[idx]
        confidence = float(probs[idx]) * 100
        lo, hi     = SPEED[terrain]
        speed      = int((lo + hi) / 2)

        all_scores = {DISPLAY[c]: round(float(probs[i]) * 100, 1)
                      for i, c in enumerate(CLASSES)}

        return jsonify({
            "terrain"    : DISPLAY[terrain],
            "confidence" : round(confidence, 1),
            "risk"       : RISK[terrain],
            "speed"      : speed,
            "speed_range": f"{lo}–{hi}",
            "time_ms"    : round(ms, 1),
            "fps"        : round(1000 / ms, 1),
            "all_scores" : all_scores,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Roboyaan Terrain Classifier")
    print("  Open → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
