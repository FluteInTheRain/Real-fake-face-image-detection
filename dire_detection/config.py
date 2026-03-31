"""
config.py - Shared configuration for Part 2: DIRE Fake Image Detection.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "SSD-1B-Traditional")   # Shared with Part 1
OUTPUT_DIR = os.path.join(ROOT_DIR, "dire_detection", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_ID   = "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces"
CONFIG_NAME  = "base_transforms"
TRAIN_LIMIT  = 100
VAL_LIMIT    = 20
BATCH_SIZE   = 4

# ─── DIRE Extractor ──────────────────────────────────────────────────────────
INVERSION_STEPS = 15

# ─── Classifier Training ─────────────────────────────────────────────────────
EPOCHS        = 3
LEARNING_RATE = 1e-4

if __name__ == "__main__":
    print(f"[Config] ROOT_DIR:   {ROOT_DIR}")
    print(f"[Config] MODEL_PATH: {MODEL_PATH}")
    print(f"[Config] OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"[Config] DEVICE:     {DEVICE}")
    print(f"[Config] Model exists: {os.path.isdir(MODEL_PATH)}")
