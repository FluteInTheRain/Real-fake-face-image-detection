"""
config.py - Shared configuration for Part 1: Stable Diffusion Research.
All experiment scripts import from this file to ensure consistency.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
# Resolve path relative to THIS file so scripts can be run from any directory 
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "SSD-1B-Traditional")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sd_research", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ─── Common Experiment Parameters ────────────────────────────────────────────
SEED            = 42
NEGATIVE_PROMPT = "ugly, blurry, deformed, poorly drawn, bad anatomy, artifact, watermark"
CFG_SCALE       = 7.0

PROMPTS = {
    "Macro":    "A highly detailed macro photography of a glowing blue butterfly resting on a dew-covered leaf, cinematic lighting, 8k resolution, photorealistic.",
    "Portrait": "A close-up portrait photo of an elderly woman with deep wrinkles and silver hair, natural sunlight, highly detailed skin texture, 85mm lens.",
    "Cyberpunk":"A futuristic cyberpunk sports car parked in a neon-lit alleyway, sharp edges, metallic reflections, unreal engine 5 render, highly detailed.",
    "Vector":   "A minimalist flat vector illustration of a cute cat drinking coffee, solid pastel colors, clean white background, studio ghibli style.",
}

if __name__ == "__main__":
    print(f"[Config] ROOT_DIR:   {ROOT_DIR}")
    print(f"[Config] MODEL_PATH: {MODEL_PATH}")
    print(f"[Config] OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"[Config] DEVICE:     {DEVICE}")
    print(f"[Config] Model exists: {os.path.isdir(MODEL_PATH)}")
