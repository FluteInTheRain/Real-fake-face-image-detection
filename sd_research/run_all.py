"""
run_all.py - Run all 2 experiments sequentially.

Usage:
    cd /Users/khang/Downloads/lab02
    source .venv/bin/activate
    python sd_research/run_all.py

    # Or run a single experiment:
    python sd_research/experiment1_steps.py
    python sd_research/experiment2_cfg.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

from config import (
    MODEL_PATH,
    OUTPUT_DIR,
    DEVICE,
    SEED,
    NEGATIVE_PROMPT,
    CFG_SCALE,
    PROMPTS,
)

import experiment1_steps as exp1
import experiment2_cfg as exp2


def main():
    print("=" * 60)
    print("LAB02 – PART 1: Stable Diffusion Research")
    print(f"Device: {DEVICE.upper()}  |  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Load pipeline ONCE, reuse for all experiments ──────────────────────
    print("\n[Setup] Loading SSD-1B pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # float32 required: float16 causes NaN on MPS
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.to(torch.float32)  # Ensure VAE stays float32 for stable decode
    print("[Setup] Pipeline ready.\n")

    # ── Experiment 1 ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Running Experiment 1: Steps vs Latency / SSIM")
    print("─" * 60)
    STEPS_LIST = [8, 12, 16, 20, 25, 30, 50]
    results1, baseline_images, generated_images = exp1.run_experiment(pipe, STEPS_LIST)
    results1 = exp1.compute_ssim(results1, baseline_images, generated_images)
    exp1.plot_results(results1)

    # ── Experiment 2 ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Running Experiment 2: CFG Scale Sensitivity")
    print("─" * 60)
    clip_model, clip_processor = exp2.load_clip()
    results2 = exp2.run_experiment(
        pipe, clip_model, clip_processor, cfg_scales=[3.0, 5.0, 7.0, 9.0, 12.0]
    )
    exp2.plot_results(results2)
    del clip_model, clip_processor  # Free memory before stress test

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"All experiments complete!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
