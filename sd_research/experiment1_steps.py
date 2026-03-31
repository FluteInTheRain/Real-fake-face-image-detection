"""
experiment1_steps.py - Experiment 1: Inference Steps vs Latency & SSIM.

Research Question:
    At what number of inference steps does the SSD-1B model reach convergence
    (SSIM >= 0.95 relative to the 50-step reference), while minimising latency?

Output:
    outputs/exp1_individual_prompts.png
    outputs/exp1_average.png
    outputs/exp1_generated_images/   (all generated PNGs)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

from config import MODEL_PATH, OUTPUT_DIR, DEVICE, SEED, NEGATIVE_PROMPT, CFG_SCALE, PROMPTS

IMG_DIR = os.path.join(OUTPUT_DIR, "exp1_generated_images")
os.makedirs(IMG_DIR, exist_ok=True)


def load_pipeline():
    print(f"[Exp1] Loading pipeline from: {MODEL_PATH}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # float32 required on MPS to avoid NaN/black images
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.to(torch.float32)
    return pipe


def run_experiment(pipe, steps_list):
    """Generate images at each step count and record latency."""
    os.makedirs(IMG_DIR, exist_ok=True)  # Ensure directory exists at runtime
    results, baseline_images, generated_images = [], {}, {}

    for p_name, prompt_text in PROMPTS.items():
        print(f"\n  Prompt: {p_name}")
        generated_images[p_name] = {}

        for steps in steps_list:
            generator = torch.Generator(device=DEVICE).manual_seed(SEED)
            t0 = time.time()
            image = pipe(
                prompt=prompt_text, negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps, guidance_scale=CFG_SCALE,
                generator=generator, output_type="pil",
            ).images[0]
            latency = time.time() - t0

            generated_images[p_name][steps] = image
            if steps == max(steps_list):
                baseline_images[p_name] = np.array(image)

            results.append({"Prompt_Type": p_name, "Steps": steps, "Latency": latency})
            image.save(os.path.join(IMG_DIR, f"{p_name}_{steps}steps.png"))
            print(f"    Steps={steps:3d} | Latency={latency:.2f}s")

    return results, baseline_images, generated_images


def compute_ssim(results, baseline_images, generated_images):
    """Append SSIM scores to results."""
    for row in results:
        curr = np.array(generated_images[row["Prompt_Type"]][row["Steps"]])
        base = baseline_images[row["Prompt_Type"]]
        score, _ = ssim(base, curr, full=True, channel_axis=-1)
        row["SSIM"] = score
    return results


def plot_results(results):
    df = pd.DataFrame(results)

    # 2x2 individual grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, pname in enumerate(df["Prompt_Type"].unique()):
        if i >= 4: break
        ax1 = axes.flatten()[i]
        ax2 = ax1.twinx()
        sub = df[df["Prompt_Type"] == pname].sort_values("Steps")
        ax1.plot(sub["Steps"], sub["Latency"], "o-", color="tab:red", label="Latency")
        ax2.plot(sub["Steps"], sub["SSIM"], "s-", color="tab:blue", label="SSIM")
        ax2.axhline(0.95, color="gray", linestyle=":", label="0.95 Threshold")
        ax1.set_title(pname); ax1.set_xlabel("Steps")
        ax1.set_ylabel("Latency (s)", color="tab:red")
        ax2.set_ylabel("SSIM", color="tab:blue")
        ax1.grid(alpha=0.4)
    fig.suptitle("Exp 1: Individual Prompt Convergence", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(OUTPUT_DIR, "exp1_individual_prompts.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"[Exp1] Saved: {p}")

    # Average plot
    df_mean = df.groupby("Steps").agg({"Latency": "mean", "SSIM": "mean"}).reset_index()
    print("\n=== Experiment 1 – Average Results ==="); print(df_mean.round(4).to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(df_mean["Steps"], df_mean["Latency"], "o-", color="tab:red"); ax1.set_ylabel("Mean Latency (s)", color="tab:red")
    ax2.plot(df_mean["Steps"], df_mean["SSIM"],    "s-", color="tab:blue"); ax2.set_ylabel("Mean SSIM", color="tab:blue")
    ax2.axhline(0.95, color="gray", linestyle=":", label="0.95 Threshold")
    ax1.set_xlabel("Inference Steps"); plt.title("Exp 1: Average Convergence (SSD-1B / MPS)")
    fig.tight_layout(); plt.grid(alpha=0.4)
    p = os.path.join(OUTPUT_DIR, "exp1_average.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"[Exp1] Saved: {p}")


if __name__ == "__main__":
    STEPS_LIST = [8, 12, 16, 20, 25, 30, 50]
    pipe = load_pipeline()
    results, baseline_images, generated_images = run_experiment(pipe, STEPS_LIST)
    results = compute_ssim(results, baseline_images, generated_images)
    plot_results(results)
    print("\n[Exp1] Done.")
