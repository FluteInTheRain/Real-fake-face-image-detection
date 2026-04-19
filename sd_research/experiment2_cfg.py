"""
experiment2_cfg.py - Experiment 2: CFG Scale Sensitivity.

Research Question:
    How does the Classifier-Free Guidance (CFG) scale affect prompt adherence
    (CLIP Score) and over-saturation risk (HSV saturation channel) in SSD-1B?

Output:
    outputs/exp2_individual_prompts.png
    outputs/exp2_average.png
    outputs/exp2_generated_images/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPModel, CLIPProcessor

from config import MODEL_PATH, OUTPUT_DIR, DEVICE, SEED, PROMPTS, NEGATIVE_PROMPT

IMG_DIR = os.path.join(OUTPUT_DIR, "exp2_generated_images")

# CFG Scales to test (same prompts as Experiment 1)
CFG_SCALES = [3.0, 5.0, 7.0, 9.0, 12.0, 15.0]


def load_pipeline():
    print(f"[Exp2] Loading pipeline from: {MODEL_PATH}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # float32 required on MPS to avoid NaN
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.to(torch.float32)
    return pipe


def load_clip():
    print("[Exp2] Loading CLIP (ViT-L/14)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def run_experiment(pipe, clip_model, clip_processor, cfg_scales=CFG_SCALES, inference_steps=20):
    os.makedirs(IMG_DIR, exist_ok=True)
    results = []
    # Use same prompt order as Experiment 1 (Macro, Portrait, Cyberpunk, Vector)
    for p_name, prompt_text in PROMPTS.items():
        print(f"\n  Prompt: {p_name}")
        for cfg in cfg_scales:
            generator = torch.Generator(device=DEVICE).manual_seed(SEED)
            image = pipe(
                prompt=prompt_text, negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=inference_steps, guidance_scale=cfg,
                generator=generator, output_type="pil",
            ).images[0]

            # CLIP Score
            inputs = clip_processor(text=[prompt_text], images=image, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                clip_score = clip_model(**inputs).logits_per_image.item()

            # Saturation
            hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
            sat = float(np.mean(hsv[:, :, 1]))

            results.append({"Prompt_Type": p_name, "CFG_Scale": cfg, "CLIP_Score": round(clip_score, 2), "Mean_Saturation": round(sat, 2)})
            image.save(os.path.join(IMG_DIR, f"{p_name}_cfg{cfg}.png"))
            print(f"    CFG={cfg:>4} | CLIP={clip_score:.2f} | Sat={sat:.2f}")
            gc.collect()

    return results


def plot_results(results):
    df = pd.DataFrame(results)

    # 2x2 individual
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, pname in enumerate(df["Prompt_Type"].unique()):
        if i >= 4: break
        ax1 = axes.flatten()[i]; ax2 = ax1.twinx()
        sub = df[df["Prompt_Type"] == pname].sort_values("CFG_Scale")
        ax1.plot(sub["CFG_Scale"], sub["CLIP_Score"],      "o-", color="tab:blue", label="CLIP Score")
        ax2.plot(sub["CFG_Scale"], sub["Mean_Saturation"], "s--", color="tab:red",  label="Saturation")
        ax2.axhline(150, color="gray", linestyle=":", label="Saturation Warning")
        ax1.set_title(pname); ax1.set_xlabel("CFG Scale")
        ax1.set_ylabel("CLIP Score", color="tab:blue")
        ax2.set_ylabel("Mean Saturation", color="tab:red")
        ax1.grid(alpha=0.4)
    fig.suptitle("Exp 2: CFG Scale – Prompt Adherence vs Saturation", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(OUTPUT_DIR, "exp2_individual_prompts.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"[Exp2] Saved: {p}")

    # Average
    df_mean = df.groupby("CFG_Scale").agg({"CLIP_Score": "mean", "Mean_Saturation": "mean"}).reset_index()
    print("\n=== Experiment 2 – Average Results by CFG ==="); print(df_mean.to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(10, 6)); ax2 = ax1.twinx()
    ax1.plot(df_mean["CFG_Scale"], df_mean["CLIP_Score"],      "o-", color="tab:blue"); ax1.set_ylabel("Mean CLIP Score", color="tab:blue")
    ax2.plot(df_mean["CFG_Scale"], df_mean["Mean_Saturation"], "s--", color="tab:red");  ax2.set_ylabel("Mean Saturation", color="tab:red")
    ax2.axhline(150, color="gray", linestyle=":", label="Saturation Warning")
    ax1.set_xlabel("CFG Scale"); plt.title("Exp 2: Trade-off Adherence vs Saturation (SSD-1B / MPS)")
    fig.tight_layout(); plt.grid(alpha=0.4)
    p = os.path.join(OUTPUT_DIR, "exp2_average.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"[Exp2] Saved: {p}")


if __name__ == "__main__":
    pipe = load_pipeline()
    clip_model, clip_processor = load_clip()
    results = run_experiment(pipe, clip_model, clip_processor, inference_steps=20)
    plot_results(results)
    print("\n[Exp2] Done.")
