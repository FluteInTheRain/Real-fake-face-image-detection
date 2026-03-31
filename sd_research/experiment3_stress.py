"""
experiment3_stress.py - Experiment 3: Stress Test (Batch Size & Memory).

Research Question:
    What is the maximum stable batch size for SSD-1B on Apple Silicon (MPS /
    Unified Memory)?  How do real-time latent callbacks affect memory and
    throughput?

    Note: MPS does not expose VRAM counters the way CUDA does.
    We use process RSS memory (via psutil) as a unified-memory proxy.

Output:
    outputs/exp3_stress_test.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

from config import MODEL_PATH, OUTPUT_DIR, DEVICE

try:
    import psutil
    _PROCESS = psutil.Process(os.getpid())
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("[Exp3] Warning: psutil not found. RAM tracking disabled. pip install psutil")


def _ram_gb():
    if HAVE_PSUTIL:
        return _PROCESS.memory_info().rss / (1024 ** 3)
    return 0.0


def load_pipeline():
    print(f"[Exp3] Loading pipeline from: {MODEL_PATH}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # float32 required on MPS to avoid NaN
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.to(torch.float32)
    pipe.vae.enable_slicing()
    return pipe


def run_stress_test(pipe, batch_sizes, callback_freqs, prompt, negative_prompt, steps=20, cfg=7.0):
    results = []
    print("=" * 60)
    print("EXPERIMENT 3: Stress Test – Batch Size & Callbacks")
    print("=" * 60)

    for batch in batch_sizes:
        for freq in callback_freqs:
            freq_label = "None" if freq == 0 else f"Every {freq} steps"
            print(f"\n  Batch={batch} | Callback={freq_label}")

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            def stress_callback(pipe, step_index, timestep, callback_kwargs):
                if freq > 0 and step_index % freq == 0:
                    _ = callback_kwargs["latents"].mean().item()
                return callback_kwargs

            try:
                ram_before = _ram_gb()
                t0 = time.time()

                _ = pipe(
                    prompt=[prompt] * batch,
                    negative_prompt=[negative_prompt] * batch,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    callback_on_step_end=stress_callback if freq > 0 else None,
                    callback_on_step_end_tensor_inputs=["latents"] if freq > 0 else [],
                ).images

                duration = time.time() - t0
                ram_delta = _ram_gb() - ram_before
                throughput = batch / duration

                results.append({
                    "Batch_Size": batch, "Callback_Freq": freq_label,
                    "Status": "✅ Success",
                    "RAM_Delta_GB": round(ram_delta, 2),
                    "Throughput_img_s": round(throughput, 4),
                })
                print(f"  -> ✅ | RAM+{ram_delta:.2f}GB | {throughput:.4f} img/s")

            except Exception as e:
                oom = "out of memory" in str(e).lower() or "oom" in str(e).lower()
                status = "❌ OOM" if oom else f"❌ Error"
                print(f"  -> {status}: {e}")
                results.append({
                    "Batch_Size": batch, "Callback_Freq": freq_label,
                    "Status": status,
                    "RAM_Delta_GB": float("nan"),
                    "Throughput_img_s": 0.0,
                })
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    return results


def plot_results(results):
    df = pd.DataFrame(results)
    print("\n=== Experiment 3 – Stress Test Summary ===")
    print(df.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label in df["Callback_Freq"].unique():
        sub = df[df["Callback_Freq"] == label]
        ax1.plot(sub["Batch_Size"], sub["RAM_Delta_GB"], marker="o", label=label)
        ax2.plot(sub["Batch_Size"], sub["Throughput_img_s"], marker="s", linestyle="--", label=label)

    ax1.set_title("RAM Usage Delta (GB) — MPS Unified Memory")
    ax1.set_xlabel("Batch Size"); ax1.set_ylabel("RAM Δ (GB)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title("System Throughput (Images/Sec)")
    ax2.set_xlabel("Batch Size"); ax2.set_ylabel("Img/s")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Exp 3: Stress Test – Batch Size & Callback Overhead (SSD-1B / MPS)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "exp3_stress_test.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Exp3] Saved: {p}")


if __name__ == "__main__":
    pipe = load_pipeline()
    stress_results = run_stress_test(
        pipe,
        batch_sizes=[1, 2, 4],      # Keep conservative for 16GB Mac RAM
        callback_freqs=[0, 5, 1],
        prompt="A cinematic shot of a futuristic sports car, high resolution.",
        negative_prompt="blurry, low quality, artifacts",
        steps=20, cfg=7.0,
    )
    plot_results(stress_results)
    print("\n[Exp3] Done.")
