"""
dire_extractor.py - Core DIRE (Diffusion Reconstruction Error) algorithm.

Algorithm steps:
  1. Encode image → VAE latent space.
  2. Add noise via DDIM forward process (Inversion).
  3. Denoise via DDIM reverse process (Reconstruction).
  4. Compute |original_latents - reconstructed_latents| as the error map.

Supports Apple Silicon MPS and CPU.
Uses the local SSD-1B-Traditional model (shared with Part 1).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel

from config import MODEL_PATH, DEVICE, INVERSION_STEPS


class DIREExtractor:
    """Extracts DIRE error maps from a batch of images using a local diffusion model."""

    def __init__(self, model_id: str = MODEL_PATH, device: str = DEVICE):
        print(f"[DIRE] Initializing on [{device.upper()}]")
        print(f"[DIRE] Loading from:  {model_id}")
        self.device = torch.device(device)
        self.dtype  = torch.float32   # float32 is safer for MPS

        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)
        self.unet.eval()

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(50)

        print(f"[DIRE] Ready. VAE scaling factor: {self.vae.config.scaling_factor}")

    @torch.no_grad()
    def compute_dire_map(self, pixel_images: torch.Tensor, inversion_steps: int = INVERSION_STEPS) -> torch.Tensor:
        """
        Compute the DIRE error map for a batch of images.

        Args:
            pixel_images:    (B, 3, H, W) tensor normalised to [-1, 1].
            inversion_steps: Number of DDIM timesteps for inversion/reconstruction.

        Returns:
            error_map: (B, C_lat, H_lat, W_lat) absolute reconstruction error.
        """
        images = pixel_images.to(self.device, dtype=self.dtype)

        # Step 1 – Encode to latent space
        original_latents = self.vae.encode(images).latent_dist.sample()
        original_latents = original_latents * self.vae.config.scaling_factor

        # Step 2 – DDIM forward: add noise at timestep t
        t_idx = min(inversion_steps, len(self.scheduler.timesteps) - 1)
        t     = self.scheduler.timesteps[-t_idx]
        noise = torch.randn_like(original_latents)
        noisy = self.scheduler.add_noise(original_latents, noise, torch.tensor([t], device=self.device))

        # Step 3 – DDIM reverse: denoise step by step (unconditional)
        cross_attn_dim = self.unet.config.cross_attention_dim
        encoder_hs = torch.zeros(
            (pixel_images.shape[0], 1, cross_attn_dim),
            device=self.device, dtype=self.dtype
        )
        reconstructed = noisy.clone()
        for timestep in self.scheduler.timesteps[-t_idx:]:
            noise_pred = self.unet(reconstructed, timestep, encoder_hidden_states=encoder_hs).sample
            reconstructed = self.scheduler.step(noise_pred, timestep, reconstructed).prev_sample

        # Step 4 – DIRE = |original - reconstructed|
        return torch.abs(original_latents - reconstructed)


if __name__ == "__main__":
    print("=" * 60)
    print("DIRE Extractor smoke test")
    print("=" * 60)
    extractor = DIREExtractor()
    dummy = torch.randn(2, 3, 256, 256)
    error_map = extractor.compute_dire_map(dummy, inversion_steps=10)
    print(f"  Input:     {dummy.shape}")
    print(f"  Error map: {error_map.shape}")
    print(f"  Range:     [{error_map.min():.4f}, {error_map.max():.4f}]")
    print("Smoke test PASSED.")
