"""
train_classifier.py - Train a ResNet-18 binary classifier on DIRE error maps.

Workflow:
  1. Load real/fake image pairs from HuggingFace dataset.
  2. For each batch, compute DIRE error maps (latent space).
  3. Feed error maps into modified ResNet-18 (4-channel input).
  4. Train with Binary Cross-Entropy loss.
  5. Validate, plot ROC-AUC, and save the model checkpoint.

Run:
    cd /Users/khang/Downloads/lab02
    source .venv/bin/activate
    python part2_dire_detection/train_classifier.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from config import DEVICE, OUTPUT_DIR, EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAIN_LIMIT, VAL_LIMIT, INVERSION_STEPS
from data_loader import get_dataloaders
from dire_extractor import DIREExtractor
from evaluate_and_plot import plot_roc_curve


def build_classifier(in_channels: int = 4, device: str = DEVICE) -> nn.Module:
    """
    Modified ResNet-18 for DIRE error map classification.

    in_channels=4 because the VAE produces 4-channel latent tensors.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def train():
    print("=" * 60)
    print(f"Part 2: DIRE Classifier Training  |  Device: {DEVICE.upper()}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    train_loader, val_loader = get_dataloaders(
        train_limit=TRAIN_LIMIT, val_limit=VAL_LIMIT, batch_size=BATCH_SIZE
    )

    # ── DIRE Extractor ────────────────────────────────────────────────────────
    print("\n[2/4] Initializing DIRE extractor ...")
    extractor = DIREExtractor()

    # ── Classifier ───────────────────────────────────────────────────────────
    print("\n[3/4] Building ResNet-18 classifier ...")
    classifier = build_classifier(in_channels=4, device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # ── Training Loop ────────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {EPOCHS} epoch(s) ...\n")
    for epoch in range(EPOCHS):
        classifier.train()
        total_loss, num_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            labels = labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)

            dire_maps = extractor.compute_dire_map(images, inversion_steps=INVERSION_STEPS)

            optimizer.zero_grad()
            outputs = classifier(dire_maps.to(torch.float32))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n" + "─" * 60 + "\nValidation ...\n")
    classifier.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            dire_maps = extractor.compute_dire_map(images, inversion_steps=INVERSION_STEPS)
            probs = torch.sigmoid(classifier(dire_maps.to(torch.float32))).cpu().squeeze()
            all_labels.extend(labels.tolist())
            all_scores.extend(probs.tolist() if probs.dim() > 0 else [probs.item()])

    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plot_roc_curve(all_labels, all_scores, save_path=roc_path)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(OUTPUT_DIR, "dire_classifier.pt")
    torch.save(classifier.state_dict(), ckpt_path)
    print(f"\n[Classifier] Model saved: {ckpt_path}")
    print("Training complete!")


if __name__ == "__main__":
    train()
