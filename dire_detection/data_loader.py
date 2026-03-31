"""
data_loader.py - Dataset loader for Part 2: DIRE fake image detection.

Loads real (FFHQ) and SDXL-generated face images from HuggingFace using
streaming mode to avoid full dataset download.

The dataset only has a 'train' split, so we carve out a validation portion
by skipping the first `train_limit` items.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import DATASET_ID, CONFIG_NAME, TRAIN_LIMIT, VAL_LIMIT, BATCH_SIZE


class SDXLFaceDataset(Dataset):
    """
    PyTorch Dataset wrapping the BitMind FFHQ vs SDXL face dataset.

    Args:
        skip:  Number of items to skip at the start (used to carve val set).
        limit: Max number of items to load. None = load all.
    """

    # Known label key candidates — auto-detected from the first item
    _LABEL_CANDIDATES = ["label", "is_fake", "class", "target", "fake"]

    def __init__(self, skip: int = 0, limit: int = None):
        print(f"[DataLoader] Loading dataset: skip={skip}, limit={limit}")

        stream = load_dataset(DATASET_ID, CONFIG_NAME, split="train", streaming=True)

        self.items: list = []
        skipped = count = 0
        for item in stream:
            if skipped < skip:
                skipped += 1
                continue
            if limit is not None and count >= limit:
                break
            self.items.append(item)
            count += 1

        # Auto-detect label key
        self._label_key = None
        if self.items:
            keys = list(self.items[0].keys())
            print(f"[DataLoader] Item keys: {keys}")
            for candidate in self._LABEL_CANDIDATES:
                if candidate in keys:
                    self._label_key = candidate
                    break
            if self._label_key is None:
                print(f"[DataLoader] WARNING: No label key found in {keys}. Defaulting to 0.")

        print(f"[DataLoader] Loaded {len(self.items)} images. Label key: '{self._label_key}'")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["image"].convert("RGB")
        label = int(item[self._label_key]) if self._label_key else 0
        return self.transform(image), label


def get_dataloaders(train_limit=TRAIN_LIMIT, val_limit=VAL_LIMIT, batch_size=BATCH_SIZE):
    """Build train and validation DataLoaders."""
    train_ds = SDXLFaceDataset(skip=0,           limit=train_limit)
    val_ds   = SDXLFaceDataset(skip=train_limit, limit=val_limit)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 60)
    print("DataLoader smoke test")
    print("=" * 60)
    train_loader, val_loader = get_dataloaders(train_limit=10, val_limit=5, batch_size=4)
    for imgs, labels in train_loader:
        print(f"  Train batch -> images: {imgs.shape}, labels: {labels}")
        break
    for imgs, labels in val_loader:
        print(f"  Val   batch -> images: {imgs.shape}, labels: {labels}")
        break
    print("Smoke test PASSED.")
