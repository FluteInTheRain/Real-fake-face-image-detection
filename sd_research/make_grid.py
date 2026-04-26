"""
make_grid.py - Stitch experiment output images into a single documentation-ready grid.

Usage:
    cd /Users/khang/Downloads/lab02
    source .venv/bin/activate

    # Auto-detect and build grids for all experiments:
    python sd_research/make_grid.py

    # Specific experiment only:
    python sd_research/make_grid.py --exp 1
    python sd_research/make_grid.py --exp 2

Output:
    sd_research/outputs/grid_exp1_steps.png
    sd_research/outputs/grid_exp2_cfg.png
"""

import os, sys, re, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import OUTPUT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
THUMB_W = 256  # width of each thumbnail in the grid
THUMB_H = 256  # height
CAPTION_H = 40  # height of caption bar below each image
HEADER_H = 52  # height of the row header (prompt label) on the left
PAD = 10  # padding between cells
FONT_SIZE = 18  # caption font size
HEADER_W = 140  # width of the left-hand prompt label column
BG_COLOR = (245, 245, 248)  # light grey background
HEADER_BG = (50, 50, 70)  # dark header background
HEADER_COLOR = (255, 255, 255)  # white text on header
CAPTION_BG = (230, 230, 235)  # caption bar background
CAPTION_COLOR = (30, 30, 30)  # caption text color
TITLE_H = 64  # title bar height at top


def _load_font(size):
    """Try to load a built-in system font; fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_text_centered(draw, text, x, y, w, h, font, color):
    """Draw text centered in a bounding box (x, y, x+w, y+h)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, font=font, fill=color)


def build_grid(
    image_dir: str,
    groups: dict,  # { "PromptName": [(label, filepath), ...] }
    title: str,
    output_path: str,
    col_header: str = "Parameter",
):
    """
    Build and save a documentation grid image.

    Each row = one prompt.
    Each column = one parameter value (steps / CFG / etc.).
    Cells = thumbnail + caption.
    """
    font_caption = _load_font(FONT_SIZE - 2)
    font_header = _load_font(FONT_SIZE)
    font_title = _load_font(FONT_SIZE + 4)

    prompt_names = list(groups.keys())
    n_rows = len(prompt_names)
    n_cols = max(len(v) for v in groups.values())

    cell_w = THUMB_W + PAD
    cell_h = THUMB_H + CAPTION_H + PAD

    total_w = HEADER_W + n_cols * cell_w + PAD
    total_h = TITLE_H + n_rows * cell_h + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # ── Title bar ────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, total_w, TITLE_H], fill=HEADER_BG)
    _draw_text_centered(draw, title, 0, 0, total_w, TITLE_H, font_title, HEADER_COLOR)

    # ── Column headers (first row above images) ───────────────────────────────
    # (derived from the first group's labels)
    sample_labels = [label for label, _ in list(groups.values())[0]]
    for c, label in enumerate(sample_labels):
        x = HEADER_W + c * cell_w + PAD
        draw.rectangle(
            [x, TITLE_H, x + THUMB_W, TITLE_H + CAPTION_H - 4], fill=HEADER_BG
        )
        _draw_text_centered(
            draw,
            str(label),
            x,
            TITLE_H,
            THUMB_W,
            CAPTION_H - 4,
            font_caption,
            HEADER_COLOR,
        )

    # ── Rows ─────────────────────────────────────────────────────────────────
    y_offset = TITLE_H + CAPTION_H
    for row_idx, p_name in enumerate(prompt_names):
        y = y_offset + row_idx * cell_h

        # Left header: prompt name
        draw.rectangle([0, y, HEADER_W - PAD, y + cell_h - PAD], fill=HEADER_BG)
        _draw_text_centered(
            draw, p_name, 0, y, HEADER_W - PAD, cell_h - PAD, font_header, HEADER_COLOR
        )

        # Cells
        for col_idx, (label, fpath) in enumerate(groups[p_name]):
            x = HEADER_W + col_idx * cell_w + PAD
            cx = x
            cy = y + PAD

            if fpath and os.path.exists(fpath):
                img = (
                    Image.open(fpath)
                    .convert("RGB")
                    .resize((THUMB_W, THUMB_H), Image.LANCZOS)
                )
            else:
                # Placeholder if image missing
                img = Image.new("RGB", (THUMB_W, THUMB_H), (180, 180, 200))
                d = ImageDraw.Draw(img)
                d.text(
                    (10, THUMB_H // 2 - 10),
                    "N/A",
                    fill=(100, 100, 100),
                    font=font_caption,
                )

            canvas.paste(img, (cx, cy))

            # Caption bar
            cap_y = cy + THUMB_H
            draw.rectangle(
                [cx, cap_y, cx + THUMB_W, cap_y + CAPTION_H - PAD], fill=CAPTION_BG
            )
            _draw_text_centered(
                draw,
                str(label),
                cx,
                cap_y,
                THUMB_W,
                CAPTION_H - PAD,
                font_caption,
                CAPTION_COLOR,
            )

    canvas.save(output_path, dpi=(300, 300))
    print(f"[Grid] Saved: {output_path}  ({total_w}×{total_h} px)")
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Experiment-specific parsers
# ─────────────────────────────────────────────────────────────────────────────


def build_exp1_grid():
    """Group exp1 images by prompt, ordered by step count."""
    img_dir = os.path.join(OUTPUT_DIR, "exp1_generated_images")
    if not os.path.isdir(img_dir):
        print("[Grid] exp1_generated_images/ not found. Run Experiment 1 first.")
        return

    # Discover files: {prompt_name: [(steps, filepath), ...]}
    pattern = re.compile(r"^(.+?)_(\d+)steps\.png$")
    groups_raw = {}
    for fname in sorted(os.listdir(img_dir)):
        m = pattern.match(fname)
        if m:
            p_name, steps = m.group(1), int(m.group(2))
            groups_raw.setdefault(p_name, []).append(
                (steps, os.path.join(img_dir, fname))
            )

    # Sort each prompt's entries by step count
    groups = {
        p: [(f"{s} steps", fp) for s, fp in sorted(entries)]
        for p, entries in sorted(groups_raw.items())
    }

    out = os.path.join(OUTPUT_DIR, "grid_exp1_steps.png")
    build_grid(
        image_dir=img_dir,
        groups=groups,
        title="Experiment 1: Inference Steps vs Output Quality (SSD-1B / MPS)",
        output_path=out,
        col_header="Steps",
    )


def build_exp2_grid():
    """Group exp2 images by prompt, ordered by CFG scale."""
    img_dir = os.path.join(OUTPUT_DIR, "exp2_generated_images")
    if not os.path.isdir(img_dir):
        print("[Grid] exp2_generated_images/ not found. Run Experiment 2 first.")
        return

    # Discover files: {prompt_name: [(cfg_float, filepath), ...]}
    pattern = re.compile(r"^(.+?)_cfg([\d.]+)\.png$")
    groups_raw = {}
    for fname in sorted(os.listdir(img_dir)):
        m = pattern.match(fname)
        if m:
            p_name, cfg = m.group(1), float(m.group(2))
            groups_raw.setdefault(p_name, []).append(
                (cfg, os.path.join(img_dir, fname))
            )

    groups = {
        p: [(f"CFG {c:.1f}", fp) for c, fp in sorted(entries)]
        for p, entries in sorted(groups_raw.items())
    }

    out = os.path.join(OUTPUT_DIR, "grid_exp2_cfg.png")
    build_grid(
        image_dir=img_dir,
        groups=groups,
        title="Experiment 2: CFG Scale Sensitivity — Prompt Adherence vs Saturation (SSD-1B / MPS)",
        output_path=out,
        col_header="CFG Scale",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch experiment images into a grid for docs."
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2],
        default=None,
        help="Which experiment to grid (1 or 2). Default: both.",
    )
    args = parser.parse_args()

    if args.exp is None or args.exp == 1:
        build_exp1_grid()
    if args.exp is None or args.exp == 2:
        build_exp2_grid()
