"""Generate LCX evolution filmstrip from matrix_history.jsonl.

Outputs to lcx_film/ directory:
  - lcx_step_NNN.png     — LCX state at each sampled step
  - delta_NNN_to_MMM.png — amplified change between steps
  - filmstrip.png        — all steps side by side
  - evolution.png         — step 0 | step 499 | delta (big)
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Color anchors (same as frame_renderer)
CYAN = np.array([0, 131, 143], dtype=np.float32)
GRAY = np.array([96, 125, 139], dtype=np.float32)
FUCHSIA = np.array([194, 24, 91], dtype=np.float32)
BG = np.array([30, 30, 30], dtype=np.uint8)


def color_signed(val):
    val = max(-1.0, min(1.0, val))
    if val <= 0:
        t = val + 1.0
        return (1 - t) * CYAN + t * GRAY
    else:
        return (1 - val) * GRAY + val * FUCHSIA


def render_grid(values, side=8, cell_px=48, gap=2):
    arr = np.array(values, dtype=np.float32).flatten()
    n = side * side
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))
    img_size = side * cell_px + (side + 1) * gap
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = BG
    for i in range(n):
        r, c = i // side, i % side
        color = color_signed(arr[i]).astype(np.uint8)
        y0 = gap + r * (cell_px + gap)
        x0 = gap + c * (cell_px + gap)
        img[y0:y0+cell_px, x0:x0+cell_px] = color
    return Image.fromarray(img)


def add_label(img, text, font_size=20):
    """Add a text label below the image."""
    w, h = img.size
    label_h = font_size + 12
    new = Image.new('RGB', (w, h + label_h), (30, 30, 30))
    new.paste(img, (0, 0))
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("consola.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, h + 4), text, fill=(200, 200, 200), font=font)
    return new


def main():
    history_path = r"S:\AI\work\VRAXION_DEV\Diamond Code\logs\swarm\matrix_history.jsonl"
    out_dir = r"S:\AI\work\VRAXION_DEV\Diamond Code\lcx_film"
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    with open(history_path) as f:
        for line in f:
            entries.append(json.loads(line))

    total = len(entries)
    print(f"Loaded {total} steps from matrix_history.jsonl")

    # Sample steps for filmstrip
    sample_steps = [0, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, total - 1]
    sample_steps = [s for s in sample_steps if s < total]

    cell_px = 48
    gap = 2

    # 1. Individual step images
    for s in sample_steps:
        e = entries[s]
        img = render_grid(e['lcx_after'], cell_px=cell_px, gap=gap)
        img = add_label(img, f"Step {e['step']}  norm={e['lcx_norm']:.3f}")
        img.save(os.path.join(out_dir, f"lcx_step_{e['step']:04d}.png"))
        print(f"  Saved lcx_step_{e['step']:04d}.png")

    # 2. Delta images (vs step 0)
    lcx_0 = np.array(entries[0]['lcx_after'])
    for s in sample_steps[1:]:
        lcx_s = np.array(entries[s]['lcx_after'])
        delta = (lcx_s - lcx_0) * 3.0  # 3x amplification
        img = render_grid(delta.tolist(), cell_px=cell_px, gap=gap)
        img = add_label(img, f"Delta 0->{entries[s]['step']} (3x)")
        img.save(os.path.join(out_dir, f"delta_000_to_{entries[s]['step']:04d}.png"))

    # 3. Filmstrip — all sampled steps in a row
    strip_imgs = []
    for s in sample_steps:
        e = entries[s]
        img = render_grid(e['lcx_after'], cell_px=32, gap=1)
        img = add_label(img, f"s{e['step']}", font_size=14)
        strip_imgs.append(img)

    strip_w = sum(im.width for im in strip_imgs) + (len(strip_imgs) - 1) * 4
    strip_h = max(im.height for im in strip_imgs)
    filmstrip = Image.new('RGB', (strip_w, strip_h), (30, 30, 30))
    x = 0
    for im in strip_imgs:
        filmstrip.paste(im, (x, 0))
        x += im.width + 4
    filmstrip.save(os.path.join(out_dir, "filmstrip.png"))
    print(f"  Saved filmstrip.png ({strip_w}x{strip_h})")

    # 4. Evolution poster — step 0 | step 499 | delta (big, labeled)
    big_px = 64
    lcx_last = np.array(entries[-1]['lcx_after'])
    delta_full = lcx_last - lcx_0

    img_0 = render_grid(lcx_0.tolist(), cell_px=big_px, gap=3)
    img_0 = add_label(img_0, f"Step 0  (norm {entries[0]['lcx_norm']:.3f})", font_size=22)

    img_end = render_grid(lcx_last.tolist(), cell_px=big_px, gap=3)
    img_end = add_label(img_end, f"Step {entries[-1]['step']}  (norm {entries[-1]['lcx_norm']:.3f})", font_size=22)

    img_delta = render_grid(delta_full.tolist(), cell_px=big_px, gap=3)
    img_delta = add_label(img_delta, f"Delta (raw, max={np.abs(delta_full).max():.2f})", font_size=22)

    poster_gap = 20
    pw = img_0.width * 3 + poster_gap * 4
    ph = max(img_0.height, img_end.height, img_delta.height) + 50
    poster = Image.new('RGB', (pw, ph), (24, 27, 31))
    draw = ImageDraw.Draw(poster)
    try:
        font = ImageFont.truetype("consola.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((pw // 2 - 200, 8), "LCX Evolution (500 steps)", fill=(200, 200, 200), font=font)

    y_off = 42
    poster.paste(img_0, (poster_gap, y_off))
    poster.paste(img_end, (poster_gap * 2 + img_0.width, y_off))
    poster.paste(img_delta, (poster_gap * 3 + img_0.width * 2, y_off))

    poster.save(os.path.join(out_dir, "evolution.png"))
    print(f"  Saved evolution.png ({pw}x{ph})")

    print(f"\nAll images in: {out_dir}")


if __name__ == "__main__":
    main()
