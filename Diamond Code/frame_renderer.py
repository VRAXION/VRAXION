"""
Render LCX frames as PNG images for Grafana live display.
Writes to live/ directory, served by HTTP server on :8088.

Colors: dark cyan (#00838F) -> blue-gray (#607D8B) -> warm fuchsia (#C2185B)
"""

import os
import numpy as np
from PIL import Image

# Color anchors
CYAN = np.array([0, 131, 143], dtype=np.float32)      # -1 or 0
GRAY = np.array([96, 125, 139], dtype=np.float32)      # 0 (midpoint for signed)
FUCHSIA = np.array([194, 24, 91], dtype=np.float32)    # +1

LIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live")


def _ensure_dir():
    os.makedirs(LIVE_DIR, exist_ok=True)


def _color_signed(val):
    """Map [-1, 1] -> cyan-gray-fuchsia."""
    val = max(-1.0, min(1.0, val))
    if val <= 0:
        t = (val + 1.0)  # 0..1 maps cyan..gray
        return (1 - t) * CYAN + t * GRAY
    else:
        return (1 - val) * GRAY + val * FUCHSIA


def _color_unsigned(val):
    """Map [0, 1] -> cyan-fuchsia (no gray midpoint)."""
    val = max(0.0, min(1.0, val))
    return (1 - val) * CYAN + val * FUCHSIA


def render_grid(values, filepath, side=8, cell_px=48, signed=True, gap=2):
    """Render side x side grid as PNG with cell gaps."""
    _ensure_dir()
    arr = np.array(values, dtype=np.float32).flatten()
    n = side * side
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))

    img_size = side * cell_px + (side + 1) * gap
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]  # dark background for gaps

    color_fn = _color_signed if signed else _color_unsigned
    for i in range(n):
        r, c = i // side, i % side
        color = color_fn(arr[i]).astype(np.uint8)
        y0 = gap + r * (cell_px + gap)
        x0 = gap + c * (cell_px + gap)
        img[y0:y0+cell_px, x0:x0+cell_px] = color

    Image.fromarray(img).save(os.path.join(LIVE_DIR, filepath))


def render_strip(values, filepath, cell_px=48, gap=2, signed=False):
    """Render 1 x N strip as PNG."""
    _ensure_dir()
    arr = np.array(values, dtype=np.float32).flatten()
    n = len(arr)

    w = n * cell_px + (n + 1) * gap
    h = cell_px + 2 * gap
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]

    color_fn = _color_signed if signed else _color_unsigned
    for i, val in enumerate(arr):
        color = color_fn(val).astype(np.uint8)
        x0 = gap + i * (cell_px + gap)
        img[gap:gap+cell_px, x0:x0+cell_px] = color

    Image.fromarray(img).save(os.path.join(LIVE_DIR, filepath))


def render_mask_matrix(mask_dict, num_bits, filepath="mask_matrix.png", cell_px=48, gap=2):
    """Render beings x bits mask matrix as PNG.
    mask_dict: {being_id: [bit_indices]} e.g. {0: [1,3,4,6], 1: [0,2,5,7]}
    """
    _ensure_dir()
    n_beings = len(mask_dict)
    rows = n_beings
    cols = num_bits

    w = cols * cell_px + (cols + 1) * gap
    h = rows * cell_px + (rows + 1) * gap
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]

    OFF = np.array([40, 50, 55], dtype=np.uint8)  # dark, not assigned

    for being_id in range(n_beings):
        bits = set(mask_dict.get(being_id, []))
        for bit in range(num_bits):
            color = FUCHSIA.astype(np.uint8) if bit in bits else OFF
            y0 = gap + being_id * (cell_px + gap)
            x0 = gap + bit * (cell_px + gap)
            img[y0:y0+cell_px, x0:x0+cell_px] = color

    Image.fromarray(img).save(os.path.join(LIVE_DIR, filepath))


def render_delta(before, after, filepath, side=8, cell_px=48, gap=2, amplify=10.0):
    """Render amplified (after - before) delta grid. Shows what changed this step."""
    _ensure_dir()
    b = np.array(before, dtype=np.float32).flatten()
    a = np.array(after, dtype=np.float32).flatten()
    n = side * side
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))

    delta = (a - b) * amplify  # amplify tiny changes

    img_size = side * cell_px + (side + 1) * gap
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]

    for i in range(n):
        r, c = i // side, i % side
        color = _color_signed(delta[i]).astype(np.uint8)
        y0 = gap + r * (cell_px + gap)
        x0 = gap + c * (cell_px + gap)
        img[y0:y0+cell_px, x0:x0+cell_px] = color

    Image.fromarray(img).save(os.path.join(LIVE_DIR, filepath))


def render_rect(values, filepath, rows, cols, cell_px=48, gap=2, signed=False):
    """Render rows x cols rectangular grid as PNG."""
    _ensure_dir()
    arr = np.array(values, dtype=np.float32).flatten()
    n = rows * cols
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))

    w = cols * cell_px + (cols + 1) * gap
    h = rows * cell_px + (rows + 1) * gap
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]

    color_fn = _color_signed if signed else _color_unsigned
    for i in range(n):
        r, c = i // cols, i % cols
        color = color_fn(arr[i]).astype(np.uint8)
        y0 = gap + r * (cell_px + gap)
        x0 = gap + c * (cell_px + gap)
        img[y0:y0+cell_px, x0:x0+cell_px] = color

    Image.fromarray(img).save(os.path.join(LIVE_DIR, filepath))


def render_all(input_vals, output_vals, lcx_before, lcx_after, num_bits=8):
    """Render all 5 frame PNGs in one call."""
    render_grid(input_vals, "input.png", side=num_bits, signed=False)
    render_grid(output_vals, "output.png", side=num_bits, signed=True)
    render_grid(lcx_before, "lcx_before.png", side=num_bits, signed=True)
    render_grid(lcx_after, "lcx_after.png", side=num_bits, signed=True)
    render_delta(lcx_before, lcx_after, "lcx_delta.png", side=num_bits, amplify=10.0)
