"""Variable-precision deploy simulation.

Given each cell's slack (measured), assign the minimum bits needed.
Compute total deploy size INCLUDING the metadata overhead (which cell gets
which precision class).

Precision classes:
  class A (4-bit) : slack > 0.05   (8 values: ±0.05, ±0.1, ...)
  class B (6-bit) : slack > 0.01   (64 values: step ~0.02)
  class C (8-bit) : slack > 0.003  (256 values: step ~0.006)
  class D (10-bit): slack > 0.001  (1024 values: step ~0.0015)
  class E (12-bit): slack > 0.0003 (4096 values: step ~0.0004)
  class F (16-bit): slack <= 0.0003 (full fp16)

Encoding: per-cell 3-bit class tag (6 classes) + packed class-bit value.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SLACK = Path("output/merger_single_w_slack/slack.npy")
CHAMPION = Path("output/merger_single_w_fp16_all/final_fp16.json")


def main():
    slack = np.load(SLACK)
    print(f"Slack shape: {slack.shape}, cells: {slack.size}")
    print(f"Slack: min={slack.min():.6f}, median={np.median(slack):.6f}, max={slack.max():.6f}")

    thresholds = [
        ("A (4-bit)",  0.05,  4),
        ("B (6-bit)",  0.01,  6),
        ("C (8-bit)",  0.003, 8),
        ("D (10-bit)", 0.001, 10),
        ("E (12-bit)", 0.0003, 12),
        ("F (16-bit)", 0.0,   16),
    ]

    # Assign each cell to the most aggressive class where its slack >= threshold
    flat = slack.flatten()
    class_idx = np.full(flat.size, len(thresholds) - 1, dtype=int)  # default: F
    # Iterate from most aggressive to least
    for ci, (name, thr, bits) in enumerate(thresholds):
        mask = flat >= thr
        # Only assign if not already assigned to a more aggressive class
        not_yet = class_idx == len(thresholds) - 1
        if ci == len(thresholds) - 1:
            pass  # F is default, don't overwrite
        else:
            # Only the first pass assigns
            newly = mask & not_yet
            class_idx[newly] = ci

    # Actually, let me redo this more carefully:
    class_idx = np.full(flat.size, -1, dtype=int)
    for ci, (name, thr, bits) in enumerate(thresholds):
        if ci == len(thresholds) - 1:
            # F: catch everything unassigned
            unassigned = class_idx == -1
            class_idx[unassigned] = ci
        else:
            # Assign to this class if slack >= threshold AND not yet assigned
            mask = (flat >= thr) & (class_idx == -1)
            class_idx[mask] = ci

    # Report class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    total_bits = 0
    for ci, (name, thr, bits) in enumerate(thresholds):
        count = (class_idx == ci).sum()
        cl_bits = count * bits
        total_bits += cl_bits
        print(f"  {name:12s} slack>={thr:.4f}: {count:5d} cells ({100*count/flat.size:5.1f}%)  total_bits={cl_bits}")

    # Metadata: 3-bit class tag per cell
    metadata_bits = 3 * flat.size
    total_bytes = int(np.ceil((total_bits + metadata_bits) / 8))

    # Plus per-class alpha (4 B * 6 classes) if linear int, or per-class fp16 lookup
    alpha_bytes = 4 * (len(thresholds) - 1)  # 5 alphas; class F has no alpha (fp16 direct)

    print(f"\n=== DEPLOY SIZE (variable precision) ===")
    print(f"  Data bits       : {total_bits}")
    print(f"  Metadata bits   : {metadata_bits} (3 bit/cell class tag)")
    print(f"  Class alphas    : {alpha_bytes} B")
    print(f"  Total W         : {total_bytes + alpha_bytes} B")
    print(f"  vs pure fp16 W  : 5184 B")
    print(f"  delta           : {(total_bytes + alpha_bytes) - 5184:+d} B")

    # Alternative: 2-bit class tag if we reduce to 4 classes
    print(f"\n=== ALTERNATIVE: 4 classes only (2-bit tag) ===")
    thresholds4 = [
        ("A (4-bit)",  0.05,  4),
        ("B (8-bit)",  0.003, 8),
        ("C (12-bit)", 0.0003, 12),
        ("D (16-bit)", 0.0,   16),
    ]
    class_idx4 = np.full(flat.size, -1, dtype=int)
    for ci, (name, thr, bits) in enumerate(thresholds4):
        if ci == len(thresholds4) - 1:
            class_idx4[class_idx4 == -1] = ci
        else:
            mask = (flat >= thr) & (class_idx4 == -1)
            class_idx4[mask] = ci

    total_bits4 = 0
    for ci, (name, thr, bits) in enumerate(thresholds4):
        count = (class_idx4 == ci).sum()
        total_bits4 += count * bits
        print(f"  {name:12s} slack>={thr:.4f}: {count:5d} cells ({100*count/flat.size:5.1f}%)")

    metadata4 = 2 * flat.size
    total4 = int(np.ceil((total_bits4 + metadata4) / 8)) + 4 * (len(thresholds4) - 1)
    print(f"  Total W: {total4} B (vs 5184 B pure fp16, delta {total4-5184:+d} B)")

    # Simpler still: just 2 classes — fp16 + one smaller
    print(f"\n=== SIMPLEST: 2 classes (1-bit tag: loose vs tight) ===")
    # Loose: slack > 0.003 (can go to 8 bit). Tight: the rest (stay fp16).
    for thr_loose, loose_bits in [(0.05, 4), (0.01, 6), (0.003, 8), (0.001, 10)]:
        loose = flat >= thr_loose
        n_loose = loose.sum()
        n_tight = flat.size - n_loose
        data_bits = n_loose * loose_bits + n_tight * 16
        meta_bits = flat.size  # 1 bit/cell
        alpha_b = 4  # just one alpha for the loose class
        total = int(np.ceil((data_bits + meta_bits) / 8)) + alpha_b
        print(f"  loose={loose_bits}bit (slack>={thr_loose:.3f}): {n_loose} cells loose, {n_tight} cells tight => {total} B (delta {total-5184:+d} B)")

    # Most aggressive: identify cells that could be stored with just a bit each (sign only)
    print(f"\n=== EXTREME: 1-bit sign cells ===")
    # Cells where slack is so large that sign alone suffices
    # If slack > |value|/2, we can ignore the value magnitude entirely — just sign matters
    # Load W
    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W_uint16 = np.array(m["W_fp16"], dtype=np.uint16)
    W_fp16 = W_uint16.view(np.float16).astype(np.float32).flatten()
    abs_W = np.abs(W_fp16)
    # A cell can be 1-bit (sign only) if slack > |value|  (can fall anywhere between 0 and |value| -> 0 is fine, just need sign preserved)
    # Actually simpler: if we snap to 0 or some "large" representative value
    # For this test: slack > |value|/2
    one_bit_mask = slack.flatten() > abs_W / 2
    n_one_bit = one_bit_mask.sum()
    print(f"  Cells where slack > |W|/2: {n_one_bit} ({100*n_one_bit/flat.size:.1f}%)")


if __name__ == "__main__":
    main()
