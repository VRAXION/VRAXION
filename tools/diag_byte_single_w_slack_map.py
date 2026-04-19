"""Measure per-cell 'slack' — how much can each cell be perturbed before
lossless breaks?

For each cell (i,j):
  Binary search for max |delta| such that lossless stays 100%.
  Positive and negative directions separately; slack = min of both.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import (
    SingleWMirror, load_byte_pairs, metrics, DEVICE,
)

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
OUT_DIR = Path("output/merger_single_w_slack")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_lossless(model, data):
    with torch.no_grad():
        y = model(data)
        return (torch.sign(y) == torch.sign(data)).all(dim=1).all().item()


def find_slack(model, data, i, j, probes):
    """Probes is a list of |delta| values, sorted increasing.
    Returns max delta at which lossless still holds on both sides, or 0.
    """
    orig = model.W.data[i, j].item()
    pos_max = 0.0
    neg_max = 0.0
    with torch.no_grad():
        for d in probes:
            model.W.data[i, j] = orig + d
            if is_lossless(model, data):
                pos_max = d
            else:
                break
        model.W.data[i, j] = orig
        for d in probes:
            model.W.data[i, j] = orig - d
            if is_lossless(model, data):
                neg_max = d
            else:
                break
        model.W.data[i, j] = orig
    return min(pos_max, neg_max), pos_max, neg_max


def main():
    print(f"=== PER-CELL SLACK MAP ===")
    print(f"Source: {CHAMPION}")

    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float32)

    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(m["b1"], dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(m["b2"], dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(m["c19_c"], dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(m["c19_rho"], dtype=torch.float32, device=DEVICE))
    data = load_byte_pairs().to(DEVICE)
    assert is_lossless(model, data), "baseline not lossless"

    # Probe values (log-spaced in magnitude)
    probes = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    print(f"Probes: {probes}")
    print(f"Cells: 32 * 81 = 2592. Tests per cell: 2 * {len(probes)} = {2*len(probes)}.")

    t0 = time.time()
    slack = np.zeros((32, 81), dtype=np.float32)
    pos_map = np.zeros((32, 81), dtype=np.float32)
    neg_map = np.zeros((32, 81), dtype=np.float32)
    for i in range(32):
        for j in range(81):
            s, p, n = find_slack(model, data, i, j, probes)
            slack[i, j] = s
            pos_map[i, j] = p
            neg_map[i, j] = n
        if (i + 1) % 4 == 0:
            el = time.time() - t0
            print(f"  row {i+1}/32: slack min={slack[:i+1].min():.5f} max={slack[:i+1].max():.5f} ({el:.0f}s)", flush=True)

    # Analyze
    print(f"\n=== SLACK DISTRIBUTION ===")
    flat = slack.flatten()
    for thr in probes:
        n_below = (flat < thr).sum()
        print(f"  cells with slack < {thr:.4f}: {n_below} ({100*n_below/flat.size:.1f}%)")
    print(f"  min slack : {flat.min():.6f}")
    print(f"  median    : {np.median(flat):.6f}")
    print(f"  max tested: {flat.max():.6f}")

    # Histogram
    print(f"\n  Histogram:")
    bins_edges = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]
    counts, _ = np.histogram(flat, bins=bins_edges)
    for i in range(len(bins_edges) - 1):
        print(f"    [{bins_edges[i]:.4f} .. {bins_edges[i+1]:.4f}]  {counts[i]:5d} ({100*counts[i]/flat.size:.1f}%)")

    # Budget math: what deploy size if each cell uses bits proportional to -log2(slack)?
    # More slack = fewer bits needed
    print(f"\n=== PRECISION-AWARE BIT BUDGET ===")
    print(f"  Per cell bit estimate: log2(range / slack)")
    range_W = 2 * np.abs(W).max()  # full range
    # Avoid log2(0): clamp to smallest probe
    slack_clamped = np.maximum(flat, probes[0])
    bits_per_cell = np.log2(range_W / slack_clamped)
    total_bits = bits_per_cell.sum()
    print(f"  Total bits: {total_bits:.0f} ({total_bits/8:.0f} bytes)")
    print(f"  Mean bits/cell: {bits_per_cell.mean():.2f}")
    print(f"  If aligned to 8 bits/cell uniformly: 2592 bytes")

    # Save
    np.save(OUT_DIR / "slack.npy", slack)
    with open(OUT_DIR / "slack_summary.json", "w") as f:
        json.dump({
            "probes": probes,
            "slack_min": float(flat.min()),
            "slack_median": float(np.median(flat)),
            "slack_max": float(flat.max()),
            "bins_edges": bins_edges,
            "bins_counts": counts.tolist(),
            "mean_bits_per_cell": float(bits_per_cell.mean()),
            "total_bits": float(total_bits),
        }, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/slack.npy and slack_summary.json  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
