"""Diag: greedy exhaustive int8 assignment on a fresh float baseline.

For each float cell (sorted smallest |w| first), try every int in [-127..127]
and keep the first one that maintains 100% lossless. No retrain, no rollback.
Shows the structural ceiling of single-pass exhaustive greedy int8.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_free_int8 import (
    FreeInt8Merger, load_source_json, load_byte_pairs,
    metrics, DEVICE,
)

SOURCE = "output/merger_exact_quick/01_exact_float/final_model.json"
INT_RANGE = 127


def build_model_from_float_baseline(state, device):
    """Build a FreeInt8Merger with all cells as free float (no frozen, no int8)."""
    H = state["H"]
    in_dim = state["in_dim"]
    out_dim = state["out_dim"]

    # Synthesize a FreeInt8Merger state dict
    W1 = np.array(state["W1"])
    W2 = np.array(state["W2"])
    synth_state = {
        "H": H, "in_dim": in_dim, "out_dim": out_dim,
        "codebook_W1": [0.0],  # dummy single entry
        "codebook_W2": [0.0],
        "W1_idx": np.zeros(W1.shape, dtype=int).tolist(),
        "W2_idx": np.zeros(W2.shape, dtype=int).tolist(),
        "W1_frozen_mask": np.zeros(W1.shape, dtype=int).tolist(),
        "W2_frozen_mask": np.zeros(W2.shape, dtype=int).tolist(),
        "W1_float": W1.tolist(),
        "W2_float": W2.tolist(),
        "b1": state["b1"],
        "b2": state["b2"],
        "db1": state["db1"],
        "db2": state["db2"],
        "c19_c": state["c19_c"],
        "c19_rho": state["c19_rho"],
    }
    model = FreeInt8Merger(synth_state, device).to(device)
    # Set alpha_free to max_abs / 127
    with torch.no_grad():
        a_f1 = max(np.abs(W1).max() / 127.0, 1e-6)
        a_f2 = max(np.abs(W2).max() / 127.0, 1e-6)
        model.alpha_free_W1.data.fill_(a_f1)
        model.alpha_free_W2.data.fill_(a_f2)
    return model


def main():
    state = load_source_json(SOURCE)
    model = build_model_from_float_baseline(state, DEVICE)
    data = load_byte_pairs().to(DEVICE)

    ll0, pd0 = metrics(model, data)
    print(f"Baseline: ll={ll0:.4f}%, pd={pd0:.4f}%")
    print(f"alpha_W1 = {model.alpha_free_W1.item():.6f}")
    print(f"alpha_W2 = {model.alpha_free_W2.item():.6f}")

    # Order cells by |w| ascending
    W1 = model.W1_float.data.cpu().numpy()
    W2 = model.W2_float.data.cpu().numpy()
    cells = []
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            cells.append(("W1", i, j, abs(W1[i, j])))
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            cells.append(("W2", i, j, abs(W2[i, j])))
    cells.sort(key=lambda c: c[3])
    print(f"\nTotal cells: {len(cells)}")

    n_accept = 0
    n_fail = 0
    t0 = time.time()
    alpha_W1 = model.alpha_free_W1.item()
    alpha_W2 = model.alpha_free_W2.item()

    for idx, (mat, i, j, _) in enumerate(cells):
        if mat == "W1":
            w = model.W1_float.data[i, j].item()
            alpha = alpha_W1
        else:
            w = model.W2_float.data[i, j].item()
            alpha = alpha_W2
        nearest = int(round(w / alpha))
        nearest = max(-INT_RANGE, min(INT_RANGE, nearest))

        # Try nearest first, then expand outward
        candidates = [nearest]
        for d in range(1, INT_RANGE + 1):
            if nearest - d >= -INT_RANGE:
                candidates.append(nearest - d)
            if nearest + d <= INT_RANGE:
                candidates.append(nearest + d)

        accepted = False
        with torch.no_grad():
            for iv in candidates:
                if mat == "W1":
                    model.W1_int8_mask[i, j] = True
                    model.W1_int8[i, j] = float(iv)
                else:
                    model.W2_int8_mask[i, j] = True
                    model.W2_int8[i, j] = float(iv)
                ll, _ = metrics(model, data)
                if ll >= 100.0:
                    accepted = True
                    break
                else:
                    # Reset mask only (don't retry from scratch in next iter)
                    if mat == "W1":
                        model.W1_int8_mask[i, j] = False
                    else:
                        model.W2_int8_mask[i, j] = False

        if accepted:
            n_accept += 1
        else:
            n_fail += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(cells) - idx - 1) / max(rate, 0.1)
            print(f"  #{idx+1}/{len(cells)}  acc={n_accept}  fail={n_fail}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  ll={metrics(model, data)[0]:.4f}%", flush=True)

    t_total = time.time() - t0
    ll, pd = metrics(model, data)
    print(f"\n=== DONE ===")
    print(f"  Processed: {len(cells)}")
    print(f"  Accepted int8: {n_accept} ({100*n_accept/len(cells):.1f}%)")
    print(f"  Failed (stays float): {n_fail}")
    print(f"  Final lossless: {ll:.4f}%")
    print(f"  Time: {t_total:.0f}s")

    # Deploy byte estimate (simplified)
    int8_cells = n_accept
    float_cells = n_fail
    deploy = int8_cells + float_cells * 4 + 8 + (model.b1.numel() + model.b2.numel() + model.db1.numel() + model.db2.numel() + 2 * model.H) * 4
    print(f"  Deploy est: {deploy} B ({deploy/1024:.2f} KB)")


if __name__ == "__main__":
    main()
