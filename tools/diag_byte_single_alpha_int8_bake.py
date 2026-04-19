"""Diag: freeze model with single (or two) best alpha values, all weights int8.

Test 1: SINGLE global alpha for all weights (W1 and W2 share).
Test 2: TWO alphas (alpha_W1, alpha_W2 grid-search).

For each alpha combo:
  W_int = round(W / alpha).clip(-127, 127)
  W_reconstructed = W_int * alpha
  Check lossless on 65536 byte pairs.

Report best alpha(s) + lossless + deploy byte estimate.
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


def build_model_from_float_baseline(state, device):
    H = state["H"]; in_dim = state["in_dim"]; out_dim = state["out_dim"]
    W1 = np.array(state["W1"]); W2 = np.array(state["W2"])
    synth_state = {
        "H": H, "in_dim": in_dim, "out_dim": out_dim,
        "codebook_W1": [0.0], "codebook_W2": [0.0],
        "W1_idx": np.zeros(W1.shape, dtype=int).tolist(),
        "W2_idx": np.zeros(W2.shape, dtype=int).tolist(),
        "W1_frozen_mask": np.zeros(W1.shape, dtype=int).tolist(),
        "W2_frozen_mask": np.zeros(W2.shape, dtype=int).tolist(),
        "W1_float": W1.tolist(), "W2_float": W2.tolist(),
        "b1": state["b1"], "b2": state["b2"],
        "db1": state["db1"], "db2": state["db2"],
        "c19_c": state["c19_c"], "c19_rho": state["c19_rho"],
    }
    return FreeInt8Merger(synth_state, device).to(device), W1, W2


def test_single_alpha(model, W1_orig, W2_orig, alpha, data):
    """Replace W1, W2 with int8 × alpha reconstruction. Return lossless."""
    W1_int = np.clip(np.round(W1_orig / alpha), -127, 127)
    W2_int = np.clip(np.round(W2_orig / alpha), -127, 127)
    W1_rec = W1_int * alpha
    W2_rec = W2_int * alpha
    with torch.no_grad():
        model.W1_float.data.copy_(torch.tensor(W1_rec, dtype=torch.float32).to(DEVICE))
        model.W2_float.data.copy_(torch.tensor(W2_rec, dtype=torch.float32).to(DEVICE))
    ll, pd = metrics(model, data)
    return ll, pd


def test_two_alpha(model, W1_orig, W2_orig, alpha_W1, alpha_W2, data):
    W1_int = np.clip(np.round(W1_orig / alpha_W1), -127, 127)
    W2_int = np.clip(np.round(W2_orig / alpha_W2), -127, 127)
    W1_rec = W1_int * alpha_W1
    W2_rec = W2_int * alpha_W2
    with torch.no_grad():
        model.W1_float.data.copy_(torch.tensor(W1_rec, dtype=torch.float32).to(DEVICE))
        model.W2_float.data.copy_(torch.tensor(W2_rec, dtype=torch.float32).to(DEVICE))
    ll, pd = metrics(model, data)
    return ll, pd


def main():
    state = load_source_json(SOURCE)
    model, W1, W2 = build_model_from_float_baseline(state, DEVICE)
    data = load_byte_pairs().to(DEVICE)

    max_W1 = float(np.abs(W1).max())
    max_W2 = float(np.abs(W2).max())
    print(f"Max |W1| = {max_W1:.4f}, Max |W2| = {max_W2:.4f}")
    print(f"Baseline before bake: ll={metrics(model, data)[0]:.4f}%")

    # TEST 1: SINGLE global alpha
    print("\n=== TEST 1: SINGLE global alpha (W1 & W2 share) ===")
    alpha_min = max_W1 / 127  # ensure W1 max fits int8
    alphas = np.geomspace(alpha_min, alpha_min * 4, 30)
    print(f"Grid: {len(alphas)} values in [{alpha_min:.6f}, {alpha_min*4:.6f}]")
    best_single = (-1, None, None)
    for a in alphas:
        ll, pd = test_single_alpha(model, W1, W2, a, data)
        if ll > best_single[0]:
            best_single = (ll, pd, a)
        if ll >= 100.0:
            print(f"  alpha={a:.6f}  ll={ll:.4f}%  pd={pd:.4f}%  OK 100%")
    print(f"\n  BEST SINGLE: alpha={best_single[2]:.6f}, ll={best_single[0]:.4f}%, pd={best_single[1]:.4f}%")

    # TEST 2: TWO alphas (per-matrix)
    print("\n=== TEST 2: TWO alphas (alpha_W1, alpha_W2) ===")
    a1_min = max_W1 / 127
    a2_min = max_W2 / 127
    # Grid: each from min to min*8 (logarithmic)
    a1_grid = np.geomspace(a1_min, a1_min * 8, 20)
    a2_grid = np.geomspace(a2_min, a2_min * 8, 20)
    print(f"Grid: {len(a1_grid)}x{len(a2_grid)} = {len(a1_grid)*len(a2_grid)} combos")

    t0 = time.time()
    best_two = (-1, None, None, None)
    lossless_combos = []
    for a1 in a1_grid:
        for a2 in a2_grid:
            ll, pd = test_two_alpha(model, W1, W2, a1, a2, data)
            if ll > best_two[0]:
                best_two = (ll, pd, a1, a2)
            if ll >= 100.0:
                lossless_combos.append((a1, a2, ll, pd))
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  BEST TWO: alpha_W1={best_two[2]:.6f}, alpha_W2={best_two[3]:.6f}, ll={best_two[0]:.4f}%, pd={best_two[1]:.4f}%")
    print(f"  Lossless combos found: {len(lossless_combos)}")
    if lossless_combos:
        print(f"  Sample lossless: alpha_W1={lossless_combos[0][0]:.6f}, alpha_W2={lossless_combos[0][1]:.6f}")

    # Deploy byte estimate (if we have 100% lossless bake)
    if best_two[0] >= 100.0:
        total_cells = W1.size + W2.size
        int8_bytes = total_cells  # 1 byte per cell
        alpha_bytes = 8  # 2 floats
        bias_bytes = (len(state["b1"]) + len(state["b2"]) + len(state["db1"]) + len(state["db2"])) * 4
        c19_bytes = 2 * state["H"] * 4
        total = int8_bytes + alpha_bytes + bias_bytes + c19_bytes
        print(f"\n  PURE INT8 BAKE deploy: {total} B ({total/1024:.2f} KB)")
        print(f"  VS hibrid baseline:    7312 B (7.14 KB)")


if __name__ == "__main__":
    main()
