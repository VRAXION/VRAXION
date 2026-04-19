"""Diag: exhaustive int search on single cell.

For a handful of free float cells, try EVERY int value in [-127, 127]
with the current alpha, and see how many maintain 100.0000% lossless.
Tests whether the GPT strict pipeline's top-16 cutoff was too narrow.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_free_int8 import (
    FreeInt8Merger, load_source_json, load_byte_pairs,
    metrics, DEVICE,
)
from diag_byte_pair_merger_absorb_float import restore_full_state

SOURCE = "output/merger_exact_quick/03_strict_int8/final_model.json"
NUM_CELLS_TO_TEST = 8
INT_RANGE = 127


def main():
    state = load_source_json(SOURCE)
    model = FreeInt8Merger(state, DEVICE).to(DEVICE)
    restore_full_state(model, state)
    data = load_byte_pairs().to(DEVICE)

    ll0, pd0 = metrics(model, data)
    print(f"Baseline: ll={ll0:.4f}%, pd={pd0:.4f}%\n")
    print(f"alpha_W1 = {model.alpha_free_W1.item():.6f}")
    print(f"alpha_W2 = {model.alpha_free_W2.item():.6f}")

    # Find free cells (not frozen, not int8)
    with torch.no_grad():
        W1_free = (~model.W1_frozen_mask & ~model.W1_int8_mask).cpu().numpy()
        W2_free = (~model.W2_frozen_mask & ~model.W2_int8_mask).cpu().numpy()
    w1_free_idx = np.argwhere(W1_free)
    w2_free_idx = np.argwhere(W2_free)

    print(f"\nFree cells: W1={len(w1_free_idx)}, W2={len(w2_free_idx)}")

    # Pick NUM_CELLS_TO_TEST cells total, mixed W1/W2, spread over the free set
    np.random.seed(0)
    test_cells = []
    for _ in range(NUM_CELLS_TO_TEST // 2):
        i = np.random.randint(len(w1_free_idx))
        test_cells.append(("W1", int(w1_free_idx[i][0]), int(w1_free_idx[i][1])))
    for _ in range(NUM_CELLS_TO_TEST - NUM_CELLS_TO_TEST // 2):
        i = np.random.randint(len(w2_free_idx))
        test_cells.append(("W2", int(w2_free_idx[i][0]), int(w2_free_idx[i][1])))

    print(f"\nTesting {len(test_cells)} cells with exhaustive int search [-{INT_RANGE}..{INT_RANGE}]:\n")
    print(f"{'Cell':>12} {'w_orig':>10} {'near_int':>9} {'100% ints':>12} {'best_range':>20}")
    print("-" * 75)

    alpha_W1 = model.alpha_free_W1.item()
    alpha_W2 = model.alpha_free_W2.item()

    for cell in test_cells:
        mat, i, j = cell
        if mat == "W1":
            w_orig = model.W1_float.data[i, j].item()
            alpha = alpha_W1
        else:
            w_orig = model.W2_float.data[i, j].item()
            alpha = alpha_W2
        near_int = int(round(w_orig / alpha))

        # Save original
        if mat == "W1":
            orig_val = model.W1_float.data[i, j].item()
            orig_mask = bool(model.W1_int8_mask[i, j].item())
            orig_int = float(model.W1_int8[i, j].item())
        else:
            orig_val = model.W2_float.data[i, j].item()
            orig_mask = bool(model.W2_int8_mask[i, j].item())
            orig_int = float(model.W2_int8[i, j].item())

        # Exhaustive try all int values
        good_ints = []
        with torch.no_grad():
            if mat == "W1":
                model.W1_int8_mask[i, j] = True
            else:
                model.W2_int8_mask[i, j] = True

            for iv in range(-INT_RANGE, INT_RANGE + 1):
                if mat == "W1":
                    model.W1_int8[i, j] = float(iv)
                else:
                    model.W2_int8[i, j] = float(iv)
                ll, _ = metrics(model, data)
                if ll >= 100.0:
                    good_ints.append(iv)

            # Restore cell
            if mat == "W1":
                model.W1_int8_mask[i, j] = orig_mask
                model.W1_int8[i, j] = orig_int
                model.W1_float.data[i, j] = orig_val
            else:
                model.W2_int8_mask[i, j] = orig_mask
                model.W2_int8[i, j] = orig_int
                model.W2_float.data[i, j] = orig_val

        if good_ints:
            best_range = f"[{min(good_ints)}..{max(good_ints)}]"
        else:
            best_range = "NONE"
        print(f"  {mat}[{i:>2},{j:>2}]  {w_orig:+.4f}  {near_int:>+5}   {len(good_ints):>4}/255   {best_range:>20}")


if __name__ == "__main__":
    main()
