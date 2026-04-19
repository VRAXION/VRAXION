"""Dictionary compression prototype: cells as (ref, coef) combinations.

Strategy:
  1. Find K 'generator' cells — the ones that maximally cover other cells via
     integer multiples within per-cell slack.
  2. For each non-generator cell, encode as (ref_idx, small_int_coef).
  3. If no combo works within slack, fall back to float/fp16 for that cell.

  Deploy:
    K generators * fp16 (2 B each)
    For each covered cell:  ceil(log2(K)) bits ref + 4 bits coef  = e.g. 10+4=14 bits
    For each uncovered cell: fp16 value directly (16 bits)
    Per-cell 1-bit flag (covered/fallback)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
SLACK = Path("output/merger_single_w_slack/slack.npy")


def find_best_covering(W, slack, K, coef_range=range(-7, 8)):
    """Greedy: select K cells as generators. Goal: max number of other cells
    expressible as coef * gen within their slack."""
    n = len(W)
    W_abs_max = np.abs(W).max()
    # Start: pick the cell with smallest magnitude as first generator (helps tiny cells)
    gen_set = set()
    # Strategy: pick cells that cover the most other cells
    print(f"  Greedy generator selection (K={K}, coefs={list(coef_range)})...")
    for k in range(K):
        best_gain = 0
        best_idx = -1
        for cand in range(n):
            if cand in gen_set: continue
            # Count cells that would be newly covered if we added 'cand'
            gain = 0
            for i in range(n):
                if i == cand or i in gen_set: continue
                # Is i expressible as coef * W[cand] within slack[i]?
                for coef in coef_range:
                    if coef == 0: continue
                    if abs(W[i] - coef * W[cand]) < slack[i]:
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
        if best_idx == -1: break
        gen_set.add(best_idx)
        if (k+1) % 10 == 0:
            print(f"    Selected {k+1} generators, last gain: {best_gain}")
    return sorted(gen_set)


def encode_cells(W, slack, generators, coef_range=range(-7, 8)):
    """For each cell, try to encode as (gen_idx, coef) within slack.
    Return: encoding[i] = (gen_idx, coef) or None if not encodable.
    """
    n = len(W)
    encoding = [None] * n
    gen_arr = np.array(generators)
    gen_W = W[gen_arr]
    for i in range(n):
        if i in generators:
            encoding[i] = ("GEN", None)
            continue
        best_err = np.inf
        best_pair = None
        for gi, gw in enumerate(gen_W):
            for coef in coef_range:
                if coef == 0: continue
                err = abs(W[i] - coef * gw)
                if err < slack[i] and err < best_err:
                    best_err = err
                    best_pair = (gi, coef)
        encoding[i] = best_pair
    return encoding


def main():
    with open(CHAMPION) as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float64).flatten()
    slack = np.load(SLACK).flatten()
    n = len(W)

    print(f"=== DICTIONARY COMPRESSION PROTOTYPE ===")
    print(f"N cells: {n}")
    print(f"Cells with slack > 0: {(slack > 1e-10).sum()}")
    print(f"Cells with slack = 0 (razor edge): {(slack <= 1e-10).sum()}")

    # Test different K values
    for K in [16, 32, 64, 128]:
        print(f"\n--- K={K} generators ---")
        t0 = time.time()
        gens = find_best_covering(W, slack, K, coef_range=range(-7, 8))
        enc = encode_cells(W, slack, gens, coef_range=range(-7, 8))

        n_gen = len([e for e in enc if e == ("GEN", None)])
        n_covered = len([e for e in enc if e not in (None, ("GEN", None))])
        n_fallback = len([e for e in enc if e is None])
        coverage = 100 * (n_covered) / (n - n_gen)
        print(f"  Generators: {n_gen}, covered: {n_covered}, fallback: {n_fallback}")
        print(f"  Coverage of non-generators: {coverage:.1f}%")

        # Deploy size estimate:
        # Each generator: 2 B (fp16)
        # Each covered cell: log2(K) bits ref + 4 bits coef + 1 bit flag = ceil(log2(K))+5 bits
        # Each fallback cell: 16 bits fp16 + 1 bit flag = 17 bits
        bits_per_covered = int(np.ceil(np.log2(K))) + 4 + 1
        bits_per_fallback = 16 + 1
        total_bits = n_gen * 16 + n_covered * bits_per_covered + n_fallback * bits_per_fallback
        total_bytes = int(np.ceil(total_bits / 8))
        print(f"  Per-cell bits: gen=16, covered={bits_per_covered}, fallback={bits_per_fallback}")
        print(f"  Total W: {total_bytes} B (vs 5184 B fp16, delta {total_bytes-5184:+d} B)")
        print(f"  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
