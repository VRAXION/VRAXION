"""Fast dictionary compression prototype (vectorized).

Precompute a coverage matrix: cover[i, j, c] = W[i] is within slack[i] of coef[c] * W[j].
Then greedy pick generators to maximize union coverage.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
SLACK = Path("output/merger_single_w_slack/slack.npy")


def build_coverage(W, slack, coefs):
    """cover[i, j] = True if W[i] ≈ coef * W[j] for any coef in coefs, within slack[i].

    Returns: cover (n x n boolean), best_coef (n x n, -1 if no cover).
    """
    n = len(W)
    cover = np.zeros((n, n), dtype=bool)
    best_coef = np.full((n, n), 0, dtype=np.int8)
    for c in coefs:
        if c == 0: continue
        # targets[j] = c * W[j]
        targets = c * W  # shape (n,)
        # errors[i, j] = |W[i] - targets[j]|
        err = np.abs(W[:, None] - targets[None, :])  # (n, n)
        # fits[i, j] = True if err < slack[i]
        fits = err < slack[:, None]
        # Only update where not yet covered
        newly = fits & (~cover)
        cover |= newly
        best_coef[newly] = c
    # Zero the diagonal (a cell isn't its own ref with coef=1)
    np.fill_diagonal(cover, False)
    return cover, best_coef


def greedy_select_generators(cover, K):
    """Pick K generator cells to maximize union coverage."""
    n = cover.shape[0]
    chosen = []
    remaining_coverage = np.zeros(n, dtype=bool)
    for k in range(K):
        # For each candidate j, count how many uncovered cells it would newly cover
        # (plus j itself is a generator — always "covered" as itself)
        new_cov = cover & (~remaining_coverage)[:, None]
        new_counts = new_cov.sum(axis=0)
        # Prefer candidates not yet chosen
        for c in chosen:
            new_counts[c] = -1
        best = int(np.argmax(new_counts))
        if new_counts[best] <= 0:
            break
        chosen.append(best)
        remaining_coverage |= cover[:, best]
        remaining_coverage[best] = True  # generator is covered as itself
    return chosen, remaining_coverage


def main():
    with open(CHAMPION) as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float64).flatten()
    slack = np.load(SLACK).flatten()
    n = len(W)

    print(f"=== FAST DICTIONARY COMPRESSION ===")
    print(f"N cells: {n}, slack>0 cells: {(slack > 1e-10).sum()}")

    coefs = list(range(-7, 8))

    t0 = time.time()
    print(f"\nBuilding coverage matrix (n={n}, coefs={coefs})...")
    cover, best_coef = build_coverage(W, slack, coefs)
    print(f"  Done ({time.time()-t0:.1f}s). Coverage entries: {cover.sum()}")
    # Self-coverage check
    total_possible = cover.any(axis=1).sum()
    print(f"  Cells that have AT LEAST ONE valid (j,c) covering them: {total_possible}/{n} ({100*total_possible/n:.1f}%)")

    for K in [8, 16, 32, 64, 128, 256]:
        t1 = time.time()
        chosen, coverage = greedy_select_generators(cover, K)
        n_gen = len(chosen)
        n_covered = coverage.sum() - n_gen
        n_fallback = n - coverage.sum()
        print(f"\nK={K}: generators={n_gen}, covered={n_covered} ({100*n_covered/max(1,n-n_gen):.1f}% of non-gen), fallback={n_fallback}  ({time.time()-t1:.1f}s)")

        # Deploy size
        idx_bits = int(np.ceil(np.log2(max(K, 2))))
        coef_bits = int(np.ceil(np.log2(len(coefs) + 1)))  # include 0 marker
        # Per-cell:
        #   generators: 1 bit flag + 16 bits fp16 value = 17 bits
        #   covered:    1 bit flag + idx_bits + coef_bits bits
        #   fallback:   1 bit flag + 16 bits fp16 value
        gen_bits = 17
        cov_bits = 1 + idx_bits + coef_bits
        fb_bits = 17
        total_bits = n_gen * gen_bits + n_covered * cov_bits + n_fallback * fb_bits
        total_bytes = int(np.ceil(total_bits / 8))

        print(f"  idx_bits={idx_bits}, coef_bits={coef_bits}, per-covered-cell={cov_bits} bits")
        print(f"  Deploy W: {total_bytes} B (vs fp16 5184 B, delta {total_bytes-5184:+d} B)")


if __name__ == "__main__":
    main()
