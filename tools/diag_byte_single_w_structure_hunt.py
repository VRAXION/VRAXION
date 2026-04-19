"""Hunt for hidden structure in the champion W (32x81 = 2592 cells).

Three tests:
  1. SVD rank: how many independent 'directions' is W using? Low rank = big win.
  2. Dictionary learning: can K atoms + sparse combos reconstruct W?
  3. GCD unit: is there a common 'step' that divides most values?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")


def svd_test(W):
    print("\n" + "="*60)
    print("TEST 1: SVD RANK — hány független irányt használ W?")
    print("="*60)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    print(f"W shape: {W.shape}, singular values count: {len(s)}")
    print(f"\nSingular values (first 32):")
    total = s.sum()
    cumul = 0
    print(f"  #   s_i         %_of_total  cumul_%")
    for i, sv in enumerate(s):
        cumul += sv
        print(f"  {i:2d}  {sv:10.6f}  {100*sv/total:6.2f}   {100*cumul/total:6.2f}")

    # Low-rank truncation tests
    print(f"\nLow-rank reconstruction error (rank K):")
    print(f"  K  | max_err     | rmse        | W_params saved")
    full_params = W.size
    for K in [32, 24, 16, 12, 8, 6, 4, 2]:
        W_k = (U[:, :K] * s[:K]) @ Vt[:K, :]
        err = W - W_k
        max_err = np.abs(err).max()
        rmse = np.sqrt((err**2).mean())
        # Storage: U_K (32*K) + s_K (K) + Vt_K (K*81)
        stored = 32*K + K + K*81
        saved = full_params - stored
        print(f"  {K:2d}  | {max_err:.6f}  | {rmse:.6f}  | {stored:4d} vs {full_params} ({100*stored/full_params:5.1f}%)")


def gcd_test(W):
    print("\n" + "="*60)
    print("TEST 2: GCD UNIT — van-e közös 'lépéskvantum'?")
    print("="*60)
    # For each candidate unit, count how many cells are 'close' to an integer multiple
    W_flat = np.abs(W.flatten())
    W_flat = W_flat[W_flat > 1e-6]  # skip near-zero
    for unit in [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00001]:
        rem = (W_flat / unit) - np.round(W_flat / unit)
        err = np.abs(rem * unit)  # error in original scale
        # Fraction within 10% of unit
        pct_close = (err < unit * 0.1).mean() * 100
        max_err = err.max()
        mean_err = err.mean()
        print(f"  unit={unit:.6f}: {pct_close:5.1f}% fit, max_err={max_err:.6f}, mean_err={mean_err:.6f}")


def sparse_dictionary_test(W, K_list=(4, 8, 16, 32), max_coef_options=(3, 5, 7)):
    """Fit K atoms, express each cell as sum of small-integer * atom.

    Each cell = sum_k coef[k] * atom[k], coef in {-c, ..., -1, 0, 1, ..., c}
    where c = (max_coef_options - 1) / 2.
    """
    print("\n" + "="*60)
    print("TEST 3: DICTIONARY — hány alap tudja lefedni a W-t?")
    print("="*60)
    W_flat = W.flatten()
    for K in K_list:
        # K-SVD lite: alternate between atom update and coef fit (with integer constraint)
        # Init atoms: top-K density peaks
        atoms = np.random.randn(K) * W.std() * 0.5
        atoms = np.sort(atoms)

        for max_coef_count in max_coef_options:
            half = (max_coef_count - 1) // 2
            coef_range = np.arange(-half, half + 1)  # e.g., [-1, 0, 1]

            # For each cell, find best (coef1, ..., coefK) combination via exhaustive or greedy
            # Exhaustive is max_coef_count^K combos — doable for K<=6, too big for K=32
            if K <= 4:
                import itertools
                combos = np.array(list(itertools.product(coef_range, repeat=K)))
                values = combos @ atoms  # all possible values
                # For each cell, find nearest
                idx = np.argmin(np.abs(W_flat[:, None] - values[None, :]), axis=1)
                recon = values[idx]
            else:
                # Greedy matching pursuit: choose coef one atom at a time
                recon = np.zeros_like(W_flat)
                residual = W_flat.copy()
                coefs = np.zeros((W_flat.size, K), dtype=int)
                # Iterate
                for _ in range(K):
                    # For each cell, find best (atom_idx, coef) pair
                    # value added = coef * atoms[k]
                    # test: what's the impact of adding coef * atoms[k] on residual?
                    best_change = np.zeros(W_flat.size)
                    best_k = np.zeros(W_flat.size, dtype=int)
                    best_c = np.zeros(W_flat.size, dtype=int)
                    for k in range(K):
                        for c in coef_range:
                            if c == 0: continue
                            change = np.abs(residual)**2 - np.abs(residual - c * atoms[k])**2
                            mask = change > best_change
                            best_change = np.where(mask, change, best_change)
                            best_k = np.where(mask, k, best_k)
                            best_c = np.where(mask, c, best_c)
                    # Apply
                    for i in range(W_flat.size):
                        if best_change[i] > 0:
                            coefs[i, best_k[i]] += best_c[i]
                            residual[i] -= best_c[i] * atoms[best_k[i]]
                recon = coefs @ atoms

            # Measure
            err = W_flat - recon
            max_err = np.abs(err).max()
            rmse = np.sqrt((err**2).mean())
            bits_per_coef = int(np.ceil(np.log2(max_coef_count)))
            bits_per_cell = K * bits_per_coef
            cells_bytes = 2592 * bits_per_cell / 8
            atoms_bytes = K * 4
            total = cells_bytes + atoms_bytes
            print(f"  K={K:2d} coefs={max_coef_count}: max_err={max_err:.6f}, rmse={rmse:.6f}, "
                  f"bits/cell={bits_per_cell}, deploy={int(total):4d} B")


def unique_value_pairs(W):
    print("\n" + "="*60)
    print("TEST 4: PÁR-STRUKTÚRA — vannak-e egymás többszörösei?")
    print("="*60)
    vals = W.flatten()
    # Check: how many pairs (i,j) where vals[i] == 2 * vals[j]?
    # Approx within tolerance
    tol = 0.0005
    n_doubles = 0
    n_triples = 0
    n_halves = 0
    n_negatives = 0
    for v in vals[:100]:  # sample
        matches_2 = (np.abs(vals - 2*v) < tol).sum()
        matches_3 = (np.abs(vals - 3*v) < tol).sum()
        matches_h = (np.abs(vals - v/2) < tol).sum()
        matches_n = (np.abs(vals - (-v)) < tol).sum()
        n_doubles += matches_2
        n_triples += matches_3
        n_halves += matches_h
        n_negatives += matches_n
    print(f"  Sample of 100 values:")
    print(f"    matches for 2×v : {n_doubles} (avg {n_doubles/100:.1f} per value)")
    print(f"    matches for 3×v : {n_triples} (avg {n_triples/100:.1f} per value)")
    print(f"    matches for v/2 : {n_halves} (avg {n_halves/100:.1f} per value)")
    print(f"    matches for -v  : {n_negatives} (avg {n_negatives/100:.1f} per value)")


def main():
    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float32)  # 32x81
    print(f"=== STRUCTURE HUNT on W (shape {W.shape}, {W.size} cells) ===")
    print(f"Source: {CHAMPION}")

    svd_test(W)
    gcd_test(W)
    unique_value_pairs(W)
    sparse_dictionary_test(W, K_list=(3, 4), max_coef_options=(3, 5, 7))


if __name__ == "__main__":
    main()
