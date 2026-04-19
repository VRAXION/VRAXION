"""Analyze the 2592-cell single-W champion for compression opportunities.

Loads output/merger_single_w_exhaustive_fix/final_model.json and reports:
- W: range, mean, std, abs_mean, unique count, density peaks, symmetry, outliers
- b1, b2: same stats
- c19_c, c19_rho: same stats
- Clustering potential (k-means on W at various k)
- Bit-width estimates for different quantization strategies
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MODEL = Path("output/merger_single_w_exhaustive_fix/final_model.json")


def stat_block(name, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    uniq = np.unique(arr)
    print(f"\n=== {name} ===")
    print(f"  shape/count : {arr.shape}, N={arr.size}")
    print(f"  unique vals : {uniq.size}")
    print(f"  min / max   : {arr.min():+.6f} / {arr.max():+.6f}")
    print(f"  mean / std  : {arr.mean():+.6f} / {arr.std():.6f}")
    print(f"  abs_mean    : {np.abs(arr).mean():.6f}")
    print(f"  median      : {np.median(arr):+.6f}")
    # Percentiles
    for p in [1, 5, 25, 75, 95, 99]:
        print(f"  p{p:02d}         : {np.percentile(arr, p):+.6f}")
    # Symmetry around zero
    pos = (arr > 0).sum()
    neg = (arr < 0).sum()
    zero = (arr == 0).sum()
    print(f"  sign split  : pos={pos} ({100*pos/arr.size:.1f}%) neg={neg} ({100*neg/arr.size:.1f}%) zero={zero}")
    # Outliers (>3 std)
    thr = 3 * arr.std()
    outl = (np.abs(arr - arr.mean()) > thr).sum()
    print(f"  outliers>3s : {outl} ({100*outl/arr.size:.1f}%)")


def histogram_block(name, arr, n_bins=20):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = counts.max()
    print(f"\n--- {name} histogram ({n_bins} bins) ---")
    for i in range(n_bins):
        bar_len = int(50 * counts[i] / max_count) if max_count else 0
        bar = "#" * bar_len
        print(f"  [{edges[i]:+.4f}, {edges[i+1]:+.4f}] {counts[i]:5d} {bar}")


def clustering_estimate(arr, ks=(16, 32, 64, 128, 256, 512)):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    print(f"\n--- K-means clustering (1-D) on {arr.size} values ---")
    print(f"  k     | max_abs_err    | rmse           | bits")
    for k in ks:
        # Simple 1-D k-means: sort, equal-count quantiles, iterate a few times
        v = np.sort(arr)
        # init centers: quantiles
        qs = np.linspace(0, 1, k + 1)
        centers = np.array([np.quantile(v, (qs[i] + qs[i+1]) / 2) for i in range(k)])
        for _ in range(30):
            # assign
            idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
            # update
            new_centers = np.array([arr[idx == j].mean() if (idx == j).any() else centers[j]
                                    for j in range(k)])
            if np.allclose(new_centers, centers, atol=1e-9):
                break
            centers = new_centers
        idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
        recon = centers[idx]
        err = arr - recon
        max_err = np.abs(err).max()
        rmse = np.sqrt((err ** 2).mean())
        bits = int(np.ceil(np.log2(k)))
        print(f"  {k:5d} | {max_err:.6f}      | {rmse:.6f}      | {bits} bits/cell")


def int8_snap_analysis(arr, name):
    """If we quantize linearly to int8 (one global alpha), how much error?"""
    arr = np.asarray(arr, dtype=np.float32).flatten()
    abs_max = np.abs(arr).max()
    alpha = abs_max / 127
    q = np.round(arr / alpha).astype(np.int32)
    q = np.clip(q, -127, 127)
    recon = q * alpha
    err = arr - recon
    print(f"\n--- {name} int8 linear snap (alpha={alpha:.6e}) ---")
    print(f"  max_abs_err : {np.abs(err).max():.6f}")
    print(f"  rmse        : {np.sqrt((err ** 2).mean()):.6f}")
    print(f"  unique ints : {len(np.unique(q))}")


def main():
    if not MODEL.exists():
        print(f"ERR: {MODEL} not found"); return
    with open(MODEL, "r") as f:
        m = json.load(f)
    print("=== Single-W CHAMPION ANALYSIS ===")
    print(f"Source: {MODEL}")
    print(f"Architecture: {m['architecture']}")
    print(f"H={m['H']}, in_dim={m['in_dim']}, out_dim={m['out_dim']}")
    print(f"Starting lossless: {m.get('starting_lossless', '?')}%")
    print(f"Final lossless   : {m['final_lossless']}%")
    print(f"Tweak            : cell {m.get('tweak_cell', '?')} += {m.get('tweak_delta', '?')}")

    W = np.array(m["W"], dtype=np.float32)
    b1 = np.array(m["b1"], dtype=np.float32)
    b2 = np.array(m["b2"], dtype=np.float32)
    c19_c = np.array(m["c19_c"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho"], dtype=np.float32)

    # ---- MAIN W MATRIX ----
    stat_block("W matrix (32x81 = 2592 cells)", W)
    histogram_block("W", W, n_bins=30)
    int8_snap_analysis(W, "W")
    clustering_estimate(W.flatten(), ks=(16, 32, 64, 128, 256, 512))

    # Row/col patterns
    print(f"\n--- W row/col abs_mean ---")
    row_abs = np.abs(W).mean(axis=1)
    col_abs = np.abs(W).mean(axis=0)
    print(f"  rows (32)   abs_mean range: {row_abs.min():.6f} .. {row_abs.max():.6f}")
    print(f"  cols (81)   abs_mean range: {col_abs.min():.6f} .. {col_abs.max():.6f}")

    # Per-row int8 alpha (32 alphas, one per row)
    print(f"\n--- W per-row int8 analysis (32 alphas) ---")
    max_errs = []
    all_unique = set()
    for r in range(W.shape[0]):
        row = W[r]
        amax = np.abs(row).max()
        alpha = amax / 127
        q = np.round(row / alpha).astype(np.int32).clip(-127, 127)
        recon = q * alpha
        max_errs.append(np.abs(row - recon).max())
        all_unique.update(int(v) for v in q)
    print(f"  worst row max_err: {max(max_errs):.6f}")
    print(f"  mean  row max_err: {np.mean(max_errs):.6f}")
    print(f"  total unique int values across rows: {len(all_unique)}")

    # Per-col int8 alpha (81 alphas)
    print(f"\n--- W per-col int8 analysis (81 alphas) ---")
    max_errs_col = []
    for c in range(W.shape[1]):
        col = W[:, c]
        amax = np.abs(col).max()
        alpha = amax / 127
        q = np.round(col / alpha).astype(np.int32).clip(-127, 127)
        recon = q * alpha
        max_errs_col.append(np.abs(col - recon).max())
    print(f"  worst col max_err: {max(max_errs_col):.6f}")
    print(f"  mean  col max_err: {np.mean(max_errs_col):.6f}")

    # ---- BIAS 1 ----
    stat_block("b1 (81 values, pre-C19)", b1)
    int8_snap_analysis(b1, "b1")

    # ---- BIAS 2 ----
    stat_block("b2 (32 values, post-W.T)", b2)
    int8_snap_analysis(b2, "b2")

    # ---- C19 ----
    stat_block("c19_c (81 values)", c19_c)
    stat_block("c19_rho (81 values)", c19_rho)

    # ---- DEPLOY BYTE ESTIMATE ----
    print(f"\n=== DEPLOY SIZE ESTIMATES ===")
    print(f"  Float32 dump (naive):")
    n_vals = W.size + b1.size + b2.size + c19_c.size + c19_rho.size
    print(f"    {n_vals} values x 4 bytes = {n_vals * 4} bytes ({n_vals * 4 / 1024:.2f} KB)")

    print(f"\n  If W -> 8-bit codebook (256 entries, 1 byte/cell):")
    w_code = 256 * 4 + W.size  # 256 float32 codebook + 1 byte/cell
    rest = (b1.size + b2.size + c19_c.size + c19_rho.size) * 4
    print(f"    codebook 1024 B + W cells {W.size} B + rest {rest} B")
    print(f"    = {w_code + rest} bytes ({(w_code + rest) / 1024:.2f} KB)")

    print(f"\n  If W -> 6-bit codebook (64 entries, 6 bit/cell):")
    w_code6 = 64 * 4 + int(np.ceil(W.size * 6 / 8))
    print(f"    codebook 256 B + W cells {int(np.ceil(W.size * 6 / 8))} B + rest {rest} B")
    print(f"    = {w_code6 + rest} bytes ({(w_code6 + rest) / 1024:.2f} KB)")

    print(f"\n  If W -> int8 global linear (1 alpha, 1 byte/cell):")
    w_int8 = 4 + W.size
    print(f"    alpha 4 B + W cells {W.size} B + rest {rest} B")
    print(f"    = {w_int8 + rest} bytes ({(w_int8 + rest) / 1024:.2f} KB)")

    print(f"\n  If W -> int4 codebook (16 entries, 4 bit/cell):")
    w_int4 = 16 * 4 + W.size // 2
    print(f"    codebook 64 B + W cells {W.size // 2} B + rest {rest} B")
    print(f"    = {w_int4 + rest} bytes ({(w_int4 + rest) / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
