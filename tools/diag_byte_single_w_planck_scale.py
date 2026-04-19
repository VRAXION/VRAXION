"""Find the 'Planck scale' of the champion W — the natural step resolution.

Questions:
  1. What's the gap between adjacent sorted values? (Median / distribution)
  2. Is the distribution linear or log-scale?
  3. What 255 bins (codebook) naturally fit this distribution?
     Compare strategies: linear, log, Lloyd-Max, density-weighted.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")


def gap_analysis(W_sorted):
    gaps = np.diff(W_sorted)
    nonzero = gaps[gaps > 0]
    print(f"\n=== ADJACENT-VALUE GAPS ===")
    print(f"  n gaps          : {len(gaps)}")
    print(f"  min gap         : {gaps.min():.8f}")
    print(f"  median gap      : {np.median(gaps):.8f}")
    print(f"  mean gap        : {gaps.mean():.8f}")
    print(f"  max gap         : {gaps.max():.8f}")
    print(f"  p25 / p50 / p75 : {np.percentile(gaps, 25):.8f} / {np.percentile(gaps, 50):.8f} / {np.percentile(gaps, 75):.8f}")
    print(f"  p90 / p95 / p99 : {np.percentile(gaps, 90):.8f} / {np.percentile(gaps, 95):.8f} / {np.percentile(gaps, 99):.8f}")
    # Histogram of gaps
    bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    hist, _ = np.histogram(gaps, bins=bins)
    print(f"\n  Gap histogram (log scale):")
    for i in range(len(bins) - 1):
        print(f"    [{bins[i]:.0e} .. {bins[i+1]:.0e}]  {hist[i]:5d}")


def linearity_test(W):
    """Is the magnitude distribution linear or log?
    Plot log(|W|) vs rank, see if it's a straight line (uniform log distribution).
    """
    absW = np.abs(W[W != 0])
    sorted_abs = np.sort(absW)
    print(f"\n=== LINEARITY TEST ===")
    # Uniform in log-space would give: log(sorted[k]) ~ a * k + b (linear)
    # Uniform in linear-space: sorted[k] ~ a * k + b (linear)
    log_vals = np.log(sorted_abs)
    # Fit both
    x = np.arange(len(sorted_abs))
    # Linear fit
    a_lin, b_lin = np.polyfit(x, sorted_abs, 1)
    lin_residual = np.abs(sorted_abs - (a_lin * x + b_lin)).mean()
    # Log fit
    a_log, b_log = np.polyfit(x, log_vals, 1)
    log_residual = np.abs(log_vals - (a_log * x + b_log)).mean()
    # Power fit (sqrt)
    sqrt_vals = np.sqrt(sorted_abs)
    a_sq, b_sq = np.polyfit(x, sqrt_vals, 1)
    sq_residual = np.abs(sqrt_vals - (a_sq * x + b_sq)).mean()

    print(f"  Linear   fit residual: {lin_residual:.6f}  (lower = more linear)")
    print(f"  Log      fit residual: {log_residual:.6f}  (lower = more log-shaped)")
    print(f"  Sqrt     fit residual: {sq_residual:.6f}  (lower = sqrt-shaped)")

    # Best shape
    names = ["linear", "log", "sqrt"]
    residuals = [lin_residual, log_residual, sq_residual]
    best = names[np.argmin(residuals)]
    print(f"  -> best shape: {best}")


def lloyd_max_quantizer(arr, k, n_iter=200):
    """Lloyd-Max: iteratively optimal non-uniform quantizer (1-D k-means)."""
    arr = arr.flatten()
    # Init: uniform quantiles
    qs = np.linspace(0, 1, k + 1)
    centers = np.array([np.quantile(arr, (qs[i] + qs[i+1]) / 2) for i in range(k)])
    for _ in range(n_iter):
        # Assign each value to nearest center
        idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
        new_centers = np.zeros(k, dtype=np.float64)
        for j in range(k):
            m = idx == j
            if m.any():
                new_centers[j] = arr[m].mean()
            else:
                new_centers[j] = centers[j]
        if np.allclose(new_centers, centers, atol=1e-12):
            centers = new_centers
            break
        centers = new_centers
    # Final assignment
    idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
    recon = centers[idx]
    err = arr - recon
    return centers, idx, err


def linear_uniform_quant(arr, k=255):
    """Uniform symmetric: 255 bins spanning [-amax, amax]."""
    amax = np.abs(arr).max()
    alpha = 2 * amax / (k - 1)
    # Snap
    q = np.round((arr + amax) / alpha).astype(int).clip(0, k - 1)
    recon = q * alpha - amax
    err = arr - recon
    return q, recon, err, alpha


def log_quant(arr, k=255):
    """Log-spaced in magnitude, sign preserved. k must be odd to include 0."""
    k = k if k % 2 == 1 else k - 1
    half = (k - 1) // 2
    absW = np.abs(arr)
    amax = absW.max()
    amin = absW[absW > 0].min()  # min nonzero magnitude
    # Log-spaced magnitudes
    log_centers_mag = np.exp(np.linspace(np.log(amin), np.log(amax), half))
    # Full centers: [-max ... -min, 0, +min ... +max]
    centers = np.concatenate([-log_centers_mag[::-1], [0.0], log_centers_mag])
    idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
    recon = centers[idx]
    err = arr - recon
    return centers, idx, recon, err


def density_centers(arr, k=255, bw=0.01):
    """Density-based: find peaks using simple histogram."""
    bins = np.linspace(arr.min(), arr.max(), k + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
    recon = centers[idx]
    err = arr - recon
    return centers, idx, recon, err


def compare_quantizers(W, k=255):
    print(f"\n=== QUANTIZER COMPARISON (k={k}) ===")
    W_flat = W.flatten()

    # Linear uniform
    q, recon, err, alpha = linear_uniform_quant(W_flat, k=k)
    print(f"\n  LINEAR UNIFORM (step={alpha:.6f}):")
    print(f"    max_err : {np.abs(err).max():.6f}")
    print(f"    rmse    : {np.sqrt((err**2).mean()):.6f}")
    print(f"    unique  : {len(np.unique(q))}")

    # Lloyd-Max
    centers, idx, err_lm = lloyd_max_quantizer(W_flat, k=k, n_iter=100)
    print(f"\n  LLOYD-MAX (optimal MSE):")
    print(f"    max_err : {np.abs(err_lm).max():.6f}")
    print(f"    rmse    : {np.sqrt((err_lm**2).mean()):.6f}")
    print(f"    min cen : {centers.min():.6f}")
    print(f"    max cen : {centers.max():.6f}")
    print(f"    smallest gap between centers: {np.min(np.diff(np.sort(centers))):.8f}")
    print(f"    largest  gap between centers: {np.max(np.diff(np.sort(centers))):.8f}")

    # Log
    log_centers, idx, recon, err_log = log_quant(W_flat, k=k)
    print(f"\n  LOG-SPACED (magnitude, sign preserved):")
    print(f"    max_err : {np.abs(err_log).max():.6f}")
    print(f"    rmse    : {np.sqrt((err_log**2).mean()):.6f}")
    print(f"    centers count: {len(log_centers)}")

    return {"linear": (alpha, q, recon), "lloyd_max": (centers, idx, W_flat - err_lm)}


def tweak_needed_estimate(W, k=255):
    """What's the expected tweak magnitude after quant?

    For each cell, the quant error is in [-step/2, +step/2] (for linear).
    The float model required single-cell tweak 0.0003 to close the last pair.
    How does quant error compare?
    """
    _, _, err, alpha = linear_uniform_quant(W.flatten(), k=k)
    print(f"\n=== TWEAK FEASIBILITY ===")
    print(f"  Critical tweak for float model : 0.000295 (last pair)")
    print(f"  Linear int8 step / 2           : {alpha/2:.6f}  ({(alpha/2)/0.000295:.0f}x critical)")
    print(f"  => Linear int8 error is {int(alpha/2/0.000295)}x the critical threshold.")
    print(f"  Conclusion: quant breaks many pairs, need per-cell escape OR retraining.")


def show_top_density_peaks(W, n_peaks=20, bw=0.005):
    """Find where the weights actually cluster in magnitude."""
    W_flat = np.sort(W.flatten())
    print(f"\n=== DENSITY PEAKS (bandwidth={bw}) ===")
    # Simple histogram
    bins = np.arange(W_flat.min(), W_flat.max() + bw, bw)
    hist, _ = np.histogram(W_flat, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    # Top n peaks
    top_idx = np.argsort(hist)[-n_peaks:][::-1]
    print(f"  Top {n_peaks} peaks (center +/- bw/2):")
    for rank, i in enumerate(top_idx, 1):
        print(f"    #{rank:2d}  center={centers[i]:+.4f}  count={hist[i]}")


def main():
    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float64)  # higher precision for analysis
    print(f"=== PLANCK-SCALE ANALYSIS on W (32x81 = 2592 cells) ===")
    print(f"Source: {CHAMPION}")
    print(f"min/max: {W.min():+.6f} / {W.max():+.6f}")
    print(f"abs_mean: {np.abs(W).mean():.6f}, std: {W.std():.6f}")

    W_sorted = np.sort(W.flatten())
    gap_analysis(W_sorted)
    linearity_test(W.flatten())
    show_top_density_peaks(W.flatten(), n_peaks=15, bw=0.003)

    for k in [255, 127, 63]:
        compare_quantizers(W, k=k)
    tweak_needed_estimate(W, k=255)


if __name__ == "__main__":
    main()
