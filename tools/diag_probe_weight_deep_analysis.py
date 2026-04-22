"""Deep analysis of Block C weight + activation distributions.

For a freshly trained relu model (same recipe as the sweep):
  - Full percentile table per tensor
  - Per-row amax distribution (shows if tensor is heterogeneous -> per-channel quant helps)
  - ASCII histogram of weight magnitudes
  - Fine α sweep (200 points in [0.01, 1.0]) with BOTH weight-MSE and
    output-reconstruction-MSE curves — the real objective is output MSE
  - Rate-distortion: reconstruction MSE at bits ∈ {2,3,4,5,6,8} with α=1 (RTN)
    and with best α per bit → "how many bits do these weights actually need?"
  - Post-activation stats for the same input batch

Outputs everything to stdout as a readable plain-text report.

Run:
    python3 tools/diag_probe_weight_deep_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Python"))
from block_a_byte_unit import ByteEncoder  # noqa: E402

CONTEXT = 8
H = 128
LR = 0.05
MOMENTUM = 0.9
BATCH = 256
SEED = 1337
EPOCHS = 3
CORPUS = REPO_ROOT / "output" / "data" / "fineweb_edu_100mb.txt"
CORPUS_BYTES = 1_000_000   # same as the sweep


# ---------- activation (relu for the probe) ----------

def relu_forward(z):
    return np.maximum(z, 0.0)


# ---------- quantization helpers (symmetric, [-qmax, qmax]) ----------

def sym_quant(x, scale, qmax):
    q = np.round(x / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def rtn_reconstruct(x, bits, alpha=1.0):
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(x)))
    if amax == 0.0 or alpha <= 0:
        return x.copy(), 0.0
    scale = (alpha * amax) / qmax
    return sym_quant(x, scale, qmax), scale


# ---------- reports ----------

def percentile_table(name, W):
    flat = W.ravel()
    qs = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    vals = np.percentile(flat, qs)
    print(f"\n{name}: shape={W.shape}  n={flat.size:,}  "
          f"mean={flat.mean():+.5f}  std={flat.std():.5f}")
    print(f"  abs_max={np.abs(flat).max():.5f}   pos_frac={(flat>0).mean()*100:.2f}%")
    for q, v in zip(qs, vals):
        print(f"    p{q:5.1f} = {v:+.5f}")


def ascii_histogram(name, x, bins=40, width=60):
    """Horizontal ASCII histogram of abs(x)."""
    a = np.abs(x.ravel())
    lo, hi = 0.0, float(a.max()) * 1.0001
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    peak = counts.max() or 1
    print(f"\n{name}  |x|  histogram  (range 0 → {hi:.4f})")
    for c, left in zip(counts, edges):
        bar = "#" * int(round(width * c / peak))
        print(f"  {left:7.4f} |{bar:<{width}} {c}")


def per_row_amax_stats(name, W):
    """Is each row independent? If per-row amax varies wildly, per-channel
    quant unlocks huge dynamic range."""
    row_amax = np.max(np.abs(W), axis=1)
    print(f"\n{name}  per-row amax stats  (rows={W.shape[0]})")
    qs = [0, 25, 50, 75, 100]
    for q in qs:
        print(f"    row_amax p{q:3d} = {np.percentile(row_amax, q):.5f}")
    col_amax = np.max(np.abs(W), axis=0)
    print(f"  per-col amax  (cols={W.shape[1]})")
    for q in qs:
        print(f"    col_amax p{q:3d} = {np.percentile(col_amax, q):.5f}")
    print(f"  tensor amax = {np.abs(W).max():.5f}")
    print(f"  ratio max_row_amax / tensor_amax = {row_amax.max()/np.abs(W).max():.4f} "
          f"(1.0 means no per-row heterogeneity)")
    print(f"  ratio min_row_amax / tensor_amax = {row_amax.min()/np.abs(W).max():.4f} "
          f"(close to 1 = uniform rows; small = per-row quant unlocks range)")


def rate_distortion(name, W, X_calib):
    """Reconstruction MSE at different bit widths.

    Reports BOTH raw weight MSE (‖W-Wq‖) and output MSE (‖X_calib W - X_calib Wq‖),
    at RTN (α=1) and at best α from a fine 200-point grid.
    """
    print(f"\n{name}  rate-distortion  (α=1 RTN vs best α)")
    print(f"  bits |  α     |  W-MSE (×1e6)  |  out-MSE (×1e6)")
    ref = X_calib @ W
    for bits in [2, 3, 4, 5, 6, 8]:
        Wq_rtn, _ = rtn_reconstruct(W, bits, alpha=1.0)
        w_mse_rtn = float(np.mean((Wq_rtn - W) ** 2))
        o_mse_rtn = float(np.mean((X_calib @ Wq_rtn - ref) ** 2))

        best_w_mse = float("inf"); best_o_mse = float("inf"); best_alpha = 1.0
        for a in np.linspace(0.05, 1.0, 200):
            Wq, _ = rtn_reconstruct(W, bits, alpha=float(a))
            o_mse = float(np.mean((X_calib @ Wq - ref) ** 2))
            if o_mse < best_o_mse:
                best_o_mse = o_mse; best_alpha = float(a)
                best_w_mse = float(np.mean((Wq - W) ** 2))
        print(f"  int{bits:2d}| α=1.00 |  {w_mse_rtn*1e6:11.3f}  |  {o_mse_rtn*1e6:11.3f}   <- RTN")
        print(f"       | α={best_alpha:.2f} |  {best_w_mse*1e6:11.3f}  |  {best_o_mse*1e6:11.3f}   <- best")


def fine_alpha_sweep(name, W, X_calib, bits=4, n=200):
    """Fine α sweep — print curve as an ASCII line for both W-MSE and out-MSE."""
    ref = X_calib @ W
    alphas = np.linspace(0.05, 1.0, n)
    w_mses = np.empty(n); o_mses = np.empty(n)
    for i, a in enumerate(alphas):
        Wq, _ = rtn_reconstruct(W, bits, alpha=float(a))
        w_mses[i] = float(np.mean((Wq - W) ** 2))
        o_mses[i] = float(np.mean((X_calib @ Wq - ref) ** 2))

    print(f"\n{name}  fine α sweep (int{bits}, n={n}) — output-MSE curve")
    lo, hi = o_mses.min(), o_mses.max()
    best_i = int(o_mses.argmin())
    for step_i in range(0, n, 5):  # 40 rows
        a = alphas[step_i]; m = o_mses[step_i]
        norm = (m - lo) / (hi - lo + 1e-12)
        bar = "#" * int(round(60 * (1 - norm)))  # higher bar = lower MSE
        mark = " <<< BEST" if step_i == best_i else ""
        print(f"  α={a:.3f}  |{bar:<60}  MSE={m*1e6:.3f}{mark}")
    print(f"  best α = {alphas[best_i]:.3f}   min out-MSE = {o_mses[best_i]*1e6:.3f}")
    print(f"  α=1.00: out-MSE = {o_mses[-1]*1e6:.3f}  "
          f"(ratio best/RTN = {o_mses[best_i]/o_mses[-1]:.3f})")


def effective_bits_estimate(name, W):
    """Entropy-based estimate of how many bits the weights 'want'.

    Approximation: estimate Gaussian-equivalent entropy, compare to what int-N
    levels can carry (log2(2^N) = N). If data is heavy-tailed, true entropy is
    smaller than Gaussian and int-N saturates lower bit counts quickly.
    """
    flat = W.ravel()
    std = float(flat.std())
    gauss_entropy_nats = 0.5 * np.log(2 * np.pi * np.e * std * std)
    gauss_entropy_bits = gauss_entropy_nats / np.log(2)
    # Quantize to a fine grid and measure discrete entropy
    for bits in [4, 6, 8]:
        qmax = 2 ** (bits - 1) - 1
        amax = float(np.abs(flat).max()) or 1.0
        scale = amax / qmax
        codes = np.round(flat / scale).clip(-qmax, qmax).astype(np.int32)
        u, c = np.unique(codes, return_counts=True)
        p = c / c.sum()
        H_empirical = float(-(p * np.log2(p)).sum())
        print(f"  int{bits}  RTN codebook uses {len(u)}/{2*qmax+1} levels, "
              f"empirical entropy = {H_empirical:.3f} bits "
              f"(max possible = {np.log2(2*qmax+1):.3f})")
    print(f"  {name}  continuous Gaussian-approx entropy = {gauss_entropy_bits:.3f} bits "
          f"(reference only)")


# ---------- setup + train ----------

def main():
    print("=" * 72)
    print("Block C deep weight/activation distribution analysis")
    print("=" * 72)

    assert CORPUS.exists(), f"missing {CORPUS}"
    with CORPUS.open("rb") as f:
        raw = f.read(CORPUS_BYTES)
    print(f"Corpus: {CORPUS}  bytes={len(raw):,}")

    enc = ByteEncoder.load_default()
    lut = enc._lut_f32
    arr = np.frombuffer(raw, dtype=np.uint8)
    N = len(arr) - CONTEXT
    ctx = np.empty((N, CONTEXT), dtype=np.uint8)
    for i in range(CONTEXT):
        ctx[:, i] = arr[i : i + N]
    y = arr[CONTEXT : CONTEXT + N].astype(np.int64)
    X = lut[ctx].reshape(ctx.shape[0], -1)

    rng = np.random.default_rng(SEED)
    in_dim = X.shape[1]
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, 256)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), BATCH):
            idx = perm[i : i + BATCH]
            xb = X[idx]; yb = y[idx]; B = len(yb)
            z = xb @ W1 + b1
            a = relu_forward(z)
            logits = a @ W2 + b2
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m); p = e / e.sum(axis=1, keepdims=True)
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW2 = a.T @ dlogits; db2 = dlogits.sum(axis=0)
            da = dlogits @ W2.T
            dz = da * (z > 0)
            dW1 = xb.T @ dz; db1 = dz.sum(axis=0)
            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - LR * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - LR * db2; b2 += vb2
        # no eval between epochs — minimal probe

    # pick calibration batch (first 16K train windows)
    Xcal_for_W1 = X[:16384]
    z_cal = Xcal_for_W1 @ W1 + b1
    act_cal = relu_forward(z_cal)
    # calib input for W2 is the post-activation hidden
    Xcal_for_W2 = act_cal

    # ==== W1 deep analysis ====
    print("\n" + "=" * 72); print("W1"); print("=" * 72)
    percentile_table("W1", W1)
    per_row_amax_stats("W1", W1)
    ascii_histogram("W1", W1, bins=30, width=60)
    effective_bits_estimate("W1", W1)
    rate_distortion("W1", W1, Xcal_for_W1)
    fine_alpha_sweep("W1", W1, Xcal_for_W1, bits=4)

    # ==== W2 deep analysis ====
    print("\n" + "=" * 72); print("W2"); print("=" * 72)
    percentile_table("W2", W2)
    per_row_amax_stats("W2", W2)
    ascii_histogram("W2", W2, bins=30, width=60)
    effective_bits_estimate("W2", W2)
    rate_distortion("W2", W2, Xcal_for_W2)
    fine_alpha_sweep("W2", W2, Xcal_for_W2, bits=4)

    # ==== activation deep analysis ====
    print("\n" + "=" * 72); print("post-relu activation (calibration batch)")
    print("=" * 72)
    percentile_table("act_cal", act_cal)
    ascii_histogram("act_cal", act_cal, bins=30, width=60)
    effective_bits_estimate("act_cal", act_cal)


if __name__ == "__main__":
    main()
