"""Block C multi-seed float-to-plateau + clip-aware PTQ int4.

Pipeline:
  Phase A  — train each (variant, seed) to plateau (early-stop on test CE).
  Phase B1 — for each float champion, do clip-aware PTQ int4 on W1, W2 and
             the post-activation. For each tensor, grid-search the clip ratio
             α that minimizes output-MSE vs float (classic calibration PTQ).
  Phase C  — report mean ± std per variant at float and at int4.

"Smart rounding" here = clip-aware scale search, not naive RTN. If a variant
is naive-RTN worst-case (long activation tails, e.g. swish), the α grid will
find a smaller scale that keeps the bulk precise at the cost of saturating
outliers. This is the cheap tier before AdaRound / GPTQ.

Run:
    python3 tools/diag_block_c_multiseed_ptq.py \
        --corpus output/data/fineweb_edu_100mb.txt --max-bytes 1000000 \
        --out output/block_c_multiseed_ptq --seeds 5 --max-epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
TEST_FRAC = 0.1
PATIENCE = 1  # early-stop patience on test CE
def _default_clip_grid(bits: int) -> np.ndarray:
    """α grid density matched to what the int bitwidth can resolve. Range [0.3, 1.0]
    covers heavy-tailed weight distributions (empirically the narrower [0.5, 1.0]
    grid hit the α=0.50 floor on every tensor)."""
    n = max(8, min(2 ** bits, 33))
    return np.linspace(0.3, 1.0, n)
CALIB_BATCHES = 64  # batches to use for PTQ calibration

DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_multiseed_ptq"


# -------- activations --------

def act_relu(z):
    a = np.maximum(z, 0.0)
    return a, lambda g: g * (z > 0).astype(np.float32)


def act_tanh(z):
    a = np.tanh(z)
    return a, lambda g: g * (1.0 - a * a)


def act_gelu(z):
    c = np.sqrt(2.0 / np.pi).astype(np.float32); k = 0.044715
    u = c * (z + k * z ** 3); t = np.tanh(u)
    a = 0.5 * z * (1.0 + t)
    def grad(g):
        du_dz = c * (1.0 + 3.0 * k * z * z)
        return g * (0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * du_dz)
    return a, grad


def act_swish(z):
    s = 1.0 / (1.0 + np.exp(-z))
    a = z * s
    return a, lambda g: g * (s + z * s * (1.0 - s))


def act_beukers_single(z):
    denom = 1.0 + np.abs(z)
    a = z / denom
    return a, lambda g: g * (1.0 / (denom * denom))


def act_beukers_gate(z):
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]; y = z[..., H_out:]
    p = x * y; denom = 1.0 + np.abs(p)
    a = p / denom
    def grad(g):
        gp = g * (1.0 / (denom * denom))
        return np.concatenate([gp * y, gp * x], axis=-1)
    return a, grad


def act_swiglu(z):
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]; y = z[..., H_out:]
    s = 1.0 / (1.0 + np.exp(-y))
    a = x * s
    def grad(g):
        return np.concatenate([g * s, g * x * s * (1.0 - s)], axis=-1)
    return a, grad


VARIANTS = {
    "relu":           (act_relu,           H),
    "tanh":           (act_tanh,           H),
    "gelu":           (act_gelu,           H),
    "swish":          (act_swish,          H),
    "beukers_single": (act_beukers_single, H),
    "beukers_gate":   (act_beukers_gate,   2 * H),
    "swiglu":         (act_swiglu,         2 * H),
}


def make_windows(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    N = len(arr) - CONTEXT
    ctx = np.empty((N, CONTEXT), dtype=np.uint8)
    for i in range(CONTEXT):
        ctx[:, i] = arr[i : i + N]
    return ctx, arr[CONTEXT : CONTEXT + N].astype(np.int64)


def embed_context(ctx_bytes, lut):
    return lut[ctx_bytes].reshape(ctx_bytes.shape[0], -1)


def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m); p = e / e.sum(axis=1, keepdims=True)
    return p, float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def eval_model(W1, b1, W2, b2, act_fn, Xte, yte):
    z = Xte @ W1 + b1
    a, _ = act_fn(z)
    logits = a @ W2 + b2
    p, test_ce = softmax_ce(logits, yte)
    pred = logits.argmax(axis=1)
    return test_ce, float((pred == yte).mean())


def train_float_to_plateau(seed, act_fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, max_epochs):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, 256)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    Ntr = len(ytr)
    best_test_ce = float("inf")
    best_state = None
    bad_epochs = 0
    curve = []

    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xtr[idx]; yb = ytr[idx]; B = len(yb)
            z = xb @ W1 + b1
            a, grad_act = act_fn(z)
            logits = a @ W2 + b2
            p, loss = softmax_ce(logits, yb); losses.append(loss)
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW2 = a.T @ dlogits; db2 = dlogits.sum(axis=0)
            da = dlogits @ W2.T
            dz = grad_act(da)
            dW1 = xb.T @ dz; db1 = dz.sum(axis=0)
            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - LR * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - LR * db2; b2 += vb2

        test_ce, test_acc = eval_model(W1, b1, W2, b2, act_fn, Xte, yte)
        train_ce = float(np.mean(losses))
        curve.append({"epoch": epoch, "train_ce": train_ce,
                      "test_ce": test_ce, "test_acc": test_acc})
        if test_ce < best_test_ce - 1e-4:
            best_test_ce = test_ce
            best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > PATIENCE:
                break

    return best_state, curve


# -------- PTQ: clip-aware scale search --------

def _quantize(x, scale, qmax):
    q = np.round(x / scale).clip(-qmax - 1.0, qmax)
    return (q * scale).astype(np.float32)


def ptq_weight_by_output(W, X_calib, bits):
    """Pick α so that ||X_calib @ W - X_calib @ Wq||_F is minimized."""
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(W)))
    if amax == 0.0:
        return W.copy(), 1.0
    ref = X_calib @ W
    best_mse = float("inf"); best_alpha = 1.0; best_W = W
    for a in _default_clip_grid(bits):
        scale = (a * amax) / qmax
        Wq = _quantize(W, scale, qmax)
        out = X_calib @ Wq
        mse = float(np.mean((out - ref) ** 2))
        if mse < best_mse:
            best_mse = mse; best_alpha = float(a); best_W = Wq
    return best_W, best_alpha


def ptq_activation_scale(a_calib, bits):
    """Pick α for activation tensor so that ||a - aq||_F is minimized."""
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(a_calib)))
    if amax == 0.0:
        return 1.0
    best_mse = float("inf"); best_alpha = 1.0
    for a in _default_clip_grid(bits):
        scale = (a * amax) / qmax
        aq = _quantize(a_calib, scale, qmax)
        mse = float(np.mean((aq - a_calib) ** 2))
        if mse < best_mse:
            best_mse = mse; best_alpha = float(a)
    return best_alpha


def apply_activation_quant(a, alpha, bits):
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(a)))
    if amax == 0.0:
        return a
    scale = (alpha * amax) / qmax
    return _quantize(a, scale, qmax)


def eval_ptq(W1, b1, W2, b2, act_fn, Xcal, ycal, Xte, yte, bits):
    # 1. PTQ weights via output-MSE
    W1q, a1 = ptq_weight_by_output(W1, Xcal, bits)
    # build post-W1 activation on calibration set for W2 calibration + activation α
    z_cal = Xcal @ W1q + b1
    act_cal, _ = act_fn(z_cal)
    alpha_act = ptq_activation_scale(act_cal, bits)
    # quantize calib activations for W2 output-mse
    act_cal_q = apply_activation_quant(act_cal, alpha_act, bits)
    W2q, a2 = ptq_weight_by_output(W2, act_cal_q, bits)

    # 2. full forward at eval
    z = Xte @ W1q + b1
    a, _ = act_fn(z)
    aq = apply_activation_quant(a, alpha_act, bits)
    logits = aq @ W2q + b2
    p, test_ce = softmax_ce(logits, yte)
    pred = logits.argmax(axis=1)
    return test_ce, float((pred == yte).mean()), {
        "alpha_W1": a1, "alpha_W2": a2, "alpha_act": alpha_act,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Block C multi-seed + clip-aware PTQ int{args.bits} ==")
    print(f"Corpus: {args.corpus}  seeds: {args.seeds}  max_epochs: {args.max_epochs}")
    if args.max_bytes > 0:
        with args.corpus.open("rb") as f:
            raw = f.read(args.max_bytes)
    else:
        raw = args.corpus.read_bytes()
    print(f"Corpus bytes: {len(raw):,}")

    enc = ByteEncoder.load_default()
    lut = enc._lut_f32
    ctx_bytes, y = make_windows(raw)
    N = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_test = int(N * TEST_FRAC)
    te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    Xtr = embed_context(ctx_bytes[tr_idx], lut)
    Xte = embed_context(ctx_bytes[te_idx], lut)
    ytr = y[tr_idx]; yte = y[te_idx]
    in_dim = Xtr.shape[1]

    # Calibration set: first CALIB_BATCHES * BATCH windows of train (no leakage from test)
    n_cal = min(CALIB_BATCHES * BATCH, len(ytr))
    Xcal = Xtr[:n_cal]; ycal = ytr[:n_cal]
    print(f"Windows: train={len(tr_idx):,}  test={len(te_idx):,}  calib={n_cal:,}")
    print(f"Input dim: {in_dim}")

    all_results = []
    t_start = time.time()
    for name, (fn, pre_dim) in VARIANTS.items():
        print(f"\n=== {name}  (pre_dim={pre_dim}) ===")
        variant_runs = []
        for seed in range(args.seeds):
            t0 = time.time()
            state, curve = train_float_to_plateau(
                seed, fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, args.max_epochs)
            W1, b1, W2, b2 = state
            f_test_ce, f_test_acc = eval_model(W1, b1, W2, b2, fn, Xte, yte)
            q_test_ce, q_test_acc, alphas = eval_ptq(
                W1, b1, W2, b2, fn, Xcal, ycal, Xte, yte, args.bits)
            dt = time.time() - t0
            variant_runs.append({
                "seed": seed, "epochs_trained": len(curve),
                "float_test_ce": f_test_ce, "float_test_acc": f_test_acc,
                f"int{args.bits}_test_ce": q_test_ce,
                f"int{args.bits}_test_acc": q_test_acc,
                "ptq_alphas": alphas, "seconds": dt, "curve": curve,
            })
            print(f"  seed={seed}  ep={len(curve)}  "
                  f"float={f_test_acc*100:.2f}%  int{args.bits}={q_test_acc*100:.2f}%  "
                  f"αs=W1:{alphas['alpha_W1']:.2f} W2:{alphas['alpha_W2']:.2f} "
                  f"act:{alphas['alpha_act']:.2f}  t={dt:.1f}s", flush=True)

        f_accs = np.array([r["float_test_acc"] for r in variant_runs])
        q_accs = np.array([r[f"int{args.bits}_test_acc"] for r in variant_runs])
        all_results.append({
            "name": name, "pre_dim": pre_dim, "runs": variant_runs,
            "float_acc_mean": float(f_accs.mean()),
            "float_acc_std": float(f_accs.std()),
            f"int{args.bits}_acc_mean": float(q_accs.mean()),
            f"int{args.bits}_acc_std": float(q_accs.std()),
            f"int{args.bits}_drop_mean": float((f_accs - q_accs).mean()),
        })
        r = all_results[-1]
        print(f"  >> {name}: float {r['float_acc_mean']*100:.2f}% ± {r['float_acc_std']*100:.2f}  "
              f"int{args.bits} {r[f'int{args.bits}_acc_mean']*100:.2f}% ± "
              f"{r[f'int{args.bits}_acc_std']*100:.2f}  "
              f"drop {r[f'int{args.bits}_drop_mean']*100:.2f}%")

    dt_total = time.time() - t_start
    print(f"\nTotal wall time: {dt_total/60:.1f} min")

    # Ranking by int{bits} mean acc
    all_results.sort(key=lambda r: -r[f"int{args.bits}_acc_mean"])
    print(f"\n== Ranking by int{args.bits} mean acc ==")
    for r in all_results:
        print(f"  {r['name']:16s}  float={r['float_acc_mean']*100:5.2f}±{r['float_acc_std']*100:.2f}  "
              f"int{args.bits}={r[f'int{args.bits}_acc_mean']*100:5.2f}±{r[f'int{args.bits}_acc_std']*100:.2f}  "
              f"drop={r[f'int{args.bits}_drop_mean']*100:.2f}%")

    out_path = out_dir / f"results_int{args.bits}.json"
    out_path.write_text(json.dumps({
        "corpus": str(args.corpus), "bits": args.bits, "seeds": args.seeds,
        "max_epochs": args.max_epochs, "patience": PATIENCE,
        "clip_grid": _default_clip_grid(args.bits).tolist(),
        "context": CONTEXT, "hidden": H,
        "lr": LR, "momentum": MOMENTUM, "batch": BATCH,
        "corpus_bytes": len(raw), "train_windows": int(len(tr_idx)),
        "test_windows": int(len(te_idx)), "calib_windows": n_cal,
        "input_dim": in_dim, "total_seconds": dt_total,
        "results": all_results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
