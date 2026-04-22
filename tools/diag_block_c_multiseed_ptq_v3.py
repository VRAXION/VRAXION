"""Block C multi-seed float-to-plateau + smart PTQ int4 (v3).

v3 changes over v2:
  * per-channel (output-axis) weight scale — each output column of W gets its
    own scale. Deep analysis showed per-row amax varies 9× for W1, so a single
    global scale wastes 13/15 levels on small rows.
  * symmetric quantization [-qmax, qmax] instead of [-qmax-1, qmax] — the
    weight distribution is essentially symmetric, the extra negative slot
    introduces a systematic bias.
  * unsigned quantization [0, 2^bits - 1] for relu activations — 75% of relu
    output is zero, so the negative half of signed int4 is literally wasted.
    Effectively turns uint4 into int5 resolution for one-sided activations.
  * C19 added as an 8th variant (fixed c=1.0, rho=1.0) for reference.
    Expected behaviour: strong float, heavy PTQ drop — confirms B-block finding.

The α grid is still 16 points on [0.1, 1.0]. Per-channel scale moves outliers
out of the game, so a wider grid is less important here.

Run:
    python3 tools/diag_block_c_multiseed_ptq_v3.py \
        --corpus output/data/fineweb_edu_100mb.txt --max-bytes 1000000 \
        --out output/block_c_multiseed_ptq_v3_fineweb --seeds 5 --max-epochs 5 --bits 4
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
PATIENCE = 1
CALIB_BATCHES = 64

DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_multiseed_ptq_v3"


def _default_clip_grid(bits: int) -> np.ndarray:
    n = max(8, min(2 ** bits, 33))
    return np.linspace(0.1, 1.0, n)


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


def act_c19_parametric(z, c_vec, rho_vec):
    """C19 piecewise-quadratic periodic activation, per-channel learnable c, rho.

    Mirrors Python/block_b_merger/merger.py::_c19. c and rho are shape (pre_dim,).

    Returns (a, grad_fn) where grad_fn(g_out) = (grad_z, grad_c, grad_rho).
    Gradients wrt c, rho use the clamped effective values: grad is zero on
    the clamp boundary (c < 0.1 or rho < 0).
    """
    # Clamp (straight-through: gradient is zero where the clamp is active).
    c_eff = np.maximum(c_vec, np.float32(0.1))
    rho_eff = np.maximum(rho_vec, np.float32(0.0))
    c_active = (c_vec >= np.float32(0.1)).astype(np.float32)
    rho_active = (rho_vec >= np.float32(0.0)).astype(np.float32)

    L = np.float32(6.0) * c_eff                 # (pre_dim,)
    scaled = z / c_eff                          # (B, pre_dim)
    n_floor = np.floor(scaled).astype(np.float32)
    t = scaled - n_floor
    h = t * (1.0 - t)
    sgn = np.where((n_floor.astype(np.int64) % 2) == 0,
                   np.float32(1.0), np.float32(-1.0))

    interior = c_eff * (sgn * h + rho_eff * h * h)
    tail_pos = (z >= L)
    tail_neg = (z <= -L)
    tail = tail_pos | tail_neg

    a = np.where(tail_pos, z - L,
          np.where(tail_neg, z + L, interior)).astype(np.float32)

    def grad(g_out):
        # Interior derivative wrt z:  (sgn + 2 rho h) * (1 - 2t)
        # Tail derivative wrt z: 1.
        dz_interior = (sgn + 2.0 * rho_eff * h) * (1.0 - 2.0 * t)
        dz = np.where(tail, np.float32(1.0), dz_interior.astype(np.float32))
        grad_z = g_out * dz

        # Gradient wrt c:
        #   interior: y = c (sgn h + rho h^2);  dy/dc = g - (sgn + 2 rho h)(1-2t)(n+t)
        #             where g = sgn h + rho h^2 and t := scaled - n, so x/c = n+t
        #   tail_pos : y = z - 6c   -> dy/dc = -6
        #   tail_neg : y = z + 6c   -> dy/dc = +6
        g_val = sgn * h + rho_eff * h * h
        dc_interior = g_val - (sgn + 2.0 * rho_eff * h) * (1.0 - 2.0 * t) * (n_floor + t)
        dc = np.where(tail_pos, np.float32(-6.0),
                np.where(tail_neg, np.float32(+6.0),
                         dc_interior.astype(np.float32)))
        grad_c_per_channel = (g_out * dc).sum(axis=0) * c_active

        # Gradient wrt rho:
        #   interior: dy/drho = c * h^2
        #   tail: 0
        drho_interior = c_eff * h * h
        drho = np.where(tail, np.float32(0.0), drho_interior.astype(np.float32))
        grad_rho_per_channel = (g_out * drho).sum(axis=0) * rho_active

        return grad_z, grad_c_per_channel, grad_rho_per_channel

    return a, grad


def _c19_closure(c_vec, rho_vec):
    """Bind (c, rho) into an act_fn with the standard (a, grad_z) interface."""
    def fn(z):
        a, grad_full = act_c19_parametric(z, c_vec, rho_vec)
        # Wrap grad so callers that only consume grad_z keep working.
        def grad(g_out):
            gz, _, _ = grad_full(g_out)
            return gz
        return a, grad
    return fn


# variant name -> (forward, pre_activation_dim, one_sided?, parametric?)
# one_sided == True  -> unsigned activation quant works (bulk in positive half)
# parametric == True -> activation has learnable per-channel c, rho (only c19 today)
VARIANTS = {
    "relu":           (act_relu,           H,     True,  False),
    "tanh":           (act_tanh,           H,     False, False),
    "gelu":           (act_gelu,           H,     False, False),
    "swish":          (act_swish,          H,     False, False),
    "beukers_single": (act_beukers_single, H,     False, False),
    "beukers_gate":   (act_beukers_gate,   2 * H, False, False),
    "swiglu":         (act_swiglu,         2 * H, False, False),
    "c19":            (None,               H,     False, True),   # built per-run from c,rho
}

C19_INIT_C = 1.0
C19_INIT_RHO = 0.5
C19_LR_MULT = 0.1  # c/rho learn on a slower time-scale than the weights


def c19_init_params(pre_dim: int):
    c = np.full(pre_dim, C19_INIT_C, dtype=np.float32)
    rho = np.full(pre_dim, C19_INIT_RHO, dtype=np.float32)
    return c, rho


# -------- data --------

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


# -------- float training --------

def train_float_to_plateau(seed, act_fn_fixed, pre_dim, Xtr, ytr, Xte, yte,
                           in_dim, max_epochs, parametric=False):
    """If parametric=True, we're training c19 with learnable c, rho per-channel.

    act_fn_fixed is ignored in that case; we build the act_fn each batch from
    the current c, rho via _c19_closure (for eval) and directly via
    act_c19_parametric (for training, so we capture all three gradients).
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, 256)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    if parametric:
        c_vec, rho_vec = c19_init_params(pre_dim)
        vc = np.zeros_like(c_vec); vrho = np.zeros_like(rho_vec)
    else:
        c_vec = rho_vec = vc = vrho = None

    Ntr = len(ytr)
    best_test_ce = float("inf"); best_state = None; bad = 0
    curve = []

    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xtr[idx]; yb = ytr[idx]; B = len(yb)
            z = xb @ W1 + b1

            if parametric:
                a, grad_full = act_c19_parametric(z, c_vec, rho_vec)
            else:
                a, grad_act = act_fn_fixed(z)

            logits = a @ W2 + b2
            p, loss = softmax_ce(logits, yb); losses.append(loss)
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW2 = a.T @ dlogits; db2 = dlogits.sum(axis=0)
            da = dlogits @ W2.T

            if parametric:
                dz, dc_batch, drho_batch = grad_full(da)
            else:
                dz = grad_act(da); dc_batch = drho_batch = None

            dW1 = xb.T @ dz; db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - LR * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - LR * db2; b2 += vb2

            if parametric:
                vc = MOMENTUM * vc - (LR * C19_LR_MULT) * dc_batch
                vrho = MOMENTUM * vrho - (LR * C19_LR_MULT) * drho_batch
                c_vec += vc
                rho_vec += vrho
                # Clamp so future forward passes are stable (c>=0.1, rho>=0).
                np.maximum(c_vec, np.float32(0.1), out=c_vec)
                np.maximum(rho_vec, np.float32(0.0), out=rho_vec)

        if parametric:
            act_fn_eval = _c19_closure(c_vec, rho_vec)
        else:
            act_fn_eval = act_fn_fixed
        test_ce, test_acc = eval_model(W1, b1, W2, b2, act_fn_eval, Xte, yte)
        train_ce = float(np.mean(losses))
        extra_str = ""
        if parametric:
            extra_str = f"  c_μ={c_vec.mean():.3f}  rho_μ={rho_vec.mean():.3f}"
        curve.append({"epoch": epoch, "train_ce": train_ce,
                      "test_ce": test_ce, "test_acc": test_acc})
        if test_ce < best_test_ce - 1e-4:
            best_test_ce = test_ce
            if parametric:
                best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                              c_vec.copy(), rho_vec.copy())
            else:
                best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break

    return best_state, curve


# -------- PTQ: per-channel symmetric weights, unsigned/signed activations --------

def _sym_quantize(x, scale, qmax):
    # Symmetric: clip to [-qmax, qmax] (one fewer level than asym [-qmax-1, qmax]
    # but cleaner for symmetric weight distributions).
    q = np.round(x / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def _unsigned_quantize(x, scale, qmax_unsigned, zero_point=0.0):
    # Unsigned: clip to [0, qmax_unsigned]. Caller must pass nonnegative x.
    q = np.round((x - zero_point) / scale).clip(0.0, qmax_unsigned)
    return (q * scale + zero_point).astype(np.float32)


def ptq_weight_per_channel(W, X_calib, bits, clip_grid):
    """Per-output-channel scale, symmetric. α grid-searched per channel.

    Vectorised: for each α in the grid we quantise every column in one shot and
    evaluate all columns' MSEs with a single matmul. So total cost is
    `len(grid)` matmuls, regardless of how many channels there are.

    Returns (Wq, alphas, scales) with alphas / scales of shape (out,).
    """
    qmax = float(2 ** (bits - 1) - 1)
    out_dim = W.shape[1]
    col_amax = np.max(np.abs(W), axis=0)          # (out,)
    zero_cols = col_amax == 0.0

    ref = X_calib @ W                              # (B, out)
    best_mse = np.full(out_dim, np.inf, dtype=np.float32)
    best_alpha = np.ones(out_dim, dtype=np.float32)
    best_Wq_cols = W.copy()

    for a in clip_grid:
        scales = (float(a) * col_amax) / qmax       # (out,)
        safe_scales = np.where(scales == 0, np.float32(1.0), scales)
        q = np.round(W / safe_scales).clip(-qmax, qmax)
        Wq_alpha = (q * safe_scales).astype(np.float32)
        Wq_alpha[:, zero_cols] = 0.0                # keep zero-mass cols at zero
        out_alpha = X_calib @ Wq_alpha              # (B, out)
        mse = np.mean((out_alpha - ref) ** 2, axis=0)  # (out,)
        better = mse < best_mse
        best_mse[better] = mse[better]
        best_alpha[better] = float(a)
        # Copy the columns that improved into the running champion.
        cols_to_replace = np.where(better)[0]
        if cols_to_replace.size:
            best_Wq_cols[:, cols_to_replace] = Wq_alpha[:, cols_to_replace]

    scales_out = (best_alpha * col_amax) / qmax
    scales_out = np.where(col_amax == 0.0, np.float32(1.0), scales_out).astype(np.float32)
    return best_Wq_cols.astype(np.float32), best_alpha.astype(np.float32), scales_out


def ptq_activation_alpha(a_calib, bits, one_sided, clip_grid):
    """Find α that minimises ||a - aq||. Returns (alpha, is_unsigned).

    For one-sided activations (e.g. ReLU, gelu, swish) we use UNSIGNED quant,
    which doubles the effective resolution on the positive half.
    """
    if one_sided:
        qmax_u = float(2 ** bits - 1)  # e.g. 15 for int4 unsigned
        amax = float(np.max(a_calib))  # positive side only; assume >= 0
        if amax == 0.0:
            return 1.0, True
        best_mse = float("inf"); best_alpha = 1.0
        for a in clip_grid:
            scale = (a * amax) / qmax_u
            aq = _unsigned_quantize(a_calib, scale, qmax_u)
            mse = float(np.mean((aq - a_calib) ** 2))
            if mse < best_mse:
                best_mse = mse; best_alpha = float(a)
        return best_alpha, True
    else:
        qmax = float(2 ** (bits - 1) - 1)
        amax = float(np.max(np.abs(a_calib)))
        if amax == 0.0:
            return 1.0, False
        best_mse = float("inf"); best_alpha = 1.0
        for a in clip_grid:
            scale = (a * amax) / qmax
            aq = _sym_quantize(a_calib, scale, qmax)
            mse = float(np.mean((aq - a_calib) ** 2))
            if mse < best_mse:
                best_mse = mse; best_alpha = float(a)
        return best_alpha, False


def apply_activation_quant(a, alpha, bits, one_sided):
    if one_sided:
        qmax_u = float(2 ** bits - 1)
        amax = float(np.max(a)) if a.size else 0.0
        if amax == 0.0:
            return a
        scale = (alpha * amax) / qmax_u
        return _unsigned_quantize(a, scale, qmax_u)
    else:
        qmax = float(2 ** (bits - 1) - 1)
        amax = float(np.max(np.abs(a))) if a.size else 0.0
        if amax == 0.0:
            return a
        scale = (alpha * amax) / qmax
        return _sym_quantize(a, scale, qmax)


def eval_ptq(W1, b1, W2, b2, act_fn, one_sided, Xcal, Xte, yte, bits):
    """Per-channel PTQ eval. act_fn is already the live forward (C19 closure
    or fixed variant), so this function doesn't need to know about C19."""
    clip_grid = _default_clip_grid(bits)
    # Step 1: per-channel PTQ of W1 using raw input
    W1q, alphas_W1, scales_W1 = ptq_weight_per_channel(W1, Xcal, bits, clip_grid)
    # Step 2: calibrate activation α using W1q
    z_cal = Xcal @ W1q + b1
    act_cal, _ = act_fn(z_cal)
    alpha_act, act_unsigned = ptq_activation_alpha(act_cal, bits, one_sided, clip_grid)
    act_cal_q = apply_activation_quant(act_cal, alpha_act, bits, act_unsigned)
    # Step 3: per-channel PTQ of W2 using the quantized calibration activation
    W2q, alphas_W2, scales_W2 = ptq_weight_per_channel(W2, act_cal_q, bits, clip_grid)

    # Final eval
    z = Xte @ W1q + b1
    a, _ = act_fn(z)
    aq = apply_activation_quant(a, alpha_act, bits, act_unsigned)
    logits = aq @ W2q + b2
    p, test_ce = softmax_ce(logits, yte)
    pred = logits.argmax(axis=1)
    return test_ce, float((pred == yte).mean()), {
        "W1_alpha_mean": float(alphas_W1.mean()),
        "W1_alpha_median": float(np.median(alphas_W1)),
        "W2_alpha_mean": float(alphas_W2.mean()),
        "W2_alpha_median": float(np.median(alphas_W2)),
        "act_alpha": alpha_act,
        "act_unsigned": act_unsigned,
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

    print(f"== Block C multi-seed + PER-CHANNEL PTQ int{args.bits} (v3) ==")
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

    n_cal = min(CALIB_BATCHES * BATCH, len(ytr))
    Xcal = Xtr[:n_cal]
    print(f"Windows: train={len(tr_idx):,}  test={len(te_idx):,}  calib={n_cal:,}")

    all_results = []
    t_start = time.time()
    for name, (fn, pre_dim, one_sided, parametric) in VARIANTS.items():
        print(f"\n=== {name}  (pre_dim={pre_dim}, one_sided_act={one_sided}, "
              f"parametric={parametric}) ===")
        runs = []
        for seed in range(args.seeds):
            t0 = time.time()
            state, curve = train_float_to_plateau(
                seed, fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, args.max_epochs,
                parametric=parametric)
            if parametric:
                W1, b1, W2, b2, c_vec, rho_vec = state
                live_fn = _c19_closure(c_vec, rho_vec)
                c19_params = {"c_mean": float(c_vec.mean()),
                              "c_std": float(c_vec.std()),
                              "rho_mean": float(rho_vec.mean()),
                              "rho_std": float(rho_vec.std())}
            else:
                W1, b1, W2, b2 = state
                live_fn = fn
                c19_params = None

            f_ce, f_acc = eval_model(W1, b1, W2, b2, live_fn, Xte, yte)
            q_ce, q_acc, alphas = eval_ptq(
                W1, b1, W2, b2, live_fn, one_sided, Xcal, Xte, yte, args.bits)
            dt = time.time() - t0
            run_rec = {
                "seed": seed, "epochs_trained": len(curve),
                "float_test_ce": f_ce, "float_test_acc": f_acc,
                f"int{args.bits}_test_ce": q_ce,
                f"int{args.bits}_test_acc": q_acc,
                "ptq_alphas": alphas, "seconds": dt, "curve": curve,
            }
            if c19_params is not None:
                run_rec["c19_params"] = c19_params
            runs.append(run_rec)
            extra = ""
            if c19_params is not None:
                extra = f"  c_μ={c19_params['c_mean']:.2f}  rho_μ={c19_params['rho_mean']:.2f}"
            print(f"  seed={seed}  ep={len(curve)}  "
                  f"float={f_acc*100:.2f}%  int{args.bits}={q_acc*100:.2f}%  "
                  f"αW1_μ={alphas['W1_alpha_mean']:.2f}  "
                  f"αW2_μ={alphas['W2_alpha_mean']:.2f}  "
                  f"α_act={alphas['act_alpha']:.2f}"
                  f"{'_u' if alphas['act_unsigned'] else '_s'}{extra}  t={dt:.1f}s",
                  flush=True)

        f_accs = np.array([r["float_test_acc"] for r in runs])
        q_accs = np.array([r[f"int{args.bits}_test_acc"] for r in runs])
        summary = {
            "name": name, "pre_dim": pre_dim, "one_sided_act": one_sided,
            "runs": runs,
            "float_acc_mean": float(f_accs.mean()),
            "float_acc_std": float(f_accs.std()),
            f"int{args.bits}_acc_mean": float(q_accs.mean()),
            f"int{args.bits}_acc_std": float(q_accs.std()),
            f"int{args.bits}_drop_mean": float((f_accs - q_accs).mean()),
        }
        all_results.append(summary)
        print(f"  >> {name}: float {summary['float_acc_mean']*100:.2f}% "
              f"± {summary['float_acc_std']*100:.2f}  "
              f"int{args.bits} {summary[f'int{args.bits}_acc_mean']*100:.2f}% "
              f"± {summary[f'int{args.bits}_acc_std']*100:.2f}  "
              f"drop {summary[f'int{args.bits}_drop_mean']*100:.2f}%")

    dt_total = time.time() - t_start
    all_results.sort(key=lambda r: -r[f"int{args.bits}_acc_mean"])
    print(f"\nTotal wall time: {dt_total/60:.1f} min")
    print(f"\n== Ranking by int{args.bits} mean acc (v3: per-channel + unsigned act) ==")
    for r in all_results:
        print(f"  {r['name']:16s}  "
              f"float={r['float_acc_mean']*100:5.2f}±{r['float_acc_std']*100:.2f}  "
              f"int{args.bits}={r[f'int{args.bits}_acc_mean']*100:5.2f}"
              f"±{r[f'int{args.bits}_acc_std']*100:.2f}  "
              f"drop={r[f'int{args.bits}_drop_mean']*100:.2f}%")

    out_path = out_dir / f"results_int{args.bits}.json"
    out_path.write_text(json.dumps({
        "corpus": str(args.corpus), "bits": args.bits, "seeds": args.seeds,
        "max_epochs": args.max_epochs, "patience": PATIENCE,
        "clip_grid": _default_clip_grid(args.bits).tolist(),
        "context": CONTEXT, "hidden": H, "lr": LR, "momentum": MOMENTUM,
        "batch": BATCH, "corpus_bytes": len(raw),
        "train_windows": int(len(tr_idx)), "test_windows": int(len(te_idx)),
        "calib_windows": n_cal, "input_dim": in_dim, "total_seconds": dt_total,
        "quant_scheme": "per_channel_symmetric_weights__unsigned_activation_for_one_sided",
        "results": all_results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
