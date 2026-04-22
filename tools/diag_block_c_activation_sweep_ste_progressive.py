"""Block C STE sweep — progressive (one-by-one) quantization.

Motivation: all-at-once int4 QAT diverges or underperforms across all 7
activation variants (beukers_gate dropped from 45.33% float to 32.70% int4,
worse: train loss grows epoch-over-epoch). Classic QAT fix: ramp up the
quantization one tensor at a time, with a decaying learning rate.

Phases (each = 1 epoch):
  P0  float warmup       no quant        LR = base_lr
  P1  +W1 quant          W1q only        LR = base_lr / 2
  P2  +W2 quant          W1q + W2q       LR = base_lr / 4
  P3  +activation quant  W1q + W2q + aq  LR = base_lr / 8
  P4  warmdown           all on          LR = base_lr / 16

Eval always uses the fully-quantized forward (W1q + W2q + aq).

Run:
    python3 tools/diag_block_c_activation_sweep_ste_progressive.py \
        --corpus output/data/fineweb_edu_100mb.txt --max-bytes 5000000 \
        --out output/block_c_activation_sweep_ste_progressive_fineweb \
        --bits 4 --base-lr 0.02
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
MOMENTUM = 0.9
BATCH = 256
SEED = 1337
TEST_FRAC = 0.1

DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_activation_sweep_ste_progressive"


def fake_quant(x: np.ndarray, bits: int) -> np.ndarray:
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax == 0.0:
        return x.copy()
    scale = amax / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax - 1.0, qmax)
    return (q * scale).astype(np.float32)


def identity(x: np.ndarray, bits: int) -> np.ndarray:  # noqa: ARG001
    return x


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


PHASES = [
    # (name, quant_W1, quant_W2, quant_act, lr_scale)
    ("P0_float",    False, False, False, 1.0),
    ("P1_W1",       True,  False, False, 0.5),
    ("P2_W1W2",     True,  True,  False, 0.25),
    ("P3_W1W2A",    True,  True,  True,  0.125),
    ("P4_warmdown", True,  True,  True,  0.0625),
]


def train_one(name, act_fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, bits, base_lr):
    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, 256)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    curve = []
    t0 = time.time()

    for phase_name, qW1, qW2, qA, lr_scale in PHASES:
        lr = base_lr * lr_scale
        Ntr = len(ytr)
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xtr[idx]; yb = ytr[idx]; B = len(yb)

            W1f = fake_quant(W1, bits) if qW1 else W1
            W2f = fake_quant(W2, bits) if qW2 else W2

            z = xb @ W1f + b1
            a, grad_act = act_fn(z)
            af = fake_quant(a, bits) if qA else a
            logits = af @ W2f + b2
            p, loss = softmax_ce(logits, yb); losses.append(loss)

            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW2 = af.T @ dlogits
            db2 = dlogits.sum(axis=0)
            da = dlogits @ W2f.T   # STE through fake_quant(W2) and fake_quant(a)
            dz = grad_act(da)
            dW1 = xb.T @ dz
            db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - lr * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - lr * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - lr * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - lr * db2; b2 += vb2

        # Eval at full quant regardless of phase, so curves are comparable.
        W1q = fake_quant(W1, bits); W2q = fake_quant(W2, bits)
        z = Xte @ W1q + b1
        a, _ = act_fn(z)
        aq = fake_quant(a, bits)
        logits = aq @ W2q + b2
        _, test_ce = softmax_ce(logits, yte)
        pred = logits.argmax(axis=1)
        test_acc = float((pred == yte).mean())
        train_ce = float(np.mean(losses))
        curve.append({"phase": phase_name, "lr": lr, "qW1": qW1, "qW2": qW2, "qA": qA,
                      "train_ce": train_ce, "test_ce": test_ce, "test_acc": test_acc})
        print(f"  [{name:16s}][int{bits}] {phase_name:12s} lr={lr:.5f}  "
              f"train_ce={train_ce:.4f}  test_ce_q={test_ce:.4f}  "
              f"test_acc_q={test_acc*100:.2f}%", flush=True)

    dt = time.time() - t0
    return {"name": name, "bits": bits, "pre_dim": pre_dim, "base_lr": base_lr,
            "param_count": int(W1.size + b1.size + W2.size + b2.size),
            "seconds": dt, "curve": curve,
            "final_test_ce": curve[-1]["test_ce"],
            "final_test_acc": curve[-1]["test_acc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--base-lr", type=float, default=0.02,
                    help="LR at phase 0; halved each phase.")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Block C STE sweep — PROGRESSIVE (int{args.bits}, base_lr={args.base_lr}) ==")
    print(f"Corpus: {args.corpus}")
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
    rng = np.random.default_rng(SEED); perm = rng.permutation(N)
    n_test = int(N * TEST_FRAC)
    te_idx = perm[:n_test]; tr_idx = perm[n_test:]
    print(f"Windows: train={len(tr_idx)}  test={len(te_idx)}  (context={CONTEXT})")

    Xtr = embed_context(ctx_bytes[tr_idx], lut)
    Xte = embed_context(ctx_bytes[te_idx], lut)
    ytr = y[tr_idx]; yte = y[te_idx]
    in_dim = Xtr.shape[1]

    results = []
    for name, (fn, pre_dim) in VARIANTS.items():
        print(f"\n-- {name}  (pre_dim={pre_dim}, int{args.bits}) --")
        r = train_one(name, fn, pre_dim, Xtr, ytr, Xte, yte, in_dim,
                      bits=args.bits, base_lr=args.base_lr)
        results.append(r)

    results.sort(key=lambda r: r["final_test_ce"])
    print(f"\n== Ranking (progressive int{args.bits}, lower CE better) ==")
    for r in results:
        print(f"  {r['name']:16s}  test_ce={r['final_test_ce']:.4f}  "
              f"test_acc={r['final_test_acc']*100:.2f}%  "
              f"params={r['param_count']:,}  time={r['seconds']:.1f}s")

    out_path = out_dir / f"results_int{args.bits}.json"
    out_path.write_text(json.dumps({
        "corpus": str(args.corpus), "bits": args.bits, "base_lr": args.base_lr,
        "phases": [p[0] for p in PHASES],
        "context": CONTEXT, "hidden": H, "momentum": MOMENTUM, "batch": BATCH,
        "test_frac": TEST_FRAC, "corpus_bytes": len(raw),
        "train_windows": int(len(tr_idx)), "test_windows": int(len(te_idx)),
        "input_dim": in_dim, "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
