"""Block C activation sweep — next-byte prediction from 8-byte context.

Fair isolation test for choosing the Block C neural activation function.

Setup:
  - Corpus: alice_corpus.txt (100 KB).
  - Input: 8 consecutive bytes -> Block A LUT embedding -> flat 128-dim vector.
  - Hidden: H=128 units, single layer, swept activation.
  - Output: 256-way softmax (predict the 9th byte).
  - Loss: cross-entropy. Metric: test top-1 accuracy + CE.
  - Optimizer: SGD + momentum, shared hyperparameters across all variants.
  - Parameter budget equalized: GLU-style variants get 2H pre-activation so
    first-layer param count matches standard (in * 2H total either way).

Run:
    python3 tools/diag_block_c_activation_sweep.py
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
EPOCHS = 3
BATCH = 256
SEED = 1337
TEST_FRAC = 0.1

DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_activation_sweep"


def make_windows(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    N = len(arr) - CONTEXT
    ctx = np.empty((N, CONTEXT), dtype=np.uint8)
    for i in range(CONTEXT):
        ctx[:, i] = arr[i : i + N]
    y = arr[CONTEXT : CONTEXT + N].astype(np.int64)
    return ctx, y


def embed_context(ctx_bytes: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # ctx_bytes: (B, CONTEXT) uint8 -> (B, CONTEXT*16) float32
    return lut[ctx_bytes].reshape(ctx_bytes.shape[0], -1)


# -------- activations (forward + grad wrt pre-activation) --------

def act_relu(z):
    a = np.maximum(z, 0.0)
    def grad(g_out):
        return g_out * (z > 0).astype(np.float32)
    return a, grad


def act_tanh(z):
    a = np.tanh(z)
    def grad(g_out):
        return g_out * (1.0 - a * a)
    return a, grad


def act_gelu(z):
    # tanh approximation
    c = np.sqrt(2.0 / np.pi).astype(np.float32)
    k = 0.044715
    u = c * (z + k * z ** 3)
    t = np.tanh(u)
    a = 0.5 * z * (1.0 + t)
    def grad(g_out):
        du_dz = c * (1.0 + 3.0 * k * z * z)
        dt_du = 1.0 - t * t
        da_dz = 0.5 * (1.0 + t) + 0.5 * z * dt_du * du_dz
        return g_out * da_dz
    return a, grad


def act_swish(z):
    s = 1.0 / (1.0 + np.exp(-z))
    a = z * s
    def grad(g_out):
        return g_out * (s + z * s * (1.0 - s))
    return a, grad


def act_beukers_single(z):
    # x / (1 + |x|), single-input cousin of the Beukers gate
    abs_z = np.abs(z)
    denom = 1.0 + abs_z
    a = z / denom
    def grad(g_out):
        # d/dz [z / (1+|z|)] = 1 / (1+|z|)^2  (away from 0; |z|'s kink cancels)
        return g_out * (1.0 / (denom * denom))
    return a, grad


def act_beukers_gate(z):
    # z has last dim = 2H ; split (x, y); a = xy/(1+|xy|) ; output dim = H
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]
    y = z[..., H_out:]
    p = x * y
    abs_p = np.abs(p)
    denom = 1.0 + abs_p
    a = p / denom
    def grad(g_out):
        # da/dp = 1 / (1+|p|)^2
        da_dp = 1.0 / (denom * denom)
        gp = g_out * da_dp
        gx = gp * y
        gy = gp * x
        return np.concatenate([gx, gy], axis=-1)
    return a, grad


def act_swiglu(z):
    H_out = z.shape[-1] // 2
    x = z[..., :H_out]
    y = z[..., H_out:]
    s = 1.0 / (1.0 + np.exp(-y))
    a = x * s
    def grad(g_out):
        gx = g_out * s
        gy = g_out * x * s * (1.0 - s)
        return np.concatenate([gx, gy], axis=-1)
    return a, grad


VARIANTS = {
    "relu":            (act_relu,            H),
    "tanh":            (act_tanh,            H),
    "gelu":            (act_gelu,            H),
    "swish":           (act_swish,           H),
    "beukers_single":  (act_beukers_single,  H),
    "beukers_gate":    (act_beukers_gate,    2 * H),  # outputs H
    "swiglu":          (act_swiglu,          2 * H),  # outputs H
}


def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    ll = -np.log(p[np.arange(len(y)), y] + 1e-12)
    return p, ll.mean()


def train_one(name, act_fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, epochs=EPOCHS):
    rng = np.random.default_rng(SEED)
    # He init scaled for stability
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    # output layer: activation output dim = H for all variants (GLU halves)
    out_dim_hidden = H  # by construction
    W2 = rng.normal(0.0, np.sqrt(2.0 / out_dim_hidden), size=(out_dim_hidden, 256)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)

    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    Ntr = len(ytr)
    curve = []
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xtr[idx]; yb = ytr[idx]
            B = len(yb)

            z = xb @ W1 + b1
            a, grad_act = act_fn(z)
            logits = a @ W2 + b2
            p, loss = softmax_ce(logits, yb)
            losses.append(loss)

            dlogits = p.copy()
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            dW2 = a.T @ dlogits
            db2 = dlogits.sum(axis=0)
            da = dlogits @ W2.T
            dz = grad_act(da)
            dW1 = xb.T @ dz
            db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - LR * dW1
            vb1 = MOMENTUM * vb1 - LR * db1
            vW2 = MOMENTUM * vW2 - LR * dW2
            vb2 = MOMENTUM * vb2 - LR * db2
            W1 += vW1; b1 += vb1
            W2 += vW2; b2 += vb2

        # epoch eval
        z = Xte @ W1 + b1
        a, _ = act_fn(z)
        logits = a @ W2 + b2
        p, test_ce = softmax_ce(logits, yte)
        pred = logits.argmax(axis=1)
        test_acc = float((pred == yte).mean())
        train_ce = float(np.mean(losses))
        curve.append({"epoch": epoch + 1, "train_ce": train_ce,
                      "test_ce": float(test_ce), "test_acc": test_acc})
        print(f"  [{name:16s}] epoch {epoch+1}/{EPOCHS}  "
              f"train_ce={train_ce:.4f}  test_ce={test_ce:.4f}  test_acc={test_acc*100:.2f}%",
              flush=True)

    dt = time.time() - t0
    return {"name": name, "pre_dim": pre_dim, "out_dim": out_dim_hidden,
            "param_count": int(W1.size + b1.size + W2.size + b2.size),
            "epochs": epochs, "seconds": dt, "curve": curve,
            "final_test_ce": curve[-1]["test_ce"],
            "final_test_acc": curve[-1]["test_acc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help="Path to UTF-8 text corpus.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output directory for results.json.")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="Cap corpus read size in bytes (0 = no cap).")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs
    print("== Block C activation sweep ==")
    print(f"Corpus: {args.corpus}")
    if args.max_bytes > 0:
        with args.corpus.open("rb") as f:
            raw = f.read(args.max_bytes)
    else:
        raw = args.corpus.read_bytes()
    print(f"Corpus bytes: {len(raw):,}")

    print("Loading Block A LUT for input embedding...")
    enc = ByteEncoder.load_default()
    lut = enc._lut_f32  # (256, 16) float32 — public via encode_bytes, but direct matrix is faster

    ctx_bytes, y = make_windows(raw)
    N = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_test = int(N * TEST_FRAC)
    te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    print(f"Windows: train={len(tr_idx)}  test={len(te_idx)}  (context={CONTEXT})")

    Xtr = embed_context(ctx_bytes[tr_idx], lut)
    Xte = embed_context(ctx_bytes[te_idx], lut)
    ytr = y[tr_idx]; yte = y[te_idx]
    in_dim = Xtr.shape[1]
    print(f"Input dim: {in_dim}  (context * LUT_DIM = {CONTEXT}*16)")

    results = []
    for name, (fn, pre_dim) in VARIANTS.items():
        print(f"\n-- {name}  (pre_dim={pre_dim}) --")
        r = train_one(name, fn, pre_dim, Xtr, ytr, Xte, yte, in_dim, epochs=epochs)
        results.append(r)

    results.sort(key=lambda r: r["final_test_ce"])

    print("\n== Ranking by test CE (lower is better) ==")
    for r in results:
        print(f"  {r['name']:16s}  test_ce={r['final_test_ce']:.4f}  "
              f"test_acc={r['final_test_acc']*100:.2f}%  "
              f"params={r['param_count']:,}  "
              f"time={r['seconds']:.1f}s")

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps({
        "corpus": str(args.corpus),
        "context": CONTEXT, "hidden": H, "lr": LR, "momentum": MOMENTUM,
        "epochs": epochs, "batch": BATCH, "test_frac": TEST_FRAC,
        "corpus_bytes": len(raw), "train_windows": int(len(tr_idx)),
        "test_windows": int(len(te_idx)), "input_dim": in_dim,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
