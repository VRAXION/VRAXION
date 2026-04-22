"""One-shot probe: train a small model, dump W1/W2 distribution statistics."""
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
CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
EPOCHS = 2


def act_relu(z):
    a = np.maximum(z, 0.0)
    return a, lambda g: g * (z > 0).astype(np.float32)


def stats(name, W):
    flat = W.ravel()
    pos_frac = float((flat > 0).mean())
    p01, p10, p25, p50, p75, p90, p99 = np.percentile(flat, [1, 10, 25, 50, 75, 90, 99])
    pos_peak = float(flat.max())
    neg_peak = float(flat.min())
    print(f"\n{name}: shape={W.shape}  mean={flat.mean():+.5f}  median={p50:+.5f}  "
          f"std={flat.std():.5f}")
    print(f"  pos_fraction = {pos_frac*100:.2f}%     (symmetric baseline = 50%)")
    print(f"  abs  max  = {np.abs(flat).max():.5f}")
    print(f"  + peak    = {pos_peak:+.5f}   |  - peak    = {neg_peak:+.5f}")
    print(f"  p99       = {p99:+.5f}   |  p01       = {p01:+.5f}")
    print(f"  p90       = {p90:+.5f}   |  p10       = {p10:+.5f}")
    print(f"  p75       = {p75:+.5f}   |  p25       = {p25:+.5f}")
    pos_mass = float(np.abs(flat[flat > 0]).sum())
    neg_mass = float(np.abs(flat[flat < 0]).sum())
    print(f"  ∑|pos| = {pos_mass:.3f}   |  ∑|neg| = {neg_mass:.3f}   "
          f"ratio pos/neg = {pos_mass/neg_mass:.4f}")


def main():
    raw = CORPUS.read_bytes()
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

    print("=== At He-init (before any training) ===")
    stats("W1 init", W1)
    stats("W2 init", W2)

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), BATCH):
            idx = perm[i : i + BATCH]
            xb = X[idx]; yb = y[idx]; B = len(yb)
            z = xb @ W1 + b1
            a, grad_act = act_relu(z)
            logits = a @ W2 + b2
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m); p = e / e.sum(axis=1, keepdims=True)
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW2 = a.T @ dlogits; db2 = dlogits.sum(axis=0)
            da = dlogits @ W2.T
            dz = grad_act(da)
            dW1 = xb.T @ dz; db1 = dz.sum(axis=0)
            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - LR * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - LR * db2; b2 += vb2

    print(f"\n=== After {EPOCHS} epochs of relu training on alice ===")
    stats("W1 trained", W1)
    stats("W2 trained", W2)


if __name__ == "__main__":
    main()
