"""Block C byte round-trip reconstruction — find minimum H for 100% lossless.

Task mirrors Block B's A→B chain (chain_a_b.rs), but with a tunable hidden
width and pluggable activation. For each H in the sweep, trains a
tied-mirror autoencoder:

    x_bytes (K bytes)
        -> Block A LUT: (K, 16) float, flatten to (K*16,)
        -> z = x @ W + b1                  # W shape (K*16, H)
        -> a = activation(z)
        -> y = a @ W.T + b2                # (K*16,)
        -> reshape (K, 16)
        -> nearest-byte per 16-dim slice  => reconstructed K bytes

Training loss: L2 distance in LUT space (continuous, differentiable).
Evaluation: discrete byte round-trip on a held-out chunk set. A chunk is a
PASS only if ALL K bytes decode exactly.

Metric: chunk-level 100% round-trip rate.

Usage — plug the v3 winner in (default = beukers_gate):

    python3 tools/diag_block_c_reconstruction_h_sweep.py \
        --activation beukers_gate \
        --corpus output/data/fineweb_edu_100mb.txt \
        --max-bytes 2000000 \
        --k 8 \
        --h-grid 16,24,32,48,64,96,128,192,256,384,512 \
        --epochs 20 \
        --out output/block_c_reconstruction
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

BATCH = 256
LR = 0.05
MOMENTUM = 0.9
SEED = 1337
TEST_FRAC = 0.1
PATIENCE = 2

DEFAULT_CORPUS = REPO_ROOT / "output" / "data" / "fineweb_edu_100mb.txt"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_reconstruction"


# -------- activations (forward + grad wrt pre-activation) --------

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


# GLU-style activations halve the output dim, so their "pre_dim" is 2×H.
ACTIVATIONS = {
    "relu":           (act_relu,           1),
    "tanh":           (act_tanh,           1),
    "gelu":           (act_gelu,           1),
    "swish":          (act_swish,          1),
    "beukers_single": (act_beukers_single, 1),
    "beukers_gate":   (act_beukers_gate,   2),   # pre_dim = 2H, out = H
    "swiglu":         (act_swiglu,         2),
}


# -------- data --------

def make_chunks(raw: bytes, k: int) -> np.ndarray:
    """Slide a k-byte window across the corpus, return (N, k) uint8."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    N = len(arr) - k + 1
    out = np.empty((N, k), dtype=np.uint8)
    for i in range(k):
        out[:, i] = arr[i : i + N]
    return out


def embed_chunks(chunks: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """chunks: (N, K) uint8  ->  (N, K*16) float32 via Block A LUT."""
    return lut[chunks].reshape(chunks.shape[0], -1)


# -------- decode: nearest-LUT per 16-dim slice --------

def nearest_bytes(y_flat: np.ndarray, lut: np.ndarray, k: int) -> np.ndarray:
    """y_flat: (N, K*16)  ->  (N, K) uint8 reconstructed bytes."""
    y = y_flat.reshape(-1, k, 16)                    # (N, K, 16)
    # distance to each of 256 LUT vectors, per slot
    # lut: (256, 16)
    # we want squared dist -> pick argmin per (N, K)
    # expand: (N, K, 1, 16) - (256, 16) -> (N, K, 256)
    # using matmul trick:  d = ||y||^2 - 2 y@lut.T + ||lut||^2
    y_sq = np.sum(y * y, axis=-1, keepdims=True)     # (N, K, 1)
    lut_sq = np.sum(lut * lut, axis=-1)              # (256,)
    cross = y @ lut.T                                 # (N, K, 256)
    d = y_sq - 2.0 * cross + lut_sq                   # (N, K, 256)
    return np.argmin(d, axis=-1).astype(np.uint8)


# -------- training --------

def train_autoencoder(act_name, K, H, Xtr, Xte, targets_test, bytes_test,
                      lut, epochs):
    """Untied encoder/decoder. Tied-mirror breaks for GLU variants because
    the activation halves the width (pre_dim=2H -> H). We initialise W2
    from W1.T for non-GLU cases so behaviour is close to tied at t=0, and
    let the weights diverge during training."""
    act_fn, mult = ACTIVATIONS[act_name]
    pre_dim = mult * H
    out_dim_hidden = H  # activation output dim is always H
    in_dim = K * 16

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    # Mirror-init W2 from the first H columns of W1 (if GLU) or full W1.T (if not)
    W2 = W1[:, :out_dim_hidden].T.copy().astype(np.float32)
    b2 = np.zeros(in_dim, dtype=np.float32)

    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

    Ntr = Xtr.shape[0]
    best_loss = float("inf"); best_state = None; bad = 0
    curve = []

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xtr[idx]; tb = xb
            z = xb @ W1 + b1
            a, grad_act = act_fn(z)       # (B, H)
            y = a @ W2 + b2                # (B, in_dim)
            resid = y - tb
            loss = float(np.mean(resid * resid))
            losses.append(loss)
            dy = (2.0 / resid.size) * resid
            dW2 = a.T @ dy                 # (H, in_dim)
            db2 = dy.sum(axis=0)
            da = dy @ W2.T                 # (B, H)
            dz = grad_act(da)              # (B, pre_dim)
            dW1 = xb.T @ dz                # (in_dim, pre_dim)
            db1 = dz.sum(axis=0)
            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vW2 = MOMENTUM * vW2 - LR * dW2; W2 += vW2
            vb2 = MOMENTUM * vb2 - LR * db2; b2 += vb2

        z = Xte @ W1 + b1
        a, _ = act_fn(z)
        y = a @ W2 + b2
        resid = y - Xte
        test_loss = float(np.mean(resid * resid))

        bytes_hat = nearest_bytes(y, lut, K)
        match_all = np.all(bytes_hat == bytes_test, axis=1)
        roundtrip_pct = float(match_all.mean()) * 100.0
        train_loss = float(np.mean(losses))
        curve.append({"epoch": epoch, "train_loss": train_loss,
                      "test_loss": test_loss, "roundtrip_pct": roundtrip_pct})
        print(f"    ep {epoch:2d}  train={train_loss:.5f}  test={test_loss:.5f}  "
              f"roundtrip={roundtrip_pct:.2f}%", flush=True)
        if test_loss < best_loss - 1e-5:
            best_loss = test_loss
            best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break
        if roundtrip_pct >= 100.0 - 1e-9:
            break

    W1, b1, W2, b2 = best_state
    z = Xte @ W1 + b1
    a, _ = act_fn(z)
    y = a @ W2 + b2
    bytes_hat = nearest_bytes(y, lut, K)
    roundtrip_pct = float(np.all(bytes_hat == bytes_test, axis=1).mean()) * 100.0
    return {
        "activation": act_name, "K": K, "H": H, "pre_dim": pre_dim,
        "in_dim": in_dim, "epochs_run": len(curve), "curve": curve,
        "final_test_loss": float(best_loss),
        "final_roundtrip_pct": roundtrip_pct,
        "param_count": int(W1.size + b1.size + W2.size + b2.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="beukers_gate",
                    choices=list(ACTIVATIONS.keys()),
                    help="Winner activation from v3 sweep. Default: beukers_gate.")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--max-bytes", type=int, default=2_000_000,
                    help="Corpus read cap (bytes).")
    ap.add_argument("--k", type=int, default=8,
                    help="Chunk length in bytes (Block C operates on K-byte units).")
    ap.add_argument("--h-grid", type=str,
                    default="16,24,32,48,64,96,128,192,256,384,512",
                    help="Comma-separated hidden widths to sweep.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h_grid = [int(x) for x in args.h_grid.split(",")]

    print(f"== Block C reconstruction H sweep  (activation={args.activation}) ==")
    print(f"Corpus: {args.corpus}   K={args.k}   H grid={h_grid}")
    with args.corpus.open("rb") as f:
        raw = f.read(args.max_bytes) if args.max_bytes > 0 else f.read()
    print(f"Corpus bytes: {len(raw):,}")

    enc = ByteEncoder.load_default()
    lut = enc._lut_f32.astype(np.float32)

    chunks = make_chunks(raw, args.k)
    rng = np.random.default_rng(0)
    perm = rng.permutation(chunks.shape[0])
    n_test = int(chunks.shape[0] * TEST_FRAC)
    te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    Xtr = embed_chunks(chunks[tr_idx], lut)
    Xte = embed_chunks(chunks[te_idx], lut)
    bytes_test = chunks[te_idx]
    print(f"Chunks: train={len(tr_idx):,}  test={len(te_idx):,}")

    results = []
    t_start = time.time()
    first_100 = None
    for H in h_grid:
        print(f"\n-- H={H} --")
        t0 = time.time()
        r = train_autoencoder(args.activation, args.k, H,
                              Xtr, Xte, Xte, bytes_test, lut, args.epochs)
        r["seconds"] = time.time() - t0
        results.append(r)
        print(f"  H={H}: roundtrip={r['final_roundtrip_pct']:.2f}%  "
              f"param={r['param_count']:,}  t={r['seconds']:.1f}s")
        if first_100 is None and r["final_roundtrip_pct"] >= 100.0 - 1e-9:
            first_100 = H
            print(f"  ★ First H with 100% round-trip = {H}")
            break

    dt = time.time() - t_start
    print(f"\n== Summary ==  total {dt/60:.1f} min")
    for r in results:
        flag = "  ★" if r["H"] == first_100 else ""
        print(f"  H={r['H']:4d}  roundtrip={r['final_roundtrip_pct']:6.2f}%  "
              f"params={r['param_count']:,}  t={r['seconds']:.1f}s{flag}")

    out_path = args.out / f"hsweep_{args.activation}_k{args.k}.json"
    out_path.write_text(json.dumps({
        "activation": args.activation, "K": args.k,
        "h_grid": h_grid, "epochs_budget": args.epochs,
        "corpus": str(args.corpus), "corpus_bytes": len(raw),
        "train_chunks": int(len(tr_idx)), "test_chunks": int(len(te_idx)),
        "first_100_pct_H": first_100,
        "total_seconds": dt,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
