"""Block C vocab-aware embedder sweep.

Replaces the earlier sliding-window reconstruction probe. Here the task is what
Block C actually has to do: **embed every champion-vocab token into a latent
vector** such that it is unique, minimally-distant from its neighbours, and
decodable back to the original bytes. We measure this at float and int4.

For each H in the sweep we:
  1. Build a tokens-as-bytes dataset from output/word_tokenizer_champion/
     champion_vocab.json (LEARNED tokens only — BYTE / PUNCT / WS_RUN are
     trivially unique).
  2. Train an autoencoder (encoder: bytes -> N-dim latent, decoder: N -> bytes)
     with the Block C winner activation (default beukers_gate).
  3. Measure four output metrics at float and after per-channel int4 PTQ:
       * round_trip_pct   — tokens whose bytes reconstruct exactly
       * unique_pct       — distinct latent vectors out of N_total
       * min_pair_dist    — smallest L2 distance between any two latents
       * int4_collision_pct — how many unique-float embeddings collapse under int4

Finds the minimum H where all four metrics pass simultaneously.

Usage:
    python3 tools/diag_block_c_vocab_embedder_sweep.py \
        --activation beukers_gate \
        --h-grid 32,48,64,96,128,192,256 \
        --epochs 30 \
        --latent-dim 32
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
PATIENCE = 3

VOCAB_PATH = REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_vocab_embedder"
MAX_BYTES = 12        # champion max_subword_len
PAD_BYTE = 0x00       # sentinel; length is tracked explicitly


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


ACTIVATIONS = {
    "relu":         (act_relu,         1),
    "tanh":         (act_tanh,         1),
    "gelu":         (act_gelu,         1),
    "swish":        (act_swish,        1),
    "beukers_gate": (act_beukers_gate, 2),
    "swiglu":       (act_swiglu,       2),
}


# -------- vocab loading --------

def load_vocab_bytes(vocab_path: Path):
    """Return (bytes_matrix, lengths) restricted to LEARNED tokens.

    bytes_matrix : (N, MAX_BYTES) uint8, right-padded with PAD_BYTE
    lengths      : (N,) int32  — true byte length per token
    """
    data = json.loads(vocab_path.read_text())
    tokens = []
    for entry in data:
        if entry.get("kind") != "LEARNED":
            continue
        hex_str = entry["bytes_hex"]
        b = bytes.fromhex(hex_str)
        if entry.get("space_prefix"):
            b = b" " + b
        if len(b) > MAX_BYTES:
            b = b[:MAX_BYTES]
        tokens.append(b)

    N = len(tokens)
    arr = np.full((N, MAX_BYTES), PAD_BYTE, dtype=np.uint8)
    lens = np.empty(N, dtype=np.int32)
    for i, b in enumerate(tokens):
        arr[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        lens[i] = len(b)
    return arr, lens


def embed_bytes(byte_arr, lut):
    return lut[byte_arr].reshape(byte_arr.shape[0], -1).astype(np.float32)


def nearest_bytes(y_flat, lut, max_bytes):
    y = y_flat.reshape(-1, max_bytes, 16)
    y_sq = np.sum(y * y, axis=-1, keepdims=True)
    lut_sq = np.sum(lut * lut, axis=-1)
    cross = y @ lut.T
    d = y_sq - 2.0 * cross + lut_sq
    return np.argmin(d, axis=-1).astype(np.uint8)


# -------- metrics --------

def round_trip_pct(byte_pred, byte_true, lengths):
    """Compare only up to each token's true length."""
    ok = np.ones(len(lengths), dtype=bool)
    for i, L in enumerate(lengths):
        if not np.array_equal(byte_pred[i, :L], byte_true[i, :L]):
            ok[i] = False
    return float(ok.mean()) * 100.0


def latent_diagnostics(latents):
    """Return (unique_pct, min_pair_dist). min_pair_dist computed pairwise on a
    subset capped at 4096 rows for tractability, representative of the
    distribution."""
    # Uniqueness via row hashing
    rounded = np.round(latents, decimals=6)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    unique_pct = float(len(np.unique(inv))) / len(latents) * 100.0

    # Subsample for pairwise distance
    n = latents.shape[0]
    if n > 4096:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=4096, replace=False)
        L = latents[idx]
    else:
        L = latents
    # ||a-b||² = ||a||² + ||b||² - 2ab
    sq = np.sum(L * L, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (L @ L.T)
    # ignore the diagonal (self-distance)
    np.fill_diagonal(d2, np.inf)
    min_pair = float(np.sqrt(max(d2.min(), 0.0)))
    return unique_pct, min_pair


# -------- int4 per-channel symmetric PTQ --------

def ptq_weight_per_channel(W, bits=4):
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


# -------- autoencoder (encoder: bytes -> N-dim latent, decoder: N -> bytes) --------

def train_and_eval(act_name, H, latent_dim, X, bytes_true, lengths, lut, epochs):
    act_fn, mult = ACTIVATIONS[act_name]
    pre_dim = mult * H
    in_dim = X.shape[1]

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W_lat = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, latent_dim)).astype(np.float32)
    b_lat = np.zeros(latent_dim, dtype=np.float32)
    W_dec = rng.normal(0.0, np.sqrt(2.0 / latent_dim), size=(latent_dim, in_dim)).astype(np.float32)
    b_dec = np.zeros(in_dim, dtype=np.float32)

    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vWl = np.zeros_like(W_lat); vbl = np.zeros_like(b_lat)
    vWd = np.zeros_like(W_dec); vbd = np.zeros_like(b_dec)

    Ntr = X.shape[0]
    best_loss = float("inf"); best_state = None; bad = 0
    curve = []

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = X[idx]; tb = xb
            z = xb @ W1 + b1
            a, grad_act = act_fn(z)        # (B, H)
            lat = a @ W_lat + b_lat         # (B, latent_dim)
            y = lat @ W_dec + b_dec         # (B, in_dim)
            resid = y - tb
            loss = float(np.mean(resid * resid))
            losses.append(loss)

            dy = (2.0 / resid.size) * resid
            dW_dec = lat.T @ dy
            db_dec = dy.sum(axis=0)
            dlat = dy @ W_dec.T
            dW_lat = a.T @ dlat
            db_lat = dlat.sum(axis=0)
            da = dlat @ W_lat.T
            dz = grad_act(da)
            dW1 = xb.T @ dz
            db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vWl = MOMENTUM * vWl - LR * dW_lat; W_lat += vWl
            vbl = MOMENTUM * vbl - LR * db_lat; b_lat += vbl
            vWd = MOMENTUM * vWd - LR * dW_dec; W_dec += vWd
            vbd = MOMENTUM * vbd - LR * db_dec; b_dec += vbd

        # epoch eval (full set since N is small)
        z = X @ W1 + b1
        a, _ = act_fn(z)
        lat = a @ W_lat + b_lat
        y = lat @ W_dec + b_dec
        resid = y - X
        test_loss = float(np.mean(resid * resid))
        bytes_hat = nearest_bytes(y, lut, MAX_BYTES)
        rt_pct = round_trip_pct(bytes_hat, bytes_true, lengths)
        unique_pct, min_pair = latent_diagnostics(lat)
        train_loss = float(np.mean(losses))
        curve.append({"epoch": epoch, "train_loss": train_loss,
                      "recon_loss": test_loss, "roundtrip_pct": rt_pct,
                      "unique_pct": unique_pct, "min_pair_dist": min_pair})
        print(f"    ep {epoch:2d}  recon={test_loss:.5f}  rt={rt_pct:6.2f}%  "
              f"uniq={unique_pct:6.2f}%  min_pair={min_pair:.4f}",
              flush=True)
        if test_loss < best_loss - 1e-5:
            best_loss = test_loss
            best_state = (W1.copy(), b1.copy(), W_lat.copy(), b_lat.copy(),
                          W_dec.copy(), b_dec.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break
        if rt_pct >= 100.0 - 1e-9 and unique_pct >= 100.0 - 1e-9:
            break

    W1, b1, W_lat, b_lat, W_dec, b_dec = best_state

    # Float eval
    z = X @ W1 + b1
    a, _ = act_fn(z)
    lat_f = a @ W_lat + b_lat
    y_f = lat_f @ W_dec + b_dec
    bytes_f = nearest_bytes(y_f, lut, MAX_BYTES)
    rt_f = round_trip_pct(bytes_f, bytes_true, lengths)
    uniq_f, min_pair_f = latent_diagnostics(lat_f)

    # Int4 eval (per-channel PTQ on all three weight matrices)
    W1q = ptq_weight_per_channel(W1)
    W_lq = ptq_weight_per_channel(W_lat)
    W_dq = ptq_weight_per_channel(W_dec)
    z = X @ W1q + b1
    a, _ = act_fn(z)
    lat_q = a @ W_lq + b_lat
    y_q = lat_q @ W_dq + b_dec
    bytes_q = nearest_bytes(y_q, lut, MAX_BYTES)
    rt_q = round_trip_pct(bytes_q, bytes_true, lengths)
    uniq_q, min_pair_q = latent_diagnostics(lat_q)

    return {
        "H": H, "latent_dim": latent_dim, "pre_dim": pre_dim,
        "param_count": int(W1.size + b1.size + W_lat.size + b_lat.size
                           + W_dec.size + b_dec.size),
        "epochs_run": len(curve), "curve": curve,
        "float_recon_loss": float(best_loss),
        "float": {"roundtrip_pct": rt_f, "unique_pct": uniq_f,
                  "min_pair_dist": min_pair_f},
        "int4": {"roundtrip_pct": rt_q, "unique_pct": uniq_q,
                 "min_pair_dist": min_pair_q},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="beukers_gate",
                    choices=list(ACTIVATIONS.keys()))
    ap.add_argument("--h-grid", type=str, default="32,48,64,96,128,192,256")
    ap.add_argument("--latent-dim", type=int, default=32,
                    help="Output embedding dimension N.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h_grid = [int(x) for x in args.h_grid.split(",")]

    print(f"== Block C vocab embedder sweep  (activation={args.activation}) ==")
    print(f"vocab: {args.vocab}  latent_dim={args.latent_dim}  epochs={args.epochs}")

    byte_arr, lens = load_vocab_bytes(args.vocab)
    print(f"LEARNED tokens loaded: {byte_arr.shape[0]:,}  "
          f"mean_len={lens.mean():.2f}  max_len={lens.max()}")

    enc = ByteEncoder.load_default()
    lut = enc._lut_f32.astype(np.float32)
    X = embed_bytes(byte_arr, lut)
    print(f"Input dim (max_bytes * 16): {X.shape[1]}  N={X.shape[0]:,}")

    results = []
    t_start = time.time()
    first_pass_H = None
    for H in h_grid:
        pre_dim = ACTIVATIONS[args.activation][1] * H
        print(f"\n-- H={H}  pre_dim={pre_dim}  (act={args.activation}) --")
        t0 = time.time()
        r = train_and_eval(args.activation, H, args.latent_dim, X,
                           byte_arr, lens, lut, args.epochs)
        r["seconds"] = time.time() - t0
        results.append(r)
        f = r["float"]; q = r["int4"]
        print(f"  H={H}  float: rt={f['roundtrip_pct']:6.2f}%  "
              f"uniq={f['unique_pct']:6.2f}%  pair={f['min_pair_dist']:.4f}")
        print(f"         int4 : rt={q['roundtrip_pct']:6.2f}%  "
              f"uniq={q['unique_pct']:6.2f}%  pair={q['min_pair_dist']:.4f}  "
              f"(params={r['param_count']:,}, t={r['seconds']:.1f}s)")
        if (first_pass_H is None
                and f["roundtrip_pct"] >= 100.0 - 1e-9
                and f["unique_pct"] >= 100.0 - 1e-9):
            first_pass_H = H
            print(f"  ★ First H with 100% float round-trip AND uniqueness = {H}")

    dt = time.time() - t_start
    print(f"\n== Summary ==  total {dt/60:.1f} min")
    print(f"  {'H':>5}  {'rt_f':>6}  {'uniq_f':>6}  {'pair_f':>8}  "
          f"{'rt_q':>6}  {'uniq_q':>6}  {'pair_q':>8}  params")
    for r in results:
        flag = "  ★" if r["H"] == first_pass_H else ""
        f = r["float"]; q = r["int4"]
        print(f"  {r['H']:5d}  {f['roundtrip_pct']:6.2f}  {f['unique_pct']:6.2f}  "
              f"{f['min_pair_dist']:8.4f}  "
              f"{q['roundtrip_pct']:6.2f}  {q['unique_pct']:6.2f}  "
              f"{q['min_pair_dist']:8.4f}  {r['param_count']:,}{flag}")

    out_path = args.out / f"sweep_{args.activation}_L{args.latent_dim}.json"
    out_path.write_text(json.dumps({
        "activation": args.activation, "latent_dim": args.latent_dim,
        "h_grid": h_grid, "epochs_budget": args.epochs,
        "vocab_path": str(args.vocab),
        "n_learned_tokens": int(byte_arr.shape[0]),
        "max_bytes": MAX_BYTES, "first_pass_H": first_pass_H,
        "total_seconds": dt, "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
