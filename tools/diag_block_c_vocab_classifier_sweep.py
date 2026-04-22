"""Block C vocab classifier sweep (pivot from reconstruction).

Reconstruction autoencoder stalled around 10% round-trip because:
  * padding bytes dominated the L2 loss
  * nothing explicitly punished identical embeddings

Pivot: train as a **classifier** over the LEARNED-token vocabulary.
Cross-entropy naturally pushes every token to its own class, so uniqueness is
guaranteed whenever classification accuracy is 100%.

Architecture:

    bytes (12, uint8) pad 0x00
       -> Block A LUT: (12, 16) float, flatten to (192,)
       -> encoder: W1 (192, pre_dim) -> activation -> (H,)
       -> latent head: W_lat (H, N) -> LATENT (N,)            ← Block C output
       -> classifier: W_cls (N, vocab_size) -> softmax logits
       -> loss: cross-entropy vs true token-id

Metrics per H (at float and after per-channel int4 PTQ):

    classification_acc  — fraction of tokens classified correctly
    unique_pct          — distinct latent rows (rounded to 6 decimals)
    min_pair_dist       — smallest pairwise L2 distance (subsample 4096)

Round-trip is not learned; it is a deterministic lookup table
(token_id -> bytes) the deploy pipeline already owns, so once
classification is 100% the whole chain is 100% lossless.

Run:
    python3 tools/diag_block_c_vocab_classifier_sweep.py \
        --activation beukers_gate --latent-dim 32 \
        --h-grid 32,48,64,96,128,192,256 --epochs 30
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
LR = 0.1        # SGD+momentum benefits from a higher LR here; with 32K-way
                # softmax and 32K examples the per-batch gradient on W_cls is
                # very sparse (only 256 active columns' bias row per step),
                # so a healthy LR is needed to actually move it.
MOMENTUM = 0.9
SEED = 1337
PATIENCE = 5
MAX_BYTES = 12

VOCAB_PATH = REPO_ROOT / "output" / "word_tokenizer_champion" / "champion_vocab.json"
DEFAULT_OUT = REPO_ROOT / "output" / "block_c_vocab_classifier"


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


# -------- data --------

def load_vocab_bytes(vocab_path: Path):
    data = json.loads(vocab_path.read_text())
    tokens = []
    for entry in data:
        if entry.get("kind") != "LEARNED":
            continue
        b = bytes.fromhex(entry["bytes_hex"])
        if entry.get("space_prefix"):
            b = b" " + b
        if len(b) > MAX_BYTES:
            b = b[:MAX_BYTES]
        tokens.append(b)
    N = len(tokens)
    arr = np.zeros((N, MAX_BYTES), dtype=np.uint8)
    lens = np.empty(N, dtype=np.int32)
    for i, b in enumerate(tokens):
        arr[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        lens[i] = len(b)
    return arr, lens


def embed_bytes(byte_arr, lut):
    return lut[byte_arr].reshape(byte_arr.shape[0], -1).astype(np.float32)


# -------- metrics --------

def latent_diagnostics(latents):
    rounded = np.round(latents, decimals=6)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    unique_pct = float(len(np.unique(inv))) / len(latents) * 100.0
    n = latents.shape[0]
    if n > 4096:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=4096, replace=False)
        L = latents[idx]
    else:
        L = latents
    sq = np.sum(L * L, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (L @ L.T)
    np.fill_diagonal(d2, np.inf)
    min_pair = float(np.sqrt(max(d2.min(), 0.0)))
    return unique_pct, min_pair


def ptq_weight_per_channel(W, bits=4):
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


# -------- training --------

def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m); p = e / e.sum(axis=1, keepdims=True)
    ll = -np.log(p[np.arange(len(y)), y] + 1e-12)
    return p, float(ll.mean())


def train_and_eval(act_name, H, latent_dim, X, y_cls, lut, vocab_size, epochs):
    act_fn, mult = ACTIVATIONS[act_name]
    pre_dim = mult * H
    in_dim = X.shape[1]

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, pre_dim)).astype(np.float32)
    b1 = np.zeros(pre_dim, dtype=np.float32)
    W_lat = rng.normal(0.0, np.sqrt(2.0 / H), size=(H, latent_dim)).astype(np.float32)
    b_lat = np.zeros(latent_dim, dtype=np.float32)
    W_cls = rng.normal(0.0, np.sqrt(2.0 / latent_dim), size=(latent_dim, vocab_size)).astype(np.float32)
    b_cls = np.zeros(vocab_size, dtype=np.float32)

    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vWl = np.zeros_like(W_lat); vbl = np.zeros_like(b_lat)
    vWc = np.zeros_like(W_cls); vbc = np.zeros_like(b_cls)

    Ntr = X.shape[0]
    best_ce = float("inf"); best_state = None; bad = 0
    curve = []

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(Ntr)
        losses = []
        for i in range(0, Ntr, BATCH):
            idx = perm[i : i + BATCH]
            xb = X[idx]; yb = y_cls[idx]; B = len(yb)
            z = xb @ W1 + b1
            a, grad_act = act_fn(z)
            lat = a @ W_lat + b_lat
            logits = lat @ W_cls + b_cls
            p, loss = softmax_ce(logits, yb)
            losses.append(loss)
            dlogits = p.copy(); dlogits[np.arange(B), yb] -= 1.0; dlogits /= B
            dW_cls = lat.T @ dlogits; db_cls = dlogits.sum(axis=0)
            dlat = dlogits @ W_cls.T
            dW_lat = a.T @ dlat; db_lat = dlat.sum(axis=0)
            da = dlat @ W_lat.T
            dz = grad_act(da)
            dW1 = xb.T @ dz; db1 = dz.sum(axis=0)

            vW1 = MOMENTUM * vW1 - LR * dW1; W1 += vW1
            vb1 = MOMENTUM * vb1 - LR * db1; b1 += vb1
            vWl = MOMENTUM * vWl - LR * dW_lat; W_lat += vWl
            vbl = MOMENTUM * vbl - LR * db_lat; b_lat += vbl
            vWc = MOMENTUM * vWc - LR * dW_cls; W_cls += vWc
            vbc = MOMENTUM * vbc - LR * db_cls; b_cls += vbc

        # full-set eval — chunked to avoid building a (N, vocab) logits matrix
        # (at N=vocab=32K that's 4 GB; we process 512 rows at a time instead).
        EVAL_CHUNK = 512
        preds = np.empty(Ntr, dtype=np.int64)
        ce_sum = 0.0
        lat_chunks = []
        for s in range(0, Ntr, EVAL_CHUNK):
            e = min(s + EVAL_CHUNK, Ntr)
            z = X[s:e] @ W1 + b1
            a, _ = act_fn(z)
            lat_chunk = a @ W_lat + b_lat
            lat_chunks.append(lat_chunk)
            logits = lat_chunk @ W_cls + b_cls
            mlog = logits.max(axis=1, keepdims=True)
            ex = np.exp(logits - mlog)
            pr = ex / ex.sum(axis=1, keepdims=True)
            yb_chunk = y_cls[s:e]
            ce_sum += float(-np.log(pr[np.arange(e - s), yb_chunk] + 1e-12).sum())
            preds[s:e] = logits.argmax(axis=1)
        test_ce = ce_sum / Ntr
        lat = np.concatenate(lat_chunks, axis=0)
        acc = float((preds == y_cls).mean()) * 100.0
        uniq, min_pair = latent_diagnostics(lat)
        train_ce = float(np.mean(losses))
        curve.append({"epoch": epoch, "train_ce": train_ce, "test_ce": test_ce,
                      "acc_pct": acc, "unique_pct": uniq, "min_pair": min_pair})
        print(f"    ep {epoch:2d}  train_ce={train_ce:.4f}  test_ce={test_ce:.4f}  "
              f"acc={acc:6.2f}%  uniq={uniq:6.2f}%  pair={min_pair:.4f}",
              flush=True)
        if test_ce < best_ce - 1e-4:
            best_ce = test_ce
            best_state = (W1.copy(), b1.copy(), W_lat.copy(), b_lat.copy(),
                          W_cls.copy(), b_cls.copy())
            bad = 0
        else:
            bad += 1
            if bad > PATIENCE:
                break
        if acc >= 100.0 - 1e-9:
            break

    W1, b1, W_lat, b_lat, W_cls, b_cls = best_state

    def _chunked_eval(W1_in, W_lat_in, W_cls_in):
        preds = np.empty(Ntr, dtype=np.int64)
        lat_chunks = []
        EVAL_CHUNK = 512
        for s in range(0, Ntr, EVAL_CHUNK):
            e = min(s + EVAL_CHUNK, Ntr)
            z = X[s:e] @ W1_in + b1
            a, _ = act_fn(z)
            lat_chunk = a @ W_lat_in + b_lat
            lat_chunks.append(lat_chunk)
            logits = lat_chunk @ W_cls_in + b_cls
            preds[s:e] = logits.argmax(axis=1)
        acc = float((preds == y_cls).mean()) * 100.0
        lat = np.concatenate(lat_chunks, axis=0)
        uniq, pair = latent_diagnostics(lat)
        return acc, uniq, pair

    acc_f, uniq_f, pair_f = _chunked_eval(W1, W_lat, W_cls)

    # Int4 metrics (per-channel PTQ on all three weight matrices)
    W1q = ptq_weight_per_channel(W1)
    W_lq = ptq_weight_per_channel(W_lat)
    W_cq = ptq_weight_per_channel(W_cls)
    acc_q, uniq_q, pair_q = _chunked_eval(W1q, W_lq, W_cq)

    return {
        "H": H, "latent_dim": latent_dim, "pre_dim": pre_dim,
        "param_count": int(W1.size + b1.size + W_lat.size + b_lat.size
                           + W_cls.size + b_cls.size),
        "epochs_run": len(curve), "curve": curve,
        "float": {"cls_acc": acc_f, "unique_pct": uniq_f, "min_pair": pair_f},
        "int4": {"cls_acc": acc_q, "unique_pct": uniq_q, "min_pair": pair_q},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="beukers_gate",
                    choices=list(ACTIVATIONS.keys()))
    ap.add_argument("--h-grid", type=str, default="32,48,64,96,128,192,256")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h_grid = [int(x) for x in args.h_grid.split(",")]

    print(f"== Block C vocab classifier sweep  (activation={args.activation}) ==")
    print(f"vocab: {args.vocab}  latent_dim={args.latent_dim}  epochs={args.epochs}")

    byte_arr, lens = load_vocab_bytes(args.vocab)
    N = byte_arr.shape[0]
    y_cls = np.arange(N, dtype=np.int64)          # token-id is the class label
    print(f"LEARNED tokens loaded: {N:,}  "
          f"mean_len={lens.mean():.2f}  max_len={lens.max()}")

    enc = ByteEncoder.load_default()
    lut = enc._lut_f32.astype(np.float32)
    X = embed_bytes(byte_arr, lut)
    print(f"Input dim: {X.shape[1]}  vocab_size={N}")

    results = []
    t_start = time.time()
    first_100 = None
    for H in h_grid:
        pre_dim = ACTIVATIONS[args.activation][1] * H
        print(f"\n-- H={H}  pre_dim={pre_dim}  (act={args.activation}) --")
        t0 = time.time()
        r = train_and_eval(args.activation, H, args.latent_dim, X, y_cls, lut,
                           vocab_size=N, epochs=args.epochs)
        r["seconds"] = time.time() - t0
        results.append(r)
        f = r["float"]; q = r["int4"]
        print(f"  H={H}  float: acc={f['cls_acc']:6.2f}%  uniq={f['unique_pct']:6.2f}%  "
              f"pair={f['min_pair']:.4f}")
        print(f"         int4 : acc={q['cls_acc']:6.2f}%  uniq={q['unique_pct']:6.2f}%  "
              f"pair={q['min_pair']:.4f}  "
              f"(params={r['param_count']:,}, t={r['seconds']:.1f}s)")
        if first_100 is None and f["cls_acc"] >= 100.0 - 1e-9:
            first_100 = H
            print(f"  ★ First H with 100% float classification accuracy = {H}")

    dt = time.time() - t_start
    print(f"\n== Summary ==  total {dt/60:.1f} min")
    print(f"  {'H':>5}  {'acc_f':>6}  {'uniq_f':>7}  {'pair_f':>8}  "
          f"{'acc_q':>6}  {'uniq_q':>7}  {'pair_q':>8}  params")
    for r in results:
        flag = "  ★" if r["H"] == first_100 else ""
        f = r["float"]; q = r["int4"]
        print(f"  {r['H']:5d}  {f['cls_acc']:6.2f}  {f['unique_pct']:7.2f}  "
              f"{f['min_pair']:8.4f}  "
              f"{q['cls_acc']:6.2f}  {q['unique_pct']:7.2f}  "
              f"{q['min_pair']:8.4f}  {r['param_count']:,}{flag}")

    out_path = args.out / f"sweep_{args.activation}_L{args.latent_dim}.json"
    out_path.write_text(json.dumps({
        "activation": args.activation, "latent_dim": args.latent_dim,
        "h_grid": h_grid, "epochs_budget": args.epochs,
        "vocab_path": str(args.vocab),
        "n_learned_tokens": int(N),
        "max_bytes": MAX_BYTES, "first_100_pct_H": first_100,
        "total_seconds": dt, "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
