"""Byte mirror autoencoder with different activations in the encoder.

Test: does adding a nonlinearity to the encoder (while keeping tied-weight
linear decoder) preserve 100% lossless at dim=16?

Activations tested:
  - linear (baseline, no activation)
  - relu
  - tanh
  - c19 (rho=8, c=1.0 — the validated VRAXION winner)
  - leaky_relu (alpha=0.01)

Architecture:
  encode: latent = activation(input @ Wq + bias)
  decode: logits = latent @ Wq.t() + out_bias     (tied W, linear decode)

All with int8 QAT STE, dual loss (recon + context), dim=16.
"""

from __future__ import annotations
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
LATENT_DIM = 16
SEED = 42
EPOCHS = 30
BATCH = 8192
N_SAMPLES = 200_000
LR = 0.01
VOCAB = 256


class Int8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        return torch.clamp(torch.round(w / scale), -127, 127) * scale

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantize_int8(W):
    scale = W.abs().max().detach().clamp(min=1e-8) / 127.0
    return Int8STE.apply(W, scale)


def byte_to_bits(b):
    bits = torch.zeros(b.shape[0], N_BITS, device=b.device)
    for i in range(N_BITS):
        bits[:, i] = (b >> i) & 1
    return bits


# ── C19 activation (vectorized) ───────────────────────────

def c19_activation(x, c=1.0, rho=8.0):
    """C19 oscillating activation, vectorized PyTorch."""
    c = max(c, 0.1)
    L = 6.0 * c
    # Clamp to linear outside [-L, L]
    out = torch.where(x >= L, x - L, torch.where(x <= -L, x + L, torch.zeros_like(x)))
    # Inside [-L, L]: periodic bumps
    mask = (x > -L) & (x < L)
    if mask.any():
        scaled = x[mask] / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
        out[mask] = c * (sgn * h + rho * h * h)
    return out


# ── Activations registry ──────────────────────────────────

ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: F.relu(x),
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "tanh": lambda x: torch.tanh(x),
    "c19_rho8": lambda x: c19_activation(x, c=1.0, rho=8.0),
}


def load_bigrams(path, n, seed=SEED):
    raw = Path(path).read_bytes()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(raw) - 2, (n,), generator=gen)
    return arr[offs].long(), arr[offs + 1].long()


def train_and_test(act_name, act_fn, cur, nxt, corpus_path):
    torch.manual_seed(SEED)
    W = (torch.randn(N_BITS, LATENT_DIM, device=DEVICE) * 0.3).detach().requires_grad_(True)
    enc_bias = torch.zeros(LATENT_DIM, device=DEVICE).requires_grad_(True)
    dec_bias = torch.zeros(N_BITS, device=DEVICE).requires_grad_(True)
    V = (torch.randn(LATENT_DIM, VOCAB, device=DEVICE) * 0.1).detach().requires_grad_(True)
    opt = torch.optim.Adam([W, enc_bias, dec_bias, V], lr=LR)

    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            Wq = quantize_int8(W)
            raw = inp @ Wq + enc_bias
            latent = act_fn(raw)
            logits_rec = latent @ Wq.t() + dec_bias
            loss_rec = F.binary_cross_entropy_with_logits(logits_rec, bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0

    # Eval: lossless
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        Wq = quantize_int8(W)
        latent = act_fn(inp @ Wq + enc_bias)
        logits = latent @ Wq.t() + dec_bias
        pred = (logits > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())

        # Min pairwise dist (collision)
        dists = torch.cdist(latent.unsqueeze(0), latent.unsqueeze(0)).squeeze(0)
        dists.fill_diagonal_(float('inf'))
        min_dist = dists.min().item()

    # Downstream char-LM
    CTX = 8; MASK_POS = 4
    raw_bytes = Path(corpus_path).read_bytes()
    arr = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)

    def sample(n, seed):
        gen = torch.Generator().manual_seed(seed)
        offs = torch.randint(0, len(arr) - CTX - 1, (n,), generator=gen)
        idx_mat = offs.unsqueeze(1) + torch.arange(CTX).unsqueeze(0)
        chunks = arr[idx_mat].long()
        targets = chunks[:, MASK_POS].clone()
        chunks[:, MASK_POS] = 32
        return chunks.to(DEVICE), targets.to(DEVICE)

    eval_x, eval_y = sample(5000, 99)
    train_x, train_y = sample(20000, 42)

    with torch.no_grad():
        Wq = quantize_int8(W)
        def embed(chunks):
            flat = chunks.flatten()
            b = byte_to_bits(flat).float() * 2.0 - 1.0
            lat = act_fn(b @ Wq + enc_bias)
            return lat.view(chunks.shape[0], CTX, -1).reshape(chunks.shape[0], -1)
        train_feat = embed(train_x)
        eval_feat = embed(eval_x)

    D = train_feat.shape[1]
    torch.manual_seed(SEED)
    P = (torch.randn(D, VOCAB, device=DEVICE) * 0.01).detach().requires_grad_(True)
    pb = torch.zeros(VOCAB, device=DEVICE).requires_grad_(True)
    opt2 = torch.optim.Adam([P, pb], lr=0.005)
    for _ in range(100):
        loss = F.cross_entropy(train_feat @ P + pb, train_y)
        opt2.zero_grad(); loss.backward(); opt2.step()
    with torch.no_grad():
        eval_acc = (eval_feat @ P + pb).argmax(1).eq(eval_y).float().mean().item() * 100

    return {
        "act": act_name, "byte_acc": byte_acc, "missed": missed,
        "min_dist": min_dist, "eval_acc": eval_acc, "time": elapsed,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}, dim={LATENT_DIM}")
    print(f"Byte mirror: activation sweep (encoder-side only, tied linear decoder)")
    print(f"Dual loss (recon + context), int8 QAT STE\n")

    cur, nxt = load_bigrams(corpus, N_SAMPLES)

    results = []
    for name in ["linear", "relu", "leaky_relu", "tanh", "c19_rho8"]:
        print(f">>> {name}...")
        r = train_and_test(name, ACTIVATIONS[name], cur, nxt, corpus)
        print(f"    lossless={r['byte_acc']:.2f}%  missed={r['missed']}/256  "
              f"min_dist={r['min_dist']:.4f}  downstream={r['eval_acc']:.2f}%  [{r['time']:.1f}s]")
        results.append(r)

    print(f"\n{'='*72}")
    print(f"{'activation':<14} {'lossless':>10} {'missed':>8} {'min_dist':>10} {'downstream':>12} {'time':>8}")
    print(f"{'='*72}")
    for r in results:
        flag = " <<< PASS" if r["byte_acc"] == 100.0 else ""
        print(f"{r['act']:<14} {r['byte_acc']:>9.2f}% {r['missed']:>8d} {r['min_dist']:>10.4f} "
              f"{r['eval_acc']:>11.2f}% {r['time']:>7.1f}s{flag}")


if __name__ == "__main__":
    main()
