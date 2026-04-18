"""Byte tokenizer unit architecture sweep.

Goal: find the most compact unit with best downstream signal.
All with int8 QAT, dual loss (recon + context).

Architectures tested:
  A: 8->16 linear tied          (baseline, 128 params)
  B: 8->16 ReLU tied            (+ activation, 144 params)
  C: 8->32->16 ReLU+linear tied (2-layer, 656 params)
  D: 8->16->16 ReLU+ReLU tied   (2-layer both active, 400 params)
  E: 8->16 ReLU untied          (no mirror, 256+16 params)
  F: 8->24->16 ReLU+linear tied (2-layer mid-width, 464 params)

Measured: downstream char-LM eval%, lossless%, param count, inference cost.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
OUT_DIM = 16
SEED = 42
EPOCHS = 40
BATCH = 8192
N_BIGRAMS = 200_000
LR = 0.01
VOCAB = 256
CTX = 8
MASK_POS = 4


class Int8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        return torch.clamp(torch.round(w / scale), -127, 127) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None

def q8(W):
    scale = W.abs().max().detach().clamp(min=1e-8) / 127.0
    return Int8STE.apply(W, scale)

def byte_to_bits(b):
    bits = torch.zeros(b.shape[0], N_BITS, device=b.device)
    for i in range(N_BITS):
        bits[:, i] = (b >> i) & 1
    return bits

def load_bigrams(path, n):
    raw = Path(path).read_bytes()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(raw) - 2, (n,), generator=gen)
    return arr[offs].long(), arr[offs + 1].long()


# ── Architecture definitions ──────────────────────────────

class UnitA(torch.nn.Module):
    """8->16 linear tied."""
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(N_BITS, OUT_DIM) * 0.3)
    def encode(self, x):
        return x @ q8(self.W)
    def decode(self, z):
        return z @ q8(self.W).t()
    def param_count(self):
        return N_BITS * OUT_DIM

class UnitB(torch.nn.Module):
    """8->16 ReLU tied."""
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(N_BITS, OUT_DIM) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(OUT_DIM))
    def encode(self, x):
        return F.relu(x @ q8(self.W) + self.b)
    def decode(self, z):
        return z @ q8(self.W).t()
    def param_count(self):
        return N_BITS * OUT_DIM + OUT_DIM

class UnitC(torch.nn.Module):
    """8->32 ReLU -> 32->16 linear, tied."""
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, 32) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(32))
        self.W2 = torch.nn.Parameter(torch.randn(32, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
    def encode(self, x):
        h = F.relu(x @ q8(self.W1) + self.b1)
        return h @ q8(self.W2) + self.b2
    def decode(self, z):
        h = z @ q8(self.W2).t()
        return h @ q8(self.W1).t()
    def param_count(self):
        return N_BITS * 32 + 32 + 32 * OUT_DIM + OUT_DIM

class UnitD(torch.nn.Module):
    """8->16 ReLU -> 16->16 ReLU, tied."""
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, OUT_DIM) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.W2 = torch.nn.Parameter(torch.randn(OUT_DIM, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
    def encode(self, x):
        h = F.relu(x @ q8(self.W1) + self.b1)
        return F.relu(h @ q8(self.W2) + self.b2)
    def decode(self, z):
        h = z @ q8(self.W2).t()
        return h @ q8(self.W1).t()
    def param_count(self):
        return N_BITS * OUT_DIM + OUT_DIM + OUT_DIM * OUT_DIM + OUT_DIM

class UnitE(torch.nn.Module):
    """8->16 ReLU, UNTIED (separate encoder/decoder W)."""
    def __init__(self):
        super().__init__()
        self.We = torch.nn.Parameter(torch.randn(N_BITS, OUT_DIM) * 0.3)
        self.be = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.Wd = torch.nn.Parameter(torch.randn(OUT_DIM, N_BITS) * 0.3)
    def encode(self, x):
        return F.relu(x @ q8(self.We) + self.be)
    def decode(self, z):
        return z @ q8(self.Wd)
    def param_count(self):
        return N_BITS * OUT_DIM + OUT_DIM + OUT_DIM * N_BITS

class UnitF(torch.nn.Module):
    """8->24 ReLU -> 24->16 linear, tied."""
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, 24) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(24))
        self.W2 = torch.nn.Parameter(torch.randn(24, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
    def encode(self, x):
        h = F.relu(x @ q8(self.W1) + self.b1)
        return h @ q8(self.W2) + self.b2
    def decode(self, z):
        h = z @ q8(self.W2).t()
        return h @ q8(self.W1).t()
    def param_count(self):
        return N_BITS * 24 + 24 + 24 * OUT_DIM + OUT_DIM


UNITS = {
    "A linear-tied": UnitA,
    "B relu-tied": UnitB,
    "C 8-32-16 tied": UnitC,
    "D relu-relu tied": UnitD,
    "E relu-untied": UnitE,
    "F 8-24-16 tied": UnitF,
}


def train_and_eval(name, unit_cls, cur, nxt, corpus_path):
    torch.manual_seed(SEED)
    unit = unit_cls().to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    all_params = list(unit.parameters()) + [V]
    opt = torch.optim.Adam(all_params, lr=LR)

    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            latent = unit.encode(inp)
            logits_rec = unit.decode(latent)
            loss_rec = F.binary_cross_entropy_with_logits(logits_rec, bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    # Lossless test
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        latent = unit.encode(inp)
        logits = unit.decode(latent)
        pred = (logits > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())

    # Downstream char-LM
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

    train_x, train_y = sample(20000, 42)
    eval_x, eval_y = sample(5000, 99)

    with torch.no_grad():
        def embed(chunks):
            flat = chunks.flatten()
            b = byte_to_bits(flat).float() * 2.0 - 1.0
            lat = unit.encode(b)
            return lat.view(chunks.shape[0], CTX, -1).reshape(chunks.shape[0], -1)
        train_feat = embed(train_x)
        eval_feat = embed(eval_x)

    D = train_feat.shape[1]
    torch.manual_seed(SEED)
    P = torch.nn.Parameter(torch.randn(D, VOCAB, device=DEVICE) * 0.01)
    pb = torch.nn.Parameter(torch.zeros(VOCAB, device=DEVICE))
    opt2 = torch.optim.Adam([P, pb], lr=0.005)
    for _ in range(100):
        loss = F.cross_entropy(train_feat @ P + pb, train_y)
        opt2.zero_grad(); loss.backward(); opt2.step()
    with torch.no_grad():
        eval_acc = (eval_feat @ P + pb).argmax(1).eq(eval_y).float().mean().item() * 100

    return {
        "name": name,
        "params": unit.param_count(),
        "byte_acc": byte_acc,
        "missed": missed,
        "eval_acc": eval_acc,
        "time": train_time,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}, output_dim={OUT_DIM}")
    print(f"Byte unit architecture sweep — compact + best downstream\n")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    results = []
    for name, cls in UNITS.items():
        print(f">>> {name}...")
        r = train_and_eval(name, cls, cur, nxt, corpus)
        print(f"    params={r['params']:>5d}  lossless={r['byte_acc']:>6.2f}%  "
              f"downstream={r['eval_acc']:>5.2f}%  [{r['time']:.1f}s]")
        results.append(r)

    # Sort by downstream
    results.sort(key=lambda r: -r["eval_acc"])

    print(f"\n{'='*78}")
    print(f"{'UNIT':<20} {'params':>7} {'int8 bytes':>10} {'lossless':>10} {'downstream':>12} {'rank':>6}")
    print(f"{'='*78}")
    for i, r in enumerate(results):
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        print(f"{r['name']:<20} {r['params']:>7d} {r['params']:>9d}B {ll:>10} {r['eval_acc']:>11.2f}% {i+1:>6}")

    winner = results[0]
    print(f"\n>>> WINNER: {winner['name']}")
    print(f"    {winner['params']} params = {winner['params']} bytes int8")
    print(f"    downstream: {winner['eval_acc']:.2f}%")
    print(f"    lossless: {winner['byte_acc']:.2f}%")


if __name__ == "__main__":
    main()
