"""Fix the lossless gap: get 100% reconstruction WITH nonlinear activation.

Two strategies tested on 1H SiLU 8->32->16 (current: 99.6% lossless):
  A: Heavy recon weight (5.0 * recon + 0.1 * ctx) — push harder
  B: 2-phase training:
       Phase 1: recon-only until 100% lossless
       Phase 2: freeze W, train only V (context head) on frozen latents
  C: Even heavier (10.0 * recon + 0.1 * ctx) + more epochs
  D: 2-phase with GELU (best downstream activation)

Also test: does the frozen embedding still give good downstream signal?
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
    s = W.abs().max().detach().clamp(min=1e-8) / 127.0
    return Int8STE.apply(W, s)

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


class Unit1H(torch.nn.Module):
    def __init__(self, H=32, act="silu"):
        super().__init__()
        self.act_name = act
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.H = H

    def act(self, x):
        if self.act_name == "silu": return F.silu(x)
        if self.act_name == "gelu": return F.gelu(x)
        return F.relu(x)

    def encode(self, x):
        return self.act(x @ q8(self.W1) + self.b1) @ q8(self.W2) + self.b2

    def decode(self, z):
        return z @ q8(self.W2).t() @ q8(self.W1).t()

    def param_count(self):
        return N_BITS * self.H + self.H + self.H * OUT_DIM + OUT_DIM


def eval_lossless(unit):
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        pred = (unit.decode(unit.encode(inp)) > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())
    return byte_acc, missed


def eval_downstream(unit, corpus_path):
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
    opt = torch.optim.Adam([P, pb], lr=0.005)
    for _ in range(100):
        loss = F.cross_entropy(train_feat @ P + pb, train_y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (eval_feat @ P + pb).argmax(1).eq(eval_y).float().mean().item() * 100


def strategy_A(act, recon_w, epochs, cur, nxt, corpus):
    """Single-phase dual loss with heavy recon weight."""
    torch.manual_seed(SEED)
    unit = Unit1H(32, act).to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    opt = torch.optim.Adam(list(unit.parameters()) + [V], lr=LR)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            latent = unit.encode(inp)
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent), bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = recon_w * loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()
    elapsed = time.time() - t0

    byte_acc, missed = eval_lossless(unit)
    ds = eval_downstream(unit, corpus)
    return byte_acc, missed, ds, elapsed


def strategy_B(act, epochs_p1, epochs_p2, cur, nxt, corpus):
    """2-phase: recon-only → freeze W, train V only."""
    torch.manual_seed(SEED)
    unit = Unit1H(32, act).to(DEVICE)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)
    N = cur_d.shape[0]

    # Phase 1: recon only
    opt1 = torch.optim.Adam(unit.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(epochs_p1):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            latent = unit.encode(inp)
            loss = F.binary_cross_entropy_with_logits(unit.decode(latent), bits)
            opt1.zero_grad(); loss.backward(); opt1.step()

        ba, mi = eval_lossless(unit)
        if ba == 100.0:
            print(f"    Phase 1: 100% lossless reached at ep={ep}")
            break

    ba_p1, mi_p1 = eval_lossless(unit)
    print(f"    Phase 1 done: lossless={ba_p1:.2f}%  missed={mi_p1}")

    # Freeze encoder weights
    for p in unit.parameters():
        p.requires_grad_(False)

    # Phase 2: context head only on frozen latents
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    opt2 = torch.optim.Adam([V], lr=LR)
    for ep in range(epochs_p2):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH):
            idx = perm[start:start + BATCH]
            bits = byte_to_bits(cur_d[idx]).float()
            inp = bits * 2.0 - 1.0
            with torch.no_grad():
                latent = unit.encode(inp)
            loss = F.cross_entropy(latent @ V, nxt_d[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()

    elapsed = time.time() - t0

    byte_acc, missed = eval_lossless(unit)
    ds = eval_downstream(unit, corpus)
    return byte_acc, missed, ds, elapsed


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Fixing lossless gap: 1H 8->32->16 with SiLU and GELU\n")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    results = []

    # Strategy A: heavy recon weight
    for act in ["silu", "gelu"]:
        for rw, ep in [(1.0, 50), (5.0, 50), (10.0, 80), (20.0, 100)]:
            label = f"A {act} rw={rw} ep={ep}"
            print(f">>> {label}...")
            ba, mi, ds, t = strategy_A(act, rw, ep, cur, nxt, corpus)
            ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
            print(f"    lossless={ll}  missed={mi}  downstream={ds:.2f}%  [{t:.1f}s]")
            results.append({"label": label, "byte_acc": ba, "missed": mi, "eval_acc": ds, "time": t})

    # Strategy B: 2-phase
    for act in ["silu", "gelu"]:
        label = f"B {act} 2-phase"
        print(f">>> {label}...")
        ba, mi, ds, t = strategy_B(act, 80, 30, cur, nxt, corpus)
        ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
        print(f"    lossless={ll}  missed={mi}  downstream={ds:.2f}%  [{t:.1f}s]")
        results.append({"label": label, "byte_acc": ba, "missed": mi, "eval_acc": ds, "time": t})

    # Sort by: lossless first (100% on top), then downstream
    results.sort(key=lambda r: (-r["byte_acc"], -r["eval_acc"]))

    print(f"\n{'='*75}")
    print(f"{'CONFIG':<28} {'lossless':>10} {'missed':>8} {'downstream':>12}")
    print(f"{'='*75}")
    for r in results:
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        star = " ***" if r["byte_acc"] == 100.0 else ""
        print(f"{r['label']:<28} {ll:>10} {r['missed']:>8} {r['eval_acc']:>11.2f}%{star}")


if __name__ == "__main__":
    main()
