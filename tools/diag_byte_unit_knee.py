"""Find the knee: minimum unit size that still achieves 100% lossless.

C19 + L-BFGS (the winner). Sweep:
  - H (hidden width): 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 48
  - output_dim: 8, 10, 12, 14, 16

For each combo: train, check lossless, check downstream.
Find the smallest config that hits 100%.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
SEED = 42
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

def c19_vec(x, c, rho):
    c_s = c.clamp(min=0.1)
    rho_s = rho.clamp(min=0.0)
    L = 6.0 * c_s
    scaled = x / c_s
    n = scaled.floor()
    t = scaled - n
    h_val = t * (1.0 - t)
    sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
    interior = c_s * (sgn * h_val + rho_s * h_val * h_val)
    return torch.where(x >= L, x - L, torch.where(x <= -L, x + L, interior))


class ByteUnit(torch.nn.Module):
    def __init__(self, H, out_dim):
        super().__init__()
        self.H = H
        self.out_dim = out_dim
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, out_dim) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(out_dim))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def encode(self, x):
        return c19_vec(x @ q8(self.W1) + self.b1, self.c, self.rho) @ q8(self.W2) + self.b2

    def decode(self, z):
        return z @ q8(self.W2).t() @ q8(self.W1).t()

    def param_count(self):
        return N_BITS * self.H + self.H + self.H * self.out_dim + self.out_dim + self.H * 2


def train_lbfgs(unit, cur_d, nxt_d, out_dim):
    V = torch.nn.Parameter(torch.randn(out_dim, VOCAB, device=DEVICE) * 0.1)
    all_params = list(unit.parameters()) + [V]
    opt = torch.optim.LBFGS(all_params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe",
                             history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12)

    all_b = torch.arange(256, device=DEVICE)
    all_bits = byte_to_bits(all_b).float()
    all_inp = all_bits * 2.0 - 1.0

    n_ctx = min(50000, cur_d.shape[0])
    ctx_cur = cur_d[:n_ctx]
    ctx_nxt = nxt_d[:n_ctx]
    ctx_bits = byte_to_bits(ctx_cur).float()
    ctx_inp = ctx_bits * 2.0 - 1.0

    for outer in range(150):
        def closure():
            opt.zero_grad()
            latent_all = unit.encode(all_inp)
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent_all), all_bits)
            latent_ctx = unit.encode(ctx_inp)
            loss_ctx = F.cross_entropy(latent_ctx @ V, ctx_nxt)
            loss = loss_rec + 0.1 * loss_ctx
            loss.backward()
            return loss
        opt.step(closure)


def eval_lossless(unit):
    with torch.no_grad():
        all_b = torch.arange(256, device=DEVICE)
        bits = byte_to_bits(all_b).float()
        inp = bits * 2.0 - 1.0
        pred = (unit.decode(unit.encode(inp)) > 0).float()
        byte_acc = (pred == bits).all(dim=1).float().mean().item() * 100
        missed = int((~(pred == bits).all(dim=1)).sum().item())
    return byte_acc, missed


def eval_downstream(unit, corpus_path, out_dim):
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


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"KNEE FINDER: C19 + L-BFGS, minimum unit size for 100% lossless\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    H_vals = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 48]
    out_vals = [8, 10, 12, 16]

    results = []

    # First: sweep H at fixed out_dim=16 (find H knee)
    print("=== SWEEP H (output_dim=16 fixed) ===")
    print(f"{'H':>4} {'params':>7} {'lossless':>10} {'missed':>8} {'downstream':>12} {'time':>8}")
    print("-" * 60)

    for H in H_vals:
        torch.manual_seed(SEED)
        unit = ByteUnit(H, 16).to(DEVICE)
        pc = unit.param_count()
        t0 = time.time()
        train_lbfgs(unit, cur_d, nxt_d, 16)
        elapsed = time.time() - t0
        ba, mi = eval_lossless(unit)
        ds = eval_downstream(unit, corpus, 16)
        ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
        marker = " <<<" if ba == 100.0 else ""
        print(f"{H:>4} {pc:>7} {ll:>10} {mi:>8} {ds:>11.2f}% {elapsed:>7.1f}s{marker}")
        results.append({"H": H, "out": 16, "params": pc, "byte_acc": ba, "missed": mi, "eval_acc": ds})

    # Find minimum H for 100%
    perfect_h = [r for r in results if r["byte_acc"] == 100.0]
    if perfect_h:
        knee_h = min(perfect_h, key=lambda r: r["H"])
        print(f"\n>>> H knee: H={knee_h['H']} ({knee_h['params']} params)")
    else:
        knee_h = None
        print(f"\n>>> No H achieved 100% lossless at out=16!")

    # Then: sweep output_dim at knee H (find output knee)
    best_H = knee_h["H"] if knee_h else 32
    print(f"\n=== SWEEP output_dim (H={best_H} fixed) ===")
    print(f"{'out':>4} {'params':>7} {'lossless':>10} {'missed':>8} {'downstream':>12} {'time':>8}")
    print("-" * 60)

    results2 = []
    for od in out_vals:
        torch.manual_seed(SEED)
        unit = ByteUnit(best_H, od).to(DEVICE)
        pc = unit.param_count()
        t0 = time.time()
        train_lbfgs(unit, cur_d, nxt_d, od)
        elapsed = time.time() - t0
        ba, mi = eval_lossless(unit)
        ds = eval_downstream(unit, corpus, od)
        ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
        marker = " <<<" if ba == 100.0 else ""
        print(f"{od:>4} {pc:>7} {ll:>10} {mi:>8} {ds:>11.2f}% {elapsed:>7.1f}s{marker}")
        results2.append({"H": best_H, "out": od, "params": pc, "byte_acc": ba, "missed": mi, "eval_acc": ds})

    perfect_out = [r for r in results2 if r["byte_acc"] == 100.0]
    if perfect_out:
        knee_out = min(perfect_out, key=lambda r: r["out"])
        print(f"\n>>> Output dim knee: out={knee_out['out']} ({knee_out['params']} params)")

    # Final summary
    all_perfect = [r for r in results + results2 if r["byte_acc"] == 100.0]
    if all_perfect:
        smallest = min(all_perfect, key=lambda r: r["params"])
        best_ds = max(all_perfect, key=lambda r: r["eval_acc"])
        print(f"\n{'='*65}")
        print(f"  KNEE (smallest 100% lossless):")
        print(f"    H={smallest['H']}, out={smallest['out']}, params={smallest['params']}")
        print(f"    downstream={smallest['eval_acc']:.2f}%")
        print(f"  BEST downstream at 100% lossless:")
        print(f"    H={best_ds['H']}, out={best_ds['out']}, params={best_ds['params']}")
        print(f"    downstream={best_ds['eval_acc']:.2f}%")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
