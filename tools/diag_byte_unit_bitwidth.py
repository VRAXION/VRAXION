"""Bitwidth sweep: how far can we quantize the byte unit weights?

Architecture: C19 + L-BFGS, H=24, out=16 (the knee winner).
Sweep weight precision: int8, int6, int5, int4, int3, ternary (int2), binary (int1).

For each: train with L-BFGS + STE at that precision, measure lossless + downstream.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
H = 24
OUT_DIM = 16
SEED = 42
VOCAB = 256
CTX = 8
MASK_POS = 4


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


# ── Multi-precision STE ────────────────────────────────────

class QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale, max_val):
        q = torch.clamp(torch.round(w / scale), -max_val, max_val)
        return q * scale
    @staticmethod
    def backward(ctx, g):
        return g, None, None

def quantize(W, bits):
    """Quantize W to given bit width. bits=8 → int8, bits=1 → binary {-1,+1}."""
    if bits >= 16:
        return W  # float, no quantization
    max_val = (2 ** (bits - 1)) - 1  # int8: 127, int4: 7, int1: 1 (binary), etc.
    if bits == 1:
        max_val = 1  # binary: {-1, +1}
    scale = W.abs().max().detach().clamp(min=1e-8) / max(max_val, 1)
    return QuantSTE.apply(W, scale, max_val)


BITWIDTHS = [
    (16, "float32",  "no quant"),
    (8,  "int8",     "256 levels"),
    (6,  "int6",     "64 levels"),
    (5,  "int5",     "32 levels"),
    (4,  "int4",     "16 levels"),
    (3,  "int3",     "8 levels"),
    (2,  "ternary",  "{-1,0,+1}"),
    (1,  "binary",   "{-1,+1}"),
]


class ByteUnit(torch.nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def qW1(self): return quantize(self.W1, self.bits)
    def qW2(self): return quantize(self.W2, self.bits)

    def encode(self, x):
        return c19_vec(x @ self.qW1() + self.b1, self.c, self.rho) @ self.qW2() + self.b2

    def decode(self, z):
        return z @ self.qW2().t() @ self.qW1().t()

    def weight_count(self):
        return N_BITS * H + H * OUT_DIM  # only weight params (not biases, not c/rho)

    def total_params(self):
        return N_BITS * H + H + H * OUT_DIM + OUT_DIM + H * 2

    def storage_bytes(self):
        """Actual bytes needed to store weights at this precision."""
        n_weights = self.weight_count()
        if self.bits >= 16:
            return n_weights * 4  # float32
        elif self.bits == 1:
            return (n_weights + 7) // 8  # 1 bit per weight, packed
        else:
            return (n_weights * self.bits + 7) // 8  # packed bits
        # Plus biases + c + rho always float32
        # (simplified: just weight storage for comparison)


def train_lbfgs(unit, cur_d, nxt_d):
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    all_params = list(unit.parameters()) + [V]
    opt = torch.optim.LBFGS(all_params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe",
                             history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12)

    all_b = torch.arange(256, device=DEVICE)
    all_bits = byte_to_bits(all_b).float()
    all_inp = all_bits * 2.0 - 1.0

    n_ctx = min(50000, cur_d.shape[0])
    ctx_inp = byte_to_bits(cur_d[:n_ctx]).float() * 2.0 - 1.0
    ctx_nxt = nxt_d[:n_ctx]

    for outer in range(200):
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


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Bitwidth sweep: C19 + L-BFGS, H={H}, out={OUT_DIM}")
    print(f"How far can we quantize and keep 100% lossless?\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    n_weights = N_BITS * H + H * OUT_DIM  # 192 + 384 = 576 weights

    results = []
    print(f"{'bits':>5} {'name':>8} {'w_bytes':>8} {'lossless':>10} {'missed':>8} {'downstream':>12} {'time':>8}")
    print("=" * 72)

    for bits, name, desc in BITWIDTHS:
        torch.manual_seed(SEED)
        unit = ByteUnit(bits).to(DEVICE)
        t0 = time.time()
        train_lbfgs(unit, cur_d, nxt_d)
        elapsed = time.time() - t0
        ba, mi = eval_lossless(unit)
        ds = eval_downstream(unit, corpus)

        w_bytes = unit.storage_bytes()
        ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
        marker = " <<<" if ba == 100.0 else ""
        print(f"{bits:>5} {name:>8} {w_bytes:>7}B {ll:>10} {mi:>8} {ds:>11.2f}% {elapsed:>7.1f}s{marker}")
        results.append({"bits": bits, "name": name, "desc": desc,
                        "w_bytes": w_bytes, "byte_acc": ba, "missed": mi,
                        "eval_acc": ds, "time": elapsed})

    # Summary
    print(f"\n{'='*72}")
    print(f"  BITWIDTH PARETO (C19 + L-BFGS, H={H}, out={OUT_DIM})")
    print(f"{'='*72}")
    print(f"{'precision':<10} {'storage':>10} {'compression':>12} {'lossless':>10} {'downstream':>12}")
    print(f"{'-'*72}")
    float_bytes = n_weights * 4
    for r in results:
        comp = f"{float_bytes / max(r['w_bytes'],1):.0f}x"
        ll = "100%" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        print(f"{r['name']:<10} {r['w_bytes']:>9}B {comp:>12} {ll:>10} {r['eval_acc']:>11.2f}%")

    # Find the knee
    perfect = [r for r in results if r["byte_acc"] == 100.0]
    if perfect:
        smallest = min(perfect, key=lambda r: r["w_bytes"])
        print(f"\n>>> MINIMUM PRECISION FOR 100% LOSSLESS: {smallest['name']}")
        print(f"    storage: {smallest['w_bytes']} bytes ({float_bytes/smallest['w_bytes']:.0f}x compression vs float32)")
        print(f"    downstream: {smallest['eval_acc']:.2f}%")


if __name__ == "__main__":
    main()
