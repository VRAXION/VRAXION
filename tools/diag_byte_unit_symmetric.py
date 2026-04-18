"""Symmetric mirror byte unit: activation on BOTH encoder AND decoder.

Previous bug: decoder was always linear, couldn't invert the encoder's nonlinearity.
Fix: add matching activation on decoder side too.

Configs:
  - asym_silu:     encoder SiLU, decoder LINEAR (previous baseline)
  - sym_silu:      encoder SiLU, decoder SiLU (symmetric!)
  - sym_gelu:      encoder GELU, decoder GELU
  - sym_silu_skip: symmetric SiLU + residual skip (x @ W_skip added to latent)
  - sym_gelu_skip: symmetric GELU + skip
  - sym_c19:       symmetric C19 learnable (c, rho per neuron)
  - sym_c19_skip:  symmetric C19 + skip
  - sym_beukers:   symmetric soft-sign x/(1+|x|) (Beukers single-input)

All 1H, H=32, 16D output, int8 STE QAT, dual loss (rw=1.0).
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
H = 32
SEED = 42
EPOCHS = 60
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

def beukers_single(x):
    return x / (1.0 + x.abs())


class SymmetricUnit(torch.nn.Module):
    def __init__(self, act_name="sym_silu", use_skip=False):
        super().__init__()
        self.act_name = act_name
        self.use_skip = use_skip

        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1_enc = torch.nn.Parameter(torch.zeros(H))
        self.b1_dec = torch.nn.Parameter(torch.zeros(N_BITS))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2_enc = torch.nn.Parameter(torch.zeros(OUT_DIM))
        self.b2_dec = torch.nn.Parameter(torch.zeros(H))

        if use_skip:
            self.W_skip = torch.nn.Parameter(torch.randn(N_BITS, OUT_DIM) * 0.1)

        if "c19" in act_name:
            # Separate c, rho for encoder (H neurons) and decoder (H neurons)
            self.c_enc = torch.nn.Parameter(torch.ones(H))
            self.rho_enc = torch.nn.Parameter(torch.full((H,), 4.0))
            self.c_dec = torch.nn.Parameter(torch.ones(H))
            self.rho_dec = torch.nn.Parameter(torch.full((H,), 4.0))

    def act_enc(self, x):
        if "silu" in self.act_name: return F.silu(x)
        if "gelu" in self.act_name: return F.gelu(x)
        if "c19" in self.act_name: return c19_vec(x, self.c_enc, self.rho_enc)
        if "beukers" in self.act_name: return beukers_single(x)
        return x  # linear

    def act_dec(self, x):
        if "asym" in self.act_name: return x  # asymmetric = decoder is linear
        if "silu" in self.act_name: return F.silu(x)
        if "gelu" in self.act_name: return F.gelu(x)
        if "c19" in self.act_name: return c19_vec(x, self.c_dec, self.rho_dec)
        if "beukers" in self.act_name: return beukers_single(x)
        return x

    def encode(self, x):
        h = self.act_enc(x @ q8(self.W1) + self.b1_enc)
        latent = h @ q8(self.W2) + self.b2_enc
        if self.use_skip:
            latent = latent + x @ q8(self.W_skip)
        return latent

    def decode(self, z):
        h = self.act_dec(z @ q8(self.W2).t() + self.b2_dec)
        return h @ q8(self.W1).t() + self.b1_dec

    def param_count(self):
        # W1 + b1_enc + b1_dec + W2 + b2_enc + b2_dec
        total = N_BITS * H + H + N_BITS + H * OUT_DIM + OUT_DIM + H
        if self.use_skip:
            total += N_BITS * OUT_DIM
        if "c19" in self.act_name:
            total += H * 4  # c_enc, rho_enc, c_dec, rho_dec
        return total


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


def train_unit(unit, cur, nxt):
    unit = unit.to(DEVICE)
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    opt = torch.optim.Adam(list(unit.parameters()) + [V], lr=LR)
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
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent), bits)
            loss_ctx = F.cross_entropy(latent @ V, nxt_d[idx])
            loss = loss_rec + 0.1 * loss_ctx
            opt.zero_grad(); loss.backward(); opt.step()
    return time.time() - t0


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}, H={H}, out={OUT_DIM}, epochs={EPOCHS}")
    print(f"Symmetric mirror byte unit: act on BOTH encoder AND decoder\n")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    configs = [
        ("asym_silu (baseline)", "asym_silu", False),
        ("sym_silu",             "sym_silu",  False),
        ("sym_silu + skip",      "sym_silu",  True),
        ("sym_gelu",             "sym_gelu",  False),
        ("sym_gelu + skip",      "sym_gelu",  True),
        ("sym_c19 learnable",    "sym_c19",   False),
        ("sym_c19 + skip",       "sym_c19",   True),
        ("sym_beukers",          "sym_beukers", False),
        ("sym_beukers + skip",   "sym_beukers", True),
    ]

    results = []
    for label, act, skip in configs:
        torch.manual_seed(SEED)
        unit = SymmetricUnit(act, skip)
        pc = unit.param_count()
        print(f">>> {label} ({pc} params)...")
        t = train_unit(unit, cur, nxt)
        ba, mi = eval_lossless(unit)
        ds = eval_downstream(unit, corpus)
        ll = "PASS" if ba == 100.0 else f"{ba:.1f}%"
        print(f"    lossless={ll}  missed={mi}  downstream={ds:.2f}%  [{t:.1f}s]")
        results.append({"label": label, "params": pc, "byte_acc": ba,
                        "missed": mi, "eval_acc": ds, "time": t})

    results.sort(key=lambda r: (-r["byte_acc"], -r["eval_acc"]))

    print(f"\n{'='*80}")
    print(f"{'CONFIG':<24} {'params':>7} {'lossless':>10} {'downstream':>12} {'rank':>6}")
    print(f"{'='*80}")
    for i, r in enumerate(results):
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        star = " ***" if r["byte_acc"] == 100.0 and r["eval_acc"] > 40 else ""
        print(f"{r['label']:<24} {r['params']:>7d} {ll:>10} {r['eval_acc']:>11.2f}% {i+1:>6}{star}")

    # Highlight winners
    ll_winners = [r for r in results if r["byte_acc"] == 100.0]
    if ll_winners:
        best = max(ll_winners, key=lambda r: r["eval_acc"])
        print(f"\n>>> LOSSLESS + BEST DOWNSTREAM: {best['label']} — {best['eval_acc']:.2f}%, {best['params']} params ***")
    else:
        near = max(results, key=lambda r: r["byte_acc"])
        print(f"\n>>> CLOSEST TO LOSSLESS: {near['label']} — {near['byte_acc']:.2f}%, downstream={near['eval_acc']:.2f}%")


if __name__ == "__main__":
    main()
