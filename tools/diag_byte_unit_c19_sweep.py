"""2-layer byte unit: activation sweep with learnable C19.

Architecture: 8 -> H (activation) -> 16 (linear), tied mirror.
H in {32, 48} to test if wider hidden helps lossless.

Activations:
  - relu (reference winner)
  - c19_fixed (c=1.0, rho=8.0)
  - c19_learnable (per-neuron trainable c and rho)
  - c19_learnable_init1 (c init=0.5, rho init=1.0 — gentle start)
  - gelu
  - silu (swish)

Key question: can C19 with learnable params achieve 100% lossless + best downstream?
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
EPOCHS = 50
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


# ── C19 vectorized with per-neuron c, rho ─────────────────

def c19_vec(x, c, rho):
    """C19 activation, vectorized. c, rho: (D,) broadcastable."""
    c_safe = c.clamp(min=0.1)
    rho_safe = rho.clamp(min=0.0)
    L = 6.0 * c_safe

    # Linear tails
    above = x - L
    below = x + L

    # Interior: periodic bumps
    scaled = x / c_safe
    n = scaled.floor()
    t = scaled - n
    h = t * (1.0 - t)
    sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
    interior = c_safe * (sgn * h + rho_safe * h * h)

    out = torch.where(x >= L, above, torch.where(x <= -L, below, interior))
    return out


# ── Unit definitions ──────────────────────────────────────

class Unit2Layer(torch.nn.Module):
    """8 -> H (activation) -> 16 (linear), tied mirror decoder."""
    def __init__(self, H, act_type="relu", c_init=1.0, rho_init=8.0):
        super().__init__()
        self.H = H
        self.act_type = act_type
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))

        if "c19" in act_type:
            self.c = torch.nn.Parameter(torch.full((H,), c_init))
            self.rho = torch.nn.Parameter(torch.full((H,), rho_init))

    def activation(self, x):
        if self.act_type == "relu":
            return F.relu(x)
        elif self.act_type == "gelu":
            return F.gelu(x)
        elif self.act_type == "silu":
            return F.silu(x)
        elif self.act_type == "c19_fixed":
            c = torch.ones(self.H, device=x.device)
            rho = torch.full((self.H,), 8.0, device=x.device)
            return c19_vec(x, c, rho)
        elif self.act_type in ("c19_learn", "c19_learn_gentle"):
            return c19_vec(x, self.c, self.rho)
        return x

    def encode(self, x):
        h = self.activation(x @ q8(self.W1) + self.b1)
        return h @ q8(self.W2) + self.b2

    def decode(self, z):
        h = z @ q8(self.W2).t()
        return h @ q8(self.W1).t()

    def param_count(self):
        base = N_BITS * self.H + self.H + self.H * OUT_DIM + OUT_DIM
        if "c19" in self.act_type and self.act_type != "c19_fixed":
            base += self.H * 2  # c + rho per neuron
        return base


def train_and_eval(label, unit, cur, nxt, corpus_path):
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
    elapsed = time.time() - t0

    # Lossless
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

    # Report C19 learned params if applicable
    c_rho_str = ""
    if hasattr(unit, 'c') and unit.act_type != "c19_fixed":
        c_rho_str = f"  c=[{unit.c.min().item():.2f}..{unit.c.max().item():.2f}]  rho=[{unit.rho.min().item():.2f}..{unit.rho.max().item():.2f}]"

    return {
        "label": label, "params": unit.param_count(),
        "byte_acc": byte_acc, "missed": missed,
        "eval_acc": eval_acc, "time": elapsed,
        "c_rho": c_rho_str,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}, output={OUT_DIM}D, epochs={EPOCHS}")
    print(f"2-layer byte unit: activation sweep including C19 learnable\n")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    configs = [
        ("relu H=32",           32, "relu",             1.0, 8.0),
        ("relu H=48",           48, "relu",             1.0, 8.0),
        ("gelu H=32",           32, "gelu",             1.0, 8.0),
        ("silu H=32",           32, "silu",             1.0, 8.0),
        ("c19_fixed H=32",      32, "c19_fixed",        1.0, 8.0),
        ("c19_learn H=32",      32, "c19_learn",        1.0, 8.0),
        ("c19_gentle H=32",     32, "c19_learn_gentle", 0.5, 1.0),
        ("c19_learn H=48",      48, "c19_learn",        1.0, 8.0),
        ("c19_gentle H=48",     48, "c19_learn_gentle", 0.5, 1.0),
    ]

    results = []
    for label, H, act, c_init, rho_init in configs:
        torch.manual_seed(SEED)
        unit = Unit2Layer(H, act, c_init, rho_init).to(DEVICE)
        print(f">>> {label} ({unit.param_count()} params)...")
        r = train_and_eval(label, unit, cur, nxt, corpus)
        ll_str = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        print(f"    lossless={ll_str}  downstream={r['eval_acc']:.2f}%  [{r['time']:.1f}s]{r['c_rho']}")
        results.append(r)

    results.sort(key=lambda r: -r["eval_acc"])

    print(f"\n{'='*82}")
    print(f"{'CONFIG':<22} {'params':>7} {'lossless':>10} {'downstream':>12} {'rank':>6}")
    print(f"{'='*82}")
    for i, r in enumerate(results):
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        star = " ***" if r["byte_acc"] == 100.0 and r["eval_acc"] > 40 else ""
        print(f"{r['label']:<22} {r['params']:>7d} {ll:>10} {r['eval_acc']:>11.2f}% {i+1:>6}{star}")

    # Find best that is also lossless
    lossless_winners = [r for r in results if r["byte_acc"] == 100.0]
    if lossless_winners:
        w = max(lossless_winners, key=lambda r: r["eval_acc"])
        print(f"\n>>> BEST LOSSLESS: {w['label']} — downstream={w['eval_acc']:.2f}%, {w['params']} params")
    else:
        print(f"\n>>> No config achieved 100% lossless.")

    overall = results[0]
    print(f">>> BEST OVERALL: {overall['label']} — downstream={overall['eval_acc']:.2f}%, lossless={overall['byte_acc']:.2f}%, {overall['params']} params")


if __name__ == "__main__":
    main()
