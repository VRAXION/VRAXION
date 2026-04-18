"""Byte unit with L-BFGS (Hessian-based) optimization.

Hypothesis: Adam gets stuck in C19's oscillating landscape.
L-BFGS uses curvature info → might escape shallow local minima.

Our problem is TINY (816 params, 256 bytes) = perfect for L-BFGS.

Test: same architectures as before, but L-BFGS full-batch instead of Adam mini-batch.
  - SiLU + Adam (reference)
  - SiLU + L-BFGS
  - GELU + L-BFGS
  - C19 learnable + Adam (reference: catastrophic)
  - C19 learnable + L-BFGS (the real test!)
  - C19 learnable + L-BFGS + higher recon weight

All 1H 8->32->16, int8 STE QAT, dual loss.
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
    def __init__(self, act="silu"):
        super().__init__()
        self.act_name = act
        self.W1 = torch.nn.Parameter(torch.randn(N_BITS, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, OUT_DIM) * 0.3)
        self.b2 = torch.nn.Parameter(torch.zeros(OUT_DIM))
        if act == "c19":
            self.c = torch.nn.Parameter(torch.ones(H))
            self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def activation(self, x):
        if self.act_name == "silu": return F.silu(x)
        if self.act_name == "gelu": return F.gelu(x)
        if self.act_name == "c19": return c19_vec(x, self.c, self.rho)
        return F.relu(x)

    def encode(self, x):
        return self.activation(x @ q8(self.W1) + self.b1) @ q8(self.W2) + self.b2

    def decode(self, z):
        return z @ q8(self.W2).t() @ q8(self.W1).t()

    def param_count(self):
        p = N_BITS * H + H + H * OUT_DIM + OUT_DIM
        if self.act_name == "c19": p += H * 2
        return p


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


def train_adam(unit, cur_d, nxt_d, recon_w=1.0, epochs=60, lr=0.01):
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    opt = torch.optim.Adam(list(unit.parameters()) + [V], lr=lr)
    N = cur_d.shape[0]
    BATCH = 8192

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


def train_lbfgs(unit, cur_d, nxt_d, recon_w=1.0, max_iter=200, lr=1.0):
    """Full-batch L-BFGS training."""
    V = torch.nn.Parameter(torch.randn(OUT_DIM, VOCAB, device=DEVICE) * 0.1)
    all_params = list(unit.parameters()) + [V]
    opt = torch.optim.LBFGS(all_params, lr=lr, max_iter=20, line_search_fn="strong_wolfe",
                             history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12)

    # Full batch: ALL bigrams (or a large subsample for context loss)
    # For reconstruction: use all 256 bytes
    all_b = torch.arange(256, device=DEVICE)
    all_bits = byte_to_bits(all_b).float()
    all_inp = all_bits * 2.0 - 1.0

    # Subsample bigrams for context loss (keep manageable)
    n_ctx = min(50000, cur_d.shape[0])
    ctx_cur = cur_d[:n_ctx]
    ctx_nxt = nxt_d[:n_ctx]
    ctx_bits = byte_to_bits(ctx_cur).float()
    ctx_inp = ctx_bits * 2.0 - 1.0

    for outer in range(max_iter):
        def closure():
            opt.zero_grad()
            # Reconstruction on all 256 bytes
            latent_all = unit.encode(all_inp)
            loss_rec = F.binary_cross_entropy_with_logits(unit.decode(latent_all), all_bits)
            # Context on bigram subsample
            latent_ctx = unit.encode(ctx_inp)
            loss_ctx = F.cross_entropy(latent_ctx @ V, ctx_nxt)
            loss = recon_w * loss_rec + 0.1 * loss_ctx
            loss.backward()
            return loss

        opt.step(closure)

        if outer % 20 == 0 or outer == max_iter - 1:
            ba, mi = eval_lossless(unit)
            if ba == 100.0:
                print(f"      iter={outer}: 100% lossless reached!")
                return


def run_config(label, act, optimizer, recon_w, cur_d, nxt_d, corpus):
    torch.manual_seed(SEED)
    unit = ByteUnit(act).to(DEVICE)
    pc = unit.param_count()

    t0 = time.time()
    if optimizer == "adam":
        train_adam(unit, cur_d, nxt_d, recon_w=recon_w, epochs=60)
    elif optimizer == "lbfgs":
        train_lbfgs(unit, cur_d, nxt_d, recon_w=recon_w, max_iter=150)
    elapsed = time.time() - t0

    ba, mi = eval_lossless(unit)
    ds = eval_downstream(unit, corpus)

    ll_str = "PASS" if ba == 100.0 else f"{ba:.1f}%"

    # C19 params
    extra = ""
    if hasattr(unit, 'c'):
        extra = f"  c=[{unit.c.min().item():.2f}..{unit.c.max().item():.2f}] rho=[{unit.rho.min().item():.2f}..{unit.rho.max().item():.2f}]"

    print(f"    lossless={ll_str}  missed={mi}  downstream={ds:.2f}%  [{elapsed:.1f}s]{extra}")
    return {"label": label, "params": pc, "byte_acc": ba, "missed": mi,
            "eval_acc": ds, "time": elapsed}


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"L-BFGS vs Adam: can Hessian info fix C19 and improve lossless?\n")

    cur, nxt = load_bigrams(corpus, 200_000)
    cur_d, nxt_d = cur.to(DEVICE), nxt.to(DEVICE)

    configs = [
        # (label, act, optimizer, recon_w)
        ("SiLU Adam rw=1",       "silu", "adam",  1.0),
        ("SiLU L-BFGS rw=1",     "silu", "lbfgs", 1.0),
        ("SiLU L-BFGS rw=5",     "silu", "lbfgs", 5.0),
        ("GELU Adam rw=1",       "gelu", "adam",  1.0),
        ("GELU L-BFGS rw=1",     "gelu", "lbfgs", 1.0),
        ("C19 Adam rw=1",        "c19",  "adam",  1.0),
        ("C19 L-BFGS rw=1",      "c19",  "lbfgs", 1.0),
        ("C19 L-BFGS rw=5",      "c19",  "lbfgs", 5.0),
        ("C19 L-BFGS rw=10",     "c19",  "lbfgs", 10.0),
    ]

    results = []
    for label, act, opt_name, rw in configs:
        print(f">>> {label}...")
        r = run_config(label, act, opt_name, rw, cur_d, nxt_d, corpus)
        results.append(r)

    results.sort(key=lambda r: (-r["byte_acc"], -r["eval_acc"]))

    print(f"\n{'='*78}")
    print(f"{'CONFIG':<24} {'lossless':>10} {'missed':>8} {'downstream':>12} {'time':>8}")
    print(f"{'='*78}")
    for r in results:
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        star = " ***" if r["byte_acc"] == 100.0 else ""
        print(f"{r['label']:<24} {ll:>10} {r['missed']:>8} {r['eval_acc']:>11.2f}% {r['time']:>7.1f}s{star}")

    # Key comparison
    print(f"\n>>> Key question: Does L-BFGS fix C19?")
    c19_adam = next((r for r in results if "C19 Adam" in r["label"]), None)
    c19_lbfgs = [r for r in results if "C19 L-BFGS" in r["label"]]
    if c19_adam:
        print(f"    C19 + Adam:  lossless={c19_adam['byte_acc']:.1f}%")
    for r in c19_lbfgs:
        print(f"    {r['label']}: lossless={r['byte_acc']:.1f}%  downstream={r['eval_acc']:.2f}%")


if __name__ == "__main__":
    main()
