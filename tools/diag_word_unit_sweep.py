"""Word embedder unit sweep: find the best way to combine byte embeddings.

Uses the baked byte tokenizer (from byte_unit_winner_int4.json).
Tests different word-unit architectures on a real task:
  predict the NEXT byte given a word's embedding.

Candidates:
  A) Flat concat: N x 16D = N*16D input → C19 neurons → 16D output
  B) Pairwise tree: pairs of 16D → 16D, repeated in rounds (tournament)
  C) Mean pool: average all byte embeddings (zero params, baseline)
  D) Max pool: max over all byte embeddings (zero params)
  E) Weighted sum: learnable attention weight per position → weighted avg
  F) Conv-style: sliding window of 2-3 bytes → C19 → pool

All with int4 QAT + L-BFGS where applicable.
Context: 8 byte window, predict next byte.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
VOCAB = 256
CTX = 8
EMBED_DIM = 16
N_BIGRAMS = 200_000


class Int8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        return torch.clamp(torch.round(w / scale), -127, 127) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None, None

def q4(W):
    s = W.abs().max().detach().clamp(min=1e-8) / 7.0
    return Int8STE.apply(W, s, None) if False else _q4(W)

def _q4(W):
    s = W.abs().max().detach().clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(W / s), -7, 7)
    return q * s + (W - (q * s)).detach()  # STE trick: forward=quantized, backward=float

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


# ── Load frozen byte embeddings ───────────────────────────

def load_byte_embedder():
    """Load the baked byte unit and compute 256 embeddings."""
    with open("tools/byte_unit_winner_int4.json") as f:
        data = json.load(f)
    W1 = torch.tensor(data["W1_int4"], dtype=torch.float32, device=DEVICE)
    W2 = torch.tensor(data["W2_int4"], dtype=torch.float32, device=DEVICE)
    b1 = torch.tensor(data["bias1"], device=DEVICE)
    b2 = torch.tensor(data["bias2"], device=DEVICE)
    c = torch.tensor(data["c19_c"], device=DEVICE)
    rho = torch.tensor(data["c19_rho"], device=DEVICE)
    sW1 = data["scale_W1"]
    sW2 = data["scale_W2"]

    # Pre-compute all 256 byte embeddings
    all_b = torch.arange(256, device=DEVICE)
    bits = torch.zeros(256, 8, device=DEVICE)
    for i in range(8):
        bits[:, i] = (all_b >> i) & 1
    inp = bits * 2.0 - 1.0

    hidden = c19_vec(inp @ (W1 * sW1) + b1, c, rho)
    embeddings = hidden @ (W2 * sW2) + b2  # (256, 16)
    return embeddings.detach()


# ── Data loading ──────────────────────────────────────────

def load_samples(path, n, seed=SEED):
    raw = Path(path).read_bytes()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(arr) - CTX - 2, (n,), generator=gen)
    idx = offs.unsqueeze(1) + torch.arange(CTX + 1).unsqueeze(0)
    chunks = arr[idx].long()
    context = chunks[:, :CTX]   # 8 bytes of context
    target = chunks[:, CTX]     # next byte to predict
    return context.to(DEVICE), target.to(DEVICE)


def embed_context(byte_embs, context):
    """(256, 16) embeddings + (N, CTX) byte indices → (N, CTX, 16) embedded."""
    return byte_embs[context]  # (N, CTX, 16)


# ── Word unit candidates ─────────────────────────────────

class MeanPool(torch.nn.Module):
    """Baseline: just average all byte embeddings."""
    def __init__(self):
        super().__init__()
    def forward(self, x):  # (N, CTX, 16) → (N, 16)
        return x.mean(dim=1)
    def param_count(self): return 0

class MaxPool(torch.nn.Module):
    """Baseline: max over all byte embeddings."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.max(dim=1).values
    def param_count(self): return 0

class FlatConcat(torch.nn.Module):
    """Flatten CTX*16D → C19 hidden → 16D output."""
    def __init__(self, H=48):
        super().__init__()
        self.H = H
        D_in = CTX * EMBED_DIM  # 128
        self.W1 = torch.nn.Parameter(torch.randn(D_in, H) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, EMBED_DIM) * 0.1)
        self.b2 = torch.nn.Parameter(torch.zeros(EMBED_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def forward(self, x):  # (N, CTX, 16) → (N, 16)
        flat = x.reshape(x.shape[0], -1)  # (N, 128)
        h = c19_vec(flat @ _q4(self.W1) + self.b1, self.c, self.rho)
        return h @ _q4(self.W2) + self.b2

    def param_count(self):
        return CTX * EMBED_DIM * self.H + self.H + self.H * EMBED_DIM + EMBED_DIM + self.H * 2

class PairwiseTree(torch.nn.Module):
    """Tournament: pair 2x16D → 16D, repeat until 1 left. Single shared unit."""
    def __init__(self, H=32):
        super().__init__()
        self.H = H
        D_in = 2 * EMBED_DIM  # 32
        self.W1 = torch.nn.Parameter(torch.randn(D_in, H) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, EMBED_DIM) * 0.1)
        self.b2 = torch.nn.Parameter(torch.zeros(EMBED_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def pair_merge(self, a, b):
        """Merge two (N, 16) → (N, 16)."""
        cat = torch.cat([a, b], dim=-1)  # (N, 32)
        h = c19_vec(cat @ _q4(self.W1) + self.b1, self.c, self.rho)
        return h @ _q4(self.W2) + self.b2

    def forward(self, x):  # (N, CTX, 16) → (N, 16)
        # Pad to power of 2 if needed
        N, T, D = x.shape
        while T > 1:
            if T % 2 == 1:
                x = torch.cat([x, x[:, -1:, :]], dim=1)  # duplicate last
                T += 1
            pairs = []
            for i in range(0, T, 2):
                pairs.append(self.pair_merge(x[:, i, :], x[:, i+1, :]))
            x = torch.stack(pairs, dim=1)
            T = x.shape[1]
        return x[:, 0, :]

    def param_count(self):
        return 2 * EMBED_DIM * self.H + self.H + self.H * EMBED_DIM + EMBED_DIM + self.H * 2

class PairwiseTreeMulti(torch.nn.Module):
    """Tournament with SEPARATE unit per round (3 rounds for CTX=8)."""
    def __init__(self, H=24):
        super().__init__()
        self.H = H
        self.n_rounds = 3  # log2(8) = 3
        D_in = 2 * EMBED_DIM
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_rounds):
            layer = torch.nn.ModuleDict({
                'W1': torch.nn.Linear(1, 1),  # placeholder
            })
            self.layers.append(layer)

        # Actually build properly
        self.W1s = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(D_in, H) * 0.1) for _ in range(self.n_rounds)])
        self.b1s = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(H)) for _ in range(self.n_rounds)])
        self.W2s = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(H, EMBED_DIM) * 0.1) for _ in range(self.n_rounds)])
        self.b2s = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(EMBED_DIM)) for _ in range(self.n_rounds)])
        self.cs = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(H)) for _ in range(self.n_rounds)])
        self.rhos = torch.nn.ParameterList([torch.nn.Parameter(torch.full((H,), 4.0)) for _ in range(self.n_rounds)])

    def forward(self, x):  # (N, 8, 16) → (N, 16)
        for r in range(self.n_rounds):
            N, T, D = x.shape
            pairs = []
            for i in range(0, T, 2):
                cat = torch.cat([x[:, i, :], x[:, i+1, :]], dim=-1)
                h = c19_vec(cat @ _q4(self.W1s[r]) + self.b1s[r], self.cs[r], self.rhos[r])
                pairs.append(h @ _q4(self.W2s[r]) + self.b2s[r])
            x = torch.stack(pairs, dim=1)
        return x[:, 0, :]

    def param_count(self):
        per = 2 * EMBED_DIM * self.H + self.H + self.H * EMBED_DIM + EMBED_DIM + self.H * 2
        return per * self.n_rounds

class WeightedSum(torch.nn.Module):
    """Learnable per-position attention weights."""
    def __init__(self):
        super().__init__()
        self.pos_weights = torch.nn.Parameter(torch.zeros(CTX))

    def forward(self, x):  # (N, CTX, 16) → (N, 16)
        w = F.softmax(self.pos_weights, dim=0)  # (CTX,)
        return (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    def param_count(self): return CTX

class ConvPool(torch.nn.Module):
    """Sliding window: 3 consecutive bytes → C19 → max pool."""
    def __init__(self, H=24):
        super().__init__()
        self.H = H
        D_in = 3 * EMBED_DIM  # 48
        self.W1 = torch.nn.Parameter(torch.randn(D_in, H) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(H))
        self.W2 = torch.nn.Parameter(torch.randn(H, EMBED_DIM) * 0.1)
        self.b2 = torch.nn.Parameter(torch.zeros(EMBED_DIM))
        self.c = torch.nn.Parameter(torch.ones(H))
        self.rho = torch.nn.Parameter(torch.full((H,), 4.0))

    def forward(self, x):  # (N, CTX, 16) → (N, 16)
        N, T, D = x.shape
        # Pad: duplicate first and last
        x_pad = torch.cat([x[:, :1, :], x, x[:, -1:, :]], dim=1)  # (N, T+2, D)
        windows = []
        for i in range(T):
            w = x_pad[:, i:i+3, :].reshape(N, -1)  # (N, 48)
            windows.append(w)
        stacked = torch.stack(windows, dim=1)  # (N, T, 48)
        # Apply C19 to each window
        flat = stacked.reshape(N * T, -1)
        h = c19_vec(flat @ _q4(self.W1) + self.b1, self.c, self.rho)
        out = h @ _q4(self.W2) + self.b2  # (N*T, 16)
        out = out.reshape(N, T, EMBED_DIM)
        return out.max(dim=1).values  # max pool over positions

    def param_count(self):
        return 3 * EMBED_DIM * self.H + self.H + self.H * EMBED_DIM + EMBED_DIM + self.H * 2


# ── Training + evaluation ─────────────────────────────────

def train_and_eval(label, model, byte_embs, train_ctx, train_tgt, eval_ctx, eval_tgt):
    model = model.to(DEVICE)
    V = torch.nn.Parameter(torch.randn(EMBED_DIM, VOCAB, device=DEVICE) * 0.1)
    all_params = list(model.parameters()) + [V]
    trainable = sum(p.numel() for p in all_params if p.requires_grad)

    # Use L-BFGS for small models, Adam for larger
    use_lbfgs = trainable < 5000

    if use_lbfgs:
        opt = torch.optim.LBFGS(all_params, lr=1.0, max_iter=15, line_search_fn="strong_wolfe",
                                 history_size=30)
        train_emb = embed_context(byte_embs, train_ctx)

        t0 = time.time()
        for outer in range(100):
            def closure():
                opt.zero_grad()
                word_emb = model(train_emb)
                loss = F.cross_entropy(word_emb @ V, train_tgt)
                loss.backward()
                return loss
            opt.step(closure)
    else:
        opt = torch.optim.Adam(all_params, lr=0.005)
        train_emb = embed_context(byte_embs, train_ctx)

        t0 = time.time()
        BATCH = 4096
        N = train_ctx.shape[0]
        for ep in range(40):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, BATCH):
                idx = perm[start:start + BATCH]
                word_emb = model(train_emb[idx])
                loss = F.cross_entropy(word_emb @ V, train_tgt[idx])
                opt.zero_grad(); loss.backward(); opt.step()

    elapsed = time.time() - t0

    with torch.no_grad():
        eval_emb = embed_context(byte_embs, eval_ctx)
        word_emb = model(eval_emb)
        logits = word_emb @ V
        eval_acc = logits.argmax(1).eq(eval_tgt).float().mean().item() * 100

        train_word = model(train_emb[:5000])
        train_acc = (train_word @ V).argmax(1).eq(train_tgt[:5000]).float().mean().item() * 100

    return {
        "label": label, "params": model.param_count(), "trainable": trainable,
        "train_acc": train_acc, "eval_acc": eval_acc, "time": elapsed,
        "optimizer": "L-BFGS" if use_lbfgs else "Adam",
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}")
    print(f"Word embedder sweep: best way to combine byte embeddings\n")

    byte_embs = load_byte_embedder()
    print(f"Byte embeddings loaded: {byte_embs.shape}")

    train_ctx, train_tgt = load_samples(corpus, 50000, seed=42)
    eval_ctx, eval_tgt = load_samples(corpus, 10000, seed=99)
    print(f"Train: {train_ctx.shape[0]}, Eval: {eval_ctx.shape[0]}")
    print(f"Task: given 8 byte context, predict next byte\n")

    # Flat byte baseline (no word unit, just concat raw embeddings + linear)
    print(">>> Flat baseline (concat 8x16D, linear probe)...")
    with torch.no_grad():
        tr_emb = embed_context(byte_embs, train_ctx).reshape(train_ctx.shape[0], -1)
        ev_emb = embed_context(byte_embs, eval_ctx).reshape(eval_ctx.shape[0], -1)
    torch.manual_seed(SEED)
    P = torch.nn.Parameter(torch.randn(CTX * EMBED_DIM, VOCAB, device=DEVICE) * 0.01)
    pb = torch.nn.Parameter(torch.zeros(VOCAB, device=DEVICE))
    opt_b = torch.optim.Adam([P, pb], lr=0.005)
    for _ in range(100):
        loss = F.cross_entropy(tr_emb @ P + pb, train_tgt)
        opt_b.zero_grad(); loss.backward(); opt_b.step()
    with torch.no_grad():
        bl_acc = (ev_emb @ P + pb).argmax(1).eq(eval_tgt).float().mean().item() * 100
    print(f"    Flat baseline eval: {bl_acc:.2f}% (no word unit, raw concat)\n")

    configs = [
        ("A) Mean pool",        MeanPool()),
        ("B) Max pool",         MaxPool()),
        ("C) Weighted sum",     WeightedSum()),
        ("D) Pair-tree shared", PairwiseTree(H=32)),
        ("E) Pair-tree x3",    PairwiseTreeMulti(H=24)),
        ("F) Flat C19 H=32",   FlatConcat(H=32)),
        ("G) Flat C19 H=48",   FlatConcat(H=48)),
        ("H) Conv3 pool H=24", ConvPool(H=24)),
    ]

    results = []
    for label, model in configs:
        torch.manual_seed(SEED)
        # Reinit params
        for p in model.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, 0, 0.1)
            elif p.dim() == 1 and p.shape[0] > EMBED_DIM:
                torch.nn.init.ones_(p)  # c params

        print(f">>> {label} ({model.param_count()} params)...")
        r = train_and_eval(label, model, byte_embs, train_ctx, train_tgt, eval_ctx, eval_tgt)
        print(f"    train={r['train_acc']:.2f}%  eval={r['eval_acc']:.2f}%  [{r['time']:.1f}s] ({r['optimizer']})")
        results.append(r)

    results.sort(key=lambda r: -r["eval_acc"])

    print(f"\n{'='*80}")
    print(f"{'WORD UNIT':<24} {'params':>7} {'eval%':>8} {'train%':>8} {'opt':>8} {'time':>7}")
    print(f"{'='*80}")
    print(f"{'Flat baseline (ref)':<24} {'—':>7} {bl_acc:>7.2f}% {'—':>8} {'Adam':>8} {'—':>7}")
    for r in results:
        delta = r['eval_acc'] - bl_acc
        star = " ***" if r['eval_acc'] > bl_acc + 1 else ""
        print(f"{r['label']:<24} {r['params']:>7} {r['eval_acc']:>7.2f}% {r['train_acc']:>7.2f}% {r['optimizer']:>8} {r['time']:>6.1f}s{star}")

    winner = results[0]
    print(f"\n>>> WINNER: {winner['label']} — eval={winner['eval_acc']:.2f}% (+{winner['eval_acc']-bl_acc:.2f}pp vs flat baseline)")


if __name__ == "__main__":
    main()
