"""Deep byte unit sweep: 2 hidden layers + C19 residual.

Architecture: 8 → H1 (act) → H2 (act) → 16 (linear), tied mirror decoder.
Also tests C19+residual: output = alpha * C19(x) + (1-alpha) * x per neuron.

Configs:
  - 1H relu     8→32→16              (baseline from before)
  - 2H relu     8→32→24→16           (2 hidden, relu)
  - 2H relu     8→48→32→16           (2 hidden, wider)
  - 2H gelu     8→32→24→16           (2 hidden, gelu)
  - 2H gelu     8→48→32→16           (2 hidden, wider gelu)
  - 2H silu     8→32→24→16           (2 hidden, silu)
  - 1H c19res   8→32→16              (c19 + residual bypass)
  - 2H c19res   8→32→24→16           (2 hidden, c19 residual)
  - 2H c19res   8→48→32→16           (2 hidden, wider c19 residual)

All int8 QAT, dual loss (recon + context), dim=16 output.
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
    h = t * (1.0 - t)
    sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
    interior = c_s * (sgn * h + rho_s * h * h)
    return torch.where(x >= L, x - L, torch.where(x <= -L, x + L, interior))


ACT_FNS = {
    "relu": lambda x, **kw: F.relu(x),
    "gelu": lambda x, **kw: F.gelu(x),
    "silu": lambda x, **kw: F.silu(x),
}


class DeepUnit(torch.nn.Module):
    def __init__(self, hidden_sizes, act_name="relu"):
        """hidden_sizes: list of hidden dims, e.g. [32] or [32, 24]."""
        super().__init__()
        self.act_name = act_name
        self.n_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        # Build encoder weights
        dims = [N_BITS] + list(hidden_sizes) + [OUT_DIM]
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for i in range(len(dims) - 1):
            self.weights.append(torch.nn.Parameter(torch.randn(dims[i], dims[i+1]) * 0.3))
            self.biases.append(torch.nn.Parameter(torch.zeros(dims[i+1])))

        # C19 residual params (per hidden neuron)
        self.is_c19res = act_name == "c19res"
        if self.is_c19res:
            self.c_params = torch.nn.ParameterList()
            self.rho_params = torch.nn.ParameterList()
            self.alpha_params = torch.nn.ParameterList()
            for hs in hidden_sizes:
                self.c_params.append(torch.nn.Parameter(torch.ones(hs)))
                self.rho_params.append(torch.nn.Parameter(torch.full((hs,), 4.0)))
                self.alpha_params.append(torch.nn.Parameter(torch.full((hs,), 0.5)))

    def activation(self, x, layer_idx):
        if self.is_c19res:
            alpha = torch.sigmoid(self.alpha_params[layer_idx])
            c19_out = c19_vec(x, self.c_params[layer_idx], self.rho_params[layer_idx])
            return alpha * c19_out + (1.0 - alpha) * x
        return ACT_FNS[self.act_name](x)

    def encode(self, x):
        h = x
        for i in range(len(self.weights)):
            h = h @ q8(self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:  # activation on hidden, NOT on output
                h = self.activation(h, i)
        return h

    def decode(self, z):
        h = z
        for i in range(len(self.weights) - 1, -1, -1):
            h = h @ q8(self.weights[i]).t()
        return h

    def param_count(self):
        total = 0
        dims = [N_BITS] + list(self.hidden_sizes) + [OUT_DIM]
        for i in range(len(dims) - 1):
            total += dims[i] * dims[i+1] + dims[i+1]
        if self.is_c19res:
            for hs in self.hidden_sizes:
                total += hs * 3  # c, rho, alpha
        return total


def train_and_eval(label, unit, cur, nxt, corpus_path):
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

    # Downstream
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

    # C19 residual: report learned alphas
    alpha_str = ""
    if unit.is_c19res:
        for li, ap in enumerate(unit.alpha_params):
            a = torch.sigmoid(ap)
            alpha_str += f" L{li+1}_alpha=[{a.min().item():.2f}..{a.max().item():.2f}]"

    return {
        "label": label, "params": unit.param_count(),
        "byte_acc": byte_acc, "missed": missed,
        "eval_acc": eval_acc, "time": elapsed, "alpha": alpha_str,
    }


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else \
        "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

    print(f"Device: {DEVICE}, output={OUT_DIM}D, epochs={EPOCHS}")
    print(f"Deep byte unit sweep: 1H vs 2H, relu/gelu/silu/c19res\n")

    cur, nxt = load_bigrams(corpus, N_BIGRAMS)

    configs = [
        # (label, hidden_sizes, act)
        ("1H relu 8-32-16",      [32],     "relu"),
        ("1H gelu 8-32-16",      [32],     "gelu"),
        ("1H silu 8-32-16",      [32],     "silu"),
        ("1H c19res 8-32-16",    [32],     "c19res"),
        ("2H relu 8-32-24-16",   [32, 24], "relu"),
        ("2H relu 8-48-32-16",   [48, 32], "relu"),
        ("2H gelu 8-32-24-16",   [32, 24], "gelu"),
        ("2H gelu 8-48-32-16",   [48, 32], "gelu"),
        ("2H silu 8-32-24-16",   [32, 24], "silu"),
        ("2H silu 8-48-32-16",   [48, 32], "silu"),
        ("2H c19res 8-32-24-16", [32, 24], "c19res"),
        ("2H c19res 8-48-32-16", [48, 32], "c19res"),
    ]

    results = []
    for label, hs, act in configs:
        torch.manual_seed(SEED)
        unit = DeepUnit(hs, act)
        print(f">>> {label} ({unit.param_count()} params)...")
        r = train_and_eval(label, unit, cur, nxt, corpus)
        ll_str = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        print(f"    lossless={ll_str}  downstream={r['eval_acc']:.2f}%  [{r['time']:.1f}s]{r['alpha']}")
        results.append(r)

    results.sort(key=lambda r: -r["eval_acc"])

    print(f"\n{'='*85}")
    print(f"{'CONFIG':<26} {'params':>7} {'bytes':>7} {'lossless':>10} {'downstream':>12} {'rank':>6}")
    print(f"{'='*85}")
    for i, r in enumerate(results):
        ll = "PASS" if r["byte_acc"] == 100.0 else f"{r['byte_acc']:.1f}%"
        star = " ***" if r["byte_acc"] >= 99.5 and r["eval_acc"] >= 43 else ""
        print(f"{r['label']:<26} {r['params']:>7d} {r['params']:>6d}B {ll:>10} {r['eval_acc']:>11.2f}% {i+1:>6}{star}")

    # Best lossless
    ll_best = [r for r in results if r["byte_acc"] >= 99.5]
    if ll_best:
        w = max(ll_best, key=lambda r: r["eval_acc"])
        print(f"\n>>> BEST >=99.5% LOSSLESS: {w['label']} — downstream={w['eval_acc']:.2f}%, {w['params']} params")

    # Best overall
    w = results[0]
    print(f">>> BEST DOWNSTREAM: {w['label']} — downstream={w['eval_acc']:.2f}%, lossless={w['byte_acc']:.2f}%, {w['params']} params")


if __name__ == "__main__":
    main()
