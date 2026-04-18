"""L1 Byte Pair Merger v2: targeted experiments to push toward 100% lossless.

Phase 1 showed that:
  - Hidden size is irrelevant (H=48 == H=128)
  - Output dim is the only variable that matters
  - Even out=32 (no compression) only reaches 73% lossless
  - The bottleneck is the tied-weight mirror architecture itself

V2 tests:
  A) Heavy recon weight (rw=5, rw=10) — worked for byte embedder
  B) Sign-based loss (directly optimizing sign match, not MSE)
  C) 2-phase training: phase1 = MSE, phase2 = freeze encoder, retrain decoder
  D) Larger models: out=32 with H=128, H=256
  E) No-mirror baseline: untied encoder+decoder (upper bound)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_EPOCHS = 500
PATIENCE = 50
PRINT_EVERY = 10

LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8.json")


# ---------------------------------------------------------------------------
# C19 activation (vectorized, learnable per-neuron c and rho)
# ---------------------------------------------------------------------------

class C19Activation(nn.Module):
    def __init__(self, dim: int, c_init: float = 1.0, rho_init: float = 8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        out_pos = x - L
        out_neg = x + L
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0,
                          torch.ones_like(n),
                          -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        out = torch.where(x >= L, out_pos,
              torch.where(x <= -L, out_neg, interior))
        return out


# ---------------------------------------------------------------------------
# Model A: Tied-weight mirror (same as v1 but with sign loss option)
# ---------------------------------------------------------------------------

class BytePairMergerTied(nn.Module):
    def __init__(self, hidden: int, output_dim: int, input_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(input_dim))
        self.c19 = C19Activation(hidden, c_init=1.0, rho_init=8.0)

    def encode(self, x):
        h = self.c19(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z):
        h = z @ self.W2.t() + self.db1
        return h @ self.W1.t() + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model B: UNTIED encoder/decoder (upper bound test)
# ---------------------------------------------------------------------------

class BytePairMergerUntied(nn.Module):
    def __init__(self, hidden: int, output_dim: int, input_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        # Encoder
        self.eW1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.eb1 = nn.Parameter(torch.zeros(hidden))
        self.eW2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.eb2 = nn.Parameter(torch.zeros(output_dim))
        self.c19_enc = C19Activation(hidden, c_init=1.0, rho_init=8.0)
        # Decoder (separate weights)
        self.dW1 = nn.Parameter(torch.randn(output_dim, hidden) * 0.1)
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.dW2 = nn.Parameter(torch.randn(hidden, input_dim) * 0.1)
        self.db2 = nn.Parameter(torch.zeros(input_dim))
        self.c19_dec = C19Activation(hidden, c_init=1.0, rho_init=8.0)

    def encode(self, x):
        h = self.c19_enc(x @ self.eW1 + self.eb1)
        return h @ self.eW2 + self.eb2

    def decode(self, z):
        h = self.c19_dec(z @ self.dW1 + self.db1)
        return h @ self.dW2 + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> torch.Tensor:
    with open(LUT_PATH, "r") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut_int8 = blob["lut"]
    lut = torch.tensor(lut_int8, dtype=torch.float32) * scale
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    emb_a = lut[idx_a]
    emb_b = lut[idx_b]
    return torch.cat([emb_a, emb_b], dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model, data):
    with torch.no_grad():
        x_hat, z = model(data)
        sign_match = (torch.sign(x_hat) == torch.sign(data))
        lossless = sign_match.all(dim=1).float().mean().item() * 100
        per_dim = sign_match.float().mean().item() * 100
        # Also compute margin: how far are wrong signs from the boundary?
        wrong = ~sign_match
        if wrong.any():
            margin = (x_hat * data)[wrong].mean().item()  # negative = wrong sign
        else:
            margin = 0.0
    return lossless, per_dim, margin


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, data, loss_fn, name, max_ep=MAX_EPOCHS, patience=PATIENCE):
    model = model.to(DEVICE)
    data_dev = data.to(DEVICE)
    n_params = model.param_count()
    print(f"\n--- {name}: params={n_params} ---", flush=True)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    best_loss = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, max_ep + 1):
        def closure():
            optimizer.zero_grad()
            x_hat, z = model(data_dev)
            loss = loss_fn(x_hat, data_dev, z)
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        loss_f = loss_val.item() if torch.is_tensor(loss_val) else loss_val

        if loss_f < best_loss - 1e-7:
            best_loss = loss_f
            no_improve = 0
        else:
            no_improve += 1

        if epoch % PRINT_EVERY == 0 or epoch == 1 or no_improve >= patience:
            ll, pd, mg = compute_metrics(model, data_dev)
            print(f"  epoch {epoch:4d}: loss={loss_f:.6f}, "
                  f"lossless={ll:6.2f}%, per-dim={pd:6.2f}%", flush=True)

        if no_improve >= patience:
            print(f"  -> converged at epoch {epoch}", flush=True)
            break

    elapsed = time.time() - t0
    ll, pd, mg = compute_metrics(model, data_dev)
    print(f"  FINAL: lossless={ll:.2f}%, per-dim={pd:.2f}%, "
          f"margin={mg:.4f}, time={elapsed:.1f}s", flush=True)
    return {"name": name, "lossless": ll, "per_dim": pd, "params": n_params,
            "time": elapsed}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(x_hat, x, z):
    return F.mse_loss(x_hat, x)

def sign_aware_loss(x_hat, x, z):
    """MSE + penalty for wrong signs."""
    mse = F.mse_loss(x_hat, x)
    # sign penalty: want x_hat * x > 0 for all dims
    sign_prod = x_hat * x  # positive = correct sign
    penalty = F.relu(-sign_prod).mean()  # penalize negative products
    return mse + 5.0 * penalty

def heavy_recon_loss(x_hat, x, z):
    """MSE with heavy weight to push sign correctness."""
    mse = F.mse_loss(x_hat, x)
    sign_prod = x_hat * x
    penalty = F.relu(-sign_prod).mean()
    return mse + 10.0 * penalty

def margin_loss(x_hat, x, z):
    """Maximize margin on correct sign predictions."""
    sign_prod = x_hat * x  # want this positive and large
    # Hinge-like: max(0, 0.5 - sign_prod)
    hinge = F.relu(0.5 - sign_prod).mean()
    mse = F.mse_loss(x_hat, x)
    return hinge + 0.1 * mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== L1 BYTE PAIR MERGER V2 ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Goal: push tied-weight mirror toward 100% lossless\n", flush=True)

    data = load_data()
    print(f"Data: {data.shape}, range [{data.min():.4f}, {data.max():.4f}]",
          flush=True)

    results = []

    # ---- Experiment 1: Sign-aware loss on tied H=64 out=32 ----
    print("\n" + "="*60, flush=True)
    print("EXP 1: Sign-aware loss functions (tied, H=64, out=32)", flush=True)
    print("="*60, flush=True)

    torch.manual_seed(SEED)
    m = BytePairMergerTied(hidden=64, output_dim=32)
    r = train(m, data, mse_loss, "1a: MSE baseline (tied H=64 out=32)")
    results.append(r)

    torch.manual_seed(SEED)
    m = BytePairMergerTied(hidden=64, output_dim=32)
    r = train(m, data, sign_aware_loss, "1b: sign_aware rw=5 (tied H=64 out=32)")
    results.append(r)

    torch.manual_seed(SEED)
    m = BytePairMergerTied(hidden=64, output_dim=32)
    r = train(m, data, heavy_recon_loss, "1c: heavy_recon rw=10 (tied H=64 out=32)")
    results.append(r)

    torch.manual_seed(SEED)
    m = BytePairMergerTied(hidden=64, output_dim=32)
    r = train(m, data, margin_loss, "1d: margin_loss (tied H=64 out=32)")
    results.append(r)

    # ---- Experiment 2: Larger hidden (tied, out=32) ----
    print("\n" + "="*60, flush=True)
    print("EXP 2: Larger hidden (tied, out=32, sign_aware loss)", flush=True)
    print("="*60, flush=True)

    for H in [128, 256]:
        torch.manual_seed(SEED)
        m = BytePairMergerTied(hidden=H, output_dim=32)
        r = train(m, data, sign_aware_loss, f"2: tied H={H} out=32 sign_aware")
        results.append(r)

    # ---- Experiment 3: UNTIED (upper bound) ----
    print("\n" + "="*60, flush=True)
    print("EXP 3: UNTIED encoder/decoder (upper bound)", flush=True)
    print("="*60, flush=True)

    for H, out in [(64, 16), (64, 24), (64, 32), (128, 32)]:
        torch.manual_seed(SEED)
        m = BytePairMergerUntied(hidden=H, output_dim=out)
        r = train(m, data, sign_aware_loss,
                  f"3: untied H={H} out={out} sign_aware")
        results.append(r)

    # ---- Experiment 4: Best untied with out=16 (actual 2:1 compression) ----
    print("\n" + "="*60, flush=True)
    print("EXP 4: Untied H=128/256, out=16 (real 2:1 compression)", flush=True)
    print("="*60, flush=True)

    for H in [128, 256]:
        torch.manual_seed(SEED)
        m = BytePairMergerUntied(hidden=H, output_dim=16)
        r = train(m, data, sign_aware_loss,
                  f"4: untied H={H} out=16 sign_aware")
        results.append(r)

    # ---- Summary ----
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Name':<45} | {'Lossless':>8} | {'Per-dim':>8} | {'Params':>7}",
          flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['name']:<45} | {r['lossless']:>7.2f}% | "
              f"{r['per_dim']:>7.2f}% | {r['params']:>7}", flush=True)

    best = max(results, key=lambda r: r["lossless"])
    print(f"\nBEST: {best['name']} -> {best['lossless']:.2f}% lossless",
          flush=True)


if __name__ == "__main__":
    main()
