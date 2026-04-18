"""Level 1 Byte Pair Merger sweep: find the best tied-weight mirror autoencoder
that compresses 2 consecutive byte embeddings (2 x 16D = 32D) into a smaller latent.

Architecture per config:
  encoder: 32D -> hidden (C19 activation, learnable c/rho per neuron) -> output_dim
  decoder: output_dim -> hidden (linear, tied W_enc^T) -> 32D reconstructed
  Asymmetric: C19 on encoder side only, linear decode (byte embedder winner pattern).

Optimizer: L-BFGS (strong_wolfe) full-batch on all 65,536 byte pairs.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_EPOCHS = 1000
PATIENCE = 100
PRINT_EVERY = 10

LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")

CONFIGS = [
    # Phase 4: push out=16 (2:1 compression) with larger hidden capacity
    {"name": "P16-128", "hidden": 128, "output": 16},
    {"name": "P16-256", "hidden": 256, "output": 16},
    {"name": "P16-384", "hidden": 384, "output": 16},
    {"name": "P16-512", "hidden": 512, "output": 16},
    # Also try longer training (more epochs) at H=128
    {"name": "P20-128", "hidden": 128, "output": 20},
    {"name": "P24-128", "hidden": 128, "output": 24},
]

# ---------------------------------------------------------------------------
# C19 activation  (vectorized, learnable per-neuron c and rho)
# ---------------------------------------------------------------------------

class C19Activation(nn.Module):
    """Piecewise oscillating activation with learnable c and rho per neuron."""

    def __init__(self, dim: int, c_init: float = 1.0, rho_init: float = 8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c  # (dim,)

        # Linear tails outside [-L, L]
        out_pos = x - L  # x >= L
        out_neg = x + L  # x <= -L

        # Interior: oscillating bumps
        scaled = x / c  # broadcast: (batch, dim) / (dim,)
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0,
                          torch.ones_like(n),
                          -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)

        # Compose via masks
        out = torch.where(x >= L, out_pos,
              torch.where(x <= -L, out_neg, interior))
        return out


# ---------------------------------------------------------------------------
# Tied-weight mirror autoencoder
# ---------------------------------------------------------------------------

class BytePairMerger(nn.Module):
    """
    Encoder:  32D --W1--> hidden (C19) --W2--> output_dim
    Decoder:  output_dim --W2^T--> hidden (linear) --W1^T--> 32D
    Tied weights, asymmetric activation (C19 encoder, linear decoder).
    """

    def __init__(self, hidden: int, output_dim: int, input_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

        # Encoder weights (decoder reuses transposes)
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dim))

        # Decoder biases (separate from encoder)
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(input_dim))

        # C19 activation on encoder hidden layer (learnable per-neuron)
        self.c19 = C19Activation(hidden, c_init=1.0, rho_init=8.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c19(x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z @ self.W2.t() + self.db1       # linear (no activation)
        out = h @ self.W1.t() + self.db2      # linear
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_byte_pair_data() -> torch.Tensor:
    """Load byte embedder LUT, build all 65536 byte-pair inputs (32D each)."""
    with open(LUT_PATH, "r") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut_int8 = blob["lut"]  # list of 256 lists, each 16 ints

    # Build float LUT: (256, 16)
    lut = torch.tensor(lut_int8, dtype=torch.float32) * scale

    # All 65536 pairs: byte_a in [0,255], byte_b in [0,255]
    # pair (a, b) -> concat(lut[a], lut[b]) = 32D
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)  # 0,0,...,0, 1,1,...,1, ...
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)  # 0,1,...,255, 0,1,...,255, ...
    emb_a = lut[idx_a]  # (65536, 16)
    emb_b = lut[idx_b]  # (65536, 16)
    pairs = torch.cat([emb_a, emb_b], dim=1)  # (65536, 32)
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_config(cfg: dict, data: torch.Tensor) -> dict:
    name = cfg["name"]
    hidden = cfg["hidden"]
    output_dim = cfg["output"]

    torch.manual_seed(SEED)
    model = BytePairMerger(hidden=hidden, output_dim=output_dim).to(DEVICE)
    data_dev = data.to(DEVICE)

    n_params = model.param_count()
    print(f"\n--- Config {name}: H={hidden}, out={output_dim}, params={n_params} ---",
          flush=True)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    best_loss = float("inf")
    no_improve = 0
    t0 = time.time()
    final_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # L-BFGS requires closure
        def closure():
            optimizer.zero_grad()
            x_hat, _z = model(data_dev)
            # MSE reconstruction loss on the 32D float embeddings
            loss = F.mse_loss(x_hat, data_dev)
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        loss_f = loss_val.item() if torch.is_tensor(loss_val) else loss_val

        # Early stopping
        if loss_f < best_loss - 1e-7:
            best_loss = loss_f
            no_improve = 0
        else:
            no_improve += 1

        # Metrics for progress
        if epoch % PRINT_EVERY == 0 or epoch == 1 or no_improve >= PATIENCE:
            with torch.no_grad():
                x_hat, _ = model(data_dev)
                # Lossless: sign of each dimension must match
                sign_match = (torch.sign(x_hat) == torch.sign(data_dev))
                lossless = sign_match.all(dim=1).float().mean().item() * 100
                per_dim = sign_match.float().mean().item() * 100
            print(f"  epoch {epoch:4d}: loss={loss_f:.6f}, "
                  f"lossless={lossless:6.2f}%, per-dim={per_dim:6.2f}%",
                  flush=True)

        final_epoch = epoch
        if no_improve >= PATIENCE:
            print(f"  -> converged at epoch {epoch} (patience {PATIENCE})", flush=True)
            break

    elapsed = time.time() - t0

    # Final evaluation
    with torch.no_grad():
        x_hat, z = model(data_dev)
        sign_match = (torch.sign(x_hat) == torch.sign(data_dev))
        lossless_count = sign_match.all(dim=1).sum().item()
        lossless_pct = lossless_count / 65536 * 100
        per_dim_pct = sign_match.float().mean().item() * 100

    lut_kb = 65536 * output_dim * 1 / 1024  # int8 = 1 byte each

    print(f"  FINAL: lossless={lossless_pct:.2f}% ({int(lossless_count)}/65536), "
          f"per-dim={per_dim_pct:.2f}%, time={elapsed:.1f}s", flush=True)
    print(f"  LUT size: {lut_kb:.0f} KB (65536 x {output_dim} x 1B)", flush=True)

    return {
        "name": name,
        "hidden": hidden,
        "output": output_dim,
        "params": n_params,
        "lossless_count": int(lossless_count),
        "lossless_pct": lossless_pct,
        "per_dim_pct": per_dim_pct,
        "time": elapsed,
        "lut_kb": lut_kb,
        "epochs": final_epoch,
        "final_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== L1 BYTE PAIR MERGER SWEEP ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Input: 2 x 16D byte embeddings (32D total)", flush=True)
    print(f"Total byte pairs: 65,536", flush=True)
    print(f"Optimizer: L-BFGS (strong_wolfe, full-batch)", flush=True)
    print(f"Activation: C19 (learnable c, rho per neuron) on encoder, linear decoder",
          flush=True)
    print(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}", flush=True)

    data = load_byte_pair_data()
    print(f"Data shape: {data.shape}, range: [{data.min().item():.4f}, {data.max().item():.4f}]",
          flush=True)

    results = []
    for cfg in CONFIGS:
        res = train_config(cfg, data)
        results.append(res)

    # Summary table
    print("\n=== SUMMARY ===", flush=True)
    header = (f"{'Config':<8}| {'Hidden':<7}| {'OutDim':<7}| {'Params':<7}| "
              f"{'Lossless':<10}| {'Per-dim':<9}| {'Time':<7}| {'LUT KB':<7}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    best_idx = 0
    best_lossless = -1.0
    for i, r in enumerate(results):
        print(f"{r['name']:<8}| {r['hidden']:<7}| {r['output']:<7}| {r['params']:<7}| "
              f"{r['lossless_pct']:>7.2f}% | {r['per_dim_pct']:>7.2f}% | "
              f"{r['time']:>5.1f}s | {r['lut_kb']:>5.0f}",
              flush=True)
        # Best = highest lossless; tie-break by fewer params
        if (r["lossless_pct"] > best_lossless or
            (r["lossless_pct"] == best_lossless and
             r["params"] < results[best_idx]["params"])):
            best_lossless = r["lossless_pct"]
            best_idx = i

    w = results[best_idx]
    print(f"\nWINNER: Config {w['name']} "
          f"(H={w['hidden']}, out={w['output']}, "
          f"lossless={w['lossless_pct']:.2f}%, "
          f"params={w['params']}, "
          f"LUT={w['lut_kb']:.0f}KB)",
          flush=True)


if __name__ == "__main__":
    main()
