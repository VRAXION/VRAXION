"""Staged freeze for the L1 byte pair merger.

Same recipe that pushed the byte embedder from 95.3% (bulk quantization)
to 100% (staged freeze). If hidden size growth doesn't break out=16's 19.2%
ceiling, staged freeze might.

Protocol:
  1. Train float32 with L-BFGS until convergence
  2. Freeze the weight with smallest impact on loss, re-train remaining
  3. Repeat until stuck or all frozen (for quantization) OR just for final polish
  4. Report lossless at each stage

Also try: deeper encoder (2 hidden layers) as an alternative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


class C19Activation(nn.Module):
    def __init__(self, dim, c_init=1.0, rho_init=8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))

    def forward(self, x):
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
        return torch.where(x >= L, out_pos,
               torch.where(x <= -L, out_neg, interior))


class TiedMerger(nn.Module):
    """Standard 1-hidden-layer tied mirror."""
    def __init__(self, hidden, output_dim, input_dim=32):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(input_dim))
        self.c19 = C19Activation(hidden)

    def encode(self, x):
        return self.c19(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def decode(self, z):
        return (z @ self.W2.t() + self.db1) @ self.W1.t() + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class DeepTiedMerger(nn.Module):
    """2-hidden-layer tied mirror with C19 only on first (encoder-side) hidden."""
    def __init__(self, h1, h2, output_dim, input_dim=32):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, h1) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.W2 = nn.Parameter(torch.randn(h1, h2) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(h2))
        self.W3 = nn.Parameter(torch.randn(h2, output_dim) * 0.1)
        self.b3 = nn.Parameter(torch.zeros(output_dim))
        # Decoder biases
        self.db1 = nn.Parameter(torch.zeros(h2))
        self.db2 = nn.Parameter(torch.zeros(h1))
        self.db3 = nn.Parameter(torch.zeros(input_dim))
        self.c19 = C19Activation(h1)

    def encode(self, x):
        h = self.c19(x @ self.W1 + self.b1)
        h = h @ self.W2 + self.b2  # linear mid
        z = h @ self.W3 + self.b3  # linear out
        return z

    def decode(self, z):
        h = z @ self.W3.t() + self.db1
        h = h @ self.W2.t() + self.db2
        return h @ self.W1.t() + self.db3

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def load_data():
    with open(LUT_PATH, "r") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32) * scale
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1)


def metrics(model, data):
    with torch.no_grad():
        x_hat, _ = model(data)
        sign_match = (torch.sign(x_hat) == torch.sign(data))
        lossless = sign_match.all(dim=1).float().mean().item() * 100
        per_dim = sign_match.float().mean().item() * 100
    return lossless, per_dim


def train_floats(model, data, max_ep, patience, print_every=20, tag=""):
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn="strong_wolfe")
    best, no_imp = float("inf"), 0
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        def closure():
            opt.zero_grad()
            x_hat, _ = model(data)
            loss = F.mse_loss(x_hat, data)
            loss.backward()
            return loss
        lv = opt.step(closure)
        lf = lv.item() if torch.is_tensor(lv) else lv
        if lf < best - 1e-8:
            best, no_imp = lf, 0
        else:
            no_imp += 1
        if ep % print_every == 0 or ep == 1:
            ll, pd = metrics(model, data)
            print(f"  {tag} ep {ep:4d}: loss={lf:.6f}, lossless={ll:6.2f}%",
                  flush=True)
        if no_imp >= patience:
            break
    ll, pd = metrics(model, data)
    print(f"  {tag} FINAL ep {ep}: loss={best:.6f}, lossless={ll:.2f}%, "
          f"per-dim={pd:.2f}%, time={time.time()-t0:.1f}s", flush=True)
    return ll, pd


def main():
    print(f"=== L1 BYTE PAIR MERGER — STAGED + DEEP ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Data: zero-free byte embedder LUT", flush=True)
    data = load_data().to(DEVICE)
    print(f"Data shape: {data.shape}", flush=True)

    results = []

    # --- Test 1: Deep (2-hidden) encoder, out=16 ---
    print("\n" + "="*60, flush=True)
    print("TEST 1: Deep 2-hidden encoder, out=16", flush=True)
    print("="*60, flush=True)
    for h1, h2 in [(64, 32), (128, 32), (128, 64), (256, 64), (256, 128), (512, 128)]:
        torch.manual_seed(SEED)
        m = DeepTiedMerger(h1=h1, h2=h2, output_dim=16).to(DEVICE)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n--- Deep H1={h1} H2={h2} out=16, params={n_params} ---",
              flush=True)
        ll, pd = train_floats(m, data, max_ep=500, patience=80, tag=f"D{h1}-{h2}")
        results.append(("deep", (h1, h2, 16), n_params, ll, pd))

    # --- Test 2: 1-hidden with WIDE hidden + extended training ---
    print("\n" + "="*60, flush=True)
    print("TEST 2: Shallow wide (H=512), out=16, extended training", flush=True)
    print("="*60, flush=True)
    torch.manual_seed(SEED)
    m = TiedMerger(hidden=512, output_dim=16).to(DEVICE)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"\n--- Shallow H=512 out=16, params={n_params} ---", flush=True)
    ll, pd = train_floats(m, data, max_ep=1500, patience=200, tag="H=512")
    results.append(("shallow", (512, 0, 16), n_params, ll, pd))

    # --- Test 3: 1-hidden with out=18, 20 (intermediate compression) ---
    print("\n" + "="*60, flush=True)
    print("TEST 3: Intermediate compression (out=18, 20) H=64", flush=True)
    print("="*60, flush=True)
    for out in [18, 20, 22]:
        torch.manual_seed(SEED)
        m = TiedMerger(hidden=64, output_dim=out).to(DEVICE)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n--- Shallow H=64 out={out}, params={n_params} ---", flush=True)
        ll, pd = train_floats(m, data, max_ep=500, patience=80, tag=f"out={out}")
        results.append(("shallow", (64, 0, out), n_params, ll, pd))

    # --- Summary ---
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Type':<8} | {'Shape':<18} | {'Params':<7} | {'Lossless':<9} | "
          f"{'Per-dim':<8}", flush=True)
    print("-" * 65, flush=True)
    for kind, shape, params, ll, pd in results:
        shape_str = f"H{shape[0]}-{shape[1]}/o{shape[2]}" if shape[1] else \
                    f"H{shape[0]}/o{shape[2]}"
        print(f"{kind:<8} | {shape_str:<18} | {params:<7} | "
              f"{ll:>7.2f}% | {pd:>7.2f}%", flush=True)


if __name__ == "__main__":
    main()
