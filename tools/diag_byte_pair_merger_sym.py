"""L1 Byte Pair Merger — SYMMETRIC C19 mirror.

Previous experiments used asymmetric: C19 on encoder, LINEAR on decoder
(byte embedder winner pattern). But byte embedder had 8-bit input, merger has
32D float input — much richer, may need C19 on decoder side too.

This test: TIED weights BUT C19 applied on BOTH encoder and decoder hidden.
Never tested this exact combo before.
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
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0,
                          torch.ones_like(n), -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        return torch.where(x >= L, x - L,
               torch.where(x <= -L, x + L, interior))


class SymMergerTied(nn.Module):
    """TIED weights with C19 on BOTH encoder and decoder hidden layers.

    Forward:  x -> W1 -> b1 -> C19_enc -> W2 -> b2 -> z
    Backward: z -> W2.t() -> db1 -> C19_dec -> W1.t() -> db2 -> x_hat
    W1, W2 shared; C19 has separate learnable c/rho per side.
    """
    def __init__(self, hidden, output_dim, input_dim=32,
                 shared_c19=False):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, output_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(input_dim))
        self.c19_enc = C19Activation(hidden)
        if shared_c19:
            self.c19_dec = self.c19_enc
        else:
            self.c19_dec = C19Activation(hidden)

    def encode(self, x):
        return self.c19_enc(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def decode(self, z):
        return self.c19_dec(z @ self.W2.t() + self.db1) @ self.W1.t() + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class AsymMergerTied(nn.Module):
    """Reference: tied mirror with LINEAR decoder (previous winner pattern)."""
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


def train(model, data, max_ep=500, patience=80, print_every=20, tag=""):
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
            print(f"  {tag} ep {ep:4d}: loss={lf:.6f}, "
                  f"lossless={ll:6.2f}%", flush=True)
        if no_imp >= patience:
            break
    ll, pd = metrics(model, data)
    print(f"  {tag} FINAL ep {ep}: loss={best:.6f}, lossless={ll:.2f}%, "
          f"per-dim={pd:.2f}%, time={time.time()-t0:.1f}s", flush=True)
    return ll, pd


def main():
    print(f"=== SYMMETRIC C19 MIRROR TEST ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    data = load_data().to(DEVICE)
    print(f"Data: {data.shape}\n", flush=True)

    results = []

    # --- Baseline: asymmetric (linear decoder), H=64, out=16 ---
    torch.manual_seed(SEED)
    m = AsymMergerTied(hidden=64, output_dim=16).to(DEVICE)
    p = sum(x.numel() for x in m.parameters())
    print(f"\n--- REF: Asym tied (linear dec) H=64 out=16, params={p} ---",
          flush=True)
    ll, pd = train(m, data, tag="asym-64-16")
    results.append(("asym", 64, 16, p, ll))

    # --- Symmetric: C19 on both sides, separate c/rho ---
    for H, out in [(64, 16), (128, 16), (64, 20), (64, 24), (64, 32)]:
        torch.manual_seed(SEED)
        m = SymMergerTied(hidden=H, output_dim=out).to(DEVICE)
        p = sum(x.numel() for x in m.parameters())
        print(f"\n--- SYM tied (C19 both) H={H} out={out}, params={p} ---",
              flush=True)
        ll, pd = train(m, data, tag=f"sym-{H}-{out}")
        results.append(("sym", H, out, p, ll))

    # --- Symmetric with shared C19 (truly tied, same c/rho on both sides) ---
    for H, out in [(64, 16), (64, 32)]:
        torch.manual_seed(SEED)
        m = SymMergerTied(hidden=H, output_dim=out, shared_c19=True).to(DEVICE)
        p = sum(x.numel() for x in m.parameters())
        print(f"\n--- SYM-SHARED tied (shared C19) H={H} out={out}, "
              f"params={p} ---", flush=True)
        ll, pd = train(m, data, tag=f"sym-shared-{H}-{out}")
        results.append(("sym-shared", H, out, p, ll))

    # --- Summary ---
    print("\n" + "="*60, flush=True)
    print("SUMMARY — Symmetric C19 vs Asymmetric (linear dec)", flush=True)
    print("="*60, flush=True)
    print(f"{'Kind':<12} | {'H':<5} | {'Out':<5} | {'Params':<6} | "
          f"{'Lossless':<8}", flush=True)
    for kind, H, out, p, ll in results:
        print(f"{kind:<12} | {H:<5} | {out:<5} | {p:<6} | {ll:>7.2f}%",
              flush=True)


if __name__ == "__main__":
    main()
