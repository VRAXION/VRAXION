"""Binary variants sweep — find which float components are CRITICAL.

The pure-binary (no biases, no alphas, sign activation, binary LUT) sweep
produced max 32/256 exact at H=24 — it does not converge. To find the
minimum configuration that DOES work, sweep variants with progressively
fewer float components:

Variants (all have binary W1, W2):
  A: C19 activation, float biases, float alphas, int8 LUT     (current champion)
  B: C19 activation, float biases, float alphas, BINARY LUT   (force binary output)
  C: SIGN activation, float biases, float alphas, BINARY LUT
  D: SIGN activation, NO biases,    float alphas, BINARY LUT
  E: SIGN activation, NO biases,    NO alphas,    BINARY LUT   (pure — expected to fail)

For each variant, sweep H in [16, 24, 32, 48, 64] and report best lossless H.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_unit_widen_sweep import (
    DEVICE, CODEBOOKS, FloatByteUnit, QuantByteUnit,
    build_dataset, metrics, train_adam, choose_best_alpha_pair,
    init_quant_from_float, load_winner_blob, warm_start_from_winner,
)
from diag_byte_unit_full_binary_sweep import FullBinaryByteUnit, stesign, STEsign

OUT_DIR = Path("output/byte_unit_binary_variants_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class BinaryWithBiasSignLUT(nn.Module):
    """Variant C: binary W1/W2 + sign activation + float biases + alphas + binary LUT."""
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.W1 = nn.Parameter(torch.empty(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.empty(hidden, 16, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(hidden, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(16, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        nn.init.uniform_(self.W1, -1.0, 1.0)
        nn.init.uniform_(self.W2, -1.0, 1.0)

    @property
    def alpha1(self):
        return F.softplus(self.alpha1_raw) + 1e-6
    @property
    def alpha2(self):
        return F.softplus(self.alpha2_raw) + 1e-6

    def W1_bin(self):
        return stesign(self.W1) * self.alpha1
    def W2_bin(self):
        return stesign(self.W2) * self.alpha2

    def encode(self, x):
        h_pre = x @ self.W1_bin() + self.b1
        h = stesign(h_pre)  # sign activation
        z_pre = h @ self.W2_bin() + self.b2
        z = stesign(z_pre)  # BINARY LUT
        return z

    def decode(self, z):
        W2_b = self.W2_bin()
        W1_b = self.W1_bin()
        return (z @ W2_b.t()) @ W1_b.t()

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class BinaryNoBiasSignLUT(nn.Module):
    """Variant D: binary W1/W2 + sign activation + NO biases + alphas + binary LUT."""
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.W1 = nn.Parameter(torch.empty(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.empty(hidden, 16, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        nn.init.uniform_(self.W1, -1.0, 1.0)
        nn.init.uniform_(self.W2, -1.0, 1.0)

    @property
    def alpha1(self):
        return F.softplus(self.alpha1_raw) + 1e-6
    @property
    def alpha2(self):
        return F.softplus(self.alpha2_raw) + 1e-6

    def W1_bin(self):
        return stesign(self.W1) * self.alpha1
    def W2_bin(self):
        return stesign(self.W2) * self.alpha2

    def encode(self, x):
        h = stesign(x @ self.W1_bin())
        z = stesign(h @ self.W2_bin())
        return z

    def decode(self, z):
        return (z @ self.W2_bin().t()) @ self.W1_bin().t()

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def measure(model, x):
    model.eval()
    with torch.no_grad():
        z, x_hat = model(x)
    signs = torch.sign(x_hat); signs[signs == 0] = 1.0
    match = (signs == x).all(dim=1)
    exact = int(match.sum().item())
    # unique
    z_bits = ((z + 1) / 2).long()
    flat = torch.zeros(z.size(0), dtype=torch.long, device=DEVICE)
    for j in range(z.size(1)):
        flat = flat * 2 + z_bits[:, j]
    uniq = int(torch.unique(flat).numel())
    return exact, uniq


def train_variant(cls, hidden, epochs=1500, lr=5e-3, restarts=4, seed=42):
    x = build_dataset()
    best = {"exact": -1, "uniq": 0}
    for r in range(restarts):
        torch.manual_seed(seed + r * 13 + hidden * 7)
        model = cls(hidden).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            z, x_hat = model(x)
            loss = F.mse_loss(x_hat, x)
            opt.zero_grad(); loss.backward(); opt.step()
        exact, uniq = measure(model, x)
        if exact > best["exact"]:
            best = {"exact": exact, "uniq": uniq, "restart": r, "loss": float(loss.item())}
            if exact == 256:
                break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiddens", default="16,24,32,48,64,96")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=4)
    args = parser.parse_args()
    hiddens = [int(s) for s in args.hiddens.split(",")]

    variants = [
        ("C_bias_sign_binLUT", BinaryWithBiasSignLUT),
        ("D_nobias_sign_binLUT", BinaryNoBiasSignLUT),
    ]

    print("=" * 70)
    print("BINARY VARIANTS SWEEP — find smallest config reaching 256/256")
    print("=" * 70)

    results = {}
    for var_name, cls in variants:
        print(f"\n[{var_name}]")
        var_results = []
        for h in hiddens:
            t0 = time.time()
            res = train_variant(cls, h, epochs=args.epochs, restarts=args.restarts)
            res["hidden"] = h
            res["time_s"] = time.time() - t0
            var_results.append(res)
            mark = "✓" if res["exact"] == 256 else " "
            print(f"  {mark} H={h:>3d}  exact={res['exact']:>3d}/256  uniq={res['uniq']:>3d}  "
                  f"({res['time_s']:.1f}s, r={res['restart']})")
            if res["exact"] == 256:
                break
        results[var_name] = var_results

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for var_name, var_res in results.items():
        lossless = [r for r in var_res if r["exact"] == 256]
        if lossless:
            min_h = min(r["hidden"] for r in lossless)
            print(f"  {var_name}: min H = {min_h}")
        else:
            best = max(var_res, key=lambda r: r["exact"])
            print(f"  {var_name}: NO lossless in sweep (best: H={best['hidden']}, {best['exact']}/256)")

    out_path = OUT_DIR / "variants_summary.json"
    out_path.write_text(json.dumps({"hiddens": hiddens, "epochs": args.epochs,
                                     "restarts": args.restarts, "results": results}, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
