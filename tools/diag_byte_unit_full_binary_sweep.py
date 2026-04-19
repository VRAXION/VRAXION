"""Full-binary byte unit minimum H sweep.

Not just binary WEIGHTS — binary EVERYTHING:
  - W1, W2: binary {-1, +1}
  - hidden activation: sign function (binary output)
  - LUT output: binary (sign-snapped)
  - NO biases, NO alpha scales, NO C19, NO float anywhere in the forward path

Question: what is the smallest H that keeps the full-binary pipeline 256/256
lossless? If any H works, that's a ~10x size reduction over the current
binary-weights + int8-LUT champion.

Architecture:
  x in {-1, +1}^8
    -> W1 (8 x H, binary)
    -> sign()
    -> h in {-1, +1}^H
    -> W2 (H x 16, binary)
    -> sign()
    -> z in {-1, +1}^16   (this IS the LUT entry, binary)
  decode: z @ W2.T @ W1.T -> take sign to recover byte bits

Training: STE (straight-through estimator) for both sign() and binary
weight quantization. Adam on float shadow weights; forward uses binary.

Storage if successful at H=X:
  W1:  8 * X bits
  W2:  X * 16 bits
  LUT: 256 * 16 bits = 512 B (same regardless of X)
  total binary L0 = W1 + W2 + 512 B

Usage:
  python tools/diag_byte_unit_full_binary_sweep.py
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("output/byte_unit_full_binary_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> torch.Tensor:
    """256 x 8 bipolar bit vectors for each byte 0..255."""
    bytes_idx = torch.arange(256, dtype=torch.long, device=DEVICE)
    bits = torch.stack([(bytes_idx >> i) & 1 for i in range(8)], dim=1).float()
    return bits * 2.0 - 1.0  # {-1, +1}


class STEsign(torch.autograd.Function):
    """Straight-through sign: forward sign, backward identity."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.sign(x)
        out[out == 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # clip gradient where input is too large (optional; keeps training stable)
        mask = (x.abs() <= 1.0).float()
        return grad_out * mask


def stesign(x):
    return STEsign.apply(x)


class FullBinaryByteUnit(nn.Module):
    """Everything binary: W1, W2, activation, output."""
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        # Float "shadow" weights that get STE-binarized on the forward pass.
        self.W1_shadow = nn.Parameter(torch.empty(8, hidden, device=DEVICE))
        self.W2_shadow = nn.Parameter(torch.empty(hidden, 16, device=DEVICE))
        nn.init.uniform_(self.W1_shadow, -1.0, 1.0)
        nn.init.uniform_(self.W2_shadow, -1.0, 1.0)

    def W1_bin(self):
        return stesign(self.W1_shadow)

    def W2_bin(self):
        return stesign(self.W2_shadow)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x in {-1, +1}^8
        h_pre = x @ self.W1_bin()   # (B, H)
        h = stesign(h_pre)           # binary hidden
        z_pre = h @ self.W2_bin()    # (B, 16)
        z = stesign(z_pre)           # BINARY LUT entry
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # tied mirror decode, produces 8-dim float (take sign for byte)
        return (z @ self.W2_bin().t()) @ self.W1_bin().t()

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def measure_exact(model: FullBinaryByteUnit, x: torch.Tensor) -> tuple[int, int]:
    """Return (bytes_exact, unique_latents)."""
    model.eval()
    with torch.no_grad():
        z, x_hat = model(x)
    signs = torch.sign(x_hat)
    signs[signs == 0] = 1.0
    per_byte_match = (signs == x).all(dim=1)
    bytes_exact = int(per_byte_match.sum().item())
    # unique latents: count unique rows in z (binary)
    z_bits = ((z + 1) / 2).long()  # {0, 1}
    z_flat = torch.zeros(z.size(0), dtype=torch.long, device=DEVICE)
    for j in range(z.size(1)):
        z_flat = z_flat * 2 + z_bits[:, j]
    uniq = int(torch.unique(z_flat).numel())
    return bytes_exact, uniq


def train_one(hidden: int, epochs: int = 2000, lr: float = 5e-3, seed: int = 42,
              restart_n: int = 3) -> dict:
    """Train full-binary byte unit at given H, try a few random restarts."""
    x = build_dataset()
    best = {"bytes_exact": -1, "unique_latents": 0, "hidden": hidden}
    for r in range(restart_n):
        torch.manual_seed(seed + r * 13 + hidden * 7)
        model = FullBinaryByteUnit(hidden).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            z, x_hat = model(x)
            # L2 reconstruction loss; since x is bipolar and x_hat is signed,
            # this encourages x_hat to align with x.
            loss_recon = F.mse_loss(x_hat, x)
            # Mild "separation" term: penalize duplicate z vectors.
            # (Optional; helps training avoid the all-same-latent trap.)
            z_bits = ((z + 1) / 2)
            # pairwise bit-similarity — want low on average (not identical)
            z_flat = z_bits.reshape(256, -1)
            sep_penalty = (z_flat @ z_flat.t() / z_flat.size(1) - 0.5).pow(2).mean() * 0.1
            loss = loss_recon + sep_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()

        exact, uniq = measure_exact(model, x)
        if exact > best["bytes_exact"]:
            best = {
                "bytes_exact": exact,
                "unique_latents": uniq,
                "hidden": hidden,
                "restart": r,
                "final_loss": float(loss.item()),
            }
            if exact == 256:
                break  # early exit on perfect

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiddens", default="16,24,32,48,64,96,128,160,192,256")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=3)
    args = parser.parse_args()

    hiddens = [int(s) for s in args.hiddens.split(",")]
    print("=" * 70)
    print("FULL-BINARY BYTE UNIT MINIMUM-H SWEEP")
    print("  everything binary: W1, W2, activation, LUT. no biases, no scale.")
    print(f"  hiddens={hiddens}  epochs={args.epochs}  restarts={args.restarts}")
    print("=" * 70)

    results = []
    for h in hiddens:
        t0 = time.time()
        res = train_one(h, epochs=args.epochs, lr=args.lr, seed=args.seed,
                        restart_n=args.restarts)
        res["time_s"] = time.time() - t0
        results.append(res)
        mark = "✓" if res["bytes_exact"] == 256 else " "
        print(f"  {mark} H={h:>4d}  exact={res['bytes_exact']:>3d}/256  "
              f"uniq_latents={res['unique_latents']:>3d}/256  "
              f"({res['time_s']:.1f}s, restart={res.get('restart',0)})")
        if res["bytes_exact"] == 256:
            # Found a lossless H — can stop (but continue to check if smaller also works)
            pass

    # Find minimum H that was 100% exact
    lossless = [r for r in results if r["bytes_exact"] == 256]
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if lossless:
        min_h = min(r["hidden"] for r in lossless)
        print(f"  Minimum H for 100% full-binary lossless: {min_h}")
        print(f"  Storage at H={min_h}:")
        W1_bits = 8 * min_h
        W2_bits = min_h * 16
        LUT_bits = 256 * 16
        total = (W1_bits + W2_bits + LUT_bits) / 8
        print(f"    W1:  {W1_bits/8:.0f} B  ({W1_bits} bits)")
        print(f"    W2:  {W2_bits/8:.0f} B  ({W2_bits} bits)")
        print(f"    LUT: {LUT_bits/8:.0f} B  ({LUT_bits} bits)")
        print(f"    total: {total:.0f} B")
        print(f"  vs current binary-weights+int8-LUT champion: ~10,500 B")
        print(f"  reduction: {100*(1 - total/10500):.1f}%")
    else:
        best = max(results, key=lambda r: r["bytes_exact"])
        print(f"  NO H in the sweep reached 100% full-binary lossless.")
        print(f"  Best: H={best['hidden']}, {best['bytes_exact']}/256 exact.")
        print(f"  This means full-binary (with no biases, no scale, no C19) may need:")
        print(f"    - larger H (extend sweep)")
        print(f"    - longer training")
        print(f"    - or keeping SOME float components (biases / alpha / activation scale)")

    out_path = OUT_DIR / "sweep_summary.json"
    out_path.write_text(json.dumps({
        "sweep_hiddens": hiddens,
        "epochs": args.epochs,
        "lr": args.lr,
        "restarts": args.restarts,
        "results": results,
        "lossless_hiddens": [r["hidden"] for r in lossless],
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
