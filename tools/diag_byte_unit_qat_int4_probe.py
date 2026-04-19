"""Probe the shipped L0 byte unit under STE-style int4 quantization.

Purpose:
  - verify apples-to-apples against the ternary QAT probe
  - confirm that the shipped int4 winner is already exact under the same
    tied-mirror decode formula and quantized forward path
  - optionally run a few optimization steps to see whether the exact point is
    stable under the current objective

The byte unit is:
  8 signed bits -> 24 C19 hidden -> 16D latent
  decode: (z @ W2.T) @ W1.T
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINNER_PATH = Path(__file__).with_name("byte_unit_winner_int4.json")


class C19Activation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_raw = nn.Parameter(torch.ones(dim, device=DEVICE))
        self.rho_raw = nn.Parameter(torch.ones(dim, device=DEVICE) * 8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        return torch.where(x >= L, x - L, torch.where(x <= -L, x + L, interior))


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return STERound.apply(x)


def byte_to_signed_bits(byte: int) -> list[float]:
    return [1.0 if ((byte >> i) & 1) else -1.0 for i in range(8)]


def build_dataset() -> torch.Tensor:
    x = np.array([byte_to_signed_bits(i) for i in range(256)], dtype=np.float32)
    return torch.tensor(x, device=DEVICE)


class QATByteUnitInt4(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(8, 24, device=DEVICE))
        self.W2 = nn.Parameter(torch.zeros(24, 16, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(24, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(16, device=DEVICE))
        self.c19 = C19Activation(24)

    @property
    def alpha1(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha1_raw) + 1e-6

    @property
    def alpha2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha2_raw) + 1e-6

    def quant_W1(self) -> torch.Tensor:
        a = self.alpha1
        code = ste_round(self.W1 / a).clamp(-8.0, 7.0)
        return a * code

    def quant_W2(self) -> torch.Tensor:
        a = self.alpha2
        code = ste_round(self.W2 / a).clamp(-8.0, 7.0)
        return a * code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        W1_q = self.quant_W1()
        W2_q = self.quant_W2()
        h = self.c19(x @ W1_q + self.b1)
        return h @ W2_q + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        W1_q = self.quant_W1()
        W2_q = self.quant_W2()
        return (z @ W2_q.t()) @ W1_q.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def load_int4_winner(model: QATByteUnitInt4) -> dict:
    blob = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    W1 = np.array(blob["W1_int4"], dtype=np.float32) * float(blob["scale_W1"])
    W2 = np.array(blob["W2_int4"], dtype=np.float32) * float(blob["scale_W2"])
    with torch.no_grad():
        model.W1.copy_(torch.tensor(W1, device=DEVICE))
        model.W2.copy_(torch.tensor(W2, device=DEVICE))
        model.b1.copy_(torch.tensor(blob["bias1"], dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(blob["bias2"], dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(blob["c19_c"], dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(blob["c19_rho"], dtype=torch.float32, device=DEVICE))
        model.alpha1_raw.copy_(
            torch.log(torch.expm1(torch.tensor(float(blob["scale_W1"]), device=DEVICE)))
        )
        model.alpha2_raw.copy_(
            torch.log(torch.expm1(torch.tensor(float(blob["scale_W2"]), device=DEVICE)))
        )
    return blob


@torch.no_grad()
def metrics(model: QATByteUnitInt4, x: torch.Tensor) -> dict:
    z, x_hat = model(x)
    ok_bits = torch.sign(x_hat).eq(torch.sign(x))
    byte_exact = ok_bits.all(dim=1)
    q1 = torch.clamp(torch.round(model.quant_W1() / model.alpha1.clamp(min=1e-6)), -8, 7)
    q2 = torch.clamp(torch.round(model.quant_W2() / model.alpha2.clamp(min=1e-6)), -8, 7)
    return {
        "lossless": float(byte_exact.float().mean().item() * 100.0),
        "bad_bytes": int((~byte_exact).sum().item()),
        "per_bit": float(ok_bits.float().mean().item() * 100.0),
        "alpha1": float(model.alpha1.item()),
        "alpha2": float(model.alpha2.item()),
        "w1_int_min": int(q1.min().item()),
        "w1_int_max": int(q1.max().item()),
        "w2_int_min": int(q2.min().item()),
        "w2_int_max": int(q2.max().item()),
    }


def objective(model: QATByteUnitInt4, x: torch.Tensor) -> torch.Tensor:
    _, x_hat = model(x)
    mse = ((x_hat - x) ** 2).mean()
    sign_hinge = torch.relu(-x_hat * torch.sign(x)).mean()
    return mse + 2.0 * sign_hinge


def train_adam(model: QATByteUnitInt4, x: torch.Tensor, epochs: int, lr: float) -> None:
    if epochs <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        loss = objective(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            m = metrics(model, x)
            print(
                f"  [adam {ep+1:4d}] loss={loss.item():.6f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_bytes']:3d} "
                f"bit={m['per_bit']:.2f}% a1={m['alpha1']:.6f} a2={m['alpha2']:.6f}",
                flush=True,
            )
            if m["lossless"] >= 100.0:
                return


def train_lbfgs(model: QATByteUnitInt4, x: torch.Tensor, max_outer: int) -> None:
    if max_outer <= 0:
        return
    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=50,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    best = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stall = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            loss = objective(model, x)
            loss.backward()
            return loss
        loss = float(opt.step(closure).item())
        m = metrics(model, x)
        if m["lossless"] > best + 1e-6:
            best = m["lossless"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 5 == 0 or m["lossless"] >= 100.0:
            print(
                f"  [lbfgs {outer+1:3d}] loss={loss:.6f} ll={m['lossless']:.2f}% "
                f"bad={m['bad_bytes']:3d} bit={m['per_bit']:.2f}% stall={stall}",
                flush=True,
            )
        if m["lossless"] >= 100.0 or stall >= 8:
            break
    model.load_state_dict(best_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam-epochs", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--lbfgs-outer", type=int, default=0)
    parser.add_argument("--freeze-alpha", action="store_true")
    parser.add_argument("--out", default="output/byte_unit_qat_int4_probe")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BYTE UNIT INT4 PROBE (STE) — 8 -> 24 -> 16")
    print("=" * 70)

    x = build_dataset()
    model = QATByteUnitInt4().to(DEVICE)
    blob = load_int4_winner(model)
    if args.freeze_alpha:
        model.alpha1_raw.requires_grad_(False)
        model.alpha2_raw.requires_grad_(False)

    print("\n[1] Warm-start exact check from shipped int4 winner")
    m0 = metrics(model, x)
    for k in ("lossless", "bad_bytes", "per_bit", "alpha1", "alpha2", "w1_int_min", "w1_int_max", "w2_int_min", "w2_int_max"):
        print(f"  {k:>12}: {m0[k]}")

    t0 = time.time()
    print("\n[2] Optional polish")
    train_adam(model, x, args.adam_epochs, args.adam_lr)
    train_lbfgs(model, x, args.lbfgs_outer)
    final = metrics(model, x)
    elapsed = time.time() - t0

    print("\n[3] Final metrics")
    for k in ("lossless", "bad_bytes", "per_bit", "alpha1", "alpha2", "w1_int_min", "w1_int_max", "w2_int_min", "w2_int_max"):
        print(f"  {k:>12}: {final[k]}")
    print(f"  {'time_s':>12}: {elapsed:.2f}")

    payload = {
        "architecture": blob["architecture"],
        "precision": "int4 weights + STE probe",
        "scale_W1": final["alpha1"],
        "scale_W2": final["alpha2"],
        "metrics": {**final, "time_s": elapsed},
    }
    save_path = out_dir / "byte_unit_qat_int4_probe.json"
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
