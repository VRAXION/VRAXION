"""Latent-dimension sweep for the L0 byte unit.

Question:
  For the exact byte->latent->byte codec lane, how small can the latent
  dimension get before 256/256 round-trip breaks, and does a wider latent
  buy anything useful?

Protocol per config:
  1. Build float model: 8 -> H -> D tied-mirror byte unit.
  2. Float Adam warmup on all 256 bytes.
  3. Static alpha search for the chosen codebook.
  4. Fixed-alpha QAT Adam polish with STE codebook snapping.
  5. Full exactness check on all 256 bytes.

This is the A-block codec lane only. No downstream task is mixed in here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


THIS = Path(__file__).resolve()
BASE = THIS.with_name("diag_byte_unit_widen_sweep.py")
spec = importlib.util.spec_from_file_location("byte_widen", BASE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


DEVICE = mod.DEVICE


class FloatByteUnitLatent(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, activation: str):
        super().__init__()
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.W1 = nn.Parameter(torch.empty(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.empty(hidden, latent_dim, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(hidden, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(latent_dim, device=DEVICE))
        self.act = mod.make_activation(activation, hidden)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.activation_name in {"relu", "leaky_relu"}:
            nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        else:
            nn.init.xavier_uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W2.t()) @ self.W1.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class QuantByteUnitLatent(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, activation: str, codebook: tuple[float, ...]):
        super().__init__()
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.W1 = nn.Parameter(torch.zeros(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.zeros(hidden, latent_dim, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(hidden, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(latent_dim, device=DEVICE))
        self.act = mod.make_activation(activation, hidden)
        self.register_buffer("codebook", torch.tensor(codebook, dtype=torch.float32, device=DEVICE))

    @property
    def alpha1(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha1_raw) + 1e-6

    @property
    def alpha2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha2_raw) + 1e-6

    def quant_W1(self) -> torch.Tensor:
        return mod.ste_codebook(self.W1, self.alpha1 * self.codebook)

    def quant_W2(self) -> torch.Tensor:
        return mod.ste_codebook(self.W2, self.alpha2 * self.codebook)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x @ self.quant_W1() + self.b1)
        return h @ self.quant_W2() + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.quant_W2().t()) @ self.quant_W1().t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def choose_best_alpha_pair(float_model: FloatByteUnitLatent, x_np: np.ndarray, codebook: tuple[float, ...]) -> tuple[float, float, float]:
    W1 = float_model.W1.detach().cpu().numpy()
    W2 = float_model.W2.detach().cpu().numpy()
    b1 = float_model.b1.detach().cpu().numpy()
    b2 = float_model.b2.detach().cpu().numpy()
    levels = np.array(codebook, dtype=np.float32)
    if float_model.activation_name == "c19":
        act = float_model.act
        assert isinstance(act, mod.C19Activation)
        c = act.c_raw.detach().cpu().numpy()
        rho = act.rho_raw.detach().cpu().numpy()
    else:
        c = None
        rho = None
    base1 = float(np.abs(W1).mean() + 1e-8)
    base2 = float(np.abs(W2).mean() + 1e-8)
    best = None
    for m1 in (0.0625, 0.125, 0.167, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
        for m2 in (0.0625, 0.125, 0.167, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
            a1 = base1 * m1
            a2 = base2 * m2
            W1q = mod.nearest_codebook_np(W1, a1 * levels)
            W2q = mod.nearest_codebook_np(W2, a2 * levels)
            h = mod.activation_np(float_model.activation_name, x_np @ W1q + b1, c, rho)
            z = h @ W2q + b2
            y = (z @ W2q.T) @ W1q.T
            lossless = float(((np.sign(y) == np.sign(x_np)).all(axis=1)).mean() * 100.0)
            cand = (lossless, a1, a2)
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return float(best[0]), float(best[1]), float(best[2])


def init_quant_from_float(quant: QuantByteUnitLatent, float_model: FloatByteUnitLatent, a1: float, a2: float) -> None:
    with torch.no_grad():
        quant.W1.copy_(float_model.W1.detach())
        quant.W2.copy_(float_model.W2.detach())
        quant.b1.copy_(float_model.b1.detach())
        quant.b2.copy_(float_model.b2.detach())
        quant.alpha1_raw.copy_(torch.log(torch.expm1(torch.tensor(a1, device=DEVICE))))
        quant.alpha2_raw.copy_(torch.log(torch.expm1(torch.tensor(a2, device=DEVICE))))
        if quant.activation_name == "c19":
            assert isinstance(quant.act, mod.C19Activation) and isinstance(float_model.act, mod.C19Activation)
            quant.act.c_raw.copy_(float_model.act.c_raw.detach())
            quant.act.rho_raw.copy_(float_model.act.rho_raw.detach())


@dataclass
class SweepResult:
    hidden: int
    latent_dim: int
    activation: str
    codebook_name: str
    float_lossless: float
    float_bad: int
    static_lossless: float
    final_lossless: float
    final_bad: int
    final_per_bit: float
    unique_latents: int
    alpha1: float
    alpha2: float
    time_s: float


def run_one(hidden: int, latent_dim: int, activation: str, codebook_name: str, x: torch.Tensor, x_np: np.ndarray, args) -> SweepResult:
    t0 = time.time()
    print("\n" + "=" * 78)
    print(f"CONFIG hidden={hidden} latent={latent_dim} activation={activation} codebook={codebook_name}")
    print("=" * 78)

    seed_offset = sum(ord(ch) for ch in (activation + "|" + codebook_name))
    torch.manual_seed(args.seed + hidden * 17 + latent_dim * 31 + seed_offset)
    np.random.seed(args.seed + hidden * 17 + latent_dim * 31 + seed_offset)

    float_model = FloatByteUnitLatent(hidden, latent_dim, activation).to(DEVICE)
    m0 = mod.metrics(float_model, x)
    print(f"[float init] ll={m0['lossless']:.2f}% bad={m0['bad_bytes']} bit={m0['per_bit']:.2f}%")

    if m0["lossless"] < 100.0 and args.float_epochs > 0:
        print("[float warmup]")
        mf = mod.train_adam(
            float_model,
            x,
            epochs=args.float_epochs,
            lr=args.float_lr,
            print_every=max(50, args.float_epochs // 4),
            tag="float",
        )
    else:
        mf = m0
    print(f"[float final] ll={mf['lossless']:.2f}% bad={mf['bad_bytes']} bit={mf['per_bit']:.2f}% uniq={mf['unique_latents']}")

    codebook = mod.CODEBOOKS[codebook_name]
    static_ll, a1, a2 = choose_best_alpha_pair(float_model, x_np, codebook)
    print(f"[static snap] ll={static_ll:.2f}% a1={a1:.6f} a2={a2:.6f}")

    quant_model = QuantByteUnitLatent(hidden, latent_dim, activation, codebook).to(DEVICE)
    init_quant_from_float(quant_model, float_model, a1, a2)
    quant_model.alpha1_raw.requires_grad_(False)
    quant_model.alpha2_raw.requires_grad_(False)
    mq0 = mod.metrics(quant_model, x)
    print(f"[qat warm] ll={mq0['lossless']:.2f}% bad={mq0['bad_bytes']} bit={mq0['per_bit']:.2f}%")

    if args.qat_epochs > 0:
        print("[qat fixed-alpha Adam]")
        mq = mod.train_adam(
            quant_model,
            x,
            epochs=args.qat_epochs,
            lr=args.qat_lr,
            print_every=max(50, args.qat_epochs // 4),
            tag="qat",
        )
    else:
        mq = mq0
    print(f"[qat final] ll={mq['lossless']:.2f}% bad={mq['bad_bytes']} bit={mq['per_bit']:.2f}% uniq={mq['unique_latents']}")

    return SweepResult(
        hidden=hidden,
        latent_dim=latent_dim,
        activation=activation,
        codebook_name=codebook_name,
        float_lossless=mf["lossless"],
        float_bad=mf["bad_bytes"],
        static_lossless=static_ll,
        final_lossless=mq["lossless"],
        final_bad=mq["bad_bytes"],
        final_per_bit=mq["per_bit"],
        unique_latents=mq["unique_latents"],
        alpha1=a1,
        alpha2=a2,
        time_s=time.time() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dims", default="8,10,12,16,24")
    parser.add_argument("--hiddens", default="8,12,16,24")
    parser.add_argument("--activations", default="identity,tanh,relu,c19")
    parser.add_argument("--codebooks", default="binary")
    parser.add_argument("--float-epochs", type=int, default=150)
    parser.add_argument("--float-lr", type=float, default=2e-3)
    parser.add_argument("--qat-epochs", type=int, default=150)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="output/byte_unit_latent_dim_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = mod.build_dataset()
    x_np = x.detach().cpu().numpy()
    latent_dims = [int(s) for s in args.latent_dims.split(",") if s.strip()]
    hiddens = [int(s) for s in args.hiddens.split(",") if s.strip()]
    activations = [s.strip() for s in args.activations.split(",") if s.strip()]
    codebooks = [s.strip() for s in args.codebooks.split(",") if s.strip()]

    print("=" * 78)
    print("BYTE UNIT LATENT-DIM SWEEP")
    print("=" * 78)
    print(f"device={DEVICE}")
    print(f"latent_dims={latent_dims}")
    print(f"hiddens={hiddens}")
    print(f"activations={activations}")
    print(f"codebooks={codebooks}")
    print(f"float_epochs={args.float_epochs} qat_epochs={args.qat_epochs}")

    results: list[SweepResult] = []
    for cb in codebooks:
        for act in activations:
            for d in latent_dims:
                for h in hiddens:
                    results.append(run_one(h, d, act, cb, x, x_np, args))

    ranked = sorted(
        results,
        key=lambda r: (
            r.final_lossless,
            r.final_per_bit,
            -r.final_bad,
            -r.unique_latents,
            -r.latent_dim,
            -r.hidden,
        ),
        reverse=True,
    )

    exact_ranked = sorted(
        [r for r in results if r.final_lossless >= 100.0],
        key=lambda r: (r.latent_dim, r.hidden, -r.final_per_bit, r.time_s),
    )

    summary = {
        "device": str(DEVICE),
        "latent_dims": latent_dims,
        "hiddens": hiddens,
        "activations": activations,
        "codebooks": codebooks,
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "results": [r.__dict__ for r in results],
        "ranked": [r.__dict__ for r in ranked],
        "exact_ranked": [r.__dict__ for r in exact_ranked],
    }
    save_path = out_dir / "summary.json"
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("TOP RESULTS")
    print("=" * 78)
    for i, r in enumerate(ranked[:12], start=1):
        print(
            f"{i:2d}. D={r.latent_dim:<3d} H={r.hidden:<3d} act={r.activation:<8s} cb={r.codebook_name:<8s} "
            f"final={r.final_lossless:6.2f}% bad={r.final_bad:3d} bit={r.final_per_bit:6.2f}% "
            f"uniq={r.unique_latents:3d} t={r.time_s:5.1f}s"
        )

    print("\n" + "=" * 78)
    print("EXACT FRONTIER")
    print("=" * 78)
    if not exact_ranked:
        print("No exact configurations found.")
    else:
        for i, r in enumerate(exact_ranked[:12], start=1):
            print(
                f"{i:2d}. D={r.latent_dim:<3d} H={r.hidden:<3d} act={r.activation:<8s} "
                f"bit={r.final_per_bit:6.2f}% uniq={r.unique_latents:3d} t={r.time_s:5.1f}s"
            )
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
