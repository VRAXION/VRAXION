"""Widen sweep for the L0 byte unit under low-bit quantization.

Question:
  Can a wider 8 -> H -> 16 tied-mirror byte unit rescue lower bitwidths
  (pure 2-bit / pure 3-bit), and does activation choice matter?

Protocol per config:
  1. Build float model (optionally warm-start C19 from shipped H=24 winner).
  2. Float Adam warmup on all 256 bytes.
  3. Static alpha search for the chosen codebook.
  4. Fixed-alpha QAT Adam polish with STE codebook snapping.
  5. Full exactness check on all 256 bytes.

This is intentionally a bounded first-pass screen, not the final overnight run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINNER_PATH = Path(__file__).with_name("byte_unit_winner_int4.json")


CODEBOOKS: dict[str, tuple[float, ...]] = {
    "2bit_sym13": (-3.0, -1.0, 1.0, 3.0),
    "3bit_sym1248": (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0),
}


def byte_to_signed_bits(byte: int) -> list[float]:
    return [1.0 if ((byte >> i) & 1) else -1.0 for i in range(8)]


def build_dataset() -> torch.Tensor:
    x = np.array([byte_to_signed_bits(i) for i in range(256)], dtype=np.float32)
    return torch.tensor(x, device=DEVICE)


class C19Activation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), 3.0, device=DEVICE))
        self.rho_raw = nn.Parameter(torch.full((dim,), 1.0, device=DEVICE))

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


class IdentityActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ReLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class LeakyReLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0.01 * x)


class SiLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SoftplusActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)


class TanhActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


def make_activation(name: str, hidden: int) -> nn.Module:
    if name == "c19":
        return C19Activation(hidden)
    if name == "relu":
        return ReLUActivation()
    if name == "leaky_relu":
        return LeakyReLUActivation()
    if name == "silu":
        return SiLUActivation()
    if name == "softplus":
        return SoftplusActivation()
    if name == "tanh":
        return TanhActivation()
    if name == "identity":
        return IdentityActivation()
    raise ValueError(f"unknown activation: {name}")


class FloatByteUnit(nn.Module):
    def __init__(self, hidden: int, activation: str):
        super().__init__()
        self.hidden = hidden
        self.activation_name = activation
        self.W1 = nn.Parameter(torch.empty(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.empty(hidden, 16, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(hidden, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(16, device=DEVICE))
        self.act = make_activation(activation, hidden)
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


class STECodebook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, 1)
        dist = (flat - levels.reshape(1, -1)).abs()
        idx = dist.argmin(dim=1)
        return levels[idx].reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_levels: torch.Tensor | None = None):
        return grad_output, None


def ste_codebook(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return STECodebook.apply(x, levels)


class QuantByteUnit(nn.Module):
    def __init__(self, hidden: int, activation: str, codebook: tuple[float, ...]):
        super().__init__()
        self.hidden = hidden
        self.activation_name = activation
        self.W1 = nn.Parameter(torch.zeros(8, hidden, device=DEVICE))
        self.W2 = nn.Parameter(torch.zeros(hidden, 16, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(hidden, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(16, device=DEVICE))
        self.act = make_activation(activation, hidden)
        self.register_buffer("codebook", torch.tensor(codebook, dtype=torch.float32, device=DEVICE))

    @property
    def alpha1(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha1_raw) + 1e-6

    @property
    def alpha2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha2_raw) + 1e-6

    def quant_W1(self) -> torch.Tensor:
        return ste_codebook(self.W1, self.alpha1 * self.codebook)

    def quant_W2(self) -> torch.Tensor:
        return ste_codebook(self.W2, self.alpha2 * self.codebook)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x @ self.quant_W1() + self.b1)
        return h @ self.quant_W2() + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.quant_W2().t()) @ self.quant_W1().t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def load_winner_blob() -> dict:
    return json.loads(WINNER_PATH.read_text(encoding="utf-8"))


def warm_start_from_winner(model: FloatByteUnit, blob: dict) -> bool:
    if model.activation_name != "c19" or model.hidden < 24:
        return False
    W1 = np.array(blob["W1_int4"], dtype=np.float32) * float(blob["scale_W1"])
    W2 = np.array(blob["W2_int4"], dtype=np.float32) * float(blob["scale_W2"])
    with torch.no_grad():
        model.W1.zero_()
        model.W2.zero_()
        model.b1.zero_()
        model.b2.copy_(torch.tensor(blob["bias2"], dtype=torch.float32, device=DEVICE))
        model.W1[:, :24].copy_(torch.tensor(W1, dtype=torch.float32, device=DEVICE))
        model.W2[:24, :].copy_(torch.tensor(W2, dtype=torch.float32, device=DEVICE))
        model.b1[:24].copy_(torch.tensor(blob["bias1"], dtype=torch.float32, device=DEVICE))
        act = model.act
        assert isinstance(act, C19Activation)
        act.c_raw.fill_(3.0)
        act.rho_raw.fill_(1.0)
        act.c_raw[:24].copy_(torch.tensor(blob["c19_c"], dtype=torch.float32, device=DEVICE))
        act.rho_raw[:24].copy_(torch.tensor(blob["c19_rho"], dtype=torch.float32, device=DEVICE))
    return True


@torch.no_grad()
def metrics(model: nn.Module, x: torch.Tensor) -> dict:
    z, x_hat = model(x)
    ok_bits = torch.sign(x_hat).eq(torch.sign(x))
    byte_exact = ok_bits.all(dim=1)
    return {
        "lossless": float(byte_exact.float().mean().item() * 100.0),
        "bad_bytes": int((~byte_exact).sum().item()),
        "per_bit": float(ok_bits.float().mean().item() * 100.0),
        "unique_latents": int(np.unique(np.round(z.detach().cpu().numpy(), 6), axis=0).shape[0]),
    }


def objective(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    _, x_hat = model(x)
    mse = ((x_hat - x) ** 2).mean()
    sign_hinge = torch.relu(-x_hat * torch.sign(x)).mean()
    return mse + 2.0 * sign_hinge


def train_adam(model: nn.Module, x: torch.Tensor, epochs: int, lr: float, print_every: int = 100, tag: str = "adam") -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        loss = objective(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        m = metrics(model, x)
        if m["lossless"] > best + 1e-6:
            best = m["lossless"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % print_every == 0 or ep == 0:
            print(
                f"  [{tag} {ep+1:4d}] loss={loss.item():.6f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_bytes']:3d} "
                f"bit={m['per_bit']:.2f}% uniq={m['unique_latents']:3d}",
                flush=True,
            )
        if m["lossless"] >= 100.0:
            return m
    model.load_state_dict(best_state)
    return metrics(model, x)


def c19_np(x: np.ndarray, c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.1, None)
    rho = np.clip(rho, 0.0, None)
    L = 6.0 * c
    scaled = x / c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sgn = np.where((n.astype(np.int64) % 2) == 0, 1.0, -1.0)
    interior = c * (sgn * h + rho * h * h)
    return np.where(x >= L, x - L, np.where(x <= -L, x + L, interior))


def activation_np(kind: str, pre: np.ndarray, c: np.ndarray | None = None, rho: np.ndarray | None = None) -> np.ndarray:
    if kind == "c19":
        assert c is not None and rho is not None
        return c19_np(pre, c, rho)
    if kind == "relu":
        return np.maximum(pre, 0.0)
    if kind == "leaky_relu":
        return np.where(pre > 0.0, pre, 0.01 * pre)
    if kind == "silu":
        return pre * (1.0 / (1.0 + np.exp(-pre)))
    if kind == "softplus":
        return np.log1p(np.exp(-np.abs(pre))) + np.maximum(pre, 0.0)
    if kind == "tanh":
        return np.tanh(pre)
    if kind == "identity":
        return pre
    raise ValueError(kind)


def nearest_codebook_np(W: np.ndarray, levels: np.ndarray) -> np.ndarray:
    flat = W.reshape(-1, 1)
    dist = np.abs(flat - levels.reshape(1, -1))
    idx = dist.argmin(axis=1)
    return levels[idx].reshape(W.shape)


def choose_best_alpha_pair(float_model: FloatByteUnit, x_np: np.ndarray, codebook: tuple[float, ...]) -> tuple[float, float, float]:
    W1 = float_model.W1.detach().cpu().numpy()
    W2 = float_model.W2.detach().cpu().numpy()
    b1 = float_model.b1.detach().cpu().numpy()
    b2 = float_model.b2.detach().cpu().numpy()
    levels = np.array(codebook, dtype=np.float32)
    if float_model.activation_name == "c19":
        act = float_model.act
        assert isinstance(act, C19Activation)
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
            W1q = nearest_codebook_np(W1, a1 * levels)
            W2q = nearest_codebook_np(W2, a2 * levels)
            h = activation_np(float_model.activation_name, x_np @ W1q + b1, c, rho)
            z = h @ W2q + b2
            y = (z @ W2q.T) @ W1q.T
            lossless = float(((np.sign(y) == np.sign(x_np)).all(axis=1)).mean() * 100.0)
            cand = (lossless, a1, a2)
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return float(best[0]), float(best[1]), float(best[2])


def init_quant_from_float(quant: QuantByteUnit, float_model: FloatByteUnit, a1: float, a2: float) -> None:
    with torch.no_grad():
        quant.W1.copy_(float_model.W1.detach())
        quant.W2.copy_(float_model.W2.detach())
        quant.b1.copy_(float_model.b1.detach())
        quant.b2.copy_(float_model.b2.detach())
        quant.alpha1_raw.copy_(torch.log(torch.expm1(torch.tensor(a1, device=DEVICE))))
        quant.alpha2_raw.copy_(torch.log(torch.expm1(torch.tensor(a2, device=DEVICE))))
        if quant.activation_name == "c19":
            assert isinstance(quant.act, C19Activation) and isinstance(float_model.act, C19Activation)
            quant.act.c_raw.copy_(float_model.act.c_raw.detach())
            quant.act.rho_raw.copy_(float_model.act.rho_raw.detach())


@dataclass
class SweepResult:
    hidden: int
    activation: str
    codebook_name: str
    warm_started: bool
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


def run_one(hidden: int, activation: str, codebook_name: str, x: torch.Tensor, x_np: np.ndarray, args, blob: dict) -> SweepResult:
    t0 = time.time()
    print("\n" + "=" * 78)
    print(f"CONFIG hidden={hidden} activation={activation} codebook={codebook_name}")
    print("=" * 78)

    seed_offset = sum(ord(ch) for ch in (activation + "|" + codebook_name))
    torch.manual_seed(args.seed + hidden * 17 + seed_offset)
    np.random.seed(args.seed + hidden * 17 + seed_offset)

    float_model = FloatByteUnit(hidden, activation).to(DEVICE)
    warm_started = warm_start_from_winner(float_model, blob)
    m0 = metrics(float_model, x)
    print(f"[float init] ll={m0['lossless']:.2f}% bad={m0['bad_bytes']} bit={m0['per_bit']:.2f}% warm_start={warm_started}")

    if m0["lossless"] < 100.0 and args.float_epochs > 0:
        print("[float warmup]")
        mf = train_adam(float_model, x, epochs=args.float_epochs, lr=args.float_lr, print_every=max(50, args.float_epochs // 4), tag="float")
    else:
        mf = m0
    print(f"[float final] ll={mf['lossless']:.2f}% bad={mf['bad_bytes']} bit={mf['per_bit']:.2f}% uniq={mf['unique_latents']}")

    codebook = CODEBOOKS[codebook_name]
    static_ll, a1, a2 = choose_best_alpha_pair(float_model, x_np, codebook)
    print(f"[static snap] ll={static_ll:.2f}% a1={a1:.6f} a2={a2:.6f}")

    quant_model = QuantByteUnit(hidden, activation, codebook).to(DEVICE)
    init_quant_from_float(quant_model, float_model, a1, a2)
    quant_model.alpha1_raw.requires_grad_(False)
    quant_model.alpha2_raw.requires_grad_(False)
    mq0 = metrics(quant_model, x)
    print(f"[qat warm] ll={mq0['lossless']:.2f}% bad={mq0['bad_bytes']} bit={mq0['per_bit']:.2f}%")

    if args.qat_epochs > 0:
        print("[qat fixed-alpha Adam]")
        mq = train_adam(quant_model, x, epochs=args.qat_epochs, lr=args.qat_lr, print_every=max(50, args.qat_epochs // 4), tag="qat")
    else:
        mq = mq0
    print(f"[qat final] ll={mq['lossless']:.2f}% bad={mq['bad_bytes']} bit={mq['per_bit']:.2f}% uniq={mq['unique_latents']}")

    return SweepResult(
        hidden=hidden,
        activation=activation,
        codebook_name=codebook_name,
        warm_started=warm_started,
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
    parser.add_argument("--hiddens", default="24,32,48,64")
    parser.add_argument("--activations", default="c19,relu")
    parser.add_argument("--codebooks", default="2bit_sym13,3bit_sym1248")
    parser.add_argument("--float-epochs", type=int, default=300)
    parser.add_argument("--float-lr", type=float, default=2e-3)
    parser.add_argument("--qat-epochs", type=int, default=250)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="output/byte_unit_widen_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = build_dataset()
    x_np = x.detach().cpu().numpy()
    blob = load_winner_blob()

    hiddens = [int(s) for s in args.hiddens.split(",") if s.strip()]
    activations = [s.strip() for s in args.activations.split(",") if s.strip()]
    codebooks = [s.strip() for s in args.codebooks.split(",") if s.strip()]

    print("=" * 78)
    print("BYTE UNIT WIDEN SWEEP — LOW-BIT SCREEN")
    print("=" * 78)
    print(f"hiddens={hiddens}")
    print(f"activations={activations}")
    print(f"codebooks={codebooks}")
    print(f"float_epochs={args.float_epochs} qat_epochs={args.qat_epochs}")

    results: list[SweepResult] = []
    for cb in codebooks:
        for act in activations:
            for h in hiddens:
                results.append(run_one(h, act, cb, x, x_np, args, blob))

    ranked = sorted(results, key=lambda r: (r.final_lossless, r.final_per_bit, -r.final_bad), reverse=True)
    summary = {
        "hiddens": hiddens,
        "activations": activations,
        "codebooks": codebooks,
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "results": [r.__dict__ for r in results],
        "ranked": [r.__dict__ for r in ranked],
    }
    save_path = out_dir / "summary.json"
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RANKED RESULTS")
    print("=" * 78)
    for i, r in enumerate(ranked[:12], start=1):
        print(
            f"{i:2d}. H={r.hidden:<3d} act={r.activation:<4s} cb={r.codebook_name:<12s} "
            f"float={r.float_lossless:6.2f}% static={r.static_lossless:6.2f}% "
            f"final={r.final_lossless:6.2f}% bad={r.final_bad:3d} bit={r.final_per_bit:6.2f}% "
            f"t={r.time_s:5.1f}s"
        )
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
