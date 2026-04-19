"""Probe pure 2-bit weight quantization on the shipped L0 byte unit.

We test several true 4-level codebooks against the current exact int4 winner:
  - static warm-start search for best alpha pair per codebook
  - short fixed-alpha Adam-only STE polish on the best candidate

Why fixed-alpha + Adam-only?
  - the int4 probe showed the shipped exact point is stable under tiny Adam steps
  - free alpha / LBFGS polish can destroy exactness even for int4
  - this isolates the actual capacity of pure 2-bit much more cleanly
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


CODEBOOKS: dict[str, tuple[float, ...]] = {
    "sym12": (-2.0, -1.0, 1.0, 2.0),
    "sym13": (-3.0, -1.0, 1.0, 3.0),
    "halfstep": (-1.5, -0.5, 0.5, 1.5),
    "sym24": (-4.0, -2.0, 2.0, 4.0),
}


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


class STECodebook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, 1)
        dist = (flat - levels.reshape(1, -1)).abs()
        idx = dist.argmin(dim=1)
        out = levels[idx].reshape_as(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_levels: torch.Tensor | None = None):
        return grad_output, None


def ste_codebook(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return STECodebook.apply(x, levels)


def byte_to_signed_bits(byte: int) -> list[float]:
    return [1.0 if ((byte >> i) & 1) else -1.0 for i in range(8)]


def build_dataset() -> torch.Tensor:
    x = np.array([byte_to_signed_bits(i) for i in range(256)], dtype=np.float32)
    return torch.tensor(x, device=DEVICE)


def nearest_codebook_np(W: np.ndarray, levels: np.ndarray) -> np.ndarray:
    flat = W.reshape(-1, 1)
    dist = np.abs(flat - levels.reshape(1, -1))
    idx = dist.argmin(axis=1)
    return levels[idx].reshape(W.shape)


class QATByteUnit2Bit(nn.Module):
    def __init__(self, codebook: tuple[float, ...]):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(8, 24, device=DEVICE))
        self.W2 = nn.Parameter(torch.zeros(24, 16, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.b1 = nn.Parameter(torch.zeros(24, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(16, device=DEVICE))
        self.c19 = C19Activation(24)
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


def load_winner_blob() -> dict:
    return json.loads(WINNER_PATH.read_text(encoding="utf-8"))


def load_float_winner_into(model: QATByteUnit2Bit, blob: dict, a1: float, a2: float) -> None:
    W1 = np.array(blob["W1_int4"], dtype=np.float32) * float(blob["scale_W1"])
    W2 = np.array(blob["W2_int4"], dtype=np.float32) * float(blob["scale_W2"])
    with torch.no_grad():
        model.W1.copy_(torch.tensor(W1, device=DEVICE))
        model.W2.copy_(torch.tensor(W2, device=DEVICE))
        model.b1.copy_(torch.tensor(blob["bias1"], dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(blob["bias2"], dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(blob["c19_c"], dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(blob["c19_rho"], dtype=torch.float32, device=DEVICE))
        model.alpha1_raw.copy_(torch.log(torch.expm1(torch.tensor(a1, device=DEVICE))))
        model.alpha2_raw.copy_(torch.log(torch.expm1(torch.tensor(a2, device=DEVICE))))


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


def choose_best_alpha_pair(W1: np.ndarray, W2: np.ndarray, blob: dict, codebook: tuple[float, ...]) -> tuple[float, float, float]:
    c = np.array(blob["c19_c"], dtype=np.float32)
    rho = np.array(blob["c19_rho"], dtype=np.float32)
    b1 = np.array(blob["bias1"], dtype=np.float32)
    b2 = np.array(blob["bias2"], dtype=np.float32)
    x = np.array([byte_to_signed_bits(i) for i in range(256)], dtype=np.float32)
    levels = np.array(codebook, dtype=np.float32)
    base1 = float(np.abs(W1).mean())
    base2 = float(np.abs(W2).mean())
    best = None
    for m1 in (0.125, 0.167, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
        for m2 in (0.125, 0.167, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
            a1 = base1 * m1
            a2 = base2 * m2
            W1q = nearest_codebook_np(W1, a1 * levels)
            W2q = nearest_codebook_np(W2, a2 * levels)
            z = c19_np(x @ W1q + b1, c, rho) @ W2q + b2
            y = (z @ W2q.T) @ W1q.T
            lossless = float(((np.sign(y) == np.sign(x)).all(axis=1)).mean() * 100.0)
            cand = (lossless, a1, a2)
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return float(best[0]), float(best[1]), float(best[2])


@torch.no_grad()
def metrics(model: QATByteUnit2Bit, x: torch.Tensor) -> dict:
    z, x_hat = model(x)
    ok_bits = torch.sign(x_hat).eq(torch.sign(x))
    byte_exact = ok_bits.all(dim=1)
    q1 = model.quant_W1().detach().cpu().numpy()
    q2 = model.quant_W2().detach().cpu().numpy()
    codebook = sorted({float(v) for v in np.unique(np.concatenate([q1.reshape(-1), q2.reshape(-1)]))})
    return {
        "lossless": float(byte_exact.float().mean().item() * 100.0),
        "bad_bytes": int((~byte_exact).sum().item()),
        "per_bit": float(ok_bits.float().mean().item() * 100.0),
        "alpha1": float(model.alpha1.item()),
        "alpha2": float(model.alpha2.item()),
        "used_levels": codebook,
    }


def objective(model: QATByteUnit2Bit, x: torch.Tensor) -> torch.Tensor:
    _, x_hat = model(x)
    mse = ((x_hat - x) ** 2).mean()
    sign_hinge = torch.relu(-x_hat * torch.sign(x)).mean()
    return mse + 2.0 * sign_hinge


def train_adam(model: QATByteUnit2Bit, x: torch.Tensor, epochs: int, lr: float) -> None:
    if epochs <= 0:
        return
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
        if (ep + 1) % 100 == 0 or ep == 0:
            print(
                f"  [adam {ep+1:4d}] loss={loss.item():.6f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_bytes']:3d} "
                f"bit={m['per_bit']:.2f}% a1={m['alpha1']:.6f} a2={m['alpha2']:.6f}",
                flush=True,
            )
        if m["lossless"] >= 100.0:
            return
    model.load_state_dict(best_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", default="all", choices=["all", *CODEBOOKS.keys()])
    parser.add_argument("--adam-epochs", type=int, default=400)
    parser.add_argument("--adam-lr", type=float, default=5e-4)
    parser.add_argument("--out", default="output/byte_unit_qat_2bit_probe")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BYTE UNIT PURE 2-BIT PROBE — 8 -> 24 -> 16")
    print("=" * 70)

    blob = load_winner_blob()
    W1 = np.array(blob["W1_int4"], dtype=np.float32) * float(blob["scale_W1"])
    W2 = np.array(blob["W2_int4"], dtype=np.float32) * float(blob["scale_W2"])
    x = build_dataset()

    names = list(CODEBOOKS) if args.codebook == "all" else [args.codebook]
    results = []

    print("\n[1] Static warm-start sweep")
    for name in names:
        codebook = CODEBOOKS[name]
        ll, a1, a2 = choose_best_alpha_pair(W1, W2, blob, codebook)
        results.append({"name": name, "codebook": codebook, "warm_lossless": ll, "alpha1": a1, "alpha2": a2})
        print(f"  {name:>8} {codebook} -> ll={ll:.2f}% a1={a1:.6f} a2={a2:.6f}")

    best = max(results, key=lambda r: r["warm_lossless"])
    print("\n[2] Best codebook")
    print(f"  {best['name']} {best['codebook']} -> ll={best['warm_lossless']:.2f}%")

    model = QATByteUnit2Bit(best["codebook"]).to(DEVICE)
    load_float_winner_into(model, blob, best["alpha1"], best["alpha2"])
    model.alpha1_raw.requires_grad_(False)
    model.alpha2_raw.requires_grad_(False)

    m0 = metrics(model, x)
    print("\n[3] Warm-start metrics")
    for k in ("lossless", "bad_bytes", "per_bit", "alpha1", "alpha2", "used_levels"):
        print(f"  {k:>12}: {m0[k]}")

    print("\n[4] Fixed-alpha Adam polish")
    t0 = time.time()
    train_adam(model, x, args.adam_epochs, args.adam_lr)
    final = metrics(model, x)
    elapsed = time.time() - t0

    print("\n[5] Final metrics")
    for k in ("lossless", "bad_bytes", "per_bit", "alpha1", "alpha2", "used_levels"):
        print(f"  {k:>12}: {final[k]}")
    print(f"  {'time_s':>12}: {elapsed:.2f}")

    payload = {
        "architecture": "C19 1H, 8->24->16, tied mirror, pure 2-bit probe",
        "best_codebook_name": best["name"],
        "best_codebook": list(best["codebook"]),
        "static_results": results,
        "warm_metrics": m0,
        "final_metrics": {**final, "time_s": elapsed},
    }
    save_path = out_dir / "byte_unit_qat_2bit_probe.json"
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
