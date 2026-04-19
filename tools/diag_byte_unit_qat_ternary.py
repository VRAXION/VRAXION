"""QAT ternary experiment for the L0 byte unit (8 -> 24 -> 16, tied mirror).

Goal:
  - start from the shipped int4 byte unit winner
  - replace W1/W2 with ternary weights via STE in forward
  - keep b1/b2/C19 float for the first experiment
  - optimize exact roundtrip on all 256 bytes

Roundtrip metric:
  byte -> 8 signed bits -> encoder -> 16D latent -> tied linear decode -> 8 signed bits
  exact byte = all 8 reconstructed bit signs match the original bits

This is intentionally a small, exhaustive experiment; the full dataset is only 256 bytes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINNER_PATH = Path(__file__).with_name("byte_unit_winner_int4.json")


class C19Activation(nn.Module):
    def __init__(self, dim: int, c_init: float = 1.0, rho_init: float = 8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init, device=DEVICE))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init, device=DEVICE))

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


class STETernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, alpha)
        thr = 0.5 * alpha
        out = torch.zeros_like(x)
        out = torch.where(x > thr, alpha.expand_as(out), out)
        out = torch.where(x < -thr, -alpha.expand_as(out), out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, alpha = ctx.saved_tensors
        grad = grad_output.clone()
        # Standard clipped STE: let gradient flow near the active region.
        clip = (x.abs() <= (2.0 * alpha)).to(grad.dtype)
        return grad * clip, None


def ste_ternary(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return STETernary.apply(x, alpha)


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


class QATByteUnit(nn.Module):
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
        code = ste_round(self.W1 / a).clamp(-1.0, 1.0)
        return a * code

    def quant_W2(self) -> torch.Tensor:
        a = self.alpha2
        code = ste_round(self.W2 / a).clamp(-1.0, 1.0)
        return a * code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        W1_q = self.quant_W1()
        W2_q = self.quant_W2()
        h = self.c19(x @ W1_q + self.b1)
        return h @ W2_q + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        W1_q = self.quant_W1()
        W2_q = self.quant_W2()
        # Public docs and shipped winner imply a plain tied linear mirror.
        return (z @ W2_q.t()) @ W1_q.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def load_int4_winner(model: QATByteUnit) -> None:
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
        a1, a2 = choose_best_alpha_pair(W1, W2, blob)
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


def ternary_static(W: np.ndarray, alpha: float) -> np.ndarray:
    code = np.clip(np.round(W / alpha), -1.0, 1.0)
    return alpha * code


def choose_best_alpha_pair(W1: np.ndarray, W2: np.ndarray, blob: dict) -> tuple[float, float]:
    c = np.array(blob["c19_c"], dtype=np.float32)
    rho = np.array(blob["c19_rho"], dtype=np.float32)
    b1 = np.array(blob["bias1"], dtype=np.float32)
    b2 = np.array(blob["bias2"], dtype=np.float32)
    x = np.array([byte_to_signed_bits(i) for i in range(256)], dtype=np.float32)
    base1 = float(np.abs(W1).mean())
    base2 = float(np.abs(W2).mean())
    best = None
    for m1 in (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
        for m2 in (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0):
            a1 = base1 * m1
            a2 = base2 * m2
            W1q = ternary_static(W1, a1)
            W2q = ternary_static(W2, a2)
            z = c19_np(x @ W1q + b1, c, rho) @ W2q + b2
            y = (z @ W2q.T) @ W1q.T
            lossless = float(((np.sign(y) == np.sign(x)).all(axis=1)).mean() * 100.0)
            cand = (lossless, a1, a2)
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return float(best[1]), float(best[2])


@torch.no_grad()
def metrics(model: QATByteUnit, x: torch.Tensor) -> dict:
    z, x_hat = model(x)
    sign = torch.sign(x_hat)
    ok_bits = sign.eq(torch.sign(x))
    byte_exact = ok_bits.all(dim=1)
    latent = z.detach().cpu().numpy()
    uniq = np.unique(np.round(latent, 6), axis=0).shape[0]
    W1_q = model.quant_W1()
    W2_q = model.quant_W2()
    tern1 = (W1_q / model.alpha1.clamp(min=1e-6)).round().detach()
    tern2 = (W2_q / model.alpha2.clamp(min=1e-6)).round().detach()
    return {
        "lossless": float(byte_exact.float().mean().item() * 100.0),
        "bad_bytes": int((~byte_exact).sum().item()),
        "per_bit": float(ok_bits.float().mean().item() * 100.0),
        "unique_latents": int(uniq),
        "alpha1": float(model.alpha1.item()),
        "alpha2": float(model.alpha2.item()),
        "w1_zero_pct": float((tern1 == 0).float().mean().item() * 100.0),
        "w2_zero_pct": float((tern2 == 0).float().mean().item() * 100.0),
    }


def objective(model: QATByteUnit, x: torch.Tensor, latent_margin: float = 0.0) -> torch.Tensor:
    z, x_hat = model(x)
    mse = ((x_hat - x) ** 2).mean()
    sign_hinge = torch.relu(-x_hat * torch.sign(x)).mean()
    loss = mse + 2.0 * sign_hinge
    if latent_margin > 0.0:
        # Encourage byte latents not to collapse on top of each other.
        d = torch.cdist(z, z, p=2)
        mask = ~torch.eye(d.shape[0], device=d.device, dtype=torch.bool)
        min_other = d.masked_fill(~mask, 1e9).min(dim=1).values
        loss = loss + latent_margin * torch.relu(0.5 - min_other).mean()
    return loss


def train_adam(model: QATByteUnit, x: torch.Tensor, epochs: int, lr: float, latent_margin: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    for ep in range(epochs):
        loss = objective(model, x, latent_margin=latent_margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            m = metrics(model, x)
            best = max(best, m["lossless"])
            print(
                f"  [adam {ep+1:4d}] loss={loss.item():.6f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_bytes']:3d} "
                f"bit={m['per_bit']:.2f}% uniq={m['unique_latents']:3d} "
                f"a1={m['alpha1']:.4f} a2={m['alpha2']:.4f}",
                flush=True,
            )
            if m["lossless"] >= 100.0:
                return


def train_lbfgs(model: QATByteUnit, x: torch.Tensor, max_outer: int, latent_margin: float) -> None:
    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=50,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    stall = 0
    best = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            loss = objective(model, x, latent_margin=latent_margin)
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
                f"  [lbfgs {outer+1:3d}] loss={loss:.6f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_bytes']:3d} "
                f"bit={m['per_bit']:.2f}% uniq={m['unique_latents']:3d} stall={stall}",
                flush=True,
            )
        if m["lossless"] >= 100.0:
            return
        if stall >= 12:
            break
    model.load_state_dict(best_state)


def save_artifact(model: QATByteUnit, out_path: Path, final_metrics: dict) -> None:
    with torch.no_grad():
        W1_q = model.quant_W1()
        W2_q = model.quant_W2()
        t1 = torch.clamp(torch.round(W1_q / model.alpha1.clamp(min=1e-6)), -1, 1).cpu().numpy().astype(int)
        t2 = torch.clamp(torch.round(W2_q / model.alpha2.clamp(min=1e-6)), -1, 1).cpu().numpy().astype(int)
        payload = {
            "architecture": "C19 1H, 8->24->16, tied mirror, ternary QAT",
            "precision": "ternary weights + STE",
            "alpha1": float(model.alpha1.item()),
            "alpha2": float(model.alpha2.item()),
            "W1_ternary": t1.tolist(),
            "W2_ternary": t2.tolist(),
            "bias1": model.b1.detach().cpu().numpy().tolist(),
            "bias2": model.b2.detach().cpu().numpy().tolist(),
            "c19_c": model.c19.c_raw.detach().cpu().numpy().tolist(),
            "c19_rho": model.c19.rho_raw.detach().cpu().numpy().tolist(),
            "metrics": final_metrics,
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/byte_unit_qat_ternary")
    parser.add_argument("--adam-epochs", type=int, default=1500)
    parser.add_argument("--adam-lr", type=float, default=5e-3)
    parser.add_argument("--lbfgs-outer", type=int, default=60)
    parser.add_argument("--latent-margin", type=float, default=0.0)
    parser.add_argument("--freeze-alpha", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BYTE UNIT TERNARY QAT (STE) — 8 -> 24 -> 16")
    print("=" * 70)

    x = build_dataset()
    model = QATByteUnit().to(DEVICE)
    load_int4_winner(model)
    if args.freeze_alpha:
        model.alpha1_raw.requires_grad_(False)
        model.alpha2_raw.requires_grad_(False)

    print("\n[1] Warm-started from shipped int4 winner")
    m0 = metrics(model, x)
    for k in ("lossless", "bad_bytes", "per_bit", "unique_latents", "alpha1", "alpha2", "w1_zero_pct", "w2_zero_pct"):
        print(f"  {k:>14}: {m0[k]}")

    t0 = time.time()
    print(f"\n[2] Adam QAT  (freeze_alpha={args.freeze_alpha})")
    train_adam(model, x, epochs=args.adam_epochs, lr=args.adam_lr, latent_margin=args.latent_margin)

    print("\n[3] LBFGS polish")
    train_lbfgs(model, x, max_outer=args.lbfgs_outer, latent_margin=args.latent_margin)

    final = metrics(model, x)
    elapsed = time.time() - t0

    print("\n[4] Final metrics")
    for k in ("lossless", "bad_bytes", "per_bit", "unique_latents", "alpha1", "alpha2", "w1_zero_pct", "w2_zero_pct"):
        print(f"  {k:>14}: {final[k]}")
    print(f"  {'time_s':>14}: {elapsed:.1f}")

    save_path = out_dir / "byte_unit_qat_ternary.json"
    save_artifact(model, save_path, {**final, "time_s": elapsed})
    print(f"\nSaved: {save_path}")

    w_bytes = math.ceil((8 * 24 + 24 * 16) * 2 / 8) + 8  # pack ternary into 2 bits/weight + 2 scales
    aux_bytes = (24 + 16 + 24 + 24) * 2  # rough fp16 bias + c19
    total = w_bytes + aux_bytes
    print("\nDeploy estimate (ternary W, fp16 bias/C19):")
    print(f"  W packed + scales: {w_bytes} B")
    print(f"  bias/C19 fp16    : {aux_bytes} B")
    print(f"  total            : {total} B ({total/1024:.2f} KB)")


if __name__ == "__main__":
    main()
