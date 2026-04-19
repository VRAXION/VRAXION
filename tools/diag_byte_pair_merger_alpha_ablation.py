"""Ablate the global alpha in the 7-bit identity merger recipe.

Compares:
  - trainable alpha: Wq = ste(W, alpha * codebook)
  - no alpha:       Wq = ste(W, codebook)

This answers whether the current exact 7-bit identity line fundamentally needs
the shared scale, or whether pure integer levels are enough.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
LUT_PATH = ROOT / "byte_embedder_lut_int8_nozero.json"
CODEBOOK = tuple(float(v) for v in range(-63, 64) if v != 0)


def load_byte_pairs() -> torch.Tensor:
    blob = json.loads(LUT_PATH.read_text(encoding="utf-8"))
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32, device=DEVICE) * scale
    idx_a = torch.arange(256, device=DEVICE).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256, device=DEVICE).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1)


class STECodebook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, 1)
        dist = (flat - levels.reshape(1, -1)).abs()
        idx = dist.argmin(dim=1)
        return levels[idx].reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output, grad_levels=None):
        return grad_output, None


def ste_codebook(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return STECodebook.apply(x, levels)


class IdentityActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MergerSingleW(nn.Module):
    def __init__(self, H: int, use_alpha: bool, w_scale: float = 0.2):
        super().__init__()
        self.H = H
        self.use_alpha = use_alpha
        self.quantize = True
        self.W = nn.Parameter(torch.randn(32, H, device=DEVICE) * w_scale)
        self.b1 = nn.Parameter(torch.zeros(H, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(32, device=DEVICE))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.act = IdentityActivation()
        self.register_buffer("codebook", torch.tensor(CODEBOOK, dtype=torch.float32, device=DEVICE))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_raw) + 1e-6

    def quant_W(self) -> torch.Tensor:
        if not self.quantize:
            return self.W
        levels = self.codebook
        if self.use_alpha:
            levels = self.alpha * levels
        return ste_codebook(self.W, levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.quant_W()
        h = self.act(x @ W + self.b1)
        return h @ W.t() + self.b2


@torch.no_grad()
def metrics(model: MergerSingleW, data: torch.Tensor) -> dict:
    y = model(data)
    sign_match = torch.sign(y) == torch.sign(data)
    return {
        "lossless": float(sign_match.all(dim=1).float().mean().item() * 100.0),
        "per_dim": float(sign_match.float().mean().item() * 100.0),
        "bad_pairs": int((~sign_match.all(dim=1)).sum().item()),
    }


def objective(model: MergerSingleW, data: torch.Tensor) -> torch.Tensor:
    y = model(data)
    mse = ((y - data) ** 2).mean()
    sign_hinge = torch.relu(-y * torch.sign(data)).mean()
    return mse + 0.5 * sign_hinge


def set_alpha_raw(model: MergerSingleW, raw_val: float) -> None:
    with torch.no_grad():
        model.alpha_raw.fill_(raw_val)


def static_alpha_search(model: MergerSingleW, data: torch.Tensor, steps: int = 50) -> float:
    if not model.use_alpha:
        return metrics(model, data)["lossless"]
    with torch.no_grad():
        W_abs = model.W.detach().abs().flatten()
        lo = float(W_abs.min().item()) + 1e-8
        hi = float(W_abs.max().item()) + 1e-4
    best_ll = -1.0
    best_alpha = None
    saved = model.alpha_raw.detach().clone()
    for a_val in np.linspace(lo, hi, steps):
        raw_val = float(np.log(np.exp(a_val) - 1.0) if a_val > 1e-6 else -5.0)
        set_alpha_raw(model, raw_val)
        ll = metrics(model, data)["lossless"]
        if ll > best_ll:
            best_ll = ll
            best_alpha = float(a_val)
    with torch.no_grad():
        model.alpha_raw.copy_(saved)
    if best_alpha is not None:
        raw_val = float(np.log(np.exp(best_alpha) - 1.0) if best_alpha > 1e-6 else -5.0)
        set_alpha_raw(model, raw_val)
    return best_ll


def train_adam(model: MergerSingleW, data: torch.Tensor, epochs: int, lr: float, tag: str) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_ll = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        loss = objective(model, data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        m = metrics(model, data)
        if m["lossless"] > best_ll + 1e-6:
            best_ll = m["lossless"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 200 == 0 or ep == 0:
            print(f"    [{tag} {ep+1:4d}] loss={loss.item():.5f} ll={m['lossless']:.2f}% bad={m['bad_pairs']:5d}", flush=True)
        if m["lossless"] >= 100.0:
            return m
    model.load_state_dict(best_state)
    return metrics(model, data)


def train_lbfgs(model: MergerSingleW, data: torch.Tensor, max_outer: int, patience: int, tag: str) -> dict:
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    best_ll = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stall = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            loss = objective(model, data)
            loss.backward()
            return loss
        loss_val = opt.step(closure).item()
        m = metrics(model, data)
        if m["lossless"] > best_ll + 1e-4:
            best_ll = m["lossless"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 10 == 0 or outer == 0:
            print(f"    [{tag} {outer+1:3d}] loss={loss_val:.6f} ll={m['lossless']:.2f}% bad={m['bad_pairs']:5d} stall={stall}", flush=True)
        if m["lossless"] >= 100.0:
            print(f"    -> LOSSLESS @ outer {outer+1}", flush=True)
            return m
        if stall >= patience:
            print(f"    -> plateau (stall={stall})", flush=True)
            break
    model.load_state_dict(best_state)
    return metrics(model, data)


def run_one(seed: int, use_alpha: bool) -> dict:
    data = load_byte_pairs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MergerSingleW(120, use_alpha=use_alpha).to(DEVICE)
    model.quantize = False
    train_adam(model, data, 1500, 2e-3, "float")
    train_lbfgs(model, data, 150, 25, "float-lbfgs")
    model.quantize = True
    static_ll = static_alpha_search(model, data, steps=50)
    warm = metrics(model, data)
    train_adam(model, data, 800, 5e-4, "qat")
    final = train_lbfgs(model, data, 150, 25, "qat-lbfgs")
    out = {
        "seed": seed,
        "mode": "trainable_alpha" if use_alpha else "no_alpha",
        "static_lossless": static_ll,
        "warm_lossless": warm["lossless"],
        "final_lossless": final["lossless"],
        "final_bad": final["bad_pairs"],
        "final_per_dim": final["per_dim"],
    }
    if use_alpha:
        out["alpha"] = float(model.alpha.item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--out", default="S:/Git/VRAXION/output/merger_alpha_ablation")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        print("=" * 78, flush=True)
        print(f"SEED {seed} trainable_alpha", flush=True)
        print("=" * 78, flush=True)
        rows.append(run_one(seed, True))
        print("=" * 78, flush=True)
        print(f"SEED {seed} no_alpha", flush=True)
        print("=" * 78, flush=True)
        rows.append(run_one(seed, False))

    summary = {"seeds": seeds, "results": rows}
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}", flush=True)


if __name__ == "__main__":
    main()
