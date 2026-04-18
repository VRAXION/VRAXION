"""Shared utilities for exact H=81 byte-pair merger experiments.

The exact-first pipeline uses three stages:

  1. Pure float exact trainer.
  2. Exact lookup-codebook freeze.
  3. Strict staged int8 freeze.

These helpers keep the core math and exactness checks identical across stages.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LUT_IN_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class C19Activation(nn.Module):
    def __init__(self, dim: int, c_init: float = 1.0, rho_init: float = 8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(
            n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n)
        )
        interior = c * (sgn * h + rho * h * h)
        return torch.where(
            x >= L, x - L, torch.where(x <= -L, x + L, interior)
        )


class PureFloatMerger(nn.Module):
    """Unquantized 32 -> H -> 32 mirror-tied merger."""

    def __init__(self, hidden: int = 81, in_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.H = hidden
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W1 = nn.Parameter(torch.randn(in_dim, hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hidden, out_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        self.db1 = nn.Parameter(torch.zeros(hidden))
        self.db2 = nn.Parameter(torch.zeros(out_dim))
        self.c19 = C19Activation(hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.c19(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W2.t() + self.db1) @ self.W1.t() + self.db2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_byte_pairs() -> torch.Tensor:
    with open(LUT_IN_PATH, "r", encoding="utf-8") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32) * scale
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1)


def eval_stats(
    model: nn.Module, data: torch.Tensor
) -> dict[str, float | int | bool]:
    with torch.no_grad():
        x_hat, _ = model(data)
        sign_match = torch.sign(x_hat) == torch.sign(data)
        pair_ok = sign_match.all(dim=1)
        exact = bool(pair_ok.all().item())
        lossless = pair_ok.float().mean().item() * 100.0
        per_dim = sign_match.float().mean().item() * 100.0
        mse = F.mse_loss(x_hat, data).item()
        bad_pairs = int((~pair_ok).sum().item())
        bad_dims = int((~sign_match).sum().item())
    return {
        "exact": exact,
        "lossless": lossless,
        "per_dim": per_dim,
        "mse": mse,
        "bad_pairs": bad_pairs,
        "bad_dims": bad_dims,
    }


def _masked_lbfgs_grad_zero(model: nn.Module) -> None:
    # Some quantized models carry frozen masks alongside float storage.
    if hasattr(model, "W1_float") and hasattr(model, "W1_frozen_mask"):
        grad = getattr(model.W1_float, "grad", None)
        if grad is not None:
            mask = model.W1_frozen_mask
            if hasattr(model, "W1_int8_mask"):
                mask = mask | model.W1_int8_mask
            grad[mask] = 0.0
    if hasattr(model, "W2_float") and hasattr(model, "W2_frozen_mask"):
        grad = getattr(model.W2_float, "grad", None)
        if grad is not None:
            mask = model.W2_frozen_mask
            if hasattr(model, "W2_int8_mask"):
                mask = mask | model.W2_int8_mask
            grad[mask] = 0.0


def train_adam(
    model: nn.Module,
    data: torch.Tensor,
    n_epochs: int,
    lr: float = 1e-3,
    print_every: int = 200,
    tag: str = "adam",
    log=None,
) -> dict[str, float | int | bool]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_stats = eval_stats(model, data)
    for ep in range(n_epochs):
        opt.zero_grad()
        x_hat, _ = model(data)
        loss = F.mse_loss(x_hat, data)
        loss.backward()
        _masked_lbfgs_grad_zero(model)
        opt.step()
        if ep == n_epochs - 1 or (print_every > 0 and (ep + 1) % print_every == 0):
            last_stats = eval_stats(model, data)
            msg = (
                f"  [{tag} ep {ep+1}] loss={loss.item():.6e} "
                f"ll={last_stats['lossless']:.4f}% "
                f"bad_pairs={last_stats['bad_pairs']} "
                f"mse={last_stats['mse']:.6e}"
            )
            if log is None:
                print(msg, flush=True)
            else:
                log(msg)
            if last_stats["exact"]:
                return last_stats
    return last_stats


def train_lbfgs_plateau(
    model: nn.Module,
    data: torch.Tensor,
    max_outer: int = 200,
    patience: int = 30,
    print_every: int = 10,
    tag: str = "lbfgs",
    log=None,
) -> dict[str, float | int | bool]:
    opt = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    best_loss = float("inf")
    no_imp = 0
    best_snapshot = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }
    last_stats = eval_stats(model, data)

    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            x_hat, _ = model(data)
            loss = F.mse_loss(x_hat, data)
            loss.backward()
            _masked_lbfgs_grad_zero(model)
            return loss

        lv = opt.step(closure)
        loss = float(lv.item() if torch.is_tensor(lv) else lv)
        last_stats = eval_stats(model, data)

        if loss < best_loss - 1e-12:
            best_loss = loss
            no_imp = 0
            best_snapshot = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        else:
            no_imp += 1

        if outer == max_outer - 1 or (print_every > 0 and (outer + 1) % print_every == 0):
            msg = (
                f"  [{tag} outer {outer+1}] loss={loss:.6e} "
                f"ll={last_stats['lossless']:.4f}% "
                f"bad_pairs={last_stats['bad_pairs']} "
                f"mse={last_stats['mse']:.6e} "
                f"no_imp={no_imp}"
            )
            if log is None:
                print(msg, flush=True)
            else:
                log(msg)

        if last_stats["exact"]:
            return last_stats
        if no_imp >= patience:
            break

    model.load_state_dict(best_snapshot, strict=True)
    return eval_stats(model, data)


def _effective_from_codebook_state(d: dict) -> tuple[np.ndarray, np.ndarray]:
    codebook_W1 = np.array(d["codebook_W1"], dtype=np.float32)
    codebook_W2 = np.array(d["codebook_W2"], dtype=np.float32)
    W1_idx = np.array(d["W1_idx"], dtype=np.int64)
    W2_idx = np.array(d["W2_idx"], dtype=np.int64)
    W1_frozen = np.array(d["W1_frozen_mask"]).astype(bool)
    W2_frozen = np.array(d["W2_frozen_mask"]).astype(bool)
    W1_float = np.array(d["W1_float"], dtype=np.float32)
    W2_float = np.array(d["W2_float"], dtype=np.float32)

    W1_eff = W1_float.copy()
    W2_eff = W2_float.copy()

    if "W1_int8_mask" in d and "W1_int8" in d:
        W1_i8_mask = np.array(d["W1_int8_mask"]).astype(bool)
        W2_i8_mask = np.array(d["W2_int8_mask"]).astype(bool)
        a1 = float(d["alpha_free_W1"])
        a2 = float(d["alpha_free_W2"])
        W1_eff = np.where(W1_i8_mask, a1 * np.array(d["W1_int8"], dtype=np.float32), W1_eff)
        W2_eff = np.where(W2_i8_mask, a2 * np.array(d["W2_int8"], dtype=np.float32), W2_eff)

    W1_eff = np.where(W1_frozen, codebook_W1[W1_idx], W1_eff)
    W2_eff = np.where(W2_frozen, codebook_W2[W2_idx], W2_eff)
    return W1_eff, W2_eff


def _effective_from_alpha_int_state(d: dict) -> tuple[np.ndarray, np.ndarray]:
    a1 = float(d["alpha_W1"])
    a2 = float(d.get("alpha_W2", a1))
    W1_frozen = np.array(d["W1_frozen_mask"]).astype(bool)
    W2_frozen = np.array(d["W2_frozen_mask"]).astype(bool)
    W1_int = np.array(d["W1_int"], dtype=np.float32)
    W2_int = np.array(d["W2_int"], dtype=np.float32)
    W1_float = np.array(d.get("W1_float", np.zeros_like(W1_int)), dtype=np.float32)
    W2_float = np.array(d.get("W2_float", np.zeros_like(W2_int)), dtype=np.float32)
    W1_eff = np.where(W1_frozen, a1 * W1_int, W1_float)
    W2_eff = np.where(W2_frozen, a2 * W2_int, W2_float)
    return W1_eff, W2_eff


def load_effective_pure_float(path: str | Path) -> PureFloatMerger:
    d = load_json(path)
    if "W1" in d and "W2" in d:
        W1_eff = np.array(d["W1"], dtype=np.float32)
        W2_eff = np.array(d["W2"], dtype=np.float32)
    elif "codebook_W1" in d:
        W1_eff, W2_eff = _effective_from_codebook_state(d)
    elif "alpha_W1" in d and "W1_int" in d:
        W1_eff, W2_eff = _effective_from_alpha_int_state(d)
    else:
        raise ValueError(f"Unsupported merger artifact format: {path}")

    H = int(d["H"])
    in_dim = int(d["in_dim"])
    out_dim = int(d["out_dim"])
    model = PureFloatMerger(hidden=H, in_dim=in_dim, out_dim=out_dim)
    with torch.no_grad():
        model.W1.copy_(torch.tensor(W1_eff, dtype=torch.float32))
        model.W2.copy_(torch.tensor(W2_eff, dtype=torch.float32))
        model.b1.copy_(torch.tensor(d["b1"], dtype=torch.float32))
        model.b2.copy_(torch.tensor(d["b2"], dtype=torch.float32))
        model.db1.copy_(torch.tensor(d["db1"], dtype=torch.float32))
        model.db2.copy_(torch.tensor(d["db2"], dtype=torch.float32))
        model.c19.c_raw.copy_(torch.tensor(d["c19_c"], dtype=torch.float32))
        model.c19.rho_raw.copy_(torch.tensor(d["c19_rho"], dtype=torch.float32))
    return model


def export_pure_float_json(
    out_path: str | Path,
    model: PureFloatMerger,
    stats: dict[str, float | int | bool],
    meta: dict | None = None,
) -> None:
    payload = {
        "architecture": "exact pure float merger",
        "H": model.H,
        "in_dim": model.in_dim,
        "out_dim": model.out_dim,
        "lossless": float(stats["lossless"]),
        "per_dim": float(stats["per_dim"]),
        "mse": float(stats["mse"]),
        "bad_pairs": int(stats["bad_pairs"]),
        "bad_dims": int(stats["bad_dims"]),
        "W1": model.W1.detach().cpu().numpy().tolist(),
        "W2": model.W2.detach().cpu().numpy().tolist(),
        "b1": model.b1.detach().cpu().numpy().tolist(),
        "b2": model.b2.detach().cpu().numpy().tolist(),
        "db1": model.db1.detach().cpu().numpy().tolist(),
        "db2": model.db2.detach().cpu().numpy().tolist(),
        "c19_c": model.c19.c_raw.detach().cpu().numpy().tolist(),
        "c19_rho": model.c19.rho_raw.detach().cpu().numpy().tolist(),
    }
    if meta:
        payload.update(meta)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
