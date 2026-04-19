"""L1 Merger widen sweep — activation x codebook x hidden-dim.

Parallel to diag_byte_unit_widen_sweep.py, but for the L1 single-W mirror-tied
merger: forward(x) = act(x @ W + b1) @ W.T + b2 with in_dim=32 (two byte
latents concatenated) and hidden H.

Protocol per config:
  1. Adam warmup on all 65536 byte pairs (float).
  2. Fixed-alpha STE codebook polish (Adam) if codebook != "float".
  3. Optional LBFGS polish.
  4. Full exactness check on all 65536 pairs (sign-match byte_exact).

Intended as a first-pass screen; defaults tuned for ~30-60s/config on GPU.
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
LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


CODEBOOKS: dict[str, tuple[float, ...] | None] = {
    "float": None,
    "binary": (-1.0, 1.0),
    "ternary": (-1.0, 0.0, 1.0),
    "2bit_sym13": (-3.0, -1.0, 1.0, 3.0),
    "3bit_sym1248": (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0),
    "4bit_int": tuple(float(v) for v in range(-8, 9) if v != 0),
    "4bit_pow2": (-128.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    "5bit_int": tuple(float(v) for v in range(-15, 16) if v != 0),
    "6bit_int": tuple(float(v) for v in range(-31, 32) if v != 0),
    "7bit_int": tuple(float(v) for v in range(-63, 64) if v != 0),
    "8bit_int": tuple(float(v) for v in range(-127, 128) if v != 0),
    "9bit_int": tuple(float(v) for v in range(-255, 256) if v != 0),
    "10bit_int": tuple(float(v) for v in range(-511, 512) if v != 0),
}


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


class ReLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class TanhActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class IdentityActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LeakyReLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0.01 * x)


def make_activation(name: str, hidden: int) -> nn.Module:
    if name == "c19":
        return C19Activation(hidden)
    if name == "relu":
        return ReLUActivation()
    if name == "tanh":
        return TanhActivation()
    if name == "identity":
        return IdentityActivation()
    if name == "leaky_relu":
        return LeakyReLUActivation()
    raise ValueError(f"unknown activation: {name}")


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


class MergerSingleW(nn.Module):
    def __init__(self, in_dim: int, H: int, activation: str, codebook: tuple | None, w_scale: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.W = nn.Parameter(torch.randn(in_dim, H, device=DEVICE) * w_scale)
        self.b1 = nn.Parameter(torch.zeros(H, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(in_dim, device=DEVICE))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.act = make_activation(activation, H).to(DEVICE)
        if codebook is not None:
            self.register_buffer("codebook", torch.tensor(codebook, dtype=torch.float32, device=DEVICE))
            self.use_codebook = True
        else:
            self.use_codebook = False

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_raw) + 1e-6

    def quant_W(self) -> torch.Tensor:
        if self.use_codebook:
            return ste_codebook(self.W, self.alpha * self.codebook)
        return self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.quant_W()
        h = self.act(x @ W + self.b1)
        return h @ W.t() + self.b2


class MergerDualW(nn.Module):
    """Byte unit-style dual-W mirror tied: encode = act(x@W1+b1)@W2+b2, decode = (z@W2.T)@W1.T."""
    def __init__(self, in_dim: int, H: int, activation: str, codebook: tuple | None, w_scale: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.W1 = nn.Parameter(torch.randn(in_dim, H, device=DEVICE) * w_scale)
        self.W2 = nn.Parameter(torch.randn(H, in_dim, device=DEVICE) * w_scale)
        self.b1 = nn.Parameter(torch.zeros(H, device=DEVICE))
        self.b2 = nn.Parameter(torch.zeros(in_dim, device=DEVICE))
        self.alpha1_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self.act = make_activation(activation, H).to(DEVICE)
        if codebook is not None:
            self.register_buffer("codebook", torch.tensor(codebook, dtype=torch.float32, device=DEVICE))
            self.use_codebook = True
        else:
            self.use_codebook = False

    @property
    def alpha1(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha1_raw) + 1e-6

    @property
    def alpha2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha2_raw) + 1e-6

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha1

    def quant_W1(self) -> torch.Tensor:
        if self.use_codebook:
            return ste_codebook(self.W1, self.alpha1 * self.codebook)
        return self.W1

    def quant_W2(self) -> torch.Tensor:
        if self.use_codebook:
            return ste_codebook(self.W2, self.alpha2 * self.codebook)
        return self.W2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self.quant_W1()
        W2 = self.quant_W2()
        z = self.act(x @ W1 + self.b1) @ W2 + self.b2
        return (z @ W2.t()) @ W1.t()


def load_byte_pairs() -> torch.Tensor:
    with open(LUT_PATH, "r", encoding="utf-8") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32) * scale
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1).to(DEVICE)


@torch.no_grad()
def metrics(model: MergerSingleW, data: torch.Tensor) -> dict:
    y = model(data)
    sign_match = torch.sign(y) == torch.sign(data)
    lossless = float(sign_match.all(dim=1).float().mean().item() * 100.0)
    per_dim = float(sign_match.float().mean().item() * 100.0)
    bad_pairs = int((~sign_match.all(dim=1)).sum().item())
    return {"lossless": lossless, "per_dim": per_dim, "bad_pairs": bad_pairs}


def objective(model: MergerSingleW, data: torch.Tensor) -> torch.Tensor:
    y = model(data)
    mse = ((y - data) ** 2).mean()
    sign_hinge = torch.relu(-y * torch.sign(data)).mean()
    return mse + 0.5 * sign_hinge


def _set_alpha(model, raw_val: float) -> None:
    with torch.no_grad():
        if hasattr(model, "alpha1_raw"):
            model.alpha1_raw.fill_(raw_val)
            model.alpha2_raw.fill_(raw_val)
        else:
            model.alpha_raw.fill_(raw_val)


def _get_alpha_raw(model):
    if hasattr(model, "alpha1_raw"):
        return model.alpha1_raw.detach().clone()
    return model.alpha_raw.detach().clone()


def _restore_alpha(model, saved) -> None:
    with torch.no_grad():
        if hasattr(model, "alpha1_raw"):
            model.alpha1_raw.copy_(saved)
            model.alpha2_raw.copy_(saved)
        else:
            model.alpha_raw.copy_(saved)


def static_alpha_search(model, data: torch.Tensor, steps: int = 50) -> float:
    if not model.use_codebook:
        return 0.0
    with torch.no_grad():
        if hasattr(model, "W1"):
            W_abs = torch.cat([model.W1.detach().abs().flatten(), model.W2.detach().abs().flatten()])
        else:
            W_abs = model.W.detach().abs().flatten()
        lo = float(W_abs.min().item()) + 1e-8
        hi = float(W_abs.max().item()) + 1e-4
    best_ll = -1.0
    best_alpha = None
    saved = _get_alpha_raw(model)
    for a_val in np.linspace(lo, hi, steps):
        raw_val = float(np.log(np.exp(a_val) - 1.0) if a_val > 1e-6 else -5.0)
        _set_alpha(model, raw_val)
        m = metrics(model, data)
        if m["lossless"] > best_ll:
            best_ll = m["lossless"]
            best_alpha = float(a_val)
    _restore_alpha(model, saved)
    if best_alpha is not None:
        raw_val = float(np.log(np.exp(best_alpha) - 1.0) if best_alpha > 1e-6 else -5.0)
        _set_alpha(model, raw_val)
    return best_ll


def train_lbfgs(model: MergerSingleW, data: torch.Tensor, max_outer: int, patience: int = 20, tag: str = "lbfgs", print_every: int = 10) -> dict:
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
        if (outer + 1) % print_every == 0 or outer == 0:
            print(
                f"    [{tag} {outer+1:3d}] loss={loss_val:.6f} "
                f"ll={m['lossless']:6.2f}% bad={m['bad_pairs']:5d} stall={stall}",
                flush=True,
            )
        if m["lossless"] >= 100.0:
            print(f"    -> LOSSLESS @ outer {outer+1}", flush=True)
            return m
        if stall >= patience:
            print(f"    -> plateau (stall={stall})", flush=True)
            break
    model.load_state_dict(best_state)
    return metrics(model, data)


def train_adam(model: MergerSingleW, data: torch.Tensor, epochs: int, lr: float, tag: str = "adam", print_every: int = 200) -> dict:
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
        if (ep + 1) % print_every == 0 or ep == 0:
            print(
                f"    [{tag} {ep+1:4d}] loss={loss.item():.5f} "
                f"ll={m['lossless']:.2f}% bad={m['bad_pairs']:5d} pd={m['per_dim']:.2f}%",
                flush=True,
            )
        if m["lossless"] >= 100.0:
            return m
    model.load_state_dict(best_state)
    return metrics(model, data)


def run_one(in_dim: int, H: int, activation: str, codebook_name: str, data: torch.Tensor, args) -> dict:
    codebook = CODEBOOKS[codebook_name]
    print(f"\n{'=' * 78}", flush=True)
    print(f"CONFIG arch={args.arch} in_dim={in_dim} H={H} activation={activation} codebook={codebook_name}", flush=True)
    print("=" * 78, flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    if args.arch == "dual":
        model = MergerDualW(in_dim, H, activation, codebook).to(DEVICE)
    else:
        model = MergerSingleW(in_dim, H, activation, codebook).to(DEVICE)
    init_m = metrics(model, data)
    print(f"[init] ll={init_m['lossless']:.2f}% bad={init_m['bad_pairs']} pd={init_m['per_dim']:.2f}%", flush=True)

    print(f"[float warmup] Adam {args.float_epochs}ep lr={args.float_lr}", flush=True)
    float_m = train_adam(model, data, args.float_epochs, args.float_lr, tag="float")
    print(f"[float adam] ll={float_m['lossless']:.2f}% bad={float_m['bad_pairs']} pd={float_m['per_dim']:.2f}%", flush=True)

    if args.lbfgs_outer > 0:
        print(f"[float LBFGS] max_outer={args.lbfgs_outer} patience={args.lbfgs_patience}", flush=True)
        if model.use_codebook:
            model.use_codebook = False
            float_m = train_lbfgs(model, data, args.lbfgs_outer, patience=args.lbfgs_patience, tag="float-lbfgs")
            model.use_codebook = True
        else:
            float_m = train_lbfgs(model, data, args.lbfgs_outer, patience=args.lbfgs_patience, tag="float-lbfgs")
        print(f"[float final] ll={float_m['lossless']:.2f}% bad={float_m['bad_pairs']} pd={float_m['per_dim']:.2f}%", flush=True)

    if model.use_codebook:
        static_ll = static_alpha_search(model, data, steps=args.alpha_steps)
        print(f"[static alpha] best_ll={static_ll:.2f}% alpha={model.alpha.item():.5f}", flush=True)
        warm_m = metrics(model, data)
        print(f"[qat warm] ll={warm_m['lossless']:.2f}% bad={warm_m['bad_pairs']}", flush=True)
        print(f"[qat polish] Adam {args.qat_epochs}ep lr={args.qat_lr}", flush=True)
        qat_m = train_adam(model, data, args.qat_epochs, args.qat_lr, tag="qat")
        print(f"[qat adam] ll={qat_m['lossless']:.2f}% bad={qat_m['bad_pairs']} pd={qat_m['per_dim']:.2f}%", flush=True)
        if args.lbfgs_outer > 0:
            print(f"[qat LBFGS] max_outer={args.lbfgs_outer} patience={args.lbfgs_patience}", flush=True)
            qat_m = train_lbfgs(model, data, args.lbfgs_outer, patience=args.lbfgs_patience, tag="qat-lbfgs")
            print(f"[qat final] ll={qat_m['lossless']:.2f}% bad={qat_m['bad_pairs']} pd={qat_m['per_dim']:.2f}%", flush=True)
        final_m = qat_m
        static_ll_report = static_ll
    else:
        final_m = float_m
        static_ll_report = float_m["lossless"]

    elapsed = time.time() - t0
    return {
        "in_dim": in_dim,
        "hidden": H,
        "activation": activation,
        "codebook_name": codebook_name,
        "float_lossless": float_m["lossless"],
        "float_bad": float_m["bad_pairs"],
        "static_lossless": static_ll_report,
        "final_lossless": final_m["lossless"],
        "final_bad": final_m["bad_pairs"],
        "final_per_dim": final_m["per_dim"],
        "weights_count": (2 if args.arch == "dual" else 1) * in_dim * H,
        "time_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--arch", choices=["single", "dual"], default="single")
    parser.add_argument("--hiddens", default="64,81")
    parser.add_argument("--activations", default="c19,relu,tanh,identity")
    parser.add_argument("--codebooks", default="float,binary,ternary")
    parser.add_argument("--float-epochs", type=int, default=800)
    parser.add_argument("--float-lr", type=float, default=2e-3)
    parser.add_argument("--qat-epochs", type=int, default=400)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--alpha-steps", type=int, default=50)
    parser.add_argument("--lbfgs-outer", type=int, default=150)
    parser.add_argument("--lbfgs-patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="output/byte_pair_merger_widen_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hiddens = [int(s) for s in args.hiddens.split(",") if s.strip()]
    activations = [s.strip() for s in args.activations.split(",") if s.strip()]
    codebooks = [s.strip() for s in args.codebooks.split(",") if s.strip()]

    print("=" * 78, flush=True)
    print("L1 MERGER WIDEN SWEEP (single-W mirror tied)", flush=True)
    print("=" * 78, flush=True)
    print(f"in_dim={args.in_dim} hiddens={hiddens}", flush=True)
    print(f"activations={activations}", flush=True)
    print(f"codebooks={codebooks}", flush=True)
    print(f"float_epochs={args.float_epochs} qat_epochs={args.qat_epochs}", flush=True)
    print(f"seed={args.seed}", flush=True)

    data = load_byte_pairs()
    print(f"Loaded {data.shape[0]} byte-pair vectors, in_dim={data.shape[1]}", flush=True)

    results: list[dict] = []
    for H in hiddens:
        for act in activations:
            for cb in codebooks:
                res = run_one(args.in_dim, H, act, cb, data, args)
                results.append(res)

    ranked = sorted(
        results,
        key=lambda r: (-r["final_lossless"], r["final_bad"], -r["final_per_dim"], r["weights_count"]),
    )
    summary = {
        "in_dim": args.in_dim,
        "hiddens": hiddens,
        "activations": activations,
        "codebooks": codebooks,
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "seed": args.seed,
        "results": results,
        "ranked": ranked,
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print("RANKED RESULTS", flush=True)
    print("=" * 78, flush=True)
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:2d}. H={r['hidden']:3d} act={r['activation']:<8s} cb={r['codebook_name']:<12s} "
            f"float={r['float_lossless']:6.2f}% final={r['final_lossless']:6.2f}% "
            f"bad={r['final_bad']:5d} pd={r['final_per_dim']:6.2f}% t={r['time_s']:5.1f}s",
            flush=True,
        )
    print(f"\nSaved: {save}", flush=True)


if __name__ == "__main__":
    main()
