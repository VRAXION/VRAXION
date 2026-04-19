"""L1 Merger — SINGLE-W MIRROR TIED (one matrix only) baseline.

Architecture:
  forward(x) = C19(x @ W + b1) @ W.T + b2

Single weight matrix W (32 x H), bias b1 (H) + b2 (32), plus C19 params.
The encoder and decoder share W entirely (Wᵀ for the return leg).

This retries Cluster 10's single-W ceiling (73% lossless) with modern
techniques: multiple restarts, Adam + LBFGS, sign-aware loss, longer training.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LUT_IN_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


class C19Activation(nn.Module):
    def __init__(self, dim, c_init=1.0, rho_init=8.0):
        super().__init__()
        self.c_raw = nn.Parameter(torch.full((dim,), c_init))
        self.rho_raw = nn.Parameter(torch.full((dim,), rho_init))

    def forward(self, x):
        c = self.c_raw.clamp(min=0.1)
        rho = self.rho_raw.clamp(min=0.0)
        L = 6.0 * c
        scaled = x / c
        n = scaled.floor()
        t = scaled - n
        h = t * (1.0 - t)
        sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
        interior = c * (sgn * h + rho * h * h)
        return torch.where(
            x >= L, x - L, torch.where(x <= -L, x + L, interior)
        )


class SingleWMirror(nn.Module):
    """Forward: y = C19(x @ W + b1) @ W.T + b2.  Single W (in_dim x H)."""
    def __init__(self, in_dim, H, device, w_scale=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.W = nn.Parameter(torch.randn(in_dim, H, device=device) * w_scale)
        self.b1 = nn.Parameter(torch.zeros(H, device=device))
        self.b2 = nn.Parameter(torch.zeros(in_dim, device=device))
        self.c19 = C19Activation(H).to(device)

    def forward(self, x):
        h = self.c19(x @ self.W + self.b1)
        y = h @ self.W.t() + self.b2
        return y


def load_byte_pairs():
    with open(LUT_IN_PATH, "r") as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32) * scale
    idx_a = torch.arange(256).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1)


def metrics(model, data):
    with torch.no_grad():
        y = model(data)
        sign_match = torch.sign(y) == torch.sign(data)
        lossless = sign_match.all(dim=1).float().mean().item() * 100
        per_dim = sign_match.float().mean().item() * 100
        bad_pairs = (~sign_match.all(dim=1)).sum().item()
    return lossless, per_dim, bad_pairs


def train_adam(model, data, n_epochs, lr=1e-3, silent=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        opt.zero_grad()
        y = model(data)
        mse = ((y - data) ** 2).mean()
        sign_hinge = torch.relu(-y * torch.sign(data)).mean()
        loss = mse + 0.5 * sign_hinge
        loss.backward()
        opt.step()
        if not silent and (ep + 1) % 250 == 0:
            ll, pd, bp = metrics(model, data)
            print(f"    [adam ep {ep+1}] loss={loss.item():.4f} ll={ll:.2f}% bad={bp}", flush=True)


def train_lbfgs(model, data, max_outer=100, tol_stall=5):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=100,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")
    best_ll = 0.0
    stall = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            y = model(data)
            mse = ((y - data) ** 2).mean()
            sign_hinge = torch.relu(-y * torch.sign(data)).mean()
            loss = mse + 0.5 * sign_hinge
            loss.backward()
            return loss
        loss = opt.step(closure).item()
        ll, pd, bp = metrics(model, data)
        if ll > best_ll + 0.01:
            best_ll = ll
            stall = 0
        else:
            stall += 1
        if (outer + 1) % 10 == 0:
            print(f"    [lbfgs outer {outer+1}] loss={loss:.6f} ll={ll:.4f}% bad={bp} stall={stall}", flush=True)
        if ll >= 100.0:
            print(f"    -> LOSSLESS @ outer {outer+1}", flush=True)
            return True
        if stall >= tol_stall:
            print(f"    -> plateau (stall={stall})", flush=True)
            return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=81)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--adam-epochs", type=int, default=2000)
    parser.add_argument("--lbfgs-outer", type=int, default=150)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SINGLE-W MIRROR TIED (in=32, H={args.H}) ===", flush=True)
    print(f"  Single W: {32 * args.H} weight cells (half of asymm 2-matrix)", flush=True)

    data = load_byte_pairs().to(DEVICE)

    best_restart = None
    best_ll = 0.0
    t0 = time.time()

    for r in range(args.restarts):
        torch.manual_seed(1000 + r)
        np.random.seed(1000 + r)
        print(f"\n--- Restart {r+1}/{args.restarts} (seed={1000+r}) ---", flush=True)
        model = SingleWMirror(32, args.H, DEVICE).to(DEVICE)

        ll0, _, bp0 = metrics(model, data)
        print(f"  init: ll={ll0:.4f}% bad={bp0}", flush=True)

        print(f"  [Phase 1] Adam {args.adam_epochs} epochs", flush=True)
        train_adam(model, data, args.adam_epochs)
        ll, pd, bp = metrics(model, data)
        print(f"  After Adam: ll={ll:.4f}% bad={bp}", flush=True)

        print(f"  [Phase 2] LBFGS max {args.lbfgs_outer} outers", flush=True)
        lossless = train_lbfgs(model, data, max_outer=args.lbfgs_outer)

        ll, pd, bp = metrics(model, data)
        print(f"  Restart {r+1}: ll={ll:.4f}% pd={pd:.4f}% bad={bp}", flush=True)

        if ll > best_ll:
            best_ll = ll
            best_restart = r + 1
            # Save best model
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if ll >= 100.0:
            print(f"  >>> FOUND 100% LOSSLESS @ restart {r+1}, seed={1000+r} <<<", flush=True)
            break

    t_total = time.time() - t0
    print(f"\n=== DONE ===", flush=True)
    print(f"  Best restart: {best_restart}, lossless: {best_ll:.4f}%", flush=True)
    print(f"  Time: {t_total:.0f}s", flush=True)

    # If lossless found, save artifact
    final_state = {
        "architecture": "single-W mirror tied (SingleWMirror)",
        "H": args.H, "in_dim": 32, "out_dim": 32,
        "best_lossless": best_ll,
        "best_restart": best_restart,
        "time_s": t_total,
        "weights_count": 32 * args.H,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(final_state, f, indent=2)
    print(f"  Saved summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
