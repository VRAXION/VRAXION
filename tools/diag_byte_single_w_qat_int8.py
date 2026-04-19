"""QAT (Quantization-Aware Training) with int8 STE for single-W mirror.

Fine-tune the existing float champion to become int8-compatible:
  Forward:  W_q = alpha * round_ste(W / alpha)  [clipped to int8]
  Backward: gradient flows through (straight-through)

  b1, b2, c19 stay float (can also be quantized later).

Strategy:
  1. Warm start from the existing 100% float champion.
  2. Train with STE-quantized W_q in forward, loss = MSE + sign-hinge.
  3. Multi-phase: Adam (low bit-width might need longer warmup) + LBFGS finish.
  4. At end: snap W to int8 and verify lossless.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import (
    load_byte_pairs, metrics, DEVICE, C19Activation,
)

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def ste_round(x):
    return STERound.apply(x)


class QATSingleW(nn.Module):
    def __init__(self, in_dim, H, device, n_bits=8):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.n_bits = n_bits
        self.max_int = 2**(n_bits - 1) - 1  # e.g., 127 for int8
        self.W = nn.Parameter(torch.zeros(in_dim, H, device=device))
        self.alpha = nn.Parameter(torch.tensor(1.0, device=device))
        self.b1 = nn.Parameter(torch.zeros(H, device=device))
        self.b2 = nn.Parameter(torch.zeros(in_dim, device=device))
        self.c19 = C19Activation(H).to(device)

    def quant_W(self):
        # W_q = alpha * clip(round(W/alpha), -max_int, max_int)
        scaled = self.W / self.alpha
        rounded = ste_round(scaled)
        clipped = torch.clamp(rounded, -self.max_int, self.max_int)
        return self.alpha * clipped

    def forward(self, x):
        W = self.quant_W()
        h = self.c19(x @ W + self.b1)
        return h @ W.t() + self.b2


def load_from_champion(model, champion_path):
    with open(champion_path, "r") as f:
        m = json.load(f)
    W = np.array(m["W"], dtype=np.float32)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(m["b1"], dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(m["b2"], dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(m["c19_c"], dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(m["c19_rho"], dtype=torch.float32, device=DEVICE))
        # Alpha init: amax / max_int
        amax = np.abs(W).max()
        model.alpha.copy_(torch.tensor(amax / model.max_int, dtype=torch.float32, device=DEVICE))


def train_adam(model, data, n_epochs=2000, lr=1e-3, hinge=0.5, log_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_ll = 0.0
    best_state = None
    for ep in range(n_epochs):
        opt.zero_grad()
        y = model(data)
        mse = ((y - data) ** 2).mean()
        sign_hinge = torch.relu(-y * torch.sign(data)).mean()
        loss = mse + hinge * sign_hinge
        loss.backward()
        opt.step()
        if (ep + 1) % log_every == 0:
            ll, pd, bp = metrics(model, data)
            if ll > best_ll:
                best_ll = ll
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [adam {ep+1}] loss={loss.item():.4e} ll={ll:.4f}% bad={bp} best={best_ll:.4f}%", flush=True)
    return best_state or {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_lbfgs(model, data, max_outer=200, stall=25, hinge=1.0, log_every=5):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=100,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=100, line_search_fn="strong_wolfe")
    best_ll = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    s = 0
    for outer in range(max_outer):
        def closure():
            opt.zero_grad()
            y = model(data)
            mse = ((y - data) ** 2).mean()
            sign_hinge = torch.relu(-y * torch.sign(data)).mean()
            loss = mse + hinge * sign_hinge
            loss.backward()
            return loss
        loss = opt.step(closure).item()
        ll, pd, bp = metrics(model, data)
        if ll > best_ll + 1e-4:
            best_ll = ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            s = 0
        else:
            s += 1
        if (outer + 1) % log_every == 0 or ll >= 100.0:
            print(f"  [lbfgs {outer+1}] loss={loss:.4e} ll={ll:.4f}% bad={bp} stall={s} best={best_ll:.4f}%", flush=True)
        if ll >= 100.0:
            return best_state
        if s >= stall:
            model.load_state_dict(best_state)
            return best_state
    model.load_state_dict(best_state)
    return best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--out", default="output/merger_single_w_qat_int8")
    parser.add_argument("--adam-epochs", type=int, default=3000)
    parser.add_argument("--lbfgs-outer", type=int, default=200)
    parser.add_argument("--lbfgs-hinge", type=float, default=1.0)
    parser.add_argument("--warm-start", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== QAT INT{args.bits} STE TRAINING ===", flush=True)
    print(f"Target: W quantized to int{args.bits} via STE, bias/c19 float.", flush=True)

    data = load_byte_pairs().to(DEVICE)

    model = QATSingleW(32, 81, DEVICE, n_bits=args.bits).to(DEVICE)
    if args.warm_start:
        print(f"Warm start from: {CHAMPION}", flush=True)
        load_from_champion(model, CHAMPION)

    ll, _, bp = metrics(model, data)
    print(f"Initial (quantized): ll={ll:.4f}% bad={bp}", flush=True)
    print(f"Initial alpha: {model.alpha.item():.6f}", flush=True)

    t0 = time.time()

    # Phase 1: Adam warmup
    print(f"\n--- Phase 1: Adam {args.adam_epochs} epochs ---", flush=True)
    best_state = train_adam(model, data, n_epochs=args.adam_epochs, lr=5e-4, hinge=0.5, log_every=100)
    model.load_state_dict(best_state)
    ll, _, bp = metrics(model, data)
    print(f"After Adam: ll={ll:.4f}% bad={bp} alpha={model.alpha.item():.6f}", flush=True)

    # Phase 2: LBFGS finish (moderate hinge)
    print(f"\n--- Phase 2: LBFGS hinge={args.lbfgs_hinge}, max {args.lbfgs_outer} outers ---", flush=True)
    best_state = train_lbfgs(model, data, max_outer=args.lbfgs_outer, stall=25, hinge=args.lbfgs_hinge)
    model.load_state_dict(best_state)
    ll, _, bp = metrics(model, data)
    print(f"After LBFGS: ll={ll:.4f}% bad={bp}", flush=True)

    # Phase 3: Heavy hinge push if not there yet
    if ll < 100.0:
        print(f"\n--- Phase 3: Heavy hinge=5.0, stall=30 ---", flush=True)
        best_state = train_lbfgs(model, data, max_outer=300, stall=30, hinge=5.0)
        model.load_state_dict(best_state)
        ll, _, bp = metrics(model, data)
        print(f"After heavy LBFGS: ll={ll:.4f}% bad={bp}", flush=True)

    # Final: extract int representation
    with torch.no_grad():
        W_int = torch.round(model.W / model.alpha).clamp(-model.max_int, model.max_int)
        # Reconstruct using the true int values (identical to quant_W in this case)
        W_q = (model.alpha * W_int).cpu().numpy()

    # Verify
    model_eval = QATSingleW(32, 81, DEVICE, n_bits=args.bits).to(DEVICE)
    model_eval.load_state_dict(model.state_dict())
    ll_f, _, bp_f = metrics(model_eval, data)
    print(f"\nFinal: ll={ll_f:.6f}% bad={bp_f} time={time.time()-t0:.0f}s", flush=True)

    # Save
    artifact = {
        "architecture": f"QAT int{args.bits} single-W",
        "H": 81, "in_dim": 32,
        "bits": args.bits,
        "final_lossless": ll_f,
        "final_bad": bp_f,
        "W_int": W_int.cpu().numpy().astype(int).tolist(),
        "W_alpha": model.alpha.item(),
        "b1": model.b1.data.cpu().numpy().tolist(),
        "b2": model.b2.data.cpu().numpy().tolist(),
        "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
        "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
    }
    path = out_dir / f"qat_int{args.bits}.json"
    with open(path, "w") as f:
        json.dump(artifact, f)
    print(f"Saved: {path}", flush=True)

    # Size estimate (bias/c19 as fp16 for minimal deploy)
    bit_per_w = args.bits
    n_W = 2592
    W_bytes = int(np.ceil(n_W * bit_per_w / 8)) + 4  # alpha
    bias_bytes = (81 + 32) * 2 + 4  # fp16 bias + alpha? just fp16 for now
    c19_bytes = (81 + 81) * 2
    total = W_bytes + bias_bytes + c19_bytes
    print(f"\nDeploy estimate (W int{args.bits}, bias/c19 fp16):")
    print(f"  W: {W_bytes} B")
    print(f"  bias (fp16): {bias_bytes} B")
    print(f"  c19 (fp16): {c19_bytes} B")
    print(f"  Total: {total} B ({total/1024:.2f} KB)")


if __name__ == "__main__":
    main()
