"""Single-W int8 LINEAR quantization pipeline.

W -> int8 linear (1 global alpha, 2592 int8 cells).
Starting error: ~0.003 max — proven tolerable (single-cell tweak needed 0.0003).
Retrain: alpha trainable, int8 values frozen, b1/b2/c19 trainable.

If plateau < 100%, per-cell rollback: some cells can be "bumped" to neighbor ints.
Final: exhaustive single-cell int perturbation (the same technique that closed
the float model's last bad pair).
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
    SingleWMirror, load_byte_pairs, metrics, DEVICE, C19Activation,
)

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
OUT_DIR = Path("output/merger_single_w_int8_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Int8LinearSingleW(nn.Module):
    """Single-W where W = alpha * ints, with trainable alpha and frozen ints."""
    def __init__(self, in_dim, H, alpha_init, ints_init, b1, b2, c19_c, c19_rho, device):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32, device=device))
        self.register_buffer("W_ints", torch.tensor(ints_init, dtype=torch.float32, device=device))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32, device=device))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32, device=device))
        self.c19 = C19Activation(H).to(device)
        with torch.no_grad():
            self.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=device))
            self.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=device))

    def build_W(self):
        return self.alpha * self.W_ints

    def forward(self, x):
        W = self.build_W()
        h = self.c19(x @ W + self.b1)
        return h @ W.t() + self.b2


def retrain(model, data, max_outer=200, stall=25, hinge=0.5, log_every=5):
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
            print(f"  [retrain {outer+1}] loss={loss:.6e} ll={ll:.6f}% bad={bp} stall={s} best={best_ll:.4f}%", flush=True)
        if ll >= 100.0:
            print(f"  -> LOSSLESS", flush=True)
            return True, best_state
        if s >= stall:
            print(f"  -> plateau, best={best_ll:.4f}%", flush=True)
            model.load_state_dict(best_state)
            return False, best_state
    model.load_state_dict(best_state)
    return False, best_state


def per_cell_int_tweak(model, data, max_rounds=3):
    """For each cell, try +1/-1 int perturbations. Accept if lossless improves."""
    print(f"\n=== Per-cell int tweak ===")
    ll, _, bp = metrics(model, data)
    print(f"Start: ll={ll:.6f}% bad={bp}")
    if bp == 0: return True

    ints_orig = model.W_ints.data.clone()
    n_cells = ints_orig.numel()
    total_accept = 0

    with torch.no_grad():
        for round_idx in range(max_rounds):
            any_improved = False
            best_ll = ll
            t0 = time.time()
            for i in range(ints_orig.shape[0]):
                for j in range(ints_orig.shape[1]):
                    orig = model.W_ints.data[i, j].item()
                    for delta in [-1, 1, -2, 2, -3, 3, -5, 5]:
                        new = orig + delta
                        if new < -127 or new > 127:
                            continue
                        model.W_ints.data[i, j] = new
                        ll_t, _, bp_t = metrics(model, data)
                        if ll_t > best_ll:
                            best_ll = ll_t
                            ll = ll_t
                            any_improved = True
                            total_accept += 1
                            print(f"  [round {round_idx+1}] W[{i},{j}] int {int(orig)} -> {int(new)}: ll={ll_t:.6f}% bad={bp_t}", flush=True)
                            orig = new  # update baseline for this cell
                            break
                    else:
                        # No improvement from any delta — restore
                        model.W_ints.data[i, j] = orig
                        continue
                    # We accepted a delta — keep the new value
                if (i + 1) % 8 == 0:
                    print(f"  round {round_idx+1} row {i+1}/{ints_orig.shape[0]}: ll={ll:.6f}% bad={bp_t} ({time.time()-t0:.0f}s)", flush=True)
            ll, _, bp = metrics(model, data)
            print(f"  Round {round_idx+1} done: ll={ll:.6f}% bad={bp}, accepts={total_accept}", flush=True)
            if bp == 0:
                return True
            if not any_improved:
                print(f"  No improvements this round — stop.")
                break
    return bp == 0


def bias_int8(arr, n_bits=8):
    """Symmetric int8 snap. Returns (ints, alpha)."""
    amax = np.abs(arr).max()
    max_int = 2**(n_bits-1) - 1
    alpha = amax / max_int
    ints = np.round(arr / alpha).clip(-max_int, max_int).astype(np.int32)
    return ints, alpha


def main():
    print(f"=== SINGLE-W INT8 LINEAR PIPELINE ===")
    print(f"Source: {CHAMPION}")

    with open(CHAMPION, "r") as f:
        m = json.load(f)

    W = np.array(m["W"], dtype=np.float32)
    b1 = np.array(m["b1"], dtype=np.float32)
    b2 = np.array(m["b2"], dtype=np.float32)
    c19_c = np.array(m["c19_c"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho"], dtype=np.float32)

    # ---- Float sanity ----
    model_f = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model_f.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model_f.b1.copy_(torch.tensor(b1, dtype=torch.float32, device=DEVICE))
        model_f.b2.copy_(torch.tensor(b2, dtype=torch.float32, device=DEVICE))
        model_f.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=DEVICE))
        model_f.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=DEVICE))
    data = load_byte_pairs().to(DEVICE)
    ll, pd, bp = metrics(model_f, data)
    print(f"Float sanity: ll={ll:.6f}% bad={bp}")

    # ---- Stage 1: int8 linear snap on W ----
    amax = np.abs(W).max()
    alpha0 = amax / 127.0
    W_ints = np.round(W / alpha0).clip(-127, 127).astype(np.int32)
    W_recon = W_ints * alpha0
    max_err = np.abs(W - W_recon).max()
    print(f"\n========== STAGE 1: W int8 snap ==========")
    print(f"  amax={amax:.6f}, alpha={alpha0:.8f}, max_err={max_err:.6f}")
    print(f"  unique ints used: {len(np.unique(W_ints))}")

    model_q = Int8LinearSingleW(32, 81, alpha0, W_ints, b1, b2, c19_c, c19_rho, DEVICE).to(DEVICE)
    ll, _, bp = metrics(model_q, data)
    print(f"  After snap: ll={ll:.6f}% bad={bp}")

    if ll < 100.0:
        print(f"\n  --- Retrain (alpha + b1 + b2 + c19 trainable, ints frozen) ---")
        t0 = time.time()
        lossless, state = retrain(model_q, data, max_outer=200, stall=25, hinge=0.5)
        ll, _, bp = metrics(model_q, data)
        print(f"  After retrain: ll={ll:.6f}% bad={bp} ({time.time()-t0:.0f}s)")

        if ll < 100.0:
            print(f"\n  --- Heavy retrain (hinge=5.0) ---")
            t0 = time.time()
            lossless, state = retrain(model_q, data, max_outer=200, stall=25, hinge=5.0)
            ll, _, bp = metrics(model_q, data)
            print(f"  After heavy retrain: ll={ll:.6f}% bad={bp} ({time.time()-t0:.0f}s)")

        if ll < 100.0 and bp <= 200:
            print(f"\n  --- Per-cell int tweak ---")
            t0 = time.time()
            lossless = per_cell_int_tweak(model_q, data, max_rounds=2)
            ll, _, bp = metrics(model_q, data)
            print(f"  After int tweak: ll={ll:.6f}% bad={bp} ({time.time()-t0:.0f}s)")

    final_W_ints = model_q.W_ints.data.cpu().numpy().astype(np.int32)
    final_alpha = model_q.alpha.data.cpu().numpy().item()
    final_b1 = model_q.b1.data.cpu().numpy()
    final_b2 = model_q.b2.data.cpu().numpy()
    final_c19_c = model_q.c19.c_raw.data.cpu().numpy()
    final_c19_rho = model_q.c19.rho_raw.data.cpu().numpy()

    # ---- Stage 2: b1, b2 int8 snap + retrain ----
    stage1_ll = ll
    stage1_bp = bp
    print(f"\n========== STAGE 2: b1, b2 int8 snap ==========")

    b1_ints, b1_alpha = bias_int8(final_b1)
    b2_ints, b2_alpha = bias_int8(final_b2)
    print(f"  b1: alpha={b1_alpha:.8f}, max_err={np.abs(final_b1 - b1_ints*b1_alpha).max():.6f}")
    print(f"  b2: alpha={b2_alpha:.8f}, max_err={np.abs(final_b2 - b2_ints*b2_alpha).max():.6f}")

    # Apply to model by setting b1, b2 to quantized values and making them frozen
    class Int8LinearFullQ(nn.Module):
        """W int8 + b1 int8 + b2 int8, only c19 trainable."""
        def __init__(self, W_ints, W_alpha, b1_ints, b1_alpha, b2_ints, b2_alpha,
                     c19_c, c19_rho, device):
            super().__init__()
            self.register_buffer("W_ints", torch.tensor(W_ints, dtype=torch.float32, device=device))
            self.W_alpha = nn.Parameter(torch.tensor(W_alpha, dtype=torch.float32, device=device))
            self.register_buffer("b1_ints", torch.tensor(b1_ints, dtype=torch.float32, device=device))
            self.b1_alpha = nn.Parameter(torch.tensor(b1_alpha, dtype=torch.float32, device=device))
            self.register_buffer("b2_ints", torch.tensor(b2_ints, dtype=torch.float32, device=device))
            self.b2_alpha = nn.Parameter(torch.tensor(b2_alpha, dtype=torch.float32, device=device))
            self.c19 = C19Activation(81).to(device)
            with torch.no_grad():
                self.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=device))
                self.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=device))

        def forward(self, x):
            W = self.W_alpha * self.W_ints
            b1 = self.b1_alpha * self.b1_ints
            b2 = self.b2_alpha * self.b2_ints
            h = self.c19(x @ W + b1)
            return h @ W.t() + b2

    model_full = Int8LinearFullQ(final_W_ints, final_alpha,
                                  b1_ints, b1_alpha, b2_ints, b2_alpha,
                                  final_c19_c, final_c19_rho, DEVICE).to(DEVICE)
    ll, _, bp = metrics(model_full, data)
    print(f"  After full snap: ll={ll:.6f}% bad={bp}")

    if ll < 100.0:
        print(f"  --- Retrain alphas + c19 ---")
        t0 = time.time()
        lossless, _ = retrain(model_full, data, max_outer=200, stall=25, hinge=5.0)
        ll, _, bp = metrics(model_full, data)
        print(f"  After retrain: ll={ll:.6f}% bad={bp} ({time.time()-t0:.0f}s)")

    # ---- Save artifact ----
    artifact = {
        "architecture": "single-W int8 linear (W + b1 + b2)",
        "H": 81, "in_dim": 32, "out_dim": 32,
        "stage1_lossless": stage1_ll,
        "stage1_bad": stage1_bp,
        "final_lossless": ll,
        "final_bad": bp,
        "W_alpha": model_full.W_alpha.item(),
        "W_ints": model_full.W_ints.data.cpu().numpy().astype(int).tolist(),
        "b1_alpha": model_full.b1_alpha.item(),
        "b1_ints": model_full.b1_ints.data.cpu().numpy().astype(int).tolist(),
        "b2_alpha": model_full.b2_alpha.item(),
        "b2_ints": model_full.b2_ints.data.cpu().numpy().astype(int).tolist(),
        "c19_c": model_full.c19.c_raw.data.cpu().numpy().tolist(),
        "c19_rho": model_full.c19.rho_raw.data.cpu().numpy().tolist(),
    }
    out_path = OUT_DIR / "final_int8_quant.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f)
    print(f"\nSaved: {out_path}")

    # ---- Size breakdown ----
    n_W = 2592
    n_b1 = 81
    n_b2 = 32
    n_c19 = 81 + 81
    total = (1 + n_W + 1 + n_b1 + 1 + n_b2) + n_c19 * 4
    print(f"\n=== DEPLOY BYTES ===")
    print(f"  W : alpha 4 B + ints {n_W} B = {4 + n_W} B")
    print(f"  b1: alpha 4 B + ints {n_b1} B = {4 + n_b1} B")
    print(f"  b2: alpha 4 B + ints {n_b2} B = {4 + n_b2} B")
    print(f"  c19_c + c19_rho: {n_c19} x 4 B = {n_c19 * 4} B")
    print(f"  TOTAL: {4 + n_W + 4 + n_b1 + 4 + n_b2 + n_c19 * 4} B ({(4 + n_W + 4 + n_b1 + 4 + n_b2 + n_c19 * 4)/1024:.2f} KB)")


if __name__ == "__main__":
    main()
