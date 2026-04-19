"""Hybrid W quantization: most cells int8, a few escape to float.

Strategy:
  1. Snap all cells int8 (loss: 91.9%).
  2. Greedy: find cells whose 'escape' (revert to original float) gives
     the biggest lossless improvement.
  3. Accept top K escapes, retrain (alpha + escape-floats + bias + c19).
  4. Iterate until lossless OR escape budget exhausted.

Deploy:
  - W: 1 alpha (4 B) + 2592 int8 (2592 B) = 2596 B
  - + escape: K floats (4K B) + K indices (2K B assuming 16-bit) = 6K B
  - b1, b2: int8 with per-vector alpha
  - c19: float (162 * 4 = 648 B)
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
OUT_DIR = Path("output/merger_single_w_hybrid_escape")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class HybridSingleW(nn.Module):
    """W = alpha * W_ints * (1 - escape_mask) + W_escape * escape_mask

    - W_ints: frozen int representation (as float)
    - W_alpha: trainable scalar
    - W_escape: trainable per-cell float (only effective where mask=1)
    - escape_mask: buffer, 0 or 1
    - b1, b2, c19: trainable
    """
    def __init__(self, in_dim, H, W_ints, alpha, escape_mask, W_escape,
                 b1, b2, c19_c, c19_rho, device):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.register_buffer("W_ints", torch.tensor(W_ints, dtype=torch.float32, device=device))
        self.W_alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, device=device))
        self.register_buffer("escape_mask", torch.tensor(escape_mask, dtype=torch.float32, device=device))
        self.W_escape = nn.Parameter(torch.tensor(W_escape, dtype=torch.float32, device=device))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32, device=device))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32, device=device))
        self.c19 = C19Activation(H).to(device)
        with torch.no_grad():
            self.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=device))
            self.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=device))

    def build_W(self):
        int_part = self.W_alpha * self.W_ints
        return (1.0 - self.escape_mask) * int_part + self.escape_mask * self.W_escape

    def forward(self, x):
        W = self.build_W()
        h = self.c19(x @ W + self.b1)
        return h @ W.t() + self.b2


def retrain(model, data, max_outer=100, stall=15, hinge=0.5, log_every=5, tag=""):
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
            print(f"    [{tag} {outer+1}] loss={loss:.4e} ll={ll:.6f}% bad={bp} stall={s} best={best_ll:.4f}%", flush=True)
        if ll >= 100.0:
            return True, best_state
        if s >= stall:
            model.load_state_dict(best_state)
            return False, best_state
    model.load_state_dict(best_state)
    return False, best_state


def rank_cells_by_escape_benefit(model, W_orig, data):
    """For each non-escaped cell, measure ll improvement if we unsnap it.

    Returns list of ((i,j), ll_improvement) sorted descending.
    """
    ll0, _, bp0 = metrics(model, data)
    in_dim, H = 32, 81
    benefits = []
    # Compute current full W
    with torch.no_grad():
        alpha = model.W_alpha.item()
        # For speed: just swap W_ints temporarily. When a cell is already escaped
        # (escape_mask=1), we don't rank it.
        cur_mask = model.escape_mask.data.cpu().numpy()
        cur_escape_vals = model.W_escape.data.cpu().numpy()
        for i in range(in_dim):
            for j in range(H):
                if cur_mask[i, j] == 1.0:
                    continue
                # Save state
                old_escape_mask = model.escape_mask.data[i, j].item()
                old_escape_val = model.W_escape.data[i, j].item()
                # Set: escape=1, escape_val = original float
                model.escape_mask.data[i, j] = 1.0
                model.W_escape.data[i, j] = float(W_orig[i, j])
                ll, _, bp = metrics(model, data)
                benefits.append(((i, j), ll - ll0))
                # Restore
                model.escape_mask.data[i, j] = old_escape_mask
                model.W_escape.data[i, j] = old_escape_val
    benefits.sort(key=lambda x: -x[1])
    return benefits, ll0, bp0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-escapes", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=20,
                        help="cells to escape per iteration")
    parser.add_argument("--retrain-stall", type=int, default=12)
    args = parser.parse_args()

    print(f"=== HYBRID INT8 + ESCAPE PIPELINE ===", flush=True)
    print(f"Source: {CHAMPION}")

    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W_orig = np.array(m["W"], dtype=np.float32)
    b1 = np.array(m["b1"], dtype=np.float32)
    b2 = np.array(m["b2"], dtype=np.float32)
    c19_c = np.array(m["c19_c"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho"], dtype=np.float32)

    data = load_byte_pairs().to(DEVICE)

    # Initial int8 snap
    amax = np.abs(W_orig).max()
    alpha = amax / 127.0
    W_ints = np.round(W_orig / alpha).clip(-127, 127).astype(np.float32)
    escape_mask = np.zeros_like(W_orig, dtype=np.float32)
    W_escape = W_orig.copy()

    model = HybridSingleW(32, 81, W_ints, alpha, escape_mask, W_escape,
                           b1, b2, c19_c, c19_rho, DEVICE).to(DEVICE)
    ll, _, bp = metrics(model, data)
    print(f"Initial int8 snap: ll={ll:.6f}%, bad={bp}", flush=True)

    # Quick retrain (alpha + b1/b2/c19) — no escapes yet
    print(f"\n--- Initial retrain (alpha + bias + c19) ---", flush=True)
    # Mask out W_escape from training initially (escape_mask is all 0, so its grad
    # goes to masked-out positions; not critical but clean).
    retrain(model, data, max_outer=50, stall=10, hinge=0.5, tag="init")
    ll, _, bp = metrics(model, data)
    print(f"After initial retrain: ll={ll:.6f}%, bad={bp}", flush=True)

    # Iterative escape + retrain
    iteration = 0
    total_escapes = 0
    while total_escapes < args.max_escapes and ll < 100.0:
        iteration += 1
        print(f"\n========== ITERATION {iteration} ==========", flush=True)
        print(f"  Current: ll={ll:.6f}% bad={bp}, escapes used={total_escapes}/{args.max_escapes}", flush=True)

        t0 = time.time()
        print(f"  Ranking cells by escape benefit...", flush=True)
        benefits, _, _ = rank_cells_by_escape_benefit(model, W_orig, data)
        # Filter to positive improvements
        pos = [b for b in benefits if b[1] > 0]
        print(f"  {len(pos)} cells have >0 escape benefit (top 5: {[(b[0], f'{b[1]:.4f}%') for b in pos[:5]]})  ({time.time()-t0:.0f}s)", flush=True)

        if len(pos) == 0:
            print(f"  No positive improvements — stop escaping.", flush=True)
            break

        # Accept top batch_size
        n_accept = min(args.batch_size, len(pos), args.max_escapes - total_escapes)
        with torch.no_grad():
            for k in range(n_accept):
                (i, j), imp = pos[k]
                model.escape_mask.data[i, j] = 1.0
                model.W_escape.data[i, j] = float(W_orig[i, j])
        total_escapes += n_accept

        ll_before, _, bp_before = metrics(model, data)
        print(f"  Escaped {n_accept} cells -> ll={ll_before:.6f}% bad={bp_before}", flush=True)

        # Retrain
        print(f"  Retraining...", flush=True)
        t0 = time.time()
        lossless, _ = retrain(model, data, max_outer=80, stall=args.retrain_stall, hinge=0.5, tag=f"it{iteration}")
        ll, _, bp = metrics(model, data)
        print(f"  After retrain: ll={ll:.6f}% bad={bp} ({time.time()-t0:.0f}s)", flush=True)

        if lossless:
            print(f"  >>> LOSSLESS <<<", flush=True)
            break

    # Final heavy retrain
    if ll < 100.0:
        print(f"\n--- Heavy retrain (hinge=5.0) ---", flush=True)
        retrain(model, data, max_outer=150, stall=20, hinge=5.0, tag="heavy")
        ll, _, bp = metrics(model, data)
        print(f"After heavy retrain: ll={ll:.6f}% bad={bp}", flush=True)

    # Collect
    final_mask = model.escape_mask.data.cpu().numpy()
    final_ints = model.W_ints.data.cpu().numpy().astype(int)
    final_alpha = model.W_alpha.item()
    final_escape = model.W_escape.data.cpu().numpy()
    final_b1 = model.b1.data.cpu().numpy()
    final_b2 = model.b2.data.cpu().numpy()
    final_c19_c = model.c19.c_raw.data.cpu().numpy()
    final_c19_rho = model.c19.rho_raw.data.cpu().numpy()

    n_escape = int(final_mask.sum())
    print(f"\n=== DONE ===", flush=True)
    print(f"  Final lossless: {ll:.6f}% bad={bp}", flush=True)
    print(f"  Escape cells: {n_escape} / 2592 ({100*n_escape/2592:.1f}%)", flush=True)

    artifact = {
        "architecture": "single-W hybrid int8+escape",
        "H": 81, "in_dim": 32, "out_dim": 32,
        "final_lossless": ll, "final_bad": bp,
        "n_escape": n_escape,
        "W_alpha": final_alpha,
        "W_ints": final_ints.tolist(),
        "W_escape": final_escape.tolist(),
        "escape_mask": final_mask.astype(int).tolist(),
        "b1": final_b1.tolist(),
        "b2": final_b2.tolist(),
        "c19_c": final_c19_c.tolist(),
        "c19_rho": final_c19_rho.tolist(),
    }
    out_path = OUT_DIR / "final_hybrid.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f)
    print(f"Saved: {out_path}", flush=True)

    # Deploy size
    n_W = 2592
    # Escape cells: index (2 bytes: enough for 2592) + float value (4 bytes) = 6 B each
    escape_b = n_escape * 6
    int_b = 4 + n_W  # alpha + ints
    b1_int = 4 + 81
    b2_int = 4 + 32
    c19_b = (81 + 81) * 4
    total = int_b + escape_b + b1_int + b2_int + c19_b
    print(f"\n=== DEPLOY BYTES ===", flush=True)
    print(f"  W ints (alpha+{n_W}): {int_b} B", flush=True)
    print(f"  Escape ({n_escape} cells x 6B): {escape_b} B", flush=True)
    print(f"  b1 (int8 + alpha): {b1_int} B", flush=True)
    print(f"  b2 (int8 + alpha): {b2_int} B", flush=True)
    print(f"  c19 float (162 x 4): {c19_b} B", flush=True)
    print(f"  TOTAL: {total} B ({total/1024:.2f} KB)", flush=True)


if __name__ == "__main__":
    main()
