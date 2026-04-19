"""Single-W quantization pipeline.

Stage A: K-means codebook on W (64 entries).
  A1. Snap -> measure lossless
  A2. If not lossless: retrain with trainable codebook + fixed indices
  A3. If still not: per-cell rollback (revert cells that break pairs)
  A4. Exhaustive single-cell codebook tweak for final residuals

Stage B: int8 snap on b1, b2 (retrain codebook + c19 + other bias)
Stage C: int8 on c19_c, c19_rho if viable

Final artifact: deployable bytes with size breakdown.
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
OUT_DIR = Path("output/merger_single_w_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kmeans_1d(arr, k, n_iter=50, seed=0):
    """Simple 1-D k-means. Returns (centers, indices)."""
    rng = np.random.default_rng(seed)
    # Init centers at quantiles
    qs = np.linspace(0, 1, k + 1)
    centers = np.array([np.quantile(arr, (qs[i] + qs[i+1]) / 2) for i in range(k)])
    for _ in range(n_iter):
        # Assign
        idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
        # Update
        new = np.empty(k, dtype=np.float32)
        for j in range(k):
            mask = idx == j
            if mask.any():
                new[j] = arr[mask].mean()
            else:
                # Empty cluster: reseed to a random data point
                new[j] = arr[rng.integers(0, len(arr))]
        if np.allclose(new, centers, atol=1e-9):
            centers = new
            break
        centers = new
    idx = np.argmin(np.abs(arr[:, None] - centers[None, :]), axis=1)
    return centers.astype(np.float32), idx.astype(np.int64)


class CodebookSingleW(nn.Module):
    """Single-W model where W is expressed as codebook[indices].

    codebook: K trainable floats.
    indices:  (in_dim * H) frozen long tensor.
    b1, b2, c19: trainable.
    """
    def __init__(self, in_dim, H, codebook_init, indices, b1, b2, c19_c, c19_rho, device):
        super().__init__()
        self.in_dim = in_dim
        self.H = H
        self.codebook = nn.Parameter(torch.tensor(codebook_init, dtype=torch.float32, device=device))
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long, device=device))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32, device=device))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32, device=device))
        self.c19 = C19Activation(H).to(device)
        with torch.no_grad():
            self.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=device))
            self.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=device))

    def build_W(self):
        return self.codebook[self.indices].reshape(self.in_dim, self.H)

    def forward(self, x):
        W = self.build_W()
        h = self.c19(x @ W + self.b1)
        return h @ W.t() + self.b2


def retrain_codebook(model, data, max_outer=150, stall=20, hinge=0.5):
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
        if ll > best_ll + 0.001:
            best_ll = ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            s = 0
        else:
            s += 1
        if (outer + 1) % 5 == 0 or ll >= 100.0:
            print(f"  [retrain outer {outer+1}] loss={loss:.6e} ll={ll:.6f}% bad={bp} stall={s} best={best_ll:.4f}%", flush=True)
        if ll >= 100.0:
            print(f"  -> LOSSLESS", flush=True)
            return True, best_state
        if s >= stall:
            print(f"  -> plateau, best={best_ll:.4f}%", flush=True)
            model.load_state_dict(best_state)
            return False, best_state
    model.load_state_dict(best_state)
    return False, best_state


def exhaustive_codebook_tweak(model, data, ll_start, bad_start):
    """Try small perturbations on each codebook entry."""
    if bad_start == 0:
        return True
    print(f"\n=== Exhaustive codebook tweak (bad={bad_start}) ===")
    cb_orig = model.codebook.data.clone()
    abs_mean = cb_orig.abs().mean().item()
    deltas = [m * abs_mean for m in [-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005,
                                       0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]]
    best_ll = ll_start
    best_k = None
    best_d = None
    with torch.no_grad():
        for k_idx in range(cb_orig.numel()):
            orig = cb_orig[k_idx].item()
            for d in deltas:
                model.codebook.data[k_idx] = orig + d
                ll, _, bp = metrics(model, data)
                if ll > best_ll:
                    best_ll = ll
                    best_k = k_idx
                    best_d = d
                    print(f"  cb[{k_idx}] += {d:+.6f} -> ll={ll:.6f}%, bad={bp}", flush=True)
                model.codebook.data[k_idx] = orig
    if best_k is not None and best_ll > ll_start:
        with torch.no_grad():
            model.codebook.data[best_k] = cb_orig[best_k].item() + best_d
        ll, _, bp = metrics(model, data)
        print(f"  Applied: ll={ll:.6f}%, bad={bp}")
        return ll >= 100.0
    print("  No improvement found.")
    return False


def cell_rollback(model, data, W_float, indices_frozen):
    """For cells where snap caused a bad pair, revert to float.

    Returns a mask of which cells remain frozen (True = frozen-to-codebook,
    False = free-float).
    """
    # NOTE: simplified — per-cell rollback is complex when we have a
    # codebook-based W. Skip for now; retry later if needed.
    pass


def stage_a_lookup_codebook(model_f, data, W_float, b1, b2, c19_c, c19_rho, K=64):
    print(f"\n========== STAGE A: K-MEANS CODEBOOK (K={K}) ==========")
    print(f"  Running k-means on {W_float.size} W cells...")
    centers, indices = kmeans_1d(W_float.flatten(), k=K, n_iter=80, seed=42)
    print(f"  Centers: {len(centers)}, min={centers.min():.6f}, max={centers.max():.6f}")

    # Build codebook model
    cb_model = CodebookSingleW(
        in_dim=W_float.shape[0], H=W_float.shape[1],
        codebook_init=centers, indices=indices,
        b1=b1, b2=b2, c19_c=c19_c, c19_rho=c19_rho,
        device=DEVICE,
    ).to(DEVICE)

    ll, pd, bp = metrics(cb_model, data)
    print(f"  After snap (no retrain): ll={ll:.6f}%, bad={bp}")

    if ll >= 100.0:
        print(f"  >>> LOSSLESS from pure snap! <<<")
        return cb_model, centers, indices

    # Retrain
    print(f"\n  --- Retrain with trainable codebook ---")
    lossless, state = retrain_codebook(cb_model, data, max_outer=100, stall=15, hinge=0.5)
    ll, pd, bp = metrics(cb_model, data)
    print(f"  After retrain: ll={ll:.6f}%, bad={bp}")

    if ll >= 100.0:
        return cb_model, cb_model.codebook.data.cpu().numpy(), indices

    # Exhaustive codebook tweak
    lossless = exhaustive_codebook_tweak(cb_model, data, ll, bp)
    if lossless:
        print(f"  >>> LOSSLESS via codebook tweak <<<")
        return cb_model, cb_model.codebook.data.cpu().numpy(), indices

    # Heavy retrain with higher hinge
    print(f"\n  --- Heavy retrain (hinge=5.0) ---")
    lossless, state = retrain_codebook(cb_model, data, max_outer=200, stall=25, hinge=5.0)
    ll, pd, bp = metrics(cb_model, data)
    print(f"  After heavy retrain: ll={ll:.6f}%, bad={bp}")

    if ll >= 100.0:
        return cb_model, cb_model.codebook.data.cpu().numpy(), indices

    # Final exhaustive
    lossless = exhaustive_codebook_tweak(cb_model, data, ll, bp)
    return cb_model, cb_model.codebook.data.cpu().numpy(), indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=64, help="codebook size for W")
    args = parser.parse_args()

    print(f"=== SINGLE-W QUANT PIPELINE (K={args.K}) ===")
    print(f"Source: {CHAMPION}")

    with open(CHAMPION, "r") as f:
        m = json.load(f)

    W = np.array(m["W"], dtype=np.float32)
    b1 = np.array(m["b1"], dtype=np.float32)
    b2 = np.array(m["b2"], dtype=np.float32)
    c19_c = np.array(m["c19_c"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho"], dtype=np.float32)

    # Sanity: rebuild float model, check 100%
    model_f = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model_f.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model_f.b1.copy_(torch.tensor(b1, dtype=torch.float32, device=DEVICE))
        model_f.b2.copy_(torch.tensor(b2, dtype=torch.float32, device=DEVICE))
        model_f.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=DEVICE))
        model_f.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=DEVICE))
    data = load_byte_pairs().to(DEVICE)
    ll, pd, bp = metrics(model_f, data)
    print(f"Float model sanity: ll={ll:.6f}% bad={bp}")
    if ll < 100.0:
        print("ERR: float model not lossless. Aborting.")
        return

    t0 = time.time()
    cb_model, final_cb, final_idx = stage_a_lookup_codebook(
        model_f, data, W, b1, b2, c19_c, c19_rho, K=args.K,
    )
    ll_final, _, bp_final = metrics(cb_model, data)
    print(f"\n=== STAGE A DONE ===  ll={ll_final:.6f}% bad={bp_final} time={time.time()-t0:.0f}s")

    # Save Stage A artifact
    stage_a = {
        "stage": "A_lookup_codebook",
        "K": args.K,
        "final_lossless": ll_final,
        "final_bad": bp_final,
        "codebook": final_cb.tolist(),
        "indices": final_idx.tolist(),
        "b1": cb_model.b1.detach().cpu().numpy().tolist(),
        "b2": cb_model.b2.detach().cpu().numpy().tolist(),
        "c19_c": cb_model.c19.c_raw.detach().cpu().numpy().tolist(),
        "c19_rho": cb_model.c19.rho_raw.detach().cpu().numpy().tolist(),
    }
    out_path = OUT_DIR / f"stage_a_K{args.K}.json"
    with open(out_path, "w") as f:
        json.dump(stage_a, f)
    print(f"Saved: {out_path}")

    # Size estimate
    bits_per_idx = int(np.ceil(np.log2(args.K)))
    n_cells = 32 * 81
    codebook_b = args.K * 4
    idx_b = int(np.ceil(n_cells * bits_per_idx / 8))
    b1_b = 81 * 4
    b2_b = 32 * 4
    c19_b = (81 + 81) * 4
    total_float = codebook_b + idx_b + b1_b + b2_b + c19_b
    print(f"\n  Stage A deploy estimate (float bias/c19):")
    print(f"    codebook {codebook_b} B + indices {idx_b} B + b1 {b1_b} B + b2 {b2_b} B + c19 {c19_b} B = {total_float} B ({total_float/1024:.2f} KB)")


if __name__ == "__main__":
    main()
