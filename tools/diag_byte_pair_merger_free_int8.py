"""L1 Merger — FREE CELL INT8 freeze (cell-by-cell).

After lookup codebook gravity run, ~900 cells remain float.
This pass quantizes each into int8 × alpha_free (per-matrix alpha),
one cell at a time with lossless check and periodic retrain.

Protocol:
  1. Load lookup codebook result (two-level state: frozen via lookup, free as float).
  2. Initialize alpha_free_W1, alpha_free_W2 from max|free value| / 127.
  3. For each free cell (smallest-|w| first):
     a. Compute nearest int: round(w / alpha_free).
     b. Try candidates: nearest, nearest±1, nearest±2, nearest±3 (ordered by |delta|).
     c. Accept first that maintains lossless ≥ 99.99%.
     d. Skip if no candidate works.
  4. Retrain every 50 accepts: codebook + alpha_free + bias + C19.
  5. Per-50 cell checkpoint.
  6. Final retrain + save.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LUT_IN_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")
LOSSLESS_FLOOR = 99.99
RETRAIN_MAX_ITER = 50
RETRAIN_PATIENCE = 15
RETRAIN_EVERY = 50  # retrain after N accepts
CAND_WIDTH = 3  # try nearest, ±1, ±2, ±3


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
        sgn = torch.where(
            n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n)
        )
        interior = c * (sgn * h + rho * h * h)
        return torch.where(
            x >= L, x - L, torch.where(x <= -L, x + L, interior)
        )


class FreeInt8Merger(nn.Module):
    """Three-layer weight: lookup-frozen | int8-frozen | float."""

    def __init__(self, state, device):
        super().__init__()
        self.in_dim = state["in_dim"]
        self.out_dim = state["out_dim"]
        self.H = state["H"]

        # Lookup codebook (from previous run, trainable)
        self.codebook_W1 = nn.Parameter(
            torch.tensor(state["codebook_W1"], dtype=torch.float32).to(device)
        )
        self.codebook_W2 = nn.Parameter(
            torch.tensor(state["codebook_W2"], dtype=torch.float32).to(device)
        )

        # Lookup indices (buffer)
        self.register_buffer("W1_idx", torch.tensor(state["W1_idx"], dtype=torch.long).to(device))
        self.register_buffer("W2_idx", torch.tensor(state["W2_idx"], dtype=torch.long).to(device))

        # Lookup-frozen masks (from previous run)
        self.register_buffer("W1_frozen_mask", torch.tensor(state["W1_frozen_mask"], dtype=torch.bool).to(device))
        self.register_buffer("W2_frozen_mask", torch.tensor(state["W2_frozen_mask"], dtype=torch.bool).to(device))

        # NEW: int8-frozen masks (start: all False)
        shape1 = self.W1_frozen_mask.shape
        shape2 = self.W2_frozen_mask.shape
        self.register_buffer("W1_int8_mask", torch.zeros(shape1, dtype=torch.bool, device=device))
        self.register_buffer("W2_int8_mask", torch.zeros(shape2, dtype=torch.bool, device=device))

        # NEW: int8 values (one per cell, default 0)
        self.register_buffer("W1_int8", torch.zeros(shape1, dtype=torch.float32, device=device))
        self.register_buffer("W2_int8", torch.zeros(shape2, dtype=torch.float32, device=device))

        # Float storage for still-free cells (also trainable)
        self.W1_float = nn.Parameter(
            torch.tensor(state["W1_float"], dtype=torch.float32).to(device)
        )
        self.W2_float = nn.Parameter(
            torch.tensor(state["W2_float"], dtype=torch.float32).to(device)
        )

        # Compute initial alpha_free from free cells
        with torch.no_grad():
            free1 = self.W1_float[~self.W1_frozen_mask]
            free2 = self.W2_float[~self.W2_frozen_mask]
            a_f1 = max(free1.abs().max().item() / 127.0, 1e-6)
            a_f2 = max(free2.abs().max().item() / 127.0, 1e-6)
        self.alpha_free_W1 = nn.Parameter(torch.tensor(a_f1, dtype=torch.float32).to(device))
        self.alpha_free_W2 = nn.Parameter(torch.tensor(a_f2, dtype=torch.float32).to(device))

        # Bias + C19
        self.b1 = nn.Parameter(torch.tensor(state["b1"], dtype=torch.float32).to(device))
        self.b2 = nn.Parameter(torch.tensor(state["b2"], dtype=torch.float32).to(device))
        self.db1 = nn.Parameter(torch.tensor(state["db1"], dtype=torch.float32).to(device))
        self.db2 = nn.Parameter(torch.tensor(state["db2"], dtype=torch.float32).to(device))
        self.c19 = C19Activation(self.H)
        with torch.no_grad():
            self.c19.c_raw.copy_(torch.tensor(state["c19_c"], dtype=torch.float32).to(device))
            self.c19.rho_raw.copy_(torch.tensor(state["c19_rho"], dtype=torch.float32).to(device))

    def W1_eff(self):
        looked = self.codebook_W1[self.W1_idx]
        int8_eff = self.alpha_free_W1 * self.W1_int8
        out = torch.where(self.W1_int8_mask, int8_eff, self.W1_float)
        out = torch.where(self.W1_frozen_mask, looked, out)
        return out

    def W2_eff(self):
        looked = self.codebook_W2[self.W2_idx]
        int8_eff = self.alpha_free_W2 * self.W2_int8
        out = torch.where(self.W2_int8_mask, int8_eff, self.W2_float)
        out = torch.where(self.W2_frozen_mask, looked, out)
        return out

    def encode(self, x):
        return self.c19(x @ self.W1_eff() + self.b1) @ self.W2_eff() + self.b2

    def decode(self, z):
        return (z @ self.W2_eff().t() + self.db1) @ self.W1_eff().t() + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def still_free_count(self):
        free_w1 = (~self.W1_frozen_mask & ~self.W1_int8_mask).sum().item()
        free_w2 = (~self.W2_frozen_mask & ~self.W2_int8_mask).sum().item()
        return free_w1, free_w2

    def total_cells(self):
        return self.W1_float.numel() + self.W2_float.numel()


def load_source_json(path):
    with open(path, "r") as f:
        return json.load(f)


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
        x_hat, _ = model(data)
        sign_match = torch.sign(x_hat) == torch.sign(data)
        lossless = sign_match.all(dim=1).float().mean().item() * 100
        per_dim = sign_match.float().mean().item() * 100
    return lossless, per_dim


def pick_free_cell(model, skipped):
    """Smallest |w|-first among cells that are neither lookup-frozen nor int8-frozen."""
    with torch.no_grad():
        W1_abs = model.W1_float.abs().clone()
        W2_abs = model.W2_float.abs().clone()
        W1_abs[model.W1_frozen_mask | model.W1_int8_mask] = float("inf")
        W2_abs[model.W2_frozen_mask | model.W2_int8_mask] = float("inf")
        for (mat, i, j) in skipped:
            if mat == "W1":
                W1_abs[i, j] = float("inf")
            else:
                W2_abs[i, j] = float("inf")
        all_abs = torch.cat([W1_abs.flatten(), W2_abs.flatten()])
        free_count = (~torch.isinf(all_abs)).sum().item()
        if free_count == 0:
            return None
        val, flat_idx = torch.min(all_abs, dim=0)

    idx = flat_idx.item()
    n1 = model.W1_float.numel()
    if idx < n1:
        i = int(idx // model.W1_float.shape[1])
        j = int(idx % model.W1_float.shape[1])
        return ("W1", i, j)
    idx2 = idx - n1
    i = int(idx2 // model.W2_float.shape[1])
    j = int(idx2 % model.W2_float.shape[1])
    return ("W2", i, j)


def candidate_ints(w, alpha, width=CAND_WIDTH, max_abs=127):
    """Return candidate int values ranked by |w/alpha - v|."""
    wn = w / max(alpha, 1e-8)
    nearest = int(round(wn))
    nearest = max(-max_abs, min(max_abs, nearest))
    cands = []
    for delta in range(width + 1):
        if delta == 0:
            cands.append(nearest)
        else:
            if nearest - delta >= -max_abs:
                cands.append(nearest - delta)
            if nearest + delta <= max_abs:
                cands.append(nearest + delta)
    # deduplicate preserving order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def try_commit_int8(model, cell, int_v, data):
    mat, i, j = cell
    if mat == "W1":
        orig_mask = bool(model.W1_int8_mask[i, j].item())
        orig_val = float(model.W1_int8[i, j].item())
        orig_float = float(model.W1_float.data[i, j].item())
        model.W1_int8_mask[i, j] = True
        model.W1_int8[i, j] = float(int_v)
        model.W1_float.data[i, j] = 0.0
    else:
        orig_mask = bool(model.W2_int8_mask[i, j].item())
        orig_val = float(model.W2_int8[i, j].item())
        orig_float = float(model.W2_float.data[i, j].item())
        model.W2_int8_mask[i, j] = True
        model.W2_int8[i, j] = float(int_v)
        model.W2_float.data[i, j] = 0.0
    ll, _ = metrics(model, data)
    return ll, (mat, i, j, orig_mask, orig_val, orig_float)


def rollback(model, saved):
    mat, i, j, orig_mask, orig_val, orig_float = saved
    if mat == "W1":
        model.W1_int8_mask[i, j] = orig_mask
        model.W1_int8[i, j] = orig_val
        model.W1_float.data[i, j] = orig_float
    else:
        model.W2_int8_mask[i, j] = orig_mask
        model.W2_int8[i, j] = orig_val
        model.W2_float.data[i, j] = orig_float


def retrain(model, data, max_iter=RETRAIN_MAX_ITER, patience=RETRAIN_PATIENCE):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50,
                            line_search_fn="strong_wolfe")
    best, no_imp = float("inf"), 0
    for _ in range(max_iter):
        def closure():
            opt.zero_grad()
            x_hat, _ = model(data)
            loss = F.mse_loss(x_hat, data)
            loss.backward()
            with torch.no_grad():
                if model.W1_float.grad is not None:
                    mask = model.W1_frozen_mask | model.W1_int8_mask
                    model.W1_float.grad[mask] = 0.0
                if model.W2_float.grad is not None:
                    mask = model.W2_frozen_mask | model.W2_int8_mask
                    model.W2_float.grad[mask] = 0.0
            return loss
        lv = opt.step(closure)
        lf = lv.item() if torch.is_tensor(lv) else lv
        if lf < best - 1e-9:
            best, no_imp = lf, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    return best


def save_checkpoint(path, model, skipped, n_processed, n_accepted, n_skipped):
    torch.save({
        "codebook_W1": model.codebook_W1.data.cpu(),
        "codebook_W2": model.codebook_W2.data.cpu(),
        "W1_idx": model.W1_idx.cpu(),
        "W2_idx": model.W2_idx.cpu(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu(),
        "W1_int8_mask": model.W1_int8_mask.cpu(),
        "W2_int8_mask": model.W2_int8_mask.cpu(),
        "W1_int8": model.W1_int8.cpu(),
        "W2_int8": model.W2_int8.cpu(),
        "alpha_free_W1": model.alpha_free_W1.data.cpu(),
        "alpha_free_W2": model.alpha_free_W2.data.cpu(),
        "W1_float": model.W1_float.data.cpu(),
        "W2_float": model.W2_float.data.cpu(),
        "b1": model.b1.data.cpu(),
        "b2": model.b2.data.cpu(),
        "db1": model.db1.data.cpu(),
        "db2": model.db2.data.cpu(),
        "c19_c": model.c19.c_raw.data.cpu(),
        "c19_rho": model.c19.rho_raw.data.cpu(),
        "skipped": [list(c) for c in skipped],
        "n_processed": n_processed,
        "n_accepted": n_accepted,
        "n_skipped": n_skipped,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_path = out_dir / "telemetry.jsonl"

    print(f"=== FREE CELL INT8 (cell-by-cell) ===", flush=True)
    print(f"Source: {args.source}", flush=True)

    state = load_source_json(args.source)
    model = FreeInt8Merger(state, DEVICE).to(DEVICE)
    data = load_byte_pairs().to(DEVICE)

    total = model.total_cells()
    free_w1, free_w2 = model.still_free_count()
    lookup_frozen = model.W1_frozen_mask.sum().item() + model.W2_frozen_mask.sum().item()
    print(f"Start: H={model.H}, total={total}", flush=True)
    print(f"  lookup-frozen: {lookup_frozen}", flush=True)
    print(f"  still-free:    {free_w1 + free_w2}", flush=True)
    print(f"  alpha_free_W1={model.alpha_free_W1.item():.6f}", flush=True)
    print(f"  alpha_free_W2={model.alpha_free_W2.item():.6f}", flush=True)
    ll0, pd0 = metrics(model, data)
    print(f"Baseline lossless={ll0:.4f}%, per-dim={pd0:.4f}%", flush=True)

    skipped = set()
    n_processed = 0
    n_accepted = 0
    n_skipped = 0
    t0 = time.time()

    while True:
        cell = pick_free_cell(model, skipped)
        if cell is None:
            print("\nNo more free cells.", flush=True)
            break
        n_processed += 1

        mat, i, j = cell
        if mat == "W1":
            w = model.W1_float.data[i, j].item()
            alpha = model.alpha_free_W1.item()
        else:
            w = model.W2_float.data[i, j].item()
            alpha = model.alpha_free_W2.item()

        candidates = candidate_ints(w, alpha)
        accepted = False
        best_ll = -1.0
        used_v = None
        n_tries = 0
        for int_v in candidates:
            n_tries += 1
            ll, saved = try_commit_int8(model, cell, int_v, data)
            if ll > best_ll:
                best_ll = ll
            if ll >= LOSSLESS_FLOOR:
                accepted = True
                used_v = int_v
                break
            else:
                rollback(model, saved)

        if accepted:
            n_accepted += 1
            status = f"ACCEPT v={used_v}"
        else:
            skipped.add(cell)
            n_skipped += 1
            status = f"SKIP best_ll={best_ll:.2f}%"

        free_w1, free_w2 = model.still_free_count()

        row = {
            "cell": list(cell), "w": w, "alpha": alpha,
            "accepted": accepted, "used_value": used_v,
            "n_tries": n_tries, "best_ll": best_ll,
            "still_free": free_w1 + free_w2,
        }
        with open(telem_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if n_processed % 50 == 0 or not accepted:
            ll, _ = metrics(model, data)
            print(
                f"  #{n_processed} [{status}]: tries={n_tries}, "
                f"still_free={free_w1 + free_w2}, "
                f"accepted={n_accepted}, skipped={n_skipped}, ll={ll:.4f}%",
                flush=True,
            )

        # Retrain periodically
        if n_accepted > 0 and n_accepted % RETRAIN_EVERY == 0 and accepted:
            print(f"  [retrain @ {n_accepted} accepts]", flush=True)
            retrain(model, data)
            save_checkpoint(out_dir / "checkpoint.pt", model, skipped,
                            n_processed, n_accepted, n_skipped)

        # Per-50 cell checkpoint (safety)
        if n_processed % 100 == 0:
            save_checkpoint(out_dir / "checkpoint.pt", model, skipped,
                            n_processed, n_accepted, n_skipped)

        if n_processed > 5000:
            break

    # Final retrain
    print(f"\n--- FINAL RETRAIN ---", flush=True)
    retrain(model, data, max_iter=100, patience=30)
    ll, pd = metrics(model, data)

    t_total = time.time() - t0
    free_w1, free_w2 = model.still_free_count()
    n_int8_W1 = model.W1_int8_mask.sum().item()
    n_int8_W2 = model.W2_int8_mask.sum().item()

    print(f"\n=== DONE ===")
    print(f"  Cells processed: {n_processed}")
    print(f"  Accepted (int8-frozen): {n_accepted}")
    print(f"    W1: {n_int8_W1}, W2: {n_int8_W2}")
    print(f"  Skipped (still float): {n_skipped}")
    print(f"  Still free floats: {free_w1 + free_w2}")
    print(f"  alpha_free_W1={model.alpha_free_W1.item():.6f}")
    print(f"  alpha_free_W2={model.alpha_free_W2.item():.6f}")
    print(f"  Lossless: {ll:.4f}%, per-dim: {pd:.4f}%")
    print(f"  Time: {t_total:.0f}s")

    # Deploy size estimate
    cb1_sz = len(model.codebook_W1)
    cb2_sz = len(model.codebook_W2)
    bits_W1 = max(1, int(np.ceil(np.log2(max(cb1_sz, 2)))))
    bits_W2 = max(1, int(np.ceil(np.log2(max(cb2_sz, 2)))))
    idx_bytes_W1 = int(np.ceil(model.W1_idx.numel() * bits_W1 / 8))
    idx_bytes_W2 = int(np.ceil(model.W2_idx.numel() * bits_W2 / 8))
    cb_bytes = (cb1_sz + cb2_sz) * 4
    int8_bytes = n_int8_W1 + n_int8_W2  # 1 B each
    free_float_bytes = (free_w1 + free_w2) * 4
    # Mask overhead
    mask_bytes = (model.W1_frozen_mask.numel() + model.W2_frozen_mask.numel()) // 8
    int8_mask_bytes = (model.W1_int8_mask.numel() + model.W2_int8_mask.numel()) // 8
    misc_bytes = (model.b1.numel() + model.b2.numel() + model.db1.numel() +
                  model.db2.numel() + 2 * model.H + 2) * 4  # +2 for alpha_free
    total_bytes = (idx_bytes_W1 + idx_bytes_W2 + cb_bytes +
                   int8_bytes + free_float_bytes +
                   mask_bytes + int8_mask_bytes + misc_bytes)
    print(f"\n  Deploy size estimate:")
    print(f"    idx_W1 ({bits_W1}bit):    {idx_bytes_W1} B")
    print(f"    idx_W2 ({bits_W2}bit):    {idx_bytes_W2} B")
    print(f"    codebooks:           {cb_bytes} B")
    print(f"    int8 values:         {int8_bytes} B")
    print(f"    free floats:         {free_float_bytes} B")
    print(f"    masks (lookup+int8): {mask_bytes + int8_mask_bytes} B")
    print(f"    biases + C19 + a:    {misc_bytes} B")
    print(f"    TOTAL:               {total_bytes} B ({total_bytes/1024:.2f} KB)")

    artifact = {
        "architecture": "lookup + free-int8 hybrid",
        "source": args.source,
        "H": model.H, "in_dim": model.in_dim, "out_dim": model.out_dim,
        "total_cells": total,
        "lookup_frozen": lookup_frozen,
        "int8_frozen": n_int8_W1 + n_int8_W2,
        "still_free": free_w1 + free_w2,
        "codebook_W1": model.codebook_W1.data.cpu().numpy().tolist(),
        "codebook_W2": model.codebook_W2.data.cpu().numpy().tolist(),
        "alpha_free_W1": float(model.alpha_free_W1.item()),
        "alpha_free_W2": float(model.alpha_free_W2.item()),
        "lossless": ll, "per_dim": pd,
        "n_processed": n_processed, "n_accepted": n_accepted, "n_skipped": n_skipped,
        "time_s": t_total,
        "deploy_bytes_est": total_bytes,
        "W1_idx": model.W1_idx.cpu().numpy().astype(int).tolist(),
        "W2_idx": model.W2_idx.cpu().numpy().astype(int).tolist(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu().numpy().astype(int).tolist(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu().numpy().astype(int).tolist(),
        "W1_int8_mask": model.W1_int8_mask.cpu().numpy().astype(int).tolist(),
        "W2_int8_mask": model.W2_int8_mask.cpu().numpy().astype(int).tolist(),
        "W1_int8": model.W1_int8.cpu().numpy().astype(int).tolist(),
        "W2_int8": model.W2_int8.cpu().numpy().astype(int).tolist(),
        "W1_float": model.W1_float.data.cpu().numpy().tolist(),
        "W2_float": model.W2_float.data.cpu().numpy().tolist(),
        "b1": model.b1.data.cpu().numpy().tolist(),
        "b2": model.b2.data.cpu().numpy().tolist(),
        "db1": model.db1.data.cpu().numpy().tolist(),
        "db2": model.db2.data.cpu().numpy().tolist(),
        "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
        "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
    }
    with open(out_dir / "final_model.json", "w") as f:
        json.dump(artifact, f)
    print(f"  Saved: {out_dir / 'final_model.json'}")


if __name__ == "__main__":
    main()
