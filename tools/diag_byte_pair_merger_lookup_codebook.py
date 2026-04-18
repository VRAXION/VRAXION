"""L1 Merger — LOOKUP CODEBOOK with gravity-driven peaks.

User insight (2026-04-18): stop using int*alpha fixed grid.
Instead store a small lookup table (8-16 values per matrix) and
per-cell codebook index. Any real value allowed. Codebook values
are TRAINABLE — the peaks drift toward the cells they anchor
(literal gravity), and cells gravitate to the peaks.

Protocol:
  1. Load halfscale pentary source (5 initial codebook entries per matrix).
  2. Loop:
     a. KDE on free cells (per matrix) → find top density peak(s).
     b. Only accept peak if mass >= MIN_MASS.
     c. Append peak value to codebook (becomes trainable param).
     d. For each free cell near peak (within RADIUS): try freeze to new idx,
        accept if lossless >= threshold.
     e. Retrain: codebook + bias + C19 + remaining free W_float.
     f. Stop when no peak meets threshold.
  3. Final retrain + save.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LUT_IN_PATH = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")
LOSSLESS_FLOOR = 99.99
RETRAIN_MAX_ITER = 60
RETRAIN_PATIENCE = 20
MIN_PEAK_MASS = 15
PEAK_BANDWIDTH = 0.04
PEAK_RADIUS = 0.15  # normalized-space radius for candidate cells near peak
MAX_CODEBOOK_SIZE = 20  # per matrix


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


class LookupMerger(nn.Module):
    def __init__(self, state, device):
        super().__init__()
        self.in_dim = state["in_dim"]
        self.out_dim = state["out_dim"]
        self.H = state["H"]

        a1 = float(state["alpha_W1"])
        a2 = float(state["alpha_W2"])
        # Initial codebook: {-2α, -α, 0, +α, +2α} per matrix
        cb1 = torch.tensor([-2*a1, -a1, 0.0, a1, 2*a1], dtype=torch.float32)
        cb2 = torch.tensor([-2*a2, -a2, 0.0, a2, 2*a2], dtype=torch.float32)
        self.codebook_W1 = nn.Parameter(cb1.to(device).clone())
        self.codebook_W2 = nn.Parameter(cb2.to(device).clone())

        # Map old ints {-2,-1,0,1,2} to indices {0,1,2,3,4}
        W1_int = state["W1_int"].long()
        W2_int = state["W2_int"].long()
        self.register_buffer("W1_idx", (W1_int + 2).clamp(0, 4).to(device))
        self.register_buffer("W2_idx", (W2_int + 2).clamp(0, 4).to(device))

        # Frozen masks
        self.register_buffer("W1_frozen_mask", state["W1_frozen_mask"].to(device).clone())
        self.register_buffer("W2_frozen_mask", state["W2_frozen_mask"].to(device).clone())

        # Free cells: raw float values
        self.W1_float = nn.Parameter(state["W1"].to(device).clone())
        self.W2_float = nn.Parameter(state["W2"].to(device).clone())

        # Bias + C19
        self.b1 = nn.Parameter(state["b1"].to(device).clone())
        self.b2 = nn.Parameter(state["b2"].to(device).clone())
        self.db1 = nn.Parameter(state["db1"].to(device).clone())
        self.db2 = nn.Parameter(state["db2"].to(device).clone())
        self.c19 = C19Activation(self.H)
        with torch.no_grad():
            self.c19.c_raw.copy_(state["c19_c"].to(device))
            self.c19.rho_raw.copy_(state["c19_rho"].to(device))

    def W1_eff(self):
        looked = self.codebook_W1[self.W1_idx]
        return torch.where(self.W1_frozen_mask, looked, self.W1_float)

    def W2_eff(self):
        looked = self.codebook_W2[self.W2_idx]
        return torch.where(self.W2_frozen_mask, looked, self.W2_float)

    def encode(self, x):
        return self.c19(x @ self.W1_eff() + self.b1) @ self.W2_eff() + self.b2

    def decode(self, z):
        return (z @ self.W2_eff().t() + self.db1) @ self.W1_eff().t() + self.db2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def free_count(self):
        return (
            (~self.W1_frozen_mask).sum().item(),
            (~self.W2_frozen_mask).sum().item(),
        )

    def total_cells(self):
        return self.W1_float.numel() + self.W2_float.numel()

    def add_codebook_entry(self, matrix, value):
        """Return new index. Codebook grows (Parameter recreated)."""
        if matrix == "W1":
            new = torch.cat([
                self.codebook_W1.data,
                torch.tensor([value], dtype=torch.float32, device=self.codebook_W1.device)
            ])
            self.codebook_W1 = nn.Parameter(new)
            return len(new) - 1
        else:
            new = torch.cat([
                self.codebook_W2.data,
                torch.tensor([value], dtype=torch.float32, device=self.codebook_W2.device)
            ])
            self.codebook_W2 = nn.Parameter(new)
            return len(new) - 1

    def remove_unused_codebook_entries(self):
        """Prune codebook entries not referenced by any frozen cell."""
        for mat in ["W1", "W2"]:
            if mat == "W1":
                cb = self.codebook_W1
                idx = self.W1_idx
                mask = self.W1_frozen_mask
            else:
                cb = self.codebook_W2
                idx = self.W2_idx
                mask = self.W2_frozen_mask
            used = set(idx[mask].cpu().tolist())
            if len(used) == cb.numel():
                continue
            # Remap
            sorted_used = sorted(used)
            remap = {old: new for new, old in enumerate(sorted_used)}
            new_cb = cb.data[sorted_used].clone()
            new_idx = idx.clone()
            for old, new in remap.items():
                new_idx[idx == old] = new
            if mat == "W1":
                self.codebook_W1 = nn.Parameter(new_cb)
                self.W1_idx.copy_(new_idx)
            else:
                self.codebook_W2 = nn.Parameter(new_cb)
                self.W2_idx.copy_(new_idx)


def load_source_json(path):
    with open(path, "r") as f:
        d = json.load(f)
    return {
        "H": d["H"],
        "in_dim": d["in_dim"],
        "out_dim": d["out_dim"],
        "W1": torch.tensor(d["W1_float"], dtype=torch.float32),
        "W2": torch.tensor(d["W2_float"], dtype=torch.float32),
        "W1_frozen_mask": torch.tensor(d["W1_frozen_mask"], dtype=torch.bool),
        "W2_frozen_mask": torch.tensor(d["W2_frozen_mask"], dtype=torch.bool),
        "W1_int": torch.tensor(d["W1_int"], dtype=torch.float32),
        "W2_int": torch.tensor(d["W2_int"], dtype=torch.float32),
        "alpha_W1": d["alpha_W1"],
        "alpha_W2": d["alpha_W2"],
        "b1": torch.tensor(d["b1"], dtype=torch.float32),
        "b2": torch.tensor(d["b2"], dtype=torch.float32),
        "db1": torch.tensor(d["db1"], dtype=torch.float32),
        "db2": torch.tensor(d["db2"], dtype=torch.float32),
        "c19_c": torch.tensor(d["c19_c"], dtype=torch.float32),
        "c19_rho": torch.tensor(d["c19_rho"], dtype=torch.float32),
    }


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


def kde_peaks(values, bandwidth=PEAK_BANDWIDTH, grid_n=800, min_mass=MIN_PEAK_MASS):
    """Return list of (peak_value, mass) sorted by mass descending.
    values: 1D numpy array (normalized = value/alpha OR absolute — caller decides)
    """
    if len(values) == 0:
        return []
    lo = float(values.min()) - 0.1
    hi = float(values.max()) + 0.1
    grid = np.linspace(lo, hi, grid_n)
    diff = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1) / (bandwidth * np.sqrt(2*np.pi) * len(values))

    peaks = []
    radius_bins = 5
    for i in range(radius_bins, len(density)-radius_bins):
        window = density[i-radius_bins:i+radius_bins+1]
        if density[i] == window.max() and density[i] > 0:
            px = grid[i]
            mass = int(((values >= px - 0.08) & (values <= px + 0.08)).sum())
            if mass >= min_mass:
                peaks.append((float(px), mass))

    peaks.sort(key=lambda x: -x[1])
    return peaks


def candidate_cells_near(model, matrix, target_value, radius=PEAK_RADIUS):
    """Return free cells within |value - target| <= radius."""
    with torch.no_grad():
        if matrix == "W1":
            W = model.W1_float.data
            mask = model.W1_frozen_mask
        else:
            W = model.W2_float.data
            mask = model.W2_frozen_mask
        diff = (W - target_value).abs()
        cond = (~mask) & (diff <= radius)
        ii, jj = torch.where(cond)
        dists = diff[ii, jj]
        order = torch.argsort(dists)
        return [(matrix, int(ii[k].item()), int(jj[k].item())) for k in order]


def try_freeze(model, cell, new_idx, data):
    mat, i, j = cell
    if mat == "W1":
        orig_mask = bool(model.W1_frozen_mask[i, j].item())
        orig_idx = int(model.W1_idx[i, j].item())
        orig_float = float(model.W1_float.data[i, j].item())
        model.W1_frozen_mask[i, j] = True
        model.W1_idx[i, j] = new_idx
        model.W1_float.data[i, j] = 0.0
    else:
        orig_mask = bool(model.W2_frozen_mask[i, j].item())
        orig_idx = int(model.W2_idx[i, j].item())
        orig_float = float(model.W2_float.data[i, j].item())
        model.W2_frozen_mask[i, j] = True
        model.W2_idx[i, j] = new_idx
        model.W2_float.data[i, j] = 0.0
    ll, _ = metrics(model, data)
    return ll, (mat, i, j, orig_mask, orig_idx, orig_float)


def rollback(model, saved):
    mat, i, j, orig_mask, orig_idx, orig_float = saved
    if mat == "W1":
        model.W1_frozen_mask[i, j] = orig_mask
        model.W1_idx[i, j] = orig_idx
        model.W1_float.data[i, j] = orig_float
    else:
        model.W2_frozen_mask[i, j] = orig_mask
        model.W2_idx[i, j] = orig_idx
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
                    model.W1_float.grad[model.W1_frozen_mask] = 0.0
                if model.W2_float.grad is not None:
                    model.W2_float.grad[model.W2_frozen_mask] = 0.0
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


def save_checkpoint(path, model, iter_n, n_accept_total):
    torch.save({
        "codebook_W1": model.codebook_W1.data.cpu().clone(),
        "codebook_W2": model.codebook_W2.data.cpu().clone(),
        "W1_idx": model.W1_idx.cpu().clone(),
        "W2_idx": model.W2_idx.cpu().clone(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu().clone(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu().clone(),
        "W1_float": model.W1_float.data.cpu().clone(),
        "W2_float": model.W2_float.data.cpu().clone(),
        "b1": model.b1.data.cpu().clone(),
        "b2": model.b2.data.cpu().clone(),
        "db1": model.db1.data.cpu().clone(),
        "db2": model.db2.data.cpu().clone(),
        "c19_c": model.c19.c_raw.data.cpu().clone(),
        "c19_rho": model.c19.rho_raw.data.cpu().clone(),
        "iter_n": iter_n,
        "n_accept_total": n_accept_total,
    }, path)


def log_row(telem_path, row):
    with open(telem_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--min-mass", type=int, default=MIN_PEAK_MASS)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_path = out_dir / "telemetry.jsonl"

    print(f"=== LOOKUP CODEBOOK (gravity-driven) ===", flush=True)
    print(f"Source: {args.source}", flush=True)

    state = load_source_json(args.source)
    model = LookupMerger(state, DEVICE).to(DEVICE)
    data = load_byte_pairs().to(DEVICE)

    total = model.total_cells()
    free_w1, free_w2 = model.free_count()
    print(f"Start: H={model.H}, total={total}, "
          f"frozen={total - free_w1 - free_w2}, free={free_w1 + free_w2}",
          flush=True)
    ll0, pd0 = metrics(model, data)
    print(f"Baseline lossless={ll0:.4f}%, per-dim={pd0:.4f}%", flush=True)
    print(f"Codebook W1 ({len(model.codebook_W1)}): "
          f"{[f'{v:+.3f}' for v in model.codebook_W1.data.cpu().tolist()]}",
          flush=True)
    print(f"Codebook W2 ({len(model.codebook_W2)}): "
          f"{[f'{v:+.3f}' for v in model.codebook_W2.data.cpu().tolist()]}",
          flush=True)

    t0 = time.time()
    iter_n = 0
    n_accept_total = 0

    while iter_n < args.max_iters:
        iter_n += 1
        print(f"\n--- ITER {iter_n} ---", flush=True)

        # KDE peak finding per matrix (absolute weight space, not normalized)
        with torch.no_grad():
            free_W1_vals = model.W1_float.data[~model.W1_frozen_mask].cpu().numpy()
            free_W2_vals = model.W2_float.data[~model.W2_frozen_mask].cpu().numpy()

        # Exclude values too close to existing codebook entries (avoid dupes)
        def filter_near_existing(vals, codebook_vals, min_dist=0.08):
            keep = np.ones_like(vals, dtype=bool)
            for cv in codebook_vals:
                keep &= np.abs(vals - cv) > min_dist
            return vals[keep]

        cb_W1_vals = model.codebook_W1.data.cpu().numpy()
        cb_W2_vals = model.codebook_W2.data.cpu().numpy()
        free_W1_exc = filter_near_existing(free_W1_vals, cb_W1_vals)
        free_W2_exc = filter_near_existing(free_W2_vals, cb_W2_vals)

        peaks_W1 = kde_peaks(free_W1_exc, min_mass=args.min_mass)
        peaks_W2 = kde_peaks(free_W2_exc, min_mass=args.min_mass)

        print(f"Peaks W1 (excl.): {[(f'{p:+.3f}', m) for p, m in peaks_W1[:5]]}",
              flush=True)
        print(f"Peaks W2 (excl.): {[(f'{p:+.3f}', m) for p, m in peaks_W2[:5]]}",
              flush=True)

        # Pick the single best peak across both matrices
        best_candidate = None  # (matrix, value, mass)
        if peaks_W1:
            v, m = peaks_W1[0]
            best_candidate = ("W1", v, m)
        if peaks_W2:
            v, m = peaks_W2[0]
            if best_candidate is None or m > best_candidate[2]:
                best_candidate = ("W2", v, m)

        if best_candidate is None:
            print("No more peaks meet threshold. STOP.", flush=True)
            break
        if len(getattr(model, f"codebook_{best_candidate[0]}")) >= MAX_CODEBOOK_SIZE:
            print(f"Codebook_{best_candidate[0]} at max size. STOP.", flush=True)
            break

        mat, peak_val, peak_mass = best_candidate
        print(f"CHOSEN peak: matrix={mat}, value={peak_val:+.4f}, mass={peak_mass}",
              flush=True)

        # Add to codebook
        new_idx = model.add_codebook_entry(mat, peak_val)
        print(f"  codebook_{mat} grown to {len(getattr(model, f'codebook_{mat}'))} entries, "
              f"new_idx={new_idx}", flush=True)

        # Try to freeze all cells near peak
        candidates = candidate_cells_near(model, mat, peak_val, radius=PEAK_RADIUS)
        print(f"  {len(candidates)} candidate cells within radius {PEAK_RADIUS}", flush=True)

        n_accept = 0
        n_try = 0
        saves = []
        for cell in candidates:
            n_try += 1
            ll, saved = try_freeze(model, cell, new_idx, data)
            if ll >= LOSSLESS_FLOOR:
                n_accept += 1
                saves.append(saved)
            else:
                rollback(model, saved)

        print(f"  tried={n_try}, accepted={n_accept}", flush=True)
        n_accept_total += n_accept

        # Check if codebook entry unused → remove it
        used_cells = (
            (model.W1_frozen_mask & (model.W1_idx == new_idx)).sum().item()
            if mat == "W1"
            else (model.W2_frozen_mask & (model.W2_idx == new_idx)).sum().item()
        )
        if used_cells == 0:
            # Strip the unused entry
            if mat == "W1":
                model.codebook_W1 = nn.Parameter(model.codebook_W1.data[:-1])
            else:
                model.codebook_W2 = nn.Parameter(model.codebook_W2.data[:-1])
            print(f"  unused peak — removed from codebook", flush=True)

        # Retrain
        if n_accept > 0:
            print(f"  retraining...", flush=True)
            t_r = time.time()
            final_loss = retrain(model, data)
            ll, pd = metrics(model, data)
            print(f"  retrain done: loss={final_loss:.6f}, "
                  f"lossless={ll:.4f}%, time={time.time()-t_r:.1f}s", flush=True)

        free_w1, free_w2 = model.free_count()
        frozen = total - free_w1 - free_w2
        ll, pd = metrics(model, data)

        row = {
            "iter": iter_n,
            "peak_matrix": mat,
            "peak_value": peak_val,
            "peak_mass": peak_mass,
            "n_accept": n_accept,
            "n_try": n_try,
            "n_accept_total": n_accept_total,
            "frozen_total": frozen,
            "lossless": ll,
            "per_dim": pd,
            "codebook_W1_size": len(model.codebook_W1),
            "codebook_W2_size": len(model.codebook_W2),
            "time_s": time.time() - t0,
        }
        log_row(telem_path, row)

        save_checkpoint(out_dir / "checkpoint.pt", model, iter_n, n_accept_total)
        print(f"  frozen={frozen}/{total} ({100*frozen/total:.1f}%), "
              f"lossless={ll:.4f}%", flush=True)

    # Final pass: clean up codebook + final retrain
    print(f"\n--- FINAL RETRAIN ---", flush=True)
    model.remove_unused_codebook_entries()
    retrain(model, data, max_iter=100, patience=30)
    ll, pd = metrics(model, data)
    free_w1, free_w2 = model.free_count()
    frozen = total - free_w1 - free_w2

    t_total = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  Iters: {iter_n}")
    print(f"  Total accepted: {n_accept_total}")
    print(f"  Frozen: {frozen}/{total} ({100*frozen/total:.1f}%)")
    print(f"  Codebook W1 ({len(model.codebook_W1)}): "
          f"{[f'{v:+.4f}' for v in model.codebook_W1.data.cpu().tolist()]}")
    print(f"  Codebook W2 ({len(model.codebook_W2)}): "
          f"{[f'{v:+.4f}' for v in model.codebook_W2.data.cpu().tolist()]}")
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
    # Biases + C19 + masks
    free_float_bytes = (free_w1 + free_w2) * 4  # float32
    mask_bytes = (model.W1_frozen_mask.numel() + model.W2_frozen_mask.numel()) // 8
    misc_bytes = (model.b1.numel() + model.b2.numel() + model.db1.numel() +
                  model.db2.numel() + 2 * model.H) * 4
    total_bytes = idx_bytes_W1 + idx_bytes_W2 + cb_bytes + free_float_bytes + mask_bytes + misc_bytes
    print(f"\n  Deploy size estimate:")
    print(f"    idx_W1 ({bits_W1}bit): {idx_bytes_W1} B")
    print(f"    idx_W2 ({bits_W2}bit): {idx_bytes_W2} B")
    print(f"    codebooks:             {cb_bytes} B")
    print(f"    free floats:           {free_float_bytes} B")
    print(f"    masks:                 {mask_bytes} B")
    print(f"    biases + C19:          {misc_bytes} B")
    print(f"    TOTAL:                 {total_bytes} B ({total_bytes/1024:.2f} KB)")

    artifact = {
        "architecture": "lookup codebook gravity",
        "source": args.source,
        "H": model.H,
        "in_dim": model.in_dim,
        "out_dim": model.out_dim,
        "total_cells": total,
        "frozen_cells": frozen,
        "pct_frozen": 100 * frozen / total,
        "codebook_W1": model.codebook_W1.data.cpu().numpy().tolist(),
        "codebook_W2": model.codebook_W2.data.cpu().numpy().tolist(),
        "lossless": ll,
        "per_dim": pd,
        "n_iters": iter_n,
        "n_accept_total": n_accept_total,
        "time_s": t_total,
        "deploy_bytes_est": total_bytes,
        "W1_idx": model.W1_idx.cpu().numpy().astype(int).tolist(),
        "W2_idx": model.W2_idx.cpu().numpy().astype(int).tolist(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu().numpy().astype(int).tolist(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu().numpy().astype(int).tolist(),
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
