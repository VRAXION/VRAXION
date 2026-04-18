"""Exact lookup-codebook stage from a pure-float H=81 winner.

This is the exact-only version of the gravity/codebook idea:

  1. Load an exact pure-float merger checkpoint.
  2. Find density peaks in the remaining free weights.
  3. Add one trainable codebook entry for a chosen peak.
  4. Freeze nearby cells one by one, requiring exact 100.0000% after each.
  5. LBFGS plateau.
  6. Keep the round only if exactness survives the refit.

The output format is intentionally compatible with the strict staged int8
runner's source loader.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from diag_byte_pair_merger_exact_utils import (
    DEVICE,
    C19Activation,
    eval_stats,
    load_byte_pairs,
    load_json,
    train_lbfgs_plateau,
)


class ExactLookupMerger(nn.Module):
    def __init__(self, state: dict, device: torch.device):
        super().__init__()
        self.H = int(state["H"])
        self.in_dim = int(state["in_dim"])
        self.out_dim = int(state["out_dim"])

        self.codebook_W1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))
        self.codebook_W2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))

        self.register_buffer(
            "W1_idx",
            torch.zeros((self.in_dim, self.H), dtype=torch.long, device=device),
        )
        self.register_buffer(
            "W2_idx",
            torch.zeros((self.H, self.out_dim), dtype=torch.long, device=device),
        )
        self.register_buffer(
            "W1_frozen_mask",
            torch.zeros((self.in_dim, self.H), dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "W2_frozen_mask",
            torch.zeros((self.H, self.out_dim), dtype=torch.bool, device=device),
        )

        self.W1_float = nn.Parameter(torch.tensor(state["W1"], dtype=torch.float32, device=device))
        self.W2_float = nn.Parameter(torch.tensor(state["W2"], dtype=torch.float32, device=device))
        self.b1 = nn.Parameter(torch.tensor(state["b1"], dtype=torch.float32, device=device))
        self.b2 = nn.Parameter(torch.tensor(state["b2"], dtype=torch.float32, device=device))
        self.db1 = nn.Parameter(torch.tensor(state["db1"], dtype=torch.float32, device=device))
        self.db2 = nn.Parameter(torch.tensor(state["db2"], dtype=torch.float32, device=device))
        self.c19 = C19Activation(self.H)
        with torch.no_grad():
            self.c19.c_raw.copy_(torch.tensor(state["c19_c"], dtype=torch.float32, device=device))
            self.c19.rho_raw.copy_(torch.tensor(state["c19_rho"], dtype=torch.float32, device=device))

    def W1_eff(self) -> torch.Tensor:
        looked = self.codebook_W1[self.W1_idx]
        return torch.where(self.W1_frozen_mask, looked, self.W1_float)

    def W2_eff(self) -> torch.Tensor:
        looked = self.codebook_W2[self.W2_idx]
        return torch.where(self.W2_frozen_mask, looked, self.W2_float)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.c19(x @ self.W1_eff() + self.b1) @ self.W2_eff() + self.b2

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W2_eff().t() + self.db1) @ self.W1_eff().t() + self.db2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def total_cells(self) -> int:
        return int(self.W1_float.numel() + self.W2_float.numel())

    def free_count(self) -> tuple[int, int]:
        return (
            int((~self.W1_frozen_mask).sum().item()),
            int((~self.W2_frozen_mask).sum().item()),
        )

    def add_codebook_entry(self, matrix: str, value: float) -> int:
        if matrix == "W1":
            new = torch.cat(
                [
                    self.codebook_W1.data,
                    torch.tensor([value], dtype=torch.float32, device=self.codebook_W1.device),
                ]
            )
            self.codebook_W1 = nn.Parameter(new)
            return int(len(new) - 1)
        new = torch.cat(
            [
                self.codebook_W2.data,
                torch.tensor([value], dtype=torch.float32, device=self.codebook_W2.device),
            ]
        )
        self.codebook_W2 = nn.Parameter(new)
        return int(len(new) - 1)


def kde_peaks(
    values: np.ndarray,
    bandwidth: float,
    grid_n: int,
    min_mass: int,
    mass_radius: float,
) -> list[tuple[float, int]]:
    if len(values) == 0:
        return []
    lo = float(values.min()) - 0.1
    hi = float(values.max()) + 0.1
    grid = np.linspace(lo, hi, grid_n)
    diff = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1) / (
        bandwidth * np.sqrt(2 * np.pi) * len(values)
    )
    peaks: list[tuple[float, int]] = []
    radius_bins = 5
    for i in range(radius_bins, len(density) - radius_bins):
        window = density[i - radius_bins : i + radius_bins + 1]
        if density[i] != window.max() or density[i] <= 0:
            continue
        px = float(grid[i])
        mass = int(((values >= px - mass_radius) & (values <= px + mass_radius)).sum())
        if mass >= min_mass:
            peaks.append((px, mass))
    peaks.sort(key=lambda x: -x[1])
    return peaks


def candidate_cells_near(
    model: ExactLookupMerger, matrix: str, target_value: float, radius: float
) -> list[tuple[str, int, int]]:
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
        return [
            (matrix, int(ii[k].item()), int(jj[k].item()))
            for k in order
        ]


def set_lookup_cell(
    model: ExactLookupMerger, cell: tuple[str, int, int], new_idx: int
) -> tuple[str, int, int, bool, int, float]:
    mat, i, j = cell
    if mat == "W1":
        orig_mask = bool(model.W1_frozen_mask[i, j].item())
        orig_idx = int(model.W1_idx[i, j].item())
        orig_float = float(model.W1_float.data[i, j].item())
        model.W1_frozen_mask[i, j] = True
        model.W1_idx[i, j] = new_idx
        model.W1_float.data[i, j] = 0.0
        return (mat, i, j, orig_mask, orig_idx, orig_float)
    orig_mask = bool(model.W2_frozen_mask[i, j].item())
    orig_idx = int(model.W2_idx[i, j].item())
    orig_float = float(model.W2_float.data[i, j].item())
    model.W2_frozen_mask[i, j] = True
    model.W2_idx[i, j] = new_idx
    model.W2_float.data[i, j] = 0.0
    return (mat, i, j, orig_mask, orig_idx, orig_float)


def rollback_lookup_cell(
    model: ExactLookupMerger,
    saved: tuple[str, int, int, bool, int, float],
) -> None:
    mat, i, j, orig_mask, orig_idx, orig_float = saved
    if mat == "W1":
        model.W1_frozen_mask[i, j] = orig_mask
        model.W1_idx[i, j] = orig_idx
        model.W1_float.data[i, j] = orig_float
    else:
        model.W2_frozen_mask[i, j] = orig_mask
        model.W2_idx[i, j] = orig_idx
        model.W2_float.data[i, j] = orig_float


def export_final_json(
    out_path: Path,
    source: str,
    model: ExactLookupMerger,
    history: list[dict],
    stats: dict[str, float | int | bool],
    elapsed: float,
) -> None:
    free_w1, free_w2 = model.free_count()
    total = model.total_cells()
    frozen = total - free_w1 - free_w2
    cb1_sz = len(model.codebook_W1)
    cb2_sz = len(model.codebook_W2)
    bits_w1 = max(1, int(np.ceil(np.log2(max(cb1_sz, 2)))))
    bits_w2 = max(1, int(np.ceil(np.log2(max(cb2_sz, 2)))))
    idx_bytes_w1 = int(np.ceil(model.W1_idx.numel() * bits_w1 / 8))
    idx_bytes_w2 = int(np.ceil(model.W2_idx.numel() * bits_w2 / 8))
    cb_bytes = (cb1_sz + cb2_sz) * 4
    free_float_bytes = (free_w1 + free_w2) * 4
    mask_bytes = (model.W1_frozen_mask.numel() + model.W2_frozen_mask.numel()) // 8
    misc_bytes = (
        model.b1.numel()
        + model.b2.numel()
        + model.db1.numel()
        + model.db2.numel()
        + 2 * model.H
    ) * 4
    total_bytes = idx_bytes_w1 + idx_bytes_w2 + cb_bytes + free_float_bytes + mask_bytes + misc_bytes

    payload = {
        "architecture": "lookup codebook gravity exact",
        "source": source,
        "H": model.H,
        "in_dim": model.in_dim,
        "out_dim": model.out_dim,
        "total_cells": total,
        "frozen_cells": frozen,
        "pct_frozen": 100.0 * frozen / total,
        "codebook_W1": model.codebook_W1.data.cpu().numpy().tolist(),
        "codebook_W2": model.codebook_W2.data.cpu().numpy().tolist(),
        "lossless": float(stats["lossless"]),
        "per_dim": float(stats["per_dim"]),
        "mse": float(stats["mse"]),
        "bad_pairs": int(stats["bad_pairs"]),
        "bad_dims": int(stats["bad_dims"]),
        "n_iters": len(history),
        "n_accept_total": int(sum(int(row["accepted"]) for row in history)),
        "time_s": elapsed,
        "deploy_bytes_est": total_bytes,
        "history": history,
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-iters", type=int, default=24)
    parser.add_argument("--max-codebook-size", type=int, default=20)
    parser.add_argument("--min-mass", type=int, default=15)
    parser.add_argument("--bandwidth", type=float, default=0.04)
    parser.add_argument("--peak-radius", type=float, default=0.15)
    parser.add_argument("--mass-radius", type=float, default=0.08)
    parser.add_argument("--grid-n", type=int, default=800)
    parser.add_argument("--retrain-outer", type=int, default=60)
    parser.add_argument("--retrain-patience", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runlog_path = out_dir / "run.log"
    telemetry_path = out_dir / "telemetry.jsonl"
    source_state = load_json(args.source)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(runlog_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    model = ExactLookupMerger(source_state, DEVICE).to(DEVICE)
    data = load_byte_pairs().to(DEVICE)
    start_stats = eval_stats(model, data)
    if not start_stats["exact"]:
        raise SystemExit("Source is not exact; exact lookup stage requires 100% input.")

    history: list[dict] = []
    t0 = time.time()

    log("=== EXACT LOOKUP CODEBOOK ===")
    log(f"Source: {args.source}")
    log(
        f"Baseline: lossless={start_stats['lossless']:.4f}% "
        f"bad_pairs={start_stats['bad_pairs']} "
        f"free={sum(model.free_count())}"
    )

    for iter_n in range(1, args.max_iters + 1):
        peaks: list[tuple[str, float, int]] = []
        with torch.no_grad():
            free_w1 = model.W1_float.data[~model.W1_frozen_mask].detach().cpu().numpy()
            free_w2 = model.W2_float.data[~model.W2_frozen_mask].detach().cpu().numpy()
        peaks.extend(("W1", v, m) for v, m in kde_peaks(free_w1, args.bandwidth, args.grid_n, args.min_mass, args.mass_radius))
        peaks.extend(("W2", v, m) for v, m in kde_peaks(free_w2, args.bandwidth, args.grid_n, args.min_mass, args.mass_radius))
        peaks.sort(key=lambda row: -row[2])

        if not peaks:
            log(f"Iter {iter_n}: no peaks above min-mass={args.min_mass}. STOP.")
            break

        committed = False
        for mat, peak_val, peak_mass in peaks:
            cb_size = len(model.codebook_W1) if mat == "W1" else len(model.codebook_W2)
            if cb_size >= args.max_codebook_size:
                continue

            round_base = copy.deepcopy(model)
            new_idx = model.add_codebook_entry(mat, float(peak_val))
            candidates = candidate_cells_near(model, mat, float(peak_val), args.peak_radius)
            accepted_cells = 0

            for cell in candidates:
                saved = set_lookup_cell(model, cell, new_idx)
                stats = eval_stats(model, data)
                row = {
                    "iter": iter_n,
                    "matrix": mat,
                    "peak_value": float(peak_val),
                    "peak_mass": int(peak_mass),
                    "cell": [cell[1], cell[2]],
                    "exact": bool(stats["exact"]),
                    "lossless": float(stats["lossless"]),
                    "bad_pairs": int(stats["bad_pairs"]),
                    "mse": float(stats["mse"]),
                }
                with open(telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if stats["exact"]:
                    accepted_cells += 1
                    continue
                rollback_lookup_cell(model, saved)

            if accepted_cells == 0:
                model = round_base
                continue

            retrained = train_lbfgs_plateau(
                model,
                data,
                max_outer=args.retrain_outer,
                patience=args.retrain_patience,
                print_every=10,
                tag=f"lookup-it{iter_n}",
                log=log,
            )
            if not retrained["exact"]:
                model = round_base
                log(
                    f"Iter {iter_n}: rollback peak {mat}@{peak_val:+.4f} "
                    f"after non-exact refit ({retrained['lossless']:.4f}%)."
                )
                continue

            free_now = sum(model.free_count())
            entry = {
                "iter": iter_n,
                "matrix": mat,
                "peak_value": float(peak_val),
                "peak_mass": int(peak_mass),
                "accepted": int(accepted_cells),
                "codebook_size_W1": int(len(model.codebook_W1)),
                "codebook_size_W2": int(len(model.codebook_W2)),
                "free_after": int(free_now),
                "mse": float(retrained["mse"]),
            }
            history.append(entry)
            log(
                f"Iter {iter_n}: ACCEPT peak {mat}@{peak_val:+.4f} "
                f"mass={peak_mass} accepted={accepted_cells} free={free_now}"
            )
            committed = True
            torch.save(
                {
                    "iter": iter_n,
                    "history": history,
                    "state_dict": model.state_dict(),
                },
                out_dir / "checkpoint.pt",
            )
            break

        if not committed:
            log(f"Iter {iter_n}: no exact-committable peak found. STOP.")
            break

    final_stats = eval_stats(model, data)
    elapsed = time.time() - t0
    if not final_stats["exact"]:
        raise SystemExit("Exact lookup stage ended non-exact; refusing to export.")

    export_final_json(
        out_dir / "final_model.json",
        args.source,
        model,
        history,
        final_stats,
        elapsed,
    )

    log("\n=== DONE ===")
    log(f"Iters: {len(history)}")
    log(f"Accepted total: {sum(int(row['accepted']) for row in history)}")
    log(
        f"Final: lossless={final_stats['lossless']:.4f}% "
        f"bad_pairs={final_stats['bad_pairs']} "
        f"free={sum(model.free_count())}"
    )
    log(f"Saved: {out_dir / 'final_model.json'}")


if __name__ == "__main__":
    main()
