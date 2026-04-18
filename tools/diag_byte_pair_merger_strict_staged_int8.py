"""L1 Merger — strict staged int8 diagnostic.

Exact-only staged freeze:

  1. Start from an exact 100% lookup-codebook winner.
  2. Re-plateau the float degrees of freedom if needed.
  3. Build a shortlist of the globally easiest remaining float cells by
     quantization error to int8 * alpha_free.
  4. For each shortlist candidate:
       - snap one cell to one int8 value
       - short LBFGS refit
       - require exact 100.0000% lossless
  5. Keep only the globally best accepted single-cell move.
  6. Immediately full-refit / plateau.
  7. Repeat until no exact single-cell move is found.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_free_int8 import (  # noqa: E402
    DEVICE,
    FreeInt8Merger,
    candidate_ints,
    load_byte_pairs,
    load_source_json,
    retrain,
)


def clone_state(model: FreeInt8Merger) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_state(
    model: FreeInt8Merger, snapshot: dict[str, torch.Tensor]
) -> None:
    model.load_state_dict(snapshot, strict=True)


def eval_stats(
    model: FreeInt8Merger, data: torch.Tensor
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


def set_int8_cell(
    model: FreeInt8Merger, cell: tuple[str, int, int], int_v: int
) -> None:
    mat, i, j = cell
    if mat == "W1":
        model.W1_int8_mask[i, j] = True
        model.W1_int8[i, j] = float(int_v)
        model.W1_float.data[i, j] = 0.0
    else:
        model.W2_int8_mask[i, j] = True
        model.W2_int8[i, j] = float(int_v)
        model.W2_float.data[i, j] = 0.0


def still_free_cells(model: FreeInt8Merger) -> int:
    free_w1, free_w2 = model.still_free_count()
    return free_w1 + free_w2


def build_shortlist(
    model: FreeInt8Merger, top_k: int
) -> list[dict[str, float | int | str]]:
    with torch.no_grad():
        alpha1 = float(model.alpha_free_W1.item())
        alpha2 = float(model.alpha_free_W2.item())

        q1 = (model.W1_float / max(alpha1, 1e-8)).round().clamp(-127, 127)
        q2 = (model.W2_float / max(alpha2, 1e-8)).round().clamp(-127, 127)
        err1 = (model.W1_float - q1 * alpha1).abs()
        err2 = (model.W2_float - q2 * alpha2).abs()

        err1[model.W1_frozen_mask | model.W1_int8_mask] = float("inf")
        err2[model.W2_frozen_mask | model.W2_int8_mask] = float("inf")

        items: list[dict[str, float | int | str]] = []
        for i, j in torch.nonzero(~torch.isinf(err1), as_tuple=False).tolist():
            w = float(model.W1_float.data[i, j].item())
            nearest = int(round(w / max(alpha1, 1e-8)))
            nearest = max(-127, min(127, nearest))
            items.append(
                {
                    "mat": "W1",
                    "i": i,
                    "j": j,
                    "w": w,
                    "alpha": alpha1,
                    "nearest": nearest,
                    "err": float(err1[i, j].item()),
                }
            )
        for i, j in torch.nonzero(~torch.isinf(err2), as_tuple=False).tolist():
            w = float(model.W2_float.data[i, j].item())
            nearest = int(round(w / max(alpha2, 1e-8)))
            nearest = max(-127, min(127, nearest))
            items.append(
                {
                    "mat": "W2",
                    "i": i,
                    "j": j,
                    "w": w,
                    "alpha": alpha2,
                    "nearest": nearest,
                    "err": float(err2[i, j].item()),
                }
            )

    items.sort(key=lambda row: (row["err"], abs(row["w"])))
    return items[:top_k]


def save_checkpoint(
    path: Path,
    model: FreeInt8Merger,
    history: list[dict[str, float | int | str | bool]],
    rounds_run: int,
) -> None:
    artifact = {
        "history": history,
        "rounds_run": rounds_run,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(artifact, path)


def export_final_json(
    out_path: Path,
    source: str,
    model: FreeInt8Merger,
    history: list[dict[str, float | int | str | bool]],
    stats: dict[str, float | int | bool],
    time_s: float,
) -> None:
    free_w1, free_w2 = model.still_free_count()
    n_int8_w1 = int(model.W1_int8_mask.sum().item())
    n_int8_w2 = int(model.W2_int8_mask.sum().item())
    lookup_frozen = int(
        model.W1_frozen_mask.sum().item() + model.W2_frozen_mask.sum().item()
    )

    cb1_sz = len(model.codebook_W1)
    cb2_sz = len(model.codebook_W2)
    bits_w1 = max(1, int(np.ceil(np.log2(max(cb1_sz, 2)))))
    bits_w2 = max(1, int(np.ceil(np.log2(max(cb2_sz, 2)))))
    idx_bytes_w1 = int(np.ceil(model.W1_idx.numel() * bits_w1 / 8))
    idx_bytes_w2 = int(np.ceil(model.W2_idx.numel() * bits_w2 / 8))
    cb_bytes = (cb1_sz + cb2_sz) * 4
    int8_bytes = n_int8_w1 + n_int8_w2
    free_float_bytes = (free_w1 + free_w2) * 4
    mask_bytes = (model.W1_frozen_mask.numel() + model.W2_frozen_mask.numel()) // 8
    int8_mask_bytes = (model.W1_int8_mask.numel() + model.W2_int8_mask.numel()) // 8
    misc_bytes = (
        model.b1.numel()
        + model.b2.numel()
        + model.db1.numel()
        + model.db2.numel()
        + 2 * model.H
        + 2
    ) * 4
    total_bytes = (
        idx_bytes_w1
        + idx_bytes_w2
        + cb_bytes
        + int8_bytes
        + free_float_bytes
        + mask_bytes
        + int8_mask_bytes
        + misc_bytes
    )

    artifact = {
        "architecture": "strict staged int8 diagnostic",
        "source": source,
        "H": model.H,
        "in_dim": model.in_dim,
        "out_dim": model.out_dim,
        "total_cells": model.total_cells(),
        "lookup_frozen": lookup_frozen,
        "int8_frozen": n_int8_w1 + n_int8_w2,
        "still_free": free_w1 + free_w2,
        "alpha_free_W1": float(model.alpha_free_W1.item()),
        "alpha_free_W2": float(model.alpha_free_W2.item()),
        "lossless": float(stats["lossless"]),
        "per_dim": float(stats["per_dim"]),
        "mse": float(stats["mse"]),
        "bad_pairs": int(stats["bad_pairs"]),
        "bad_dims": int(stats["bad_dims"]),
        "time_s": time_s,
        "deploy_bytes_est": total_bytes,
        "rounds_run": len(history),
        "accepted_moves": history,
        "codebook_W1": model.codebook_W1.data.cpu().numpy().tolist(),
        "codebook_W2": model.codebook_W2.data.cpu().numpy().tolist(),
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--candidate-width", type=int, default=2)
    parser.add_argument("--short-retrain-iters", type=int, default=10)
    parser.add_argument("--short-patience", type=int, default=5)
    parser.add_argument("--full-retrain-iters", type=int, default=40)
    parser.add_argument("--full-patience", type=int, default=12)
    parser.add_argument("--max-accepts", type=int, default=20)
    parser.add_argument("--stall-rounds", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry.jsonl"
    runlog_path = out_dir / "run.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(runlog_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== STRICT STAGED INT8 DIAGNOSTIC ===")
    log(f"Source: {args.source}")
    log(f"Device: {DEVICE}")

    state = load_source_json(args.source)
    model = FreeInt8Merger(state, DEVICE).to(DEVICE)
    data = load_byte_pairs().to(DEVICE)

    start_stats = eval_stats(model, data)
    log(
        "Baseline: "
        f"lossless={start_stats['lossless']:.4f}% "
        f"bad_pairs={start_stats['bad_pairs']} "
        f"free={still_free_cells(model)} "
        f"alpha_W1={model.alpha_free_W1.item():.6f} "
        f"alpha_W2={model.alpha_free_W2.item():.6f}"
    )
    if not start_stats["exact"]:
        log("Baseline not exact; running initial plateau.")
        retrain(model, data, max_iter=100, patience=25)
        start_stats = eval_stats(model, data)
        log(
            "After plateau: "
            f"lossless={start_stats['lossless']:.4f}% "
            f"bad_pairs={start_stats['bad_pairs']}"
        )

    history: list[dict[str, float | int | str | bool]] = []
    t0 = time.time()
    no_progress_rounds = 0
    rounds_run = 0

    while len(history) < args.max_accepts:
        rounds_run += 1
        baseline_snapshot = clone_state(model)
        shortlist = build_shortlist(model, args.top_k)
        if not shortlist:
            log("No remaining free cells.")
            break

        best_move = None
        best_stats = None

        for rank, row in enumerate(shortlist, start=1):
            cell = (str(row["mat"]), int(row["i"]), int(row["j"]))
            w = float(row["w"])
            alpha = float(row["alpha"])
            for int_v in candidate_ints(w, alpha, width=args.candidate_width):
                restore_state(model, baseline_snapshot)
                set_int8_cell(model, cell, int_v)
                retrain(
                    model,
                    data,
                    max_iter=args.short_retrain_iters,
                    patience=args.short_patience,
                )
                stats = eval_stats(model, data)
                trial = {
                    "round": rounds_run,
                    "rank": rank,
                    "mat": cell[0],
                    "i": cell[1],
                    "j": cell[2],
                    "w": w,
                    "alpha": alpha,
                    "err": float(row["err"]),
                    "nearest": int(row["nearest"]),
                    "int_v": int(int_v),
                    "exact": bool(stats["exact"]),
                    "lossless": float(stats["lossless"]),
                    "bad_pairs": int(stats["bad_pairs"]),
                    "mse": float(stats["mse"]),
                }
                with open(telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trial) + "\n")

                if not stats["exact"]:
                    continue

                if best_move is None or (
                    stats["mse"],
                    row["err"],
                    abs(int_v - int(row["nearest"])),
                    rank,
                ) < (
                    best_stats["mse"],
                    best_move["err"],
                    abs(int(best_move["int_v"]) - int(best_move["nearest"])),
                    int(best_move["rank"]),
                ):
                    best_move = trial
                    best_stats = stats

        restore_state(model, baseline_snapshot)

        if best_move is None:
            no_progress_rounds += 1
            log(
                f"Round {rounds_run}: no exact single-cell move found in top-{args.top_k}; "
                f"stall={no_progress_rounds}/{args.stall_rounds}"
            )
            if no_progress_rounds >= args.stall_rounds:
                break
            continue

        no_progress_rounds = 0
        chosen_cell = (
            str(best_move["mat"]),
            int(best_move["i"]),
            int(best_move["j"]),
        )
        set_int8_cell(model, chosen_cell, int(best_move["int_v"]))
        retrain(
            model,
            data,
            max_iter=args.full_retrain_iters,
            patience=args.full_patience,
        )
        post_stats = eval_stats(model, data)
        if not post_stats["exact"]:
            restore_state(model, baseline_snapshot)
            log(
                f"Round {rounds_run}: best short candidate failed full plateau "
                f"({post_stats['lossless']:.4f}%, bad_pairs={post_stats['bad_pairs']}). Stopping."
            )
            break

        free_now = still_free_cells(model)
        accept_row = {
            **best_move,
            "post_lossless": float(post_stats["lossless"]),
            "post_bad_pairs": int(post_stats["bad_pairs"]),
            "post_mse": float(post_stats["mse"]),
            "free_after": free_now,
        }
        history.append(accept_row)
        save_checkpoint(out_dir / "checkpoint.pt", model, history, rounds_run)
        log(
            f"Round {rounds_run}: ACCEPT {chosen_cell[0]}[{chosen_cell[1]},{chosen_cell[2]}] "
            f"-> {int(best_move['int_v'])} "
            f"(rank={best_move['rank']}, err={best_move['err']:.6f}, mse={post_stats['mse']:.3e}) "
            f"free={free_now}"
        )

    final_stats = eval_stats(model, data)
    elapsed = time.time() - t0
    log(
        "\n=== DONE ===\n"
        f"Accepted: {len(history)}\n"
        f"Still free: {still_free_cells(model)}\n"
        f"Lossless: {final_stats['lossless']:.4f}%\n"
        f"Bad pairs: {final_stats['bad_pairs']}\n"
        f"MSE: {final_stats['mse']:.6e}\n"
        f"Time: {elapsed:.0f}s"
    )
    export_final_json(
        out_dir / "final_model.json",
        args.source,
        model,
        history,
        final_stats,
        elapsed,
    )

    if not final_stats["exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
