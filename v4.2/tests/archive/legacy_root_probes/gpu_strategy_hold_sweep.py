"""GPU microprobe for fixed-window strategy cadence.

This compares the current reject-triggered two-bit controller against
window-held variants where the slow strategy bits (`signal`, `grow`) stay fixed
for `strategy_hold_steps` attempts before they are reconsidered.

Only the strategy cadence changes:
  - `signal` / `grow` are either reject-triggered (`two_bit_random35`) or
    window-held (`two_bit_hold{N}`)
  - `intensity` still adapts per-attempt
  - `loss_pct` still mutates per-attempt and reverts with the mask on reject
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from tests.gpu_strategy_ab import retention_from_loss, rollback_two_bit
from tests.gpu_strategy_plateau import mask_density, mutate_two_bit
from tests.gpu_int_mood_ab import CONFIGS, gpu_init, make_eval_runner


DEFAULT_CANDIDATES = (
    "two_bit_random35",
    "two_bit_hold25",
    "two_bit_hold50",
    "two_bit_hold100",
    "two_bit_hold200",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384,V256_N768", help="Comma-separated config names")
    ap.add_argument("--attempts", type=int, default=16000)
    ap.add_argument("--seeds", default="42,77,123")
    ap.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidate names",
    )
    ap.add_argument("--log-name", default="gpu_strategy_hold_sweep")
    return ap.parse_args()


def parse_csv(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_candidate(name: str):
    if name == "two_bit_random35":
        return {"name": name, "mode": "random35", "strategy_hold_steps": 0}
    match = re.fullmatch(r"two_bit_hold(\d+)", name)
    if not match:
        raise ValueError(f"Unknown candidate: {name}")
    return {"name": name, "mode": "hold", "strategy_hold_steps": int(match.group(1))}


def maybe_flip_strategy(controller: dict, gen: torch.Generator, device: torch.device):
    changed = False
    if float(torch.rand((), generator=gen, device=device).item()) < 0.35:
        controller["signal"] = 1 - controller["signal"]
        changed = True
    if float(torch.rand((), generator=gen, device=device).item()) < 0.35:
        controller["grow"] = 1 - controller["grow"]
        changed = True
    return changed


def run_one(config_name: str, seed: int, attempts: int, candidate_name: str, log_q=None):
    candidate = parse_candidate(candidate_name)
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, _leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(vocab, neurons, targets, out_start, device)
    loss_pct = torch.tensor(15, device=device, dtype=torch.int16)
    controller = {"signal": 0, "grow": 1, "intensity": 7}

    score, acc = eval_runner(mask, retention_from_loss(loss_pct))
    best_score = score.clone()
    best_acc = acc.clone()
    accepted = 0
    window_changes = 0
    hold_steps = candidate["strategy_hold_steps"]
    window_start_best_score = float(best_score.item())
    window_start_accepted = accepted
    window_start_signal = controller["signal"]
    window_start_grow = controller["grow"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        prev, changes = mutate_two_bit(mask, loss_pct, controller, gen, diag_mask)
        new_score, new_acc = eval_runner(mask, retention_from_loss(loss_pct))
        if bool((new_score > score).item()):
            score = new_score
            accepted += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_two_bit(mask, loss_pct, prev, changes)
            if candidate["mode"] == "random35":
                if maybe_flip_strategy(controller, gen, mask.device):
                    window_changes += 1

        if candidate["mode"] == "hold" and att % hold_steps == 0:
            if float(best_score.item()) <= window_start_best_score:
                if maybe_flip_strategy(controller, gen, mask.device):
                    window_changes += 1
            window_start_best_score = float(best_score.item())
            window_start_accepted = accepted
            window_start_signal = controller["signal"]
            window_start_grow = controller["grow"]

        if att % 4000 == 0:
            mode = "SIGNAL" if controller["signal"] else ("GROW" if controller["grow"] else "SHRINK")
            log_msg(
                log_q,
                (
                    f"{config_name:10s} {candidate_name:16s} seed={seed:3d} att={att:5d} "
                    f"best_acc={best_acc.item()*100:5.1f}% score={best_score.item():.4f} "
                    f"density={mask_density(mask):.4f} accepted={accepted:5d} "
                    f"loss={int(loss_pct.item()):2d}% {mode:6s} int={controller['intensity']:2d} "
                    f"window_changes={window_changes:3d}"
                ),
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    result = {
        "config": config_name,
        "seed": seed,
        "candidate": candidate_name,
        "strategy_hold_steps": hold_steps,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_density": mask_density(mask),
        "final_loss_pct": int(loss_pct.item()),
        "final_signal": int(controller["signal"]),
        "final_grow": int(controller["grow"]),
        "final_intensity": int(controller["intensity"]),
        "window_changes": int(window_changes),
        "window_start_best_score": float(window_start_best_score),
        "window_start_accepted": int(window_start_accepted),
        "window_start_signal": int(window_start_signal),
        "window_start_grow": int(window_start_grow),
    }
    return result


def print_result(log_q, row):
    mode = "SIGNAL" if row["final_signal"] else ("GROW" if row["final_grow"] else "SHRINK")
    hold = row["strategy_hold_steps"] if row["strategy_hold_steps"] else "-"
    log_msg(
        log_q,
        (
            f"{row['config']:10s} {row['candidate']:16s} seed={row['seed']:3d} "
            f"acc={row['best_acc']*100:5.1f}% score={row['best_score']:.4f} "
            f"aps={row['attempts_per_sec']:.1f} hold={hold:>3} "
            f"density={row['final_density']:.4f} loss={row['final_loss_pct']:2d}% "
            f"{mode:6s} int={row['final_intensity']:2d} window_changes={row['window_changes']:3d}"
        ),
    )


def summarize(results, configs, candidates, log_q):
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")
    for config in configs:
        log_msg(log_q, f"-- {config} --")
        summary_rows = []
        for candidate in candidates:
            rows = [r for r in results if r["config"] == config and r["candidate"] == candidate]
            accs = np.array([r["best_acc"] for r in rows], dtype=np.float64)
            scores = np.array([r["best_score"] for r in rows], dtype=np.float64)
            aps = np.array([r["attempts_per_sec"] for r in rows], dtype=np.float64)
            changes = np.array([r["window_changes"] for r in rows], dtype=np.float64)
            row = {
                "candidate": candidate,
                "mean_acc": float(accs.mean()),
                "mean_score": float(scores.mean()),
                "p10_acc": float(np.percentile(accs, 10)),
                "worst_acc": float(accs.min()),
                "mean_aps": float(aps.mean()),
                "mean_changes": float(changes.mean()),
            }
            summary_rows.append(row)
        summary_rows.sort(
            key=lambda r: (
                r["mean_acc"],
                r["mean_score"],
                r["p10_acc"],
                r["mean_aps"],
            ),
            reverse=True,
        )
        for row in summary_rows:
            log_msg(
                log_q,
                (
                    f"{row['candidate']:16s} mean_acc={row['mean_acc']*100:5.1f}% "
                    f"mean_score={row['mean_score']:.4f} p10={row['p10_acc']*100:5.1f}% "
                    f"worst={row['worst_acc']*100:5.1f}% aps={row['mean_aps']:.1f} "
                    f"mean_changes={row['mean_changes']:.1f}"
                ),
            )
        if summary_rows:
            winner = summary_rows[0]["candidate"]
            log_msg(log_q, f"WINNER {config}: {winner}")
        log_msg(log_q, "")
        log_msg(log_q, f"SUMMARY_JSON {json.dumps({'config': config, 'rows': summary_rows})}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    configs = parse_csv(args.configs)
    seeds = parse_csv_ints(args.seeds)
    candidates = parse_csv(args.candidates)
    for config in configs:
        if config not in CONFIGS:
            raise SystemExit(f"Unknown config: {config}")
    for candidate in candidates:
        parse_candidate(candidate)

    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            (
                f"GPU STRATEGY HOLD SWEEP attempts={args.attempts} configs={configs} "
                f"seeds={seeds} candidates={candidates}"
            ),
        )
        log_msg(log_q, "=" * 120)
        results = []
        for config in configs:
            for candidate in candidates:
                for seed in seeds:
                    row = run_one(config, seed, args.attempts, candidate, log_q)
                    results.append(row)
                    print_result(log_q, row)
                    log_msg(log_q, f"RESULT_JSON {json.dumps(row)}")
        summarize(results, configs, candidates, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    main()
