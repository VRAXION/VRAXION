"""GPU initial-density sweep with plateau-oriented early stop.

The goal is to stop when the search looks practically frozen, not merely when
an arbitrary attempt budget is exhausted. We therefore watch checkpointed
trajectories and stop only when the committed state shows negligible drift for
multiple consecutive windows. A large safety cap remains only as a failsafe.

Operational plateau definition:
  - checkpoint every ``CHECKPOINT_EVERY`` attempts
  - look at a rolling window of the most recent checkpoints
  - if best score, density, and accepted-mutation count all drift only within
    small tolerances for ``PLATEAU_PATIENCE`` consecutive windows
  - and at least ``MIN_ATTEMPTS`` have elapsed
  - then we stop and call that a practical plateau
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from tests.gpu_full_evo_prototype import (
    BenchConfig,
    gpu_init_from_cpu,
    make_eval_runner,
    mutate_gpu_reference,
    rollback_gpu_reference,
)


SEEDS = [42, 77]
SAFETY_CAP_ATTEMPTS = 200000
MIN_ATTEMPTS = 4000
CHECKPOINT_EVERY = 1000
PLATEAU_WINDOW = 4
PLATEAU_PATIENCE = 3
DENSITY_EPS = 0.0015
SCORE_EPS = 0.0010
ACCEPT_EPS = 48

CONFIGS = [
    ("V64_N192", 64, 192),
    ("V128_N384", 128, 384),
    ("V256_N768", 256, 768),
]

INITIAL_DENSITIES = [0.000, 0.005, 0.010, 0.020, 0.040, 0.060, 0.100, 0.150]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all", help="Comma-separated config names or 'all'")
    ap.add_argument("--densities", default="all", help="Comma-separated densities or 'all'")
    ap.add_argument("--seeds", default="all", help="Comma-separated seeds or 'all'")
    ap.add_argument("--safety-cap", type=int, default=SAFETY_CAP_ATTEMPTS)
    ap.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--plateau-window", type=int, default=PLATEAU_WINDOW)
    ap.add_argument("--plateau-patience", type=int, default=PLATEAU_PATIENCE)
    ap.add_argument("--density-eps", type=float, default=DENSITY_EPS)
    ap.add_argument("--score-eps", type=float, default=SCORE_EPS)
    ap.add_argument("--accept-eps", type=int, default=ACCEPT_EPS)
    return ap.parse_args()


def mask_density(mask: torch.Tensor) -> float:
    n = mask.shape[0]
    total = n * n - n
    return float((mask != 0).sum().item()) / float(total)


def window_frozen(history, window: int, density_eps: float, score_eps: float, accept_eps: int) -> bool:
    if len(history) < window:
        return False
    chunk = history[-window:]
    density_delta = abs(chunk[-1]["density"] - chunk[0]["density"])
    score_delta = abs(chunk[-1]["best_score"] - chunk[0]["best_score"])
    accept_delta = chunk[-1]["accepted"] - chunk[0]["accepted"]
    return density_delta <= density_eps and score_delta <= score_eps and accept_delta <= accept_eps


def run_one(
    name: str,
    vocab: int,
    neurons: int,
    density: float,
    seed: int,
    safety_cap: int,
    min_attempts: int,
    checkpoint_every: int,
    plateau_window: int,
    plateau_patience: int,
    density_eps: float,
    score_eps: float,
    accept_eps: int,
    log_q=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    cfg = BenchConfig(name, vocab, neurons, density)
    device = torch.device("cuda")

    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, mood_x, mood_z, leak, targets, out_start = gpu_init_from_cpu(cfg, seed, device)
    diag_mask = ~torch.eye(cfg.neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(cfg, targets, out_start, device, compile_eval=False)

    init_density = mask_density(mask)
    _, score, acc = eval_runner(mask, leak)
    score = score.clone()
    best_score = score.clone()
    best_acc = acc.clone()

    accepted = 0
    checkpoint_history = [
        {"att": 0, "density": init_density, "best_score": float(best_score.item()), "accepted": 0}
    ]
    frozen_windows = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stop_reason = "safety_cap"
    stop_att = safety_cap

    for att in range(1, safety_cap + 1):
        delta = mutate_gpu_reference(mask, mood_x, mood_z, leak, gen, diag_mask)
        _, new_score, new_acc = eval_runner(mask, leak)

        if bool((new_score > score).item()):
            score = new_score
            accepted += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_gpu_reference(mask, mood_x, mood_z, leak, delta)

        if att % checkpoint_every == 0:
            density_now = mask_density(mask)
            point = {
                "att": att,
                "density": density_now,
                "best_score": float(best_score.item()),
                "accepted": accepted,
            }
            checkpoint_history.append(point)
            frozen = window_frozen(checkpoint_history, plateau_window, density_eps, score_eps, accept_eps)
            frozen_windows = frozen_windows + 1 if frozen else 0
            log_msg(
                log_q,
                f"{name:10s} init={density:0.3f} seed={seed:3d} att={att:5d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} "
                f"density={density_now:0.4f} leak={float(leak.item()):0.3f} "
                f"accepted={accepted:4d} frozen_win={frozen_windows:2d}",
            )

            if att >= min_attempts and frozen_windows >= plateau_patience:
                stop_reason = "plateau"
                stop_att = att
                break

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    final_density = mask_density(mask)
    result = {
        "config": name,
        "seed": seed,
        "init_density": density,
        "init_density_real": init_density,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "final_density": final_density,
        "final_leak": float(leak.item()),
        "accepted": accepted,
        "stop_reason": stop_reason,
        "stop_attempt": stop_att,
        "frozen_windows": frozen_windows,
        "attempts_per_sec": stop_att / dt if dt > 0 else float("inf"),
        "checkpoint_history": checkpoint_history,
    }
    log_msg(
        log_q,
        f"{name:10s} init={density:0.3f} seed={seed:3d} "
        f"FINAL acc={result['best_acc']*100:5.1f}% score={result['best_score']:.4f} "
        f"density={init_density:0.4f}->{final_density:0.4f} leak={result['final_leak']:0.3f} "
        f"accepted={accepted:4d} stop={stop_reason}@{stop_att} frozen_win={frozen_windows:2d} "
        f"aps={result['attempts_per_sec']:.1f}",
    )
    return result


def print_summary(results, args):
    configs = sorted({r["config"] for r in results})
    print("\n" + "=" * 110, flush=True)
    print(
        f"PLATEAU RULE: MIN_ATTEMPTS={args.min_attempts}, CHECKPOINT_EVERY={args.checkpoint_every}, "
        f"WINDOW={args.plateau_window}, PATIENCE={args.plateau_patience}, "
        f"DENSITY_EPS={args.density_eps}, SCORE_EPS={args.score_eps}, ACCEPT_EPS={args.accept_eps}",
        flush=True,
    )

    for cfg_name in configs:
        rows = [r for r in results if r["config"] == cfg_name]
        print(f"\n--- {cfg_name} SUMMARY ---", flush=True)
        print(
        f"  {'init_d':>7s} {'acc_mean':>9s} {'acc_best':>9s} {'final_d':>9s} "
        f"{'stop_att':>9s} {'aps':>8s} {'plateau%':>9s}",
            flush=True,
        )
        by_density = {}
        for row in rows:
            by_density.setdefault(row["init_density"], []).append(row)
        ranked = []
        for density in sorted(by_density):
            group = by_density[density]
            acc_mean = statistics.mean(r["best_acc"] for r in group) * 100.0
            acc_best = max(r["best_acc"] for r in group) * 100.0
            final_d = statistics.mean(r["final_density"] for r in group)
            stop_att = statistics.mean(r["stop_attempt"] for r in group)
            aps = statistics.mean(r["attempts_per_sec"] for r in group)
            plateau_rate = 100.0 * statistics.mean(1.0 if r["stop_reason"] == "plateau" else 0.0 for r in group)
            ranked.append((density, acc_mean, acc_best, final_d, stop_att, aps, plateau_rate))
            print(
                f"  {density:7.3f} {acc_mean:9.1f} {acc_best:9.1f} {final_d:9.4f} "
                f"{stop_att:9.0f} {aps:8.1f} {plateau_rate:8.1f}%",
                flush=True,
            )

        winner = max(ranked, key=lambda x: x[1])
        print(
            f"  BEST mean acc: init_density={winner[0]:0.3f} "
            f"acc={winner[1]:.1f}% final_density={winner[3]:0.4f} "
            f"mean_stop={winner[4]:.0f}",
            flush=True,
        )


def parse_subset(args):
    cfg_map = {name: (name, vocab, neurons) for name, vocab, neurons in CONFIGS}
    if args.configs == "all":
        cfgs = CONFIGS
    else:
        names = [x.strip() for x in args.configs.split(",") if x.strip()]
        cfgs = [cfg_map[n] for n in names]

    if args.densities == "all":
        densities = INITIAL_DENSITIES
    else:
        densities = [float(x.strip()) for x in args.densities.split(",") if x.strip()]

    if args.seeds == "all":
        seeds = SEEDS
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return cfgs, densities, seeds


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.", flush=True)
        return 1

    args = parse_args()
    configs, densities, seeds = parse_subset(args)

    total = len(configs) * len(densities) * len(seeds)
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")

    print("GPU INITIAL-DENSITY PLATEAU SWEEP", flush=True)
    print(
        f"configs={len(configs)} densities={len(densities)} seeds={len(seeds)} "
        f"total_runs={total}",
        flush=True,
    )
    print(
        f"safety_cap={args.safety_cap} min_attempts={args.min_attempts} "
        f"window={args.plateau_window} patience={args.plateau_patience}",
        flush=True,
    )
    print("=" * 110, flush=True)

    results = []
    with live_log("density_plateau", log_dir=log_dir) as (log_q, log_path):
        log_msg(log_q, f"Starting {total} sequential GPU runs")
        for cfg_name, vocab, neurons in configs:
            for density in densities:
                for seed in seeds:
                    results.append(
                        run_one(
                            cfg_name,
                            vocab,
                            neurons,
                            density,
                            seed,
                            args.safety_cap,
                            args.min_attempts,
                            args.checkpoint_every,
                            args.plateau_window,
                            args.plateau_patience,
                            args.density_eps,
                            args.score_eps,
                            args.accept_eps,
                            log_q=log_q,
                        )
                    )
        log_msg(log_q, "Done")

    print_summary(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
