"""Deterministic charge-rate sweep for current v4.2 semantics.

Supports both:
  - coarse fixed+learnable sweeps
  - focused holdout sweeps on selected configs/rates

Current v4.2 semantics:
  - ternary mask in {-1, 0, +1}
  - fixed gain = 2.0
  - no separate weight matrix
  - score = 0.5 * accuracy + 0.5 * target_probability
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from model.graph import SelfWiringGraph


GAIN = 2.0
DEFAULT_SEEDS = [42, 77, 123]
DEFAULT_BUDGET = int(os.getenv("VRX_SWEEP_BUDGET", "32000"))
DEFAULT_WORKERS = int(os.getenv("VRX_SWEEP_WORKERS", "22"))

CONFIG_CATALOG = {
    "V16_N80": (16, 80, 0.06, 0.5),
    "V64_N192": (64, 192, 0.06, 0.5),
    "V64_dense": (64, 192, 0.15, 0.5),
    "V64_sparse": (64, 192, 0.02, 0.5),
    "V128_N384": (128, 384, 0.06, 0.5),
    "V128_dense": (128, 384, 0.15, 0.5),
}

DEFAULT_CONFIGS = [
    "V16_N80",
    "V64_N192",
    "V64_dense",
    "V64_sparse",
    "V128_N384",
]

DEFAULT_FIXED_RATES = [0.10, 0.20, 0.30, 0.50, 0.70]
DEFAULT_LEARNABLE_INIT = 0.30


def parse_float_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help=f"Comma-separated config names from: {', '.join(CONFIG_CATALOG)}",
    )
    ap.add_argument(
        "--fixed-rates",
        default=",".join(f"{x:.2f}" for x in DEFAULT_FIXED_RATES),
        help="Comma-separated fixed charge_rate candidates.",
    )
    ap.add_argument(
        "--include-learnable",
        action="store_true",
        help="Include the learnable charge_rate control run.",
    )
    ap.add_argument(
        "--learnable-init",
        type=float,
        default=DEFAULT_LEARNABLE_INIT,
        help="Initial charge_rate for the learnable control.",
    )
    ap.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds.",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Attempts per job.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Process workers.",
    )
    ap.add_argument(
        "--log-prefix",
        default="charge_rate_sweep",
        help="Base name used by live_log() for the output log.",
    )
    return ap.parse_args()


def make_modes(fixed_rates: list[float], include_learnable: bool, learnable_init: float):
    modes = []
    for rate in fixed_rates:
        modes.append((f"fix_{rate:.2f}", rate, False))
    if include_learnable:
        modes.append(("learnable", learnable_init, True))
    return modes


def run_one(
    net_name,
    V,
    N,
    density,
    threshold,
    mode_name,
    cr_init,
    learnable,
    seed,
    budget,
    log_q=None,
):
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)
    leak = 0.85
    cr = cr_init

    def eval_b(lk, c_rate):
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            weff = mask_f * np.float32(GAIN)
            raw = acts @ weff + acts * net.self_conn
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * np.float32(c_rate)
            charges *= np.float32(lk)
            acts = np.maximum(charges - net.threshold, 0.0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start : net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    _, score = eval_b(leak, cr)
    best_acc = 0.0
    for _ in range(budget):
        sm = net.mask.copy()
        mx_s = net.mood_x
        mz_s = net.mood_z
        lk_s = leak
        cr_s = cr

        net.mutate_with_mood()
        if random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)
        if learnable and random.random() < 0.2:
            cr = np.clip(cr + random.gauss(0, 0.03), 0.01, 1.0)

        a, s = eval_b(leak, cr)
        if s > score:
            score = s
            best_acc = max(best_acc, a)
        else:
            net.mask = sm
            net.mood_x = mx_s
            net.mood_z = mz_s
            leak = lk_s
            cr = cr_s

    log_msg(
        log_q,
        f"{net_name:12s} {mode_name:10s} seed={seed:3d} "
        f"acc={best_acc*100:5.1f}% leak={leak:.3f} cr={cr:.3f}",
    )
    return {
        "net": net_name,
        "mode": mode_name,
        "seed": seed,
        "acc": best_acc,
        "leak": leak,
        "cr": cr,
    }


def summarize_results(all_results, config_names, mode_names):
    print(f"\n{'='*95}", flush=True)
    print(f"  {'':12s}", end="")
    for mode in mode_names:
        print(f" {mode:>10s}", end="")
    print()
    for net in config_names:
        print(f"  {net:12s}", end="")
        for mode in mode_names:
            runs = [r for r in all_results if r["net"] == net and r["mode"] == mode]
            if runs:
                mean = np.mean([r["acc"] for r in runs]) * 100.0
                print(f" {mean:9.1f}%", end="")
            else:
                print(f" {'--':>10s}", end="")
        print()

    learnable_rows = [r for r in all_results if r["mode"] == "learnable"]
    if learnable_rows:
        print("\nLEARNABLE CR CONVERGENCE:", flush=True)
        for net in config_names:
            runs = [r for r in learnable_rows if r["net"] == net]
            if not runs:
                continue
            crs = [r["cr"] for r in runs]
            lks = [r["leak"] for r in runs]
            print(
                f"  {net:12s}: cr={np.mean(crs):.3f} +/-{np.std(crs):.3f} "
                f"leak={np.mean(lks):.3f}",
                flush=True,
            )
    print(f"{'='*95}", flush=True)


def main():
    args = parse_args()
    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    fixed_rates = parse_float_csv(args.fixed_rates)
    seeds = parse_int_csv(args.seeds)
    modes = make_modes(fixed_rates, args.include_learnable, args.learnable_init)

    jobs = []
    for name in config_names:
        if name not in CONFIG_CATALOG:
            raise SystemExit(f"Unknown config: {name}")
        V, N, density, threshold = CONFIG_CATALOG[name]
        for mode_name, cr_init, learnable in modes:
            for seed in seeds:
                jobs.append(
                    (
                        name,
                        V,
                        N,
                        density,
                        threshold,
                        mode_name,
                        cr_init,
                        learnable,
                        seed,
                        args.budget,
                    )
                )

    print(
        f"CHARGE RATE SWEEP: {len(jobs)} jobs, {args.workers} workers, "
        f"{args.budget} budget, configs={config_names}, "
        f"fixed_rates={fixed_rates}, include_learnable={args.include_learnable}",
        flush=True,
    )
    print("=" * 95, flush=True)

    all_results = []
    with live_log(args.log_prefix) as (log_q, log_path):
        log_msg(log_q, f"Starting {len(jobs)} jobs")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")
        log_msg(log_q, f"Log path: {log_path}")

    summarize_results(all_results, config_names, [m[0] for m in modes])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
