"""Threshold sweep on the current graph API.

Compare fixed uniform theta values against learnable-theta runs that start from
the same initial threshold.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from lib.log import live_log, log_msg
from tests.harness import ParameterSweepConfig, build_sweep_net, mutate_structure, run_parameter_search

V = 64; N = 192
DEFAULT_BUDGET = int(os.getenv("VRX_SWEEP_BUDGET", "16000"))
DEFAULT_SEEDS = [42, 77, 123]
DEFAULT_WORKERS = int(os.getenv("VRX_SWEEP_WORKERS", "20"))
THETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80]


def parse_float_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-values", default=",".join(f"{v:.2f}" for v in THETA_VALUES))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--log-prefix", default="dynamic_threshold")
    return ap.parse_args()


def run_one(mode, theta_init, seed, budget, log_q=None):
    config = ParameterSweepConfig(
        vocab=V,
        neurons=N,
        density=0.06,
        threshold=theta_init,
        ticks=8,
        budget=budget,
    )
    net, perm = build_sweep_net(config, seed)

    def propose(net, context):
        mutate_structure(net)
        if mode == "learnable" and random.random() < 0.35:
            net.mutate(forced_op="theta")

    outcome = run_parameter_search(net, perm, config, propose_fn=propose)
    label = f"{mode}_theta={theta_init:.2f}"
    log_msg(
        log_q,
        f"{label:20s} seed={seed:3d} acc={outcome.best_acc*100:5.1f}% theta={net.threshold:.3f}",
    )
    return {
        'mode': mode,
        'theta_init': theta_init,
        'seed': seed,
        'acc': outcome.best_acc,
        'theta_final': net.threshold,
    }


def main():
    args = parse_args()
    theta_values = parse_float_csv(args.theta_values)
    seeds = parse_int_csv(args.seeds)

    # Fixed-vs-learnable theta sweep
    jobs = []
    for theta_init in theta_values:
        for seed in seeds:
            jobs.append(('fixed', theta_init, seed))
            jobs.append(('learnable', theta_init, seed))

    total = len(jobs)
    print(f"THETA SWEEP: {total} jobs ({len(theta_values)} theta values x 2 modes x {len(seeds)} seeds)", flush=True)
    print(f"theta init values: {theta_values}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with live_log(args.log_prefix) as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, m, c, s, args.budget, log_q) for m, c, s in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    groups = defaultdict(list)
    for r in all_results:
        key = f"{r['mode']}_theta={r['theta_init']:.2f}"
        groups[key].append(r['acc'])

    print(f"\n{'='*70}", flush=True)
    print("RESULTS:", flush=True)
    ranked = [(k, np.mean(v)*100, np.std(v)*100) for k, v in groups.items()]
    ranked.sort(key=lambda x: -x[1])
    for name, mean, std in ranked:
        marker = ' <<<' if name == ranked[0][0] else ''
        print(f"  {name:20s}  acc={mean:5.1f}% +/-{std:.1f}%{marker}", flush=True)
    print("\nFINAL THETA MEANS:", flush=True)
    theta_groups = defaultdict(list)
    for row in all_results:
        theta_groups[f"{row['mode']}_theta={row['theta_init']:.2f}"].append(row['theta_final'])
    for name, vals in sorted(theta_groups.items()):
        print(f"  {name:20s} theta={np.mean(vals):.3f} +/-{np.std(vals):.3f}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
