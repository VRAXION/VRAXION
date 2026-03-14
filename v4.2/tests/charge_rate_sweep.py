"""Charge rate: fix vs learnable across different configs. Parallel.

Current v4.2 semantics:
  - ternary mask in {-1, 0, +1}
  - fixed gain = 2.0
  - no separate weight matrix
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

SEEDS = [42, 77, 123]
BUDGET = int(os.getenv("VRX_SWEEP_BUDGET", "32000"))
MAX_WORKERS = int(os.getenv("VRX_SWEEP_WORKERS", "22"))
GAIN = 2.0

NET_CONFIGS = [
    ("V16_N80",    16,   80, 0.06, 0.5),
    ("V64_N192",   64,  192, 0.06, 0.5),
    ("V64_dense",  64,  192, 0.15, 0.5),
    ("V64_sparse", 64,  192, 0.02, 0.5),
    ("V128_N384", 128,  384, 0.06, 0.5),
]

MODES = [
    ("fix_0.1",  0.1, False),
    ("fix_0.2",  0.2, False),
    ("fix_0.3",  0.3, False),
    ("fix_0.5",  0.5, False),
    ("fix_0.7",  0.7, False),
    ("learnable", 0.3, True),
]


def run_one(net_name, V, N, density, threshold, mode_name, cr_init, learnable, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
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
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            weff = mask_f * np.float32(GAIN)
            raw = acts @ weff + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * np.float32(c_rate)
            charges *= np.float32(lk)
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5*acc + 0.5*tp

    _, score = eval_b(leak, cr)
    best_acc = 0.0
    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = leak; cr_s = cr

        net.mutate_with_mood()
        if random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)
        if learnable and random.random() < 0.2:
            cr = np.clip(cr + random.gauss(0, 0.03), 0.01, 1.0)

        a, s = eval_b(leak, cr)
        if s > score: score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s
            leak = lk_s; cr = cr_s

    log_msg(log_q, f"{net_name:12s} {mode_name:10s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={leak:.3f} cr={cr:.3f}")
    return {'net': net_name, 'mode': mode_name, 'seed': seed,
            'acc': best_acc, 'leak': leak, 'cr': cr}


def main():
    jobs = []
    for net_name, V, N, d, th in NET_CONFIGS:
        for mode_name, cr_init, learnable in MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, mode_name, cr_init, learnable, seed))

    total = len(jobs)
    print(
        f"CHARGE RATE SWEEP: {total} jobs, {MAX_WORKERS} workers, "
        f"{BUDGET} budget",
        flush=True,
    )
    print("=" * 75, flush=True)

    all_results = []
    with live_log('charge_rate_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    print(f"\n{'='*75}", flush=True)
    nets = [c[0] for c in NET_CONFIGS]
    modes = [m[0] for m in MODES]
    print(f"  {'':12s}", end='')
    for m in modes: print(f" {m:>10s}", end='')
    print()
    for net in nets:
        print(f"  {net:12s}", end='')
        for m in modes:
            runs = [r for r in all_results if r['net']==net and r['mode']==m]
            if runs:
                mean = np.mean([r['acc'] for r in runs]) * 100
                print(f" {mean:9.1f}%", end='')
            else:
                print(f"       -- ", end='')
        print()

    # Learnable convergence
    print(f"\nLEARNABLE CR CONVERGENCE:", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='learnable']
        if runs:
            crs = [r['cr'] for r in runs]
            lks = [r['leak'] for r in runs]
            print(f"  {net:12s}: cr={np.mean(crs):.3f} +/-{np.std(crs):.3f}  "
                  f"leak={np.mean(lks):.3f}", flush=True)
    print(f"{'='*75}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
