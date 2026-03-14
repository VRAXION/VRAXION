"""Computed leak from density vs fix vs learnable — parallel."""
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
GAIN = 2.0
K = 0.46  # fitted from data

NET_CONFIGS = [
    ("V8_N32",     8,   32, 0.06, 0.5,  8000),
    ("V16_N80",   16,   80, 0.06, 0.5,  8000),
    ("V64_N192",  64,  192, 0.06, 0.5, 16000),
    ("V64_dense", 64,  192, 0.15, 0.5, 16000),
    ("V64_sparse",64,  192, 0.02, 0.5, 16000),
    ("V64_th02",  64,  192, 0.06, 0.2, 16000),
    ("V128_N384",128,  384, 0.06, 0.5, 16000),
]

MODES = [
    ("fix_best", False),     # best static per config (hardcoded from mega sweep)
    ("computed", False),     # leak = f(density) realtime
    ("learnable", True),     # co-evolved
]

BEST_STATIC = {
    "V8_N32": 0.99, "V16_N80": 0.99, "V64_N192": 0.99,
    "V64_dense": 0.95, "V64_sparse": 0.99, "V64_th02": 0.85, "V128_N384": 0.99,
}


def compute_leak(mask, N):
    density = np.count_nonzero(mask) / (N * N)
    return max(0.80, min(0.99, 1.0 - density * K))


def run_one(net_name, V, N, density, threshold, budget, mode_name, learnable, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)

    if mode_name == "fix_best":
        leak = BEST_STATIC[net_name]
    elif mode_name == "computed":
        leak = compute_leak(net.mask, N)
    else:
        leak = 0.85

    def eval_b(lk):
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            Weff = mask_f * np.float32(GAIN)
            raw = acts @ Weff + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
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

    _, score = eval_b(leak)
    best_acc = 0.0
    for att in range(budget):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = leak
        net.mutate_with_mood()

        if mode_name == "computed":
            leak = compute_leak(net.mask, N)
        elif learnable and random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)

        a, s = eval_b(leak)
        if s > score: score = s; best_acc = max(best_acc, a)
        else: net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; leak = lk_s

    log_msg(log_q, f"{net_name:12s} {mode_name:10s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={leak:.3f}")
    return {'net': net_name, 'mode': mode_name, 'seed': seed,
            'acc': best_acc, 'leak': leak}


def main():
    jobs = []
    for net_name, V, N, d, th, budget in NET_CONFIGS:
        for mode_name, learnable in MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, budget, mode_name, learnable, seed))

    total = len(jobs)
    print(f"COMPUTED LEAK TEST: {total} jobs, 20 workers", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with live_log('computed_leak') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    print(f"\n{'='*70}", flush=True)
    nets = [c[0] for c in NET_CONFIGS]
    modes = [m[0] for m in MODES]
    print(f"  {'':12s}", end='')
    for m in modes: print(f" {m:>10s}", end='')
    print()
    for net in nets:
        print(f"  {net:12s}", end='')
        for m in modes:
            runs = [r['acc'] for r in all_results if r['net']==net and r['mode']==m]
            if runs: print(f" {np.mean(runs)*100:9.1f}%", end='')
            else: print(f"        -- ", end='')
        print()
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
