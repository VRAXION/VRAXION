"""Static leak values vs learnable — wildly different configs, parallel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

SEEDS = [42, 77]
BUDGET = 16000
GAIN = 2.0

# Different network configs to test universality
NET_CONFIGS = [
    ("V8_N32",     8,   32, 0.06, 0.5,  8000),
    ("V16_N80",   16,   80, 0.06, 0.5,  8000),
    ("V64_N192",  64,  192, 0.06, 0.5, 16000),
    ("V64_dense", 64,  192, 0.15, 0.5, 16000),
    ("V64_sparse",64,  192, 0.02, 0.5, 16000),
    ("V64_th02",  64,  192, 0.06, 0.2, 16000),
    ("V128_N384",128,  384, 0.06, 0.5, 16000),
]

# Leak modes to compare
LEAK_MODES = [
    ("fix_085", 0.85, False),
    ("fix_090", 0.90, False),
    ("fix_095", 0.95, False),
    ("fix_098", 0.98, False),
    ("fix_099", 0.99, False),
    ("learnable", 0.85, True),
]


def run_one(net_name, V, N, density, threshold, budget, leak_name, leak_init, learnable, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)
    leak = leak_init

    def eval_b(lk):
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
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
        if learnable and random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)
        a, s = eval_b(leak)
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s
            leak = lk_s

    log_msg(log_q, f"{net_name:12s} {leak_name:10s} seed={seed:2d} "
            f"acc={best_acc*100:5.1f}% leak={leak:.3f}")
    return {'net': net_name, 'leak_mode': leak_name, 'seed': seed,
            'acc': best_acc, 'final_leak': leak}


def main():
    jobs = []
    for net_name, V, N, d, th, budget in NET_CONFIGS:
        for leak_name, leak_init, learnable in LEAK_MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, budget,
                             leak_name, leak_init, learnable, seed))

    total = len(jobs)
    print(f"STATIC vs LEARNABLE LEAK: {total} jobs, 22 workers", flush=True)
    print(f"Net configs: {len(NET_CONFIGS)}, Leak modes: {len(LEAK_MODES)}", flush=True)
    print("=" * 75, flush=True)

    all_results = []
    with live_log('leak_static_vs_learn') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary per net config
    print(f"\n{'='*75}", flush=True)
    for net_name, _, _, _, _, _ in NET_CONFIGS:
        print(f"\n  {net_name}:", flush=True)
        for leak_name, _, _ in LEAK_MODES:
            runs = [r for r in all_results
                    if r['net'] == net_name and r['leak_mode'] == leak_name]
            if runs:
                mean = np.mean([r['acc'] for r in runs]) * 100
                lk = np.mean([r['final_leak'] for r in runs])
                print(f"    {leak_name:10s}  acc={mean:5.1f}%  leak={lk:.3f}", flush=True)

    # Overall: learnable vs best static per net
    print(f"\n{'='*75}", flush=True)
    print("LEARNABLE vs BEST STATIC per config:", flush=True)
    for net_name, _, _, _, _, _ in NET_CONFIGS:
        learn_runs = [r['acc'] for r in all_results
                      if r['net'] == net_name and r['leak_mode'] == 'learnable']
        best_static = 0
        best_static_name = ""
        for leak_name, _, learnable in LEAK_MODES:
            if learnable:
                continue
            runs = [r['acc'] for r in all_results
                    if r['net'] == net_name and r['leak_mode'] == leak_name]
            if runs and np.mean(runs) > best_static:
                best_static = np.mean(runs)
                best_static_name = leak_name
        learn_mean = np.mean(learn_runs) * 100 if learn_runs else 0
        diff = learn_mean - best_static * 100
        print(f"  {net_name:12s}  learn={learn_mean:5.1f}%  "
              f"best_static={best_static*100:5.1f}% ({best_static_name})  "
              f"diff={diff:+.1f}pp", flush=True)
    print(f"{'='*75}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
