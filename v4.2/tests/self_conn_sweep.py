"""Self-connection sweep: fix values + learnable.

self_conn has NEVER been tested. Currently hardcoded at 0.1.
Test: 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, and learnable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg
from scipy import sparse

SEEDS = [42, 77, 123]
BUDGET = 32000

NET_CONFIGS = [
    ("V64_N192",  64, 192, 0.06, 0.5),
    ("V128_N384", 128, 384, 0.06, 0.5),
]

MODES = [
    ("fix_0.00", 0.00, False),
    ("fix_0.05", 0.05, False),
    ("fix_0.10", 0.10, False),
    ("fix_0.20", 0.20, False),
    ("fix_0.50", 0.50, False),
    ("fix_1.00", 1.00, False),
    ("learnable", 0.10, True),
]


def run_one(net_name, V, N, density, threshold, mode_name, sc_init, learnable, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)
    sc = sc_init

    def eval_b(self_conn):
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * np.float32(self_conn)
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
            charges *= np.float32(net.leak)
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    _, score = eval_b(sc)
    best_acc = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak
        sc_s = sc

        net.mutate_with_mood()
        if learnable and random.random() < 0.2:
            sc = np.clip(sc + random.gauss(0, 0.05), 0.0, 2.0)

        a, s = eval_b(sc)
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s
            sc = sc_s

    log_msg(log_q, f"{net_name:12s} {mode_name:10s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% leak={net.leak:.3f} sc={sc:.3f}")
    return {
        'net': net_name, 'mode': mode_name, 'seed': seed,
        'acc': best_acc, 'leak': net.leak, 'sc': sc,
    }


def main():
    jobs = []
    for net_name, V, N, d, th in NET_CONFIGS:
        for mode_name, sc_init, learnable in MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, mode_name, sc_init, learnable, seed))

    total = len(jobs)
    print(f"SELF-CONNECTION SWEEP: {total} jobs, 22 workers, 32K budget", flush=True)
    print("=" * 90, flush=True)

    all_results = []
    with live_log('self_conn_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary
    nets = [c[0] for c in NET_CONFIGS]
    modes = [m[0] for m in MODES]
    print(f"\n{'='*90}", flush=True)
    print("ACCURACY:", flush=True)
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

    # Learnable convergence
    print(f"\nLEARNABLE SELF_CONN CONVERGENCE:", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='learnable']
        if runs:
            scs = [r['sc'] for r in runs]
            print(f"  {net:12s}: sc={np.mean(scs):.3f} +/-{np.std(scs):.3f} "
                  f"({', '.join(f'{s:.3f}' for s in scs)})", flush=True)

    print(f"{'='*90}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
