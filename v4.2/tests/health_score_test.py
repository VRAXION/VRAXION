"""Test: does health score as auxiliary objective improve training? Parallel.
Also: find the universal health formula across wildly different configs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg
from scipy import sparse

SEEDS = [42, 77, 123]
BUDGET = 32000

NET_CONFIGS = [
    ("V16_N80",    16,   80, 0.06, 0.5),
    ("V64_N192",   64,  192, 0.06, 0.5),
    ("V64_dense",  64,  192, 0.15, 0.5),
    ("V64_sparse", 64,  192, 0.02, 0.5),
    ("V128_N384", 128,  384, 0.06, 0.5),
]

# Modes: baseline (acc only) vs health-augmented (acc + health)
MODES = [
    ("acc_only", 0.0),        # pure accuracy
    ("health_01", 0.01),      # 1% health weight
    ("health_05", 0.05),      # 5% health weight
    ("health_10", 0.10),      # 10% health weight
    ("health_20", 0.20),      # 20% health weight
]


def compute_health(charges, threshold, clip_factor):
    """Compute health score from charge state. Free after forward pass."""
    clip_bound = threshold * clip_factor
    dead = (np.abs(charges) < 0.01).mean()
    sat = (np.abs(charges) > clip_bound * 0.95).mean()
    spread = np.std(charges)
    # Health: reward spread, penalize dead + saturated
    health = spread / (1.0 + dead + sat)
    return health, dead, sat, spread


def run_one(net_name, V, N, density, threshold, mode_name, health_weight, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=threshold)
    perm = np.random.permutation(V)
    leak = 0.85

    def eval_b(lk):
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * net.self_conn
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
        acc_score = 0.5 * acc + 0.5 * tp

        health, dead, sat, spread = compute_health(charges, net.threshold, net.clip_factor)

        if health_weight > 0:
            combined = (1 - health_weight) * acc_score + health_weight * health
        else:
            combined = acc_score

        return acc, combined, health, dead, sat, spread

    _, score, _, _, _, _ = eval_b(leak)
    best_acc = 0.0
    final_health = 0.0
    final_dead = 0.0
    final_sat = 0.0
    final_spread = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = leak

        net.mutate_with_mood()
        if random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)

        a, s, h, d, sa, sp = eval_b(leak)
        if s > score:
            score = s; best_acc = max(best_acc, a)
            final_health = h; final_dead = d; final_sat = sa; final_spread = sp
        else:
            net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s; leak = lk_s

    eff_drive = net.charge_rate / (1 - leak) if leak < 1 else 999
    log_msg(log_q, f"{net_name:12s} {mode_name:12s} seed={seed:3d} "
            f"acc={best_acc*100:5.1f}% hlth={final_health:.3f} "
            f"dead={final_dead*100:.0f}% sat={final_sat*100:.0f}% "
            f"leak={leak:.3f} drive={eff_drive:.1f}")
    return {
        'net': net_name, 'mode': mode_name, 'seed': seed,
        'acc': best_acc, 'health': final_health,
        'dead': final_dead, 'sat': final_sat, 'spread': final_spread,
        'leak': leak, 'eff_drive': eff_drive,
    }


def main():
    jobs = []
    for net_name, V, N, d, th in NET_CONFIGS:
        for mode_name, hw in MODES:
            for seed in SEEDS:
                jobs.append((net_name, V, N, d, th, mode_name, hw, seed))

    total = len(jobs)
    print(f"HEALTH SCORE TEST: {total} jobs, 22 workers, 32K budget", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    with live_log('health_score') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, *j, log_q) for j in jobs]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Summary: acc per config per mode
    print(f"\n{'='*80}", flush=True)
    print("ACCURACY:", flush=True)
    nets = [c[0] for c in NET_CONFIGS]
    modes = [m[0] for m in MODES]
    print(f"  {'':12s}", end='')
    for m in modes: print(f" {m:>12s}", end='')
    print()
    for net in nets:
        print(f"  {net:12s}", end='')
        for m in modes:
            runs = [r['acc'] for r in all_results if r['net']==net and r['mode']==m]
            if runs: print(f" {np.mean(runs)*100:11.1f}%", end='')
            else: print(f"          -- ", end='')
        print()

    # Health metrics for acc_only (baseline dynamics)
    print(f"\nHEALTH DYNAMICS (acc_only baseline):", flush=True)
    print(f"  {'':12s} {'dead%':>6s} {'sat%':>6s} {'spread':>7s} {'health':>7s} {'leak':>6s} {'drive':>6s}", flush=True)
    for net in nets:
        runs = [r for r in all_results if r['net']==net and r['mode']=='acc_only']
        if runs:
            print(f"  {net:12s} {np.mean([r['dead'] for r in runs])*100:5.1f}% "
                  f"{np.mean([r['sat'] for r in runs])*100:5.1f}% "
                  f"{np.mean([r['spread'] for r in runs]):7.4f} "
                  f"{np.mean([r['health'] for r in runs]):7.4f} "
                  f"{np.mean([r['leak'] for r in runs]):6.3f} "
                  f"{np.mean([r['eff_drive'] for r in runs]):6.1f}", flush=True)

    # Correlation: health vs accuracy across all configs
    print(f"\nCORRELATION health vs accuracy (all runs):", flush=True)
    accs = [r['acc'] for r in all_results]
    healths = [r['health'] for r in all_results]
    if len(accs) > 2:
        corr = np.corrcoef(accs, healths)[0, 1]
        print(f"  Pearson r = {corr:.3f}", flush=True)

    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
