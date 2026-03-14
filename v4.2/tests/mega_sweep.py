"""MEGA SWEEP: wildly different configs to test if C/weak/strong are universal."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

SEEDS = [42, 77]  # 2 seeds for speed

# Wildly different configs: vary EVERYTHING
CONFIGS = [
    # (name, V, N, density, threshold, leak, charge_rate, clip_factor, task, budget)
    # --- SIZE variations ---
    ("tiny_V8",        8,   32, 0.06, 0.5, 0.85, 0.3, 2.0, "perm", 8000),
    ("small_V16",     16,   80, 0.06, 0.5, 0.85, 0.3, 2.0, "perm", 8000),
    ("medium_V64",    64,  192, 0.06, 0.5, 0.85, 0.3, 2.0, "perm", 16000),
    ("large_V128",   128,  384, 0.06, 0.5, 0.85, 0.3, 2.0, "perm", 16000),
    # --- DENSITY variations ---
    ("sparse_d002",   64,  192, 0.02, 0.5, 0.85, 0.3, 2.0, "perm", 16000),
    ("dense_d015",    64,  192, 0.15, 0.5, 0.85, 0.3, 2.0, "perm", 16000),
    ("dense_d030",    64,  192, 0.30, 0.5, 0.85, 0.3, 2.0, "perm", 16000),
    # --- LEAK variations ---
    ("leak_070",      64,  192, 0.06, 0.5, 0.70, 0.3, 2.0, "perm", 16000),
    ("leak_080",      64,  192, 0.06, 0.5, 0.80, 0.3, 2.0, "perm", 16000),
    ("leak_090",      64,  192, 0.06, 0.5, 0.90, 0.3, 2.0, "perm", 16000),
    ("leak_095",      64,  192, 0.06, 0.5, 0.95, 0.3, 2.0, "perm", 16000),
    # --- THRESHOLD variations ---
    ("thresh_02",     64,  192, 0.06, 0.2, 0.85, 0.3, 2.0, "perm", 16000),
    ("thresh_10",     64,  192, 0.06, 1.0, 0.85, 0.3, 2.0, "perm", 16000),
    ("thresh_20",     64,  192, 0.06, 2.0, 0.85, 0.3, 2.0, "perm", 16000),
    # --- CHARGE_RATE variations ---
    ("crate_01",      64,  192, 0.06, 0.5, 0.85, 0.1, 2.0, "perm", 16000),
    ("crate_05",      64,  192, 0.06, 0.5, 0.85, 0.5, 2.0, "perm", 16000),
    ("crate_10",      64,  192, 0.06, 0.5, 0.85, 1.0, 2.0, "perm", 16000),
    # --- CLIP variations ---
    ("clip_10",       64,  192, 0.06, 0.5, 0.85, 0.3, 1.0, "perm", 16000),
    ("clip_40",       64,  192, 0.06, 0.5, 0.85, 0.3, 4.0, "perm", 16000),
    ("clip_80",       64,  192, 0.06, 0.5, 0.85, 0.3, 8.0, "perm", 16000),
    # --- EXTREME combos ---
    ("fast_leak",     64,  192, 0.06, 0.3, 0.70, 0.5, 2.0, "perm", 16000),
    ("slow_thick",    64,  192, 0.15, 1.0, 0.95, 0.1, 4.0, "perm", 16000),
    ("wild",          64,  192, 0.10, 0.8, 0.75, 0.8, 3.0, "perm", 16000),
]


def run_one(cfg, seed, log_q=None):
    name, V, N, density, threshold, leak, charge_rate, clip_factor, task, budget = cfg
    np.random.seed(seed); random.seed(seed)

    net = SelfWiringGraph(N, V, density=density, threshold=threshold, leak=leak)
    net.charge_rate = charge_rate
    net.clip_factor = clip_factor

    if task == "perm":
        perm = np.random.permutation(V)
    targets = perm

    def eval_b():
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            neuron_act = np.abs(acts).max(axis=0)
            is_strong = neuron_act >= net.dyn_C
            row_weight = np.where(is_strong, np.float32(net.w_strong),
                                  np.float32(net.w_weak))
            Weff = mask_f * row_weight[:, None]
            raw = acts @ Weff + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
            charges *= net.leak
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return acc, 0.5*acc + 0.5*tp

    _, score = eval_b()
    best_acc = 0.0
    for att in range(budget):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        net.mutate_with_mood()
        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s

    log_msg(log_q, f"{name:16s} seed={seed:2d} V={V:3d} N={N:3d} d={density:.2f} "
            f"th={threshold:.1f} lk={leak:.2f} cr={charge_rate:.1f} cl={clip_factor:.1f} "
            f"acc={best_acc*100:5.1f}%")
    return {
        'name': name, 'seed': seed, 'V': V, 'N': N,
        'density': density, 'threshold': threshold, 'leak': leak,
        'charge_rate': charge_rate, 'clip_factor': clip_factor,
        'acc': best_acc,
    }


def main():
    total = len(CONFIGS) * len(SEEDS)
    print(f"MEGA SWEEP: {len(CONFIGS)} configs x {len(SEEDS)} seeds = {total} jobs", flush=True)
    print(f"Workers: 22 (leave 2 cores free)", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    with live_log('mega_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=22) as pool:
            futures = [pool.submit(run_one, cfg, seed, log_q)
                       for cfg in CONFIGS for seed in SEEDS]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Group by name
    groups = defaultdict(list)
    for r in all_results:
        groups[r['name']].append(r)

    print(f"\n{'='*80}", flush=True)
    print(f"{'Name':16s} {'V':>3s} {'N':>3s} {'d':>5s} {'th':>4s} {'lk':>4s} "
          f"{'cr':>3s} {'cl':>3s} {'Acc':>6s}", flush=True)
    print("-" * 80, flush=True)

    ranked = []
    for name, runs in groups.items():
        mean_acc = np.mean([r['acc'] for r in runs]) * 100
        r0 = runs[0]
        ranked.append((name, r0['V'], r0['N'], r0['density'], r0['threshold'],
                       r0['leak'], r0['charge_rate'], r0['clip_factor'], mean_acc))
    ranked.sort(key=lambda x: -x[-1])

    for name, V, N, d, th, lk, cr, cl, acc in ranked:
        print(f"{name:16s} {V:3d} {N:3d} {d:5.2f} {th:4.1f} {lk:4.2f} "
              f"{cr:3.1f} {cl:3.1f} {acc:5.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"BEST: {ranked[0][0]} ({ranked[0][-1]:.1f}%)", flush=True)
    print(f"WORST: {ranked[-1][0]} ({ranked[-1][-1]:.1f}%)", flush=True)

    # Check if dynamic threshold works across ALL configs
    baseline = [x[-1] for x in ranked if x[0] == 'medium_V64']
    print(f"\nBaseline (medium_V64): {baseline[0]:.1f}%" if baseline else "", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
