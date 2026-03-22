"""Test: learnable leak + gain vs fixed, parallel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

V = 64; N = 192; BUDGET = 16000; SEEDS = [42, 77, 123]


def run_one(mode, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    # Learnable params (start at current defaults)
    learn_leak = 0.85
    learn_gain = 2.0

    def eval_b(lk, gn):
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            Weff = mask_f * np.float32(gn)
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

    if mode == 'fixed':
        lk, gn = 0.85, 2.0
    elif mode == 'fixed_best':
        lk, gn = 0.95, 2.0
    else:
        lk, gn = learn_leak, learn_gain

    _, score = eval_b(lk, gn)
    best_acc = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = lk; gn_s = gn

        # Mood mutation
        net.mutate_with_mood()

        # Learnable leak + gain mutation (if learnable mode)
        if mode == 'learn_both':
            if random.random() < 0.2:
                lk = np.clip(lk + random.gauss(0, 0.03), 0.5, 0.99)
            if random.random() < 0.2:
                gn = np.clip(gn + random.gauss(0, 0.2), 0.5, 5.0)
        elif mode == 'learn_leak':
            if random.random() < 0.2:
                lk = np.clip(lk + random.gauss(0, 0.03), 0.5, 0.99)
        elif mode == 'learn_gain':
            if random.random() < 0.2:
                gn = np.clip(gn + random.gauss(0, 0.2), 0.5, 5.0)

        a, s = eval_b(lk, gn)
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s
            lk = lk_s; gn = gn_s

    log_msg(log_q, f"{mode:15s} seed={seed:3d} acc={best_acc*100:5.1f}% "
            f"leak={lk:.3f} gain={gn:.2f}")
    return {'mode': mode, 'seed': seed, 'acc': best_acc, 'leak': lk, 'gain': gn}


MODES = ['fixed', 'fixed_best', 'learn_leak', 'learn_gain', 'learn_both']


def main():
    total = len(MODES) * len(SEEDS)
    print(f"LEARNABLE LEAK+GAIN: {total} jobs parallel", flush=True)
    print(f"Modes: {MODES}", flush=True)
    print("=" * 65, flush=True)

    all_results = []
    with live_log('learnable_leak_gain') as (log_q, log_path):
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, mode, seed, log_q)
                       for mode in MODES for seed in SEEDS]
            for fut in as_completed(futures):
                all_results.append(fut.result())

    groups = defaultdict(list)
    for r in all_results:
        groups[r['mode']].append(r)

    print(f"\n{'='*65}", flush=True)
    ranked = []
    for mode, runs in groups.items():
        mean_acc = np.mean([r['acc'] for r in runs]) * 100
        std_acc = np.std([r['acc'] for r in runs]) * 100
        mean_leak = np.mean([r['leak'] for r in runs])
        mean_gain = np.mean([r['gain'] for r in runs])
        ranked.append((mode, mean_acc, std_acc, mean_leak, mean_gain))
    ranked.sort(key=lambda x: -x[1])

    for mode, acc, std, lk, gn in ranked:
        print(f"  {mode:15s}  acc={acc:5.1f}% +/-{std:.1f}%  "
              f"leak={lk:.3f} gain={gn:.2f}", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
