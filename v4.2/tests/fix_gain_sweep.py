"""Fix gain vs dynamic check — parallel."""
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

CONFIGS = [
    ('dynamic_check', None),
    ('fix_1.25', 1.25),
    ('fix_1.50', 1.50),
    ('fix_1.75', 1.75),
    ('fix_2.00', 2.00),
    ('fix_2.50', 2.50),
]


def run_one(name, gain, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    def eval_b():
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            if gain is None:
                neuron_act = np.abs(acts).max(axis=0)
                rw = np.where(neuron_act >= 0.5, np.float32(2.0), np.float32(1.5))
                Weff = mask_f * rw[:, None]
            else:
                Weff = mask_f * np.float32(gain)
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
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, 0.5*acc + 0.5*tp

    _, score = eval_b()
    best_acc = 0.0
    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        net.mutate_with_mood()
        a, s = eval_b()
        if s > score: score = s; best_acc = max(best_acc, a)
        else: net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s

    log_msg(log_q, f"{name:15s} seed={seed:3d} acc={best_acc*100:5.1f}%")
    return {'name': name, 'seed': seed, 'acc': best_acc}


def main():
    total = len(CONFIGS) * len(SEEDS)
    print(f"FIX GAIN SWEEP: {total} jobs parallel", flush=True)
    print("=" * 55, flush=True)

    all_results = []
    with live_log('fix_gain') as (log_q, log_path):
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, name, gain, seed, log_q)
                       for name, gain in CONFIGS for seed in SEEDS]
            for fut in as_completed(futures):
                all_results.append(fut.result())

    groups = defaultdict(list)
    for r in all_results:
        groups[r['name']].append(r['acc'])

    print(f"\n{'='*55}", flush=True)
    ranked = [(n, np.mean(a)*100, np.std(a)*100) for n, a in groups.items()]
    ranked.sort(key=lambda x: -x[1])
    for name, mean, std in ranked:
        print(f"  {name:15s}  acc={mean:5.1f}% +/-{std:.1f}%", flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
