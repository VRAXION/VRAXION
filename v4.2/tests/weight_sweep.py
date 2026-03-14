"""Ternary vs Binary weight sweep — parallel."""
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


def run_one(name, weak, mid, strong, n_levels, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    if n_levels == 3:
        net.W_strong = np.random.randint(0, 3, (N, N), dtype=np.int8)

    def make_weff():
        if n_levels == 2:
            return (weak + net.W_strong.astype(np.float32) * (strong - weak)) * net.mask.astype(np.float32)
        else:
            w = np.where(net.W_strong == 0, np.float32(weak),
                np.where(net.W_strong == 1, np.float32(mid), np.float32(strong)))
            return w * net.mask.astype(np.float32)

    def eval_b():
        Weff = make_weff()
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
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
        sm = net.mask.copy(); sw = net.W_strong.copy()
        mx_s = net.mood_x; mz_s = net.mood_z

        if random.random() < 0.2:
            net.mood_x = np.clip(net.mood_x + random.gauss(0, 0.15), 0, 1)
        if random.random() < 0.2:
            net.mood_z = np.clip(net.mood_z + random.gauss(0, 0.15), 0, 1)
        n_changes = max(1, int(1 + net.mood_z * 14))

        for _ in range(n_changes):
            alive = np.argwhere(net.mask != 0)
            dead = np.argwhere(net.mask == 0)
            dead = dead[dead[:, 0] != dead[:, 1]] if len(dead) > 0 else dead

            if net.mood_x < 0.33:
                if random.random() < 0.7 and len(dead) > 0:
                    i = dead[random.randint(0, len(dead)-1)]
                    net.mask[i[0], i[1]] = 1 if random.random() > 0.5 else -1
                elif len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    net.mask[i[0], i[1]] *= -1
            elif net.mood_x < 0.66:
                r = random.random()
                if r < 0.6 and len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    old = net.mask[i[0], i[1]]; net.mask[i[0], i[1]] = 0
                    nc = random.randint(0, N-1)
                    while nc == i[0]: nc = random.randint(0, N-1)
                    net.mask[i[0], nc] = old
                elif r < 0.8 and len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    net.mask[i[0], i[1]] *= -1
                elif len(dead) > 0:
                    i = dead[random.randint(0, len(dead)-1)]
                    net.mask[i[0], i[1]] = 1 if random.random() > 0.5 else -1
            else:
                if random.random() < 0.5 and len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    net.mask[i[0], i[1]] *= -1
                elif len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    if n_levels == 2:
                        net.W_strong[i[0], i[1]] = not net.W_strong[i[0], i[1]]
                    else:
                        net.W_strong[i[0], i[1]] = np.int8(random.randint(0, 2))

        a, s = eval_b()
        if s > score: score = s; best_acc = max(best_acc, a)
        else: net.mask = sm; net.W_strong = sw; net.mood_x = mx_s; net.mood_z = mz_s

    log_msg(log_q, f"{name:22s} seed={seed:3d} acc={best_acc*100:5.1f}% "
            f"({n_levels}L: {weak:.3f}/{mid:.3f}/{strong:.3f})")
    return {'name': name, 'seed': seed, 'acc': best_acc, 'n_levels': n_levels}


CONFIGS = [
    ('tern_0.5_1.0_1.5',   0.5, 1.0, 1.5, 3),
    ('tern_0.4_1.0_2.6',   0.382, 1.0, 2.618, 3),
    ('tern_0.5_1.0_3.0',   0.5, 1.0, 3.0, 3),
    ('tern_0.1_1.0_5.0',   0.1, 1.0, 5.0, 3),
    ('tern_phi',            0.618, 1.0, 1.618, 3),
    ('bin_0.5_1.5',         0.5, 0.0, 1.5, 2),
    ('bin_1.0_3.0',         1.0, 0.0, 3.0, 2),
    ('bin_phi_sq',          0.382, 0.0, 2.618, 2),
]


def main():
    total = len(CONFIGS) * len(SEEDS)
    print(f"TERNARY vs BINARY WEIGHT SWEEP (V={V}, N={N}, {BUDGET} att)", flush=True)
    print(f"Configs: {len(CONFIGS)}, Seeds: {SEEDS}, Jobs: {total}, Workers: 20", flush=True)
    print("=" * 75, flush=True)

    all_results = []
    with live_log('weight_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = []
            for name, weak, mid, strong, n_levels in CONFIGS:
                for seed in SEEDS:
                    fut = pool.submit(run_one, name, weak, mid, strong, n_levels, seed, log_q)
                    futures.append(fut)
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, f"All {total} jobs complete")

    print(f"\n{'='*75}", flush=True)
    print("RANKED:", flush=True)
    groups = defaultdict(list)
    for r in all_results:
        groups[r['name']].append(r['acc'])
    ranked = []
    for name, accs in groups.items():
        lvl = '3L' if any(r['n_levels'] == 3 and r['name'] == name for r in all_results) else '2L'
        ranked.append((name, np.mean(accs)*100, np.std(accs)*100, lvl))
    ranked.sort(key=lambda x: -x[1])
    for name, mean, std, lvl in ranked:
        print(f"  {name:22s} [{lvl}] acc={mean:5.1f}% +/-{std:.1f}%", flush=True)
    print(f"\nWINNER: {ranked[0][0]} ({ranked[0][1]:.1f}%)", flush=True)
    print(f"{'='*75}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
