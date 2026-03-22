"""Does weight contrast matter when leak is learnable? Parallel test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from lib.log import live_log, log_msg

V = 64; N = 192; BUDGET = 32000; SEEDS = [42, 77, 123]


def run_one(mode, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)
    leak = 0.85

    # For binary/ternary modes, create weight array
    gain = 2.0  # for learnable_gain mode
    if mode == 'binary_1.5_2.0':
        W_strong = np.random.rand(N, N) > 0.5  # bool
    elif mode == 'ternary_1.5_1.75_2.0':
        W_level = np.random.randint(0, 3, (N, N), dtype=np.int8)

    def eval_b(lk):
        mask_f = net.mask.astype(np.float32)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            if mode == 'fix_2.0':
                Weff = mask_f * 2.0
            elif mode == 'fix_1.75':
                Weff = mask_f * 1.75
            elif mode == 'learnable_gain':
                Weff = mask_f * np.float32(gain)
            elif mode == 'binary_1.5_2.0':
                Weff = mask_f * (1.5 + W_strong.astype(np.float32) * 0.5)
            elif mode == 'ternary_1.5_1.75_2.0':
                w = np.where(W_level == 0, 1.5, np.where(W_level == 1, 1.75, 2.0)).astype(np.float32)
                Weff = mask_f * w
            elif mode == 'dynamic_1.5_2.0':
                neuron_act = np.abs(acts).max(axis=0)
                rw = np.where(neuron_act >= 0.5, np.float32(2.0), np.float32(1.5))
                Weff = mask_f * rw[:, None]
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
    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z
        lk_s = leak
        if mode == 'binary_1.5_2.0':
            sw = W_strong.copy()
        elif mode == 'ternary_1.5_1.75_2.0':
            sw = W_level.copy()
        else:
            sw = None
        gn_s = gain

        # Mood + leak + gain mutation
        if random.random() < 0.2:
            net.mood_x = np.clip(net.mood_x + random.gauss(0, 0.15), 0, 1)
        if random.random() < 0.2:
            net.mood_z = np.clip(net.mood_z + random.gauss(0, 0.15), 0, 1)
        if random.random() < 0.2:
            leak = np.clip(leak + random.gauss(0, 0.03), 0.5, 0.99)
        if mode == 'learnable_gain' and random.random() < 0.2:
            gain = np.clip(gain + random.gauss(0, 0.2), 0.5, 5.0)

        # Mask mutation
        n_ch = max(1, int(1 + net.mood_z * 14))
        for _ in range(n_ch):
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
                # Refiner: flip mask OR toggle weight (if available)
                if sw is not None and random.random() < 0.3 and len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    if mode == 'binary_1.5_2.0':
                        W_strong[i[0], i[1]] = not W_strong[i[0], i[1]]
                    elif mode == 'ternary_1.5_1.75_2.0':
                        W_level[i[0], i[1]] = np.int8(random.randint(0, 2))
                elif len(alive) > 0:
                    i = alive[random.randint(0, len(alive)-1)]
                    net.mask[i[0], i[1]] *= -1

        a, s = eval_b(leak)
        if s > score: score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s; leak = lk_s; gain = gn_s
            if mode == 'binary_1.5_2.0': W_strong[:] = sw
            elif mode == 'ternary_1.5_1.75_2.0': W_level[:] = sw

    log_msg(log_q, f"{mode:22s} seed={seed:3d} acc={best_acc*100:5.1f}% leak={leak:.3f} gain={gain:.2f}")
    return {'mode': mode, 'seed': seed, 'acc': best_acc, 'leak': leak, 'gain': gain}


MODES = ['fix_2.0', 'fix_1.75', 'binary_1.5_2.0', 'ternary_1.5_1.75_2.0', 'dynamic_1.5_2.0', 'learnable_gain']


def main():
    total = len(MODES) * len(SEEDS)
    print(f"WEIGHT CONTRAST + LEARNABLE LEAK: {total} jobs", flush=True)
    print("=" * 65, flush=True)

    all_results = []
    with live_log('weight_with_leak') as (log_q, log_path):
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, m, s, log_q) for m in MODES for s in SEEDS]
            for fut in as_completed(futures):
                all_results.append(fut.result())

    groups = defaultdict(list)
    for r in all_results:
        groups[r['mode']].append(r)
    print(f"\n{'='*65}", flush=True)
    ranked = [(m, np.mean([r['acc'] for r in g])*100, np.std([r['acc'] for r in g])*100,
               np.mean([r['leak'] for r in g]))
              for m, g in groups.items()]
    ranked.sort(key=lambda x: -x[1])
    for m, acc, std, lk in ranked:
        print(f"  {m:22s}  acc={acc:5.1f}% +/-{std:.1f}%  leak={lk:.3f}", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
