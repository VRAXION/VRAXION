"""Extend weight sweep into unexplored regions."""
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

# Fill in the missing corners
CONFIGS = [
    # Higher weak values
    (2.50, 3.0), (2.50, 3.5), (2.50, 4.0), (2.50, 5.0),
    (3.00, 3.5), (3.00, 4.0), (3.00, 5.0), (3.00, 7.0),
    # Lower strong for high weak (the diagonal)
    (1.50, 1.5), (1.75, 1.5), (1.75, 2.0), (1.75, 2.5), (1.75, 3.0),
    (2.00, 2.0), (2.00, 2.25),
    (2.50, 2.5), (2.50, 2.75),
    # Phi diagonal points
    (1.00, 1.618), (1.25, 2.023), (1.50, 2.427), (1.75, 2.832), (2.00, 3.236),
    # Very close ratio points
    (1.50, 1.75), (2.00, 2.5), (2.50, 3.25),
]

# Remove duplicates and cases where strong <= weak
CONFIGS = list(set((round(w,3), round(s,3)) for w, s in CONFIGS if s > w))


def run_one(weak, strong, seed, log_q=None):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    def eval_b():
        Weff = (weak + net.W_strong.astype(np.float32) * (strong - weak)) * net.mask.astype(np.float32)
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
        net.mutate_with_mood()
        a, s = eval_b()
        if s > score: score = s; best_acc = max(best_acc, a)
        else: net.mask = sm; net.W_strong = sw; net.mood_x = mx_s; net.mood_z = mz_s

    log_msg(log_q, f"w={weak:.3f} s={strong:.3f} seed={seed:3d} acc={best_acc*100:5.1f}%")
    return {'weak': weak, 'strong': strong, 'seed': seed, 'acc': best_acc}


def main():
    total = len(CONFIGS) * len(SEEDS)
    print(f"EXTEND SWEEP: {len(CONFIGS)} configs x {len(SEEDS)} seeds = {total} jobs", flush=True)
    print(f"Configs: {sorted(CONFIGS)}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with live_log('weight_extend') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_one, w, s, seed, log_q)
                       for w, s in CONFIGS for seed in SEEDS]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    groups = defaultdict(list)
    for r in all_results:
        groups[(r['weak'], r['strong'])].append(r['acc'])

    print(f"\n{'='*70}", flush=True)
    ranked = [(w, s, np.mean(a)*100, np.std(a)*100, s/w)
              for (w, s), a in groups.items()]
    ranked.sort(key=lambda x: -x[2])
    print("TOP 10:", flush=True)
    for i, (w, s, m, sd, ratio) in enumerate(ranked[:10]):
        print(f"  {i+1:2d}. w={w:.3f} s={s:.3f} ratio={ratio:.3f} acc={m:.1f}%+/-{sd:.1f}%", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
