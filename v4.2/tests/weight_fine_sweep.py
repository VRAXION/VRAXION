"""Fine-grained binary weight sweep to find Pareto front / knee."""
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

WEAK_VALUES = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
STRONG_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]


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
        else: net.mask = sm; net.resync_alive(); net.W_strong = sw; net.mood_x = mx_s; net.mood_z = mz_s

    log_msg(log_q, f"w={weak:.2f} s={strong:.1f} seed={seed:3d} acc={best_acc*100:5.1f}%")
    return {'weak': weak, 'strong': strong, 'seed': seed, 'acc': best_acc,
            'ratio': strong/weak if weak > 0 else 999}


def main():
    configs = [(w, s) for w in WEAK_VALUES for s in STRONG_VALUES if s > w]
    total = len(configs) * len(SEEDS)
    print(f"FINE WEIGHT SWEEP: {len(configs)} configs x {len(SEEDS)} seeds = {total} jobs", flush=True)
    print(f"Weak: {WEAK_VALUES}", flush=True)
    print(f"Strong: {STRONG_VALUES}", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with live_log('weight_fine_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} jobs on 20 workers")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = []
            for weak, strong in configs:
                for seed in SEEDS:
                    futures.append(pool.submit(run_one, weak, strong, seed, log_q))
            for fut in as_completed(futures):
                all_results.append(fut.result())
        log_msg(log_q, "Done")

    # Group and rank
    groups = defaultdict(list)
    for r in all_results:
        key = (r['weak'], r['strong'])
        groups[key].append(r['acc'])

    print(f"\n{'='*70}", flush=True)
    print("HEATMAP (weak x strong -> mean acc %):", flush=True)
    print(f"{'weak':>6s}", end="")
    for s in STRONG_VALUES:
        print(f"  s={s:<4.1f}", end="")
    print()
    for w in WEAK_VALUES:
        print(f"{w:6.2f}", end="")
        for s in STRONG_VALUES:
            if s <= w:
                print(f"  {'---':>5s}", end="")
            else:
                accs = groups.get((w, s), [])
                if accs:
                    print(f"  {np.mean(accs)*100:5.1f}", end="")
                else:
                    print(f"  {'???':>5s}", end="")
        print()

    print(f"\n{'='*70}", flush=True)
    print("TOP 10:", flush=True)
    ranked = []
    for (w, s), accs in groups.items():
        ranked.append((w, s, np.mean(accs)*100, np.std(accs)*100, s/w))
    ranked.sort(key=lambda x: -x[2])
    for i, (w, s, mean, std, ratio) in enumerate(ranked[:10]):
        print(f"  {i+1:2d}. weak={w:.2f} strong={s:.1f}  ratio={ratio:.1f}:1  "
              f"acc={mean:5.1f}% +/-{std:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
