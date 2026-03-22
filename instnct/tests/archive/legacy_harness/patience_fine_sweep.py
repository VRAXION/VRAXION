"""Fine patience sweep: 0-10 range to find exact optimum.
We know: p=10 best (51.4%), flip baseline (50.6%), p=25 already drops.
Now: is p=1,2,3,5,7 even better?
p=0 = flip EVERY reject (baseline), p=1 = review every attempt.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123, 0, 1]
BUDGET = 32000
V, N, DENSITY = 64, 192, 0.06
PATIENCE_VALUES = [1, 2, 3, 5, 7, 10, 15]


def evaluate(net, perm):
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
    tp = probs[np.arange(V), perm[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def run_one(patience, seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score
    accepts = 0
    changes = 0

    for att in range(BUDGET):
        state = net.save_state()
        net.mutate()
        s = evaluate(net, perm)
        if s > score:
            score = s; best = max(best, score)
            accepts += 1
        else:
            net.restore_state(state)

        if (att + 1) % patience == 0:
            rate = accepts / patience
            if rate < 0.10:
                net.signal = np.int8(1 - int(net.signal))
                net.grow = np.int8(1 - int(net.grow))
                changes += 1
            elif rate < 0.15:
                if random.random() < 0.5:
                    net.signal = np.int8(1 - int(net.signal))
                else:
                    net.grow = np.int8(1 - int(net.grow))
                changes += 1
            accepts = 0

    return {'patience': patience, 'seed': seed, 'score': best, 'changes': changes}


def run_flip(seed):
    """Baseline: flip on reject."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score
    for att in range(BUDGET):
        state = net.save_state()
        net.mutate()
        s = evaluate(net, perm)
        if s > score:
            score = s; best = max(best, score)
        else:
            net.restore_state(state)
            if random.random() < 0.35:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < 0.35:
                net.grow = np.int8(1 - int(net.grow))
    return {'patience': 'flip', 'seed': seed, 'score': best, 'changes': 0}


def main():
    jobs_flip = [('flip', s) for s in SEEDS]
    jobs_window = [(p, s) for p in PATIENCE_VALUES for s in SEEDS]

    total = len(jobs_flip) + len(jobs_window)
    print(f"FINE PATIENCE SWEEP [1-15] + flip baseline", flush=True)
    print(f"V={V} N={N} budget={BUDGET} seeds={SEEDS}", flush=True)
    print(f"Total jobs: {total}, workers: 22", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = []
        for _, s in jobs_flip:
            futures.append(pool.submit(run_flip, s))
        for p, s in jobs_window:
            futures.append(pool.submit(run_one, p, s))
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            label = f"flip" if r['patience'] == 'flip' else f"p={r['patience']}"
            print(f"  {label:>8s} seed={r['seed']}: {r['score']*100:5.1f}%", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    flip_scores = [r['score'] for r in all_results if r['patience'] == 'flip']
    bm = np.mean(flip_scores) * 100

    print(f"  {'patience':>10s} {'mean':>7s} {'std':>7s} {'vs_flip':>8s}", flush=True)
    print(f"  {'flip':>10s} {bm:6.1f}% {np.std(flip_scores)*100:6.1f}pp     ---", flush=True)

    for p in PATIENCE_VALUES:
        scores = [r['score'] for r in all_results if r['patience'] == p]
        m = np.mean(scores) * 100
        s = np.std(scores) * 100
        diff = m - bm
        marker = " <<<" if diff > 0 else ""
        print(f"  {p:>10d} {m:6.1f}% {s:6.1f}pp {diff:+7.1f}pp{marker}", flush=True)

    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
