"""Patience sweep: find the optimal strategy review interval.

Tests patience values [10, 25, 50, 100, 200, 500] + flip_on_reject baseline.
PARALLEL execution — all CPU cores used.
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
PATIENCE_VALUES = [10, 25, 50, 100, 200, 500]


def evaluate(net, perm):
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
    tp = probs[np.arange(V), perm[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def run_one(mode, patience, seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score
    accepts = 0
    strategy_changes = 0

    for att in range(BUDGET):
        state = net.save_state()
        net.mutate()
        s = evaluate(net, perm)
        if s > score:
            score = s; best = max(best, score)
            if mode == 'window':
                accepts += 1
        else:
            net.restore_state(state)
            if mode == 'flip':
                if random.random() < 0.35:
                    net.signal = np.int8(1 - int(net.signal))
                if random.random() < 0.35:
                    net.grow = np.int8(1 - int(net.grow))

        # Window review
        if mode == 'window' and (att + 1) % patience == 0:
            rate = accepts / patience
            if rate < 0.10:
                net.signal = np.int8(1 - int(net.signal))
                net.grow = np.int8(1 - int(net.grow))
                strategy_changes += 1
            elif rate < 0.15:
                if random.random() < 0.5:
                    net.signal = np.int8(1 - int(net.signal))
                else:
                    net.grow = np.int8(1 - int(net.grow))
                strategy_changes += 1
            accepts = 0

    return {'mode': mode, 'patience': patience, 'seed': seed,
            'score': best, 'changes': strategy_changes}


def main():
    # Build all jobs
    jobs = []
    for seed in SEEDS:
        jobs.append(('flip', 0, seed))
    for patience in PATIENCE_VALUES:
        for seed in SEEDS:
            jobs.append(('window', patience, seed))

    total = len(jobs)
    print(f"PATIENCE SWEEP (PARALLEL)", flush=True)
    print(f"V={V} N={N} budget={BUDGET} seeds={SEEDS}", flush=True)
    print(f"Patience values: {PATIENCE_VALUES} + flip baseline", flush=True)
    print(f"Total jobs: {total}, workers: 22", flush=True)
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    all_results = []
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            label = f"flip" if r['mode'] == 'flip' else f"p={r['patience']}"
            print(f"  {label:>8s} seed={r['seed']}: {r['score']*100:5.1f}% "
                  f"changes={r['changes']}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.0f}s", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("PATIENCE vs QUALITY:", flush=True)
    print(f"  {'patience':>10s} {'mean':>7s} {'std':>7s} {'vs_flip':>8s}", flush=True)

    # Baseline
    flip_scores = [r['score'] for r in all_results if r['mode'] == 'flip']
    bm = np.mean(flip_scores) * 100
    bs = np.std(flip_scores) * 100
    print(f"  {'flip':>10s} {bm:6.1f}% {bs:6.1f}pp     ---", flush=True)

    for p in PATIENCE_VALUES:
        scores = [r['score'] for r in all_results
                  if r['mode'] == 'window' and r['patience'] == p]
        m = np.mean(scores) * 100
        s = np.std(scores) * 100
        diff = m - bm
        changes = np.mean([r['changes'] for r in all_results
                           if r['mode'] == 'window' and r['patience'] == p])
        marker = " <<<" if m >= max(
            np.mean([r['score'] for r in all_results
                     if r['patience'] == pp]) * 100
            for pp in PATIENCE_VALUES) - 0.1 else ""
        print(f"  {p:>10d} {m:6.1f}% {s:6.1f}pp {diff:+7.1f}pp "
              f"(avg {changes:.0f} changes){marker}", flush=True)

    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
