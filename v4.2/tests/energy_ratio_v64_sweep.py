"""Energy ratio sweep V64: the consensus Phase 1 test.
energy[src] × energy[dst] edge scoring, rejection sampling.
energy = accumulated abs activation across all ticks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
SEEDS = list(range(10))
BUDGET = 16000


def run_one(ratio, seed):
    SelfWiringGraph.ENERGY_RATIO = ratio
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(64)
    perm = np.random.permutation(64)
    t0 = time.perf_counter()
    best = train(net, perm, 64, max_attempts=BUDGET, ticks=8, verbose=False)
    return ratio, seed, best, time.perf_counter() - t0


def main():
    jobs = [(r, s) for r in RATIOS for s in SEEDS]
    total = len(jobs)
    print(f'ENERGY RATIO V64 SWEEP: {total} jobs, 22 workers, {BUDGET} budget', flush=True)
    print(f'Ratios: {RATIOS}, Seeds: {SEEDS}', flush=True)
    print('=' * 70, flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for fut in as_completed(futures):
            r, s, score, elapsed = fut.result()
            results.setdefault(r, []).append(score)
            print(f'  ratio={r:.1f} seed={s}: {score*100:.1f}% ({elapsed:.0f}s)', flush=True)

    print(f'\n{"="*70}', flush=True)
    print(f'  {"ratio":>6s} {"mean":>7s} {"std":>6s} {"vs_0%":>8s}', flush=True)
    baseline = np.mean(results.get(0.0, [0])) * 100
    for r in RATIOS:
        scores = results.get(r, [])
        m = np.mean(scores) * 100
        s = np.std(scores) * 100
        diff = m - baseline
        marker = ' <<<' if m >= max(np.mean(results.get(rr, [0])) * 100 for rr in RATIOS) - 0.1 else ''
        print(f'  {r:6.1f} {m:6.1f}% {s:5.1f}pp {diff:+7.1f}pp{marker}', flush=True)
    print(f'{"="*70}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
