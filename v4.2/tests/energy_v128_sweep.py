"""V128 energy ratio sweep — parallel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

RATIOS = [0.0, 0.3, 0.5, 0.6]
SEEDS = [42, 77, 123]


def run_one(ratio, seed):
    SelfWiringGraph.ENERGY_RATIO = ratio
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(128)
    perm = np.random.permutation(128)
    t0 = time.perf_counter()
    best = train(net, perm, 128, max_attempts=32000, ticks=8, verbose=False)
    return ratio, seed, best, time.perf_counter() - t0


def main():
    jobs = [(r, s) for r in RATIOS for s in SEEDS]
    print(f'V128 ENERGY SWEEP: {len(jobs)} jobs parallel', flush=True)
    print('=' * 60, flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for fut in as_completed(futures):
            r, s, score, elapsed = fut.result()
            results.setdefault(r, []).append(score)
            print(f'  ratio={r:.1f} seed={s}: {score*100:.1f}% ({elapsed:.0f}s)', flush=True)

    print(f'\n{"="*60}', flush=True)
    for r in RATIOS:
        scores = results.get(r, [])
        if scores:
            print(f'  ratio={r:.1f}: mean={np.mean(scores)*100:.1f}% std={np.std(scores)*100:.1f}pp', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
