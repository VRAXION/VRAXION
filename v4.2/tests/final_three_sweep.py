"""Final 3 unswept params: shrink ratio, loss_step, init_density.
These need code-level patching so we monkey-patch per run.
Track learnable param convergence too (final loss_pct, signal, grow, intensity)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

SEEDS = [42, 77, 123, 0, 1]
BUDGET = 16000


def run_shrink(ratio_remove, seed):
    """Patch shrink remove/rewire ratio."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(64)
    perm = np.random.permutation(64)

    # Monkey-patch mutate to use custom shrink ratio
    orig_mutate = net.mutate
    def patched_mutate():
        if random.randint(1, 20) <= 7:
            net.intensity = np.int8(max(1, min(15, int(net.intensity) + random.choice([-1, 1]))))
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        undo = []
        for _ in range(int(net.intensity)):
            if net.signal:
                net._flip(undo)
            else:
                if net.grow:
                    net._add(undo)
                else:
                    if random.randint(1, 10) <= ratio_remove:
                        net._remove(undo)
                    else:
                        net._rewire(undo)
        if any(e[0] in ('A', 'R', 'W') for e in undo):
            net.alive = list(net.alive_set)
        return undo
    net.mutate = patched_mutate

    best = train(net, perm, 64, max_attempts=BUDGET, ticks=8, verbose=False)
    return ('shrink', f'{ratio_remove}/10', seed, best,
            int(net.loss_pct), int(net.signal), int(net.grow), int(net.intensity))


def run_loss_step(step, seed):
    """Patch loss_pct step size."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(64)
    perm = np.random.permutation(64)

    orig_mutate = net.mutate
    def patched_mutate():
        if random.randint(1, 20) <= 7:
            net.intensity = np.int8(max(1, min(15, int(net.intensity) + random.choice([-1, 1]))))
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-step, step))))
        undo = []
        for _ in range(int(net.intensity)):
            if net.signal:
                net._flip(undo)
            else:
                if net.grow:
                    net._add(undo)
                else:
                    if random.randint(1, 10) <= 7:
                        net._remove(undo)
                    else:
                        net._rewire(undo)
        if any(e[0] in ('A', 'R', 'W') for e in undo):
            net.alive = list(net.alive_set)
        return undo
    net.mutate = patched_mutate

    best = train(net, perm, 64, max_attempts=BUDGET, ticks=8, verbose=False)
    return ('loss_step', f'+-{step}', seed, best,
            int(net.loss_pct), int(net.signal), int(net.grow), int(net.intensity))


def run_density(density, seed):
    """Different init density."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(192, 64, density=density)
    perm = np.random.permutation(64)
    best = train(net, perm, 64, max_attempts=BUDGET, ticks=8, verbose=False)
    return ('density', f'{density:.2f}', seed, best,
            int(net.loss_pct), int(net.signal), int(net.grow), int(net.intensity))


def main():
    jobs = []
    # Shrink ratio: X/10 remove, rest rewire
    for ratio in [3, 5, 7, 9, 10]:
        for seed in SEEDS:
            jobs.append(('shrink', ratio, seed))
    # Loss step: ±N
    for step in [1, 2, 3, 5, 7]:
        for seed in SEEDS:
            jobs.append(('loss_step', step, seed))
    # Init density
    for d in [0.02, 0.04, 0.06, 0.10, 0.15]:
        for seed in SEEDS:
            jobs.append(('density', d, seed))

    total = len(jobs)
    print(f'FINAL 3 SWEEP: {total} jobs, 22 workers, {BUDGET} budget', flush=True)
    print('=' * 80, flush=True)

    results = {}
    convergence = {}
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = {}
        for j in jobs:
            if j[0] == 'shrink':
                futures[pool.submit(run_shrink, j[1], j[2])] = j
            elif j[0] == 'loss_step':
                futures[pool.submit(run_loss_step, j[1], j[2])] = j
            elif j[0] == 'density':
                futures[pool.submit(run_density, j[1], j[2])] = j

        for fut in as_completed(futures):
            sweep, label, seed, score, loss, sig, grow, inten = fut.result()
            results.setdefault((sweep, label), []).append(score)
            convergence.setdefault((sweep, label), []).append(
                {'loss': loss, 'signal': sig, 'grow': grow, 'intensity': inten})
            print(f'  {sweep:10s} {label:>6s} seed={seed}: {score*100:.1f}% '
                  f'loss={loss} sig={sig} grow={grow} int={inten}', flush=True)

    # Summary
    print(f'\n{"="*80}', flush=True)
    for sweep in ['shrink', 'loss_step', 'density']:
        print(f'\n{sweep}:', flush=True)
        labels = sorted(set(l for s, l in results if s == sweep))
        for label in labels:
            scores = results.get((sweep, label), [])
            convs = convergence.get((sweep, label), [])
            m = np.mean(scores) * 100
            s = np.std(scores) * 100
            avg_loss = np.mean([c['loss'] for c in convs])
            avg_int = np.mean([c['intensity'] for c in convs])
            print(f'  {label:>6s}: {m:.1f}% +/-{s:.1f}pp  '
                  f'converged: loss={avg_loss:.0f} int={avg_int:.0f}', flush=True)
    print(f'{"="*80}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
