"""Sweep all remaining unswept params. One at a time, baseline vs alternatives.
V64, 16K budget, 5 seeds, parallel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

SEEDS = [42, 77, 123, 0, 1]
BUDGET = 16000


def run_one(sweep, label, overrides, seed):
    # Apply overrides to class/instance
    np.random.seed(seed); random.seed(seed)
    for k, v in overrides.get('class', {}).items():
        setattr(SelfWiringGraph, k, v)
    net = SelfWiringGraph(64)
    for k, v in overrides.get('instance', {}).items():
        setattr(net, k, v)
    perm = np.random.permutation(64)
    best = train(net, perm, 64, max_attempts=BUDGET, ticks=8, verbose=False)
    # Reset class attrs
    SelfWiringGraph.LOSS_DRIFT = 0.2
    SelfWiringGraph.PATIENCE = 0.35
    return sweep, label, seed, best


SWEEPS = {
    'LOSS_DRIFT': [
        ('0.05', {'class': {'LOSS_DRIFT': 0.05}}),
        ('0.10', {'class': {'LOSS_DRIFT': 0.10}}),
        ('0.20', {'class': {'LOSS_DRIFT': 0.20}}),  # current
        ('0.35', {'class': {'LOSS_DRIFT': 0.35}}),
        ('0.50', {'class': {'LOSS_DRIFT': 0.50}}),
    ],
    'loss_step': [
        ('+-1', {'class': {}}),  # need custom mutate — skip, use +-3 variants below
        ('+-2', {'class': {}}),
        ('+-3', {'class': {}}),  # current
        ('+-5', {'class': {}}),
    ],
    'shrink_ratio': [
        ('50/50', {'class': {}}),  # 50% remove, 50% rewire
        ('70/30', {'class': {}}),  # current
        ('90/10', {'class': {}}),
        ('100/0', {'class': {}}),  # pure remove
    ],
    'intensity_start': [
        ('1', {'instance': {'intensity': np.int8(1)}}),
        ('3', {'instance': {'intensity': np.int8(3)}}),
        ('7', {'instance': {'intensity': np.int8(7)}}),  # current
        ('12', {'instance': {'intensity': np.int8(12)}}),
        ('15', {'instance': {'intensity': np.int8(15)}}),
    ],
    'signal_start': [
        ('0_struct', {'instance': {'signal': np.int8(0)}}),  # current
        ('1_signal', {'instance': {'signal': np.int8(1)}}),
    ],
    'grow_start': [
        ('0_shrink', {'instance': {'grow': np.int8(0)}}),
        ('1_grow', {'instance': {'grow': np.int8(1)}}),  # current
    ],
    'loss_pct_start': [
        ('1', {'instance': {'loss_pct': np.int8(1)}}),
        ('5', {'instance': {'loss_pct': np.int8(5)}}),
        ('15', {'instance': {'loss_pct': np.int8(15)}}),  # current
        ('30', {'instance': {'loss_pct': np.int8(30)}}),
    ],
    'init_density': [
        ('0.02', {'class': {}}),
        ('0.04', {'class': {}}),
        ('0.06', {'class': {}}),  # current
        ('0.10', {'class': {}}),
        ('0.15', {'class': {}}),
    ],
}

# Only run sweeps that don't need code changes (loss_step, shrink_ratio, init_density need patching)
SIMPLE_SWEEPS = ['LOSS_DRIFT', 'intensity_start', 'signal_start', 'grow_start', 'loss_pct_start']


def main():
    jobs = []
    for sweep in SIMPLE_SWEEPS:
        for label, overrides in SWEEPS[sweep]:
            for seed in SEEDS:
                jobs.append((sweep, label, overrides, seed))

    total = len(jobs)
    print(f'REMAINING PARAMS SWEEP: {total} jobs, 22 workers, {BUDGET} budget', flush=True)
    print(f'Sweeps: {SIMPLE_SWEEPS}', flush=True)
    print('=' * 70, flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for fut in as_completed(futures):
            sweep, label, seed, score = fut.result()
            results.setdefault((sweep, label), []).append(score)
            print(f'  {sweep:18s} {label:10s} seed={seed}: {score*100:.1f}%', flush=True)

    print(f'\n{"="*70}', flush=True)
    for sweep in SIMPLE_SWEEPS:
        print(f'\n{sweep}:', flush=True)
        labels = [l for l, _ in SWEEPS[sweep]]
        for label in labels:
            scores = results.get((sweep, label), [])
            if scores:
                m = np.mean(scores) * 100
                s = np.std(scores) * 100
                current = ' (current)' if any(
                    label in ['0.20', '7', '0_struct', '1_grow', '15']
                    and label == l for l, _ in SWEEPS[sweep]
                ) else ''
                print(f'  {label:>10s}: {m:.1f}% +/-{s:.1f}pp{current}', flush=True)
    print(f'{"="*70}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
