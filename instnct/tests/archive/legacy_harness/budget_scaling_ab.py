"""Budget scaling: flip vs darwinian at 16K, 32K, 64K.
Which one LEARNS FASTER with more budget?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123]
BUDGETS = [16000, 32000, 64000]
V, N, DENSITY = 64, 192, 0.06


def evaluate(net, perm):
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
    tp = probs[np.arange(V), perm[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def run(mode, seed, budget):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score

    for att in range(budget):
        state = net.save_state()

        if mode == 'darwinian':
            if random.random() < 0.35:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < 0.35:
                net.grow = np.int8(1 - int(net.grow))

        net.mutate()
        s = evaluate(net, perm)
        if s > score:
            score = s; best = max(best, score)
        else:
            net.restore_state(state)
            if mode == 'flip':
                if random.random() < 0.35:
                    net.signal = np.int8(1 - int(net.signal))
                if random.random() < 0.35:
                    net.grow = np.int8(1 - int(net.grow))
    return best


def main():
    print("BUDGET SCALING: FLIP vs DARWINIAN", flush=True)
    print(f"Budgets={BUDGETS}, Seeds={SEEDS}", flush=True)
    print("=" * 70, flush=True)

    results = {}
    for budget in BUDGETS:
        for mode in ['flip', 'darwinian']:
            scores = []
            for seed in SEEDS:
                t0 = time.perf_counter()
                s = run(mode, seed, budget)
                elapsed = time.perf_counter() - t0
                scores.append(s)
                print(f"  {mode:10s} budget={budget:5d} seed={seed}: {s*100:5.1f}% ({elapsed:.0f}s)", flush=True)
            key = (mode, budget)
            results[key] = scores
            print(f"  {mode:10s} budget={budget:5d} MEAN: {np.mean(scores)*100:.1f}%", flush=True)
            print(flush=True)

    # Summary: learning curves
    print(f"\n{'='*70}", flush=True)
    print("LEARNING CURVES:", flush=True)
    print(f"  {'budget':>6s} {'flip':>8s} {'darwin':>8s} {'diff':>8s} {'flip_slope':>10s} {'darwin_slope':>12s}", flush=True)

    prev_flip = prev_darwin = None
    for budget in BUDGETS:
        fm = np.mean(results[('flip', budget)]) * 100
        dm = np.mean(results[('darwinian', budget)]) * 100
        diff = fm - dm

        f_slope = f"{fm - prev_flip:+.1f}pp" if prev_flip else "---"
        d_slope = f"{dm - prev_darwin:+.1f}pp" if prev_darwin else "---"

        print(f"  {budget:6d} {fm:7.1f}% {dm:7.1f}% {diff:+7.1f}pp {f_slope:>10s} {d_slope:>12s}", flush=True)
        prev_flip = fm; prev_darwin = dm

    print(f"\nVERDICT:", flush=True)
    flip_gain = np.mean(results[('flip', 64000)]) - np.mean(results[('flip', 16000)])
    darwin_gain = np.mean(results[('darwinian', 64000)]) - np.mean(results[('darwinian', 16000)])
    print(f"  Flip  16K→64K gain: {flip_gain*100:+.1f}pp", flush=True)
    print(f"  Darwin 16K→64K gain: {darwin_gain*100:+.1f}pp", flush=True)
    if darwin_gain > flip_gain + 0.01:
        print(f"  → DARWINIAN scales better!", flush=True)
    elif flip_gain > darwin_gain + 0.01:
        print(f"  → FLIP scales better!", flush=True)
    else:
        print(f"  → SAME scaling!", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
