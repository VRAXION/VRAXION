"""Window-based strategy: run N attempts with fixed strategy,
then evaluate acceptance rate and decide whether to switch.

Compare: flip_on_reject vs window_25 vs window_50 vs window_100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123, 0, 1]
BUDGET = 32000
V, N, DENSITY = 64, 192, 0.06


def evaluate(net, perm):
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
    tp = probs[np.arange(V), perm[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def run_flip_on_reject(seed):
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
    return best


def run_window(seed, window_size):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score
    accepts_in_window = 0

    for att in range(BUDGET):
        state = net.save_state()
        net.mutate()
        s = evaluate(net, perm)
        if s > score:
            score = s; best = max(best, score)
            accepts_in_window += 1
        else:
            net.restore_state(state)

        # End of window: evaluate strategy and maybe switch
        if (att + 1) % window_size == 0:
            accept_rate = accepts_in_window / window_size
            # If accept rate is low, try different strategy
            if accept_rate < 0.10:
                # Both bits flip — big change
                net.signal = np.int8(1 - int(net.signal))
                net.grow = np.int8(1 - int(net.grow))
            elif accept_rate < 0.15:
                # One bit flips — small change
                if random.random() < 0.5:
                    net.signal = np.int8(1 - int(net.signal))
                else:
                    net.grow = np.int8(1 - int(net.grow))
            # else: accept rate OK, keep strategy
            accepts_in_window = 0

    return best


def main():
    modes = [
        ('flip_reject', None),
        ('window_25', 25),
        ('window_50', 50),
        ('window_100', 100),
    ]

    print(f"WINDOW STRATEGY A/B TEST", flush=True)
    print(f"V={V} N={N} budget={BUDGET} seeds={SEEDS}", flush=True)
    print("=" * 70, flush=True)

    all_results = {}
    for mode_name, window in modes:
        scores = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            if window is None:
                s = run_flip_on_reject(seed)
            else:
                s = run_window(seed, window)
            elapsed = time.perf_counter() - t0
            scores.append(s)
            print(f"  {mode_name:14s} seed={seed}: {s*100:5.1f}% ({elapsed:.0f}s)", flush=True)
        all_results[mode_name] = scores
        m = np.mean(scores) * 100
        std = np.std(scores) * 100
        print(f"  {mode_name:14s} MEAN: {m:.1f}% +/- {std:.1f}pp\n", flush=True)

    print(f"{'='*70}", flush=True)
    print("SUMMARY:", flush=True)
    for mode_name, _ in modes:
        scores = all_results[mode_name]
        m = np.mean(scores) * 100
        std = np.std(scores) * 100
        print(f"  {mode_name:14s}: {m:.1f}% +/- {std:.1f}pp", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
