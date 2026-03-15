"""10-seed A/B: flip-on-reject vs darwinian strategy.
V64 16K budget — enough to see variance clearly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph

SEEDS = list(range(10))
BUDGET = 16000
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
            # Flip strategy on reject
            if random.random() < 0.35:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < 0.35:
                net.grow = np.int8(1 - int(net.grow))
    return best


def run_darwinian(seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)

    score = evaluate(net, perm)
    best = score
    for att in range(BUDGET):
        state = net.save_state()
        # Strategy mutates HERE (survives reject)
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
            # Strategy NOT reverted
    return best


def main():
    print(f"10-SEED STRATEGY A/B TEST", flush=True)
    print(f"V={V} N={N} budget={BUDGET} seeds={SEEDS}", flush=True)
    print("=" * 60, flush=True)

    flip_scores = []
    darwin_scores = []

    for seed in SEEDS:
        t0 = time.perf_counter()
        fs = run_flip_on_reject(seed)
        t1 = time.perf_counter()
        ds = run_darwinian(seed)
        t2 = time.perf_counter()
        flip_scores.append(fs)
        darwin_scores.append(ds)
        winner = "FLIP" if fs > ds else "DARWIN" if ds > fs else "TIE"
        print(f"  seed={seed}: flip={fs*100:5.1f}% darwin={ds*100:5.1f}% "
              f"diff={( fs-ds)*100:+5.1f}pp {winner} "
              f"({t1-t0:.0f}s / {t2-t1:.0f}s)", flush=True)

    fm = np.mean(flip_scores) * 100
    dm = np.mean(darwin_scores) * 100
    fs_std = np.std(flip_scores) * 100
    ds_std = np.std(darwin_scores) * 100

    print(f"\n{'='*60}", flush=True)
    print(f"  FLIP ON REJECT:  {fm:.1f}% +/- {fs_std:.1f}pp", flush=True)
    print(f"  DARWINIAN:       {dm:.1f}% +/- {ds_std:.1f}pp", flush=True)
    print(f"  DIFF:            {fm-dm:+.1f}pp mean, {fs_std-ds_std:+.1f}pp std", flush=True)
    print(f"\n  Flip wins: {sum(1 for f,d in zip(flip_scores, darwin_scores) if f>d)}/10", flush=True)
    print(f"  Darwin wins: {sum(1 for f,d in zip(flip_scores, darwin_scores) if d>f)}/10", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
