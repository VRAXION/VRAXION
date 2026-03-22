"""N/V capacity test: can N/V=1 shared reach 100%?

Test on EASY tasks where N/V=3 reaches 100%.
If N/V=1 also reaches 100% → internal neurons useless.
If N/V=1 stuck at <90% → internal neurons needed.

V=16 (easy, 100% reachable fast)
V=32 (medium)
V=64 (our standard)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123]
BUDGET = 32000


def evaluate(net, targets, V, ticks=8):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return acc, 0.5 * acc + 0.5 * tp


def run_one(V, nv_ratio, seed):
    N = max(V, V * nv_ratio)  # N/V=1 means N=V (shared I/O)
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    _, score = evaluate(net, perm, V)
    best_acc = 0.0
    best_score = score
    stale = 0

    for att in range(BUDGET):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        acc, s = evaluate(net, perm, V)

        if s > score:
            score = s
            best_acc = max(best_acc, acc)
            best_score = max(best_score, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if random.random() < net.PATIENCE:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < net.PATIENCE:
                net.grow = np.int8(1 - int(net.grow))

        if best_acc >= 1.0 or stale >= 6000:
            break

    return {
        'V': V, 'ratio': nv_ratio, 'N': N, 'seed': seed,
        'acc': best_acc, 'score': best_score,
        'conns': net.count_connections(), 'att': att + 1,
        'io': 'shared' if N == V else 'split',
    }


def main():
    configs = []
    for V in [16, 32, 64]:
        for nv in [1, 2, 3, 4]:
            for seed in SEEDS:
                configs.append((V, nv, seed))

    total = len(configs)
    print(f"N/V CAPACITY TEST: {total} jobs, 22 workers, {BUDGET} budget", flush=True)
    print(f"V=[16,32,64] × N/V=[1,2,3,4] × 3 seeds", flush=True)
    print(f"Question: can N/V=1 (shared I/O) reach 100%?", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = [pool.submit(run_one, *c) for c in configs]
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            print(f"  V={r['V']:3d} N/V={r['ratio']} ({r['io']:6s}) seed={r['seed']}: "
                  f"acc={r['acc']*100:5.1f}% N={r['N']} att={r['att']}", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    for V in [16, 32, 64]:
        print(f"\nV={V} (random perm, 32K budget):", flush=True)
        print(f"  {'N/V':>4s} {'N':>4s} {'I/O':>6s} {'mean_acc':>9s} {'100%?':>6s} {'matmul':>8s}", flush=True)
        for nv in [1, 2, 3, 4]:
            runs = [r for r in all_results if r['V'] == V and r['ratio'] == nv]
            if runs:
                N = runs[0]['N']
                io = runs[0]['io']
                accs = [r['acc'] for r in runs]
                m = np.mean(accs) * 100
                reached = sum(1 for a in accs if a >= 0.99)
                marker = " <<<" if reached == len(accs) else ""
                print(f"  {nv:>4d} {N:>4d} {io:>6s} {m:8.1f}% {reached}/{len(accs):>4s} {N*N:>7d}{marker}",
                      flush=True)

    print(f"\n{'='*80}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
