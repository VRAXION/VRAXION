"""Quick fill: patience 8 and 9 between the 7-10 gap."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def run_one(patience, seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)
    score = evaluate(net, perm)
    best = score
    accepts = 0
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
            elif rate < 0.15:
                if random.random() < 0.5:
                    net.signal = np.int8(1 - int(net.signal))
                else:
                    net.grow = np.int8(1 - int(net.grow))
            accepts = 0
    return patience, seed, best


def main():
    jobs = [(p, s) for p in [8, 9] for s in SEEDS]
    print(f"PATIENCE 8-9 GAP FILL, {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        results = {}
        for fut in as_completed(futures):
            p, seed, score = fut.result()
            results.setdefault(p, []).append(score)
            print(f"  p={p} seed={seed}: {score*100:.1f}%", flush=True)

    for p in [8, 9]:
        scores = results[p]
        print(f"\np={p}: mean={np.mean(scores)*100:.1f}% std={np.std(scores)*100:.1f}pp", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
