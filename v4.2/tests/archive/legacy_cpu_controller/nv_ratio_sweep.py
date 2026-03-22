"""N/V ratio sweep on synthetic + English bigram tasks.

Tests N = 2V, 3V, 4V, 5V across:
- Synthetic random permutation (V=64)
- English letter bigram (V=27: a-z + space, from real text)
- Synthetic random permutation (V=128)

Is the middle "thinking" layer needed? Or can input→output direct?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph
from collections import Counter

SEEDS = [42, 77, 123]
BUDGET = 32000
NV_RATIOS = [2, 3, 4, 5]

# English text for bigram task
ENGLISH_TEXT = """the quick brown fox jumps over the lazy dog and the cat sat on the mat
while the rain in spain falls mainly on the plain the old man and the sea
to be or not to be that is the question whether tis nobler in the mind
it was the best of times it was the worst of times it was the age of wisdom
all happy families are alike each unhappy family is unhappy in its own way
call me ishmael some years ago never mind how long precisely having little
it is a truth universally acknowledged that a single man in possession of
in the beginning god created the heavens and the earth and the earth was
one morning when gregor samsa woke from troubled dreams he found himself
far out in the uncharted backwaters of the unfashionable end of the western"""


def make_bigram_targets(text, vocab_size=27):
    """Build bigram target: for each char, what's the most likely next char?
    a=0, b=1, ..., z=25, space=26"""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    chars = []
    for c in text.lower():
        if 'a' <= c <= 'z':
            chars.append(ord(c) - ord('a'))
        elif c == ' ':
            chars.append(26)
    for i in range(len(chars) - 1):
        counts[chars[i], chars[i + 1]] += 1
    # Target: most likely next char for each input char
    targets = np.argmax(counts, axis=1)
    return targets


def evaluate(net, targets, V, ticks=8):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def run_one(task_name, V, nv_ratio, targets, seed):
    N = V * nv_ratio
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V)

    score = evaluate(net, targets, V)
    best = score
    stale = 0

    for att in range(BUDGET):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_score = evaluate(net, targets, V)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if random.random() < net.PATIENCE:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < net.PATIENCE:
                net.grow = np.int8(1 - int(net.grow))

        if best >= 0.99 or stale >= 6000:
            break

    return {
        'task': task_name, 'V': V, 'ratio': nv_ratio, 'N': N,
        'seed': seed, 'score': best, 'conns': net.count_connections(),
    }


def main():
    # Build tasks
    bigram_targets = make_bigram_targets(ENGLISH_TEXT, 27)

    jobs = []
    for nv in NV_RATIOS:
        # Synthetic perm V=64
        for seed in SEEDS:
            np.random.seed(seed)
            perm64 = np.random.permutation(64)
            jobs.append(('synth_V64', 64, nv, perm64, seed))

        # English bigram V=27
        for seed in SEEDS:
            jobs.append(('english_V27', 27, nv, bigram_targets, seed))

        # Synthetic perm V=128 (slower)
        for seed in SEEDS:
            np.random.seed(seed)
            perm128 = np.random.permutation(128)
            jobs.append(('synth_V128', 128, nv, perm128, seed))

    total = len(jobs)
    print(f"N/V RATIO SWEEP: {total} jobs, 22 workers, {BUDGET} budget", flush=True)
    print(f"Ratios: {NV_RATIOS}, Tasks: synth_V64, english_V27, synth_V128", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    with ProcessPoolExecutor(max_workers=22) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            print(f"  {r['task']:12s} N/V={r['ratio']} seed={r['seed']}: "
                  f"{r['score']*100:5.1f}% N={r['N']} conns={r['conns']}", flush=True)

    # Summary per task
    print(f"\n{'='*80}", flush=True)
    for task in ['synth_V64', 'english_V27', 'synth_V128']:
        print(f"\n{task}:", flush=True)
        print(f"  {'N/V':>5s} {'N':>5s} {'mean':>7s} {'std':>6s} {'matmul_size':>12s}", flush=True)
        for nv in NV_RATIOS:
            scores = [r['score'] for r in all_results
                      if r['task'] == task and r['ratio'] == nv]
            if scores:
                V = [r['V'] for r in all_results
                     if r['task'] == task and r['ratio'] == nv][0]
                N = V * nv
                m = np.mean(scores) * 100
                s = np.std(scores) * 100
                marker = " <<<" if m >= max(
                    np.mean([r['score'] for r in all_results
                             if r['task'] == task and r['ratio'] == rr]) * 100
                    for rr in NV_RATIOS) - 0.1 else ""
                print(f"  {nv:>5d} {N:>5d} {m:6.1f}% {s:5.1f}pp {N*N:>10d} cells{marker}",
                      flush=True)

    print(f"\n{'='*80}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
