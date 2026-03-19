"""Overcomplete Projection Sweep — does a larger passive IO space help?
=====================================================================
Tests the hypothesis: same info (V tokens), projected into increasingly
overcomplete hidden spaces (H >> V), gives the mask more "surface area"
to learn from — richer grounding.

Ratios tested: 3 (baseline), 9, 15, 21, 27
Tasks: English bigram V=27, synthetic permutation V=27

This is essentially testing whether an overcomplete / redundant
representation — like a reservoir — gives the mutation engine better
footholds for learning.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.passive_io import PassiveIOGraph, train_passive

SEEDS = [42, 77, 123]
BUDGET = 8000
STALE = 5000
H_RATIOS = [3, 6, 9, 12]

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
    """Build bigram target: for each char, most likely next char."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    chars = []
    for c in text.lower():
        if 'a' <= c <= 'z':
            chars.append(ord(c) - ord('a'))
        elif c == ' ':
            chars.append(26)
    for i in range(len(chars) - 1):
        counts[chars[i], chars[i + 1]] += 1
    return np.argmax(counts, axis=1)


def run_one(task_name, V, h_ratio, targets, seed):
    np.random.seed(seed)
    random.seed(seed)

    net = PassiveIOGraph(V, h_ratio=h_ratio, proj='random')
    H = net.H

    # Train
    best = train_passive(net, targets, V,
                         max_attempts=BUDGET, ticks=8,
                         stale_limit=STALE, verbose=False)

    return {
        'task': task_name,
        'V': V,
        'h_ratio': h_ratio,
        'H': H,
        'seed': seed,
        'score': best,
        'conns': net.count_connections(),
        'mask_cells': H * H,
    }


def main():
    V = 27
    bigram_targets = make_bigram_targets(ENGLISH_TEXT, V)

    jobs = []
    for hr in H_RATIOS:
        # English bigram
        for seed in SEEDS:
            jobs.append(('english_bigram', V, hr, bigram_targets, seed))

        # Synthetic random permutation
        for seed in SEEDS:
            np.random.seed(seed)
            perm = np.random.permutation(V)
            jobs.append(('synth_perm', V, hr, perm, seed))

    total = len(jobs)
    workers = min(total, max(1, multiprocessing.cpu_count() - 1))

    print(f"OVERCOMPLETE PROJECTION SWEEP", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"V={V}, h_ratios={H_RATIOS}, budget={BUDGET}, stale={STALE}", flush=True)
    print(f"Tasks: english_bigram, synth_perm | Seeds: {SEEDS}", flush=True)
    print(f"Total jobs: {total}, workers: {workers}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\nDimension summary:", flush=True)
    for hr in H_RATIOS:
        H = V * hr
        print(f"  h_ratio={hr:2d} → H={H:5d}, mask={H*H:>10,d} cells, "
              f"W_in=({V},{H})", flush=True)
    print(flush=True)

    all_results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            all_results.append(r)
            elapsed = time.time() - t0
            print(f"  [{i:2d}/{total}] {r['task']:14s} h_ratio={r['h_ratio']:2d} "
                  f"H={r['H']:5d} seed={r['seed']:3d}: "
                  f"{r['score']*100:5.1f}% conns={r['conns']:5d} "
                  f"({elapsed:.0f}s)", flush=True)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS  (total time: {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    for task in ['english_bigram', 'synth_perm']:
        print(f"\n{task}:", flush=True)
        print(f"  {'h_ratio':>7s} {'H':>5s} {'mask_cells':>10s} "
              f"{'mean':>7s} {'std':>6s} {'min':>6s} {'max':>6s} "
              f"{'conns':>6s}", flush=True)
        print(f"  {'-'*60}", flush=True)

        best_mean = 0
        rows = []
        for hr in H_RATIOS:
            scores = [r['score'] for r in all_results
                      if r['task'] == task and r['h_ratio'] == hr]
            conns = [r['conns'] for r in all_results
                     if r['task'] == task and r['h_ratio'] == hr]
            if scores:
                H = V * hr
                m = np.mean(scores) * 100
                s = np.std(scores) * 100
                mn = np.min(scores) * 100
                mx = np.max(scores) * 100
                mc = int(np.mean(conns))
                best_mean = max(best_mean, m)
                rows.append((hr, H, H*H, m, s, mn, mx, mc))

        for hr, H, mc_cells, m, s, mn, mx, mc in rows:
            marker = " <<<" if m >= best_mean - 0.1 else ""
            print(f"  {hr:7d} {H:5d} {mc_cells:>10,d} "
                  f"{m:6.1f}% {s:5.1f}pp {mn:5.1f}% {mx:5.1f}% "
                  f"{mc:5d}{marker}", flush=True)

    # Scaling analysis
    print(f"\n{'='*70}", flush=True)
    print(f"SCALING ANALYSIS — score improvement vs compute cost", flush=True)
    print(f"{'='*70}", flush=True)
    baseline_scores = {}
    for task in ['english_bigram', 'synth_perm']:
        baseline_scores[task] = np.mean([
            r['score'] for r in all_results
            if r['task'] == task and r['h_ratio'] == 3
        ]) * 100

    for task in ['english_bigram', 'synth_perm']:
        print(f"\n{task} (baseline h_ratio=3: {baseline_scores[task]:.1f}%):", flush=True)
        for hr in H_RATIOS:
            scores = [r['score'] for r in all_results
                      if r['task'] == task and r['h_ratio'] == hr]
            if scores:
                H = V * hr
                m = np.mean(scores) * 100
                delta = m - baseline_scores[task]
                cost_ratio = (H * H) / ((V * 3) ** 2)
                print(f"  h_ratio={hr:2d}: {m:5.1f}% (Δ={delta:+5.1f}pp) "
                      f"cost={cost_ratio:5.1f}x", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
