"""
FINAL HEAD-TO-HEAD: PassiveIO-scaled3x vs Original Active IO
=============================================================
10 seeds, 15k budget — definitive comparison.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph, train
from model.passive_io import PassiveIOGraph, train_passive

V = 27
BUDGET = 15_000
TICKS = 8
STALE = 12_000
SEEDS = [42, 123, 777, 999, 314, 55, 1337, 2024, 7, 8888]


def run_original(seed):
    np.random.seed(seed)
    pyrandom.seed(seed)
    net = SelfWiringGraph(V)
    targets = np.random.permutation(V)
    t0 = time.time()
    score = train(net, targets, V, max_attempts=BUDGET, ticks=TICKS,
                  stale_limit=STALE, verbose=False)
    dt = time.time() - t0
    return score, net.count_connections(), dt


def run_passive_3x(seed):
    np.random.seed(seed)
    pyrandom.seed(seed)
    # Build net, then scale projections
    net = PassiveIOGraph(V, h_ratio=3, proj='random')
    # Deterministic projection from separate rng
    rng = np.random.RandomState(seed + 10000)
    input_projection = rng.randn(V, V * 3).astype(np.float32)
    input_projection /= np.linalg.norm(input_projection, axis=1, keepdims=True)
    output_projection = rng.randn(V * 3, V).astype(np.float32)
    output_projection /= np.linalg.norm(output_projection, axis=0, keepdims=True)
    net.input_projection = input_projection * 3.0
    net.output_projection = output_projection * 3.0
    targets = np.random.permutation(V)
    t0 = time.time()
    score = train_passive(net, targets, V, max_attempts=BUDGET, ticks=TICKS,
                          stale_limit=STALE, verbose=False)
    dt = time.time() - t0
    return score, net.count_connections(), dt


print(f"{'='*80}")
print(f"  FINAL: Original Active IO vs PassiveIO-scaled3x")
print(f"  V={V}, Budget={BUDGET}, Ticks={TICKS}, Seeds={len(SEEDS)}")
print(f"{'='*80}\n")

orig_scores, pass_scores = [], []
orig_conns, pass_conns = [], []
orig_times, pass_times = [], []

print(f"  {'Seed':>6s}  │ {'Original':>10s} {'Conns':>6s} {'Time':>5s}  │ "
      f"{'Passive3x':>10s} {'Conns':>6s} {'Time':>5s}  │ {'Winner':>10s} {'Delta':>6s}")
print(f"  {'─'*85}")

for seed in SEEDS:
    s1, c1, t1 = run_original(seed)
    s2, c2, t2 = run_passive_3x(seed)
    orig_scores.append(s1); pass_scores.append(s2)
    orig_conns.append(c1); pass_conns.append(c2)
    orig_times.append(t1); pass_times.append(t2)

    winner = "PASSIVE" if s2 > s1 else "ORIGINAL" if s1 > s2 else "TIE"
    delta = (s2 - s1) * 100
    print(f"  {seed:>6d}  │ {s1*100:>9.1f}% {c1:>6d} {t1:>4.1f}s  │ "
          f"{s2*100:>9.1f}% {c2:>6d} {t2:>4.1f}s  │ {winner:>10s} {delta:>+5.1f}pp")

print(f"  {'─'*85}")

# Summary
o_avg = np.mean(orig_scores)
p_avg = np.mean(pass_scores)
o_std = np.std(orig_scores)
p_std = np.std(pass_scores)
wins_passive = sum(1 for s1, s2 in zip(orig_scores, pass_scores) if s2 > s1)
wins_original = sum(1 for s1, s2 in zip(orig_scores, pass_scores) if s1 > s2)
ties = len(SEEDS) - wins_passive - wins_original

print(f"\n  ÖSSZESÍTÉS:")
print(f"  {'':>20s}  {'Original':>12s}  {'Passive-3x':>12s}")
print(f"  {'─'*50}")
print(f"  {'Avg score':<20s}  {o_avg*100:>11.1f}%  {p_avg*100:>11.1f}%")
print(f"  {'Std':<20s}  {o_std*100:>11.1f}%  {p_std*100:>11.1f}%")
print(f"  {'Best':<20s}  {max(orig_scores)*100:>11.1f}%  {max(pass_scores)*100:>11.1f}%")
print(f"  {'Worst':<20s}  {min(orig_scores)*100:>11.1f}%  {min(pass_scores)*100:>11.1f}%")
print(f"  {'Median':<20s}  {np.median(orig_scores)*100:>11.1f}%  {np.median(pass_scores)*100:>11.1f}%")
print(f"  {'Avg conns':<20s}  {np.mean(orig_conns):>11.0f}  {np.mean(pass_conns):>11.0f}")
print(f"  {'Avg time':<20s}  {np.mean(orig_times):>10.1f}s  {np.mean(pass_times):>10.1f}s")
print(f"\n  Win/Loss/Tie:  Passive {wins_passive} — Original {wins_original} — Tie {ties}")
print(f"  Avg delta:     {(p_avg - o_avg)*100:+.1f} percentage points")

print(f"\n{'='*80}")
print("DONE")
