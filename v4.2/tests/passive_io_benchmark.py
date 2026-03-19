"""
PassiveIO vs Original — Head-to-head benchmark
================================================
Compares:
  1. Original SelfWiringGraph (IN/OUT are neurons)
  2. PassiveIOGraph proj='random'
  3. PassiveIOGraph proj='identity'
  4. PassiveIOGraph proj='hadamard'

Same budget, same seeds, same task.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph, train
from model.passive_io import PassiveIOGraph, train_passive

SEEDS = [42, 123, 777]
V = 27
BUDGET = 10_000
TICKS = 8
STALE = 8_000

results = {}

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

def run_passive(seed, proj):
    np.random.seed(seed)
    pyrandom.seed(seed)
    net = PassiveIOGraph(V, h_ratio=3, proj=proj)
    targets = np.random.permutation(V)
    t0 = time.time()
    score = train_passive(net, targets, V, max_attempts=BUDGET, ticks=TICKS,
                          stale_limit=STALE, verbose=False)
    dt = time.time() - t0
    return score, net.count_connections(), dt


configs = [
    ("Original", run_original),
    ("Passive-random", lambda s: run_passive(s, 'random')),
    ("Passive-identity", lambda s: run_passive(s, 'identity')),
    ("Passive-hadamard", lambda s: run_passive(s, 'hadamard')),
]

print(f"{'='*80}")
print(f"  PassiveIO vs Original Benchmark")
print(f"  V={V}, Budget={BUDGET}, Ticks={TICKS}, Seeds={SEEDS}")
print(f"{'='*80}")

all_results = {}
for name, fn in configs:
    all_results[name] = []
    for seed in SEEDS:
        print(f"  Running {name} seed={seed}...", end=" ", flush=True)
        score, conns, dt = fn(seed)
        all_results[name].append((score, conns, dt))
        print(f"score={score*100:.1f}% conns={conns} time={dt:.1f}s")

# ── Summary ──
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  {'Config':<20s} │ {'Avg Score':>10s} │ {'Avg Conns':>10s} │ {'Avg Time':>9s}")
print(f"  {'─'*60}")
for name in all_results:
    scores = [r[0] for r in all_results[name]]
    conns = [r[1] for r in all_results[name]]
    times = [r[2] for r in all_results[name]]
    print(f"  {name:<20s} │ {np.mean(scores)*100:>9.1f}% │ {np.mean(conns):>10.0f} │ {np.mean(times):>8.1f}s")

# ── Per-seed comparison ──
print(f"\n  PER-SEED SCORES:")
print(f"  {'Seed':<6s}", end="")
for name in all_results:
    print(f"  {name:>18s}", end="")
print()
print(f"  {'─'*80}")
for i, seed in enumerate(SEEDS):
    print(f"  {seed:<6d}", end="")
    for name in all_results:
        s = all_results[name][i][0]
        print(f"  {s*100:>17.1f}%", end="")
    print()

# ── Detailed PassiveIO topology for one run ──
print(f"\n{'='*80}")
print(f"  PassiveIO-identity topology detail (seed=42)")
print(f"{'='*80}")
np.random.seed(42)
pyrandom.seed(42)
net = PassiveIOGraph(V, h_ratio=3, proj='identity')
targets = np.random.permutation(V)
train_passive(net, targets, V, max_attempts=BUDGET, ticks=TICKS,
              stale_limit=STALE, verbose=True)

mask = net.mask
H = net.H
alive = (mask != 0)
n_edges = int(alive.sum())
in_deg = alive.sum(axis=0)
out_deg = alive.sum(axis=1)
total_deg = in_deg + out_deg

print(f"\n  Hidden-only mask: {H}×{H}")
print(f"  Edges: {n_edges} / {H*(H-1)} ({n_edges/(H*(H-1))*100:.1f}%)")
print(f"  In-degree:  min={int(in_deg.min())} max={int(in_deg.max())} mean={in_deg.mean():.1f}")
print(f"  Out-degree: min={int(out_deg.min())} max={int(out_deg.max())} mean={out_deg.mean():.1f}")

# W_in and W_out analysis
print(f"\n  W_in shape: {net.W_in.shape}  (fixed, not learned)")
print(f"  W_out shape: {net.W_out.shape}  (fixed, not learned)")
print(f"  W_in sparsity: {(net.W_in == 0).mean()*100:.0f}%")
print(f"  W_out sparsity: {(net.W_out == 0).mean()*100:.0f}%")

# Check: can signals actually flow through?
print(f"\n  Signal flow test (identity proj):")
for inp_idx in range(min(3, V)):
    inp = np.zeros(V, dtype=np.float32)
    inp[inp_idx] = 1.0
    projected = inp @ net.W_in
    nonzero = int((projected != 0).sum())
    print(f"    Input[{inp_idx}] → {nonzero}/{H} hidden neurons activated by projection")

print(f"\n{'='*80}")
print("DONE")
