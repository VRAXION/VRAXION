"""Sweep: which add strategy works best during the rewire phase?
Tests different add frequencies, burst sizes, and frontloading."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
SEEDS = [42, 77, 123]
MAX_ATT = 50000

STRATEGIES = [
    {"name": "no_adds",         "add_every": 0,   "add_burst": 1},
    {"name": "very_sparse",     "add_every": 200, "add_burst": 1},
    {"name": "sparse",          "add_every": 100, "add_burst": 1},
    {"name": "moderate",        "add_every": 50,  "add_burst": 1},  # current default
    {"name": "frequent",        "add_every": 25,  "add_burst": 1},
    {"name": "very_frequent",   "add_every": 10,  "add_burst": 1},
    {"name": "burst_sparse",    "add_every": 200, "add_burst": 5},
    {"name": "burst_moderate",  "add_every": 100, "add_burst": 3},
    {"name": "frontload",       "add_every": 200, "add_burst": 1,
     "frontload_until": 500, "frontload_every": 10},
]

# Shared crystal settings
CRYSTAL = dict(
    crystal_budget=5000,
    crystal_window=300,
    crystal_min_rate=0.003,
)

print(f"{'='*75}")
print(f"  ADD STRATEGY SWEEP — V={V}, budget={MAX_ATT}, seeds={SEEDS}")
print(f"{'='*75}")

all_results = {}

for strat in STRATEGIES:
    name = strat["name"]
    kwargs = {k: v for k, v in strat.items() if k != "name"}
    all_results[name] = []

    for seed in SEEDS:
        np.random.seed(seed)
        targets = np.random.randint(0, V, size=V)
        net = SelfWiringGraph(V)

        sc, acc, kept, cycles = train_cyclic(
            net, targets, V,
            score_fn=score_batch,
            ticks=8,
            max_att=MAX_ATT,
            stale_limit=2000,
            **kwargs,
            **CRYSTAL,
            verbose=False,
        )
        conns = net.count_connections()
        all_results[name].append({
            "score": sc, "acc": acc, "conns": conns,
            "kept": kept, "cycles": cycles,
        })
        print(f"  {name:20s} seed={seed} → score={sc*100:.1f}% "
              f"acc={acc*100:.0f}% conns={conns:4d} cycles={cycles}")

# ── Summary table ──
print(f"\n{'='*75}")
print(f"  SUMMARY (mean over {len(SEEDS)} seeds)")
print(f"{'='*75}")
print(f"  {'strategy':20s} {'score':>7s} {'acc':>5s} {'conns':>6s} {'cycles':>7s}")
print(f"  {'-'*20} {'-'*7} {'-'*5} {'-'*6} {'-'*7}")

# Collect for ranking
summary = []
for name in [s["name"] for s in STRATEGIES]:
    runs = all_results[name]
    mean_sc = np.mean([r["score"] for r in runs])
    mean_acc = np.mean([r["acc"] for r in runs])
    mean_conns = np.mean([r["conns"] for r in runs])
    mean_cyc = np.mean([r["cycles"] for r in runs])
    summary.append((name, mean_sc, mean_acc, mean_conns, mean_cyc))
    print(f"  {name:20s} {mean_sc*100:6.1f}% {mean_acc*100:4.0f}% "
          f"{mean_conns:5.0f}  {mean_cyc:6.1f}")

# ── Rank by composite: score - 0.001 * conns (reward sparsity) ──
print(f"\n  RANKING (score - 0.001*conns composite):")
ranked = sorted(summary, key=lambda x: x[1] - 0.001 * x[3], reverse=True)
for i, (name, sc, acc, conns, cyc) in enumerate(ranked):
    composite = sc - 0.001 * conns
    marker = " ★" if i == 0 else ""
    print(f"  {i+1}. {name:20s}  composite={composite*100:.2f}  "
          f"(score={sc*100:.1f}% conns={conns:.0f}){marker}")
