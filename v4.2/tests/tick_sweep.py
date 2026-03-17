"""Sweep: what tick count is optimal?
Fair test: sweep across multiple seeds, report per-seed and average."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
SEEDS = [42, 77, 123, 256, 999]
MAX_ATT = 40000
TICK_VALUES = [2, 4, 6, 8, 10, 12, 16]

print(f"{'='*75}")
print(f"  TICK SWEEP — V={V}, budget={MAX_ATT}, seeds={SEEDS}")
print(f"  Tick values: {TICK_VALUES}")
print(f"{'='*75}")

# results[ticks] = [(score, acc, conns), ...]
results = {t: [] for t in TICK_VALUES}

for seed in SEEDS:
    print(f"\n  --- Seed {seed} ---")
    for ticks in TICK_VALUES:
        np.random.seed(seed)
        targets = np.random.randint(0, V, size=V)
        net = SelfWiringGraph(V)

        sc, acc, kept, cyc = train_cyclic(
            net, targets, V,
            score_fn=score_batch,
            ticks=ticks,
            max_att=MAX_ATT,
            stale_limit=2000,
            crystal_budget=5000,
            crystal_window=300,
            crystal_min_rate=0.003,
            verbose=False,
        )
        conns = net.count_connections()
        results[ticks].append((sc, acc, conns))
        print(f"    ticks={ticks:2d} → score={sc*100:.1f}% acc={acc*100:.0f}% conns={conns:4d}")

# ── Summary ──
print(f"\n{'='*75}")
print(f"  SUMMARY (mean ± std over {len(SEEDS)} seeds)")
print(f"{'='*75}")
print(f"  {'ticks':>5s} {'score':>10s} {'acc':>8s} {'conns':>8s}")
print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*8}")

best_composite = -999
best_tick = None
for t in TICK_VALUES:
    scores = [r[0] for r in results[t]]
    accs = [r[1] for r in results[t]]
    conns = [r[2] for r in results[t]]
    mean_sc = np.mean(scores)
    std_sc = np.std(scores)
    mean_acc = np.mean(accs)
    mean_conns = np.mean(conns)
    composite = mean_sc - 0.001 * mean_conns
    if composite > best_composite:
        best_composite = composite
        best_tick = t
    marker = ""
    print(f"  {t:5d} {mean_sc*100:5.1f}±{std_sc*100:.1f}% "
          f"{mean_acc*100:5.0f}%   {mean_conns:6.0f}{marker}")

print(f"\n  Best composite: ticks={best_tick} (score - 0.001*conns = {best_composite*100:.2f})")

# Per-seed best tick
print(f"\n  Per-seed best tick (by score):")
for i, seed in enumerate(SEEDS):
    seed_results = [(t, results[t][i][0]) for t in TICK_VALUES]
    best_t, best_s = max(seed_results, key=lambda x: x[1])
    print(f"    seed={seed}: best ticks={best_t} (score={best_s*100:.1f}%)")
