"""A/B comparison: plain rewire-only vs cyclic (rewire + crystal).
Same seed, same budget, same V. Which one wins on score AND sparsity?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

BUDGET = 50000
V = 16
SEEDS = [42, 77, 123, 256, 999]

print(f"{'='*70}")
print(f"  A/B TEST: Plain Rewire vs Cyclic (Rewire + Strict Crystal)")
print(f"  V={V}, budget={BUDGET}, seeds={SEEDS}")
print(f"{'='*70}\n")

results_plain = []
results_cyclic = []

for seed in SEEDS:
    print(f"\n{'─'*70}")
    print(f"  SEED {seed}")
    print(f"{'─'*70}")

    # ── A) Plain rewire-only (use train_cyclic with crystal_budget=0) ──
    np.random.seed(seed)
    targets = np.random.randint(0, V, size=V)
    net_a = SelfWiringGraph(V)
    init_conns = net_a.count_connections()

    print(f"\n  [A] PLAIN REWIRE (crystal_budget=0)")
    sc_a, acc_a, kept_a, cyc_a = train_cyclic(
        net_a, targets, V,
        score_fn=score_batch,
        ticks=8,
        max_att=BUDGET,
        stale_limit=BUDGET,  # never stop from stale — use full budget on rewire
        add_every=50,
        crystal_budget=0,    # NO crystal phase
        crystal_window=300,
        crystal_min_rate=0.003,
        verbose=False,
    )
    conns_a = net_a.count_connections()
    print(f"  → Score: {sc_a*100:.1f}%  Acc: {acc_a*100:.0f}%  "
          f"Conns: {init_conns}→{conns_a}  Kept: {kept_a}")
    results_plain.append((sc_a, acc_a, conns_a, kept_a))

    # ── B) Cyclic (rewire + strict crystal) ──
    np.random.seed(seed)
    targets = np.random.randint(0, V, size=V)
    net_b = SelfWiringGraph(V)

    print(f"  [B] CYCLIC (rewire + strict crystal)")
    sc_b, acc_b, kept_b, cyc_b = train_cyclic(
        net_b, targets, V,
        score_fn=score_batch,
        ticks=8,
        max_att=BUDGET,
        stale_limit=2000,
        add_every=50,
        crystal_budget=5000,
        crystal_window=300,
        crystal_min_rate=0.003,
        verbose=False,
    )
    conns_b = net_b.count_connections()
    print(f"  → Score: {sc_b*100:.1f}%  Acc: {acc_b*100:.0f}%  "
          f"Conns: {init_conns}→{conns_b}  Kept: {kept_b}  Cycles: {cyc_b}")
    results_cyclic.append((sc_b, acc_b, conns_b, kept_b))

    delta_sc = (sc_b - sc_a) * 100
    delta_conns = conns_a - conns_b
    winner = "CYCLIC" if (sc_b >= sc_a and conns_b <= conns_a) else \
             "PLAIN" if (sc_a > sc_b and conns_a <= conns_b) else "MIXED"
    print(f"  Δscore: {delta_sc:+.1f}pp  Δconns: {delta_conns:+d}  → {winner}")

# ── Summary ──
print(f"\n{'='*70}")
print(f"  SUMMARY across {len(SEEDS)} seeds")
print(f"{'='*70}")
print(f"  {'':>8} {'Score':>8} {'Acc':>6} {'Conns':>8}")
print(f"  {'PLAIN':>8} {np.mean([r[0] for r in results_plain])*100:7.1f}% "
      f"{np.mean([r[1] for r in results_plain])*100:5.0f}% "
      f"{np.mean([r[2] for r in results_plain]):7.0f}")
print(f"  {'CYCLIC':>8} {np.mean([r[0] for r in results_cyclic])*100:7.1f}% "
      f"{np.mean([r[1] for r in results_cyclic])*100:5.0f}% "
      f"{np.mean([r[2] for r in results_cyclic]):7.0f}")

avg_sc_delta = np.mean([c[0]-p[0] for p,c in zip(results_plain, results_cyclic)]) * 100
avg_conn_delta = np.mean([p[2]-c[2] for p,c in zip(results_plain, results_cyclic)])
print(f"\n  Δscore (cyclic - plain): {avg_sc_delta:+.1f}pp")
print(f"  Δconns (plain - cyclic): {avg_conn_delta:+.0f} fewer edges with cyclic")
print(f"  Sparsity gain: {avg_conn_delta/np.mean([r[2] for r in results_plain])*100:.0f}% fewer connections")
