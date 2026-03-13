"""
VRAXION v22 -- Chain Population Rebalancing
=============================================
v2 showed Refiner dominates (16/24 wins), Flipper/Scout barely contribute.
Test different slave mixes to find the optimal population.

Populations:
  A: Original   — 1 Scout + 1 Refiner + 1 Flipper + 1 Rewirer (4 slaves)
  B: Refiner-heavy — 2 Refiner(soft) + 1 Refiner(aggressive) + 1 Rewirer (4 slaves)
  C: Minimal    — 1 Refiner + 1 Rewirer (2 slaves, same budget/slave, 2x faster)
  D: Gradient   — Refiner(0.01) + Refiner(0.03) + Refiner(0.07) + Rewirer (4 slaves)
                   Different "zoom levels" of refinement
  E: Dynamic    — All 4 original, but winner gets 2x budget next round
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph


def score_batch(net, targets, V, ticks=8):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    return 0.5 * acc + 0.5 * tp, acc


# ============================================================
#  PARAMETRIC SLAVES
# ============================================================

def make_refiner(struct_rate=0.02, weight_prob=0.7):
    """Factory: refiner with configurable aggressiveness."""
    def fn(net, targets, V, budget, ticks=8):
        best_sc, best_acc = score_batch(net, targets, V, ticks)
        best_mask = net.mask.copy(); best_W = net.W.copy()
        for _ in range(budget):
            sm = net.mask.copy(); sw = net.W.copy()
            if random.random() < weight_prob:
                net.mutate_weights()
            else:
                net.mutate_structure(struct_rate)
            sc, ac = score_batch(net, targets, V, ticks)
            if sc > best_sc:
                best_sc = sc; best_acc = ac
                best_mask = net.mask.copy(); best_W = net.W.copy()
            else:
                net.mask = sm; net.W = sw
        return best_sc, best_acc, best_mask, best_W
    return fn


def make_scout(rate=0.15):
    def fn(net, targets, V, budget, ticks=8):
        best_sc, best_acc = score_batch(net, targets, V, ticks)
        best_mask = net.mask.copy(); best_W = net.W.copy()
        for _ in range(budget):
            sm = net.mask.copy(); sw = net.W.copy()
            net.mutate_structure(rate)
            sc, ac = score_batch(net, targets, V, ticks)
            if sc > best_sc:
                best_sc = sc; best_acc = ac
                best_mask = net.mask.copy(); best_W = net.W.copy()
            else:
                net.mask = sm; net.W = sw
        return best_sc, best_acc, best_mask, best_W
    return fn


def slave_rewirer(net, targets, V, budget, ticks=8):
    old_flip = net.flip_rate; net.flip_rate = 0.0
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy(); best_W = net.W.copy()
    for _ in range(budget):
        sm = net.mask.copy(); sw = net.W.copy()
        net.mutate_structure(0.05)
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc; best_acc = ac
            best_mask = net.mask.copy(); best_W = net.W.copy()
        else:
            net.mask = sm; net.W = sw
    net.flip_rate = old_flip
    return best_sc, best_acc, best_mask, best_W


# ============================================================
#  POPULATIONS
# ============================================================

POPULATIONS = {
    "A_original": [
        ("Scout",   make_scout(0.15)),
        ("Refiner", make_refiner(0.02, 0.7)),
        ("Flipper", make_refiner(0.05, 0.0)),  # structure-only, like old flipper
        ("Rewirer", slave_rewirer),
    ],
    "B_refiner_heavy": [
        ("Ref_soft",  make_refiner(0.01, 0.8)),   # very gentle
        ("Ref_mid",   make_refiner(0.03, 0.6)),   # medium
        ("Ref_aggr",  make_refiner(0.07, 0.4)),   # aggressive
        ("Rewirer",   slave_rewirer),
    ],
    "C_minimal": [
        ("Refiner",  make_refiner(0.02, 0.7)),
        ("Rewirer",  slave_rewirer),
    ],
    "D_gradient": [
        ("Ref_001",  make_refiner(0.01, 0.8)),   # microscope
        ("Ref_003",  make_refiner(0.03, 0.6)),   # magnifying glass
        ("Ref_007",  make_refiner(0.07, 0.4)),   # wide angle
        ("Rewirer",  slave_rewirer),              # structural
    ],
}


def train_population(net, targets, V, slaves, budget_per_slave=2000,
                     rounds=4, ticks=8, dynamic=False):
    """Train with a given slave population."""
    champion_sc, champion_acc = score_batch(net, targets, V, ticks)
    champion_mask = net.mask.copy()
    champion_W = net.W.copy()
    wins = {name: 0 for name, _ in slaves}
    budgets = {name: budget_per_slave for name, _ in slaves}

    for r in range(rounds):
        round_best_sc = champion_sc
        round_best_acc = champion_acc
        round_best_mask = champion_mask
        round_best_W = champion_W
        round_winner = None

        for name, fn in slaves:
            net.mask = champion_mask.copy()
            net.W = champion_W.copy()
            b = budgets[name]
            sc, ac, m, w = fn(net, targets, V, b, ticks)

            if sc > round_best_sc:
                round_best_sc = sc
                round_best_acc = ac
                round_best_mask = m
                round_best_W = w
                round_winner = name

        if round_best_sc > champion_sc:
            champion_sc = round_best_sc
            champion_acc = round_best_acc
            champion_mask = round_best_mask
            champion_W = round_best_W
            if round_winner:
                wins[round_winner] += 1

        # Dynamic: winner gets 2x budget, losers get 0.75x
        if dynamic and round_winner:
            for name, _ in slaves:
                if name == round_winner:
                    budgets[name] = min(budgets[name] * 2, budget_per_slave * 4)
                else:
                    budgets[name] = max(int(budgets[name] * 0.75), budget_per_slave // 4)

        if champion_acc >= 0.99:
            break

    net.mask = champion_mask.copy()
    net.W = champion_W.copy()

    total_att = sum(budgets[n] for n, _ in slaves) * rounds  # approximate
    return champion_acc, wins


# ============================================================
#  RUN
# ============================================================
print("=" * 65)
print("  CHAIN POPULATION REBALANCING")
print("=" * 65)

V = 64; N = 256
BUDGET = 2000
ROUNDS = 4
SEEDS = [42, 77, 123]

# Also test E_dynamic (original slaves + dynamic budget)
all_pops = list(POPULATIONS.items()) + [
    ("E_dynamic", POPULATIONS["A_original"]),  # same slaves, dynamic flag
]

for pop_name, slaves in all_pops:
    is_dynamic = (pop_name == "E_dynamic")
    n_slaves = len(slaves)
    total_compute = n_slaves * BUDGET * ROUNDS
    slave_names = "+".join(n for n, _ in slaves)

    print(f"\n  --- {pop_name} ({n_slaves} slaves, {total_compute//1000}K compute) ---")
    print(f"      [{slave_names}]")

    accs = []
    all_wins = {}

    for seed in SEEDS:
        np.random.seed(seed); random.seed(seed)
        perm = np.random.permutation(V)

        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net = SelfWiringGraph(N, V)
        t0 = time.time()
        acc, wins = train_population(net, perm, V, slaves, BUDGET, ROUNDS,
                                     dynamic=is_dynamic)
        elapsed = time.time() - t0

        accs.append(acc)
        for k, v in wins.items():
            all_wins[k] = all_wins.get(k, 0) + v

        print(f"    seed={seed:3d}  acc={acc*100:5.1f}%  ({elapsed:.1f}s)  ", end="")
        top_winner = max(wins, key=wins.get) if any(v > 0 for v in wins.values()) else "-"
        print(f"top={top_winner}")

    avg = np.mean(accs) * 100
    print(f"    AVG: {avg:.1f}%")
    if all_wins:
        print(f"    Wins: ", end="")
        for name, count in sorted(all_wins.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"{name}={count} ", end="")
        print()

# Single baseline for reference
print(f"\n  --- BASELINE: Single searcher (8K att) ---")
single_accs = []
for seed in SEEDS:
    np.random.seed(seed); random.seed(seed)
    perm = np.random.permutation(V)
    np.random.seed(seed + 1000); random.seed(seed + 1000)
    net = SelfWiringGraph(N, V)
    best_sc, best_acc = score_batch(net, perm, V)
    stale = 0
    for att in range(8000):
        sm = net.mask.copy(); sw = net.W.copy()
        if att < 2500:
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()
        sc, ac = score_batch(net, perm, V)
        if sc > best_sc:
            best_sc = sc; best_acc = ac; stale = 0
        else:
            net.mask = sm; net.W = sw; stale += 1
        if best_acc >= 0.99: break
        if stale >= 6000: break
    single_accs.append(best_acc)
    print(f"    seed={seed:3d}  acc={best_acc*100:5.1f}%")
print(f"    AVG: {np.mean(single_accs)*100:.1f}%")


print(f"\n{'='*65}")
print(f"  RANKING (V={V})")
print(f"{'='*65}", flush=True)
