"""
VRAXION v22 -- Chain Ensemble v2: Full Budget per Slave
========================================================
v1 tanulsag: budget splitting = vesztes. A slave-eknek TOBB ido kell.

v2 strategia:
  - Minden slave TELJES budget-et kap (nem osztjuk el)
  - Kevesebb round, de slave-enkent SOKKAL tobb attempt
  - Alternativa: same total compute, de 4x tobb munka (fair: 4 core)

Comparison:
  A: Single searcher, 8K attempts
  B: Chain v2 — 4 slaves x 2K attempts x 4 rounds = 32K total (4x compute)
  C: Chain v2 "fair" — 4 slaves x 500 att x 4 rounds = 8K total (same compute)
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
#  SLAVE STRATEGIES (same as v1)
# ============================================================

def slave_scout(net, targets, V, budget, ticks=8):
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy(); best_W = net.W.copy()
    for _ in range(budget):
        sm = net.mask.copy(); sw = net.W.copy()
        net.mutate_structure(0.15)
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc; best_acc = ac
            best_mask = net.mask.copy(); best_W = net.W.copy()
        else:
            net.mask = sm; net.W = sw
    return best_sc, best_acc, best_mask, best_W


def slave_refiner(net, targets, V, budget, ticks=8):
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy(); best_W = net.W.copy()
    for _ in range(budget):
        sm = net.mask.copy(); sw = net.W.copy()
        if random.random() < 0.7:
            net.mutate_weights()
        else:
            net.mutate_structure(0.02)
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc; best_acc = ac
            best_mask = net.mask.copy(); best_W = net.W.copy()
        else:
            net.mask = sm; net.W = sw
    return best_sc, best_acc, best_mask, best_W


def slave_flipper(net, targets, V, budget, ticks=8):
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy(); best_W = net.W.copy()
    for _ in range(budget):
        sm = net.mask.copy()
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            net.mask[idx[:, 0], idx[:, 1]] *= -1
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc; best_acc = ac
            best_mask = net.mask.copy(); best_W = net.W.copy()
        else:
            net.mask = sm
    return best_sc, best_acc, best_mask, best_W


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


SLAVES = [
    ("Scout",   slave_scout),
    ("Refiner", slave_refiner),
    ("Flipper", slave_flipper),
    ("Rewirer", slave_rewirer),
]


def train_chain_v2(net, targets, V, budget_per_slave, rounds=4, ticks=8, verbose=True):
    """Each slave gets full budget_per_slave per round. No splitting."""
    champion_sc, champion_acc = score_batch(net, targets, V, ticks)
    champion_mask = net.mask.copy()
    champion_W = net.W.copy()
    wins = {name: 0 for name, _ in SLAVES}
    total_att = 0

    for r in range(rounds):
        round_best_sc = champion_sc
        round_best_mask = champion_mask
        round_best_W = champion_W
        round_best_acc = champion_acc
        round_winner = None

        for name, fn in SLAVES:
            net.mask = champion_mask.copy()
            net.W = champion_W.copy()
            sc, ac, m, w = fn(net, targets, V, budget_per_slave, ticks)
            total_att += budget_per_slave

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

        if verbose:
            w = round_winner if round_winner else "-"
            print(f"      round {r+1}: acc={champion_acc*100:5.1f}%  winner={w}")

        if champion_acc >= 0.99:
            break

    net.mask = champion_mask.copy()
    net.W = champion_W.copy()
    return champion_acc, total_att, wins


def train_single(net, targets, V, total_attempts=8000, ticks=8):
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    stale = 0
    for att in range(total_attempts):
        sm = net.mask.copy(); sw = net.W.copy()
        if att < 2500:
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc; best_acc = ac; stale = 0
        else:
            net.mask = sm; net.W = sw; stale += 1
        if best_acc >= 0.99: break
        if stale >= 6000: break
    return best_acc


# ============================================================
#  RUN
# ============================================================
print("=" * 65)
print("  CHAIN ENSEMBLE v2 — Full Budget per Slave")
print("=" * 65)

CONFIGS = [
    # V,   N,   single_att, slave_budget, rounds
    (32,  128,  8000,       2000,         4),    # chain: 4*2000*4 = 32K (4x compute)
    (64,  256,  8000,       2000,         4),    # chain: 32K
]

SEEDS = [42, 77, 123]

for V, N, S_ATT, SL_BUD, ROUNDS in CONFIGS:
    chain_total = len(SLAVES) * SL_BUD * ROUNDS
    print(f"\n  --- V={V}, N={N} ---")
    print(f"  Single: {S_ATT} att | Chain: {len(SLAVES)}x{SL_BUD}x{ROUNDS}r = {chain_total} att ({chain_total/S_ATT:.0f}x compute)")
    print()

    single_accs = []
    chain_accs = []
    all_wins = {}

    for seed in SEEDS:
        np.random.seed(seed); random.seed(seed)
        perm = np.random.permutation(V)

        # A: Single
        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_s = SelfWiringGraph(N, V)
        t0 = time.time()
        acc_s = train_single(net_s, perm, V, S_ATT)
        t_s = time.time() - t0

        # B: Chain v2
        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_c = SelfWiringGraph(N, V)
        t0 = time.time()
        print(f"    seed={seed} chain rounds:")
        acc_c, actual, wins = train_chain_v2(net_c, perm, V, SL_BUD, ROUNDS)
        t_c = time.time() - t0

        single_accs.append(acc_s)
        chain_accs.append(acc_c)
        for k, v in wins.items():
            all_wins[k] = all_wins.get(k, 0) + v

        delta = (acc_c - acc_s) * 100
        print(f"    => single={acc_s*100:5.1f}%  chain={acc_c*100:5.1f}%  "
              f"delta={delta:+5.1f}%  ({t_s:.1f}s vs {t_c:.1f}s)")
        print()

    avg_s = np.mean(single_accs) * 100
    avg_c = np.mean(chain_accs) * 100
    print(f"    AVG: single={avg_s:.1f}%  chain={avg_c:.1f}%  delta={avg_c - avg_s:+.1f}%")
    if all_wins:
        print(f"    Wins: ", end="")
        for name, count in sorted(all_wins.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"{name}={count} ", end="")
        print()


print(f"\n{'='*65}")
print(f"  INSIGHT")
print(f"{'='*65}")
print(f"  v1 (budget split) lost because slaves had too few attempts.")
print(f"  v2 gives full budget per slave (4x total compute).")
print(f"  If chain STILL loses: specialization doesn't help this arch.")
print(f"  If chain wins: diversity helps, but only with enough budget.")
print(f"{'='*65}", flush=True)
