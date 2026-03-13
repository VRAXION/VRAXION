"""
VRAXION v22 -- Chain Ensemble: Master + Specialized Slaves
===========================================================
Master holds the champion. N slaves propose mutations with different
strategies. Master picks the best, broadcasts as new jackpot.

Slave specializations:
  - Scout:    high mutation rate (0.15), structure only, fast & reckless
  - Refiner:  low mutation rate (0.02), weights + light structure, careful
  - Flipper:  only flip mutations (the strongest operator, flip_rate=1.0)
  - Rewirer:  only rewire/add/remove (structural diversity, flip_rate=0.0)

Comparison:
  A: Single searcher (baseline, same total attempts)
  B: Chain ensemble (same total attempts split across slaves)
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
#  SLAVE STRATEGIES
# ============================================================

def slave_scout(net, targets, V, budget, ticks=8):
    """Aggressive explorer: high mutation rate, structure only."""
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy()
    best_W = net.W.copy()

    for _ in range(budget):
        saved_m = net.mask.copy()
        saved_w = net.W.copy()
        net.mutate_structure(0.15)  # aggressive rate
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            best_mask = net.mask.copy()
            best_W = net.W.copy()
        else:
            net.mask = saved_m
            net.W = saved_w

    return best_sc, best_acc, best_mask, best_W


def slave_refiner(net, targets, V, budget, ticks=8):
    """Careful tuner: low mutation rate, mostly weights."""
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy()
    best_W = net.W.copy()

    for _ in range(budget):
        saved_m = net.mask.copy()
        saved_w = net.W.copy()
        if random.random() < 0.7:
            net.mutate_weights()
        else:
            net.mutate_structure(0.02)  # gentle
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            best_mask = net.mask.copy()
            best_W = net.W.copy()
        else:
            net.mask = saved_m
            net.W = saved_w

    return best_sc, best_acc, best_mask, best_W


def slave_flipper(net, targets, V, budget, ticks=8):
    """Flip specialist: only sign flips (strongest mutation operator)."""
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy()
    best_W = net.W.copy()

    for _ in range(budget):
        saved_m = net.mask.copy()
        # Manual flip: toggle signs on random alive connections
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            rows, cols = idx[:, 0], idx[:, 1]
            net.mask[rows, cols] *= -1

        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            best_mask = net.mask.copy()
            best_W = net.W.copy()
        else:
            net.mask = saved_m

    return best_sc, best_acc, best_mask, best_W


def slave_rewirer(net, targets, V, budget, ticks=8):
    """Structural diversity: add/remove/rewire, no flips."""
    old_flip = net.flip_rate
    net.flip_rate = 0.0  # disable flip branch

    best_sc, best_acc = score_batch(net, targets, V, ticks)
    best_mask = net.mask.copy()
    best_W = net.W.copy()

    for _ in range(budget):
        saved_m = net.mask.copy()
        saved_w = net.W.copy()
        net.mutate_structure(0.05)
        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            best_mask = net.mask.copy()
            best_W = net.W.copy()
        else:
            net.mask = saved_m
            net.W = saved_w

    net.flip_rate = old_flip
    return best_sc, best_acc, best_mask, best_W


# ============================================================
#  CHAIN ENSEMBLE
# ============================================================

SLAVE_POOL = [
    ("Scout",   slave_scout),
    ("Refiner", slave_refiner),
    ("Flipper", slave_flipper),
    ("Rewirer", slave_rewirer),
]


def train_chain(net, targets, V, total_attempts=8000, ticks=8,
                rounds=None, verbose=True):
    """
    Master-slave chain training.
    Each round: 4 slaves get budget = total_attempts / (4 * rounds).
    After each round, master picks best, broadcasts as new jackpot.
    """
    n_slaves = len(SLAVE_POOL)
    if rounds is None:
        rounds = 20  # default: 20 rounds

    budget_per_slave = max(1, total_attempts // (n_slaves * rounds))
    actual_total = budget_per_slave * n_slaves * rounds

    champion_sc, champion_acc = score_batch(net, targets, V, ticks)
    champion_mask = net.mask.copy()
    champion_W = net.W.copy()

    wins_by_slave = {name: 0 for name, _ in SLAVE_POOL}
    round_log = []

    for r in range(rounds):
        round_best_sc = champion_sc
        round_best_acc = champion_acc
        round_best_mask = champion_mask
        round_best_W = champion_W
        round_winner = None

        for name, fn in SLAVE_POOL:
            # Each slave starts from champion (jackpot broadcast)
            net.mask = champion_mask.copy()
            net.W = champion_W.copy()

            sc, ac, m, w = fn(net, targets, V, budget_per_slave, ticks)

            if sc > round_best_sc:
                round_best_sc = sc
                round_best_acc = ac
                round_best_mask = m
                round_best_W = w
                round_winner = name

        # Update champion if improved
        if round_best_sc > champion_sc:
            champion_sc = round_best_sc
            champion_acc = round_best_acc
            champion_mask = round_best_mask
            champion_W = round_best_W
            if round_winner:
                wins_by_slave[round_winner] += 1

        round_log.append(champion_acc)

        if verbose and (r + 1) % 5 == 0:
            att_so_far = (r + 1) * n_slaves * budget_per_slave
            winner_str = round_winner if round_winner else "-"
            print(f"    [round {r+1:3d}] acc={champion_acc*100:5.1f}% "
                  f"att={att_so_far:5d}  winner={winner_str}")

        if champion_acc >= 0.99:
            break

    # Restore champion to net
    net.mask = champion_mask.copy()
    net.W = champion_W.copy()

    return champion_acc, actual_total, wins_by_slave, round_log


def train_single(net, targets, V, total_attempts=8000, ticks=8, verbose=True):
    """Baseline: single searcher with same total attempts."""
    best_sc, best_acc = score_batch(net, targets, V, ticks)
    stale = 0

    for att in range(total_attempts):
        saved_m = net.mask.copy()
        saved_w = net.W.copy()

        # Same mix as default train: structure first, then both
        if att < 2500:
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        sc, ac = score_batch(net, targets, V, ticks)
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            stale = 0
        else:
            net.mask = saved_m
            net.W = saved_w
            stale += 1

        if verbose and (att + 1) % 2000 == 0:
            print(f"    [att {att+1:5d}] acc={best_acc*100:5.1f}%")

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc


# ============================================================
#  RUN
# ============================================================
print("=" * 60)
print("  CHAIN ENSEMBLE vs SINGLE SEARCHER")
print("=" * 60)

CONFIGS = [
    (16,  80,  8000),   # easy
    (32, 128,  8000),   # medium
    (64, 256,  8000),   # hard
]

SEEDS = [42, 77, 123]

for V, N, TOTAL in CONFIGS:
    print(f"\n  --- V={V}, N={N}, {TOTAL} total attempts ---\n")

    single_accs = []
    chain_accs = []
    all_wins = {}

    for seed in SEEDS:
        # A: Single searcher
        np.random.seed(seed); random.seed(seed)
        perm = np.random.permutation(V)

        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_s = SelfWiringGraph(N, V)
        t0 = time.time()
        acc_s = train_single(net_s, perm, V, TOTAL, verbose=False)
        t_s = time.time() - t0

        # B: Chain ensemble
        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_c = SelfWiringGraph(N, V)
        t0 = time.time()
        acc_c, actual, wins, log = train_chain(net_c, perm, V, TOTAL, verbose=False)
        t_c = time.time() - t0

        single_accs.append(acc_s)
        chain_accs.append(acc_c)

        for k, v in wins.items():
            all_wins[k] = all_wins.get(k, 0) + v

        delta = (acc_c - acc_s) * 100
        print(f"    seed={seed:3d}  single={acc_s*100:5.1f}%  chain={acc_c*100:5.1f}%  "
              f"delta={delta:+5.1f}%  ({t_s:.1f}s vs {t_c:.1f}s)")

    avg_s = np.mean(single_accs) * 100
    avg_c = np.mean(chain_accs) * 100
    print(f"    AVG:     single={avg_s:.1f}%  chain={avg_c:.1f}%  delta={avg_c - avg_s:+.1f}%")

    if all_wins:
        print(f"    Wins by slave: ", end="")
        for name, count in sorted(all_wins.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"{name}={count} ", end="")
        print()


# ============================================================
#  VERDICT
# ============================================================
print(f"\n{'='*60}")
print(f"  VERDICT")
print(f"{'='*60}")
print(f"  Chain > Single: specialization + diversity helps search")
print(f"  Chain = Single: overhead cancels out diversity benefit")
print(f"  Chain < Single: splitting budget hurts (not enough per slave)")
print(f"{'='*60}", flush=True)
