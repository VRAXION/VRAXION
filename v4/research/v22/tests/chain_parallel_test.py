"""
VRAXION v22 -- Parallel Chain Ensemble
========================================
Slaves run on separate CPU cores via multiprocessing.
Master collects results, picks winner, broadcasts jackpot.

Comparison:
  A: Sequential chain (1 core, v2 style)
  B: Parallel chain (N cores, same work)
  C: Population variants (parallel)

Measures wall-clock speedup from parallelization.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from v22_best_config import SelfWiringGraph


# ============================================================
#  WORKER FUNCTION (runs in child process)
# ============================================================

def worker_run(slave_type, mask, W, N, V, targets, budget, seed,
               threshold=0.5, leak=0.85):
    """
    Standalone worker: creates net from mask+W, runs strategy, returns best.
    Fully self-contained — no shared state, safe for multiprocessing.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Reconstruct network from mask + W
    net = SelfWiringGraph.__new__(SelfWiringGraph)
    net.N = N
    net.V = V
    net.flip_rate = 0.30
    net.last_acc = 0.0
    net.threshold = threshold
    net.leak = leak
    net.mask = mask.copy()
    net.W = W.copy()
    net.state = np.zeros(N, dtype=np.float32)
    net.charge = np.zeros(N, dtype=np.float32)
    net.addr = np.zeros((N, 4), dtype=np.float32)
    net.target_W = np.zeros((N, 4), dtype=np.float32)
    if N >= 2 * V:
        net.io_mode = 'split'
        net.out_start = N - V
    else:
        net.io_mode = 'shared'
        net.out_start = 0

    # Score function
    def score():
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == targets).mean()
        tp = probs[np.arange(V), targets].mean()
        return 0.5 * acc + 0.5 * tp, acc

    best_sc, best_acc = score()
    best_mask = net.mask.copy()
    best_W = net.W.copy()

    for _ in range(budget):
        sm = net.mask.copy()
        sw = net.W.copy()

        if slave_type == 'scout':
            net.mutate_structure(0.15)
        elif slave_type == 'refiner_soft':
            if random.random() < 0.8:
                net.mutate_weights()
            else:
                net.mutate_structure(0.01)
        elif slave_type == 'refiner':
            if random.random() < 0.7:
                net.mutate_weights()
            else:
                net.mutate_structure(0.02)
        elif slave_type == 'refiner_aggr':
            if random.random() < 0.4:
                net.mutate_weights()
            else:
                net.mutate_structure(0.07)
        elif slave_type == 'rewirer':
            net.flip_rate = 0.0
            net.mutate_structure(0.05)
            net.flip_rate = 0.30
        else:
            net.mutate_structure(0.05)

        sc, ac = score()
        if sc > best_sc:
            best_sc = sc
            best_acc = ac
            best_mask = net.mask.copy()
            best_W = net.W.copy()
        else:
            net.mask = sm
            net.W = sw

    return slave_type, best_sc, best_acc, best_mask, best_W


# ============================================================
#  PARALLEL CHAIN TRAINER
# ============================================================

def train_parallel(net, targets, V, slave_types, budget_per_slave=2000,
                   rounds=4, max_workers=None):
    """Parallel chain: each slave on a separate process."""
    if max_workers is None:
        max_workers = min(len(slave_types), mp.cpu_count())

    # Initial score
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    champion_acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    champion_sc = 0.5 * champion_acc + 0.5 * tp

    champion_mask = net.mask.copy()
    champion_W = net.W.copy()
    wins = {s: 0 for s in slave_types}

    for r in range(rounds):
        # Launch all slaves in parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for i, stype in enumerate(slave_types):
                seed = random.randint(0, 2**31) + i * 1000
                f = pool.submit(
                    worker_run, stype,
                    champion_mask, champion_W,
                    net.N, V, targets, budget_per_slave, seed,
                    net.threshold, net.leak
                )
                futures[f] = stype

            # Collect results
            round_best_sc = champion_sc
            round_best_acc = champion_acc
            round_best_mask = champion_mask
            round_best_W = champion_W
            round_winner = None

            for f in as_completed(futures):
                stype, sc, ac, m, w = f.result()
                if sc > round_best_sc:
                    round_best_sc = sc
                    round_best_acc = ac
                    round_best_mask = m
                    round_best_W = w
                    round_winner = stype

        # Update champion
        if round_best_sc > champion_sc:
            champion_sc = round_best_sc
            champion_acc = round_best_acc
            champion_mask = round_best_mask
            champion_W = round_best_W
            if round_winner:
                wins[round_winner] += 1

        print(f"      round {r+1}: acc={champion_acc*100:5.1f}%  "
              f"winner={round_winner if round_winner else '-'}")

        if champion_acc >= 0.99:
            break

    net.mask = champion_mask.copy()
    net.W = champion_W.copy()
    return champion_acc, wins


def train_sequential(net, targets, V, slave_types, budget_per_slave=2000,
                     rounds=4):
    """Sequential chain (same logic, 1 core) for timing comparison."""
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    champion_acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    champion_sc = 0.5 * champion_acc + 0.5 * tp
    champion_mask = net.mask.copy()
    champion_W = net.W.copy()

    for r in range(rounds):
        round_best_sc = champion_sc
        round_best_acc = champion_acc
        round_best_mask = champion_mask
        round_best_W = champion_W
        round_winner = None

        for stype in slave_types:
            seed = random.randint(0, 2**31)
            _, sc, ac, m, w = worker_run(
                stype, champion_mask, champion_W,
                net.N, V, targets, budget_per_slave, seed,
                net.threshold, net.leak
            )
            if sc > round_best_sc:
                round_best_sc = sc
                round_best_acc = ac
                round_best_mask = m
                round_best_W = w
                round_winner = stype

        if round_best_sc > champion_sc:
            champion_sc = round_best_sc
            champion_acc = round_best_acc
            champion_mask = round_best_mask
            champion_W = round_best_W

        if champion_acc >= 0.99:
            break

    net.mask = champion_mask.copy()
    net.W = champion_W.copy()
    return champion_acc


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    N_CORES = mp.cpu_count()
    print("=" * 65)
    print(f"  PARALLEL CHAIN ENSEMBLE ({N_CORES} CPU cores available)")
    print("=" * 65)

    V = 64; N = 256
    BUDGET = 2000
    ROUNDS = 4
    SEEDS = [42, 77, 123]

    # Population configs to test
    POPS = {
        "A_original":      ['scout', 'refiner', 'refiner_aggr', 'rewirer'],
        "B_refiner_heavy":  ['refiner_soft', 'refiner', 'refiner_aggr', 'rewirer'],
        "C_minimal":        ['refiner', 'rewirer'],
    }

    for pop_name, slaves in POPS.items():
        n_slaves = len(slaves)
        print(f"\n  --- {pop_name} ({n_slaves} slaves) ---")
        print(f"      [{', '.join(slaves)}]")

        for seed in SEEDS:
            np.random.seed(seed); random.seed(seed)
            perm = np.random.permutation(V)

            # Sequential
            np.random.seed(seed + 1000); random.seed(seed + 1000)
            net_seq = SelfWiringGraph(N, V)
            t0 = time.time()
            acc_seq = train_sequential(net_seq, perm, V, slaves, BUDGET, ROUNDS)
            t_seq = time.time() - t0

            # Parallel
            np.random.seed(seed + 1000); random.seed(seed + 1000)
            net_par = SelfWiringGraph(N, V)
            t0 = time.time()
            print(f"\n    seed={seed} parallel:")
            acc_par, wins = train_parallel(net_par, perm, V, slaves, BUDGET, ROUNDS)
            t_par = time.time() - t0

            speedup = t_seq / t_par if t_par > 0 else 0
            print(f"    => seq={acc_seq*100:5.1f}% ({t_seq:.1f}s)  "
                  f"par={acc_par*100:5.1f}% ({t_par:.1f}s)  "
                  f"speedup={speedup:.1f}x")
            if wins:
                top = sorted(wins.items(), key=lambda x: -x[1])
                print(f"       wins: {' '.join(f'{n}={c}' for n, c in top if c > 0)}")

    print(f"\n{'='*65}")
    print(f"  Expected: ~{min(4, N_CORES)}x wall-clock speedup")
    print(f"  Same accuracy (different RNG per process)")
    print(f"{'='*65}", flush=True)
