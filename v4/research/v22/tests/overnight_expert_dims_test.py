"""
VRAXION v22 -- Overnight Expert Differentiation Mega-Test
==========================================================
6 dimensions of expert specialization, 17 configs, 3 seeds each.
Parallel execution on 20 CPU cores.

DIM 1: Tick variation (4/8/16)
DIM 2: Region locality (first_half/second_half/input_zone/output_zone/random_blob)
DIM 3: Odd/Even interleave
DIM 4: Learned affinity channel (fixed/learned)
DIM 5: Eval strategy (full/quarter/class_weighted)
DIM 6: Ensemble composition (tick_mixed/region_split) -- uses parallel slaves
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from v22_best_config import SelfWiringGraph
from v22_log import live_log, log_msg

# =============================================================
# Global test params
# =============================================================
V = 64
N = 256
BUDGET = 2000
ROUNDS = 4
SEEDS = [42, 77, 123]
MAX_WORKERS = 20

# =============================================================
# Scoring
# =============================================================
def score_batch(net, targets, V, ticks=8):
    logits = net.forward_batch(ticks=ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    return 0.5 * acc + 0.5 * tp, acc

def score_quarter(net, targets, V, subset_idx, ticks=8):
    """Score only a subset of inputs (faster but noisier)."""
    logits_full = net.forward_batch(ticks=ticks)
    logits = logits_full[subset_idx]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    sub_targets = targets[subset_idx]
    preds = np.argmax(probs, axis=1)
    acc = (preds == sub_targets).mean()
    tp = probs[np.arange(len(subset_idx)), sub_targets].mean()
    return 0.5 * acc + 0.5 * tp, acc

def score_class_weighted(net, targets, V, class_weights, ticks=8):
    """Weighted scoring -- harder classes count more."""
    logits = net.forward_batch(ticks=ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    correct = (preds == targets).astype(float)
    tp = probs[np.arange(V), targets]
    w = class_weights / class_weights.sum()
    acc = (correct * w).sum()
    tp_w = (tp * w).sum()
    return 0.5 * acc + 0.5 * tp_w, correct.mean()

# =============================================================
# Region helpers
# =============================================================
def get_allowed_neurons(region, N, V, out_start, seed=42):
    """Return set of neuron indices allowed for mutation."""
    if region is None or region == 'unrestricted':
        return None  # no restriction
    elif region == 'first_half':
        return set(range(N // 2))
    elif region == 'second_half':
        return set(range(N // 2, N))
    elif region == 'input_zone':
        # input neurons (0..V) + 16 neighbors
        return set(range(min(V + 16, N)))
    elif region == 'output_zone':
        # output neurons (out_start..N) + 16 before
        return set(range(max(0, out_start - 16), N))
    elif region == 'middle':
        # internal neurons only (not input, not output)
        return set(range(V, out_start)) if out_start > V else set(range(V, N))
    elif region == 'random_blob':
        rng = np.random.RandomState(seed + 999)
        blob_size = int(N * 0.3)
        return set(rng.choice(N, blob_size, replace=False).tolist())
    elif region == 'odd':
        return set(range(1, N, 2))
    elif region == 'even':
        return set(range(0, N, 2))
    else:
        return None

def constrained_mutate_structure(net, rate, allowed):
    """Mutate structure but only touch neurons in allowed set."""
    if allowed is None:
        net.mutate_structure(rate)
        return

    # Save full mask
    old_mask = net.mask.copy()
    old_W = net.W.copy()

    # Do normal mutation
    net.mutate_structure(rate)

    # Revert changes outside allowed region
    for i in range(net.N):
        if i not in allowed:
            net.mask[i, :] = old_mask[i, :]
            net.mask[:, i] = old_mask[:, i]
            net.W[i, :] = old_W[i, :]
            net.W[:, i] = old_W[:, i]

def constrained_mutate_weights(net, allowed):
    """Mutate weights but only for neurons in allowed set."""
    if allowed is None:
        net.mutate_weights()
        return

    old_W = net.W.copy()
    net.mutate_weights()

    for i in range(net.N):
        if i not in allowed:
            net.W[i, :] = old_W[i, :]
            net.W[:, i] = old_W[:, i]

# =============================================================
# Affinity channel
# =============================================================
def affinity_mutate_structure(net, rate, expert_ch, neuron_chs):
    """Mutation strength modulated by channel affinity."""
    # Compute per-neuron weight: close channel = strong mutation
    weights = 1.0 - np.abs(neuron_chs - expert_ch)
    weights = np.clip(weights, 0.1, 1.0)  # floor at 10%

    # Scale rate by mean affinity (global effect)
    effective_rate = rate * weights.mean()
    net.mutate_structure(effective_rate)

def update_neuron_channels(neuron_chs, expert_ch, net, old_mask, learning_rate=0.05):
    """Move neuron channels toward expert if mutation was kept."""
    changed = np.any(net.mask != old_mask, axis=1) | np.any(net.mask != old_mask, axis=0)
    # Neurons that were changed move toward expert channel
    neuron_chs[changed] += learning_rate * (expert_ch - neuron_chs[changed])
    return neuron_chs

# =============================================================
# Single config runner
# =============================================================
def run_config(config_name, config, seed, V=64, N=256, budget=2000, rounds=4, log_q=None):
    """Run one config with one seed. Returns result dict."""
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N, V)
    targets = np.random.permutation(V)
    out_start = net.out_start

    ticks = config.get('ticks', 8)
    region = config.get('region', None)
    eval_mode = config.get('eval_mode', 'full')
    affinity = config.get('affinity', None)
    mut_style = config.get('mut_style', 'refiner_aggr')

    allowed = get_allowed_neurons(region, N, V, out_start, seed)

    # Affinity channels
    expert_ch = config.get('expert_channel', 0.5)
    neuron_chs = None
    if affinity:
        neuron_chs = np.random.rand(N).astype(np.float32)

    # Quarter eval subset (fixed for this run)
    quarter_idx = np.random.choice(V, V // 4, replace=False) if eval_mode == 'quarter' else None

    # Class weights for weighted eval
    class_weights = np.ones(V, dtype=np.float32)

    # Score function
    def score():
        if eval_mode == 'quarter':
            return score_quarter(net, targets, V, quarter_idx, ticks=ticks)
        elif eval_mode == 'class_weighted':
            return score_class_weighted(net, targets, V, class_weights, ticks=ticks)
        else:
            return score_batch(net, targets, V, ticks=ticks)

    # Full eval (always full, for final measurement)
    def score_full():
        return score_batch(net, targets, V, ticks=ticks)

    t0 = time.time()
    best_sc, best_acc = score()
    best_mask = net.mask.copy()
    best_W = net.W.copy()
    total_kept = 0

    for rnd in range(rounds):
        for att in range(budget):
            sm = net.mask.copy()
            sw = net.W.copy()

            # Mutation
            if affinity and neuron_chs is not None:
                affinity_mutate_structure(net, 0.07, expert_ch, neuron_chs)
                if random.random() < 0.4:
                    net.mutate_weights()
            elif mut_style == 'refiner_aggr':
                if random.random() < 0.4:
                    constrained_mutate_weights(net, allowed)
                else:
                    constrained_mutate_structure(net, 0.07, allowed)
            elif mut_style == 'refiner':
                if random.random() < 0.7:
                    constrained_mutate_weights(net, allowed)
                else:
                    constrained_mutate_structure(net, 0.02, allowed)
            elif mut_style == 'scout':
                constrained_mutate_structure(net, 0.15, allowed)
            else:
                constrained_mutate_structure(net, 0.05, allowed)

            sc, ac = score()

            if sc > best_sc:
                best_sc = sc
                best_acc = ac
                best_mask = net.mask.copy()
                best_W = net.W.copy()
                total_kept += 1

                # Update class weights (harder classes get higher weight)
                if eval_mode == 'class_weighted':
                    logits = net.forward_batch(ticks=ticks)
                    preds = np.argmax(logits, axis=1)
                    for c in range(V):
                        if preds[c] == targets[c]:
                            class_weights[c] *= 0.95  # easier -> lower weight
                        else:
                            class_weights[c] *= 1.1   # harder -> higher weight
                    class_weights = np.clip(class_weights, 0.1, 10.0)

                # Update affinity channels (learned mode)
                if affinity == 'learned' and neuron_chs is not None:
                    neuron_chs = update_neuron_channels(neuron_chs, expert_ch, net, sm)
            else:
                net.mask = sm
                net.W = sw

        # Restore best for next round
        net.mask = best_mask.copy()
        net.W = best_W.copy()

    elapsed = time.time() - t0

    # Final eval always with full batch
    net.mask = best_mask.copy()
    net.W = best_W.copy()
    _, final_acc = score_full()

    log_msg(log_q, f"{config_name:25s} seed={seed} acc={final_acc*100:5.1f}% time={elapsed:.0f}s")
    return {
        'config': config_name,
        'seed': seed,
        'acc': final_acc,
        'score': best_sc,
        'kept': total_kept,
        'time': elapsed,
        'ticks': ticks,
        'region': region or 'unrestricted',
        'eval_mode': eval_mode,
        'affinity': affinity or 'none',
    }

# =============================================================
# Ensemble runner (DIM 6)
# =============================================================
def run_ensemble(ensemble_name, slave_configs, seed, V=64, N=256,
                 budget_per_slave=2000, rounds=4, log_q=None):
    """Run a multi-slave ensemble with jackpot broadcast."""
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N, V)
    targets = np.random.permutation(V)

    champion_mask = net.mask.copy()
    champion_W = net.W.copy()
    champion_sc, champion_acc = score_batch(net, targets, V, ticks=8)
    wins = {}

    t0 = time.time()

    for rnd in range(rounds):
        round_best_sc = champion_sc
        round_best_acc = champion_acc
        round_best_mask = champion_mask.copy()
        round_best_W = champion_W.copy()
        round_winner = None

        for sname, sconfig in slave_configs.items():
            # Each slave starts from champion
            np.random.seed(seed + rnd * 100 + hash(sname) % 1000)
            random.seed(seed + rnd * 100 + hash(sname) % 1000)

            net.mask = champion_mask.copy()
            net.W = champion_W.copy()

            ticks = sconfig.get('ticks', 8)
            region = sconfig.get('region', None)
            allowed = get_allowed_neurons(region, N, V, net.out_start, seed)

            best_sc = champion_sc
            best_mask = net.mask.copy()
            best_W = net.W.copy()

            for att in range(budget_per_slave):
                sm = net.mask.copy()
                sw = net.W.copy()

                if random.random() < 0.4:
                    constrained_mutate_weights(net, allowed)
                else:
                    constrained_mutate_structure(net, 0.07, allowed)

                sc, ac = score_batch(net, targets, V, ticks=ticks)
                if sc > best_sc:
                    best_sc = sc
                    best_mask = net.mask.copy()
                    best_W = net.W.copy()
                else:
                    net.mask = sm
                    net.W = sw

            if best_sc > round_best_sc:
                round_best_sc = best_sc
                round_best_mask = best_mask.copy()
                round_best_W = best_W.copy()
                # Get actual acc
                net.mask = best_mask.copy()
                net.W = best_W.copy()
                _, round_best_acc = score_batch(net, targets, V, ticks=8)
                round_winner = sname

        if round_winner:
            champion_sc = round_best_sc
            champion_acc = round_best_acc
            champion_mask = round_best_mask.copy()
            champion_W = round_best_W.copy()
            wins[round_winner] = wins.get(round_winner, 0) + 1

    elapsed = time.time() - t0

    log_msg(log_q, f"{ensemble_name:25s} seed={seed} acc={champion_acc*100:5.1f}% "
            f"wins={wins} time={elapsed:.0f}s")
    return {
        'config': ensemble_name,
        'seed': seed,
        'acc': champion_acc,
        'score': champion_sc,
        'kept': sum(wins.values()),
        'time': elapsed,
        'ticks': 'mixed',
        'region': 'ensemble',
        'eval_mode': 'full',
        'affinity': 'none',
        'wins': wins,
    }

# =============================================================
# Config definitions
# =============================================================
CONFIGS = {
    # DIM 1: Tick variation
    'tick_4':   {'ticks': 4},
    'tick_8':   {'ticks': 8},
    'tick_16':  {'ticks': 16},

    # DIM 2: Region locality
    'region_first_half':   {'region': 'first_half'},
    'region_second_half':  {'region': 'second_half'},
    'region_input_zone':   {'region': 'input_zone'},
    'region_output_zone':  {'region': 'output_zone'},
    'region_random_blob':  {'region': 'random_blob'},

    # DIM 3: Odd/Even
    'odd_neurons':  {'region': 'odd'},
    'even_neurons': {'region': 'even'},

    # DIM 4: Affinity
    'affinity_fixed':   {'affinity': 'fixed', 'expert_channel': 0.5},
    'affinity_learned': {'affinity': 'learned', 'expert_channel': 0.5},

    # DIM 5: Eval strategy
    'eval_full':           {'eval_mode': 'full'},
    'eval_quarter':        {'eval_mode': 'quarter'},
    'eval_class_weighted': {'eval_mode': 'class_weighted'},
}

ENSEMBLE_CONFIGS = {
    # DIM 6: Ensemble composition
    'ens_tick_mixed': {
        'slave_4tick':  {'ticks': 4},
        'slave_8tick':  {'ticks': 8},
        'slave_16tick': {'ticks': 16},
        'slave_8tick_aggr': {'ticks': 8},
    },
    'ens_region_split': {
        'slave_input':  {'region': 'input_zone'},
        'slave_output': {'region': 'output_zone'},
        'slave_middle': {'region': 'middle'},
        'slave_free':   {'region': None},
    },
}

# =============================================================
# Main
# =============================================================
def main():
    print("=" * 70)
    print("  OVERNIGHT EXPERT DIFFERENTIATION MEGA-TEST")
    print("=" * 70)
    print(f"  V={V}, N={N}, budget={BUDGET}, rounds={ROUNDS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Configs: {len(CONFIGS)} single + {len(ENSEMBLE_CONFIGS)} ensemble")
    print(f"  Total jobs: {len(CONFIGS)*len(SEEDS) + len(ENSEMBLE_CONFIGS)*len(SEEDS)}")
    print(f"  Max workers: {MAX_WORKERS}")
    print(f"  {'=' * 70}")
    sys.stdout.flush()

    all_results = []
    t_start = time.time()

    with live_log('overnight_expert_dims') as (log_q, log_path):
        log_msg(log_q, f"Starting test run")

        # Phase 1: Single configs (parallel)
        print(f"\n  PHASE 1: Single config tests ({len(CONFIGS)*len(SEEDS)} jobs)")
        print(f"  {'-' * 50}")
        sys.stdout.flush()

        jobs = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for cname, cfg in CONFIGS.items():
                for seed in SEEDS:
                    fut = pool.submit(run_config, cname, cfg, seed, V, N, BUDGET, ROUNDS,
                                      log_q=log_q)
                    jobs.append((cname, seed, fut))

            for cname, seed, fut in jobs:
                try:
                    res = fut.result(timeout=600)
                    all_results.append(res)
                except Exception as ex:
                    log_msg(log_q, f"[ERR] {cname} seed={seed}: {ex}")

        # Phase 2: Ensemble configs (sequential)
        print(f"\n  PHASE 2: Ensemble tests ({len(ENSEMBLE_CONFIGS)*len(SEEDS)} jobs)")
        print(f"  {'-' * 50}")
        sys.stdout.flush()

        for ename, slave_cfgs in ENSEMBLE_CONFIGS.items():
            for seed in SEEDS:
                try:
                    res = run_ensemble(ename, slave_cfgs, seed, V, N, BUDGET, ROUNDS,
                                       log_q=log_q)
                    all_results.append(res)
                except Exception as ex:
                    log_msg(log_q, f"[ERR] {ename} seed={seed}: {ex}")

        log_msg(log_q, f"All jobs complete ({len(all_results)} results)")

    total_time = time.time() - t_start

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'overnight_expert_dims_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'config', 'seed', 'acc', 'score', 'kept', 'time',
            'ticks', 'region', 'eval_mode', 'affinity'])
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\n  CSV saved: {csv_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY (sorted by mean accuracy)")
    print(f"{'=' * 70}\n")

    # Group by config
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r['config']].append(r)

    ranked = []
    for cname, runs in groups.items():
        accs = [r['acc'] for r in runs]
        times = [r['time'] for r in runs]
        ranked.append({
            'config': cname,
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'min_acc': np.min(accs),
            'max_acc': np.max(accs),
            'mean_time': np.mean(times),
        })

    ranked.sort(key=lambda x: -x['mean_acc'])

    print(f"  {'Config':25s} {'Mean':>6s} {'Std':>5s} {'Min':>6s} {'Max':>6s} {'Time':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")
    for r in ranked:
        print(f"  {r['config']:25s} {r['mean_acc']*100:5.1f}% {r['std_acc']*100:4.1f}% "
              f"{r['min_acc']*100:5.1f}% {r['max_acc']*100:5.1f}% {r['mean_time']:5.0f}s")

    # Dimension winners
    print(f"\n  DIMENSION WINNERS:")
    print(f"  {'-' * 50}")
    dims = {
        'DIM1 Ticks': ['tick_4', 'tick_8', 'tick_16'],
        'DIM2 Region': ['region_first_half', 'region_second_half',
                        'region_input_zone', 'region_output_zone', 'region_random_blob'],
        'DIM3 Odd/Even': ['odd_neurons', 'even_neurons'],
        'DIM4 Affinity': ['affinity_fixed', 'affinity_learned'],
        'DIM5 Eval': ['eval_full', 'eval_quarter', 'eval_class_weighted'],
        'DIM6 Ensemble': ['ens_tick_mixed', 'ens_region_split'],
    }

    dim_winners = {}
    for dim_name, members in dims.items():
        dim_ranked = [r for r in ranked if r['config'] in members]
        if dim_ranked:
            winner = dim_ranked[0]
            dim_winners[dim_name] = winner['config']
            print(f"  {dim_name:20s} -> {winner['config']:25s} ({winner['mean_acc']*100:.1f}%)")

    print(f"\n  Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
