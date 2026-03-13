"""
Capacitor Hyperparameter Sweep — Sequential, One-at-a-Time
===========================================================
Each sweep finds the best value, which becomes the new default
for all subsequent sweeps.

Order (most sensitive first):
  1. threshold  [0.05 .. 0.7]
  2. leak       [0.50 .. 0.99]
  3. acc_rate   [0.1 .. 1.0]
  4. self_fb    [0.0 .. 0.2]
  5. charge precision (int8 vs float16 vs float32) with new params

Final validation: all winners on 16/32/64-class (8K + 16K).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Quantization (reused from charge_precision_test)
# ============================================================

def quantize_noop(charge, lo, hi):
    return np.clip(charge, lo, hi)

def quantize_float16(charge, lo, hi):
    return np.clip(charge, lo, hi).astype(np.float16).astype(np.float32)

def quantize_int8(charge, lo, hi):
    charge = np.clip(charge, lo, hi)
    scale = (hi - lo) / 255.0
    return (np.round((charge - lo) / scale) * scale + lo).astype(np.float32)

QUANTIZERS = {
    'float32': quantize_noop,
    'float16': quantize_float16,
    'int8': quantize_int8,
}


# ============================================================
#  Training with configurable capacitor params
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=8000,
                 threshold=0.5, leak=0.85, acc_rate=0.3, self_fb=0.1,
                 charge_mode='float32', ticks=8):
    """Train with configurable capacitor hyperparams."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V, threshold=threshold, leak=leak)
    quantize = QUANTIZERS.get(charge_mode, quantize_noop)

    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    clamp_lo = -threshold * 2
    clamp_hi = threshold * 2

    def forward_custom(world):
        act = net.state.copy()
        Weff = net.W * net.mask

        for t in range(ticks):
            if t == 0:
                act[:V] = world

            raw = act @ Weff + act * self_fb

            net.charge += raw * acc_rate
            net.charge *= leak

            net.charge = quantize(net.charge, clamp_lo, clamp_hi)

            act = np.maximum(net.charge - threshold, 0.0)

        net.state = act.copy()
        return net.charge[:V]

    def evaluate():
        net.reset()
        correct = 0
        total_score = 0.0

        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = forward_custom(world)
                probs = softmax(logits[:V])

                if p == 1:
                    acc_i = 1.0 if np.argmax(probs) == targets[i] else 0.0
                    tp = float(probs[targets[i]])
                    total_score += 0.5 * acc_i + 0.5 * tp
                    if acc_i > 0:
                        correct += 1

        acc = correct / len(inputs)
        score = total_score / len(inputs)
        net.last_acc = acc
        return score, acc

    score, acc = evaluate()
    best_score = score
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score, new_acc = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if best_acc >= 0.99 or stale >= 3500:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    return {
        'label': label,
        'seed': seed,
        'n_classes': n_classes,
        'acc': best_acc,
        'score': best_score,
        'kept': kept,
        'accept_rate': accept_rate,
        'time': elapsed,
        'conns': net.count_connections(),
    }


def worker(args):
    return train_config(**args)


def aggregate_results(results):
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    summary = {}
    for label, runs in groups.items():
        accs = [r['acc'] for r in runs]
        rates = [r['accept_rate'] for r in runs]
        times = [r['time'] for r in runs]
        summary[label] = {
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'accept_mean': np.mean(rates),
            'time_mean': np.mean(times),
            'n': len(runs),
        }
    return summary


def print_sweep(param_name, summary, default_label=None):
    default_acc = summary[default_label]['acc_mean'] if default_label and default_label in summary else None

    print(f"\n  {'='*80}")
    print(f"  {'Param':<10s} {'Value':<12s} {'Mean_Acc':>10s} {'Std':>7s} {'Accept%':>9s} {'Time':>6s}", end="")
    if default_acc is not None:
        print(f" {'vs_Default':>11s}", end="")
    print()
    print(f"  {'-'*80}")

    for label in sorted(summary, key=lambda k: summary[k]['acc_mean'], reverse=True):
        s = summary[label]
        # Extract value from label
        val = label.split('=')[1] if '=' in label else label
        delta_str = ""
        if default_acc is not None:
            delta = (s['acc_mean'] - default_acc) * 100
            delta_str = f"  {delta:+6.1f}%"
        print(f"  {param_name:<10s} {val:<12s} {s['acc_mean']*100:8.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['accept_mean']:8.3f}% {s['time_mean']:5.0f}s{delta_str}")

    # Find best
    best_label = max(summary, key=lambda k: summary[k]['acc_mean'])
    best_val = best_label.split('=')[1] if '=' in best_label else best_label
    print(f"\n  >>> WINNER: {param_name}={best_val} ({summary[best_label]['acc_mean']*100:.1f}%)")
    sys.stdout.flush()
    return best_label


def run_sweep(param_name, values, fixed_params, n_classes=16, n_neurons=256,
              max_attempts=8000, n_workers=24):
    """Run a parameter sweep. Returns best value."""

    configs = []
    for val in values:
        params = dict(fixed_params)
        params[param_name] = val
        label = f"{param_name}={val}"
        for seed in SEEDS:
            configs.append(dict(
                label=label, seed=seed,
                n_classes=n_classes, n_neurons=n_neurons,
                max_attempts=max_attempts, **params))

    print(f"\n  Running {len(configs)} jobs ({param_name} sweep, {n_classes}-class, {max_attempts} att)...",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    summary = aggregate_results(results)

    # Find default label
    default_val = fixed_params.get(param_name, None)
    default_label = f"{param_name}={default_val}" if default_val is not None else None

    best_label = print_sweep(param_name, summary, default_label)
    best_val = float(best_label.split('=')[1])
    return best_val


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Task: 16-class, 256 neurons, 8K attempts per job")
    print(f"  Scoring: combined (0.5*acc + 0.5*target_prob)")

    # Starting defaults
    defaults = {
        'threshold': 0.5,
        'leak': 0.85,
        'acc_rate': 0.3,
        'self_fb': 0.1,
        'charge_mode': 'float32',
    }

    # =========================================================
    # SWEEP 1: THRESHOLD
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  SWEEP 1: THRESHOLD (most sensitive)")
    print(f"  Fixed: leak={defaults['leak']}, acc={defaults['acc_rate']}, sfb={defaults['self_fb']}")
    print(f"{'#'*80}", flush=True)

    best_threshold = run_sweep(
        'threshold',
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7],
        defaults, n_workers=n_workers)

    defaults['threshold'] = best_threshold
    print(f"\n  Updated defaults: threshold={best_threshold}")

    # =========================================================
    # SWEEP 2: LEAK
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  SWEEP 2: LEAK")
    print(f"  Fixed: threshold={defaults['threshold']}, acc={defaults['acc_rate']}, sfb={defaults['self_fb']}")
    print(f"{'#'*80}", flush=True)

    best_leak = run_sweep(
        'leak',
        [0.50, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99],
        defaults, n_workers=n_workers)

    defaults['leak'] = best_leak
    print(f"\n  Updated defaults: threshold={defaults['threshold']}, leak={best_leak}")

    # =========================================================
    # SWEEP 3: ACC_RATE
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  SWEEP 3: ACC_RATE (charge accumulation rate)")
    print(f"  Fixed: threshold={defaults['threshold']}, leak={defaults['leak']}, sfb={defaults['self_fb']}")
    print(f"{'#'*80}", flush=True)

    best_acc_rate = run_sweep(
        'acc_rate',
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        defaults, n_workers=n_workers)

    defaults['acc_rate'] = best_acc_rate
    print(f"\n  Updated defaults: threshold={defaults['threshold']}, leak={defaults['leak']}, acc={best_acc_rate}")

    # =========================================================
    # SWEEP 4: SELF_FEEDBACK
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  SWEEP 4: SELF_FEEDBACK")
    print(f"  Fixed: threshold={defaults['threshold']}, leak={defaults['leak']}, acc={defaults['acc_rate']}")
    print(f"{'#'*80}", flush=True)

    best_sfb = run_sweep(
        'self_fb',
        [0.0, 0.02, 0.05, 0.1, 0.15, 0.2],
        defaults, n_workers=n_workers)

    defaults['self_fb'] = best_sfb
    print(f"\n  Updated defaults: threshold={defaults['threshold']}, leak={defaults['leak']}, "
          f"acc={defaults['acc_rate']}, sfb={best_sfb}")

    # =========================================================
    # SWEEP 5: CHARGE PRECISION (with optimized params)
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  SWEEP 5: CHARGE PRECISION (with optimized params)")
    print(f"  Params: threshold={defaults['threshold']}, leak={defaults['leak']}, "
          f"acc={defaults['acc_rate']}, sfb={defaults['self_fb']}")
    print(f"{'#'*80}", flush=True)

    precision_configs = []
    for mode in ['float32', 'float16', 'int8']:
        label = f"charge_mode={mode}"
        for seed in SEEDS:
            precision_configs.append(dict(
                label=label, seed=seed,
                n_classes=16, n_neurons=256, max_attempts=8000,
                threshold=defaults['threshold'], leak=defaults['leak'],
                acc_rate=defaults['acc_rate'], self_fb=defaults['self_fb'],
                charge_mode=mode))

    print(f"\n  Running {len(precision_configs)} jobs (precision sweep)...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        prec_results = list(pool.map(worker, precision_configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    prec_summary = aggregate_results(prec_results)
    best_prec_label = print_sweep('precision', prec_summary, 'charge_mode=float32')
    best_mode = best_prec_label.split('=')[1]
    defaults['charge_mode'] = best_mode

    # =========================================================
    # FINAL VALIDATION
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  FINAL VALIDATION — Optimized vs Original Defaults")
    print(f"  Optimized: threshold={defaults['threshold']}, leak={defaults['leak']}, "
          f"acc={defaults['acc_rate']}, sfb={defaults['self_fb']}, charge={defaults['charge_mode']}")
    print(f"  Original:  threshold=0.5, leak=0.85, acc=0.3, sfb=0.1, charge=float32")
    print(f"{'#'*80}", flush=True)

    val_configs = []
    for nc, max_att in [(16, 8000), (32, 8000), (64, 8000), (64, 16000)]:
        att_label = f"{max_att//1000}K"
        # Optimized
        for seed in SEEDS:
            val_configs.append(dict(
                label=f'{nc}c OPTIMIZED {att_label}', seed=seed,
                n_classes=nc, n_neurons=256, max_attempts=max_att,
                threshold=defaults['threshold'], leak=defaults['leak'],
                acc_rate=defaults['acc_rate'], self_fb=defaults['self_fb'],
                charge_mode=defaults['charge_mode']))
        # Original defaults
        for seed in SEEDS:
            val_configs.append(dict(
                label=f'{nc}c ORIGINAL {att_label}', seed=seed,
                n_classes=nc, n_neurons=256, max_attempts=max_att,
                threshold=0.5, leak=0.85, acc_rate=0.3, self_fb=0.1,
                charge_mode='float32'))

    print(f"\n  Running {len(val_configs)} validation jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        val_results = list(pool.map(worker, val_configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    val_summary = aggregate_results(val_results)

    print(f"\n  {'='*80}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*80}")
    print(f"  {'Config':<30s} {'Acc':>8s} {'Std':>7s} {'Accept%':>9s} {'Time':>6s}")
    print(f"  {'-'*70}")
    for label in sorted(val_summary, key=lambda k: (
            int(k.split('c ')[0]), 'ORIGINAL' in k)):
        s = val_summary[label]
        print(f"  {label:<30s} {s['acc_mean']*100:6.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['accept_mean']:8.3f}% {s['time_mean']:5.0f}s")
    sys.stdout.flush()

    # Summary
    print(f"\n  {'='*80}")
    print(f"  OPTIMIZED PARAMETERS")
    print(f"  {'='*80}")
    print(f"  threshold:    {defaults['threshold']}")
    print(f"  leak:         {defaults['leak']}")
    print(f"  acc_rate:     {defaults['acc_rate']}")
    print(f"  self_fb:      {defaults['self_fb']}")
    print(f"  charge_mode:  {defaults['charge_mode']}")
    print(f"  {'='*80}")
    print(f"  DONE")
    print(f"  {'='*80}", flush=True)
