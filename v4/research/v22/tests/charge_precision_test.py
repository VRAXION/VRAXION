"""
Charge Precision Benchmark
===========================
The mask is 2-bit (ternary), weights are 1-bit (binary).
But the charge is float32 — 32 bits per neuron per tick.

Question: how much precision does the charge ACTUALLY need?
If int8 works, the entire model fits in L1 cache on microcontrollers.

Variants:
  1. float32 — current baseline (32 bit)
  2. float16 — half precision (16 bit)
  3. int8 — 256 levels mapped to [-1, 1] (8 bit)
  4. int4 — 16 levels mapped to [-1, 1] (4 bit)
  5. binary — 0 or 1 only (1 bit)
  6. ternary — -1, 0, +1 (2 bit)

The quantization happens AFTER each capacitor update step:
  charge += raw * 0.3
  charge *= leak
  charge = quantize(charge)  # <-- here

All tests parallel, 5 seeds, combined scoring.
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
#  Quantization functions
# ============================================================

def quantize_noop(charge, clamp_lo, clamp_hi):
    """Float32 — no quantization (baseline)."""
    return np.clip(charge, clamp_lo, clamp_hi)


def quantize_float16(charge, clamp_lo, clamp_hi):
    """Float16 — half precision."""
    charge = np.clip(charge, clamp_lo, clamp_hi)
    return charge.astype(np.float16).astype(np.float32)


def quantize_int8(charge, clamp_lo, clamp_hi):
    """Int8 — 256 levels in [clamp_lo, clamp_hi]."""
    charge = np.clip(charge, clamp_lo, clamp_hi)
    # Map to [0, 255], round, map back
    scale = (clamp_hi - clamp_lo) / 255.0
    quantized = np.round((charge - clamp_lo) / scale) * scale + clamp_lo
    return quantized.astype(np.float32)


def quantize_int4(charge, clamp_lo, clamp_hi):
    """Int4 — 16 levels in [clamp_lo, clamp_hi]."""
    charge = np.clip(charge, clamp_lo, clamp_hi)
    scale = (clamp_hi - clamp_lo) / 15.0
    quantized = np.round((charge - clamp_lo) / scale) * scale + clamp_lo
    return quantized.astype(np.float32)


def quantize_binary(charge, clamp_lo, clamp_hi):
    """Binary — 0 or 1 (fire or not)."""
    return np.where(charge > 0, np.float32(1.0), np.float32(0.0))


def quantize_ternary(charge, clamp_lo, clamp_hi):
    """Ternary — -1, 0, or +1."""
    out = np.zeros_like(charge)
    threshold = (clamp_hi - clamp_lo) * 0.25  # fire at 25% of range
    out[charge > threshold] = 1.0
    out[charge < -threshold] = -1.0
    return out


QUANTIZERS = {
    'float32': quantize_noop,
    'float16': quantize_float16,
    'int8': quantize_int8,
    'int4': quantize_int4,
    'binary': quantize_binary,
    'ternary': quantize_ternary,
}

BITS_PER_NEURON = {
    'float32': 32,
    'float16': 16,
    'int8': 8,
    'int4': 4,
    'binary': 1,
    'ternary': 2,
}


# ============================================================
#  Training with quantized charge
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=4000,
                 charge_mode='float32', ticks=8):
    """Train with quantized charge. Returns result dict."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    quantize = QUANTIZERS[charge_mode]

    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    clamp_lo = -net.threshold * 2
    clamp_hi = net.threshold * 2

    def forward_quantized(world):
        """Forward pass with charge quantization after each tick."""
        act = net.state.copy()
        Weff = net.W * net.mask

        for t in range(ticks):
            if t == 0:
                act[:V] = world

            raw = act @ Weff + act * 0.1

            # Capacitor dynamics
            net.charge += raw * 0.3
            net.charge *= net.leak

            # QUANTIZE charge here
            net.charge = quantize(net.charge, clamp_lo, clamp_hi)

            # Threshold output
            act = np.maximum(net.charge - net.threshold, 0.0)

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
                logits = forward_quantized(world)
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

    bits = BITS_PER_NEURON[charge_mode]
    # Total model size: 3 bits per connection (mask+weight) + charge bits per neuron
    conn_bits = net.count_connections() * 3
    charge_bits = n_neurons * bits
    total_bytes = (conn_bits + charge_bits) // 8

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
        'charge_bits': bits,
        'total_bytes': total_bytes,
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
        bytes_list = [r['total_bytes'] for r in runs]
        summary[label] = {
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'accept_mean': np.mean(rates),
            'time_mean': np.mean(times),
            'bytes_mean': np.mean(bytes_list),
            'n': len(runs),
        }
    return summary


def print_summary(title, summary):
    print(f"\n  {'='*75}")
    print(f"  {title}")
    print(f"  {'='*75}")
    print(f"  {'Config':<30s} {'Acc':>7s} {'Std':>6s} {'Accept%':>8s} {'Size':>8s} {'Time':>6s}")
    print(f"  {'-'*70}")
    for label in sorted(summary, key=lambda k: summary[k]['acc_mean'], reverse=True):
        s = summary[label]
        print(f"  {label:<30s} {s['acc_mean']*100:6.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['accept_mean']:7.3f}% {s['bytes_mean']:7.0f}B {s['time_mean']:5.0f}s")
    sys.stdout.flush()


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 4
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Scoring: combined (0.5*acc + 0.5*target_prob)")

    # === PHASE 1: Full sweep on 16-class ===
    print(f"\n{'#'*75}")
    print(f"  PHASE 1: Charge Precision Sweep -- 16-class, 256 neurons, 4K attempts")
    print(f"{'#'*75}", flush=True)

    configs = []
    for mode in QUANTIZERS:
        label = f'16c {mode} ({BITS_PER_NEURON[mode]}b)'
        for seed in SEEDS:
            configs.append(dict(
                label=label, seed=seed,
                n_classes=16, n_neurons=256,
                max_attempts=4000, charge_mode=mode, ticks=8))

    print(f"  Running {len(configs)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p1 = list(pool.map(worker, configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    summary_p1 = aggregate_results(results_p1)
    print_summary("16-CLASS CHARGE PRECISION", summary_p1)

    # Find best and classify
    best_label = max(summary_p1, key=lambda k: summary_p1[k]['acc_mean'])
    baseline_acc = summary_p1[[k for k in summary_p1 if 'float32' in k][0]]['acc_mean']
    print(f"\n  Best: {best_label} ({summary_p1[best_label]['acc_mean']*100:.1f}%)")
    print(f"  Baseline (float32): {baseline_acc*100:.1f}%")

    # Show which modes are within 5% of baseline
    print(f"\n  Modes within 5% of float32:")
    for label in sorted(summary_p1, key=lambda k: summary_p1[k]['acc_mean'], reverse=True):
        s = summary_p1[label]
        delta = (s['acc_mean'] - baseline_acc) * 100
        if abs(delta) <= 5.0:
            print(f"    {label}: {s['acc_mean']*100:.1f}% (delta={delta:+.1f}%, size={s['bytes_mean']:.0f}B)")
    sys.stdout.flush()

    # === PHASE 2: Top modes on 32 and 64-class ===
    # Take modes within 5% of baseline + always include float32
    viable = [label for label in summary_p1
              if abs(summary_p1[label]['acc_mean'] - baseline_acc) <= 0.05
              or 'float32' in label]

    print(f"\n{'#'*75}")
    print(f"  PHASE 2: Viable modes on 32-class and 64-class")
    print(f"  Testing: {[v.replace('16c ', '') for v in viable]}")
    print(f"{'#'*75}", flush=True)

    configs_p2 = []
    for nc in [32, 64]:
        for label_16 in viable:
            # Extract mode from 16-class label
            mode = label_16.split('16c ')[1].split(' (')[0]
            label = f'{nc}c {mode} ({BITS_PER_NEURON[mode]}b)'
            for seed in SEEDS:
                configs_p2.append(dict(
                    label=label, seed=seed,
                    n_classes=nc, n_neurons=256,
                    max_attempts=4000, charge_mode=mode, ticks=8))

    print(f"  Running {len(configs_p2)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_p2 = list(pool.map(worker, configs_p2))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    results_32 = [r for r in results_p2 if r['n_classes'] == 32]
    results_64 = [r for r in results_p2 if r['n_classes'] == 64]
    summary_32 = aggregate_results(results_32)
    summary_64 = aggregate_results(results_64)
    print_summary("32-CLASS RESULTS", summary_32)
    print_summary("64-CLASS RESULTS", summary_64)

    # === PHASE 3: Extended 8K on 64-class with best low-precision mode ===
    # Find the lowest-bit mode that's within 5% of float32 on 64-class
    float32_64_label = [k for k in summary_64 if 'float32' in k]
    if float32_64_label:
        baseline_64 = summary_64[float32_64_label[0]]['acc_mean']
        candidates = []
        for label in summary_64:
            s = summary_64[label]
            delta = abs(s['acc_mean'] - baseline_64)
            mode = label.split(f'64c ')[1].split(' (')[0]
            bits = BITS_PER_NEURON.get(mode, 32)
            if delta <= 0.05:
                candidates.append((bits, mode, label))
        candidates.sort()  # sort by bits ascending

        if candidates:
            best_low = candidates[0]  # lowest bit count that works
            best_mode = best_low[1]
            print(f"\n  Lowest viable precision on 64-class: {best_mode} ({best_low[0]} bits)")

            print(f"\n{'#'*75}")
            print(f"  PHASE 3: Extended 8K -- 64-class, {best_mode} vs float32")
            print(f"{'#'*75}", flush=True)

            configs_p3 = []
            for seed in SEEDS:
                configs_p3.append(dict(
                    label=f'64c {best_mode} 8K',
                    seed=seed, n_classes=64, n_neurons=256,
                    max_attempts=8000, charge_mode=best_mode, ticks=8))
                configs_p3.append(dict(
                    label='64c float32 8K',
                    seed=seed, n_classes=64, n_neurons=256,
                    max_attempts=8000, charge_mode='float32', ticks=8))

            print(f"  Running {len(configs_p3)} jobs on {n_workers} cores...", flush=True)
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results_p3 = list(pool.map(worker, configs_p3))
            print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

            summary_p3 = aggregate_results(results_p3)
            print_summary("64-CLASS EXTENDED (8K)", summary_p3)

    # === FINAL ===
    print(f"\n{'#'*75}")
    print(f"  DONE")
    print(f"{'#'*75}", flush=True)
