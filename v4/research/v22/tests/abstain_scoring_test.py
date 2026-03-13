"""
Abstain Scoring Benchmark
==========================
From introspection insight #1: the brain does NOT answer when it doesn't
have enough information. "I don't know" is an ACTIVE response.

Current problem: the network is FORCED to answer even when uncertain.
A wrong guess on class B penalizes a mutation that helped class A.
This is the core of the 64-class interference wall.

Abstain mechanism: if softmax entropy is above a threshold (output too flat),
that input is SKIPPED in scoring — neither correct nor incorrect.
This means mutations that make hard classes uncertain don't get penalized,
while mutations that make easy classes more confident get rewarded.

Combined with combined scoring (0.5*acc + 0.5*target_prob) for maximum effect.

All tests parallel, 5 seeds, capacitor forward.
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
#  Scoring functions
# ============================================================

def entropy(probs):
    """Shannon entropy in bits."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log2(p))


def max_entropy(n_classes):
    """Maximum entropy for n_classes (uniform distribution)."""
    return np.log2(n_classes)


def score_accuracy_only(probs, target, n_classes, abstain_threshold=None):
    """Standard accuracy scoring. Returns (score, counted)."""
    return (1.0 if np.argmax(probs) == target else 0.0), True


def score_combined(probs, target, n_classes, abstain_threshold=None):
    """Combined: 0.5*accuracy + 0.5*target_prob. Returns (score, counted)."""
    acc = 1.0 if np.argmax(probs) == target else 0.0
    tp = float(probs[target])
    return 0.5 * acc + 0.5 * tp, True


def score_abstain_accuracy(probs, target, n_classes, abstain_threshold=0.8):
    """Accuracy but skip uncertain outputs.
    If entropy > threshold * max_entropy: abstain (not counted).
    """
    ent = entropy(probs)
    max_ent = max_entropy(n_classes)
    if ent > abstain_threshold * max_ent:
        return 0.0, False  # abstain — not counted
    return (1.0 if np.argmax(probs) == target else 0.0), True


def score_abstain_combined(probs, target, n_classes, abstain_threshold=0.8):
    """Combined scoring but skip uncertain outputs.
    Best of both: fine-grained signal + don't penalize uncertainty.
    """
    ent = entropy(probs)
    max_ent = max_entropy(n_classes)
    if ent > abstain_threshold * max_ent:
        return 0.0, False  # abstain
    acc = 1.0 if np.argmax(probs) == target else 0.0
    tp = float(probs[target])
    return 0.5 * acc + 0.5 * tp, True


def score_confidence_weighted(probs, target, n_classes, abstain_threshold=None):
    """Score weighted by confidence (1 - normalized entropy).
    Uncertain outputs contribute less, not zero.
    """
    acc = 1.0 if np.argmax(probs) == target else 0.0
    tp = float(probs[target])
    base = 0.5 * acc + 0.5 * tp
    ent = entropy(probs)
    max_ent = max_entropy(n_classes)
    confidence = 1.0 - (ent / max_ent)  # 0=uniform, 1=certain
    return base * confidence, True


# ============================================================
#  Training with configurable scoring
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=4000,
                 score_fn_name='accuracy', abstain_threshold=0.8):
    """Train with a specific scoring function."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    score_fns = {
        'accuracy': score_accuracy_only,
        'combined': score_combined,
        'abstain_acc': score_abstain_accuracy,
        'abstain_combined': score_abstain_combined,
        'confidence_weighted': score_confidence_weighted,
    }
    score_fn = score_fns[score_fn_name]

    def evaluate():
        net.reset()
        total_score = 0.0
        counted = 0
        correct = 0
        abstained = 0

        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks=8)
                probs = softmax(logits[:V])

                if p == 1:
                    s, was_counted = score_fn(probs, targets[i], V, abstain_threshold)
                    if was_counted:
                        total_score += s
                        counted += 1
                        if np.argmax(probs) == targets[i]:
                            correct += 1
                    else:
                        abstained += 1

        # Score = average over counted inputs (not total)
        avg_score = total_score / max(counted, 1)
        acc = correct / len(inputs)  # real accuracy (always over all inputs)
        net.last_acc = acc
        return avg_score, acc, abstained

    score, acc, _ = evaluate()
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
        new_score, new_acc, abstained = evaluate()

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


def print_summary(title, summary):
    print(f"\n  {'='*70}")
    print(f"  {title}")
    print(f"  {'='*70}")
    print(f"  {'Config':<35s} {'Acc':>7s} {'Std':>6s} {'Accept%':>8s} {'Time':>6s}")
    print(f"  {'-'*65}")
    for label in sorted(summary, key=lambda k: summary[k]['acc_mean'], reverse=True):
        s = summary[label]
        print(f"  {label:<35s} {s['acc_mean']*100:6.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['accept_mean']:7.3f}% {s['time_mean']:5.0f}s")
    sys.stdout.flush()


# ============================================================
#  Test configurations
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]


def build_scoring_sweep(n_classes, n_neurons):
    """Test all scoring modes on a given task."""
    configs = []
    base = dict(n_classes=n_classes, n_neurons=n_neurons, max_attempts=4000)

    modes = [
        ('accuracy', 'accuracy', 0.8),
        ('combined', 'combined', 0.8),
        ('abstain_acc_0.8', 'abstain_acc', 0.8),
        ('abstain_acc_0.7', 'abstain_acc', 0.7),
        ('abstain_acc_0.9', 'abstain_acc', 0.9),
        ('abstain_combined_0.8', 'abstain_combined', 0.8),
        ('abstain_combined_0.7', 'abstain_combined', 0.7),
        ('abstain_combined_0.9', 'abstain_combined', 0.9),
        ('confidence_weighted', 'confidence_weighted', 0.8),
    ]

    for label_suffix, fn_name, threshold in modes:
        label = f'{n_classes}c {label_suffix}'
        for seed in SEEDS:
            configs.append(dict(
                label=label, seed=seed,
                score_fn_name=fn_name, abstain_threshold=threshold,
                **base))
    return configs


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    n_workers = os.cpu_count() or 4
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")

    # === Phase 1: Full sweep on 16-class ===
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Scoring Sweep on 16-class (256 neurons)")
    print(f"{'#'*70}", flush=True)

    configs_16 = build_scoring_sweep(16, 256)
    print(f"  Running {len(configs_16)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_16 = list(pool.map(worker, configs_16))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)
    summary_16 = aggregate_results(results_16)
    print_summary("16-CLASS RESULTS", summary_16)

    # Find best scoring mode
    best_16 = max(summary_16, key=lambda k: summary_16[k]['acc_mean'])
    print(f"\n  Best 16-class: {best_16}", flush=True)

    # === Phase 2: Top 3 modes on 32-class and 64-class ===
    top3 = sorted(summary_16, key=lambda k: summary_16[k]['acc_mean'], reverse=True)[:3]
    # Always include accuracy and combined as baselines
    baselines = ['16c accuracy', '16c combined']
    for b in baselines:
        if b not in top3:
            top3.append(b)

    print(f"\n{'#'*70}")
    print(f"  PHASE 2: Top configs on 32-class and 64-class")
    print(f"  Testing: {[t.replace('16c ', '') for t in top3]}")
    print(f"{'#'*70}", flush=True)

    configs_phase2 = []
    for nc in [32, 64]:
        for label_16 in top3:
            # Extract scoring mode from 16-class label
            mode_part = label_16.replace('16c ', '')

            # Parse mode and threshold
            if mode_part == 'accuracy':
                fn, thr = 'accuracy', 0.8
            elif mode_part == 'combined':
                fn, thr = 'combined', 0.8
            elif mode_part == 'confidence_weighted':
                fn, thr = 'confidence_weighted', 0.8
            elif mode_part.startswith('abstain_combined_'):
                fn = 'abstain_combined'
                thr = float(mode_part.split('_')[-1])
            elif mode_part.startswith('abstain_acc_'):
                fn = 'abstain_acc'
                thr = float(mode_part.split('_')[-1])
            else:
                continue

            label = f'{nc}c {mode_part}'
            for seed in SEEDS:
                configs_phase2.append(dict(
                    label=label, seed=seed,
                    n_classes=nc, n_neurons=256,
                    max_attempts=4000,
                    score_fn_name=fn, abstain_threshold=thr))

    print(f"  Running {len(configs_phase2)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_phase2 = list(pool.map(worker, configs_phase2))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    results_32 = [r for r in results_phase2 if r['n_classes'] == 32]
    results_64 = [r for r in results_phase2 if r['n_classes'] == 64]

    summary_32 = aggregate_results(results_32)
    summary_64 = aggregate_results(results_64)
    print_summary("32-CLASS RESULTS", summary_32)
    print_summary("64-CLASS RESULTS", summary_64)

    # === Phase 3: Best overall + longer run on 64-class ===
    best_64 = max(summary_64, key=lambda k: summary_64[k]['acc_mean'])
    print(f"\n  Best 64-class: {best_64}", flush=True)

    # Run the winner with 8000 attempts to see if it keeps climbing
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Extended run — best config on 64-class, 8000 attempts")
    print(f"{'#'*70}", flush=True)

    # Parse best config
    best_mode = best_64.replace('64c ', '')
    if best_mode == 'accuracy':
        best_fn, best_thr = 'accuracy', 0.8
    elif best_mode == 'combined':
        best_fn, best_thr = 'combined', 0.8
    elif best_mode == 'confidence_weighted':
        best_fn, best_thr = 'confidence_weighted', 0.8
    elif best_mode.startswith('abstain_combined_'):
        best_fn = 'abstain_combined'
        best_thr = float(best_mode.split('_')[-1])
    elif best_mode.startswith('abstain_acc_'):
        best_fn = 'abstain_acc'
        best_thr = float(best_mode.split('_')[-1])
    else:
        best_fn, best_thr = 'combined', 0.8

    configs_ext = []
    for seed in SEEDS:
        configs_ext.append(dict(
            label=f'64c {best_mode} 8K',
            seed=seed, n_classes=64, n_neurons=256,
            max_attempts=8000,
            score_fn_name=best_fn, abstain_threshold=best_thr))
        # Baseline comparison
        configs_ext.append(dict(
            label='64c accuracy 8K',
            seed=seed, n_classes=64, n_neurons=256,
            max_attempts=8000,
            score_fn_name='accuracy', abstain_threshold=0.8))

    print(f"  Running {len(configs_ext)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_ext = list(pool.map(worker, configs_ext))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)
    summary_ext = aggregate_results(results_ext)
    print_summary("64-CLASS EXTENDED (8K attempts)", summary_ext)

    # === FINAL ===
    print(f"\n{'#'*70}")
    print(f"  DONE")
    print(f"{'#'*70}", flush=True)
