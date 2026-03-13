"""
Two Entities — Diff Signal as Rewiring Guide
==============================================
Entity 1 (Thinker): forward pass through the network
Entity 2 (Weaver): diff signal propagates through SAME wiring,
                    leaving traces that guide rewiring

All tests run in PARALLEL across multiple seeds using ProcessPoolExecutor.
All forward passes use CAPACITOR neuron model (canonical config).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Diff rewiring (Entity 2 — the Weaver)
# ============================================================

def rewire_from_diff(net, diff, ticks=4, decay=0.7):
    """Diff signal propagates through the network's own wiring.
    Where it leaves strong traces, connections change.

    diff: target_onehot - softmax(output), size V
    ticks: how far the diff signal propagates
    decay: how quickly the signal fades (0.5=local, 1.0=global)
    """
    signal = np.zeros(net.N, dtype=np.float32)
    signal[:net.V] = diff  # diff enters at I/O neurons

    Weff = net.W * net.mask

    # Diff propagates through the SAME wiring — linearly (no activation)
    for t in range(ticks):
        signal = signal * decay
        raw = signal @ Weff
        signal = raw
        np.clip(signal, -5, 5, out=signal)

    # Trace: absolute signal strength per neuron
    trace = np.abs(signal)
    if trace.max() < 1e-6:
        return 0  # no signal reached anywhere

    threshold = np.percentile(trace, 75)  # top 25% neurons affected
    active_neurons = np.where(trace > threshold)[0]

    changes = 0
    for ni in active_neurons:
        ni = int(ni)
        diff_strength = signal[ni]
        if abs(diff_strength) < 0.01:
            continue

        if diff_strength > 0:
            # Positive trace: this neuron needs MORE excitation
            conns_to = np.argwhere(net.mask[:, ni] != 0).flatten()
            inhib = [int(c) for c in conns_to if net.mask[int(c), ni] < 0]
            if inhib and random.random() < 0.5:
                src = random.choice(inhib)
                net.mask[src, ni] = 1
                changes += 1
            else:
                disconnected = np.where(net.mask[:, ni] == 0)[0]
                disconnected = disconnected[disconnected != ni]
                if len(disconnected) > 0:
                    src = int(np.random.choice(disconnected))
                    net.mask[src, ni] = 1
                    changes += 1
        else:
            # Negative trace: this neuron needs MORE inhibition
            conns_to = np.argwhere(net.mask[:, ni] != 0).flatten()
            excit = [int(c) for c in conns_to if net.mask[int(c), ni] > 0]
            if excit and random.random() < 0.5:
                src = random.choice(excit)
                net.mask[src, ni] = -1
                changes += 1
            else:
                disconnected = np.where(net.mask[:, ni] == 0)[0]
                disconnected = disconnected[disconnected != ni]
                if len(disconnected) > 0:
                    src = int(np.random.choice(disconnected))
                    net.mask[src, ni] = -1
                    changes += 1

    return changes


# ============================================================
#  Evaluate helper (capacitor forward)
# ============================================================

def evaluate(net, inputs, targets, vocab, ticks=6, return_diff=False):
    """Evaluate network. Optionally return avg diff signal."""
    net.reset()
    correct = 0
    total_diff = np.zeros(vocab, dtype=np.float32) if return_diff else None

    for p in range(2):
        for i in range(len(inputs)):
            world = np.zeros(vocab, dtype=np.float32)
            world[inputs[i]] = 1.0
            logits = net.forward(world, ticks, mode='capacitor')
            probs = softmax(logits[:vocab])

            if p == 1:
                if np.argmax(probs) == targets[i]:
                    correct += 1
                if return_diff:
                    target_oh = np.zeros(vocab, dtype=np.float32)
                    target_oh[targets[i]] = 1.0
                    total_diff += (target_oh - probs)

    acc = correct / len(inputs)
    net.last_acc = acc
    if return_diff:
        return acc, total_diff / len(inputs)
    return acc


# ============================================================
#  Training modes
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=4000,
                 diff_ratio=0.0, diff_ticks=4, diff_decay=0.7,
                 use_combined_scoring=False):
    """Train with configurable diff rewiring ratio.

    diff_ratio: fraction of attempts that use diff rewiring (rest = random mutation)
    diff_ticks: how many ticks diff signal propagates
    diff_decay: decay rate of diff signal
    use_combined_scoring: if True, use 0.5*acc + 0.5*target_prob for selection
    """
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    score = evaluate(net, inputs, targets, V)
    best = score
    kept_diff = 0
    kept_random = 0
    total_diff = 0
    total_random = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False

    t0 = time.time()

    for att in range(max_attempts):
        use_diff = random.random() < diff_ratio

        state = net.save_state()

        if use_diff:
            # Entity 2: diff-guided rewiring
            total_diff += 1
            acc_before, avg_diff = evaluate(net, inputs, targets, V, return_diff=True)
            rewire_from_diff(net, avg_diff, ticks=diff_ticks, decay=diff_decay)
        else:
            # Random mutation (Entity 1 baseline)
            total_random += 1
            if phase == "STRUCTURE":
                net.mutate_structure(0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(0.02)
                else:
                    net.mutate_weights()

        net.self_wire()

        if use_combined_scoring:
            # Combined scoring: 0.5*accuracy + 0.5*target_prob
            net.reset()
            correct = 0
            target_prob_sum = 0.0
            for p in range(2):
                for i in range(len(inputs)):
                    world = np.zeros(V, dtype=np.float32)
                    world[inputs[i]] = 1.0
                    logits = net.forward(world, 6, mode='capacitor')
                    probs = softmax(logits[:V])
                    if p == 1:
                        if np.argmax(probs) == targets[i]:
                            correct += 1
                        target_prob_sum += probs[targets[i]]
            acc_part = correct / len(inputs)
            prob_part = target_prob_sum / len(inputs)
            new_score = 0.5 * acc_part + 0.5 * prob_part
            net.last_acc = acc_part
        else:
            new_score = evaluate(net, inputs, targets, V)

        if new_score > score:
            score = new_score
            if use_diff:
                kept_diff += 1
            else:
                kept_random += 1
            stale = 0
            best = max(best, score if not use_combined_scoring else net.last_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if best >= 0.99 or stale >= 3500:
            break

    elapsed = time.time() - t0

    diff_accept = (kept_diff / max(total_diff, 1)) * 100
    rand_accept = (kept_random / max(total_random, 1)) * 100

    return {
        'label': label,
        'seed': seed,
        'n_classes': n_classes,
        'acc': best,
        'kept_diff': kept_diff,
        'kept_random': kept_random,
        'total_diff': total_diff,
        'total_random': total_random,
        'diff_accept_pct': diff_accept,
        'rand_accept_pct': rand_accept,
        'conns': net.count_connections(),
        'time': elapsed,
    }


# ============================================================
#  Worker function for multiprocessing
# ============================================================

def worker(args):
    """Unpacks args and runs train_config."""
    return train_config(**args)


# ============================================================
#  Result aggregation
# ============================================================

def aggregate_results(results, group_key='label'):
    """Group results by label, compute mean and std."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(r)

    summary = {}
    for label, runs in groups.items():
        accs = [r['acc'] for r in runs]
        times = [r['time'] for r in runs]
        diff_accepts = [r['diff_accept_pct'] for r in runs]
        rand_accepts = [r['rand_accept_pct'] for r in runs]
        summary[label] = {
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'time_mean': np.mean(times),
            'diff_accept_mean': np.mean(diff_accepts),
            'rand_accept_mean': np.mean(rand_accepts),
            'n_runs': len(runs),
        }
    return summary


def print_summary(title, summary):
    """Pretty-print aggregated results."""
    print(f"\n  {'='*75}")
    print(f"  {title}")
    print(f"  {'='*75}")
    print(f"  {'Config':<30s} {'Acc':>8s} {'Std':>6s} {'DiffAcc%':>9s} "
          f"{'RandAcc%':>9s} {'Time':>6s} {'N':>3s}")
    print(f"  {'-'*75}")

    # Sort by accuracy descending
    for label in sorted(summary, key=lambda k: summary[k]['acc_mean'], reverse=True):
        s = summary[label]
        print(f"  {label:<30s} {s['acc_mean']*100:7.1f}% {s['acc_std']*100:5.1f}% "
              f"{s['diff_accept_mean']:8.3f}% {s['rand_accept_mean']:8.3f}% "
              f"{s['time_mean']:5.0f}s {s['n_runs']:3d}")
    sys.stdout.flush()


# ============================================================
#  Test definitions
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]


def build_test_A():
    """Test A: Diff ratio comparison on 16-class, 256 neurons."""
    configs = []
    base = dict(n_classes=16, n_neurons=256, max_attempts=4000)

    ratios = [
        ('A: random_only',     0.0),
        ('A: diff_only',       1.0),
        ('A: diff70_rand30',   0.7),
        ('A: diff50_rand50',   0.5),
        ('A: diff30_rand70',   0.3),
        ('A: diff10_rand90',   0.1),
    ]
    for label, ratio in ratios:
        for seed in SEEDS:
            configs.append(dict(label=label, seed=seed, diff_ratio=ratio, **base))
    return configs


def build_test_B():
    """Test B: Diff tick count sweep on 16-class."""
    configs = []
    base = dict(n_classes=16, n_neurons=256, max_attempts=4000, diff_ratio=0.5)

    for ticks in [1, 2, 4, 6]:
        label = f'B: diff_ticks={ticks}'
        for seed in SEEDS:
            configs.append(dict(label=label, seed=seed, diff_ticks=ticks, **base))
    return configs


def build_test_D():
    """Test D: Diff decay sweep on 16-class."""
    configs = []
    base = dict(n_classes=16, n_neurons=256, max_attempts=4000, diff_ratio=0.5)

    for decay in [0.5, 0.7, 0.9, 1.0]:
        label = f'D: diff_decay={decay}'
        for seed in SEEDS:
            configs.append(dict(label=label, seed=seed, diff_decay=decay, **base))
    return configs


def build_test_C(best_diff_ratio, best_diff_ticks, best_diff_decay):
    """Test C: Winner config on 32-class and 64-class."""
    configs = []
    for nc in [32, 64]:
        # Baseline
        for seed in SEEDS:
            configs.append(dict(
                label=f'C: {nc}c baseline',
                seed=seed, n_classes=nc, n_neurons=256,
                max_attempts=4000, diff_ratio=0.0))
        # Winner
        for seed in SEEDS:
            configs.append(dict(
                label=f'C: {nc}c diff_winner',
                seed=seed, n_classes=nc, n_neurons=256,
                max_attempts=4000, diff_ratio=best_diff_ratio,
                diff_ticks=best_diff_ticks, diff_decay=best_diff_decay))
    return configs


def build_test_E(best_diff_ratio, best_diff_ticks, best_diff_decay):
    """Test E: Combined scoring + diff rewiring on 16/32/64-class."""
    configs = []
    for nc in [16, 32, 64]:
        # Combined scoring + diff
        for seed in SEEDS:
            configs.append(dict(
                label=f'E: {nc}c combined+diff',
                seed=seed, n_classes=nc, n_neurons=256,
                max_attempts=4000, diff_ratio=best_diff_ratio,
                diff_ticks=best_diff_ticks, diff_decay=best_diff_decay,
                use_combined_scoring=True))
        # Combined scoring only (no diff)
        for seed in SEEDS:
            configs.append(dict(
                label=f'E: {nc}c combined_only',
                seed=seed, n_classes=nc, n_neurons=256,
                max_attempts=4000, diff_ratio=0.0,
                use_combined_scoring=True))
    return configs


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    n_workers = os.cpu_count() or 4
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")

    # === TEST A: Diff ratio sweep ===
    print(f"\n{'#'*75}")
    print(f"  TEST A: Diff Ratio Sweep (16-class, 256 neurons)")
    print(f"{'#'*75}", flush=True)

    configs_A = build_test_A()
    print(f"  Running {len(configs_A)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_A = list(pool.map(worker, configs_A))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)
    summary_A = aggregate_results(results_A)
    print_summary("TEST A RESULTS: Diff Ratio Sweep", summary_A)

    # Find best diff ratio
    best_A = max(summary_A, key=lambda k: summary_A[k]['acc_mean'])
    best_ratio_str = best_A.split('_')[-1] if 'diff' in best_A else '0.0'
    # Parse ratio from label
    ratio_map = {'A: random_only': 0.0, 'A: diff_only': 1.0,
                 'A: diff70_rand30': 0.7, 'A: diff50_rand50': 0.5,
                 'A: diff30_rand70': 0.3, 'A: diff10_rand90': 0.1}
    best_diff_ratio = ratio_map.get(best_A, 0.5)
    print(f"\n  Best ratio: {best_A} -> diff_ratio={best_diff_ratio}", flush=True)

    # === TEST B + D in parallel ===
    print(f"\n{'#'*75}")
    print(f"  TEST B+D: Diff Ticks + Decay Sweep (parallel)")
    print(f"{'#'*75}", flush=True)

    configs_BD = build_test_B() + build_test_D()
    print(f"  Running {len(configs_BD)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_BD = list(pool.map(worker, configs_BD))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    results_B = [r for r in results_BD if r['label'].startswith('B:')]
    results_D = [r for r in results_BD if r['label'].startswith('D:')]

    summary_B = aggregate_results(results_B)
    summary_D = aggregate_results(results_D)
    print_summary("TEST B RESULTS: Diff Ticks Sweep", summary_B)
    print_summary("TEST D RESULTS: Diff Decay Sweep", summary_D)

    # Find best ticks and decay
    best_B = max(summary_B, key=lambda k: summary_B[k]['acc_mean'])
    best_D = max(summary_D, key=lambda k: summary_D[k]['acc_mean'])
    best_diff_ticks = int(best_B.split('=')[1])
    best_diff_decay = float(best_D.split('=')[1])
    print(f"\n  Best ticks: {best_B} -> diff_ticks={best_diff_ticks}")
    print(f"  Best decay: {best_D} -> diff_decay={best_diff_decay}", flush=True)

    # === TEST C: Winner on 32/64-class ===
    print(f"\n{'#'*75}")
    print(f"  TEST C: Winner on 32/64-class")
    print(f"  Config: ratio={best_diff_ratio}, ticks={best_diff_ticks}, decay={best_diff_decay}")
    print(f"{'#'*75}", flush=True)

    configs_C = build_test_C(best_diff_ratio, best_diff_ticks, best_diff_decay)
    print(f"  Running {len(configs_C)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_C = list(pool.map(worker, configs_C))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)
    summary_C = aggregate_results(results_C)
    print_summary("TEST C RESULTS: 32/64-class", summary_C)

    # === TEST E: Combined scoring + diff ===
    print(f"\n{'#'*75}")
    print(f"  TEST E: Combined Scoring + Diff Rewiring")
    print(f"  Config: ratio={best_diff_ratio}, ticks={best_diff_ticks}, decay={best_diff_decay}")
    print(f"{'#'*75}", flush=True)

    configs_E = build_test_E(best_diff_ratio, best_diff_ticks, best_diff_decay)
    print(f"  Running {len(configs_E)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_E = list(pool.map(worker, configs_E))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)
    summary_E = aggregate_results(results_E)
    print_summary("TEST E RESULTS: Combined Scoring + Diff", summary_E)

    # === FINAL SUMMARY ===
    print(f"\n{'#'*75}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*75}")
    all_summary = {}
    all_summary.update(summary_A)
    all_summary.update(summary_B)
    all_summary.update(summary_C)
    all_summary.update(summary_D)
    all_summary.update(summary_E)
    print_summary("ALL RESULTS", all_summary)

    print(f"\n  DONE", flush=True)
