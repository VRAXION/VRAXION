"""
Learnable Params — Invariant Search Across Sizes
==================================================
All 4 capacitor params (threshold, leak, acc_rate, self_fb) are learnable
simultaneously. Evolution finds the optimum. We OBSERVE where they converge
across different task sizes to find size-independent invariants.

Method:
  - 15% of mutations go to param perturbation (any of the 4)
  - 85% to topology (structure + weights, as usual)
  - Combined scoring (0.5*acc + 0.5*target_prob)
  - Log params every 2000 attempts

Test matrix (all parallel, 5 seeds):
  16-class,  80 neurons,  8K attempts
  32-class, 160 neurons,  8K attempts
  64-class, 320 neurons, 16K attempts

Analysis: classify each param as INVARIANT / SIZE-DEPENDENT / NOISE.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import math
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  LearnableNet — all 4 capacitor params evolve
# ============================================================

class LearnableNet(SelfWiringGraph):
    """SelfWiringGraph with learnable capacitor hyperparams."""

    def __init__(self, n_neurons, vocab, **kwargs):
        super().__init__(n_neurons, vocab, **kwargs)
        # Learnable params — start from current defaults
        self.threshold = 0.5
        self.leak = 0.85
        self.acc_rate = 0.3
        self.self_fb = 0.1

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world

            raw = act @ Weff + act * self.self_fb

            self.charge += raw * self.acc_rate
            self.charge *= self.leak

            act = np.maximum(self.charge - self.threshold, 0.0)

            clamp = self.threshold * 2
            self.charge = np.clip(self.charge, -clamp, clamp)

        self.state = act.copy()
        return self.charge[:self.V]

    def save_state(self):
        base = super().save_state()
        return base + (self.threshold, self.leak, self.acc_rate, self.self_fb)

    def restore_state(self, s):
        super().restore_state(s[:6])
        self.threshold = s[6]
        self.leak = s[7]
        self.acc_rate = s[8]
        self.self_fb = s[9]

    def mutate_params(self):
        """Perturb one random capacitor param."""
        param = random.choice(['threshold', 'leak', 'acc_rate', 'self_fb'])
        if param == 'threshold':
            self.threshold += random.gauss(0, 0.03)
            self.threshold = max(0.01, min(1.5, self.threshold))
        elif param == 'leak':
            self.leak += random.gauss(0, 0.02)
            self.leak = max(0.3, min(0.99, self.leak))
        elif param == 'acc_rate':
            self.acc_rate += random.gauss(0, 0.03)
            self.acc_rate = max(0.01, min(1.5, self.acc_rate))
        elif param == 'self_fb':
            self.self_fb += random.gauss(0, 0.02)
            self.self_fb = max(-0.3, min(0.5, self.self_fb))


# ============================================================
#  Training with learnable params
# ============================================================

def train_config(label, n_classes, n_neurons, seed, max_attempts=8000, ticks=8):
    """Train with all 4 params learnable. Returns result + param trajectory."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = LearnableNet(n_neurons, V)

    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    # Param trajectory log
    trajectory = []

    def evaluate():
        net.reset()
        correct = 0
        total_score = 0.0

        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks)
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

        # Mutation: 15% params, 85% topology
        r = random.random()
        if r < 0.15:
            net.mutate_params()
        elif phase == "STRUCTURE":
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

        # Log every 2000 attempts
        if (att + 1) % 2000 == 0:
            trajectory.append({
                'step': att + 1,
                'acc': best_acc,
                'threshold': net.threshold,
                'leak': net.leak,
                'acc_rate': net.acc_rate,
                'self_fb': net.self_fb,
            })

        if best_acc >= 0.99 or stale >= 3500:
            # Log final state too
            if not trajectory or trajectory[-1]['step'] != att + 1:
                trajectory.append({
                    'step': att + 1,
                    'acc': best_acc,
                    'threshold': net.threshold,
                    'leak': net.leak,
                    'acc_rate': net.acc_rate,
                    'self_fb': net.self_fb,
                })
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    # Log final if not logged yet
    if not trajectory or trajectory[-1]['step'] != att + 1:
        trajectory.append({
            'step': att + 1,
            'acc': best_acc,
            'threshold': net.threshold,
            'leak': net.leak,
            'acc_rate': net.acc_rate,
            'self_fb': net.self_fb,
        })

    return {
        'label': label,
        'seed': seed,
        'n_classes': n_classes,
        'n_neurons': n_neurons,
        'acc': best_acc,
        'score': best_score,
        'kept': kept,
        'accept_rate': accept_rate,
        'time': elapsed,
        'conns': net.count_connections(),
        # Final converged params
        'threshold': net.threshold,
        'leak': net.leak,
        'acc_rate': net.acc_rate,
        'self_fb': net.self_fb,
        # Trajectory
        'trajectory': trajectory,
    }


def worker(args):
    return train_config(**args)


# ============================================================
#  Analysis
# ============================================================

def print_trajectories(results):
    """Print param trajectories per size."""
    groups = defaultdict(list)
    for r in results:
        groups[r['n_classes']].append(r)

    print(f"\n  {'='*90}")
    print(f"  PARAM TRAJECTORIES")
    print(f"  {'='*90}")

    for nc in sorted(groups):
        print(f"\n  --- {nc}-class ---")
        print(f"  {'size':>4s} | {'seed':>5s} | {'step':>5s} | {'acc':>5s} | "
              f"{'threshold':>9s} | {'leak':>6s} | {'acc_rate':>8s} | {'self_fb':>7s}")
        print(f"  {'-'*75}")
        for r in sorted(groups[nc], key=lambda x: x['seed']):
            for t in r['trajectory']:
                print(f"  {nc:4d} | {r['seed']:5d} | {t['step']:5d} | {t['acc']*100:4.0f}% | "
                      f"{t['threshold']:9.4f} | {t['leak']:6.4f} | {t['acc_rate']:8.4f} | {t['self_fb']:7.4f}")
    sys.stdout.flush()


def analyze_convergence(results):
    """Compute convergence stats and classify params."""
    groups = defaultdict(list)
    for r in results:
        groups[r['n_classes']].append(r)

    params = ['threshold', 'leak', 'acc_rate', 'self_fb']

    # Per-size stats
    size_stats = {}
    for nc in sorted(groups):
        runs = groups[nc]
        stats = {}
        for p in params:
            vals = [r[p] for r in runs]
            stats[p] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'values': vals,
            }
        size_stats[nc] = stats

    # Print convergence table
    sizes = sorted(size_stats.keys())
    print(f"\n  {'='*90}")
    print(f"  CONVERGENCE ANALYSIS (final values)")
    print(f"  {'='*90}")

    header = f"  {'Param':<12s}"
    for nc in sizes:
        header += f" | {nc}c mean (std)"
    header += f" | {'Cross-size':>12s} | {'Seed var':>10s} | {'Category':<14s}"
    print(header)
    print(f"  {'-'*90}")

    categories = {}
    for p in params:
        line = f"  {p:<12s}"
        means_across_sizes = []
        seed_vars = []

        for nc in sizes:
            s = size_stats[nc][p]
            line += f" | {s['mean']:6.4f} ({s['std']:5.4f})"
            means_across_sizes.append(s['mean'])
            seed_vars.append(s['std'])

        cross_size_std = np.std(means_across_sizes)
        avg_seed_std = np.mean(seed_vars)

        # Classify
        if cross_size_std < 0.05:
            category = "INVARIANT"
        elif avg_seed_std > cross_size_std:
            category = "NOISE"
        else:
            category = "SIZE-DEPENDENT"

        categories[p] = category
        line += f" | {cross_size_std:12.4f} | {avg_seed_std:10.4f} | {category:<14s}"
        print(line)

    sys.stdout.flush()
    return size_stats, categories


def analyze_scaling(size_stats, categories):
    """For SIZE-DEPENDENT params, fit scaling functions."""
    sizes = sorted(size_stats.keys())

    print(f"\n  {'='*90}")
    print(f"  SCALING ANALYSIS")
    print(f"  {'='*90}")

    for p, cat in categories.items():
        if cat != "SIZE-DEPENDENT":
            # For INVARIANT, just report the global mean
            all_vals = []
            for nc in sizes:
                all_vals.extend(size_stats[nc][p]['values'])
            global_mean = np.mean(all_vals)
            global_std = np.std(all_vals)
            print(f"\n  {p} ({cat}): global mean = {global_mean:.4f} (std {global_std:.4f})")
            continue

        # SIZE-DEPENDENT: try to fit
        means = [size_stats[nc][p]['mean'] for nc in sizes]
        print(f"\n  {p} (SIZE-DEPENDENT):")
        print(f"    Raw data: ", end="")
        for nc, m in zip(sizes, means):
            print(f"{nc}c={m:.4f}  ", end="")
        print()

        # Try linear: param = a + b * nc
        if len(sizes) >= 2:
            x = np.array(sizes, dtype=float)
            y = np.array(means)

            # Linear fit
            coeffs_lin = np.polyfit(x, y, 1)
            pred_lin = np.polyval(coeffs_lin, x)
            err_lin = np.mean((y - pred_lin)**2)

            # Log fit: param = a + b * log2(nc)
            x_log = np.log2(x)
            coeffs_log = np.polyfit(x_log, y, 1)
            pred_log = np.polyval(coeffs_log, x_log)
            err_log = np.mean((y - pred_log)**2)

            # Sqrt fit: param = a + b * sqrt(nc)
            x_sqrt = np.sqrt(x)
            coeffs_sqrt = np.polyfit(x_sqrt, y, 1)
            pred_sqrt = np.polyval(coeffs_sqrt, x_sqrt)
            err_sqrt = np.mean((y - pred_sqrt)**2)

            fits = [
                ('linear', f'{p} = {coeffs_lin[1]:.4f} + {coeffs_lin[0]:.6f} * nc', err_lin),
                ('log2', f'{p} = {coeffs_log[1]:.4f} + {coeffs_log[0]:.4f} * log2(nc)', err_log),
                ('sqrt', f'{p} = {coeffs_sqrt[1]:.4f} + {coeffs_sqrt[0]:.4f} * sqrt(nc)', err_sqrt),
            ]

            fits.sort(key=lambda x: x[2])
            for name, formula, err in fits:
                marker = " <<<" if err == fits[0][2] else ""
                print(f"    {name:>8s}: {formula}  (MSE={err:.6f}){marker}")

    sys.stdout.flush()


def print_accuracy_summary(results):
    """Print accuracy per size."""
    groups = defaultdict(list)
    for r in results:
        groups[r['n_classes']].append(r)

    print(f"\n  {'='*90}")
    print(f"  ACCURACY SUMMARY")
    print(f"  {'='*90}")
    print(f"  {'Config':<25s} {'Acc':>7s} {'Std':>6s} {'Accept%':>8s} {'Time':>6s}")
    print(f"  {'-'*60}")

    for nc in sorted(groups):
        runs = groups[nc]
        accs = [r['acc'] for r in runs]
        rates = [r['accept_rate'] for r in runs]
        times = [r['time'] for r in runs]
        nn = runs[0]['n_neurons']
        att = max(t['step'] for r in runs for t in r['trajectory'])
        label = f"{nc}c / {nn}n / {att//1000}K"
        print(f"  {label:<25s} {np.mean(accs)*100:5.1f}% {np.std(accs)*100:4.1f}% "
              f"{np.mean(rates):7.3f}% {np.mean(times):5.0f}s")
    sys.stdout.flush()


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")
    print(f"  All 4 params learnable: threshold, leak, acc_rate, self_fb")
    print(f"  Mutation split: 15% params, 85% topology")
    print(f"  Scoring: combined (0.5*acc + 0.5*target_prob)")

    # =========================================================
    # BUILD CONFIGS
    # =========================================================
    configs = []

    test_matrix = [
        (16,  80,  8000),
        (32, 160,  8000),
        (64, 320, 16000),
    ]

    for nc, nn, max_att in test_matrix:
        for seed in SEEDS:
            configs.append(dict(
                label=f'{nc}c/{nn}n',
                n_classes=nc, n_neurons=nn, seed=seed,
                max_attempts=max_att, ticks=8))

    print(f"\n{'#'*80}")
    print(f"  LEARNABLE PARAMS — INVARIANT SEARCH")
    print(f"  Configs: {len(configs)} jobs ({len(test_matrix)} sizes x {len(SEEDS)} seeds)")
    for nc, nn, ma in test_matrix:
        print(f"    {nc}-class, {nn} neurons, {ma//1000}K attempts")
    print(f"{'#'*80}", flush=True)

    print(f"\n  Running {len(configs)} jobs on {n_workers} cores...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, configs))
    total_time = time.time() - t0
    print(f"  Completed in {total_time:.0f}s", flush=True)

    # =========================================================
    # ANALYSIS
    # =========================================================

    print_accuracy_summary(results)
    print_trajectories(results)
    size_stats, categories = analyze_convergence(results)
    analyze_scaling(size_stats, categories)

    # =========================================================
    # EXTRA: 8-class and 128-class if time permits
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  EXTRA SIZES: 8-class and 128-class")
    print(f"{'#'*80}", flush=True)

    extra_configs = []
    extra_matrix = [
        (8,   48,  4000),
        (128, 640, 16000),
    ]
    for nc, nn, max_att in extra_matrix:
        for seed in SEEDS:
            extra_configs.append(dict(
                label=f'{nc}c/{nn}n',
                n_classes=nc, n_neurons=nn, seed=seed,
                max_attempts=max_att, ticks=8))

    print(f"\n  Running {len(extra_configs)} extra jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        extra_results = list(pool.map(worker, extra_configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Combine all results
    all_results = results + extra_results

    print_accuracy_summary(all_results)
    print_trajectories(extra_results)  # only print extra trajectories

    print(f"\n{'#'*80}")
    print(f"  FULL ANALYSIS — ALL 5 SIZES")
    print(f"{'#'*80}", flush=True)

    all_size_stats, all_categories = analyze_convergence(all_results)
    analyze_scaling(all_size_stats, all_categories)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print(f"\n{'#'*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*80}")

    print(f"\n  INVARIANT CONSTANTS (same for all sizes):")
    for p, cat in all_categories.items():
        if cat == "INVARIANT":
            all_vals = []
            for nc in sorted(all_size_stats):
                all_vals.extend(all_size_stats[nc][p]['values'])
            print(f"    {p} = {np.mean(all_vals):.4f} (std {np.std(all_vals):.4f})")

    print(f"\n  SIZE-DEPENDENT PARAMS (need scaling formula):")
    for p, cat in all_categories.items():
        if cat == "SIZE-DEPENDENT":
            for nc in sorted(all_size_stats):
                print(f"    {nc}c: {p} = {all_size_stats[nc][p]['mean']:.4f}")

    print(f"\n  NOISE (high seed variance, not significant):")
    for p, cat in all_categories.items():
        if cat == "NOISE":
            all_vals = []
            for nc in sorted(all_size_stats):
                all_vals.extend(all_size_stats[nc][p]['values'])
            print(f"    {p} = {np.mean(all_vals):.4f} (std {np.std(all_vals):.4f}) — use default")

    print(f"\n  Total runtime: {total_time + (time.time() - t0):.0f}s")
    print(f"  {'='*80}")
    print(f"  DONE")
    print(f"  {'='*80}", flush=True)
