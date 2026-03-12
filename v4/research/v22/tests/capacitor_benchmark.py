"""
VRAXION v22 — Capacitor Neuron Benchmark
==========================================
Compares capacitor neuron variants against leaky_relu baseline
on 16-class, 32-class, and 64-class lookup tasks.

Capacitor neuron: charge accumulates over ticks, only fires above threshold.
This temporal integration compensates for limited weight precision (binary 0.5/1.5).

Configs tested:
  a) leaky_relu (baseline) — with overflow clipping
  b) capacitor t=0.3 leak=0.85
  c) capacitor t=0.5 leak=0.85
  d) capacitor t=0.5 leak=0.90
  e) cap_bipolar t=0.5 leak=0.85
"""

import numpy as np
import random
import time
import sys
import json
from datetime import datetime


class SelfWiringGraph:
    """Self-wiring graph with configurable activation (leaky_relu / capacitor / cap_bipolar)."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 activation='leaky_relu', threshold=0.5, leak=0.85):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.activation = activation
        self.threshold = threshold
        self.leak = leak

        # Ternary mask
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Binary weights
        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        # 4D addresses for self-wiring
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

        # Capacitor charge (used by capacitor/cap_bipolar activations)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0  # Reset charge for every new eval

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            # First tick: inject input
            if t == 0:
                act[:self.V] = world

            # Signal propagation
            raw = act @ Weff + act * 0.1

            if self.activation == 'leaky_relu':
                act = np.where(raw > 0, raw, np.float32(0.01) * raw)
                # Overflow protection
                act = np.clip(act, -10, 10)

            elif self.activation == 'capacitor':
                # Capacitor dynamics
                self.charge += raw * 0.3
                self.charge *= self.leak
                # Threshold output
                act = np.maximum(self.charge - self.threshold, 0.0)
                # Clamp charge
                self.charge = np.clip(self.charge,
                                      -self.threshold * 2,
                                      self.threshold * 2)

            elif self.activation == 'cap_bipolar':
                # Bipolar capacitor dynamics
                self.charge += raw * 0.3
                self.charge *= self.leak
                # Positive AND negative overflow
                pos_overflow = np.maximum(self.charge - self.threshold, 0.0)
                neg_overflow = np.minimum(self.charge + self.threshold, 0.0)
                act = pos_overflow + neg_overflow
                # Clamp charge
                self.charge = np.clip(self.charge,
                                      -self.threshold * 2,
                                      self.threshold * 2)

        self.state = act.copy()

        # Output: read CHARGE for capacitor variants (richer signal for softmax)
        if self.activation in ('capacitor', 'cap_bipolar'):
            return self.charge[:self.V]
        else:
            return act[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def charge_stats(self):
        """Return charge distribution stats for internal neurons."""
        if self.activation in ('capacitor', 'cap_bipolar'):
            internal = self.charge[self.V:]
            return {
                'mean': float(np.mean(internal)),
                'std': float(np.std(internal)),
                'min': float(np.min(internal)),
                'max': float(np.max(internal)),
            }
        return None

    # === State management ===

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy())

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()

    # === Mutation operators ===

    def mutate_structure(self, rate=0.05):
        r = random.random()
        if r < self.flip_rate:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate * 0.5))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.mask[r2, c] *= -1
        else:
            action = random.choice(['add_pos', 'add_neg', 'remove', 'rewire'])
            if action in ('add_pos', 'add_neg'):
                dead = np.argwhere(self.mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    n = max(1, int(len(dead) * rate))
                    idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
                    sign = 1.0 if action == 'add_pos' else -1.0
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        self.mask[r2, c] = sign
                        self.W[r2, c] = random.choice([np.float32(0.5), np.float32(1.5)])
            elif action == 'remove':
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        self.mask[int(idx[j][0]), int(idx[j][1])] = 0
            else:  # rewire
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    for j in range(len(idx)):
                        r2, c = int(idx[j][0]), int(idx[j][1])
                        old_sign = self.mask[r2, c]
                        old_w = self.W[r2, c]
                        self.mask[r2, c] = 0
                        nc = random.randint(0, self.N - 1)
                        while nc == r2:
                            nc = random.randint(0, self.N - 1)
                        self.mask[r2, nc] = old_sign
                        self.W[r2, nc] = old_w

    def mutate_weights(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            for j in range(len(idx)):
                r2, c = int(idx[j][0]), int(idx[j][1])
                self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)

    def self_wire(self):
        if self.last_acc < 0.3:
            top_k, max_new = 2, 1
        elif self.last_acc < 0.7:
            top_k, max_new = 3, 2
        else:
            top_k, max_new = 5, 3
        act = self.state
        a2 = np.abs(act[self.V:])
        if a2.sum() < 0.01:
            return
        nc = min(top_k, len(a2))
        top = np.argpartition(a2, -nc)[-nc:] + self.V
        new = 0
        for ni in top:
            ni = int(ni)
            if np.abs(act[ni]) < 0.1:
                continue
            tgt = self.addr[ni] + np.abs(act[ni]) * self.target_W[ni]
            d = ((self.addr - tgt) ** 2).sum(axis=1)
            d[ni] = float('inf')
            near = int(np.argmin(d))
            if self.mask[ni, near] == 0:
                self.mask[ni, near] = random.choice([-1.0, 1.0])
                self.W[ni, near] = random.choice([np.float32(0.5), np.float32(1.5)])
                new += 1
            if new >= max_new:
                break


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8,
          log_interval=2000, label=""):
    """Train with detailed logging every log_interval attempts."""

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks)
                probs = softmax(logits)
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / len(inputs)
        net.last_acc = acc
        return acc

    score = evaluate()
    best = score
    phase = "STRUCTURE"
    kept = 0
    total = 0
    stale = 0
    switched = False
    logs = []

    for att in range(max_attempts):
        state = net.save_state()
        total += 1

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        # Logging
        if (att + 1) % log_interval == 0:
            pos, neg = net.pos_neg_ratio()
            accept_rate = kept / total * 100 if total > 0 else 0
            charge_s = net.charge_stats()

            log_entry = {
                'attempt': att + 1,
                'accuracy': best,
                'connections': net.count_connections(),
                'pos': pos,
                'neg': neg,
                'kept': kept,
                'total': total,
                'accept_rate': accept_rate,
                'phase': phase,
                'charge_stats': charge_s,
            }
            logs.append(log_entry)

            charge_str = ""
            if charge_s:
                charge_str = (f" | Charge: mean={charge_s['mean']:.3f} "
                              f"std={charge_s['std']:.3f} "
                              f"min={charge_s['min']:.3f} "
                              f"max={charge_s['max']:.3f}")

            print(f"  [{att+1:6d}] {label:20s} | Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Kept: {kept:4d}/{total} ({accept_rate:4.1f}%) | "
                  f"Phase: {phase}{charge_str}")

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, kept, logs


# === Config definitions ===

CONFIGS = {
    'leaky_relu': {'activation': 'leaky_relu', 'threshold': 0.0, 'leak': 0.0},
    'cap_t03_l85': {'activation': 'capacitor', 'threshold': 0.3, 'leak': 0.85},
    'cap_t05_l85': {'activation': 'capacitor', 'threshold': 0.5, 'leak': 0.85},
    'cap_t05_l90': {'activation': 'capacitor', 'threshold': 0.5, 'leak': 0.90},
    'cap_bipolar_t05': {'activation': 'cap_bipolar', 'threshold': 0.5, 'leak': 0.85},
}


def run_benchmark(task_classes, n_neurons, max_attempts, configs_to_run, seed=42, ticks=8):
    """Run benchmark for a set of configurations."""
    results = {}

    perm = None  # Will be generated per seed

    for name, cfg in configs_to_run.items():
        print(f"\n{'='*70}")
        print(f"  Config: {name} | {task_classes}-class | {n_neurons} neurons | {max_attempts} attempts")
        print(f"  activation={cfg['activation']} threshold={cfg['threshold']} leak={cfg['leak']}")
        print(f"{'='*70}")

        np.random.seed(seed)
        random.seed(seed)

        perm = np.random.permutation(task_classes)
        inputs = list(range(task_classes))

        net = SelfWiringGraph(
            n_neurons, task_classes,
            activation=cfg['activation'],
            threshold=cfg['threshold'],
            leak=cfg['leak'],
        )

        t0 = time.time()
        best_acc, total_kept, logs = train(
            net, inputs, perm, task_classes,
            max_attempts=max_attempts, ticks=ticks,
            log_interval=2000, label=name,
        )
        elapsed = time.time() - t0

        pos, neg = net.pos_neg_ratio()
        results[name] = {
            'accuracy': best_acc,
            'connections': net.count_connections(),
            'pos': pos,
            'neg': neg,
            'kept': total_kept,
            'time': elapsed,
            'logs': logs,
        }

        print(f"\n  FINAL [{name}]: {best_acc*100:.1f}% | "
              f"Conns: {net.count_connections()} (+:{pos} -:{neg}) | "
              f"Time: {elapsed:.1f}s")

    return results


def print_summary(results, task_label):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {task_label}")
    print(f"{'='*70}")
    print(f"  {'Config':<25s} {'Acc':>7s} {'Conns':>6s} {'Kept':>6s} {'Time':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*8}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, r in sorted_results:
        print(f"  {name:<25s} {r['accuracy']*100:6.1f}% {r['connections']:6d} {r['kept']:6d} {r['time']:7.1f}s")
    print()


def main():
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== A) 16-class, 80 neurons, 16000 attempts =====
    print("\n" + "#" * 70)
    print("#  BENCHMARK A: 16-class lookup, 80 neurons, 16000 attempts")
    print("#" * 70)

    results_16 = run_benchmark(
        task_classes=16, n_neurons=80,
        max_attempts=16000, configs_to_run=CONFIGS,
    )
    print_summary(results_16, "16-class, 80 neurons")
    all_results['16_class'] = results_16

    # ===== B) 32-class, 160 neurons, 20000 attempts =====
    print("\n" + "#" * 70)
    print("#  BENCHMARK B: 32-class lookup, 160 neurons, 20000 attempts")
    print("#" * 70)

    results_32 = run_benchmark(
        task_classes=32, n_neurons=160,
        max_attempts=20000, configs_to_run=CONFIGS,
    )
    print_summary(results_32, "32-class, 160 neurons")
    all_results['32_class'] = results_32

    # ===== C) 64-class, 320 neurons, 50000 attempts =====
    # Pick the best capacitor config from 16+32 results
    cap_configs = {k: v for k, v in CONFIGS.items() if k != 'leaky_relu'}
    best_cap_name = None
    best_cap_score = -1

    for name in cap_configs:
        score_16 = results_16.get(name, {}).get('accuracy', 0)
        score_32 = results_32.get(name, {}).get('accuracy', 0)
        combined = score_16 + score_32
        if combined > best_cap_score:
            best_cap_score = combined
            best_cap_name = name

    configs_64 = {
        'leaky_relu': CONFIGS['leaky_relu'],
        best_cap_name: CONFIGS[best_cap_name],
    }

    print(f"\n  Best capacitor for 64-class test: {best_cap_name}")

    print("\n" + "#" * 70)
    print("#  BENCHMARK C: 64-class lookup, 320 neurons, 50000 attempts")
    print("#" * 70)

    results_64 = run_benchmark(
        task_classes=64, n_neurons=320,
        max_attempts=50000, configs_to_run=configs_64,
    )
    print_summary(results_64, "64-class, 320 neurons")
    all_results['64_class'] = results_64

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("  FINAL CROSS-TASK COMPARISON")
    print("=" * 70)

    for task_label, results in all_results.items():
        print(f"\n  {task_label}:")
        sorted_r = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for name, r in sorted_r:
            print(f"    {name:<25s} {r['accuracy']*100:6.1f}% ({r['time']:.0f}s)")

    # Determine overall winner
    cap_wins = 0
    lr_wins = 0
    for task_label, results in all_results.items():
        lr_acc = results.get('leaky_relu', {}).get('accuracy', 0)
        best_cap_acc = max(
            (r['accuracy'] for n, r in results.items() if n != 'leaky_relu'),
            default=0
        )
        if best_cap_acc > lr_acc:
            cap_wins += 1
        elif lr_acc > best_cap_acc:
            lr_wins += 1

    print(f"\n  Capacitor wins: {cap_wins}/3 tasks")
    print(f"  Leaky ReLU wins: {lr_wins}/3 tasks")

    if cap_wins >= 2:
        print("\n  >>> CAPACITOR IS THE NEW CHAMPION <<<")
    elif lr_wins >= 2:
        print("\n  >>> LEAKY RELU REMAINS CHAMPION <<<")
    else:
        print("\n  >>> TIE — NEED MORE DATA <<<")

    # Save JSON results (without logs for readability)
    json_results = {}
    for task, results in all_results.items():
        json_results[task] = {}
        for name, r in results.items():
            json_results[task][name] = {
                'accuracy': r['accuracy'],
                'connections': r['connections'],
                'pos': r['pos'],
                'neg': r['neg'],
                'kept': r['kept'],
                'time': round(r['time'], 1),
            }

    json_path = f"/home/user/VRAXION/v4/research/v22/tests/capacitor_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    return all_results, cap_wins >= 2


if __name__ == "__main__":
    main()
