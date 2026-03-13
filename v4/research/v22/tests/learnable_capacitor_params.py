"""
VRAXION v22 — Learnable Capacitor Parameters via Evolution
============================================================
The capacitor neuron has 4 parameters that were hand-tuned:
  threshold=0.5, leak=0.85, acc_rate=0.3, self_fb=0.1

A manual sweep (testing one-at-a-time) found:
  threshold → 0.3  (+25%)
  leak      → 0.95 (+18.7%)
  acc_rate  → 0.5  (+18.7%)
  self_fb   → 0.0  (+6.2%)

But sweep tested INDEPENDENTLY — interactions are unknown.
Can evolution find the optimum by co-adapting all 4 simultaneously?

Tests:
  A) Global learnable: 16-class, 80N, 8000 attempts, 5 seeds
     - Log parameter convergence every 1000 attempts
     - Does evolution agree with the sweep?

  B) Per-neuron learnable: each neuron has its own threshold + leak
     - 16-class, 80N, 8000 attempts, 5 seeds
     - Log distribution stats (mean, std, min, max)

  C) 32 and 64-class with winning config from A/B
"""

import numpy as np
import random
import time
import json
from datetime import datetime


# ============================================================
# Network with GLOBAL learnable capacitor params
# ============================================================

class CapacitorNetGlobal:
    """Capacitor neuron with 4 globally learnable parameters."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0

        # Learnable capacitor params (evolve alongside topology)
        self.threshold = 0.5
        self.leak = 0.85
        self.acc_rate = 0.3
        self.self_fb = 0.1

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
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

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
            self.charge = np.clip(self.charge,
                                  -self.threshold * 2,
                                  self.threshold * 2)

        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy(),
                self.threshold, self.leak, self.acc_rate, self.self_fb)

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()
        self.threshold, self.leak, self.acc_rate, self.self_fb = s[6], s[7], s[8], s[9]

    def mutate(self):
        """15% param mutation, 85% topology mutation."""
        r = random.random()
        if r < 0.15:
            self._mutate_params()
        else:
            self._mutate_topology()

    def _mutate_params(self):
        """Mutate one of the 4 learnable capacitor parameters."""
        param = random.choice(['threshold', 'leak', 'acc_rate', 'self_fb'])
        if param == 'threshold':
            self.threshold += random.gauss(0, 0.03)
            self.threshold = max(0.01, min(1.0, self.threshold))
        elif param == 'leak':
            self.leak += random.gauss(0, 0.02)
            self.leak = max(0.5, min(0.99, self.leak))
        elif param == 'acc_rate':
            self.acc_rate += random.gauss(0, 0.03)
            self.acc_rate = max(0.05, min(1.0, self.acc_rate))
        elif param == 'self_fb':
            self.self_fb += random.gauss(0, 0.02)
            self.self_fb = max(0.0, min(0.5, self.self_fb))

    def _mutate_topology(self):
        """Structure + weight mutation (same as v22_best_config)."""
        phase_r = random.random()
        if phase_r < 0.3:
            # Weight mutation
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * 0.05))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)
        else:
            # Structure mutation
            rate = 0.05
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


# ============================================================
# Network with PER-NEURON learnable params (threshold + leak)
# ============================================================

class CapacitorNetPerNeuron:
    """Capacitor neuron with per-neuron threshold and leak."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0

        # Per-neuron learnable params
        self.threshold = np.full(n_neurons, 0.5, dtype=np.float32)
        self.leak = np.full(n_neurons, 0.85, dtype=np.float32)
        # Global acc_rate and self_fb (keep simple — only 2 per-neuron params)
        self.acc_rate = 0.3
        self.self_fb = 0.1

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

        # 4D addresses
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * self.self_fb
            self.charge += raw * self.acc_rate
            self.charge *= self.leak  # per-neuron leak (vectorized)
            act = np.maximum(self.charge - self.threshold, 0.0)  # per-neuron threshold
            # Clamp charge per-neuron
            self.charge = np.clip(self.charge,
                                  -self.threshold * 2,
                                  self.threshold * 2)

        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy(), self.charge.copy(),
                self.threshold.copy(), self.leak.copy(),
                self.acc_rate, self.self_fb)

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()
        self.charge = s[5].copy()
        self.threshold = s[6].copy()
        self.leak = s[7].copy()
        self.acc_rate, self.self_fb = s[8], s[9]

    def mutate(self):
        """15% per-neuron param mutation, 85% topology mutation."""
        r = random.random()
        if r < 0.15:
            self._mutate_per_neuron_params()
        else:
            self._mutate_topology()

    def _mutate_per_neuron_params(self):
        """Mutate one random neuron's threshold or leak."""
        ni = random.randint(0, self.N - 1)
        if random.random() < 0.5:
            self.threshold[ni] += random.gauss(0, 0.03)
            self.threshold[ni] = max(0.01, min(1.0, self.threshold[ni]))
        else:
            self.leak[ni] += random.gauss(0, 0.02)
            self.leak[ni] = max(0.5, min(0.99, self.leak[ni]))

    def _mutate_topology(self):
        """Same topology mutation as global version."""
        phase_r = random.random()
        if phase_r < 0.3:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * 0.05))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)
        else:
            rate = 0.05
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
                else:
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


# ============================================================
# Fixed-param baseline (for comparison)
# ============================================================

class CapacitorNetFixed:
    """Capacitor with fixed default params (baseline)."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 threshold=0.5, leak=0.85, acc_rate=0.3, self_fb=0.1):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.threshold = threshold
        self.leak = leak
        self.acc_rate = acc_rate
        self.self_fb = self_fb

        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0
        self.addr[vocab:, 3] = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

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
            self.charge = np.clip(self.charge,
                                  -self.threshold * 2,
                                  self.threshold * 2)
        self.state = act.copy()
        return self.charge[:self.V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

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

    def mutate(self):
        """Topology-only mutation (no param mutation)."""
        self._mutate_topology()

    def _mutate_topology(self):
        phase_r = random.random()
        if phase_r < 0.3:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * 0.05))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)
        else:
            rate = 0.05
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
                else:
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


# ============================================================
# Softmax + scoring
# ============================================================

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def score_combined(probs, target, vocab):
    acc = 1.0 if np.argmax(probs) == target else 0.0
    return 0.5 * acc + 0.5 * float(probs[target])


# ============================================================
# Evaluate
# ============================================================

def evaluate(net, inputs, targets, vocab, ticks=8):
    """Two-pass evaluation. Returns (combined_score, accuracy)."""
    net.reset()
    total_score = 0.0
    correct = 0
    n = len(inputs)

    for p in range(2):
        for i in range(n):
            world = np.zeros(vocab, dtype=np.float32)
            world[inputs[i]] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1:
                total_score += score_combined(probs, targets[i], vocab)
                if np.argmax(probs) == targets[i]:
                    correct += 1

    return total_score / n, correct / n


# ============================================================
# Training loop (unified for all net types)
# ============================================================

def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8,
          log_interval=1000, label="", is_global=False, is_per_neuron=False):
    """Train using mutation+selection with combined scoring."""

    score, accuracy = evaluate(net, inputs, targets, vocab, ticks)
    best_score = score
    best_acc = accuracy
    kept = 0
    stale = 0
    logs = []

    for att in range(max_attempts):
        state = net.save_state()
        net.mutate()

        new_score, new_acc = evaluate(net, inputs, targets, vocab, ticks)

        if new_score > score:
            score = new_score
            accuracy = new_acc
            kept += 1
            stale = 0
            if score > best_score:
                best_score = score
                best_acc = accuracy
        else:
            net.restore_state(state)
            stale += 1

        # Logging
        if (att + 1) % log_interval == 0:
            pos, neg = net.pos_neg_ratio()
            conns = net.count_connections()

            log_entry = {
                'step': att + 1,
                'accuracy': round(best_acc, 4),
                'connections': conns,
                'kept': kept,
            }

            # Global params
            if is_global:
                log_entry['threshold'] = round(net.threshold, 4)
                log_entry['leak'] = round(net.leak, 4)
                log_entry['acc_rate'] = round(net.acc_rate, 4)
                log_entry['self_fb'] = round(net.self_fb, 4)

                print(f"  [{att+1:5d}] {label:20s} | Acc: {best_acc*100:5.1f}% | "
                      f"thresh={net.threshold:.3f} leak={net.leak:.3f} "
                      f"acc_r={net.acc_rate:.3f} self_fb={net.self_fb:.3f} | "
                      f"Conns: {conns:4d} | Kept: {kept:3d}")

            # Per-neuron params
            elif is_per_neuron:
                t_mean = float(np.mean(net.threshold))
                t_std = float(np.std(net.threshold))
                t_min = float(np.min(net.threshold))
                t_max = float(np.max(net.threshold))
                l_mean = float(np.mean(net.leak))
                l_std = float(np.std(net.leak))
                l_min = float(np.min(net.leak))
                l_max = float(np.max(net.leak))

                log_entry['threshold_mean'] = round(t_mean, 4)
                log_entry['threshold_std'] = round(t_std, 4)
                log_entry['threshold_min'] = round(t_min, 4)
                log_entry['threshold_max'] = round(t_max, 4)
                log_entry['leak_mean'] = round(l_mean, 4)
                log_entry['leak_std'] = round(l_std, 4)
                log_entry['leak_min'] = round(l_min, 4)
                log_entry['leak_max'] = round(l_max, 4)

                print(f"  [{att+1:5d}] {label:20s} | Acc: {best_acc*100:5.1f}% | "
                      f"thresh: {t_mean:.3f}+/-{t_std:.3f} [{t_min:.2f},{t_max:.2f}] | "
                      f"leak: {l_mean:.3f}+/-{l_std:.3f} [{l_min:.2f},{l_max:.2f}] | "
                      f"Conns: {conns:4d} | Kept: {kept:3d}")

            # Fixed params
            else:
                print(f"  [{att+1:5d}] {label:20s} | Acc: {best_acc*100:5.1f}% | "
                      f"Conns: {conns:4d} | Kept: {kept:3d}")

            logs.append(log_entry)

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc, kept, logs


# ============================================================
# TEST A: Global learnable — 16-class, 5 seeds
# ============================================================

def test_a_global_learnable():
    """16-class, 80 neurons, 8000 attempts, 5 seeds.
    Compare: fixed_default vs fixed_sweep vs global_learnable."""

    V = 16
    N = 80
    MAX_ATT = 8000
    SEEDS = [42, 123, 456, 789, 1337]

    print("\n" + "#" * 70)
    print("#  TEST A: Global Learnable Capacitor Params")
    print(f"#  {V}-class, {N} neurons, {MAX_ATT} attempts, {len(SEEDS)} seeds")
    print("#  Configs: fixed_default | fixed_sweep | global_learnable")
    print("#" * 70)

    configs = {
        'fixed_default': {'threshold': 0.5, 'leak': 0.85, 'acc_rate': 0.3, 'self_fb': 0.1},
        'fixed_sweep':   {'threshold': 0.3, 'leak': 0.95, 'acc_rate': 0.5, 'self_fb': 0.0},
        'global_learn':  None,  # learnable
    }

    all_results = {}

    for cfg_name, cfg_params in configs.items():
        seed_results = []

        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  {cfg_name} | seed={seed}")
            print(f"{'='*60}")

            np.random.seed(seed)
            random.seed(seed)
            perm = np.random.permutation(V)
            inputs = list(range(V))

            if cfg_name == 'global_learn':
                net = CapacitorNetGlobal(N, V)
                is_global = True
            else:
                net = CapacitorNetFixed(N, V, **cfg_params)
                is_global = False

            t0 = time.time()
            best_acc, total_kept, logs = train(
                net, inputs, perm, V,
                max_attempts=MAX_ATT, ticks=8, log_interval=1000,
                label=f"{cfg_name}_s{seed}",
                is_global=is_global,
            )
            elapsed = time.time() - t0

            result = {
                'seed': seed,
                'accuracy': best_acc,
                'kept': total_kept,
                'time': round(elapsed, 1),
                'connections': net.count_connections(),
                'logs': logs,
            }

            # Record final params for global learnable
            if cfg_name == 'global_learn':
                result['final_threshold'] = round(net.threshold, 4)
                result['final_leak'] = round(net.leak, 4)
                result['final_acc_rate'] = round(net.acc_rate, 4)
                result['final_self_fb'] = round(net.self_fb, 4)

            seed_results.append(result)

            print(f"\n  DONE: {best_acc*100:.1f}% in {elapsed:.1f}s")
            if cfg_name == 'global_learn':
                print(f"  Final params: thresh={net.threshold:.4f} leak={net.leak:.4f} "
                      f"acc_rate={net.acc_rate:.4f} self_fb={net.self_fb:.4f}")

        all_results[cfg_name] = seed_results

    # Summary
    print(f"\n{'='*70}")
    print(f"  TEST A SUMMARY — 16-class Global Learnable")
    print(f"{'='*70}")
    print(f"  {'Config':<20s} {'Mean Acc':>9s} {'Std':>7s} {'Best':>7s} {'Worst':>7s}")
    print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

    for cfg_name, seed_results in all_results.items():
        accs = [r['accuracy'] for r in seed_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        best_acc = np.max(accs)
        worst_acc = np.min(accs)
        print(f"  {cfg_name:<20s} {mean_acc*100:7.1f}% {std_acc*100:6.1f}% "
              f"{best_acc*100:6.1f}% {worst_acc*100:6.1f}%")

    # Show convergence of global learnable params
    if 'global_learn' in all_results:
        print(f"\n  PARAMETER CONVERGENCE (global_learnable):")
        print(f"  {'Seed':>6s} {'threshold':>10s} {'leak':>8s} {'acc_rate':>10s} {'self_fb':>8s} {'Acc':>7s}")
        print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*7}")
        for r in all_results['global_learn']:
            print(f"  {r['seed']:6d} {r['final_threshold']:10.4f} {r['final_leak']:8.4f} "
                  f"{r['final_acc_rate']:10.4f} {r['final_self_fb']:8.4f} {r['accuracy']*100:6.1f}%")

        # Averages
        ft = [r['final_threshold'] for r in all_results['global_learn']]
        fl = [r['final_leak'] for r in all_results['global_learn']]
        fa = [r['final_acc_rate'] for r in all_results['global_learn']]
        fs = [r['final_self_fb'] for r in all_results['global_learn']]
        print(f"\n  MEAN:  thresh={np.mean(ft):.4f}+/-{np.std(ft):.4f}  "
              f"leak={np.mean(fl):.4f}+/-{np.std(fl):.4f}  "
              f"acc_rate={np.mean(fa):.4f}+/-{np.std(fa):.4f}  "
              f"self_fb={np.mean(fs):.4f}+/-{np.std(fs):.4f}")
        print(f"  SWEEP: thresh=0.3000  leak=0.9500  acc_rate=0.5000  self_fb=0.0000")
        print(f"  {'MATCH' if abs(np.mean(ft) - 0.3) < 0.1 and abs(np.mean(fl) - 0.95) < 0.05 else 'DIVERGE'}: "
              f"Evolution {'validates' if abs(np.mean(ft) - 0.3) < 0.1 else 'contradicts'} sweep")

    return all_results


# ============================================================
# TEST B: Per-neuron learnable — 16-class, 5 seeds
# ============================================================

def test_b_per_neuron():
    """16-class, 80 neurons, 8000 attempts, 5 seeds.
    Compare: global_learnable vs per_neuron_learnable."""

    V = 16
    N = 80
    MAX_ATT = 8000
    SEEDS = [42, 123, 456, 789, 1337]

    print("\n" + "#" * 70)
    print("#  TEST B: Per-Neuron Learnable (threshold + leak)")
    print(f"#  {V}-class, {N} neurons, {MAX_ATT} attempts, {len(SEEDS)} seeds")
    print("#" * 70)

    all_results = {}

    for mode in ['global_learn', 'per_neuron']:
        seed_results = []

        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  {mode} | seed={seed}")
            print(f"{'='*60}")

            np.random.seed(seed)
            random.seed(seed)
            perm = np.random.permutation(V)
            inputs = list(range(V))

            if mode == 'global_learn':
                net = CapacitorNetGlobal(N, V)
                is_global, is_pn = True, False
            else:
                net = CapacitorNetPerNeuron(N, V)
                is_global, is_pn = False, True

            t0 = time.time()
            best_acc, total_kept, logs = train(
                net, inputs, perm, V,
                max_attempts=MAX_ATT, ticks=8, log_interval=1000,
                label=f"{mode}_s{seed}",
                is_global=is_global, is_per_neuron=is_pn,
            )
            elapsed = time.time() - t0

            result = {
                'seed': seed,
                'accuracy': best_acc,
                'kept': total_kept,
                'time': round(elapsed, 1),
                'connections': net.count_connections(),
                'logs': logs,
            }

            if mode == 'per_neuron':
                result['threshold_distribution'] = {
                    'mean': round(float(np.mean(net.threshold)), 4),
                    'std': round(float(np.std(net.threshold)), 4),
                    'min': round(float(np.min(net.threshold)), 4),
                    'max': round(float(np.max(net.threshold)), 4),
                    'q25': round(float(np.percentile(net.threshold, 25)), 4),
                    'q50': round(float(np.percentile(net.threshold, 50)), 4),
                    'q75': round(float(np.percentile(net.threshold, 75)), 4),
                }
                result['leak_distribution'] = {
                    'mean': round(float(np.mean(net.leak)), 4),
                    'std': round(float(np.std(net.leak)), 4),
                    'min': round(float(np.min(net.leak)), 4),
                    'max': round(float(np.max(net.leak)), 4),
                    'q25': round(float(np.percentile(net.leak, 25)), 4),
                    'q50': round(float(np.percentile(net.leak, 50)), 4),
                    'q75': round(float(np.percentile(net.leak, 75)), 4),
                }

            seed_results.append(result)

        all_results[mode] = seed_results

    # Summary
    print(f"\n{'='*70}")
    print(f"  TEST B SUMMARY — Per-Neuron vs Global Learnable")
    print(f"{'='*70}")
    print(f"  {'Config':<20s} {'Mean Acc':>9s} {'Std':>7s} {'Best':>7s} {'Worst':>7s}")
    print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

    for mode, seed_results in all_results.items():
        accs = [r['accuracy'] for r in seed_results]
        print(f"  {mode:<20s} {np.mean(accs)*100:7.1f}% {np.std(accs)*100:6.1f}% "
              f"{np.max(accs)*100:6.1f}% {np.min(accs)*100:6.1f}%")

    # Per-neuron distribution analysis
    if 'per_neuron' in all_results:
        print(f"\n  PER-NEURON PARAM DISTRIBUTIONS:")
        for r in all_results['per_neuron']:
            td = r['threshold_distribution']
            ld = r['leak_distribution']
            print(f"  seed={r['seed']:4d} | thresh: mean={td['mean']:.3f} std={td['std']:.3f} "
                  f"[{td['min']:.2f},{td['max']:.2f}] | "
                  f"leak: mean={ld['mean']:.3f} std={ld['std']:.3f} "
                  f"[{ld['min']:.2f},{ld['max']:.2f}] | "
                  f"acc={r['accuracy']*100:.1f}%")

        # Convergence check
        t_stds = [r['threshold_distribution']['std'] for r in all_results['per_neuron']]
        l_stds = [r['leak_distribution']['std'] for r in all_results['per_neuron']]
        mean_t_std = np.mean(t_stds)
        mean_l_std = np.mean(l_stds)
        print(f"\n  Avg threshold std: {mean_t_std:.4f} ({'CONVERGED' if mean_t_std < 0.05 else 'SPREAD'})")
        print(f"  Avg leak std:      {mean_l_std:.4f} ({'CONVERGED' if mean_l_std < 0.03 else 'SPREAD'})")

    return all_results


# ============================================================
# TEST C: 32 and 64-class with winning config
# ============================================================

def test_c_scale(winner_type):
    """Run 32-class and 64-class with the winning config from A/B."""

    configs = [
        # (V, N, max_att)
        (32, 160, 12000),
        (64, 320, 20000),
    ]
    SEEDS = [42, 123, 456]

    print("\n" + "#" * 70)
    print(f"#  TEST C: Scale Test — 32 and 64-class")
    print(f"#  Winner: {winner_type}")
    print("#" * 70)

    all_results = {}

    for V, N, MAX_ATT in configs:
        task_results = {}

        for mode in ['fixed_default', 'fixed_sweep', winner_type]:
            seed_results = []

            for seed in SEEDS:
                print(f"\n{'='*60}")
                print(f"  {mode} | {V}-class | seed={seed}")
                print(f"{'='*60}")

                np.random.seed(seed)
                random.seed(seed)
                perm = np.random.permutation(V)
                inputs = list(range(V))

                is_global = False
                is_pn = False

                if mode == 'fixed_default':
                    net = CapacitorNetFixed(N, V, threshold=0.5, leak=0.85,
                                           acc_rate=0.3, self_fb=0.1)
                elif mode == 'fixed_sweep':
                    net = CapacitorNetFixed(N, V, threshold=0.3, leak=0.95,
                                           acc_rate=0.5, self_fb=0.0)
                elif mode == 'global_learn':
                    net = CapacitorNetGlobal(N, V)
                    is_global = True
                elif mode == 'per_neuron':
                    net = CapacitorNetPerNeuron(N, V)
                    is_pn = True

                t0 = time.time()
                best_acc, total_kept, logs = train(
                    net, inputs, perm, V,
                    max_attempts=MAX_ATT, ticks=8, log_interval=2000,
                    label=f"{mode}_{V}c_s{seed}",
                    is_global=is_global, is_per_neuron=is_pn,
                )
                elapsed = time.time() - t0

                result = {
                    'seed': seed,
                    'accuracy': best_acc,
                    'kept': total_kept,
                    'time': round(elapsed, 1),
                    'connections': net.count_connections(),
                }

                if is_global:
                    result['final_threshold'] = round(net.threshold, 4)
                    result['final_leak'] = round(net.leak, 4)
                    result['final_acc_rate'] = round(net.acc_rate, 4)
                    result['final_self_fb'] = round(net.self_fb, 4)

                if is_pn:
                    result['threshold_mean'] = round(float(np.mean(net.threshold)), 4)
                    result['leak_mean'] = round(float(np.mean(net.leak)), 4)

                seed_results.append(result)

                print(f"  DONE: {best_acc*100:.1f}% in {elapsed:.1f}s")

            task_results[mode] = seed_results

        all_results[f'{V}_class'] = task_results

        # Task summary
        print(f"\n{'='*70}")
        print(f"  {V}-CLASS SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Config':<20s} {'Mean Acc':>9s} {'Std':>7s} {'Best':>7s}")
        print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*7}")
        for mode, seed_results in task_results.items():
            accs = [r['accuracy'] for r in seed_results]
            print(f"  {mode:<20s} {np.mean(accs)*100:7.1f}% {np.std(accs)*100:6.1f}% "
                  f"{np.max(accs)*100:6.1f}%")

    return all_results


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_data = {}

    print("=" * 70)
    print("  VRAXION v22 — Learnable Capacitor Parameters via Evolution")
    print("  Can evolution find better params than manual sweep?")
    print("=" * 70)
    print()
    print("  Manual sweep optima (one-at-a-time):")
    print("    threshold → 0.3  (+25%)")
    print("    leak      → 0.95 (+18.7%)")
    print("    acc_rate  → 0.5  (+18.7%)")
    print("    self_fb   → 0.0  (+6.2%)")
    print()

    # === TEST A: Global learnable ===
    results_a = test_a_global_learnable()
    all_data['test_a'] = results_a

    # === TEST B: Per-neuron learnable ===
    results_b = test_b_per_neuron()
    all_data['test_b'] = results_b

    # Determine winner for Test C
    global_accs = [r['accuracy'] for r in results_b.get('global_learn', [])]
    pn_accs = [r['accuracy'] for r in results_b.get('per_neuron', [])]
    global_mean = np.mean(global_accs) if global_accs else 0
    pn_mean = np.mean(pn_accs) if pn_accs else 0

    winner = 'per_neuron' if pn_mean > global_mean else 'global_learn'
    print(f"\n  >>> WINNER for Test C: {winner} "
          f"(global={global_mean*100:.1f}% vs per_neuron={pn_mean*100:.1f}%)")

    # === TEST C: Scale ===
    results_c = test_c_scale(winner)
    all_data['test_c'] = results_c

    # === Final comparison table ===
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Config':<20s} {'16-class':>10s} {'32-class':>10s} {'64-class':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    # 16-class from Test A
    for cfg in ['fixed_default', 'fixed_sweep', 'global_learn']:
        accs_16 = [r['accuracy'] for r in results_a.get(cfg, [])]
        mean_16 = f"{np.mean(accs_16)*100:.1f}%" if accs_16 else "N/A"

        # 32/64 from Test C
        accs_32 = [r['accuracy'] for r in results_c.get('32_class', {}).get(cfg, [])]
        mean_32 = f"{np.mean(accs_32)*100:.1f}%" if accs_32 else "N/A"
        accs_64 = [r['accuracy'] for r in results_c.get('64_class', {}).get(cfg, [])]
        mean_64 = f"{np.mean(accs_64)*100:.1f}%" if accs_64 else "N/A"

        print(f"  {cfg:<20s} {mean_16:>10s} {mean_32:>10s} {mean_64:>10s}")

    # Per-neuron from Test B
    pn_16 = [r['accuracy'] for r in results_b.get('per_neuron', [])]
    mean_pn_16 = f"{np.mean(pn_16)*100:.1f}%" if pn_16 else "N/A"
    accs_pn_32 = [r['accuracy'] for r in results_c.get('32_class', {}).get('per_neuron', [])]
    mean_pn_32 = f"{np.mean(accs_pn_32)*100:.1f}%" if accs_pn_32 else "N/A"
    accs_pn_64 = [r['accuracy'] for r in results_c.get('64_class', {}).get('per_neuron', [])]
    mean_pn_64 = f"{np.mean(accs_pn_64)*100:.1f}%" if accs_pn_64 else "N/A"
    print(f"  {'per_neuron':<20s} {mean_pn_16:>10s} {mean_pn_32:>10s} {mean_pn_64:>10s}")

    # Save JSON
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_path = f"/home/user/VRAXION/v4/research/v22/tests/learnable_params_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=convert)
    print(f"\n  Results saved to: {json_path}")

    return all_data


if __name__ == "__main__":
    main()
