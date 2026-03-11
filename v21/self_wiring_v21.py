"""
VRAXION v21 — Full Combo: Mutation + Self-Wiring + Reward-Modulated Target Learning
====================================================================================
Combines ALL best-performing components from v18 and v19b:

1. Activation: leaky_relu (best compromise at scale)
2. Addresses: 3D+1D (spatial + functional type) — 2x faster than random
3. Self-wiring with target_W direction learning
4. Reward-modulated Hebbian weight learning (from v19b)
5. Mutation: structure (add/remove connections) + weight perturbation (from v18)
6. Epoch-based training with per-sample updates
7. Connection cap: prevents explosion, forces quality over quantity

Scaling test: 32-class, 64-class, 128-class

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-11
"""

import numpy as np
import time
import random
from collections import defaultdict


# =============================================================================
# Network
# =============================================================================

class SelfWiringNetV21:
    """
    Full combo self-wiring graph network.
    leaky_relu activation + mutation + self-wiring + reward-modulated learning.
    """

    def __init__(self, n_neurons=160, n_in=32, n_out=32, n_ticks=8,
                 decay=0.5, top_k_wire=5, target_lr=0.01, threshold_lr=0.003,
                 weight_lr=0.02, leaky_slope=0.01, max_conn_ratio=6.0):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.decay = decay
        self.top_k_wire = top_k_wire
        self.target_lr = target_lr
        self.threshold_lr = threshold_lr
        self.weight_lr = weight_lr
        self.leaky_slope = leaky_slope
        # Max connections = ratio * n_neurons (prevents explosion)
        self.max_connections = int(max_conn_ratio * n_neurons)

        # Neuron properties
        self.addresses = self._init_3d_1d_addresses()
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.2
        self.thresholds = np.full(n_neurons, 0.3, dtype=np.float32)

        # Connection structure
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self._init_connections()
        self.W = (np.random.randn(n_neurons, n_neurons).astype(np.float32) * 0.3 + 0.2)
        self.W *= self.mask

        # Effective weights cache
        self._effective_W = self.W * self.mask
        self._effective_W_dirty = False

        # Tracking
        self.conn_origin = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.conn_age = np.zeros((n_neurons, n_neurons), dtype=np.int32)
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self._sw_proposals = 0
        self._sw_accepts = 0
        self._reward_positive = 0
        self._reward_negative = 0
        self._last_activations = np.zeros(n_neurons, dtype=np.float32)
        self._tick_activations = []

        # Best state snapshot for mutation
        self._best_W = self.W.copy()
        self._best_mask = self.mask.copy()
        self._best_acc = 0.0

    def _init_3d_1d_addresses(self):
        n = self.n_neurons
        addresses = np.zeros((n, 4), dtype=np.float32)
        addresses[:, :3] = np.random.rand(n, 3).astype(np.float32)
        for i in range(n):
            if i < self.n_in:
                addresses[i, 3] = 0.0
            elif i >= n - self.n_out:
                addresses[i, 3] = 1.0
            else:
                addresses[i, 3] = 0.5
        return addresses

    def _init_connections(self):
        n = self.n_neurons
        n_in = self.n_in
        n_out = self.n_out
        internal_start = n_in
        internal_end = n - n_out
        n_internal = internal_end - internal_start
        if n_internal == 0:
            return

        for i in range(n_in):
            k = min(3, n_internal)
            targets = random.sample(range(internal_start, internal_end), k)
            for j in targets:
                self.mask[i, j] = 1.0

        for i in range(internal_start, internal_end):
            others = [x for x in range(internal_start, internal_end) if x != i]
            if others:
                for j in random.sample(others, min(2, len(others))):
                    self.mask[i, j] = 1.0
            for _ in range(2):
                out_j = random.randint(n - n_out, n - 1)
                self.mask[i, out_j] = 1.0

        n_extra = int(n * n * 0.02)
        for _ in range(n_extra):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                self.mask[i, j] = 1.0

    def _update_effective_W(self):
        self._effective_W = self.W * self.mask
        self._effective_W_dirty = False

    def forward(self, input_vec):
        """Forward pass with leaky_relu activation and sustained input."""
        self.state = np.zeros(self.n_neurons, dtype=np.float32)

        if self._effective_W_dirty:
            self._update_effective_W()
        eW = self._effective_W
        self._tick_activations = []

        for tick in range(self.n_ticks):
            np.maximum(self.state[:self.n_in], input_vec, out=self.state[:self.n_in])
            # Leaky ReLU: continuous activation
            pre = self.state - self.thresholds
            activated = np.where(pre > 0, pre, self.leaky_slope * pre)
            self._tick_activations.append(activated.copy())
            incoming = activated @ eW
            self.state = self.state * self.decay + incoming

        pre = self.state - self.thresholds
        self._last_activations = np.where(pre > 0, pre, self.leaky_slope * pre)
        return self.state[-self.n_out:]

    def forward_batch(self, input_batch):
        """Batched forward pass for evaluation."""
        B = input_batch.shape[0]
        states = np.zeros((B, self.n_neurons), dtype=np.float32)

        if self._effective_W_dirty:
            self._update_effective_W()
        eW = self._effective_W
        thresholds = self.thresholds

        for tick in range(self.n_ticks):
            np.maximum(states[:, :self.n_in], input_batch,
                       out=states[:, :self.n_in])
            pre = states - thresholds[np.newaxis, :]
            activated = np.where(pre > 0, pre, self.leaky_slope * pre)
            incoming = activated @ eW
            states = states * self.decay + incoming

        return states[:, -self.n_out:]

    def self_wire(self):
        """Self-wiring: active neurons propose new connections based on target_W."""
        # Don't wire if already at cap
        n_conns = int(self.mask.sum())
        if n_conns >= self.max_connections:
            return []

        activated = self._last_activations
        active_mask = np.abs(activated) > 0.01
        n_active = active_mask.sum()
        if n_active == 0:
            return []

        active_states = np.abs(self.state) * active_mask
        k = min(self.top_k_wire, int(n_active))
        top_indices = np.argpartition(active_states, -k)[-k:]

        new_connections = []
        for i in top_indices:
            self._sw_proposals += 1
            target_addr = self.addresses[i] + np.abs(activated[i]) * self.target_W[i]
            dists = np.linalg.norm(self.addresses - target_addr[np.newaxis, :], axis=1)
            dists[i] = np.inf

            for _ in range(5):
                j = dists.argmin()
                if self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = np.random.randn() * 0.15
                    self.conn_origin[i, j] = 1.0
                    self.conn_age[i, j] = 0
                    new_connections.append((i, j))
                    self._sw_accepts += 1
                    self._effective_W_dirty = True
                    break
                else:
                    dists[j] = np.inf

        return new_connections

    def reward_update(self, target_idx, predicted_idx, new_connections):
        """Reward-modulated weight update — adapted for leaky_relu."""
        correct = (target_idx == predicted_idx)

        if correct:
            self._reward_positive += 1
        else:
            self._reward_negative += 1

        activated = self._last_activations
        n = self.n_neurons
        out_start = n - self.n_out

        if not correct:
            # Normalize activation to [0,1] for cleaner Hebbian signal
            act_abs = np.abs(activated)
            act_max = act_abs.max()
            if act_max > 0:
                active_vec = act_abs / act_max
            else:
                active_vec = act_abs

            target_neuron = out_start + target_idx
            pred_neuron = out_start + predicted_idx

            # Strengthen: active → target output
            self.W[:, target_neuron] += self.weight_lr * active_vec * self.mask[:, target_neuron]
            # Weaken: active → predicted output
            self.W[:, pred_neuron] -= self.weight_lr * active_vec * self.mask[:, pred_neuron]

            # Deep trace-back
            if len(self._tick_activations) >= 2:
                late_act = self._tick_activations[-1]
                early_act = self._tick_activations[len(self._tick_activations) // 2]

                late_abs = np.abs(late_act)
                late_max = late_abs.max()
                if late_max > 0:
                    late_norm = late_abs / late_max
                else:
                    late_norm = late_abs

                early_abs = np.abs(early_act)
                early_max = early_abs.max()
                if early_max > 0:
                    early_norm = early_abs / early_max
                else:
                    early_norm = early_abs

                feeds_target = late_norm * self.mask[:, target_neuron]
                feeds_pred = late_norm * self.mask[:, pred_neuron]

                delta = np.outer(early_norm, feeds_target - feeds_pred) * self.mask
                self.W += self.weight_lr * 0.1 * delta

            self._effective_W_dirty = True

        # Clamp weights
        np.clip(self.W, -2.0, 2.0, out=self.W)
        self.W *= self.mask
        self._effective_W_dirty = True

        # Target_W update
        reward = 1.0 if correct else -1.0
        for (i, j) in new_connections:
            direction = self.addresses[j] - self.addresses[i]
            norm = np.linalg.norm(direction) + 1e-8
            direction /= norm
            self.target_W[i] += self.target_lr * np.abs(activated[i]) * direction * reward

        # Threshold neuromodulation
        active_mask = np.abs(activated) > 0.01
        if active_mask.any():
            if correct:
                self.thresholds[active_mask] -= self.threshold_lr
            else:
                self.thresholds[active_mask] += self.threshold_lr * 0.5
            np.clip(self.thresholds, 0.01, 5.0, out=self.thresholds)

    # =========================================================================
    # Mutation (from v18)
    # =========================================================================

    def mutate_structure(self, n_add=3, n_remove=3):
        """Structural mutation: add random, remove weakest."""
        n = self.n_neurons
        n_conns = int(self.mask.sum())

        # Add (only if below cap)
        added = 0
        if n_conns < self.max_connections:
            for _ in range(n_add * 3):
                if added >= n_add:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j and self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = np.random.randn() * 0.15
                    self.conn_age[i, j] = 0
                    added += 1

        # Remove weakest (keep minimum)
        n_conns = int(self.mask.sum())
        min_conns = max(self.n_in + self.n_out, n // 2)
        n_to_remove = min(n_remove, n_conns - min_conns)

        if n_to_remove > 0:
            active_W = np.abs(self.W) * self.mask
            # Protect young connections
            young = self.conn_age < 30
            active_W[young & (self.mask > 0)] = np.inf
            active_W[self.mask == 0] = np.inf

            flat = active_W.flatten()
            remove_idx = np.argpartition(flat, n_to_remove)[:n_to_remove]
            for idx in remove_idx:
                if flat[idx] < np.inf:
                    r, c = divmod(idx, n)
                    self.mask[r, c] = 0.0
                    self.W[r, c] = 0.0
                    self.conn_origin[r, c] = 0.0
                    self.conn_age[r, c] = 0

        self._effective_W_dirty = True

    def mutate_weights(self, rate=0.08, scale=0.15):
        """Weight mutation: random perturbations to a fraction of weights."""
        active = self.mask > 0
        n_active = int(active.sum())
        if n_active == 0:
            return

        n_mutate = max(1, int(n_active * rate))
        active_indices = np.argwhere(active)
        chosen = active_indices[np.random.choice(len(active_indices),
                                min(n_mutate, len(active_indices)), replace=False)]

        for r, c in chosen:
            self.W[r, c] += np.random.randn() * scale

        np.clip(self.W, -2.0, 2.0, out=self.W)
        self.W *= self.mask
        self._effective_W_dirty = True

    def save_best(self, acc):
        """Save best state for rollback on catastrophic forgetting."""
        if acc > self._best_acc:
            self._best_acc = acc
            self._best_W = self.W.copy()
            self._best_mask = self.mask.copy()

    def rollback_to_best(self):
        """Rollback to best known state (anti-catastrophic-forgetting)."""
        self.W = self._best_W.copy()
        self.mask = self._best_mask.copy()
        self._effective_W_dirty = True

    def age_connections(self):
        self.conn_age += (self.mask > 0).astype(np.int32)

    def prune_over_cap(self):
        """Hard prune to stay under connection cap."""
        n_conns = int(self.mask.sum())
        if n_conns <= self.max_connections:
            return 0

        n_to_remove = n_conns - self.max_connections
        active_W = np.abs(self.W) * self.mask
        active_W[self.mask == 0] = np.inf

        flat = active_W.flatten()
        remove_idx = np.argpartition(flat, n_to_remove)[:n_to_remove]
        removed = 0
        for idx in remove_idx:
            if flat[idx] < np.inf:
                r, c = divmod(idx, self.n_neurons)
                self.mask[r, c] = 0.0
                self.W[r, c] = 0.0
                self.conn_origin[r, c] = 0.0
                self.conn_age[r, c] = 0
                removed += 1

        self._effective_W_dirty = True
        return removed


# =============================================================================
# Task
# =============================================================================

def make_association_task(n_classes=32):
    indices = list(range(n_classes))
    targets = indices.copy()
    random.shuffle(targets)
    input_batch = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        input_batch[i, indices[i]] = 1.0
    return input_batch, targets, dict(zip(indices, targets))


def evaluate_accuracy_batch(net, input_batch, targets):
    outputs = net.forward_batch(input_batch)
    predictions = outputs.argmax(axis=1)
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


# =============================================================================
# Training — v21 Full Combo
# =============================================================================

def train_v21(n_classes=32, n_neurons=160, max_attempts=200_000,
              log_interval=500, target_acc=1.0):
    print("=" * 70)
    print("VRAXION v21 — Full Combo: Mutation + Self-Wiring + Reward Learning")
    print("=" * 70)
    print(f"Config: {n_classes} classes, {n_neurons} neurons, 8 ticks")
    print(f"Activation: leaky_relu (slope=0.01)")
    print(f"Addresses: 3D+1D (spatial + functional type)")
    print(f"Learning: reward-modulated Hebbian (normalized activations)")
    print(f"  target_lr=0.01, weight_lr=0.02, threshold_lr=0.003")
    print(f"Mutation: structure (add=3, remove=3) + weight (rate=8%, scale=0.15)")
    print(f"Self-wiring: top_k=5, address-directed")
    print(f"Connection cap: {int(6.0 * n_neurons)}")
    print(f"Anti-forgetting: rollback to best on stale-out")
    print(f"Solved: accuracy >= 100%")
    print(f"Backend: numpy")
    print("=" * 70)

    net = SelfWiringNetV21(
        n_neurons=n_neurons, n_in=n_classes, n_out=n_classes,
        n_ticks=8, decay=0.5, top_k_wire=5,
        target_lr=0.01, threshold_lr=0.003, weight_lr=0.02,
        leaky_slope=0.01, max_conn_ratio=6.0,
    )

    input_batch, targets, mapping = make_association_task(n_classes)
    pairs = [(input_batch[i], targets[i]) for i in range(n_classes)]

    print(f"\nMapping (first 8): {dict(list(mapping.items())[:8])}")

    acc = evaluate_accuracy_batch(net, input_batch, targets)
    best_acc = acc
    best_acc_step = 0
    stale_count = 0
    stale_limit = 20_000
    rollback_count = 0

    print(f"\nInitial accuracy: {acc*100:.1f}%")
    print(f"Initial connections: {net.mask.sum():.0f}")
    print(f"\nStarting training...\n")

    start_time = time.time()
    max_epochs = max_attempts // n_classes
    step = 0

    for epoch in range(1, max_epochs + 1):
        order = list(range(n_classes))
        random.shuffle(order)

        for idx in order:
            step += 1
            inp, target = pairs[idx]

            output = net.forward(inp)
            predicted = int(output.argmax())

            new_conns = []
            if predicted != target:
                new_conns = net.self_wire()

            net.reward_update(target, predicted, new_conns)

        net.age_connections()

        # Mutation every epoch
        net.mutate_structure(n_add=3, n_remove=3)
        net.mutate_weights(rate=0.08, scale=0.15)

        # Enforce connection cap
        net.prune_over_cap()

        # Evaluate
        acc = evaluate_accuracy_batch(net, input_batch, targets)
        net.save_best(acc)

        if acc > best_acc:
            best_acc = acc
            best_acc_step = step
            stale_count = 0
        else:
            stale_count += n_classes

        # Log
        if epoch % (log_interval // n_classes + 1) == 0 or acc > best_acc - 0.01:
            elapsed = time.time() - start_time
            n_conns = int(net.mask.sum())
            print(f"[Epoch {epoch:>5d} Step {step:>7d}]  acc={acc*100:>5.1f}%  "
                  f"best={best_acc*100:.1f}% (@{best_acc_step})  "
                  f"conns={n_conns}  stale={stale_count}  time={elapsed:.1f}s")

        # SUCCESS
        if acc >= target_acc:
            elapsed = time.time() - start_time
            n_conns = int(net.mask.sum())
            print(f"\n{'='*70}")
            print(f"SUCCESS! 100% accuracy at epoch {epoch}, step {step}")
            print(f"Time: {elapsed:.1f}s")
            print(f"Connections: {n_conns}")
            print(f"Rollbacks: {rollback_count}")
            print(f"{'='*70}")
            return {
                'solved': True,
                'epoch': epoch,
                'step': step,
                'time': elapsed,
                'connections': n_conns,
                'best_acc': best_acc,
                'n_classes': n_classes,
                'n_neurons': n_neurons,
            }

        # Stale-out: rollback + perturbation
        if stale_count >= stale_limit:
            elapsed = time.time() - start_time
            print(f"\n[STALE-OUT epoch {epoch}] best={best_acc*100:.1f}% "
                  f"(@{best_acc_step}) time={elapsed:.1f}s")

            # Rollback to best known state
            net.rollback_to_best()
            rollback_count += 1

            # Perturbation to escape local minimum
            net.top_k_wire = min(net.top_k_wire + 1, 10)
            net.target_W += np.random.randn(*net.target_W.shape).astype(np.float32) * 0.05
            net.mutate_weights(rate=0.15, scale=0.2)
            net.mutate_structure(n_add=5, n_remove=2)

            print(f"  Rollback #{rollback_count}, perturbation, top_k={net.top_k_wire}")
            stale_count = 0

    elapsed = time.time() - start_time
    n_conns = int(net.mask.sum())
    print(f"\n{'='*70}")
    print(f"MAX ATTEMPTS ({max_attempts}). Best: {best_acc*100:.1f}% (@{best_acc_step})")
    print(f"Time: {elapsed:.1f}s  Connections: {n_conns}  Rollbacks: {rollback_count}")
    print(f"{'='*70}")
    return {
        'solved': False,
        'epoch': epoch,
        'step': step,
        'time': elapsed,
        'connections': n_conns,
        'best_acc': best_acc,
        'n_classes': n_classes,
        'n_neurons': n_neurons,
    }


# =============================================================================
# Scaling Test
# =============================================================================

def run_scaling_test():
    print("\n" + "=" * 70)
    print("VRAXION v21 — SCALING TEST")
    print("=" * 70)

    configs = [
        {'n_classes': 32,  'n_neurons': 160,  'max_attempts': 200_000},
        {'n_classes': 64,  'n_neurons': 320,  'max_attempts': 400_000},
        {'n_classes': 128, 'n_neurons': 640,  'max_attempts': 800_000},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'#'*70}")
        print(f"# SCALING TEST: {cfg['n_classes']}-class, {cfg['n_neurons']} neurons")
        print(f"{'#'*70}\n")

        np.random.seed(42)
        random.seed(42)

        result = train_v21(
            n_classes=cfg['n_classes'],
            n_neurons=cfg['n_neurons'],
            max_attempts=cfg['max_attempts'],
            log_interval=500,
            target_acc=1.0,
        )
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("SCALING TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Classes':>8} {'Neurons':>8} {'Solved':>8} {'Best Acc':>10} "
          f"{'Steps':>10} {'Time(s)':>10} {'Conns':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_classes']:>8} {r['n_neurons']:>8} "
              f"{'YES' if r['solved'] else 'NO':>8} "
              f"{r['best_acc']*100:>9.1f}% "
              f"{r['step']:>10} "
              f"{r['time']:>10.1f} "
              f"{r['connections']:>8}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    run_scaling_test()
