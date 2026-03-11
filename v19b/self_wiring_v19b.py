"""
VRAXION v19b — Self-Wiring Graph Network with Reward-Modulated Target Learning
===============================================================================
NO mutation. NO backprop. Pure self-wiring + reward-modulated learning.

Key features:
- Binary activation with per-neuron learnable threshold (neuromodulation)
- 3D+1D address initialization (spatial + functional type)
- Per-sample per-output reward-modulated Hebbian/anti-Hebbian weight learning
- Three-factor Hebbian target_W learning for self-wiring direction
- Deep trace-back credit assignment through tick activations
- Threshold neuromodulation (active neurons adjust sensitivity based on reward)
- Sustained input (re-injected every tick)
- Pure numpy for speed (torch CPU BLAS is broken on some systems)

32-class A→B association task, 160 neurons.

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-11
"""

import numpy as np
import time
import math
import random
from collections import defaultdict


# =============================================================================
# Network Monitor
# =============================================================================

class NetworkMonitor:
    def __init__(self, log_interval=500):
        self.log_interval = log_interval
        self.history = defaultdict(list)
        self.step = 0

    def log(self, net, extra=None):
        self.step += 1
        if self.step % self.log_interval != 0:
            return

        mask = net.mask
        n = net.n_neurons
        n_conns = mask.sum()
        density = n_conns / (n * n)

        in_deg = mask.sum(axis=0)
        out_deg = mask.sum(axis=1)
        max_in = in_deg.max()
        max_out = out_deg.max()
        avg_deg = n_conns / n if n > 0 else 0
        hubs_in = (in_deg > 2 * avg_deg).sum()
        hubs_out = (out_deg > 2 * avg_deg).sum()

        sw_conns = net.conn_origin.sum()
        total = n_conns

        ages = net.conn_age[mask > 0]
        if len(ages) > 0:
            mean_age = ages.mean()
            max_age = ages.max()
            young = (ages < 100).sum()
            old = (ages >= 100).sum()
        else:
            mean_age = max_age = young = old = 0

        activated = net._last_activations
        active_frac = (activated > 0).mean()

        addr_range = net.addresses.max(axis=0) - net.addresses.min(axis=0)
        target_mag = np.abs(net.target_W).mean()

        thresh_mean = net.thresholds.mean()
        thresh_std = net.thresholds.std()
        thresh_min = net.thresholds.min()
        thresh_max = net.thresholds.max()

        accept_rate = net._sw_accepts / max(net._sw_proposals, 1)

        active_W = net.W[mask > 0]
        if len(active_W) > 0:
            w_mean = active_W.mean()
            w_std = active_W.std()
            w_pos = (active_W > 0).sum()
            w_neg = (active_W < 0).sum()
        else:
            w_mean = w_std = 0
            w_pos = w_neg = 0

        print(f"\n{'='*70}")
        print(f"[MONITOR] Step {self.step}")
        print(f"{'='*70}")
        print(f"[GRAPH]  conns={n_conns}  density={density:.4f}  "
              f"max_in={max_in}  max_out={max_out}  "
              f"hubs_in={hubs_in}  hubs_out={hubs_out}")
        print(f"[ORIGIN] self-wired={sw_conns:.0f}/{total:.0f} "
              f"({sw_conns/max(total,1)*100:.1f}%)")
        print(f"[WIRE]   proposals={net._sw_proposals}  accepts={net._sw_accepts}  "
              f"rate={accept_rate:.3f}")
        print(f"[AGE]    mean={mean_age:.1f}  max={max_age:.0f}  "
              f"young(<100)={young}  old(>=100)={old}")
        print(f"[ACTIV]  sparsity={1-active_frac:.3f}  "
              f"active_frac={active_frac:.3f}")
        print(f"[ADDR]   range=[{addr_range[0]:.2f},{addr_range[1]:.2f},"
              f"{addr_range[2]:.2f},{addr_range[3]:.2f}]  "
              f"target_mag={target_mag:.4f}")
        print(f"[THRESH] mean={thresh_mean:.3f}  std={thresh_std:.3f}  "
              f"min={thresh_min:.3f}  max={thresh_max:.3f}")
        print(f"[WEIGHT] mean={w_mean:.3f}  std={w_std:.3f}  "
              f"pos={w_pos}  neg={w_neg}")
        print(f"[REWARD] positive={net._reward_positive}  negative={net._reward_negative}  "
              f"neutral={net._reward_neutral}")
        if extra:
            for k, v in extra.items():
                print(f"[EXTRA]  {k}={v}")
        print(f"{'='*70}")

        self.history['step'].append(self.step)
        self.history['n_conns'].append(n_conns)
        self.history['density'].append(density)
        self.history['active_frac'].append(active_frac)
        self.history['target_mag'].append(target_mag)
        self.history['thresh_mean'].append(thresh_mean)


# =============================================================================
# v19b Self-Wiring Graph Network (numpy)
# =============================================================================

class SelfWiringNetV19b:
    """
    Self-wiring graph network with reward-modulated target learning.
    Pure numpy for speed. NO mutation, NO backprop.
    """

    def __init__(self, n_neurons=160, n_in=32, n_out=32, n_ticks=8,
                 decay=0.5, top_k_wire=5, target_lr=0.01, threshold_lr=0.003,
                 weight_lr=0.015):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.decay = decay
        self.top_k_wire = top_k_wire
        self.target_lr = target_lr
        self.threshold_lr = threshold_lr
        self.weight_lr = weight_lr

        # Neuron properties
        self.addresses = self._init_3d_1d_addresses()
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.2
        self.thresholds = np.full(n_neurons, 0.3, dtype=np.float32)

        # Connection structure
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self._init_connections()
        self.W = (np.random.randn(n_neurons, n_neurons).astype(np.float32) * 0.3 + 0.2)
        self.W *= self.mask

        # Precomputed effective weights (updated when mask/W changes)
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
        self._reward_neutral = 0
        self._last_activations = np.zeros(n_neurons, dtype=np.float32)
        self._tick_activations = []

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

        # Input → 3 random internals
        for i in range(n_in):
            k = min(3, n_internal)
            targets = random.sample(range(internal_start, internal_end), k)
            for j in targets:
                self.mask[i, j] = 1.0

        # Internal → 2 other internals + 2 outputs
        for i in range(internal_start, internal_end):
            others = [x for x in range(internal_start, internal_end) if x != i]
            if others:
                for j in random.sample(others, min(2, len(others))):
                    self.mask[i, j] = 1.0
            for _ in range(2):
                out_j = random.randint(n - n_out, n - 1)
                self.mask[i, out_j] = 1.0

        # 2% random
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
        """Forward pass with sustained input injection."""
        self.state = np.zeros(self.n_neurons, dtype=np.float32)

        if self._effective_W_dirty:
            self._update_effective_W()
        eW = self._effective_W
        thresholds = self.thresholds
        self._tick_activations = []

        for tick in range(self.n_ticks):
            # Sustained input
            np.maximum(self.state[:self.n_in], input_vec, out=self.state[:self.n_in])

            activated = (self.state > thresholds).astype(np.float32)
            self._tick_activations.append(activated.copy())
            incoming = activated @ eW
            self.state = self.state * self.decay + incoming

        self._last_activations = (self.state > thresholds).astype(np.float32)
        return self.state[-self.n_out:]

    def forward_batch(self, input_batch):
        """Batched forward pass for evaluation. input_batch: (B, n_in)"""
        B = input_batch.shape[0]
        states = np.zeros((B, self.n_neurons), dtype=np.float32)

        if self._effective_W_dirty:
            self._update_effective_W()
        eW = self._effective_W
        thresholds = self.thresholds

        for tick in range(self.n_ticks):
            np.maximum(states[:, :self.n_in], input_batch,
                       out=states[:, :self.n_in])
            activated = (states > thresholds[np.newaxis, :]).astype(np.float32)
            incoming = activated @ eW
            states = states * self.decay + incoming

        return states[:, -self.n_out:]

    def self_wire(self):
        activated = self._last_activations
        active_mask = activated > 0
        n_active = active_mask.sum()
        if n_active == 0:
            return []

        active_states = np.abs(self.state) * active_mask
        k = min(self.top_k_wire, int(n_active))
        top_indices = np.argpartition(active_states, -k)[-k:]

        new_connections = []
        for i in top_indices:
            self._sw_proposals += 1
            target_addr = self.addresses[i] + activated[i] * self.target_W[i]
            dists = np.linalg.norm(self.addresses - target_addr[np.newaxis, :], axis=1)
            dists[i] = np.inf

            for _ in range(5):
                j = dists.argmin()
                if self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = np.random.randn() * 0.1
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
        correct = (target_idx == predicted_idx)

        if correct:
            self._reward_positive += 1
        else:
            self._reward_negative += 1

        activated = self._last_activations
        n = self.n_neurons
        out_start = n - self.n_out

        if not correct:
            active_vec = (activated > 0).astype(np.float32)
            target_neuron = out_start + target_idx
            pred_neuron = out_start + predicted_idx

            # Strengthen: active → target output
            self.W[:, target_neuron] += self.weight_lr * active_vec * self.mask[:, target_neuron]
            # Weaken: active → predicted output
            self.W[:, pred_neuron] -= self.weight_lr * active_vec * self.mask[:, pred_neuron]

            # Deep trace-back (vectorized)
            if len(self._tick_activations) >= 2:
                late_act = self._tick_activations[-1]
                early_act = self._tick_activations[len(self._tick_activations) // 2]

                feeds_target = (late_act > 0).astype(np.float32) * self.mask[:, target_neuron]
                feeds_pred = (late_act > 0).astype(np.float32) * self.mask[:, pred_neuron]
                early_active = (early_act > 0).astype(np.float32)

                delta = np.outer(early_active, feeds_target - feeds_pred) * self.mask
                self.W += self.weight_lr * 0.1 * delta

            self._effective_W_dirty = True

        # Clamp weights
        np.clip(self.W, -2.0, 2.0, out=self.W)
        # Anti-Hebbian pruning: connections pushed near zero by conflicting
        # updates are dead — remove them (the network "decides" to disconnect)
        near_zero = (np.abs(self.W) < 0.01) & (self.mask > 0)
        if near_zero.any():
            self.mask[near_zero] = 0.0
            self.W[near_zero] = 0.0
            self.conn_origin[near_zero] = 0.0
            self.conn_age[near_zero] = 0
        self.W *= self.mask
        self._effective_W_dirty = True

        # Target_W update
        reward = 1.0 if correct else -1.0
        for (i, j) in new_connections:
            direction = self.addresses[j] - self.addresses[i]
            norm = np.linalg.norm(direction) + 1e-8
            direction /= norm
            self.target_W[i] += self.target_lr * activated[i] * direction * reward

        # Threshold neuromodulation
        active_mask = activated > 0
        if active_mask.any():
            if correct:
                self.thresholds[active_mask] -= self.threshold_lr
            else:
                self.thresholds[active_mask] += self.threshold_lr * 0.5
            np.clip(self.thresholds, 0.01, 5.0, out=self.thresholds)

    def remove_connection(self, i, j):
        self.mask[i, j] = 0.0
        self.W[i, j] = 0.0
        self.conn_origin[i, j] = 0.0
        self.conn_age[i, j] = 0
        self._effective_W_dirty = True

    def prune_dead_connections(self, max_age=10000, min_conns=100):
        n_conns = self.mask.sum()
        if n_conns <= min_conns:
            return 0
        old = (self.conn_age > max_age) & (self.mask > 0)
        n_pruned = min(int(old.sum()), int(n_conns - min_conns))
        if n_pruned > 0:
            old_ages = self.conn_age.copy()
            old_ages[~old] = 0
            flat = old_ages.flatten()
            prune_idx = np.argpartition(flat, -n_pruned)[-n_pruned:]
            rows = prune_idx // self.n_neurons
            cols = prune_idx % self.n_neurons
            for r, c in zip(rows, cols):
                self.remove_connection(r, c)
        return n_pruned

    def age_connections(self):
        self.conn_age += (self.mask > 0).astype(np.int32)


# =============================================================================
# Task
# =============================================================================

def make_association_task(n_classes=32):
    indices = list(range(n_classes))
    targets = indices.copy()
    random.shuffle(targets)
    # Build batch input matrix (n_classes, n_classes) — one-hot rows
    input_batch = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        input_batch[i, indices[i]] = 1.0
    return input_batch, targets, dict(zip(indices, targets))


def evaluate_accuracy_batch(net, input_batch, targets):
    """Batched evaluation — single forward pass for all samples."""
    outputs = net.forward_batch(input_batch)
    predictions = outputs.argmax(axis=1)
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


# =============================================================================
# Training
# =============================================================================

def train_v19b(n_classes=32, n_neurons=160, max_attempts=200_000,
               log_interval=500, target_acc=1.0):
    print("=" * 70)
    print("VRAXION v19b — Self-Wiring with Reward-Modulated Target Learning")
    print("=" * 70)
    print(f"Config: {n_classes} classes, {n_neurons} neurons, 8 ticks")
    print(f"Activation: binary, learnable threshold (init=0.3, [0.01, 5.0])")
    print(f"Addresses: 3D+1D (spatial + functional type)")
    print(f"Learning: per-sample per-output reward-modulated Hebbian")
    print(f"  target_W lr=0.01, weight lr=0.008, threshold lr=0.001")
    print(f"  Deep trace-back: 0.1 × weight_lr on intermediate connections")
    print(f"  Epoch-based: all samples per epoch, reduced interference")
    print(f"Mutation: NONE")
    print(f"Backend: numpy (pure CPU)")
    print("=" * 70)

    net = SelfWiringNetV19b(
        n_neurons=n_neurons, n_in=n_classes, n_out=n_classes,
        n_ticks=8, decay=0.5, top_k_wire=5,
        target_lr=0.01, threshold_lr=0.001, weight_lr=0.008,
    )

    input_batch, targets, mapping = make_association_task(n_classes)
    # Individual pairs for per-sample training
    pairs = [(input_batch[i], targets[i]) for i in range(n_classes)]

    print(f"\nMapping (first 8): {dict(list(mapping.items())[:8])}")

    monitor = NetworkMonitor(log_interval=log_interval)

    acc = evaluate_accuracy_batch(net, input_batch, targets)
    best_acc = acc
    best_acc_step = 0
    stale_count = 0
    stale_limit = 30_000

    print(f"\nInitial accuracy: {acc*100:.1f}%")
    print(f"Initial connections: {net.mask.sum():.0f}")
    print(f"\nStarting per-sample reward training...\n")

    start_time = time.time()
    max_epochs = max_attempts // n_classes
    step = 0

    for epoch in range(1, max_epochs + 1):
        # Shuffle sample order each epoch (reduces ordering bias)
        order = list(range(n_classes))
        random.shuffle(order)

        for idx in order:
            step += 1
            inp, target = pairs[idx]

            # Forward pass
            output = net.forward(inp)
            predicted = int(output.argmax())

            # Self-wire (only on wrong predictions — focus wiring on failures)
            new_conns = []
            if predicted != target:
                new_conns = net.self_wire()

            # Reward-modulated learning
            net.reward_update(target, predicted, new_conns)

        # Age connections once per epoch (not every sample)
        net.age_connections()

        # Evaluate accuracy after each epoch
        acc = evaluate_accuracy_batch(net, input_batch, targets)
        if acc > best_acc:
            best_acc = acc
            best_acc_step = step
            stale_count = 0
        else:
            stale_count += n_classes

        # Logging
        if epoch % (log_interval // n_classes + 1) == 0 or acc > best_acc - 0.01:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch:>5d} Step {step:>7d}]  acc={acc*100:>5.1f}%  "
                  f"best={best_acc*100:.1f}% (@{best_acc_step})  "
                  f"conns={net.mask.sum():.0f}  "
                  f"stale={stale_count}  "
                  f"time={elapsed:.1f}s")

            monitor.log(net, extra={
                'accuracy': f'{acc*100:.1f}%',
                'best_accuracy': f'{best_acc*100:.1f}%',
            })

        # Success
        if acc >= target_acc:
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"SUCCESS! 100% accuracy at epoch {epoch}, step {step} ({elapsed:.1f}s)")
            print(f"Connections: {net.mask.sum():.0f} "
                  f"(self-wired: {net.conn_origin.sum():.0f})")
            print(f"Threshold: [{net.thresholds.min():.3f}, {net.thresholds.max():.3f}]")
            print(f"Target_W mag: {np.abs(net.target_W).mean():.4f}")
            print(f"{'='*70}")
            return net, monitor

        # Stale-out recovery
        if stale_count >= stale_limit:
            elapsed = time.time() - start_time
            print(f"\n[STALE-OUT epoch {epoch} step {step}] best={best_acc*100:.1f}% "
                  f"(@{best_acc_step}) time={elapsed:.1f}s")
            net.top_k_wire = min(net.top_k_wire + 2, 12)
            net.target_W += np.random.randn(*net.target_W.shape).astype(np.float32) * 0.05
            net.thresholds += np.random.randn(*net.thresholds.shape).astype(np.float32) * 0.02
            np.clip(net.thresholds, 0.01, 5.0, out=net.thresholds)
            print(f"  Recovery: top_k={net.top_k_wire}, target+threshold noise")
            stale_count = 0
            stale_limit += 10_000

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"MAX ATTEMPTS ({max_attempts}). Best: {best_acc*100:.1f}% (@{best_acc_step})")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*70}")
    return net, monitor


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    print("VRAXION v19b — numpy backend")
    net, monitor = train_v19b(
        n_classes=32, n_neurons=160, max_attempts=200_000,
        log_interval=500, target_acc=1.0,
    )
