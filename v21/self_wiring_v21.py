"""
VRAXION v21 — Full Combo: Hill-Climbing Mutation + Self-Wiring + Reward Learning
=================================================================================
Combines ALL best-performing components:

1. Activation: binary threshold (clean signals for Hebbian learning)
2. Addresses: 3D+1D (spatial + functional type) — 2x faster than random
3. Self-wiring with target_W direction learning
4. Reward-modulated Hebbian weight learning (from v19b)
5. Hill-climbing mutation: try perturbation → evaluate → keep/revert (from v18)
6. Epoch-based training with per-sample updates
7. Connection cap prevents explosion

Scaling test: 32-class, 64-class, 128-class

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-11
"""

import numpy as np
import time
import random


# =============================================================================
# Network
# =============================================================================

class SelfWiringNetV21:
    """
    Full combo self-wiring graph network.
    leaky_relu + hill-climbing mutation + self-wiring + reward-modulated learning.
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
        self.conn_age = np.zeros((n_neurons, n_neurons), dtype=np.int32)
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self._sw_proposals = 0
        self._sw_accepts = 0
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
        """Forward pass with binary activation and sustained input."""
        self.state = np.zeros(self.n_neurons, dtype=np.float32)

        if self._effective_W_dirty:
            self._update_effective_W()
        eW = self._effective_W
        thresholds = self.thresholds
        self._tick_activations = []

        for tick in range(self.n_ticks):
            np.maximum(self.state[:self.n_in], input_vec, out=self.state[:self.n_in])
            activated = (self.state > thresholds).astype(np.float32)
            self._tick_activations.append(activated.copy())
            incoming = activated @ eW
            self.state = self.state * self.decay + incoming

        self._last_activations = (self.state > thresholds).astype(np.float32)
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
            activated = (states > thresholds[np.newaxis, :]).astype(np.float32)
            incoming = activated @ eW
            states = states * self.decay + incoming

        return states[:, -self.n_out:]

    def self_wire(self):
        """Self-wiring: active neurons propose connections via target_W."""
        n_conns = int(self.mask.sum())
        if n_conns >= self.max_connections:
            return []

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
                    self.W[i, j] = np.random.randn() * 0.15
                    self.conn_age[i, j] = 0
                    new_connections.append((i, j))
                    self._sw_accepts += 1
                    self._effective_W_dirty = True
                    break
                else:
                    dists[j] = np.inf

        return new_connections

    def reward_update(self, target_idx, predicted_idx, new_connections):
        """Reward-modulated weight update — binary activation signals."""
        correct = (target_idx == predicted_idx)
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

            # Deep trace-back
            if len(self._tick_activations) >= 2:
                late_act = self._tick_activations[-1]
                early_act = self._tick_activations[len(self._tick_activations) // 2]

                feeds_target = (late_act > 0).astype(np.float32) * self.mask[:, target_neuron]
                feeds_pred = (late_act > 0).astype(np.float32) * self.mask[:, pred_neuron]
                early_active = (early_act > 0).astype(np.float32)

                delta = np.outer(early_active, feeds_target - feeds_pred) * self.mask
                self.W += self.weight_lr * 0.1 * delta

            self._effective_W_dirty = True

        np.clip(self.W, -2.0, 2.0, out=self.W)
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

    def snapshot(self):
        """Save state for hill-climbing revert."""
        return {
            'W': self.W.copy(),
            'mask': self.mask.copy(),
            'thresholds': self.thresholds.copy(),
            'target_W': self.target_W.copy(),
            'conn_age': self.conn_age.copy(),
        }

    def restore(self, snap):
        """Restore from snapshot."""
        self.W = snap['W'].copy()
        self.mask = snap['mask'].copy()
        self.thresholds = snap['thresholds'].copy()
        self.target_W = snap['target_W'].copy()
        self.conn_age = snap['conn_age'].copy()
        self._effective_W_dirty = True

    def age_connections(self):
        self.conn_age += (self.mask > 0).astype(np.int32)


# =============================================================================
# Hill-Climbing Mutation
# =============================================================================

def hill_climb_mutation(net, input_batch, targets, current_acc,
                        n_mutations=10):
    """
    Try random mutations, keep if accuracy improves or stays same.
    Multi-scale: small precise perturbations + occasional big jumps.
    """
    n = net.n_neurons
    best_acc = current_acc
    improved = False

    active = np.argwhere(net.mask > 0)
    if len(active) == 0:
        return best_acc, improved

    n_active = len(active)

    for trial in range(n_mutations):
        snap = net.snapshot()

        # Alternate between different mutation types
        mut_type = trial % 4

        if mut_type == 0:
            # Small weight perturbation (fine-tuning)
            n_perturb = max(1, n_active // 20)
            chosen = active[np.random.choice(n_active, min(n_perturb, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.15

        elif mut_type == 1:
            # Medium weight perturbation
            n_perturb = max(1, n_active // 5)
            chosen = active[np.random.choice(n_active, min(n_perturb, n_active), replace=False)]
            for r, c in chosen:
                net.W[r, c] += np.random.randn() * 0.3

        elif mut_type == 2:
            # Large single-connection change (targeted exploration)
            idx = np.random.randint(n_active)
            r, c = active[idx]
            net.W[r, c] = np.random.randn() * 0.5  # Complete reset

        else:
            # Structure mutation: swap weak→new
            n_conns = int(net.mask.sum())
            if n_conns > 0:
                # Remove 1 weakest
                active_W = np.abs(net.W) * net.mask
                active_W[net.mask == 0] = np.inf
                flat = active_W.flatten()
                worst = flat.argmin()
                if flat[worst] < np.inf:
                    r, c = divmod(worst, n)
                    net.mask[r, c] = 0.0
                    net.W[r, c] = 0.0
                    net.conn_age[r, c] = 0

                # Add 1 new random
                for _ in range(10):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j and net.mask[i, j] == 0:
                        net.mask[i, j] = 1.0
                        net.W[i, j] = np.random.randn() * 0.3
                        net.conn_age[i, j] = 0
                        break

        np.clip(net.W, -2.0, 2.0, out=net.W)
        net.W *= net.mask
        net._effective_W_dirty = True

        acc = evaluate_accuracy_batch(net, input_batch, targets)
        if acc >= best_acc:
            best_acc = acc
            if acc > current_acc:
                improved = True
            # Update active list if structure changed
            if mut_type == 3:
                active = np.argwhere(net.mask > 0)
                n_active = len(active)
        else:
            net.restore(snap)

    return best_acc, improved


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
    print("VRAXION v21 — Full Combo: Hill-Climbing + Self-Wiring + Reward")
    print("=" * 70)
    print(f"Config: {n_classes} classes, {n_neurons} neurons, 8 ticks")
    print(f"Activation: binary threshold (learnable, init=0.3)")
    print(f"Addresses: 3D+1D (spatial + functional type)")
    print(f"Learning: reward-modulated Hebbian (normalized)")
    print(f"  weight_lr=0.02, threshold_lr=0.003, target_lr=0.01")
    print(f"Mutation: hill-climbing (10 multi-scale per epoch)")
    print(f"Self-wiring: top_k=5, address-directed")
    print(f"Connection cap: {int(6.0 * n_neurons)}")
    print(f"Solved: accuracy >= 100%")
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
    stale_limit = 25_000

    print(f"\nInitial accuracy: {acc*100:.1f}%")
    print(f"Initial connections: {net.mask.sum():.0f}")
    print()

    start_time = time.time()
    max_epochs = max_attempts // n_classes
    step = 0

    # SEQUENTIAL: Phase 1 = reward learning, Phase 2 = pure hill-climbing
    mode = 'reward'  # starts with reward learning
    reward_stale_limit = 15_000
    best_snap = net.snapshot()

    for epoch in range(1, max_epochs + 1):
        if mode == 'reward':
            # --- REWARD LEARNING PHASE ---
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

            # Evaluate
            acc = evaluate_accuracy_batch(net, input_batch, targets)

            if acc > best_acc:
                best_acc = acc
                best_acc_step = step
                stale_count = 0
                best_snap = net.snapshot()
            else:
                stale_count += n_classes

            # Switch to hill-climbing when reward learning plateaus
            if stale_count >= reward_stale_limit:
                elapsed = time.time() - start_time
                print(f"\n[SWITCH TO HILL-CLIMBING] epoch {epoch}, "
                      f"best={best_acc*100:.1f}%, time={elapsed:.1f}s")
                net.restore(best_snap)
                acc = best_acc
                mode = 'hillclimb'
                stale_count = 0
                stale_limit = 50_000

        else:
            # --- PURE HILL-CLIMBING PHASE ---
            step += n_classes  # count steps

            hc_acc, improved = hill_climb_mutation(
                net, input_batch, targets, acc,
                n_mutations=20,  # more mutations since no reward learning
            )

            if hc_acc > best_acc:
                best_acc = hc_acc
                best_acc_step = step
                stale_count = 0
                best_snap = net.snapshot()
            else:
                stale_count += n_classes

            acc = hc_acc

            # If stuck in HC, switch back to reward learning from best
            if stale_count >= stale_limit:
                elapsed = time.time() - start_time
                print(f"\n[SWITCH BACK TO REWARD] epoch {epoch}, "
                      f"best={best_acc*100:.1f}%, time={elapsed:.1f}s")
                net.restore(best_snap)
                # Small perturbation to explore new directions
                active = net.mask > 0
                noise = np.random.randn(*net.W.shape).astype(np.float32) * 0.05
                net.W += noise * active
                np.clip(net.W, -2.0, 2.0, out=net.W)
                net.W *= net.mask
                net._effective_W_dirty = True
                net.top_k_wire = min(net.top_k_wire + 1, 10)
                mode = 'reward'
                stale_count = 0

                acc = evaluate_accuracy_batch(net, input_batch, targets)

        # Log
        if epoch % (log_interval // n_classes + 1) == 0:
            elapsed = time.time() - start_time
            n_conns = int(net.mask.sum())
            print(f"[{mode:>8s} Epoch {epoch:>5d} Step {step:>7d}]  "
                  f"acc={acc*100:>5.1f}%  best={best_acc*100:.1f}% (@{best_acc_step})  "
                  f"conns={n_conns}  stale={stale_count}  time={elapsed:.1f}s")

        # SUCCESS
        if acc >= target_acc:
            elapsed = time.time() - start_time
            n_conns = int(net.mask.sum())
            print(f"\n{'='*70}")
            print(f"SUCCESS! 100% accuracy at epoch {epoch}, step {step}")
            print(f"Time: {elapsed:.1f}s  Connections: {n_conns}")
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

    elapsed = time.time() - start_time
    n_conns = int(net.mask.sum())
    print(f"\n{'='*70}")
    print(f"MAX ATTEMPTS ({max_attempts}). Best: {best_acc*100:.1f}% (@{best_acc_step})")
    print(f"Time: {elapsed:.1f}s  Connections: {n_conns}")
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
