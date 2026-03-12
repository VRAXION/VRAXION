"""
VRAXION v22 — Best Configuration
==================================
Self-Wiring Graph Network — all empirically validated choices.

Architecture:
  - Flat graph (no layers, no hierarchy)
  - Ternary mask (-1/0/+1) with flip mutation (30%)
  - Binary weights (0.5/1.5) — positive only, sign in mask
  - Leaky ReLU activation (continuous, like membrane potential)
  - Shared I/O (first V neurons = input + output)
  - First tick only input injection
  - Persistent state with decay 0.5

Learning:
  - Mutation + selection (keep/revert)
  - Structure phase → Both phase transition
  - Flip mutation is the most powerful operator
  - Self-wiring with inverse arousal (optional)

Results (16-class lookup):
  - 87.5% accuracy
  - 890 connections
  - 3 bits per connection (2 mask + 1 weight)

Proven on real English text: 28% bigram prediction (7.8x random)
"""

import numpy as np
import math
import random


class SelfWiringGraph:
    """The best-of-all-tests architecture."""

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30):
        self.N = n_neurons
        self.V = vocab
        self.flip_rate = flip_rate
        self.last_acc = 0.0

        # Ternary mask: -1 (inhibit), 0 (no connection), +1 (excite)
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Binary weights: 0.5 (weak) or 1.5 (strong), positive only
        self.W = np.where(
            np.random.rand(n_neurons, n_neurons) > 0.5,
            np.float32(0.5), np.float32(1.5)
        )

        # 4D addresses for self-wiring (3D spatial + 1D functional)
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.addr[:vocab, 3] = 0.0        # I/O neurons: functional = 0
        self.addr[vocab:, 3] = 0.5        # Internal: functional = 0.5
        self.target_W = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1

        # Persistent state (implicit memory via decay)
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

    def reset(self):
        self.state *= 0

    def forward(self, world, ticks=6, mode='capacitor'):
        """
        Forward pass with selectable neuron model.
        mode='leaky_relu': original continuous activation
        mode='capacitor': integrate-and-fire (biological capacitor)
        """
        if mode == 'leaky_relu':
            return self._forward_leaky_relu(world, ticks)
        else:
            return self._forward_capacitor(world, ticks)

    def _forward_leaky_relu(self, world, ticks=6):
        """Original leaky ReLU forward pass."""
        act = self.state.copy()
        Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)
            np.clip(act, -10.0, 10.0, out=act)
        self.state = act.copy()
        return act[:self.V]

    def _forward_capacitor(self, world, ticks=6):
        """Integrate-and-fire capacitor model.
        - charge accumulates from incoming signals
        - neuron fires (emits spike) when charge > threshold
        - after firing: charge resets to 0 (refractory)
        - charge leaks each tick (decay)
        This naturally creates sparse activation patterns."""
        charge = self.state.copy()  # accumulated charge per neuron
        Weff = self.W * self.mask
        threshold = 0.5  # fire threshold (sweet spot from config sweep)
        spike_strength = 1.0  # output magnitude when firing
        leak = 0.85  # charge retention per tick (sweet spot from config sweep)

        # Accumulate output reads across ticks (temporal code)
        output_acc = np.zeros(self.V, dtype=np.float32)

        for t in range(ticks):
            # Leak charge (capacitor discharge)
            charge *= leak

            # First tick: inject input current
            if t == 0:
                charge[:self.V] += world * 2.0  # strong input current

            # Incoming current from connected neurons that fired
            spikes = (charge > threshold).astype(np.float32) * spike_strength
            current = spikes @ Weff  # weighted sum of incoming spikes
            charge += current * 0.3  # integration rate

            # Fire and reset
            fired = charge > threshold
            charge[fired] = 0.0  # reset after firing (refractory)

            # Accumulate output (temporal integration)
            output_acc += charge[:self.V]

        self.state = charge.copy()
        return output_acc  # accumulated output over all ticks

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def memory_bytes(self):
        """Total model size in bytes (2 bit mask + 1 bit weight per connection)"""
        return self.count_connections() * 3 // 8

    # === State management ===

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.target_W.copy())

    def restore_state(self, s):
        self.W = s[0].copy()
        self.mask = s[1].copy()
        self.state = s[2].copy()
        self.addr = s[3].copy()
        self.target_W = s[4].copy()

    # === Mutation operators ===

    def mutate_diff_guided(self, diff, rate=0.05):
        """Diff-guided mutation: target connections of the worst output neuron.
        diff: array of size V, signed error per output neuron.
        Positive diff = output too high, negative = output too low."""
        worst_idx = int(np.argmax(np.abs(diff)))
        worst_sign = diff[worst_idx]  # negative = needs more excitation, positive = needs inhibition

        # Find connections INTO the worst output neuron
        conns_to_worst = np.argwhere(self.mask[:, worst_idx] != 0).flatten()
        dead_to_worst = np.argwhere(self.mask[:, worst_idx] == 0).flatten()
        dead_to_worst = dead_to_worst[dead_to_worst != worst_idx]  # no self

        r = random.random()
        if r < self.flip_rate and len(conns_to_worst) > 0:
            # Flip a connection to worst neuron
            src = int(np.random.choice(conns_to_worst))
            self.mask[src, worst_idx] *= -1
        elif worst_sign < 0 and len(dead_to_worst) > 0:
            # Output too low → add excitatory or flip inhibitory to excitatory
            if len(conns_to_worst) > 0 and random.random() < 0.5:
                # Flip an inhibitory to excitatory
                inhib = conns_to_worst[self.mask[conns_to_worst, worst_idx] < 0]
                if len(inhib) > 0:
                    src = int(np.random.choice(inhib))
                    self.mask[src, worst_idx] = 1.0
                    return
            # Add excitatory
            src = int(np.random.choice(dead_to_worst))
            self.mask[src, worst_idx] = 1.0
            self.W[src, worst_idx] = random.choice([np.float32(0.5), np.float32(1.5)])
        elif worst_sign > 0 and len(dead_to_worst) > 0:
            # Output too high → add inhibitory or flip excitatory to inhibitory
            if len(conns_to_worst) > 0 and random.random() < 0.5:
                excit = conns_to_worst[self.mask[conns_to_worst, worst_idx] > 0]
                if len(excit) > 0:
                    src = int(np.random.choice(excit))
                    self.mask[src, worst_idx] = -1.0
                    return
            src = int(np.random.choice(dead_to_worst))
            self.mask[src, worst_idx] = -1.0
            self.W[src, worst_idx] = random.choice([np.float32(0.5), np.float32(1.5)])
        else:
            # Fallback to random mutation
            self.mutate_structure(rate)

    def mutate_structure(self, rate=0.05):
        """Structural mutation: flip / add / remove / rewire."""
        r = random.random()

        if r < self.flip_rate:
            # FLIP: toggle sign of existing connections (+1 <-> -1)
            # This is the most powerful mutation operator!
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
        """Weight mutation: toggle 0.5 <-> 1.5 for random connections."""
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * 0.05))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
            for j in range(len(idx)):
                r2, c = int(idx[j][0]), int(idx[j][1])
                self.W[r2, c] = np.float32(1.5) if self.W[r2, c] < 1.0 else np.float32(0.5)

    # === Self-wiring (inverse arousal) ===

    def self_wire(self):
        """Active neurons propose new connections.
        Inverse arousal: more wiring when accuracy is high."""
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


    # === Diagnostics ===

    def per_class_accuracy(self, inputs, targets, vocab, ticks=6):
        """Evaluate accuracy per class. Returns dict {class_id: (correct, total)}."""
        from collections import defaultdict
        stats = defaultdict(lambda: [0, 0])  # [correct, total]
        self.reset()
        # 2 passes like normal eval
        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = self.forward(world, ticks)
                if p == 1:
                    pred = np.argmax(softmax(logits))
                    stats[targets[i]][1] += 1
                    if pred == targets[i]:
                        stats[targets[i]][0] += 1
        return dict(stats)

    def activation_map(self, inputs, vocab, ticks=6):
        """Get activation pattern for each input. Returns (n_inputs, N) array."""
        maps = []
        self.reset()
        for _ in range(2):  # 2 passes
            maps_pass = []
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                self.forward(world, ticks)
                maps_pass.append(self.state.copy())
        return np.array(maps_pass)

    def interference_test(self, inputs, targets, vocab, ticks=6, n_samples=200):
        """Test interference: mutate, measure per-class delta.
        Returns (n_samples, n_classes) array of accuracy changes."""
        n_classes = vocab
        deltas = []
        base_pc = self.per_class_accuracy(inputs, targets, vocab, ticks)

        for _ in range(n_samples):
            state = self.save_state()
            self.mutate_structure(0.05)
            new_pc = self.per_class_accuracy(inputs, targets, vocab, ticks)
            self.restore_state(state)

            row = []
            for c in range(n_classes):
                base_c = base_pc.get(c, [0, 1])
                new_c = new_pc.get(c, [0, 1])
                base_a = base_c[0] / max(base_c[1], 1)
                new_a = new_c[0] / max(new_c[1], 1)
                row.append(new_a - base_a)
            deltas.append(row)
        return np.array(deltas)

    def connection_overlap(self, inputs, vocab, ticks=6):
        """Measure how many inputs share each connection (activation overlap).
        Returns (N, N) array: overlap[i,j] = number of inputs where both i and j are active."""
        act_map = self.activation_map(inputs, vocab, ticks)
        active = (np.abs(act_map) > 0.1).astype(np.float32)  # (n_inputs, N)
        # overlap[i,j] = how many inputs activate BOTH neuron i and j
        overlap = active.T @ active  # (N, N)
        return overlap

    def diagnose(self, inputs, targets, vocab, ticks=6):
        """Full diagnostic report. Returns dict with all metrics."""
        pc = self.per_class_accuracy(inputs, targets, vocab, ticks)
        act_map = self.activation_map(inputs, vocab, ticks)

        # Active neuron stats
        active_per_input = (np.abs(act_map) > 0.1).sum(axis=1)
        active_neurons = np.unique(np.where(np.abs(act_map) > 0.1)[1])

        # Connection overlap
        active_bin = (np.abs(act_map) > 0.1).astype(np.float32)
        overlap = active_bin.T @ active_bin  # (N, N)
        # Average overlap per active connection
        alive_mask = (self.mask != 0)
        conn_overlaps = overlap[alive_mask]

        return {
            'per_class': pc,
            'n_classes': vocab,
            'active_neurons_total': len(active_neurons),
            'active_per_input_mean': float(active_per_input.mean()),
            'active_per_input_std': float(active_per_input.std()),
            'conn_overlap_mean': float(conn_overlaps.mean()) if len(conn_overlaps) > 0 else 0,
            'conn_overlap_max': float(conn_overlaps.max()) if len(conn_overlaps) > 0 else 0,
            'connections': self.count_connections(),
            'pos_neg': self.pos_neg_ratio(),
        }


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, inputs, targets, vocab, max_attempts=8000, ticks=6, verbose=True):
    """Train the network using mutation + selection."""

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):  # 2 passes for state buildup
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
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        # Mutate
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        # Evaluate
        new_score = evaluate()

        # Keep or revert
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        # Phase transition
        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        # Logging
        if verbose and (att + 1) % 1000 == 0:
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Kept: {kept:3d} | Phase: {phase}")

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, kept


# === Demo ===
if __name__ == "__main__":
    import time

    V = 16  # vocabulary / classes
    N = 80  # total neurons (V shared I/O + 64 internal)
    SEED = 42

    np.random.seed(SEED)
    random.seed(SEED)

    # Setup: random permutation task (A -> B lookup)
    perm = np.random.permutation(V)
    inputs = list(range(V))

    print("=" * 60)
    print("VRAXION v22 — Self-Wiring Graph Network")
    print("=" * 60)
    print(f"  Neurons: {N} ({V} I/O + {N-V} internal)")
    print(f"  Task: {V}-class lookup")
    print(f"  Seed: {SEED}")
    print()

    net = SelfWiringGraph(N, V)
    t0 = time.time()
    best_acc, total_kept = train(net, inputs, perm, V)
    elapsed = time.time() - t0

    pos, neg = net.pos_neg_ratio()
    print(f"\n  Final: {best_acc*100:.1f}% | "
          f"Conns: {net.count_connections()} (+:{pos} -:{neg}) | "
          f"Memory: {net.memory_bytes()} bytes | "
          f"Time: {elapsed:.1f}s")
