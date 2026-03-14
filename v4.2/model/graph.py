"""
Self-Wiring Graph Network
==========================
Flat graph with ternary mask, binary weights, and capacitor neurons.
Gradient-free: learns via mutation + selection.

Canonical decisions (empirically validated):
  - Encoding: 8-bit byte, spatial injection at tick 0
  - I/O: Split (first V = input, last V = output)
  - Activation: Capacitor (threshold=0.5, leak=0.85, charge_rate=0.3)
  - Mutation: Flip 30%, vectorized structure + weight operators
  - NaN guard: nan_to_num in forward pass
"""

import numpy as np
import random


class SelfWiringGraph:
    """Flat graph network with capacitor neuron dynamics."""

    # Capacitor hyperparameters (exposed for tuning)
    charge_rate = 0.3
    self_conn = 0.1
    clip_factor = 2.0

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 threshold=0.5, leak=0.85, io_mode='split'):
        self.N = n_neurons
        self.V = vocab
        self.io_mode = io_mode
        self.flip_rate = flip_rate
        self.last_acc = 0.0
        self.threshold = threshold
        self.leak = leak

        # Output zone: split = last V neurons, shared = first V neurons
        if io_mode == 'split' and n_neurons >= 2 * vocab:
            self.out_start = n_neurons - vocab
        else:
            self.io_mode = 'shared'
            self.out_start = 0

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

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=8):
        """Single-input forward pass with capacitor dynamics."""
        act = self.state.copy()
        Weff = self.W * self.mask
        clip_bound = self.threshold * self.clip_factor

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ Weff + act * self.self_conn
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw * self.charge_rate
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge, -clip_bound, clip_bound)

        self.state = act.copy()
        return self.charge[self.out_start:self.out_start + self.V]

    def forward_batch(self, ticks=8):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        Weff = self.W * self.mask
        V, N = self.V, self.N
        clip_bound = self.threshold * self.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)

        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = acts @ Weff + acts * self.self_conn
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * self.charge_rate
            charges *= self.leak
            acts = np.maximum(charges - self.threshold, 0.0)
            charges = np.clip(charges, -clip_bound, clip_bound)

        return charges[:, self.out_start:self.out_start + V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    def memory_bytes(self):
        """Model size in bytes (2 bit mask + 1 bit weight per connection)."""
        return self.count_connections() * 3 // 8

    # --- State management ---

    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(), self.charge.copy())

    def restore_state(self, s):
        self.W, self.mask, self.state, self.charge = (
            s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy())

    # --- Mutation operators ---

    def mutate_structure(self, rate=0.05):
        """Structural mutation: flip / add / remove / rewire (vectorized)."""
        r = random.random()

        if r < self.flip_rate:
            alive = np.argwhere(self.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate * 0.5))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                rows, cols = idx[:, 0], idx[:, 1]
                self.mask[rows, cols] *= -1
        else:
            action = random.choice(['add_pos', 'add_neg', 'remove', 'rewire'])

            if action in ('add_pos', 'add_neg'):
                dead = np.argwhere(self.mask == 0)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    n = max(1, int(len(dead) * rate))
                    idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
                    rows, cols = idx[:, 0], idx[:, 1]
                    self.mask[rows, cols] = 1.0 if action == 'add_pos' else -1.0
                    self.W[rows, cols] = np.where(
                        np.random.rand(len(rows)) > 0.5,
                        np.float32(0.5), np.float32(1.5))

            elif action == 'remove':
                alive = np.argwhere(self.mask != 0)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                    self.mask[idx[:, 0], idx[:, 1]] = 0

            else:  # rewire (sequential dependency)
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
            rows, cols = idx[:, 0], idx[:, 1]
            self.W[rows, cols] = np.where(
                self.W[rows, cols] < 1.0,
                np.float32(1.5), np.float32(0.5))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8,
          stale_limit=6000, phase_switch=2500, verbose=True):
    """Train via mutation + selection with structure/both phase transition."""

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
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

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

        if phase == "STRUCTURE" and stale > phase_switch and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if verbose and (att + 1) % 1000 == 0:
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Kept: {kept:3d} | Phase: {phase}")

        if best >= 0.99:
            break
        if stale >= stale_limit:
            break

    return best, kept
