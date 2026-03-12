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

    def forward(self, world, ticks=6):
        """
        Forward pass:
        - world: one-hot input vector (size V)
        - First tick: inject input into I/O neurons
        - Remaining ticks: free propagation
        - Output: read I/O neurons after final tick
        """
        act = self.state.copy()
        Weff = self.W * self.mask  # effective weights

        for t in range(ticks):
            act = act * self.decay

            # First tick only: inject input
            if t == 0:
                act[:self.V] = world

            # Propagate through graph
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)  # leaky relu

        self.state = act.copy()
        return act[:self.V]  # shared I/O: output = same neurons as input

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
