"""
Self-Wiring Graph Network
==========================
Flat graph with ternary mask and capacitor neurons.
Gradient-free: learns via mutation + selection.
Pure numpy + random. Zero dependencies.

The ONLY learnable matrix: int8 mask {-1, 0, +1}
Everything else is either a fixed constant or co-evolved scalar.
"""

import numpy as np
import random


class SelfWiringGraph:

    # Fixed constants (all sweep-validated)
    GAIN = 2
    CHARGE_RATE = 0.3
    SELF_CONN = 0.05
    THRESHOLD = 0.5
    CLIP_BOUND = 1.0  # was threshold * clip_factor (0.5 * 2.0)

    def __init__(self, n_neurons, vocab, density=0.06, conn_budget=0):
        self.N = n_neurons
        self.V = vocab
        self.conn_budget = conn_budget

        # Split I/O: first V = input, last V = output
        self.out_start = n_neurons - vocab if n_neurons >= 2 * vocab else 0

        # Ternary mask: the ONLY learnable matrix
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.int8)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

        # Co-evolved scalars
        self.leak = 0.85
        self.mood = 2      # 0=scout 1=rewirer 2=refiner 3=pruner
        self.intensity = 7  # n_changes per mutation (1-15)

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=8):
        """Single-input forward pass."""
        act = self.state.copy()
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ self.mask * self.GAIN + act * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw * self.CHARGE_RATE
            self.charge *= self.leak
            act = np.maximum(self.charge - self.THRESHOLD, 0.0)
            self.charge = np.clip(self.charge, -self.CLIP_BOUND, self.CLIP_BOUND)
        self.state = act.copy()
        return self.charge[self.out_start:self.out_start + self.V]

    def forward_batch(self, ticks=8):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V, N = self.V, self.N
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = acts @ self.mask * self.GAIN + acts * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * self.CHARGE_RATE
            charges *= self.leak
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -self.CLIP_BOUND, self.CLIP_BOUND)
        return charges[:, self.out_start:self.out_start + V]

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    # --- State management ---

    def save_state(self):
        return {
            'mask': self.mask.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'mood': self.mood,
            'intensity': self.intensity,
            'leak': self.leak,
        }

    def restore_state(self, s):
        self.mask[:] = s['mask']
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.mood = s['mood']
        self.intensity = s['intensity']
        self.leak = s['leak']

    # --- Mutation ---

    def mutate_with_mood(self):
        """4-zone mood-driven mutation + learnable leak."""
        # Mood step: randomly move to neighbor zone
        if random.random() < 0.35:
            self.mood = max(0, min(3, self.mood + random.choice([-1, 1])))

        # Intensity step: randomly adjust
        if random.random() < 0.35:
            self.intensity = max(1, min(15, self.intensity + random.choice([-1, 1])))

        # Leak drift
        if random.random() < 0.2:
            self.leak = np.clip(self.leak + random.gauss(0, 0.03), 0.5, 0.99)

        # Mask mutations
        for _ in range(self.intensity):
            if self.mood == 0:       # scout: grow
                if random.random() < 0.7:
                    self._add()
                else:
                    self._flip()
            elif self.mood == 1:     # rewirer: reroute
                r = random.random()
                if r < 0.6:
                    self._rewire()
                elif r < 0.8:
                    self._flip()
                else:
                    self._add()
            elif self.mood == 2:     # refiner: tune signs
                if random.random() < 0.8:
                    self._flip()
                else:
                    self._rewire()
            else:                     # pruner: shrink
                r = random.random()
                if r < 0.7:
                    self._remove()
                elif r < 0.9:
                    self._flip()
                else:
                    self._rewire()

    def _add(self):
        if self.conn_budget > 0:
            alive = np.argwhere(self.mask != 0)
            if len(alive) >= self.conn_budget:
                j = alive[random.randint(0, len(alive) - 1)]
                self.mask[j[0], j[1]] = 0
        dead = np.argwhere(self.mask == 0)
        dead = dead[dead[:, 0] != dead[:, 1]]
        if len(dead) > 0:
            i = dead[random.randint(0, len(dead) - 1)]
            self.mask[i[0], i[1]] = 1 if random.random() > 0.5 else -1

    def _flip(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            self.mask[i[0], i[1]] *= -1

    def _remove(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            self.mask[i[0], i[1]] = 0

    def _rewire(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            old = self.mask[i[0], i[1]]
            self.mask[i[0], i[1]] = 0
            nc = random.randint(0, self.N - 1)
            while nc == i[0]:
                nc = random.randint(0, self.N - 1)
            self.mask[i[0], nc] = old


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, targets, vocab, max_attempts=8000, ticks=8,
          stale_limit=6000, verbose=True):
    """Train via mutation + selection."""

    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = min(vocab, net.V)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    stale = 0

    for att in range(max_attempts):
        state = net.save_state()
        net.mutate_with_mood()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.restore_state(state)
            stale += 1

        if verbose and (att + 1) % 1000 == 0:
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] Score: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Leak: {net.leak:.4f}")

        if best >= 0.99 or stale >= stale_limit:
            break

    return best
