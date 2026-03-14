"""
Self-Wiring Graph Network
==========================
Flat graph with ternary mask and capacitor neurons.
Gradient-free: learns via mutation + selection.

Architecture:
  - Ternary mask (int8): -1 (inhibit), 0 (none), +1 (excite)
  - Fixed gain = 2.0 (proven optimal over learnable/binary/ternary/dynamic)
  - Capacitor neuron: charge accumulates, leaks, fires above threshold
  - Split I/O: first V = input, last V = output
  - 2D mood: co-evolved mutation controller
  - Learnable leak: co-evolves with mask, converges ~0.98 within ~500 att
  - Sparse CSR matmul: 14x speedup at V=128 (94% of mask is zero)

Learnable parameters:
  mask (int8 NxN):  topology — the ONLY matrix
  mood_x (float):   mutation type (scout/rewirer/refiner)
  mood_z (float):   mutation intensity (1-15 changes per attempt)
  leak (float):     capacitor decay rate

Fixed constants (all sweep-validated):
  gain = 2.0, threshold = 0.5, charge_rate = 0.3, self_conn = 0.1
"""

import numpy as np
import random
from scipy import sparse


class SelfWiringGraph:
    """Flat graph network with capacitor neuron dynamics."""

    # Fixed constants (sweep-validated, do not change)
    gain = 2.0
    charge_rate = 0.3
    self_conn = 0.1
    clip_factor = 2.0

    def __init__(self, n_neurons, vocab, density=0.06, flip_rate=0.30,
                 threshold=0.5, leak=0.85, io_mode='split'):
        self.N = n_neurons
        self.V = vocab
        self.io_mode = io_mode
        self.flip_rate = flip_rate
        self.threshold = threshold
        self.leak = leak

        # Output zone
        if io_mode == 'split' and n_neurons >= 2 * vocab:
            self.out_start = n_neurons - vocab
        else:
            self.io_mode = 'shared'
            self.out_start = 0

        # Ternary mask: -1/0/+1 (the ONLY learnable matrix)
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.int8)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

        # 2D mood: controls mutation type and intensity
        self.mood_x = 0.5
        self.mood_z = 0.5

        # Sparse CSR cache (rebuilt after mask mutation)
        self._weff_dirty = True
        self._Weff_csr = None

    def _rebuild_weff(self):
        """Rebuild sparse CSR weight matrix from mask."""
        self._Weff_csr = sparse.csr_matrix(
            self.mask.astype(np.float32) * self.gain)
        self._weff_dirty = False

    def reset(self):
        self.state *= 0
        self.charge *= 0

    def forward(self, world, ticks=8):
        """Single-input forward pass with capacitor dynamics."""
        if self._weff_dirty:
            self._rebuild_weff()
        act = self.state.copy()
        clip_bound = self.threshold * self.clip_factor

        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = np.asarray(act @ self._Weff_csr) + act * self.self_conn
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw * self.charge_rate
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge, -clip_bound, clip_bound)

        self.state = act.copy()
        return self.charge[self.out_start:self.out_start + self.V]

    def forward_batch(self, ticks=8):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        if self._weff_dirty:
            self._rebuild_weff()
        V, N = self.V, self.N
        clip_bound = self.threshold * self.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)

        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ self._Weff_csr) + acts * self.self_conn
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
        """Model size: 2 bits per connection (ternary mask only)."""
        return self.count_connections() * 2 // 8

    # --- State management ---

    def save_state(self):
        return {
            'mask': self.mask.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'mood_x': self.mood_x,
            'mood_z': self.mood_z,
            'leak': self.leak,
        }

    def restore_state(self, s):
        self.mask[:] = s['mask']
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.mood_x = s['mood_x']
        self.mood_z = s['mood_z']
        self.leak = s['leak']
        self._weff_dirty = True

    # --- Mutation operators ---

    def mutate_with_mood(self):
        """2D mood-driven mutation + learnable leak."""
        # Mood mutation
        if random.random() < 0.2:
            self.mood_x = np.clip(self.mood_x + random.gauss(0, 0.15), 0.0, 1.0)
        if random.random() < 0.2:
            self.mood_z = np.clip(self.mood_z + random.gauss(0, 0.15), 0.0, 1.0)

        # Leak mutation
        if random.random() < 0.2:
            self.leak = np.clip(self.leak + random.gauss(0, 0.03), 0.5, 0.99)

        # Mask mutation
        n_changes = max(1, int(1 + self.mood_z * 14))
        for _ in range(n_changes):
            if self.mood_x < 0.33:
                if random.random() < 0.7:
                    self._add_connection()
                else:
                    self._flip_connection()
            elif self.mood_x < 0.66:
                r = random.random()
                if r < 0.6:
                    self._rewire_connection()
                elif r < 0.8:
                    self._flip_connection()
                else:
                    self._add_connection()
            else:
                self._flip_connection()

        self._weff_dirty = True

    def _add_connection(self):
        dead = np.argwhere(self.mask == 0)
        dead = dead[dead[:, 0] != dead[:, 1]]
        if len(dead) > 0:
            i = dead[random.randint(0, len(dead) - 1)]
            self.mask[i[0], i[1]] = 1 if random.random() > 0.5 else -1

    def _flip_connection(self):
        alive = np.argwhere(self.mask != 0)
        if len(alive) > 0:
            i = alive[random.randint(0, len(alive) - 1)]
            self.mask[i[0], i[1]] *= -1

    def _rewire_connection(self):
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


def train(net, inputs, targets, vocab, max_attempts=8000, ticks=8,
          stale_limit=6000, verbose=True):
    """Train via mutation + selection using 2D mood + learnable leak."""

    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        V = min(vocab, net.V)
        acc = (preds[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    kept = 0
    stale = 0

    for att in range(max_attempts):
        state = net.save_state()
        net.mutate_with_mood()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        if verbose and (att + 1) % 1000 == 0:
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Kept: {kept:3d} | Leak: {net.leak:.4f}")

        if best >= 0.99:
            break
        if stale >= stale_limit:
            break

    return best, kept
