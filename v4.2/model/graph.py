"""
Self-Wiring Graph Network
==========================
Flat graph with ternary mask and capacitor neurons.
Gradient-free: learns via mutation + selection.
Pure numpy + random. Zero dependencies.

The ONLY learnable matrix is the int8 mask {-1, 0, +1}.
Mutations sample from an O(1) alive-edge cache; the remaining learned knobs
are small co-evolved int controller params.
"""

import numpy as np
import random


class SelfWiringGraph:

    # Fixed constants (all sweep-validated)
    GAIN = 2
    CHARGE_RATE = 0.3
    SELF_CONN = 0.05
    THRESHOLD = 0.5
    CLIP_BOUND = 1.0
    PATIENCE = 0.35   # strategy flip prob on reject (~patience=10 implicit)
    LOSS_DRIFT = 0.2  # loss_pct mutation probability per attempt

    def __init__(self, n_neurons, vocab, density=0.06):
        self.N = n_neurons
        self.V = vocab

        # Split I/O: first V = input, last V = output
        self.out_start = n_neurons - vocab if n_neurons >= 2 * vocab else 0

        # Ternary mask: the ONLY learnable matrix
        r = np.random.rand(n_neurons, n_neurons)
        self.mask = np.zeros((n_neurons, n_neurons), dtype=np.int8)
        self.mask[r < density / 2] = -1
        self.mask[r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Alive edges: list for O(1) random pick, set for O(1) undo
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

        # Persistent state
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.charge = np.zeros(n_neurons, dtype=np.float32)

        # Co-evolved learned params (all int8)
        self.loss_pct = np.int8(15)    # 1-50, charge loses loss% per tick
        self.signal = np.int8(0)       # Q1: 0=structural, 1=signal-only (flip)
        self.grow = np.int8(1)         # Q2: 0=shrink, 1=grow (only if structural)
        self.intensity = np.int8(7)    # 1-15, n_changes per mutation

    def reset(self):
        self.state *= 0
        self.charge *= 0

    @property
    def retention(self):
        return np.float32((100 - int(self.loss_pct)) * 0.01)

    @property
    def loss(self):
        """Backward-compatible alias used by ad-hoc experiment scripts."""
        return int(self.loss_pct)

    @loss.setter
    def loss(self, value):
        self.loss_pct = np.int8(max(1, min(50, int(value))))

    @property
    def leak(self):
        """Backward-compatible retention view used by older scripts."""
        return float(self.retention)

    @leak.setter
    def leak(self, value):
        # Support:
        # - float retention 0..1
        # - int legacy retention bucket 50..99
        # - int direct loss bucket 1..50
        if isinstance(value, (float, np.floating)):
            loss = int(round((1.0 - float(value)) * 100.0))
        else:
            iv = int(value)
            loss = 100 - iv if 50 <= iv <= 99 else iv
        self.loss_pct = np.int8(max(1, min(50, loss)))

    def forward(self, world, ticks=8):
        """Single-input forward pass."""
        act = self.state.copy()
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ self.mask * self.GAIN + act * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw * self.CHARGE_RATE
            self.charge *= retain
            act = np.maximum(self.charge - self.THRESHOLD, 0.0)
            self.charge = np.clip(self.charge, -self.CLIP_BOUND, self.CLIP_BOUND)
        self.state = act.copy()
        return self.charge[self.out_start:self.out_start + self.V]

    def forward_batch(self, ticks=8):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V, N = self.V, self.N
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = acts @ self.mask * self.GAIN + acts * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * self.CHARGE_RATE
            charges *= retain
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -self.CLIP_BOUND, self.CLIP_BOUND)
        return charges[:, self.out_start:self.out_start + V]

    def resync_alive(self):
        """Rebuild alive list+set from mask. Call after direct mask writes."""
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

    def count_connections(self):
        return len(self.alive)

    def pos_neg_ratio(self):
        pos = sum(1 for r, c in self.alive if self.mask[r, c] > 0)
        return pos, len(self.alive) - pos

    # --- State management ---

    def save_state(self):
        """Save only what affects the forward pass.
        Strategy bits (signal, grow, intensity) are NOT saved —
        they survive rejects and learn through differential survival."""
        return {
            'mask': self.mask.copy(),
            'alive': self.alive.copy(),
            'alive_set': self.alive_set.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'loss_pct': np.int8(self.loss_pct),
        }

    def restore_state(self, s):
        """Revert forward-pass state. Strategy bits untouched."""
        self.mask[:] = s['mask']
        if 'alive' in s:
            self.alive = s['alive'].copy()
            self.alive_set = s.get('alive_set', set(self.alive)).copy()
        else:
            self.resync_alive()
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.loss_pct = np.int8(s.get('loss_pct', s.get('loss', 15)))

    def replay(self, log):
        """Repeat logged ops in reverse = undo. O(changes) + O(alive) list rebuild.
        Flip = repeat (mask only). Structural ops update mask + set, list rebuilt once."""
        has_structural = False
        for entry in reversed(log):
            op = entry[0]
            if op == 'F':
                self.mask[entry[1], entry[2]] *= -1
            elif op == 'A':
                r, c = entry[1], entry[2]
                self.mask[r, c] = 0
                self.alive_set.discard((r, c))
                has_structural = True
            elif op == 'R':
                r, c = entry[1], entry[2]
                self.mask[r, c] = entry[3]
                self.alive_set.add((r, c))
                has_structural = True
            elif op == 'W':
                _, r, c_old, c_new = entry
                sign = self.mask[r, c_new]
                self.mask[r, c_new] = 0
                self.mask[r, c_old] = sign
                self.alive_set.discard((r, c_new))
                self.alive_set.add((r, c_old))
                has_structural = True
        if has_structural:
            self.alive = list(self.alive_set)

    # --- Mutation ---

    def mutate(self):
        """Decoupled strategy mutation: mask/loss revert, strategy survives.

        Strategy bits (signal, grow, intensity) are NOT reverted on reject.
        Only mask + loss_pct revert on reject (they affect the forward pass).
        Strategy updates happen through slower differential survival across
        multiple attempts rather than per-attempt scalar reversion.

        Decision tree:
          Q1 signal? → YES: flip only (CHEAP)
                       NO:  structural
                            Q2 grow? → YES: add
                                       NO:  remove/rewire
        """
        # Intensity drift (survives rejects)
        if random.random() < self.PATIENCE:
            self.intensity = np.int8(max(1, min(15, int(self.intensity) + random.choice([-1, 1]))))

        # Loss step (reverts with mask on reject)
        if random.random() < self.LOSS_DRIFT:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))

        # Mask mutations using current strategy — returns undo log
        undo = []
        for _ in range(int(self.intensity)):
            if self.signal:  # Q1: signal-only → flip (CHEAP)
                self._flip(undo)
            else:            # structural change
                if self.grow:    # Q2: grow → add
                    self._add(undo)
                else:            # shrink → remove or rewire
                    if random.random() < 0.7:
                        self._remove(undo)
                    else:
                        self._rewire(undo)
        return undo

    def mutate_with_mood(self):
        """Backward-compatible alias for older experiment scripts."""
        self.mutate()

    def _add(self, undo):
        r, c = random.randint(0, self.N-1), random.randint(0, self.N-1)
        if r != c and self.mask[r, c] == 0:
            self.mask[r, c] = random.choice([-1, 1])
            self.alive.append((r, c))
            self.alive_set.add((r, c))
            undo.append(('A', r, c))

    def _flip(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            self.mask[r, c] *= -1
            undo.append(('F', r, c))

    def _remove(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            old_sign = self.mask[r, c]
            self.mask[r, c] = 0
            self.alive[idx] = self.alive[-1]
            self.alive.pop()
            self.alive_set.discard((r, c))
            undo.append(('R', r, c, old_sign))

    def _rewire(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            nc = random.randint(0, self.N-1)
            if nc != r and nc != c and self.mask[r, nc] == 0:
                old = self.mask[r, c]
                self.mask[r, c] = 0
                self.mask[r, nc] = old
                self.alive[idx] = (r, nc)
                self.alive_set.discard((r, c))
                self.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))


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
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            # Strategy failed → reconsider (flip on reject)
            if random.random() < net.PATIENCE:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < net.PATIENCE:
                net.grow = np.int8(1 - int(net.grow))

        if verbose and (att + 1) % 1000 == 0:
            mode = "SIGNAL" if net.signal else ("GROW" if net.grow else "SHRINK")
            print(f"  [{att+1:5d}] Score: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | {mode} int={int(net.intensity)} | "
                  f"Loss: {int(net.loss_pct)}%")

        if best >= 0.99 or stale >= stale_limit:
            break

    return best
