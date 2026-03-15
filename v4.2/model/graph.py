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
    CLIP_BOUND = 1.0
    PATIENCE = 0.35  # strategy flip prob on reject (~patience=10 implicit)

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

    def count_connections(self):
        return int((self.mask != 0).sum())

    def pos_neg_ratio(self):
        return int((self.mask > 0).sum()), int((self.mask < 0).sum())

    # --- State management ---

    def save_state(self):
        """Save only what affects the forward pass.
        Strategy bits (signal, grow, intensity) are NOT saved —
        they survive rejects and learn through differential survival."""
        return {
            'mask': self.mask.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'loss_pct': np.int8(self.loss_pct),
        }

    def restore_state(self, s):
        """Revert forward-pass state. Strategy bits untouched."""
        self.mask[:] = s['mask']
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.loss_pct = np.int8(s.get('loss_pct', s.get('loss', 15)))

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
        if random.random() < 0.2:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))

        # Mask mutations using current strategy
        for _ in range(int(self.intensity)):
            if self.signal:  # Q1: signal-only → flip (CHEAP)
                self._flip()
            else:            # structural change
                if self.grow:    # Q2: grow → add
                    self._add()
                else:            # shrink → remove or rewire
                    if random.random() < 0.7:
                        self._remove()
                    else:
                        self._rewire()

    def mutate_with_mood(self):
        """Backward-compatible alias for older experiment scripts."""
        self.mutate()

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
        net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.restore_state(state)
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
