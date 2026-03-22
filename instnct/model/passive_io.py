"""
PassiveIO Graph — Input/Output as passive projection sockets
=============================================================
Same self-wiring mutation engine, BUT:
  - Input = fixed input_projection  (V → H) projection matrix
  - Output = fixed output_projection (H → V) projection matrix
  - Mutable mask is ONLY H × H  (hidden-to-hidden)
  - No IN/OUT neurons — they're just "sockets" / "plugs"

This forces all computation through the hidden layer.
"""

import numpy as np
import random

from model.graph import SelfWiringGraph, train as _orig_train


class PassiveIOGraph:
    """Self-wiring graph with passive I/O sockets."""

    DENSITY = 4
    DRIVE = 0.6
    THRESHOLD = 0.5
    CAP_RATIO = 120

    def __init__(self, vocab, h_ratio=3, proj='random'):
        self.V = vocab
        self.H = vocab * h_ratio          # hidden neurons only
        self.N = self.H                     # for compatibility
        self.out_start = 0                  # not used, but keeps API compat

        # ── Fixed projection matrices (NOT learned) ──
        if proj == 'random':
            # Random orthogonal-ish projection
            self.input_projection = np.random.randn(vocab, self.H).astype(np.float32)
            self.input_projection /= np.linalg.norm(self.input_projection, axis=1, keepdims=True)
            self.output_projection = np.random.randn(self.H, vocab).astype(np.float32)
            self.output_projection /= np.linalg.norm(self.output_projection, axis=0, keepdims=True)
        elif proj == 'identity':
            # First V hidden neurons = input slots, last V = output slots
            self.input_projection = np.zeros((vocab, self.H), dtype=np.float32)
            self.input_projection[np.arange(vocab), np.arange(vocab)] = 1.0
            self.output_projection = np.zeros((self.H, vocab), dtype=np.float32)
            self.output_projection[np.arange(self.H - vocab, self.H), np.arange(vocab)] = 1.0
        elif proj == 'hadamard':
            # Hadamard-like structured projection
            H_mat = self._hadamard_like(vocab, self.H)
            self.input_projection = H_mat.astype(np.float32)
            self.output_projection = H_mat.T.astype(np.float32)
        else:
            raise ValueError(f"Unknown proj: {proj}")

        # ── Mutable mask: H × H only (hidden-to-hidden) ──
        d = self.DENSITY / 100
        r = np.random.rand(self.H, self.H)
        self.mask = np.zeros((self.H, self.H), dtype=np.float32)
        self.mask[r < d / 2] = -self.DRIVE
        self.mask[r > 1 - d / 2] = self.DRIVE
        np.fill_diagonal(self.mask, 0)

        # Alive edges
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

        # Persistent state (hidden only)
        self.state = np.zeros(self.H, dtype=np.float32)
        self.charge = np.zeros(self.H, dtype=np.float32)

        # Co-evolved params
        self.loss_pct = np.int8(15)
        self.drive = np.int8(1)

    @staticmethod
    def _hadamard_like(V, H):
        """Build a pseudo-Hadamard projection V→H."""
        # Use Walsh-Hadamard if power of 2, otherwise random orthogonal rows
        mat = np.random.randn(V, H)
        # QR for orthogonality
        Q, _ = np.linalg.qr(mat.T)
        return Q.T[:V]  # V × H

    def reset(self):
        self.state *= 0
        self.charge *= 0

    @property
    def retention(self):
        return (100 - int(self.loss_pct)) * 0.01

    # ── Forward ──

    def forward(self, world, ticks=8):
        """Single input forward pass."""
        act = self.state.copy()
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                # Project input into hidden space (additive injection)
                act += world @ self.input_projection
            raw = act @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw
            self.charge *= retain
            act = np.maximum(self.charge - self.THRESHOLD, 0.0)
            self.charge = np.clip(self.charge, -1.0, 1.0)
        self.state = act.copy()
        # Project hidden state to output space
        return self.charge @ self.output_projection

    def forward_batch(self, ticks=8):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V, H = self.V, self.H
        charges = np.zeros((V, H), dtype=np.float32)
        acts = np.zeros((V, H), dtype=np.float32)
        retain = float(self.retention)
        eye = np.eye(V, dtype=np.float32)
        projected_inputs = eye @ self.input_projection  # (V, H) — precompute
        for t in range(ticks):
            if t == 0:
                acts += projected_inputs
            raw = acts @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= retain
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -1.0, 1.0)
        return charges @ self.output_projection  # (V, V) logits

    # ── Mutation (delegate to same mechanics) ──

    def resync_alive(self):
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

    def count_connections(self):
        return len(self.alive)

    def pos_neg_ratio(self):
        pos = sum(1 for r, c in self.alive if self.mask[r, c] > 0)
        return pos, len(self.alive) - pos

    def save_state(self):
        return {
            'mask': self.mask.copy(),
            'alive': self.alive.copy(),
            'alive_set': self.alive_set.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'loss_pct': np.int8(self.loss_pct),
            'drive': np.int8(self.drive),
        }

    def restore_state(self, s):
        self.mask[:] = s['mask']
        self.alive = s['alive'].copy()
        self.alive_set = s['alive_set'].copy()
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.loss_pct = np.int8(s['loss_pct'])
        self.drive = np.int8(s['drive'])

    def replay(self, log):
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

    def mutate(self, forced_op=None, n_changes=1, freeze_params=False):
        if forced_op is not None:
            undo = []
            op_map = {
                'add': self._add, 'remove': self._remove,
                'rewire': self._rewire, 'flip': self._flip,
            }
            for _ in range(max(0, int(n_changes))):
                op_map[forced_op](undo)
            return undo

        if random.randint(1, 5) == 1 and not freeze_params:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7 and not freeze_params:
            self.drive = np.int8(max(-15, min(15, int(self.drive) + random.choice([-1, 1]))))

        undo = []
        d = int(self.drive)
        if d > 0:
            for _ in range(d):
                self._add(undo)
        elif d < 0:
            for _ in range(-d):
                self._remove(undo)
        else:
            self._rewire(undo)
        return undo

    def _add(self, undo):
        cap = self.V * 3 * self.CAP_RATIO  # same cap formula
        if len(self.alive) >= cap:
            return
        r, c = random.randint(0, self.H-1), random.randint(0, self.H-1)
        if r != c and self.mask[r, c] == 0:
            self.mask[r, c] = self.DRIVE if random.randint(0, 1) else -self.DRIVE
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
            nc = random.randint(0, self.H-1)
            if nc != r and nc != c and self.mask[r, nc] == 0:
                old = self.mask[r, c]
                self.mask[r, c] = 0
                self.mask[r, nc] = old
                self.alive[idx] = (r, nc)
                self.alive_set.discard((r, c))
                self.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))


def train_passive(net, targets, vocab, max_attempts=8000, ticks=8,
                  stale_limit=6000, verbose=True):
    """Train PassiveIOGraph — same algo as original."""

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
    rewire_threshold = stale_limit // 3

    for att in range(max_attempts):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

            if stale > rewire_threshold:
                rw_undo = []
                net._rewire(rw_undo)
                rw_score = evaluate()
                if rw_score > score:
                    score = rw_score
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw_undo)

        if verbose and (att + 1) % 1000 == 0:
            print(f"  [{att+1:5d}] Score: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | drive={int(net.drive):+d} | "
                  f"Loss: {int(net.loss_pct)}%")

        if best >= 0.99 or stale >= stale_limit:
            break

    return best
