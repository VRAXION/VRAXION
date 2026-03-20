"""
Self-Wiring Graph Network
==========================
Flat graph with ternary mask and capacitor neurons.
Gradient-free: learns via mutation + selection.
Pure numpy + random. Zero dependencies.

Architecture: Passive I/O sockets + hidden-only mutable mask.
  - W_in  (V × H): fixed random projection, injects input into hidden layer
  - W_out (H × V): fixed random projection, reads output from hidden layer
  - mask  (H × H): the ONLY learnable matrix {-DRIVE, 0, +DRIVE}
  - No input/output neurons — all neurons are hidden and must learn.

TEMPORARY: W_in/W_out scaled by INJ_SCALE=3.0 to overcome threshold.
  Signal dies at injection without scaling because unit-norm projection
  values (~0.09) are far below THRESHOLD (0.5). This will be revisited
  when we rework the threshold/signal-strength balance.
"""

import numpy as np
import random
import os


class SelfWiringGraph:

    # All constants sweep-validated and locked in
    NV_RATIO = 3       # hidden neurons per vocab unit
    DENSITY = 4        # init density in percent (4% = 0.04)
    DRIVE = 0.6        # GAIN(2) × CHARGE_RATE(0.3)
    THRESHOLD = 0.5    # firing threshold
    CAP_RATIO = 120    # max alive edges = V * NV_RATIO * CAP_RATIO
    INJ_SCALE = 3.0    # TEMPORARY: projection amplitude to overcome threshold
    # Mutation int fractions: PATIENCE 7/20, LOSS_DRIFT 1/5, SHRINK 7/10, LOSS_STEP +-3

    def __init__(self, *args, **_):
        # SelfWiringGraph(64)        → V=64, N=64*NV_RATIO
        # SelfWiringGraph(192, 64)   → N=192, V=64 (explicit N)
        if len(args) == 1:
            vocab = args[0]
            self.V = vocab
            self.N = vocab * self.NV_RATIO
        else:
            self.N = args[0]
            self.V = args[1]
            vocab = self.V
        self.H = vocab * self.NV_RATIO      # hidden neurons (all neurons)
        self.N = self.H                       # compat alias

        # Passive I/O projections (fixed, not learned)
        # Separate RNG so mask init stays deterministic regardless of projection
        proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
        W_in = proj_rng.randn(vocab, self.H).astype(np.float32)
        W_in /= np.linalg.norm(W_in, axis=1, keepdims=True)
        W_out = proj_rng.randn(self.H, vocab).astype(np.float32)
        W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
        self.W_in = W_in * self.INJ_SCALE    # TEMPORARY: scale up
        self.W_out = W_out * self.INJ_SCALE   # TEMPORARY: scale up

        # Mask: H × H hidden-only, float32 with baked DRIVE {-0.6, 0, +0.6}
        d = self.DENSITY / 100
        r = np.random.rand(self.H, self.H)
        self.mask = np.zeros((self.H, self.H), dtype=np.float32)
        self.mask[r < d / 2] = -self.DRIVE
        self.mask[r > 1 - d / 2] = self.DRIVE
        np.fill_diagonal(self.mask, 0)

        # Alive edges: list for O(1) random pick, set for O(1) undo
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

        # Persistent state (hidden only)
        self.state = np.zeros(self.H, dtype=np.float32)
        self.charge = np.zeros(self.H, dtype=np.float32)

        # Co-evolved learned params
        self.loss_pct = np.int8(15)    # 1-50, charge loses loss% per tick
        self.drive = np.int8(1)        # signed: +N=add N, -N=remove N, [-15,+15]

    @property
    def out_start(self):
        """Compat shim for old tests that reference out_start."""
        return 0

    def reset(self):
        self.state *= 0
        self.charge *= 0

    @property
    def retention(self):
        return (100 - int(self.loss_pct)) * 0.01

    def forward(self, world, ticks=6):
        """Single-input forward pass. Passive I/O: inject via W_in, read via W_out."""
        act = self.state.copy()
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                act += world @ self.W_in
            raw = act @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge += raw
            self.charge *= retain
            act = np.maximum(self.charge - self.THRESHOLD, 0.0)
            self.charge = np.clip(self.charge, -1.0, 1.0)
        self.state = act.copy()
        return self.charge @ self.W_out

    def forward_batch(self, ticks=6):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V, H = self.V, self.H
        charges = np.zeros((V, H), dtype=np.float32)
        acts = np.zeros((V, H), dtype=np.float32)
        retain = float(self.retention)
        projected = np.eye(V, dtype=np.float32) @ self.W_in  # (V, H)
        for t in range(ticks):
            if t == 0:
                acts += projected
            raw = acts @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= retain
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -1.0, 1.0)
        return charges @ self.W_out

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
        """Save everything that reverts on reject: mask, charge, loss, drive."""
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
        """Revert all state including drive."""
        self.mask[:] = s['mask']
        if 'alive' in s:
            self.alive = s['alive'].copy()
            self.alive_set = s.get('alive_set', set(self.alive)).copy()
        else:
            self.resync_alive()
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.loss_pct = np.int8(s.get('loss_pct', s.get('loss', 15)))
        if 'drive' in s:
            self.drive = np.int8(s['drive'])

    # --- Disk persistence ---

    def save(self, path):
        """Save winner graph to disk (.npz). Only stores non-zero edges."""
        rows, cols = np.where(self.mask != 0)
        vals = self.mask[rows, cols]
        np.savez_compressed(path,
            V=self.V,
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            vals=vals,
            loss_pct=int(self.loss_pct),
            drive=int(self.drive),
        )

    @classmethod
    def load(cls, path):
        """Load a saved graph. Reconstructs full mask from sparse edges."""
        d = np.load(path)
        V = int(d['V'])
        net = object.__new__(cls)
        net.V = V
        net.N = V * cls.NV_RATIO
        net.out_start = net.N - V if net.N >= 2 * V else 0

        net.mask = np.zeros((net.N, net.N), dtype=np.float32)
        rows, cols, vals = d['rows'], d['cols'], d['vals']
        net.mask[rows, cols] = vals
        net.alive = list(zip(rows.tolist(), cols.tolist()))
        net.alive_set = set(net.alive)

        net.state = np.zeros(net.N, dtype=np.float32)
        net.charge = np.zeros(net.N, dtype=np.float32)
        net.loss_pct = np.int8(int(d['loss_pct']))
        net.drive = np.int8(int(d['drive']))
        return net

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

    def mutate(self, forced_op=None, n_changes=1, freeze_params=False):
        """Mutate the graph.

        Default mode keeps the current learned-drive behavior.
        Optional ``forced_op`` keeps older research harnesses working without
        changing the mainline mutation policy:
          - ``add`` / ``remove`` / ``rewire`` / ``flip``
          - ``n_changes`` repeats the structural op
          - ``freeze_params`` is accepted for compatibility; forced ops do not
            drift ``loss_pct`` or ``drive`` regardless
        """
        if forced_op is not None:
            undo = []
            op_map = {
                'add': self._add,
                'remove': self._remove,
                'rewire': self._rewire,
                'flip': self._flip,
            }
            op_fn = op_map.get(forced_op)
            if op_fn is None:
                raise ValueError(f"Unsupported forced_op: {forced_op}")
            for _ in range(max(0, int(n_changes))):
                op_fn(undo)
            return undo

        # Loss drift (reverts on reject) — 1/5 chance
        if random.randint(1, 5) == 1 and not freeze_params:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))

        # Drive drift — 7/20 chance, ±1, reverts on reject
        if random.randint(1, 20) <= 7 and not freeze_params:
            self.drive = np.int8(max(-15, min(15, int(self.drive) + random.choice([-1, 1]))))

        # Execute drive: +N=add, -N=remove, 0=rewire
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

    def mutate_with_mood(self):
        """Compatibility alias for older training utilities."""
        return self.mutate()

    def _add(self, undo):
        cap = self.V * self.NV_RATIO * self.CAP_RATIO
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

    def crystallize(self, evaluate_fn, eps=1e-6, verbose=False):
        """Pass-based crystal pruning. Remove dead-weight edges without hurting score.

        Shuffles alive edges, tries removing each exactly once per pass.
        Repeats passes until a full pass removes zero edges.

        Args:
            evaluate_fn: callable() -> float, returns current score
            eps: float tolerance for score comparison
            verbose: print pass stats
        Returns:
            total edges removed
        """
        score = evaluate_fn()
        total_removed = 0
        pass_num = 0
        while True:
            alive_snapshot = list(self.alive)
            random.shuffle(alive_snapshot)
            removed_this_pass = 0
            for r, c in alive_snapshot:
                if self.mask[r, c] == 0:
                    continue
                old_val = self.mask[r, c]
                self.mask[r, c] = 0.0
                self.alive_set.discard((r, c))
                new_score = evaluate_fn()
                if new_score >= score - eps:
                    score = new_score
                    removed_this_pass += 1
                    total_removed += 1
                else:
                    self.mask[r, c] = old_val
                    self.alive_set.add((r, c))
            self.resync_alive()
            pass_num += 1
            if verbose:
                print(f"    crystal pass {pass_num}: removed {removed_this_pass}, "
                      f"remaining {len(self.alive)}")
            if removed_this_pass == 0:
                break
        return total_removed


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, targets, vocab, max_attempts=8000, ticks=6,
          stale_limit=6000, verbose=True, save_path=None):
    """Train via mutation + selection. Saves winner to save_path if provided."""

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
            if score > best:
                best = score
                if save_path:
                    net.save(save_path)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

            # Phase 3: rewire when stale — explore new topologies
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

    if save_path:
        net.save(save_path)
        if verbose:
            print(f"  Winner saved → {save_path}")

    return best
