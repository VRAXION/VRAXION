"""
Surprise-Guided Mutation for SelfWiringGraph
=============================================
Single-sample eligibility traces + blame-directed mutation.

FINDINGS (V=16, N=48):
- Single-sample eligibility is conceptually correct over batch averaging
  (avoids cancellation of conflicting signals between samples)
- But at V=16 scale, blame-directed mutation does NOT outperform random
  mutation+selection (solve rate 3/8 vs 6/8 baseline)
- Root causes:
  1. Search space too small — random covers it efficiently
  2. Blamed edges are shared between correct/wrong predictions
  3. Fixed mutation mix (rewire/add/flip) loses the adaptive strategy
     evolution of the base class
- May show benefit at larger scales (V=64+) where random search is
  insufficient and blame can meaningfully narrow the search space
"""

import numpy as np
import random as _random
from model.graph import SelfWiringGraph, softmax


class SurpriseGraph(SelfWiringGraph):
    """SelfWiringGraph with single-sample eligibility and blame tracking.

    Provides blame-directed mutation as an alternative to random mutation.
    Use blame_mutate() for directed search, or standard mutate() for random.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Eligibility trace: which edges carried signal for a single sample
        self.eligibility = np.zeros((self.N, self.N), dtype=np.float32)

        # Per-edge blame score (decayed accumulator across update_blame calls)
        self._edge_blame = np.zeros((self.N, self.N), dtype=np.float32)

    # ── Single-sample forward with eligibility ──

    def forward_single_with_eligibility(self, input_idx, ticks=8):
        """Single-input forward that tracks which edges carried signal.

        Unlike the batch version which averages eligibility across all V
        samples (cancelling conflicting signals), this computes eligibility
        for ONE sample at a time — like the brain processing one stimulus.

        Returns: (V,) logits for this single input
        """
        N = self.N
        charge = np.zeros(N, dtype=np.float32)
        act = np.zeros(N, dtype=np.float32)
        retain = float(self.retention)

        self.eligibility *= 0

        for t in range(ticks):
            if t == 0:
                act[input_idx] = 1.0

            abs_act = np.abs(act)
            self.eligibility += np.outer(abs_act, abs_act)

            raw = act @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charge += raw
            charge *= retain
            act = np.maximum(charge - self.THRESHOLD, 0.0)
            charge = np.clip(charge, -1.0, 1.0)

        self.eligibility /= ticks
        return charge[self.out_start:self.out_start + self.V]

    # ── Blame map: which edges cause wrong answers ──

    def update_blame(self, targets, ticks=8):
        """Scan all V samples, accumulate eligibility from WRONG predictions.

        Uses a temporarily low threshold so signal can propagate even in
        sparse networks — this is an analytical pass, not a training step.

        Returns: (accuracy_at_low_threshold, n_wrong)
        """
        V = self.V
        wrong_elig = np.zeros((self.N, self.N), dtype=np.float32)
        correct = 0

        # Low threshold for signal tracing (analytical, not training)
        real_threshold = self.THRESHOLD
        self.THRESHOLD = 0.05

        for idx in range(V):
            logits = self.forward_single_with_eligibility(int(idx), ticks)
            target = int(targets[idx])

            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            pred = int(np.argmax(probs))

            if pred == target:
                correct += 1
            else:
                wrong_elig += self.eligibility

        self.THRESHOLD = real_threshold

        # Blend into decayed blame (memory across calls)
        self._edge_blame = 0.8 * self._edge_blame + 0.2 * wrong_elig
        return correct / V, V - correct

    # ── Blame-directed mutation ──

    def _pick_blamed_edge(self):
        """Pick an alive edge weighted by blame score."""
        if not self.alive:
            return None, None

        weights = np.array([
            self._edge_blame[r, c] for r, c in self.alive
        ], dtype=np.float32)

        total = weights.sum()
        if total > 0:
            weights /= total
            idx = np.random.choice(len(self.alive), p=weights)
        else:
            idx = _random.randint(0, len(self.alive) - 1)

        return idx, self.alive[idx]

    def blame_mutate(self, n_changes=None):
        """Mutation directed by blame map.

        50% blame-biased rewire, 30% random add, 20% blame-biased flip.
        Note: at V=16 this does NOT outperform random mutation.
        """
        undo = []
        iters = n_changes if n_changes is not None else int(self.intensity)
        has_blame = self._edge_blame.max() > 0

        for _ in range(iters):
            r = _random.random()
            if has_blame and r < 0.5:
                # Blame-biased rewire
                idx_and_edge = self._pick_blamed_edge()
                if idx_and_edge[0] is not None:
                    idx, (row, col) = idx_and_edge
                    nc = _random.randint(0, self.N - 1)
                    if nc != row and nc != col and self.mask[row, nc] == 0:
                        old = self.mask[row, col]
                        self.mask[row, col] = 0
                        self.mask[row, nc] = old
                        self.alive[idx] = (row, nc)
                        self.alive_set.discard((row, col))
                        self.alive_set.add((row, nc))
                        undo.append(('W', row, col, nc))
            elif r < 0.8:
                # Random add (base class)
                SelfWiringGraph._add(self, undo)
            else:
                # Blame-biased flip
                if has_blame:
                    idx_and_edge = self._pick_blamed_edge()
                    if idx_and_edge[0] is not None:
                        idx, (row, col) = idx_and_edge
                        self.mask[row, col] *= -1
                        undo.append(('F', row, col))
                else:
                    SelfWiringGraph._flip(self, undo)

        # Parameter drift (same as base class)
        if _random.randint(1, 20) <= 7:
            self.intensity = np.int8(max(1, min(15, int(self.intensity) + _random.choice([-1, 1]))))
        if _random.randint(1, 5) == 1:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + _random.randint(-3, 3))))

        return undo

    # ── Smoke test compatibility ──

    def surprise_step(self, targets, ticks=8, lr=0.1):
        """One step of blame-directed mutation + selection."""
        from lib.utils import score_batch
        V = self.V

        acc, _ = self.update_blame(targets, ticks)

        old_score, _ = score_batch(self, targets, V, ticks)
        old_loss = int(self.loss_pct)
        undo = self.blame_mutate()
        new_score, new_acc = score_batch(self, targets, V, ticks)

        if new_score > old_score:
            return new_acc, 0.0, len(undo), 0.0
        else:
            self.replay(undo)
            self.loss_pct = np.int8(old_loss)
            return old_score, 0.0, 0, 0.0


# ── Training loops ──

def train_surprise(net, targets, V, ticks=8, max_steps=5000,
                   lr=0.01, verbose=True):
    """Train using blame-directed mutation + selection."""
    from lib.utils import score_batch

    sc, best_acc = score_batch(net, targets, V, ticks)
    best_sc = sc
    stale = 0

    for step in range(max_steps):
        if step % 10 == 0:
            net.update_blame(targets, ticks)

        old_loss = int(net.loss_pct)
        undo = net.blame_mutate()
        sc, acc = score_batch(net, targets, V, ticks)

        if sc > best_sc:
            best_sc = sc
            best_acc = acc
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1

        if verbose and (step + 1) % 100 == 0:
            conns = net.count_connections()
            print(f"  [{step+1:5d}] Acc: {best_acc*100:5.1f}% | "
                  f"Score: {best_sc*100:.1f}% | "
                  f"Conns: {conns} | stale={stale}")

        if best_acc >= 0.99:
            break
        if stale >= 4000:
            break

    return best_acc


def train_hybrid(net, targets, V, ticks=8, max_steps=10000,
                 lr=0.01, explore_every=50, verbose=True):
    """Hybrid: blame-directed + periodic random mutation."""
    from lib.utils import score_batch

    sc, best_acc = score_batch(net, targets, V, ticks)
    best_sc = sc
    stale = 0

    for step in range(max_steps):
        if step % 10 == 0:
            net.update_blame(targets, ticks)

        old_loss = int(net.loss_pct)

        if explore_every and (step + 1) % explore_every == 0:
            undo = SelfWiringGraph.mutate(net)
        else:
            undo = net.blame_mutate()

        sc, acc = score_batch(net, targets, V, ticks)

        if sc > best_sc:
            best_sc = sc
            best_acc = acc
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1

        if verbose and (step + 1) % 200 == 0:
            conns = net.count_connections()
            print(f"  [{step+1:5d}] Acc: {best_acc*100:5.1f}% | "
                  f"Score: {best_sc*100:.1f}% | Conns: {conns}")

        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    return best_acc
