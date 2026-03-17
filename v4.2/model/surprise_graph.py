"""
Three-Factor Local Learning for SelfWiringGraph
================================================
Surprise-modulated Hebbian update: Δw = eligibility × surprise_delta

This replaces blind mutation+selection with directed local updates.
Each edge changes based on its OWN activity, modulated by a global
surprise signal — exactly like biological dopamine-gated plasticity.

PROTOTYPE — not yet integrated into training loop.
"""

import numpy as np
from model.graph import SelfWiringGraph, softmax


class SurpriseGraph(SelfWiringGraph):
    """SelfWiringGraph extended with three-factor local learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Eligibility trace: accumulates pre×post activity per edge
        # Same shape as mask — only nonzero where edges exist + where activity flows
        self.eligibility = np.zeros((self.N, self.N), dtype=np.float32)

        # Tendency buffer: accumulates float deltas, snaps to ternary
        # This solves the "ternary constraint" problem — changes accumulate
        # until they cross a threshold, then the mask flips
        self.tendency = np.zeros((self.N, self.N), dtype=np.float32)

        # Surprise baseline (EMA) — "what surprise level do I expect?"
        self.surprise_baseline = np.float32(np.log(self.V))  # start at max entropy
        self.baseline_alpha = 0.1  # EMA smoothing factor

    # ── Phase 1: Forward with eligibility tracking ──

    def forward_with_eligibility(self, ticks=8):
        """Batch forward that also accumulates eligibility traces.

        For each tick, for each active edge (i→j):
            eligibility[i,j] += |act_i| × |act_j|

        This tells us: "how much did this edge participate in the computation?"
        High eligibility = this edge carried signal. Low = dormant.

        Returns: (V,V) logits (same as forward_batch)
        """
        V, N = self.V, self.N
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        retain = float(self.retention)

        # Reset eligibility for this forward pass
        self.eligibility *= 0

        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)

            # ── Eligibility accumulation ──
            # For each sample s: eligibility[i,j] += |acts[s,i]| × |acts[s,j]|
            # Summed over all V samples: eligibility = |acts|^T @ |acts| (outer product)
            # This is O(V × N²) but we only need it where mask != 0
            abs_acts = np.abs(acts)  # (V, N)
            # Sum over samples: mean pre×post correlation across all inputs
            self.eligibility += (abs_acts.T @ abs_acts) / V  # (N, N)

            raw = acts @ self.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= retain
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -1.0, 1.0)

        # Normalize eligibility by ticks so it's scale-independent
        self.eligibility /= ticks

        return charges[:, self.out_start:self.out_start + V]

    # ── Phase 2: Compute surprise ──

    def compute_surprise(self, logits, targets):
        """Compute prediction error (surprise) relative to running baseline.

        Returns:
            delta: positive = better than expected, negative = worse
            raw_surprise: current cross-entropy (for logging)
        """
        V = self.V
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)

        # Per-sample surprise = -log(prob of correct answer)
        target_probs = np.clip(probs[np.arange(V), targets[:V]], 1e-10, 1.0)
        raw_surprise = (-np.log(target_probs)).mean()

        # Delta: baseline - current (positive = improving = "better than expected")
        delta = float(self.surprise_baseline - raw_surprise)

        # Update baseline EMA
        self.surprise_baseline = (
            (1 - self.baseline_alpha) * self.surprise_baseline +
            self.baseline_alpha * raw_surprise
        )

        return delta, float(raw_surprise)

    # ── Phase 3: Local three-factor update ──

    def local_update(self, delta, lr=0.01):
        """Apply three-factor update: Δtendency = lr × eligibility × delta.

        Where eligibility is high AND delta is positive → strengthen active paths.
        Where eligibility is high AND delta is negative → weaken active paths.

        When |tendency| exceeds a threshold → snap mask change:
        - Active edge with high positive tendency → keep (reinforce)
        - Active edge with high negative tendency → flip sign or remove
        - Zero edge with high eligibility → candidate for birth (add)

        Returns:
            n_changes: number of actual mask modifications
        """
        # Δtendency = lr × eligibility × delta
        # Only update where mask is nonzero (existing edges) OR eligibility is high
        dt = lr * self.eligibility * delta
        self.tendency += dt

        n_changes = 0
        SNAP_THRESHOLD = 0.5  # when |tendency| exceeds this → act

        # ── Existing edges: flip or remove ──
        active_mask = (self.mask != 0)
        flip_candidates = active_mask & (np.abs(self.tendency) > SNAP_THRESHOLD)

        if flip_candidates.any():
            rows, cols = np.where(flip_candidates)
            for r, c in zip(rows, cols):
                if self.tendency[r, c] > SNAP_THRESHOLD:
                    # Strong positive tendency → reinforce (make sure sign aligns)
                    # Nothing to do — edge is already helping
                    self.tendency[r, c] = 0  # reset
                elif self.tendency[r, c] < -SNAP_THRESHOLD:
                    # Strong negative tendency → this edge is hurting
                    if abs(self.tendency[r, c]) > 2 * SNAP_THRESHOLD:
                        # Very negative → remove entirely
                        self.mask[r, c] = 0
                        self.alive_set.discard((r, c))
                    else:
                        # Moderately negative → flip sign
                        self.mask[r, c] *= -1
                    self.tendency[r, c] = 0
                    n_changes += 1

        # ── Birth: zero edges with high eligibility ──
        # If nearby neurons are very active but no edge exists → create one
        birth_candidates = (~active_mask) & (self.eligibility > 0.3)
        if birth_candidates.any() and delta > 0:
            # Only birth during positive surprise (things are going well)
            rows, cols = np.where(birth_candidates)
            # Limit births to avoid explosion
            max_births = max(1, int(self.intensity))
            indices = np.random.choice(len(rows), min(max_births, len(rows)), replace=False)
            for idx in indices:
                r, c = int(rows[idx]), int(cols[idx])
                if r != c:  # no self-connections
                    self.mask[r, c] = self.DRIVE if np.random.rand() > 0.5 else -self.DRIVE
                    self.alive.append((r, c))
                    self.alive_set.add((r, c))
                    n_changes += 1

        # Rebuild alive list if we removed edges
        if n_changes > 0:
            self.alive = list(self.alive_set)

        # Decay tendency towards zero (prevents stale accumulation)
        self.tendency *= 0.95

        return n_changes

    # ── Convenience: one step of surprise-modulated learning ──

    def surprise_step(self, targets, ticks=8, lr=0.01):
        """Single learning step: forward → surprise → local update.

        Returns: (accuracy, surprise_delta, n_changes)
        """
        logits = self.forward_with_eligibility(ticks)
        delta, raw_surp = self.compute_surprise(logits, targets)

        # Accuracy for logging
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == targets[:self.V]).mean()

        n_changes = self.local_update(delta, lr)

        return acc, delta, n_changes, raw_surp


# ── Training loop: surprise-modulated ──

def train_surprise(net, targets, V, ticks=8, max_steps=5000,
                   lr=0.01, verbose=True):
    """Train using three-factor local learning.

    No mutation. No accept/reject. Every step directly improves the network
    by strengthening active edges during positive surprise and weakening
    them during negative surprise.
    """
    best_acc = 0.0
    for step in range(max_steps):
        acc, delta, n_changes, raw_surp = net.surprise_step(targets, ticks, lr)
        best_acc = max(best_acc, acc)

        if verbose and (step + 1) % 100 == 0:
            conns = net.count_connections()
            print(f"  [{step+1:5d}] Acc: {acc*100:5.1f}% | "
                  f"Surprise: {raw_surp:.3f} | Δ: {delta:+.4f} | "
                  f"Changes: {n_changes} | Conns: {conns}")

        if best_acc >= 0.99:
            break

    return best_acc


# ── Hybrid: surprise + mutation exploration ──

def train_hybrid(net, targets, V, ticks=8, max_steps=10000,
                 lr=0.01, explore_every=50, verbose=True):
    """Hybrid: surprise-modulated local learning + mutation exploration.

    Primary: three-factor local updates (directed, efficient)
    Secondary: occasional random mutation+selection (explore new topologies)

    The mutation+selection acts as "noise" / "exploration" that the local
    learning can then refine — similar to how biological evolution provides
    the architecture while learning fills in the weights.
    """
    from lib.utils import score_batch

    best_acc = 0.0
    best_score = 0.0

    for step in range(max_steps):
        # ── Primary: local surprise-modulated update ──
        acc, delta, n_changes, raw_surp = net.surprise_step(targets, ticks, lr)
        best_acc = max(best_acc, acc)

        # ── Secondary: occasional exploratory mutation ──
        if explore_every and (step + 1) % explore_every == 0:
            state = net.save_state()
            old_loss = int(net.loss_pct)
            undo = net.mutate()

            sc, _ = score_batch(net, targets, V, ticks)
            if sc > best_score:
                best_score = sc
            else:
                net.replay(undo)
                net.loss_pct = np.int8(old_loss)

        if verbose and (step + 1) % 200 == 0:
            conns = net.count_connections()
            print(f"  [{step+1:5d}] Acc: {acc*100:5.1f}% | "
                  f"Surprise: {raw_surp:.3f} | Δ: {delta:+.4f} | "
                  f"Conns: {conns} | Changes: {n_changes}")

        if best_acc >= 0.99:
            break

    return best_acc
