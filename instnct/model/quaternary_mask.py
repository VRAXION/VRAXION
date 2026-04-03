"""Quaternary upper-triangle connection mask for SelfWiringGraph.

Encoding per neuron pair (i,j) where i < j:
    0 = no connection
    1 = i -> j  (forward)
    2 = j -> i  (backward)
    3 = bidirectional (both)

Storage: flat uint8 array of length H*(H-1)//2 (upper triangle only).
"""

import numpy as np
import random as _random


class QuaternaryMask:

    # Class-level triu cache keyed by H
    _triu_cache = {}

    def __init__(self, H, data=None):
        self.H = H
        self.n_pairs = H * (H - 1) // 2
        if data is not None:
            self.data = np.asarray(data, dtype=np.uint8)
            assert self.data.shape == (self.n_pairs,)
        else:
            self.data = np.zeros(self.n_pairs, dtype=np.uint8)
        # Cache triu indices for this H
        if H not in QuaternaryMask._triu_cache:
            QuaternaryMask._triu_cache[H] = np.triu_indices(H, k=1)
        self._triu_i, self._triu_j = QuaternaryMask._triu_cache[H]
        # Alive tracking: indices into self.data where value != 0
        self._rebuild_alive()

    # ------------------------------------------------------------------
    # Alive tracking (mirrors graph.py self.alive pattern)
    # ------------------------------------------------------------------

    def _rebuild_alive(self):
        nz = np.where(self.data != 0)[0]
        self._alive = nz.tolist()
        self._alive_set = set(self._alive)

    def _add_alive(self, idx):
        if idx not in self._alive_set:
            self._alive.append(idx)
            self._alive_set.add(idx)

    def _remove_alive(self, idx):
        if idx in self._alive_set:
            self._alive_set.discard(idx)
            try:
                pos = self._alive.index(idx)
                self._alive[pos] = self._alive[-1]
                self._alive.pop()
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _pair_index(self, i, j):
        """Linear index for pair (i,j). Canonicalises so i < j."""
        if i > j:
            i, j = j, i
        return i * self.H - i * (i + 1) // 2 + j - i - 1

    def _is_swapped(self, i, j):
        """Return True if i > j (pair was swapped to canonical order)."""
        return i > j

    # ------------------------------------------------------------------
    # Pair access
    # ------------------------------------------------------------------

    def get_pair(self, i, j):
        """Get quaternary value for neuron pair (i,j). Canonical: i<j stored."""
        swapped = i > j
        idx = self._pair_index(i, j)
        val = int(self.data[idx])
        if swapped and val in (1, 2):
            return 3 - val  # flip perspective: 1<->2
        return val

    def set_pair(self, i, j, val):
        """Set quaternary value for neuron pair (i,j).

        val semantics are from caller's perspective:
            1 = i->j, 2 = j->i, 3 = bidir, 0 = none
        Internally stored with canonical i<j orientation.
        """
        swapped = i > j
        idx = self._pair_index(i, j)
        if swapped and val in (1, 2):
            val = 3 - val  # flip to canonical perspective
        old = int(self.data[idx])
        self.data[idx] = val
        if val == 0:
            self._remove_alive(idx)
        else:
            self._add_alive(idx)
        return old

    # ------------------------------------------------------------------
    # Conversion: bool mask <-> quaternary
    # ------------------------------------------------------------------

    @classmethod
    def from_bool_mask(cls, mask):
        """Convert H x H boolean mask to QuaternaryMask."""
        H = mask.shape[0]
        qm = cls(H)
        ii, jj = qm._triu_i, qm._triu_j
        fwd = mask[ii, jj]   # i->j edges
        bwd = mask[jj, ii]   # j->i edges
        # Encode: fwd=1, bwd=2, both=3, neither=0
        qm.data[:] = fwd.astype(np.uint8) + bwd.astype(np.uint8) * 2
        qm._rebuild_alive()
        return qm

    def to_bool_mask(self):
        """Expand quaternary back to H x H boolean mask."""
        H = self.H
        mask = np.zeros((H, H), dtype=bool)
        d = self.data
        ii, jj = self._triu_i, self._triu_j
        fwd = (d == 1) | (d == 3)
        bwd = (d == 2) | (d == 3)
        mask[ii[fwd], jj[fwd]] = True
        mask[jj[bwd], ii[bwd]] = True
        return mask

    # ------------------------------------------------------------------
    # Directed edge list (for forward pass sparse cache)
    # ------------------------------------------------------------------

    def to_directed_edges(self):
        """Return (rows, cols) directed edge arrays for sparse forward pass.

        Compatible with SelfWiringGraph._sparse_mul_1d_from_cache.
        """
        d = self.data
        ii, jj = self._triu_i, self._triu_j
        fwd = (d == 1) | (d == 3)
        bwd = (d == 2) | (d == 3)
        rows = np.concatenate([ii[fwd], jj[bwd]]).astype(np.intp)
        cols = np.concatenate([jj[fwd], ii[bwd]]).astype(np.intp)
        return rows, cols

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def count_edges(self):
        """Total directed edges. val 1/2 -> 1 edge, val 3 -> 2 edges."""
        d = self.data
        return int(np.sum(d == 1) + np.sum(d == 2) + 2 * np.sum(d == 3))

    def count_bidir(self):
        """Count of bidirectional pairs (val == 3)."""
        return int(np.sum(self.data == 3))

    def count_by_state(self):
        """Histogram {0: n, 1: n, 2: n, 3: n}."""
        d = self.data
        return {v: int(np.sum(d == v)) for v in range(4)}

    @property
    def memory_bytes(self):
        return self.data.nbytes

    def copy(self):
        return QuaternaryMask(self.H, self.data.copy())

    # ------------------------------------------------------------------
    # Loop detection — search-based, O(edges²) not O(H³)
    # ------------------------------------------------------------------

    def loop_levels(self):
        """Per-neuron loop participation: bidir (2-cycle) and triangle (3-cycle).

        Returns (has_bidir, in_tri) — two bool arrays of shape (H,).
        has_bidir[i] = True if neuron i is in any bidirectional pair.
        in_tri[i] = True if neuron i is in any directed 3-cycle (i→j→k→i).

        Bidir: O(n_pairs) — just check data == 3.
        Triangle: O(edges × avg_degree) — neighbor walk, not M³.
        """
        H = self.H
        d = self.data
        ii, jj = self._triu_i, self._triu_j

        # --- Bidir: pairs with value 3 ---
        bidir_mask = (d == 3)
        has_bidir = np.zeros(H, dtype=bool)
        if np.any(bidir_mask):
            has_bidir[ii[bidir_mask]] = True
            has_bidir[jj[bidir_mask]] = True

        # --- Build adjacency list for directed edges (forward only) ---
        fwd = (d == 1) | (d == 3)
        bwd = (d == 2) | (d == 3)
        # All directed edges: source → target
        src = np.concatenate([ii[fwd], jj[bwd]])
        tgt = np.concatenate([jj[fwd], ii[bwd]])

        # Build adjacency: neighbors[i] = list of neurons i points to
        neighbors = [[] for _ in range(H)]
        for s, t in zip(src.tolist(), tgt.tolist()):
            neighbors[s].append(t)

        # Build incoming set for O(1) "does k→i exist?" check
        incoming = [set() for _ in range(H)]
        for s, t in zip(src.tolist(), tgt.tolist()):
            incoming[t].add(s)

        # --- Triangle search: i→j→k→i ---
        # For each edge i→j, check if any neighbor k of j also points back to i.
        # "k→i" means k is in incoming[i].
        in_tri = np.zeros(H, dtype=bool)
        nbr_sets = [set(n) for n in neighbors]  # for fast intersection
        for i in range(H):
            if in_tri[i]:
                continue
            inc_i = incoming[i]
            if not inc_i:
                continue
            for j in neighbors[i]:
                # Any k in neighbors[j] ∩ incoming[i] (and k != i)?
                overlap = nbr_sets[j] & inc_i
                overlap.discard(i)
                if overlap:
                    in_tri[i] = True
                    in_tri[j] = True
                    for k in overlap:
                        in_tri[k] = True
                    break

        return has_bidir, in_tri

    def find_triangles(self):
        """Find all directed 3-cycles. Returns list of (i,j,k) where i->j->k->i.

        Used by crystallize_loop_aware to group edges into atomic loop units.
        """
        H = self.H
        d = self.data
        ii, jj = self._triu_i, self._triu_j
        fwd = (d == 1) | (d == 3)
        bwd = (d == 2) | (d == 3)
        src = np.concatenate([ii[fwd], jj[bwd]])
        tgt = np.concatenate([jj[fwd], ii[bwd]])

        neighbors = [[] for _ in range(H)]
        for s, t in zip(src.tolist(), tgt.tolist()):
            neighbors[s].append(t)
        incoming = [set() for _ in range(H)]
        for s, t in zip(src.tolist(), tgt.tolist()):
            incoming[t].add(s)

        triangles = []
        seen = set()
        nbr_sets = [set(n) for n in neighbors]
        for i in range(H):
            inc_i = incoming[i]
            if not inc_i:
                continue
            for j in neighbors[i]:
                overlap = nbr_sets[j] & inc_i
                overlap.discard(i)
                for k in overlap:
                    tri = tuple(sorted([i, j, k]))
                    if tri not in seen:
                        seen.add(tri)
                        triangles.append((i, j, k))
        return triangles

    def edge_in_triangle(self):
        """For each directed edge, is it part of any triangle?

        Returns a set of (src, tgt) tuples that are triangle members.
        """
        triangles = self.find_triangles()
        loop_edges = set()
        for i, j, k in triangles:
            loop_edges.add((i, j))
            loop_edges.add((j, k))
            loop_edges.add((k, i))
        return loop_edges

    def loop_levels_brute(self):
        """Reference implementation via M³ diagonal. O(H³). For validation only."""
        m = self.to_bool_mask().astype(np.float32)
        has_bidir = np.any(self.to_bool_mask() & self.to_bool_mask().T, axis=1)
        in_tri = (np.diag(m @ m @ m) > 0)
        return has_bidir, in_tri

    # ------------------------------------------------------------------
    # Mutations — each returns an undo record list
    # ------------------------------------------------------------------

    def _random_alive(self, rng):
        """Pick random alive pair index. Returns None if empty."""
        if not self._alive:
            return None
        return self._alive[rng.randint(0, len(self._alive) - 1)]

    def _random_dead(self, rng, max_tries=20):
        """Pick random pair index with value 0. Returns None after max_tries."""
        for _ in range(max_tries):
            idx = rng.randint(0, self.n_pairs - 1)
            if self.data[idx] == 0:
                return idx
        return None

    def _random_with_val(self, rng, vals, max_tries=30):
        """Pick random alive pair whose value is in vals set."""
        if not self._alive:
            return None
        for _ in range(max_tries):
            idx = self._alive[rng.randint(0, len(self._alive) - 1)]
            if self.data[idx] in vals:
                return idx
        return None

    def mutate_add(self, rng, undo):
        """Add edge to an empty pair. Direction 1 or 2, 50/50."""
        idx = self._random_dead(rng)
        if idx is None:
            return
        new_val = 1 if rng.random() < 0.5 else 2
        self.data[idx] = new_val
        self._add_alive(idx)
        undo.append(('QA', idx, 0))

    def mutate_remove(self, rng, undo):
        """Remove a unidirectional edge (val 1 or 2 -> 0)."""
        idx = self._random_with_val(rng, {1, 2})
        if idx is None:
            return
        old = int(self.data[idx])
        self.data[idx] = 0
        self._remove_alive(idx)
        undo.append(('QR', idx, old))

    def mutate_remove_smart(self, rng, undo, tri_neurons=None):
        """Remove edge, preferring dead-end (non-loop) edges.

        If tri_neurons is provided (bool array from loop_levels()),
        try to pick an edge where at least one endpoint is NOT in a triangle.
        Falls back to random remove after max_tries.
        """
        if not self._alive:
            return
        ii, jj = self._triu_i, self._triu_j
        # Try to find a non-loop edge first
        if tri_neurons is not None:
            for _ in range(30):
                idx = self._alive[rng.randint(0, len(self._alive) - 1)]
                if self.data[idx] not in (1, 2):
                    continue
                i, j = int(ii[idx]), int(jj[idx])
                if not tri_neurons[i] or not tri_neurons[j]:
                    # At least one endpoint is not in a triangle = dead-end
                    old = int(self.data[idx])
                    self.data[idx] = 0
                    self._remove_alive(idx)
                    undo.append(('QR', idx, old))
                    return
        # Fallback: regular random remove
        self.mutate_remove(rng, undo)

    def mutate_flip(self, rng, undo):
        """Atomic direction reversal: 1 <-> 2."""
        idx = self._random_with_val(rng, {1, 2})
        if idx is None:
            return
        old = int(self.data[idx])
        self.data[idx] = 3 - old  # 1->2, 2->1
        undo.append(('QF', idx, old))

    def mutate_upgrade(self, rng, undo):
        """Upgrade unidirectional to bidirectional: 1->3 or 2->3."""
        idx = self._random_with_val(rng, {1, 2})
        if idx is None:
            return
        old = int(self.data[idx])
        self.data[idx] = 3
        undo.append(('QU', idx, old))

    def mutate_downgrade(self, rng, undo):
        """Downgrade bidirectional to unidirectional: 3 -> 1 or 2."""
        idx = self._random_with_val(rng, {3})
        if idx is None:
            return
        new_val = 1 if rng.random() < 0.5 else 2
        self.data[idx] = new_val
        undo.append(('QD', idx, 3))

    def mutate_rewire(self, rng, undo):
        """Move an edge from one pair to another (same direction)."""
        idx_old = self._random_with_val(rng, {1, 2})
        if idx_old is None:
            return
        idx_new = self._random_dead(rng)
        if idx_new is None:
            return
        old_val = int(self.data[idx_old])
        self.data[idx_old] = 0
        self._remove_alive(idx_old)
        self.data[idx_new] = old_val
        self._add_alive(idx_new)
        undo.append(('QW', idx_old, old_val, idx_new))

    def apply_undo(self, undo_log):
        """Revert mutations from undo log (reverse order)."""
        for entry in reversed(undo_log):
            op = entry[0]
            if op == 'QA':
                _, idx, _ = entry
                self.data[idx] = 0
                self._remove_alive(idx)
            elif op == 'QR':
                _, idx, old = entry
                self.data[idx] = old
                self._add_alive(idx)
            elif op == 'QF':
                _, idx, old = entry
                self.data[idx] = old
            elif op == 'QU':
                _, idx, old = entry
                self.data[idx] = old
            elif op == 'QD':
                _, idx, old = entry
                self.data[idx] = old
            elif op == 'QW':
                _, idx_old, old_val, idx_new = entry
                self.data[idx_new] = 0
                self._remove_alive(idx_new)
                self.data[idx_old] = old_val
                self._add_alive(idx_old)
