"""
INSTNCT — Self-Wiring Graph kernel
==================================
Hidden-only recurrent substrate with fixed passive I/O projections.
Gradient-free learning happens by mutating the ternary hidden mask and the
per-neuron theta / decay vectors, then keeping only improved candidates.

Runtime contract:
  - input_projection  (V × H): fixed projection from vocab-space into hidden space
  - output_projection (H × V): fixed projection from hidden charge into vocab logits
  - mask             (H × H): learnable hidden graph, int8 {-1, 0, +1}
  - edge_magnitude   float32: scalar multiplied into sparse cache at sync time
  - theta            (H,): per-neuron firing threshold
  - decay            (H,): per-neuron decay rate
  - state            (H,): hidden activation after thresholding
  - charge           (H,): hidden pre-threshold charge, clamped to nonnegative values
"""

import numpy as np
import random


class SelfWiringGraph:

    DEFAULT_HIDDEN_RATIO = 3
    DEFAULT_DENSITY = 4
    DEFAULT_EDGE_MAGNITUDE = 1.0
    DEFAULT_CAP_RATIO = 120
    DEFAULT_PROJECTION_SCALE = 3.0
    DEFAULT_THETA = 0.1
    DEFAULT_DECAY = 0.15

    def __init__(
        self,
        vocab,
        *,
        hidden=None,
        hidden_ratio=DEFAULT_HIDDEN_RATIO,
        density=None,
        theta_init=DEFAULT_THETA,
        decay_init=DEFAULT_DECAY,
        edge_magnitude=DEFAULT_EDGE_MAGNITUDE,
        projection_scale=DEFAULT_PROJECTION_SCALE,
        cap_ratio=DEFAULT_CAP_RATIO,
        seed=None,
    ):
        self.V = int(vocab)
        if self.V <= 0:
            raise ValueError("vocab must be a positive integer")

        self.hidden_ratio = int(hidden_ratio)
        if hidden is None:
            if self.hidden_ratio <= 0:
                raise ValueError("hidden_ratio must be positive when hidden is omitted")
            self.H = self.V * self.hidden_ratio
        else:
            self.H = int(hidden)
            if self.H <= 0:
                raise ValueError("hidden must be a positive integer")

        self.density_fraction = self._density_to_fraction(
            self.DEFAULT_DENSITY if density is None else density
        )
        self.edge_magnitude = np.float32(edge_magnitude)
        self.projection_scale = np.float32(projection_scale)
        self.cap_ratio = int(cap_ratio)
        if self.cap_ratio <= 0:
            raise ValueError("cap_ratio must be positive")
        if self.edge_magnitude <= 0:
            raise ValueError("edge_magnitude must be positive")

        if seed is None:
            proj_rng = np.random.RandomState(np.random.randint(0, 2**31))
            init_rand = np.random.rand
        else:
            init_rng = np.random.RandomState(int(seed))
            proj_rng = np.random.RandomState(int(init_rng.randint(0, 2**31)))
            init_rand = init_rng.rand

        hidden = self.H
        input_projection = proj_rng.randn(self.V, hidden).astype(np.float32)
        input_projection /= np.linalg.norm(input_projection, axis=1, keepdims=True)
        output_projection = proj_rng.randn(hidden, self.V).astype(np.float32)
        output_projection /= np.linalg.norm(output_projection, axis=0, keepdims=True)
        self.input_projection = input_projection * self.projection_scale
        self.output_projection = output_projection * self.projection_scale

        # Mask: H × H hidden-only, int8 ternary {-1, 0, +1}.
        # edge_magnitude is applied at sparse-cache sync time, not stored in mask.
        r = init_rand(hidden, hidden)
        self.mask = np.zeros((hidden, hidden), dtype=np.int8)
        self.mask[r < self.density_fraction / 2] = -1
        self.mask[r > 1 - self.density_fraction / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Alive edges: canonical row-major cache + set for O(1) undo membership.
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)
        self._sync_sparse_idx()

        # Persistent state (hidden only)
        self.state = np.zeros(self.H, dtype=np.float32)
        self.charge = np.zeros(self.H, dtype=np.float32)

        # Co-evolved learned params
        self.loss_pct = np.int8(15)
        self.mutation_drive = np.int8(1)  # signed: +N=add N, -N=remove N, 0=rewire
        self.theta = np.full(self.H, float(theta_init), dtype=np.float32)
        self.decay = np.full(self.H, float(decay_init), dtype=np.float32)

    @staticmethod
    def _density_to_fraction(density):
        density = float(density)
        if density < 0:
            raise ValueError("density must be non-negative")
        if density > 1.0:
            density /= 100.0
        return float(np.clip(density, 0.0, 1.0))

    def reset(self):
        self.state *= 0
        self.charge *= 0

    @property
    def theta_mean(self):
        return float(np.mean(self.theta))

    @theta_mean.setter
    def theta_mean(self, value):
        self.theta.fill(np.float32(value))

    @property
    def retention_vec(self):
        """Per-neuron retention = 1.0 - decay."""
        return 1.0 - self.decay

    @property
    def retention_mean(self):
        return float(np.mean(self.retention_vec))

    @retention_mean.setter
    def retention_mean(self, value):
        self.decay.fill(np.float32(1.0 - float(np.clip(value, 0.0, 1.0))))

    @property
    def decay_mean(self):
        return float(np.mean(self.decay))

    @decay_mean.setter
    def decay_mean(self, value):
        self.decay.fill(np.float32(value))

    @staticmethod
    def build_sparse_cache(mask, edge_magnitude=1.0):
        """Build boolean sparse cache: separate pos/neg index arrays.

        Returns (pos_rows, pos_cols, neg_rows, neg_cols) for multiply-free
        sparse forward pass. Legacy callers using (rows, cols, vals) should
        migrate to the boolean format.

        For backward compat with static rollout methods that still accept
        edge_magnitude, the old 3-tuple format is returned when
        edge_magnitude != 1.0.
        """
        if edge_magnitude != 1.0:
            # Legacy path: float multiplication
            rows, cols = np.where(mask != 0)
            if len(rows) == 0:
                return (
                    np.empty(0, dtype=np.intp),
                    np.empty(0, dtype=np.intp),
                    np.empty(0, dtype=np.float32),
                )
            vals = mask[rows, cols].astype(np.float32) * np.float32(edge_magnitude)
            return rows.astype(np.intp), cols.astype(np.intp), vals

        # Boolean path: no float multiply needed
        pos = mask == 1
        neg = mask == -1
        pr, pc = np.where(pos)
        nr, nc = np.where(neg)
        return (
            pr.astype(np.intp), pc.astype(np.intp),
            nr.astype(np.intp), nc.astype(np.intp),
        )

    @staticmethod
    def _sparse_mul_1d_from_cache(H, act, sparse_cache):
        """Sparse act @ mask for 1D act vector. O(edges) instead of O(H^2).

        Supports both boolean 4-tuple (pos_r, pos_c, neg_r, neg_c) and
        legacy 3-tuple (rows, cols, vals) cache formats.
        """
        raw = np.zeros(H, dtype=np.float32)
        if len(sparse_cache) == 4:
            pr, pc, nr, nc = sparse_cache
            if len(pr):
                np.add.at(raw, pc, act[pr])
            if len(nr):
                np.subtract.at(raw, nc, act[nr])
        else:
            rows, cols, vals = sparse_cache
            if len(rows):
                np.add.at(raw, cols, act[rows] * vals)
        return raw

    @staticmethod
    def _sparse_mul_2d_from_cache(H, acts, sparse_cache):
        """Sparse acts @ mask for 2D batch. O(batch * edges) instead of O(batch * H^2).

        Supports both boolean 4-tuple and legacy 3-tuple cache formats.
        """
        B = acts.shape[0]
        raw = np.zeros((B, H), dtype=np.float32)
        if len(sparse_cache) == 4:
            pr, pc, nr, nc = sparse_cache
            if len(pr):
                np.add.at(raw, (slice(None), pc), acts[:, pr])
            if len(nr):
                np.subtract.at(raw, (slice(None), nc), acts[:, nr])
        else:
            rows, cols, vals = sparse_cache
            if len(rows):
                np.add.at(raw, (slice(None), cols), acts[:, rows] * vals)
        return raw

    def _sparse_mul_1d(self, act):
        return self._sparse_mul_1d_from_cache(self.H, act, self._sp_cache)

    def _sparse_mul_2d(self, acts):
        return self._sparse_mul_2d_from_cache(self.H, acts, self._sp_cache)

    @staticmethod
    def rollout_token(
        injected,
        *,
        mask,
        theta,
        decay,
        ticks,
        input_duration=1,
        state=None,
        charge=None,
        sparse_cache=None,
        edge_magnitude=1.0,
    ):
        mask = np.asarray(mask)
        # Support both int8 ternary mask and legacy float32 mask.
        # For dense path, always produce float32 with magnitude baked in.
        if mask.dtype != np.float32:
            mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
        else:
            mask_f32 = mask
        theta = np.asarray(theta, dtype=np.float32)
        decay = np.asarray(decay, dtype=np.float32)
        H = mask.shape[0]
        if mask.shape != (H, H):
            raise ValueError(f"mask must be square, got {mask.shape}")
        if theta.shape != (H,) or decay.shape != (H,):
            raise ValueError(
                f"theta and decay must both be shape {(H,)}, got {theta.shape} and {decay.shape}"
            )
        injected = np.asarray(injected, dtype=np.float32)
        if injected.shape != (H,):
            raise ValueError(f"injected must have shape {(H,)}, got {injected.shape}")

        act = np.zeros(H, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32).copy()
        cur_charge = np.zeros(H, dtype=np.float32) if charge is None else np.asarray(charge, dtype=np.float32).copy()
        ret = 1.0 - decay
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1

        for tick in range(int(ticks)):
            if tick < int(input_duration):
                act = act + injected
            raw = (
                SelfWiringGraph._sparse_mul_1d_from_cache(H, act, sparse_cache)
                if use_sparse else act @ mask_f32
            )
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            cur_charge += raw
            cur_charge *= ret
            act = np.maximum(cur_charge - theta, 0.0)
            cur_charge = np.maximum(cur_charge, 0.0)
        return act, cur_charge

    @staticmethod
    def rollout_token_batch(
        injected_batch,
        *,
        mask,
        theta,
        decay,
        ticks,
        input_duration=1,
        acts=None,
        charges=None,
        sparse_cache=None,
        edge_magnitude=1.0,
    ):
        mask = np.asarray(mask)
        if mask.dtype != np.float32:
            mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
        else:
            mask_f32 = mask
        theta = np.asarray(theta, dtype=np.float32)
        decay = np.asarray(decay, dtype=np.float32)
        H = mask.shape[0]
        if mask.shape != (H, H):
            raise ValueError(f"mask must be square, got {mask.shape}")
        injected_batch = np.asarray(injected_batch, dtype=np.float32)
        if injected_batch.ndim != 2 or injected_batch.shape[1] != H:
            raise ValueError(
                f"injected_batch must have shape (batch, {H}), got {injected_batch.shape}"
            )
        batch = injected_batch.shape[0]
        cur_acts = np.zeros((batch, H), dtype=np.float32) if acts is None else np.asarray(acts, dtype=np.float32).copy()
        cur_charges = (
            np.zeros((batch, H), dtype=np.float32)
            if charges is None else np.asarray(charges, dtype=np.float32).copy()
        )
        ret = 1.0 - decay
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1

        for tick in range(int(ticks)):
            if tick < int(input_duration):
                cur_acts = cur_acts + injected_batch
            raw = (
                SelfWiringGraph._sparse_mul_2d_from_cache(H, cur_acts, sparse_cache)
                if use_sparse else cur_acts @ mask_f32
            )
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            cur_charges += raw
            cur_charges *= ret
            cur_acts = np.maximum(cur_charges - theta, 0.0)
            cur_charges = np.maximum(cur_charges, 0.0)
        return cur_acts, cur_charges

    def readout(self, hidden_state):
        """Project one hidden-state vector into output-logit space."""
        hidden = np.asarray(hidden_state, dtype=np.float32)
        if hidden.shape != (self.H,):
            raise ValueError(f"readout expects shape {(self.H,)}, got {hidden.shape}")
        return hidden @ self.output_projection

    def readout_batch(self, hidden_states):
        """Project a batch of hidden-state vectors into output-logit space."""
        hidden = np.asarray(hidden_states, dtype=np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != self.H:
            raise ValueError(
                f"readout_batch expects shape (batch, {self.H}), got {hidden.shape}"
            )
        return hidden @ self.output_projection

    def forward(self, world, ticks=6):
        """Single-input forward pass. Passive I/O: inject via input_projection, read via output_projection."""
        world_vec = np.asarray(world, dtype=np.float32).copy()
        np.nan_to_num(world_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_token(
            injected,
            mask=self.mask,
            theta=self.theta,
            decay=self.decay,
            ticks=ticks,
            state=self.state,
            charge=self.charge,
            sparse_cache=self._sp_cache,
            edge_magnitude=self.edge_magnitude,
        )
        return self.readout(self.charge)

    def forward_batch(self, ticks=6):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V = self.V
        projected = np.eye(V, dtype=np.float32) @ self.input_projection
        _, charges = self.rollout_token_batch(
            projected,
            mask=self.mask,
            theta=self.theta,
            decay=self.decay,
            ticks=ticks,
            sparse_cache=self._sp_cache,
            edge_magnitude=self.edge_magnitude,
        )
        return self.readout_batch(charges)

    def resync_alive(self):
        """Rebuild alive list+set from mask. Call after direct mask writes."""
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)
        self._sync_sparse_idx()

    def _sync_sparse_idx(self):
        """Precompute boolean sparse cache for multiply-free forward pass.

        Splits alive edges into positive and negative index arrays.
        The forward pass uses add/subtract instead of float multiplication.
        """
        if self.alive:
            rows = np.array([r for r, c in self.alive], dtype=np.intp)
            cols = np.array([c for r, c in self.alive], dtype=np.intp)
            signs = self.mask[rows, cols]
            pos = signs > 0
            neg = signs < 0
            self._sp_cache = (
                rows[pos], cols[pos],
                rows[neg], cols[neg],
            )
            # Legacy accessors (backward compat for external callers)
            self._sp_rows = rows
            self._sp_cols = cols
            self._sp_vals = signs.astype(np.float32) * self.edge_magnitude
        else:
            self._sp_cache = (
                np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
                np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
            )
            self._sp_rows = np.empty(0, dtype=np.intp)
            self._sp_cols = np.empty(0, dtype=np.intp)
            self._sp_vals = np.empty(0, dtype=np.float32)

    def count_connections(self):
        return len(self.alive)

    def pos_neg_ratio(self):
        pos = int(np.sum(self.mask > 0))
        return pos, len(self.alive) - pos

    # --- State management ---

    def save_state(self):
        """Save everything that reverts on reject: mask, charge, loss, mutation drive, theta."""
        return {
            'mask': self.mask.copy(),
            'alive': self.alive.copy(),
            'alive_set': self.alive_set.copy(),
            'state': self.state.copy(),
            'charge': self.charge.copy(),
            'loss_pct': np.int8(self.loss_pct),
            'mutation_drive': np.int8(self.mutation_drive),
            'theta': self.theta.copy(),
            'decay': self.decay.copy(),
        }

    def restore_state(self, s):
        """Revert all state including mutation drive."""
        saved_mask = s['mask']
        if saved_mask.dtype != self.mask.dtype:
            saved_mask = np.sign(saved_mask).astype(np.int8)
        self.mask[:] = saved_mask
        if 'alive' in s:
            self.alive = s['alive'].copy()
            self.alive_set = s.get('alive_set', set(self.alive)).copy()
        else:
            self.resync_alive()
        self._sync_sparse_idx()
        self.state[:] = s['state']
        self.charge[:] = s['charge']
        self.loss_pct = np.int8(s.get('loss_pct', s.get('loss', 15)))
        if 'mutation_drive' in s:
            self.mutation_drive = np.int8(s['mutation_drive'])
        elif 'drive' in s:
            self.mutation_drive = np.int8(s['drive'])
        if 'theta' in s:
            self.theta[:] = s['theta']
        if 'decay' in s:
            self.decay[:] = s['decay']

    # --- Breeding / crossover ---

    @classmethod
    def breed(cls, parent_a, parent_b, *, fitness_a=None, fitness_b=None, seed=None):
        """Breed two parent graphs via union with sign-only edges.

        Takes all edges from both parents. Where both parents independently
        evolved the same edge with the same sign, the child keeps ±1 (no
        magnitude boost — topology carries importance, not weight magnitude).
        Where only one parent has an edge, it's kept. No pruning.

        Parameters
        ----------
        parent_a, parent_b : SelfWiringGraph
            Parents with identical V, H, and projections.
        fitness_a, fitness_b : float, optional
            Fitness scores. When both parents have an edge but disagree on
            sign, the fitter parent's value is kept.
        seed : int, optional
            RNG seed for reproducible tie-breaking.
        """
        if parent_a.V != parent_b.V or parent_a.H != parent_b.H:
            raise ValueError("Parents must have same V and H")
        if not np.array_equal(parent_a.input_projection, parent_b.input_projection):
            raise ValueError("Parents must share the same input projection")

        rng = np.random.RandomState(seed)
        fa = fitness_a if fitness_a is not None else 1.0
        fb = fitness_b if fitness_b is not None else 1.0

        # Build child via object.__new__ to skip __init__ random generation
        child = object.__new__(cls)
        child.V = parent_a.V
        child.H = parent_a.H
        child.hidden_ratio = parent_a.hidden_ratio
        child.projection_scale = parent_a.projection_scale
        child.edge_magnitude = parent_a.edge_magnitude
        child.cap_ratio = parent_a.cap_ratio
        child.density_fraction = 0.0
        child.input_projection = parent_a.input_projection.copy()
        child.output_projection = parent_a.output_projection.copy()

        a_mask = parent_a.mask
        b_mask = parent_b.mask
        a_has = a_mask != 0
        b_has = b_mask != 0
        both = a_has & b_has
        same_sign = both & (np.sign(a_mask) == np.sign(b_mask))
        disagree = both & ~same_sign
        only_a = a_has & ~b_has
        only_b = ~a_has & b_has

        child.mask = np.zeros((child.H, child.H), dtype=np.int8)
        child.mask[only_a] = a_mask[only_a]
        child.mask[only_b] = b_mask[only_b]

        # Agreement edges: same sign → keep ±1 (no magnitude boost).
        # Topology (fan-in/fan-out) carries importance, not weight magnitude.
        child.mask[same_sign] = np.sign(a_mask[same_sign]).astype(np.int8)

        # Disagreement edges: fitness-weighted pick, normal magnitude (±1)
        if np.any(disagree):
            pick_a = rng.rand(child.H, child.H) < (fa / (fa + fb + 1e-10))
            child.mask[disagree] = np.where(
                pick_a[disagree], a_mask[disagree], b_mask[disagree]).astype(np.int8)

        np.fill_diagonal(child.mask, 0)
        child.resync_alive()

        # Average meta-params with small noise
        child.theta = ((parent_a.theta + parent_b.theta) / 2.0).astype(np.float32)
        child.decay = ((parent_a.decay + parent_b.decay) / 2.0).astype(np.float32)
        child.loss_pct = np.int8((int(parent_a.loss_pct) + int(parent_b.loss_pct)) // 2)
        child.mutation_drive = np.int8((int(parent_a.mutation_drive) + int(parent_b.mutation_drive)) // 2)

        # Fresh state
        child.state = np.zeros(child.H, dtype=np.float32)
        child.charge = np.zeros(child.H, dtype=np.float32)

        return child

    def save(self, path):
        """Save winner graph to disk (.npz) with exact forward-path projections."""
        rows, cols = np.where(self.mask != 0)
        vals = self.mask[rows, cols].astype(np.int8)
        np.savez_compressed(path,
            schema_version=np.int16(3),
            V=self.V,
            H=self.H,
            hidden_ratio=np.int32(self.hidden_ratio),
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            vals=vals,
            loss_pct=int(self.loss_pct),
            mutation_drive=int(self.mutation_drive),
            theta=self.theta,
            decay=self.decay,
            projection_scale=np.float32(self.projection_scale),
            edge_magnitude=np.float32(self.edge_magnitude),
            cap_ratio=np.int32(self.cap_ratio),
            input_projection=self.input_projection,
            output_projection=self.output_projection,
        )

    @classmethod
    def load(cls, path):
        """Load a saved graph. New-format checkpoints round-trip exactly."""
        with np.load(path) as d:
            required = {
                'V', 'H', 'rows', 'cols', 'vals', 'loss_pct',
                'theta', 'decay', 'input_projection', 'output_projection',
            }
            missing = sorted(required.difference(d.files))
            if missing:
                raise ValueError(
                    "Checkpoint is missing required fields for exact round-trip load: "
                    + ", ".join(missing)
                )

            V = int(d['V'])
            H = int(d['H'])
            rows = np.array(d['rows'], dtype=np.int32)
            cols = np.array(d['cols'], dtype=np.int32)
            raw_vals = np.array(d['vals'])
            # Backward compat: old checkpoints stored float vals with baked magnitude
            if raw_vals.dtype in (np.float32, np.float64):
                vals = np.sign(raw_vals).astype(np.int8)
            else:
                vals = raw_vals.astype(np.int8)
            input_projection = np.array(d['input_projection'], dtype=np.float32)
            output_projection = np.array(d['output_projection'], dtype=np.float32)
            theta = np.array(d['theta'], dtype=np.float32)
            decay = np.array(d['decay'], dtype=np.float32)
            loss_pct = np.int8(int(d['loss_pct']))
            mutation_drive = np.int8(int(d['mutation_drive'])) if 'mutation_drive' in d.files else np.int8(int(d['drive']))
            hidden_ratio = int(d['hidden_ratio']) if 'hidden_ratio' in d.files else max(1, H // max(V, 1))
            projection_scale = np.float32(d['projection_scale']) if 'projection_scale' in d.files else np.float32(cls.DEFAULT_PROJECTION_SCALE)
            edge_magnitude = np.float32(d['edge_magnitude']) if 'edge_magnitude' in d.files else np.float32(cls.DEFAULT_EDGE_MAGNITUDE)
            cap_ratio = int(d['cap_ratio']) if 'cap_ratio' in d.files else int(cls.DEFAULT_CAP_RATIO)

        if input_projection.shape != (V, H):
            raise ValueError(
                f"input_projection shape mismatch: expected {(V, H)}, got {input_projection.shape}"
            )
        if output_projection.shape != (H, V):
            raise ValueError(
                f"output_projection shape mismatch: expected {(H, V)}, got {output_projection.shape}"
            )
        if theta.shape != (H,):
            raise ValueError(f"theta shape mismatch: expected {(H,)}, got {theta.shape}")
        if decay.shape != (H,):
            raise ValueError(f"decay shape mismatch: expected {(H,)}, got {decay.shape}")
        if rows.shape != cols.shape or rows.shape != vals.shape:
            raise ValueError("rows, cols, and vals must have identical shapes")
        if np.any(rows < 0) or np.any(rows >= H) or np.any(cols < 0) or np.any(cols >= H):
            raise ValueError("checkpoint edge indices fall outside hidden graph bounds")

        net = object.__new__(cls)
        net.V = V
        net.H = H
        net.hidden_ratio = hidden_ratio
        net.projection_scale = projection_scale
        net.edge_magnitude = edge_magnitude
        net.cap_ratio = cap_ratio
        net.density_fraction = 0.0
        net.input_projection = input_projection
        net.output_projection = output_projection
        net.mask = np.zeros((H, H), dtype=np.int8)
        net.mask[rows, cols] = vals
        np.fill_diagonal(net.mask, 0)
        net.state = np.zeros(H, dtype=np.float32)
        net.charge = np.zeros(H, dtype=np.float32)
        net.loss_pct = loss_pct
        net.mutation_drive = mutation_drive
        net.theta = theta
        net.decay = decay
        net.resync_alive()
        return net

    def replay(self, log):
        """Repeat logged ops in reverse = undo. O(changes) + O(alive) list rebuild.
        Flip = repeat (mask only). Structural ops update mask + set, list rebuilt once."""
        has_structural = False
        for entry in reversed(log):
            op = entry[0]
            if op == 'F':
                self.mask[entry[1], entry[2]] *= np.int8(-1)
            elif op == 'A':
                r, c = entry[1], entry[2]
                self.mask[r, c] = np.int8(0)
                self.alive_set.discard((r, c))
                has_structural = True
            elif op == 'R':
                r, c = entry[1], entry[2]
                self.mask[r, c] = np.int8(entry[3])
                self.alive_set.add((r, c))
                has_structural = True
            elif op == 'W':
                _, r, c_old, c_new = entry
                sign = self.mask[r, c_new]
                self.mask[r, c_new] = np.int8(0)
                self.mask[r, c_old] = sign
                self.alive_set.discard((r, c_new))
                self.alive_set.add((r, c_old))
                has_structural = True
            elif op == 'T':
                self.theta[entry[1]] = np.float32(entry[2])
            elif op == 'D':
                self.decay[entry[1]] = np.float32(entry[2])
            elif op == 'L':
                self.loss_pct = np.int8(entry[1])
            elif op == 'G':
                self.mutation_drive = np.int8(entry[1])
        if has_structural:
            self.resync_alive()
        else:
            self._sync_sparse_idx()

    # --- Mutation ---

    def mutate(self, forced_op=None, n_changes=1, freeze_params=False):
        """Mutate the graph.

        Default mode keeps the current learned-drive behavior.
        Optional ``forced_op`` keeps older research harnesses working without
        changing the mainline mutation policy:
          - ``add`` / ``remove`` / ``rewire`` / ``flip``
          - ``n_changes`` repeats the structural op
          - ``freeze_params`` is accepted for compatibility; forced ops do not
            drift ``loss_pct`` or ``mutation_drive`` regardless
        """
        if forced_op is not None:
            undo = []
            op_map = {
                'add': self._add,
                'remove': self._remove,
                'rewire': self._rewire,
                'flip': self._flip,
                'theta': self._theta_mutate,
                'decay': self._decay_mutate,
            }
            op_fn = op_map.get(forced_op)
            if op_fn is None:
                raise ValueError(f"Unsupported forced_op: {forced_op}")
            for _ in range(max(0, int(n_changes))):
                op_fn(undo)
            self.resync_alive()
            return undo

        # Loss drift (reverts on reject) — 1/5 chance
        undo = []

        if random.randint(1, 5) == 1 and not freeze_params:
            undo.append(('L', int(self.loss_pct)))
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))

        # Drive drift — 7/20 chance, ±1, reverts on reject
        if random.randint(1, 20) <= 7 and not freeze_params:
            undo.append(('G', int(self.mutation_drive)))
            self.mutation_drive = np.int8(max(-15, min(15, int(self.mutation_drive) + random.choice([-1, 1]))))

        # Theta (threshold) drift — 1/5 chance, 1 random neuron, random value [0, 1]
        if random.randint(1, 5) == 1 and not freeze_params:
            self._theta_mutate(undo)

        # Execute drive: +N=add, -N=remove, 0=rewire
        d = int(self.mutation_drive)
        if d > 0:
            for _ in range(d):
                self._add(undo)
        elif d < 0:
            for _ in range(-d):
                self._remove(undo)
        else:
            self._rewire(undo)
        self.resync_alive()
        return undo

    def _add(self, undo):
        cap = self.H * self.cap_ratio
        if len(self.alive) >= cap:
            return
        r, c = random.randint(0, self.H-1), random.randint(0, self.H-1)
        if r != c and self.mask[r, c] == 0:
            self.mask[r, c] = np.int8(1) if random.randint(0, 1) else np.int8(-1)
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
            old_sign = int(self.mask[r, c])
            self.mask[r, c] = np.int8(0)
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
                self.mask[r, c] = np.int8(0)
                self.mask[r, nc] = old
                self.alive[idx] = (r, nc)
                self.alive_set.discard((r, c))
                self.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))

    def _theta_mutate(self, undo):
        """Mutate one random neuron's threshold to a random value [0, 1]."""
        idx = random.randint(0, self.H - 1)
        old_val = float(self.theta[idx])
        self.theta[idx] = np.float32(random.random())
        undo.append(('T', idx, old_val))

    def _decay_mutate(self, undo):
        """Mutate one random neuron's decay rate to a random value [0.01, 0.5]."""
        idx = random.randint(0, self.H - 1)
        old_val = float(self.decay[idx])
        self.decay[idx] = np.float32(random.uniform(0.01, 0.5))
        undo.append(('D', idx, old_val))

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
                self.mask[r, c] = np.int8(0)
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
