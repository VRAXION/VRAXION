"""
INSTNCT — Self-Wiring Graph kernel
==================================
Hidden-only recurrent substrate with fixed passive I/O projections.
Gradient-free learning happens by mutating the binary hidden mask and the
per-neuron theta / decay vectors, then keeping only improved candidates.

Runtime contract:
  - input_projection  (V × H): fixed projection from vocab-space into hidden space
  - output_projection (H × V): fixed projection from hidden charge into vocab logits
  - mask             (H × H): learnable hidden graph, boolean (True/False)
  - theta            (H,): per-neuron firing threshold
  - decay            (H,): per-neuron decay rate
  - freq             (H,): per-neuron firing frequency (Musical Gating)
  - phase            (H,): per-neuron firing phase (Musical Gating)
  - rho              (H,): per-neuron modulation depth (Musical Gating)
  - polarity         (H,): per-neuron polarity (+1 excitatory, -1 inhibitory)
  - state            (H,): hidden activation (binary spikes) after gating
  - charge           (H,): hidden pre-threshold charge (0-15)
"""

import numpy as np
import random


class SelfWiringGraph:

    DEFAULT_HIDDEN_RATIO = 3
    DEFAULT_DENSITY = 4
    DEFAULT_EDGE_MAGNITUDE = 1.0
    DEFAULT_CAP_RATIO = 120
    DEFAULT_PROJECTION_SCALE = 3.0
    DEFAULT_THETA = 15.0
    DEFAULT_DECAY = 1.0
    MAX_CHARGE = 15.0
    DEFAULT_RHO = 0.3  # Starting modulation depth
    DEFAULT_INHIBITORY_FRACTION = 0.10  # Fly-realistic: fewer but broader I-neurons
    POLARITY_FLIP_PROB = 10  # 1-in-N chance per mutate step
    DEFAULT_INPUT_MODE = 'projection'  # 'projection' (legacy) or 'sdr'
    DEFAULT_SDR_K = 13                 # active neurons per byte (20% of sdr_dim)
    DEFAULT_SDR_DIM = 64               # SDR input dimension (legacy default)
    DEFAULT_OUTPUT_DIM = 160           # output readout dim (legacy default)
    PHI = (1 + 5**0.5) / 2            # golden ratio 1.618...
    # Phi overlap mode: in_dim = out_dim = round(H/phi), K = 20% of in_dim
    # Overlap zone = 2*(H/phi) - H neurons are both input and output
    # Validated: 20.8% at H=256 (beats non-overlap 20.0%)

    def _build_sdr_table(self, rng=None):
        """Build sparse distributed representation: 256 x sdr_dim, K active per row."""
        if rng is None:
            rng = np.random.RandomState(42)
        self.sdr_table = np.zeros((256, self.sdr_dim), dtype=np.float32)
        for v in range(256):
            active = rng.choice(self.sdr_dim, size=self.sdr_k, replace=False)
            self.sdr_table[v, active] = 1.0

    @staticmethod
    def _checkpoint_index_dtype(hidden_size):
        """Use the smallest safe integer dtype for edge indices on disk."""
        return np.uint16 if int(hidden_size) <= np.iinfo(np.uint16).max else np.uint32

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
        input_mode=None,
        sdr_k=None,
        sdr_dim=None,
        output_dim=None,
        phi_overlap=False,
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

        # Phi overlap: auto-compute I/O dims from H using golden ratio
        self.phi_overlap = phi_overlap
        if phi_overlap:
            input_mode = input_mode or 'sdr'
            phi_dim = int(round(self.H / self.PHI))
            sdr_dim = sdr_dim or phi_dim
            output_dim = output_dim or phi_dim
            sdr_k = sdr_k or max(1, int(round(phi_dim * 0.20)))

        # SDR input mode: sparse distributed byte representation
        self.input_mode = input_mode or self.DEFAULT_INPUT_MODE
        self.sdr_table = None
        self.sdr_k = sdr_k or self.DEFAULT_SDR_K
        self.sdr_dim = sdr_dim or self.DEFAULT_SDR_DIM
        if self.input_mode == 'sdr':
            self._build_sdr_table(proj_rng)

        # Tentacle output projection (output_dim × V, smaller than full H × V)
        self.output_dim = output_dim or self.DEFAULT_OUTPUT_DIM
        self._output_proj_tentacle = None
        if self.input_mode == 'sdr':
            tp = proj_rng.randn(self.output_dim, self.V).astype(np.float32)
            tp /= np.linalg.norm(tp, axis=0, keepdims=True)
            self._output_proj_tentacle = tp * self.projection_scale

        # Polarity: bool (True=excitatory, False=inhibitory). 10% inhibitory.
        self.polarity = np.ones(self.H, dtype=np.bool_)
        inhib_mask = init_rand(self.H) < self.DEFAULT_INHIBITORY_FRACTION
        self.polarity[inhib_mask] = False
        self._polarity_f32 = np.where(self.polarity, 1.0, -1.0).astype(np.float32)

        # Mask: H × H boolean (native bool, minimal dtype).
        # Fly-realistic topology: Inhibitory neurons have 2x out-degree (hub-property)
        r = init_rand(hidden, hidden)
        effective_density = np.full(hidden, self.density_fraction)
        effective_density[~self.polarity] *= 2.0
        self.mask = (r < effective_density[:, np.newaxis])  # dtype=bool, 1 byte/elem
        np.fill_diagonal(self.mask, False)
        self._mask_f32_cache = None  # lazy precomputed float32 for dense matmul

        # Alive edges: canonical row-major cache + set for O(1) undo membership.
        rows, cols = np.where(self.mask)
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

        # Refractory period: 1-tick resting state after firing.
        # Proven +7.5% win in A/B testing.
        self.refractory = np.zeros(self.H, dtype=np.int8)

        # Musical Gating: Freq, Phase and Learnable Rho
        # Every neuron has its own internal clock and sensitivity to it.
        self.freq = init_rand(self.H).astype(np.float32) * 1.5 + 0.5
        self.phase = init_rand(self.H).astype(np.float32) * 2.0 * np.pi
        self.rho = np.full(self.H, self.DEFAULT_RHO, dtype=np.float32)

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
        self.refractory *= 0

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
        """Build binary sparse cache: (rows, cols) index arrays.

        Binary mask {0, 1}: forward pass is pure addition, no multiply.
        Returns a 2-tuple (rows, cols).

        Legacy 3-tuple (rows, cols, vals) returned when edge_magnitude != 1.0
        for backward compat with old callers.
        """
        rows, cols = np.where(mask != 0)
        if edge_magnitude != 1.0:
            if len(rows) == 0:
                return (
                    np.empty(0, dtype=np.intp),
                    np.empty(0, dtype=np.intp),
                    np.empty(0, dtype=np.float32),
                )
            vals = mask[rows, cols].astype(np.float32) * np.float32(edge_magnitude)
            return rows.astype(np.intp), cols.astype(np.intp), vals
        return rows.astype(np.intp), cols.astype(np.intp)

    @staticmethod
    def _sparse_mul_1d_from_cache(H, act, sparse_cache):
        """Sparse act @ mask for 1D act vector. O(edges) instead of O(H^2).

        Binary 2-tuple (rows, cols): pure add, no multiply.
        Legacy 3-tuple (rows, cols, vals): multiply path.
        """
        raw = np.zeros(H, dtype=np.float32)
        if len(sparse_cache) == 2:
            rows, cols = sparse_cache
            if len(rows):
                np.add.at(raw, cols, act[rows])
        else:
            rows, cols, vals = sparse_cache
            if len(rows):
                np.add.at(raw, cols, act[rows] * vals)
        return raw

    @staticmethod
    def _sparse_mul_2d_from_cache(H, acts, sparse_cache):
        """Sparse acts @ mask for 2D batch. O(batch * edges) instead of O(batch * H^2).

        Binary 2-tuple: pure add. Legacy 3-tuple: multiply path.
        """
        B = acts.shape[0]
        raw = np.zeros((B, H), dtype=np.float32)
        if len(sparse_cache) == 2:
            rows, cols = sparse_cache
            if len(rows):
                np.add.at(raw, (slice(None), cols), acts[:, rows])
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
        polarity=None,
        refractory=None,
        freq=None,
        phase=None,
        rho=None,
    ):
        mask = np.asarray(mask)
        # Dense path needs float32. Accept precomputed or compute once.
        if mask.dtype == np.float32:
            mask_f32 = mask
        elif mask.dtype == np.bool_:
            # Bool → float32 only when dense path needed (sparse skips this)
            mask_f32 = None  # lazy — only computed if dense path used
        else:
            mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
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
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1

        for tick in range(int(ticks)):
            # 1. DECAY
            cur_charge = np.maximum(cur_charge - decay, 0.0)
            
            # 2. INPUT
            if tick < int(input_duration):
                act = act + injected
            
            # 3. PROPAGATE
            if use_sparse:
                raw = SelfWiringGraph._sparse_mul_1d_from_cache(H, act, sparse_cache)
            else:
                if mask_f32 is None:
                    mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
                raw = act @ mask_f32
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            cur_charge += raw
            
            # 4. CLAMP
            np.clip(cur_charge, 0.0, SelfWiringGraph.MAX_CHARGE, out=cur_charge)
            
            # 5. SPIKE DECISION with Refractory and C19 Soft-Wave Modulation
            effective_theta = theta
            if freq is not None and phase is not None:
                wave = np.sin(tick * freq + phase)
                # Use per-neuron learnable rho if provided, else use default.
                curr_rho = rho if rho is not None else SelfWiringGraph.DEFAULT_RHO
                effective_theta = np.clip(
                    theta * (1.0 + curr_rho * wave),
                    1.0, SelfWiringGraph.MAX_CHARGE
                )

            if refractory is not None:
                can_fire = (refractory == 0)
                fired = (cur_charge >= effective_theta) & can_fire
                refractory[refractory > 0] -= 1
                refractory[fired] = 1 
            else:
                fired = (cur_charge >= effective_theta)

            act = fired.astype(np.float32) # Digital spike: always 1.0
            if polarity is not None:
                act = act * polarity
            
            # 6. RESET FIRED NEURONS
            cur_charge[fired] = 0.0 # Hard reset for now, matching Int4 Brain win
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
        polarity=None,
        refractory=None,
        freq=None,
        phase=None,
        rho=None,
    ):
        mask = np.asarray(mask)
        if mask.dtype == np.float32:
            mask_f32 = mask
        elif mask.dtype == np.bool_:
            mask_f32 = None  # lazy
        else:
            mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
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
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1
        batch_refractory = np.zeros((batch, H), dtype=np.int32) if refractory is not None else None

        for tick in range(int(ticks)):
            # 1. LEAK
            cur_charges = np.maximum(cur_charges - decay, 0.0)

            # 2. INPUT
            if tick < int(input_duration):
                cur_acts = cur_acts + injected_batch

            # 3. PROPAGATE
            if use_sparse:
                raw = SelfWiringGraph._sparse_mul_2d_from_cache(H, cur_acts, sparse_cache)
            else:
                if mask_f32 is None:
                    mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude)
                raw = cur_acts @ mask_f32
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            cur_charges += raw
            
            # 4. CLAMP
            np.clip(cur_charges, 0.0, SelfWiringGraph.MAX_CHARGE, out=cur_charges)
            
            # 5. SPIKE with C19 Soft-Wave Gating
            effective_theta = theta
            if freq is not None and phase is not None:
                wave = np.sin(tick * freq + phase)
                # Use per-neuron learnable rho if provided, else use default.
                curr_rho = rho if rho is not None else SelfWiringGraph.DEFAULT_RHO
                effective_theta = np.clip(
                    theta * (1.0 + curr_rho * wave),
                    1.0, SelfWiringGraph.MAX_CHARGE
                )
            
            if refractory is not None:
                can_fire = (batch_refractory == 0)
                fired = (cur_charges >= effective_theta) & can_fire
                batch_refractory[batch_refractory > 0] -= 1
                batch_refractory[fired] = 1
            else:
                fired = (cur_charges >= effective_theta)
            cur_acts = fired.astype(np.float32)
            if polarity is not None:
                cur_acts = cur_acts * polarity
            
            # 6. RESET
            cur_charges[fired] = 0.0
        return cur_acts, cur_charges

    def readout(self, vec):
        """Project one state/charge vector into output-logit space."""
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (self.H,):
            raise ValueError(f"readout expects shape {(self.H,)}, got {v.shape}")
        if self.input_mode == 'sdr' and self._output_proj_tentacle is not None:
            out_slice = v[self.H - self.output_dim:]
            return out_slice @ self._output_proj_tentacle
        return v @ self.output_projection

    def readout_batch(self, vecs):
        """Project a batch of state/charge vectors into output-logit space."""
        v = np.asarray(vecs, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.H:
            raise ValueError(
                f"readout_batch expects shape (batch, {self.H}), got {v.shape}"
            )
        if self.input_mode == 'sdr' and self._output_proj_tentacle is not None:
            out_slice = v[:, self.H - self.output_dim:]
            return out_slice @ self._output_proj_tentacle
        return v @ self.output_projection

    def forward(self, world, ticks=12):
        """Single-input forward pass. Passive I/O: inject via input_projection, read via output_projection."""
        world_vec = np.asarray(world, dtype=np.float32).copy()
        np.nan_to_num(world_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.input_mode == 'sdr' and self.sdr_table is not None:
            byte_idx = int(np.argmax(world_vec))
            injected = np.zeros(self.H, dtype=np.float32)
            injected[0:self.sdr_dim] = self.sdr_table[byte_idx]
        else:
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
            polarity=self._polarity_f32,
            refractory=self.refractory,
            freq=self.freq,
            phase=self.phase,
            rho=self.rho,
        )
        # SDR mode: charge readout (validated 14.1% vs 10.3% state)
        # Legacy mode: state readout (original design)
        if self.input_mode == 'sdr':
            return self.readout(self.charge)
        return self.readout(self.state)

    def forward_batch(self, ticks=12):
        """Batch forward: all V inputs simultaneously. Returns (V, V) logits."""
        V = self.V
        if self.input_mode == 'sdr' and self.sdr_table is not None:
            projected = np.zeros((V, self.H), dtype=np.float32)
            projected[:, 0:self.sdr_dim] = self.sdr_table[:V]
        else:
            projected = np.eye(V, dtype=np.float32) @ self.input_projection
        acts, charges = self.rollout_token_batch(
            projected,
            mask=self.mask,
            theta=self.theta,
            decay=self.decay,
            ticks=ticks,
            sparse_cache=self._sp_cache,
            edge_magnitude=self.edge_magnitude,
            polarity=self._polarity_f32,
            refractory=self.refractory,
            freq=self.freq,
            phase=self.phase,
            rho=self.rho,
        )
        # SDR mode: charge readout; Legacy: spike readout
        if self.input_mode == 'sdr':
            return self.readout_batch(charges)
        return self.readout_batch(acts)

    def resync_alive(self):
        """Rebuild alive list+set from mask. Call after direct mask writes."""
        rows, cols = np.where(self.mask)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)
        self._mask_f32_cache = None  # invalidate dense matmul cache
        self._sync_sparse_idx()

    def _sync_sparse_idx(self):
        """Precompute binary sparse cache for multiply-free forward pass.

        Binary mask {0, 1}: stores (rows, cols) only. Forward pass is pure
        addition — no sign, no magnitude, no float multiply.
        """
        if self.alive:
            rows = np.array([r for r, c in self.alive], dtype=np.intp)
            cols = np.array([c for r, c in self.alive], dtype=np.intp)
            self._sp_cache = (rows, cols)
            # Legacy accessors (backward compat for external callers)
            self._sp_rows = rows
            self._sp_cols = cols
        else:
            self._sp_cache = (
                np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
            )
            self._sp_rows = np.empty(0, dtype=np.intp)
            self._sp_cols = np.empty(0, dtype=np.intp)

    def count_connections(self):
        return len(self.alive)

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
            'polarity': self.polarity.copy(),
            'freq': self.freq.copy(),
            'phase': self.phase.copy(),
            'rho': self.rho.copy(),
        }

    def restore_state(self, s):
        """Revert all state including mutation drive."""
        saved_mask = s['mask']
        if saved_mask.dtype != self.mask.dtype:
            saved_mask = (saved_mask != 0)
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
        if 'polarity' in s:
            raw_pol = s['polarity']
            self.polarity[:] = raw_pol if raw_pol.dtype == np.bool_ else (raw_pol > 0)
            self._polarity_f32[:] = np.where(self.polarity, 1.0, -1.0)
        if 'freq' in s:
            self.freq[:] = s['freq']
        if 'phase' in s:
            self.phase[:] = s['phase']
        if 'rho' in s:
            self.rho[:] = s['rho']

    # --- Breeding / crossover ---

    @classmethod
    def breed(cls, parent_a, parent_b, *, fitness_a=None, fitness_b=None, seed=None):
        """Breed two parent graphs via binary union.

        Child gets every edge that either parent has. Binary mask {0, 1}:
        no sign, no magnitude — topology is the only learnable structure.

        Parameters
        ----------
        parent_a, parent_b : SelfWiringGraph
            Parents with identical V, H, and projections.
        fitness_a, fitness_b : float, optional
            Accepted for API compat; unused in binary union.
        seed : int, optional
            RNG seed (accepted for API compat; union is deterministic).
        """
        if parent_a.V != parent_b.V or parent_a.H != parent_b.H:
            raise ValueError("Parents must have same V and H")
        if not np.array_equal(parent_a.input_projection, parent_b.input_projection):
            raise ValueError("Parents must share the same input projection")

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

        # Binary union: child has edge wherever either parent has one
        child.mask = ((parent_a.mask != 0) | (parent_b.mask != 0))  # bool dtype
        np.fill_diagonal(child.mask, False)
        child._mask_f32_cache = None
        child.resync_alive()

        # Polarity: random mix from parents
        child.polarity = parent_a.polarity.copy()
        mix_mask = np.random.rand(child.H) < 0.5
        child.polarity[mix_mask] = parent_b.polarity[mix_mask]
        child._polarity_f32 = np.where(child.polarity, 1.0, -1.0).astype(np.float32)

        # Average meta-params with small noise
        child.theta = ((parent_a.theta + parent_b.theta) / 2.0).astype(np.float32)
        child.decay = ((parent_a.decay + parent_b.decay) / 2.0).astype(np.float32)
        child.freq = ((parent_a.freq + parent_b.freq) / 2.0).astype(np.float32)
        child.phase = ((parent_a.phase + parent_b.phase) / 2.0).astype(np.float32)
        child.rho = ((parent_a.rho + parent_b.rho) / 2.0).astype(np.float32)
        child.loss_pct = np.int8((int(parent_a.loss_pct) + int(parent_b.loss_pct)) // 2)
        child.mutation_drive = np.int8((int(parent_a.mutation_drive) + int(parent_b.mutation_drive)) // 2)

        # SDR + tentacle mode propagation
        child.input_mode = parent_a.input_mode
        child.sdr_k = parent_a.sdr_k
        child.sdr_dim = parent_a.sdr_dim
        child.sdr_table = parent_a.sdr_table.copy() if parent_a.sdr_table is not None else None
        child.output_dim = parent_a.output_dim
        child._output_proj_tentacle = parent_a._output_proj_tentacle.copy() if parent_a._output_proj_tentacle is not None else None
        child.phi_overlap = parent_a.phi_overlap

        # Fresh state
        child.state = np.zeros(child.H, dtype=np.float32)
        child.charge = np.zeros(child.H, dtype=np.float32)

        return child

    def save(self, path):
        """Save winner graph to disk (.npz) with exact forward-path projections."""
        rows, cols = np.where(self.mask)
        index_dtype = self._checkpoint_index_dtype(self.H)
        vals = np.ones(rows.shape[0], dtype=np.bool_)
        np.savez_compressed(path,
            schema_version=np.int16(4),
            V=self.V,
            H=self.H,
            hidden_ratio=np.int32(self.hidden_ratio),
            rows=rows.astype(index_dtype, copy=False),
            cols=cols.astype(index_dtype, copy=False),
            vals=vals,
            loss_pct=int(self.loss_pct),
            mutation_drive=int(self.mutation_drive),
            theta=self.theta,
            decay=self.decay,
            polarity=np.where(self.polarity, np.int8(1), np.int8(-1)),  # save as int8 for compat
            freq=self.freq,
            phase=self.phase,
            rho=self.rho,
            projection_scale=np.float32(self.projection_scale),
            edge_magnitude=np.float32(self.edge_magnitude),
            cap_ratio=np.int32(self.cap_ratio),
            input_projection=self.input_projection,
            output_projection=self.output_projection,
            **(  # SDR + tentacle fields (optional, backward compatible)
                {'input_mode': np.array([ord(c) for c in self.input_mode], dtype=np.uint8),
                 'sdr_table': self.sdr_table,
                 'sdr_k': np.int32(self.sdr_k),
                 'sdr_dim': np.int32(self.sdr_dim),
                 'output_dim': np.int32(self.output_dim),
                 'output_proj_tentacle': self._output_proj_tentacle,
                 'phi_overlap': np.bool_(self.phi_overlap)}
                if self.input_mode == 'sdr' and self.sdr_table is not None
                else {}
            ),
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
            # Backward compat: old checkpoints stored float or int8 ternary vals.
            # Convert any nonzero value to 1 (binary mask).
            vals = (raw_vals != 0)
            input_projection = np.array(d['input_projection'], dtype=np.float32)
            output_projection = np.array(d['output_projection'], dtype=np.float32)
            theta = np.array(d['theta'], dtype=np.float32)
            decay = np.array(d['decay'], dtype=np.float32)
            if 'freq' in d.files:
                freq = np.array(d['freq'], dtype=np.float32)
                phase = np.array(d['phase'], dtype=np.float32)
                rho = np.array(d['rho'], dtype=np.float32) if 'rho' in d.files else np.full(H, cls.DEFAULT_RHO, dtype=np.float32)
            else:
                # Legacy: default 1.0 freq (always top phase)
                freq = np.ones(H, dtype=np.float32)
                phase = np.zeros(H, dtype=np.float32)
                rho = np.zeros(H, dtype=np.float32) # No wave modulation for legacy

            if 'polarity' in d.files:
                raw_pol = np.array(d['polarity'])
                polarity = (raw_pol > 0) if raw_pol.dtype != np.bool_ else raw_pol
            else:
                polarity = np.ones(H, dtype=np.bool_)
            loss_pct = np.int8(int(d['loss_pct']))
            mutation_drive = np.int8(int(d['mutation_drive'])) if 'mutation_drive' in d.files else np.int8(int(d['drive']))
            hidden_ratio = int(d['hidden_ratio']) if 'hidden_ratio' in d.files else max(1, H // max(V, 1))
            projection_scale = np.float32(d['projection_scale']) if 'projection_scale' in d.files else np.float32(cls.DEFAULT_PROJECTION_SCALE)
            edge_magnitude = np.float32(d['edge_magnitude']) if 'edge_magnitude' in d.files else np.float32(cls.DEFAULT_EDGE_MAGNITUDE)
            cap_ratio = int(d['cap_ratio']) if 'cap_ratio' in d.files else int(cls.DEFAULT_CAP_RATIO)

            # Extract SDR fields while npz is still open (optional, backward compatible)
            _sdr_input_mode = ''.join(chr(c) for c in d['input_mode']) if 'input_mode' in d else 'projection'
            _sdr_table = np.array(d['sdr_table'], dtype=np.float32) if 'sdr_table' in d else None
            _sdr_k = int(d['sdr_k']) if 'sdr_k' in d else cls.DEFAULT_SDR_K
            _sdr_dim = int(d['sdr_dim']) if 'sdr_dim' in d else cls.DEFAULT_SDR_DIM
            _output_dim = int(d['output_dim']) if 'output_dim' in d else cls.DEFAULT_OUTPUT_DIM
            _output_proj_tentacle = np.array(d['output_proj_tentacle'], dtype=np.float32) if 'output_proj_tentacle' in d else None
            _phi_overlap = bool(d['phi_overlap']) if 'phi_overlap' in d else False

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
        net.mask = np.zeros((H, H), dtype=np.bool_)
        net.mask[rows, cols] = vals
        np.fill_diagonal(net.mask, False)
        net._mask_f32_cache = None
        net.state = np.zeros(H, dtype=np.float32)
        net.charge = np.zeros(H, dtype=np.float32)
        net.loss_pct = loss_pct
        net.mutation_drive = mutation_drive
        net.theta = theta
        net.decay = decay
        net.freq = freq
        net.phase = phase
        net.rho = rho
        net.polarity = polarity.astype(np.bool_) if polarity.dtype != np.bool_ else polarity
        net._polarity_f32 = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
        net.refractory = np.zeros(H, dtype=np.int8)

        # SDR mode (from locals extracted inside the with-block above)
        net.input_mode = _sdr_input_mode
        net.sdr_table = _sdr_table
        net.sdr_k = _sdr_k
        net.sdr_dim = _sdr_dim
        net.output_dim = _output_dim
        net._output_proj_tentacle = _output_proj_tentacle
        net.phi_overlap = _phi_overlap

        net.resync_alive()
        return net

    def replay(self, log):
        """Repeat logged ops in reverse = undo. O(changes) + O(alive) list rebuild.
        Structural ops update mask + set, list rebuilt once."""
        has_structural = False
        for entry in reversed(log):
            op = entry[0]
            if op == 'A':
                r, c = entry[1], entry[2]
                self.mask[r, c] = False
                self.alive_set.discard((r, c))
                has_structural = True
            elif op == 'R':
                r, c = entry[1], entry[2]
                self.mask[r, c] = True
                self.alive_set.add((r, c))
                has_structural = True
            elif op == 'W':
                _, r, c_old, c_new = entry
                self.mask[r, c_new] = False
                self.mask[r, c_old] = True
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
            elif op == 'P':
                old_val = entry[2]
                self.polarity[entry[1]] = bool(old_val > 0) if not isinstance(old_val, bool) else old_val
                self._polarity_f32[entry[1]] = 1.0 if self.polarity[entry[1]] else -1.0
            elif op == 'FR':
                self.freq[entry[1]] = np.float32(entry[2])
            elif op == 'PH':
                self.phase[entry[1]] = np.float32(entry[2])
            elif op == 'RH':
                self.rho[entry[1]] = np.float32(entry[2])
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
                'add_loop': self._add_loop,
                'remove': self._remove,
                'rewire': self._rewire,
                'flip': self._flip,
                'theta': self._theta_mutate,
                'decay': self._decay_mutate,
                'polarity': self._polarity_mutate,
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

        # Polarity flip drift: 1-in-N chance, reverts on reject
        if random.randint(1, self.POLARITY_FLIP_PROB) == 1 and not freeze_params:
            self._polarity_mutate(undo)

        # Musical Gating drift: 1-in-10 chance
        if random.randint(1, 10) == 1 and not freeze_params:
            idx = random.randint(0, self.H - 1)
            if random.random() < 0.5:
                undo.append(('FR', idx, float(self.freq[idx])))
                self.freq[idx] = np.float32(random.uniform(0.5, 2.0))
            else:
                undo.append(('PH', idx, float(self.phase[idx])))
                self.phase[idx] = np.float32(random.uniform(0, 2*np.pi))

        # Rho (modulation depth) drift: 1-in-10 chance
        if random.randint(1, 10) == 1 and not freeze_params:
            idx = random.randint(0, self.H - 1)
            undo.append(('RH', idx, float(self.rho[idx])))
            self.rho[idx] = np.clip(self.rho[idx] + random.uniform(-0.1, 0.1), 0.0, 1.0)

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
        
        # Fly-realistic bias: if source is excitatory, maybe skip adding 
        # to maintain the 2x ratio for inhibitory neurons.
        if self.polarity[r] == 1 and random.random() < 0.5:
            return

        if r != c and self.mask[r, c] == 0:
            self.mask[r, c] = True
            self.alive.append((r, c))
            self.alive_set.add((r, c))
            undo.append(('A', r, c))

    def _flip(self, undo):
        """Legacy flip: rewire to a random target (sign flip is meaningless in binary)."""
        self._rewire(undo)

    def _remove(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            old_val = bool(self.mask[r, c])
            self.mask[r, c] = False
            self.alive[idx] = self.alive[-1]
            self.alive.pop()
            self.alive_set.discard((r, c))
            undo.append(('R', r, c, old_val))

    def _rewire(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            nc = random.randint(0, self.H-1)
            if nc != r and nc != c and self.mask[r, nc] == 0:
                self.mask[r, c] = False
                self.mask[r, nc] = True
                self.alive[idx] = (r, nc)
                self.alive_set.discard((r, c))
                self.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))

    def _theta_mutate(self, undo):
        """Mutate one random neuron's threshold to a value in [1, MAX_CHARGE]."""
        idx = random.randint(0, self.H - 1)
        old_val = float(self.theta[idx])
        self.theta[idx] = np.float32(random.randint(1, int(self.MAX_CHARGE)))
        undo.append(('T', idx, old_val))

    def _decay_mutate(self, undo):
        """Mutate one random neuron's decay rate to a value in [0, 2]."""
        idx = random.randint(0, self.H - 1)
        old_val = float(self.decay[idx])
        # Simple integer-like decay steps for int4 dynamics
        self.decay[idx] = np.float32(random.choice([0.5, 1.0, 2.0]))
        undo.append(('D', idx, old_val))

    def _add_loop(self, undo, max_len=6):
        """Add a complete loop of random length [2, max_len].

        Picks a random start neuron, chains through random distinct neurons,
        then closes the loop back to start.  All edges are added atomically.
        If any edge in the chain already exists or nodes collide, bail out.
        """
        cap = self.H * self.cap_ratio
        loop_len = random.randint(2, max(2, max_len))
        if len(self.alive) + loop_len > cap:
            return
        nodes = [random.randint(0, self.H - 1)]
        for _ in range(loop_len - 1):
            n = random.randint(0, self.H - 1)
            if n in nodes:
                return  # collision, bail
            nodes.append(n)
        # Check all edges are free
        edges = []
        for i in range(loop_len):
            r, c = nodes[i], nodes[(i + 1) % loop_len]
            if self.mask[r, c] != 0:
                return  # edge exists, bail
            edges.append((r, c))
        # Commit all edges atomically
        for r, c in edges:
            self.mask[r, c] = True
            self.alive.append((r, c))
            self.alive_set.add((r, c))
            undo.append(('A', r, c))

    def _polarity_mutate(self, undo):
        """Flip one random neuron's polarity."""
        idx = random.randint(0, self.H - 1)
        old_pol = bool(self.polarity[idx])
        self.polarity[idx] = ~self.polarity[idx]  # bool NOT
        self._polarity_f32[idx] = 1.0 if self.polarity[idx] else -1.0
        undo.append(('P', idx, old_pol))

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
                self.mask[r, c] = False
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
