"""
Swarm Byte Ring Model

Key innovation: N independent beings sharing ONE ring memory.
Each being has its own pointer and hidden state, but all write to shared memory.

Hypothesis: Multiple weak learners with spatial diversity can match/exceed single strong learner.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import math


def fibonacci_split(num_bits: int, min_k: int = 4) -> list:
    """
    Auto-determine beings and K values from Fibonacci halving.

    Each being gets half of remaining bits, down to min_k.
    Number of beings emerges naturally from the split.

    Returns:
        List of K values (descending). len(result) = number of beings.
    """
    k_values = []
    remaining = num_bits
    while remaining >= min_k * 2:
        k = remaining // 2
        k_values.append(k)
        remaining -= k
    if remaining > 0:
        k_values.append(remaining)
    return k_values


def fibonacci_k_schedule(num_beings: int, num_bits: int, min_k: int = 2) -> list:
    """
    Generate Fibonacci-halving K values for a heterogeneous swarm.

    Being 0 gets the most bits (up to num_bits/2), each subsequent being
    gets roughly half, down to min_k. Remaining beings get min_k.

    Args:
        num_beings: Number of beings.
        num_bits: Total input/output bits.
        min_k: Minimum bits per being (default 2).

    Returns:
        List of K values, one per being (descending order).
    """
    k_values = []
    remaining = num_bits
    for i in range(num_beings):
        k = max(min_k, remaining // 2)
        k = min(k, num_bits)  # Can't see more than total
        k_values.append(k)
        remaining = remaining - k
        if remaining < min_k:
            remaining = num_bits  # Reset for next "octave"
    return k_values


def fibonacci_tick_periods(k_values: list = None, num_beings: int = 0) -> list:
    """Map K values (or being indices) to Fibonacci tick periods.

    Larger K -> longer period (fires less often, thinks deeply).
    Smaller K -> period=1 (fires every tick, vibrates fast).

    Args:
        k_values: List of K values per being (from fibonacci_k_schedule).
                  If None, uses num_beings with cycling fibonacci assignment.
        num_beings: Number of beings (used when k_values is None for flat swarm).

    Returns:
        List of tick periods, one per being. Period N means "fire every N ticks".
    """
    FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    if k_values is not None:
        unique_ks = sorted(set(k_values), reverse=True)  # largest first
        n = len(unique_ks)
        # Largest K -> largest fib period, smallest K -> period 1
        fib_map = [FIB[min(i + 1, len(FIB) - 1)] for i in range(n)]  # ascending: [1,2,3,5,8,13,21,...]
        k_to_period = {k: fib_map[n - 1 - i] for i, k in enumerate(unique_ks)}
        return [k_to_period[k] for k in k_values]
    else:
        # Flat swarm: cycle fibonacci by being index
        n = min(num_beings, len(FIB))
        return [FIB[min((num_beings - 1 - i) % n + 1, len(FIB) - 1)] for i in range(num_beings)]


def fibonacci_hidden_dims(k_values: list, max_hidden: int = 4096, min_hidden: int = 128) -> list:
    """Compute per-being hidden dims from K values. Largest K gets max_hidden, scales linearly.

    Args:
        k_values: List of K values per being (from fibonacci_k_schedule).
        max_hidden: Hidden dim for the largest being (whale).
        min_hidden: Floor hidden dim for the smallest being (ant).

    Returns:
        List of H_i values, one per being.
    """
    k_max = max(k_values)
    dims = []
    for k in k_values:
        h = int(max_hidden * k / k_max)
        h = max(min_hidden, h)
        h = ((h + 15) // 16) * 16  # Round to 16 for GPU tensor core alignment
        dims.append(h)
    return dims


def generate_receptive_masks(
    num_beings: int,
    num_bits: int = 8,
    bits_per_being: int = 4,
    min_coverage: int = 2,
    max_coverage: int = 0,
    seed: int = 42,
    fibonacci: bool = False,
) -> torch.Tensor:
    """
    Generate random binary masks ensuring every bit has min_coverage beings.

    Each being gets a random subset of bits as its receptive field.
    Coverage repair ensures no bit is left uncovered.

    Args:
        num_beings: Number of beings in the swarm.
        num_bits: Total input/output bits (default 8).
        bits_per_being: How many bits each being sees (uniform mode).
        min_coverage: Every bit must be covered by at least this many beings.
        max_coverage: No bit covered by more than this many beings (0 = auto).
        seed: Random seed for reproducibility.
        fibonacci: If True, use Fibonacci-halving K schedule instead of uniform.

    Returns:
        masks: [num_beings, num_bits] binary tensor (1 = being reads/writes this bit).
    """
    # Compute per-being K values
    if fibonacci:
        k_values = fibonacci_k_schedule(num_beings, num_bits, min_k=4)
    else:
        k_values = [bits_per_being] * num_beings

    total_slots = sum(k_values)
    required_slots = num_bits * min_coverage
    if total_slots < required_slots:
        raise ValueError(
            f"Cannot satisfy min_coverage={min_coverage}: "
            f"total slots = {total_slots}, "
            f"but need {num_bits} x {min_coverage} = {required_slots}"
        )

    gen = torch.Generator()
    gen.manual_seed(seed)

    # Special case: exact 1:1 mapping (K=1, num_beings >= num_bits)
    if bits_per_being == 1 and num_beings >= num_bits and min_coverage == 1 and not fibonacci:
        masks = torch.zeros(num_beings, num_bits)
        perm = torch.randperm(num_bits, generator=gen)
        for i in range(num_beings):
            masks[i, perm[i % num_bits]] = 1.0
        return masks

    masks = torch.zeros(num_beings, num_bits)
    for i in range(num_beings):
        perm = torch.randperm(num_bits, generator=gen)[:k_values[i]]
        masks[i, perm] = 1.0

    # Coverage repair: ensure every bit has >= min_coverage beings
    for _ in range(1000):
        coverage = masks.sum(dim=0)  # [num_bits]
        under = (coverage < min_coverage).nonzero(as_tuple=True)[0]
        if len(under) == 0:
            break
        # Pick an under-covered bit
        bit_fix = under[0].item()
        # Find beings that DON'T cover this bit
        uncovering = (masks[:, bit_fix] < 0.5).nonzero(as_tuple=True)[0]
        if len(uncovering) == 0:
            break
        # Pick a random uncovering being
        being_fix = uncovering[torch.randint(len(uncovering), (1,), generator=gen).item()].item()
        # Find this being's bits that have excess coverage (> min_coverage)
        being_bits = (masks[being_fix] > 0.5).nonzero(as_tuple=True)[0]
        excess = [b.item() for b in being_bits if coverage[b.item()] > min_coverage]
        if excess:
            # Swap: remove an excess bit, add the under-covered bit
            drop = excess[torch.randint(len(excess), (1,), generator=gen).item()]
            masks[being_fix, drop] = 0.0
            masks[being_fix, bit_fix] = 1.0
        else:
            # No excess to swap, just add (being gets bits_per_being + 1)
            masks[being_fix, bit_fix] = 1.0

    # Phase 2: Drain over-covered bits for balanced coverage
    avg_k = sum(k_values) / len(k_values)
    max_cov = max_coverage if max_coverage > 0 else int(num_beings * avg_k / num_bits + 1)
    for _ in range(1000):
        coverage = masks.sum(dim=0)
        over = (coverage > max_cov).nonzero(as_tuple=True)[0]
        if len(over) == 0:
            break
        bit_drain = over[0].item()
        covering = (masks[:, bit_drain] > 0.5).nonzero(as_tuple=True)[0]
        if len(covering) == 0:
            break
        being_drain = covering[torch.randint(len(covering), (1,), generator=gen).item()].item()
        under = (coverage < max_cov).nonzero(as_tuple=True)[0]
        candidates = [u.item() for u in under if masks[being_drain, u.item()] < 0.5]
        if candidates:
            add_bit = candidates[torch.randint(len(candidates), (1,), generator=gen).item()]
            masks[being_drain, bit_drain] = 0.0
            masks[being_drain, add_bit] = 1.0
        else:
            break

    # Final check
    coverage = masks.sum(dim=0)
    if (coverage < min_coverage).any():
        raise RuntimeError(
            f"Coverage repair failed. Coverage: {coverage.int().tolist()}"
        )

    return masks


def generate_combinatorial_masks(
    num_beings: int,
    num_bits: int = 256,
    bits_per_being: int = 0,
    min_coverage: int = 2,
    max_coverage: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate masks maximizing pairwise bit-combination diversity.

    Greedy algorithm: each new being picks bits that create the most
    NEW (bit_i, bit_j) pairs not yet covered by existing beings.
    All beings get the same K bits (uniform size).

    Args:
        num_beings: Number of beings in the swarm.
        num_bits: Total input/output bits.
        bits_per_being: Bits per being (0 = auto: round(num_bits / phi)).
        min_coverage: Every bit must be covered by at least this many beings.
        max_coverage: No bit covered by more than this many beings (0 = auto).
        seed: Random seed for deterministic tiebreaking.

    Returns:
        masks: [num_beings, num_bits] binary tensor.
    """
    PHI = (1 + math.sqrt(5)) / 2
    K = bits_per_being if bits_per_being > 0 else round(num_bits / PHI)
    K = max(2, min(K, num_bits))

    gen = torch.Generator()
    gen.manual_seed(seed)

    pair_coverage = torch.zeros(num_bits, num_bits, dtype=torch.long)
    bit_coverage = torch.zeros(num_bits, dtype=torch.long)
    masks = torch.zeros(num_beings, num_bits)

    for b in range(num_beings):
        selected = []
        available = list(range(num_bits))

        # First bit: least-covered, deterministic tiebreak
        min_cov = bit_coverage[available].min().item()
        candidates = [j for j in available if bit_coverage[j].item() == min_cov]
        # Deterministic tiebreak via hash
        tiebreak = torch.zeros(len(candidates))
        for ci, c in enumerate(candidates):
            tiebreak[ci] = ((seed * 7 + b * PHI + c * PHI * PHI) % 1.0)
        first = candidates[tiebreak.argmax().item()]
        selected.append(first)
        available.remove(first)

        # Greedy: pick bits maximizing new pair coverage
        for _ in range(K - 1):
            best_score = -1.0
            best_idx = 0

            for idx, bit in enumerate(available):
                # Count new pairs with already-selected bits
                new_pairs = 0
                for s in selected:
                    lo, hi = min(bit, s), max(bit, s)
                    if pair_coverage[lo, hi] == 0:
                        new_pairs += 1

                # Score: new pairs (primary), least covered (secondary), hash tiebreak
                score = (new_pairs * 10000.0
                         - bit_coverage[bit].item()
                         + ((seed * 13 + b * PHI + bit * PHI * PHI) % 1.0) * 0.001)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            chosen = available.pop(best_idx)
            selected.append(chosen)

        # Update masks and coverage
        for i in selected:
            masks[b, i] = 1.0
            bit_coverage[i] += 1
            for j in selected:
                if i != j:
                    pair_coverage[min(i, j), max(i, j)] += 1

    # Coverage repair: ensure every bit has >= min_coverage beings
    for _ in range(1000):
        cov = masks.sum(dim=0)
        under = (cov < min_coverage).nonzero(as_tuple=True)[0]
        if len(under) == 0:
            break
        bit_fix = under[0].item()
        uncovering = (masks[:, bit_fix] < 0.5).nonzero(as_tuple=True)[0]
        if len(uncovering) == 0:
            break
        being_fix = uncovering[torch.randint(len(uncovering), (1,), generator=gen).item()].item()
        being_bits = (masks[being_fix] > 0.5).nonzero(as_tuple=True)[0]
        excess = [bb.item() for bb in being_bits if cov[bb.item()] > min_coverage]
        if excess:
            drop = excess[torch.randint(len(excess), (1,), generator=gen).item()]
            masks[being_fix, drop] = 0.0
            masks[being_fix, bit_fix] = 1.0
        else:
            masks[being_fix, bit_fix] = 1.0

    # Drain over-covered bits
    avg_k = K
    max_cov = max_coverage if max_coverage > 0 else int(num_beings * avg_k / num_bits + 1)
    for _ in range(1000):
        cov = masks.sum(dim=0)
        over = (cov > max_cov).nonzero(as_tuple=True)[0]
        if len(over) == 0:
            break
        bit_drain = over[0].item()
        covering = (masks[:, bit_drain] > 0.5).nonzero(as_tuple=True)[0]
        if len(covering) == 0:
            break
        being_drain = covering[torch.randint(len(covering), (1,), generator=gen).item()].item()
        under_bits = (cov < max_cov).nonzero(as_tuple=True)[0]
        swap_candidates = [u.item() for u in under_bits if masks[being_drain, u.item()] < 0.5]
        if swap_candidates:
            add_bit = swap_candidates[torch.randint(len(swap_candidates), (1,), generator=gen).item()]
            masks[being_drain, bit_drain] = 0.0
            masks[being_drain, add_bit] = 1.0
        else:
            break

    # Final coverage check
    final_cov = masks.sum(dim=0)
    if (final_cov < min_coverage).any():
        raise RuntimeError(
            f"Coverage repair failed. Coverage: {final_cov.int().tolist()}"
        )

    # Diagnostic: pair coverage stats
    total_pairs = num_bits * (num_bits - 1) // 2
    covered_pairs = int((pair_coverage > 0).sum().item()) // 2
    print(f"  Combinatorial masks: {covered_pairs}/{total_pairs} pairs covered "
          f"({100 * covered_pairs / max(total_pairs, 1):.1f}%), K={K}, {num_beings} beings")

    return masks


class BeingParameters(nn.Module):
    """
    Per-being parameters (pointer destinations, jump gate, context strength).
    Each being in the swarm has its own instance of these.

    Phase embedding based on golden ratio (φ) for non-interfering specialization.
    Optional input_mask for receptive field restriction.
    """

    def __init__(self, num_memory_positions: int, embedding_dim: int,
                 being_id: int = 0, input_mask: torch.Tensor = None,
                 hidden_dim: int = None):
        super().__init__()
        hdim = hidden_dim if hidden_dim is not None else embedding_dim
        self.hidden_dim = hdim

        # Learned jump destinations for each memory position
        self.pointer_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Jump gate: decides whether to jump or walk
        self.jump_gate = nn.Linear(hdim, 1)
        nn.init.constant_(self.jump_gate.bias, 0.5)  # Start jumping (prevents dead gate)

        # Context strength: how much to weight memory reads
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        # GOLDEN RATIO PHASE EMBEDDING (fixed, not trainable)
        # φ = 1.618033988749895 creates maximal dispersion
        # Each being gets unique phase: being_id × φ
        # Encoded as sin/cos harmonics for smooth, non-interfering patterns
        PHI = 1.618033988749895
        phase = being_id * PHI

        # Create multi-harmonic embedding (16 dimensions)
        # Uses harmonics 1×, 2×, 3×, 4×, 5×, 6×, 7×, 8× of base phase
        phase_embedding = []
        for harmonic in range(1, 9):  # 8 harmonics × 2 (sin+cos) = 16D
            phase_embedding.append(math.sin(harmonic * phase))
            phase_embedding.append(math.cos(harmonic * phase))

        # Register as buffer (not trainable parameter)
        self.register_buffer(
            'phase_embedding',
            torch.tensor(phase_embedding, dtype=torch.float32)
        )

        # Receptive field mask: which bits this being reads/writes
        if input_mask is not None:
            self.register_buffer('input_mask', input_mask.clone())
        else:
            self.register_buffer('input_mask', None)

        # Store being ID for reference
        self.being_id = being_id


class SwarmByteRingModel(nn.Module):
    """
    Swarm of N independent beings sharing a single ring memory.

    Shared components (all beings use these):
      - input_proj (8 → embedding_dim)
      - output_proj (embedding_dim → 8)
      - processing_layers (depth-1 additional layers)

    Per-being components (each being has its own):
      - pointer_destinations (learned jump targets)
      - jump_gate (decides when to jump vs walk)
      - context_strength (scales memory read influence)
      - pointer_position (runtime state)
      - hidden_state (runtime state)

    Output: Combined being outputs (mean or ring-attention gated)
    """

    def __init__(
        self,
        num_memory_positions: int = 64,
        embedding_dim: int = 64,
        num_beings: int = 2,
        depth: int = 2,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
        combiner_mode: str = 'mean',
        num_bits: int = 8,
        bits_per_being: int = 0,
        min_coverage: int = 2,
        mask_seed: int = 42,
        fibonacci: bool = False,
        combinatorial: bool = False,
        think_ticks: int = 0,
        temporal_fibonacci: bool = False,
        capacity_fibonacci: bool = False,
        max_hidden: int = 4096,
        min_hidden: int = 128,
        full_view: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_memory_positions = num_memory_positions
        self.num_beings = num_beings
        self.num_bits = num_bits
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        self.depth = depth
        self.combiner_mode = combiner_mode
        self.think_ticks = think_ticks
        self.full_view = full_view
        self.combinatorial = combinatorial
        self._pair_coverage_frac = None

        if combinatorial and capacity_fibonacci:
            raise ValueError("combinatorial uses uniform sizes -- capacity_fibonacci is incompatible")

        # Auto-derive K from golden ratio for combinatorial mode
        PHI = (1 + math.sqrt(5)) / 2
        if combinatorial and bits_per_being == 0:
            bits_per_being = round(num_bits / PHI)
            print(f"  Combinatorial auto: K = round({num_bits}/phi) = {bits_per_being}")

        self.bits_per_being = bits_per_being
        self.min_coverage = min_coverage

        # SHARED COMPONENTS (all beings use these)
        if embedding_dim > num_bits:
            self.input_proj = nn.Linear(num_bits * 2, embedding_dim)  # *2 for [input + GEM]
            self.output_proj = nn.Linear(embedding_dim, num_bits)
        else:
            self.input_proj = None
            self.output_proj = None

        # Multi-layer processing (if depth > 1)
        if depth > 1:
            self.processing_layers = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(depth - 1)
            ])
        else:
            self.processing_layers = None

        # LayerNorm before inter-timestep tanh to prevent saturation
        self.state_norm = nn.LayerNorm(embedding_dim)

        # RECEPTIVE FIELD MASKS (if bits_per_being > 0)
        if bits_per_being > 0:
            if combinatorial:
                masks = generate_combinatorial_masks(
                    num_beings=num_beings,
                    num_bits=num_bits,
                    bits_per_being=bits_per_being,
                    min_coverage=min_coverage,
                    seed=mask_seed,
                )
                # Compute pair coverage fraction once
                pc = torch.zeros(num_bits, num_bits)
                for i in range(num_beings):
                    active = masks[i].nonzero(as_tuple=True)[0]
                    for ai in range(len(active)):
                        for bi in range(ai + 1, len(active)):
                            pc[active[ai], active[bi]] = 1
                total_pairs = num_bits * (num_bits - 1) // 2
                self._pair_coverage_frac = float((pc > 0).sum().item()) / max(total_pairs, 1)
            else:
                masks = generate_receptive_masks(
                    num_beings=num_beings,
                    num_bits=num_bits,
                    bits_per_being=bits_per_being,
                    min_coverage=min_coverage,
                    seed=mask_seed,
                    fibonacci=fibonacci,
                )
            self.register_buffer('receptive_masks', masks)  # [num_beings, num_bits]
        else:
            self.register_buffer('receptive_masks', None)

        # HETEROGENEOUS MODE: per-being input/output projections
        # Each being gets its own K→D input and D→K output projection.
        # Big ants have bigger K = more input info. All share D for ring compat.
        self.heterogeneous = self.receptive_masks is not None and bits_per_being > 0
        if self.heterogeneous:
            self.being_input_projs = nn.ModuleList()
            self.being_output_projs = nn.ModuleList()
            for i in range(num_beings):
                if self.full_view:
                    self.being_input_projs.append(nn.Linear(num_bits * 2, embedding_dim))  # *2 for [input + GEM]
                    self.being_output_projs.append(nn.Linear(embedding_dim, num_bits))
                else:
                    k_i = int(self.receptive_masks[i].sum().item())
                    self.being_input_projs.append(nn.Linear(k_i + num_bits, embedding_dim))  # +num_bits for GEM
                    self.being_output_projs.append(nn.Linear(embedding_dim, k_i))
            # Shared projections not used in heterogeneous mode
            self.input_proj = None
            self.output_proj = None

        # TEMPORAL FIBONACCI TICK SCHEDULING
        self.temporal_fibonacci = temporal_fibonacci
        if temporal_fibonacci:
            if self.receptive_masks is not None:
                k_values = [int(self.receptive_masks[i].sum().item())
                            for i in range(num_beings)]
                tick_periods = fibonacci_tick_periods(k_values=k_values)
            else:
                tick_periods = fibonacci_tick_periods(num_beings=num_beings)
            self.register_buffer('tick_periods',
                                 torch.tensor(tick_periods, dtype=torch.long))
        else:
            self.tick_periods = None

        # Per-being state: 'null' (skip/no influence), 'active' (training), 'frozen' (learned but locked)
        self.being_states = {i: 'active' for i in range(num_beings)}  # default: all active

        # CAPACITY FIBONACCI: per-being hidden dimensions
        self.capacity_fibonacci = capacity_fibonacci
        if capacity_fibonacci:
            assert self.heterogeneous, (
                "capacity_fibonacci requires fibonacci=True with bits_per_being > 0"
            )
            k_values = fibonacci_k_schedule(num_beings, num_bits, min_k=4)
            self.hidden_dims = fibonacci_hidden_dims(k_values, max_hidden, min_hidden)

            # Ring bridge projections: D <-> H_i
            self.ring_read_projs = nn.ModuleList([
                nn.Linear(embedding_dim, h_i) for h_i in self.hidden_dims
            ])
            self.ring_write_projs = nn.ModuleList([
                nn.Linear(h_i, embedding_dim) for h_i in self.hidden_dims
            ])

            # Override being input/output projs: K_i <-> H_i (was K_i <-> D)
            self.being_input_projs = nn.ModuleList()
            self.being_output_projs = nn.ModuleList()
            for i in range(num_beings):
                h_i = self.hidden_dims[i]
                if self.full_view:
                    self.being_input_projs.append(nn.Linear(num_bits * 2, h_i))  # *2 for [input + GEM]
                    self.being_output_projs.append(nn.Linear(h_i, num_bits))
                else:
                    k_i = int(self.receptive_masks[i].sum().item())
                    self.being_input_projs.append(nn.Linear(k_i + num_bits, h_i))  # +num_bits for GEM
                    self.being_output_projs.append(nn.Linear(h_i, k_i))

            # Per-being processing layers (can't share, dims differ)
            if depth > 1:
                self.being_processing_layers = nn.ModuleList()
                for i in range(num_beings):
                    h_i = self.hidden_dims[i]
                    layers = nn.ModuleList([
                        nn.Linear(h_i, h_i) for _ in range(depth - 1)
                    ])
                    self.being_processing_layers.append(layers)
                self.processing_layers = None  # Disable shared layers
            else:
                self.being_processing_layers = None
        else:
            self.hidden_dims = [embedding_dim] * num_beings
            self.ring_read_projs = None
            self.ring_write_projs = None
            self.being_processing_layers = None

        # PER-BEING COMPONENTS (each being has its own)
        # Pass being_id for golden ratio phase embeddings + optional input mask
        self.beings = nn.ModuleList([
            BeingParameters(
                num_memory_positions, embedding_dim, being_id=i,
                input_mask=self.receptive_masks[i] if self.receptive_masks is not None else None,
                hidden_dim=self.hidden_dims[i] if capacity_fibonacci else None,
            )
            for i in range(num_beings)
        ])

        # RING-ATTENTION COMBINER (Option A: zero new parameters)
        # Learnable temperature for softmax sharpness
        if combiner_mode == 'ring_attention':
            self.gate_temperature = nn.Parameter(torch.tensor(1.0))

        # LEARNED COMBINER GATE: per-being-per-bit reliability weights
        # Solves the masked combiner deadlock where ~10 beings per bit
        # disagree with equal confidence → neutral average → chance output.
        # Gate learns which beings to trust for which bits.
        if combiner_mode == 'masked' and self.heterogeneous:
            self.combiner_gate = nn.Parameter(torch.zeros(num_beings, num_bits))
        else:
            self.combiner_gate = None

        # GLOBAL EMBEDDING MATRIX (GEM) 💎: persistent dynamic embedding
        # 1:1 mapped to input bits. Updated each step via golden ratio EMA.
        # Not a parameter (not trained by optimizer) — updated at runtime.
        self.register_buffer('gem', torch.zeros(num_bits))
        self.gem_write_head = nn.Linear(embedding_dim, num_bits)
        self._phi_inv = 0.6180339887  # 1/φ — golden ratio inverse

        # VECTORIZED PATH PRECOMPUTATION (combinatorial mode only)
        # Pre-stack per-being index tensors so _forward_vectorized() can
        # use batched ops instead of a Python for-loop over beings.
        if combinatorial and self.heterogeneous:
            # vis_indices: [num_beings, K] — which bits each being reads/writes
            vis_list = []
            for i in range(num_beings):
                vis_list.append(self.receptive_masks[i].nonzero(as_tuple=True)[0])
            self.register_buffer('_vec_vis_indices', torch.stack(vis_list))

            # phase_embeddings padded to embedding_dim: [num_beings, D]
            phase_list = []
            for being in self.beings:
                pe = being.phase_embedding  # [16]
                if embedding_dim >= 16:
                    padded = torch.zeros(embedding_dim)
                    padded[:16] = pe
                else:
                    padded = pe[:embedding_dim]
                phase_list.append(padded)
            self.register_buffer('_vec_phase', torch.stack(phase_list))

    def _gaussian_attention_weights(
        self,
        pointer_position: torch.Tensor,
        num_positions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian attention weights around pointer."""
        B = pointer_position.size(0)
        K = self.attention_radius

        offsets = torch.arange(-K, K + 1, device=pointer_position.device)
        base = torch.floor(pointer_position).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()

        # Circular distance
        half = num_positions / 2.0
        delta = torch.remainder(indices_float - pointer_position.unsqueeze(1) + half, num_positions) - half

        # Gaussian logits
        logits = -(delta ** 2) / self.attention_temperature
        weights = torch.softmax(logits, dim=1)

        return indices, weights

    def _hard_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator for hard binary decisions."""
        probs = torch.sigmoid(logits)
        hard = (probs > 0.5).float()
        return hard - probs.detach() + probs

    def _ring_attention_combine(self, being_stack, hidden_states_list, memory_ring):
        """
        Ring-attention combiner: use ring memory as query to weight beings.

        The ring has seen what all beings wrote. Its summary acts as a
        "judge" -- dot product with each being's hidden state measures
        alignment. High alignment = this being's output fits the context.

        Args:
            being_stack:        [num_beings, B, 8] raw logits from all beings
            hidden_states_list: list of [B, H_i] tensors (may have different H_i)
            memory_ring:        [B, M, D] shared ring memory

        Returns:
            combined: [B, 8] weighted combination (on probability level)
            gate_weights: [B, num_beings] the attention weights (for logging)
        """
        # Ring summary = mean across all positions
        r = memory_ring.mean(dim=1)  # [B, D]

        # Score each being by projecting to D and dot product with ring summary
        scale = math.sqrt(self.embedding_dim)
        scores_list = []
        for i, h_i in enumerate(hidden_states_list):
            if self.ring_write_projs is not None:
                h_proj = self.ring_write_projs[i](h_i)  # [B, H_i] -> [B, D]
            else:
                h_proj = h_i  # already [B, D]
            score = (h_proj * r).sum(dim=-1) / scale  # [B]
            scores_list.append(score)
        scores = torch.stack(scores_list, dim=1)  # [B, num_beings]
        scores = scores * self.gate_temperature

        # Softmax gate weights
        gate_weights = torch.softmax(scores, dim=-1)  # [B, num_beings]

        # Combine at PROBABILITY level (GPT Pro's advice: probabilities, not logits)
        probs = torch.sigmoid(being_stack)  # [num_beings, B, 8]
        # probs: [num_beings, B, 8], gate: [B, num_beings]
        # Rearrange: probs -> [B, num_beings, 8], gate -> [B, num_beings, 1]
        probs_t = probs.permute(1, 0, 2)  # [B, num_beings, 8]
        w = gate_weights.unsqueeze(-1)  # [B, num_beings, 1]
        p_combined = (w * probs_t).sum(dim=1)  # [B, 8]

        # Convert back to logit space for loss compatibility
        # logit = log(p / (1-p)), clamp to avoid inf
        p_combined = p_combined.clamp(1e-6, 1.0 - 1e-6)
        combined = torch.log(p_combined / (1.0 - p_combined))  # [B, 8]

        return combined, gate_weights

    def gate_entropy_loss(self, gate_weights):
        """
        Entropy regularizer to prevent gate collapse.

        Encourages the gate to USE all beings, not always pick one.
        H(w) = -sum(w * log(w)). Maximize entropy = minimize -H.

        Args:
            gate_weights: [B, num_beings] softmax weights

        Returns:
            neg_entropy: scalar, add to loss (lower = more entropy = better)
        """
        # Per-sample entropy, mean across batch
        log_w = torch.log(gate_weights + 1e-8)
        entropy = -(gate_weights * log_w).sum(dim=-1)  # [B]
        # We want to MAXIMIZE entropy, so return negative
        # Max entropy = log(num_beings), normalize to [0, 1]
        max_entropy = math.log(self.num_beings)
        neg_entropy = 1.0 - entropy.mean() / max_entropy
        return neg_entropy

    def _masked_combine(self, being_stack: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Masked combiner: per-bit confidence-weighted aggregation from covering beings only.

        For each output bit, only beings whose receptive mask includes that bit contribute.
        Weights are proportional to confidence (distance from 0.5 in probability space).

        Args:
            being_stack: [num_beings, B, 8] raw logits from all beings.
            training: If True, soft weighting. If False, pick most confident.

        Returns:
            combined: [B, 8] in logit space.
        """
        masks = self.receptive_masks  # [N, 8]
        probs = torch.sigmoid(being_stack)  # [N, B, 8]
        confidence = (probs - 0.5).abs() * 2.0  # [N, B, 8] range [0, 1]
        if self.full_view:
            # All beings vote on all bits — no mask gating
            mask_exp = torch.ones(masks.size(0), 1, masks.size(1),
                                  device=being_stack.device)
        else:
            mask_exp = masks.unsqueeze(1)  # [N, 1, 8]

        # Learned gate: per-being-per-bit reliability weights
        if self.combiner_gate is not None:
            gate = torch.sigmoid(self.combiner_gate).unsqueeze(1)  # [N, 1, num_bits]
        else:
            gate = 1.0

        if training:
            # Soft: mask × gate only — NO confidence weighting during training.
            # Confidence weighting creates a gradient dead zone: uncertain beings
            # get weight≈0 → gradient≈0 → can never learn. Equal weighting ensures
            # all covering beings receive gradient signal.
            weights = mask_exp * gate  # [N, B, num_bits]
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
            p_combined = (weights * probs).sum(dim=0)  # [B, num_bits]
        else:
            # Hard: pick most confident covering being (gate-weighted)
            conf_masked = confidence * gate
            conf_masked[mask_exp.expand_as(conf_masked) < 0.5] = -1.0
            best_idx = conf_masked.argmax(dim=0)  # [B, num_bits]
            p_combined = probs.gather(0, best_idx.unsqueeze(0)).squeeze(0)  # [B, num_bits]

        # Eps-stabilized logit — no hard clamp, so gradient always flows.
        # (Hard clamp has zero gradient outside range, a second gradient killer.)
        eps = 1e-6
        return torch.log(p_combined + eps) - torch.log(1.0 - p_combined + eps)

    def _is_contributing(self, being_idx: int) -> bool:
        """Check if being should contribute to spatial metrics (pointer positions)."""
        return self.being_states.get(being_idx, 'null') != 'null'

    def _forward_vectorized(
        self, x: torch.Tensor, return_stats: bool = False, return_being_outputs: bool = False,
    ):
        """
        Vectorized forward pass for combinatorial mode.

        Replaces the sequential being loop with batched tensor ops.
        All beings read the same ring state, process in parallel, and write
        simultaneously (synchronous update instead of sequential dependency).

        Only valid when combinatorial=True (uniform K, no capacity_fibonacci).
        """
        B, T, _ = x.shape
        N = self.num_beings
        D = self.embedding_dim
        M = self.num_memory_positions
        device = x.device
        dtype = x.dtype

        # -- Gather per-being parameters (fresh each call for gradient flow) --
        inp_w = torch.stack([p.weight for p in self.being_input_projs])   # [N, D, K]
        inp_b = torch.stack([p.bias for p in self.being_input_projs])     # [N, D]
        out_w = torch.stack([p.weight for p in self.being_output_projs])  # [N, K_out, D]
        out_b = torch.stack([p.bias for p in self.being_output_projs])    # [N, K_out]
        jg_w = torch.stack([b.jump_gate.weight for b in self.beings]).squeeze(1)  # [N, D]
        jg_b = torch.stack([b.jump_gate.bias for b in self.beings]).squeeze(1)    # [N]
        all_dests = torch.stack([b.pointer_destinations for b in self.beings])     # [N, M]
        ctx_scales = torch.sigmoid(
            torch.stack([b.context_strength for b in self.beings])
        ).reshape(N, 1, 1)  # [N, 1, 1]

        vis_indices = self._vec_vis_indices   # [N, K]
        phase_bias = self._vec_phase.unsqueeze(1)  # [N, 1, D]

        # Active being mask (null = skip entirely)
        active_mask = torch.tensor(
            [self.being_states.get(i, 'null') != 'null' for i in range(N)],
            device=device, dtype=torch.bool,
        )  # [N]

        # -- Initialize ring memory and per-being state --
        memory_ring = torch.zeros(B, M, D, device=device, dtype=dtype)

        pointer_positions = torch.empty(N, B, device=device, dtype=dtype)
        for n in range(N):
            lo = n * M / N
            hi = (n + 1) * M / N
            pointer_positions[n].uniform_(lo, hi)

        hidden_states = torch.zeros(N, B, D, device=device, dtype=dtype)

        # Output accumulators
        combined_outputs = []
        outputs_per_being = [[] for _ in range(N)] if return_being_outputs else None
        gate_weights_for_entropy = []

        # Stats accumulators
        if return_stats:
            jump_counts_per_being = [[] for _ in range(N)]
            pointer_positions_log = []
            output_disagreements = []
            gate_weights_log = []
            diag_hidden_abs = []   # mean |h| per timestep
            diag_hidden_max = []   # max |h| per timestep
            diag_ring_read_norm = []  # ring read norm per timestep

        # Pre-expand vis_indices for scatter: [N, B, K]
        if not self.full_view:
            vis_exp_out = vis_indices.unsqueeze(1).expand(-1, B, -1)

        # ===================== TIMESTEP LOOP =====================
        for t in range(T):
            input_bits = x[:, t, :]  # [B, num_bits]
            # GEM 💎: concatenate persistent embedding for vectorized path
            _gem_exp_v = self.gem.unsqueeze(0).expand(B, -1)  # [B, num_bits]

            # -- Temporal fibonacci dormancy mask --
            if self.tick_periods is not None:
                tick_active = (t % self.tick_periods) == 0  # [N]
            else:
                tick_active = torch.ones(N, device=device, dtype=torch.bool)
            step_active = active_mask & tick_active  # [N]
            step_mask = step_active.float().reshape(N, 1, 1)  # [N, 1, 1]

            # 1. Gather visible bits & project (with GEM concatenated)
            if self.full_view:
                input_with_gem = torch.cat([input_bits, _gem_exp_v], dim=-1)  # [B, num_bits*2]
                being_input = torch.einsum('bk,ndk->bnd', input_with_gem, inp_w) + inp_b
            else:
                visible = input_bits[:, vis_indices]  # [B, N, K]
                # Expand GEM for each being: [B, N, num_bits]
                _gem_being = _gem_exp_v.unsqueeze(1).expand(-1, N, -1)
                visible_with_gem = torch.cat([visible, _gem_being], dim=-1)  # [B, N, K+num_bits]
                being_input = torch.einsum('bnk,ndk->bnd', visible_with_gem, inp_w) + inp_b
            being_input = being_input.permute(1, 0, 2)  # [N, B, D]

            # 2. Batched ring read (all beings read SAME ring state)
            flat_ptrs = pointer_positions.reshape(-1)  # [N*B]
            indices, weights = self._gaussian_attention_weights(flat_ptrs, M)
            W = indices.size(1)  # 2*attention_radius + 1
            indices = indices.reshape(N, B, W)
            weights = weights.reshape(N, B, W)

            mem_exp = memory_ring.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, M, D]
            idx_gather = indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [N, B, W, D]
            neighborhood = mem_exp.gather(2, idx_gather)  # [N, B, W, D]
            context_reads = (weights.unsqueeze(-1) * neighborhood).sum(dim=2)  # [N, B, D]

            if return_stats:
                diag_ring_read_norm.append(context_reads.norm().item())

            # 3. Combine: input + scaled context + phase bias
            combined = being_input + ctx_scales * context_reads + 0.1 * phase_bias

            # 4. State update (LayerNorm prevents tanh saturation)
            state_update = torch.tanh(self.state_norm(combined + hidden_states))
            if self.processing_layers is not None:
                for layer in self.processing_layers:
                    state_update = state_update + torch.nn.functional.softsign(layer(state_update))

            # Apply dormancy: keep old state for dormant/null beings
            hidden_states = torch.where(step_mask.bool(), state_update, hidden_states)

            # GEM 💎 WRITE (vectorized): average across beings and batch
            _gem_sig_v = torch.nn.functional.softsign(self.gem_write_head(state_update))  # [N, B, num_bits]
            _gem_upd_v = _gem_sig_v.mean(dim=0).mean(dim=0)  # average N,B → [num_bits]
            self.gem = self._phi_inv * self.gem.detach() + (1.0 - self._phi_inv) * _gem_upd_v

            if return_stats:
                diag_hidden_abs.append(hidden_states.abs().mean().item())
                diag_hidden_max.append(hidden_states.abs().max().item())

            # 5. Ring write (all active beings simultaneously via single scatter_add)
            write_contrib = (
                hidden_states.unsqueeze(2).expand(-1, -1, W, -1)
                * weights.unsqueeze(-1)
                * step_mask.unsqueeze(2)  # zero out dormant writes
            )  # [N, B, W, D]

            # Flatten beings into scatter dimension: [B, N*W, D]
            idx_scatter = (
                indices.permute(1, 0, 2)
                .reshape(B, -1)
                .unsqueeze(-1)
                .expand(-1, -1, D)
            )  # [B, N*W, D]
            contribs_flat = write_contrib.permute(1, 0, 2, 3).reshape(B, -1, D)
            memory_ring = memory_ring.scatter_add(1, idx_scatter, contribs_flat)

            # 6. Pointer update (jump or walk) — soft blend for gradient flow
            current_pos = pointer_positions.long().clamp(0, M - 1)  # [N, B]
            jump_targets = all_dests.gather(1, current_pos).float().clamp(0, M - 1)  # [N, B]
            jump_logits = (
                (hidden_states * jg_w.unsqueeze(1)).sum(-1) + jg_b.unsqueeze(1)
            )  # [N, B]
            jump_prob = torch.sigmoid(jump_logits)  # [N, B] — continuous, differentiable
            walk_pos = (pointer_positions + 1.0) % M
            new_ptrs = jump_prob * jump_targets + (1.0 - jump_prob) * walk_pos
            pointer_positions = torch.where(step_active.unsqueeze(1), new_ptrs, pointer_positions)

            # 7. Output projection
            if self.full_view:
                being_outputs_t = (
                    torch.einsum('nbd,nkd->nbk', hidden_states, out_w)
                    + out_b.unsqueeze(1)
                )  # [N, B, num_bits]
            else:
                output_k = (
                    torch.einsum('nbd,nkd->nbk', hidden_states, out_w)
                    + out_b.unsqueeze(1)
                )  # [N, B, K]
                being_outputs_t = torch.zeros(N, B, self.num_bits, device=device, dtype=dtype)
                being_outputs_t.scatter_(2, vis_exp_out, output_k)

            # Zero null beings
            being_outputs_t[~active_mask] = 0.0

            if return_being_outputs:
                for n in range(N):
                    outputs_per_being[n].append(being_outputs_t[n])

            # Stats: per-being jump rates, pointer positions
            if return_stats:
                for n in range(N):
                    if not active_mask[n] or not tick_active[n]:
                        jump_counts_per_being[n].append(-1.0)
                    else:
                        jump_counts_per_being[n].append(jump_prob[n].mean().item())
                ptr_t = [
                    pointer_positions[n].mean().item()
                    for n in range(N) if self._is_contributing(n)
                ]
                pointer_positions_log.append(ptr_t)

            # -------- THINK TICKS --------
            if self.think_ticks > 0:
                for _tt in range(self.think_ticks):
                    if self.tick_periods is not None:
                        global_tick = t * (1 + self.think_ticks) + 1 + _tt
                        tt_active = (global_tick % self.tick_periods) == 0
                        tt_step = active_mask & tt_active
                    else:
                        tt_step = active_mask
                    tt_mask = tt_step.float().reshape(N, 1, 1)

                    # Ring read
                    flat_p = pointer_positions.reshape(-1)
                    idx_tt, wgt_tt = self._gaussian_attention_weights(flat_p, M)
                    W_tt = idx_tt.size(1)
                    idx_tt = idx_tt.reshape(N, B, W_tt)
                    wgt_tt = wgt_tt.reshape(N, B, W_tt)

                    mem_tt = memory_ring.unsqueeze(0).expand(N, -1, -1, -1)
                    idxg_tt = idx_tt.unsqueeze(-1).expand(-1, -1, -1, D)
                    nbr_tt = mem_tt.gather(2, idxg_tt)
                    ctx_tt = (wgt_tt.unsqueeze(-1) * nbr_tt).sum(dim=2)  # [N, B, D]

                    # No input — context + phase only
                    comb_tt = ctx_scales * ctx_tt + 0.1 * phase_bias
                    su_tt = torch.tanh(self.state_norm(comb_tt + hidden_states))
                    if self.processing_layers is not None:
                        for layer in self.processing_layers:
                            su_tt = su_tt + torch.nn.functional.softsign(layer(su_tt))
                    hidden_states = torch.where(tt_mask.bool(), su_tt, hidden_states)

                    # Ring write
                    wc_tt = (
                        hidden_states.unsqueeze(2).expand(-1, -1, W_tt, -1)
                        * wgt_tt.unsqueeze(-1) * tt_mask.unsqueeze(2)
                    )
                    ids_tt = idx_tt.permute(1, 0, 2).reshape(B, -1).unsqueeze(-1).expand(-1, -1, D)
                    cfs_tt = wc_tt.permute(1, 0, 2, 3).reshape(B, -1, D)
                    memory_ring = memory_ring.scatter_add(1, ids_tt, cfs_tt)

                    # Pointer update — soft blend for gradient flow
                    cp_tt = pointer_positions.long().clamp(0, M - 1)
                    jt_tt = all_dests.gather(1, cp_tt).float().clamp(0, M - 1)
                    jl_tt = (hidden_states * jg_w.unsqueeze(1)).sum(-1) + jg_b.unsqueeze(1)
                    jp_tt = torch.sigmoid(jl_tt)
                    wp_tt = (pointer_positions + 1.0) % M
                    np_tt = jp_tt * jt_tt + (1.0 - jp_tt) * wp_tt
                    pointer_positions = torch.where(tt_step.unsqueeze(1), np_tt, pointer_positions)

                # Regenerate outputs after think ticks
                if self.full_view:
                    being_outputs_t = (
                        torch.einsum('nbd,nkd->nbk', hidden_states, out_w)
                        + out_b.unsqueeze(1)
                    )
                else:
                    output_k = (
                        torch.einsum('nbd,nkd->nbk', hidden_states, out_w)
                        + out_b.unsqueeze(1)
                    )
                    being_outputs_t = torch.zeros(N, B, self.num_bits, device=device, dtype=dtype)
                    being_outputs_t.scatter_(2, vis_exp_out, output_k)
                being_outputs_t[~active_mask] = 0.0

                if return_being_outputs:
                    for n in range(N):
                        outputs_per_being[n][-1] = being_outputs_t[n]

            # -------- COMBINE OUTPUTS --------
            if self.combiner_mode == 'masked' and self.receptive_masks is not None:
                combined_t = self._masked_combine(being_outputs_t, self.training)
            elif self.combiner_mode == 'ring_attention':
                hidden_list = [hidden_states[n] for n in range(N)]
                combined_t, gate_w = self._ring_attention_combine(
                    being_outputs_t, hidden_list, memory_ring,
                )
                gate_weights_for_entropy.append(gate_w)
                if return_stats:
                    gate_weights_log.append(gate_w.detach())
            else:
                if self.training:
                    combined_t = being_outputs_t.mean(dim=0)
                else:
                    being_binary = (being_outputs_t > 0.0).float()
                    vote_sum = being_binary.sum(dim=0)
                    combined_t = (vote_sum > N / 2.0).float() * 2.0 - 1.0

            combined_outputs.append(combined_t)

            if return_stats:
                output_disagreements.append(being_outputs_t.std(dim=0).mean().item())

        # ===================== END TIMESTEP LOOP =====================
        output = torch.stack(combined_outputs, dim=1)  # [B, T, num_bits]

        # Entropy loss (ring_attention only)
        if self.combiner_mode == 'ring_attention' and gate_weights_for_entropy:
            all_gw = torch.stack(gate_weights_for_entropy)
            self._last_entropy_loss = self.gate_entropy_loss(all_gw.reshape(-1, N))
        else:
            self._last_entropy_loss = torch.tensor(0.0)

        if not return_stats:
            return output

        # -- Compute stats (mirrors sequential path) --
        all_jump_counts = []
        for being_counts in jump_counts_per_being:
            all_jump_counts.extend(c for c in being_counts if c >= 0.0)
        avg_jump_rate = sum(all_jump_counts) / len(all_jump_counts) if all_jump_counts else 0.5

        def circular_distance(a, b, ring=M):
            d = abs(a - b)
            return min(d, ring - d)

        circular_spreads = []
        all_pointer_positions = []
        for t_idx in range(T):
            pts = pointer_positions_log[t_idx]
            all_pointer_positions.extend(int(p) % M for p in pts)
            dists = [
                circular_distance(pts[i], pts[j])
                for i in range(len(pts)) for j in range(i + 1, len(pts))
            ]
            if dists:
                circular_spreads.append(sum(dists) / len(dists))
        avg_spread = sum(circular_spreads) / len(circular_spreads) if circular_spreads else 0.0

        avg_disagree = (
            sum(output_disagreements) / len(output_disagreements) if output_disagreements else 0.0
        )

        jump_rates = []
        for counts in jump_counts_per_being:
            active = [c for c in counts if c >= 0.0]
            jump_rates.append(sum(active) / len(active) if active else 0.0)

        stats = {
            'jump_gate': avg_jump_rate,
            'circular_spread': avg_spread,
            'pointer_spread': avg_spread,
            'output_disagreement': avg_disagree,
            'jump_rates_per_being': jump_rates,
            'pointer_positions_all': all_pointer_positions,
            'diag_hidden_abs': diag_hidden_abs,
            'diag_hidden_max': diag_hidden_max,
            'diag_ring_read_norm': diag_ring_read_norm,
            'diag_ctx_scales': [
                torch.sigmoid(b.context_strength).item() for b in self.beings
            ],
        }

        if self.receptive_masks is not None and not self.full_view:
            masks = self.receptive_masks
            bit_cov = masks.sum(dim=0)
            stats['bit_coverage'] = bit_cov.int().tolist()
            stats['min_bit_coverage'] = int(bit_cov.min().item())
            stats['max_bit_coverage'] = int(bit_cov.max().item())
            stats['avg_bit_coverage'] = float(bit_cov.float().mean().item())
            n_masks = masks.size(0)
            ham_sum = 0.0
            ham_count = 0
            for i in range(n_masks):
                for j in range(i + 1, n_masks):
                    ham_sum += (masks[i] != masks[j]).float().sum().item()
                    ham_count += 1
            stats['mask_diversity'] = ham_sum / ham_count if ham_count > 0 else 0.0

        if self.tick_periods is not None:
            periods = self.tick_periods.tolist()
            avg_active = sum(1.0 / p for p in periods)
            stats['avg_active_per_tick'] = avg_active
            stats['activation_ratio'] = avg_active / N

        if self._pair_coverage_frac is not None:
            stats['pair_coverage'] = self._pair_coverage_frac

        if self.combiner_mode == 'ring_attention' and return_stats and gate_weights_log:
            all_gates = torch.stack(gate_weights_log)
            avg_gate = all_gates.mean(dim=(0, 1))
            stats['gate_weights'] = avg_gate.tolist()
            log_g = torch.log(all_gates + 1e-8)
            entropy = -(all_gates * log_g).sum(dim=-1).mean()
            max_ent = math.log(N)
            stats['gate_entropy'] = float(entropy.item() / max_ent)
            stats['gate_temperature'] = float(self.gate_temperature.item())

        if return_being_outputs:
            all_bo = []
            for n in range(N):
                all_bo.append(torch.stack(outputs_per_being[n]))
            stats['being_outputs'] = torch.stack(all_bo)  # [N, T, B, num_bits]

        return output, stats

    def forward(self, x: torch.Tensor, return_stats: bool = False, return_being_outputs: bool = False):
        """
        Forward pass with N beings sharing one ring.

        Args:
            x: [B, T, 8] byte sequence
            return_stats: If True, return (output, stats) with swarm metrics
            return_being_outputs: If True, include per-being outputs in stats

        Returns:
            output: [B, T, 8] predicted sequence (mean of all beings)
            stats: Dict with swarm metrics (if return_stats=True)
                - jump_gate: Average jump activation across all beings
                - circular_spread: Mean pairwise circular distance (FIXED from linear std)
                - output_disagreement: Std of being outputs (ensemble diversity)
                - being_outputs: [num_beings, T, B, 8] ALL timestep outputs (if return_being_outputs=True)
                - jump_rates_per_being: List of jump rates per being
                - pointer_positions_all: List of all pointer positions (for coverage)
        """
        # GEM 💎: detach from previous step's computation graph
        # Preserves values but drops grad_fn so optimizer.step() doesn't invalidate
        self.gem = self.gem.detach()

        # Dispatch to vectorized path for combinatorial mode
        if self.combinatorial and hasattr(self, '_vec_vis_indices'):
            return self._forward_vectorized(x, return_stats, return_being_outputs)

        B, T, _ = x.shape

        # SHARED ring memory (all beings write here)
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )

        # PER-BEING state initialization
        being_states = []
        for being_idx in range(self.num_beings):
            # Spatial offset initialization: spread beings across ring
            offset_start = being_idx * self.num_memory_positions / self.num_beings
            offset_end = (being_idx + 1) * self.num_memory_positions / self.num_beings
            pointer_init = torch.empty(B, device=x.device).uniform_(offset_start, offset_end)

            being_states.append({
                'pointer_position': pointer_init,
                'hidden_state': torch.zeros(B, self.hidden_dims[being_idx], device=x.device, dtype=x.dtype),
            })

        # Track outputs per being (for ensemble averaging and per-being stats)
        outputs_per_being = [[] for _ in range(self.num_beings)]
        combined_outputs_per_t = []  # Store combined output per timestep
        gate_weights_for_entropy = []  # For entropy regularizer (with grad)

        # Track metrics (if requested)
        if return_stats:
            jump_counts_per_being = [[] for _ in range(self.num_beings)]
            pointer_positions_log = []
            output_disagreements = []
            gate_weights_log = []

        # TIMESTEP LOOP
        for t in range(T):
            # 1. Shared input projection (all beings see same input)
            input_bits = x[:, t, :]
            # GEM 💎: concatenate persistent embedding alongside raw input
            gem_expanded = self.gem.unsqueeze(0).expand(input_bits.size(0), -1)  # [B, num_bits]
            input_bits_with_gem = torch.cat([input_bits, gem_expanded], dim=-1)  # [B, num_bits*2]
            if self.input_proj is not None:
                input_vec = self.input_proj(input_bits_with_gem)
            else:
                input_vec = input_bits_with_gem

            # 2. Each being processes independently
            being_outputs_t = []
            hidden_states_t = []
            pointer_positions_t = []

            for being_idx in range(self.num_beings):
                being = self.beings[being_idx]
                state = being_states[being_idx]

                # NULL BEINGS: skip entirely (output zeros, no influence)
                if self.being_states.get(being_idx, 'null') == 'null':
                    output_bits = torch.zeros(x.size(0), self.num_bits, device=x.device, dtype=x.dtype)
                    being_outputs_t.append(output_bits)
                    outputs_per_being[being_idx].append(output_bits)
                    hidden_states_t.append(state['hidden_state'].clone())
                    if return_stats:
                        jump_counts_per_being[being_idx].append(-1.0)
                        if self._is_contributing(being_idx):
                            pointer_positions_t.append(state['pointer_position'].mean().item())
                    continue

                # TEMPORAL FIBONACCI: skip dormant beings
                if self.tick_periods is not None and t % self.tick_periods[being_idx].item() != 0:
                    # Dormant: produce stale output from current hidden state
                    if self.full_view:
                        output_bits = self.being_output_projs[being_idx](state['hidden_state'])
                    elif self.heterogeneous:
                        vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                        output_k = self.being_output_projs[being_idx](state['hidden_state'])
                        output_bits = torch.zeros(state['hidden_state'].size(0), self.num_bits,
                                                  device=x.device, dtype=x.dtype)
                        output_bits[:, vis_idx] = output_k
                    elif self.output_proj is not None:
                        output_bits = self.output_proj(state['hidden_state'])
                    else:
                        output_bits = state['hidden_state']
                    being_outputs_t.append(output_bits)
                    outputs_per_being[being_idx].append(output_bits)
                    hidden_states_t.append(state['hidden_state'].clone())
                    if return_stats:
                        jump_counts_per_being[being_idx].append(-1.0)  # sentinel: dormant tick
                        if self._is_contributing(being_idx):
                            pointer_positions_t.append(state['pointer_position'].mean().item())
                    continue

                # 2a. Read from being's pointer position
                indices, weights = self._gaussian_attention_weights(
                    state['pointer_position'], self.num_memory_positions
                )
                indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                neighborhood = memory_ring.gather(1, indices_exp)
                context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)
                context_scale = torch.sigmoid(being.context_strength)

                # CAPACITY FIBONACCI: project ring context D -> H_i
                if self.ring_read_projs is not None:
                    context_read = self.ring_read_projs[being_idx](context_read)

                # 2b. Combine input + context (with optional receptive field masking)
                if self.full_view:
                    # FULL VIEW: all bits projected through per-being bottleneck
                    being_input_vec = self.being_input_projs[being_idx](input_bits_with_gem)  # [B, num_bits*2] → [B, H_i]
                elif self.heterogeneous:
                    # HETEROGENEOUS: extract visible bits + full GEM, use per-being projection
                    vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                    visible_bits = input_bits[:, vis_idx]  # [B, K_i]
                    visible_with_gem = torch.cat([visible_bits, gem_expanded], dim=-1)  # [B, K_i + num_bits]
                    being_input_vec = self.being_input_projs[being_idx](visible_with_gem)  # [B, D]
                elif being.input_mask is not None:
                    # Homogeneous with masks: zero out non-visible bits + GEM
                    masked_bits = input_bits * being.input_mask.unsqueeze(0)  # [B, 8]
                    masked_with_gem = torch.cat([masked_bits, gem_expanded], dim=-1)  # [B, num_bits*2]
                    if self.input_proj is not None:
                        being_input_vec = self.input_proj(masked_with_gem)
                    else:
                        being_input_vec = masked_with_gem
                else:
                    being_input_vec = input_vec  # No mask: use shared projection
                combined_input = being_input_vec + context_scale * context_read

                # 2b.1. Add golden ratio phase embedding (forces specialization)
                # Expand phase_embedding from [16] to [B, H_i]
                h_i = self.hidden_dims[being_idx]
                B = combined_input.size(0)
                if being.phase_embedding.size(0) == h_i:
                    phase_bias = being.phase_embedding.unsqueeze(0).expand(B, -1)
                else:
                    phase_16d = being.phase_embedding.unsqueeze(0).expand(B, -1)
                    if h_i >= 16:
                        padding = torch.zeros(B, h_i - 16,
                                            device=phase_16d.device, dtype=phase_16d.dtype)
                        phase_bias = torch.cat([phase_16d, padding], dim=1)
                    else:
                        phase_bias = phase_16d[:, :h_i]

                # Add phase bias (small scale to not dominate)
                combined_input = combined_input + 0.1 * phase_bias

                # 2c. State update (LayerNorm prevents tanh saturation)
                state_update = torch.tanh(self.state_norm(combined_input + state['hidden_state']))

                # Additional processing layers (with residual connections)
                if self.being_processing_layers is not None:
                    for layer in self.being_processing_layers[being_idx]:
                        state_update = state_update + torch.nn.functional.softsign(layer(state_update))
                elif self.processing_layers is not None:
                    for layer in self.processing_layers:
                        state_update = state_update + torch.nn.functional.softsign(layer(state_update))

                state['hidden_state'] = state_update

                # GEM 💎 WRITE: golden ratio EMA update from this being's hidden state
                _gem_signal = torch.nn.functional.softsign(self.gem_write_head(state_update))  # [B, num_bits]
                _gem_update = _gem_signal.mean(dim=0)  # average across batch → [num_bits]
                self.gem = self._phi_inv * self.gem.detach() + (1.0 - self._phi_inv) * _gem_update

                # 2d. Write to SHARED ring (scatter_add accumulation)
                # Gradient flows through ring writes for collaborative learning.
                # CAPACITY FIBONACCI: project H_i -> D before ring write
                if self.ring_write_projs is not None:
                    write_vec = self.ring_write_projs[being_idx](state_update)
                else:
                    write_vec = state_update
                update_broadcast = write_vec.unsqueeze(1).expand(-1, weights.size(1), -1)
                contribution = weights.unsqueeze(-1) * update_broadcast
                memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

                # 2e. Update pointer (jump or walk) — soft blend for gradient flow
                current_pos = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                jump_target = being.pointer_destinations[current_pos].float().clamp(0, self.num_memory_positions - 1)
                jump_logits = being.jump_gate(state_update).squeeze(-1)
                jump_prob = torch.sigmoid(jump_logits)
                walk_position = (state['pointer_position'] + 1.0) % self.num_memory_positions
                state['pointer_position'] = jump_prob * jump_target + (1.0 - jump_prob) * walk_position

                # Track jump activation (if stats requested)
                if return_stats:
                    jump_counts_per_being[being_idx].append(jump_prob.mean().item())
                    pointer_positions_t.append(state['pointer_position'].mean().item())

                # 2f. Generate output
                if self.full_view:
                    # FULL VIEW: project H_i → all bits
                    output_bits = self.being_output_projs[being_idx](state['hidden_state'])  # [B, H_i] → [B, num_bits]
                elif self.heterogeneous:
                    # Per-being output: D → K_i visible bits, placed into full vector
                    vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                    output_k = self.being_output_projs[being_idx](state['hidden_state'])  # [B, K_i]
                    output_bits = torch.zeros(
                        state['hidden_state'].size(0), self.num_bits,
                        device=x.device, dtype=x.dtype,
                    )
                    output_bits[:, vis_idx] = output_k
                elif self.output_proj is not None:
                    output_bits = self.output_proj(state['hidden_state'])
                else:
                    output_bits = state['hidden_state']

                being_outputs_t.append(output_bits)
                outputs_per_being[being_idx].append(output_bits)
                hidden_states_t.append(state['hidden_state'].clone())

            # 2g. THINK TICKS: extra ring rounds without input injection
            #     Beings read what others wrote, process, write back. No new input.
            if self.think_ticks > 0:
                for _tt in range(self.think_ticks):
                    for being_idx in range(self.num_beings):
                        being = self.beings[being_idx]
                        state = being_states[being_idx]

                        # NULL BEINGS: skip in think ticks
                        if self.being_states.get(being_idx, 'null') == 'null':
                            continue

                        # TEMPORAL FIBONACCI: skip dormant beings in think ticks
                        if self.tick_periods is not None:
                            global_tick = t * (1 + self.think_ticks) + 1 + _tt
                            if global_tick % self.tick_periods[being_idx].item() != 0:
                                continue

                        # Read from ring
                        indices, weights = self._gaussian_attention_weights(
                            state['pointer_position'], self.num_memory_positions
                        )
                        indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                        neighborhood = memory_ring.gather(1, indices_exp)
                        context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)
                        context_scale = torch.sigmoid(being.context_strength)

                        # CAPACITY FIBONACCI: project ring context D -> H_i
                        if self.ring_read_projs is not None:
                            context_read = self.ring_read_projs[being_idx](context_read)

                        # No input -- only ring context + hidden state
                        combined_input = context_scale * context_read

                        # Phase embedding (identity signal)
                        h_i_tt = self.hidden_dims[being_idx]
                        B_tt = combined_input.size(0)
                        if being.phase_embedding.size(0) == h_i_tt:
                            phase_bias = being.phase_embedding.unsqueeze(0).expand(B_tt, -1)
                        else:
                            phase_16d = being.phase_embedding.unsqueeze(0).expand(B_tt, -1)
                            if h_i_tt >= 16:
                                padding = torch.zeros(B_tt, h_i_tt - 16,
                                                      device=phase_16d.device, dtype=phase_16d.dtype)
                                phase_bias = torch.cat([phase_16d, padding], dim=1)
                            else:
                                phase_bias = phase_16d[:, :h_i_tt]
                        combined_input = combined_input + 0.1 * phase_bias

                        # State update (LayerNorm prevents tanh saturation)
                        state_update = torch.tanh(self.state_norm(combined_input + state['hidden_state']))
                        if self.being_processing_layers is not None:
                            for layer in self.being_processing_layers[being_idx]:
                                state_update = state_update + torch.nn.functional.softsign(layer(state_update))
                        elif self.processing_layers is not None:
                            for layer in self.processing_layers:
                                state_update = state_update + torch.nn.functional.softsign(layer(state_update))
                        state['hidden_state'] = state_update

                        # Write to ring
                        if self.ring_write_projs is not None:
                            write_vec = self.ring_write_projs[being_idx](state_update)
                        else:
                            write_vec = state_update
                        update_broadcast = write_vec.unsqueeze(1).expand(-1, weights.size(1), -1)
                        contribution = weights.unsqueeze(-1) * update_broadcast
                        memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

                        # Move pointer — soft blend for gradient flow
                        current_pos = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                        jump_target = being.pointer_destinations[current_pos].float().clamp(0, self.num_memory_positions - 1)
                        jump_logits = being.jump_gate(state_update).squeeze(-1)
                        jump_prob = torch.sigmoid(jump_logits)
                        walk_position = (state['pointer_position'] + 1.0) % self.num_memory_positions
                        state['pointer_position'] = jump_prob * jump_target + (1.0 - jump_prob) * walk_position

                # Regenerate outputs from post-thinking hidden states
                being_outputs_t = []
                hidden_states_t = []
                for being_idx in range(self.num_beings):
                    state = being_states[being_idx]

                    # NULL BEINGS: output zeros, don't overwrite per-being history
                    if self.being_states.get(being_idx, 'null') == 'null':
                        being_outputs_t.append(torch.zeros(
                            x.size(0), self.num_bits, device=x.device, dtype=x.dtype))
                        hidden_states_t.append(state['hidden_state'].clone())
                        continue

                    if self.full_view:
                        output_bits = self.being_output_projs[being_idx](state['hidden_state'])
                    elif self.heterogeneous:
                        vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                        output_k = self.being_output_projs[being_idx](state['hidden_state'])
                        output_bits = torch.zeros(
                            state['hidden_state'].size(0), self.num_bits,
                            device=state['hidden_state'].device, dtype=state['hidden_state'].dtype,
                        )
                        output_bits[:, vis_idx] = output_k
                    elif self.output_proj is not None:
                        output_bits = self.output_proj(state['hidden_state'])
                    else:
                        output_bits = state['hidden_state']
                    being_outputs_t.append(output_bits)
                    outputs_per_being[being_idx][-1] = output_bits  # Replace pre-think output
                    hidden_states_t.append(state['hidden_state'].clone())

            # 3. Aggregate outputs across beings
            being_stack = torch.stack(being_outputs_t)  # [num_beings, B, 8]

            if self.combiner_mode == 'masked' and self.receptive_masks is not None:
                mean_output_t = self._masked_combine(being_stack, self.training)
            elif self.combiner_mode == 'ring_attention':
                mean_output_t, gate_w = self._ring_attention_combine(
                    being_stack, hidden_states_t, memory_ring
                )
                gate_weights_for_entropy.append(gate_w)  # keep grad for entropy loss
                if return_stats:
                    gate_weights_log.append(gate_w.detach())
            else:
                if self.training:
                    # TRAINING: Soft voting (average) for gradient flow
                    mean_output_t = being_stack.mean(dim=0)
                else:
                    # EVAL: Hard voting (majority wins)
                    being_binary = (being_stack > 0.0).float()
                    vote_sum = being_binary.sum(dim=0)
                    majority_threshold = self.num_beings / 2.0
                    mean_output_t = (vote_sum > majority_threshold).float() * 2.0 - 1.0

            # Store combined output for this timestep
            combined_outputs_per_t.append(mean_output_t)

            # Track ensemble diversity (if stats requested)
            if return_stats:
                pointer_positions_log.append(pointer_positions_t)
                # Output disagreement: std of being outputs
                being_outputs_stacked = torch.stack(being_outputs_t)  # [num_beings, B, 8]
                disagreement = being_outputs_stacked.std(dim=0).mean().item()  # scalar
                output_disagreements.append(disagreement)

        # 4. Stack combined outputs across timesteps -> [B, T, 8]
        output = torch.stack(combined_outputs_per_t, dim=1)

        # Compute gate entropy loss (ring_attention only, for regularizer)
        if self.combiner_mode == 'ring_attention' and gate_weights_for_entropy:
            all_gw = torch.stack(gate_weights_for_entropy)  # [T, B, num_beings]
            self._last_entropy_loss = self.gate_entropy_loss(
                all_gw.reshape(-1, self.num_beings)  # [T*B, num_beings]
            )
        else:
            self._last_entropy_loss = torch.tensor(0.0)

        # 5. Return with optional stats
        if return_stats:
            # Jump gate: average across all beings and timesteps (skip dormant ticks)
            all_jump_counts = []
            for being_counts in jump_counts_per_being:
                active_counts = [c for c in being_counts if c >= 0.0]  # -1.0 = dormant
                all_jump_counts.extend(active_counts)
            avg_jump_rate = sum(all_jump_counts) / len(all_jump_counts) if all_jump_counts else 0.5

            # Circular pointer spread: mean pairwise circular distance (FIXED)
            def circular_distance(a, b, N=self.num_memory_positions):
                diff = abs(a - b)
                return min(diff, N - diff)

            circular_spreads = []
            all_pointer_positions = []  # For coverage calculation
            for t in range(T):
                positions_t = pointer_positions_log[t]  # List of positions for this timestep
                all_pointer_positions.extend([int(p) % self.num_memory_positions for p in positions_t])

                # Compute mean pairwise circular distance
                distances = []
                for i in range(len(positions_t)):
                    for j in range(i+1, len(positions_t)):
                        distances.append(circular_distance(positions_t[i], positions_t[j]))
                if distances:
                    circular_spreads.append(sum(distances) / len(distances))

            avg_circular_spread = sum(circular_spreads) / len(circular_spreads) if circular_spreads else 0.0

            # Output disagreement: average over timesteps
            avg_output_disagreement = sum(output_disagreements) / len(output_disagreements) if output_disagreements else 0.0

            # Per-being jump rates (skip dormant ticks marked as -1.0)
            jump_rates_per_being = []
            for counts in jump_counts_per_being:
                active = [c for c in counts if c >= 0.0]
                jump_rates_per_being.append(
                    sum(active) / len(active) if active else 0.0
                )

            stats = {
                'jump_gate': avg_jump_rate,
                'circular_spread': avg_circular_spread,  # FIXED metric
                'pointer_spread': avg_circular_spread,  # Backward compat alias
                'output_disagreement': avg_output_disagreement,
                'jump_rates_per_being': jump_rates_per_being,
                'pointer_positions_all': all_pointer_positions,
            }

            # Receptive field mask stats (skip in full_view — all beings see all bits)
            if self.receptive_masks is not None and not self.full_view:
                masks = self.receptive_masks  # [num_beings, 8]
                bit_cov = masks.sum(dim=0)  # [8]
                stats['bit_coverage'] = bit_cov.int().tolist()
                stats['min_bit_coverage'] = int(bit_cov.min().item())
                stats['max_bit_coverage'] = int(bit_cov.max().item())
                stats['avg_bit_coverage'] = float(bit_cov.float().mean().item())
                # Mask diversity: avg pairwise Hamming distance
                N = masks.size(0)
                ham_sum = 0.0
                ham_count = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        ham_sum += (masks[i] != masks[j]).float().sum().item()
                        ham_count += 1
                stats['mask_diversity'] = ham_sum / ham_count if ham_count > 0 else 0.0

            # Temporal fibonacci stats
            if self.tick_periods is not None:
                periods = self.tick_periods.tolist()
                avg_active = sum(1.0 / p for p in periods)
                stats['avg_active_per_tick'] = avg_active
                stats['activation_ratio'] = avg_active / self.num_beings

            if self.capacity_fibonacci:
                stats['hidden_dims'] = self.hidden_dims
                stats['max_hidden'] = max(self.hidden_dims)
                stats['min_hidden'] = min(self.hidden_dims)

            if self._pair_coverage_frac is not None:
                stats['pair_coverage'] = self._pair_coverage_frac

            # Gate stats (ring-attention only)
            if self.combiner_mode == 'ring_attention' and gate_weights_log:
                # gate_weights_log: list of [B, num_beings] tensors
                all_gates = torch.stack(gate_weights_log)  # [T, B, num_beings]
                avg_gate = all_gates.mean(dim=(0, 1))  # [num_beings]
                stats['gate_weights'] = avg_gate.tolist()
                # Gate entropy (0 = collapsed, 1 = uniform)
                log_g = torch.log(all_gates + 1e-8)
                entropy = -(all_gates * log_g).sum(dim=-1).mean()
                max_ent = math.log(self.num_beings)
                stats['gate_entropy'] = float(entropy.item() / max_ent)
                # Gate temperature
                stats['gate_temperature'] = float(self.gate_temperature.item())

            # Add per-being outputs if requested
            if return_being_outputs:
                # Get ALL timestep outputs for each being
                all_being_outputs = []
                for being_idx in range(self.num_beings):
                    # Stack all timesteps: [T, B, 8]
                    being_sequence = torch.stack(outputs_per_being[being_idx])
                    all_being_outputs.append(being_sequence)
                # Stack to [num_beings, T, B, 8]
                stats['being_outputs'] = torch.stack(all_being_outputs)

            return output, stats
        else:
            return output


if __name__ == "__main__":
    print("=" * 70)
    print("SWARM BYTE RING MODEL TEST")
    print("=" * 70)
    print()

    # Test 1: Backward compat (N=8)
    for num_beings in [2, 4]:
        print(f"Testing {num_beings}-being swarm (N=8, backward compat):")
        model = SwarmByteRingModel(
            num_memory_positions=64, embedding_dim=64,
            num_beings=num_beings, depth=2, num_bits=8,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total_params:,}")

        x = torch.randint(0, 2, (4, 16, 8)).float()
        output, stats = model(x, return_stats=True)
        assert output.shape == (4, 16, 8), f"Expected (4,16,8), got {output.shape}"

        model.zero_grad()
        target = torch.randint(0, 2, (4, 16, 8)).float()
        loss = nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  Grads: {has_grads}/{sum(1 for _ in model.parameters())}  OK")
        print()

    # Test 2: N=64, K=8, 16 beings (standardized architecture)
    print("-" * 70)
    print("Testing N=64 architecture (16 beings, K=8, D=64):")
    print()

    model_64 = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=16, depth=2, num_bits=64,
        combiner_mode='masked', bits_per_being=8, min_coverage=2,
    )

    total_params_64 = sum(p.numel() for p in model_64.parameters())
    print(f"  Params: {total_params_64:,}")

    masks = model_64.receptive_masks
    print(f"  Masks shape: {masks.shape}")
    coverage = masks.sum(dim=0)
    print(f"  Coverage: min={int(coverage.min())}, max={int(coverage.max())}, "
          f"mean={coverage.mean():.1f}")

    # Check jump gate init
    for i, being in enumerate(model_64.beings):
        bias_val = being.jump_gate.bias.item()
        if i == 0:
            print(f"  Jump gate bias (being 0): {bias_val:.2f} (should be 0.5)")
        assert abs(bias_val - 0.5) < 0.01, f"Being {i} jump gate bias = {bias_val}"

    x = torch.randint(0, 2, (4, 16, 64)).float()
    output_64, stats_64 = model_64(x, return_stats=True)
    print(f"  Output shape: {output_64.shape}")
    assert output_64.shape == (4, 16, 64), f"Expected (4,16,64), got {output_64.shape}"
    print(f"  Min bit coverage: {stats_64['min_bit_coverage']}")
    print(f"  Max bit coverage: {stats_64['max_bit_coverage']}")
    print(f"  Mask diversity: {stats_64['mask_diversity']:.3f}")

    # Check all beings jump at init
    jump_rates = stats_64['jump_rates_per_being']
    dead_jumps = [i for i, r in enumerate(jump_rates) if r < 0.01]
    if dead_jumps:
        print(f"  WARNING: Dead jump gates at beings {dead_jumps}")
    else:
        print(f"  All {len(jump_rates)} beings jumping (min={min(jump_rates):.3f})")

    # Gradient flow
    model_64.zero_grad()
    target = torch.randint(0, 2, (4, 16, 64)).float()
    loss = nn.functional.binary_cross_entropy_with_logits(output_64, target)
    loss.backward()
    has_grads = sum(1 for p in model_64.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p = sum(1 for _ in model_64.parameters())
    print(f"  Grads: {has_grads}/{total_p}  OK")
    print()

    # Test 3: Combinatorial mode (vectorized forward)
    print("-" * 70)
    print("Testing COMBINATORIAL VECTORIZED (16 beings, N=64, D=64):")
    print()

    model_comb = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=16, depth=2, num_bits=64,
        combiner_mode='masked', bits_per_being=0,  # auto K from phi
        min_coverage=2, combinatorial=True,
    )
    total_params_comb = sum(p.numel() for p in model_comb.parameters())
    print(f"  Params: {total_params_comb:,}")
    print(f"  Has _vec_vis_indices: {hasattr(model_comb, '_vec_vis_indices')}")
    print(f"  vis_indices shape: {model_comb._vec_vis_indices.shape}")
    print(f"  phase shape: {model_comb._vec_phase.shape}")

    x_comb = torch.randint(0, 2, (4, 16, 64)).float()
    output_comb, stats_comb = model_comb(x_comb, return_stats=True, return_being_outputs=True)
    print(f"  Output shape: {output_comb.shape}")
    assert output_comb.shape == (4, 16, 64), f"Expected (4,16,64), got {output_comb.shape}"
    print(f"  being_outputs shape: {stats_comb['being_outputs'].shape}")
    assert stats_comb['being_outputs'].shape[0] == 16
    assert stats_comb['being_outputs'].shape[1] == 16  # T
    assert stats_comb['being_outputs'].shape[2] == 4   # B
    print(f"  Jump gate: {stats_comb['jump_gate']:.3f}")
    print(f"  Circular spread: {stats_comb['circular_spread']:.3f}")
    print(f"  Output disagreement: {stats_comb['output_disagreement']:.3f}")

    # Gradient flow
    model_comb.zero_grad()
    target_comb = torch.randint(0, 2, (4, 16, 64)).float()
    loss_comb = nn.functional.binary_cross_entropy_with_logits(output_comb, target_comb)
    loss_comb.backward()
    has_grads_comb = sum(1 for p in model_comb.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p_comb = sum(1 for _ in model_comb.parameters())
    print(f"  Grads: {has_grads_comb}/{total_p_comb}  OK")

    # Test with think_ticks
    print()
    print("  Testing with think_ticks=2:")
    model_think = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=8, depth=2, num_bits=64,
        combiner_mode='masked', combinatorial=True, think_ticks=2,
    )
    x_think = torch.randint(0, 2, (2, 8, 64)).float()
    out_think = model_think(x_think)
    assert out_think.shape == (2, 8, 64), f"Think ticks: Expected (2,8,64), got {out_think.shape}"
    model_think.zero_grad()
    loss_think = nn.functional.binary_cross_entropy_with_logits(out_think, torch.randint(0, 2, (2, 8, 64)).float())
    loss_think.backward()
    print(f"  Shape: {out_think.shape}  Grads OK")

    # Test with temporal_fibonacci
    print()
    print("  Testing with temporal_fibonacci:")
    model_tf = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=8, depth=2, num_bits=64,
        combiner_mode='masked', combinatorial=True, temporal_fibonacci=True,
    )
    x_tf = torch.randint(0, 2, (2, 8, 64)).float()
    out_tf, stats_tf = model_tf(x_tf, return_stats=True)
    assert out_tf.shape == (2, 8, 64)
    print(f"  Shape: {out_tf.shape}  activation_ratio: {stats_tf.get('activation_ratio', 'N/A')}")
    print()

    print("=" * 70)
    print("All tests passed!")
    print(f"N=8 backward compat:      OK")
    print(f"N=64 standardized:        OK ({total_params_64:,} params)")
    print(f"N=64 combinatorial (vec): OK ({total_params_comb:,} params)")
    print("=" * 70)
