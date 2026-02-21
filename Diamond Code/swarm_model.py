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


# Jump gate sigmoid temperature: controls sharpness of jump/walk decision.
# Probe sweep (v1+v2): TAU=0.5 optimal, plateau at 0.25-0.5, TAU=0.1 hurts.
# At tau=0.5: sigmoid(2/0.5)=0.98 — near-binary, decisive pointer navigation.
JUMP_SIGMOID_TAU = 0.5

# State update EMA blend: old_hidden vs new_input (ALPHA * old + (1-ALPHA) * new).
# Probe sweep (v1+v2): plateau at 0.2-0.3, peak at 0.2. Below 0.1 collapses (memory starvation).
# Golden ratio 0.618 was suboptimal (-1.4%). Model wants fast input integration.
STATE_EMA_ALPHA = 0.2

# Exclusive effort weights: each effort level owns one LCX channel only
EFFORT_WEIGHTS = torch.tensor([
    [1.0, 0.0, 0.0],  # effort=0 (fast):   R only
    [0.0, 1.0, 0.0],  # effort=1 (medium): G only
    [0.0, 0.0, 1.0],  # effort=2 (slow):   B only
])


def c19_activation(x, rho=4.0):
    """C19 periodic parabolic wave activation (probe-validated: +4.2% vs GELU).
    Piecewise parabolic arcs with alternating sign, linear tails beyond 6*pi."""
    l = 6.0 * math.pi
    inv_pi = 1.0 / math.pi
    scaled = x * inv_pi
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = math.pi * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


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


def generate_phi_stride_assignments(num_ants: int, grid_size: int = 64,
                                     slots_per_ant: int = 1) -> torch.Tensor:
    """Assign each ant to slots_per_ant pixels via golden ratio stride.

    The golden angle ensures maximum spread: each new ant lands in
    the largest remaining gap, like sunflower seed packing.
    Collisions are resolved by advancing to the next phi-stride position.

    Args:
        num_ants: Number of ants (beings) to assign.
        grid_size: Total pixels in the grid (default 64 for 8x8).
        slots_per_ant: How many LCX slots each ant can write.

    Returns:
        Tensor of shape [num_ants, slots_per_ant] with pixel indices.
    """
    PHI = (1 + math.sqrt(5)) / 2
    total_slots = num_ants * slots_per_ant
    assert total_slots <= grid_size, \
        f"Cannot assign {total_slots} unique slots in grid of {grid_size}"
    used = set()
    positions = []
    i = 0
    while len(positions) < total_slots:
        pos = int((i * PHI * grid_size) % grid_size)
        if pos not in used:
            used.add(pos)
            positions.append(pos)
        i += 1
    return torch.tensor(positions, dtype=torch.long).view(num_ants, slots_per_ant)


class BeingParameters(nn.Module):
    """
    Per-being parameters (pointer destinations, jump gate, context strength).
    Each being in the swarm has its own instance of these.

    Phase embedding based on golden ratio (φ) for non-interfering specialization.
    Optional input_mask for receptive field restriction.
    """

    def __init__(self, num_memory_positions: int, embedding_dim: int,
                 being_id: int = 0, input_mask: torch.Tensor = None,
                 hidden_dim: int = None, num_pointers: int = 1):
        super().__init__()
        hdim = hidden_dim if hidden_dim is not None else embedding_dim
        self.hidden_dim = hdim
        self.num_pointers = num_pointers

        # Learned jump destinations for each memory position (Pointer A)
        self.pointer_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Jump gate: decides whether to jump or walk (Pointer A — explorer bias)
        self.jump_gate = nn.Linear(hdim, 1)
        nn.init.constant_(self.jump_gate.bias, 0.5)  # Start jumping (prevents dead gate)

        # Pointer B: independent destinations + jump gate (walker bias)
        if num_pointers >= 2:
            self.pointer_destinations_b = nn.Parameter(
                torch.rand(num_memory_positions) * num_memory_positions
            )
            self.jump_gate_b = nn.Linear(hdim, 1)
            nn.init.constant_(self.jump_gate_b.bias, -0.5)  # Walker bias (symmetry break)

        # Pointer C: independent destinations + jump gate (neutral bias)
        if num_pointers >= 3:
            self.pointer_destinations_c = nn.Parameter(
                torch.rand(num_memory_positions) * num_memory_positions
            )
            self.jump_gate_c = nn.Linear(hdim, 1)
            nn.init.constant_(self.jump_gate_c.bias, 0.0)  # Neutral (symmetry break)

        # Context strength: how much to weight memory reads (shared between pointers)
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
        depth: 'int | list' = 2,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
        combiner_mode: str = 'mean',
        num_bits: int = 8,
        bits_per_being: int = 0,
        min_coverage: int = 2,
        mask_seed: int = 42,
        fibonacci: bool = False,
        combinatorial: bool = False,
        think_ticks: 'int | list' = 0,
        temporal_fibonacci: bool = False,
        capacity_fibonacci: bool = False,
        max_hidden: int = 4096,
        min_hidden: int = 128,
        full_view: bool = False,
        use_lcx: bool = False,
        slots_per_being: int = 1,
        lcx_mode: str = "dense",
        lcx_num_slots: int = 256,
        lcx_key_dim: int = 64,
        lcx_top_k: int = 4,
        lcx_num_levels: int = 3,
        lcx_level_slots: list = None,
        num_pointers: int = 1,
        byte_token_mode: bool = False,
        being_modes: 'list | None' = None,
    ):
        super().__init__()

        self.byte_token_mode = byte_token_mode
        self.embedding_dim = embedding_dim
        self.num_memory_positions = num_memory_positions
        self.num_beings = num_beings
        self.num_bits = num_bits
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        # Per-being depth: expand int to list, validate list length
        if isinstance(depth, int):
            self.depth_per_being = [depth] * num_beings
        else:
            assert len(depth) == num_beings, \
                f"depth list length {len(depth)} != num_beings {num_beings}"
            self.depth_per_being = list(depth)
        self.depth = max(self.depth_per_being)  # backward compat
        self.combiner_mode = combiner_mode
        # Per-being think_ticks: expand int to list, validate list length
        if isinstance(think_ticks, int):
            self.think_ticks_per_being = [think_ticks] * num_beings
        else:
            assert len(think_ticks) == num_beings, \
                f"think_ticks list length {len(think_ticks)} != num_beings {num_beings}"
            self.think_ticks_per_being = list(think_ticks)
        self.think_ticks = max(self.think_ticks_per_being)  # backward compat
        self.full_view = full_view
        self.combinatorial = combinatorial
        self.use_lcx = use_lcx
        self.num_pointers = num_pointers
        self.effort_level = 0  # RGB LCX channel: 0=R(fast), 1=G(medium), 2=B(slow) — dense LCX only
        self._lcx_hash_mode = (use_lcx and lcx_mode == "hash")
        self.lcx_mode = lcx_mode
        self.lcx_num_slots = lcx_num_slots
        self.lcx_key_dim = lcx_key_dim
        self.lcx_top_k = lcx_top_k
        self._lcx_num_levels = lcx_num_levels if (use_lcx and lcx_mode == "hash") else 0
        self._lcx_level_slots = lcx_level_slots or ([int(lcx_num_slots * (10 ** i)) for i in range(lcx_num_levels)] if (use_lcx and lcx_mode == "hash") else [])
        # Level cap: tick N unlocks level N+1. Max active = think_ticks.
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
        # Dense LCX: input is broadcast 8x8 + LCX 8x8 (element-wise addition) = 64 values
        # Hash LCX: input bits only (LCX added in embedding space after projection)
        # GEM: input bits + GEM = num_bits * 2
        if use_lcx and lcx_mode == "hash":
            _input_width = num_bits  # hash LCX: raw bits only, LCX added post-projection
        elif use_lcx:
            _input_width = num_bits * num_bits  # dense LCX: broadcast grid
        else:
            _input_width = num_bits * 2  # GEM: bits + GEM
        # BYTE TOKEN MODE: discrete byte embedding + C19 bottleneck encoder/decoder
        # Replaces input_proj/output_proj with learned byte-level encoding.
        # Each byte (0-255) gets its own embedding vector — lossless by construction.
        # Output is 256-class classification (CrossEntropyLoss).
        if byte_token_mode:
            _bn = max(8, embedding_dim // 10)  # bottleneck dim (10:1 ratio)
            self._byte_bn_dim = _bn
            self._output_dim = 256
            # Encoder: Embedding(256, bn) → Lin → C19 → Lin → C19 → Lin → D
            self.byte_embed = nn.Embedding(256, _bn)
            self.byte_enc1 = nn.Linear(_bn, _bn)
            self.byte_enc2 = nn.Linear(_bn, embedding_dim)
            # Decoder: D → Lin → C19 → Lin → C19 → Lin → 256 logits
            self.byte_dec1 = nn.Linear(embedding_dim, _bn)
            self.byte_dec2 = nn.Linear(_bn, _bn)
            self.byte_dec3 = nn.Linear(_bn, 256)
            # Orthogonal init (lever 7)
            for layer in [self.byte_enc1, self.byte_enc2, self.byte_dec1, self.byte_dec2, self.byte_dec3]:
                nn.init.orthogonal_(layer.weight)
            self.input_proj = None
            self.output_proj = None
        else:
            self._output_dim = num_bits
            # Standard projections
            self.input_proj = nn.Linear(_input_width, embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, num_bits)

        # Multi-layer processing (if depth > 1)
        # Pre-LN + C19 residual blocks (standard transformer pattern)
        _all_same_depth = len(set(self.depth_per_being)) == 1
        if _all_same_depth and self.depth > 1:
            # Shared processing layers (all beings same depth)
            self.processing_layers = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(self.depth - 1)
            ])
            self.processing_norms = nn.ModuleList([
                nn.LayerNorm(embedding_dim)
                for _ in range(self.depth - 1)
            ])
        else:
            self.processing_layers = None
            self.processing_norms = None

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
        # Defaults (overridden below if heterogeneous or capacity_fibonacci)
        self.being_input_projs = None
        self.being_output_projs = None
        self.heterogeneous = self.receptive_masks is not None and bits_per_being > 0
        if self.heterogeneous:
            if use_lcx:
                # LCX v3: all ants get uniform 64-wide input (broadcast+add),
                # so we use the SHARED input_proj. Only need per-being OUTPUT projs.
                self.being_input_projs = None
                self.being_output_projs = nn.ModuleList()
                for i in range(num_beings):
                    if self.full_view:
                        self.being_output_projs.append(nn.Linear(embedding_dim, num_bits))
                    else:
                        k_i = int(self.receptive_masks[i].sum().item())
                        self.being_output_projs.append(nn.Linear(embedding_dim, k_i))
                # Keep shared input_proj (created above), disable shared output_proj
                self.output_proj = None
            else:
                # GEM path: per-being input AND output projections (variable width)
                self.being_input_projs = nn.ModuleList()
                self.being_output_projs = nn.ModuleList()
                for i in range(num_beings):
                    if self.full_view:
                        _fw = num_bits * 2
                        self.being_input_projs.append(nn.Linear(_fw, embedding_dim))
                        self.being_output_projs.append(nn.Linear(embedding_dim, num_bits))
                    else:
                        k_i = int(self.receptive_masks[i].sum().item())
                        _pw = k_i + num_bits
                        self.being_input_projs.append(nn.Linear(_pw, embedding_dim))
                        self.being_output_projs.append(nn.Linear(embedding_dim, k_i))
                # Shared projections not used in GEM heterogeneous mode
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

        # Being modes: 'thinker' (full gradient) or 'scanner' (no_grad, writes only)
        if being_modes is None:
            self.being_modes = ['thinker'] * num_beings
        else:
            assert len(being_modes) == num_beings, \
                f"being_modes length {len(being_modes)} != num_beings {num_beings}"
            assert all(m in ('thinker', 'scanner') for m in being_modes), \
                f"Invalid being_mode(s): {being_modes}. Must be 'thinker' or 'scanner'."
            self.being_modes = list(being_modes)

        # CAPACITY FIBONACCI: per-being hidden dimensions
        self.capacity_fibonacci = capacity_fibonacci
        if capacity_fibonacci:
            assert self.heterogeneous, (
                "capacity_fibonacci requires fibonacci=True with bits_per_being > 0"
            )
            k_values = fibonacci_k_schedule(num_beings, num_bits, min_k=2)
            self.hidden_dims = fibonacci_hidden_dims(k_values, max_hidden, min_hidden)

            # Ring bridge projections: D <-> H_i
            self.ring_read_projs = nn.ModuleList([
                nn.Linear(embedding_dim, h_i) for h_i in self.hidden_dims
            ])
            self.ring_write_projs = nn.ModuleList([
                nn.Linear(h_i, embedding_dim) for h_i in self.hidden_dims
            ])

            # Override being input/output projs: K_i <-> H_i (was K_i <-> D)
            if use_lcx and lcx_mode != 'hash':
                # Dense LCX v3: shared input proj (lcx_size → D), then per-being D → H_i bridge
                # Hash LCX uses num_bits-wide input (already set), not num_bits²
                self.input_proj = nn.Linear(num_bits * num_bits, embedding_dim)
                self.being_input_projs = nn.ModuleList([
                    nn.Linear(embedding_dim, h_i) for h_i in self.hidden_dims
                ])
                self.being_output_projs = nn.ModuleList()
                for i in range(num_beings):
                    h_i = self.hidden_dims[i]
                    if self.full_view:
                        self.being_output_projs.append(nn.Linear(h_i, num_bits))
                    else:
                        k_i = int(self.receptive_masks[i].sum().item())
                        self.being_output_projs.append(nn.Linear(h_i, k_i))
            else:
                self.being_input_projs = nn.ModuleList()
                self.being_output_projs = nn.ModuleList()
                for i in range(num_beings):
                    h_i = self.hidden_dims[i]
                    if self.full_view:
                        _fw = num_bits * 2
                        self.being_input_projs.append(nn.Linear(_fw, h_i))
                        self.being_output_projs.append(nn.Linear(h_i, num_bits))
                    else:
                        k_i = int(self.receptive_masks[i].sum().item())
                        _pw = k_i + num_bits
                        self.being_input_projs.append(nn.Linear(_pw, h_i))
                        self.being_output_projs.append(nn.Linear(h_i, k_i))

            # Per-being LayerNorm (hidden dims differ from embedding_dim)
            self.being_state_norms = nn.ModuleList([
                nn.LayerNorm(h_i) for h_i in self.hidden_dims
            ])

            # Per-being processing layers (can't share, dims differ)
            # Pre-LN + C19 residual blocks per being
            if depth > 1:
                self.being_processing_layers = nn.ModuleList()
                self.being_processing_norms = nn.ModuleList()
                for i in range(num_beings):
                    h_i = self.hidden_dims[i]
                    layers = nn.ModuleList([
                        nn.Linear(h_i, h_i) for _ in range(depth - 1)
                    ])
                    norms = nn.ModuleList([
                        nn.LayerNorm(h_i) for _ in range(depth - 1)
                    ])
                    self.being_processing_layers.append(layers)
                    self.being_processing_norms.append(norms)
                self.processing_layers = None  # Disable shared layers
                self.processing_norms = None
            else:
                self.being_processing_layers = None
                self.being_processing_norms = None
        else:
            self.hidden_dims = [embedding_dim] * num_beings
            self.ring_read_projs = None
            self.ring_write_projs = None
            self.being_state_norms = None
            # Heterogeneous depth: per-being processing layers (same D, different depth)
            if not _all_same_depth:
                self.being_processing_layers = nn.ModuleList()
                self.being_processing_norms = nn.ModuleList()
                for i in range(num_beings):
                    d_i = self.depth_per_being[i]
                    n_layers = max(0, d_i - 1)
                    layers = nn.ModuleList([
                        nn.Linear(embedding_dim, embedding_dim) for _ in range(n_layers)
                    ])
                    norms = nn.ModuleList([
                        nn.LayerNorm(embedding_dim) for _ in range(n_layers)
                    ])
                    self.being_processing_layers.append(layers)
                    self.being_processing_norms.append(norms)
            else:
                self.being_processing_layers = None
                self.being_processing_norms = None

        # PER-BEING COMPONENTS (each being has its own)
        # Pass being_id for golden ratio phase embeddings + optional input mask
        self.beings = nn.ModuleList([
            BeingParameters(
                num_memory_positions, embedding_dim, being_id=i,
                input_mask=self.receptive_masks[i] if self.receptive_masks is not None else None,
                hidden_dim=self.hidden_dims[i] if capacity_fibonacci else None,
                num_pointers=num_pointers,
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

        # MEMORY SYSTEM: LCX (grayscale 2D scratchpad) or GEM (legacy 1-channel EMA)
        if use_lcx and lcx_mode == "hash":
            # FLAT GOLDEN RATIO LCX: independent per-level buffers, flat cosine search.
            # Slot counts follow phi × 10^n: [618, 6180, 61800, ...].
            # Each level is searched independently — no parent-child, no quadtree.
            # Read: flat cosine similarity → top-K → softmax weighted sum.
            # Write: same routing → EMA blend at top-K slots.
            # Lazy allocation: only L0 is allocated at init (always needed for per-timestep read/write).
            # Higher levels (L1, L2, ...) are allocated on first access during think ticks.
            # This saves ~1.7 GB VRAM at Beta (L2 idle until Gamma).
            self._lcx_allocated_levels = set()
            for _lvl in range(self._lcx_num_levels):
                _n = self._lcx_level_slots[_lvl] if _lvl < len(self._lcx_level_slots) else self._lcx_level_slots[-1]
                if _lvl == 0:
                    _k = torch.randn(_n, lcx_key_dim)
                    _k = torch.nn.functional.normalize(_k, dim=-1)
                    self.register_buffer(f'lcx_keys_{_lvl}', _k)
                    self.register_buffer(f'lcx_values_{_lvl}', torch.randn(_n, embedding_dim) * 0.01)
                    self.register_buffer(f'lcx_heat_{_lvl}', torch.zeros(_n, dtype=torch.int16))
                    self.register_buffer(f'lcx_valid_{_lvl}', torch.zeros(_n, dtype=torch.bool))
                    self._lcx_allocated_levels.add(_lvl)
                else:
                    # Lazy: register as None, allocate on first access
                    self.register_buffer(f'lcx_keys_{_lvl}', None)
                    self.register_buffer(f'lcx_values_{_lvl}', None)
                    self.register_buffer(f'lcx_heat_{_lvl}', None)
                    self.register_buffer(f'lcx_valid_{_lvl}', None)
            # Initialize hash planes for L0 bucketed search (if large enough)
            _n0 = self._lcx_level_slots[0] if self._lcx_level_slots else 0
            if _n0 >= self._LCX_MIN_SLOTS_FOR_BUCKETING:
                self._lcx_init_hash_planes(0, _n0, _k.device, _k.dtype)
            # Temperature: tau sweep (probe #93) confirmed flat across [0.1, 2.0] — keep 1.0
            self._lcx_route_temps = [1.0 for _lvl in range(self._lcx_num_levels)]
            self._lcx_total_slots = sum(self._lcx_level_slots)
            self.lcx_route_query = nn.Linear(embedding_dim, lcx_key_dim)
            self.lcx_write_gate = nn.Linear(embedding_dim, 1)
            nn.init.constant_(self.lcx_write_gate.bias, 0.0)  # balanced init (sigmoid=0.5)
            # Zoom gate: auto-effort (model decides "need more detail?")
            self.zoom_gate = nn.Linear(embedding_dim, 1)
            nn.init.constant_(self.zoom_gate.bias, -4.0)  # sigmoid(-4)=1.8% — prevents noise while allowing gradient flow (was -5.0, frozen by AGC)
            # LCX read bottleneck: D → D//10 → C19 → D//10 → C19 → D
            # Translates raw LCX read vectors into hidden-compatible representations.
            _bn_dim = max(1, embedding_dim // 10)  # 618 at D=6180
            self.lcx_bn_layers = nn.ModuleList([
                nn.Linear(embedding_dim, _bn_dim),
                nn.Linear(_bn_dim, _bn_dim),
                nn.Linear(_bn_dim, embedding_dim),
            ])
            # Orthogonal init for bottleneck (probe: +0.8% vs kaiming, preserves norms through C19)
            for _bn_layer in self.lcx_bn_layers:
                nn.init.orthogonal_(_bn_layer.weight)
                nn.init.zeros_(_bn_layer.bias)
            # Think token: learned "I'm thinking" signal for think ticks
            self.think_token = nn.Parameter(torch.randn(embedding_dim) * 0.02)
            # Stubs for dense LCX layers (not used in hash mode)
            self.register_buffer('lcx', None)
            self.register_buffer('gem', None)
            self.lcx_propose = None
            self.lcx_gate = None
            self.gem_write_head = None
            self._phi_inv = None
            self.slots_per_being = -1
            self.register_buffer('pixel_assignments', None)
        elif use_lcx:
            # Dense LCX: num_bits × num_bits grayscale image (flat buffer)
            # Input = broadcast(8 bits → 8x8) + LCX(8x8), element-wise addition
            lcx_size = num_bits * num_bits
            self.register_buffer('lcx', torch.zeros(3, lcx_size))  # RGB: 3 effort-depth channels
            self.register_buffer('gem', None)
            self.lcx_propose = nn.Linear(embedding_dim, lcx_size)   # softsign Δ
            self.lcx_gate = nn.Linear(embedding_dim, lcx_size)      # sigmoid G
            nn.init.constant_(self.lcx_gate.bias, -0.486)           # logit(0.382) ≈ phi init
            self.gem_write_head = None
            self._phi_inv = None
            # Think token (also useful in dense mode)
            self.think_token = nn.Parameter(torch.randn(embedding_dim) * 0.02)
            # Hash LCX stubs (not used in dense mode)
            self.lcx_route_query = None
            self.lcx_write_gate = None
            self.lcx_bn_layers = None
            self.register_buffer('lcx_keys', None)
            self.register_buffer('lcx_values', None)
            # Phi-stride pixel assignments: per-slot ownership (VRA-88)
            # slots_per_being: 1=flowchart (one slot each), -1=global write (giant ant),
            #                  K=K phi-stride slots per being
            self.slots_per_being = slots_per_being
            if slots_per_being == -1 or slots_per_being >= lcx_size or num_beings == 1:
                # Global write: being writes to ALL LCX slots (auto for single-being)
                self.register_buffer('pixel_assignments', None)
            else:
                self.register_buffer(
                    'pixel_assignments',
                    generate_phi_stride_assignments(num_beings, lcx_size, slots_per_being),
                )
        else:
            # GEM: legacy 1-channel golden ratio EMA
            self.register_buffer('gem', torch.zeros(num_bits))
            self.register_buffer('lcx', None)
            self.gem_write_head = nn.Linear(embedding_dim, num_bits)
            self._phi_inv = 0.6180339887  # 1/φ — golden ratio inverse
            self.lcx_propose = None
            self.lcx_gate = None
            self.lcx_route_query = None
            self.lcx_write_gate = None
            self.lcx_bn_layers = None
            self.register_buffer('lcx_keys', None)
            self.register_buffer('lcx_values', None)
            self.think_token = None

        # Initialize attributes that may be set externally by live_controls / training loop
        # (avoids getattr fallbacks and makes the contract explicit)
        self._effort_auto = False
        self._allowed_levels = None
        self._eval_skip_think = False
        self._current_stage = 'INFANT'
        self._last_zoom_gate = 0.0

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
        """Compute uniform attention weights over window around pointer.

        Probe v1-v3 showed Gaussian shape adds zero value vs mean-pool
        (90.26% vs 90.20%, within noise). Uniform is simpler and ~8% faster.
        """
        B = pointer_position.size(0)
        K = self.attention_radius
        window_size = 2 * K + 1

        offsets = torch.arange(-K, K + 1, device=pointer_position.device)
        base = torch.floor(pointer_position).long().clamp(0, num_positions - 1)
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions

        # Uniform weights: mean-pool over neighborhood
        weights = torch.ones(B, window_size, device=pointer_position.device) / window_size

        return indices, weights

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

        # Clear LCX auxiliary loss accumulators for this forward pass
        if self.training and self._lcx_hash_mode:
            self._lcx_write_gate_accum = []
            self._lcx_read_weights_accum = []
            self._lcx_zoom_gate_accum = []
            self._lcx_score_margin_accum = []

        # Rebuild bucket indices for bucketed LCX search (once per forward pass)
        if self._lcx_hash_mode:
            self._lcx_rebuild_all_bucket_indices()

        # -- Gather per-being parameters (fresh each call for gradient flow) --
        if self.being_input_projs is not None:
            inp_w = torch.stack([p.weight for p in self.being_input_projs])   # [N, D, K]
            inp_b = torch.stack([p.bias for p in self.being_input_projs])     # [N, D]
        else:
            inp_w = None  # LCX v3: uses shared input_proj
            inp_b = None
        out_w = torch.stack([p.weight for p in self.being_output_projs])  # [N, K_out, D]
        out_b = torch.stack([p.bias for p in self.being_output_projs])    # [N, K_out]
        jg_w = torch.stack([b.jump_gate.weight for b in self.beings]).squeeze(1)  # [N, D]
        jg_b = torch.stack([b.jump_gate.bias for b in self.beings]).squeeze(1)    # [N]
        all_dests = torch.stack([b.pointer_destinations for b in self.beings])     # [N, M]
        if self.num_pointers >= 2:
            jg_w_b = torch.stack([b.jump_gate_b.weight for b in self.beings]).squeeze(1)  # [N, D]
            jg_b_b = torch.stack([b.jump_gate_b.bias for b in self.beings]).squeeze(1)    # [N]
            all_dests_b = torch.stack([b.pointer_destinations_b for b in self.beings])     # [N, M]
        if self.num_pointers >= 3:
            jg_w_c = torch.stack([b.jump_gate_c.weight for b in self.beings]).squeeze(1)  # [N, D]
            jg_b_c = torch.stack([b.jump_gate_c.bias for b in self.beings]).squeeze(1)    # [N]
            all_dests_c = torch.stack([b.pointer_destinations_c for b in self.beings])     # [N, M]
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
            if self.num_pointers >= 3:
                third = (hi - lo) / 3
                pointer_positions[n].uniform_(lo, lo + third)
            elif self.num_pointers >= 2:
                mid = (lo + hi) / 2
                pointer_positions[n].uniform_(lo, mid)
            else:
                pointer_positions[n].uniform_(lo, hi)

        if self.num_pointers >= 2:
            pointer_positions_b = torch.empty(N, B, device=device, dtype=dtype)
            for n in range(N):
                lo = n * M / N
                hi = (n + 1) * M / N
                if self.num_pointers >= 3:
                    third = (hi - lo) / 3
                    pointer_positions_b[n].uniform_(lo + third, lo + 2 * third)
                else:
                    mid = (lo + hi) / 2
                    pointer_positions_b[n].uniform_(mid, hi)

        if self.num_pointers >= 3:
            pointer_positions_c = torch.empty(N, B, device=device, dtype=dtype)
            for n in range(N):
                lo = n * M / N
                hi = (n + 1) * M / N
                third = (hi - lo) / 3
                pointer_positions_c[n].uniform_(lo + 2 * third, hi)

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

            # -- Temporal fibonacci dormancy mask --
            if self.tick_periods is not None:
                tick_active = (t % self.tick_periods) == 0  # [N]
            else:
                tick_active = torch.ones(N, device=device, dtype=torch.bool)
            step_active = active_mask & tick_active  # [N]
            step_mask = step_active.float().reshape(N, 1, 1)  # [N, 1, 1]

            # 1. Gather visible bits & project (with memory)
            if self._lcx_hash_mode:
                # Hash LCX: project raw bits, add LCX context in embedding space
                per_being = []
                for n_idx in range(N):
                    vis_mask = vis_indices[n_idx]  # [K]
                    vis_bits = input_bits[:, vis_mask]  # [B, K]
                    # Pad to full num_bits width for input_proj
                    full_bits = torch.zeros(B, self.num_bits, device=device, dtype=dtype)
                    full_bits[:, vis_mask] = vis_bits
                    proj = self.input_proj(full_bits)  # [B, D]
                    per_being.append(proj)
                being_input = torch.stack(per_being, dim=0)  # [N, B, D]
                # Add hash LCX context (same context for all beings — shared scratchpad)
                # Read once, broadcast to all N beings
                # NOTE: queries with mean of per-being masked inputs (differs from sequential
                # path which queries with full input per-being — equivalent for num_beings=1)
                lcx_ctx = self._lcx_flat_read(being_input.mean(dim=0), level=0)  # [B, D]
                being_input = being_input + self._lcx_bottleneck(lcx_ctx).unsqueeze(0)  # [N, B, D]
            elif self.lcx is not None:
                # Dense LCX: per-ant masked broadcast+add → shared projection
                per_being = []
                for n_idx in range(N):
                    vis_mask = vis_indices[n_idx]  # [K]
                    vis_bits = input_bits[:, vis_mask]  # [B, K]
                    combined = self._read_with_lcx(vis_bits, mask=vis_mask)  # [B, lcx_size]
                    proj = self.input_proj(combined)  # [B, D]
                    per_being.append(proj)
                being_input = torch.stack(per_being, dim=0)  # [N, B, D]
            elif self.gem is not None:
                _gem_exp_v = self.gem.unsqueeze(0).expand(B, -1)  # [B, num_bits]
                if self.full_view:
                    input_with_gem = torch.cat([input_bits, _gem_exp_v], dim=-1)  # [B, num_bits*2]
                    being_input = torch.einsum('bk,ndk->bnd', input_with_gem, inp_w) + inp_b
                else:
                    visible = input_bits[:, vis_indices]  # [B, N, K]
                    _gem_being = _gem_exp_v.unsqueeze(1).expand(-1, N, -1)
                    visible_with_gem = torch.cat([visible, _gem_being], dim=-1)  # [B, N, K+num_bits]
                    being_input = torch.einsum('bnk,ndk->bnd', visible_with_gem, inp_w) + inp_b
                being_input = being_input.permute(1, 0, 2)  # [N, B, D]
            else:
                # No memory active (INFANT mode: hash-mode model with LCX disabled)
                per_being = []
                for n_idx in range(N):
                    vis_mask = vis_indices[n_idx]  # [K]
                    vis_bits = input_bits[:, vis_mask]  # [B, K]
                    full_bits = torch.zeros(B, self.num_bits, device=device, dtype=dtype)
                    full_bits[:, vis_mask] = vis_bits
                    proj = self.input_proj(full_bits)  # [B, D]
                    per_being.append(proj)
                being_input = torch.stack(per_being, dim=0)  # [N, B, D]

            # 2. Batched ring read (all beings read SAME ring state)
            flat_ptrs_a = pointer_positions.reshape(-1)  # [N*B]
            indices_a, weights_a = self._gaussian_attention_weights(flat_ptrs_a, M)
            W = indices_a.size(1)  # 2*attention_radius + 1
            indices_a = indices_a.reshape(N, B, W)
            weights_a = weights_a.reshape(N, B, W)

            mem_exp = memory_ring.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, M, D]
            idx_gather_a = indices_a.unsqueeze(-1).expand(-1, -1, -1, D)  # [N, B, W, D]
            neighborhood_a = mem_exp.gather(2, idx_gather_a)  # [N, B, W, D]
            context_reads_a = (weights_a.unsqueeze(-1) * neighborhood_a).sum(dim=2)  # [N, B, D]

            if self.num_pointers >= 2:
                flat_ptrs_b = pointer_positions_b.reshape(-1)
                indices_b, weights_b = self._gaussian_attention_weights(flat_ptrs_b, M)
                indices_b = indices_b.reshape(N, B, W)
                weights_b = weights_b.reshape(N, B, W)
                idx_gather_b = indices_b.unsqueeze(-1).expand(-1, -1, -1, D)
                neighborhood_b = mem_exp.gather(2, idx_gather_b)
                context_reads_b = (weights_b.unsqueeze(-1) * neighborhood_b).sum(dim=2)

            if self.num_pointers >= 3:
                flat_ptrs_c = pointer_positions_c.reshape(-1)
                indices_c, weights_c = self._gaussian_attention_weights(flat_ptrs_c, M)
                indices_c = indices_c.reshape(N, B, W)
                weights_c = weights_c.reshape(N, B, W)
                idx_gather_c = indices_c.unsqueeze(-1).expand(-1, -1, -1, D)
                neighborhood_c = mem_exp.gather(2, idx_gather_c)
                context_reads_c = (weights_c.unsqueeze(-1) * neighborhood_c).sum(dim=2)

            if self.num_pointers >= 3:
                # Competitive attention: score each read against hidden state, softmax-select
                reads_stack = torch.stack([context_reads_a, context_reads_b, context_reads_c], dim=2)  # [N, B, 3, D]
                attn_scores = (hidden_states.unsqueeze(2) * reads_stack).sum(dim=-1)  # [N, B, 3]
                attn_scores = attn_scores / (D ** 0.5)
                attn_weights = 0.25 * torch.softmax(attn_scores, dim=-1) + 0.25  # [N, B, 3] min 25% per pointer
                context_reads = (attn_weights.unsqueeze(-1) * reads_stack).sum(dim=2)  # [N, B, D]
            elif self.num_pointers >= 2:
                context_reads = 0.5 * (context_reads_a + context_reads_b)
            else:
                context_reads = context_reads_a

            if return_stats:
                diag_ring_read_norm.append(context_reads.norm().item())

            # 3. Combine: input + scaled context + phase bias
            combined = being_input + ctx_scales * context_reads + 0.1 * phase_bias

            # 4. State update (phi EMA — norm hidden only, input path clean)
            state_update = STATE_EMA_ALPHA * self.state_norm(hidden_states) + (1 - STATE_EMA_ALPHA) * combined
            if self.processing_layers is not None:
                for norm, layer in zip(self.processing_norms, self.processing_layers):
                    state_update = state_update + c19_activation(layer(norm(state_update)))

            # Apply dormancy: keep old state for dormant/null beings
            hidden_states = torch.where(step_mask.bool(), state_update, hidden_states)

            # MEMORY WRITE (vectorized): per-slot ownership (VRA-88)
            if self._lcx_hash_mode:
                # Hash LCX write: route state_update to top-k slots
                # Average across beings for a single write per step
                # NOTE: sequential path writes per-being (equivalent for num_beings=1)
                write_state = hidden_states.mean(dim=0)    # [B, D]
                write_content = state_update.mean(dim=0)   # [B, D]
                self._lcx_flat_write(write_state, write_content, level=0)
            elif self.lcx is not None and self.pixel_assignments is not None:
                slots = self.pixel_assignments                          # [N, K]
                delta = torch.nn.functional.softsign(self.lcx_propose(state_update))  # [N, B, lcx_size]
                gate = torch.sigmoid(self.lcx_gate(state_update))                     # [N, B, lcx_size]
                delta_b = delta.mean(dim=1)                             # [N, lcx_size]
                gate_b = gate.mean(dim=1)                               # [N, lcx_size]
                # Gather each being's assigned K slots
                N_active, K = slots.shape
                arange_n = torch.arange(N_active, device=delta.device).unsqueeze(1).expand(-1, K)
                delta_at_slots = delta_b[arange_n, slots]               # [N, K]
                gate_at_slots = gate_b[arange_n, slots]                 # [N, K]
                # Flatten for scatter: [N*K] unique slots (phi-stride guarantees no overlap)
                flat_slots = slots.reshape(-1)                          # [N*K]
                flat_delta = delta_at_slots.reshape(-1)                 # [N*K]
                flat_gate = gate_at_slots.reshape(-1)                   # [N*K]
                w = EFFORT_WEIGHTS[self.effort_level].to(self.lcx.device)  # [3]
                new_lcx = self.lcx.detach().clone()
                for c in range(3):
                    old_c = self.lcx[c].detach()
                    old_at_slots = old_c[flat_slots]
                    new_at_slots = (1.0 - flat_gate * w[c]) * old_at_slots + flat_gate * w[c] * flat_delta
                    new_lcx[c] = old_c.scatter(0, flat_slots, new_at_slots)
                self.lcx = new_lcx.clamp(-1.0, 1.0)
            elif self.lcx is not None:
                # Legacy global write (backwards compat when pixel_assignments missing)
                delta = torch.nn.functional.softsign(self.lcx_propose(state_update))  # [N, B, lcx_size]
                gate = torch.sigmoid(self.lcx_gate(state_update))                      # [N, B, lcx_size]
                delta_avg = delta.mean(dim=0).mean(dim=0)  # [lcx_size]
                gate_avg = gate.mean(dim=0).mean(dim=0)    # [lcx_size]
                w = EFFORT_WEIGHTS[self.effort_level].to(self.lcx.device)  # [3]
                new_lcx = self.lcx.detach().clone()
                for c in range(3):
                    old_c = self.lcx[c].detach()
                    new_lcx[c] = (1.0 - gate_avg * w[c]) * old_c + gate_avg * w[c] * delta_avg
                self.lcx = new_lcx.clamp(-1.0, 1.0)
            elif self.gem_write_head is not None and self.gem is not None:
                _gem_sig_v = torch.nn.functional.softsign(self.gem_write_head(state_update))  # [N, B, num_bits]
                _gem_upd_v = _gem_sig_v.mean(dim=0).mean(dim=0)  # average N,B → [num_bits]
                self.gem = self._phi_inv * self.gem.detach() + (1.0 - self._phi_inv) * _gem_upd_v
            # else: no memory write (INFANT mode: hash-mode model with LCX disabled)

            if return_stats:
                diag_hidden_abs.append(hidden_states.abs().mean().item())
                diag_hidden_max.append(hidden_states.abs().max().item())

            # 5. Ring write — Pointer A (all active beings simultaneously via single scatter_add)
            write_contrib_a = (
                hidden_states.unsqueeze(2).expand(-1, -1, W, -1)
                * weights_a.unsqueeze(-1)
                * step_mask.unsqueeze(2)  # zero out dormant writes
            )  # [N, B, W, D]

            # Flatten beings into scatter dimension: [B, N*W, D]
            idx_scatter_a = (
                indices_a.permute(1, 0, 2)
                .reshape(B, -1)
                .unsqueeze(-1)
                .expand(-1, -1, D)
            )  # [B, N*W, D]
            contribs_flat_a = write_contrib_a.permute(1, 0, 2, 3).reshape(B, -1, D)
            memory_ring = memory_ring.scatter_add(1, idx_scatter_a, contribs_flat_a)

            if self.num_pointers >= 2:
                # Ring write — Pointer B
                write_contrib_b = (
                    hidden_states.unsqueeze(2).expand(-1, -1, W, -1)
                    * weights_b.unsqueeze(-1)
                    * step_mask.unsqueeze(2)
                )
                idx_scatter_b = (
                    indices_b.permute(1, 0, 2)
                    .reshape(B, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, D)
                )
                contribs_flat_b = write_contrib_b.permute(1, 0, 2, 3).reshape(B, -1, D)
                memory_ring = memory_ring.scatter_add(1, idx_scatter_b, contribs_flat_b)

            if self.num_pointers >= 3:
                # Ring write — Pointer C
                write_contrib_c = (
                    hidden_states.unsqueeze(2).expand(-1, -1, W, -1)
                    * weights_c.unsqueeze(-1)
                    * step_mask.unsqueeze(2)
                )
                idx_scatter_c = (
                    indices_c.permute(1, 0, 2)
                    .reshape(B, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, D)
                )
                contribs_flat_c = write_contrib_c.permute(1, 0, 2, 3).reshape(B, -1, D)
                memory_ring = memory_ring.scatter_add(1, idx_scatter_c, contribs_flat_c)

            # 6. Pointer A update (jump or walk) — soft blend for gradient flow
            current_pos_a = pointer_positions.long().clamp(0, M - 1)  # [N, B]
            jump_targets_a = all_dests.gather(1, current_pos_a).float().clamp(0, M - 1)  # [N, B]
            jump_logits_a = (
                (hidden_states * jg_w.unsqueeze(1)).sum(-1) + jg_b.unsqueeze(1)
            )  # [N, B]
            jump_prob_a = torch.sigmoid(jump_logits_a / JUMP_SIGMOID_TAU)
            walk_pos_a = (pointer_positions + 1.0) % M
            new_ptrs_a = jump_prob_a * jump_targets_a + (1.0 - jump_prob_a) * walk_pos_a
            pointer_positions = torch.where(step_active.unsqueeze(1), new_ptrs_a, pointer_positions)

            if self.num_pointers >= 2:
                # Pointer B update (independent walker)
                current_pos_b = pointer_positions_b.long().clamp(0, M - 1)
                jump_targets_b = all_dests_b.gather(1, current_pos_b).float().clamp(0, M - 1)
                jump_logits_b = (
                    (hidden_states * jg_w_b.unsqueeze(1)).sum(-1) + jg_b_b.unsqueeze(1)
                )
                jump_prob_b = torch.sigmoid(jump_logits_b / JUMP_SIGMOID_TAU)
                walk_pos_b = (pointer_positions_b + 1.0) % M
                new_ptrs_b = jump_prob_b * jump_targets_b + (1.0 - jump_prob_b) * walk_pos_b
                pointer_positions_b = torch.where(step_active.unsqueeze(1), new_ptrs_b, pointer_positions_b)

            if self.num_pointers >= 3:
                # Pointer C update (neutral)
                current_pos_c = pointer_positions_c.long().clamp(0, M - 1)
                jump_targets_c = all_dests_c.gather(1, current_pos_c).float().clamp(0, M - 1)
                jump_logits_c = (
                    (hidden_states * jg_w_c.unsqueeze(1)).sum(-1) + jg_b_c.unsqueeze(1)
                )
                jump_prob_c = torch.sigmoid(jump_logits_c / JUMP_SIGMOID_TAU)
                walk_pos_c = (pointer_positions_c + 1.0) % M
                new_ptrs_c = jump_prob_c * jump_targets_c + (1.0 - jump_prob_c) * walk_pos_c
                pointer_positions_c = torch.where(step_active.unsqueeze(1), new_ptrs_c, pointer_positions_c)

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
                    elif self.num_pointers >= 3:
                        avg_jp = (jump_prob_a[n].mean().item() + jump_prob_b[n].mean().item() + jump_prob_c[n].mean().item()) / 3.0
                        jump_counts_per_being[n].append(avg_jp)
                    elif self.num_pointers >= 2:
                        avg_jp = 0.5 * (jump_prob_a[n].mean().item() + jump_prob_b[n].mean().item())
                        jump_counts_per_being[n].append(avg_jp)
                    else:
                        jump_counts_per_being[n].append(jump_prob_a[n].mean().item())
                ptr_t = [
                    pointer_positions[n].mean().item()
                    for n in range(N) if self._is_contributing(n)
                ]
                if self.num_pointers >= 2:
                    ptr_t.extend([
                        pointer_positions_b[n].mean().item()
                        for n in range(N) if self._is_contributing(n)
                    ])
                if self.num_pointers >= 3:
                    ptr_t.extend([
                        pointer_positions_c[n].mean().item()
                        for n in range(N) if self._is_contributing(n)
                    ])
                pointer_positions_log.append(ptr_t)

            # -------- THINK TICKS --------
            _eff_think_ticks = 0 if (not self.training and getattr(self, '_eval_skip_think', False)) else self.think_ticks
            if _eff_think_ticks > 0:
                for _tt in range(_eff_think_ticks):
                    # Per-being think tick quota masking
                    _tt_limit = torch.tensor(self.think_ticks_per_being, device=device)  # [N]
                    tt_being_active = (_tt < _tt_limit)  # [N] bool

                    if self.tick_periods is not None:
                        _tt_per = torch.tensor(self.think_ticks_per_being, device=device)
                        global_tick = t * (1 + _tt_per) + 1 + _tt  # [N] tensor
                        tt_active = (global_tick % self.tick_periods) == 0
                        tt_step = active_mask & tt_active & tt_being_active
                    else:
                        tt_step = active_mask & tt_being_active
                    tt_mask = tt_step.float().reshape(N, 1, 1)

                    # Ring read — Pointer A
                    flat_p_a = pointer_positions.reshape(-1)
                    idx_tt_a, wgt_tt_a = self._gaussian_attention_weights(flat_p_a, M)
                    W_tt = idx_tt_a.size(1)
                    idx_tt_a = idx_tt_a.reshape(N, B, W_tt)
                    wgt_tt_a = wgt_tt_a.reshape(N, B, W_tt)

                    mem_tt = memory_ring.unsqueeze(0).expand(N, -1, -1, -1)
                    idxg_tt_a = idx_tt_a.unsqueeze(-1).expand(-1, -1, -1, D)
                    nbr_tt_a = mem_tt.gather(2, idxg_tt_a)
                    ctx_tt_a = (wgt_tt_a.unsqueeze(-1) * nbr_tt_a).sum(dim=2)  # [N, B, D]

                    if self.num_pointers >= 2:
                        # Ring read — Pointer B
                        flat_p_b = pointer_positions_b.reshape(-1)
                        idx_tt_b, wgt_tt_b = self._gaussian_attention_weights(flat_p_b, M)
                        idx_tt_b = idx_tt_b.reshape(N, B, W_tt)
                        wgt_tt_b = wgt_tt_b.reshape(N, B, W_tt)
                        idxg_tt_b = idx_tt_b.unsqueeze(-1).expand(-1, -1, -1, D)
                        nbr_tt_b = mem_tt.gather(2, idxg_tt_b)
                        ctx_tt_b = (wgt_tt_b.unsqueeze(-1) * nbr_tt_b).sum(dim=2)

                    if self.num_pointers >= 3:
                        # Ring read — Pointer C
                        flat_p_c = pointer_positions_c.reshape(-1)
                        idx_tt_c, wgt_tt_c = self._gaussian_attention_weights(flat_p_c, M)
                        idx_tt_c = idx_tt_c.reshape(N, B, W_tt)
                        wgt_tt_c = wgt_tt_c.reshape(N, B, W_tt)
                        idxg_tt_c = idx_tt_c.unsqueeze(-1).expand(-1, -1, -1, D)
                        nbr_tt_c = mem_tt.gather(2, idxg_tt_c)
                        ctx_tt_c = (wgt_tt_c.unsqueeze(-1) * nbr_tt_c).sum(dim=2)

                    if self.num_pointers >= 3:
                        # Competitive attention: score each read against hidden state
                        reads_tt = torch.stack([ctx_tt_a, ctx_tt_b, ctx_tt_c], dim=2)  # [N, B, 3, D]
                        attn_s_tt = (hidden_states.unsqueeze(2) * reads_tt).sum(dim=-1) / (D ** 0.5)
                        attn_w_tt = 0.25 * torch.softmax(attn_s_tt, dim=-1) + 0.25  # [N, B, 3] min 25% per pointer
                        ctx_tt = (attn_w_tt.unsqueeze(-1) * reads_tt).sum(dim=2)
                    elif self.num_pointers >= 2:
                        ctx_tt = 0.5 * (ctx_tt_a + ctx_tt_b)
                    else:
                        ctx_tt = ctx_tt_a

                    # No input — context + phase + think token
                    comb_tt = ctx_scales * ctx_tt + 0.1 * phase_bias
                    if self.think_token is not None:
                        comb_tt = comb_tt + self.think_token  # "I'm thinking" signal
                    su_tt = STATE_EMA_ALPHA * self.state_norm(hidden_states) + (1 - STATE_EMA_ALPHA) * comb_tt
                    if self.processing_layers is not None:
                        for norm, layer in zip(self.processing_norms, self.processing_layers):
                            su_tt = su_tt + c19_activation(layer(norm(su_tt)))
                    hidden_states = torch.where(tt_mask.bool(), su_tt, hidden_states)

                    # Flat LCX read/write during think ticks
                    # Each tick reads the next level up, clamped to top level.
                    # tt=0 → L1, tt=1 → L2, ..., tt>=top → re-read top (iterative retrieval).
                    if self._lcx_hash_mode:
                        _lcx_level = min(_tt + 1, self._lcx_num_levels - 1)
                        _do_lcx = self._lcx_num_levels > 0
                        if _do_lcx:
                            _allowed = getattr(self, '_allowed_levels', None)
                            if _allowed is not None and _lcx_level not in _allowed:
                                _do_lcx = False
                        if _do_lcx:
                            _query = hidden_states.mean(dim=0)
                            lcx_tt_ctx = self._lcx_flat_read(_query, level=_lcx_level)
                            lcx_tt_ctx = self._lcx_bottleneck(lcx_tt_ctx)
                            # zoom_gate: L2-normalize input to bound pre-activation
                            with torch.amp.autocast('cuda', enabled=False):
                                _zg_dtype = self.zoom_gate.weight.dtype
                                _q32 = torch.nn.functional.normalize(_query.to(_zg_dtype), dim=-1)
                                zg = torch.sigmoid(self.zoom_gate(_q32))
                            lcx_tt_ctx = lcx_tt_ctx * zg
                            self._last_zoom_gate = zg.mean().item()
                            # Stash zoom_gate output for anti-saturation aux loss
                            if self.training:
                                if not hasattr(self, '_lcx_zoom_gate_accum'):
                                    self._lcx_zoom_gate_accum = []
                                self._lcx_zoom_gate_accum.append(zg)
                            # Early-stop during eval only
                            if not self.training and zg.max().item() < 0.1:
                                break
                            hidden_states = hidden_states + lcx_tt_ctx.unsqueeze(0) * tt_mask
                            # Write at this level
                            _wv = hidden_states.mean(dim=0)
                            _wk = su_tt.mean(dim=0)
                            self._lcx_flat_write(_wv, _wk, level=_lcx_level)

                    # Ring write — Pointer A
                    wc_tt_a = (
                        hidden_states.unsqueeze(2).expand(-1, -1, W_tt, -1)
                        * wgt_tt_a.unsqueeze(-1) * tt_mask.unsqueeze(2)
                    )
                    ids_tt_a = idx_tt_a.permute(1, 0, 2).reshape(B, -1).unsqueeze(-1).expand(-1, -1, D)
                    cfs_tt_a = wc_tt_a.permute(1, 0, 2, 3).reshape(B, -1, D)
                    memory_ring = memory_ring.scatter_add(1, ids_tt_a, cfs_tt_a)

                    if self.num_pointers >= 2:
                        # Ring write — Pointer B
                        wc_tt_b = (
                            hidden_states.unsqueeze(2).expand(-1, -1, W_tt, -1)
                            * wgt_tt_b.unsqueeze(-1) * tt_mask.unsqueeze(2)
                        )
                        ids_tt_b = idx_tt_b.permute(1, 0, 2).reshape(B, -1).unsqueeze(-1).expand(-1, -1, D)
                        cfs_tt_b = wc_tt_b.permute(1, 0, 2, 3).reshape(B, -1, D)
                        memory_ring = memory_ring.scatter_add(1, ids_tt_b, cfs_tt_b)

                    if self.num_pointers >= 3:
                        # Ring write — Pointer C
                        wc_tt_c = (
                            hidden_states.unsqueeze(2).expand(-1, -1, W_tt, -1)
                            * wgt_tt_c.unsqueeze(-1) * tt_mask.unsqueeze(2)
                        )
                        ids_tt_c = idx_tt_c.permute(1, 0, 2).reshape(B, -1).unsqueeze(-1).expand(-1, -1, D)
                        cfs_tt_c = wc_tt_c.permute(1, 0, 2, 3).reshape(B, -1, D)
                        memory_ring = memory_ring.scatter_add(1, ids_tt_c, cfs_tt_c)

                    # Pointer A update — soft blend for gradient flow
                    cp_tt_a = pointer_positions.long().clamp(0, M - 1)
                    jt_tt_a = all_dests.gather(1, cp_tt_a).float().clamp(0, M - 1)
                    jl_tt_a = (hidden_states * jg_w.unsqueeze(1)).sum(-1) + jg_b.unsqueeze(1)
                    jp_tt_a = torch.sigmoid(jl_tt_a / JUMP_SIGMOID_TAU)
                    wp_tt_a = (pointer_positions + 1.0) % M
                    np_tt_a = jp_tt_a * jt_tt_a + (1.0 - jp_tt_a) * wp_tt_a
                    pointer_positions = torch.where(tt_step.unsqueeze(1), np_tt_a, pointer_positions)

                    if self.num_pointers >= 2:
                        # Pointer B update (independent walker)
                        cp_tt_b = pointer_positions_b.long().clamp(0, M - 1)
                        jt_tt_b = all_dests_b.gather(1, cp_tt_b).float().clamp(0, M - 1)
                        jl_tt_b = (hidden_states * jg_w_b.unsqueeze(1)).sum(-1) + jg_b_b.unsqueeze(1)
                        jp_tt_b = torch.sigmoid(jl_tt_b / JUMP_SIGMOID_TAU)
                        wp_tt_b = (pointer_positions_b + 1.0) % M
                        np_tt_b = jp_tt_b * jt_tt_b + (1.0 - jp_tt_b) * wp_tt_b
                        pointer_positions_b = torch.where(tt_step.unsqueeze(1), np_tt_b, pointer_positions_b)

                    if self.num_pointers >= 3:
                        # Pointer C update (neutral)
                        cp_tt_c = pointer_positions_c.long().clamp(0, M - 1)
                        jt_tt_c = all_dests_c.gather(1, cp_tt_c).float().clamp(0, M - 1)
                        jl_tt_c = (hidden_states * jg_w_c.unsqueeze(1)).sum(-1) + jg_b_c.unsqueeze(1)
                        jp_tt_c = torch.sigmoid(jl_tt_c / JUMP_SIGMOID_TAU)
                        wp_tt_c = (pointer_positions_c + 1.0) % M
                        np_tt_c = jp_tt_c * jt_tt_c + (1.0 - jp_tt_c) * wp_tt_c
                        pointer_positions_c = torch.where(tt_step.unsqueeze(1), np_tt_c, pointer_positions_c)

                    # Detach between think ticks (VRAM fix): only last tick gets gradient
                    if _tt < _eff_think_ticks - 1:
                        hidden_states = hidden_states.detach()
                        memory_ring = memory_ring.detach()

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

        # LCX auxiliary losses for gradient flow
        if self.training and self._lcx_hash_mode:
            _wg_acc = getattr(self, '_lcx_write_gate_accum', [])
            if _wg_acc:
                _all_gates = torch.cat([g.reshape(-1) for g in _wg_acc])
                self._lcx_write_gate_aux_loss = -torch.log(_all_gates.mean() + 1e-8)
            else:
                self._lcx_write_gate_aux_loss = torch.tensor(0.0, device=output.device)
            _rw_acc = getattr(self, '_lcx_read_weights_accum', [])
            if _rw_acc:
                _all_w = torch.cat([w.reshape(-1, w.shape[-1]) for w in _rw_acc], dim=0)
                self._lcx_read_attn_aux_loss = -torch.log(_all_w + 1e-8).mean()
            else:
                self._lcx_read_attn_aux_loss = torch.tensor(0.0, device=output.device)
            # Zoom gate anti-saturation: binary entropy loss keeps gate alive
            # Maximized at zg=0.5, zero at zg=0 or zg=1.  Negated → minimize → push toward 0.5.
            _zg_acc = getattr(self, '_lcx_zoom_gate_accum', [])
            if _zg_acc:
                _all_zg = torch.cat([z.reshape(-1) for z in _zg_acc])
                _zg_clamped = _all_zg.clamp(1e-6, 1.0 - 1e-6)
                self._lcx_zoom_gate_aux_loss = -(
                    _zg_clamped * torch.log(_zg_clamped) +
                    (1 - _zg_clamped) * torch.log(1 - _zg_clamped)
                ).mean()
            else:
                self._lcx_zoom_gate_aux_loss = torch.tensor(0.0, device=output.device)
            # Score margin telemetry (not a loss — pure diagnostic)
            _sm_acc = getattr(self, '_lcx_score_margin_accum', [])
            if _sm_acc:
                self._last_score_margin = sum(m for m, _ in _sm_acc) / len(_sm_acc)
                self._last_score_top1 = sum(t for _, t in _sm_acc) / len(_sm_acc)
            else:
                self._last_score_margin = 0.0
                self._last_score_top1 = 0.0
        else:
            self._lcx_write_gate_aux_loss = torch.tensor(0.0, device=output.device)
            self._lcx_read_attn_aux_loss = torch.tensor(0.0, device=output.device)
            self._lcx_zoom_gate_aux_loss = torch.tensor(0.0, device=output.device)

        # Detach LCX buffers — gradient highway is within-step only
        # Per-level buffers: writes are in-place with no_grad, safety net.
        # Skip unallocated levels (lazy allocation — they are None).
        if self._lcx_hash_mode:
            _alloc = getattr(self, '_lcx_allocated_levels', set(range(self._lcx_num_levels)))
            for _lvl in range(self._lcx_num_levels):
                if _lvl not in _alloc:
                    continue
                _k, _v = self._lcx_level_bufs(_lvl)
                if _v.requires_grad:
                    setattr(self, f'lcx_values_{_lvl}', _v.detach())
                if _k.requires_grad:
                    setattr(self, f'lcx_keys_{_lvl}', _k.detach())

        return output, stats

    def _read_with_lcx(self, input_bits, mask=None):
        """Element-wise addition: broadcast(input) + LCX.

        LCX v3: broadcast each input bit across its row to form an 8x8 grid,
        then add element-wise to the LCX 8x8 grid. Spatial binding: bit i
        lives in row i of the combined matrix.

        Args:
            input_bits: [B, num_bits] full input, or [B, K_i] if masked
            mask: optional index tensor [K_i] mapping masked bits to positions.
                  If provided, unseen bit rows get 0 + LCX (LCX only).

        Returns: [B, lcx_size] combined = input_8x8 + LCX_8x8 (flattened)
        """
        B = input_bits.size(0)
        nb = self.num_bits

        # Build full-width input (unseen bits = 0)
        if mask is not None:
            full_input = torch.zeros(B, nb, device=input_bits.device, dtype=input_bits.dtype)
            full_input[:, mask] = input_bits
        else:
            full_input = input_bits  # [B, num_bits]

        # Broadcast: each bit fills its row → [B, 8, 8] → [B, 64]
        input_grid = full_input.unsqueeze(-1).expand(-1, -1, nb).reshape(B, -1)

        # Weighted blend of ALL 3 RGB channels (golden-ratio weights by effort level)
        w = EFFORT_WEIGHTS[self.effort_level].to(self.lcx.device)  # [3]
        active_lcx = (self.lcx * w.view(3, 1)).sum(dim=0)          # [lcx_size]
        lcx_expanded = active_lcx.unsqueeze(0).expand(B, -1)      # [B, lcx_size]
        return input_grid + lcx_expanded  # [B, lcx_size]

    def _lcx_level_bufs(self, level: int):
        """Return (keys, values) for a given zoom level.
        Lazy-allocates the level if not yet on GPU.
        In double-buffer mode, returns golden buffers during reads."""
        level = min(level, self._lcx_num_levels - 1)
        # Double-buffer: redirect reads to golden copy
        if getattr(self, '_db_mode', False) and getattr(self, '_db_reading', False):
            g = self._db_golden
            return g[f'L{level}_keys'], g[f'L{level}_values']
        if hasattr(self, '_lcx_allocated_levels') and level not in self._lcx_allocated_levels:
            self._allocate_lcx_level(level)
        return getattr(self, f'lcx_keys_{level}'), getattr(self, f'lcx_values_{level}')

    # --- Bucketed LCX search infrastructure ---

    _LCX_TARGET_BUCKET_SIZE = 512   # aim for ~512 slots per bucket
    _LCX_MIN_SLOTS_FOR_BUCKETING = 1024  # don't bucket tiny levels

    def _lcx_compute_bucket(self, vec: torch.Tensor, level: int) -> torch.Tensor:
        """SimHash: project onto random planes, convert sign bits to bucket ID.
        vec: [..., key_dim] -> [...] int64 bucket IDs."""
        planes = getattr(self, f'_lcx_hash_planes_{level}', None)
        if planes is None:
            return torch.zeros(vec.shape[:-1], dtype=torch.long, device=vec.device)
        bits = (vec @ planes > 0).long()                       # [..., num_bits]
        powers = getattr(self, f'_lcx_hash_powers_{level}')    # [num_bits]
        return (bits * powers).sum(dim=-1)                      # [...]

    def _lcx_build_bucket_index(self, level: int):
        """Build padded bucket index from current keys. Called once per forward pass."""
        keys = getattr(self, f'lcx_keys_{level}', None)
        if keys is None:
            return
        _n = keys.shape[0]
        if _n < self._LCX_MIN_SLOTS_FOR_BUCKETING:
            # Small level: no bucketing, set flag
            setattr(self, f'_lcx_bucketed_{level}', False)
            return

        # Ensure hash planes exist (created at allocation, but needed for L0 init too)
        if not hasattr(self, f'_lcx_hash_planes_{level}') or getattr(self, f'_lcx_hash_planes_{level}', None) is None:
            self._lcx_init_hash_planes(level, _n, keys.device, keys.dtype)

        num_buckets = getattr(self, f'_lcx_num_buckets_{level}')
        bucket_ids = self._lcx_compute_bucket(keys, level)         # [S]

        # Build padded index: [num_buckets, max_bucket_size]
        counts = torch.zeros(num_buckets, dtype=torch.long, device=keys.device)
        for b_id in range(num_buckets):
            counts[b_id] = (bucket_ids == b_id).sum()
        max_bucket_size = int(counts.max().item())
        if max_bucket_size == 0:
            max_bucket_size = 1  # edge case: empty buckets

        bucket_index = torch.zeros(num_buckets, max_bucket_size,
                                   dtype=torch.long, device=keys.device)
        fill_pos = torch.zeros(num_buckets, dtype=torch.long, device=keys.device)
        # Vectorized scatter: sort slots by bucket
        sorted_buckets, sort_order = bucket_ids.sort()
        for i in range(num_buckets):
            mask = sorted_buckets == i
            indices = sort_order[mask]
            cnt = indices.shape[0]
            if cnt > 0:
                bucket_index[i, :cnt] = indices

        setattr(self, f'_lcx_bucket_ids_{level}', bucket_ids)
        setattr(self, f'_lcx_bucket_index_{level}', bucket_index)
        setattr(self, f'_lcx_bucket_counts_{level}', counts)
        setattr(self, f'_lcx_max_bucket_size_{level}', max_bucket_size)
        setattr(self, f'_lcx_bucketed_{level}', True)

    def _lcx_init_hash_planes(self, level: int, num_slots: int, device, dtype):
        """Create SimHash random projection planes for a level."""
        num_buckets = max(1, num_slots // self._LCX_TARGET_BUCKET_SIZE)
        num_hash_bits = max(1, int(math.ceil(math.log2(num_buckets)))) if num_buckets > 1 else 1
        num_buckets = 2 ** num_hash_bits  # round up to power of 2

        planes = torch.randn(self.lcx_key_dim, num_hash_bits, device=device, dtype=dtype)
        planes = torch.nn.functional.normalize(planes, dim=0)
        powers = (2 ** torch.arange(num_hash_bits, device=device)).long()

        # register_buffer so planes/powers survive .to(device) calls
        self.register_buffer(f'_lcx_hash_planes_{level}', planes, persistent=False)
        self.register_buffer(f'_lcx_hash_powers_{level}', powers, persistent=False)
        setattr(self, f'_lcx_num_buckets_{level}', num_buckets)
        print(f"  [LCX] L{level} bucketed search: {num_slots} slots / {num_buckets} buckets "
              f"= ~{num_slots // num_buckets} slots/bucket ({num_hash_bits}-bit SimHash)", flush=True)

    def _lcx_rebuild_all_bucket_indices(self):
        """Rebuild bucket indices for all allocated levels. Call once per forward pass."""
        if not hasattr(self, '_lcx_allocated_levels'):
            return
        for lvl in self._lcx_allocated_levels:
            self._lcx_build_bucket_index(lvl)

    def _allocate_lcx_level(self, level: int):
        """Lazy-allocate an LCX level's buffers on GPU. Called on first access."""
        if level in self._lcx_allocated_levels:
            return
        _n = self._lcx_level_slots[level] if level < len(self._lcx_level_slots) else self._lcx_level_slots[-1]
        device = self.lcx_keys_0.device
        dtype = self.lcx_keys_0.dtype

        _k = torch.randn(_n, self.lcx_key_dim, device=device, dtype=dtype)
        _k = torch.nn.functional.normalize(_k, dim=-1)
        setattr(self, f'lcx_keys_{level}', _k)
        setattr(self, f'lcx_values_{level}',
                torch.randn(_n, self.embedding_dim, device=device, dtype=dtype) * 0.01)
        setattr(self, f'lcx_heat_{level}',
                torch.zeros(_n, dtype=torch.int16, device=device))
        setattr(self, f'lcx_valid_{level}',
                torch.zeros(_n, dtype=torch.bool, device=device))
        self._lcx_allocated_levels.add(level)

        _mb = _n * (self.lcx_key_dim + self.embedding_dim) * 4 / 1024 / 1024
        print(f"  [LCX] Lazy-allocated L{level}: {_n} slots ({_mb:.0f} MB)", flush=True)

        # Initialize hash planes + bucket index for bucketed search
        if _n >= self._LCX_MIN_SLOTS_FOR_BUCKETING:
            self._lcx_init_hash_planes(level, _n, device, dtype)
            self._lcx_build_bucket_index(level)

    def resize_lcx(self, new_slots: int):
        """Resize LCX to a single level with new_slots slots, preserving memories.

        Handles all cases:
        - Growing 1 level (copy old, pad with empty)
        - Shrinking 1 level (keep hottest slots)
        - Merging N levels → 1 level (concatenate all, then resize)

        Memories are preserved (hottest-first if shrinking).
        Hash planes and bucket indices are rebuilt for the new size.
        Model weights (lcx_route_query, zoom_gate, etc.) are untouched.

        Call during training when hitting a plateau:
            model.resize_lcx(200_000)  # grow from 100K to 200K
        """
        if not self._lcx_hash_mode:
            print("  [LCX] resize_lcx: LCX hash mode not active, nothing to do.")
            return

        device = self.lcx_keys_0.device
        dtype = self.lcx_keys_0.dtype

        # ── 1. Gather all existing slots from all allocated levels ──
        all_keys, all_values, all_heat, all_valid = [], [], [], []
        old_levels = sorted(self._lcx_allocated_levels)

        for lvl in old_levels:
            k = getattr(self, f'lcx_keys_{lvl}', None)
            if k is None:
                continue
            all_keys.append(k.detach())
            all_values.append(getattr(self, f'lcx_values_{lvl}').detach())
            all_heat.append(getattr(self, f'lcx_heat_{lvl}').detach())
            all_valid.append(getattr(self, f'lcx_valid_{lvl}').detach())

        if all_keys:
            merged_keys = torch.cat(all_keys, dim=0)
            merged_values = torch.cat(all_values, dim=0)
            merged_heat = torch.cat(all_heat, dim=0)
            merged_valid = torch.cat(all_valid, dim=0)
            old_total = merged_keys.shape[0]
        else:
            old_total = 0

        # ── 2. If shrinking, keep hottest slots first ──
        if old_total > new_slots:
            _, sort_idx = merged_heat.sort(descending=True)
            merged_keys = merged_keys[sort_idx[:new_slots]]
            merged_values = merged_values[sort_idx[:new_slots]]
            merged_heat = merged_heat[sort_idx[:new_slots]]
            merged_valid = merged_valid[sort_idx[:new_slots]]
            kept = new_slots
        else:
            kept = old_total

        # ── 3. Create new buffers ──
        new_keys = torch.randn(new_slots, self.lcx_key_dim, device=device, dtype=dtype)
        new_keys = torch.nn.functional.normalize(new_keys, dim=-1)
        new_values = torch.zeros(new_slots, self.embedding_dim, device=device, dtype=dtype)
        new_heat = torch.zeros(new_slots, dtype=torch.int16, device=device)
        new_valid = torch.zeros(new_slots, dtype=torch.bool, device=device)

        # ── 4. Copy preserved slots into the front ──
        if kept > 0:
            new_keys[:kept] = merged_keys[:kept]
            new_values[:kept] = merged_values[:kept]
            new_heat[:kept] = merged_heat[:kept]
            new_valid[:kept] = merged_valid[:kept]

        # ── 5. Clean up old levels (L1+) ──
        for lvl in old_levels:
            if lvl == 0:
                continue
            # Remove registered buffers for this level
            for suffix in ['keys', 'values', 'heat', 'valid']:
                name = f'lcx_{suffix}_{lvl}'
                if name in self._buffers:
                    del self._buffers[name]
                elif hasattr(self, name):
                    delattr(self, name)
            # Remove hash/bucket state
            for prefix in ['_lcx_hash_planes_', '_lcx_hash_powers_',
                           '_lcx_num_buckets_', '_lcx_bucket_ids_',
                           '_lcx_bucket_index_', '_lcx_bucket_counts_',
                           '_lcx_max_bucket_size_', '_lcx_bucketed_']:
                name = f'{prefix}{lvl}'
                if name in self._buffers:
                    del self._buffers[name]
                elif hasattr(self, name):
                    delattr(self, name)

        # ── 6. Clean up old L0 hash/bucket state ──
        for prefix in ['_lcx_hash_planes_', '_lcx_hash_powers_',
                       '_lcx_num_buckets_', '_lcx_bucket_ids_',
                       '_lcx_bucket_index_', '_lcx_bucket_counts_',
                       '_lcx_max_bucket_size_', '_lcx_bucketed_']:
            name = f'{prefix}0'
            if name in self._buffers:
                del self._buffers[name]
            elif hasattr(self, name):
                delattr(self, name)

        # ── 7. Install new L0 buffers ──
        # register_buffer updates existing buffer if name already registered
        self.register_buffer('lcx_keys_0', new_keys)
        self.register_buffer('lcx_values_0', new_values)
        self.register_buffer('lcx_heat_0', new_heat)
        self.register_buffer('lcx_valid_0', new_valid)

        # ── 8. Update model config ──
        self._lcx_num_levels = 1
        self._lcx_level_slots = [new_slots]
        self._lcx_total_slots = new_slots
        self._lcx_allocated_levels = {0}
        self._lcx_route_temps = [1.0]

        # ── 9. Re-init hash planes + bucket index ──
        if new_slots >= self._LCX_MIN_SLOTS_FOR_BUCKETING:
            self._lcx_init_hash_planes(0, new_slots, device, dtype)
            self._lcx_build_bucket_index(0)

        # Free old tensors
        del all_keys, all_values, all_heat, all_valid
        if old_total > 0:
            del merged_keys, merged_values, merged_heat, merged_valid

        _mb = new_slots * (self.lcx_key_dim + self.embedding_dim) * 4 / 1024 / 1024
        print(f"  [LCX] Resized: {old_total:,} slots ({len(old_levels)} levels) "
              f"-> {new_slots:,} slots (1 level, {_mb:.0f} MB)")
        print(f"  [LCX] Preserved {kept:,}/{old_total:,} memories "
              f"({new_slots - kept:,} new empty slots)")

    def lcx_heat_stats(self):
        """Return per-level heat statistics for logging/telemetry."""
        stats = {}
        if not hasattr(self, '_lcx_allocated_levels'):
            return stats
        for lvl in range(self._lcx_num_levels):
            heat = getattr(self, f'lcx_heat_{lvl}', None)
            if heat is not None and lvl in self._lcx_allocated_levels:
                h = heat.float()
                stats[f'L{lvl}_hot_slots'] = int((h > 0).sum().item())
                stats[f'L{lvl}_total_slots'] = heat.shape[0]
                stats[f'L{lvl}_max_heat'] = int(h.max().item())
                stats[f'L{lvl}_mean_heat'] = float(h.mean().item())
            valid = getattr(self, f'lcx_valid_{lvl}', None)
            if valid is not None and lvl in self._lcx_allocated_levels:
                stats[f'L{lvl}_valid_slots'] = int(valid.sum().item())
        stats['allocated_levels'] = len(self._lcx_allocated_levels)
        return stats

    # --- LCX double-buffer (golden read / scratch write / sleep cycle) ---

    def snapshot_lcx(self) -> dict:
        """Clone all LCX buffers for allocated levels. Returns CPU copies."""
        snap = {}
        if not hasattr(self, '_lcx_allocated_levels'):
            return snap
        for lvl in self._lcx_allocated_levels:
            snap[f'L{lvl}_keys'] = getattr(self, f'lcx_keys_{lvl}').detach().cpu().clone()
            snap[f'L{lvl}_values'] = getattr(self, f'lcx_values_{lvl}').detach().cpu().clone()
            snap[f'L{lvl}_heat'] = getattr(self, f'lcx_heat_{lvl}').cpu().clone()
            snap[f'L{lvl}_valid'] = getattr(self, f'lcx_valid_{lvl}').cpu().clone()
        return snap

    def restore_lcx(self, snap: dict):
        """Restore LCX buffers from a snapshot (moves to model device)."""
        for lvl in self._lcx_allocated_levels:
            dev = getattr(self, f'lcx_keys_{lvl}').device
            getattr(self, f'lcx_keys_{lvl}').copy_(snap[f'L{lvl}_keys'].to(dev))
            getattr(self, f'lcx_values_{lvl}').copy_(snap[f'L{lvl}_values'].to(dev))
            getattr(self, f'lcx_heat_{lvl}').copy_(snap[f'L{lvl}_heat'].to(dev))
            getattr(self, f'lcx_valid_{lvl}').copy_(snap[f'L{lvl}_valid'].to(dev))

    def enable_double_buffer(self, golden_snap=None):
        """Enable double-buffer mode: reads from golden, writes to live scratch.
        If no golden_snap provided, snapshots current LCX as initial golden."""
        if golden_snap is None:
            golden_snap = self.snapshot_lcx()
        # Move golden to model device for fast reads
        dev = next(self.parameters()).device
        self._db_golden = {k: v.to(dev) for k, v in golden_snap.items()}
        self._db_mode = True
        self._db_reading = False

    def disable_double_buffer(self, keep_golden=False):
        """Disable double-buffer mode. If keep_golden, restore golden to live buffers."""
        if keep_golden and hasattr(self, '_db_golden'):
            self.restore_lcx(self._db_golden)
        self._db_mode = False
        self._db_reading = False
        if hasattr(self, '_db_golden'):
            del self._db_golden

    # --- LCX bottleneck projection ---

    def _lcx_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Translate LCX read from memory-space to brain-space: D→618→C19→618→C19→D."""
        if self.lcx_bn_layers is None:
            return x
        for i, layer in enumerate(self.lcx_bn_layers):
            x = layer(x)
            if i < len(self.lcx_bn_layers) - 1:
                x = c19_activation(x)
        return x

    # --- Byte token encode/decode ---

    def _byte_encode(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Byte token (0-255) → D-dimensional vector via C19 bottleneck.
        Input: [B] Long tensor of byte values.
        Output: [B, D] float tensor."""
        emb = self.byte_embed(byte_ids)                 # [B, bn]
        h = c19_activation(self.byte_enc1(emb))         # [B, bn]
        return self.byte_enc2(h)                        # [B, D]

    def _byte_decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """D-dimensional hidden → 256 byte logits via C19 bottleneck.
        Input: [B, D] float tensor.
        Output: [B, 256] logits (raw, no softmax)."""
        h = c19_activation(self.byte_dec1(hidden))      # [B, bn]
        h = c19_activation(self.byte_dec2(h))           # [B, bn]
        return self.byte_dec3(h)                        # [B, 256]

    # --- Flat LCX read/write ---

    def _lcx_flat_read(self, state: torch.Tensor, level: int) -> torch.Tensor:
        """Bucketed cosine similarity read at a single level.
        SimHash query -> bucket -> search within bucket -> topk -> softmax weighted sum.
        For small levels (<1024 slots), falls back to full search.
        In double-buffer mode, reads from golden copy via _db_reading flag."""
        _db = getattr(self, '_db_mode', False)
        if _db:
            self._db_reading = True
        try:
            return self._lcx_flat_read_impl(state, level)
        finally:
            if _db:
                self._db_reading = False

    def _lcx_flat_read_impl(self, state: torch.Tensor, level: int) -> torch.Tensor:
        """Inner implementation of flat LCX read."""
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        B = state.shape[0]
        k = self.lcx_top_k

        query = self.lcx_route_query(state)                              # [B, key_dim]
        query = torch.nn.functional.normalize(query, dim=-1)

        keys, values = self._lcx_level_bufs(level)
        valid = getattr(self, f'lcx_valid_{level}', None)
        temp = self._lcx_route_temps[min(level, len(self._lcx_route_temps) - 1)]

        is_bucketed = getattr(self, f'_lcx_bucketed_{level}', False)

        if not is_bucketed:
            # --- Original full search (small levels) ---
            scores = query @ keys.detach().clone().T                     # [B, S]
            if valid is not None and valid.any() and not valid.all():
                scores = scores.masked_fill(~valid.unsqueeze(0), float('-inf'))
            if temp != 1.0:
                scores = scores / temp
            eff_k = min(k, scores.shape[-1])
            topk_scores, topk_idx = scores.topk(eff_k, dim=-1)          # [B, K]
            weights = torch.nn.functional.softmax(topk_scores, dim=-1)   # [B, K]
            topk_values = values.detach()[topk_idx]                      # [B, K, D]
        else:
            # --- Bucketed search (large levels) ---
            bucket_index = getattr(self, f'_lcx_bucket_index_{level}')   # [num_buckets, max_bkt]
            bucket_counts = getattr(self, f'_lcx_bucket_counts_{level}') # [num_buckets]
            max_bkt = getattr(self, f'_lcx_max_bucket_size_{level}')

            query_bucket = self._lcx_compute_bucket(query, level)        # [B]
            # Gather the slot indices for each query's bucket
            sel_indices = bucket_index[query_bucket]                     # [B, max_bkt]
            sel_counts = bucket_counts[query_bucket]                     # [B]

            # Gather keys for the selected slots only
            flat_idx = sel_indices.clamp(min=0)                          # safe index for empty padding
            sel_keys = keys.detach()[flat_idx]                           # [B, max_bkt, key_dim]

            # Batched cosine similarity within bucket
            scores = torch.bmm(query.unsqueeze(1),
                               sel_keys.transpose(1, 2)).squeeze(1)     # [B, max_bkt]

            # Mask padding positions
            range_idx = torch.arange(max_bkt, device=scores.device).unsqueeze(0)  # [1, max_bkt]
            pad_mask = range_idx >= sel_counts.unsqueeze(1)              # [B, max_bkt]
            scores = scores.masked_fill(pad_mask, float('-inf'))

            # Mask invalid (unwritten) slots
            if valid is not None and valid.any() and not valid.all():
                sel_valid = valid[flat_idx]                              # [B, max_bkt]
                scores = scores.masked_fill(~sel_valid, float('-inf'))

            if temp != 1.0:
                scores = scores / temp

            eff_k = min(k, max_bkt)
            topk_scores, topk_local = scores.topk(eff_k, dim=-1)        # [B, K]
            # Map local bucket indices back to global slot indices
            topk_idx = flat_idx.gather(1, topk_local)                    # [B, K]
            weights = torch.nn.functional.softmax(topk_scores, dim=-1)   # [B, K]
            topk_values = values.detach()[topk_idx]                      # [B, K, D]

        context = (weights.unsqueeze(-1) * topk_values).sum(dim=1)       # [B, D]

        # Stash read weights for auxiliary entropy loss
        if self.training:
            if not hasattr(self, '_lcx_read_weights_accum'):
                self._lcx_read_weights_accum = []
            self._lcx_read_weights_accum.append(weights)
            # Score margin diagnostic (pure telemetry, no gradient)
            with torch.no_grad():
                _n_cand = scores.shape[-1]
                if _n_cand > eff_k:
                    _kp1 = scores.topk(eff_k + 1, dim=-1).values  # [B, K+1]
                    _margin = _kp1[:, -2] - _kp1[:, -1]           # last winner - first loser
                else:
                    _margin = torch.zeros(1, device=scores.device)
                if not hasattr(self, '_lcx_score_margin_accum'):
                    self._lcx_score_margin_accum = []
                self._lcx_score_margin_accum.append(
                    (_margin.mean().item(), topk_scores[:, 0].mean().item())
                )

        if squeeze:
            context = context.squeeze(0)
        return context

    def _lcx_flat_write(self, state: torch.Tensor,
                        write_content: torch.Tensor, level: int):
        """Flat cosine route → EMA write at a single level.
        Anti-rut: random slot write (no quadtree siblings)."""
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            write_content = write_content.unsqueeze(0)
        B = state.shape[0]
        k = self.lcx_top_k

        # Write gate WITH gradient for auxiliary loss
        # L2-normalize input to bound pre-activation (hidden magnitudes grow unbounded), fp32
        with torch.amp.autocast('cuda', enabled=False):
            _wg_dtype = self.lcx_write_gate.weight.dtype  # fp32 on GPU, fp64 if model.double()
            _wc_norm = torch.nn.functional.normalize(write_content.to(_wg_dtype), dim=-1)
            gate_for_aux = torch.sigmoid(self.lcx_write_gate(_wc_norm))  # [B, 1]
        if self.training:
            if not hasattr(self, '_lcx_write_gate_accum'):
                self._lcx_write_gate_accum = []
            self._lcx_write_gate_accum.append(gate_for_aux)

        with torch.no_grad():
            query = self.lcx_route_query(state)
            query = torch.nn.functional.normalize(query, dim=-1)

            keys, values = self._lcx_level_bufs(level)
            scores = query @ keys.T                                      # [B, S]
            eff_k = min(k, scores.shape[-1])
            topk_scores, topk_idx = scores.topk(eff_k, dim=-1)          # [B, K]

            # EMA write at this level
            weights = torch.nn.functional.softmax(topk_scores, dim=-1)
            gate = gate_for_aux.detach()  # detached for buffer write (no grad through EMA)
            # Asymmetric nudge: slower at deeper levels to preserve diversity
            nudge_rate = 0.005 if level > 0 else 0.01

            # Heat counter + valid mask for hot-bin tracking
            heat = getattr(self, f'lcx_heat_{level}', None)
            valid = getattr(self, f'lcx_valid_{level}', None)

            for b in range(B):
                for j in range(topk_idx.shape[1]):
                    idx = topk_idx[b, j].item()
                    w = weights[b, j] * gate[b, 0]
                    values[idx] = (1.0 - w) * values[idx] + w * write_content[b]
                    keys[idx] = torch.nn.functional.normalize(
                        (1.0 - nudge_rate) * keys[idx] + nudge_rate * query[b],
                        dim=-1)
                    if heat is not None:
                        heat[idx] = min(heat[idx] + 1, 32767)  # int16 cap
                    if valid is not None:
                        valid[idx] = True

            # Anti-rut: write weakly to a random slot (flat — no siblings)
            num_slots = keys.shape[0]
            rand_idx = torch.randint(num_slots, (B,), device=keys.device)
            for b in range(B):
                ri = rand_idx[b].item()
                values[ri] = 0.99 * values[ri] + 0.01 * write_content[b]
                if valid is not None:
                    valid[ri] = True

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
        # Detach memory from previous step's computation graph
        if self.lcx is not None:
            self.lcx = self.lcx.detach()
        elif self.gem is not None:
            self.gem = self.gem.detach()

        # Dispatch to vectorized path for combinatorial mode
        if self.combinatorial and hasattr(self, '_vec_vis_indices'):
            return self._forward_vectorized(x, return_stats, return_being_outputs)

        if self.byte_token_mode:
            B, T = x.shape  # x is [B, T] Long tensor of byte values
            _float_dtype = next(self.parameters()).dtype  # match model dtype (fp32 or fp64)
        else:
            B, T, _ = x.shape  # x is [B, T, num_bits] float tensor
            _float_dtype = x.dtype  # inherit float32/float64 from input

        # Clear LCX auxiliary loss accumulators for this forward pass
        if self.training and self._lcx_hash_mode:
            self._lcx_write_gate_accum = []
            self._lcx_read_weights_accum = []
            self._lcx_zoom_gate_accum = []
            self._lcx_score_margin_accum = []

        # Rebuild bucket indices for bucketed LCX search (once per forward pass)
        if self._lcx_hash_mode:
            self._lcx_rebuild_all_bucket_indices()

        # SHARED ring memory (all beings write here)
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=_float_dtype
        )

        # PER-BEING state initialization
        being_states = []
        for being_idx in range(self.num_beings):
            # Spatial offset initialization: spread beings across ring
            offset_start = being_idx * self.num_memory_positions / self.num_beings
            offset_end = (being_idx + 1) * self.num_memory_positions / self.num_beings
            seg_len = offset_end - offset_start
            # Pointer A: first third (or half/full depending on num_pointers)
            if self.num_pointers >= 3:
                third = seg_len / 3
                pointer_init_a = torch.empty(B, device=x.device).uniform_(offset_start, offset_start + third)
            elif self.num_pointers >= 2:
                mid = offset_start + seg_len / 2
                pointer_init_a = torch.empty(B, device=x.device).uniform_(offset_start, mid)
            else:
                pointer_init_a = torch.empty(B, device=x.device).uniform_(offset_start, offset_end)

            state_dict = {
                'pointer_position': pointer_init_a,
                'hidden_state': torch.zeros(B, self.hidden_dims[being_idx], device=x.device, dtype=_float_dtype),
            }
            # Pointer B: second segment
            if self.num_pointers >= 2:
                if self.num_pointers >= 3:
                    state_dict['pointer_position_b'] = torch.empty(B, device=x.device).uniform_(
                        offset_start + third, offset_start + 2 * third)
                else:
                    state_dict['pointer_position_b'] = torch.empty(B, device=x.device).uniform_(mid, offset_end)
            # Pointer C: last third
            if self.num_pointers >= 3:
                state_dict['pointer_position_c'] = torch.empty(B, device=x.device).uniform_(
                    offset_start + 2 * third, offset_end)

            being_states.append(state_dict)

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

        # ─── TWO-PHASE FORWARD: Scanner Pre-Sweep ───
        _scanner_idxs = [i for i, m in enumerate(self.being_modes) if m == 'scanner']
        _thinker_idxs = [i for i, m in enumerate(self.being_modes) if m == 'thinker']
        _has_phase_split = len(_scanner_idxs) > 0 and len(_thinker_idxs) > 0
        _active_beings = _thinker_idxs if _has_phase_split else list(range(self.num_beings))

        if _has_phase_split:
            assert self._lcx_hash_mode, \
                "Two-phase forward (scanner+thinker) requires hash LCX mode"
            with torch.no_grad():
                for _ts in range(T):
                    # Input for this timestep
                    if self.byte_token_mode:
                        _s_bytes = x[:, _ts]
                    else:
                        _s_bits = x[:, _ts, :]

                    for being_idx in _scanner_idxs:
                        being = self.beings[being_idx]
                        state = being_states[being_idx]

                        if self.being_states.get(being_idx, 'null') == 'null':
                            continue
                        if self.tick_periods is not None and _ts % self.tick_periods[being_idx].item() != 0:
                            continue

                        # Ring read (Pointer A)
                        indices_a, weights_a = self._gaussian_attention_weights(
                            state['pointer_position'], self.num_memory_positions)
                        indices_exp_a = indices_a.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                        neighborhood_a = memory_ring.gather(1, indices_exp_a)
                        context_read_a = (weights_a.unsqueeze(-1) * neighborhood_a).sum(dim=1)

                        if self.num_pointers >= 2:
                            indices_b, weights_b = self._gaussian_attention_weights(
                                state['pointer_position_b'], self.num_memory_positions)
                            indices_exp_b = indices_b.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                            neighborhood_b = memory_ring.gather(1, indices_exp_b)
                            context_read_b = (weights_b.unsqueeze(-1) * neighborhood_b).sum(dim=1)

                        if self.num_pointers >= 3:
                            indices_c, weights_c = self._gaussian_attention_weights(
                                state['pointer_position_c'], self.num_memory_positions)
                            indices_exp_c = indices_c.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                            neighborhood_c = memory_ring.gather(1, indices_exp_c)
                            context_read_c = (weights_c.unsqueeze(-1) * neighborhood_c).sum(dim=1)

                        # Combine pointer reads
                        if self.num_pointers >= 3:
                            _D = self.embedding_dim
                            reads_stack = torch.stack([context_read_a, context_read_b, context_read_c], dim=1)
                            attn_scores = (state['hidden_state'].unsqueeze(1) * reads_stack).sum(dim=-1) / (_D ** 0.5)
                            attn_weights = 0.25 * torch.softmax(attn_scores, dim=-1) + 0.25
                            context_read = (attn_weights.unsqueeze(-1) * reads_stack).sum(dim=1)
                        elif self.num_pointers >= 2:
                            context_read = 0.5 * (context_read_a + context_read_b)
                        else:
                            context_read = context_read_a

                        context_scale = torch.sigmoid(being.context_strength)
                        if self.ring_read_projs is not None:
                            context_read = self.ring_read_projs[being_idx](context_read)

                        # Input combination (hash LCX mode)
                        if self.byte_token_mode:
                            being_input_vec = self._byte_encode(_s_bytes)
                        else:
                            being_input_vec = self.input_proj(_s_bits)
                        lcx_ctx = self._lcx_flat_read(being_input_vec, level=0)
                        being_input_vec = being_input_vec + self._lcx_bottleneck(lcx_ctx)
                        if self.being_input_projs is not None:
                            being_input_vec = self.being_input_projs[being_idx](being_input_vec)
                        combined_input = being_input_vec + context_scale * context_read

                        # Phase embedding
                        h_i = self.hidden_dims[being_idx]
                        _B = combined_input.size(0)
                        if being.phase_embedding.size(0) == h_i:
                            phase_bias = being.phase_embedding.unsqueeze(0).expand(_B, -1)
                        else:
                            phase_16d = being.phase_embedding.unsqueeze(0).expand(_B, -1)
                            if h_i >= 16:
                                padding = torch.zeros(_B, h_i - 16, device=phase_16d.device, dtype=phase_16d.dtype)
                                phase_bias = torch.cat([phase_16d, padding], dim=1)
                            else:
                                phase_bias = phase_16d[:, :h_i]
                        combined_input = combined_input + 0.1 * phase_bias

                        # State update (phi EMA)
                        _norm = self.being_state_norms[being_idx] if self.being_state_norms is not None else self.state_norm
                        state_update = STATE_EMA_ALPHA * _norm(state['hidden_state']) + (1 - STATE_EMA_ALPHA) * combined_input
                        if self.being_processing_layers is not None:
                            for norm, layer in zip(self.being_processing_norms[being_idx], self.being_processing_layers[being_idx]):
                                state_update = state_update + c19_activation(layer(norm(state_update)))
                        elif self.processing_layers is not None:
                            for norm, layer in zip(self.processing_norms, self.processing_layers):
                                state_update = state_update + c19_activation(layer(norm(state_update)))
                        state['hidden_state'] = state_update

                        # LCX write
                        if self.ring_write_projs is not None:
                            write_vec = self.ring_write_projs[being_idx](state_update)
                        else:
                            write_vec = state_update
                        self._lcx_flat_write(state['hidden_state'], write_vec, level=0)

                        # Ring write (Pointer A)
                        update_broadcast_a = write_vec.unsqueeze(1).expand(-1, weights_a.size(1), -1)
                        contribution_a = weights_a.unsqueeze(-1) * update_broadcast_a
                        memory_ring = memory_ring.scatter_add(1, indices_exp_a, contribution_a)

                        if self.num_pointers >= 2:
                            update_broadcast_b = write_vec.unsqueeze(1).expand(-1, weights_b.size(1), -1)
                            contribution_b = weights_b.unsqueeze(-1) * update_broadcast_b
                            memory_ring = memory_ring.scatter_add(1, indices_exp_b, contribution_b)

                        if self.num_pointers >= 3:
                            update_broadcast_c = write_vec.unsqueeze(1).expand(-1, weights_c.size(1), -1)
                            contribution_c = weights_c.unsqueeze(-1) * update_broadcast_c
                            memory_ring = memory_ring.scatter_add(1, indices_exp_c, contribution_c)

                        # Pointer update (A)
                        current_pos_a = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                        jump_target_a = being.pointer_destinations[current_pos_a].float().clamp(0, self.num_memory_positions - 1)
                        jump_logits_a = being.jump_gate(state_update).squeeze(-1)
                        jump_prob_a = torch.sigmoid(jump_logits_a / JUMP_SIGMOID_TAU)
                        walk_position_a = (state['pointer_position'] + 1.0) % self.num_memory_positions
                        state['pointer_position'] = jump_prob_a * jump_target_a + (1.0 - jump_prob_a) * walk_position_a

                        if self.num_pointers >= 2:
                            current_pos_b = state['pointer_position_b'].long().clamp(0, self.num_memory_positions - 1)
                            jump_target_b = being.pointer_destinations_b[current_pos_b].float().clamp(0, self.num_memory_positions - 1)
                            jump_logits_b = being.jump_gate_b(state_update).squeeze(-1)
                            jump_prob_b = torch.sigmoid(jump_logits_b / JUMP_SIGMOID_TAU)
                            walk_position_b = (state['pointer_position_b'] + 1.0) % self.num_memory_positions
                            state['pointer_position_b'] = jump_prob_b * jump_target_b + (1.0 - jump_prob_b) * walk_position_b

                        if self.num_pointers >= 3:
                            current_pos_c = state['pointer_position_c'].long().clamp(0, self.num_memory_positions - 1)
                            jump_target_c = being.pointer_destinations_c[current_pos_c].float().clamp(0, self.num_memory_positions - 1)
                            jump_logits_c = being.jump_gate_c(state_update).squeeze(-1)
                            jump_prob_c = torch.sigmoid(jump_logits_c / JUMP_SIGMOID_TAU)
                            walk_position_c = (state['pointer_position_c'] + 1.0) % self.num_memory_positions
                            state['pointer_position_c'] = jump_prob_c * jump_target_c + (1.0 - jump_prob_c) * walk_position_c
            # ─── END Scanner Pre-Sweep ───

        # TIMESTEP LOOP (thinker beings only when phase split active)
        for t in range(T):
            # 1. Shared input projection (all beings see same input)
            if self.byte_token_mode:
                input_bytes = x[:, t]       # [B] Long tensor of byte values
                input_bits = None           # not used in byte token mode
            else:
                input_bits = x[:, t, :]     # [B, num_bits] float tensor
            # Memory: Hash LCX (embedding space) or Dense LCX (input space) or GEM
            if self.byte_token_mode:
                # Byte token mode: encode bytes directly, skip raw-bits path
                input_with_mem = None
                input_vec = self._byte_encode(input_bytes)  # [B, D]
            elif self._lcx_hash_mode:
                # Hash LCX: raw bits only (LCX context added per-being in step 2b)
                input_with_mem = input_bits  # [B, num_bits]
            elif self.lcx is not None:
                # Dense LCX: full input broadcast+add (no mask for shared path)
                input_with_mem = self._read_with_lcx(input_bits)  # [B, lcx_size=64]
            elif self.gem is not None:
                gem_expanded = self.gem.unsqueeze(0).expand(input_bits.size(0), -1)  # [B, num_bits]
                input_with_mem = torch.cat([input_bits, gem_expanded], dim=-1)  # [B, num_bits*2]
            else:
                # No memory active (INFANT mode: hash-mode model with LCX disabled)
                input_with_mem = input_bits  # [B, num_bits]
            if not self.byte_token_mode:
                if self.input_proj is not None:
                    input_vec = self.input_proj(input_with_mem)
                else:
                    input_vec = input_with_mem

            # 2. Each being processes independently
            being_outputs_t = []
            hidden_states_t = []
            pointer_positions_t = []

            for being_idx in _active_beings:
                being = self.beings[being_idx]
                state = being_states[being_idx]

                # NULL BEINGS: skip entirely (output zeros, no influence)
                if self.being_states.get(being_idx, 'null') == 'null':
                    output_bits = torch.zeros(x.size(0), self._output_dim, device=x.device, dtype=_float_dtype)
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
                    if self.full_view and self.being_output_projs is not None:
                        output_bits = self.being_output_projs[being_idx](state['hidden_state'])
                    elif self.heterogeneous:
                        vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                        output_k = self.being_output_projs[being_idx](state['hidden_state'])
                        output_bits = torch.zeros(state['hidden_state'].size(0), self.num_bits,
                                                  device=x.device, dtype=_float_dtype)
                        output_bits[:, vis_idx] = output_k
                    elif self.byte_token_mode:
                        output_bits = self._byte_decode(state['hidden_state'])
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

                # Scanner no_grad: disable gradient for scanner beings
                _is_scanner = self.being_modes[being_idx] == 'scanner'
                _prev_grad = torch.is_grad_enabled()
                if _is_scanner:
                    torch.set_grad_enabled(False)

                # 2a. Read from being's pointer position(s)
                indices_a, weights_a = self._gaussian_attention_weights(
                    state['pointer_position'], self.num_memory_positions
                )
                indices_exp_a = indices_a.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                neighborhood_a = memory_ring.gather(1, indices_exp_a)
                context_read_a = (weights_a.unsqueeze(-1) * neighborhood_a).sum(dim=1)

                if self.num_pointers >= 2:
                    # Pointer B: independent read from second position
                    indices_b, weights_b = self._gaussian_attention_weights(
                        state['pointer_position_b'], self.num_memory_positions
                    )
                    indices_exp_b = indices_b.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                    neighborhood_b = memory_ring.gather(1, indices_exp_b)
                    context_read_b = (weights_b.unsqueeze(-1) * neighborhood_b).sum(dim=1)

                if self.num_pointers >= 3:
                    # Pointer C: independent read from third position
                    indices_c, weights_c = self._gaussian_attention_weights(
                        state['pointer_position_c'], self.num_memory_positions
                    )
                    indices_exp_c = indices_c.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                    neighborhood_c = memory_ring.gather(1, indices_exp_c)
                    context_read_c = (weights_c.unsqueeze(-1) * neighborhood_c).sum(dim=1)

                if self.num_pointers >= 3:
                    # Competitive attention: score each read against hidden state, softmax-select
                    _D = self.embedding_dim
                    reads_stack = torch.stack([context_read_a, context_read_b, context_read_c], dim=1)  # [B, 3, D]
                    attn_scores = (state['hidden_state'].unsqueeze(1) * reads_stack).sum(dim=-1)  # [B, 3]
                    attn_scores = attn_scores / (_D ** 0.5)
                    attn_weights = 0.25 * torch.softmax(attn_scores, dim=-1) + 0.25  # [B, 3] min 25% per pointer
                    context_read = (attn_weights.unsqueeze(-1) * reads_stack).sum(dim=1)  # [B, D]
                elif self.num_pointers >= 2:
                    context_read = 0.5 * (context_read_a + context_read_b)
                else:
                    context_read = context_read_a

                context_scale = torch.sigmoid(being.context_strength)

                # CAPACITY FIBONACCI: project ring context D -> H_i
                if self.ring_read_projs is not None:
                    context_read = self.ring_read_projs[being_idx](context_read)

                # 2b. Combine input + context (with optional receptive field masking)
                if self._lcx_hash_mode:
                    # Hash LCX: project input to embedding space, add LCX context
                    if self.byte_token_mode:
                        being_input_vec = self._byte_encode(input_bytes)  # [B, D]
                    else:
                        being_input_vec = self.input_proj(input_bits)  # [B, D]
                    lcx_ctx = self._lcx_flat_read(being_input_vec, level=0)  # [B, D]
                    being_input_vec = being_input_vec + self._lcx_bottleneck(lcx_ctx)
                elif self.lcx is not None:
                    # Dense LCX: broadcast+add with per-ant mask → shared projection
                    if self.heterogeneous and not self.full_view:
                        vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                        vis_bits = input_bits[:, vis_idx]
                        combined_lcx = self._read_with_lcx(vis_bits, mask=vis_idx)  # [B, 64]
                    elif being.input_mask is not None:
                        masked_bits = input_bits * being.input_mask.unsqueeze(0)
                        combined_lcx = self._read_with_lcx(masked_bits)  # [B, 64]
                    else:
                        combined_lcx = input_with_mem  # already computed above [B, 64]
                    being_input_vec = self.input_proj(combined_lcx) if self.input_proj is not None else combined_lcx
                    # Bridge D → H_i for capacity_fibonacci compatibility
                    if self.being_input_projs is not None:
                        being_input_vec = self.being_input_projs[being_idx](being_input_vec)
                elif self.full_view and self.being_input_projs is not None:
                    being_input_vec = self.being_input_projs[being_idx](input_with_mem)
                elif self.heterogeneous:
                    vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                    visible_bits = input_bits[:, vis_idx]
                    visible_with_mem = torch.cat([visible_bits, gem_expanded], dim=-1)
                    being_input_vec = self.being_input_projs[being_idx](visible_with_mem)
                elif being.input_mask is not None:
                    masked_bits = input_bits * being.input_mask.unsqueeze(0)
                    masked_with_mem = torch.cat([masked_bits, gem_expanded], dim=-1)
                    if self.input_proj is not None:
                        being_input_vec = self.input_proj(masked_with_mem)
                    else:
                        being_input_vec = masked_with_mem
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

                # 2c. State update (phi EMA — norm hidden only, input path clean)
                _norm = self.being_state_norms[being_idx] if self.being_state_norms is not None else self.state_norm
                state_update = STATE_EMA_ALPHA * _norm(state['hidden_state']) + (1 - STATE_EMA_ALPHA) * combined_input

                # Additional processing layers (Pre-LN + C19 residual)
                if self.being_processing_layers is not None:
                    for norm, layer in zip(self.being_processing_norms[being_idx], self.being_processing_layers[being_idx]):
                        state_update = state_update + c19_activation(layer(norm(state_update)))
                elif self.processing_layers is not None:
                    for norm, layer in zip(self.processing_norms, self.processing_layers):
                        state_update = state_update + c19_activation(layer(norm(state_update)))

                state['hidden_state'] = state_update

                # CAPACITY FIBONACCI: project H_i -> D for memory writes
                if self.ring_write_projs is not None:
                    write_vec = self.ring_write_projs[being_idx](state_update)
                else:
                    write_vec = state_update

                # MEMORY WRITE (sequential): per-slot ownership (VRA-88)
                if self._lcx_hash_mode:
                    # Hash LCX write: route state to top-k slots
                    self._lcx_flat_write(state['hidden_state'], write_vec, level=0)
                elif self.lcx is not None and self.pixel_assignments is not None:
                    slots = self.pixel_assignments[being_idx]  # [K] this being's pixel(s)
                    delta = torch.nn.functional.softsign(self.lcx_propose(write_vec))  # [B, lcx_size]
                    gate = torch.sigmoid(self.lcx_gate(write_vec))                     # [B, lcx_size]
                    delta_at_slots = delta[:, slots].mean(dim=0)    # [K] batch-mean per slot
                    gate_at_slots = gate[:, slots].mean(dim=0)      # [K]
                    w = EFFORT_WEIGHTS[self.effort_level].to(self.lcx.device)  # [3]
                    new_lcx = self.lcx.detach().clone()
                    for c in range(3):
                        old_c = self.lcx[c].detach()
                        old_at_slots = old_c[slots]
                        new_at_slots = (1.0 - gate_at_slots * w[c]) * old_at_slots + gate_at_slots * w[c] * delta_at_slots
                        new_lcx[c] = old_c.scatter(0, slots, new_at_slots)
                    self.lcx = new_lcx.clamp(-1.0, 1.0)
                elif self.lcx is not None:
                    # Legacy global write (backwards compat when pixel_assignments missing)
                    delta = torch.nn.functional.softsign(self.lcx_propose(write_vec))  # [B, lcx_size]
                    gate = torch.sigmoid(self.lcx_gate(write_vec))                      # [B, lcx_size]
                    delta_avg = delta.mean(dim=0)   # [lcx_size]
                    gate_avg = gate.mean(dim=0)     # [lcx_size]
                    w = EFFORT_WEIGHTS[self.effort_level].to(self.lcx.device)  # [3]
                    new_lcx = self.lcx.detach().clone()
                    for c in range(3):
                        old_c = self.lcx[c].detach()
                        new_lcx[c] = (1.0 - gate_avg * w[c]) * old_c + gate_avg * w[c] * delta_avg
                    self.lcx = new_lcx.clamp(-1.0, 1.0)
                elif self.gem_write_head is not None and self.gem is not None:
                    _gem_signal = torch.nn.functional.softsign(self.gem_write_head(write_vec))  # [B, num_bits]
                    _gem_update = _gem_signal.mean(dim=0)  # average across batch → [num_bits]
                    self.gem = self._phi_inv * self.gem.detach() + (1.0 - self._phi_inv) * _gem_update
                # else: no memory write (INFANT mode: hash-mode model with LCX disabled)

                # 2d. Write to SHARED ring (scatter_add accumulation)
                # Gradient flows through ring writes for collaborative learning.
                # Pointer A write
                update_broadcast_a = write_vec.unsqueeze(1).expand(-1, weights_a.size(1), -1)
                contribution_a = weights_a.unsqueeze(-1) * update_broadcast_a
                memory_ring = memory_ring.scatter_add(1, indices_exp_a, contribution_a)

                if self.num_pointers >= 2:
                    # Pointer B write (same write_vec, different neighborhood)
                    update_broadcast_b = write_vec.unsqueeze(1).expand(-1, weights_b.size(1), -1)
                    contribution_b = weights_b.unsqueeze(-1) * update_broadcast_b
                    memory_ring = memory_ring.scatter_add(1, indices_exp_b, contribution_b)

                if self.num_pointers >= 3:
                    # Pointer C write (same write_vec, different neighborhood)
                    update_broadcast_c = write_vec.unsqueeze(1).expand(-1, weights_c.size(1), -1)
                    contribution_c = weights_c.unsqueeze(-1) * update_broadcast_c
                    memory_ring = memory_ring.scatter_add(1, indices_exp_c, contribution_c)

                # 2e. Update pointer A (jump or walk) — soft blend for gradient flow
                current_pos_a = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                jump_target_a = being.pointer_destinations[current_pos_a].float().clamp(0, self.num_memory_positions - 1)
                jump_logits_a = being.jump_gate(state_update).squeeze(-1)
                jump_prob_a = torch.sigmoid(jump_logits_a / JUMP_SIGMOID_TAU)
                walk_position_a = (state['pointer_position'] + 1.0) % self.num_memory_positions
                state['pointer_position'] = jump_prob_a * jump_target_a + (1.0 - jump_prob_a) * walk_position_a

                if self.num_pointers >= 2:
                    # Update pointer B (independent jump/walk with walker bias)
                    current_pos_b = state['pointer_position_b'].long().clamp(0, self.num_memory_positions - 1)
                    jump_target_b = being.pointer_destinations_b[current_pos_b].float().clamp(0, self.num_memory_positions - 1)
                    jump_logits_b = being.jump_gate_b(state_update).squeeze(-1)
                    jump_prob_b = torch.sigmoid(jump_logits_b / JUMP_SIGMOID_TAU)
                    walk_position_b = (state['pointer_position_b'] + 1.0) % self.num_memory_positions
                    state['pointer_position_b'] = jump_prob_b * jump_target_b + (1.0 - jump_prob_b) * walk_position_b

                if self.num_pointers >= 3:
                    # Update pointer C (neutral bias)
                    current_pos_c = state['pointer_position_c'].long().clamp(0, self.num_memory_positions - 1)
                    jump_target_c = being.pointer_destinations_c[current_pos_c].float().clamp(0, self.num_memory_positions - 1)
                    jump_logits_c = being.jump_gate_c(state_update).squeeze(-1)
                    jump_prob_c = torch.sigmoid(jump_logits_c / JUMP_SIGMOID_TAU)
                    walk_position_c = (state['pointer_position_c'] + 1.0) % self.num_memory_positions
                    state['pointer_position_c'] = jump_prob_c * jump_target_c + (1.0 - jump_prob_c) * walk_position_c

                # Track jump activation (if stats requested)
                if return_stats:
                    if self.num_pointers >= 3:
                        avg_jump = (jump_prob_a.mean().item() + jump_prob_b.mean().item() + jump_prob_c.mean().item()) / 3.0
                        jump_counts_per_being[being_idx].append(avg_jump)
                        pointer_positions_t.append(state['pointer_position'].mean().item())
                        pointer_positions_t.append(state['pointer_position_b'].mean().item())
                        pointer_positions_t.append(state['pointer_position_c'].mean().item())
                    elif self.num_pointers >= 2:
                        avg_jump = 0.5 * (jump_prob_a.mean().item() + jump_prob_b.mean().item())
                        jump_counts_per_being[being_idx].append(avg_jump)
                        pointer_positions_t.append(state['pointer_position'].mean().item())
                        pointer_positions_t.append(state['pointer_position_b'].mean().item())
                    else:
                        jump_counts_per_being[being_idx].append(jump_prob_a.mean().item())
                        pointer_positions_t.append(state['pointer_position'].mean().item())

                # 2f. Generate output
                if self.full_view and self.being_output_projs is not None:
                    # FULL VIEW with per-being output: project H_i → all bits
                    output_bits = self.being_output_projs[being_idx](state['hidden_state'])  # [B, H_i] → [B, num_bits]
                elif self.heterogeneous:
                    # Per-being output: D → K_i visible bits, placed into full vector
                    vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                    output_k = self.being_output_projs[being_idx](state['hidden_state'])  # [B, K_i]
                    output_bits = torch.zeros(
                        state['hidden_state'].size(0), self.num_bits,
                        device=x.device, dtype=_float_dtype,
                    )
                    output_bits[:, vis_idx] = output_k
                elif self.byte_token_mode:
                    output_bits = self._byte_decode(state['hidden_state'])
                elif self.output_proj is not None:
                    output_bits = self.output_proj(state['hidden_state'])
                else:
                    output_bits = state['hidden_state']

                being_outputs_t.append(output_bits)
                outputs_per_being[being_idx].append(output_bits)
                hidden_states_t.append(state['hidden_state'].clone())

                # Restore gradient state after scanner processing
                if _is_scanner:
                    torch.set_grad_enabled(_prev_grad)

            # 2g. THINK TICKS: extra ring rounds without input injection
            #     Beings read what others wrote, process, write back. No new input.
            _eff_think_ticks = 0 if (not self.training and getattr(self, '_eval_skip_think', False)) else self.think_ticks
            if _eff_think_ticks > 0:
                for _tt in range(_eff_think_ticks):
                    for being_idx in range(self.num_beings):
                        being = self.beings[being_idx]
                        state = being_states[being_idx]

                        # Per-being think tick quota: skip if exhausted
                        if _tt >= self.think_ticks_per_being[being_idx]:
                            continue

                        # NULL BEINGS: skip in think ticks
                        if self.being_states.get(being_idx, 'null') == 'null':
                            continue

                        # TEMPORAL FIBONACCI: skip dormant beings in think ticks
                        if self.tick_periods is not None:
                            global_tick = t * (1 + self.think_ticks_per_being[being_idx]) + 1 + _tt
                            if global_tick % self.tick_periods[being_idx].item() != 0:
                                continue

                        # Scanner no_grad: disable gradient for scanner beings in think ticks
                        _is_scanner_tt = self.being_modes[being_idx] == 'scanner'
                        _prev_grad_tt = torch.is_grad_enabled()
                        if _is_scanner_tt:
                            torch.set_grad_enabled(False)

                        # Read from ring (Pointer A)
                        indices_a, weights_a = self._gaussian_attention_weights(
                            state['pointer_position'], self.num_memory_positions
                        )
                        indices_exp_a = indices_a.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                        neighborhood_a = memory_ring.gather(1, indices_exp_a)
                        context_read_a = (weights_a.unsqueeze(-1) * neighborhood_a).sum(dim=1)

                        if self.num_pointers >= 2:
                            # Pointer B read
                            indices_b, weights_b = self._gaussian_attention_weights(
                                state['pointer_position_b'], self.num_memory_positions
                            )
                            indices_exp_b = indices_b.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                            neighborhood_b = memory_ring.gather(1, indices_exp_b)
                            context_read_b = (weights_b.unsqueeze(-1) * neighborhood_b).sum(dim=1)

                        if self.num_pointers >= 3:
                            # Pointer C read
                            indices_c, weights_c = self._gaussian_attention_weights(
                                state['pointer_position_c'], self.num_memory_positions
                            )
                            indices_exp_c = indices_c.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                            neighborhood_c = memory_ring.gather(1, indices_exp_c)
                            context_read_c = (weights_c.unsqueeze(-1) * neighborhood_c).sum(dim=1)

                        if self.num_pointers >= 3:
                            # Competitive attention: score each read against hidden state
                            _D = self.embedding_dim
                            reads_stack = torch.stack([context_read_a, context_read_b, context_read_c], dim=1)  # [B, 3, D]
                            attn_scores = (state['hidden_state'].unsqueeze(1) * reads_stack).sum(dim=-1) / (_D ** 0.5)
                            attn_weights = 0.25 * torch.softmax(attn_scores, dim=-1) + 0.25  # [B, 3] min 25% per pointer
                            context_read = (attn_weights.unsqueeze(-1) * reads_stack).sum(dim=1)
                        elif self.num_pointers >= 2:
                            context_read = 0.5 * (context_read_a + context_read_b)
                        else:
                            context_read = context_read_a

                        context_scale = torch.sigmoid(being.context_strength)

                        # CAPACITY FIBONACCI: project ring context D -> H_i
                        if self.ring_read_projs is not None:
                            context_read = self.ring_read_projs[being_idx](context_read)

                        # No input -- only ring context + hidden state + think token
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
                        if self.think_token is not None:
                            combined_input = combined_input + self.think_token[:h_i_tt]  # "I'm thinking" signal

                        # State update (phi EMA — norm hidden only, input path clean)
                        # Matches vectorized path: single EMA + direct assign (no double blend)
                        _norm_tt = self.being_state_norms[being_idx] if self.being_state_norms is not None else self.state_norm
                        state_update = STATE_EMA_ALPHA * _norm_tt(state['hidden_state']) + (1 - STATE_EMA_ALPHA) * combined_input
                        if self.being_processing_layers is not None:
                            for norm, layer in zip(self.being_processing_norms[being_idx], self.being_processing_layers[being_idx]):
                                state_update = state_update + c19_activation(layer(norm(state_update)))
                        elif self.processing_layers is not None:
                            for norm, layer in zip(self.processing_norms, self.processing_layers):
                                state_update = state_update + c19_activation(layer(norm(state_update)))
                        state['hidden_state'] = state_update

                        # Flat LCX read/write during think tick
                        # Each tick reads the next level up, clamped to top level.
                        # tt=0 → L1, tt=1 → L2, ..., tt>=top → re-read top (iterative retrieval).
                        if self._lcx_hash_mode:
                            _lcx_level = min(_tt + 1, self._lcx_num_levels - 1)
                            _do_lcx = self._lcx_num_levels > 0
                            if _do_lcx:
                                _allowed = getattr(self, '_allowed_levels', None)
                                if _allowed is not None and _lcx_level not in _allowed:
                                    _do_lcx = False
                            if _do_lcx:
                                _query = state['hidden_state']
                                lcx_tt_ctx = self._lcx_flat_read(_query, level=_lcx_level)
                                lcx_tt_ctx = self._lcx_bottleneck(lcx_tt_ctx)
                                # zoom_gate: L2-normalize input to bound pre-activation
                                with torch.amp.autocast('cuda', enabled=False):
                                    _zg_dtype = self.zoom_gate.weight.dtype
                                    _q32 = torch.nn.functional.normalize(_query.to(_zg_dtype), dim=-1)
                                    zg = torch.sigmoid(self.zoom_gate(_q32))
                                lcx_tt_ctx = lcx_tt_ctx * zg
                                self._last_zoom_gate = zg.mean().item()
                                # Stash zoom_gate output for anti-saturation aux loss
                                if self.training:
                                    if not hasattr(self, '_lcx_zoom_gate_accum'):
                                        self._lcx_zoom_gate_accum = []
                                    self._lcx_zoom_gate_accum.append(zg)
                                state['hidden_state'] = state['hidden_state'] + lcx_tt_ctx
                                # Write at this level
                                self._lcx_flat_write(state['hidden_state'], state_update,
                                                     level=_lcx_level)

                        # Write to ring (Pointer A)
                        if self.ring_write_projs is not None:
                            write_vec = self.ring_write_projs[being_idx](state_update)
                        else:
                            write_vec = state_update
                        update_broadcast_a = write_vec.unsqueeze(1).expand(-1, weights_a.size(1), -1)
                        contribution_a = weights_a.unsqueeze(-1) * update_broadcast_a
                        memory_ring = memory_ring.scatter_add(1, indices_exp_a, contribution_a)

                        if self.num_pointers >= 2:
                            # Pointer B write (same write_vec, different neighborhood)
                            update_broadcast_b = write_vec.unsqueeze(1).expand(-1, weights_b.size(1), -1)
                            contribution_b = weights_b.unsqueeze(-1) * update_broadcast_b
                            memory_ring = memory_ring.scatter_add(1, indices_exp_b, contribution_b)

                        if self.num_pointers >= 3:
                            # Pointer C write (same write_vec, different neighborhood)
                            update_broadcast_c = write_vec.unsqueeze(1).expand(-1, weights_c.size(1), -1)
                            contribution_c = weights_c.unsqueeze(-1) * update_broadcast_c
                            memory_ring = memory_ring.scatter_add(1, indices_exp_c, contribution_c)

                        # Move pointer A — soft blend for gradient flow
                        current_pos_a = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                        jump_target_a = being.pointer_destinations[current_pos_a].float().clamp(0, self.num_memory_positions - 1)
                        jump_logits_a = being.jump_gate(state_update).squeeze(-1)
                        jump_prob_a = torch.sigmoid(jump_logits_a / JUMP_SIGMOID_TAU)
                        walk_position_a = (state['pointer_position'] + 1.0) % self.num_memory_positions
                        state['pointer_position'] = jump_prob_a * jump_target_a + (1.0 - jump_prob_a) * walk_position_a

                        if self.num_pointers >= 2:
                            # Move pointer B (independent walker)
                            current_pos_b = state['pointer_position_b'].long().clamp(0, self.num_memory_positions - 1)
                            jump_target_b = being.pointer_destinations_b[current_pos_b].float().clamp(0, self.num_memory_positions - 1)
                            jump_logits_b = being.jump_gate_b(state_update).squeeze(-1)
                            jump_prob_b = torch.sigmoid(jump_logits_b / JUMP_SIGMOID_TAU)
                            walk_position_b = (state['pointer_position_b'] + 1.0) % self.num_memory_positions
                            state['pointer_position_b'] = jump_prob_b * jump_target_b + (1.0 - jump_prob_b) * walk_position_b

                        if self.num_pointers >= 3:
                            # Move pointer C (neutral)
                            current_pos_c = state['pointer_position_c'].long().clamp(0, self.num_memory_positions - 1)
                            jump_target_c = being.pointer_destinations_c[current_pos_c].float().clamp(0, self.num_memory_positions - 1)
                            jump_logits_c = being.jump_gate_c(state_update).squeeze(-1)
                            jump_prob_c = torch.sigmoid(jump_logits_c / JUMP_SIGMOID_TAU)
                            walk_position_c = (state['pointer_position_c'] + 1.0) % self.num_memory_positions
                            state['pointer_position_c'] = jump_prob_c * jump_target_c + (1.0 - jump_prob_c) * walk_position_c

                        # Restore gradient state after scanner think tick processing
                        if _is_scanner_tt:
                            torch.set_grad_enabled(_prev_grad_tt)

                    # Detach between think ticks (VRAM fix): only last tick gets gradient
                    if _tt < _eff_think_ticks - 1:
                        for being_idx in range(self.num_beings):
                            being_states[being_idx]['hidden_state'] = being_states[being_idx]['hidden_state'].detach()
                        memory_ring = memory_ring.detach()

                # Regenerate outputs from post-thinking hidden states
                being_outputs_t = []
                hidden_states_t = []
                for being_idx in _active_beings:
                    state = being_states[being_idx]

                    # NULL BEINGS: output zeros, don't overwrite per-being history
                    if self.being_states.get(being_idx, 'null') == 'null':
                        being_outputs_t.append(torch.zeros(
                            x.size(0), self.num_bits, device=x.device, dtype=x.dtype))
                        hidden_states_t.append(state['hidden_state'].clone())
                        continue

                    if self.full_view and self.being_output_projs is not None:
                        output_bits = self.being_output_projs[being_idx](state['hidden_state'])
                    elif self.heterogeneous:
                        vis_idx = self.receptive_masks[being_idx].nonzero(as_tuple=True)[0]
                        output_k = self.being_output_projs[being_idx](state['hidden_state'])
                        output_bits = torch.zeros(
                            state['hidden_state'].size(0), self.num_bits,
                            device=state['hidden_state'].device, dtype=state['hidden_state'].dtype,
                        )
                        output_bits[:, vis_idx] = output_k
                    elif self.byte_token_mode:
                        output_bits = self._byte_decode(state['hidden_state'])
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
                    majority_threshold = being_stack.size(0) / 2.0
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

        # LCX auxiliary losses for gradient flow
        if self.training and self._lcx_hash_mode:
            _wg_acc = getattr(self, '_lcx_write_gate_accum', [])
            if _wg_acc:
                _all_gates = torch.cat([g.reshape(-1) for g in _wg_acc])
                self._lcx_write_gate_aux_loss = -torch.log(_all_gates.mean() + 1e-8)
            else:
                self._lcx_write_gate_aux_loss = torch.tensor(0.0, device=x.device)
            _rw_acc = getattr(self, '_lcx_read_weights_accum', [])
            if _rw_acc:
                _all_w = torch.cat([w.reshape(-1, w.shape[-1]) for w in _rw_acc], dim=0)
                self._lcx_read_attn_aux_loss = -torch.log(_all_w + 1e-8).mean()
            else:
                self._lcx_read_attn_aux_loss = torch.tensor(0.0, device=x.device)
            # Zoom gate anti-saturation aux loss
            _zg_acc = getattr(self, '_lcx_zoom_gate_accum', [])
            if _zg_acc:
                _all_zg = torch.cat([z.reshape(-1) for z in _zg_acc])
                _zg_clamped = _all_zg.clamp(1e-6, 1.0 - 1e-6)
                self._lcx_zoom_gate_aux_loss = -(
                    _zg_clamped * torch.log(_zg_clamped) +
                    (1 - _zg_clamped) * torch.log(1 - _zg_clamped)
                ).mean()
            else:
                self._lcx_zoom_gate_aux_loss = torch.tensor(0.0, device=x.device)
            # Score margin telemetry (not a loss — pure diagnostic)
            _sm_acc = getattr(self, '_lcx_score_margin_accum', [])
            if _sm_acc:
                self._last_score_margin = sum(m for m, _ in _sm_acc) / len(_sm_acc)
                self._last_score_top1 = sum(t for _, t in _sm_acc) / len(_sm_acc)
            else:
                self._last_score_margin = 0.0
                self._last_score_top1 = 0.0
        else:
            self._lcx_write_gate_aux_loss = torch.tensor(0.0, device=x.device)
            self._lcx_read_attn_aux_loss = torch.tensor(0.0, device=x.device)
            self._lcx_zoom_gate_aux_loss = torch.tensor(0.0, device=x.device)

        # 3h. Detach LCX buffers — gradient highway is within-step only
        # Per-level buffers: writes are in-place with no_grad, safety net.
        # Skip unallocated levels (lazy allocation — they are None).
        if self._lcx_hash_mode:
            _alloc = getattr(self, '_lcx_allocated_levels', set(range(self._lcx_num_levels)))
            for _lvl in range(self._lcx_num_levels):
                if _lvl not in _alloc:
                    continue
                _k, _v = self._lcx_level_bufs(_lvl)
                if _v.requires_grad:
                    setattr(self, f'lcx_values_{_lvl}', _v.detach())
                if _k.requires_grad:
                    setattr(self, f'lcx_keys_{_lvl}', _k.detach())

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
                positions_t = [p for p in pointer_positions_log[t] if not (p != p)]  # filter NaN
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
                    if len(outputs_per_being[being_idx]) > 0:
                        being_sequence = torch.stack(outputs_per_being[being_idx])
                    else:
                        # Scanner beings in phase split produce no output
                        being_sequence = torch.zeros(T, B, self._output_dim,
                                                     device=x.device, dtype=_float_dtype)
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

    # ---- TRIPLE POINTER COMPETITIVE ATTENTION TEST ----
    print("-" * 70)
    print("Testing TRIPLE RING POINTERS with competitive attention (num_pointers=3):")
    print()

    # Sequential path (num_beings=1)
    model_dp = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=1, depth=2, num_bits=8, num_pointers=3,
        think_ticks=1,
    )
    total_params_dp = sum(p.numel() for p in model_dp.parameters())
    print(f"  Params: {total_params_dp:,}")
    assert hasattr(model_dp.beings[0], 'pointer_destinations_b'), "pointer_destinations_b missing"
    assert hasattr(model_dp.beings[0], 'jump_gate_b'), "jump_gate_b missing"
    assert hasattr(model_dp.beings[0], 'pointer_destinations_c'), "pointer_destinations_c missing"
    assert hasattr(model_dp.beings[0], 'jump_gate_c'), "jump_gate_c missing"
    print(f"  Pointer A bias: {model_dp.beings[0].jump_gate.bias.item():.2f}")
    print(f"  Pointer B bias: {model_dp.beings[0].jump_gate_b.bias.item():.2f}")
    print(f"  Pointer C bias: {model_dp.beings[0].jump_gate_c.bias.item():.2f}")

    x_dp = torch.randint(0, 2, (4, 16, 8)).float()
    output_dp, stats_dp = model_dp(x_dp, return_stats=True)
    assert output_dp.shape == (4, 16, 8), f"Expected (4,16,8), got {output_dp.shape}"
    print(f"  Output: {output_dp.shape}  OK")
    print(f"  Jump gate (avg): {stats_dp['jump_gate']:.3f}")
    print(f"  Positions logged: {len(stats_dp['pointer_positions_all'])}")

    model_dp.zero_grad()
    target_dp = torch.randint(0, 2, (4, 16, 8)).float()
    loss_dp = nn.functional.binary_cross_entropy_with_logits(output_dp, target_dp)
    loss_dp.backward()
    jgb_grad = model_dp.beings[0].jump_gate_b.weight.grad
    assert jgb_grad is not None and jgb_grad.abs().sum() > 0, "Pointer B jump_gate must receive gradients"
    jgc_grad = model_dp.beings[0].jump_gate_c.weight.grad
    assert jgc_grad is not None and jgc_grad.abs().sum() > 0, "Pointer C jump_gate must receive gradients"
    print(f"  Pointer B jump_gate grad norm: {jgb_grad.norm().item():.4e}  OK")
    print(f"  Pointer C jump_gate grad norm: {jgc_grad.norm().item():.4e}  OK")

    # Vectorized path (combinatorial=True)
    model_dp_v = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=4, depth=2, num_bits=64,
        combiner_mode='masked', combinatorial=True,
        num_pointers=3, think_ticks=1,
    )
    x_dp_v = torch.randint(0, 2, (2, 8, 64)).float()
    out_dp_v, stats_dp_v = model_dp_v(x_dp_v, return_stats=True)
    assert out_dp_v.shape == (2, 8, 64)
    model_dp_v.zero_grad()
    loss_dp_v = nn.functional.binary_cross_entropy_with_logits(out_dp_v, torch.randint(0, 2, (2, 8, 64)).float())
    loss_dp_v.backward()
    jgb_grad_v = model_dp_v.beings[0].jump_gate_b.weight.grad
    assert jgb_grad_v is not None and jgb_grad_v.abs().sum() > 0
    jgc_grad_v = model_dp_v.beings[0].jump_gate_c.weight.grad
    assert jgc_grad_v is not None and jgc_grad_v.abs().sum() > 0
    print(f"  Vectorized path: {out_dp_v.shape}  Grads B+C OK")
    print()

    # ---- HASH LCX TESTS ----
    print("-" * 70)
    print("Testing HASH LCX (256 slots, top-4, 64-d keys):")
    print()

    # Test 1: Single being with zoom hash LCX (sequential path)
    model_hash = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=1, depth=2, num_bits=8,
        use_lcx=True, lcx_mode="hash", lcx_num_slots=32, lcx_key_dim=16, lcx_top_k=4,
        lcx_num_levels=3, lcx_level_slots=[32, 128, 512],
    )
    assert model_hash._lcx_hash_mode, "Hash mode flag should be True"
    # Per-level flat buffers: exact slot counts, no grid rounding
    assert hasattr(model_hash, 'lcx_keys_0'), "Per-level lcx_keys_0 should exist"
    assert model_hash.lcx_keys_0.shape == (32, 16), f"L0 keys shape: {model_hash.lcx_keys_0.shape}"
    assert model_hash.lcx_keys_2.shape == (512, 16), f"L2 keys shape: {model_hash.lcx_keys_2.shape}"
    assert model_hash.lcx_values_0.shape == (32, 64), f"L0 values shape: {model_hash.lcx_values_0.shape}"
    assert model_hash._lcx_level_slots == [32, 128, 512], f"Level slots: {model_hash._lcx_level_slots}"
    assert model_hash.lcx_propose is None, "Dense lcx_propose should be None in hash mode"
    assert model_hash.think_token is not None, "think_token should exist"
    assert hasattr(model_hash, 'zoom_gate'), "zoom_gate should exist"

    hash_params = sum(p.numel() for p in model_hash.parameters())
    print(f"  Zoom Hash LCX params: {hash_params:,} ({model_hash._lcx_num_levels} levels)")

    # Forward pass
    x_hash = torch.randint(0, 2, (2, 4, 8)).float()
    out_hash = model_hash(x_hash)
    assert out_hash.shape == (2, 4, 8), f"Hash LCX: Expected (2,4,8), got {out_hash.shape}"
    print(f"  Forward shape: {out_hash.shape}  OK")

    # Check L0 got written (values should differ from init after forward pass)
    _, vals_L0 = model_hash._lcx_level_bufs(0)
    l0_count = vals_L0.shape[0]
    nonzero_slots = (vals_L0.abs().sum(dim=-1) > 0.011).sum().item()  # > init noise (0.01 scale)
    print(f"  L0 slots with data: {nonzero_slots}/{l0_count}")
    assert nonzero_slots > 0, "At least one L0 slot should have been written"

    # Gradient flow
    model_hash.zero_grad()
    target_hash = torch.randint(0, 2, (2, 4, 8)).float()
    loss_hash = nn.functional.binary_cross_entropy_with_logits(out_hash, target_hash)
    loss_hash.backward()
    grads_hash = sum(1 for p in model_hash.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_hash = sum(1 for _ in model_hash.parameters())
    print(f"  Grads: {grads_hash}/{total_hash}  OK")

    # Test 2: Hash LCX with think ticks (verify detach + think token)
    print()
    print("  Testing hash LCX + think_ticks=2:")
    model_hash_tt = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=1, depth=2, num_bits=8,
        use_lcx=True, lcx_mode="hash", lcx_num_slots=32, lcx_key_dim=16, lcx_top_k=4,
        think_ticks=2,
    )
    x_hash_tt = torch.randint(0, 2, (2, 4, 8)).float()
    out_hash_tt = model_hash_tt(x_hash_tt)
    assert out_hash_tt.shape == (2, 4, 8), f"Hash+TT: Expected (2,4,8), got {out_hash_tt.shape}"
    model_hash_tt.zero_grad()
    loss_hash_tt = nn.functional.binary_cross_entropy_with_logits(out_hash_tt, torch.randint(0, 2, (2, 4, 8)).float())
    loss_hash_tt.backward()
    # Verify think_token got gradient
    assert model_hash_tt.think_token.grad is not None, "think_token should have gradient"
    assert model_hash_tt.think_token.grad.abs().sum() > 0, "think_token gradient should be non-zero"
    print(f"  Shape: {out_hash_tt.shape}  think_token grad norm: {model_hash_tt.think_token.grad.norm():.4f}  OK")

    # Test 3: Hash LCX vectorized path (combinatorial + multi-being)
    print()
    print("  Testing zoom hash LCX vectorized (8 beings, combinatorial):")
    model_hash_vec = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=8, depth=2, num_bits=64,
        combiner_mode='masked', combinatorial=True,
        use_lcx=True, lcx_mode="hash", lcx_num_slots=64, lcx_key_dim=16, lcx_top_k=4,
        lcx_num_levels=2, lcx_level_slots=[64, 256],
    )
    hash_vec_params = sum(p.numel() for p in model_hash_vec.parameters())
    print(f"  Params: {hash_vec_params:,}")
    x_hash_vec = torch.randint(0, 2, (2, 8, 64)).float()
    out_hash_vec = model_hash_vec(x_hash_vec)
    assert out_hash_vec.shape == (2, 8, 64), f"Hash vec: Expected (2,8,64), got {out_hash_vec.shape}"
    model_hash_vec.zero_grad()
    loss_hash_vec = nn.functional.binary_cross_entropy_with_logits(out_hash_vec, torch.randint(0, 2, (2, 8, 64)).float())
    loss_hash_vec.backward()
    grads_hash_vec = sum(1 for p in model_hash_vec.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Shape: {out_hash_vec.shape}  Grads: {grads_hash_vec}/{sum(1 for _ in model_hash_vec.parameters())}  OK")

    # Test 4: Dense LCX still works (regression check)
    print()
    print("  Testing dense LCX (regression check):")
    model_dense = SwarmByteRingModel(
        num_memory_positions=64, embedding_dim=64,
        num_beings=1, depth=2, num_bits=8,
        use_lcx=True, lcx_mode="dense",
    )
    assert not model_dense._lcx_hash_mode, "Dense mode should not set hash flag"
    assert model_dense.lcx is not None, "Dense lcx buffer should exist"
    x_dense = torch.randint(0, 2, (2, 4, 8)).float()
    out_dense = model_dense(x_dense)
    assert out_dense.shape == (2, 4, 8)
    model_dense.zero_grad()
    loss_dense = nn.functional.binary_cross_entropy_with_logits(out_dense, torch.randint(0, 2, (2, 4, 8)).float())
    loss_dense.backward()
    print(f"  Shape: {out_dense.shape}  Grads OK")
    print()

    print("=" * 70)
    print("All tests passed!")
    print(f"N=8 backward compat:      OK")
    print(f"N=64 standardized:        OK ({total_params_64:,} params)")
    print(f"N=64 combinatorial (vec): OK ({total_params_comb:,} params)")
    print(f"Flat LCX (sequential):    OK ({hash_params:,} params)")
    print(f"Flat LCX (vectorized):    OK ({hash_vec_params:,} params)")
    print(f"Dense LCX (regression):   OK")
    print("=" * 70)
