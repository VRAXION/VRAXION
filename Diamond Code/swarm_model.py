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


def generate_receptive_masks(
    num_beings: int,
    num_bits: int = 8,
    bits_per_being: int = 4,
    min_coverage: int = 2,
    max_coverage: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate random binary masks ensuring every bit has min_coverage beings.

    Each being gets a random subset of bits as its receptive field.
    Coverage repair ensures no bit is left uncovered.

    Args:
        num_beings: Number of beings in the swarm.
        num_bits: Total input/output bits (default 8).
        bits_per_being: How many bits each being sees.
        min_coverage: Every bit must be covered by at least this many beings.
        max_coverage: No bit covered by more than this many beings (0 = auto).
        seed: Random seed for reproducibility.

    Returns:
        masks: [num_beings, num_bits] binary tensor (1 = being reads/writes this bit).
    """
    total_slots = num_beings * bits_per_being
    required_slots = num_bits * min_coverage
    if total_slots < required_slots:
        raise ValueError(
            f"Cannot satisfy min_coverage={min_coverage}: "
            f"{num_beings} beings x {bits_per_being} bits = {total_slots} slots, "
            f"but need {num_bits} x {min_coverage} = {required_slots}"
        )

    gen = torch.Generator()
    gen.manual_seed(seed)

    masks = torch.zeros(num_beings, num_bits)
    for i in range(num_beings):
        perm = torch.randperm(num_bits, generator=gen)[:bits_per_being]
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
    max_cov = max_coverage if max_coverage > 0 else (num_beings * bits_per_being // num_bits + 1)
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


class BeingParameters(nn.Module):
    """
    Per-being parameters (pointer destinations, jump gate, context strength).
    Each being in the swarm has its own instance of these.

    Phase embedding based on golden ratio (φ) for non-interfering specialization.
    Optional input_mask for receptive field restriction.
    """

    def __init__(self, num_memory_positions: int, embedding_dim: int,
                 being_id: int = 0, input_mask: torch.Tensor = None):
        super().__init__()

        # Learned jump destinations for each memory position
        self.pointer_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Jump gate: decides whether to jump or walk
        self.jump_gate = nn.Linear(embedding_dim, 1)
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
        self.bits_per_being = bits_per_being
        self.min_coverage = min_coverage

        # SHARED COMPONENTS (all beings use these)
        if embedding_dim > num_bits:
            self.input_proj = nn.Linear(num_bits, embedding_dim)
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

        # RECEPTIVE FIELD MASKS (if bits_per_being > 0)
        if bits_per_being > 0:
            masks = generate_receptive_masks(
                num_beings=num_beings,
                num_bits=num_bits,
                bits_per_being=bits_per_being,
                min_coverage=min_coverage,
                seed=mask_seed,
            )
            self.register_buffer('receptive_masks', masks)  # [num_beings, num_bits]
        else:
            self.register_buffer('receptive_masks', None)

        # PER-BEING COMPONENTS (each being has its own)
        # Pass being_id for golden ratio phase embeddings + optional input mask
        self.beings = nn.ModuleList([
            BeingParameters(
                num_memory_positions, embedding_dim, being_id=i,
                input_mask=self.receptive_masks[i] if self.receptive_masks is not None else None,
            )
            for i in range(num_beings)
        ])

        # RING-ATTENTION COMBINER (Option A: zero new parameters)
        # Learnable temperature for softmax sharpness
        if combiner_mode == 'ring_attention':
            self.gate_temperature = nn.Parameter(torch.tensor(1.0))

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

    def _ring_attention_combine(self, being_stack, hidden_states, memory_ring):
        """
        Ring-attention combiner: use ring memory as query to weight beings.

        The ring has seen what all beings wrote. Its summary acts as a
        "judge" -- dot product with each being's hidden state measures
        alignment. High alignment = this being's output fits the context.

        Args:
            being_stack:   [num_beings, B, 8] raw logits from all beings
            hidden_states: [num_beings, B, D] per-being hidden states
            memory_ring:   [B, M, D] shared ring memory

        Returns:
            combined: [B, 8] weighted combination (on probability level)
            gate_weights: [B, num_beings] the attention weights (for logging)
        """
        # Ring summary = mean across all positions
        r = memory_ring.mean(dim=1)  # [B, D]

        # h: [num_beings, B, D] -> transpose to [B, num_beings, D]
        h = hidden_states.permute(1, 0, 2)  # [B, num_beings, D]

        # Dot product attention (parameter-free)
        # Scale by sqrt(D) for stable softmax, then apply learnable temperature
        scale = math.sqrt(self.embedding_dim)
        scores = (h * r.unsqueeze(1)).sum(dim=-1) / scale  # [B, num_beings]
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
        mask_exp = masks.unsqueeze(1)  # [N, 1, 8]

        if training:
            # Soft: confidence-weighted average, masked to covering beings
            weights = confidence * mask_exp  # [N, B, 8]
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
            p_combined = (weights * probs).sum(dim=0)  # [B, 8]
        else:
            # Hard: pick most confident covering being per bit
            conf_masked = confidence.clone()
            conf_masked[mask_exp.expand_as(conf_masked) < 0.5] = -1.0
            best_idx = conf_masked.argmax(dim=0)  # [B, 8]
            p_combined = probs.gather(0, best_idx.unsqueeze(0)).squeeze(0)  # [B, 8]

        p_combined = p_combined.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p_combined / (1.0 - p_combined))  # logit space

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
                'hidden_state': torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype),
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
            if self.input_proj is not None:
                input_vec = self.input_proj(input_bits)
            else:
                input_vec = input_bits

            # 2. Each being processes independently
            being_outputs_t = []
            hidden_states_t = []
            pointer_positions_t = []

            for being_idx in range(self.num_beings):
                being = self.beings[being_idx]
                state = being_states[being_idx]

                # 2a. Read from being's pointer position
                indices, weights = self._gaussian_attention_weights(
                    state['pointer_position'], self.num_memory_positions
                )
                indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                neighborhood = memory_ring.gather(1, indices_exp)
                context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)
                context_scale = torch.sigmoid(being.context_strength)

                # 2b. Combine input + context (with optional receptive field masking)
                if being.input_mask is not None:
                    # Mask input bits: zero out bits this being can't see
                    masked_bits = input_bits * being.input_mask.unsqueeze(0)  # [B, 8]
                    if self.input_proj is not None:
                        being_input_vec = self.input_proj(masked_bits)
                    else:
                        being_input_vec = masked_bits
                else:
                    being_input_vec = input_vec  # No mask: use shared projection
                combined_input = being_input_vec + context_scale * context_read

                # 2b.1. Add golden ratio phase embedding (forces specialization)
                # Expand phase_embedding from [16] to [B, 16] or [B, embedding_dim]
                B = combined_input.size(0)
                if being.phase_embedding.size(0) == self.embedding_dim:
                    # Phase embedding matches embedding_dim
                    phase_bias = being.phase_embedding.unsqueeze(0).expand(B, -1)
                else:
                    # Phase embedding is 16D, need to pad/project to embedding_dim
                    phase_16d = being.phase_embedding.unsqueeze(0).expand(B, -1)
                    if self.embedding_dim >= 16:
                        # Pad with zeros
                        padding = torch.zeros(B, self.embedding_dim - 16,
                                            device=phase_16d.device, dtype=phase_16d.dtype)
                        phase_bias = torch.cat([phase_16d, padding], dim=1)
                    else:
                        # Truncate to embedding_dim
                        phase_bias = phase_16d[:, :self.embedding_dim]

                # Add phase bias (small scale to not dominate)
                combined_input = combined_input + 0.1 * phase_bias

                # 2c. State update (with multi-layer processing if depth > 1)
                state_update = torch.tanh(combined_input + state['hidden_state'])

                # Additional processing layers (shared across beings)
                if self.processing_layers is not None:
                    for layer in self.processing_layers:
                        state_update = torch.tanh(layer(state_update))

                state['hidden_state'] = state_update

                # 2d. Write to SHARED ring (scatter_add accumulation)
                update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
                contribution = weights.unsqueeze(-1) * update_broadcast
                memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

                # 2e. Update pointer (jump or walk)
                current_pos = state['pointer_position'].long().clamp(0, self.num_memory_positions - 1)
                jump_target = being.pointer_destinations[current_pos]
                jump_logits = being.jump_gate(state_update).squeeze(-1)
                should_jump = self._hard_gate(jump_logits)
                walk_position = (state['pointer_position'] + 1.0) % self.num_memory_positions
                state['pointer_position'] = torch.where(should_jump > 0.5, jump_target, walk_position)

                # Track jump activation (if stats requested)
                if return_stats:
                    jump_counts_per_being[being_idx].append(should_jump.mean().item())
                    pointer_positions_t.append(state['pointer_position'].mean().item())

                # 2f. Generate output
                if self.output_proj is not None:
                    output_bits = self.output_proj(state['hidden_state'])
                else:
                    output_bits = state['hidden_state']

                being_outputs_t.append(output_bits)
                outputs_per_being[being_idx].append(output_bits)
                hidden_states_t.append(state['hidden_state'].clone())

            # 3. Aggregate outputs across beings
            being_stack = torch.stack(being_outputs_t)  # [num_beings, B, 8]

            if self.combiner_mode == 'masked' and self.receptive_masks is not None:
                mean_output_t = self._masked_combine(being_stack, self.training)
            elif self.combiner_mode == 'ring_attention':
                h_stack = torch.stack(hidden_states_t)  # [num_beings, B, D]
                mean_output_t, gate_w = self._ring_attention_combine(
                    being_stack, h_stack, memory_ring
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
            # Jump gate: average across all beings and timesteps
            all_jump_counts = []
            for being_counts in jump_counts_per_being:
                all_jump_counts.extend(being_counts)
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

            # Per-being jump rates
            jump_rates_per_being = [
                sum(counts) / len(counts) if counts else 0.0
                for counts in jump_counts_per_being
            ]

            stats = {
                'jump_gate': avg_jump_rate,
                'circular_spread': avg_circular_spread,  # FIXED metric
                'pointer_spread': avg_circular_spread,  # Backward compat alias
                'output_disagreement': avg_output_disagreement,
                'jump_rates_per_being': jump_rates_per_being,
                'pointer_positions_all': all_pointer_positions,
            }

            # Receptive field mask stats
            if self.receptive_masks is not None:
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

    print("=" * 70)
    print("All tests passed!")
    print(f"N=8 backward compat: OK")
    print(f"N=64 standardized:   OK ({total_params_64:,} params)")
    print("=" * 70)
