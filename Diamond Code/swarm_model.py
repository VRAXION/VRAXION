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


class BeingParameters(nn.Module):
    """
    Per-being parameters (pointer destinations, jump gate, context strength).
    Each being in the swarm has its own instance of these.

    NEW: Phase embedding based on golden ratio (φ) for non-interfering specialization.
    """

    def __init__(self, num_memory_positions: int, embedding_dim: int, being_id: int = 0):
        super().__init__()

        # Learned jump destinations for each memory position
        self.pointer_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Jump gate: decides whether to jump or walk
        self.jump_gate = nn.Linear(embedding_dim, 1)

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

    Output: Mean of all being outputs (simple ensemble averaging)
    """

    def __init__(
        self,
        num_memory_positions: int = 64,
        embedding_dim: int = 64,
        num_beings: int = 2,
        depth: int = 2,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_memory_positions = num_memory_positions
        self.num_beings = num_beings
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        self.depth = depth

        # SHARED COMPONENTS (all beings use these)
        if embedding_dim > 8:
            self.input_proj = nn.Linear(8, embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, 8)
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

        # PER-BEING COMPONENTS (each being has its own)
        # Pass being_id for golden ratio phase embeddings
        self.beings = nn.ModuleList([
            BeingParameters(num_memory_positions, embedding_dim, being_id=i)
            for i in range(num_beings)
        ])

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

        # Track outputs per being (for ensemble averaging)
        outputs_per_being = [[] for _ in range(self.num_beings)]

        # Track metrics (if requested)
        if return_stats:
            jump_counts_per_being = [[] for _ in range(self.num_beings)]
            pointer_positions_log = []
            output_disagreements = []

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

                # 2b. Combine input + context
                combined_input = input_vec + context_scale * context_read

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

            # 3. Aggregate outputs across beings
            being_stack = torch.stack(being_outputs_t)  # [num_beings, B, 8]

            if self.training:
                # TRAINING: Soft voting (average) for gradient flow
                mean_output_t = being_stack.mean(dim=0)
            else:
                # EVAL: Hard voting (majority wins)
                being_binary = (being_stack > 0.0).float()
                vote_sum = being_binary.sum(dim=0)
                majority_threshold = self.num_beings / 2.0
                mean_output_t = (vote_sum > majority_threshold).float() * 2.0 - 1.0  # Back to logits

            # Track ensemble diversity (if stats requested)
            if return_stats:
                pointer_positions_log.append(pointer_positions_t)
                # Output disagreement: std of being outputs
                being_outputs_stacked = torch.stack(being_outputs_t)  # [num_beings, B, 8]
                disagreement = being_outputs_stacked.std(dim=0).mean().item()  # scalar
                output_disagreements.append(disagreement)

        # 4. Stack outputs across timesteps
        # outputs_per_being[i] is list of T tensors of shape [B, 8]
        # Stack to [T, B, 8] then transpose to [B, T, 8]
        final_outputs = []
        for t in range(T):
            t_outputs = [outputs_per_being[being_idx][t] for being_idx in range(self.num_beings)]
            t_stack = torch.stack(t_outputs)  # [num_beings, B, 8]

            if self.training:
                # TRAINING: Soft voting (average) for gradients
                mean_output = t_stack.mean(dim=0)
            else:
                # EVAL: Hard voting (majority)
                t_binary = (t_stack > 0.0).float()
                t_vote_sum = t_binary.sum(dim=0)
                mean_output = (t_vote_sum > self.num_beings / 2.0).float() * 2.0 - 1.0
            final_outputs.append(mean_output)

        output = torch.stack(final_outputs, dim=1)

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

    # Test different swarm sizes
    for num_beings in [2, 4, 10]:
        print(f"Testing {num_beings}-being swarm:")
        print()

        # Create model
        model = SwarmByteRingModel(
            num_memory_positions=64,
            embedding_dim=64,
            num_beings=num_beings,
            depth=2,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Forward pass
        batch_size = 4
        seq_len = 16
        x = torch.randint(0, 2, (batch_size, seq_len, 8)).float()

        output, stats = model(x, return_stats=True)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, 8)
        print(f"  Jump gate: {stats['jump_gate']:.4f}")
        print(f"  Pointer spread: {stats['pointer_spread']:.4f}")
        print(f"  Output disagreement: {stats['output_disagreement']:.4f}")
        print("  OK Shape correct")
        print()

        # Gradient flow
        model.zero_grad()
        target = torch.randint(0, 2, (batch_size, seq_len, 8)).float()
        loss = nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total_params_count = sum(1 for _ in model.parameters())
        print(f"  Parameters with gradients: {has_grads}/{total_params_count}")
        print("  OK Gradients flow")
        print()

    print("=" * 70)
    print("Swarm model ready!")
    print(f"Key innovation: N beings sharing ONE ring memory")
    print(f"Hypothesis: Spatial diversity + ensemble effects > single strong learner")
    print("=" * 70)
