"""
Byte Ring Model - Configurable Embedding Dimension

A ring memory model that operates on 8-bit bytes with flexible internal representation.

Key differences from RingMemoryModel:
- Input/Output: Always [B, T, 8] bits (byte-sized)
- Internal embedding_dim: Configurable (default 32)
- If embedding_dim > 8: Uses small Linear(8, dim) and Linear(dim, 8) projections
- If embedding_dim = 8: Direct byte-to-byte (no projections)
- No classification head - reconstruction only

Parameter efficiency:
- embedding_dim=8: ~74 params (no projections)
- embedding_dim=16: ~338 params (8×16 + 16×8 projections)
- embedding_dim=32: ~610 params (8×32 + 32×8 projections)
- embedding_dim=64: ~1,154 params (8×64 + 64×8 projections)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class ByteRingModel(nn.Module):
    """
    Ring memory for byte processing with configurable internal dimension.

    Input: [batch, seq_len, 8] binary bits
    Output: [batch, seq_len, 8] binary bits

    Internal representation: embedding_dim (default 32)
    - If embedding_dim > 8: Uses projections 8→dim→8
    - If embedding_dim = 8: Direct byte-to-byte (no projections)
    """

    def __init__(
        self,
        num_memory_positions: int = 64,
        embedding_dim: int = 32,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
        mobius: bool = False,
    ):
        super().__init__()

        # Configurable embedding dimension (default 32)
        self.embedding_dim = embedding_dim

        self.num_memory_positions = num_memory_positions
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        self.mobius = mobius

        # Möbius helix config
        self.mobius_scale = 2 if mobius else 1
        self.ring_range = int(num_memory_positions * self.mobius_scale)

        # Input/output projections (if embedding_dim != 8)
        if embedding_dim > 8:
            self.input_proj = nn.Linear(8, embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, 8)
        else:
            self.input_proj = None
            self.output_proj = None

        # Pointer control - emergent routing
        self.jump_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Content-based jump gate (embedding_dim → 1)
        self.jump_gate = nn.Linear(embedding_dim, 1)

        # Context mixing strength
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        # Möbius phase embeddings (embedding_dim D)
        if mobius:
            self.phase_embed = nn.ParameterList([
                nn.Parameter(torch.randn(embedding_dim) * 0.1),  # cos component
                nn.Parameter(torch.randn(embedding_dim) * 0.1),  # sin component
            ])
        else:
            self.phase_embed = None

    def _gaussian_attention_weights(
        self,
        pointer_position: torch.Tensor,  # [B]
        num_positions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gaussian attention weights around pointer.

        Returns:
            indices: [B, 2K+1] - position indices
            weights: [B, 2K+1] - attention weights (sum to 1)
        """
        B = pointer_position.size(0)
        K = self.attention_radius

        # Neighborhood offsets: [-K, ..., 0, ..., +K]
        offsets = torch.arange(-K, K + 1, device=pointer_position.device)

        # Base index (floor of pointer)
        base = torch.floor(pointer_position).long().clamp(0, num_positions - 1)

        # Circular indices
        indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions
        indices_float = indices.float()

        # Circular distance
        delta = self._circular_distance(
            pointer_position.unsqueeze(1),
            indices_float,
            num_positions
        )

        # Gaussian logits
        logits = -(delta ** 2) / self.attention_temperature
        weights = torch.softmax(logits, dim=1)

        return indices, weights

    def _circular_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        num_positions: int,
    ) -> torch.Tensor:
        """Shortest distance on circular ring."""
        half = num_positions / 2.0
        return torch.remainder(b - a + half, num_positions) - half

    def _hard_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Straight-Through Estimator for hard binary decisions.

        Forward: Hard binary (0 or 1)
        Backward: Soft gradients from sigmoid
        """
        probs = torch.sigmoid(logits)
        hard = (probs > 0.5).float()
        # STE: forward uses hard, backward uses probs
        return hard - probs.detach() + probs

    def forward(
        self,
        x: torch.Tensor,  # [B, T, 8] - binary bits
    ) -> torch.Tensor:
        """
        Forward pass: byte sequence → ring memory → byte sequence

        Args:
            x: Input byte sequence [batch, seq_len, 8] (binary 0/1)

        Returns:
            output: Predicted byte sequence [batch, seq_len, 8]
        """
        B, T, _ = x.shape

        # Initialize ring memory (embedding_dim vectors at each position)
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )

        # Hidden state (embedding_dim vector)
        hidden_state = torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype)

        # Initialize pointer
        pointer_position = torch.empty(B, device=x.device).uniform_(0, self.num_memory_positions)

        # Möbius holonomy state (±1)
        holonomy_state = torch.randint(0, 2, (B,), device=x.device, dtype=x.dtype) * 2.0 - 1.0

        outputs = []

        # Process sequence
        for t in range(T):
            # 1. Read input bits and project (if needed)
            input_bits = x[:, t, :]  # [B, 8]

            if self.input_proj is not None:
                input_vec = self.input_proj(input_bits)  # [B, embedding_dim]
            else:
                input_vec = input_bits  # [B, 8]

            # 2. Read from ring (Gaussian attention)
            indices, weights = self._gaussian_attention_weights(
                pointer_position,
                self.num_memory_positions
            )

            # Gather neighborhood
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            neighborhood = memory_ring.gather(1, indices_exp)  # [B, 2K+1, embedding_dim]

            # Weighted sum
            context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)  # [B, embedding_dim]

            # Möbius helix: phase modulation
            if self.mobius and self.phase_embed is not None:
                theta = (pointer_position / float(self.num_memory_positions)) * (2.0 * math.pi)
                phase_cos = torch.cos(theta).unsqueeze(1)  # [B, 1]
                phase_sin = torch.sin(theta).unsqueeze(1)  # [B, 1]
                context_read = context_read + holonomy_state.unsqueeze(1) * (
                    phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]
                )

            # 3. Context injection
            context_scale = torch.sigmoid(self.context_strength)
            combined_input = input_vec + context_scale * context_read

            # 4. State update (tanh activation)
            state_update = torch.tanh(combined_input + hidden_state)
            hidden_state = state_update

            # 5. Write to ring (scatter-add)
            update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

            # 6. Pointer update - hard discrete jumps
            old_pointer_position = pointer_position.clone()

            # Current position
            current_pos = pointer_position.long().clamp(0, self.num_memory_positions - 1)

            # Jump destination
            jump_target = self.jump_destinations[current_pos]

            # Content-based jump decision
            jump_logits = self.jump_gate(state_update).squeeze(-1)
            should_jump = self._hard_gate(jump_logits)

            # Walk position
            walk_position = (pointer_position + 1.0) % self.num_memory_positions

            # JUMP or WALK
            pointer_position = torch.where(
                should_jump > 0.5,
                jump_target,
                walk_position
            )

            # Möbius holonomy: flip state on wrap
            if self.mobius:
                wrapped = (pointer_position < 1.0) & (old_pointer_position >= self.num_memory_positions - 1.0)
                holonomy_state = torch.where(wrapped, -holonomy_state, holonomy_state)

            # 7. Output = hidden state → project to 8 bits (if needed)
            if self.output_proj is not None:
                output_bits = self.output_proj(hidden_state)  # [B, 8]
            else:
                output_bits = hidden_state  # [B, 8]

            outputs.append(output_bits)

        # Stack outputs: [B, T, 8]
        output = torch.stack(outputs, dim=1)

        return output


if __name__ == "__main__":
    print("=" * 70)
    print("BYTE RING MODEL TESTS - DIMENSION SWEEP")
    print("=" * 70)
    print()

    # Test all embedding dimensions
    test_dims = [8, 16, 32, 64]
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, 2, (batch_size, seq_len, 8)).float()

    print("Test 1: Parameter Counts Across Dimensions")
    print("-" * 70)
    for dim in test_dims:
        model = ByteRingModel(num_memory_positions=64, embedding_dim=dim, mobius=False)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"embedding_dim={dim:2d}: {total_params:5,} params", end="")

        # Check if projections exist
        if model.input_proj is not None:
            input_p = sum(p.numel() for p in model.input_proj.parameters())
            output_p = sum(p.numel() for p in model.output_proj.parameters())
            print(f"  (proj: {input_p} + {output_p})")
        else:
            print("  (no projections)")
    print()

    # Test 2: Forward pass for all dimensions
    print("Test 2: Forward Pass Across Dimensions")
    print("-" * 70)
    for dim in test_dims:
        model = ByteRingModel(num_memory_positions=64, embedding_dim=dim, mobius=False)
        output = model(x)
        assert output.shape == (batch_size, seq_len, 8), f"Shape mismatch for dim={dim}!"
        print(f"embedding_dim={dim:2d}: OK  Output shape {output.shape}")
    print()

    # Test 3: Gradient flow for 32D
    print("Test 3: Gradient Flow (embedding_dim=32)")
    print("-" * 70)
    model = ByteRingModel(num_memory_positions=64, embedding_dim=32, mobius=False)
    model.zero_grad()
    output = model(x)
    target = torch.randint(0, 2, (batch_size, seq_len, 8)).float()
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)
    loss.backward()

    has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_param_tensors = sum(1 for _ in model.parameters())
    print(f"Parameters with gradients: {has_grads}/{total_param_tensors}")
    assert has_grads > 0, "No gradients!"
    print("OK Gradients flow")
    print()

    # Test 4: Möbius variant (32D)
    print("Test 4: Möbius Variant (embedding_dim=32)")
    print("-" * 70)
    mobius_model = ByteRingModel(num_memory_positions=64, embedding_dim=32, mobius=True)
    mobius_params = sum(p.numel() for p in mobius_model.parameters())
    standard_params = sum(p.numel() for p in model.parameters())
    print(f"Standard: {standard_params:,} params")
    print(f"Möbius:   {mobius_params:,} params")
    print(f"Increase: +{mobius_params - standard_params} (phase embeddings)")

    mobius_output = mobius_model(x)
    assert mobius_output.shape == (batch_size, seq_len, 8), "Möbius shape mismatch!"
    print("OK Möbius forward pass works")
    print()

    print("=" * 70)
    print("All tests passed!")
    print("Dimension range: 8D (74 params) to 64D (~1,154 params)")
    print("=" * 70)
