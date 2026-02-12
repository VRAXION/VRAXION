"""
Dual-Pointer Byte Ring Model

Key innovation: TWO independent pointers instead of one.
This allows reading from MULTIPLE memory locations simultaneously.

Hypothesis: This will enable addition (needs both a AND b).
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class DualPointerByteRingModel(nn.Module):
    """
    Ring memory with TWO pointers for multi-location access.

    Pointer 1: "Primary" - tracks current focus
    Pointer 2: "Secondary" - tracks previous important info

    At each timestep, reads from BOTH positions and combines them.
    """

    def __init__(
        self,
        num_memory_positions: int = 64,
        embedding_dim: int = 32,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
        depth: int = 1,
        use_dual_pointers: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_memory_positions = num_memory_positions
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        self.depth = depth
        self.use_dual_pointers = use_dual_pointers

        # Input/output projections
        if embedding_dim > 8:
            self.input_proj = nn.Linear(8, embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, 8)
        else:
            self.input_proj = None
            self.output_proj = None

        # POINTER 1 (always present)
        self.pointer1_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )
        self.pointer1_jump_gate = nn.Linear(embedding_dim, 1)
        self.context1_strength = nn.Parameter(torch.tensor(0.2))

        # POINTER 2 (optional - only if use_dual_pointers=True)
        if use_dual_pointers:
            self.pointer2_destinations = nn.Parameter(
                torch.rand(num_memory_positions) * num_memory_positions
            )
            self.pointer2_jump_gate = nn.Linear(embedding_dim, 1)
            self.context2_strength = nn.Parameter(torch.tensor(0.2))
        else:
            self.pointer2_destinations = None
            self.pointer2_jump_gate = None
            self.context2_strength = None

        # Multi-layer processing (if depth > 1)
        if depth > 1:
            self.processing_layers = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(depth - 1)
            ])
        else:
            self.processing_layers = None

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

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        Forward pass with DUAL pointers.

        Args:
            x: [B, T, 8] byte sequence
            return_stats: If True, return (output, stats) with jump gate stats

        Returns:
            output: [B, T, 8] predicted sequence
            stats: Dict with jump_gate activation rate (if return_stats=True)
        """
        B, T, _ = x.shape

        # Initialize ring memory
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )

        # Hidden state
        hidden_state = torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype)

        # Initialize pointer1 (always)
        pointer1_position = torch.empty(B, device=x.device).uniform_(0, self.num_memory_positions * 0.3)

        # Initialize pointer2 (only if dual mode)
        if self.use_dual_pointers:
            pointer2_position = torch.empty(B, device=x.device).uniform_(
                self.num_memory_positions * 0.7, self.num_memory_positions
            )

        outputs = []
        jump_counts = [] if return_stats else None

        for t in range(T):
            # 1. Read input
            input_bits = x[:, t, :]

            if self.input_proj is not None:
                input_vec = self.input_proj(input_bits)
            else:
                input_vec = input_bits

            # 2. Read from pointer1 (always)
            indices1, weights1 = self._gaussian_attention_weights(
                pointer1_position, self.num_memory_positions
            )
            indices1_exp = indices1.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            neighborhood1 = memory_ring.gather(1, indices1_exp)
            context_read1 = (weights1.unsqueeze(-1) * neighborhood1).sum(dim=1)
            context_scale1 = torch.sigmoid(self.context1_strength)

            # Read from pointer2 (only if dual mode)
            if self.use_dual_pointers:
                indices2, weights2 = self._gaussian_attention_weights(
                    pointer2_position, self.num_memory_positions
                )
                indices2_exp = indices2.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
                neighborhood2 = memory_ring.gather(1, indices2_exp)
                context_read2 = (weights2.unsqueeze(-1) * neighborhood2).sum(dim=1)
                context_scale2 = torch.sigmoid(self.context2_strength)

                # 3. Combine BOTH contexts
                combined_input = (
                    input_vec
                    + context_scale1 * context_read1
                    + context_scale2 * context_read2
                )
            else:
                # 3. Single context only
                combined_input = input_vec + context_scale1 * context_read1

            # 4. State update (with multi-layer processing if depth > 1)
            state_update = torch.tanh(combined_input + hidden_state)

            # Additional processing layers (more neurons = more computation)
            if self.processing_layers is not None:
                for layer in self.processing_layers:
                    state_update = torch.tanh(layer(state_update))

            hidden_state = state_update

            # 5. Write to ring (from pointer1 position)
            update_broadcast = state_update.unsqueeze(1).expand(-1, weights1.size(1), -1)
            contribution = weights1.unsqueeze(-1) * update_broadcast
            memory_ring = memory_ring.scatter_add(1, indices1_exp, contribution)

            # 6. Update pointer1 (always)
            current_pos1 = pointer1_position.long().clamp(0, self.num_memory_positions - 1)
            jump_target1 = self.pointer1_destinations[current_pos1]
            jump_logits1 = self.pointer1_jump_gate(state_update).squeeze(-1)
            should_jump1 = self._hard_gate(jump_logits1)
            walk_position1 = (pointer1_position + 1.0) % self.num_memory_positions
            pointer1_position = torch.where(should_jump1 > 0.5, jump_target1, walk_position1)

            # Track jump gate activation (if stats requested)
            if return_stats:
                jump_counts.append(should_jump1.mean().item())

            # Update pointer2 (only if dual mode)
            if self.use_dual_pointers:
                current_pos2 = pointer2_position.long().clamp(0, self.num_memory_positions - 1)
                jump_target2 = self.pointer2_destinations[current_pos2]
                jump_logits2 = self.pointer2_jump_gate(state_update).squeeze(-1)
                should_jump2 = self._hard_gate(jump_logits2)
                walk_position2 = (pointer2_position + 1.0) % self.num_memory_positions
                pointer2_position = torch.where(should_jump2 > 0.5, jump_target2, walk_position2)

            # 7. Output
            if self.output_proj is not None:
                output_bits = self.output_proj(hidden_state)
            else:
                output_bits = hidden_state

            outputs.append(output_bits)

        # Return output with optional stats
        output = torch.stack(outputs, dim=1)
        if return_stats:
            stats = {
                'jump_gate': sum(jump_counts) / len(jump_counts) if jump_counts else 0.5
            }
            return output, stats
        else:
            return output


if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-POINTER BYTE RING MODEL TEST")
    print("=" * 70)
    print()

    # Create model
    model = DualPointerByteRingModel(num_memory_positions=64, embedding_dim=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Forward pass
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, 2, (batch_size, seq_len, 8)).float()
    print(f"Input shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 8)
    print("OK Shape correct")
    print()

    # Gradient flow
    model.zero_grad()
    target = torch.randint(0, 2, (batch_size, seq_len, 8)).float()
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)
    loss.backward()

    has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Parameters with gradients: {has_grads}/{sum(1 for _ in model.parameters())}")
    print("OK Gradients flow")
    print()

    print("=" * 70)
    print("Dual-pointer model ready!")
    print(f"Key innovation: TWO pointers = can read from TWO locations")
    print(f"This might enable addition (needs both a AND b)")
    print("=" * 70)
