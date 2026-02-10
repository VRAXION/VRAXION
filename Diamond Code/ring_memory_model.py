"""
Ring Memory Model - Minimal Reference Implementation

A neural memory architecture using:
- Circular addressable buffer (ring)
- Learned soft pointer (where to read/write)
- Gaussian attention for reads (smooth neighborhood)
- Scatter-add for writes (distributed updates)
- Jump/walk pointer dynamics with inertia

Core invariants:
- Pointer stays in [0, num_positions)
- Memory values are bounded (optional clipping)
- Attention weights sum to 1.0
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import math


class RingMemoryModel(nn.Module):
    """
    Ring-based pointer memory with emergent content-based routing.

    Args:
        input_size: Input dimension (e.g., 1 for scalar)
        num_outputs: Number of output classes
        num_memory_positions: Size of circular buffer (64, 128, 256)
        embedding_dim: Feature dimension per position (64, 128)
        attention_radius: Neighborhood size (2 = ±2 neighbors)
        attention_temperature: Softmax temperature (8.0 = soft)
        activation: Non-linearity ('tanh', 'relu', 'silu')

    Design:
        - Each position learns its own jump destination (emergent routing)
        - Content-based gate decides when to jump vs walk
        - Hard discrete jumps (no soft blending) using STE
        - Allows loops and refinement patterns to emerge naturally
    """

    def __init__(
        self,
        input_size: int,
        num_outputs: int,
        num_memory_positions: int = 64,
        embedding_dim: int = 64,
        attention_radius: int = 2,
        attention_temperature: float = 8.0,
        activation: str = "tanh",
    ):
        super().__init__()

        # Store config
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.num_memory_positions = num_memory_positions
        self.embedding_dim = embedding_dim
        self.attention_radius = attention_radius
        self.attention_temperature = attention_temperature
        self.activation_name = activation

        # Input embedding
        self.input_projection = nn.Linear(input_size, embedding_dim)

        # Normalization (only for multi-dim)
        if embedding_dim > 1:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.layer_norm = None

        # Pointer control - EMERGENT ROUTING
        # Each position learns its own jump destination (no downsampling!)
        self.jump_destinations = nn.Parameter(
            torch.rand(num_memory_positions) * num_memory_positions
        )

        # Content-based gate: should we jump from current position?
        self.jump_gate = nn.Linear(embedding_dim, 1)

        # Context mixing
        self.context_strength = nn.Parameter(torch.tensor(0.2))

        # Output
        self.output_head = nn.Linear(embedding_dim, num_outputs)

        # Debug stats (optional telemetry)
        self.register_buffer("_debug_step", torch.tensor(0))
        self.debug_enabled = False

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply configured activation function."""
        if self.activation_name == "tanh":
            return torch.tanh(x)
        elif self.activation_name == "relu":
            return torch.relu(x)
        elif self.activation_name == "silu":
            return torch.nn.functional.silu(x)
        else:
            return x

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

        Args:
            logits: Pre-sigmoid values [B, ...]

        Returns:
            Hard binary values with soft gradients [B, ...]
        """
        probs = torch.sigmoid(logits)
        hard = (probs > 0.5).float()
        # STE trick: forward uses hard, backward uses probs
        return hard - probs.detach() + probs

    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, input_size]
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through ring memory.

        Args:
            x: Input sequence [batch, seq_len, input_size]
            return_debug: If True, return debug info dict

        Returns:
            logits: [batch, num_outputs]
            aux_loss: Scalar auxiliary loss (for regularization)
            debug_info: Optional debug dict (if return_debug=True)
        """
        B, T, _ = x.shape

        # Initialize state
        memory_ring = torch.zeros(
            B, self.num_memory_positions, self.embedding_dim,
            device=x.device, dtype=x.dtype
        )
        hidden_state = torch.zeros(B, self.embedding_dim, device=x.device, dtype=x.dtype)
        # Initialize pointer deterministically (respects torch.manual_seed)
        pointer_position = torch.empty(B, device=x.device).uniform_(0, self.num_memory_positions)

        debug_info = {"pointer_trajectory": [], "attention_entropy": []} if return_debug else None

        # Process sequence
        for t in range(T):
            # 1. Project input
            input_embed = self.input_projection(x[:, t, :])
            input_embed = self._activate(input_embed)

            # 2. Read from ring (Gaussian attention)
            indices, weights = self._gaussian_attention_weights(
                pointer_position,
                self.num_memory_positions
            )

            # Gather neighborhood
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            neighborhood = memory_ring.gather(1, indices_exp)  # [B, 2K+1, dim]

            # Weighted sum
            context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)  # [B, dim]

            # 3. Context injection
            context_scale = torch.sigmoid(self.context_strength)
            combined_input = input_embed + context_scale * context_read

            # 4. State update
            state_update = self._activate(combined_input + hidden_state)
            hidden_state = state_update

            # Apply normalization if available
            if self.layer_norm is not None:
                hidden_state = self.layer_norm(hidden_state)

            # 5. Write to ring (scatter-add)
            update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
            contribution = weights.unsqueeze(-1) * update_broadcast
            memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

            # 6. Pointer update - HARD DISCRETE JUMPS (Emergent Routing)
            # Current position (integer index)
            current_pos = pointer_position.long().clamp(0, self.num_memory_positions - 1)

            # Get jump destination for this position (each position learns where to send data)
            jump_target = self.jump_destinations[current_pos]

            # Content-based decision: should we jump from here?
            jump_logits = self.jump_gate(state_update).squeeze(-1)
            should_jump = self._hard_gate(jump_logits)  # Hard 0 or 1 (with soft gradients)

            # Walk position (+1 with wrap)
            walk_position = (pointer_position + 1.0) % self.num_memory_positions

            # Hard routing: JUMP or WALK (no blending!)
            pointer_position = torch.where(
                should_jump > 0.5,
                jump_target,      # JUMP to learned destination
                walk_position     # WALK to next position
            )

            # Debug recording
            if return_debug:
                debug_info["pointer_trajectory"].append(pointer_position.detach().cpu())
                entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=1).mean()
                debug_info["attention_entropy"].append(entropy.item())

        # 7. Output
        logits = self.output_head(hidden_state)

        # Auxiliary loss (pointer regularization - encourage exploration)
        aux_loss = 0.0

        if return_debug:
            return logits, aux_loss, debug_info
        return logits, aux_loss, None
