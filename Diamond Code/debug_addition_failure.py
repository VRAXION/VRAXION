"""
Debug: Why Addition Fails

Visualize what the model actually does at each timestep for addition task.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from byte_ring_model import ByteRingModel
from byte_data import generate_addition_task


def analyze_forward_pass():
    """Trace through one forward pass on addition task."""
    print("=" * 70)
    print("DEBUGGING ADDITION FAILURE")
    print("=" * 70)
    print()

    # Create small model
    model = ByteRingModel(num_memory_positions=8, embedding_dim=32, mobius=False)

    # Generate one addition example: a + b = sum
    x, y = generate_addition_task(n_samples=1, seq_len=4, max_value=20, seed=42)

    # Decode values
    a_val = sum(int(x[0, 0, j].item()) * (2 ** (7 - j)) for j in range(8))
    b_val = sum(int(x[0, 1, j].item()) * (2 ** (7 - j)) for j in range(8))
    sum_val = sum(int(y[0, 2, j].item()) * (2 ** (7 - j)) for j in range(8))

    print(f"Test case: {a_val} + {b_val} = {sum_val}")
    print()
    print(f"Input sequence:  {[a_val, b_val, 0, 0]}")
    print(f"Target sequence: {[a_val, b_val, sum_val, 0]}")
    print()

    # Manual forward pass with instrumentation
    B, T, _ = x.shape
    device = x.device

    # Initialize
    memory_ring = torch.zeros(B, model.num_memory_positions, model.embedding_dim, device=device, dtype=x.dtype)
    hidden_state = torch.zeros(B, model.embedding_dim, device=device, dtype=x.dtype)
    pointer_position = torch.rand(B, device=device) * model.num_memory_positions

    print("TIMESTEP-BY-TIMESTEP ANALYSIS:")
    print("=" * 70)

    for t in range(T):
        print(f"\n[t={t}]")

        # Input
        input_bits = x[:, t, :]
        input_val = sum(int(input_bits[0, j].item()) * (2 ** (7 - j)) for j in range(8))
        target_val = sum(int(y[0, t, j].item()) * (2 ** (7 - j)) for j in range(8))

        print(f"  Input byte:  {input_val:3d} (decimal)")
        print(f"  Target byte: {target_val:3d} (decimal)")

        # Project input
        if model.input_proj is not None:
            input_vec = model.input_proj(input_bits)
            input_magnitude = input_vec.abs().mean().item()
            print(f"  Input projection magnitude: {input_magnitude:.4f}")
        else:
            input_vec = input_bits

        # Read from ring
        indices, weights = model._gaussian_attention_weights(pointer_position, model.num_memory_positions)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, model.embedding_dim)
        neighborhood = memory_ring.gather(1, indices_exp)
        context_read = (weights.unsqueeze(-1) * neighborhood).sum(dim=1)
        context_magnitude = context_read.abs().mean().item()

        print(f"  Pointer position: {pointer_position.item():.2f}")
        print(f"  Context read magnitude: {context_magnitude:.4f}")

        # Combine
        context_scale = torch.sigmoid(model.context_strength)
        combined_input = input_vec + context_scale * context_read

        # Update state
        state_update = torch.tanh(combined_input + hidden_state)
        hidden_magnitude = state_update.abs().mean().item()
        print(f"  Hidden state magnitude: {hidden_magnitude:.4f}")

        # Write to ring
        update_broadcast = state_update.unsqueeze(1).expand(-1, weights.size(1), -1)
        contribution = weights.unsqueeze(-1) * update_broadcast
        memory_ring = memory_ring.scatter_add(1, indices_exp, contribution)

        # Output
        if model.output_proj is not None:
            output_bits = model.output_proj(state_update)
        else:
            output_bits = state_update

        # Decode output
        output_val = sum(int((output_bits[0, j] > 0.5).item()) * (2 ** (7 - j)) for j in range(8))
        match_str = "MATCH" if output_val == target_val else "WRONG"
        print(f"  Output byte: {output_val:3d} (decimal) [{match_str}]")

        # Update pointer (simplified)
        hidden_state = state_update
        current_pos = pointer_position.long().clamp(0, model.num_memory_positions - 1)
        jump_target = model.jump_destinations[current_pos]
        jump_logits = model.jump_gate(state_update).squeeze(-1)
        should_jump = (torch.sigmoid(jump_logits) > 0.5).float()
        walk_position = (pointer_position + 1.0) % model.num_memory_positions
        pointer_position = torch.where(should_jump > 0.5, jump_target, walk_position)

    print()
    print("=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print()
    print("At t=2 (when sum is needed):")
    print("  - Input is ZERO (no signal!)")
    print("  - Context reads ONE blob from ring (wherever pointer happens to be)")
    print("  - Hidden state carries forward info, but it's tanh-saturated")
    print("  - NO mechanism to retrieve BOTH 'a' and 'b' simultaneously")
    print("  - NO dedicated arithmetic unit to compute a+b")
    print()
    print("Why echoing works:")
    print("  - At t=0: Input=a, just copy to output (short circuit)")
    print("  - At t=1: Input=b, just copy to output (short circuit)")
    print()
    print("Why addition fails:")
    print("  - At t=2: Input=0, need to fetch a AND b from memory, compute sum")
    print("  - But architecture only supports: read ONE location + blend with input")
    print("  - No multi-value retrieval, no arithmetic computation space")
    print()
    print("=" * 70)


if __name__ == "__main__":
    analyze_forward_pass()
