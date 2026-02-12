"""
Verify Mobius holonomy bit flips correctly on pointer wrap.

Tests that TRUE Mobius implementation:
1. Detects wrap events (position crosses from high to low)
2. Flips holonomy state (-1 <-> +1) on each wrap
3. Runs without crashes or NaN values
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel


def main():
    print("=" * 70)
    print("TRUE MOBIUS HOLONOMY TEST")
    print("=" * 70)
    print()

    # Small ring for easy verification (wraps happen more frequently)
    num_positions = 8
    embedding_dim = 16

    print(f"Model: {num_positions} positions, {embedding_dim}D embedding")
    print("Testing: holonomy flipping on pointer wrap")
    print()

    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=True,  # Enable TRUE Mobius
    )

    # Create input that forces pointer to walk around ring multiple times
    # With zero input, pointer should walk +1 each step and wrap around
    num_timesteps = 100
    x = torch.zeros(1, num_timesteps, 1)  # Batch=1, 100 timesteps, 1D input

    print(f"Running {num_timesteps} timesteps with zero input...")
    print("(Pointer should walk +1 each step and wrap at boundaries)")
    print()

    # Forward pass with debug info
    logits, aux_loss, debug_info = model(x, return_debug=True)

    # Extract trajectories
    pointer_traj = torch.stack(debug_info['pointer_trajectory']).squeeze()  # [T]
    holonomy_traj = torch.stack(debug_info['holonomy_trajectory']).squeeze()  # [T]

    print(f"Pointer trajectory (first 20 steps):")
    print(pointer_traj[:20].numpy())
    print()

    print(f"Holonomy trajectory (first 20 steps):")
    print(holonomy_traj[:20].numpy())
    print()

    # Detect wrap events
    wraps = []
    holonomy_flips = []

    for t in range(1, len(pointer_traj)):
        # Wrap: pointer drops from high to low
        if pointer_traj[t] < 1.0 and pointer_traj[t-1] >= num_positions - 1.0:
            wraps.append(t)
            print(f"[WRAP] Step {t}: position {pointer_traj[t-1]:.2f} -> {pointer_traj[t]:.2f}")

        # Holonomy flip: sign changes
        if holonomy_traj[t] != holonomy_traj[t-1]:
            holonomy_flips.append(t)
            print(f"[FLIP] Step {t}: holonomy {holonomy_traj[t-1]:.1f} -> {holonomy_traj[t]:.1f}")

    print()
    print("=" * 70)
    print(f"Total wraps detected: {len(wraps)}")
    print(f"Total holonomy flips: {len(holonomy_flips)}")
    print()

    # Verify wraps and flips align
    if len(wraps) == len(holonomy_flips):
        print("SUCCESS: Number of wraps matches number of holonomy flips")

        # Check if they happen at the same timesteps
        if wraps == holonomy_flips:
            print("SUCCESS: Wraps and flips occur at the same timesteps")
        else:
            print("WARNING: Wraps and flips detected at different timesteps")
            print(f"  Wraps at: {wraps[:5]}...")
            print(f"  Flips at: {holonomy_flips[:5]}...")
    else:
        print(f"WARNING: Mismatch - {len(wraps)} wraps but {len(holonomy_flips)} flips")

    print()

    # Check for NaN or Inf
    has_nan_logits = torch.isnan(logits).any().item()
    has_inf_logits = torch.isinf(logits).any().item()
    has_nan_ptr = torch.isnan(pointer_traj).any().item()
    has_inf_ptr = torch.isinf(pointer_traj).any().item()

    if has_nan_logits or has_inf_logits or has_nan_ptr or has_inf_ptr:
        print("ERROR: NaN or Inf values detected!")
        print(f"  Logits: NaN={has_nan_logits}, Inf={has_inf_logits}")
        print(f"  Pointer: NaN={has_nan_ptr}, Inf={has_inf_ptr}")
    else:
        print("SUCCESS: No NaN or Inf values detected")

    print()
    print("=" * 70)
    print()

    # Show holonomy state over time
    print("Holonomy state coverage:")
    num_positive = (holonomy_traj == 1.0).sum().item()
    num_negative = (holonomy_traj == -1.0).sum().item()
    print(f"  +1: {num_positive}/{len(holonomy_traj)} steps ({100*num_positive/len(holonomy_traj):.1f}%)")
    print(f"  -1: {num_negative}/{len(holonomy_traj)} steps ({100*num_negative/len(holonomy_traj):.1f}%)")
    print()

    if num_positive > 0 and num_negative > 0:
        print("SUCCESS: Both holonomy states used (double-cover active)")
    else:
        print("WARNING: Only one holonomy state used (no wraps or flips disabled)")

    print()
    print("Test complete!")


if __name__ == "__main__":
    main()
