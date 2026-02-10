"""
Debug script to verify jump mechanism is working correctly.

Tests:
1. Jump gate responds to input
2. Jump destinations are learned
3. Pointer actually moves when jumping
"""

import torch
from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def main():
    print("=" * 70)
    print("JUMP MECHANISM DEBUG")
    print("=" * 70)
    print()

    # Create model
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=64,
        embedding_dim=64,
    )

    print("1. Initial jump_destinations (should be random 0-63):")
    print(f"   {model.jump_destinations.detach().cpu().numpy()[:10]}... (first 10)")
    print()

    # Generate test data
    x_test, y_test, _ = generate_assoc_clean(n_samples=10, seq_len=32, keys=2, pairs=1, seed=42)

    # Forward pass with debug
    model.eval()
    with torch.no_grad():
        logits, aux_loss, debug_info = model(x_test, return_debug=True)

    # Check jump decisions
    jump_decisions = torch.stack(debug_info['jump_decisions'])  # [seq_len, batch]
    print(f"2. Jump decisions shape: {jump_decisions.shape}")
    print(f"   Jump rate: {jump_decisions.float().mean().item():.4f}")
    print(f"   Jump decisions per timestep (first sample):")
    for t in range(min(10, jump_decisions.size(0))):
        jumps = jump_decisions[t, 0].item()
        print(f"      t={t}: {jumps:.2f}")
    print()

    # Check pointer trajectory
    pointer_traj = torch.stack(debug_info['pointer_trajectory'])  # [seq_len, batch]
    print(f"3. Pointer trajectory (first sample):")
    for t in range(min(10, pointer_traj.size(0))):
        pos = pointer_traj[t, 0].item()
        jump = jump_decisions[t, 0].item()
        print(f"      t={t}: pos={pos:.2f}, jumped={'YES' if jump > 0.5 else 'NO'}")
    print()

    # Check if pointer moves sequentially (walk) or jumps
    pointer_diffs = []
    for t in range(1, min(10, pointer_traj.size(0))):
        prev_pos = pointer_traj[t-1, 0].item()
        curr_pos = pointer_traj[t, 0].item()
        diff = (curr_pos - prev_pos) % 64
        pointer_diffs.append(diff)
        expected = "WALK (+1)" if abs(diff - 1.0) < 0.1 else f"JUMP (+{diff:.1f})"
        print(f"      t={t-1}->{t}: {prev_pos:.1f} -> {curr_pos:.1f} = {expected}")
    print()

    print("=" * 70)
    print("SUMMARY:")
    print(f"  Jump mechanism active: {jump_decisions.float().mean().item() > 0.01}")
    print(f"  Pointer movement: {'sequential' if all(abs(d - 1.0) < 0.1 for d in pointer_diffs) else 'jumping'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
