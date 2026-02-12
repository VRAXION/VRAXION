"""
Visualize the routing graph to understand why position 31 fails to reach 25-26.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from visualize_routing import detect_loops
import numpy as np


def trace_routing_path(jump_destinations, start_pos, max_steps=10):
    """
    Trace the routing path from a starting position.

    Returns list of positions visited.
    """
    path = [start_pos]
    visited = {start_pos}
    current = start_pos

    for _ in range(max_steps):
        next_pos = int(jump_destinations[current].item()) % len(jump_destinations)
        path.append(next_pos)

        if next_pos in visited:
            # Hit a cycle
            break
        visited.add(next_pos)
        current = next_pos

    return path


def visualize_routing(checkpoint_path):
    """Visualize the routing graph and analyze position 31 -> 25."""

    print("=" * 70)
    print("ROUTING GRAPH ANALYSIS - Why Position 31 Fails")
    print("=" * 70)
    print()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=64,
        embedding_dim=256,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract jump destinations
    jump_dest = model.jump_destinations.detach().cpu().numpy()
    num_pos = len(jump_dest)

    print(f"Ring size: {num_pos} positions")
    print()

    # Detect routing patterns
    cycles, self_loops = detect_loops(model.jump_destinations)
    print(f"Detected cycles: {len(cycles)}")
    print(f"Self-loops (refinement stations): {len(self_loops)}")
    print()

    # Focus on the failure positions: 25, 26, 31
    print("=" * 70)
    print("FOCUS: Positions 25, 26 (key-value) and 31 (query)")
    print("=" * 70)
    print()

    positions_of_interest = [25, 26, 31]

    for pos in positions_of_interest:
        target = int(jump_dest[pos]) % num_pos
        print(f"Position {pos:2d} -> jumps to -> Position {target:2d}")

    print()
    print("-" * 70)
    print("Routing Paths (following jumps):")
    print("-" * 70)
    print()

    # Trace from position 31 (where query is)
    print("FROM POSITION 31 (query position):")
    path_31 = trace_routing_path(model.jump_destinations, 31, max_steps=15)
    path_str = " -> ".join(map(str, path_31))
    print(f"  Path: {path_str}")

    if 25 in path_31 or 26 in path_31:
        reach_pos = 25 if 25 in path_31 else 26
        steps = path_31.index(reach_pos)
        print(f"  ✓ CAN reach position {reach_pos} in {steps} jumps")
    else:
        print(f"  ✗ CANNOT reach positions 25 or 26!")
        print(f"     This explains the failures!")
    print()

    # Trace from position 25 (where key-value is)
    print("FROM POSITION 25 (key-value position):")
    path_25 = trace_routing_path(model.jump_destinations, 25, max_steps=10)
    path_str = " -> ".join(map(str, path_25))
    print(f"  Path: {path_str}")

    if 31 in path_25:
        steps = path_25.index(31)
        print(f"  ✓ CAN reach position 31 in {steps} jumps")
    else:
        print(f"  ✗ CANNOT reach position 31")
    print()

    # Show surrounding positions
    print("-" * 70)
    print("Nearby Positions (context):")
    print("-" * 70)
    print()
    print("Positions 20-35 (failure region):")
    for pos in range(20, 36):
        target = int(jump_dest[pos]) % num_pos

        # Calculate jump distance
        if target > pos:
            dist = target - pos
            direction = f"+{dist}"
        else:
            dist = pos - target
            direction = f"-{dist}"

        # Mark special positions
        marker = ""
        if pos == 25:
            marker = "  <- KEY POSITION"
        elif pos == 26:
            marker = "  <- VALUE POSITION"
        elif pos == 31:
            marker = "  <- QUERY POSITION (FAILS HERE!)"
        elif pos in self_loops:
            marker = "  <- SELF-LOOP (refinement station)"

        print(f"  Pos {pos:2d} -> {target:2d} (jump {direction:>4s}){marker}")

    print()

    # Jump distance statistics
    print("-" * 70)
    print("Jump Distance Analysis:")
    print("-" * 70)
    print()

    jump_distances = []
    for i in range(num_pos):
        target = int(jump_dest[i]) % num_pos
        # Calculate circular distance
        if target >= i:
            dist = target - i
        else:
            dist = num_pos - i + target
        jump_distances.append(dist)

    mean_dist = np.mean(jump_distances)
    std_dist = np.std(jump_distances)

    print(f"Mean jump distance: {mean_dist:.1f} positions")
    print(f"Std jump distance: {std_dist:.1f} positions")
    print()

    # Position 31 specifically
    pos_31_target = int(jump_dest[31]) % num_pos
    if pos_31_target >= 31:
        dist_31 = pos_31_target - 31
    else:
        dist_31 = num_pos - 31 + pos_31_target

    print(f"Position 31 jump distance: {dist_31} positions")
    print(f"  (needs to reach position 25, which is {31-25}=6 positions BACKWARD)")
    print()

    # Reachability matrix for positions 25-35
    print("-" * 70)
    print("Reachability Matrix (can position X reach Y in ≤5 jumps?):")
    print("-" * 70)
    print()

    reach_range = range(20, 36)
    print("     ", end="")
    for target in reach_range:
        print(f"{target:3d}", end="")
    print()

    for start in reach_range:
        print(f"{start:3d}: ", end="")
        path = trace_routing_path(model.jump_destinations, start, max_steps=5)
        for target in reach_range:
            if target in path:
                steps = path.index(target)
                if steps == 0:
                    print("  ●", end="")  # Same position
                else:
                    print(f"{steps:3d}", end="")  # Number of jumps
            else:
                print("  .", end="")  # Not reachable
        print()

    print()
    print("Legend: ● = same position, number = jumps needed, . = not reachable")
    print()
    print("=" * 70)


if __name__ == "__main__":
    visualize_routing("checkpoints/assoc_clean_step_1900.pt")
