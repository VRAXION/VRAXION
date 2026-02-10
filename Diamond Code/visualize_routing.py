"""
Visualize emergent routing patterns in RingMemoryModel.

Shows:
- Jump destination graph (directed adjacency)
- Loop detection (cycles, refinement stations)
- Jump probability heatmap per position
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List
sys.path.insert(0, str(Path(__file__).parent))
from ring_memory_model import RingMemoryModel


def detect_loops(jump_destinations: torch.Tensor, max_depth: int = 10) -> Tuple[List[List[int]], List[int]]:
    """
    Detect loops in jump destinations.

    Args:
        jump_destinations: [num_positions] tensor of jump targets
        max_depth: Maximum cycle length to detect

    Returns:
        cycles: List of cycles (lists of positions)
        self_loops: List of positions that jump to themselves
    """
    jump_dest = jump_destinations.detach().cpu().numpy()
    num_pos = len(jump_dest)

    self_loops = []
    cycles = []

    visited_global = set()

    for start in range(num_pos):
        if start in visited_global:
            continue

        path = [start]
        visited_local = {start}
        current = int(jump_dest[start]) % num_pos

        for _ in range(max_depth):
            if current == start:
                # Self-loop
                if len(path) == 1:
                    self_loops.append(start)
                else:
                    cycles.append(path)
                visited_global.update(path)
                break

            if current in visited_local:
                # Cycle detected (but doesn't return to start)
                cycle_start_idx = path.index(current)
                cycle = path[cycle_start_idx:]
                cycles.append(cycle)
                visited_global.update(cycle)
                break

            path.append(current)
            visited_local.add(current)
            current = int(jump_dest[current]) % num_pos
        else:
            # No loop found within max_depth
            visited_global.update(path)

    return cycles, self_loops


def visualize_routing_graph(model: RingMemoryModel, save_path: str = "routing_graph.png") -> None:
    """
    Visualize jump destinations as a directed graph.

    Args:
        model: Trained RingMemoryModel
        save_path: Path to save the figure
    """
    jump_dest = model.jump_destinations.detach().cpu().numpy()
    num_pos = len(jump_dest)

    # Detect loops
    cycles, self_loops = detect_loops(model.jump_destinations)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ============================================
    # Plot 1: Adjacency Matrix
    # ============================================
    ax = axes[0, 0]
    adj_matrix = np.zeros((num_pos, num_pos))
    for i in range(num_pos):
        target = int(jump_dest[i]) % num_pos
        adj_matrix[i, target] = 1

    ax.imshow(adj_matrix, cmap='Blues', aspect='auto')
    ax.set_xlabel("Target Position")
    ax.set_ylabel("Source Position")
    ax.set_title(f"Jump Destination Adjacency Matrix ({num_pos}x{num_pos})")
    ax.grid(False)

    # Highlight self-loops
    for pos in self_loops:
        ax.plot(pos, pos, 'ro', markersize=8, label='Self-loop' if pos == self_loops[0] else "")

    if self_loops:
        ax.legend()

    # ============================================
    # Plot 2: Jump Distance Histogram
    # ============================================
    ax = axes[0, 1]
    jump_distances = []
    for i in range(num_pos):
        target = int(jump_dest[i]) % num_pos
        # Circular distance
        dist = min(abs(target - i), num_pos - abs(target - i))
        jump_distances.append(dist)

    ax.hist(jump_distances, bins=np.arange(0, num_pos//2 + 2) - 0.5, edgecolor='black')
    ax.set_xlabel("Jump Distance (circular)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Jump Distances (avg={np.mean(jump_distances):.1f})")
    ax.grid(True, alpha=0.3)

    # ============================================
    # Plot 3: Circular Routing Visualization
    # ============================================
    ax = axes[1, 0]
    ax.set_aspect('equal')

    # Draw circle for positions
    theta = np.linspace(0, 2*np.pi, num_pos, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Draw positions
    ax.plot(x, y, 'o', color='lightgray', markersize=4, zorder=1)

    # Draw some jump arrows (sample to avoid clutter)
    sample_indices = np.linspace(0, num_pos-1, min(20, num_pos), dtype=int)
    for i in sample_indices:
        target = int(jump_dest[i]) % num_pos
        if i != target:  # Skip self-loops for clarity
            ax.annotate('', xy=(x[target], y[target]), xytext=(x[i], y[i]),
                       arrowprops=dict(arrowstyle='->', lw=0.5, color='blue', alpha=0.3))

    # Highlight self-loops
    for pos in self_loops:
        ax.plot(x[pos], y[pos], 'ro', markersize=10, zorder=3, label='Refinement Station' if pos == self_loops[0] else "")

    # Highlight cycles
    colors = ['green', 'orange', 'purple', 'cyan']
    for cycle_idx, cycle in enumerate(cycles[:4]):  # Max 4 cycles
        cycle_x = [x[p] for p in cycle]
        cycle_y = [y[p] for p in cycle]
        ax.plot(cycle_x, cycle_y, 'o-', color=colors[cycle_idx % len(colors)],
               markersize=8, lw=2, zorder=2, label=f'Cycle {cycle_idx+1} (len={len(cycle)})')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(f"Circular Routing Visualization (sample of {len(sample_indices)} jumps)")
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    # ============================================
    # Plot 4: Statistics
    # ============================================
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    EMERGENT ROUTING STATISTICS
    ===========================

    Total Positions: {num_pos}

    Self-Loops (Refinement Stations): {len(self_loops)}
      Positions: {self_loops[:10]}{'...' if len(self_loops) > 10 else ''}

    Detected Cycles: {len(cycles)}
      """

    for cycle_idx, cycle in enumerate(cycles[:5]):
        cycle_str = ' -> '.join(map(str, cycle)) + ' -> ' + str(cycle[0])
        stats_text += f"\n      Cycle {cycle_idx+1}: {cycle_str}"

    if len(cycles) > 5:
        stats_text += f"\n      ... and {len(cycles)-5} more"

    stats_text += f"""

    Jump Distance Stats:
      Mean: {np.mean(jump_distances):.2f}
      Median: {np.median(jump_distances):.2f}
      Max: {np.max(jump_distances)}

    Routing Patterns:
      Local jumps (<5): {sum(1 for d in jump_distances if d < 5)} ({sum(1 for d in jump_distances if d < 5)/num_pos*100:.1f}%)
      Medium jumps (5-15): {sum(1 for d in jump_distances if 5 <= d < 15)} ({sum(1 for d in jump_distances if 5 <= d < 15)/num_pos*100:.1f}%)
      Long jumps (>15): {sum(1 for d in jump_distances if d >= 15)} ({sum(1 for d in jump_distances if d >= 15)/num_pos*100:.1f}%)
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved routing visualization to: {save_path}")
    plt.close()


def print_routing_summary(model: RingMemoryModel) -> None:
    """
    Print summary of emergent routing patterns.

    Args:
        model: Trained RingMemoryModel
    """
    jump_dest = model.jump_destinations.detach().cpu().numpy()
    num_pos = len(jump_dest)

    cycles, self_loops = detect_loops(model.jump_destinations)

    print("="*70)
    print("EMERGENT ROUTING SUMMARY")
    print("="*70)
    print()

    print(f"Total positions: {num_pos}")
    print(f"Refinement stations (self-loops): {len(self_loops)}")

    if self_loops:
        print(f"  Positions: {self_loops}")

    print()
    print(f"Detected cycles: {len(cycles)}")

    for cycle_idx, cycle in enumerate(cycles[:10]):
        cycle_str = ' -> '.join(map(str, cycle)) + ' -> ' + str(cycle[0])
        print(f"  Cycle {cycle_idx+1} (len={len(cycle)}): {cycle_str}")

    if len(cycles) > 10:
        print(f"  ... and {len(cycles)-10} more cycles")

    print()

    # Jump distance analysis
    jump_distances = []
    for i in range(num_pos):
        target = int(jump_dest[i]) % num_pos
        dist = min(abs(target - i), num_pos - abs(target - i))
        jump_distances.append(dist)

    print(f"Jump distances:")
    print(f"  Mean: {np.mean(jump_distances):.2f}")
    print(f"  Median: {np.median(jump_distances):.2f}")
    print(f"  Max: {np.max(jump_distances)}")

    print()
    print("="*70)


if __name__ == "__main__":
    # Example: Load and visualize a trained model
    print("Example: Training a model and visualizing emergent routing...")
    print()

    # Train a simple model
    torch.manual_seed(42)
    x = torch.randint(0, 10, (100, 16, 1)).float()
    y = x[:, -1, 0].long()

    model = RingMemoryModel(
        input_size=1,
        num_outputs=10,
        num_memory_positions=64,
        embedding_dim=256,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training for 1000 steps...")
    for step in range(1000):
        optimizer.zero_grad()
        logits, aux_loss, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc*100:.1f}%")

    print()
    print_routing_summary(model)
    print()
    visualize_routing_graph(model, "routing_graph.png")
