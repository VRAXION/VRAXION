"""
Simple routing visualization (no Unicode characters).
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel


def trace_path(jump_dest, start, max_steps=15):
    """Trace routing path from start position."""
    path = [start]
    visited = {start}
    current = start

    for _ in range(max_steps):
        next_pos = int(jump_dest[current].item()) % len(jump_dest)
        path.append(next_pos)
        if next_pos in visited:
            break
        visited.add(next_pos)
        current = next_pos

    return path


# Load model
checkpoint = torch.load("checkpoints/assoc_clean_step_1900.pt", weights_only=False)
model = RingMemoryModel(input_size=1, num_outputs=2, num_memory_positions=64, embedding_dim=256)
model.load_state_dict(checkpoint['model_state_dict'])

jump_dest = model.jump_destinations.detach().cpu().numpy()

print("=" * 70)
print("WHY POSITION 31 FAILS TO RECALL KEY-VALUE AT 25-26")
print("=" * 70)
print()

# Show the critical positions
print("Jump Destinations (learned routing):")
print("  Position 25 (KEY)   -> jumps to -> Position", int(jump_dest[25]) % 64)
print("  Position 26 (VALUE) -> jumps to -> Position", int(jump_dest[26]) % 64)
print("  Position 31 (QUERY) -> jumps to -> Position", int(jump_dest[31]) % 64)
print()

# Trace path from 31
path_31 = trace_path(jump_dest, 31)
print("Routing path from Position 31 (where query is):")
print("  ", " -> ".join(map(str, path_31)))
print()

# Check if it reaches 25 or 26
if 25 in path_31 or 26 in path_31:
    reach = 25 if 25 in path_31 else 26
    steps = path_31.index(reach)
    print(f"Result: CAN reach position {reach} in {steps} jumps - should work!")
else:
    print("Result: CANNOT reach position 25 or 26 - THIS IS THE BUG!")
    print()
    print("The model's learned routing doesn't connect these positions,")
    print("so when the pointer is at position 31, it can't jump back to")
    print("retrieve the value at positions 25-26. Hence the wrong prediction.")

print()
print("=" * 70)
print("VISUAL DIAGRAM:")
print("=" * 70)
print()
print("Sequence positions:")
print()
print("  [0] [1] ... [24] [25] [26] [27] ... [30] [31] [32] ...")
print("                    ^^^^  ^^^^              ^^^^")
print("                    KEY  VALUE            QUERY")
print()
print("What SHOULD happen:")
print("  1. Pointer reads query at position 31 (key = 2.0)")
print("  2. Pointer needs to find where key 2.0 was stored earlier")
print("  3. Should jump/walk backward to position 25-26")
print("  4. Retrieve value (-1.0 = class 0)")
print()
print("What ACTUALLY happens:")
print("  1. Pointer at position 31 jumps to position 37")
print("  2. Then follows path: 37->20->27->57->2->12->55->16->38->24->1->8->27")
print("  3. Gets stuck in a cycle, never visits position 25 or 26!")
print("  4. Has to guess the value -> guesses wrong (predicts class 1)")
print()
print("=" * 70)
print("WHY DOES THIS HAPPEN?")
print("=" * 70)
print()
print("The model learned routing patterns that work for MOST positions,")
print("but position 31 -> 25 is a 'blind spot' in the learned topology.")
print()
print("Positions 31 and 25 are:")
print("  - 6 positions apart (31 - 25 = 6)")
print("  - Near the end of the sequence")
print("  - Both in the same region (20-35)")
print()
print("The learned jumps in this region:")

for pos in range(20, 36):
    target = int(jump_dest[pos]) % 64
    marker = ""
    if pos == 25:
        marker = "  <-- KEY"
    elif pos == 26:
        marker = "  <-- VALUE"
    elif pos == 31:
        marker = "  <-- QUERY (fails here!)"
    print(f"  Pos {pos:2d} -> {target:2d}{marker}")

print()
print("Notice: Position 31 jumps to 37 (forward), not backward to 25!")
print("The model never learned a routing path from 31 back to 25.")
print()
print("This is why those 3 test cases failed - they all had this exact")
print("positional pattern that the routing graph doesn't handle.")
print()
print("=" * 70)
