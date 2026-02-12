"""
Test the new hard discrete jump mechanism.

Tests:
1. Baseline: Force walk-only (disable jumps) - should reach 98%
2. With hard jumps enabled - should reach >90%
3. Monitor gradient flow through jump_destinations
4. Visualize emergent routing patterns
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ring_memory_model import RingMemoryModel
from visualize_routing import detect_loops

print("="*70)
print("HARD DISCRETE JUMPS - Emergent Routing Test")
print("="*70)
print()

# Data
torch.manual_seed(42)
x_data = torch.randint(0, 10, (100, 16, 1)).float()
y_labels = x_data[:, -1, 0].long()

print(f"Task: COPY (predict last token)")
print(f"Data: {x_data.shape}, Labels: {y_labels.shape}, Classes: 10")
print(f"Random baseline: ~10%")
print()


# ==============================================================================
# TEST 1: Baseline - Force walk only (disable jumps)
# ==============================================================================
print("[TEST 1] Baseline: Walk-only (jumps disabled)")
print("-"*70)

class WalkOnlyModel(RingMemoryModel):
    """Force walk-only by making jump gate always output 0."""

    def _hard_gate(self, logits):
        # Force should_jump = 0 (always walk)
        return torch.zeros_like(logits)

torch.manual_seed(42)
model1 = WalkOnlyModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

max_acc1 = 0.0
for step in range(1000):
    optimizer1.zero_grad()
    logits, aux_loss, _ = model1(x_data)
    loss = torch.nn.functional.cross_entropy(logits, y_labels) + aux_loss
    loss.backward()
    optimizer1.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y_labels).float().mean().item()
    if acc > max_acc1:
        max_acc1 = acc

    if step % 200 == 0:
        print(f"  Step {step:4d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print(f"  Best: {max_acc1*100:.1f}%")
print(f"  Result: {'PASS' if max_acc1 > 0.90 else 'FAIL'} (target >90%, expect ~98%)")
print()


# ==============================================================================
# TEST 2: With hard jumps enabled
# ==============================================================================
print("[TEST 2] With hard discrete jumps (emergent routing)")
print("-"*70)

torch.manual_seed(42)
model2 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

max_acc2 = 0.0
jump_dest_changes = []  # Track if jump_destinations change

# Save initial jump destinations
initial_jump_dest = model2.jump_destinations.detach().clone()

for step in range(1000):
    optimizer2.zero_grad()
    logits, aux_loss, _ = model2(x_data)
    loss = torch.nn.functional.cross_entropy(logits, y_labels) + aux_loss
    loss.backward()

    # Check gradient on jump_destinations
    if model2.jump_destinations.grad is not None:
        grad_norm = model2.jump_destinations.grad.norm().item()
    else:
        grad_norm = 0.0

    optimizer2.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y_labels).float().mean().item()

        # Track changes in jump_destinations
        dest_change = (model2.jump_destinations - initial_jump_dest).norm().item()
        jump_dest_changes.append(dest_change)

    if acc > max_acc2:
        max_acc2 = acc

    if step % 200 == 0:
        print(f"  Step {step:4d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%, grad_norm={grad_norm:.4f}")

print(f"  Best: {max_acc2*100:.1f}%")
print(f"  Jump destinations changed: {jump_dest_changes[-1]:.4f} (0 = no change)")
print(f"  Result: {'PASS' if max_acc2 > 0.90 else 'FAIL'} (target >90%)")
print()


# ==============================================================================
# TEST 3: Analyze emergent routing patterns
# ==============================================================================
print("[TEST 3] Emergent Routing Pattern Analysis")
print("-"*70)

with torch.no_grad():
    # Detect emergent routing patterns
    cycles, self_loops = detect_loops(model2.jump_destinations)

    print(f"  Total positions: {model2.num_memory_positions}")
    print(f"  Self-loops (refinement stations): {len(self_loops)}")
    print(f"  Detected cycles: {len(cycles)}")
    print()

    # Show some examples
    if self_loops:
        print(f"  Example self-loops:")
        for pos in self_loops[:5]:
            print(f"    Position {pos} -> {pos} (stay)")
        print()

    if cycles:
        print(f"  Example cycles:")
        for cycle in cycles[:3]:
            cycle_str = ' -> '.join(map(str, cycle)) + ' -> ' + str(cycle[0])
            print(f"    {cycle_str}")

    print()


# ==============================================================================
# SUMMARY
# ==============================================================================
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Walk-only (baseline): {max_acc1*100:.1f}%")
print(f"With hard jumps:      {max_acc2*100:.1f}%")
print()

if max_acc1 > 0.90 and max_acc2 > 0.90:
    print("[SUCCESS] Both mechanisms reach >90% accuracy!")
    if max_acc2 > max_acc1:
        print(f"  Hard jumps BETTER than walk-only by {(max_acc2-max_acc1)*100:.1f}%")
    elif max_acc2 < max_acc1:
        print(f"  Walk-only better by {(max_acc1-max_acc2)*100:.1f}% (jumps may need tuning)")
    else:
        print(f"  Both perform equally")
elif max_acc1 > 0.90:
    print(f"[PARTIAL] Walk-only works but hard jumps fail ({max_acc2*100:.1f}%)")
    print("  Possible issues: STE not working, jump destinations not learning")
elif max_acc2 > 0.90:
    print(f"[INTERESTING] Hard jumps work but walk-only fails ({max_acc1*100:.1f}%)")
else:
    print(f"[FAIL] Neither mechanism reaches >90%")
    print("  Both need investigation")

print("="*70)
