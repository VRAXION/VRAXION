"""
Exact copy of RingMemoryModel forward pass.

Test if I can reproduce the 49% failure by using the EXACT code.
"""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

print("="*70)
print("EXACT RING MEMORY MODEL TEST")
print("="*70)
print()

# Data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()


def train_and_test(model, name, steps=500):
    """Train model and return best accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_acc = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        logits, aux_loss, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        if acc > max_acc:
            max_acc = acc

        if step % 100 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

    result = "PASS" if max_acc > 0.90 else "FAIL"
    print(f"  Best: {max_acc*100:.1f}% [{result}]")
    return max_acc


# ==============================================================================
# TEST 1: Original RingMemoryModel (256-dim)
# ==============================================================================
print("[TEST 1] RingMemoryModel (256-dim, 256 positions)")
print("-"*70)

torch.manual_seed(42)
model = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=256,
    embedding_dim=256,
)

train_and_test(model, "RingMemory-256")
print()


# ==============================================================================
# TEST 2: Disable pointer interpolation (use full-size jump_targets)
# ==============================================================================
print("[TEST 2] What if jump_targets is NOT downsampled?")
print("-"*70)

torch.manual_seed(42)
model2 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,  # Small so no downsampling
    embedding_dim=256,
)

# Force no downsampling
model2.pointer_downsample = 1
model2.jump_targets = torch.nn.Parameter(torch.rand(64) * 64)
model2.jump_bias = torch.nn.Parameter(torch.zeros(64))

train_and_test(model2, "No-Downsample")
print()


# ==============================================================================
# TEST 3: Disable deadzone
# ==============================================================================
print("[TEST 3] What if deadzone threshold = 0 (no deadzone)?")
print("-"*70)

torch.manual_seed(42)
model3 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=256,
    embedding_dim=256,
    movement_threshold=0.0,  # Disable deadzone
)

train_and_test(model3, "No-Deadzone")
print()


# ==============================================================================
# TEST 4: Disable inertia
# ==============================================================================
print("[TEST 4] What if inertia = 0 (no momentum)?")
print("-"*70)

torch.manual_seed(42)
model4 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=256,
    embedding_dim=256,
    pointer_momentum=0.0,  # Disable inertia
)

train_and_test(model4, "No-Inertia")
print()


# ==============================================================================
# TEST 5: Simplest possible config
# ==============================================================================
print("[TEST 5] Simplest config (no inertia, no deadzone, small radius)")
print("-"*70)

torch.manual_seed(42)
model5 = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=256,
    attention_radius=2,
    attention_temperature=8.0,
    pointer_momentum=0.0,  # No inertia
    movement_threshold=0.0,  # No deadzone
    activation="tanh",
)

train_and_test(model5, "Simplest")
print()


print("="*70)
print("SUMMARY")
print("="*70)
print("Which config breaks?")
print("="*70)
