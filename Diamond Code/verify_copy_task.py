"""Verify: Model learns COPY task >90% in <100 steps."""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

print("="*70)
print("DIAMOND CODE - COPY Task Verification")
print("="*70)
print()
print("Task: Predict last token from sequence")
print("Target: >90% accuracy in <100 steps")
print("="*70)
print()

# Generate COPY task data
torch.manual_seed(42)
x = torch.randint(0, 10, (100, 16, 1)).float()
y = x[:, -1, 0].long()  # Predict last token

print(f"Data shape: {x.shape}")
print(f"Labels shape: {y.shape}")
print(f"Num classes: 10")
print()

# Create model
model = RingMemoryModel(
    input_size=1,
    num_outputs=10,
    num_memory_positions=64,
    embedding_dim=64,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...")
print()

max_acc = 0.0
max_acc_step = 0

for step in range(100):
    optimizer.zero_grad()
    logits, aux_loss, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean().item()

    if acc > max_acc:
        max_acc = acc
        max_acc_step = step

    if step % 10 == 0:
        print(f"Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

print()
print("="*70)
print("RESULT:")
print("="*70)
print(f"Best accuracy: {max_acc*100:.1f}% (at step {max_acc_step})")
print(f"Target:        >90.0%")
print()

if max_acc > 0.90:
    print("[PASS] Model learns COPY task >90% in <100 steps!")
    print("="*70)
    exit(0)
else:
    print(f"[FAIL] Model only reached {max_acc*100:.1f}%, target was >90%")
    print("="*70)
    exit(1)
