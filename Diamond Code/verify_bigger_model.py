"""Test RingMemoryModel with bigger capacity."""

import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

print("="*70)
print("RING MEMORY MODEL - BIGGER CAPACITY TEST")
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

# Test different model sizes
configs = [
    ("Small (64-dim)", 64, 64),
    ("Medium (128-dim)", 128, 128),
    ("Large (256-dim)", 256, 256),
]

for name, num_positions, embed_dim in configs:
    print(f"[{name}]")
    print("-"*70)

    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=10,
        num_memory_positions=num_positions,
        embedding_dim=embed_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_acc = 0.0
    max_acc_step = 0

    for step in range(500):
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

        if step % 100 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:5.1f}%")

    print(f"  Best: {max_acc*100:.1f}% at step {max_acc_step}")
    print(f"  Result: {'PASS' if max_acc > 0.90 else 'FAIL'}")
    print()

print("="*70)
print("CONCLUSION")
print("="*70)
print("Does Ring Memory work with adequate model capacity?")
print("="*70)
