"""Test 2-dim slots - is the problem scalar vs vector?"""

import os
import sys
sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

ROOT = "S:/AI/work/VRAXION_DEV/Golden Draft"
os.environ['VRX_ROOT'] = ROOT
os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'
os.environ['VRX_PRECISION'] = 'fp64'
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['VRX_PTR_INERTIA_OVERRIDE'] = '0.6'
os.environ['VRX_AGC_ENABLED'] = '0'
os.environ['VRX_LR'] = '0.001'

import torch
import torch.nn as nn
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway

print("="*70)
print("2-DIM TEST - Scalar vs Vector Threshold?")
print("="*70)
print()
print("Hypothesis:")
print("  1-dim (scalar) = broken (cannot form vector representations)")
print("  2-dim (vector) = works (minimal vector space)")
print()
print("Testing 2-dim on binary task...")
print("="*70)
print()

# Binary task
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
x = x.to(torch.float64)
y = y.long()

# 2-dim model (capacity-matched: 2 × 2048 = 4096 units like 64 × 64)
model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=2048, slot_dim=2)
model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Model: {sum(p.numel() for p in model.parameters())} params")
print(f"Capacity: 2 dims × 2048 positions = 4096 storage units")
print(f"Data: {x.shape}, labels: {y.shape}")
print()

# Train for 50 steps
losses = []
accs = []

for step in range(50):
    optimizer.zero_grad()

    # Forward
    logits, aux_loss = model(x)
    loss = criterion(logits, y) + aux_loss

    # Backward
    loss.backward()
    optimizer.step()

    # Metrics
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

    losses.append(loss.item())
    accs.append(acc)

    if step % 10 == 0:
        print(f"Step {step:3d}: loss={loss.item():.4f}, acc={acc*100:.1f}%")

print()
print("="*70)
print("RESULT:")
print("="*70)
print(f"Initial: loss={losses[0]:.4f}, acc={accs[0]*100:.1f}%")
print(f"Final:   loss={losses[-1]:.4f}, acc={accs[-1]*100:.1f}%")
print(f"Change:  loss={losses[-1]-losses[0]:+.4f}, acc={(accs[-1]-accs[0])*100:+.1f}%")
print()

print("COMPARISON:")
print("  1-dim:  loss 203→112, acc 51%→49% (BROKEN - scalar)")
print("  2-dim:  loss {:.0f}→{:.0f}, acc {:.0f}%→{:.0f}%".format(
    losses[0], losses[-1], accs[0]*100, accs[-1]*100
), end="")

if accs[-1] > 0.9:
    print(" (WORKS! - vector)")
    print()
    print("[!!] THRESHOLD FOUND: 2-dim is the minimum!")
    print("     The problem: 1-dim scalars cannot form vector space")
elif accs[-1] > 0.7:
    print(" (PARTIAL - vector helps)")
    print()
    print("[~] 2-dim helps but not enough")
elif accs[-1] > 0.55:
    print(" (BARELY - still struggling)")
    print()
    print("[~] 2-dim slightly better than 1-dim")
else:
    print(" (BROKEN - vector not enough)")
    print()
    print("[!!] 2-dim ALSO fails - problem is deeper")
    print("     Need to test 4-dim, 8-dim, 16-dim...")

print("="*70)
