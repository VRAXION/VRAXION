"""Quick check: Can 1-dim learn AT ALL?"""

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
print("QUICK 1-DIM CHECK (without LayerNorm)")
print("="*70)
print()

# Binary task
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
x = x.to(torch.float64)
y = y.long()

# 1-dim model
model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=4096, slot_dim=1)
model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Model: {sum(p.numel() for p in model.parameters())} params")
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

if accs[-1] > 0.9:
    print("[OK] 1-dim CAN learn (90%+ accuracy)")
elif accs[-1] > 0.7:
    print("[~] 1-dim learns something (70-90%)")
elif accs[-1] > 0.55:
    print("[~] 1-dim barely learns (55-70%)")
else:
    print("[!!] 1-dim CANNOT learn (≤55% = random)")

print("="*70)
