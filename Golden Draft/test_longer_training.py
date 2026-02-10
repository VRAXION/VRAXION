"""Test with MORE steps - maybe 10 isn't enough?"""

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
import numpy as np
torch.set_num_threads(20)

torch.manual_seed(42)
np.random.seed(42)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("LONGER TRAINING TEST - 100 steps instead of 10")
print("="*70)
print()

x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
batch_x, batch_y = next(iter(loader))
batch_x = batch_x.to(torch.float64)
batch_y = batch_y.long()

torch.manual_seed(42)

model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)
model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

losses = []
accs = []

for step in range(100):
    optimizer.zero_grad()
    logits, aux_loss = model(batch_x)
    loss = criterion(logits, batch_y) + aux_loss
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == batch_y).float().mean().item()

    losses.append(loss.item())
    accs.append(acc)

    if step % 10 == 0:
        print(f"Step {step:3d}: loss={loss.item():.2f}, acc={acc*100:.1f}%")

print()
print("="*70)
print("RESULT:")
print("="*70)
print(f"Init (step 0):   loss={losses[0]:.2f}, acc={accs[0]*100:.1f}%")
print(f"Final (step 99): loss={losses[-1]:.2f}, acc={accs[-1]*100:.1f}%")
print(f"Best accuracy: {max(accs)*100:.1f}% (step {accs.index(max(accs))})")
print()

if max(accs) > 0.7:
    print("Model CAN learn with enough steps!")
elif max(accs) > 0.6:
    print("Slight improvement, but still struggling")
else:
    print("Model CANNOT learn even with 100 steps")

print("="*70)
