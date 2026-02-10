"""Test with FIXED random seed to eliminate data variance."""

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

# FIXED SEED
torch.manual_seed(42)
np.random.seed(42)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("FIXED SEED TEST - Eliminate randomness")
print("="*70)
print()
print("Using seed=42 for reproducible results")
print("Testing 64x64 baseline - should get ~68-75% if working")
print("="*70)
print()

# Generate data with FIXED seed
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
batch_x, batch_y = next(iter(loader))
batch_x = batch_x.to(torch.float64)
batch_y = batch_y.long()

print(f"Batch: {batch_x.shape}")
print(f"Labels: {batch_y.tolist()}")
print()

# Run 5 trials with SAME data, SAME seed
print("Running 5 trials with SAME data:")
print()

for trial in range(5):
    print(f"--- Trial {trial+1} ---")

    # Reset seed for model init
    torch.manual_seed(42 + trial)

    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)
    model = model.to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []

    for step in range(10):
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

    print(f"  Init:  loss={losses[0]:.2f}, acc={accs[0]*100:.1f}%")
    print(f"  Final: loss={losses[-1]:.2f}, acc={accs[-1]*100:.1f}%")
    print(f"  Delta: {(accs[-1] - accs[0])*100:+.1f}%")
    print()

print("="*70)
print("INTERPRETATION:")
print("="*70)
print()
print("If all 5 trials get similar results (65-75%):")
print("  => Model IS learning, just high variance")
print()
print("If results vary wildly (30-70%):")
print("  => Initialization matters more than learning")
print()
print("If all stuck at 50% (+/- 10%):")
print("  => Model fundamentally cannot learn this task")
print("="*70)
