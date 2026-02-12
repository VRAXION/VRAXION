"""Test pointer stuck - version 3 using last_ptr_int."""

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
os.environ['VRX_HEARTBEAT_STEPS'] = '1000'

import torch
import numpy as np
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader
from tools.instnct_train_steps import train_steps

print("="*70)
print("POINTER STUCK TEST v3 - Using last_ptr_int")
print("="*70)
print()

# Train binary model
print("Training binary model (100 steps)...")
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)

train_steps(
    model=model,
    loader=train_loader,
    steps=100,
    dataset_name="pointer_stuck_test_v3",
    model_name="test_pointer_stuck_v3"
)

print("\nCollecting pointer positions from last_ptr_int...")

# Create test set
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

model.eval()
pointer_positions = []

with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)
        _ = model(sample)

        # Read last_ptr_int after forward
        if hasattr(model, 'last_ptr_int') and model.last_ptr_int is not None:
            ptr_val = int(model.last_ptr_int.item())
            pointer_positions.append(ptr_val)

if not pointer_positions:
    print("\n[!!] ERROR: last_ptr_int is None or not set")
    sys.exit(1)

# Analyze distribution
pointer_positions = np.array(pointer_positions)
unique_positions = len(np.unique(pointer_positions))
most_common = np.bincount(pointer_positions, minlength=64).argmax()
most_common_count = np.bincount(pointer_positions, minlength=64).max()

# Calculate entropy
counts = np.bincount(pointer_positions, minlength=64)
probs = counts / counts.sum()
entropy = -np.sum(probs * np.log(probs + 1e-12))
max_entropy = np.log(64)
entropy_ratio = entropy / max_entropy

print()
print("="*70)
print("POINTER POSITION ANALYSIS")
print("="*70)
print()
print(f"Samples analyzed:    {len(pointer_positions)}")
print(f"Ring size:           0-63 (64 positions)")
print()
print(f"Unique positions:    {unique_positions} / 64 ({100*unique_positions/64:.1f}%)")
print(f"Most common:         Position {most_common} ({most_common_count} samples, {100*most_common_count/len(pointer_positions):.1f}%)")
print(f"Min position:        {pointer_positions.min()}")
print(f"Max position:        {pointer_positions.max()}")
print(f"Mean position:       {pointer_positions.mean():.1f}")
print(f"Std dev:             {pointer_positions.std():.2f}")
print()
print(f"Entropy:             {entropy:.2f} / {max_entropy:.2f} ({100*entropy_ratio:.1f}% of max)")
print()

# Histogram
print("Distribution (10 bins):")
hist, bins = np.histogram(pointer_positions, bins=10, range=(0, 64))
max_count = hist.max()
for i in range(10):
    bin_start = int(bins[i])
    bin_end = int(bins[i+1])
    bar = '#' * max(1, int(40 * hist[i] / max_count))
    print(f"  {bin_start:2d}-{bin_end:2d}: {bar:<40} ({hist[i]:2d})")
print()

# Detailed position counts (top 10)
top_positions = np.argsort(counts)[::-1][:10]
print("Top 10 most common positions:")
for pos in top_positions:
    if counts[pos] > 0:
        print(f"  Position {pos:2d}: {counts[pos]:2d} samples ({100*counts[pos]/len(pointer_positions):5.1f}%)")
print()

# Verdict
print("="*70)
print("VERDICT:")
print("="*70)
print()

if unique_positions <= 5:
    print("[!!] POINTER IS STUCK!")
    print(f"     Only {unique_positions} unique positions out of 64")
    print(f"     {100*most_common_count/len(pointer_positions):.1f}% of samples point to position {most_common}")
    print()
    print("ROOT CAUSE: Pointer mechanism is NOT learning to diversify")
    print("            All samples collapse to ~same pointer position")

elif unique_positions < 20:
    print("[~] POINTER IS PARTIALLY STUCK")
    print(f"     Only {unique_positions} positions used (expect ~64 for full diversity)")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum")
    print()
    print("Pointer is moving but collapsing to a few favorite positions")

elif entropy_ratio < 0.5:
    print("[~] POINTER IS BIASED")
    print(f"     {unique_positions} positions used, but distribution is uneven")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum (expect ~90%+)")
    print()
    print("Pointer can reach many positions but has strong preferences")

else:
    print("[OK] POINTER IS WORKING!")
    print(f"     {unique_positions} unique positions (good diversity)")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum")
    print()
    print("Pointer mechanism is learning and diversifying across samples")
    print("Internal collapse problem must be elsewhere (hidden computation, etc.)")

print()
print("="*70)
