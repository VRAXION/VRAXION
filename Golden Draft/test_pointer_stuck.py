"""Simple test: Is the pointer stuck at one position?"""

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
os.environ['VRX_HEARTBEAT_STEPS'] = '1000'  # Suppress logs

import torch
import numpy as np
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader
from tools.instnct_train_steps import train_steps

print("="*70)
print("POINTER STUCK TEST")
print("="*70)
print()
print("Question: Is the pointer stuck at one position, or moving?")
print()
print("Prediction:")
print("  - If STUCK:   All samples point to ~same position (e.g., all at 32)")
print("  - If MOVING:  Samples spread across ring (0-63)")
print()
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
    dataset_name="pointer_stuck_test",
    model_name="test_pointer_stuck"
)

print("\nCollecting pointer positions...")

# Create test set with diverse samples
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

model.eval()
pointer_positions = []

with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)  # Single sample [1, 16, 1]

        # Run forward pass
        output = model(sample)

        # Try to get pointer position from model
        # The pointer should be stored in model after forward pass
        if hasattr(model, 'ptr_int') and model.ptr_int is not None:
            # ptr_int is the integer pointer position
            ptr_pos = int(model.ptr_int.item())
            pointer_positions.append(ptr_pos)
        elif hasattr(model, 'ptr_float') and model.ptr_float is not None:
            # ptr_float is the float pointer position, round it
            ptr_pos = int(round(model.ptr_float.item()))
            pointer_positions.append(ptr_pos)

if not pointer_positions:
    print("\n[!!] ERROR: Could not capture pointer positions!")
    print("     Model might not be storing ptr_int/ptr_float")
    print("     This needs investigation in the model code")
    sys.exit(1)

# Analyze distribution
pointer_positions = np.array(pointer_positions)
unique_positions = len(np.unique(pointer_positions))
most_common = np.bincount(pointer_positions).argmax()
most_common_count = np.bincount(pointer_positions).max()
entropy = -np.sum((np.bincount(pointer_positions, minlength=64) / len(pointer_positions)) *
                  np.log(np.bincount(pointer_positions, minlength=64) / len(pointer_positions) + 1e-12))
max_entropy = np.log(64)  # For uniform distribution
entropy_ratio = entropy / max_entropy

print()
print("="*70)
print("POINTER POSITION ANALYSIS")
print("="*70)
print()
print(f"Samples analyzed:    {len(pointer_positions)}")
print(f"Ring size:           0-63 (64 positions)")
print()
print(f"Unique positions:    {unique_positions} / 64")
print(f"Most common:         Position {most_common} ({most_common_count} samples, {100*most_common_count/len(pointer_positions):.1f}%)")
print(f"Min position:        {pointer_positions.min()}")
print(f"Max position:        {pointer_positions.max()}")
print(f"Mean position:       {pointer_positions.mean():.1f}")
print(f"Std dev:             {pointer_positions.std():.2f}")
print()
print(f"Entropy:             {entropy:.2f} / {max_entropy:.2f} ({100*entropy_ratio:.1f}% of max)")
print()

# Histogram (simple text-based)
print("Distribution (10 bins):")
hist, bins = np.histogram(pointer_positions, bins=10, range=(0, 64))
for i in range(10):
    bin_start = int(bins[i])
    bin_end = int(bins[i+1])
    bar = '#' * int(hist[i] / 2)  # Scale down for display
    print(f"  {bin_start:2d}-{bin_end:2d}: {bar} ({hist[i]})")
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
    print("ROOT CAUSE: Pointer mechanism is NOT learning to move")
    print("            All samples collapse to same pointer position")
    print()
    print("WHY: Pointer update gradients might be:")
    print("  - Zero (not flowing)")
    print("  - Too small (vanishing)")
    print("  - Stuck in local minimum")

elif unique_positions < 20:
    print("[~] POINTER IS PARTIALLY STUCK")
    print(f"     Only {unique_positions} unique positions (expect ~64 for full diversity)")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum")
    print()
    print("Pointer is moving but collapsing to a few modes")

elif entropy_ratio < 0.5:
    print("[~] POINTER IS BIASED")
    print(f"     {unique_positions} positions used, but heavily biased")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum (expect ~90%+)")
    print()
    print("Pointer can move but prefers certain positions")

else:
    print("[OK] POINTER IS WORKING!")
    print(f"     {unique_positions} unique positions (good diversity)")
    print(f"     Entropy: {100*entropy_ratio:.1f}% of maximum")
    print()
    print("Pointer mechanism is learning and diversifying")
    print("Problem must be elsewhere (hidden state computation, output head, etc.)")

print()
print("="*70)
