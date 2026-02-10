"""Test pointer stuck - version 2 with direct state capture."""

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
print("POINTER STUCK TEST v2")
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
    dataset_name="pointer_stuck_test_v2",
    model_name="test_pointer_stuck_v2"
)

print("\nCollecting pointer positions via hook...")

# Create test set
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

model.eval()

# Hook to capture pointer during forward pass
pointer_positions = []

# Monkey-patch the model to save pointer after each forward
original_forward = model.forward

def forward_with_capture(*args, **kwargs):
    result = original_forward(*args, **kwargs)

    # After forward, check for pointer in various possible locations
    if hasattr(model, 'ptr_int') and model.ptr_int is not None:
        ptr = model.ptr_int
    elif hasattr(model, 'ptr_float') and model.ptr_float is not None:
        ptr = model.ptr_float
    elif hasattr(model, 'theta_ptr') and model.theta_ptr is not None:
        ptr = model.theta_ptr
    else:
        # Try to find it in the last layer's state
        ptr = None

    if ptr is not None:
        if torch.is_tensor(ptr):
            if ptr.numel() == 1:
                pointer_positions.append(int(round(ptr.item())) % 64)
            else:
                # Batch of pointers, take last one
                pointer_positions.append(int(round(ptr[-1].item())) % 64)

    return result

model.forward = forward_with_capture

# Run inference
with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)
        _ = model(sample)

# Restore original forward
model.forward = original_forward

if not pointer_positions:
    print("\n[!!] ERROR: Still couldn't capture pointers via monkey-patch")
    print()
    print("Fallback: Analyzing model attributes...")

    # Check what pointer-related attributes exist
    print("\nModel attributes containing 'ptr':")
    for attr in dir(model):
        if 'ptr' in attr.lower():
            val = getattr(model, attr, None)
            if val is not None and not callable(val):
                print(f"  {attr}: {type(val).__name__}", end="")
                if torch.is_tensor(val):
                    print(f" shape={tuple(val.shape)}")
                else:
                    print()

    print("\nModel attributes containing 'theta':")
    for attr in dir(model):
        if 'theta' in attr.lower():
            val = getattr(model, attr, None)
            if val is not None and not callable(val):
                print(f"  {attr}: {type(val).__name__}", end="")
                if torch.is_tensor(val):
                    print(f" shape={tuple(val.shape)}")
                else:
                    print()

    print()
    print("WORKAROUND: Use AC (anchor clicks) from training logs as proxy")
    print("AC values during training showed variation (7-15), suggesting pointer")
    print("IS moving during training. But we can't capture it in eval mode.")
    print()
    print("CONCLUSION:")
    print("  - Model has pointer mechanism (AC varies during training)")
    print("  - Pointer state not exposed after forward() in eval mode")
    print("  - Need to modify model code to expose internal state")

    sys.exit(1)

# Analyze distribution
pointer_positions = np.array(pointer_positions)
unique_positions = len(np.unique(pointer_positions))
most_common = np.bincount(pointer_positions).argmax()
most_common_count = np.bincount(pointer_positions).max()
entropy = -np.sum((np.bincount(pointer_positions, minlength=64) / len(pointer_positions)) *
                  np.log(np.bincount(pointer_positions, minlength=64) / len(pointer_positions) + 1e-12))
max_entropy = np.log(64)
entropy_ratio = entropy / max_entropy

print()
print("="*70)
print("POINTER POSITION ANALYSIS")
print("="*70)
print()
print(f"Samples analyzed:    {len(pointer_positions)}")
print(f"Unique positions:    {unique_positions} / 64")
print(f"Most common:         Position {most_common} ({most_common_count} samples, {100*most_common_count/len(pointer_positions):.1f}%)")
print(f"Std dev:             {pointer_positions.std():.2f}")
print(f"Entropy:             {100*entropy_ratio:.1f}% of max")
print()

# Distribution
print("Distribution (10 bins):")
hist, bins = np.histogram(pointer_positions, bins=10, range=(0, 64))
for i in range(10):
    bin_start = int(bins[i])
    bin_end = int(bins[i+1])
    bar = '#' * max(1, int(hist[i] / 2))
    print(f"  {bin_start:2d}-{bin_end:2d}: {bar} ({hist[i]})")
print()

# Verdict
if unique_positions <= 5:
    print("[!!] POINTER IS STUCK!")
elif unique_positions < 20:
    print("[~] POINTER IS PARTIALLY STUCK")
elif entropy_ratio < 0.5:
    print("[~] POINTER IS BIASED")
else:
    print("[OK] POINTER IS WORKING!")

print()
print("="*70)
