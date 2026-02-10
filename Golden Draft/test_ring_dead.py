"""Test if the ring (memory buffer) is learning diverse states."""

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
print("RING STATE TEST - Is the ring learning?")
print("="*70)
print()
print("Question: Do different ring positions store different information?")
print()
print("Prediction:")
print("  - DEAD ring:  All positions ~same value (variance < 0.01)")
print("  - LIVE ring:  Positions differ (variance > 0.1)")
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
    dataset_name="ring_dead_test",
    model_name="test_ring_dead"
)

print("\nCapturing ring states during inference...")

# Create test samples
test_x, test_y, _ = task_copy(n_samples=10, seq_len=16, vocab_size=2)

model.eval()

# Hook to capture ring state during forward pass
ring_states = []

# The ring is part of the forward pass state
# We need to hook into the model to capture it
# Let me check what ring-related attributes exist after forward

print("\nChecking model for ring-related attributes...")
print()

# First, do a forward pass to populate any ring state
with torch.no_grad():
    _ = model(test_x[0:1].to(torch.float64))

# Check for ring attributes
ring_attrs = []
for attr in dir(model):
    if 'ring' in attr.lower() and not attr.startswith('_') and not callable(getattr(model, attr, None)):
        val = getattr(model, attr, None)
        if val is not None:
            print(f"Found: model.{attr}")
            if torch.is_tensor(val):
                print(f"  Type: Tensor, shape: {tuple(val.shape)}")
                ring_attrs.append((attr, val))
            else:
                print(f"  Type: {type(val).__name__}")

if not ring_attrs:
    print("\n[!!] No ring tensors found in model attributes")
    print("     Ring might be internal to forward pass only")
    print()
    print("Checking for slot/state buffers instead...")

    for attr in ['ring_states', 'slot_states', 'h', 'hidden', 'state_ring']:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if torch.is_tensor(val):
                print(f"Found: model.{attr}, shape: {tuple(val.shape)}")
                ring_attrs.append((attr, val))

if not ring_attrs:
    print("\n[!!] ERROR: Cannot find ring state buffer")
    print()
    print("Ring is likely computed dynamically during forward() and not stored")
    print("This requires modifying the model code to expose internal states")
    print()
    print("WORKAROUND: Test theta_ptr_reduced (learned pointer parameters)")
    print()

    # Check theta_ptr variance as proxy
    if hasattr(model, 'theta_ptr_reduced'):
        theta_ptr = model.theta_ptr_reduced.data
        print(f"theta_ptr_reduced shape: {theta_ptr.shape}")
        print(f"theta_ptr values:")
        print(f"  Min:     {theta_ptr.min().item():.4f}")
        print(f"  Max:     {theta_ptr.max().item():.4f}")
        print(f"  Mean:    {theta_ptr.mean().item():.4f}")
        print(f"  Std dev: {theta_ptr.std().item():.4f}")
        print(f"  Variance:{theta_ptr.var().item():.6f}")
        print()

        # Plot distribution
        theta_np = theta_ptr.cpu().numpy()
        print("Distribution (10 bins):")
        hist, bins = np.histogram(theta_np, bins=10)
        max_count = hist.max()
        for i in range(10):
            bar = '#' * max(1, int(40 * hist[i] / max_count))
            print(f"  {bins[i]:6.2f}-{bins[i+1]:6.2f}: {bar:<40} ({hist[i]:2d})")
        print()

        # Verdict on theta_ptr
        theta_var = theta_ptr.var().item()
        if theta_var < 0.01:
            print("[!!] theta_ptr has VERY LOW variance!")
            print("     Pointer parameters are nearly identical")
            print("     This could prevent diverse pointer movement")
        elif theta_var < 1.0:
            print("[~] theta_ptr has MODERATE variance")
            print(f"    Std dev: {theta_ptr.std().item():.4f}")
        else:
            print("[OK] theta_ptr has GOOD variance")
            print("     Pointer parameters are diverse")

    # Check theta_gate as well
    print()
    if hasattr(model, 'theta_gate_reduced'):
        theta_gate = model.theta_gate_reduced.data
        print(f"theta_gate_reduced shape: {theta_gate.shape}")
        print(f"theta_gate variance: {theta_gate.var().item():.6f}")
        print(f"theta_gate std dev:  {theta_gate.std().item():.4f}")

    sys.exit(0)

# Analyze ring states
print()
print("="*70)
print("RING STATE ANALYSIS")
print("="*70)

for attr_name, ring_tensor in ring_attrs:
    print(f"\nAnalyzing: {attr_name}")
    print(f"Shape: {tuple(ring_tensor.shape)}")

    # Calculate variance
    ring_var = ring_tensor.var().item()
    ring_std = ring_tensor.std().item()
    ring_mean = ring_tensor.mean().item()

    print(f"  Mean:     {ring_mean:.6f}")
    print(f"  Std dev:  {ring_std:.6f}")
    print(f"  Variance: {ring_var:.6f}")
    print(f"  Range:    [{ring_tensor.min().item():.4f}, {ring_tensor.max().item():.4f}]")

    # Verdict
    print()
    if ring_var < 0.01:
        print(f"  [!!] VERY LOW VARIANCE - Ring appears DEAD")
        print(f"       All positions contain nearly identical values")
    elif ring_var < 0.1:
        print(f"  [~] LOW VARIANCE - Ring barely learning")
    else:
        print(f"  [OK] GOOD VARIANCE - Ring is learning")

print()
print("="*70)
