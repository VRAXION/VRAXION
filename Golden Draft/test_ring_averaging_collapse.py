"""Test if ring averaging (fused = gathered.mean) collapses variance."""

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
print("RING AVERAGING COLLAPSE TEST")
print("="*70)
print()
print("Question: Does ring averaging collapse variance?")
print()
print("Findings so far:")
print("  - input_proj has variance = 0.0086 (LOW but not collapsed)")
print("  - Final hidden has variance = 0.00004 (COLLAPSED)")
print("  - 200x drop happens AFTER input_proj")
print()
print("Hypothesis:")
print("  Ring averaging: fused = gathered.mean(dim=1)")
print("  Averages multiple ring positions into single vector")
print("  -> Could be collapsing all diversity")
print()
print("Test Strategy:")
print("  Compare variance of:")
print("  1. Raw input tokens (should be high)")
print("  2. Input after one forward pass (should show collapse)")
print("  3. Measure variance REDUCTION ratio")
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
    dataset_name="ring_averaging_test",
    model_name="test_ring_averaging"
)

print("\nAnalyzing variance collapse through forward pass...")
print()

# Create diverse test samples
print("Creating test samples...")
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

# Measure input variance
print("Input token variance:")
input_variance = test_x.var().item()
input_std = test_x.std().item()
unique_vals = len(torch.unique(test_x))
print(f"  Variance: {input_variance:.6f}")
print(f"  Std dev:  {input_std:.6f}")
print(f"  Unique values: {unique_vals}")
print()

# Now measure output variance after forward pass
model.eval()

# Hook to capture the final hidden state before output head
hidden_states = []

def capture_head_input(module, input, output):
    # input[0] is the hidden state passed to the head
    hidden_states.append(input[0].detach().cpu())

hook = model.head.register_forward_hook(capture_head_input)

with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)
        _ = model(sample)

hook.remove()

# Analyze hidden state variance
if hidden_states:
    all_hidden = torch.cat(hidden_states, dim=0)  # [100, 64]

    # Per-sample variance (variance across dimensions for each sample)
    per_sample_var = all_hidden.var(dim=1)
    print("Hidden state variance (per-sample):")
    print(f"  Mean:  {per_sample_var.mean().item():.8f}")
    print(f"  Std:   {per_sample_var.std().item():.8f}")
    print(f"  Min:   {per_sample_var.min().item():.8f}")
    print(f"  Max:   {per_sample_var.max().item():.8f}")
    print()

    # Cross-sample variance (variance across samples for each dimension)
    cross_sample_var = all_hidden.var(dim=0)
    print("Hidden state variance (cross-sample, per-dimension):")
    print(f"  Mean:  {cross_sample_var.mean().item():.8f}")
    print(f"  Std:   {cross_sample_var.std().item():.8f}")
    print(f"  Min:   {cross_sample_var.min().item():.8f}")
    print(f"  Max:   {cross_sample_var.max().item():.8f}")
    print()

    # Overall statistics
    print("Overall hidden state statistics:")
    print(f"  Shape: {all_hidden.shape}")
    print(f"  Mean:  {all_hidden.mean().item():.6f}")
    print(f"  Std:   {all_hidden.std().item():.6f}")
    print(f"  Range: [{all_hidden.min().item():.4f}, {all_hidden.max().item():.4f}]")
    print()

    # Check if all samples have nearly identical hidden states
    # If variance across samples is very low, they've collapsed to same state
    cross_sample_mean_var = cross_sample_var.mean().item()

    print("="*70)
    print("VARIANCE REDUCTION ANALYSIS")
    print("="*70)
    print()

    print(f"Input variance:         {input_variance:.8f}")
    print(f"Hidden variance:        {cross_sample_mean_var:.8f}")
    print(f"Reduction ratio:        {cross_sample_mean_var / (input_variance + 1e-12):.6f}x")
    print(f"Reduction magnitude:    {input_variance / (cross_sample_mean_var + 1e-12):.1f}x DROP")
    print()

    # Check per-sample variance (within each sample)
    per_sample_mean_var = per_sample_var.mean().item()
    print(f"Per-sample variance:    {per_sample_mean_var:.8f}")
    print()

    print("="*70)
    print("DIAGNOSIS")
    print("="*70)
    print()

    if cross_sample_mean_var < 0.0001:
        print("[!!] EXTREME CROSS-SAMPLE COLLAPSE")
        print()
        print("All samples produce nearly IDENTICAL hidden states!")
        print()
        print("This means:")
        print("  - Different inputs -> Same internal representation")
        print("  - Model can't distinguish between samples")
        print()
        print("Why binary works (58-60%) but 10-class fails (10%):")
        print("  - Binary: Even with identical states, random 50/50 split")
        print("  - 10-class: Need to distinguish 10 classes with collapsed states")
        print()
        print("ROOT CAUSE CANDIDATES:")
        print("  1. Ring averaging collapses all ring positions to same value")
        print("  2. Ring state itself is collapsed (all positions identical)")
        print("  3. Recurrent h update saturates to a fixed point")
        print()

    if per_sample_mean_var < 0.001:
        print("[!!] EXTREME PER-SAMPLE COLLAPSE")
        print()
        print("Each hidden state has all 64 dimensions nearly identical!")
        print()
        print("This means:")
        print("  - Hidden = [x, x, x, ..., x] (all same value)")
        print("  - No diversity within each sample")
        print()
        print("ROOT CAUSE:")
        print("  - Ring averaging: fused = gathered.mean(dim=1)")
        print("  - If all gathered ring slots have same value,")
        print("    averaging them produces the same value")
        print()

    # Check if the problem is ring state collapse
    print()
    print("NEXT DIAGNOSTIC:")
    print("  Check if ring state itself is collapsed")
    print("  (all 64 ring positions have identical values)")
    print()
    print("="*70)

else:
    print("[!!] ERROR: No hidden states captured")
    print("     Hook might have failed")

print()
