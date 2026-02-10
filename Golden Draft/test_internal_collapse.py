"""Test if internal representations collapse to single state."""

import os
import sys
from pathlib import Path

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
print("INTERNAL COLLAPSE TEST")
print("="*70)
print()
print("Hypothesis: Internal states collapse to single averaged point")
print("Prediction:")
print("  - Binary (works):   HIGH variance (different samples, different states)")
print("  - 3-class (fails):  LOW variance (collapse to averaged state)")
print("  - 10-class (fails): LOW variance (collapse to averaged state)")
print()
print("="*70)
print()

results = []

for num_classes in [2, 3, 10]:
    print(f"\n{'='*70}")
    print(f"TESTING {num_classes}-CLASS")
    print(f"{'='*70}\n")

    # Generate data
    x, y, _ = task_copy(n_samples=1000, seq_len=16, vocab_size=num_classes)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=100, shuffle=False)  # 10 batches of 100

    # Create and train model
    print(f"Training {num_classes}-class model for 100 steps...")
    model = AbsoluteHallway(input_dim=1, num_classes=num_classes, ring_len=64, slot_dim=64)

    train_steps(
        model=model,
        loader=train_loader,
        steps=100,
        dataset_name=f"collapse_test_{num_classes}class",
        model_name=f"test_collapse_{num_classes}class"
    )

    print(f"\nCollecting internal states...")
    model.eval()

    # We need to modify model to track states
    # Add a hook to capture hidden state before output head
    hidden_states = []
    pointer_positions = []

    def capture_hook(module, input, output):
        # Input to the head is the hidden state
        hidden_states.append(input[0].detach().cpu())

    # Register hook on output head
    hook = model.head.register_forward_hook(capture_hook)

    # Run inference
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(torch.float64)
            outputs = model(batch_x)

            # Try to get pointer positions if available
            if hasattr(model, 'ptr_int') and model.ptr_int is not None:
                pointer_positions.append(model.ptr_int.detach().cpu())

    hook.remove()

    # Analyze variance
    if hidden_states:
        # Concatenate all hidden states
        all_hidden = torch.cat(hidden_states, dim=0)  # [N, 64]

        # Calculate variance across samples (for each dimension)
        hidden_var_per_dim = all_hidden.var(dim=0)  # [64]
        hidden_mean_var = hidden_var_per_dim.mean().item()
        hidden_std_var = hidden_var_per_dim.std().item()

        # Calculate total variance (average across all dimensions and samples)
        hidden_total_var = all_hidden.var().item()

        print(f"\nHIDDEN STATE VARIANCE:")
        print(f"  Mean variance per dim: {hidden_mean_var:.4f}")
        print(f"  Total variance:        {hidden_total_var:.4f}")
        print(f"  Shape: {all_hidden.shape}")
        print(f"  Range: [{all_hidden.min():.2f}, {all_hidden.max():.2f}]")
    else:
        hidden_mean_var = 0
        hidden_total_var = 0

    if pointer_positions:
        all_ptrs = torch.cat(pointer_positions, dim=0)  # [N]
        ptr_var = all_ptrs.float().var().item()
        ptr_std = all_ptrs.float().std().item()
        ptr_mean = all_ptrs.float().mean().item()
        ptr_unique = len(all_ptrs.unique())

        print(f"\nPOINTER POSITION VARIANCE:")
        print(f"  Variance:      {ptr_var:.4f}")
        print(f"  Std dev:       {ptr_std:.4f}")
        print(f"  Mean position: {ptr_mean:.2f}")
        print(f"  Unique positions: {ptr_unique} / 64")
        print(f"  Range: [{all_ptrs.min()}, {all_ptrs.max()}]")
    else:
        ptr_var = 0
        ptr_std = 0
        ptr_unique = 0

    results.append({
        'num_classes': num_classes,
        'hidden_mean_var': hidden_mean_var,
        'hidden_total_var': hidden_total_var,
        'ptr_var': ptr_var,
        'ptr_std': ptr_std,
        'ptr_unique': ptr_unique
    })

# Summary
print("\n" + "="*70)
print("SUMMARY: VARIANCE COMPARISON")
print("="*70)
print()
print(f"{'Classes':<10} {'Hidden Var':<12} {'Ptr Var':<12} {'Ptr Unique':<12} {'Status'}")
print("-"*70)

for r in results:
    nc = r['num_classes']
    h_var = r['hidden_total_var']
    p_var = r['ptr_var']
    p_uniq = r['ptr_unique']

    # Determine status
    if nc == 2:
        status = "BASELINE (works)"
    else:
        # Compare to binary
        binary_h = results[0]['hidden_total_var']
        binary_p = results[0]['ptr_var']

        h_ratio = h_var / binary_h if binary_h > 0 else 0
        p_ratio = p_var / binary_p if binary_p > 0 else 0

        if h_ratio < 0.5 or p_ratio < 0.5:
            status = "COLLAPSED!"
        else:
            status = "Similar"

    print(f"{nc:<10} {h_var:<12.4f} {p_var:<12.4f} {p_uniq:<12} {status}")

print()
print("="*70)
print("INTERPRETATION:")
print()

binary_h = results[0]['hidden_total_var']
binary_p = results[0]['ptr_var']

collapsed = False
for r in results[1:]:  # Skip binary
    h_ratio = r['hidden_total_var'] / binary_h if binary_h > 0 else 1
    p_ratio = r['ptr_var'] / binary_p if binary_p > 0 else 1

    if h_ratio < 0.5 or p_ratio < 0.5:
        collapsed = True
        print(f"[!!] {r['num_classes']}-class shows COLLAPSE:")
        if h_ratio < 0.5:
            print(f"     Hidden variance {h_ratio*100:.1f}% of binary")
        if p_ratio < 0.5:
            print(f"     Pointer variance {p_ratio*100:.1f}% of binary")
        print()

if collapsed:
    print("=> HYPOTHESIS CONFIRMED!")
    print("   Internal states collapse to single averaged point for 3+ classes")
    print("   Binary works because 2-way split is easier even with collapse")
    print()
    print("ROOT CAUSE: Pointer/hidden state NOT learning to diversify")
else:
    print("=> Hypothesis NOT confirmed")
    print("   Variance is similar across all class counts")
    print("   Problem must be elsewhere (output head, gradients, etc.)")

print()
print("="*70)
