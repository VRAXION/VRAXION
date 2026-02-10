"""Full diagnostic biopsy - compare 64-dim vs 1-dim to understand WHY things fail."""

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
import torch.nn as nn
import numpy as np
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*80)
print("FULL DIAGNOSTIC BIOPSY - Understanding the Failure")
print("="*80)
print()
print("Comparing:")
print("  Patient A: 64-dim slots (barely works - 58% binary)")
print("  Patient B: 1-dim slots  (dead - 0% binary)")
print()
print("Measurements:")
print("  1. Gradient flow (are gradients reaching all layers?)")
print("  2. Activation values (what numbers are we seeing?)")
print("  3. Weight changes (is learning happening?)")
print("  4. Ring state (what's stored in memory?)")
print("  5. Loss landscape (stuck? moving?)")
print()
print("="*80)
print()

# Create simple binary task
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Get one batch for testing
batch_x, batch_y = next(iter(loader))
batch_x = batch_x.to(torch.float64)
batch_y = batch_y.long()

print("Test batch:")
print(f"  Input shape: {batch_x.shape}")
print(f"  Labels: {batch_y[:10].tolist()}")
print()

# =============================================================================
# PATIENT A: 64-dim (baseline)
# =============================================================================

print("="*80)
print("PATIENT A: 64-dim Baseline")
print("="*80)
print()

model_64 = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)
model_64 = model_64.to(torch.float64)  # Match input dtype
optimizer_64 = torch.optim.Adam(model_64.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Model params: {sum(p.numel() for p in model_64.parameters())}")
print()

# Training loop with diagnostics
print("Training for 10 steps with full diagnostics...")
print()

for step in range(10):
    optimizer_64.zero_grad()

    # Forward pass
    logits, aux_loss = model_64(batch_x)
    loss = criterion(logits, batch_y) + aux_loss

    # Backward pass
    loss.backward()

    # DIAGNOSTIC 1: Gradient magnitudes
    grad_norms = {}
    for name, param in model_64.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # DIAGNOSTIC 2: Activation statistics
    with torch.no_grad():
        # Get activations at different points
        input_proj_out = model_64.input_proj(batch_x[:, 0, :])  # First timestep
        input_proj_out = model_64._apply_activation(input_proj_out)

        # Predictions
        preds = logits.argmax(dim=1)
        acc = (preds == batch_y).float().mean().item()

    # DIAGNOSTIC 3: Weight statistics
    weight_stats = {}
    for name, param in model_64.named_parameters():
        weight_stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item()
        }

    # Update
    optimizer_64.step()

    # Print diagnostics
    print(f"Step {step}:")
    print(f"  Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%")
    print(f"  Input projection output: mean={input_proj_out.mean():.4f}, std={input_proj_out.std():.4f}")
    print(f"  Logits: {logits[0].detach().numpy()}")
    print(f"  Gradient norms:")
    for name, norm in sorted(grad_norms.items()):
        print(f"    {name:30s}: {norm:.6f}")
    print()

print()
print("Patient A Summary (after 10 steps):")
print(f"  Final loss: {loss.item():.4f}")
print(f"  Final accuracy: {acc*100:.1f}%")
print(f"  Total gradient norm: {sum(grad_norms.values()):.4f}")
print()

# Save final state for comparison
final_loss_64 = loss.item()
final_acc_64 = acc
final_grad_64 = sum(grad_norms.values())

# =============================================================================
# PATIENT B: 1-dim
# =============================================================================

print("="*80)
print("PATIENT B: 1-dim (Big Ring)")
print("="*80)
print()

model_1d = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=4096, slot_dim=1)
model_1d = model_1d.to(torch.float64)  # Match input dtype
optimizer_1d = torch.optim.Adam(model_1d.parameters(), lr=0.001)

print(f"Model params: {sum(p.numel() for p in model_1d.parameters())}")
print()

print("Training for 10 steps with full diagnostics...")
print()

for step in range(10):
    optimizer_1d.zero_grad()

    # Forward pass
    logits, aux_loss = model_1d(batch_x)
    loss = criterion(logits, batch_y) + aux_loss

    # Backward pass
    loss.backward()

    # DIAGNOSTIC 1: Gradient magnitudes
    grad_norms = {}
    for name, param in model_1d.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # DIAGNOSTIC 2: Activation statistics
    with torch.no_grad():
        # Get activations at different points
        input_proj_out = model_1d.input_proj(batch_x[:, 0, :])
        input_proj_out = model_1d._apply_activation(input_proj_out)

        # Predictions
        preds = logits.argmax(dim=1)
        acc = (preds == batch_y).float().mean().item()

    # DIAGNOSTIC 3: Weight statistics
    weight_stats = {}
    for name, param in model_1d.named_parameters():
        weight_stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item()
        }

    # Update
    optimizer_1d.step()

    # Print diagnostics
    print(f"Step {step}:")
    print(f"  Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%")
    print(f"  Input projection output: mean={input_proj_out.mean():.4f}, std={input_proj_out.std():.4f}")
    print(f"  Logits: {logits[0].detach().numpy()}")
    print(f"  Gradient norms:")
    for name, norm in sorted(grad_norms.items()):
        print(f"    {name:30s}: {norm:.6f}")
    print()

print()
print("Patient B Summary (after 10 steps):")
print(f"  Final loss: {loss.item():.4f}")
print(f"  Final accuracy: {acc*100:.1f}%")
print(f"  Total gradient norm: {sum(grad_norms.values()):.4f}")
print()

# Save final state
final_loss_1d = loss.item()
final_acc_1d = acc
final_grad_1d = sum(grad_norms.values())

# =============================================================================
# COMPARISON
# =============================================================================

print("="*80)
print("DIAGNOSIS - Side-by-Side Comparison")
print("="*80)
print()

print(f"{'Metric':<30s} {'Patient A (64-dim)':<20s} {'Patient B (1-dim)':<20s} {'Verdict'}")
print("-"*80)
print(f"{'Final Loss':<30s} {final_loss_64:<20.4f} {final_loss_1d:<20.4f}", end=" ")
if abs(final_loss_64 - final_loss_1d) < 0.1:
    print("SIMILAR")
elif final_loss_64 < final_loss_1d:
    print("A BETTER")
else:
    print("B BETTER")

print(f"{'Final Accuracy':<30s} {final_acc_64*100:<20.1f} {final_acc_1d*100:<20.1f}", end=" ")
if abs(final_acc_64 - final_acc_1d) < 0.05:
    print("SIMILAR")
elif final_acc_64 > final_acc_1d:
    print("A BETTER")
else:
    print("B BETTER")

print(f"{'Total Gradient Flow':<30s} {final_grad_64:<20.4f} {final_grad_1d:<20.4f}", end=" ")
if abs(final_grad_64 - final_grad_1d) < 0.1:
    print("SIMILAR")
elif final_grad_64 > final_grad_1d:
    print("A STRONGER")
else:
    print("B STRONGER")

print()
print("="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print()

if final_grad_1d < 0.01:
    print("[!!] Patient B has VANISHING GRADIENTS")
    print("     -> Learning is not happening at all!")
    print("     -> Gradients too small to update weights")
    print()

if final_loss_1d > 0.69:
    print("[!!] Patient B stuck at random guessing")
    print("     -> Loss ~0.69 = log(2) (random binary baseline)")
    print("     -> Model is not learning anything")
    print()

if abs(final_acc_1d - 0.5) < 0.05:
    print("[!!] Patient B at 50% accuracy (random)")
    print("     -> Just guessing randomly")
    print()

print("RECOMMENDATION:")
if final_grad_1d < 0.01:
    print("  The 1-dim config has gradient flow problems.")
    print("  Either:")
    print("    1. Architecture doesn't support 1-dim slots")
    print("    2. Need different initialization")
    print("    3. Need different learning rate")
    print("    4. Fundamental incompatibility with 1-dim")
else:
    print("  Gradients are flowing, but something else is wrong.")
    print("  Need deeper investigation of:")
    print("    - Ring addressing mechanism")
    print("    - Information capacity")
    print("    - Loss landscape")

print()
print("="*80)
