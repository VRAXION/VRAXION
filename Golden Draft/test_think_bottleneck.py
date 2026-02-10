"""Test if think_proj bottleneck (64->21->64) is where variance collapses."""

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
print("THINK_PROJ BOTTLENECK TEST")
print("="*70)
print()
print("Question: Does the 64->21->64 bottleneck collapse variance?")
print()
print("Hypothesis:")
print("  - Learned params have variance=5.84 (diverse)")
print("  - Hidden states have variance=0.0008 (collapsed)")
print("  -> Something in forward pass collapses activations")
print()
print("Test Plan:")
print("  1. Train binary model (works at 60%)")
print("  2. Hook into layers during forward:")
print("     - BEFORE think_proj (64-dim)")
print("     - AT bottleneck (21-dim)")
print("     - AFTER think_proj (64-dim)")
print("     - AT output head (64-dim)")
print("  3. Measure variance at each point")
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
    dataset_name="think_bottleneck_test",
    model_name="test_think_bottleneck"
)

print("\nCapturing layer activations during inference...")

# Create test set
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

model.eval()

# Storage for activations at each layer
activations = {
    'before_think': [],
    'at_bottleneck': [],
    'after_think': [],
    'at_head': []
}

# Find the think_proj layers in the model
print("\nInspecting model architecture...")
print()

# AbsoluteHallway structure:
# - model.think_down: nn.Linear(64, 21)  # First part of bottleneck
# - model.think_up: nn.Linear(21, 64)    # Second part of bottleneck
# - model.head: LocationExpertRouter (output head)

# Check if these exist
for attr in ['think_down', 'think_up', 'head']:
    if hasattr(model, attr):
        layer = getattr(model, attr)
        print(f"Found: model.{attr}")
        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
            print(f"  Type: Linear, {layer.in_features} -> {layer.out_features}")
        else:
            print(f"  Type: {type(layer).__name__}")

print()

# Register hooks
def make_hook(name):
    def hook(module, input, output):
        # input is tuple of tensors, output is tensor
        if isinstance(output, torch.Tensor):
            activations[name].append(output.detach().cpu())
    return hook

# We need to hook into the intermediate layers
# But AbsoluteHallway doesn't expose them directly in forward()
# Let's check the actual forward pass structure

print("Reading AbsoluteHallway.forward() to find hook points...")
print()

# Strategy: Monkey-patch the model to capture intermediate states
original_forward = model.forward

def forward_with_capture(x):
    # Run the original forward but capture intermediates
    # We'll need to replicate parts of the forward pass

    # First, get the initial hidden state after ring operations
    # This requires calling parts of forward() manually

    # For now, let's just capture what we can from the output
    result = original_forward(x)

    # Try to capture from think_down and think_up if they were called
    # But we can't hook them mid-forward without modifying the model

    return result

# Alternative: Use register_forward_hook on the layers
if hasattr(model, 'think_down'):
    hook1 = model.think_down.register_forward_hook(make_hook('at_bottleneck'))

if hasattr(model, 'think_up'):
    hook2 = model.think_up.register_forward_hook(make_hook('after_think'))

if hasattr(model, 'head'):
    # Head receives the final 64-dim representation
    def head_input_hook(module, input, output):
        # input[0] is the 64-dim representation
        activations['at_head'].append(input[0].detach().cpu())
    hook3 = model.head.register_forward_hook(head_input_hook)

# For 'before_think', we need to hook earlier
# Let me check if there's a layer before think_down

print("Running inference with hooks...")
print()

with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)
        _ = model(sample)

# Remove hooks
if hasattr(model, 'think_down'):
    hook1.remove()
if hasattr(model, 'think_up'):
    hook2.remove()
if hasattr(model, 'head'):
    hook3.remove()

# Analyze variance at each point
print("="*70)
print("VARIANCE ANALYSIS")
print("="*70)
print()

# Function to calculate and display variance
def analyze_activations(name, acts_list):
    if not acts_list:
        print(f"\n[!!] No activations captured for {name}")
        return None

    # Concatenate all samples
    all_acts = torch.cat(acts_list, dim=0)

    # Calculate variance per dimension
    var_per_dim = all_acts.var(dim=0)
    mean_var = var_per_dim.mean().item()
    std_var = var_per_dim.std().item()
    min_var = var_per_dim.min().item()
    max_var = var_per_dim.max().item()

    # Overall statistics
    mean_val = all_acts.mean().item()
    std_val = all_acts.std().item()
    min_val = all_acts.min().item()
    max_val = all_acts.max().item()

    print(f"\n{name.upper()}")
    print(f"  Shape: {all_acts.shape}")
    print(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Mean: {mean_val:.6f}, Std dev: {std_val:.6f}")
    print()
    print(f"  Variance (per-dimension):")
    print(f"    Mean:  {mean_var:.8f}")
    print(f"    Std:   {std_var:.8f}")
    print(f"    Range: [{min_var:.8f}, {max_var:.8f}]")

    # Verdict
    if mean_var < 0.001:
        print(f"  [!!] VERY LOW VARIANCE - collapsed")
    elif mean_var < 0.01:
        print(f"  [~] LOW VARIANCE - partially collapsed")
    elif mean_var < 0.1:
        print(f"  [~] MODERATE VARIANCE")
    else:
        print(f"  [OK] GOOD VARIANCE - diverse")

    return mean_var

# Analyze each layer
variances = {}

for name in ['at_bottleneck', 'after_think', 'at_head']:
    var = analyze_activations(name, activations[name])
    if var is not None:
        variances[name] = var

print()
print("="*70)
print("VARIANCE PROGRESSION")
print("="*70)
print()

if variances:
    # Sort by appearance in forward pass (conceptually)
    layer_order = ['at_bottleneck', 'after_think', 'at_head']

    print("Layer progression:")
    for i, name in enumerate(layer_order):
        if name in variances:
            var = variances[name]
            bar = '#' * max(1, int(40 * var / max(variances.values())))
            print(f"  {i+1}. {name:20s}: {var:.8f}  {bar}")

    # Find the collapse point
    print()
    print("Analysis:")

    if 'at_bottleneck' in variances and 'after_think' in variances:
        ratio = variances['after_think'] / (variances['at_bottleneck'] + 1e-12)
        print(f"  Bottleneck (21-dim):     {variances['at_bottleneck']:.8f}")
        print(f"  After expansion (64-dim): {variances['after_think']:.8f}")
        print(f"  Ratio (after/before):     {ratio:.4f}x")
        print()

        if ratio < 0.5:
            print("  [!!] VARIANCE DROPS after bottleneck expansion")
            print("      The think_up layer (21->64) is losing diversity")
        elif ratio > 2.0:
            print("  [OK] Variance increases after expansion (expected)")
        else:
            print("  [~] Variance roughly preserved through bottleneck")

    if 'after_think' in variances and 'at_head' in variances:
        ratio = variances['at_head'] / (variances['after_think'] + 1e-12)
        print()
        print(f"  After think_proj:  {variances['after_think']:.8f}")
        print(f"  At output head:    {variances['at_head']:.8f}")
        print(f"  Ratio (head/think):{ratio:.4f}x")
        print()

        if ratio < 0.5:
            print("  [!!] VARIANCE DROPS before output head")
            print("      Some operation between think_proj and head is collapsing")
        else:
            print("  [~] Variance preserved to output head")

    # Overall verdict
    print()
    print("="*70)
    print("VERDICT:")
    print("="*70)
    print()

    if all(v < 0.001 for v in variances.values()):
        print("[!!] ALL layers have collapsed variance (<0.001)")
        print("     The collapse happens BEFORE think_down (before 21-dim bottleneck)")
        print()
        print("ROOT CAUSE: The collapse is NOT from think_proj!")
        print("            It happens earlier in the forward pass:")
        print("            - Ring operations?")
        print("            - Pointer-based readout?")
        print("            - Input projection?")
        print()
        print("NEXT TEST: Hook into ring operations (before think_proj)")

    elif variances['at_bottleneck'] > 0.01 and variances.get('after_think', 0) < 0.001:
        print("[!!] BOTTLENECK is the collapse point!")
        print("     Variance drops in the 21-dim bottleneck")
        print()
        print("ROOT CAUSE: think_proj (64->21->64) is too narrow")
        print()
        print("FIX: Increase think_dim from 21 to 32 or 64")

    else:
        print("[~] Collapse pattern unclear from these hooks")
        print("    Need more hook points to pinpoint exact location")

print()
print("="*70)
