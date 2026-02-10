"""Test if input_proj (first layer) is where variance collapses."""

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
print("INPUT_PROJ COLLAPSE TEST")
print("="*70)
print()
print("Question: Does the first layer (input_proj) collapse variance?")
print()
print("Hypothesis:")
print("  - The collapse happens BEFORE think_proj (ruled out bottleneck)")
print("  - Suspects: input_proj, ring readout, or activations")
print()
print("Test Plan:")
print("  1. Train binary model (works at 60%)")
print("  2. Hook into layers during forward:")
print("     - AFTER input_proj (very first layer)")
print("     - AFTER activation")
print("     - AFTER ring context addition (gru_in)")
print("     - FINAL hidden state (h)")
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
    dataset_name="input_proj_collapse_test",
    model_name="test_input_proj_collapse"
)

print("\nCapturing layer activations during inference...")

# Create test set
test_x, test_y, _ = task_copy(n_samples=100, seq_len=16, vocab_size=2)

model.eval()

# Storage for activations at each layer
activations = {
    'after_input_proj': [],
    'after_activation': [],
    'after_ring_context': [],
    'final_h': []
}

print("\nInspecting model architecture...")
print()

# Check if input_proj exists
if hasattr(model, 'input_proj'):
    print(f"Found: model.input_proj")
    print(f"  Type: Linear, {model.input_proj.in_features} -> {model.input_proj.out_features}")
else:
    print("[!!] ERROR: model.input_proj not found")
    sys.exit(1)

print()
print("Monkey-patching forward() to capture intermediate states...")
print()

# We need to monkey-patch the forward method to capture intermediates
# because they're computed inside the forward loop and not exposed

original_forward = model.forward

captured_count = 0

def forward_with_capture(x, return_xray=False):
    global captured_count

    # Only capture for the first 100 samples (to avoid memory issues)
    if captured_count >= 100:
        return original_forward(x, return_xray)

    # We'll manually replicate the first few steps of forward to capture intermediates
    B, T, _ = x.shape
    device = x.device

    # Initialize states (simplified, just enough to capture what we need)
    h = torch.zeros(B, model.slot_dim, device=device, dtype=x.dtype)

    # Process just the last timestep (since we're doing COPY task, last token matters)
    t = T - 1

    # STEP 1: input_proj
    inp_raw = model.input_proj(x[:, t, :])
    activations['after_input_proj'].append(inp_raw.detach().cpu())

    # STEP 2: activation
    inp_activated = model._apply_activation(inp_raw)
    activations['after_activation'].append(inp_activated.detach().cpu())

    # STEP 3: For ring context, we'd need to compute the full ring state
    # which is complex. Skip for now and just capture h at the end.

    captured_count += 1

    # Now run the actual forward to get correct output
    return original_forward(x, return_xray)

model.forward = forward_with_capture

# Run inference
with torch.no_grad():
    for i in range(100):
        sample = test_x[i:i+1].to(torch.float64)
        _ = model(sample)

# Restore original forward
model.forward = original_forward

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

for name in ['after_input_proj', 'after_activation', 'after_ring_context', 'final_h']:
    if activations[name]:  # Only analyze if we captured something
        var = analyze_activations(name, activations[name])
        if var is not None:
            variances[name] = var

print()
print("="*70)
print("VARIANCE PROGRESSION")
print("="*70)
print()

if variances:
    # Sort by appearance in forward pass
    layer_order = ['after_input_proj', 'after_activation', 'after_ring_context', 'final_h']

    print("Layer progression:")
    max_var = max(variances.values()) if variances.values() else 1.0
    for i, name in enumerate(layer_order):
        if name in variances:
            var = variances[name]
            bar = '#' * max(1, int(40 * var / max_var))
            print(f"  {i+1}. {name:25s}: {var:.8f}  {bar}")

    # Analysis
    print()
    print("Analysis:")
    print()

    if 'after_input_proj' in variances:
        var_proj = variances['after_input_proj']
        print(f"  After input_proj:  {var_proj:.8f}")

        if var_proj < 0.001:
            print("  [!!] COLLAPSE AT FIRST LAYER!")
            print("      input_proj is mapping all inputs to nearly identical values")
            print()
            print("ROOT CAUSE: input_proj (first linear layer) is the problem")
            print()
            print("Possible reasons:")
            print("  1. Weights initialized poorly (all near zero or identical)")
            print("  2. Input tokens are too similar (all 0.0 or 1.0)")
            print("  3. Layer is saturated (all outputs map to same value)")
            print()
            print("FIX: Check input_proj weight initialization")
        else:
            print("  [OK] input_proj has good variance")
            print("      Collapse happens LATER in the forward pass")

    if 'after_input_proj' in variances and 'after_activation' in variances:
        ratio = variances['after_activation'] / (variances['after_input_proj'] + 1e-12)
        print()
        print(f"  After input_proj:  {variances['after_input_proj']:.8f}")
        print(f"  After activation:  {variances['after_activation']:.8f}")
        print(f"  Ratio (act/proj):  {ratio:.4f}x")
        print()

        if ratio < 0.1:
            print("  [!!] ACTIVATION FUNCTION is collapsing variance!")
            print("      All different inputs map to same activation output")
            print()
            print("ROOT CAUSE: Activation saturation")
            print()
            print("Possible reasons:")
            print("  1. All values near zero -> activation outputs all near zero")
            print("  2. Activation function has narrow active range")
            print()
            print("FIX: Check activation function and input magnitudes")
        elif ratio < 0.5:
            print("  [~] Activation reduces variance (expected)")
        else:
            print("  [OK] Activation preserves variance")

    # Overall verdict
    print()
    print("="*70)
    print("VERDICT:")
    print("="*70)
    print()

    if 'after_input_proj' in variances and variances['after_input_proj'] < 0.001:
        print("[!!] COLLAPSE POINT: input_proj (FIRST LAYER)")
        print()
        print("The very first linear layer is mapping all inputs to identical values.")
        print()
        print("This explains why:")
        print("  - Binary works (58-60%): Even with collapsed states, 2-way split is easy")
        print("  - 10-class fails (10%): Collapsed states can't distinguish 10 classes")
        print()
        print("NEXT STEP: Inspect input_proj weights and input token distribution")

    elif 'after_activation' in variances and variances['after_activation'] < 0.001:
        print("[!!] COLLAPSE POINT: Activation function (SECOND STEP)")
        print()
        print("The activation function is saturating all inputs to the same output.")
        print()
        print("NEXT STEP: Check activation function and input magnitudes")

    else:
        print("[~] Collapse happens later (after ring context or in h update)")
        print()
        print("NEXT STEP: Hook into ring operations and h update")

else:
    print("[!!] No activations captured - something went wrong")

print()
print("="*70)
