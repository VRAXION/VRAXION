"""Inspect model architecture for binary vs 10-class."""

import os
import sys
sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

# Minimal env
os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'
os.environ['VRX_ROOT'] = "S:/AI/work/VRAXION_DEV/Golden Draft"

import torch
from vraxion.instnct.absolute_hallway import AbsoluteHallway

print("="*70)
print("MODEL ARCHITECTURE COMPARISON: Binary vs 10-Class")
print("="*70)
print()

# Binary model
print("--- BINARY (2-class) ---")
model_2 = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)
params_2 = sum(p.numel() for p in model_2.parameters())
print(f"Total params: {params_2:,}")
print()
print("Named modules:")
for name, module in model_2.named_modules():
    if name and not any(x in name for x in ['activation', 'dropout']):
        print(f"  {name}: {module.__class__.__name__}")
print()
print("Parameters:")
for name, param in model_2.named_parameters():
    print(f"  {name}: {tuple(param.shape)} = {param.numel():,} params")
print()

# 10-class model
print("--- 10-CLASS ---")
model_10 = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=64, slot_dim=64)
params_10 = sum(p.numel() for p in model_10.parameters())
print(f"Total params: {params_10:,}")
print()
print("Named modules:")
for name, module in model_10.named_modules():
    if name and not any(x in name for x in ['activation', 'dropout']):
        print(f"  {name}: {module.__class__.__name__}")
print()
print("Parameters:")
for name, param in model_10.named_parameters():
    print(f"  {name}: {tuple(param.shape)} = {param.numel():,} params")
print()

print("="*70)
print("DIFFERENCE")
print("="*70)
print(f"Param difference: {params_10 - params_2:,} params")
print(f"Expected head diff: (10-2) * 64 = {8 * 64} weight + 8 bias = {8 * 64 + 8}")
print()

# Check if head weight/bias sizes make sense
if hasattr(model_2, 'head'):
    print("Binary head:")
    print(f"  weight shape: {model_2.head.weight.shape}")
    print(f"  bias shape: {model_2.head.bias.shape}")
    print()

if hasattr(model_10, 'head'):
    print("10-class head:")
    print(f"  weight shape: {model_10.head.weight.shape}")
    print(f"  bias shape: {model_10.head.bias.shape}")
    print()

print("="*70)
