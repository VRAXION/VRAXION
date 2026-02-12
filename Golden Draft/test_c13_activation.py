"""Test C13 activation - self-gated smooth activation."""

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
torch.set_num_threads(20)

# Patch ACT_NAME before importing model
import vraxion.instnct.absolute_hallway as ah_module
ah_module.ACT_NAME = "c13"

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader
from tools.instnct_train_steps import train_steps

print("="*70)
print("C13 ACTIVATION TEST - Self-gated smooth activation")
print("="*70)
print()
print("C13 formula:")
print("  t = 1.0 + (x / 2.0)")
print("  t = clamp(t, 0, 1)")
print("  output = x * t^2")
print()
print("Properties:")
print("  - Bounded below (like tanh)")
print("  - Can grow above (like ReLU)")
print("  - Smooth self-gating (t^2)")
print()
print("Testing on 10-class COPY task:")
print("  Baseline (tanh): ~10%")
print()
print("="*70)
print()

# Quick test on 10-class only
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=10)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=64, slot_dim=64)

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="c13_test_10class",
    model_name="test_c13_10class"
)

acc = result.get('final_accuracy', 0) * 100
print(f"\n{'='*70}")
print(f"RESULT: C13 Activation")
print(f"{'='*70}")
print(f"10-class accuracy: {acc:.1f}%")
print(f"Baseline (tanh):   10.0%")
print(f"Change:            {acc - 10:.1f}%")
print()

if acc > 50:
    print("[OK] C13 WORKS! Solved the collapse!")
elif acc > 20:
    print("[~] C13 helps somewhat")
elif acc < 1:
    print("[!!] C13 BROKE IT (0% accuracy)")
else:
    print("[~] C13 doesn't help much")

print(f"{'='*70}")
