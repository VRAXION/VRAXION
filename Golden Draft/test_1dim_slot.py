"""Test if using 1-dim slots (instead of 64-dim) fixes the collapse."""

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

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader
from tools.instnct_train_steps import train_steps

print("="*70)
print("1-DIM SLOT TEST - Eliminate dimensional collapse!")
print("="*70)
print()
print("Hypothesis:")
print("  If 64 dims collapse to 1 value anyway...")
print("  Why not just use 1 dimension from the start?")
print()
print("Change: slot_dim = 1 (was 64)")
print()
print("Expected:")
print("  - No dimensional collapse (only 1 dim!)")
print("  - Model forced to use POSITIONAL diversity")
print("  - 64 ring positions provide the capacity")
print()
print("Testing on 10-class COPY:")
print("  Baseline (64-dim slots): 10%")
print("  Target (1-dim slots): >50%?")
print()
print("="*70)
print()

# Test 10-class with 1-dim slots
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=10)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Creating model with slot_dim=1 (was 64)...")
model = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=64, slot_dim=1)
print(f"Model created: {sum(p.numel() for p in model.parameters())} params")
print()

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="1dim_test_10class",
    model_name="test_1dim_10class"
)

acc = result.get('final_accuracy', 0) * 100
print(f"\n{'='*70}")
print(f"RESULT: 1-Dim Slots")
print(f"{'='*70}")
print(f"10-class accuracy: {acc:.1f}%")
print(f"Baseline (64-dim):  10.0%")
print(f"Change:            {acc - 10:.1f}%")
print()

if acc > 50:
    print("[OK] 1-DIM WORKS! Solved the collapse!")
    print()
    print("The trick: Use positional diversity instead of dimensional!")
    print("64 ring positions > 64 collapsed dimensions")
elif acc > 20:
    print("[~] 1-dim helps somewhat")
elif acc < 1:
    print("[!!] 1-dim BROKE IT (0% accuracy)")
else:
    print("[~] 1-dim doesn't help much")

print(f"{'='*70}")
