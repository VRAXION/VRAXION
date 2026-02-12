"""Test if sum instead of mean helps with collapse."""

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
print("SUM vs MEAN TEST - Does summing help?")
print("="*70)
print()
print("Change made: fused = gathered.sum(dim=1)  # Was: .mean(dim=1)")
print()
print("Testing on 3 tasks:")
print("  1. Binary (2-class) - baseline was ~58%")
print("  2. 3-class - baseline was ~35%")
print("  3. 10-class - baseline was ~10%")
print()
print("If sum helps, we should see higher accuracy!")
print()
print("="*70)
print()

results = []

# Test 1: Binary (2-class)
print("\n" + "="*70)
print("TEST 1: BINARY (2-class COPY)")
print("="*70)
print()

x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=64)

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="sum_test_binary",
    model_name="test_sum_binary"
)

binary_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {binary_acc:.1f}%")
print(f"Baseline (mean): ~58%")
print(f"Change: {binary_acc - 58:.1f}%")

results.append({
    "task": "Binary (2-class)",
    "baseline": 58,
    "with_sum": binary_acc,
    "change": binary_acc - 58
})

# Test 2: 3-class
print("\n" + "="*70)
print("TEST 2: 3-CLASS COPY")
print("="*70)
print()

x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=3)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=3, ring_len=64, slot_dim=64)

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="sum_test_3class",
    model_name="test_sum_3class"
)

three_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {three_acc:.1f}%")
print(f"Baseline (mean): ~35%")
print(f"Change: {three_acc - 35:.1f}%")

results.append({
    "task": "3-class",
    "baseline": 35,
    "with_sum": three_acc,
    "change": three_acc - 35
})

# Test 3: 10-class
print("\n" + "="*70)
print("TEST 3: 10-CLASS COPY")
print("="*70)
print()

x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=10)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=64, slot_dim=64)

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="sum_test_10class",
    model_name="test_sum_10class"
)

ten_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {ten_acc:.1f}%")
print(f"Baseline (mean): ~10%")
print(f"Change: {ten_acc - 10:.1f}%")

results.append({
    "task": "10-class",
    "baseline": 10,
    "with_sum": ten_acc,
    "change": ten_acc - 10
})

# Summary
print("\n" + "="*70)
print("SUMMARY: SUM vs MEAN")
print("="*70)
print()

for r in results:
    status = "IMPROVED" if r['change'] > 2 else "NO CHANGE" if abs(r['change']) < 2 else "WORSE"
    print(f"{r['task']:20s}: {r['baseline']:5.1f}% -> {r['with_sum']:5.1f}%  ({r['change']:+5.1f}%)  [{status}]")

print()
print("="*70)
print("VERDICT:")
print("="*70)
print()

if any(r['change'] > 5 for r in results):
    print("[OK] Summing HELPS! Significant improvement on some tasks.")
    print()
    print("This confirms that averaging was losing information.")
    print("Next step: Try other fixes (LayerNorm, dimension coupling, etc.)")
elif any(r['change'] > 2 for r in results):
    print("[~] Summing helps SLIGHTLY.")
    print()
    print("Small improvement, but root cause (dimensional collapse) remains.")
    print("Need deeper fixes: prevent h dimensions from collapsing.")
else:
    print("[!!] Summing does NOT help.")
    print()
    print("The averaging at ring readout is NOT the main problem.")
    print("Root cause is deeper: h dimensions collapse BEFORE averaging.")
    print()
    print("Need to fix dimensional collapse in recurrent update:")
    print("  - Add LayerNorm")
    print("  - Add dimension mixing")
    print("  - Use grouped updates")

print()
print("="*70)
