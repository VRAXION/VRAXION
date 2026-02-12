"""Test if LayerNorm fixes dimensional collapse."""

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
print("LAYERNORM FIX TEST")
print("="*70)
print()
print("Change made: Added LayerNorm after h update")
print()
print("  h = upd")
print("  h = self.h_layernorm(h)  # <- FORCES dimensional diversity")
print()
print("This forces all 64 dimensions to have mean=0, std=1")
print("-> Prevents collapse to [x, x, x, ...]")
print()
print("Testing on 3 tasks:")
print("  1. Binary (2-class) - baseline was ~58%")
print("  2. 3-class - baseline was ~35%")
print("  3. 10-class - baseline was ~10%")
print()
print("If LayerNorm works, 10-class should JUMP UP!")
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
    dataset_name="layernorm_test_binary",
    model_name="test_layernorm_binary"
)

binary_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {binary_acc:.1f}%")
print(f"Baseline (no LayerNorm): ~58%")
print(f"Change: {binary_acc - 58:.1f}%")

results.append({
    "task": "Binary (2-class)",
    "baseline": 58,
    "with_layernorm": binary_acc,
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
    dataset_name="layernorm_test_3class",
    model_name="test_layernorm_3class"
)

three_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {three_acc:.1f}%")
print(f"Baseline (no LayerNorm): ~35%")
print(f"Change: {three_acc - 35:.1f}%")

results.append({
    "task": "3-class",
    "baseline": 35,
    "with_layernorm": three_acc,
    "change": three_acc - 35
})

# Test 3: 10-class
print("\n" + "="*70)
print("TEST 3: 10-CLASS COPY (THE BIG TEST!)")
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
    dataset_name="layernorm_test_10class",
    model_name="test_layernorm_10class"
)

ten_acc = result.get('final_accuracy', 0) * 100
print(f"\nFinal accuracy: {ten_acc:.1f}%")
print(f"Baseline (no LayerNorm): ~10%")
print(f"Change: {ten_acc - 10:.1f}%")

results.append({
    "task": "10-class",
    "baseline": 10,
    "with_layernorm": ten_acc,
    "change": ten_acc - 10
})

# Summary
print("\n" + "="*70)
print("SUMMARY: LayerNorm Fix")
print("="*70)
print()

for r in results:
    if r['change'] > 20:
        status = "HUGE WIN!"
    elif r['change'] > 10:
        status = "BIG IMPROVEMENT"
    elif r['change'] > 5:
        status = "IMPROVEMENT"
    elif abs(r['change']) < 2:
        status = "NO CHANGE"
    elif r['change'] < -5:
        status = "BROKE IT"
    else:
        status = "SLIGHT CHANGE"

    print(f"{r['task']:20s}: {r['baseline']:5.1f}% -> {r['with_layernorm']:5.1f}%  ({r['change']:+5.1f}%)  [{status}]")

print()
print("="*70)
print("VERDICT:")
print("="*70)
print()

ten_class_result = results[2]

if ten_class_result['with_layernorm'] > 50:
    print("[OK] LAYERNORM WORKS! 10-class is now >50%!")
    print()
    print("The dimensional collapse is FIXED!")
    print("Model can now distinguish 10 different classes.")
    print()
    print("Root cause was: h dimensions collapsing to [x, x, x, ...]")
    print("Fix: LayerNorm forces dimensions to spread out (mean=0, std=1)")
    print()
    print("NEXT STEPS:")
    print("  - Test on harder tasks (assoc_clean, etc.)")
    print("  - Tune LayerNorm placement/settings")
    print("  - Celebrate! 🎉")

elif ten_class_result['with_layernorm'] > 20:
    print("[~] LAYERNORM HELPS but not enough")
    print()
    print(f"10-class improved from 10% to {ten_class_result['with_layernorm']:.1f}%")
    print("But still far from the target (should be 90-100%)")
    print()
    print("Possible issues:")
    print("  - LayerNorm placement (maybe need it elsewhere too)")
    print("  - Still some collapse happening")
    print("  - Need additional fixes (residual, mixing, etc.)")

else:
    print("[!!] LAYERNORM DOESN'T HELP")
    print()
    print("10-class accuracy didn't improve significantly")
    print()
    print("Possible reasons:")
    print("  - LayerNorm applied in wrong place")
    print("  - Collapse happening elsewhere")
    print("  - Different root cause than we thought")
    print()
    print("NEXT STEPS:")
    print("  - Try LayerNorm in different locations")
    print("  - Try dimension mixing instead")
    print("  - Re-examine the collapse mechanism")

print()
print("="*70)
