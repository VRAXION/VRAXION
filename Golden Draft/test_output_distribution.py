"""Check what classes the model actually predicts."""

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

import torch
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("OUTPUT DISTRIBUTION TEST")
print("="*70)
print()

# Test 3-class after training
print("Testing 3-CLASS model after training...")
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=3)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load trained model (if exists) or create fresh
model = AbsoluteHallway(input_dim=1, num_classes=3, ring_len=64, slot_dim=64)

# Quick train (just 20 steps to see pattern)
from tools.instnct_train_steps import train_steps
result = train_steps(
    model=model,
    loader=loader,
    steps=50,
    dataset_name="output_dist_3class",
    model_name="test_output_dist"
)

print("\nEvaluating predictions...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(torch.float64)  # Match model dtype
        outputs = model(batch_x)
        # Model returns (logits, movement_cost)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch_y.cpu().tolist())

print()
print("="*70)
print("PREDICTION ANALYSIS")
print("="*70)
print()

# Count predictions
from collections import Counter
pred_counts = Counter(all_preds)
target_counts = Counter(all_targets)

print("Target distribution (ground truth):")
for i in range(3):
    count = target_counts.get(i, 0)
    pct = 100 * count / len(all_targets)
    print(f"  Class {i}: {count:4d} ({pct:5.1f}%)")
print()

print("Prediction distribution (what model outputs):")
for i in range(3):
    count = pred_counts.get(i, 0)
    pct = 100 * count / len(all_preds) if all_preds else 0
    print(f"  Class {i}: {count:4d} ({pct:5.1f}%)")
print()

# Check if model uses all 3 classes or just 2
classes_used = len([c for c in pred_counts.values() if c > 0])
print(f"Classes actually used: {classes_used} out of 3")
print()

if classes_used < 3:
    unused = [i for i in range(3) if pred_counts.get(i, 0) == 0]
    print(f"[!!] Model NEVER predicts class(es): {unused}")
    print(f"     This confirms binary limitation!")
elif max(pred_counts.values()) > 0.9 * len(all_preds):
    dominant = max(pred_counts, key=pred_counts.get)
    print(f"[!] Model mostly predicts class {dominant} ({100*pred_counts[dominant]/len(all_preds):.1f}%)")
    print(f"    This suggests collapse to single mode")
else:
    print(f"[OK] Model uses all 3 classes")
    print(f"     Binary limitation NOT confirmed")
    print(f"     Problem might be elsewhere (gradient flow, LR, etc.)")

print()
print("="*70)
