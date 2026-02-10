"""Quick test: Does 3-class work or fail like 10-class?"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

# Environment setup
ROOT = "S:/AI/work/VRAXION_DEV/Golden Draft"
os.environ['VRX_ROOT'] = ROOT
os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'
os.environ['VRX_PRECISION'] = 'fp64'
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'
os.environ['VRX_PTR_INERTIA_OVERRIDE'] = '0.6'
os.environ['VRX_AGC_ENABLED'] = '0'
os.environ['VRX_GRAD_CLIP'] = '0.0'
os.environ['VRX_UPDATE_SCALE'] = '1.0'
os.environ['VRX_LR'] = '0.001'
os.environ['VRX_SYNTH'] = '0'
os.environ['VRX_HEARTBEAT_STEPS'] = '10'
os.environ['VRX_DEBUG_STATS'] = '0'

LOG_PATH = Path(ROOT) / "logs/probe/copy_test_3class.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ['VRX_DASHBOARD_LOG'] = str(LOG_PATH)

import torch
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from tools.instnct_train_steps import train_steps
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("CRITICAL TEST: 3-CLASS COPY")
print("="*70)
print()
print("Question: Does model break at binary->3 or gradually degrade?")
print()
print("If 3-class WORKS (>40%): Gradual degradation with num_classes")
print("If 3-class FAILS (~33%): Cliff at binary boundary")
print()
print("="*70)
print()

# Generate data
print("Generating 3-class COPY task...")
x, y, num_classes = task_copy(n_samples=5000, seq_len=16, vocab_size=3)
print(f"Generated: {x.shape[0]} samples, seq_len={x.shape[1]}, classes={num_classes}")

# Sanity check
print(f"\nLabel distribution:")
print(f"  Min: {y.min().item()}, Max: {y.max().item()}")
print(f"  Unique: {sorted(y.unique().tolist())}")
print(f"  Random baseline: {100/num_classes:.1f}%")
print()

# Create DataLoader
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model
print("Creating model...")
model = AbsoluteHallway(
    input_dim=1,
    num_classes=num_classes,
    ring_len=64,
    slot_dim=64,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} parameters")
print()

# Train
print("Training for 200 steps...")
print()

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="copy_3class",
    model_name="test_copy_3class"
)

final_acc = result.get('final_accuracy', 0.0) * 100
random_baseline = 100 / num_classes

print()
print("="*70)
print("RESULTS: 3-CLASS COPY")
print("="*70)
print(f"Final accuracy:  {final_acc:.1f}%")
print(f"Random baseline: {random_baseline:.1f}%")
print(f"Improvement:     {final_acc - random_baseline:+.1f} points")
print()

if final_acc > 50:
    print("[OK] 3-CLASS WORKS!")
    print("     => Problem is GRADUAL degradation with num_classes")
    print("     => Model can learn beyond binary, just gets worse")
    print()
    print("Next: Test 4-class, 5-class to find degradation curve")
elif final_acc > random_baseline + 5:
    print("[~] 3-CLASS MARGINAL")
    print("     => Model struggles immediately above binary")
    print("     => Suggests softmax saturation or gradient issue")
    print()
    print("Next: Check gradient norms and learning rate")
else:
    print("[!!] 3-CLASS FAILS!")
    print("     => CLIFF at binary boundary")
    print("     => Model architecture limited to 2 classes")
    print()
    print("Next: Investigate output head - hardcoded binary somewhere?")

print()
print("="*70)
