"""Test 1-dim + big ring on 2-class (binary) task."""

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
print("1-DIM + BIG RING - Binary (2-class) Test")
print("="*70)
print()
print("Question: Does 1-dim improve SIMPLE tasks?")
print()
print("Config:")
print("  slot_dim = 1")
print("  ring_len = 4,096")
print()
print("Testing on BINARY (2-class) COPY:")
print("  Baseline (64×64): ~58%")
print("  Target (1×4096): 90%+?")
print()
print("="*70)
print()

# Test binary with 1-dim big ring
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=4096, slot_dim=1)
print(f"Model: {sum(p.numel() for p in model.parameters())} params")
print()

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="1dim_binary_test",
    model_name="test_1dim_binary"
)

acc = result.get('final_accuracy', 0) * 100
print(f"\n{'='*70}")
print(f"RESULT: 1-Dim + Big Ring on Binary Task")
print(f"{'='*70}")
print(f"Binary accuracy:    {acc:.1f}%")
print(f"Baseline (64×64):   58.0%")
print(f"Change:            {acc - 58:.1f}%")
print()

if acc > 90:
    print("[OK] HUGE WIN! 1-dim dominates on simple tasks!")
elif acc > 70:
    print("[OK] Significant improvement!")
elif acc > 58:
    print("[~] Slight improvement")
elif acc < 1:
    print("[!!] Broken (0%)")
else:
    print("[~] No better than baseline")

print(f"{'='*70}")
