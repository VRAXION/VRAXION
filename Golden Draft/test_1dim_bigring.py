"""Test 1-dim slots with HUGE ring to match capacity."""

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
print("1-DIM SLOTS + BIG RING - Capacity-matched test")
print("="*70)
print()
print("Fair comparison:")
print("  Original: 64 dims × 64 positions = 4,096 storage units")
print("  New:      1 dim   × 4,096 positions = 4,096 storage units")
print()
print("This gives equal CAPACITY but different structure:")
print("  - Original: dimensional diversity within each slot")
print("  - New: positional/temporal diversity across slots")
print()
print("Testing on 10-class COPY:")
print("  Baseline (64×64): 10%")
print("  Target (1×4096): >50%?")
print()
print("="*70)
print()

# Test 10-class with 1-dim slots and huge ring
x, y, num_classes = task_copy(n_samples=1000, seq_len=16, vocab_size=10)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Creating model with slot_dim=1, ring_len=4096...")
model = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=4096, slot_dim=1)
print(f"Model created: {sum(p.numel() for p in model.parameters())} params")
print(f"Ring capacity: 4,096 positions × 1 dim = 4,096 units")
print()

result = train_steps(
    model=model,
    loader=train_loader,
    steps=200,
    dataset_name="1dim_bigring_test",
    model_name="test_1dim_bigring"
)

acc = result.get('final_accuracy', 0) * 100
print(f"\n{'='*70}")
print(f"RESULT: 1-Dim Slots + Big Ring (4,096 positions)")
print(f"{'='*70}")
print(f"10-class accuracy: {acc:.1f}%")
print(f"Baseline (64×64):   10.0%")
print(f"Change:            {acc - 10:.1f}%")
print()

if acc > 50:
    print("[OK] CAPACITY-MATCHED 1-DIM WORKS!")
    print()
    print("KEY INSIGHT:")
    print("  Positional diversity > Dimensional diversity")
    print("  4,096 temporal slots > 64 spatial dimensions")
elif acc > 20:
    print("[~] Helps somewhat, but not enough")
elif acc < 1:
    print("[!!] Still broken (0%)")
else:
    print("[~] Slight improvement over baseline")

print(f"{'='*70}")
