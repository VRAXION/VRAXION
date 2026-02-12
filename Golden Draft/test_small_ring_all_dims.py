"""Test if the problem is LARGE RING, not low dimensions."""

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

import torch
import torch.nn as nn
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

print("="*70)
print("HYPOTHESIS TEST: Small ring (64) for ALL dimensions")
print("="*70)
print()
print("If the problem is LARGE RINGS (not low dims), then:")
print("  - All dimensions should work with ring_len=64")
print("  - Pointer can converge in small space")
print()
print("Testing: 1, 2, 4, 8, 16, 32, 64 dims × 64 ring")
print("="*70)
print()

# Binary task
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
batch_x, batch_y = next(iter(loader))
batch_x = batch_x.to(torch.float64)
batch_y = batch_y.long()

RING_LEN = 64  # FIXED small ring for all

results = []

for slot_dim in [1, 2, 4, 8, 16, 32, 64]:
    print(f"{'='*70}")
    print(f"slot_dim={slot_dim}, ring_len={RING_LEN} (capacity={slot_dim * RING_LEN})")
    print(f"{'='*70}")

    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=RING_LEN, slot_dim=slot_dim)
    model = model.to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []

    for step in range(10):
        optimizer.zero_grad()
        logits, aux_loss = model(batch_x)
        loss = criterion(logits, batch_y) + aux_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == batch_y).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

    final_acc = accs[-1]
    final_loss = losses[-1]
    status = "WORKS" if final_acc > 0.65 else "BROKEN"

    print(f"Init:  loss={losses[0]:.2f}, acc={accs[0]*100:.1f}%")
    print(f"Final: loss={final_loss:.2f}, acc={final_acc*100:.1f}%")
    print(f"=> {status}")
    print()

    results.append({
        'dim': slot_dim,
        'acc_final': final_acc,
        'status': status
    })

print("="*70)
print("RESULTS - All with ring_len=64")
print("="*70)
print()
print(f"{'Dim':<6} {'Capacity':<10} {'Final Acc':<12} {'Status'}")
print("-"*70)

for r in results:
    capacity = r['dim'] * RING_LEN
    print(f"{r['dim']:<6} {capacity:<10} {r['acc_final']*100:<12.1f} {r['status']}")

print()
working_count = sum(1 for r in results if r['status'] == 'WORKS')
print(f"Working: {working_count}/7")
print()

if working_count >= 5:
    print("HYPOTHESIS CONFIRMED:")
    print("  The problem is LARGE RINGS, not low dimensions!")
    print("  Small ring (64) allows pointer to converge.")
elif working_count == 0:
    print("HYPOTHESIS REJECTED:")
    print("  Even small rings don't work - deeper problem!")
else:
    print("MIXED RESULTS:")
    print("  Ring size is ONE factor, but not the only one.")

print("="*70)
