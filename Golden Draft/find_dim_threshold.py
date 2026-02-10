"""Find the minimum slot_dim that actually works."""

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

print("="*70)
print("DIMENSION SWEEP - Find the minimum working slot_dim")
print("="*70)
print()
print("Testing: 1, 2, 4, 8, 16, 32, 64 dimensions")
print("Task: Binary (2-class) COPY")
print("Steps: 50 per dimension")
print("="*70)
print()

# Binary task
x, y, num_classes = task_copy(n_samples=100, seq_len=16, vocab_size=2)
x = x.to(torch.float64)
y = y.long()

results = []

for slot_dim in [1, 2, 4, 8, 16, 32, 64]:
    # Capacity-matched ring length: 4096 total storage units
    ring_len = 4096 // slot_dim

    print(f"\n{'='*70}")
    print(f"Testing slot_dim={slot_dim}, ring_len={ring_len}")
    print(f"Capacity: {slot_dim} x {ring_len} = {slot_dim * ring_len} units")
    print(f"{'='*70}")

    # Create model
    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=ring_len, slot_dim=slot_dim)
    model = model.to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    # Train for 50 steps
    losses = []
    accs = []

    for step in range(50):
        optimizer.zero_grad()

        # Forward
        logits, aux_loss = model(x)
        loss = criterion(logits, y) + aux_loss

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

    # Report
    final_acc = accs[-1]
    final_loss = losses[-1]

    status = "WORKS" if final_acc > 0.7 else "BROKEN"

    print(f"Initial: loss={losses[0]:.1f}, acc={accs[0]*100:.1f}%")
    print(f"Final:   loss={final_loss:.1f}, acc={final_acc*100:.1f}%")
    print(f"Status:  {status}")

    results.append({
        'dim': slot_dim,
        'ring_len': ring_len,
        'loss_init': losses[0],
        'loss_final': final_loss,
        'acc_init': accs[0],
        'acc_final': final_acc,
        'status': status
    })

# Summary table
print("\n" + "="*70)
print("SUMMARY - Dimension Threshold Search")
print("="*70)
print()
print(f"{'Dim':<6} {'Ring':<6} {'Loss0':<8} {'LossF':<8} {'Acc0':<6} {'AccF':<6} {'Status'}")
print("-"*70)

for r in results:
    print(f"{r['dim']:<6} {r['ring_len']:<6} {r['loss_init']:<8.1f} {r['loss_final']:<8.1f} "
          f"{r['acc_init']*100:<6.1f} {r['acc_final']*100:<6.1f} {r['status']}")

print()
print("="*70)
print("THRESHOLD ANALYSIS:")
print("="*70)

# Find threshold
working = [r for r in results if r['status'] == 'WORKS']
broken = [r for r in results if r['status'] == 'BROKEN']

if working:
    min_working = min(working, key=lambda r: r['dim'])
    print(f"Minimum working dimension: {min_working['dim']}")
    print(f"  Final accuracy: {min_working['acc_final']*100:.1f}%")

    if broken:
        max_broken = max(broken, key=lambda r: r['dim'])
        print(f"Maximum broken dimension: {max_broken['dim']}")
        print()
        print(f"THRESHOLD: Between {max_broken['dim']} and {min_working['dim']} dimensions")
else:
    print("ALL DIMENSIONS BROKEN (even 64-dim!)")
    print("This suggests a deeper architectural problem.")

print("="*70)
