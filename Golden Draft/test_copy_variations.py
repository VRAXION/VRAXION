"""Test COPY task with different parameters to find what works.

Hypothesis: Model needs longer sequences or fewer output classes.
"""

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

import torch
torch.set_num_threads(20)

from tools.diagnostic_tasks import task_copy
from tools.instnct_train_steps import train_steps
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

# Test configurations
TESTS = [
    {
        "name": "BASELINE",
        "seq_len": 16,
        "vocab_size": 10,
        "desc": "Original (FAILED at 9.2%)"
    },
    {
        "name": "LONG_SEQ",
        "seq_len": 256,
        "vocab_size": 10,
        "desc": "Match assoc_clean seq_len"
    },
    {
        "name": "BINARY",
        "seq_len": 16,
        "vocab_size": 2,
        "desc": "Match assoc_clean output classes"
    },
    {
        "name": "BOTH",
        "seq_len": 256,
        "vocab_size": 2,
        "desc": "Match BOTH seq_len AND classes"
    },
]

results = []

for test_config in TESTS:
    name = test_config['name']
    seq_len = test_config['seq_len']
    vocab_size = test_config['vocab_size']
    desc = test_config['desc']

    print("\n" + "="*70)
    print(f"TEST: {name}")
    print("="*70)
    print(f"Description:    {desc}")
    print(f"Seq length:     {seq_len}")
    print(f"Vocab size:     {vocab_size} classes")
    print(f"Random chance:  {100/vocab_size:.1f}%")
    print("="*70)
    print()

    # Set log path
    LOG_PATH = Path(ROOT) / f"logs/probe/copy_test_{name.lower()}.log"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ['VRX_DASHBOARD_LOG'] = str(LOG_PATH)

    # Generate data
    print(f"Generating data...")
    x, y, num_classes = task_copy(n_samples=5000, seq_len=seq_len, vocab_size=vocab_size)
    print(f"Generated: {x.shape[0]} samples, seq_len={x.shape[1]}, classes={num_classes}")

    # SANITY CHECK: Verify labels span full range
    print(f"\nLabel distribution check:")
    print(f"  Min label: {y.min().item()}")
    print(f"  Max label: {y.max().item()}")
    print(f"  Unique labels: {sorted(y.unique().tolist())}")
    print(f"  Expected range: 0-{vocab_size-1}")
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
    print(f"Training for 200 steps...")
    print()

    result = train_steps(
        model=model,
        loader=train_loader,
        steps=200,
        dataset_name=f"copy_{name.lower()}",
        model_name=f"test_copy_{name.lower()}"
    )

    final_acc = result.get('final_accuracy', 0.0) * 100
    random_baseline = 100 / vocab_size
    improvement = final_acc - random_baseline

    print()
    print("="*70)
    print(f"RESULTS: {name}")
    print("="*70)
    print(f"Accuracy:       {final_acc:.1f}%")
    print(f"Random chance:  {random_baseline:.1f}%")
    print(f"Improvement:    {improvement:+.1f} points")
    print()

    if improvement > 20:
        status = "[OK] WORKS!"
    elif improvement > 5:
        status = "[~] Marginal"
    else:
        status = "[!!] BROKEN"

    print(f"Status: {status}")
    print()

    results.append({
        "name": name,
        "desc": desc,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "accuracy": final_acc,
        "random": random_baseline,
        "improvement": improvement,
        "status": status
    })

# Final summary
print("\n" + "="*70)
print("SUMMARY: COPY TASK VARIATIONS")
print("="*70)
print()
print(f"{'Test':<15} {'Seq':<6} {'Classes':<8} {'Acc':<8} {'Random':<8} {'Δ':<8} {'Status':<15}")
print("-"*70)

for r in results:
    print(f"{r['name']:<15} {r['seq_len']:<6} {r['vocab_size']:<8} {r['accuracy']:>6.1f}% {r['random']:>6.1f}% {r['improvement']:>+6.1f}  {r['status']:<15}")

print()
print("="*70)
print("INTERPRETATION:")
print()

# Find which factors matter
long_seq_works = any(r['seq_len'] == 256 and r['improvement'] > 20 for r in results)
binary_works = any(r['vocab_size'] == 2 and r['improvement'] > 20 for r in results)
both_needed = results[3]['improvement'] > 20 and not (long_seq_works or binary_works)

if both_needed:
    print("=> Model needs BOTH long sequences AND binary output")
    print("   This suggests SEVERE capacity limitations")
elif long_seq_works:
    print("=> Model needs LONG sequences (256+) to learn")
    print("   Short sequences (16) don't give pointer time to converge")
elif binary_works:
    print("=> Model needs SIMPLE outputs (2 classes)")
    print("   10-class output space is too hard for this architecture")
else:
    print("=> Model FAILS on ALL variations")
    print("   Core architecture is fundamentally broken")

print()
