"""Quick test: Run probe for 10 steps to verify it works"""

import sys
import os

# Add paths
sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

# Configure environment
os.environ['VRX_SYNTH'] = '1'
os.environ['VRX_SYNTH_MODE'] = 'assoc_clean'
os.environ['VRX_BATCH_SIZE'] = '16'
os.environ['VRX_MAX_SAMPLES'] = '500'  # Smaller for quick test
os.environ['VRX_SYNTH_LEN'] = '256'
os.environ['VRX_ASSOC_KEYS'] = '4'
os.environ['VRX_ASSOC_PAIRS'] = '3'
os.environ['VAR_RUN_SEED'] = '42'

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from vraxion.instnct.absolute_hallway import AbsoluteHallway
from tools.instnct_data import get_seq_mnist_loader

print("=" * 60)
print("QUICK TEST: Probe verification (10 steps)")
print("=" * 60)
print()

# Load data
print("Loading assoc_clean dataset...")
train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
print(f"Dataset loaded: num_classes={num_classes}")

# Get first batch to check shapes
train_iter = iter(train_loader)
sample_batch = next(train_iter)
inputs, targets = sample_batch
print(f"Input shape:  {inputs.shape}")
print(f"Target shape: {targets.shape}")
print(f"Target range: [{targets.min().item()}, {targets.max().item()}]")
print()

# Reset iterator
train_iter = iter(train_loader)

# Create model
model = AbsoluteHallway(
    input_dim=1,
    num_classes=num_classes,
    ring_len=64,
    slot_dim=64,
).to("cpu")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} parameters")
print()

# Setup training
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Running 10 training steps...")
print()

for step in range(1, 11):
    step_start = time.time()

    # Get batch
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    inputs, targets = batch
    inputs = inputs.to("cpu")
    targets = targets.to("cpu")

    # Train
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    # Accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()

    step_time = time.time() - step_start

    print(f"Step {step:2d}: loss={loss.item():.4f}, acc={acc:.2%}, time={step_time:.2f}s")

print()
print("=" * 60)
print("TEST PASSED: Probe runs correctly")
print("=" * 60)
print()
print("Next: Run full probe with 2000 steps")
print("  python tools/_scratch/long_run_probe_agc_off.py")
