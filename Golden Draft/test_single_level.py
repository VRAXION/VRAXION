"""Run a single diagnostic level interactively.

Usage:
    python test_single_level.py L1_COPY
    python test_single_level.py L3_PARITY
    etc.
"""

import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python test_single_level.py <level_name>")
    print()
    print("Available levels:")
    print("  L1_COPY      - Copy last token (sanity check)")
    print("  L2_1BACK     - Predict 1 position back")
    print("  L3_PARITY    - Count 1s, even/odd")
    print("  L4_COUNT     - Count target, classify bins")
    print("  L5_FIRSTLAST - First == last token?")
    print("  L6_MAJORITY  - Most common token")
    print("  L7_ASSOC     - Assoc clean (current baseline)")
    print("  L8_NESTED    - Nested parity (XOR)")
    print("  L9_ASSOC_BYTE - Assoc byte (16-way)")
    print("  L10_REVERSAL - Predict first token")
    sys.exit(1)

LEVEL = sys.argv[1].upper()

# Add paths
sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ROOT = "S:/AI/work/VRAXION_DEV/Golden Draft"
os.environ['VRX_ROOT'] = ROOT

# Device and precision
os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'
os.environ['VRX_PRECISION'] = 'fp64'

# CPU optimization
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'

# Pointer manual steering
os.environ['VRX_PTR_INERTIA_OVERRIDE'] = '0.6'

# AGC disabled
os.environ['VRX_AGC_ENABLED'] = '0'
os.environ['VRX_GRAD_CLIP'] = '0.0'
os.environ['VRX_UPDATE_SCALE'] = '1.0'

# Learning config
os.environ['VRX_LR'] = '0.001'

# Disable synth mode initially
os.environ['VRX_SYNTH'] = '0'

# Logging
os.environ['VRX_HEARTBEAT_STEPS'] = '10'
os.environ['VRX_DEBUG_STATS'] = '0'

# Dashboard log
LOG_PATH = Path(ROOT) / f"logs/probe/{LEVEL.lower()}_test.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ['VRX_DASHBOARD_LOG'] = str(LOG_PATH)

import torch
torch.set_num_threads(20)

from tools.diagnostic_tasks import *
from tools.instnct_train_steps import train_steps
from tools.instnct_data import get_seq_mnist_loader
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# LEVEL CONFIGURATIONS
# ============================================================================

LEVEL_CONFIGS = {
    "L1_COPY": {
        "fn": task_copy,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10},
        "steps": 100,
        "target": 98,
        "desc": "Copy last token"
    },
    "L2_1BACK": {
        "fn": task_nback,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10, "n_back": 1},
        "steps": 150,
        "target": 95,
        "desc": "Predict 1 position back"
    },
    "L3_PARITY": {
        "fn": task_parity,
        "kwargs": {"n_samples": 5000, "seq_len": 16},
        "steps": 200,
        "target": 85,
        "desc": "Count 1s, even/odd"
    },
    "L4_COUNT": {
        "fn": task_count_range,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 10, "target_token": 5},
        "steps": 250,
        "target": 75,
        "desc": "Count target token, bins"
    },
    "L5_FIRSTLAST": {
        "fn": task_first_last_match,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 10},
        "steps": 250,
        "target": 70,
        "desc": "First == last?"
    },
    "L6_MAJORITY": {
        "fn": task_majority_vote,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 5},
        "steps": 300,
        "target": 65,
        "desc": "Most common token"
    },
    "L7_ASSOC": {
        "fn": None,  # Use existing
        "kwargs": None,
        "steps": 300,
        "target": 64,
        "desc": "Assoc clean (baseline)"
    },
    "L8_NESTED": {
        "fn": task_nested_parity,
        "kwargs": {"n_samples": 5000, "seq_len": 32},
        "steps": 300,
        "target": 55,
        "desc": "Nested parity (XOR)"
    },
    "L9_ASSOC_BYTE": {
        "fn": None,  # Use existing
        "kwargs": None,
        "steps": 300,
        "target": 40,
        "desc": "Assoc byte (16-way)"
    },
    "L10_REVERSAL": {
        "fn": task_reversal,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10},
        "steps": 300,
        "target": 30,
        "desc": "Predict first token"
    },
}

if LEVEL not in LEVEL_CONFIGS:
    print(f"Unknown level: {LEVEL}")
    print(f"Available: {', '.join(LEVEL_CONFIGS.keys())}")
    sys.exit(1)

config = LEVEL_CONFIGS[LEVEL]

print("=" * 70)
print(f"TESTING LEVEL: {LEVEL}")
print("=" * 70)
print()
print(f"Task:           {config['desc']}")
print(f"Target:         {config['target']}% accuracy")
print(f"Steps:          {config['steps']}")
print(f"Device:         CPU (fp64)")
print(f"Model:          AbsoluteHallway")
print(f"Log:            {LOG_PATH}")
print()
print("=" * 70)
print()

# Generate or load data
if config['fn'] is not None:
    # Custom diagnostic task
    print("Generating data...")
    x, y, num_classes = config['fn'](**config['kwargs'])
    print(f"Generated: {x.shape[0]} samples, seq_len={x.shape[1]}, classes={num_classes}")

    # Create DataLoader
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
else:
    # Use existing assoc tasks
    if LEVEL == "L7_ASSOC":
        os.environ['VRX_SYNTH'] = '1'
        os.environ['VRX_SYNTH_MODE'] = 'assoc_clean'
        os.environ['VRX_MAX_SAMPLES'] = '5000'
        os.environ['VRX_SYNTH_LEN'] = '256'
        os.environ['VRX_ASSOC_KEYS'] = '4'
        os.environ['VRX_ASSOC_PAIRS'] = '3'
        os.environ['VRX_BATCH_SIZE'] = '16'
        train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
        print(f"Loaded assoc_clean: classes={num_classes}")
    elif LEVEL == "L9_ASSOC_BYTE":
        os.environ['VRX_SYNTH'] = '1'
        os.environ['VRX_SYNTH_MODE'] = 'assoc_byte'
        os.environ['VRX_MAX_SAMPLES'] = '5000'
        os.environ['VRX_SYNTH_LEN'] = '256'
        os.environ['VRX_ASSOC_KEYS'] = '4'
        os.environ['VRX_ASSOC_PAIRS'] = '3'
        os.environ['VRX_ASSOC_VAL_RANGE'] = '16'
        os.environ['VRX_BATCH_SIZE'] = '16'
        train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
        print(f"Loaded assoc_byte: classes={num_classes}")

print()

# Create model
print("Creating model...")
model = AbsoluteHallway(
    input_dim=1,
    num_classes=num_classes,
    ring_len=64,
    slot_dim=64,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized: {total_params:,} parameters")
print()

# Train
print(f"Training for {config['steps']} steps...")
print("(Watch the output below - you'll see accuracy every 10 steps)")
print()
print("=" * 70)
print()

result = train_steps(
    model=model,
    loader=train_loader,
    steps=config['steps'],
    dataset_name=LEVEL,
    model_name=f"test_{LEVEL.lower()}"
)

# Results
final_acc = result.get('final_accuracy', 0.0) * 100
target = config['target']
passed = "PASS" if final_acc >= target else "FAIL"

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"Level:          {LEVEL}")
print(f"Target:         {target}%")
print(f"Actual:         {final_acc:.1f}%")
print(f"Status:         {passed}")
print()

if passed == "PASS":
    print("[OK] Model PASSED this level!")
    print(f"     Exceeded target by {final_acc - target:.1f} percentage points")
else:
    print("[!!] Model FAILED this level!")
    print(f"     Fell short by {target - final_acc:.1f} percentage points")

print()
print("=" * 70)
print()

if LEVEL == "L1_COPY" and passed == "FAIL":
    print("WARNING: Model failed the sanity check (L1_COPY)!")
    print("This indicates the model is fundamentally broken.")
    print("Recommendation: Stop testing and investigate core architecture.")
elif LEVEL == "L1_COPY":
    print("Good! Sanity check passed. Model can learn trivial tasks.")
    print("Next: python test_single_level.py L2_1BACK")
elif passed == "PASS":
    print(f"Good! Try the next level to find the breaking point.")
else:
    print(f"Breaking point found at {LEVEL}.")
    print("This tells you the model's capacity ceiling.")
