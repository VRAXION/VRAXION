"""Run diagnostic task ladder to find model breaking point.

Runs Levels 1-10 sequentially, each with fresh model instance.
Reports pass/fail for each level and identifies exact breaking point.
"""

import os
import sys
from pathlib import Path

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

# Pointer manual steering (from previous fixes)
os.environ['VRX_PTR_INERTIA_OVERRIDE'] = '0.6'

# AGC disabled
os.environ['VRX_AGC_ENABLED'] = '0'
os.environ['VRX_GRAD_CLIP'] = '0.0'
os.environ['VRX_UPDATE_SCALE'] = '1.0'

# Learning config
os.environ['VRX_LR'] = '0.001'

# Disable synth mode (we'll generate data directly)
os.environ['VRX_SYNTH'] = '0'

# Logging
os.environ['VRX_HEARTBEAT_STEPS'] = '10'
os.environ['VRX_DEBUG_STATS'] = '0'

# Dashboard log
LOG_PATH = Path(ROOT) / "logs/probe/diagnostic_ladder.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ['VRX_DASHBOARD_LOG'] = str(LOG_PATH)

import torch
torch.set_num_threads(20)

from tools.diagnostic_tasks import *
from tools.instnct_train_steps import train_steps
from tools.instnct_data import get_seq_mnist_loader  # For L7, L9 (assoc tasks)
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# LEVEL DEFINITIONS
# ============================================================================

LEVELS = [
    {
        "name": "L1_COPY",
        "fn": task_copy,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10},
        "steps": 100,
        "target": 98,
        "desc": "Copy last token (sanity check)"
    },
    {
        "name": "L2_1BACK",
        "fn": task_nback,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10, "n_back": 1},
        "steps": 150,
        "target": 95,
        "desc": "Predict token 1 position back"
    },
    {
        "name": "L3_PARITY",
        "fn": task_parity,
        "kwargs": {"n_samples": 5000, "seq_len": 16},
        "steps": 200,
        "target": 85,
        "desc": "Count 1s, output even/odd"
    },
    {
        "name": "L4_COUNT",
        "fn": task_count_range,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 10, "target_token": 5, "bins": [2, 5]},
        "steps": 250,
        "target": 75,
        "desc": "Count target token, classify into bins"
    },
    {
        "name": "L5_FIRSTLAST",
        "fn": task_first_last_match,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 10},
        "steps": 250,
        "target": 70,
        "desc": "Check if first == last token"
    },
    {
        "name": "L6_MAJORITY",
        "fn": task_majority_vote,
        "kwargs": {"n_samples": 5000, "seq_len": 32, "vocab_size": 5},
        "steps": 300,
        "target": 65,
        "desc": "Find most common token"
    },
    {
        "name": "L7_ASSOC",
        "fn": None,  # Use existing assoc_clean
        "kwargs": None,
        "steps": 300,
        "target": 64,
        "desc": "Binary associative memory (current baseline)"
    },
    {
        "name": "L8_NESTED",
        "fn": task_nested_parity,
        "kwargs": {"n_samples": 5000, "seq_len": 32},
        "steps": 300,
        "target": 55,
        "desc": "Parity(first_half) XOR Parity(second_half)"
    },
    {
        "name": "L9_ASSOC_BYTE",
        "fn": None,  # Use existing assoc_byte
        "kwargs": None,
        "steps": 300,
        "target": 40,
        "desc": "Multi-class associative memory (16-way)"
    },
    {
        "name": "L10_REVERSAL",
        "fn": task_reversal,
        "kwargs": {"n_samples": 5000, "seq_len": 16, "vocab_size": 10},
        "steps": 300,
        "target": 30,
        "desc": "Predict first token (sequence reversal)"
    },
]


def create_dataloader(x, y, batch_size=16, shuffle=True):
    """Create PyTorch DataLoader from tensors."""
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_level(level_config):
    """Run a single diagnostic level.

    Args:
        level_config: Level dictionary with name, fn, kwargs, steps, target

    Returns:
        dict with level name, target, actual accuracy, status (PASS/FAIL)
    """
    name = level_config['name']
    fn = level_config['fn']
    kwargs = level_config['kwargs']
    steps = level_config['steps']
    target = level_config['target']
    desc = level_config['desc']

    print(f"\n{'='*70}")
    print(f"LEVEL: {name}")
    print(f"Task: {desc}")
    print(f"Target accuracy: {target}%")
    print(f"Steps: {steps}")
    print(f"{'='*70}\n")

    # Generate data
    if fn is not None:
        # Custom diagnostic task
        x, y, num_classes = fn(**kwargs)
        print(f"Generated {x.shape[0]} samples, {x.shape[1]} seq_len, {num_classes} classes")

        # Create DataLoader
        train_loader = create_dataloader(x, y, batch_size=16, shuffle=True)

    else:
        # Use existing assoc_clean/assoc_byte from instnct_data
        if name == "L7_ASSOC":
            # assoc_clean
            os.environ['VRX_SYNTH'] = '1'
            os.environ['VRX_SYNTH_MODE'] = 'assoc_clean'
            os.environ['VRX_MAX_SAMPLES'] = '5000'
            os.environ['VRX_SYNTH_LEN'] = '256'
            os.environ['VRX_ASSOC_KEYS'] = '4'
            os.environ['VRX_ASSOC_PAIRS'] = '3'
            os.environ['VRX_BATCH_SIZE'] = '16'
            train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
            print(f"Loaded assoc_clean dataset, {num_classes} classes")

        elif name == "L9_ASSOC_BYTE":
            # assoc_byte
            os.environ['VRX_SYNTH'] = '1'
            os.environ['VRX_SYNTH_MODE'] = 'assoc_byte'
            os.environ['VRX_MAX_SAMPLES'] = '5000'
            os.environ['VRX_SYNTH_LEN'] = '256'
            os.environ['VRX_ASSOC_KEYS'] = '4'
            os.environ['VRX_ASSOC_PAIRS'] = '3'
            os.environ['VRX_ASSOC_VAL_RANGE'] = '16'
            os.environ['VRX_BATCH_SIZE'] = '16'
            train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
            print(f"Loaded assoc_byte dataset, {num_classes} classes")

        else:
            raise ValueError(f"Unknown level without task function: {name}")

    # Create fresh model
    print(f"Creating model...")
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
    print(f"Training for {steps} steps...")
    try:
        result = train_steps(
            model=model,
            loader=train_loader,
            steps=steps,
            dataset_name=name,
            model_name="diagnostic_ladder"
        )

        # Extract final accuracy (modified train_steps to return this)
        final_acc = result.get('final_accuracy', 0.0) * 100

    except Exception as e:
        print(f"\n[ERROR] Training failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        final_acc = 0.0

    # Determine pass/fail
    passed = "PASS" if final_acc >= target else "FAIL"

    print(f"\n{'='*70}")
    print(f"[{passed}] {name}: {final_acc:.1f}% (target {target}%)")
    print(f"{'='*70}\n")

    return {
        "level": name,
        "desc": desc,
        "target": target,
        "actual": final_acc,
        "status": passed
    }


def main():
    """Run full diagnostic ladder."""
    print("=" * 70)
    print("DIAGNOSTIC TASK LADDER")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Device:         CPU (fp64)")
    print(f"  Model:          AbsoluteHallway (2,820 params)")
    print(f"  Pointer:        Manual steering (inertia=0.6)")
    print(f"  AGC:            DISABLED")
    print(f"  Levels:         10 (L1-L10)")
    print(f"  Dashboard log:  {LOG_PATH}")
    print()
    print("=" * 70)
    print()

    results = []

    for level_config in LEVELS:
        result = run_level(level_config)
        results.append(result)

    # Final report
    print("\n" + "="*70)
    print("DIAGNOSTIC LADDER RESULTS")
    print("="*70)
    print()
    print(f"{'Level':<15} {'Description':<40} {'Target':>6} {'Actual':>7} {'Status':>6}")
    print("-"*70)

    for r in results:
        status_icon = "[OK]" if r['status'] == "PASS" else "[!!]"
        print(f"{r['level']:<15} {r['desc']:<40} {r['target']:>5}% {r['actual']:>6.1f}% {status_icon:>6}")

    print()
    print("="*70)

    # Find breaking point
    breaking_level = None
    for i, r in enumerate(results):
        if r['status'] == "FAIL":
            breaking_level = (i + 1, r['level'])
            break

    if breaking_level:
        print(f"\n>> MODEL BREAKS AT: Level {breaking_level[0]} ({breaking_level[1]})")
        print()
        print("Interpretation:")
        level_num = breaking_level[0]
        if level_num == 1:
            print("  Model is COMPLETELY BROKEN (fails sanity check).")
            print("  Action: Full rewrite needed (diamond level code).")
        elif level_num <= 3:
            print("  Model has basic issues with simple memory/counting.")
            print("  Action: Debug core architecture before testing harder tasks.")
        elif level_num <= 6:
            print("  Model capacity ceiling reached.")
            print("  Action: Either increase model size OR accept current ceiling.")
        else:
            print("  Model is healthy! Breaking on hard tasks is expected.")
            print("  Action: Test on bigger model to see if it can scale further.")
    else:
        print("\n>> MODEL PASSES ALL LEVELS!")
        print()
        print("Interpretation:")
        print("  Model is exceptionally strong for its size (2,820 params).")
        print("  Action: Consider testing on even harder tasks or bigger datasets.")

    print()
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Review dashboard at http://localhost:8501")
    print("  2. Check individual level logs for gradient stability")
    print("  3. If needed, run tune_task_difficulty.py for fine-grained analysis")
    print()


if __name__ == "__main__":
    main()
