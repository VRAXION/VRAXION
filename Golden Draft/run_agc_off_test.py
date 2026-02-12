"""Proper AGC OFF Test - Using Production Training Infrastructure

This uses the REAL training pipeline (instnct_train_steps.py) with proper:
- Automatic checkpointing
- Error handling
- Telemetry logging
- Settings management

No more standalone fragmented scripts.
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, "S:/AI/Golden Code")
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

# ============================================================================
# CONFIGURATION - AGC OFF Test (TOT-H007)
# ============================================================================

# Task configuration
os.environ['VRX_SYNTH'] = '1'
os.environ['VRX_SYNTH_MODE'] = 'assoc_clean'
os.environ['VRX_BATCH_SIZE'] = '16'
os.environ['VRX_MAX_SAMPLES'] = '5000'
os.environ['VRX_SYNTH_LEN'] = '256'
os.environ['VRX_ASSOC_KEYS'] = '4'
os.environ['VRX_ASSOC_PAIRS'] = '3'
os.environ['VAR_RUN_SEED'] = '42'

# CPU Optimization (20 of 24 threads for ~90% utilization)
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'

# Training configuration via vraxion.settings
ROOT = "S:/AI/work/VRAXION_DEV/Golden Draft"
os.environ['VRX_ROOT'] = ROOT
os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'  # CRITICAL: Use VAR_COMPUTE_DEVICE, not VRX_DEVICE
os.environ['VRX_PRECISION'] = 'fp64'  # Double precision
os.environ['VRX_LR'] = '0.001'

# Pointer manual control - disable neural adaptive heads (updated 2026-02-10)
# Neural heads (inertia_head, deadzone_head, walk_head) were computing poor values
# Manual override forces use of module constants for Goldilocks orbit tuning
os.environ['VRX_PTR_INERTIA_OVERRIDE'] = '0.6'  # Disable neural heads, use constant

# AGC DISABLED - This is the critical setting
os.environ['VRX_AGC_ENABLED'] = '0'  # ACTUAL AGC disable flag (not grad_clip!)
os.environ['VRX_GRAD_CLIP'] = '0.0'  # Legacy gradient clipping (separate from AGC)
os.environ['VRX_UPDATE_SCALE'] = '1.0'

# Checkpointing (save every 100 steps)
os.environ['VRX_CHECKPOINT_PATH'] = os.path.join(ROOT, 'checkpoints/agc_off_test.pt')
os.environ['VRX_SAVE_EVERY_STEPS'] = '100'
os.environ['VRX_SAVE_HISTORY'] = '1'  # Keep history checkpoints

# Logging
os.environ['VRX_HEARTBEAT_STEPS'] = '10'  # Log every 10 steps
os.environ['VRX_DEBUG_STATS'] = '0'

# Trace logging for dashboard
os.environ['VRX_TRAIN_TRACE'] = '1'
os.environ['VRX_TRAIN_TRACE_PATH'] = os.path.join(ROOT, 'traces/current/train_steps_trace.jsonl')

# Dashboard log output
LOG_PATH = Path(ROOT) / "logs/probe/probe_live.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ['VRX_DASHBOARD_LOG'] = str(LOG_PATH)

import torch
torch.set_num_threads(20)  # PyTorch thread optimization

from tools.instnct_data import get_seq_mnist_loader
from tools.instnct_train_steps import train_steps
from vraxion.instnct.absolute_hallway import AbsoluteHallway

def main():
    print("=" * 70)
    print("AGC OFF TEST - Production Infrastructure")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Task:           assoc_clean (binary associative memory)")
    print(f"  AGC:            DISABLED (grad_clip=0.0)")
    print(f"  Steps:          2000")
    print(f"  Batch size:     16")
    print(f"  Device:         CPU")
    print(f"  Precision:      fp64 (double)")
    print(f"  CPU threads:    20/24 (~90% utilization)")
    print(f"  Checkpoints:    Every 100 steps -> {os.environ['VRX_CHECKPOINT_PATH']}")
    print(f"  Dashboard log:  {LOG_PATH}")
    print()
    print("=" * 70)
    print()

    # Create checkpoint directory
    Path(os.environ['VRX_CHECKPOINT_PATH']).parent.mkdir(parents=True, exist_ok=True)

    # Create trace directory
    Path(os.environ['VRX_TRAIN_TRACE_PATH']).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
    print(f"Dataset loaded: num_classes={num_classes}")
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

    # Check if we can resume from checkpoint
    checkpoint_path = Path(os.environ['VRX_CHECKPOINT_PATH'])
    if checkpoint_path.exists():
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from step {start_step}")
        print()
    else:
        print("No checkpoint found, starting from scratch")
        print()

    # Run training using production infrastructure
    print("Starting training with production infrastructure...")
    print("This includes:")
    print("  [+] Automatic checkpointing")
    print("  [+] Error handling")
    print("  [+] Telemetry logging")
    print("  [+] Gradient monitoring")
    print()

    try:
        result = train_steps(
            model=model,
            loader=train_loader,
            steps=2000,
            dataset_name="assoc_clean",
            model_name="agc_off_test"
        )

        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print()
        print("Results:")
        print(f"  Final loss: {result.get('loss', 'N/A')}")
        print(f"  Total time: {result.get('elapsed_time', 'N/A')}s")
        print()
        print("Next steps:")
        print("  1. Check dashboard at http://localhost:8501")
        print("  2. Run correlation analysis:")
        print("     python tools/_scratch/analyze_pointer_correlation.py")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Training interrupted by user")
        print("=" * 70)
        print()
        print("Checkpoint saved - you can resume with:")
        print(f"  python {__file__}")
        print()
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR during training")
        print("=" * 70)
        print(f"Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()
        print(f"Latest checkpoint: {os.environ['VRX_CHECKPOINT_PATH']}")
        print()
        raise

if __name__ == "__main__":
    main()
