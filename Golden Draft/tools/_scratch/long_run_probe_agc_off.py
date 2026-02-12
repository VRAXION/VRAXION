"""Long-run Probe: AGC OFF - Extended training observation

Validates TOT-H007 finding that AGC is unnecessary for single-being training.
Runs extended training (2000 steps) with AGC disabled to observe long-term
performance and stability.

THIS VERSION USES THE REAL ASSOC_CLEAN DATASET (not fake random noise).

Dashboard: http://localhost:8501
Launch dashboard first:
    python -m streamlit run tools/live_dashboard.py -- --log logs/probe/probe_live.log
"""

import sys
import os

# Add Golden Code to path FIRST (before any vraxion imports)
sys.path.insert(0, "S:/AI/Golden Code")

# Add Golden Draft tools to path (for instnct_data)
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Golden Draft")

import time
from pathlib import Path

# Configure assoc_clean task BEFORE importing data loader
os.environ['VRX_SYNTH'] = '1'
os.environ['VRX_SYNTH_MODE'] = 'assoc_clean'
os.environ['VRX_BATCH_SIZE'] = '16'
os.environ['VRX_MAX_SAMPLES'] = '5000'
os.environ['VRX_SYNTH_LEN'] = '256'
os.environ['VRX_ASSOC_KEYS'] = '4'
os.environ['VRX_ASSOC_PAIRS'] = '3'
os.environ['VAR_RUN_SEED'] = '42'  # Training seed

# Enable trace logging for per-sequence loss analysis
os.environ['VRX_TRAIN_TRACE'] = '1'
os.environ['VRX_TRAIN_TRACE_PATH'] = 'traces/current/train_steps_trace.jsonl'

# CPU Performance Optimization: Use ~90% of Ryzen threads (20 of 24)
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'

import torch
import torch.nn as nn
import torch.optim as optim

# Set PyTorch to use 20 threads (leaving 4 for OS/other processes)
torch.set_num_threads(20)

from vraxion.instnct.absolute_hallway import AbsoluteHallway
from tools.instnct_data import get_seq_mnist_loader


def main():
    print("=" * 70)
    print("LONG-RUN PROBE: AGC OFF (TOT-H007 Validation) - CPU OPTIMIZED")
    print("=" * 70)
    print()
    print("CRITICAL FIX: Now using REAL assoc_clean dataset (not random noise)")
    print()
    print("Configuration:")
    print("  Task:       assoc_clean (binary associative memory)")
    print("  AGC:        DISABLED (testing TOT-H007 finding)")
    print("  Steps:      2000 (extended observation)")
    print("  Batch size: 16")
    print("  Keys:       4 distinct keys")
    print("  Pairs:      3 key-value pairs per sequence")
    print("  Device:     CPU")
    print("  CPU Threads: 20/24 (targeting ~90% utilization)")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print()
    print("Dashboard:    http://localhost:8501")
    print("Log file:     logs/probe/probe_live.log")
    print("Trace file:   traces/current/train_steps_trace.jsonl")
    print()
    print("-" * 70)
    print()

    # Setup
    device = "cpu"
    log_path = Path("S:/AI/work/VRAXION_DEV/Golden Draft/logs/probe/probe_live.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup trace directory
    trace_path = Path("S:/AI/work/VRAXION_DEV/Golden Draft/traces/current")
    trace_path.mkdir(parents=True, exist_ok=True)

    # Clear log file
    log_path.write_text("")

    # Get real assoc_clean data loader (training seed=42)
    print("Loading real assoc_clean dataset (seed=42)...")
    train_loader, num_classes, _ = get_seq_mnist_loader(train=True)
    print(f"Dataset loaded: num_classes={num_classes}")

    # Verify it's binary classification
    if num_classes != 2:
        print(f"WARNING: Expected num_classes=2 for assoc_clean, got {num_classes}")

    # Create iterator
    train_iter = iter(train_loader)

    # Inspect first batch to verify input shape
    sample_batch = next(train_iter)
    sample_inputs, sample_targets = sample_batch
    print(f"Input shape:  {sample_inputs.shape}  (expected: [batch, seq, 1])")
    print(f"Target shape: {sample_targets.shape}  (expected: [batch])")
    print(f"Target range: [{sample_targets.min().item()}, {sample_targets.max().item()}]  (expected: [0, 1])")
    print()

    # Reset iterator
    train_iter = iter(train_loader)

    # Model configuration (matching wiki-documented config)
    # NOTE: The plan acknowledges ring_len=64, slot_dim=64 gives ~9,777 params (not 2,820)
    # The "2,820 params" claim in docs is an error - we use the actual working config
    input_dim = 1       # assoc_clean outputs [batch, seq, 1]
    ring_len = 64       # From wiki-documented config
    slot_dim = 64       # From wiki-documented config

    model = AbsoluteHallway(
        input_dim=input_dim,
        num_classes=num_classes,  # Should be 2 for binary classification
        ring_len=ring_len,
        slot_dim=slot_dim,
    ).to(device)

    # Count params (print ACTUAL count, no lies)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {total_params:,} parameters")
    print(f"  (ring_len={ring_len}, slot_dim={slot_dim}, input_dim={input_dim}, num_classes={num_classes})")
    print()

    # Debug: Check if satiety is enabled
    print("Checking model satiety configuration...")
    if hasattr(model, 'satiety_enabled'):
        print(f"  Satiety enabled: {model.satiety_enabled}")
    if hasattr(model, 'SATIETY_THRESH'):
        print(f"  Satiety threshold: {model.SATIETY_THRESH}")
    print()

    # Optimizer (standard Adam)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training config
    num_steps = 2000
    log_interval = 10

    # Tracking
    losses = []
    accs = []
    ma_window = 100  # Moving average window

    print(f"Starting training: {num_steps} steps")
    print(f"AGC: DISABLED (no gradient normalization)")
    print(f"Expected: ~1.5s/step (real task), ~64.8% final accuracy")
    print()

    start_time = time.time()

    for step in range(1, num_steps + 1):
        try:
            step_start = time.time()

            # Get batch from real data loader
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            optimizer.zero_grad()
            logits, move_penalty = model(inputs)

            # Extract pointer telemetry (already computed by model)
            flip_rate = getattr(model, 'ptr_flip_rate', 0.0) or 0.0
            orbit = getattr(model, 'ptr_orbit', 0.0) or 0.0
            residual = getattr(model, 'ptr_residual_mean', 0.0) or 0.0
            anchor_clicks = getattr(model, 'ptr_anchor_clicks', 0) or 0
            max_dwell = getattr(model, 'ptr_max_dwell', 0.0) or 0.0
            mean_dwell = getattr(model, 'ptr_mean_dwell', 0.0) or 0.0

            # Extract control state
            inertia = getattr(model, 'ptr_inertia', 0.0) or 0.0
            deadzone = getattr(model, 'ptr_deadzone', 0.0) or 0.0
            walk_prob = getattr(model, 'ptr_walk_prob', 0.0) or 0.0

            # Loss
            loss = criterion(logits, targets)

            # Backward (NO AGC - plain Adam optimizer)
            loss.backward()

            # Compute gradient norm for monitoring
            total_grad_norm = 0.0
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                total_grad_norm = total_grad_norm ** 0.5

            # Optimizer step (no AGC intervention)
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == targets).float().mean().item()

            # Track
            losses.append(loss.item())
            accs.append(acc)

            # Compute moving averages
            loss_ma = sum(losses[-ma_window:]) / min(len(losses), ma_window)
            acc_ma = sum(accs[-ma_window:]) / min(len(accs), ma_window)

            step_time = time.time() - step_start

            # Dashboard-compatible log format
            # Format: step N | loss X.XXXXXX | acc=X.XXXX RD:X.XXXX traction=X.XXXX shard=0/0
            log_line = (
                f"step {step} | loss {loss.item():.6f} | "
                f"acc={acc:.4f} acc_ma={acc_ma:.4f} "
                f"gnorm={total_grad_norm:.3f} "
                f"s_per_step={step_time:.3f} "
                f"flip_rate={flip_rate:.4f} orbit={orbit:.1f} residual={residual:.4f} anchor_clicks={anchor_clicks} "
                f"inertia={inertia:.3f} deadzone={deadzone:.4f} walk={walk_prob:.3f} "
                f"agc=OFF shard=0/0"
            )

            # Write to log file (dashboard reads this)
            with open(log_path, "a") as f:
                f.write(log_line + "\n")

            # Console output
            if step % log_interval == 0 or step == 1:
                elapsed = time.time() - start_time
                eta = (elapsed / step) * (num_steps - step)
                print(
                    f"[{step:4d}/{num_steps}] "
                    f"loss={loss.item():.4f} (ma={loss_ma:.4f}) | "
                    f"acc={acc:.2%} (ma={acc_ma:.2%}) | "
                    f"gnorm={total_grad_norm:.2f} | "
                    f"{step_time:.2f}s/step | "
                    f"ETA={eta/60:.1f}m"
                )

        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print(f"TRAINING INTERRUPTED at step {step}/{num_steps}")
            print("=" * 70)
            break
        except Exception as e:
            print()
            print("=" * 70)
            print(f"ERROR at step {step}/{num_steps}")
            print(f"Exception: {type(e).__name__}: {e}")
            print("=" * 70)
            import traceback
            traceback.print_exc()
            break

    # Training loop completed - check if we finished all steps
    if step == num_steps:
        print()
        print("=" * 70)
        print(f"SUCCESS: Training loop completed all {num_steps} steps")
        print("=" * 70)
        print()
    else:
        print()
        print("=" * 70)
        print(f"WARNING: Training loop exited early at step {step}/{num_steps}")
        print("=" * 70)
        print()

    # Generalization test (different seed)
    print()
    print("=" * 70)
    print("GENERALIZATION TEST (different seed)")
    print("=" * 70)
    print()
    print("Testing on fresh data (seed=99, different from training seed=42)...")

    os.environ['VAR_RUN_SEED'] = '99'  # Different seed!
    eval_loader, eval_num_classes, _ = get_seq_mnist_loader(train=True)

    model.eval()
    eval_correct = 0
    eval_total = 0
    eval_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            preds = logits.argmax(dim=1)
            eval_correct += (preds == targets).sum().item()
            eval_total += targets.size(0)

            loss = criterion(logits, targets)
            eval_losses.append(loss.item())

            # Only process a reasonable eval set (same size as training)
            if batch_idx >= 300:
                break

    eval_acc = eval_correct / eval_total
    eval_loss = sum(eval_losses) / len(eval_losses)

    print(f"Eval batches:      {len(eval_losses)}")
    print(f"Eval samples:      {eval_total}")
    print(f"Eval loss:         {eval_loss:.4f}")
    print(f"Eval accuracy:     {eval_acc:.1%}")
    print(f"vs. chance (50%):  {'+' if eval_acc > 0.5 else '-'}{abs(eval_acc - 0.5):.1%}")
    print()

    # Final summary
    print("=" * 70)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("=" * 70)
    total_time = time.time() - start_time
    final_loss = losses[-1]
    final_loss_ma = sum(losses[-ma_window:]) / ma_window
    final_acc = accs[-1]
    final_acc_ma = sum(accs[-ma_window:]) / ma_window

    print()
    print("Training performance:")
    print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Avg step time:      {total_time/num_steps:.3f}s")
    print(f"  Final loss:         {final_loss:.4f}")
    print(f"  Final loss (MA100): {final_loss_ma:.4f}")
    print(f"  Final acc:          {final_acc:.2%}")
    print(f"  Final acc (MA100):  {final_acc_ma:.2%}")
    print()
    print("Generalization performance:")
    print(f"  Eval accuracy:      {eval_acc:.1%}")
    print(f"  vs. baseline:       {eval_acc - 0.648:.2%} (target: 64.8%)")
    print()
    print("Findings:")
    print(f"  - AGC OFF: Model trained for {num_steps} steps without gradient clipping")
    print(f"  - Training acc (MA100): {final_acc_ma:.1%}")
    print(f"  - Generalization acc: {eval_acc:.1%} (vs 50% chance)")
    print(f"  - Dataset: REAL assoc_clean (not random noise)")
    print(f"  - Speed: {total_time/num_steps:.2f}s/step (expected ~1.5s)")
    print()

    # TOT-H007 verdict
    if eval_acc > 0.60:
        status = "SUPPORTED"
        verdict = "Model generalizes well without AGC"
    elif eval_acc > 0.55:
        status = "PARTIALLY SUPPORTED"
        verdict = "Model learns but below baseline (needs investigation)"
    else:
        status = "NEEDS REVIEW"
        verdict = "Model not learning effectively (check config/data)"

    print(f"TOT-H007 status: {status}")
    print(f"  {verdict}")
    print()
    print("Dashboard log: logs/probe/probe_live.log")
    print("=" * 70)


if __name__ == "__main__":
    main()
