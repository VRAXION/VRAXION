"""
Test: 128D Embedding (75% parameter reduction)

Configuration:
- Single pointer
- 3 layers (depth)
- 128D embedding (vs 256D)

Goal: Test if we can cut 75% of params/compute while maintaining 100% accuracy.
"""

import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dual_pointer_model import DualPointerByteRingModel
from byte_data import generate_addition_task, byte_accuracy, bit_accuracy


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def main():
    print("=" * 70)
    print("TEST: 128D EMBEDDING (75% PARAMETER REDUCTION)")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Pointers: Single")
    print("  Depth: 3 layers")
    print("  Embedding: 128D (vs 256D baseline)")
    print()
    print("Question: Can we cut 75% of params/compute and still reach 100%?")
    print("=" * 70)
    print()

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=128,  # REDUCED FROM 256D
        depth=3,
        use_dual_pointers=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    baseline_params = 136266
    savings = baseline_params - total_params
    savings_pct = (savings / baseline_params) * 100

    print(f"Parameters: {total_params:,}")
    print(f"Baseline (256D): {baseline_params:,}")
    print(f"Savings: {savings:,} params ({savings_pct:.1f}% reduction)")
    print()

    # Training setup
    seq_len = 16
    batch_size = 32
    max_value = 100
    num_steps = 10000
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set
    x_eval, y_eval = generate_addition_task(
        n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999
    )

    # Log setup
    log_path = Path(__file__).parent / "logs" / "test_128d.log"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # Track best
    best_sum_acc = 0.0
    best_step = 0
    reached_100_step = -1

    print(f"Training for {num_steps:,} steps (no early stopping)...")
    print(f"Log: {log_path}")
    print()
    print("Dashboard: http://localhost:8501")
    print("=" * 70)
    print()

    start_time = time.time()

    with open(log_path, "w") as log_file:
        for step in range(num_steps):
            step_start = time.time()

            # Generate training data
            x_train, y_train = generate_addition_task(
                n_samples=batch_size,
                seq_len=seq_len,
                max_value=max_value,
                seed=42 + step + 1000000,
            )

            # Train step
            optimizer.zero_grad()
            output = model(x_train)
            loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start

            # Eval
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    eval_output = model(x_eval)
                    sum_acc = position_accuracy(eval_output, y_eval, 2)
                    byte_acc = byte_accuracy(eval_output, y_eval)
                    bit_acc = bit_accuracy(eval_output, y_eval)
                model.train()

                # Track best
                if sum_acc > best_sum_acc:
                    best_sum_acc = sum_acc
                    best_step = step

                # Track when 100% first reached
                if reached_100_step == -1 and sum_acc >= 1.0:
                    reached_100_step = step
                    print(f"  >> REACHED 100% at step {step}!")

                # Log to file (dashboard format)
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"sum_acc={sum_acc:.4f} byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
                    f"s_per_step={step_time:.3f}\n"
                )
                log_file.write(log_line)
                log_file.flush()

                # Console output every 500 steps
                if step % 500 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"step {step:5d} | loss {loss.item():.6f} | "
                        f"sum_acc={sum_acc*100:5.1f}% | byte_acc={byte_acc*100:5.1f}% | "
                        f"time={elapsed:.1f}s"
                    )

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best sum accuracy: {best_sum_acc*100:.1f}% (at step {best_step})")
    if reached_100_step >= 0:
        print(f"Reached 100% at step: {reached_100_step}")
    print(f"Training time: {total_time:.1f}s")
    print(f"Parameters: {total_params:,} ({savings_pct:.1f}% smaller than 256D)")
    print()

    if best_sum_acc >= 1.0:
        print("SUCCESS! 128D embedding achieves 100%")
        print(f"  75% parameter reduction with NO accuracy loss!")
        print(f"  Optimal architecture: Single + 3 layers + 128D")
    elif best_sum_acc >= 0.90:
        print(f"PARTIAL SUCCESS! {best_sum_acc*100:.1f}% accuracy")
        print(f"  Close to 100%, might reach it with more steps")
    else:
        print(f"FAILURE! Only {best_sum_acc*100:.1f}% accuracy")
        print(f"  128D insufficient, need 256D")

    print("=" * 70)


if __name__ == "__main__":
    main()
