"""
Test A6: Single Pointer + Deep (3 layers) + Large Embedding (256D)

This tests if dual pointers are necessary, or if depth + capacity is sufficient.

Baseline (known working): Dual + 3 layers + 256D = 100%
This test: Single + 3 layers + 256D = ???
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
    print("TEST A6: SINGLE POINTER + DEEP + LARGE EMBEDDING")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Pointers: SINGLE (vs dual)")
    print("  Depth: 3 layers")
    print("  Embedding: 256D")
    print()
    print("Question: Are dual pointers necessary, or is depth+capacity enough?")
    print("=" * 70)
    print()

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=256,
        depth=3,
        use_dual_pointers=False,  # SINGLE POINTER MODE
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
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
    log_path = Path(__file__).parent / "logs" / "a6_test.log"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # Track best
    best_sum_acc = 0.0
    best_step = 0

    print(f"Training for up to {num_steps:,} steps...")
    print(f"Log: {log_path}")
    print()
    print("Watch dashboard at: http://localhost:8501")
    print("  (Launch: python -m streamlit run diamond_dashboard.py -- --log logs/a6_test.log)")
    print()
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

                # Log to file (dashboard format)
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"sum_acc={sum_acc:.4f} byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
                    f"s_per_step={step_time:.3f}\n"
                )
                log_file.write(log_line)
                log_file.flush()

                # Console output every 100 steps
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"step {step:5d} | loss {loss.item():.6f} | "
                        f"sum_acc={sum_acc*100:5.1f}% | byte_acc={byte_acc*100:5.1f}% | "
                        f"time={elapsed:.1f}s"
                    )

                # Track milestones (no early stopping - run full 10K)
                if sum_acc >= 0.95 and best_sum_acc < 0.95:
                    print(f"  >> Reached 95% at step {step}")

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best sum accuracy: {best_sum_acc*100:.1f}% (at step {best_step})")
    print(f"Training time: {total_time:.1f}s")
    print()

    if best_sum_acc >= 0.90:
        print("CONCLUSION: Dual pointers are NOT necessary!")
        print("  Minimal architecture: Single pointer + 3 layers + 256D")
        print(f"  Parameters: {total_params:,}")
    elif best_sum_acc >= 0.50:
        print("PARTIAL: Single pointer learns something, but not enough")
        print("  Dual pointers may accelerate, but not required for learning")
    else:
        print("CONCLUSION: Dual pointers ARE necessary!")
        print("  Cannot learn addition with single pointer, even with depth+capacity")

    print("=" * 70)


if __name__ == "__main__":
    main()
