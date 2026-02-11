"""
FINAL DEFINITIVE TEST: Can a scaled-up individual model learn addition?

Architecture:
- Dual pointers (read from TWO locations)
- 256D embedding (vs 32D - more neurons)
- 3 layers of processing per timestep (vs 1 - more depth)
- 64 ring positions (memory)

Parameters: ~200K (vs 650-750 before)

This tests: Is addition possible with just MORE neurons and MORE depth?
Or is the swarm fundamentally necessary?

Training: 10,000 steps on byte addition (0-100)
Success criterion: >90% sum accuracy at position 2
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
    print("FINAL TEST: SCALED INDIVIDUAL MODEL ON ADDITION")
    print("=" * 70)
    print()
    print("Question: Can ONE model learn addition if it has:")
    print("  - More neurons (256D embedding)")
    print("  - More depth (3 layers per timestep)")
    print("  - Dual pointers (multi-location access)")
    print()
    print("Or is swarm fundamentally necessary?")
    print("=" * 70)
    print()

    # Configuration
    seq_len = 16
    batch_size = 32
    max_value = 100
    num_steps = 10000
    eval_interval = 50

    # Create log directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "diamond_probe.log"

    # Clear previous log
    if log_path.exists():
        log_path.unlink()

    print("Creating scaled-up model...")
    torch.manual_seed(42)

    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=256,  # 8x larger than before
        depth=3,  # 3x deeper per timestep
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Embedding: 256D (was 32D)")
    print(f"Depth: 3 layers/timestep (was 1)")
    print(f"Pointers: 2 (dual)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set
    x_eval, y_eval = generate_addition_task(
        n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999
    )

    # Track best
    best_sum_acc = 0.0
    best_byte_acc = 0.0
    best_step = 0

    print(f"Training for {num_steps:,} steps...")
    print(f"Logging to: {log_path}")
    print()
    print("Launch dashboard with:")
    print(f"  python -m streamlit run diamond_dashboard.py -- --log {log_path}")
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
                    best_byte_acc = byte_acc
                    best_step = step

                # Log to file (dashboard format)
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"sum_acc={sum_acc:.4f} byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
                    f"s_per_step={step_time:.3f}\n"
                )
                log_file.write(log_line)
                log_file.flush()

                # Console output
                if step % 500 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"step {step:5d} | loss {loss.item():.6f} | "
                        f"sum_acc={sum_acc*100:5.1f}% | byte_acc={byte_acc*100:5.1f}% | "
                        f"elapsed={elapsed:.1f}s"
                    )

                # Early stopping if perfect
                if sum_acc >= 0.99:
                    print()
                    print("=" * 70)
                    print("SUCCESS! Model learned addition!")
                    print(f"Reached 99% sum accuracy at step {step}")
                    print("=" * 70)
                    break

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best sum accuracy: {best_sum_acc*100:.1f}% (at step {best_step})")
    print(f"Best byte accuracy: {best_byte_acc*100:.1f}%")
    print()

    if best_sum_acc >= 0.90:
        print("SUCCESS! Individual model CAN learn addition!")
        print("  -> Swarm is for harder tasks, not basic computation")
        print("  -> Needed: More neurons + more depth")
    elif best_sum_acc >= 0.50:
        print("PARTIAL SUCCESS! Model learned something, but not enough.")
        print("  -> May need even more capacity, or architectural change")
        print("  -> Consider: Even deeper (5+ layers), wider (512D), or swarm")
    else:
        print("FAILURE! Even scaled up, model cannot learn addition.")
        print("  -> Problem is architectural, not just capacity")
        print("  -> Swarm may be fundamentally necessary")
        print("  -> Or need explicit arithmetic module")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
