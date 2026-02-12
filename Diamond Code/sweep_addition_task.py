"""
Dimension Sweep - Addition Task

Tests all embedding dimensions on byte addition:
Input:  [a, b, 0, 0, ...]
Target: [a, b, sum, 0, ...]

Tests compositional reasoning: can the model learn arithmetic?
"""

import torch
import torch.nn as nn
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from byte_ring_model import ByteRingModel
from byte_data import (
    generate_addition_task,
    byte_accuracy,
    bit_accuracy,
)


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def train_single_config(embedding_dim, num_steps=2000, verbose=False):
    """Train model on addition task."""
    seq_len = 16
    batch_size = 32
    num_positions = 64
    mobius = False
    max_value = 100  # Limit to 0-100 for easier learning

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing embedding_dim={embedding_dim}")
        print(f"{'='*70}")

    # Create model
    torch.manual_seed(42)
    model = ByteRingModel(
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=mobius,
    )

    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"Parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set
    x_eval, y_eval = generate_addition_task(n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999)

    # Track best
    best_byte_acc = 0.0
    best_bit_acc = 0.0
    best_sum_acc = 0.0  # Accuracy on position 2 (the sum)
    converged_step = -1
    final_loss = 0.0

    start_time = time.time()

    # Training loop
    for step in range(num_steps):
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

        final_loss = loss.item()

        # Eval every 20 steps
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                eval_output = model(x_eval)
                eval_bit_acc = bit_accuracy(eval_output, y_eval)
                eval_byte_acc = byte_accuracy(eval_output, y_eval)

                # Check accuracy at each position
                echo_a_acc = position_accuracy(eval_output, y_eval, 0)
                echo_b_acc = position_accuracy(eval_output, y_eval, 1)
                sum_acc = position_accuracy(eval_output, y_eval, 2)

            model.train()

            if eval_byte_acc > best_byte_acc:
                best_byte_acc = eval_byte_acc
            if eval_bit_acc > best_bit_acc:
                best_bit_acc = eval_bit_acc
            if sum_acc > best_sum_acc:
                best_sum_acc = sum_acc

            # Check convergence (80%+ on sum position)
            if converged_step == -1 and sum_acc >= 0.80:
                converged_step = step

            if verbose and step % 200 == 0:
                print(f"step {step:4d} | loss {loss.item():.6f} | "
                      f"sum_acc={sum_acc:.4f} | byte_acc={eval_byte_acc:.4f} | "
                      f"echo=[{echo_a_acc:.2f}, {echo_b_acc:.2f}]")

    training_time = time.time() - start_time

    if verbose:
        print(f"\nResults:")
        print(f"  Best bit accuracy:  {best_bit_acc*100:.1f}%")
        print(f"  Best byte accuracy: {best_byte_acc*100:.1f}%")
        print(f"  Best sum accuracy:  {best_sum_acc*100:.1f}% (position 2)")
        print(f"  Converged at step:  {converged_step if converged_step >= 0 else 'N/A'}")
        print(f"  Training time:      {training_time:.1f}s")

    return {
        'embedding_dim': embedding_dim,
        'total_params': total_params,
        'best_bit_acc': best_bit_acc,
        'best_byte_acc': best_byte_acc,
        'best_sum_acc': best_sum_acc,
        'converged_step': converged_step,
        'final_loss': final_loss,
        'training_time': training_time,
    }


def main():
    print("=" * 70)
    print("BYTE RING MODEL - ADDITION TASK SWEEP")
    print("=" * 70)
    print()
    print("Task: Learn byte addition (a + b = sum)")
    print("Format: Input [a, b, 0, ...] -> Target [a, b, sum, 0, ...]")
    print("Training: 2000 steps per configuration")
    print("Dimensions: 8, 16, 32, 64")
    print()

    # Test all dimensions
    test_dims = [8, 16, 32, 64]
    results = []

    for dim in test_dims:
        result = train_single_config(dim, num_steps=2000, verbose=True)
        results.append(result)

    # Save results to CSV
    log_path = Path(__file__).parent / "logs" / "diamond" / "addition_task_sweep.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w', newline='') as f:
        fieldnames = ['embedding_dim', 'total_params', 'best_bit_acc', 'best_byte_acc',
                      'best_sum_acc', 'converged_step', 'final_loss', 'training_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print()
    print("=" * 70)
    print("SUMMARY - ADDITION TASK")
    print("=" * 70)
    print(f"{'Dim':>4} | {'Params':>6} | {'Sum Acc':>8} | {'Byte Acc':>8} | {'Conv Step':>10} | {'Time':>6}")
    print("-" * 70)
    for r in results:
        conv_str = str(r['converged_step']) if r['converged_step'] >= 0 else "N/A"
        print(f"{r['embedding_dim']:>4} | {r['total_params']:>6,} | "
              f"{r['best_sum_acc']*100:>7.1f}% | {r['best_byte_acc']*100:>7.1f}% | "
              f"{conv_str:>10} | {r['training_time']:>5.1f}s")

    print()
    print("Key metric: Sum Acc = accuracy on position 2 (the actual computation)")
    print(f"Results saved to: {log_path}")
    print()

    # Recommendation
    best_dim = max(results, key=lambda r: r['best_sum_acc'])
    print(f"Best dimension: {best_dim['embedding_dim']}D with {best_dim['best_sum_acc']*100:.1f}% sum accuracy")

    # Check if any reached 80%+ on sum
    winners = [r for r in results if r['best_sum_acc'] >= 0.80]
    if winners:
        smallest_winner = min(winners, key=lambda r: r['total_params'])
        print(f"Recommended: {smallest_winner['embedding_dim']}D ({smallest_winner['total_params']:,} params) - smallest to reach 80%+ on addition")
    else:
        print("Note: No configuration reached 80%+ on addition (task may be too hard)")

    print("=" * 70)


if __name__ == "__main__":
    main()
