"""
Dimension Sweep - Bitwise Logic Tasks

Tests if model can learn simple bitwise operations:
- AND: a & b (each output bit = a_i AND b_i)
- OR:  a | b (each output bit = a_i OR b_i)
- XOR: a ^ b (each output bit = a_i XOR b_i)

These are MUCH simpler than addition (no carry across bits).
If model can't learn these, it can't do ANY logic.
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
    generate_logic_task,
    byte_accuracy,
    bit_accuracy,
)


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def train_single_config(embedding_dim, operation="and", num_steps=1000, verbose=False):
    """Train model on bitwise logic task."""
    seq_len = 16
    batch_size = 32
    num_positions = 64
    mobius = False

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing embedding_dim={embedding_dim} on {operation.upper()}")
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
    x_eval, y_eval = generate_logic_task(n_samples=100, seq_len=seq_len, operation=operation, seed=9999)

    # Track best
    best_byte_acc = 0.0
    best_bit_acc = 0.0
    best_logic_acc = 0.0  # Accuracy on position 2 (the logic result)
    converged_step = -1
    final_loss = 0.0

    start_time = time.time()

    # Training loop
    for step in range(num_steps):
        # Generate training data
        x_train, y_train = generate_logic_task(
            n_samples=batch_size,
            seq_len=seq_len,
            operation=operation,
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
                logic_acc = position_accuracy(eval_output, y_eval, 2)

            model.train()

            if eval_byte_acc > best_byte_acc:
                best_byte_acc = eval_byte_acc
            if eval_bit_acc > best_bit_acc:
                best_bit_acc = eval_bit_acc
            if logic_acc > best_logic_acc:
                best_logic_acc = logic_acc

            # Check convergence (95%+ on logic position)
            if converged_step == -1 and logic_acc >= 0.95:
                converged_step = step

            if verbose and step % 100 == 0:
                print(f"step {step:4d} | loss {loss.item():.6f} | "
                      f"{operation.upper()}_acc={logic_acc:.4f} | byte_acc={eval_byte_acc:.4f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"\nResults:")
        print(f"  Best bit accuracy:   {best_bit_acc*100:.1f}%")
        print(f"  Best byte accuracy:  {best_byte_acc*100:.1f}%")
        print(f"  Best {operation.upper()} accuracy:   {best_logic_acc*100:.1f}% (position 2)")
        print(f"  Converged at step:   {converged_step if converged_step >= 0 else 'N/A'}")
        print(f"  Training time:       {training_time:.1f}s")

    return {
        'embedding_dim': embedding_dim,
        'operation': operation,
        'total_params': total_params,
        'best_bit_acc': best_bit_acc,
        'best_byte_acc': best_byte_acc,
        'best_logic_acc': best_logic_acc,
        'converged_step': converged_step,
        'final_loss': final_loss,
        'training_time': training_time,
    }


def main():
    print("=" * 70)
    print("BYTE RING MODEL - BITWISE LOGIC SWEEP")
    print("=" * 70)
    print()
    print("Task: Learn bitwise operations (AND, OR, XOR)")
    print("Format: Input [a, b, 0, ...] -> Target [a, b, a OP b, 0, ...]")
    print("Training: 1000 steps per configuration")
    print("Operations: AND, OR, XOR (element-wise, no carry)")
    print("Dimensions: 8, 16, 32, 64")
    print()

    # Test all dimensions and operations
    test_dims = [8, 16, 32, 64]
    test_ops = ["and", "or", "xor"]
    results = []

    for op in test_ops:
        print(f"\n{'='*70}")
        print(f"OPERATION: {op.upper()}")
        print(f"{'='*70}")

        for dim in test_dims:
            result = train_single_config(dim, operation=op, num_steps=1000, verbose=True)
            results.append(result)

    # Save results to CSV
    log_path = Path(__file__).parent / "logs" / "diamond" / "logic_task_sweep.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w', newline='') as f:
        fieldnames = ['operation', 'embedding_dim', 'total_params', 'best_bit_acc', 'best_byte_acc',
                      'best_logic_acc', 'converged_step', 'final_loss', 'training_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Summary by operation
    print()
    print("=" * 70)
    print("SUMMARY - BITWISE LOGIC TASKS")
    print("=" * 70)

    for op in test_ops:
        print(f"\n{op.upper()} Operation:")
        print(f"{'Dim':>4} | {'Params':>6} | {op.upper()+' Acc':>8} | {'Conv Step':>10}")
        print("-" * 50)
        op_results = [r for r in results if r['operation'] == op]
        for r in op_results:
            conv_str = str(r['converged_step']) if r['converged_step'] >= 0 else "N/A"
            print(f"{r['embedding_dim']:>4} | {r['total_params']:>6,} | "
                  f"{r['best_logic_acc']*100:>7.1f}% | {conv_str:>10}")

    print()
    print(f"Results saved to: {log_path}")
    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for op in test_ops:
        op_results = [r for r in results if r['operation'] == op]
        best = max(op_results, key=lambda r: r['best_logic_acc'])
        winners = [r for r in op_results if r['best_logic_acc'] >= 0.95]

        print(f"\n{op.upper()}:")
        print(f"  Best: {best['embedding_dim']}D with {best['best_logic_acc']*100:.1f}% accuracy")

        if winners:
            smallest = min(winners, key=lambda r: r['total_params'])
            print(f"  Smallest to reach 95%: {smallest['embedding_dim']}D ({smallest['total_params']:,} params)")
        else:
            print(f"  None reached 95% - task may be too hard")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
