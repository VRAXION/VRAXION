"""
Dimension Sweep for Byte Ring Model

Tests all embedding dimensions (8, 16, 32, 64) on repeated byte task.
Outputs CSV with results for comparison.
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
    generate_repeated_byte,
    byte_accuracy,
    bit_accuracy,
)


def train_single_config(embedding_dim, num_steps=500, verbose=False):
    """
    Train model with given embedding_dim on repeated byte task.

    Returns:
        dict with results: {
            'embedding_dim': int,
            'total_params': int,
            'best_bit_acc': float,
            'best_byte_acc': float,
            'converged_step': int,
            'final_loss': float,
            'training_time': float,
        }
    """
    # Config
    seq_len = 16
    batch_size = 32
    num_positions = 64
    mobius = False

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
    x_eval, y_eval = generate_repeated_byte(n_samples=100, seq_len=seq_len, seed=9999)

    # Track best
    best_byte_acc = 0.0
    best_bit_acc = 0.0
    converged_step = -1
    final_loss = 0.0

    start_time = time.time()

    # Training loop
    for step in range(num_steps):
        # Generate training data
        x_train, y_train = generate_repeated_byte(
            n_samples=batch_size,
            seq_len=seq_len,
            seed=42 + step + 1000000,
        )

        # Train step
        optimizer.zero_grad()
        output = model(x_train)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        # Eval every 10 steps
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_output = model(x_eval)
                eval_bit_acc = bit_accuracy(eval_output, y_eval)
                eval_byte_acc = byte_accuracy(eval_output, y_eval)
            model.train()

            if eval_byte_acc > best_byte_acc:
                best_byte_acc = eval_byte_acc
            if eval_bit_acc > best_bit_acc:
                best_bit_acc = eval_bit_acc

            # Check convergence (95%+ byte accuracy)
            if converged_step == -1 and eval_byte_acc >= 0.95:
                converged_step = step

            if verbose and step % 50 == 0:
                print(f"step {step:3d} | loss {loss.item():.6f} | "
                      f"bit={eval_bit_acc:.4f} | byte={eval_byte_acc:.4f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"\nResults:")
        print(f"  Best bit accuracy:  {best_bit_acc*100:.1f}%")
        print(f"  Best byte accuracy: {best_byte_acc*100:.1f}%")
        print(f"  Converged at step:  {converged_step if converged_step >= 0 else 'N/A'}")
        print(f"  Training time:      {training_time:.1f}s")

    return {
        'embedding_dim': embedding_dim,
        'total_params': total_params,
        'best_bit_acc': best_bit_acc,
        'best_byte_acc': best_byte_acc,
        'converged_step': converged_step,
        'final_loss': final_loss,
        'training_time': training_time,
    }


def main():
    print("=" * 70)
    print("BYTE RING MODEL - DIMENSION SWEEP")
    print("=" * 70)
    print()
    print("Task: Repeated byte (easiest)")
    print("Training: 500 steps per configuration")
    print("Dimensions: 8, 16, 32, 64")
    print()

    # Test all dimensions
    test_dims = [8, 16, 32, 64]
    results = []

    for dim in test_dims:
        result = train_single_config(dim, num_steps=500, verbose=True)
        results.append(result)

    # Save results to CSV
    log_path = Path(__file__).parent / "logs" / "diamond" / "dimension_sweep.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w', newline='') as f:
        fieldnames = ['embedding_dim', 'total_params', 'best_bit_acc', 'best_byte_acc',
                      'converged_step', 'final_loss', 'training_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dim':>4} | {'Params':>6} | {'Bit Acc':>8} | {'Byte Acc':>8} | {'Conv Step':>10} | {'Time':>6}")
    print("-" * 70)
    for r in results:
        conv_str = str(r['converged_step']) if r['converged_step'] >= 0 else "N/A"
        print(f"{r['embedding_dim']:>4} | {r['total_params']:>6,} | "
              f"{r['best_bit_acc']*100:>7.1f}% | {r['best_byte_acc']*100:>7.1f}% | "
              f"{conv_str:>10} | {r['training_time']:>5.1f}s")

    print()
    print(f"Results saved to: {log_path}")
    print()

    # Recommendation
    best_dim = max(results, key=lambda r: r['best_byte_acc'])
    print(f"Best dimension: {best_dim['embedding_dim']}D with {best_dim['best_byte_acc']*100:.1f}% byte accuracy")

    # Check if 32D or 64D reached 95%+
    winners = [r for r in results if r['best_byte_acc'] >= 0.95]
    if winners:
        smallest_winner = min(winners, key=lambda r: r['total_params'])
        print(f"Recommended: {smallest_winner['embedding_dim']}D ({smallest_winner['total_params']:,} params) - smallest to reach 95%+")
    else:
        print("WARNING: No configuration reached 95%+ byte accuracy!")

    print("=" * 70)


if __name__ == "__main__":
    main()
