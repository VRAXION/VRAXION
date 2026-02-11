"""
Quick Test: Can Dual Pointers Learn Addition?

Compare single-pointer vs dual-pointer on addition task.
"""

import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from byte_ring_model import ByteRingModel
from dual_pointer_model import DualPointerByteRingModel
from byte_data import generate_addition_task, byte_accuracy, bit_accuracy


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def train_model(model, model_name, num_steps=1000):
    """Train a model on addition and return best sum accuracy."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    seq_len = 16
    batch_size = 32
    max_value = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set
    x_eval, y_eval = generate_addition_task(n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999)

    best_sum_acc = 0.0

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

        # Eval every 50 steps
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_output = model(x_eval)
                sum_acc = position_accuracy(eval_output, y_eval, 2)
                byte_acc = byte_accuracy(eval_output, y_eval)
            model.train()

            if sum_acc > best_sum_acc:
                best_sum_acc = sum_acc

            if step % 200 == 0:
                print(f"step {step:4d} | loss {loss.item():.6f} | sum_acc={sum_acc:.4f} | byte_acc={byte_acc:.4f}")

    print(f"\nBest sum accuracy: {best_sum_acc*100:.1f}%")
    return best_sum_acc


def main():
    print("=" * 70)
    print("DUAL-POINTER vs SINGLE-POINTER ON ADDITION")
    print("=" * 70)
    print()

    torch.manual_seed(42)

    # Test single pointer (baseline)
    print("Creating SINGLE-POINTER model (baseline)...")
    single_model = ByteRingModel(num_memory_positions=64, embedding_dim=32)
    single_params = sum(p.numel() for p in single_model.parameters())
    print(f"Parameters: {single_params:,}")

    single_acc = train_model(single_model, "Single-Pointer (32D)", num_steps=1000)

    # Test dual pointer
    print("\n\nCreating DUAL-POINTER model...")
    dual_model = DualPointerByteRingModel(num_memory_positions=64, embedding_dim=32)
    dual_params = sum(p.numel() for p in dual_model.parameters())
    print(f"Parameters: {dual_params:,}")

    dual_acc = train_model(dual_model, "Dual-Pointer (32D)", num_steps=1000)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single-Pointer (32D, {single_params} params): {single_acc*100:.1f}% sum accuracy")
    print(f"Dual-Pointer (32D, {dual_params} params):   {dual_acc*100:.1f}% sum accuracy")
    print()

    if dual_acc > single_acc + 0.05:
        print(f"SUCCESS! Dual pointers improved by {(dual_acc - single_acc)*100:.1f}%")
        print("The ability to read from TWO locations helps!")
    elif dual_acc > single_acc:
        print(f"Slight improvement: +{(dual_acc - single_acc)*100:.1f}%")
    else:
        print("No improvement. Dual pointers alone don't fix the problem.")
        print("Issue may be deeper than just multi-location access.")

    print("=" * 70)


if __name__ == "__main__":
    main()
