"""
Train Byte Ring Model - Autoencoder Task

Test if direct byte I/O works (no input/output heads).
Task: Reconstruct random byte sequences.
"""

import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from byte_ring_model import ByteRingModel
from byte_data import (
    generate_random_bytes,
    generate_repeated_byte,
    byte_accuracy,
    bit_accuracy,
    bytes_to_string,
)


def main():
    print("=" * 70)
    print("BYTE AUTOENCODER TRAINING")
    print("=" * 70)
    print()

    # Task config
    seq_len = 16  # Sequence length (bytes)
    batch_size = 32
    task_type = "repeated"  # Options: "repeated", "random", "copy"

    # Model config
    num_positions = 64
    embedding_dim = 32  # Default: 32D internal representation
    mobius = False  # Start with standard ring

    print(f"Task: Byte autoencoding - {task_type} pattern")
    print(f"Sequence length: {seq_len} bytes")
    print(f"Batch size: {batch_size}")
    print(f"Model: {num_positions} positions, embedding_dim={embedding_dim}, {'Möbius' if mobius else 'Standard'}")
    print()

    # Create model
    torch.manual_seed(42)
    model = ByteRingModel(
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=mobius,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set (use task_type)
    if task_type == "repeated":
        x_eval, y_eval = generate_repeated_byte(n_samples=100, seq_len=seq_len, seed=9999)
    else:
        x_eval, y_eval = generate_random_bytes(n_samples=100, seq_len=seq_len, seed=9999)

    # Log file
    log_path = Path(__file__).parent / "logs" / "diamond" / "byte_autoencoder.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        f.write("")

    print(f"Logging to: {log_path}")
    print()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
            f.flush()

    log("=" * 70)
    log(f"Byte Autoencoder: {task_type} task, {seq_len} bytes, {num_positions} positions")
    log(f"Model: {total_params} params, embedding_dim={embedding_dim} ({'Möbius' if mobius else 'Standard'})")
    log(f"Optimizer: AdamW(lr=0.001, weight_decay=0.01)")
    log("=" * 70)

    best_byte_acc = 0.0
    best_bit_acc = 0.0
    step = 0

    try:
        # Run for 1000 steps
        while step < 1000:
            step_start = time.time()

            # Generate fresh training data based on task type
            if task_type == "repeated":
                x_train, y_train = generate_repeated_byte(
                    n_samples=batch_size,
                    seq_len=seq_len,
                    seed=42 + step + 1000000,
                )
            else:
                x_train, y_train = generate_random_bytes(
                    n_samples=batch_size,
                    seq_len=seq_len,
                    seed=42 + step + 1000000,
                )

            # Train step
            optimizer.zero_grad()
            output = model(x_train)

            # Loss: Binary cross-entropy per bit
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, y_train)
            loss.backward()
            optimizer.step()

            # Training metrics
            train_bit_acc = bit_accuracy(output, y_train)
            train_byte_acc = byte_accuracy(output, y_train)

            step_time = time.time() - step_start

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

                log(f"step {step} | loss {loss.item():.6f} | "
                    f"bit_acc={eval_bit_acc:.4f} | byte_acc={eval_byte_acc:.4f} | "
                    f"train_bit={train_bit_acc:.4f} | train_byte={train_byte_acc:.4f} | "
                    f"best_bit={best_bit_acc:.4f} | best_byte={best_byte_acc:.4f} | "
                    f"s_per_step={step_time:.3f}")

            step += 1

    except KeyboardInterrupt:
        pass

    log("")
    log("=" * 70)
    log(f"Training completed at step {step}")
    log(f"Best bit accuracy: {best_bit_acc*100:.1f}%")
    log(f"Best byte accuracy: {best_byte_acc*100:.1f}%")
    log("=" * 70)

    # Final evaluation with examples
    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print(f"Best bit accuracy: {best_bit_acc*100:.1f}%")
    print(f"Best byte accuracy: {best_byte_acc*100:.1f}%")
    print()

    # Show some examples
    model.eval()
    with torch.no_grad():
        if task_type == "repeated":
            test_x, test_y = generate_repeated_byte(n_samples=3, seq_len=8, seed=12345)
        else:
            test_x, test_y = generate_random_bytes(n_samples=3, seq_len=8, seed=12345)
        test_output = model(test_x)
        test_pred = (test_output > 0.5).float()

        print("Example predictions:")
        for i in range(3):
            print(f"\nSample {i}:")
            print(f"  Input:  {bytes_to_string(test_x[i])}")
            print(f"  Pred:   {bytes_to_string(test_pred[i])}")
            match = (test_pred[i] == test_y[i]).all(dim=-1)
            matches = match.sum().item()
            print(f"  Match:  {matches}/{len(match)} bytes correct")

    print()
    print("=" * 70)
    print("SUCCESS CRITERIA:")
    print("=" * 70)
    if best_byte_acc >= 0.80:
        print("✓ PASS: Can reconstruct bytes > 80% accuracy")
    else:
        print(f"✗ FAIL: Only {best_byte_acc*100:.1f}% byte accuracy (need 80%+)")

    if best_bit_acc >= 0.90:
        print("✓ PASS: Can reconstruct bits > 90% accuracy")
    else:
        print(f"✗ FAIL: Only {best_bit_acc*100:.1f}% bit accuracy (need 90%+)")

    print()
    print(f"Model parameters: {total_params} (vs ~1400 with heads)")
    print("=" * 70)


if __name__ == "__main__":
    main()
