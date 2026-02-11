"""
Ablation Study: What Makes Addition Work?

Systematically test all combinations of features:
- Single vs Dual pointers
- Shallow (1 layer) vs Deep (3 layers)
- Small (32D) vs Large (256D) embedding

Goal: Identify minimal requirements for 100% sum accuracy on addition.
"""

import torch
import torch.nn as nn
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dual_pointer_model import DualPointerByteRingModel
from byte_data import generate_addition_task, byte_accuracy, bit_accuracy


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def train_config(use_dual, depth, embed_dim, label, num_steps=5000):
    """Train a single configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"  Dual pointers: {use_dual}")
    print(f"  Depth: {depth} layers")
    print(f"  Embedding: {embed_dim}D")
    print(f"{'='*70}")

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=embed_dim,
        depth=depth,
        use_dual_pointers=use_dual,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Training setup
    seq_len = 16
    batch_size = 32
    max_value = 100
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set
    x_eval, y_eval = generate_addition_task(
        n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999
    )

    # Create log directory and file
    log_dir = Path(__file__).parent / "logs" / "ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label.lower().replace(' ', '_').replace(':', '')}.log"

    # Clear previous log
    if log_path.exists():
        log_path.unlink()

    # Track best
    best_sum_acc = 0.0
    best_byte_acc = 0.0
    best_step = 0
    converged_step = -1

    start_time = time.time()

    print(f"\nTraining for up to {num_steps:,} steps...")
    print(f"Logging to: {log_path.name}")
    print()

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

                # Check convergence
                if converged_step == -1 and sum_acc >= 0.95:
                    converged_step = step

                # Log to file
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"sum_acc={sum_acc:.4f} byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
                    f"s_per_step={step_time:.3f}\n"
                )
                log_file.write(log_line)
                log_file.flush()

                # Console output
                if step % 500 == 0:
                    print(
                        f"step {step:5d} | loss {loss.item():.6f} | "
                        f"sum_acc={sum_acc*100:5.1f}% | byte_acc={byte_acc*100:5.1f}%"
                    )

                # Early stopping if perfect
                if sum_acc >= 0.99:
                    print()
                    print(f"SUCCESS! Reached 99% sum accuracy at step {step}")
                    converged_step = step
                    break

    training_time = time.time() - start_time

    print()
    print(f"Best sum accuracy: {best_sum_acc*100:.1f}% (at step {best_step})")
    print(f"Converged at step: {converged_step if converged_step >= 0 else 'N/A'}")
    print(f"Training time: {training_time:.1f}s")

    return {
        "config": label,
        "use_dual": use_dual,
        "depth": depth,
        "embed_dim": embed_dim,
        "params": total_params,
        "best_sum_acc": best_sum_acc,
        "best_byte_acc": best_byte_acc,
        "converged_step": converged_step,
        "training_time": training_time,
    }


def main():
    print("=" * 70)
    print("ABLATION STUDY: WHAT MAKES ADDITION WORK?")
    print("=" * 70)
    print()
    print("Testing 8 configurations to identify critical components:")
    print("  - Single vs Dual pointers (multi-location access)")
    print("  - Shallow (1) vs Deep (3) layers (computation)")
    print("  - Small (32D) vs Large (256D) embedding (capacity)")
    print()
    print("Hypothesis: Depth > Embedding > Dual pointers")
    print("=" * 70)

    # Test matrix
    configs = [
        # (use_dual, depth, embed_dim, label)
        (False, 1, 32, "Baseline"),
        (True, 1, 32, "A1: +Dual"),
        (False, 3, 32, "A2: +Depth"),
        (False, 1, 256, "A3: +Embed"),
        (True, 3, 32, "A4: Dual+Depth"),
        (True, 1, 256, "A5: Dual+Embed"),
        (False, 3, 256, "A6: Depth+Embed"),
        (True, 3, 256, "Current"),
    ]

    results = []

    for use_dual, depth, embed_dim, label in configs:
        result = train_config(use_dual, depth, embed_dim, label, num_steps=5000)
        results.append(result)

    # Save results to CSV
    csv_path = Path(__file__).parent / "logs" / "ablation" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "config",
            "use_dual",
            "depth",
            "embed_dim",
            "params",
            "best_sum_acc",
            "best_byte_acc",
            "converged_step",
            "training_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Print summary
    print()
    print("=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Config':<20} | {'Params':>8} | {'Sum Acc':>8} | {'Converged':>10}")
    print("-" * 70)

    for r in results:
        conv_str = str(r["converged_step"]) if r["converged_step"] >= 0 else "N/A"
        print(
            f"{r['config']:<20} | {r['params']:>8,} | {r['best_sum_acc']*100:>7.1f}% | {conv_str:>10}"
        )

    print()
    print(f"Results saved to: {csv_path}")
    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Find configs that work (>90% sum acc)
    working = [r for r in results if r["best_sum_acc"] >= 0.90]

    if working:
        print(f"Configurations with 90%+ sum accuracy: {len(working)}")
        for r in working:
            print(f"  - {r['config']}: {r['best_sum_acc']*100:.1f}%")

        # Find minimal
        minimal = min(working, key=lambda r: r["params"])
        print()
        print(f"Minimal working configuration:")
        print(f"  Config: {minimal['config']}")
        print(f"  Parameters: {minimal['params']:,}")
        print(f"  Sum accuracy: {minimal['best_sum_acc']*100:.1f}%")
        print(f"  Dual pointers: {minimal['use_dual']}")
        print(f"  Depth: {minimal['depth']}")
        print(f"  Embedding: {minimal['embed_dim']}D")
    else:
        print("No configurations reached 90% sum accuracy!")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
