"""
Basic Competence Benchmark: ADD, AND, OR, XOR

Tests a single model config on 4 fundamental operations sequentially.
Saves checkpoints for each operation.

Usage:
    python benchmark_basic_ops.py

Config tested:
    - Single pointer + 3 layers + 128D (optimal from ablation)
"""

import torch
import torch.nn as nn
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dual_pointer_model import DualPointerByteRingModel
from byte_data import generate_addition_task, generate_logic_task, byte_accuracy, bit_accuracy


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def weight_reset(m):
    """Reset learnable parameters to random init."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def test_single_operation(model, operation, num_steps=10000, log_dir=None):
    """
    Test model on a single operation.

    Args:
        model: DualPointerByteRingModel instance
        operation: 'add', 'and', 'or', or 'xor'
        num_steps: Training steps (default 10K)
        log_dir: Directory for logs and checkpoints

    Returns:
        dict with {accuracy, converged_step, time, checkpoint_path}
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {operation.upper()}")
    print(f"{'='*70}\n")

    # Reset model weights
    model.apply(weight_reset)

    # Setup
    seq_len = 16
    batch_size = 32
    max_value = 100
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Task generator
    if operation == 'add':
        task_fn = lambda **kw: generate_addition_task(**kw)
    else:
        task_fn = lambda **kw: generate_logic_task(operation=operation, **kw)

    # Eval set
    x_eval, y_eval = task_fn(n_samples=100, seq_len=seq_len, max_value=max_value, seed=9999)

    # Log setup
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{operation}.log"
        if log_path.exists():
            log_path.unlink()
        log_file = open(log_path, "w")
    else:
        log_file = None

    # Track best
    best_acc = 0.0
    best_step = 0
    converged_step = -1
    start_time = time.time()

    print(f"Training for {num_steps:,} steps...")
    print(f"Dashboard: python -m streamlit run diamond_dashboard.py -- --log logs/benchmark/{operation}.log")
    print()

    for step in range(num_steps):
        step_start = time.time()

        # Generate training data
        x_train, y_train = task_fn(
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
                eval_output, eval_stats = model(x_eval, return_stats=True)
                op_acc = position_accuracy(eval_output, y_eval, 2)  # Position 2 = result
                byte_acc = byte_accuracy(eval_output, y_eval)
                bit_acc = bit_accuracy(eval_output, y_eval)
                jump_rate = eval_stats['jump_gate']
            model.train()

            # Track best
            if op_acc > best_acc:
                best_acc = op_acc
                best_step = step

            # Track convergence (first time hitting 95%)
            if converged_step == -1 and op_acc >= 0.95:
                converged_step = step
                print(f"  >> CONVERGED at step {step} ({op_acc*100:.1f}%)")

            # Log
            if log_file:
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"op_acc={op_acc:.4f} byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
                    f"jump_gate={jump_rate:.4f} "
                    f"s_per_step={step_time:.3f}\n"
                )
                log_file.write(log_line)
                log_file.flush()

            # Console every 500 steps
            if step % 500 == 0:
                elapsed = time.time() - start_time
                print(
                    f"step {step:5d} | loss {loss.item():.6f} | "
                    f"op_acc={op_acc*100:5.1f}% | byte_acc={byte_acc*100:5.1f}% | "
                    f"time={elapsed:.1f}s"
                )

    total_time = time.time() - start_time

    if log_file:
        log_file.close()

    # Save checkpoint
    checkpoint_path = None
    if log_dir and best_acc >= 0.95:
        checkpoint_path = log_dir / f"{operation}_best.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'operation': operation,
            'accuracy': best_acc,
            'step': best_step,
        }, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

    # Results
    result = {
        'operation': operation,
        'accuracy': best_acc,
        'converged_step': converged_step,
        'best_step': best_step,
        'time_seconds': total_time,
        'passed': best_acc >= 0.95,
        'checkpoint': str(checkpoint_path) if checkpoint_path else None,
    }

    print(f"\n{'='*70}")
    print(f"RESULT: {operation.upper()}")
    print(f"{'='*70}")
    print(f"Best accuracy: {best_acc*100:.1f}% (at step {best_step})")
    if converged_step >= 0:
        print(f"Converged: step {converged_step}")
    print(f"Time: {total_time:.1f}s")
    print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"{'='*70}\n")

    return result


def main():
    print("="*70)
    print("BASIC COMPETENCE BENCHMARK")
    print("="*70)
    print()
    print("Model Configuration:")
    print("  - Single pointer")
    print("  - 3 processing layers")
    print("  - 128D embedding")
    print()
    print("Operations: ADD, AND, OR, XOR")
    print("Steps per operation: 10,000")
    print("Success threshold: 95% accuracy")
    print("="*70)
    print()

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=128,
        depth=3,
        use_dual_pointers=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Log directory
    log_dir = Path(__file__).parent / "logs" / "benchmark"

    # Test each operation
    operations = ['add', 'and', 'or', 'xor']
    results = {}

    for op in operations:
        result = test_single_operation(model, op, num_steps=10000, log_dir=log_dir)
        results[op] = result

        # Short break between operations
        time.sleep(2)

    # Overall summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()

    for op in operations:
        r = results[op]
        status = "PASS" if r['passed'] else "FAIL"
        converge = f"{r['converged_step']}" if r['converged_step'] >= 0 else "Never"
        print(f"{op.upper():4s} | {r['accuracy']*100:5.1f}% | converged: {converge:>6s} | {status}")

    print()

    # Overall pass/fail
    all_passed = all(r['passed'] for r in results.values())
    overall_acc = min(r['accuracy'] for r in results.values())

    print(f"Overall minimum accuracy: {overall_acc*100:.1f}%")
    print(f"Verdict: {'ALL OPERATIONS PASSED' if all_passed else 'FAILED'}")
    print()

    if all_passed:
        print("BASIC COMPETENCE ACHIEVED")
        print("  Model ready for advanced testing")
    else:
        failed = [op for op, r in results.items() if not r['passed']]
        print(f"INSUFFICIENT COMPETENCE")
        print(f"  Failed operations: {', '.join(failed)}")

    print("="*70)

    # Save results JSON
    results_path = log_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'model_config': {
                'embedding_dim': 128,
                'depth': 3,
                'use_dual_pointers': False,
                'parameters': total_params,
            },
            'operations': results,
            'overall': {
                'passed': all_passed,
                'min_accuracy': overall_acc,
            }
        }, f, indent=2)

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
