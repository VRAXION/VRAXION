"""
Multi-Task Benchmark: Learn ALL operations in one training session

Tests if model can learn ADD, AND, OR, XOR simultaneously by looking at
an operation code in the input.

Input format: [a, b, OP_CODE, 0, 0, ...]
Output format: [a, b, result, 0, 0, ...]

OP_CODE encoding (position 2):
- ADD: [1,0,0,0,0,0,0,0] (bit 0 set)
- AND: [0,1,0,0,0,0,0,0] (bit 1 set)
- OR:  [0,0,1,0,0,0,0,0] (bit 2 set)
- XOR: [0,0,0,1,0,0,0,0] (bit 3 set)

Usage:
    python benchmark_multitask.py
"""

import torch
import torch.nn as nn
import sys
import time
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dual_pointer_model import DualPointerByteRingModel
from byte_data import byte_accuracy, bit_accuracy


def int_to_bits(x):
    """Convert integer to 8-bit tensor."""
    bits = []
    for i in range(8):
        bits.append((x >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32)


def generate_multitask_batch(n_samples, seq_len=16, max_value=100, seed=None):
    """
    Generate batch with mixed operations.

    Each sample randomly picks one of: ADD, AND, OR, XOR

    Returns:
        x: [B, T, 8] input sequence
        y: [B, T, 8] target sequence
        ops: [B] operation indices (0=add, 1=and, 2=or, 3=xor)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Operation codes (one-hot in bits 0-3 of position 2)
    op_codes = {
        'add': int_to_bits(1 << 0),  # [1,0,0,0,0,0,0,0]
        'and': int_to_bits(1 << 1),  # [0,1,0,0,0,0,0,0]
        'or':  int_to_bits(1 << 2),  # [0,0,1,0,0,0,0,0]
        'xor': int_to_bits(1 << 3),  # [0,0,0,1,0,0,0,0]
    }

    x_batch = []
    y_batch = []
    op_indices = []

    for _ in range(n_samples):
        # Pick random operation
        op_name = random.choice(['add', 'and', 'or', 'xor'])
        op_idx = {'add': 0, 'and': 1, 'or': 2, 'xor': 3}[op_name]
        op_indices.append(op_idx)

        # Generate operands
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)

        # Compute result based on operation
        if op_name == 'add':
            result = (a + b) % 256
        elif op_name == 'and':
            result = a & b
        elif op_name == 'or':
            result = a | b
        else:  # xor
            result = a ^ b

        # Build input sequence: [a, b, OP_CODE, 0, 0, ...]
        x_seq = torch.zeros(seq_len, 8)
        x_seq[0, :] = int_to_bits(a)
        x_seq[1, :] = int_to_bits(b)
        x_seq[2, :] = op_codes[op_name]  # Operation code

        # Build target sequence: [a, b, result, 0, 0, ...]
        y_seq = torch.zeros(seq_len, 8)
        y_seq[0, :] = int_to_bits(a)
        y_seq[1, :] = int_to_bits(b)
        y_seq[2, :] = int_to_bits(result)

        x_batch.append(x_seq)
        y_batch.append(y_seq)

    return torch.stack(x_batch), torch.stack(y_batch), torch.tensor(op_indices)


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def per_operation_accuracy(output, target, op_indices):
    """Calculate accuracy for each operation separately."""
    accuracies = {}
    for op_idx, op_name in enumerate(['add', 'and', 'or', 'xor']):
        mask = (op_indices == op_idx)
        if mask.sum() > 0:
            op_output = output[mask]
            op_target = target[mask]
            accuracies[op_name] = position_accuracy(op_output, op_target, 2)
        else:
            accuracies[op_name] = 0.0
    return accuracies


def main():
    print("="*70)
    print("MULTI-TASK BENCHMARK: ALL OPERATIONS IN ONE SESSION")
    print("="*70)
    print()
    print("Model learns ADD, AND, OR, XOR simultaneously")
    print("Operation code in input position 2")
    print("Single training run - no resets")
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

    # Training setup
    seq_len = 16
    batch_size = 32
    max_value = 100
    num_steps = 10000
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set (25 samples per operation = 100 total)
    eval_batches = []
    for op_seed in range(25):
        for op_idx in range(4):
            x, y, ops = generate_multitask_batch(
                n_samples=1,
                seq_len=seq_len,
                max_value=max_value,
                seed=9999 + op_seed * 4 + op_idx
            )
            eval_batches.append((x, y, ops))

    x_eval = torch.cat([x for x, _, _ in eval_batches])
    y_eval = torch.cat([y for _, y, _ in eval_batches])
    ops_eval = torch.cat([ops for _, _, ops in eval_batches])

    print(f"Eval set: {len(x_eval)} samples (25 per operation)")
    print()

    # Log setup
    log_dir = Path(__file__).parent / "logs" / "benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "multitask.log"
    if log_path.exists():
        log_path.unlink()

    # Track best
    best_overall = 0.0
    best_step = 0
    converged = {op: -1 for op in ['add', 'and', 'or', 'xor']}

    print(f"Training for {num_steps:,} steps...")
    print(f"Log: {log_path}")
    print()
    print("Dashboard: http://localhost:8501")
    print(f"  Launch: python -m streamlit run diamond_dashboard.py -- --log logs/benchmark/multitask.log")
    print("="*70)
    print()

    start_time = time.time()

    with open(log_path, "w") as log_file:
        for step in range(num_steps):
            step_start = time.time()

            # Generate mixed training batch
            x_train, y_train, ops_train = generate_multitask_batch(
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

                    # Overall accuracy
                    overall_acc = position_accuracy(eval_output, y_eval, 2)
                    byte_acc = byte_accuracy(eval_output, y_eval)
                    bit_acc = bit_accuracy(eval_output, y_eval)
                    jump_rate = eval_stats['jump_gate']

                    # Per-operation accuracy
                    op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)

                model.train()

                # Track best
                if overall_acc > best_overall:
                    best_overall = overall_acc
                    best_step = step

                # Track convergence per operation
                for op_name, acc in op_accs.items():
                    if converged[op_name] == -1 and acc >= 0.95:
                        converged[op_name] = step
                        print(f"  >> {op_name.upper()} reached 95% at step {step}")

                # Log to file
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"overall={overall_acc:.4f} "
                    f"add={op_accs['add']:.4f} and={op_accs['and']:.4f} "
                    f"or={op_accs['or']:.4f} xor={op_accs['xor']:.4f} "
                    f"byte_acc={byte_acc:.4f} bit_acc={bit_acc:.4f} "
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
                        f"overall={overall_acc*100:5.1f}% | "
                        f"add={op_accs['add']*100:5.1f}% and={op_accs['and']*100:5.1f}% "
                        f"or={op_accs['or']*100:5.1f}% xor={op_accs['xor']*100:5.1f}% | "
                        f"time={elapsed:.1f}s"
                    )

    total_time = time.time() - start_time

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_output, _ = model(x_eval, return_stats=True)
        final_overall = position_accuracy(eval_output, y_eval, 2)
        final_op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)

    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Overall accuracy: {final_overall*100:.1f}% (best: {best_overall*100:.1f}% at step {best_step})")
    print()
    print("Per-operation accuracy:")
    for op_name in ['add', 'and', 'or', 'xor']:
        acc = final_op_accs[op_name]
        conv = converged[op_name]
        status = "PASS" if acc >= 0.95 else "FAIL"
        conv_str = f"step {conv}" if conv >= 0 else "Never"
        print(f"  {op_name.upper():4s}: {acc*100:5.1f}% | converged: {conv_str:>10s} | {status}")

    print()
    print(f"Training time: {total_time:.1f}s")
    print()

    # Overall pass/fail
    all_passed = all(final_op_accs[op] >= 0.95 for op in ['add', 'and', 'or', 'xor'])

    if all_passed:
        print("SUCCESS! Model learned all 4 operations in one session")
        print("  Multi-task learning validated")
    elif final_overall >= 0.75:
        print("PARTIAL SUCCESS! Model learned multiple operations")
        failed = [op for op in ['add', 'and', 'or', 'xor'] if final_op_accs[op] < 0.95]
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("FAILURE! Multi-task learning too difficult")
        print(f"  Only {final_overall*100:.1f}% overall accuracy")

    print("="*70)

    # Save results
    results = {
        'model_config': {
            'embedding_dim': 128,
            'depth': 3,
            'use_dual_pointers': False,
            'parameters': total_params,
        },
        'training': {
            'num_steps': num_steps,
            'time_seconds': total_time,
        },
        'results': {
            'overall': final_overall,
            'operations': final_op_accs,
            'convergence': converged,
        },
        'passed': all_passed,
    }

    results_path = log_dir / "multitask_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
