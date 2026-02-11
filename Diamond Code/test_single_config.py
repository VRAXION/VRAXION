"""
Test a single model configuration on multi-task benchmark.

Usage:
    python test_single_config.py --embedding 64 --depth 2

Logs everything (loss, accuracy, jump gates) for dashboard viewing.
"""

import torch
import torch.nn as nn
import sys
import time
import json
import argparse
from pathlib import Path
import random

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
    """Generate batch with mixed operations."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    op_codes = {
        'add': int_to_bits(1 << 0),
        'and': int_to_bits(1 << 1),
        'or':  int_to_bits(1 << 2),
        'xor': int_to_bits(1 << 3),
    }

    x_batch = []
    y_batch = []
    op_indices = []

    for _ in range(n_samples):
        op_name = random.choice(['add', 'and', 'or', 'xor'])
        op_idx = {'add': 0, 'and': 1, 'or': 2, 'xor': 3}[op_name]
        op_indices.append(op_idx)

        a = random.randint(0, max_value)
        b = random.randint(0, max_value)

        if op_name == 'add':
            result = (a + b) % 256
        elif op_name == 'and':
            result = a & b
        elif op_name == 'or':
            result = a | b
        else:
            result = a ^ b

        x_seq = torch.zeros(seq_len, 8)
        x_seq[0, :] = int_to_bits(a)
        x_seq[1, :] = int_to_bits(b)
        x_seq[2, :] = op_codes[op_name]

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
    parser = argparse.ArgumentParser(description='Test single model config')
    parser.add_argument('--embedding', type=int, default=64, help='Embedding dimension (default: 64)')
    parser.add_argument('--depth', type=int, default=2, help='Number of layers (default: 2)')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps (default: 5000)')
    args = parser.parse_args()

    embedding_dim = args.embedding
    depth = args.depth
    num_steps = args.steps

    print("="*70)
    print("SINGLE CONFIG TEST: Multi-Task Benchmark")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Embedding: {embedding_dim}D")
    print(f"  Depth: {depth} layers")
    print(f"  Steps: {num_steps:,}")
    print("="*70)
    print()

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=embedding_dim,
        depth=depth,
        use_dual_pointers=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Training setup
    seq_len = 16
    batch_size = 32
    max_value = 100
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set (25 samples per operation = 100 total)
    eval_batches = []
    for op_seed in range(25):
        for op_idx in range(4):
            x, y, ops = generate_multitask_batch(
                n_samples=1, seq_len=seq_len, max_value=max_value,
                seed=9999 + op_seed * 4 + op_idx
            )
            eval_batches.append((x, y, ops))

    x_eval = torch.cat([x for x, _, _ in eval_batches])
    y_eval = torch.cat([y for _, y, _ in eval_batches])
    ops_eval = torch.cat([ops for _, _, ops in eval_batches])

    print(f"Eval set: {len(x_eval)} samples (25 per operation)")
    print()

    # Log setup
    log_dir = Path(__file__).parent / "logs" / "config_test"
    log_dir.mkdir(parents=True, exist_ok=True)

    config_name = f"{embedding_dim}d_{depth}layers"
    log_path = log_dir / f"{config_name}.log"
    if log_path.exists():
        log_path.unlink()

    print(f"Log: {log_path}")
    print()
    print("Dashboard: http://localhost:8501")
    print(f"  Launch: python -m streamlit run diamond_dashboard.py -- --log logs/config_test/{config_name}.log")
    print("="*70)
    print()

    # Track best and convergence
    best_overall = 0.0
    best_step = 0
    converged = {op: -1 for op in ['add', 'and', 'or', 'xor']}

    start_time = time.time()

    with open(log_path, "w") as log_file:
        for step in range(num_steps):
            step_start = time.time()

            # Generate training batch
            x_train, y_train, _ = generate_multitask_batch(
                n_samples=batch_size, seq_len=seq_len, max_value=max_value,
                seed=42 + step + 1000000
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
                    overall_acc = position_accuracy(eval_output, y_eval, 2)
                    op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)
                    jump_rate = eval_stats['jump_gate']
                model.train()

                # Track best
                if overall_acc > best_overall:
                    best_overall = overall_acc
                    best_step = step

                # Track convergence
                for op_name, acc in op_accs.items():
                    if converged[op_name] == -1 and acc >= 0.95:
                        converged[op_name] = step
                        print(f"  >> {op_name.upper()} reached 95% at step {step}")

                # Log to file (dashboard format)
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"overall={overall_acc:.4f} "
                    f"add={op_accs['add']:.4f} and={op_accs['and']:.4f} "
                    f"or={op_accs['or']:.4f} xor={op_accs['xor']:.4f} "
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
                        f"add={op_accs['add']*100:4.0f}% and={op_accs['and']*100:4.0f}% "
                        f"or={op_accs['or']*100:4.0f}% xor={op_accs['xor']*100:4.0f}% | "
                        f"jump={jump_rate*100:4.0f}% | time={elapsed:.1f}s"
                    )

    total_time = time.time() - start_time

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_output, eval_stats = model(x_eval, return_stats=True)
        final_overall = position_accuracy(eval_output, y_eval, 2)
        final_op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)
        final_jump = eval_stats['jump_gate']

    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Config: {embedding_dim}D, {depth} layers ({total_params:,} params)")
    print(f"Overall accuracy: {final_overall*100:.1f}% (best: {best_overall*100:.1f}% at step {best_step})")
    print()
    print("Per-operation:")
    for op_name in ['add', 'and', 'or', 'xor']:
        acc = final_op_accs[op_name]
        conv = converged[op_name]
        status = "PASS" if acc >= 0.95 else "FAIL"
        conv_str = f"step {conv}" if conv >= 0 else "Never"
        print(f"  {op_name.upper():4s}: {acc*100:5.1f}% | converged: {conv_str:>10s} | {status}")
    print()
    print(f"Final jump gate: {final_jump*100:.1f}%")
    print(f"Training time: {total_time:.1f}s")
    print()

    # Pass/fail
    passed = final_overall >= 0.95
    if passed:
        print("STATUS: PASS - Model achieved multi-task competence")
    else:
        print(f"STATUS: FAIL - Only {final_overall*100:.1f}% accuracy")

    print("="*70)

    # Save results
    result = {
        'config': {
            'embedding_dim': embedding_dim,
            'depth': depth,
            'parameters': total_params,
        },
        'final': {
            'overall_acc': final_overall,
            'add_acc': final_op_accs['add'],
            'and_acc': final_op_accs['and'],
            'or_acc': final_op_accs['or'],
            'xor_acc': final_op_accs['xor'],
            'jump_gate': final_jump,
            'best_overall': best_overall,
        },
        'convergence': converged,
        'training_time': total_time,
        'passed': passed,
    }

    result_path = log_dir / f"{config_name}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
