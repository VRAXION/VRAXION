"""
Minimal Model Sweep: Find where multi-task learning breaks

Systematically reduce embedding dimension and depth to find minimal
viable architecture for multi-task learning.

Tests:
1. Baseline: 128D, 3 layers (known working)
2. Half embedding: 64D, 3 layers
3. Quarter embedding: 32D, 3 layers
4. Shallow: 128D, 2 layers
5. Very shallow: 128D, 1 layer
6. Half + shallow: 64D, 2 layers
7. Minimal: 32D, 1 layer

All results saved automatically (logs, plots, summary JSON).
"""

import torch
import torch.nn as nn
import sys
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

sys.path.insert(0, str(Path(__file__).parent))

from dual_pointer_model import DualPointerByteRingModel
from byte_data import byte_accuracy, bit_accuracy
import random


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


def test_config(embedding_dim, depth, num_steps=5000, sweep_dir=None):
    """
    Test a single model configuration.

    Returns dict with all metrics and history.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {embedding_dim}D embedding, {depth} layers")
    print(f"{'='*70}")

    # Create model
    torch.manual_seed(42)
    model = DualPointerByteRingModel(
        num_memory_positions=64,
        embedding_dim=embedding_dim,
        depth=depth,
        use_dual_pointers=False,
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

    # History tracking
    history = {
        'steps': [],
        'loss': [],
        'overall_acc': [],
        'add_acc': [],
        'and_acc': [],
        'or_acc': [],
        'xor_acc': [],
        'jump_gate': [],
    }

    # Track convergence
    converged = {op: -1 for op in ['add', 'and', 'or', 'xor']}
    best_overall = 0.0

    start_time = time.time()

    for step in range(num_steps):
        # Generate training batch
        x_train, y_train, _ = generate_multitask_batch(
            n_samples=batch_size, seq_len=seq_len, max_value=max_value,
            seed=42 + step + 1000000
        )

        # Train
        optimizer.zero_grad()
        output = model(x_train)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)
        loss.backward()
        optimizer.step()

        # Eval
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_output, eval_stats = model(x_eval, return_stats=True)
                overall_acc = position_accuracy(eval_output, y_eval, 2)
                op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)
                jump_rate = eval_stats['jump_gate']
            model.train()

            # Record history
            history['steps'].append(step)
            history['loss'].append(loss.item())
            history['overall_acc'].append(overall_acc)
            history['add_acc'].append(op_accs['add'])
            history['and_acc'].append(op_accs['and'])
            history['or_acc'].append(op_accs['or'])
            history['xor_acc'].append(op_accs['xor'])
            history['jump_gate'].append(jump_rate)

            # Track convergence
            for op_name, acc in op_accs.items():
                if converged[op_name] == -1 and acc >= 0.95:
                    converged[op_name] = step
                    print(f"  {op_name.upper()} reached 95% at step {step}")

            if overall_acc > best_overall:
                best_overall = overall_acc

            # Console every 1000 steps
            if step % 1000 == 0:
                print(
                    f"step {step:5d} | loss {loss.item():.6f} | "
                    f"overall={overall_acc*100:5.1f}% | "
                    f"add={op_accs['add']*100:4.0f}% and={op_accs['and']*100:4.0f}% "
                    f"or={op_accs['or']*100:4.0f}% xor={op_accs['xor']*100:4.0f}% | "
                    f"jump={jump_rate*100:4.0f}%"
                )

    total_time = time.time() - start_time

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_output, _ = model(x_eval, return_stats=True)
        final_overall = position_accuracy(eval_output, y_eval, 2)
        final_op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)

    # Results
    result = {
        'config': {
            'embedding_dim': embedding_dim,
            'depth': depth,
            'parameters': total_params,
        },
        'training': {
            'num_steps': num_steps,
            'time_seconds': total_time,
        },
        'final': {
            'overall_acc': final_overall,
            'add_acc': final_op_accs['add'],
            'and_acc': final_op_accs['and'],
            'or_acc': final_op_accs['or'],
            'xor_acc': final_op_accs['xor'],
            'best_overall': best_overall,
        },
        'convergence': converged,
        'history': history,
        'passed': final_overall >= 0.95,
    }

    print(f"\nFinal: {final_overall*100:.1f}% overall | Best: {best_overall*100:.1f}%")
    print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")

    # Save individual result
    if sweep_dir:
        config_name = f"{embedding_dim}d_{depth}layers"
        result_path = sweep_dir / f"{config_name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {result_path}")

    return result


def plot_sweep_results(results, sweep_dir):
    """Generate comparison plots for all configs."""

    # Figure 1: Final accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    configs = [f"{r['config']['embedding_dim']}D\n{r['config']['depth']}L" for r in results]
    params = [r['config']['parameters'] for r in results]
    overall_accs = [r['final']['overall_acc'] * 100 for r in results]

    colors = ['green' if r['passed'] else 'red' for r in results]

    # Overall accuracy bar chart
    ax1.bar(configs, overall_accs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=95, color='blue', linestyle='--', label='95% threshold')
    ax1.set_ylabel('Overall Accuracy (%)')
    ax1.set_title('Multi-Task Performance by Configuration')
    ax1.set_ylim([0, 105])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Accuracy vs parameters scatter
    ax2.scatter(params, overall_accs, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.axhline(y=95, color='blue', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('Overall Accuracy (%)')
    ax2.set_title('Accuracy vs Model Size')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Annotate points
    for i, (p, acc, cfg) in enumerate(zip(params, overall_accs, configs)):
        ax2.annotate(cfg.replace('\n', ' '), (p, acc),
                     textcoords="offset points", xytext=(5,5), fontsize=8)

    plt.tight_layout()
    plt.savefig(sweep_dir / 'sweep_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {sweep_dir / 'sweep_comparison.png'}")
    plt.close()

    # Figure 2: Per-operation accuracy heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    ops = ['add', 'and', 'or', 'xor']
    op_data = []
    for r in results:
        op_data.append([r['final'][f'{op}_acc'] * 100 for op in ops])

    im = ax.imshow(op_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([op.upper() for op in ops])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title('Per-Operation Accuracy Heatmap (%)')

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(ops)):
            text = ax.text(j, i, f'{op_data[i][j]:.0f}',
                          ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(sweep_dir / 'per_operation_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {sweep_dir / 'per_operation_heatmap.png'}")
    plt.close()

    # Figure 3: Training curves for all configs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i] if i < 4 else None
        if ax is None:
            continue

        h = r['history']
        config_name = f"{r['config']['embedding_dim']}D, {r['config']['depth']}L"

        ax.plot(h['steps'], [a*100 for a in h['overall_acc']],
                label='Overall', linewidth=2, color='black')
        ax.plot(h['steps'], [a*100 for a in h['add_acc']], label='ADD', alpha=0.7)
        ax.plot(h['steps'], [a*100 for a in h['and_acc']], label='AND', alpha=0.7)
        ax.plot(h['steps'], [a*100 for a in h['or_acc']], label='OR', alpha=0.7)
        ax.plot(h['steps'], [a*100 for a in h['xor_acc']], label='XOR', alpha=0.7)

        ax.axhline(y=95, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{config_name} ({r["config"]["parameters"]:,} params)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

    # Hide unused subplots
    for i in range(len(results), 4):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(sweep_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {sweep_dir / 'training_curves.png'}")
    plt.close()


def main():
    print("="*70)
    print("MINIMAL MODEL SWEEP: Find Breaking Point")
    print("="*70)
    print()
    print("Testing 7 configurations to find minimal viable architecture")
    print("All results saved automatically (no manual dashboard needed)")
    print("="*70)

    # Create sweep directory
    sweep_dir = Path(__file__).parent / "logs" / "sweep_multitask"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Test configurations (embedding_dim, depth)
    configs = [
        (128, 3),  # Baseline (known working)
        (64, 3),   # Half embedding
        (128, 2),  # Shallow
        (64, 2),   # Half + shallow
        (32, 3),   # Quarter embedding
        (128, 1),  # Very shallow
        (32, 1),   # Minimal
    ]

    results = []

    for embedding_dim, depth in configs:
        result = test_config(
            embedding_dim=embedding_dim,
            depth=depth,
            num_steps=5000,  # 5K steps for speed
            sweep_dir=sweep_dir
        )
        results.append(result)

    # Generate plots
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    plot_sweep_results(results, sweep_dir)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<15} {'Params':<10} {'Overall':<10} {'Status'}")
    print("-"*70)

    for r in results:
        config = f"{r['config']['embedding_dim']}D, {r['config']['depth']}L"
        params = f"{r['config']['parameters']:,}"
        overall = f"{r['final']['overall_acc']*100:.1f}%"
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{config:<15} {params:<10} {overall:<10} {status}")

    # Find minimal working config
    working = [r for r in results if r['passed']]
    if working:
        minimal = min(working, key=lambda r: r['config']['parameters'])
        print()
        print(f"Minimal working config: {minimal['config']['embedding_dim']}D, "
              f"{minimal['config']['depth']}L ({minimal['config']['parameters']:,} params)")
    else:
        print("\nNo configurations passed!")

    # Save summary
    summary = {
        'configs_tested': len(results),
        'configs_passed': len(working),
        'results': results,
        'minimal_working': minimal['config'] if working else None,
    }

    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {sweep_dir}")
    print(f"  - Individual configs: *.json")
    print(f"  - Comparison plots: *.png")
    print(f"  - Summary: sweep_summary.json")
    print("="*70)


if __name__ == "__main__":
    main()
