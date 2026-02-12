"""
Deep debug of TRUE Möbius training dynamics.

Tracks:
- Holonomy flip events and frequency
- Gradient norms (jump_destinations vs others)
- Pointer wrapping behavior
- Phase modulation values
- Loss/accuracy around flip events
- Jump gate activation patterns
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def analyze_gradients(model):
    """Extract gradient norms by parameter group."""
    jump_grad_norm = 0.0
    other_grad_norm = 0.0

    if model.jump_destinations.grad is not None:
        jump_grad_norm = model.jump_destinations.grad.norm().item()

    for name, param in model.named_parameters():
        if 'jump_destinations' not in name and param.grad is not None:
            other_grad_norm += param.grad.norm().item() ** 2

    other_grad_norm = np.sqrt(other_grad_norm)

    return jump_grad_norm, other_grad_norm


def main():
    print("=" * 80)
    print("DEEP DEBUG: TRUE MÖBIUS TRAINING DYNAMICS")
    print("=" * 80)
    print()

    # Config
    num_keys = 2
    num_pairs = 2
    seq_len = 32
    num_positions = 64
    embedding_dim = 64
    batch_size = 100
    num_steps = 500

    print(f"Task: {num_keys} keys, {num_pairs} pairs, seq_len={seq_len}")
    print(f"Model: {num_positions}x{embedding_dim}D, batch_size={batch_size}")
    print(f"Training: {num_steps} steps")
    print()

    # Create model with TRUE Möbius
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=True,  # TRUE Möbius with holonomy
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=seq_len, keys=num_keys, pairs=num_pairs, seed=9999
    )

    # Tracking arrays
    steps = []
    losses = []
    accuracies = []
    jump_grads = []
    other_grads = []
    flip_counts_per_step = []
    wrap_counts_per_step = []
    jump_rates = []
    holonomy_distributions = []  # NEW: Track % of samples with holonomy=+1

    # Detailed flip tracking
    flip_events = []  # List of (step, sample_idx, timestep, old_holonomy, new_holonomy, old_ptr, new_ptr)

    print("Starting training with detailed instrumentation...")
    print()
    print(f"{'Step':<6} {'Loss':<8} {'Acc':<6} {'JumpG':<7} {'OtherG':<7} {'Flips':<6} {'Wraps':<6} {'JumpR':<6}")
    print("-" * 80)

    for step in range(num_steps):
        # Generate fresh training data
        x_train, y_train, _ = generate_assoc_clean(
            n_samples=batch_size, seq_len=seq_len, keys=num_keys, pairs=num_pairs,
            seed=42 + step + 1000000
        )

        # Forward pass with debug info
        optimizer.zero_grad()
        logits, aux_loss, debug_info = model(x_train, return_debug=True)
        loss = F.cross_entropy(logits, y_train) + aux_loss
        loss.backward()

        # Analyze gradients BEFORE optimizer step
        jump_grad, other_grad = analyze_gradients(model)

        optimizer.step()

        # Training accuracy
        train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()

        # Extract debug info
        holonomy_traj = torch.stack(debug_info['holonomy_trajectory'])  # [T, B]
        pointer_traj = torch.stack(debug_info['pointer_trajectory'])  # [T, B]
        jump_decisions = torch.stack(debug_info['jump_decisions'])  # [T, B]

        # Track holonomy distribution (% of samples with holonomy=+1)
        holonomy_pct_positive = (holonomy_traj == 1.0).float().mean().item()
        holonomy_distributions.append(holonomy_pct_positive)

        # Count flips and wraps per sample
        num_flips = 0
        num_wraps = 0

        for b in range(batch_size):
            holonomy_sample = holonomy_traj[:, b]
            pointer_sample = pointer_traj[:, b]

            for t in range(1, seq_len):
                # Flip detection
                if holonomy_sample[t] != holonomy_sample[t-1]:
                    num_flips += 1
                    flip_events.append({
                        'step': step,
                        'sample': b,
                        'timestep': t,
                        'old_holonomy': holonomy_sample[t-1].item(),
                        'new_holonomy': holonomy_sample[t].item(),
                        'old_ptr': pointer_sample[t-1].item(),
                        'new_ptr': pointer_sample[t].item(),
                        'loss': loss.item(),
                    })

                # Wrap detection (position crosses from high to low)
                if pointer_sample[t] < 1.0 and pointer_sample[t-1] >= num_positions - 1.0:
                    num_wraps += 1

        # Jump gate rate
        jump_rate = jump_decisions.float().mean().item()

        # Record metrics
        steps.append(step)
        losses.append(loss.item())
        accuracies.append(train_acc)
        jump_grads.append(jump_grad)
        other_grads.append(other_grad)
        flip_counts_per_step.append(num_flips)
        wrap_counts_per_step.append(num_wraps)
        jump_rates.append(jump_rate)

        # Eval every 50 steps
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(x_eval, return_debug=False)
                eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
            model.train()

            print(f"{step:<6} {loss.item():<8.4f} {eval_acc:<6.3f} {jump_grad:<7.4f} {other_grad:<7.4f} "
                  f"{num_flips:<6} {num_wraps:<6} {jump_rate:<6.3f}")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Summary statistics
    total_flips = sum(flip_counts_per_step)
    total_wraps = sum(wrap_counts_per_step)
    avg_flips_per_step = total_flips / num_steps
    avg_wraps_per_step = total_wraps / num_steps

    print(f"Total holonomy flips: {total_flips}")
    print(f"Total wraps detected: {total_wraps}")
    print(f"Flips per step (avg): {avg_flips_per_step:.1f}")
    print(f"Wraps per step (avg): {avg_wraps_per_step:.1f}")
    print(f"Flip/wrap ratio: {total_flips / max(total_wraps, 1):.2f}")
    print()

    # Expected wraps calculation
    # With 64 positions, 32 timesteps, walking +1 each step:
    # Starting position random in [0, 64), walking 32 steps
    # Expected wraps ≈ 32/64 = 0.5 per sequence on average
    expected_wraps_per_seq = seq_len / num_positions
    expected_wraps_per_batch = expected_wraps_per_seq * batch_size
    print(f"Expected wraps per step (theoretical): {expected_wraps_per_batch:.1f}")
    print(f"Actual vs expected: {avg_wraps_per_step / expected_wraps_per_batch:.2%}")
    print()

    # Gradient analysis
    print(f"Jump grad (mean): {np.mean(jump_grads):.4f}")
    print(f"Other grad (mean): {np.mean(other_grads):.4f}")
    print(f"Jump/Other grad ratio: {np.mean(jump_grads) / max(np.mean(other_grads), 1e-8):.4f}")
    print()

    # Jump gate analysis
    print(f"Jump gate activation (mean): {np.mean(jump_rates):.3f}")
    print(f"Jump gate activation (max): {np.max(jump_rates):.3f}")
    print()

    # Holonomy distribution analysis
    holonomy_mean = np.mean(holonomy_distributions)
    holonomy_std = np.std(holonomy_distributions)
    print(f"Holonomy distribution (% +1):")
    print(f"  Mean: {holonomy_mean:.3f}")
    print(f"  Std:  {holonomy_std:.3f}")
    print(f"  Range: [{np.min(holonomy_distributions):.3f}, {np.max(holonomy_distributions):.3f}]")

    # Calculate correlation between holonomy distribution and accuracy
    correlation = np.corrcoef(holonomy_distributions, accuracies)[0, 1]
    print(f"\nHolonomy-Accuracy Correlation: {correlation:.3f}")
    if abs(correlation) > 0.5:
        print(f"  -> STRONG correlation! Holonomy distribution likely drives oscillations.")
    elif abs(correlation) > 0.3:
        print(f"  -> MODERATE correlation. Holonomy may partially explain oscillations.")
    else:
        print(f"  -> WEAK correlation. Holonomy distribution not the main cause.")
    print()

    # Flip event analysis
    if len(flip_events) > 0:
        print(f"Flip events (first 10):")
        for i, event in enumerate(flip_events[:10]):
            print(f"  [{i}] Step {event['step']}, sample {event['sample']}, t={event['timestep']}: "
                  f"holonomy {event['old_holonomy']:.1f} -> {event['new_holonomy']:.1f}, "
                  f"ptr {event['old_ptr']:.2f} -> {event['new_ptr']:.2f}, loss={event['loss']:.4f}")
        print()

    # Check for gradient spikes around flips
    if len(flip_events) > 0:
        flip_steps = [e['step'] for e in flip_events]
        flip_step_set = set(flip_steps)

        # Compare gradient norms on steps with flips vs without
        flips_jump_grads = [jump_grads[s] for s in flip_step_set if s < len(jump_grads)]
        no_flips_jump_grads = [jump_grads[s] for s in range(len(jump_grads)) if s not in flip_step_set]

        if len(flips_jump_grads) > 0 and len(no_flips_jump_grads) > 0:
            print(f"Gradient norms on steps WITH flips: {np.mean(flips_jump_grads):.4f}")
            print(f"Gradient norms on steps WITHOUT flips: {np.mean(no_flips_jump_grads):.4f}")
            print(f"Spike ratio: {np.mean(flips_jump_grads) / max(np.mean(no_flips_jump_grads), 1e-8):.2f}x")
            print()

    # Create visualizations
    print("Generating visualizations...")
    output_dir = Path(__file__).parent / "logs" / "mobius_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Training dynamics
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(steps, losses, linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title('Loss over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(steps, accuracies, linewidth=0.5, alpha=0.7)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    # Gradient norms
    axes[1, 0].plot(steps, jump_grads, label='Jump destinations', linewidth=0.5, alpha=0.7)
    axes[1, 0].plot(steps, other_grads, label='Other params', linewidth=0.5, alpha=0.7)
    axes[1, 0].set_title('Gradient Norms')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Flips and wraps
    axes[1, 1].plot(steps, flip_counts_per_step, label='Holonomy flips', linewidth=0.5, alpha=0.7)
    axes[1, 1].plot(steps, wrap_counts_per_step, label='Wraps detected', linewidth=0.5, alpha=0.7)
    axes[1, 1].axhline(y=expected_wraps_per_batch, color='r', linestyle='--',
                       label=f'Expected wraps ({expected_wraps_per_batch:.1f})', alpha=0.5)
    axes[1, 1].set_title('Flips and Wraps per Step')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Jump gate activation
    axes[2, 0].plot(steps, jump_rates, linewidth=0.5, alpha=0.7)
    axes[2, 0].set_title('Jump Gate Activation Rate')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Activation Rate')
    axes[2, 0].grid(True, alpha=0.3)

    # Holonomy Distribution vs Accuracy (KEY DIAGNOSTIC)
    ax_hol = axes[2, 1]
    ax_hol.plot(steps, holonomy_distributions, label='% Holonomy +1',
                linewidth=1.5, alpha=0.8, color='#00D9FF')
    ax_hol.plot(steps, accuracies, label='Accuracy',
                linewidth=1.5, alpha=0.8, color='#FFB000', linestyle='--')
    ax_hol.set_title(f'Holonomy Distribution vs Accuracy (r={correlation:.3f})')
    ax_hol.set_xlabel('Step')
    ax_hol.set_ylabel('Value')
    ax_hol.legend()
    ax_hol.grid(True, alpha=0.3)
    ax_hol.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "training_dynamics.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_dynamics.png'}")

    # Figure 2: Flip event distribution
    if len(flip_events) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Flips over training time
        flip_step_counts = {}
        for event in flip_events:
            s = event['step']
            flip_step_counts[s] = flip_step_counts.get(s, 0) + 1

        steps_with_flips = sorted(flip_step_counts.keys())
        counts_with_flips = [flip_step_counts[s] for s in steps_with_flips]

        axes[0, 0].scatter(steps_with_flips, counts_with_flips, alpha=0.5, s=10)
        axes[0, 0].set_title('Flip Events Over Training')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Number of Flips')
        axes[0, 0].grid(True, alpha=0.3)

        # Flip timestep distribution
        flip_timesteps = [e['timestep'] for e in flip_events]
        axes[0, 1].hist(flip_timesteps, bins=32, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Flip Timestep Distribution')
        axes[0, 1].set_xlabel('Timestep within Sequence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Pointer position at flip
        flip_ptr_positions = [e['new_ptr'] for e in flip_events]
        axes[1, 0].hist(flip_ptr_positions, bins=64, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Pointer Position at Flip')
        axes[1, 0].set_xlabel('Pointer Position')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)

        # Holonomy transition matrix
        transitions = {'1_to_-1': 0, '-1_to_1': 0}
        for event in flip_events:
            if event['old_holonomy'] > 0:
                transitions['1_to_-1'] += 1
            else:
                transitions['-1_to_1'] += 1

        axes[1, 1].bar(transitions.keys(), transitions.values(), alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Holonomy Transition Counts')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "flip_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'flip_analysis.png'}")

    print()
    print("=" * 80)
    print("Debug analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
