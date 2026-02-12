"""Loss Root Cause Analyzer for VRAXION

Analyzes per-sequence loss patterns from training traces to identify:
- Hard sequences that consistently cause high loss
- Correlations between sequence properties and loss
- Loss spike events and their causes

Usage:
  python tools/_scratch/loss_root_cause_analyzer.py \\
      --trace traces/current/train_steps_trace.jsonl \\
      --plot-out scratch/loss_analysis.png \\
      --hard-batch-out scratch/hard_batch.pt \\
      --threshold 1.5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


def find_hard_sequences(trace_path: str, threshold: float = 2.0) -> Dict:
    """Find sequences that consistently cause high loss.

    Args:
        trace_path: Path to train_steps_trace.jsonl
        threshold: Multiplier on global mean loss (sequences > threshold * mean are flagged)

    Returns:
        Dict mapping hash -> {"mean_loss": float, "count": int, "targets": List[int], "std_loss": float}
    """
    print(f"[1/4] Finding hard sequences (threshold={threshold}x mean)...")

    sequence_stats = defaultdict(lambda: {"losses": [], "targets": []})
    global_losses = []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)
            batch_losses = trace.get("batch_losses")
            batch_targets = trace.get("batch_targets")
            batch_hashes = trace.get("batch_hashes")

            if batch_losses and batch_targets and batch_hashes:
                for loss, target, h in zip(batch_losses, batch_targets, batch_hashes):
                    sequence_stats[h]["losses"].append(loss)
                    sequence_stats[h]["targets"].append(target)
                    global_losses.append(loss)

    global_mean = np.mean(global_losses)
    global_std = np.std(global_losses)
    threshold_loss = threshold * global_mean

    print(f"  Global loss: {global_mean:.4f} ± {global_std:.4f}")
    print(f"  Threshold: {threshold_loss:.4f}")
    print(f"  Unique sequences: {len(sequence_stats)}")

    # Compute stats for each sequence
    hard_sequences = {}
    for h, data in sequence_stats.items():
        losses = data["losses"]
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        if mean_loss > threshold_loss:
            hard_sequences[h] = {
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "count": len(losses),
                "targets": data["targets"],
                "severity": mean_loss / global_mean  # How many times worse than average
            }

    print(f"  Hard sequences found: {len(hard_sequences)} ({100*len(hard_sequences)/max(1,len(sequence_stats)):.1f}%)")

    return hard_sequences


def correlate_sequence_properties(trace_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Correlate sequence properties with loss.

    For assoc_clean task: target value might correlate with loss.

    Returns:
        Tuple of (property_values, loss_values) for scatter plotting
    """
    print(f"[2/4] Correlating sequence properties with loss...")

    targets = []
    losses = []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)
            batch_losses = trace.get("batch_losses")
            batch_targets = trace.get("batch_targets")

            if batch_losses and batch_targets:
                targets.extend(batch_targets)
                losses.extend(batch_losses)

    targets = np.array(targets)
    losses = np.array(losses)

    if len(targets) > 0 and len(losses) > 0:
        pearson_r, pearson_p = stats.pearsonr(targets, losses)
        spearman_r, spearman_p = stats.spearmanr(targets, losses)

        print(f"  Target vs Loss correlation:")
        print(f"    Pearson:  r={pearson_r:+.3f}, p={pearson_p:.4f}")
        print(f"    Spearman: ρ={spearman_r:+.3f}, p={spearman_p:.4f}")

    return targets, losses


def detect_loss_spikes(trace_path: str, sigma: float = 2.0) -> List[Dict]:
    """Detect and group loss spike events.

    Args:
        trace_path: Path to train_steps_trace.jsonl
        sigma: Number of std deviations above mean to flag as spike

    Returns:
        List of spike events: [{"step": int, "batch_mean_loss": float, "examples": [...]}]
    """
    print(f"[3/4] Detecting loss spikes (σ={sigma})...")

    step_losses = []
    step_nums = []
    step_examples = []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            trace = json.loads(line)
            batch_losses = trace.get("batch_losses")
            batch_targets = trace.get("batch_targets")
            batch_hashes = trace.get("batch_hashes")
            step = trace.get("step")

            if batch_losses and step is not None:
                batch_mean = np.mean(batch_losses)
                step_losses.append(batch_mean)
                step_nums.append(step)

                # Store examples for visualization
                examples = []
                if batch_targets and batch_hashes:
                    for loss, target, h in zip(batch_losses, batch_targets, batch_hashes):
                        examples.append({"loss": loss, "target": target, "hash": h})
                step_examples.append(examples)

    global_mean = np.mean(step_losses)
    global_std = np.std(step_losses)
    threshold = global_mean + sigma * global_std

    print(f"  Global batch mean: {global_mean:.4f} ± {global_std:.4f}")
    print(f"  Spike threshold: {threshold:.4f}")

    # Find spikes
    spikes = []
    for step, loss, examples in zip(step_nums, step_losses, step_examples):
        if loss > threshold:
            spikes.append({
                "step": step,
                "batch_mean_loss": loss,
                "severity": (loss - global_mean) / global_std,
                "examples": examples
            })

    print(f"  Spikes found: {len(spikes)} ({100*len(spikes)/max(1,len(step_losses)):.1f}% of steps)")

    # Group consecutive spikes into events
    events = []
    if spikes:
        current_event = [spikes[0]]
        for spike in spikes[1:]:
            if spike["step"] - current_event[-1]["step"] <= 5:  # Within 5 steps = same event
                current_event.append(spike)
            else:
                events.append(current_event)
                current_event = [spike]
        events.append(current_event)

        print(f"  Spike events: {len(events)} (grouped consecutive spikes)")

    return spikes


def export_hard_batch(trace_path: str, out_path: str, top_n: int = 100) -> None:
    """Export hardest sequences to .pt file for focused training.

    Args:
        trace_path: Path to train_steps_trace.jsonl
        out_path: Path to save .pt file
        top_n: Number of hardest sequences to export
    """
    print(f"[4/4] Exporting top {top_n} hardest sequences...")

    # This requires the actual input tensors, which aren't in the trace
    # For now, just export the hash/target pairs for manual reconstruction
    hard_seqs = find_hard_sequences(trace_path, threshold=1.0)

    # Sort by severity
    sorted_seqs = sorted(hard_seqs.items(), key=lambda x: x[1]["severity"], reverse=True)
    top_seqs = sorted_seqs[:top_n]

    # Export metadata (hash, target, loss stats)
    export_data = {
        "hashes": [h for h, _ in top_seqs],
        "mean_losses": [data["mean_loss"] for _, data in top_seqs],
        "targets": [data["targets"][0] if data["targets"] else None for _, data in top_seqs],
        "counts": [data["count"] for _, data in top_seqs],
    }

    torch.save(export_data, out_path)
    print(f"  Saved to: {out_path}")
    print(f"  Note: Contains metadata only (hash/target/loss). Reconstruct inputs from dataset.")


def plot_analysis(trace_path: str, out_path: str, threshold: float = 1.5, sigma: float = 2.0) -> None:
    """Generate comprehensive analysis plots.

    Args:
        trace_path: Path to train_steps_trace.jsonl
        out_path: Path to save plot image
        threshold: Threshold for hard sequence detection
        sigma: Sigma for spike detection
    """
    print(f"\nGenerating plots...")

    # Run all analyses
    hard_seqs = find_hard_sequences(trace_path, threshold)
    targets, losses = correlate_sequence_properties(trace_path)
    spikes = detect_loss_spikes(trace_path, sigma)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Loss Root Cause Analysis", fontsize=16, fontweight='bold')

    # Top-left: Histogram of per-sequence mean loss
    ax1 = axes[0, 0]
    if hard_seqs:
        all_mean_losses = [data["mean_loss"] for data in hard_seqs.values()]
        ax1.hist(all_mean_losses, bins=30, color='#ff00ff', alpha=0.7, edgecolor='black')
        ax1.axvline(threshold * np.mean(losses), color='#00e5ff', linestyle='--', linewidth=2, label=f'{threshold}x mean')
        ax1.set_xlabel("Mean Loss per Sequence")
        ax1.set_ylabel("Count")
        ax1.set_title("A: Hard Sequence Distribution")
        ax1.legend()
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No hard sequences found", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("A: Hard Sequence Distribution")

    # Top-right: Timeline of loss spike events
    ax2 = axes[0, 1]
    if spikes:
        spike_steps = [s["step"] for s in spikes]
        spike_losses = [s["batch_mean_loss"] for s in spikes]
        spike_severities = [s["severity"] for s in spikes]

        scatter = ax2.scatter(spike_steps, spike_losses, c=spike_severities,
                            cmap='viridis', s=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Batch Mean Loss")
        ax2.set_title("B: Loss Spike Timeline")
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Severity (σ)')
    else:
        ax2.text(0.5, 0.5, "No spikes detected", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("B: Loss Spike Timeline")

    # Bottom-left: Scatter plot of target vs loss
    ax3 = axes[1, 0]
    if len(targets) > 0 and len(losses) > 0:
        # Subsample for readability
        if len(targets) > 5000:
            indices = np.random.choice(len(targets), 5000, replace=False)
            targets_sub = targets[indices]
            losses_sub = losses[indices]
        else:
            targets_sub = targets
            losses_sub = losses

        ax3.scatter(targets_sub, losses_sub, alpha=0.3, s=10, color='#8b1a8b')

        # Add trend line
        z = np.polyfit(targets, losses, 1)
        p = np.poly1d(z)
        target_range = np.linspace(targets.min(), targets.max(), 100)
        ax3.plot(target_range, p(target_range), "#00e5ff", linewidth=2, linestyle='--', label='Linear fit')

        pearson_r, _ = stats.pearsonr(targets, losses)
        ax3.set_xlabel("Target Value")
        ax3.set_ylabel("Loss")
        ax3.set_title(f"C: Target vs Loss (r={pearson_r:+.3f})")
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No correlation data", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("C: Target vs Loss")

    # Bottom-right: Top 10 hardest sequences
    ax4 = axes[1, 1]
    if hard_seqs:
        sorted_seqs = sorted(hard_seqs.items(), key=lambda x: x[1]["mean_loss"], reverse=True)
        top_10 = sorted_seqs[:10]

        hashes_short = [str(h)[:8] + "..." for h, _ in top_10]
        mean_losses_top = [data["mean_loss"] for _, data in top_10]
        counts = [data["count"] for _, data in top_10]

        y_pos = np.arange(len(top_10))
        bars = ax4.barh(y_pos, mean_losses_top, color='#ff00ff', alpha=0.7, edgecolor='black')

        # Annotate with counts
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax4.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f' n={count}', va='center', fontsize=9)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(hashes_short, fontsize=8)
        ax4.set_xlabel("Mean Loss")
        ax4.set_title("D: Top 10 Hardest Sequences")
        ax4.grid(alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, "No hard sequences", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("D: Top 10 Hardest Sequences")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze loss patterns from training traces")
    parser.add_argument("--trace", required=True, help="Path to train_steps_trace.jsonl")
    parser.add_argument("--plot-out", default="scratch/loss_analysis.png", help="Output plot path")
    parser.add_argument("--hard-batch-out", default="scratch/hard_batch.pt", help="Output hard batch path")
    parser.add_argument("--threshold", type=float, default=1.5, help="Hard sequence threshold (multiplier on mean)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Spike detection sigma")
    parser.add_argument("--top-n", type=int, default=100, help="Number of hard sequences to export")

    args = parser.parse_args()

    print("="*60)
    print("LOSS ROOT CAUSE ANALYZER")
    print("="*60)
    print(f"Trace file: {args.trace}")
    print(f"Plot output: {args.plot_out}")
    print(f"Hard batch output: {args.hard_batch_out}")
    print("="*60 + "\n")

    # Ensure output directories exist
    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.hard_batch_out).parent.mkdir(parents=True, exist_ok=True)

    # Generate comprehensive analysis plot
    plot_analysis(args.trace, args.plot_out, args.threshold, args.sigma)

    # Export hard batch
    export_hard_batch(args.trace, args.hard_batch_out, args.top_n)

    print("="*60)
    print("✓ Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
