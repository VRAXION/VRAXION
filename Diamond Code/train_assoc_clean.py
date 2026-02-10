"""
Train Diamond Code on assoc_clean task for direct comparison with VRAXION.

VRAXION Baseline (AbsoluteHallway, 2,820 params): 64.8% accuracy

This script tests if Diamond Code's learned routing helps on associative recall.
"""

import torch
import argparse
import time
import sys
from pathlib import Path

# Add Diamond Code to path
sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from visualize_routing import detect_loops
from assoc_clean_data import generate_assoc_clean


def main():
    parser = argparse.ArgumentParser(description="Train Diamond Code on assoc_clean")
    parser.add_argument('--steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--log', type=str, default='logs/diamond/assoc_clean.log', help='Log file path')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--assoc-keys', type=int, default=2, help='Number of keys')
    parser.add_argument('--assoc-pairs', type=int, default=1, help='Number of pairs per sequence')
    parser.add_argument('--num-positions', type=int, default=64, help='Memory ring positions')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--routing-update-freq', type=int, default=50,
                        help='Update routing metrics every N steps')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                        help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create log directory if needed
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directory if needed
    if args.checkpoint_freq > 0:
        checkpoint_path = Path(args.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_file = open(args.log, 'w', encoding='utf-8')

    def log(msg):
        """Write to both console and file."""
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    # Set random seed
    torch.manual_seed(args.seed)

    # Generate assoc_clean data
    log("=" * 70)
    log("DIAMOND CODE - Associative Recall Benchmark")
    log("=" * 70)
    log(f"Generating assoc_clean dataset...")

    x_data, y_labels, actual_seq_len = generate_assoc_clean(
        n_samples=args.batch_size,
        seq_len=args.seq_len,
        keys=args.assoc_keys,
        pairs=args.assoc_pairs,
        seed=args.seed
    )

    log(f"Task: assoc_clean (associative recall)")
    log(f"Data: {x_data.shape}, Labels: {y_labels.shape}, Classes: 2 (binary)")
    log(f"Keys: {args.assoc_keys}, Pairs: {args.assoc_pairs}, Seq len: {actual_seq_len}")
    log(f"Model: {args.num_positions} positions, {args.embedding_dim}D embeddings")
    log(f"Training: {args.steps} steps, lr={args.lr}, batch_size={args.batch_size}")
    log(f"Log file: {args.log}")
    log("")
    log("VRAXION BASELINE (AbsoluteHallway, 2,820 params): 64.8% accuracy")
    log("Goal: Match or exceed VRAXION's performance")
    log("=" * 70)
    log("")

    # Create model
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,  # Binary classification
        num_memory_positions=args.num_positions,
        embedding_dim=args.embedding_dim,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,}")
    log("")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Track initial jump destinations for parameter delta
    initial_jump_dest = model.jump_destinations.detach().clone()

    log("Starting training...")
    log("")

    # Tracking
    max_acc = 0.0
    acc_history = []

    # Training loop
    for step in range(args.steps):
        step_start = time.time()

        # Forward pass with debug info
        optimizer.zero_grad()
        logits, aux_loss, debug_info = model(x_data, return_debug=True)
        loss = torch.nn.functional.cross_entropy(logits, y_labels) + aux_loss

        # Backward pass
        loss.backward()

        # Compute metrics before optimizer step
        with torch.no_grad():
            # Accuracy
            acc = (logits.argmax(dim=1) == y_labels).float().mean().item()
            acc_history.append(acc)
            if acc > max_acc:
                max_acc = acc

            # Jump gate activation rate
            if step > 0 and 'pointer_trajectory' in debug_info:
                ptr_traj = debug_info['pointer_trajectory']
                if len(ptr_traj) > 1:
                    jumps = 0
                    total_moves = 0
                    for i in range(1, len(ptr_traj)):
                        prev_pos = ptr_traj[i-1].mean().item()
                        curr_pos = ptr_traj[i].mean().item()
                        delta = abs(curr_pos - prev_pos)
                        if delta > 1.5 and delta < (args.num_positions - 1.5):
                            jumps += 1
                        total_moves += 1
                    jump_gate_rate = jumps / max(total_moves, 1)
                else:
                    jump_gate_rate = 0.0
            else:
                jump_gate_rate = 0.0

            # Attention entropy
            if 'attention_entropy' in debug_info and len(debug_info['attention_entropy']) > 0:
                att_ent = sum(debug_info['attention_entropy']) / len(debug_info['attention_entropy'])
            else:
                att_ent = 0.0

            # Pointer position
            if 'pointer_trajectory' in debug_info and len(debug_info['pointer_trajectory']) > 0:
                ptr_pos = debug_info['pointer_trajectory'][-1].mean().item()
            else:
                ptr_pos = 0.0

            # Gradient norms
            jump_grad_norm = model.jump_destinations.grad.norm().item() if model.jump_destinations.grad is not None else 0.0
            other_grad_norm = 0.0
            for name, param in model.named_parameters():
                if 'jump_destinations' not in name and param.grad is not None:
                    other_grad_norm += param.grad.norm().item() ** 2
            other_grad_norm = other_grad_norm ** 0.5

        # Optimizer step
        optimizer.step()

        # Timing
        step_time = time.time() - step_start

        # Log training metrics (every step)
        log(f"step {step} | loss {loss.item():.6f} | "
            f"acc={acc:.4f} | jump_gate={jump_gate_rate:.2f} | "
            f"att_ent={att_ent:.2f} | ptr_pos={ptr_pos:.1f} | "
            f"s_per_step={step_time:.3f}")

        # Log routing metrics (every N steps)
        if step % args.routing_update_freq == 0 and step > 0:
            with torch.no_grad():
                cycles, self_loops = detect_loops(model.jump_destinations)
                num_cycles = len(cycles)
                num_self_loops = len(self_loops)

                log(f"routing_update step={step} | "
                    f"cyc={num_cycles} | sl={num_self_loops} | "
                    f"grad_j={jump_grad_norm:.3f} | grad_o={other_grad_norm:.3f}")

        # Save checkpoint (every N steps)
        if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0 and step > 0:
            checkpoint_file = checkpoint_path / f"assoc_clean_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'acc': acc,
                'max_acc': max_acc,
            }, checkpoint_file)
            log(f"checkpoint saved: {checkpoint_file}")

    # Final summary
    log("")
    log("=" * 70)
    log("Training Complete")
    log("=" * 70)

    with torch.no_grad():
        # Final accuracy
        logits, _, _ = model(x_data)
        final_acc = (logits.argmax(dim=1) == y_labels).float().mean().item()

        # Final routing stats
        cycles, self_loops = detect_loops(model.jump_destinations)

        # Parameter delta
        param_delta = (model.jump_destinations - initial_jump_dest).norm().item()

        # Compute mean accuracy over last 100 steps
        last_100_acc = sum(acc_history[-100:]) / min(100, len(acc_history))

    log(f"Final accuracy: {final_acc*100:.1f}%")
    log(f"Best accuracy: {max_acc*100:.1f}%")
    log(f"Mean accuracy (last 100 steps): {last_100_acc*100:.1f}%")
    log("")
    log(f"Refinement stations (self-loops): {len(self_loops)}")
    log(f"Detected cycles: {len(cycles)}")
    log(f"Jump destination parameter delta: {param_delta:.4f}")
    log("")

    # Comparison to VRAXION
    vraxion_acc = 64.8
    log("COMPARISON TO VRAXION:")
    log(f"  VRAXION (AbsoluteHallway): {vraxion_acc:.1f}%")
    log(f"  Diamond Code (this run):  {max_acc*100:.1f}%")

    if max_acc * 100 > vraxion_acc:
        improvement = max_acc * 100 - vraxion_acc
        log(f"  Result: BETTER by +{improvement:.1f}% ✓")
    elif max_acc * 100 < vraxion_acc:
        gap = vraxion_acc - max_acc * 100
        log(f"  Result: WORSE by -{gap:.1f}%")
    else:
        log(f"  Result: TIED")

    log("")
    log(f"Log saved to: {args.log}")
    log("=" * 70)

    log_file.close()


if __name__ == "__main__":
    main()
