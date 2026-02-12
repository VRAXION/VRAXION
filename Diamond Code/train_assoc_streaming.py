"""
Diamond Code - Streaming Associative Recall Training

Tests TRUE generalization by generating fresh data every step.
No memorization possible - must learn the algorithm!

Compared to static training:
- Static: Same 100 sequences for 2000 steps -> 100% (memorization)
- Streaming: NEW 100 sequences EVERY step -> ??? (algorithm learning)
"""

import torch
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from visualize_routing import detect_loops
from assoc_clean_data import generate_assoc_clean


def main():
    parser = argparse.ArgumentParser(description="Streaming assoc_clean training")
    parser.add_argument('--steps', type=int, default=5000, help='Number of training steps')
    parser.add_argument('--log', type=str, default='logs/diamond/assoc_streaming.log', help='Log file')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--assoc-keys', type=int, default=2, help='Number of keys (default 2, try 4 for harder)')
    parser.add_argument('--assoc-pairs', type=int, default=1, help='Pairs per sequence (default 1, try 2 for harder)')
    parser.add_argument('--num-positions', type=int, default=64, help='Memory ring positions')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval-freq', type=int, default=100, help='Eval every N steps')
    parser.add_argument('--eval-samples', type=int, default=1000, help='Eval set size')
    parser.add_argument('--routing-update-freq', type=int, default=50, help='Routing metrics every N steps')
    parser.add_argument('--checkpoint-freq', type=int, default=500, help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for model init only)')

    args = parser.parse_args()

    # Create directories
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Set random seed for model initialization only
    torch.manual_seed(args.seed)

    log("=" * 70)
    log("DIAMOND CODE - Streaming Associative Recall")
    log("=" * 70)
    log(f"Task: assoc_clean with FRESH DATA every step (no memorization!)")
    log(f"Keys: {args.assoc_keys}, Pairs: {args.assoc_pairs}, Seq len: {args.seq_len}")
    log(f"Model: {args.num_positions} positions, {args.embedding_dim}D embeddings")
    log(f"Training: {args.steps} steps, lr={args.lr}, batch_size={args.batch_size}")
    log(f"Eval: every {args.eval_freq} steps on {args.eval_samples} fresh samples")
    log("")
    log("CHALLENGE: Can the model learn the ALGORITHM without memorizing?")
    log("=" * 70)
    log("")

    # Create model
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=args.num_positions,
        embedding_dim=args.embedding_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,}")
    log("")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Track initial jump destinations
    initial_jump_dest = model.jump_destinations.detach().clone()

    # Generate FIXED eval set (never seen during training)
    log("Generating fixed evaluation set...")
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=args.eval_samples,
        seq_len=args.seq_len,
        keys=args.assoc_keys,
        pairs=args.assoc_pairs,
        seed=9999  # Fixed seed for eval
    )
    log(f"Eval set: {args.eval_samples} sequences (seed=9999, never seen in training)")
    log("")

    log("Starting streaming training...")
    log("")

    # Tracking
    best_eval_acc = 0.0
    running_train_acc = 0.0
    step_counter = 0

    # Training loop
    for step in range(args.steps):
        step_start = time.time()

        # Generate FRESH training data every step!
        # Use step number as seed so it's reproducible but always different
        x_train, y_train, _ = generate_assoc_clean(
            n_samples=args.batch_size,
            seq_len=args.seq_len,
            keys=args.assoc_keys,
            pairs=args.assoc_pairs,
            seed=args.seed + step + 1000000  # Different seed every step!
        )

        # Forward pass
        optimizer.zero_grad()
        logits, aux_loss, debug_info = model(x_train, return_debug=True)
        loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss

        # Backward pass
        loss.backward()

        # Compute metrics
        with torch.no_grad():
            # Training accuracy (on this batch)
            train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()
            running_train_acc = 0.9 * running_train_acc + 0.1 * train_acc  # EMA

            # Jump gate activation
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

        # Log training metrics
        log(f"step {step} | loss {loss.item():.6f} | "
            f"acc={train_acc:.4f} | jump_gate={jump_gate_rate:.2f} | "
            f"att_ent={att_ent:.2f} | ptr_pos={ptr_pos:.1f} | "
            f"s_per_step={step_time:.3f}")

        # Evaluation on fixed test set
        if step % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(x_eval)
                eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
                eval_loss = torch.nn.functional.cross_entropy(eval_logits, y_eval).item()

            model.train()

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

            log(f"eval step={step} | eval_loss={eval_loss:.6f} | "
                f"eval_acc={eval_acc:.4f} | best_eval_acc={best_eval_acc:.4f}")

        # Routing metrics
        if step % args.routing_update_freq == 0 and step > 0:
            with torch.no_grad():
                cycles, self_loops = detect_loops(model.jump_destinations)
                num_cycles = len(cycles)
                num_self_loops = len(self_loops)

                log(f"routing_update step={step} | "
                    f"cyc={num_cycles} | sl={num_self_loops} | "
                    f"grad_j={jump_grad_norm:.3f} | grad_o={other_grad_norm:.3f}")

        # Checkpoints
        if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0 and step > 0:
            checkpoint_file = checkpoint_path / f"assoc_streaming_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'train_acc': train_acc,
                'eval_acc': eval_acc if step % args.eval_freq == 0 else None,
                'best_eval_acc': best_eval_acc,
            }, checkpoint_file)
            log(f"checkpoint saved: {checkpoint_file}")

    # Final evaluation
    log("")
    log("=" * 70)
    log("Training Complete")
    log("=" * 70)

    model.eval()
    with torch.no_grad():
        # Final eval
        eval_logits, _, _ = model(x_eval)
        final_eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
        final_eval_loss = torch.nn.functional.cross_entropy(eval_logits, y_eval).item()

        # Routing stats
        cycles, self_loops = detect_loops(model.jump_destinations)
        param_delta = (model.jump_destinations - initial_jump_dest).norm().item()

    log(f"Final eval accuracy: {final_eval_acc*100:.1f}%")
    log(f"Best eval accuracy: {best_eval_acc*100:.1f}%")
    log(f"Final eval loss: {final_eval_loss:.4f}")
    log("")
    log(f"Refinement stations (self-loops): {len(self_loops)}")
    log(f"Detected cycles: {len(cycles)}")
    log(f"Jump destination parameter delta: {param_delta:.4f}")
    log("")

    # Comparison
    log("COMPARISON:")
    log(f"  Static training (memorization): 100.0% (train), 99.7% (test)")
    log(f"  Streaming training (algorithm): {best_eval_acc*100:.1f}% (test)")
    log(f"  VRAXION baseline: 64.8%")
    log("")

    if best_eval_acc * 100 > 90:
        log("Result: SUCCESS - Model learned the algorithm!")
    elif best_eval_acc * 100 > 70:
        log("Result: PARTIAL - Model partially learned the pattern")
    else:
        log("Result: FAILED - Model could not generalize without memorization")

    log("")
    log(f"Log saved to: {args.log}")
    log("=" * 70)

    log_file.close()


if __name__ == "__main__":
    main()
