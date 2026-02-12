"""
Test Diamond Code on progressively harder tasks with dashboard.

Model: 64×64D (516 params) - minimal working configuration
Watch real-time on dashboard to see where it breaks!
"""

import torch
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def main():
    # TASK CONFIG - EDIT THESE TO TEST DIFFERENT DIFFICULTIES
    num_keys = 2      # Change: 2 (easy), 4, 8, 16 (hard)
    num_pairs = 2     # Change: 1 (easy), 2, 3, 5 (hard)
    seq_len = 32      # Change: 32 (easy), 64, 96, 128 (hard) - Back to baseline for debugging

    # MODEL CONFIG - Keep at minimal working size
    num_positions = 64
    embedding_dim = 64

    print("=" * 70)
    print("DIAMOND CODE - HARDER TASK TEST")
    print("=" * 70)
    print(f"Task: {num_keys} keys, {num_pairs} pairs, seq_len={seq_len}")
    print(f"Model: {num_positions}x{embedding_dim}D")
    print()

    # Create model with Möbius helix (2x effective memory)
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=True,  # Enable Möbius helix for doubled memory capacity
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=seq_len, keys=num_keys, pairs=num_pairs, seed=9999
    )

    # Log file
    log_path = Path(__file__).parent / "logs" / "diamond" / "probe_live.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear old log
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
    log(f"Harder Task: {num_keys} keys, {num_pairs} pairs, len={seq_len}")
    log(f"Model: {num_positions}x{embedding_dim}D = {total_params} params")
    log("=" * 70)

    best_eval_acc = 0.0
    step = 0

    try:
        # Run indefinitely
        while True:
            step_start = time.time()

            # Fresh training data (streaming)
            x_train, y_train, _ = generate_assoc_clean(
                n_samples=100, seq_len=seq_len, keys=num_keys, pairs=num_pairs,
                seed=42 + step + 1000000
            )

            # Train step
            optimizer.zero_grad()
            logits, aux_loss, _ = model(x_train)  # Discard routing_info during training
            loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss
            loss.backward()
            optimizer.step()

            # Training accuracy
            train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()

            step_time = time.time() - step_start

            # Eval every 10 steps (increased frequency to capture oscillations)
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    eval_logits, _, eval_routing = model(x_eval, return_debug=True)
                    eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()

                    # Jump gate stats
                    eval_jump_decisions = torch.stack(eval_routing['jump_decisions'])
                    eval_jump_rate = eval_jump_decisions.float().mean().item()

                    # Holonomy distribution
                    eval_holonomy_traj = torch.stack(eval_routing['holonomy_trajectory'])  # [T, B]
                    holonomy_pct = (eval_holonomy_traj == 1.0).float().mean().item()

                    # MÖBIUS DIAGNOSTICS
                    # Pointer position stats (detect synchronization)
                    eval_pointer_traj = torch.stack(eval_routing['pointer_trajectory'])  # [T, B]
                    ptr_mean = eval_pointer_traj.mean().item()
                    ptr_std = eval_pointer_traj.std().item()
                    ptr_min = eval_pointer_traj.min().item()
                    ptr_max = eval_pointer_traj.max().item()

                    # Wrap events (count position jumps from high to low)
                    ptr_diffs = eval_pointer_traj[1:] - eval_pointer_traj[:-1]
                    wrap_events = (ptr_diffs < -num_positions * 0.5).sum().item()  # Big negative jump = wrap

                    # Phase coverage (how much of 0-64 range is used)
                    ptr_range = ptr_max - ptr_min
                    coverage_pct = ptr_range / num_positions

                model.train()

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc

                # Dashboard-compatible log format with Möbius diagnostics
                log(f"step {step} | loss {loss.item():.6f} | "
                    f"acc={eval_acc:.4f} | jump_gate={eval_jump_rate:.2f} | "
                    f"holonomy_pct={holonomy_pct:.3f} | "
                    f"ptr_std={ptr_std:.2f} | wraps={wrap_events} | coverage={coverage_pct:.3f} | "
                    f"train_acc={train_acc:.4f} | best={best_eval_acc:.4f} | "
                    f"s_per_step={step_time:.3f}")

            step += 1

    except KeyboardInterrupt:
        log("")
        log("=" * 70)
        log(f"Training stopped at step {step}")
        log(f"Best eval accuracy: {best_eval_acc*100:.1f}%")
        log("=" * 70)


if __name__ == "__main__":
    main()
