"""
Hard Task Test - Standard Ring

Task: 4 keys, 4 pairs, seq_len=64
This is 4× harder than baseline (2 keys, 2 pairs, seq_len=32)

Control test: If Möbius doesn't outperform Standard on this
memory-intensive task, then holonomy doesn't double memory.
"""

import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def main():
    # HARD TASK CONFIG (same as Möbius test)
    num_keys = 4      # 2× baseline
    num_pairs = 4     # 2× baseline
    seq_len = 64      # 2× baseline

    # MODEL CONFIG (same capacity)
    num_positions = 64
    embedding_dim = 64

    print("=" * 70)
    print("HARD TASK TEST - STANDARD RING")
    print("=" * 70)
    print(f"Task: {num_keys} keys, {num_pairs} pairs, seq_len={seq_len}")
    print(f"Model: {num_positions}x{embedding_dim}D")
    print(f"Optimizer: AdamW(lr=0.001, weight_decay=0.01)")
    print(f"Topology: STANDARD RING (control)")
    print()
    print("Hypothesis: Standard should struggle vs Möbius on hard tasks")
    print("Reason: Limited to 64 memory positions (no holonomy)")
    print()

    # Create model WITHOUT Möbius helix
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=False,  # STANDARD RING
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    # AdamW optimizer (same as Möbius test)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set (SAME as Möbius test)
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=seq_len, keys=num_keys, pairs=num_pairs, seed=9999
    )

    # Log file
    log_path = Path(__file__).parent / "logs" / "diamond" / "hard_standard.log"
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
    log(f"Hard Task (Standard): {num_keys} keys, {num_pairs} pairs, len={seq_len}")
    log(f"Model: {num_positions}x{embedding_dim}D = {total_params} params")
    log(f"Optimizer: AdamW(lr=0.001, weight_decay=0.01)")
    log(f"Topology: STANDARD RING")
    log("=" * 70)

    best_eval_acc = 0.0
    step = 0

    try:
        # Run for 3000 steps (same as Möbius test)
        while step < 3000:
            step_start = time.time()

            # Fresh training data (streaming)
            x_train, y_train, _ = generate_assoc_clean(
                n_samples=100, seq_len=seq_len, keys=num_keys, pairs=num_pairs,
                seed=42 + step + 1000000
            )

            # Train step
            optimizer.zero_grad()
            logits, aux_loss, _ = model(x_train)
            loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss
            loss.backward()
            optimizer.step()

            # Training accuracy
            train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()

            step_time = time.time() - step_start

            # Eval every 10 steps
            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    eval_logits, _, eval_routing = model(x_eval, return_debug=True)
                    eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()

                    # Jump gate stats (should be 0 for standard ring)
                    eval_jump_decisions = torch.stack(eval_routing['jump_decisions'])
                    eval_jump_rate = eval_jump_decisions.float().mean().item()

                    # Holonomy (should always be 1.0 for standard ring)
                    eval_holonomy_traj = torch.stack(eval_routing['holonomy_trajectory'])
                    holonomy_pct = (eval_holonomy_traj == 1.0).float().mean().item()

                    # Pointer stats
                    eval_pointer_traj = torch.stack(eval_routing['pointer_trajectory'])
                    ptr_std = eval_pointer_traj.std().item()

                    # Wrap events
                    ptr_diffs = eval_pointer_traj[1:] - eval_pointer_traj[:-1]
                    wrap_events = (ptr_diffs < -num_positions * 0.5).sum().item()

                    # Coverage
                    ptr_range = eval_pointer_traj.max().item() - eval_pointer_traj.min().item()
                    coverage_pct = ptr_range / num_positions

                model.train()

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc

                # Dashboard-compatible log format
                log(f"step {step} | loss {loss.item():.6f} | "
                    f"acc={eval_acc:.4f} | jump_gate={eval_jump_rate:.2f} | "
                    f"holonomy_pct={holonomy_pct:.3f} | "
                    f"ptr_std={ptr_std:.2f} | wraps={wrap_events} | coverage={coverage_pct:.3f} | "
                    f"train_acc={train_acc:.4f} | best={best_eval_acc:.4f} | "
                    f"s_per_step={step_time:.3f}")

            step += 1

    except KeyboardInterrupt:
        pass

    log("")
    log("=" * 70)
    log(f"Training completed at step {step}")
    log(f"Best eval accuracy: {best_eval_acc*100:.1f}%")
    log("=" * 70)


if __name__ == "__main__":
    main()
