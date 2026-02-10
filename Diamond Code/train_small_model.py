"""
Train small Diamond Code model with dashboard logging.

Model: 16 positions × 32D embeddings (~500 params)
Task: Streaming assoc_clean (2 keys, 1 pair, seq_len=32)
Steps: Runs indefinitely - stop manually when satisfied
"""

import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def main():
    # Model config
    num_positions = 64
    embedding_dim = 32

    print("=" * 70)
    print("SMALL MODEL TRAINING - Dashboard Mode")
    print("=" * 70)
    print(f"Model: {num_positions} positions × {embedding_dim}D embeddings")
    print("Task: Streaming assoc_clean (2 keys, 1 pair, seq_len=32)")
    print("Steps: Unlimited (stop manually)")
    print()

    # Create model
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=32, keys=2, pairs=1, seed=9999
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
    log(f"Small Model Training: {num_positions}×{embedding_dim}D = {total_params} params")
    log("=" * 70)

    best_eval_acc = 0.0
    step = 0

    try:
        # Run indefinitely
        while True:
            step_start = time.time()

            # Fresh training data (streaming)
            x_train, y_train, _ = generate_assoc_clean(
                n_samples=100, seq_len=32, keys=2, pairs=1,
                seed=42 + step + 1000000
            )

            # Train step
            optimizer.zero_grad()
            logits, aux_loss, routing_info = model(x_train, return_debug=True)
            loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss
            loss.backward()
            optimizer.step()

            # Training accuracy
            train_acc = (logits.argmax(dim=1) == y_train).float().mean().item()

            # Extract jump gate activation rate
            # jump_decisions is a list of [batch] tensors (one per timestep)
            jump_decisions = torch.stack(routing_info['jump_decisions'])  # [seq_len, batch]
            jump_gate_rate = jump_decisions.float().mean().item()

            step_time = time.time() - step_start

            # Eval every 50 steps
            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    eval_logits, _, eval_routing = model(x_eval, return_debug=True)
                    eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
                    eval_jump_decisions = torch.stack(eval_routing['jump_decisions'])
                    eval_jump_rate = eval_jump_decisions.float().mean().item()
                model.train()

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc

                # Dashboard-compatible log format
                log(f"step {step} | loss {loss.item():.6f} | "
                    f"acc={eval_acc:.4f} | jump_gate={eval_jump_rate:.2f} | "
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
