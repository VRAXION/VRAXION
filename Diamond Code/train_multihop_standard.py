"""
Multi-hop Reasoning Test - Standard Topology (1-hop baseline)

Task: 1-hop simple recall (NO composition)
- 5 chains × 1 hop = 5 facts to remember
- Simple key->value lookup (A->B, no chaining)
- Tests pure recall capacity without compositional stress

Purpose: Isolate classification capacity bottleneck
- If 1-hop works (~60%+): composition was the bottleneck
- If 1-hop fails (~15%): 15-way classification is too hard
- Expected: ~60-75% for both topologies (easy task)
"""

import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from multihop_task import generate_multihop


def main():
    # 1-HOP BASELINE CONFIG (v3 - isolate classification bottleneck)
    num_chains = 5       # Number of independent fact chains
    chain_length = 1     # 1-hop = simple recall, NO composition
    vocab_size = 15      # Number of unique entity IDs

    # Calculated sequence length: chains × hops × 3 (key, val, LINK) + 3 (QUERY, key, ?)
    seq_len = num_chains * chain_length * 3 + 3

    # MODEL CONFIG (same capacity as Möbius)
    num_positions = 64
    embedding_dim = 64

    print("=" * 70)
    print("1-HOP BASELINE TEST - STANDARD TOPOLOGY")
    print("=" * 70)
    print(f"Task: {num_chains} chains × {chain_length} hop = {num_chains * chain_length} facts")
    print(f"Vocab size: {vocab_size} entities (15-way classification)")
    print(f"Sequence length: {seq_len} tokens")
    print(f"Model: {num_positions}×{embedding_dim}D")
    print(f"Optimizer: AdamW(lr=0.001, weight_decay=0.01)")
    print(f"Topology: STANDARD (baseline)")
    print()
    print("Purpose: Test if model can learn 15-way classification")
    print("Expected: ~60-75% (easy task, no compositional stress)")
    print("If fails: 15-way classification is too hard for 7K params")
    print()

    # Create model with Standard ring (no Möbius)
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=vocab_size,  # Predict entity IDs
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=False,  # STANDARD TOPOLOGY
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Fixed eval set (500 samples, seed for reproducibility - SAME AS MÖBIUS)
    x_eval, y_eval, _ = generate_multihop(
        n_samples=500,
        num_chains=num_chains,
        chain_length=chain_length,
        vocab_size=vocab_size,
        seed=9999,  # Same seed as Möbius for fair comparison
    )

    # Log file
    log_path = Path(__file__).parent / "logs" / "diamond" / "multihop_standard.log"
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
    log(f"1-hop baseline (Standard): {num_chains} chains, {vocab_size} entities, len={seq_len}")
    log(f"Model: {num_positions}×{embedding_dim}D = {total_params} params")
    log(f"Optimizer: AdamW(lr=0.001, weight_decay=0.01)")
    log(f"Topology: STANDARD | Purpose: isolate classification bottleneck")
    log("=" * 70)

    best_eval_acc = 0.0
    step = 0

    try:
        # Run for 3000 steps (same as Möbius)
        while step < 3000:
            step_start = time.time()

            # Fresh training data (streaming, same seed scheme as Möbius)
            x_train, y_train, _ = generate_multihop(
                n_samples=100,
                num_chains=num_chains,
                chain_length=chain_length,
                vocab_size=vocab_size,
                seed=42 + step + 1000000,
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

                    # Jump gate stats
                    eval_jump_decisions = torch.stack(eval_routing['jump_decisions'])
                    eval_jump_rate = eval_jump_decisions.float().mean().item()

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

                # Dashboard-compatible log format (no holonomy_pct for Standard)
                log(f"step {step} | loss {loss.item():.6f} | "
                    f"acc={eval_acc:.4f} | jump_gate={eval_jump_rate:.2f} | "
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

    # Summary for comparison
    print()
    print("=" * 70)
    print("FINAL RESULTS - STANDARD (1-HOP BASELINE)")
    print("=" * 70)
    print(f"Task: {chain_length}-hop simple recall ({num_chains} chains)")
    print(f"Best eval accuracy: {best_eval_acc*100:.1f}%")
    print(f"Random baseline: {100/vocab_size:.1f}%")
    print()
    print("Interpretation:")
    if best_eval_acc >= 0.60:
        print("  SUCCESS: Model can learn 15-way classification")
        print("  Conclusion: 2-hop failure was due to COMPOSITION, not classification")
        print("  Next step: Try 10-way or 2-way with 2-hop composition")
    elif best_eval_acc >= 0.30:
        print("  PARTIAL: Model struggles with 15-way but learns something")
        print("  Conclusion: Classification capacity is tight, composition too hard")
        print("  Next step: Try 10-way 1-hop or 2-way 2-hop")
    else:
        print("  FAILURE: Model cannot learn 15-way classification")
        print("  Conclusion: Classification capacity is the primary bottleneck")
        print("  Next step: Reduce to 10-way or binary classification")
    print("=" * 70)


if __name__ == "__main__":
    main()
