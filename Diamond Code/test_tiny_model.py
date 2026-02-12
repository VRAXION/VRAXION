"""
Quick test of the tiniest viable model - 8 positions, 16D embeddings.

Find the absolute minimum: does Diamond Code work with ~70 parameters?
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def main():
    print("=" * 70)
    print("TINY MODEL TEST - Absolute Minimum Diamond Code")
    print("=" * 70)
    print()
    print("Testing: 8 positions x 16D embeddings")
    print("Task: 2 keys, 1 pair, seq_len=32 (streaming assoc_clean)")
    print("Steps: 500")
    print()

    # Create tiny model
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=8,
        embedding_dim=16,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=32, keys=2, pairs=1, seed=9999
    )

    best_eval_acc = 0.0

    # Training loop
    for step in range(500):
        # Fresh training data each step (streaming)
        x_train, y_train, _ = generate_assoc_clean(
            n_samples=100, seq_len=32, keys=2, pairs=1,
            seed=42 + step + 1000000
        )

        # Train step
        optimizer.zero_grad()
        logits, aux_loss, _ = model(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss
        loss.backward()
        optimizer.step()

        # Eval every 50 steps
        if step % 50 == 0 or step == 499:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(x_eval)
                eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
            model.train()

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

            print(f"Step {step:3d}: loss={loss.item():.6f} | acc={eval_acc*100:5.1f}% | best={best_eval_acc*100:5.1f}%")

    print()
    print("=" * 70)
    print(f"FINAL RESULT: {best_eval_acc*100:.1f}% accuracy")
    print()

    if best_eval_acc >= 0.95:
        print("PASS: Tiny model (70 params) can learn the task!")
    elif best_eval_acc >= 0.70:
        print("WEAK: Tiny model struggles but shows some learning")
    else:
        print("FAIL: Model too small to learn the task")
    print("=" * 70)


if __name__ == "__main__":
    main()
