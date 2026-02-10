"""
Find minimum viable model size - shrink until it breaks!

Keep task constant (streaming assoc_clean), reduce model capacity.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def test_model_size(num_positions, embedding_dim, steps=2500, eval_freq=250):
    """Test a specific model configuration."""

    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"Testing: ring={num_positions}, embed={embedding_dim}, params={total_params:,}")
    print(f"{'='*70}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=32, keys=2, pairs=1, seed=9999
    )

    best_eval_acc = 0.0

    # Training loop
    for step in range(steps):
        # Fresh training data each step
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

        # Eval
        if step % eval_freq == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(x_eval)
                eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
            model.train()

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

            print(f"  Step {step:4d}: eval_acc={eval_acc*100:5.1f}%, best={best_eval_acc*100:5.1f}%")

    return best_eval_acc, total_params


def main():
    print("="*70)
    print("FIND MINIMUM VIABLE MODEL - Shrink Until It Breaks")
    print("="*70)
    print("\nTask: Streaming assoc_clean (2 keys, 1 pair, len=32)")
    print("Strategy: Keep task constant, reduce model capacity")
    print()

    results = []

    # Baseline (we know this works)
    print("\n" + "="*70)
    print("BASELINE (known to work)")
    print("="*70)
    acc, params = test_model_size(num_positions=64, embedding_dim=256, steps=2500)
    results.append(("64 pos × 256 dim", params, acc))

    # Test 1: Reduce embedding dimension (keep ring size)
    print("\n" + "="*70)
    print("TEST 1: Smaller Embeddings (ring=64)")
    print("="*70)

    for embed_dim in [128, 64, 32]:
        acc, params = test_model_size(num_positions=64, embedding_dim=embed_dim, steps=2500)
        results.append((f"64 pos × {embed_dim} dim", params, acc))

    # Test 2: Reduce ring size (keep embedding)
    print("\n" + "="*70)
    print("TEST 2: Smaller Ring (embed=256)")
    print("="*70)

    for ring_size in [32, 16]:
        acc, params = test_model_size(num_positions=ring_size, embedding_dim=256, steps=2500)
        results.append((f"{ring_size} pos × 256 dim", params, acc))

    # Test 3: Both small
    print("\n" + "="*70)
    print("TEST 3: Both Small")
    print("="*70)

    for ring_size, embed_dim in [(32, 128), (16, 64), (8, 32)]:
        acc, params = test_model_size(num_positions=ring_size, embedding_dim=embed_dim, steps=2500)
        results.append((f"{ring_size} pos × {embed_dim} dim", params, acc))

    # Test 4: TINY (extreme test)
    print("\n" + "="*70)
    print("TEST 4: TINY MODEL (extreme)")
    print("="*70)

    acc, params = test_model_size(num_positions=8, embedding_dim=16, steps=2500)
    results.append(("8 pos × 16 dim", params, acc))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Where Does Size Matter?")
    print("="*70)
    print()

    # Sort by parameter count
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    for config, params, acc in results_sorted:
        status = "PASS" if acc >= 0.95 else "OK" if acc >= 0.85 else "WEAK" if acc >= 0.70 else "FAIL"
        bar = "█" * int(acc * 50)
        pct_of_baseline = (params / results_sorted[0][1]) * 100
        print(f"{status:6s} {acc*100:5.1f}% {bar:50s} {config:20s} ({params:6,} params = {pct_of_baseline:5.1f}%)")

    print()
    print("="*70)
    print("Key Questions:")
    print("  - What's the MINIMUM params for 95%+ accuracy?")
    print("  - Does ring size or embedding dim matter more?")
    print("  - How much overhead does the baseline have?")
    print("="*70)


if __name__ == "__main__":
    main()
