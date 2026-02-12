"""
Find the limits of Diamond Code - where does it plateau?

Tests progressively harder variants to find breaking points.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def quick_test(
    num_keys=2,
    num_pairs=1,
    seq_len=32,
    num_positions=64,
    embedding_dim=64,  # Test minimal working model (516 params)
    steps=1000,
    eval_freq=100,
    seed=42
):
    """
    Quick test on a configuration.

    Returns final eval accuracy.
    """
    # Create model
    torch.manual_seed(seed)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,  # Always binary for assoc_clean
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Fixed eval set
    x_eval, y_eval, actual_len = generate_assoc_clean(
        n_samples=500,
        seq_len=seq_len,
        keys=num_keys,
        pairs=num_pairs,
        seed=9999
    )

    print(f"\nTesting: keys={num_keys}, pairs={num_pairs}, seq_len={seq_len} (actual={actual_len})")
    print(f"Model: {num_positions} positions, {embedding_dim}D, {sum(p.numel() for p in model.parameters()):,} params")

    best_eval_acc = 0.0

    # Training loop
    for step in range(steps):
        # Generate fresh data
        x_train, y_train, _ = generate_assoc_clean(
            n_samples=100,
            seq_len=seq_len,
            keys=num_keys,
            pairs=num_pairs,
            seed=seed + step + 1000000
        )

        # Train step
        optimizer.zero_grad()
        logits, aux_loss, _ = model(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train) + aux_loss
        loss.backward()
        optimizer.step()

        # Eval
        if step % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                eval_logits, _, _ = model(x_eval)
                eval_acc = (eval_logits.argmax(dim=1) == y_eval).float().mean().item()
            model.train()

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

            print(f"  Step {step:4d}: eval_acc={eval_acc*100:5.1f}%, best={best_eval_acc*100:5.1f}%")

    print(f"RESULT: {best_eval_acc*100:.1f}%")
    return best_eval_acc


def main():
    print("=" * 70)
    print("DIAMOND CODE - LIMIT FINDING")
    print("=" * 70)
    print()
    print("Testing progressively harder configurations to find breaking points.")
    print()

    results = []

    # Baseline (we know this works)
    print("\n" + "=" * 70)
    print("BASELINE (known to work)")
    print("=" * 70)
    acc = quick_test(num_keys=2, num_pairs=1, seq_len=32, steps=1000)
    results.append(("Baseline (2 keys, 1 pair, len=32)", acc))

    # Test 1: More keys
    print("\n" + "=" * 70)
    print("TEST 1: More Keys (Harder Classification)")
    print("=" * 70)

    for num_keys in [4, 8]:
        acc = quick_test(num_keys=num_keys, num_pairs=1, seq_len=32, steps=1000)
        results.append((f"{num_keys} keys, 1 pair, len=32", acc))

    # Test 2: More pairs
    print("\n" + "=" * 70)
    print("TEST 2: More Pairs (Multi-Association)")
    print("=" * 70)

    for num_pairs in [2, 3]:
        acc = quick_test(num_keys=2, num_pairs=num_pairs, seq_len=48, steps=1000)
        results.append((f"2 keys, {num_pairs} pairs, len=48", acc))

    # Test 3: Longer sequences
    print("\n" + "=" * 70)
    print("TEST 3: Longer Sequences (Long-Range Dependencies)")
    print("=" * 70)

    for seq_len in [64, 96]:
        acc = quick_test(num_keys=2, num_pairs=1, seq_len=seq_len, steps=1000)
        results.append((f"2 keys, 1 pair, len={seq_len}", acc))

    # Test 4: WEIRD - Sequence longer than ring
    print("\n" + "=" * 70)
    print("TEST 4: WEIRD - Sequence Longer Than Ring")
    print("=" * 70)

    acc = quick_test(num_keys=2, num_pairs=1, seq_len=128, num_positions=64, steps=1000)
    results.append((f"2 keys, 1 pair, len=128 (ring=64)", acc))

    # Test 5: WEIRD - Tiny model
    print("\n" + "=" * 70)
    print("TEST 5: WEIRD - Tiny Model")
    print("=" * 70)

    acc = quick_test(num_keys=2, num_pairs=1, seq_len=32, num_positions=16, embedding_dim=64, steps=1000)
    results.append((f"2 keys, 1 pair, len=32 (16 pos, 64D)", acc))

    # Test 6: Combined hard
    print("\n" + "=" * 70)
    print("TEST 6: Combined Hard Mode")
    print("=" * 70)

    acc = quick_test(num_keys=4, num_pairs=2, seq_len=64, steps=1500)
    results.append((f"4 keys, 2 pairs, len=64", acc))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - WHERE DOES IT BREAK?")
    print("=" * 70)
    print()

    for config, acc in results:
        status = "PASS" if acc > 0.90 else "STRUGGLE" if acc > 0.70 else "FAIL"
        bar = "█" * int(acc * 40)
        print(f"{status:8s} {acc*100:5.1f}% {bar:40s} {config}")

    print()
    print("=" * 70)
    print("Thresholds: PASS >= 90%, STRUGGLE 70-90%, FAIL < 70%")
    print("=" * 70)


if __name__ == "__main__":
    main()
