"""
Standalone assoc_clean data generator for Diamond Code.

Extracted from VRAXION's instnct_data.py for direct comparison testing.

Task: Associative recall
- Keys: positive tokens (2.0, 3.0, ...)
- Values: -1.0 (class 0) or -2.0 (class 1)
- Query: last token is a key, predict associated value
"""

import torch
import random


def generate_assoc_clean(
    n_samples=100,
    seq_len=16,
    keys=2,
    pairs=1,
    seed=42,
):
    """
    Generate assoc_clean dataset.

    Args:
        n_samples: Number of sequences
        seq_len: Target sequence length (auto-adjusted if too short)
        keys: Number of unique keys (default 2)
        pairs: Number of key-value pairs per sequence (default 1)
        seed: Random seed

    Returns:
        (x_data, y_labels, actual_seq_len)
        x_data: [n_samples, seq_len, 1] float tensor
        y_labels: [n_samples] long tensor (binary: 0 or 1)
        actual_seq_len: Actual sequence length used (may be > seq_len)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    min_len = pairs * 2 + 1  # Need space for pairs + query
    max_bumps = 5

    # Ensure sequence is long enough
    if seq_len < min_len:
        print(f"[assoc_clean] bumping seq_len {seq_len} -> {min_len} (min required)")
        seq_len = min_len

    # Try to place non-overlapping pairs
    for bump_attempt in range(max_bumps + 1):
        x = torch.zeros((n_samples, seq_len, 1), dtype=torch.float32)
        y = torch.zeros((n_samples,), dtype=torch.long)
        max_start = seq_len - 3  # Reserve last position for query

        success = True
        for idx in range(n_samples):
            used = set()
            pair_specs = []

            # Shuffle possible starting positions
            starts = list(range(0, max_start + 1))
            random.shuffle(starts)

            # Place pairs without overlap
            for cand in starts:
                if cand in used or (cand + 1) in used:
                    continue

                used.add(cand)
                used.add(cand + 1)

                # Generate key-value pair
                key_id = random.randint(0, keys - 1)
                val = random.randint(0, 1)
                key_token = float(2 + key_id)
                val_token = -1.0 if val == 0 else -2.0

                x[idx, cand, 0] = key_token
                x[idx, cand + 1, 0] = val_token
                pair_specs.append((key_id, val, key_token))

                if len(pair_specs) >= pairs:
                    break

            # Check if we placed enough pairs
            if len(pair_specs) < pairs:
                success = False
                break

            # Place query (one of the keys) at the end
            _, q_val, q_token = random.choice(pair_specs)
            x[idx, -1, 0] = q_token
            y[idx] = q_val

        if success:
            return x, y, seq_len

        # Failed to place pairs, increase sequence length
        new_len = seq_len + max(2, pairs) * 2
        print(f"[assoc_clean] bumping seq_len {seq_len} -> {new_len} (placement failed)")
        seq_len = new_len

    raise RuntimeError(f"assoc_clean: failed to place {pairs} non-overlapping pairs after {max_bumps} bumps")


def test_assoc_clean():
    """Test the assoc_clean generator."""
    print("=" * 70)
    print("Testing assoc_clean data generator")
    print("=" * 70)

    x, y, seq_len = generate_assoc_clean(n_samples=5, seq_len=16, keys=2, pairs=1, seed=42)

    print(f"Generated {x.shape[0]} sequences, length {seq_len}")
    print()

    # Show first sequence
    print("Example sequence:")
    seq = x[0, :, 0]
    label = y[0].item()

    for i, token in enumerate(seq):
        if token == 0.0:
            print(f"  [{i:2d}] {token:5.1f}  (padding)")
        elif token > 0:
            print(f"  [{i:2d}] {token:5.1f}  <- KEY")
        elif token < 0:
            val_class = 0 if token == -1.0 else 1
            print(f"  [{i:2d}] {token:5.1f}  <- VALUE (class {val_class})")

    print()
    print(f"Query (last token): {seq[-1].item():.1f}")
    print(f"Label: {label}")
    print()

    # Verify answer is correct
    query_key = seq[-1].item()
    for i in range(len(seq) - 1):
        if seq[i].item() == query_key and i + 1 < len(seq):
            expected_val = 0 if seq[i + 1].item() == -1.0 else 1
            print(f"Found pair at position {i}: key={query_key:.1f}, value={seq[i+1].item():.1f} -> class {expected_val}")
            if expected_val == label:
                print("✓ Label matches!")
            else:
                print("✗ Label mismatch!")
            break

    print("=" * 70)


if __name__ == "__main__":
    test_assoc_clean()
