"""Diagnostic task ladder for model testing.

Each function generates a single task level with tunable difficulty.
All tasks return (x, y, num_classes) where:
- x: input sequences [n_samples, seq_len, 1] float32
- y: target labels [n_samples] long
- num_classes: number of output classes

Fresh random data generated on each call - no memorization possible.
"""

import torch
import random


def task_copy(n_samples=5000, seq_len=16, vocab_size=10, seed=None):
    """Level 1: Copy last token.

    Task: Predict the final token in the sequence.
    Difficulty: Trivial - just remember last seen token.
    Expected: 98-100% accuracy in <20 steps.

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens (0 to vocab_size-1)
        seed: Random seed for reproducibility

    Returns:
        (x, y, num_classes) tuple
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()
    y = x[:, -1, 0].long()
    return x, y, vocab_size


def task_nback(n_samples=5000, seq_len=16, vocab_size=10, n_back=1, seed=None):
    """Level 2: Predict token from N positions back.

    Task: Predict the token N positions before the end.
    Difficulty: Easy - requires N-step memory.
    Expected: 95-98% accuracy for n_back=1 in <50 steps.

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens
        n_back: How many positions back to predict (1=second-to-last)
        seed: Random seed

    Returns:
        (x, y, num_classes) tuple
    """
    if seed is not None:
        torch.manual_seed(seed)

    if seq_len <= n_back:
        raise ValueError(f"seq_len ({seq_len}) must be > n_back ({n_back})")

    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()
    y = x[:, -(n_back+1), 0].long()
    return x, y, vocab_size


def task_parity(n_samples=5000, seq_len=16, seed=None):
    """Level 3: Count 1s, output even/odd.

    Task: Count number of 1s in sequence, predict 0 (even) or 1 (odd).
    Difficulty: Easy-medium - requires counting.
    Expected: 85-95% accuracy in <100 steps.

    Tuning: Increase seq_len (16->32->64->128) for harder counting.

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        seed: Random seed

    Returns:
        (x, y, num_classes) where num_classes=2
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Binary sequences (0.0, 1.0)
    x = torch.randint(0, 2, (n_samples, seq_len, 1)).float()

    # Count 1s per sequence
    counts = x.sum(dim=1).squeeze()

    # Parity: 0 if even, 1 if odd
    y = (counts % 2).long()

    return x, y, 2


def task_count_range(n_samples=5000, seq_len=32, vocab_size=10, target_token=5, bins=None, seed=None):
    """Level 4: Count target token occurrences, classify into bins.

    Task: Count how many times target_token appears, output bin index.
    Difficulty: Medium - requires precise counting and binning.
    Expected: 75-85% accuracy in <150 steps.

    Tuning:
    - Narrow bins (harder): bins=[0, 1, 2, 3, 4] (5 classes)
    - Wider bins (easier): bins=[2, 5] (3 classes: <=2, 3-5, >5)
    - Longer sequences (harder): seq_len=64, 128

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens
        target_token: Token to count
        bins: Bin thresholds (e.g., [2, 5] -> classes: <=2, 3-5, >5)
        seed: Random seed

    Returns:
        (x, y, num_classes) where num_classes=len(bins)+1
    """
    if bins is None:
        bins = [2, 5]  # Default: 3 bins (0-2, 3-5, 6+)

    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()

    # Count target token per sequence
    counts = (x.squeeze() == target_token).sum(dim=1)

    # Bin the counts
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        count = counts[i].item()
        bin_idx = len(bins)  # Default: highest bin
        for j, threshold in enumerate(bins):
            if count <= threshold:
                bin_idx = j
                break
        y[i] = bin_idx

    return x, y, len(bins) + 1


def task_first_last_match(n_samples=5000, seq_len=32, vocab_size=10, seed=None):
    """Level 5: Check if first token == last token.

    Task: Output 1 if first token matches last token, 0 otherwise.
    Difficulty: Medium - requires long-range memory (beginning vs end).
    Expected: 70-80% accuracy in <200 steps.

    Tuning: Increase seq_len (32->64->128) for longer memory span.

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens
        seed: Random seed

    Returns:
        (x, y, num_classes) where num_classes=2
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random sequences
    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()

    # Force 50% to match (for balanced dataset)
    for i in range(n_samples):
        if i % 2 == 0:
            # Make first == last
            x[i, -1, 0] = x[i, 0, 0]

    # Check if first == last
    y = (x[:, 0, 0] == x[:, -1, 0]).long()

    return x, y, 2


def task_majority_vote(n_samples=5000, seq_len=32, vocab_size=5, seed=None):
    """Level 6: Find most common token in sequence.

    Task: Predict the token that appears most frequently.
    Difficulty: Medium-hard - requires frequency tracking.
    Expected: 65-75% accuracy in <250 steps.

    Tuning:
    - Increase vocab_size (5->10->20) - more ties, harder
    - Increase seq_len (32->64) - longer sequences

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens
        seed: Random seed

    Returns:
        (x, y, num_classes) where num_classes=vocab_size
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()

    # Find most frequent token per sequence
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        seq = x[i, :, 0]
        # Count frequencies
        counts = torch.zeros(vocab_size)
        for token in range(vocab_size):
            counts[token] = (seq == token).sum()
        # Argmax (ties broken by first occurrence)
        y[i] = counts.argmax()

    return x, y, vocab_size


def task_nested_parity(n_samples=5000, seq_len=32, seed=None):
    """Level 8: Parity of parity (compositional reasoning).

    Task: Compute parity(first_half) XOR parity(second_half).
    Difficulty: Very hard - requires compositional reasoning.
    Expected: 50-60% accuracy (might fail completely).

    Tuning: Increase seq_len for even harder.

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length (should be even for clean split)
        seed: Random seed

    Returns:
        (x, y, num_classes) where num_classes=2
    """
    if seed is not None:
        torch.manual_seed(seed)

    if seq_len % 2 != 0:
        raise ValueError(f"seq_len ({seq_len}) must be even for nested_parity")

    # Binary sequences
    x = torch.randint(0, 2, (n_samples, seq_len, 1)).float()

    half = seq_len // 2

    # Compute parity for each half
    first_half = x[:, :half, 0].sum(dim=1) % 2
    second_half = x[:, half:, 0].sum(dim=1) % 2

    # XOR the two parities
    y = (first_half.long() ^ second_half.long())

    return x, y, 2


def task_reversal(n_samples=5000, seq_len=16, vocab_size=10, seed=None):
    """Level 10: Sequence reversal (predict first token).

    Task: Predict the first token in the sequence (perfect memory + reversal).
    Difficulty: Extremely hard - requires full sequence memory.
    Expected: <30% accuracy (likely fails).

    Args:
        n_samples: Number of sequences
        seq_len: Sequence length
        vocab_size: Number of distinct tokens
        seed: Random seed

    Returns:
        (x, y, num_classes) tuple
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randint(0, vocab_size, (n_samples, seq_len, 1)).float()
    y = x[:, 0, 0].long()
    return x, y, vocab_size


# Helper function to create train/eval splits
def create_splits(x, y, train_ratio=0.8, seed=None):
    """Split data into train/eval sets.

    Args:
        x: Input sequences [n_samples, seq_len, 1]
        y: Target labels [n_samples]
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for shuffle

    Returns:
        (x_train, y_train, x_eval, y_eval) tuple
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_samples = x.shape[0]
    indices = torch.randperm(n_samples)

    split_idx = int(n_samples * train_ratio)

    train_idx = indices[:split_idx]
    eval_idx = indices[split_idx:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_eval = x[eval_idx]
    y_eval = y[eval_idx]

    return x_train, y_train, x_eval, y_eval


# Expose all task generators
__all__ = [
    'task_copy',
    'task_nback',
    'task_parity',
    'task_count_range',
    'task_first_last_match',
    'task_majority_vote',
    'task_nested_parity',
    'task_reversal',
    'create_splits',
]


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 70)
    print("DIAGNOSTIC TASKS - SANITY CHECK")
    print("=" * 70)
    print()

    tasks = [
        ("L1_COPY", task_copy, {"n_samples": 100, "seq_len": 16, "vocab_size": 10}),
        ("L2_1BACK", task_nback, {"n_samples": 100, "seq_len": 16, "vocab_size": 10, "n_back": 1}),
        ("L3_PARITY", task_parity, {"n_samples": 100, "seq_len": 16}),
        ("L4_COUNT", task_count_range, {"n_samples": 100, "seq_len": 32, "vocab_size": 10}),
        ("L5_FIRSTLAST", task_first_last_match, {"n_samples": 100, "seq_len": 32, "vocab_size": 10}),
        ("L6_MAJORITY", task_majority_vote, {"n_samples": 100, "seq_len": 32, "vocab_size": 5}),
        ("L8_NESTED", task_nested_parity, {"n_samples": 100, "seq_len": 32}),
        ("L10_REVERSAL", task_reversal, {"n_samples": 100, "seq_len": 16, "vocab_size": 10}),
    ]

    for name, task_fn, kwargs in tasks:
        x, y, num_classes = task_fn(**kwargs)
        print(f"{name:<15} x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}  num_classes={num_classes}")

        # Check label distribution
        unique, counts = torch.unique(y, return_counts=True)
        print(f"{'':15} Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
        print()

    print("All tasks generated successfully!")
