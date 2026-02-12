"""
Evaluate trained Diamond Code model on fresh assoc_clean data.

Tests true generalization (not memorization).
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def eval_model(checkpoint_path, n_test_samples=1000, seed=9999):
    """
    Evaluate model on fresh test data.

    Args:
        checkpoint_path: Path to saved checkpoint
        n_test_samples: Number of test sequences
        seed: Random seed (different from training!)
    """
    print("=" * 70)
    print("DIAMOND CODE - Generalization Evaluation")
    print("=" * 70)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Create model with same architecture
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=64,
        embedding_dim=256,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_step = checkpoint.get('step', 'unknown')
    train_acc = checkpoint.get('acc', 'unknown')
    train_max_acc = checkpoint.get('max_acc', 'unknown')

    print(f"Checkpoint from step: {train_step}")
    print(f"Training accuracy at checkpoint: {train_acc}")
    print(f"Max training accuracy: {train_max_acc}")
    print()

    # Generate FRESH test data (different seed!)
    print(f"Generating {n_test_samples} FRESH test sequences (seed={seed})...")
    x_test, y_test, seq_len = generate_assoc_clean(
        n_samples=n_test_samples,
        seq_len=32,
        keys=2,
        pairs=1,
        seed=seed  # Different seed = new data!
    )
    print(f"Test data: {x_test.shape}, seq_len={seq_len}")
    print()

    # Evaluate in batches
    batch_size = 100
    n_batches = (n_test_samples + batch_size - 1) // batch_size

    all_correct = 0
    all_losses = []

    print("Running evaluation...")
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_test_samples)

            x_batch = x_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]

            # Forward pass
            logits, aux_loss, _ = model(x_batch)
            loss = torch.nn.functional.cross_entropy(logits, y_batch, reduction='none')

            # Accuracy
            predictions = logits.argmax(dim=1)
            correct = (predictions == y_batch).sum().item()
            all_correct += correct

            # Store losses
            all_losses.extend(loss.tolist())

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                running_acc = all_correct / end_idx
                print(f"  Batch {i+1}/{n_batches}: accuracy={running_acc*100:.1f}%")

    # Final results
    test_acc = all_correct / n_test_samples
    mean_loss = sum(all_losses) / len(all_losses)
    max_loss = max(all_losses)
    min_loss = min(all_losses)

    # Loss statistics
    loss_std = (sum((l - mean_loss)**2 for l in all_losses) / len(all_losses)) ** 0.5

    print()
    print("=" * 70)
    print("GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"Test accuracy: {test_acc*100:.2f}% ({all_correct}/{n_test_samples} correct)")
    print(f"Mean loss: {mean_loss:.4f}")
    print(f"Loss std: {loss_std:.4f}")
    print(f"Loss range: [{min_loss:.4f}, {max_loss:.4f}]")
    print()

    # Comparison
    print("COMPARISON:")
    print(f"  Training accuracy: {train_max_acc*100 if isinstance(train_max_acc, float) else train_max_acc}%")
    print(f"  Test accuracy:     {test_acc*100:.2f}%")

    if isinstance(train_max_acc, float):
        gap = (train_max_acc - test_acc) * 100
        if gap > 5:
            print(f"  Generalization gap: -{gap:.1f}% (overfitting!)")
        elif gap > 0:
            print(f"  Generalization gap: -{gap:.1f}% (slight overfitting)")
        else:
            print(f"  Generalization: EXCELLENT (test >= train)")

    print()
    print("VRAXION BASELINE:")
    print(f"  VRAXION (AbsoluteHallway): 64.8%")
    print(f"  Diamond Code (test):       {test_acc*100:.2f}%")

    if test_acc * 100 > 64.8:
        improvement = test_acc * 100 - 64.8
        print(f"  Result: BETTER by +{improvement:.1f}%")
    else:
        gap = 64.8 - test_acc * 100
        print(f"  Result: WORSE by -{gap:.1f}%")

    print("=" * 70)

    return test_acc, mean_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Diamond Code on assoc_clean")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--test-samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=9999, help='Test data seed (should differ from training)')

    args = parser.parse_args()

    eval_model(args.checkpoint, args.test_samples, args.seed)
