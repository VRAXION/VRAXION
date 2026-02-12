"""
Analyze which test cases the model failed on.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


# Load checkpoint
checkpoint = torch.load("checkpoints/assoc_clean_step_1900.pt")

model = RingMemoryModel(
    input_size=1,
    num_outputs=2,
    num_memory_positions=64,
    embedding_dim=256,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate same test data
x_test, y_test, seq_len = generate_assoc_clean(
    n_samples=1000,
    seq_len=32,
    keys=2,
    pairs=1,
    seed=9999
)

# Find failures
print("=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)
print()

with torch.no_grad():
    logits, aux_loss, _ = model(x_test)
    predictions = logits.argmax(dim=1)

    # Find mistakes
    mistakes = (predictions != y_test).nonzero(as_tuple=True)[0]

    print(f"Total mistakes: {len(mistakes)}/1000")
    print()

    if len(mistakes) > 0:
        print("Failed sequences:")
        for i, idx in enumerate(mistakes[:10]):  # Show first 10
            idx = idx.item()
            seq = x_test[idx, :, 0]
            true_label = y_test[idx].item()
            pred_label = predictions[idx].item()
            confidence = torch.softmax(logits[idx], dim=0)

            print(f"\nMistake {i+1}:")
            print(f"  Sequence index: {idx}")
            print(f"  True label: {true_label}")
            print(f"  Predicted: {pred_label}")
            print(f"  Confidence: {confidence[pred_label].item():.3f}")

            # Show sequence
            nonzero = (seq != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) > 0:
                print(f"  Sequence (non-zero positions):")
                for pos in nonzero:
                    token = seq[pos].item()
                    if pos == len(seq) - 1:
                        print(f"    [{pos:2d}] {token:5.1f}  <- QUERY")
                    elif token > 0:
                        print(f"    [{pos:2d}] {token:5.1f}  <- key")
                    else:
                        val_class = 0 if token == -1.0 else 1
                        print(f"    [{pos:2d}] {token:5.1f}  <- value (class {val_class})")
    else:
        print("No mistakes! Perfect accuracy!")

print()
print("=" * 70)
