"""
Debug: Check if holonomy is actually flipping during training.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


torch.manual_seed(42)

# Same config as train_harder_tasks.py
model = RingMemoryModel(
    input_size=1,
    num_outputs=2,
    num_memory_positions=64,
    embedding_dim=64,
    mobius=True,
)

# Generate sample data (2 pairs task)
x_train, y_train, _ = generate_assoc_clean(
    n_samples=10, seq_len=32, keys=2, pairs=2, seed=12345
)

print("Running forward pass with TRUE Möbius...")
logits, aux_loss, debug_info = model(x_train, return_debug=True)

# Check holonomy trajectory
holonomy_traj = torch.stack(debug_info['holonomy_trajectory'])  # [T, B]
pointer_traj = torch.stack(debug_info['pointer_trajectory'])  # [T, B]

print(f"\nHolonomy trajectory shape: {holonomy_traj.shape}")
print(f"Pointer trajectory shape: {pointer_traj.shape}")
print()

# For first sample in batch
holonomy_sample = holonomy_traj[:, 0]  # [T]
pointer_sample = pointer_traj[:, 0]  # [T]

print("First sample holonomy over 32 timesteps:")
print(holonomy_sample.numpy())
print()

print("First sample pointer positions:")
print(pointer_sample.numpy())
print()

# Count flips
num_flips = 0
for t in range(1, len(holonomy_sample)):
    if holonomy_sample[t] != holonomy_sample[t-1]:
        num_flips += 1
        print(f"Flip at step {t}: holonomy {holonomy_sample[t-1]:.1f} → {holonomy_sample[t]:.1f}, "
              f"pointer {pointer_sample[t-1]:.2f} → {pointer_sample[t]:.2f}")

print()
print(f"Total flips in 32 timesteps: {num_flips}")
print()

# Check holonomy usage across all samples
unique_holonomy = torch.unique(holonomy_traj)
print(f"Unique holonomy values: {unique_holonomy.numpy()}")

num_pos_1 = (holonomy_traj == 1.0).sum().item()
num_neg_1 = (holonomy_traj == -1.0).sum().item()
total = holonomy_traj.numel()

print(f"+1: {num_pos_1}/{total} ({100*num_pos_1/total:.1f}%)")
print(f"-1: {num_neg_1}/{total} ({100*num_neg_1/total:.1f}%)")
