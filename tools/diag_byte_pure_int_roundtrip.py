"""Diag: pure int-rounding roundtrip test.

Take a lossless float baseline, round all weights to int at various scales,
reconstruct, and check if lossless is preserved. No codebook, no LUT —
just int = round(W * scale), then W_deploy = int / scale.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_free_int8 import (
    FreeInt8Merger, load_source_json, load_byte_pairs,
    metrics, DEVICE,
)
from diag_byte_pair_merger_absorb_float import restore_full_state

SOURCE = "output/merger_absorb_float/final_model.json"


def main():
    state = load_source_json(SOURCE)
    model = FreeInt8Merger(state, DEVICE).to(DEVICE)
    restore_full_state(model, state)
    data = load_byte_pairs().to(DEVICE)

    # Sanity check: baseline
    ll0, pd0 = metrics(model, data)
    print(f"Baseline (current hybrid): ll={ll0:.4f}%, pd={pd0:.4f}%\n")

    # Extract effective W1, W2 (the actual float values that the model uses)
    with torch.no_grad():
        W1 = model.W1_eff().cpu().numpy()
        W2 = model.W2_eff().cpu().numpy()
        b1 = model.b1.data.cpu().numpy()
        b2 = model.b2.data.cpu().numpy()
        db1 = model.db1.data.cpu().numpy()
        db2 = model.db2.data.cpu().numpy()

    print(f"W1 range: [{W1.min():.4f}, {W1.max():.4f}], abs_max={np.abs(W1).max():.4f}")
    print(f"W2 range: [{W2.min():.4f}, {W2.max():.4f}], abs_max={np.abs(W2).max():.4f}")
    print(f"Total weight cells: {W1.size + W2.size}\n")

    # Test scales: W_int = round(W * scale); W_reconstruct = W_int / scale
    print(f"{'Scale':>8} {'max int':>8} {'bits/cell':>9} {'deploy KB':>10} {'lossless':>10} {'per_dim':>9}")
    print("-" * 70)

    total_cells = W1.size + W2.size
    for scale in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        W1_int = np.round(W1 * scale).astype(np.int64)
        W2_int = np.round(W2 * scale).astype(np.int64)
        max_abs_int = max(np.abs(W1_int).max(), np.abs(W2_int).max())
        # bits needed for 2-complement: max(1, ceil(log2(max_abs_int + 1)) + 1)
        bits = max(2, int(np.ceil(np.log2(max_abs_int + 1))) + 1)

        # Install reconstructed weights into the model (without gradient)
        with torch.no_grad():
            # Save originals
            W1_orig = model.W1_float.data.clone()
            W2_orig = model.W2_float.data.clone()
            frozen1 = model.W1_frozen_mask.clone()
            frozen2 = model.W2_frozen_mask.clone()
            int8_1 = model.W1_int8_mask.clone()
            int8_2 = model.W2_int8_mask.clone()

            # Fully unfreeze and set all weights to reconstructed float
            model.W1_frozen_mask.zero_()
            model.W2_frozen_mask.zero_()
            model.W1_int8_mask.zero_()
            model.W2_int8_mask.zero_()
            model.W1_float.data.copy_(torch.tensor(W1_int / scale, dtype=torch.float32).to(DEVICE))
            model.W2_float.data.copy_(torch.tensor(W2_int / scale, dtype=torch.float32).to(DEVICE))

        ll, pd = metrics(model, data)

        deploy_bytes = int(np.ceil(total_cells * bits / 8)) + 4  # +4 for scale metadata
        # bias + misc (keep float)
        deploy_bytes += (b1.size + b2.size + db1.size + db2.size) * 4
        # c19 params
        deploy_bytes += 2 * model.H * 4

        marker = " OK" if ll >= 100.0 else ("" if ll >= 99.9 else " BAD")
        print(f"{scale:>8} {max_abs_int:>8} {bits:>9} {deploy_bytes/1024:>9.2f} {ll:>9.4f}% {pd:>8.4f}%{marker}")

        # Restore original state for next iteration
        with torch.no_grad():
            model.W1_float.data.copy_(W1_orig)
            model.W2_float.data.copy_(W2_orig)
            model.W1_frozen_mask.copy_(frozen1)
            model.W2_frozen_mask.copy_(frozen2)
            model.W1_int8_mask.copy_(int8_1)
            model.W2_int8_mask.copy_(int8_2)


if __name__ == "__main__":
    main()
