"""Analyze why the tied-weight mirror merger caps at 73.18% lossless.

Questions:
  1. What's the distribution of the 32D pair inputs?
  2. Are many values near zero (sign-flip vulnerable)?
  3. Which pairs fail? Is there a pattern?
  4. Can we identify the 17,575 unsolvable pairs?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LUT_PATH = Path(__file__).with_name("byte_embedder_lut_int8.json")


def main():
    with open(LUT_PATH) as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut_int8 = np.array(blob["lut"], dtype=np.float32)
    lut = lut_int8 * scale  # (256, 16)

    print(f"LUT shape: {lut.shape}, scale={scale:.6f}")
    print(f"Per-byte embedding range: [{lut.min():.4f}, {lut.max():.4f}]")

    # Build all 65,536 byte pairs
    idx_a = np.arange(256).repeat(256)
    idx_b = np.tile(np.arange(256), 256)
    pairs = np.concatenate([lut[idx_a], lut[idx_b]], axis=1)  # (65536, 32)
    print(f"\nPair shape: {pairs.shape}")
    print(f"Pair range: [{pairs.min():.4f}, {pairs.max():.4f}]")

    # --- 1. Distribution of values ---
    print("\n=== VALUE DISTRIBUTION ===")
    abs_vals = np.abs(pairs).flatten()
    print(f"Total dimensions to match (65536 x 32): {abs_vals.size:,}")
    for thresh in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        count = (abs_vals < thresh).sum()
        pct = count / abs_vals.size * 100
        print(f"  |value| < {thresh:.2f}: {count:>10,} ({pct:5.2f}%)")

    # --- 2. Per-dimension stats ---
    print("\n=== PER-DIMENSION STATS ===")
    print("Dim | Min     | Max     | Mean    | Std     | NearZero%")
    for d in range(32):
        col = pairs[:, d]
        near_zero = (np.abs(col) < 0.1).mean() * 100
        print(f"  {d:2d} | {col.min():7.3f} | {col.max():7.3f} | "
              f"{col.mean():7.3f} | {col.std():.3f} | {near_zero:5.2f}%")

    # --- 3. Pairs with any near-zero dim (most vulnerable) ---
    min_abs_per_pair = np.abs(pairs).min(axis=1)
    print("\n=== PAIRS BY MIN ABS DIM ===")
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
        count = (min_abs_per_pair < thresh).sum()
        pct = count / 65536 * 100
        print(f"  min|dim| < {thresh:.2f}: {count:>6,} pairs ({pct:5.2f}%)")

    # --- 4. Unique byte embeddings count ---
    print("\n=== BYTE EMBEDDING UNIQUENESS ===")
    # How many near-zero values per byte?
    near_zero_counts = (np.abs(lut) < 0.1).sum(axis=1)
    print(f"Bytes with near-zero dims (count):")
    for c in sorted(set(near_zero_counts)):
        n_bytes = (near_zero_counts == c).sum()
        print(f"  {c} near-zero dims: {n_bytes} bytes")

    # --- 5. Sign pattern uniqueness ---
    sign_patterns = np.sign(pairs)  # (65536, 32)
    # Count unique sign patterns
    sign_tuples = [tuple(row.tolist()) for row in sign_patterns.astype(int)]
    unique_signs = len(set(sign_tuples))
    print(f"\n=== SIGN PATTERNS ===")
    print(f"Unique sign patterns among 65,536 pairs: {unique_signs:,}")
    print(f"Duplicates: {65536 - unique_signs:,}")

    # --- 6. Check if LUT has any exact zeros ---
    print(f"\n=== EXACT ZEROS ===")
    print(f"Exact zeros in LUT (int8=0): {(lut_int8 == 0).sum()}")
    print(f"  bytes affected: {(lut_int8 == 0).any(axis=1).sum()}")

    # --- 7. Pair input entropy estimate ---
    print(f"\n=== INFO ESTIMATE ===")
    # 32 dims, each with ~256 possible values -> 32*8 = 256 bits max
    # But real data lies on low-dim manifold
    # SVD to see effective rank
    u, s, vt = np.linalg.svd(pairs, full_matrices=False)
    s_norm = s / s.sum()
    cumul = np.cumsum(s_norm)
    print(f"Singular value decomposition of pair matrix:")
    print(f"  Top 1 SV captures: {cumul[0]*100:.1f}%")
    print(f"  Top 4 SV captures: {cumul[3]*100:.1f}%")
    print(f"  Top 8 SV captures: {cumul[7]*100:.1f}%")
    print(f"  Top 16 SV captures: {cumul[15]*100:.1f}%")
    print(f"  Top 24 SV captures: {cumul[23]*100:.1f}%")
    print(f"  Top 32 SV captures: {cumul[31]*100:.1f}%")
    # Effective rank
    entropy = -(s_norm * np.log(s_norm + 1e-10)).sum()
    eff_rank = np.exp(entropy)
    print(f"  Effective rank: {eff_rank:.1f}")


if __name__ == "__main__":
    main()
