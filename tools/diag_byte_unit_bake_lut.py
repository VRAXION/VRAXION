"""Bake the byte unit into a LUT — zero computation at inference.

Since there are only 256 possible inputs, pre-compute all 256 latent vectors.
Then try to simplify/quantize the LUT itself:
  1. Raw float32 LUT (256 x 16 = 16KB)
  2. Int8 quantized LUT (4KB)
  3. Int4 quantized LUT (2KB)
  4. Common factor analysis — can we factorize?
  5. Check: does the mirror decoder still work on quantized latents?
  6. Deduplicate / cluster analysis — any byte pairs with near-identical latents?

Also: can we bake the DECODER into a second LUT? (latent → byte)
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BITS = 8
H = 24
OUT_DIM = 16


def c19(x, c, rho):
    c = max(c, 0.1)
    rho = max(rho, 0.0)
    L = 6.0 * c
    if x >= L: return x - L
    if x <= -L: return x + L
    s = x / c
    n = int(s) if s >= 0 else int(s) - 1
    t = s - n
    h = t * (1.0 - t)
    sgn = 1.0 if n % 2 == 0 else -1.0
    return c * (sgn * h + rho * h * h)


def main():
    # Load baked weights
    with open("tools/byte_unit_winner_int4.json") as f:
        data = json.load(f)

    W1 = np.array(data["W1_int4"], dtype=np.float64)  # (8, 24)
    W2 = np.array(data["W2_int4"], dtype=np.float64)  # (24, 16)
    b1 = np.array(data["bias1"])
    b2 = np.array(data["bias2"])
    c_arr = np.array(data["c19_c"])
    rho_arr = np.array(data["c19_rho"])
    sW1 = data["scale_W1"]
    sW2 = data["scale_W2"]

    # Compute all 256 latent vectors
    print("Computing all 256 latent vectors...\n")
    latents = np.zeros((256, OUT_DIM))
    bits_all = np.zeros((256, 8))

    for byte_val in range(256):
        bits = np.array([(byte_val >> i) & 1 for i in range(8)], dtype=np.float64)
        bits_all[byte_val] = bits
        inp = bits * 2.0 - 1.0  # {-1, +1}

        # Layer 1: inp @ W1*sW1 + b1 -> C19
        hidden = np.zeros(H)
        for j in range(H):
            dot = b1[j]
            for i in range(8):
                dot += inp[i] * W1[i, j] * sW1
            hidden[j] = c19(dot, c_arr[j], rho_arr[j])

        # Layer 2: hidden @ W2*sW2 + b2
        for k in range(OUT_DIM):
            s = b2[k]
            for j in range(H):
                s += hidden[j] * W2[j, k] * sW2
            latents[byte_val, k] = s

    # Verify lossless with float latents
    def decode_and_check(lat_table):
        correct = 0
        for byte_val in range(256):
            lat = lat_table[byte_val]
            # Decode: lat @ W2^T*sW2 @ W1^T*sW1
            h2 = np.zeros(H)
            for j in range(H):
                for k in range(OUT_DIM):
                    h2[j] += lat[k] * W2[j, k] * sW2
            recon = np.zeros(8)
            for i in range(8):
                for j in range(H):
                    recon[i] += h2[j] * W1[i, j] * sW1
            recon_bits = (recon > 0).astype(int)
            if np.array_equal(recon_bits, bits_all[byte_val].astype(int)):
                correct += 1
        return correct

    float_correct = decode_and_check(latents)
    print(f"Float32 LUT: {float_correct}/256 lossless")
    print(f"  Size: {256 * 16 * 4} bytes (16KB)")

    # Int8 quantized LUT
    lat_min = latents.min()
    lat_max = latents.max()
    print(f"\nLatent range: [{lat_min:.4f}, {lat_max:.4f}]")

    def quantize_lut(lat, bits):
        max_val = (2 ** (bits - 1)) - 1
        scale = max(abs(lat.min()), abs(lat.max())) / max_val
        q = np.clip(np.round(lat / scale), -max_val, max_val)
        return q * scale, scale, q.astype(int)

    for qbits, label in [(8, "int8"), (6, "int6"), (5, "int5"), (4, "int4"), (3, "int3")]:
        q_lat, scale, q_int = quantize_lut(latents, qbits)
        correct = decode_and_check(q_lat)
        storage = (256 * 16 * qbits + 7) // 8
        print(f"\n{label} LUT: {correct}/256 lossless")
        print(f"  Size: {storage} bytes ({storage/1024:.1f}KB)")
        print(f"  Scale: {scale:.6f}")

    # Analysis: unique latent rows
    print(f"\n{'='*60}")
    print("LUT ANALYSIS")
    print(f"{'='*60}")

    # Pairwise distances
    from itertools import combinations
    dists = []
    for i, j in combinations(range(256), 2):
        d = np.linalg.norm(latents[i] - latents[j])
        dists.append((d, i, j))
    dists.sort()

    print(f"\nClosest 10 byte pairs (latent distance):")
    for d, i, j in dists[:10]:
        ci = chr(i) if 32 <= i < 127 else '?'
        cj = chr(j) if 32 <= j < 127 else '?'
        print(f"  {i:3d} '{ci}' <-> {j:3d} '{cj}': dist={d:.6f}")

    print(f"\nFarthest 5 byte pairs:")
    for d, i, j in dists[-5:]:
        ci = chr(i) if 32 <= i < 127 else '?'
        cj = chr(j) if 32 <= j < 127 else '?'
        print(f"  {i:3d} '{ci}' <-> {j:3d} '{cj}': dist={d:.6f}")

    # Per-dimension stats
    print(f"\nPer-dimension range of latent table:")
    for k in range(OUT_DIM):
        col = latents[:, k]
        print(f"  z{k:02d}: [{col.min():>8.4f}, {col.max():>8.4f}]  spread={col.max()-col.min():.4f}  unique_int4={len(np.unique(np.round(col / (max(abs(col.min()), abs(col.max())) / 7))))}")

    # Common factor analysis on int4 weights
    print(f"\n{'='*60}")
    print("COMMON FACTOR ANALYSIS (W1 int4)")
    print(f"{'='*60}")
    # Check if rows of W1 share GCD patterns
    for i in range(8):
        row = W1[i].astype(int)
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            from math import gcd
            from functools import reduce
            g = reduce(gcd, [abs(v) for v in nonzero])
            print(f"  b{i}: GCD of nonzeros = {g}  row={row.tolist()}")

    # Final summary: best bake format
    print(f"\n{'='*60}")
    print("DEPLOYMENT OPTIONS")
    print(f"{'='*60}")

    # Option 1: raw LUT
    print(f"\n  A) Float32 LUT (zero compute, table lookup)")
    print(f"     256 x 16 x 4B = 16,384 bytes")
    print(f"     Lossless: {float_correct}/256")
    print(f"     Inference: lut[byte] -> 16D float")

    # Option 2: int8 LUT
    q_lat8, sc8, qi8 = quantize_lut(latents, 8)
    c8 = decode_and_check(q_lat8)
    print(f"\n  B) Int8 LUT (zero compute, 1 scale multiply)")
    print(f"     256 x 16 x 1B + 4B scale = 4,100 bytes")
    print(f"     Lossless: {c8}/256")
    print(f"     Inference: lut_int8[byte] * scale -> 16D float")

    # Option 3: keep neural network (288B weights, needs C19 compute)
    print(f"\n  C) Neural network (288B int4 weights, C19 compute)")
    print(f"     288 bytes weights + 352B params = 640 bytes")
    print(f"     Lossless: 256/256")
    print(f"     Inference: 8 bit -> matmul -> C19 -> matmul -> 16D")

    # Option 4: hybrid — LUT encode, neural decode (for mirror)
    print(f"\n  D) Hybrid: LUT encode + neural decode")
    print(f"     LUT: 4,100B (int8) for encode")
    print(f"     Decoder: 288B (int4 weights) for mirror reconstruct")
    print(f"     Total: 4,388 bytes")
    print(f"     Encode: lut[byte] (instant)")
    print(f"     Decode: matmul chain (for verification)")


if __name__ == "__main__":
    main()
