"""Check where the zeros in the byte embedder LUT come from.

Hypothesis: the neural network produces small non-zero floats, which get
rounded to zero during int8 quantization (scale = 0.132).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LUT_INT8 = Path(__file__).with_name("byte_embedder_lut_int8.json")
WINNER = Path(__file__).with_name("byte_unit_winner_int4.json")


def c19(x, c, rho):
    c = max(c, 0.1)
    rho = max(rho, 0.0)
    L = 6.0 * c
    if x >= L: return x - L
    if x <= -L: return x + L
    s = x / c
    n = int(np.floor(s))
    t = s - n
    h = t * (1.0 - t)
    sgn = 1.0 if n % 2 == 0 else -1.0
    return c * (sgn * h + rho * h * h)


def main():
    # Load int8 LUT
    with open(LUT_INT8) as f:
        blob = json.load(f)
    scale = blob["scale"]
    lut_int8 = np.array(blob["lut"], dtype=np.int32)

    # Load trained weights to recompute the float LUT
    with open(WINNER) as f:
        data = json.load(f)
    W1 = np.array(data["W1_int4"], dtype=np.float64)
    W2 = np.array(data["W2_int4"], dtype=np.float64)
    b1 = np.array(data["bias1"])
    b2 = np.array(data["bias2"])
    c_arr = np.array(data["c19_c"])
    rho_arr = np.array(data["c19_rho"])
    sW1, sW2 = data["scale_W1"], data["scale_W2"]

    # Recompute the float LUT from the trained model
    lut_float = np.zeros((256, 16))
    for byte_val in range(256):
        bits = np.array([(byte_val >> i) & 1 for i in range(8)],
                        dtype=np.float64) * 2.0 - 1.0
        hidden = np.zeros(24)
        for j in range(24):
            dot = b1[j]
            for i in range(8):
                dot += bits[i] * W1[i, j] * sW1
            hidden[j] = c19(dot, c_arr[j], rho_arr[j])
        for k in range(16):
            s = b2[k]
            for j in range(24):
                s += hidden[j] * W2[j, k] * sW2
            lut_float[byte_val, k] = s

    print(f"Int8 LUT scale: {scale:.6f}")
    print(f"Quantization threshold: any |float| < {scale/2:.6f} becomes 0")
    print()

    # Count zeros in int8 LUT
    n_zeros = (lut_int8 == 0).sum()
    print(f"Zeros in int8 LUT: {n_zeros}")
    print(f"Bytes with at least one zero dim: {(lut_int8 == 0).any(axis=1).sum()}")
    print()

    # Find corresponding float values
    print("=== WHAT THE FLOAT VALUES ACTUALLY WERE (at int8=0 positions) ===")
    zero_mask = (lut_int8 == 0)
    float_at_zeros = lut_float[zero_mask]
    print(f"Number of positions: {len(float_at_zeros)}")
    print(f"Float range at these positions: "
          f"[{float_at_zeros.min():.4f}, {float_at_zeros.max():.4f}]")
    print(f"Mean abs: {np.abs(float_at_zeros).mean():.4f}")
    print()

    print("Top 20 largest abs values that still rounded to 0:")
    sorted_abs = sorted(np.abs(float_at_zeros), reverse=True)[:20]
    for v in sorted_abs:
        print(f"  |{v:.5f}|  (scale/2 = {scale/2:.5f})")

    print()
    print("=== THE FIX: what if we used a smaller scale? ===")
    # For the current range, try alternate scales
    lut_max = max(abs(lut_float.min()), abs(lut_float.max()))
    print(f"LUT max abs value: {lut_max:.4f}")
    print(f"Current scale: {scale:.6f} (= max/127)")

    # Try with int16 instead
    scale_int16 = lut_max / 32767
    lut_int16 = np.round(lut_float / scale_int16).astype(np.int32)
    zeros_int16 = (lut_int16 == 0).sum()
    print(f"\nIf we used int16 instead:")
    print(f"  scale = {scale_int16:.8f}")
    print(f"  zeros = {zeros_int16} (vs {n_zeros} in int8)")

    # Alternative: boost small values away from zero
    # Use dead-zone exclusion: if |val| < 0.5*scale, round away from 0
    lut_int8_safe = np.where(
        np.abs(lut_float) < scale,
        np.sign(lut_float) * 1,  # push tiny values to +/-1
        np.round(lut_float / scale)
    ).astype(np.int32)
    # Need to handle exact zero floats
    lut_int8_safe = np.where(
        (lut_int8_safe == 0) & (lut_float != 0),
        np.where(lut_float > 0, 1, -1),
        lut_int8_safe
    ).astype(np.int32)
    zeros_safe = (lut_int8_safe == 0).sum()
    print(f"\nIf we used 'no-zero' int8 (push tiny values to +/-1):")
    print(f"  zeros = {zeros_safe}")

    # Check reconstruction quality if we use the no-zero LUT
    # Reconstruct using the no-zero int8 through the decoder
    lut_safe_float = lut_int8_safe * scale
    print(f"\nLUT diff (safe vs original int8): "
          f"max={np.abs(lut_safe_float - lut_int8*scale).max():.4f}")


if __name__ == "__main__":
    main()
