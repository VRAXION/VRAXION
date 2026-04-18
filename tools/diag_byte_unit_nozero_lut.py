"""Regenerate the byte embedder LUT as a zero-free int8 table.

The problem: the original int8 LUT contained 40 exact zeros because the
trained neural network produced small non-zero values that rounded to 0
during quantization (scale = 0.132, so anything |x| < 0.066 rounded to 0).

The fix: clamp any would-be-zero to +/-1 (preserving its sign from the
original float value). This adds at most 1 scale unit (~0.132) of error
per affected dimension, but eliminates zeros entirely.

Verify:
  - Byte embedder still 100% lossless (sign preserved -> mirror decoder works)
  - Zero values removed from LUT
  - Export new LUT as byte_embedder_lut_int8_nozero.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

WINNER = Path(__file__).with_name("byte_unit_winner_int4.json")
OUT_JSON = Path(__file__).with_name("byte_embedder_lut_int8_nozero.json")


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
    with open(WINNER) as f:
        data = json.load(f)
    W1 = np.array(data["W1_int4"], dtype=np.float64)
    W2 = np.array(data["W2_int4"], dtype=np.float64)
    b1 = np.array(data["bias1"])
    b2 = np.array(data["bias2"])
    c_arr = np.array(data["c19_c"])
    rho_arr = np.array(data["c19_rho"])
    sW1, sW2 = data["scale_W1"], data["scale_W2"]

    # Recompute the float LUT (256 x 16)
    print("Computing 256 float latents from trained weights...")
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

    # Quantize to int8 with same scale as before
    lut_max = max(abs(lut_float.min()), abs(lut_float.max()))
    scale = lut_max / 127.0
    lut_int8 = np.round(lut_float / scale).astype(np.int32)

    # Count zeros before fix
    zeros_before = (lut_int8 == 0).sum()
    print(f"\nOriginal int8 LUT:")
    print(f"  scale = {scale:.6f}")
    print(f"  zeros = {zeros_before}")

    # Apply the no-zero fix: push any zero to +/-1 based on original float sign
    # If the original float was positive (even if tiny), push to +1; else -1.
    zero_mask = (lut_int8 == 0)
    # For the tiny-but-nonzero floats, use their float sign
    fix_signs = np.sign(lut_float).astype(np.int32)
    # If exact 0 float (shouldn't happen but guard), default to +1
    fix_signs = np.where(fix_signs == 0, 1, fix_signs)

    lut_int8_fixed = lut_int8.copy()
    lut_int8_fixed[zero_mask] = fix_signs[zero_mask]

    zeros_after = (lut_int8_fixed == 0).sum()
    print(f"\nFixed int8 LUT (no-zero):")
    print(f"  zeros = {zeros_after}")
    print(f"  values changed: {zero_mask.sum()}")
    print(f"  max change per cell: 1 (= {scale:.4f} in float)")

    # Verify byte embedder is still 100% lossless through mirror decoder
    print(f"\nVerifying lossless roundtrip through mirror decoder...")

    def verify(lut_vals_int8, scale):
        correct = 0
        for byte_val in range(256):
            lat = lut_vals_int8[byte_val] * scale  # float latent
            # Mirror decode: lat @ W2^T * sW2 -> hidden (linear, no C19)
            h2 = np.zeros(24)
            for j in range(24):
                for k in range(16):
                    h2[j] += lat[k] * W2[j, k] * sW2
            # Then: h2 @ W1^T * sW1 -> 8D
            recon = np.zeros(8)
            for i in range(8):
                for j in range(24):
                    recon[i] += h2[j] * W1[i, j] * sW1
            # Sign of reconstructed = decoded bits
            recon_bits = (recon > 0).astype(int)
            orig_bits = np.array([(byte_val >> i) & 1 for i in range(8)])
            if np.array_equal(recon_bits, orig_bits):
                correct += 1
        return correct

    orig_ok = verify(lut_int8, scale)
    fixed_ok = verify(lut_int8_fixed, scale)
    print(f"  Original int8 LUT: {orig_ok}/256 lossless")
    print(f"  Fixed int8 LUT:    {fixed_ok}/256 lossless")

    # Check pairs: how many now have no zero across 32 dims?
    idx_a = np.arange(256).repeat(256)
    idx_b = np.tile(np.arange(256), 256)
    pairs = np.concatenate([lut_int8_fixed[idx_a], lut_int8_fixed[idx_b]],
                           axis=1)
    pairs_with_zero = (pairs == 0).any(axis=1).sum()
    print(f"\nPairs with any zero dim: {pairs_with_zero}/65536")
    print(f"(should be 0 now — previously 17,575)")

    # Export
    export = {
        "format": "int8_lut_nozero",
        "description": "Byte embedder LUT, int8, zero-free (tiny floats "
                       "pushed to +/-1). Inference: lut[byte] * scale",
        "lossless": f"{fixed_ok}/256",
        "scale": float(scale),
        "lut": lut_int8_fixed.tolist(),
        "dimensions": {"entries": 256, "embed_dim": 16, "bits_per_value": 8},
        "storage_bytes": 256 * 16 + 4,
        "zero_free": True,
        "changes_from_original": int(zero_mask.sum()),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nExported: {OUT_JSON}")


if __name__ == "__main__":
    main()
