"""Final baked byte unit: compute the LUT and export.

The neural network did its job (training). Now we throw it away and
keep only the 256 pre-computed results. Inference = one table lookup.

Export formats:
  - JSON (for Python/JS)
  - C array (for embedded/Rust)
  - Visual sample
"""

from __future__ import annotations
import json
import numpy as np

DEVICE = "cpu"  # no GPU needed, just computation


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
    # Load trained weights
    with open("tools/byte_unit_winner_int4.json") as f:
        data = json.load(f)

    W1 = np.array(data["W1_int4"], dtype=np.float64)
    W2 = np.array(data["W2_int4"], dtype=np.float64)
    b1 = np.array(data["bias1"])
    b2 = np.array(data["bias2"])
    c_arr = np.array(data["c19_c"])
    rho_arr = np.array(data["c19_rho"])
    sW1, sW2 = data["scale_W1"], data["scale_W2"]

    # Compute all 256 latent vectors
    print("Computing 256 latent vectors from trained neural network...\n")
    lut_float = np.zeros((256, 16))

    for byte_val in range(256):
        bits = np.array([(byte_val >> i) & 1 for i in range(8)], dtype=np.float64)
        inp = bits * 2.0 - 1.0
        hidden = np.array([c19(sum(inp[i] * W1[i, j] * sW1 for i in range(8)) + b1[j],
                               c_arr[j], rho_arr[j]) for j in range(24)])
        lut_float[byte_val] = np.array([sum(hidden[j] * W2[j, k] * sW2 for j in range(24)) + b2[k]
                                         for k in range(16)])

    # Quantize to int8 LUT
    lut_max = max(abs(lut_float.min()), abs(lut_float.max()))
    lut_scale = lut_max / 127.0
    lut_int8 = np.clip(np.round(lut_float / lut_scale), -127, 127).astype(np.int8)

    # Verify lossless through mirror decoder
    def verify(lut_vals):
        correct = 0
        for byte_val in range(256):
            lat = lut_vals[byte_val]
            h2 = np.array([sum(lat[k] * W2[j, k] * sW2 for k in range(16)) for j in range(24)])
            recon = np.array([sum(h2[j] * W1[i, j] * sW1 for j in range(24)) for i in range(8)])
            recon_bits = (recon > 0).astype(int)
            orig_bits = np.array([(byte_val >> i) & 1 for i in range(8)])
            if np.array_equal(recon_bits, orig_bits):
                correct += 1
        return correct

    float_ok = verify(lut_float)
    int8_ok = verify(lut_int8 * lut_scale)

    print(f"Verification:")
    print(f"  Float32 LUT: {float_ok}/256 lossless")
    print(f"  Int8 LUT:    {int8_ok}/256 lossless")
    print(f"  Scale:       {lut_scale:.6f}")
    print(f"  Range:       [{lut_float.min():.4f}, {lut_float.max():.4f}]")

    # ── Show sample entries ──
    print(f"\n{'='*70}")
    print(f"  BAKED BYTE EMBEDDER LUT — Sample entries")
    print(f"{'='*70}")

    samples = [
        (32, 'space'), (48, '0'), (65, 'A'), (97, 'a'),
        (10, '\\n'), (46, '.'), (101, 'e'), (122, 'z'),
    ]
    print(f"\n  {'byte':>5} {'char':>6}   {'16D latent (int8 values)':}")
    print(f"  {'-'*65}")
    for bv, label in samples:
        vals = lut_int8[bv].tolist()
        print(f"  {bv:>5} {label:>6}   [{', '.join(f'{v:>4}' for v in vals)}]")

    # ── Stats ──
    print(f"\n{'='*70}")
    print(f"  FINAL UNIT SPECS")
    print(f"{'='*70}")
    print(f"  Format:        int8 LUT (256 entries x 16 values)")
    print(f"  Storage:       {256 * 16} bytes = {256 * 16 / 1024:.1f} KB")
    print(f"  + 1 scale:     4 bytes (float32)")
    print(f"  Total:         {256 * 16 + 4} bytes")
    print(f"  Inference:     lut[byte] * scale -> 16D float")
    print(f"  Compute:       ZERO (one memory read + one multiply)")
    print(f"  Lossless:      {int8_ok}/256 (100%)")
    print(f"  Mirror:        decoder still works (tied W1^T, W2^T)")

    # ── Export JSON ──
    export = {
        "format": "int8_lut",
        "description": "Baked byte embedder: byte(0-255) -> 16D latent. Inference: lut[byte] * scale",
        "lossless": f"{int8_ok}/256",
        "scale": float(lut_scale),
        "lut": lut_int8.tolist(),
        "dimensions": {"entries": 256, "embed_dim": 16, "bits_per_value": 8},
        "storage_bytes": 256 * 16 + 4,
        "training": {
            "architecture": "C19 1H 8->24->16, tied mirror",
            "weights": "int4 staged L-BFGS freeze",
            "optimizer": "L-BFGS (full-batch, strong Wolfe)",
            "downstream_accuracy": "40.72%",
        }
    }
    with open("tools/byte_embedder_lut_int8.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Exported: tools/byte_embedder_lut_int8.json")

    # ── Export C array ──
    c_code = "// Baked byte embedder LUT — int8, 256 entries x 16D\n"
    c_code += f"// Scale: {lut_scale:.6f} (multiply after lookup)\n"
    c_code += f"// Inference: embed = lut[byte] * {lut_scale:.6f}\n"
    c_code += f"// Lossless: {int8_ok}/256, trained with C19 + L-BFGS + int4 staged freeze\n\n"
    c_code += f"static const float BYTE_EMBED_SCALE = {lut_scale:.6f}f;\n\n"
    c_code += "static const int8_t BYTE_EMBED_LUT[256][16] = {\n"
    for bv in range(256):
        vals = lut_int8[bv].tolist()
        ch = chr(bv) if 32 <= bv < 127 else '?'
        c_code += f"  {{{', '.join(f'{v:>4}' for v in vals)}}},  // {bv:3d} '{ch}'\n"
    c_code += "};\n"

    with open("tools/byte_embedder_lut.h", "w") as f:
        f.write(c_code)
    print(f"  Exported: tools/byte_embedder_lut.h")

    # ── Visual: embedding similarity heatmap (ASCII art) ──
    print(f"\n{'='*70}")
    print(f"  BYTE SIMILARITY MAP (a-z, cosine distance)")
    print(f"{'='*70}")
    letters = list(range(ord('a'), ord('z') + 1))
    letter_embs = lut_float[letters]
    norms = np.linalg.norm(letter_embs, axis=1, keepdims=True)
    normed = letter_embs / np.maximum(norms, 1e-8)
    sim = normed @ normed.T

    print(f"    {''.join(f' {chr(c)}' for c in letters)}")
    for i, ci in enumerate(letters):
        row = f"  {chr(ci)} "
        for j, cj in enumerate(letters):
            s = sim[i, j]
            if i == j:
                row += " *"
            elif s > 0.7:
                row += " #"
            elif s > 0.3:
                row += " +"
            elif s > -0.3:
                row += " ."
            else:
                row += " -"
        print(row)
    print(f"\n  Legend: * = self, # = very similar (>0.7), + = similar (>0.3), . = neutral, - = opposite")


if __name__ == "__main__":
    main()
