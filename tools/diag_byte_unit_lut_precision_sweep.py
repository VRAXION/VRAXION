"""Test: what's the smallest LUT precision that preserves 100% lossless?

We have binary weights already. The baked LUT is int8. The question: can we
quantize the LUT values themselves down (int4, int2, int1) and still have
the tied-mirror decoder reconstruct all 256 bytes correctly?

Tests each precision by:
  1. Loading the frozen binary+C19+H=16 champion
  2. Encoding all 256 bytes -> (256, 16) float LUT
  3. Quantizing the LUT to N levels (int8, int4, int2, int1)
  4. Running the decoder on the quantized LUT
  5. Checking byte reconstruction (256/256?)

If binary (int1) still gives 256/256, then the DEPLOY LUT can be 512 B.

Usage:
  python tools/diag_byte_unit_lut_precision_sweep.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_unit_widen_sweep import (
    DEVICE, CODEBOOKS, QuantByteUnit, build_dataset, metrics,
)

CHAMPION_DIR = Path("output/byte_unit_champion_binary_c19_h16")
OUT_DIR = Path("output/byte_unit_lut_precision_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN = 16
ACTIVATION = "c19"
CODEBOOK_NAME = "binary"
CODEBOOK = CODEBOOKS[CODEBOOK_NAME]


def reload_champion():
    blob = json.loads((CHAMPION_DIR / "byte_unit_winner_binary.json").read_text(encoding="utf-8"))
    model = QuantByteUnit(HIDDEN, ACTIVATION, CODEBOOK).to(DEVICE)

    W1_idx = np.array(blob["W1_binary_idx"], dtype=np.int64)
    W1_levels = np.array(blob["W1_levels"], dtype=np.float32)
    W1 = W1_levels[W1_idx]
    W2_idx = np.array(blob["W2_binary_idx"], dtype=np.int64)
    W2_levels = np.array(blob["W2_levels"], dtype=np.float32)
    W2 = W2_levels[W2_idx]

    with torch.no_grad():
        model.W1.copy_(torch.tensor(W1, device=DEVICE))
        model.W2.copy_(torch.tensor(W2, device=DEVICE))
        model.b1.copy_(torch.tensor(blob["b1"], device=DEVICE))
        model.b2.copy_(torch.tensor(blob["b2"], device=DEVICE))
        a1_raw = math.log(math.exp(blob["alpha1"]) - 1)
        a2_raw = math.log(math.exp(blob["alpha2"]) - 1)
        model.alpha1_raw.copy_(torch.tensor(a1_raw, device=DEVICE))
        model.alpha2_raw.copy_(torch.tensor(a2_raw, device=DEVICE))
        for name, p in model.act.named_parameters():
            p.copy_(torch.tensor(blob["activation_params"][name], device=DEVICE))

    return model


def quantize_symmetric(lut: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Symmetric linear quantization to signed N-bit integer range."""
    if bits == 8:
        levels = 127  # int8: -127..+127
    elif bits == 4:
        levels = 7    # int4: -7..+7
    elif bits == 2:
        levels = 1    # int2: -1, +1 (actually this is -1..+1 range)
    elif bits == 1:
        levels = 1    # int1: -1, +1 signum-like
    else:
        raise ValueError(f"unsupported bits: {bits}")

    absmax = float(np.abs(lut).max())
    if absmax == 0:
        return lut.astype(np.int32), 1.0
    scale = absmax / levels
    q = np.round(lut / scale).clip(-levels, levels)
    return q.astype(np.int32), scale


def quantize_binary_sign(lut: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure binary: sign(lut) * mean_abs per dim."""
    mean_abs = float(np.abs(lut).mean())
    signs = np.sign(lut).astype(np.int32)
    # Handle zeros by mapping to +1
    signs[signs == 0] = 1
    return signs, mean_abs


def quantize_ternary(lut: np.ndarray) -> tuple[np.ndarray, float]:
    """Ternary: map to {-1, 0, +1} based on magnitude threshold."""
    absmax = float(np.abs(lut).max())
    thr = absmax / 3  # heuristic threshold
    q = np.zeros_like(lut, dtype=np.int32)
    q[lut > thr] = 1
    q[lut < -thr] = -1
    scale = absmax / 1
    return q, scale


def decode_via_lut(lut: torch.Tensor, model) -> torch.Tensor:
    """Decode a (256, 16) LUT back to (256, 8) bit-signs via the model's decoder."""
    with torch.no_grad():
        x_hat = model.decode(lut)  # (256, 8)
    return x_hat


def byte_match(x_hat_signs: np.ndarray, x_orig: np.ndarray) -> int:
    """Count how many of 256 bytes decode correctly."""
    # x_orig is (256, 8) with -1/+1 per bit
    # x_hat_signs is predicted signs
    # Bytes match iff all 8 signs match
    match_per_byte = (x_hat_signs == x_orig).all(axis=1)
    return int(match_per_byte.sum())


def main():
    print("=" * 70)
    print("LUT PRECISION SWEEP — how low can we go and keep 256/256 lossless?")
    print("=" * 70)

    print(f"\n[1] Loading binary+C19+H=16 champion from {CHAMPION_DIR}")
    model = reload_champion()
    model.eval()

    x = build_dataset()  # (256, 8) with -1/+1 per bit
    x_np = x.cpu().numpy()
    m = metrics(model, x)
    print(f"    weight-reload lossless: {m['lossless']:.2f}% (sanity)")

    with torch.no_grad():
        float_lut = model.encode(x).cpu().numpy()  # (256, 16) float
    print(f"    float LUT shape: {float_lut.shape}  range: [{float_lut.min():+.3f}, {float_lut.max():+.3f}]")

    results = []

    for bits in [8, 4, 2, 1]:
        q_lut, scale = quantize_symmetric(float_lut, bits)
        dequant = q_lut.astype(np.float32) * scale
        dequant_t = torch.tensor(dequant, device=DEVICE)
        x_hat = decode_via_lut(dequant_t, model).cpu().numpy()
        x_hat_signs = np.sign(x_hat).astype(np.int32)
        x_hat_signs[x_hat_signs == 0] = 1
        matches = byte_match(x_hat_signs, x_np.astype(np.int32))
        # Size calc
        raw_bytes = 256 * 16 * bits / 8
        print(f"  [symmetric int{bits}]: {matches:>3d}/256 bytes OK  "
              f"raw LUT = {raw_bytes:>5.0f} B  scale={scale:.4e}")
        results.append({
            "method": f"symmetric_int{bits}",
            "bits_per_value": bits,
            "bytes_decoded": matches,
            "lossless": matches == 256,
            "raw_lut_bytes": int(raw_bytes),
            "scale": float(scale),
        })

    # Try sign-based binary (different from symmetric_int1)
    q_bin, mean_abs = quantize_binary_sign(float_lut)
    dequant = q_bin.astype(np.float32) * mean_abs
    dequant_t = torch.tensor(dequant, device=DEVICE)
    x_hat = decode_via_lut(dequant_t, model).cpu().numpy()
    x_hat_signs = np.sign(x_hat).astype(np.int32)
    x_hat_signs[x_hat_signs == 0] = 1
    matches = byte_match(x_hat_signs, x_np.astype(np.int32))
    print(f"  [binary sign]:          {matches:>3d}/256 bytes OK  "
          f"raw LUT = {256*16/8:>5.0f} B  (1 bit per value)")
    results.append({
        "method": "binary_sign",
        "bits_per_value": 1,
        "bytes_decoded": matches,
        "lossless": matches == 256,
        "raw_lut_bytes": 256*16//8,
        "scale": float(mean_abs),
    })

    # Try ternary
    q_tern, scale_t = quantize_ternary(float_lut)
    dequant = q_tern.astype(np.float32) * scale_t
    dequant_t = torch.tensor(dequant, device=DEVICE)
    x_hat = decode_via_lut(dequant_t, model).cpu().numpy()
    x_hat_signs = np.sign(x_hat).astype(np.int32)
    x_hat_signs[x_hat_signs == 0] = 1
    matches = byte_match(x_hat_signs, x_np.astype(np.int32))
    print(f"  [ternary]:              {matches:>3d}/256 bytes OK  "
          f"raw LUT ~= {256*16*1.585/8:>5.0f} B packed")
    results.append({
        "method": "ternary",
        "bits_per_value": 1.585,
        "bytes_decoded": matches,
        "lossless": matches == 256,
        "raw_lut_bytes_packed": int(256*16*1.585/8),
        "scale": float(scale_t),
    })

    print("\n" + "=" * 70)
    print("VERDICT — smallest lossless LUT precision")
    print("=" * 70)
    lossless_opts = [r for r in results if r["lossless"]]
    if lossless_opts:
        smallest = min(lossless_opts, key=lambda r: r.get("raw_lut_bytes", 99999))
        print(f"  Lossless methods found: {len(lossless_opts)}")
        for r in lossless_opts:
            raw_b = r.get("raw_lut_bytes", r.get("raw_lut_bytes_packed", "?"))
            print(f"    - {r['method']:<20} {raw_b} B raw")
        print(f"\n  SMALLEST LOSSLESS: {smallest['method']} -> {smallest.get('raw_lut_bytes', smallest.get('raw_lut_bytes_packed'))} B")
        print(f"  vs current int8 LUT: 4096 B")
    else:
        print(f"  NO method below int8 stayed lossless. The int8 LUT is minimum for this decoder.")
        best = max(results, key=lambda r: r["bytes_decoded"])
        print(f"  Best non-lossless: {best['method']} -> {best['bytes_decoded']}/256")

    out_path = OUT_DIR / "precision_sweep.json"
    out_path.write_text(json.dumps({
        "champion": "binary + C19 + H=16",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
