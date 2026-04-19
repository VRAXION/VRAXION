"""Throw standard compressors (zlib/gzip, bz2, lzma/7zip) at the raw weights.

Gives a real-world floor for what general-purpose lossless compression achieves.
Shannon entropy is the theoretical limit; LZMA/bzip2 get close to it on
redundant data.
"""
from __future__ import annotations
import json, zlib, bz2, lzma, struct
from pathlib import Path
import numpy as np

ART = Path("output/merger_single_w_hybrid_k/final_hybrid.json")


def test_compressors(data: bytes, label: str):
    """Run all standard compressors on a byte buffer."""
    print(f"\n=== {label} ===")
    print(f"  Raw size: {len(data)} B")
    results = {}
    # zlib (gzip core)
    for lvl in [6, 9]:
        c = zlib.compress(data, level=lvl)
        results[f"zlib-{lvl}"] = len(c)
    # bz2
    for lvl in [6, 9]:
        c = bz2.compress(data, compresslevel=lvl)
        results[f"bz2-{lvl}"] = len(c)
    # lzma (xz/7zip core)
    for preset in [6, 9, 9 | lzma.PRESET_EXTREME]:
        c = lzma.compress(data, preset=preset)
        label_p = f"lzma-{preset}" if preset < 16 else "lzma-9E"
        results[label_p] = len(c)

    for name, size in sorted(results.items(), key=lambda x: x[1]):
        pct = 100 * size / len(data)
        print(f"  {name:12s}: {size:5d} B  ({pct:5.1f}% of raw)")
    return results


def main():
    with open(ART) as f:
        m = json.load(f)

    # Concatenate all values as fp16 bytes
    parts_fp16 = {}
    parts_fp32 = {}
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        arr32 = np.array(m[f"{comp}_final"], dtype=np.float32).flatten()
        parts_fp32[comp] = arr32
        parts_fp16[comp] = arr32.astype(np.float16)

    all_fp16 = np.concatenate([parts_fp16[c] for c in ["W", "b1", "b2", "c19_c", "c19_rho"]])
    all_fp32 = np.concatenate([parts_fp32[c] for c in ["W", "b1", "b2", "c19_c", "c19_rho"]])

    print("=" * 70)
    print("STANDARD COMPRESSION TEST on hybrid-K weights")
    print("=" * 70)
    print(f"  Total cells: {len(all_fp16)}")
    print(f"  fp16 naive:  {len(all_fp16) * 2} B")
    print(f"  fp32 naive:  {len(all_fp32) * 4} B")
    print(f"  Hybrid-K champion: 4008 B")
    print(f"  Shannon raw floor: 2687 B")

    # 1. Raw fp16 concatenation
    raw_fp16 = all_fp16.tobytes()
    r1 = test_compressors(raw_fp16, "RAW FP16 (2867 x 2B = 5734 B)")

    # 2. Per-component fp16
    print(f"\n{'='*70}")
    print(f"PER-COMPONENT (fp16) — lzma-9E only for brevity")
    print(f"{'='*70}")
    per_comp_total = 0
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        data = parts_fp16[comp].tobytes()
        c = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
        per_comp_total += len(c)
        print(f"  {comp:10s}: raw {len(data):4d}B -> lzma {len(c):4d}B ({100*len(c)/len(data):.1f}%)")
    print(f"  TOTAL:      {per_comp_total} B")

    # 3. Raw fp32 (more unique values, usually worse to compress)
    raw_fp32 = all_fp32.tobytes()
    r3 = test_compressors(raw_fp32, "RAW FP32 (2867 x 4B = 11468 B)")

    # 4. Sorted fp16 — sometimes sorting reveals structure
    sorted_fp16 = np.sort(all_fp16.view(np.uint16))
    # Delta-encode the sorted values
    deltas = np.diff(sorted_fp16).astype(np.int32)
    deltas_bytes = deltas.tobytes()
    first = struct.pack("<H", int(sorted_fp16[0]))
    r4 = test_compressors(first + deltas_bytes, "SORTED FP16 + DELTAS")

    # 5. Grouped by sign (positive + negative separately)
    pos = all_fp32[all_fp32 > 0].astype(np.float16).tobytes()
    neg = all_fp32[all_fp32 < 0].astype(np.float16).tobytes()
    zer = np.zeros(int((all_fp32 == 0).sum()), dtype=np.float16).tobytes()
    combined = pos + neg + zer
    r5 = test_compressors(combined, "SIGN-GROUPED FP16")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — where does 'zip it' land?")
    print(f"{'='*70}")
    print(f"  Naive fp16:             5734 B    (100%)")
    print(f"  Hybrid-K champion:      4008 B    ({100*4008/5734:.1f}%)")
    best_raw = min(r1.values())
    best_sorted = min(r4.values())
    best_grouped = min(r5.values())
    print(f"  Best on raw fp16:       {best_raw} B  ({100*best_raw/5734:.1f}%)")
    print(f"  Best on sorted+delta:   {best_sorted} B  ({100*best_sorted/5734:.1f}%)")
    print(f"  Best on sign-grouped:   {best_grouped} B  ({100*best_grouped/5734:.1f}%)")
    print(f"  Per-component lzma:     {per_comp_total} B  ({100*per_comp_total/5734:.1f}%)")
    print(f"  Shannon raw floor:      2687 B    ({100*2687/5734:.1f}%) [theoretical]")
    print(f"  Shannon topology floor: 2422 B    ({100*2422/5734:.1f}%) [theoretical]")

    print(f"\nINTERPRETATION:")
    best_overall = min(best_raw, best_sorted, best_grouped, per_comp_total)
    gap = best_overall - 2422
    print(f"  Best standard compressor: {best_overall} B")
    print(f"  Gap from Shannon floor:   {gap} B ({100*gap/2422:+.1f}%)")
    print(f"  Gap from current champ:   {best_overall - 4008:+d} B ({100*(best_overall-4008)/4008:+.1f}%)")


if __name__ == "__main__":
    main()
