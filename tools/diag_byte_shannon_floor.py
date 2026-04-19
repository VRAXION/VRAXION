"""Shannon entropy floor analysis for the hybrid-K champion (4008 B).

Questions answered:
  1. What is the absolute minimum bytes if we Huffman-code the raw fp16 values?
  2. What is the minimum if we Huffman-code the (sign, coef, gen_idx) triplets?
  3. How much overhead does the current fixed-width encoding carry?
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np

ART = Path("output/merger_single_w_hybrid_k/final_hybrid.json")


def shannon(counts: Counter) -> float:
    """Shannon H in bits: -sum p_i * log2(p_i)."""
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-30)))


def main():
    with open(ART) as f:
        m = json.load(f)

    W = np.array(m["W_final"], dtype=np.float32).flatten()
    b1 = np.array(m["b1_final"], dtype=np.float32)
    b2 = np.array(m["b2_final"], dtype=np.float32)
    c19_c = np.array(m["c19_c_final"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho_final"], dtype=np.float32)
    all_vals = np.concatenate([W, b1, b2, c19_c, c19_rho])
    fp16_buckets = all_vals.astype(np.float16).view(np.uint16)

    print("=" * 70)
    print(f"SHANNON ENTROPY FLOOR — hybrid-K champion artifact")
    print("=" * 70)
    print(f"Total cells: {len(all_vals)}")
    print(f"Unique fp16 buckets: {len(set(fp16_buckets.tolist()))}")
    print(f"FP16 naive: {len(all_vals)*2} B")
    print(f"Current deploy: 4008 B (3.91 KB)")

    # -------------------------------------------------------------------
    # [1] Shannon on the raw fp16 value distribution (ignores any encoding).
    #     This is the *absolute info-theoretic floor* at fp16 resolution.
    # -------------------------------------------------------------------
    counts_raw = Counter(fp16_buckets.tolist())
    H_raw = shannon(counts_raw)
    floor_raw_bits = len(all_vals) * H_raw
    floor_raw_B = int(np.ceil(floor_raw_bits / 8))
    print(f"\n{'-'*70}")
    print(f"[1] RAW FP16 VALUE DISTRIBUTION")
    print(f"{'-'*70}")
    print(f"  Unique values: {len(counts_raw)} / {len(all_vals)} cells")
    most_common = counts_raw.most_common(5)
    print(f"  Top 5 most repeated fp16 buckets:")
    for bucket, cnt in most_common:
        val = np.array([bucket], dtype=np.uint16).view(np.float16)[0]
        print(f"    {float(val):+.6f}  -> {cnt} cells ({100*cnt/len(all_vals):.1f}%)")
    print(f"  Shannon H: {H_raw:.3f} bits/cell")
    print(f"  Huffman floor: {floor_raw_B} B (vs fp16 {len(all_vals)*2} B)")
    print(f"  Savings vs fp16: {len(all_vals)*2 - floor_raw_B} B ({100*(1-floor_raw_B/(len(all_vals)*2)):.1f}%)")

    # -------------------------------------------------------------------
    # [2] Shannon on encoded symbols (sign, coef, gen_idx) per component.
    #     This tells us: can Huffman squeeze the current hybrid encoding tighter?
    # -------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"[2] ENCODED SYMBOL DISTRIBUTION (sign, coef, gen_idx)")
    print(f"{'-'*70}")
    total_encoded_bits_current = 0
    total_encoded_bits_shannon = 0
    total_encoded_cells = 0
    total_fallback_cells = 0
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        encs = m["encodings"][comp]
        if not encs:
            continue
        # Each triplet (sign, coef, gen_idx) is one symbol
        symbols = [tuple(v) for v in encs.values()]
        counts = Counter(symbols)
        H = shannon(counts)
        n = len(symbols)
        K = len(m["generators"][comp])
        idx_bits = max(1, int(np.ceil(np.log2(max(K, 2)))))
        coefs = [abs(s[1]) for s in symbols]
        coef_bits = max(1, int(np.ceil(np.log2(max(coefs) + 1))))
        current_per = 1 + 1 + coef_bits + idx_bits  # tag + sign + coef + idx
        total_encoded_bits_current += n * current_per
        total_encoded_bits_shannon += n * H
        total_encoded_cells += n
        arr_size = {"W": 2592, "b1": 81, "b2": 32, "c19_c": 81, "c19_rho": 81}[comp]
        total_fallback_cells += arr_size - n
        print(f"  [{comp:8s}] n={n:4d} unique={len(counts):3d} | H={H:5.3f} bits | current={current_per} bits | K={K} coef_bits={coef_bits} idx_bits={idx_bits}")

    # -------------------------------------------------------------------
    # [3] Grand total with hybrid approach
    # -------------------------------------------------------------------
    fallback_bits = total_fallback_cells * 17  # 1 tag + 16 fp16
    # Generator storage (fp16)
    gen_bytes_total = sum(len(m["generators"][c]) * 2 for c in m["generators"])
    gen_bits = gen_bytes_total * 8

    current_total_bits = total_encoded_bits_current + fallback_bits + gen_bits
    shannon_total_bits = total_encoded_bits_shannon + fallback_bits + gen_bits
    # Plus shannon on fallback fp16 values (those also repeat some)
    # Collect fallback values for each component
    fallback_vals = []
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        arr = {"W": W, "b1": b1, "b2": b2, "c19_c": c19_c, "c19_rho": c19_rho}[comp]
        encs = m["encodings"][comp]
        enc_idxs = set(int(k) for k in encs.keys())
        for i in range(arr.size):
            if i not in enc_idxs:
                fallback_vals.append(float(arr.flatten()[i]))
    fb_fp16 = np.array(fallback_vals, dtype=np.float16).view(np.uint16)
    fb_counts = Counter(fb_fp16.tolist())
    H_fb = shannon(fb_counts) if len(fb_counts) > 1 else 0.0
    shannon_fb_bits = len(fb_fp16) * H_fb + len(fb_fp16) * 1  # +1 tag bit each
    shannon_hybrid_bits = total_encoded_bits_shannon + shannon_fb_bits + gen_bits

    current_total_B = int(np.ceil(current_total_bits / 8))
    shannon_hybrid_B = int(np.ceil(shannon_hybrid_bits / 8))

    print(f"\n{'-'*70}")
    print(f"[3] FALLBACK VALUE DISTRIBUTION ({len(fallback_vals)} fp16 cells)")
    print(f"{'-'*70}")
    print(f"  Unique fp16 buckets among fallbacks: {len(fb_counts)}")
    print(f"  Shannon H of fallback values: {H_fb:.3f} bits/cell")
    print(f"  Current: 16 bits/cell × {len(fb_fp16)} = {len(fb_fp16)*2} B")
    print(f"  Huffman floor: {int(np.ceil(len(fb_fp16)*H_fb/8))} B (saves {len(fb_fp16)*2 - int(np.ceil(len(fb_fp16)*H_fb/8))} B)")

    print(f"\n{'='*70}")
    print(f"SUMMARY — where is the theoretical floor?")
    print(f"{'='*70}")
    print(f"  Naive fp16 (lookup):            {len(all_vals)*2} B")
    print(f"  Current hybrid-K champion:      4008 B (3.91 KB)")
    print(f"  Shannon on encoded + fallback:  {shannon_hybrid_B} B  <- optimal Huffman on current topology")
    print(f"  Shannon on raw fp16 values:     {floor_raw_B} B  <- absolute floor (ignore topology)")
    print()
    print(f"  Gap from current to Huffman-on-topology: {4008 - shannon_hybrid_B} B ({100*(4008-shannon_hybrid_B)/4008:+.1f}%)")
    print(f"  Gap from current to raw Shannon:         {4008 - floor_raw_B} B ({100*(4008-floor_raw_B)/4008:+.1f}%)")
    print()
    print(f"  INTERPRETATION:")
    if shannon_hybrid_B < 4008:
        pct = 100 * (4008 - shannon_hybrid_B) / 4008
        print(f"    -> Huffman-coding the current encoding would save ~{4008-shannon_hybrid_B} B ({pct:.1f}%)")
    if floor_raw_B < 4008:
        pct = 100 * (4008 - floor_raw_B) / 4008
        print(f"    -> Optimal Huffman on raw fp16 values: ~{4008-floor_raw_B} B below current ({pct:.1f}%)")
        print(f"    -> But this ignores that we also need to *run* the model, so structure matters")


if __name__ == "__main__":
    main()
