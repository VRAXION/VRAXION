"""Deep dive: find every simplification opportunity in the generator-compressed model.

Tests:
  1. Per-component K sweep (K=2,4,8,16,32) — find min-deploy K per component.
  2. Cross-component generator overlap — can different components share generators?
  3. Generator hierarchy / recipe — can the generators themselves be expressed as
     multiples of a smaller root-generator set?
  4. Coefficient usage analysis — which coefs (1..7) are actually used?
     Can we shrink coef range?
  5. Unused generators — any K's never picked? Drop them.
  6. Cell redundancy — do multiple cells map to same (sign, coef, gen)? Could dedup.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import (
    SingleWMirror, load_byte_pairs, metrics, DEVICE,
)

STAGED = Path("output/merger_single_w_generator_all/final_all_components.json")
OUT_DIR = Path("output/merger_single_w_deep_dive")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_lossless(model, data):
    with torch.no_grad():
        y = model(data)
        return (torch.sign(y) == torch.sign(data)).all(dim=1).all().item()


def build_model(W, b1, b2, c19_c, c19_rho):
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(b1, dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(b2, dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=DEVICE))
    return model


def main():
    with open(STAGED) as f:
        m = json.load(f)

    W = np.array(m["W_final"], dtype=np.float32)
    b1 = np.array(m["b1_final"], dtype=np.float32)
    b2 = np.array(m["b2_final"], dtype=np.float32)
    c19_c = np.array(m["c19_c_final"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho_final"], dtype=np.float32)

    comps = {
        "W": (W.flatten(), np.array(None)),  # W doesn't have gens in this artifact (stage 1 from earlier)
        "b1": (b1, np.array(m["b1_generators"])),
        "b2": (b2, np.array(m["b2_generators"])),
        "c19_c": (c19_c, np.array(m["c19_c_generators"])),
        "c19_rho": (c19_rho, np.array(m["c19_rho_generators"])),
    }
    encs = {
        "b1": m["b1_encodings"],
        "b2": m["b2_encodings"],
        "c19_c": m["c19_c_encodings"],
        "c19_rho": m["c19_rho_encodings"],
    }

    # Load W generators from original staged artifact
    W_staged = Path("output/merger_single_w_generator_staged/final_staged.json")
    with open(W_staged) as f:
        ws = json.load(f)
    W_generators = np.array(ws["generators"])
    comps["W"] = (W.flatten(), W_generators)

    print("=" * 70)
    print("DEEP DIVE — generator compression inventory")
    print("=" * 70)

    # ---- TEST 1: Generator overlap across components ----
    print("\n" + "="*60)
    print("1. GENERATOR OVERLAP (cross-component sharing potential)")
    print("="*60)
    tol = 0.0005
    all_gens = {}
    for name, (_, gens) in comps.items():
        all_gens[name] = gens
        print(f"\n  {name}: {len(gens)} generators, range [{gens.min():.6f}, {gens.max():.6f}]")

    # Check for overlap between component pairs
    print("\n  Pairwise overlaps (tolerance=0.0005):")
    for a in comps:
        for b in comps:
            if a >= b: continue
            ga, gb = all_gens[a], all_gens[b]
            shared = 0
            for x in ga:
                if np.any(np.abs(gb - x) < tol):
                    shared += 1
            print(f"    {a:10s} vs {b:10s}: {shared} shared ({100*shared/min(len(ga),len(gb)):.0f}%)")

    # ---- TEST 2: Full pooled generator set ----
    print("\n" + "="*60)
    print("2. POOLED GENERATORS (single shared dictionary)")
    print("="*60)
    pooled = np.concatenate([gens for _, gens in comps.values()])
    pooled_unique = np.unique(np.round(pooled / tol).astype(int)) * tol
    print(f"  All generators: {len(pooled)} total, {len(pooled_unique)} unique (tol=0.0005)")
    print(f"  If shared: save {len(pooled) - len(pooled_unique)} * 2 = {(len(pooled) - len(pooled_unique)) * 2} B on generator storage")

    # ---- TEST 3: Coefficient usage ----
    print("\n" + "="*60)
    print("3. COEFFICIENT USAGE")
    print("="*60)
    for name, e in encs.items():
        if not e: continue
        coefs_used = [abs(v[1]) for v in e.values()]
        signs_used = [v[0] for v in e.values()]
        idxs_used = [v[2] for v in e.values()]
        unique_coefs = sorted(set(coefs_used))
        unique_idxs = sorted(set(idxs_used))
        pos = signs_used.count(1)
        neg = signs_used.count(-1)
        print(f"\n  {name}: {len(e)} cells encoded")
        print(f"    coefs used: {unique_coefs} (max {max(coefs_used)})")
        print(f"    gen idxs used: {unique_idxs} (out of 16)")
        print(f"    signs: +{pos} / -{neg}")

        # If only coefs 1-3 used, we could use 2 bits instead of 3
        if max(coefs_used) <= 3:
            print(f"    -> could use 2 bits for coef instead of 3: save {len(e)} bits")

    # ---- TEST 4: Generator recipe (hierarchy) ----
    print("\n" + "="*60)
    print("4. GENERATOR RECIPE — can generators be expressed as multiples of fewer roots?")
    print("="*60)
    for name, (_, gens) in comps.items():
        if len(gens) == 0: continue
        # Find smallest gen (the "root")
        g_sorted = np.sort(gens)
        root = g_sorted[0]
        # Check: g_sorted[i] ≈ k * root for small integer k?
        ratios = g_sorted / root
        int_ratios = np.round(ratios)
        residuals = np.abs(ratios - int_ratios)
        print(f"\n  {name}: root = {root:.6f}")
        print(f"    integer ratios (round): {int_ratios.astype(int)}")
        print(f"    residuals (|ratio - int|): max={residuals.max():.4f}, mean={residuals.mean():.4f}")

    # ---- TEST 5: Unused generators ----
    print("\n" + "="*60)
    print("5. UNUSED GENERATORS")
    print("="*60)
    for name, (_, gens) in comps.items():
        if name not in encs or not encs[name]: continue
        used = set(v[2] for v in encs[name].values())
        unused = [i for i in range(len(gens)) if i not in used]
        print(f"  {name}: {len(unused)} unused out of {len(gens)} ({[gens[i] for i in unused]})")

    # ---- TEST 6: K sweep per component ----
    print("\n" + "="*60)
    print("6. K SWEEP per component (theoretical minimum deploy)")
    print("="*60)
    # For each component, try different K values and estimate deploy size
    for name, (arr, _) in comps.items():
        print(f"\n  {name} ({len(arr)} cells):")
        # Using simplified bits-per-cell estimate (not rerunning full pipeline)
        # Per cell: 1 bit tag + (3 sign + 3 coef + log2(K) idx) = 7 + ceil(log2(K)) bits for encoded
        #           or 1 bit tag + 16 fp16 = 17 bits for fallback
        # We don't know exact acceptance without rerunning — use current as estimate
        fp16_bytes = len(arr) * 2
        print(f"    fp16: {fp16_bytes} B")
        for K in [2, 4, 8, 16]:
            if K > len(arr) // 2:
                continue  # K too large
            idx_bits = max(1, int(np.ceil(np.log2(K))))
            encoded_bits = 1 + 1 + 3 + idx_bits  # tag + sign + coef + idx
            gen_bytes = K * 2
            # Assume 80% acceptance for small K, 95% for large (rough)
            pct_accept = 0.60 if K == 2 else 0.75 if K == 4 else 0.85 if K == 8 else 0.95
            n_enc = int(len(arr) * pct_accept)
            n_fb = len(arr) - n_enc
            est_bits = n_enc * encoded_bits + n_fb * 17
            est_bytes = gen_bytes + int(np.ceil(est_bits / 8))
            delta = est_bytes - fp16_bytes
            print(f"    K={K:2d}: ~{est_bytes} B (delta {delta:+d} B)")


if __name__ == "__main__":
    main()
