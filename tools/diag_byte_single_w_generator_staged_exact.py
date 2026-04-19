"""Staged exact generator compression.

Given a generator set (from GPT's static proto), convert cells one by one to
(sign, coef, gen) or (sign1, coef1, gen1, sign2, coef2, gen2) form. After each
conversion, do a FULL 65536-pair sign-match check. Accept if still lossless,
else roll back.

This produces a real, verified-lossless deploy artifact.
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

CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
GEN_PROTO = Path("output/merger_single_w_generator_value_proto.json")
OUT_DIR = Path("output/merger_single_w_generator_staged")
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


def find_best_fit(w_target, generators, coef_range=range(1, 8), sign=(-1, 1)):
    """Return best representation of w_target as (sign, coef, gen_idx) or None.

    Returns: (approx_value, sign, coef, gen_idx) or None
    """
    best_err = np.inf
    best = None
    for s in sign:
        for gi, g in enumerate(generators):
            for c in coef_range:
                approx = s * c * g
                err = abs(w_target - approx)
                if err < best_err:
                    best_err = err
                    best = (approx, s, c, gi)
    return best, best_err


def find_best_fit2(w_target, generators, coef_range=range(1, 8), sign=(-1, 1)):
    """Two-generator: w_target ≈ s1*c1*g1 + s2*c2*g2.  O(K^2 * coefs^2) — slow but fine."""
    best_err = np.inf
    best = None
    gens = list(generators)
    for gi, g1 in enumerate(gens):
        for gj, g2 in enumerate(gens):
            if gi >= gj: continue  # symmetric — skip duplicates
            for s1 in sign:
                for s2 in sign:
                    for c1 in coef_range:
                        for c2 in coef_range:
                            approx = s1*c1*g1 + s2*c2*g2
                            err = abs(w_target - approx)
                            if err < best_err:
                                best_err = err
                                best = (approx, s1, c1, gi, s2, c2, gj)
    return best, best_err


def main():
    print("=== STAGED EXACT GENERATOR COMPRESSION ===", flush=True)

    # Load champion
    with open(CHAMPION) as f:
        m = json.load(f)
    W_orig = np.array(m["W"], dtype=np.float32)
    b1 = np.array(m["b1"], dtype=np.float32)
    b2 = np.array(m["b2"], dtype=np.float32)
    c19_c = np.array(m["c19_c"], dtype=np.float32)
    c19_rho = np.array(m["c19_rho"], dtype=np.float32)

    # Load generators from GPT's proto
    with open(GEN_PROTO) as f:
        gp = json.load(f)
    generators = np.array(gp["best"]["generators"], dtype=np.float64)
    print(f"Loaded K={len(generators)} generators from {GEN_PROTO}")
    print(f"  generators: {generators}")

    data = load_byte_pairs().to(DEVICE)

    # Start with fp16-snapped W (our proven-lossless baseline with 1-ulp fix)
    # Actually, start from the fp16 champion directly
    fp16_champ = Path("output/merger_single_w_fp16_all/final_fp16.json")
    with open(fp16_champ) as f:
        fp = json.load(f)
    W_fp16 = np.array(fp["W_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b1_fp16 = np.array(fp["b1_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b2_fp16 = np.array(fp["b2_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_c_fp16 = np.array(fp["c19_c_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_rho_fp16 = np.array(fp["c19_rho_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)

    W_cur = W_fp16.copy()
    model = build_model(W_cur, b1_fp16, b2_fp16, c19_c_fp16, c19_rho_fp16)
    assert is_lossless(model, data), "fp16 baseline not lossless"
    print(f"FP16 baseline: lossless OK", flush=True)

    # Priority order: cells whose fp16 value is BEST approximable by a (coef * gen).
    # These should convert easily.
    n = W_orig.size
    shapes = (32, 81)
    fits = []
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            best, err = find_best_fit(W_fp16[i, j], generators)
            if best is not None:
                fits.append(((i, j), err, best))
    fits.sort(key=lambda x: x[1])  # smallest error first

    # Statistics of fit errors vs slack
    slack = np.load("output/merger_single_w_slack/slack.npy")
    print(f"\nFit quality stats (vs per-cell slack):")
    within_slack = 0
    for (i, j), err, _ in fits:
        if err < slack[i, j]:
            within_slack += 1
    print(f"  {within_slack}/{len(fits)} cells have single-gen fit within slack ({100*within_slack/len(fits):.1f}%)")

    # Staged conversion: try each candidate cell in order of best single-gen fit
    print(f"\n=== STAGE 1: single-gen conversions ===", flush=True)
    t0 = time.time()
    n_accept = 0
    n_reject = 0
    encodings = {}  # (i,j) -> ('1gen', s, c, gi) or ('2gen', ...)

    # Do full lossless check in batches for speed (torch tensor once)
    # Use is_lossless which does a full check on data
    for idx, ((i, j), err, (approx, s, c, gi)) in enumerate(fits):
        if err > 0.5:  # definitely won't fit — skip
            continue
        orig_val = W_cur[i, j]
        W_cur[i, j] = approx
        model = build_model(W_cur, b1_fp16, b2_fp16, c19_c_fp16, c19_rho_fp16)
        if is_lossless(model, data):
            n_accept += 1
            encodings[(i, j)] = ('1gen', s, c, gi)
        else:
            W_cur[i, j] = orig_val
            n_reject += 1
        if (idx + 1) % 100 == 0:
            print(f"  Tested {idx+1}/{len(fits)}, accept={n_accept}, reject={n_reject} ({time.time()-t0:.0f}s)", flush=True)

    t_stage1 = time.time() - t0
    print(f"\nStage 1 done: accept={n_accept}, reject={n_reject} ({t_stage1:.0f}s)", flush=True)
    model = build_model(W_cur, b1_fp16, b2_fp16, c19_c_fp16, c19_rho_fp16)
    assert is_lossless(model, data), "lossless broken after stage 1"

    # Count results
    n_1gen = sum(1 for v in encodings.values() if v[0] == '1gen')
    n_2gen = sum(1 for v in encodings.values() if v[0] == '2gen')
    n_fallback = n - len(encodings)
    print(f"\nCell distribution: {n_1gen} single-gen, {n_2gen} double-gen, {n_fallback} fallback (fp16)")

    # Deploy size estimate
    gen_bytes = len(generators) * 2  # fp16 generators
    # Per-cell encoding:
    #   tag: 2 bits (1gen, 2gen, fallback, reserved)
    #   1gen: sign (1 bit) + coef (3 bits, 1..7) + gen_idx (log2(K)=4 bits) = 8 bits
    #   2gen: sign*2 + coef*2 + idx*2 = 1+3+4+1+3+4 = 16 bits
    #   fallback: 16 bits fp16
    n_bits_1gen = 2 + 8
    n_bits_2gen = 2 + 16
    n_bits_fb = 2 + 16
    total_bits = n_1gen * n_bits_1gen + n_2gen * n_bits_2gen + n_fallback * n_bits_fb
    W_bytes = int(np.ceil(total_bits / 8)) + gen_bytes

    print(f"\nDeploy W estimate:")
    print(f"  Generators: {gen_bytes} B")
    print(f"  1gen cells: {n_1gen} × {n_bits_1gen} bits = {n_1gen*n_bits_1gen//8} B")
    print(f"  2gen cells: {n_2gen} × {n_bits_2gen} bits = {n_2gen*n_bits_2gen//8} B")
    print(f"  Fallback:   {n_fallback} × {n_bits_fb} bits = {n_fallback*n_bits_fb//8} B")
    print(f"  Total W:    {W_bytes} B (vs fp16 5184 B, delta {W_bytes-5184:+d} B)")
    print(f"\nDeploy total:")
    total_final = W_bytes + (81 + 32 + 81 + 81) * 2
    print(f"  W {W_bytes} + bias/c19 fp16 {(81+32+81+81)*2} = {total_final} B ({total_final/1024:.2f} KB)")
    print(f"  vs fp16 champion {5734} B ({5734/1024:.2f} KB)")

    # Save artifact
    artifact = {
        "architecture": "single-W generator-value staged exact",
        "H": 81, "in_dim": 32,
        "lossless": True,
        "n_1gen": n_1gen, "n_2gen": n_2gen, "n_fallback": n_fallback,
        "generators": generators.tolist(),
        "encodings": {f"{i},{j}": list(v) for (i, j), v in encodings.items()},
        "W_fp32_final": W_cur.tolist(),
        "b1_fp16": b1_fp16.astype(np.float16).view(np.uint16).tolist(),
        "b2_fp16": b2_fp16.astype(np.float16).view(np.uint16).tolist(),
        "c19_c_fp16": c19_c_fp16.astype(np.float16).view(np.uint16).tolist(),
        "c19_rho_fp16": c19_rho_fp16.astype(np.float16).view(np.uint16).tolist(),
        "deploy_W_bytes": W_bytes,
        "deploy_total_bytes": total_final,
    }
    with open(OUT_DIR / "final_staged.json", "w") as f:
        json.dump(artifact, f)
    print(f"\nSaved: {OUT_DIR / 'final_staged.json'}")


if __name__ == "__main__":
    main()
