"""Extend staged exact generator compression to ALL 2867 values (W + b1 + b2 + c19_c + c19_rho).

For each component:
  1. Compute its slack map (or use existing).
  2. Build a generator set sized for its range.
  3. Staged exact: try each cell as (sign, coef, gen_idx); verify with FULL lossless check; roll back if broken.
  4. Report accepted vs fallback counts.

Start from the W-optimized state (output/merger_single_w_generator_staged/final_staged.json).
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

START_ARTIFACT = Path("output/merger_single_w_generator_staged/final_staged.json")
FP16_CHAMP = Path("output/merger_single_w_fp16_all/final_fp16.json")
OUT_DIR = Path("output/merger_single_w_generator_all")
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


def kmeans_magnitudes(values, K, n_iter=200):
    """K-means on absolute values (positive generators)."""
    abs_vals = np.abs(values[values != 0])
    if len(abs_vals) == 0:
        return np.array([])
    # Init at quantiles
    qs = np.linspace(0, 1, K + 2)[1:-1]
    centers = np.quantile(abs_vals, qs).astype(np.float64)
    for _ in range(n_iter):
        d = np.abs(abs_vals[:, None] - centers[None, :])
        idx = np.argmin(d, axis=1)
        new = centers.copy()
        for j in range(K):
            m = idx == j
            if m.any(): new[j] = abs_vals[m].mean()
        if np.allclose(new, centers, atol=1e-12): break
        centers = new
    return np.sort(np.unique(np.maximum(centers, 1e-12)))


def find_best_fit(target, generators, coefs=range(1, 8), signs=(-1, 1)):
    """Best (sign * coef * generator) approximation to target."""
    best_err = np.inf
    best = None
    for s in signs:
        for gi, g in enumerate(generators):
            for c in coefs:
                approx = s * c * g
                err = abs(target - approx)
                if err < best_err:
                    best_err = err
                    best = (approx, s, c, gi)
    return best, best_err


def staged_convert_component(component_name, arr, slack_arr, generators,
                              data, build_fn, arr_idx_in_model):
    """Try to convert each value in arr to (sign, coef, gen_idx).
    Full lossless check after each. Returns (new_arr, encodings_dict).
    """
    print(f"\n=== Staging: {component_name} ({arr.size} values, K={len(generators)} generators) ===", flush=True)
    new_arr = arr.copy()
    encodings = {}

    # Sort indices by best single-gen fit error (smallest first)
    fits = []
    for i in range(arr.size):
        best, err = find_best_fit(arr[i], generators)
        fits.append((i, err, best))
    fits.sort(key=lambda x: x[1])

    t0 = time.time()
    n_accept = 0
    n_reject = 0
    for rank, (i, err, (approx, s, c, gi)) in enumerate(fits):
        if err > 0.5 * max(1.0, np.abs(arr).max()):  # too far off
            continue
        orig = new_arr[i]
        new_arr[i] = approx
        model = build_fn(new_arr, arr_idx_in_model)
        if is_lossless(model, data):
            n_accept += 1
            encodings[i] = (s, c, gi)
        else:
            new_arr[i] = orig
            n_reject += 1
        if (rank + 1) % 50 == 0:
            print(f"  [{component_name}] rank {rank+1}/{len(fits)}: accept={n_accept} reject={n_reject} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  [{component_name}] DONE: accept={n_accept}/{arr.size} ({100*n_accept/arr.size:.1f}%), reject={n_reject}", flush=True)
    return new_arr, encodings


def main():
    print("=== STAGED EXACT GENERATOR ON ALL COMPONENTS ===", flush=True)

    # Start from W-optimized state
    with open(START_ARTIFACT) as f:
        sa = json.load(f)
    W = np.array(sa["W_fp32_final"], dtype=np.float32)
    b1_fp16 = np.array(sa["b1_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b2_fp16 = np.array(sa["b2_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_c_fp16 = np.array(sa["c19_c_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_rho_fp16 = np.array(sa["c19_rho_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    W_generators = np.array(sa["generators"], dtype=np.float64)

    data = load_byte_pairs().to(DEVICE)

    # Verify starting state is lossless
    model = build_model(W, b1_fp16, b2_fp16, c19_c_fp16, c19_rho_fp16)
    assert is_lossless(model, data), "Starting state not lossless!"
    print(f"Starting lossless OK", flush=True)
    print(f"W generators (K={len(W_generators)}): already applied ({sa.get('n_1gen', 'N/A')} cells)")

    # Now build generator sets for each remaining component
    b1_gens = kmeans_magnitudes(b1_fp16, K=16)
    b2_gens = kmeans_magnitudes(b2_fp16, K=16)
    c19_c_gens = kmeans_magnitudes(c19_c_fp16, K=16)
    c19_rho_gens = kmeans_magnitudes(c19_rho_fp16, K=16)
    print(f"b1 generators: {b1_gens}")
    print(f"b2 generators: {b2_gens}")
    print(f"c19_c generators: {c19_c_gens}")
    print(f"c19_rho generators: {c19_rho_gens}")

    # Stage conversions — need proper build_fn for each component
    state = {"W": W, "b1": b1_fp16.copy(), "b2": b2_fp16.copy(),
             "c19_c": c19_c_fp16.copy(), "c19_rho": c19_rho_fp16.copy()}
    all_encodings = {"b1": {}, "b2": {}, "c19_c": {}, "c19_rho": {}}

    for comp_name, gens in [("b1", b1_gens), ("b2", b2_gens),
                              ("c19_c", c19_c_gens), ("c19_rho", c19_rho_gens)]:
        arr = state[comp_name].copy()
        # For slack, we don't have per-cell maps for these, use dummy big slack — the full lossless check is authoritative
        slack_arr = np.ones_like(arr) * 1.0  # just a placeholder

        def build_fn(candidate_arr, comp_name=comp_name):
            new_state = dict(state)
            new_state[comp_name] = candidate_arr
            return build_model(new_state["W"], new_state["b1"], new_state["b2"],
                                new_state["c19_c"], new_state["c19_rho"])

        new_arr, encs = staged_convert_component(
            comp_name, arr, slack_arr, gens, data, build_fn, comp_name,
        )
        state[comp_name] = new_arr
        all_encodings[comp_name] = encs

    # Final size
    print(f"\n=== FINAL DEPLOY SIZE ===", flush=True)
    # W: from starting artifact
    W_bytes = sa["deploy_W_bytes"]
    print(f"  W: {W_bytes} B (from stage 1)")

    # For each other component, count generator entries + encoded cells + fallback fp16
    for comp_name, gens in [("b1", b1_gens), ("b2", b2_gens),
                              ("c19_c", c19_c_gens), ("c19_rho", c19_rho_gens)]:
        arr = state[comp_name]
        encs = all_encodings[comp_name]
        n_1gen = len(encs)
        n_fb = arr.size - n_1gen
        gen_b = len(gens) * 2  # fp16 generators
        # Per cell: 1-bit tag + content (8 bits for 1gen, 16 bits for fallback)
        cells_bits = n_1gen * 9 + n_fb * 17
        comp_bytes = gen_b + int(np.ceil(cells_bits / 8))
        fp16_b = arr.size * 2
        print(f"  {comp_name:10s}: gens {gen_b} B + cells {int(np.ceil(cells_bits/8))} B = {comp_bytes} B  (vs fp16 {fp16_b}, n_1gen={n_1gen}/{arr.size}={100*n_1gen/arr.size:.0f}%)")

    # Grand total
    b1_bytes = 16*2 + int(np.ceil((len(all_encodings['b1'])*9 + (81-len(all_encodings['b1']))*17)/8))
    b2_bytes = 16*2 + int(np.ceil((len(all_encodings['b2'])*9 + (32-len(all_encodings['b2']))*17)/8))
    c19_c_bytes = 16*2 + int(np.ceil((len(all_encodings['c19_c'])*9 + (81-len(all_encodings['c19_c']))*17)/8))
    c19_rho_bytes = 16*2 + int(np.ceil((len(all_encodings['c19_rho'])*9 + (81-len(all_encodings['c19_rho']))*17)/8))

    total = W_bytes + b1_bytes + b2_bytes + c19_c_bytes + c19_rho_bytes
    print(f"\n  TOTAL: {total} B ({total/1024:.2f} KB)")
    print(f"  vs fp16 champion: 5734 B (delta {total - 5734:+d} B)")

    # Save
    artifact = {
        "architecture": "generator-staged all components",
        "W_bytes": W_bytes,
        "W_final": state["W"].tolist(),
        "b1_final": state["b1"].tolist(),
        "b2_final": state["b2"].tolist(),
        "c19_c_final": state["c19_c"].tolist(),
        "c19_rho_final": state["c19_rho"].tolist(),
        "b1_generators": b1_gens.tolist(),
        "b2_generators": b2_gens.tolist(),
        "c19_c_generators": c19_c_gens.tolist(),
        "c19_rho_generators": c19_rho_gens.tolist(),
        "b1_encodings": {str(k): list(v) for k, v in all_encodings["b1"].items()},
        "b2_encodings": {str(k): list(v) for k, v in all_encodings["b2"].items()},
        "c19_c_encodings": {str(k): list(v) for k, v in all_encodings["c19_c"].items()},
        "c19_rho_encodings": {str(k): list(v) for k, v in all_encodings["c19_rho"].items()},
        "deploy_total_bytes": total,
    }
    with open(OUT_DIR / "final_all_components.json", "w") as f:
        json.dump(artifact, f)
    print(f"\nSaved: {OUT_DIR / 'final_all_components.json'}")


if __name__ == "__main__":
    main()
