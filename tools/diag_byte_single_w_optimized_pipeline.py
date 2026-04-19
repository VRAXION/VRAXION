"""Optimized full-pipeline: per-component K choice + unused-gen pruning + coef-tight encoding.

Target: push the 4212 B generator-all baseline as low as possible.

Strategy:
  1. W: K=8 generators, re-run staged exact (was K=16).
  2. b1/c19_c/c19_rho: K=4 generators.
  3. b2: K=4 generators.
  4. After staging: prune unused generators, tighten coef encoding (per-component).
  5. Final: compute exact deploy bytes.
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

FP16_CHAMP = Path("output/merger_single_w_fp16_all/final_fp16.json")
OUT_DIR = Path("output/merger_single_w_optimized")
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
    abs_vals = np.abs(values[np.abs(values) > 1e-8])
    if len(abs_vals) == 0:
        return np.array([])
    K = min(K, len(abs_vals))
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


def staged_exact(comp_name, state, comp_key, generators, data, coefs=range(1, 8)):
    arr_orig = state[comp_key].copy()
    orig_shape = arr_orig.shape
    arr = arr_orig.flatten()
    fits = [(i, *find_best_fit(float(arr[i]), generators, coefs)) for i in range(arr.size)]
    fits.sort(key=lambda x: x[2])  # by error

    print(f"\n  [{comp_name}] Staging {arr.size} cells (K={len(generators)}):", flush=True)
    n_accept = n_reject = 0
    encs = {}
    t0 = time.time()
    for rank, (i, best, err) in enumerate(fits):
        if best is None: continue
        approx, s, c, gi = best
        if err > 1.0 * max(1.0, np.abs(arr).max()):
            continue
        orig = arr[i]
        arr[i] = approx
        reshaped = arr.reshape(orig_shape)
        trial_state = dict(state)
        trial_state[comp_key] = reshaped
        model = build_model(trial_state["W"], trial_state["b1"], trial_state["b2"],
                             trial_state["c19_c"], trial_state["c19_rho"])
        if is_lossless(model, data):
            n_accept += 1
            encs[i] = (s, c, gi)
        else:
            arr[i] = orig
            n_reject += 1
        if (rank + 1) % 200 == 0:
            print(f"    rank {rank+1}/{len(fits)}: accept={n_accept} reject={n_reject} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{comp_name}] DONE: accept={n_accept}/{arr.size} ({100*n_accept/arr.size:.1f}%)  ({time.time()-t0:.0f}s)", flush=True)
    return arr.reshape(orig_shape), encs


def prune_and_size(arr, gens, encs, comp_name):
    """Drop unused gens, compact indices, choose min coef bits."""
    if not encs:
        return gens, encs, arr.size * 2  # all fallback fp16

    # Find used gens
    used_idxs = sorted(set(v[2] for v in encs.values()))
    new_idx_map = {old: new for new, old in enumerate(used_idxs)}
    compact_gens = np.array([gens[i] for i in used_idxs])
    compact_encs = {i: (s, c, new_idx_map[gi]) for i, (s, c, gi) in encs.items()}

    # Determine coef bits
    coefs_used = sorted(set(v[1] for v in compact_encs.values()))
    coef_bits = max(1, int(np.ceil(np.log2(max(coefs_used) + 1))))
    idx_bits = max(1, int(np.ceil(np.log2(max(len(compact_gens), 2)))))

    # Bit layout:
    #   tag: 1 bit (encoded / fallback)
    #   encoded: sign(1) + coef(coef_bits) + idx(idx_bits)
    #   fallback: fp16(16)
    enc_bits = 1 + 1 + coef_bits + idx_bits
    fb_bits = 1 + 16

    n_enc = len(compact_encs)
    n_fb = arr.size - n_enc
    gen_bytes = len(compact_gens) * 2
    total_bits = n_enc * enc_bits + n_fb * fb_bits
    total_bytes = gen_bytes + int(np.ceil(total_bits / 8))

    print(f"\n  [{comp_name}] optimized encoding:")
    print(f"    used gens: {len(compact_gens)}/{len(gens)} (pruned {len(gens)-len(compact_gens)})")
    print(f"    coef bits: {coef_bits}, idx bits: {idx_bits}")
    print(f"    encoded cells: {n_enc} @ {enc_bits} bits, fallback: {n_fb} @ {fb_bits} bits")
    print(f"    total: {total_bytes} B (vs fp16 {arr.size*2} B, delta {total_bytes - arr.size*2:+d} B)")
    return compact_gens, compact_encs, total_bytes


def main():
    print("=== OPTIMIZED FULL PIPELINE ===", flush=True)
    with open(FP16_CHAMP) as f:
        fp = json.load(f)
    W = np.array(fp["W_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b1 = np.array(fp["b1_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b2 = np.array(fp["b2_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_c = np.array(fp["c19_c_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_rho = np.array(fp["c19_rho_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)

    data = load_byte_pairs().to(DEVICE)
    state = {"W": W, "b1": b1, "b2": b2, "c19_c": c19_c, "c19_rho": c19_rho}

    # Build generators per-component
    K_config = {"W": 8, "b1": 4, "b2": 4, "c19_c": 4, "c19_rho": 4}
    gens_map = {}
    for k, K in K_config.items():
        gens_map[k] = kmeans_magnitudes(state[k].flatten(), K)
        print(f"{k}: K={K} generators: {gens_map[k]}")

    # Staging (W first, then others on top)
    print(f"\n{'='*50}\nSTAGED EXACT\n{'='*50}")

    # W first
    arr, encs_W = staged_exact("W", state, "W", gens_map["W"], data)
    state["W"] = arr

    encs_map = {"W": encs_W}
    for comp in ["b1", "b2", "c19_c", "c19_rho"]:
        arr, encs_c = staged_exact(comp, state, comp, gens_map[comp], data)
        state[comp] = arr
        encs_map[comp] = encs_c

    # Final verification
    model = build_model(state["W"], state["b1"], state["b2"], state["c19_c"], state["c19_rho"])
    assert is_lossless(model, data), "Lossless BROKEN at end!"
    print(f"\nFinal verification: 100% lossless OK")

    # Prune + size
    print(f"\n{'='*50}\nFINAL DEPLOY SIZE\n{'='*50}")
    total = 0
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        arr = state[comp].flatten() if comp == "W" else state[comp]
        pruned_gens, pruned_encs, b = prune_and_size(arr, gens_map[comp], encs_map[comp], comp)
        gens_map[comp] = pruned_gens
        encs_map[comp] = pruned_encs
        total += b

    print(f"\n{'='*50}")
    print(f"  GRAND TOTAL: {total} B ({total/1024:.2f} KB)")
    print(f"  vs fp16 champion (5734 B): {total - 5734:+d} B ({100*(total-5734)/5734:+.1f}%)")
    print(f"  vs prior gen-all (4212 B): {total - 4212:+d} B ({100*(total-4212)/4212:+.1f}%)")
    print(f"{'='*50}")

    # Save
    artifact = {
        "architecture": "generator-compressed optimized (per-component K, pruned gens, tight coef)",
        "K_config": K_config,
        "W_final": state["W"].tolist(),
        "b1_final": state["b1"].tolist(),
        "b2_final": state["b2"].tolist(),
        "c19_c_final": state["c19_c"].tolist(),
        "c19_rho_final": state["c19_rho"].tolist(),
        "generators": {k: v.tolist() for k, v in gens_map.items()},
        "encodings": {k: {str(i): list(v) for i, v in encs.items()} for k, encs in encs_map.items()},
        "deploy_total_bytes": total,
        "deploy_total_kb": total / 1024,
    }
    with open(OUT_DIR / "final_optimized.json", "w") as f:
        json.dump(artifact, f)
    print(f"\nSaved: {OUT_DIR / 'final_optimized.json'}")


if __name__ == "__main__":
    main()
