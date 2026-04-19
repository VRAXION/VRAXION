"""Hybrid K per-component — optimized mix.

Use K=16 for W, b1, c19_c; K=4 for b2, c19_rho.
Based on deep-dive finding that some components benefit from smaller K
(c19_rho concentrates tightly, b2 has few values so big-K overhead dominates).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_single_w_optimized_pipeline import (
    is_lossless, build_model, kmeans_magnitudes, find_best_fit,
    staged_exact, prune_and_size, DEVICE,
)
from diag_byte_pair_merger_single_w_mirror import load_byte_pairs

FP16_CHAMP = Path("output/merger_single_w_fp16_all/final_fp16.json")
OUT_DIR = Path("output/merger_single_w_hybrid_k")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== HYBRID K OPTIMIZATION ===", flush=True)
    with open(FP16_CHAMP) as f:
        fp = json.load(f)
    W = np.array(fp["W_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b1 = np.array(fp["b1_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    b2 = np.array(fp["b2_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_c = np.array(fp["c19_c_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)
    c19_rho = np.array(fp["c19_rho_fp16"], dtype=np.uint16).view(np.float16).astype(np.float32)

    data = load_byte_pairs().to(DEVICE)
    state = {"W": W, "b1": b1, "b2": b2, "c19_c": c19_c, "c19_rho": c19_rho}

    K_config = {"W": 16, "b1": 16, "b2": 4, "c19_c": 16, "c19_rho": 4}
    gens_map = {}
    for k, K in K_config.items():
        gens_map[k] = kmeans_magnitudes(state[k].flatten(), K)
        print(f"{k} (K={K}): {gens_map[k]}")

    # Staging (W first)
    encs_map = {}
    order = ["W", "c19_rho", "c19_c", "b1", "b2"]  # rho first to lock its best values
    for comp in order:
        arr, encs = staged_exact(comp, state, comp, gens_map[comp], data)
        state[comp] = arr
        encs_map[comp] = encs

    model = build_model(state["W"], state["b1"], state["b2"], state["c19_c"], state["c19_rho"])
    assert is_lossless(model, data), "BROKEN"
    print(f"\nFinal verification: 100% lossless OK")

    print(f"\n{'='*50}\nFINAL DEPLOY SIZE (HYBRID K)\n{'='*50}")
    total = 0
    for comp in ["W", "b1", "b2", "c19_c", "c19_rho"]:
        arr = state[comp].flatten() if comp == "W" else state[comp]
        pruned_gens, pruned_encs, b = prune_and_size(arr, gens_map[comp], encs_map[comp], comp)
        gens_map[comp] = pruned_gens
        encs_map[comp] = pruned_encs
        total += b

    print(f"\n{'='*50}")
    print(f"  GRAND TOTAL: {total} B ({total/1024:.2f} KB)")
    print(f"  vs fp16 (5734): {total - 5734:+d} B")
    print(f"  vs K=16 all-components (4212): {total - 4212:+d} B")
    print(f"  vs K=8 optimized (4448): {total - 4448:+d} B")

    with open(OUT_DIR / "final_hybrid.json", "w") as f:
        json.dump({
            "K_config": K_config,
            "deploy_total_bytes": total,
            "W_final": state["W"].tolist(),
            "b1_final": state["b1"].tolist(),
            "b2_final": state["b2"].tolist(),
            "c19_c_final": state["c19_c"].tolist(),
            "c19_rho_final": state["c19_rho"].tolist(),
            "generators": {k: v.tolist() for k, v in gens_map.items()},
            "encodings": {k: {str(i): list(v) for i, v in encs.items()} for k, encs in encs_map.items()},
        }, f)


if __name__ == "__main__":
    main()
