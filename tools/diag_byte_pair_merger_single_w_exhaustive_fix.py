"""Exhaustive single-cell tweak to fix the last bad pair(s).

Loads a near-lossless single-W model (e.g. 99.9985% with 1 bad pair)
and for each cell tries small perturbations to see if any one tweak
pushes the model to 100% lossless.
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

# Will switch to final_push output if available
SOURCE_CANDIDATES = [
    "output/merger_single_w_final_push/best_model.pt",
    "output/merger_single_w_continue/best_model.pt",
]
OUT_DIR = Path("output/merger_single_w_exhaustive_fix")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    source = None
    for c in SOURCE_CANDIDATES:
        if Path(c).exists():
            source = c
            break
    if source is None:
        print("No source model found."); return
    print(f"=== EXHAUSTIVE CELL TWEAK FIX ===")
    print(f"Source: {source}")

    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    state = torch.load(source, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    data = load_byte_pairs().to(DEVICE)

    ll0, pd0, bp0 = metrics(model, data)
    print(f"Loaded: ll={ll0:.6f}% bad={bp0}")

    if bp0 == 0:
        print("Already lossless! Nothing to fix.")
        return

    if bp0 > 20:
        print(f"Too many bad pairs ({bp0}) for single-cell exhaustive. Skipping.")
        return

    # Get baseline W and compute perturbation deltas (relative to abs mean)
    W_orig = model.W.data.clone()
    w_abs_mean = W_orig.abs().mean().item()
    print(f"W shape: {W_orig.shape}, abs_mean={w_abs_mean:.6f}")

    # Deltas: proportional steps of various magnitudes
    delta_multipliers = [-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005,
                          0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    deltas_abs = [d * w_abs_mean for d in delta_multipliers]
    print(f"Testing {len(deltas_abs)} deltas per cell; total evals: {W_orig.numel() * len(deltas_abs)}")

    t0 = time.time()
    best_ll = ll0
    best_cell = None
    best_delta = None
    n_improvements = 0

    with torch.no_grad():
        for i in range(W_orig.shape[0]):
            for j in range(W_orig.shape[1]):
                orig = W_orig[i, j].item()
                for delta in deltas_abs:
                    model.W.data[i, j] = orig + delta
                    ll, pd, bp = metrics(model, data)
                    if ll > best_ll:
                        best_ll = ll
                        best_cell = (i, j)
                        best_delta = delta
                        n_improvements += 1
                        print(f"  [improvement #{n_improvements}] W[{i},{j}] += {delta:+.6f} -> ll={ll:.6f}%, bad={bp}", flush=True)
                    # restore
                    model.W.data[i, j] = orig
            if (i + 1) % 8 == 0:
                elapsed = time.time() - t0
                print(f"  progress: row {i+1}/{W_orig.shape[0]}, best_ll={best_ll:.6f}%, time={elapsed:.0f}s", flush=True)

    t_total = time.time() - t0
    print(f"\n=== SCAN DONE ===  time={t_total:.0f}s, improvements tried={n_improvements}")
    print(f"Best: ll={best_ll:.6f}% via W[{best_cell}] += {best_delta}")

    # Apply best tweak if improved
    if best_cell is not None and best_ll > ll0:
        i, j = best_cell
        with torch.no_grad():
            model.W.data[i, j] = W_orig[i, j].item() + best_delta
        ll_check, pd, bp = metrics(model, data)
        print(f"Applied: ll={ll_check:.6f}%, bad={bp}")

        torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

        with open(OUT_DIR / "final_model.json", "w") as f:
            json.dump({
                "architecture": "single-W + exhaustive 1-cell tweak",
                "H": 81, "in_dim": 32, "out_dim": 32,
                "starting_lossless": ll0,
                "starting_bad_pairs": bp0,
                "final_lossless": ll_check,
                "final_bad_pairs": bp,
                "tweak_cell": list(best_cell),
                "tweak_delta": best_delta,
                "weights_count": 2592,
                "W": model.W.data.cpu().numpy().tolist(),
                "b1": model.b1.data.cpu().numpy().tolist(),
                "b2": model.b2.data.cpu().numpy().tolist(),
                "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
                "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
            }, f)
        print(f"Saved: {OUT_DIR / 'final_model.json'}")
    else:
        print("No single-cell tweak improved lossless.")


if __name__ == "__main__":
    main()
