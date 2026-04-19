"""Exhaustive fp16-grid search for the 2 cells that break fp16 lossless.

Strategy:
  1. Fp16-cast everything.
  2. Identify bad cells (the 2 that break).
  3. For each bad cell, try neighboring fp16 values (+/- 1..N ulps).
  4. If any single-cell fp16 value fixes the bad pairs, accept.
  5. If not, try pairs.
  6. If still not, try all 2592 cells (broader search) — in fp16 grid.
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
OUT_DIR = Path("output/merger_single_w_fp16_all")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cast_fp16(arr):
    """Cast to fp16 and back to fp32."""
    return torch.tensor(arr, dtype=torch.float32).to(torch.float16).to(torch.float32).numpy()


def fp16_neighbors(val, n_ulps=8):
    """Return fp16 values at val + k*ulp for k in -n..n."""
    t = torch.tensor([val], dtype=torch.float16)
    bits = t.view(torch.int16)
    results = []
    for k in range(-n_ulps, n_ulps + 1):
        nb = (bits + k).view(torch.float16)
        results.append(nb.to(torch.float32).item())
    return sorted(set(results))


def build_model(W, b1, b2, c19_c, c19_rho):
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(b1, dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(b2, dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=DEVICE))
    return model


def find_bad_cells(W, b1, b2, c19_c, c19_rho, data):
    """Identify which W cells are involved in bad pairs after fp16 snap."""
    model = build_model(W, b1, b2, c19_c, c19_rho)
    with torch.no_grad():
        y = model(data)
        sign_match = torch.sign(y) == torch.sign(data)
        bad_pair_mask = ~sign_match.all(dim=1)
        bad_pair_indices = bad_pair_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    return bad_pair_indices


def main():
    print(f"=== FP16 ALL-IN SEARCH (eliminate escape cells) ===")
    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W32 = np.array(m["W"], dtype=np.float32)
    b1_32 = np.array(m["b1"], dtype=np.float32)
    b2_32 = np.array(m["b2"], dtype=np.float32)
    c19_c_32 = np.array(m["c19_c"], dtype=np.float32)
    c19_rho_32 = np.array(m["c19_rho"], dtype=np.float32)
    data = load_byte_pairs().to(DEVICE)

    # Cast all to fp16
    W16 = cast_fp16(W32)
    b1_16 = cast_fp16(b1_32)
    b2_16 = cast_fp16(b2_32)
    c19_c_16 = cast_fp16(c19_c_32)
    c19_rho_16 = cast_fp16(c19_rho_32)

    model = build_model(W16, b1_16, b2_16, c19_c_16, c19_rho_16)
    ll, _, bp = metrics(model, data)
    print(f"Starting fp16: ll={ll:.6f}% bad={bp}")

    if bp == 0:
        print("Already lossless!")
        return

    bad_pair_idx = find_bad_cells(W16, b1_16, b2_16, c19_c_16, c19_rho_16, data)
    print(f"Bad pair indices: {bad_pair_idx.tolist()}")

    # Phase 1: Single-W-cell exhaustive fp16 tweak
    print(f"\n=== PHASE 1: Single W cell fp16 tweak ===")
    t0 = time.time()
    best_ll = ll
    best_cell = None
    best_val = None
    W_test = W16.copy()

    for i in range(32):
        for j in range(81):
            orig_val = W_test[i, j]
            neighbors = fp16_neighbors(orig_val, n_ulps=8)
            for nb in neighbors:
                if nb == orig_val:
                    continue
                W_test[i, j] = nb
                model = build_model(W_test, b1_16, b2_16, c19_c_16, c19_rho_16)
                ll_t, _, bp_t = metrics(model, data)
                if ll_t > best_ll:
                    best_ll = ll_t
                    best_cell = (i, j)
                    best_val = nb
                    print(f"  [W improvement] W[{i},{j}]: {orig_val:.6e} -> {nb:.6e} => ll={ll_t:.6f}% bad={bp_t}", flush=True)
            W_test[i, j] = orig_val
        if (i + 1) % 8 == 0:
            print(f"  row {i+1}/32, best_ll={best_ll:.6f}% ({time.time()-t0:.0f}s)", flush=True)

    if best_cell is not None and best_ll >= 100.0:
        i, j = best_cell
        W_test[i, j] = best_val
        print(f"\n  >>> LOSSLESS via W[{i},{j}] -> {best_val:.6e} <<<")
        # Save and return
        save_artifact(W_test, b1_16, b2_16, c19_c_16, c19_rho_16, ll=100.0, bp=0)
        return

    # If partial improvement, apply and try bias cells
    if best_cell is not None:
        i, j = best_cell
        W_test[i, j] = best_val
        print(f"  Applied W[{i},{j}] -> {best_val:.6e}, best_ll={best_ll:.6f}%")

    # Phase 2: Single b1/b2/c19 cell exhaustive fp16 tweak
    print(f"\n=== PHASE 2: bias / c19 single cell tweak ===")
    ll_cur, _, bp_cur = metrics(build_model(W_test, b1_16, b2_16, c19_c_16, c19_rho_16), data)
    print(f"  Current: ll={ll_cur:.6f}% bad={bp_cur}")

    found = False
    param_arrays = {"b1": b1_16, "b2": b2_16, "c19_c": c19_c_16, "c19_rho": c19_rho_16}
    for pname, parr in param_arrays.items():
        parr_test = parr.copy()
        for k in range(parr_test.size):
            orig_val = parr_test[k]
            neighbors = fp16_neighbors(orig_val, n_ulps=8)
            for nb in neighbors:
                if nb == orig_val: continue
                parr_test[k] = nb
                m_dict = {"b1": b1_16.copy(), "b2": b2_16.copy(), "c19_c": c19_c_16.copy(), "c19_rho": c19_rho_16.copy()}
                m_dict[pname] = parr_test
                model = build_model(W_test, m_dict["b1"], m_dict["b2"], m_dict["c19_c"], m_dict["c19_rho"])
                ll_t, _, bp_t = metrics(model, data)
                if ll_t > ll_cur:
                    ll_cur = ll_t
                    print(f"  [{pname} improvement] {pname}[{k}]: {orig_val:.6e} -> {nb:.6e} => ll={ll_t:.6f}% bad={bp_t}", flush=True)
                    if ll_t >= 100.0:
                        # Accept permanently
                        param_arrays[pname] = parr_test.copy()
                        if pname == "b1": b1_16 = param_arrays["b1"]
                        elif pname == "b2": b2_16 = param_arrays["b2"]
                        elif pname == "c19_c": c19_c_16 = param_arrays["c19_c"]
                        elif pname == "c19_rho": c19_rho_16 = param_arrays["c19_rho"]
                        save_artifact(W_test, b1_16, b2_16, c19_c_16, c19_rho_16, ll=100.0, bp=0)
                        found = True
                        break
            parr_test[k] = orig_val
            if found: break
        if found: break

    if found:
        return

    # Phase 3: 2-cell combinations on W (bad pairs only)
    print(f"\n=== PHASE 3: Pairwise W cell tweak ===")
    ll_cur, _, bp_cur = metrics(build_model(W_test, b1_16, b2_16, c19_c_16, c19_rho_16), data)
    print(f"  Current: ll={ll_cur:.6f}% bad={bp_cur}")
    if bp_cur == 0:
        print("  Already lossless from phase 1/2!")
        save_artifact(W_test, b1_16, b2_16, c19_c_16, c19_rho_16, ll=ll_cur, bp=bp_cur)
        return

    # Pick top-N candidate cells by impact: cells whose single tweak gave the biggest improvement
    # For simplicity, iterate all pairs (i1,j1,i2,j2), try all combinations — too slow for full.
    # Instead, do: top-50 cells x top-50 cells
    print(f"  (Full pairwise skipped — too slow. Saving current best.)")

    save_artifact(W_test, b1_16, b2_16, c19_c_16, c19_rho_16, ll=ll_cur, bp=bp_cur)


def save_artifact(W, b1, b2, c19_c, c19_rho, ll, bp):
    out = {
        "architecture": "fp16-all (no escape)" if bp == 0 else f"fp16-mostly + {bp} bad",
        "H": 81, "in_dim": 32,
        "final_lossless": ll, "final_bad": bp,
        "W_fp16": W.astype(np.float16).view(np.uint16).tolist(),
        "b1_fp16": b1.astype(np.float16).view(np.uint16).tolist(),
        "b2_fp16": b2.astype(np.float16).view(np.uint16).tolist(),
        "c19_c_fp16": c19_c.astype(np.float16).view(np.uint16).tolist(),
        "c19_rho_fp16": c19_rho.astype(np.float16).view(np.uint16).tolist(),
    }
    path = OUT_DIR / "final_fp16.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  Saved: {path} (ll={ll:.6f}% bad={bp})")
    # Size (binary deploy would be 2 B per fp16 value)
    total = (2592 + 81 + 32 + 81 + 81) * 2
    print(f"  Deploy (binary): {total} B ({total/1024:.2f} KB)")


if __name__ == "__main__":
    main()
