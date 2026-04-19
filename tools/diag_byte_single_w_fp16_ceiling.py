"""Fp16 global ceiling test.

Cast every parameter to fp16 and back to fp32 (simulates fp16 storage).
Measure lossless. Progressively back off if fails.

Scopes:
  1. ALL fp16 (W, b1, b2, c19_c, c19_rho)
  2. W + bias fp16, c19 fp32
  3. W fp16 only, rest fp32
  4. W fp16 + per-cell escape (cells that break)

Also test bfloat16 for comparison.
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
OUT_DIR = Path("output/merger_single_w_fp16")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cast_roundtrip(arr, dtype):
    """Cast numpy float32 to dtype and back, simulating lossy storage."""
    t = torch.tensor(arr, dtype=torch.float32)
    t = t.to(dtype).to(torch.float32)
    return t.numpy()


def build_model(W, b1, b2, c19_c, c19_rho):
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(b1, dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(b2, dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(c19_c, dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(c19_rho, dtype=torch.float32, device=DEVICE))
    return model


def test_config(name, W, b1, b2, c19_c, c19_rho, data, size_bytes):
    model = build_model(W, b1, b2, c19_c, c19_rho)
    ll, pd, bp = metrics(model, data)
    status = "OK" if ll >= 100.0 else "BAD"
    print(f"  {status}  {name:40s}  ll={ll:.6f}%  bad={bp:5d}  size={size_bytes} B ({size_bytes/1024:.2f} KB)")
    return ll, bp


def main():
    print(f"=== FP16 GLOBAL CEILING TEST ===")
    print(f"Source: {CHAMPION}")

    with open(CHAMPION, "r") as f:
        m = json.load(f)
    W32 = np.array(m["W"], dtype=np.float32)
    b1_32 = np.array(m["b1"], dtype=np.float32)
    b2_32 = np.array(m["b2"], dtype=np.float32)
    c19_c_32 = np.array(m["c19_c"], dtype=np.float32)
    c19_rho_32 = np.array(m["c19_rho"], dtype=np.float32)

    data = load_byte_pairs().to(DEVICE)

    # Baseline
    print(f"\n[Baseline fp32 everything]")
    test_config("fp32 baseline", W32, b1_32, b2_32, c19_c_32, c19_rho_32, data,
                (2592 + 81 + 32 + 81 + 81) * 4)

    # Fp16 cast variants
    print(f"\n[FP16 variants]")
    W16 = cast_roundtrip(W32, torch.float16)
    b1_16 = cast_roundtrip(b1_32, torch.float16)
    b2_16 = cast_roundtrip(b2_32, torch.float16)
    c19_c_16 = cast_roundtrip(c19_c_32, torch.float16)
    c19_rho_16 = cast_roundtrip(c19_rho_32, torch.float16)

    # All fp16
    size_all_fp16 = (2592 + 81 + 32 + 81 + 81) * 2
    ll_all, bp_all = test_config("ALL fp16", W16, b1_16, b2_16, c19_c_16, c19_rho_16,
                                  data, size_all_fp16)

    # W fp16 only (bias+c19 fp32)
    size_w_fp16_only = 2592 * 2 + (81 + 32 + 81 + 81) * 4
    test_config("W fp16 only (bias+c19 fp32)", W16, b1_32, b2_32, c19_c_32, c19_rho_32,
                data, size_w_fp16_only)

    # W+bias fp16, c19 fp32
    size_wb_fp16 = (2592 + 81 + 32) * 2 + (81 + 81) * 4
    test_config("W+bias fp16, c19 fp32", W16, b1_16, b2_16, c19_c_32, c19_rho_32,
                data, size_wb_fp16)

    # W fp32, bias fp16, c19 fp32
    size_bias_fp16 = 2592 * 4 + (81 + 32) * 2 + (81 + 81) * 4
    test_config("W fp32, bias fp16, c19 fp32", W32, b1_16, b2_16, c19_c_32, c19_rho_32,
                data, size_bias_fp16)

    # c19 fp16 only (rest fp32)
    size_c19_fp16 = (2592 + 81 + 32) * 4 + (81 + 81) * 2
    test_config("c19 fp16 only (rest fp32)", W32, b1_32, b2_32, c19_c_16, c19_rho_16,
                data, size_c19_fp16)

    # BF16 variants
    print(f"\n[BF16 variants]")
    Wbf = cast_roundtrip(W32, torch.bfloat16)
    b1_bf = cast_roundtrip(b1_32, torch.bfloat16)
    b2_bf = cast_roundtrip(b2_32, torch.bfloat16)
    c19_c_bf = cast_roundtrip(c19_c_32, torch.bfloat16)
    c19_rho_bf = cast_roundtrip(c19_rho_32, torch.bfloat16)
    test_config("ALL bf16", Wbf, b1_bf, b2_bf, c19_c_bf, c19_rho_bf, data, size_all_fp16)

    # Per-cell escape for ALL fp16
    if ll_all < 100.0:
        print(f"\n[FP16 WITH ESCAPE — recover broken cells]")
        # For each W cell that differs between fp16 and fp32, check if reverting to fp32 helps
        delta_mask = (W32 != W16)
        n_diff = delta_mask.sum()
        print(f"  W cells changed by fp16 cast: {n_diff} / 2592")

        # Greedy: try reverting cells one by one in order of magnitude of change
        diff_mag = np.abs(W32 - W16)
        sorted_idx = np.argsort(-diff_mag.flatten())  # largest diff first

        W_cur = W16.copy()
        # Keep bias + c19 at fp16 (they may not be problem)
        model = build_model(W_cur, b1_16, b2_16, c19_c_16, c19_rho_16)
        ll, _, bp = metrics(model, data)
        print(f"  Start: ll={ll:.6f}% bad={bp}")

        escaped = []
        for k_idx, flat_idx in enumerate(sorted_idx):
            i, j = flat_idx // 81, flat_idx % 81
            if diff_mag[i, j] == 0:
                break  # no more diffs
            # Try reverting this cell to fp32
            old_val = W_cur[i, j]
            W_cur[i, j] = W32[i, j]
            model = build_model(W_cur, b1_16, b2_16, c19_c_16, c19_rho_16)
            ll_new, _, bp_new = metrics(model, data)
            if bp_new < bp:
                escaped.append((int(i), int(j)))
                bp = bp_new
                ll = ll_new
                if (len(escaped) % 10 == 0) or bp == 0:
                    print(f"  Escaped {len(escaped)} cells: ll={ll:.6f}% bad={bp}", flush=True)
            else:
                W_cur[i, j] = old_val  # revert
            if bp == 0:
                break
            if len(escaped) >= 300:
                break

        print(f"\n  Final with {len(escaped)} escapes:")
        model = build_model(W_cur, b1_16, b2_16, c19_c_16, c19_rho_16)
        ll_final, _, bp_final = metrics(model, data)
        # Size: fp16 bulk + fp32 escapes (need indices too)
        # Each escape: 2 B (index) + 2 B extra (store delta as fp16-to-fp32 diff, 2 B extra)
        # or simpler: each escape = 2 B index + 4 B fp32 value = 6 B
        size_escape = size_all_fp16 + len(escaped) * 6
        status = "OK" if ll_final >= 100.0 else "BAD"
        print(f"  {status}  ll={ll_final:.6f}% bad={bp_final} size={size_escape} B ({size_escape/1024:.2f} KB)")

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
