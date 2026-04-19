"""Generator-value compression prototype for single-W champion.

Idea:
  - Learn a small dictionary of generator magnitudes from the weight values.
  - Encode each weight as either:
      1) c * g_j
      2) a * g_j + b * g_k
      3) fp16 fallback
  - Acceptance is local: fit must land within the per-cell slack.

This is intentionally a static fit / byte-estimate probe, not an end-to-end
exact deploy format yet. The goal is to test whether "few generator values +
small integer coefficients" is even competitive with the current fp16 bulk
baseline on W itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


CHAMPION = Path("output/merger_single_w_exhaustive_fix/final_model.json")
SLACK = Path("output/merger_single_w_slack/slack.npy")
OUT = Path("output/merger_single_w_generator_value_proto.json")


def kmeans_1d(data: np.ndarray, k: int, n_iter: int = 100) -> np.ndarray:
    """Simple 1-D kmeans for positive magnitudes."""
    if len(data) == 0:
        return np.zeros((0,), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(data, qs).astype(np.float64)
    for _ in range(n_iter):
        d = np.abs(data[:, None] - centers[None, :])
        idx = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new_centers[j] = data[m].mean()
        if np.allclose(new_centers, centers, atol=1e-12):
            break
        centers = new_centers
    centers = np.sort(np.unique(np.maximum(centers, 1e-12)))
    return centers


def fit_one_generator(
    w: float,
    slack: float,
    gens: np.ndarray,
    coef_range: range,
) -> tuple[float, tuple[int, int] | None]:
    best_err = np.inf
    best = None
    for gi, g in enumerate(gens):
        for c in coef_range:
            if c == 0:
                continue
            err = abs(w - c * g)
            if err < best_err:
                best_err = err
                best = (gi, c)
    if best is not None and best_err <= slack:
        return best_err, best
    return best_err, None


def fit_two_generator(
    w: float,
    slack: float,
    gens: np.ndarray,
    coef_vals: list[int],
) -> tuple[float, tuple[int, int, int, int] | None]:
    best_err = np.inf
    best = None
    K = len(gens)
    for j in range(K):
        gj = gens[j]
        for k in range(j, K):
            gk = gens[k]
            for a in coef_vals:
                for b in coef_vals:
                    if a == 0 and b == 0:
                        continue
                    err = abs(w - (a * gj + b * gk))
                    if err < best_err:
                        best_err = err
                        best = (j, k, a, b)
    if best is not None and best_err <= slack:
        return best_err, best
    return best_err, None


def bits_for_signed_states(states: int) -> int:
    return int(math.ceil(math.log2(states)))


def main() -> None:
    with open(CHAMPION, "r", encoding="utf-8") as f:
        blob = json.load(f)
    W = np.array(blob["W"], dtype=np.float64).flatten()
    slack = np.load(SLACK).astype(np.float64).flatten()

    abs_pos = np.abs(W[np.abs(W) > 0])

    print("=== GENERATOR-VALUE PROTOTYPE ===")
    print(f"N cells: {len(W)}")
    print(f"Median slack: {np.median(slack):.6f}")
    print("Goal: beat W fp16 bulk = 5184 B while keeping static fit within slack.")

    results = []

    # The current exact deploy stores W as fp16 bulk: 2592 * 2 = 5184 B.
    fp16_w_bytes = len(W) * 2

    for K in [8, 16, 32]:
        gens = kmeans_1d(abs_pos, K)
        if len(gens) == 0:
            continue
        gen_bits = len(gens) * 16
        ref_bits = int(math.ceil(math.log2(len(gens))))
        one_coef_bits = bits_for_signed_states(15)   # [-7..7]
        two_coef_bits = bits_for_signed_states(7)    # [-3..3]

        counts = {"fallback": 0, "one": 0, "two": 0}
        err_sum = {"one": 0.0, "two": 0.0, "fallback": 0.0}

        for wi, si in zip(W, slack):
            e1, r1 = fit_one_generator(wi, si, gens, range(-7, 8))
            if r1 is not None:
                counts["one"] += 1
                err_sum["one"] += e1
                continue

            e2, r2 = fit_two_generator(wi, si, gens, [-3, -2, -1, 0, 1, 2, 3])
            if r2 is not None:
                counts["two"] += 1
                err_sum["two"] += e2
            else:
                counts["fallback"] += 1
                err_sum["fallback"] += min(e1, e2)

        # 2 mode bits per cell:
        #   00 fallback fp16
        #   01 one-generator
        #   10 two-generator
        bits_one = 2 + ref_bits + one_coef_bits
        bits_two = 2 + 2 * ref_bits + 2 * two_coef_bits
        bits_fallback = 2 + 16

        total_bits = gen_bits
        total_bits += counts["one"] * bits_one
        total_bits += counts["two"] * bits_two
        total_bits += counts["fallback"] * bits_fallback
        total_bytes = int(math.ceil(total_bits / 8))

        result = {
            "K": int(len(gens)),
            "generators": gens.tolist(),
            "counts": counts,
            "coverage_pct": 100.0 * (counts["one"] + counts["two"]) / len(W),
            "bits": {
                "generator_table": gen_bits,
                "per_one": bits_one,
                "per_two": bits_two,
                "per_fallback": bits_fallback,
                "total": total_bits,
            },
            "bytes": {
                "total": total_bytes,
                "delta_vs_fp16_W": total_bytes - fp16_w_bytes,
            },
            "mean_fit_error": {
                "one": 0.0 if counts["one"] == 0 else err_sum["one"] / counts["one"],
                "two": 0.0 if counts["two"] == 0 else err_sum["two"] / counts["two"],
            },
        }
        results.append(result)

        print(
            f"K={len(gens):2d}  one={counts['one']:4d}  two={counts['two']:4d}  "
            f"fallback={counts['fallback']:4d}  coverage={result['coverage_pct']:.1f}%  "
            f"W_bytes={total_bytes}  delta={total_bytes - fp16_w_bytes:+d}"
        )

    best = min(results, key=lambda r: r["bytes"]["total"]) if results else None
    summary = {
        "fp16_w_bytes": fp16_w_bytes,
        "results": results,
        "best": best,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best is not None:
        print(
            f"\nBest static W fit: K={best['K']}, "
            f"{best['bytes']['total']} B (delta {best['bytes']['delta_vs_fp16_W']:+d} vs fp16 W)"
        )
        print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
