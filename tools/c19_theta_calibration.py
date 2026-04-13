#!/usr/bin/env python3
"""Per-head theta calibration sweep for all 9 grid3_copy_bit_N c19 ensembles.

For each head, compute per-sample ensemble scores on the train set, then sweep
theta in [-0.5, +0.5] to find the threshold that maximizes train acc. Verify
val/test acc at that theta. Compare to theta=0 baseline.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_grow_explorer import gen_data
from c19_manual_explorer import lut_lookup


def ensemble_score(baked_neurons, input_bits):
    if not baked_neurons:
        return 0.0
    score = 0.0
    sigs = list(input_bits)
    for neuron in baked_neurons:
        dot = 0
        for w, p in zip(neuron["weights"], neuron["parents"]):
            if w != 0:
                dot += w * sigs[p]
        lut_val = lut_lookup(neuron["lut"], neuron["lut_min_dot"], dot)
        score += neuron["alpha"] * lut_val
        sigs.append(1 if lut_val >= 0.0 else 0)
    return score


def acc_at_theta(scores, labels, theta: float) -> float:
    ok = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= theta else 0
        if pred == y:
            ok += 1
    return 100.0 * ok / len(labels)


def sweep_theta(tr_scores, tr_labels, va_scores, va_labels, te_scores, te_labels):
    all_scores = sorted(set(tr_scores))
    if not all_scores:
        return None
    # Candidate thetas = midpoints between adjacent unique scores, plus endpoints
    thetas = sorted(set([-0.5, 0.0, 0.5] + all_scores))
    # Also midpoints
    midpoints = []
    for i in range(len(all_scores) - 1):
        midpoints.append((all_scores[i] + all_scores[i + 1]) / 2)
    thetas = sorted(set(thetas + midpoints))

    best = {"theta": 0.0, "train": 0.0, "val": 0.0, "test": 0.0}
    for theta in thetas:
        tr_acc = acc_at_theta(tr_scores, tr_labels, theta)
        if tr_acc > best["train"]:
            best = {
                "theta": theta,
                "train": tr_acc,
                "val": acc_at_theta(va_scores, va_labels, theta),
                "test": acc_at_theta(te_scores, te_labels, theta),
            }
    return best


def main():
    print("=" * 78)
    print("c19 per-head theta calibration for grid3_copy_bit_0..8")
    print("=" * 78)
    print()
    print(f"{'bit':<4} {'neurons':<8} {'theta0 val/test':<20} {'best_theta':<12} {'swept val/test':<18}")
    print("-" * 78)

    results = []
    for bit_idx in range(9):
        state_path = Path("target") / "c19_manual_grow" / f"grid3_copy_bit_{bit_idx}" / "state.json"
        if not state_path.exists():
            print(f"  {bit_idx}  <no state>")
            continue
        state = json.loads(state_path.read_text())
        baked = state.get("neurons", [])
        (train, val, test, n_in) = gen_data(f"grid3_copy_bit_{bit_idx}", 42)
        tr_scores = [ensemble_score(baked, x) for x in train[0]]
        va_scores = [ensemble_score(baked, x) for x in val[0]]
        te_scores = [ensemble_score(baked, x) for x in test[0]]
        theta0_val = acc_at_theta(va_scores, val[1], 0.0)
        theta0_test = acc_at_theta(te_scores, test[1], 0.0)

        swept = sweep_theta(tr_scores, train[1], va_scores, val[1], te_scores, test[1])
        if swept is None:
            print(f"  {bit_idx}  {len(baked)}       {theta0_val:5.2f}/{theta0_test:5.2f}      (empty)")
            continue

        print(f"  {bit_idx}  {len(baked):<7}  {theta0_val:5.2f}/{theta0_test:5.2f}        "
              f"{swept['theta']:+7.4f}    {swept['val']:5.2f}/{swept['test']:5.2f}")
        results.append({
            "bit": bit_idx,
            "neurons": len(baked),
            "theta0_val": theta0_val,
            "theta0_test": theta0_test,
            "best_theta": swept["theta"],
            "best_train": swept["train"],
            "swept_val": swept["val"],
            "swept_test": swept["test"],
        })

    print()
    if results:
        delta_vals = [r["swept_val"] - r["theta0_val"] for r in results]
        delta_tests = [r["swept_test"] - r["theta0_test"] for r in results]
        mean_dv = sum(delta_vals) / len(delta_vals)
        mean_dt = sum(delta_tests) / len(delta_tests)
        improved = sum(1 for r in results if r["swept_val"] > r["theta0_val"])
        print(f"theta calibration improvement: mean val delta = {mean_dv:+.3f}pp, mean test delta = {mean_dt:+.3f}pp")
        print(f"heads where theta>0 helps val: {improved}/{len(results)}")
    out_path = Path("target") / "c19_grid3_copy" / "theta_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
