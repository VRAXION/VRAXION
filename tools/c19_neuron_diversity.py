#!/usr/bin/env python3
"""Compare two c19 neurons in a baked ensemble — agreement, correlation, LUT overlap."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_grow_explorer import gen_data
from c19_manual_explorer import neuron_eval_c19_pred


def analyze(task: str):
    state_path = Path("target") / "c19_manual_grow" / task / "state.json"
    state = json.loads(state_path.read_text())
    baked = state["neurons"]
    if len(baked) < 2:
        print(f"need >= 2 neurons, got {len(baked)}")
        return
    n0, n1 = baked[0], baked[1]

    (train, val, test, n_in) = gen_data(task, 42)

    print(f"=== {task} — N0 vs N1 diversity analysis ===\n")
    print(f"N0 parents={n0['parents']} weights={n0['weights']} c={n0['c']} rho={n0['rho']} alpha={n0['alpha']:+.4f}")
    print(f"N1 parents={n1['parents']} weights={n1['weights']} c={n1['c']} rho={n1['rho']} alpha={n1['alpha']:+.4f}\n")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n0_outs = []
        n1_outs = []
        for x, y in zip(split[0], split[1]):
            o0 = neuron_eval_c19_pred(n0, x)
            o1 = neuron_eval_c19_pred(n1, x + [o0])
            n0_outs.append(o0)
            n1_outs.append(o1)
        n = len(n0_outs)
        both_1 = sum(1 for a, b in zip(n0_outs, n1_outs) if a == 1 and b == 1)
        both_0 = sum(1 for a, b in zip(n0_outs, n1_outs) if a == 0 and b == 0)
        n0_only = sum(1 for a, b in zip(n0_outs, n1_outs) if a == 1 and b == 0)
        n1_only = sum(1 for a, b in zip(n0_outs, n1_outs) if a == 0 and b == 1)
        agree = both_1 + both_0
        n0_acc = 100.0 * sum(1 for o, y in zip(n0_outs, split[1]) if o == y) / n
        n1_acc = 100.0 * sum(1 for o, y in zip(n1_outs, split[1]) if o == y) / n
        n0_mean = sum(n0_outs) / n
        n1_mean = sum(n1_outs) / n
        cov = sum((a - n0_mean) * (b - n1_mean) for a, b in zip(n0_outs, n1_outs)) / n
        n0_var = sum((a - n0_mean) ** 2 for a in n0_outs) / n
        n1_var = sum((b - n1_mean) ** 2 for b in n1_outs) / n
        corr = cov / math.sqrt(n0_var * n1_var) if (n0_var * n1_var) > 0 else 0.0
        print(f"[{name}] n={n}")
        print(f"  N0 raw acc: {n0_acc:5.2f}%  (+ rate {100*sum(n0_outs)/n:5.1f}%)")
        print(f"  N1 raw acc: {n1_acc:5.2f}%  (+ rate {100*sum(n1_outs)/n:5.1f}%)")
        print(f"  both=1: {both_1:4d}   both=0: {both_0:4d}   N0_only: {n0_only:4d}   N1_only: {n1_only:4d}")
        print(f"  agreement: {100*agree/n:5.2f}%   disagreement: {100*(n-agree)/n:5.2f}%")
        print(f"  pearson correlation (output-level): {corr:+.4f}")
        print()

    # LUT level comparison
    print("=== LUT comparison ===")
    n0_lut = n0["lut"]
    n1_lut = n1["lut"]
    n0_min = n0["lut_min_dot"]
    n1_min = n1["lut_min_dot"]
    print(f"  N0 lut_min={n0_min}  len={len(n0_lut)}")
    print(f"  N1 lut_min={n1_min}  len={len(n1_lut)}")
    dots_n0 = list(range(n0_min, n0_min + len(n0_lut)))
    dots_n1 = list(range(n1_min, n1_min + len(n1_lut)))
    print(f"  N0 dots: {dots_n0}")
    print(f"  N1 dots: {dots_n1}")
    print(f"  N0 vals: {[f'{v:+.3f}' for v in n0_lut]}")
    print(f"  N1 vals: {[f'{v:+.3f}' for v in n1_lut]}")
    # If the dots overlap
    common_dots = sorted(set(dots_n0) & set(dots_n1))
    if common_dots:
        print(f"  common dots: {common_dots}")
        for d in common_dots:
            v0 = n0_lut[d - n0_min]
            v1 = n1_lut[d - n1_min]
            diff = v1 - v0
            print(f"    dot={d:+d}: N0={v0:+.4f}  N1={v1:+.4f}  diff={diff:+.4f}")
    # Sign pattern
    n0_signs = [("+" if v > 0.001 else ("-" if v < -0.001 else "0")) for v in n0_lut]
    n1_signs = [("+" if v > 0.001 else ("-" if v < -0.001 else "0")) for v in n1_lut]
    print(f"  N0 sign pattern: [{','.join(n0_signs)}]")
    print(f"  N1 sign pattern: [{','.join(n1_signs)}]")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "grid3_center"
    analyze(task)
