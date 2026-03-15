"""Analyze coarse + holdout charge-rate sweep logs with a deterministic decision ladder.

Priority order:
  1. constant
  2. simple formula
  3. learnable

The winner is chosen by lexicographic regret, then complexity.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


NET_DENSITY = {
    "V16_N80": 0.06,
    "V64_N192": 0.06,
    "V64_dense": 0.15,
    "V64_sparse": 0.02,
    "V128_N384": 0.06,
    "V128_dense": 0.15,
}

COMPLEXITY = {
    "constant": 0,
    "piecewise": 1,
    "linear": 2,
    "learnable": 3,
}


def parse_args() -> argparse.Namespace:
    default_logs = sorted(
        Path(__file__).resolve().parent.joinpath("logs").glob("charge_rate_sweep_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=default_logs[-1:] if default_logs else [],
        help="One or more charge_rate_sweep logs. Defaults to latest coarse log only.",
    )
    ap.add_argument("--const-worst-threshold-pp", type=float, default=2.0)
    ap.add_argument("--const-mean-threshold-pp", type=float, default=1.0)
    ap.add_argument("--formula-win-worst-pp", type=float, default=2.0)
    ap.add_argument("--formula-win-mean-pp", type=float, default=1.0)
    ap.add_argument("--learnable-win-worst-pp", type=float, default=2.0)
    ap.add_argument("--learnable-win-mean-pp", type=float, default=1.0)
    return ap.parse_args()


def parse_rows(log_paths: list[Path]):
    pat = re.compile(
        r"\] (\S+)\s+(\S+)\s+seed=\s*(\d+) acc=\s*([0-9.]+)% leak=([0-9.]+) cr=([0-9.]+)"
    )
    rows = []
    for log_path in log_paths:
        for line in log_path.read_text().splitlines():
            m = pat.search(line)
            if not m:
                continue
            net, mode, seed, acc, leak, cr = m.groups()
            rows.append(
                {
                    "source": str(log_path),
                    "net": net,
                    "mode": mode,
                    "seed": int(seed),
                    "acc": float(acc) / 100.0,
                    "leak": float(leak),
                    "cr": float(cr),
                }
            )
    return rows


def summarize(rows):
    means = defaultdict(dict)
    for net in sorted({r["net"] for r in rows}):
        modes = sorted({r["mode"] for r in rows if r["net"] == net})
        for mode in modes:
            vals = [r["acc"] for r in rows if r["net"] == net and r["mode"] == mode]
            if vals:
                means[net][mode] = sum(vals) / len(vals)

    best_fixed = {}
    fixed_grid = sorted(
        {
            float(mode.split("_", 1)[1])
            for net_modes in means.values()
            for mode in net_modes
            if mode.startswith("fix_")
        }
    )
    for net, md in means.items():
        fixed = {k: v for k, v in md.items() if k.startswith("fix_")}
        mode = max(fixed, key=fixed.get)
        best_fixed[net] = {
            "mode": mode,
            "cr": float(mode.split("_", 1)[1]),
            "acc": fixed[mode],
            "density": NET_DENSITY.get(net),
        }

    learnable = {}
    for net in means:
        vals = [r for r in rows if r["net"] == net and r["mode"] == "learnable"]
        if vals:
            learnable[net] = {
                "acc": sum(r["acc"] for r in vals) / len(vals),
                "cr": sum(r["cr"] for r in vals) / len(vals),
                "leak": sum(r["leak"] for r in vals) / len(vals),
            }

    return means, best_fixed, learnable, fixed_grid


def score_candidate(name, klass, per_net):
    regrets = [item["opt"] - item["acc"] for item in per_net]
    return {
        "name": name,
        "class": klass,
        "complexity": COMPLEXITY[klass],
        "worst_regret_pp": max(regrets) * 100.0,
        "mean_regret_pp": (sum(regrets) / len(regrets)) * 100.0,
        "per_net": per_net,
    }


def lookup_fixed_mode(means_for_net, target_rate: float):
    modes = []
    for mode in means_for_net:
        if not mode.startswith("fix_"):
            continue
        rate = float(mode.split("_", 1)[1])
        modes.append((abs(rate - target_rate), mode))
    if not modes:
        raise KeyError(f"No fixed mode for target_rate={target_rate}")
    modes.sort(key=lambda item: (item[0], item[1]))
    return modes[0][1]


def nearest_fixed_mode(pred, fixed_grid):
    nearest = min(fixed_grid, key=lambda x: abs(x - pred))
    return nearest


def make_candidates(means, best_fixed, learnable, fixed_grid):
    candidates = []

    constant_values = sorted(
        {
            float(mode.split("_", 1)[1])
            for net_modes in means.values()
            for mode in net_modes
            if mode.startswith("fix_")
        }
    )
    for cr in constant_values:
        per_net = []
        for net, meta in best_fixed.items():
            mode = lookup_fixed_mode(means[net], cr)
            per_net.append(
                {
                    "net": net,
                    "pred": cr,
                    "mode": mode,
                    "acc": means[net][mode],
                    "opt": meta["acc"],
                }
            )
        candidates.append(score_candidate(f"CONSTANT({cr:.2f})", "constant", per_net))

    for cutoff in [0.08, 0.10, 0.12]:
        per_net = []
        for net, meta in best_fixed.items():
            pred = 0.20 if meta["density"] is not None and meta["density"] > cutoff else 0.30
            nearest = nearest_fixed_mode(pred, fixed_grid)
            mode = lookup_fixed_mode(means[net], nearest)
            per_net.append(
                {
                    "net": net,
                    "pred": pred,
                    "mode": mode,
                    "acc": means[net][mode],
                    "opt": meta["acc"],
                }
            )
        candidates.append(
            score_candidate(f"PIECEWISE_DENSITY(d0={cutoff:.2f})", "piecewise", per_net)
        )

    linear_specs = [
        ("LINEAR_DENSITY(a=0.315,b=-0.77)", lambda d: max(0.20, min(0.30, 0.315 - 0.77 * d))),
        ("LINEAR_DENSITY(a=0.300,b=-0.67)", lambda d: max(0.20, min(0.30, 0.300 - 0.67 * d))),
    ]
    for name, law in linear_specs:
        per_net = []
        for net, meta in best_fixed.items():
            pred = law(meta["density"])
            nearest = nearest_fixed_mode(pred, fixed_grid)
            mode = lookup_fixed_mode(means[net], nearest)
            per_net.append(
                {
                    "net": net,
                    "pred": pred,
                    "mode": mode,
                    "acc": means[net][mode],
                    "opt": meta["acc"],
                }
            )
        candidates.append(score_candidate(name, "linear", per_net))

    if learnable:
        per_net = []
        for net, meta in best_fixed.items():
            if net not in learnable:
                break
            per_net.append(
                {
                    "net": net,
                    "pred": learnable[net]["cr"],
                    "mode": "learnable",
                    "acc": learnable[net]["acc"],
                    "opt": meta["acc"],
                }
            )
        if len(per_net) == len(best_fixed):
            candidates.append(score_candidate("LEARNABLE", "learnable", per_net))

    candidates.sort(key=lambda r: (r["worst_regret_pp"], r["mean_regret_pp"], r["complexity"], r["name"]))
    return candidates


def constant_gate(cand, args):
    return (
        cand["worst_regret_pp"] <= args.const_worst_threshold_pp
        and cand["mean_regret_pp"] <= args.const_mean_threshold_pp
    )


def pick_winner(candidates, args):
    constants = [c for c in candidates if c["class"] == "constant"]
    best_constant = min(constants, key=lambda c: (c["worst_regret_pp"], c["mean_regret_pp"], abs(float(c["name"][9:-1]) - 0.30), c["name"]))
    if constant_gate(best_constant, args):
        return best_constant, "constant_pass"

    piecewise = [c for c in candidates if c["class"] == "piecewise"]
    linear = [c for c in candidates if c["class"] == "linear"]
    formulas = sorted(piecewise + linear, key=lambda c: (c["worst_regret_pp"], c["mean_regret_pp"], c["complexity"], c["name"]))
    if formulas:
        best_formula = formulas[0]
        if (
            best_formula["worst_regret_pp"] <= args.const_worst_threshold_pp
            and (
                best_constant["worst_regret_pp"] - best_formula["worst_regret_pp"] >= args.formula_win_worst_pp
                or best_constant["mean_regret_pp"] - best_formula["mean_regret_pp"] >= args.formula_win_mean_pp
            )
        ):
            return best_formula, "formula_pass"

    learnables = [c for c in candidates if c["class"] == "learnable"]
    if learnables:
        best_formula_or_const = min(
            [best_constant] + formulas[:1],
            key=lambda c: (c["worst_regret_pp"], c["mean_regret_pp"], c["complexity"]),
        )
        best_learnable = learnables[0]
        if (
            best_formula_or_const["worst_regret_pp"] - best_learnable["worst_regret_pp"] >= args.learnable_win_worst_pp
            and best_formula_or_const["mean_regret_pp"] - best_learnable["mean_regret_pp"] >= args.learnable_win_mean_pp
        ):
            return best_learnable, "learnable_pass"

    return best_constant, "fallback_best_constant"


def main() -> int:
    args = parse_args()
    rows = parse_rows(args.logs)
    means, best_fixed, learnable, fixed_grid = summarize(rows)
    candidates = make_candidates(means, best_fixed, learnable, fixed_grid)
    winner, reason = pick_winner(candidates, args)

    print("LOGS")
    for p in args.logs:
        print(f"  {p}")

    print("\nBEST FIX PER CONFIG")
    for net, meta in best_fixed.items():
        print(
            f"  {net:12s} density={meta['density']:.2f} "
            f"best={meta['mode']:10s} acc={meta['acc']*100:5.1f}%"
        )

    if learnable:
        print("\nLEARNABLE FINAL MEANS")
        for net, meta in learnable.items():
            geff = meta["cr"] / max(1e-9, 1.0 - meta["leak"])
            print(
                f"  {net:12s} acc={meta['acc']*100:5.1f}% "
                f"cr={meta['cr']:.3f} leak={meta['leak']:.3f} geff={geff:.2f}"
            )

    print("\nSCORECARD")
    for cand in candidates:
        if cand["class"] == "constant":
            passed = constant_gate(cand, args)
        elif cand["class"] == "piecewise":
            passed = False
        elif cand["class"] == "linear":
            passed = False
        else:
            passed = False
        status = "PASS" if passed else "PENDING/FAIL"
        print(
            f"  {cand['name']:28s} class={cand['class']:9s} "
            f"worst_regret_pp={cand['worst_regret_pp']:5.2f} "
            f"mean_regret_pp={cand['mean_regret_pp']:5.2f} "
            f"complexity={cand['complexity']} status={status}"
        )

    print("\nWINNER")
    print(f"  winner={winner['name']}")
    print(f"  reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
