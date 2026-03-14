"""Analyze v4.2 charge-rate sweep logs and score simple law candidates.

This is a meta-analysis layer over completed charge_rate_sweep logs:
  - parse per-config fixed/learnable results
  - identify config-level best fixed charge_rate
  - score simple candidate laws by regret against the observed optimum
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


NET_DENSITY = {
    "V16_N80": 0.06,
    "V64_N192": 0.06,
    "V64_dense": 0.15,
    "V64_sparse": 0.02,
    "V128_N384": 0.06,
}

FIXED_GRID = [0.1, 0.2, 0.3, 0.5, 0.7]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=max(
            Path(__file__).resolve().parent.joinpath("logs").glob("charge_rate_sweep_*.log"),
            key=lambda p: p.stat().st_mtime,
        ),
        help="Path to a charge_rate_sweep log file (defaults to latest).",
    )
    return ap.parse_args()


def parse_rows(log_path: Path):
    pat = re.compile(
        r"\] (\S+)\s+(\S+)\s+seed=\s*(\d+) acc=\s*([0-9.]+)% leak=([0-9.]+) cr=([0-9.]+)"
    )
    rows = []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        net, mode, seed, acc, leak, cr = m.groups()
        rows.append(
            {
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
    for net in NET_DENSITY:
        modes = sorted({r["mode"] for r in rows if r["net"] == net})
        for mode in modes:
            vals = [r["acc"] for r in rows if r["net"] == net and r["mode"] == mode]
            if vals:
                means[net][mode] = sum(vals) / len(vals)

    best = {}
    for net, md in means.items():
        fixed = {k: v for k, v in md.items() if k.startswith("fix_")}
        mode = max(fixed, key=fixed.get)
        best[net] = {
            "mode": mode,
            "cr": float(mode.split("_")[1]),
            "acc": fixed[mode],
            "density": NET_DENSITY[net],
        }

    learnable = {}
    for net in NET_DENSITY:
        lr = [r for r in rows if r["net"] == net and r["mode"] == "learnable"]
        if lr:
            learnable[net] = {
                "acc": sum(r["acc"] for r in lr) / len(lr),
                "cr": sum(r["cr"] for r in lr) / len(lr),
                "leak": sum(r["leak"] for r in lr) / len(lr),
            }

    return means, best, learnable


def score_laws(means, best):
    candidate_laws = {
        "const_0.30": lambda d: 0.30,
        "const_0.20": lambda d: 0.20,
        "piecewise_dense": lambda d: 0.20 if d > 0.10 else 0.30,
        "linear_density": lambda d: max(0.20, min(0.30, 0.315 - 0.77 * d)),
        "power_density": lambda d: max(0.20, min(0.30, 0.30 * (0.06 / max(d, 1e-9)) ** 0.20)),
    }

    score_rows = []
    for name, law in candidate_laws.items():
        per_net = []
        regrets = []
        ratios = []
        for net, meta in best.items():
            pred = law(meta["density"])
            nearest = min(FIXED_GRID, key=lambda x: abs(x - pred))
            mode = f"fix_{nearest:.1f}"
            acc = means[net][mode]
            opt = meta["acc"]
            regrets.append(opt - acc)
            ratios.append(acc / opt if opt > 0 else 0.0)
            per_net.append(
                {
                    "net": net,
                    "density": meta["density"],
                    "pred": pred,
                    "nearest": nearest,
                    "acc": acc,
                    "opt": opt,
                }
            )
        score_rows.append(
            {
                "name": name,
                "mean_regret": sum(regrets) / len(regrets),
                "worst_regret": max(regrets),
                "mean_ratio": sum(ratios) / len(ratios),
                "worst_ratio": min(ratios),
                "per_net": per_net,
            }
        )

    score_rows.sort(key=lambda r: (r["worst_regret"], r["mean_regret"]))
    return score_rows


def main() -> int:
    args = parse_args()
    rows = parse_rows(args.log)
    means, best, learnable = summarize(rows)
    score_rows = score_laws(means, best)

    print(f"LOG: {args.log}")
    print("\nBEST FIX PER CONFIG")
    for net, meta in best.items():
        print(
            f"  {net:12s} density={meta['density']:.2f} "
            f"best={meta['mode']:8s} acc={meta['acc']*100:5.1f}%"
        )

    print("\nLEARNABLE FINAL MEANS")
    for net, meta in learnable.items():
        print(
            f"  {net:12s} acc={meta['acc']*100:5.1f}% "
            f"cr={meta['cr']:.3f} leak={meta['leak']:.3f} "
            f"geff={meta['cr']/max(1e-9, 1.0-meta['leak']):.2f}"
        )

    print("\nCANDIDATE LAW SCORECARD")
    for row in score_rows:
        print(
            f"  {row['name']:16s} "
            f"mean_reg={row['mean_regret']*100:5.2f}pp "
            f"worst_reg={row['worst_regret']*100:5.2f}pp "
            f"mean_ratio={row['mean_ratio']*100:6.2f}% "
            f"worst_ratio={row['worst_ratio']*100:6.2f}%"
        )
        for item in row["per_net"]:
            print(
                f"    {item['net']:12s} d={item['density']:.2f} "
                f"pred={item['pred']:.3f} use={item['nearest']:.1f} "
                f"acc={item['acc']*100:5.1f}% opt={item['opt']*100:5.1f}%"
            )

    best_row = score_rows[0]
    print(
        "\nRECOMMENDATION\n"
        f"  best_law={best_row['name']} "
        f"worst_regret={best_row['worst_regret']*100:.2f}pp "
        f"mean_regret={best_row['mean_regret']*100:.2f}pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
