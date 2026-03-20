"""Matrix runner for ranked-prune regularization A/B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_rank_prune_regularization_ab import CONFIGS, parse_csv_ints, run_case


POLICIES = [
    {"name": "no_prune", "policy": "no_prune", "warmup": 0, "prune_interval": 0, "prune_frac": 0.0},
    {"name": "prune_w512_i512_f0.005", "policy": "periodic_prune", "warmup": 512, "prune_interval": 512, "prune_frac": 0.005},
    {"name": "prune_w1024_i512_f0.005", "policy": "periodic_prune", "warmup": 1024, "prune_interval": 512, "prune_frac": 0.005},
    {"name": "prune_w1024_i512_f0.01", "policy": "periodic_prune", "warmup": 1024, "prune_interval": 512, "prune_frac": 0.01},
    {"name": "prune_w1024_i1024_f0.01", "policy": "periodic_prune", "warmup": 1024, "prune_interval": 1024, "prune_frac": 0.01},
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="V64_N192", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", default="42,77,123")
    ap.add_argument("--attempts", default="4096,8192")
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--log-every", type=int, default=512)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", default="")
    return ap.parse_args()


def median(rows: list[dict], key: str) -> float:
    return float(np.median([float(r[key]) for r in rows]))


def summarize_budget(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in sorted({str(r["policy_name"]) for r in rows}):
        sub = [r for r in rows if str(r["policy_name"]) == name]
        out[name] = {
            "train_score_median": median(sub, "train_score_final"),
            "holdout_score_median": median(sub, "holdout_score_final"),
            "train_acc_median": median(sub, "train_acc_final"),
            "holdout_acc_median": median(sub, "holdout_acc_final"),
            "best_holdout_score_median": median(sub, "best_holdout_score"),
            "final_edges_median": median(sub, "final_edges"),
            "wall_ms_median": median(sub, "wall_ms"),
            "deterministic_all": float(all(bool(r["deterministic"]) for r in sub)),
        }
    return out


def budget_verdict(summary: dict[str, dict[str, float]]) -> dict[str, dict[str, object]]:
    base = summary["no_prune"]
    out: dict[str, dict[str, object]] = {}
    for name, stats in summary.items():
        if name == "no_prune":
            continue
        hold_gain = stats["holdout_score_median"] - base["holdout_score_median"]
        train_delta = stats["train_score_median"] - base["train_score_median"]
        edge_ratio = stats["final_edges_median"] / max(base["final_edges_median"], 1.0)
        positive = (
            stats["deterministic_all"] == 1.0
            and hold_gain >= 0.01
            and train_delta >= -0.03
        )
        research_only = (
            not positive
            and stats["deterministic_all"] == 1.0
            and hold_gain >= 0.005
        )
        out[name] = {
            "holdout_gain": hold_gain,
            "train_delta": train_delta,
            "edge_ratio_vs_baseline": edge_ratio,
            "positive": positive,
            "research_only": research_only,
        }
    return out


def overall_verdict(budget_rows: list[dict[str, object]]) -> dict[str, object]:
    positives: dict[str, int] = {}
    research: dict[str, int] = {}
    for b in budget_rows:
        for name, row in b["verdict"].items():
            if row["positive"]:
                positives[name] = positives.get(name, 0) + 1
            if row["research_only"]:
                research[name] = research.get(name, 0) + 1
    winner = None
    for name, count in positives.items():
        if count >= 2:
            winner = name
            break
    return {
        "positive_budget_counts": positives,
        "research_only_budget_counts": research,
        "winner": winner,
        "overall_positive": winner is not None,
    }


def main() -> int:
    args = parse_args()
    cfg = CONFIGS[args.config]
    seeds = parse_csv_ints(args.seeds)
    attempts_list = parse_csv_ints(args.attempts)

    results: list[dict[str, object]] = []
    for attempts in attempts_list:
        for policy in POLICIES:
            for seed in seeds:
                row = run_case(
                    cfg=cfg,
                    seed=seed,
                    attempts=attempts,
                    train_frac=args.train_frac,
                    warmup=policy["warmup"],
                    prune_interval=policy["prune_interval"],
                    prune_frac=policy["prune_frac"],
                    log_every=args.log_every,
                    eps=args.eps,
                    policy=policy["policy"],
                )
                row["attempt_budget"] = attempts
                row["policy_name"] = policy["name"]
                results.append(row)
                print(
                    f"attempts={attempts:5d} {policy['name']:24s} seed={seed:3d} "
                    f"train={row['train_score_final']:.4f} hold={row['holdout_score_final']:.4f} "
                    f"best_hold={row['best_holdout_score']:.4f} edges={row['final_edges']:4d} det={row['deterministic']}"
                )

    budget_rows = []
    for attempts in attempts_list:
        rows = [r for r in results if int(r["attempt_budget"]) == attempts]
        summary = summarize_budget(rows)
        verdict = budget_verdict(summary)
        budget_rows.append({"attempt_budget": attempts, "summary": summary, "verdict": verdict})

    out = {
        "branch_intent": "gpu ranked-prune regularization matrix",
        "args": vars(args),
        "results": results,
        "budgets": budget_rows,
        "overall_verdict": overall_verdict(budget_rows),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_rank_prune_regularization_matrix_{stamp}.json"
    output_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {output_path}")
    print(json.dumps(out["overall_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
