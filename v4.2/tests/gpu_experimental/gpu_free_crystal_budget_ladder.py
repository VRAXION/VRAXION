"""Deterministic GPU orchestrator for free-crystal budget ladders.

Runs the exact free mid-crystal schedule probe across multiple budgets and
produces a single source-of-truth JSON report. If V64 shows a real signal,
optionally confirms on V128 using the winning crystal count only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_free_crystal_frequency_ab import (
    DEFAULT_CONFIGS,
    DEFAULT_CRYSTALS,
    DEFAULT_SEEDS,
    run_case,
)
from tests.gpu_experimental.gpu_crystal_pass_ab import parse_csv_ints


DEFAULT_BUDGETS = (2048, 4096, 8192)
DEFAULT_V128_BUDGETS = (4096, 8192)
DEFAULT_V128_LATE_BUDGETS = (8192, 16384)


def parse_int_csv_or_default(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = (raw or "").strip()
    if not raw:
        return default
    return tuple(parse_csv_ints(raw))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64", help="Comma-separated V sizes to run initially")
    ap.add_argument("--seeds", default="42,77,123", help="Comma-separated V64 seeds")
    ap.add_argument("--budgets", default="2048,4096,8192", help="Comma-separated V64 budgets")
    ap.add_argument("--crystals", default="0,1,2,4", help="Comma-separated crystal counts")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--score-tol", type=float, default=5e-3)
    ap.add_argument("--v128-seeds", default="42,77", help="Comma-separated V128 confirm seeds")
    ap.add_argument("--compile-crystal-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def summarize_budget_matrix(results: list[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for V in sorted({int(r["V"]) for r in results}):
        rows_v = [r for r in results if int(r["V"]) == V]
        summary[str(V)] = {}
        for budget in sorted({int(r["budget"]) for r in rows_v}):
            rows_b = [r for r in rows_v if int(r["budget"]) == budget]
            summary[str(V)][str(budget)] = {}
            for n_crystals in sorted({int(r["n_crystals"]) for r in rows_b}):
                rows = [r for r in rows_b if int(r["n_crystals"]) == n_crystals]
                summary[str(V)][str(budget)][str(n_crystals)] = {
                    "n_cases": len(rows),
                    "score_after_growth_mean": float(sum(r["score_after_growth"] for r in rows) / len(rows)),
                    "score_after_growth_median": float(_median([r["score_after_growth"] for r in rows])),
                    "final_edges_mean": float(sum(r["final_edges"] for r in rows) / len(rows)),
                    "final_edges_median": float(_median([r["final_edges"] for r in rows])),
                    "wall_ms_mean": float(sum(r["wall_ms"] for r in rows) / len(rows)),
                    "wall_ms_median": float(_median([r["wall_ms"] for r in rows])),
                    "promotions_mean": float(sum(r["promotions"] for r in rows) / len(rows)),
                    "crystal_removed_total_mean": float(sum(r["crystal_removed_total"] for r in rows) / len(rows)),
                    "crystal_passes_total_mean": float(sum(r["crystal_passes_total"] for r in rows) / len(rows)),
                }
    return summary


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty list")
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def evaluate_budget_gate(summary: dict, results: list[dict], score_tol: float) -> dict:
    by_v: dict[str, dict[str, object]] = {}
    winner_by_v: dict[str, dict[str, object] | None] = {}
    overall_positive = False

    for V_str, budgets in summary.items():
        budget_rows = []
        positive_counts: dict[int, int] = {}
        bake_counts: dict[int, int] = {}
        for budget_str, crystal_rows in budgets.items():
            baseline = crystal_rows["0"]
            baseline_score = float(baseline["score_after_growth_median"])
            baseline_edges = float(baseline["final_edges_median"])
            baseline_wall = float(baseline["wall_ms_median"])
            candidates = []
            for n_key, stats in crystal_rows.items():
                if n_key == "0":
                    continue
                rows = [
                    r for r in results
                    if str(r["V"]) == V_str
                    and str(r["budget"]) == budget_str
                    and str(r["n_crystals"]) == n_key
                ]
                deterministic_all = all(bool(r["deterministic"]) for r in rows)
                score = float(stats["score_after_growth_median"])
                edges = float(stats["final_edges_median"])
                wall = float(stats["wall_ms_median"])
                score_ok = score >= baseline_score - score_tol
                positive = (score >= baseline_score + 3e-3) or (score_ok and edges <= 0.9 * baseline_edges)
                bake_ready = positive and deterministic_all and wall <= 1.5 * baseline_wall
                research_only_positive = positive and deterministic_all and wall > 1.5 * baseline_wall
                if positive:
                    positive_counts[int(n_key)] = positive_counts.get(int(n_key), 0) + 1
                if bake_ready:
                    bake_counts[int(n_key)] = bake_counts.get(int(n_key), 0) + 1
                overall_positive = overall_positive or positive
                candidates.append({
                    "n_crystals": int(n_key),
                    "score_after_growth_median": score,
                    "final_edges_median": edges,
                    "wall_ms_median": wall,
                    "deterministic_all": deterministic_all,
                    "score_ok": score_ok,
                    "positive": positive,
                    "bake_ready": bake_ready,
                    "research_only_positive": research_only_positive,
                })
            budget_rows.append({
                "budget": int(budget_str),
                "baseline_no_crystal_score_median": baseline_score,
                "baseline_no_crystal_edges_median": baseline_edges,
                "baseline_no_crystal_wall_median": baseline_wall,
                "candidates": candidates,
            })

        winner = None
        if positive_counts:
            valid = sorted(
                (
                    (n_crystals, count, bake_counts.get(n_crystals, 0))
                    for n_crystals, count in positive_counts.items()
                    if count >= 2
                ),
                key=lambda item: (-item[1], -item[2], item[0]),
            )
            if valid:
                winner = {
                    "n_crystals": valid[0][0],
                    "positive_budget_count": valid[0][1],
                    "bake_ready_budget_count": valid[0][2],
                }

        by_v[V_str] = {
            "budgets": budget_rows,
            "positive_budget_counts": positive_counts,
            "bake_ready_budget_counts": bake_counts,
            "winner": winner,
        }
        winner_by_v[V_str] = winner

    return {
        "overall_positive": overall_positive,
        "by_v": by_v,
        "winner_by_v": winner_by_v,
    }


def choose_v128_budgets(v64_rows: list[dict], winner_crystals: int) -> tuple[int, ...]:
    positive_budgets = []
    for row in v64_rows:
        for cand in row["candidates"]:
            if cand["n_crystals"] == winner_crystals and cand["positive"]:
                positive_budgets.append(int(row["budget"]))
    if any(b in (2048, 4096) for b in positive_budgets):
        return DEFAULT_V128_BUDGETS
    return DEFAULT_V128_LATE_BUDGETS


def run_matrix(
    *,
    V: int,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    crystals: tuple[int, ...],
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> list[dict]:
    results: list[dict] = []
    for budget in budgets:
        for n_crystals in crystals:
            for seed in seeds:
                row = run_case(
                    V=V,
                    seed=seed,
                    n_crystals=n_crystals,
                    total_evals=budget,
                    ticks=ticks,
                    eps=eps,
                    device=device,
                    compile_crystal_eval=compile_crystal_eval,
                )
                row["budget"] = int(budget)
                results.append(row)
                print(
                    f"V={V:3d} budget={budget:5d} n_crystals={n_crystals:2d} seed={seed:3d} "
                    f"score={row['score_after_growth']*100:6.2f}% "
                    f"edges={row['final_edges']:4d} "
                    f"cr_removed={row['crystal_removed_total']:4d} "
                    f"wall_ms={row['wall_ms']:8.1f}"
                )
    return results


def main() -> int:
    args = parse_args()
    configs = parse_int_csv_or_default(args.configs, DEFAULT_CONFIGS)
    if configs != (64,):
        # Keep the first stage decision-complete and narrow. V128 is conditional.
        configs = tuple(v for v in configs if v == 64) or (64,)
    seeds_v64 = parse_int_csv_or_default(args.seeds, DEFAULT_SEEDS)
    seeds_v128 = parse_int_csv_or_default(args.v128_seeds, (42, 77))
    budgets_v64 = parse_int_csv_or_default(args.budgets, DEFAULT_BUDGETS)
    crystals = parse_int_csv_or_default(args.crystals, DEFAULT_CRYSTALS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results: list[dict] = []

    v64_results = run_matrix(
        V=64,
        seeds=seeds_v64,
        budgets=budgets_v64,
        crystals=crystals,
        ticks=args.ticks,
        eps=args.eps,
        device=device,
        compile_crystal_eval=args.compile_crystal_eval,
    )
    all_results.extend(v64_results)

    summary_v64 = summarize_budget_matrix(v64_results)
    verdict_v64 = evaluate_budget_gate(summary_v64, v64_results, args.score_tol)

    v128_results: list[dict] = []
    v128_confirm = None
    winner = verdict_v64["winner_by_v"].get("64")
    if winner is not None:
        winner_crystals = int(winner["n_crystals"])
        budgets_v128 = choose_v128_budgets(verdict_v64["by_v"]["64"]["budgets"], winner_crystals)
        v128_results = run_matrix(
            V=128,
            seeds=seeds_v128,
            budgets=budgets_v128,
            crystals=(winner_crystals,),
            ticks=args.ticks,
            eps=args.eps,
            device=device,
            compile_crystal_eval=args.compile_crystal_eval,
        )
        all_results.extend(v128_results)
        summary_v128 = summarize_budget_matrix(v128_results)
        verdict_v128 = evaluate_budget_gate(summary_v128, v128_results, args.score_tol)
        v128_confirm = {
            "winner_crystals": winner_crystals,
            "budgets": list(budgets_v128),
            "summary": summary_v128,
            "verdict": verdict_v128,
        }

    full_summary = summarize_budget_matrix(all_results)
    payload = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "branch_intent": "gpu deterministic free crystal budget ladder",
        "args": vars(args),
        "results": all_results,
        "v64_summary": summary_v64,
        "v64_verdict": verdict_v64,
        "v128_confirm": v128_confirm,
        "full_summary": full_summary,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    output_path = args.output
    if not output_path:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(logs_dir / f"gpu_free_crystal_budget_ladder_{stamp}.json")

    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {output_path}")
    print(json.dumps(verdict_v64, indent=2))
    if v128_confirm is not None:
        print(json.dumps(v128_confirm["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
