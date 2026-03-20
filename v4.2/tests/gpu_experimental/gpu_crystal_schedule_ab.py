"""GPU schedule A/B for add-only growth with periodic crystal.

Goal:
  - compare continuous add-only growth against split grow->crystal schedules
  - keep the current CPU canonical trainer untouched
  - reuse the current PassiveIO GPU scoring semantics

This is an experimental scheduler probe, not a canonical trainer replacement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_crystal_pass_ab import (
    CrystalState,
    crystal_pass_based,
    make_case,
    make_score_runner,
    parse_csv_ints,
    set_seeds,
)
from tests.gpu_experimental.gpu_swarm_v1 import batch_scores, sample_dead_edges
from model.graph import SelfWiringGraph


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

DEFAULT_CONFIGS = (64,)
DEFAULT_SEEDS = (42, 77, 123)
DEFAULT_SEGMENTS = (1, 2, 4)
DEFAULT_TOTAL_EVALS = {
    64: 4096,
    128: 8192,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77,123", help="Comma-separated integer seeds")
    ap.add_argument("--segments", default="1,2,4", help="Comma-separated segment counts; 1 means continuous")
    ap.add_argument("--total-evals", type=int, default=0, help="Override total add proposals per run")
    ap.add_argument("--k", type=int, default=1, help="Batch width per growth step")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--verdict-score-tol", type=float, default=5e-3, help="Practical score tolerance vs continuous")
    ap.add_argument("--compile-crystal-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def allocate_segment_budgets(total_evals: int, segments: int) -> list[int]:
    base = total_evals // segments
    rem = total_evals % segments
    return [base + (1 if i < rem else 0) for i in range(segments)]


def run_growth_phase(
    master_mask: torch.Tensor,
    alive: list[tuple[int, int]],
    alive_set: set[tuple[int, int]],
    master_score: float,
    case,
    rng: random.Random,
    phase_evals: int,
    K: int,
    ticks: int,
    eps: float,
) -> tuple[float, int]:
    promotions = 0
    candidate_evals = 0

    while candidate_evals < phase_evals:
        batch_k = min(K, phase_evals - candidate_evals)
        edges = sample_dead_edges(rng, case.H, alive_set, batch_k)
        if not edges:
            break

        candidate_masks = master_mask.unsqueeze(0).repeat(len(edges), 1, 1)
        signs = []
        for i, (row, col) in enumerate(edges):
            sign = SelfWiringGraph.DRIVE if rng.random() > 0.5 else -SelfWiringGraph.DRIVE
            candidate_masks[i, row, col] = sign
            signs.append(sign)

        scores = batch_scores(candidate_masks, case, ticks)
        candidate_evals += len(edges)
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        if best_score > master_score + eps:
            row, col = edges[best_idx]
            master_mask[row, col] = signs[best_idx]
            alive.append((row, col))
            alive_set.add((row, col))
            master_score = best_score
            promotions += 1

    return master_score, promotions


def run_schedule_once(
    V: int,
    seed: int,
    segments: int,
    total_evals: int,
    K: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> dict:
    set_seeds(seed)
    case, _ = make_case(V, seed, device)
    master_mask = torch.zeros((case.H, case.H), dtype=torch.float32, device=device)
    master_score = float(batch_scores(master_mask.unsqueeze(0), case, ticks)[0].item())
    alive: list[tuple[int, int]] = []
    alive_set: set[tuple[int, int]] = set()
    grow_rng = random.Random(seed + 600_000 + V * 1000 + segments * 10 + K)
    score_runner = make_score_runner(case, ticks, device, compile_crystal_eval)
    segment_budgets = allocate_segment_budgets(total_evals, segments)

    promotions_total = 0
    mid_crystal_removed_total = 0
    mid_crystal_passes_total = 0
    t0 = time.perf_counter()

    for idx, phase_budget in enumerate(segment_budgets):
        master_score, promotions = run_growth_phase(
            master_mask=master_mask,
            alive=alive,
            alive_set=alive_set,
            master_score=master_score,
            case=case,
            rng=grow_rng,
            phase_evals=phase_budget,
            K=K,
            ticks=ticks,
            eps=eps,
        )
        promotions_total += promotions

        if idx < len(segment_budgets) - 1 and alive:
            state = CrystalState(
                mask=master_mask.clone(),
                alive=list(alive),
                alive_set=set(alive_set),
                score=master_score,
            )
            crystal_rng = random.Random(seed + 700_000 + V * 1000 + segments * 10 + K + idx)
            crystal = crystal_pass_based(
                state=state,
                score_runner=score_runner,
                rng=crystal_rng,
                eps=eps,
                floor_score=master_score - eps,
                max_passes=0,
                max_wall_ms=0,
            )
            master_mask = state.mask
            alive = list(state.alive)
            alive_set = set(state.alive_set)
            master_score = float(crystal["score_after"])
            mid_crystal_removed_total += int(crystal["removed_count"])
            mid_crystal_passes_total += int(crystal["passes"])

    edges_before_final = len(alive)
    final_state = CrystalState(
        mask=master_mask.clone(),
        alive=list(alive),
        alive_set=set(alive_set),
        score=master_score,
    )
    final_rng = random.Random(seed + 800_000 + V * 1000 + segments * 10 + K)
    final_crystal = crystal_pass_based(
        state=final_state,
        score_runner=score_runner,
        rng=final_rng,
        eps=eps,
        floor_score=master_score - eps,
        max_passes=0,
        max_wall_ms=0,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "V": V,
        "seed": seed,
        "segments": segments,
        "K": K,
        "best_score_before_final_crystal": master_score,
        "best_score_after_final_crystal": float(final_crystal["score_after"]),
        "promotions": promotions_total,
        "candidate_evals": total_evals,
        "wall_ms": wall_ms,
        "candidate_per_sec": (total_evals / max(wall_ms / 1000.0, 1e-9)),
        "edges_before_final_crystal": edges_before_final,
        "final_edges": int(final_crystal["edges_after"]),
        "mid_crystal_removed_total": mid_crystal_removed_total,
        "mid_crystal_passes_total": mid_crystal_passes_total,
        "final_crystal_removed": int(final_crystal["removed_count"]),
        "final_crystal_passes": int(final_crystal["passes"]),
        "mask_hash": str(final_crystal["mask_hash"]),
    }


def run_case(
    V: int,
    seed: int,
    segments: int,
    total_evals: int,
    K: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> dict:
    first = run_schedule_once(V, seed, segments, total_evals, K, ticks, eps, device, compile_crystal_eval)
    second = run_schedule_once(V, seed, segments, total_evals, K, ticks, eps, device, compile_crystal_eval)
    deterministic = (
        abs(first["best_score_after_final_crystal"] - second["best_score_after_final_crystal"]) <= eps
        and abs(first["best_score_before_final_crystal"] - second["best_score_before_final_crystal"]) <= eps
        and first["promotions"] == second["promotions"]
        and first["final_edges"] == second["final_edges"]
        and first["mask_hash"] == second["mask_hash"]
    )
    return {**first, "deterministic": deterministic}


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for V in sorted({r["V"] for r in results}):
        summary[str(V)] = {}
        rows_v = [r for r in results if r["V"] == V]
        for segments in sorted({r["segments"] for r in rows_v}):
            rows = [r for r in rows_v if r["segments"] == segments]
            summary[str(V)][str(segments)] = {
                "n_cases": len(rows),
                "score_after_final_crystal_mean": float(np.mean([r["best_score_after_final_crystal"] for r in rows])),
                "score_after_final_crystal_median": float(np.median([r["best_score_after_final_crystal"] for r in rows])),
                "final_edges_mean": float(np.mean([r["final_edges"] for r in rows])),
                "final_edges_median": float(np.median([r["final_edges"] for r in rows])),
                "wall_ms_mean": float(np.mean([r["wall_ms"] for r in rows])),
                "wall_ms_median": float(np.median([r["wall_ms"] for r in rows])),
                "mid_crystal_removed_total_mean": float(np.mean([r["mid_crystal_removed_total"] for r in rows])),
                "promotions_mean": float(np.mean([r["promotions"] for r in rows])),
            }
    return summary


def verdict(results: list[dict], score_tol: float) -> dict:
    by_v: dict[str, dict[str, object]] = {}
    overall_positive = False

    for V in sorted({r["V"] for r in results}):
        rows_v = [r for r in results if r["V"] == V]
        baseline_rows = [r for r in rows_v if r["segments"] == 1]
        baseline_score = float(np.median([r["best_score_after_final_crystal"] for r in baseline_rows]))
        baseline_edges = float(np.median([r["final_edges"] for r in baseline_rows]))
        comparisons = []
        for segments in sorted({r["segments"] for r in rows_v if r["segments"] > 1}):
            rows = [r for r in rows_v if r["segments"] == segments]
            score_median = float(np.median([r["best_score_after_final_crystal"] for r in rows]))
            edges_median = float(np.median([r["final_edges"] for r in rows]))
            deterministic_ok = all(r["deterministic"] is True for r in rows)
            score_ok = score_median >= baseline_score - score_tol
            edge_ok = edges_median <= baseline_edges
            positive = deterministic_ok and (score_median > baseline_score + 1e-6 or (score_ok and edge_ok))
            overall_positive = overall_positive or positive
            comparisons.append(
                {
                    "segments": segments,
                    "positive": positive,
                    "deterministic_ok": deterministic_ok,
                    "score_after_final_crystal_median": score_median,
                    "final_edges_median": edges_median,
                    "score_ok": score_ok,
                    "edge_ok": edge_ok,
                }
            )
        by_v[str(V)] = {
            "baseline_segments_1_score_median": baseline_score,
            "baseline_segments_1_edges_median": baseline_edges,
            "comparisons": comparisons,
        }

    return {
        "overall_positive": overall_positive,
        "by_v": by_v,
        "gate": {
            "positive": f"segment>1 median score >= baseline - {score_tol} and median edges <= baseline, or strictly better score",
            "determinism": "all repeated runs deterministic",
        },
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    device = torch.device("cuda")
    configs = parse_csv_ints(args.configs)
    seeds = parse_csv_ints(args.seeds)
    segments_list = parse_csv_ints(args.segments)

    results = []
    for V in configs:
        total_evals = args.total_evals or DEFAULT_TOTAL_EVALS.get(V, max(2048, 64 * V))
        for segments in segments_list:
            for seed in seeds:
                row = run_case(
                    V=V,
                    seed=seed,
                    segments=segments,
                    total_evals=total_evals,
                    K=args.k,
                    ticks=args.ticks,
                    eps=args.eps,
                    device=device,
                    compile_crystal_eval=args.compile_crystal_eval,
                )
                results.append(row)
                print(
                    f"V={V} seg={segments} K={args.k} seed={seed} "
                    f"score={row['best_score_after_final_crystal']*100:6.2f}% "
                    f"edges={row['final_edges']:4d} "
                    f"mid_rm={row['mid_crystal_removed_total']:4d} "
                    f"wall_ms={row['wall_ms']:8.1f}"
                )

    summary = summarize(results)
    gate = verdict(results, args.verdict_score_tol)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "branch_intent": "gpu periodic crystal schedule A/B for PassiveIO mainline",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": gate,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_crystal_schedule_ab_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    print(json.dumps(gate, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
