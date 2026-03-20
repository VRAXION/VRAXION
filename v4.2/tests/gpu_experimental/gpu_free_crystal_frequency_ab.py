"""Deterministic GPU probe for free mid-crystal frequency schedules.

Goal:
  - reproduce the CPU-side "grow -> crystal -> grow" finding on GPU
  - compare exact free mid-training crystal counts under the same add-only
    proposal stream
  - keep canonical CPU training untouched

Definition:
  - total_evals counts only add proposals
  - crystals are FREE: they do not consume add budget
  - n_crystals inserts that many pass-based crystal events between N+1 growth
    segments
  - no final crystal by default; this mirrors the CPU-side schedule claim
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
    mask_hash,
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
DEFAULT_CRYSTALS = (0, 1, 2, 4)
DEFAULT_TOTAL_EVALS = {
    64: 2048,
    128: 4096,
}


def parse_int_csv_or_default(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = (raw or "").strip()
    if not raw:
        return default
    return tuple(parse_csv_ints(raw))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77,123", help="Comma-separated integer seeds")
    ap.add_argument("--crystals", default="0,1,2,4", help="Comma-separated free mid-crystal counts")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--score-tol", type=float, default=5e-3, help="Practical score tolerance vs no-crystal")
    ap.add_argument("--total-evals", type=int, default=0, help="Override total add proposals per run")
    ap.add_argument("--compile-crystal-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def allocate_segment_budgets(total_evals: int, n_segments: int) -> list[int]:
    base = total_evals // n_segments
    rem = total_evals % n_segments
    return [base + (1 if i < rem else 0) for i in range(n_segments)]


def candidate_score_for_add(
    master_mask: torch.Tensor,
    case,
    ticks: int,
    row: int,
    col: int,
    sign: float,
) -> float:
    candidate_mask = master_mask.unsqueeze(0).clone()
    candidate_mask[0, row, col] = sign
    return float(batch_scores(candidate_mask, case, ticks)[0].item())


def run_growth_phase(
    *,
    master_mask: torch.Tensor,
    alive: list[tuple[int, int]],
    alive_set: set[tuple[int, int]],
    master_score: float,
    case,
    rng: random.Random,
    phase_evals: int,
    ticks: int,
    eps: float,
) -> tuple[float, int]:
    promotions = 0
    candidate_evals = 0

    while candidate_evals < phase_evals:
        edges = sample_dead_edges(rng, case.H, alive_set, 1)
        if not edges:
            break
        row, col = edges[0]
        sign = SelfWiringGraph.DRIVE if rng.random() > 0.5 else -SelfWiringGraph.DRIVE
        new_score = candidate_score_for_add(master_mask, case, ticks, row, col, sign)
        candidate_evals += 1

        if new_score > master_score + eps:
            master_mask[row, col] = sign
            alive.append((row, col))
            alive_set.add((row, col))
            master_score = new_score
            promotions += 1

    return master_score, promotions


def maybe_crystal(
    *,
    alive: list[tuple[int, int]],
    alive_set: set[tuple[int, int]],
    master_mask: torch.Tensor,
    master_score: float,
    case,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
    crystal_seed: int,
    eval_index: int,
) -> tuple[torch.Tensor, list[tuple[int, int]], set[tuple[int, int]], float, dict]:
    score_runner = make_score_runner(case, ticks, device, compile_crystal_eval)
    state = CrystalState(
        mask=master_mask.clone(),
        alive=list(alive),
        alive_set=set(alive_set),
        score=master_score,
    )
    score_before = master_score
    edges_before = len(alive)
    crystal = crystal_pass_based(
        state=state,
        score_runner=score_runner,
        rng=random.Random(crystal_seed),
        eps=eps,
        floor_score=score_before - eps,
        max_passes=0,
        max_wall_ms=0,
    )
    event = {
        "eval_index": eval_index,
        "score_before": score_before,
        "score_after": float(crystal["score_after"]),
        "edges_before": edges_before,
        "edges_after": int(crystal["edges_after"]),
        "removed_count": int(crystal["removed_count"]),
        "passes": int(crystal["passes"]),
    }
    return state.mask, list(state.alive), set(state.alive_set), float(crystal["score_after"]), event


def run_once(
    *,
    V: int,
    seed: int,
    n_crystals: int,
    total_evals: int,
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

    # Keep add-only proposal order invariant across crystal-count policies.
    grow_rng = random.Random(seed + 950_000 + V * 1000)
    budgets = allocate_segment_budgets(total_evals, n_crystals + 1)

    promotions_total = 0
    crystal_removed_total = 0
    crystal_passes_total = 0
    crystal_events: list[dict[str, object]] = []
    t0 = time.perf_counter()
    consumed = 0

    for seg_idx, phase_budget in enumerate(budgets):
        master_score, promotions = run_growth_phase(
            master_mask=master_mask,
            alive=alive,
            alive_set=alive_set,
            master_score=master_score,
            case=case,
            rng=grow_rng,
            phase_evals=phase_budget,
            ticks=ticks,
            eps=eps,
        )
        promotions_total += promotions
        consumed += phase_budget

        if seg_idx < len(budgets) - 1 and alive:
            crystal_seed = seed + 960_000 + V * 1000 + consumed * 10 + seg_idx
            master_mask, alive, alive_set, master_score, event = maybe_crystal(
                alive=alive,
                alive_set=alive_set,
                master_mask=master_mask,
                master_score=master_score,
                case=case,
                ticks=ticks,
                eps=eps,
                device=device,
                compile_crystal_eval=compile_crystal_eval,
                crystal_seed=crystal_seed,
                eval_index=consumed,
            )
            crystal_removed_total += int(event["removed_count"])
            crystal_passes_total += int(event["passes"])
            crystal_events.append(event)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "V": V,
        "seed": seed,
        "n_crystals": n_crystals,
        "ticks": ticks,
        "K": 1,
        "total_evals": total_evals,
        "score_after_growth": master_score,
        "promotions": promotions_total,
        "candidate_evals": total_evals,
        "wall_ms": wall_ms,
        "candidate_per_sec": total_evals / max(wall_ms / 1000.0, 1e-9),
        "final_edges": len(alive),
        "crystal_removed_total": crystal_removed_total,
        "crystal_passes_total": crystal_passes_total,
        "crystal_events": crystal_events,
        "mask_hash": mask_hash(master_mask),
    }


def run_case(
    *,
    V: int,
    seed: int,
    n_crystals: int,
    total_evals: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> dict:
    first = run_once(
        V=V,
        seed=seed,
        n_crystals=n_crystals,
        total_evals=total_evals,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_crystal_eval=compile_crystal_eval,
    )
    second = run_once(
        V=V,
        seed=seed,
        n_crystals=n_crystals,
        total_evals=total_evals,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_crystal_eval=compile_crystal_eval,
    )
    deterministic = (
        abs(first["score_after_growth"] - second["score_after_growth"]) <= eps
        and first["promotions"] == second["promotions"]
        and first["final_edges"] == second["final_edges"]
        and first["crystal_removed_total"] == second["crystal_removed_total"]
        and first["mask_hash"] == second["mask_hash"]
    )
    return {**first, "deterministic": deterministic}


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for V in sorted({r["V"] for r in results}):
        rows_v = [r for r in results if r["V"] == V]
        summary[str(V)] = {}
        for n_crystals in sorted({r["n_crystals"] for r in rows_v}):
            rows = [r for r in rows_v if r["n_crystals"] == n_crystals]
            summary[str(V)][str(n_crystals)] = {
                "n_cases": len(rows),
                "score_after_growth_mean": float(np.mean([r["score_after_growth"] for r in rows])),
                "score_after_growth_median": float(np.median([r["score_after_growth"] for r in rows])),
                "final_edges_mean": float(np.mean([r["final_edges"] for r in rows])),
                "final_edges_median": float(np.median([r["final_edges"] for r in rows])),
                "wall_ms_mean": float(np.mean([r["wall_ms"] for r in rows])),
                "wall_ms_median": float(np.median([r["wall_ms"] for r in rows])),
                "promotions_mean": float(np.mean([r["promotions"] for r in rows])),
                "crystal_removed_total_mean": float(np.mean([r["crystal_removed_total"] for r in rows])),
                "crystal_passes_total_mean": float(np.mean([r["crystal_passes_total"] for r in rows])),
            }
    return summary


def stage_verdict(summary: dict, results: list[dict], score_tol: float) -> dict:
    out: dict[str, dict[str, object]] = {}
    any_positive = False
    for V_str, row in summary.items():
        baseline = row["0"]
        baseline_score = float(baseline["score_after_growth_median"])
        baseline_edges = float(baseline["final_edges_median"])
        baseline_wall = float(baseline["wall_ms_median"])
        candidates = []
        for n_key, stats in row.items():
            if n_key == "0":
                continue
            rows = [r for r in results if str(r["V"]) == V_str and str(r["n_crystals"]) == n_key]
            deterministic_all = all(bool(r["deterministic"]) for r in rows)
            score = float(stats["score_after_growth_median"])
            edges = float(stats["final_edges_median"])
            wall = float(stats["wall_ms_median"])
            score_ok = score >= baseline_score - score_tol
            positive = (score >= baseline_score + 3e-3) or (score_ok and edges <= 0.9 * baseline_edges)
            bake_ready = positive and deterministic_all and wall <= 1.5 * baseline_wall
            research_only_positive = positive and deterministic_all and wall > 1.5 * baseline_wall
            any_positive = any_positive or positive
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
        out[V_str] = {
            "baseline_no_crystal_score_median": baseline_score,
            "baseline_no_crystal_edges_median": baseline_edges,
            "baseline_no_crystal_wall_median": baseline_wall,
            "candidates": candidates,
        }
    return {"overall_positive": any_positive, "by_v": out}


def main() -> int:
    args = parse_args()
    configs = parse_int_csv_or_default(args.configs, DEFAULT_CONFIGS)
    seeds = parse_int_csv_or_default(args.seeds, DEFAULT_SEEDS)
    crystals = parse_int_csv_or_default(args.crystals, DEFAULT_CRYSTALS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[dict] = []
    for V in configs:
        total_evals = args.total_evals or DEFAULT_TOTAL_EVALS.get(V, DEFAULT_TOTAL_EVALS[64])
        for n_crystals in crystals:
            for seed in seeds:
                row = run_case(
                    V=V,
                    seed=seed,
                    n_crystals=n_crystals,
                    total_evals=total_evals,
                    ticks=args.ticks,
                    eps=args.eps,
                    device=device,
                    compile_crystal_eval=args.compile_crystal_eval,
                )
                results.append(row)
                print(
                    f"V={V:3d} n_crystals={n_crystals:2d} seed={seed:3d} "
                    f"score={row['score_after_growth']*100:6.2f}% "
                    f"edges={row['final_edges']:4d} "
                    f"cr_removed={row['crystal_removed_total']:4d} "
                    f"wall_ms={row['wall_ms']:8.1f}"
                )

    summary = summarize(results)
    verdict = stage_verdict(summary, results, args.score_tol)
    payload = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "branch_intent": "gpu deterministic free crystal frequency A/B",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": verdict,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    output_path = args.output
    if not output_path:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(logs_dir / f"gpu_free_crystal_frequency_ab_{stamp}.json")

    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {output_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
