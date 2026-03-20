"""Deterministic GPU A/B for state-triggered mid-crystal scheduling.

Goal:
  - compare continuous add-only growth + final crystal against
    stale-triggered mid-crystal policies
  - keep canonical CPU training untouched
  - isolate scheduler quality with K=1 so swarm width does not confound results
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
DEFAULT_TOTAL_EVALS = {
    64: 2048,
    128: 4096,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77,123", help="Comma-separated integer seeds")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--verdict-score-tol", type=float, default=5e-3, help="0.5pp tolerance vs continuous")
    ap.add_argument("--total-evals", type=int, default=0, help="Override total add proposals per run")
    ap.add_argument("--compile-crystal-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def policy_grid_for_v(V: int) -> list[dict[str, object]]:
    return [
        {"policy": "continuous_end", "stale_trigger": 0, "max_mid_crystals": 0},
        {"policy": "stale_once", "stale_trigger": V, "max_mid_crystals": 1},
        {"policy": "stale_once", "stale_trigger": 2 * V, "max_mid_crystals": 1},
        {"policy": "stale_once", "stale_trigger": 4 * V, "max_mid_crystals": 1},
        {"policy": "stale_repeat2", "stale_trigger": V, "max_mid_crystals": 2},
        {"policy": "stale_repeat2", "stale_trigger": 2 * V, "max_mid_crystals": 2},
        {"policy": "stale_repeat2", "stale_trigger": 4 * V, "max_mid_crystals": 2},
    ]


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


def maybe_mid_crystal(
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
    policy: str,
    stale_trigger: int,
    max_mid_crystals: int,
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
    # Keep the add-only proposal stream invariant across policies so scheduler
    # differences come only from mid-crystal intervention, not from different
    # random growth trajectories.
    rng = random.Random(seed + 910_000 + V * 1000)

    promotions = 0
    failed_proposals = 0
    stale = 0
    candidate_evals = 0
    mid_crystals = 0
    mid_crystal_removed_total = 0
    mid_crystal_passes_total = 0
    mid_crystal_events: list[dict[str, object]] = []
    t0 = time.perf_counter()

    while candidate_evals < total_evals:
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
            stale = 0
        else:
            failed_proposals += 1
            stale += 1

        should_crystal = (
            policy != "continuous_end"
            and max_mid_crystals > 0
            and stale_trigger > 0
            and stale >= stale_trigger
            and mid_crystals < max_mid_crystals
            and len(alive) > 0
        )
        if should_crystal:
            crystal_seed = seed + 920_000 + V * 1000 + candidate_evals * 10 + mid_crystals
            master_mask, alive, alive_set, master_score, event = maybe_mid_crystal(
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
                eval_index=candidate_evals,
            )
            mid_crystals += 1
            mid_crystal_removed_total += int(event["removed_count"])
            mid_crystal_passes_total += int(event["passes"])
            mid_crystal_events.append(event)
            stale = 0

    best_score_before_final_crystal = master_score
    edges_before_final_crystal = len(alive)
    final_mask, final_alive, final_alive_set, final_score, final_event = maybe_mid_crystal(
        alive=alive,
        alive_set=alive_set,
        master_mask=master_mask,
        master_score=master_score,
        case=case,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_crystal_eval=compile_crystal_eval,
        crystal_seed=seed + 930_000 + V * 1000,
        eval_index=candidate_evals,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "V": V,
        "seed": seed,
        "policy": policy,
        "stale_trigger": stale_trigger,
        "max_mid_crystals": max_mid_crystals,
        "ticks": ticks,
        "K": 1,
        "total_evals": total_evals,
        "best_score_before_final_crystal": best_score_before_final_crystal,
        "best_score_after_final_crystal": final_score,
        "promotions": promotions,
        "failed_proposals": failed_proposals,
        "candidate_evals": candidate_evals,
        "wall_ms": wall_ms,
        "edges_before_final_crystal": edges_before_final_crystal,
        "final_edges": len(final_alive),
        "mid_crystals": mid_crystals,
        "mid_crystal_removed_total": mid_crystal_removed_total,
        "mid_crystal_passes_total": mid_crystal_passes_total,
        "final_crystal_removed": int(final_event["removed_count"]),
        "final_crystal_passes": int(final_event["passes"]),
        "mask_hash": mask_hash(final_mask),
        "mid_crystal_events": mid_crystal_events,
        "final_crystal_event": final_event,
    }


def run_case(
    *,
    V: int,
    seed: int,
    policy: str,
    stale_trigger: int,
    max_mid_crystals: int,
    total_evals: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> dict:
    first = run_once(
        V=V,
        seed=seed,
        policy=policy,
        stale_trigger=stale_trigger,
        max_mid_crystals=max_mid_crystals,
        total_evals=total_evals,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_crystal_eval=compile_crystal_eval,
    )
    second = run_once(
        V=V,
        seed=seed,
        policy=policy,
        stale_trigger=stale_trigger,
        max_mid_crystals=max_mid_crystals,
        total_evals=total_evals,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_crystal_eval=compile_crystal_eval,
    )

    deterministic = (
        abs(first["best_score_after_final_crystal"] - second["best_score_after_final_crystal"]) <= eps
        and first["final_edges"] == second["final_edges"]
        and first["mid_crystals"] == second["mid_crystals"]
        and first["promotions"] == second["promotions"]
        and first["mask_hash"] == second["mask_hash"]
    )
    first["deterministic"] = deterministic
    return first


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for V in sorted({r["V"] for r in results}):
        summary[str(V)] = {}
        rows_v = [r for r in results if r["V"] == V]
        keys = sorted(
            {
                (r["policy"], int(r["stale_trigger"]), int(r["max_mid_crystals"]))
                for r in rows_v
            },
            key=lambda x: (x[0], x[1], x[2]),
        )
        for policy, stale_trigger, max_mid_crystals in keys:
            rows = [
                r for r in rows_v
                if r["policy"] == policy
                and int(r["stale_trigger"]) == stale_trigger
                and int(r["max_mid_crystals"]) == max_mid_crystals
            ]
            label = f"{policy}@{stale_trigger}x{max_mid_crystals}"
            summary[str(V)][label] = {
                "n_cases": len(rows),
                "score_after_final_crystal_mean": float(np.mean([r["best_score_after_final_crystal"] for r in rows])),
                "score_after_final_crystal_median": float(np.median([r["best_score_after_final_crystal"] for r in rows])),
                "final_edges_mean": float(np.mean([r["final_edges"] for r in rows])),
                "final_edges_median": float(np.median([r["final_edges"] for r in rows])),
                "wall_ms_mean": float(np.mean([r["wall_ms"] for r in rows])),
                "wall_ms_median": float(np.median([r["wall_ms"] for r in rows])),
                "mid_crystals_mean": float(np.mean([r["mid_crystals"] for r in rows])),
                "mid_crystal_removed_total_mean": float(np.mean([r["mid_crystal_removed_total"] for r in rows])),
                "promotions_mean": float(np.mean([r["promotions"] for r in rows])),
            }
    return summary


def stage_verdict(results: list[dict], score_tol: float) -> dict:
    by_v: dict[str, dict[str, object]] = {}
    overall_positive = False

    for V in sorted({r["V"] for r in results}):
        rows_v = [r for r in results if r["V"] == V]
        baseline_rows = [r for r in rows_v if r["policy"] == "continuous_end"]
        baseline_score = float(np.median([r["best_score_after_final_crystal"] for r in baseline_rows]))
        baseline_edges = float(np.median([r["final_edges"] for r in baseline_rows]))
        baseline_wall = float(np.median([r["wall_ms"] for r in baseline_rows]))

        candidates = []
        for policy, stale_trigger, max_mid_crystals in sorted(
            {
                (r["policy"], int(r["stale_trigger"]), int(r["max_mid_crystals"]))
                for r in rows_v
                if r["policy"] != "continuous_end"
            },
            key=lambda x: (x[0], x[1], x[2]),
        ):
            rows = [
                r for r in rows_v
                if r["policy"] == policy
                and int(r["stale_trigger"]) == stale_trigger
                and int(r["max_mid_crystals"]) == max_mid_crystals
            ]
            score_median = float(np.median([r["best_score_after_final_crystal"] for r in rows]))
            edges_median = float(np.median([r["final_edges"] for r in rows]))
            wall_median = float(np.median([r["wall_ms"] for r in rows]))
            deterministic_ok = all(r["deterministic"] is True for r in rows)
            score_ok = score_median >= baseline_score - score_tol
            positive = score_median >= baseline_score + 0.003 or (score_ok and edges_median <= 0.9 * baseline_edges)
            bake_ready = positive and deterministic_ok and wall_median <= 1.5 * baseline_wall
            research_only_positive = positive and deterministic_ok and wall_median > 1.5 * baseline_wall
            overall_positive = overall_positive or positive
            candidates.append(
                {
                    "policy": policy,
                    "stale_trigger": stale_trigger,
                    "max_mid_crystals": max_mid_crystals,
                    "score_after_final_crystal_median": score_median,
                    "final_edges_median": edges_median,
                    "wall_ms_median": wall_median,
                    "deterministic_all": deterministic_ok,
                    "score_ok": score_ok,
                    "positive": positive,
                    "bake_ready": bake_ready,
                    "research_only_positive": research_only_positive,
                }
            )
        by_v[str(V)] = {
            "baseline_continuous_score_median": baseline_score,
            "baseline_continuous_edges_median": baseline_edges,
            "baseline_continuous_wall_median": baseline_wall,
            "candidates": candidates,
        }

    return {
        "overall_positive": overall_positive,
        "by_v": by_v,
        "gate": {
            "score": f"score_after_final_crystal >= continuous_end - {score_tol}",
            "positive": "score >= baseline + 0.003 OR (score within tolerance AND final_edges <= 0.9 * baseline)",
            "wall_guard": "wall_ms <= 1.5 * baseline for bake-ready; otherwise research-only positive",
            "determinism": "all repeated runs deterministic",
        },
    }


def best_v64_candidate(results: list[dict]) -> dict | None:
    verdict = stage_verdict([r for r in results if r["V"] == 64], 5e-3)
    rows = verdict["by_v"].get("64", {}).get("candidates", [])
    positive_rows = [r for r in rows if r["positive"]]
    if not positive_rows:
        return None
    positive_rows.sort(
        key=lambda r: (
            not r["bake_ready"],
            -float(r["score_after_final_crystal_median"]),
            float(r["final_edges_median"]),
            float(r["wall_ms_median"]),
        )
    )
    return positive_rows[0]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    device = torch.device("cuda")
    configs = parse_csv_ints(args.configs)
    seeds = parse_csv_ints(args.seeds)
    results: list[dict] = []

    for V in configs:
        total_evals = args.total_evals or DEFAULT_TOTAL_EVALS.get(V, max(2048, 64 * V))
        for spec in policy_grid_for_v(V):
            for seed in seeds:
                row = run_case(
                    V=V,
                    seed=seed,
                    policy=str(spec["policy"]),
                    stale_trigger=int(spec["stale_trigger"]),
                    max_mid_crystals=int(spec["max_mid_crystals"]),
                    total_evals=total_evals,
                    ticks=args.ticks,
                    eps=args.eps,
                    device=device,
                    compile_crystal_eval=args.compile_crystal_eval,
                )
                results.append(row)
                print(
                    f"V={V} policy={row['policy']} trig={row['stale_trigger']:3d} "
                    f"midmax={row['max_mid_crystals']} seed={seed} "
                    f"score={row['best_score_after_final_crystal']*100:6.2f}% "
                    f"edges={row['final_edges']:4d} "
                    f"mid={row['mid_crystals']} "
                    f"wall_ms={row['wall_ms']:8.1f}"
                )

    summary = summarize(results)
    verdict = stage_verdict(results, args.verdict_score_tol)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "branch_intent": "gpu deterministic state-triggered crystal A/B",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": verdict,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_state_crystal_ab_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
