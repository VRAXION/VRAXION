"""GPU add-only swarm width A/B for the current PassiveIO mainline.

Goal:
  - compare K=1 vs wider batched candidate search on GPU
  - keep the current CPU canonical trainer untouched
  - use empty-start, add-only growth, then pass-based crystal finalization

This is a search-speed probe, not a canonical trainer replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
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
from model.graph import SelfWiringGraph


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

DEFAULT_CONFIGS = (64, 128)
DEFAULT_SEEDS = (42, 77, 123, 321)
DEFAULT_KS = (1, 32, 64)
DEFAULT_TOTAL_EVALS = {
    64: 4096,
    128: 8192,
}


@dataclass
class SwarmRun:
    V: int
    K: int
    seed: int
    best_score_before_crystal: float
    best_score_after_crystal: float
    promotions: int
    candidate_evals: int
    wall_ms: float
    candidate_per_sec: float
    edges_before_crystal: int
    final_edges: int
    crystal_removed: int
    crystal_passes: int
    mask_hash: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64,128", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77,123,321", help="Comma-separated integer seeds")
    ap.add_argument("--ks", default="1,32,64", help="Comma-separated swarm widths")
    ap.add_argument("--total-evals", type=int, default=0, help="Override total candidate evaluations per run")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--verdict-score-tol", type=float, default=5e-3, help="Stage B score tolerance vs K=1 (0.5pp default)")
    ap.add_argument("--compile-crystal-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def make_empty_master(V: int, seed: int, device: torch.device):
    case, _ = make_case(V, seed, device)
    master_mask = torch.zeros((case.H, case.H), dtype=torch.float32, device=device)
    return case, master_mask


def sample_dead_edges(
    rng: random.Random,
    H: int,
    alive_set: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    picked: list[tuple[int, int]] = []
    used = set()
    max_slots = H * (H - 1) - len(alive_set)
    count = min(count, max_slots)
    while len(picked) < count:
        r = rng.randrange(H)
        c = rng.randrange(H)
        edge = (r, c)
        if r == c or edge in alive_set or edge in used:
            continue
        used.add(edge)
        picked.append(edge)
    return picked


def batch_scores(
    candidate_masks: torch.Tensor,
    case,
    ticks: int,
) -> torch.Tensor:
    K = candidate_masks.shape[0]
    charges = torch.zeros((K, case.V, case.H), dtype=torch.float32, device=candidate_masks.device)
    acts = torch.zeros_like(charges)
    projected = case.projected.unsqueeze(0).expand(K, -1, -1)
    targets = case.targets.unsqueeze(0).expand(K, -1)
    row_idx = torch.arange(case.V, device=candidate_masks.device, dtype=torch.long)

    for t in range(ticks):
        if t == 0:
            acts.copy_(projected)
        raw = torch.matmul(acts, candidate_masks)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= case.retain
        acts = torch.clamp(charges - SelfWiringGraph.THRESHOLD, min=0.0)
        charges = torch.clamp(charges, -1.0, 1.0)

    logits = torch.matmul(charges, case.W_out)
    probs = torch.softmax(logits, dim=2)
    preds = torch.argmax(probs, dim=2)
    acc = (preds == targets).to(torch.float32).mean(dim=1)
    tp = probs[:, row_idx, case.targets].mean(dim=1)
    return 0.5 * acc + 0.5 * tp


def run_once(
    V: int,
    K: int,
    seed: int,
    total_evals: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> SwarmRun:
    set_seeds(seed)
    case, master_mask = make_empty_master(V, seed, device)
    master_score = float(batch_scores(master_mask.unsqueeze(0), case, ticks)[0].item())
    alive: list[tuple[int, int]] = []
    alive_set: set[tuple[int, int]] = set()
    rng = random.Random(seed + 400_000 + V * 1000 + K)

    candidate_evals = 0
    promotions = 0
    t0 = time.perf_counter()

    while candidate_evals < total_evals:
        batch_k = min(K, total_evals - candidate_evals)
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

    edges_before_crystal = len(alive)
    score_runner = make_score_runner(case, ticks, device, compile_crystal_eval)
    state = CrystalState(
        mask=master_mask.clone(),
        alive=list(alive),
        alive_set=set(alive_set),
        score=master_score,
    )
    crystal_rng = random.Random(seed + 500_000 + V * 1000 + K)
    crystal = crystal_pass_based(
        state=state,
        score_runner=score_runner,
        rng=crystal_rng,
        eps=eps,
        floor_score=master_score - eps,
        max_passes=0,
        max_wall_ms=0,
    )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return SwarmRun(
        V=V,
        K=K,
        seed=seed,
        best_score_before_crystal=master_score,
        best_score_after_crystal=float(crystal["score_after"]),
        promotions=promotions,
        candidate_evals=candidate_evals,
        wall_ms=wall_ms,
        candidate_per_sec=(candidate_evals / max(wall_ms / 1000.0, 1e-9)),
        edges_before_crystal=edges_before_crystal,
        final_edges=int(crystal["edges_after"]),
        crystal_removed=int(crystal["removed_count"]),
        crystal_passes=int(crystal["passes"]),
        mask_hash=str(crystal["mask_hash"]),
    )


def run_case(
    V: int,
    K: int,
    seed: int,
    total_evals: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_crystal_eval: bool,
) -> dict:
    first = run_once(V, K, seed, total_evals, ticks, eps, device, compile_crystal_eval)
    second = run_once(V, K, seed, total_evals, ticks, eps, device, compile_crystal_eval)
    deterministic = (
        abs(first.best_score_after_crystal - second.best_score_after_crystal) <= eps
        and abs(first.best_score_before_crystal - second.best_score_before_crystal) <= eps
        and first.promotions == second.promotions
        and first.candidate_evals == second.candidate_evals
        and first.final_edges == second.final_edges
        and first.mask_hash == second.mask_hash
    )
    return {
        "V": V,
        "K": K,
        "seed": seed,
        "best_score_before_crystal": first.best_score_before_crystal,
        "best_score_after_crystal": first.best_score_after_crystal,
        "promotions": first.promotions,
        "candidate_evals": first.candidate_evals,
        "wall_ms": first.wall_ms,
        "candidate_per_sec": first.candidate_per_sec,
        "edges_before_crystal": first.edges_before_crystal,
        "final_edges": first.final_edges,
        "crystal_removed": first.crystal_removed,
        "crystal_passes": first.crystal_passes,
        "mask_hash": first.mask_hash,
        "deterministic": deterministic,
    }


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for V in sorted({r["V"] for r in results}):
        summary[str(V)] = {}
        rows_v = [r for r in results if r["V"] == V]
        for K in sorted({r["K"] for r in rows_v}):
            rows = [r for r in rows_v if r["K"] == K]
            summary[str(V)][str(K)] = {
                "n_cases": len(rows),
                "score_before_crystal_mean": float(np.mean([r["best_score_before_crystal"] for r in rows])),
                "score_after_crystal_mean": float(np.mean([r["best_score_after_crystal"] for r in rows])),
                "score_after_crystal_median": float(np.median([r["best_score_after_crystal"] for r in rows])),
                "promotions_mean": float(np.mean([r["promotions"] for r in rows])),
                "wall_ms_mean": float(np.mean([r["wall_ms"] for r in rows])),
                "wall_ms_median": float(np.median([r["wall_ms"] for r in rows])),
                "candidate_per_sec_mean": float(np.mean([r["candidate_per_sec"] for r in rows])),
                "final_edges_mean": float(np.mean([r["final_edges"] for r in rows])),
            }
    return summary


def stage_b_verdict(results: list[dict], score_tol: float) -> dict:
    by_v: dict[str, dict[str, object]] = {}
    overall_pass = True

    for V in sorted({r["V"] for r in results}):
        rows_v = [r for r in results if r["V"] == V]
        baseline_rows = [r for r in rows_v if r["K"] == 1]
        if not baseline_rows:
            continue
        baseline_score = float(np.median([r["best_score_after_crystal"] for r in baseline_rows]))
        baseline_wall = float(np.median([r["wall_ms"] for r in baseline_rows]))
        deterministic_ok = all(r["deterministic"] is True for r in rows_v)

        candidates = []
        passed_any = False
        for K in sorted({r["K"] for r in rows_v if r["K"] > 1}):
            rows = [r for r in rows_v if r["K"] == K]
            score_median = float(np.median([r["best_score_after_crystal"] for r in rows]))
            wall_median = float(np.median([r["wall_ms"] for r in rows]))
            score_ok = score_median >= baseline_score - score_tol
            if V == 64:
                wall_ok = wall_median <= 0.5 * baseline_wall
            else:
                wall_ok = wall_median <= 0.65 * baseline_wall
            passed = score_ok and wall_ok and deterministic_ok
            passed_any = passed_any or passed
            candidates.append(
                {
                    "K": K,
                    "passed": passed,
                    "score_after_crystal_median": score_median,
                    "wall_ms_median": wall_median,
                    "score_ok": score_ok,
                    "wall_ok": wall_ok,
                }
            )

        overall_pass = overall_pass and passed_any and deterministic_ok
        by_v[str(V)] = {
            "baseline_k1_score_median": baseline_score,
            "baseline_k1_wall_median": baseline_wall,
            "deterministic_all": deterministic_ok,
            "passed": passed_any and deterministic_ok,
            "candidates": candidates,
        }

    return {
        "overall_pass": overall_pass,
        "by_v": by_v,
        "gate": {
            "score_after_crystal": f"K>1 median >= K1 median - {score_tol}",
            "wall_v64": "K>1 median wall <= 0.5 * K1 median wall",
            "wall_v128+": "K>1 median wall <= 0.65 * K1 median wall",
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
    ks = parse_csv_ints(args.ks)

    results = []
    for V in configs:
        total_evals = args.total_evals or DEFAULT_TOTAL_EVALS.get(V, max(2048, 64 * V))
        for K in ks:
            for seed in seeds:
                row = run_case(
                    V=V,
                    K=K,
                    seed=seed,
                    total_evals=total_evals,
                    ticks=args.ticks,
                    eps=args.eps,
                    device=device,
                    compile_crystal_eval=args.compile_crystal_eval,
                )
                results.append(row)
                print(
                    f"V={V} K={K} seed={seed} "
                    f"score={row['best_score_after_crystal']*100:6.2f}% "
                    f"promotions={row['promotions']:4d} "
                    f"edges={row['final_edges']:4d} "
                    f"wall_ms={row['wall_ms']:8.1f} "
                    f"cand_s={row['candidate_per_sec']:8.1f}"
                )

    summary = summarize(results)
    verdict = stage_b_verdict(results, args.verdict_score_tol)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "branch_intent": "gpu add-only swarm width A/B for PassiveIO mainline",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": verdict,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_swarm_v1_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
