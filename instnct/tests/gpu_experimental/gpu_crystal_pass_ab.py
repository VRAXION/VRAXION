"""GPU crystal A/B for current PassiveIO mainline.

Purpose:
  - keep the current CPU canonical trainer untouched
  - evaluate crystal pruning variants with GPU scoring only
  - compare retry/random crystal against pass-based crystal
  - start from deterministic seeded "winner-like" graphs grown by add-only search

Why this exists:
  CPU results now suggest crystal is not a cosmetic cleanup. A graph that
  survives one safe remove becomes a new system; more edges can become
  removable only after earlier removals are accepted. This makes pass-based
  fixed-point crystal a strong first GPU target before full swarm search.
"""

from __future__ import annotations

import argparse
import hashlib
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

from model.graph import SelfWiringGraph


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

DEFAULT_CONFIGS = (64, 128)
DEFAULT_SEEDS = (42, 77, 123)
DEFAULT_GROW_ATTEMPTS = {
    64: 4096,
    128: 8192,
}


@dataclass
class GpuCase:
    V: int
    H: int
    W_in: torch.Tensor
    W_out: torch.Tensor
    projected: torch.Tensor
    targets: torch.Tensor
    retain: float


@dataclass
class EvalBuffers:
    charges: torch.Tensor
    acts: torch.Tensor
    row_idx: torch.Tensor


@dataclass
class CrystalState:
    mask: torch.Tensor
    alive: list[tuple[int, int]]
    alive_set: set[tuple[int, int]]
    score: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64,128", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77,123", help="Comma-separated integer seeds")
    ap.add_argument("--grow-attempts", type=int, default=0, help="Override seeded add-only growth attempts")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--verdict-score-tol", type=float, default=5e-5, help="Practical score tolerance for Stage A gate")
    ap.add_argument("--compile-eval", action="store_true")
    ap.add_argument("--retry-patience-mode", default="edges", choices=["fixed", "edges"])
    ap.add_argument("--retry-patience", type=int, default=500)
    ap.add_argument("--retry-cap-mode", default="coupon95", choices=["none", "coupon95"])
    ap.add_argument("--max-passes", type=int, default=0, help="Optional hard cap for pass crystal")
    ap.add_argument("--max-wall-ms", type=int, default=0, help="Optional hard wall cap per crystal variant")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_case(V: int, seed: int, device: torch.device) -> tuple[GpuCase, torch.Tensor]:
    set_seeds(seed)
    net = SelfWiringGraph(V)
    perm = np.random.permutation(V).astype(np.int64)
    W_in = torch.from_numpy(net.W_in.copy()).to(device=device, dtype=torch.float32)
    W_out = torch.from_numpy(net.W_out.copy()).to(device=device, dtype=torch.float32)
    projected = W_in.clone()
    targets = torch.from_numpy(perm).to(device=device, dtype=torch.long)
    case = GpuCase(
        V=V,
        H=net.H,
        W_in=W_in,
        W_out=W_out,
        projected=projected,
        targets=targets,
        retain=float(net.retention),
    )
    mask = torch.from_numpy(net.mask.copy()).to(device=device, dtype=torch.float32)
    return case, mask


def make_eval_buffers(case: GpuCase, device: torch.device) -> EvalBuffers:
    return EvalBuffers(
        charges=torch.empty((case.V, case.H), dtype=torch.float32, device=device),
        acts=torch.empty((case.V, case.H), dtype=torch.float32, device=device),
        row_idx=torch.arange(case.V, dtype=torch.long, device=device),
    )


def gpu_score(
    mask: torch.Tensor,
    case: GpuCase,
    ticks: int,
    buffers: EvalBuffers,
) -> float:
    charges = buffers.charges
    acts = buffers.acts
    charges.zero_()
    acts.zero_()
    for t in range(ticks):
        if t == 0:
            acts.copy_(case.projected)
        raw = acts @ mask
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= case.retain
        acts = torch.clamp(charges - SelfWiringGraph.THRESHOLD, min=0.0)
        charges = torch.clamp(charges, -1.0, 1.0)
    logits = charges @ case.W_out
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == case.targets).to(torch.float32).mean()
    tp = probs[buffers.row_idx, case.targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return float(score.item())


def make_score_runner(case: GpuCase, ticks: int, device: torch.device, compile_eval: bool):
    buffers = make_eval_buffers(case, device)

    def score_runner(mask: torch.Tensor) -> torch.Tensor:
        charges = buffers.charges
        acts = buffers.acts
        charges.zero_()
        acts.zero_()
        for t in range(ticks):
            if t == 0:
                acts.copy_(case.projected)
            raw = acts @ mask
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= case.retain
            acts = torch.clamp(charges - SelfWiringGraph.THRESHOLD, min=0.0)
            charges = torch.clamp(charges, -1.0, 1.0)
        logits = charges @ case.W_out
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == case.targets).to(torch.float32).mean()
        tp = probs[buffers.row_idx, case.targets].mean()
        return 0.5 * acc + 0.5 * tp

    if compile_eval:
        compiled = torch.compile(score_runner, mode="reduce-overhead", fullgraph=False)

        def wrapped(mask: torch.Tensor) -> float:
            return float(compiled(mask).item())

        return wrapped

    def eager(mask: torch.Tensor) -> float:
        return gpu_score(mask, case, ticks, buffers)

    return eager


def random_dead_edge(rng: random.Random, H: int, alive_set: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        r = rng.randrange(H)
        c = rng.randrange(H)
        if r != c and (r, c) not in alive_set:
            return r, c


def build_start_state(
    V: int,
    seed: int,
    grow_attempts: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_eval: bool,
) -> tuple[GpuCase, CrystalState, dict]:
    case, accepted_mask = make_case(V, seed, device)
    score_runner = make_score_runner(case, ticks, device, compile_eval)
    alive_np = torch.nonzero(accepted_mask != 0, as_tuple=False).cpu().numpy()
    alive = [(int(r), int(c)) for r, c in alive_np]
    alive_set = set(alive)
    score = score_runner(accepted_mask)

    rng = random.Random(seed + 100_000 + V)
    candidate_mask = accepted_mask.clone()
    kept = 0

    for _ in range(grow_attempts):
        row, col = random_dead_edge(rng, case.H, alive_set)
        sign = SelfWiringGraph.DRIVE if rng.random() > 0.5 else -SelfWiringGraph.DRIVE
        old = float(candidate_mask[row, col].item())
        candidate_mask.copy_(accepted_mask)
        candidate_mask[row, col] = sign
        new_score = score_runner(candidate_mask)
        if new_score > score + eps:
            accepted_mask[row, col] = sign
            alive.append((row, col))
            alive_set.add((row, col))
            score = new_score
            kept += 1
        else:
            candidate_mask[row, col] = old

    state = CrystalState(mask=accepted_mask.clone(), alive=list(alive), alive_set=set(alive_set), score=score)
    meta = {
        "grow_attempts": grow_attempts,
        "kept_adds": kept,
        "edges_after_grow": len(alive),
        "score_after_grow": score,
    }
    return case, state, meta


def clone_state(state: CrystalState) -> CrystalState:
    return CrystalState(
        mask=state.mask.clone(),
        alive=list(state.alive),
        alive_set=set(state.alive_set),
        score=float(state.score),
    )


def retry_attempt_cap(edges_before: int, mode: str) -> int:
    if mode == "none":
        return 0
    if edges_before <= 1:
        return max(1, edges_before)
    return int(math.ceil(edges_before * (math.log(edges_before) + 3.0)))


def retry_patience(edges_before: int, mode: str, fixed_value: int) -> int:
    if mode == "edges":
        return max(1, edges_before)
    return max(1, fixed_value)


def crystal_retry_random(
    state: CrystalState,
    score_runner,
    rng: random.Random,
    eps: float,
    floor_score: float,
    patience_mode: str,
    fixed_patience: int,
    cap_mode: str,
    max_wall_ms: int,
) -> dict:
    edges_before = len(state.alive)
    patience = retry_patience(edges_before, patience_mode, fixed_patience)
    attempt_cap = retry_attempt_cap(edges_before, cap_mode)
    stale = 0
    attempts = 0
    removed = 0
    t0 = time.perf_counter()
    stop_reason = "unknown"

    while state.alive:
        if attempt_cap and attempts >= attempt_cap:
            stop_reason = "attempt_cap"
            break
        if stale >= patience:
            stop_reason = "patience"
            break
        if max_wall_ms and (time.perf_counter() - t0) * 1000.0 >= max_wall_ms:
            stop_reason = "wall_cap"
            break

        idx = rng.randrange(len(state.alive))
        row, col = state.alive[idx]
        old_val = float(state.mask[row, col].item())
        state.mask[row, col] = 0.0
        new_score = score_runner(state.mask)
        attempts += 1
        if new_score >= floor_score:
            state.score = new_score
            last = state.alive[-1]
            state.alive[idx] = last
            state.alive.pop()
            state.alive_set.discard((row, col))
            removed += 1
            stale = 0
        else:
            state.mask[row, col] = old_val
            stale += 1
    else:
        stop_reason = "empty"

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "variant": "retry_random",
        "score_after": state.score,
        "edges_after": len(state.alive),
        "removed_count": removed,
        "removed_pct": (removed / edges_before) if edges_before else 0.0,
        "passes": 0,
        "attempted_removes": attempts,
        "stale_at_end": stale,
        "patience": patience,
        "attempt_cap": attempt_cap,
        "stop_reason": stop_reason,
        "wall_ms": wall_ms,
        "mask_hash": mask_hash(state.mask),
    }


def crystal_pass_based(
    state: CrystalState,
    score_runner,
    rng: random.Random,
    eps: float,
    floor_score: float,
    max_passes: int,
    max_wall_ms: int,
) -> dict:
    edges_before = len(state.alive)
    passes = 0
    attempts = 0
    removed = 0
    t0 = time.perf_counter()
    stop_reason = "zero_remove_pass"

    while True:
        if max_passes and passes >= max_passes:
            stop_reason = "pass_cap"
            break
        if max_wall_ms and (time.perf_counter() - t0) * 1000.0 >= max_wall_ms:
            stop_reason = "wall_cap"
            break
        if not state.alive:
            stop_reason = "empty"
            break

        alive_snapshot = list(state.alive)
        rng.shuffle(alive_snapshot)
        removed_this_pass = 0
        for row, col in alive_snapshot:
            if max_wall_ms and (time.perf_counter() - t0) * 1000.0 >= max_wall_ms:
                stop_reason = "wall_cap"
                break
            if (row, col) not in state.alive_set:
                continue
            old_val = float(state.mask[row, col].item())
            state.mask[row, col] = 0.0
            new_score = score_runner(state.mask)
            attempts += 1
            if new_score >= floor_score:
                state.score = new_score
                state.alive_set.discard((row, col))
                removed += 1
                removed_this_pass += 1
            else:
                state.mask[row, col] = old_val
        # Canonicalize edge order between passes so seeded shuffles stay reproducible.
        state.alive = sorted(state.alive_set)
        passes += 1
        if removed_this_pass == 0:
            stop_reason = "zero_remove_pass"
            break
        if stop_reason == "wall_cap":
            break

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "variant": "pass_based",
        "score_after": state.score,
        "edges_after": len(state.alive),
        "removed_count": removed,
        "removed_pct": (removed / edges_before) if edges_before else 0.0,
        "passes": passes,
        "attempted_removes": attempts,
        "stale_at_end": 0,
        "patience": 0,
        "attempt_cap": 0,
        "stop_reason": stop_reason,
        "wall_ms": wall_ms,
        "mask_hash": mask_hash(state.mask),
    }


def mask_hash(mask: torch.Tensor) -> str:
    arr = mask.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def run_case(
    V: int,
    seed: int,
    grow_attempts: int,
    ticks: int,
    eps: float,
    device: torch.device,
    compile_eval: bool,
    retry_patience_mode: str,
    retry_patience_fixed: int,
    retry_cap_mode: str,
    max_passes: int,
    max_wall_ms: int,
) -> dict:
    case, start_state, grow_meta = build_start_state(
        V=V,
        seed=seed,
        grow_attempts=grow_attempts,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_eval=compile_eval,
    )
    score_runner = make_score_runner(case, ticks, device, compile_eval)
    edges_before = len(start_state.alive)
    score_before = float(start_state.score)
    floor_score = score_before - eps

    retry_rng1 = random.Random(seed + 200_000 + V)
    retry_rng2 = random.Random(seed + 200_000 + V)
    pass_rng1 = random.Random(seed + 300_000 + V)
    pass_rng2 = random.Random(seed + 300_000 + V)

    retry_1 = crystal_retry_random(
        clone_state(start_state), score_runner, retry_rng1, eps, floor_score, retry_patience_mode, retry_patience_fixed, retry_cap_mode, max_wall_ms
    )
    retry_2 = crystal_retry_random(
        clone_state(start_state), score_runner, retry_rng2, eps, floor_score, retry_patience_mode, retry_patience_fixed, retry_cap_mode, max_wall_ms
    )
    pass_1 = crystal_pass_based(
        clone_state(start_state), score_runner, pass_rng1, eps, floor_score, max_passes, max_wall_ms
    )
    pass_2 = crystal_pass_based(
        clone_state(start_state), score_runner, pass_rng2, eps, floor_score, max_passes, max_wall_ms
    )

    if max_wall_ms:
        retry_det = None
        pass_det = None
    else:
        retry_det = (
            abs(retry_1["score_after"] - retry_2["score_after"]) <= eps
            and retry_1["edges_after"] == retry_2["edges_after"]
            and retry_1["removed_count"] == retry_2["removed_count"]
            and retry_1["mask_hash"] == retry_2["mask_hash"]
        )
        pass_det = (
            abs(pass_1["score_after"] - pass_2["score_after"]) <= eps
            and pass_1["edges_after"] == pass_2["edges_after"]
            and pass_1["removed_count"] == pass_2["removed_count"]
            and pass_1["passes"] == pass_2["passes"]
            and pass_1["mask_hash"] == pass_2["mask_hash"]
        )

    return {
        "V": V,
        "seed": seed,
        "grow_meta": grow_meta,
        "score_before": score_before,
        "edges_before": edges_before,
        "retry_random": {**retry_1, "deterministic": retry_det},
        "pass_based": {**pass_1, "deterministic": pass_det},
    }


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for V in sorted({r["V"] for r in results}):
        rows = [r for r in results if r["V"] == V]
        retry_removed = [r["retry_random"]["removed_pct"] for r in rows]
        pass_removed = [r["pass_based"]["removed_pct"] for r in rows]
        retry_time = [r["retry_random"]["wall_ms"] for r in rows]
        pass_time = [r["pass_based"]["wall_ms"] for r in rows]
        retry_attempts = [r["retry_random"]["attempted_removes"] for r in rows]
        pass_attempts = [r["pass_based"]["attempted_removes"] for r in rows]
        retry_score_gain = [r["retry_random"]["score_after"] - r["score_before"] for r in rows]
        pass_score_gain = [r["pass_based"]["score_after"] - r["score_before"] for r in rows]
        summary[str(V)] = {
            "n_cases": len(rows),
            "retry_removed_pct_mean": float(np.mean(retry_removed)),
            "pass_removed_pct_mean": float(np.mean(pass_removed)),
            "retry_removed_pct_median": float(np.median(retry_removed)),
            "pass_removed_pct_median": float(np.median(pass_removed)),
            "retry_wall_ms_mean": float(np.mean(retry_time)),
            "pass_wall_ms_mean": float(np.mean(pass_time)),
            "retry_wall_ms_median": float(np.median(retry_time)),
            "pass_wall_ms_median": float(np.median(pass_time)),
            "retry_attempts_mean": float(np.mean(retry_attempts)),
            "pass_attempts_mean": float(np.mean(pass_attempts)),
            "retry_attempts_median": float(np.median(retry_attempts)),
            "pass_attempts_median": float(np.median(pass_attempts)),
            "retry_score_gain_mean": float(np.mean(retry_score_gain)),
            "pass_score_gain_mean": float(np.mean(pass_score_gain)),
            "pass_minus_retry_removed_pct": float(np.mean(pass_removed) - np.mean(retry_removed)),
        }
    return summary


def stage_a_verdict(results: list[dict], eps: float, score_tol: float) -> dict:
    by_v: dict[str, dict[str, object]] = {}
    overall_pass = True

    for V in sorted({r["V"] for r in results}):
        rows = [r for r in results if r["V"] == V]
        retry_score_after = np.array([r["retry_random"]["score_after"] for r in rows], dtype=np.float64)
        pass_score_after = np.array([r["pass_based"]["score_after"] for r in rows], dtype=np.float64)
        retry_removed = np.array([r["retry_random"]["removed_pct"] for r in rows], dtype=np.float64)
        pass_removed = np.array([r["pass_based"]["removed_pct"] for r in rows], dtype=np.float64)
        retry_attempts = np.array([r["retry_random"]["attempted_removes"] for r in rows], dtype=np.float64)
        pass_attempts = np.array([r["pass_based"]["attempted_removes"] for r in rows], dtype=np.float64)
        retry_wall = np.array([r["retry_random"]["wall_ms"] for r in rows], dtype=np.float64)
        pass_wall = np.array([r["pass_based"]["wall_ms"] for r in rows], dtype=np.float64)

        retry_det_all = all(r["retry_random"]["deterministic"] is True for r in rows)
        pass_det_all = all(r["pass_based"]["deterministic"] is True for r in rows)

        retry_score_median = float(np.median(retry_score_after))
        pass_score_median = float(np.median(pass_score_after))
        retry_removed_median = float(np.median(retry_removed))
        pass_removed_median = float(np.median(pass_removed))
        retry_attempts_median = float(np.median(retry_attempts))
        pass_attempts_median = float(np.median(pass_attempts))
        retry_wall_median = float(np.median(retry_wall))
        pass_wall_median = float(np.median(pass_wall))

        score_ok = pass_score_median >= retry_score_median - score_tol
        removed_ok = pass_removed_median >= retry_removed_median - 0.01
        efficiency_ok = (
            pass_attempts_median <= 0.75 * retry_attempts_median
            or pass_wall_median <= 0.75 * retry_wall_median
        )
        deterministic_ok = retry_det_all and pass_det_all
        passed = score_ok and removed_ok and efficiency_ok and deterministic_ok
        overall_pass = overall_pass and passed

        by_v[str(V)] = {
            "passed": passed,
            "retry_score_after_median": retry_score_median,
            "pass_score_after_median": pass_score_median,
            "retry_removed_pct_median": retry_removed_median,
            "pass_removed_pct_median": pass_removed_median,
            "retry_attempts_median": retry_attempts_median,
            "pass_attempts_median": pass_attempts_median,
            "retry_wall_ms_median": retry_wall_median,
            "pass_wall_ms_median": pass_wall_median,
            "retry_deterministic_all": retry_det_all,
            "pass_deterministic_all": pass_det_all,
            "gate_score_ok": score_ok,
            "gate_removed_ok": removed_ok,
            "gate_efficiency_ok": efficiency_ok,
            "gate_deterministic_ok": deterministic_ok,
        }

    return {
        "overall_pass": overall_pass,
        "by_v": by_v,
        "gate": {
            "score_after": "pass_median >= retry_median - eps",
            "score_after_practical": f"pass_median >= retry_median - {score_tol}",
            "removed_pct": "pass_median >= retry_median - 0.01",
            "efficiency": "pass_attempts_median <= 0.75 * retry_attempts_median OR pass_wall_median <= 0.75 * retry_wall_median",
            "determinism": "all repeated runs deterministic for both variants",
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
    grow_attempts_override = args.grow_attempts

    results = []
    for V in configs:
        grow_attempts = grow_attempts_override or DEFAULT_GROW_ATTEMPTS.get(V, max(2048, 64 * V))
        for seed in seeds:
            row = run_case(
                V=V,
                seed=seed,
                grow_attempts=grow_attempts,
                ticks=args.ticks,
                eps=args.eps,
                device=device,
                compile_eval=args.compile_eval,
                retry_patience_mode=args.retry_patience_mode,
                retry_patience_fixed=args.retry_patience,
                retry_cap_mode=args.retry_cap_mode,
                max_passes=args.max_passes,
                max_wall_ms=args.max_wall_ms,
            )
            results.append(row)
            print(
                f"V={V} seed={seed} "
                f"grow_edges={row['grow_meta']['edges_after_grow']} "
                f"retry_removed={row['retry_random']['removed_pct']*100:5.1f}% "
                f"pass_removed={row['pass_based']['removed_pct']*100:5.1f}% "
                f"retry_ms={row['retry_random']['wall_ms']:7.1f} "
                f"pass_ms={row['pass_based']['wall_ms']:7.1f}"
            )

    summary = summarize(results)
    verdict = stage_a_verdict(results, args.eps, args.verdict_score_tol)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "branch_intent": "gpu crystal A/B for PassiveIO mainline",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": verdict,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_crystal_pass_ab_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
