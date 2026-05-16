from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from run_stable_loop_phase_lock_wavefield_probe import (
    CONTRACT as _PREV_CONTRACT,
    PHASE_CLASSES,
    PublicCase,
    aggregate as _unused_aggregate,
    make_cases,
    neighbors_sum,
    phase_probabilities,
    public_arrays,
    unit_phase,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_006_WAVEFIELD_MUTATION_FITNESS_CONTRACT.md"

ARMS = (
    "IDENTITY_BASELINE",
    "ORACLE_COMPLEX_WEIGHTS",
    "HARD_ARGMAX_MUTATION",
    "SOFT_PROB_MUTATION",
    "SOFT_NLL_MUTATION",
)


@dataclass
class CellWeights:
    # features: in_re, in_im, gr, gi, in_re*gr, in_im*gi, in_re*gi, in_im*gr
    re: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    im: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def parse_csv(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in parse_csv(text):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def oracle_weights() -> CellWeights:
    w = CellWeights(re=[0.0] * 8, im=[0.0] * 8)
    w.re[4] = 1.0
    w.re[5] = -1.0
    w.im[6] = 1.0
    w.im[7] = 1.0
    return w


def mutate_weights(weights: CellWeights, rng: random.Random, scale: float) -> CellWeights:
    out = copy.deepcopy(weights)
    target = out.re if rng.random() < 0.5 else out.im
    idx = rng.randrange(len(target))
    target[idx] += rng.gauss(0.0, scale)
    return out


def weight_distance_to_oracle(weights: CellWeights) -> float:
    oracle = oracle_weights()
    diffs = [(a - b) ** 2 for a, b in zip(weights.re, oracle.re)]
    diffs += [(a - b) ** 2 for a, b in zip(weights.im, oracle.im)]
    return float(math.sqrt(sum(diffs)))


def cell_wavefield_target(public: PublicCase, weights: CellWeights, *, steps: int) -> complex:
    free, gate = public_arrays(public)
    state = np.zeros((public.width, public.width), dtype=np.complex64)
    sy, sx = public.source
    ty, tx = public.target
    source_z = unit_phase(public.source_phase)
    gr = gate.real.astype(np.float32)
    gi = gate.imag.astype(np.float32)
    w_re = np.array(weights.re, dtype=np.float32)
    w_im = np.array(weights.im, dtype=np.float32)

    for _ in range(steps):
        incoming = neighbors_sum(state)
        ir = incoming.real.astype(np.float32)
        ii = incoming.imag.astype(np.float32)
        feats = (
            ir,
            ii,
            gr,
            gi,
            ir * gr,
            ii * gi,
            ir * gi,
            ii * gr,
        )
        out_re = sum(w * f for w, f in zip(w_re, feats))
        out_im = sum(w * f for w, f in zip(w_im, feats))
        next_state = (out_re + 1j * out_im) * free
        state = 0.15 * state + next_state
        state[sy, sx] += source_z
        mag = np.abs(state)
        too_big = mag > 4.0
        state[too_big] = state[too_big] / mag[too_big] * 4.0
    return complex(state[ty, tx])


def evaluate_weights(cases: list[Any], weights: CellWeights, steps: int) -> dict[str, float]:
    probs: list[float] = []
    nlls: list[float] = []
    accs: list[int] = []
    margins: list[float] = []
    for bundle in cases:
        z = cell_wavefield_target(bundle.public, weights, steps=steps)
        p = phase_probabilities(z)
        label = bundle.private.label
        pred = max(range(PHASE_CLASSES), key=lambda k: p[k])
        probs.append(p[label])
        nlls.append(-math.log(max(p[label], 1.0e-9)))
        accs.append(int(pred == label))
        ss = sorted(p, reverse=True)
        margins.append(ss[0] - ss[1])
    return {
        "phase_argmax_accuracy": float(np.mean(accs)),
        "correct_phase_probability_at_target": float(np.mean(probs)),
        "target_nll": float(np.mean(nlls)),
        "target_probability_margin": float(np.mean(margins)),
        "weight_distance_to_oracle": weight_distance_to_oracle(weights),
    }


def fitness(metrics: dict[str, float], arm: str) -> float:
    if arm == "HARD_ARGMAX_MUTATION":
        return metrics["phase_argmax_accuracy"]
    if arm == "SOFT_PROB_MUTATION":
        return metrics["correct_phase_probability_at_target"]
    if arm == "SOFT_NLL_MUTATION":
        return -metrics["target_nll"]
    raise ValueError(f"no mutable fitness for {arm}")


def search(job: dict[str, Any], train_cases: list[Any], eval_cases: list[Any]) -> dict[str, Any]:
    out = Path(job["out"])
    arm = job["arm"]
    seed = int(job["seed"])
    steps = int(job["steps"])
    search_steps = int(job["search_steps"])
    checkpoint_interval = int(job["checkpoint_interval"])
    rng = random.Random(seed * 1_000_003 + hash(arm) % 999_983)
    job_id = f"{arm}__seed{seed}"
    progress_path = out / "job_progress" / f"{job_id}.jsonl"
    append_jsonl(progress_path, {"event": "job_start", "job_id": job_id, "time": time.time()})

    if arm == "IDENTITY_BASELINE":
        weights = CellWeights()
        metrics = evaluate_weights(eval_cases, weights, steps)
        metrics.update({"arm": arm, "seed": seed, "job_id": job_id, "accepted_mutations": 0, "evaluated_mutations": 0})
        append_jsonl(progress_path, {"event": "job_done", **metrics, "time": time.time()})
        return metrics
    if arm == "ORACLE_COMPLEX_WEIGHTS":
        weights = oracle_weights()
        metrics = evaluate_weights(eval_cases, weights, steps)
        metrics.update({"arm": arm, "seed": seed, "job_id": job_id, "accepted_mutations": 0, "evaluated_mutations": 0})
        append_jsonl(progress_path, {"event": "job_done", **metrics, "time": time.time()})
        return metrics

    weights = CellWeights()
    train_metrics = evaluate_weights(train_cases, weights, steps)
    best_score = fitness(train_metrics, arm)
    accepted = 0
    scale = float(job["mutation_scale"])
    for step in range(1, search_steps + 1):
        candidate = mutate_weights(weights, rng, scale)
        cand_metrics = evaluate_weights(train_cases, candidate, steps)
        cand_score = fitness(cand_metrics, arm)
        if cand_score >= best_score:
            weights = candidate
            best_score = cand_score
            accepted += 1
            append_jsonl(
                out / "candidate_log.jsonl",
                {
                    "event": "accepted",
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "score": best_score,
                    **cand_metrics,
                    "time": time.time(),
                },
            )
        if step % checkpoint_interval == 0 or step == search_steps:
            eval_metrics = evaluate_weights(eval_cases, weights, steps)
            append_jsonl(
                progress_path,
                {
                    "event": "checkpoint",
                    "job_id": job_id,
                    "step": step,
                    "accepted": accepted,
                    "train_score": best_score,
                    **eval_metrics,
                    "time": time.time(),
                },
            )

    metrics = evaluate_weights(eval_cases, weights, steps)
    metrics.update(
        {
            "arm": arm,
            "seed": seed,
            "job_id": job_id,
            "accepted_mutations": accepted,
            "evaluated_mutations": search_steps,
            "acceptance_rate": accepted / max(1, search_steps),
        }
    )
    append_jsonl(progress_path, {"event": "job_done", **metrics, "time": time.time()})
    return metrics


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    width = int(job["width"])
    train_n = int(job["train_examples"])
    eval_n = int(job["eval_examples"])
    train_cases = make_cases(seed + 100_000, train_n, width)
    eval_cases = make_cases(seed + 200_000, eval_n, width)
    return search(job, train_cases, eval_cases)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    arm_summary = []
    for arm, vals in sorted(by_arm.items()):
        arm_summary.append(
            {
                "arm": arm,
                "jobs": len(vals),
                "phase_argmax_accuracy": float(np.mean([v["phase_argmax_accuracy"] for v in vals])),
                "correct_phase_probability_at_target": float(np.mean([v["correct_phase_probability_at_target"] for v in vals])),
                "target_nll": float(np.mean([v["target_nll"] for v in vals])),
                "target_probability_margin": float(np.mean([v["target_probability_margin"] for v in vals])),
                "weight_distance_to_oracle": float(np.mean([v["weight_distance_to_oracle"] for v in vals])),
                "acceptance_rate": float(np.mean([v.get("acceptance_rate", 0.0) for v in vals])),
            }
        )
    lookup = {r["arm"]: r for r in arm_summary}
    verdicts: list[str] = []
    hard = lookup.get("HARD_ARGMAX_MUTATION")
    soft = lookup.get("SOFT_NLL_MUTATION") or lookup.get("SOFT_PROB_MUTATION")
    oracle = lookup.get("ORACLE_COMPLEX_WEIGHTS")
    if soft and hard and soft["correct_phase_probability_at_target"] >= hard["correct_phase_probability_at_target"] + 0.05:
        verdicts.append("SOFT_WAVEFIELD_FITNESS_BEATS_HARD_ARGMAX")
    if soft and oracle and soft["correct_phase_probability_at_target"] >= oracle["correct_phase_probability_at_target"] - 0.10:
        verdicts.append("MUTATION_APPROACHES_ORACLE_COMPLEX_CELL")
    if soft and soft["phase_argmax_accuracy"] < 0.80:
        verdicts.append("WAVEFIELD_MUTATION_STILL_NOT_SOLVED")
    if not verdicts:
        verdicts.append("NO_CLEAR_WAVEFIELD_MUTATION_ADVANTAGE")
    return {"arm_summary": arm_summary, "verdicts": verdicts, "updated_at": time.time()}


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines = ["# STABLE_LOOP_PHASE_LOCK_006_WAVEFIELD_MUTATION_FITNESS Report", "", "## Verdicts", ""]
    lines += [f"- {v}" for v in summary.get("verdicts", [])]
    lines += [
        "",
        "## Arm Summary",
        "",
        "| Arm | Argmax | Correct probability | NLL | Dist to oracle | Acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("arm_summary", []):
        lines.append(
            "| {arm} | {acc:.3f} | {prob:.3f} | {nll:.3f} | {dist:.3f} | {accept:.3f} |".format(
                arm=row["arm"],
                acc=row["phase_argmax_accuracy"],
                prob=row["correct_phase_probability_at_target"],
                nll=row["target_nll"],
                dist=row["weight_distance_to_oracle"],
                accept=row["acceptance_rate"],
            )
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--search-steps", type=int, default=120)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--arms", default=",".join(ARMS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    jobs = [
        {
            "out": str(out),
            "seed": seed,
            "arm": arm,
            "width": args.width,
            "steps": args.steps,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "search_steps": args.search_steps,
            "mutation_scale": args.mutation_scale,
            "checkpoint_interval": args.checkpoint_interval,
        }
        for seed in seeds
        for arm in arms
    ]
    write_json(out / "queue.json", {"jobs": jobs, "device": args.device, "created_at": time.time()})
    if CONTRACT.exists():
        (out / "contract_snapshot.md").write_text(CONTRACT.read_text(encoding="utf-8"), encoding="utf-8")
    elif _PREV_CONTRACT.exists():
        (out / "contract_snapshot.md").write_text(_PREV_CONTRACT.read_text(encoding="utf-8"), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    last_heartbeat = 0.0
    with cf.ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in cf.as_completed(futures):
            row = future.result()
            rows.append(row)
            append_jsonl(out / "metrics.jsonl", row)
            summary = aggregate(rows)
            summary.update({"completed_jobs": len(rows), "total_jobs": len(jobs)})
            write_json(out / "summary.json", summary)
            write_report(out, summary)
            append_jsonl(out / "progress.jsonl", {"event": "job_done", "job_id": row["job_id"], "completed": len(rows), "total": len(jobs), "time": time.time()})
            now = time.time()
            if now - last_heartbeat >= args.heartbeat_sec:
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(rows), "total": len(jobs), "time": now})
                last_heartbeat = now
    summary = aggregate(rows)
    summary.update({"completed_jobs": len(rows), "total_jobs": len(jobs)})
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
