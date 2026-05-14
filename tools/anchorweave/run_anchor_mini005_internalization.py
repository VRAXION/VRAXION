#!/usr/bin/env python3
"""Parallel ANCHOR-MINI-005 process-internalization sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_005_INTERNALIZATION_CONTRACT.md")
CARRIERS = [
    "SPARSE_DIRECT",
    "SPARSE_ORACLE_ROUTED",
    "SPARSE_LEARNED_PROCESS",
    "SPARSE_LEARNED_HYBRID",
    "SPARSE_SHUFFLED_PROCESS",
]
CANDIDATE_COUNT = 4
CATEGORY_COUNT = 4
GOAL_COUNT = 16
EFFECT_COUNT = 24

JOB_COMPLETE = "ANCHOR_MINI_005_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_005_JOB_INVALID_STRESS"

STATUS_STRONG = "ANCHOR_MINI_005_INTERNALIZATION_STRONG_POSITIVE"
STATUS_WEAK = "ANCHOR_MINI_005_INTERNALIZATION_WEAK_POSITIVE"
STATUS_ORACLE_ONLY = "ANCHOR_MINI_005_ORACLE_ONLY"
STATUS_NEGATIVE = "ANCHOR_MINI_005_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_005_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_005_RESOURCE_BLOCKED"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return sorted(dict.fromkeys(seeds))


def parse_csv(raw: str, cast: Any = str) -> list[Any]:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-005 internalization sweep.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2125")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--carriers", default=",".join(CARRIERS))
    parser.add_argument("--train-examples", default="1024")
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--surface", default="0.90/0.90")
    parser.add_argument("--max-steps", default="1600")
    parser.add_argument("--proposals", type=int, default=9)
    parser.add_argument("--edge-cap", type=int, default=20)
    parser.add_argument("--aux-weight", type=float, default=4.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None, help="debug cap after queue construction")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed(metrics_path: Path) -> set[str]:
    completed: set[str] = set()
    if not metrics_path.exists():
        return completed
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(row["job_id"])
    return completed


def ensure_binary(root: Path, skip_build: bool) -> Path:
    exe = root / "target" / "release" / "examples" / (
        "evolve_anchor_mini005.exe" if os.name == "nt" else "evolve_anchor_mini005"
    )
    if skip_build:
        if not exe.exists():
            raise FileNotFoundError(f"missing built example: {exe}")
        return exe
    subprocess.run(
        ["cargo", "build", "--release", "-p", "instnct-core", "--example", "evolve_anchor_mini005"],
        cwd=root,
        check=True,
    )
    if not exe.exists():
        raise FileNotFoundError(f"cargo build succeeded but binary is missing: {exe}")
    return exe


def effect_category(effect_id: int) -> int:
    return effect_id % CATEGORY_COUNT


def choose_surface_shortcut(
    *,
    split: str,
    answer_slot: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> int:
    wrong_slots = [slot for slot in range(CANDIDATE_COUNT) if slot != answer_slot]
    if split == "train":
        if rng.random() < train_surface_gold_prob:
            return answer_slot
        return rng.choice(wrong_slots)
    if split == "eval":
        if rng.random() < eval_surface_wrong_prob:
            return rng.choice(wrong_slots)
        return answer_slot
    raise RuntimeError(f"unknown split: {split}")


def make_surface_priors(shortcut_slot: int, rng: random.Random) -> list[float]:
    surface = [0.03 + 0.24 * rng.random() for _ in range(CANDIDATE_COUNT)]
    surface[shortcut_slot] = 0.82 + 0.17 * rng.random()
    return surface


def make_balanced_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> list[dict[str, Any]]:
    effects_by_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) == category]
        for category in range(CATEGORY_COUNT)
    }
    rows: list[dict[str, Any]] = []
    for index in range(count):
        goal_id = rng.randrange(GOAL_COUNT)
        goal_category = goal_id % CATEGORY_COUNT
        answer_slot = rng.randrange(CANDIDATE_COUNT)
        non_goal_categories = [category for category in range(CATEGORY_COUNT) if category != goal_category]
        rng.shuffle(non_goal_categories)
        effect_ids: list[int] = []
        distractor_index = 0
        for slot in range(CANDIDATE_COUNT):
            if slot == answer_slot:
                effect_ids.append(rng.choice(effects_by_category[goal_category]))
            else:
                effect_ids.append(rng.choice(effects_by_category[non_goal_categories[distractor_index]]))
                distractor_index += 1
        effect_categories = [effect_category(effect) for effect in effect_ids]
        match_bits = [int(category == goal_category) for category in effect_categories]
        shortcut = choose_surface_shortcut(
            split=split,
            answer_slot=answer_slot,
            rng=rng,
            train_surface_gold_prob=train_surface_gold_prob,
            eval_surface_wrong_prob=eval_surface_wrong_prob,
        )
        shifted_bits = [int(category == (goal_category + 1) % CATEGORY_COUNT) for category in effect_categories]
        rows.append(
            {
                "example_id": f"{split}_{index:05d}",
                "split": split,
                "goal_id": goal_id,
                "goal_category": goal_category,
                "effect_ids": effect_ids,
                "effect_categories": effect_categories,
                "surface_priors": make_surface_priors(shortcut, rng),
                "answer_label": answer_slot,
                "match_bits": match_bits,
                "surface_shortcut_label": shortcut,
                "surface_shortcut_is_gold": shortcut == answer_slot,
                "shuffled_goal_category": (goal_category + 1) % CATEGORY_COUNT,
                "shuffled_effect_categories": effect_categories,
                "shuffled_match_bits": shifted_bits,
            }
        )
    return rows


def make_dataset(
    *,
    dataset_path: Path,
    seed: int,
    train_examples: int,
    eval_examples: int,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> None:
    if dataset_path.exists():
        return
    rng = random.Random(seed)
    train_rows = make_balanced_examples(
        split="train",
        count=train_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    eval_rows = make_balanced_examples(
        split="eval",
        count=eval_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_path,
        {
            "metadata": {
                "source": "run_anchor_mini005_internalization.py",
                "mini": "ANCHOR-MINI-005",
                "seed": seed,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "train_surface_gold_prob": train_surface_gold_prob,
                "eval_surface_wrong_prob": eval_surface_wrong_prob,
                "category_distractors": "balanced_one_per_non_goal_category",
                "wrong_process_control": "goal_category_plus_one_mod_4",
                "visibility_note": "non-oracle sparse carriers receive raw goal/effect categories, not match_bits",
            },
            "train_examples": train_rows,
            "eval_examples": eval_rows,
        },
    )


def surface_id(raw: str) -> str:
    return raw.replace("/", "_").replace(".", "p")


def job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['carrier']}_seed{job['seed']}_n{job['train_examples']}"
        f"_s{surface_id(job['surface'])}_steps{job['max_steps']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    carriers = parse_csv(args.carriers)
    unknown = sorted(set(carriers) - set(CARRIERS))
    if unknown:
        raise ValueError(f"unknown carriers: {unknown}")
    train_sizes = parse_csv(args.train_examples, int)
    max_steps_values = parse_csv(args.max_steps, int)
    surfaces = parse_csv(args.surface)
    queue: list[dict[str, Any]] = []
    for seed in seeds:
        for carrier in carriers:
            for train_examples in train_sizes:
                for max_steps in max_steps_values:
                    for surface in surfaces:
                        train_prob_raw, eval_prob_raw = surface.split("/", 1)
                        job = {
                            "carrier": carrier,
                            "seed": seed,
                            "train_examples": train_examples,
                            "eval_examples": args.eval_examples,
                            "train_surface_gold_prob": float(train_prob_raw),
                            "eval_surface_wrong_prob": float(eval_prob_raw),
                            "surface": surface,
                            "max_steps": max_steps,
                            "proposals": args.proposals,
                            "edge_cap": args.edge_cap,
                            "aux_weight": args.aux_weight,
                        }
                        job["job_id"] = job_id(job)
                        queue.append(job)
    if args.max_jobs is not None:
        queue = queue[: args.max_jobs]
    return queue


def run_job(
    *,
    root: Path,
    exe: Path,
    out: Path,
    job: dict[str, Any],
    progress_path: Path,
    metrics_path: Path,
    lock: threading.Lock,
) -> dict[str, Any]:
    jid = job["job_id"]
    job_dir = out / "jobs" / jid
    dataset_path = out / "datasets" / (
        f"seed{job['seed']}_n{job['train_examples']}_eval{job['eval_examples']}_s{surface_id(job['surface'])}.json"
    )
    make_dataset(
        dataset_path=dataset_path,
        seed=job["seed"],
        train_examples=job["train_examples"],
        eval_examples=job["eval_examples"],
        train_surface_gold_prob=job["train_surface_gold_prob"],
        eval_surface_wrong_prob=job["eval_surface_wrong_prob"],
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(progress_path, {"event": "job_start", "job_id": jid, "job": job, "time": time.time()}, lock)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    started = time.time()
    proc = subprocess.run(
        [
            str(exe),
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(job_dir),
            "--carrier",
            job["carrier"],
            "--seed",
            str(job["seed"]),
            "--max-steps",
            str(job["max_steps"]),
            "--proposals",
            str(job["proposals"]),
            "--edge-cap",
            str(job["edge_cap"]),
            "--aux-weight",
            str(job["aux_weight"]),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - started
    report_path = job_dir / "report.json"
    if proc.returncode != 0 or not report_path.exists():
        row = {
            "event": "job_complete",
            "job_id": jid,
            "job": job,
            "status": STATUS_BLOCKED,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 3),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-2000:],
            "time": time.time(),
        }
    else:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "event": "job_complete",
            "job_id": jid,
            "job": job,
            "status": payload["status"],
            "elapsed_s": round(elapsed, 3),
            "stress": payload["stress"],
            "metrics": payload["metrics"],
            "feature_visibility": payload["feature_visibility"],
            "edge_count": payload["metrics"]["edge_count"],
            "report_path": str(report_path),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "time": time.time(),
        }
    append_jsonl(metrics_path, row, lock)
    append_jsonl(progress_path, {"event": "job_done", "job_id": jid, "status": row["status"], "time": time.time()}, lock)
    return row


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], queue_size: int, budget_reached: bool) -> dict[str, Any]:
    completed = [row for row in rows if row.get("event") == "job_complete"]
    valid = [row for row in completed if row["status"] == JOB_COMPLETE]
    blocked = [row for row in completed if row["status"] == STATUS_BLOCKED]
    invalid = [row for row in completed if row["status"] == JOB_INVALID]
    by_carrier: dict[str, dict[str, Any]] = {}
    for carrier in CARRIERS:
        subset = [row for row in completed if row["job"]["carrier"] == carrier]
        valid_subset = [row for row in subset if row["status"] == JOB_COMPLETE]
        by_carrier[carrier] = {
            "jobs": len(subset),
            "valid_jobs": len(valid_subset),
            "blocked_jobs": len([row for row in subset if row["status"] == STATUS_BLOCKED]),
            "invalid_jobs": len([row for row in subset if row["status"] == JOB_INVALID]),
            "ood_accuracy": mean([row["metrics"]["answer_eval_ood_accuracy"] for row in valid_subset]),
            "shortcut_trap_rate": mean([row["metrics"]["shortcut_trap_rate"] for row in valid_subset]),
            "process_bit_accuracy": mean([row["metrics"]["process_bit_accuracy"] for row in valid_subset]),
            "process_exact_row_accuracy": mean([row["metrics"]["process_exact_row_accuracy"] for row in valid_subset]),
            "true_process_bit_accuracy": mean([row["metrics"]["true_process_bit_accuracy"] for row in valid_subset]),
            "true_process_exact_row_accuracy": mean([row["metrics"]["true_process_exact_row_accuracy"] for row in valid_subset]),
            "edge_count": mean([row["metrics"]["edge_count"] for row in valid_subset]),
        }
    direct = by_carrier["SPARSE_DIRECT"]
    oracle = by_carrier["SPARSE_ORACLE_ROUTED"]
    learned = by_carrier["SPARSE_LEARNED_PROCESS"]
    hybrid = by_carrier["SPARSE_LEARNED_HYBRID"]
    shuffled = by_carrier["SPARSE_SHUFFLED_PROCESS"]
    valid_seed_count = len({
        row["job"]["seed"]
        for row in valid
        if row["job"]["carrier"] in {"SPARSE_DIRECT", "SPARSE_LEARNED_PROCESS", "SPARSE_SHUFFLED_PROCESS"}
    })
    oracle_gap = oracle["ood_accuracy"] - learned["ood_accuracy"]
    internalization_gap = learned["ood_accuracy"] - direct["ood_accuracy"]
    shuffled_control_gap = learned["ood_accuracy"] - shuffled["ood_accuracy"]
    conditions = {
        "valid_seeds_at_least_80": valid_seed_count >= 80,
        "oracle_upper_bound_good": oracle["ood_accuracy"] >= 0.90 and oracle["shortcut_trap_rate"] <= 0.10,
        "learned_beats_direct_by_0p25": learned["ood_accuracy"] >= direct["ood_accuracy"] + 0.25,
        "learned_beats_shuffled_by_0p25": learned["ood_accuracy"] >= shuffled["ood_accuracy"] + 0.25,
        "learned_trap_rate_le_0p25": learned["shortcut_trap_rate"] <= 0.25,
        "learned_process_true_accuracy_high": learned["true_process_bit_accuracy"] >= 0.90,
        "hybrid_directionally_positive": hybrid["ood_accuracy"] > direct["ood_accuracy"]
        and hybrid["shortcut_trap_rate"] < direct["shortcut_trap_rate"],
        "shuffled_not_reproducing": shuffled["ood_accuracy"] <= learned["ood_accuracy"] - 0.25,
        "no_blocked_jobs": not blocked,
    }
    if not completed:
        final_status = STATUS_BLOCKED
    elif not valid and invalid:
        final_status = STATUS_INVALID
    elif all(conditions.values()):
        final_status = STATUS_STRONG
    elif conditions["oracle_upper_bound_good"] and not (
        conditions["learned_beats_direct_by_0p25"] and conditions["learned_trap_rate_le_0p25"]
    ):
        final_status = STATUS_ORACLE_ONLY
    elif (
        conditions["learned_beats_direct_by_0p25"]
        and conditions["learned_beats_shuffled_by_0p25"]
        and conditions["learned_trap_rate_le_0p25"]
    ):
        final_status = STATUS_WEAK
    elif not valid:
        final_status = STATUS_INVALID
    else:
        final_status = STATUS_NEGATIVE
    return {
        "status": final_status,
        "completed_jobs": len(completed),
        "queue_size": queue_size,
        "budget_reached": budget_reached,
        "valid_jobs": len(valid),
        "invalid_jobs": len(invalid),
        "blocked_jobs": len(blocked),
        "valid_seed_count": valid_seed_count,
        "by_carrier": by_carrier,
        "gaps": {
            "oracle_gap": oracle_gap,
            "internalization_gap": internalization_gap,
            "shuffled_control_gap": shuffled_control_gap,
        },
        "conditions": conditions,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-005 Process Internalization Sweep",
        "",
        f"Status: `{summary['status']}`",
        f"Completed jobs: `{summary['completed_jobs']}` / `{summary['queue_size']}`",
        f"Valid jobs: `{summary['valid_jobs']}`",
        f"Valid seed count: `{summary['valid_seed_count']}`",
        f"Budget reached: `{summary['budget_reached']}`",
        "",
        "## By Carrier",
        "",
        "| carrier | jobs | valid | OOD acc | trap rate | process bit | true process bit | edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for carrier, row in summary["by_carrier"].items():
        lines.append(
            f"| `{carrier}` | {row['jobs']} | {row['valid_jobs']} | "
            f"{row['ood_accuracy']:.3f} | {row['shortcut_trap_rate']:.3f} | "
            f"{row['process_bit_accuracy']:.3f} | {row['true_process_bit_accuracy']:.3f} | "
            f"{row['edge_count']:.1f} |"
        )
    lines.extend(["", "## Gaps", ""])
    for key, value in summary["gaps"].items():
        lines.append(f"- `{key}`: `{value:.3f}`")
    lines.extend(["", "## Conditions", ""])
    for key, value in summary["conditions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a toy process-masking test. It does not prove natural-language AnchorCell behavior, full INSTNCT recurrent behavior, or symbol grounding at scale.",
            "",
            "Oracle carriers are upper bounds only. The internalization claim depends on `SPARSE_LEARNED_PROCESS`, where match bits are labels/fitness targets but not eval inputs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> int:
    root = repo_root()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    contract = root / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, out / "contract_snapshot.md")
    queue = build_queue(args)
    write_json(out / "queue.json", queue)
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    completed = load_completed(metrics_path)
    remaining = [job for job in queue if job["job_id"] not in completed]
    exe = ensure_binary(root, args.skip_build)
    lock = threading.Lock()
    budget_seconds = None
    if args.budget_hours is not None:
        budget_seconds = args.budget_hours * 3600.0
    if args.budget_minutes is not None:
        budget_seconds = args.budget_minutes * 60.0
    started = time.time()
    budget_reached = False
    append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "time": started,
            "jobs": args.jobs,
            "queue_size": len(queue),
            "remaining": len(remaining),
        },
        lock,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        in_flight = {}
        iterator = iter(remaining)
        while True:
            while len(in_flight) < args.jobs:
                if budget_seconds is not None and time.time() - started >= budget_seconds:
                    budget_reached = True
                    break
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                future = executor.submit(
                    run_job,
                    root=root,
                    exe=exe,
                    out=out,
                    job=job,
                    progress_path=progress_path,
                    metrics_path=metrics_path,
                    lock=lock,
                )
                in_flight[future] = job["job_id"]
            if not in_flight:
                break
            for future in as_completed(list(in_flight), timeout=None):
                in_flight.pop(future)
                _ = future.result()
                break
            if budget_reached and not in_flight:
                break
    rows = read_metrics(metrics_path)
    summary = summarize(rows, len(queue), budget_reached)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", summary)
    append_jsonl(progress_path, {"event": "run_done", "summary": summary, "time": time.time()}, lock)
    print(f"wrote {out}")
    print(f"status: {summary['status']}")
    return 0 if summary["status"] != STATUS_BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
