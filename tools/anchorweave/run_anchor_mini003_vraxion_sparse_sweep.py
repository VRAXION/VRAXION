#!/usr/bin/env python3
"""Parallel ANCHOR-MINI-004 sparse carrier sweep for MINI-003."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
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


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_004_VRAXION_SPARSE_CONTRACT.md")
MINI003_MODULE_DIR = Path(__file__).resolve().parent
CARRIERS = [
    "SPARSE_DIRECT",
    "SPARSE_AUX_DIRECT",
    "SPARSE_ROUTED",
    "SPARSE_HYBRID",
    "SPARSE_SHUFFLED_ROUTED",
]
STATUS_STRONG = "ANCHOR_MINI_004_VRAXION_SPARSE_STRONG_POSITIVE"
STATUS_WEAK = "ANCHOR_MINI_004_VRAXION_SPARSE_WEAK_POSITIVE"
STATUS_NEGATIVE = "ANCHOR_MINI_004_VRAXION_SPARSE_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_004_VRAXION_SPARSE_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_004_VRAXION_SPARSE_RESOURCE_BLOCKED"


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
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-004 sparse carrier sweep.")
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
    parser.add_argument("--edge-cap", type=int, default=16)
    parser.add_argument("--aux-weight", type=float, default=2.0)
    parser.add_argument("--skip-build", action="store_true")
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
    exe = root / "target" / "release" / "examples" / ("evolve_anchor_mini003.exe" if os.name == "nt" else "evolve_anchor_mini003")
    if skip_build:
        if not exe.exists():
            raise FileNotFoundError(f"missing built example: {exe}")
        return exe
    subprocess.run(
        ["cargo", "build", "--release", "-p", "instnct-core", "--example", "evolve_anchor_mini003"],
        cwd=root,
        check=True,
    )
    if not exe.exists():
        raise FileNotFoundError(f"cargo build succeeded but binary is missing: {exe}")
    return exe


def import_mini003() -> Any:
    sys.path.insert(0, str(MINI003_MODULE_DIR))
    import run_anchor_mini003 as mini003  # type: ignore

    return mini003


def make_dataset(
    *,
    mini003: Any,
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
    train_rows = mini003.make_examples(
        split="train",
        count=train_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    eval_rows = mini003.make_examples(
        split="eval",
        count=eval_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    shuffle_rng = random.Random(seed + 700_004)
    shuffled_train = mini003.shuffled_aux_targets(train_rows, shuffle_rng)
    shuffled_eval = mini003.shuffled_aux_targets(eval_rows, shuffle_rng)

    def serialize(rows: list[Any], shuffled: list[tuple[int, list[int], list[int]]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row, aux in zip(rows, shuffled):
            payload = asdict(row)
            payload["shuffled_goal_category"] = aux[0]
            payload["shuffled_effect_categories"] = aux[1]
            payload["shuffled_match_bits"] = aux[2]
            out.append(payload)
        return out

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_path,
        {
            "metadata": {
                "source": "run_anchor_mini003.py",
                "seed": seed,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "train_surface_gold_prob": train_surface_gold_prob,
                "eval_surface_wrong_prob": eval_surface_wrong_prob,
            },
            "train_examples": serialize(train_rows, shuffled_train),
            "eval_examples": serialize(eval_rows, shuffled_eval),
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
    return queue


def run_job(
    *,
    root: Path,
    exe: Path,
    out: Path,
    job: dict[str, Any],
    mini003: Any,
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
        mini003=mini003,
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
    valid = [row for row in completed if row["status"] == "ANCHOR_MINI_004_JOB_COMPLETE"]
    blocked = [row for row in completed if row["status"] == STATUS_BLOCKED]
    by_carrier: dict[str, dict[str, Any]] = {}
    for carrier in CARRIERS:
        subset = [row for row in completed if row["job"]["carrier"] == carrier]
        valid_subset = [row for row in subset if row["status"] == "ANCHOR_MINI_004_JOB_COMPLETE"]
        by_carrier[carrier] = {
            "jobs": len(subset),
            "valid_jobs": len(valid_subset),
            "blocked_jobs": len([row for row in subset if row["status"] == STATUS_BLOCKED]),
            "ood_accuracy": mean([row["metrics"]["answer_eval_ood_accuracy"] for row in valid_subset]),
            "shortcut_trap_rate": mean([row["metrics"]["shortcut_trap_rate"] for row in valid_subset]),
            "process_bit_accuracy": mean([row["metrics"]["process_bit_accuracy"] for row in valid_subset]),
            "edge_count": mean([row["metrics"]["edge_count"] for row in valid_subset]),
        }
    direct = by_carrier["SPARSE_DIRECT"]
    routed = by_carrier["SPARSE_ROUTED"]
    shuffled = by_carrier["SPARSE_SHUFFLED_ROUTED"]
    hybrid = by_carrier["SPARSE_HYBRID"]
    aux_direct = by_carrier["SPARSE_AUX_DIRECT"]
    valid_seed_count = len({
        row["job"]["seed"]
        for row in valid
        if row["job"]["carrier"] in {"SPARSE_DIRECT", "SPARSE_ROUTED", "SPARSE_SHUFFLED_ROUTED"}
    })
    conditions = {
        "valid_seeds_at_least_50": valid_seed_count >= 50,
        "routed_beats_direct_by_0p25": routed["ood_accuracy"] >= direct["ood_accuracy"] + 0.25,
        "routed_beats_shuffled_by_0p25": routed["ood_accuracy"] >= shuffled["ood_accuracy"] + 0.25,
        "routed_trap_rate_le_0p25": routed["shortcut_trap_rate"] <= 0.25,
        "hybrid_directionally_positive": hybrid["ood_accuracy"] > direct["ood_accuracy"]
        and hybrid["shortcut_trap_rate"] < direct["shortcut_trap_rate"],
        "aux_direct_not_equal_routed": aux_direct["ood_accuracy"] < routed["ood_accuracy"] - 0.10,
        "no_blocked_jobs": not blocked,
    }
    if not completed:
        final_status = STATUS_BLOCKED
    elif all(conditions.values()):
        final_status = STATUS_STRONG
    elif (
        conditions["routed_beats_direct_by_0p25"]
        and conditions["routed_beats_shuffled_by_0p25"]
        and conditions["routed_trap_rate_le_0p25"]
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
        "blocked_jobs": len(blocked),
        "valid_seed_count": valid_seed_count,
        "by_carrier": by_carrier,
        "conditions": conditions,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-004 VRAXION Sparse Carrier Sweep",
        "",
        f"Status: `{summary['status']}`",
        f"Completed jobs: `{summary['completed_jobs']}` / `{summary['queue_size']}`",
        f"Valid jobs: `{summary['valid_jobs']}`",
        f"Valid seed count: `{summary['valid_seed_count']}`",
        f"Budget reached: `{summary['budget_reached']}`",
        "",
        "## By Carrier",
        "",
        "| carrier | jobs | valid | ood_acc | trap_rate | process_bit_acc | edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for carrier, row in summary["by_carrier"].items():
        lines.append(
            f"| `{carrier}` | {row['jobs']} | {row['valid_jobs']} | "
            f"{row['ood_accuracy']:.3f} | {row['shortcut_trap_rate']:.3f} | "
            f"{row['process_bit_accuracy']:.3f} | {row['edge_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Conditions",
            "",
        ]
    )
    for key, value in summary["conditions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a toy sparse mutation-selection carrier test. It does not prove natural-language AnchorCell behavior, full INSTNCT recurrent behavior, or symbol grounding at scale.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows = []
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
    mini003 = import_mini003()
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
                    mini003=mini003,
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
