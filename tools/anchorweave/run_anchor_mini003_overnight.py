#!/usr/bin/env python3
"""Run an append-only ANCHOR-MINI-003 overnight robustness sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


HEAD_MODES = ["compatibility", "hybrid", "direct_match"]
TRAIN_EXAMPLES = [64, 128, 256, 512, 1024]
HIDDEN_DIMS = [16, 32, 48]
AUX_WEIGHTS = [0.5, 1.0, 2.0]
SURFACE_SETTINGS = [(0.70, 0.70), (0.80, 0.80), (0.90, 0.90), (0.95, 0.95)]
SEED_POOL = list(range(2026, 2226))
STATUSES_POSITIVE = {"ANCHOR_MINI_003_STRONG_POSITIVE", "ANCHOR_MINI_003_WEAK_POSITIVE"}
STATUS_STRONG = "ANCHOR_MINI_003_STRONG_POSITIVE"
STATUS_INVALID = "ANCHOR_MINI_003_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_003_RESOURCE_BLOCKED"


@dataclass(frozen=True)
class Job:
    job_id: str
    seed: int
    answer_head_mode: str
    train_examples: int
    hidden: int
    aux_loss_weight: float
    train_surface_gold_prob: float
    eval_surface_wrong_prob: float


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-003 overnight robustness sweep.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--queue-seed", type=int, default=4242)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.012)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_queue(queue_seed: int) -> list[Job]:
    raw: list[Job] = []
    for head_mode in HEAD_MODES:
        for train_examples in TRAIN_EXAMPLES:
            for hidden in HIDDEN_DIMS:
                for aux_weight in AUX_WEIGHTS:
                    for train_surface, eval_surface in SURFACE_SETTINGS:
                        for seed in SEED_POOL:
                            raw.append(
                                Job(
                                    job_id=(
                                        f"{head_mode}_n{train_examples}_h{hidden}_w{str(aux_weight).replace('.', 'p')}_"
                                        f"s{int(train_surface * 100):02d}_{int(eval_surface * 100):02d}_seed{seed}"
                                    ),
                                    seed=seed,
                                    answer_head_mode=head_mode,
                                    train_examples=train_examples,
                                    hidden=hidden,
                                    aux_loss_weight=aux_weight,
                                    train_surface_gold_prob=train_surface,
                                    eval_surface_wrong_prob=eval_surface,
                                )
                            )
    rng = random.Random(queue_seed)
    rng.shuffle(raw)
    return raw


def completed_job_ids(metrics_path: Path) -> set[str]:
    return {str(row["job_id"]) for row in read_jsonl(metrics_path) if row.get("event") == "job_complete"}


def job_command(args: argparse.Namespace, job: Job, job_out: Path) -> list[str]:
    runner = repo_root() / "tools" / "anchorweave" / "run_anchor_mini003.py"
    return [
        sys.executable,
        str(runner),
        "--out",
        str(job_out),
        "--seeds",
        str(job.seed),
        "--answer-head-mode",
        job.answer_head_mode,
        "--train-examples",
        str(job.train_examples),
        "--eval-examples",
        str(args.eval_examples),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--hidden",
        str(job.hidden),
        "--aux-loss-weight",
        str(job.aux_loss_weight),
        "--train-surface-gold-prob",
        str(job.train_surface_gold_prob),
        "--eval-surface-wrong-prob",
        str(job.eval_surface_wrong_prob),
    ]


def run_job(args: argparse.Namespace, job: Job, out: Path) -> dict[str, Any]:
    job_out = out / "jobs" / job.job_id
    job_out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cmd = job_command(args, job, job_out)
    completed = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report_path = job_out / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = report.get("status", STATUS_BLOCKED)
        aggregate = report.get("aggregate_metrics", {})
    else:
        report = {}
        status = STATUS_BLOCKED
        aggregate = {}
    return {
        "event": "job_complete",
        "job_id": job.job_id,
        "job": asdict(job),
        "status": status,
        "returncode": completed.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "report_path": str(report_path),
        "aggregate_metrics": aggregate,
    }


def bucket_key(row: dict[str, Any], field: str) -> str:
    return str(row["job"][field])


def rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    return sum(int(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def mean_metric(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    values = []
    for row in rows:
        arm_metrics = row.get("aggregate_metrics", {}).get(arm)
        if arm_metrics and metric in arm_metrics:
            values.append(float(arm_metrics[metric]))
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], queue_size: int, budget_reached: bool) -> dict[str, Any]:
    completed = [row for row in rows if row.get("event") == "job_complete"]
    summary: dict[str, Any] = {
        "queue_size": queue_size,
        "completed_jobs": len(completed),
        "budget_reached": budget_reached,
        "strong_positive_rate": rate(completed, lambda row: row.get("status") == STATUS_STRONG),
        "positive_rate": rate(completed, lambda row: row.get("status") in STATUSES_POSITIVE),
        "invalid_rate": rate(completed, lambda row: row.get("status") == STATUS_INVALID),
        "blocked_rate": rate(completed, lambda row: row.get("status") == STATUS_BLOCKED or row.get("returncode") != 0),
        "slices": {},
    }
    for field in ["answer_head_mode", "train_examples", "hidden", "aux_loss_weight", "train_surface_gold_prob"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in completed:
            grouped[bucket_key(row, field)].append(row)
        summary["slices"][field] = {
            key: {
                "jobs": len(group_rows),
                "strong_positive_rate": rate(group_rows, lambda row: row.get("status") == STATUS_STRONG),
                "positive_rate": rate(group_rows, lambda row: row.get("status") in STATUSES_POSITIVE),
                "answer_only_shortcut_trap_rate": mean_metric(group_rows, "ANSWER_ONLY", "shortcut_trap_rate"),
                "anchor_shortcut_trap_rate": mean_metric(group_rows, "ANCHOR_MULTI_TASK", "shortcut_trap_rate"),
                "shuffled_shortcut_trap_rate": mean_metric(group_rows, "SHUFFLED_ANCHOR_MULTI_TASK", "shortcut_trap_rate"),
            }
            for key, group_rows in sorted(grouped.items())
        }
    stable_by_train: list[int] = []
    for key, item in summary["slices"].get("train_examples", {}).items():
        if item["jobs"] >= 3 and item["strong_positive_rate"] >= 0.95:
            stable_by_train.append(int(key))
    summary["minimum_train_examples_for_stable_positive"] = min(stable_by_train) if stable_by_train else None
    return summary


def write_report_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-003 Overnight Robustness Sweep",
        "",
        f"Completed jobs: `{summary['completed_jobs']}` / `{summary['queue_size']}`",
        f"Budget reached: `{summary['budget_reached']}`",
        f"Strong positive rate: `{summary['strong_positive_rate']:.3f}`",
        f"Positive rate: `{summary['positive_rate']:.3f}`",
        f"Invalid rate: `{summary['invalid_rate']:.3f}`",
        "",
        "## By Answer Head Mode",
        "",
        "| mode | jobs | strong | positive | answer_only_trap | anchor_trap | shuffled_trap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary["slices"].get("answer_head_mode", {}).items():
        lines.append(
            f"| `{mode}` | {item['jobs']} | {item['strong_positive_rate']:.3f} | {item['positive_rate']:.3f} | "
            f"{item['answer_only_shortcut_trap_rate']:.3f} | {item['anchor_shortcut_trap_rate']:.3f} | "
            f"{item['shuffled_shortcut_trap_rate']:.3f} |"
        )
    lines.extend(["", "## By Train Examples", ""])
    lines.extend(["| train_examples | jobs | strong | positive | anchor_trap |", "|---:|---:|---:|---:|---:|"])
    for train_examples, item in summary["slices"].get("train_examples", {}).items():
        lines.append(
            f"| {train_examples} | {item['jobs']} | {item['strong_positive_rate']:.3f} | "
            f"{item['positive_rate']:.3f} | {item['anchor_shortcut_trap_rate']:.3f} |"
        )
    lines.extend(["", "## Interpretation Rules", ""])
    lines.extend(
        [
            "- Compatibility-only positives mean the signal works when decisions route through decomposition.",
            "- Hybrid positives mean the signal survives some direct bypass pressure.",
            "- Direct-match positives mean auxiliary supervision transfers even without forced category compatibility.",
            "- Collapses identify useful boundaries, not failures of the validated MINI-003 default.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs_dir = out / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    queue = build_queue(args.queue_seed)
    write_json(out / "queue.json", {"queue_seed": args.queue_seed, "jobs": [asdict(job) for job in queue]})
    budget_s = (args.budget_minutes * 60.0) if args.budget_minutes is not None else (args.budget_hours * 3600.0)
    started = time.time()
    done = completed_job_ids(metrics_path)
    budget_reached = False
    completed_now = 0
    for job in queue:
        if job.job_id in done:
            continue
        if args.max_jobs is not None and completed_now >= args.max_jobs:
            break
        if time.time() - started >= budget_s:
            budget_reached = True
            break
        append_jsonl(progress_path, {"event": "job_start", "job_id": job.job_id, "job": asdict(job), "time": time.time()})
        row = run_job(args, job, out)
        append_jsonl(metrics_path, row)
        append_jsonl(progress_path, {"event": "job_done", "job_id": job.job_id, "status": row["status"], "time": time.time()})
        completed_now += 1
    rows = read_jsonl(metrics_path)
    summary = summarize(rows, len(queue), budget_reached)
    write_json(out / "summary.json", summary)
    write_report_md(out / "report.md", summary)
    print(f"wrote {out}")
    print(f"completed_now: {completed_now}")
    print(f"completed_total: {summary['completed_jobs']}")
    print(f"strong_positive_rate: {summary['strong_positive_rate']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
