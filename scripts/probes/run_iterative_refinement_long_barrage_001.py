#!/usr/bin/env python3
"""ITERATIVE_REFINEMENT_LONG_BARRAGE_001.

Detached-friendly long barrage for the iterative refinement dynamics probe.

Runs many small ITERATIVE_REFINEMENT_DYNAMICS_001 configurations until a time
budget is reached, recording which settings require an external stop and which
settings learn an internal target-hold transition.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "ITERATIVE_REFINEMENT_LONG_BARRAGE_001"
DEFAULT_OUT = Path("target/pilot_wave/iterative_refinement_long_barrage_001/day_run")
RUNNER = REPO_ROOT / "scripts/probes/run_iterative_refinement_dynamics_001.py"
BOUNDARY_TEXT = (
    "ITERATIVE_REFINEMENT_LONG_BARRAGE_001 is a local long-run stress harness for toy iterative state "
    "refinement. It does not claim language understanding, GPT-like readiness, production readiness, "
    "safety alignment, consciousness, or open-domain reasoning."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SystemExit("--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise SystemExit("--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--time-budget-hours", type=float, default=8.5)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--per-run-timeout-sec", type=int, default=1800)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    return args


def base_plan(seed0: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seed = seed0
    for include_hold in [False, True]:
        for max_train_steps in [5, 10, 20, 40, 60]:
            jobs.append(
                {
                    "job_name": f"{'hold' if include_hold else 'nohold'}_mts{max_train_steps}_s{seed}",
                    "seed": seed,
                    "max_train_steps": max_train_steps,
                    "steps": 2500 if max_train_steps <= 20 else 3500,
                    "hidden": 192,
                    "no_stop_extra_ticks": 20,
                    "include_target_hold_examples": include_hold,
                    "phase": "coverage_matrix",
                }
            )
            seed += 1
    for extra_ticks in [50, 100, 250, 500]:
        for include_hold in [False, True]:
            jobs.append(
                {
                    "job_name": f"{'hold' if include_hold else 'nohold'}_extra{extra_ticks}_s{seed}",
                    "seed": seed,
                    "max_train_steps": 60,
                    "steps": 3500,
                    "hidden": 192,
                    "no_stop_extra_ticks": extra_ticks,
                    "include_target_hold_examples": include_hold,
                    "phase": "no_stop_extra_tick_stress",
                }
            )
            seed += 1
    for hidden in [64, 96, 128, 256]:
        for include_hold in [False, True]:
            jobs.append(
                {
                    "job_name": f"{'hold' if include_hold else 'nohold'}_hidden{hidden}_s{seed}",
                    "seed": seed,
                    "max_train_steps": 60,
                    "steps": 3500,
                    "hidden": hidden,
                    "no_stop_extra_ticks": 100,
                    "include_target_hold_examples": include_hold,
                    "phase": "capacity_matrix",
                }
            )
            seed += 1
    return jobs


def random_job(rng: random.Random, index: int) -> dict[str, Any]:
    include_hold = rng.random() < 0.5
    max_train_steps = rng.choice([5, 8, 10, 15, 20, 30, 40, 60])
    hidden = rng.choice([64, 96, 128, 192, 256])
    extra_ticks = rng.choice([20, 50, 100, 250, 500])
    steps = rng.choice([1200, 2000, 2500, 3500, 5000])
    seed = rng.randint(5000, 999999)
    return {
        "job_name": f"adaptive_{index:04d}_{'hold' if include_hold else 'nohold'}_mts{max_train_steps}_h{hidden}_x{extra_ticks}_s{seed}",
        "seed": seed,
        "max_train_steps": max_train_steps,
        "steps": steps,
        "hidden": hidden,
        "no_stop_extra_ticks": extra_ticks,
        "include_target_hold_examples": include_hold,
        "phase": "adaptive_random",
    }


def command_for(job: dict[str, Any], run_out: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--out",
        rel(run_out),
        "--seed",
        str(job["seed"]),
        "--max-train-steps",
        str(job["max_train_steps"]),
        "--steps",
        str(job["steps"]),
        "--hidden",
        str(job["hidden"]),
        "--no-stop-extra-ticks",
        str(job["no_stop_extra_ticks"]),
    ]
    if job["include_target_hold_examples"]:
        cmd.append("--include-target-hold-examples")
    return cmd


def flatten_summary(job: dict[str, Any], run_out: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        **job,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "run_out": rel(run_out),
        "summary_exists": False,
    }
    summary_path = run_out / "summary.json"
    if not summary_path.exists():
        row["status"] = "missing_summary"
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    row.update(
        {
            "summary_exists": True,
            "status": summary.get("status"),
            "verdicts": ",".join(summary.get("verdicts", [])),
            "teacher_forced_transition_accuracy_final": metrics.get("teacher_forced_transition_accuracy_final"),
            "learned_free_run_convergence_rate": metrics.get("learned_free_run_convergence_rate"),
            "learned_final_target_accuracy": metrics.get("learned_final_target_accuracy"),
            "learned_free_run_transition_accuracy": metrics.get("learned_free_run_transition_accuracy"),
            "learned_wrong_direction_rate": metrics.get("learned_wrong_direction_rate"),
            "learned_cycle_rate": metrics.get("learned_cycle_rate"),
            "learned_max_stable_horizon": metrics.get("learned_max_stable_horizon"),
            "no_stop_learned_final_at_stop_state_rate": metrics.get("no_stop_learned_final_at_stop_state_rate"),
            "no_stop_learned_exit_after_stop_state_rate": metrics.get("no_stop_learned_exit_after_stop_state_rate"),
            "no_stop_learned_post_stop_zero_delta_rate": metrics.get("no_stop_learned_post_stop_zero_delta_rate"),
            "no_stop_learned_mean_abs_error_after_extra_ticks": metrics.get("no_stop_learned_mean_abs_error_after_extra_ticks"),
            "no_stop_learned_runaway_rate": metrics.get("no_stop_learned_runaway_rate"),
            "wall_clock_sec": metrics.get("wall_clock_sec"),
        }
    )
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("summary_exists")]
    positives = [row for row in completed if row.get("status") == "positive"]
    internal_hold = [row for row in completed if "NO_STOP_INTERNAL_HOLD_EMERGED" in str(row.get("verdicts", ""))]
    external_required = [row for row in completed if "NO_STOP_EXTERNAL_STOP_REQUIRED" in str(row.get("verdicts", ""))]
    failed = [row for row in completed if row.get("status") != "positive"]
    by_phase: dict[str, dict[str, Any]] = {}
    for phase in sorted({row.get("phase", "") for row in completed}):
        phase_rows = [row for row in completed if row.get("phase") == phase]
        by_phase[phase] = {
            "run_count": len(phase_rows),
            "positive_count": sum(row.get("status") == "positive" for row in phase_rows),
            "internal_hold_count": sum("NO_STOP_INTERNAL_HOLD_EMERGED" in str(row.get("verdicts", "")) for row in phase_rows),
            "external_stop_required_count": sum("NO_STOP_EXTERNAL_STOP_REQUIRED" in str(row.get("verdicts", "")) for row in phase_rows),
        }
    return {
        "run_count": len(rows),
        "completed_count": len(completed),
        "positive_count": len(positives),
        "failed_count": len(failed),
        "internal_hold_count": len(internal_hold),
        "external_stop_required_count": len(external_required),
        "best_no_stop_final_at_stop_state_rate": max(
            [float(row.get("no_stop_learned_final_at_stop_state_rate") or 0.0) for row in completed],
            default=0.0,
        ),
        "worst_positive_no_stop_final_at_stop_state_rate": min(
            [
                float(row.get("no_stop_learned_final_at_stop_state_rate") or 0.0)
                for row in positives
            ],
            default=0.0,
        ),
        "by_phase": by_phase,
    }


def write_report(out: Path, rows: list[dict[str, Any]], agg: dict[str, Any], final: bool) -> None:
    lines = [
        "# ITERATIVE_REFINEMENT_LONG_BARRAGE_001 Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{'completed' if final else 'running'}`",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(agg, indent=2, sort_keys=True),
        "```",
        "",
        "## Recent Runs",
        "",
        "| job | status | hold aug | mts | hidden | extra | no-stop final | exit-after-stop | verdict note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows[-20:]:
        verdict = "internal_hold" if "NO_STOP_INTERNAL_HOLD_EMERGED" in str(row.get("verdicts", "")) else "external_required"
        lines.append(
            "| {job_name} | {status} | {include_target_hold_examples} | {max_train_steps} | {hidden} | "
            "{no_stop_extra_ticks} | {final:.3f} | {exit:.3f} | {verdict} |".format(
                job_name=row.get("job_name"),
                status=row.get("status"),
                include_target_hold_examples=row.get("include_target_hold_examples"),
                max_train_steps=row.get("max_train_steps"),
                hidden=row.get("hidden"),
                no_stop_extra_ticks=row.get("no_stop_extra_ticks"),
                final=float(row.get("no_stop_learned_final_at_stop_state_rate") or 0.0),
                exit=float(row.get("no_stop_learned_exit_after_stop_state_rate") or 0.0),
                verdict=verdict,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The barrage maps stability and stop behavior for a toy transition loop. It does not prove open-domain reasoning.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def main() -> int:
    args = parse_args()
    out: Path = args.out
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.time_budget_hours * 3600.0
    rng = random.Random(args.seed)
    planned = base_plan(args.seed)
    write_json(
        out / "queue.json",
        {
            "schema_version": "iterative_refinement_long_barrage_queue_v1",
            "milestone": MILESTONE,
            "time_budget_hours": args.time_budget_hours,
            "seed": args.seed,
            "planned_jobs": planned,
            "boundary": BOUNDARY_TEXT,
        },
    )
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "start", "out": rel(out), "budget_hours": args.time_budget_hours})
    rows: list[dict[str, Any]] = []
    job_index = 0
    while time.time() < deadline:
        if job_index < len(planned):
            job = planned[job_index]
        else:
            job = random_job(rng, job_index)
        job_index += 1
        run_out = runs_dir / f"{job_index:04d}_{job['job_name']}"
        cmd = command_for(job, run_out)
        append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "job_start", "job_index": job_index, "job": job, "cmd": cmd})
        started = time.time()
        log_path = run_out / "subprocess.log"
        run_out.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.per_run_timeout_sec,
                    check=False,
                )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "job_timeout", "job_index": job_index, "job_name": job["job_name"]})
        elapsed = time.time() - started
        row = flatten_summary(job, run_out, returncode, elapsed)
        rows.append(row)
        append_jsonl(out / "run_metrics.jsonl", row)
        agg = aggregate(rows)
        write_json(out / "summary.json", {"schema_version": "iterative_refinement_long_barrage_summary_v1", "status": "running", "aggregate": agg, "latest": row})
        write_csv(out / "metrics.csv", rows)
        write_report(out, rows, agg, final=False)
        append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "job_end", "job_index": job_index, "row": row})
    agg = aggregate(rows)
    write_json(
        out / "summary.json",
        {
            "schema_version": "iterative_refinement_long_barrage_summary_v1",
            "status": "completed",
            "aggregate": agg,
            "completed_at": utc_now(),
            "boundary": BOUNDARY_TEXT,
        },
    )
    write_csv(out / "metrics.csv", rows)
    write_report(out, rows, agg, final=True)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "completed", "aggregate": agg})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
