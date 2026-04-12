from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "instnct-core"
EXAMPLE = "neuron_grower"
DEFAULT_GOLDEN = ROOT / "instnct-core" / "tests" / "fixtures" / "grower_regression_golden.json"

FINAL_RE = re.compile(r"FINAL:\s+(?P<neurons>\d+)\s+neurons,\s+depth=(?P<depth>\d+),\s+hidden=(?P<hidden>\w+)")
SCORES_RE = re.compile(r"train=(?P<train>\d+(?:\.\d+)?)%\s+val=(?P<val>\d+(?:\.\d+)?)%\s+test=(?P<test>\d+(?:\.\d+)?)%")
TIME_RE = re.compile(r"Time:\s+(?P<seconds>\d+(?:\.\d+)?)s")


@dataclass
class TaskSpec:
    name: str
    max_neurons: int
    stall: int


TASKS = [
    TaskSpec("four_parity", 8, 8),
    TaskSpec("four_popcount_2", 8, 8),
    TaskSpec("is_digit_gt_4", 10, 10),
    TaskSpec("diagonal_xor", 10, 10),
    TaskSpec("full_parity_4", 12, 12),
    TaskSpec("digit_parity", 12, 12),
]


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def parse_metrics(stdout: str) -> dict[str, object]:
    final = FINAL_RE.search(stdout)
    scores = SCORES_RE.search(stdout)
    timing = TIME_RE.search(stdout)
    if not final or not scores or not timing:
        raise ValueError(f"failed to parse grower output:\n{stdout}")
    stalled = "Stalled " in stdout
    return {
        "neurons": int(final.group("neurons")),
        "depth": int(final.group("depth")),
        "hidden": final.group("hidden") == "true",
        "train": float(scores.group("train")),
        "val": float(scores.group("val")),
        "test": float(scores.group("test")),
        "seconds": float(timing.group("seconds")),
        "stalled": stalled,
    }


def aggregate(task_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    vals = [m["val"] for m in task_metrics.values()]
    tests = [m["test"] for m in task_metrics.values()]
    neurons = [m["neurons"] for m in task_metrics.values()]
    depths = [m["depth"] for m in task_metrics.values()]
    stalls = sum(1 for m in task_metrics.values() if m["stalled"])
    return {
        "task_count": len(task_metrics),
        "mean_val": round(statistics.mean(vals), 3),
        "median_val": round(statistics.median(vals), 3),
        "max_val": round(max(vals), 3),
        "mean_test": round(statistics.mean(tests), 3),
        "median_test": round(statistics.median(tests), 3),
        "max_test": round(max(tests), 3),
        "mean_neurons": round(statistics.mean(neurons), 3),
        "max_depth": max(depths),
        "stall_count": stalls,
    }


def compare_to_golden(metrics: dict[str, object], golden: dict[str, object]) -> list[str]:
    errors: list[str] = []
    golden_tasks = golden.get("tasks", {})
    for task, exp in golden_tasks.items():
        got = metrics["tasks"].get(task)
        if got is None:
            errors.append(f"missing task in current run: {task}")
            continue
        for key in ("neurons", "depth", "hidden", "train", "val", "test", "stalled"):
            if got[key] != exp[key]:
                errors.append(f"{task}.{key}: expected {exp[key]!r}, got {got[key]!r}")
    golden_summary = golden.get("summary", {})
    for key, exp in golden_summary.items():
        got = metrics["summary"].get(key)
        if got != exp:
            errors.append(f"summary.{key}: expected {exp!r}, got {got!r}")
    return errors


def render_summary(metrics: dict[str, object], golden_errors: list[str]) -> str:
    lines = [
        "# Grower Regression Summary",
        "",
        "## Aggregate",
        "",
        f"- task_count: {metrics['summary']['task_count']}",
        f"- mean_val: {metrics['summary']['mean_val']}",
        f"- median_val: {metrics['summary']['median_val']}",
        f"- max_val: {metrics['summary']['max_val']}",
        f"- mean_test: {metrics['summary']['mean_test']}",
        f"- median_test: {metrics['summary']['median_test']}",
        f"- max_test: {metrics['summary']['max_test']}",
        f"- mean_neurons: {metrics['summary']['mean_neurons']}",
        f"- max_depth: {metrics['summary']['max_depth']}",
        f"- stall_count: {metrics['summary']['stall_count']}",
        "",
        "## Per Task",
        "",
        "| Task | Train | Val | Test | Neurons | Depth | Hidden | Stalled | Seconds |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for task, data in metrics["tasks"].items():
        lines.append(
            f"| {task} | {data['train']:.1f} | {data['val']:.1f} | {data['test']:.1f} | "
            f"{data['neurons']} | {data['depth']} | {data['hidden']} | {data['stalled']} | {data['seconds']:.1f} |"
        )
    lines.extend(["", "## Golden Check", ""])
    if golden_errors:
        lines.append("FAILED")
        lines.append("")
        lines.extend([f"- {err}" for err in golden_errors])
    else:
        lines.append("PASS")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical grower regression sweep and emit an append-only evidence bundle.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output directory. Defaults to target/grower-regression/<timestamp>.")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--write-golden", type=Path, default=None, help="Write the current metrics as the golden file.")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.report_dir or (ROOT / "target" / "grower-regression" / timestamp)
    report_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = report_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    build_cmd = ["cargo", "build", "--example", EXAMPLE, "--release"]
    run(build_cmd, CORE)

    tasks_out: dict[str, dict[str, object]] = {}
    commands: list[str] = []
    for spec in TASKS:
        out_dir = runs_dir / spec.name
        cmd = [
            "cargo", "run", "--example", EXAMPLE, "--release", "--",
            "--task", spec.name,
            "--data-seed", str(args.data_seed),
            "--search-seed", str(args.search_seed),
            "--max-neurons", str(spec.max_neurons),
            "--stall", str(spec.stall),
            "--out-dir", str(out_dir),
        ]
        stdout = run(cmd, CORE)
        (report_dir / f"{spec.name}.stdout.txt").write_text(stdout, encoding="utf-8")
        tasks_out[spec.name] = parse_metrics(stdout)
        commands.append(" ".join(cmd))

    metrics = {
        "timestamp_utc": timestamp,
        "data_seed": args.data_seed,
        "search_seed": args.search_seed,
        "tasks": tasks_out,
        "summary": aggregate(tasks_out),
    }

    env = {
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "cargo_version": run(["cargo", "--version"], ROOT).strip(),
        "rustc_version": run(["rustc", "--version"], ROOT).strip(),
        "workspace_version": os.environ.get("CARGO_PKG_VERSION", ""),
    }

    golden_errors: list[str] = []
    if args.golden.exists():
        golden = json.loads(args.golden.read_text(encoding="utf-8"))
        golden_errors = compare_to_golden(metrics, golden)

    (report_dir / "run_cmd.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (report_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.md").write_text(render_summary(metrics, golden_errors), encoding="utf-8")
    (report_dir / "golden_check.json").write_text(json.dumps({"errors": golden_errors}, indent=2) + "\n", encoding="utf-8")

    if args.write_golden:
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        args.write_golden.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if golden_errors:
        print(render_summary(metrics, golden_errors))
        return 1

    print(render_summary(metrics, golden_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
