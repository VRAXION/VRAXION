"""
run_grid3_curriculum.py — Orchestrate the 3x3 grid pattern recognition curriculum.

For each of 10 grid3 pattern tasks, spawn `SEEDS_PER_TASK` parallel cargo subprocesses
running `instnct-core/examples/grid3_curriculum`. Each subprocess writes a trace.json
to `target/grid3_curriculum/<task>/seed_<N>/trace.json`. After all 200 runs complete,
pick the best seed per task (by val_acc, then neurons, then seed), copy the winning
trace to `docs/pages/brain_replay/traces/<task>.json`, and write a `tasks.json` index
alongside. The script also emits an append-only evidence bundle under
`target/grid3_curriculum/<UTC-timestamp>/`.

# IMPORTANT: Why we build BEFORE spawning the pool
We call `cargo build --release --example grid3_curriculum` synchronously ONCE before
creating the ProcessPoolExecutor. Parallel cargo subprocess spawns race on the cargo
lock file (`target/.cargo-lock`) and will serialize anyway on the very first build,
wasting minutes of wall time (worst case: deadlock). By doing a single up-front build,
all downstream `cargo run` calls find the example already compiled and incur zero
rebuild cost. If the build fails we exit 1 immediately — no point fanning out to 200
workers that will all hit the same compile error.

# Scope
- Stdlib only. No click / rich / tqdm. Python 3.8+ compatible (no PEP 604 unions,
  no walrus; typing.Optional / typing.List / typing.Dict for annotations).
- Windows-compatible: all paths go through pathlib.Path, shell=False subprocess.
- Pipeline: (1) build, (2) run 200 jobs in a ProcessPoolExecutor, (3) pick best-of-20
  per task, (4) publish winning traces to docs/pages/brain_replay/, (5) write evidence
  bundle.

# Exit codes
    0  all 10 tasks had at least one successful run with val_acc > 50.0
    1  cargo build failed
    2  at least one task had zero successful runs (all seeds failed)
    3  publish phase failed (cannot write into docs/pages/brain_replay/)

Usage examples:
    python tools/run_grid3_curriculum.py
    python tools/run_grid3_curriculum.py --seeds-per-task 5 --tasks grid3_center
    python tools/run_grid3_curriculum.py --dry-run --seeds-per-task 2 --tasks grid3_center
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "instnct-core"
EXAMPLE = "grid3_curriculum"
DOCS_TRACES_DIR = ROOT / "docs" / "pages" / "brain_replay" / "traces"
DOCS_TASKS_INDEX = ROOT / "docs" / "pages" / "brain_replay" / "tasks.json"
OUT_ROOT = ROOT / "target" / "grid3_curriculum"

BASELINE = 50.0
DEFAULT_SEEDS_PER_TASK = 20
DEFAULT_DATA_SEED = 42
DEFAULT_MAX_NEURONS = 32
SUBPROCESS_TIMEOUT_SEC = 900  # 15 min cap per (task, seed)

TASKS: List[str] = [
    "grid3_horizontal_line",
    "grid3_vertical_line",
    "grid3_diagonal",
    "grid3_center",
    "grid3_corner",
    "grid3_diag_xor",
    "grid3_full_parity",
    "grid3_majority",
    "grid3_symmetry_h",
    "grid3_top_heavy",
]


# ---------------------------------------------------------------------------
# Command construction helpers
# ---------------------------------------------------------------------------

def build_run_cmd(task: str, seed: int, out_dir: Path, data_seed: int, max_neurons: int) -> List[str]:
    """Construct the cargo-run argv for one (task, seed)."""
    return [
        "cargo", "run", "--release", "--example", EXAMPLE, "-p", "instnct-core", "--",
        "--task", task,
        "--search-seed", str(seed),
        "--data-seed", str(data_seed),
        "--out-dir", str(out_dir),
        "--max-neurons", str(max_neurons),
    ]


def build_build_cmd() -> List[str]:
    """Construct the cargo-build argv used ONCE before the pool."""
    return [
        "cargo", "build", "--release", "--example", EXAMPLE, "-p", "instnct-core",
    ]


# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------

def build_example() -> None:
    """
    Run `cargo build --release --example grid3_curriculum -p instnct-core` inside ROOT.
    This must be called ONCE, synchronously, before the ProcessPoolExecutor starts.
    Raises RuntimeError on non-zero exit so main() can translate to exit code 1.
    """
    cmd = build_build_cmd()
    print("[grid3] building example... (" + " ".join(cmd) + ")")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("cargo not found on PATH: " + str(exc))
    if proc.returncode != 0:
        raise RuntimeError(
            "cargo build failed (exit {code})\nSTDOUT:\n{out}\nSTDERR:\n{err}".format(
                code=proc.returncode,
                out=proc.stdout,
                err=proc.stderr,
            )
        )
    print("[grid3] build ok")


# ---------------------------------------------------------------------------
# Per-seed runner (invoked inside worker processes)
# ---------------------------------------------------------------------------

def run_single_seed(
    task: str,
    seed: int,
    out_root: Path,
    data_seed: int,
    max_neurons: int,
) -> Dict[str, object]:
    """
    Spawn one cargo-run subprocess for (task, seed). Returns a flat dict with the
    fields the orchestrator needs for pick_best + evidence bundle. Never raises:
    all failures are recorded in the returned dict via result["error"].
    """
    out_dir = out_root / task / "seed_{}".format(seed)
    result: Dict[str, object] = {
        "task": task,
        "seed": seed,
        "out_dir": str(out_dir),
        "returncode": -1,
        "trace_path": None,
        "best_val_acc": None,
        "best_test_acc": None,
        "total_neurons": None,
        "max_depth": None,
        "stall_count": None,
        "error": None,
        "stdout_tail": "",
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["error"] = "mkdir failed: {}".format(exc)
        return result

    cmd = build_run_cmd(task, seed, out_dir, data_seed, max_neurons)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(CORE),
            text=True,
            capture_output=True,
            check=False,
            shell=False,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        result["error"] = "timeout after {}s".format(exc.timeout)
        return result
    except FileNotFoundError as exc:
        result["error"] = "cargo not found: {}".format(exc)
        return result
    except OSError as exc:
        result["error"] = "subprocess error: {}".format(exc)
        return result

    result["returncode"] = int(proc.returncode)
    if proc.stdout:
        result["stdout_tail"] = proc.stdout[-500:]
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-300:]
        result["error"] = "exit {}: {}".format(proc.returncode, stderr_tail)
        return result

    trace_file = out_dir / "trace.json"
    if not trace_file.exists():
        result["error"] = "trace.json missing at {}".format(trace_file)
        return result
    try:
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = "trace parse failed: {}".format(exc)
        return result

    final = trace.get("final") or {}
    try:
        result["trace_path"] = str(trace_file)
        result["best_val_acc"] = float(final.get("best_val_acc", 0.0))
        result["best_test_acc"] = float(final.get("best_test_acc", 0.0))
        result["total_neurons"] = int(final.get("total_neurons", 0))
        result["max_depth"] = int(final.get("max_depth", 0))
        result["stall_count"] = int(final.get("stall_count", 0))
    except (TypeError, ValueError) as exc:
        result["error"] = "final field cast failed: {}".format(exc)
    return result


# ---------------------------------------------------------------------------
# Best-of-N selection
# ---------------------------------------------------------------------------

def pick_best(task: str, results: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """
    From a list of per-seed result dicts, return the winner or None.

    Tie-break order (spec §Worker HARNESS):
        1. Higher best_val_acc wins.
        2. On tie, fewer total_neurons wins (cheaper network).
        3. On further tie, lower search_seed wins (deterministic).

    Only considers results where returncode == 0, trace_path is set, and
    best_val_acc > BASELINE (i.e. strictly better than random).
    """
    qualified: List[Dict[str, object]] = []
    for r in results:
        if r.get("returncode") != 0:
            continue
        if r.get("trace_path") is None:
            continue
        val = r.get("best_val_acc")
        if val is None:
            continue
        try:
            if float(val) > BASELINE:
                qualified.append(r)
        except (TypeError, ValueError):
            continue
    if not qualified:
        return None
    qualified.sort(key=lambda r: (
        -float(r["best_val_acc"]),       # type: ignore[arg-type]
        int(r["total_neurons"] or 0),    # type: ignore[arg-type]
        int(r["seed"]),                  # type: ignore[arg-type]
    ))
    return qualified[0]


# ---------------------------------------------------------------------------
# Publishing phase
# ---------------------------------------------------------------------------

def publish_traces(winners: Dict[str, Optional[Dict[str, object]]]) -> None:
    """
    Copy each winning trace.json into docs/pages/brain_replay/traces/<task>.json.
    Creates the parent directory if missing. Raises OSError on first failure
    so main() can convert to exit code 3.
    """
    DOCS_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    for task, w in winners.items():
        if w is None:
            continue
        src_str = w.get("trace_path")
        if not src_str:
            continue
        src = Path(str(src_str))
        dst = DOCS_TRACES_DIR / "{}.json".format(task)
        shutil.copyfile(str(src), str(dst))


def write_tasks_index(winners: Dict[str, Optional[Dict[str, object]]]) -> None:
    """
    Write docs/pages/brain_replay/tasks.json with shape:
      {
        "generated_utc": "<iso>",
        "tasks": [ {task, best_val_acc, total_neurons, winner_seed, max_depth}, ... ]
      }

    Order matches the canonical TASKS order from the spec.
    """
    entries: List[Dict[str, object]] = []
    for task in TASKS:
        w = winners.get(task)
        if w is None:
            continue
        entries.append({
            "task": task,
            "best_val_acc": w.get("best_val_acc"),
            "total_neurons": w.get("total_neurons"),
            "max_depth": w.get("max_depth"),
            "winner_seed": w.get("seed"),
        })
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tasks": entries,
    }
    DOCS_TASKS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    DOCS_TASKS_INDEX.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------

def render_summary(
    winners: Dict[str, Optional[Dict[str, object]]],
    all_results: Dict[str, List[Dict[str, object]]],
    seeds_per_task: int,
) -> str:
    lines: List[str] = [
        "# Grid3 Curriculum Summary",
        "",
        "## Winners",
        "",
        "| Task | Val Acc | Test Acc | Neurons | Depth | Winner Seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        w = winners.get(task)
        if w is None:
            lines.append("| {} | N/A | N/A | N/A | N/A | N/A |".format(task))
            continue
        val = w.get("best_val_acc")
        test = w.get("best_test_acc")
        nrn = w.get("total_neurons")
        dep = w.get("max_depth")
        seed = w.get("seed")
        val_s = "{:.1f}".format(float(val)) if val is not None else "N/A"
        test_s = "{:.1f}".format(float(test)) if test is not None else "N/A"
        nrn_s = "{}".format(nrn) if nrn is not None else "N/A"
        dep_s = "{}".format(dep) if dep is not None else "N/A"
        seed_s = "{}".format(seed) if seed is not None else "N/A"
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            task, val_s, test_s, nrn_s, dep_s, seed_s,
        ))
    lines.extend([
        "",
        "## Aggregate",
        "",
        "- total_tasks: {}".format(len(TASKS)),
        "- seeds_per_task: {}".format(seeds_per_task),
        "- total_jobs: {}".format(sum(len(v) for v in all_results.values())),
        "- tasks_solved: {}".format(sum(1 for w in winners.values() if w is not None)),
    ])
    failed_tasks: List[str] = []
    for task in TASKS:
        rs = all_results.get(task, [])
        ok = sum(1 for r in rs if r.get("returncode") == 0 and r.get("trace_path"))
        if ok == 0:
            failed_tasks.append(task)
    lines.extend([
        "- tasks_with_zero_success: {}".format(len(failed_tasks)),
    ])
    if failed_tasks:
        lines.append("")
        lines.append("### Tasks with zero successful runs")
        lines.append("")
        for t in failed_tasks:
            lines.append("- {}".format(t))
    return "\n".join(lines) + "\n"


def write_evidence_bundle(
    report_dir: Path,
    args_namespace: argparse.Namespace,
    winners: Dict[str, Optional[Dict[str, object]]],
    all_results: Dict[str, List[Dict[str, object]]],
    env: Dict[str, object],
    build_cmd: List[str],
    cmd_template: List[str],
) -> None:
    """Write run_cmd.txt, env.json, curriculum_index.json, summary.md."""
    report_dir.mkdir(parents=True, exist_ok=True)
    # run_cmd.txt — the python invocation + the cargo build and cargo run templates
    invoked = "python tools/run_grid3_curriculum.py " + " ".join(
        "--{}={}".format(k.replace("_", "-"), v)
        for k, v in vars(args_namespace).items()
        if v is not None and not isinstance(v, bool)
    )
    run_cmd_lines = [
        invoked,
        "# build command (run once before the pool):",
        " ".join(build_cmd),
        "# run command template (one per (task, seed)):",
        " ".join(cmd_template),
    ]
    (report_dir / "run_cmd.txt").write_text("\n".join(run_cmd_lines) + "\n", encoding="utf-8")
    (report_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

    flat: List[Dict[str, object]] = []
    for task in TASKS:
        flat.extend(all_results.get(task, []))
    (report_dir / "curriculum_index.json").write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "results": flat,
            "winners": {
                task: winners.get(task)
                for task in TASKS
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(
        render_summary(winners, all_results, args_namespace.seeds_per_task),
        encoding="utf-8",
    )


def collect_env(max_workers: int, seeds_per_task: int, tasks_subset: List[str]) -> Dict[str, object]:
    def _probe(cmd: List[str]) -> str:
        try:
            p = subprocess.run(
                cmd, cwd=str(ROOT), text=True, capture_output=True,
                check=False, shell=False,
            )
            return (p.stdout or "").strip()
        except (OSError, FileNotFoundError):
            return "unknown"
    return {
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "cargo_version": _probe(["cargo", "--version"]),
        "rustc_version": _probe(["rustc", "--version"]),
        "max_workers": max_workers,
        "seeds_per_task": seeds_per_task,
        "tasks": tasks_subset,
    }


# ---------------------------------------------------------------------------
# Argparse and main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3x3 grid pattern curriculum: 10 tasks x N seeds in parallel, "
            "pick best-of-N per task, publish winning traces to docs/pages/brain_replay/."
        ),
    )
    parser.add_argument(
        "--seeds-per-task", type=int, default=DEFAULT_SEEDS_PER_TASK,
        help="Seeds per task (default: {}).".format(DEFAULT_SEEDS_PER_TASK),
    )
    parser.add_argument(
        "--data-seed", type=int, default=DEFAULT_DATA_SEED,
        help="Data seed (fixed across search seeds so results are comparable).",
    )
    parser.add_argument(
        "--max-neurons", type=int, default=DEFAULT_MAX_NEURONS,
        help="Max neurons passed through to the cargo example.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Optional subset of tasks to run; defaults to all 10.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help="Override evidence bundle dir (default: target/grid3_curriculum/<timestamp>/).",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Skip the copy into docs/pages/brain_replay/ (useful for local dry runs).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the jobs that would run without spawning cargo.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Task filter
    if args.tasks:
        unknown = [t for t in args.tasks if t not in TASKS]
        if unknown:
            print("ERROR: unknown task(s): {}".format(", ".join(unknown)), file=sys.stderr)
            print("valid tasks: {}".format(", ".join(TASKS)), file=sys.stderr)
            return 2
        tasks_subset = [t for t in TASKS if t in set(args.tasks)]
    else:
        tasks_subset = list(TASKS)

    seeds_per_task = int(args.seeds_per_task)
    if seeds_per_task < 1:
        print("ERROR: --seeds-per-task must be >= 1", file=sys.stderr)
        return 2

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    total_jobs = len(tasks_subset) * seeds_per_task

    # Dry run: print the plan and exit 0
    if args.dry_run:
        print("[grid3] DRY RUN — no subprocesses will be spawned")
        print("[grid3] would build: {}".format(" ".join(build_build_cmd())))
        print("[grid3] tasks ({}): {}".format(len(tasks_subset), ", ".join(tasks_subset)))
        print("[grid3] seeds per task: {}".format(seeds_per_task))
        print("[grid3] total jobs: {}".format(total_jobs))
        print("[grid3] max workers: {}".format(max_workers))
        print("[grid3] out_root: {}".format(OUT_ROOT))
        print("[grid3] docs_traces_dir: {}".format(DOCS_TRACES_DIR))
        for task in tasks_subset:
            for seed in range(1, seeds_per_task + 1):
                out_dir = OUT_ROOT / task / "seed_{}".format(seed)
                cmd = build_run_cmd(task, seed, out_dir, args.data_seed, args.max_neurons)
                print("[dry-run] {}/seed_{}: {}".format(task, seed, " ".join(cmd)))
        print("[grid3] dry-run ok")
        return 0

    # Build phase — one synchronous cargo build before the pool.
    try:
        build_example()
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1

    # Run phase — ProcessPoolExecutor over (task, seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_task_results: Dict[str, List[Dict[str, object]]] = {t: [] for t in tasks_subset}

    print("[grid3] scheduled {} jobs across {} workers".format(total_jobs, max_workers))
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for task in tasks_subset:
            for seed in range(1, seeds_per_task + 1):
                fut = pool.submit(
                    run_single_seed,
                    task,
                    seed,
                    OUT_ROOT,
                    args.data_seed,
                    args.max_neurons,
                )
                futures[fut] = (task, seed)
        for fut in concurrent.futures.as_completed(futures):
            task, seed = futures[fut]
            done += 1
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover — belt and suspenders
                result = {
                    "task": task,
                    "seed": seed,
                    "out_dir": str(OUT_ROOT / task / "seed_{}".format(seed)),
                    "returncode": -1,
                    "trace_path": None,
                    "best_val_acc": None,
                    "best_test_acc": None,
                    "total_neurons": None,
                    "max_depth": None,
                    "stall_count": None,
                    "error": "worker crashed: {}".format(exc),
                    "stdout_tail": "",
                }
            per_task_results[task].append(result)
            if result.get("error"):
                print("[{}/{}] {}/seed_{} FAIL ({})".format(
                    done, total_jobs, task, seed, result["error"],
                ))
            else:
                print("[{}/{}] {}/seed_{} val_acc={} neurons={}".format(
                    done, total_jobs, task, seed,
                    result.get("best_val_acc"),
                    result.get("total_neurons"),
                ))

    # Pick winners
    print("[grid3] picking winners...")
    winners: Dict[str, Optional[Dict[str, object]]] = {}
    for task in tasks_subset:
        winners[task] = pick_best(task, per_task_results[task])

    # Print winners table
    print("[grid3] task                          val   neurons  seed")
    for task in tasks_subset:
        w = winners[task]
        if w is None:
            print("{:<30s} FAIL  (no successful seed)".format(task))
        else:
            print("{:<30s} {:>5.1f}   {:>5}   {:>4}".format(
                task,
                float(w.get("best_val_acc") or 0.0),  # type: ignore[arg-type]
                int(w.get("total_neurons") or 0),     # type: ignore[arg-type]
                int(w.get("seed") or 0),              # type: ignore[arg-type]
            ))

    # Check: every task must have at least ONE successful run (not just a winner)
    zero_success_tasks: List[str] = []
    for task in tasks_subset:
        rs = per_task_results.get(task, [])
        if not any(r.get("returncode") == 0 and r.get("trace_path") for r in rs):
            zero_success_tasks.append(task)

    # Publish phase
    publish_error: Optional[str] = None
    if not args.no_publish:
        try:
            print("[grid3] publishing {} traces".format(
                sum(1 for w in winners.values() if w is not None),
            ))
            publish_traces(winners)
            write_tasks_index(winners)
        except (OSError, RuntimeError) as exc:
            publish_error = "publish failed: {}".format(exc)
            print("ERROR: {}".format(publish_error), file=sys.stderr)
    else:
        print("[grid3] --no-publish: skipping docs/pages/brain_replay/ writes")

    # Evidence bundle
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.report_dir or (OUT_ROOT / timestamp)
    env = collect_env(max_workers, seeds_per_task, tasks_subset)
    cmd_template = build_run_cmd("<task>", 0, OUT_ROOT / "<task>" / "seed_<N>", args.data_seed, args.max_neurons)
    try:
        write_evidence_bundle(
            report_dir=report_dir,
            args_namespace=args,
            winners=winners,
            all_results=per_task_results,
            env=env,
            build_cmd=build_build_cmd(),
            cmd_template=cmd_template,
        )
        print("[grid3] evidence: {}".format(report_dir))
    except OSError as exc:
        print("ERROR: failed to write evidence bundle at {}: {}".format(report_dir, exc), file=sys.stderr)
        # evidence-bundle failure is not fatal — we still return the right status code
        # for termination criteria below.

    # Termination criteria — order matters
    if publish_error is not None:
        return 3
    if zero_success_tasks:
        print("FAIL: {} tasks had zero successful runs: {}".format(
            len(zero_success_tasks), ", ".join(zero_success_tasks),
        ))
        return 2
    solved = sum(1 for w in winners.values() if w is not None)
    if solved < len(tasks_subset):
        print("FAIL: {}/{} tasks solved above baseline".format(solved, len(tasks_subset)))
        return 2
    print("PASS: {}/{} tasks solved".format(solved, len(tasks_subset)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
