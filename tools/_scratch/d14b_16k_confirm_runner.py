#!/usr/bin/env python3
"""D14b sharded 16k confirm runner.

Runs the D10r-v8 artifact/state-identity gate on the best D14 non-control
candidate and its strongest control comparator. This is a research runner:
it writes restartable generated output under output/ and does not promote
checkpoints by itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
DEFAULT_OUT = Path("output/phase_d14b_h384_basin_confirm_20260501")
ARTIFACT_CONTROLS = (
    "random_projection_null,"
    "state_shuffle_shared,"
    "state_shuffle_projection_consistent,"
    "no_network_random_state"
)
DEFAULT_TARGETS = {
    "rank06_non_control_9_26_19_53": Path(
        "output/phase_d14_h384_state_anchored_basin_atlas_20260501/"
        "bounded_fast_confirm/quadtree_hot/candidates/top_06.ckpt"
    ),
    "rank02_control_18_54": Path(
        "output/phase_d14_h384_state_anchored_basin_atlas_20260501/"
        "bounded_fast_confirm/quadtree_hot/candidates/top_02.ckpt"
    ),
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_event(out_root: Path, event: dict[str, Any]) -> None:
    path = out_root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": now(), **event}) + "\n")


def chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def rel(path: Path) -> str:
    return str(path)


def parse_targets(spec: str) -> dict[str, Path]:
    if not spec:
        return DEFAULT_TARGETS
    targets: dict[str, Path] = {}
    for part in spec.split(","):
        label, raw_path = part.split("=", 1)
        targets[label.strip()] = Path(raw_path.strip())
    return targets


def target_command(args: argparse.Namespace, target_label: str, target_path: Path, shard_dir: Path, seeds: list[int]) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d10r_hardened_eval.py",
        "--device",
        args.device,
        "--baseline",
        rel(BASELINE),
        "--positive",
        rel(target_path),
        "--positive-label",
        target_label,
        "--eval-len",
        str(args.eval_len),
        "--eval-seeds",
        ",".join(str(seed) for seed in seeds),
        "--control-repeats",
        str(args.control_repeats),
        "--artifact-controls",
        ARTIFACT_CONTROLS,
        "--alternate-baseline-checkpoints",
        "",
        "--no-state-zone-diagnostics",
        "--out",
        str(shard_dir),
    ]


def load_summary(shard_dir: Path) -> dict[str, Any] | None:
    path = shard_dir / "d10r_run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def progress_map(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "D14B H384 BASIN FAMILY CONFIRM MAP",
            "",
            "[1] D13 top_01 package",
            "    DONE",
            "",
            "[2] D14 atlas",
            "    DONE: D14_MULTI_BASIN_SIGNAL at bounded confirm",
            "",
            "[3] D14b 16k/30seed confirm",
            f"    CURRENT: {status.get('completed_shards', 0)}/{status.get('total_shards', 0)} shards",
            f"    verdict: {status.get('verdict', 'RUNNING')}",
            "",
            "[4] Next",
            "    PASS -> H384 basin-family evidence strengthens",
            "    FAIL/control-dominant -> keep D14 as atlas signal only",
            "",
        ]
    )


def write_status(out_root: Path, status: dict[str, Any], wake_reason: str | None = None) -> None:
    write_json(out_root / "status.json", status)
    (out_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
    if wake_reason:
        write_json(
            out_root / "wake_trigger.json",
            {
                "ts": now(),
                "reason": wake_reason,
                "verdict": status.get("verdict"),
                "status_path": str(out_root / "status.json"),
                "progress_map": str(out_root / "progress_map.md"),
            },
        )


def summarize(out_root: Path, targets: dict[str, Path], shards: list[list[int]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target_label in targets:
        for idx, seeds in enumerate(shards):
            shard_dir = out_root / target_label / f"shard_{idx:02d}"
            summary = load_summary(shard_dir)
            if not summary:
                rows.append({"target": target_label, "idx": idx, "seeds": seeds, "status": "pending"})
                continue
            ckpt = summary["checkpoint_summaries"][0]
            rows.append(
                {
                    "target": target_label,
                    "idx": idx,
                    "seeds": seeds,
                    "status": "done",
                    "verdict": summary["verdict"],
                    "artifact_gate_pass": bool(summary.get("artifact_gate_pass")),
                    "trusted_mo_ci_low": float(ckpt["trusted_mo_ci_low"]),
                    "real_mo_delta_ci_low": float(ckpt["real_mo_delta_ci_low"]),
                    "blocking_control_families": summary.get("blocking_control_families", []),
                    "elapsed_s": float(summary.get("elapsed_s", 0.0)),
                }
            )
    done = [row for row in rows if row["status"] == "done"]
    failed = [
        row
        for row in done
        if row.get("verdict") != "D10R_V8_STATE_IDENTITY_PASS" or not row.get("artifact_gate_pass")
    ]
    target_stats: dict[str, dict[str, Any]] = {}
    for target_label in targets:
        target_done = [row for row in done if row["target"] == target_label]
        target_failed = [row for row in failed if row["target"] == target_label]
        target_stats[target_label] = {
            "completed": len(target_done),
            "failed": len(target_failed),
            "min_trusted_mo_ci_low": min((row["trusted_mo_ci_low"] for row in target_done), default=None),
            "min_real_mo_delta_ci_low": min((row["real_mo_delta_ci_low"] for row in target_done), default=None),
            "mean_elapsed_s": (
                sum(float(row["elapsed_s"]) for row in target_done) / len(target_done) if target_done else None
            ),
        }
    total_shards = len(targets) * len(shards)
    if failed:
        verdict = "D14B_16K_SHARDED_FAIL"
    elif len(done) == total_shards:
        non_control = target_stats.get("rank06_non_control_9_26_19_53", {})
        control = target_stats.get("rank02_control_18_54", {})
        if non_control.get("failed", 0) == 0 and control.get("failed", 0) == 0:
            verdict = "D14B_NON_CONTROL_AND_CONTROL_PASS"
        else:
            verdict = "D14B_PARTIAL_PASS"
    else:
        verdict = "RUNNING"
    return {
        "ts": now(),
        "verdict": verdict,
        "eval_len": None,
        "total_shards": total_shards,
        "completed_shards": len(done),
        "failed_shards": len(failed),
        "target_stats": target_stats,
        "shards": rows,
    }


def run_shard(args: argparse.Namespace, out_root: Path, target_label: str, target_path: Path, idx: int, seeds: list[int], status: dict[str, Any]) -> None:
    shard_dir = out_root / target_label / f"shard_{idx:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if load_summary(shard_dir):
        append_event(out_root, {"event": "shard_skip_existing", "target": target_label, "idx": idx, "seeds": seeds})
        return
    cmd = target_command(args, target_label, target_path, shard_dir, seeds)
    append_event(out_root, {"event": "shard_start", "target": target_label, "idx": idx, "seeds": seeds, "cmd": cmd})
    started = time.perf_counter()
    log_path = shard_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        while True:
            code = proc.poll()
            elapsed = time.perf_counter() - started
            status.update(
                {
                    "ts": now(),
                    "verdict": "RUNNING",
                    "active_target": target_label,
                    "active_shard": idx,
                    "active_seeds": seeds,
                    "active_pid": proc.pid,
                    "active_elapsed_s": elapsed,
                }
            )
            write_status(out_root, status, "heartbeat")
            append_event(out_root, {"event": "heartbeat", "target": target_label, "idx": idx, "elapsed_s": elapsed, "pid": proc.pid})
            if code is not None:
                break
            time.sleep(args.heartbeat_s)
    elapsed = time.perf_counter() - started
    append_event(out_root, {"event": "shard_end", "target": target_label, "idx": idx, "returncode": code, "elapsed_s": elapsed})
    if code != 0:
        status.update({"verdict": "D14B_SHARD_PROCESS_FAIL", "failed_target": target_label, "failed_shard": idx})
        write_status(out_root, status, "shard_failed")
        raise SystemExit(f"{target_label}/shard_{idx:02d} failed with code {code}; see {log_path}")


def write_report(out_root: Path, status: dict[str, Any]) -> None:
    lines = [
        "# D14b H384 Basin Confirm Report",
        "",
        f"Verdict: `{status['verdict']}`",
        "",
        f"Completed shards: {status['completed_shards']} / {status['total_shards']}",
        f"Failed shards: {status['failed_shards']}",
        "",
        "## Target Summary",
        "",
        "| target | completed | failed | min trusted CI low | min real CI low | mean elapsed s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, stats in status["target_stats"].items():
        lines.append(
            "| {target} | {completed} | {failed} | {trusted} | {real} | {elapsed} |".format(
                target=target,
                completed=stats["completed"],
                failed=stats["failed"],
                trusted="" if stats["min_trusted_mo_ci_low"] is None else f"{stats['min_trusted_mo_ci_low']:.6f}",
                real="" if stats["min_real_mo_delta_ci_low"] is None else f"{stats['min_real_mo_delta_ci_low']:.6f}",
                elapsed="" if stats["mean_elapsed_s"] is None else f"{stats['mean_elapsed_s']:.1f}",
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This runner compares the D14 non-control child candidate against the strongest D14 control tile.",
        "- A full promotion-grade D14b decision requires all 30 seeds per target.",
        "- Partial runs are timing/evidence probes only.",
    ]
    (out_root / "D14B_H384_BASIN_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--targets", default="")
    parser.add_argument("--eval-len", type=int, default=16000)
    parser.add_argument("--seed-start", type=int, default=972001)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--max-shards-per-target", type=int, default=0, help="0 means run all remaining shards")
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--heartbeat-s", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.targets)
    required = [BASELINE, *targets.values()]
    missing = [str(path) for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"missing required checkpoints: {missing}")

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    shards = chunked(seeds, args.shard_size)
    manifest = {
        "ts": now(),
        "baseline": str(BASELINE),
        "targets": {label: str(path) for label, path in targets.items()},
        "eval_len": args.eval_len,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "shard_size": args.shard_size,
        "control_repeats": args.control_repeats,
        "artifact_controls": ARTIFACT_CONTROLS.split(","),
        "shards": [{"idx": idx, "seeds": shard} for idx, shard in enumerate(shards)],
    }
    write_json(out_root / "manifest.json", manifest)
    status = summarize(out_root, targets, shards)
    status["eval_len"] = args.eval_len
    write_status(out_root, status, "start")
    append_event(out_root, {"event": "runner_start", "status": status})
    if args.dry_run:
        print(json.dumps(status, indent=2))
        write_report(out_root, status)
        return 0

    for target_label, target_path in targets.items():
        ran = 0
        for idx, shard in enumerate(shards):
            if args.max_shards_per_target > 0 and ran >= args.max_shards_per_target:
                break
            if load_summary(out_root / target_label / f"shard_{idx:02d}"):
                continue
            run_shard(args, out_root, target_label, target_path, idx, shard, status)
            ran += 1
            status = summarize(out_root, targets, shards)
            status["eval_len"] = args.eval_len
            write_status(out_root, status, "shard_complete")
            write_report(out_root, status)
            print(json.dumps(status, indent=2), flush=True)
            if status["verdict"] == "D14B_16K_SHARDED_FAIL":
                return 2

    status = summarize(out_root, targets, shards)
    status["eval_len"] = args.eval_len
    if status["verdict"] == "RUNNING":
        status["verdict"] = "D14B_16K_SHARDED_PARTIAL"
    write_status(out_root, status, "runner_stop")
    write_report(out_root, status)
    append_event(out_root, {"event": "runner_stop", "status": status})
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
