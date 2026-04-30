#!/usr/bin/env python3
"""Shard runner for the D10u top_01 16k promotion confirm.

This is a research runner, not release code. It runs the D10r-v8 artifact gate
on the exported D10u top_01 checkpoint in restartable shards and writes
append-only status/progress files under output/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
TARGET = Path(
    "output/phase_d10u_focused_ladder_20260430/bounded/"
    "candidates/top_01_seed_2042_edge_threshold_coadapted.ckpt"
)
DEFAULT_OUT = Path("output/phase_d10u_top01_d10r_confirm_20260430/confirm_16000_30seed_sharded")
ARTIFACT_CONTROLS = (
    "random_projection_null,"
    "state_shuffle_shared,"
    "state_shuffle_projection_consistent,"
    "no_network_random_state"
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_event(out_root: Path, event: dict) -> None:
    path = out_root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": now(), **event}) + "\n")


def chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def progress_map(status: dict) -> str:
    return "\n".join(
        [
            "D10U TOP_01 RELEASE-CANDIDATE CONFIRM MAP",
            "",
            "[1] top_01 short scout",
            "    DONE: STRICT_TRUSTED at eval_len=128",
            "",
            "[2] bounded confirm",
            "    DONE: D10R_V8_STATE_IDENTITY_PASS at eval_len=1000",
            "",
            "[3] longer confirm",
            "    DONE: D10R_V8_STATE_IDENTITY_PASS at eval_len=4000",
            "",
            "[4] promotion-grade confirm",
            f"    CURRENT: {status.get('completed_shards', 0)}/{status.get('total_shards', 0)} shards",
            f"    verdict: {status.get('verdict', 'RUNNING')}",
            "",
            "[5] release-candidate package",
            "    BLOCKED until 16k/30-seed shard set passes",
            "",
        ]
    )


def write_status(out_root: Path, status: dict, wake_reason: str | None = None) -> None:
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


def shard_command(args: argparse.Namespace, shard_dir: Path, seeds: list[int]) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d10r_hardened_eval.py",
        "--device",
        args.device,
        "--baseline",
        str(BASELINE),
        "--positive",
        str(TARGET),
        "--positive-label",
        "d10u_top01_state_anchored",
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


def load_shard_summary(shard_dir: Path) -> dict | None:
    path = shard_dir / "d10r_run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_existing(out_root: Path, shards: list[list[int]]) -> dict:
    rows = []
    for idx, seeds in enumerate(shards):
        shard_dir = out_root / f"shard_{idx:02d}"
        summary = load_shard_summary(shard_dir)
        if not summary:
            rows.append({"idx": idx, "seeds": seeds, "status": "pending"})
            continue
        ckpt = summary["checkpoint_summaries"][0]
        rows.append(
            {
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
    if failed:
        verdict = "D10U_TOP01_16K_SHARDED_FAIL"
    elif len(done) == len(shards):
        verdict = "D10U_TOP01_16K_SHARDED_PASS"
    else:
        verdict = "RUNNING"
    return {
        "ts": now(),
        "verdict": verdict,
        "total_shards": len(shards),
        "completed_shards": len(done),
        "failed_shards": len(failed),
        "min_trusted_mo_ci_low": min((row["trusted_mo_ci_low"] for row in done), default=None),
        "min_real_mo_delta_ci_low": min((row["real_mo_delta_ci_low"] for row in done), default=None),
        "shards": rows,
    }


def run_shard(args: argparse.Namespace, out_root: Path, idx: int, seeds: list[int], status: dict) -> None:
    shard_dir = out_root / f"shard_{idx:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if load_shard_summary(shard_dir):
        append_event(out_root, {"event": "shard_skip_existing", "idx": idx, "seeds": seeds})
        return

    cmd = shard_command(args, shard_dir, seeds)
    append_event(out_root, {"event": "shard_start", "idx": idx, "seeds": seeds, "cmd": cmd})
    log_path = shard_dir / "run.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            code = proc.poll()
            elapsed = time.perf_counter() - started
            status.update(
                {
                    "ts": now(),
                    "verdict": "RUNNING",
                    "active_shard": idx,
                    "active_seeds": seeds,
                    "active_pid": proc.pid,
                    "active_elapsed_s": elapsed,
                }
            )
            write_status(out_root, status, "heartbeat")
            append_event(out_root, {"event": "heartbeat", "idx": idx, "elapsed_s": elapsed, "pid": proc.pid})
            if code is not None:
                break
            time.sleep(args.heartbeat_s)
    elapsed = time.perf_counter() - started
    append_event(out_root, {"event": "shard_end", "idx": idx, "returncode": code, "elapsed_s": elapsed})
    if code != 0:
        status.update({"verdict": "D10U_TOP01_16K_SHARD_PROCESS_FAIL", "failed_shard": idx})
        write_status(out_root, status, "shard_failed")
        raise SystemExit(f"shard_{idx:02d} failed with code {code}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-len", type=int, default=16000)
    parser.add_argument("--seed-start", type=int, default=970101)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means run all remaining shards")
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--heartbeat-s", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in [BASELINE, TARGET] if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"missing required checkpoints: {missing}")

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    shards = chunked(seeds, args.shard_size)
    manifest = {
        "ts": now(),
        "baseline": str(BASELINE),
        "target": str(TARGET),
        "eval_len": args.eval_len,
        "seed_count": args.seed_count,
        "shard_size": args.shard_size,
        "control_repeats": args.control_repeats,
        "artifact_controls": ARTIFACT_CONTROLS.split(","),
        "state_zone_diagnostics": False,
        "shards": [{"idx": idx, "seeds": shard} for idx, shard in enumerate(shards)],
    }
    write_json(out_root / "manifest.json", manifest)

    status = summarize_existing(out_root, shards)
    write_status(out_root, status, "start")
    append_event(out_root, {"event": "runner_start", "status": status})
    if args.dry_run:
        print(json.dumps(status, indent=2))
        return 0

    run_budget = args.max_shards if args.max_shards > 0 else len(shards)
    ran = 0
    for idx, shard in enumerate(shards):
        if ran >= run_budget:
            break
        if load_shard_summary(out_root / f"shard_{idx:02d}"):
            continue
        run_shard(args, out_root, idx, shard, status)
        status = summarize_existing(out_root, shards)
        write_status(out_root, status, "shard_complete")
        print(json.dumps(status, indent=2), flush=True)
        if status["verdict"] == "D10U_TOP01_16K_SHARDED_FAIL":
            return 2
        ran += 1

    status = summarize_existing(out_root, shards)
    if status["verdict"] == "RUNNING":
        status["verdict"] = "D10U_TOP01_16K_SHARDED_PARTIAL"
    write_status(out_root, status, "runner_stop")
    append_event(out_root, {"event": "runner_stop", "status": status})
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
