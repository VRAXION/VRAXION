#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


MILESTONE = "E63_POCKET_OBSERVATORY_VISUAL_DEBUG_DASHBOARD"
SCHEMA_VERSION = "pocket_observatory_v1"
DECISIONS = {
    "e63_pocket_observatory_dashboard_ready",
    "e63_pocket_observatory_sample_only",
    "e63_pocket_observatory_invalid_artifact",
}
REQUIRED = [
    "backend_manifest.json",
    "observatory_snapshot.json",
    "pocket_state.json",
    "pocket_events.jsonl",
    "flow_snapshot.json",
    "proposal_snapshot.json",
    "agency_decisions.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "report.md",
    "index.html",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def static_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard.py")
    checker = Path("scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard_check.py")
    for path in [runner, checker]:
        ast.parse(path.read_text(encoding="utf-8"))
    source = runner.read_text(encoding="utf-8")
    forbidden = ["tensorflow", "torch.", "optimizer", "backward(", "loss.backward", "eval("]
    for token in forbidden:
        if token in source:
            failures.append(f"runner contains forbidden training/runtime token: {token}")
    return failures


def validate(root: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing required artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(root / "checker_summary.json", result)
        return result

    failures.extend(static_check())
    manifest = read_json(root / "backend_manifest.json")
    snapshot = read_json(root / "observatory_snapshot.json")
    pocket_state = read_json(root / "pocket_state.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    replay = read_json(root / "deterministic_replay.json")
    events = read_jsonl(root / "pocket_events.jsonl")
    decisions = read_jsonl(root / "agency_decisions.jsonl")
    html = (root / "index.html").read_text(encoding="utf-8")

    if manifest.get("milestone") != MILESTONE or summary.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        failures.append("snapshot schema mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("invalid decision label")
    if decision.get("decision") != summary.get("decision"):
        failures.append("decision mismatch between decision.json and summary.json")
    if aggregate.get("pocket_count") != len(snapshot.get("pockets", [])):
        failures.append("aggregate pocket_count does not match snapshot pockets")
    if aggregate.get("false_commits") != 0:
        failures.append("sample artifact should have zero false commits")
    if len(snapshot.get("cycles", [])) < 8:
        failures.append("not enough cycles for heatmap playback")
    if len(snapshot.get("pockets", [])) < 6:
        failures.append("not enough pockets for observatory sample")
    if not snapshot.get("heatmap"):
        failures.append("missing heatmap rows")
    if not events:
        failures.append("missing pocket events")
    if not decisions:
        failures.append("missing Agency decisions")
    if len(pocket_state.get("pockets", [])) != len(snapshot.get("pockets", [])):
        failures.append("pocket_state and snapshot pocket counts differ")
    if "https://" in html or "http://" in html:
        failures.append("dashboard must not depend on external network/CDN URLs")
    for token in ["Activity Heatmap", "Flow Field Commit Grid", "Auto-refresh", "embedded-data"]:
        if token not in html:
            failures.append(f"dashboard missing expected UI token {token}")
    for name, expected in replay.get("artifact_hashes", {}).items():
        path = root / name
        if not path.exists():
            failures.append(f"replay references missing artifact {name}")
        elif sha256_file(path) != expected:
            failures.append(f"deterministic replay hash mismatch for {name}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": decision.get("decision"),
        "pocket_count": aggregate.get("pocket_count"),
        "cycle_count": aggregate.get("cycle_count"),
    }
    if write_summary:
        write_json(root / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard")
    parser.add_argument("--sample-only", default="")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    result = validate(root, args.write_summary)
    if args.write_summary and args.sample_only:
        write_json(root / "sample_only_checker_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
