#!/usr/bin/env python3
"""Checker for E118 CoreCandidate cross-source no-harm gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E118_CORE_CANDIDATE_CROSS_SOURCE_NO_HARM_GAUNTLET"
REQUIRED = [
    "run_manifest.json",
    "source_manifest.json",
    "operator_cross_source_results.json",
    "source_family_report.json",
    "row_level_samples.json",
    "hard_negative_samples.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "partial_aggregate_snapshot.json",
    "progress.jsonl",
    "report.md",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    manifest = read_json(root / "run_manifest.json")
    source = read_json(root / "source_manifest.json")
    rows = read_json(root / "operator_cross_source_results.json")["rows"]
    source_rows = read_json(root / "source_family_report.json")["rows"]
    samples = read_json(root / "row_level_samples.json")["rows"]
    hard_samples = read_json(root / "hard_negative_samples.json")["rows"]
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report = (root / "report.md").read_text(encoding="utf-8")

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    boundary = manifest.get("boundary", "")
    for phrase in ["cross-source no-harm", "not PermaCore", "not TrueGolden", "not automatic Core promotion"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    if "PermaCore promotion confirmed" in report or "TrueGolden promotion confirmed" in report:
        failures.append("forbidden promotion claim")
    if len(source.get("source_families", [])) < 8:
        failures.append("too few source families")
    if source.get("fineweb_sample_count", 0) <= 0:
        failures.append("missing FineWeb-derived samples")
    if aggregate.get("candidate_count") != 136:
        failures.append("candidate count must be 136")
    if aggregate.get("actual_300k_count", 0) < 77:
        failures.append("actual 300k count below E117 target reach count")
    if len(rows) != aggregate.get("candidate_count"):
        failures.append("operator row count mismatch")
    if aggregate.get("source_family_count") != len(source_rows):
        failures.append("source family row count mismatch")
    if aggregate.get("case_count", 0) < 15000:
        failures.append("case count too small")
    for key in [
        "hard_negative_total",
        "false_commit_total",
        "unsupported_answer_total",
        "wrong_scope_call_total",
        "negative_transfer_total",
        "synthetic_imprint_total",
    ]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if aggregate.get("cross_source_no_harm_pass_count") != aggregate.get("candidate_count"):
        failures.append("not all candidates passed cross-source no-harm")
    if aggregate.get("cross_source_no_harm_remaining_count") != 0:
        failures.append("remaining cross-source candidates nonzero")
    if hard_samples:
        failures.append("hard negative samples not empty")
    for row in rows:
        if not row.get("cross_source_no_harm_pass"):
            failures.append(f"operator did not pass: {row.get('operator_id')}")
        if row.get("source_family_coverage") != aggregate.get("source_family_count"):
            failures.append(f"source coverage mismatch: {row.get('operator_id')}")
        if row.get("hard_negative_count") != 0 or row.get("negative_transfer_count") != 0:
            failures.append(f"operator hard/negative transfer: {row.get('operator_id')}")
        if "actual_300k_reached" not in row:
            failures.append(f"operator missing actual 300k marker: {row.get('operator_id')}")
    if len(samples) < 64:
        failures.append("row-level samples too sparse")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    replay_payload = {
        "contract": CONTRACT,
        "candidate_ids": [row["operator_id"] for row in rows],
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "source_rows": source_rows,
        "sample_hash": stable_hash(samples[:16]),
    }
    if replay.get("hash") != stable_hash(replay_payload) or not replay.get("hash_match"):
        failures.append("deterministic replay mismatch")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e118_core_candidate_cross_source_no_harm_confirmed":
        failures.append("decision not confirmed")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e118_core_candidate_cross_source_no_harm_gauntlet")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.out)
    failures = check(root)
    payload = {
        "artifact_contract": CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "target_checker_passed": not failures,
    }
    if args.write_summary:
        (root / "checker_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
