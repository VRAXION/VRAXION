#!/usr/bin/env python3
"""Checker for E117 alpha-Weave targeted pressure gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E117_ALPHA_WEAVE_TARGETED_PRESSURE_GAUNTLET"
REQUIRED = [
    "run_manifest.json",
    "gauntlet_manifest.json",
    "operator_gauntlet_results.json",
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


def deterministic_hash(payload: Any) -> str:
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
    gauntlet = read_json(root / "gauntlet_manifest.json")
    rows = read_json(root / "operator_gauntlet_results.json")["rows"]
    samples = read_json(root / "row_level_samples.json")["rows"]
    hard_negative_samples = read_json(root / "hard_negative_samples.json")["rows"]
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    report_text = (root / "report.md").read_text(encoding="utf-8")
    progress_lines = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    boundary = manifest.get("boundary", "")
    for phrase in ["targeted pressure gauntlet only", "not final training", "not PermaCore", "not TrueGolden", "not automatic Core promotion"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    if "PermaCore promotion confirmed" in report_text or "TrueGolden promotion confirmed" in report_text:
        failures.append("forbidden promotion claim in report")
    if not gauntlet.get("source_e116_summary"):
        failures.append("missing E116 source summary")
    if aggregate.get("hard_negative_total") != 0:
        failures.append("hard negatives detected")
    for key in [
        "false_commit_total",
        "wrong_scope_call_total",
        "unsupported_answer_total",
        "over_budget_total",
        "public_leak_total",
        "schema_failure_total",
        "metadata_failure_total",
    ]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if aggregate.get("target_reach_count") != aggregate.get("target_operator_count"):
        failures.append("not all target operators reached next activation limit")
    if aggregate.get("targeted_needed_remaining_count") != 0:
        failures.append("targeted needed remaining nonzero")
    if len(rows) != aggregate.get("target_operator_count"):
        failures.append("operator row count mismatch")
    if aggregate.get("qualified_activation_total") != aggregate.get("scheduled_case_count"):
        failures.append("qualified activation does not match scheduled cases")
    if hard_negative_samples:
        failures.append("hard negative sample list is not empty")
    for row in rows:
        if row.get("hard_negative") != 0:
            failures.append(f"operator hard negative: {row.get('operator_id')}")
        if not row.get("reaches_permacore_probation_after_e117_gauntlet"):
            failures.append(f"operator target not reached: {row.get('operator_id')}")
        if row.get("remaining_after_e117_gauntlet") != 0:
            failures.append(f"operator still remaining: {row.get('operator_id')}")
        if row.get("qualified_activation", 0) <= 0:
            failures.append(f"operator has no qualified activation: {row.get('operator_id')}")

    source_summary = gauntlet.get("source_e116_summary", {})
    if source_summary.get("scheduled_case_count") != aggregate.get("scheduled_case_count"):
        failures.append("scheduled case count does not match E116 source")
    if source_summary.get("generated_cell_packs") != aggregate.get("generated_cell_packs"):
        failures.append("generated cell pack count does not match E116 source")
    if len(progress_lines) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress_lines):
        failures.append("progress incomplete")

    replay_payload = {
        "contract": CONTRACT,
        "schema": read_json(Path("docs/research/ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json")) if Path("docs/research/ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json").exists() else None,
        "source_summary": source_summary,
        "operator_rows": rows,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "sample_hash": deterministic_hash(samples[:8]),
    }
    if replay.get("hash") != deterministic_hash(replay_payload) or not replay.get("hash_match"):
        failures.append("deterministic replay mismatch")
    if decision.get("failure_count") != 0:
        failures.append("decision failure_count nonzero")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e117_alpha_weave_targeted_pressure_gauntlet")
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
