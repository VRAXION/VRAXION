#!/usr/bin/env python3
"""Checker for E123 orange-baseline FineWeb new-skill discovery probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E123_ORANGE_BASELINE_FINEWEB_NEW_SKILL_DISCOVERY_PROBE"
REQUIRED = [
    "run_manifest.json",
    "dataset_report.json",
    "orange_library_report.json",
    "candidate_discovery_report.json",
    "negative_card_interaction_report.json",
    "text_io_probe_report.json",
    "row_level_samples.jsonl",
    "candidate_examples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def read_json(path: Path) -> dict[str, Any]:
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
    dataset = read_json(root / "dataset_report.json")
    library = read_json(root / "orange_library_report.json")
    candidates = read_json(root / "candidate_discovery_report.json")["rows"]
    interactions = read_json(root / "negative_card_interaction_report.json")["rows"]
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [line for line in (root / "candidate_examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report_text = (root / "report.md").read_text(encoding="utf-8")

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for text in ["discovery only", "no promotion", "not Core", "not PermaCore", "not TrueGolden", "not final training", "not Gemma-style generation"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    for phrase in ["promotion confirmed", "PermaCore", "TrueGolden", "Gemma-level"]:
        if phrase in report_text and phrase not in boundary:
            failures.append(f"forbidden claim in report: {phrase}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if dataset.get("rows_seen") != aggregate.get("rows_seen"):
        failures.append("dataset/aggregate row mismatch")
    if aggregate.get("rows_seen", 0) < 10_000:
        failures.append("too few rows scanned")
    if library.get("active_operator_count") != 144 or aggregate.get("active_operator_count") != 144:
        failures.append("E122 active orange count mismatch")
    if not aggregate.get("orange_only_confirmed"):
        failures.append("orange-only baseline not confirmed")
    if library.get("negative_card_count", 0) <= 0 or aggregate.get("negative_card_count", 0) <= 0:
        failures.append("negative cards missing")
    if aggregate.get("candidate_count") != len(candidates):
        failures.append("candidate count mismatch")
    if aggregate.get("new_farm_candidate_count") != len([row for row in candidates if row.get("suggested_status") == "NewFarmCandidate"]):
        failures.append("new farm candidate count mismatch")
    if aggregate.get("covered_candidate_count") != len([row for row in candidates if row.get("suggested_status") == "CoveredByOrangeBaseline"]):
        failures.append("covered candidate count mismatch")
    if aggregate.get("negative_card_false_block_count") != 0:
        failures.append("negative card false block detected")
    if aggregate.get("normal_router_callable_cards") != 0:
        failures.append("negative cards leaked to normal router")
    if aggregate.get("negative_card_blocked_bad_variant_count", 0) <= 0:
        failures.append("negative cards did not block any bad variants")
    if not candidates:
        failures.append("candidate rows missing")
    for row in candidates:
        if row.get("support_count", 0) <= 0:
            failures.append(f"candidate support missing: {row.get('candidate_id')}")
        if row.get("suggested_status") == "NewFarmCandidate" and row.get("avg_orange_coverage", 1.0) >= 0.75:
            failures.append(f"new candidate coverage too high: {row.get('candidate_id')}")
    if len(interactions) != len(candidates):
        failures.append("negative-card interaction count mismatch")
    for row in interactions:
        if row.get("normal_router_callable_cards") != 0 or row.get("false_block_count") != 0:
            failures.append(f"bad negative-card interaction: {row.get('candidate_id')}")
    if not samples:
        failures.append("row-level samples missing")
    if not examples:
        failures.append("candidate examples missing")
    if len(progress) < 3 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "candidate_rows": candidates,
        "card_interactions": interactions,
        "dataset": manifest.get("dataset"),
        "contract": CONTRACT,
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
    parser.add_argument("--out", default="target/pilot_wave/e123_orange_baseline_fineweb_new_skill_discovery_probe")
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
