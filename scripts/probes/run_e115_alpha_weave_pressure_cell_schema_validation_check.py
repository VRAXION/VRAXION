#!/usr/bin/env python3
"""Checker for E115 alpha-Weave pressure-cell schema validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E115_ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_VALIDATION"
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SCHEMA = REPO_ROOT / "docs" / "research" / "ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json"
REQUIRED = [
    "run_manifest.json",
    "alpha_weave_pressure_cell_schema_v1.json",
    "sample_cell_pack.json",
    "public_input_samples.json",
    "machine_solve_view.json",
    "schema_validation_report.json",
    "adversarial_validation_report.json",
    "control_results.json",
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


def walk_public(obj: Any, forbidden_keys: set[str], forbidden_text: set[str], path: str = "public_input") -> list[str]:
    failures: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in forbidden_keys:
                failures.append(f"{path}: forbidden public key {key}")
            failures.extend(walk_public(value, forbidden_keys, forbidden_text, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            failures.extend(walk_public(value, forbidden_keys, forbidden_text, f"{path}[{index}]"))
    elif isinstance(obj, str):
        lower = obj.lower()
        for token in forbidden_text:
            if token.lower() in lower:
                failures.append(f"{path}: forbidden public token {token}")
    return failures


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    manifest = read_json(root / "run_manifest.json")
    schema = read_json(root / "alpha_weave_pressure_cell_schema_v1.json")
    cell = read_json(root / "sample_cell_pack.json")
    schema_report = read_json(root / "schema_validation_report.json")
    adversarial = read_json(root / "adversarial_validation_report.json")["rows"]
    controls = read_json(root / "control_results.json")["rows"]
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    boundary = manifest.get("boundary", "")
    for phrase in ["schema validation only", "not final training", "not PermaCore", "not TrueGolden"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    if schema.get("schema_version") != "AlphaWeavePressureCell-v1":
        failures.append("schema version mismatch")
    if CANONICAL_SCHEMA.exists() and schema != read_json(CANONICAL_SCHEMA):
        failures.append("schema artifact differs from tracked canonical schema")
    if not schema_report.get("schema_valid") or schema_report.get("failures"):
        failures.append("schema validation report has failures")
    forbidden_keys = set(schema.get("public_forbidden_keys", []))
    forbidden_text = set(schema.get("public_forbidden_text", []))
    failures.extend(walk_public(cell.get("public_input", {}), forbidden_keys, forbidden_text))
    for variant in cell.get("adversarial_variants", []):
        failures.extend(walk_public(variant.get("public_input", {}), forbidden_keys, forbidden_text, variant.get("variant_id", "variant")))
    if aggregate.get("oracle_leak_rate") != 0.0:
        failures.append("oracle leak rate nonzero")
    if aggregate.get("target_operator_leak_rate") != 0.0:
        failures.append("target operator leak rate nonzero")
    if aggregate.get("primary_success_rate") != 1.0:
        failures.append("primary policy did not solve all variants")
    if aggregate.get("false_commit_rate") != 0.0:
        failures.append("false commit rate nonzero")
    if aggregate.get("wrong_scope_call_rate") != 0.0:
        failures.append("wrong scope call rate nonzero")
    if aggregate.get("unsupported_answer_rate") != 0.0:
        failures.append("unsupported answer rate nonzero")
    if aggregate.get("over_budget_rate") != 0.0:
        failures.append("over budget rate nonzero")
    if not aggregate.get("controls_all_invalid_as_general_policy"):
        failures.append("not all adversarial controls were invalidated")
    if not adversarial or any(not row.get("success") for row in adversarial):
        failures.append("adversarial variant failure")
    if not controls or any(not row.get("invalid_as_general_policy") for row in controls):
        failures.append("control invalidation mismatch")
    replay_payload = {
        "contract": CONTRACT,
        "schema": schema,
        "cell": cell,
        "variant_results": adversarial,
        "control_results": controls,
        "aggregate": aggregate,
    }
    if replay.get("hash") != deterministic_hash(replay_payload) or not replay.get("hash_match"):
        failures.append("deterministic replay mismatch")
    if decision.get("failure_count") != 0:
        failures.append("decision failure_count nonzero")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e115_alpha_weave_pressure_cell_schema_validation")
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
