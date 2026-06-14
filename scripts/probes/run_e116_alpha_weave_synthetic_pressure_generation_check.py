#!/usr/bin/env python3
"""Checker for E116 alpha-Weave synthetic pressure generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E116_ALPHA_WEAVE_SYNTHETIC_PRESSURE_GENERATION"
REQUIRED = [
    "run_manifest.json",
    "generation_manifest.json",
    "synthetic_origin_report.json",
    "rare_operator_input_report.json",
    "generated_cells.jsonl",
    "operator_target_coverage.json",
    "activation_projection_report.json",
    "leakage_check_report.json",
    "public_sample_cells.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "partial_aggregate_snapshot.json",
    "progress.jsonl",
    "human_machine_sample_report.md",
    "report.md",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def public_contains_forbidden(obj: Any) -> list[str]:
    failures: list[str] = []
    forbidden = {
        "synthetic_codex_generated",
        "codex",
        "target_skill",
        "target_operators",
        "hidden_oracle",
        "expected_answer",
        "expected_action",
    }
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in forbidden:
                failures.append(f"forbidden key {key}")
            failures.extend(public_contains_forbidden(value))
    elif isinstance(obj, list):
        for value in obj:
            failures.extend(public_contains_forbidden(value))
    elif isinstance(obj, str):
        lower = obj.lower()
        for token in forbidden:
            if token in lower:
                failures.append(f"forbidden token {token}")
    return failures


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    manifest = read_json(root / "run_manifest.json")
    generation = read_json(root / "generation_manifest.json")
    origin = read_json(root / "synthetic_origin_report.json")
    coverage = read_json(root / "operator_target_coverage.json")["rows"]
    projection = read_json(root / "activation_projection_report.json")["rows"]
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
    for phrase in ["synthetic pressure data generation only", "not final training", "not PermaCore", "not TrueGolden"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    if generation.get("data_origin") != "synthetic_codex_generated":
        failures.append("generation data_origin missing")
    if not generation.get("human_review_status"):
        failures.append("human_review_status missing")
    if origin.get("public_leak_failure_count") != 0:
        failures.append("public leak failures detected")
    if aggregate.get("schema_failure_count") != 0:
        failures.append("schema failures detected")
    if aggregate.get("public_leak_failure_count") != 0:
        failures.append("aggregate public leak failures detected")
    if aggregate.get("targeted_needed_remaining_count") != 0:
        failures.append("not all rare operators reached targeted next limit")
    if aggregate.get("target_reach_count") != aggregate.get("rare_operator_count"):
        failures.append("target reach count mismatch")
    if len(coverage) != aggregate.get("rare_operator_count"):
        failures.append("coverage operator count mismatch")
    if len(projection) != len(coverage):
        failures.append("projection/coverage count mismatch")
    for row in coverage:
        if row.get("qualified_synthetic_pressure_activation", 0) <= 0:
            failures.append(f"no synthetic activation: {row.get('operator_id')}")
        if not row.get("reaches_permacore_probation_after_targeted_pressure"):
            failures.append(f"target not reached: {row.get('operator_id')}")

    generated_path = root / "generated_cells.jsonl"
    generated_count = 0
    sample_cells = []
    with generated_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            generated_count += 1
            if generated_count <= 32:
                cell = json.loads(line)
                metadata = cell.get("training_metadata", {})
                if metadata.get("data_origin") != "synthetic_codex_generated":
                    failures.append(f"missing synthetic origin: {cell.get('cell_id')}")
                if not metadata.get("synthetic_disclosure"):
                    failures.append(f"missing synthetic disclosure: {cell.get('cell_id')}")
                failures.extend(f"{cell.get('cell_id')}: {failure}" for failure in public_contains_forbidden(cell.get("public_input", {})))
                for variant in cell.get("adversarial_variants", []):
                    failures.extend(f"{cell.get('cell_id')}:{variant.get('variant_id')}: {failure}" for failure in public_contains_forbidden(variant.get("public_input", {})))
                sample_cells.append({
                    "cell_id": cell.get("cell_id"),
                    "training_metadata": metadata,
                })
    if generated_count != aggregate.get("generated_cell_packs"):
        failures.append("generated cell count mismatch")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")

    replay_payload = {
        "contract": CONTRACT,
        "schema": read_json(root / "run_manifest.json").get("schema_version") and read_json(Path("docs/research/ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json")) if Path("docs/research/ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json").exists() else None,
        "operators": coverage,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "generated_hash": deterministic_hash({"cell_count": generated_count, "first_samples": read_json(root / "public_sample_cells.json")["rows"][:3]}),
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
    parser.add_argument("--out", default="target/pilot_wave/e116_alpha_weave_synthetic_pressure_generation")
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
