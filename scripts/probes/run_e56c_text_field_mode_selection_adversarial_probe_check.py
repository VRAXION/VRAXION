#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E56C_TEXT_FIELD_MODE_SELECTION_ADVERSARIAL_PROBE"
SYSTEMS = {
    "always_fast_default",
    "always_long_capped",
    "always_clean_long",
    "naive_length_router",
    "clean_long_without_cost_guard",
    "three_mode_agency_router",
    "oracle_mode_selector",
    "random_mode_control",
}
STAGES = {
    "A0_short_answerable",
    "A1_boundary_overlap_answerable",
    "A2_medium_needs_long_capped",
    "A3_long_clean_required",
    "A4_long_lure_relevant_early",
    "A5_missing_evidence_must_ask",
    "A6_oversize_requires_multi_cycle",
    "A7_adversarial_decoy_requires_clean",
}
DECISIONS = {
    "e56c_three_mode_agency_selector_adversarial_confirmed",
    "e56c_single_clean_long_mode_cost_overfit_detected",
    "e56c_length_router_insufficient_under_adversarial_mix",
    "e56c_clean_long_required_as_default",
    "e56c_mode_policy_unresolved",
    "e56c_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "mode_selection_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "adversarial_report.json",
    "mode_policy_recommendation.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "stage_metrics_sample.json",
    "row_level_sample.jsonl",
    "adversarial_report_sample.json",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e56c_text_field_mode_selection_adversarial_probe.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    system_results: dict[str, Any] = {}
    stage_metrics: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            continue
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = {
                "success": mean([1.0 if row["success"] else 0.0 for row in stage_rows]),
                "mode_accuracy": mean([1.0 if row["mode_accuracy"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in stage_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in stage_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in stage_rows]),
                "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in stage_rows]),
                "mean_cost": mean([float(row["chosen_cost"]) for row in stage_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        adversarial_rows = [row for row in system_rows if row["adversarial"]]
        system_results[system] = {
            "by_stage": by_stage,
            "overall": {
                "success": mean([1.0 if row["success"] else 0.0 for row in system_rows]),
                "mode_accuracy": mean([1.0 if row["mode_accuracy"] else 0.0 for row in system_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in system_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in system_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in system_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in system_rows]),
                "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in system_rows]),
                "mean_cost": mean([float(row["chosen_cost"]) for row in system_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in system_rows]),
                "adversarial_success": mean([1.0 if row["success"] else 0.0 for row in adversarial_rows]),
                "adversarial_mode_accuracy": mean([1.0 if row["mode_accuracy"] else 0.0 for row in adversarial_rows]),
                "adversarial_false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in adversarial_rows]),
                "row_count": len(system_rows),
            },
        }
    for stage in STAGES:
        stage_metrics[stage] = {system: system_results[system]["by_stage"][stage] for system in SYSTEMS if system in system_results}
    return stage_metrics, system_results


def compare_float(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    schema = read_json(sample_dir / "sample_schema.json")
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if set(schema.get("systems", [])) != SYSTEMS:
        failures.append("sample systems mismatch")
    if set(schema.get("stages", [])) != STAGES:
        failures.append("sample stages mismatch")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample replay did not pass")
    if not rows:
        failures.append("sample row-level data empty")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(out / "checker_summary.json", result)
        return result

    failures.extend(static_policy_check())
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    recommendation = read_json(out / "mode_policy_recommendation.json")
    reported_system_results = read_json(out / "system_results.json")
    reported_stage_metrics = read_json(out / "stage_metrics.json")
    rows = read_jsonl(out / "row_level_results.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("manifest systems mismatch")
    if set(manifest.get("stages", [])) != STAGES:
        failures.append("manifest stages mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flag is not false")
    if aggregate.get("decision") != decision.get("decision") or decision.get("decision") not in DECISIONS:
        failures.append("decision artifact mismatch or invalid decision")
    if summary.get("decision") != aggregate.get("decision"):
        failures.append("summary decision mismatch")
    if not rows:
        failures.append("row-level results empty")

    row_systems = {row.get("system") for row in rows}
    row_stages = {row.get("stage") for row in rows}
    if row_systems != SYSTEMS:
        failures.append(f"row system coverage mismatch: {sorted(row_systems)}")
    if row_stages != STAGES:
        failures.append(f"row stage coverage mismatch: {sorted(row_stages)}")
    if not any(row.get("adversarial") for row in rows):
        failures.append("no adversarial row-level cases")

    stage_metrics, system_results = summarize(rows)
    for system in SYSTEMS:
        for key, value in system_results[system]["overall"].items():
            if isinstance(value, (int, float)):
                compare_float(f"{system}.overall.{key}", value, reported_system_results[system]["overall"][key], failures)
    for stage in STAGES:
        for system in SYSTEMS:
            compare_float(
                f"{stage}.{system}.success",
                stage_metrics[stage][system]["success"],
                reported_stage_metrics[stage][system]["success"],
                failures,
            )

    agency = reported_system_results["three_mode_agency_router"]["overall"]
    if aggregate.get("decision") == "e56c_three_mode_agency_selector_adversarial_confirmed":
        if agency["success"] < 0.98:
            failures.append("confirmed decision but agency success below 0.98")
        if agency["mode_accuracy"] < 0.98:
            failures.append("confirmed decision but agency mode accuracy below 0.98")
        if agency["adversarial_success"] < 0.98:
            failures.append("confirmed decision but adversarial success below 0.98")
        if agency["false_commit_rate"] > 0.01:
            failures.append("confirmed decision but false commit above 0.01")
        if agency["overpay_rate"] > 0.01:
            failures.append("confirmed decision but overpay above 0.01")
    if recommendation.get("recommended_policy") != "three_mode_agency_router":
        failures.append("recommended policy is not three_mode_agency_router")

    for name, expected_hash in replay.get("artifact_hashes", {}).items():
        path = out / name
        if not path.exists():
            failures.append(f"replay hash references missing artifact {name}")
        elif file_sha256(path) != expected_hash:
            failures.append(f"deterministic replay hash mismatch for {name}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_only_checker_passed": sample_result["passed"],
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e56c_text_field_mode_selection_adversarial_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e56c_text_field_mode_selection_adversarial_probe")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            write_json(Path(args.sample_only) / "sample_only_checker_summary.json", result)
    else:
        result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
