#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E57_OUTPUT_EGRESS_FIELD_MULTI_RESOLUTION_RENDERER_PROBE"
SYSTEMS = {
    "compact_only_output",
    "short_text_only_output",
    "long_text_only_output",
    "direct_pocket_to_text_unsafe",
    "naive_length_egress_router",
    "agency_committed_single_resolution",
    "agency_committed_multi_resolution_renderer",
    "oracle_egress_reference",
    "random_output_control",
}
STAGES = {
    "R0_compact_action_only",
    "R1_short_text_answer",
    "R2_long_trace_answer",
    "R3_multires_summary_plus_detail",
    "R4_unresolved_must_ask",
    "R5_stale_proposal_leak_attack",
    "R6_utf8_boundary_text",
    "R7_long_input_compact_answer",
}
DECISIONS = {
    "e57_multi_resolution_egress_renderer_confirmed",
    "e57_single_resolution_output_sufficient",
    "e57_output_stale_proposal_leak_detected",
    "e57_output_renderer_policy_unresolved",
    "e57_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "egress_mode_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "multi_resolution_report.json",
    "egress_policy_recommendation.json",
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
    "multi_resolution_report_sample.json",
    "row_level_sample.jsonl",
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
    runner = Path("scripts/probes/run_e57_output_egress_field_multi_resolution_renderer_probe.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage_metrics: dict[str, Any] = {}
    system_results: dict[str, Any] = {}
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
                "false_output_rate": mean([1.0 if row["false_output"] else 0.0 for row in stage_rows]),
                "wrong_confident_output_rate": mean([1.0 if row["wrong_confident_output"] else 0.0 for row in stage_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in stage_rows]),
                "stale_proposal_leak_rate": mean([1.0 if row["stale_proposal_leak"] else 0.0 for row in stage_rows]),
                "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in stage_rows]),
                "utf8_valid": mean([1.0 if row["utf8_valid"] else 0.0 for row in stage_rows]),
                "trace_backed_output": mean([1.0 if row["trace_backed_output"] else 0.0 for row in stage_rows]),
                "multi_resolution_write_success": mean([1.0 if row["multi_resolution_write_success"] else 0.0 for row in stage_rows]),
                "mean_cost": mean([float(row["chosen_cost"]) for row in stage_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        adversarial_rows = [row for row in system_rows if row["adversarial"]]
        multires_rows = [row for row in system_rows if row["requires_multi_resolution"]]
        system_results[system] = {
            "by_stage": by_stage,
            "overall": {
                "success": mean([1.0 if row["success"] else 0.0 for row in system_rows]),
                "mode_accuracy": mean([1.0 if row["mode_accuracy"] else 0.0 for row in system_rows]),
                "false_output_rate": mean([1.0 if row["false_output"] else 0.0 for row in system_rows]),
                "wrong_confident_output_rate": mean([1.0 if row["wrong_confident_output"] else 0.0 for row in system_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in system_rows]),
                "stale_proposal_leak_rate": mean([1.0 if row["stale_proposal_leak"] else 0.0 for row in system_rows]),
                "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in system_rows]),
                "utf8_valid": mean([1.0 if row["utf8_valid"] else 0.0 for row in system_rows]),
                "trace_backed_output": mean([1.0 if row["trace_backed_output"] else 0.0 for row in system_rows]),
                "multi_resolution_write_success": mean([1.0 if row["multi_resolution_write_success"] else 0.0 for row in multires_rows]),
                "adversarial_success": mean([1.0 if row["success"] else 0.0 for row in adversarial_rows]),
                "adversarial_false_output_rate": mean([1.0 if row["false_output"] else 0.0 for row in adversarial_rows]),
                "mean_cost": mean([float(row["chosen_cost"]) for row in system_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in system_rows]),
                "row_count": len(system_rows),
            },
        }
    for stage in STAGES:
        stage_metrics[stage] = {system: system_results[system]["by_stage"][stage] for system in SYSTEMS if system in system_results}
    multi_report = {
        system: {
            "multi_resolution_write_success": metrics["overall"]["multi_resolution_write_success"],
            "overall_success": metrics["overall"]["success"],
            "trace_backed_output": metrics["overall"]["trace_backed_output"],
            "stale_proposal_leak_rate": metrics["overall"]["stale_proposal_leak_rate"],
            "net_utility": metrics["overall"]["net_utility"],
        }
        for system, metrics in system_results.items()
    }
    return stage_metrics, system_results, multi_report


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
    recommendation = read_json(out / "egress_policy_recommendation.json")
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
    if not any(row.get("requires_multi_resolution") for row in rows):
        failures.append("no multi-resolution row-level cases")
    if not any(row.get("stale_proposal_present") for row in rows):
        failures.append("no stale proposal adversarial cases")

    stage_metrics, system_results, multi_report = summarize(rows)
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

    multi = reported_system_results["agency_committed_multi_resolution_renderer"]["overall"]
    if aggregate.get("decision") == "e57_multi_resolution_egress_renderer_confirmed":
        if multi["success"] < 0.98:
            failures.append("confirmed decision but multi renderer success below 0.98")
        if multi["mode_accuracy"] < 0.98:
            failures.append("confirmed decision but multi renderer mode accuracy below 0.98")
        if multi["multi_resolution_write_success"] < 0.98:
            failures.append("confirmed decision but multires write success below 0.98")
        if multi["false_output_rate"] > 0.01:
            failures.append("confirmed decision but false output above 0.01")
        if multi["stale_proposal_leak_rate"] > 0.01:
            failures.append("confirmed decision but stale leak above 0.01")
        if multi["trace_backed_output"] < 0.98:
            failures.append("confirmed decision but trace-backed output below 0.98")
    if recommendation.get("recommended_policy") != "agency_committed_multi_resolution_renderer":
        failures.append("recommended policy is not agency_committed_multi_resolution_renderer")

    unsafe = reported_system_results["direct_pocket_to_text_unsafe"]["overall"]
    if unsafe["stale_proposal_leak_rate"] <= 0.05:
        failures.append("unsafe direct pocket-to-text control did not expose stale proposal leak")

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
    parser.add_argument("--out", default="target/pilot_wave/e57_output_egress_field_multi_resolution_renderer_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e57_output_egress_field_multi_resolution_renderer_probe")
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
