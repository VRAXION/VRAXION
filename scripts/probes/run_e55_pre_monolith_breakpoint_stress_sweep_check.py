#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E55_PRE_MONOLITH_BREAKPOINT_STRESS_SWEEP"
SYSTEMS = {
    "current_pre_monolith_stack",
    "shortcut_or_raw_commit_control",
    "oracle_reference",
}
STAGES = [
    "S0_symbolic_controlled_evidence",
    "S1_noisy_text_controlled",
    "S2_adversarial_text_contrast",
    "S3_real_like_weak_text",
    "S4_missing_evidence_information_seeking",
    "S5_binary_packet_clean",
    "S6_binary_packet_noise10",
    "S7_binary_continuous_guarded",
    "S8_binary_bit_slip_resync",
    "S9_proposal_agency_adversarial",
    "S10_persistent_library_governance",
]
DECISIONS = {
    "e55_pre_monolith_breakpoints_localized",
    "e55_pre_monolith_text_frontier_still_open",
    "e55_pre_monolith_binary_resync_frontier_open",
    "e55_pre_monolith_all_sweep_clean",
    "e55_pre_monolith_core_regression_detected",
    "e55_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "stress_sweep_manifest.json",
    "stage_generation_report.json",
    "row_level_results.jsonl",
    "stage_metrics.json",
    "system_results.json",
    "breakpoint_sweep_report.json",
    "bottleneck_localization_report.json",
    "adversarial_stress_report.json",
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
    "stage_metrics_sample.json",
    "system_results_sample.json",
    "breakpoint_sweep_sample.json",
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e55_pre_monolith_breakpoint_stress_sweep.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = {
                "success": mean([1.0 if row["success"] else 0.0 for row in stage_rows]),
                "answer_correct": mean([1.0 if row["answer_correct"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "wrong_confident": mean([1.0 if row.get("wrong_confident") else 0.0 for row in stage_rows]),
                "avg_steps": mean([float(row.get("steps", 0)) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        out[system] = {
            "by_stage": by_stage,
            "overall": {
                "success": mean([1.0 if row["success"] else 0.0 for row in system_rows]),
                "answer_correct": mean([1.0 if row["answer_correct"] else 0.0 for row in system_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in system_rows]),
                "wrong_confident": mean([1.0 if row.get("wrong_confident") else 0.0 for row in system_rows]),
                "row_count": len(system_rows),
            },
        }
    return out


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
    stage_metrics = read_json(out / "stage_metrics.json")
    system_results = read_json(out / "system_results.json")
    breakpoint = read_json(out / "breakpoint_sweep_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("manifest system set mismatch")
    if manifest.get("stages") != STAGES:
        failures.append("manifest stage order mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not progress or not heartbeat:
        failures.append("empty required row/progress/heartbeat artifact")

    systems_seen = {row["system"] for row in rows}
    stages_seen = {row["stage"] for row in rows}
    if systems_seen != SYSTEMS:
        failures.append(f"row system mismatch {systems_seen}")
    if stages_seen != set(STAGES):
        failures.append(f"row stage mismatch {stages_seen}")

    recomputed = summarize(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]
        for stage in STAGES:
            for key, value in metrics["by_stage"][stage].items():
                compare_float(f"{system}.{stage}.{key}", value, reported["by_stage"][stage][key], failures)
                compare_float(f"stage_metrics.{stage}.{system}.{key}", value, stage_metrics[stage][system][key], failures)
        for key, value in metrics["overall"].items():
            compare_float(f"{system}.overall.{key}", value, reported["overall"][key], failures)

    primary = system_results["current_pre_monolith_stack"]["by_stage"]
    required_clean = [
        "S0_symbolic_controlled_evidence",
        "S1_noisy_text_controlled",
        "S4_missing_evidence_information_seeking",
        "S5_binary_packet_clean",
        "S6_binary_packet_noise10",
        "S7_binary_continuous_guarded",
        "S9_proposal_agency_adversarial",
        "S10_persistent_library_governance",
    ]
    for stage in required_clean:
        if primary[stage]["success"] < 0.95:
            failures.append(f"required clean stage regressed: {stage} success={primary[stage]['success']}")
    expected_frontier = ["S3_real_like_weak_text", "S8_binary_bit_slip_resync"]
    if not any(primary[stage]["success"] < 0.95 for stage in expected_frontier):
        failures.append("expected frontier stages did not expose any breakpoint")
    if breakpoint.get("first_failing_stage") not in set(STAGES) | {None}:
        failures.append("invalid first failing stage")
    if aggregate.get("first_failing_stage") != breakpoint.get("first_failing_stage"):
        failures.append("aggregate first failing stage mismatch")

    sample = validate_sample(sample_dir)
    if not sample["passed"]:
        failures.append("sample-only validation failed")
        failures.extend(f"sample: {failure}" for failure in sample["failures"])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_only_checker_passed": sample["passed"],
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e55_pre_monolith_breakpoint_stress_sweep")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e55_pre_monolith_breakpoint_stress_sweep")
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
