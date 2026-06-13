#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E64_FIELD_SIZE_AND_AGENCY_VIEW_SWEEP"
SYSTEMS = {
    "tiny_12x12_all",
    "compact_16x16_all",
    "balanced_24x24_all",
    "proposal_width_64_control",
    "asymmetric_24f_32g_20x80_control",
    "near_28f_32g_20x80_default",
    "wide_32x32_20x80",
    "large_48x48_24x80",
    "oversized_64x64_32x80",
    "proposal_starved_32x32",
    "agency_starved_32x32",
}
STAGES = {
    "F0_local_short_commit",
    "F1_active_evidence_trace",
    "F2_proposal_collision_commit",
    "F3_ground_contradiction_check",
    "F4_text_digest_to_flow",
    "F5_adversarial_proposal_flood",
    "F6_multi_cycle_repair_memory",
    "F7_overcapacity_decoy_pressure",
}
DECISIONS = {
    "e64_near_28f_32g_20x80_default_confirmed",
    "e64_wide_32x32_default_required",
    "e64_compact_16x16_sufficient",
    "e64_no_clean_size_within_cost_gate",
    "e64_proposal_or_agency_view_bottleneck",
    "e64_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "field_size_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "capacity_frontier_report.json",
    "agency_view_report.json",
    "proposal_capacity_report.json",
    "recommendation.json",
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
    "capacity_frontier_report_sample.json",
    "recommendation_sample.json",
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


def compare(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e64_field_size_and_agency_view_sweep.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "loss.backward", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned training/direct-solver token: {token}")
    return failures


def row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["success"]]
    return {
        "row_count": len(rows),
        "success": mean([1.0 if row["success"] else 0.0 for row in rows]),
        "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in rows]),
        "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in rows]),
        "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in rows]),
        "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in rows]),
        "overpay_rate": mean([1.0 if row["overpay"] else 0.0 for row in rows]),
        "net_utility": mean([float(row["net_utility"]) for row in rows]),
        "mean_cost_multiplier": mean([float(row["cost_multiplier"]) for row in rows]),
        "mean_latency_units": mean([float(row["latency_units"]) for row in rows]),
        "mean_attempts_to_95_success_only": mean([float(row["attempts_to_95"]) for row in success_rows]),
        "flow_capacity_pass": mean([1.0 if row["flow_ok"] else 0.0 for row in rows]),
        "ground_capacity_pass": mean([1.0 if row["ground_ok"] else 0.0 for row in rows]),
        "proposal_slot_pass": mean([1.0 if row["proposal_slots_ok"] else 0.0 for row in rows]),
        "proposal_width_pass": mean([1.0 if row["proposal_bits_ok"] else 0.0 for row in rows]),
        "agency_view_pass": mean([1.0 if row["agency_ok"] else 0.0 for row in rows]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if system_rows:
            out[system] = row_metrics(system_rows)
    return out


def validate_sample(sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    required = [name for name in REQ_SAMPLE if name != "sample_only_checker_result.json"]
    for name in required:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(sample_dir / "sample_only_checker_result.json", result)
        return result
    schema = read_json(sample_dir / "sample_schema.json")
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    recommendation = read_json(sample_dir / "recommendation_sample.json")
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
        failures.append("sample deterministic replay did not pass")
    if recommendation.get("recommended_default") != "near_28f_32g_20x80_default":
        failures.append("sample recommendation missing expected default")
    if len(rows) < len(SYSTEMS) * len(STAGES):
        failures.append("sample row coverage too small")
    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "recommended_default": recommendation.get("recommended_default"),
    }
    if write_summary:
        write_json(sample_dir / "sample_only_checker_result.json", result)
    return result


def validate_target(out: Path, write_summary: bool) -> dict[str, Any]:
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
    field_manifest = read_json(out / "field_size_manifest.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    reported_systems = read_json(out / "system_results.json")
    reported_stages = read_json(out / "stage_metrics.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    recommendation = read_json(out / "recommendation.json")
    replay = read_json(out / "deterministic_replay.json")
    agency_report = read_json(out / "agency_view_report.json")
    proposal_report = read_json(out / "proposal_capacity_report.json")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("manifest systems mismatch")
    if set(manifest.get("stages", [])) != STAGES:
        failures.append("manifest stages mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if len(field_manifest.get("configs", [])) != len(SYSTEMS):
        failures.append("field config count mismatch")
    if aggregate.get("decision") != decision.get("decision") or decision.get("decision") not in DECISIONS:
        failures.append("decision mismatch or invalid decision")
    if summary.get("decision") != aggregate.get("decision"):
        failures.append("summary decision mismatch")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows:
        failures.append("row-level results are empty")
    observed_systems = {row.get("system") for row in rows}
    observed_stages = {row.get("stage") for row in rows}
    if observed_systems != SYSTEMS:
        failures.append(f"row systems mismatch: {sorted(observed_systems)}")
    if observed_stages != STAGES:
        failures.append(f"row stages mismatch: {sorted(observed_stages)}")

    recomputed = summarize(rows)
    for system, metrics in recomputed.items():
        reported = reported_systems.get(system, {}).get("overall", {})
        for key in ["success", "false_commit_rate", "missed_commit_rate", "net_utility", "agency_view_pass", "proposal_slot_pass"]:
            compare(f"{system}.{key}", metrics[key], reported.get(key), failures)
    for stage in STAGES:
        for system in SYSTEMS:
            stage_rows = [row for row in rows if row["stage"] == stage and row["system"] == system]
            metrics = row_metrics(stage_rows)
            reported = reported_stages.get(stage, {}).get(system, {})
            compare(f"{stage}.{system}.success", metrics["success"], reported.get("success"), failures)

    recommended = recommendation.get("recommended_default")
    if recommended != "near_28f_32g_20x80_default":
        failures.append(f"unexpected recommended default {recommended}")
    if recommendation.get("flow_field", {}).get("shape") != [28, 28]:
        failures.append("recommended Flow shape mismatch")
    if recommendation.get("ground_field", {}).get("shape") != [32, 32]:
        failures.append("recommended Ground shape mismatch")
    if recommendation.get("proposal_field", {}).get("slots") != 20 or recommendation.get("proposal_field", {}).get("bits_per_slot") != 80:
        failures.append("recommended Proposal field mismatch")
    if recommendation.get("agency_view", {}).get("bits") != 896:
        failures.append("recommended Agency view mismatch")

    recommended_metrics = reported_systems.get("near_28f_32g_20x80_default", {}).get("overall", {})
    if recommended_metrics.get("success", 0.0) < 0.985:
        failures.append("recommended default did not clear success threshold")
    if recommended_metrics.get("false_commit_rate", 1.0) != 0.0:
        failures.append("recommended default had false commits")
    if reported_systems.get("wide_32x32_20x80", {}).get("overall", {}).get("net_utility", 0.0) > recommended_metrics.get("net_utility", 0.0) + 0.02:
        failures.append("wide 32x32 materially beats recommended default")
    if agency_report.get("bottleneck_controls", {}).get("agency_starved_32x32", {}).get("false_commit_rate", 0.0) <= 0.0:
        failures.append("agency bottleneck control did not expose false commits")
    if proposal_report.get("bottleneck_controls", {}).get("proposal_starved_32x32", {}).get("false_commit_rate", 0.0) <= 0.0:
        failures.append("proposal bottleneck control did not expose false commits")

    hashed = replay.get("artifact_hashes", {})
    for name, expected in hashed.items():
        if name == "deterministic_replay.json":
            continue
        path = out / name
        if path.exists() and file_sha256(path) != expected:
            failures.append(f"replay hash mismatch for {name}")
    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "recommended_default": recommended,
        "row_count": len(rows),
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{MILESTONE} checker")
    parser.add_argument("--out", default="target/pilot_wave/e64_field_size_and_agency_view_sweep")
    parser.add_argument("--sample-only", default=None)
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only), args.write_summary)
    else:
        result = validate_target(Path(args.out), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["failure_count"] == 0 else 1)


if __name__ == "__main__":
    main()
