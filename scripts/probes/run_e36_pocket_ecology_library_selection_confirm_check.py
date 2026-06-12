#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E36_POCKET_ECOLOGY_LIBRARY_SELECTION_CONFIRM"
DECISIONS = {
    "e36_pocket_ecology_selection_confirmed",
    "e36_pocket_ecology_selection_partial",
    "e36_pocket_ecology_negative_transfer",
    "e36_no_ecology_advantage_detected",
    "e36_invalid_artifact_or_oracle_detected",
}
SYSTEMS = {
    "no_library_scratch",
    "random_library_import",
    "unfiltered_library_import",
    "evaluated_library_import",
    "evaluated_library_plus_adapter",
    "wrong_toxic_pocket_control",
    "oracle_invalid_control",
}
VALID_SYSTEMS = SYSTEMS - {"oracle_invalid_control"}
SPLITS = {
    "same_packet_clean",
    "same_continuous_stream",
    "target_packet_clean",
    "target_continuous_stream",
    "target_adversarial_decoy",
    "target_bit_insert",
    "target_bit_drop",
}
TARGET_SPLITS = {split for split in SPLITS if split.startswith("target_")}
STABLE_TARGET_SPLITS = {
    "target_packet_clean",
    "target_continuous_stream",
    "target_adversarial_decoy",
}
BITSLIP_TARGET_SPLITS = {"target_bit_insert", "target_bit_drop"}
CANDIDATES = {
    "protocol_framing_ingress_v001",
    "protocol_framing_no_adapter",
    "dirty_start_only_decoder",
    "wrong_rotated_codebook_pocket",
    "dormant_unused_pocket",
}
REQ_TARGET = [
    "backend_manifest.json",
    "ecology_world_report.json",
    "candidate_pocket_report.json",
    "pocket_value_report.json",
    "ecology_selection_report.json",
    "paired_ablation_report.json",
    "selection_history.jsonl",
    "row_level_results.jsonl",
    "system_results.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "selection_history_sample.jsonl",
    "pocket_value_sample.json",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows) if rows else 0.0


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def row_utility(row: dict[str, Any]) -> float:
    return (
        (1.0 if row.get("closed_loop_success") else 0.0)
        + 0.35 * (1.0 if row.get("trace_exact") else 0.0)
        - 5.0 * float(row.get("wrong_feature_write_rate", 0.0))
        - 3.0 * float(row.get("false_frame_commit_rate", 0.0))
        - 0.02 * float(row.get("step_count", 0.0))
        - float(row.get("call_cost", 0.0))
    )


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-12) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["torch", "tensorflow", "jax", "backward(", "AdamW", "SGD(", "optim.", "sympy", "eval(", "hash("]:
        if token in source:
            failures.append(f"runner contains banned nondeterministic/gradient/oracle token: {token}")
    return failures


def recompute_system_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    sys_rows = [row for row in rows if row.get("system") == system]
    target_rows = [row for row in sys_rows if row.get("split") in TARGET_SPLITS]
    stable_rows = [row for row in sys_rows if row.get("split") in STABLE_TARGET_SPLITS]
    bitslip_rows = [row for row in sys_rows if row.get("split") in BITSLIP_TARGET_SPLITS]
    return {
        "row_count": len(sys_rows),
        "target_world_success": metric(target_rows, "closed_loop_success"),
        "stable_target_success": metric(stable_rows, "closed_loop_success"),
        "bitslip_target_success": metric(bitslip_rows, "closed_loop_success"),
        "answer_correct": metric(sys_rows, "answer_correct"),
        "trace_exact": metric(sys_rows, "trace_exact"),
        "wrong_confident_answer": metric(sys_rows, "wrong_confident_answer"),
        "wrong_feature_write_rate": mean_value(target_rows, "wrong_feature_write_rate"),
        "false_frame_commit_rate": mean_value(target_rows, "false_frame_commit_rate"),
        "avg_steps": mean_value(sys_rows, "step_count"),
        "utility": sum(row_utility(row) for row in target_rows) / len(target_rows) if target_rows else 0.0,
    }


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    system_metrics = read_json(sample_dir / "system_metrics_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    pocket_values = read_json(sample_dir / "pocket_value_sample.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "selection_history_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if schema.get("milestone") != MILESTONE or schema.get("pocket_ecology") is not True:
        failures.append("sample schema missing E36 pocket ecology marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if set(system_metrics) != SYSTEMS:
        failures.append("sample system metrics missing required systems")
    if set(pocket_values) != CANDIDATES:
        failures.append("sample pocket values missing required candidates")
    statuses = {cid: value.get("status") for cid, value in pocket_values.items()}
    if statuses.get("wrong_rotated_codebook_pocket") != "banned":
        failures.append("sample toxic pocket was not banned")
    if statuses.get("dormant_unused_pocket") != "deprecated":
        failures.append("sample dormant pocket was not deprecated")
    if not rows or any("candidate_id" not in row or "world_id" not in row for row in rows):
        failures.append("sample row-level ecology fields missing")
    if not history:
        failures.append("sample selection history empty")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    runner = Path(__file__).resolve().with_name("run_e36_pocket_ecology_library_selection_confirm.py")
    failures.extend(static_runner_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    world_report = read_json(out / "ecology_world_report.json")
    candidates = read_json(out / "candidate_pocket_report.json")
    pocket_values = read_json(out / "pocket_value_report.json")
    ecology = read_json(out / "ecology_selection_report.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    history = read_jsonl(out / "selection_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if set(system_results) != SYSTEMS or set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("missing required systems")
    if set(candidates) != CANDIDATES or set(pocket_values) != CANDIDATES:
        failures.append("missing required pocket candidates")
    if set(world_report.get("splits", [])) != SPLITS:
        failures.append("split mismatch")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not history or not progress or not heartbeat:
        failures.append("empty required row/history/progress/heartbeat artifact")
    if any(row.get("system") not in SYSTEMS for row in rows):
        failures.append("unknown system in row-level results")
    if any(row.get("split") not in SPLITS for row in rows):
        failures.append("unknown split in row-level results")
    if any("candidate_id" not in row or "world_id" not in row or "output_hash" not in row for row in rows):
        failures.append("row-level ecology fields missing")

    for system in SYSTEMS:
        recomputed = recompute_system_metrics(rows, system)
        reported = system_results.get(system, {})
        if recomputed["row_count"] != reported.get("row_count"):
            failures.append(f"row_count mismatch {system}")
        for key in [
            "target_world_success",
            "stable_target_success",
            "bitslip_target_success",
            "answer_correct",
            "trace_exact",
            "wrong_confident_answer",
            "wrong_feature_write_rate",
            "false_frame_commit_rate",
            "avg_steps",
            "utility",
        ]:
            compare_float(f"{system}.{key}", recomputed[key], reported.get(key), failures)

    statuses = {cid: value.get("status") for cid, value in pocket_values.items()}
    promoted = [cid for cid, status in statuses.items() if status in {"stable", "core"}]
    banned = [cid for cid, status in statuses.items() if status == "banned"]
    deprecated = [cid for cid, status in statuses.items() if status == "deprecated"]
    if statuses.get("wrong_rotated_codebook_pocket") != "banned":
        failures.append("toxic rotated pocket was not banned")
    if statuses.get("dormant_unused_pocket") != "deprecated":
        failures.append("dormant pocket was not deprecated")
    if "dirty_start_only_decoder" in promoted:
        failures.append("dirty scratch decoder was promoted as stable/core library pocket")
    if ecology.get("evaluated_selected_candidate") not in CANDIDATES:
        failures.append("invalid evaluated selected candidate")
    if sorted(ecology.get("promoted_candidates", [])) != sorted(promoted):
        failures.append("ecology promoted list mismatch")
    if sorted(ecology.get("banned_candidates", [])) != sorted(banned):
        failures.append("ecology banned list mismatch")
    if sorted(ecology.get("deprecated_candidates", [])) != sorted(deprecated):
        failures.append("ecology deprecated list mismatch")

    evaluated = system_results.get("evaluated_library_plus_adapter", {})
    random_lib = system_results.get("random_library_import", {})
    toxic = system_results.get("wrong_toxic_pocket_control", {})
    scratch = system_results.get("no_library_scratch", {})
    if aggregate.get("decision") in {"e36_pocket_ecology_selection_confirmed", "e36_pocket_ecology_selection_partial"}:
        if "protocol_framing_ingress_v001" not in promoted:
            failures.append("positive/partial decision without promoted protocol framing pocket")
        if evaluated.get("stable_target_success", 0.0) < 0.98:
            failures.append("evaluated stable target too weak for positive/partial decision")
        if evaluated.get("wrong_feature_write_rate", 1.0) > 0.005:
            failures.append("evaluated write safety too weak for positive/partial decision")
        if evaluated.get("target_world_success", 0.0) < random_lib.get("target_world_success", 0.0) + 0.10:
            failures.append("evaluated library did not beat random library sufficiently")
        if toxic.get("target_world_success", 1.0) >= evaluated.get("target_world_success", 0.0) - 0.05:
            failures.append("toxic pocket control too close to evaluated library")
        if not banned or not deprecated:
            failures.append("positive/partial decision without banned and deprecated lifecycle outcomes")
    if aggregate.get("decision") == "e36_pocket_ecology_selection_confirmed":
        if evaluated.get("target_world_success", 0.0) < scratch.get("target_world_success", 0.0):
            failures.append("confirmed decision while evaluated target score is below scratch")
        if evaluated.get("bitslip_target_success", 0.0) < 0.90:
            failures.append("confirmed decision while bit-slip transfer is still unsolved")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample_result,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
