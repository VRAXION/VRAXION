#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


MILESTONE = "E26_HARD_SKIP_TEXT_REASONING_FAILURE_MAP"
SYSTEMS = {
    "flow_pocket_hard_skip_primary",
    "parser_only_control",
    "flow_pocket_marker_shortcut_ablation",
    "flow_pocket_stale_rule_retention_ablation",
    "flow_pocket_answer_only_ablation",
    "flow_pocket_no_memory_ablation",
    "flow_pocket_no_paraphrase_generalization_ablation",
    "flow_pocket_no_evidence_span_tracking_ablation",
    "mlp_text_feature_gradient_baseline",
    "gru_text_gradient_baseline",
    "tiny_transformer_text_gradient_baseline",
    "tiny_transformer_text_curriculum_gradient",
    "random_static_control",
    "oracle_text_parser_invalid_control",
    "direct_rule_engine_invalid_control",
}
VALID_SYSTEMS = {name for name in SYSTEMS if not name.endswith("_invalid_control")}
NEURAL_SYSTEMS = {
    "mlp_text_feature_gradient_baseline",
    "gru_text_gradient_baseline",
    "tiny_transformer_text_gradient_baseline",
    "tiny_transformer_text_curriculum_gradient",
}
DECISIONS = {
    "e26_hard_skip_text_reasoning_confirmed",
    "e26_hard_skip_text_reasoning_partial",
    "e26_hard_skip_text_reasoning_failed_with_failure_map",
    "e26_invalid_oracle_or_artifact_detected",
}
HARD_FAMILIES = [
    {"name": "stage1_bridge_single_shift", "stage": 1, "expected_bottleneck": "none"},
    {"name": "stage2_multi_rule_document", "stage": 2, "expected_bottleneck": "multi_rule_tracking"},
    {"name": "stage3_long_decoy_dense", "stage": 3, "expected_bottleneck": "evidence_retrieval_under_decoys"},
    {"name": "stage4_temporal_disorder", "stage": 4, "expected_bottleneck": "temporal_order_reconstruction"},
    {"name": "stage5_missing_evidence_ambiguous", "stage": 5, "expected_bottleneck": "missing_or_underdetermined_evidence"},
    {"name": "stage6_indirect_language", "stage": 6, "expected_bottleneck": "paraphrase_semantics"},
    {"name": "stage7_long_context_memory", "stage": 7, "expected_bottleneck": "long_context_memory"},
]
SPLITS = [item["name"] for item in HARD_FAMILIES]
FAMILY_STAGE = {item["name"]: item["stage"] for item in HARD_FAMILIES}
FAMILY_BOTTLENECK = {item["name"]: item["expected_bottleneck"] for item in HARD_FAMILIES}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "aggregate_metrics.json",
    "training_curve_report.json",
    "trace_discovery_report.json",
    "ruleshift_generalization_report.json",
    "failure_map_report.json",
    "baseline_comparison_report.json",
    "leakage_audit.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "row_level_results.jsonl",
    "system_results.json",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "trace_discovery_sample.jsonl",
    "training_curve_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "leakage_sample_audit.json",
    "boundary_claims_sample_report.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def assert_close(name: str, a: float, b: float | None, failures: list[str], tol: float = 1e-9) -> None:
    if b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol):
        failures.append(f"metric mismatch {name}: {a} != {b}")


def static_policy_check(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sympy":
                    failures.append("runner imports sympy")
        if isinstance(node, ast.ImportFrom) and node.module == "sympy":
            failures.append("runner imports sympy")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            failures.append("runner calls Python eval")
    return failures


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    scenarios = sorted({row["scenario"] for row in rows})
    for system in sorted({row["system"] for row in rows}):
        sys_rows = [row for row in rows if row["system"] == system]
        splits = {split: [row for row in sys_rows if row["split"] == split] for split in sorted({row["split"] for row in sys_rows})}
        by_scenario = {scenario: [row for row in sys_rows if row["scenario"] == scenario] for scenario in scenarios}
        split_metrics = {f"{split}_composition_success": metric(splits.get(split, []), "composition_success") for split in SPLITS}
        stage1_rows = splits.get("stage1_bridge_single_shift", [])
        output[system] = {
            **split_metrics,
            "heldout_answer_accuracy": metric(stage1_rows, "answer_correct"),
            "heldout_trace_exact": metric(stage1_rows, "trace_exact"),
            "evidence_span_validity": metric(sys_rows, "evidence_span_valid"),
            "wrong_trace_rate": metric(sys_rows, "wrong_trace_rate"),
            "hallucinated_rule_update_rate": metric(sys_rows, "hallucinated_rule_update"),
            "overall_composition_success": metric(sys_rows, "composition_success"),
            "overall_answer_accuracy": metric(sys_rows, "answer_correct"),
            "overall_trace_exact": metric(sys_rows, "trace_exact"),
            "trace_bit_accuracy": mean([float(row["trace_bit_accuracy"]) for row in sys_rows]) if sys_rows else 0.0,
            "scenario_composition_success": {k: metric(v, "composition_success") for k, v in by_scenario.items()},
        }
    return output


def validate_invalid_controls(rows: list[dict[str, Any]], failures: list[str], prefix: str = "") -> None:
    for invalid_name in {"direct_rule_engine_invalid_control", "oracle_text_parser_invalid_control"}:
        invalid_rows = [row for row in rows if row.get("system") == invalid_name]
        if not invalid_rows or not all(row.get("invalid_oracle_control") and not row.get("valid_primary_system") for row in invalid_rows[:50]):
            failures.append(f"{prefix}invalid control {invalid_name} missing/not marked")


def validate_sample(sample_dir: Path, expected_run_id: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append("missing sample file " + name)
    if failures:
        return {"passed": False, "failures": failures}
    manifest = read_json(sample_dir / "artifact_sample_manifest.json")
    run_id = manifest.get("run_id")
    if expected_run_id and run_id != expected_run_id:
        failures.append("sample run_id mismatch")
    if manifest.get("milestone") != MILESTONE:
        failures.append("sample milestone mismatch")
    for name, expected_hash in manifest.get("sample_file_hashes", {}).items():
        path = sample_dir / name
        if not path.exists():
            failures.append("sample hash path missing " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            failures.append("sample hash mismatch " + name)
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    traces = read_jsonl(sample_dir / "trace_discovery_sample.jsonl")
    curves = read_jsonl(sample_dir / "training_curve_sample.jsonl")
    metrics = read_json(sample_dir / "aggregate_metrics_sample.json")
    systems = read_json(sample_dir / "system_metrics_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    if len(rows) < 700:
        failures.append("sample row count below 700")
    if len(traces) < 200:
        failures.append("trace sample count below 200")
    if len(curves) < 20:
        failures.append("training curve sample too small")
    if set(systems) != SYSTEMS:
        failures.append("sample system set mismatch")
    if schema.get("failure_map") is not True or schema.get("trace_locked") is not True:
        failures.append("sample schema missing trace-locked failure-map flags")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("sample leakage in valid system")
    validate_invalid_controls(rows, failures, "sample ")
    leakage = read_json(sample_dir / "leakage_sample_audit.json")
    if not leakage.get("passed") or leakage.get("explicit_shift_assignment_visible") or leakage.get("structured_support_rows_visible"):
        failures.append("sample leakage audit failed")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if not replay.get("passed"):
        failures.append("sample deterministic replay failed")
    boundary = read_json(sample_dir / "boundary_claims_sample_report.json")
    if boundary.get("forbidden_claims_present"):
        failures.append("sample boundary failure")
    if metrics.get("sample_row_count") != len(rows):
        failures.append("sample row count mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "run_id": run_id,
        "sample_row_count": len(rows),
        "sample_trace_count": len(traces),
        "sample_curve_count": len(curves),
        "sample_metrics": metrics,
    }


def validate_failure_map(decision: dict[str, Any], systems: dict[str, Any], failure_map: dict[str, Any], failures: list[str]) -> None:
    if failure_map.get("decision") != decision.get("decision"):
        failures.append("failure map decision mismatch")
    if failure_map.get("hard_families") != HARD_FAMILIES:
        failures.append("hard families mismatch")
    flow = systems.get("flow_pocket_hard_skip_primary", {})
    family_scores = {split: flow.get(f"{split}_composition_success") for split in SPLITS}
    if failure_map.get("family_scores") != family_scores:
        failures.append("failure map family scores mismatch")
    first_failing = next((split for split in SPLITS if float(family_scores.get(split) or 0.0) < 0.80), None)
    best_stage = max((FAMILY_STAGE[split] for split in SPLITS if float(family_scores.get(split) or 0.0) >= 0.80), default=0)
    if failure_map.get("first_failing_family") != first_failing:
        failures.append("first failing family mismatch")
    if failure_map.get("best_stage_passed") != best_stage:
        failures.append("best stage passed mismatch")
    if failure_map.get("failure_signature") != FAMILY_BOTTLENECK.get(first_failing or "", "none"):
        failures.append("failure signature mismatch")
    dec = decision.get("decision")
    if dec == "e26_hard_skip_text_reasoning_confirmed":
        if first_failing is not None:
            failures.append("confirmed decision has failing family")
        if min(float(flow.get(f"{split}_composition_success") or 0.0) for split in SPLITS) < 0.80:
            failures.append("confirmed decision below all-family gate")
    elif dec == "e26_hard_skip_text_reasoning_partial":
        if first_failing is None:
            failures.append("partial decision without failing family")
        if best_stage < 1:
            failures.append("partial decision without any passed stage")
        if not failure_map.get("recommended_repair"):
            failures.append("partial decision missing recommended repair")
    elif dec == "e26_hard_skip_text_reasoning_failed_with_failure_map":
        if best_stage > 0:
            failures.append("failed-with-map decision still passed a stage")


def validate_target(out: Path, sample_dir: Path, runner_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append("missing target file " + name)
    if failures:
        return {"passed": False, "failures": failures, "checker_failure_count": len(failures)}
    failures.extend(static_policy_check(runner_path))
    summary = read_json(out / "summary.json")
    decision = read_json(out / "decision.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    manifest = read_json(out / "backend_manifest.json")
    task = read_json(out / "task_generation_report.json")
    systems = read_json(out / "system_results.json")
    failure_map = read_json(out / "failure_map_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if summary.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE or manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("unknown decision")
    if decision.get("decision") != summary.get("decision") or decision.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS or set(systems) != SYSTEMS:
        failures.append("system set mismatch")
    if task.get("hard_skip_failure_map") is not True:
        failures.append("hard-skip failure-map flag missing")
    if task.get("explicit_shift_assignment_visible") is not False:
        failures.append("explicit shift assignment visible")
    if task.get("structured_support_rows_visible") is not False:
        failures.append("structured support rows visible")
    if task.get("naturalized_text_evidence_stream") is not True:
        failures.append("naturalized text evidence stream flag missing")
    if task.get("splits") != SPLITS:
        failures.append("split list mismatch")
    if task.get("hard_families") != HARD_FAMILIES:
        failures.append("task hard families mismatch")
    if len(rows) < 15000:
        failures.append("target row-level results below 15000")
    if not progress:
        failures.append("progress.jsonl empty")
    if not heartbeat:
        failures.append("hardware_heartbeat.jsonl empty")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("valid target row used direct/sympy/oracle leakage")
    validate_invalid_controls(rows, failures)
    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        recorded = systems.get(system, {})
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sk, sv in value.items():
                    assert_close(f"{system}.{key}.{sk}", sv, recorded.get(key, {}).get(sk), failures)
            elif key in recorded:
                assert_close(f"{system}.{key}", value, recorded.get(key), failures)
    replay = read_json(out / "deterministic_replay.json")
    expected_hash = digest([{k: row[k] for k in ["episode_id", "system", "predicted_answer", "answer_correct", "trace_exact", "composition_success", "output_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_hash or replay.get("deterministic_replay_match_rate", 0.0) < 1.0 or not replay.get("passed"):
        failures.append("deterministic replay hash mismatch")
    leakage = read_json(out / "leakage_audit.json")
    if not leakage.get("passed") or leakage.get("explicit_shift_assignment_visible") or leakage.get("structured_support_rows_visible"):
        failures.append("target leakage audit failed")
    validate_failure_map(decision, systems, failure_map, failures)
    sample_result = validate_sample(sample_dir, summary.get("run_id"))
    if not sample_result["passed"]:
        failures += ["sample:" + item for item in sample_result["failures"]]
    return {
        "passed": not failures,
        "failures": failures,
        "checker_failure_count": len(failures),
        "target_checker_passed": not failures,
        "sample_only_checker_passed": sample_result["passed"],
        "decision": decision.get("decision") if not failures else "e26_invalid_oracle_or_artifact_detected",
        "failure_map": failure_map,
        "recomputed_system_metrics": recomputed,
        "sample_result": sample_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        output = {
            "decision": "e26_sample_only_replay_passed" if result["passed"] else "e26_invalid_oracle_or_artifact_detected",
            "checker_failure_count": 0 if result["passed"] else len(result["failures"]),
            "sample_only_checker_passed": result["passed"],
            **result,
        }
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    runner_path = Path(__file__).with_name("run_e26_hard_skip_text_reasoning_failure_map.py")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), runner_path)
    if args.write_summary:
        out = Path(args.out)
        summary = read_json(out / "summary.json")
        decision = read_json(out / "decision.json")
        summary["checker_failure_count"] = result["checker_failure_count"]
        summary["target_checker_passed"] = result["target_checker_passed"]
        summary["sample_only_checker_passed"] = result["sample_only_checker_passed"]
        if result["failures"]:
            summary["decision"] = "e26_invalid_oracle_or_artifact_detected"
            decision["decision"] = "e26_invalid_oracle_or_artifact_detected"
            decision["positive_gate_passed"] = False
        decision["checker_failure_count"] = result["checker_failure_count"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "e26_checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
