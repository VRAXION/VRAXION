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


MILESTONE = "E23_TRACE_LOCKED_ONLINE_RULESHIFT_COMPOSITION_VS_NEURAL_BASELINES"
SYSTEMS = {
    "flow_pocket_online_state_primary",
    "flow_pocket_no_ruleshift_update_ablation",
    "flow_pocket_answer_only_ablation",
    "mlp_answer_only_gradient_baseline",
    "mlp_trace_locked_gradient_baseline",
    "gru_trace_locked_gradient_baseline",
    "tiny_transformer_trace_locked_gradient_baseline",
    "tiny_transformer_curriculum_trace_locked",
    "random_static_control",
    "direct_rule_engine_invalid_control",
}
VALID_SYSTEMS = SYSTEMS - {"direct_rule_engine_invalid_control"}
DECISIONS = {
    "e23_flow_pocket_online_ruleshift_trace_advantage_confirmed",
    "e23_neural_online_adaptation_stronger",
    "e23_answer_accuracy_without_trace_validity",
    "e23_no_clear_winner",
    "e23_invalid_oracle_or_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "aggregate_metrics.json",
    "training_curve_report.json",
    "trace_validity_report.json",
    "ruleshift_generalization_report.json",
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
    "trace_locked_sample.jsonl",
    "training_curve_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "leakage_sample_audit.json",
    "boundary_claims_sample_report.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def assert_close(name: str, a: float, b: float | None, failures: list[str], tol: float = 1e-9) -> None:
    if b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol):
        failures.append(f"metric mismatch {name}: {a} != {b}")


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        sys_rows = [row for row in rows if row["system"] == system]
        splits = {split: [row for row in sys_rows if row["split"] == split] for split in sorted({row["split"] for row in sys_rows})}
        output[system] = {
            "heldout_composition_success": metric(splits.get("heldout", []), "composition_success"),
            "ood_composition_success": metric(splits.get("ood", []), "composition_success"),
            "counterfactual_composition_success": metric(splits.get("counterfactual", []), "composition_success"),
            "adversarial_composition_success": metric(splits.get("adversarial", []), "composition_success"),
            "heldout_answer_accuracy": metric(splits.get("heldout", []), "answer_correct"),
            "heldout_trace_exact": metric(splits.get("heldout", []), "trace_exact"),
            "overall_answer_accuracy": metric(sys_rows, "answer_correct"),
            "overall_trace_exact": metric(sys_rows, "trace_exact"),
            "overall_composition_success": metric(sys_rows, "composition_success"),
            "trace_bit_accuracy": mean([float(row["trace_bit_accuracy"]) for row in sys_rows]) if sys_rows else 0.0,
        }
    return output


def static_policy_check(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text())
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
    for name, expected_hash in manifest.get("sample_file_hashes", {}).items():
        path = sample_dir / name
        if not path.exists():
            failures.append("sample hash path missing " + name)
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected_hash:
            failures.append("sample hash mismatch " + name)
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    trace_rows = read_jsonl(sample_dir / "trace_locked_sample.jsonl")
    curves = read_jsonl(sample_dir / "training_curve_sample.jsonl")
    systems = read_json(sample_dir / "system_metrics_sample.json")
    metrics = read_json(sample_dir / "aggregate_metrics_sample.json")
    if len(rows) < 700:
        failures.append("sample row count below 700")
    if len(trace_rows) < 200:
        failures.append("trace sample count below 200")
    if len(curves) < 20:
        failures.append("training curve sample too small")
    if set(systems) != SYSTEMS:
        failures.append("sample system set mismatch")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("sample leakage in valid system")
    invalid_rows = [row for row in rows if row.get("system") == "direct_rule_engine_invalid_control"]
    if not invalid_rows or not all(row.get("invalid_oracle_control") and not row.get("valid_primary_system") for row in invalid_rows[:20]):
        failures.append("invalid direct control missing/not marked")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if not replay.get("passed"):
        failures.append("sample deterministic replay failed")
    leakage = read_json(sample_dir / "leakage_sample_audit.json")
    if not leakage.get("passed"):
        failures.append("sample leakage audit failed")
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
        "sample_trace_count": len(trace_rows),
        "sample_curve_count": len(curves),
        "sample_metrics": metrics,
    }


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
    systems = read_json(out / "system_results.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if summary.get("milestone") != MILESTONE:
        failures.append("summary milestone mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("unknown decision")
    if decision.get("decision") != summary.get("decision") or decision.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS or set(systems) != SYSTEMS:
        failures.append("target system set mismatch")
    if len(rows) < 10000:
        failures.append("target row-level results below 10000")
    if not progress:
        failures.append("progress.jsonl empty")
    if not heartbeat:
        failures.append("hardware heartbeat empty")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("valid target row used direct/sympy/oracle leakage")
    invalid_rows = [row for row in rows if row.get("system") == "direct_rule_engine_invalid_control"]
    if not invalid_rows or not all(row.get("invalid_oracle_control") and not row.get("valid_primary_system") for row in invalid_rows[:50]):
        failures.append("invalid direct control missing/not marked")
    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        recorded = systems.get(system, {})
        for key, value in metrics.items():
            if key in recorded:
                assert_close(system + "." + key, value, recorded.get(key), failures)
    replay = read_json(out / "deterministic_replay.json")
    expected_hash = digest([{k: row[k] for k in ["episode_id", "system", "predicted_answer", "answer_correct", "trace_exact", "composition_success", "output_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_hash or replay.get("deterministic_replay_match_rate", 0.0) < 1.0:
        failures.append("deterministic replay hash mismatch")
    leakage = read_json(out / "leakage_audit.json")
    if not leakage.get("passed"):
        failures.append("target leakage audit failed")
    sample_result = validate_sample(sample_dir, summary.get("run_id"))
    if not sample_result["passed"]:
        failures += ["sample:" + item for item in sample_result["failures"]]
    if decision.get("decision") == "e23_flow_pocket_online_ruleshift_trace_advantage_confirmed":
        flow = systems["flow_pocket_online_state_primary"]
        best_neural = max((systems[name] for name in systems if "gradient" in name or "transformer" in name), key=lambda item: item.get("heldout_composition_success") or 0.0)
        if flow["heldout_composition_success"] < best_neural["heldout_composition_success"] + 0.10:
            failures.append("flow positive decision without >=0.10 heldout composition margin")
        if min(flow["heldout_composition_success"], flow["ood_composition_success"], flow["counterfactual_composition_success"]) < 0.80:
            failures.append("flow positive decision below OOD/counterfactual gate")
    result = {
        "passed": not failures,
        "failures": failures,
        "checker_failure_count": len(failures),
        "target_checker_passed": not failures,
        "sample_only_checker_passed": sample_result["passed"],
        "decision": decision.get("decision") if not failures else "e23_invalid_oracle_or_artifact_detected",
        "recomputed_system_metrics": recomputed,
        "sample_result": sample_result,
    }
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
        output = {"decision": "e23_sample_only_replay_passed" if result["passed"] else "e23_invalid_oracle_or_artifact_detected", "checker_failure_count": 0 if result["passed"] else len(result["failures"]), "sample_only_checker_passed": result["passed"], **result}
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    runner_path = Path(__file__).with_name("run_e23_trace_locked_online_ruleshift_composition_vs_neural_baselines.py")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), runner_path)
    if args.write_summary:
        out = Path(args.out)
        summary = read_json(out / "summary.json")
        decision = read_json(out / "decision.json")
        summary["checker_failure_count"] = result["checker_failure_count"]
        summary["target_checker_passed"] = result["target_checker_passed"]
        summary["sample_only_checker_passed"] = result["sample_only_checker_passed"]
        if result["failures"]:
            summary["decision"] = "e23_invalid_oracle_or_artifact_detected"
            decision["decision"] = "e23_invalid_oracle_or_artifact_detected"
            decision["positive_gate_passed"] = False
        decision["checker_failure_count"] = result["checker_failure_count"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        (out / "e23_checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
