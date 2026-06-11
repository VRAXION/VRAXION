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


MILESTONE = "E22_COST_EFFICIENCY_VS_GRADIENT_BASELINES_CONFIRM"
VALID_SYSTEMS = {
    "flow_pocket_curriculum_primary",
    "flow_pocket_no_curriculum_ablation",
    "monolithic_mutation_baseline",
    "mlp_gradient_baseline",
    "gru_lstm_gradient_baseline",
    "tiny_transformer_gradient_baseline",
    "tiny_transformer_plus_curriculum",
    "random_static_controls",
}
SYSTEMS = sorted(VALID_SYSTEMS | {"oracle_sympy_direct_eval_invalid_controls"})
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "aggregate_metrics.json",
    "cost_curve_report.json",
    "accuracy_to_cost_report.json",
    "latency_report.json",
    "resource_usage_report.json",
    "trace_validity_report.json",
    "baseline_comparison_report.json",
    "leakage_audit.json",
    "deterministic_replay.json",
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
    "cost_curve_sample.jsonl",
    "row_level_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "leakage_sample_audit.json",
    "boundary_claims_sample_report.json",
    "sample_schema.json",
]
DECISIONS = {
    "e22_flow_pocket_cost_efficiency_confirmed",
    "e22_neural_baseline_more_efficient",
    "e22_transformer_curriculum_matches_flow_pocket",
    "e22_flow_pocket_accuracy_positive_but_cost_not",
    "e22_no_clear_efficiency_winner",
    "e22_invalid_oracle_or_artifact_detected",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def metric(rows: list[dict[str, Any]], key: str = "canonical_answer_correct") -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assert_close(name: str, a: float, b: float | None, failures: list[str], tol: float = 1e-9) -> None:
    if b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol):
        failures.append(f"metric mismatch {name}: {a} != {b}")


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
            failures.append("sample manifest hash path missing " + name)
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected_hash:
            failures.append("sample hash mismatch " + name)
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    curves = read_jsonl(sample_dir / "cost_curve_sample.jsonl")
    metrics = read_json(sample_dir / "aggregate_metrics_sample.json")
    systems = read_json(sample_dir / "system_metrics_sample.json")
    if len(rows) < 500:
        failures.append("row-level sample below 500")
    if len(curves) < 8:
        failures.append("cost curve sample too small")
    if set(systems) != set(SYSTEMS):
        failures.append("sample system set mismatch")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("valid sample row used direct/sympy/oracle leakage")
    if not any(row.get("system") == "oracle_sympy_direct_eval_invalid_controls" and row.get("invalid_oracle_control") for row in rows):
        failures.append("invalid oracle control missing or not marked")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if not replay.get("passed"):
        failures.append("sample deterministic replay failed")
    leakage = read_json(sample_dir / "leakage_sample_audit.json")
    if not leakage.get("passed"):
        failures.append("sample leakage audit failed")
    boundary = read_json(sample_dir / "boundary_claims_sample_report.json")
    if boundary.get("forbidden_claims_present"):
        failures.append("sample boundary claim failure")
    if metrics.get("sample_row_count") != len(rows):
        failures.append("sample row count mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "run_id": run_id,
        "sample_row_count": len(rows),
        "sample_cost_curve_count": len(curves),
        "sample_system_count": len(systems),
        "sample_metrics": metrics,
    }


def static_policy_check(runner_path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(runner_path.read_text())
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


def recompute_system_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        sys_rows = [row for row in rows if row["system"] == system]
        phases = {phase: [row for row in sys_rows if row.get("phase") == phase] for phase in sorted({row.get("phase") for row in sys_rows})}
        output[system] = {
            "heldout_accuracy": metric(phases.get("heldout_transfer", [])),
            "locked_hard_accuracy": metric(phases.get("locked_hard_posttest", [])),
            "locked_hard_pretest_accuracy": metric(phases.get("locked_hard_pretest", [])),
            "answer_accuracy": metric(sys_rows, "answer_correct"),
            "route_accuracy": metric(sys_rows, "route_correct"),
            "trace_validity": metric(sys_rows, "trace_valid"),
            "renderer_faithfulness": metric(sys_rows, "renderer_faithful"),
        }
    return output


def validate_target(out: Path, sample_dir: Path, runner_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append("missing target file " + name)
    if failures:
        return {"passed": False, "failures": failures}
    failures.extend(static_policy_check(runner_path))
    summary = read_json(out / "summary.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    manifest = read_json(out / "backend_manifest.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if summary.get("milestone") != MILESTONE:
        failures.append("summary milestone mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("unknown decision label")
    if decision.get("decision") != summary.get("decision") or decision.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch between target files")
    if set(system_results) != set(SYSTEMS):
        failures.append("target system set mismatch")
    if set(manifest.get("systems", [])) != set(SYSTEMS):
        failures.append("backend manifest system set mismatch")
    if not progress:
        failures.append("progress.jsonl empty")
    if not heartbeat:
        failures.append("hardware_heartbeat.jsonl empty")
    if len(rows) < 5000:
        failures.append("row-level target below 5000 rows")
    if any(row.get("system") in VALID_SYSTEMS and (row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary")) for row in rows):
        failures.append("valid target row used direct/sympy/oracle leakage")
    oracle_rows = [row for row in rows if row.get("system") == "oracle_sympy_direct_eval_invalid_controls"]
    if not oracle_rows or not all(row.get("invalid_oracle_control") and not row.get("valid_primary_system") for row in oracle_rows[:100]):
        failures.append("oracle/direct invalid control missing or incorrectly marked")
    recomputed = recompute_system_metrics(rows)
    for system, metrics in recomputed.items():
        recorded = system_results.get(system, {})
        for key in ["heldout_accuracy", "locked_hard_accuracy", "answer_accuracy", "route_accuracy", "trace_validity", "renderer_faithfulness"]:
            assert_close(system + "." + key, metrics[key], recorded.get(key), failures)
    replay = read_json(out / "deterministic_replay.json")
    expected_row_hash = digest([{k: row[k] for k in ["episode_id", "system", "predicted_label", "canonical_answer_correct", "output_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_row_hash or replay.get("deterministic_replay_match_rate", 0.0) < 1.0:
        failures.append("target deterministic replay hash mismatch")
    leakage = read_json(out / "leakage_audit.json")
    if not leakage.get("passed"):
        failures.append("target leakage audit failed")
    sample_result = validate_sample(sample_dir, summary.get("run_id"))
    if not sample_result["passed"]:
        failures += ["sample:" + item for item in sample_result["failures"]]
    if aggregate.get("artifact_sample_pack_passed") is not True:
        failures.append("aggregate did not mark artifact sample pack passed")
    if summary.get("hardware_dependency_status", {}).get("pytorch_available") and not any(system_results[name].get("status") == "trained" or system_results[name].get("dependency_status") == "pytorch_available" for name in VALID_SYSTEMS if "gradient" in name or "transformer" in name):
        failures.append("PyTorch available but no neural baseline training recorded")
    if decision.get("decision") == "e22_flow_pocket_cost_efficiency_confirmed":
        flow = system_results["flow_pocket_curriculum_primary"]
        best_neural = max((system_results[name] for name in system_results if name in {"mlp_gradient_baseline", "gru_lstm_gradient_baseline", "tiny_transformer_gradient_baseline", "tiny_transformer_plus_curriculum"}), key=lambda item: item.get("heldout_accuracy") or 0.0)
        if flow.get("heldout_accuracy", 0.0) + 1e-9 < best_neural.get("heldout_accuracy", 0.0):
            failures.append("flow positive decision despite lower neural heldout accuracy")
        if flow.get("trace_validity", 0.0) < 0.85:
            failures.append("flow positive decision despite weak trace validity")
    result = {
        "passed": not failures,
        "failures": failures,
        "checker_failure_count": len(failures),
        "target_checker_passed": not failures,
        "sample_only_checker_passed": sample_result["passed"],
        "decision": decision.get("decision") if not failures else "e22_invalid_oracle_or_artifact_detected",
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
        output = {
            "decision": "e22_sample_only_replay_passed" if result["passed"] else "e22_invalid_oracle_or_artifact_detected",
            "checker_failure_count": 0 if result["passed"] else len(result["failures"]),
            "sample_only_checker_passed": result["passed"],
            **result,
        }
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    runner_path = Path(__file__).with_name("run_e22_cost_efficiency_vs_gradient_baselines_confirm.py")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), runner_path)
    if args.write_summary:
        out = Path(args.out)
        summary = read_json(out / "summary.json")
        decision = read_json(out / "decision.json")
        summary["checker_failure_count"] = result["checker_failure_count"]
        summary["target_checker_passed"] = result["target_checker_passed"]
        summary["sample_only_checker_passed"] = result["sample_only_checker_passed"]
        if result["failures"]:
            summary["decision"] = "e22_invalid_oracle_or_artifact_detected"
            decision["decision"] = "e22_invalid_oracle_or_artifact_detected"
            decision["positive_gate_passed"] = False
        decision["checker_failure_count"] = result["checker_failure_count"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        (out / "e22_checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
