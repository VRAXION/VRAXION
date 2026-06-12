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


MILESTONE = "E32_FLOW_FIELD_CAPACITY_AND_TRACE_LEDGER_REPAIR_CONFIRM"
SYSTEMS = {
    "baseline_d96_p8",
    "capacity_flow_d192_p8",
    "trace_ledger_weighted_d96_p8",
    "trace_ledger_weighted_d192_p8",
    "span_bucket_aux_d96_p8",
    "ingress_event_aux_d96_p8",
    "combined_capacity_aux_d192_p8",
    "random_static_control",
}
RUNGS = [
    "R0_explicit_controlled_evidence",
    "R1_final_mixed_canonical",
    "R2_naturalized_text_canonical",
    "R3_paraphrase_variation",
    "R4_decoy_density",
    "R5_temporal_disorder",
    "R6_unresolved_answerable_minimal_pairs",
    "R7_long_context_evidence_span",
    "R8_indirect_implication_language",
    "R9_mined_real_text_weak_labels",
]
NON_ANSWER_ACTIONS = {"ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED"}
DECISIONS = {
    "e32_capacity_only_repair_confirmed",
    "e32_trace_ledger_auxiliary_positive",
    "e32_span_auxiliary_positive",
    "e32_ingress_auxiliary_positive",
    "e32_combined_capacity_auxiliary_positive",
    "e32_no_repair_confirmed",
    "e32_artifact_invalid",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "repair_plan.json",
    "repair_comparison_report.json",
    "training_curve_report.json",
    "row_level_results.jsonl",
    "trace_ledger.jsonl",
    "flow_field_snapshot.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "repair_comparison_sample.json",
    "training_curve_sample.jsonl",
    "trace_ledger_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def assert_close(name: str, actual: float, expected: float | None, failures: list[str], tol: float = 1e-9) -> None:
    if expected is None or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        failures.append(f"metric mismatch {name}: {actual} != {expected}")


def static_policy_check(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            failures.append("runner calls Python eval")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sympy":
                    failures.append("runner imports sympy")
        if isinstance(node, ast.ImportFrom) and node.module == "sympy":
            failures.append("runner imports sympy")
    return failures


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for system in sorted({row["system"] for row in rows}):
        sys_rows = [row for row in rows if row["system"] == system]
        heldout = [row for row in sys_rows if row["split"] == "heldout"]
        validation = [row for row in sys_rows if row["split"] == "validation"]
        rung_metrics: dict[str, dict[str, Any]] = {}
        for rung in RUNGS:
            rung_rows = [row for row in heldout if row["rung"] == rung]
            rung_metrics[rung] = {
                "heldout_resolution_success": metric(rung_rows, "resolution_success"),
                "heldout_action_accuracy": metric(rung_rows, "action_correct"),
                "heldout_trace_exact": metric(rung_rows, "trace_exact"),
                "heldout_trace_bit_accuracy": mean([float(row["trace_bit_accuracy"]) for row in rung_rows]),
                "row_count": len(rung_rows),
            }
        output[system] = {
            "heldout_resolution_success": metric(heldout, "resolution_success"),
            "validation_resolution_success": metric(validation, "resolution_success"),
            "heldout_action_accuracy": metric(heldout, "action_correct"),
            "heldout_trace_exact": metric(heldout, "trace_exact"),
            "heldout_trace_bit_accuracy": mean([float(row["trace_bit_accuracy"]) for row in heldout]),
            "wrong_confident_answer_on_unresolved": metric([row for row in sys_rows if row["target_action"] in NON_ANSWER_ACTIONS], "wrong_confident_answer_on_unresolved"),
            "false_ask_on_answerable": metric([row for row in sys_rows if row["target_action"] == "ANSWER"], "false_ask_on_answerable"),
            "rung_metrics": rung_metrics,
        }
    return output


def validate_sample(sample_dir: Path, expected_run_id: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append("missing sample file " + name)
    if failures:
        return {"passed": False, "failures": failures}
    manifest = read_json(sample_dir / "artifact_sample_manifest.json")
    if manifest.get("milestone") != MILESTONE:
        failures.append("sample milestone mismatch")
    if expected_run_id and manifest.get("run_id") != expected_run_id:
        failures.append("sample run_id mismatch")
    for name, expected_hash in manifest.get("sample_file_hashes", {}).items():
        path = sample_dir / name
        if not path.exists():
            failures.append("sample hash path missing " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            failures.append("sample hash mismatch " + name)
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    curves = read_jsonl(sample_dir / "training_curve_sample.jsonl")
    traces = read_jsonl(sample_dir / "trace_ledger_sample.jsonl")
    metrics = read_json(sample_dir / "system_metrics_sample.json")
    comp = read_json(sample_dir / "repair_comparison_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if len(rows) < 160:
        failures.append("sample row count below 160")
    if not curves:
        failures.append("sample training curve empty")
    if len(traces) < 50:
        failures.append("sample trace ledger too small")
    if set(metrics) != SYSTEMS:
        failures.append("sample system set mismatch")
    if "systems" not in comp or "baseline_heldout_resolution" not in comp:
        failures.append("sample repair comparison missing fields")
    if schema.get("milestone") != MILESTONE or set(schema.get("systems", [])) != SYSTEMS:
        failures.append("sample schema mismatch")
    if not replay.get("passed"):
        failures.append("sample replay failed")
    return {"passed": not failures, "failures": failures, "run_id": manifest.get("run_id")}


def validate_target(out: Path, sample_dir: Path | None, runner_path: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append("missing target file " + name)
    if runner_path and runner_path.exists():
        failures.extend(static_policy_check(runner_path))
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures, "warnings": warnings}
    backend = read_json(out / "backend_manifest.json")
    metrics = read_json(out / "system_results.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    replay = read_json(out / "deterministic_replay.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    comp = read_json(out / "repair_comparison_report.json")
    if backend.get("milestone") != MILESTONE:
        failures.append("backend milestone mismatch")
    if set(metrics) != SYSTEMS:
        failures.append("system set mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("unknown decision label")
    if aggregate.get("decision") != decision.get("decision"):
        failures.append("decision mismatch")
    if not replay.get("passed") or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay failed")
    expected_hash = digest([{k: row[k] for k in ["episode_id", "system", "split", "rung", "target_action", "predicted_action", "action_correct", "trace_exact", "trace_bit_accuracy", "resolution_success", "text_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_hash:
        failures.append("row-level replay hash mismatch")
    if not progress:
        failures.append("progress jsonl empty")
    if not heartbeat:
        failures.append("hardware heartbeat empty")
    if "systems" not in comp:
        failures.append("repair comparison missing systems")
    recomputed = summarize(rows)
    for system in SYSTEMS:
        stored = metrics.get(system, {})
        rec = recomputed.get(system, {})
        for key in [
            "heldout_resolution_success",
            "validation_resolution_success",
            "heldout_action_accuracy",
            "heldout_trace_exact",
            "heldout_trace_bit_accuracy",
            "wrong_confident_answer_on_unresolved",
            "false_ask_on_answerable",
        ]:
            assert_close(f"{system}.{key}", rec.get(key, 0.0), stored.get(key), failures)
        for rung in RUNGS:
            if stored.get("rung_metrics", {}).get(rung, {}).get("row_count", 0) <= 0:
                failures.append(f"missing rung rows for {system}.{rung}")
    sample_result = None
    if sample_dir:
        sample_result = validate_sample(sample_dir, backend.get("run_id"))
        if not sample_result.get("passed"):
            failures.extend("sample: " + item for item in sample_result.get("failures", []))
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "warnings": warnings,
        "decision": decision.get("decision"),
        "run_id": backend.get("run_id"),
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
        if args.write_summary:
            (Path(args.sample_only) / "sample_only_checker_result.json").write_text(
                json.dumps({"sample_only_checker_passed": result["passed"], "checker_failure_count": len(result.get("failures", [])), **result}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        parser.error("--out and --artifact-sample-dir are required unless --sample-only is used")
    runner_path = Path(__file__).with_name("run_e32_flow_field_capacity_and_trace_ledger_repair_confirm.py")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), runner_path)
    if args.write_summary:
        (Path(args.out) / "target_checker_result.json").write_text(
            json.dumps({"target_checker_passed": result["passed"], "checker_failure_count": result["failure_count"], **result}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
