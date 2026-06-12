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


MILESTONE = "E33_BRIDGE_BREAKPOINT_SATURATION_LADDER"
SYSTEMS = {
    "small_workspace_d96",
    "large_workspace_d192",
    "large_workspace_trace_focus_d192",
    "oracle_text_interpreter_d96",
    "random_static_control",
}
STEPS = [
    "S0_structured_events_no_text",
    "S1_clean_symbolic_sentences",
    "S2_naturalized_templates",
    "S3_paraphrase_variation",
    "S4_decoy_dense_text",
    "S5_temporal_order_shuffle",
    "S6_missing_info_minimal_pairs",
    "S7_long_context_evidence",
    "S8_indirect_language",
    "S9_weak_mined_real_text",
]
NON_ANSWER_ACTIONS = {"ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED"}
DECISIONS = {
    "e33_controlled_bridge_clean_until_real_text_break",
    "e33_breaks_before_real_text",
    "e33_capacity_bottleneck_before_text",
    "e33_ingress_codec_bottleneck_before_text",
    "e33_weak_real_text_data_bottleneck_localized",
    "e33_no_clean_saturation_detected",
    "e33_artifact_invalid",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "saturation_plan.json",
    "saturation_ladder_report.json",
    "training_curve_report.json",
    "row_level_results.jsonl",
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
    "saturation_ladder_sample.json",
    "training_curve_sample.jsonl",
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
    out: dict[str, dict[str, Any]] = {system: {"system": system, "steps": {}} for system in SYSTEMS}
    for system in SYSTEMS:
        for step in STEPS:
            step_rows = [row for row in rows if row["system"] == system and row["step"] == step]
            res = metric(step_rows, "resolution_success")
            trace = metric(step_rows, "trace_exact")
            out[system]["steps"][step] = {
                "resolution_success": res,
                "action_accuracy": metric(step_rows, "action_correct"),
                "trace_exact": trace,
                "trace_bit_accuracy": mean([float(row["trace_bit_accuracy"]) for row in step_rows]),
                "wrong_confident_answer_on_unresolved": metric([row for row in step_rows if row["target_action"] in NON_ANSWER_ACTIONS], "wrong_confident_answer_on_unresolved"),
                "false_ask_on_answerable": metric([row for row in step_rows if row["target_action"] == "ANSWER"], "false_ask_on_answerable"),
                "clean_98": res >= 0.98 and trace >= 0.98,
                "perfect_100": res == 1.0 and trace == 1.0,
                "row_count": len(step_rows),
            }
    return out


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
    metrics = read_json(sample_dir / "system_metrics_sample.json")
    ladder = read_json(sample_dir / "saturation_ladder_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if len(rows) < 180:
        failures.append("sample row count below 180")
    if not curves:
        failures.append("sample training curve empty")
    if set(metrics) != SYSTEMS:
        failures.append("sample system set mismatch")
    if schema.get("milestone") != MILESTONE or set(schema.get("systems", [])) != SYSTEMS:
        failures.append("sample schema mismatch")
    if schema.get("steps") != STEPS:
        failures.append("sample step order mismatch")
    if "last_clean_step_by_system" not in ladder or "first_failed_step_by_system" not in ladder:
        failures.append("sample ladder missing clean/fail fields")
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
    task = read_json(out / "task_generation_report.json")
    metrics = read_json(out / "system_results.json")
    ladder = read_json(out / "saturation_ladder_report.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    replay = read_json(out / "deterministic_replay.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if backend.get("milestone") != MILESTONE:
        failures.append("backend milestone mismatch")
    if set(metrics) != SYSTEMS:
        failures.append("system set mismatch")
    if task.get("steps") != STEPS:
        failures.append("step order mismatch")
    if decision.get("decision") not in DECISIONS:
        failures.append("unknown decision label")
    if aggregate.get("decision") != decision.get("decision"):
        failures.append("decision mismatch")
    if "last_clean_step_by_system" not in ladder or "first_failed_step_by_system" not in ladder:
        failures.append("ladder missing clean/fail fields")
    if not replay.get("passed") or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay failed")
    expected_hash = digest([{k: row[k] for k in ["episode_id", "system", "split", "step", "rung", "target_action", "predicted_action", "action_correct", "trace_exact", "trace_bit_accuracy", "resolution_success", "text_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_hash:
        failures.append("row-level replay hash mismatch")
    if not progress:
        failures.append("progress jsonl empty")
    if not heartbeat:
        failures.append("hardware heartbeat empty")
    recomputed = summarize(rows)
    for system in SYSTEMS:
        for step in STEPS:
            stored = metrics.get(system, {}).get("steps", {}).get(step, {})
            rec = recomputed[system]["steps"][step]
            if stored.get("row_count", 0) <= 0:
                failures.append(f"missing rows for {system}.{step}")
            for key in ["resolution_success", "action_accuracy", "trace_exact", "trace_bit_accuracy", "wrong_confident_answer_on_unresolved", "false_ask_on_answerable"]:
                assert_close(f"{system}.{step}.{key}", rec[key], stored.get(key), failures)
            if bool(stored.get("clean_98")) != bool(rec["clean_98"]):
                failures.append(f"clean_98 mismatch {system}.{step}")
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
    runner_path = Path(__file__).with_name("run_e33_bridge_breakpoint_saturation_ladder.py")
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
