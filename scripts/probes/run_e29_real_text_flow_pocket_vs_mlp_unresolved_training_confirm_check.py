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


MILESTONE = "E29_REAL_TEXT_FLOW_POCKET_VS_MLP_UNRESOLVED_TRAINING_CONFIRM"
SYSTEMS = {
    "flow_pocket_matrix_text_gradient",
    "tiny_hash_mlp_real_text_gradient",
    "keyword_regex_reference",
    "majority_answer_baseline",
    "random_control",
}
ACTIONS = ["ANSWER", "ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED"]
NON_ANSWER_ACTIONS = {"ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED"}
DECISIONS = {
    "e29_flow_pocket_matrix_beats_mlp_on_real_text_unresolved",
    "e29_flow_pocket_matrix_matches_mlp_with_better_abstention",
    "e29_mlp_baseline_beats_flow_pocket_matrix",
    "e29_real_text_needs_contrastive_bridge_for_both",
    "e29_no_clear_real_text_winner",
}
REQ_TARGET = [
    "backend_manifest.json",
    "dataset_mining_report.json",
    "mined_real_text_examples.jsonl",
    "training_curve_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "row_level_results.jsonl",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


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
        by_split = {split: [row for row in sys_rows if row["split"] == split] for split in sorted({row["split"] for row in sys_rows})}
        by_action = {action: [row for row in sys_rows if row["target_action"] == action] for action in ACTIONS}
        output[system] = {
            "overall_action_accuracy": metric(sys_rows, "correct_action"),
            "wrong_confident_answer_on_unresolved": metric([row for row in sys_rows if row["target_action"] in NON_ANSWER_ACTIONS], "wrong_confident_answer_on_unresolved"),
            "false_ask_on_answerable": metric([row for row in sys_rows if row["target_action"] == "ANSWER"], "false_ask_on_answerable"),
            "non_answer_justified_rate": metric([row for row in sys_rows if row["target_action"] in NON_ANSWER_ACTIONS], "non_answer_justified"),
            "target_action_accuracy": {action: metric(action_rows, "correct_action") for action, action_rows in by_action.items()},
            **{f"{split}_action_accuracy": metric(by_split.get(split, []), "correct_action") for split in ["train", "validation", "heldout", "phrase_holdout"]},
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
    metrics = read_json(sample_dir / "system_metrics_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    if len(rows) < 150:
        failures.append("sample row count below 150")
    if not curves:
        failures.append("sample training curve empty")
    if set(metrics) != SYSTEMS:
        failures.append("sample system set mismatch")
    if schema.get("milestone") != MILESTONE or not schema.get("flow_pocket_vs_mlp"):
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
    expected_hash = digest([{k: row[k] for k in ["example_id", "system", "target_action", "predicted_action", "correct_action", "split", "text_hash"]} for row in rows])
    if replay.get("row_level_results_sha256") != expected_hash:
        failures.append("row-level replay hash mismatch")
    if not progress:
        failures.append("progress jsonl empty")
    if not heartbeat:
        failures.append("hardware heartbeat empty")
    recomputed = summarize(rows)
    for system in SYSTEMS:
        stored = metrics.get(system, {})
        rec = recomputed.get(system, {})
        for key in ["overall_action_accuracy", "wrong_confident_answer_on_unresolved", "false_ask_on_answerable", "heldout_action_accuracy", "phrase_holdout_action_accuracy"]:
            assert_close(f"{system}.{key}", rec.get(key, 0.0), stored.get(key), failures)
    if metrics["flow_pocket_matrix_text_gradient"].get("parameter_count") == metrics["tiny_hash_mlp_real_text_gradient"].get("parameter_count"):
        warnings.append("flow and mlp parameter counts match unexpectedly")
    sample_result = None
    if sample_dir:
        sample_result = validate_sample(sample_dir, backend.get("run_id"))
        if not sample_result.get("passed"):
            failures.extend("sample: " + item for item in sample_result.get("failures", []))
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "warnings": warnings, "decision": decision.get("decision"), "run_id": backend.get("run_id"), "sample_result": sample_result}


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
            (Path(args.sample_only) / "sample_only_checker_result.json").write_text(json.dumps({"sample_only_checker_passed": result["passed"], "checker_failure_count": len(result.get("failures", [])), **result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    runner = Path(__file__).with_name("run_e29_real_text_flow_pocket_vs_mlp_unresolved_training_confirm.py")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), runner)
    if args.write_summary:
        (Path(args.out) / "target_checker_result.json").write_text(json.dumps(result | {"target_checker_passed": result["passed"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
