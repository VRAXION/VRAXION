#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E34B_ACTIVE_EVIDENCE_WORLD_WITH_NOISY_TEXT_OBSERVATIONS"
DECISIONS = {
    "e34b_noisy_text_active_evidence_confirmed",
    "e34b_text_extraction_bottleneck_detected",
    "e34b_active_policy_no_efficiency_advantage",
    "e34b_noisy_text_active_evidence_failed",
    "e34b_artifact_invalid",
}
SYSTEMS = {
    "learned_mutation_text_policy",
    "forced_initial_text_answer",
    "random_text_action_control",
    "ask_all_text_until_unique",
    "keyword_shortcut_text_control",
    "oracle_text_policy_reference",
}
SPLITS = {"heldout", "ood", "counterfactual", "adversarial"}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "text_observation_report.json",
    "policy_initial_state.json",
    "policy_final_state.json",
    "parameter_diff.json",
    "mutation_history.jsonl",
    "row_level_results.jsonl",
    "system_results.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows)


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_tokens = [
        "torch",
        "tensorflow",
        "jax",
        "backward(",
        "AdamW",
        "SGD(",
        "RMSprop",
        "optim.",
        "loss_fn",
        "GradientTape",
        "sympy",
    ]
    for token in banned_tokens:
        if token in source:
            failures.append(f"runner contains banned gradient/oracle token: {token}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            failures.append("runner calls Python eval")
    return failures


def recompute_system_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    sys_rows = [row for row in rows if row.get("system") == system]
    split_rows = {split: [row for row in sys_rows if row.get("split") == split] for split in SPLITS}
    return {
        "row_count": len(sys_rows),
        "closed_loop_success": metric(sys_rows, "closed_loop_success"),
        "answer_correct": metric(sys_rows, "answer_correct"),
        "trace_exact": metric(sys_rows, "trace_exact"),
        "wrong_confident_answer": metric(sys_rows, "wrong_confident_answer"),
        "false_ask": metric(sys_rows, "false_ask"),
        "redundant_actions": metric(sys_rows, "redundant_action"),
        "avg_steps": mean_value(sys_rows, "step_count"),
        "avg_inspects": mean_value(sys_rows, "inspect_count"),
        "text_extraction_accuracy": mean_value(sys_rows, "text_extraction_accuracy"),
        "first_useful_evidence_action": metric(sys_rows, "first_useful_evidence_action"),
        "split_closed_loop_success": {split: metric(split_rows[split], "closed_loop_success") for split in SPLITS},
        "split_text_extraction_accuracy": {split: mean_value(split_rows[split], "text_extraction_accuracy") for split in SPLITS},
        "split_avg_steps": {split: mean_value(split_rows[split], "step_count") for split in SPLITS},
    }


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-12) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


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
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema does not assert gradient_descent_used=false")
    if schema.get("noisy_text_observations") is not True:
        failures.append("sample schema missing noisy_text_observations=true")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample replay did not pass")
    if not rows:
        failures.append("empty sample row_level_sample.jsonl")
    if not history:
        failures.append("empty sample mutation_history_sample.jsonl")
    if set(system_metrics) != SYSTEMS:
        failures.append("sample system metrics missing required systems")
    if not {row.get("system") for row in rows}.issubset(SYSTEMS):
        failures.append("sample row contains unknown system")
    if any("initial_text" not in row or "extraction_events" not in row for row in rows):
        failures.append("sample row missing text/extraction fields")
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "run_id": aggregate.get("run_id"),
    }


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

    runner = Path(__file__).resolve().with_name("run_e34b_active_evidence_world_with_noisy_text_observations.py")
    failures.extend(static_runner_policy_check(runner))

    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    parameter_diff = read_json(out / "parameter_diff.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    task = read_json(out / "task_generation_report.json")
    text_report = read_json(out / "text_observation_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    history = read_jsonl(out / "mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("manifest gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across decision/summary/aggregate")
    if decision.get("checker_failure_count") != 0:
        failures.append("decision.json does not report checker_failure_count=0")
    if set(system_results) != SYSTEMS:
        failures.append("system_results missing required systems")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("backend manifest missing required systems")
    if set(task.get("actions", [])) != {"INSPECT_TEXT(feature)", "ANSWER(cause)"}:
        failures.append("task report action set mismatch")
    if text_report.get("hidden_truth_used_by_primary") is not False:
        failures.append("text report does not assert hidden_truth_used_by_primary=false")
    if text_report.get("semantic_lane_labels_used") is not False:
        failures.append("text report does not assert semantic_lane_labels_used=false")
    if not rows:
        failures.append("empty row_level_results.jsonl")
    if not history:
        failures.append("empty mutation_history.jsonl")
    if not progress:
        failures.append("empty progress.jsonl")
    if not heartbeat:
        failures.append("empty hardware_heartbeat.jsonl")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if parameter_diff.get("changed") is not True or parameter_diff.get("initial_hash") == parameter_diff.get("final_hash"):
        failures.append("missing real parameter diff")
    if int(parameter_diff.get("accepted_mutations", 0)) <= 0:
        failures.append("no accepted mutations")
    if int(parameter_diff.get("rejected_mutations", 0)) <= 0:
        failures.append("no rejected mutations")
    if int(parameter_diff.get("rollback_count", 0)) <= 0:
        failures.append("no rollback count")
    if any(row.get("system") not in SYSTEMS for row in rows):
        failures.append("row-level result contains unknown system")
    if any(row.get("split") not in SPLITS for row in rows):
        failures.append("row-level result contains unknown split")
    if any("actions" not in row or not row["actions"] for row in rows):
        failures.append("row-level result missing action trace")
    if any("initial_text" not in row or "extraction_events" not in row for row in rows):
        failures.append("row-level result missing text/extraction fields")

    if set(system_results) == SYSTEMS:
        for system in SYSTEMS:
            recomputed = recompute_system_metrics(rows, system)
            reported = system_results[system]
            if recomputed["row_count"] != reported.get("row_count"):
                failures.append(f"row_count mismatch {system}")
            for key in [
                "closed_loop_success",
                "answer_correct",
                "trace_exact",
                "wrong_confident_answer",
                "false_ask",
                "redundant_actions",
                "avg_steps",
                "avg_inspects",
                "text_extraction_accuracy",
                "first_useful_evidence_action",
            ]:
                compare_float(f"{system}.{key}", recomputed[key], reported.get(key), failures)
            for split in SPLITS:
                compare_float(
                    f"{system}.split_closed_loop_success.{split}",
                    recomputed["split_closed_loop_success"][split],
                    reported.get("split_closed_loop_success", {}).get(split),
                    failures,
                )
                compare_float(
                    f"{system}.split_text_extraction_accuracy.{split}",
                    recomputed["split_text_extraction_accuracy"][split],
                    reported.get("split_text_extraction_accuracy", {}).get(split),
                    failures,
                )
                compare_float(
                    f"{system}.split_avg_steps.{split}",
                    recomputed["split_avg_steps"][split],
                    reported.get("split_avg_steps", {}).get(split),
                    failures,
                )

    learned = system_results.get("learned_mutation_text_policy", {})
    ask_all = system_results.get("ask_all_text_until_unique", {})
    forced = system_results.get("forced_initial_text_answer", {})
    random_control = system_results.get("random_text_action_control", {})
    shortcut = system_results.get("keyword_shortcut_text_control", {})
    if aggregate.get("decision") == "e34b_noisy_text_active_evidence_confirmed":
        if learned.get("closed_loop_success", 0.0) < 0.95:
            failures.append("confirmed decision but learned success below threshold")
        if learned.get("trace_exact", 0.0) < 0.95:
            failures.append("confirmed decision but learned trace below threshold")
        if learned.get("text_extraction_accuracy", 0.0) < 0.96:
            failures.append("confirmed decision but text extraction below threshold")
        if learned.get("wrong_confident_answer", 1.0) > 0.03:
            failures.append("confirmed decision but learned wrong-confident too high")
        if learned.get("avg_steps", 999.0) >= ask_all.get("avg_steps", -1.0):
            failures.append("confirmed decision but learned is not more step-efficient than ask-all")
        if random_control.get("closed_loop_success", 1.0) >= learned.get("closed_loop_success", 0.0) - 0.20:
            failures.append("confirmed decision but random control too close")
        if forced.get("wrong_confident_answer", 0.0) < 0.80:
            failures.append("confirmed decision but forced-answer is not a failing control")
        if shortcut.get("split_closed_loop_success", {}).get("adversarial", 1.0) >= learned.get("split_closed_loop_success", {}).get("adversarial", 0.0) - 0.20:
            failures.append("confirmed decision but keyword shortcut adversarial control did not fail enough")

    sample = validate_sample(sample_dir)
    if not sample["passed"]:
        failures.extend([f"sample: {item}" for item in sample["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["target_checker_passed"] = result["passed"]
        summary["sample_only_checker_passed"] = sample["passed"]
        summary["checker_failure_count"] = len(failures)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        parser.error("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
