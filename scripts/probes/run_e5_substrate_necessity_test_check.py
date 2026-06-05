#!/usr/bin/env python3
"""Checker for E5 substrate necessity test artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e5_substrate_necessity_test.py"
CHECKER = "scripts/probes/run_e5_substrate_necessity_test_check.py"
SYSTEMS = (
    "e4_top_down_hierarchical_router",
    "tiny_mlp_gradient",
    "tiny_mlp_mutation_only",
    "tiny_recurrent_gradient",
    "tiny_recurrent_mutation_only",
    "hybrid_neural_frontend_mutation_router",
    "flat_detail_scanner",
    "bottom_up_evidence_scanner",
    "random_classifier",
)
MUTATION_SYSTEMS = (
    "e4_top_down_hierarchical_router",
    "tiny_mlp_mutation_only",
    "tiny_recurrent_mutation_only",
    "hybrid_neural_frontend_mutation_router",
    "flat_detail_scanner",
    "bottom_up_evidence_scanner",
)
GRADIENT_SYSTEMS = ("tiny_mlp_gradient", "tiny_recurrent_gradient", "hybrid_neural_frontend_mutation_router")
VALID_DECISIONS = {
    "e5_neural_net_not_required_for_current_proxy",
    "e5_neural_substrate_viable",
    "e5_gradient_neural_viable_mutation_neural_not_yet",
    "e5_mutation_neural_substrate_viable",
    "e5_hybrid_neural_representation_plus_mutation_router_preferred",
    "e5_task_too_easy_redesign_required",
    "e5_neural_accuracy_without_abstraction_routing",
    "e5_leak_or_artifact_detected",
    "e5_neural_substrate_not_validated_on_current_proxy",
}
BASE_REQUIRED = (
    "e5_backend_manifest.json",
    "e5_task_generation_report.json",
    "e5_substrate_comparison_report.json",
    "e5_leakage_and_memorization_report.json",
    "e5_training_cost_report.json",
    "e5_no_synthetic_metric_audit.json",
    "e5_deterministic_replay_report.json",
    "e5_accept_reject_rollback_report.json",
    "e5_generation_metrics.json",
    "e5_row_level_eval_sample_heldout.json",
    "e5_row_level_eval_sample_ood.json",
    "e5_row_level_eval_sample_counterfactual.json",
    "e5_row_level_eval_sample_adversarial.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
)
HASH_ARTIFACTS = (
    "e5_substrate_comparison_report.json",
    "e5_leakage_and_memorization_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_SYSTEM_METRICS = (
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
    "heldout_verdict_accuracy",
    "heldout_level_accuracy",
    "heldout_causal_path_accuracy",
    "heldout_stopping_depth_accuracy",
    "heldout_over_detail_rate",
    "heldout_irrelevant_branch_rate",
    "generalization_gap",
    "parameter_count",
    "branch_order_shuffled_usefulness",
    "abstraction_routing_passed_with_leak_controls",
)


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_artifacts() -> list[str]:
    names = list(BASE_REQUIRED)
    for system in SYSTEMS:
        names.append(f"e5_candidate_{system}_summary.json")
        names.append(f"e5_parameter_diff_{system}.json")
        if system in MUTATION_SYSTEMS:
            names.append(f"e5_mutation_history_{system}.json")
        if system in GRADIENT_SYSTEMS:
            names.append(f"e5_training_history_{system}.json")
    return names


def ast_scan_runner() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    torch_import_seen = False
    adamw_seen = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_import_seen = True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "torch":
                torch_import_seen = True
        elif isinstance(node, ast.Attribute):
            if node.attr == "AdamW":
                adamw_seen = True
    if not torch_import_seen:
        failures.append("TORCH_IMPORT_MISSING_FOR_GRADIENT_BASELINE")
    if not adamw_seen:
        failures.append("ADAMW_MISSING_FOR_GRADIENT_BASELINE")
    forbidden_mutation_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {"run_vector_mutation_search", "mutate_vector_candidate", "vector_predict", "run_e4_style_search"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_function_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_mutation_calls:
                        failures.append(f"MUTATION_ONLY_BACKPROP_OR_OPTIMIZER_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in {"AdamW", "SGD", "RMSprop"}:
                    failures.append(f"MUTATION_ONLY_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
    return failures


def compare_replay(primary: Path, replay: Path) -> dict[str, Any]:
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_path = primary / name
        replay_path = replay / name
        primary_hash = file_sha256(primary_path)
        replay_hash = file_sha256(replay_path)
        comparisons[name] = {
            "primary_exists": primary_path.exists(),
            "replay_exists": replay_path.exists(),
            "primary_hash": primary_hash,
            "replay_hash": replay_hash,
            "match": primary_hash is not None and primary_hash == replay_hash,
        }
    return {
        "external_replay_compared": True,
        "external_replay_path": replay.as_posix(),
        "external_replay_passed": all(row["match"] for row in comparisons.values()),
        "external_hash_comparisons": comparisons,
    }


def update_external_replay_report(out: Path, replay: Path) -> dict[str, Any]:
    report_path = out / "e5_deterministic_replay_report.json"
    report = load_json(report_path)
    report.update(compare_replay(out, replay))
    report["deterministic_replay_passed"] = bool(report.get("internal_replay_passed") and report.get("external_replay_passed"))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def progress_rows(out: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def check_artifacts(out: Path, replay: Path | None) -> list[str]:
    failures: list[str] = []
    for name in required_artifacts():
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e5_backend_manifest.json")
    task = load_json(out / "e5_task_generation_report.json")
    substrate = load_json(out / "e5_substrate_comparison_report.json")
    leakage = load_json(out / "e5_leakage_and_memorization_report.json")
    training_cost = load_json(out / "e5_training_cost_report.json")
    audit = load_json(out / "e5_no_synthetic_metric_audit.json")
    rollback = load_json(out / "e5_accept_reject_rollback_report.json")
    generation = load_json(out / "e5_generation_metrics.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    deterministic = load_json(out / "e5_deterministic_replay_report.json")
    if replay is not None:
        deterministic = update_external_replay_report(out, replay)
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("UNKNOWN_DECISION")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if set(substrate.get("systems", {})) != set(SYSTEMS):
        failures.append("SUBSTRATE_SYSTEMS_MISMATCH")
    if set(aggregate.get("systems", {})) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEMS_MISMATCH")
    if manifest.get("gradient_backprop_allowed") is not True:
        failures.append("GRADIENT_BASELINE_NOT_DECLARED")
    if manifest.get("mutation_backend_used") is not True:
        failures.append("MUTATION_BACKEND_NOT_DECLARED")
    if manifest.get("row_level_predictions_used") is not True:
        failures.append("ROW_LEVEL_FLAG_MISSING")

    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if split not in task.get("splits", {}):
            failures.append(f"SPLIT_MISSING:{split}")
        elif task["splits"][split].get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    for key in (
        "synthetic_harness_only",
        "static_metric_dictionary_used",
        "hardcoded_improvement_used",
        "mutation_only_optimizer_used",
    ):
        if audit.get(key) is not False:
            failures.append(f"BAD_AUDIT_FLAG:{key}")
    if audit.get("row_level_predictions_used") is not True:
        failures.append("ROW_LEVEL_EVAL_NOT_PROVEN")
    if audit.get("gradient_backprop_allowed_for_gradient_systems_only") is not True:
        failures.append("GRADIENT_SCOPE_NOT_PROVEN")

    for key in (
        "route_labels_used_for_scoring",
        "route_names_used_for_scoring",
        "candidate_order_used_as_feature",
        "hidden_correct_label_used_for_scoring",
        "row_targets_available_to_score_function",
        "route_index_leak_detected",
        "candidate_name_leak_detected",
    ):
        if leakage.get(key) is not False and decision.get("decision") != "e5_leak_or_artifact_detected":
            failures.append(f"LEAKAGE_FLAG_BAD:{key}")
    if leakage.get("leakage_sentinel_passed") is not True and decision.get("decision") != "e5_leak_or_artifact_detected":
        failures.append("LEAKAGE_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("shuffled_answer_label_control_passed") is not True and decision.get("decision") != "e5_leak_or_artifact_detected":
        failures.append("SHUFFLED_LABEL_CONTROL_FAILED")
    if leakage.get("branch_order_control_passed") is not True and decision.get("decision") != "e5_leak_or_artifact_detected":
        failures.append("BRANCH_ORDER_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("branch_order_control_failures") and decision.get("decision") != "e5_leak_or_artifact_detected":
        failures.append("BRANCH_ORDER_FAILURES_WITHOUT_LEAK_DECISION")

    if rollback.get("rollback_test_passed") is not True:
        failures.append("ROLLBACK_TEST_FAILED")
    if rollback.get("accepted_mutation_count_total", 0) < 1:
        failures.append("NO_ACCEPTED_MUTATIONS_TOTAL")
    if rollback.get("rejected_mutation_count_total", 0) < 1:
        failures.append("NO_REJECTED_MUTATIONS_TOTAL")
    if rollback.get("rejected_mutation_count_total") != rollback.get("rollback_count_total"):
        failures.append("ROLLBACK_TOTAL_MISMATCH")

    for system in SYSTEMS:
        metrics = aggregate.get("systems", {}).get(system)
        if not metrics:
            failures.append(f"SYSTEM_METRICS_MISSING:{system}")
            continue
        for key in REQUIRED_SYSTEM_METRICS:
            if key not in metrics:
                failures.append(f"SYSTEM_METRIC_MISSING:{system}:{key}")
        if system != "random_classifier" and metrics.get("parameter_count", 0) < 1:
            failures.append(f"PARAMETER_COUNT_MISSING:{system}")
        diff = load_json(out / f"e5_parameter_diff_{system}.json")
        if system != "random_classifier" and diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_PARAMETER_DIFF:{system}")
        if system in MUTATION_SYSTEMS:
            hist = load_json(out / f"e5_mutation_history_{system}.json")
            if hist.get("mutation_attempt_count", 0) < 1:
                failures.append(f"NO_MUTATION_HISTORY:{system}")
            if hist.get("accepted_mutation_count", 0) < 1:
                failures.append(f"NO_ACCEPTED_MUTATIONS:{system}")
            if hist.get("rejected_mutation_count", 0) < 1:
                failures.append(f"NO_REJECTED_MUTATIONS:{system}")
            if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
                failures.append(f"ROLLBACK_MISMATCH:{system}")
            if len(generation.get(system, [])) != manifest.get("generations"):
                failures.append(f"GENERATION_COUNT_MISMATCH:{system}")
        if system in ("tiny_mlp_gradient", "tiny_recurrent_gradient"):
            hist = load_json(out / f"e5_training_history_{system}.json")
            if len(hist.get("history", [])) != manifest.get("gradient_epochs"):
                failures.append(f"EPOCH_COUNT_MISMATCH:{system}")
            if hist.get("backprop_used") is not True:
                failures.append(f"BACKPROP_NOT_DECLARED:{system}")
        if system == "hybrid_neural_frontend_mutation_router":
            hist = load_json(out / "e5_training_history_hybrid_neural_frontend_mutation_router.json")
            if len(hist.get("history", [])) != manifest.get("gradient_epochs"):
                failures.append("HYBRID_FRONTEND_EPOCH_COUNT_MISMATCH")

    for system in SYSTEMS:
        cost = training_cost.get("systems", {}).get(system)
        if not cost:
            failures.append(f"TRAINING_COST_MISSING:{system}")
        elif system != "random_classifier" and cost.get("parameter_count", 0) < 1:
            failures.append(f"TRAINING_COST_PARAMETER_COUNT_MISSING:{system}")

    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if replay is not None and deterministic.get("external_replay_passed") is not True:
        failures.append("EXTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")

    rows = progress_rows(out)
    events = {row.get("event") for row in rows}
    for event in ("startup", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if "generation_complete" not in events:
        failures.append("PROGRESS_EVENT_MISSING:generation_complete")
    if "epoch_complete" not in events:
        failures.append("PROGRESS_EVENT_MISSING:epoch_complete")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e5_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(SYSTEMS):
            failures.append(f"ROW_SAMPLE_SYSTEMS_MISMATCH:{split}")

    failures.extend(ast_scan_runner())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    replay = resolve_out(args.replay_out) if args.replay_out else None
    failures = check_artifacts(out, replay)
    result = {
        "failure_count": len(failures),
        "failures": failures,
        "out": out.as_posix(),
        "replay_out": replay.as_posix() if replay else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
