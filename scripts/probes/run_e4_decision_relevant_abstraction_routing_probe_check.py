#!/usr/bin/env python3
"""Checker for E4 decision-relevant abstraction routing artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e4_decision_relevant_abstraction_routing_probe.py"
CHECKER = "scripts/probes/run_e4_decision_relevant_abstraction_routing_probe_check.py"
SYSTEMS = (
    "flat_detail_scanner",
    "bottom_up_evidence_scanner",
    "top_down_hierarchical_router",
    "dynamic_state_medium_router",
)
VALID_DECISIONS = {
    "e4_decision_relevant_abstraction_routing_confirmed",
    "e4_flat_detail_scanning_sufficient",
    "e4_answer_level_selection_failure",
    "e4_overbranching_failure",
    "e4_leak_or_task_artifact_detected",
}
BASE_REQUIRED = (
    "e4_backend_manifest.json",
    "e4_task_generation_report.json",
    "e4_routing_report.json",
    "e4_control_baseline_report.json",
    "e4_leakage_sentinel_report.json",
    "e4_no_synthetic_metric_audit.json",
    "e4_deterministic_replay_report.json",
    "e4_accept_reject_rollback_report.json",
    "e4_generation_metrics.json",
    "e4_row_level_eval_sample_heldout.json",
    "e4_row_level_eval_sample_ood.json",
    "e4_row_level_eval_sample_counterfactual.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
)
HASH_ARTIFACTS = (
    "e4_routing_report.json",
    "e4_control_baseline_report.json",
    "e4_leakage_sentinel_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
SYSTEM_REQUIRED_METRICS = (
    "heldout_verdict_accuracy",
    "heldout_level_accuracy",
    "heldout_causal_path_accuracy",
    "heldout_stopping_depth_accuracy",
    "heldout_over_detail_rate",
    "heldout_under_detail_rate",
    "heldout_irrelevant_branch_rate",
    "heldout_top_down_path_consistency",
    "heldout_usefulness",
    "heldout_detail_efficiency",
    "ood_usefulness",
    "counterfactual_usefulness",
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
        names.extend(
            [
                f"e4_candidate_{system}_initial.json",
                f"e4_candidate_{system}_final.json",
                f"e4_parameter_diff_{system}.json",
                f"e4_mutation_history_{system}.json",
            ]
        )
    return names


def ast_scan(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_imports = {"torch", "tensorflow", "jax", "sklearn"}
    forbidden_calls = {"backward", "grad", "fit", "partial_fit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_imports:
                    failures.append(f"FORBIDDEN_IMPORT:{alias.name}:{path.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in forbidden_imports:
                failures.append(f"FORBIDDEN_IMPORT_FROM:{node.module}:{path.name}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in forbidden_calls:
                failures.append(f"FORBIDDEN_OPTIMIZER_OR_BACKPROP_CALL:{name}:{path.name}")
    return failures


def check_static_files() -> list[str]:
    failures: list[str] = []
    for rel in (RUNNER, CHECKER):
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{rel}")
        else:
            failures.extend(ast_scan(path))
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
    report_path = out / "e4_deterministic_replay_report.json"
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

    manifest = load_json(out / "e4_backend_manifest.json")
    task = load_json(out / "e4_task_generation_report.json")
    routing = load_json(out / "e4_routing_report.json")
    controls = load_json(out / "e4_control_baseline_report.json")
    leakage = load_json(out / "e4_leakage_sentinel_report.json")
    audit = load_json(out / "e4_no_synthetic_metric_audit.json")
    rollback = load_json(out / "e4_accept_reject_rollback_report.json")
    generation = load_json(out / "e4_generation_metrics.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    deterministic = load_json(out / "e4_deterministic_replay_report.json")
    if replay is not None:
        deterministic = update_external_replay_report(out, replay)
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("UNKNOWN_DECISION")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if set(routing.get("systems", {})) != set(SYSTEMS):
        failures.append("ROUTING_SYSTEMS_MISMATCH")
    if set(aggregate.get("systems", {})) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEMS_MISMATCH")
    if manifest.get("reference_systems") != ["oracle_reference_only"]:
        failures.append("ORACLE_REFERENCE_MISSING_OR_NOT_REFERENCE")
    if manifest.get("candidate_state_created") is not True or manifest.get("mutation_backend_used") is not True:
        failures.append("BACKEND_FLAGS_NOT_PROVEN")

    for split in ("train", "validation", "heldout", "ood", "counterfactual"):
        if split not in task.get("splits", {}):
            failures.append(f"SPLIT_MISSING:{split}")
        elif task["splits"][split].get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    for key in (
        "synthetic_harness_only",
        "static_metric_dictionary_used",
        "hardcoded_improvement_used",
        "gradient_backprop_used",
        "real_optimizer_detected",
    ):
        if audit.get(key) is not False:
            failures.append(f"BAD_AUDIT_FLAG:{key}")
    if audit.get("row_level_predictions_used") is not True:
        failures.append("ROW_LEVEL_EVAL_NOT_PROVEN")

    if controls.get("controls_do_not_solve_task") is not True and decision.get("decision") != "e4_leak_or_task_artifact_detected":
        failures.append("CONTROLS_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("leakage_sentinel_passed") is not True and decision.get("decision") != "e4_leak_or_task_artifact_detected":
        failures.append("LEAKAGE_FAILED_WITHOUT_LEAK_DECISION")
    for key in (
        "route_labels_used_for_scoring",
        "route_names_used_for_scoring",
        "candidate_order_used_as_feature",
        "hidden_correct_label_used_for_scoring",
        "row_targets_available_to_score_function",
        "route_index_leak_detected",
        "candidate_name_leak_detected",
    ):
        if leakage.get(key) is not False and decision.get("decision") != "e4_leak_or_task_artifact_detected":
            failures.append(f"LEAKAGE_FLAG_BAD:{key}")

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
        for key in SYSTEM_REQUIRED_METRICS:
            if key not in metrics:
                failures.append(f"SYSTEM_METRIC_MISSING:{system}:{key}")
        diff = load_json(out / f"e4_parameter_diff_{system}.json")
        hist = load_json(out / f"e4_mutation_history_{system}.json")
        if diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_PARAMETER_DIFF:{system}")
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

    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if replay is not None and deterministic.get("external_replay_passed") is not True:
        failures.append("EXTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")

    rows = progress_rows(out)
    events = {row.get("event") for row in rows}
    for event in ("startup", "generation_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    generation_events = [row for row in rows if row.get("event") == "generation_complete"]
    expected_generations = int(manifest.get("generations", 0)) * len(SYSTEMS)
    if len(generation_events) < expected_generations:
        failures.append("PROGRESS_GENERATION_EVENT_COUNT_TOO_LOW")

    for split in ("heldout", "ood", "counterfactual"):
        sample = load_json(out / f"e4_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(SYSTEMS):
            failures.append(f"ROW_SAMPLE_SYSTEMS_MISMATCH:{split}")

    failures.extend(check_static_files())
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
