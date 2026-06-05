#!/usr/bin/env python3
"""Checker for E1 real-backend continuous state medium probe."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/e1_real_backend_continuous_state_medium_probe")
RUNNER = "scripts/probes/run_e1_real_backend_continuous_state_medium_probe.py"
CHECKER = "scripts/probes/run_e1_real_backend_continuous_state_medium_probe_check.py"
DOCS = (
    "docs/research/E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE_CONTRACT.md",
    "docs/research/E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "e1_online_check_report.md",
    "e1_backend_manifest.json",
    "e1_task_generation_report.json",
    "e1_candidate_flat_initial.json",
    "e1_candidate_flat_final.json",
    "e1_candidate_state_medium_initial.json",
    "e1_candidate_state_medium_final.json",
    "e1_candidate_gated_state_medium_initial.json",
    "e1_candidate_gated_state_medium_final.json",
    "e1_parameter_diff_flat.json",
    "e1_parameter_diff_state_medium.json",
    "e1_parameter_diff_gated_state_medium.json",
    "e1_mutation_history_flat.json",
    "e1_mutation_history_state_medium.json",
    "e1_mutation_history_gated_state_medium.json",
    "e1_generation_metrics.json",
    "e1_row_level_eval_sample_train.json",
    "e1_row_level_eval_sample_heldout.json",
    "e1_row_level_eval_sample_ood.json",
    "e1_row_level_eval_sample_counterfactual.json",
    "e1_control_baseline_report.json",
    "e1_flat_baseline_failure_audit.json",
    "e1_state_medium_leakage_audit.json",
    "e1_flat_vs_state_medium_comparison_report.json",
    "e1_state_dynamics_report.json",
    "e1_convergence_stability_report.json",
    "e1_accept_reject_rollback_report.json",
    "e1_no_synthetic_metric_audit.json",
    "e1_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
HASH_ARTIFACTS = (
    "e1_candidate_flat_final.json",
    "e1_candidate_state_medium_final.json",
    "e1_candidate_gated_state_medium_final.json",
    "e1_parameter_diff_flat.json",
    "e1_parameter_diff_state_medium.json",
    "e1_parameter_diff_gated_state_medium.json",
    "e1_generation_metrics.json",
    "e1_control_baseline_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
FORBIDDEN_IMPORTS = {"torch", "jax", "tensorflow"}
FORBIDDEN_CALL_NAMES = {"backward", "fit"}
VALID_DECISIONS = {
    "e1_continuous_state_medium_probe_positive",
    "e1_gated_continuous_state_medium_probe_positive",
    "e1_flat_resistance_remains_preferred",
    "e1_no_state_medium_advantage_detected",
    "e1_task_too_easy_or_leaky",
    "e1_state_medium_instability_detected",
    "e1_invalid_synthetic_metric_regression",
}


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def ast_scan(path: Path) -> list[str]:
    failures = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    failures.append(f"FORBIDDEN_IMPORT:{alias.name}:{path.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS:
                failures.append(f"FORBIDDEN_IMPORT_FROM:{module}:{path.name}")
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            tail = func.rsplit(".", 1)[-1]
            if tail in FORBIDDEN_CALL_NAMES:
                failures.append(f"FORBIDDEN_OPTIMIZER_OR_BACKPROP_CALL:{func}:{path.name}")
    return failures


def compare_replay(primary: Path, replay: Path) -> dict[str, Any]:
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_path = primary / name
        replay_path = replay / name
        comparisons[name] = {
            "primary_exists": primary_path.exists(),
            "replay_exists": replay_path.exists(),
            "primary_hash": file_sha256(primary_path),
            "replay_hash": file_sha256(replay_path),
        }
        comparisons[name]["match"] = (
            comparisons[name]["primary_hash"] is not None
            and comparisons[name]["primary_hash"] == comparisons[name]["replay_hash"]
        )
    return {
        "external_replay_compared": True,
        "external_replay_path": replay.as_posix(),
        "external_replay_passed": all(row["match"] for row in comparisons.values()),
        "external_hash_comparisons": comparisons,
    }


def check_static_files() -> list[str]:
    failures = []
    for rel in (RUNNER, CHECKER, *DOCS):
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{rel}")
    for rel in (RUNNER, CHECKER):
        path = REPO_ROOT / rel
        if path.exists():
            failures.extend(ast_scan(path))
    return failures


def check_decision_logic(decision: dict[str, Any], aggregate: dict[str, Any], controls: dict[str, Any], dynamics: dict[str, Any]) -> list[str]:
    failures = []
    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("UNKNOWN_DECISION")
        return failures
    if decision["decision"] == "e1_task_too_easy_or_leaky" and controls.get("controls_do_not_solve_task") is True:
        failures.append("TASK_TOO_EASY_DECISION_BUT_CONTROLS_PASSED")
    if decision["decision"] == "e1_state_medium_instability_detected" and dynamics.get("finite_state_dynamics_passed") is True:
        failures.append("INSTABILITY_DECISION_BUT_DYNAMICS_FINITE")
    flat_h = aggregate["flat_final_heldout_accuracy"]
    flat_o = aggregate["flat_final_ood_accuracy"]
    flat_c = aggregate["flat_final_counterfactual_accuracy"]
    gated_positive = (
        aggregate["gated_state_medium_final_heldout_accuracy"] >= flat_h + 0.03
        and aggregate["gated_state_medium_final_ood_accuracy"] >= flat_o + 0.03
        and aggregate["gated_state_medium_final_counterfactual_accuracy"] >= flat_c + 0.03
    )
    state_positive = (
        aggregate["state_medium_final_heldout_accuracy"] >= flat_h + 0.03
        and aggregate["state_medium_final_ood_accuracy"] >= flat_o + 0.03
        and aggregate["state_medium_final_counterfactual_accuracy"] >= flat_c + 0.03
    )
    if decision["decision"] == "e1_gated_continuous_state_medium_probe_positive" and not gated_positive:
        failures.append("GATED_POSITIVE_DECISION_WITHOUT_REQUIRED_DELTAS")
    if decision["decision"] == "e1_continuous_state_medium_probe_positive" and not (state_positive or gated_positive):
        failures.append("STATE_POSITIVE_DECISION_WITHOUT_REQUIRED_DELTAS")
    if decision["decision"] in {"e1_flat_resistance_remains_preferred", "e1_no_state_medium_advantage_detected"} and (state_positive or gated_positive):
        failures.append("NON_POSITIVE_DECISION_DESPITE_REQUIRED_DELTAS")
    return failures


def check_artifacts(out: Path, replay: Path | None) -> list[str]:
    failures = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e1_backend_manifest.json")
    task = load_json(out / "e1_task_generation_report.json")
    diffs = {
        "flat": load_json(out / "e1_parameter_diff_flat.json"),
        "state_medium": load_json(out / "e1_parameter_diff_state_medium.json"),
        "gated_state_medium": load_json(out / "e1_parameter_diff_gated_state_medium.json"),
    }
    histories = {
        "flat": load_json(out / "e1_mutation_history_flat.json"),
        "state_medium": load_json(out / "e1_mutation_history_state_medium.json"),
        "gated_state_medium": load_json(out / "e1_mutation_history_gated_state_medium.json"),
    }
    generations = load_json(out / "e1_generation_metrics.json")
    controls = load_json(out / "e1_control_baseline_report.json")
    flat_audit = load_json(out / "e1_flat_baseline_failure_audit.json")
    leakage_audit = load_json(out / "e1_state_medium_leakage_audit.json")
    comparison = load_json(out / "e1_flat_vs_state_medium_comparison_report.json")
    dynamics = load_json(out / "e1_state_dynamics_report.json")
    rollback = load_json(out / "e1_accept_reject_rollback_report.json")
    audit = load_json(out / "e1_no_synthetic_metric_audit.json")
    deterministic = load_json(out / "e1_deterministic_replay_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    progress = read_jsonl(out / "progress.jsonl")

    for key, expected in {
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "synthetic_harness_only": False,
    }.items():
        if manifest.get(key) is not expected:
            failures.append(f"BAD_MANIFEST_FLAG:{key}")
    if manifest.get("numpy_version") is None:
        failures.append("NUMPY_VERSION_NOT_RECORDED")

    if task.get("feature_convention") != "normalized_worse_when_larger":
        failures.append("BAD_FEATURE_CONVENTION")
    for split in ("train", "validation", "heldout", "ood", "counterfactual"):
        if split not in task.get("splits", {}):
            failures.append(f"SPLIT_MISSING:{split}")
    if len(task.get("routes", [])) != 7:
        failures.append("ROUTE_COUNT_BAD")
    if len(task.get("features", [])) != 18:
        failures.append("FEATURE_COUNT_BAD")

    for system, diff in diffs.items():
        if diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_ACTUAL_PARAMETER_DIFF:{system}")
        if diff.get("changed_parameter_count", 0) <= 0:
            failures.append(f"NO_CHANGED_PARAMETERS:{system}")
        if histories[system].get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATIONS:{system}")
        if histories[system].get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{system}")
        if histories[system].get("rollback_count", 0) != histories[system].get("rejected_mutation_count", 0):
            failures.append(f"ROLLBACK_REJECT_MISMATCH:{system}")

    for system in ("flat", "state_medium", "gated_state_medium"):
        if len(generations.get("systems", {}).get(system, [])) != manifest.get("generations"):
            failures.append(f"GENERATION_COUNT_MISMATCH:{system}")

    if controls.get("controls_do_not_solve_task") is not True:
        failures.append("CONTROLS_SOLVE_TASK")
    if controls.get("non_oracle_controls_below_0_90_heldout", 0) < 7:
        failures.append("TOO_FEW_CONTROLS_BELOW_THRESHOLD")
    oracle = controls.get("control_metrics", {}).get("oracle_reference_only", {})
    if oracle.get("reference_only") is not True or oracle.get("used_as_candidate") is not False:
        failures.append("ORACLE_NOT_REFERENCE_ONLY")

    if dynamics.get("finite_state_dynamics_passed") is not True:
        failures.append("FINITE_STATE_DYNAMICS_FAILED")
    if aggregate.get("flat_failure_audit_passed") is not True:
        failures.append("AGGREGATE_FLAT_AUDIT_FAILED")
    if aggregate.get("leakage_audit_passed") is not True:
        failures.append("AGGREGATE_LEAKAGE_AUDIT_FAILED")
    if flat_audit.get("flat_failure_audit_passed") is not True:
        failures.append("FLAT_FAILURE_AUDIT_FAILED")
    if flat_audit.get("score_direction_correct_on_simple_sanity_subset") is not True:
        failures.append("FLAT_SCORE_DIRECTION_SANITY_FAILED")
    if leakage_audit.get("leakage_audit_passed") is not True:
        failures.append("LEAKAGE_AUDIT_FAILED")
    for system in ("state_medium", "gated_state_medium"):
        row = leakage_audit.get("systems", {}).get(system, {})
        if row.get("candidate_order_shuffle_passed") is not True:
            failures.append(f"CANDIDATE_ORDER_SHUFFLE_FAILED:{system}")
        if row.get("feature_permutation_sanity_passed") is not True:
            failures.append(f"FEATURE_PERMUTATION_SANITY_FAILED:{system}")
    for key in ("route_labels_used_for_scoring", "route_names_used_for_scoring", "candidate_order_used_as_feature", "hidden_correct_route_index_used_for_scoring"):
        if leakage_audit.get(key) is not False:
            failures.append(f"LEAKAGE_FLAG_BAD:{key}")
    if rollback.get("rollback_test_executed") is not True or rollback.get("rollback_test_passed") is not True:
        failures.append("ROLLBACK_TEST_FAILED")
    if rollback.get("accepted_mutation_count_total", 0) < 1:
        failures.append("NO_ACCEPTED_MUTATIONS_TOTAL")
    if rollback.get("rejected_mutation_count_total", 0) < 1:
        failures.append("NO_REJECTED_MUTATIONS_TOTAL")

    for key, expected in {
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
    }.items():
        if audit.get(key) is not expected:
            failures.append(f"BAD_AUDIT_FLAG:{key}")
        if aggregate.get(key) is not expected:
            failures.append(f"BAD_AGGREGATE_FLAG:{key}")
        if decision.get(key) is not expected:
            failures.append(f"BAD_DECISION_FLAG:{key}")

    proof_checks = {
        "real_candidate_state_created": True,
        "real_mutation_operator_used": True,
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "before_after_parameter_diff_written": True,
        "actual_parameter_diff_found": True,
        "rollback_test_executed": True,
        "rollback_test_passed": True,
        "deterministic_replay_passed": True,
        "state_medium_parameters_mutated": True,
        "finite_state_dynamics_passed": True,
        "flat_failure_audit_passed": True,
        "leakage_audit_passed": True,
    }
    for key, expected in proof_checks.items():
        if decision.get(key) is not expected:
            failures.append(f"BAD_PROOF_FIELD:{key}")
    if decision.get("accepted_mutation_count_total", 0) < 1:
        failures.append("BAD_PROOF_ACCEPTED_TOTAL")
    if decision.get("rejected_mutation_count_total", 0) < 1:
        failures.append("BAD_PROOF_REJECTED_TOTAL")

    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if aggregate.get("deterministic_replay_passed") is not True:
        failures.append("AGGREGATE_REPLAY_FAILED")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if comparison.get("flat_final_heldout_accuracy") != aggregate.get("flat_final_heldout_accuracy"):
        failures.append("COMPARISON_AGGREGATE_MISMATCH")

    events = {row.get("event") for row in progress}
    for event in ("startup", "generation_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    for sample_name in (
        "e1_row_level_eval_sample_train.json",
        "e1_row_level_eval_sample_heldout.json",
        "e1_row_level_eval_sample_ood.json",
        "e1_row_level_eval_sample_counterfactual.json",
    ):
        sample = load_json(out / sample_name)
        if not sample:
            failures.append(f"EMPTY_SAMPLE:{sample_name}")
        for system in ("flat", "state_medium", "gated_state_medium"):
            if not sample.get(system):
                failures.append(f"EMPTY_SAMPLE_SYSTEM:{sample_name}:{system}")
            elif "predicted_route" not in sample[system][0] or "scores" not in sample[system][0]:
                failures.append(f"BAD_SAMPLE:{sample_name}:{system}")

    failures.extend(check_decision_logic(decision, aggregate, controls, dynamics))

    if replay is not None and replay.exists():
        replay_result = compare_replay(out, replay)
        merged = dict(deterministic)
        merged.update(replay_result)
        write_json(out / "e1_deterministic_replay_report.json", merged)
        if not replay_result["external_replay_passed"]:
            failures.append("EXTERNAL_REPLAY_HASH_MISMATCH")

    return sorted(set(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--replay-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    if args.replay_out is None:
        default_replay = Path(str(out) + "_replay")
        replay = default_replay if default_replay.exists() else None
    else:
        replay = resolve_out(args.replay_out)
    failures = check_static_files() + check_artifacts(out, replay)
    print(json.dumps({"out": out.as_posix(), "replay_out": replay.as_posix() if replay else None, "failure_count": len(failures), "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
