#!/usr/bin/env python3
"""Checker for E2 real-backend state-medium conductivity ordering audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit")
RUNNER = "scripts/probes/run_e2_real_backend_state_medium_conductivity_ordering_audit.py"
CHECKER = "scripts/probes/run_e2_real_backend_state_medium_conductivity_ordering_audit_check.py"
DOCS = (
    "docs/research/E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT_CONTRACT.md",
    "docs/research/E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "e2_online_check_report.md",
    "e2_backend_manifest.json",
    "e2_task_generation_report.json",
    "e2_candidate_flat_initial.json",
    "e2_candidate_flat_final.json",
    "e2_candidate_state_medium_initial.json",
    "e2_candidate_state_medium_final.json",
    "e2_candidate_trajectory_readout_initial.json",
    "e2_candidate_trajectory_readout_final.json",
    "e2_candidate_stability_readout_initial.json",
    "e2_candidate_stability_readout_final.json",
    "e2_parameter_diff_flat.json",
    "e2_parameter_diff_state_medium.json",
    "e2_parameter_diff_trajectory_readout.json",
    "e2_parameter_diff_stability_readout.json",
    "e2_mutation_history_flat.json",
    "e2_mutation_history_state_medium.json",
    "e2_mutation_history_trajectory_readout.json",
    "e2_mutation_history_stability_readout.json",
    "e2_generation_metrics.json",
    "e2_row_level_eval_sample_train.json",
    "e2_row_level_eval_sample_heldout.json",
    "e2_row_level_eval_sample_ood.json",
    "e2_row_level_eval_sample_counterfactual.json",
    "e2_row_level_eval_sample_adversarial.json",
    "e2_conductivity_ordering_report.json",
    "e2_logical_vs_wrong_gap_report.json",
    "e2_attractor_basin_report.json",
    "e2_state_trajectory_report.json",
    "e2_perturbation_recovery_report.json",
    "e2_counterfactual_ordering_report.json",
    "e2_ood_ordering_report.json",
    "e2_control_baseline_report.json",
    "e2_leakage_sentinel_report.json",
    "e2_no_synthetic_metric_audit.json",
    "e2_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
HASH_ARTIFACTS = (
    "e2_candidate_state_medium_final.json",
    "e2_candidate_trajectory_readout_final.json",
    "e2_candidate_stability_readout_final.json",
    "e2_conductivity_ordering_report.json",
    "e2_attractor_basin_report.json",
    "e2_control_baseline_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
SYSTEMS = ("flat", "state_medium", "trajectory_readout", "stability_readout")
STATE_SYSTEMS = ("state_medium", "trajectory_readout", "stability_readout")
ORDERING_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
FORBIDDEN_IMPORTS = {"torch", "jax", "tensorflow"}
FORBIDDEN_CALL_NAMES = {"backward", "fit"}
VALID_DECISIONS = {
    "e2_state_medium_conductivity_ordering_confirmed",
    "e2_temporal_projection_readout_positive",
    "e2_flat_resistance_sufficient",
    "e2_leak_or_task_too_easy_detected",
    "e2_state_medium_instability_detected",
    "e2_no_conductivity_ordering_detected",
    "e2_invalid_synthetic_metric_regression",
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


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


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


def ast_scan(path: Path, check_gate_tokens: bool = False) -> list[str]:
    failures = []
    text = path.read_text(encoding="utf-8")
    if check_gate_tokens:
        lowered = text.lower()
        for needle in ("shortcut_" + "gate", "route_" + "gate", "named_" + "gate"):
            if needle in lowered:
                failures.append(f"FORBIDDEN_GATE_STRING:{needle}:{path.name}")
    tree = ast.parse(text, filename=str(path))
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


def check_static_files() -> list[str]:
    failures = []
    for rel in (RUNNER, CHECKER, *DOCS):
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{rel}")
    for rel in (RUNNER, CHECKER):
        path = REPO_ROOT / rel
        if path.exists():
            failures.extend(ast_scan(path, check_gate_tokens=(rel == RUNNER)))
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
    report_path = out / "e2_deterministic_replay_report.json"
    report = load_json(report_path)
    report.update(compare_replay(out, replay))
    report["deterministic_replay_passed"] = bool(report.get("internal_replay_passed") and report.get("external_replay_passed"))
    write_json(report_path, report)
    return report


def conductivity_split_pass(metrics: dict[str, Any], split: str) -> bool:
    gap_keys = (
        "logical_vs_best_wrong_gap",
        "logical_vs_shortcut_gap",
        "logical_vs_noise_gap",
        "logical_vs_illogical_gap",
        "logical_vs_surface_wrong_gap",
        "logical_vs_contradiction_gap",
    )
    if any(metrics.get(key, -999.0) <= 0.0 for key in gap_keys):
        return False
    if metrics.get("perturbation_recovery_rate", 0.0) < 0.85:
        return False
    if metrics.get("shortcut_attractor_rate", 1.0) > 0.05:
        return False
    if metrics.get("contradiction_attractor_rate", 1.0) > 0.05:
        return False
    if split == "counterfactual" and metrics.get("ordering_stability", 0.0) < 0.90:
        return False
    if split == "ood" and metrics.get("ordering_stability", 0.0) < 0.85:
        return False
    return True


def check_artifacts(out: Path, replay: Path | None) -> list[str]:
    failures = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e2_backend_manifest.json")
    task = load_json(out / "e2_task_generation_report.json")
    conductivity = load_json(out / "e2_conductivity_ordering_report.json")
    controls = load_json(out / "e2_control_baseline_report.json")
    leakage = load_json(out / "e2_leakage_sentinel_report.json")
    audit = load_json(out / "e2_no_synthetic_metric_audit.json")
    deterministic = load_json(out / "e2_deterministic_replay_report.json")
    if replay is not None:
        deterministic = update_external_replay_report(out, replay)
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    generations = load_json(out / "e2_generation_metrics.json")
    progress = read_jsonl(out / "progress.jsonl")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("UNKNOWN_DECISION")
    for key, expected in {
        "candidate_state_created": True,
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "explicit_hand_designed_gate_module_used": False,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "synthetic_harness_only": False,
    }.items():
        if manifest.get(key) is not expected:
            failures.append(f"BAD_MANIFEST_FLAG:{key}")
    if task.get("logical_route_index") == 0:
        failures.append("LOGICAL_ROUTE_IS_INDEX_ZERO")
    if task.get("feature_convention") != "normalized_worse_when_larger":
        failures.append("BAD_FEATURE_CONVENTION")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if split not in task.get("splits", {}):
            failures.append(f"SPLIT_MISSING:{split}")
    if len(task.get("routes", [])) != 8:
        failures.append("ROUTE_COUNT_BAD")
    if len(task.get("features", [])) != 18:
        failures.append("FEATURE_COUNT_BAD")

    for system in SYSTEMS:
        diff = load_json(out / f"e2_parameter_diff_{system}.json")
        history = load_json(out / f"e2_mutation_history_{system}.json")
        if diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_ACTUAL_PARAMETER_DIFF:{system}")
        if history.get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATIONS:{system}")
        if history.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{system}")
        if history.get("rollback_count", 0) != history.get("rejected_mutation_count", 0):
            failures.append(f"ROLLBACK_REJECT_MISMATCH:{system}")
        if len(generations.get("systems", {}).get(system, [])) != manifest.get("generations"):
            failures.append(f"GENERATION_COUNT_MISMATCH:{system}")

    for system, row in conductivity.get("systems", {}).items():
        for split in ORDERING_SPLITS:
            metrics = row.get("split_metrics", {}).get(split, {})
            expected = conductivity_split_pass(metrics, split)
            actual = row.get("split_pass", {}).get(split)
            if actual is not expected:
                failures.append(f"CONDUCTIVITY_SPLIT_PASS_MISMATCH:{system}:{split}")
        if row.get("conductivity_ordering_passed") is not all(row.get("split_pass", {}).values()):
            failures.append(f"CONDUCTIVITY_SYSTEM_PASS_MISMATCH:{system}")

    if controls.get("controls_do_not_solve_task") is not True and decision.get("decision") != "e2_leak_or_task_too_easy_detected":
        failures.append("CONTROLS_SOLVE_TASK_WITHOUT_LEAK_DECISION")
    cm = controls.get("control_metrics", {})
    for name in ("untrained_state_medium_control", "route_index_order_control", "candidate_name_leak_control"):
        if cm.get(name, {}).get("heldout_logical_path_selected_rate", 1.0) >= 0.90:
            failures.append(f"CONTROL_SOLVES_TASK:{name}")
    if cm.get("oracle_reference_only", {}).get("reference_only") is not True or cm.get("oracle_reference_only", {}).get("used_as_candidate") is not False:
        failures.append("ORACLE_NOT_REFERENCE_ONLY")
    if cm.get("shuffled_route_order_control", {}).get("preserved_outcome") is not True:
        failures.append("SHUFFLED_ROUTE_ORDER_CONTROL_FAILED")

    for key, expected in {
        "route_index_leak_detected": False,
        "candidate_name_leak_detected": False,
        "shuffled_route_order_passed": True,
        "leakage_sentinel_passed": True,
    }.items():
        if leakage.get(key) is not expected and decision.get("decision") != "e2_leak_or_task_too_easy_detected":
            failures.append(f"BAD_LEAKAGE_FLAG:{key}")
    for key in ("route_labels_used_for_scoring", "route_names_used_for_scoring", "candidate_order_used_as_feature", "hidden_correct_route_index_used_for_scoring"):
        if leakage.get(key) is not False:
            failures.append(f"LEAKAGE_CONSUMPTION_FLAG_BAD:{key}")

    for key, expected in {
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "explicit_hand_designed_gate_module_used": False,
    }.items():
        if audit.get(key) is not expected:
            failures.append(f"BAD_AUDIT_FLAG:{key}")
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
        "route_index_leak_detected": False,
        "candidate_name_leak_detected": False,
        "shuffled_route_order_passed": True,
    }
    for key, expected in proof_checks.items():
        if decision.get(key) is not expected and not (decision.get("decision") == "e2_leak_or_task_too_easy_detected" and key in {"route_index_leak_detected", "candidate_name_leak_detected", "shuffled_route_order_passed"}):
            failures.append(f"BAD_PROOF_FIELD:{key}")
    if decision.get("accepted_mutation_count_total", 0) < 1:
        failures.append("BAD_PROOF_ACCEPTED_TOTAL")
    if decision.get("rejected_mutation_count_total", 0) < 1:
        failures.append("BAD_PROOF_REJECTED_TOTAL")
    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if replay is not None and deterministic.get("external_replay_passed") is not True:
        failures.append("EXTERNAL_REPLAY_FAILED")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")

    events = {row.get("event") for row in progress}
    for event in ("startup", "generation_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    failures.extend(check_static_files())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--replay-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    replay = resolve_out(args.replay_out) if args.replay_out else None
    failures = check_artifacts(out, replay)
    payload = {
        "failure_count": len(failures),
        "failures": failures,
        "out": out.as_posix(),
        "replay_out": replay.as_posix() if replay else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
