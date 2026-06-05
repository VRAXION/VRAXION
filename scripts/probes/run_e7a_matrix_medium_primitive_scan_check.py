#!/usr/bin/env python3
"""Checker for E7A matrix-medium primitive scan artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a_matrix_medium_primitive_scan.py"
CHECKER = "scripts/probes/run_e7a_matrix_medium_primitive_scan_check.py"
SYSTEMS = (
    "random_classifier",
    "linear_matrix_depth1",
    "linear_matrix_depth3",
    "linear_matrix_depth6",
    "tanh_matrix_depth3",
    "relu_matrix_depth3",
    "c19_fixed_matrix_depth3",
    "c19_rho0_matrix_depth3",
    "c19_c_mut_matrix_depth3",
    "c19_rho_mut_matrix_depth3",
    "c19_c_rho_mut_matrix_depth3",
    "c19_fixed_recurrent_fixed6",
    "c19_c_rho_mut_recurrent_fixed6",
    "c19_fixed_recurrent_halting6",
    "c19_c_rho_mut_recurrent_halting6",
    "c19_c_rho_mut_recurrent_halting_restart6",
)
MUTATION_SYSTEMS = tuple(system for system in SYSTEMS if system != "random_classifier")
BASE_REQUIRED = (
    "e7a_backend_manifest.json",
    "e7a_task_report.json",
    "e7a_primitive_scan_report.json",
    "e7a_collapse_audit.json",
    "e7a_c19_parameter_mode_audit.json",
    "e7a_halting_audit.json",
    "e7a_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a_row_level_eval_sample_heldout.json",
    "e7a_row_level_eval_sample_ood.json",
    "e7a_row_level_eval_sample_counterfactual.json",
    "e7a_row_level_eval_sample_adversarial.json",
)
HASH_ARTIFACTS = (
    "e7a_primitive_scan_report.json",
    "e7a_collapse_audit.json",
    "e7a_c19_parameter_mode_audit.json",
    "e7a_halting_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_METRICS = (
    "heldout_macro_family_accuracy",
    "ood_macro_family_accuracy",
    "counterfactual_macro_family_accuracy",
    "adversarial_macro_family_accuracy",
    "heldout_per_family_accuracy",
    "heldout_mean_steps",
    "heldout_halt_efficiency",
    "heldout_state_mean_abs",
    "heldout_state_std",
    "generalization_gap",
    "parameter_count",
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
        names.append(f"e7a_candidate_{system}_summary.json")
        if system in MUTATION_SYSTEMS:
            names.append(f"e7a_parameter_diff_{system}.json")
            names.append(f"e7a_mutation_history_{system}.json")
    return names


def progress_rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def ast_scan() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    text = runner.read_text(encoding="utf-8")
    for token in ("c19_rho0_matrix_depth3", "c19_c_mut_matrix_depth3", "c19_rho_mut_matrix_depth3", "c19_c_rho_mut_matrix_depth3"):
        if token not in text:
            failures.append(f"C19_MODE_MISSING_IN_RUNNER:{token}")
    if "final_e7_verdict_intentionally_deferred" not in text:
        failures.append("FINAL_VERDICT_DEFER_FLAG_MISSING")
    tree = ast.parse(text)
    if not any(isinstance(node, ast.Name) and node.id == "ProcessPoolExecutor" for node in ast.walk(tree)):
        failures.append("PARALLEL_EXECUTION_SUPPORT_MISSING")
    return failures


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in required_artifacts():
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a_backend_manifest.json")
    task = load_json(out / "e7a_task_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    deterministic = load_json(out / "e7a_deterministic_replay_report.json")
    collapse = load_json(out / "e7a_collapse_audit.json")
    c19 = load_json(out / "e7a_c19_parameter_mode_audit.json")
    halting = load_json(out / "e7a_halting_audit.json")

    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if set(aggregate.get("systems", {})) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEMS_MISMATCH")
    if decision.get("decision") != "e7a_observational_primitive_scan_complete":
        failures.append("BAD_DECISION_FOR_PRIMITIVE_SCAN")
    if decision.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("FINAL_E7_VERDICT_NOT_DEFERRED")
    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")
    for name, row in deterministic.get("hash_comparisons", {}).items():
        if name in HASH_ARTIFACTS and row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")

    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")
    if task.get("families") != ["linear", "xor", "ring", "wave"]:
        failures.append("TASK_FAMILIES_MISMATCH")

    if collapse.get("linear_depth_systems_are_functionally_collapsible") is not True:
        failures.append("LINEAR_COLLAPSE_AUDIT_FAILED")
    for system in ("c19_fixed_matrix_depth3", "c19_rho0_matrix_depth3", "c19_c_mut_matrix_depth3", "c19_rho_mut_matrix_depth3", "c19_c_rho_mut_matrix_depth3"):
        if system not in c19.get("matrix_depth3_systems", {}):
            failures.append(f"C19_MATRIX_MODE_AUDIT_MISSING:{system}")
    for system in ("c19_fixed_recurrent_fixed6", "c19_c_rho_mut_recurrent_fixed6", "c19_fixed_recurrent_halting6", "c19_c_rho_mut_recurrent_halting6", "c19_c_rho_mut_recurrent_halting_restart6"):
        if system not in halting.get("systems", {}):
            failures.append(f"HALTING_AUDIT_MISSING:{system}")

    for system, metrics in aggregate.get("systems", {}).items():
        for key in REQUIRED_METRICS:
            if key not in metrics:
                failures.append(f"SYSTEM_METRIC_MISSING:{system}:{key}")
        if system != "random_classifier" and metrics.get("parameter_count", 0) < 1:
            failures.append(f"PARAMETER_COUNT_MISSING:{system}")

    generations = manifest.get("settings", {}).get("generations")
    for system in MUTATION_SYSTEMS:
        hist = load_json(out / f"e7a_mutation_history_{system}.json")
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{system}")
        if hist.get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATIONS:{system}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{system}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{system}")
        candidate = load_json(out / f"e7a_candidate_{system}_summary.json")
        if len(candidate.get("generation_metrics", [])) != generations:
            failures.append(f"GENERATION_COUNT_MISMATCH:{system}")
        diff = load_json(out / f"e7a_parameter_diff_{system}.json")
        if diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_PARAMETER_DIFF:{system}")

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "generation_complete", "system_start", "system_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if manifest.get("settings", {}).get("execution_mode") == "parallel" and "parallel_systems_start" not in events:
        failures.append("PARALLEL_PROGRESS_MISSING")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e7a_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(SYSTEMS):
            failures.append(f"ROW_SAMPLE_SYSTEMS_MISMATCH:{split}")

    failures.extend(ast_scan())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = check_artifacts(out)
    print(json.dumps({"failure_count": len(failures), "failures": failures, "out": out.as_posix()}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
