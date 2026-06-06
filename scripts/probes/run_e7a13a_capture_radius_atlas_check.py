#!/usr/bin/env python3
"""Checker for E7A13A capture radius atlas artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a13a_capture_radius_atlas.py"
CHECKER = "scripts/probes/run_e7a13a_capture_radius_atlas_check.py"
CORRUPTION_MODES = (
    "random_bit_flip_shell",
    "least_sensitive_bit_flip_shell",
    "most_sensitive_bit_flip_shell",
    "block_corruption_shell",
    "scale_perturbation_shell",
    "bits_plus_scale_corruption_shell",
)
VALID_DECISIONS = {
    "e7a13a_capture_radius_measured",
    "e7a13a_invalid_artifact_detected",
}
VALID_CLASSIFICATIONS = {
    "sharp_capture_boundary",
    "smooth_falloff",
    "ragged_island_basin",
    "no_measurable_repair_basin",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "center_seed_report.json",
    "shell_metrics.json",
    "repair_metrics.json",
    "capture_radius_report.json",
    "falloff_model_report.json",
    "deterministic_replay.json",
    "summary.json",
    "final_summary.md",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
)
REQUIRED_SHELL_FIELDS = (
    "shell_id",
    "requested_distance",
    "corruption_mode",
    "replicate",
    "requested_flip_count",
    "actual_flipped_key_count",
    "raw_hamming_distance_to_center",
    "normalized_hamming_distance_to_center",
    "sensitivity_weighted_bit_distance_to_center",
    "output_distance_to_center_seed",
    "output_distance_to_teacher",
    "seed_eval_before_repair",
    "eval_gap_to_center_seed",
    "eval_gap_to_qat_reference",
    "seed_solve_passed_before_repair",
    "corrupted_hash",
)
REQUIRED_REPAIR_FIELDS = (
    "shell_id",
    "budget_multiplier",
    "repair_generations",
    "seed_eval_before_repair",
    "seed_eval_after_repair",
    "repair_gain",
    "accepted_mutations",
    "rejected_mutations",
    "rollback_count",
    "mutation_attempts",
    "accepted_by_operator",
    "rejected_by_operator",
    "budget_to_best_eval",
    "solve_passed",
    "recovered_to_within_epsilon",
    "final_eval",
    "final_hash",
    "row_level_samples",
    "history_tail",
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


def progress_rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def static_scan() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    text = runner.read_text(encoding="utf-8")
    for token in (
        "E7A13A_CAPTURE_RADIUS_ATLAS",
        "random_bit_flip_shell",
        "least_sensitive_bit_flip_shell",
        "most_sensitive_bit_flip_shell",
        "block_corruption_shell",
        "scale_perturbation_shell",
        "bits_plus_scale_corruption_shell",
        "sensitivity_weighted_bit_distance_to_center",
        "mutation_history_snapshots",
        "current_best_candidate_summary",
        "partial_aggregate_snapshot",
        "deterministic_replay",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_function_names = {"corrupt_seed", "distance_metrics", "score_candidate", "mutate_for_repair", "repair_worker"}
    forbidden_calls = {"backward", "step", "zero_grad"}
    forbidden_optimizer_attrs = {"AdamW", "SGD", "RMSprop", "Optimizer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_function_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_calls:
                        failures.append(f"MUTATION_REPAIR_BACKPROP_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in forbidden_optimizer_attrs:
                    failures.append(f"MUTATION_REPAIR_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
    return failures


def check_shells(failures: list[str], manifest: dict[str, Any], shells: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = manifest.get("settings", {})
    distances = settings.get("distances", [])
    shell_replicates = int(settings.get("shell_replicates", 0))
    expected_count = len(CORRUPTION_MODES) * len(distances) * shell_replicates
    rows = shells.get("rows", [])
    if shells.get("shell_count") != len(rows):
        failures.append("SHELL_COUNT_FIELD_MISMATCH")
    if len(rows) != expected_count:
        failures.append(f"SHELL_ROW_COUNT_MISMATCH:expected={expected_count}:actual={len(rows)}")
    if tuple(shells.get("required_corruption_modes", [])) != CORRUPTION_MODES:
        failures.append("SHELL_CORRUPTION_MODE_SET_MISMATCH")
    seen = set()
    shell_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        shell_id = row.get("shell_id")
        if not shell_id:
            failures.append("SHELL_ID_MISSING")
            continue
        if shell_id in seen:
            failures.append(f"DUPLICATE_SHELL_ID:{shell_id}")
        seen.add(shell_id)
        shell_by_id[shell_id] = row
        for key in REQUIRED_SHELL_FIELDS:
            if key not in row:
                failures.append(f"SHELL_FIELD_MISSING:{shell_id}:{key}")
        if row.get("corruption_mode") not in CORRUPTION_MODES:
            failures.append(f"SHELL_MODE_UNKNOWN:{shell_id}:{row.get('corruption_mode')}")
        if row.get("requested_distance") not in distances:
            failures.append(f"SHELL_DISTANCE_UNKNOWN:{shell_id}:{row.get('requested_distance')}")
        for key in ("normalized_hamming_distance_to_center", "sensitivity_weighted_bit_distance_to_center"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                failures.append(f"SHELL_DISTANCE_FIELD_INVALID:{shell_id}:{key}:{value}")
        if row.get("corruption_mode") != "scale_perturbation_shell" and row.get("requested_distance", 0) > 0 and row.get("raw_hamming_distance_to_center", 0) < 1:
            failures.append(f"SHELL_NO_BIT_CHANGE:{shell_id}")
    return shell_by_id


def check_repairs(failures: list[str], manifest: dict[str, Any], shell_by_id: dict[str, dict[str, Any]], repairs: dict[str, Any]) -> None:
    settings = manifest.get("settings", {})
    budget_multipliers = settings.get("budget_multipliers", [])
    expected_count = len(shell_by_id) * len(budget_multipliers)
    rows = repairs.get("rows", [])
    if repairs.get("repair_run_count") != len(rows):
        failures.append("REPAIR_COUNT_FIELD_MISMATCH")
    if len(rows) != expected_count:
        failures.append(f"REPAIR_ROW_COUNT_MISMATCH:expected={expected_count}:actual={len(rows)}")
    if repairs.get("budget_multipliers") != budget_multipliers:
        failures.append("REPAIR_BUDGET_MULTIPLIERS_MISMATCH")
    expected_pairs = {(shell_id, int(budget)) for shell_id in shell_by_id for budget in budget_multipliers}
    seen_pairs = set()
    for row in rows:
        shell_id = row.get("shell_id")
        budget = row.get("budget_multiplier")
        pair = (shell_id, int(budget)) if shell_id is not None and budget is not None else None
        if pair in seen_pairs:
            failures.append(f"DUPLICATE_REPAIR_PAIR:{shell_id}:{budget}")
        if pair:
            seen_pairs.add(pair)
        for key in REQUIRED_REPAIR_FIELDS:
            if key not in row:
                failures.append(f"REPAIR_FIELD_MISSING:{shell_id}:{budget}:{key}")
        if shell_id not in shell_by_id:
            failures.append(f"REPAIR_UNKNOWN_SHELL:{shell_id}")
        if budget not in budget_multipliers:
            failures.append(f"REPAIR_UNKNOWN_BUDGET:{shell_id}:{budget}")
        if row.get("mutation_attempts", 0) < 1:
            failures.append(f"REPAIR_NO_MUTATION_ATTEMPTS:{shell_id}:{budget}")
        if row.get("rejected_mutations", 0) < 1:
            failures.append(f"REPAIR_NO_REJECTED_MUTATIONS:{shell_id}:{budget}")
        if row.get("rejected_mutations") != row.get("rollback_count"):
            failures.append(f"REPAIR_ROLLBACK_MISMATCH:{shell_id}:{budget}")
        if shell_id in shell_by_id:
            before = shell_by_id[shell_id].get("seed_eval_before_repair")
            if row.get("seed_eval_before_repair") != before:
                failures.append(f"REPAIR_BEFORE_EVAL_MISMATCH:{shell_id}:{budget}")
        if row.get("seed_eval_after_repair") != row.get("final_eval"):
            failures.append(f"REPAIR_AFTER_FINAL_MISMATCH:{shell_id}:{budget}")
        if not isinstance(row.get("row_level_samples"), dict) or not row.get("row_level_samples"):
            failures.append(f"REPAIR_ROW_LEVEL_SAMPLES_MISSING:{shell_id}:{budget}")
    missing_pairs = expected_pairs - seen_pairs
    if missing_pairs:
        failures.append(f"REPAIR_PAIR_SET_MISSING:{len(missing_pairs)}")
    skipped = repairs.get("skipped_budgets", [])
    for item in skipped:
        if "budget_multiplier" not in item or not item.get("reason"):
            failures.append("SKIPPED_BUDGET_REASON_MISSING")


def check_reports(
    failures: list[str],
    out: Path,
    manifest: dict[str, Any],
    center: dict[str, Any],
    capture: dict[str, Any],
    falloff: dict[str, Any],
    replay: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    for key in ("center_source", "center_hash", "center_eval_accuracy", "center_solve_passed", "scale_storage_mode", "parameter_count", "bit_cost"):
        if key not in center:
            failures.append(f"CENTER_FIELD_MISSING:{key}")
    if not center.get("center_hash"):
        failures.append("CENTER_HASH_EMPTY")
    if center.get("scale_storage_mode") != "block_per_tensor":
        failures.append("CENTER_SCALE_MODE_NOT_BLOCK")
    if not isinstance(center.get("bit_cost"), dict) or "total_bit_cost" not in center.get("bit_cost", {}):
        failures.append("CENTER_BIT_COST_BAD")

    classification = capture.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
        failures.append("CAPTURE_CLASSIFICATION_INVALID")
    if classification != falloff.get("classification"):
        failures.append("FALLOFF_CLASSIFICATION_MISMATCH")
    if not capture.get("buckets"):
        failures.append("CAPTURE_BUCKETS_EMPTY")
    if not capture.get("by_distance"):
        failures.append("CAPTURE_BY_DISTANCE_EMPTY")
    for row in capture.get("by_distance", []):
        for key in ("distance", "mean_repair_gain", "mean_final_eval", "recovery_rate_to_center", "solve_rate", "n_buckets"):
            if key not in row:
                failures.append(f"CAPTURE_DISTANCE_FIELD_MISSING:{key}")
    if falloff.get("best_model") not in {"flat_null", "exponential_falloff", "power_law_falloff", "logistic_falloff"}:
        failures.append("FALLOFF_BEST_MODEL_INVALID")
    for model in ("flat_null", "exponential_falloff", "power_law_falloff", "logistic_falloff"):
        if model not in falloff.get("models", {}):
            failures.append(f"FALLOFF_MODEL_MISSING:{model}")

    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for name, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")
    if summary.get("decision") not in VALID_DECISIONS:
        failures.append("SUMMARY_DECISION_INVALID")
    if summary.get("classification") != classification:
        failures.append("SUMMARY_CLASSIFICATION_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True:
        failures.append("SUMMARY_REPLAY_FLAG_BAD")
    if summary.get("repair_run_count", 0) < 1 or summary.get("shell_count", 0) < 1:
        failures.append("SUMMARY_COUNTS_BAD")
    if manifest.get("broad_claims_intentionally_deferred") is not True or summary.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_BAD")

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "center_seed_built", "repair_jobs_submitted", "repair_job_complete", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    for dirname in ("partial_status", "mutation_history_snapshots", "current_best_candidate_summary"):
        folder = out / dirname
        if not folder.exists() or not any(folder.rglob("*.json")):
            failures.append(f"PARTIAL_OUTPUT_DIR_EMPTY:{dirname}")
    partial = load_json(out / "partial_aggregate_snapshot.json")
    if partial.get("completed_repair_jobs") != summary.get("repair_run_count"):
        failures.append("PARTIAL_AGGREGATE_COMPLETED_COUNT_MISMATCH")

    report_text = (out / "final_summary.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning", "dark matter"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "backend_manifest.json")
    center = load_json(out / "center_seed_report.json")
    shells = load_json(out / "shell_metrics.json")
    repairs = load_json(out / "repair_metrics.json")
    capture = load_json(out / "capture_radius_report.json")
    falloff = load_json(out / "falloff_model_report.json")
    replay = load_json(out / "deterministic_replay.json")
    summary = load_json(out / "summary.json")

    if manifest.get("milestone") != "E7A13A_CAPTURE_RADIUS_ATLAS":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("corruption_modes", [])) != CORRUPTION_MODES:
        failures.append("MANIFEST_CORRUPTION_MODES_MISMATCH")
    settings = manifest.get("settings", {})
    if not settings.get("distances"):
        failures.append("SETTINGS_DISTANCES_EMPTY")
    if not settings.get("budget_multipliers"):
        failures.append("SETTINGS_BUDGETS_EMPTY")
    if settings.get("base_repair_generations", 0) < 1 or settings.get("mutation_population", 0) < 2:
        failures.append("MUTATION_BUDGET_TOO_SMALL")

    shell_by_id = check_shells(failures, manifest, shells)
    check_repairs(failures, manifest, shell_by_id, repairs)
    check_reports(failures, out, manifest, center, capture, falloff, replay, summary)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a13a_capture_radius_atlas")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a13a_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    (out / "checker_report.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "checker_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
