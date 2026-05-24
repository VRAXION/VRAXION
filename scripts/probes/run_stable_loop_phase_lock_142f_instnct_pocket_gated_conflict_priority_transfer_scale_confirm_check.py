#!/usr/bin/env python3
"""Checker for 142F conflict/priority transfer scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_142f_instnct_pocket_gated_conflict_priority_transfer_scale_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_142f_instnct_pocket_gated_conflict_priority_transfer_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_142f_instnct_pocket_gated_conflict_priority_transfer_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
EXPECTED_FAMILIES = {
    "TWO_FIELD_PRIORITY_RULE",
    "DUAL_POCKET_CONFLICT",
    "TABLE_RULE_PRIORITY_OVERRIDE",
    "VISIBLE_VALUE_LOSES_TO_PRIORITY",
    "NOISY_DISTRACTOR_PRIORITY_TRAP",
    "SAME_TEMPLATE_DIFFERENT_PRIORITY_CONTRAST",
}
REQUIRED_CONTROLS = {
    "A_ONLY_CONTROL",
    "B_ONLY_CONTROL",
    "ALWAYS_A_SHORTCUT_CONTROL",
    "ALWAYS_B_SHORTCUT_CONTROL",
    "TABLE_DEFAULT_CONTROL",
    "RULE_DEFAULT_CONTROL",
    "VISIBLE_VALUE_CONTROL",
    "NOISY_DISTRACTOR_CONTROL",
    "CLOSED_POCKET_ABLATION_CONTROL",
    "PRIORITY_DEFAULT_SHORTCUT_CONTROL",
    "SAME_TEMPLATE_PRIORITY_INVERSION_CONTROL",
    "PREFIX_ONLY_CONTROL",
}
REQUIRED_ARTIFACTS = [
    "queue.json", "progress.jsonl", "upstream_142a_manifest.json",
    "eval_config.json", "helper_provenance_verification.json", "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json", "forbidden_input_rejection_report.json",
    "conflict_priority_eval_manifest.json", "priority_rule_manifest.json", "conflict_pair_manifest.json",
    "explicit_marker_audit.json", "eval_rows.jsonl", "mutation_candidate_results.jsonl",
    "mutation_search_trace.jsonl", "selection_report.json", "fitness_landscape.json",
    "raw_generation_trace.jsonl", "raw_generation_results.jsonl", "pocket_trace.jsonl",
    "pocket_ablation_results.jsonl", "scoring_results.jsonl", "contrast_group_results.jsonl",
    "priority_inversion_pairs.jsonl", "control_results.jsonl", "control_arm_report.json",
    "priority_control_report.json", "generated_before_scoring_report.json", "freshness_leakage_audit.json",
    "conflict_priority_transfer_metrics.json", "aggregate_metrics.json", "helper_request_audit.json",
    "canonical_metric_alias_report.json", "per_seed_gate_report.json", "per_family_gate_report.json",
    "per_seed_metrics.json", "per_family_metrics.json", "per_winner_metrics.json",
    "winner_distribution_report.json", "per_winner_gate_report.json",
    "priority_inversion_report.json", "priority_inversion_pair_report.json",
    "same_template_opposite_winner_report.json", "shortcut_report.json",
    "wrong_priority_field_report.json", "priority_default_shortcut_report.json",
    "arm_comparison.json", "determinism_replay_report.json", "decision.json", "summary.json", "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored", "raw_assistant_capability_restored", "structured_tool_capability_restored",
    "gpt_like_readiness_claimed", "open_domain_assistant_readiness_claimed", "production_chat_claimed",
    "public_api_claimed", "deployment_readiness_claimed", "safety_alignment_claimed",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_FLAG_NOT_FALSE:{key}")


def ast_scan(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
            failures.append(f"OLD_RUNNER_IMPORT:{path.name}")
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            failures.append(f"TORCH_IMPORT_NOT_ALLOWED:{path.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "backward", "step"}:
                failures.append(f"TRAINING_CALL_NOT_ALLOWED:{path.name}:{name}")
            if path.name == Path(CHECKER).name and name == "raw_generate":
                failures.append(f"CHECKER_RAW_GENERATE_NOT_ALLOWED:{path.name}")
    return failures


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_upstreams(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_142a_manifest.json")
    if upstream.get("decision") != "instnct_pocket_gated_conflict_priority_transfer_probe_positive":
        failures.append(f"BAD_142A_DECISION:{upstream.get('decision')}")
    if upstream.get("next") != "142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM":
        failures.append(f"BAD_142A_NEXT:{upstream.get('next')}")
    exact = {
        "main_final_answer_accuracy": 1.0,
        "priority_rule_accuracy": 1.0,
        "conflict_resolution_accuracy": 1.0,
        "priority_inversion_accuracy": 1.0,
        "same_template_opposite_winner_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "pocket_ablation_delta_final_answer_accuracy": 1.0,
    }
    for key, expected in exact.items():
        if upstream.get(key) != expected:
            failures.append(f"BAD_142A_METRIC:{key}:{upstream.get(key)}")
    audit = upstream.get("helper_request_audit", {})
    if audit.get("all_requests_allowed_keys_only") is not True or audit.get("forbidden_keys_present_count") != 0:
        failures.append(f"BAD_142A_HELPER_AUDIT:{audit}")
    if upstream.get("per_seed_gate_passed") is not True or upstream.get("per_family_gate_passed") is not True:
        failures.append("BAD_142A_HARDENED_GATE_REPORTS")


def require_config(root: Path, failures: list[str]) -> None:
    config = load_json(root / "eval_config.json")
    for key, expected in {
        "train_allowed": False,
        "training_performed": False,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
    }.items():
        if config.get(key) is not expected:
            failures.append(f"BOUNDARY_NOT_{expected}:{key}:{config.get(key)}")
    if config.get("helper_generation_allowed") is not True or config.get("shared_helper_only") is not True:
        failures.append("HELPER_ONLY_GENERATION_BOUNDARY_MISSING")
    require_false_flags(config, failures)


def require_infrastructure(root: Path, failures: list[str]) -> None:
    provenance = load_json(root / "helper_provenance_verification.json")
    ast_report = load_json(root / "ast_shortcut_scan_report.json")
    canary = load_json(root / "expected_output_canary_report.json")
    forbidden = load_json(root / "forbidden_input_rejection_report.json")
    generated = load_json(root / "generated_before_scoring_report.json")
    leakage = load_json(root / "freshness_leakage_audit.json")
    determinism = load_json(root / "determinism_replay_report.json")
    if provenance.get("adapter_backend_name") != "repo_local_instnct_mutation_graph":
        failures.append(f"BAD_HELPER_BACKEND:{provenance.get('adapter_backend_name')}")
    if provenance.get("strict_pocket_gated_symbols_present") is not True:
        failures.append("STRICT_POCKET_SYMBOLS_MISSING")
    if ast_report.get("passed") is not True:
        failures.append(f"AST_SCAN_FAILED:{ast_report.get('failures')}")
    if canary.get("passed") is not True or forbidden.get("passed") is not True:
        failures.append("FORBIDDEN_INPUT_CANARY_FAILED")
    if generated.get("generated_text_produced_before_scoring") is not True:
        failures.append("GENERATED_AFTER_SCORING_OR_MISSING")
    if generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("BAD_HELPER_REQUEST_KEYS_IN_GENERATION_REPORT")
    if generated.get("expected_or_scorer_metadata_in_helper_requests") is not False:
        failures.append("HELPER_REQUEST_METADATA_LEAK")
    if leakage.get("leakage_rejected") is not True:
        failures.append("LEAKAGE_NOT_REJECTED")
    if determinism.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_FAILED")


def require_manifests(root: Path, failures: list[str]) -> None:
    manifest = load_json(root / "conflict_priority_eval_manifest.json")
    priority = load_json(root / "priority_rule_manifest.json")
    pairs = load_json(root / "conflict_pair_manifest.json")
    winner_distribution = load_json(root / "winner_distribution_report.json")
    marker = load_json(root / "explicit_marker_audit.json")
    if manifest.get("row_count", 0) < 2880:
        failures.append(f"ROW_COUNT_TOO_LOW:{manifest.get('row_count')}")
    if manifest.get("family_count") != 6 or set(manifest.get("families", [])) != EXPECTED_FAMILIES:
        failures.append(f"BAD_FAMILY_SET:{manifest.get('families')}")
    if manifest.get("scaffold_variant_count", 0) < 72:
        failures.append(f"SCAFFOLD_VARIANT_COUNT_TOO_LOW:{manifest.get('scaffold_variant_count')}")
    if priority.get("has_a_b_table_rule_winners") is not True:
        failures.append(f"PRIORITY_WINNER_COVERAGE_MISSING:{priority.get('winner_source_counts')}")
    for key in ["a_wins_rate", "b_wins_rate", "table_wins_rate", "rule_wins_rate"]:
        if winner_distribution.get(key, 0.0) < 0.15:
            failures.append(f"WINNER_RATE_TOO_LOW:{key}:{winner_distribution.get(key)}")
    if winner_distribution.get("winner_distribution_balanced") is not True:
        failures.append("WINNER_DISTRIBUTION_NOT_BALANCED")
    if pairs.get("priority_inversion_pair_rate", 0.0) < 0.50:
        failures.append(f"PRIORITY_INVERSION_PAIR_RATE_TOO_LOW:{pairs.get('priority_inversion_pair_rate')}")
    if pairs.get("priority_inversion_pair_count", 0) < 1000:
        failures.append(f"PRIORITY_INVERSION_PAIR_COUNT_TOO_LOW:{pairs.get('priority_inversion_pair_count')}")
    if pairs.get("same_template_opposite_winner_pairs", 0) < 1000:
        failures.append(f"SAME_TEMPLATE_OPPOSITE_WINNER_PAIRS_TOO_LOW:{pairs.get('same_template_opposite_winner_pairs')}")
    if marker.get("direct_pocket_value_marker_rate") != 0.0:
        failures.append(f"DIRECT_POCKET_VALUE_MARKER_PRESENT:{marker.get('direct_pocket_value_marker_rate')}")
    if marker.get("visible_wrong_value_row_rate") != 1.0 or marker.get("noisy_distractor_row_rate") != 1.0:
        failures.append("VISIBLE_OR_NOISY_ROWS_NOT_FULL_COVERAGE")


def require_generation_trace(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "helper_request_audit.json")
    if audit.get("all_requests_allowed_keys_only") is not True:
        failures.append("HELPER_REQUEST_AUDIT_KEYS_NOT_ALLOWED_ONLY")
    if audit.get("no_forbidden_keys_in_accepted_generation_requests") is not True:
        failures.append("FORBIDDEN_KEYS_IN_ACCEPTED_HELPER_REQUESTS")
    if audit.get("forbidden_keys_present_count") != 0:
        failures.append(f"FORBIDDEN_KEY_COUNT_NONZERO:{audit.get('forbidden_keys_present_count')}")
    if audit.get("raw_generate_allowed_in_runner") is not True or audit.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_RUNNER_CHECKER_BOUNDARY_BAD")
    if audit.get("shared_helper_only") is not True:
        failures.append("HELPER_REQUEST_AUDIT_NOT_SHARED_HELPER_ONLY")
    if set(audit.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_AUDIT_ALLOWED_KEYS:{audit.get('allowed_helper_keys')}")
    rows = read_jsonl(root / "raw_generation_trace.jsonl")
    if not rows:
        failures.append("RAW_GENERATION_TRACE_EMPTY")
    for index, row in enumerate(rows[:50]):
        request = row.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append(f"BAD_HELPER_REQUEST_KEYS:{index}:{sorted(request)}")
        if row.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append(f"BAD_RAW_BACKEND:{index}:{row.get('backend_name')}")
        if row.get("generated_before_scoring") is not True:
            failures.append(f"GENERATED_BEFORE_SCORING_FALSE:{index}")


def require_controls(root: Path, failures: list[str]) -> None:
    report = load_json(root / "priority_control_report.json")
    controls = read_jsonl(root / "control_results.jsonl")
    present = {row.get("control") for row in controls}
    if report.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_ALL_FAIL")
    missing = sorted(REQUIRED_CONTROLS - present)
    if missing:
        failures.append(f"MISSING_CONTROLS:{missing}")
    for row in controls:
        if row.get("control") in REQUIRED_CONTROLS and row.get("control_failed") is not True:
            failures.append(f"REQUIRED_CONTROL_PASSED:{row.get('control')}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "arm_comparison.json")
    aggregate = load_json(root / "aggregate_metrics.json")
    alias_report = load_json(root / "canonical_metric_alias_report.json")
    seed_gate = load_json(root / "per_seed_gate_report.json")
    family_gate = load_json(root / "per_family_gate_report.json")
    winner_gate = load_json(root / "per_winner_gate_report.json")
    winner_distribution = load_json(root / "winner_distribution_report.json")
    shortcuts = load_json(root / "shortcut_report.json")
    inversion = load_json(root / "priority_inversion_report.json")
    inversion_pairs = load_json(root / "priority_inversion_pair_report.json")
    opposite = load_json(root / "same_template_opposite_winner_report.json")
    wrong = load_json(root / "wrong_priority_field_report.json")
    default = load_json(root / "priority_default_shortcut_report.json")
    equals_zero = [
        "wrong_priority_field_rate",
        "priority_default_shortcut_rate",
        "always_a_shortcut_rate",
        "always_b_shortcut_rate",
        "table_default_shortcut_rate",
        "rule_default_shortcut_rate",
        "single_field_shortcut_rate",
        "visible_bypass_violation_rate",
        "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
    ]
    for key in equals_zero:
        if metrics.get(key) != 0.0:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    floors = {
        "main_final_answer_accuracy": 0.90,
        "priority_rule_accuracy": 0.90,
        "conflict_resolution_accuracy": 0.90,
        "priority_inversion_accuracy": 0.85,
        "same_template_opposite_winner_accuracy": 0.85,
        "main_pocket_writeback_rate": 0.95,
        "main_contrast_group_accuracy": 0.90,
        "pocket_ablation_delta_final_answer_accuracy": 0.85,
    }
    for key, floor in floors.items():
        if metrics.get(key, 0.0) < floor:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}<{floor}")
    if metrics.get("ablation_final_answer_accuracy", 1.0) > 0.05:
        failures.append(f"ABLATION_ACCURACY_TOO_HIGH:{metrics.get('ablation_final_answer_accuracy')}")
    if metrics.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_METRIC_NOT_TRUE")
    for key in ["priority_rule_accuracy", "conflict_resolution_accuracy", "wrong_priority_field_rate", "priority_default_shortcut_rate", "priority_inversion_accuracy", "same_template_opposite_winner_accuracy", "always_a_shortcut_rate", "always_b_shortcut_rate", "table_default_shortcut_rate", "rule_default_shortcut_rate", "priority_inversion_pair_count"]:
        if key not in aggregate or key not in alias_report.get("canonical_metrics", {}):
            failures.append(f"CANONICAL_PRIORITY_METRIC_MISSING:{key}")
    gates = aggregate.get("infrastructure_gates", {})
    for key in ["expected_output_canary_passed", "ast_scan_passed", "leakage_rejected", "controls_failed", "generated_text_before_scoring", "helper_request_keys_allowed_only", "no_expected_scorer_oracle_metadata", "deterministic_replay_passed"]:
        if gates.get(key) is not True:
            failures.append(f"INFRASTRUCTURE_GATE_FAILED:{key}")
    if seed_gate.get("passed") is not True:
        failures.append("PER_SEED_GATE_REPORT_FAILED")
    if family_gate.get("passed") is not True:
        failures.append("PER_FAMILY_GATE_REPORT_FAILED")
    if winner_gate.get("passed") is not True:
        failures.append("PER_WINNER_GATE_REPORT_FAILED")
    if winner_distribution.get("passed") is not True:
        failures.append("WINNER_DISTRIBUTION_REPORT_FAILED")
    if shortcuts.get("passed") is not True:
        failures.append("SHORTCUT_REPORT_FAILED")
    if inversion.get("passed") is not True or inversion.get("priority_inversion_accuracy", 0.0) < 0.85:
        failures.append("PRIORITY_INVERSION_REPORT_FAILED")
    if inversion_pairs.get("priority_inversion_pair_count", 0) < 1000:
        failures.append("PRIORITY_INVERSION_PAIR_REPORT_COUNT_TOO_LOW")
    if opposite.get("same_template_opposite_winner_pairs", 0) < 1000:
        failures.append("SAME_TEMPLATE_OPPOSITE_WINNER_REPORT_COUNT_TOO_LOW")
    if wrong.get("wrong_priority_field_selected") is not False:
        failures.append("WRONG_PRIORITY_FIELD_SELECTED")
    if default.get("priority_default_shortcut_detected") is not False:
        failures.append("PRIORITY_DEFAULT_SHORTCUT_DETECTED")


def require_selection(root: Path, failures: list[str]) -> None:
    selection = load_json(root / "selection_report.json")
    if selection.get("selected_candidate") != "open_priority_resolved_final_all_markers":
        failures.append(f"BAD_SELECTED_CANDIDATE:{selection.get('selected_candidate')}")
    if selection.get("gradient_used") is not False:
        failures.append("GRADIENT_USED_NOT_FALSE")
    if selection.get("selected_by_fitness") is not True:
        failures.append("SELECTION_NOT_BY_FITNESS")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    if decision.get("decision") != "instnct_pocket_gated_conflict_priority_transfer_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if decision.get("value_grounding_claimed") is not False or decision.get("architecture_superiority_claimed") is not False:
        failures.append("BROAD_OR_ARCHITECTURE_CLAIMED")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)


def require_report(root: Path, failures: list[str]) -> None:
    report = (root / "report.md").read_text(encoding="utf-8")
    for phrase in [
        "constrained helper/backend conflict-priority final selection",
        "not open-ended reasoning",
        "not general composition",
        "not GPT-like readiness",
        "not open-domain reasoning",
        "not broad assistant capability",
        "not production/public",
        "not architecture superiority",
    ]:
        if phrase not in report:
            failures.append(f"REPORT_MISSING_BOUNDARY:{phrase}")


def require_docs(failures: list[str]) -> None:
    for doc in DOCS:
        path = REPO_ROOT / doc
        text = path.read_text(encoding="utf-8")
        for phrase in [
            "142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM",
            "priority inversion",
            "winner distribution",
            "helper_request_audit.json",
            "not GPT-like readiness",
            "not broad assistant capability",
        ]:
            if phrase not in text:
                failures.append(f"DOC_MISSING_PHRASE:{doc}:{phrase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 142F conflict/priority transfer scale confirm artifacts.")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()

    failures: list[str] = []
    require_changed_files(failures)
    for path in [REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]:
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{path}")
        else:
            failures.extend(ast_scan(path))
    for doc in DOCS:
        if not (REPO_ROOT / doc).exists():
            failures.append(f"MISSING_TRACKED_FILE:{doc}")
    if all((REPO_ROOT / doc).exists() for doc in DOCS):
        require_docs(failures)
    require_artifacts(root, failures)
    if not failures:
        require_upstreams(root, failures)
        require_config(root, failures)
        require_infrastructure(root, failures)
        require_manifests(root, failures)
        require_generation_trace(root, failures)
        require_controls(root, failures)
        require_metrics(root, failures)
        require_selection(root, failures)
        require_decision(root, failures)
        require_report(root, failures)
    if failures:
        print("142F CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("142F CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
