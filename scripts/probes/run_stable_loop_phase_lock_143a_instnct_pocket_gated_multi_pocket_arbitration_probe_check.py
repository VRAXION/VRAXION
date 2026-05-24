#!/usr/bin/env python3
"""Checker for 143A multi-pocket arbitration probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143a_instnct_pocket_gated_multi_pocket_arbitration_probe/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143a_instnct_pocket_gated_multi_pocket_arbitration_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143a_instnct_pocket_gated_multi_pocket_arbitration_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
EXPECTED_FAMILIES = {
    "THREE_POCKET_RULE_SELECT",
    "QUORUM_TWO_OF_THREE",
    "RECENCY_OVERRIDE",
    "TIE_BREAK_ARBITRATION",
    "STALE_POCKET_LOSES",
    "SAME_TEMPLATE_ARBITRATION_INVERSION",
    "RULE_HIERARCHY_CONFLICT",
}
REQUIRED_CONTROLS = {
    "FIRST_POCKET_CONTROL",
    "LAST_POCKET_CONTROL",
    "DEFAULT_POCKET_CONTROL",
    "STALE_POCKET_CONTROL",
    "VISIBLE_BYPASS_CONTROL",
    "NOISY_DISTRACTOR_CONTROL",
    "CLOSED_POCKET_ABLATION_CONTROL",
    "QUORUM_WRONG_CONTROL",
    "RECENCY_WRONG_CONTROL",
    "TIE_BREAK_WRONG_CONTROL",
    "RESOLVED_FINAL_MARKER_ECHO_CONTROL",
    "POCKET_LABEL_PERMUTATION_CONTROL",
    "RULE_HIERARCHY_CONFLICT_CONTROL",
    "SAME_VALUES_DIFFERENT_RULE_CONTROL",
    "SAME_RULE_DIFFERENT_VALUES_CONTROL",
    "PREFIX_ONLY_CONTROL",
}
REQUIRED_ARTIFACTS = [
    "queue.json", "progress.jsonl", "upstream_142z_manifest.json", "upstream_142f_manifest.json",
    "eval_config.json", "helper_provenance_verification.json", "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json", "forbidden_input_rejection_report.json",
    "multi_pocket_eval_manifest.json", "multi_pocket_manifest.json", "arbitration_rule_manifest.json",
    "explicit_marker_audit.json", "eval_rows.jsonl", "mutation_candidate_results.jsonl",
    "mutation_search_trace.jsonl", "selection_report.json", "fitness_landscape.json",
    "raw_generation_trace.jsonl", "raw_generation_results.jsonl", "pocket_trace.jsonl",
    "pocket_ablation_results.jsonl", "scoring_results.jsonl", "contrast_group_results.jsonl",
    "arbitration_inversion_pairs.jsonl", "control_results.jsonl", "control_arm_report.json",
    "multi_pocket_control_report.json", "generated_before_scoring_report.json", "freshness_leakage_audit.json",
    "multi_pocket_arbitration_metrics.json", "aggregate_metrics.json", "helper_request_audit.json",
    "canonical_metric_alias_report.json", "per_seed_gate_report.json", "per_family_gate_report.json",
    "per_seed_metrics.json", "per_family_metrics.json", "per_pocket_metrics.json",
    "pocket_distribution_report.json", "winner_distribution_report.json", "per_pocket_gate_report.json",
    "priority_inversion_report.json", "priority_inversion_pair_report.json",
    "same_template_opposite_winner_report.json", "shortcut_report.json",
    "resolved_final_marker_echo_report.json", "pocket_label_permutation_report.json",
    "same_values_different_rule_report.json", "same_rule_different_values_report.json",
    "rule_hierarchy_conflict_report.json", "default_pocket_shortcut_report.json",
    "first_pocket_shortcut_report.json", "stale_pocket_shortcut_report.json",
    "visible_noisy_shortcut_report.json", "arm_comparison.json", "determinism_replay_report.json",
    "decision.json", "summary.json", "report.md",
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
    upstream_z = load_json(root / "upstream_142z_manifest.json")
    upstream_f = load_json(root / "upstream_142f_manifest.json")
    if upstream_z.get("decision") != "multi_pocket_arbitration_probe_recommended":
        failures.append(f"BAD_142Z_DECISION:{upstream_z.get('decision')}")
    if upstream_z.get("next") != "143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE":
        failures.append(f"BAD_142Z_NEXT:{upstream_z.get('next')}")
    if upstream_f.get("decision") != "instnct_pocket_gated_conflict_priority_transfer_scale_confirmed":
        failures.append(f"BAD_142F_DECISION:{upstream_f.get('decision')}")
    if upstream_f.get("next") != "142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN":
        failures.append(f"BAD_142F_NEXT:{upstream_f.get('next')}")
    audit = upstream_f.get("helper_request_audit", {})
    if audit.get("all_requests_allowed_keys_only") is not True or audit.get("forbidden_keys_present_count") != 0:
        failures.append(f"BAD_142F_HELPER_AUDIT:{audit}")
    if upstream_f.get("per_seed_gate_passed") is not True or upstream_f.get("per_family_gate_passed") is not True:
        failures.append("BAD_142F_HARDENED_GATE_REPORTS")


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
    if config.get("runner_may_call_raw_generate") is not True or config.get("checker_may_call_raw_generate") is not False:
        failures.append("RAW_GENERATE_RUNNER_CHECKER_BOUNDARY_BAD")
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
    manifest = load_json(root / "multi_pocket_eval_manifest.json")
    pocket = load_json(root / "multi_pocket_manifest.json")
    arbitration = load_json(root / "arbitration_rule_manifest.json")
    distribution = load_json(root / "pocket_distribution_report.json")
    marker = load_json(root / "explicit_marker_audit.json")
    if manifest.get("row_count") != 1008:
        failures.append(f"BAD_ROW_COUNT:{manifest.get('row_count')}")
    if manifest.get("family_count") != 7 or set(manifest.get("families", [])) != EXPECTED_FAMILIES:
        failures.append(f"BAD_FAMILY_SET:{manifest.get('families')}")
    if pocket.get("has_all_pocket_winners") is not True:
        failures.append(f"POCKET_WINNER_COVERAGE_MISSING:{pocket.get('selected_pocket_counts')}")
    if distribution.get("pocket_distribution_balanced") is not True:
        failures.append("POCKET_DISTRIBUTION_NOT_BALANCED")
    for key in ["pocket_a_wins_rate", "pocket_b_wins_rate", "pocket_c_wins_rate"]:
        if distribution.get(key, 0.0) < 0.20:
            failures.append(f"POCKET_RATE_TOO_LOW:{key}:{distribution.get(key)}")
    if arbitration.get("arbitration_inversion_pair_count", 0) < 200:
        failures.append(f"ARBITRATION_INVERSION_PAIR_COUNT_TOO_LOW:{arbitration.get('arbitration_inversion_pair_count')}")
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


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    comparison = load_json(root / "arm_comparison.json")
    required_exact_zero = [
        "default_pocket_shortcut_rate", "first_pocket_shortcut_rate", "last_pocket_shortcut_rate",
        "stale_pocket_shortcut_rate", "resolved_final_marker_echo_rate",
        "visible_bypass_violation_rate", "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
    ]
    for key in required_exact_zero:
        if metrics.get(key) != 0.0:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    for key, threshold in {
        "main_final_answer_accuracy": 0.70,
        "multi_pocket_arbitration_accuracy": 0.70,
        "quorum_rule_accuracy": 0.65,
        "recency_rule_accuracy": 0.65,
        "tie_break_accuracy": 0.65,
        "rule_hierarchy_conflict_accuracy": 0.65,
        "priority_inversion_accuracy": 0.70,
        "same_template_opposite_winner_accuracy": 0.70,
        "pocket_label_permutation_accuracy": 0.70,
        "same_values_different_rule_accuracy": 0.70,
        "same_rule_different_values_accuracy": 0.70,
        "main_pocket_writeback_rate": 0.80,
        "main_contrast_group_accuracy": 0.70,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    if metrics.get("ablation_final_answer_accuracy", 1.0) > 0.15:
        failures.append(f"ABLATION_TOO_HIGH:{metrics.get('ablation_final_answer_accuracy')}")
    if metrics.get("pocket_ablation_delta_final_answer_accuracy", 0.0) < 0.50:
        failures.append(f"ABLATION_DELTA_TOO_LOW:{metrics.get('pocket_ablation_delta_final_answer_accuracy')}")
    if metrics.get("deterministic_replay_passed") is not True:
        failures.append("AGGREGATE_DETERMINISM_FALSE")
    if comparison.get("priority_inversion_pair_count", 0) < 200:
        failures.append(f"COMPARISON_INVERSION_PAIR_COUNT_TOO_LOW:{comparison.get('priority_inversion_pair_count')}")
    infra = metrics.get("infrastructure_gates", {})
    for key in [
        "expected_output_canary_passed", "ast_scan_passed", "leakage_rejected",
        "controls_failed", "generated_text_before_scoring",
        "helper_request_keys_allowed_only", "no_expected_scorer_oracle_metadata",
        "deterministic_replay_passed",
    ]:
        if infra.get(key) is not True:
            failures.append(f"INFRA_GATE_FAILED:{key}:{infra.get(key)}")


def require_controls(root: Path, failures: list[str]) -> None:
    report = load_json(root / "control_arm_report.json")
    if report.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_ALL_FAIL")
    missing = REQUIRED_CONTROLS - set(report.get("required_controls", []))
    if missing:
        failures.append(f"MISSING_CONTROLS:{sorted(missing)}")
    for control in REQUIRED_CONTROLS:
        if report.get(f"{control.lower()}_failed") is not True:
            failures.append(f"CONTROL_NOT_FAILED:{control}")


def require_hardening_reports(root: Path, failures: list[str]) -> None:
    for name in [
        "resolved_final_marker_echo_report.json",
        "pocket_label_permutation_report.json",
        "same_values_different_rule_report.json",
        "same_rule_different_values_report.json",
        "rule_hierarchy_conflict_report.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
        "per_pocket_gate_report.json",
        "shortcut_report.json",
    ]:
        payload = load_json(root / name)
        if payload.get("passed") is not True:
            failures.append(f"HARDENING_REPORT_FAILED:{name}:{payload}")
    echo = load_json(root / "resolved_final_marker_echo_report.json")
    if echo.get("resolved_final_marker_echo_rate") != 0.0:
        failures.append(f"RESOLVED_MARKER_ECHO_RATE_NONZERO:{echo.get('resolved_final_marker_echo_rate')}")
    if echo.get("resolved_final_marker_echo_control_failed") is not True:
        failures.append("RESOLVED_MARKER_ECHO_CONTROL_NOT_FAILED")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_pocket_arbitration_probe_positive":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_POSITIVE":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    if decision.get("architecture_superiority_claimed") is not False:
        failures.append("ARCHITECTURE_SUPERIORITY_CLAIMED")
    if decision.get("value_grounding_claimed") is not False:
        failures.append("VALUE_GROUNDING_CLAIMED")


def require_docs(failures: list[str]) -> None:
    required = [
        "constrained helper/backend",
        "multi-pocket arbitration",
        "resolved-final-marker echo",
        "not open-ended reasoning",
        "not GPT-like",
        "not broad assistant capability",
        "not production/public API/deployment/safety readiness",
        "not architecture superiority",
    ]
    for rel_path in DOCS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_DOC:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in required:
            if phrase not in text:
                failures.append(f"DOC_MISSING_PHRASE:{rel_path}:{phrase}")


def run_check(root: Path) -> list[str]:
    failures: list[str] = []
    require_changed_files(failures)
    for path in [REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]:
        if not path.exists():
            failures.append(f"MISSING_SCRIPT:{path}")
        else:
            failures.extend(ast_scan(path))
    require_artifacts(root, failures)
    if not failures:
        require_upstreams(root, failures)
        require_config(root, failures)
        require_infrastructure(root, failures)
        require_manifests(root, failures)
        require_generation_trace(root, failures)
        require_metrics(root, failures)
        require_controls(root, failures)
        require_hardening_reports(root, failures)
        require_decision(root, failures)
    require_docs(failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143A multi-pocket arbitration probe artifacts")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_check(root)
    if failures:
        print("143A CHECK FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("143A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
