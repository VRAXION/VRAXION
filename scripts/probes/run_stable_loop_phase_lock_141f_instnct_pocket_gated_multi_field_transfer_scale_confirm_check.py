#!/usr/bin/env python3
"""Checker for 141F multi-field transfer scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
EXPECTED_FAMILIES = {
    "FIELD_A_PLUS_FIELD_B_TO_FINAL",
    "POCKET_SOURCE_TABLE_RULE_FIELD",
    "DUAL_POCKET_PRIORITY_CONFLICT",
    "MULTI_FIELD_SAME_TEMPLATE_CONTRAST",
    "DISTRACTOR_FIELD_MIX",
    "INTERMEDIATE_FIELD_CHAIN",
}
REQUIRED_CONTROLS = {
    "FIELD_A_ONLY_CONTROL",
    "FIELD_B_ONLY_CONTROL",
    "INTERMEDIATE_COPY_CONTROL",
    "VISIBLE_TARGET_BYPASS_CONTROL",
    "NOISY_DISTRACTOR_CONTROL",
    "CLOSED_POCKET_ABLATION_CONTROL",
    "SINGLE_FIELD_SHORTCUT_CONTROL",
    "PRIORITY_CONFLICT_WRONG_FIELD_CONTROL",
    "PREFIX_ONLY_CONTROL",
}
REQUIRED_ARTIFACTS = [
    "queue.json", "progress.jsonl", "upstream_141a_manifest.json",
    "eval_config.json", "helper_provenance_verification.json", "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json", "forbidden_input_rejection_report.json",
    "multi_field_eval_manifest.json", "multi_field_binding_manifest.json", "explicit_marker_audit.json",
    "eval_rows.jsonl", "mutation_candidate_results.jsonl", "mutation_search_trace.jsonl", "selection_report.json",
    "fitness_landscape.json", "raw_generation_trace.jsonl", "raw_generation_results.jsonl", "pocket_trace.jsonl",
    "pocket_ablation_results.jsonl", "scoring_results.jsonl", "contrast_group_results.jsonl",
    "control_results.jsonl", "control_arm_report.json", "visible_bypass_control_report.json",
    "noisy_distractor_control_report.json", "generated_before_scoring_report.json", "freshness_leakage_audit.json",
    "multi_field_transfer_metrics.json", "aggregate_metrics.json", "field_shortcut_report.json", "priority_conflict_report.json",
    "single_field_shortcut_report.json", "per_seed_metrics.json", "per_family_metrics.json", "arm_comparison.json",
    "determinism_replay_report.json", "decision.json", "summary.json", "report.md",
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
        if not line.strip():
            continue
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
    return failures


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_upstreams(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_141a_manifest.json")
    if upstream.get("decision") != "instnct_pocket_gated_multi_field_transfer_probe_positive":
        failures.append(f"BAD_141A_DECISION:{upstream.get('decision')}")
    if upstream.get("next") != "141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM":
        failures.append(f"BAD_141A_NEXT:{upstream.get('next')}")
    if upstream.get("selected_candidate") != "open_multi_field_final_all_markers":
        failures.append(f"BAD_141A_SELECTED_CANDIDATE:{upstream.get('selected_candidate')}")
    if upstream.get("main_final_answer_accuracy") != 1.0:
        failures.append(f"BAD_141A_FINAL_ACCURACY:{upstream.get('main_final_answer_accuracy')}")
    if upstream.get("main_multi_field_binding_accuracy") != 1.0:
        failures.append(f"BAD_141A_BINDING_ACCURACY:{upstream.get('main_multi_field_binding_accuracy')}")
    if upstream.get("main_pocket_writeback_rate") != 1.0:
        failures.append(f"BAD_141A_WRITEBACK_RATE:{upstream.get('main_pocket_writeback_rate')}")
    if upstream.get("pocket_ablation_delta") != 1.0:
        failures.append(f"BAD_141A_ABLATION_DELTA:{upstream.get('pocket_ablation_delta')}")
    if upstream.get("deterministic_replay_passed") is not True:
        failures.append("BAD_141A_DETERMINISM")


def require_config(root: Path, failures: list[str]) -> None:
    config = load_json(root / "eval_config.json")
    if config.get("train_allowed") is not False or config.get("training_performed") is not False:
        failures.append("TRAINING_BOUNDARY_NOT_FALSE")
    if config.get("helper_backend_modification_allowed") is not False:
        failures.append("HELPER_BACKEND_MODIFICATION_ALLOWED")
    if config.get("public_request_key_change_allowed") is not False:
        failures.append("PUBLIC_REQUEST_KEY_CHANGE_ALLOWED")
    if config.get("source_checkpoint_mutation_allowed") is not False:
        failures.append("SOURCE_CHECKPOINT_MUTATION_ALLOWED")
    for key in ["runtime_surface_mutated", "release_surface_mutated", "product_surface_mutated", "root_license_changed"]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_NOT_FALSE:{key}")
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
    eval_manifest = load_json(root / "multi_field_eval_manifest.json")
    binding_manifest = load_json(root / "multi_field_binding_manifest.json")
    audit = load_json(root / "explicit_marker_audit.json")
    if eval_manifest.get("row_count", 0) < 2880:
        failures.append(f"ROW_COUNT_TOO_LOW:{eval_manifest.get('row_count')}")
    if eval_manifest.get("family_count") != 6:
        failures.append(f"BAD_FAMILY_COUNT:{eval_manifest.get('family_count')}")
    if set(eval_manifest.get("families", [])) != EXPECTED_FAMILIES:
        failures.append(f"BAD_FAMILIES:{eval_manifest.get('families')}")
    if eval_manifest.get("scaffold_variant_count", 0) < 72:
        failures.append(f"SCAFFOLD_VARIANTS_TOO_LOW:{eval_manifest.get('scaffold_variant_count')}")
    if binding_manifest.get("final_distinct_from_fields_rate") != 1.0:
        failures.append(f"FINAL_NOT_DISTINCT_FROM_FIELDS:{binding_manifest.get('final_distinct_from_fields_rate')}")
    if audit.get("direct_pocket_value_marker_rate") != 0.0:
        failures.append(f"DIRECT_POCKET_VALUE_MARKER_PRESENT:{audit.get('direct_pocket_value_marker_rate')}")
    if audit.get("implicit_or_minimal_gate_row_rate", 0.0) < 0.70:
        failures.append(f"IMPLICIT_GATE_RATE_TOO_LOW:{audit.get('implicit_or_minimal_gate_row_rate')}")


def require_generation_trace(root: Path, failures: list[str]) -> None:
    rows = read_jsonl(root / "raw_generation_trace.jsonl")
    if not rows:
        failures.append("RAW_GENERATION_TRACE_EMPTY")
        return
    for index, row in enumerate(rows[:50]):
        request = row.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append(f"BAD_HELPER_REQUEST_KEYS:{index}:{sorted(request)}")
        if row.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append(f"BAD_RAW_BACKEND:{index}:{row.get('backend_name')}")
        if row.get("generated_before_scoring") is not True:
            failures.append(f"GENERATED_BEFORE_SCORING_FALSE:{index}")


def require_controls(root: Path, failures: list[str]) -> None:
    control_report = load_json(root / "control_arm_report.json")
    controls = read_jsonl(root / "control_results.jsonl")
    present = {row.get("control") for row in controls}
    if control_report.get("controls_failed") is not True:
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
    per_seed = load_json(root / "per_seed_metrics.json")
    required_equal = {
        "single_field_shortcut_rate": 0.0,
        "field_a_shortcut_rate": 0.0,
        "field_b_shortcut_rate": 0.0,
        "intermediate_copy_shortcut_rate": 0.0,
        "visible_bypass_violation_rate": 0.0,
        "noisy_distractor_violation_rate": 0.0,
        "direct_pocket_value_marker_rate": 0.0,
        "priority_conflict_wrong_field_rate": 0.0,
    }
    for key, expected in required_equal.items():
        if metrics.get(key) != expected:
            failures.append(f"METRIC_NOT_{expected}:{key}:{metrics.get(key)}")
    lower_bounds = {
        "main_final_answer_accuracy": 0.90,
        "main_multi_field_binding_accuracy": 0.90,
        "main_pocket_writeback_rate": 0.95,
        "main_contrast_group_accuracy": 0.90,
        "pocket_ablation_delta": 0.85,
    }
    for key, floor in lower_bounds.items():
        if metrics.get(key, 0.0) < floor:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}<{floor}")
    if metrics.get("ablation_final_answer_accuracy", 1.0) > 0.05:
        failures.append(f"ABLATION_ACCURACY_TOO_HIGH:{metrics.get('ablation_final_answer_accuracy')}")
    if metrics.get("deterministic_replay_passed") is not True:
        failures.append("METRIC_DETERMINISM_NOT_TRUE")
    if metrics.get("value_grounding_claimed") is not False:
        failures.append("VALUE_GROUNDING_CLAIMED")
    if metrics.get("architecture_superiority_claimed") is not False:
        failures.append("ARCHITECTURE_SUPERIORITY_CLAIMED")
    canonical = set(aggregate.get("canonical_metric_names", []))
    for name in [
        "direct_pocket_value_marker_rate",
        "main_final_answer_accuracy",
        "main_multi_field_binding_accuracy",
        "main_pocket_writeback_rate",
        "priority_conflict_wrong_field_rate",
    ]:
        if name not in canonical:
            failures.append(f"CANONICAL_METRIC_MISSING:{name}")
        if name not in aggregate:
            failures.append(f"AGGREGATE_METRIC_MISSING:{name}")
    gates = aggregate.get("infrastructure_gates", {})
    for key in [
        "expected_output_canary_passed",
        "ast_scan_passed",
        "leakage_rejected",
        "controls_failed",
        "generated_text_before_scoring",
        "helper_request_keys_allowed_only",
        "no_expected_scorer_oracle_metadata",
        "deterministic_replay_passed",
    ]:
        if gates.get(key) is not True:
            failures.append(f"INFRASTRUCTURE_GATE_FAILED:{key}")
    for seed, seed_metrics in per_seed.get("main", {}).items():
        seed_ablation = per_seed.get("ablation", {}).get(seed, {})
        if seed_metrics.get("final_answer_accuracy", 0.0) < 0.85:
            failures.append(f"SEED_FINAL_ACCURACY_BELOW_GATE:{seed}")
        if seed_metrics.get("multi_field_binding_accuracy", 0.0) < 0.85:
            failures.append(f"SEED_BINDING_ACCURACY_BELOW_GATE:{seed}")
        if seed_metrics.get("pocket_writeback_rate", 0.0) < 0.90:
            failures.append(f"SEED_WRITEBACK_BELOW_GATE:{seed}")
        if seed_ablation.get("final_answer_accuracy", 1.0) > 0.10:
            failures.append(f"SEED_ABLATION_TOO_HIGH:{seed}")
        for key in [
            "single_field_shortcut_rate",
            "field_a_shortcut_rate",
            "field_b_shortcut_rate",
            "priority_conflict_wrong_field_rate",
            "visible_bypass_violation_rate",
            "noisy_distractor_violation_rate",
        ]:
            if seed_metrics.get(key) != 0.0:
                failures.append(f"SEED_METRIC_NOT_ZERO:{seed}:{key}:{seed_metrics.get(key)}")


def require_selection(root: Path, failures: list[str]) -> None:
    selection = load_json(root / "selection_report.json")
    if selection.get("selected_candidate") != "open_multi_field_final_all_markers":
        failures.append(f"BAD_SELECTED_CANDIDATE:{selection.get('selected_candidate')}")
    if selection.get("gradient_used") is not False:
        failures.append("GRADIENT_USED_NOT_FALSE")
    if selection.get("selected_by_fitness") is not True:
        failures.append("SELECTION_NOT_BY_FITNESS")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_field_transfer_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if decision.get("pocket_mechanism_claimed") is not True:
        failures.append("POCKET_MECHANISM_NOT_CLAIMED_ON_POSITIVE")
    if decision.get("multi_field_transfer_scale_confirmed") is not True:
        failures.append("SCALE_CONFIRM_FLAG_NOT_TRUE")
    if decision.get("per_seed_gate_failures") != []:
        failures.append(f"PER_SEED_GATE_FAILURES:{decision.get('per_seed_gate_failures')}")
    if decision.get("value_grounding_claimed") is not False:
        failures.append("BROAD_VALUE_GROUNDING_CLAIMED")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)


def require_report(root: Path, failures: list[str]) -> None:
    report = (root / "report.md").read_text(encoding="utf-8")
    required_phrases = [
        "multi-field final selection under controlled helper manifest",
        "not open-ended reasoning",
        "not general composition",
        "not GPT-like readiness",
        "not open-domain reasoning",
        "not broad assistant capability",
        "not production/public",
        "not general architecture superiority",
    ]
    for phrase in required_phrases:
        if phrase not in report:
            failures.append(f"REPORT_MISSING_BOUNDARY:{phrase}")


def require_docs(failures: list[str]) -> None:
    for doc in DOCS:
        path = REPO_ROOT / doc
        text = path.read_text(encoding="utf-8")
        for phrase in [
            "141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM",
            "multi-field",
            "single-field shortcut",
            "not GPT-like readiness",
            "not broad assistant capability",
            "not open-ended reasoning",
            "not general composition",
        ]:
            if phrase not in text:
                failures.append(f"DOC_MISSING_PHRASE:{doc}:{phrase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 141F multi-field transfer scale confirm artifacts.")
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
        print("141F CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("141F CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
