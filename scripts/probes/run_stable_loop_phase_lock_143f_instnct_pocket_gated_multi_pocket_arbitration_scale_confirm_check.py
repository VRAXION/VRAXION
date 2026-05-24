#!/usr/bin/env python3
"""Checker for 143F multi-pocket arbitration scale/dependency probe."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM_RESULT.md",
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
PASS_DECISIONS = {
    "instnct_pocket_gated_multi_pocket_arbitration_scale_confirmed",
    "resolved_final_marker_dependency_confirmed",
}
REQUIRED_ARTIFACTS = [
    "queue.json", "progress.jsonl", "upstream_143a_manifest.json",
    "eval_config.json", "helper_provenance_verification.json", "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json", "forbidden_input_rejection_report.json",
    "multi_pocket_eval_manifest.json", "multi_pocket_manifest.json", "arbitration_rule_manifest.json",
    "explicit_marker_audit.json", "resolved_marker_present_marker_audit.json",
    "no_resolved_final_marker_marker_audit.json", "eval_rows.jsonl",
    "resolved_marker_present_subset_rows.jsonl", "no_resolved_final_marker_subset_rows.jsonl",
    "selection_report.json", "mutation_candidate_results.jsonl", "mutation_search_trace.jsonl",
    "fitness_landscape.json", "raw_generation_trace.jsonl", "raw_generation_results.jsonl",
    "pocket_trace.jsonl", "pocket_ablation_results.jsonl",
    "no_resolved_final_marker_subset_results.jsonl", "no_resolved_final_marker_subset_scoring.jsonl",
    "no_resolved_abc_static_marker_control_results.jsonl", "scoring_results.jsonl",
    "contrast_group_results.jsonl", "arbitration_inversion_pairs.jsonl",
    "control_results.jsonl", "control_arm_report.json", "multi_pocket_control_report.json",
    "generated_before_scoring_report.json", "freshness_leakage_audit.json",
    "multi_pocket_arbitration_metrics.json", "aggregate_metrics.json", "arm_comparison.json",
    "helper_request_audit.json", "canonical_metric_alias_report.json",
    "per_seed_gate_report.json", "per_family_gate_report.json", "per_pocket_gate_report.json",
    "per_seed_metrics.json", "per_family_metrics.json", "per_pocket_metrics.json",
    "pocket_distribution_report.json", "winner_distribution_report.json", "shortcut_report.json",
    "resolved_final_marker_echo_report.json", "priority_inversion_report.json",
    "priority_inversion_pair_report.json", "no_resolved_prompt_scan_report.json",
    "no_resolved_final_marker_subset_manifest.json", "no_resolved_final_marker_subset_report.json",
    "resolved_marker_present_subset_report.json", "resolved_marker_dependency_report.json",
    "no_resolved_final_marker_shortcut_report.json", "no_resolved_abc_static_marker_control_report.json",
    "no_resolved_explicit_winner_label_subset_report.json",
    "no_resolved_rule_derived_winner_subset_report.json",
    "determinism_replay_report.json", "decision.json", "summary.json", "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored", "raw_assistant_capability_restored", "structured_tool_capability_restored",
    "gpt_like_readiness_claimed", "open_domain_assistant_readiness_claimed", "production_chat_claimed",
    "public_api_claimed", "deployment_readiness_claimed", "safety_alignment_claimed",
]
BOUNDARY_PHRASES = [
    "constrained helper/backend",
    "not open-ended reasoning",
    "not general composition",
    "not GPT-like",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]
NO_RESOLVED_BANNED_PATTERNS = [
    r"arbitrated[-_ ]?final",
    r"selected[-_ ]?final",
    r"winner[-_ ]?value",
    r"final[-_ ]?target",
    r"expected[-_ ]?answer",
    r"gold[-_ ]?output",
    r"answer[-_ ]?value",
    r"resolved[-_ ]?final",
    r"ANSWER\s*=",
    r"TARGET\s*=",
    r"GOLD\s*=",
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


def require_config(root: Path, failures: list[str]) -> None:
    config = load_json(root / "eval_config.json")
    expected_false = {
        "train_allowed",
        "training_performed",
        "helper_backend_modification_allowed",
        "public_request_key_change_allowed",
        "source_checkpoint_mutation_allowed",
        "runtime_surface_mutated",
        "release_surface_mutated",
        "product_surface_mutated",
        "root_license_changed",
    }
    for key in expected_false:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_NOT_FALSE:{key}:{config.get(key)}")
    if config.get("helper_generation_allowed") is not True or config.get("shared_helper_only") is not True:
        failures.append("HELPER_ONLY_GENERATION_BOUNDARY_MISSING")
    if config.get("runner_may_call_raw_generate") is not True or config.get("checker_may_call_raw_generate") is not False:
        failures.append("RAW_GENERATE_RUNNER_CHECKER_BOUNDARY_BAD")
    if set(config.get("families", [])) != EXPECTED_FAMILIES:
        failures.append(f"BAD_CONFIG_FAMILIES:{config.get('families')}")
    require_false_flags(config, failures)


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143a_manifest.json")
    if upstream.get("decision") != "instnct_pocket_gated_multi_pocket_arbitration_probe_positive":
        failures.append(f"BAD_143A_DECISION:{upstream.get('decision')}")
    if upstream.get("next") != "143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM":
        failures.append(f"BAD_143A_NEXT:{upstream.get('next')}")
    exact = {
        "main_final_answer_accuracy": 1.0,
        "multi_pocket_arbitration_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "ablation_final_answer_accuracy": 0.0,
        "pocket_ablation_delta_final_answer_accuracy": 1.0,
        "resolved_final_marker_echo_rate": 0.0,
    }
    for key, expected in exact.items():
        if upstream.get(key) != expected:
            failures.append(f"BAD_143A_METRIC:{key}:{upstream.get(key)}")
    if upstream.get("resolved_final_marker_echo_control_failed") is not True:
        failures.append("BAD_143A_ECHO_CONTROL")
    if upstream.get("deterministic_replay_passed") is not True:
        failures.append("BAD_143A_DETERMINISM")


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


def require_helper_audit(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "helper_request_audit.json")
    if set(audit.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_AUDIT_ALLOWED_KEYS:{audit.get('allowed_helper_keys')}")
    for key, expected in {
        "all_requests_allowed_keys_only": True,
        "forbidden_keys_present_count": 0,
        "no_forbidden_keys_in_accepted_generation_requests": True,
        "raw_generate_allowed_in_runner": True,
        "raw_generate_allowed_in_checker": False,
        "shared_helper_only": True,
    }.items():
        if audit.get(key) != expected:
            failures.append(f"BAD_HELPER_AUDIT:{key}:{audit.get(key)}")
    rows = read_jsonl(root / "raw_generation_trace.jsonl")
    if not rows:
        failures.append("RAW_GENERATION_TRACE_EMPTY")
    for index, row in enumerate(rows[:100]):
        request = row.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append(f"BAD_HELPER_REQUEST_KEYS:{index}:{sorted(request)}")
        if row.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append(f"BAD_RAW_BACKEND:{index}:{row.get('backend_name')}")
        if row.get("generated_before_scoring") is not True:
            failures.append(f"GENERATED_BEFORE_SCORING_FALSE:{index}")


def require_manifests(root: Path, failures: list[str]) -> None:
    manifest = load_json(root / "multi_pocket_eval_manifest.json")
    marker = load_json(root / "explicit_marker_audit.json")
    no_marker = load_json(root / "no_resolved_final_marker_marker_audit.json")
    pocket = load_json(root / "multi_pocket_manifest.json")
    distribution = load_json(root / "pocket_distribution_report.json")
    if manifest.get("resolved_marker_present_row_count") < 2688:
        failures.append(f"RESOLVED_ROW_COUNT_TOO_LOW:{manifest.get('resolved_marker_present_row_count')}")
    if manifest.get("no_resolved_final_marker_row_count") < 672:
        failures.append(f"NO_RESOLVED_ROW_COUNT_TOO_LOW:{manifest.get('no_resolved_final_marker_row_count')}")
    if manifest.get("family_count") != 7 or set(manifest.get("families", [])) != EXPECTED_FAMILIES:
        failures.append(f"BAD_FAMILY_SET:{manifest.get('families')}")
    if manifest.get("scaffold_variant_count", 0) < 84:
        failures.append(f"SCAFFOLD_VARIANT_COUNT_TOO_LOW:{manifest.get('scaffold_variant_count')}")
    if pocket.get("has_all_pocket_winners") is not True:
        failures.append(f"POCKET_WINNER_COVERAGE_MISSING:{pocket.get('selected_pocket_counts')}")
    if distribution.get("passed") is not True:
        failures.append("POCKET_DISTRIBUTION_NOT_BALANCED")
    if marker.get("direct_pocket_value_marker_rate") != 0.0:
        failures.append(f"DIRECT_POCKET_VALUE_MARKER_PRESENT:{marker.get('direct_pocket_value_marker_rate')}")
    if no_marker.get("resolved_final_marker_row_rate") != 0.0:
        failures.append(f"NO_RESOLVED_HAS_RESOLVED_MARKER:{no_marker.get('resolved_final_marker_row_rate')}")


def require_no_resolved(root: Path, failures: list[str]) -> None:
    rows = read_jsonl(root / "no_resolved_final_marker_subset_rows.jsonl")
    scan = load_json(root / "no_resolved_prompt_scan_report.json")
    manifest = load_json(root / "no_resolved_final_marker_subset_manifest.json")
    report = load_json(root / "no_resolved_final_marker_subset_report.json")
    abc = load_json(root / "no_resolved_abc_static_marker_control_report.json")
    if scan.get("passed") is not True:
        failures.append(f"NO_RESOLVED_PROMPT_SCAN_FAILED:{scan.get('banned_matches')}")
    for row in rows[:2000]:
        prompt = row.get("prompt", "")
        for pattern in NO_RESOLVED_BANNED_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                failures.append(f"NO_RESOLVED_BANNED_PROMPT_MARKER:{row.get('row_id')}:{pattern}")
                break
        if row.get("no_resolved_variant") == "no_resolved_rule_derived_winner" and re.search(r"\b(winner|selected_pocket_id)\s*=", prompt, re.IGNORECASE):
            failures.append(f"RULE_DERIVED_HAS_EXPLICIT_WINNER:{row.get('row_id')}")
    if manifest.get("payload_markers") != [
        "arbitrated final:",
        "quorum-selected final:",
        "recency-selected final:",
        "tie-break selected final:",
        "hierarchy-selected final:",
    ]:
        failures.append(f"NO_RESOLVED_MANIFEST_NOT_FINAL_MARKERS_STATIC:{manifest.get('payload_markers')}")
    if manifest.get("payload_markers_static") is not True or manifest.get("per_row_manifest_switching_allowed") is not False:
        failures.append("NO_RESOLVED_STATIC_MANIFEST_FLAGS_BAD")
    for key, expected in {
        "no_resolved_unique_checkpoint_path_count": 1,
        "no_resolved_unique_checkpoint_hash_count": 1,
        "no_resolved_manifest_payload_markers_static": True,
        "no_resolved_per_row_manifest_switch_rate": 0.0,
        "no_resolved_per_row_payload_marker_switch_rate": 0.0,
    }.items():
        if report.get(key) != expected:
            failures.append(f"BAD_NO_RESOLVED_STATIC_METRIC:{key}:{report.get(key)}")
    if abc.get("diagnostic_only") is not True or abc.get("payload_markers_static") is not True:
        failures.append("ABC_STATIC_DIAGNOSTIC_FLAGS_BAD")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    comparison = load_json(root / "arm_comparison.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    if decision.get("decision") not in PASS_DECISIONS:
        failures.append(f"UNACCEPTED_143F_DECISION:{decision.get('decision')}")
    if decision.get("decision") == "resolved_final_marker_dependency_confirmed":
        gates = {
            "resolved_marker_present_subset_accuracy": (0.85, ">="),
            "no_resolved_final_marker_subset_accuracy": (0.60, "<"),
            "no_resolved_final_marker_subset_fallback_rate": (0.0, ">"),
            "resolved_marker_dependency_delta": (0.25, ">="),
            "no_resolved_final_marker_subset_shortcut_rate": (0.0, "=="),
            "no_resolved_final_marker_subset_unexpected_value_rate": (0.0, "=="),
            "no_resolved_final_marker_subset_visible_rate": (0.0, "=="),
            "no_resolved_final_marker_subset_noisy_rate": (0.0, "=="),
            "no_resolved_final_marker_subset_train_namespace_rate": (0.0, "=="),
        }
    else:
        gates = {
            "main_final_answer_accuracy": (0.85, ">="),
            "resolved_marker_present_subset_accuracy": (0.85, ">="),
            "no_resolved_final_marker_subset_accuracy": (0.60, ">="),
            "no_resolved_final_marker_subset_writeback_rate": (0.70, ">="),
            "no_resolved_final_marker_subset_shortcut_rate": (0.0, "=="),
        }
    for key, (threshold, comparator) in gates.items():
        value = comparison.get(key)
        if comparator == ">=" and not (value >= threshold):
            failures.append(f"METRIC_GATE_FAIL:{key}:{value}<{threshold}")
        if comparator == ">" and not (value > threshold):
            failures.append(f"METRIC_GATE_FAIL:{key}:{value}<={threshold}")
        if comparator == "<" and not (value < threshold):
            failures.append(f"METRIC_GATE_FAIL:{key}:{value}>={threshold}")
        if comparator == "==" and value != threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}:{value}!={threshold}")
    normal_zero = [
        "default_pocket_shortcut_rate",
        "first_pocket_shortcut_rate",
        "last_pocket_shortcut_rate",
        "stale_pocket_shortcut_rate",
        "resolved_final_marker_echo_rate",
        "visible_bypass_violation_rate",
        "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
    ]
    for key in normal_zero:
        if comparison.get(key) != 0.0:
            failures.append(f"NORMAL_SHORTCUT_NONZERO:{key}:{comparison.get(key)}")
    if comparison.get("ablation_final_answer_accuracy", 1.0) > 0.10:
        failures.append(f"ABLATION_TOO_HIGH:{comparison.get('ablation_final_answer_accuracy')}")
    if comparison.get("pocket_ablation_delta_final_answer_accuracy", 0.0) < 0.75:
        failures.append(f"ABLATION_DELTA_TOO_LOW:{comparison.get('pocket_ablation_delta_final_answer_accuracy')}")
    infra = metrics.get("infrastructure_gates", {})
    for key in ["expected_output_canary_passed", "ast_scan_passed", "leakage_rejected", "controls_failed", "generated_text_before_scoring", "helper_request_keys_allowed_only", "no_expected_scorer_oracle_metadata", "deterministic_replay_passed"]:
        if infra.get(key) is not True:
            failures.append(f"INFRA_GATE_FAIL:{key}:{infra.get(key)}")
    if infra.get("forbidden_helper_metadata_count") != 0:
        failures.append(f"FORBIDDEN_HELPER_METADATA_COUNT:{infra.get('forbidden_helper_metadata_count')}")
    for payload_name, payload in {"decision": decision, "summary": summary}.items():
        require_false_flags(payload, failures)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_PHRASE_MISSING:{payload_name}:{phrase}")


def require_reports(root: Path, failures: list[str]) -> None:
    control = load_json(root / "control_arm_report.json")
    echo = load_json(root / "resolved_final_marker_echo_report.json")
    seed = load_json(root / "per_seed_gate_report.json")
    family = load_json(root / "per_family_gate_report.json")
    pocket = load_json(root / "per_pocket_gate_report.json")
    shortcut = load_json(root / "shortcut_report.json")
    dependency = load_json(root / "resolved_marker_dependency_report.json")
    if control.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    if echo.get("passed") is not True or echo.get("resolved_final_marker_echo_control_failed") is not True:
        failures.append("RESOLVED_ECHO_REPORT_NOT_GATED")
    if seed.get("passed") is not True or family.get("passed") is not True or pocket.get("passed") is not True:
        failures.append("PER_SEED_FAMILY_POCKET_GATES_FAILED")
    if shortcut.get("passed") is not True:
        failures.append("SHORTCUT_REPORT_FAILED")
    if dependency.get("resolved_marker_dependency_delta", 0.0) < 0.25:
        failures.append("DEPENDENCY_DELTA_TOO_LOW")


def require_docs(failures: list[str]) -> None:
    for rel_path in DOCS:
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_PHRASE_MISSING:{rel_path}:{phrase}")
        if "NO_RESOLVED_FINAL_MARKER_SUBSET" not in text:
            failures.append(f"DOC_NO_RESOLVED_SUBSET_MISSING:{rel_path}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    for path in [REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]:
        failures.extend(ast_scan(path))
    require_artifacts(root, failures)
    if failures:
        return failures
    require_config(root, failures)
    require_upstream(root, failures)
    require_infrastructure(root, failures)
    require_helper_audit(root, failures)
    require_manifests(root, failures)
    require_no_resolved(root, failures)
    require_metrics(root, failures)
    require_reports(root, failures)
    require_docs(failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143F multi-pocket arbitration scale/dependency probe")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143F CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143F CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
