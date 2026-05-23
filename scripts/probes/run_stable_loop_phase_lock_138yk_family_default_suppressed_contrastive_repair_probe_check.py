#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138YK family-default-suppressed repair/probe."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
SOURCE_CHECKPOINT = "target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke/checkpoints/target_138yi_family_contrastive_value/model.pt"
TARGET_CHECKPOINT = "target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke/checkpoints/target_138yk_family_default_suppressed/model.pt"

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yj_manifest.json",
    "upstream_138yd_manifest.json",
    "upstream_138yi_manifest.json",
    "determinism_manifest.json",
    "train_config.json",
    "eval_config.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "helper_provenance_verification.json",
    "forbidden_input_rejection_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "train_rows.jsonl",
    "eval_rows.jsonl",
    "eval_row_hashes.json",
    "family_default_value_bank.json",
    "hard_negative_default_rows.jsonl",
    "contrast_group_manifest.json",
    "ood_family_value_manifest.json",
    "freshness_leakage_audit.json",
    "training_metrics.jsonl",
    "training_objective_report.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "scoring_results.jsonl",
    "contrast_group_results.jsonl",
    "family_default_suppression_metrics.json",
    "intra_family_contrastive_metrics.json",
    "family_default_attractor_report.json",
    "high_frequency_value_replay_report.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "control_results.jsonl",
    "control_arm_report.json",
    "generated_before_scoring_report.json",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "aggregate_metrics.json",
    "determinism_replay_report.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
    "evidence_rebuild_status.json",
    "decision.json",
    "summary.json",
    "report.md",
]
OPTIONAL_ARTIFACTS = ["post_wrapper_carrier_proxy_report.json"]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
    "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_POSITIVE",
    "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_FAILS",
    "FAMILY_DEFAULT_SUPPRESSED_TRAINING_PATH_MISSING",
    "family_default_value_bank",
    "hard_negative_default_rows",
    "HARD_NEGATIVE_DEFAULT_CONTROL",
    "PEER_EXPECTED_VALUE_CONFUSION_CONTROL",
    "shared_raw_generation_helper.py",
    "generated_text",
    "expected-output canary",
    "If family default values are still used, it fails",
    "Raw assistant capability remains quarantined",
    "Structured/tool capability remains invalidated",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "row_answer",
    "eval_family",
    "answer",
    "expected_values",
}
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
]
FINAL_EVAL_FALSE_FLAGS = [
    "expected_output_used_for_generation",
    "expected_payload_used_for_generation",
    "scorer_metadata_used_for_generation",
    "oracle_rerank_used",
    "verifier_rerank_used",
    "llm_judge_used",
    "teacher_forcing_used",
    "constrained_decoding_used",
    "json_mode_used",
    "grammar_decoder_used",
    "regex_fixer_used",
    "post_generation_repair_used",
    "retry_loop_used",
    "best_of_n_used",
    "actual_tool_execution_used",
    "runtime_tool_call_used",
]
CONTROLS = {
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_ANSWER_CONTROL",
    "DISTRACTOR_COPY_CONTROL",
    "STALE_CHAT_FRAGMENT_CONTROL",
    "TRAIN_NAMESPACE_REPLAY_CONTROL",
    "PREFIX_ONLY_CONTROL",
    "GENERIC_VALUE_CONTROL",
    "FAMILY_DEFAULT_VALUE_CONTROL",
    "SAME_VALUE_FOR_ALL_ROWS_CONTROL",
    "PARROT_COPY_CONTROL",
    "HIGH_FREQUENCY_TRAIN_VALUE_CONTROL",
    "HARD_NEGATIVE_DEFAULT_CONTROL",
    "PEER_EXPECTED_VALUE_CONFUSION_CONTROL",
}
KNOWN_DECISIONS = {
    "family_default_suppressed_contrastive_repair_positive",
    "no_value_improvement",
    "family_default_shortcut_persists",
    "hard_negative_default_failure",
    "contrastive_objective_still_too_weak",
    "high_frequency_value_replay_detected",
    "parrot_trap_copy_shortcut_detected",
    "stale_chat_rollout_failure",
    "namespace_rollout_failure",
    "nondeterministic_family_default_suppressed_probe",
    "family_default_suppressed_eval_leakage",
    "scorer_or_task_weakness",
    "raw_helper_integrity_failure",
    "family_default_suppressed_training_path_missing",
}
KNOWN_VERDICTS = {
    "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_POSITIVE",
    "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_FAILS",
    "FAMILY_DEFAULT_SUPPRESSED_TRAINING_PATH_MISSING",
    "DETERMINISM_REPLAY_MISMATCH",
}
POSITIVE_MIN = {
    "answer_value_accuracy": 0.25,
    "exact_answer_accuracy": 0.20,
    "value_after_prefix_accuracy": 0.25,
    "intra_family_contrastive_accuracy": 0.30,
    "intra_family_unique_correct_value_rate": 0.25,
    "rule_derived_value_accuracy": 0.20,
    "table_derived_value_accuracy": 0.20,
    "composition_derived_value_accuracy": 0.15,
    "ood_symbol_value_accuracy": 0.15,
}
POSITIVE_MAX = {
    "intra_family_mode_collapse_rate": 0.60,
    "family_default_attractor_rate": 0.35,
    "family_default_reuse_rate": 0.35,
    "family_dominant_wrong_value_rate": 0.35,
    "multi_expected_to_single_default_rate": 0.30,
    "same_value_for_all_rows_rate": 0.20,
    "hard_negative_default_violation_rate": 0.20,
    "stale_chat_fragment_rate": 0.10,
    "train_namespace_leak_rate": 0.05,
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "remains quarantined", "remains invalidated"]
FORBIDDEN_CLAIMS = {
    "RAW_ASSISTANT_CAPABILITY_RESTORED_FALSE_CLAIM": ["raw assistant capability restored"],
    "STRUCTURED_TOOL_CAPABILITY_RESTORED_FALSE_CLAIM": ["structured/tool capability restored", "structured tool capability restored"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like readiness improved"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_keys(payload: dict[str, Any], keys: list[str], verdict: str, failures: list[str]) -> None:
    for key in keys:
        if key not in payload:
            failures.append(f"{verdict}:{key}")


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 240) : match.start()]
        if any(marker in window for marker in NEGATION_MARKERS):
            return True
    return False


def find_false_claims(text: str) -> list[str]:
    failures: list[str] = []
    lowered = text.lower()
    for verdict, phrases in FORBIDDEN_CLAIMS.items():
        for phrase in phrases:
            if phrase.lower() in lowered and not claim_is_negated(text, phrase):
                failures.append(verdict)
                break
    return failures


def runtime_surface_mutation_detected() -> bool:
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        return True
    return False


def read_files() -> tuple[list[str], dict[str, str]]:
    missing: list[str] = []
    files: dict[str, str] = {}
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def ast_scan_source(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    old_runner_re = re.compile(r"^run_stable_loop_phase_lock_(?!138yk_family_default_suppressed_contrastive_repair_probe)")
    forbidden_calls = {
        "oracle_rerank",
        "verifier_rerank",
        "llm_judge",
        "grammar_decoder",
        "constrained_decoding",
        "regex_fixer",
        "json_fixer",
        "json_mode",
        "best_of_n",
        "retry_loop",
        "post_generation_repair",
        "runtime_tool_call",
        "actual_tool_execution",
    }
    forbidden_strings = ["docs/product", "docs/releases", "sdk", "deploy", "service_start"]
    for path in paths:
        if not path.exists():
            failures.append(f"MISSING_AST_SOURCE:{path}")
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == Path(RUNNER).name and "shutil.rmtree" in text:
            failures.append("DELETE_OR_CONSOLIDATE_PATH_DETECTED")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if old_runner_re.match(alias.name):
                        failures.append("OLD_RUNNER_IMPORT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if old_runner_re.match(module):
                    failures.append("OLD_RUNNER_IMPORT_DETECTED")
            if isinstance(node, ast.Assign):
                targets = " ".join(ast.unparse(target) for target in node.targets)
                value = ast.unparse(node.value)
                if re.search(r"generated_text|generated", targets) and any(token in value for token in ["expected_output", "expected_payload", "expected_answer", "gold_output", "target_json"]):
                    failures.append("AST_GENERATED_TEXT_FROM_EXPECTED_MATERIAL")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).lower()
                if any(token in name for token in forbidden_calls):
                    failures.append("ORACLE_SHORTCUT_DETECTED")
                if any(token in name for token in forbidden_strings):
                    failures.append("RUNTIME_RELEASE_SURFACE_CALL_DETECTED")
    return failures


def require_false_flags(payload: dict[str, Any], failures: list[str], allow_reasoning_subtrack: bool = False) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_FLAG_NOT_FALSE:{key}")
    if not allow_reasoning_subtrack and payload.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not False:
        failures.append("BOUNDARY_FLAG_NOT_FALSE:reasoning_subtrack_real_raw_evidence_partially_restored")


def check_upstreams(failures: list[str]) -> None:
    yj = load_json(SMOKE_ROOT / "upstream_138yj_manifest.json")
    yd = load_json(SMOKE_ROOT / "upstream_138yd_manifest.json")
    yi = load_json(SMOKE_ROOT / "upstream_138yi_manifest.json")
    if yj.get("decision") != "family_default_suppressed_contrastive_objective_plan_complete" or yj.get("next") != "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE":
        failures.append("UPSTREAM_138YJ_ROUTE_MISMATCH")
    if yj.get("hard_negative_default_plan_present") is not True or yj.get("family_default_value_bank_required") is not True:
        failures.append("UPSTREAM_138YJ_HARD_NEGATIVE_PLAN_MISSING")
    if yd.get("decision") != "family_default_shortcut_analysis_complete" or yd.get("root_cause") != "contrastive_objective_too_weak":
        failures.append("UPSTREAM_138YD_ROUTE_MISMATCH")
    if yd.get("contrast_group_default_shortcut_rate") != 0.78125 or yd.get("multi_expected_to_single_default_rate") != 0.6822916666666666 or yd.get("family_default_control_failed") is not True:
        failures.append("UPSTREAM_138YD_EVIDENCE_PROFILE_MISMATCH")
    for key in ["objective_explicitly_penalizes_family_default", "objective_explicitly_penalizes_same_value_for_all_rows", "objective_rewards_intra_family_distinct_values"]:
        if yd.get(key) is not False:
            failures.append(f"UPSTREAM_138YD_OBJECTIVE_PROFILE_MISMATCH:{key}")
    if yi.get("decision") != "high_frequency_train_value_replay_detected" or yi.get("verdict") != "FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_FAILS":
        failures.append("UPSTREAM_138YI_ROUTE_MISMATCH")
    required_yi = {
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "intra_family_contrastive_accuracy": 0.0,
        "intra_family_mode_collapse_rate": 0.9427083333333334,
        "family_default_attractor_rate": 0.78125,
        "family_default_shortcut_detected": True,
        "high_frequency_train_value_replay_detected": True,
        "parrot_trap_detected": False,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
        "determinism_replay_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
        "generated_text_before_scoring": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
    }
    for key, expected in required_yi.items():
        if yi.get(key) != expected:
            failures.append(f"UPSTREAM_138YI_PROFILE_MISMATCH:{key}")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    for rel in OPTIONAL_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_OPTIONAL_DIAGNOSTIC_ARTIFACT:{rel}")
    if failures:
        return failures

    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 20:
        failures.append("PROGRESS_NOT_REFRESHED")
    progress_events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "determinism setup",
        "dataset build",
        "family default value bank build",
        "hard negative row build",
        "training start",
        "target checkpoint write",
        "final_eval generation",
        "scoring",
        "family default suppression analysis",
        "controls",
        "determinism replay",
        "decision",
        "final verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    value_metrics = load_json(SMOKE_ROOT / "value_grounding_metrics.json")
    contrast_metrics = load_json(SMOKE_ROOT / "intra_family_contrastive_metrics.json")
    suppression = load_json(SMOKE_ROOT / "family_default_suppression_metrics.json")
    family_default = load_json(SMOKE_ROOT / "family_default_attractor_report.json")
    high_frequency = load_json(SMOKE_ROOT / "high_frequency_value_replay_report.json")
    parrot = load_json(SMOKE_ROOT / "parrot_trap_report.json")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    before = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    scan = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    source = load_json(SMOKE_ROOT / "source_checkpoint_integrity_manifest.json")
    target = load_json(SMOKE_ROOT / "target_checkpoint_integrity_manifest.json")
    provenance = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    ood = load_json(SMOKE_ROOT / "ood_family_value_manifest.json")
    evidence = load_json(SMOKE_ROOT / "evidence_rebuild_status.json")
    bank = load_json(SMOKE_ROOT / "family_default_value_bank.json")
    train_config = load_json(SMOKE_ROOT / "train_config.json")
    eval_config = load_json(SMOKE_ROOT / "eval_config.json")
    training_objective = load_json(SMOKE_ROOT / "training_objective_report.json")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    raw = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    scoring = read_jsonl(SMOKE_ROOT / "scoring_results.jsonl")
    eval_rows = read_jsonl(SMOKE_ROOT / "eval_rows.jsonl")
    hard_rows = read_jsonl(SMOKE_ROOT / "hard_negative_default_rows.jsonl")
    contrast_rows = read_jsonl(SMOKE_ROOT / "contrast_group_results.jsonl")
    per_seed = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")

    check_upstreams(failures)

    if decision.get("decision") not in KNOWN_DECISIONS:
        failures.append("DECISION_MISMATCH")
    if decision.get("verdict") not in KNOWN_VERDICTS:
        failures.append("VERDICT_MISMATCH")
    positive = decision.get("decision") == "family_default_suppressed_contrastive_repair_positive"
    require_false_flags(decision, failures, allow_reasoning_subtrack=positive)
    require_false_flags(summary, failures, allow_reasoning_subtrack=positive)
    for key in FINAL_EVAL_FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append(f"FINAL_EVAL_FORBIDDEN_FLAG_FAILURE:{key}")
    if decision.get("shared_raw_generation_helper_used") is not True:
        failures.append("HELPER_ONLY_FINAL_EVAL_NOT_RECORDED")

    if canary.get("expected_output_canary_passed") is not True:
        failures.append("EXPECTED_OUTPUT_CANARY_FAILED")
    if scan.get("ast_shortcut_scan_passed") is not True:
        failures.append("AST_SHORTCUT_SCAN_FAILED")
    if leakage.get("leakage_rejected") is not True:
        failures.append("LEAKAGE_NOT_REJECTED")
    if before.get("generated_text_produced_before_scoring") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILURE")
    if replay.get("determinism_replay_passed") is not True and decision.get("decision") != "nondeterministic_family_default_suppressed_probe":
        failures.append("DETERMINISM_REPLAY_NOT_ENFORCED")
    if source.get("source_checkpoint_unchanged") is not True or provenance.get("source_checkpoint_unchanged") is not True:
        failures.append("SOURCE_CHECKPOINT_MUTATION")
    if source.get("source_checkpoint_path") != SOURCE_CHECKPOINT:
        failures.append("SOURCE_CHECKPOINT_PATH_MISMATCH")
    if target.get("target_checkpoint_changed") is not True or provenance.get("target_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_CHANGED")
    if target.get("target_checkpoint_path") != TARGET_CHECKPOINT or provenance.get("selected_checkpoint_path") != TARGET_CHECKPOINT:
        failures.append("TARGET_CHECKPOINT_PATH_MISMATCH")
    if ood.get("eval_values_held_out_from_train") is not True or ood.get("train_eval_value_namespaces_disjoint") is not True:
        failures.append("OOD_FAMILY_VALUE_REQUIREMENT_MISSING")

    if not bank.get("families") or not bank.get("per_family_forbidden_default_values"):
        failures.append("FAMILY_DEFAULT_VALUE_BANK_EMPTY")
    if len(hard_rows) != len(eval_rows) or len(eval_rows) == 0:
        failures.append("HARD_NEGATIVE_ROW_COUNT_MISMATCH")
    for row in hard_rows[:50]:
        if row.get("family_default_hard_negative") is not True or row.get("pass_requires_no_family_default") is not True:
            failures.append("HARD_NEGATIVE_ROW_FLAG_MISSING")
        if not isinstance(row.get("forbidden_family_default_values"), list):
            failures.append("HARD_NEGATIVE_FORBIDDEN_DEFAULTS_MISSING")
    if any("forbidden_family_default_values" not in row or "hard_negative_default_violation" not in row for row in scoring):
        failures.append("SCORING_HARD_NEGATIVE_FIELDS_MISSING")

    if controls.get("controls_failed") is not True:
        failures.append("SCORER_CONTROLS_DID_NOT_FAIL")
    if set(controls.get("controls", {})) != CONTROLS:
        failures.append("SCORER_CONTROLS_INCOMPLETE")
    if controls.get("controls", {}).get("HARD_NEGATIVE_DEFAULT_CONTROL", {}).get("failed") is not True:
        failures.append("HARD_NEGATIVE_DEFAULT_CONTROL_NOT_FAILED")
    if controls.get("controls", {}).get("PEER_EXPECTED_VALUE_CONFUSION_CONTROL", {}).get("failed") is not True:
        failures.append("PEER_EXPECTED_VALUE_CONFUSION_CONTROL_NOT_FAILED")
    if controls.get("controls_called_helper") is not False:
        failures.append("SCORER_CONTROLS_CALLED_HELPER")

    if len(traces) != len(scoring) or len(scoring) != len(eval_rows) or len(raw) != len(scoring):
        failures.append("TRACE_RAW_SCORING_ROW_COUNT_MISMATCH")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_ALLOWED_KEYS_FAILURE")
        if set(request) & FORBIDDEN_HELPER_KEYS:
            failures.append("HELPER_REQUEST_FORBIDDEN_METADATA")
        if trace.get("generated_before_scoring") is not True:
            failures.append("GENERATED_BEFORE_SCORING_FAILURE")
        for key in ["helper_request_hash", "generated_text_hash", "generation_trace_hash", "model_checkpoint_hash", "generation_config_hash"]:
            if not trace.get(key):
                failures.append(f"RAW_GENERATION_TRACE_FIELD_MISSING:{key}")

    required_metrics = [
        "answer_value_accuracy",
        "exact_answer_accuracy",
        "value_after_prefix_accuracy",
        "intra_family_contrastive_accuracy",
        "intra_family_unique_correct_value_rate",
        "intra_family_mode_collapse_rate",
        "family_default_attractor_rate",
        "family_default_reuse_rate",
        "family_dominant_wrong_value_rate",
        "multi_expected_to_single_default_rate",
        "same_value_for_all_rows_rate",
        "hard_negative_default_violation_rate",
        "hard_negative_default_control_failed",
        "rule_derived_value_accuracy",
        "table_derived_value_accuracy",
        "composition_derived_value_accuracy",
        "ood_symbol_value_accuracy",
        "train_namespace_leak_rate",
        "stale_chat_fragment_rate",
        "parrot_trap_detected",
        "high_frequency_train_value_replay_detected",
        "family_default_shortcut_detected",
    ]
    require_keys(aggregate, required_metrics, "AGGREGATE_METRIC_MISSING", failures)
    require_keys(value_metrics, ["answer_value_accuracy", "exact_answer_accuracy", "value_after_prefix_accuracy", "intra_family_contrastive_accuracy", "family_default_reuse_rate", "hard_negative_default_violation_rate"], "VALUE_METRIC_MISSING", failures)
    require_keys(contrast_metrics, ["contrast_group_count", "intra_family_contrastive_accuracy", "multi_expected_to_single_default_rate", "same_value_for_all_rows_rate", "hard_negative_default_violation_rate"], "CONTRAST_METRIC_MISSING", failures)
    for key in ["family_default_attractor_rate", "family_default_reuse_rate", "family_dominant_wrong_value_rate", "multi_expected_to_single_default_rate", "same_value_for_all_rows_rate", "hard_negative_default_violation_rate", "hard_negative_default_control_failed"]:
        if suppression.get(key) != aggregate.get(key):
            failures.append(f"SUPPRESSION_METRIC_MISMATCH:{key}")
    if family_default.get("family_default_shortcut_detected") != aggregate.get("family_default_shortcut_detected"):
        failures.append("FAMILY_DEFAULT_REPORT_MISMATCH")
    if high_frequency.get("high_frequency_train_value_replay_detected") != aggregate.get("high_frequency_train_value_replay_detected"):
        failures.append("HIGH_FREQUENCY_REPORT_MISMATCH")
    if parrot.get("parrot_trap_detected") != aggregate.get("parrot_trap_detected"):
        failures.append("PARROT_TRAP_REPORT_MISMATCH")
    if any("pass" not in row or "same_value_for_all_rows" not in row or "hard_negative_default_emitted" not in row for row in contrast_rows):
        failures.append("CONTRAST_RESULT_FIELDS_MISSING")

    if train_config.get("source_checkpoint_path") != SOURCE_CHECKPOINT or train_config.get("target_checkpoint_path") != TARGET_CHECKPOINT:
        failures.append("TRAIN_CONFIG_CHECKPOINT_POLICY_MISMATCH")
    if train_config.get("family_default_value_bank_used") is not True or train_config.get("positive_can_depend_on_train_loss") is not False:
        failures.append("TRAIN_CONFIG_HARD_NEGATIVE_POLICY_MISSING")
    if eval_config.get("helper_path") != "scripts/probes/shared_raw_generation_helper.py" or eval_config.get("family_default_hard_negative_success_allowed") is not False:
        failures.append("EVAL_CONFIG_HELPER_OR_DEFAULT_POLICY_MISMATCH")
    if training_objective.get("objective_explicitly_penalizes_family_default") is not True:
        failures.append("TRAINING_OBJECTIVE_DEFAULT_PENALTY_MISSING")
    if training_objective.get("positive_can_depend_on_train_loss") is not False:
        failures.append("TRAINING_OBJECTIVE_LOSS_ONLY_OVERCLAIM")

    if positive:
        gates = [aggregate.get(key, 0.0) >= threshold for key, threshold in POSITIVE_MIN.items()]
        gates.extend(aggregate.get(key, 1.0) <= threshold for key, threshold in POSITIVE_MAX.items())
        gates.extend(
            [
                aggregate.get("family_default_shortcut_detected") is False,
                aggregate.get("high_frequency_train_value_replay_detected") is False,
                aggregate.get("parrot_trap_detected") is False,
                controls.get("controls_failed") is True,
                replay.get("determinism_replay_passed") is True,
                evidence.get("reasoning_subtrack_real_raw_evidence_partially_restored") is True,
            ]
        )
        for row in per_seed:
            gates.extend(
                [
                    row.get("answer_value_accuracy", 0.0) >= 0.20,
                    row.get("intra_family_contrastive_accuracy", 0.0) >= 0.25,
                    row.get("intra_family_mode_collapse_rate", 1.0) <= 0.65,
                    row.get("family_default_attractor_rate", 1.0) <= 0.45,
                    row.get("hard_negative_default_violation_rate", 1.0) <= 0.25,
                    row.get("rule_derived_value_accuracy", 0.0) >= 0.15,
                    row.get("table_derived_value_accuracy", 0.0) >= 0.15,
                ]
            )
        if not all(gates):
            failures.append("POSITIVE_GATE_FAILURE")
    else:
        if evidence.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not False:
            failures.append("CLEAN_NEGATIVE_EVIDENCE_FLAG_FAILURE")

    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined_text = "\n".join(files.values())
    docs_text = "\n".join(files.get(rel, "") for rel in DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("UNAUTHORIZED_REPO_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138YK check failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138YK check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
