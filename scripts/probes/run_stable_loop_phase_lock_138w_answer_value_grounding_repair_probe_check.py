#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138W answer-value grounding repair probe."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138w_answer_value_grounding_repair_probe/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138w_answer_value_grounding_repair_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138w_answer_value_grounding_repair_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138v_manifest.json",
    "upstream_138s_manifest.json",
    "upstream_138i_manifest.json",
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
    "ood_value_grounding_manifest.json",
    "freshness_leakage_audit.json",
    "training_metrics.jsonl",
    "training_objective_report.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "scoring_results.jsonl",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "post_wrapper_carrier_proxy_report.json",
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
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE",
    "ANSWER_VALUE_GROUNDING_REPAIR_POSITIVE",
    "ANSWER_VALUE_GROUNDING_REPAIR_FAILS",
    "VALUE_GROUNDING_TRAINING_PATH_MISSING",
    "DETERMINISM_REPLAY_MISMATCH",
    "Parrot Trap",
    "PARROT_COPY_CONTROL",
    "post_wrapper_garbage_token_rate",
    "value_after_prefix_accuracy",
    "rule_derived_value_accuracy",
    "table_derived_value_accuracy",
    "ood_symbol_value_accuracy",
    "Wrapper-Induced Amnesia",
    "Residual Signal Carrier",
    "diagnostic_gap",
    "shared_raw_generation_helper.py",
    "generated_text",
    "expected-output canary",
    "If it only copies prompt values, it fails",
    "If it cannot derive held-out values, it fails",
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
    "PARROT_COPY_CONTROL",
}
KNOWN_DECISIONS = {
    "answer_value_grounding_repair_positive",
    "no_value_grounding_improvement",
    "parrot_trap_copy_shortcut_detected",
    "wrapper_success_without_value_grounding_persists",
    "wrapper_induced_amnesia_proxy_failure",
    "stale_chat_rollout_failure",
    "nondeterministic_value_grounding_probe",
    "value_grounding_eval_leakage",
    "scorer_or_task_weakness",
    "raw_helper_integrity_failure",
    "value_grounding_training_path_missing",
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
    for verdict, phrases in FORBIDDEN_CLAIMS.items():
        for phrase in phrases:
            if phrase.lower() in text.lower() and not claim_is_negated(text, phrase):
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
    old_runner_re = re.compile(r"^(run_stable_loop_phase_lock_|run_deck_local_)")
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
    for path in paths:
        if not path.exists():
            failures.append(f"MISSING_AST_SOURCE:{path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    return failures


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 15:
        failures.append("PROGRESS_NOT_REFRESHED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    value_metrics = load_json(SMOKE_ROOT / "value_grounding_metrics.json")
    parrot = load_json(SMOKE_ROOT / "parrot_trap_report.json")
    carrier = load_json(SMOKE_ROOT / "post_wrapper_carrier_proxy_report.json")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    before = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    scan = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    source = load_json(SMOKE_ROOT / "source_checkpoint_integrity_manifest.json")
    target = load_json(SMOKE_ROOT / "target_checkpoint_integrity_manifest.json")
    provenance = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    ood = load_json(SMOKE_ROOT / "ood_value_grounding_manifest.json")
    evidence = load_json(SMOKE_ROOT / "evidence_rebuild_status.json")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    scoring = read_jsonl(SMOKE_ROOT / "scoring_results.jsonl")
    eval_rows = read_jsonl(SMOKE_ROOT / "eval_rows.jsonl")

    if decision.get("decision") not in KNOWN_DECISIONS:
        failures.append("DECISION_MISMATCH")
    if decision.get("verdict") not in {"ANSWER_VALUE_GROUNDING_REPAIR_POSITIVE", "ANSWER_VALUE_GROUNDING_REPAIR_FAILS", "VALUE_GROUNDING_TRAINING_PATH_MISSING", "DETERMINISM_REPLAY_MISMATCH"}:
        failures.append("DECISION_MISMATCH")
    require_false_flags(decision, failures)
    require_false_flags(load_json(SMOKE_ROOT / "summary.json"), failures)
    for key in FINAL_EVAL_FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append("FINAL_EVAL_FORBIDDEN_FLAG_FAILURE")

    if canary.get("expected_output_canary_passed") is not True:
        failures.append("EXPECTED_OUTPUT_CANARY_FAILED")
    if scan.get("ast_shortcut_scan_passed") is not True:
        failures.append("AST_SHORTCUT_SCAN_FAILED")
    if leakage.get("leakage_rejected") is not True:
        failures.append("LEAKAGE_NOT_REJECTED")
    if before.get("generated_text_produced_before_scoring") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILURE")
    if replay.get("determinism_replay_passed") is not True and decision.get("decision") != "nondeterministic_value_grounding_probe":
        failures.append("DETERMINISM_REPLAY_NOT_ENFORCED")
    if source.get("source_checkpoint_unchanged") is not True or provenance.get("source_checkpoint_unchanged") is not True:
        failures.append("SOURCE_CHECKPOINT_MUTATION")
    if target.get("target_checkpoint_changed") is not True or provenance.get("target_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_CHANGED")
    if ood.get("eval_values_held_out_from_train") is not True or ood.get("train_eval_value_namespaces_disjoint") is not True:
        failures.append("OOD_VALUE_GROUNDING_REQUIREMENT_MISSING")

    if controls.get("controls_failed") is not True:
        failures.append("SCORER_CONTROLS_DID_NOT_FAIL")
    if set(controls.get("controls", {})) != CONTROLS:
        failures.append("SCORER_CONTROLS_INCOMPLETE")
    if controls.get("controls_called_helper") is not False:
        failures.append("SCORER_CONTROLS_CALLED_HELPER")

    if len(traces) != len(scoring) or len(scoring) != len(eval_rows):
        failures.append("TRACE_SCORING_ROW_COUNT_MISMATCH")
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
                failures.append("RAW_GENERATION_TRACE_FIELD_MISSING")

    for key in ["answer_value_accuracy", "exact_answer_accuracy", "value_after_prefix_accuracy", "prompt_value_copy_accuracy", "rule_derived_value_accuracy", "table_derived_value_accuracy", "composition_derived_value_accuracy", "ood_symbol_value_accuracy", "prefix_success_value_failure_rate", "eval_namespace_success_value_failure_rate", "no_stale_wrong_value_rate"]:
        if key not in value_metrics or key not in aggregate:
            failures.append("VALUE_METRIC_MISSING")
    for key in ["value_after_prefix_accuracy", "value_position_error_rate", "empty_value_after_prefix_rate", "generic_value_after_prefix_rate", "post_wrapper_garbage_token_rate", "value_emission_latency_mean", "value_emission_latency_p95", "repeated_token_after_prefix_rate"]:
        if key not in carrier:
            failures.append("POST_WRAPPER_PROXY_MISSING")
    if carrier.get("hidden_state_residual_signal_measurement") != "diagnostic_gap":
        failures.append("HIDDEN_STATE_OVERCLAIM")
    if parrot.get("parrot_trap_detected") != aggregate.get("parrot_trap_detected"):
        failures.append("PARROT_TRAP_REPORT_MISMATCH")
    if aggregate.get("parrot_trap_detected") and decision.get("decision") == "answer_value_grounding_repair_positive":
        failures.append("PARROT_TRAP_ALLOWED_POSITIVE")

    positive = decision.get("decision") == "answer_value_grounding_repair_positive"
    if positive:
        gates = [
            aggregate.get("answer_value_accuracy", 0.0) >= 0.25,
            aggregate.get("exact_answer_accuracy", 0.0) >= 0.20,
            aggregate.get("value_after_prefix_accuracy", 0.0) >= 0.25,
            aggregate.get("rule_derived_value_accuracy", 0.0) >= 0.20,
            aggregate.get("table_derived_value_accuracy", 0.0) >= 0.20,
            aggregate.get("post_wrapper_garbage_token_rate", 1.0) <= 0.20,
            aggregate.get("parrot_trap_detected") is False,
            replay.get("determinism_replay_passed") is True,
            evidence.get("reasoning_subtrack_real_raw_evidence_partially_restored") is True,
        ]
        if not all(gates):
            failures.append("POSITIVE_GATE_FAILURE")
    else:
        if evidence.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not False:
            failures.append("CLEAN_NEGATIVE_EVIDENCE_FLAG_FAILURE")

    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(json.dumps(load_json(SMOKE_ROOT / "summary.json")) + "\n" + report))
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
        print("138W checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138W checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
