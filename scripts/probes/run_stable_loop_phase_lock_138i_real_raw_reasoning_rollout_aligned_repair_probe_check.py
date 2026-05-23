#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138I rollout-aligned repair probe."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138h_manifest.json",
    "upstream_138ga_manifest.json",
    "upstream_138r_manifest.json",
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
    "freshness_leakage_audit.json",
    "training_metrics.jsonl",
    "training_objective_report.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "scoring_results.jsonl",
    "control_results.jsonl",
    "control_arm_report.json",
    "generated_before_scoring_report.json",
    "namespace_metrics.json",
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
    "STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE",
    "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_POSITIVE",
    "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS",
    "ROLLOUT_ALIGNED_TRAINING_PATH_MISSING",
    "DETERMINISM_REPLAY_MISMATCH",
    "TRAIN_NAMESPACE_REPLAY_CONTROL",
    "namespace_rollout_failure",
    "shared_raw_generation_helper.py",
    "generated_text",
    "expected-output canary",
    "answer_prefix_alone_is_success",
    "checkpoint_change_alone_is_evidence",
    "train_loss_alone_is_evidence",
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
    "eval_family_expected_values",
    "row_answer",
    "target_json",
    "gold_output",
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
NAMESPACE_GATES = {
    "post_train_namespace_leak_rate": 0.05,
    "post_eval_namespace_emission_accuracy": 0.45,
    "post_answer_prefix_accuracy": 0.60,
    "post_answer_value_accuracy": 0.35,
    "post_stale_user_assistant_fragment_rate": 0.10,
    "post_off_prompt_output_rate": 0.30,
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

    train_config = load_json(SMOKE_ROOT / "train_config.json")
    eval_config = load_json(SMOKE_ROOT / "eval_config.json")
    if train_config.get("train_examples") != 60000 or train_config.get("seeds") != [2291, 2292, 2293]:
        failures.append("FULL_CONFIG_NOT_USED")
    if train_config.get("positive_can_depend_on_train_loss") is not False or train_config.get("old_runners_imported") is not False:
        failures.append("LOSS_ONLY_OR_OLD_RUNNER_ALLOWED")
    if "rollout-aligned" not in str(train_config.get("training_objective", "")):
        failures.append("ROLLOUT_ALIGNED_OBJECTIVE_MISSING")
    if eval_config.get("eval_rows_per_family") != 96 or eval_config.get("max_new_tokens") != 96:
        failures.append("FULL_CONFIG_NOT_USED")
    if eval_config.get("helper_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    if eval_config.get("answer_prefix_alone_is_success") is not False or eval_config.get("namespace_metrics_required") is not True:
        failures.append("NAMESPACE_GATE_CONFIG_MISSING")
    for flag in FINAL_EVAL_FALSE_FLAGS:
        if eval_config.get(flag) is not False:
            failures.append("RAW_FINAL_EVAL_FLAG_FAILURE")

    upstream_138h = load_json(SMOKE_ROOT / "upstream_138h_manifest.json")
    if upstream_138h.get("upstream_138h_verified") is not True or upstream_138h.get("next") != "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE":
        failures.append("UPSTREAM_138H_NOT_COMPLETE")
    if upstream_138h.get("primary_bottleneck") != "train_namespace_rollout_alignment_failure":
        failures.append("UPSTREAM_138H_NOT_COMPLETE")
    upstream_138ga = load_json(SMOKE_ROOT / "upstream_138ga_manifest.json")
    if upstream_138ga.get("upstream_138ga_verified") is not True or upstream_138ga.get("primary_label_counts") != {"train_namespace_overlap": 38}:
        failures.append("UPSTREAM_138GA_NOT_DISAMBIGUATED")
    if upstream_138ga.get("near_match_row_count") != 38 or upstream_138ga.get("total_scored_row_count") != 960 or upstream_138ga.get("meaningful_near_match_rate") != 0.0:
        failures.append("UPSTREAM_138GA_NOT_DISAMBIGUATED")
    upstream_138r = load_json(SMOKE_ROOT / "upstream_138r_manifest.json")
    if upstream_138r.get("upstream_138r_verified") is not True or upstream_138r.get("decision") != "teacher_forcing_or_training_objective_failure":
        failures.append("UPSTREAM_138R_NOT_CLEAN_NEGATIVE")
    if upstream_138r.get("mean_real_raw_reasoning_accuracy") != 0.0 or upstream_138r.get("expected_token_inclusion_rate") != 0.0:
        failures.append("UPSTREAM_138R_NOT_CLEAN_NEGATIVE")

    determinism = load_json(SMOKE_ROOT / "determinism_manifest.json")
    for key in ["source_checkpoint_hash", "target_checkpoint_hash", "dataset_hash", "train_config_hash", "eval_config_hash", "helper_source_hash"]:
        if not determinism.get(key):
            failures.append("DETERMINISM_MANIFEST_INCOMPLETE")
    if determinism.get("wall_clock_or_uuid_influences_dataset_train_eval_decision_or_score") is not False:
        failures.append("DETERMINISM_MANIFEST_INCOMPLETE")

    source = load_json(SMOKE_ROOT / "source_checkpoint_integrity_manifest.json")
    target = load_json(SMOKE_ROOT / "target_checkpoint_integrity_manifest.json")
    if source.get("source_checkpoint_unchanged") is not True:
        failures.append("SOURCE_CHECKPOINT_MUTATION_DETECTED")
    if target.get("helper_strict_load_passed") is not True or target.get("target_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_HELPER_COMPATIBLE")
    for manifest in [source, target]:
        if manifest.get("checkpoint_extra_keys") != [] or manifest.get("checkpoint_missing_keys") != []:
            failures.append("CHECKPOINT_KEY_SET_MISMATCH")

    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    if forbidden.get("all_rejected") is not True:
        failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    if canary.get("expected_output_canary_passed") is not True or canary.get("original_helper_request_hash") != canary.get("shadow_helper_request_hash"):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    request_json = f"{canary.get('original_helper_request_json', '')}\n{canary.get('shadow_helper_request_json', '')}"
    for forbidden_name in FORBIDDEN_HELPER_KEYS:
        if forbidden_name in request_json:
            failures.append("CANARY_FORBIDDEN_FIELDS_PRESENT")
    scan = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    if scan.get("ast_shortcut_scan_passed") is not True or scan.get("findings"):
        failures.append("AST_SHORTCUT_SCAN_FAILED")

    train_manifest = load_json(SMOKE_ROOT / "train_dataset_manifest.json")
    eval_manifest = load_json(SMOKE_ROOT / "eval_dataset_manifest.json")
    train_rows = read_jsonl(SMOKE_ROOT / "train_rows.jsonl")
    eval_rows = read_jsonl(SMOKE_ROOT / "eval_rows.jsonl")
    if train_manifest.get("row_count") != 60000 or len(train_rows) != 60000:
        failures.append("TRAIN_DATASET_MANIFEST_INVALID")
    if eval_manifest.get("row_count") != 960 or len(eval_rows) != 960:
        failures.append("EVAL_DATASET_MANIFEST_INVALID")
    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_rejected") is not True:
        failures.append("REASONING_REPAIR_EVAL_LEAKAGE")
    if leakage.get("train_eval_namespaces_disjoint") is not True or leakage.get("train_eval_row_hash_overlap") != 0:
        failures.append("TRAIN_EVAL_NAMESPACE_OR_HASH_LEAKAGE")
    if leakage.get("exact_prompt_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0 or leakage.get("exact_expected_output_overlap") != 0:
        failures.append("TRAIN_EVAL_LEAKAGE")

    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    raw_results = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    scoring = read_jsonl(SMOKE_ROOT / "scoring_results.jsonl")
    if len(traces) != 960 or len(raw_results) != 960 or len(scoring) != 960:
        failures.append("FULL_EVAL_ROW_COUNT_MISMATCH")
    for trace in traces:
        helper_request = trace.get("helper_request", {})
        if set(helper_request) != ALLOWED_HELPER_KEYS:
            failures.append("RAW_HELPER_REQUEST_SCHEMA_FAILURE")
        if set(helper_request) & FORBIDDEN_HELPER_KEYS:
            failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
        if set(trace.get("helper_request_allowed_keys", [])) != ALLOWED_HELPER_KEYS:
            failures.append("RAW_HELPER_REQUEST_SCHEMA_FAILURE")
        if trace.get("generated_before_scoring") is not True:
            failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
        for key in ["row_id", "seed", "helper_request_hash", "generated_text_hash", "generation_trace_hash", "model_checkpoint_hash", "generation_config_hash"]:
            if trace.get(key) in {None, ""}:
                failures.append("RAW_TRACE_FIELD_MISSING")

    before = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    if before.get("generated_text_produced_before_scoring") is not True or before.get("scoring_did_not_feed_back_into_generation") is not True:
        failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
    if before.get("helper_requests_built_without_expected_or_scorer_metadata") is not True:
        failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    if controls.get("controls_called_helper") is not False or controls.get("controls_failed") is not True:
        failures.append("SCORER_OR_TASK_WEAKNESS")
    if "TRAIN_NAMESPACE_REPLAY_CONTROL" not in controls.get("controls", {}):
        failures.append("TRAIN_NAMESPACE_REPLAY_CONTROL_MISSING")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    if replay.get("determinism_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_MISMATCH")

    training_rows = read_jsonl(SMOKE_ROOT / "training_metrics.jsonl")
    training_report = load_json(SMOKE_ROOT / "training_objective_report.json")
    if not training_rows or training_rows[-1].get("train_step_count", 0) <= 0:
        failures.append("TRAINING_METRICS_MISSING")
    if training_report.get("positive_can_depend_on_train_loss") is not False:
        failures.append("LOSS_ONLY_SUCCESS_ALLOWED")
    for key in ["rollout_alignment_metric_initial", "rollout_alignment_metric_final", "namespace_loss_proxy_initial", "namespace_loss_proxy_final", "stale_fragment_penalty_metric_initial", "stale_fragment_penalty_metric_final"]:
        if key not in training_report:
            failures.append("ROLLOUT_ALIGNMENT_PROXY_MISSING")

    namespace = load_json(SMOKE_ROOT / "namespace_metrics.json")
    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    for key, threshold in NAMESPACE_GATES.items():
        metric_source = aggregate if key in {"post_stale_user_assistant_fragment_rate", "post_off_prompt_output_rate"} else namespace
        if key not in metric_source or key not in aggregate:
            failures.append("NAMESPACE_METRIC_MISSING")
        elif key.endswith("_rate") and metric_source[key] > threshold:
            pass
        elif not key.endswith("_rate") and metric_source[key] < threshold:
            pass
    for key in [
        "baseline_train_namespace_leak_rate",
        "post_train_namespace_leak_rate",
        "baseline_eval_namespace_emission_accuracy",
        "post_eval_namespace_emission_accuracy",
        "baseline_answer_value_accuracy",
        "post_answer_value_accuracy",
        "post_answer_prefix_accuracy",
        "post_namespace_accuracy",
        "post_exact_answer_accuracy",
        "train_namespace_leak_rate_reduced",
        "eval_namespace_emission_accuracy_improved",
        "answer_value_accuracy_improved",
        "namespace_gates_passed",
    ]:
        if key not in namespace:
            failures.append("NAMESPACE_METRIC_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    allowed_decisions = {
        "real_raw_reasoning_rollout_aligned_repair_positive": "139R_REAL_RAW_REASONING_REPAIR_SCALE_CONFIRM",
        "no_rollout_improvement": "138I_FAILURE_ANALYSIS",
        "namespace_rollout_failure": "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS",
        "stale_chat_rollout_failure": "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS",
        "nondeterministic_repair_probe": "138N_DETERMINISM_FAILURE_ANALYSIS",
        "reasoning_repair_eval_leakage": "138L_REASONING_REPAIR_EVAL_LEAKAGE_REDESIGN",
        "scorer_or_task_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
        "raw_helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        "rollout_aligned_training_path_missing": "138IA_ROLLOUT_ALIGNED_TRAINING_HELPER_INTEGRATION_PLAN",
        "teacher_forcing_or_training_objective_failure": "138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS",
    }
    if decision.get("decision") not in allowed_decisions or decision.get("next") != allowed_decisions.get(decision.get("decision")):
        failures.append("DECISION_MISMATCH")

    positive_gates = (
        bool(aggregate.get("positive_reasoning_repair_gates_passed"))
        and bool(decision.get("helper_only_rollout_accuracy_improved"))
        and namespace.get("post_train_namespace_leak_rate", 1.0) <= NAMESPACE_GATES["post_train_namespace_leak_rate"]
        and namespace.get("post_eval_namespace_emission_accuracy", 0.0) >= NAMESPACE_GATES["post_eval_namespace_emission_accuracy"]
        and namespace.get("post_answer_prefix_accuracy", 0.0) >= NAMESPACE_GATES["post_answer_prefix_accuracy"]
        and namespace.get("post_answer_value_accuracy", 0.0) >= NAMESPACE_GATES["post_answer_value_accuracy"]
        and aggregate.get("post_stale_user_assistant_fragment_rate", 1.0) <= NAMESPACE_GATES["post_stale_user_assistant_fragment_rate"]
        and aggregate.get("post_off_prompt_output_rate", 1.0) <= NAMESPACE_GATES["post_off_prompt_output_rate"]
        and controls.get("controls_failed") is True
        and leakage.get("leakage_rejected") is True
        and replay.get("determinism_replay_passed") is True
    )
    if decision.get("decision") == "real_raw_reasoning_rollout_aligned_repair_positive":
        if decision.get("verdict") != "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_POSITIVE" or positive_gates is not True:
            failures.append("POSITIVE_WITHOUT_ROLLOUT_NAMESPACE_GATES")
        if decision.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not True:
            failures.append("POSITIVE_WITHOUT_PARTIAL_REASONING_FLAG")
    else:
        if decision.get("verdict") not in {"REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS", "DETERMINISM_REPLAY_MISMATCH", "ROLLOUT_ALIGNED_TRAINING_PATH_MISSING"}:
            failures.append("DECISION_MISMATCH")
        if decision.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")

    for key in FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")
    for key in ["answer_prefix_alone_is_success", "checkpoint_change_alone_is_evidence", "train_loss_alone_is_evidence"]:
        if decision.get(key) is not False:
            failures.append("FAKE_POSITIVE_GUARD_MISSING")
    for key in ["shared_raw_generation_helper_used", "expected_output_canary_passed", "ast_shortcut_scan_passed", "generated_text_produced_before_scoring", "source_checkpoint_unchanged", "target_checkpoint_changed"]:
        if decision.get(key) is not True:
            failures.append("DECISION_GATE_MISSING")

    evidence = load_json(SMOKE_ROOT / "evidence_rebuild_status.json")
    if evidence.get("raw_assistant_capability_restored") is not False or evidence.get("structured_tool_capability_restored") is not False:
        failures.append("BOUNDARY_CLAIM_FAILURE")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    require_false_flags(summary, failures)
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report))

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples:
        failures.append("HUMAN_READABLE_SAMPLES_MISSING")
    for sample in samples[:10]:
        for key in ["row_id", "family", "prompt", "generated_text", "expected_output", "pass", "failure_reason", "namespace_label", "helper_trace_hash"]:
            if key not in sample:
                failures.append("HUMAN_READABLE_SAMPLE_FIELD_MISSING")
                break
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
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138I checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138I checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
