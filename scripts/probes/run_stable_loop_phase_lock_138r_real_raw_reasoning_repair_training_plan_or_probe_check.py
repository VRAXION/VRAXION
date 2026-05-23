#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138R real-raw reasoning repair probe."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_137b_manifest.json",
    "upstream_137r_manifest.json",
    "upstream_136r_manifest.json",
    "upstream_135e_manifest.json",
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
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "scoring_results.jsonl",
    "control_results.jsonl",
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
    "STABLE_LOOP_PHASE_LOCK_138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
    "REAL_RAW_REASONING_REPAIR_PROBE_POSITIVE",
    "REAL_RAW_REASONING_REPAIR_PROBE_FAILS",
    "REAL_RAW_REASONING_TRAINING_HELPER_MISSING",
    "DETERMINISM_REPLAY_MISMATCH",
    "shared_raw_generation_helper.py",
    "generated_text",
    "expected-output canary",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
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
SEED_GATES = {
    "real_raw_reasoning_accuracy": 0.45,
    "provided_fact_qa_accuracy": 0.60,
    "single_step_reasoning_accuracy": 0.55,
    "two_step_reasoning_accuracy": 0.40,
    "rule_chaining_accuracy": 0.40,
    "table_rule_reasoning_accuracy": 0.40,
    "small_arithmetic_accuracy": 0.40,
    "contradiction_resolution_accuracy": 0.35,
    "hallucination_trap_pass_rate": 0.50,
}
MAX_RATE_GATES = {"stale_chat_fragment_rate": 0.10, "off_prompt_output_rate": 0.30}
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
        if not line.strip():
            continue
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


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    train_config = load_json(SMOKE_ROOT / "train_config.json")
    eval_config = load_json(SMOKE_ROOT / "eval_config.json")
    if train_config.get("train_examples") != 60000 or train_config.get("seeds") != [2281, 2282, 2283]:
        failures.append("FULL_CONFIG_NOT_USED")
    if eval_config.get("eval_rows_per_family") != 96 or eval_config.get("max_new_tokens") != 96:
        failures.append("FULL_CONFIG_NOT_USED")
    if eval_config.get("helper_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    for flag in FINAL_EVAL_FALSE_FLAGS:
        if eval_config.get(flag) is not False:
            failures.append("RAW_FINAL_EVAL_FLAG_FAILURE")

    upstream_137b = load_json(SMOKE_ROOT / "upstream_137b_manifest.json")
    if upstream_137b.get("upstream_137b_verified") is not True or upstream_137b.get("next") != "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE":
        failures.append("UPSTREAM_137B_NOT_COMPLETE")
    upstream_137r = load_json(SMOKE_ROOT / "upstream_137r_manifest.json")
    if upstream_137r.get("upstream_137r_clean_negative_verified") is not True or upstream_137r.get("mean_real_raw_reasoning_accuracy") != 0.0:
        failures.append("UPSTREAM_137R_NOT_CLEAN_NEGATIVE")
    upstream_136r = load_json(SMOKE_ROOT / "upstream_136r_manifest.json")
    if upstream_136r.get("upstream_136r_verified") is not True or upstream_136r.get("upstream_135e_verified") is not True:
        failures.append("UPSTREAM_CHAIN_NOT_VERIFIED")

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
    if target.get("helper_strict_load_passed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_HELPER_COMPATIBLE")
    if target.get("target_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_WRITTEN")
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

    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    raw_results = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    scoring = read_jsonl(SMOKE_ROOT / "scoring_results.jsonl")
    if len(traces) != 960 or len(raw_results) != 960 or len(scoring) != 960:
        failures.append("FULL_EVAL_ROW_COUNT_MISMATCH")
    for trace in traces:
        if set(trace.get("helper_request", {})) != ALLOWED_HELPER_KEYS:
            failures.append("RAW_HELPER_REQUEST_SCHEMA_FAILURE")
        if set(trace.get("helper_request", {})) & FORBIDDEN_HELPER_KEYS:
            failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
        if set(trace.get("helper_request_allowed_keys", [])) != ALLOWED_HELPER_KEYS:
            failures.append("RAW_HELPER_REQUEST_SCHEMA_FAILURE")
        if trace.get("generated_before_scoring") is not True:
            failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
        for key in ["helper_request_hash", "generated_text_hash", "generation_trace_hash", "model_checkpoint_hash", "generation_config_hash"]:
            if not trace.get(key):
                failures.append("RAW_TRACE_FIELD_MISSING")

    before = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    if before.get("generated_text_produced_before_scoring") is not True or before.get("scoring_did_not_feed_back_into_generation") is not True:
        failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    if controls.get("controls_called_helper") is not False or controls.get("controls_failed") is not True:
        failures.append("SCORER_OR_TASK_WEAKNESS")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    if replay.get("determinism_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_MISMATCH")

    training_rows = read_jsonl(SMOKE_ROOT / "training_metrics.jsonl")
    if not training_rows or training_rows[-1].get("train_step_count", 0) <= 0:
        failures.append("TRAINING_METRICS_MISSING")
    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    seed_rows = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")
    if len(seed_rows) != 3:
        failures.append("PER_SEED_METRICS_MISSING")
    per_seed_positive = all(
        all(row.get(key, 0.0) >= threshold for key, threshold in SEED_GATES.items())
        and all(row.get(key, 1.0) <= threshold for key, threshold in MAX_RATE_GATES.items())
        for row in seed_rows
    )
    positive_gates = (
        bool(aggregate.get("positive_reasoning_repair_gates_passed"))
        and bool(aggregate.get("all_seeds_passed_independently"))
        and aggregate.get("mean_real_raw_reasoning_accuracy", 0.0) >= 0.50
        and aggregate.get("expected_token_inclusion_rate", 0.0) >= 0.50
        and aggregate.get("near_match_rate", 0.0) >= 0.50
        and per_seed_positive
    )

    decision = load_json(SMOKE_ROOT / "decision.json")
    allowed_decisions = {
        "real_raw_reasoning_repair_probe_positive": "139R_REAL_RAW_REASONING_REPAIR_SCALE_CONFIRM",
        "real_raw_reasoning_repair_probe_failed": "138B_REAL_RAW_REASONING_REPAIR_FAILURE_ANALYSIS",
        "teacher_forcing_or_training_objective_failure": "138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS",
        "stale_chat_rollout_failure": "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS",
        "nondeterministic_repair_probe": "138N_DETERMINISM_FAILURE_ANALYSIS",
        "reasoning_repair_eval_leakage": "138L_REASONING_REPAIR_EVAL_LEAKAGE_REDESIGN",
        "scorer_or_task_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
    }
    if decision.get("decision") not in allowed_decisions or decision.get("next") != allowed_decisions.get(decision.get("decision")):
        failures.append("DECISION_MISMATCH")
    if decision.get("decision") == "real_raw_reasoning_repair_probe_positive":
        if decision.get("verdict") != "REAL_RAW_REASONING_REPAIR_PROBE_POSITIVE" or positive_gates is not True or decision.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not True:
            failures.append("POSITIVE_WITHOUT_REASONING_GATES")
    else:
        if decision.get("verdict") not in {"REAL_RAW_REASONING_REPAIR_PROBE_FAILS", "DETERMINISM_REPLAY_MISMATCH"}:
            failures.append("DECISION_MISMATCH")
        if decision.get("reasoning_subtrack_real_raw_evidence_partially_restored") is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")
    for key in FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")
    for key in ["shared_raw_generation_helper_used", "expected_output_canary_passed", "ast_shortcut_scan_passed", "generated_text_produced_before_scoring", "source_checkpoint_unchanged", "target_checkpoint_changed"]:
        if decision.get(key) is not True:
            failures.append("DECISION_GATE_MISSING")

    evidence = load_json(SMOKE_ROOT / "evidence_rebuild_status.json")
    if evidence.get("raw_assistant_capability_restored") is not False or evidence.get("structured_tool_capability_restored") is not False:
        failures.append("BOUNDARY_CLAIM_FAILURE")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    if summary.get("raw_assistant_capability_restored") is not False:
        failures.append("BOUNDARY_CLAIM_FAILURE")
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
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138R checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138R checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
