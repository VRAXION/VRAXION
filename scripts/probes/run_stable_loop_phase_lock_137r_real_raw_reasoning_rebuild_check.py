#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_137R real-raw reasoning rebuild."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_137r_real_raw_reasoning_rebuild/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_137r_real_raw_reasoning_rebuild.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_137r_real_raw_reasoning_rebuild_check.py"
REQUIRED_SOURCE = [RUNNER, CHECKER]
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD_RESULT.md",
]
ALLOWED_MUTATIONS = set(REQUIRED_SOURCE + REQUIRED_DOCS)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_136r_manifest.json",
    "upstream_135e_manifest.json",
    "upstream_135d_manifest.json",
    "helper_provenance_verification.json",
    "forbidden_input_rejection_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "reasoning_eval_config.json",
    "reasoning_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "scoring_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "aggregate_metrics.json",
    "generated_before_scoring_report.json",
    "evidence_rebuild_status.json",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD",
    "REAL_RAW_REASONING_REBUILD_POSITIVE",
    "REAL_RAW_REASONING_REBUILD_FAILS",
    "shared_raw_generation_helper.py",
    "expected-output canary",
    "generated_text",
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
FALSE_CLAIM_KEYS = [
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
    "real_raw_reasoning_accuracy": 0.70,
    "single_step_reasoning_accuracy": 0.80,
    "two_step_reasoning_accuracy": 0.70,
    "rule_chaining_accuracy": 0.70,
    "table_rule_reasoning_accuracy": 0.70,
    "small_arithmetic_accuracy": 0.70,
    "contradiction_resolution_accuracy": 0.65,
    "hallucination_trap_pass_rate": 0.70,
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
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


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
    for rel in REQUIRED_SOURCE + REQUIRED_DOCS:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in REQUIRED_DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def ast_scan_source(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    old_runner_re = re.compile(r"^(run_stable_loop_phase_lock_|run_deck_local_)")
    forbidden_call_names = {
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Scanner(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []

            def in_generation_context(self) -> bool:
                return any("generate" in name or "raw_" in name for name in self.function_stack)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if old_runner_re.match(alias.name):
                        failures.append("OLD_RUNNER_IMPORT_DETECTED")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                if old_runner_re.match(module):
                    failures.append("OLD_RUNNER_IMPORT_DETECTED")
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name.lower())
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Assign(self, node: ast.Assign) -> None:
                targets = " ".join(ast.unparse(target) for target in node.targets)
                value = ast.unparse(node.value)
                if re.search(r"generated_text|generated", targets) and any(token in value for token in ["expected_output", "expected_payload", "expected_answer", "gold_output", "target_json"]):
                    failures.append("AST_GENERATED_TEXT_FROM_EXPECTED_MATERIAL")
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                text = ast.unparse(node.value) if node.value is not None else ""
                if self.in_generation_context() and any(token in text for token in ["expected_output", "expected_payload", "expected_answer"]):
                    failures.append("AST_EXPECTED_OUTPUT_IN_GENERATION_PATH")
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = ast.unparse(node.func).lower()
                if any(token in name for token in forbidden_call_names):
                    failures.append("AST_SHORTCUT_SCAN_FAILED")
                self.generic_visit(node)

        Scanner().visit(tree)
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    upstream_136r = load_json(SMOKE_ROOT / "upstream_136r_manifest.json")
    upstream_135e = load_json(SMOKE_ROOT / "upstream_135e_manifest.json")
    upstream_135d = load_json(SMOKE_ROOT / "upstream_135d_manifest.json")
    if upstream_136r.get("verdict") != "REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE":
        failures.append("UPSTREAM_136R_NOT_POSITIVE")
    if upstream_135e.get("verdict") != "SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE":
        failures.append("UPSTREAM_135E_NOT_POSITIVE")
    if upstream_135d.get("decision") != "global_raw_evidence_rebuild_plan_complete":
        failures.append("UPSTREAM_135D_NOT_POSITIVE")

    config = load_json(SMOKE_ROOT / "reasoning_eval_config.json")
    if config.get("helper_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    if config.get("eval_rows_per_family") != 96:
        failures.append("FULL_CONFIG_NOT_USED")
    for flag in FINAL_EVAL_FALSE_FLAGS:
        if config.get(flag) is not False:
            failures.append("RAW_FINAL_EVAL_FLAG_FAILURE")

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

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_rejected") is not True:
        failures.append("REASONING_EVAL_LEAKAGE")

    rows = read_jsonl(SMOKE_ROOT / "reasoning_dataset.jsonl")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    raw_results = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    scoring = read_jsonl(SMOKE_ROOT / "scoring_results.jsonl")
    if len(rows) != 960 or len(traces) != len(rows) or len(raw_results) != len(rows) or len(scoring) != len(rows):
        failures.append("FULL_REASONING_ROW_COUNT_MISMATCH")
    for trace in traces[: len(traces)]:
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

    provenance = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    if provenance.get("helper_import_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    if provenance.get("checkpoint_hash_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if provenance.get("requested_checkpoint_hash") != provenance.get("selected_checkpoint_sha256"):
        failures.append("CHECKPOINT_HASH_MISMATCH")
    if provenance.get("checkpoint_extra_keys") != [] or provenance.get("checkpoint_missing_keys") != []:
        failures.append("CHECKPOINT_KEY_SET_MISMATCH")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    seed_rows = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")
    if len(seed_rows) != 5:
        failures.append("PER_SEED_METRICS_MISSING")
    per_seed_positive = all(all(row.get(key, 0.0) >= threshold for key, threshold in SEED_GATES.items()) for row in seed_rows)
    positive_gates = bool(aggregate.get("positive_reasoning_gates_passed")) and bool(aggregate.get("all_seeds_passed_independently")) and aggregate.get("mean_real_raw_reasoning_accuracy", 0.0) >= 0.75 and per_seed_positive

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("verdict") == "REAL_RAW_REASONING_REBUILD_POSITIVE":
        if decision.get("decision") != "real_raw_reasoning_evidence_rebuilt" or decision.get("next") != "138R_REAL_RAW_MULTI_TURN_STATE_REBUILD":
            failures.append("DECISION_MISMATCH")
        if positive_gates is not True or decision.get("reasoning_subtrack_real_raw_evidence_restored") is not True:
            failures.append("POSITIVE_WITHOUT_REASONING_GATES")
    elif decision.get("verdict") == "REAL_RAW_REASONING_REBUILD_FAILS":
        if decision.get("decision") != "real_raw_reasoning_not_restored" or decision.get("next") != "137B_REAL_RAW_REASONING_REPAIR_PLAN":
            failures.append("DECISION_MISMATCH")
        if positive_gates is True or decision.get("reasoning_subtrack_real_raw_evidence_restored") is not False:
            failures.append("NEGATIVE_DESPITE_PASSING_REASONING_GATES")
    else:
        failures.append("DECISION_MISMATCH")
    for key in FALSE_CLAIM_KEYS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")

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
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / HELPER, REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("137R checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("137R checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
