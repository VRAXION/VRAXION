#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_136R real-raw core rebuild."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_136r_real_raw_core_capability_minimal_rebuild/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_136r_real_raw_core_capability_minimal_rebuild.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_136r_real_raw_core_capability_minimal_rebuild_check.py"
REQUIRED_SOURCE = [RUNNER, CHECKER]
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_RESULT.md",
]
ALLOWED_MUTATIONS = set(REQUIRED_SOURCE + REQUIRED_DOCS)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_135e_manifest.json",
    "eval_config.json",
    "backend_discovery_report.json",
    "forbidden_input_rejection_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "core_raw_eval_rows.jsonl",
    "raw_generation_trace.jsonl",
    "generated_text_hashes.json",
    "core_raw_metrics.json",
    "no_oracle_metadata_report.json",
    "raw_generation_helper_provenance.json",
    "helper_integrity_manifest.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
    "REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE",
    "shared_raw_generation_helper.py",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "generated_text_before_scoring",
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
    "integrated_policy_used_during_final_eval",
    "decoder_reference_used_during_final_eval",
    "oracle_rerank_used",
    "expected_answer_used_during_eval",
    "teacher_forcing_used_during_final_eval",
    "verifier_rerank_used",
    "llm_judge_used",
    "actual_tool_execution_used",
    "runtime_tool_call_used",
    "constrained_decoding_used",
    "json_mode_used",
    "grammar_decoder_used",
    "post_generation_repair_used",
    "retry_loop_used",
    "best_of_n_used",
]
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

    upstream = load_json(SMOKE_ROOT / "upstream_135e_manifest.json")
    if upstream.get("upstream_135e_verified") is not True:
        failures.append("UPSTREAM_135E_NOT_VERIFIED")
    if upstream.get("decision") != "shared_raw_generation_helper_and_canary_ready":
        failures.append("UPSTREAM_135E_DECISION_MISMATCH")
    if upstream.get("next") != "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD":
        failures.append("UPSTREAM_135E_NEXT_MISMATCH")

    config = load_json(SMOKE_ROOT / "eval_config.json")
    if config.get("helper_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    for flag in FINAL_EVAL_FALSE_FLAGS:
        if config.get(flag) is not False:
            failures.append("RAW_FINAL_EVAL_FLAG_FAILURE")

    backend = load_json(SMOKE_ROOT / "backend_discovery_report.json")
    if backend.get("status") != "selected" or not backend.get("selected"):
        failures.append("RAW_GENERATION_BACKEND_MISSING")

    provenance = load_json(SMOKE_ROOT / "raw_generation_helper_provenance.json")
    for key in [
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "requested_checkpoint_hash",
        "model_checkpoint_hash",
        "backend_name",
        "backend_version",
        "torch_available",
        "device",
        "generation_config",
        "generation_config_hash",
        "helper_source_sha256",
        "helper_import_path",
        "backend_load_status",
        "checkpoint_key_count",
        "checkpoint_expected_key_count",
        "checkpoint_extra_keys",
        "checkpoint_missing_keys",
        "checkpoint_shape_summary",
        "strict_load_state_dict",
    ]:
        if key not in provenance or provenance[key] in ("", None):
            failures.append("HELPER_PROVENANCE_INCOMPLETE")
    if provenance.get("helper_import_path") != HELPER:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    if provenance.get("requested_checkpoint_hash") != provenance.get("selected_checkpoint_sha256"):
        failures.append("CHECKPOINT_HASH_MISMATCH")
    if provenance.get("model_checkpoint_hash") != provenance.get("selected_checkpoint_sha256"):
        failures.append("CHECKPOINT_HASH_MISMATCH")
    if provenance.get("checkpoint_extra_keys") != [] or provenance.get("checkpoint_missing_keys") != []:
        failures.append("CHECKPOINT_KEY_SET_MISMATCH")
    if provenance.get("strict_load_state_dict") is not True or provenance.get("backend_load_status") != "strict_load_state_dict_passed":
        failures.append("STRICT_LOAD_STATE_DICT_MISSING")

    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    if forbidden.get("all_rejected") is not True:
        failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
    if len(forbidden.get("rows", [])) < 10 or len(forbidden.get("generation_config_rows", [])) < 3:
        failures.append("FORBIDDEN_INPUT_TEST_TOO_SMALL")

    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    if canary.get("expected_output_canary_passed") is not True:
        failures.append("ORACLE_SHORTCUT_DETECTED")
    if canary.get("original_helper_request_hash") != canary.get("shadow_helper_request_hash"):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    for original, shadow in [
        ("generated_text_original_hash", "generated_text_shadow_hash"),
        ("generation_trace_hash_original", "generation_trace_hash_shadow"),
        ("token_count_original", "token_count_shadow"),
        ("stop_reason_original", "stop_reason_shadow"),
        ("model_checkpoint_hash_original", "model_checkpoint_hash_shadow"),
        ("generation_config_hash_original", "generation_config_hash_shadow"),
    ]:
        if canary.get(original) != canary.get(shadow):
            failures.append("ORACLE_SHORTCUT_DETECTED")
    request_json = f"{canary.get('original_helper_request_json', '')}\n{canary.get('shadow_helper_request_json', '')}"
    for forbidden_name in FORBIDDEN_HELPER_KEYS:
        if forbidden_name in request_json:
            failures.append("CANARY_FORBIDDEN_FIELDS_PRESENT")

    scan = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    if scan.get("ast_shortcut_scan_passed") is not True or scan.get("findings"):
        failures.append("AST_SHORTCUT_SCAN_FAILED")

    rows = read_jsonl(SMOKE_ROOT / "core_raw_eval_rows.jsonl")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    if len(rows) < 8 or len(traces) != len(rows):
        failures.append("CORE_RAW_ROW_COUNT_INVALID")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("RAW_HELPER_REQUEST_SCHEMA_FAILURE")
        if set(request) & FORBIDDEN_HELPER_KEYS:
            failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
        response = trace.get("response", {})
        if not response.get("generated_text") or response.get("token_count", 0) <= 0:
            failures.append("GENERATED_TEXT_MISSING")
        if trace.get("generated_text_before_scoring") is not True:
            failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")

    hashes = load_json(SMOKE_ROOT / "generated_text_hashes.json")
    if hashes.get("all_generated_text_exists") is not True or hashes.get("all_token_counts_positive") is not True:
        failures.append("GENERATED_TEXT_MISSING")

    metrics = load_json(SMOKE_ROOT / "core_raw_metrics.json")
    if metrics.get("semantic_accuracy_not_gate") is not True or metrics.get("semantic_accuracy_is_diagnostic_only") is not True:
        failures.append("SEMANTIC_ACCURACY_USED_AS_GATE")
    if metrics.get("generated_text_before_scoring") is not True:
        failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
    for flag in FINAL_EVAL_FALSE_FLAGS:
        if metrics.get("final_eval_flags", {}).get(flag) is not False:
            failures.append("RAW_FINAL_EVAL_FLAG_FAILURE")

    oracle = load_json(SMOKE_ROOT / "no_oracle_metadata_report.json")
    if oracle.get("forbidden_metadata_reached_helper") is not False:
        failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
    if oracle.get("generated_text_produced_before_scoring") is not True:
        failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "real_raw_core_capability_minimal_rebuild_recorded":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "137R_REAL_RAW_REASONING_REBUILD":
        failures.append("NEXT_MISMATCH")
    if decision.get("shared_raw_generation_helper_used") is not True:
        failures.append("SHARED_RAW_GENERATION_HELPER_NOT_USED")
    if decision.get("expected_output_canary_passed") is not True:
        failures.append("ORACLE_SHORTCUT_DETECTED")
    if decision.get("generated_text_before_scoring") is not True:
        failures.append("GENERATED_TEXT_NOT_BEFORE_SCORING")
    for key in FALSE_CLAIM_KEYS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")

    summary = load_json(SMOKE_ROOT / "summary.json")
    if summary.get("raw_assistant_capability_restored") is not False:
        failures.append("BOUNDARY_CLAIM_FAILURE")
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
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    source_paths = [REPO_ROOT / HELPER, REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]
    failures.extend(ast_scan_source(source_paths))
    failures.extend(check_artifacts())
    if failures:
        print("136R checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("136R checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
