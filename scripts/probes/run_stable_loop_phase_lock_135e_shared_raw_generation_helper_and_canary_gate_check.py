#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_135E."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_135e_shared_raw_generation_helper_and_canary_gate/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_135e_shared_raw_generation_helper_and_canary_gate.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_135e_shared_raw_generation_helper_and_canary_gate_check.py"
REQUIRED_SOURCE = [HELPER, RUNNER, CHECKER]
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_RESULT.md",
]
ALLOWED_MUTATIONS = set(REQUIRED_SOURCE + REQUIRED_DOCS)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_135d_manifest.json",
    "helper_config.json",
    "raw_generation_helper_contract.json",
    "raw_generation_helper_provenance.json",
    "backend_discovery_report.json",
    "forbidden_input_rejection_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "raw_generation_smoke_rows.jsonl",
    "raw_generation_trace.jsonl",
    "generated_text_hashes.json",
    "helper_integrity_manifest.json",
    "future_milestone_requirements.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE",
    "SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE",
    "RAW_GENERATION_BACKEND_MISSING",
    "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "AST_SHORTCUT_SCAN_FAILED",
    "shared_raw_generation_helper_and_canary_ready",
    "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "remains quarantined", "remains invalidated"]
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

        Scanner().visit(tree)
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    upstream = load_json(SMOKE_ROOT / "upstream_135d_manifest.json")
    if upstream.get("upstream_135d_verified") is not True:
        failures.append("UPSTREAM_135D_NOT_VERIFIED")
    if upstream.get("decision") != "global_raw_evidence_rebuild_plan_complete":
        failures.append("UPSTREAM_135D_DECISION_MISMATCH")

    contract = load_json(SMOKE_ROOT / "raw_generation_helper_contract.json")
    allowed = set(contract.get("allowed_request_keys", []))
    if allowed != {"prompt", "checkpoint_hash", "checkpoint_path", "seed", "max_new_tokens", "generation_config"}:
        failures.append("STRICT_REQUEST_SCHEMA_MISSING")
    if contract.get("unknown_fields_rejected") is not True:
        failures.append("STRICT_REQUEST_SCHEMA_MISSING")
    for forbidden in ["expected_output", "expected_payload", "expected_answer", "target_json", "row_answer", "gold_output", "eval_family", "answer"]:
        if forbidden not in contract.get("forbidden_request_keys", []):
            failures.append("STRICT_REQUEST_SCHEMA_MISSING")

    backend = load_json(SMOKE_ROOT / "backend_discovery_report.json")
    if backend.get("status") != "selected" or not backend.get("selected"):
        failures.append("RAW_GENERATION_BACKEND_MISSING")

    provenance = load_json(SMOKE_ROOT / "raw_generation_helper_provenance.json")
    required_provenance = [
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "model_checkpoint_hash",
        "backend_name",
        "backend_version",
        "torch_available",
        "device",
        "generation_config_hash",
        "helper_source_sha256",
    ]
    for key in required_provenance:
        if key not in provenance or provenance[key] in ("", None):
            failures.append("HELPER_PROVENANCE_INCOMPLETE")
    for key, expected in [
        ("real_raw_generation_backend_used", True),
        ("raw_generation_backend_missing", False),
        ("fake_helper_used", False),
        ("simulated_model_output_used", False),
    ]:
        if provenance.get(key) != expected:
            failures.append("HELPER_PROVENANCE_INCOMPLETE")

    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    if forbidden.get("all_rejected") is not True:
        failures.append("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED")
    if len(forbidden.get("rows", [])) < 10:
        failures.append("FORBIDDEN_INPUT_TEST_TOO_SMALL")

    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    if canary.get("expected_output_canary_passed") is not True:
        failures.append("ORACLE_SHORTCUT_DETECTED")
    if canary.get("prompt_identical") is not True or canary.get("helper_requests_identical") is not True:
        failures.append("CANARY_REQUEST_MISMATCH")
    if canary.get("forbidden_fields_absent_from_helper_requests") is not True:
        failures.append("CANARY_FORBIDDEN_FIELDS_PRESENT")
    for key in ["generated_text", "generation_trace_hash", "token_count", "stop_reason"]:
        if canary.get("generation_side_fields_identical", {}).get(key) is not True:
            failures.append("ORACLE_SHORTCUT_DETECTED")
    for request_key in list(canary.get("helper_request_original", {})) + list(canary.get("helper_request_shadow", {})):
        if request_key not in allowed:
            failures.append("CANARY_FORBIDDEN_FIELDS_PRESENT")

    scan = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    if scan.get("ast_shortcut_scan_passed") is not True or scan.get("findings"):
        failures.append("AST_SHORTCUT_SCAN_FAILED")
    if scan.get("ast_scan_used") is not True:
        failures.append("AST_SCAN_NOT_USED")

    smoke_rows = read_jsonl(SMOKE_ROOT / "raw_generation_smoke_rows.jsonl")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    if not (8 <= len(smoke_rows) <= 32) or len(traces) != len(smoke_rows):
        failures.append("SMOKE_ROW_COUNT_INVALID")
    for trace in traces:
        if trace.get("generated_text_exists") is not True:
            failures.append("GENERATED_TEXT_MISSING")
        request = trace.get("helper_request", {})
        if set(request) != allowed:
            failures.append("STRICT_REQUEST_SCHEMA_MISSING")
        response = trace.get("response", {})
        for key in ["generated_text", "token_count", "stop_reason", "generation_trace_hash", "model_checkpoint_hash", "generation_config_hash", "helper_backend", "helper_version"]:
            if key not in response:
                failures.append("RAW_RESPONSE_FIELD_MISSING")

    hashes = load_json(SMOKE_ROOT / "generated_text_hashes.json")
    if hashes.get("all_generated_text_exists") is not True:
        failures.append("GENERATED_TEXT_MISSING")

    future = load_json(SMOKE_ROOT / "future_milestone_requirements.json")
    for milestone in [
        "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
        "137R_REAL_RAW_REASONING_REBUILD",
        "138R_REAL_RAW_MULTI_TURN_STATE_REBUILD",
        "139R_REAL_RAW_HALLUCINATION_REFUSAL_REBUILD",
        "140R_REAL_RAW_INJECTION_PRIORITY_REBUILD",
        "141R_REAL_RAW_STRUCTURED_TOOL_REBUILD",
        "142R_REAL_RAW_CEILING_AND_GAP_REMAP",
    ]:
        if milestone not in future.get("applies_to", []):
            failures.append("FUTURE_MILESTONE_LOCK_INCOMPLETE")
    if future.get("no_future_raw_capability_milestone_may_pass_without_these") is not True:
        failures.append("FUTURE_MILESTONE_LOCK_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "shared_raw_generation_helper_and_canary_ready":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD":
        failures.append("NEXT_MISMATCH")
    for key in [
        "upstream_135d_verified",
        "shared_raw_generation_helper_written",
        "backend_discovery_complete",
        "real_raw_generation_backend_used",
        "forbidden_input_rejection_passed",
        "expected_output_canary_passed",
        "ast_shortcut_scan_passed",
        "generated_text_exists",
        "generation_trace_hash_written",
        "model_checkpoint_hash_written",
        "generation_config_hash_written",
        "helper_provenance_written",
    ]:
        if decision.get(key) is not True:
            failures.append("HARD_GATE_NOT_PASSED")
    for key, expected in [
        ("raw_generation_backend_missing", False),
        ("fake_helper_used", False),
        ("simulated_model_output_used", False),
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("root_license_changed", False),
        ("raw_assistant_capability_restored", False),
        ("structured_tool_capability_restored", False),
        ("gpt_like_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if decision.get(key) != expected:
            failures.append("SIDE_EFFECT_OR_OVERCLAIM_DETECTED")

    summary = load_json(SMOKE_ROOT / "summary.json")
    verdicts = set(summary.get("verdicts", []))
    for verdict in [
        "SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE",
        "REAL_REPO_LOCAL_RAW_BACKEND_USED",
        "FORBIDDEN_INPUT_REJECTION_PASSED",
        "EXPECTED_OUTPUT_CANARY_PASSED",
        "AST_SHORTCUT_SCAN_PASSED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
        "NO_CAPABILITY_RESTORED",
    ]:
        if verdict not in verdicts:
            failures.append(f"VERDICT_MISSING:{verdict}")
    for key, expected in [
        ("infrastructure_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("root_license_changed", False),
        ("old_runners_modified", False),
        ("raw_assistant_capability_restored", False),
        ("structured_tool_capability_restored", False),
        ("gpt_like_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("SIDE_EFFECT_OR_OVERCLAIM_DETECTED")

    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in [
        "startup",
        "upstream_verification",
        "helper_import",
        "backend_discovery",
        "forbidden_input_rejection_tests",
        "smoke_row_build",
        "canary_build",
        "raw_generation_smoke",
        "AST scan",
        "decision_writing",
        "final_verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    for phrase in [
        "135E establishes raw generation infrastructure only",
        "Raw assistant capability remains quarantined",
        "Structured/tool capability remains invalidated",
        "No capability restored",
        "Not GPT-like readiness",
        "Not open-domain assistant readiness",
        "Not production chat",
        "Not public API",
        "Not deployment readiness",
        "Not safety alignment",
    ]:
        if phrase not in text_to_scan:
            failures.append("BOUNDARY_TEXT_MISSING")
    failures.extend(find_false_claims(text_to_scan))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_REQUIRED_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(ast_scan_source([REPO_ROOT / HELPER, REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if args.check_only:
        failures.extend(check_artifacts())
    if failures:
        print("135E checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("135E checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
