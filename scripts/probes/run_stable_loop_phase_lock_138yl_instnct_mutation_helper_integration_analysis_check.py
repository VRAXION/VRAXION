#!/usr/bin/env python3
"""Checker for 138YL INSTNCT mutation helper integration analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yl_instnct_mutation_helper_integration_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yl_instnct_mutation_helper_integration_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yl_instnct_mutation_helper_integration_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YL_INSTNCT_MUTATION_HELPER_INTEGRATION_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YL_INSTNCT_MUTATION_HELPER_INTEGRATION_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_138yk_manifest.json",
    "mutation_microprobe_manifest.json",
    "phase_lock_004_manifest.json",
    "helper_backend_audit.json",
    "instnct_mutation_surface_map.json",
    "generator_contract_gap_analysis.json",
    "gradient_vs_mutation_credit_report.json",
    "external_research_manifest.json",
    "integration_options.json",
    "integration_risk_register.json",
    "target_138ym_milestone_plan.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "reasoning_subtrack_real_raw_evidence_partially_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
]
FORBIDDEN_IMPORTS = {"torch", "shared_raw_generation_helper"}
FORBIDDEN_CALL_TOKENS = {
    "raw_generate",
    "load_checkpoint",
    "train",
    "fit",
    "backward",
    "optimizer",
    "manual_seed",
    "forward",
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "quarantined", "invalidated"]
FORBIDDEN_CLAIMS = {
    "RAW_ASSISTANT_CAPABILITY_RESTORED_FALSE_CLAIM": ["raw assistant capability restored"],
    "STRUCTURED_TOOL_CAPABILITY_RESTORED_FALSE_CLAIM": ["structured/tool capability restored", "structured tool capability restored"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like readiness"],
    "OPEN_DOMAIN_ASSISTANT_FALSE_CLAIM": ["open-domain assistant ready", "open-domain assistant readiness"],
    "PRODUCTION_CHAT_FALSE_CLAIM": ["production chat ready", "production readiness"],
    "PUBLIC_API_FALSE_CLAIM": ["public API ready", "public API readiness"],
    "DEPLOYMENT_FALSE_CLAIM": ["deployment ready", "deployment readiness"],
    "SAFETY_ALIGNMENT_FALSE_CLAIM": ["safety aligned", "safety alignment"],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    out: list[str] = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        out.append(line[3:].replace("\\", "/"))
    return out


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 220) : match.start()]
        after = lowered[match.end() : min(len(lowered), match.end() + 80)]
        if any(marker in window for marker in NEGATION_MARKERS) or any(marker in after for marker in [" false", " not", " no"]):
            return True
    return False


def find_false_claims(text: str) -> list[str]:
    failures: list[str] = []
    for code, phrases in FORBIDDEN_CLAIMS.items():
        for phrase in phrases:
            if phrase.lower() in text.lower() and not claim_is_negated(text, phrase):
                failures.append(code)
                break
    return failures


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_FLAG_NOT_FALSE:{key}")


def ast_scan(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    failures.append(f"FORBIDDEN_IMPORT:{alias.name}:{path.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_IMPORTS or module.startswith("run_stable_loop_phase_lock_"):
                failures.append(f"FORBIDDEN_IMPORT_FROM:{module}:{path.name}")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(token in name for token in FORBIDDEN_CALL_TOKENS):
                if name not in {"append_progress"}:
                    failures.append(f"FORBIDDEN_CALL:{name}:{path.name}")
    return failures


def check_changed_files() -> list[str]:
    failures: list[str] = []
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")
    return failures


def check_static_files() -> list[str]:
    failures: list[str] = []
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if rel in DOCS and len(text.strip()) < 400:
            failures.append(f"DOC_TOO_SHORT:{rel}")
        failures.extend(find_false_claims(text))
    failures.extend(ast_scan(REPO_ROOT / RUNNER))
    failures.extend(ast_scan(REPO_ROOT / CHECKER))
    return sorted(set(failures))


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    config = load_json(SMOKE_ROOT / "analysis_config.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    helper = load_json(SMOKE_ROOT / "helper_backend_audit.json")
    gaps = load_json(SMOKE_ROOT / "generator_contract_gap_analysis.json")
    mutation = load_json(SMOKE_ROOT / "mutation_microprobe_manifest.json")
    phase004 = load_json(SMOKE_ROOT / "phase_lock_004_manifest.json")
    upstream = load_json(SMOKE_ROOT / "upstream_138yk_manifest.json")
    plan = load_json(SMOKE_ROOT / "target_138ym_milestone_plan.json")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_ARTIFACT_ONLY")
    for key in [
        "training_performed",
        "new_helper_inference_run",
        "shared_helper_called",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_backend_modified",
    ]:
        if config.get(key) is not False:
            failures.append(f"ARTIFACT_ONLY_BOUNDARY_FAILURE:{key}")

    if decision.get("decision") != "instnct_mutation_helper_integration_analysis_complete":
        failures.append("BAD_DECISION")
    if decision.get("next") != "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN":
        failures.append("BAD_NEXT")
    if decision.get("instnct_helper_adapter_required") is not True:
        failures.append("ADAPTER_REQUIRED_NOT_TRUE")
    if decision.get("real_raw_value_grounding_comparison_ready") is not False:
        failures.append("COMPARISON_READY_TOO_EARLY")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)

    if helper.get("helper_backend") != "repo_local_checkpoint_byte_lm":
        failures.append("HELPER_BACKEND_NOT_BYTE_LM")
    if helper.get("adapter_required_for_instnct_raw_generation") is not True:
        failures.append("HELPER_ADAPTER_GAP_NOT_DETECTED")
    if gaps.get("blocking_gap_count", 0) < 3:
        failures.append("CONTRACT_GAPS_TOO_WEAK")
    if "HIGHWAY_POCKET_MUTATION_POSITIVE" not in mutation.get("verdicts", []):
        failures.append("MUTATION_TOY_SIGNAL_MISSING")
    if phase004.get("phase_credit_assignment_not_solved") is not True:
        failures.append("PHASE004_NEGATIVE_NOT_PRESERVED")
    if upstream.get("decision") != "family_default_shortcut_persists":
        failures.append("UPSTREAM_138YK_DECISION_NOT_PRESERVED")
    if upstream.get("source_backend_name") != "byte_gru_lm":
        failures.append("UPSTREAM_138YK_BACKEND_NOT_BYTE_GRU")

    if plan.get("milestone") != "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN":
        failures.append("TARGET_PLAN_BAD_MILESTONE")
    for term in [
        "adapter_contract.json",
        "instnct_checkpoint_contract.json",
        "prompt_encoder_contract.json",
        "iterative_propagation_schedule.json",
        "output_decoder_contract.json",
        "determinism_plan.json",
    ]:
        if term not in plan.get("required_artifacts", []):
            failures.append(f"TARGET_PLAN_MISSING:{term}")

    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "source surface audit",
        "contract gap analysis",
        "credit mechanism analysis",
        "integration option selection",
        "target 138YM plan writing",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    for term in [
        "artifact-only",
        "byte-GRU",
        "repo_local_checkpoint_byte_lm",
        "138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN",
        "Raw assistant capability remains quarantined",
        "not GPT-like readiness",
    ]:
        if term not in report_text:
            failures.append(f"REPORT_TERM_MISSING:{term}")
    failures.extend(find_false_claims(report_text))
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YL integration analysis")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {
        "schema_version": "phase_138yl_checker_result_v1",
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
