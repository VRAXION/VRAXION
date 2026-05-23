#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138G objective failure analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138r_manifest.json",
    "objective_failure_config.json",
    "teacher_forcing_vs_rollout_report.json",
    "rollout_output_pattern_report.json",
    "train_eval_answer_namespace_report.json",
    "prompt_answer_alignment_report.json",
    "first_mismatch_report.json",
    "stop_behavior_report.json",
    "scoring_strictness_recheck.json",
    "checkpoint_objective_gap_report.json",
    "diagnostic_gap_register.json",
    "objective_failure_root_cause.json",
    "next_objective_redesign_requirements.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS",
    "artifact-only analysis",
    "diagnostic_gap",
    "objective_failure_ambiguous",
    "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN",
    "no new inference",
    "shared_raw_generation_helper.py",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
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
TAG_TYPES = {"artifact_observed", "computed_from_artifact", "diagnostic_gap", "inference"}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "remains quarantined", "remains invalidated", "artifact-only"]
FORBIDDEN_CLAIMS = {
    "REASONING_RESTORED_FALSE_CLAIM": ["reasoning restored", "reasoning is restored"],
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
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    forbidden_calls = {
        "raw_generate",
        "discover_backend",
        "load_checkpoint",
        "manual_seed",
        "forward",
        "train_target_model",
        "optimizer",
        "backward",
        "step",
        "Start-Process",
    }
    old_runner_re = re.compile(r"^run_stable_loop_phase_lock_")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports or old_runner_re.match(alias.name):
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or old_runner_re.match(module):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
    return failures


def check_tagged_entry(item: Any) -> bool:
    return isinstance(item, dict) and item.get("evidence_type") in TAG_TYPES and "source" in item and "value" in item


def check_tagged_report(report: dict[str, Any], failures: list[str], name: str) -> None:
    if report.get("evidence_tags_present") is not True:
        failures.append(f"EVIDENCE_TAGS_MISSING:{name}")
    for container_key in ["fields", "metrics", "computed"]:
        container = report.get(container_key, {})
        if isinstance(container, dict):
            for key, item in container.items():
                if not check_tagged_entry(item):
                    failures.append(f"UNTAGGED_FIELD:{name}:{container_key}:{key}")
    if "classification" in report and not check_tagged_entry(report["classification"]):
        failures.append(f"UNTAGGED_FIELD:{name}:classification")
    if "selected_root_cause" in report and not check_tagged_entry(report["selected_root_cause"]):
        failures.append(f"UNTAGGED_FIELD:{name}:selected_root_cause")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    manifest = load_json(SMOKE_ROOT / "upstream_138r_manifest.json")
    if manifest.get("upstream_138r_verified") is not True:
        failures.append("UPSTREAM_138R_ARTIFACT_MISSING")
    expected_manifest = {
        "verdict": "REAL_RAW_REASONING_REPAIR_PROBE_FAILS",
        "decision": "teacher_forcing_or_training_objective_failure",
        "next": "138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS",
        "determinism_replay_passed": True,
        "mean_real_raw_reasoning_accuracy": 0.0,
        "expected_token_inclusion_rate": 0.0,
        "helper_canary_ast_leakage_controls_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            failures.append("UPSTREAM_138R_ARTIFACT_MISSING")

    config = load_json(SMOKE_ROOT / "objective_failure_config.json")
    for key in ["artifact_only_analysis", "no_training", "no_new_inference", "no_helper_calls_for_new_generations", "no_torch_forward_passes"]:
        if config.get(key) is not True:
            failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")

    teacher = load_json(SMOKE_ROOT / "teacher_forcing_vs_rollout_report.json")
    patterns = load_json(SMOKE_ROOT / "rollout_output_pattern_report.json")
    namespace = load_json(SMOKE_ROOT / "train_eval_answer_namespace_report.json")
    alignment = load_json(SMOKE_ROOT / "prompt_answer_alignment_report.json")
    mismatch = load_json(SMOKE_ROOT / "first_mismatch_report.json")
    stop = load_json(SMOKE_ROOT / "stop_behavior_report.json")
    scoring = load_json(SMOKE_ROOT / "scoring_strictness_recheck.json")
    checkpoint = load_json(SMOKE_ROOT / "checkpoint_objective_gap_report.json")
    root_cause = load_json(SMOKE_ROOT / "objective_failure_root_cause.json")
    for name, report in [
        ("teacher", teacher),
        ("patterns", patterns),
        ("namespace", namespace),
        ("alignment", alignment),
        ("mismatch", mismatch),
        ("stop", stop),
        ("scoring", scoring),
        ("checkpoint", checkpoint),
        ("root_cause", root_cause),
    ]:
        check_tagged_report(report, failures, name)

    if teacher["fields"]["teacher_forced_loss_initial"]["evidence_type"] != "diagnostic_gap":
        failures.append("TEACHER_FORCED_LOSS_OVERCLAIM")
    if teacher["fields"]["teacher_forced_loss_final"]["evidence_type"] != "diagnostic_gap":
        failures.append("TEACHER_FORCED_LOSS_OVERCLAIM")
    if teacher["computed"]["teacher_forced_loss_improved_claim_allowed"]["value"] is not False:
        failures.append("TEACHER_FORCED_LOSS_OVERCLAIM")
    if patterns["metrics"]["row_count"]["value"] != 960:
        failures.append("ROLLOUT_PATTERN_REPORT_INVALID")
    if patterns["metrics"]["expected_token_inclusion_rate"]["value"] != 0.0:
        failures.append("ROLLOUT_PATTERN_REPORT_INVALID")
    if namespace["fields"]["generated_train_namespace_token_rate"]["value"] <= 0.0:
        failures.append("NAMESPACE_ANALYSIS_MISSING")
    if scoring["fields"]["near_match_rate"]["value"] <= 0.0:
        failures.append("EXPECTED_NONZERO_NEAR_MATCH_NOT_FOUND")
    if scoring["classification"]["value"] != "scoring_or_task_weakness_possible":
        failures.append("SCORING_STRICTNESS_RECHECK_INVALID")
    if root_cause["selected_root_cause"]["value"] != "objective_failure_ambiguous":
        failures.append("ROOT_CAUSE_OVERCLAIM")

    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    if gaps.get("diagnostic_gap_count", 0) < 2:
        failures.append("DIAGNOSTIC_GAPS_NOT_RECORDED")

    requirements = load_json(SMOKE_ROOT / "next_objective_redesign_requirements.json")
    required_gates = {
        "helper-only final eval",
        "generated_text before scoring",
        "expected-output canary",
        "AST shortcut scan",
        "deterministic replay",
        "controls fail",
        "leakage rejected",
        "clean negative accepted",
    }
    rejects = {
        "teacher-forcing-only success",
        "loss-only success",
        "expected-output construction",
        "old runner imports",
        "oracle/rerank/verifier/LLM judge",
        "post-generation repair",
        "threshold weakening to force positive",
    }
    if not required_gates.issubset(set(requirements.get("required_gates", []))):
        failures.append("NEXT_OBJECTIVE_REDESIGN_REQUIREMENTS_INCOMPLETE")
    if not rejects.issubset(set(requirements.get("explicitly_reject", []))):
        failures.append("NEXT_OBJECTIVE_REDESIGN_REQUIREMENTS_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "objective_failure_ambiguous" or decision.get("next") != "138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION":
        failures.append("DECISION_MISMATCH")
    if decision.get("artifact_only_analysis") is not True or decision.get("new_model_inference_run") is not False or decision.get("shared_helper_called_for_new_generation") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    for key in FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")

    summary = load_json(SMOKE_ROOT / "summary.json")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    if summary.get("artifact_only_analysis") is not True or summary.get("training_performed") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report_text))
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
        print("138G checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138G checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
