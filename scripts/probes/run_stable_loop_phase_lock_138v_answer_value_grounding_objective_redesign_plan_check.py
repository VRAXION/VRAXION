#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138V answer-value grounding plan."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138v_answer_value_grounding_objective_redesign_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138v_answer_value_grounding_objective_redesign_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138v_answer_value_grounding_objective_redesign_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138s_manifest.json",
    "wrapper_value_gap_summary.json",
    "no_stale_value_failure_summary.json",
    "residual_signal_carrier_requirements.json",
    "wrapper_induced_amnesia_hypothesis.json",
    "value_grounding_objective_requirements.json",
    "ood_value_grounding_eval_requirements.json",
    "stale_secondary_gate_requirements.json",
    "next_138w_milestone_plan.json",
    "anti_shortcut_requirements.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN",
    "Wrapper-Induced Amnesia",
    "Residual Signal Carrier",
    "diagnostic_gap",
    "wrapper_success_without_value_grounding",
    "138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE",
    "answer_value_accuracy improves from 0.0",
    "prefix_success_value_failure_rate decreases",
    "shared_raw_generation_helper.py only",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
    "Not GPT-like readiness",
    "Not open-domain assistant readiness",
    "Not production chat",
    "Not public API",
    "Not deployment readiness",
    "Not safety alignment",
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
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "remains quarantined", "remains invalidated", "planning-only"]
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    old_runner_re = re.compile(r"^run_stable_loop_phase_lock_")
    forbidden_calls = {"raw_generate", "manual_seed", "forward", "backward", "optimizer", "load_checkpoint"}
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

    progress = load_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 10:
        failures.append("PROGRESS_NOT_REFRESHED")

    manifest = load_json(SMOKE_ROOT / "upstream_138s_manifest.json")
    if manifest.get("upstream_138s_verified") is not True or manifest.get("upstream_138i_helper_integrity_verified") is not True:
        failures.append("UPSTREAM_138S_ARTIFACT_MISSING")
    if manifest.get("decision") != "stale_chat_rollout_failure_analysis_complete" or manifest.get("next") != "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN":
        failures.append("UPSTREAM_138S_ROUTE_MISMATCH")
    if manifest.get("primary_diagnosis") != "answer_value_grounding_failure_decoupled_from_stale_chat":
        failures.append("UPSTREAM_138S_DIAGNOSIS_MISMATCH")
    if manifest.get("answer_prefix_accuracy") != 1.0 or manifest.get("eval_namespace_emission_accuracy") != 1.0:
        failures.append("WRAPPER_VALUE_GAP_MISSING")
    if manifest.get("answer_value_accuracy") != 0.0 or manifest.get("wrapper_prefix_learned_without_value_grounding") is not True:
        failures.append("WRAPPER_VALUE_GAP_MISSING")
    if manifest.get("P_wrong_value_given_stale_chat") != 1.0 or manifest.get("P_wrong_value_given_no_stale_chat") != 1.0:
        failures.append("NO_STALE_VALUE_FAILURE_MISSING")
    if manifest.get("canary_ast_controls_leakage_determinism_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")
    if manifest.get("no_expected_or_scorer_metadata_reached_helper_requests") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")

    wrapper = load_json(SMOKE_ROOT / "wrapper_value_gap_summary.json")
    no_stale = load_json(SMOKE_ROOT / "no_stale_value_failure_summary.json")
    residual = load_json(SMOKE_ROOT / "residual_signal_carrier_requirements.json")
    amnesia = load_json(SMOKE_ROOT / "wrapper_induced_amnesia_hypothesis.json")
    value_reqs = load_json(SMOKE_ROOT / "value_grounding_objective_requirements.json")
    ood = load_json(SMOKE_ROOT / "ood_value_grounding_eval_requirements.json")
    stale = load_json(SMOKE_ROOT / "stale_secondary_gate_requirements.json")
    plan = load_json(SMOKE_ROOT / "next_138w_milestone_plan.json")
    anti = load_json(SMOKE_ROOT / "anti_shortcut_requirements.json")
    decision = load_json(SMOKE_ROOT / "decision.json")

    if wrapper.get("answer_prefix_accuracy") != 1.0 or wrapper.get("eval_namespace_emission_accuracy") != 1.0:
        failures.append("WRAPPER_VALUE_GAP_SUMMARY_INVALID")
    if wrapper.get("answer_value_accuracy") != 0.0 or wrapper.get("prefix_success_value_failure_rate") != 1.0:
        failures.append("WRAPPER_VALUE_GAP_SUMMARY_INVALID")
    if wrapper.get("wrapper_prefix_learned_without_value_grounding") is not True:
        failures.append("WRAPPER_VALUE_GAP_SUMMARY_INVALID")
    if no_stale.get("P_wrong_value_given_stale_chat") != 1.0 or no_stale.get("P_wrong_value_given_no_stale_chat") != 1.0:
        failures.append("NO_STALE_VALUE_FAILURE_SUMMARY_INVALID")
    if "secondary failure" not in no_stale.get("conclusion", ""):
        failures.append("NO_STALE_VALUE_FAILURE_SUMMARY_INVALID")
    if residual.get("hidden_state_residual_signal_measurement") != "diagnostic_gap":
        failures.append("HIDDEN_STATE_OVERCLAIM")
    if residual.get("not_a_hidden_state_claim") is not True:
        failures.append("HIDDEN_STATE_OVERCLAIM")
    for metric in [
        "answer_value_accuracy",
        "exact_answer_accuracy",
        "value_after_prefix_accuracy",
        "value_position_error_rate",
        "empty_value_after_prefix_rate",
        "generic_value_after_prefix_rate",
        "prompt_value_copy_accuracy",
        "rule_derived_value_accuracy",
        "table_derived_value_accuracy",
    ]:
        if metric not in residual.get("output_level_proxy_metrics", []):
            failures.append("RESIDUAL_SIGNAL_PROXY_MISSING")
    layers = residual.get("layer_separation", {})
    if layers.get("wrapper_reflex", {}).get("must_not_count_as_grounding") is not True:
        failures.append("RESIDUAL_SIGNAL_PROXY_MISSING")
    if layers.get("value_carrier", {}).get("hidden_state_claim_status") != "diagnostic_gap_without_instrumentation":
        failures.append("HIDDEN_STATE_OVERCLAIM")
    if layers.get("value_grounding", {}).get("status") != "next_probe_target":
        failures.append("VALUE_GROUNDING_REWARD_MISSING")
    if amnesia.get("status") != "planning_hypothesis" or amnesia.get("hidden_state_claim_status") != "diagnostic_gap_without_instrumentation":
        failures.append("WRAPPER_INDUCED_AMNESIA_OVERCLAIM")
    if "correct value after ANSWER=E" not in value_reqs.get("must_directly_reward", []):
        failures.append("VALUE_GROUNDING_REWARD_MISSING")
    if "ANSWER=E with wrong value" not in value_reqs.get("must_directly_penalize", []):
        failures.append("VALUE_GROUNDING_PENALTY_MISSING")
    if "train/eval value namespaces disjoint" not in ood.get("requirements", []):
        failures.append("OOD_VALUE_GROUNDING_REQUIREMENT_MISSING")
    if stale.get("stale_chat_is_secondary_hard_gate") is not True:
        failures.append("STALE_SECONDARY_GATE_MISSING")
    if plan.get("milestone") != "138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE":
        failures.append("NEXT_138W_PLAN_INVALID")
    for required in ["shared_raw_generation_helper.py only", "generated_text before scoring", "deterministic replay", "controls fail", "leakage rejected"]:
        if required not in plan.get("required_path", []):
            failures.append("NEXT_138W_GATE_MISSING")
    for required in [
        "answer_value_accuracy improves from 0.0",
        "exact_answer_accuracy improves from 0.0",
        "prefix_success_value_failure_rate decreases",
        "P(wrong_value | no_stale_chat) decreases from 1.0",
        "value_after_prefix_accuracy improves from 0.0",
        "value_position_error_rate decreases",
    ]:
        if required not in plan.get("positive_gates", []):
            failures.append("NEXT_138W_POSITIVE_GATE_MISSING")
    if anti.get("threshold_weakening_to_force_positive_allowed") is not False or anti.get("expected_output_may_enter_generation") is not False:
        failures.append("ANTI_SHORTCUT_REQUIREMENT_MISSING")
    for phrase in ["train-loss-only success", "prefix-only success", "namespace-only success", "old runner imports"]:
        if phrase not in anti.get("explicit_rejects", []):
            failures.append("ANTI_SHORTCUT_REQUIREMENT_MISSING")

    if decision.get("decision") != "answer_value_grounding_objective_redesign_plan_complete" or decision.get("next") != "138W_ANSWER_VALUE_GROUNDING_REPAIR_PROBE":
        failures.append("DECISION_MISMATCH")
    if decision.get("hidden_state_residual_signal_measurement") != "diagnostic_gap":
        failures.append("HIDDEN_STATE_OVERCLAIM")
    if decision.get("planning_only") is not True or decision.get("training_performed") is not False or decision.get("new_model_inference_run") is not False:
        failures.append("PLANNING_ONLY_BOUNDARY_FAILURE")
    require_false_flags(decision, failures)

    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    require_false_flags(summary, failures)
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
        print("138V checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138V checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
