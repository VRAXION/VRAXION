#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_121."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan.py",
    "scripts/probes/run_stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_120_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_116_manifest.json",
    "upstream_115_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "post_reasoning_failure_priority_map.json",
    "breakpoint_analysis.json",
    "root_vs_symptom_analysis.json",
    "repair_target_selection.json",
    "training_design_options.json",
    "eval_gate_proposal.json",
    "risk_register.json",
    "111_failure_prevention_map.json",
    "next_milestone_plan.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN",
    "planning only",
    "TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE",
    "UPSTREAM_120_POST_REASONING_MAP_VERIFIED",
    "MULTI_TURN_STATE_REPAIR_TARGET_SELECTED",
    "122_MULTI_TURN_STATE_REPAIR",
    "TIER_4_MULTI_TURN_STATE_UPDATE",
    "multi_turn_state_failure",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
    "HUNGARIAN_ASSISTANT_CLAIM_DETECTED": ["Hungarian assistant ready"],
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
        window = lowered[max(0, match.start() - 220) : match.start()]
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
    for rel in REQUIRED_DOCS + REQUIRED_SOURCE:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in REQUIRED_DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def require_plan_terms(plan_text: str) -> list[str]:
    failures: list[str] = []
    required_terms = [
        "multi-turn corrections",
        "active vs stale state tracking",
        "override chains",
        "slot updates across turns",
        "table/doc facts plus state updates",
        "bounded refusal with state carry",
        "stale-state decoys",
        "train/eval namespace disjointness",
        "anti-memorization rows",
        "leakage audit against 112-121 artifacts",
        "scheduled sampling or rollout-style objective if training is used",
        "raw-only final eval",
        "no teacher-forcing-only success",
        "oracle rerank",
        "expected-answer metadata",
        "decoder reference",
        "integrated policy",
        "multi_turn_state_accuracy",
        "state_tracking_accuracy",
        "multi_turn_correction_accuracy",
        "stale_state_rejection_accuracy",
        "override_chain_accuracy",
        "active_slot_after_update_accuracy",
        "retention metrics",
        "collapse metrics",
        "namespace drift metrics",
        "leakage metrics",
    ]
    for term in required_terms:
        if term not in plan_text:
            failures.append(f"NEXT_MILESTONE_PLAN_MISSING:{term}")
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    verdicts = set(summary.get("verdicts", []))
    if "TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE" not in verdicts:
        failures.append("TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_RESULT_MISSING")
    for key, expected in [
        ("planning_only", True),
        ("analysis_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("inference_run_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("gpt_like_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "TRAINING_SIDE_EFFECT_DETECTED")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("inference_run_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
    ]:
        if metrics.get(key) != expected:
            failures.append("TRAINING_SIDE_EFFECT_DETECTED")

    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in [
        "startup",
        "upstream_verification",
        "artifact_loading",
        "failure_prioritization",
        "root_symptom_analysis",
        "repair_target_selection",
        "eval_gate_proposal",
        "decision_writing",
        "final_verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    priority = load_json(SMOKE_ROOT / "post_reasoning_failure_priority_map.json")
    if priority.get("first_breakpoint_tier") != "TIER_4_MULTI_TURN_STATE_UPDATE":
        failures.append("FAILURE_PRIORITY_MAP_MISSING")
    if priority.get("selected_priority_failure") != "multi_turn_state_failure":
        failures.append("FAILURE_PRIORITY_MAP_MISSING")
    if priority.get("reasoning_regression_rejected") is not True or float(priority.get("reasoning_failure_rate", 1.0)) != 0.0:
        failures.append("REASONING_PRESERVATION_GATE_MISSING")
    if priority.get("selection_rule") != "first_breakpoint_tier outranks global failure count":
        failures.append("FAILURE_PRIORITY_MAP_MISSING")

    breakpoint = load_json(SMOKE_ROOT / "breakpoint_analysis.json")
    if breakpoint.get("first_breakpoint_tier") != "TIER_4_MULTI_TURN_STATE_UPDATE":
        failures.append("BREAKPOINT_ANALYSIS_MISSING")

    root = load_json(SMOKE_ROOT / "root_vs_symptom_analysis.json")
    if root.get("first_breakpoint_outranks_global_count") is not True or root.get("later_tier_target_selected_first") is not False:
        failures.append("ROOT_VS_SYMPTOM_ANALYSIS_MISSING")

    selection = load_json(SMOKE_ROOT / "repair_target_selection.json")
    if selection.get("selected_next_milestone") != "122_MULTI_TURN_STATE_REPAIR" or selection.get("selected_repair_target") != "multi_turn_state_first":
        failures.append("REPAIR_TARGET_SELECTION_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("selected_next_milestone") != "122_MULTI_TURN_STATE_REPAIR":
        failures.append("DECISION_MISSING")
    for key, expected in [
        ("first_breakpoint_tier", "TIER_4_MULTI_TURN_STATE_UPDATE"),
        ("primary_next_repair_target", "multi_turn_state_failure"),
        ("reasoning_regression_rejected", True),
        ("reasoning_failure_rate", 0.0),
    ]:
        if decision.get(key) != expected:
            failures.append("DECISION_MISSING")
    for key in [
        "why_not_hallucination_refusal_first",
        "why_not_format_injection_first",
        "why_not_long_context_first",
        "why_not_more_general_training",
        "why_not_deploy_polish",
        "why_not_architecture_pivot",
    ]:
        if not decision.get(key):
            failures.append("DECISION_MISSING")

    next_plan = load_json(SMOKE_ROOT / "next_milestone_plan.json")
    if next_plan.get("milestone_name") != "122_MULTI_TURN_STATE_REPAIR":
        failures.append("NEXT_MILESTONE_PLAN_MISSING")
    plan_text = json.dumps(next_plan, sort_keys=True)
    failures.extend(require_plan_terms(plan_text))
    reasoning_gates = next_plan.get("reasoning_preservation_gates", {})
    if reasoning_gates.get("tier4_reasoning_accuracy") != ">= 0.97":
        failures.append("REASONING_PRESERVATION_GATE_MISSING")
    if reasoning_gates.get("tier8_reasoning_combo_accuracy") != ">= 0.90":
        failures.append("REASONING_PRESERVATION_GATE_MISSING")
    if reasoning_gates.get("reasoning_failure_rate") != "<= 0.05":
        failures.append("REASONING_PRESERVATION_GATE_MISSING")

    for rel in ["eval_gate_proposal.json", "risk_register.json", "111_failure_prevention_map.json", "training_design_options.json"]:
        if not load_json(SMOKE_ROOT / rel):
            failures.append(f"MISSING_ARTIFACT_CONTENT:{rel}")
    prevention_text = json.dumps(load_json(SMOKE_ROOT / "111_failure_prevention_map.json"), sort_keys=True)
    for term in ["train_eval_namespace_disjointness", "anti_memorization_rows", "no_teacher_forcing_only_success", "no_oracle_rerank", "no_decoder_reference", "no_integrated_policy_during_final_eval"]:
        if term not in prevention_text:
            failures.append(f"111_FAILURE_PREVENTION_MISSING:{term}")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
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
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if args.check_only:
        failures.extend(check_artifacts())
    if failures:
        print("121 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("121 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
