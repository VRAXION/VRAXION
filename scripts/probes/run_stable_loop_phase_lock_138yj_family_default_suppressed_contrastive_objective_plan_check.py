#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138YJ family-default-suppressed plan."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yj_family_default_suppressed_contrastive_objective_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yj_family_default_suppressed_contrastive_objective_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yj_family_default_suppressed_contrastive_objective_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yd_manifest.json",
    "upstream_138yh_manifest.json",
    "upstream_138yi_manifest.json",
    "analysis_config.json",
    "family_default_failure_summary.json",
    "objective_weakness_diagnosis.json",
    "family_default_suppression_requirements.json",
    "strengthened_contrast_group_design.json",
    "hard_negative_family_default_policy.json",
    "anti_shortcut_requirements.json",
    "target_138yk_training_objective_spec.json",
    "target_138yk_eval_gate_spec.json",
    "target_138yk_failure_routes.json",
    "next_138yk_milestone_plan.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "family_default_suppressed_contrastive_objective_plan_complete",
    "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
    "family_default_value_bank",
    "hard_negative_default_rows",
    "deterministic replay",
    "shared_raw_generation_helper.py",
    "contrastive_objective_too_weak",
    "diagnostic_gap",
    "artifact-only",
    "planning-only",
    "Raw assistant capability remains quarantined",
    "Structured/tool capability remains invalidated",
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
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "remains quarantined", "remains invalidated"]
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 260) : match.start()]
        after = lowered[match.end() : min(len(lowered), match.end() + 80)]
        if any(marker in window for marker in NEGATION_MARKERS) or any(marker in after for marker in ["false", "not ", "no "]):
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


def ast_scan(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    forbidden_calls = {"raw_generate", "load_checkpoint", "manual_seed", "forward", "backward", "optimizer", "train_target_model"}
    for path in paths:
        if not path.exists():
            failures.append(f"MISSING_AST_SOURCE:{path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        failures.append("PLANNING_ONLY_BOUNDARY_FAILURE")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or module.startswith("run_stable_loop_phase_lock_"):
                    failures.append("PLANNING_ONLY_BOUNDARY_FAILURE")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("PLANNING_ONLY_BOUNDARY_FAILURE")
    return failures


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_CLAIM_FAILURE:{key}")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 11:
        failures.append("PROGRESS_NOT_REFRESHED")
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "family default failure summary",
        "objective weakness diagnosis",
        "suppression requirement drafting",
        "contrast group redesign",
        "hard negative policy",
        "anti-shortcut design",
        "target 138YK plan writing",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    upstream_yd = load_json(SMOKE_ROOT / "upstream_138yd_manifest.json")
    upstream_yh = load_json(SMOKE_ROOT / "upstream_138yh_manifest.json")
    upstream_yi = load_json(SMOKE_ROOT / "upstream_138yi_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    failure_summary = load_json(SMOKE_ROOT / "family_default_failure_summary.json")
    weakness = load_json(SMOKE_ROOT / "objective_weakness_diagnosis.json")
    suppression = load_json(SMOKE_ROOT / "family_default_suppression_requirements.json")
    contrast = load_json(SMOKE_ROOT / "strengthened_contrast_group_design.json")
    hard_negative = load_json(SMOKE_ROOT / "hard_negative_family_default_policy.json")
    training_spec = load_json(SMOKE_ROOT / "target_138yk_training_objective_spec.json")
    eval_spec = load_json(SMOKE_ROOT / "target_138yk_eval_gate_spec.json")
    routes = load_json(SMOKE_ROOT / "target_138yk_failure_routes.json")
    milestone = load_json(SMOKE_ROOT / "next_138yk_milestone_plan.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if upstream_yd.get("decision") != "family_default_shortcut_analysis_complete" or upstream_yd.get("root_cause") != "contrastive_objective_too_weak":
        failures.append("UPSTREAM_138YD_ROUTE_MISMATCH")
    if upstream_yd.get("contrast_group_default_shortcut_rate") != 0.78125 or upstream_yd.get("multi_expected_to_single_default_rate") != 0.6822916666666666:
        failures.append("UPSTREAM_138YD_METRIC_PROFILE_MISMATCH")
    if upstream_yd.get("objective_explicitly_penalizes_family_default") is not False or upstream_yd.get("objective_rewards_intra_family_distinct_values") is not False:
        failures.append("UPSTREAM_138YD_OBJECTIVE_PROFILE_MISMATCH")
    if upstream_yh.get("global_top5_train_all_replay_rate") != 0.0 or upstream_yh.get("family_top5_train_all_replay_rate") != 0.0:
        failures.append("UPSTREAM_138YH_FALSIFICATION_PROFILE_MISMATCH")
    if upstream_yi.get("verified") is not True or upstream_yi.get("determinism_replay_passed") is not True or upstream_yi.get("train_namespace_leak_rate") != 0.0:
        failures.append("UPSTREAM_138YI_INTEGRITY_MISSING")

    for key in ["artifact_only", "planning_only", "training_performed", "new_model_inference_run", "shared_helper_called", "torch_forward_pass_run", "checkpoint_mutation_performed", "runtime_surface_mutated", "root_license_changed"]:
        expected = True if key in {"artifact_only", "planning_only"} else False
        if config.get(key) is not expected:
            failures.append(f"PLANNING_ONLY_BOUNDARY_FAILURE:{key}")

    if failure_summary.get("root_cause") != "contrastive_objective_too_weak":
        failures.append("FAILURE_SUMMARY_ROOT_MISMATCH")
    if failure_summary.get("contrast_group_default_shortcut_rate") != 0.78125 or failure_summary.get("answer_value_accuracy") != 0.0:
        failures.append("FAILURE_SUMMARY_METRIC_MISMATCH")
    if weakness.get("objective_pressure_against_family_default_reuse_insufficient") is not True:
        failures.append("OBJECTIVE_WEAKNESS_NOT_RECORDED")
    for key in ["family_default_value_bank", "per_family_forbidden_default_values", "required_suppression_mechanisms", "required_138yk_metrics"]:
        if key not in suppression:
            failures.append(f"SUPPRESSION_REQUIREMENT_MISSING:{key}")
    if not any(item.startswith("hard_negative_default_rows") for item in milestone.get("required_artifacts", [])):
        failures.append("HARD_NEGATIVE_ROWS_NOT_REQUIRED")
    if "family_default_value_bank.json" not in milestone.get("required_artifacts", []):
        failures.append("FAMILY_DEFAULT_VALUE_BANK_NOT_REQUIRED")
    if milestone.get("milestone") != "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE":
        failures.append("NEXT_138YK_MILESTONE_MISMATCH")
    if milestone.get("shared_raw_generation_helper_only_for_final_eval") is not True or milestone.get("target_checkpoint_under_target_only") is not True:
        failures.append("NEXT_138YK_HELPER_OR_TARGET_POLICY_MISSING")
    if "at least one hard negative default value from prior failure artifacts" not in contrast.get("group_requirements", []):
        failures.append("CONTRAST_GROUP_HARD_NEGATIVE_MISSING")
    if hard_negative.get("row_failure_rule") is None:
        failures.append("HARD_NEGATIVE_POLICY_INCOMPLETE")
    if "family default wrong value" not in training_spec.get("objective_penalizes", []):
        failures.append("TRAINING_SPEC_DEFAULT_PENALTY_MISSING")
    if eval_spec.get("generated_text_before_scoring") is not True or eval_spec.get("final_eval_helper_only") != "scripts/probes/shared_raw_generation_helper.py":
        failures.append("EVAL_GATE_HELPER_ONLY_MISSING")
    if routes.get("clean_negative_routes", {}).get("family_default_shortcut_persists") != "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS":
        failures.append("FAILURE_ROUTE_MISSING")

    if decision.get("decision") != "family_default_suppressed_contrastive_objective_plan_complete" or decision.get("next") != "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE":
        failures.append("DECISION_MISMATCH")
    for field in ["output_head_prior", "hidden_state_carrier", "grower_scout_behavior", "topological_inhibition"]:
        if not any(gap.get("field") == field and gap.get("status") == "diagnostic_gap" for gap in gaps.get("gaps", [])):
            failures.append(f"DIAGNOSTIC_GAP_MISSING:{field}")

    require_false_flags(decision, failures)
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
    failures.extend(ast_scan([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138YJ checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138YJ checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
