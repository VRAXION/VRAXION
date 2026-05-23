#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138YD family-default shortcut analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yd_family_default_shortcut_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yd_family_default_shortcut_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yd_family_default_shortcut_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yh_manifest.json",
    "upstream_138yi_manifest.json",
    "analysis_config.json",
    "family_default_shortcut_map.json",
    "default_value_origin_report.json",
    "family_template_shortcut_report.json",
    "contrast_group_default_failure_report.json",
    "objective_shortcut_reward_report.json",
    "scorer_dataset_shortcut_report.json",
    "family_default_root_cause.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
    "family_default_shortcut_analysis_complete",
    "template_induced_family_default_shortcut",
    "objective_allows_family_default_shortcut",
    "contrastive_objective_too_weak",
    "dataset_low_intra_family_value_diversity",
    "scorer_family_default_weakness",
    "model_family_default_attractor_output_behavior",
    "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "artifact-only",
    "family-default shortcut",
    "diagnostic_gap",
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
ROOTS = {
    "template_induced_family_default_shortcut",
    "objective_allows_family_default_shortcut",
    "contrastive_objective_too_weak",
    "dataset_low_intra_family_value_diversity",
    "scorer_family_default_weakness",
    "model_family_default_attractor_output_behavior",
    "mixed_family_default_shortcut",
    "family_default_shortcut_ambiguous",
}
ROUTES = {
    "template_induced_family_default_shortcut": "138YT_TEMPLATE_DECONFOUNDING_VALUE_GROUNDING_PLAN",
    "objective_allows_family_default_shortcut": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "contrastive_objective_too_weak": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "dataset_low_intra_family_value_diversity": "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN",
    "scorer_family_default_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
    "model_family_default_attractor_output_behavior": "138YJ_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_OBJECTIVE_PLAN",
    "mixed_family_default_shortcut": "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET",
    "family_default_shortcut_ambiguous": "138YDB_FAMILY_DEFAULT_SHORTCUT_MANUAL_REVIEW_PACKET",
}
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


def ast_scan(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    forbidden_calls = {"raw_generate", "load_checkpoint", "manual_seed", "forward", "backward", "optimizer", "train_target_model", "train("}
    for path in paths:
        if not path.exists():
            failures.append(f"MISSING_AST_SOURCE:{path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or module.startswith("run_stable_loop_phase_lock_"):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
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
    if len(progress) < 12:
        failures.append("PROGRESS_NOT_REFRESHED")
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "family default mapping",
        "default origin analysis",
        "template shortcut analysis",
        "contrast group failure analysis",
        "objective shortcut analysis",
        "scorer/dataset shortcut analysis",
        "root cause selection",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    upstream_yh = load_json(SMOKE_ROOT / "upstream_138yh_manifest.json")
    upstream_yi = load_json(SMOKE_ROOT / "upstream_138yi_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    shortcut = load_json(SMOKE_ROOT / "family_default_shortcut_map.json")
    origin = load_json(SMOKE_ROOT / "default_value_origin_report.json")
    template = load_json(SMOKE_ROOT / "family_template_shortcut_report.json")
    contrast = load_json(SMOKE_ROOT / "contrast_group_default_failure_report.json")
    objective = load_json(SMOKE_ROOT / "objective_shortcut_reward_report.json")
    scorer = load_json(SMOKE_ROOT / "scorer_dataset_shortcut_report.json")
    root = load_json(SMOKE_ROOT / "family_default_root_cause.json")
    recommendation = load_json(SMOKE_ROOT / "next_repair_recommendation.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if upstream_yh.get("decision") != "high_frequency_value_replay_analysis_complete" or upstream_yh.get("root_cause") != "family_default_shortcut_replay":
        failures.append("UPSTREAM_138YH_ROUTE_MISMATCH")
    if upstream_yh.get("global_top5_train_all_replay_rate") != 0.0 or upstream_yh.get("family_top5_train_all_replay_rate") != 0.0:
        failures.append("UPSTREAM_138YH_FALSIFICATION_PROFILE_MISMATCH")
    if upstream_yh.get("strict_train_all_membership_rate") != 0.13671875 or upstream_yh.get("family_default_attractor_rate") != 0.78125:
        failures.append("UPSTREAM_138YH_METRIC_PROFILE_MISMATCH")
    if upstream_yi.get("verified") is not True or upstream_yi.get("controls_failed") is not True or upstream_yi.get("determinism_replay_passed") is not True:
        failures.append("UPSTREAM_138YI_INTEGRITY_MISSING")
    if upstream_yi.get("stale_chat_fragment_rate") != 0.0 or upstream_yi.get("train_namespace_leak_rate") != 0.0:
        failures.append("UPSTREAM_138YI_FAILURE_PROFILE_MISMATCH")

    for key in ["artifact_only", "training_performed", "new_model_inference_run", "shared_helper_called", "torch_forward_pass_run", "checkpoint_mutation_performed", "runtime_surface_mutated", "root_license_changed"]:
        expected = True if key == "artifact_only" else False
        if config.get(key) is not expected:
            failures.append(f"ARTIFACT_ONLY_BOUNDARY_FAILURE:{key}")

    if shortcut.get("family_count") != 8 or shortcut.get("row_count") != 768:
        failures.append("FAMILY_DEFAULT_MAP_INCOMPLETE")
    allowed_labels = {"strong_family_default_shortcut", "moderate_family_default_shortcut", "weak_family_default_shortcut", "no_family_default_shortcut", "ambiguous_family_default"}
    for family, payload in shortcut.get("families", {}).items():
        if payload.get("family_default_label") not in allowed_labels:
            failures.append(f"FAMILY_LABEL_INVALID:{family}")
        if "dominant_default_value" not in payload or "contrast_groups_collapsed_to_default" not in payload:
            failures.append(f"FAMILY_DEFAULT_FIELD_MISSING:{family}")
    if origin.get("family_count") != 8:
        failures.append("DEFAULT_ORIGIN_INCOMPLETE")
    if template.get("template_default_correlation_rate") is None or template.get("family_default_correlation_rate") is None:
        failures.append("TEMPLATE_CORRELATION_MISSING")
    if contrast.get("group_count") != 192 or contrast.get("contrast_group_default_shortcut_rate", 0.0) <= 0.0:
        failures.append("CONTRAST_GROUP_DEFAULT_REPORT_INVALID")
    if objective.get("positive_can_depend_on_train_loss") is not False:
        failures.append("TRAIN_LOSS_SUCCESS_OVERCLAIM")
    if not objective.get("diagnostic_gap"):
        failures.append("OBJECTIVE_DIAGNOSTIC_GAP_MISSING")
    if scorer.get("family_default_control_failed") is not True:
        failures.append("FAMILY_DEFAULT_CONTROL_NOT_FAILED")

    if root.get("root_cause") not in ROOTS:
        failures.append("ROOT_CAUSE_UNKNOWN")
    expected_next = ROUTES.get(root.get("root_cause"))
    if recommendation.get("recommended_next") != expected_next or decision.get("next") != expected_next:
        failures.append("RECOMMENDATION_ROUTE_MISMATCH")
    if decision.get("decision") != "family_default_shortcut_analysis_complete":
        failures.append("DECISION_MISMATCH")
    if root.get("internal_mechanism_claim_status") != "diagnostic_gap_without_logits_hidden_state_or_grower_scout_artifacts":
        failures.append("INTERNAL_MECHANISM_DIAGNOSTIC_GAP_MISSING")
    for field in ["output_head_prior", "hidden_state_family_default_mechanism", "grower_scout_behavior"]:
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
        print("138YD checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138YD checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
