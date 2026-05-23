#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138YF family-specific value attractor plan."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yf_family_specific_value_attractor_repair_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yf_family_specific_value_attractor_repair_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yf_family_specific_value_attractor_repair_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138u_manifest.json",
    "upstream_138w_manifest.json",
    "analysis_config.json",
    "family_specific_attractor_summary.json",
    "train_membership_reconciliation.json",
    "intra_family_mode_collapse_report.json",
    "intra_family_contrastive_objective_requirements.json",
    "deep_scout_forcing_hypothesis.json",
    "carrier_proxy_requirements.json",
    "anti_shortcut_requirements.json",
    "next_138yi_milestone_plan.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN",
    "family_specific_train_value_attractor",
    "Scout-First Laziness",
    "Missing Intra-Family Variance",
    "intra_family_contrastive_accuracy",
    "intra_family_mode_collapse_rate",
    "family_default_attractor_rate",
    "138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE",
    "family_specific_value_attractor_repair_plan_complete",
    "diagnostic_gap",
    "artifact-only",
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
    "SCOUT_INTERNAL_OVERCLAIM": ["scout/grower behavior measured", "actual scout behavior measured"],
}
EXPECTED_STRICT_TRAIN_RATE = 0.09895833333333333


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12


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
            failures.append("BOUNDARY_CLAIM_FAILURE")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 10:
        failures.append("PROGRESS_NOT_REFRESHED")

    upstream_138u = load_json(SMOKE_ROOT / "upstream_138u_manifest.json")
    upstream_138w = load_json(SMOKE_ROOT / "upstream_138w_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    summary_report = load_json(SMOKE_ROOT / "family_specific_attractor_summary.json")
    reconciliation = load_json(SMOKE_ROOT / "train_membership_reconciliation.json")
    collapse = load_json(SMOKE_ROOT / "intra_family_mode_collapse_report.json")
    contrastive = load_json(SMOKE_ROOT / "intra_family_contrastive_objective_requirements.json")
    scout = load_json(SMOKE_ROOT / "deep_scout_forcing_hypothesis.json")
    carrier = load_json(SMOKE_ROOT / "carrier_proxy_requirements.json")
    shortcuts = load_json(SMOKE_ROOT / "anti_shortcut_requirements.json")
    next_plan = load_json(SMOKE_ROOT / "next_138yi_milestone_plan.json")
    risk = load_json(SMOKE_ROOT / "risk_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if upstream_138u.get("decision") != "wrong_value_attractor_analysis_complete" or upstream_138u.get("root_cause") != "family_specific_train_value_attractor" or upstream_138u.get("next") != "138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN":
        failures.append("UPSTREAM_138U_ROUTE_MISMATCH")
    if upstream_138u.get("wrong_specific_value_rate") != 1.0 or upstream_138u.get("expected_value_candidate_rate") != 0.0:
        failures.append("UPSTREAM_138U_PROFILE_MISMATCH")
    if not close(upstream_138u.get("generated_values_seen_in_train_rate"), EXPECTED_STRICT_TRAIN_RATE):
        failures.append("STRICT_TRAIN_MEMBERSHIP_MISMATCH")
    for key in ["not_global_train_value_prior", "not_high_frequency_train_value_prior", "not_prompt_copy_parrot_trap"]:
        if upstream_138u.get(key) is not True:
            failures.append("OVERCLAIM_REJECTION_MISSING")
    if upstream_138w.get("helper_integrity_passed") is not True or upstream_138w.get("determinism_replay_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")
    if upstream_138w.get("parrot_trap_detected") is not False or upstream_138w.get("stale_chat_fragment_rate") != 0.0 or upstream_138w.get("train_namespace_leak_rate") != 0.0:
        failures.append("UPSTREAM_138W_PROFILE_MISMATCH")

    if config.get("artifact_only") is not True or config.get("planning_only") is not True or config.get("new_inference_run") is not False or config.get("shared_helper_called") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")
    if summary_report.get("root_cause") != "family_specific_train_value_attractor":
        failures.append("SUMMARY_ROOT_CAUSE_MISMATCH")
    if summary_report.get("coarse_family_routing_appears_present", {}).get("evidence_type") != "inference":
        failures.append("COARSE_ROUTING_EVIDENCE_TAG_MISSING")
    if summary_report.get("value_specific_grounding_is_absent", {}).get("answer_value_accuracy") != 0.0:
        failures.append("VALUE_GROUNDING_ABSENCE_MISSING")
    if summary_report.get("not_global_train_value_prior") is not True or summary_report.get("not_high_frequency_train_value_prior") is not True or summary_report.get("not_prompt_copy_parrot_trap") is not True:
        failures.append("REJECTED_ALTERNATIVE_MISSING")

    if reconciliation.get("upstream_138wv_train_seen_value_label_rate") != 1.0 or not close(reconciliation.get("strict_138u_train_row_membership_rate"), EXPECTED_STRICT_TRAIN_RATE):
        failures.append("TRAIN_MEMBERSHIP_RECONCILIATION_FAILURE")
    if reconciliation.get("global_memorized_lookup_claimed") is not False or reconciliation.get("high_frequency_train_lookup_claimed") is not False:
        failures.append("MEMORIZATION_OVERCLAIM")

    families = collapse.get("families", {})
    if collapse.get("family_count") != 8 or len(families) != 8:
        failures.append("FAMILY_COLLAPSE_REPORT_INCOMPLETE")
    if collapse.get("overall_intra_family_mode_collapse_rate", 0.0) <= 0.0:
        failures.append("INTRA_FAMILY_COLLAPSE_NOT_MEASURED")
    for data in families.values():
        if data.get("per_family_expected_unique_value_count", 0) < 90:
            failures.append("EXPECTED_VALUE_DIVERSITY_MISMATCH")
        if data.get("per_family_correct_unique_value_count") != 0:
            failures.append("CORRECT_VALUE_DIVERSITY_NOT_ZERO")
        if data.get("intra_family_correct_value_diversity_rate") != 0.0:
            failures.append("CORRECT_DIVERSITY_RATE_NOT_ZERO")

    if contrastive.get("core_fix") != "intra_family_contrastive_objective":
        failures.append("CONTRASTIVE_OBJECTIVE_MISSING")
    for metric in ["intra_family_contrastive_accuracy", "family_default_attractor_rate", "per_family_rule_derived_value_accuracy", "per_family_table_derived_value_accuracy", "per_family_ood_symbol_value_accuracy"]:
        if metric not in contrastive.get("required_metrics_for_next_probe", []):
            failures.append("CONTRASTIVE_METRIC_MISSING")
    if scout.get("measured_directly") is not False or scout.get("diagnostic_gap") is not True or scout.get("status") != "design_hypothesis":
        failures.append("SCOUT_HYPOTHESIS_OVERCLAIM")
    if carrier.get("hidden_state_or_graph_carrier_measurement") != "diagnostic_gap" or carrier.get("instrumented_internal_state") is not False:
        failures.append("CARRIER_PROXY_OVERCLAIM")
    for reject in ["family-level format success only", "high-frequency train value replay", "family default value replay", "threshold weakening"]:
        if reject not in shortcuts.get("explicit_rejects", []):
            failures.append("ANTI_SHORTCUT_REJECT_MISSING")
    if next_plan.get("milestone") != "138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE":
        failures.append("NEXT_PLAN_ROUTE_MISMATCH")
    for gate in ["shared_raw_generation_helper.py only", "generated_text before scoring", "deterministic replay", "controls fail", "leakage rejected"]:
        if gate not in next_plan.get("required_integrity_gates", []):
            failures.append("NEXT_PLAN_GATE_MISSING")
    if not risk.get("risks"):
        failures.append("RISK_REGISTER_EMPTY")

    if decision.get("decision") != "family_specific_value_attractor_repair_plan_complete" or decision.get("next") != "138YI_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PROBE":
        failures.append("DECISION_ROUTE_MISMATCH")
    if decision.get("scout_first_laziness_status") != "design_hypothesis_not_measured_mechanism":
        failures.append("SCOUT_DECISION_OVERCLAIM")
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
        print("138YF checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138YF checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
