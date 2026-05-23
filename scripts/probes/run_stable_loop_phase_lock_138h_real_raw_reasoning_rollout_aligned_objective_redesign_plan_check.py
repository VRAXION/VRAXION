#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138H objective redesign plan."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138h_real_raw_reasoning_rollout_aligned_objective_redesign_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138h_real_raw_reasoning_rollout_aligned_objective_redesign_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138h_real_raw_reasoning_rollout_aligned_objective_redesign_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138ga_manifest.json",
    "upstream_138g_manifest.json",
    "upstream_138r_manifest.json",
    "rollout_alignment_failure_summary.json",
    "objective_redesign_requirements.json",
    "train_eval_namespace_policy.json",
    "rollout_aligned_training_design.json",
    "final_eval_gate_design.json",
    "anti_shortcut_requirements.json",
    "next_138i_milestone_plan.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN",
    "planning-only",
    "train_namespace_rollout_alignment_failure",
    "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE",
    "shared_raw_generation_helper.py",
    "generated_text before scoring",
    "teacher-forcing-only success",
    "loss-only success",
    "ANSWER=T",
    "ANSWER=E",
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
    forbidden_calls = {
        "raw_generate",
        "manual_seed",
        "forward",
        "backward",
        "optimizer",
        "Start-Process",
    }
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


def is_tagged(item: Any) -> bool:
    return isinstance(item, dict) and item.get("evidence_type") in TAG_TYPES and "source" in item and "value" in item


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    manifest_ga = load_json(SMOKE_ROOT / "upstream_138ga_manifest.json")
    if manifest_ga.get("upstream_138ga_verified") is not True:
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")
    if manifest_ga.get("decision") != "objective_failure_disambiguated":
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")
    if manifest_ga.get("next") != "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN":
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")
    if manifest_ga.get("near_match_row_count") != 38 or manifest_ga.get("total_scored_row_count") != 960:
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")
    if manifest_ga.get("primary_label_counts") != {"train_namespace_overlap": 38}:
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")
    if manifest_ga.get("meaningful_near_match_rate") != 0.0:
        failures.append("UPSTREAM_138GA_ARTIFACT_MISSING")

    manifest_g = load_json(SMOKE_ROOT / "upstream_138g_manifest.json")
    if manifest_g.get("upstream_138g_verified") is not True or manifest_g.get("decision") != "objective_failure_ambiguous":
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_g.get("teacher_forced_loss_fields_diagnostic_gap") is not True:
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")

    manifest_r = load_json(SMOKE_ROOT / "upstream_138r_manifest.json")
    if manifest_r.get("upstream_138r_verified") is not True:
        failures.append("UPSTREAM_138R_ARTIFACT_MISSING")
    if manifest_r.get("mean_real_raw_reasoning_accuracy") != 0.0:
        failures.append("UPSTREAM_138R_ARTIFACT_MISSING")
    if manifest_r.get("expected_token_inclusion_rate") != 0.0:
        failures.append("UPSTREAM_138R_ARTIFACT_MISSING")
    if manifest_r.get("helper_canary_ast_leakage_controls_determinism_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")

    failure_summary = load_json(SMOKE_ROOT / "rollout_alignment_failure_summary.json")
    for key in [
        "primary_bottleneck",
        "near_match_disambiguation",
        "meaningful_near_match_rate",
        "rollout_accuracy",
        "expected_token_inclusion_rate",
        "teacher_forced_loss_fields",
        "train_loss_decreased",
        "interpretation",
    ]:
        if not is_tagged(failure_summary.get(key)):
            failures.append(f"EVIDENCE_TAG_MISSING:{key}")
    if failure_summary["primary_bottleneck"]["value"] != "train_namespace_rollout_alignment_failure":
        failures.append("PRIMARY_BOTTLENECK_MISMATCH")

    requirements = load_json(SMOKE_ROOT / "objective_redesign_requirements.json")
    for item in ["train namespace replay", "helper-only autoregressive free rollout", "loss-only success rejection", "teacher-forcing-only success rejection"]:
        if item not in requirements.get("must_target", []):
            failures.append("OBJECTIVE_REDESIGN_REQUIREMENTS_INCOMPLETE")
    for item in ["teacher-forcing-only success", "loss-only success", "threshold weakening", "expected-output construction", "post-generation repair"]:
        if item not in requirements.get("must_not_optimize_for", []):
            failures.append("OBJECTIVE_REDESIGN_REQUIREMENTS_INCOMPLETE")

    namespace = load_json(SMOKE_ROOT / "train_eval_namespace_policy.json")
    required_metrics = {
        "train_namespace_leak_rate",
        "eval_namespace_emission_accuracy",
        "answer_prefix_accuracy",
        "answer_value_accuracy",
        "stale_chat_fragment_rate",
        "off_prompt_output_rate",
    }
    if namespace.get("train_namespace") != "ANSWER=T..." or namespace.get("eval_namespace") != "ANSWER=E...":
        failures.append("NAMESPACE_POLICY_INCOMPLETE")
    if not required_metrics.issubset(set(namespace.get("required_metrics_for_138i", []))):
        failures.append("NAMESPACE_POLICY_INCOMPLETE")

    design = load_json(SMOKE_ROOT / "rollout_aligned_training_design.json")
    names = {item.get("name") for item in design.get("objective_components", [])}
    if not {"output_namespace_alignment", "free_rollout_alignment", "scoring_format_discipline"}.issubset(names):
        failures.append("ROLLOUT_ALIGNED_TRAINING_DESIGN_INCOMPLETE")
    if design.get("source_checkpoint_immutable") is not True or design.get("target_checkpoint_under_target_only") is not True:
        failures.append("ROLLOUT_ALIGNED_TRAINING_DESIGN_INCOMPLETE")

    gates = load_json(SMOKE_ROOT / "final_eval_gate_design.json")
    gate_flags = [
        "shared_raw_generation_helper_only",
        "generated_text_before_scoring",
        "helper_request_contains_no_expected_or_scorer_metadata",
        "expected_output_canary_required",
        "ast_shortcut_scan_required",
        "deterministic_replay_required",
        "controls_must_fail",
        "leakage_rejected_required",
        "source_checkpoint_unchanged_required",
        "target_checkpoint_under_target_only",
    ]
    for key in gate_flags:
        if gates.get(key) is not True:
            failures.append("FINAL_EVAL_GATE_DESIGN_INCOMPLETE")
    routes = gates.get("clean_negative_routes", {})
    if routes.get("no_rollout_improvement", {}).get("next") != "138I_FAILURE_ANALYSIS":
        failures.append("FINAL_EVAL_GATE_DESIGN_INCOMPLETE")
    if routes.get("namespace_leak_persists", {}).get("next") != "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS":
        failures.append("FINAL_EVAL_GATE_DESIGN_INCOMPLETE")

    anti = load_json(SMOKE_ROOT / "anti_shortcut_requirements.json")
    required_rejects = {
        "teacher-forcing-only success",
        "loss-only success",
        "threshold weakening",
        "expected-output construction",
        "old runner imports",
        "helper/backend modification to improve score",
        "oracle/rerank/verifier/LLM judge",
        "constrained decoding",
        "JSON mode",
        "regex fixer",
        "post-generation repair",
        "retry loop",
        "best-of-n",
    }
    if not required_rejects.issubset(set(anti.get("explicitly_reject", []))):
        failures.append("ANTI_SHORTCUT_REQUIREMENTS_INCOMPLETE")

    plan = load_json(SMOKE_ROOT / "next_138i_milestone_plan.json")
    if plan.get("milestone") != "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE":
        failures.append("NEXT_138I_PLAN_INCOMPLETE")
    if plan.get("primary_bottleneck") != "train_namespace_rollout_alignment_failure":
        failures.append("NEXT_138I_PLAN_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "rollout_aligned_objective_redesign_plan_complete":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE":
        failures.append("DECISION_MISMATCH")
    if decision.get("primary_bottleneck") != "train_namespace_rollout_alignment_failure":
        failures.append("PRIMARY_BOTTLENECK_MISMATCH")
    if decision.get("no_capability_restored") is not True:
        failures.append("BOUNDARY_CLAIM_FAILURE")
    if not isinstance(decision.get("evidence_summary"), dict) or not isinstance(decision.get("required_138i_gates"), list):
        failures.append("DECISION_MACHINE_READABILITY_FAILURE")
    if decision.get("artifact_only_planning") is not True:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    if decision.get("training_performed") is not False or decision.get("new_model_inference_run") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    if decision.get("shared_helper_called") is not False or decision.get("torch_forward_pass_run") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")

    summary = load_json(SMOKE_ROOT / "summary.json")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    for key in FALSE_FLAGS:
        if decision.get(key) is not False or summary.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")
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
        print("138H checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138H checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
