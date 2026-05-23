#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138YH high-frequency replay analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yh_high_frequency_value_replay_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yh_high_frequency_value_replay_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yh_high_frequency_value_replay_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yi_manifest.json",
    "upstream_138yf_manifest.json",
    "analysis_config.json",
    "replay_value_extraction_report.json",
    "train_value_frequency_report.json",
    "replay_rank_report.json",
    "family_replay_shape_report.json",
    "contrast_group_replay_report.json",
    "objective_reward_artifact_report.json",
    "scorer_dataset_artifact_report.json",
    "root_cause_report.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS",
    "high_frequency_value_replay_analysis_complete",
    "global_high_frequency_train_value_replay",
    "family_local_high_frequency_value_replay",
    "family_default_shortcut_replay",
    "same_value_for_all_rows_collapse",
    "objective_missing_frequency_penalty",
    "dataset_low_value_diversity_artifact",
    "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
    "TR value replay",
    "ANSWER=T namespace leakage",
    "strict train membership",
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
ROOTS = {
    "global_high_frequency_train_value_replay",
    "family_local_high_frequency_value_replay",
    "family_default_shortcut_replay",
    "same_value_for_all_rows_collapse",
    "objective_missing_frequency_penalty",
    "dataset_low_value_diversity_artifact",
    "mixed_high_frequency_replay",
    "high_frequency_replay_ambiguous",
}
ROUTES = {
    "global_high_frequency_train_value_replay": "138YHG_GLOBAL_VALUE_FREQUENCY_PENALTY_PLAN",
    "family_local_high_frequency_value_replay": "138YHL_FAMILY_LOCAL_FREQUENCY_PENALTY_PLAN",
    "family_default_shortcut_replay": "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS",
    "same_value_for_all_rows_collapse": "138YS_SAME_VALUE_COLLAPSE_ANALYSIS",
    "dataset_low_value_diversity_artifact": "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN",
    "objective_missing_frequency_penalty": "138YJ_FREQUENCY_SUPPRESSED_INTRA_FAMILY_OBJECTIVE_PLAN",
    "mixed_high_frequency_replay": "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET",
    "high_frequency_replay_ambiguous": "138YHB_HIGH_FREQUENCY_REPLAY_MANUAL_REVIEW_PACKET",
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
    for event in ["startup", "upstream verification", "value extraction", "train frequency analysis", "replay rank analysis", "family replay shape", "contrast group replay", "root cause selection", "decision", "final verdict"]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    upstream = load_json(SMOKE_ROOT / "upstream_138yi_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    extraction = load_json(SMOKE_ROOT / "replay_value_extraction_report.json")
    train_freq = load_json(SMOKE_ROOT / "train_value_frequency_report.json")
    ranks = load_json(SMOKE_ROOT / "replay_rank_report.json")
    family = load_json(SMOKE_ROOT / "family_replay_shape_report.json")
    contrast = load_json(SMOKE_ROOT / "contrast_group_replay_report.json")
    objective = load_json(SMOKE_ROOT / "objective_reward_artifact_report.json")
    dataset = load_json(SMOKE_ROOT / "scorer_dataset_artifact_report.json")
    root = load_json(SMOKE_ROOT / "root_cause_report.json")
    recommendation = load_json(SMOKE_ROOT / "next_repair_recommendation.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if upstream.get("decision") != "high_frequency_train_value_replay_detected" or upstream.get("next") != "138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS":
        failures.append("UPSTREAM_138YI_ROUTE_MISMATCH")
    if upstream.get("high_frequency_train_value_replay_detected") is not True or upstream.get("family_default_shortcut_detected") is not True:
        failures.append("UPSTREAM_138YI_PROFILE_MISMATCH")
    for key in ["canary", "ast", "leakage", "controls", "determinism"]:
        if key == "canary" and upstream.get("verified") is not True:
            failures.append("UPSTREAM_138YI_NOT_VERIFIED")
    if upstream.get("train_namespace_leak_rate") != 0.0:
        failures.append("ANSWER_T_LEAK_CONFUSED_WITH_TR_REPLAY")

    for key in ["artifact_only", "training_performed", "new_model_inference_run", "shared_helper_called", "torch_forward_pass_run", "checkpoint_mutation_performed"]:
        expected = True if key == "artifact_only" else False
        if config.get(key) is not expected:
            failures.append(f"ARTIFACT_ONLY_BOUNDARY_FAILURE:{key}")

    if extraction.get("row_count") != 768 or len(extraction.get("rows", [])) != 768:
        failures.append("REPLAY_EXTRACTION_ROW_COUNT_MISMATCH")
    if extraction.get("tr_prefix_replay_rate") != 1.0 or extraction.get("ev_expected_candidate_rate") != 0.0:
        failures.append("REPLAY_EXTRACTION_PROFILE_MISMATCH")
    required_row_fields = {"row_id", "family", "seed", "contrast_group_id", "expected_value", "generated_value", "generated_text", "prompt", "helper_trace_hash", "pass_fail", "failure_reason", "generated_value_source"}
    if any(not required_row_fields.issubset(row) for row in extraction.get("rows", [])):
        failures.append("REPLAY_EXTRACTION_ROW_FIELD_MISSING")

    for key in ["generated_values_seen_in_train_expected_rate", "generated_values_seen_in_train_prompt_rate", "generated_values_seen_in_train_all_rate", "generated_value_membership"]:
        if key not in train_freq:
            failures.append(f"TRAIN_FREQUENCY_FIELD_MISSING:{key}")
    for key in [
        "generated_values_top1_global_train_all_rate",
        "generated_values_top5_global_train_all_rate",
        "generated_values_top10_global_train_all_rate",
        "generated_values_top1_family_train_all_rate",
        "generated_values_top5_family_train_all_rate",
        "generated_values_top10_family_train_all_rate",
    ]:
        if key not in ranks:
            failures.append(f"REPLAY_RANK_FIELD_MISSING:{key}")
    if family.get("family_count") != 8:
        failures.append("FAMILY_REPLAY_SHAPE_INCOMPLETE")
    if contrast.get("group_count") != 192:
        failures.append("CONTRAST_GROUP_COUNT_MISMATCH")
    if objective.get("positive_can_depend_on_train_loss") is not False:
        failures.append("TRAIN_LOSS_SUCCESS_OVERCLAIM")
    if not objective.get("diagnostic_gap"):
        failures.append("OBJECTIVE_DIAGNOSTIC_GAP_MISSING")
    if dataset.get("train_eval_value_overlap") != 0:
        failures.append("TRAIN_EVAL_VALUE_OVERLAP")

    if root.get("root_cause") not in ROOTS:
        failures.append("ROOT_CAUSE_UNKNOWN")
    if recommendation.get("recommended_next") != ROUTES.get(root.get("root_cause")) or decision.get("next") != recommendation.get("recommended_next"):
        failures.append("RECOMMENDATION_ROUTE_MISMATCH")
    if root.get("root_cause") == "family_default_shortcut_replay":
        if root.get("family_default_attractor_rate", 0.0) < 0.75:
            failures.append("FAMILY_DEFAULT_ROOT_UNSUPPORTED")
    if root.get("root_cause") == "global_high_frequency_train_value_replay" and ranks.get("generated_values_top1_global_train_all_rate", 0.0) < 0.50:
        failures.append("GLOBAL_ROOT_UNSUPPORTED")
    if decision.get("decision") != "high_frequency_value_replay_analysis_complete":
        failures.append("DECISION_MISMATCH")
    if not any(gap.get("field") == "output_head_prior" and gap.get("status") == "diagnostic_gap" for gap in gaps.get("gaps", [])):
        failures.append("OUTPUT_HEAD_DIAGNOSTIC_GAP_MISSING")

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
        print("138YH checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138YH checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
