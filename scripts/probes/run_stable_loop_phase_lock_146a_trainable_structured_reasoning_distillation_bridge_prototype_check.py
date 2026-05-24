#!/usr/bin/env python3
"""Checker for 146A trainable structured reasoning distillation bridge prototype."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_145z_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "teacher_trace_manifest.json",
    "training_config.json",
    "model_input_audit.json",
    "value_token_leakage_report.json",
    "dataset_split_audit.json",
    "shortcut_scanner_report.json",
    "baseline_report.json",
    "ablation_report.json",
    "evaluation_report.json",
    "oracle_shortcut_audit.json",
    "model_artifact_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "rule_metadata_reasoning_claimed",
    "natural_language_rule_reasoning_claimed",
    "open_ended_arbitration_claimed",
    "gpt_like_readiness_claimed",
    "gemma_like_capability_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "architecture_superiority_claimed",
]
BOUNDARY_PHRASES = [
    "constrained model-facing distillation evidence only",
    "canonical structured prompts only",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/Gemma-like assistant capability",
    "not production readiness",
    "not architecture superiority",
]
FORBIDDEN_MODEL_INPUT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"selected_pocket_id",
        r"\bwinner\s*=\s*pocket_[abc]\b",
        r"final_selected",
        r"derived_selected",
        r"answer[-_ ]?value",
        r"gold[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"teacher_trace",
        r"per-row oracle metadata",
        r"\bANSWER\s*=",
        r"\bGOLD\s*=",
        r"\bTARGET\s*=",
        r"\bEXPECTED\s*=",
    ]
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_false_flags(payload: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"{prefix}_BOUNDARY_FLAG_NOT_FALSE:{key}:{payload.get(key)}")


def scan_python(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "shared_raw_generation_helper":
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "tensorflow", "requests", "socket", "urllib", "http.client", "shared_raw_generation_helper"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if path.name.endswith("_check.py") and name == "raw_generate":
                failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
            if name in {"raw_generate", "load_checkpoint", "save_checkpoint", "download", "urlopen", "request"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
            if name in {"backward", "step", "forward"}:
                failures.append(f"TRAINING_FRAMEWORK_CALL_NOT_ALLOWED:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["146A", "trainable structured reasoning distillation", "raw canonical structured text"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
        if path.suffix == ".py":
            failures.extend(scan_python(path))
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    if helper_source != git_show_head(HELPER):
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if (root / "progress.jsonl").exists() and (root / "progress.jsonl").stat().st_size == 0:
        failures.append("PROGRESS_JSONL_EMPTY")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_145z_manifest.json")
    decision = upstream.get("decision", {})
    target = upstream.get("target_146a_milestone_plan", {})
    gap = upstream.get("model_facing_bridge_gap_analysis", {})
    if decision.get("decision") != "trainable_structured_reasoning_distillation_bridge_plan_recommended":
        failures.append(f"BAD_145Z_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "trainable_structured_reasoning_distillation_bridge_plan":
        failures.append(f"BAD_145Z_SELECTED_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE":
        failures.append(f"BAD_145Z_NEXT:{decision.get('next')}")
    if target.get("implementation_ready") is not True:
        failures.append("TARGET_146A_NOT_IMPLEMENTATION_READY")
    if gap.get("structured_helper_stack_scale_confirmed") is not True:
        failures.append("UPSTREAM_STACK_NOT_CONFIRMED")
    if gap.get("trainable_model_internalization_untested") is not True:
        failures.append("UPSTREAM_MODEL_INTERNALIZATION_NOT_UNTESTED")
    if gap.get("natural_language_rule_reasoning_untested") is not True:
        failures.append("UPSTREAM_NL_REASONING_NOT_UNTESTED")
    if gap.get("open_ended_arbitration_claimed") is not False:
        failures.append("UPSTREAM_OPEN_ENDED_ARBITRATION_CLAIMED")
    if gap.get("gpt_like_or_gemma_like_capability_claimed") is not False:
        failures.append("UPSTREAM_GPT_GEMMA_CLAIMED")


def require_curriculum(root: Path, failures: list[str]) -> None:
    rows = []
    expected_counts = {"curriculum_train.jsonl": 2400, "curriculum_validation.jsonl": 600, "curriculum_test.jsonl": 600, "curriculum_ood_test.jsonl": 600}
    for filename, expected_count in expected_counts.items():
        split_rows = load_jsonl(root / filename)
        if len(split_rows) != expected_count:
            failures.append(f"BAD_SPLIT_ROW_COUNT:{filename}:{len(split_rows)}")
        rows.extend(split_rows)
    for row in rows:
        model_input = row.get("model_input", "")
        if not isinstance(model_input, str) or "rule_block=" not in model_input:
            failures.append(f"MODEL_INPUT_NOT_RAW_CANONICAL_TEXT:{row.get('row_id')}")
            break
        for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
            if pattern.search(model_input):
                failures.append(f"MODEL_INPUT_FORBIDDEN_FIELD:{row.get('row_id')}:{pattern.pattern}")
                break
    if any(key in rows[0].get("model_input", "") for key in ["parsed_rule_blocks", "final_selected", "selected_pocket_id"]):
        failures.append("MODEL_INPUT_CONTAINS_PARSED_OR_ORACLE_FIELD")


def require_audits(root: Path, failures: list[str]) -> None:
    audit_names = [
        "model_input_audit.json",
        "value_token_leakage_report.json",
        "dataset_split_audit.json",
        "shortcut_scanner_report.json",
        "baseline_report.json",
        "ablation_report.json",
        "evaluation_report.json",
        "oracle_shortcut_audit.json",
        "model_artifact_audit.json",
    ]
    for name in audit_names:
        payload = load_json(root / name)
        if payload.get("passed") is not True:
            failures.append(f"AUDIT_NOT_PASSED:{name}")
    input_audit = load_json(root / "model_input_audit.json")
    if input_audit.get("model_input_is_raw_canonical_text") is not True:
        failures.append("MODEL_INPUT_AUDIT_NOT_RAW_TEXT")
    if input_audit.get("parsed_symbolic_rule_features_passed_to_model") is not False:
        failures.append("PARSED_SYMBOLIC_FEATURES_PASSED_TO_MODEL")
    model_artifact = load_json(root / "model_artifact_audit.json")
    if model_artifact.get("external_api_calls") is not False or model_artifact.get("large_model_download") is not False:
        failures.append("MODEL_ARTIFACT_EXTERNAL_OR_DOWNLOAD")
    if model_artifact.get("manual_symbolic_rule_features_passed_to_model") is not False:
        failures.append("MODEL_ARTIFACT_SYMBOLIC_FEATURES")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    min_thresholds = {
        "teacher_label_reproduction_accuracy": 0.80,
        "selected_pocket_prediction_accuracy": 0.80,
        "final_value_prediction_accuracy": 0.80,
        "final_value_from_predicted_pocket_accuracy": 0.80,
        "heldout_template_accuracy": 0.70,
        "ood_composition_accuracy": 0.60,
        "candidate_value_permutation_accuracy": 0.70,
        "candidate_value_shuffle_consistency": 0.70,
    }
    for key, threshold in min_thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    max_thresholds = {
        "oracle_ablation_accuracy": 0.20,
        "no_priority_ablation_accuracy": 0.35,
        "shuffled_priority_ablation_accuracy": 0.35,
        "no_rule_blocks_ablation_accuracy": 0.35,
        "shuffled_label_control_accuracy": 0.35,
        "test_template_overlap_rate": 0.05,
    }
    for key, threshold in max_thresholds.items():
        if metrics.get(key, 1.0) > threshold:
            failures.append(f"METRIC_ABOVE_GATE:{key}:{metrics.get(key)}")
    zero_metrics = [
        "shortcut_scanner_violation_count",
        "train_validation_leakage_count",
        "value_token_contains_pocket_id_rate",
        "value_token_contains_rule_type_rate",
        "value_token_overlap_train_test_rate",
    ]
    for key in zero_metrics:
        if metrics.get(key) not in {0, 0.0}:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_NOT_TRUE")
    if metrics.get("selected_pocket_prediction_accuracy", 0.0) < metrics.get("best_baseline_accuracy", 1.0) + 0.10:
        failures.append("MODEL_DOES_NOT_BEAT_BASELINE_MARGIN")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    config = load_json(root / "analysis_config.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "trainable_structured_reasoning_distillation_bridge_prototype_positive":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if config.get("raw_generate_allowed") is not False or config.get("external_api_allowed") is not False:
        failures.append("CONFIG_ALLOWS_RAW_GENERATE_OR_EXTERNAL_API")
    if config.get("shared_helper_modification_allowed") is not False:
        failures.append("CONFIG_ALLOWS_HELPER_MODIFICATION")
    for payload_name, payload in [("decision", decision), ("summary", summary), ("config", config)]:
        require_false_flags(payload, failures, payload_name)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_MISSING:{payload_name}:{phrase}")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            failures.append(f"DOC_TOO_SHORT:{doc}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_MISSING:{doc}:{phrase}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_curriculum(root, failures)
    require_audits(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 146A trainable structured reasoning distillation bridge prototype")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("146A CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("146A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
