#!/usr/bin/env python3
"""Checker for 146H trainable structured reasoning distillation bridge scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm_check.py"
PHASE_146A_RUNNER = "scripts/probes/run_stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_146a_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "teacher_trace_manifest.json",
    "training_config.json",
    "model_feature_audit.json",
    "feature_path_audit.json",
    "same_model_family_audit.json",
    "baseline_margin_report.json",
    "per_seed_gate_report.json",
    "per_family_ood_report.json",
    "split_stability_report.json",
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


def ast_scan(path: Path, failures: list[str]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"shared_raw_generation_helper", "torch", "tensorflow", "requests", "socket", "urllib", "http.client"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"shared_raw_generation_helper", "torch", "tensorflow", "requests", "socket", "urllib", "http.client"}:
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if path.name.endswith("_check.py") and name == "raw_generate":
                failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
            if name in {"raw_generate", "download", "urlopen", "request", "load_checkpoint", "save_checkpoint"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["146H", "trainable structured reasoning distillation", "raw canonical structured text"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
        if path.suffix == ".py":
            ast_scan(path, failures)
    if (REPO_ROOT / HELPER).read_text(encoding="utf-8") != git_show_head(HELPER):
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")
    phase_146a = (REPO_ROOT / PHASE_146A_RUNNER).read_text(encoding="utf-8")
    if "def token_features(text: str" not in phase_146a:
        failures.append("TOKEN_FEATURES_FUNCTION_MISSING_IN_146A")
    if "token_features(row[\"model_input\"]" not in phase_146a:
        failures.append("146A_FEATURE_EXTRACTOR_NOT_MODEL_INPUT_ONLY")
    if "predict_one(row[input_key])" not in phase_146a:
        failures.append("146A_PREDICT_PATH_NOT_INPUT_KEY_BASED")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if (root / "progress.jsonl").exists() and (root / "progress.jsonl").stat().st_size == 0:
        failures.append("PROGRESS_JSONL_EMPTY")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_146a_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "trainable_structured_reasoning_distillation_bridge_prototype_positive":
        failures.append(f"BAD_146A_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_146A_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM":
        failures.append(f"BAD_146A_NEXT:{decision.get('next')}")
    exact_gates = {
        "selected_pocket_prediction_accuracy": metrics.get("selected_pocket_prediction_accuracy", 0.0) >= 0.90,
        "final_value_from_predicted_pocket_accuracy": metrics.get("final_value_from_predicted_pocket_accuracy", 0.0) >= 0.90,
        "heldout_template_accuracy": metrics.get("heldout_template_accuracy") == 1.0,
        "ood_composition_accuracy": metrics.get("ood_composition_accuracy", 0.0) >= 0.70,
        "shortcut_scanner_violation_count": metrics.get("shortcut_scanner_violation_count") == 0,
        "train_validation_leakage_count": metrics.get("train_validation_leakage_count") == 0,
        "value_token_overlap_train_test_rate": metrics.get("value_token_overlap_train_test_rate") == 0.0,
        "deterministic_replay_passed": metrics.get("deterministic_replay_passed") is True,
    }
    for key, passed in exact_gates.items():
        if not passed:
            failures.append(f"UPSTREAM_146A_GATE_FAILED:{key}:{metrics.get(key)}")


def require_curriculum(root: Path, failures: list[str]) -> None:
    expected_counts = {
        "curriculum_train.jsonl": 9600,
        "curriculum_validation.jsonl": 2400,
        "curriculum_test.jsonl": 2400,
        "curriculum_ood_test.jsonl": 2400,
    }
    all_rows = []
    for filename, expected_count in expected_counts.items():
        rows = load_jsonl(root / filename)
        if len(rows) != expected_count:
            failures.append(f"BAD_SPLIT_ROW_COUNT:{filename}:{len(rows)}")
        all_rows.extend(rows)
    for row in all_rows:
        model_input = row.get("model_input", "")
        if not isinstance(model_input, str) or "priority=" not in model_input:
            failures.append(f"MODEL_INPUT_NOT_RAW_CANONICAL_TEXT:{row.get('row_id')}")
            break
        for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
            if pattern.search(model_input):
                failures.append(f"MODEL_INPUT_FORBIDDEN_FIELD:{row.get('row_id')}:{pattern.pattern}")
                return


def require_feature_audits(root: Path, failures: list[str]) -> None:
    model_feature = load_json(root / "model_feature_audit.json")
    feature_path = load_json(root / "feature_path_audit.json")
    same_model = load_json(root / "same_model_family_audit.json")
    expected_false = [
        "parsed_rule_features_used",
        "teacher_trace_features_used",
        "selected_pocket_oracle_features_used",
        "answer_or_value_label_features_used",
    ]
    if model_feature.get("uses_raw_text_ngram_features_only") is not True:
        failures.append("MODEL_FEATURES_NOT_RAW_NGRAM_ONLY")
    for key in expected_false:
        if model_feature.get(key) is not False:
            failures.append(f"MODEL_FEATURE_AUDIT_NOT_FALSE:{key}:{model_feature.get(key)}")
    required_feature_path = {
        "feature_extractor_function_name": "token_features",
        "feature_extractor_input_field": "model_input",
        "feature_extractor_uses_only_model_input": True,
        "feature_extractor_reads_teacher_trace": False,
        "feature_extractor_reads_selected_pocket_label": False,
        "feature_extractor_reads_final_value_label": False,
        "feature_extractor_reads_candidate_values_as_labels": False,
        "train_X_source_field": "model_input",
        "validation_X_source_field": "model_input",
        "test_X_source_field": "model_input",
        "ood_X_source_field": "model_input",
    }
    for key, expected in required_feature_path.items():
        if feature_path.get(key) != expected:
            failures.append(f"FEATURE_PATH_AUDIT_BAD:{key}:{feature_path.get(key)}")
    required_same_model = {
        "same_model_family_as_146a": True,
        "same_raw_text_ngram_feature_policy_as_146a": True,
        "new_model_architecture_introduced": False,
        "external_model_or_api_used": False,
    }
    for key, expected in required_same_model.items():
        if same_model.get(key) != expected:
            failures.append(f"SAME_MODEL_AUDIT_BAD:{key}:{same_model.get(key)}")
    for name in ["model_feature_audit.json", "feature_path_audit.json", "same_model_family_audit.json"]:
        if load_json(root / name).get("passed") is not True:
            failures.append(f"AUDIT_NOT_PASSED:{name}")


def require_reports(root: Path, failures: list[str]) -> None:
    report_names = [
        "split_stability_report.json",
        "model_input_audit.json",
        "value_token_leakage_report.json",
        "dataset_split_audit.json",
        "shortcut_scanner_report.json",
        "baseline_report.json",
        "ablation_report.json",
        "evaluation_report.json",
        "oracle_shortcut_audit.json",
        "model_artifact_audit.json",
        "baseline_margin_report.json",
        "per_seed_gate_report.json",
        "per_family_ood_report.json",
    ]
    for name in report_names:
        payload = load_json(root / name)
        if payload.get("passed") is not True:
            failures.append(f"REPORT_NOT_PASSED:{name}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    min_thresholds = {
        "selected_pocket_prediction_accuracy": 0.88,
        "final_value_from_predicted_pocket_accuracy": 0.88,
        "heldout_template_accuracy": 0.85,
        "ood_composition_accuracy": 0.70,
        "minimum_ood_family_accuracy": 0.50,
        "margin_over_best_baseline": 0.15,
        "test_margin_over_best_baseline": 0.15,
        "ood_margin_over_best_baseline": 0.10,
    }
    for key, threshold in min_thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    max_thresholds = {
        "shuffled_label_control_accuracy": 0.35,
        "test_template_overlap_rate": 0.05,
        "oracle_ablation_accuracy": 0.20,
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
        "value_token_overlap_train_ood_rate",
        "collapsed_ood_family_count",
    ]
    for key in zero_metrics:
        if metrics.get(key) not in {0, 0.0}:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_NOT_TRUE")
    if metrics.get("passed") is not True:
        failures.append("AGGREGATE_METRICS_NOT_PASSED")


def require_per_seed_and_family(root: Path, failures: list[str]) -> None:
    per_seed = load_json(root / "per_seed_gate_report.json")
    if per_seed.get("seed_count") != 4:
        failures.append(f"BAD_SEED_COUNT:{per_seed.get('seed_count')}")
    for item in per_seed.get("seeds", []):
        if item.get("selected_pocket_prediction_accuracy", 0.0) < 0.85:
            failures.append(f"PER_SEED_SELECTED_BELOW_GATE:{item.get('seed')}:{item.get('selected_pocket_prediction_accuracy')}")
        if item.get("final_value_from_predicted_pocket_accuracy", 0.0) < 0.85:
            failures.append(f"PER_SEED_FINAL_BELOW_GATE:{item.get('seed')}:{item.get('final_value_from_predicted_pocket_accuracy')}")
        if item.get("ood_composition_accuracy", 0.0) < 0.65:
            failures.append(f"PER_SEED_OOD_BELOW_GATE:{item.get('seed')}:{item.get('ood_composition_accuracy')}")
        if item.get("margin_over_best_baseline", 0.0) < 0.10:
            failures.append(f"PER_SEED_MARGIN_BELOW_GATE:{item.get('seed')}:{item.get('margin_over_best_baseline')}")
        if item.get("shortcut_scanner_violation_count") != 0:
            failures.append(f"PER_SEED_SHORTCUT_VIOLATION:{item.get('seed')}")
        if item.get("value_token_overlap_train_test_rate") != 0.0:
            failures.append(f"PER_SEED_VALUE_OVERLAP:{item.get('seed')}")
        if not item.get("ood_family_breakdown"):
            failures.append(f"PER_SEED_OOD_FAMILY_MISSING:{item.get('seed')}")
    per_family = load_json(root / "per_family_ood_report.json")
    if per_family.get("minimum_ood_family_accuracy", 0.0) < 0.50:
        failures.append(f"MINIMUM_OOD_FAMILY_BELOW_GATE:{per_family.get('minimum_ood_family_accuracy')}")
    if per_family.get("collapsed_ood_family_count") != 0:
        failures.append(f"COLLAPSED_OOD_FAMILY_COUNT:{per_family.get('collapsed_ood_family_count')}")
    if per_family.get("no_ood_family_below_minimum") is not True:
        failures.append("OOD_FAMILY_BELOW_MINIMUM")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    config = load_json(root / "analysis_config.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "trainable_structured_reasoning_distillation_bridge_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if config.get("raw_generate_allowed") is not False or config.get("external_api_allowed") is not False:
        failures.append("CONFIG_ALLOWS_RAW_GENERATE_OR_EXTERNAL_API")
    if config.get("shared_helper_modification_allowed") is not False:
        failures.append("CONFIG_ALLOWS_HELPER_MODIFICATION")
    if config.get("new_model_architecture_allowed") is not False:
        failures.append("CONFIG_ALLOWS_NEW_MODEL_ARCHITECTURE")
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
    require_feature_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_per_seed_and_family(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 146H trainable structured reasoning distillation bridge scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("146H CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("146H CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
