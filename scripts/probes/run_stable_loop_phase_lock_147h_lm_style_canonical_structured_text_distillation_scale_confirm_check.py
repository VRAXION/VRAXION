#!/usr/bin/env python3
"""Checker for 147H LM-style canonical structured text distillation scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_147h_lm_style_canonical_structured_text_distillation_scale_confirm/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_147h_lm_style_canonical_structured_text_distillation_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_147h_lm_style_canonical_structured_text_distillation_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_147a_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "teacher_trace_manifest.json",
    "training_config.json",
    "lm_training_metrics.jsonl",
    "generation_eval_report.json",
    "selected_label_generation_report.json",
    "final_value_copy_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "label_distribution_report.json",
    "ood_generation_family_report.json",
    "ood_split_definition_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "value_token_leakage_report.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "label_byte_generation_report.json",
    "same_model_family_audit.json",
    "model_input_audit.json",
    "feature_path_audit.json",
    "per_seed_gate_report.json",
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
EXPECTED_DECISION = "lm_style_canonical_structured_text_distillation_scale_confirmed"
EXPECTED_VERDICT = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED"
EXPECTED_NEXT = "147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN"
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
        r"\bANSWER\s*=",
        r"\bGOLD\s*=",
        r"\bTARGET\s*=",
        r"\bEXPECTED\s*=",
        r"^SELECTED=",
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
                if alias.name in {"shared_raw_generation_helper", "requests", "socket", "urllib", "http.client", "tensorflow"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
                if path.name.endswith("_check.py") and alias.name == "torch":
                    failures.append(f"CHECKER_TORCH_IMPORT_NOT_ALLOWED:{path.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"raw_generate", "urlopen", "download"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
            if path.name.endswith("_check.py") and name in {"backward", "step", "train"}:
                failures.append(f"CHECKER_TRAINING_CALL_NOT_ALLOWED:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["147H", "scale confirm", "selected label byte", "canonical structured prompts"]:
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
    progress = root / "progress.jsonl"
    if progress.exists() and progress.stat().st_size == 0:
        failures.append("PROGRESS_JSONL_EMPTY")
    metrics_log = root / "lm_training_metrics.jsonl"
    if metrics_log.exists() and metrics_log.stat().st_size == 0:
        failures.append("LM_TRAINING_METRICS_EMPTY")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_147a_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    expected = {
        "selected_label_generation_accuracy": 1.0,
        "final_value_from_generated_label_accuracy": 1.0,
        "generated_output_schema_valid_rate": 1.0,
        "ood_selected_accuracy": 1.0,
        "shuffled_target_control_accuracy": 0.0,
        "shortcut_scanner_violation_count": 0,
        "generation_deterministic_replay_passed": True,
    }
    if decision.get("decision") != "lm_style_canonical_structured_text_distillation_prototype_positive":
        failures.append(f"BAD_147A_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_147A_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM":
        failures.append(f"BAD_147A_NEXT:{decision.get('next')}")
    for key, value in expected.items():
        if metrics.get(key) != value:
            failures.append(f"UPSTREAM_147A_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    if upstream.get("passed") is not True or upstream.get("failed_checks"):
        failures.append(f"UPSTREAM_147A_FAILED:{upstream.get('failed_checks')}")


def require_curriculum(root: Path, failures: list[str]) -> None:
    expected_counts = {
        "curriculum_train.jsonl": 9600,
        "curriculum_validation.jsonl": 2400,
        "curriculum_test.jsonl": 2400,
        "curriculum_ood_test.jsonl": 2400,
    }
    for filename, expected_count in expected_counts.items():
        rows = load_jsonl(root / filename)
        if len(rows) != expected_count:
            failures.append(f"BAD_SPLIT_ROW_COUNT:{filename}:{len(rows)}")
        labels = {row.get("selected_pocket_label") for row in rows}
        if labels != {"A", "B", "C", "fallback"}:
            failures.append(f"LABELS_NOT_COMPLETE:{filename}:{sorted(labels)}")
        for row in rows[:1000]:
            model_input = row.get("model_input", "")
            for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
                if pattern.search(model_input):
                    failures.append(f"FORBIDDEN_MODEL_INPUT:{filename}:{row.get('row_id')}:{pattern.pattern}")
                    break


def require_audits(root: Path, failures: list[str]) -> None:
    audit_files = [
        "generated_schema_report.json",
        "generation_input_audit.json",
        "label_distribution_report.json",
        "ood_generation_family_report.json",
        "ood_split_definition_report.json",
        "anti_memorization_report.json",
        "baseline_margin_report.json",
        "shuffled_target_control_report.json",
        "shortcut_scanner_report.json",
        "leakage_audit.json",
        "value_token_leakage_report.json",
        "model_artifact_audit.json",
        "deterministic_replay_report.json",
        "label_byte_generation_report.json",
        "same_model_family_audit.json",
        "model_input_audit.json",
        "feature_path_audit.json",
        "per_seed_gate_report.json",
    ]
    for filename in audit_files:
        payload = load_json(root / filename)
        if payload.get("passed") is not True:
            failures.append(f"AUDIT_NOT_PASSED:{filename}")
    label_byte = load_json(root / "label_byte_generation_report.json")
    if label_byte.get("schema_prefix_fixed_by_runner") is not True:
        failures.append("SCHEMA_PREFIX_NOT_FIXED_BY_RUNNER")
    if label_byte.get("selected_line_wrapper_deterministic") is not True:
        failures.append("SELECTED_LINE_WRAPPER_NOT_DETERMINISTIC")
    if label_byte.get("model_generates_full_selected_line") is not False:
        failures.append("FULL_LINE_GENERATION_OVERCLAIMED")
    ood = load_json(root / "ood_generation_family_report.json")
    if ood.get("minimum_ood_family_row_count", 0) < 40:
        failures.append(f"OOD_FAMILY_ROW_COUNT_TOO_LOW:{ood.get('minimum_ood_family_row_count')}")
    same = load_json(root / "same_model_family_audit.json")
    for key in ["same_model_family_as_147a", "same_architecture_as_147a", "same_byte_vocab_size_as_147a", "same_training_objective_as_147a", "no_new_model_architecture", "no_external_model_or_api", "no_pretrained_weights", "cpu_only"]:
        if same.get(key) is not True:
            failures.append(f"SAME_MODEL_AUDIT_BAD:{key}:{same.get(key)}")
    generation_input = load_json(root / "generation_input_audit.json")
    required_false = [
        "eval_generation_input_ends_with_output_delimiter",
        "eval_generation_input_contains_target_selected_label",
        "eval_generation_input_contains_target_label_byte_after_prefix",
        "eval_generation_input_contains_answer_value",
        "eval_generation_input_contains_gold_or_expected",
    ]
    for key in required_false:
        if generation_input.get(key) is not False:
            failures.append(f"GENERATION_INPUT_FALSE_GATE_BAD:{key}:{generation_input.get(key)}")
    if generation_input.get("eval_generation_input_ends_with_selected_prefix") is not True:
        failures.append("GENERATION_INPUT_NOT_SELECTED_PREFIX")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    thresholds = {
        "selected_label_generation_accuracy": 0.85,
        "selected_label_byte_accuracy": 0.85,
        "final_value_from_generated_label_accuracy": 0.85,
        "heldout_template_selected_accuracy": 0.75,
        "ood_selected_accuracy": 0.70,
        "generated_output_schema_valid_rate": 0.95,
        "heldout_priority_order_accuracy": 0.50,
        "heldout_block_order_accuracy": 0.50,
        "heldout_template_accuracy": 0.60,
        "heldout_rule_composition_accuracy": 0.50,
        "minimum_ood_family_accuracy": 0.50,
        "minimum_per_label_generation_accuracy": 0.40,
        "test_margin_over_best_baseline": 0.10,
        "ood_margin_over_best_baseline": 0.05,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_TOO_LOW:{key}:{metrics.get(key)}")
    exact = {
        "multiple_selected_line_rate": 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "extra_text_generation_rate": 0.0,
        "collapsed_ood_family_count": 0,
        "shortcut_scanner_violation_count": 0,
        "train_eval_prompt_overlap_count": 0,
        "train_ood_prompt_overlap_count": 0,
        "normalized_train_eval_prompt_overlap_count": 0,
        "normalized_train_ood_prompt_overlap_count": 0,
        "value_token_overlap_train_test_rate": 0.0,
        "value_token_overlap_train_ood_rate": 0.0,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            failures.append(f"METRIC_EXACT_MISMATCH:{key}:{metrics.get(key)}")
    if metrics.get("malformed_selected_label_rate", 1.0) > 0.05:
        failures.append(f"MALFORMED_LABEL_RATE_HIGH:{metrics.get('malformed_selected_label_rate')}")
    for key in [
        "schema_prefix_fixed_by_runner",
        "selected_line_wrapper_deterministic",
        "train_loss_improves",
        "eval_loss_improves",
        "validation_loss_not_nan",
        "generation_deterministic_replay_passed",
        "every_label_appears_in_every_split",
    ]:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_BOOL_NOT_TRUE:{key}:{metrics.get(key)}")
    if metrics.get("model_generates_full_selected_line") is not False:
        failures.append("MODEL_FULL_LINE_GENERATION_OVERCLAIM")
    if metrics.get("minimum_ood_family_row_count", 0) < 40:
        failures.append("MINIMUM_OOD_FAMILY_ROW_COUNT_TOO_LOW")
    if metrics.get("selected_label_generation_accuracy", 0.0) < metrics.get("best_baseline_accuracy", 1.0) + 0.10:
        failures.append("BASELINE_MARGIN_TOO_LOW")
    if metrics.get("shuffled_target_control_accuracy", 1.0) > 0.35:
        failures.append(f"SHUFFLED_TARGET_CONTROL_TOO_HIGH:{metrics.get('shuffled_target_control_accuracy')}")
    for report in metrics.get("per_seed_reports", []):
        seed = report.get("seed")
        if report.get("selected_label_generation_accuracy", 0.0) < 0.80:
            failures.append(f"PER_SEED_SELECTED_LOW:{seed}")
        if report.get("schema_valid_rate", 0.0) < 0.90:
            failures.append(f"PER_SEED_SCHEMA_LOW:{seed}")
        if report.get("ood_selected_accuracy", 0.0) < 0.60:
            failures.append(f"PER_SEED_OOD_LOW:{seed}")
        if report.get("generation_deterministic_replay_passed") is not True:
            failures.append(f"PER_SEED_REPLAY_FAILED:{seed}")
    if metrics.get("passed") is not True:
        failures.append("AGGREGATE_METRICS_NOT_PASSED")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    config = load_json(root / "analysis_config.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != EXPECTED_DECISION:
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != EXPECTED_VERDICT:
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != EXPECTED_NEXT:
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    for payload_name, payload in [("decision", decision), ("summary", summary), ("config", config)]:
        require_false_flags(payload, failures, payload_name)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_MISSING:{payload_name}:{phrase}")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 147H LM-style canonical structured text distillation scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures: list[str] = []
    if not args.skip_changed_files_check:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if not failures:
        require_upstream(root, failures)
        require_curriculum(root, failures)
        require_audits(root, failures)
        require_metrics(root, failures)
        require_decision(root, failures)
    if failures:
        print("147H CHECK FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("147H CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
