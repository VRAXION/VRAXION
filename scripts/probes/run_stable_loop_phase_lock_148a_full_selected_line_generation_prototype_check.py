#!/usr/bin/env python3
"""Checker for 148A full selected-line generation prototype."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_148a_full_selected_line_generation_prototype/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_148a_full_selected_line_generation_prototype.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_148a_full_selected_line_generation_prototype_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_147z_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "teacher_trace_manifest.json",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "training_config.json",
    "training_metrics.jsonl",
    "generation_prefix_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "full_line_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "label_distribution_report.json",
    "per_label_generation_report.json",
    "anti_memorization_report.json",
    "ood_generation_family_report.json",
    "ood_split_definition_report.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "value_token_leakage_report.json",
    "model_artifact_audit.json",
    "model_input_audit.json",
    "feature_path_audit.json",
    "deterministic_replay_report.json",
    "evaluation_report.json",
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
EXPECTED_DECISION = "full_selected_line_generation_prototype_positive"
EXPECTED_VERDICT = "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_POSITIVE"
EXPECTED_NEXT = "148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM"
VALID_LINES = {"SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"}
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
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "shared_raw_generation_helper":
            failures.append(f"FORBIDDEN_IMPORT:{path.name}:shared_raw_generation_helper")
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
            if path.name.endswith("_check.py") and name in {"backward", "step", "train", "fit", "forward"}:
                failures.append(f"CHECKER_TRAINING_CALL_NOT_ALLOWED:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["148A", "full SELECTED=<label> line", "canonical structured prompts"]:
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


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_147z_manifest.json")
    decision = upstream.get("decision", {})
    target = upstream.get("target_148a_milestone_plan", {})
    if decision.get("decision") != "full_selected_line_generation_prototype_plan_recommended":
        failures.append(f"BAD_147Z_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "full_selected_line_generation_prototype":
        failures.append(f"BAD_147Z_SELECTED_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE":
        failures.append(f"BAD_147Z_NEXT:{decision.get('next')}")
    if target.get("implementation_ready") is not True:
        failures.append("TARGET_148A_NOT_IMPLEMENTATION_READY")
    generation = target.get("generation_input_policy", {})
    for key, expected in {
        "eval_generation_input_ends_with_output_delimiter": True,
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "model_generates_selected_prefix": True,
        "model_generates_full_selected_line": True,
    }.items():
        if generation.get(key) != expected:
            failures.append(f"TARGET_GENERATION_POLICY_BAD:{key}:{generation.get(key)}")
    raw = target.get("raw_generation_policy", {})
    for key, expected in {
        "raw_generated_text_stored": True,
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
    }.items():
        if raw.get(key) != expected:
            failures.append(f"TARGET_RAW_POLICY_BAD:{key}:{raw.get(key)}")
    decoding = target.get("decoding_policy", {})
    for key, expected in {
        "autoregressive_generation_used": True,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
        "stop_on_newline_or_max_len": True,
    }.items():
        if decoding.get(key) != expected:
            failures.append(f"TARGET_DECODING_POLICY_BAD:{key}:{decoding.get(key)}")


def require_curriculum(root: Path, failures: list[str]) -> None:
    for filename in ["curriculum_train.jsonl", "curriculum_validation.jsonl", "curriculum_test.jsonl", "curriculum_ood_test.jsonl"]:
        rows = load_jsonl(root / filename)
        if not rows:
            failures.append(f"EMPTY_CURRICULUM:{filename}")
        labels = {row.get("selected_pocket_label") for row in rows}
        if labels != {"A", "B", "C", "fallback"}:
            failures.append(f"LABELS_NOT_COMPLETE:{filename}:{sorted(labels)}")
        for row in rows:
            model_input = row.get("model_input", "")
            for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
                if pattern.search(model_input):
                    failures.append(f"FORBIDDEN_MODEL_INPUT:{filename}:{row.get('row_id')}:{pattern.pattern}")
                    break


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


def require_audits(root: Path, failures: list[str]) -> None:
    prefix = load_json(root / "generation_prefix_audit.json")
    raw = load_json(root / "raw_generation_audit.json")
    decoding = load_json(root / "decoding_audit.json")
    schema = load_json(root / "generated_schema_report.json")
    full = load_json(root / "full_line_generation_report.json")
    labels = load_json(root / "label_distribution_report.json")
    replay = load_json(root / "deterministic_replay_report.json")
    baseline = load_json(root / "baseline_margin_report.json")
    shortcut = load_json(root / "shortcut_scanner_report.json")
    leakage = load_json(root / "leakage_audit.json")
    value = load_json(root / "value_token_leakage_report.json")
    artifact = load_json(root / "model_artifact_audit.json")
    for name, payload in [
        ("generation_prefix_audit", prefix),
        ("raw_generation_audit", raw),
        ("decoding_audit", decoding),
        ("generated_schema_report", schema),
        ("full_line_generation_report", full),
        ("label_distribution_report", labels),
        ("deterministic_replay_report", replay),
        ("baseline_margin_report", baseline),
        ("shortcut_scanner_report", shortcut),
        ("leakage_audit", leakage),
        ("value_token_leakage_report", value),
        ("model_artifact_audit", artifact),
        ("model_input_audit", load_json(root / "model_input_audit.json")),
        ("feature_path_audit", load_json(root / "feature_path_audit.json")),
        ("anti_memorization_report", load_json(root / "anti_memorization_report.json")),
        ("ood_generation_family_report", load_json(root / "ood_generation_family_report.json")),
    ]:
        if payload.get("passed") is not True:
            failures.append(f"AUDIT_NOT_PASSED:{name}")
    for key, expected in {
        "eval_generation_input_ends_with_output_delimiter": True,
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "model_generates_selected_prefix": True,
        "model_generates_full_selected_line": True,
    }.items():
        if prefix.get(key) != expected:
            failures.append(f"PREFIX_AUDIT_BAD:{key}:{prefix.get(key)}")
    for key, expected in {
        "raw_generated_text_stored": True,
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
        "label_repair_used": False,
    }.items():
        if raw.get(key) != expected:
            failures.append(f"RAW_AUDIT_BAD:{key}:{raw.get(key)}")
    for key, expected in {
        "autoregressive_generation_used": True,
        "full_selected_line_target_used": True,
        "first_byte_only_training_used": False,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
        "stop_on_newline_or_max_len": True,
    }.items():
        if decoding.get(key) != expected:
            failures.append(f"DECODING_AUDIT_BAD:{key}:{decoding.get(key)}")
    if "max_new_bytes" not in decoding:
        failures.append("DECODING_AUDIT_MISSING_MAX_NEW_BYTES")
    if labels.get("fallback_full_line_accuracy", 0.0) < 0.40 or labels.get("minimum_per_label_full_line_accuracy", 0.0) < 0.40:
        failures.append("LABEL_FULL_LINE_ACCURACY_TOO_LOW")
    if shortcut.get("shortcut_scanner_violation_count") != 0:
        failures.append("SHORTCUT_SCANNER_VIOLATIONS")
    if leakage.get("train_eval_prompt_overlap_count") != 0 or leakage.get("train_ood_prompt_overlap_count") != 0:
        failures.append("PROMPT_LEAKAGE_DETECTED")
    if value.get("value_token_overlap_train_test_rate") != 0.0:
        failures.append("VALUE_TOKEN_LEAKAGE_DETECTED")
    if artifact.get("external_model_or_api_used") is not False or artifact.get("model_download_used") is not False:
        failures.append("EXTERNAL_MODEL_OR_DOWNLOAD_USED")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    expectations = {
        "selected_prefix_generation_accuracy": 0.70,
        "selected_label_generation_accuracy": 0.70,
        "full_selected_line_exact_match_rate": 0.70,
        "selected_label_extracted_from_full_line_accuracy": 0.70,
        "final_value_from_generated_line_accuracy": 0.70,
        "generated_output_schema_valid_rate": 0.80,
        "ood_full_line_accuracy": 0.50,
    }
    for key, threshold in expectations.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_TOO_LOW:{key}:{metrics.get(key)}")
    exact = {
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
        "label_repair_used": False,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "multiple_selected_line_rate": 0.0,
        "shortcut_scanner_violation_count": 0,
        "train_eval_prompt_overlap_count": 0,
        "train_ood_prompt_overlap_count": 0,
        "value_token_overlap_train_test_rate": 0.0,
        "generation_deterministic_replay_passed": True,
        "full_selected_line_target_used": True,
        "first_byte_only_training_used": False,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            failures.append(f"METRIC_EXACT_MISMATCH:{key}:{metrics.get(key)}")
    if metrics.get("extra_text_generation_rate", 1.0) > 0.20:
        failures.append(f"EXTRA_TEXT_RATE_TOO_HIGH:{metrics.get('extra_text_generation_rate')}")
    if metrics.get("shuffled_target_control_accuracy", 1.0) > 0.35:
        failures.append(f"SHUFFLED_TARGET_TOO_HIGH:{metrics.get('shuffled_target_control_accuracy')}")
    if metrics.get("full_line_generation_accuracy", 0.0) < metrics.get("best_baseline_accuracy", 1.0) + 0.10:
        failures.append("BASELINE_MARGIN_TOO_LOW")
    if metrics.get("passed") is not True:
        failures.append("AGGREGATE_METRICS_NOT_PASSED")


def require_raw_outputs(root: Path, failures: list[str]) -> None:
    evaluation = load_json(root / "evaluation_report.json")
    for row in evaluation.get("rows", []):
        raw = row.get("raw_generated_text", "")
        scored = raw[:-1] if raw.endswith("\n") else raw
        if row.get("schema_valid") and scored not in VALID_LINES:
            failures.append(f"SCHEMA_VALID_FROM_INVALID_RAW:{row.get('row_id')}:{raw!r}")
        if row.get("schema_valid") and row.get("raw_generated_text") != row.get("scored_generated_text") + "\n":
            failures.append(f"SCHEMA_NOT_RAW_WITH_TRAILING_NEWLINE:{row.get('row_id')}")
        if "ANSWER=" in raw or "selected_pocket_id" in raw:
            failures.append(f"RAW_OUTPUT_SHORTCUT:{row.get('row_id')}")
    train_corpus = (root / "sequence_train_corpus.txt").read_text(encoding="utf-8")
    if "SELECTED=<label>\\n" in train_corpus:
        failures.append("TRAIN_CORPUS_CONTAINS_PLACEHOLDER_TARGET")
    if "<OUTPUT>\nSELECTED=" not in train_corpus:
        failures.append("TRAIN_CORPUS_MISSING_FULL_SELECTED_LINE_TARGET")


def check(root: Path, skip_changed_files_check: bool) -> list[str]:
    failures: list[str] = []
    if not skip_changed_files_check:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_curriculum(root, failures)
    require_decision(root, failures)
    require_audits(root, failures)
    require_metrics(root, failures)
    require_raw_outputs(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(SMOKE_ROOT), help="148A smoke artifact root")
    parser.add_argument("--check-only", action="store_true", help="Compatibility flag; checker always checks only")
    parser.add_argument("--skip-changed-files-check", action="store_true", help="Allow running while later milestone files are present")
    args = parser.parse_args()
    failures = check(Path(args.root), args.skip_changed_files_check)
    if failures:
        print("148A CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("148A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
