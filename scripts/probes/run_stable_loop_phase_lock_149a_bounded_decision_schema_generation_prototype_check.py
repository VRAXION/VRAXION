#!/usr/bin/env python3
"""Checker for 149A bounded decision schema generation prototype."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype_check.py"
CONTRACT = "docs/research/STABLE_LOOP_PHASE_LOCK_149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_CONTRACT.md"
RESULT = "docs/research/STABLE_LOOP_PHASE_LOCK_149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_RESULT.md"
ALLOWED_MUTATIONS = {RUNNER, CHECKER, CONTRACT, RESULT}
EXPECTED_DECISION = "bounded_decision_schema_generation_prototype_positive"
EXPECTED_VERDICT = "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE"
EXPECTED_NEXT = "149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM"
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_148z_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "training_config.json",
    "lm_training_metrics.jsonl",
    "bounded_decision_schema_report.json",
    "selected_line_generation_report.json",
    "reason_code_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "schema_prefix_audit.json",
    "raw_schema_generation_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "final_value_copy_report.json",
    "label_distribution_report.json",
    "reason_code_distribution_report.json",
    "reason_code_semantics_report.json",
    "ood_bounded_schema_family_report.json",
    "ood_split_definition_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "value_token_leakage_report.json",
    "feature_path_audit.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
BOUNDARY_PHRASES = [
    "constrained model-facing distillation evidence only",
    "canonical structured prompts only",
    "bounded two-line decision schema generation only",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/Gemma-like assistant capability",
    "not production readiness",
    "not architecture superiority",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status_paths() -> set[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        paths.add(line[3:].replace("\\", "/"))
    return paths


def helper_unchanged() -> bool:
    helper = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
    if not helper.exists():
        return True
    result = subprocess.run(
        ["git", "show", "HEAD:scripts/probes/shared_raw_generation_helper.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout == helper.read_text(encoding="utf-8")


def check_static_source() -> list[str]:
    failures: list[str] = []
    runner_text = (REPO_ROOT / RUNNER).read_text(encoding="utf-8")
    checker_text = (REPO_ROOT / CHECKER).read_text(encoding="utf-8")
    raw_generate_call = "raw_" + "generate("
    torch_import = "import " + "torch"
    if raw_generate_call in runner_text or raw_generate_call in checker_text:
        failures.append("raw_generate call found")
    if "import shared_raw_generation_helper" in runner_text or "from shared_raw_generation_helper" in runner_text:
        failures.append("shared_raw_generation_helper imported by runner")
    if ("torch" + ".") in checker_text or torch_import in checker_text:
        failures.append("checker must not import/use torch")
    tree = ast.parse(checker_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"train", "fit", "backward", "step", "forward"}:
            failures.append("checker appears to train or run model code")
            break
    required_runner_terms = [
        "schema_from_raw",
        "runner_prepends_reason_code",
        "reason_code_extracted_from_substring",
        "full_bounded_schema_target_used",
        "constrained_label_or_reason_only_decoding_used",
    ]
    for term in required_runner_terms:
        if term not in runner_text:
            failures.append(f"runner missing source guardrail term: {term}")
    return failures


def check_artifacts(root: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]


def check_boundary(root: Path) -> list[str]:
    failures: list[str] = []
    docs_and_reports = [
        REPO_ROOT / CONTRACT,
        REPO_ROOT / RESULT,
        root / "report.md",
    ]
    json_docs = [root / "decision.json", root / "summary.json"]
    for path in docs_and_reports:
        text = path.read_text(encoding="utf-8")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"{path.name} missing boundary phrase: {phrase}")
    for path in json_docs:
        payload = load_json(path)
        boundary = payload.get("boundary", "")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in boundary:
                failures.append(f"{path.name} missing boundary phrase: {phrase}")
        for flag in [
            "natural_language_rule_reasoning_claimed",
            "open_ended_arbitration_claimed",
            "gemma_like_capability_claimed",
            "deployment_readiness_claimed",
            "architecture_superiority_claimed",
        ]:
            if payload.get(flag) is not False:
                failures.append(f"{path.name} broad flag not false: {flag}")
    return failures


def check_upstream(root: Path) -> list[str]:
    failures: list[str] = []
    upstream = load_json(root / "upstream_148z_manifest.json")
    decision = upstream.get("decision", {})
    target = upstream.get("target_149a_milestone_plan", {})
    if decision.get("decision") != "bounded_decision_schema_generation_prototype_plan_recommended":
        failures.append("upstream 148Z decision mismatch")
    if decision.get("selected_option") != "bounded_decision_schema_generation_prototype":
        failures.append("upstream 148Z selected_option mismatch")
    if decision.get("next") != "149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE":
        failures.append("upstream 148Z next mismatch")
    if target.get("implementation_ready") is not True:
        failures.append("target_149a implementation_ready not true")
    return failures


def expect(payload: dict[str, Any], key: str, op: str, value: Any, failures: list[str], name: str) -> None:
    actual = payload.get(key)
    ok = False
    if op == ">=":
        ok = actual is not None and actual >= value
    elif op == "<=":
        ok = actual is not None and actual <= value
    elif op == "==":
        ok = actual == value
    elif op == "is":
        ok = actual is value
    if not ok:
        failures.append(f"{name}.{key} expected {op} {value}, got {actual!r}")


def check_metrics(root: Path) -> list[str]:
    failures: list[str] = []
    metrics = load_json(root / "aggregate_metrics.json")
    decision = load_json(root / "decision.json")
    if decision.get("decision") != EXPECTED_DECISION:
        failures.append(f"decision mismatch: {decision.get('decision')}")
    if decision.get("verdict") != EXPECTED_VERDICT:
        failures.append(f"verdict mismatch: {decision.get('verdict')}")
    if decision.get("next") != EXPECTED_NEXT:
        failures.append(f"next mismatch: {decision.get('next')}")
    if decision.get("positive_gate_passed") is not True:
        failures.append("positive gate not passed")
    for key, threshold in [
        ("selected_line_generation_accuracy", 0.70),
        ("reason_code_generation_accuracy", 0.60),
        ("reason_code_semantic_accuracy", 0.60),
        ("selected_reason_pair_exact_match_rate", 0.60),
        ("full_bounded_schema_exact_match_rate", 0.60),
        ("generated_output_schema_valid_rate", 0.75),
        ("final_value_from_generated_schema_accuracy", 0.70),
        ("ood_bounded_schema_accuracy", 0.45),
        ("selected_line_accuracy_over_best_baseline", 0.10),
        ("reason_code_accuracy_over_best_baseline", 0.05),
        ("minimum_per_reason_code_accuracy", 0.35),
    ]:
        expect(metrics, key, ">=", threshold, failures, "aggregate_metrics")
    for key, threshold in [
        ("shuffled_target_control_accuracy", 0.35),
        ("extra_text_generation_rate", 0.20),
    ]:
        expect(metrics, key, "<=", threshold, failures, "aggregate_metrics")
    for key, value in [
        ("answer_value_generation_rate", 0.0),
        ("selected_pocket_id_generation_rate", 0.0),
        ("free_text_reason_generation_rate", 0.0),
        ("shortcut_scanner_violation_count", 0),
        ("train_eval_prompt_overlap_count", 0),
        ("train_ood_prompt_overlap_count", 0),
        ("value_token_overlap_train_test_rate", 0.0),
        ("eval_generation_input_contains_selected_line", False),
        ("eval_generation_input_contains_reason_code", False),
        ("runner_prepends_selected_line", False),
        ("runner_prepends_reason_code", False),
        ("deterministic_schema_wrapper_used", False),
        ("post_generation_repair_used", False),
        ("selected_line_extracted_from_substring", False),
        ("reason_code_extracted_from_substring", False),
        ("casing_repair_used", False),
        ("prefix_repair_used", False),
        ("label_repair_used", False),
        ("reason_code_repair_used", False),
        ("selected_line_only_training_used", False),
        ("constrained_label_or_reason_only_decoding_used", False),
        ("every_reason_code_seen_in_train_validation_test_ood", True),
        ("raw_generated_text_stored", True),
        ("schema_scored_from_raw_generated_text", True),
        ("autoregressive_generation_used", True),
        ("full_bounded_schema_target_used", True),
        ("generation_deterministic_replay_passed", True),
    ]:
        expect(metrics, key, "==", value, failures, "aggregate_metrics")
    if metrics.get("passed") is not True:
        failures.append("aggregate_metrics.passed is not true")
    return failures


def check_reports(root: Path) -> list[str]:
    failures: list[str] = []
    for report_name in [
        "bounded_decision_schema_report.json",
        "reason_code_semantics_report.json",
        "schema_prefix_audit.json",
        "raw_schema_generation_audit.json",
        "generation_input_audit.json",
        "decoding_audit.json",
        "generated_schema_report.json",
        "baseline_margin_report.json",
        "shortcut_scanner_report.json",
        "leakage_audit.json",
        "value_token_leakage_report.json",
        "model_artifact_audit.json",
        "deterministic_replay_report.json",
    ]:
        payload = load_json(root / report_name)
        if payload.get("passed") is not True:
            failures.append(f"{report_name} did not pass")
    train_config = load_json(root / "training_config.json")
    if train_config.get("train_target_sequence") != "SELECTED=<label>\\nREASON_CODE=<bounded_code>\\n":
        failures.append("training_config train_target_sequence mismatch")
    if train_config.get("selected_line_only_training_used") is not False:
        failures.append("training_config selected_line_only_training_used not false")
    if train_config.get("opaque_value_token_generation_required") is not False:
        failures.append("training_config opaque value generation flag not false")
    return failures


def check_changed_files(skip: bool) -> list[str]:
    if skip:
        return []
    changed = git_status_paths()
    extra = sorted(path for path in changed if path not in ALLOWED_MUTATIONS)
    return [f"unexpected changed file: {path}" for path in extra]


def check(root: Path, skip_changed_files_check: bool) -> list[str]:
    failures: list[str] = []
    failures.extend(check_changed_files(skip_changed_files_check))
    if not helper_unchanged():
        failures.append("shared_raw_generation_helper.py changed")
    failures.extend(check_static_source())
    missing = check_artifacts(root)
    failures.extend(f"missing artifact: {name}" for name in missing)
    if missing:
        return failures
    failures.extend(check_upstream(root))
    failures.extend(check_metrics(root))
    failures.extend(check_reports(root))
    failures.extend(check_boundary(root))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 149A bounded decision schema generation prototype")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true", help="Compatibility flag; checker always checks only")
    parser.add_argument("--skip-changed-files-check", action="store_true", help="Allow running while later milestone files are present")
    args = parser.parse_args()
    failures = check(Path(args.root), args.skip_changed_files_check)
    if failures:
        print("149A CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("149A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
