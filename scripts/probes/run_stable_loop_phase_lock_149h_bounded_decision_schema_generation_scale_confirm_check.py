#!/usr/bin/env python3
"""Checker for 149H bounded decision schema scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_149h_bounded_decision_schema_generation_scale_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_149h_bounded_decision_schema_generation_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_149h_bounded_decision_schema_generation_scale_confirm_check.py"
CONTRACT = "docs/research/STABLE_LOOP_PHASE_LOCK_149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM_CONTRACT.md"
RESULT = "docs/research/STABLE_LOOP_PHASE_LOCK_149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM_RESULT.md"
ALLOWED_MUTATIONS = {RUNNER, CHECKER, CONTRACT, RESULT}
ACCEPTED_DECISIONS = {
    "bounded_decision_schema_generation_scale_confirmed",
    "bounded_decision_schema_generation_scale_edge_pocket_routing_bottleneck_confirmed",
}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_149a_manifest.json",
    "compute_probe.json",
    "machine_utilization_report.json",
    "worker_status_report.json",
    "seed_completion_report.json",
    "per_seed_gate_report.json",
    "seed_variance_report.json",
    "row_level_generation_report.jsonl",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "bounded_schema_generation_report.json",
    "hub_routing_diagnostic_report.json",
    "failure_category_report.json",
    "per_seed_failure_category_report.json",
    "selected_label_confusion_matrix.json",
    "reason_code_confusion_matrix.json",
    "selected_reason_pair_confusion_report.json",
    "selected_label_by_reason_code_report.json",
    "selected_label_by_priority_order_report.json",
    "priority_order_holdout_report.json",
    "ood_bounded_schema_family_report.json",
    "schema_prefix_audit.json",
    "raw_schema_generation_audit.json",
    "decoding_audit.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
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
        if line:
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
    raw_generation_call = "raw_" + "generate("
    if raw_generation_call in runner_text or raw_generation_call in checker_text:
        failures.append("raw generation helper call found")
    if "import shared_raw_generation_helper" in runner_text or "from shared_raw_generation_helper" in runner_text:
        failures.append("shared_raw_generation_helper imported by runner")
    torch_attr = "torch" + "."
    torch_import = "import " + "torch"
    if torch_attr in checker_text or torch_import in checker_text:
        failures.append("checker must not import/use torch")
    tree = ast.parse(checker_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"train", "fit", "backward", "step", "forward"}:
            failures.append("checker appears to train or run model code")
            break
    for term in [
        "ProcessPoolExecutor",
        "machine_utilization_report.json",
        "pocket_routing_failure_reason_correct_selected_wrong",
        "schema_scored_from_raw_generated_text",
        "selected_line_only_training_used",
    ]:
        if term not in runner_text:
            failures.append(f"runner missing required term: {term}")
    return failures


def check_artifacts(root: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]


def check_boundary(root: Path) -> list[str]:
    failures: list[str] = []
    for path in [REPO_ROOT / CONTRACT, REPO_ROOT / RESULT, root / "report.md"]:
        text = path.read_text(encoding="utf-8")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"{path.name} missing boundary phrase: {phrase}")
    for path in [root / "decision.json", root / "summary.json"]:
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


def expect(payload: dict[str, Any], key: str, op: str, value: Any, failures: list[str], name: str) -> None:
    actual = payload.get(key)
    if op == ">=":
        ok = actual is not None and actual >= value
    elif op == "<=":
        ok = actual is not None and actual <= value
    elif op == "==":
        ok = actual == value
    elif op == "is":
        ok = actual is value
    else:
        ok = False
    if not ok:
        failures.append(f"{name}.{key} expected {op} {value!r}, got {actual!r}")


def check_upstream(root: Path) -> list[str]:
    failures: list[str] = []
    upstream = load_json(root / "upstream_149a_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "bounded_decision_schema_generation_prototype_positive":
        failures.append("upstream 149A decision mismatch")
    if decision.get("verdict") != "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE":
        failures.append("upstream 149A verdict mismatch")
    if decision.get("next") != "149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM":
        failures.append("upstream 149A next mismatch")
    expect(metrics, "full_bounded_schema_exact_match_rate", ">=", 0.73, failures, "upstream.metrics")
    expect(metrics, "selected_line_generation_accuracy", ">=", 0.74, failures, "upstream.metrics")
    expect(metrics, "reason_code_generation_accuracy", ">=", 0.98, failures, "upstream.metrics")
    expect(metrics, "generated_output_schema_valid_rate", "==", 1.0, failures, "upstream.metrics")
    expect(metrics, "shuffled_target_control_accuracy", "<=", 0.05, failures, "upstream.metrics")
    expect(metrics, "generation_deterministic_replay_passed", "is", True, failures, "upstream.metrics")
    return failures


def check_seed_artifacts(root: Path) -> list[str]:
    failures: list[str] = []
    seed_completion = load_json(root / "seed_completion_report.json")
    if seed_completion.get("all_seeds_completed") is not True:
        failures.append("not all seed workers completed")
    for seed in seed_completion.get("requested_seeds", []):
        seed_root = root / f"seed_{seed}"
        for name in [
            "progress.jsonl",
            "training_metrics_summary.json",
            "partial_eval_report.json",
            "partial_failure_category_report.json",
            "partial_decision.json",
            "aggregate_metrics.json",
        ]:
            if not (seed_root / name).exists():
                failures.append(f"missing seed artifact seed_{seed}/{name}")
    return failures


def check_metrics(root: Path) -> list[str]:
    failures: list[str] = []
    metrics = load_json(root / "aggregate_metrics.json")
    decision = load_json(root / "decision.json")
    failure = load_json(root / "failure_category_report.json")
    hub = load_json(root / "hub_routing_diagnostic_report.json")
    priority = load_json(root / "priority_order_holdout_report.json")
    if decision.get("decision") not in ACCEPTED_DECISIONS:
        failures.append(f"unexpected decision: {decision.get('decision')}")
    if decision.get("next") != "149Z_BOUNDED_DECISION_SCHEMA_GENERATION_NEXT_DECISION_PLAN":
        failures.append(f"unexpected next: {decision.get('next')}")
    if decision.get("positive_gate_passed") is not True:
        failures.append("positive gate not passed")
    for key, threshold in [
        ("generated_output_schema_valid_rate", 0.95),
        ("reason_code_generation_accuracy", 0.90),
        ("reason_code_semantic_accuracy", 0.90),
        ("selected_line_generation_accuracy", 0.70),
        ("minimum_per_label_selected_accuracy", 0.40),
        ("minimum_per_reason_code_accuracy", 0.70),
    ]:
        expect(metrics, key, ">=", threshold, failures, "metrics")
    for key, value in [
        ("eval_generation_input_contains_selected_line", False),
        ("eval_generation_input_contains_reason_code", False),
        ("runner_prepends_selected_line", False),
        ("runner_prepends_reason_code", False),
        ("deterministic_schema_wrapper_used", False),
        ("raw_generated_text_stored", True),
        ("schema_scored_from_raw_generated_text", True),
        ("post_generation_repair_used", False),
        ("selected_line_extracted_from_substring", False),
        ("reason_code_extracted_from_substring", False),
        ("full_bounded_schema_target_used", True),
        ("selected_line_only_training_used", False),
        ("constrained_label_or_reason_only_decoding_used", False),
        ("generation_deterministic_replay_passed", True),
    ]:
        expect(metrics, key, "is", value, failures, "metrics")
    expect(metrics, "shortcut_scanner_violation_count", "==", 0, failures, "metrics")
    expect(metrics, "train_eval_prompt_overlap_count", "==", 0, failures, "metrics")
    expect(metrics, "train_ood_prompt_overlap_count", "==", 0, failures, "metrics")
    expect(metrics, "value_token_overlap_train_test_rate", "==", 0.0, failures, "metrics")
    for key in [
        "priority_order_holdout_selected_accuracy",
        "priority_order_holdout_reason_accuracy",
        "priority_order_holdout_pair_accuracy",
        "block_order_holdout_selected_accuracy",
        "invalid_high_priority_selected_accuracy",
    ]:
        if key not in priority or key not in metrics:
            failures.append(f"missing priority diagnostic: {key}")
    if "pocket_routing_failure_rate" not in failure or "selector_reason_failure_rate" not in failure:
        failures.append("failure category rates missing")
    diagnosis = hub.get("diagnosis", {})
    if diagnosis.get("schema_generation_is_failure_source") is not False:
        failures.append("schema unexpectedly diagnosed as failure source")
    if decision.get("decision") == "bounded_decision_schema_generation_scale_confirmed":
        expect(metrics, "selected_line_generation_accuracy", ">=", 0.80, failures, "strong")
        expect(metrics, "full_bounded_schema_exact_match_rate", ">=", 0.80, failures, "strong")
        expect(metrics, "ood_bounded_schema_accuracy", ">=", 0.70, failures, "strong")
        expect(metrics, "priority_order_holdout_selected_accuracy", ">=", 0.50, failures, "strong")
        expect(metrics, "pocket_routing_failure_rate", "<=", 0.25, failures, "strong")
    if decision.get("decision") == "bounded_decision_schema_generation_scale_edge_pocket_routing_bottleneck_confirmed":
        if not (metrics.get("pocket_routing_failure_rate", 0.0) > metrics.get("selector_reason_failure_rate", 1.0)):
            failures.append("edge route requires pocket routing failure to dominate selector/reason failure")
        if metrics.get("dominant_failure_category") != "pocket_routing_failure_reason_correct_selected_wrong":
            failures.append("edge route requires pocket routing dominant failure category")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 149H bounded decision schema scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    failures: list[str] = []
    dirty = git_status_paths()
    unexpected = dirty - ALLOWED_MUTATIONS
    if unexpected:
        failures.append(f"unexpected dirty tracked paths: {sorted(unexpected)}")
    if not helper_unchanged():
        failures.append("shared_raw_generation_helper.py changed from HEAD")
    failures.extend(check_static_source())
    missing = check_artifacts(root)
    if missing:
        failures.append(f"missing artifacts: {missing}")
    else:
        failures.extend(check_upstream(root))
        failures.extend(check_seed_artifacts(root))
        failures.extend(check_metrics(root))
        failures.extend(check_boundary(root))
    if failures:
        print("149H CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("149H CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
