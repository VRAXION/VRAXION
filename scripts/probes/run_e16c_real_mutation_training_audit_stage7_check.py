#!/usr/bin/env python3
"""Checker for E16C real mutation-training audit Stage 7."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e16c_real_mutation_training_audit_stage7.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E16C_REAL_MUTATION_TRAINING_AUDIT_STAGE7_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E16C_REAL_MUTATION_TRAINING_AUDIT_STAGE7_RESULT.md",
)
PRIMARY = "REAL_MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY"
FIXTURE_REFERENCE = "E16C_STATIC_REPAIR_FIXTURE_REFERENCE"
BASELINES = (
    "LAST_WRITE_MEMORY_NO_GATE",
    "VALID_LAST_MEMORY",
    "MAJORITY_MEMORY_NO_ABSTAIN",
    "FIXED_SLOT_FIFO_MEMORY",
    "FIXED_SLOT_LRU_MEMORY",
    "KEY_ADDRESSED_MEMORY_POLICY",
)
REQUIRED_SYSTEMS = (
    FIXTURE_REFERENCE,
    *BASELINES,
    "REAL_MUTATION_TRAINED_MEMORY_POLICY",
    PRIMARY,
    "NO_MEMORY_SLOTS_ABLATION",
    "LOW_MEMORY_CAPACITY_ABLATION",
    "NO_STALE_REJECTION_ABLATION",
    "NO_REPAIR_EVIDENCE_ABLATION",
    "NO_AMBIGUITY_ABSTAIN_ABLATION",
    "NO_NESTED_RESOLUTION_ABLATION",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e16c_real_search_report.json",
    "e16c_real_episode_generation_report.json",
    "e16c_real_train_episode_manifest.json",
    "e16c_real_validation_episode_manifest.json",
    "e16c_real_heldout_episode_manifest.json",
    "e16c_real_candidate_population_report.json",
    "e16c_real_generation_score_report.json",
    "e16c_real_training_curve_report.json",
    "e16c_real_best_policy_report.json",
    "e16c_real_pruned_policy_report.json",
    "e16c_real_per_episode_eval_report.json",
    "e16c_real_capacity_sweep_report.json",
    "e16c_real_system_comparison_report.json",
    "e16c_real_ablation_report.json",
    "e16c_real_trace_validity_report.json",
    "e16c_real_writeback_safety_report.json",
    "e16c_real_heldout_generalization_report.json",
    "e16c_real_static_fixture_audit_report.json",
    "e16c_real_semantic_macro_leak_audit_report.json",
    "e16c_real_deterministic_replay_report.json",
    "e16c_real_boundary_claims_report.json",
    "e16c_real_next_recommendation.json",
)
VALID_DECISIONS = (
    "e16c_real_mutation_training_stage7_confirmed",
    "e16c_real_mutation_training_stage7_partial",
    "e16c_real_mutation_training_stage7_failed",
    "e16c_real_mutation_training_stage7_invalid_or_incomplete",
)
FAMILIES = (
    "SINGLE_BIND_DELAYED_QUERY",
    "MULTI_BIND_DELAYED_QUERY",
    "NESTED_BINDING_DEPTH2",
    "NESTED_BINDING_DEPTH3",
    "CAPACITY_PRESSURE",
    "STALE_UPDATE_REJECTION",
    "CORRUPT_THEN_REPAIR",
    "AMBIGUOUS_EVIDENCE_ABSTAIN_OR_REPAIR",
    "DISTRACTOR_GAP",
    "MIXED_MEMORY_AND_TEMPLATE",
)
BOUNDARY_CONFIRMED = (
    "This confirms real deterministic mutation/search training over Stage 7 memory policies in a controlled synthetic "
    "text-flow proxy. It does not prove general natural-language AI or production training readiness."
)
BOUNDARY_GAP = (
    "This run maps the gap between fixture-style repair and real mutation training for Stage 7 memory policies. "
    "It does not confirm production training readiness."
)
GATE_THRESHOLDS = {
    "multi_sentence_binding_accuracy": (0.75, "min"),
    "long_horizon_recall": (0.75, "min"),
    "ambiguous_abstain_accuracy": (0.80, "min"),
    "nested_depth2_accuracy": (0.75, "min"),
    "nested_depth3_accuracy": (0.65, "min"),
    "capacity_pressure_accuracy": (0.70, "min"),
    "stale_update_rejection_rate": (0.85, "min"),
    "corrupt_then_repair_success_rate": (0.80, "min"),
    "distractor_gap_survival": (0.80, "min"),
    "trace_validity": (0.95, "min"),
    "wrong_writeback_rate": (0.02, "max"),
    "destructive_overwrite_rate": (0.02, "max"),
    "branch_contamination_rate": (0.0, "eq"),
    "semantic_slot_leak_detected": (False, "eq"),
    "macro_leak_detected": (False, "eq"),
    "privileged_control_selected_as_primary": (False, "eq"),
    "static_fixture_selected_as_primary": (False, "eq"),
    "aggregate_recomputed_from_episode_logs": (True, "eq"),
    "deterministic_replay_passed": (True, "eq"),
    "checker_failure_count": (0, "eq"),
}
ALLOWED_MICRO_OPS = {
    "READ_TOKEN",
    "COMPARE_TOKEN",
    "WRITE_MEMORY_SLOT",
    "READ_MEMORY_SLOT",
    "CLEAR_MEMORY_SLOT",
    "SCORE_MEMORY_SLOT",
    "ROUTE_KEY",
    "ROUTE_VALUE",
    "UPDATE_CONFIDENCE",
    "REJECT_STALE",
    "APPLY_REPAIR_EVIDENCE",
    "ABSTAIN_IF_AMBIGUOUS",
    "RESOLVE_NESTED",
    "GATED_COMMIT",
    "EMIT_OUTPUT",
}
FORBIDDEN_MACROS = {"BIND", "QUERY", "MEMORY_LOOKUP_MACRO", "KEY_VALUE_BIND_MACRO", "ORACLE_LOOKUP"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return rounded(sum(values) / len(values))


def exact_rate(rows: list[dict[str, Any]], families: set[str] | None = None) -> float:
    subset = [row for row in rows if families is None or row["family"] in families]
    return rate(sum(1 for row in subset if row["exact"]), len(subset))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    binding_families = set(FAMILIES) - {"AMBIGUOUS_EVIDENCE_ABSTAIN_OR_REPAIR"}
    recall_families = {"MULTI_BIND_DELAYED_QUERY", "NESTED_BINDING_DEPTH2", "NESTED_BINDING_DEPTH3", "CAPACITY_PRESSURE", "DISTRACTOR_GAP"}
    heldout_binding = {"NESTED_BINDING_DEPTH2", "NESTED_BINDING_DEPTH3", "CAPACITY_PRESSURE", "STALE_UPDATE_REJECTION", "CORRUPT_THEN_REPAIR"}
    return {
        "multi_sentence_binding_accuracy": exact_rate(rows, binding_families),
        "long_horizon_recall": exact_rate(rows, recall_families),
        "ambiguous_abstain_accuracy": exact_rate(rows, {"AMBIGUOUS_EVIDENCE_ABSTAIN_OR_REPAIR"}),
        "nested_depth2_accuracy": exact_rate(rows, {"NESTED_BINDING_DEPTH2"}),
        "nested_depth3_accuracy": exact_rate(rows, {"NESTED_BINDING_DEPTH3"}),
        "capacity_pressure_accuracy": exact_rate(rows, {"CAPACITY_PRESSURE"}),
        "stale_update_rejection_rate": exact_rate(rows, {"STALE_UPDATE_REJECTION"}),
        "corrupt_then_repair_success_rate": exact_rate(rows, {"CORRUPT_THEN_REPAIR"}),
        "distractor_gap_survival": exact_rate(rows, {"DISTRACTOR_GAP"}),
        "single_bind_delayed_query_accuracy": exact_rate(rows, {"SINGLE_BIND_DELAYED_QUERY"}),
        "multi_bind_delayed_query_accuracy": exact_rate(rows, {"MULTI_BIND_DELAYED_QUERY"}),
        "mixed_memory_template_accuracy": exact_rate(rows, {"MIXED_MEMORY_AND_TEMPLATE"}),
        "trace_validity": mean([row["trace_validity"] for row in rows]),
        "wrong_writeback_rate": rate(sum(1 for row in rows if row["wrong_writeback"]), len(rows)),
        "destructive_overwrite_rate": rate(sum(1 for row in rows if row["destructive_overwrite"]), len(rows)),
        "branch_contamination_rate": rate(sum(1 for row in rows if row["branch_contamination"]), len(rows)),
        "stale_write_rejection_rate": rate(sum(1 for row in rows if row["stale_write_rejected"]), len(rows)),
        "gate_false_accept_rate": rate(sum(1 for row in rows if row["gate_false_accept"]), len(rows)),
        "gate_false_reject_rate": rate(sum(1 for row in rows if row["gate_false_reject"]), len(rows)),
        "heldout_vocab_accuracy": exact_rate([row for row in rows if row["heldout_vocab"]]),
        "randomized_codebook_generalization": exact_rate(rows),
        "heldout_binding_pattern_accuracy": exact_rate(rows, heldout_binding),
        "heldout_gap_length_accuracy": exact_rate(rows, {"DISTRACTOR_GAP"}),
        "cost_per_episode": mean([row["cost"] for row in rows]),
        "cost_per_tick": rounded(mean([row["cost"] for row in rows]) / 3.0),
        "average_memory_slots_used": mean([row["memory_slots_used"] for row in rows]),
        "max_memory_slots_used": max((row["memory_slots_used"] for row in rows), default=0),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    positives = (
        metrics["multi_sentence_binding_accuracy"]
        + metrics["long_horizon_recall"]
        + metrics["ambiguous_abstain_accuracy"]
        + metrics["nested_depth2_accuracy"]
        + metrics["nested_depth3_accuracy"]
        + metrics["capacity_pressure_accuracy"]
        + metrics["stale_update_rejection_rate"]
        + metrics["corrupt_then_repair_success_rate"]
        + metrics["distractor_gap_survival"]
        + metrics["trace_validity"]
    )
    penalties = metrics["wrong_writeback_rate"] * 2.0 + metrics["destructive_overwrite_rate"] * 2.0 + metrics["cost_per_episode"] * 0.004
    return rounded(positives - penalties)


def gate_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, (threshold, mode) in GATE_THRESHOLDS.items():
        value = metrics.get(key)
        if mode == "min":
            checks[f"{key}_at_least_{threshold}"] = value >= threshold
        elif mode == "max":
            checks[f"{key}_at_most_{threshold}"] = value <= threshold
        else:
            checks[f"{key}_equals_{threshold}"] = value == threshold
    return checks


def source_audit(failures: list[dict[str, Any]]) -> None:
    text = RUNNER.read_text(encoding="utf-8")
    lowered = text.lower()
    forbidden_patterns = (
        "def base_metrics",
        "def capacity_sweep_rows",
        "repaired_stage7_binding_accuracy = 0.872",
        '"repaired_stage7_binding_accuracy": 0.872',
        "interpolat",
        "static metric rows",
    )
    for pattern in forbidden_patterns:
        if pattern in lowered:
            add_failure(failures, "STATIC_FIXTURE_SOURCE_PATTERN", pattern)
    tree = ast.parse(text)
    allowed = {"__future__", "argparse", "dataclasses", "hashlib", "json", "pathlib", "random", "typing"}
    blocked = {"torch", "tensorflow", "keras", "jax", "numpy", "sklearn", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in blocked:
                add_failure(failures, "NEURAL_OR_EXTERNAL_IMPORT", name)
            elif root and root not in allowed:
                add_failure(failures, "NON_STDLIB_IMPORT_REVIEW_REQUIRED", name)


def recompute_training_curve(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generations = sorted({row["generation"] for row in score_rows})
    curve = []
    for generation in generations:
        rows = [row for row in score_rows if row["generation"] == generation]
        best = max(rows, key=lambda row: row["validation_score"])
        curve.append(
            {
                "generation": generation,
                "best_candidate_id": best["candidate_id"],
                "best_train_score": best["train_score"],
                "best_validation_score": best["validation_score"],
                "best_validation_binding_accuracy": best["validation_metrics"]["multi_sentence_binding_accuracy"],
                "best_validation_long_horizon_recall": best["validation_metrics"]["long_horizon_recall"],
            }
        )
    return curve


def compare_metric_block(failures: list[dict[str, Any]], label: str, expected: dict[str, Any], actual: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if actual.get(key) != expected.get(key):
            add_failure(failures, "RECOMPUTED_METRIC_MISMATCH", f"{label}:{key}:{actual.get(key)} != {expected.get(key)}")


def check_boundary(out: Path, failures: list[dict[str, Any]]) -> None:
    paths = [out / "report.md", out / "e16c_real_boundary_claims_report.json", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY_CONFIRMED not in text and BOUNDARY_GAP not in text:
            add_failure(failures, "BOUNDARY_TEXT_MISSING", str(path))
        scrubbed = text.replace(BOUNDARY_CONFIRMED, "").replace(BOUNDARY_GAP, "")
        lower = scrubbed.lower()
        blocked = ("agi", "consciousness", "open-natural-language", "open natural language", "general natural language", "d99", "d100")
        for token in blocked:
            if token in lower:
                add_failure(failures, "BROAD_CLAIM_TOKEN_FOUND", f"{path}:{token}")


def check_policy_ops(report: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    for section in ("policy",):
        ops = report.get(section, {}).get("micro_ops", [])
        if not ops:
            add_failure(failures, "POLICY_OPS_MISSING", section)
        for op in ops:
            if op not in ALLOWED_MICRO_OPS:
                add_failure(failures, "NON_MICRO_OP_IN_POLICY", op)
            if op in FORBIDDEN_MACROS:
                add_failure(failures, "FORBIDDEN_MACRO_IN_POLICY", op)


def check_reports(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    per_episode = load_json(out / "e16c_real_per_episode_eval_report.json")
    generation_scores = load_json(out / "e16c_real_generation_score_report.json")
    training_curve = load_json(out / "e16c_real_training_curve_report.json")
    capacity = load_json(out / "e16c_real_capacity_sweep_report.json")
    ablation = load_json(out / "e16c_real_ablation_report.json")
    static_audit = load_json(out / "e16c_real_static_fixture_audit_report.json")
    semantic = load_json(out / "e16c_real_semantic_macro_leak_audit_report.json")
    replay = load_json(out / "e16c_real_deterministic_replay_report.json")
    best_policy = load_json(out / "e16c_real_best_policy_report.json")
    pruned_policy = load_json(out / "e16c_real_pruned_policy_report.json")
    train_manifest = load_json(out / "e16c_real_train_episode_manifest.json")
    validation_manifest = load_json(out / "e16c_real_validation_episode_manifest.json")
    heldout_manifest = load_json(out / "e16c_real_heldout_episode_manifest.json")
    systems = aggregate.get("systems", {})
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_MISMATCH", str(decision.get("primary_system")))
    if aggregate.get("primary_system") != PRIMARY:
        add_failure(failures, "AGGREGATE_PRIMARY_MISMATCH", str(aggregate.get("primary_system")))
    if systems.get(FIXTURE_REFERENCE, {}).get("invalid_for_proof") is not True:
        add_failure(failures, "FIXTURE_REFERENCE_NOT_MARKED_INVALID", FIXTURE_REFERENCE)
    if static_audit.get("source_fixture_audit_passed") is not True:
        add_failure(failures, "STATIC_FIXTURE_AUDIT_NOT_PASSED", "e16c_real_static_fixture_audit_report.json")
    if static_audit.get("static_fixture_selected_as_primary") is not False:
        add_failure(failures, "STATIC_FIXTURE_SELECTED_AS_PRIMARY", "e16c_real_static_fixture_audit_report.json")
    for manifest_a, manifest_b, label in (
        (train_manifest, validation_manifest, "train_validation"),
        (train_manifest, heldout_manifest, "train_heldout"),
        (validation_manifest, heldout_manifest, "validation_heldout"),
    ):
        ids_a = {episode["episode_id"] for episode in manifest_a.get("episodes", [])}
        ids_b = {episode["episode_id"] for episode in manifest_b.get("episodes", [])}
        if ids_a.intersection(ids_b):
            add_failure(failures, "SPLIT_EPISODE_OVERLAP", label)
    if not per_episode.get("derived_from_policy_execution"):
        add_failure(failures, "PER_EPISODE_NOT_DERIVED_FROM_EXECUTION", "e16c_real_per_episode_eval_report.json")
    rows = per_episode.get("rows", [])
    metric_keys = (
        "multi_sentence_binding_accuracy",
        "long_horizon_recall",
        "ambiguous_abstain_accuracy",
        "nested_depth2_accuracy",
        "nested_depth3_accuracy",
        "capacity_pressure_accuracy",
        "stale_update_rejection_rate",
        "corrupt_then_repair_success_rate",
        "distractor_gap_survival",
        "trace_validity",
        "wrong_writeback_rate",
        "destructive_overwrite_rate",
        "branch_contamination_rate",
        "heldout_vocab_accuracy",
        "randomized_codebook_generalization",
        "heldout_binding_pattern_accuracy",
        "heldout_gap_length_accuracy",
        "cost_per_episode",
        "cost_per_tick",
    )
    for system in REQUIRED_SYSTEMS:
        if system == FIXTURE_REFERENCE:
            continue
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            add_failure(failures, "MISSING_PER_EPISODE_ROWS_FOR_SYSTEM", system)
            continue
        recomputed = compute_metrics(system_rows)
        compare_metric_block(failures, system, recomputed, systems[system], metric_keys)
    if PRIMARY in systems:
        primary = systems[PRIMARY]
        expected_gate = gate_checks(primary)
        for key, value in expected_gate.items():
            if aggregate.get("positive_gate", {}).get("checks", {}).get(key) is not value:
                add_failure(failures, "GATE_MATH_MISMATCH", key)
        gate_passed = all(expected_gate.values())
        if aggregate.get("positive_gate", {}).get("passed") is not gate_passed:
            add_failure(failures, "GATE_PASSED_MISMATCH", "aggregate_metrics.json")
        baseline_metrics = [systems[name] for name in BASELINES]
        best_baseline = max(baseline_metrics, key=score_metrics)
        best_baseline_name = next(name for name in BASELINES if systems[name] is best_baseline)
        if primary.get("best_baseline_system") != best_baseline_name:
            add_failure(failures, "BEST_BASELINE_MISMATCH", str(primary.get("best_baseline_system")))
        if primary.get("delta_vs_best_baseline_binding_accuracy") != rounded(primary["multi_sentence_binding_accuracy"] - best_baseline["multi_sentence_binding_accuracy"]):
            add_failure(failures, "DELTA_BINDING_MISMATCH", str(primary.get("delta_vs_best_baseline_binding_accuracy")))
        if primary.get("delta_vs_best_baseline_long_horizon_recall") != rounded(primary["long_horizon_recall"] - best_baseline["long_horizon_recall"]):
            add_failure(failures, "DELTA_RECALL_MISMATCH", str(primary.get("delta_vs_best_baseline_long_horizon_recall")))
        expected_decision = "e16c_real_mutation_training_stage7_confirmed" if gate_passed else (
            "e16c_real_mutation_training_stage7_partial"
            if primary["delta_vs_best_baseline_binding_accuracy"] >= 0.05 or primary["delta_vs_best_baseline_long_horizon_recall"] >= 0.05 or primary["delta_vs_best_baseline_trace_validity"] >= 0.03
            else "e16c_real_mutation_training_stage7_failed"
        )
        if decision.get("decision") != expected_decision:
            add_failure(failures, "DECISION_MATH_MISMATCH", f"{decision.get('decision')} != {expected_decision}")
        if decision.get("positive_gate_passed") is not gate_passed:
            add_failure(failures, "DECISION_GATE_MISMATCH", "decision.json")
        if summary.get("aggregate_recomputed_from_episode_logs") is not True:
            add_failure(failures, "SUMMARY_RECOMPUTE_FLAG_NOT_TRUE", "summary.json")
    recomputed_curve = recompute_training_curve(generation_scores.get("rows", []))
    if training_curve.get("curve") != recomputed_curve:
        add_failure(failures, "TRAINING_CURVE_RECOMPUTE_MISMATCH", "e16c_real_training_curve_report.json")
    for row in capacity.get("rows", []):
        slot = str(row["slot_count"])
        slot_rows = capacity.get("per_slot_episode_rows", {}).get(slot, [])
        if not slot_rows:
            add_failure(failures, "CAPACITY_SLOT_ROWS_MISSING", slot)
            continue
        metrics = compute_metrics(slot_rows)
        expected = {
            "binding_accuracy": metrics["multi_sentence_binding_accuracy"],
            "long_horizon_recall": metrics["long_horizon_recall"],
            "nested_depth2_accuracy": metrics["nested_depth2_accuracy"],
            "nested_depth3_accuracy": metrics["nested_depth3_accuracy"],
            "capacity_pressure_accuracy": metrics["capacity_pressure_accuracy"],
            "cost_per_episode": metrics["cost_per_episode"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                add_failure(failures, "CAPACITY_RECOMPUTE_MISMATCH", f"{slot}:{key}")
    if capacity.get("slot_counts") != [1, 2, 3, 4, 6, 8, 12]:
        add_failure(failures, "CAPACITY_SLOT_SET_MISMATCH", str(capacity.get("slot_counts")))
    expectations = ablation.get("expectations", {})
    for key, value in expectations.items():
        if value is not True:
            add_failure(failures, "ABLATION_EXPECTATION_FAILED", key)
    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e16c_real_semantic_macro_leak_audit_report.json")
    if semantic.get("macro_leak_detected") is not False:
        add_failure(failures, "MACRO_LEAK_FLAG_NOT_FALSE", "e16c_real_semantic_macro_leak_audit_report.json")
    if semantic.get("runtime_receives_expected_answer") is not False:
        add_failure(failures, "EXPECTED_ANSWER_RUNTIME_FLAG_NOT_FALSE", "e16c_real_semantic_macro_leak_audit_report.json")
    if replay.get("internal_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e16c_real_deterministic_replay_report.json")
    check_policy_ops(best_policy, failures)
    check_policy_ops(pruned_policy, failures)


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            add_failure(failures, "MISSING_ARTIFACT", artifact)
    for doc in DOCS:
        if not doc.exists():
            add_failure(failures, "MISSING_DOC", str(doc))
    if RUNNER.exists():
        source_audit(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_reports(out, failures)
    check_boundary(out, failures)
    result = {"schema_version": "e16c_real_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e16c_real_mutation_training_audit_stage7")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
