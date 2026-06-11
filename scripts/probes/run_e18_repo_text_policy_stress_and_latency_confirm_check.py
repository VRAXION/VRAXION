#!/usr/bin/env python3
"""Checker for E18 repo-text policy stress and latency confirmation probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any


RUNNER = Path(__file__).with_name("run_e18_repo_text_policy_stress_and_latency_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM_RESULT.md",
)
BOUNDARY_TEXT = (
    "This is a real-repository-text stress and latency audit for a controlled Flow text policy. "
    "It uses local project documents and adversarial deterministic task wrappers. "
    "It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness."
)
E17_REFERENCE = "E17_POLICY_REFERENCE"
STATIC = "STATIC_KEYWORD_BASELINE"
BM25 = "BM25_LIKE_BASELINE"
HEADING = "HEADING_PATH_WEIGHTED_BASELINE"
SOURCE_ORACLE = "SOURCE_PATH_ORACLE_CONTROL"
FIELD_ORACLE = "FIELD_NAME_ORACLE_CONTROL"
HAND = "HAND_AUTHORED_EXTRACTOR_CONTROL"
RANDOM_BASELINE = "RANDOM_POLICY_BASELINE"
UNPRUNED = "MUTATION_TRAINED_STRESS_POLICY"
PRIMARY = "MUTATION_TRAINED_PRUNED_STRESS_POLICY_PRIMARY"
ABLATIONS = (
    "NO_SOURCE_PATH_FEATURE_ABLATION",
    "NO_HEADING_PATH_FEATURE_ABLATION",
    "NO_TABLE_PARSER_ABLATION",
    "NO_NUMERIC_PARSER_ABLATION",
    "NO_ABSTAIN_POLICY_ABLATION",
    "NO_DISTRACTOR_REJECTION_ABLATION",
    "NO_LONG_CONTEXT_MEMORY_ABLATION",
    "NO_PARAPHRASE_ALIAS_ABLATION",
    "NO_CANONICAL_DECODER_STRICTNESS_ABLATION",
)
SYSTEMS = (
    E17_REFERENCE,
    STATIC,
    BM25,
    HEADING,
    SOURCE_ORACLE,
    FIELD_ORACLE,
    HAND,
    RANDOM_BASELINE,
    UNPRUNED,
    PRIMARY,
    *ABLATIONS,
)
FAMILIES = (
    "NO_SOURCE_PATH_FIELD_EXTRACTION",
    "PARAPHRASED_FIELD_EXTRACTION",
    "SAME_KEY_CONFLICT_RETRIEVAL",
    "SAME_MILESTONE_DISTRACTOR",
    "TARGET_NOT_FIRST_LONG_CONTEXT",
    "TABLE_NUMERIC_STRESS",
    "METRIC_DELTA_STRESS",
    "CROSS_DOC_CHAIN_STRESS",
    "CAVEAT_BOUNDARY_PARAPHRASE",
    "AMBIGUOUS_OR_MISSING_EVIDENCE",
    "ADVERSARIAL_NOISY_CONTEXT",
    "LATENCY_COST_STRESS",
    "SOURCE_PATH_HINT_ABLATION",
    "FIELD_NAME_HINT_ABLATION",
    "HELDOUT_FUTURE_DOCS",
)
VALID_DECISIONS = (
    "e18_repo_text_policy_stress_and_latency_confirmed",
    "e18_repo_text_policy_stress_and_latency_partial_downshifted",
    "e18_repo_text_policy_stress_and_latency_partial",
    "e18_repo_text_policy_stress_and_latency_failed",
    "e18_repo_text_policy_stress_and_latency_invalid_or_incomplete",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e18_search_report.json",
    "e18_corpus_manifest.json",
    "e18_corpus_split_report.json",
    "e18_episode_generation_report.json",
    "e18_train_episode_manifest.json",
    "e18_validation_episode_manifest.json",
    "e18_heldout_episode_manifest.json",
    "e18_stress_episode_manifest.json",
    "e18_candidate_population_report.json",
    "e18_generation_score_report.json",
    "e18_training_curve_report.json",
    "e18_checkpoint_report.json",
    "e18_best_policy_report.json",
    "e18_pruned_policy_report.json",
    "e18_per_episode_eval_report.json",
    "e18_system_comparison_report.json",
    "e18_task_family_report.json",
    "e18_ablation_report.json",
    "e18_source_path_hint_ablation_report.json",
    "e18_field_name_hint_ablation_report.json",
    "e18_same_key_conflict_report.json",
    "e18_same_milestone_distractor_report.json",
    "e18_target_not_first_report.json",
    "e18_table_numeric_report.json",
    "e18_long_context_memory_report.json",
    "e18_abstain_ambiguity_report.json",
    "e18_latency_report.json",
    "e18_trace_validity_report.json",
    "e18_writeback_safety_report.json",
    "e18_renderer_faithfulness_report.json",
    "e18_source_fixture_audit_report.json",
    "e18_deterministic_replay_report.json",
    "e18_boundary_claims_report.json",
    "e18_failure_map_report.json",
    "e18_next_recommendation.json",
    "checkpoint_latest.json",
    "training_progress.jsonl",
)
GATE_THRESHOLDS = {
    "exact_answer_accuracy": (0.78, "min"),
    "canonical_object_accuracy": (0.75, "min"),
    "evidence_chunk_accuracy": (0.80, "min"),
    "retrieval_top1_accuracy": (0.80, "min"),
    "no_source_path_accuracy": (0.70, "min"),
    "paraphrased_field_accuracy": (0.70, "min"),
    "same_key_conflict_accuracy": (0.70, "min"),
    "same_milestone_distractor_accuracy": (0.70, "min"),
    "target_not_first_accuracy": (0.75, "min"),
    "table_row_extraction_accuracy": (0.70, "min"),
    "metric_delta_accuracy": (0.70, "min"),
    "noisy_context_repair_accuracy": (0.70, "min"),
    "long_context_memory_accuracy": (0.70, "min"),
    "ambiguity_handling_accuracy": (0.75, "min"),
    "hallucinated_answer_rate": (0.05, "max"),
    "wrong_evidence_rate": (0.10, "max"),
    "trace_validity": (0.90, "min"),
    "renderer_faithfulness": (0.98, "min"),
}


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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return rounded(sorted_values[index])


def milestone_score(path: str) -> int:
    match = re.search(r"(?:^|/)E(\d+)([A-Z]?)", path)
    if not match:
        return -1
    suffix = match.group(2)
    return int(match.group(1)) * 100 + (ord(suffix) - 64 if suffix else 0)


def accuracy(rows: list[dict[str, Any]], key: str = "exact") -> float:
    return rate(sum(1 for row in rows if row[key]), len(rows))


def family_accuracy(rows: list[dict[str, Any]], family: str, key: str = "exact") -> float:
    return accuracy([row for row in rows if row["family"] == family], key)


def hint_accuracy(rows: list[dict[str, Any]], predicate: Any) -> float:
    return accuracy([row for row in rows if predicate(row)])


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_abstain = [row for row in rows if row["expected_status"] in {"missing_evidence", "ambiguous"}]
    predicted_abstain = [row for row in rows if row["predicted_status"] in {"missing_evidence", "ambiguous"}]
    file_groups: dict[str, list[dict[str, Any]]] = {}
    stress_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["file_path"]:
            file_groups.setdefault(row["file_path"], []).append(row)
            if row["split"] == "stress":
                stress_groups.setdefault(row["file_path"], []).append(row)
    latencies = [row["total_latency_ms"] for row in rows]
    source_hint = hint_accuracy(rows, lambda row: row["hint_profile"].get("source_path_hint") is True)
    no_source = hint_accuracy(rows, lambda row: row["hint_profile"].get("source_path_hint") is False)
    field_hint = hint_accuracy(rows, lambda row: row["hint_profile"].get("field_name_hint") is True)
    no_field_hint = hint_accuracy(rows, lambda row: row["hint_profile"].get("field_name_hint") is False)
    total_latency_s = sum(latencies) / 1000.0
    return {
        "episode_count": len(rows),
        "exact_answer_accuracy": accuracy(rows),
        "canonical_object_accuracy": accuracy(rows, "canonical_exact"),
        "evidence_chunk_accuracy": accuracy(rows, "evidence_exact"),
        "retrieval_top1_accuracy": rate(sum(1 for row in rows if row["retrieval_evaluated"] and row["retrieval_exact"]), sum(1 for row in rows if row["retrieval_evaluated"])),
        "no_source_path_accuracy": no_source,
        "paraphrased_field_accuracy": hint_accuracy(rows, lambda row: row["hint_profile"].get("paraphrased") is True),
        "same_key_conflict_accuracy": family_accuracy(rows, "SAME_KEY_CONFLICT_RETRIEVAL"),
        "same_milestone_distractor_accuracy": family_accuracy(rows, "SAME_MILESTONE_DISTRACTOR"),
        "target_not_first_accuracy": hint_accuracy(rows, lambda row: row["hint_profile"].get("target_not_first") is True),
        "table_row_extraction_accuracy": family_accuracy(rows, "TABLE_NUMERIC_STRESS"),
        "metric_delta_accuracy": family_accuracy(rows, "METRIC_DELTA_STRESS"),
        "cross_doc_chain_accuracy": family_accuracy(rows, "CROSS_DOC_CHAIN_STRESS"),
        "caveat_boundary_accuracy": family_accuracy(rows, "CAVEAT_BOUNDARY_PARAPHRASE"),
        "noisy_context_repair_accuracy": family_accuracy(rows, "ADVERSARIAL_NOISY_CONTEXT"),
        "long_context_memory_accuracy": family_accuracy(rows, "TARGET_NOT_FIRST_LONG_CONTEXT"),
        "ambiguity_handling_accuracy": family_accuracy(rows, "AMBIGUOUS_OR_MISSING_EVIDENCE"),
        "missing_evidence_accuracy": accuracy([row for row in rows if row["expected_status"] == "missing_evidence"]),
        "source_path_hint_dependency_delta": rounded(source_hint - no_source),
        "field_name_hint_dependency_delta": rounded(field_hint - no_field_hint),
        "hallucinated_answer_rate": rate(sum(1 for row in rows if row["hallucinated"]), len(rows)),
        "wrong_evidence_rate": rate(sum(1 for row in rows if row["wrong_evidence"]), len(rows)),
        "trace_validity": mean([row["trace_validity"] for row in rows]),
        "wrong_writeback_rate": rate(sum(1 for row in rows if row["wrong_writeback"]), len(rows)),
        "destructive_overwrite_rate": rate(sum(1 for row in rows if row["destructive_overwrite"]), len(rows)),
        "branch_contamination_rate": rate(sum(1 for row in rows if row["branch_contamination"]), len(rows)),
        "renderer_faithfulness": mean([row["renderer_faithful"] for row in rows]),
        "cost_per_episode": mean([row["cost"] for row in rows]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": rounded(max(latencies) if latencies else 0.0),
        "episodes_per_second": rounded(len(rows) / max(total_latency_s, 0.001)),
        "retrieval_latency_p50_ms": percentile([row["retrieval_latency_ms"] for row in rows], 50),
        "extraction_latency_p50_ms": percentile([row["extraction_latency_ms"] for row in rows], 50),
        "decode_latency_p50_ms": percentile([row["decode_latency_ms"] for row in rows], 50),
        "heldout_file_accuracy": mean([accuracy(group) for group in file_groups.values()]),
        "stress_file_accuracy": mean([accuracy(group) for group in stress_groups.values()]),
        "heldout_future_doc_accuracy": accuracy([row for row in rows if row["future_doc"]]),
        "unseen_heading_accuracy": accuracy([row for row in rows if row["out_of_train_heading"]]),
        "unseen_milestone_accuracy": accuracy([row for row in rows if row["future_doc"]]),
        "abstain_precision": rate(sum(1 for row in predicted_abstain if row["exact"]), len(predicted_abstain)),
        "abstain_recall": rate(sum(1 for row in expected_abstain if row["predicted_status"] == row["expected_status"]), len(expected_abstain)),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    positives = (
        metrics["exact_answer_accuracy"] * 2.0
        + metrics["no_source_path_accuracy"] * 1.5
        + metrics["same_key_conflict_accuracy"] * 1.2
        + metrics["target_not_first_accuracy"]
        + metrics["ambiguity_handling_accuracy"]
        + metrics["evidence_chunk_accuracy"]
        + metrics["renderer_faithfulness"]
    )
    penalties = metrics["hallucinated_answer_rate"] * 2.5 + metrics["wrong_evidence_rate"] * 1.6 + metrics["latency_p95_ms"] * 0.001 + metrics["cost_per_episode"] * 0.01
    return rounded(positives - penalties)


def recompute_training_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for generation in sorted({row["generation"] for row in rows}):
        generation_rows = [row for row in rows if row["generation"] == generation]
        best = max(generation_rows, key=lambda row: row["validation_score"])
        curve.append(
            {
                "generation": generation,
                "best_candidate_id": best["candidate_id"],
                "best_train_score": best["train_score"],
                "best_validation_score": best["validation_score"],
                "best_validation_no_source_path_accuracy": best["validation_metrics"]["no_source_path_accuracy"],
                "best_validation_same_key_conflict_accuracy": best["validation_metrics"]["same_key_conflict_accuracy"],
            }
        )
    return curve


def gate_checks(metrics: dict[str, Any], systems: dict[str, dict[str, Any]], summary: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "run_budget_class_not_partial_downshifted": summary["run_budget_class"] != "partial_downshifted",
        "actual_generations_at_least_40": summary["generations_completed"] >= 40,
        "actual_population_at_least_64": summary["population_size"] >= 64,
        "actual_heldout_episodes_at_least_800": summary["heldout_episode_count"] >= 800,
        "actual_stress_episodes_at_least_800": summary["stress_episode_count"] >= 800,
        "latency_p95_ms_reported_and_finite": metrics["latency_p95_ms"] >= 0.0 and metrics["latency_p95_ms"] < 60_000,
        "beats_bm25_no_source_path_by_0.05": rounded(metrics["no_source_path_accuracy"] - systems[BM25]["no_source_path_accuracy"]) >= 0.05,
        "beats_static_same_key_by_0.08": rounded(metrics["same_key_conflict_accuracy"] - systems[STATIC]["same_key_conflict_accuracy"]) >= 0.08,
        "source_fixture_audit_passed": summary["source_fixture_audit_passed"] is True,
        "aggregate_recomputed_from_episode_logs": summary["aggregate_recomputed_from_episode_logs"] is True,
        "checker_failure_count_zero": summary["checker_failure_count"] == 0,
    }
    for key, (threshold, mode) in GATE_THRESHOLDS.items():
        checks[f"{key}_{'at_least' if mode == 'min' else 'at_most'}_{threshold}"] = metrics[key] >= threshold if mode == "min" else metrics[key] <= threshold
    return checks


def compare_metrics(failures: list[dict[str, Any]], label: str, expected: dict[str, Any], actual: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if expected.get(key) != actual.get(key):
            add_failure(failures, "RECOMPUTED_METRIC_MISMATCH", f"{label}:{key}:{actual.get(key)} != {expected.get(key)}")


def source_audit(failures: list[dict[str, Any]]) -> None:
    text = RUNNER.read_text(encoding="utf-8")
    lowered = text.lower()
    for pattern in ("static final metric table", "hardcoded final primary", "interpolated training curve", "torch", "tensorflow", "sklearn"):
        if pattern in lowered:
            add_failure(failures, "SOURCE_FORBIDDEN_PATTERN", pattern)
    tree = ast.parse(text)
    allowed = {"__future__", "argparse", "dataclasses", "hashlib", "importlib", "json", "pathlib", "random", "re", "statistics", "sys", "time", "typing"}
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
    if "def execute_policy" in text:
        body = text.split("def execute_policy", 1)[1].split("\ndef ", 1)[0]
        if ".expected_" in body:
            add_failure(failures, "EXPECTED_USED_DURING_INFERENCE", "execute_policy")
        if ".family" in body:
            add_failure(failures, "RAW_FAMILY_USED_DURING_INFERENCE", "execute_policy")


def check_boundary(out: Path, failures: list[dict[str, Any]]) -> None:
    for path in [out / "report.md", *DOCS]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY_TEXT not in text:
            add_failure(failures, "BOUNDARY_TEXT_MISSING", str(path))
        lower = text.replace(BOUNDARY_TEXT, "").lower()
        for claim in ("proves general natural-language", "proves internet-scale", "production ready", "confirms agi", "confirms consciousness"):
            if claim in lower:
                add_failure(failures, "BROAD_POSITIVE_CLAIM_FOUND", f"{path}:{claim}")


def check_reports(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    per_episode = load_json(out / "e18_per_episode_eval_report.json")
    generation_scores = load_json(out / "e18_generation_score_report.json")
    curve_report = load_json(out / "e18_training_curve_report.json")
    split = load_json(out / "e18_corpus_split_report.json")
    source_fixture = load_json(out / "e18_source_fixture_audit_report.json")
    replay = load_json(out / "e18_deterministic_replay_report.json")
    failure_map = load_json(out / "e18_failure_map_report.json")
    systems = aggregate.get("systems", {})
    for system in SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY or aggregate.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_MISMATCH", f"{decision.get('primary_system')} / {aggregate.get('primary_system')}")
    for invalid in (SOURCE_ORACLE, FIELD_ORACLE, HAND):
        if systems.get(invalid, {}).get("invalid_for_primary") is not True:
            add_failure(failures, "INVALID_CONTROL_NOT_MARKED", invalid)
        if decision.get("primary_system") == invalid:
            add_failure(failures, "INVALID_CONTROL_SELECTED_AS_PRIMARY", invalid)
    leakage = split.get("leakage_audit", {})
    if leakage.get("split_by_file") is not True or leakage.get("passed") is not True:
        add_failure(failures, "SPLIT_LEAKAGE_AUDIT_FAILED", str(leakage))
    for key in ("train_validation_overlap", "train_heldout_overlap", "train_stress_overlap", "validation_heldout_overlap", "validation_stress_overlap", "heldout_stress_overlap"):
        if leakage.get(key):
            add_failure(failures, "SPLIT_FILE_OVERLAP", key)
    if per_episode.get("derived_from_policy_execution") is not True:
        add_failure(failures, "PER_EPISODE_NOT_DERIVED_FROM_EXECUTION", "e18_per_episode_eval_report.json")
    if source_fixture.get("source_fixture_audit_passed") is not True:
        add_failure(failures, "SOURCE_FIXTURE_AUDIT_NOT_PASSED", "e18_source_fixture_audit_report.json")
    for key in ("metrics_are_static_tables", "training_curve_interpolated", "source_path_oracle_selected_as_primary", "field_name_oracle_selected_as_primary", "hand_authored_extractor_selected_as_primary", "raw_task_family_labels_route_answer_selection", "hardcoded_final_primary_numbers", "neural_dependencies_used"):
        if source_fixture.get(key) is not False:
            add_failure(failures, "SOURCE_AUDIT_FLAG_NOT_FALSE", key)
    if replay.get("deterministic_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e18_deterministic_replay_report.json")
    rows = per_episode.get("rows", [])
    metric_keys = (
        "episode_count",
        "exact_answer_accuracy",
        "canonical_object_accuracy",
        "evidence_chunk_accuracy",
        "retrieval_top1_accuracy",
        "no_source_path_accuracy",
        "paraphrased_field_accuracy",
        "same_key_conflict_accuracy",
        "same_milestone_distractor_accuracy",
        "target_not_first_accuracy",
        "table_row_extraction_accuracy",
        "metric_delta_accuracy",
        "cross_doc_chain_accuracy",
        "caveat_boundary_accuracy",
        "noisy_context_repair_accuracy",
        "long_context_memory_accuracy",
        "ambiguity_handling_accuracy",
        "missing_evidence_accuracy",
        "source_path_hint_dependency_delta",
        "field_name_hint_dependency_delta",
        "hallucinated_answer_rate",
        "wrong_evidence_rate",
        "trace_validity",
        "wrong_writeback_rate",
        "destructive_overwrite_rate",
        "branch_contamination_rate",
        "renderer_faithfulness",
        "cost_per_episode",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "episodes_per_second",
        "retrieval_latency_p50_ms",
        "extraction_latency_p50_ms",
        "decode_latency_p50_ms",
        "heldout_file_accuracy",
        "stress_file_accuracy",
        "heldout_future_doc_accuracy",
        "unseen_heading_accuracy",
        "unseen_milestone_accuracy",
        "abstain_precision",
        "abstain_recall",
    )
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            add_failure(failures, "MISSING_PER_EPISODE_ROWS_FOR_SYSTEM", system)
            continue
        recomputed = compute_metrics(system_rows)
        compare_metrics(failures, system, recomputed, systems[system], metric_keys)
    if PRIMARY in systems and BM25 in systems and STATIC in systems:
        primary = systems[PRIMARY]
        if primary.get("delta_vs_bm25_no_source_path_accuracy") != rounded(primary["no_source_path_accuracy"] - systems[BM25]["no_source_path_accuracy"]):
            add_failure(failures, "BM25_DELTA_MISMATCH", str(primary.get("delta_vs_bm25_no_source_path_accuracy")))
        if primary.get("delta_vs_static_same_key_conflict_accuracy") != rounded(primary["same_key_conflict_accuracy"] - systems[STATIC]["same_key_conflict_accuracy"]):
            add_failure(failures, "STATIC_DELTA_MISMATCH", str(primary.get("delta_vs_static_same_key_conflict_accuracy")))
        expected_gate = gate_checks(primary, systems, summary)
        if aggregate.get("positive_gate", {}).get("checks") != expected_gate:
            add_failure(failures, "GATE_CHECKS_MISMATCH", "aggregate_metrics.json")
        gate_passed = all(expected_gate.values())
        if aggregate.get("positive_gate", {}).get("passed") is not gate_passed:
            add_failure(failures, "GATE_PASSED_MISMATCH", "aggregate_metrics.json")
        expected_decision = "e18_repo_text_policy_stress_and_latency_partial_downshifted" if summary.get("run_budget_class") == "partial_downshifted" else (
            "e18_repo_text_policy_stress_and_latency_confirmed" if gate_passed else (
                "e18_repo_text_policy_stress_and_latency_partial"
                if primary["delta_vs_bm25_no_source_path_accuracy"] >= 0.02 or primary["delta_vs_static_same_key_conflict_accuracy"] >= 0.02
                else "e18_repo_text_policy_stress_and_latency_failed"
            )
        )
        if decision.get("decision") != expected_decision:
            add_failure(failures, "DECISION_MATH_MISMATCH", f"{decision.get('decision')} != {expected_decision}")
        if decision.get("positive_gate_passed") is not gate_passed:
            add_failure(failures, "DECISION_GATE_MISMATCH", "decision.json")
    recomputed_curve = recompute_training_curve(generation_scores.get("rows", []))
    if curve_report.get("curve") != recomputed_curve:
        add_failure(failures, "TRAINING_CURVE_RECOMPUTE_MISMATCH", "e18_training_curve_report.json")
    if recomputed_curve:
        overfit = rounded(recomputed_curve[-1]["best_train_score"] - recomputed_curve[-1]["best_validation_score"])
        if curve_report.get("overfit_gap") != overfit or summary.get("overfit_gap") != overfit:
            add_failure(failures, "OVERFIT_GAP_RECOMPUTE_MISMATCH", str(overfit))
    if failure_map.get("failure_map_complete") is not True:
        add_failure(failures, "FAILURE_MAP_INCOMPLETE", "e18_failure_map_report.json")
    if summary.get("aggregate_recomputed_from_episode_logs") is not True:
        add_failure(failures, "SUMMARY_RECOMPUTE_FLAG_NOT_TRUE", "summary.json")


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
    result = {"schema_version": "e18_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
