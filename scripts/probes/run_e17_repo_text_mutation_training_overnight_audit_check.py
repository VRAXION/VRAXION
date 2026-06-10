#!/usr/bin/env python3
"""Checker for E17 real-repository-text mutation-training overnight audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any


RUNNER = Path(__file__).with_name("run_e17_repo_text_mutation_training_overnight_audit.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT_RESULT.md",
)
STATIC = "STATIC_KEYWORD_BASELINE"
BM25 = "BM25_LIKE_BASELINE"
HEADING = "HEADING_PATH_WEIGHTED_BASELINE"
HAND = "HAND_AUTHORED_EXTRACTOR_CONTROL"
RANDOM_BASELINE = "RANDOM_POLICY_BASELINE"
UNPRUNED = "MUTATION_TRAINED_REPO_TEXT_POLICY"
PRIMARY = "MUTATION_TRAINED_PRUNED_REPO_TEXT_POLICY_PRIMARY"
ABLATIONS = (
    "NO_HEADING_PATH_ABLATION",
    "NO_TABLE_PARSER_ABLATION",
    "NO_NUMERIC_PARSER_ABLATION",
    "NO_ABSTAIN_POLICY_ABLATION",
    "NO_DISTRACTOR_REJECTION_ABLATION",
    "NO_LONG_CONTEXT_MEMORY_ABLATION",
    "NO_CANONICAL_DECODER_STRICTNESS_ABLATION",
)
SYSTEMS = (STATIC, BM25, HEADING, HAND, RANDOM_BASELINE, UNPRUNED, PRIMARY, *ABLATIONS)
FAMILIES = (
    "FIELD_EXTRACTION",
    "METRIC_COMPARISON",
    "RESULT_SUMMARY_CANONICAL",
    "DOCUMENT_RETRIEVAL",
    "CROSS_DOC_NEXT_CHAIN",
    "CAVEAT_BOUNDARY_DETECTION",
    "NOISY_CONTEXT_REPAIR",
    "LONG_CONTEXT_MEMORY",
    "TABLE_ROW_EXTRACTION",
    "AMBIGUOUS_OR_MISSING_EVIDENCE",
)
BOUNDARY_TEXT = (
    "This is an overnight real-repository-text mutation-training audit for a controlled Flow text policy. "
    "It uses real local project documents, but task wrappers and labels are deterministically generated from those documents. "
    "It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness."
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e17_search_report.json",
    "e17_corpus_manifest.json",
    "e17_corpus_split_report.json",
    "e17_episode_generation_report.json",
    "e17_train_episode_manifest.json",
    "e17_validation_episode_manifest.json",
    "e17_heldout_episode_manifest.json",
    "e17_candidate_population_report.json",
    "e17_generation_score_report.json",
    "e17_training_curve_report.json",
    "e17_checkpoint_report.json",
    "e17_best_policy_report.json",
    "e17_pruned_policy_report.json",
    "e17_per_episode_eval_report.json",
    "e17_system_comparison_report.json",
    "e17_task_family_report.json",
    "e17_ablation_report.json",
    "e17_retrieval_report.json",
    "e17_extraction_report.json",
    "e17_table_numeric_report.json",
    "e17_long_context_memory_report.json",
    "e17_abstain_ambiguity_report.json",
    "e17_trace_validity_report.json",
    "e17_writeback_safety_report.json",
    "e17_renderer_faithfulness_report.json",
    "e17_source_fixture_audit_report.json",
    "e17_deterministic_replay_report.json",
    "e17_boundary_claims_report.json",
    "e17_failure_map_report.json",
    "e17_next_recommendation.json",
    "checkpoint_latest.json",
    "training_progress.jsonl",
)
VALID_DECISIONS = (
    "e17_repo_text_mutation_training_overnight_confirmed",
    "e17_repo_text_mutation_training_overnight_partial",
    "e17_repo_text_mutation_training_overnight_failed",
    "e17_repo_text_mutation_training_overnight_invalid_or_incomplete",
)
GATE_THRESHOLDS = {
    "exact_answer_accuracy": (0.70, "min"),
    "canonical_object_accuracy": (0.70, "min"),
    "evidence_chunk_accuracy": (0.75, "min"),
    "retrieval_top1_accuracy": (0.75, "min"),
    "field_extraction_accuracy": (0.80, "min"),
    "table_row_extraction_accuracy": (0.70, "min"),
    "noisy_context_repair_accuracy": (0.65, "min"),
    "long_context_memory_accuracy": (0.65, "min"),
    "ambiguity_handling_accuracy": (0.70, "min"),
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


def family_accuracy(rows: list[dict[str, Any]], family: str, key: str = "exact") -> float:
    subset = [row for row in rows if row["family"] == family]
    return rate(sum(1 for row in subset if row[key]), len(subset))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_abstain = [row for row in rows if row["expected_status"] in {"missing_evidence", "ambiguous"}]
    predicted_abstain = [row for row in rows if row["predicted_status"] in {"missing_evidence", "ambiguous"}]
    file_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["file_path"]:
            file_groups.setdefault(row["file_path"], []).append(row)
    return {
        "episode_count": len(rows),
        "exact_answer_accuracy": rate(sum(1 for row in rows if row["exact"]), len(rows)),
        "canonical_object_accuracy": rate(sum(1 for row in rows if row["canonical_exact"]), len(rows)),
        "evidence_chunk_accuracy": rate(sum(1 for row in rows if row["evidence_exact"]), len(rows)),
        "retrieval_top1_accuracy": rate(sum(1 for row in rows if row["retrieval_evaluated"] and row["retrieval_exact"]), sum(1 for row in rows if row["retrieval_evaluated"])),
        "field_extraction_accuracy": family_accuracy(rows, "FIELD_EXTRACTION"),
        "metric_comparison_accuracy": family_accuracy(rows, "METRIC_COMPARISON"),
        "result_summary_accuracy": family_accuracy(rows, "RESULT_SUMMARY_CANONICAL", "canonical_exact"),
        "cross_doc_chain_accuracy": family_accuracy(rows, "CROSS_DOC_NEXT_CHAIN"),
        "caveat_boundary_accuracy": family_accuracy(rows, "CAVEAT_BOUNDARY_DETECTION"),
        "noisy_context_repair_accuracy": family_accuracy(rows, "NOISY_CONTEXT_REPAIR"),
        "long_context_memory_accuracy": family_accuracy(rows, "LONG_CONTEXT_MEMORY"),
        "table_row_extraction_accuracy": family_accuracy(rows, "TABLE_ROW_EXTRACTION"),
        "abstain_precision": rate(sum(1 for row in predicted_abstain if row["exact"]), len(predicted_abstain)),
        "abstain_recall": rate(sum(1 for row in expected_abstain if row["predicted_status"] == row["expected_status"]), len(expected_abstain)),
        "ambiguity_handling_accuracy": family_accuracy(rows, "AMBIGUOUS_OR_MISSING_EVIDENCE"),
        "hallucinated_answer_rate": rate(sum(1 for row in rows if row["hallucinated"]), len(rows)),
        "wrong_evidence_rate": rate(sum(1 for row in rows if row["wrong_evidence"]), len(rows)),
        "trace_validity": mean([row["trace_validity"] for row in rows]),
        "wrong_writeback_rate": rate(sum(1 for row in rows if row["wrong_writeback"]), len(rows)),
        "destructive_overwrite_rate": rate(sum(1 for row in rows if row["destructive_overwrite"]), len(rows)),
        "branch_contamination_rate": rate(sum(1 for row in rows if row["branch_contamination"]), len(rows)),
        "renderer_faithfulness": mean([row["renderer_faithful"] for row in rows]),
        "cost_per_episode": mean([row["cost"] for row in rows]),
        "cost_per_chunk": rounded(sum(row["cost"] for row in rows) / max(1, sum(row["context_chunk_count"] for row in rows))),
        "heldout_file_accuracy": mean([rate(sum(1 for row in group if row["exact"]), len(group)) for group in file_groups.values()]),
        "heldout_document_accuracy": mean([rate(sum(1 for row in group if row["exact"]), len(group)) for group in file_groups.values()]),
        "heldout_milestone_accuracy": rate(sum(1 for row in rows if row["file_path"] and re.search(r"/E\\d|^E\\d", row["file_path"]) and row["exact"]), sum(1 for row in rows if row["file_path"] and re.search(r"/E\\d|^E\\d", row["file_path"]))),
        "heldout_table_accuracy": family_accuracy(rows, "TABLE_ROW_EXTRACTION"),
        "heldout_numeric_accuracy": family_accuracy(rows, "METRIC_COMPARISON"),
        "out_of_train_heading_accuracy": rate(sum(1 for row in rows if row["out_of_train_heading"] and row["exact"]), sum(1 for row in rows if row["out_of_train_heading"])),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    positives = (
        metrics["exact_answer_accuracy"] * 2.2
        + metrics["canonical_object_accuracy"] * 1.8
        + metrics["evidence_chunk_accuracy"] * 1.4
        + metrics["retrieval_top1_accuracy"] * 1.2
        + metrics["ambiguity_handling_accuracy"] * 1.2
        + metrics["trace_validity"]
        + metrics["renderer_faithfulness"]
    )
    penalties = metrics["hallucinated_answer_rate"] * 2.0 + metrics["wrong_evidence_rate"] * 1.5 + metrics["cost_per_episode"] * 0.015
    return rounded(positives - penalties)


def gate_checks(metrics: dict[str, Any], systems: dict[str, dict[str, Any]], flags: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, (threshold, mode) in GATE_THRESHOLDS.items():
        value = metrics.get(key)
        if mode == "min":
            checks[f"{key}_at_least_{threshold}"] = value >= threshold
        else:
            checks[f"{key}_at_most_{threshold}"] = value <= threshold
    checks["exact_beats_bm25_by_0.08"] = rounded(metrics["exact_answer_accuracy"] - systems[BM25]["exact_answer_accuracy"]) >= 0.08
    checks["canonical_beats_static_by_0.08"] = rounded(metrics["canonical_object_accuracy"] - systems[STATIC]["canonical_object_accuracy"]) >= 0.08
    checks["aggregate_recomputed_from_episode_logs"] = flags["aggregate_recomputed_from_episode_logs"] is True
    checks["source_fixture_audit_passed"] = flags["source_fixture_audit_passed"] is True
    checks["deterministic_replay_passed"] = flags["deterministic_replay_passed"] is True
    checks["checker_failure_count_zero"] = flags["checker_failure_count"] == 0
    return checks


def recompute_training_curve(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for generation in sorted({row["generation"] for row in score_rows}):
        rows = [row for row in score_rows if row["generation"] == generation]
        best = max(rows, key=lambda row: row["validation_score"])
        curve.append(
            {
                "generation": generation,
                "best_candidate_id": best["candidate_id"],
                "best_train_score": best["train_score"],
                "best_validation_score": best["validation_score"],
                "best_validation_exact_answer_accuracy": best["validation_metrics"]["exact_answer_accuracy"],
                "best_validation_canonical_object_accuracy": best["validation_metrics"]["canonical_object_accuracy"],
            }
        )
    return curve


def compare_metric_block(failures: list[dict[str, Any]], label: str, expected: dict[str, Any], actual: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if actual.get(key) != expected.get(key):
            add_failure(failures, "RECOMPUTED_METRIC_MISMATCH", f"{label}:{key}:{actual.get(key)} != {expected.get(key)}")


def source_audit(failures: list[dict[str, Any]]) -> None:
    text = RUNNER.read_text(encoding="utf-8")
    lowered = text.lower()
    forbidden = (
        "def base_metrics",
        "static final metric table",
        "hardcoded final primary",
        "interpolated training curve",
        "torch",
        "tensorflow",
        "sklearn",
    )
    for pattern in forbidden:
        if pattern in lowered:
            add_failure(failures, "SOURCE_FORBIDDEN_PATTERN", pattern)
    tree = ast.parse(text)
    allowed = {"__future__", "argparse", "dataclasses", "hashlib", "json", "pathlib", "random", "re", "time", "typing"}
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
    marker = "def execute_policy"
    if marker in text:
        body = text.split(marker, 1)[1].split("\ndef ", 1)[0]
        if ".expected_answer" in body or ".expected_canonical" in body:
            add_failure(failures, "EXPECTED_USED_DURING_INFERENCE", "execute_policy")
        if ".family" in body:
            add_failure(failures, "RAW_FAMILY_USED_DURING_INFERENCE", "execute_policy")


def check_boundary(out: Path, failures: list[dict[str, Any]]) -> None:
    paths = [out / "report.md", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY_TEXT not in text:
            add_failure(failures, "BOUNDARY_TEXT_MISSING", str(path))
        lower = text.replace(BOUNDARY_TEXT, "").lower()
        positive_claims = (
            "proves general natural-language",
            "proves internet-scale",
            "production ready",
            "confirms agi",
            "confirms consciousness",
        )
        for claim in positive_claims:
            if claim in lower:
                add_failure(failures, "BROAD_POSITIVE_CLAIM_FOUND", f"{path}:{claim}")


def check_reports(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    per_episode = load_json(out / "e17_per_episode_eval_report.json")
    generation_scores = load_json(out / "e17_generation_score_report.json")
    training_curve = load_json(out / "e17_training_curve_report.json")
    split = load_json(out / "e17_corpus_split_report.json")
    source_fixture = load_json(out / "e17_source_fixture_audit_report.json")
    replay = load_json(out / "e17_deterministic_replay_report.json")
    failure_map = load_json(out / "e17_failure_map_report.json")
    systems = aggregate.get("systems", {})
    for system in SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY or aggregate.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_MISMATCH", f"{decision.get('primary_system')} / {aggregate.get('primary_system')}")
    if systems.get(HAND, {}).get("invalid_for_proof") is not True:
        add_failure(failures, "HAND_CONTROL_NOT_MARKED_INVALID", HAND)
    if decision.get("primary_system") == HAND:
        add_failure(failures, "HAND_CONTROL_SELECTED_AS_PRIMARY", HAND)
    leakage = split.get("leakage_audit", {})
    if leakage.get("split_by_file") is not True or leakage.get("passed") is not True:
        add_failure(failures, "SPLIT_LEAKAGE_AUDIT_FAILED", str(leakage))
    train_files = set(split.get("train_files", []))
    heldout_files = set(split.get("heldout_files", []))
    validation_files = set(split.get("validation_files", []))
    if train_files.intersection(heldout_files):
        add_failure(failures, "TRAIN_HELDOUT_FILE_OVERLAP", str(sorted(train_files.intersection(heldout_files))))
    if train_files.intersection(validation_files):
        add_failure(failures, "TRAIN_VALIDATION_FILE_OVERLAP", str(sorted(train_files.intersection(validation_files))))
    if validation_files.intersection(heldout_files):
        add_failure(failures, "VALIDATION_HELDOUT_FILE_OVERLAP", str(sorted(validation_files.intersection(heldout_files))))
    if per_episode.get("derived_from_policy_execution") is not True:
        add_failure(failures, "PER_EPISODE_NOT_DERIVED_FROM_EXECUTION", "e17_per_episode_eval_report.json")
    if source_fixture.get("source_fixture_audit_passed") is not True:
        add_failure(failures, "SOURCE_FIXTURE_AUDIT_NOT_PASSED", "e17_source_fixture_audit_report.json")
    if source_fixture.get("primary_metrics_from_static_final_tables") is not False:
        add_failure(failures, "STATIC_FINAL_METRIC_TABLE_FLAG_NOT_FALSE", "e17_source_fixture_audit_report.json")
    if source_fixture.get("oracle_expected_answers_used_during_inference") is not False:
        add_failure(failures, "ORACLE_EXPECTED_ANSWERS_USED_FLAG_NOT_FALSE", "e17_source_fixture_audit_report.json")
    if source_fixture.get("raw_task_family_labels_route_answer_selection") is not False:
        add_failure(failures, "RAW_TASK_FAMILY_ROUTE_FLAG_NOT_FALSE", "e17_source_fixture_audit_report.json")
    if replay.get("deterministic_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e17_deterministic_replay_report.json")

    metric_keys = (
        "episode_count",
        "exact_answer_accuracy",
        "canonical_object_accuracy",
        "evidence_chunk_accuracy",
        "retrieval_top1_accuracy",
        "field_extraction_accuracy",
        "metric_comparison_accuracy",
        "result_summary_accuracy",
        "cross_doc_chain_accuracy",
        "caveat_boundary_accuracy",
        "noisy_context_repair_accuracy",
        "long_context_memory_accuracy",
        "table_row_extraction_accuracy",
        "abstain_precision",
        "abstain_recall",
        "ambiguity_handling_accuracy",
        "hallucinated_answer_rate",
        "wrong_evidence_rate",
        "trace_validity",
        "wrong_writeback_rate",
        "destructive_overwrite_rate",
        "branch_contamination_rate",
        "renderer_faithfulness",
        "cost_per_episode",
        "cost_per_chunk",
        "heldout_file_accuracy",
        "heldout_document_accuracy",
        "heldout_milestone_accuracy",
        "heldout_table_accuracy",
        "heldout_numeric_accuracy",
        "out_of_train_heading_accuracy",
    )
    rows = per_episode.get("rows", [])
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            add_failure(failures, "MISSING_PER_EPISODE_ROWS_FOR_SYSTEM", system)
            continue
        recomputed = compute_metrics(system_rows)
        compare_metric_block(failures, system, recomputed, systems[system], metric_keys)
    if PRIMARY in systems and BM25 in systems and STATIC in systems:
        primary = systems[PRIMARY]
        if primary.get("delta_vs_bm25_exact_answer_accuracy") != rounded(primary["exact_answer_accuracy"] - systems[BM25]["exact_answer_accuracy"]):
            add_failure(failures, "BM25_DELTA_MISMATCH", str(primary.get("delta_vs_bm25_exact_answer_accuracy")))
        if primary.get("delta_vs_static_canonical_object_accuracy") != rounded(primary["canonical_object_accuracy"] - systems[STATIC]["canonical_object_accuracy"]):
            add_failure(failures, "STATIC_DELTA_MISMATCH", str(primary.get("delta_vs_static_canonical_object_accuracy")))
        flags = {
            "aggregate_recomputed_from_episode_logs": aggregate.get("aggregate_recomputed_from_episode_logs"),
            "source_fixture_audit_passed": aggregate.get("source_fixture_audit_passed"),
            "deterministic_replay_passed": aggregate.get("deterministic_replay_passed"),
            "checker_failure_count": decision.get("checker_failure_count"),
        }
        expected_gate = gate_checks(primary, systems, flags)
        if aggregate.get("positive_gate", {}).get("checks") != expected_gate:
            add_failure(failures, "GATE_CHECKS_MISMATCH", "aggregate_metrics.json")
        gate_passed = all(expected_gate.values())
        if aggregate.get("positive_gate", {}).get("passed") is not gate_passed:
            add_failure(failures, "GATE_PASSED_MISMATCH", "aggregate_metrics.json")
        if decision.get("positive_gate_passed") is not gate_passed:
            add_failure(failures, "DECISION_GATE_MISMATCH", "decision.json")
        meaningful = primary["delta_vs_bm25_exact_answer_accuracy"] >= 0.03 or primary["delta_vs_static_canonical_object_accuracy"] >= 0.03
        expected_decision = "e17_repo_text_mutation_training_overnight_confirmed" if gate_passed else (
            "e17_repo_text_mutation_training_overnight_partial" if meaningful else "e17_repo_text_mutation_training_overnight_failed"
        )
        if decision.get("decision") != expected_decision:
            add_failure(failures, "DECISION_MATH_MISMATCH", f"{decision.get('decision')} != {expected_decision}")
    recomputed_curve = recompute_training_curve(generation_scores.get("rows", []))
    if training_curve.get("curve") != recomputed_curve:
        add_failure(failures, "TRAINING_CURVE_RECOMPUTE_MISMATCH", "e17_training_curve_report.json")
    if recomputed_curve:
        expected_overfit = rounded(recomputed_curve[-1]["best_train_score"] - recomputed_curve[-1]["best_validation_score"])
        if training_curve.get("overfit_gap") != expected_overfit:
            add_failure(failures, "OVERFIT_GAP_RECOMPUTE_MISMATCH", str(training_curve.get("overfit_gap")))
        if summary.get("overfit_gap") != expected_overfit:
            add_failure(failures, "SUMMARY_OVERFIT_GAP_MISMATCH", str(summary.get("overfit_gap")))
    if failure_map.get("failure_map_complete") is not True:
        add_failure(failures, "FAILURE_MAP_INCOMPLETE", "e17_failure_map_report.json")
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
    result = {"schema_version": "e17_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e17_repo_text_mutation_training_overnight_audit")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
