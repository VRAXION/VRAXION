#!/usr/bin/env python3
"""Checker for E18B full-budget repo text stress artifacts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

PRIMARY = "MUTATION_TRAINED_PRUNED_STRESS_POLICY_PRIMARY"
INVALID_PRIMARY = {"SOURCE_PATH_ORACLE_CONTROL", "FIELD_NAME_ORACLE_CONTROL", "HAND_AUTHORED_EXTRACTOR_CONTROL"}
FULL_MINIMUMS = {
    "generations_completed": 40,
    "population_size": 64,
    "heldout_episode_count": 800,
    "stress_episode_count": 800,
    "candidate_count_evaluated": 2560,
    "checkpoint_count": 40,
}
FULL_GATES = {
    "exact_answer_accuracy": (">=", 0.78),
    "canonical_object_accuracy": (">=", 0.75),
    "evidence_chunk_accuracy": (">=", 0.80),
    "retrieval_top1_accuracy": (">=", 0.80),
    "no_source_path_accuracy": (">=", 0.70),
    "paraphrased_field_accuracy": (">=", 0.70),
    "same_key_conflict_accuracy": (">=", 0.70),
    "same_milestone_distractor_accuracy": (">=", 0.70),
    "target_not_first_accuracy": (">=", 0.75),
    "noisy_context_repair_accuracy": (">=", 0.70),
    "long_context_memory_accuracy": (">=", 0.70),
    "ambiguity_handling_accuracy": (">=", 0.75),
    "hallucinated_answer_rate": ("<=", 0.05),
    "wrong_evidence_rate": ("<=", 0.10),
    "trace_validity": (">=", 0.90),
    "renderer_faithfulness": (">=", 0.98),
}
FAMILY_METRICS = {
    "no_source_path_accuracy": "NO_SOURCE_PATH_FIELD_EXTRACTION",
    "paraphrased_field_accuracy": "PARAPHRASED_FIELD_EXTRACTION",
    "same_key_conflict_accuracy": "SAME_KEY_CONFLICT_RETRIEVAL",
    "same_milestone_distractor_accuracy": "SAME_MILESTONE_DISTRACTOR",
    "target_not_first_accuracy": "TARGET_NOT_FIRST_LONG_CONTEXT",
    "noisy_context_repair_accuracy": "ADVERSARIAL_NOISY_CONTEXT",
    "long_context_memory_accuracy": "LONG_CONTEXT_MEMORY",
    "table_row_extraction_accuracy": "TABLE_NUMERIC_STRESS",
    "metric_delta_accuracy": "METRIC_DELTA_STRESS",
    "ambiguity_handling_accuracy": "AMBIGUOUS_OR_MISSING_EVIDENCE",
    "missing_evidence_accuracy": "AMBIGUOUS_OR_MISSING_EVIDENCE",
}
REQUIRED_ARTIFACTS = [
    "decision.json", "summary.json", "aggregate_metrics.json", "report.md", "e18b_search_report.json",
    "e18b_corpus_manifest.json", "e18b_corpus_split_report.json", "e18b_episode_generation_report.json",
    "e18b_train_episode_manifest.json", "e18b_validation_episode_manifest.json", "e18b_heldout_episode_manifest.json",
    "e18b_stress_episode_manifest.json", "e18b_candidate_population_report.json", "e18b_generation_score_report.json",
    "e18b_training_curve_report.json", "e18b_checkpoint_report.json", "e18b_best_policy_report.json", "e18b_pruned_policy_report.json",
    "e18b_per_episode_eval_report.json", "e18b_system_comparison_report.json", "e18b_task_family_report.json", "e18b_ablation_report.json",
    "e18b_source_path_hint_ablation_report.json", "e18b_field_name_hint_ablation_report.json", "e18b_same_key_conflict_report.json",
    "e18b_same_milestone_distractor_report.json", "e18b_target_not_first_report.json", "e18b_table_numeric_report.json",
    "e18b_long_context_memory_report.json", "e18b_abstain_ambiguity_report.json", "e18b_latency_report.json", "e18b_trace_validity_report.json",
    "e18b_writeback_safety_report.json", "e18b_renderer_faithfulness_report.json", "e18b_source_fixture_audit_report.json",
    "e18b_deterministic_replay_report.json", "e18b_boundary_claims_report.json", "e18b_failure_map_report.json", "e18b_next_recommendation.json",
    "checkpoint_latest.json", "training_progress.jsonl",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(vals: List[float], pct: float) -> float:
    if not vals: return 0.0
    ordered = sorted(vals)
    k = (len(ordered)-1)*pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi: return float(ordered[lo])
    return float(ordered[lo]*(hi-k)+ordered[hi]*(k-lo))


def summarize(rows: List[Dict[str, Any]], system: str, split: str) -> Dict[str, float]:
    selected = [r for r in rows if r.get("system") == system and r.get("split") == split]
    def mb(key: str, subset: List[Dict[str, Any]] | None = None) -> float:
        subset = selected if subset is None else subset
        return sum(1 for r in subset if r.get(key)) / len(subset) if subset else 0.0
    out = {
        "episode_count": float(len(selected)),
        "exact_answer_accuracy": mb("exact_answer"),
        "canonical_object_accuracy": mb("canonical_object"),
        "evidence_chunk_accuracy": mb("evidence_chunk_correct"),
        "retrieval_top1_accuracy": mb("retrieval_top1_correct"),
        "hallucinated_answer_rate": mb("hallucinated_answer"),
        "wrong_evidence_rate": mb("wrong_evidence"),
        "trace_validity": mb("trace_valid"),
        "renderer_faithfulness": mb("renderer_faithful"),
    }
    for metric, family in FAMILY_METRICS.items():
        fam_rows = [r for r in selected if r.get("family") == family]
        out[metric] = mb("exact_answer", fam_rows)
    lat = [float(r.get("latency_ms", 0.0)) for r in selected]
    out.update({
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "latency_max_ms": max(lat) if lat else 0.0,
        "episodes_per_second": 1000.0 / (sum(lat)/len(lat)) if lat else 0.0,
    })
    return out


def close(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(float(a)-float(b)) <= eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-summary", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    failures: List[str] = []
    warnings: List[str] = []

    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"missing required artifact: {name}")
    if failures:
        print(json.dumps({"checker_failure_count": len(failures), "failures": failures}, indent=2))
        return 1

    summary = load_json(out / "summary.json")
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    logs_obj = load_json(out / "e18b_per_episode_eval_report.json")
    logs = logs_obj.get("logs", [])
    split = load_json(out / "e18b_corpus_split_report.json")
    source_audit = load_json(out / "e18b_source_fixture_audit_report.json")
    checkpoints = load_json(out / "e18b_checkpoint_report.json")
    generation_scores = load_json(out / "e18b_generation_score_report.json").get("generation_scores", [])
    curve = load_json(out / "e18b_training_curve_report.json").get("training_curve", [])
    pruned = load_json(out / "e18b_pruned_policy_report.json")
    boundary = load_json(out / "e18b_boundary_claims_report.json")

    primary = summary.get("primary_system")
    if primary in INVALID_PRIMARY:
        failures.append(f"invalid oracle/control selected as primary: {primary}")
    if pruned.get("oracle_control_selected_as_primary"):
        failures.append("pruned policy report says oracle control selected as primary")
    if not logs_obj.get("aggregate_recomputed_from_episode_logs"):
        failures.append("per-episode report does not declare recomputation")
    if not source_audit.get("source_fixture_audit_passed"):
        failures.append("source fixture audit failed")
    if not source_audit.get("split_leakage_audit_passed"):
        failures.append("split leakage audit failed")
    paths_by_split = split.get("splits", {})
    seen: Dict[str, str] = {}
    for split_name, paths in paths_by_split.items():
        for p in paths:
            if p in seen:
                failures.append(f"split overlap for {p}: {seen[p]} and {split_name}")
            seen[p] = split_name
    if not aggregate.get("aggregate_recomputed_from_episode_logs"):
        failures.append("aggregate_metrics does not declare recomputation from episode logs")

    heldout = summarize(logs, PRIMARY, "heldout")
    stress = summarize(logs, PRIMARY, "stress")
    bm25 = summarize(logs, "BM25_LIKE_BASELINE", "stress")
    static = summarize(logs, "STATIC_KEYWORD_BASELINE", "stress")
    stress["delta_vs_bm25_no_source_path_accuracy"] = stress["no_source_path_accuracy"] - bm25["no_source_path_accuracy"]
    stress["delta_vs_static_same_key_conflict_accuracy"] = stress["same_key_conflict_accuracy"] - static["same_key_conflict_accuracy"]
    with_hint = [r for r in logs if r.get("split") == "stress" and r.get("system") == PRIMARY and r.get("source_path_hint")]
    no_hint = [r for r in logs if r.get("split") == "stress" and r.get("system") == PRIMARY and not r.get("source_path_hint")]
    stress["source_path_hint_dependency_delta"] = (sum(r.get("exact_answer", False) for r in with_hint)/len(with_hint) if with_hint else 0.0) - (sum(r.get("exact_answer", False) for r in no_hint)/len(no_hint) if no_hint else 0.0)
    exact_hint = [r for r in logs if r.get("split") == "stress" and r.get("system") == PRIMARY and r.get("field_name_hint") in ["decision","next","primary_system","checker_failure_count","run_budget_class","positive_gate_passed"]]
    other_hint = [r for r in logs if r.get("split") == "stress" and r.get("system") == PRIMARY and r.get("field_name_hint") not in ["decision","next","primary_system","checker_failure_count","run_budget_class","positive_gate_passed"]]
    stress["field_name_hint_dependency_delta"] = (sum(r.get("exact_answer", False) for r in exact_hint)/len(exact_hint) if exact_hint else 0.0) - (sum(r.get("exact_answer", False) for r in other_hint)/len(other_hint) if other_hint else 0.0)

    for split_name, recomputed in [("heldout", heldout), ("stress", stress)]:
        recorded = aggregate.get(split_name, {})
        for key, value in recomputed.items():
            if key in recorded and not close(recorded[key], value, 1e-7):
                failures.append(f"aggregate mismatch {split_name}.{key}: recorded={recorded[key]} recomputed={value}")

    # Recompute training curve from generation scores.
    for item in curve:
        gen = item.get("generation")
        rows = [r for r in generation_scores if r.get("generation") == gen]
        if not rows:
            failures.append(f"training curve generation {gen} has no generation scores")
            continue
        best = max(float(r["validation_score"]) for r in rows)
        mean = statistics.fmean(float(r["validation_score"]) for r in rows)
        if not close(item.get("best_validation_score", -1), best, 1e-9) or not close(item.get("mean_validation_score", -1), mean, 1e-9):
            failures.append(f"training curve mismatch at generation {gen}")

    actual = summary.get("actual_budget", {})
    full_budget_met = all(int(actual.get(k, 0)) >= v for k, v in FULL_MINIMUMS.items())
    if decision.get("decision") == "e18b_full_budget_repo_text_stress_confirmed" and not full_budget_met:
        failures.append("full confirmed with too-small actual budget")
    full_gates_pass = full_budget_met and not failures
    for metric, (op, threshold) in FULL_GATES.items():
        value = stress.get(metric, 0.0)
        if op == ">=" and value < threshold: full_gates_pass = False
        if op == "<=" and value > threshold: full_gates_pass = False
    if decision.get("decision") == "e18b_full_budget_repo_text_stress_confirmed" and not full_gates_pass:
        failures.append("full confirmed without satisfying recomputed full gates")
    if int(actual.get("checkpoint_count", 0)) != int(checkpoints.get("checkpoint_count", -1)):
        failures.append("checkpoint count mismatch")
    if not math.isfinite(stress.get("latency_p95_ms", float("nan"))):
        failures.append("latency_p95_ms is not finite")

    report_text = (out / "report.md").read_text(encoding="utf-8")
    positive_broad_claims = [
        "proves general natural-language AI", "proves general natural language AI", "internet-scale LLM behavior confirmed",
        "production readiness confirmed", "AGI confirmed", "consciousness confirmed", "neural benchmark confirmed",
    ]
    for phrase in positive_broad_claims:
        if phrase.lower() in report_text.lower():
            failures.append(f"forbidden broad claim appears: {phrase}")
    if boundary.get("broad_claims_detected"):
        failures.append("boundary claims report detected broad claims")

    checker_failure_count = len(failures)
    corrected_decision = decision.get("decision")
    positive_gate = bool(summary.get("positive_gate_passed"))
    if checker_failure_count:
        corrected_decision = "e18b_full_budget_repo_text_stress_invalid_or_incomplete"
        positive_gate = False
    elif decision.get("decision") == "e18b_full_budget_repo_text_stress_confirmed" and not full_budget_met:
        corrected_decision = "e18b_full_budget_repo_text_stress_invalid_or_incomplete"
        positive_gate = False

    check_summary = {
        "checker_failure_count": checker_failure_count,
        "failures": failures,
        "warnings": warnings,
        "decision": corrected_decision,
        "positive_gate_passed": positive_gate,
        "full_budget_met": full_budget_met,
        "full_confirmation_forbidden": decision.get("decision") != "e18b_full_budget_repo_text_stress_confirmed",
        "source_fixture_audit_passed": source_audit.get("source_fixture_audit_passed"),
        "aggregate_recomputed_from_episode_logs": checker_failure_count == 0,
        "recomputed_heldout_metrics": heldout,
        "recomputed_stress_metrics": stress,
    }
    if args.write_summary:
        # Patch summary/decision with checker count and recomputed marker; keep runner budget/decision semantics unless invalid.
        summary["checker_failure_count"] = checker_failure_count
        summary["aggregate_recomputed_from_episode_logs"] = checker_failure_count == 0
        summary["heldout_metrics"] = heldout
        summary["stress_metrics"] = stress
        if checker_failure_count:
            summary["decision"] = corrected_decision
            summary["positive_gate_passed"] = False
            decision["decision"] = corrected_decision
            decision["positive_gate_passed"] = False
        decision["checker_failure_count"] = checker_failure_count
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "e18b_checker_summary.json").write_text(json.dumps(check_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(check_summary, indent=2, sort_keys=True))
    return 1 if checker_failure_count else 0

if __name__ == "__main__":
    raise SystemExit(main())
