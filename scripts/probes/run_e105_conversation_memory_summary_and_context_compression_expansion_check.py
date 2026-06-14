#!/usr/bin/env python3
"""Checker for E105 conversation memory summary/context compression artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "operator_library_manifest.json",
    "task_generation_report.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "selection_frequency_report.json",
    "counterfactual_report.json",
    "operator_lifecycle_report.json",
    "mutation_summary.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_level_samples.jsonl",
    "operator_evolution_history.jsonl",
]

SAMPLE_REQUIRED = [
    "sample_manifest.json",
    "operator_library_manifest.json",
    "task_generation_report.json",
    "aggregate_metrics.json",
    "selection_frequency_report.json",
    "counterfactual_report.json",
    "operator_lifecycle_report.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check_common(root: Path, sample_only: bool) -> list[str]:
    failures: list[str] = []
    required = SAMPLE_REQUIRED if sample_only else REQUIRED
    for name in required:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    library = read_json(root / "operator_library_manifest.json")
    task = read_json(root / "task_generation_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    frequency = read_json(root / "selection_frequency_report.json")
    counterfactual = read_json(root / "counterfactual_report.json")
    lifecycle = read_json(root / "operator_lifecycle_report.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != "E105_CONVERSATION_MEMORY_SUMMARY_AND_CONTEXT_COMPRESSION_EXPANSION":
            failures.append("sample artifact contract mismatch")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != "E105_CONVERSATION_MEMORY_SUMMARY_AND_CONTEXT_COMPRESSION_EXPANSION":
            failures.append("artifact contract mismatch")
        if "not open-domain summarization" not in manifest.get("boundary", ""):
            failures.append("boundary caveat missing")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [line for line in (root / "operator_evolution_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_progress_dir = root / "seed_progress"
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")
        if not seed_progress_dir.exists() or not list(seed_progress_dir.glob("seed_*.jsonl")):
            failures.append("missing per-seed progress")
        if len(history) < len(manifest.get("seeds", [])) * manifest.get("generations", 0):
            failures.append("operator evolution history too sparse")
        if len(samples) < 300:
            failures.append("row-level samples too sparse")

    if library.get("canonical_term") != "Operator":
        failures.append("canonical term is not Operator")
    if library.get("legacy_alias") != "Pocket":
        failures.append("legacy alias missing")
    families = set(library.get("families", []))
    for family in ["Lens", "Guard", "Scribe", "T-Stab"]:
        if family not in families:
            failures.append(f"missing family: {family}")
    if task.get("case_count", 0) < 7000:
        failures.append("too few context compression cases")
    if task.get("context_compression_summary") is not True:
        failures.append("context_compression_summary not true")
    if task.get("open_domain_summarization") is not False:
        failures.append("open-domain summarization unexpectedly enabled")
    if task.get("direct_summary_without_evidence_allowed") is not False:
        failures.append("direct summary without evidence unexpectedly allowed")
    if task.get("requires_citation_pointer_preservation") is not True:
        failures.append("citation pointer preservation missing")
    if task.get("requires_unresolved_dependency_preservation") is not True:
        failures.append("unresolved dependency preservation missing")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e105_context_compression_summary_expansion_confirmed":
        failures.append("decision not confirmed")
    for key in [
        "validation_context_compression_success_min",
        "adversarial_context_compression_success_min",
        "validation_required_fact_preservation_min",
        "validation_unresolved_dependency_preservation_min",
        "validation_citation_pointer_validity_min",
        "validation_stale_fact_exclusion_min",
        "validation_context_reentry_success_min",
        "validation_compression_ratio_validity_min",
    ]:
        if aggregate.get(key) != 1.0:
            failures.append(f"{key} != 1.0")
    for key in [
        "adversarial_hallucinated_summary_rate_max",
        "adversarial_lost_dependency_rate_max",
        "adversarial_stale_summary_rate_max",
        "adversarial_cross_thread_bleed_rate_max",
        "adversarial_overcompression_rate_max",
    ]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} != 0")
    if aggregate.get("accepted_mutations_total", 0) <= 0 or aggregate.get("rejected_mutations_total", 0) <= 0 or aggregate.get("rollback_count_total", 0) <= 0:
        failures.append("missing mutation accept/reject/rollback evidence")
    if summary.get("stable_operator_candidate_count", 0) < 8:
        failures.append("too few stable Operator candidates")
    if summary.get("unsafe_final_selected") != 0:
        failures.append("unsafe operators selected")
    required_stable = {
        "context_window_pressure_lens",
        "summary_relevance_span_selector_lens",
        "required_fact_preservation_guard",
        "unresolved_dependency_preservation_t_stab",
        "citation_pointer_compaction_scribe",
        "obsolete_turn_prune_guard",
        "summary_drift_detection_guard",
        "compressed_context_reentry_scribe",
    }
    stable_top = set(frequency.get("stable_top", []))
    missing_stable = sorted(required_stable - stable_top)
    if missing_stable:
        failures.append(f"missing stable operators: {missing_stable}")
    unsafe_rows = [row for row in frequency.get("rows", []) if row.get("role") == "unsafe"]
    if any(row.get("selected_frequency", 1.0) > 0.0 for row in unsafe_rows):
        failures.append("unsafe operator has nonzero frequency")
    lifecycle_rows = lifecycle.get("operator_lifecycle_table", [])
    for family in ["Lens", "Guard", "Scribe", "T-Stab"]:
        if not any(row.get("family") == family and row.get("final_status") == "StableOperatorCandidate" for row in lifecycle_rows):
            failures.append(f"no stable {family}")
    cf_summary = counterfactual.get("summary", {})
    if not any(values.get("mean_context_compression_success_loss", 0.0) > 0.0 for values in cf_summary.values()):
        failures.append("counterfactual has no positive compression contribution")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "selection_frequency": frequency,
        "counterfactual_summary": cf_summary,
        "lifecycle": lifecycle,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e105_conversation_memory_summary_and_context_compression_expansion")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E105_CONVERSATION_MEMORY_SUMMARY_AND_CONTEXT_COMPRESSION_EXPANSION_CHECK",
        "root": str(root),
        "sample_only": bool(args.sample_only),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary and not args.sample_only:
        (Path(args.out) / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
