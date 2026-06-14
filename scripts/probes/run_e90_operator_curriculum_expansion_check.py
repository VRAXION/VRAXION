#!/usr/bin/env python3
"""Checker for E90 Operator curriculum expansion artifacts."""

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
        if manifest.get("artifact_contract") != "E90_OPERATOR_CURRICULUM_EXPANSION":
            failures.append("sample artifact contract mismatch")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != "E90_OPERATOR_CURRICULUM_EXPANSION":
            failures.append("artifact contract mismatch")
        if "not open-domain language understanding" not in manifest.get("boundary", ""):
            failures.append("boundary missing language-understanding caveat")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [line for line in (root / "operator_evolution_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_progress_dir = root / "seed_progress"
        seed_progress_files = list(seed_progress_dir.glob("seed_*.jsonl")) if seed_progress_dir.exists() else []
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")
        if not seed_progress_files:
            failures.append("missing per-seed progress")
        if len(history) < len(manifest.get("seeds", [])) * manifest.get("generations", 0):
            failures.append("operator evolution history too sparse")
        if len(samples) < 300:
            failures.append("row-level samples too sparse")

    if library.get("canonical_term") != "Operator":
        failures.append("library canonical term is not Operator")
    if library.get("legacy_alias") != "Pocket":
        failures.append("legacy alias missing")
    families = set(library.get("families", []))
    for family in ["alpha_syncer", "T-Stab", "Guard", "Lens", "Scribe"]:
        if family not in families:
            failures.append(f"missing family: {family}")
    if task.get("case_count", 0) < 8000:
        failures.append("too few generated text-evidence cases")
    if decision.get("failure_count") != 0:
        failures.append("decision failure_count != 0")
    if decision.get("decision") != "e90_operator_curriculum_expansion_confirmed":
        failures.append("decision not confirmed")
    for key in [
        "validation_resolution_success_min",
        "adversarial_resolution_success_min",
        "validation_trace_validity_min",
        "validation_evidence_span_validity_min",
    ]:
        if aggregate.get(key) != 1.0:
            failures.append(f"{key} != 1.0")
    for key in ["adversarial_wrong_confident_max", "validation_false_ask_max", "adversarial_false_commit_max"]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} != 0")
    if aggregate.get("accepted_mutations_total", 0) <= 0:
        failures.append("no accepted mutation evidence")
    if aggregate.get("rejected_mutations_total", 0) <= 0:
        failures.append("no rejected mutation evidence")
    if aggregate.get("rollback_count_total", 0) <= 0:
        failures.append("no rollback evidence")
    if summary.get("stable_operator_candidate_count", 0) < 10:
        failures.append("too few stable Operator candidates")
    if summary.get("unsafe_final_selected") != 0:
        failures.append("unsafe operators selected in final set")
    stable_top = set(frequency.get("stable_top", []))
    required_stable = {
        "visible_claim_binding_alpha_syncer",
        "numeric_value_binding_alpha_syncer",
        "temporal_rule_shift_t_stab",
        "false_alarm_temporal_t_stab",
        "revoked_binding_guard",
        "contradiction_guard",
        "unresolved_state_info_guard",
        "inactive_quote_scope_guard",
        "evidence_span_lens",
        "canonical_answer_scribe",
    }
    missing_stable = sorted(required_stable - stable_top)
    if missing_stable:
        failures.append(f"missing stable useful operators: {missing_stable}")
    unsafe_rows = [row for row in frequency.get("rows", []) if row.get("role") == "unsafe"]
    if any(row.get("selected_frequency", 1.0) > 0.0 for row in unsafe_rows):
        failures.append("unsafe operator has nonzero final frequency")
    lifecycle_rows = lifecycle.get("operator_lifecycle_table", [])
    if not any(row.get("family") == "alpha_syncer" and row.get("final_status") == "StableOperatorCandidate" for row in lifecycle_rows):
        failures.append("no stable alpha_syncer Operator")
    if not any(row.get("family") == "T-Stab" and row.get("final_status") == "StableOperatorCandidate" for row in lifecycle_rows):
        failures.append("no stable T-Stab Operator")
    if not any(row.get("family") == "Guard" and row.get("final_status") == "StableOperatorCandidate" for row in lifecycle_rows):
        failures.append("no stable Guard Operator")
    cf_summary = counterfactual.get("summary", {})
    if not any(values.get("mean_resolution_loss", 0.0) > 0.0 for values in cf_summary.values()):
        failures.append("counterfactual report has no positive resolution contribution")
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
    parser.add_argument("--out", default="target/pilot_wave/e90_operator_curriculum_expansion")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E90_OPERATOR_CURRICULUM_EXPANSION_CHECK",
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
