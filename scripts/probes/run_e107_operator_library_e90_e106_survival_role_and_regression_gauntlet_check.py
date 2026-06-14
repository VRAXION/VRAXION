#!/usr/bin/env python3
"""Checker for E107 Operator Library E90-E106 survival gauntlet artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E107_OPERATOR_LIBRARY_E90_E106_SURVIVAL_ROLE_AND_REGRESSION_GAUNTLET"

REQUIRED = [
    "run_manifest.json",
    "operator_library_manifest.json",
    "task_generation_report.json",
    "source_inventory_report.json",
    "neighborhood_results.json",
    "survival_role_report.json",
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
    "source_inventory_report.json",
    "neighborhood_results.json",
    "survival_role_report.json",
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
    inventory = read_json(root / "source_inventory_report.json")
    neighborhood = read_json(root / "neighborhood_results.json")
    survival = read_json(root / "survival_role_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    frequency = read_json(root / "selection_frequency_report.json")
    counterfactual = read_json(root / "counterfactual_report.json")
    lifecycle = read_json(root / "operator_lifecycle_report.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("sample artifact contract mismatch")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("artifact contract mismatch")
        boundary = manifest.get("boundary", "")
        if "not final training" not in boundary or "not open-domain capability evaluation" not in boundary:
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
        expected_history = manifest.get("neighborhood_count", 0) * len(manifest.get("seeds", []))
        if len(history) < expected_history:
            failures.append("operator evolution history too sparse")
        if len(samples) < 200:
            failures.append("row-level samples too sparse")

    if library.get("canonical_term") != "Operator":
        failures.append("canonical term is not Operator")
    if library.get("legacy_alias") != "Pocket":
        failures.append("legacy alias missing")
    groups = set(library.get("groups", []))
    expected_groups = {f"E{number}" for number in range(90, 107)}
    if groups != expected_groups:
        failures.append(f"E90-E106 group set mismatch: {sorted(groups)}")
    if inventory.get("candidate_operator_count", 0) < 130:
        failures.append("too few candidate operators")
    if inventory.get("operator_group_count") != 17:
        failures.append("operator group count is not 17")
    if task.get("quality_control_gauntlet") is not True:
        failures.append("quality_control_gauntlet not true")
    if task.get("operator_survival_ranking") is not True:
        failures.append("operator_survival_ranking not true")
    if task.get("open_domain_capability_eval") is not False:
        failures.append("open-domain capability eval unexpectedly enabled")
    if task.get("final_training") is not False:
        failures.append("final training unexpectedly enabled")
    if task.get("full_library_scan_allowed_as_success") is not False:
        failures.append("full-library scan success unexpectedly allowed")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e107_operator_library_survival_role_regression_gauntlet_confirmed":
        failures.append("decision not confirmed")
    for key in [
        "survival_success_min",
        "adversarial_survival_success_min",
        "group_coverage_min",
        "family_coverage_min",
        "role_assignment_coverage",
    ]:
        if aggregate.get(key) != 1.0:
            failures.append(f"{key} != 1.0")
    if aggregate.get("focus_coverage_min", 0.0) < 0.5:
        failures.append("focus coverage below 0.5")
    for key in ["unsafe_control_selected_rate", "full_library_overreach_rate", "cost_blowup_rate"]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} != 0")
    if aggregate.get("seed_count", 0) < 16:
        failures.append("seed count below 16")
    if aggregate.get("neighborhood_count", 0) < 14:
        failures.append("too few neighborhoods")
    if aggregate.get("stable_support_count", 0) <= 0 or aggregate.get("specialist_count", 0) <= 0:
        failures.append("missing stable/specialist role split")
    if aggregate.get("accepted_mutations_total", 0) <= 0 or aggregate.get("rejected_mutations_total", 0) <= 0 or aggregate.get("rollback_count_total", 0) <= 0:
        failures.append("missing accept/reject/rollback evidence")
    rows = neighborhood.get("rows", [])
    if len(rows) != aggregate.get("case_count"):
        failures.append("neighborhood rows do not match case count")
    if any(row.get("unsafe_control_selected") for row in rows):
        failures.append("unsafe control selected in neighborhood")
    controls = survival.get("controls", [])
    if not controls or any(row.get("final_status") not in {"Quarantine", "Deprecated"} for row in controls):
        failures.append("controls not quarantined/deprecated")
    lifecycle_rows = lifecycle.get("operator_lifecycle_table", [])
    statuses = {row.get("final_status") for row in lifecycle_rows if row.get("role") == "candidate"}
    if "StableSupport" not in statuses or "Specialist" not in statuses:
        failures.append("role report lacks StableSupport/Specialist")
    cf_summary = counterfactual.get("summary", {})
    if not any(values.get("mean_survival_score_loss", 0.0) > 0.0 for values in cf_summary.values()):
        failures.append("counterfactual has no positive survival contribution")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "selection_frequency": frequency,
        "counterfactual_summary": cf_summary,
        "lifecycle": lifecycle,
        "survival_role": survival,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E107_OPERATOR_LIBRARY_E90_E106_SURVIVAL_ROLE_AND_REGRESSION_GAUNTLET_CHECK",
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
