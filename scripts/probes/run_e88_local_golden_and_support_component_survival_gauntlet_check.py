#!/usr/bin/env python3
"""Checker for E88 LocalGolden/support survival gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "task_generation_report.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "component_survival_table.json",
    "counterfactual_ablation.json",
    "challenger_sweep.json",
    "negative_scope_report.json",
    "reload_import_stress_report.json",
    "long_horizon_no_harm_report.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_level_samples.jsonl",
]

SAMPLE_REQUIRED = [
    "sample_manifest.json",
    "component_survival_table.json",
    "counterfactual_ablation.json",
    "challenger_sweep.json",
    "negative_scope_report.json",
    "long_horizon_no_harm_report.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
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

    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    component = read_json(root / "component_survival_table.json")
    counterfactual = read_json(root / "counterfactual_ablation.json")
    challenger = read_json(root / "challenger_sweep.json")
    negative = read_json(root / "negative_scope_report.json")
    long_horizon = read_json(root / "long_horizon_no_harm_report.json")
    replay = read_json(root / "deterministic_replay.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != "E88_LOCAL_GOLDEN_AND_SUPPORT_COMPONENT_SURVIVAL_GAUNTLET":
            failures.append("sample artifact contract mismatch")
    else:
        manifest = read_json(root / "run_manifest.json")
        task = read_json(root / "task_generation_report.json")
        aggregate = read_json(root / "aggregate_metrics.json")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_progress_dir = root / "seed_progress"
        seed_progress_files = list(seed_progress_dir.glob("seed_*.jsonl")) if seed_progress_dir.exists() else []
        if manifest.get("artifact_contract") != "E88_LOCAL_GOLDEN_AND_SUPPORT_COMPONENT_SURVIVAL_GAUNTLET":
            failures.append("artifact contract mismatch")
        if "not open-domain model training" not in manifest.get("boundary", ""):
            failures.append("boundary mismatch")
        if task.get("case_count", 0) < 100_000:
            failures.append("too few cases")
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")
        if not seed_progress_files:
            failures.append("missing per-seed progress")
        if len(samples) < 200:
            failures.append("row-level samples too sparse")
        for key in [
            "validation_action_min",
            "adversarial_action_min",
            "negative_scope_no_call_rate",
            "reload_match_rate",
            "tamper_block_rate",
            "token_swap_block_rate",
            "unsafe_global_scope_block_rate",
            "long_horizon_no_harm_rate",
        ]:
            if aggregate.get(key) != 1.0:
                failures.append(f"{key} != 1.0")
        for key in [
            "validation_false_call_max",
            "adversarial_false_call_max",
            "validation_false_commit_max",
            "adversarial_false_commit_max",
            "challenger_beats_total",
        ]:
            if aggregate.get(key) != 0:
                failures.append(f"{key} != 0")
        payload = {"aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"}, "lifecycle": component, "variants": challenger}
        if replay.get("hash") != deterministic_hash(payload):
            failures.append("deterministic replay hash mismatch")

    if decision.get("failure_count") != 0:
        failures.append("decision failure_count != 0")
    if decision.get("decision") != "e88_local_golden_survival_gauntlet_confirmed":
        failures.append("unexpected decision")
    statuses = summary.get("component_statuses", {})
    if statuses.get("calc_scribe_v003") != "SpecialistGoldenCandidate":
        failures.append("CALC-SCRIBE did not reach SpecialistGoldenCandidate")
    if statuses.get("calc_scribe_native_seed") != "LocalGoldenConfirmed":
        failures.append("native seed not LocalGoldenConfirmed")
    if "Banned" not in set(statuses.values()):
        failures.append("no banned unsafe control recorded")
    if "Quarantine" not in set(statuses.values()):
        failures.append("no quarantined unsafe control recorded")
    if "Redundant" not in set(statuses.values()):
        failures.append("no redundant control recorded")
    if negative.get("negative_scope_no_call_rate") != 1.0:
        failures.append("negative scope no-call rate != 1.0")
    if long_horizon.get("long_horizon_no_harm_rate") != 1.0:
        failures.append("long horizon no-harm rate != 1.0")
    if len(component.get("component_survival_table", [])) < 16:
        failures.append("component survival table too small")
    if len(counterfactual.get("component_rows", [])) == 0:
        failures.append("missing counterfactual rows")
    if len(challenger.get("variants", [])) < 8:
        failures.append("challenger sweep too small")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e88_local_golden_and_support_component_survival_gauntlet")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E88_LOCAL_GOLDEN_AND_SUPPORT_COMPONENT_SURVIVAL_GAUNTLET_CHECK",
        "root": str(root),
        "sample_only": bool(args.sample_only),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary and not args.sample_only:
        (Path(args.out) / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
