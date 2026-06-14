#!/usr/bin/env python3
"""Checker for E108 external transfer/no-harm gauntlet artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E108_EXTERNAL_DATASET_OPERATOR_TRANSFER_AND_NEGATIVE_SCOPE_GAUNTLET"

REQUIRED = [
    "run_manifest.json",
    "dataset_manifest.json",
    "e107_role_input_report.json",
    "external_dataset_report.json",
    "policy_results.json",
    "role_transfer_report.json",
    "operator_usage_report.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "counterfactual_report.json",
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
    "dataset_manifest.json",
    "e107_role_input_report.json",
    "external_dataset_report.json",
    "policy_results.json",
    "role_transfer_report.json",
    "operator_usage_report.json",
    "aggregate_metrics.json",
    "counterfactual_report.json",
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
    for name in SAMPLE_REQUIRED if sample_only else REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    dataset = read_json(root / "dataset_manifest.json")
    e107_input = read_json(root / "e107_role_input_report.json")
    external = read_json(root / "external_dataset_report.json")
    policy = read_json(root / "policy_results.json")
    transfer = read_json(root / "role_transfer_report.json")
    usage = read_json(root / "operator_usage_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    counterfactual = read_json(root / "counterfactual_report.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("sample artifact contract mismatch")
        sample_count = manifest.get("sample_policy_eval_count")
        source_count = manifest.get("source_policy_eval_count")
        if not isinstance(sample_count, int) or sample_count < 512:
            failures.append("sample policy row count too small")
        if not isinstance(source_count, int) or source_count < sample_count:
            failures.append("sample policy source count invalid")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("artifact contract mismatch")
        boundary = manifest.get("boundary", "")
        for text in ["not Golden promotion", "not Core promotion", "not final training"]:
            if text not in boundary:
                failures.append(f"boundary caveat missing: {text}")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [line for line in (root / "operator_evolution_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")
        if not (root / "seed_progress").exists() or not list((root / "seed_progress").glob("seed_*.jsonl")):
            failures.append("missing per-seed progress")
        if len(samples) < 300:
            failures.append("row-level samples too sparse")
        if len(history) < len(manifest.get("seeds", [])):
            failures.append("operator evolution history too sparse")

    if dataset.get("artifact_contract") != CONTRACT:
        failures.append("dataset contract mismatch")
    if dataset.get("not_raw_web_claim") is not True:
        failures.append("dataset web-claim caveat missing")
    if len(dataset.get("families", [])) < 6:
        failures.append("too few external families")
    if set(dataset.get("splits", [])) != {"validation", "adversarial", "negative_scope"}:
        failures.append("split set mismatch")
    if e107_input.get("role_count", 0) < 130:
        failures.append("too few E107 roles")
    if external.get("phase_1") != "frozen E107 role policy; no new training":
        failures.append("phase 1 frozen policy missing")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e108_external_transfer_no_harm_positive":
        failures.append("decision not confirmed")
    if aggregate.get("seed_count", 0) < 16:
        failures.append("seed count below 16")
    if aggregate.get("case_count", 0) < 10000:
        failures.append("case count too small")
    for key in ["negative_transfer_rate", "wrong_scope_call_rate", "false_commit_rate", "false_answer_rate", "unsupported_answer_rate"]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} != 0")
    if aggregate.get("no_harm_rate") != 1.0 or aggregate.get("negative_scope_success") != 1.0:
        failures.append("no-harm/negative-scope not clean")
    if aggregate.get("external_validation_success", 0.0) < 0.98 or aggregate.get("external_adversarial_success", 0.0) < 0.98:
        failures.append("external success below threshold")
    if aggregate.get("activated_gain_mean", 0.0) <= 0.0 or aggregate.get("ablation_loss_mean", 0.0) <= 0.0:
        failures.append("missing activated gain/ablation signal")
    if aggregate.get("full_library_scan_negative_transfer_rate", 0.0) <= 0.0:
        failures.append("full-library overreach control did not fail")
    if aggregate.get("external_transfer_candidate_count", 0) <= 0 or aggregate.get("scoped_transfer_candidate_count", 0) <= 0:
        failures.append("missing transfer candidate split")
    if transfer.get("status_counts", {}).get("Quarantine", 0) != 0:
        failures.append("operator quarantined under frozen no-harm policy")
    rows = policy.get("rows", [])
    if sample_only:
        if len(rows) != read_json(root / "sample_manifest.json").get("sample_policy_eval_count"):
            failures.append("sample policy rows do not match manifest")
    elif len(rows) != aggregate.get("policy_eval_count"):
        failures.append("policy rows do not match eval count")
    frozen_rows = [row for row in rows if row.get("policy") == "e107_frozen_role_policy"]
    if any(row.get("invalid_oracle") for row in frozen_rows):
        failures.append("invalid oracle leaked into frozen policy")
    usage_rows = usage.get("rows", [])
    if len(usage_rows) != e107_input.get("role_count"):
        failures.append("operator usage row count mismatch")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "role_transfer": transfer,
        "operator_usage": usage,
        "counterfactual_summary": counterfactual.get("summary", {}),
        "dataset_manifest": dataset,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e108_external_dataset_operator_transfer_and_negative_scope_gauntlet")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E108_EXTERNAL_DATASET_OPERATOR_TRANSFER_AND_NEGATIVE_SCOPE_GAUNTLET_CHECK",
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
