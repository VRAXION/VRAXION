#!/usr/bin/env python3
"""Checker for E84 CALC-SCRIBE transfer + negative-scope probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "task_generation_report.json",
    "seed_results.json",
    "system_results.json",
    "aggregate_metrics.json",
    "negative_scope_report.json",
    "decision.json",
    "report.md",
    "row_level_samples.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e84_calc_scribe_transfer_negative_scope_probe")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []
    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")
    if not failures:
        manifest = read_json(out / "run_manifest.json")
        generation = read_json(out / "task_generation_report.json")
        systems = read_json(out / "system_results.json")
        negative = read_json(out / "negative_scope_report.json")
        decision = read_json(out / "decision.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (out / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest.get("artifact_contract") != "E84_CALC_SCRIBE_TRANSFER_AND_NEGATIVE_SCOPE_PROBE":
            failures.append("artifact contract mismatch")
        if manifest.get("scope") != "visible_calc_trace_validator":
            failures.append("scope mismatch")
        if "gsm8k_solver" not in manifest.get("not_claims", []):
            failures.append("missing gsm8k_solver not-claim")
        expected_formats = {
            "native_angle",
            "native_angle_spaced",
            "square_calc",
            "arrow_calc",
            "plain_equation_line",
            "unicode_operator_line",
            "context_wrapped_native",
            "wrong_result_native",
            "wrong_result_square",
            "broken_marker_native",
            "word_problem_no_marker",
            "final_answer_no_calc_trace",
            "rationale_without_calc_markers",
        }
        if not expected_formats.issubset(set(generation.get("format_families", []))):
            failures.append("missing required format families")
        if generation.get("case_count", 0) < 10_000:
            failures.append("too few generated transfer cases")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if len(samples) < 100:
            failures.append("row-level samples too sparse")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        primary = systems.get("calc_scribe_v004_transfer_router", {})
        native = systems.get("calc_scribe_v003_native_reload", {})
        overbroad = systems.get("overbroad_word_problem_solver_control", {})
        always = systems.get("always_commit_control", {})
        for split in ["validation", "adversarial"]:
            if primary.get(split, {}).get("action_min") != 1.0:
                failures.append(f"primary {split} action_min != 1.0")
            if primary.get(split, {}).get("false_call_max") != 0.0:
                failures.append(f"primary {split} false_call_max != 0")
            if primary.get(split, {}).get("false_commit_max") != 0.0:
                failures.append(f"primary {split} false_commit_max != 0")
        for key in ["valid_commit_min", "invalid_reject_min", "no_marker_no_call_min"]:
            if primary.get("validation", {}).get(key) != 1.0:
                failures.append(f"primary validation {key} != 1.0")
        format_min = primary.get("format_min", {})
        for name in expected_formats:
            if format_min.get(name) != 1.0:
                failures.append(f"primary format not clean: {name}")
        if native.get("validation", {}).get("action_min", 1.0) >= 1.0:
            failures.append("native-only control unexpectedly matched transfer router")
        if overbroad.get("validation", {}).get("false_call_max", 0.0) <= 0.0:
            failures.append("overbroad control did not produce false calls")
        if always.get("validation", {}).get("false_commit_max", 0.0) <= 0.0:
            failures.append("always-commit control did not produce false commits")
        if negative.get("primary_validation_no_marker_no_call_min") != 1.0:
            failures.append("negative-scope primary no-call min != 1.0")
        if negative.get("primary_adversarial_false_call_max") != 0.0:
            failures.append("negative-scope adversarial false call max != 0")
    summary = {
        "checker": "E84_CALC_SCRIBE_TRANSFER_NEGATIVE_SCOPE_CHECK",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary:
        (out / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
