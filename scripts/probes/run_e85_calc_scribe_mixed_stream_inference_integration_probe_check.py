#!/usr/bin/env python3
"""Checker for E85 CALC-SCRIBE mixed-stream inference integration probe."""

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
    "integration_gate_report.json",
    "decision.json",
    "report.md",
    "row_level_samples.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e85_calc_scribe_mixed_stream_inference_integration_probe")
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
        gate = read_json(out / "integration_gate_report.json")
        decision = read_json(out / "decision.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (out / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest.get("artifact_contract") != "E85_CALC_SCRIBE_MIXED_STREAM_INFERENCE_INTEGRATION":
            failures.append("artifact contract mismatch")
        if manifest.get("scope") != "visible_calc_trace_validator":
            failures.append("scope mismatch")
        if "gsm8k_solver" not in manifest.get("not_claims", []):
            failures.append("missing gsm8k_solver not-claim")
        required_families = {
            "native_trace",
            "arrow_trace",
            "square_trace",
            "context_trace",
            "wrong_visible_trace",
            "gsm8k_question_no_trace",
            "gsm8k_final_answer_no_trace",
            "gsm8k_rationale_markers_stripped",
            "fineweb_text_no_trace",
            "fineweb_numeric_no_trace",
        }
        if not required_families.issubset(set(generation.get("route_families", []))):
            failures.append("missing required route families")
        if generation.get("case_count", 0) < 60_000:
            failures.append("too few mixed-stream cases")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if len(samples) < 100:
            failures.append("row-level samples too sparse")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        primary = systems.get("managed_active_set_transfer_router", {})
        native = systems.get("native_only_active_set", {})
        scan = systems.get("full_library_scan_no_scope_guard", {})
        alias = systems.get("alias_string_router_control", {})
        for split in ["validation", "adversarial"]:
            if primary.get(split, {}).get("route_min") != 1.0:
                failures.append(f"primary {split} route_min != 1.0")
            if primary.get(split, {}).get("action_min") != 1.0:
                failures.append(f"primary {split} action_min != 1.0")
            if primary.get(split, {}).get("false_call_max") != 0.0:
                failures.append(f"primary {split} false_call_max != 0")
            if primary.get(split, {}).get("false_commit_max") != 0.0:
                failures.append(f"primary {split} false_commit_max != 0")
        for family in required_families:
            if primary.get("family_action_min", {}).get(family) != 1.0:
                failures.append(f"primary family not clean: {family}")
        if native.get("validation", {}).get("action_min", 1.0) >= 1.0:
            failures.append("native-only control unexpectedly clean")
        if scan.get("validation", {}).get("false_call_max", 0.0) <= 0.0:
            failures.append("full scan control did not false-call")
        if alias.get("validation", {}).get("false_call_max", 0.0) <= 0.0:
            failures.append("alias router control did not false-call")
        if gate.get("primary_validation_route_min") != 1.0:
            failures.append("gate primary route min != 1")
        if gate.get("primary_validation_false_call_max") != 0.0:
            failures.append("gate primary false call max != 0")
    summary = {
        "checker": "E85_CALC_SCRIBE_MIXED_STREAM_INFERENCE_INTEGRATION_CHECK",
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
