#!/usr/bin/env python3
"""Static closure check for STABLE_LOOP_PHASE_LOCK_055.

This checker reads committed files only. It does not run cargo, does not run
the frontend build, and does not write target artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EPSILON = 1e-9
SCHEMA_VERSION = "visual_snapshot_v1"
RUN_ID = "stable_loop_phase_lock_055_real_run_replay_closure"
REQUIRED_EVENT_KINDS = {"mutation", "prune", "repair", "crystallize"}
REQUIRED_BOUNDARY_PHRASES = [
    "real-metric visual projection",
    "not raw internal topology",
    "not a new training result",
    "not production dashboard",
    "not production API",
    "not full VRAXION",
    "not language grounding",
    "not consciousness",
]
EXPECTED_METRICS = {
    0: {
        "source_arm": "NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE",
        "heldout_score": 0.060546875,
        "ood_score": 0.048828125,
        "family_min_accuracy": 0.0,
        "unique_output_count": 1,
        "expected_output_class_count": 75,
        "top_output_rate": 1.0,
        "collapse_detected": True,
    },
    50: {
        "source_arm": "FROZEN_EVAL_048_REFERENCE",
        "heldout_score": 0.166015625,
        "ood_score": 0.15625,
        "family_min_accuracy": 0.0,
        "unique_output_count": 4,
        "expected_output_class_count": 75,
        "top_output_rate": 0.8935546875,
        "collapse_detected": True,
    },
    100: {
        "source_arm": "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER",
        "heldout_score": 1.0,
        "ood_score": 1.0,
        "family_min_accuracy": 1.0,
        "hard_distractor_accuracy": 1.0,
        "long_ood_accuracy": 1.0,
        "unique_output_count": 75,
        "expected_output_class_count": 75,
        "top_output_rate": 0.0732421875,
        "majority_output_rate": 0.0546875,
        "output_entropy": 5.40437231483324,
        "collapse_detected": False,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        print("This checker supports --check-only only.", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[2]
    visual = repo / "docs/research/visual_samples/055_real_run_replay_closure/visual"
    sample_bundle = repo / "tools/visual_lab/src/lib/sample-bundle.ts"
    demo_readme = repo / "docs/research/STABLE_LOOP_PHASE_LOCK_055_VISUAL_DEMO_README.md"
    contract = repo / "docs/research/STABLE_LOOP_PHASE_LOCK_055_VISUAL_SECTION_CLOSURE_REAL_RUN_REPLAY_CONTRACT.md"
    result = repo / "docs/research/STABLE_LOOP_PHASE_LOCK_055_VISUAL_SECTION_CLOSURE_REAL_RUN_REPLAY_RESULT.md"
    route_files = [
        repo / "tools/visual_lab/src/routes/topology/+page.svelte",
        repo / "tools/visual_lab/src/routes/playback/+page.svelte",
        repo / "tools/visual_lab/src/routes/diff/+page.svelte",
        repo / "tools/visual_lab/src/routes/metrics/+page.svelte",
    ]
    required_files = [
        visual / "schema_version.json",
        visual / "run_manifest.json",
        visual / "checkpoint_index.jsonl",
        visual / "metrics.jsonl",
        visual / "mutation_events.jsonl",
        visual / "route_traces.jsonl",
        visual / "pocket_summaries.jsonl",
        visual / "graph/checkpoint_000.json",
        visual / "graph/checkpoint_050.json",
        visual / "graph/checkpoint_100.json",
        visual / "ticks/checkpoint_100_tick_000.json",
        visual / "ticks/checkpoint_100_tick_001.json",
        sample_bundle,
        demo_readme,
        contract,
        result,
        *route_files,
    ]

    missing_files = [str(path.relative_to(repo)) for path in required_files if not path.exists()]
    missing_sections: list[str] = []
    alignment_failures: list[str] = []
    verdicts: list[str] = []

    schema_ok = False
    event_kinds: set[str] = set()
    bundle_selector_ok = False
    default_bundle_ok = False
    route_loader_ok = False
    boundary_ok = False

    if not missing_files:
        schema = read_json(visual / "schema_version.json")
        manifest = read_json(visual / "run_manifest.json")
        schema_ok = schema.get("schema_version") == SCHEMA_VERSION
        if not schema_ok:
            alignment_failures.append("schema_version != visual_snapshot_v1")
        if manifest.get("run_id") != RUN_ID:
            alignment_failures.append("run_manifest.run_id mismatch")

        metrics = {row["checkpoint"]: row for row in read_jsonl(visual / "metrics.jsonl")}
        alignment_failures.extend(validate_metrics(metrics))

        events = read_jsonl(visual / "mutation_events.jsonl")
        event_kinds = {str(row.get("kind")) for row in events}
        missing_event_kinds = sorted(REQUIRED_EVENT_KINDS - event_kinds)
        if missing_event_kinds:
            alignment_failures.append(f"missing event kinds: {missing_event_kinds}")

        sample_text = sample_bundle.read_text(encoding="utf-8")
        bundle_selector_ok = all(
            token in sample_text
            for token in [
                "055_real_run_replay_closure",
                "054_larger_playback_smoke",
                "053_real_run_ingest",
                "052_smoke_minimal",
            ]
        )
        default_bundle_ok = "activeSampleBundle = closureReplayBundle" in sample_text

        route_loader_ok = all(
            "visualSampleBundles" in path.read_text(encoding="utf-8")
            and "bundleById" in path.read_text(encoding="utf-8")
            for path in route_files
        )

        doc_text = "\n".join(path.read_text(encoding="utf-8") for path in [demo_readme, contract, result])
        boundary_ok = all(phrase in doc_text for phrase in REQUIRED_BOUNDARY_PHRASES)
        if not boundary_ok:
            missing_sections.append("claim boundary")

        readme_text = demo_readme.read_text(encoding="utf-8")
        for section in [
            "055_real_run_replay_closure",
            "Topology",
            "Playback",
            "Diff",
            "Metrics",
            "checkpoint 000",
            "checkpoint 050",
            "checkpoint 100",
            "049/050",
            "not a new model result",
        ]:
            if section not in readme_text:
                missing_sections.append(f"demo README: {section}")

        if not bundle_selector_ok:
            alignment_failures.append("055 or prior bundles missing from selector source")
        if not default_bundle_ok:
            alignment_failures.append("055 is not default bundle")
        if not route_loader_ok:
            alignment_failures.append("one or more routes do not use shared bundle loader")

    check_pass = (
        not missing_files
        and not missing_sections
        and not alignment_failures
        and schema_ok
        and event_kinds >= REQUIRED_EVENT_KINDS
        and bundle_selector_ok
        and default_bundle_ok
        and route_loader_ok
        and boundary_ok
    )

    if check_pass:
        verdicts.extend(
            [
                "VISUAL_REAL_RUN_REPLAY_POSITIVE",
                "VISUAL_METRICS_ALIGNMENT_POSITIVE",
                "VISUAL_EVENT_TIMELINE_REAL_DERIVED",
                "VISUAL_DEMO_BUNDLE_POSITIVE",
                "VISUAL_SECTION_V1_CLOSED",
                "PRODUCTION_DASHBOARD_NOT_READY",
            ]
        )
    else:
        verdicts.append("VISUAL_SECTION_V1_NOT_CLOSED")
        if alignment_failures:
            verdicts.append("VISUAL_METRICS_ALIGNMENT_FAILS")
        if REQUIRED_EVENT_KINDS - event_kinds:
            verdicts.append("VISUAL_EVENT_TIMELINE_MISSING")
        if missing_files or missing_sections:
            verdicts.append("VISUAL_DEMO_BUNDLE_FAILS")

    summary = {
        "check_pass": check_pass,
        "missing_files": missing_files,
        "missing_sections": missing_sections,
        "alignment_failures": alignment_failures,
        "schema_ok": schema_ok,
        "event_kinds": sorted(event_kinds),
        "bundle_selector_ok": bundle_selector_ok,
        "default_bundle_ok": default_bundle_ok,
        "route_loader_ok": route_loader_ok,
        "boundary_ok": boundary_ok,
        "verdicts": verdicts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if check_pass else 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_metrics(metrics: dict[int, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for checkpoint, expected in EXPECTED_METRICS.items():
        row = metrics.get(checkpoint)
        if row is None:
            failures.append(f"missing checkpoint {checkpoint}")
            continue
        for key, expected_value in expected.items():
            observed = row.get(key)
            if isinstance(expected_value, float):
                if not isinstance(observed, (int, float)) or abs(float(observed) - expected_value) > EPSILON:
                    failures.append(
                        f"checkpoint {checkpoint} {key}: expected {expected_value}, observed {observed}"
                    )
            else:
                if observed != expected_value:
                    failures.append(
                        f"checkpoint {checkpoint} {key}: expected {expected_value!r}, observed {observed!r}"
                    )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
