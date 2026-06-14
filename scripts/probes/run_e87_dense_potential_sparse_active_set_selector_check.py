#!/usr/bin/env python3
"""Checker for E87 dense-potential sparse active-set selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "library_manifest.json",
    "task_generation_report.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "selection_frequency_report.json",
    "selector_evolution_report.json",
    "counterfactual_report.json",
    "mutation_summary.json",
    "deterministic_replay.json",
    "decision.json",
    "report.md",
    "selector_history.jsonl",
    "row_level_samples.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e87_dense_potential_sparse_active_set_selector")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    failures: list[str] = []
    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")
    if not failures:
        manifest = read_json(out / "run_manifest.json")
        library = read_json(out / "library_manifest.json")
        task = read_json(out / "task_generation_report.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        frequency = read_json(out / "selection_frequency_report.json")
        counterfactual = read_json(out / "counterfactual_report.json")
        mutation = read_json(out / "mutation_summary.json")
        replay = read_json(out / "deterministic_replay.json")
        decision = read_json(out / "decision.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [line for line in (out / "selector_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (out / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_progress_dir = out / "seed_progress"
        seed_progress_files = list(seed_progress_dir.glob("seed_*.jsonl")) if seed_progress_dir.exists() else []
        if manifest.get("artifact_contract") != "E87_DENSE_POTENTIAL_SPARSE_ACTIVE_SET_SELECTOR":
            failures.append("artifact contract mismatch")
        if "not open-domain model training" not in manifest.get("boundary", ""):
            failures.append("boundary missing model-training boundary")
        if manifest.get("dense_potential_size", 0) != len(library.get("pockets", [])):
            failures.append("library size mismatch")
        if task.get("case_count", 0) < 100_000:
            failures.append("too few generated cases")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if not seed_progress_files:
            failures.append("missing per-seed progress files")
        expected_history_min = len(manifest.get("seeds", [])) * manifest.get("generations", 0)
        if len(history) < expected_history_min:
            failures.append("selector history too sparse")
        if len(samples) < 200:
            failures.append("row-level samples too sparse")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        if decision.get("decision") != "e87_dense_potential_sparse_selector_confirmed":
            failures.append("decision not confirmed")
        for key in [
            "validation_action_min",
            "adversarial_action_min",
        ]:
            if aggregate.get(key) != 1.0:
                failures.append(f"{key} != 1.0")
        for key in [
            "validation_false_call_max",
            "adversarial_false_call_max",
            "validation_false_commit_max",
            "adversarial_false_commit_max",
        ]:
            if aggregate.get(key) != 0.0:
                failures.append(f"{key} != 0")
        if aggregate.get("active_set_reduction_mean", 0.0) < 0.40:
            failures.append("active-set reduction too low")
        if aggregate.get("top_k_jaccard_mean", 0.0) < 0.85:
            failures.append("stable top jaccard too low")
        if aggregate.get("unsafe_final_selection_count") != 0:
            failures.append("unsafe pockets selected in final active set")
        if aggregate.get("accepted_mutations_total", 0) <= 0:
            failures.append("no accepted mutation evidence")
        if aggregate.get("rejected_mutations_total", 0) <= 0:
            failures.append("no rejected mutation evidence")
        if aggregate.get("rollback_count_total", 0) <= 0:
            failures.append("no rollback evidence")
        if len(frequency.get("stable_top", [])) < 5:
            failures.append("stable top set too small")
        unsafe_rows = [row for row in frequency.get("rows", []) if row.get("role") == "unsafe"]
        if any(row.get("selected_frequency", 1.0) > 0.0 for row in unsafe_rows):
            failures.append("unsafe pocket has nonzero final frequency")
        cf_summary = counterfactual.get("summary", {})
        if not any(values.get("mean_action_loss", 0.0) > 0.0 for values in cf_summary.values()):
            failures.append("counterfactual report has no positive contribution")
        replay_payload = {
            "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
            "selection_frequency": frequency,
            "counterfactual_summary": cf_summary,
        }
        if replay.get("hash") != deterministic_hash(replay_payload):
            failures.append("deterministic replay hash mismatch")
        if mutation.get("accepted_mutations_total") != aggregate.get("accepted_mutations_total"):
            failures.append("mutation summary accepted total mismatch")

    summary = {
        "checker": "E87_DENSE_POTENTIAL_SPARSE_ACTIVE_SET_SELECTOR_CHECK",
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
