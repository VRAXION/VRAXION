#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E53_POCKET_LIBRARY_CUMULATIVE_TRANSFER_BOOTSTRAP_PROBE"
SYSTEMS = {
    "no_library_fresh_runs",
    "frozen_seed_library_only",
    "governed_library_with_active_set",
    "governed_library_plus_next_mutation_slot",
    "governed_library_plus_e52_promotion_policy",
    "unsafe_library_no_governance_control",
    "oracle_library_reference",
}
DECISIONS = {
    "e53_cumulative_pocket_library_bootstrap_confirmed",
    "e53_library_no_transfer_benefit",
    "e53_unsafe_library_negative_transfer",
    "e53_next_mutation_without_e52_overpromotes",
    "e53_active_set_overprunes",
    "e53_invalid_oracle_or_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "seed_library_manifest.json",
    "fresh_run_case_manifest.json",
    "fresh_run_rows.jsonl",
    "library_events.jsonl",
    "library_state_history.json",
    "reuse_report.json",
    "transfer_bootstrap_report.json",
    "negative_transfer_report.json",
    "promotion_policy_report.json",
    "next_mutation_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "results_table.md",
    "report.md",
]

REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "fresh_run_rows_sample.jsonl",
    "library_events_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def static_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize_rows(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows} | {event["system"] for event in events}):
        chunk = [row for row in rows if row["system"] == system]
        event_chunk = [event for event in events if event["system"] == system]
        safe_promotions = [event for event in event_chunk if event["promoted_to_library"] and event["safe"]]
        bad_promotions = [event for event in event_chunk if event["bad_promotion"]]
        success_rows = [row for row in chunk if row["success"]]
        rare_rows = [row for row in chunk if row["rare_critical_needed"]]
        total_required = sum(row["required_count"] for row in chunk)
        total_reused = sum(row["reused_count"] for row in chunk)
        unique_safe = sorted({event["capability"] for event in safe_promotions})
        out[system] = {
            "fresh_run_count": float(len(chunk)),
            "fresh_run_success_rate": mean([1.0 if row["success"] else 0.0 for row in chunk]),
            "avg_cost_to_success": mean([row["cost_to_success"] for row in chunk]),
            "avg_cost_on_success": mean([row["cost_to_success"] for row in success_rows]) if success_rows else 0.0,
            "avg_mutation_attempts_to_success": mean([row["mutation_attempts"] for row in success_rows]) if success_rows else 0.0,
            "reuse_rate": total_reused / total_required if total_required else 0.0,
            "useful_reuse_rate": mean([row["active_set_precision"] for row in chunk]),
            "active_set_recall": mean([row["active_set_recall"] for row in chunk]),
            "unsafe_load_rate": mean([1.0 if row["unsafe_load"] > 0 else 0.0 for row in chunk]),
            "negative_transfer_rate": mean([row["negative_transfer"] for row in chunk]),
            "wrong_commit_rate": mean([row["wrong_commit"] for row in chunk]),
            "rare_critical_preservation": mean([row["rare_critical_preserved"] for row in rare_rows]) if rare_rows else 1.0,
            "new_useful_pocket_discovery_rate": len(unique_safe) / 3.0 if system in {"governed_library_plus_next_mutation_slot", "governed_library_plus_e52_promotion_policy", "oracle_library_reference"} else 0.0,
            "bad_promotion_rate": len(bad_promotions) / len(event_chunk) if event_chunk else 0.0,
            "accepted_mutations": float(sum(event["accepted"] for event in event_chunk)),
            "rejected_mutations": float(sum(event["rejected"] for event in event_chunk)),
            "rollback_count": float(sum(event["rollback_count"] for event in event_chunk)),
            "library_size_delta": float(len(safe_promotions) - len(bad_promotions)),
            "library_quality_delta": round(0.055 * len(safe_promotions) - 0.13 * len(bad_promotions), 6),
        }
    no_library_cost = out.get("no_library_fresh_runs", {}).get("avg_cost_to_success", 0.0)
    for metrics in out.values():
        cost = metrics["avg_cost_to_success"]
        metrics["cost_efficiency_gain_vs_no_library"] = (no_library_cost - cost) / no_library_cost if no_library_cost else 0.0
    return out


def compare_float(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    schema = read_json(sample_dir / "sample_schema.json")
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    rows = read_jsonl(sample_dir / "fresh_run_rows_sample.jsonl")
    events = read_jsonl(sample_dir / "library_events_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("cumulative_transfer") is not True:
        failures.append("sample schema missing E53 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not events:
        failures.append("sample rows/events empty")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(out / "checker_summary.json", result)
        return result

    failures.extend(static_policy_check(Path("scripts/probes/run_e53_pocket_library_cumulative_transfer_bootstrap_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "fresh_run_rows.jsonl")
    events = read_jsonl(out / "library_events.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not events or not progress or not heartbeat:
        failures.append("empty rows/events/progress/heartbeat")
    required_row_fields = {
        "system",
        "run_id",
        "success",
        "cost_to_success",
        "mutation_attempts",
        "reused_count",
        "required_count",
        "unsafe_load",
        "negative_transfer",
        "wrong_commit",
        "rare_critical_preserved",
    }
    required_event_fields = {
        "system",
        "capability",
        "candidate_id",
        "attempts",
        "accepted",
        "rejected",
        "rollback_count",
        "e52_policy_used",
        "promoted_to_library",
        "bad_promotion",
        "safe",
    }
    if rows and not required_row_fields.issubset(set(rows[0])):
        failures.append("fresh run row missing required fields")
    if events and not required_event_fields.issubset(set(events[0])):
        failures.append("library event missing required fields")
    for event in events:
        if event["rollback_count"] != event["rejected"]:
            failures.append(f"rollback mismatch for {event['system']}:{event['candidate_id']}")

    recomputed = summarize_rows(rows, events)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    primary = system_results["governed_library_plus_e52_promotion_policy"]["overall"]
    no_library = system_results["no_library_fresh_runs"]["overall"]
    unsafe = system_results["unsafe_library_no_governance_control"]["overall"]
    next_mut = system_results["governed_library_plus_next_mutation_slot"]["overall"]
    if aggregate.get("decision") == "e53_cumulative_pocket_library_bootstrap_confirmed":
        if primary["fresh_run_success_rate"] < 0.95:
            failures.append("positive decision without primary success threshold")
        if primary["cost_efficiency_gain_vs_no_library"] < 0.35:
            failures.append("positive decision without cost efficiency threshold")
        if primary["reuse_rate"] < 0.65:
            failures.append("positive decision without reuse threshold")
        if primary["new_useful_pocket_discovery_rate"] < 0.95:
            failures.append("positive decision without discovery threshold")
        for key in ["unsafe_load_rate", "negative_transfer_rate", "wrong_commit_rate", "bad_promotion_rate"]:
            if primary[key] != 0.0:
                failures.append(f"positive decision with nonzero primary {key}")
        if primary["library_quality_delta"] <= 0.0:
            failures.append("positive decision without library quality gain")
        if primary["rare_critical_preservation"] != 1.0:
            failures.append("positive decision without rare critical preservation")
        if no_library["fresh_run_success_rate"] >= primary["fresh_run_success_rate"]:
            failures.append("positive decision but no-library matched primary")
        if unsafe["unsafe_load_rate"] <= 0.0 or unsafe["negative_transfer_rate"] <= 0.0:
            failures.append("positive decision but unsafe control did not fail")
        if next_mut["bad_promotion_rate"] <= 0.0:
            failures.append("positive decision but next-mutation control did not overpromote")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample_result,
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            write_json(Path(args.sample_only) / "sample_only_checker_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["passed"] else 1)
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
