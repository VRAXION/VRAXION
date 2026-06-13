#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E52_GOLDEN_DISC_SCORING_AND_PROMOTION_POLICY_PROBE"
SYSTEMS = {
    "final_answer_only_promotion",
    "immediate_only_promotion",
    "popularity_promotion",
    "scalar_average_score_promotion",
    "full_vector_policy",
    "full_vector_policy_plus_challenger",
    "oracle_lifecycle_reference",
}
DECISIONS = {
    "e52_golden_disc_scoring_policy_confirmed",
    "e52_policy_partial",
    "e52_overpromotion_detected",
    "e52_rare_critical_false_prune_detected",
    "e52_invalid_oracle_or_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "pocket_score_inputs.json",
    "promotion_rows.jsonl",
    "score_vector_report.json",
    "hard_safety_gate_report.json",
    "challenger_report.json",
    "shadow_import_report.json",
    "rare_critical_report.json",
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
    "promotion_rows_sample.jsonl",
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        core_expected = [row for row in chunk if row["expected_status"] in {"Core", "True Golden Disc"}]
        rare = [row for row in chunk if row["rare_critical"]]
        credit_hijack = [row for row in chunk if row["credit_hijack"]]
        delayed_poison = [row for row in chunk if row["delayed_poison"]]
        negative_transfer = [row for row in chunk if row["negative_transfer"] > 0.0]
        redundant = [row for row in chunk if row["redundant_clone"]]
        unsafe_high = [row for row in chunk if row["unsafe_high_utility"]]
        scope_bad = [row for row in chunk if row["scope_violation"]]
        positive_pred = [
            row
            for row in chunk
            if row["predicted_status"] in {"Stable", "Local Golden", "Semi-Perma", "Core", "True Golden Disc"}
        ]
        harmful_pred = [
            row
            for row in chunk
            if row["credit_hijack"] or row["delayed_poison"] or row["unsafe_high_utility"]
        ]
        out[system] = {
            "pocket_count": float(len(chunk)),
            "promotion_accuracy": mean([1.0 if row["correct"] else 0.0 for row in chunk]),
            "weighted_lifecycle_credit": mean([row["credit"] for row in chunk]),
            "bad_core_promotion_rate": mean([1.0 if row["bad_core_promotion"] else 0.0 for row in chunk]),
            "missed_core_rate": mean([1.0 if row["missed_core"] else 0.0 for row in core_expected]) if core_expected else 0.0,
            "rare_critical_preservation": mean([1.0 if not row["rare_critical_false_prune"] else 0.0 for row in rare]) if rare else 1.0,
            "credit_hijack_block_rate": mean([1.0 if row["predicted_status"] in {"Deprecated", "Quarantine", "Candidate"} else 0.0 for row in credit_hijack]) if credit_hijack else 1.0,
            "delayed_poison_detection": mean([1.0 if row["predicted_status"] == "Quarantine" else 0.0 for row in delayed_poison]) if delayed_poison else 1.0,
            "negative_transfer_detection": mean([1.0 if row["predicted_status"] not in {"Core", "True Golden Disc", "Semi-Perma"} else 0.0 for row in negative_transfer]) if negative_transfer else 1.0,
            "redundant_clone_rejection": mean([1.0 if row["predicted_status"] in {"Deprecated", "Candidate"} else 0.0 for row in redundant]) if redundant else 1.0,
            "unsafe_high_utility_block_rate": mean([1.0 if not row["unsafe_high_utility_bad"] else 0.0 for row in unsafe_high]) if unsafe_high else 1.0,
            "scope_violation_block_rate": mean([1.0 if row["predicted_status"] not in {"Core", "True Golden Disc", "Semi-Perma"} else 0.0 for row in scope_bad]) if scope_bad else 1.0,
            "reload_transfer_success": mean([row["reload_transfer"] for row in positive_pred]) if positive_pred else 0.0,
            "long_horizon_no_harm": mean([1.0 if row["negative_transfer"] == 0.0 and row["wrong_commit"] == 0.0 else 0.0 for row in positive_pred]) if positive_pred else 0.0,
            "prune_false_positive": mean([1.0 if row["rare_critical_false_prune"] else 0.0 for row in chunk]),
            "demotion_correctness": mean([1.0 if row["predicted_status"] in {"Deprecated", "Quarantine"} else 0.0 for row in harmful_pred]) if harmful_pred else 0.0,
        }
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
    rows = read_jsonl(sample_dir / "promotion_rows_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("golden_disc_scoring") is not True:
        failures.append("sample schema missing E52 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows:
        failures.append("sample promotion rows empty")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e52_golden_disc_scoring_and_promotion_policy_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    pockets = read_json(out / "pocket_score_inputs.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "promotion_rows.jsonl")
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
    if not pockets or not rows or not progress or not heartbeat:
        failures.append("empty pockets/rows/progress/heartbeat")

    required_row_fields = {
        "system",
        "pocket_id",
        "expected_status",
        "predicted_status",
        "correct",
        "credit",
        "vector_score",
        "safety_gate_pass",
        "challenger_pass",
        "shadow_import_pass",
        "rare_critical",
        "credit_hijack",
        "delayed_poison",
        "redundant_clone",
        "unsafe_high_utility",
        "scope_violation",
        "bad_core_promotion",
        "missed_core",
        "rare_critical_false_prune",
        "unsafe_high_utility_bad",
    }
    if rows and not required_row_fields.issubset(set(rows[0])):
        failures.append("promotion row missing required fields")

    recomputed = summarize_rows(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    primary = system_results["full_vector_policy_plus_challenger"]["overall"]
    scalar = system_results["scalar_average_score_promotion"]["overall"]
    full = system_results["full_vector_policy"]["overall"]
    popularity = system_results["popularity_promotion"]["overall"]
    if aggregate.get("decision") == "e52_golden_disc_scoring_policy_confirmed":
        thresholds = {
            "promotion_accuracy": 0.90,
            "weighted_lifecycle_credit": 0.95,
            "rare_critical_preservation": 1.0,
            "credit_hijack_block_rate": 1.0,
            "delayed_poison_detection": 1.0,
            "negative_transfer_detection": 1.0,
            "redundant_clone_rejection": 1.0,
            "unsafe_high_utility_block_rate": 1.0,
            "scope_violation_block_rate": 1.0,
            "long_horizon_no_harm": 1.0,
        }
        for key, threshold in thresholds.items():
            if primary[key] < threshold:
                failures.append(f"positive decision without {key} threshold")
        if primary["bad_core_promotion_rate"] != 0.0 or primary["missed_core_rate"] != 0.0:
            failures.append("positive decision with bad/missed core promotion")
        if primary["reload_transfer_success"] < 0.90:
            failures.append("positive decision without reload transfer success")
        if scalar["bad_core_promotion_rate"] <= 0.0:
            failures.append("positive decision but scalar average did not overpromote")
        if full["bad_core_promotion_rate"] <= 0.0:
            failures.append("positive decision but vector without challenger did not overpromote")
        if popularity["rare_critical_preservation"] >= 1.0:
            failures.append("positive decision but popularity preserved rare-critical pocket")

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
