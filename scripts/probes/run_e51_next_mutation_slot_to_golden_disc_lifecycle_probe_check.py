#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E51_NEXT_MUTATION_SLOT_TO_GOLDEN_DISC_LIFECYCLE_PROBE"
SYSTEMS = {
    "no_candidate_baseline",
    "parallel_candidate_spam_control",
    "light_probe_only_control",
    "refinement_without_uniqueness_control",
    "next_mutation_slot_to_golden_disc",
    "oracle_lifecycle_reference",
}
DECISIONS = {
    "e51_next_mutation_to_golden_disc_positive",
    "e51_light_probe_insufficient",
    "e51_parallel_candidate_spam_unsafe",
    "e51_refinement_without_uniqueness_overpromotes",
    "e51_no_unique_golden_value_detected",
    "e51_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "candidate_pool.json",
    "lifecycle_rows.jsonl",
    "light_probe_report.json",
    "active_refinement_report.json",
    "s_rank_report.json",
    "golden_disc_registry.json",
    "challenger_sweep_report.json",
    "prune_crystallization_report.json",
    "mutation_history.jsonl",
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
    "golden_registry_sample.json",
    "lifecycle_rows_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any] | list[Any]:
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
        golden = [row for row in chunk if row["predicted_stage"] == "GOLDEN_DISC"]
        expected_golden = [row for row in chunk if row["expected_stage"] == "GOLDEN_DISC"]
        light_pass = [row for row in chunk if row["light_probe_pass"]]
        stable_or_golden = [row for row in chunk if row["predicted_stage"] in {"STABLE", "S_RANK", "GOLDEN_DISC"}]
        active_slots = len([row for row in chunk if row["predicted_stage"] != "DISCARD"])
        golden_precision = mean([1.0 if row["expected_stage"] == "GOLDEN_DISC" else 0.0 for row in golden]) if golden else 0.0
        missed_golden_rate = mean([1.0 if row["missed_golden"] else 0.0 for row in expected_golden]) if expected_golden else 0.0
        challenger_defense = mean([1.0 if row["candidate_score"] >= row["challenger_best"] else 0.0 for row in golden]) if golden else 0.0
        prune_stability = mean([1.0 if row["prune_delta"] >= -0.01 else 0.0 for row in golden]) if golden else 0.0
        golden_quality = mean([mean([row["heldout"], row["ood"], row["counterfactual"], row["adversarial"], row["trace"]]) for row in golden]) if golden else 0.0
        out[system] = {
            "candidate_count": float(len(chunk)),
            "exact_stage_accuracy": mean([1.0 if row["predicted_stage"] == row["expected_stage"] else 0.0 for row in chunk]),
            "single_slot_integrity": 0.0 if active_slots > 1 and system == "parallel_candidate_spam_control" else 1.0,
            "slot_violation_rate": mean([1.0 if row["slot_violation"] else 0.0 for row in chunk]),
            "light_probe_precision": mean([1.0 if row["wrong_commit"] <= 0.03 and row["ood"] >= 0.90 else 0.0 for row in light_pass]) if light_pass else 0.0,
            "active_refinement_quality": mean([mean([row["heldout"], row["ood"], row["counterfactual"], row["adversarial"]]) for row in stable_or_golden]) if stable_or_golden else 0.0,
            "s_rank_precision": golden_precision,
            "golden_disc_count": float(len(golden)),
            "golden_disc_quality": golden_quality,
            "unique_value_score": mean([row["unique_value"] for row in golden]) if golden else 0.0,
            "challenger_defense_rate": challenger_defense,
            "prune_stability_rate": prune_stability,
            "bad_promotion_rate": mean([1.0 if row["unsafe_promotion"] else 0.0 for row in chunk]),
            "missed_golden_rate": missed_golden_rate,
            "wrong_commit_rate": mean([row["wrong_commit"] for row in golden]) if golden else 0.0,
            "direct_flow_write_violation_rate": mean([1.0 if row["direct_flow_write_violation"] else 0.0 for row in chunk]),
            "cost_adjusted_value": golden_quality + 0.20 * golden_precision - mean([row["cost"] for row in golden]) if golden else 0.0,
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
    rows = read_jsonl(sample_dir / "lifecycle_rows_sample.jsonl")
    mutation = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    golden = read_json(sample_dir / "golden_registry_sample.json")
    if schema.get("milestone") != MILESTONE or schema.get("next_mutation_lifecycle") is not True:
        failures.append("sample schema missing E51 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not mutation or not golden:
        failures.append("sample lifecycle/mutation/golden registry empty")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e51_next_mutation_slot_to_golden_disc_lifecycle_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    candidates = read_json(out / "candidate_pool.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "lifecycle_rows.jsonl")
    mutation = read_jsonl(out / "mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    golden = read_json(out / "golden_disc_registry.json")

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
    if not candidates or not rows or not mutation or not progress or not heartbeat:
        failures.append("empty candidates/rows/mutation/progress/heartbeat")

    required_row_fields = {
        "system",
        "slot_index",
        "candidate_id",
        "predicted_stage",
        "expected_stage",
        "light_probe_pass",
        "refinement_pass",
        "s_rank_pass",
        "unique_value",
        "challenger_best",
        "candidate_score",
        "prune_delta",
        "wrong_commit",
        "slot_violation",
        "direct_flow_write_violation",
        "unsafe_promotion",
        "missed_golden",
    }
    if rows and not required_row_fields.issubset(set(rows[0])):
        failures.append("lifecycle row missing required fields")

    recomputed = summarize_rows(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    primary = system_results["next_mutation_slot_to_golden_disc"]["overall"]
    if primary.get("accepted", 0) <= 0 or primary.get("rejected", 0) <= 0:
        failures.append("primary missing accepted/rejected mutation evidence")
    if primary.get("rollback_count") != primary.get("rejected"):
        failures.append("primary rollback mismatch")
    if not primary.get("parameter_diff_written") or not primary.get("parameter_diff_hash"):
        failures.append("primary missing parameter diff/hash")
    if not golden:
        failures.append("golden registry empty")
    for uid, entry in golden.items():
        if uid != entry.get("pocket_uid"):
            failures.append(f"golden uid mismatch {uid}")
        if not entry.get("frozen_anchor") or entry.get("lifecycle") != "golden_disc":
            failures.append(f"golden entry not frozen/lifecycle locked {uid}")
        if not entry.get("content_digest") or not entry.get("token"):
            failures.append(f"golden entry missing digest/token {uid}")

    if aggregate.get("decision") == "e51_next_mutation_to_golden_disc_positive":
        checks = {
            "exact_stage_accuracy": 0.99,
            "single_slot_integrity": 1.0,
            "s_rank_precision": 1.0,
            "golden_disc_quality": 0.999,
            "challenger_defense_rate": 1.0,
            "prune_stability_rate": 1.0,
        }
        for key, threshold in checks.items():
            if primary[key] < threshold:
                failures.append(f"positive decision without {key} threshold")
        if primary["golden_disc_count"] != 1:
            failures.append("positive decision without exactly one golden disc")
        if primary["unique_value_score"] < 0.05:
            failures.append("positive decision without unique value")
        if primary["bad_promotion_rate"] != 0.0 or primary["missed_golden_rate"] != 0.0:
            failures.append("positive decision with bad/missed promotion")
        if primary["wrong_commit_rate"] != 0.0 or primary["direct_flow_write_violation_rate"] != 0.0:
            failures.append("positive decision with wrong/direct write")
        if system_results["light_probe_only_control"]["overall"]["bad_promotion_rate"] <= 0.0:
            failures.append("positive decision but light probe control did not overpromote")
        if system_results["refinement_without_uniqueness_control"]["overall"]["bad_promotion_rate"] <= 0.0:
            failures.append("positive decision but no-uniqueness control did not overpromote")

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
