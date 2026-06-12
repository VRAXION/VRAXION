#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from pathlib import Path
from typing import Any


MILESTONE = "E38A_TRAINING_EFFICIENT_PROFILE_SCOUT"
DECISIONS = {
    "e38a_training_efficient_profile_candidate_found",
    "e38a_profile_max_not_bounded_extend_sweep",
    "e38a_compute_bottleneck_before_quality",
    "e38a_invalid_artifact_detected",
}
SYSTEMS = {
    "no_library_scratch_quality_anchor",
    "stable_pocket_plus_adapter_quality_anchor",
    "profile_mutation_search",
    "gpu_batched_forward_probe",
}
REQ_TARGET = [
    "backend_manifest.json",
    "profile_config_report.json",
    "workload_generation_report.json",
    "profile_results.json",
    "quality_anchor_results.json",
    "gpu_bench_report.json",
    "profile_selection_report.json",
    "row_level_results.jsonl",
    "mutation_history.jsonl",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "profile_results_sample.json",
    "row_level_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-9) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval(", "hash("]:
        if token in source:
            failures.append(f"runner contains banned gradient/oracle/nondeterministic token: {token}")
    return failures


def summarize_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rates = [float(row["candidate_eval_per_sec"]) for row in rows]
    accepted_rates = [float(row["accepted_rate"]) for row in rows]
    accepted_sec = [float(row["accepted_mutations_per_sec"]) for row in rows]
    scores = [float(row["best_score"]) for row in rows]
    return {
        "seed_count": len(rows),
        "candidate_eval_per_sec_mean": statistics.fmean(eval_rates),
        "candidate_eval_per_sec_min": min(eval_rates),
        "accepted_rate_mean": statistics.fmean(accepted_rates),
        "accepted_mutations_per_sec_mean": statistics.fmean(accepted_sec),
        "best_score_mean": statistics.fmean(scores),
        "best_score_max": max(scores),
        "latency_p50_seconds_mean": statistics.fmean(float(row["latency_p50_seconds"]) for row in rows),
        "latency_p95_seconds_mean": statistics.fmean(float(row["latency_p95_seconds"]) for row in rows),
        "wall_time_seconds_total": sum(float(row["wall_time_seconds"]) for row in rows),
    }


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    profile_results = read_json(sample_dir / "profile_results_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if schema.get("milestone") != MILESTONE or schema.get("profile_scout") is not True:
        failures.append("sample schema missing E38A profile scout marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not profile_results:
        failures.append("sample profile results empty")
    if not rows or any("system" not in row for row in rows):
        failures.append("sample row-level rows missing system")
    if not history:
        failures.append("sample mutation history empty")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    runner = Path(__file__).resolve().with_name("run_e38a_training_efficient_profile_scout.py")
    failures.extend(static_runner_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    config = read_json(out / "profile_config_report.json")
    profile_results = read_json(out / "profile_results.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    selection = read_json(out / "profile_selection_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    history = read_jsonl(out / "mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if selection.get("selected_profile") != aggregate.get("selected_profile") or decision.get("selected_profile") != aggregate.get("selected_profile"):
        failures.append("selected profile mismatch")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not history or not progress or not heartbeat:
        failures.append("empty row/history/progress/heartbeat artifact")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("system manifest mismatch")
    if set(config.get("profiles", {})) != set(profile_results):
        failures.append("profile config/results mismatch")
    if any(row.get("system") not in SYSTEMS for row in rows):
        failures.append("unknown system in rows")

    profile_rows = [row for row in rows if row.get("system") == "profile_mutation_search"]
    for profile_id, reported in profile_results.items():
        sys_rows = [row for row in profile_rows if row.get("profile_id") == profile_id]
        if not sys_rows:
            failures.append(f"profile has no row-level mutation rows: {profile_id}")
            continue
        recomputed = summarize_profile(sys_rows)
        if recomputed["seed_count"] != reported.get("seed_count"):
            failures.append(f"seed_count mismatch {profile_id}")
        for key in [
            "candidate_eval_per_sec_mean",
            "candidate_eval_per_sec_min",
            "accepted_rate_mean",
            "accepted_mutations_per_sec_mean",
            "best_score_mean",
            "best_score_max",
            "latency_p50_seconds_mean",
            "latency_p95_seconds_mean",
            "wall_time_seconds_total",
        ]:
            compare_float(f"{profile_id}.{key}", recomputed[key], reported.get(key), failures, tolerance=1e-8)
        if int(sum(int(row.get("accepted_mutations", 0)) + int(row.get("rejected_mutations", 0)) for row in sys_rows)) <= 0:
            failures.append(f"profile had no mutation attempts: {profile_id}")

    selected = aggregate.get("selected_profile")
    if selected not in profile_results:
        failures.append("selected profile missing from profile results")
    if aggregate.get("decision") == "e38a_profile_max_not_bounded_extend_sweep":
        ordered = list(profile_results)
        if selected != ordered[-1]:
            failures.append("max-not-bounded decision did not select highest tested profile")
    if aggregate.get("decision") == "e38a_training_efficient_profile_candidate_found":
        ordered = list(profile_results)
        if selected == ordered[-1]:
            failures.append("candidate-found decision selected highest tested profile; should be max-not-bounded")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "selected_profile": aggregate.get("selected_profile"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample_result,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
