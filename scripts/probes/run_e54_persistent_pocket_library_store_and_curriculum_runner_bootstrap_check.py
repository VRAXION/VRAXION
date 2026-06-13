#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E54_PERSISTENT_POCKET_LIBRARY_STORE_AND_CURRICULUM_RUNNER_BOOTSTRAP"
SYSTEMS = {
    "artifact_report_only_control",
    "unsafe_store_no_guards_control",
    "python_persistent_store_no_stress",
    "python_persistent_store_plus_adversarial_stress",
    "oracle_store_reference",
}
DECISIONS = {
    "e54_python_persistent_library_runtime_confirmed",
    "e54_store_integrity_failure_detected",
    "e54_adversarial_guard_failure",
    "e54_promotion_pipeline_incomplete",
    "e54_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "store_schema.json",
    "curriculum_manifest.json",
    "curriculum_rows.jsonl",
    "adversarial_stress_rows.jsonl",
    "store_integrity_report.json",
    "curriculum_runner_report.json",
    "adversarial_stress_report.json",
    "promotion_pipeline_report.json",
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
    "curriculum_rows_sample.jsonl",
    "adversarial_stress_rows_sample.jsonl",
    "store_integrity_sample_report.json",
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


def summarize(curriculum: list[dict[str, Any]], stress: list[dict[str, Any]], promotions: list[dict[str, Any]], store_root: Path | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    systems = sorted({row["system"] for row in curriculum} | {row["system"] for row in stress})
    for system in systems:
        curr = [row for row in curriculum if row["system"] == system]
        rows = [row for row in stress if row["system"] == system]
        expected_block = [row for row in rows if row["expected_block"]]
        valid = [row for row in rows if not row["expected_block"]]
        promo = [row for row in promotions if row.get("system") == system]
        safe_promotions = [row for row in promo if row.get("allowed") and not row.get("bad_promotion")]
        bad_promotions = [row for row in promo if row.get("bad_promotion")]
        registry_count = 0
        artifact_count = 0
        ledger_complete = 0.0
        if store_root is not None:
            root = store_root / system
            registry_path = root / "registry.json"
            if registry_path.exists():
                registry_count = len(read_json(registry_path).get("pockets", {}))
            artifact_dir = root / "artifacts"
            artifact_count = len(list(artifact_dir.glob("*.json"))) if artifact_dir.exists() else 0
            ledger_complete = 1.0 if all((root / name).exists() for name in ["lifecycle_ledger.jsonl", "access_ledger.jsonl", "promotion_ledger.jsonl", "score_ledger.jsonl"]) else 0.0
        out[system] = {
            "curriculum_success_rate": mean([1.0 if row["success"] else 0.0 for row in curr]),
            "avg_cost_to_success": mean([row["cost_to_success"] for row in curr]),
            "reuse_rate": mean([row["reuse_rate"] for row in curr]),
            "valid_load_success_rate": mean([1.0 if row["passed"] else 0.0 for row in valid]) if valid else 0.0,
            "adversarial_block_rate": mean([1.0 if row["blocked"] else 0.0 for row in expected_block]) if expected_block else 0.0,
            "unsafe_load_rate": mean([row["unsafe_loaded"] for row in rows]),
            "digest_mismatch_block_rate": attack_rate(rows, "direct_artifact_tamper"),
            "token_swap_block_rate": attack_rate(rows, "token_pocket_swap"),
            "abi_mismatch_block_rate": attack_rate(rows, "abi_mismatch"),
            "quarantine_block_rate": attack_rate(rows, "quarantine_load"),
            "banned_block_rate": attack_rate(rows, "banned_load"),
            "stale_token_block_rate": attack_rate(rows, "stale_token"),
            "alias_rename_survival": allow_rate(rows, "alias_rename"),
            "concurrent_stale_write_block_rate": attack_rate(rows, "concurrent_stale_write"),
            "unsafe_promotion_block_rate": attack_rate(rows, "unsafe_promotion"),
            "bad_promotion_rate": len(bad_promotions) / len(promo) if promo else 0.0,
            "safe_promotion_count": float(len(safe_promotions)),
            "registry_entry_count": float(registry_count),
            "artifact_count": float(artifact_count),
            "persistent_reload_match": 1.0 if registry_count == artifact_count and registry_count > 0 else 0.0,
            "ledger_complete": ledger_complete,
            "library_quality_delta": round(0.055 * len(safe_promotions) - 0.15 * len(bad_promotions), 6),
        }
    return out


def attack_rate(rows: list[dict[str, Any]], attack_type: str) -> float:
    subset = [row for row in rows if row["attack_type"] == attack_type]
    return mean([1.0 if row["blocked"] else 0.0 for row in subset]) if subset else 0.0


def allow_rate(rows: list[dict[str, Any]], attack_type: str) -> float:
    subset = [row for row in rows if row["attack_type"] == attack_type]
    return mean([1.0 if row["allowed"] else 0.0 for row in subset]) if subset else 0.0


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
    rows = read_jsonl(sample_dir / "curriculum_rows_sample.jsonl")
    stress = read_jsonl(sample_dir / "adversarial_stress_rows_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("persistent_python_store") is not True:
        failures.append("sample schema missing E54 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not stress:
        failures.append("sample rows empty")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e54_persistent_pocket_library_store_and_curriculum_runner_bootstrap.py")))
    manifest = read_json(out / "backend_manifest.json")
    schema = read_json(out / "store_schema.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    store_report = read_json(out / "store_integrity_report.json")
    system_results = read_json(out / "system_results.json")
    curriculum = read_jsonl(out / "curriculum_rows.jsonl")
    stress = read_jsonl(out / "adversarial_stress_rows.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if schema.get("registry_schema") != "PocketLibraryStore-v1":
        failures.append("store schema mismatch")
    if set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not curriculum or not stress or not progress or not heartbeat:
        failures.append("empty required row/progress artifacts")
    primary_store = out / "persistent_library" / "python_persistent_store_plus_adversarial_stress"
    if not primary_store.exists() or not (primary_store / "registry.json").exists() or not (primary_store / "tokens.json").exists():
        failures.append("primary persistent library store missing")
    primary_stress_types = {row["attack_type"] for row in stress if row["system"] == "python_persistent_store_plus_adversarial_stress"}
    required_attacks = {
        "valid_load",
        "alias_rename",
        "token_pocket_swap",
        "abi_mismatch",
        "quarantine_load",
        "banned_load",
        "stale_token",
        "direct_artifact_tamper",
        "unsafe_promotion",
        "concurrent_stale_write",
    }
    if not required_attacks.issubset(primary_stress_types):
        failures.append("primary stress suite missing required attack types")

    promotions = []
    persistent_root = out / "persistent_library"
    if persistent_root.exists():
        for promotion_path in persistent_root.glob("*/promotion_ledger.jsonl"):
            promotions.extend(read_jsonl(promotion_path))
    recomputed = summarize(curriculum, stress, promotions, out / "persistent_library")
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    primary = system_results["python_persistent_store_plus_adversarial_stress"]["overall"]
    unsafe = system_results["unsafe_store_no_guards_control"]["overall"]
    if aggregate.get("decision") == "e54_python_persistent_library_runtime_confirmed":
        required_equal_one = [
            "curriculum_success_rate",
            "valid_load_success_rate",
            "adversarial_block_rate",
            "digest_mismatch_block_rate",
            "token_swap_block_rate",
            "abi_mismatch_block_rate",
            "quarantine_block_rate",
            "banned_block_rate",
            "stale_token_block_rate",
            "alias_rename_survival",
            "concurrent_stale_write_block_rate",
            "unsafe_promotion_block_rate",
            "persistent_reload_match",
            "ledger_complete",
        ]
        for key in required_equal_one:
            if primary[key] != 1.0:
                failures.append(f"positive decision without primary {key}=1")
        for key in ["unsafe_load_rate", "bad_promotion_rate"]:
            if primary[key] != 0.0:
                failures.append(f"positive decision with primary {key} nonzero")
        if primary["safe_promotion_count"] < 2:
            failures.append("positive decision without two safe promotions")
        if primary["library_quality_delta"] <= 0.0:
            failures.append("positive decision without library quality gain")
        if unsafe["unsafe_load_rate"] <= 0.0 or unsafe["adversarial_block_rate"] >= 1.0:
            failures.append("positive decision but unsafe control did not fail")
        if store_report.get("registry_entries") != primary["registry_entry_count"]:
            failures.append("store report registry count mismatch")

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
