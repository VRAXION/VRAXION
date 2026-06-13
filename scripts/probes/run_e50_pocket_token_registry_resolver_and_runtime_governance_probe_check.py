#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E50_POCKET_TOKEN_REGISTRY_RESOLVER_AND_RUNTIME_GOVERNANCE_PROBE"
SYSTEMS = {
    "filename_alias_router_control",
    "uid_only_no_descriptor_control",
    "descriptor_token_router_no_guard",
    "registry_guard_only_static_active_set",
    "full_library_scan_control",
    "token_registry_manager_active_set",
    "oracle_registry_reference",
}
DECISIONS = {
    "e50_pocket_token_registry_governance_positive",
    "e50_alias_filename_control_sufficient",
    "e50_token_routing_without_guard_unsafe",
    "e50_active_set_overprunes",
    "e50_registry_guard_blocks_but_routing_weak",
    "e50_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "registry_schema.json",
    "pocket_registry.json",
    "pocket_tokens.json",
    "resolver_events.jsonl",
    "governance_report.json",
    "active_set_report.json",
    "token_swap_report.json",
    "alias_rename_report.json",
    "digest_integrity_report.json",
    "stale_token_report.json",
    "registry_guard_mutation_history.jsonl",
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
    "pocket_registry_sample.json",
    "pocket_tokens_sample.json",
    "resolver_events_sample.jsonl",
    "registry_guard_mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def summarize_events(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({event["system"] for event in events}):
        chunk = [event for event in events if event["system"] == system]
        valid = [event for event in chunk if event["expected_action"] == "CALL"]
        alias_rows = [event for event in chunk if event["scenario"] == "alias_rename"]
        digest_rows = [event for event in chunk if event["expected_action"] == "BLOCK_DIGEST"]
        swap_rows = [event for event in chunk if event["expected_action"] == "BLOCK_TOKEN_SWAP"]
        unsafe_rows = [event for event in chunk if event["expected_action"] == "BLOCK_UNSAFE"]
        stale_rows = [event for event in chunk if event["expected_action"] == "REAUDIT"]
        abi_rows = [event for event in chunk if event["expected_action"] == "BLOCK_ABI"]
        active_reduction = mean([1.0 - event["active_set_size"] / event["full_library_size"] for event in chunk])
        governance_success = mean([1.0 if event["governance_success"] else 0.0 for event in chunk])
        unsafe_load_rate = mean([1.0 if event["unsafe_load"] else 0.0 for event in chunk])
        avg_cost = mean([event["cost"] for event in chunk])
        out[system] = {
            "row_count": float(len(chunk)),
            "governance_success": governance_success,
            "route_accuracy": mean([1.0 if event["route_correct"] else 0.0 for event in valid]),
            "alias_rename_survival": mean([1.0 if event["governance_success"] else 0.0 for event in alias_rows]),
            "digest_mismatch_block_rate": mean([1.0 if event["action"] == "BLOCK_DIGEST" else 0.0 for event in digest_rows]),
            "token_swap_block_rate": mean([1.0 if event["action"] == "BLOCK_TOKEN_SWAP" else 0.0 for event in swap_rows]),
            "banned_quarantine_block_rate": mean([1.0 if event["action"] == "BLOCK_UNSAFE" else 0.0 for event in unsafe_rows]),
            "stale_token_reaudit_rate": mean([1.0 if event["action"] == "REAUDIT" else 0.0 for event in stale_rows]),
            "abi_mismatch_block_rate": mean([1.0 if event["action"] == "BLOCK_ABI" else 0.0 for event in abi_rows]),
            "unsafe_load_rate": unsafe_load_rate,
            "avg_active_set_size": mean([event["active_set_size"] for event in chunk]),
            "full_library_size": mean([event["full_library_size"] for event in chunk]),
            "active_set_reduction": active_reduction,
            "avg_cost": avg_cost,
            "cost_adjusted_utility": governance_success - 0.50 * unsafe_load_rate - 0.10 * avg_cost + 0.08 * active_reduction,
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
    events = read_jsonl(sample_dir / "resolver_events_sample.jsonl")
    mutation = read_jsonl(sample_dir / "registry_guard_mutation_history_sample.jsonl")
    registry = read_json(sample_dir / "pocket_registry_sample.json")
    tokens = read_json(sample_dir / "pocket_tokens_sample.json")
    if schema.get("milestone") != MILESTONE or schema.get("pocket_token_registry") is not True:
        failures.append("sample schema missing E50 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not events or not mutation or not registry or not tokens:
        failures.append("sample registry/tokens/events/mutation empty")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e50_pocket_token_registry_resolver_and_runtime_governance_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    schema = read_json(out / "registry_schema.json")
    registry = read_json(out / "pocket_registry.json")
    tokens = read_json(out / "pocket_tokens.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    governance = read_json(out / "governance_report.json")
    events = read_jsonl(out / "resolver_events.jsonl")
    mutation = read_jsonl(out / "registry_guard_mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if schema.get("pocket_uid") != "immutable" or schema.get("human_alias") != "human_alias_only":
        failures.append("registry schema missing uid/alias contract")
    if set(system_results) != SYSTEMS or set(governance) != SYSTEMS:
        failures.append("system/report set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not registry or not tokens or not events or not mutation or not progress or not heartbeat:
        failures.append("empty registry/tokens/events/mutation/progress/heartbeat")
    for uid, entry in registry.items():
        if uid != entry.get("pocket_uid"):
            failures.append(f"registry uid mismatch for {uid}")
        if uid not in tokens:
            failures.append(f"missing token for registry uid {uid}")
        if entry.get("human_alias") == uid:
            failures.append(f"human alias equals uid for {uid}")
    for uid, token in tokens.items():
        if uid != token.get("pocket_uid"):
            failures.append(f"token uid mismatch for {uid}")
        if "descriptor_vector" not in token or "capability_signature" not in token:
            failures.append(f"token missing descriptor fields for {uid}")

    required_event_fields = {
        "system",
        "row_id",
        "scenario",
        "expected_action",
        "expected_uid",
        "token_uid",
        "content_uid",
        "proposed_uid",
        "resolved_uid",
        "action",
        "correct_action",
        "route_correct",
        "governance_success",
        "unsafe_load",
        "active_set_size",
        "full_library_size",
        "cost",
    }
    if events and not required_event_fields.issubset(set(events[0])):
        failures.append("resolver event missing required fields")

    recomputed = summarize_events(events)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    primary_report = system_results["token_registry_manager_active_set"]["overall"]
    if primary_report.get("accepted", 0) <= 0 or primary_report.get("rejected", 0) <= 0:
        failures.append("primary manager missing accepted/rejected mutation evidence")
    if primary_report.get("rollback_count") != primary_report.get("rejected"):
        failures.append("primary manager rollback mismatch")
    if not primary_report.get("parameter_diff_written") or not primary_report.get("parameter_diff_hash"):
        failures.append("primary manager missing parameter diff/hash")

    if aggregate.get("decision") == "e50_pocket_token_registry_governance_positive":
        checks = {
            "governance_success": 0.95,
            "route_accuracy": 0.95,
            "alias_rename_survival": 0.95,
            "digest_mismatch_block_rate": 1.0,
            "token_swap_block_rate": 1.0,
            "banned_quarantine_block_rate": 1.0,
            "stale_token_reaudit_rate": 1.0,
            "abi_mismatch_block_rate": 1.0,
        }
        for key, threshold in checks.items():
            if primary_report[key] < threshold:
                failures.append(f"positive decision without {key} threshold")
        if primary_report["unsafe_load_rate"] != 0.0:
            failures.append("positive decision with unsafe load")
        if primary_report["active_set_reduction"] < 0.25:
            failures.append("positive decision without active-set reduction")
        if primary_report["cost_adjusted_utility"] <= system_results["full_library_scan_control"]["overall"]["cost_adjusted_utility"]:
            failures.append("positive decision without beating full library scan cost utility")

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
