#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E33B_GRADIENTLESS_FLOW_BREAKPOINT_AUDIT"
DECISIONS = {
    "e33b_gradientless_breakpoint_localized",
    "e33b_gradientless_all_controlled_clean",
    "e33b_gradientless_breakpoint_not_reproduced",
    "e33b_gradientless_artifact_invalid",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "breakpoint_ladder_report.json",
    "system_results.json",
    "row_level_results.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "breakpoint_ladder_sample.json",
    "row_level_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]
PRIMARY = {
    "E24": "flow_pocket_unsccaffolded_discovery_primary",
    "E25": "flow_pocket_naturalized_text_discovery_primary",
    "E26": "flow_pocket_hard_skip_primary",
    "E27": "flow_pocket_unresolved_information_seeking_primary",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows)


def static_policy_check(path: Path) -> list[str]:
    failures: list[str] = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_text = ["backward(", "AdamW", "SGD(", "RMSprop", "optim.", "loss_fn", "train_neural("]
    for token in banned_text:
        if token in source:
            failures.append(f"gradient harness token present in E33B runner: {token}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            failures.append("runner calls Python eval")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sympy":
                    failures.append("runner imports sympy")
        if isinstance(node, ast.ImportFrom) and node.module == "sympy":
            failures.append("runner imports sympy")
    return failures


def validate_ladder(rows: list[dict[str, Any]], ladder: dict[str, Any], failures: list[str]) -> None:
    for milestone, primary in PRIMARY.items():
        primary_rows = [row for row in rows if row["source_milestone"] == milestone and row["system"] == primary]
        if not primary_rows:
            failures.append(f"missing row-level primary rows for {milestone}")
            continue
        split_scores = {
            split: metric([row for row in primary_rows if row["split"] == split], "composition_success")
            for split in sorted({row["split"] for row in primary_rows})
        }
        reported = ladder.get(milestone, {}).get("primary", {}).get("split_composition_success", {})
        for split, score in split_scores.items():
            if split not in reported or not math.isclose(float(score), float(reported[split]), rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"metric mismatch {milestone}/{split}: rows={score} reported={reported.get(split)}")
        if ladder.get(milestone, {}).get("gradient_descent_used") is not False:
            failures.append(f"{milestone} ladder missing gradient_descent_used=false")


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failures": failures}
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    ladder = read_json(sample_dir / "breakpoint_ladder_sample.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if not rows:
        failures.append("empty row_level_sample.jsonl")
    if not all(row.get("source_milestone") in PRIMARY for row in rows):
        failures.append("sample rows contain unknown milestone")
    if not all(ladder.get(m, {}).get("gradient_descent_used") is False for m in PRIMARY):
        failures.append("sample ladder does not assert gradient_descent_used=false for all milestones")
    return {"passed": not failures, "failures": failures, "run_id": aggregate.get("run_id")}


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
    runner = Path(__file__).resolve().with_name("run_e33b_gradientless_flow_breakpoint_audit.py")
    failures.extend(static_policy_check(runner))
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    replay = read_json(out / "deterministic_replay.json")
    manifest = read_json(out / "backend_manifest.json")
    ladder = read_json(out / "breakpoint_ladder_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    if aggregate.get("milestone") != MILESTONE:
        failures.append("aggregate milestone mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision"):
        failures.append("decision.json mismatch")
    if aggregate.get("gradient_descent_used") is not False or manifest.get("gradient_descent_used") is not False:
        failures.append("gradient_descent_used is not false")
    if manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("optimizer/backprop flags are not false")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    validate_ladder(rows, ladder, failures)
    if any(row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary") for row in rows):
        failures.append("row-level leakage/direct eval flag detected")
    sample = validate_sample(sample_dir)
    if not sample["passed"]:
        failures.extend([f"sample: {item}" for item in sample["failures"]])
    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = read_json(out / "summary.json")
        summary["target_checker_passed"] = result["passed"]
        summary["sample_only_checker_passed"] = sample["passed"]
        summary["checker_failure_count"] = len(failures)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        parser.error("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
