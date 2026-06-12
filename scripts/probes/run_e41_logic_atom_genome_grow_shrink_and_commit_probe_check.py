#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E41_LOGIC_ATOM_GENOME_GROW_SHRINK_AND_COMMIT_PROBE"
SYSTEMS = {
    "oracle_proposal_commit_reference",
    "direct_write_logic_atom_baseline",
    "proposal_without_arbiter",
    "fixed_slot_proposal_arbiter",
    "grow_shrink_logic_atom_genome",
    "full_flow_painter_control",
    "random_genome_control",
}
DECISIONS = {
    "e41_logic_atom_grow_shrink_commit_positive",
    "e41_fixed_slots_sufficient_growth_not_needed",
    "e41_direct_write_sufficient",
    "e41_arbiter_required_but_growth_failed",
    "e41_full_flow_required",
    "e41_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "proposal_commit_report.json",
    "logic_atom_genome_report.json",
    "system_results.json",
    "footprint_report.json",
    "mutation_report.json",
    "row_level_results.jsonl",
    "footprint_frames.jsonl",
    "aggregate_metrics.json",
    "deterministic_replay.json",
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
    "system_results_sample.json",
    "row_level_sample.jsonl",
    "footprint_frame_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def validate_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in ["expected_action", "action", "action_correct", "proposals", "arbiter", "false_commit", "missed_commit", "footprint", "router_trace"]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")
    fp = row.get("footprint", {})
    for key in ["read_count", "write_count", "changed_count", "read_spread_ratio", "write_spread_ratio", "scan_cell_count", "illegal_write_count", "missed_target_write_count"]:
        if key not in fp:
            failures.append(f"{prefix}: footprint missing {key}")


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        out[system] = {
            "exact_rate": sum(1.0 if row["exact"] else 0.0 for row in chunk) / len(chunk),
            "action_accuracy": sum(1.0 if row["action_correct"] else 0.0 for row in chunk) / len(chunk),
            "cell_accuracy": sum(float(row["cell_accuracy"]) for row in chunk) / len(chunk),
            "false_commit_rate": sum(1.0 if row["false_commit"] else 0.0 for row in chunk) / len(chunk),
            "missed_commit_rate": sum(1.0 if row["missed_commit"] else 0.0 for row in chunk) / len(chunk),
            "proposal_count_mean": sum(float(row["proposal_count"]) for row in chunk) / len(chunk),
            "read_spread_ratio": sum(float(row["footprint"]["read_spread_ratio"]) for row in chunk) / len(chunk),
            "write_spread_ratio": sum(float(row["footprint"]["write_spread_ratio"]) for row in chunk) / len(chunk),
            "scan_cell_count_mean": sum(float(row["footprint"]["scan_cell_count"]) for row in chunk) / len(chunk),
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
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    frames = read_jsonl(sample_dir / "footprint_frame_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("logic_atom_proposal_commit") is not True:
        failures.append("sample schema missing E41 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not frames or not history:
        failures.append("sample row/frame/history artifact empty")
    for idx, row in enumerate(rows[:40]):
        validate_row(row, failures, f"sample row {idx}")
    for idx, frame in enumerate(frames[:40]):
        for key in ["read_cells", "write_cells", "delta_cells", "read_heatmap", "write_heatmap", "delta_heatmap", "visible_protocol"]:
            if key not in frame:
                failures.append(f"sample frame {idx}: missing {key}")
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

    runner = Path(__file__).resolve().with_name("run_e41_logic_atom_genome_grow_shrink_and_commit_probe.py")
    failures.extend(static_runner_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    mutation = read_json(out / "mutation_report.json")
    proposal_report = read_json(out / "proposal_commit_report.json")
    genome_report = read_json(out / "logic_atom_genome_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    frames = read_jsonl(out / "footprint_frames.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("logic_atom_proposal_commit") is not True:
        failures.append("E41 proposal/commit marker missing")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(manifest.get("systems", [])) != SYSTEMS or set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not frames or not progress or not heartbeat:
        failures.append("empty row/frame/progress/heartbeat artifact")
    if proposal_report.get("oracle_reference_ineligible") is not True or genome_report.get("grow_shrink_enabled") is not True:
        failures.append("proposal/genome reports missing required flags")

    if {row.get("system") for row in rows} != SYSTEMS:
        failures.append("row-level system mismatch")
    for idx, row in enumerate(rows[:500]):
        validate_row(row, failures, f"row {idx}")
    for idx, frame in enumerate(frames[:160]):
        for key in ["read_cells", "write_cells", "delta_cells", "read_heatmap", "write_heatmap", "delta_heatmap", "target_patch_cells", "visible_protocol"]:
            if key not in frame:
                failures.append(f"frame {idx}: missing {key}")

    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)

    stat = mutation.get("grow_shrink_logic_atom_genome", {})
    if int(stat.get("accepted_mutations", 0)) + int(stat.get("rejected_mutations", 0)) <= 0:
        failures.append("grow_shrink_logic_atom_genome: no mutation attempts")
    if int(stat.get("rollback_count", 0)) != int(stat.get("rejected_mutations", -1)):
        failures.append("grow_shrink_logic_atom_genome: rollback mismatch")
    if "parameter_diff" not in stat or "parameter_hash" not in stat:
        failures.append("grow_shrink_logic_atom_genome: missing parameter diff/hash")

    grow = system_results["grow_shrink_logic_atom_genome"]["overall"]
    direct = system_results["direct_write_logic_atom_baseline"]["overall"]
    no_arbiter = system_results["proposal_without_arbiter"]["overall"]
    fixed = system_results["fixed_slot_proposal_arbiter"]["overall"]
    random_control = system_results["random_genome_control"]["overall"]
    full = system_results["full_flow_painter_control"]["overall"]
    if aggregate.get("decision") == "e41_logic_atom_grow_shrink_commit_positive":
        if grow["exact_rate"] < 0.95 or grow["action_accuracy"] < 0.95:
            failures.append("positive decision but grow/shrink exact or action below threshold")
        if grow["false_commit_rate"] > 0.03 or grow["missed_commit_rate"] > 0.03:
            failures.append("positive decision but grow/shrink commit errors too high")
        if direct["exact_rate"] >= 0.85 or no_arbiter["exact_rate"] >= 0.85:
            failures.append("positive decision but direct/no-arbiter control too strong")
        if fixed["exact_rate"] < 0.95:
            failures.append("positive decision but fixed slot arbiter reference failed")
        if random_control["action_accuracy"] >= 0.45:
            failures.append("positive decision but random control action accuracy too strong")
        if full["write_spread_ratio"] < 0.90:
            failures.append("positive decision but full-flow diagnostic not diffuse")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])
    result = {"passed": not failures, "failure_count": len(failures), "failures": failures, "decision": aggregate.get("decision"), "run_id": aggregate.get("run_id"), "sample_result": sample_result}
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
