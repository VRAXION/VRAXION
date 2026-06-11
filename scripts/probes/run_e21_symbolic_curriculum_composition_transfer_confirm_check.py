#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

PRIMARY = "CURRICULUM_WITH_REUSABLE_POCKETS_PRUNED_PRIMARY"
REQ_TARGET = [
    "decision.json", "summary.json", "aggregate_metrics.json", "report.md", "e21_search_report.json",
    "e21_contract_config.json", "e21_locked_hard_pretest_manifest.json", "e21_locked_hard_pretest_episodes.jsonl",
    "e21_locked_hard_posttest_report.json", "e21_curriculum_stage_report.json", "e21_learned_primitive_pocket_report.json",
    "e21_operator_reuse_report.json", "e21_generation_score_report.json", "e21_training_curve_report.json",
    "e21_checkpoint_report.json", "e21_best_policy_report.json", "e21_pruned_policy_report.json",
    "e21_per_episode_eval_report.json", "e21_ablation_report.json", "e21_system_comparison_report.json",
    "e21_trace_validity_report.json", "e21_renderer_faithfulness_report.json", "e21_leakage_audit_report.json",
    "e21_static_metric_audit_report.json", "e21_deterministic_replay_report.json", "e21_failure_map_report.json",
    "e21_next_recommendation.json", "checkpoint_latest.json", "training_progress.jsonl",
]
REQ_SAMPLE = [
    "README.md", "artifact_sample_manifest.json", "aggregate_metrics_sample.json", "sample_schema.json",
    "locked_hard_pretest_sample.jsonl", "locked_hard_posttest_sample.jsonl", "curriculum_stage_sample.jsonl",
    "heldout_transfer_sample.jsonl", "trace_sample.jsonl", "replay_sample.jsonl", "ablation_sample_summary.json",
    "system_comparison_sample_summary.json", "leakage_sample_audit.json", "sample_metric_recompute_report.json",
    "deterministic_replay_sample_report.json", "sample_only_checker_result.json", "boundary_claims_sample_report.json",
]
FORBIDDEN_PRIMARY = {"ORACLE_PARSE_TREE_CONTROL", "ORACLE_CALCULATOR_CONTROL", "PYTHON_EVAL_CONTROL", "SYMPY_CONTROL", "HAND_AUTHORED_SOLVER_CONTROL"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def metric(rows: list[dict], key: str = "canonical_answer_correct") -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def summarize(rows: list[dict]) -> dict:
    primary = [row for row in rows if row.get("system") == PRIMARY]
    undef = [row for row in primary if row.get("family") == "UNDEFINED_CASE"]
    amb = [row for row in primary if row.get("family") == "AMBIGUOUS_UNDERCONSTRAINED"]
    return {
        "episode_count": len(primary),
        "canonical_answer_accuracy": metric(primary),
        "trace_validity": metric(primary, "trace_valid"),
        "renderer_faithfulness": metric(primary, "renderer_faithful"),
        "undefined_handling_accuracy": metric(undef, "undefined_handled") if undef else 1.0,
        "ambiguous_handling_accuracy": metric(amb, "ambiguous_handled") if amb else 1.0,
        "deterministic_replay_match_rate": metric(primary, "deterministic_replay_match"),
    }


def assert_close(name: str, a: float, b: float, failures: list[str]) -> None:
    if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {name}: {a} != {b}")


def validate_sample(sample_dir: Path, expected_run_id: str | None = None) -> dict:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append("missing sample file " + name)
    if failures:
        return {"passed": False, "failures": failures}
    manifest = read_json(sample_dir / "artifact_sample_manifest.json")
    run_id = manifest.get("run_id")
    if expected_run_id and run_id != expected_run_id:
        failures.append("stale run_id")
    pre = read_jsonl(sample_dir / "locked_hard_pretest_sample.jsonl")
    post = read_jsonl(sample_dir / "locked_hard_posttest_sample.jsonl")
    held = read_jsonl(sample_dir / "heldout_transfer_sample.jsonl")
    stages = read_jsonl(sample_dir / "curriculum_stage_sample.jsonl")
    all_rows = pre + post + held + stages
    counts = {
        "committed_sample_episode_count": len(all_rows),
        "locked_hard_pretest_sample_count": len(pre),
        "locked_hard_posttest_sample_count": len(post),
        "heldout_transfer_sample_count": len(held),
        "curriculum_stage_sample_count": len(stages),
        "undefined_ambiguous_sample_count": sum(1 for row in all_rows if row.get("undefined_or_ambiguous")),
        "curriculum_stage_represented_count": len({row.get("curriculum_stage") for row in stages}),
    }
    if counts["committed_sample_episode_count"] < 500:
        failures.append("sample pack below 500 episodes")
    if counts["locked_hard_pretest_sample_count"] < 150:
        failures.append("pretest sample below 150")
    if counts["locked_hard_posttest_sample_count"] < 150:
        failures.append("posttest sample below 150")
    if counts["heldout_transfer_sample_count"] < 100:
        failures.append("heldout transfer sample below 100")
    if counts["curriculum_stage_represented_count"] < 12:
        failures.append("not all curriculum stages represented")
    if counts["undefined_ambiguous_sample_count"] < 50:
        failures.append("undefined/ambiguous samples below 50")
    if any(row.get("run_id") != run_id for row in all_rows):
        failures.append("sample row run_id mismatch")
    if any(row.get("primary_input", {}).get("oracle_parse_tree_present") or row.get("primary_input", {}).get("oracle_answer_present") or row.get("primary_input", {}).get("direct_solver_trace_present") for row in all_rows):
        failures.append("oracle leakage in sample primary input")
    if any(row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary") for row in all_rows):
        failures.append("direct solver or oracle leakage in sample rows")
    traces = read_jsonl(sample_dir / "trace_sample.jsonl")
    if any(row.get("tautological") or len(row.get("trace", [])) < 3 or not row.get("used_primitives") for row in traces):
        failures.append("tautological trace sample")
    replay = read_jsonl(sample_dir / "replay_sample.jsonl")
    if not all(row.get("deterministic_replay_match") for row in replay):
        failures.append("deterministic replay sample mismatch")
    recorded = read_json(sample_dir / "aggregate_metrics_sample.json")
    recomputed = {
        "locked_hard_pretest_accuracy": metric(pre),
        "locked_hard_posttest_accuracy": metric(post),
        "heldout_composition_transfer_accuracy": metric(held),
        "sample_episode_count": len(all_rows),
        "undefined_ambiguous_sample_count": counts["undefined_ambiguous_sample_count"],
        "trace_validity": metric(all_rows, "trace_valid"),
        "renderer_faithfulness": metric(all_rows, "renderer_faithful"),
        "deterministic_replay_match_rate": metric(all_rows, "deterministic_replay_match"),
    }
    for key, value in recomputed.items():
        assert_close("sample." + key, value, recorded.get(key), failures)
    leakage = read_json(sample_dir / "leakage_sample_audit.json")
    if leakage.get("oracle_parse_tree_leakage_detected") or leakage.get("oracle_answer_leakage_detected") or leakage.get("direct_eval_usage_detected") or leakage.get("sympy_usage_detected") or leakage.get("hand_solver_primary_detected"):
        failures.append("leakage sample audit failed")
    return {"passed": not failures, "failures": failures, "run_id": run_id, "counts": counts, "sample_metrics": recomputed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        sample_dir = Path(args.sample_only)
        result = validate_sample(sample_dir)
        output = {"decision": "e21_sample_only_replay_passed" if result["passed"] else "e21_symbolic_curriculum_composition_transfer_invalid_or_incomplete", "checker_failure_count": 0 if result["passed"] else len(result["failures"]), "sample_only_checker_passed": result["passed"], **result}
        if args.write_summary:
            (sample_dir / "sample_only_checker_result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append("missing target file " + name)
    if failures:
        print(json.dumps({"checker_failure_count": len(failures), "failures": failures}, indent=2, sort_keys=True))
        return 1
    summary = read_json(out / "summary.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    logs = read_json(out / "e21_per_episode_eval_report.json")["logs"]
    primary = [row for row in logs if row.get("system") == PRIMARY]
    pre = [row for row in primary if row.get("phase") == "pretest"]
    post = [row for row in primary if row.get("phase") == "posttest"]
    held = [row for row in primary if row.get("phase") == "heldout_transfer"]
    stress = [row for row in primary if row.get("phase") == "stress"]
    assert_close("locked_hard_pretest_accuracy", metric(pre), aggregate["locked_hard_pretest_accuracy"], failures)
    assert_close("locked_hard_posttest_accuracy", metric(post), aggregate["locked_hard_posttest_accuracy"], failures)
    assert_close("heldout_composition_transfer_accuracy", metric(held), aggregate["heldout_composition_transfer_accuracy"], failures)
    stress_summary = summarize([row for row in logs if row.get("phase") == "stress"])
    assert_close("stress canonical_answer_accuracy", stress_summary["canonical_answer_accuracy"], aggregate["stress"]["canonical_answer_accuracy"], failures)
    if summary.get("primary_system") in FORBIDDEN_PRIMARY:
        failures.append("oracle or solver control selected as primary")
    if any(row.get("direct_eval_used_by_primary") or row.get("sympy_used_by_primary") or row.get("oracle_leakage_to_primary") for row in primary):
        failures.append("direct eval, sympy, or oracle leakage detected in primary rows")
    pre_manifest = read_json(out / "e21_locked_hard_pretest_manifest.json")
    if not pre_manifest.get("locked_before_curriculum"):
        failures.append("pretest was not locked before curriculum")
    if summary["decision"] == "e21_symbolic_curriculum_composition_transfer_confirmed" and aggregate["locked_hard_pretest_accuracy"] > 0.55:
        failures.append("confirmed despite pretest too easy")
    if aggregate["locked_hard_posttest_accuracy"] - aggregate["locked_hard_pretest_accuracy"] < 0.25:
        failures.append("pre/post improvement too small")
    if aggregate["delta_vs_monolithic_equal_budget"] < 0.10:
        failures.append("monolithic baseline margin too small")
    if aggregate["delta_vs_no_reusable_pocket_transfer_ablation"] < 0.15:
        failures.append("no reusable pocket ablation margin too small")
    ablations = read_json(out / "e21_ablation_report.json")
    systems = read_json(out / "e21_system_comparison_report.json")
    if "NO_REUSABLE_POCKET_TRANSFER_ABLATION" not in ablations:
        failures.append("missing no reusable pocket ablation")
    if "MONOLITHIC_EQUAL_BUDGET_MUTATION_POLICY" not in systems:
        failures.append("missing monolithic equal budget baseline")
    leakage = read_json(out / "e21_leakage_audit_report.json")
    if leakage.get("direct_eval_usage_detected") or leakage.get("sympy_usage_detected") or leakage.get("oracle_parse_tree_leakage_detected") or leakage.get("oracle_answer_leakage_detected") or leakage.get("hand_solver_primary_detected"):
        failures.append("leakage audit failed")
    static_audit = read_json(out / "e21_static_metric_audit_report.json")
    if not static_audit.get("static_metric_audit_passed") or not static_audit.get("aggregate_recomputed_from_episode_logs"):
        failures.append("static metric audit failed")
    replay = read_json(out / "e21_deterministic_replay_report.json")
    if replay.get("deterministic_replay_match_rate", 0.0) < 0.99:
        failures.append("deterministic replay failed")
    sample_result = validate_sample(sample_dir, summary.get("run_id"))
    if not sample_result["passed"]:
        failures += ["sample:" + item for item in sample_result["failures"]]
    actual = summary.get("actual_budget", {})
    budget_ok = actual.get("generations_completed", 0) >= 100 and actual.get("population_size", 0) >= 160 and actual.get("candidate_count_evaluated", 0) >= 16000 and actual.get("heldout_episode_count", 0) >= 2400 and actual.get("stress_episode_count", 0) >= 2400 and actual.get("locked_hard_pretest_episode_count", 0) >= 1000 and actual.get("locked_hard_posttest_episode_count", 0) >= 1000 and actual.get("curriculum_stage_count", 0) >= 10 and actual.get("checkpoint_count", 0) >= 100 and sample_result.get("counts", {}).get("committed_sample_episode_count", 0) >= 500
    if summary["decision"] == "e21_symbolic_curriculum_composition_transfer_confirmed" and not budget_ok:
        failures.append("confirmed below full budget")
    decision = summary["decision"] if not failures else "e21_symbolic_curriculum_composition_transfer_invalid_or_incomplete"
    result = {
        "decision": decision,
        "checker_failure_count": len(failures),
        "failures": failures,
        "target_checker_passed": not failures,
        "sample_only_checker_passed": sample_result["passed"],
        "recomputed_pretest_accuracy": metric(pre),
        "recomputed_posttest_accuracy": metric(post),
        "recomputed_heldout_transfer_accuracy": metric(held),
        "recomputed_stress_summary": stress_summary,
        "sample_result": sample_result,
    }
    if args.write_summary:
        summary["checker_failure_count"] = len(failures)
        summary["target_checker_passed"] = not failures
        summary["sample_only_checker_passed"] = sample_result["passed"]
        if failures:
            summary["decision"] = decision
            summary["positive_gate_passed"] = False
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        dec = read_json(out / "decision.json")
        dec["checker_failure_count"] = len(failures)
        if failures:
            dec["decision"] = decision
            dec["positive_gate_passed"] = False
        (out / "decision.json").write_text(json.dumps(dec, indent=2, sort_keys=True) + "\n")
        (out / "e21_checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
