#!/usr/bin/env python3
"""Checker for E7O learned numeric pocket router composition probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7o_learned_numeric_pocket_router_composition.py"
CHECKER = "scripts/probes/run_e7o_learned_numeric_pocket_router_composition_check.py"
SYSTEMS = (
    "symbolic_proxy_pocket_router_reference",
    "float_numeric_pocket_library_router",
    "int8_numeric_pocket_library_router",
    "int4_pruned_numeric_pocket_library_router",
    "ternary_binary_numeric_pocket_router",
    "mixed_precision_numeric_pocket_router",
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "dense_graph_danger_control",
    "oracle_router_over_numeric_pockets",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "dense_graph_danger_control")
MUTATION_SYSTEMS = (
    "float_numeric_pocket_library_router",
    "int8_numeric_pocket_library_router",
    "int4_pruned_numeric_pocket_library_router",
    "ternary_binary_numeric_pocket_router",
    "mixed_precision_numeric_pocket_router",
    "monolithic_mutation_model",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
POCKET_VARIANTS = ("float", "int8", "int4_pruned", "ternary", "binary")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "numeric_pocket_training_report.json",
    "pocket_library_report.json",
    "router_training_report.json",
    "composition_report.json",
    "error_attribution_report.json",
    "system_results.json",
    "mutation_history.json",
    "leakage_report.json",
    "deterministic_replay.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
)
REQUIRED_SPLIT_METRICS = (
    "answer_accuracy",
    "route_accuracy",
    "composition_usefulness",
    "mean_route_steps",
    "pocket_error_rate",
    "router_error_rate",
    "composition_error_rate",
    "bit_budget",
    "active_parameter_count",
    "pocket_reuse_count",
    "row_level_samples",
)
REQUIRED_MEAN_METRICS = (
    "eval_mean_answer_accuracy",
    "eval_mean_composition_usefulness",
    "eval_mean_route_accuracy",
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
    "parameter_count",
    "bit_budget",
    "router_complexity",
)
VALID_DECISIONS = {
    "e7o_int4_numeric_pocket_router_composition_positive",
    "e7o_float_only_numeric_pocket_composition",
    "e7o_router_over_numeric_pockets_failure",
    "e7o_numeric_pocket_quality_bottleneck",
    "e7o_monolithic_model_preferred_for_numeric_composition",
    "e7o_numeric_pocket_router_collapses_to_graph_soup",
    "e7o_mixed_precision_numeric_pocket_router_preferred",
}


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def static_scan() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
    if failures:
        return failures
    text = runner.read_text(encoding="utf-8")
    for token in (
        "E7O_LEARNED_NUMERIC_POCKET_ROUTER_COMPOSITION",
        "CALL(pocket_id, Flow[D]) -> Flow[D]",
        "NumericPocketCore",
        "train_skill_pocket",
        "build_libraries",
        "train_mutation_router",
        "router_score",
        "error_attribution_report.json",
        "deterministic_replay",
        "hardware_heartbeat",
        "symbolic_proxy_primary",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    functions = function_nodes(tree)
    allowed_gradient_functions = {"train_skill_pocket", "train_monolithic"}
    required_gradient_functions = {"train_skill_pocket", "train_monolithic"}
    forbidden_gradient_functions = {
        "mutate_router",
        "router_score",
        "train_mutation_router",
        "train_monolithic_mutation",
        "evaluate_router_system",
        "seed_worker",
    }
    gradient_seen: set[str] = set()
    forbidden_calls = {"backward", "step", "zero_grad"}
    forbidden_optimizer_attrs = {"AdamW", "SGD", "RMSprop", "Optimizer"}
    for name, node in functions.items():
        local_uses_gradient = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                call_name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                if call_name in forbidden_calls:
                    local_uses_gradient = True
                    if name not in allowed_gradient_functions:
                        failures.append(f"BACKPROP_CALL_OUTSIDE_ALLOWED_FUNCTION:{name}:{call_name}")
            if isinstance(child, ast.Attribute) and child.attr in forbidden_optimizer_attrs:
                local_uses_gradient = True
                if name not in allowed_gradient_functions:
                    failures.append(f"OPTIMIZER_REFERENCE_OUTSIDE_ALLOWED_FUNCTION:{name}:{child.attr}")
        if local_uses_gradient:
            gradient_seen.add(name)
    for name in required_gradient_functions:
        if name not in gradient_seen:
            failures.append(f"REQUIRED_GRADIENT_PATH_MISSING:{name}")
    for name in forbidden_gradient_functions:
        node = functions.get(name)
        if node is None:
            failures.append(f"REQUIRED_FUNCTION_MISSING:{name}")
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                call_name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                if call_name in forbidden_calls:
                    failures.append(f"MUTATION_OR_ROUTER_BACKPROP_CALL:{name}:{call_name}")
            if isinstance(child, ast.Attribute) and child.attr in forbidden_optimizer_attrs:
                failures.append(f"MUTATION_OR_ROUTER_OPTIMIZER_REFERENCE:{name}:{child.attr}")
    return failures


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "backend_manifest.json")
    task = load_json(out / "task_generation_report.json")
    pocket_training = load_json(out / "numeric_pocket_training_report.json")
    library = load_json(out / "pocket_library_report.json")
    router = load_json(out / "router_training_report.json")
    composition = load_json(out / "composition_report.json")
    attribution = load_json(out / "error_attribution_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7O_LEARNED_NUMERIC_POCKET_ROUTER_COMPOSITION":
        failures.append("MANIFEST_MILESTONE_BAD")
    if manifest.get("source_milestone") != "E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE":
        failures.append("SOURCE_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if tuple(manifest.get("gradient_systems", [])) != GRADIENT_SYSTEMS:
        failures.append("MANIFEST_GRADIENT_SYSTEMS_MISMATCH")
    if tuple(manifest.get("mutation_systems", [])) != MUTATION_SYSTEMS:
        failures.append("MANIFEST_MUTATION_SYSTEMS_MISMATCH")
    if manifest.get("real_numeric_pocket_cores_used") is not True:
        failures.append("REAL_NUMERIC_POCKET_FLAG_BAD")
    if manifest.get("symbolic_proxy_primary") is not False:
        failures.append("SYMBOLIC_PROXY_PRIMARY_BAD")
    if manifest.get("mutation_router_backprop_allowed") is not False:
        failures.append("MUTATION_ROUTER_BACKPROP_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")

    if task.get("schema_version") != "e7o_task_generation_report_v1":
        failures.append("TASK_SCHEMA_BAD")
    if task.get("flow_dim") != 40:
        failures.append("TASK_FLOW_DIM_BAD")
    if tuple(task.get("skills", [])) != SKILLS:
        failures.append("TASK_SKILLS_MISMATCH")
    if not task.get("route_options"):
        failures.append("TASK_ROUTE_OPTIONS_MISSING")
    for seed in seeds:
        composition_counts = task.get("composition_row_counts", {}).get(str(seed), {})
        for split in SPLITS:
            if int(composition_counts.get(split, 0)) <= 0:
                failures.append(f"COMPOSITION_SPLIT_EMPTY:{seed}:{split}")
        pocket_counts = task.get("pocket_row_counts", {}).get(str(seed), {})
        for skill in SKILLS:
            for split in SPLITS:
                if int(pocket_counts.get(skill, {}).get(split, 0)) <= 0:
                    failures.append(f"POCKET_SPLIT_EMPTY:{seed}:{skill}:{split}")

    for name, artifact in (
        ("pocket_training", pocket_training),
        ("library", library),
        ("router", router),
        ("composition", composition),
        ("attribution", attribution),
        ("systems", systems),
        ("mutation", mutation),
        ("leakage", leakage),
        ("aggregate", aggregate),
    ):
        if not str(artifact.get("schema_version", "")).startswith("e7o_"):
            failures.append(f"SCHEMA_BAD:{name}")

    result_rows = systems.get("rows", [])
    expected_result_rows = len(seeds) * len(SYSTEMS)
    if len(result_rows) != expected_result_rows:
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_BAD:expected={expected_result_rows}:actual={len(result_rows)}")
    expected_pairs = {(system, seed) for system in SYSTEMS for seed in seeds}
    if {(row.get("system"), int(row.get("seed"))) for row in result_rows} != expected_pairs:
        failures.append("SYSTEM_RESULT_SET_BAD")
    aggregate_systems = aggregate.get("systems", {})
    if set(aggregate_systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_BAD")
    if aggregate.get("best_non_reference_system") not in set(SYSTEMS) - {"symbolic_proxy_pocket_router_reference", "oracle_router_over_numeric_pockets"}:
        failures.append("AGGREGATE_BEST_SYSTEM_BAD")

    for system in SYSTEMS:
        system_rows = [row for row in result_rows if row.get("system") == system]
        if aggregate_systems.get(system, {}).get("seed_count") != len(seeds):
            failures.append(f"AGGREGATE_SEED_COUNT_BAD:{system}")
        mean = aggregate_systems.get(system, {}).get("mean", {})
        for metric in REQUIRED_MEAN_METRICS:
            if metric not in mean:
                failures.append(f"AGGREGATE_MEAN_MISSING:{system}:{metric}")
        for row in system_rows:
            evals = row.get("evals", {})
            for split in SPLITS:
                split_eval = evals.get(split, {})
                for metric in REQUIRED_SPLIT_METRICS:
                    if metric not in split_eval:
                        failures.append(f"SPLIT_METRIC_MISSING:{system}:{row.get('seed')}:{split}:{metric}")
                if not split_eval.get("row_level_samples"):
                    failures.append(f"ROW_SAMPLES_MISSING:{system}:{row.get('seed')}:{split}")
            if system in MUTATION_SYSTEMS:
                attempts = int(row.get("mutation_attempts", 0))
                accepted = int(row.get("accepted_mutations", 0))
                rejected = int(row.get("rejected_mutations", 0))
                rollback = int(row.get("rollback_count", -1))
                if attempts <= 0 or "accepted_mutations" not in row or "rejected_mutations" not in row:
                    failures.append(f"MUTATION_COUNTS_BAD:{system}:{row.get('seed')}")
                if accepted + rejected != attempts:
                    failures.append(f"MUTATION_ACCEPT_REJECT_SUM_BAD:{system}:{row.get('seed')}")
                if rollback != rejected:
                    failures.append(f"ROLLBACK_MISMATCH:{system}:{row.get('seed')}")
                if row.get("initial_candidate_hash") == row.get("final_candidate_hash"):
                    failures.append(f"PARAMETER_DIFF_MISSING:{system}:{row.get('seed')}")
                if not row.get("parameter_diff_hash"):
                    failures.append(f"PARAMETER_DIFF_HASH_MISSING:{system}:{row.get('seed')}")
                if not row.get("history"):
                    failures.append(f"MUTATION_HISTORY_EMPTY:{system}:{row.get('seed')}")

    expected_pocket_rows = len(seeds) * len(SKILLS) * len(POCKET_VARIANTS)
    pocket_rows = pocket_training.get("rows", [])
    if len(pocket_rows) != expected_pocket_rows:
        failures.append(f"POCKET_TRAINING_ROW_COUNT_BAD:expected={expected_pocket_rows}:actual={len(pocket_rows)}")
    pocket_set = {(int(row.get("seed")), row.get("skill"), row.get("variant")) for row in pocket_rows}
    expected_pocket_set = {(seed, skill, variant) for seed in seeds for skill in SKILLS for variant in POCKET_VARIANTS}
    if pocket_set != expected_pocket_set:
        failures.append("POCKET_TRAINING_SET_BAD")
    summary_obj = pocket_training.get("summary", {})
    library_summary = library.get("summary", {})
    if summary_obj != library_summary:
        failures.append("POCKET_LIBRARY_SUMMARY_MISMATCH")
    for skill in SKILLS:
        if skill not in summary_obj:
            failures.append(f"POCKET_SUMMARY_SKILL_MISSING:{skill}")
            continue
        for variant in POCKET_VARIANTS:
            variant_summary = summary_obj[skill].get(variant, {})
            if "mean_accuracy" not in variant_summary or "mean_bit_budget" not in variant_summary or "pass_gate" not in variant_summary:
                failures.append(f"POCKET_SUMMARY_VARIANT_BAD:{skill}:{variant}")
        for variant in POCKET_VARIANTS:
            rows = [row for row in pocket_rows if row.get("skill") == skill and row.get("variant") == variant]
            for row in rows:
                standalone = row.get("standalone", {})
                if not standalone.get("state_hash"):
                    failures.append(f"POCKET_STATE_HASH_MISSING:{row.get('seed')}:{skill}:{variant}")
                for metric in ("eval_mean_accuracy", "heldout_accuracy", "ood_accuracy", "counterfactual_accuracy", "adversarial_accuracy", "bit_budget", "active_parameter_count"):
                    if metric not in standalone:
                        failures.append(f"POCKET_STANDALONE_METRIC_MISSING:{row.get('seed')}:{skill}:{variant}:{metric}")

    mutation_rows = mutation.get("rows", [])
    expected_mutation_rows = len(seeds) * len(MUTATION_SYSTEMS)
    if len(mutation_rows) != expected_mutation_rows:
        failures.append(f"MUTATION_ROW_COUNT_BAD:expected={expected_mutation_rows}:actual={len(mutation_rows)}")
    if len(router.get("rows", [])) != expected_mutation_rows:
        failures.append("ROUTER_TRAINING_ROW_COUNT_BAD")
    if len(composition.get("rows", [])) != expected_result_rows:
        failures.append("COMPOSITION_ROW_COUNT_BAD")
    if len(attribution.get("rows", [])) != expected_result_rows:
        failures.append("ATTRIBUTION_ROW_COUNT_BAD")

    if leakage.get("target_answer_in_input") is not False:
        failures.append("LEAK_TARGET_ANSWER_IN_INPUT")
    if leakage.get("expected_route_index_used_as_input") is not False:
        failures.append("LEAK_ROUTE_INDEX_USED")
    if leakage.get("symbolic_proxy_primary") is not False:
        failures.append("LEAK_SYMBOLIC_PROXY_PRIMARY")
    if leakage.get("mutation_router_uses_optimizer_or_backprop") is not False:
        failures.append("LEAK_MUTATION_ROUTER_BACKPROP")

    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, item in replay.get("hash_comparisons", {}).items():
        if item.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_LABEL_BAD")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("best_non_reference_system") != aggregate.get("best_non_reference_system"):
        failures.append("SUMMARY_BEST_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    progress = jsonl_rows(out / "progress.jsonl")
    events = {row.get("event") for row in progress}
    for event in (
        "startup",
        "tasks_generated",
        "seed_jobs_submitted",
        "pocket_gradient_epoch",
        "skill_pocket_registered",
        "router_mutation_generation",
        "monolithic_gradient_epoch",
        "monolithic_mutation_generation",
        "seed_job_complete",
        "primary_artifacts_written",
        "deterministic_replay_start",
        "deterministic_replay_complete",
        "final_artifacts_written",
    ):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if not jsonl_rows(out / "hardware_heartbeat.jsonl"):
        failures.append("HARDWARE_HEARTBEAT_EMPTY")
    partial = load_json(out / "partial_aggregate_snapshot.json")
    if "completed_rows" not in partial or "completed_pocket_rows" not in partial:
        failures.append("PARTIAL_AGGREGATE_BAD")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("agi", "consciousness", "model-scale claim", "raw-language proof", "raw-language, deployed-model"):
        if banned in report_text and "does not" not in report_text:
            failures.append(f"BROAD_CLAIM_IN_REPORT:{banned}")
    return failures


def write_summary(out: Path, failures: list[str]) -> None:
    summary_path = out / "summary.json"
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    summary["checker_failure_count"] = len(failures)
    summary["checker_failures"] = failures
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7o_learned_numeric_pocket_router_composition")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    if out.exists():
        failures.extend(check_artifacts(out))
    else:
        failures.append(f"OUT_MISSING:{out}")
    if args.write_summary and out.exists():
        write_summary(out, failures)
    result = {
        "schema_version": "e7o_checker_result_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
