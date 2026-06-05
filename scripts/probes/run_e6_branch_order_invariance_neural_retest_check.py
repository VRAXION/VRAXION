#!/usr/bin/env python3
"""Checker for E6 branch-order invariance neural retest artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e6_branch_order_invariance_neural_retest.py"
CHECKER = "scripts/probes/run_e6_branch_order_invariance_neural_retest_check.py"
SYSTEMS = (
    "e4_top_down_reference",
    "mlp_fixed_order_gradient",
    "mlp_random_order_gradient",
    "recurrent_fixed_order_gradient",
    "recurrent_random_order_gradient",
    "choicewise_shared_random_order_gradient",
    "random_classifier",
)
GRADIENT_SYSTEMS = (
    "mlp_fixed_order_gradient",
    "mlp_random_order_gradient",
    "recurrent_fixed_order_gradient",
    "recurrent_random_order_gradient",
    "choicewise_shared_random_order_gradient",
)
VALID_DECISIONS = {
    "e6_branch_order_invariance_training_succeeds",
    "e6_order_equivariant_neural_architecture_viable",
    "e6_non_neural_router_remains_preferred",
    "e6_leak_or_replay_failure",
    "e6_invariance_retest_inconclusive",
}
BASE_REQUIRED = (
    "e6_backend_manifest.json",
    "e6_task_generation_report.json",
    "e6_invariance_comparison_report.json",
    "e6_branch_order_report.json",
    "e6_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e6_mutation_history_e4_top_down_reference.json",
    "e6_row_level_eval_sample_heldout.json",
    "e6_row_level_eval_sample_ood.json",
    "e6_row_level_eval_sample_counterfactual.json",
    "e6_row_level_eval_sample_adversarial.json",
)
HASH_ARTIFACTS = (
    "e6_invariance_comparison_report.json",
    "e6_branch_order_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_SYSTEM_METRICS = (
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
    "heldout_level_accuracy",
    "heldout_causal_path_accuracy",
    "heldout_stopping_depth_accuracy",
    "heldout_over_detail_rate",
    "heldout_irrelevant_branch_rate",
    "branch_order_heldout_usefulness",
    "branch_order_ood_usefulness",
    "branch_order_counterfactual_usefulness",
    "branch_order_adversarial_usefulness",
    "normal_routing_passed",
    "branch_order_routing_passed",
    "clean_invariant_passed",
    "parameter_count",
)


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_artifacts() -> list[str]:
    names = list(BASE_REQUIRED)
    for system in SYSTEMS:
        names.append(f"e6_candidate_{system}_summary.json")
        names.append(f"e6_parameter_diff_{system}.json")
        if system in GRADIENT_SYSTEMS:
            names.append(f"e6_training_history_{system}.json")
    return names


def progress_rows(out: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compare_replay(primary: Path, replay: Path) -> dict[str, Any]:
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_path = primary / name
        replay_path = replay / name
        primary_hash = file_sha256(primary_path)
        replay_hash = file_sha256(replay_path)
        comparisons[name] = {
            "primary_exists": primary_path.exists(),
            "replay_exists": replay_path.exists(),
            "primary_hash": primary_hash,
            "replay_hash": replay_hash,
            "match": primary_hash is not None and primary_hash == replay_hash,
        }
    return {
        "external_replay_compared": True,
        "external_replay_path": replay.as_posix(),
        "external_replay_passed": all(row["match"] for row in comparisons.values()),
        "external_hash_comparisons": comparisons,
    }


def update_external_replay_report(out: Path, replay: Path) -> dict[str, Any]:
    report_path = out / "e6_deterministic_replay_report.json"
    report = load_json(report_path)
    report.update(compare_replay(out, replay))
    report["deterministic_replay_passed"] = bool(report.get("internal_replay_passed") and report.get("external_replay_passed"))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def ast_scan_runner() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    text = runner.read_text(encoding="utf-8")
    if "id(model)" in text:
        failures.append("NONDETERMINISTIC_MODEL_ID_USED_IN_RUNNER")
    if "randomize_batch_order" not in text:
        failures.append("RANDOM_ORDER_TRAINING_FUNCTION_MISSING")
    if "ProcessPoolExecutor" not in text:
        failures.append("PARALLEL_EXECUTION_SUPPORT_MISSING")
    tree = ast.parse(text)
    torch_import_seen = False
    adamw_seen = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_import_seen = True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "torch":
                torch_import_seen = True
        elif isinstance(node, ast.Attribute) and node.attr == "AdamW":
            adamw_seen = True
    if not torch_import_seen:
        failures.append("TORCH_IMPORT_MISSING")
    if not adamw_seen:
        failures.append("ADAMW_MISSING_FOR_GRADIENT_BASELINE")
    return failures


def check_artifacts(out: Path, replay: Path | None) -> list[str]:
    failures: list[str] = []
    for name in required_artifacts():
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e6_backend_manifest.json")
    task = load_json(out / "e6_task_generation_report.json")
    comparison = load_json(out / "e6_invariance_comparison_report.json")
    branch = load_json(out / "e6_branch_order_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    deterministic = load_json(out / "e6_deterministic_replay_report.json")
    if replay is not None:
        deterministic = update_external_replay_report(out, replay)
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("UNKNOWN_DECISION")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if tuple(manifest.get("gradient_systems", [])) != GRADIENT_SYSTEMS:
        failures.append("MANIFEST_GRADIENT_SYSTEMS_MISMATCH")
    if set(aggregate.get("systems", {})) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEMS_MISMATCH")
    if set(comparison.get("systems", {})) != set(SYSTEMS):
        failures.append("COMPARISON_SYSTEMS_MISMATCH")
    if set(branch.get("branch_order_scores", {})) != set(SYSTEMS):
        failures.append("BRANCH_REPORT_SYSTEMS_MISMATCH")
    if manifest.get("branch_order_randomized_training_used") is not True:
        failures.append("RANDOMIZED_TRAINING_NOT_DECLARED")
    if manifest.get("row_level_predictions_used") is not True:
        failures.append("ROW_LEVEL_FLAG_MISSING")

    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if split not in task.get("splits", {}):
            failures.append(f"SPLIT_MISSING:{split}")
        elif task["splits"][split].get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if replay is not None and deterministic.get("external_replay_passed") is not True:
        failures.append("EXTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")
    if aggregate.get("deterministic_replay_passed") is not True:
        failures.append("AGGREGATE_REPLAY_FLAG_BAD")

    if aggregate.get("label_control_passed") is not True and decision.get("decision") != "e6_leak_or_replay_failure":
        failures.append("LABEL_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
    if aggregate.get("systems", {}).get("random_classifier", {}).get("clean_invariant_passed") is True:
        failures.append("RANDOM_CLASSIFIER_PASSED_TASK_TOO_EASY")

    for system in SYSTEMS:
        metrics = aggregate.get("systems", {}).get(system)
        if not metrics:
            failures.append(f"SYSTEM_METRICS_MISSING:{system}")
            continue
        for key in REQUIRED_SYSTEM_METRICS:
            if key not in metrics:
                failures.append(f"SYSTEM_METRIC_MISSING:{system}:{key}")
        if system != "random_classifier" and metrics.get("parameter_count", 0) < 1:
            failures.append(f"PARAMETER_COUNT_MISSING:{system}")
        diff = load_json(out / f"e6_parameter_diff_{system}.json")
        if system != "random_classifier" and diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_PARAMETER_DIFF:{system}")

    mutation = load_json(out / "e6_mutation_history_e4_top_down_reference.json")
    if mutation.get("mutation_attempt_count", 0) < 1:
        failures.append("NO_TOPDOWN_MUTATION_ATTEMPTS")
    if mutation.get("accepted_mutation_count", 0) < 1:
        failures.append("NO_TOPDOWN_ACCEPTED_MUTATIONS")
    if mutation.get("rejected_mutation_count", 0) < 1:
        failures.append("NO_TOPDOWN_REJECTED_MUTATIONS")
    if mutation.get("rejected_mutation_count") != mutation.get("rollback_count"):
        failures.append("TOPDOWN_ROLLBACK_MISMATCH")

    for system in GRADIENT_SYSTEMS:
        hist = load_json(out / f"e6_training_history_{system}.json")
        if len(hist.get("history", [])) != manifest.get("gradient_epochs"):
            failures.append(f"EPOCH_COUNT_MISMATCH:{system}")
        if hist.get("backprop_used") is not True:
            failures.append(f"BACKPROP_NOT_DECLARED:{system}")
        if system.endswith("random_order_gradient") and hist.get("order_mode") != "randomized":
            failures.append(f"RANDOM_ORDER_MODE_MISSING:{system}")

    rows = progress_rows(out)
    events = {row.get("event") for row in rows}
    for event in ("startup", "final_artifacts_written", "system_start", "system_complete", "epoch_complete", "generation_complete"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if manifest.get("settings", {}).get("execution_mode") == "parallel" and "parallel_systems_start" not in events:
        failures.append("PARALLEL_PROGRESS_MISSING")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e6_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(SYSTEMS):
            failures.append(f"ROW_SAMPLE_SYSTEMS_MISMATCH:{split}")

    failures.extend(ast_scan_runner())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    replay = resolve_out(args.replay_out) if args.replay_out else None
    failures = check_artifacts(out, replay)
    result = {
        "failure_count": len(failures),
        "failures": failures,
        "out": out.as_posix(),
        "replay_out": replay.as_posix() if replay else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
