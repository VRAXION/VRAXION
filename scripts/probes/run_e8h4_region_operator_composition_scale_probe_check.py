#!/usr/bin/env python3
"""Checker for E8H4 region-operator composition scale probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts/probes/run_e8h4_region_operator_composition_scale_probe.py"
REQUIRED_ARTIFACTS = (
    "aggregate_metrics.json",
    "split_metrics.json",
    "depth_scaling_report.json",
    "operator_discovery_report.json",
    "operator_reuse_report.json",
    "mutation_history.jsonl",
    "row_level_samples.jsonl",
    "dense_shortcut_control_report.json",
    "deterministic_replay.json",
    "decision.json",
    "report.md",
)
SYSTEMS = (
    "identity_noop_baseline",
    "direct_overwrite_matrix_baseline",
    "handcoded_oracle_region_operator_reference",
    "random_region_rule_control",
    "mutation_discovered_single_operator",
    "mutation_discovered_composed_3_step",
    "mutation_discovered_composed_6_step",
    "mutation_discovered_composed_12_step",
    "mutation_discovered_composed_24_step",
    "mutation_discovered_plus_trace_check",
    "mutation_discovered_plus_prune",
    "reusable_operator_library_router",
    "dense_transform_danger_control",
    "answer_shortcut_control",
)
LEARNED_SYSTEMS = (
    "mutation_discovered_single_operator",
    "mutation_discovered_composed_3_step",
    "mutation_discovered_composed_6_step",
    "mutation_discovered_composed_12_step",
    "mutation_discovered_composed_24_step",
    "mutation_discovered_plus_trace_check",
    "mutation_discovered_plus_prune",
    "reusable_operator_library_router",
)
VALID_DECISIONS = {
    "e8h4_region_operator_composition_scale_positive",
    "e8h4_region_operator_partial_scale",
    "e8h4_single_operator_only_no_composition",
    "e8h4_trace_drift_accumulation_failure",
    "e8h4_operator_reuse_positive",
    "e8h4_mutation_search_insufficient",
    "e8h4_dense_shortcut_trace_invalid",
    "e8h4_region_operator_not_sufficient",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def add(failures: list[dict[str, str]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def depth_value(depth: dict[str, Any], system: str, length: int, metric: str) -> float:
    for row in depth.get("rows", []):
        if row.get("system") == system and int(row.get("route_length", -1)) == length:
            return float(row.get(metric, 0.0))
    return 0.0


def split_value(splits: dict[str, Any], system: str, split: str, metric: str) -> float:
    for row in splits.get("rows", []):
        if row.get("system") == system and row.get("split") == split:
            return float(row.get(metric, 0.0))
    return 0.0


def expected_decision(agg: dict[str, Any], depth: dict[str, Any], splits: dict[str, Any]) -> str:
    systems = agg["systems"]
    identity = systems["identity_noop_baseline"]["mean"]
    dense = systems["dense_transform_danger_control"]["mean"]
    shortcut = systems["answer_shortcut_control"]["mean"]
    best = max(LEARNED_SYSTEMS, key=lambda s: systems[s]["mean"]["usefulness"])
    best_mean = systems[best]["mean"]
    clean_gain = best_mean["usefulness"] > identity["usefulness"] + 0.03
    trace_ok = best_mean["trace_validity"] >= identity["trace_validity"]
    depth6_ok = depth_value(depth, best, 6, "usefulness") > depth_value(depth, "identity_noop_baseline", 6, "usefulness") + 0.03
    depth12_ok = depth_value(depth, best, 12, "usefulness") > depth_value(depth, "identity_noop_baseline", 12, "usefulness") + 0.03
    depth24_ok = depth_value(depth, best, 24, "usefulness") > depth_value(depth, "identity_noop_baseline", 24, "usefulness") + 0.03
    drift_explosive = depth_value(depth, best, 24, "drift_slope") > depth_value(depth, best, 1, "drift_slope") + 0.12
    robust = all(split_value(splits, best, split, "usefulness") > max(0.40, split_value(splits, "identity_noop_baseline", split, "usefulness") - 0.03) for split in ("ood", "counterfactual", "adversarial"))
    shortcut_invalid = (dense["usefulness"] >= best_mean["usefulness"] or shortcut["usefulness"] >= best_mean["usefulness"]) and min(dense["trace_validity"], shortcut["trace_validity"]) < best_mean["trace_validity"] - 0.05
    if clean_gain and trace_ok and depth6_ok and depth12_ok and depth24_ok and robust and not drift_explosive:
        return "e8h4_region_operator_composition_scale_positive"
    if clean_gain and trace_ok and depth6_ok and depth12_ok and robust:
        return "e8h4_region_operator_partial_scale"
    if shortcut_invalid:
        return "e8h4_dense_shortcut_trace_invalid"
    if depth_value(depth, "mutation_discovered_single_operator", 1, "usefulness") > identity["usefulness"] + 0.03 and not depth6_ok:
        return "e8h4_single_operator_only_no_composition"
    if clean_gain and not trace_ok:
        return "e8h4_trace_drift_accumulation_failure"
    if best == "reusable_operator_library_router" and clean_gain:
        return "e8h4_operator_reuse_positive"
    if not clean_gain:
        return "e8h4_region_operator_not_sufficient"
    return "e8h4_mutation_search_insufficient"


class RunnerAstGate(ast.NodeVisitor):
    def __init__(self) -> None:
        self.failures: list[dict[str, str]] = []

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            text = node.value.lower()
            if "agi" in text or "consciousness" in text or "model-scale" in text or "raw language" in text:
                # Boundary strings are allowed in docs/report text.
                pass
            if "placeholder metric" in text or "fake metric" in text:
                add(self.failures, "PLACEHOLDER_TEXT_IN_RUNNER", node.value)
        self.generic_visit(node)


def run_check(out: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add(failures, "MISSING_ARTIFACT", name)
    if failures:
        return {"schema_version": "e8h4_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    agg = load_json(out / "aggregate_metrics.json")
    splits = load_json(out / "split_metrics.json")
    depth = load_json(out / "depth_scaling_report.json")
    reuse = load_json(out / "operator_reuse_report.json")
    discovery = load_json(out / "operator_discovery_report.json")
    dense = load_json(out / "dense_shortcut_control_report.json")
    decision = load_json(out / "decision.json")
    deterministic = load_json(out / "deterministic_replay.json")
    mutation_rows = load_jsonl(out / "mutation_history.jsonl")
    sample_rows = load_jsonl(out / "row_level_samples.jsonl")

    if decision.get("decision") not in VALID_DECISIONS:
        add(failures, "INVALID_DECISION_LABEL", str(decision.get("decision")))
    if not deterministic.get("internal_replay_passed"):
        add(failures, "DETERMINISTIC_REPLAY_FAILED", "deterministic_replay.json")
    for name, item in deterministic.get("hash_comparisons", {}).items():
        if not item.get("match"):
            add(failures, "REPLAY_HASH_MISMATCH", name)
    if not decision.get("deterministic_replay_passed"):
        add(failures, "DETERMINISTIC_REPLAY_NOT_MARKED_PASS", "decision.json")

    missing_systems = [system for system in SYSTEMS if system not in agg.get("systems", {})]
    if missing_systems:
        add(failures, "MISSING_SYSTEM_METRICS", ",".join(missing_systems))
    for system in SYSTEMS:
        if not any(row.get("system") == system for row in sample_rows):
            add(failures, "MISSING_ROW_LEVEL_SAMPLE_SYSTEM", system)
    if len(sample_rows) < len(SYSTEMS):
        add(failures, "TOO_FEW_ROW_LEVEL_SAMPLES", str(len(sample_rows)))
    if not mutation_rows:
        add(failures, "MISSING_MUTATION_HISTORY_ROWS", "mutation_history.jsonl")
    else:
        accepted = max(int(row.get("accepted", 0)) for row in mutation_rows)
        rejected = max(int(row.get("rejected", 0)) for row in mutation_rows)
        rollback = max(int(row.get("rollback", 0)) for row in mutation_rows)
        if accepted <= 0 or rejected <= 0 or rollback <= 0:
            add(failures, "NO_ACCEPT_REJECT_ROLLBACK", f"{accepted}/{rejected}/{rollback}")

    expected = expected_decision(agg, depth, splits)
    if decision.get("decision") != expected:
        add(failures, "DECISION_METRIC_MISMATCH", f"expected {expected}, got {decision.get('decision')}")

    best = decision.get("detail", {}).get("best_system")
    if best in {"handcoded_oracle_region_operator_reference", "dense_transform_danger_control", "answer_shortcut_control"}:
        add(failures, "INVALID_BEST_SYSTEM", str(best))
    dense_mean = agg["systems"]["dense_transform_danger_control"]["mean"]
    shortcut_mean = agg["systems"]["answer_shortcut_control"]["mean"]
    if decision.get("decision") in {"e8h4_region_operator_composition_scale_positive", "e8h4_region_operator_partial_scale"}:
        best_mean = agg["systems"][best]["mean"]
        if dense_mean["usefulness"] >= best_mean["usefulness"] and dense_mean["trace_validity"] >= best_mean["trace_validity"] - 0.05:
            add(failures, "DENSE_SHORTCUT_COUNTED_AS_POSITIVE", "dense_transform_danger_control")
        if shortcut_mean["usefulness"] >= best_mean["usefulness"] and shortcut_mean["trace_validity"] >= best_mean["trace_validity"] - 0.05:
            add(failures, "ANSWER_SHORTCUT_COUNTED_AS_POSITIVE", "answer_shortcut_control")

    if not discovery.get("rows"):
        add(failures, "EMPTY_OPERATOR_DISCOVERY_REPORT", "operator_discovery_report.json")
    if not reuse.get("rows"):
        add(failures, "EMPTY_OPERATOR_REUSE_REPORT", "operator_reuse_report.json")
    if not dense.get("rows"):
        add(failures, "EMPTY_DENSE_CONTROL_REPORT", "dense_shortcut_control_report.json")

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    gate = RunnerAstGate()
    gate.visit(tree)
    failures.extend(gate.failures)

    return {"schema_version": "e8h4_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e8h4_region_operator_composition_scale_probe")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    result = run_check(out)
    if args.write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["failure_count"] == 0 else 1)


if __name__ == "__main__":
    main()
