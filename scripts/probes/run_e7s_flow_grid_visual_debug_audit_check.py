#!/usr/bin/env python3
"""Checker for E7S FlowGrid visual debug audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7s_flow_grid_visual_debug_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "flow_grid_frames.json",
    "flow_grid_frames.jsonl",
    "flow_grid_visualizer.html",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "deterministic_replay.json",
    "progress.jsonl",
)
SYSTEMS = (
    "current_untyped_flow_baseline",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "learned_mask_contract",
    "oracle_mask_reference",
)
VALID_DECISIONS = (
    "e7s_flow_grid_visual_debug_ready",
    "e7s_flow_grid_visual_debug_sample_only",
    "e7s_flow_grid_detected_io_corruption_pattern",
    "e7s_flow_grid_visual_debug_blocked",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_names(path: Path) -> set[str]:
    events = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.add(json.loads(line)["event"])
    return events


class ParentMap(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef):
            return cur.name
        cur = parents.get(cur)
    return None


def ast_policy_failures() -> list[str]:
    failures: list[str] = []
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    mapper = ParentMap()
    mapper.visit(tree)
    banned_function_names = {
        "train_masked_context_pocket",
        "train_context_pocket",
        "train_skill_pocket",
        "train_monolithic",
        "backward",
        "step",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned_function_names:
            fn = enclosing_function(node, mapper.parents)
            failures.append(f"TRAINING_OR_BACKPROP_CALL:{fn}:{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr in {"Adam", "AdamW", "SGD"}:
            fn = enclosing_function(node, mapper.parents)
            failures.append(f"OPTIMIZER_CALL:{fn}:{node.attr}")
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e7s_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    frames = load_json(out / "flow_grid_frames.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    html = (out / "flow_grid_visualizer.html").read_text(encoding="utf-8")
    report = (out / "report.md").read_text(encoding="utf-8").lower()

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append(f"INVALID_DECISION:{decision.get('decision')}")
    if decision.get("decision") == "e7s_flow_grid_visual_debug_blocked":
        failures.append("BLOCKED_DECISION")
    if manifest.get("training_performed") is not False or manifest.get("model_changes") is not False:
        failures.append("TRAINING_OR_MODEL_CHANGE_REPORTED")
    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        failures.append("SEMANTIC_LABELS_AS_MODEL_INPUT")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if summary.get("deterministic_replay_passed") is not True:
        failures.append("SUMMARY_REPLAY_FLAG_FALSE")

    if frames.get("flow_dim") != 40:
        failures.append(f"UNEXPECTED_FLOW_DIM:{frames.get('flow_dim')}")
    grid = frames.get("grid_shape", {})
    if grid.get("rows") != 5 or grid.get("cols") != 8:
        failures.append(f"UNEXPECTED_GRID_SHAPE:{grid}")
    if frames.get("source_type") not in {"e7r_artifact_plus_visualization_sample"}:
        failures.append(f"UNEXPECTED_SOURCE_TYPE:{frames.get('source_type')}")
    for system in SYSTEMS:
        if system not in frames.get("systems", []):
            failures.append(f"MISSING_VISUAL_SYSTEM:{system}")
        if system not in aggregate.get("systems", {}):
            failures.append(f"MISSING_AGGREGATE_SYSTEM:{system}")

    frame_rows = frames.get("frames", [])
    if len(frame_rows) < 100:
        failures.append("TOO_FEW_FRAMES")
    examples = frames.get("examples", [])
    if not examples:
        failures.append("MISSING_EXAMPLES")
    for frame in frame_rows[: min(200, len(frame_rows))]:
        for key in ("before", "after", "delta", "read_mask", "write_mask", "preserve_mask", "changed_mask", "illegal_write_mask", "preserve_corruption_mask"):
            if len(frame.get(key, [])) != frames.get("flow_dim"):
                failures.append(f"BAD_FRAME_VECTOR_LENGTH:{frame.get('frame_index')}:{key}")
        if frame.get("phase") in {"after_pocket", "delta_violation"} and "delta_magnitude" not in frame.get("metrics", {}):
            failures.append(f"MISSING_FRAME_METRIC:{frame.get('frame_index')}")

    jsonl_count = sum(1 for line in (out / "flow_grid_frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    if jsonl_count != len(frame_rows):
        failures.append(f"JSONL_FRAME_COUNT_MISMATCH:{jsonl_count}:{len(frame_rows)}")

    if "https://" in html or "http://" in html or "cdn" in html.lower():
        failures.append("HTML_HAS_EXTERNAL_DEPENDENCY")
    for token in ("Play", "read", "write", "preserve", "delta", "Route Timeline"):
        if token not in html:
            failures.append(f"HTML_MISSING_TOKEN:{token}")
    for banned_label in ("truth slot", "memory slot", "confidence slot", "result slot"):
        if banned_label in html.lower() or banned_label in report:
            failures.append(f"SEMANTIC_LANE_LABEL:{banned_label}")
    for banned_claim in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned_claim in report and "does not prove" not in report:
            failures.append(f"BANNED_CLAIM:{banned_claim}")

    progress_events = event_names(out / "progress.jsonl")
    for event in ("run_start", "frames_built", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in progress_events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7s_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        summary["checker_failure_count"] = len(failures)
        summary["checker_failures"] = failures
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
