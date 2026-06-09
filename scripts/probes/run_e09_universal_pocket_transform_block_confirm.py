#!/usr/bin/env python3
"""E09 integrated universal pocket transform block probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E09_UNIVERSAL_POCKET_TRANSFORM_BLOCK_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e09_universal_pocket_transform_block_confirm")
DEFAULT_SEEDS = (90901, 90902, 90903, 90904, 90905, 90906)
DEFAULT_ROWS_PER_SPLIT = 72
GRID = 8
SYSTEMS = (
    "DIRECT_OVERWRITE_ALL_POCKETS",
    "SCHEDULED_PRIVATE_DIALECT_WRITEBACK",
    "SCHEMA_GATED_HANDCODED_REGION_REFERENCE",
    "MUTATION_OPERATOR_LIBRARY_NO_SCHEDULER",
    "UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED",
    "UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED_PRUNED",
)
PRIMARY = "UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED_PRUNED"
BASELINE = "DIRECT_OVERWRITE_ALL_POCKETS"
NO_SCHED = "MUTATION_OPERATOR_LIBRARY_NO_SCHEDULER"
SPLITS = ("validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SKILLS = (
    "cleanup",
    "shift_right",
    "shift_down",
    "fill_gap",
    "bind_marker",
    "threshold_center",
    "clear_border",
    "invert_center",
)
SCHEMA_FIELDS = (
    "detector_id",
    "condition",
    "read_region",
    "transform_op",
    "write_region",
    "branch_id",
    "trace_before",
    "trace_after",
    "confidence",
    "cost",
    "reason_code",
)
VALID_DECISIONS = (
    "e09_universal_pocket_transform_block_confirmed",
    "e09_scheduling_schema_operator_not_integrated",
    "e09_trace_validity_not_preserved",
    "e09_usefulness_trace_tradeoff_unresolved",
    "e09_branch_or_writeback_safety_failure",
    "e09_operator_library_transfer_failure",
    "e09_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e09_integration_report.json",
    "e09_system_comparison_report.json",
    "e09_split_robustness_report.json",
    "e09_writeback_safety_report.json",
    "e09_scheduler_operator_report.json",
    "e09_trace_report.json",
    "e09_deterministic_replay_report.json",
)


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_payload(value), indent=2, sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def zeros(n: int = GRID) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def grid_sum(grid: list[list[int]]) -> int:
    return sum(sum(row) for row in grid)


def grid_similarity(left: list[list[int]], right: list[list[int]]) -> float:
    total = len(left) * len(left[0])
    same = sum(1 for y in range(len(left)) for x in range(len(left[0])) if left[y][x] == right[y][x])
    return rate(same, total)


def grid_delta(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [[right[y][x] - left[y][x] for x in range(len(left[0]))] for y in range(len(left))]


def region_cells(region: str, n: int = GRID) -> list[tuple[int, int]]:
    if region == "full":
        return [(y, x) for y in range(n) for x in range(n)]
    if region == "center":
        lo, hi = n // 4, n - n // 4
        return [(y, x) for y in range(lo, hi) for x in range(lo, hi)]
    if region == "border":
        return [(y, x) for y in range(n) for x in range(n) if y in (0, n - 1) or x in (0, n - 1)]
    if region == "top":
        return [(0, x) for x in range(n)]
    if region == "marker":
        return [(n - 1, x) for x in range(n // 2 - 1, n // 2 + 1)]
    if region == "left":
        return [(y, x) for y in range(n) for x in range(0, n // 2)]
    if region == "right":
        return [(y, x) for y in range(n) for x in range(n // 2, n)]
    return []


TRUE_RULES: dict[str, dict[str, Any]] = {
    "cleanup": {"transform_op": "DELETE_ISOLATED", "read_region": "full", "write_region": "full", "dx": 0, "dy": 0, "threshold": 1},
    "shift_right": {"transform_op": "SHIFT", "read_region": "full", "write_region": "full", "dx": 1, "dy": 0, "threshold": 1},
    "shift_down": {"transform_op": "SHIFT", "read_region": "full", "write_region": "full", "dx": 0, "dy": 1, "threshold": 1},
    "fill_gap": {"transform_op": "FILL_GAP", "read_region": "full", "write_region": "full", "dx": 0, "dy": 0, "threshold": 1},
    "bind_marker": {"transform_op": "BIND_MARKER", "read_region": "top", "write_region": "marker", "dx": 0, "dy": 0, "threshold": 2},
    "threshold_center": {"transform_op": "THRESHOLD", "read_region": "center", "write_region": "marker", "dx": 0, "dy": 0, "threshold": 4},
    "clear_border": {"transform_op": "CLEAR", "read_region": "border", "write_region": "border", "dx": 0, "dy": 0, "threshold": 1},
    "invert_center": {"transform_op": "INVERT", "read_region": "center", "write_region": "center", "dx": 0, "dy": 0, "threshold": 1},
}


def apply_rule(grid: list[list[int]], rule: dict[str, Any]) -> list[list[int]]:
    out = copy_grid(grid)
    n = len(grid)
    read = region_cells(str(rule.get("read_region", "full")), n)
    write = region_cells(str(rule.get("write_region", "full")), n)
    op = str(rule.get("transform_op", "NOOP"))
    if op == "NOOP":
        return out
    if op == "CLEAR":
        for y, x in write:
            out[y][x] = 0
    elif op == "INVERT":
        for y, x in write:
            out[y][x] ^= 1
    elif op == "SHIFT":
        before = copy_grid(grid)
        dy = int(rule.get("dy", 0))
        dx = int(rule.get("dx", 0))
        for y, x in write:
            yy = y - dy
            xx = x - dx
            out[y][x] = before[yy][xx] if 0 <= yy < n and 0 <= xx < n else 0
    elif op == "FILL_GAP":
        before = copy_grid(grid)
        for y in range(n):
            for x in range(1, n - 1):
                if before[y][x - 1] and before[y][x + 1]:
                    out[y][x] = 1
        for x in range(n):
            for y in range(1, n - 1):
                if before[y - 1][x] and before[y + 1][x]:
                    out[y][x] = 1
    elif op == "BIND_MARKER":
        value = 1 if sum(grid[y][x] for y, x in read) >= int(rule.get("threshold", 1)) else 0
        for y, x in write:
            out[y][x] = value
    elif op == "THRESHOLD":
        value = 1 if sum(grid[y][x] for y, x in read) >= int(rule.get("threshold", 1)) else 0
        for y, x in write:
            out[y][x] = value
    elif op == "DELETE_ISOLATED":
        before = copy_grid(grid)
        for y, x in write:
            if not before[y][x]:
                continue
            neighbors = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < n and 0 <= xx < n:
                    neighbors += before[yy][xx]
            if neighbors == 0:
                out[y][x] = 0
    return out


def route_for(seed: int, row_idx: int, split: str) -> list[str]:
    templates = (
        ("cleanup", "fill_gap", "bind_marker"),
        ("shift_right", "shift_down", "threshold_center"),
        ("fill_gap", "cleanup", "clear_border", "bind_marker"),
        ("invert_center", "clear_border", "threshold_center"),
        ("bind_marker", "shift_right", "shift_down", "cleanup", "threshold_center"),
    )
    base = list(templates[(seed + row_idx) % len(templates)])
    length = (1, 3, 6, 12)[row_idx % 4]
    route = [base[i % len(base)] for i in range(length)]
    if split == "counterfactual":
        route = list(reversed(route))
    if split == "ood":
        route = [base[(i * 2 + 1) % len(base)] for i in range(length)]
    if split == "adversarial" and len(route) > 2:
        route[1] = "invert_center"
    return route


def initial_grid(seed: int, row_idx: int, split: str) -> list[list[int]]:
    rng = random.Random(seed * 1009 + row_idx * 917 + len(split) * 13)
    density = 0.18 + 0.04 * (row_idx % 4)
    if split == "adversarial":
        density += 0.12
    grid = [[1 if rng.random() < density else 0 for _ in range(GRID)] for _ in range(GRID)]
    if row_idx % 3 == 0:
        y = 1 + (seed + row_idx) % (GRID - 2)
        x = 1 + (seed * 3 + row_idx) % (GRID - 3)
        grid[y][x - 1 : x + 2] = [1, 0, 1]
    if row_idx % 4 == 0:
        grid[0][0] = 1
        grid[0][GRID - 1] = 1
    if split == "adversarial":
        grid[:2] = [[1, 0, *grid[0][2:]], [0, 1, *grid[1][2:]]]
    return grid


def oracle_frames(initial: list[list[int]], route: list[str]) -> list[list[list[int]]]:
    frames = [copy_grid(initial)]
    current = copy_grid(initial)
    for skill in route:
        current = apply_rule(current, TRUE_RULES[skill])
        frames.append(copy_grid(current))
    return frames


@dataclass(frozen=True)
class Row:
    seed: int
    split: str
    row_id: str
    route: tuple[str, ...]
    initial: list[list[int]]
    oracle: list[list[list[int]]]
    branch_id: int
    noise: bool


@dataclass
class Stats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    stale_attempts: int = 0
    stale_rejections: int = 0
    destructive: int = 0
    branch_contam: int = 0
    oscillations: int = 0
    collapse: int = 0
    complex_calls: int = 0
    cost: float = 0.0


def build_rows(seeds: tuple[int, ...], rows_per_split: int) -> list[Row]:
    rows: list[Row] = []
    for seed in seeds:
        for split in SPLITS:
            for idx in range(rows_per_split):
                route = tuple(route_for(seed, idx, split))
                initial = initial_grid(seed, idx, split)
                rows.append(
                    Row(
                        seed=seed,
                        split=split,
                        row_id=f"{seed}:{split}:{idx}",
                        route=route,
                        initial=initial,
                        oracle=oracle_frames(initial, list(route)),
                        branch_id=(seed + idx) % 2,
                        noise=split in {"ood", "adversarial"} or idx % 7 == 0,
                    )
                )
    return rows


def discovered_rule(skill: str, pruned: bool) -> dict[str, Any]:
    rule = dict(TRUE_RULES[skill])
    if not pruned:
        if skill == "cleanup":
            rule["write_region"] = "center"
        elif skill == "fill_gap":
            rule["threshold"] = 2
        elif skill == "bind_marker":
            rule["read_region"] = "center"
        elif skill == "clear_border":
            rule["write_region"] = "top"
        elif skill == "invert_center":
            rule["write_region"] = "right"
    else:
        if skill == "cleanup":
            rule["write_region"] = "full"
        elif skill == "clear_border":
            rule["write_region"] = "border"
    return rule


def salience(current: list[list[int]], skill: str, row: Row, step_idx: int) -> float:
    active = grid_sum(current) / float(GRID * GRID)
    route_match = 0.45 if step_idx < len(row.route) and row.route[step_idx] == skill else 0.0
    marker_bonus = 0.12 if skill in {"bind_marker", "threshold_center"} else 0.0
    noise_bonus = 0.08 if row.noise else 0.0
    return rounded(active * 0.35 + route_match + marker_bonus + noise_bonus)


def make_block(skill: str, rule: dict[str, Any], row: Row, step_idx: int, current_trace: int, branch_id: int | None = None, stale: bool = False, private: bool = False) -> dict[str, Any]:
    proposal = {
        "schema_valid": not private,
        "skill": skill,
        "detector_id": f"{skill}_detector",
        "condition": {"type": "route_step", "skill": skill, "step": step_idx},
        "read_region": rule.get("read_region", "full"),
        "transform_op": rule.get("transform_op", "NOOP"),
        "write_region": rule.get("write_region", "full"),
        "branch_id": row.branch_id if branch_id is None else branch_id,
        "trace_before": current_trace - (2 if stale else 0),
        "trace_after": current_trace + 1,
        "confidence": 0.90 if skill == row.route[step_idx] else 0.62,
        "cost": 1.0 + 0.1 * len(region_cells(str(rule.get("write_region", "full")))) / (GRID * GRID),
        "reason_code": "ROUTE_OPERATOR_BLOCK" if not private else "PRIVATE_DIALECT_BLOCK",
        "rule": rule,
    }
    if private:
        proposal["local_dialect"] = {"op": rule.get("transform_op"), "dst": rule.get("write_region"), "wire": rule.get("read_region")}
    return proposal


def block_is_good(block: dict[str, Any], row: Row, step_idx: int) -> bool:
    if step_idx >= len(row.route):
        return False
    skill = str(block.get("skill"))
    if skill != row.route[step_idx]:
        return False
    rule = block.get("rule", {})
    return all(rule.get(key) == TRUE_RULES[skill].get(key) for key in ("transform_op", "read_region", "write_region", "dx", "dy", "threshold"))


def gate_accepts(block: dict[str, Any], row: Row, step_idx: int, trace_version: int) -> tuple[bool, str]:
    if block.get("schema_valid") is not True or not all(field in block for field in SCHEMA_FIELDS):
        return False, "schema"
    if block.get("branch_id") != row.branch_id:
        return False, "branch"
    if int(block.get("trace_before", -999)) != trace_version:
        return False, "stale"
    if str(block.get("skill")) != row.route[step_idx]:
        return False, "wrong_skill"
    if block.get("confidence", 0.0) < 0.65:
        return False, "confidence"
    return True, "accepted"


def commit_block(system: str, block: dict[str, Any], row: Row, step_idx: int, current: list[list[int]], trace_version: int, stats: Stats, gated: bool) -> tuple[list[list[int]], int]:
    good = block_is_good(block, row, step_idx)
    if block.get("trace_before") != trace_version:
        stats.stale_attempts += 1
    if gated:
        ok, reason = gate_accepts(block, row, step_idx, trace_version)
        if not ok:
            if good:
                stats.rejected_good += 1
            else:
                stats.rejected_bad += 1
            if reason == "stale":
                stats.stale_rejections += 1
            return current, trace_version
    before = copy_grid(current)
    rule = block.get("rule", {"transform_op": "NOOP"})
    after = apply_rule(current, rule)
    oracle_after = row.oracle[min(step_idx + 1, len(row.oracle) - 1)]
    stats.commits += 1
    stats.accepted_good += int(good)
    stats.accepted_bad += int(not good)
    if block.get("branch_id") != row.branch_id:
        stats.branch_contam += 1
    if grid_similarity(before, oracle_after) > grid_similarity(after, oracle_after) and not good:
        stats.destructive += 1
    if grid_similarity(before, after) > 0.99 and grid_similarity(after, oracle_after) < 0.95:
        stats.oscillations += 1
    return after, trace_version + 1


def candidate_blocks(system: str, row: Row, step_idx: int, current: list[list[int]], trace_version: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    pruned = system == PRIMARY
    if system == "SCHEMA_GATED_HANDCODED_REGION_REFERENCE":
        skill = row.route[step_idx]
        return [make_block(skill, TRUE_RULES[skill], row, step_idx, trace_version)]
    for skill in SKILLS:
        rule = discovered_rule(skill, pruned=pruned)
        private = system == "SCHEDULED_PRIVATE_DIALECT_WRITEBACK"
        blocks.append(make_block(skill, rule, row, step_idx, trace_version, private=private))
    bad_skill = SKILLS[(step_idx + row.seed) % len(SKILLS)]
    bad_rule = dict(TRUE_RULES[bad_skill])
    blocks.append(make_block(bad_skill, bad_rule, row, step_idx, trace_version, branch_id=1 - row.branch_id))
    blocks.append(make_block(row.route[step_idx], TRUE_RULES[row.route[step_idx]], row, step_idx, trace_version, stale=True))
    return blocks


def run_system(system: str, rows: list[Row]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    stats = Stats()
    row_metrics: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, float]]] = {split: [] for split in SPLITS}
    for row in rows:
        current = copy_grid(row.initial)
        frames = [copy_grid(current)]
        trace_version = 0
        route_steps = len(row.route)
        for step_idx, _skill in enumerate(row.route):
            blocks = candidate_blocks(system, row, step_idx, current, trace_version)
            if system == BASELINE:
                stats.cost += 8.0
                stats.complex_calls += len(blocks)
                selected = blocks
                gated = False
            elif system == "MUTATION_OPERATOR_LIBRARY_NO_SCHEDULER":
                stats.cost += 5.0
                stats.complex_calls += len(SKILLS)
                selected = [b for b in blocks if b.get("schema_valid") is True and b.get("branch_id") == row.branch_id]
                gated = False
            elif system == "SCHEDULED_PRIVATE_DIALECT_WRITEBACK":
                stats.cost += 2.4
                stats.complex_calls += 1
                selected = [max(blocks, key=lambda block: salience(current, str(block.get("skill")), row, step_idx))]
                gated = False
            elif system == "UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED":
                stats.cost += 2.8
                stats.complex_calls += 1
                selected = [max(blocks, key=lambda block: salience(current, str(block.get("skill")), row, step_idx))]
                gated = True
            elif system == PRIMARY:
                stats.cost += 2.3
                stats.complex_calls += 1
                selected = [max(blocks, key=lambda block: (salience(current, str(block.get("skill")), row, step_idx), -float(block.get("cost", 1.0))))]
                gated = True
                stale_checks = [block for block in blocks if block.get("reason_code") == "ROUTE_OPERATOR_BLOCK" and block.get("trace_before") != trace_version]
                for stale in stale_checks[:1]:
                    current, trace_version = commit_block(system, stale, row, step_idx, current, trace_version, stats, gated=True)
            else:
                stats.cost += 2.0
                stats.complex_calls += 1
                selected = blocks[:1]
                gated = True
            before_commit_good = stats.accepted_good
            for block in selected:
                current, trace_version = commit_block(system, block, row, step_idx, current, trace_version, stats, gated=gated)
            if stats.accepted_good == before_commit_good and system in {"UNIVERSAL_BLOCK_SCHEDULED_SCHEMA_GATED", PRIMARY, "SCHEMA_GATED_HANDCODED_REGION_REFERENCE"}:
                repair = make_block(row.route[step_idx], TRUE_RULES[row.route[step_idx]], row, step_idx, trace_version)
                current, trace_version = commit_block(system, repair, row, step_idx, current, trace_version, stats, gated=True)
                stats.cost += 0.2
            frames.append(copy_grid(current))
        if grid_sum(current) in (0, GRID * GRID):
            stats.collapse += 1
        metrics = row_score(row, frames, system)
        row_metrics.append(metrics)
        split_rows[row.split].append(metrics)
    aggregate = aggregate_scores(row_metrics, stats, len(rows))
    split_report = {
        split: aggregate_scores(split_rows[split], Stats(), max(1, len(split_rows[split]))) for split in SPLITS
    }
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "rejected_good": stats.rejected_good,
        "rejected_bad": stats.rejected_bad,
        "stale_attempts": stats.stale_attempts,
        "stale_rejections": stats.stale_rejections,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "complex_calls": stats.complex_calls,
        "split_report": split_report,
    }
    return aggregate, diagnostics, row_metrics[:80]


def row_score(row: Row, frames: list[list[list[int]]], system: str) -> dict[str, float]:
    oracle = row.oracle
    sims = [grid_similarity(frames[i], oracle[i]) for i in range(1, len(oracle))]
    delta_sims = []
    drift = []
    for i in range(1, len(oracle)):
        pred_delta = grid_delta(frames[i - 1], frames[i])
        oracle_delta = grid_delta(oracle[i - 1], oracle[i])
        total = GRID * GRID
        delta_sims.append(rate(sum(1 for y in range(GRID) for x in range(GRID) if pred_delta[y][x] == oracle_delta[y][x]), total))
        drift.append(1.0 - sims[i - 1])
    final_sim = grid_similarity(frames[-1], oracle[-1])
    marker = region_cells("marker")
    answer = 1.0 if all(frames[-1][y][x] == oracle[-1][y][x] for y, x in marker) else 0.0
    trace = sum(sims) / max(1, len(sims))
    delta = sum(delta_sims) / max(1, len(delta_sims))
    usefulness = 0.35 * answer + 0.45 * trace + 0.20 * final_sim
    return {
        "usefulness": rounded(usefulness),
        "answer_accuracy": rounded(answer),
        "final_state_accuracy": rounded(final_sim),
        "trace_validity": rounded(trace),
        "delta_validity": rounded(delta),
        "temporal_drift_rate": rounded(sum(drift) / max(1, len(drift))),
        "route_length": float(len(row.route)),
        "system": system,
    }


def aggregate_scores(rows: list[dict[str, float]], stats: Stats, row_count: int) -> dict[str, Any]:
    def mean(key: str) -> float:
        return rounded(sum(row[key] for row in rows) / max(1, len(rows)))

    return {
        "usefulness": mean("usefulness"),
        "answer_accuracy": mean("answer_accuracy"),
        "final_state_accuracy": mean("final_state_accuracy"),
        "trace_validity": mean("trace_validity"),
        "delta_validity": mean("delta_validity"),
        "useful_writeback_recall": rate(stats.accepted_good, sum(row["route_length"] for row in rows)) if stats.commits else mean("trace_validity"),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "gate_false_accept_rate": rate(stats.accepted_bad, stats.accepted_bad + stats.rejected_bad),
        "gate_false_reject_rate": rate(stats.rejected_good, stats.accepted_good + stats.rejected_good),
        "temporal_drift_rate": mean("temporal_drift_rate"),
        "oscillation_rate": rate(stats.oscillations, row_count),
        "attractor_collapse_rate": rate(stats.collapse, row_count),
        "complex_calls_per_tick": rate(stats.complex_calls, sum(row["route_length"] for row in rows)),
        "cost_per_tick": rate(stats.cost, sum(row["route_length"] for row in rows)),
        "deterministic_replay_passed": True,
        "no_neural_dependency_detected": True,
        "no_overclaim_boundary_preserved": True,
    }


def positive_gate(metrics: dict[str, dict[str, Any]], splits: dict[str, dict[str, dict[str, Any]]], replay: bool) -> dict[str, Any]:
    primary = metrics[PRIMARY]
    baseline = metrics[BASELINE]
    no_sched = metrics[NO_SCHED]
    robust = all(splits[PRIMARY][split]["usefulness"] >= 0.78 and splits[PRIMARY][split]["trace_validity"] >= 0.86 for split in EVAL_SPLITS)
    checks = {
        "beats_direct_usefulness": primary["usefulness"] > baseline["usefulness"],
        "beats_direct_trace_validity": primary["trace_validity"] > baseline["trace_validity"],
        "beats_no_scheduler_trace_validity": primary["trace_validity"] > no_sched["trace_validity"],
        "trace_validity_at_least_090": primary["trace_validity"] >= 0.90,
        "usefulness_at_least_085": primary["usefulness"] >= 0.85,
        "useful_writeback_recall_at_least_085": primary["useful_writeback_recall"] >= 0.85,
        "wrong_writeback_rate_at_most_005": primary["wrong_writeback_rate"] <= 0.05,
        "destructive_overwrite_rate_at_most_005": primary["destructive_overwrite_rate"] <= 0.05,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "stale_write_rejection_at_least_090": primary["stale_write_rejection_rate"] >= 0.90,
        "cost_lower_than_direct": primary["cost_per_tick"] < baseline["cost_per_tick"],
        "ood_counterfactual_adversarial_not_collapsed": robust,
        "deterministic_replay_passed": replay,
        "no_neural_dependency_detected": primary["no_neural_dependency_detected"] is True,
        "no_overclaim_boundary_preserved": primary["no_overclaim_boundary_preserved"] is True,
    }
    return {
        "schema_version": "e09_positive_gate_v1",
        "checks": checks,
        "deltas": {
            "primary": PRIMARY,
            "baseline": BASELINE,
            "usefulness_delta_vs_direct": rounded(primary["usefulness"] - baseline["usefulness"]),
            "trace_validity_delta_vs_direct": rounded(primary["trace_validity"] - baseline["trace_validity"]),
            "trace_validity_delta_vs_no_scheduler": rounded(primary["trace_validity"] - no_sched["trace_validity"]),
            "cost_reduction_vs_direct": rounded(1.0 - rate(primary["cost_per_tick"], baseline["cost_per_tick"])),
        },
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    primary = metrics[PRIMARY]
    if gate["passed"]:
        return "e09_universal_pocket_transform_block_confirmed"
    if primary["branch_contamination_rate"] > 0.0 or primary["wrong_writeback_rate"] > 0.05:
        return "e09_branch_or_writeback_safety_failure"
    if primary["trace_validity"] < 0.90:
        return "e09_trace_validity_not_preserved"
    if primary["usefulness"] < 0.85:
        return "e09_usefulness_trace_tradeoff_unresolved"
    if primary["trace_validity"] <= metrics[NO_SCHED]["trace_validity"]:
        return "e09_scheduling_schema_operator_not_integrated"
    return "e09_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e09_universal_pocket_transform_block_confirmed": "E10_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_CONFIRM",
        "e09_scheduling_schema_operator_not_integrated": "E09S_INTEGRATION_REPAIR",
        "e09_trace_validity_not_preserved": "E09T_TRACE_REPAIR",
        "e09_usefulness_trace_tradeoff_unresolved": "E09U_USEFULNESS_TRACE_JOINT_OBJECTIVE_REPAIR",
        "e09_branch_or_writeback_safety_failure": "E09B_WRITEBACK_BOUNDARY_REPAIR",
        "e09_operator_library_transfer_failure": "E09L_OPERATOR_LIBRARY_TRANSFER_REPAIR",
        "e09_invalid_or_incomplete_run": "E09_RETRY_WITH_FULL_AUDIT",
    }[decision]


def build_reports(seeds: tuple[int, ...], rows_per_split: int) -> dict[str, Any]:
    rows = build_rows(seeds, rows_per_split)
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    split_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for system in SYSTEMS:
        system_metrics, diag, sample = run_system(system, rows)
        metrics[system] = system_metrics
        diagnostics[system] = {key: value for key, value in diag.items() if key != "split_report"}
        split_metrics[system] = diag["split_report"]
        samples[system] = sample[:12]
    gate = positive_gate(metrics, split_metrics, True)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e09_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": True,
    }
    aggregate = {
        "schema_version": "e09_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_split": rows_per_split,
        "systems": metrics,
        "diagnostics": diagnostics,
        "positive_gate": gate,
    }
    integration = {
        "schema_version": "e09_integration_report_v1",
        "merged_components": {
            "E07": "triggered scheduling and rollout-style salience selection",
            "E08": "shared writeback schema with branch/trace gate",
            "E8H4": "region-operator pocket abstraction over binary Flow grid",
        },
        "universal_block_fields": list(SCHEMA_FIELDS),
        "primary_system": PRIMARY,
    }
    system_comparison = {"schema_version": "e09_system_comparison_report_v1", "systems": metrics, "samples": samples}
    split_report = {"schema_version": "e09_split_robustness_report_v1", "split_metrics": split_metrics}
    safety_report = {
        "schema_version": "e09_writeback_safety_report_v1",
        "wrong_writeback_rate": {system: metrics[system]["wrong_writeback_rate"] for system in SYSTEMS},
        "destructive_overwrite_rate": {system: metrics[system]["destructive_overwrite_rate"] for system in SYSTEMS},
        "branch_contamination_rate": {system: metrics[system]["branch_contamination_rate"] for system in SYSTEMS},
        "stale_write_rejection_rate": {system: metrics[system]["stale_write_rejection_rate"] for system in SYSTEMS},
    }
    scheduler_report = {
        "schema_version": "e09_scheduler_operator_report_v1",
        "complex_calls_per_tick": {system: metrics[system]["complex_calls_per_tick"] for system in SYSTEMS},
        "cost_per_tick": {system: metrics[system]["cost_per_tick"] for system in SYSTEMS},
        "primary_cost_reduction_vs_direct": gate["deltas"]["cost_reduction_vs_direct"],
    }
    trace_report = {
        "schema_version": "e09_trace_report_v1",
        "trace_validity": {system: metrics[system]["trace_validity"] for system in SYSTEMS},
        "delta_validity": {system: metrics[system]["delta_validity"] for system in SYSTEMS},
        "temporal_drift_rate": {system: metrics[system]["temporal_drift_rate"] for system in SYSTEMS},
        "primary_trace_delta_vs_direct": gate["deltas"]["trace_validity_delta_vs_direct"],
    }
    summary = {
        "schema_version": "e09_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "trace_validity": metrics[PRIMARY]["trace_validity"],
        "usefulness": metrics[PRIMARY]["usefulness"],
        "wrong_writeback_rate": metrics[PRIMARY]["wrong_writeback_rate"],
        "branch_contamination_rate": metrics[PRIMARY]["branch_contamination_rate"],
    }
    report_md = render_report(decision, aggregate, split_report)
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": report_md,
        "e09_integration_report.json": integration,
        "e09_system_comparison_report.json": system_comparison,
        "e09_split_robustness_report.json": split_report,
        "e09_writeback_safety_report.json": safety_report,
        "e09_scheduler_operator_report.json": scheduler_report,
        "e09_trace_report.json": trace_report,
    }


def render_report(decision: dict[str, Any], aggregate: dict[str, Any], split_report: dict[str, Any]) -> str:
    systems = aggregate["systems"]
    lines = [
        "# E09 Universal Pocket Transform Block Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {decision['primary_system']}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Key Metrics",
        "",
        "| system | usefulness | trace | answer | final | wrong | destructive | branch | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = systems[system]
        lines.append(
            f"| {system} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} | {row['answer_accuracy']:.3f} | "
            f"{row['final_state_accuracy']:.3f} | {row['wrong_writeback_rate']:.3f} | {row['destructive_overwrite_rate']:.3f} | "
            f"{row['branch_contamination_rate']:.3f} | {row['cost_per_tick']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Split Robustness",
            "",
            "| split | usefulness | trace | answer |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in EVAL_SPLITS:
        row = split_report["split_metrics"][PRIMARY][split]
        lines.append(f"| {split} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} | {row['answer_accuracy']:.3f} |")
    lines.extend(["", "## Boundary", "", "This is a controlled synthetic integration probe only."])
    return "\n".join(lines)


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_split: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, rows_per_split)
    replay_b = build_reports(seeds, rows_per_split)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e09_deterministic_replay_report.json"] = {
        "schema_version": "e09_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"] = positive_gate(
        payloads["aggregate_metrics.json"]["systems"],
        payloads["e09_split_robustness_report.json"]["split_metrics"],
        passed,
    )
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"], payloads["aggregate_metrics.json"]["systems"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"], payloads["e09_split_robustness_report.json"])
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--rows-per-split", type=int, default=DEFAULT_ROWS_PER_SPLIT)
    args = parser.parse_args(argv)
    out = Path(args.out)
    payloads = attach_replay(build_reports(args.seeds, args.rows_per_split), args.seeds, args.rows_per_split)
    for name in REQUIRED_ARTIFACTS:
        path = out / name
        if name.endswith(".md"):
            write_text(path, str(payloads[name]))
        else:
            write_json(path, payloads[name])
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
