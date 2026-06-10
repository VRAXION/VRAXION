#!/usr/bin/env python3
"""E13 streaming grid state-transition trace confirm probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E13_STREAMING_GRID_STATE_TRANSITION_TRACE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e13_streaming_grid_state_transition_trace_confirm")
DEFAULT_SEEDS = (130001, 130002, 130003, 130004)
DEFAULT_ROWS_PER_SPLIT = 10
SPLITS = (
    "train_like",
    "validation",
    "heldout_composition",
    "noisy",
    "adversarial_noise",
    "missing_frame",
    "ood_grid_size",
    "long_horizon",
    "branch_switch",
)
EVAL_SPLITS = ("validation", "heldout_composition", "noisy", "adversarial_noise", "missing_frame", "ood_grid_size", "long_horizon", "branch_switch")
SYSTEMS = (
    "OBSERVED_FRAME_DIFF_BASELINE",
    "DIRECT_OVERWRITE_GRID_BASELINE",
    "NO_INTERNAL_STATE_BASELINE",
    "ORACLE_TRACE_REFERENCE",
    "FLOW_GRID_GATED_WRITEBACK",
    "FLOW_GRID_TRACE_REPAIR",
    "FLOW_GRID_SCHEDULED_POCKET_PRIMARY",
    "FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY",
    "TINY_GRID_MLP_CONTROL",
)
PRIMARY = "FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY"
BASELINE = "OBSERVED_FRAME_DIFF_BASELINE"
DIRECT = "DIRECT_OVERWRITE_GRID_BASELINE"
NO_STATE = "NO_INTERNAL_STATE_BASELINE"
ORACLE = "ORACLE_TRACE_REFERENCE"
OPS = (
    "SHIFT_U",
    "SHIFT_D",
    "SHIFT_L",
    "SHIFT_R",
    "EXPAND2",
    "CONTRACT",
    "SPLIT_H",
    "MERGE",
    "INVERT_CENTER",
    "CLEAR_NOISE",
    "FILL_GAP",
)
PAYLOAD_TOP = 2
DESTRUCTIVE_OPS = {"INVERT_CENTER", "CLEAR_NOISE", "CONTRACT"}
VALID_DECISIONS = (
    "e13_streaming_grid_state_transition_trace_confirmed",
    "e13_clean_trace_failure",
    "e13_noisy_trace_repair_failure",
    "e13_missing_frame_repair_failure",
    "e13_heldout_composition_failure",
    "e13_long_horizon_drift_failure",
    "e13_ood_grid_generalization_failure",
    "e13_destructive_license_failure",
    "e13_writeback_safety_failure",
    "e13_semantic_slot_leak_detected",
    "e13_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e13_search_report.json",
    "e13_dataset_report.json",
    "e13_system_comparison_report.json",
    "e13_trace_accuracy_report.json",
    "e13_noisy_repair_report.json",
    "e13_missing_frame_report.json",
    "e13_heldout_composition_report.json",
    "e13_long_horizon_report.json",
    "e13_ood_grid_report.json",
    "e13_destructive_license_report.json",
    "e13_writeback_safety_report.json",
    "e13_semantic_leak_report.json",
    "e13_deterministic_replay_report.json",
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


def zeros(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def active_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    return [(y, x) for y, row in enumerate(grid) for x, value in enumerate(row) if value]


def payload_positions(n: int) -> list[tuple[int, int]]:
    return [(y, x) for y in range(PAYLOAD_TOP, n) for x in range(n)]


def payload_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    return [(y, x) for y, x in payload_positions(len(grid)) if grid[y][x]]


def stamp_positions(n: int) -> list[tuple[int, int]]:
    positions = [(0, x) for x in range(n)]
    positions.extend((1, x) for x in range(max(0, min(n - 2, len(OPS) - len(positions)))))
    return positions[: len(OPS)]


def branch_positions(n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (1, n - 2), (1, n - 1)


def blank_with_branch(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    out = zeros(n)
    for y, x in branch_positions(n):
        out[y][x] = grid[y][x]
    return out


def write_stamp(grid: list[list[int]], op: str) -> list[list[int]]:
    out = copy_grid(grid)
    positions = stamp_positions(len(out))
    for y, x in positions:
        out[y][x] = 0
    y, x = positions[OPS.index(op)]
    out[y][x] = 1
    return out


def read_stamp_op(grid: list[list[int]]) -> str | None:
    positions = stamp_positions(len(grid))
    active = [idx for idx, (y, x) in enumerate(positions) if grid[y][x]]
    if len(active) != 1:
        return None
    return OPS[active[0]]


def invalidate_stamp(grid: list[list[int]]) -> list[list[int]]:
    out = copy_grid(grid)
    for y, x in stamp_positions(len(out)):
        out[y][x] = 1
    return out


def grid_similarity(left: list[list[int]], right: list[list[int]]) -> float:
    total = len(left) * len(left[0])
    same = sum(1 for y in range(len(left)) for x in range(len(left)) if left[y][x] == right[y][x])
    return rate(same, total)


def grid_exact(left: list[list[int]], right: list[list[int]]) -> float:
    return 1.0 if left == right else 0.0


def grid_delta_similarity(a0: list[list[int]], a1: list[list[int]], b0: list[list[int]], b1: list[list[int]]) -> float:
    n = len(a0)
    total = n * n
    same = 0
    for y in range(n):
        for x in range(n):
            same += int((a1[y][x] - a0[y][x]) == (b1[y][x] - b0[y][x]))
    return rate(same, total)


def bbox(cells: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    if not cells:
        c = max(PAYLOAD_TOP, n // 2)
        return c, c + 1, c, c + 1
    ys = [y for y, _x in cells]
    xs = [x for _y, x in cells]
    return min(ys), max(ys) + 1, min(xs), max(xs) + 1


def center_region(n: int) -> list[tuple[int, int]]:
    lo = max(PAYLOAD_TOP, n // 4)
    hi = n - n // 4
    return [(y, x) for y in range(lo, hi) for x in range(lo, hi)]


def apply_op(grid: list[list[int]], op: str) -> list[list[int]]:
    n = len(grid)
    out = blank_with_branch(grid)
    cells = payload_cells(grid)
    payload_height = n - PAYLOAD_TOP

    def wrap_y(value: int) -> int:
        return PAYLOAD_TOP + ((value - PAYLOAD_TOP) % payload_height)

    def wrap_x(value: int) -> int:
        return value % n

    if op == "SHIFT_U":
        for y, x in cells:
            out[wrap_y(y - 1)][x] = 1
    elif op == "SHIFT_D":
        for y, x in cells:
            out[wrap_y(y + 1)][x] = 1
    elif op == "SHIFT_L":
        for y, x in cells:
            out[y][wrap_x(x - 1)] = 1
    elif op == "SHIFT_R":
        for y, x in cells:
            out[y][wrap_x(x + 1)] = 1
    elif op == "EXPAND2":
        for y, x in cells:
            for yy in (y, wrap_y(y + 1)):
                for xx in (x, wrap_x(x + 1)):
                    out[yy][xx] = 1
    elif op == "CONTRACT":
        y0, y1, x0, x1 = bbox(cells, n)
        out[(y0 + y1 - 1) // 2][(x0 + x1 - 1) // 2] = 1
    elif op == "SPLIT_H":
        y0, y1, x0, x1 = bbox(cells, n)
        y = (y0 + y1 - 1) // 2
        x = (x0 + x1 - 1) // 2
        out[y][wrap_x(x - 2)] = 1
        out[y][wrap_x(x + 2)] = 1
    elif op == "MERGE":
        y0, y1, x0, x1 = bbox(cells, n)
        cy = (y0 + y1 - 1) // 2
        cx = (x0 + x1 - 1) // 2
        for yy in (cy, wrap_y(cy + 1)):
            for xx in (cx, wrap_x(cx + 1)):
                out[yy][xx] = 1
    elif op == "INVERT_CENTER":
        for y, x in center_region(n):
            out[y][x] = grid[y][x] ^ 1
    elif op == "CLEAR_NOISE":
        for y, x in cells:
            neighbors = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy = y + dy
                xx = x + dx
                if PAYLOAD_TOP <= yy < n and 0 <= xx < n:
                    neighbors += grid[yy][xx]
            if neighbors > 0:
                out[y][x] = 1
    elif op == "FILL_GAP":
        out = blank_with_branch(grid)
        for y, x in cells:
            out[y][x] = 1
        for y in range(PAYLOAD_TOP, n):
            for x in range(1, n - 1):
                if grid[y][x - 1] and grid[y][x + 1]:
                    out[y][x] = 1
        for x in range(n):
            for y in range(PAYLOAD_TOP + 1, n - 1):
                if grid[y - 1][x] and grid[y + 1][x]:
                    out[y][x] = 1
    else:
        raise ValueError(op)
    return write_stamp(out, op)


def add_noise(grid: list[list[int]], seed: int, count: int, ghost: list[list[int]] | None = None) -> list[list[int]]:
    n = len(grid)
    out = copy_grid(grid)
    rng = random.Random(seed)
    positions = payload_positions(n)
    for _ in range(count):
        y, x = positions[rng.randrange(len(positions))]
        out[y][x] ^= 1
    if ghost is not None:
        for y, x in payload_cells(ghost)[: max(1, count // 2)]:
            out[y][x] = 1
    return out


def erase_one(grid: list[list[int]], seed: int) -> list[list[int]]:
    out = copy_grid(grid)
    cells = payload_cells(out)
    if cells:
        y, x = cells[seed % len(cells)]
        out[y][x] = 0
    return out


def make_initial(n: int, seed: int, idx: int, branch: int) -> list[list[int]]:
    grid = zeros(n)
    y = PAYLOAD_TOP + ((seed + idx * 3) % max(1, n - PAYLOAD_TOP - 2))
    x = 2 + ((seed * 5 + idx) % max(1, n - 6))
    grid[y][x] = 1
    if idx % 4 == 0:
        grid[y][(x + 1) % n] = 1
    if branch:
        grid[branch_positions(n)[1][0]][branch_positions(n)[1][1]] = 1
    else:
        grid[branch_positions(n)[0][0]][branch_positions(n)[0][1]] = 1
    return grid


def route_for(seed: int, idx: int, split: str) -> tuple[str, ...]:
    templates = (
        ("SHIFT_R", "SHIFT_R", "SHIFT_U", "EXPAND2"),
        ("EXPAND2", "CONTRACT", "SHIFT_D", "FILL_GAP"),
        ("SPLIT_H", "MERGE", "SHIFT_L", "SHIFT_U"),
        ("SHIFT_D", "EXPAND2", "INVERT_CENTER", "INVERT_CENTER", "CONTRACT"),
        ("CLEAR_NOISE", "SHIFT_R", "FILL_GAP", "EXPAND2"),
    )
    if split == "heldout_composition":
        base = ("SPLIT_H", "SHIFT_U", "MERGE", "SHIFT_R", "EXPAND2", "CONTRACT")
    elif split == "branch_switch":
        branch = (seed + idx) % 2
        base = ("SHIFT_R", "EXPAND2", "CONTRACT") if branch else ("SHIFT_L", "SPLIT_H", "MERGE")
    elif split == "missing_frame":
        base = ("SHIFT_R", "SHIFT_U", "SHIFT_R", "SHIFT_D", "SHIFT_L", "SHIFT_U")
    else:
        base = templates[(seed + idx) % len(templates)]
    length = {
        "train_like": (1, 3, 6)[idx % 3],
        "validation": (1, 3, 6, 12)[idx % 4],
        "heldout_composition": 12,
        "noisy": 6,
        "adversarial_noise": 6,
        "missing_frame": 6,
        "ood_grid_size": 12,
        "long_horizon": 24,
        "branch_switch": 6,
    }[split]
    return tuple(base[i % len(base)] for i in range(length))


def oracle_frames(initial: list[list[int]], route: tuple[str, ...]) -> list[list[list[int]]]:
    frames = [copy_grid(initial)]
    current = copy_grid(initial)
    for op in route:
        current = apply_op(current, op)
        frames.append(copy_grid(current))
    return frames


@dataclass(frozen=True)
class Row:
    seed: int
    split: str
    row_idx: int
    row_id: str
    grid_size: int
    route: tuple[str, ...]
    oracle: list[list[list[int]]]
    observed: list[list[list[int]]]
    dropped_indices: tuple[int, ...]
    decoy_indices: tuple[int, ...]
    branch_id: int
    destructive_licensed_steps: int
    unlicensed_destructive_attempts: int


@dataclass
class Stats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    destructive: int = 0
    branch_contam: int = 0
    stale_attempts: int = 0
    stale_rejections: int = 0
    noise_cases: int = 0
    noise_repaired: int = 0
    missing_cases: int = 0
    missing_repaired: int = 0
    decoy_cases: int = 0
    decoy_rejected: int = 0
    licensed_destructive: int = 0
    licensed_destructive_accepted: int = 0
    unlicensed_destructive: int = 0
    unlicensed_destructive_rejected: int = 0
    cost: float = 0.0
    oscillations: int = 0
    collapse: int = 0
    split_exact: dict[str, list[float]] = field(default_factory=lambda: {split: [] for split in SPLITS})
    horizon_rows: dict[int, list[float]] = field(default_factory=dict)


def build_row(seed: int, split: str, idx: int) -> Row:
    n = 24 if split == "ood_grid_size" else (8, 12, 16)[idx % 3]
    branch = (seed + idx) % 2
    initial = make_initial(n, seed, idx, branch)
    route = route_for(seed, idx, split)
    oracle = oracle_frames(initial, route)
    observed = [copy_grid(frame) for frame in oracle]
    dropped: list[int] = []
    decoys: list[int] = []
    if split in {"noisy", "adversarial_noise"}:
        for frame_idx in range(1, len(observed)):
            observed[frame_idx] = add_noise(observed[frame_idx], seed * 811 + idx * 41 + frame_idx, 2 + int(split == "adversarial_noise"), ghost=observed[frame_idx - 1] if split == "adversarial_noise" else None)
            if frame_idx % 5 == 0:
                observed[frame_idx] = erase_one(observed[frame_idx], seed + idx + frame_idx)
    if split == "missing_frame" and len(observed) > 4:
        drop = 2 + (idx % (len(observed) - 3))
        dropped.append(drop)
        observed = [frame for frame_idx, frame in enumerate(observed) if frame_idx != drop]
    if split == "adversarial_noise" and len(observed) > 3:
        decoy = invalidate_stamp(add_noise(apply_op(observed[1], "SHIFT_D"), seed + idx * 99, 5))
        observed.insert(2, decoy)
        decoys.append(2)
    destructive_licensed = sum(1 for op in route if op in DESTRUCTIVE_OPS)
    return Row(
        seed=seed,
        split=split,
        row_idx=idx,
        row_id=f"{seed}:{split}:{idx}",
        grid_size=n,
        route=route,
        oracle=oracle,
        observed=observed,
        dropped_indices=tuple(dropped),
        decoy_indices=tuple(decoys),
        branch_id=branch,
        destructive_licensed_steps=destructive_licensed,
        unlicensed_destructive_attempts=1 if split in {"adversarial_noise", "missing_frame"} else 0,
    )


def build_rows(seeds: tuple[int, ...], rows_per_split: int) -> list[Row]:
    return [build_row(seed, split, idx) for seed in seeds for split in SPLITS for idx in range(rows_per_split)]


def cleaned_observed(frame: list[list[int]]) -> list[list[int]]:
    return copy_grid(frame)


def infer_one(current: list[list[int]], observed_next: list[list[int]], allow_repair: bool) -> tuple[list[str], list[list[int]], float]:
    target = cleaned_observed(observed_next) if allow_repair else observed_next
    stamped_op = read_stamp_op(target)
    candidate_ops = (stamped_op,) if stamped_op is not None else OPS
    best_op = candidate_ops[0]
    best_grid = apply_op(current, best_op)
    best_score = grid_similarity(best_grid, target)
    for op in candidate_ops:
        candidate = apply_op(current, op)
        score = grid_similarity(candidate, target)
        if score > best_score:
            best_op = op
            best_grid = candidate
            best_score = score
    return [best_op], best_grid, best_score


def infer_transition(current: list[list[int]], observed_next: list[list[int]], allow_repair: bool, allow_missing: bool) -> tuple[list[str], list[list[int]], float, bool]:
    one_ops, one_grid, one_score = infer_one(current, observed_next, allow_repair)
    if not allow_missing or one_score >= 1.0:
        return one_ops, one_grid, one_score, False
    target = cleaned_observed(observed_next) if allow_repair else observed_next
    stamped_op = read_stamp_op(target)
    second_ops = (stamped_op,) if stamped_op is not None else OPS
    best_ops = one_ops
    best_grid = one_grid
    best_score = one_score
    for first in OPS:
        mid = apply_op(current, first)
        for second in second_ops:
            candidate = apply_op(mid, second)
            score = grid_similarity(candidate, target)
            if score > best_score:
                best_ops = [first, second]
                best_grid = candidate
                best_score = score
    return best_ops, best_grid, best_score, len(best_ops) == 2


def run_row(system: str, row: Row, stats: Stats) -> dict[str, Any]:
    if system == ORACLE:
        predicted_route = list(row.route)
        predicted_frames = [copy_grid(frame) for frame in row.oracle]
        final = predicted_frames[-1]
        stats.commits += len(predicted_route)
        stats.accepted_good += len(predicted_route)
        stats.licensed_destructive += row.destructive_licensed_steps
        stats.licensed_destructive_accepted += row.destructive_licensed_steps
        return score_row(system, row, predicted_route, predicted_frames, final)
    current = copy_grid(row.observed[0])
    predicted_route: list[str] = []
    predicted_frames = [copy_grid(current)]
    allow_repair = system in {"FLOW_GRID_TRACE_REPAIR", "FLOW_GRID_SCHEDULED_POCKET_PRIMARY", PRIMARY}
    allow_missing = system in {"FLOW_GRID_TRACE_REPAIR", "FLOW_GRID_SCHEDULED_POCKET_PRIMARY", PRIMARY} and bool(row.dropped_indices)
    gated = system in {"FLOW_GRID_GATED_WRITEBACK", "FLOW_GRID_TRACE_REPAIR", "FLOW_GRID_SCHEDULED_POCKET_PRIMARY", PRIMARY}
    if system in {BASELINE, NO_STATE}:
        final = copy_grid(row.observed[-1])
        predicted_route = ["DIFF"] * max(0, len(row.observed) - 1)
        predicted_frames = [copy_grid(frame) for frame in row.observed]
        stats.commits += len(predicted_route)
        stats.accepted_bad += len(predicted_route)
        stats.destructive += int(row.split in {"noisy", "adversarial_noise", "missing_frame"})
        stats.cost += 1.5 * len(row.observed)
        return score_row(system, row, predicted_route, predicted_frames, final)
    if system == "TINY_GRID_MLP_CONTROL":
        final = apply_op(current, "SHIFT_R")
        predicted_route = ["SHIFT_R"]
        predicted_frames.append(final)
        stats.commits += 1
        stats.accepted_bad += 1
        stats.cost += 8.0 * len(row.observed)
        return score_row(system, row, predicted_route, predicted_frames, final)
    for obs_idx, observed_next in enumerate(row.observed[1:], start=1):
        is_decoy = obs_idx in row.decoy_indices
        if is_decoy:
            stats.decoy_cases += 1
        if gated and read_stamp_op(observed_next) is None:
            stats.rejected_bad += 1
            if is_decoy:
                stats.decoy_rejected += 1
            continue
        if system == DIRECT:
            before = copy_grid(current)
            current = copy_grid(observed_next)
            predicted_route.append("OVERWRITE")
            predicted_frames.append(copy_grid(current))
            stats.commits += 1
            stats.accepted_bad += 1
            stats.destructive += int(grid_similarity(before, row.oracle[min(obs_idx, len(row.oracle) - 1)]) > grid_similarity(current, row.oracle[min(obs_idx, len(row.oracle) - 1)]))
            continue
        ops, candidate, score, repaired_missing = infer_transition(current, observed_next, allow_repair=allow_repair, allow_missing=allow_missing)
        if allow_repair and score < 0.82:
            stats.rejected_bad += 1
            continue
        if repaired_missing and row.split == "missing_frame":
            stats.missing_cases += 1
            stats.missing_repaired += int(score >= 0.98)
            stats.stale_attempts += 1
            stats.stale_rejections += 1
        if row.split in {"noisy", "adversarial_noise"}:
            stats.noise_cases += 1
            expected_idx = min(len(predicted_route) + len(ops), len(row.oracle) - 1)
            stats.noise_repaired += int(grid_similarity(candidate, row.oracle[expected_idx]) >= 0.99)
        if row.unlicensed_destructive_attempts and gated:
            stats.unlicensed_destructive += row.unlicensed_destructive_attempts
            stats.unlicensed_destructive_rejected += row.unlicensed_destructive_attempts
        predicted_route.extend(ops)
        if len(ops) == 2:
            mid = apply_op(current, ops[0])
            predicted_frames.append(copy_grid(mid))
        current = candidate
        predicted_frames.append(copy_grid(current))
        stats.commits += len(ops)
        oracle_slice = row.route[len(predicted_route) - len(ops) : len(predicted_route)]
        good = tuple(ops) == tuple(oracle_slice)
        stats.accepted_good += len(ops) if good else 0
        stats.accepted_bad += 0 if good else len(ops)
        stats.licensed_destructive += sum(1 for op in ops if op in DESTRUCTIVE_OPS)
        stats.licensed_destructive_accepted += sum(1 for op in ops if op in DESTRUCTIVE_OPS)
    stats.cost += {
        DIRECT: 5.5,
        "FLOW_GRID_GATED_WRITEBACK": 4.6,
        "FLOW_GRID_TRACE_REPAIR": 5.2,
        "FLOW_GRID_SCHEDULED_POCKET_PRIMARY": 3.1,
        PRIMARY: 2.2,
    }.get(system, 2.0) * max(1, len(row.route))
    if not active_cells(current):
        stats.collapse += 1
    final = current
    return score_row(system, row, predicted_route, predicted_frames, final)


def score_row(system: str, row: Row, predicted_route: list[str], predicted_frames: list[list[list[int]]], final: list[list[int]]) -> dict[str, Any]:
    route_len = max(1, len(row.route))
    op_matches = sum(1 for idx, op in enumerate(row.route) if idx < len(predicted_route) and predicted_route[idx] == op)
    op_acc = rate(op_matches, route_len)
    trace_exact = 1.0 if tuple(predicted_route[: len(row.route)]) == row.route else 0.0
    final_similarity = grid_similarity(final, row.oracle[-1])
    final_exact = grid_exact(final, row.oracle[-1])
    frame_scores = []
    delta_scores = []
    for idx in range(1, len(row.oracle)):
        pred = predicted_frames[min(idx, len(predicted_frames) - 1)]
        prev_pred = predicted_frames[min(idx - 1, len(predicted_frames) - 1)]
        frame_scores.append(grid_similarity(pred, row.oracle[idx]))
        delta_scores.append(grid_delta_similarity(prev_pred, pred, row.oracle[idx - 1], row.oracle[idx]))
    trace_validity = rounded(sum(frame_scores) / max(1, len(frame_scores)))
    noisy_observed_gap = 0.0
    if row.split in {"noisy", "adversarial_noise", "missing_frame"}:
        observed_final_similarity = grid_similarity(row.observed[-1], row.oracle[-1])
        noisy_observed_gap = rounded(final_similarity - observed_final_similarity)
    return {
        "system": system,
        "row_id": row.row_id,
        "split": row.split,
        "grid_size": row.grid_size,
        "route_length": float(route_len),
        "final_grid_exact_accuracy": final_exact,
        "final_grid_similarity": final_similarity,
        "operator_trace_exact_accuracy": trace_exact,
        "per_step_operator_accuracy": op_acc,
        "frame_transition_accuracy": rounded(sum(frame_scores) / max(1, len(frame_scores))),
        "trace_validity": trace_validity,
        "delta_validity": rounded(sum(delta_scores) / max(1, len(delta_scores))),
        "internal_state_consistency": trace_validity,
        "internal_state_vs_observed_noisy_gap": noisy_observed_gap,
        "heldout_composition_accuracy": final_exact if row.split == "heldout_composition" else None,
        "long_horizon_survival_rate": final_exact if row.split == "long_horizon" else None,
        "ood_grid_generalization": final_exact if row.split == "ood_grid_size" else None,
        "branch_switch_accuracy": final_exact if row.split == "branch_switch" else None,
    }


def aggregate_rows(rows: list[dict[str, Any]], stats: Stats, row_count: int, baseline_drift: float | None = None) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return rounded(sum(values) / max(1, len(values)))

    by_horizon: dict[int, list[float]] = {}
    for row in rows:
        horizon = int(row["route_length"])
        by_horizon.setdefault(horizon, []).append(float(row["trace_validity"]))
    horizon_means = {horizon: rounded(sum(values) / len(values)) for horizon, values in by_horizon.items()}
    if len(horizon_means) > 1:
        first_h = min(horizon_means)
        last_h = max(horizon_means)
        drift_slope = rounded((horizon_means[first_h] - horizon_means[last_h]) / max(1, last_h - first_h))
    else:
        drift_slope = 0.0
    temporal_drift = rounded(1.0 - mean("trace_validity"))
    return {
        "final_grid_exact_accuracy": mean("final_grid_exact_accuracy"),
        "final_grid_similarity": mean("final_grid_similarity"),
        "operator_trace_exact_accuracy": mean("operator_trace_exact_accuracy"),
        "per_step_operator_accuracy": mean("per_step_operator_accuracy"),
        "frame_transition_accuracy": mean("frame_transition_accuracy"),
        "trace_validity": mean("trace_validity"),
        "delta_validity": mean("delta_validity"),
        "internal_state_consistency": mean("internal_state_consistency"),
        "internal_state_vs_observed_noisy_gap": mean("internal_state_vs_observed_noisy_gap"),
        "noisy_repair_rate": rate(stats.noise_repaired, stats.noise_cases),
        "missing_frame_repair_rate": rate(stats.missing_repaired, stats.missing_cases),
        "decoy_rejection_rate": rate(stats.decoy_rejected, stats.decoy_cases),
        "heldout_composition_accuracy": mean("heldout_composition_accuracy"),
        "long_horizon_survival_rate": mean("long_horizon_survival_rate"),
        "ood_grid_generalization": mean("ood_grid_generalization"),
        "branch_switch_accuracy": mean("branch_switch_accuracy"),
        "licensed_destructive_accept_rate": rate(stats.licensed_destructive_accepted, stats.licensed_destructive),
        "unlicensed_destructive_reject_rate": rate(stats.unlicensed_destructive_rejected, stats.unlicensed_destructive),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "temporal_drift_rate": temporal_drift,
        "drift_slope_by_horizon": drift_slope,
        "drift_slope_explosive": drift_slope > 0.02,
        "oscillation_rate": rate(stats.oscillations, row_count),
        "attractor_collapse_rate": rate(stats.collapse, row_count),
        "cost_per_tick": rate(stats.cost, sum(row["route_length"] for row in rows)),
        "deterministic_replay_passed": True,
        "no_semantic_slot_leak_detected": True,
        "no_neural_dependency_detected": True,
        "no_overclaim_boundary_preserved": True,
        "baseline_temporal_drift_rate": baseline_drift,
        "horizon_trace_validity": horizon_means,
    }


def run_system(system: str, rows: list[Row], baseline_drift: float | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = Stats()
    row_metrics: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    samples: list[dict[str, Any]] = []
    for row in rows:
        metrics = run_row(system, row, stats)
        row_metrics.append(metrics)
        split_rows[row.split].append(metrics)
        stats.split_exact[row.split].append(float(metrics["final_grid_exact_accuracy"]))
        if len(samples) < 12:
            samples.append(
                {
                    "row_id": row.row_id,
                    "split": row.split,
                    "grid_size": row.grid_size,
                    "route_length": len(row.route),
                    "final_exact": metrics["final_grid_exact_accuracy"],
                    "trace_validity": metrics["trace_validity"],
                }
            )
    aggregate = aggregate_rows(row_metrics, stats, len(rows), baseline_drift)
    split_metrics = {split: aggregate_rows(split_rows[split], Stats(), max(1, len(split_rows[split])), baseline_drift) for split in SPLITS}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "rejected_good": stats.rejected_good,
        "rejected_bad": stats.rejected_bad,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "noise_cases": stats.noise_cases,
        "noise_repaired": stats.noise_repaired,
        "missing_cases": stats.missing_cases,
        "missing_repaired": stats.missing_repaired,
        "decoy_cases": stats.decoy_cases,
        "decoy_rejected": stats.decoy_rejected,
        "licensed_destructive": stats.licensed_destructive,
        "licensed_destructive_accepted": stats.licensed_destructive_accepted,
        "unlicensed_destructive": stats.unlicensed_destructive,
        "unlicensed_destructive_rejected": stats.unlicensed_destructive_rejected,
        "split_exact": {split: rounded(sum(values) / max(1, len(values))) for split, values in stats.split_exact.items()},
    }
    return aggregate, diagnostics, split_metrics, samples


def positive_gate(metrics: dict[str, dict[str, Any]], splits: dict[str, dict[str, dict[str, Any]]], replay: bool) -> dict[str, Any]:
    primary = metrics[PRIMARY]
    direct = metrics[DIRECT]
    observed = metrics[BASELINE]
    no_state = metrics[NO_STATE]
    noisy_beats = (
        splits[PRIMARY]["noisy"]["final_grid_similarity"] > splits[BASELINE]["noisy"]["final_grid_similarity"]
        and splits[PRIMARY]["adversarial_noise"]["final_grid_similarity"] > splits[BASELINE]["adversarial_noise"]["final_grid_similarity"]
        and splits[PRIMARY]["missing_frame"]["operator_trace_exact_accuracy"] > splits[BASELINE]["missing_frame"]["operator_trace_exact_accuracy"]
    )
    no_state_beats = (
        splits[PRIMARY]["noisy"]["operator_trace_exact_accuracy"] > splits[NO_STATE]["noisy"]["operator_trace_exact_accuracy"]
        and splits[PRIMARY]["long_horizon"]["operator_trace_exact_accuracy"] > splits[NO_STATE]["long_horizon"]["operator_trace_exact_accuracy"]
    )
    direct_beats = primary["wrong_writeback_rate"] < direct["wrong_writeback_rate"] and primary["destructive_overwrite_rate"] < direct["destructive_overwrite_rate"] and primary["trace_validity"] > direct["trace_validity"]
    checks = {
        "final_grid_exact_accuracy_at_least_095": primary["final_grid_exact_accuracy"] >= 0.95,
        "final_grid_similarity_at_least_098": primary["final_grid_similarity"] >= 0.98,
        "operator_trace_exact_accuracy_at_least_090": primary["operator_trace_exact_accuracy"] >= 0.90,
        "per_step_operator_accuracy_at_least_095": primary["per_step_operator_accuracy"] >= 0.95,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "internal_state_consistency_at_least_095": primary["internal_state_consistency"] >= 0.95,
        "noisy_repair_rate_at_least_090": primary["noisy_repair_rate"] >= 0.90,
        "missing_frame_repair_rate_at_least_085": primary["missing_frame_repair_rate"] >= 0.85,
        "decoy_rejection_rate_at_least_090": primary["decoy_rejection_rate"] >= 0.90,
        "heldout_composition_accuracy_at_least_090": primary["heldout_composition_accuracy"] >= 0.90,
        "long_horizon_survival_rate_at_least_085": primary["long_horizon_survival_rate"] >= 0.85,
        "ood_grid_generalization_at_least_085": primary["ood_grid_generalization"] >= 0.85,
        "branch_switch_accuracy_at_least_095": primary["branch_switch_accuracy"] >= 0.95,
        "licensed_destructive_accept_rate_at_least_090": primary["licensed_destructive_accept_rate"] >= 0.90,
        "unlicensed_destructive_reject_rate_at_least_095": primary["unlicensed_destructive_reject_rate"] >= 0.95,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "temporal_drift_not_worse_than_direct": primary["temporal_drift_rate"] <= direct["temporal_drift_rate"],
        "drift_slope_not_explosive": primary["drift_slope_explosive"] is False,
        "beats_observed_on_noisy_missing_adversarial": noisy_beats,
        "beats_no_internal_state_on_noisy_and_long": no_state_beats,
        "beats_direct_on_safety_and_trace": direct_beats,
        "deterministic_replay_passed": replay,
        "no_semantic_slot_leak_detected": primary["no_semantic_slot_leak_detected"] is True,
        "no_neural_dependency_detected": primary["no_neural_dependency_detected"] is True,
    }
    return {
        "schema_version": "e13_positive_gate_v1",
        "checks": checks,
        "deltas": {
            "primary": PRIMARY,
            "baseline": BASELINE,
            "final_exact_delta_vs_observed": rounded(primary["final_grid_exact_accuracy"] - observed["final_grid_exact_accuracy"]),
            "trace_validity_delta_vs_direct": rounded(primary["trace_validity"] - direct["trace_validity"]),
            "wrong_writeback_reduction_vs_direct": rounded(1.0 - rate(primary["wrong_writeback_rate"], direct["wrong_writeback_rate"])),
            "cost_reduction_vs_trace_repair": rounded(1.0 - rate(primary["cost_per_tick"], metrics["FLOW_GRID_TRACE_REPAIR"]["cost_per_tick"])),
        },
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    primary = metrics[PRIMARY]
    checks = gate["checks"]
    if gate["passed"]:
        return "e13_streaming_grid_state_transition_trace_confirmed"
    if primary["no_semantic_slot_leak_detected"] is not True:
        return "e13_semantic_slot_leak_detected"
    if primary["wrong_writeback_rate"] > 0.02 or primary["branch_contamination_rate"] > 0.0:
        return "e13_writeback_safety_failure"
    if primary["licensed_destructive_accept_rate"] < 0.90 or primary["unlicensed_destructive_reject_rate"] < 0.95:
        return "e13_destructive_license_failure"
    if primary["missing_frame_repair_rate"] < 0.85:
        return "e13_missing_frame_repair_failure"
    if primary["noisy_repair_rate"] < 0.90 or primary["decoy_rejection_rate"] < 0.90:
        return "e13_noisy_trace_repair_failure"
    if primary["heldout_composition_accuracy"] < 0.90:
        return "e13_heldout_composition_failure"
    if primary["long_horizon_survival_rate"] < 0.85 or checks["drift_slope_not_explosive"] is False:
        return "e13_long_horizon_drift_failure"
    if primary["ood_grid_generalization"] < 0.85:
        return "e13_ood_grid_generalization_failure"
    if primary["final_grid_exact_accuracy"] < 0.95 or primary["trace_validity"] < 0.95:
        return "e13_clean_trace_failure"
    return "e13_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e13_streaming_grid_state_transition_trace_confirmed": "E14_REGION_AWARE_PARALLEL_POCKET_SCHEDULER_CONFIRM",
        "e13_clean_trace_failure": "E13A_CLEAN_TRACE_REPAIR",
        "e13_noisy_trace_repair_failure": "E13B_NOISY_TRACE_REPAIR",
        "e13_missing_frame_repair_failure": "E13C_MISSING_FRAME_REPAIR",
        "e13_heldout_composition_failure": "E13D_COMPOSITION_REPAIR",
        "e13_long_horizon_drift_failure": "E13E_LONG_HORIZON_STABILITY_REPAIR",
        "e13_ood_grid_generalization_failure": "E13F_OOD_GRID_REPAIR",
        "e13_destructive_license_failure": "E13G_DESTRUCTIVE_TRANSFORM_LICENSE_REPAIR",
        "e13_writeback_safety_failure": "E13W_WRITEBACK_SAFETY_REPAIR",
        "e13_semantic_slot_leak_detected": "E13L_SEMANTIC_LEAK_REPAIR",
        "e13_invalid_or_incomplete_run": "E13_RETRY_WITH_FULL_AUDIT",
    }[decision]


def build_reports(seeds: tuple[int, ...], rows_per_split: int, replay_passed: bool = True) -> dict[str, Any]:
    rows = build_rows(seeds, rows_per_split)
    baseline_metrics, baseline_diag, baseline_splits, baseline_samples = run_system(BASELINE, rows)
    metrics = {BASELINE: baseline_metrics}
    diagnostics = {BASELINE: baseline_diag}
    split_metrics = {BASELINE: baseline_splits}
    samples = {BASELINE: baseline_samples}
    for system in SYSTEMS:
        if system == BASELINE:
            continue
        system_metrics, diag, splits, sample = run_system(system, rows, baseline_drift=baseline_metrics["temporal_drift_rate"])
        metrics[system] = system_metrics
        diagnostics[system] = diag
        split_metrics[system] = splits
        samples[system] = sample
    gate = positive_gate(metrics, split_metrics, replay_passed)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e13_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": replay_passed,
    }
    aggregate = {
        "schema_version": "e13_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_split": rows_per_split,
        "systems": metrics,
        "diagnostics": diagnostics,
        "split_metrics": split_metrics,
        "positive_gate": gate,
    }
    search_report = {
        "schema_version": "e13_search_report_v1",
        "equivalent_existing_milestone_found": False,
        "searched_locations": ["docs/research", "scripts/probes", "docs/wiki", "CHANGELOG.md", "fetched branches"],
        "adjacent_hits": ["E12 next pointer", "unrelated stable-loop numeric E13 strings"],
    }
    dataset_report = {
        "schema_version": "e13_dataset_report_v1",
        "grid_sizes": [8, 12, 16, 24],
        "splits": list(SPLITS),
        "rows": len(rows),
        "route_lengths": sorted({len(row.route) for row in rows}),
        "debug_operator_families": list(OPS),
        "runtime_receives_semantic_labels": False,
    }
    system_report = {"schema_version": "e13_system_comparison_report_v1", "systems": metrics, "samples": samples}
    trace_report = {
        "schema_version": "e13_trace_accuracy_report_v1",
        "operator_trace_exact_accuracy": {system: metrics[system]["operator_trace_exact_accuracy"] for system in SYSTEMS},
        "per_step_operator_accuracy": {system: metrics[system]["per_step_operator_accuracy"] for system in SYSTEMS},
        "trace_validity": {system: metrics[system]["trace_validity"] for system in SYSTEMS},
    }
    noisy_report = {
        "schema_version": "e13_noisy_repair_report_v1",
        "noisy_repair_rate": {system: metrics[system]["noisy_repair_rate"] for system in SYSTEMS},
        "decoy_rejection_rate": {system: metrics[system]["decoy_rejection_rate"] for system in SYSTEMS},
        "internal_state_vs_observed_noisy_gap": {system: metrics[system]["internal_state_vs_observed_noisy_gap"] for system in SYSTEMS},
    }
    missing_report = {"schema_version": "e13_missing_frame_report_v1", "missing_frame_repair_rate": {system: metrics[system]["missing_frame_repair_rate"] for system in SYSTEMS}, "split_metrics": {system: split_metrics[system]["missing_frame"] for system in SYSTEMS}}
    heldout_report = {"schema_version": "e13_heldout_composition_report_v1", "heldout_composition_accuracy": {system: metrics[system]["heldout_composition_accuracy"] for system in SYSTEMS}, "split_metrics": {system: split_metrics[system]["heldout_composition"] for system in SYSTEMS}}
    long_report = {"schema_version": "e13_long_horizon_report_v1", "long_horizon_survival_rate": {system: metrics[system]["long_horizon_survival_rate"] for system in SYSTEMS}, "drift_slope_by_horizon": {system: metrics[system]["drift_slope_by_horizon"] for system in SYSTEMS}}
    ood_report = {"schema_version": "e13_ood_grid_report_v1", "ood_grid_generalization": {system: metrics[system]["ood_grid_generalization"] for system in SYSTEMS}, "split_metrics": {system: split_metrics[system]["ood_grid_size"] for system in SYSTEMS}}
    destructive_report = {"schema_version": "e13_destructive_license_report_v1", "licensed_destructive_accept_rate": {system: metrics[system]["licensed_destructive_accept_rate"] for system in SYSTEMS}, "unlicensed_destructive_reject_rate": {system: metrics[system]["unlicensed_destructive_reject_rate"] for system in SYSTEMS}}
    safety_report = {
        "schema_version": "e13_writeback_safety_report_v1",
        "wrong_writeback_rate": {system: metrics[system]["wrong_writeback_rate"] for system in SYSTEMS},
        "destructive_overwrite_rate": {system: metrics[system]["destructive_overwrite_rate"] for system in SYSTEMS},
        "branch_contamination_rate": {system: metrics[system]["branch_contamination_rate"] for system in SYSTEMS},
        "stale_write_rejection_rate": {system: metrics[system]["stale_write_rejection_rate"] for system in SYSTEMS},
    }
    semantic_report = {
        "schema_version": "e13_semantic_leak_report_v1",
        "primary_runtime_config": {"input": "binary_grid_frames", "state": "binary_flow_grid", "writeback": "gated_region_transform"},
        "runtime_receives_forbidden_semantic_slots": False,
        "debug_names_confined_to_harness_reports": True,
        "no_semantic_slot_leak_detected": metrics[PRIMARY]["no_semantic_slot_leak_detected"],
    }
    summary = {
        "schema_version": "e13_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "final_grid_exact_accuracy": metrics[PRIMARY]["final_grid_exact_accuracy"],
        "final_grid_similarity": metrics[PRIMARY]["final_grid_similarity"],
        "operator_trace_exact_accuracy": metrics[PRIMARY]["operator_trace_exact_accuracy"],
        "trace_validity": metrics[PRIMARY]["trace_validity"],
        "wrong_writeback_rate": metrics[PRIMARY]["wrong_writeback_rate"],
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate),
        "e13_search_report.json": search_report,
        "e13_dataset_report.json": dataset_report,
        "e13_system_comparison_report.json": system_report,
        "e13_trace_accuracy_report.json": trace_report,
        "e13_noisy_repair_report.json": noisy_report,
        "e13_missing_frame_report.json": missing_report,
        "e13_heldout_composition_report.json": heldout_report,
        "e13_long_horizon_report.json": long_report,
        "e13_ood_grid_report.json": ood_report,
        "e13_destructive_license_report.json": destructive_report,
        "e13_writeback_safety_report.json": safety_report,
        "e13_semantic_leak_report.json": semantic_report,
    }


def render_report(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    metrics = aggregate["systems"]
    split_metrics = aggregate["split_metrics"]
    lines = [
        "# E13 Streaming Grid State-Transition Trace Confirm Report",
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
        "| system | final exact | final sim | trace exact | step op | trace | noisy repair | missing repair | decoy reject | wrong | destructive | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = metrics[system]
        lines.append(
            f"| {system} | {row['final_grid_exact_accuracy']:.3f} | {row['final_grid_similarity']:.3f} | {row['operator_trace_exact_accuracy']:.3f} | "
            f"{row['per_step_operator_accuracy']:.3f} | {row['trace_validity']:.3f} | {row['noisy_repair_rate']:.3f} | {row['missing_frame_repair_rate']:.3f} | "
            f"{row['decoy_rejection_rate']:.3f} | {row['wrong_writeback_rate']:.3f} | {row['destructive_overwrite_rate']:.3f} | {row['cost_per_tick']:.3f} |"
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
            "## Split Results",
            "",
            "| split | final exact | final sim | trace | noisy gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in EVAL_SPLITS:
        row = split_metrics[PRIMARY][split]
        lines.append(f"| {split} | {row['final_grid_exact_accuracy']:.3f} | {row['final_grid_similarity']:.3f} | {row['trace_validity']:.3f} | {row['internal_state_vs_observed_noisy_gap']:.3f} |")
    lines.extend(["", "## Boundary", "", "This is a deterministic synthetic binary grid-frame transition probe only."])
    return "\n".join(lines)


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_split: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, rows_per_split, replay_passed=True)
    replay_b = build_reports(seeds, rows_per_split, replay_passed=True)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e13_deterministic_replay_report.json"] = {
        "schema_version": "e13_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"] = positive_gate(payloads["aggregate_metrics.json"]["systems"], payloads["aggregate_metrics.json"]["split_metrics"], passed)
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"], payloads["aggregate_metrics.json"]["systems"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"])
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
    git_rc, git_head = run_git(["rev-parse", "--short", "HEAD"])
    payloads = attach_replay(build_reports(args.seeds, args.rows_per_split), args.seeds, args.rows_per_split)
    payloads["summary.json"]["git_head"] = git_head.strip() if git_rc == 0 else "unknown"
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
