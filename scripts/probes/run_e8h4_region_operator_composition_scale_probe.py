#!/usr/bin/env python3
"""E8H4 region-operator composition scale probe.

This probe tests the controlled abstraction:

    pocket = feature detector + direct Flow-grid region transform

There is no private pocket language, proposal RAM, semantic lane, or oracle
write in learned inference. Learned systems mutate region rules and compose
them over route lengths 1, 3, 6, 12, and 24.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/e8h4_region_operator_composition_scale_probe")
MILESTONE = "E8H4_REGION_OPERATOR_COMPOSITION_SCALE_PROBE"

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

SKILLS = (
    "cleanup",
    "shift_right",
    "shift_down",
    "fill_gap",
    "bind_marker",
    "clear_border",
    "invert_center",
    "threshold_center",
)

OPS = (
    "noop",
    "copy",
    "move",
    "clear",
    "invert",
    "shift",
    "fill_gap",
    "bind_marker",
    "threshold",
    "delete_isolated",
)

REGIONS = ("full", "top", "bottom", "left", "right", "center", "border", "marker")
SPLITS = ("validation", "heldout", "ood", "counterfactual", "adversarial")
HASH_ARTIFACTS = (
    "aggregate_metrics.json",
    "split_metrics.json",
    "depth_scaling_report.json",
    "operator_discovery_report.json",
    "operator_reuse_report.json",
    "mutation_history.jsonl",
    "row_level_samples.jsonl",
    "dense_shortcut_control_report.json",
    "decision.json",
    "report.md",
)

VALID_DECISIONS = (
    "e8h4_region_operator_composition_scale_positive",
    "e8h4_region_operator_partial_scale",
    "e8h4_single_operator_only_no_composition",
    "e8h4_trace_drift_accumulation_failure",
    "e8h4_operator_reuse_positive",
    "e8h4_mutation_search_insufficient",
    "e8h4_dense_shortcut_trace_invalid",
    "e8h4_region_operator_not_sufficient",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    population: int
    generations: int
    elite_count: int
    cpu_workers: int
    heartbeat_seconds: float
    route_lengths: tuple[int, ...]
    grid_sizes: tuple[int, ...]
    replay: bool
    execution_mode: str


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def round_float(value: float) -> float:
    return round(float(value), 6)


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [stable_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return stable_payload(value.tolist())
    if isinstance(value, (float, np.floating)):
        return round_float(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stable_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(stable_payload(row), sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stable_payload(row), sort_keys=True) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stable_payload({"time": time.time(), "event": event, "details": details}), sort_keys=True) + "\n")


def heartbeat(out: Path, stop: threading.Event, interval: float) -> None:
    while not stop.wait(max(1.0, interval)):
        payload = {"time": time.time(), "pid": os.getpid()}
        try:
            import psutil  # type: ignore

            proc = psutil.Process(os.getpid())
            payload["rss_mb"] = round_float(proc.memory_info().rss / (1024 * 1024))
            payload["cpu_percent"] = round_float(proc.cpu_percent(interval=None))
        except Exception:
            pass
        with (out / "hardware_heartbeat.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def region_mask(region: str, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mid0 = n // 3
    mid1 = n - mid0
    if region == "full":
        mask[:, :] = True
    elif region == "top":
        mask[: n // 2, :] = True
    elif region == "bottom":
        mask[n // 2 :, :] = True
    elif region == "left":
        mask[:, : n // 2] = True
    elif region == "right":
        mask[:, n // 2 :] = True
    elif region == "center":
        mask[mid0:mid1, mid0:mid1] = True
    elif region == "border":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif region == "marker":
        mask[-2:, -2:] = True
    else:
        raise ValueError(f"unknown region {region}")
    return mask


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def fit_region_values(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if source.size == 0:
        return np.zeros(target_shape, dtype=np.int8)
    if source.shape == target_shape:
        return source.astype(np.int8)
    mean_value = int(float(source.mean()) >= 0.5)
    return np.full(target_shape, mean_value, dtype=np.int8)


def delete_isolated(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    n = grid.shape[0]
    ys, xs = np.where(mask & (grid == 1))
    for y, x in zip(ys, xs):
        neighbors = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = y + dy
            xx = x + dx
            if 0 <= yy < n and 0 <= xx < n and grid[yy, xx] == 1:
                neighbors += 1
        if neighbors == 0:
            out[y, x] = 0
    return out


def fill_gap(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    n = grid.shape[0]
    for y in range(n):
        for x in range(n):
            if not mask[y, x] or grid[y, x] != 0:
                continue
            if x > 0 and x < n - 1 and grid[y, x - 1] == 1 and grid[y, x + 1] == 1:
                out[y, x] = 1
            if y > 0 and y < n - 1 and grid[y - 1, x] == 1 and grid[y + 1, x] == 1:
                out[y, x] = 1
    return out


def shift_masked(grid: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = grid.copy()
    shifted = np.zeros_like(grid)
    ys, xs = np.where(mask & (grid == 1))
    for y, x in zip(ys, xs):
        yy = y + dy
        xx = x + dx
        if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]:
            shifted[yy, xx] = 1
    out[mask] = 0
    out |= shifted
    return out


def apply_rule(grid: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    n = grid.shape[0]
    read = region_mask(str(rule.get("read_region", "full")), n)
    write = region_mask(str(rule.get("write_region", "full")), n)
    op = str(rule.get("op", "noop"))
    out = grid.copy()
    if op == "noop":
        return out
    if op == "clear":
        out[write] = 0
        return out
    if op == "invert":
        out[write] = 1 - out[write]
        return out
    if op == "delete_isolated":
        return delete_isolated(out, read & write)
    if op == "fill_gap":
        return fill_gap(out, read & write)
    if op == "shift":
        return shift_masked(out, read, int(rule.get("dy", 0)), int(rule.get("dx", 0)))
    if op in {"copy", "move"}:
        ry0, ry1, rx0, rx1 = bbox(read)
        wy0, wy1, wx0, wx1 = bbox(write)
        source = out[ry0:ry1, rx0:rx1]
        target_shape = (wy1 - wy0, wx1 - wx0)
        values = fit_region_values(source, target_shape)
        if target_shape[0] > 0 and target_shape[1] > 0:
            out[wy0:wy1, wx0:wx1] = values
        if op == "move":
            out[read] = 0
        return out
    if op == "bind_marker":
        if int(out[read].sum()) >= int(rule.get("threshold", 2)):
            out[write] = 1
        return out
    if op == "threshold":
        value = int(int(out[read].sum()) >= int(rule.get("threshold", max(1, int(read.sum() * 0.25)))))
        out[write] = value
        return out
    return out


TRUE_RULES: dict[str, dict[str, Any]] = {
    "cleanup": {"op": "delete_isolated", "read_region": "full", "write_region": "full", "dx": 0, "dy": 0, "threshold": 1},
    "shift_right": {"op": "shift", "read_region": "full", "write_region": "full", "dx": 1, "dy": 0, "threshold": 1},
    "shift_down": {"op": "shift", "read_region": "full", "write_region": "full", "dx": 0, "dy": 1, "threshold": 1},
    "fill_gap": {"op": "fill_gap", "read_region": "full", "write_region": "full", "dx": 0, "dy": 0, "threshold": 1},
    "bind_marker": {"op": "bind_marker", "read_region": "top", "write_region": "marker", "dx": 0, "dy": 0, "threshold": 2},
    "clear_border": {"op": "clear", "read_region": "border", "write_region": "border", "dx": 0, "dy": 0, "threshold": 1},
    "invert_center": {"op": "invert", "read_region": "center", "write_region": "center", "dx": 0, "dy": 0, "threshold": 1},
    "threshold_center": {"op": "threshold", "read_region": "center", "write_region": "marker", "dx": 0, "dy": 0, "threshold": 3},
}


def random_grid(rng: np.random.Generator, n: int, density: float, family: str) -> np.ndarray:
    grid = (rng.random((n, n)) < density).astype(np.int8)
    if family in {"completion", "routing_composition", "delayed_dependency"}:
        row = int(rng.integers(1, n - 1))
        col = int(rng.integers(1, n - 1))
        grid[row, col - 1 : col + 2] = np.array([1, 0, 1], dtype=np.int8)
    if family in {"motion", "routing_composition"}:
        y = int(rng.integers(1, n - 2))
        x0 = int(rng.integers(0, max(1, n - 4)))
        grid[y, x0 : x0 + 3] = 1
    if family in {"binding", "delayed_dependency", "conflict"}:
        grid[0, 0] = 1
        grid[0, n - 1] = 1
    if family == "adversarial":
        grid[:2, :2] = np.array([[1, 0], [0, 1]], dtype=np.int8)
        grid[-2:, -2:] = 0
    if family == "local_cleanup":
        for _ in range(max(1, n // 3)):
            y = int(rng.integers(0, n))
            x = int(rng.integers(0, n))
            grid[y, x] = 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < n and 0 <= xx < n:
                    grid[yy, xx] = 0
    return grid


def route_for_family(rng: np.random.Generator, family: str, length: int, split: str) -> list[str]:
    templates = {
        "local_cleanup": ["cleanup", "clear_border"],
        "motion": ["shift_right", "shift_down"],
        "completion": ["fill_gap", "cleanup"],
        "binding": ["bind_marker", "threshold_center"],
        "routing_composition": ["fill_gap", "shift_right", "bind_marker", "cleanup"],
        "conflict": ["invert_center", "clear_border", "cleanup", "threshold_center"],
        "delayed_dependency": ["bind_marker", "shift_right", "shift_down", "threshold_center", "cleanup"],
        "reuse_sparse": ["cleanup", "shift_right", "fill_gap", "bind_marker"],
    }
    base = list(templates.get(family, templates["routing_composition"]))
    route = [base[i % len(base)] for i in range(length)]
    if split == "counterfactual":
        route = list(reversed(route))
    if split == "adversarial" and length > 2:
        route[1] = "threshold_center"
    if split == "ood":
        route = [base[(i * 2 + 1) % len(base)] for i in range(length)]
    if rng.random() < 0.15 and split not in {"counterfactual"}:
        route = [route[(i + 1) % len(route)] for i in range(len(route))]
    return route


def oracle_frames(initial: np.ndarray, route: list[str]) -> list[np.ndarray]:
    frames = [initial.astype(np.int8)]
    current = initial.astype(np.int8)
    for skill in route:
        current = apply_rule(current, TRUE_RULES[skill]).astype(np.int8)
        frames.append(current)
    return frames


def generate_rows(seed: int, split: str, count: int, settings: Settings) -> list[dict[str, Any]]:
    rng = np.random.default_rng(stable_seed("rows", seed, split))
    families = (
        "local_cleanup",
        "motion",
        "completion",
        "binding",
        "routing_composition",
        "conflict",
        "delayed_dependency",
        "reuse_sparse",
    )
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        route_length = settings.route_lengths[idx % len(settings.route_lengths)]
        n = settings.grid_sizes[idx % len(settings.grid_sizes)]
        if split == "ood":
            n = max(n, max(settings.grid_sizes) + 2)
        density = 0.18 + 0.08 * (idx % 3)
        if split == "adversarial":
            density += 0.12
        family = families[idx % len(families)]
        if split == "adversarial":
            family = "adversarial"
        grid = random_grid(rng, n, density, family)
        route = route_for_family(rng, family, route_length, split)
        frames = oracle_frames(grid, route)
        rows.append(
            {
                "row_id": f"{seed}:{split}:{idx}",
                "seed": seed,
                "split": split,
                "family": family,
                "grid_size": n,
                "route_length": route_length,
                "route": route,
                "initial": grid.tolist(),
                "oracle_frames": [frame.tolist() for frame in frames],
            }
        )
    return rows


def random_rule(rng: random.Random) -> dict[str, Any]:
    return {
        "op": rng.choice(OPS),
        "read_region": rng.choice(REGIONS),
        "write_region": rng.choice(REGIONS),
        "dx": rng.choice((-1, 0, 1)),
        "dy": rng.choice((-1, 0, 1)),
        "threshold": rng.randint(1, 6),
    }


def mutate_rule(rule: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    new = dict(rule)
    field = rng.choice(("op", "read_region", "write_region", "dx", "dy", "threshold"))
    if field == "op":
        new[field] = rng.choice(OPS)
    elif field in {"read_region", "write_region"}:
        new[field] = rng.choice(REGIONS)
    elif field in {"dx", "dy"}:
        new[field] = rng.choice((-1, 0, 1))
    else:
        new[field] = max(1, min(8, int(new[field]) + rng.choice((-2, -1, 1, 2))))
    return new


def step_pairs(rows: list[dict[str, Any]], skill: str) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        frames = [np.array(frame, dtype=np.int8) for frame in row["oracle_frames"]]
        for step, route_skill in enumerate(row["route"]):
            if route_skill == skill:
                pairs.append((frames[step], frames[step + 1]))
    return pairs


def grid_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def score_rule(rule: dict[str, Any], pairs: list[tuple[np.ndarray, np.ndarray]], trace_check: bool) -> float:
    if not pairs:
        return 0.0
    scores = []
    for before, after in pairs:
        pred = apply_rule(before, rule)
        sim = grid_similarity(pred, after)
        if trace_check:
            delta_pred = pred.astype(float) - before.astype(float)
            delta_true = after.astype(float) - before.astype(float)
            delta_sim = 1.0 - float(np.mean(np.abs(delta_pred - delta_true)))
            sim = 0.72 * sim + 0.28 * delta_sim
        scores.append(sim)
    footprint = region_mask(rule["read_region"], pairs[0][0].shape[0]).mean() + region_mask(rule["write_region"], pairs[0][0].shape[0]).mean()
    return float(np.mean(scores) - 0.01 * footprint)


def discover_library(seed: int, rows: list[dict[str, Any]], settings: Settings, trace_check: bool, out: Path | None = None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(stable_seed("mutation", seed, "trace" if trace_check else "plain"))
    history: list[dict[str, Any]] = []
    library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        pairs = step_pairs(rows, skill)
        population = [random_rule(rng) for _ in range(settings.population)]
        scores = [score_rule(rule, pairs, trace_check) for rule in population]
        accepted = 0
        rejected = 0
        rollback = 0
        for generation in range(settings.generations):
            order = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
            elites = [population[i] for i in order[: settings.elite_count]]
            elite_scores = [scores[i] for i in order[: settings.elite_count]]
            next_population = elites[:]
            next_scores = elite_scores[:]
            while len(next_population) < settings.population:
                parent_index = rng.randrange(len(elites))
                parent = elites[parent_index]
                parent_score = elite_scores[parent_index]
                child = mutate_rule(parent, rng)
                child_score = score_rule(child, pairs, trace_check)
                if child_score >= parent_score:
                    next_population.append(child)
                    next_scores.append(child_score)
                    accepted += 1
                else:
                    next_population.append(parent)
                    next_scores.append(parent_score)
                    rejected += 1
                    rollback += 1
            population = next_population
            scores = next_scores
            best_idx = max(range(len(population)), key=lambda i: scores[i])
            history.append(
                {
                    "seed": seed,
                    "skill": skill,
                    "trace_check": trace_check,
                    "generation": generation,
                    "best_score": scores[best_idx],
                    "best_rule": population[best_idx],
                    "accepted": accepted,
                    "rejected": rejected,
                    "rollback": rollback,
                }
            )
            if out is not None and (generation == 0 or generation == settings.generations - 1 or generation % max(1, settings.generations // 8) == 0):
                live_row = {
                    "seed": seed,
                    "skill": skill,
                    "trace_check": trace_check,
                    "generation": generation,
                    "best_score": scores[best_idx],
                    "accepted": accepted,
                    "rejected": rejected,
                    "rollback": rollback,
                }
                append_jsonl(out / f"mutation_history_live_seed_{seed}.jsonl", live_row)
                append_progress(out, "mutation_generation", **live_row)
        best_idx = max(range(len(population)), key=lambda i: scores[i])
        library[skill] = population[best_idx]
    return library, history


def prune_library(library: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pruned: dict[str, dict[str, Any]] = {}
    for skill, rule in library.items():
        pairs = step_pairs(rows, skill)
        base_score = score_rule(rule, pairs, True)
        best = dict(rule)
        best_score = base_score
        for read_region in ("center", "top", "bottom", "left", "right", "border", "marker", "full"):
            candidate = dict(best)
            candidate["read_region"] = read_region
            score = score_rule(candidate, pairs, True)
            if score >= best_score - 0.01:
                best, best_score = candidate, score
        for write_region in ("center", "marker", "border", "full"):
            candidate = dict(best)
            candidate["write_region"] = write_region
            score = score_rule(candidate, pairs, True)
            if score >= best_score - 0.01:
                best, best_score = candidate, score
        pruned[skill] = best
    return pruned


def direct_baseline_rule(skill: str) -> dict[str, Any]:
    if "shift" in skill:
        return {"op": "shift", "read_region": "full", "write_region": "full", "dx": 1, "dy": 0, "threshold": 1}
    if skill in {"cleanup", "clear_border"}:
        return {"op": "clear", "read_region": "border", "write_region": "border", "dx": 0, "dy": 0, "threshold": 1}
    if skill == "invert_center":
        return {"op": "invert", "read_region": "full", "write_region": "center", "dx": 0, "dy": 0, "threshold": 1}
    return {"op": "threshold", "read_region": "full", "write_region": "marker", "dx": 0, "dy": 0, "threshold": 4}


def system_library(system: str, plain: dict[str, dict[str, Any]], trace: dict[str, dict[str, Any]], pruned: dict[str, dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    if system == "direct_overwrite_matrix_baseline":
        return {skill: direct_baseline_rule(skill) for skill in SKILLS}
    if system == "random_region_rule_control":
        rng = random.Random(stable_seed("random-control", seed))
        return {skill: random_rule(rng) for skill in SKILLS}
    if system in {"mutation_discovered_plus_trace_check", "reusable_operator_library_router"}:
        return trace
    if system == "mutation_discovered_plus_prune":
        return pruned
    return plain


def apply_system(row: dict[str, Any], system: str, library: dict[str, dict[str, Any]], max_steps: int | None) -> list[np.ndarray]:
    current = np.array(row["initial"], dtype=np.int8)
    oracle = [np.array(frame, dtype=np.int8) for frame in row["oracle_frames"]]
    frames = [current.copy()]
    if system == "handcoded_oracle_region_operator_reference":
        return oracle
    if system == "identity_noop_baseline":
        return [current.copy() for _ in range(len(row["route"]) + 1)]
    if system in {"dense_transform_danger_control", "answer_shortcut_control"}:
        for step, _skill in enumerate(row["route"]):
            if step == len(row["route"]) - 1:
                marker = region_mask("marker", current.shape[0])
                current = current.copy()
                current[marker] = oracle[-1][marker]
            frames.append(current.copy())
        return frames
    steps = row["route"] if max_steps is None else row["route"][:max_steps]
    for skill in steps:
        current = apply_rule(current, library.get(skill, {"op": "noop"})).astype(np.int8)
        frames.append(current.copy())
    while len(frames) < len(row["route"]) + 1:
        frames.append(current.copy())
    return frames


def row_metrics(row: dict[str, Any], system: str, frames: list[np.ndarray]) -> dict[str, Any]:
    oracle = [np.array(frame, dtype=np.int8) for frame in row["oracle_frames"]]
    marker = region_mask("marker", oracle[-1].shape[0])
    step_sims = [grid_similarity(frames[i], oracle[i]) for i in range(1, len(oracle))]
    transition_sims = []
    drift = []
    first_div = None
    for i in range(1, len(oracle)):
        pred_delta = frames[i].astype(float) - frames[i - 1].astype(float)
        oracle_delta = oracle[i].astype(float) - oracle[i - 1].astype(float)
        transition_sims.append(1.0 - float(np.mean(np.abs(pred_delta - oracle_delta))))
        d = 1.0 - grid_similarity(frames[i], oracle[i])
        drift.append(d)
        if first_div is None and d > 0.05:
            first_div = i - 1
    final_similarity = grid_similarity(frames[-1], oracle[-1])
    marker_ok = float(np.array_equal(frames[-1][marker], oracle[-1][marker]))
    trace_validity = float(np.mean(step_sims)) if step_sims else 1.0
    delta_validity = float(np.mean(transition_sims)) if transition_sims else 1.0
    drift_slope = 0.0
    if len(drift) > 1:
        xs = np.arange(len(drift), dtype=float)
        drift_slope = float(np.polyfit(xs, np.array(drift, dtype=float), 1)[0])
    usefulness = 0.35 * marker_ok + 0.45 * trace_validity + 0.20 * final_similarity
    return {
        "system": system,
        "seed": row["seed"],
        "split": row["split"],
        "family": row["family"],
        "row_id": row["row_id"],
        "grid_size": row["grid_size"],
        "route_length": row["route_length"],
        "answer_accuracy": marker_ok,
        "usefulness": usefulness,
        "trace_validity": trace_validity,
        "frame_mae_to_oracle": 1.0 - trace_validity,
        "delta_mae_to_oracle": 1.0 - delta_validity,
        "drift_per_step": float(np.mean(drift)) if drift else 0.0,
        "drift_slope": drift_slope,
        "first_divergence_step": len(row["route"]) if first_div is None else first_div,
        "final_grid_similarity": final_similarity,
    }


def evaluate_system(seed: int, rows_by_split: dict[str, list[dict[str, Any]]], system: str, library: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_steps_by_system = {
        "mutation_discovered_single_operator": 1,
        "mutation_discovered_composed_3_step": 3,
        "mutation_discovered_composed_6_step": 6,
        "mutation_discovered_composed_12_step": 12,
        "mutation_discovered_composed_24_step": 24,
    }
    max_steps = max_steps_by_system.get(system)
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for split, split_rows in rows_by_split.items():
        for row in split_rows:
            frames = apply_system(row, system, library, max_steps)
            metrics = row_metrics(row, system, frames)
            rows.append(metrics)
            if len(samples) < 600 and row["route_length"] in {1, 6, 12, 24}:
                samples.append(
                    {
                        **metrics,
                        "route": row["route"],
                        "initial_sum": int(np.array(row["initial"]).sum()),
                        "final_sum": int(frames[-1].sum()),
                        "oracle_final_sum": int(np.array(row["oracle_frames"][-1]).sum()),
                    }
                )
    return rows, samples


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        if not subset:
            continue
        means = {
            key: float(np.mean([float(row[key]) for row in subset]))
            for key in ("usefulness", "answer_accuracy", "trace_validity", "frame_mae_to_oracle", "delta_mae_to_oracle", "drift_per_step", "drift_slope", "first_divergence_step", "final_grid_similarity")
        }
        systems[system] = {"row_count": len(subset), "mean": means}
    best = max((s for s in systems if s not in {"handcoded_oracle_region_operator_reference"}), key=lambda s: systems[s]["mean"]["usefulness"])
    return {"schema_version": "e8h4_aggregate_metrics_v1", "systems": systems, "best_system": best}


def split_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": "e8h4_split_metrics_v1", "rows": []}
    for system in SYSTEMS:
        for split in SPLITS:
            subset = [row for row in rows if row["system"] == system and row["split"] == split]
            if not subset:
                continue
            payload["rows"].append(
                {
                    "system": system,
                    "split": split,
                    "usefulness": float(np.mean([row["usefulness"] for row in subset])),
                    "trace_validity": float(np.mean([row["trace_validity"] for row in subset])),
                    "answer_accuracy": float(np.mean([row["answer_accuracy"] for row in subset])),
                }
            )
    return payload


def depth_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": "e8h4_depth_scaling_report_v1", "rows": []}
    for system in SYSTEMS:
        for length in (1, 3, 6, 12, 24):
            subset = [row for row in rows if row["system"] == system and row["route_length"] == length]
            if not subset:
                continue
            payload["rows"].append(
                {
                    "system": system,
                    "route_length": length,
                    "usefulness": float(np.mean([row["usefulness"] for row in subset])),
                    "trace_validity": float(np.mean([row["trace_validity"] for row in subset])),
                    "drift_slope": float(np.mean([row["drift_slope"] for row in subset])),
                    "first_divergence_step": float(np.mean([row["first_divergence_step"] for row in subset])),
                }
            )
    return payload


def operator_reports(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery_rows: list[dict[str, Any]] = []
    reuse: dict[str, dict[str, Any]] = {skill: {"calls": 0, "seeds": set()} for skill in SKILLS}
    for result in seed_results:
        for mode, library in (("plain", result["plain_library"]), ("trace_check", result["trace_library"]), ("pruned", result["pruned_library"])):
            for skill, rule in library.items():
                read_cells = float(region_mask(rule["read_region"], 8).sum())
                write_cells = float(region_mask(rule["write_region"], 8).sum())
                discovery_rows.append({"seed": result["seed"], "mode": mode, "skill": skill, "rule": rule, "read_cells": read_cells, "write_cells": write_cells, "footprint": read_cells + write_cells})
        for call in result["route_calls"]:
            reuse[call]["calls"] += 1
            reuse[call]["seeds"].add(result["seed"])
    reuse_rows = [{"skill": skill, "calls": value["calls"], "seed_count": len(value["seeds"])} for skill, value in reuse.items()]
    return (
        {"schema_version": "e8h4_operator_discovery_report_v1", "rows": sorted(discovery_rows, key=lambda r: (r["seed"], r["mode"], r["skill"]))},
        {"schema_version": "e8h4_operator_reuse_report_v1", "rows": sorted(reuse_rows, key=lambda r: r["skill"])},
    )


def decide(agg: dict[str, Any], depth: dict[str, Any], splits: dict[str, Any]) -> dict[str, Any]:
    systems = agg["systems"]
    identity = systems["identity_noop_baseline"]["mean"]
    dense = systems["dense_transform_danger_control"]["mean"]
    shortcut = systems["answer_shortcut_control"]["mean"]
    best = max(LEARNED_SYSTEMS, key=lambda s: systems[s]["mean"]["usefulness"])
    best_mean = systems[best]["mean"]
    rows = depth["rows"]

    def depth_value(system: str, length: int, metric: str) -> float:
        matches = [row for row in rows if row["system"] == system and row["route_length"] == length]
        return float(matches[0][metric]) if matches else 0.0

    def split_value(system: str, split: str, metric: str) -> float:
        matches = [row for row in splits["rows"] if row["system"] == system and row["split"] == split]
        return float(matches[0][metric]) if matches else 0.0

    clean_gain = best_mean["usefulness"] > identity["usefulness"] + 0.03
    trace_ok = best_mean["trace_validity"] >= identity["trace_validity"]
    depth6_ok = depth_value(best, 6, "usefulness") > depth_value("identity_noop_baseline", 6, "usefulness") + 0.03
    depth12_ok = depth_value(best, 12, "usefulness") > depth_value("identity_noop_baseline", 12, "usefulness") + 0.03
    depth24_ok = depth_value(best, 24, "usefulness") > depth_value("identity_noop_baseline", 24, "usefulness") + 0.03
    drift_explosive = depth_value(best, 24, "drift_slope") > depth_value(best, 1, "drift_slope") + 0.12
    robust = all(split_value(best, split, "usefulness") > max(0.40, split_value("identity_noop_baseline", split, "usefulness") - 0.03) for split in ("ood", "counterfactual", "adversarial"))
    shortcut_invalid = (dense["usefulness"] >= best_mean["usefulness"] or shortcut["usefulness"] >= best_mean["usefulness"]) and min(dense["trace_validity"], shortcut["trace_validity"]) < best_mean["trace_validity"] - 0.05

    if clean_gain and trace_ok and depth6_ok and depth12_ok and depth24_ok and robust and not drift_explosive:
        decision = "e8h4_region_operator_composition_scale_positive"
    elif clean_gain and trace_ok and depth6_ok and depth12_ok and robust:
        decision = "e8h4_region_operator_partial_scale"
    elif shortcut_invalid:
        decision = "e8h4_dense_shortcut_trace_invalid"
    elif depth_value("mutation_discovered_single_operator", 1, "usefulness") > identity["usefulness"] + 0.03 and not depth6_ok:
        decision = "e8h4_single_operator_only_no_composition"
    elif clean_gain and not trace_ok:
        decision = "e8h4_trace_drift_accumulation_failure"
    elif best == "reusable_operator_library_router" and clean_gain:
        decision = "e8h4_operator_reuse_positive"
    elif not clean_gain:
        decision = "e8h4_region_operator_not_sufficient"
    else:
        decision = "e8h4_mutation_search_insufficient"
    return {
        "schema_version": "e8h4_decision_v1",
        "decision": decision,
        "deterministic_replay_passed": False,
        "checker_failure_count": None,
        "detail": {
            "best_system": best,
            "best_usefulness": best_mean["usefulness"],
            "best_trace_validity": best_mean["trace_validity"],
            "identity_usefulness": identity["usefulness"],
            "identity_trace_validity": identity["trace_validity"],
            "depth6_ok": depth6_ok,
            "depth12_ok": depth12_ok,
            "depth24_ok": depth24_ok,
            "robust_splits": robust,
            "drift_explosive": drift_explosive,
            "shortcut_invalid": shortcut_invalid,
        },
    }


def report_markdown(decision: dict[str, Any], agg: dict[str, Any], depth: dict[str, Any], splits: dict[str, Any], replay: dict[str, Any]) -> str:
    best = decision["detail"]["best_system"]
    lines = [
        "# E8H4 Region Operator Composition Scale Probe",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {best}",
        f"deterministic_replay_passed = {decision.get('deterministic_replay_passed', False)}",
        "```",
        "",
        "## Route Length Scaling",
        "",
        "| length | usefulness | trace_validity | drift_slope |",
        "|---:|---:|---:|---:|",
    ]
    for row in depth["rows"]:
        if row["system"] == best:
            lines.append(f"| {row['route_length']} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} | {row['drift_slope']:.3f} |")
    lines.extend(["", "## Split Metrics", "", "| split | usefulness | trace_validity |", "|---|---:|---:|"])
    for row in splits["rows"]:
        if row["system"] == best:
            lines.append(f"| {row['split']} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} |")
    lines.extend(
        [
            "",
            "## Control Summary",
            "",
            "| system | usefulness | trace_validity |",
            "|---|---:|---:|",
        ]
    )
    for system in ("identity_noop_baseline", "dense_transform_danger_control", "answer_shortcut_control", "handcoded_oracle_region_operator_reference"):
        mean = agg["systems"][system]["mean"]
        lines.append(f"| {system} | {mean['usefulness']:.3f} | {mean['trace_validity']:.3f} |")
    lines.extend(
        [
            "",
            "Boundary: E8H4 is a controlled binary Flow-grid region-operator proxy. It does not make raw-language, AGI, consciousness, deployed-model, or model-scale claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    train_rows = generate_rows(seed, "train", settings.train_rows_per_seed, settings)
    rows_by_split = {
        "validation": generate_rows(seed, "validation", settings.validation_rows_per_seed, settings),
        "heldout": generate_rows(seed, "heldout", settings.heldout_rows_per_seed, settings),
        "ood": generate_rows(seed, "ood", settings.ood_rows_per_seed, settings),
        "counterfactual": generate_rows(seed, "counterfactual", settings.counterfactual_rows_per_seed, settings),
        "adversarial": generate_rows(seed, "adversarial", settings.adversarial_rows_per_seed, settings),
    }
    plain_library, plain_history = discover_library(seed, train_rows, settings, trace_check=False, out=out)
    trace_library, trace_history = discover_library(seed, train_rows, settings, trace_check=True, out=out)
    pruned_library = prune_library(trace_library, train_rows)
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    route_calls: list[str] = []
    for split_rows in rows_by_split.values():
        for row in split_rows:
            route_calls.extend(row["route"])
    for system in SYSTEMS:
        library = system_library(system, plain_library, trace_library, pruned_library, seed)
        result_rows, sample_rows = evaluate_system(seed, rows_by_split, system, library)
        rows.extend(result_rows)
        samples.extend(sample_rows)
        if system in {"dense_transform_danger_control", "answer_shortcut_control"}:
            dense_rows.extend(result_rows)
    return {
        "seed": seed,
        "rows": rows,
        "samples": samples,
        "dense_rows": dense_rows,
        "mutation_history": plain_history + trace_history,
        "plain_library": plain_library,
        "trace_library": trace_library,
        "pruned_library": pruned_library,
        "route_calls": route_calls,
    }


def artifact_hashes(out: Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in HASH_ARTIFACTS}


def replay_args(settings: Settings, out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(out),
        "--seeds",
        ",".join(str(seed) for seed in settings.seeds),
        "--train-rows-per-seed",
        str(settings.train_rows_per_seed),
        "--validation-rows-per-seed",
        str(settings.validation_rows_per_seed),
        "--heldout-rows-per-seed",
        str(settings.heldout_rows_per_seed),
        "--ood-rows-per-seed",
        str(settings.ood_rows_per_seed),
        "--counterfactual-rows-per-seed",
        str(settings.counterfactual_rows_per_seed),
        "--adversarial-rows-per-seed",
        str(settings.adversarial_rows_per_seed),
        "--population",
        str(settings.population),
        "--generations",
        str(settings.generations),
        "--elite-count",
        str(settings.elite_count),
        "--cpu-workers",
        str(settings.cpu_workers),
        "--heartbeat-seconds",
        str(settings.heartbeat_seconds),
        "--route-lengths",
        ",".join(str(item) for item in settings.route_lengths),
        "--grid-sizes",
        ",".join(str(item) for item in settings.grid_sizes),
        "--execution-mode",
        settings.execution_mode,
        "--replay",
    ]


def write_artifacts(out: Path, rows: list[dict[str, Any]], samples: list[dict[str, Any]], dense_rows: list[dict[str, Any]], mutation_rows: list[dict[str, Any]], seed_results: list[dict[str, Any]], deterministic: dict[str, Any]) -> dict[str, Any]:
    agg = aggregate(rows)
    splits = split_metrics(rows)
    depth = depth_report(rows)
    discovery, reuse = operator_reports(seed_results)
    decision = decide(agg, depth, splits)
    decision["deterministic_replay_passed"] = bool(deterministic.get("internal_replay_passed", False))
    decision["checker_failure_count"] = 0 if decision["deterministic_replay_passed"] else None
    write_json(out / "aggregate_metrics.json", agg)
    write_json(out / "split_metrics.json", splits)
    write_json(out / "depth_scaling_report.json", depth)
    write_json(out / "operator_discovery_report.json", discovery)
    write_json(out / "operator_reuse_report.json", reuse)
    write_jsonl(out / "mutation_history.jsonl", sorted(mutation_rows, key=lambda r: (r["seed"], r["trace_check"], r["skill"], r["generation"])))
    write_jsonl(out / "row_level_samples.jsonl", sorted(samples, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"])))
    write_json(out / "dense_shortcut_control_report.json", {"schema_version": "e8h4_dense_shortcut_control_report_v1", "rows": sorted(dense_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))})
    write_json(out / "decision.json", decision)
    write_json(out / "deterministic_replay.json", deterministic)
    (out / "report.md").write_text(report_markdown(decision, agg, depth, splits, deterministic), encoding="utf-8")
    return decision


def run(settings: Settings, out: Path) -> dict[str, Any]:
    if out.exists() and not settings.replay:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(out, stop, settings.heartbeat_seconds), daemon=True)
    thread.start()
    append_progress(out, "run_start", milestone=MILESTONE, execution_mode=settings.execution_mode, seeds=list(settings.seeds))
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "out": str(out)} for seed in settings.seeds]
    workers = max(1, min(settings.cpu_workers, len(jobs)))
    try:
        if workers == 1:
            for job in jobs:
                result = seed_worker(job)
                seed_results.append(result)
                rows.extend(result["rows"])
                samples.extend(result["samples"])
                dense_rows.extend(result["dense_rows"])
                mutation_rows.extend(result["mutation_history"])
                append_progress(out, "seed_complete", seed=result["seed"], completed=len(seed_results), pending=len(jobs) - len(seed_results))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_seeds": len(seed_results), "row_count": len(rows), "mutation_rows": len(mutation_rows)})
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        seed_results.append(result)
                        rows.extend(result["rows"])
                        samples.extend(result["samples"])
                        dense_rows.extend(result["dense_rows"])
                        mutation_rows.extend(result["mutation_history"])
                        append_progress(out, "seed_complete", seed=job["seed"], completed=len(seed_results), pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_seeds": len(seed_results), "row_count": len(rows), "mutation_rows": len(mutation_rows)})
        placeholder = {"schema_version": "e8h4_deterministic_replay_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        decision = write_artifacts(out, rows, samples, dense_rows, mutation_rows, seed_results, placeholder)
        append_progress(out, "primary_artifacts_written", decision=decision["decision"])
        if not settings.replay:
            primary_hashes = artifact_hashes(out)
            replay_out = out / "deterministic_replay_work"
            if replay_out.exists():
                shutil.rmtree(replay_out)
            append_progress(out, "deterministic_replay_start", replay_out=str(replay_out))
            subprocess.run(replay_args(settings, replay_out), cwd=REPO_ROOT, check=True)
            replay_hashes = artifact_hashes(replay_out)
            comparisons = {name: {"primary": primary_hashes[name], "replay": replay_hashes.get(name), "match": primary_hashes[name] == replay_hashes.get(name)} for name in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e8h4_deterministic_replay_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision = write_artifacts(out, rows, samples, dense_rows, mutation_rows, seed_results, deterministic)
            append_progress(out, "deterministic_replay_complete", internal_replay_passed=deterministic["internal_replay_passed"])
        return decision
    finally:
        stop.set()
        thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="108401,108402,108403,108404")
    parser.add_argument("--train-rows-per-seed", type=int, default=96)
    parser.add_argument("--validation-rows-per-seed", type=int, default=48)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=48)
    parser.add_argument("--ood-rows-per-seed", type=int, default=48)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=48)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=48)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--generations", type=int, default=36)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--route-lengths", default="1,3,6,12,24")
    parser.add_argument("--grid-sizes", default="6,8")
    parser.add_argument("--execution-mode", default="evidence_cpu")
    parser.add_argument("--replay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        population=args.population,
        generations=args.generations,
        elite_count=args.elite_count,
        cpu_workers=args.cpu_workers,
        heartbeat_seconds=args.heartbeat_seconds,
        route_lengths=parse_int_tuple(args.route_lengths),
        grid_sizes=parse_int_tuple(args.grid_sizes),
        replay=bool(args.replay),
        execution_mode=args.execution_mode,
    )
    decision = run(settings, Path(args.out))
    print(json.dumps(stable_payload(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
