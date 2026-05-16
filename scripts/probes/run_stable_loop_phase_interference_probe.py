from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_001_RESULT.md"

MAX_GRID = 16
INPUT_CHANNELS = 4
WALL, TARGET, SRC_REAL, SRC_IMAG = 0, 1, 2, 3
EVAL_STEPS = (4, 8, 16, 24, 32)
TRAIN_STEPS = (4, 8, 16, 24)
PHASE_EPS = 1e-4
READOUT_THRESHOLD = 0.35

ALL_ARMS = (
    "ORACLE_PHASE_WAVEFRONT_S",
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "LOCAL_MESSAGE_PASSING_GNN_PHASE",
    "HARD_WALL_ABC_PHASE_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOOP",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE",
)
ORACLE_ARMS = {"ORACLE_PHASE_WAVEFRONT_S"}
LOCAL_ARMS = {
    "LOCAL_MESSAGE_PASSING_GNN_PHASE",
    "HARD_WALL_ABC_PHASE_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOOP",
}
TRAINABLE_ARMS = set(ALL_ARMS) - ORACLE_ARMS - {"LOCAL_MESSAGE_PASSING_GNN_PHASE"}
FAIR_COMPARISON_ARMS = {
    "LOCAL_MESSAGE_PASSING_GNN_PHASE",
    "HARD_WALL_ABC_PHASE_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOOP",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE",
}


@dataclass(frozen=True)
class PhaseExample:
    case_id: str
    grid: list[list[list[float]]]
    grid_size: int
    target: tuple[int, int]
    phase_classes: int
    full_label: int
    labels_by_s: dict[int, int]
    family: str
    bucket: str
    collision_bucket: str
    tags: tuple[str, ...]
    contrast_group_id: str | None
    sources: tuple[tuple[int, int, int], ...]
    target_distance: int
    collision_distance: int


@dataclass(frozen=True)
class Config:
    arm: str
    width: int
    train_mode: str


@dataclass
class JobResult:
    arm: str
    width: int
    train_mode: str
    phase_classes: int
    seed: int
    metrics: dict[str, float]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str) -> list[int]:
    out: list[int] = []
    for part in parse_csv(value):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def resolve_jobs(value: str) -> int:
    if value.startswith("auto"):
        cpu = os.cpu_count() or 1
        frac = float(value.replace("auto", "")) / 100.0
        return max(1, math.floor(cpu * frac))
    return max(1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_INTERFERENCE_001 phase wavefront interference probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-classes", type=int, choices=[2, 4], default=2)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--jobs", default="4")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def unit_phase(k: int, phase_classes: int) -> complex:
    theta = 2.0 * math.pi * k / phase_classes
    return complex(math.cos(theta), math.sin(theta))


def phase_bucket(z: complex, phase_classes: int, eps: float = PHASE_EPS) -> int:
    if abs(z) < eps:
        return 0
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / phase_classes))) % phase_classes + 1


def distance_bucket(distance: int) -> str:
    if distance <= 0:
        return "none"
    if distance <= 4:
        return "1-4"
    if distance <= 8:
        return "5-8"
    if distance <= 16:
        return "9-16"
    return "17-24"


def collision_bucket(distance: int) -> str:
    if distance < 0:
        return "none"
    return distance_bucket(distance)


def shift_no_wrap_np(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(arr)
    src_y0 = max(0, -dy)
    src_y1 = arr.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = arr.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = arr.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = arr.shape[1] - max(0, -dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def oracle_phase_labels(grid: np.ndarray, phase_classes: int, steps_values: tuple[int, ...] = EVAL_STEPS) -> tuple[dict[int, int], int, list[dict[str, float]]]:
    wall = grid[WALL] > 0.5
    free = ~wall
    target_cells = np.argwhere(grid[TARGET] > 0.5)
    if len(target_cells) != 1:
        raise ValueError("phase example must have exactly one target")
    tr, tc = target_cells[0]
    source = grid[SRC_REAL].astype(np.float32) + 1j * grid[SRC_IMAG].astype(np.float32)
    frontier = source.copy()
    reached = source.copy()
    settled = np.abs(source) > PHASE_EPS
    labels: dict[int, int] = {0: phase_bucket(reached[tr, tc], phase_classes)}
    traces: list[dict[str, float]] = []
    max_step = max(steps_values)
    for step in range(1, max_step + 1):
        proposed = np.zeros_like(frontier)
        arrival_power = np.zeros(frontier.shape, dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = shift_no_wrap_np(frontier, dy, dx)
            proposed += shifted
            arrival_power += (np.abs(shifted) > PHASE_EPS).astype(np.float32)
        proposed *= free
        arrival = (arrival_power > 0) & free
        new_cells = arrival & (~settled)
        z = proposed
        new_frontier = np.zeros_like(frontier)
        keep = new_cells & (np.abs(z) >= PHASE_EPS)
        new_frontier[keep] = z[keep]
        reached[new_cells] = z[new_cells]
        settled[new_cells] = True
        frontier = new_frontier
        traces.append(
            {
                "step": float(step),
                "frontier_norm": float(np.abs(frontier).mean()),
                "reached_norm": float(np.abs(reached).mean()),
                "settled_fraction": float(settled.mean()),
            }
        )
        if step in steps_values:
            labels[step] = phase_bucket(reached[tr, tc], phase_classes)
    full_label = labels[max_step]
    return labels, full_label, traces


def blank_grid(n: int) -> np.ndarray:
    grid = np.zeros((INPUT_CHANNELS, MAX_GRID, MAX_GRID), dtype=np.float32)
    grid[WALL] = 1.0
    grid[WALL, :n, :n] = 1.0
    return grid


def carve_cell(grid: np.ndarray, cell: tuple[int, int]) -> None:
    r, c = cell
    grid[WALL, r, c] = 0.0


def carve_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for cell in path:
        carve_cell(grid, cell)


def manhattan_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    vertical_first = rng.random() < 0.5
    if vertical_first:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    return path


def place_source(grid: np.ndarray, cell: tuple[int, int], phase: int, phase_classes: int) -> None:
    carve_cell(grid, cell)
    z = unit_phase(phase, phase_classes)
    r, c = cell
    grid[SRC_REAL, r, c] += float(z.real)
    grid[SRC_IMAG, r, c] += float(z.imag)


def place_target(grid: np.ndarray, cell: tuple[int, int]) -> None:
    carve_cell(grid, cell)
    r, c = cell
    grid[TARGET, r, c] = 1.0


def finalize_example(
    case_id: str,
    grid: np.ndarray,
    n: int,
    target: tuple[int, int],
    phase_classes: int,
    family: str,
    tags: tuple[str, ...],
    contrast_group_id: str | None,
    sources: list[tuple[int, int, int]],
    target_distance: int,
    collision_distance: int,
) -> PhaseExample:
    labels_by_s, full_label, _ = oracle_phase_labels(grid, phase_classes)
    return PhaseExample(
        case_id=case_id,
        grid=grid.tolist(),
        grid_size=n,
        target=target,
        phase_classes=phase_classes,
        full_label=full_label,
        labels_by_s={int(k): int(v) for k, v in labels_by_s.items() if k in EVAL_STEPS},
        family=family,
        bucket=distance_bucket(target_distance),
        collision_bucket=collision_bucket(collision_distance),
        tags=tags,
        contrast_group_id=contrast_group_id,
        sources=tuple(sources),
        target_distance=target_distance,
        collision_distance=collision_distance,
    )


def choose_phase_pair(rng: random.Random, phase_classes: int) -> tuple[int, int]:
    a = rng.randrange(phase_classes)
    if phase_classes == 2:
        return a, 1 - a
    return a, (a + phase_classes // 2) % phase_classes


def make_single_source(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (rng.randint(3, n - 3), rng.randint(3, n - 3))
    distance = rng.choice([4, 6, 8, 12, 16, 20])
    distance = min(distance, target[0] + target[1], (n - 1 - target[0]) + (n - 1 - target[1]), 2 * (n - 2))
    source = (max(1, target[0] - min(distance, target[0] - 1)), target[1])
    if abs(source[0] - target[0]) < distance:
        rem = distance - abs(source[0] - target[0])
        source = (source[0], max(1, target[1] - rem))
    phase = rng.randrange(phase_classes)
    grid = blank_grid(n)
    path = manhattan_path(source, target, rng)
    carve_path(grid, path)
    place_source(grid, source, phase, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "single_source_reach", ("single_source", "phase"), None, [(source[0], source[1], phase)], len(path) - 1, -1)


def make_same_phase_reinforcement(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (n // 2, n // 2)
    d = rng.choice([4, 6, 8, 10])
    left = (target[0], max(1, target[1] - d))
    right = (target[0], min(n - 2, target[1] + d))
    phase = rng.randrange(phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(left, target, rng))
    carve_path(grid, manhattan_path(right, target, rng))
    place_source(grid, left, phase, phase_classes)
    place_source(grid, right, phase, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "same_phase_reinforcement", ("reinforcement", "phase"), None, [(left[0], left[1], phase), (right[0], right[1], phase)], d, d)


def make_opposite_collision(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (n // 2, n // 2)
    d = rng.choice([3, 5, 7])
    left = (target[0], target[1] - d)
    right = (target[0], target[1] + d)
    p0, p1 = choose_phase_pair(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(left, target, rng))
    carve_path(grid, manhattan_path(right, target, rng))
    place_source(grid, left, p0, phase_classes)
    place_source(grid, right, p1, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "opposite_phase_collision", ("cancellation", "opposite_collision"), None, [(left[0], left[1], p0), (right[0], right[1], p1)], d, d)


def make_branch_cancellation(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (n // 2, min(n - 2, n // 2 + 4))
    junction = (n // 2, n // 2)
    d = rng.choice([3, 4, 5])
    up = (junction[0] - d, junction[1])
    down = (junction[0] + d, junction[1])
    p0, p1 = choose_phase_pair(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(up, junction, rng))
    carve_path(grid, manhattan_path(down, junction, rng))
    carve_path(grid, manhattan_path(junction, target, rng))
    place_source(grid, up, p0, phase_classes)
    place_source(grid, down, p1, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "branch_cancellation", ("cancellation", "branch_cancellation"), None, [(up[0], up[1], p0), (down[0], down[1], p1)], d + abs(target[1] - junction[1]), d)


def make_decoy_phase(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (n // 2, n // 2)
    source = (1, 1)
    correct = rng.randrange(phase_classes)
    wrong = (correct + 1) % phase_classes
    decoy = (target[0], min(n - 2, target[1] + 1))
    grid = blank_grid(n)
    path = manhattan_path(source, target, rng)
    carve_path(grid, path)
    carve_cell(grid, decoy)
    place_source(grid, source, correct, phase_classes)
    place_source(grid, decoy, wrong, phase_classes)
    # Re-wall the gap around the decoy except its own cell so the target-local cue is misleading.
    for dr, dc in [(-1, 0), (1, 0), (0, 1)]:
        rr, cc = decoy[0] + dr, decoy[1] + dc
        if 0 <= rr < n and 0 <= cc < n and (rr, cc) != decoy:
            grid[WALL, rr, cc] = 1.0
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "decoy_phase", ("decoy_phase", "target_local_trap"), None, [(source[0], source[1], correct), (decoy[0], decoy[1], wrong)], len(path) - 1, -1)


def make_same_target_neighborhood_pair(case_id: str, rng: random.Random, phase_classes: int, n: int, phase: int, group: str) -> PhaseExample:
    target = (n // 2, n // 2)
    source = (1, target[1])
    grid = blank_grid(n)
    path = manhattan_path(source, target, rng)
    carve_path(grid, path)
    # Fixed local target room for both members of the pair.
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            rr, cc = target[0] + dr, target[1] + dc
            if 0 <= rr < n and 0 <= cc < n:
                carve_cell(grid, (rr, cc))
    place_source(grid, source, phase, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "same_target_neighborhood_phase_contrast", ("same_target_neighborhood", "phase_contrast"), group, [(source[0], source[1], phase)], len(path) - 1, -1)


def make_k4_competition(case_id: str, rng: random.Random, phase_classes: int, n: int) -> PhaseExample:
    target = (n // 2, n // 2)
    phase = rng.randrange(phase_classes)
    side = (phase + 1) % phase_classes
    d = rng.choice([4, 5, 6])
    cells = [(target[0], target[1] - d), (target[0], target[1] + d), (target[0] - d, target[1])]
    phases = [phase, phase, side]
    grid = blank_grid(n)
    for cell in cells:
        carve_path(grid, manhattan_path(cell, target, rng))
    for cell, p in zip(cells, phases):
        place_source(grid, cell, p, phase_classes)
    place_target(grid, target)
    return finalize_example(case_id, grid, n, target, phase_classes, "K4_phase_competition", ("phase_competition", "reinforcement"), None, [(r, c, p) for (r, c), p in zip(cells, phases)], d, d)


def build_dataset(seed: int, train_examples: int, eval_examples: int, phase_classes: int) -> tuple[list[PhaseExample], list[PhaseExample]]:
    rng = random.Random(seed)

    def make_rows(count: int, split: str) -> list[PhaseExample]:
        rows: list[PhaseExample] = []
        families = ["single", "reinforce", "opposite", "branch", "decoy", "contrast_a", "contrast_b"]
        if phase_classes == 4:
            families.append("k4")
        idx = 0
        while len(rows) < count:
            family = families[idx % len(families)]
            n = rng.choice([8, 12, 16]) if split == "eval" else rng.choice([8, 12])
            case_id = f"{split}_{idx:06d}_{family}"
            if family == "single":
                row = make_single_source(case_id, rng, phase_classes, n)
            elif family == "reinforce":
                row = make_same_phase_reinforcement(case_id, rng, phase_classes, n)
            elif family == "opposite":
                row = make_opposite_collision(case_id, rng, phase_classes, n)
            elif family == "branch":
                row = make_branch_cancellation(case_id, rng, phase_classes, n)
            elif family == "decoy":
                row = make_decoy_phase(case_id, rng, phase_classes, n)
            elif family == "contrast_a":
                group = f"{split}_contrast_{idx // len(families):06d}"
                row = make_same_target_neighborhood_pair(case_id, rng, phase_classes, n, 0, group)
            elif family == "contrast_b":
                group = f"{split}_contrast_{idx // len(families):06d}"
                row = make_same_target_neighborhood_pair(case_id, rng, phase_classes, n, 1 % phase_classes, group)
            else:
                row = make_k4_competition(case_id, rng, phase_classes, n)
            rows.append(row)
            idx += 1
        return rows

    return make_rows(train_examples, "train"), make_rows(eval_examples, "eval")


def target_cell(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = torch.arange(h.shape[0], device=h.device)
    r = target[:, 0].long()
    c = target[:, 1].long()
    return h[b, :, r, c]


def phase_vectors_tensor(phase_classes: int, device: torch.device) -> torch.Tensor:
    vals = []
    for k in range(phase_classes):
        z = unit_phase(k, phase_classes)
        vals.append([float(z.real), float(z.imag)])
    return torch.tensor(vals, dtype=torch.float32, device=device)


def fixed_phase_logits(vec: torch.Tensor, phase_classes: int) -> torch.Tensor:
    phases = phase_vectors_tensor(phase_classes, vec.device)
    mag = vec.pow(2).sum(dim=1).add(1e-8).sqrt()
    none = (READOUT_THRESHOLD - mag) * 12.0
    phase_logits = (vec @ phases.T) * 12.0
    return torch.cat([none.unsqueeze(1), phase_logits], dim=1)


def examples_to_tensors(rows: list[PhaseExample], device: torch.device) -> dict[str, Any]:
    x = torch.tensor(np.array([row.grid for row in rows], dtype=np.float32), device=device)
    target = torch.tensor(np.array([row.target for row in rows], dtype=np.int64), device=device)
    full_y = torch.tensor([row.full_label for row in rows], dtype=torch.long, device=device)
    labels_by_s = {
        s: torch.tensor([row.labels_by_s[s] for row in rows], dtype=torch.long, device=device)
        for s in EVAL_STEPS
    }
    return {"x": x, "target": target, "full_y": full_y, "labels_by_s": labels_by_s}


class SummaryDirectHead(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled), trace_stats([h], x)


class TargetMarkerOnly(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1), {}


class UntiedLocalCNNPhase(nn.Module):
    def __init__(self, width: int, max_steps: int, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes
        self.inp = nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(width + INPUT_CHANNELS, width, 3, padding=1) for _ in range(max_steps)])
        self.vec_head = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.inp(x))
        traces = [h]
        for layer in self.layers[:steps]:
            h = torch.tanh(layer(torch.cat([h, x], dim=1)))
            traces.append(h)
        vec_map = self.vec_head(h) * (1.0 - x[:, WALL : WALL + 1])
        vec = target_cell(vec_map, target)
        return fixed_phase_logits(vec, self.phase_classes), trace_stats(traces, x, reached=vec_map.pow(2).sum(dim=1, keepdim=True).sqrt())


def shift_no_wrap_torch(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    src_y0 = max(0, -dy)
    src_y1 = x.shape[-2] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = x.shape[-1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = x.shape[-2] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = x.shape[-1] - max(0, -dx)
    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
    return out


def neighbor_sum_complex(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_r = torch.zeros_like(real)
    sum_i = torch.zeros_like(imag)
    arrival = torch.zeros_like(real)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = shift_no_wrap_torch(real, dy, dx)
        ii = shift_no_wrap_torch(imag, dy, dx)
        sum_r = sum_r + rr
        sum_i = sum_i + ii
        arrival = arrival + ((rr.square() + ii.square()).sqrt() > PHASE_EPS).float()
    return sum_r, sum_i, arrival


def phase_state_stats(
    x: torch.Tensor,
    frontier_traces: list[torch.Tensor],
    reached_traces: list[torch.Tensor],
    pre_wall: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    wall = x[:, WALL : WALL + 1] > 0.5
    frontier = frontier_traces[-1]
    reached = reached_traces[-1]
    if len(reached_traces) > 1:
        deltas = [(reached_traces[i] - reached_traces[i - 1]).pow(2).mean().sqrt() for i in range(1, len(reached_traces))]
        final_delta = torch.stack(deltas).mean()
    else:
        final_delta = torch.zeros((), device=x.device)
    frontier_norm = frontier.pow(2).sum(dim=1, keepdim=True).sqrt()
    reached_norm = reached.pow(2).sum(dim=1, keepdim=True).sqrt()
    return {
        "final_state_delta": final_delta,
        "phase_vector_norm_by_step": torch.stack([t.pow(2).sum(dim=1).sqrt().mean() for t in reached_traces]).mean(),
        "phase_saturation_rate": (reached.abs() > 4.0).float().mean(),
        "pre_mask_wall_pressure": torch.stack(pre_wall).mean() if pre_wall else torch.zeros((), device=x.device),
        "post_mask_wall_leak": ((reached_norm > 0.5) & wall).float().mean(),
        "frontier_norm": frontier_norm.mean(),
    }


class DeterministicPhaseMessagePassing(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        free = 1.0 - x[:, WALL : WALL + 1]
        frontier_r = x[:, SRC_REAL : SRC_REAL + 1] * free
        frontier_i = x[:, SRC_IMAG : SRC_IMAG + 1] * free
        reached_r = frontier_r.clone()
        reached_i = frontier_i.clone()
        settled = ((reached_r.square() + reached_i.square()).sqrt() > PHASE_EPS).float()
        frontier_traces = [torch.cat([frontier_r, frontier_i], dim=1)]
        reached_traces = [torch.cat([reached_r, reached_i], dim=1)]
        pre_wall = []
        for _ in range(steps):
            prop_r, prop_i, arrival = neighbor_sum_complex(frontier_r, frontier_i)
            prop_r = prop_r * free
            prop_i = prop_i * free
            new_cells = (arrival > 0).float() * (1.0 - settled) * free
            mag = (prop_r.square() + prop_i.square()).sqrt()
            keep = (mag >= PHASE_EPS).float()
            new_frontier_r = prop_r * new_cells * keep
            new_frontier_i = prop_i * new_cells * keep
            pre_wall.append(((new_frontier_r.abs() + new_frontier_i.abs()) * x[:, WALL : WALL + 1]).mean())
            reached_r = torch.where(new_cells.bool(), prop_r, reached_r)
            reached_i = torch.where(new_cells.bool(), prop_i, reached_i)
            settled = torch.maximum(settled, new_cells)
            frontier_r, frontier_i = new_frontier_r, new_frontier_i
            frontier_traces.append(torch.cat([frontier_r, frontier_i], dim=1))
            reached_traces.append(torch.cat([reached_r, reached_i], dim=1))
        vec = target_cell(torch.cat([reached_r, reached_i], dim=1), target)
        return fixed_phase_logits(vec, self.phase_classes), phase_state_stats(x, frontier_traces, reached_traces, pre_wall)


class HardWallPhaseLoop(nn.Module):
    def __init__(self, arm: str, width: int, phase_classes: int):
        super().__init__()
        self.arm = arm
        self.phase_classes = phase_classes
        self.gain_logit = nn.Parameter(torch.tensor(4.0))
        self.phase_angle = nn.Parameter(torch.tensor(0.0))
        self.latent_in = nn.Conv2d(INPUT_CHANNELS + 4, width, 1)
        self.latent_update = nn.Conv2d(width + INPUT_CHANNELS + 4, width, 3, padding=1)
        self.local_gate = nn.Conv2d(width + INPUT_CHANNELS + 4, 1, 1)
        self.phase_refine = nn.Conv2d(width, 2, 1)
        self.decay = nn.Parameter(torch.tensor(0.8))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        free = 1.0 - x[:, WALL : WALL + 1]
        frontier_r = x[:, SRC_REAL : SRC_REAL + 1] * free
        frontier_i = x[:, SRC_IMAG : SRC_IMAG + 1] * free
        reached_r = frontier_r.clone()
        reached_i = frontier_i.clone()
        settled = ((reached_r.square() + reached_i.square()).sqrt() > PHASE_EPS).float()
        h = torch.tanh(self.latent_in(torch.cat([x, reached_r, reached_i, frontier_r, frontier_i], dim=1))) * free
        frontier_traces = [torch.cat([frontier_r, frontier_i], dim=1)]
        reached_traces = [torch.cat([reached_r, reached_i], dim=1)]
        pre_wall = []
        for _ in range(steps):
            source = torch.cat([h, x, reached_r, reached_i, frontier_r, frontier_i], dim=1)
            prop_r, prop_i, arrival = neighbor_sum_complex(frontier_r, frontier_i)
            # The hard-wall phase loop is meant to test repeated local propagation,
            # not whether a randomly initialized gate can attenuate every long path
            # below the fixed readout threshold before learning starts.
            gate = 0.998 + 0.002 * torch.sigmoid(self.local_gate(source))
            if self.arm == "HARD_WALL_PRISMION_PHASE_LOOP":
                # Keep the phase primitive close to identity unless learning finds a
                # small useful correction; unconstrained per-step rotation can
                # destroy long-path phase identity and turns K2 into an optimizer
                # instability test instead of a phase-interference test.
                angle = torch.tanh(self.phase_angle) * 0.02
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                rot_r = prop_r * cos_a - prop_i * sin_a
                rot_i = prop_r * sin_a + prop_i * cos_a
                prop_r = rot_r
                prop_i = rot_i
            else:
                prop_r = prop_r
                prop_i = prop_i
            gain = 0.995 + 0.005 * torch.sigmoid(self.gain_logit)
            prop_r = prop_r * gate * gain * free
            prop_i = prop_i * gate * gain * free
            new_cells = (arrival > 0).float() * (1.0 - settled) * free
            mag = (prop_r.square() + prop_i.square()).sqrt()
            keep = (mag >= PHASE_EPS).float()
            new_frontier_r = prop_r * new_cells * keep
            new_frontier_i = prop_i * new_cells * keep
            pre_wall.append(((new_frontier_r.abs() + new_frontier_i.abs()) * x[:, WALL : WALL + 1]).mean())
            reached_r = torch.where(new_cells.bool(), prop_r, reached_r)
            reached_i = torch.where(new_cells.bool(), prop_i, reached_i)
            settled = torch.maximum(settled, new_cells)
            latent_next = torch.tanh(self.latent_update(source))
            h = torch.tanh(torch.sigmoid(self.decay) * h + latent_next) * free
            frontier_r, frontier_i = new_frontier_r, new_frontier_i
            frontier_traces.append(torch.cat([frontier_r, frontier_i], dim=1))
            reached_traces.append(torch.cat([reached_r, reached_i], dim=1))
        vec = target_cell(torch.cat([reached_r, reached_i], dim=1), target)
        return fixed_phase_logits(vec, self.phase_classes), phase_state_stats(x, frontier_traces, reached_traces, pre_wall)


def trace_stats(traces: list[torch.Tensor], x: torch.Tensor, reached: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if len(traces) > 1:
        final_delta = torch.stack([(traces[i] - traces[i - 1]).pow(2).mean().sqrt() for i in range(1, len(traces))]).mean()
    else:
        final_delta = torch.zeros((), device=x.device)
    final = traces[-1]
    stats = {
        "final_state_delta": final_delta,
        "phase_vector_norm_by_step": torch.stack([t.pow(2).mean().sqrt() for t in traces]).mean(),
        "phase_saturation_rate": (final.abs() > 4.0).float().mean(),
        "pre_mask_wall_pressure": torch.zeros((), device=x.device),
    }
    if reached is None:
        stats["post_mask_wall_leak"] = torch.zeros((), device=x.device)
    else:
        wall = x[:, WALL : WALL + 1] > 0.5
        stats["post_mask_wall_leak"] = (((reached > 0.5) & wall).float().mean())
    return stats


def make_model(config: Config, phase_classes: int) -> nn.Module | None:
    if config.arm == "SUMMARY_DIRECT_HEAD":
        return SummaryDirectHead(config.width, phase_classes)
    if config.arm == "TARGET_MARKER_ONLY":
        return TargetMarkerOnly(phase_classes)
    if config.arm == "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE":
        return UntiedLocalCNNPhase(config.width, max(EVAL_STEPS), phase_classes)
    if config.arm == "LOCAL_MESSAGE_PASSING_GNN_PHASE":
        return DeterministicPhaseMessagePassing(phase_classes)
    if config.arm in {"HARD_WALL_ABC_PHASE_LOOP", "HARD_WALL_PRISMION_PHASE_LOOP"}:
        return HardWallPhaseLoop(config.arm, config.width, phase_classes)
    return None


def iter_batches(n: int, batch_size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [np.array(idx[i : i + batch_size], dtype=np.int64) for i in range(0, n, batch_size)]


def forward_model(model: nn.Module, config: Config, batch: dict[str, Any], indices: np.ndarray, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = batch["x"][indices]
    target = batch["target"][indices]
    return model(x, target, steps=steps)


def train_model(config: Config, rows: list[PhaseExample], args: argparse.Namespace, device: torch.device, progress_path: Path, seed: int) -> nn.Module | None:
    model = make_model(config, args.phase_classes)
    if model is None:
        return None
    model.to(device)
    if config.arm not in TRAINABLE_ARMS:
        return model
    batch = examples_to_tensors(rows, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 404)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for indices in iter_batches(len(rows), args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            steps = rng.choice(TRAIN_STEPS)
            logits, _ = forward_model(model, config, batch, indices, steps)
            target = batch["labels_by_s"][steps][indices]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), **asdict(config), "seed": seed})
    return model


def predictions_for(model: nn.Module | None, config: Config, rows: list[PhaseExample], args: argparse.Namespace, device: torch.device, steps: int) -> tuple[np.ndarray, dict[str, float]]:
    if config.arm == "ORACLE_PHASE_WAVEFRONT_S":
        return np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64), {}
    assert model is not None
    model.eval()
    batch = examples_to_tensors(rows, device)
    preds: list[int] = []
    stat_rows: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            indices = np.arange(start, min(len(rows), start + args.batch_size))
            logits, stats = forward_model(model, config, batch, indices, steps)
            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().cpu().numpy().tolist())
            for key, value in stats.items():
                stat_rows[key].append(float(value.float().mean().detach().cpu()))
    return np.array(preds, dtype=np.int64), {k: float(np.mean(v)) for k, v in stat_rows.items()}


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean()) if len(gold) else math.nan


def filter_accuracy(rows: list[PhaseExample], correct: np.ndarray, predicate) -> float:
    vals = [bool(ok) for row, ok in zip(rows, correct) if predicate(row)]
    return float(np.mean(vals)) if vals else math.nan


def pair_accuracy(rows: list[PhaseExample], correct: np.ndarray) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(rows, correct):
        if row.contrast_group_id:
            groups[row.contrast_group_id].append(bool(ok))
    if not groups:
        return math.nan
    return float(np.mean([all(vals) for vals in groups.values()]))


def class_balance(rows: list[PhaseExample], steps: int) -> dict[str, float]:
    labels = [row.labels_by_s[steps] for row in rows]
    counts = Counter(labels)
    n = max(1, len(labels))
    return {
        "majority_baseline": max(counts.values()) / n if counts else math.nan,
        "random_baseline": 1.0 / (rows[0].phase_classes + 1) if rows else math.nan,
        "none_rate": counts.get(0, 0) / n,
        "num_classes_seen": float(len(counts)),
    }


def evaluate(config: Config, model: nn.Module | None, train_rows: list[PhaseExample], eval_rows: list[PhaseExample], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    metrics: dict[str, float] = {}
    acc_by_s = []
    false_future = []
    stats_by_key: dict[str, list[float]] = defaultdict(list)
    for steps in EVAL_STEPS:
        pred, stats = predictions_for(model, config, eval_rows, args, device, steps)
        gold = np.array([row.labels_by_s[steps] for row in eval_rows], dtype=np.int64)
        full_gold = np.array([row.full_label for row in eval_rows], dtype=np.int64)
        correct = pred == gold
        acc = accuracy(pred, gold)
        acc_by_s.append(acc)
        metrics[f"target_phase_accuracy_s{steps}"] = acc
        metrics[f"truncated_phase_accuracy_s{steps}"] = acc
        metrics[f"none_vs_phase_accuracy_s{steps}"] = accuracy((pred > 0).astype(np.int64), (gold > 0).astype(np.int64))
        phase_mask = gold > 0
        metrics[f"phase_bucket_accuracy_s{steps}"] = accuracy(pred[phase_mask], gold[phase_mask]) if phase_mask.any() else math.nan
        metrics[f"false_none_rate_s{steps}"] = float(((pred == 0) & (gold > 0)).mean())
        metrics[f"false_phase_rate_s{steps}"] = float(((pred > 0) & (gold == 0)).mean())
        metrics[f"wrong_phase_rate_s{steps}"] = float(((pred > 0) & (gold > 0) & (pred != gold)).mean())
        future_mask = np.array([row.full_label != row.labels_by_s[steps] for row in eval_rows], dtype=bool)
        metrics[f"false_future_collision_rate_s{steps}"] = float((pred[future_mask] == full_gold[future_mask]).mean()) if future_mask.any() else 0.0
        false_future.append(metrics[f"false_future_collision_rate_s{steps}"])
        metrics[f"cancellation_case_accuracy_s{steps}"] = filter_accuracy(eval_rows, correct, lambda row: "cancellation" in row.tags)
        metrics[f"reinforcement_case_accuracy_s{steps}"] = filter_accuracy(eval_rows, correct, lambda row: "reinforcement" in row.tags)
        metrics[f"opposite_collision_accuracy_s{steps}"] = filter_accuracy(eval_rows, correct, lambda row: "opposite_collision" in row.tags)
        metrics[f"decoy_phase_resistance_s{steps}"] = filter_accuracy(eval_rows, correct, lambda row: "decoy_phase" in row.tags)
        metrics[f"same_target_neighborhood_pair_accuracy_s{steps}"] = pair_accuracy(eval_rows, correct)
        metrics[f"false_positive_phase_after_cancel_s{steps}"] = float(((pred > 0) & (gold == 0) & np.array(["cancellation" in row.tags for row in eval_rows], dtype=bool)).mean())
        for key, value in stats.items():
            stats_by_key[key].append(value)
    metrics["target_phase_accuracy"] = float(np.mean(acc_by_s))
    metrics["truncated_phase_accuracy_by_S"] = float(np.mean(acc_by_s))
    metrics["same_weights_s_curve_accuracy"] = float(np.mean(acc_by_s))
    metrics["full_phase_accuracy_at_large_S"] = metrics["target_phase_accuracy_s32"]
    metrics["propagation_curve_score"] = metrics["same_weights_s_curve_accuracy"]
    metrics["s_matches_phase_collision_distance_score"] = float(np.mean([metrics[f"target_phase_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["false_future_collision_rate"] = float(np.mean(false_future))
    metrics["none_vs_phase_accuracy"] = float(np.mean([metrics[f"none_vs_phase_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["phase_bucket_accuracy"] = float(np.nanmean([metrics[f"phase_bucket_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["cancellation_case_accuracy"] = float(np.nanmean([metrics[f"cancellation_case_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["reinforcement_case_accuracy"] = float(np.nanmean([metrics[f"reinforcement_case_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["opposite_collision_accuracy"] = float(np.nanmean([metrics[f"opposite_collision_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["decoy_phase_resistance"] = float(np.nanmean([metrics[f"decoy_phase_resistance_s{s}"] for s in EVAL_STEPS]))
    metrics["same_target_neighborhood_pair_accuracy"] = float(np.nanmean([metrics[f"same_target_neighborhood_pair_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["false_none_rate"] = float(np.mean([metrics[f"false_none_rate_s{s}"] for s in EVAL_STEPS]))
    metrics["false_phase_rate"] = float(np.mean([metrics[f"false_phase_rate_s{s}"] for s in EVAL_STEPS]))
    metrics["wrong_phase_rate"] = float(np.mean([metrics[f"wrong_phase_rate_s{s}"] for s in EVAL_STEPS]))
    metrics["false_positive_phase_after_cancel"] = float(np.mean([metrics[f"false_positive_phase_after_cancel_s{s}"] for s in EVAL_STEPS]))
    metrics["unreachable_false_reach_all_S"] = metrics["false_phase_rate"]
    for key, values in stats_by_key.items():
        metrics[key] = float(np.mean(values))
    metrics.setdefault("post_mask_wall_leak", 0.0)
    metrics.setdefault("pre_mask_wall_pressure", 0.0)
    metrics["overrun_stability"] = float(np.mean([1.0 if acc_by_s[i + 1] >= acc_by_s[i] - 0.03 else 0.0 for i in range(len(acc_by_s) - 1)]))
    metrics["noise_recovery_accuracy"] = math.nan
    metrics["parameter_count"] = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    bal = class_balance(eval_rows, 24)
    metrics.update({f"class_balance_{k}": float(v) for k, v in bal.items()})
    return metrics


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path) -> JobResult:
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples, args.phase_classes)
    progress_path = progress_root / f"{config.arm}_w{config.width}_{config.train_mode}_seed{seed}.jsonl"
    append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
    model = train_model(config, train_rows, args, device, progress_path, seed)
    metrics = evaluate(config, model, train_rows, eval_rows, args, device)
    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "metrics": metrics})
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return JobResult(config.arm, config.width, config.train_mode, args.phase_classes, seed, metrics)


def build_queue(args: argparse.Namespace) -> list[Config]:
    arms = parse_csv(args.arms) if args.arms else list(ALL_ARMS)
    configs: list[Config] = []
    for arm in arms:
        width = 0 if arm in ORACLE_ARMS or arm == "TARGET_MARKER_ONLY" or arm == "LOCAL_MESSAGE_PASSING_GNN_PHASE" else args.width
        configs.append(Config(arm=arm, width=width, train_mode="SAME_WEIGHTS_S_CURVE"))
    return configs


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        key = f"{result.arm}|w{result.width}|{result.train_mode}"
        groups[key].append(result)
    agg: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        metric_keys = sorted({mk for row in rows for mk in row.metrics})
        metric_mean = {}
        metric_std = {}
        for mk in metric_keys:
            vals = [row.metrics[mk] for row in rows if mk in row.metrics and not math.isnan(row.metrics[mk])]
            metric_mean[mk] = float(np.mean(vals)) if vals else math.nan
            metric_std[mk] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg[key] = {
            "arm": rows[0].arm,
            "width": rows[0].width,
            "train_mode": rows[0].train_mode,
            "phase_classes": rows[0].phase_classes,
            "seeds": [row.seed for row in rows],
            "num_seeds": len(rows),
            "metric_mean": metric_mean,
            "metric_std": metric_std,
        }
    return agg


def m(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    return float(row["metric_mean"].get(key, default))


def best(agg: dict[str, dict[str, Any]], arms: set[str], metric: str = "target_phase_accuracy") -> dict[str, Any] | None:
    rows = [row for row in agg.values() if row["arm"] in arms]
    if not rows:
        return None
    return max(rows, key=lambda row: m(row, metric, -1.0))


def paired_deltas(results: list[JobResult], left_arm: str, right_arm: str, metric: str) -> dict[str, Any]:
    left = {row.seed: row.metrics.get(metric, math.nan) for row in results if row.arm == left_arm}
    right = {row.seed: row.metrics.get(metric, math.nan) for row in results if row.arm == right_arm}
    seeds = sorted(set(left) & set(right))
    deltas = [{"seed": seed, "delta": float(left[seed] - right[seed])} for seed in seeds if not math.isnan(left[seed]) and not math.isnan(right[seed])]
    vals = [row["delta"] for row in deltas]
    return {
        "left": left_arm,
        "right": right_arm,
        "metric": metric,
        "deltas": deltas,
        "mean_delta": float(np.mean(vals)) if vals else math.nan,
        "std_delta": float(np.std(vals)) if len(vals) > 1 else 0.0,
        "min_delta": float(np.min(vals)) if vals else math.nan,
        "max_delta": float(np.max(vals)) if vals else math.nan,
        "positive_seeds": int(sum(1 for v in vals if v > 0)),
        "paired_seeds": len(vals),
    }


def verdicts(agg: dict[str, dict[str, Any]], results: list[JobResult]) -> list[str]:
    labels: list[str] = []
    oracle = best(agg, {"ORACLE_PHASE_WAVEFRONT_S"})
    summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    marker = best(agg, {"TARGET_MARKER_ONLY"})
    prism = best(agg, {"HARD_WALL_PRISMION_PHASE_LOOP"})
    abc = best(agg, {"HARD_WALL_ABC_PHASE_LOOP"})
    gnn = best(agg, {"LOCAL_MESSAGE_PASSING_GNN_PHASE"})
    untied = best(agg, {"UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE"})
    if oracle and m(oracle, "target_phase_accuracy", 0.0) >= 0.99:
        task_valid = True
    else:
        task_valid = False
    if summary and m(summary, "target_phase_accuracy", 0.0) > m(summary, "class_balance_majority_baseline", 0.0) + 0.10:
        labels.append("SUMMARY_OR_TARGET_SHORTCUT_RETURNS")
    if marker and m(marker, "target_phase_accuracy", 0.0) > m(marker, "class_balance_majority_baseline", 0.0) + 0.10:
        labels.append("SUMMARY_OR_TARGET_SHORTCUT_RETURNS")
    if prism and m(prism, "post_mask_wall_leak", 0.0) > 0.02:
        labels.append("PHASE_LOOP_UNSTABLE")
    if prism and abc and gnn:
        if m(abc, "target_phase_accuracy", 0.0) >= m(prism, "target_phase_accuracy", 0.0) - 0.01 or m(gnn, "target_phase_accuracy", 0.0) >= m(prism, "target_phase_accuracy", 0.0) - 0.01:
            labels.append("CANONICAL_MESSAGE_PASSING_SUFFICIENT")
    if abc and m(abc, "target_phase_accuracy", 0.0) >= 0.985:
        labels.append("ABC_CEILING_TASK_TOO_EASY")
    if prism and abc and gnn:
        d_abc = paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "HARD_WALL_ABC_PHASE_LOOP", "target_phase_accuracy")
        d_gnn = paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE", "target_phase_accuracy")
        if (
            d_abc["mean_delta"] > 0.015
            and d_gnn["mean_delta"] > 0.015
            and m(prism, "pre_mask_wall_pressure", 1.0) <= m(abc, "pre_mask_wall_pressure", 1.0) + 0.002
            and m(prism, "false_none_rate", 1.0) <= m(abc, "false_none_rate", 1.0) + 0.01
            and m(prism, "false_phase_rate", 1.0) <= m(abc, "false_phase_rate", 1.0) + 0.01
        ):
            labels.append("PRISMION_PHASE_INTERFERENCE_POSITIVE")
    if untied and prism and m(untied, "target_phase_accuracy", 0.0) >= m(prism, "target_phase_accuracy", 0.0) - 0.01:
        labels.append("UNTIED_LOCAL_SUFFICIENT")
    if task_valid and prism and abc and gnn and m(prism, "target_phase_accuracy", 0.0) >= 0.70:
        labels.append("PHASE_INTERFERENCE_TASK_VALID")
    if task_valid and all(m(row, "target_phase_accuracy", 0.0) < 0.45 for row in [prism, abc, gnn, untied] if row):
        labels.append("TASK_TOO_HARD")
    return sorted(set(labels or ["PHASE_INTERFERENCE_PARTIAL"]))


def build_side_outputs(out_dir: Path, agg: dict[str, dict[str, Any]], results: list[JobResult], eval_rows: list[PhaseExample]) -> None:
    phase_rows = []
    collision_rows = []
    convergence_rows = []
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        base = {"arm": row["arm"], "width": row["width"], "train_mode": row["train_mode"], "phase_classes": row["phase_classes"]}
        phase_rows.append(
            {
                **base,
                "target_phase_accuracy": m(row, "target_phase_accuracy"),
                "none_vs_phase_accuracy": m(row, "none_vs_phase_accuracy"),
                "phase_bucket_accuracy": m(row, "phase_bucket_accuracy"),
                "same_weights_s_curve_accuracy": m(row, "same_weights_s_curve_accuracy"),
                "full_phase_accuracy_at_large_S": m(row, "full_phase_accuracy_at_large_S"),
            }
        )
        collision_rows.append(
            {
                **base,
                "cancellation_case_accuracy": m(row, "cancellation_case_accuracy"),
                "reinforcement_case_accuracy": m(row, "reinforcement_case_accuracy"),
                "opposite_collision_accuracy": m(row, "opposite_collision_accuracy"),
                "same_target_neighborhood_pair_accuracy": m(row, "same_target_neighborhood_pair_accuracy"),
                "decoy_phase_resistance": m(row, "decoy_phase_resistance"),
                "false_none_rate": m(row, "false_none_rate"),
                "false_phase_rate": m(row, "false_phase_rate"),
                "wrong_phase_rate": m(row, "wrong_phase_rate"),
                "false_positive_phase_after_cancel": m(row, "false_positive_phase_after_cancel"),
            }
        )
        convergence_rows.append(
            {
                **base,
                "overrun_stability": m(row, "overrun_stability"),
                "final_state_delta": m(row, "final_state_delta"),
                "phase_vector_norm_by_step": m(row, "phase_vector_norm_by_step"),
                "phase_saturation_rate": m(row, "phase_saturation_rate"),
                "pre_mask_wall_pressure": m(row, "pre_mask_wall_pressure"),
                "post_mask_wall_leak": m(row, "post_mask_wall_leak"),
            }
        )
    write_jsonl(out_dir / "phase_bucket_metrics.jsonl", phase_rows)
    write_jsonl(out_dir / "collision_metrics.jsonl", collision_rows)
    write_jsonl(out_dir / "convergence_curves.jsonl", convergence_rows)
    write_jsonl(out_dir / "phase_cases.jsonl", [asdict(row) for row in eval_rows[:128]])


def write_outputs(out_dir: Path, args: argparse.Namespace, results: list[JobResult], status: str, jobs: int, eval_rows: list[PhaseExample]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdicts(agg, results) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    build_side_outputs(out_dir, agg, results, eval_rows)
    delta_rows = [
        paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "HARD_WALL_ABC_PHASE_LOOP", "target_phase_accuracy"),
        paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE", "target_phase_accuracy"),
        paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE", "target_phase_accuracy"),
        paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "HARD_WALL_ABC_PHASE_LOOP", "false_phase_rate"),
        paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "HARD_WALL_ABC_PHASE_LOOP", "pre_mask_wall_pressure"),
    ]
    summary = {
        "status": status,
        "verdict": labels,
        "completed_jobs": len(results),
        "config": {
            "phase_classes": args.phase_classes,
            "seeds": args.seeds,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "width": args.width,
            "jobs": jobs,
            "device": args.device,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
        },
        "paired_deltas": delta_rows,
        "aggregate": agg,
    }
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# STABLE_LOOP_PHASE_INTERFERENCE_001 Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Phase classes: `{args.phase_classes}`",
        "",
        "| Arm | PhaseAcc | NoneVsPhase | PhaseBucket | Cancel | Reinforce | Opposite | Pair | Decoy | FalseNone | FalsePhase | WallPre | WallPost | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.4f}` | `{:.0f}` |".format(
                row["arm"],
                m(row, "target_phase_accuracy"),
                m(row, "none_vs_phase_accuracy"),
                m(row, "phase_bucket_accuracy"),
                m(row, "cancellation_case_accuracy"),
                m(row, "reinforcement_case_accuracy"),
                m(row, "opposite_collision_accuracy"),
                m(row, "same_target_neighborhood_pair_accuracy"),
                m(row, "decoy_phase_resistance"),
                m(row, "false_none_rate"),
                m(row, "false_phase_rate"),
                m(row, "pre_mask_wall_pressure"),
                m(row, "post_mask_wall_leak"),
                m(row, "parameter_count"),
            )
        )
    lines.extend(["", "## Paired Deltas", ""])
    for delta in delta_rows:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}` on `{delta['metric']}`: mean `{delta['mean_delta']:.4f}`, std `{delta['std_delta']:.4f}`, min `{delta['min_delta']:.4f}`, max `{delta['max_delta']:.4f}`, positive `{delta['positive_seeds']}/{delta['paired_seeds']}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    survivors = [row for row in agg.values() if m(row, "target_phase_accuracy", 0.0) >= max([m(r, "target_phase_accuracy", 0.0) for r in agg.values()] or [0.0]) - 0.03]
    write_json(out_dir / "survivor_configs.json", [{"arm": row["arm"], "width": row["width"], "train_mode": row["train_mode"]} for row in survivors])
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int, results: list[JobResult]) -> None:
    delta_abc = paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "HARD_WALL_ABC_PHASE_LOOP", "target_phase_accuracy")
    delta_gnn = paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE", "target_phase_accuracy")
    delta_untied = paired_deltas(results, "HARD_WALL_PRISMION_PHASE_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE", "target_phase_accuracy")
    lines = [
        "# STABLE_LOOP_PHASE_INTERFERENCE_001 Result",
        "",
        "## Latest Run",
        "",
        "```text",
        f"phase_classes={args.phase_classes}",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"width={args.width}",
        f"jobs={jobs}",
        f"device={args.device}",
        f"completed_jobs={completed}",
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "| Arm | PhaseAcc | Cancel | Reinforce | Opposite | Pair | Decoy | FalseNone | FalsePhase | WallPre | WallPost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.4f}` |".format(
                row["arm"],
                m(row, "target_phase_accuracy"),
                m(row, "cancellation_case_accuracy"),
                m(row, "reinforcement_case_accuracy"),
                m(row, "opposite_collision_accuracy"),
                m(row, "same_target_neighborhood_pair_accuracy"),
                m(row, "decoy_phase_resistance"),
                m(row, "false_none_rate"),
                m(row, "false_phase_rate"),
                m(row, "pre_mask_wall_pressure"),
                m(row, "post_mask_wall_leak"),
            )
        )
    lines.extend(
        [
            "",
            "## Prismion Matched Deltas",
            "",
            f"- Prismion minus ABC phase accuracy: mean `{delta_abc['mean_delta']:.4f}`, std `{delta_abc['std_delta']:.4f}`, min `{delta_abc['min_delta']:.4f}`, max `{delta_abc['max_delta']:.4f}`, positive `{delta_abc['positive_seeds']}/{delta_abc['paired_seeds']}`.",
            f"- Prismion minus GNN phase accuracy: mean `{delta_gnn['mean_delta']:.4f}`, std `{delta_gnn['std_delta']:.4f}`, min `{delta_gnn['min_delta']:.4f}`, max `{delta_gnn['max_delta']:.4f}`, positive `{delta_gnn['positive_seeds']}/{delta_gnn['paired_seeds']}`.",
            f"- Prismion minus untied-local phase accuracy: mean `{delta_untied['mean_delta']:.4f}`, std `{delta_untied['std_delta']:.4f}`, min `{delta_untied['min_delta']:.4f}`, max `{delta_untied['max_delta']:.4f}`, positive `{delta_untied['positive_seeds']}/{delta_untied['paired_seeds']}`.",
            "",
            "## Claim Boundary",
            "",
            "This does not test consciousness, full VRAXION, natural language, parser grounding, or factuality. It only tests whether Prismion-style phase/interference updates are useful in the local tied recurrent phase-wavefront mechanism implemented here.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(out_dir: Path, args: argparse.Namespace) -> list[PhaseExample]:
    _, eval_rows = build_dataset(2026, min(args.train_examples, 64), min(args.eval_examples, 128), args.phase_classes)
    write_jsonl(out_dir / "examples_sample.jsonl", [asdict(row) for row in eval_rows[:64]])
    return eval_rows


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, Any]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(config, seed) for config in configs for seed in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "queue.json", [{**asdict(config), "seed": seed} for config, seed in queue])
    eval_rows = write_samples(args.out, args)
    progress_path = args.out / "progress.jsonl"
    metrics_path = args.out / "metrics.jsonl"
    job_progress_root = args.out / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "total_jobs": len(queue), "jobs": jobs, "device": args.device})
    results: list[JobResult] = []
    write_outputs(args.out, args, results, "partial", jobs, eval_rows)
    if jobs <= 1:
        for config, seed in queue:
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
            result = run_job(config, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, asdict(result))
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results)})
            write_outputs(args.out, args, results, "partial", jobs, eval_rows)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = set()
            meta = {}
            for config, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
                fut = pool.submit(run_job, config, seed, args, job_progress_root)
                pending.add(fut)
                meta[fut] = (config, seed)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, args, results, "partial", jobs, eval_rows)
                    continue
                for fut in done:
                    config, seed = meta[fut]
                    result = fut.result()
                    results.append(result)
                    append_jsonl(metrics_path, asdict(result))
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, args, results, "partial", jobs, eval_rows)
    agg, labels = write_outputs(args.out, args, results, "complete", jobs, eval_rows)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "completed_jobs": len(results), "verdict": labels})
    return agg, labels, results


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if CONTRACT.exists():
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CONTRACT, args.out / "contract_snapshot.md")
    agg, labels, results = run_all(args, jobs)
    write_doc_result(agg, labels, args, jobs, len(results), results)
    print(json.dumps({"verdict": labels, "out": str(args.out), "completed_jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
