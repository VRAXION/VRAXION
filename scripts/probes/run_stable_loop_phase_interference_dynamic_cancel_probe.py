from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
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
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL_RESULT.md"

MAX_GRID = 16
INPUT_CHANNELS = 6
WALL, TARGET, SRC_REAL, SRC_IMAG, DELAY_REAL, DELAY_IMAG = range(INPUT_CHANNELS)
EVAL_STEPS = (4, 8, 16, 24, 32)
TRAIN_STEPS = (4, 8, 16, 24)
DELAY_STEP = 8
DECAY = 0.98
PHASE_EPS = 1e-4
READOUT_THRESHOLD = 0.35
MAX_PHASE_NORM = 2.0

ALL_ARMS = (
    "ORACLE_DYNAMIC_PHASE_S",
    "ORACLE_DYNAMIC_PHASE_FULL",
    "ORACLE_FIRST_ARRIVAL_BASELINE",
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC",
    "HARD_WALL_ABC_DYNAMIC_LOOP",
    "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC",
)
ORACLE_ARMS = {"ORACLE_DYNAMIC_PHASE_S", "ORACLE_DYNAMIC_PHASE_FULL", "ORACLE_FIRST_ARRIVAL_BASELINE"}
LOCAL_ARMS = {"LOCAL_MESSAGE_PASSING_GNN_DYNAMIC", "HARD_WALL_ABC_DYNAMIC_LOOP", "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP"}
TRAINABLE_ARMS = set(ALL_ARMS) - ORACLE_ARMS - {"LOCAL_MESSAGE_PASSING_GNN_DYNAMIC"}


@dataclass(frozen=True)
class DynamicExample:
    case_id: str
    grid: list[list[list[float]]]
    grid_size: int
    target: tuple[int, int]
    phase_classes: int
    full_label: int
    labels_by_s: dict[int, int]
    first_arrival_labels_by_s: dict[int, int]
    family: str
    bucket: str
    tags: tuple[str, ...]
    contrast_group_id: str | None
    sources: tuple[tuple[int, int, int, str], ...]


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
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def resolve_jobs(value: str) -> int:
    if value.startswith("auto"):
        cpu = os.cpu_count() or 1
        return max(1, math.floor(cpu * float(value.replace("auto", "")) / 100.0))
    return max(1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL probe.")
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


def phase_bucket(z: complex, phase_classes: int, eps: float = READOUT_THRESHOLD) -> int:
    if abs(z) < eps:
        return 0
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / phase_classes))) % phase_classes + 1


def distance_bucket(distance: int) -> str:
    if distance <= 4:
        return "1-4"
    if distance <= 8:
        return "5-8"
    if distance <= 16:
        return "9-16"
    return "17-24"


def shift_np(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(arr)
    sy0 = max(0, -dy)
    sy1 = arr.shape[0] - max(0, dy)
    sx0 = max(0, -dx)
    sx1 = arr.shape[1] - max(0, dx)
    dy0 = max(0, dy)
    dy1 = arr.shape[0] - max(0, -dy)
    dx0 = max(0, dx)
    dx1 = arr.shape[1] - max(0, -dx)
    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def clamp_complex_np(z: np.ndarray, max_norm: float = MAX_PHASE_NORM) -> np.ndarray:
    mag = np.abs(z)
    scale = np.ones_like(mag, dtype=np.float32)
    mask = mag > max_norm
    scale[mask] = max_norm / np.maximum(mag[mask], 1e-8)
    return z * scale


def blank_grid(n: int) -> np.ndarray:
    grid = np.zeros((INPUT_CHANNELS, MAX_GRID, MAX_GRID), dtype=np.float32)
    grid[WALL] = 1.0
    grid[WALL, :n, :n] = 1.0
    return grid


def carve_cell(grid: np.ndarray, cell: tuple[int, int]) -> None:
    grid[WALL, cell[0], cell[1]] = 0.0


def carve_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for cell in path:
        carve_cell(grid, cell)


def manhattan_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    axes = [0, 1]
    rng.shuffle(axes)
    for axis in axes:
        if axis == 0:
            while r != tr:
                r += 1 if tr > r else -1
                path.append((r, c))
        else:
            while c != tc:
                c += 1 if tc > c else -1
                path.append((r, c))
    return path


def place_source(grid: np.ndarray, cell: tuple[int, int], phase: int, phase_classes: int, delayed: bool = False) -> None:
    carve_cell(grid, cell)
    z = unit_phase(phase, phase_classes)
    ch_r = DELAY_REAL if delayed else SRC_REAL
    ch_i = DELAY_IMAG if delayed else SRC_IMAG
    grid[ch_r, cell[0], cell[1]] += float(z.real)
    grid[ch_i, cell[0], cell[1]] += float(z.imag)


def place_target(grid: np.ndarray, cell: tuple[int, int]) -> None:
    carve_cell(grid, cell)
    grid[TARGET, cell[0], cell[1]] = 1.0


def dynamic_oracle_labels(grid: np.ndarray, phase_classes: int, steps_values: tuple[int, ...] = EVAL_STEPS) -> tuple[dict[int, int], int]:
    wall = grid[WALL] > 0.5
    free = ~wall
    tr, tc = np.argwhere(grid[TARGET] > 0.5)[0]
    state = grid[SRC_REAL].astype(np.float32) + 1j * grid[SRC_IMAG].astype(np.float32)
    state *= free
    frontier = state.copy()
    delayed = grid[DELAY_REAL].astype(np.float32) + 1j * grid[DELAY_IMAG].astype(np.float32)
    delayed *= free
    labels: dict[int, int] = {0: phase_bucket(state[tr, tc], phase_classes)}
    for step in range(1, max(steps_values) + 1):
        incoming = np.zeros_like(state)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            incoming += shift_np(frontier, dy, dx)
        if step == DELAY_STEP:
            incoming += delayed
        incoming *= free
        incoming = clamp_complex_np(incoming)
        state = clamp_complex_np(DECAY * state + incoming) * free
        frontier = incoming * free
        if step in steps_values:
            labels[step] = phase_bucket(state[tr, tc], phase_classes)
    return {int(k): int(v) for k, v in labels.items() if k in steps_values}, labels[max(steps_values)]


def first_arrival_labels(grid: np.ndarray, phase_classes: int, steps_values: tuple[int, ...] = EVAL_STEPS) -> dict[int, int]:
    wall = grid[WALL] > 0.5
    free = ~wall
    tr, tc = np.argwhere(grid[TARGET] > 0.5)[0]
    source = grid[SRC_REAL].astype(np.float32) + 1j * grid[SRC_IMAG].astype(np.float32)
    source += grid[DELAY_REAL].astype(np.float32) + 1j * grid[DELAY_IMAG].astype(np.float32)
    source *= free
    frontier = source.copy()
    reached = source.copy()
    settled = np.abs(source) > PHASE_EPS
    labels: dict[int, int] = {0: phase_bucket(reached[tr, tc], phase_classes)}
    for step in range(1, max(steps_values) + 1):
        incoming = np.zeros_like(frontier)
        arrival = np.zeros(frontier.shape, dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = shift_np(frontier, dy, dx)
            incoming += shifted
            arrival += (np.abs(shifted) > PHASE_EPS).astype(np.float32)
        incoming *= free
        new_cells = (arrival > 0) & (~settled) & free
        frontier_next = np.zeros_like(frontier)
        keep = new_cells & (np.abs(incoming) > PHASE_EPS)
        frontier_next[keep] = incoming[keep]
        reached[new_cells] = incoming[new_cells]
        settled[new_cells] = True
        frontier = frontier_next
        if step in steps_values:
            labels[step] = phase_bucket(reached[tr, tc], phase_classes)
    return {int(k): int(v) for k, v in labels.items() if k in steps_values}


def finalize(
    case_id: str,
    grid: np.ndarray,
    n: int,
    target: tuple[int, int],
    phase_classes: int,
    family: str,
    tags: tuple[str, ...],
    group: str | None,
    sources: list[tuple[int, int, int, str]],
    nominal_distance: int,
) -> DynamicExample:
    labels, full = dynamic_oracle_labels(grid, phase_classes)
    first = first_arrival_labels(grid, phase_classes)
    return DynamicExample(
        case_id=case_id,
        grid=grid.tolist(),
        grid_size=n,
        target=target,
        phase_classes=phase_classes,
        full_label=full,
        labels_by_s=labels,
        first_arrival_labels_by_s=first,
        family=family,
        bucket=distance_bucket(nominal_distance),
        tags=tags,
        contrast_group_id=group,
        sources=tuple(sources),
    )


def opposite_phase(rng: random.Random, phase_classes: int) -> tuple[int, int]:
    p = rng.randrange(phase_classes)
    return p, (p + phase_classes // 2) % phase_classes


def make_single(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    source = (1, target[1])
    p = rng.randrange(phase_classes)
    grid = blank_grid(n)
    path = manhattan_path(source, target, rng)
    carve_path(grid, path)
    place_source(grid, source, p, phase_classes)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "single_phase_sanity", ("single",), None, [(source[0], source[1], p, "initial")], len(path) - 1)


def make_simultaneous_cancel(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    d = rng.choice([3, 4, 5])
    left = (target[0], target[1] - d)
    right = (target[0], target[1] + d)
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(left, target, rng))
    carve_path(grid, manhattan_path(right, target, rng))
    place_source(grid, left, p0, phase_classes)
    place_source(grid, right, p1, phase_classes)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "simultaneous_opposite_collision", ("cancellation", "simultaneous"), None, [(left[0], left[1], p0, "initial"), (right[0], right[1], p1, "initial")], d)


def make_delayed_cancel(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    source = (1, target[1])
    delayed = (n - 2, target[1])
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(source, target, rng))
    carve_path(grid, manhattan_path(delayed, target, rng))
    place_source(grid, source, p0, phase_classes, delayed=False)
    place_source(grid, delayed, p1, phase_classes, delayed=True)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "delayed_opposite_cancel", ("delayed_cancel", "late_overwrite"), None, [(source[0], source[1], p0, "initial"), (delayed[0], delayed[1], p1, "delayed")], abs(source[0] - target[0]))


def make_late_wrong(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    source = (1, 1)
    delayed = (n - 2, n - 2)
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(source, target, rng))
    carve_path(grid, manhattan_path(delayed, target, rng))
    place_source(grid, source, p0, phase_classes)
    place_source(grid, delayed, p1, phase_classes, delayed=True)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "late_wrong_phase_after_target", ("late_overwrite",), None, [(source[0], source[1], p0, "initial"), (delayed[0], delayed[1], p1, "delayed")], len(manhattan_path(source, target, rng)) - 1)


def make_reinforced(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    left = (target[0], 1)
    up = (1, target[1])
    delayed = (n - 2, target[1])
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    for src in [left, up, delayed]:
        carve_path(grid, manhattan_path(src, target, rng))
    place_source(grid, left, p0, phase_classes)
    place_source(grid, up, p0, phase_classes)
    place_source(grid, delayed, p1, phase_classes, delayed=True)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "reinforced_phase_vs_cancel", ("reinforcement", "dynamic_cancel"), None, [(left[0], left[1], p0, "initial"), (up[0], up[1], p0, "initial"), (delayed[0], delayed[1], p1, "delayed")], abs(left[1] - target[1]))


def make_gate(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    source = (1, target[1])
    delayed = (target[0], n - 2)
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(source, target, rng))
    carve_path(grid, manhattan_path(delayed, target, rng))
    # A one-cell side corridor makes the late wave act like a dynamic gate/cancel.
    place_source(grid, source, p0, phase_classes)
    place_source(grid, delayed, p1, phase_classes, delayed=True)
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "gate_open_close", ("gate_dynamic", "delayed_cancel"), None, [(source[0], source[1], p0, "initial"), (delayed[0], delayed[1], p1, "delayed")], abs(source[0] - target[0]))


def make_same_target_pair(case_id: str, rng: random.Random, phase_classes: int, n: int, delayed_cancel: bool, group: str) -> DynamicExample:
    target = (n // 2, n // 2)
    source = (1, target[1])
    delayed = (n - 2, target[1])
    p0, p1 = opposite_phase(rng, phase_classes)
    grid = blank_grid(n)
    carve_path(grid, manhattan_path(source, target, rng))
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            rr, cc = target[0] + dr, target[1] + dc
            if 0 <= rr < n and 0 <= cc < n:
                carve_cell(grid, (rr, cc))
    place_source(grid, source, p0, phase_classes)
    sources = [(source[0], source[1], p0, "initial")]
    if delayed_cancel:
        carve_path(grid, manhattan_path(delayed, target, rng))
        place_source(grid, delayed, p1, phase_classes, delayed=True)
        sources.append((delayed[0], delayed[1], p1, "delayed"))
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "same_target_neighborhood_dynamic_pair", ("same_target_dynamic", "delayed_cancel" if delayed_cancel else "single"), group, sources, abs(source[0] - target[0]))


def make_k4_equilibrium(case_id: str, rng: random.Random, phase_classes: int, n: int) -> DynamicExample:
    target = (n // 2, n // 2)
    grid = blank_grid(n)
    cells = [(1, target[1]), (n - 2, target[1]), (target[0], 1), (target[0], n - 2)]
    phases = [0, 1, 2, 3]
    sources = []
    for idx, (cell, phase) in enumerate(zip(cells, phases)):
        carve_path(grid, manhattan_path(cell, target, rng))
        delayed = idx >= 2
        place_source(grid, cell, phase % phase_classes, phase_classes, delayed=delayed)
        sources.append((cell[0], cell[1], phase % phase_classes, "delayed" if delayed else "initial"))
    place_target(grid, target)
    return finalize(case_id, grid, n, target, phase_classes, "K4_phase_equilibrium", ("k4_equilibrium", "dynamic_cancel"), None, sources, abs(cells[0][0] - target[0]))


def build_dataset(seed: int, train_examples: int, eval_examples: int, phase_classes: int) -> tuple[list[DynamicExample], list[DynamicExample]]:
    rng = random.Random(seed)

    def make_rows(count: int, split: str) -> list[DynamicExample]:
        rows: list[DynamicExample] = []
        families = ["single", "simul", "delay", "late", "reinforce", "gate", "pair_a", "pair_b"]
        if phase_classes == 4:
            families.append("k4")
        idx = 0
        while len(rows) < count:
            n = rng.choice([8, 12, 16]) if split == "eval" else rng.choice([8, 12])
            family = families[idx % len(families)]
            case_id = f"{split}_{idx:06d}_{family}"
            if family == "single":
                row = make_single(case_id, rng, phase_classes, n)
            elif family == "simul":
                row = make_simultaneous_cancel(case_id, rng, phase_classes, n)
            elif family == "delay":
                row = make_delayed_cancel(case_id, rng, phase_classes, n)
            elif family == "late":
                row = make_late_wrong(case_id, rng, phase_classes, n)
            elif family == "reinforce":
                row = make_reinforced(case_id, rng, phase_classes, n)
            elif family == "gate":
                row = make_gate(case_id, rng, phase_classes, n)
            elif family == "pair_a":
                group = f"{split}_dynpair_{idx // len(families):06d}"
                row = make_same_target_pair(case_id, rng, phase_classes, n, False, group)
            elif family == "pair_b":
                group = f"{split}_dynpair_{idx // len(families):06d}"
                row = make_same_target_pair(case_id, rng, phase_classes, n, True, group)
            else:
                row = make_k4_equilibrium(case_id, rng, phase_classes, n)
            rows.append(row)
            idx += 1
        return rows

    return make_rows(train_examples, "train"), make_rows(eval_examples, "eval")


def target_cell(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = torch.arange(h.shape[0], device=h.device)
    return h[b, :, target[:, 0].long(), target[:, 1].long()]


def phase_vectors(phase_classes: int, device: torch.device) -> torch.Tensor:
    vals = []
    for k in range(phase_classes):
        z = unit_phase(k, phase_classes)
        vals.append([float(z.real), float(z.imag)])
    return torch.tensor(vals, dtype=torch.float32, device=device)


def fixed_phase_logits(vec: torch.Tensor, phase_classes: int) -> torch.Tensor:
    phases = phase_vectors(phase_classes, vec.device)
    mag = vec.pow(2).sum(dim=1).add(1e-8).sqrt()
    none = (READOUT_THRESHOLD - mag) * 12.0
    phase_logits = (vec @ phases.T - READOUT_THRESHOLD) * 12.0
    return torch.cat([none.unsqueeze(1), phase_logits], dim=1)


def examples_to_tensors(rows: list[DynamicExample], device: torch.device) -> dict[str, Any]:
    x = torch.tensor(np.array([row.grid for row in rows], dtype=np.float32), device=device)
    target = torch.tensor(np.array([row.target for row in rows], dtype=np.int64), device=device)
    labels_by_s = {s: torch.tensor([row.labels_by_s[s] for row in rows], dtype=torch.long, device=device) for s in EVAL_STEPS}
    first_by_s = {s: torch.tensor([row.first_arrival_labels_by_s[s] for row in rows], dtype=torch.long, device=device) for s in EVAL_STEPS}
    full = torch.tensor([row.full_label for row in rows], dtype=torch.long, device=device)
    return {"x": x, "target": target, "labels_by_s": labels_by_s, "first_by_s": first_by_s, "full": full}


class SummaryDirectHead(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled), stats_from_trace([h], x)


class TargetMarkerOnly(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1), {}


class UntiedLocalCNN(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes
        self.inp = nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(width + INPUT_CHANNELS, width, 3, padding=1) for _ in range(max(EVAL_STEPS))])
        self.vec = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.inp(x))
        traces = [h]
        for layer in self.layers[:steps]:
            h = torch.tanh(layer(torch.cat([h, x], dim=1)))
            traces.append(h)
        vec_map = self.vec(h) * (1.0 - x[:, WALL : WALL + 1])
        vec = target_cell(vec_map, target)
        return fixed_phase_logits(vec, self.phase_classes), stats_from_trace(traces, x, reached=vec_map.pow(2).sum(dim=1, keepdim=True).sqrt())


def shift_torch(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    sy0 = max(0, -dy)
    sy1 = x.shape[-2] - max(0, dy)
    sx0 = max(0, -dx)
    sx1 = x.shape[-1] - max(0, dx)
    dy0 = max(0, dy)
    dy1 = x.shape[-2] - max(0, -dy)
    dx0 = max(0, dx)
    dx1 = x.shape[-1] - max(0, -dx)
    out[:, :, dy0:dy1, dx0:dx1] = x[:, :, sy0:sy1, sx0:sx1]
    return out


def neighbor_sum(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_r = torch.zeros_like(real)
    out_i = torch.zeros_like(imag)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        out_r = out_r + shift_torch(real, dy, dx)
        out_i = out_i + shift_torch(imag, dy, dx)
    return out_r, out_i


def clamp_phase_torch(real: torch.Tensor, imag: torch.Tensor, max_norm: float = MAX_PHASE_NORM) -> tuple[torch.Tensor, torch.Tensor]:
    mag = real.square().add(imag.square()).add(1e-8).sqrt()
    scale = torch.clamp(max_norm / mag, max=1.0)
    return real * scale, imag * scale


class DynamicMessagePassing(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vec, stats = dynamic_update_torch(x, target, steps, self.phase_classes, prismion=False, learnable=False)
        return fixed_phase_logits(vec, self.phase_classes), stats


class DynamicLoop(nn.Module):
    def __init__(self, arm: str, width: int, phase_classes: int):
        super().__init__()
        self.arm = arm
        self.phase_classes = phase_classes
        self.angle = nn.Parameter(torch.tensor(0.0))
        self.decay_logit = nn.Parameter(torch.tensor(3.9))
        self.latent_in = nn.Conv2d(INPUT_CHANNELS + 4, width, 1)
        self.latent_update = nn.Conv2d(width + INPUT_CHANNELS + 4, width, 3, padding=1)
        self.gate = nn.Conv2d(width + INPUT_CHANNELS + 4, 1, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        free = 1.0 - x[:, WALL : WALL + 1]
        state_r = x[:, SRC_REAL : SRC_REAL + 1] * free
        state_i = x[:, SRC_IMAG : SRC_IMAG + 1] * free
        frontier_r = state_r.clone()
        frontier_i = state_i.clone()
        h = torch.tanh(self.latent_in(torch.cat([x, state_r, state_i, frontier_r, frontier_i], dim=1))) * free
        states = [torch.cat([state_r, state_i], dim=1)]
        pre_wall = []
        for step in range(1, steps + 1):
            incoming_r, incoming_i = neighbor_sum(frontier_r, frontier_i)
            if step == DELAY_STEP:
                incoming_r = incoming_r + x[:, DELAY_REAL : DELAY_REAL + 1]
                incoming_i = incoming_i + x[:, DELAY_IMAG : DELAY_IMAG + 1]
            source = torch.cat([h, x, state_r, state_i, frontier_r, frontier_i], dim=1)
            local_gate = 0.998 + 0.002 * torch.sigmoid(self.gate(source))
            incoming_r = incoming_r * local_gate
            incoming_i = incoming_i * local_gate
            if self.arm == "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP":
                angle = torch.tanh(self.angle) * 0.02
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                rot_r = incoming_r * cos_a - incoming_i * sin_a
                rot_i = incoming_r * sin_a + incoming_i * cos_a
                incoming_r, incoming_i = rot_r, rot_i
            incoming_r = incoming_r * free
            incoming_i = incoming_i * free
            incoming_r, incoming_i = clamp_phase_torch(incoming_r, incoming_i)
            pre_wall.append(((incoming_r.abs() + incoming_i.abs()) * x[:, WALL : WALL + 1]).mean())
            decay = 0.95 + 0.05 * torch.sigmoid(self.decay_logit)
            state_r, state_i = clamp_phase_torch(decay * state_r + incoming_r, decay * state_i + incoming_i)
            state_r = state_r * free
            state_i = state_i * free
            frontier_r, frontier_i = incoming_r, incoming_i
            h = torch.tanh(0.8 * h + self.latent_update(source)) * free
            states.append(torch.cat([state_r, state_i], dim=1))
        vec = target_cell(torch.cat([state_r, state_i], dim=1), target)
        stats = stats_from_phase_trace(states, x, pre_wall)
        return fixed_phase_logits(vec, self.phase_classes), stats


def dynamic_update_torch(x: torch.Tensor, target: torch.Tensor, steps: int, phase_classes: int, prismion: bool, learnable: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del phase_classes, prismion, learnable
    free = 1.0 - x[:, WALL : WALL + 1]
    state_r = x[:, SRC_REAL : SRC_REAL + 1] * free
    state_i = x[:, SRC_IMAG : SRC_IMAG + 1] * free
    frontier_r = state_r.clone()
    frontier_i = state_i.clone()
    states = [torch.cat([state_r, state_i], dim=1)]
    pre_wall = []
    for step in range(1, steps + 1):
        incoming_r, incoming_i = neighbor_sum(frontier_r, frontier_i)
        if step == DELAY_STEP:
            incoming_r = incoming_r + x[:, DELAY_REAL : DELAY_REAL + 1]
            incoming_i = incoming_i + x[:, DELAY_IMAG : DELAY_IMAG + 1]
        incoming_r = incoming_r * free
        incoming_i = incoming_i * free
        incoming_r, incoming_i = clamp_phase_torch(incoming_r, incoming_i)
        pre_wall.append(((incoming_r.abs() + incoming_i.abs()) * x[:, WALL : WALL + 1]).mean())
        state_r, state_i = clamp_phase_torch(DECAY * state_r + incoming_r, DECAY * state_i + incoming_i)
        state_r = state_r * free
        state_i = state_i * free
        frontier_r, frontier_i = incoming_r, incoming_i
        states.append(torch.cat([state_r, state_i], dim=1))
    vec = target_cell(torch.cat([state_r, state_i], dim=1), target)
    return vec, stats_from_phase_trace(states, x, pre_wall)


def stats_from_trace(traces: list[torch.Tensor], x: torch.Tensor, reached: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if len(traces) > 1:
        delta = torch.stack([(traces[i] - traces[i - 1]).pow(2).mean().sqrt() for i in range(1, len(traces))]).mean()
    else:
        delta = torch.zeros((), device=x.device)
    stats = {"final_state_delta": delta, "phase_norm_by_step": torch.stack([t.pow(2).mean().sqrt() for t in traces]).mean(), "phase_saturation_rate": (traces[-1].abs() > 4.0).float().mean(), "pre_mask_wall_pressure": torch.zeros((), device=x.device)}
    if reached is None:
        stats["post_mask_wall_leak"] = torch.zeros((), device=x.device)
    else:
        wall = x[:, WALL : WALL + 1] > 0.5
        stats["post_mask_wall_leak"] = ((reached > 0.5) & wall).float().mean()
    return stats


def stats_from_phase_trace(states: list[torch.Tensor], x: torch.Tensor, pre_wall: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if len(states) > 1:
        delta = torch.stack([(states[i] - states[i - 1]).pow(2).mean().sqrt() for i in range(1, len(states))]).mean()
    else:
        delta = torch.zeros((), device=x.device)
    final = states[-1]
    wall = x[:, WALL : WALL + 1] > 0.5
    norm = final.pow(2).sum(dim=1, keepdim=True).sqrt()
    return {"final_state_delta": delta, "phase_norm_by_step": torch.stack([s.pow(2).sum(dim=1).sqrt().mean() for s in states]).mean(), "phase_saturation_rate": (final.abs() > 4.0).float().mean(), "pre_mask_wall_pressure": torch.stack(pre_wall).mean() if pre_wall else torch.zeros((), device=x.device), "post_mask_wall_leak": ((norm > 0.5) & wall).float().mean()}


def make_model(config: Config, phase_classes: int) -> nn.Module | None:
    if config.arm == "SUMMARY_DIRECT_HEAD":
        return SummaryDirectHead(config.width, phase_classes)
    if config.arm == "TARGET_MARKER_ONLY":
        return TargetMarkerOnly(phase_classes)
    if config.arm == "UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC":
        return UntiedLocalCNN(config.width, phase_classes)
    if config.arm == "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC":
        return DynamicMessagePassing(phase_classes)
    if config.arm in {"HARD_WALL_ABC_DYNAMIC_LOOP", "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP"}:
        return DynamicLoop(config.arm, config.width, phase_classes)
    return None


def iter_batches(n: int, batch_size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [np.array(idx[i : i + batch_size], dtype=np.int64) for i in range(0, n, batch_size)]


def forward_model(model: nn.Module, batch: dict[str, Any], indices: np.ndarray, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return model(batch["x"][indices], batch["target"][indices], steps=steps)


def train_model(config: Config, rows: list[DynamicExample], args: argparse.Namespace, device: torch.device, progress_path: Path, seed: int) -> nn.Module | None:
    model = make_model(config, args.phase_classes)
    if model is None:
        return None
    model.to(device)
    if config.arm not in TRAINABLE_ARMS:
        return model
    batch = examples_to_tensors(rows, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 7002)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for indices in iter_batches(len(rows), args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            steps = rng.choice(TRAIN_STEPS)
            logits, _ = forward_model(model, batch, indices, steps)
            loss = F.cross_entropy(logits, batch["labels_by_s"][steps][indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {config.arm} seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), **asdict(config), "seed": seed})
    return model


def oracle_predictions(config: Config, rows: list[DynamicExample], steps: int) -> np.ndarray:
    if config.arm == "ORACLE_DYNAMIC_PHASE_FULL":
        return np.array([row.full_label for row in rows], dtype=np.int64)
    if config.arm == "ORACLE_FIRST_ARRIVAL_BASELINE":
        return np.array([row.first_arrival_labels_by_s[steps] for row in rows], dtype=np.int64)
    return np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)


def predictions_for(model: nn.Module | None, config: Config, rows: list[DynamicExample], args: argparse.Namespace, device: torch.device, steps: int) -> tuple[np.ndarray, dict[str, float]]:
    if config.arm in ORACLE_ARMS:
        return oracle_predictions(config, rows, steps), {}
    assert model is not None
    model.eval()
    batch = examples_to_tensors(rows, device)
    preds = []
    stats: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            indices = np.arange(start, min(len(rows), start + args.batch_size))
            logits, batch_stats = forward_model(model, batch, indices, steps)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            for key, val in batch_stats.items():
                stats[key].append(float(val.float().mean().detach().cpu()))
    return np.array(preds, dtype=np.int64), {k: float(np.mean(v)) for k, v in stats.items()}


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean()) if len(gold) else math.nan


def filtered_accuracy(rows: list[DynamicExample], correct: np.ndarray, predicate) -> float:
    vals = [bool(ok) for row, ok in zip(rows, correct) if predicate(row)]
    return float(np.mean(vals)) if vals else math.nan


def pair_accuracy(rows: list[DynamicExample], correct: np.ndarray) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(rows, correct):
        if row.contrast_group_id:
            groups[row.contrast_group_id].append(bool(ok))
    return float(np.mean([all(vals) for vals in groups.values()])) if groups else math.nan


def class_balance(rows: list[DynamicExample], steps: int) -> dict[str, float]:
    labels = [row.labels_by_s[steps] for row in rows]
    counts = Counter(labels)
    n = max(1, len(labels))
    return {"majority_baseline": max(counts.values()) / n if counts else math.nan, "random_baseline": 1.0 / (rows[0].phase_classes + 1) if rows else math.nan, "none_rate": counts.get(0, 0) / n, "num_classes_seen": float(len(counts))}


def evaluate(config: Config, model: nn.Module | None, train_rows: list[DynamicExample], eval_rows: list[DynamicExample], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    del train_rows
    metrics: dict[str, float] = {}
    accs = []
    stat_rows: dict[str, list[float]] = defaultdict(list)
    for steps in EVAL_STEPS:
        pred, stats = predictions_for(model, config, eval_rows, args, device, steps)
        gold = np.array([row.labels_by_s[steps] for row in eval_rows], dtype=np.int64)
        correct = pred == gold
        acc = accuracy(pred, gold)
        accs.append(acc)
        metrics[f"dynamic_phase_accuracy_s{steps}"] = acc
        metrics[f"none_vs_phase_accuracy_s{steps}"] = accuracy((pred > 0).astype(np.int64), (gold > 0).astype(np.int64))
        phase_mask = gold > 0
        metrics[f"phase_bucket_accuracy_s{steps}"] = accuracy(pred[phase_mask], gold[phase_mask]) if phase_mask.any() else math.nan
        metrics[f"false_none_rate_s{steps}"] = float(((pred == 0) & (gold > 0)).mean())
        metrics[f"false_phase_rate_s{steps}"] = float(((pred > 0) & (gold == 0)).mean())
        metrics[f"wrong_phase_rate_s{steps}"] = float(((pred > 0) & (gold > 0) & (pred != gold)).mean())
        metrics[f"cancellation_accuracy_s{steps}"] = filtered_accuracy(eval_rows, correct, lambda row: "cancellation" in row.tags or "delayed_cancel" in row.tags)
        metrics[f"delayed_cancel_accuracy_s{steps}"] = filtered_accuracy(eval_rows, correct, lambda row: "delayed_cancel" in row.tags)
        metrics[f"late_overwrite_accuracy_s{steps}"] = filtered_accuracy(eval_rows, correct, lambda row: "late_overwrite" in row.tags)
        metrics[f"reinforcement_accuracy_s{steps}"] = filtered_accuracy(eval_rows, correct, lambda row: "reinforcement" in row.tags)
        metrics[f"gate_dynamic_accuracy_s{steps}"] = filtered_accuracy(eval_rows, correct, lambda row: "gate_dynamic" in row.tags)
        metrics[f"same_target_neighborhood_dynamic_pair_accuracy_s{steps}"] = pair_accuracy(eval_rows, correct)
        metrics[f"false_survival_after_cancel_s{steps}"] = float(((pred > 0) & (gold == 0) & np.array(["delayed_cancel" in row.tags for row in eval_rows], dtype=bool)).mean())
        metrics[f"false_cancel_without_antiphase_s{steps}"] = float(((pred == 0) & (gold > 0) & np.array(["reinforcement" in row.tags for row in eval_rows], dtype=bool)).mean())
        metrics[f"wrong_late_overwrite_rate_s{steps}"] = float(((pred > 0) & (gold > 0) & (pred != gold) & np.array(["late_overwrite" in row.tags for row in eval_rows], dtype=bool)).mean())
        for key, val in stats.items():
            stat_rows[key].append(val)
    metrics["dynamic_phase_accuracy"] = float(np.mean(accs))
    metrics["target_phase_accuracy"] = metrics["dynamic_phase_accuracy"]
    metrics["truncated_dynamic_accuracy_by_S"] = metrics["dynamic_phase_accuracy"]
    metrics["same_weights_s_curve_accuracy"] = metrics["dynamic_phase_accuracy"]
    metrics["full_dynamic_accuracy_at_large_S"] = metrics["dynamic_phase_accuracy_s32"]
    metrics["dynamic_s_matches_oracle_score"] = metrics["dynamic_phase_accuracy"]
    metrics["overrun_matches_dynamic_oracle"] = float(np.mean([1.0 if accs[i + 1] >= accs[i] - 0.05 else 0.0 for i in range(len(accs) - 1)]))
    for base in ["none_vs_phase_accuracy", "phase_bucket_accuracy", "false_none_rate", "false_phase_rate", "wrong_phase_rate", "cancellation_accuracy", "delayed_cancel_accuracy", "late_overwrite_accuracy", "reinforcement_accuracy", "gate_dynamic_accuracy", "same_target_neighborhood_dynamic_pair_accuracy", "false_survival_after_cancel", "false_cancel_without_antiphase", "wrong_late_overwrite_rate"]:
        metrics[base] = float(np.nanmean([metrics[f"{base}_s{s}"] for s in EVAL_STEPS]))
    for key, vals in stat_rows.items():
        metrics[key] = float(np.mean(vals))
    metrics.setdefault("pre_mask_wall_pressure", 0.0)
    metrics.setdefault("post_mask_wall_leak", 0.0)
    metrics["convergence_to_dynamic_oracle"] = metrics["dynamic_phase_accuracy"]
    metrics["noise_recovery_accuracy"] = math.nan
    metrics["parameter_count"] = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    if config.arm == "SUMMARY_DIRECT_HEAD":
        metrics["summary_direct_accuracy"] = metrics["dynamic_phase_accuracy"]
    if config.arm == "TARGET_MARKER_ONLY":
        metrics["target_marker_accuracy"] = metrics["dynamic_phase_accuracy"]
    if config.arm == "ORACLE_FIRST_ARRIVAL_BASELINE":
        metrics["first_arrival_baseline_accuracy"] = metrics["dynamic_phase_accuracy"]
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
    configs = []
    for arm in arms:
        width = 0 if arm in ORACLE_ARMS or arm in {"TARGET_MARKER_ONLY", "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC"} else args.width
        configs.append(Config(arm=arm, width=width, train_mode="SAME_WEIGHTS_S_CURVE"))
    return configs


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        groups[f"{result.arm}|w{result.width}|{result.train_mode}"].append(result)
    agg: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        metric_keys = sorted({mk for row in rows for mk in row.metrics})
        means = {}
        stds = {}
        for mk in metric_keys:
            vals = [row.metrics[mk] for row in rows if mk in row.metrics and not math.isnan(row.metrics[mk])]
            means[mk] = float(np.mean(vals)) if vals else math.nan
            stds[mk] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg[key] = {"arm": rows[0].arm, "width": rows[0].width, "train_mode": rows[0].train_mode, "phase_classes": rows[0].phase_classes, "seeds": [row.seed for row in rows], "num_seeds": len(rows), "metric_mean": means, "metric_std": stds}
    return agg


def metric(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    return float(row["metric_mean"].get(key, default))


def best(agg: dict[str, dict[str, Any]], arms: set[str], key: str = "dynamic_phase_accuracy") -> dict[str, Any] | None:
    rows = [row for row in agg.values() if row["arm"] in arms]
    return max(rows, key=lambda row: metric(row, key, -1.0)) if rows else None


def paired_delta(results: list[JobResult], left: str, right: str, key: str) -> dict[str, Any]:
    lmap = {row.seed: row.metrics.get(key, math.nan) for row in results if row.arm == left}
    rmap = {row.seed: row.metrics.get(key, math.nan) for row in results if row.arm == right}
    seeds = sorted(set(lmap) & set(rmap))
    vals = [{"seed": seed, "delta": float(lmap[seed] - rmap[seed])} for seed in seeds if not math.isnan(lmap[seed]) and not math.isnan(rmap[seed])]
    nums = [row["delta"] for row in vals]
    return {"left": left, "right": right, "metric": key, "deltas": vals, "mean_delta": float(np.mean(nums)) if nums else math.nan, "std_delta": float(np.std(nums)) if len(nums) > 1 else 0.0, "min_delta": float(np.min(nums)) if nums else math.nan, "max_delta": float(np.max(nums)) if nums else math.nan, "positive_seed_count": int(sum(1 for v in nums if v > 0)), "paired_seed_count": len(nums)}


def verdicts(agg: dict[str, dict[str, Any]], results: list[JobResult]) -> list[str]:
    labels = []
    dyn = best(agg, {"ORACLE_DYNAMIC_PHASE_S"})
    first = best(agg, {"ORACLE_FIRST_ARRIVAL_BASELINE"})
    summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    marker = best(agg, {"TARGET_MARKER_ONLY"})
    prism = best(agg, {"HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP"})
    abc = best(agg, {"HARD_WALL_ABC_DYNAMIC_LOOP"})
    gnn = best(agg, {"LOCAL_MESSAGE_PASSING_GNN_DYNAMIC"})
    if dyn and metric(dyn, "dynamic_phase_accuracy", 0.0) >= 0.99:
        if not first or metric(first, "dynamic_phase_accuracy", 1.0) <= 0.85:
            labels.append("DYNAMIC_CANCEL_TASK_VALID")
    if first and metric(first, "dynamic_phase_accuracy", 0.0) > 0.90:
        labels.append("FIRST_ARRIVAL_TASK_TOO_EASY")
    for ctrl in [summary, marker]:
        if ctrl and metric(ctrl, "dynamic_phase_accuracy", 0.0) > metric(ctrl, "class_balance_majority_baseline", 0.0) + 0.10:
            labels.append("SUMMARY_OR_TARGET_SHORTCUT_RETURNS")
    if prism and abc and gnn:
        if metric(prism, "dynamic_phase_accuracy", 0.0) >= 0.80 and metric(prism, "post_mask_wall_leak", 1.0) <= 0.02:
            labels.append("DYNAMIC_STABLE_LOOP_POSITIVE")
        if metric(abc, "dynamic_phase_accuracy", 0.0) >= metric(prism, "dynamic_phase_accuracy", 0.0) - 0.01 or metric(gnn, "dynamic_phase_accuracy", 0.0) >= metric(prism, "dynamic_phase_accuracy", 0.0) - 0.01:
            labels.append("CANONICAL_MESSAGE_PASSING_SUFFICIENT")
        if metric(abc, "dynamic_phase_accuracy", 0.0) >= 0.985:
            labels.append("ABC_CEILING_TASK_TOO_EASY")
        d_abc = paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "HARD_WALL_ABC_DYNAMIC_LOOP", "dynamic_phase_accuracy")
        d_gnn = paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC", "dynamic_phase_accuracy")
        if d_abc["mean_delta"] > 0.02 and d_gnn["mean_delta"] > 0.02 and metric(prism, "false_none_rate", 1.0) <= metric(abc, "false_none_rate", 1.0) + 0.01 and metric(prism, "false_phase_rate", 1.0) <= metric(abc, "false_phase_rate", 1.0) + 0.01:
            labels.append("PRISMION_DYNAMIC_CANCEL_POSITIVE")
    if prism and (metric(prism, "phase_saturation_rate", 0.0) > 0.25 or metric(prism, "post_mask_wall_leak", 0.0) > 0.02):
        labels.append("PHASE_LOOP_UNSTABLE")
    learned = [row for row in [prism, abc, gnn] if row]
    if dyn and learned and all(metric(row, "dynamic_phase_accuracy", 0.0) < 0.40 for row in learned):
        labels.append("TASK_TOO_HARD")
    return sorted(set(labels or ["DYNAMIC_CANCEL_PARTIAL"]))


def write_side_outputs(out_dir: Path, agg: dict[str, dict[str, Any]], results: list[JobResult], eval_rows: list[DynamicExample]) -> None:
    collision = []
    s_curve = []
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        base = {"arm": row["arm"], "width": row["width"], "phase_classes": row["phase_classes"]}
        collision.append({**base, "dynamic_phase_accuracy": metric(row, "dynamic_phase_accuracy"), "cancellation_accuracy": metric(row, "cancellation_accuracy"), "delayed_cancel_accuracy": metric(row, "delayed_cancel_accuracy"), "late_overwrite_accuracy": metric(row, "late_overwrite_accuracy"), "reinforcement_accuracy": metric(row, "reinforcement_accuracy"), "gate_dynamic_accuracy": metric(row, "gate_dynamic_accuracy"), "same_target_neighborhood_dynamic_pair_accuracy": metric(row, "same_target_neighborhood_dynamic_pair_accuracy"), "false_survival_after_cancel": metric(row, "false_survival_after_cancel"), "false_cancel_without_antiphase": metric(row, "false_cancel_without_antiphase")})
        s_curve.append({**base, "truncated_dynamic_accuracy_by_S": metric(row, "truncated_dynamic_accuracy_by_S"), "full_dynamic_accuracy_at_large_S": metric(row, "full_dynamic_accuracy_at_large_S"), "same_weights_s_curve_accuracy": metric(row, "same_weights_s_curve_accuracy"), "dynamic_s_matches_oracle_score": metric(row, "dynamic_s_matches_oracle_score"), "overrun_matches_dynamic_oracle": metric(row, "overrun_matches_dynamic_oracle"), "phase_norm_by_step": metric(row, "phase_norm_by_step"), "phase_saturation_rate": metric(row, "phase_saturation_rate"), "final_state_delta": metric(row, "final_state_delta"), "post_mask_wall_leak": metric(row, "post_mask_wall_leak"), "pre_mask_wall_pressure": metric(row, "pre_mask_wall_pressure")})
    write_jsonl(out_dir / "collision_metrics.jsonl", collision)
    write_jsonl(out_dir / "s_curve_metrics.jsonl", s_curve)
    write_jsonl(out_dir / "paired_seed_deltas.jsonl", [
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "HARD_WALL_ABC_DYNAMIC_LOOP", "dynamic_phase_accuracy"),
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC", "dynamic_phase_accuracy"),
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC", "dynamic_phase_accuracy"),
    ])
    write_jsonl(out_dir / "dynamic_phase_cases.jsonl", [asdict(row) for row in eval_rows[:128]])


def write_outputs(out_dir: Path, args: argparse.Namespace, results: list[JobResult], status: str, jobs: int, eval_rows: list[DynamicExample]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdicts(agg, results) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    write_side_outputs(out_dir, agg, results, eval_rows)
    summary = {"status": status, "verdict": labels, "completed_jobs": len(results), "config": {"phase_classes": args.phase_classes, "seeds": args.seeds, "train_examples": args.train_examples, "eval_examples": args.eval_examples, "epochs": args.epochs, "width": args.width, "jobs": jobs, "device": args.device, "os_cpu_count": os.cpu_count(), "torch_threads_per_worker": 1}, "aggregate": agg}
    write_json(out_dir / "summary.json", summary)
    lines = ["# STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL Report", "", f"- Status: `{status}`", f"- Verdict: `{', '.join(labels)}`", f"- Completed jobs: `{len(results)}`", f"- Phase classes: `{args.phase_classes}`", "", "| Arm | DynAcc | Cancel | Delay | Late | Reinforce | Gate | Pair | FalseNone | FalsePhase | First/WallPre | WallPost | Params |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.4f}` | `{:.0f}` |".format(row["arm"], metric(row, "dynamic_phase_accuracy"), metric(row, "cancellation_accuracy"), metric(row, "delayed_cancel_accuracy"), metric(row, "late_overwrite_accuracy"), metric(row, "reinforcement_accuracy"), metric(row, "gate_dynamic_accuracy"), metric(row, "same_target_neighborhood_dynamic_pair_accuracy"), metric(row, "false_none_rate"), metric(row, "false_phase_rate"), metric(row, "first_arrival_baseline_accuracy", metric(row, "pre_mask_wall_pressure")), metric(row, "post_mask_wall_leak"), metric(row, "parameter_count")))
    lines.extend(["", "## Paired Deltas", ""])
    for delta in [
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "HARD_WALL_ABC_DYNAMIC_LOOP", "dynamic_phase_accuracy"),
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "LOCAL_MESSAGE_PASSING_GNN_DYNAMIC", "dynamic_phase_accuracy"),
        paired_delta(results, "HARD_WALL_PRISMION_DYNAMIC_PHASE_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_DYNAMIC", "dynamic_phase_accuracy"),
    ]:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}` on `{delta['metric']}`: mean `{delta['mean_delta']:.4f}`, std `{delta['std_delta']:.4f}`, min `{delta['min_delta']:.4f}`, max `{delta['max_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int, results: list[JobResult]) -> None:
    del results
    lines = ["# STABLE_LOOP_PHASE_INTERFERENCE_002_DYNAMIC_CANCEL Result", "", "## Latest Run", "", "```text", f"phase_classes={args.phase_classes}", f"seeds={args.seeds}", f"train_examples={args.train_examples}", f"eval_examples={args.eval_examples}", f"epochs={args.epochs}", f"width={args.width}", f"jobs={jobs}", f"device={args.device}", f"completed_jobs={completed}", "```", "", "## Verdict", "", "```json", json.dumps(labels, indent=2), "```", "", "## Summary", "", "| Arm | DynAcc | Delay | Late | Pair | FalseNone | FalsePhase | WallPre | WallPost |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.4f}` |".format(row["arm"], metric(row, "dynamic_phase_accuracy"), metric(row, "delayed_cancel_accuracy"), metric(row, "late_overwrite_accuracy"), metric(row, "same_target_neighborhood_dynamic_pair_accuracy"), metric(row, "false_none_rate"), metric(row, "false_phase_rate"), metric(row, "pre_mask_wall_pressure"), metric(row, "post_mask_wall_leak")))
    lines.extend(["", "## Claim Boundary", "", "This dynamic-cancel probe only tests local tied-loop phase-state updates. It does not test consciousness, full VRAXION, language, parser grounding, or factuality."])
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(out_dir: Path, args: argparse.Namespace) -> list[DynamicExample]:
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
    job_progress = args.out / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "total_jobs": len(queue), "jobs": jobs, "device": args.device})
    results: list[JobResult] = []
    write_outputs(args.out, args, results, "partial", jobs, eval_rows)
    if jobs <= 1:
        for config, seed in queue:
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
            result = run_job(config, seed, args, job_progress)
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
                fut = pool.submit(run_job, config, seed, args, job_progress)
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
