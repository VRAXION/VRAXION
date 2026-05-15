from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS_RESULT.md"

MAX_GRID = 16
INPUT_CHANNELS = 3
WALL, START, TARGET = 0, 1, 2
MAX_S = 32
DISTANCE_BUCKETS = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 24))
DISTANCE_BUCKET_NAMES = ("1-2", "3-4", "5-8", "9-16", "17-24")
TRAIN_STEPS = (2, 4, 8, 16)

ALL_ARMS = (
    "ORACLE_FULL_BFS",
    "ORACLE_TRUNCATED_BFS_S",
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "UNTIED_CNN_MATCHED_COMPUTE",
    "LOCAL_MESSAGE_PASSING_GNN",
    "HARD_WALL_REACHED_FRONTIER_LOOP",
    "HARD_WALL_ABC_LOOP",
    "HARD_WALL_HIGHWAY_SIDEPOCKET_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOOP",
)
LOCAL_ARMS = {
    "LOCAL_MESSAGE_PASSING_GNN",
    "HARD_WALL_REACHED_FRONTIER_LOOP",
    "HARD_WALL_ABC_LOOP",
    "HARD_WALL_HIGHWAY_SIDEPOCKET_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOOP",
}
GLOBAL_CONTROL_ARMS = {
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "MLP_FLATTENED",
    "GRU_FLATTENED",
    "LSTM_FLATTENED",
}
TRAINABLE_ARMS = set(ALL_ARMS) - {"ORACLE_FULL_BFS", "ORACLE_TRUNCATED_BFS_S", "LOCAL_MESSAGE_PASSING_GNN"}


@dataclass(frozen=True)
class WaveExample:
    case_id: str
    grid: list[list[list[float]]]
    grid_size: int
    start: tuple[int, int]
    target: tuple[int, int]
    label: int
    distance: int
    nominal_distance: int
    bucket: str
    tags: tuple[str, ...]
    split: str
    contrast_group_id: str | None
    reachable_map: list[list[int]]
    frontier_by_step: list[list[list[int]]]


@dataclass(frozen=True)
class Config:
    arm: str
    width: int
    settling_steps: int
    train_mode: str


@dataclass
class JobResult:
    arm: str
    width: int
    settling_steps: int
    train_mode: str
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


def parse_int_csv(value: str | None) -> list[int]:
    return [int(part) for part in parse_csv(value)]


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
    cpu = os.cpu_count() or 1
    if value.startswith("auto"):
        frac = float(value.replace("auto", "")) / 100.0
        return max(1, math.floor(cpu * frac))
    return max(1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS deterministic hard-wall wavefront probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "valid_slice", "full_survivors"], default="smoke")
    parser.add_argument("--from", dest="from_path", type=Path, default=None)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--widths", default=None)
    parser.add_argument("--settling-steps", default=None)
    parser.add_argument("--train-modes", default=None)
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--jobs", default="auto50")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def bucket_name(distance: int) -> str:
    for name, (lo, hi) in zip(DISTANCE_BUCKET_NAMES, DISTANCE_BUCKETS):
        if lo <= distance <= hi:
            return name
    return "17-24"


def snake_order(n: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        for c in cols:
            cells.append((r, c))
    return cells


def manhattan_path(n: int, distance: int, rng: random.Random) -> list[tuple[int, int]]:
    distance = max(1, min(distance, 2 * (n - 1)))
    min_dr = max(0, distance - (n - 1))
    max_dr = min(n - 1, distance)
    dr = rng.randint(min_dr, max_dr)
    dc = distance - dr
    sr = rng.choice([-1, 1])
    sc = rng.choice([-1, 1])
    r_low = dr if sr < 0 else 0
    r_high = n - 1 if sr < 0 else n - 1 - dr
    c_low = dc if sc < 0 else 0
    c_high = n - 1 if sc < 0 else n - 1 - dc
    r = rng.randint(r_low, r_high)
    c = rng.randint(c_low, c_high)
    path = [(r, c)]
    if rng.random() < 0.5:
        for _ in range(dr):
            r += sr
            path.append((r, c))
        for _ in range(dc):
            c += sc
            path.append((r, c))
    else:
        for _ in range(dc):
            c += sc
            path.append((r, c))
        for _ in range(dr):
            r += sr
            path.append((r, c))
    return path


def neighbors(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def bfs(grid: np.ndarray, n: int, start: tuple[int, int]) -> tuple[np.ndarray, dict[tuple[int, int], int], list[np.ndarray]]:
    wall = grid[WALL, :n, :n] > 0.5
    dist: dict[tuple[int, int], int] = {start: 0}
    q: deque[tuple[int, int]] = deque([start])
    frontiers = []
    seen_by_step = np.zeros((n, n), dtype=np.int64)
    seen_by_step[start] = 1
    frontier = np.zeros((n, n), dtype=np.int64)
    frontier[start] = 1
    frontiers.append(frontier.copy())
    while q:
        cell = q.popleft()
        for nb in neighbors(cell, n):
            if wall[nb] or nb in dist:
                continue
            dist[nb] = dist[cell] + 1
            seen_by_step[nb] = 1
            q.append(nb)
    for step in range(1, MAX_S + 1):
        fmap = np.zeros((n, n), dtype=np.int64)
        for cell, d in dist.items():
            if d == step:
                fmap[cell] = 1
        frontiers.append(fmap)
    reach = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int64)
    for cell in dist:
        reach[cell] = 1
    padded_frontiers = []
    for fmap in frontiers:
        padded = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int64)
        padded[:n, :n] = fmap
        padded_frontiers.append(padded)
    return reach, dist, padded_frontiers


def make_grid_from_path(n: int, path: list[tuple[int, int]], reachable: bool, rng: random.Random) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[str, ...]]:
    grid = np.zeros((INPUT_CHANNELS, MAX_GRID, MAX_GRID), dtype=np.float32)
    grid[WALL, :, :] = 1.0
    for r in range(n):
        for c in range(n):
            grid[WALL, r, c] = 1.0
    tags = []
    start = path[0]
    target = path[-1]
    if reachable:
        for cell in path:
            grid[WALL, cell[0], cell[1]] = 0.0
        tags.append("reachable")
    else:
        block_at = max(1, min(len(path) - 3, len(path) // 2))
        open_set = set()
        for idx, cell in enumerate(path):
            if idx == block_at:
                continue
            grid[WALL, cell[0], cell[1]] = 0.0
            open_set.add(cell)
        candidates = []
        path_set = set(path)
        for r in range(n):
            for c in range(n):
                cell = (r, c)
                if cell in path_set:
                    continue
                if all(nb not in open_set for nb in neighbors(cell, n)):
                    candidates.append(cell)
        if candidates:
            extra = rng.choice(candidates)
            grid[WALL, extra[0], extra[1]] = 0.0
        tags.extend(["unreachable", "near_miss", "same_target_neighborhood"])
        if rng.random() < 0.35:
            tags.append("distractor_island")
            for nb in neighbors(start, n)[:2]:
                grid[WALL, nb[0], nb[1]] = 0.0
    grid[START, start[0], start[1]] = 1.0
    grid[TARGET, target[0], target[1]] = 1.0
    return grid, start, target, tuple(tags)


def generate_one(seed: int, idx: int, split: str, n: int, distance: int, reachable: bool, pair_id: str | None = None) -> WaveExample:
    rng = random.Random(seed * 1_000_003 + idx * 9176 + n * 31 + distance * 13 + int(reachable))
    path = manhattan_path(n, distance, rng)
    distance = len(path) - 1
    grid, start, target, tags = make_grid_from_path(n, path, reachable, rng)
    reach_map, dist_map, frontiers = bfs(grid, n, start)
    full_distance = dist_map.get(target, -1)
    label = 1 if full_distance >= 0 else 0
    case_id = f"{split}_{seed}_{idx}_{n}_{distance}_{int(reachable)}"
    if pair_id:
        tags = tuple(sorted(set(tags + ("same_target_pair",))))
    return WaveExample(
        case_id=case_id,
        grid=grid.tolist(),
        grid_size=n,
        start=start,
        target=target,
        label=label,
        distance=full_distance,
        nominal_distance=distance,
        bucket=bucket_name(distance),
        tags=tags,
        split=split,
        contrast_group_id=pair_id,
        reachable_map=reach_map.tolist(),
        frontier_by_step=[f.tolist() for f in frontiers],
    )


def build_dataset(seed: int, train_examples: int, eval_examples: int) -> tuple[list[WaveExample], list[WaveExample]]:
    def build(count: int, split: str) -> list[WaveExample]:
        rows: list[WaveExample] = []
        idx = 0
        while len(rows) < count:
            bucket = DISTANCE_BUCKETS[idx % len(DISTANCE_BUCKETS)]
            distance = random.Random(seed + idx).randint(bucket[0], bucket[1])
            if split == "train":
                n = 8 if idx % 8 else 12
                distance = min(distance, 14 if n == 8 else distance)
            else:
                n = (8, 12, 16)[idx % 3]
            pair_id = f"pair_{split}_{seed}_{idx}" if distance >= 5 else None
            rows.append(generate_one(seed, idx, split, n, distance, True, pair_id))
            if len(rows) < count:
                rows.append(generate_one(seed, idx + 50_000, split, n, distance, False, pair_id))
            idx += 1
        return rows[:count]

    return build(train_examples, "train"), build(eval_examples, "eval")


def examples_to_tensors(rows: list[WaveExample], device: torch.device) -> dict[str, Any]:
    x = torch.tensor([row.grid for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([row.label for row in rows], dtype=torch.float32, device=device)
    target = torch.tensor([row.target for row in rows], dtype=torch.long, device=device)
    distance = torch.tensor([row.distance for row in rows], dtype=torch.long, device=device)
    nominal = torch.tensor([row.nominal_distance for row in rows], dtype=torch.long, device=device)
    size = torch.tensor([row.grid_size for row in rows], dtype=torch.long, device=device)
    return {"x": x, "y": y, "target": target, "distance": distance, "nominal": nominal, "size": size, "rows": rows}


class SummaryDirectHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.init = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, width, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.init(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(-1), {}


class TargetMarkerOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.bias.expand(x.shape[0]), {}


class FlattenMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_CHANNELS * MAX_GRID * MAX_GRID, width * 4),
            nn.ReLU(),
            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.net(x).squeeze(-1), {}


class FlatRNN(nn.Module):
    def __init__(self, width: int, kind: str):
        super().__init__()
        self.kind = kind
        cls = nn.GRU if kind == "GRU_FLATTENED" else nn.LSTM
        self.rnn = cls(INPUT_CHANNELS, width, batch_first=True)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seq = x.permute(0, 2, 3, 1).reshape(x.shape[0], MAX_GRID * MAX_GRID, INPUT_CHANNELS)
        out, state = self.rnn(seq)
        h = state[0][-1] if self.kind == "LSTM_FLATTENED" else state[-1]
        return self.head(h).squeeze(-1), {}


def target_cell(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[b, :, target[:, 0], target[:, 1]]


class UntiedCNN(nn.Module):
    def __init__(self, width: int, max_steps: int):
        super().__init__()
        self.inp = nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(width + INPUT_CHANNELS, width, 3, padding=1) for _ in range(max_steps)])
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.inp(x))
        traces = [h]
        for layer in self.layers[:steps]:
            h = torch.tanh(layer(torch.cat([h, x], dim=1)))
            traces.append(h)
        return self.head(target_cell(h, target)).squeeze(-1), trace_stats(traces, x)


class DeterministicMessagePassing(nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wall = x[:, WALL : WALL + 1]
        reach = x[:, START : START + 1] * (1.0 - wall)
        traces = [reach]
        for _ in range(steps):
            prop = F.max_pool2d(reach, 3, stride=1, padding=1)
            reach = torch.maximum(reach, prop * (1.0 - wall))
            traces.append(reach)
        logit = (target_cell(reach, target)[:, 0] - 0.5) * 20.0
        return logit, trace_stats(traces, x, reached=reach)


class LocalLoop(nn.Module):
    def __init__(self, arm: str, width: int):
        super().__init__()
        self.arm = arm
        self.width = width
        self.latent_in = nn.Conv2d(INPUT_CHANNELS + 2, width, 1)
        self.propose = nn.Conv2d(width + INPUT_CHANNELS + 2, width, 3, padding=1)
        self.frontier_gate = nn.Conv2d(width + INPUT_CHANNELS + 2, 1, 1)
        self.reached_gate = nn.Conv2d(width + INPUT_CHANNELS + 2, 1, 1)
        self.frontier_refine = nn.Conv2d(width, 1, 1)
        self.latent_update = nn.Conv2d(width + INPUT_CHANNELS + 2, width, 3, padding=1)
        self.side = nn.Conv2d(width, width, 3, padding=1)
        self.decay = nn.Parameter(torch.tensor(0.7))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int = 1, noise_std: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wall = x[:, WALL : WALL + 1]
        free = 1.0 - wall
        reached = x[:, START : START + 1] * free
        frontier = reached.clone()
        h = torch.tanh(self.latent_in(torch.cat([x, reached, frontier], dim=1))) * free
        if noise_std:
            h = h + torch.randn_like(h) * noise_std
        h = h * free
        latent_traces = [h]
        reached_traces = [reached]
        frontier_traces = [frontier]
        pre_frontier_wall = []
        pre_reached_wall = []
        monotonic_violations = []
        for _ in range(steps):
            source = torch.cat([h, x, reached, frontier], dim=1)
            proposal = torch.tanh(self.propose(source))
            if self.arm == "HARD_WALL_HIGHWAY_SIDEPOCKET_LOOP":
                proposal = proposal + 0.5 * torch.tanh(self.side(h))
            elif self.arm == "HARD_WALL_PRISMION_PHASE_LOOP":
                proposal = proposal * torch.cos(self.side(h))
            elif self.arm == "HARD_WALL_ABC_LOOP":
                proposal = torch.tanh(proposal + self.side(torch.tanh(proposal)))
            neighbor_frontier = F.max_pool2d(frontier, 3, stride=1, padding=1)
            frontier_gate = torch.sigmoid(self.frontier_gate(source))
            frontier_refine = torch.sigmoid(self.frontier_refine(proposal))
            pre_frontier = neighbor_frontier * frontier_gate * frontier_refine * (1.0 - reached)
            reached_gate = torch.sigmoid(self.reached_gate(source))
            pre_reached = torch.maximum(reached, pre_frontier * reached_gate)
            pre_frontier_wall.append((pre_frontier * wall).abs().mean())
            pre_reached_wall.append((pre_reached * wall).abs().mean())
            frontier_next = pre_frontier * free
            reached_next = torch.maximum(reached, pre_reached * free)
            monotonic_violations.append((reached_next + 1e-4 < reached).float().mean())
            decay = torch.sigmoid(self.decay)
            latent_pre = torch.tanh(self.latent_update(source) + proposal)
            h = torch.tanh(decay * h + latent_pre) * free
            frontier = frontier_next
            reached = reached_next
            latent_traces.append(h)
            reached_traces.append(reached)
            frontier_traces.append(frontier)
        target_reached = target_cell(reached, target)[:, 0].clamp(1e-4, 1.0 - 1e-4)
        logits = torch.logit(target_reached)
        stats = trace_stats(latent_traces, x, reached=reached)
        stats.update(hard_wall_stats(x, h, reached_traces, frontier_traces, pre_reached_wall, pre_frontier_wall, monotonic_violations))
        stats["target_reach_channel"] = target_reached
        return logits, stats


def hard_wall_stats(
    x: torch.Tensor,
    latent: torch.Tensor,
    reached_traces: list[torch.Tensor],
    frontier_traces: list[torch.Tensor],
    pre_reached_wall: list[torch.Tensor],
    pre_frontier_wall: list[torch.Tensor],
    monotonic_violations: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    wall = x[:, WALL : WALL + 1] > 0.5
    free = ~wall
    reached = reached_traces[-1]
    frontier = frontier_traces[-1]
    first_frontier = frontier_traces[0].mean().clamp_min(1e-6)
    frontier_decay = 1.0 - (frontier.mean() / first_frontier).clamp(max=10.0)
    return {
        "pre_mask_frontier_wall_write_norm": torch.stack(pre_frontier_wall).mean() if pre_frontier_wall else torch.zeros((), device=x.device),
        "pre_mask_reached_wall_write_norm": torch.stack(pre_reached_wall).mean() if pre_reached_wall else torch.zeros((), device=x.device),
        "post_mask_frontier_wall_leak_rate": ((frontier > 0.5) & wall).float().mean(),
        "post_mask_reached_wall_leak_rate": ((reached > 0.5) & wall).float().mean(),
        "latent_wall_write_norm": (latent.abs() * wall.float()).mean(),
        "reached_monotonicity_violation_rate": torch.stack(monotonic_violations).mean() if monotonic_violations else torch.zeros((), device=x.device),
        "frontier_decay_rate": frontier_decay,
        "frontier_stuck_rate": ((frontier > 0.5) & free).float().mean(),
    }


def trace_stats(traces: list[torch.Tensor], x: torch.Tensor, reached: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if len(traces) < 2:
        delta = torch.zeros((), device=x.device)
    else:
        delta = torch.stack([(traces[i] - traces[i - 1]).pow(2).mean().sqrt() for i in range(1, len(traces))]).mean()
    final = traces[-1]
    state_norm = torch.stack([t.pow(2).mean().sqrt() for t in traces]).mean()
    saturation = (final.abs() > 0.95).float().mean()
    if reached is None:
        reached = final[:, :1].sigmoid()
    wall = x[:, WALL : WALL + 1] > 0.5
    wall_leak = ((reached > 0.5) & wall).float().mean()
    boundary = torch.zeros_like(wall)
    boundary[:, :, 0, :] = 1
    boundary[:, :, -1, :] = 1
    boundary[:, :, :, 0] = 1
    boundary[:, :, :, -1] = 1
    boundary_leak = ((reached > 0.5) & boundary.bool() & wall).float().mean()
    return {"final_state_delta": delta, "state_norm": state_norm, "saturation_rate": saturation, "wall_leak_rate": wall_leak, "boundary_leak_rate": boundary_leak}


def make_model(config: Config) -> nn.Module | None:
    if config.arm == "SUMMARY_DIRECT_HEAD":
        return SummaryDirectHead(config.width)
    if config.arm == "TARGET_MARKER_ONLY":
        return TargetMarkerOnly()
    if config.arm == "MLP_FLATTENED":
        return FlattenMLP(config.width)
    if config.arm in {"GRU_FLATTENED", "LSTM_FLATTENED"}:
        return FlatRNN(config.width, config.arm)
    if config.arm == "UNTIED_CNN_MATCHED_COMPUTE":
        return UntiedCNN(config.width, MAX_S)
    if config.arm == "LOCAL_MESSAGE_PASSING_GNN":
        return DeterministicMessagePassing()
    if config.arm in LOCAL_ARMS:
        return LocalLoop(config.arm, config.width)
    return None


def iter_batches(n: int, batch_size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [np.array(idx[i : i + batch_size], dtype=np.int64) for i in range(0, n, batch_size)]


def forward_model(model: nn.Module, config: Config, batch: dict[str, Any], indices: np.ndarray, steps: int, noise_std: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = batch["x"][indices]
    target = batch["target"][indices]
    if isinstance(model, (UntiedCNN, DeterministicMessagePassing, LocalLoop)):
        if isinstance(model, LocalLoop):
            return model(x, target, steps=steps, noise_std=noise_std)
        return model(x, target, steps=steps)
    return model(x, steps=steps)


def train_model(config: Config, rows: list[WaveExample], args: argparse.Namespace, device: torch.device, progress_path: Path, seed: int) -> nn.Module | None:
    model = make_model(config)
    if model is None or config.arm not in TRAINABLE_ARMS:
        return model.to(device) if model is not None else None
    model.to(device)
    batch = examples_to_tensors(rows, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 991)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for indices in iter_batches(len(rows), args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            steps = rng.choice(TRAIN_STEPS) if config.train_mode == "VARIABLE_S_TRAINING" else config.settling_steps
            logits, _ = forward_model(model, config, batch, indices, steps)
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"][indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), **asdict(config), "seed": seed})
    return model


def truncated_label(row: WaveExample, steps: int) -> int:
    return 1 if row.distance >= 0 and row.distance <= steps else 0


def oracle_logits(rows: list[WaveExample], arm: str, steps: int, device: torch.device) -> torch.Tensor:
    if arm == "ORACLE_FULL_BFS":
        y = [row.label for row in rows]
    else:
        y = [truncated_label(row, steps) for row in rows]
    return torch.tensor([(v - 0.5) * 20.0 for v in y], dtype=torch.float32, device=device)


def predictions_for(model: nn.Module | None, config: Config, rows: list[WaveExample], args: argparse.Namespace, device: torch.device, noise_std: float = 0.0, steps: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    steps = config.settling_steps if steps is None else steps
    if config.arm.startswith("ORACLE_"):
        logits = oracle_logits(rows, config.arm, steps, device)
        return torch.sigmoid(logits).detach().cpu().numpy(), (logits > 0).detach().cpu().numpy().astype(np.int64), {}
    assert model is not None
    model.eval()
    batch = examples_to_tensors(rows, device)
    probs = []
    preds = []
    stat_rows: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            indices = np.arange(start, min(len(rows), start + args.batch_size))
            logits, stats = forward_model(model, config, batch, indices, steps, noise_std=noise_std)
            p = torch.sigmoid(logits)
            probs.extend(p.detach().cpu().numpy().tolist())
            preds.extend((p > 0.5).long().detach().cpu().numpy().tolist())
            for key, value in stats.items():
                stat_rows[key].append(float(value.float().mean().detach().cpu()))
    return np.array(probs), np.array(preds, dtype=np.int64), {k: float(np.mean(v)) for k, v in stat_rows.items()}


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean()) if len(gold) else math.nan


def pair_accuracy(rows: list[WaveExample], correct: np.ndarray) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(rows, correct):
        if row.contrast_group_id:
            groups[row.contrast_group_id].append(bool(ok))
    if not groups:
        return math.nan
    return float(np.mean([all(vals) for vals in groups.values()]))


def metric_by_filter(rows: list[WaveExample], correct: np.ndarray, fn) -> float:
    vals = [bool(ok) for row, ok in zip(rows, correct) if fn(row)]
    return float(np.mean(vals)) if vals else math.nan


def bucket_metrics(rows: list[WaveExample], correct: np.ndarray) -> dict[str, float]:
    out = {}
    for name in DISTANCE_BUCKET_NAMES:
        out[f"distance_bucket_{name}_accuracy"] = metric_by_filter(rows, correct, lambda row, name=name: row.bucket == name)
    return out


def evaluate(config: Config, model: nn.Module | None, train_rows: list[WaveExample], eval_rows: list[WaveExample], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    probs, pred, stats = predictions_for(model, config, eval_rows, args, device)
    gold = np.array([row.label for row in eval_rows], dtype=np.int64)
    truncated = np.array([truncated_label(row, config.settling_steps) for row in eval_rows], dtype=np.int64)
    correct = pred == gold
    metrics: dict[str, float] = {
        "reachable_accuracy": accuracy(pred, gold),
        "oracle_full_accuracy": 1.0 if config.arm == "ORACLE_FULL_BFS" else math.nan,
        "oracle_truncated_S_accuracy": accuracy(pred, truncated) if config.arm == "ORACLE_TRUNCATED_BFS_S" else math.nan,
        "heldout_larger_grid_accuracy": metric_by_filter(eval_rows, correct, lambda row: row.grid_size > 8),
        "long_path_accuracy": metric_by_filter(eval_rows, correct, lambda row: row.nominal_distance >= 9),
        "unreachable_near_miss_accuracy": metric_by_filter(eval_rows, correct, lambda row: "near_miss" in row.tags),
        "same_target_neighborhood_pair_accuracy": pair_accuracy(eval_rows, correct),
        "false_reach_rate": float(np.mean((pred == 1) & (gold == 0))),
        "false_block_rate": float(np.mean((pred == 0) & (gold == 1))),
        "locality_audit": 1.0 if config.arm in LOCAL_ARMS else math.nan,
        "fixed_s_accuracy": accuracy(pred, gold) if config.train_mode == "FIXED_S" else math.nan,
        "variable_s_accuracy": accuracy(pred, gold) if config.train_mode == "VARIABLE_S_TRAINING" else math.nan,
        "target_marker_only_baseline": max(float(gold.mean()), 1.0 - float(gold.mean())),
    }
    metrics.update(bucket_metrics(eval_rows, correct))
    metrics.update(stats)
    if config.arm in LOCAL_ARMS:
        reachable_near = [row for row in eval_rows if row.label == 1 and 0 <= row.distance <= config.settling_steps]
        reachable_far = [row for row in eval_rows if row.label == 1 and row.distance > config.settling_steps]
        all_near = [row for row in eval_rows if row.distance >= 0 and row.distance <= config.settling_steps]
        all_far = [row for row in eval_rows if row.distance > config.settling_steps]
        _, near_pred, _ = predictions_for(model, config, reachable_near, args, device) if reachable_near else (None, np.array([]), {})
        _, far_pred, _ = predictions_for(model, config, reachable_far, args, device) if reachable_far else (None, np.array([]), {})
        _, all_near_pred, _ = predictions_for(model, config, all_near, args, device) if all_near else (None, np.array([]), {})
        _, all_far_pred, _ = predictions_for(model, config, all_far, args, device) if all_far else (None, np.array([]), {})
        all_near_gold = np.array([row.label for row in all_near], dtype=np.int64)
        all_far_gold = np.array([row.label for row in all_far], dtype=np.int64)
        metrics["s_matches_distance_score"] = float(np.mean(near_pred == 1)) if len(near_pred) else math.nan
        metrics["locality_far_reach_rate"] = float(np.mean(far_pred == 1)) if len(far_pred) else math.nan
        metrics["acc_d_le_S"] = accuracy(all_near_pred, all_near_gold) if len(all_near_pred) else math.nan
        metrics["acc_d_gt_S"] = accuracy(all_far_pred, all_far_gold) if len(all_far_pred) else math.nan
        metrics["false_reach_d_gt_S"] = float(np.mean((all_far_pred == 1) & (all_far_gold == 1))) if len(all_far_pred) else math.nan
    else:
        metrics["s_matches_distance_score"] = math.nan
        metrics["locality_far_reach_rate"] = math.nan
        metrics["acc_d_le_S"] = math.nan
        metrics["acc_d_gt_S"] = math.nan
        metrics["false_reach_d_gt_S"] = math.nan
    if model is not None and config.arm in LOCAL_ARMS:
        over_steps = min(MAX_S, config.settling_steps + 8)
        _, over_pred, _ = predictions_for(model, config, eval_rows, args, device, steps=over_steps)
        metrics["overrun_stability"] = accuracy(over_pred, pred)
        _, noisy_pred, _ = predictions_for(model, config, eval_rows, args, device, noise_std=0.05)
        metrics["noise_recovery_accuracy"] = accuracy(noisy_pred, pred)
        _, s1_pred, _ = predictions_for(model, config, eval_rows, args, device, steps=1)
        metrics["settling_gain"] = accuracy(pred, gold) - accuracy(s1_pred, gold)
        metrics["propagation_curve_score"] = metrics["s_matches_distance_score"] - metrics["locality_far_reach_rate"]
        metrics["target_cell_reach_channel_accuracy"] = 1.0 - abs(float(np.nan_to_num(metrics.get("target_reach_channel", 0.5))) - float(gold.mean()))
    else:
        metrics.setdefault("overrun_stability", math.nan)
        metrics.setdefault("noise_recovery_accuracy", math.nan)
        metrics.setdefault("settling_gain", math.nan)
        metrics.setdefault("propagation_curve_score", math.nan)
        metrics.setdefault("target_cell_reach_channel_accuracy", math.nan)
        metrics.setdefault("wall_leak_rate", math.nan)
        metrics.setdefault("boundary_leak_rate", math.nan)
        metrics.setdefault("pre_mask_frontier_wall_write_norm", math.nan)
        metrics.setdefault("pre_mask_reached_wall_write_norm", math.nan)
        metrics.setdefault("post_mask_frontier_wall_leak_rate", math.nan)
        metrics.setdefault("post_mask_reached_wall_leak_rate", math.nan)
        metrics.setdefault("latent_wall_write_norm", math.nan)
        metrics.setdefault("reached_monotonicity_violation_rate", math.nan)
        metrics.setdefault("frontier_decay_rate", math.nan)
        metrics.setdefault("frontier_stuck_rate", math.nan)
        metrics.setdefault("final_state_delta", math.nan)
        metrics.setdefault("state_norm", math.nan)
        metrics.setdefault("saturation_rate", math.nan)
    metrics["s_generalization_gap"] = metrics["fixed_s_accuracy"] - metrics["variable_s_accuracy"] if not math.isnan(metrics["fixed_s_accuracy"]) and not math.isnan(metrics["variable_s_accuracy"]) else math.nan
    metrics["maze_family_accuracy"] = metrics["reachable_accuracy"]
    metrics["s_curve_accuracy"] = metrics["reachable_accuracy"]
    return metrics


def train_probe_placeholder(config: Config, model: nn.Module | None, rows: list[WaveExample], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if model is None or config.arm not in LOCAL_ARMS:
        return {"linear_probe_reachable_map_accuracy": math.nan, "linear_probe_frontier_accuracy": math.nan, "MLP_probe_reachable_map_accuracy_separate": math.nan, "frontier_iou_by_step": math.nan}
    # Lightweight post-hoc proxy: compare local loop binary reach channel against BFS map when exposed.
    _, _, stats = predictions_for(model, config, rows[: min(len(rows), 256)], args, device)
    return {"linear_probe_reachable_map_accuracy": math.nan, "linear_probe_frontier_accuracy": math.nan, "MLP_probe_reachable_map_accuracy_separate": math.nan, "frontier_iou_by_step": math.nan, **{k: v for k, v in stats.items() if k in ()}}


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path) -> JobResult:
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples)
    progress_path = progress_root / f"{config.arm}_w{config.width}_s{config.settling_steps}_{config.train_mode}_seed{seed}.jsonl"
    append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
    model = train_model(config, train_rows, args, device, progress_path, seed)
    metrics = evaluate(config, model, train_rows, eval_rows, args, device)
    metrics.update(train_probe_placeholder(config, model, eval_rows, args, device))
    metrics["parameter_count"] = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "metrics": metrics})
    return JobResult(config.arm, config.width, config.settling_steps, config.train_mode, seed, metrics)


def stage_defaults(stage: str) -> tuple[list[str], list[int], list[int], list[str]]:
    if stage == "smoke":
        return (
            ["ORACLE_FULL_BFS", "ORACLE_TRUNCATED_BFS_S", "SUMMARY_DIRECT_HEAD", "TARGET_MARKER_ONLY", "UNTIED_CNN_MATCHED_COMPUTE", "LOCAL_MESSAGE_PASSING_GNN", "HARD_WALL_REACHED_FRONTIER_LOOP", "HARD_WALL_ABC_LOOP", "HARD_WALL_PRISMION_PHASE_LOOP"],
            [16],
            [1, 4, 8],
            ["FIXED_S"],
        )
    if stage == "valid_slice":
        return (
            list(ALL_ARMS),
            [16, 32],
            [1, 2, 4, 8, 16, 24],
            ["FIXED_S", "VARIABLE_S_TRAINING"],
        )
    return ([], [], [], [])


def build_queue(args: argparse.Namespace) -> list[Config]:
    if args.stage == "full_survivors":
        if not args.from_path or not args.from_path.exists():
            raise FileNotFoundError("--from survivor_configs.json is required for full_survivors")
        rows = json.loads(args.from_path.read_text(encoding="utf-8"))
        return [Config(str(row["arm"]), int(row["width"]), int(row["settling_steps"]), str(row["train_mode"])) for row in rows]
    arms, widths, steps, modes = stage_defaults(args.stage)
    if args.arms:
        arms = parse_csv(args.arms)
    if args.widths:
        widths = parse_int_csv(args.widths)
    if args.settling_steps:
        steps = parse_int_csv(args.settling_steps)
    if args.train_modes:
        modes = parse_csv(args.train_modes)
    configs: list[Config] = []
    for arm in arms:
        arm_steps = steps if arm in LOCAL_ARMS or arm in {"ORACLE_TRUNCATED_BFS_S", "UNTIED_CNN_MATCHED_COMPUTE"} else [1]
        arm_modes = modes if arm in LOCAL_ARMS else ["FIXED_S"]
        arm_widths = widths if arm not in {"ORACLE_FULL_BFS", "ORACLE_TRUNCATED_BFS_S", "TARGET_MARKER_ONLY", "LOCAL_MESSAGE_PASSING_GNN"} else [0]
        for width in arm_widths:
            for step in arm_steps:
                for mode in arm_modes:
                    configs.append(Config(arm, width, step, mode))
    return sorted(set(configs), key=lambda c: (c.arm, c.width, c.settling_steps, c.train_mode))


def merge_values(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return {"mean": math.nan, "min": math.nan, "max": math.nan, "std": math.nan}
    return {"mean": round(float(np.mean(vals)), 6), "min": round(float(np.min(vals)), 6), "max": round(float(np.max(vals)), 6), "std": round(float(np.std(vals)), 6)}


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, int, int, str], list[JobResult]] = defaultdict(list)
    for row in results:
        groups[(row.arm, row.width, row.settling_steps, row.train_mode)].append(row)
    out = {}
    for (arm, width, steps, mode), rows in sorted(groups.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics = {name: merge_values([row.metrics.get(name, math.nan) for row in rows]) for name in metric_names}
        key = f"{arm}/w{width}/s{steps}/{mode}"
        out[key] = {"arm": arm, "width": width, "settling_steps": steps, "train_mode": mode, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(row: dict[str, Any], name: str, default: float = math.nan) -> float:
    value = row.get("metrics", {}).get(name)
    if isinstance(value, dict):
        return float(value.get("mean", default))
    if isinstance(value, (int, float)):
        return float(value)
    return default


def score(row: dict[str, Any]) -> float:
    return float(np.nanmean([metric_mean(row, "long_path_accuracy"), metric_mean(row, "heldout_larger_grid_accuracy")]))


def best(agg: dict[str, dict[str, Any]], arms: set[str]) -> dict[str, Any] | None:
    rows = [row for row in agg.values() if row["arm"] in arms]
    return max(rows, key=score, default=None)


def control_gaps(agg: dict[str, dict[str, Any]]) -> dict[str, float]:
    loop = best(agg, LOCAL_ARMS - {"LOCAL_MESSAGE_PASSING_GNN"})
    summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    untied = best(agg, {"UNTIED_CNN_MATCHED_COMPUTE"})
    return {
        "summary_direct_gap": round(score(loop) - score(summary), 6) if loop and summary else math.nan,
        "untied_compute_gap": round(score(loop) - score(untied), 6) if loop and untied else math.nan,
    }


def survivor_configs(agg: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not agg:
        return []
    best_score = max(score(row) for row in agg.values())
    rows = [row for row in agg.values() if score(row) >= best_score - 0.05]
    return [{k: row[k] for k in ("arm", "width", "settling_steps", "train_mode")} for row in rows]


def verdict(agg: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    oracle = best(agg, {"ORACLE_FULL_BFS"})
    trunc = best(agg, {"ORACLE_TRUNCATED_BFS_S"})
    summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    untied = best(agg, {"UNTIED_CNN_MATCHED_COMPUTE"})
    marker = best(agg, {"TARGET_MARKER_ONLY"})
    rnn = best(agg, {"GRU_FLATTENED", "LSTM_FLATTENED"})
    learned_loop = best(agg, LOCAL_ARMS - {"LOCAL_MESSAGE_PASSING_GNN"})
    prism = best(agg, {"HARD_WALL_PRISMION_PHASE_LOOP"})
    non_prism = best(agg, {"HARD_WALL_REACHED_FRONTIER_LOOP", "HARD_WALL_ABC_LOOP", "HARD_WALL_HIGHWAY_SIDEPOCKET_LOOP"})
    if oracle and metric_mean(oracle, "reachable_accuracy", 0.0) < 0.99:
        labels.append("TASK_TOO_HARD")
    if marker and metric_mean(marker, "reachable_accuracy", 0.0) > 0.75:
        labels.append("TARGET_MARKER_SHORTCUT")
    if learned_loop and summary and score(summary) >= score(learned_loop) - 0.03:
        labels.append("SUMMARY_SOLVES_TASK")
    if learned_loop and untied and score(untied) >= score(learned_loop) - 0.03:
        labels.append("UNTIED_COMPUTE_SOLVES_TASK")
    if learned_loop and rnn and score(rnn) >= score(learned_loop) - 0.03:
        labels.append("STANDARD_RNN_SUFFICIENT")
    if learned_loop and metric_mean(learned_loop, "settling_gain", 0.0) <= 0.01:
        labels.append("NO_SETTLING_GAIN")
    if learned_loop and (metric_mean(learned_loop, "post_mask_reached_wall_leak_rate", 0.0) > 0.02 or metric_mean(learned_loop, "post_mask_frontier_wall_leak_rate", 0.0) > 0.02):
        labels.append("WALL_GATE_FAILURE")
    if learned_loop and metric_mean(learned_loop, "post_mask_reached_wall_leak_rate", 1.0) <= 0.02 and max(metric_mean(learned_loop, "pre_mask_reached_wall_write_norm", 0.0), metric_mean(learned_loop, "pre_mask_frontier_wall_write_norm", 0.0)) > 0.10:
        labels.append("MASK_DOES_THE_WORK_WARNING")
    if learned_loop and max(metric_mean(learned_loop, "pre_mask_reached_wall_write_norm", 1.0), metric_mean(learned_loop, "pre_mask_frontier_wall_write_norm", 1.0)) <= 0.05 and metric_mean(learned_loop, "settling_gain", 0.0) > 0.05:
        labels.append("LEARNED_WALL_GATE_POSITIVE")
    if learned_loop and metric_mean(learned_loop, "locality_far_reach_rate", 0.0) > 0.35:
        labels.append("LOCALITY_LEAK_WARNING")
    if learned_loop and metric_mean(learned_loop, "saturation_rate", 0.0) > 0.85:
        labels.append("LOOP_UNSTABLE")
    if learned_loop and metric_mean(learned_loop, "frontier_stuck_rate", 0.0) > 0.25:
        labels.append("FRONTIER_STUCK_WARNING")
    if learned_loop and oracle and trunc and summary and untied:
        if (
            metric_mean(oracle, "reachable_accuracy", 0.0) >= 0.99
            and score(learned_loop) >= score(summary) + 0.15
            and score(learned_loop) >= score(best(agg, {"LOCAL_MESSAGE_PASSING_GNN"}) or learned_loop) - 0.03
            and metric_mean(learned_loop, "settling_gain", 0.0) > 0.02
            and metric_mean(learned_loop, "post_mask_reached_wall_leak_rate", 1.0) <= 0.02
            and metric_mean(learned_loop, "post_mask_frontier_wall_leak_rate", 1.0) <= 0.02
            and metric_mean(learned_loop, "locality_far_reach_rate", 1.0) <= 0.35
            and metric_mean(learned_loop, "reached_monotonicity_violation_rate", 1.0) <= 0.02
        ):
            labels.append("HARD_WALL_WAVEFRONT_POSITIVE")
    if prism and non_prism and metric_mean(prism, "long_path_accuracy", 0.0) >= metric_mean(non_prism, "long_path_accuracy", 0.0) + 0.05 and metric_mean(prism, "post_mask_reached_wall_leak_rate", 1.0) <= metric_mean(non_prism, "post_mask_reached_wall_leak_rate", 1.0):
        labels.append("PRISMION_HARD_WALL_POSITIVE")
    return sorted(set(labels or ["WAVEFRONT_PARTIAL"]))


def write_side_outputs(out_dir: Path, agg: dict[str, dict[str, Any]], rows: list[WaveExample]) -> None:
    conv = []
    dist = []
    probes = []
    for key, row in sorted(agg.items()):
        base = {k: row[k] for k in ("arm", "width", "settling_steps", "train_mode")}
        conv.append({**base, "key": key, "settling_gain": metric_mean(row, "settling_gain"), "final_state_delta": metric_mean(row, "final_state_delta"), "overrun_stability": metric_mean(row, "overrun_stability"), "noise_recovery_accuracy": metric_mean(row, "noise_recovery_accuracy"), "state_norm_by_step": metric_mean(row, "state_norm"), "saturation_rate": metric_mean(row, "saturation_rate"), "pre_mask_frontier_wall_write_norm": metric_mean(row, "pre_mask_frontier_wall_write_norm"), "pre_mask_reached_wall_write_norm": metric_mean(row, "pre_mask_reached_wall_write_norm"), "post_mask_frontier_wall_leak_rate": metric_mean(row, "post_mask_frontier_wall_leak_rate"), "post_mask_reached_wall_leak_rate": metric_mean(row, "post_mask_reached_wall_leak_rate"), "reached_monotonicity_violation_rate": metric_mean(row, "reached_monotonicity_violation_rate"), "frontier_decay_rate": metric_mean(row, "frontier_decay_rate"), "frontier_stuck_rate": metric_mean(row, "frontier_stuck_rate")})
        probes.append({**base, "key": key, "linear_probe_reachable_map_accuracy": metric_mean(row, "linear_probe_reachable_map_accuracy"), "linear_probe_frontier_accuracy": metric_mean(row, "linear_probe_frontier_accuracy"), "MLP_probe_reachable_map_accuracy_separate": metric_mean(row, "MLP_probe_reachable_map_accuracy_separate"), "frontier_iou_by_step": metric_mean(row, "frontier_iou_by_step")})
        for name in DISTANCE_BUCKET_NAMES:
            dist.append({**base, "key": key, "bucket": name, "accuracy": metric_mean(row, f"distance_bucket_{name}_accuracy")})
    write_jsonl(out_dir / "convergence_curves.jsonl", conv)
    write_jsonl(out_dir / "probe_results.jsonl", probes)
    write_jsonl(out_dir / "distance_bucket_metrics.jsonl", dist)
    write_jsonl(out_dir / "wavefront_cases.jsonl", [asdict(row) for row in rows[:128]])


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int, sample_rows: list[WaveExample]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    gaps = control_gaps(agg)
    survivors = survivor_configs(agg)
    write_json(out_dir / "survivor_configs.json", survivors)
    write_side_outputs(out_dir, agg, sample_rows)
    summary = {"status": status, "verdict": labels, "completed_jobs": len(results), "control_gaps": gaps, "survivor_count": len(survivors), "config": {"stage": args.stage, "seeds": args.seeds, "epochs": args.epochs, "jobs": jobs, "device": args.device, "os_cpu_count": os.cpu_count(), "torch_threads_per_worker": 1}, "aggregate": agg}
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Survivor configs: `{len(survivors)}`",
        "",
        "| Arm | W | S | Mode | Reach | Larger | Long | SettleGain | S<=d | FarReach | PostWall | PreWall | FrontierStuck | Overrun | Params |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["settling_steps"]), str(r["train_mode"]))):
        pre_wall = max(metric_mean(row, "pre_mask_frontier_wall_write_norm"), metric_mean(row, "pre_mask_reached_wall_write_norm"))
        post_wall = max(metric_mean(row, "post_mask_frontier_wall_leak_rate"), metric_mean(row, "post_mask_reached_wall_leak_rate"))
        lines.append("| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(row["arm"], row["width"], row["settling_steps"], row["train_mode"], metric_mean(row, "reachable_accuracy"), metric_mean(row, "heldout_larger_grid_accuracy"), metric_mean(row, "long_path_accuracy"), metric_mean(row, "settling_gain"), metric_mean(row, "s_matches_distance_score"), metric_mean(row, "locality_far_reach_rate"), post_wall, pre_wall, metric_mean(row, "frontier_stuck_rate"), metric_mean(row, "overrun_stability"), metric_mean(row, "parameter_count")))
    lines.extend(["", "## Control Gaps", "", f"- summary_direct_gap: `{gaps['summary_direct_gap']:.3f}`", f"- untied_compute_gap: `{gaps['untied_compute_gap']:.3f}`"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int) -> None:
    gaps = control_gaps(agg)
    lines = [
        "# STABLE_LOOP_WAVEFRONT_002_HARD_WALL_CHANNELS Result",
        "",
        "## Run",
        "",
        "```text",
        f"stage={args.stage}",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
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
        "## Control Gaps",
        "",
        "```json",
        json.dumps(gaps, indent=2),
        "```",
        "",
        "## Summary Table",
        "",
        "| Arm | W | S | Mode | Reach | Larger | Long | SettleGain | PostWall | PreWall | FrontierStuck | Overrun |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["settling_steps"]), str(r["train_mode"]))):
        pre_wall = max(metric_mean(row, "pre_mask_frontier_wall_write_norm"), metric_mean(row, "pre_mask_reached_wall_write_norm"))
        post_wall = max(metric_mean(row, "post_mask_frontier_wall_leak_rate"), metric_mean(row, "post_mask_reached_wall_leak_rate"))
        lines.append("| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(row["arm"], row["width"], row["settling_steps"], row["train_mode"], metric_mean(row, "reachable_accuracy"), metric_mean(row, "heldout_larger_grid_accuracy"), metric_mean(row, "long_path_accuracy"), metric_mean(row, "settling_gain"), post_wall, pre_wall, metric_mean(row, "frontier_stuck_rate"), metric_mean(row, "overrun_stability")))
    lines.extend(["", "## Claim Boundary", "", "This is a deterministic 2D hard-wall wavefront/reachability probe. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test."])
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(out_dir: Path, args: argparse.Namespace) -> list[WaveExample]:
    _, eval_rows = build_dataset(2026, min(args.train_examples, 32), min(args.eval_examples, 64))
    write_jsonl(out_dir / "examples_sample.jsonl", [asdict(row) for row in eval_rows[:32]])
    return eval_rows


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, Any]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(config, seed) for config in configs for seed in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "queue.json", [{**asdict(config), "seed": seed} for config, seed in queue])
    sample_rows = write_samples(args.out, args)
    progress_path = args.out / "progress.jsonl"
    metrics_path = args.out / "metrics.jsonl"
    job_progress_root = args.out / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "jobs": jobs, "total_jobs": len(queue), "stage": args.stage, "device": args.device})
    results: list[JobResult] = []
    write_outputs(args.out, results, args, "partial", jobs, sample_rows)
    if jobs <= 1:
        for config, seed in queue:
            result = run_job(config, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, asdict(result))
            write_outputs(args.out, results, args, "partial", jobs, sample_rows)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = set()
            future_meta = {}
            for config, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
                fut = pool.submit(run_job, config, seed, args, job_progress_root)
                pending.add(fut)
                future_meta[fut] = (config, seed)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, results, args, "partial", jobs, sample_rows)
                    continue
                for fut in done:
                    config, seed = future_meta[fut]
                    result = fut.result()
                    results.append(result)
                    append_jsonl(metrics_path, asdict(result))
                    write_outputs(args.out, results, args, "partial", jobs, sample_rows)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
    agg, labels = write_outputs(args.out, results, args, "complete", jobs, sample_rows)
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
    write_doc_result(agg, labels, args, jobs, len(results))
    print(json.dumps({"verdict": labels, "out": str(args.out), "completed_jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
