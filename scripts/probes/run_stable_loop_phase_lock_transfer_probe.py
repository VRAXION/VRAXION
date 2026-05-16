from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_002_TRANSFER_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_002_TRANSFER_RESULT.md"

INPUT_CHANNELS = 6
WALL, TARGET, SRC_REAL, SRC_IMAG, GATE_REAL, GATE_IMAG = range(INPUT_CHANNELS)
EVAL_STEPS = (8, 16, 24, 32, 48)
TRAIN_STEPS = (8, 16, 24, 32)
READOUT_THRESHOLD = 0.35

ALL_ARMS = (
    "ORACLE_PHASE_LOCK_TRANSFER",
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "HARD_WALL_ABC_PHASE_LOCK_LOOP",
    "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK",
    "FIXED_PRISMION_PHASE_LOCK_LOOP",
    "LEARNED_PRISMION_PHASE_LOCK_LOOP",
    "LOCAL_BILINEAR_PHASE_LOOP",
    "COMPLEX_MULTIPLY_GNN",
)
ORACLE_ARMS = {"ORACLE_PHASE_LOCK_TRANSFER"}
FIXED_ARMS = {"ORACLE_PHASE_LOCK_TRANSFER", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "FIXED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN"}
TRAINABLE_ARMS = set(ALL_ARMS) - FIXED_ARMS
VECTOR_ARMS = {
    "HARD_WALL_ABC_PHASE_LOCK_LOOP",
    "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK",
    "FIXED_PRISMION_PHASE_LOCK_LOOP",
    "LEARNED_PRISMION_PHASE_LOCK_LOOP",
    "LOCAL_BILINEAR_PHASE_LOOP",
    "COMPLEX_MULTIPLY_GNN",
}


@dataclass(frozen=True)
class TransferExample:
    case_id: str
    grid: list[list[list[float]]]
    grid_size: int
    target: tuple[int, int]
    phase_classes: int
    labels_by_s: dict[int, int]
    full_label: int
    path_len: int
    bucket: str
    variant: str
    path_family: str
    noise_level: float
    contrast_group_id: str | None
    source_phase: float
    gate_phases: tuple[float, ...]
    path: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Config:
    arm: str
    width: int


@dataclass
class JobResult:
    arm: str
    width: int
    phase_classes: int
    seed: int
    metrics: dict[str, float]
    audit: dict[str, Any]


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
    return [part.strip() for part in (value or "").split(",") if part.strip()]


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
        return max(1, math.floor((os.cpu_count() or 1) * float(value.replace("auto", "")) / 100.0))
    return max(1, int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_LOCK_002_TRANSFER probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-classes", type=int, choices=[4], default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise-level", type=float, default=0.15)
    parser.add_argument("--mixed-cancel-phase", action="store_true")
    parser.add_argument("--jobs", default="4")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def unit_angle(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def unit_phase(k: float, phase_classes: int) -> complex:
    return unit_angle(2.0 * math.pi * k / phase_classes)


def phase_bucket(z: complex, phase_classes: int, threshold: float = READOUT_THRESHOLD) -> int:
    if abs(z) < threshold:
        return 0
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / phase_classes))) % phase_classes + 1


def bucket_len(path_len: int) -> str:
    if path_len <= 8:
        return "1-8"
    if path_len <= 16:
        return "9-16"
    if path_len <= 24:
        return "17-24"
    return "25+"


def shift_np(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(arr)
    sy0, sy1 = max(0, -dy), arr.shape[0] - max(0, dy)
    sx0, sx1 = max(0, -dx), arr.shape[1] - max(0, dx)
    dy0, dy1 = max(0, dy), arr.shape[0] - max(0, -dy)
    dx0, dx1 = max(0, dx), arr.shape[1] - max(0, -dx)
    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def phase_lock_oracle_labels(grid: np.ndarray, phase_classes: int) -> tuple[dict[int, int], int]:
    wall = grid[WALL] > 0.5
    free = ~wall
    gate = grid[GATE_REAL].astype(np.float32) + 1j * grid[GATE_IMAG].astype(np.float32)
    target = np.argwhere(grid[TARGET] > 0.5)[0]
    state = grid[SRC_REAL].astype(np.float32) + 1j * grid[SRC_IMAG].astype(np.float32)
    state *= free
    frontier = state.copy()
    written = np.abs(state) > 1e-4
    labels: dict[int, int] = {}
    for step in range(1, max(EVAL_STEPS) + 1):
        incoming = np.zeros_like(state)
        arrival = np.zeros(state.shape, dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = shift_np(frontier, dy, dx)
            incoming += shifted
            arrival += (np.abs(shifted) > 1e-4).astype(np.float32)
        incoming = incoming * gate * free
        active = (arrival > 0) & (~written) & free
        state[active] = incoming[active]
        written[active] = True
        frontier = np.zeros_like(state)
        frontier[active] = incoming[active]
        if step in EVAL_STEPS:
            labels[step] = phase_bucket(state[target[0], target[1]], phase_classes)
    return labels, labels[max(EVAL_STEPS)]


def blank_grid(n: int, max_grid: int) -> np.ndarray:
    grid = np.zeros((INPUT_CHANNELS, max_grid, max_grid), dtype=np.float32)
    grid[WALL] = 1.0
    grid[WALL, :n, :n] = 1.0
    grid[GATE_REAL, :n, :n] = 1.0
    return grid


def carve(grid: np.ndarray, cell: tuple[int, int]) -> None:
    grid[WALL, cell[0], cell[1]] = 0.0


def random_walk_path(n: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(300):
        r, c = rng.randint(1, n - 2), rng.randint(1, n - 2)
        path = [(r, c)]
        used = {(r, c)}
        for _step in range(length):
            opts = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < n - 1 and 1 <= nc < n - 1 and (nr, nc) not in used:
                    opts.append((nr, nc))
            if not opts:
                break
            r, c = rng.choice(opts)
            path.append((r, c))
            used.add((r, c))
        if len(path) == length + 1:
            return path
    return corridor_path(n, length, rng)


def corridor_path(n: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    r = rng.randint(2, n - 3)
    path: list[tuple[int, int]] = []
    for c in range(1, n - 1):
        path.append((r, c))
        if len(path) == length + 1:
            return path
    while len(path) < length + 1:
        r = max(1, min(n - 2, r + (-1 if len(path) % 2 else 1)))
        path.append((r, path[-1][1]))
    return path[: length + 1]


def zigzag_path(n: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    r = rng.randint(2, n - 3)
    c = 1
    path = [(r, c)]
    direction = 1
    while len(path) < length + 1:
        if 1 <= c + direction < n - 1:
            c += direction
        else:
            r = max(1, min(n - 2, r + 1))
            direction *= -1
        path.append((r, c))
    return path


def spiral_path(n: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    top, left, bottom, right = 1, 1, n - 2, n - 2
    path: list[tuple[int, int]] = []
    while top <= bottom and left <= right and len(path) < length + 1:
        for c in range(left, right + 1):
            path.append((top, c))
            if len(path) == length + 1:
                return path
        top += 1
        for r in range(top, bottom + 1):
            path.append((r, right))
            if len(path) == length + 1:
                return path
        right -= 1
        for c in range(right, left - 1, -1):
            path.append((bottom, c))
            if len(path) == length + 1:
                return path
        bottom -= 1
        for r in range(bottom, top - 1, -1):
            path.append((r, left))
            if len(path) == length + 1:
                return path
        left += 1
    while len(path) < length + 1:
        path.append(path[-1])
    return path[: length + 1]


def make_path(n: int, length: int, family: str, rng: random.Random) -> list[tuple[int, int]]:
    if family == "corridor":
        return corridor_path(n, length, rng)
    if family == "zigzag":
        return zigzag_path(n, length, rng)
    if family == "spiral":
        return spiral_path(n, length, rng)
    return random_walk_path(n, length, rng)


def gate_angle(gate_phase: float, phase_classes: int, rng: random.Random, noise: float) -> float:
    base = 2.0 * math.pi * gate_phase / phase_classes
    return base + (rng.uniform(-noise, noise) if noise > 0 else 0.0)


def make_example_from_path(
    case_id: str,
    rng: random.Random,
    phase_classes: int,
    max_grid: int,
    n: int,
    path: list[tuple[int, int]],
    variant: str,
    path_family: str,
    allowed_gates: list[int],
    noise: float = 0.0,
    contrast_group_id: str | None = None,
    source_phase: float | None = None,
    forced_gates: list[float] | None = None,
    gate_dropout: bool = False,
    mixed_cancel: bool = False,
) -> TransferExample:
    grid = blank_grid(n, max_grid)
    for cell in path:
        carve(grid, cell)
    source_phase = float(rng.randrange(phase_classes) if source_phase is None else source_phase)
    gates = forced_gates or [float(rng.choice(allowed_gates)) for _ in path[1:]]
    sr, sc = path[0]
    z = unit_phase(source_phase, phase_classes)
    grid[SRC_REAL, sr, sc] = float(z.real)
    grid[SRC_IMAG, sr, sc] = float(z.imag)
    if mixed_cancel and len(path) > 3:
        ar, ac = path[min(2, len(path) - 1)]
        anti = unit_phase(source_phase + phase_classes / 2.0, phase_classes)
        grid[SRC_REAL, ar, ac] = float(anti.real)
        grid[SRC_IMAG, ar, ac] = float(anti.imag)
    for idx, (cell, gate_phase) in enumerate(zip(path[1:], gates)):
        phase = 0.0 if gate_dropout and idx % 7 == 3 else gate_phase
        gz = unit_angle(gate_angle(phase, phase_classes, rng, noise))
        grid[GATE_REAL, cell[0], cell[1]] = float(gz.real)
        grid[GATE_IMAG, cell[0], cell[1]] = float(gz.imag)
    target = path[-1]
    grid[TARGET, target[0], target[1]] = 1.0
    labels, full_label = phase_lock_oracle_labels(grid, phase_classes)
    return TransferExample(
        case_id=case_id,
        grid=grid.tolist(),
        grid_size=n,
        target=target,
        phase_classes=phase_classes,
        labels_by_s=labels,
        full_label=full_label,
        path_len=len(path) - 1,
        bucket=bucket_len(len(path) - 1),
        variant=variant,
        path_family=path_family,
        noise_level=float(noise),
        contrast_group_id=contrast_group_id,
        source_phase=float(source_phase),
        gate_phases=tuple(float(g) for g in gates),
        path=tuple(path),
    )


def make_example(case_id: str, rng: random.Random, phase_classes: int, max_grid: int, split: str, idx: int, args: argparse.Namespace) -> list[TransferExample]:
    eval_mode = split == "eval"
    variant_cycle = [
        "baseline",
        "longer_paths",
        "noisy_gate_field",
        "heldout_gate_angles",
        "mixed_cancel_plus_phase_lock",
        "gate_dropout",
        "same_local_target_contrast",
        "reverse_path_consistency",
    ]
    variant = variant_cycle[idx % len(variant_cycle)]
    if not eval_mode and variant == "heldout_gate_angles":
        variant = "baseline"
    if not args.mixed_cancel_phase and variant == "mixed_cancel_plus_phase_lock":
        variant = "baseline"
    family = rng.choice(["random", "corridor", "zigzag", "spiral"])
    n = max_grid if variant == "longer_paths" else rng.choice([min(12, max_grid), min(16, max_grid), max_grid])
    n = max(8, n)
    length_choices = [6, 10, 14, 18, 22] if max_grid <= 32 else [10, 18, 26, 34, 42]
    if variant == "longer_paths":
        length_choices = [18, 22, 26, 30] if max_grid <= 32 else [26, 34, 42]
    length = min(rng.choice(length_choices), n * n // 3)
    path = make_path(n, length, family, rng)
    train_gates = [0, 1]
    eval_gates = [0, 1, 2, 3]
    allowed = [2, 3] if variant == "heldout_gate_angles" and eval_mode else (train_gates if not eval_mode else eval_gates)
    noise = args.noise_level if variant == "noisy_gate_field" else 0.0
    if variant == "same_local_target_contrast":
        group = f"{split}_pair_{idx // len(variant_cycle):06d}"
        gates = [float(rng.choice(allowed)) for _ in path[1:]]
        changed = list(gates)
        if changed:
            changed[len(changed) // 2] = (changed[len(changed) // 2] + 1.0) % phase_classes
        return [
            make_example_from_path(f"{split}_{idx:06d}_a", rng, phase_classes, max_grid, n, path, variant, family, allowed, noise, group, forced_gates=gates),
            make_example_from_path(f"{split}_{idx:06d}_b", rng, phase_classes, max_grid, n, path, variant, family, allowed, noise, group, forced_gates=changed),
        ]
    if variant == "reverse_path_consistency":
        path = list(reversed(path))
    return [
        make_example_from_path(
            f"{split}_{idx:06d}",
            rng,
            phase_classes,
            max_grid,
            n,
            path,
            variant,
            family,
            allowed,
            noise,
            gate_dropout=variant == "gate_dropout",
            mixed_cancel=variant == "mixed_cancel_plus_phase_lock",
        )
    ]


def build_dataset(seed: int, train_examples: int, eval_examples: int, phase_classes: int, max_grid: int, args: argparse.Namespace) -> tuple[list[TransferExample], list[TransferExample]]:
    rng = random.Random(seed)

    def rows(count: int, split: str) -> list[TransferExample]:
        out: list[TransferExample] = []
        idx = 0
        while len(out) < count:
            out.extend(make_example(f"{split}_{idx:06d}", rng, phase_classes, max_grid, split, idx, args))
            idx += 1
        return out[:count]

    return rows(train_examples, "train"), rows(eval_examples, "eval")


def target_cell(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = torch.arange(h.shape[0], device=h.device)
    return h[b, :, target[:, 0].long(), target[:, 1].long()]


def phase_vectors(phase_classes: int, device: torch.device) -> torch.Tensor:
    vals = []
    for k in range(phase_classes):
        z = unit_phase(k, phase_classes)
        vals.append([float(z.real), float(z.imag)])
    return torch.tensor(vals, dtype=torch.float32, device=device)


def fixed_logits(vec: torch.Tensor, phase_classes: int) -> torch.Tensor:
    phases = phase_vectors(phase_classes, vec.device)
    mag2 = vec.pow(2).sum(dim=1)
    none = (READOUT_THRESHOLD * READOUT_THRESHOLD - mag2) * 8.0
    phase_logits = (vec @ phases.T - READOUT_THRESHOLD) * 12.0
    return torch.cat([none.unsqueeze(1), phase_logits], dim=1)


def examples_to_tensors(rows: list[TransferExample], device: torch.device) -> dict[str, Any]:
    x = torch.tensor(np.array([row.grid for row in rows], dtype=np.float32), device=device)
    target = torch.tensor(np.array([row.target for row in rows], dtype=np.int64), device=device)
    labels = {s: torch.tensor([row.labels_by_s[s] for row in rows], dtype=torch.long, device=device) for s in EVAL_STEPS}
    return {"x": x, "target": target, "labels": labels}


def shift_torch(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    sy0, sy1 = max(0, -dy), x.shape[-2] - max(0, dy)
    sx0, sx1 = max(0, -dx), x.shape[-1] - max(0, dx)
    dy0, dy1 = max(0, dy), x.shape[-2] - max(0, -dy)
    dx0, dx1 = max(0, dx), x.shape[-1] - max(0, -dx)
    out[:, :, dy0:dy1, dx0:dx1] = x[:, :, sy0:sy1, sx0:sx1]
    return out


def neighbor_sum(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rr = torch.zeros_like(real)
    ii = torch.zeros_like(imag)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = rr + shift_torch(real, dy, dx)
        ii = ii + shift_torch(imag, dy, dx)
    return rr, ii


def complex_mul(a_re: torch.Tensor, a_im: torch.Tensor, b_re: torch.Tensor, b_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


class SummaryHead(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del target, steps
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled), {"activation_norm": h.abs().mean()}


class TargetOnly(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del target, steps
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1), {"activation_norm": self.logits.abs().mean()}


class UntiedCNN(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes
        self.inp = nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(width + INPUT_CHANNELS, width, 3, padding=1) for _ in range(max(EVAL_STEPS))])
        self.vec = nn.Conv2d(width, 2, 1)

    def target_vec(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.inp(x))
        for layer in self.layers[:steps]:
            h = torch.tanh(layer(torch.cat([h, x], dim=1)))
        vec_map = self.vec(h) * (1.0 - x[:, WALL : WALL + 1])
        return target_cell(vec_map, target), {"activation_norm": h.abs().mean(), "post_mask_wall_leak": torch.zeros((), device=x.device), "pre_mask_wall_pressure": torch.zeros((), device=x.device)}

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vec, st = self.target_vec(x, target, steps)
        return fixed_logits(vec, self.phase_classes), st


class PhaseLockLoop(nn.Module):
    def __init__(self, arm: str, width: int, phase_classes: int):
        super().__init__()
        self.arm = arm
        self.phase_classes = phase_classes
        if arm == "HARD_WALL_ABC_PHASE_LOCK_LOOP":
            self.h = nn.Conv2d(INPUT_CHANNELS + 2, width, 3, padding=1)
            self.vec = nn.Conv2d(width, 2, 1)
        elif arm == "LOCAL_BILINEAR_PHASE_LOOP":
            self.bilin = nn.Conv2d(8, 2, 1)
        elif arm == "LEARNED_PRISMION_PHASE_LOCK_LOOP":
            self.scale = nn.Parameter(torch.tensor([0.9, 0.9], dtype=torch.float32))
            self.mix = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def _next_vec(self, x: torch.Tensor, inc_r: torch.Tensor, inc_i: torch.Tensor, h: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gate_r = x[:, GATE_REAL : GATE_REAL + 1]
        gate_i = x[:, GATE_IMAG : GATE_IMAG + 1]
        if self.arm == "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK":
            return inc_r, inc_i, h
        if self.arm in {"FIXED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN"}:
            next_r, next_i = complex_mul(inc_r, inc_i, gate_r, gate_i)
            return next_r, next_i, h
        if self.arm == "LEARNED_PRISMION_PHASE_LOCK_LOOP":
            cm_r, cm_i = complex_mul(inc_r, inc_i, gate_r, gate_i)
            cj_r, cj_i = complex_mul(inc_r, inc_i, gate_r, -gate_i)
            next_r = self.scale[0] * cm_r + self.mix[0] * cj_r + self.bias[0]
            next_i = self.scale[1] * cm_i + self.mix[1] * cj_i + self.bias[1]
            return next_r, next_i, h
        if self.arm == "LOCAL_BILINEAR_PHASE_LOOP":
            feats = torch.cat([inc_r * gate_r, inc_r * gate_i, inc_i * gate_r, inc_i * gate_i, inc_r, inc_i, gate_r, gate_i], dim=1)
            out = self.bilin(feats)
            return out[:, 0:1], out[:, 1:2], h
        assert h is not None
        h = torch.tanh(self.h(torch.cat([x, inc_r, inc_i], dim=1)) + 0.5 * h)
        out = torch.tanh(self.vec(h))
        return out[:, 0:1], out[:, 1:2], h

    def target_vec(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        free = 1.0 - x[:, WALL : WALL + 1]
        real = x[:, SRC_REAL : SRC_REAL + 1] * free
        imag = x[:, SRC_IMAG : SRC_IMAG + 1] * free
        state_r = real.clone()
        state_i = imag.clone()
        written = real.square().add(imag.square()) > 1e-8
        h = torch.zeros((x.shape[0], self.h.out_channels, x.shape[2], x.shape[3]), device=x.device) if self.arm == "HARD_WALL_ABC_PHASE_LOCK_LOOP" else None
        activations = []
        pre_wall = []
        for _ in range(steps):
            inc_r, inc_i = neighbor_sum(real, imag)
            next_r, next_i, h = self._next_vec(x, inc_r, inc_i, h)
            wall = x[:, WALL : WALL + 1] > 0.5
            pre_wall.append(((next_r.square() + next_i.square()).sqrt() * wall.float()).mean())
            real, imag = next_r * free, next_i * free
            active = (real.square().add(imag.square()) > 1e-8) & (~written)
            state_r = torch.where(active, real, state_r)
            state_i = torch.where(active, imag, state_i)
            written = written | active
            real = torch.where(active, real, torch.zeros_like(real))
            imag = torch.where(active, imag, torch.zeros_like(imag))
            activations.append((real.abs() + imag.abs()).mean())
        vec_map = torch.cat([state_r, state_i], dim=1)
        wall = x[:, WALL : WALL + 1] > 0.5
        leak = ((vec_map.pow(2).sum(dim=1, keepdim=True).sqrt() > 0.5) & wall).float().mean()
        stats = {
            "post_mask_wall_leak": leak,
            "pre_mask_wall_pressure": torch.stack(pre_wall).mean() if pre_wall else torch.zeros((), device=x.device),
            "activation_norm": torch.stack(activations).mean() if activations else torch.zeros((), device=x.device),
        }
        return target_cell(vec_map, target), stats

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vec, st = self.target_vec(x, target, steps)
        return fixed_logits(vec, self.phase_classes), st


def make_model(config: Config, phase_classes: int) -> nn.Module | None:
    if config.arm == "SUMMARY_DIRECT_HEAD":
        return SummaryHead(config.width, phase_classes)
    if config.arm == "TARGET_MARKER_ONLY":
        return TargetOnly(phase_classes)
    if config.arm == "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK":
        return UntiedCNN(config.width, phase_classes)
    if config.arm in {
        "HARD_WALL_ABC_PHASE_LOCK_LOOP",
        "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK",
        "FIXED_PRISMION_PHASE_LOCK_LOOP",
        "LEARNED_PRISMION_PHASE_LOCK_LOOP",
        "LOCAL_BILINEAR_PHASE_LOOP",
        "COMPLEX_MULTIPLY_GNN",
    }:
        return PhaseLockLoop(config.arm, config.width, phase_classes)
    return None


def operator_audit(config: Config, model: nn.Module | None, train_audit: dict[str, float]) -> dict[str, Any]:
    mapping = {
        "ORACLE_PHASE_LOCK_TRANSFER": ("ORACLE", "fixed"),
        "SUMMARY_DIRECT_HEAD": ("GLOBAL_SUMMARY_CONTROL", "absent"),
        "TARGET_MARKER_ONLY": ("TARGET_PRIOR_CONTROL", "absent"),
        "HARD_WALL_ABC_PHASE_LOCK_LOOP": ("LEARNED_LOCAL_CONV", "absent"),
        "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK": ("CANONICAL_MESSAGE_PASSING", "absent"),
        "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK": ("UNTIED_LOCAL_COMPUTE", "learned"),
        "FIXED_PRISMION_PHASE_LOCK_LOOP": ("FIXED_OPERATOR", "fixed"),
        "LEARNED_PRISMION_PHASE_LOCK_LOOP": ("LEARNED_PRISMION_FACTOR", "learned"),
        "LOCAL_BILINEAR_PHASE_LOOP": ("GENERIC_BILINEAR", "learned"),
        "COMPLEX_MULTIPLY_GNN": ("GENERIC_COMPLEX_MULTIPLY", "fixed"),
    }
    params = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    trainable = float(sum(p.numel() for p in model.parameters() if p.requires_grad)) if model is not None else 0.0
    operator_type, phase_multiply = mapping[config.arm]
    return {
        "operator_type": operator_type,
        "phase_multiply": phase_multiply,
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "optimizer": "AdamW" if config.arm in TRAINABLE_ARMS else "none",
        **train_audit,
    }


def iter_batches(n: int, batch_size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [np.array(idx[i : i + batch_size], dtype=np.int64) for i in range(0, n, batch_size)]


def train_model(config: Config, rows: list[TransferExample], args: argparse.Namespace, device: torch.device, progress_path: Path, seed: int) -> tuple[nn.Module | None, dict[str, float]]:
    model = make_model(config, args.phase_classes)
    if model is None:
        return None, {"train_steps": 0.0, "gradient_norm_mean": 0.0, "activation_norm_mean": 0.0}
    model.to(device)
    if config.arm not in TRAINABLE_ARMS:
        return model, {"train_steps": 0.0, "gradient_norm_mean": 0.0, "activation_norm_mean": 0.0}
    batch = examples_to_tensors(rows, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 4409)
    grad_norms: list[float] = []
    activation_norms: list[float] = []
    train_steps = 0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for idx in iter_batches(len(rows), args.batch_size, rng):
            steps = rng.choice(TRAIN_STEPS)
            opt.zero_grad(set_to_none=True)
            logits, st = model(batch["x"][idx], batch["target"][idx], steps)
            loss = F.cross_entropy(logits, batch["labels"][steps][idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {config.arm} seed={seed}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_steps += 1
            grad_norms.append(float(grad.detach().cpu()))
            activation_norms.append(float(st.get("activation_norm", torch.zeros((), device=device)).detach().cpu()))
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), **asdict(config), "seed": seed})
    return model, {
        "train_steps": float(train_steps),
        "gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "activation_norm_mean": float(np.mean(activation_norms)) if activation_norms else 0.0,
    }


def oracle_pred(rows: list[TransferExample], steps: int) -> np.ndarray:
    return np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)


def predict(model: nn.Module | None, config: Config, rows: list[TransferExample], args: argparse.Namespace, device: torch.device, steps: int) -> tuple[np.ndarray, dict[str, float]]:
    if config.arm == "ORACLE_PHASE_LOCK_TRANSFER":
        return oracle_pred(rows, steps), {}
    assert model is not None
    model.eval()
    batch = examples_to_tensors(rows, device)
    preds = []
    stats: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            idx = np.arange(start, min(len(rows), start + args.batch_size))
            logits, st = model(batch["x"][idx], batch["target"][idx], steps)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            for key, value in st.items():
                stats[key].append(float(value.float().mean().detach().cpu()))
    return np.array(preds, dtype=np.int64), {k: float(np.mean(v)) for k, v in stats.items()}


def vector_predict(model: nn.Module | None, config: Config, rows: list[TransferExample], device: torch.device, steps: int, batch_size: int) -> np.ndarray | None:
    if config.arm not in VECTOR_ARMS or model is None or not hasattr(model, "target_vec"):
        return None
    model.eval()
    batch = examples_to_tensors(rows, device)
    vals = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            idx = np.arange(start, min(len(rows), start + batch_size))
            vec, _ = model.target_vec(batch["x"][idx], batch["target"][idx], steps)
            vals.extend(vec.detach().cpu().numpy().tolist())
    return np.array(vals, dtype=np.float32)


def acc(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean()) if len(gold) else math.nan


def pair_acc(rows: list[TransferExample], correct: np.ndarray) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(rows, correct):
        if row.contrast_group_id:
            groups[row.contrast_group_id].append(bool(ok))
    return float(np.mean([all(v) for v in groups.values()])) if groups else math.nan


def variant_acc(rows: list[TransferExample], correct: np.ndarray, variant: str) -> float:
    vals = [ok for row, ok in zip(rows, correct) if row.variant == variant]
    return float(np.mean(vals)) if vals else math.nan


def transform_rows(rows: list[TransferExample], control: str, seed: int) -> list[TransferExample]:
    rng = random.Random(seed)
    out = []
    grids = [np.array(row.grid, dtype=np.float32) for row in rows]
    wall_order = list(range(len(rows)))
    gate_order = list(range(len(rows)))
    rng.shuffle(wall_order)
    rng.shuffle(gate_order)
    for idx, row in enumerate(rows):
        grid = np.array(row.grid, dtype=np.float32)
        target = row.target
        if control == "gate_angle_shuffle_control":
            other = grids[gate_order[idx]]
            grid[GATE_REAL] = other[GATE_REAL]
            grid[GATE_IMAG] = other[GATE_IMAG]
        elif control == "target_cell_shuffle_control":
            free = np.argwhere(grid[WALL] < 0.5)
            tr, tc = tuple(free[(idx * 17 + 5) % len(free)])
            grid[TARGET] = 0.0
            grid[TARGET, tr, tc] = 1.0
            target = (int(tr), int(tc))
        elif control == "wall_mask_shuffle_control":
            other = grids[wall_order[idx]]
            grid[WALL] = other[WALL]
        elif control == "path_direction_shuffle_control":
            src = np.argwhere(np.abs(grid[SRC_REAL] + 1j * grid[SRC_IMAG]) > 1e-4)
            if len(src):
                tr, tc = tuple(src[0])
                grid[TARGET] = 0.0
                grid[TARGET, tr, tc] = 1.0
                target = (int(tr), int(tc))
        elif control == "phase_target_mismatch_control":
            sr = grid[SRC_REAL].copy()
            si = grid[SRC_IMAG].copy()
            rot = unit_phase(1, row.phase_classes)
            grid[SRC_REAL] = sr * rot.real - si * rot.imag
            grid[SRC_IMAG] = sr * rot.imag + si * rot.real
        out.append(replace(row, grid=grid.tolist(), target=target))
    return out


def control_accuracy(model: nn.Module | None, config: Config, rows: list[TransferExample], args: argparse.Namespace, device: torch.device, control: str) -> float:
    steps = 24
    if control == "label_shuffle_control":
        pred, _ = predict(model, config, rows, args, device, steps)
        gold = np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)
        rng = np.random.default_rng(1000 + steps)
        rng.shuffle(gold)
        return acc(pred, gold)
    controlled = transform_rows(rows, control, 8000 + len(control))
    pred, _ = predict(model, config, controlled, args, device, steps)
    gold = np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)
    return acc(pred, gold)


def vector_bucket_accuracy(vecs: np.ndarray, rows: list[TransferExample], phase_classes: int, steps: int) -> tuple[float, float]:
    preds = []
    angles = []
    for re, im in vecs:
        preds.append(phase_bucket(complex(float(re), float(im)), phase_classes))
        target = complex(float(re), float(im))
        gold_label = rows[len(preds) - 1].labels_by_s[steps]
        gold_angle = 0.0 if gold_label == 0 else 2.0 * math.pi * (gold_label - 1) / phase_classes
        pred_angle = math.atan2(target.imag, target.real)
        angles.append(abs(math.atan2(math.sin(pred_angle - gold_angle), math.cos(pred_angle - gold_angle))))
    gold = np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)
    return acc(np.array(preds, dtype=np.int64), gold), float(np.mean(angles)) if angles else math.nan


def evaluate(config: Config, model: nn.Module | None, eval_rows: list[TransferExample], args: argparse.Namespace, device: torch.device, audit: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    accs = []
    stats: dict[str, list[float]] = defaultdict(list)
    pred_by_s: dict[int, np.ndarray] = {}
    gold_by_s: dict[int, np.ndarray] = {}
    correct_by_s: dict[int, np.ndarray] = {}
    for steps in EVAL_STEPS:
        pred, st = predict(model, config, eval_rows, args, device, steps)
        gold = np.array([row.labels_by_s[steps] for row in eval_rows], dtype=np.int64)
        correct = pred == gold
        pred_by_s[steps] = pred
        gold_by_s[steps] = gold
        correct_by_s[steps] = correct
        metrics[f"phase_lock_accuracy_s{steps}"] = acc(pred, gold)
        metrics[f"phase_bucket_accuracy_s{steps}"] = acc(pred[gold > 0], gold[gold > 0]) if (gold > 0).any() else math.nan
        metrics[f"long_path_accuracy_s{steps}"] = float(np.mean([ok for ok, row in zip(correct, eval_rows) if row.bucket in {"17-24", "25+"}])) if any(row.bucket in {"17-24", "25+"} for row in eval_rows) else math.nan
        metrics[f"same_local_target_pair_accuracy_s{steps}"] = pair_acc(eval_rows, correct)
        accs.append(metrics[f"phase_lock_accuracy_s{steps}"])
        for key, value in st.items():
            stats[key].append(value)
    metrics["phase_lock_accuracy"] = float(np.mean(accs))
    metrics["transfer_accuracy"] = metrics["phase_lock_accuracy"]
    metrics["phase_bucket_accuracy"] = float(np.nanmean([metrics[f"phase_bucket_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["long_path_accuracy"] = float(np.nanmean([metrics[f"long_path_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["same_weights_s_curve_accuracy"] = metrics["phase_lock_accuracy"]
    metrics["same_local_target_pair_accuracy"] = float(np.nanmean([metrics[f"same_local_target_pair_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["paired_counterfactual_accuracy"] = metrics["same_local_target_pair_accuracy"]
    metrics["same_patch_different_answer_accuracy"] = metrics["same_local_target_pair_accuracy"]
    for variant in ["heldout_gate_angles", "noisy_gate_field", "mixed_cancel_plus_phase_lock", "reverse_path_consistency", "gate_dropout", "longer_paths"]:
        vals = []
        for steps in EVAL_STEPS:
            correct = correct_by_s[steps]
            selected = [ok for ok, row in zip(correct, eval_rows) if row.variant == variant]
            if selected:
                vals.append(float(np.mean(selected)))
        metrics[f"{variant}_accuracy"] = float(np.mean(vals)) if vals else math.nan
    metrics["heldout_gate_accuracy"] = metrics.get("heldout_gate_angles_accuracy", math.nan)
    metrics["noisy_gate_accuracy"] = metrics.get("noisy_gate_field_accuracy", math.nan)
    metrics["mixed_cancel_phase_accuracy"] = metrics.get("mixed_cancel_plus_phase_lock_accuracy", math.nan)
    metrics["reverse_path_consistency_accuracy"] = metrics.get("reverse_path_consistency_accuracy", math.nan)
    labels = [row.labels_by_s[24] for row in eval_rows]
    counts = Counter(labels)
    metrics["majority_baseline"] = max(counts.values()) / max(1, len(labels))
    metrics["random_baseline"] = 1.0 / (args.phase_classes + 1)
    metrics["chance_threshold"] = metrics["majority_baseline"] + 0.05
    for control in [
        "label_shuffle_control",
        "gate_angle_shuffle_control",
        "target_cell_shuffle_control",
        "wall_mask_shuffle_control",
        "path_direction_shuffle_control",
        "phase_target_mismatch_control",
    ]:
        metrics[control] = control_accuracy(model, config, eval_rows, args, device, control)
    for key, value in stats.items():
        metrics[key] = float(np.mean(value))
    metrics.setdefault("post_mask_wall_leak", 0.0)
    metrics.setdefault("pre_mask_wall_pressure", 0.0)
    metrics["wall_leak"] = metrics["post_mask_wall_leak"]
    metrics["pre_wall_pressure"] = metrics["pre_mask_wall_pressure"]
    metrics["parameter_count"] = float(audit["parameter_count"])
    metrics["trainable_parameter_count"] = float(audit["trainable_parameter_count"])
    metrics["train_steps"] = float(audit.get("train_steps", 0.0))
    metrics["gradient_norm_mean"] = float(audit.get("gradient_norm_mean", 0.0))
    metrics["activation_norm_mean"] = float(audit.get("activation_norm_mean", metrics.get("activation_norm", 0.0)))
    metrics["wrong_phase_rate"] = 1.0 - metrics["phase_bucket_accuracy"]
    metrics["false_none_rate"] = max(0.0, 1.0 - metrics["phase_lock_accuracy"])
    metrics["false_phase_rate"] = metrics["false_none_rate"]
    metrics["phase_drift_error"] = 1.0 - metrics["same_weights_s_curve_accuracy"]
    if config.arm in VECTOR_ARMS and model is not None:
        off_metrics = []
        off_angle = []
        for k in [8, 16]:
            _, off_rows = build_dataset(9300 + k, 64, min(512, args.eval_examples), k, args.width, args)
            vecs = vector_predict(model, config, off_rows, device, 32, args.batch_size)
            if vecs is not None:
                ok, angle = vector_bucket_accuracy(vecs, off_rows, k, 32)
                metrics[f"off_manifold_bucket_accuracy_k{k}"] = ok
                off_metrics.append(ok)
                off_angle.append(angle)
        metrics["off_manifold_bucket_accuracy"] = float(np.mean(off_metrics)) if off_metrics else math.nan
        metrics["off_manifold_angle_error"] = float(np.mean(off_angle)) if off_angle else math.nan
    return metrics


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path) -> JobResult:
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples, args.phase_classes, args.width, args)
    progress = progress_root / f"{config.arm}_w{config.width}_seed{seed}.jsonl"
    append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
    model, train_audit = train_model(config, train_rows, args, device, progress, seed)
    audit = operator_audit(config, model, train_audit)
    metrics = evaluate(config, model, eval_rows, args, device, audit)
    append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "metrics": metrics, "audit": audit})
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return JobResult(config.arm, config.width, args.phase_classes, seed, metrics, audit)


def build_queue(args: argparse.Namespace) -> list[Config]:
    arms = parse_csv(args.arms) if args.arms else list(ALL_ARMS)
    out = []
    for arm in arms:
        width = 0 if arm in ORACLE_ARMS or arm in {"TARGET_MARKER_ONLY", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "FIXED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN"} else args.width
        out.append(Config(arm, width))
    return out


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        groups[f"{result.arm}|w{result.width}"].append(result)
    agg: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        metric_keys = sorted({k for row in rows for k in row.metrics})
        means = {}
        stds = {}
        for mk in metric_keys:
            vals = [row.metrics[mk] for row in rows if mk in row.metrics and isinstance(row.metrics[mk], (int, float)) and not math.isnan(float(row.metrics[mk]))]
            means[mk] = float(np.mean(vals)) if vals else math.nan
            stds[mk] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg[key] = {
            "arm": rows[0].arm,
            "width": rows[0].width,
            "phase_classes": rows[0].phase_classes,
            "seeds": [r.seed for r in rows],
            "metric_mean": means,
            "metric_std": stds,
            "audit": rows[0].audit,
        }
    return agg


def m(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    return float(row["metric_mean"].get(key, default)) if row else default


def best(agg: dict[str, dict[str, Any]], arm: str) -> dict[str, Any] | None:
    for row in agg.values():
        if row["arm"] == arm:
            return row
    return None


def paired(results: list[JobResult], left: str, right: str, key: str) -> dict[str, Any]:
    lmap = {r.seed: r.metrics[key] for r in results if r.arm == left and key in r.metrics}
    rmap = {r.seed: r.metrics[key] for r in results if r.arm == right and key in r.metrics}
    seeds = sorted(set(lmap) & set(rmap))
    vals = [{"seed": seed, "delta": float(lmap[seed] - rmap[seed])} for seed in seeds]
    nums = [v["delta"] for v in vals]
    mean = float(np.mean(nums)) if nums else math.nan
    std = float(np.std(nums)) if len(nums) > 1 else 0.0
    lower95 = mean - 1.96 * std / math.sqrt(len(nums)) if nums else math.nan
    return {
        "left": left,
        "right": right,
        "metric": key,
        "deltas": vals,
        "mean_delta": mean,
        "lower95_delta": lower95,
        "std_delta": std,
        "min_delta": float(np.min(nums)) if nums else math.nan,
        "max_delta": float(np.max(nums)) if nums else math.nan,
        "positive_seed_count": int(sum(v > 0 for v in nums)),
        "paired_seed_count": len(nums),
    }


def shortcut_failed(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    threshold = m(row, "chance_threshold", 0.0)
    return m(row, "phase_lock_accuracy", 1.0) <= threshold and m(row, "label_shuffle_control", 1.0) <= threshold and m(row, "target_cell_shuffle_control", 1.0) <= threshold


def verdicts(agg: dict[str, dict[str, Any]], results: list[JobResult]) -> list[str]:
    labels: list[str] = []
    oracle = best(agg, "ORACLE_PHASE_LOCK_TRANSFER")
    fixed = best(agg, "FIXED_PRISMION_PHASE_LOCK_LOOP")
    learned = best(agg, "LEARNED_PRISMION_PHASE_LOCK_LOOP")
    complex_gnn = best(agg, "COMPLEX_MULTIPLY_GNN")
    bilinear = best(agg, "LOCAL_BILINEAR_PHASE_LOOP")
    abc = best(agg, "HARD_WALL_ABC_PHASE_LOCK_LOOP")
    gnn = best(agg, "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK")
    summary = best(agg, "SUMMARY_DIRECT_HEAD")
    target = best(agg, "TARGET_MARKER_ONLY")
    if oracle and fixed and m(oracle, "phase_lock_accuracy", 0.0) >= 0.99 and m(fixed, "phase_lock_accuracy", 0.0) >= 0.90 and m(fixed, "wall_leak", 1.0) <= 0.02:
        labels.append("PHASE_LOCK_TRANSFER_TASK_VALID")
    if (summary and not shortcut_failed(summary)) or (target and not shortcut_failed(target)):
        labels.append("SUMMARY_OR_TARGET_SHORTCUT_RETURNS")
    if fixed and abc and gnn and m(fixed, "phase_lock_accuracy", 0.0) > max(m(abc, "phase_lock_accuracy", 0.0), m(gnn, "phase_lock_accuracy", 0.0)) + 0.10:
        labels.append("FIXED_PRISMION_TRANSFER_POSITIVE")
    if learned and abc and gnn:
        d_abc = paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy")
        d_gnn = paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy")
        if d_abc["lower95_delta"] > 0.02 and d_gnn["lower95_delta"] > 0.02 and d_abc["positive_seed_count"] >= max(1, math.ceil(0.8 * d_abc["paired_seed_count"])):
            labels.append("LEARNED_PRISMION_TRANSFER_POSITIVE")
    if fixed and learned and m(fixed, "phase_lock_accuracy", 0.0) >= 0.90 and m(learned, "phase_lock_accuracy", 0.0) < m(fixed, "phase_lock_accuracy", 0.0) - 0.10:
        labels.append("FIXED_PRIMITIVE_ONLY")
    if fixed and complex_gnn and m(complex_gnn, "phase_lock_accuracy", 0.0) >= m(fixed, "phase_lock_accuracy", 0.0) - 0.02:
        labels.append("COMPLEX_MULTIPLY_SUFFICIENT")
    if fixed and bilinear and m(bilinear, "phase_lock_accuracy", 0.0) >= m(fixed, "phase_lock_accuracy", 0.0) - 0.02:
        labels.append("COMPLEX_MULTIPLY_SUFFICIENT")
    if learned and bilinear and complex_gnn:
        d_bi = paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_BILINEAR_PHASE_LOOP", "phase_lock_accuracy")
        d_cx = paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN", "phase_lock_accuracy")
        if d_bi["lower95_delta"] > 0.02 and d_cx["lower95_delta"] > 0.02:
            labels.append("PRISMION_UNIQUELY_USEFUL")
    strong = [row for row in [fixed, learned, complex_gnn, bilinear] if row is not None]
    if strong and all(m(row, "phase_lock_accuracy", 0.0) >= 0.97 for row in strong):
        labels.append("TASK_TOO_EASY_FOR_PRISMION_DISCRIMINATION")
    if abc and gnn and fixed and max(m(abc, "phase_lock_accuracy", 0.0), m(gnn, "phase_lock_accuracy", 0.0)) >= m(fixed, "phase_lock_accuracy", 0.0) - 0.02:
        labels.append("CANONICAL_MESSAGE_PASSING_RECOVERS")
    if fixed and m(fixed, "phase_lock_accuracy", 0.0) < 0.70:
        labels.append("TRANSFER_FAILS")
    return sorted(set(labels or ["TRANSFER_PARTIAL"]))


def write_outputs(out_dir: Path, args: argparse.Namespace, results: list[JobResult], status: str, jobs: int, sample_rows: list[TransferExample]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdicts(agg, results) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    write_jsonl(out_dir / "transfer_cases.jsonl", [asdict(row) for row in sample_rows[:128]])
    deltas = [
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_BILINEAR_PHASE_LOOP", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN", "phase_lock_accuracy"),
        paired(results, "FIXED_PRISMION_PHASE_LOCK_LOOP", "LEARNED_PRISMION_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
    ]
    write_jsonl(out_dir / "paired_seed_deltas.jsonl", deltas)
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
            "noise_level": args.noise_level,
            "mixed_cancel_phase": args.mixed_cancel_phase,
        },
        "aggregate": agg,
        "paired_deltas": deltas,
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "heldout_gate_metrics.jsonl", [{"arm": row["arm"], "heldout_gate_accuracy": m(row, "heldout_gate_accuracy")} for row in agg.values()])
    write_jsonl(out_dir / "noisy_gate_metrics.jsonl", [{"arm": row["arm"], "noisy_gate_accuracy": m(row, "noisy_gate_accuracy")} for row in agg.values()])
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_002_TRANSFER Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        "",
        "| Arm | Acc | Transfer | Pair | HeldoutGate | NoisyGate | LabelShuffle | GateShuffle | Params | Trainable | Operator |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        audit = row.get("audit", {})
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` | `{:.0f}` | `{}` |".format(
                row["arm"],
                m(row, "phase_lock_accuracy"),
                m(row, "transfer_accuracy"),
                m(row, "paired_counterfactual_accuracy"),
                m(row, "heldout_gate_accuracy"),
                m(row, "noisy_gate_accuracy"),
                m(row, "label_shuffle_control"),
                m(row, "gate_angle_shuffle_control"),
                m(row, "parameter_count"),
                m(row, "trainable_parameter_count"),
                audit.get("operator_type", ""),
            )
        )
    lines.append("\n## Paired Deltas\n")
    for delta in deltas:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}`: mean `{delta['mean_delta']:.4f}`, lower95 `{delta['lower95_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int, results: list[JobResult]) -> None:
    deltas = [
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "LOCAL_BILINEAR_PHASE_LOOP", "phase_lock_accuracy"),
        paired(results, "LEARNED_PRISMION_PHASE_LOCK_LOOP", "COMPLEX_MULTIPLY_GNN", "phase_lock_accuracy"),
        paired(results, "FIXED_PRISMION_PHASE_LOCK_LOOP", "LEARNED_PRISMION_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
    ]
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_002_TRANSFER Result",
        "",
        "## Latest Run",
        "",
        "```text",
        f"out={args.out}",
        f"phase_classes={args.phase_classes}",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"width={args.width}",
        f"jobs={jobs}",
        f"device={args.device}",
        f"noise_level={args.noise_level}",
        f"mixed_cancel_phase={args.mixed_cancel_phase}",
        f"completed_jobs={completed}",
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Summary Table",
        "",
        "| Arm | Acc | Pair | HeldoutGate | NoisyGate | LabelShuffle | GateShuffle | Params | Trainable | Operator |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        audit = row.get("audit", {})
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` | `{:.0f}` | `{}` |".format(
                row["arm"],
                m(row, "phase_lock_accuracy"),
                m(row, "paired_counterfactual_accuracy"),
                m(row, "heldout_gate_accuracy"),
                m(row, "noisy_gate_accuracy"),
                m(row, "label_shuffle_control"),
                m(row, "gate_angle_shuffle_control"),
                m(row, "parameter_count"),
                m(row, "trainable_parameter_count"),
                audit.get("operator_type", ""),
            )
        )
    lines.append("\n## Matched Deltas\n")
    for delta in deltas:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}`: mean `{delta['mean_delta']:.4f}`, lower95 `{delta['lower95_delta']:.4f}`, std `{delta['std_delta']:.4f}`, min `{delta['min_delta']:.4f}`, max `{delta['max_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run separates the previous zero-parameter fixed phase primitive from learned Prismion-style and generic complex-multiply baselines.",
            "",
            "If `COMPLEX_MULTIPLY_SUFFICIENT` appears, the correct interpretation is that local complex multiplication is the needed primitive, not that the full Prismion architecture is uniquely required.",
            "",
            "## Claim Boundary",
            "",
            "This does not prove consciousness, full VRAXION, language grounding, or general reasoning. It only tests phase-lock transfer and whether the useful primitive is Prismion-specific or generic local complex multiplication.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, Any]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(config, seed) for config in configs for seed in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "queue.json", [{**asdict(config), "seed": seed} for config, seed in queue])
    _, sample = build_dataset(2026, min(args.train_examples, 64), min(args.eval_examples, 128), args.phase_classes, args.width, args)
    write_jsonl(args.out / "examples_sample.jsonl", [asdict(row) for row in sample[:64]])
    progress = args.out / "progress.jsonl"
    metrics = args.out / "metrics.jsonl"
    job_progress = args.out / "job_progress"
    append_jsonl(progress, {"time": now_iso(), "event": "run_start", "total_jobs": len(queue), "jobs": jobs})
    results: list[JobResult] = []
    write_outputs(args.out, args, results, "partial", jobs, sample)
    if jobs <= 1:
        for config, seed in queue:
            append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
            result = run_job(config, seed, args, job_progress)
            results.append(result)
            append_jsonl(metrics, asdict(result))
            append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results)})
            write_outputs(args.out, args, results, "partial", jobs, sample)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = set()
            meta = {}
            for config, seed in queue:
                append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
                fut = pool.submit(run_job, config, seed, args, job_progress)
                pending.add(fut)
                meta[fut] = (config, seed)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, args, results, "partial", jobs, sample)
                    continue
                for fut in done:
                    config, seed = meta[fut]
                    result = fut.result()
                    results.append(result)
                    append_jsonl(metrics, asdict(result))
                    append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, args, results, "partial", jobs, sample)
    agg, labels = write_outputs(args.out, args, results, "complete", jobs, sample)
    append_jsonl(progress, {"time": now_iso(), "event": "run_complete", "completed_jobs": len(results), "verdict": labels})
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
    write_doc(agg, labels, args, jobs, len(results), results)
    print(json.dumps({"verdict": labels, "out": str(args.out), "completed_jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
