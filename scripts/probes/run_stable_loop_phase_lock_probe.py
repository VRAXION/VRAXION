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
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK_RESULT.md"

MAX_GRID = 16
INPUT_CHANNELS = 6
WALL, TARGET, SRC_REAL, SRC_IMAG, GATE_REAL, GATE_IMAG = range(INPUT_CHANNELS)
EVAL_STEPS = (4, 8, 16, 24, 32)
TRAIN_STEPS = (8, 16, 24, 32)
READOUT_THRESHOLD = 0.35
PHASE_CLASSES_DEFAULT = 4

ALL_ARMS = (
    "ORACLE_PHASE_LOCK_S",
    "SUMMARY_DIRECT_HEAD",
    "TARGET_MARKER_ONLY",
    "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK",
    "HARD_WALL_ABC_PHASE_LOCK_LOOP",
    "HARD_WALL_PRISMION_PHASE_LOCK_LOOP",
    "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK",
)
ORACLE_ARMS = {"ORACLE_PHASE_LOCK_S"}
TRAINABLE_ARMS = set(ALL_ARMS) - ORACLE_ARMS - {"LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "HARD_WALL_PRISMION_PHASE_LOCK_LOOP"}


@dataclass(frozen=True)
class LockExample:
    case_id: str
    grid: list[list[list[float]]]
    grid_size: int
    target: tuple[int, int]
    phase_classes: int
    labels_by_s: dict[int, int]
    full_label: int
    path_len: int
    bucket: str
    contrast_group_id: str | None
    source_phase: int
    gate_phases: tuple[int, ...]


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
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-classes", type=int, choices=[2, 4], default=4)
    parser.add_argument("--seeds", default="2026,2027,2028")
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


def phase_bucket(z: complex, phase_classes: int) -> int:
    if abs(z) < READOUT_THRESHOLD:
        return 0
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / phase_classes))) % phase_classes + 1


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


def bucket_len(path_len: int) -> str:
    if path_len <= 8:
        return "1-8"
    if path_len <= 16:
        return "9-16"
    return "17-24"


def blank_grid(n: int) -> np.ndarray:
    grid = np.zeros((INPUT_CHANNELS, MAX_GRID, MAX_GRID), dtype=np.float32)
    grid[WALL] = 1.0
    grid[WALL, :n, :n] = 1.0
    grid[GATE_REAL, :n, :n] = 1.0
    return grid


def carve(grid: np.ndarray, cell: tuple[int, int]) -> None:
    grid[WALL, cell[0], cell[1]] = 0.0


def random_path(n: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
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
    # Fallback deterministic corridor.
    path = []
    r = n // 2
    for c in range(1, min(n - 1, length + 2)):
        path.append((r, c))
    while len(path) < length + 1:
        r = max(1, r - 1)
        path.append((r, path[-1][1]))
    return path[: length + 1]


def make_example(case_id: str, rng: random.Random, phase_classes: int, n: int, group: str | None = None, forced_gates: list[int] | None = None) -> LockExample:
    length = rng.choice([6, 10, 14, 18, 22])
    length = min(length, n * n // 3)
    path = random_path(n, length, rng)
    grid = blank_grid(n)
    for cell in path:
        carve(grid, cell)
    source_phase = rng.randrange(phase_classes)
    gates = forced_gates or [rng.randrange(phase_classes) for _ in path[1:]]
    sr, sc = path[0]
    z = unit_phase(source_phase, phase_classes)
    grid[SRC_REAL, sr, sc] = float(z.real)
    grid[SRC_IMAG, sr, sc] = float(z.imag)
    for cell, gate_phase in zip(path[1:], gates):
        gz = unit_phase(gate_phase, phase_classes)
        grid[GATE_REAL, cell[0], cell[1]] = float(gz.real)
        grid[GATE_IMAG, cell[0], cell[1]] = float(gz.imag)
    target = path[-1]
    grid[TARGET, target[0], target[1]] = 1.0
    labels, full_label = phase_lock_oracle_labels(grid, phase_classes)
    return LockExample(case_id, grid.tolist(), n, target, phase_classes, labels, full_label, length, bucket_len(length), group, source_phase, tuple(gates))


def build_dataset(seed: int, train_examples: int, eval_examples: int, phase_classes: int) -> tuple[list[LockExample], list[LockExample]]:
    rng = random.Random(seed)

    def rows(count: int, split: str) -> list[LockExample]:
        out: list[LockExample] = []
        idx = 0
        while len(out) < count:
            n = rng.choice([8, 12, 16]) if split == "eval" else rng.choice([8, 12])
            if idx % 8 in {6, 7}:
                group = f"{split}_pair_{idx // 8:06d}"
                gates = [rng.randrange(phase_classes) for _ in range(rng.choice([10, 14, 18]))]
                if idx % 8 == 7:
                    gates = list(gates)
                    gates[len(gates) // 2] = (gates[len(gates) // 2] + 1) % phase_classes
                out.append(make_example(f"{split}_{idx:06d}", rng, phase_classes, n, group, gates))
            else:
                out.append(make_example(f"{split}_{idx:06d}", rng, phase_classes, n))
            idx += 1
        return out

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
    mag = vec.pow(2).sum(dim=1).add(1e-8).sqrt()
    none = (READOUT_THRESHOLD - mag) * 12.0
    phase_logits = (vec @ phases.T - READOUT_THRESHOLD) * 12.0
    return torch.cat([none.unsqueeze(1), phase_logits], dim=1)


def examples_to_tensors(rows: list[LockExample], device: torch.device) -> dict[str, Any]:
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


class SummaryHead(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del target, steps
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled), {"post_mask_wall_leak": torch.zeros((), device=x.device), "pre_mask_wall_pressure": torch.zeros((), device=x.device)}


class TargetOnly(nn.Module):
    def __init__(self, phase_classes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(phase_classes + 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del target, steps
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1), {}


class UntiedCNN(nn.Module):
    def __init__(self, width: int, phase_classes: int):
        super().__init__()
        self.phase_classes = phase_classes
        self.inp = nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(width + INPUT_CHANNELS, width, 3, padding=1) for _ in range(max(EVAL_STEPS))])
        self.vec = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.inp(x))
        for layer in self.layers[:steps]:
            h = torch.tanh(layer(torch.cat([h, x], dim=1)))
        vec_map = self.vec(h) * (1.0 - x[:, WALL : WALL + 1])
        return fixed_logits(target_cell(vec_map, target), self.phase_classes), {"post_mask_wall_leak": torch.zeros((), device=x.device), "pre_mask_wall_pressure": torch.zeros((), device=x.device)}


class PhaseLockLoop(nn.Module):
    def __init__(self, arm: str, width: int, phase_classes: int):
        super().__init__()
        self.arm = arm
        self.phase_classes = phase_classes
        self.learned = arm == "HARD_WALL_ABC_PHASE_LOCK_LOOP"
        self.h = nn.Conv2d(INPUT_CHANNELS + 2, width, 3, padding=1) if self.learned else None
        self.vec = nn.Conv2d(width, 2, 1) if self.learned else None

    def forward(self, x: torch.Tensor, target: torch.Tensor, steps: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        free = 1.0 - x[:, WALL : WALL + 1]
        real = x[:, SRC_REAL : SRC_REAL + 1] * free
        imag = x[:, SRC_IMAG : SRC_IMAG + 1] * free
        state_r = real.clone()
        state_i = imag.clone()
        written = real.square().add(imag.square()).sqrt() > 1e-4
        if self.arm == "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK":
            for _ in range(steps):
                inc_r, inc_i = neighbor_sum(real, imag)
                real, imag = inc_r * free, inc_i * free
                active = (real.square().add(imag.square()).sqrt() > 1e-4) & (~written)
                state_r = torch.where(active, real, state_r)
                state_i = torch.where(active, imag, state_i)
                written = written | active
        elif self.arm == "HARD_WALL_PRISMION_PHASE_LOCK_LOOP":
            gate_r = x[:, GATE_REAL : GATE_REAL + 1]
            gate_i = x[:, GATE_IMAG : GATE_IMAG + 1]
            for _ in range(steps):
                inc_r, inc_i = neighbor_sum(real, imag)
                # Complex multiplication by destination cell gate.
                next_r = inc_r * gate_r - inc_i * gate_i
                next_i = inc_r * gate_i + inc_i * gate_r
                real, imag = next_r * free, next_i * free
                active = (real.square().add(imag.square()).sqrt() > 1e-4) & (~written)
                state_r = torch.where(active, real, state_r)
                state_i = torch.where(active, imag, state_i)
                written = written | active
        else:
            h = torch.zeros((x.shape[0], self.h.out_channels, x.shape[2], x.shape[3]), device=x.device)
            for _ in range(steps):
                inc_r, inc_i = neighbor_sum(real, imag)
                h = torch.tanh(self.h(torch.cat([x, inc_r, inc_i], dim=1)) + 0.5 * h)
                out = torch.tanh(self.vec(h))
                real, imag = out[:, 0:1] * free, out[:, 1:2] * free
                active = (real.square().add(imag.square()).sqrt() > 1e-4) & (~written)
                state_r = torch.where(active, real, state_r)
                state_i = torch.where(active, imag, state_i)
                written = written | active
        vec_map = torch.cat([state_r, state_i], dim=1)
        wall = x[:, WALL : WALL + 1] > 0.5
        leak = ((vec_map.pow(2).sum(dim=1, keepdim=True).sqrt() > 0.5) & wall).float().mean()
        return fixed_logits(target_cell(vec_map, target), self.phase_classes), {"post_mask_wall_leak": leak, "pre_mask_wall_pressure": torch.zeros((), device=x.device)}


def make_model(config: Config, phase_classes: int) -> nn.Module | None:
    if config.arm == "SUMMARY_DIRECT_HEAD":
        return SummaryHead(config.width, phase_classes)
    if config.arm == "TARGET_MARKER_ONLY":
        return TargetOnly(phase_classes)
    if config.arm == "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK":
        return UntiedCNN(config.width, phase_classes)
    if config.arm in {"LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "HARD_WALL_PRISMION_PHASE_LOCK_LOOP"}:
        return PhaseLockLoop(config.arm, config.width, phase_classes)
    return None


def iter_batches(n: int, batch_size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [np.array(idx[i : i + batch_size], dtype=np.int64) for i in range(0, n, batch_size)]


def train_model(config: Config, rows: list[LockExample], args: argparse.Namespace, device: torch.device, progress_path: Path, seed: int) -> nn.Module | None:
    model = make_model(config, args.phase_classes)
    if model is None:
        return None
    model.to(device)
    if config.arm not in TRAINABLE_ARMS:
        return model
    batch = examples_to_tensors(rows, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(seed + 9903)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for idx in iter_batches(len(rows), args.batch_size, rng):
            steps = rng.choice(TRAIN_STEPS)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(batch["x"][idx], batch["target"][idx], steps)
            loss = F.cross_entropy(logits, batch["labels"][steps][idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {config.arm} seed={seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), **asdict(config), "seed": seed})
    return model


def oracle_pred(rows: list[LockExample], steps: int) -> np.ndarray:
    return np.array([row.labels_by_s[steps] for row in rows], dtype=np.int64)


def predict(model: nn.Module | None, config: Config, rows: list[LockExample], args: argparse.Namespace, device: torch.device, steps: int) -> tuple[np.ndarray, dict[str, float]]:
    if config.arm == "ORACLE_PHASE_LOCK_S":
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


def acc(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean()) if len(gold) else math.nan


def pair_acc(rows: list[LockExample], correct: np.ndarray) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(rows, correct):
        if row.contrast_group_id:
            groups[row.contrast_group_id].append(bool(ok))
    return float(np.mean([all(v) for v in groups.values()])) if groups else math.nan


def evaluate(config: Config, model: nn.Module | None, eval_rows: list[LockExample], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    metrics = {}
    accs = []
    stats: dict[str, list[float]] = defaultdict(list)
    for steps in EVAL_STEPS:
        pred, st = predict(model, config, eval_rows, args, device, steps)
        gold = np.array([row.labels_by_s[steps] for row in eval_rows], dtype=np.int64)
        correct = pred == gold
        metrics[f"phase_lock_accuracy_s{steps}"] = acc(pred, gold)
        metrics[f"none_vs_phase_accuracy_s{steps}"] = acc((pred > 0).astype(np.int64), (gold > 0).astype(np.int64))
        mask = gold > 0
        metrics[f"phase_bucket_accuracy_s{steps}"] = acc(pred[mask], gold[mask]) if mask.any() else math.nan
        metrics[f"long_path_accuracy_s{steps}"] = float(np.mean([ok for ok, row in zip(correct, eval_rows) if row.bucket == "17-24"])) if any(row.bucket == "17-24" for row in eval_rows) else math.nan
        metrics[f"same_target_neighborhood_pair_accuracy_s{steps}"] = pair_acc(eval_rows, correct)
        accs.append(metrics[f"phase_lock_accuracy_s{steps}"])
        for k, v in st.items():
            stats[k].append(v)
    metrics["phase_lock_accuracy"] = float(np.mean(accs))
    metrics["target_phase_accuracy"] = metrics["phase_lock_accuracy"]
    metrics["none_vs_phase_accuracy"] = float(np.mean([metrics[f"none_vs_phase_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["phase_bucket_accuracy"] = float(np.nanmean([metrics[f"phase_bucket_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["long_path_accuracy"] = float(np.nanmean([metrics[f"long_path_accuracy_s{s}"] for s in EVAL_STEPS]))
    metrics["same_target_neighborhood_pair_accuracy"] = float(np.nanmean([metrics[f"same_target_neighborhood_pair_accuracy_s{s}"] for s in EVAL_STEPS]))
    labels = [row.labels_by_s[24] for row in eval_rows]
    counts = Counter(labels)
    metrics["majority_baseline"] = max(counts.values()) / max(1, len(labels))
    metrics["random_baseline"] = 1.0 / (args.phase_classes + 1)
    for k, v in stats.items():
        metrics[k] = float(np.mean(v))
    metrics.setdefault("post_mask_wall_leak", 0.0)
    metrics.setdefault("pre_mask_wall_pressure", 0.0)
    metrics["parameter_count"] = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    return metrics


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path) -> JobResult:
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples, args.phase_classes)
    progress = progress_root / f"{config.arm}_w{config.width}_seed{seed}.jsonl"
    append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
    model = train_model(config, train_rows, args, device, progress, seed)
    metrics = evaluate(config, model, eval_rows, args, device)
    append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "metrics": metrics})
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return JobResult(config.arm, config.width, args.phase_classes, seed, metrics)


def build_queue(args: argparse.Namespace) -> list[Config]:
    arms = parse_csv(args.arms) if args.arms else list(ALL_ARMS)
    out = []
    for arm in arms:
        width = 0 if arm in ORACLE_ARMS or arm in {"TARGET_MARKER_ONLY", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "HARD_WALL_PRISMION_PHASE_LOCK_LOOP"} else args.width
        out.append(Config(arm, width))
    return out


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        groups[f"{result.arm}|w{result.width}"].append(result)
    agg = {}
    for key, rows in groups.items():
        metric_keys = sorted({k for row in rows for k in row.metrics})
        means = {}
        stds = {}
        for mk in metric_keys:
            vals = [row.metrics[mk] for row in rows if mk in row.metrics and not math.isnan(row.metrics[mk])]
            means[mk] = float(np.mean(vals)) if vals else math.nan
            stds[mk] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg[key] = {"arm": rows[0].arm, "width": rows[0].width, "phase_classes": rows[0].phase_classes, "seeds": [r.seed for r in rows], "metric_mean": means, "metric_std": stds}
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
    return {"left": left, "right": right, "metric": key, "deltas": vals, "mean_delta": float(np.mean(nums)) if nums else math.nan, "std_delta": float(np.std(nums)) if len(nums) > 1 else 0.0, "min_delta": float(np.min(nums)) if nums else math.nan, "max_delta": float(np.max(nums)) if nums else math.nan, "positive_seed_count": int(sum(v > 0 for v in nums)), "paired_seed_count": len(nums)}


def verdicts(agg: dict[str, dict[str, Any]], results: list[JobResult]) -> list[str]:
    labels = []
    oracle = best(agg, "ORACLE_PHASE_LOCK_S")
    prism = best(agg, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP")
    abc = best(agg, "HARD_WALL_ABC_PHASE_LOCK_LOOP")
    gnn = best(agg, "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK")
    summary = best(agg, "SUMMARY_DIRECT_HEAD")
    target = best(agg, "TARGET_MARKER_ONLY")
    if oracle and m(oracle, "phase_lock_accuracy", 0.0) >= 0.99:
        labels.append("PHASE_LOCK_TASK_VALID")
    if summary and m(summary, "phase_lock_accuracy", 0.0) > m(summary, "majority_baseline", 0.0) + 0.10:
        labels.append("SUMMARY_SHORTCUT_WARNING")
    if target and m(target, "phase_lock_accuracy", 0.0) > m(target, "majority_baseline", 0.0) + 0.10:
        labels.append("TARGET_MARKER_SHORTCUT")
    if prism and abc and gnn:
        d_abc = paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy")
        d_gnn = paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy")
        if d_abc["mean_delta"] > 0.10 and d_gnn["mean_delta"] > 0.10 and m(prism, "post_mask_wall_leak", 1.0) <= 0.02:
            labels.append("PRISMION_PHASE_LOCK_POSITIVE")
        if m(abc, "phase_lock_accuracy", 0.0) >= m(prism, "phase_lock_accuracy", 0.0) - 0.03 or m(gnn, "phase_lock_accuracy", 0.0) >= m(prism, "phase_lock_accuracy", 0.0) - 0.03:
            labels.append("CANONICAL_BASELINE_SUFFICIENT")
    return sorted(set(labels or ["PHASE_LOCK_PARTIAL"]))


def write_outputs(out_dir: Path, args: argparse.Namespace, results: list[JobResult], status: str, jobs: int, sample_rows: list[LockExample]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdicts(agg, results) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    write_jsonl(out_dir / "phase_lock_cases.jsonl", [asdict(r) for r in sample_rows[:128]])
    deltas = [
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy"),
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK", "phase_lock_accuracy"),
    ]
    write_jsonl(out_dir / "paired_seed_deltas.jsonl", deltas)
    summary = {"status": status, "verdict": labels, "completed_jobs": len(results), "config": {"phase_classes": args.phase_classes, "seeds": args.seeds, "train_examples": args.train_examples, "eval_examples": args.eval_examples, "epochs": args.epochs, "width": args.width, "jobs": jobs, "device": args.device}, "aggregate": agg, "paired_deltas": deltas}
    write_json(out_dir / "summary.json", summary)
    lines = ["# STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK Report", "", f"- Status: `{status}`", f"- Verdict: `{', '.join(labels)}`", f"- Completed jobs: `{len(results)}`", "", "| Arm | Acc | PhaseBucket | Long | Pair | Wall | Params |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(row["arm"], m(row, "phase_lock_accuracy"), m(row, "phase_bucket_accuracy"), m(row, "long_path_accuracy"), m(row, "same_target_neighborhood_pair_accuracy"), m(row, "post_mask_wall_leak"), m(row, "parameter_count")))
    lines.append("\n## Paired Deltas\n")
    for delta in deltas:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}`: mean `{delta['mean_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int, results: list[JobResult]) -> None:
    deltas = [
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "HARD_WALL_ABC_PHASE_LOCK_LOOP", "phase_lock_accuracy"),
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK", "phase_lock_accuracy"),
        paired(results, "HARD_WALL_PRISMION_PHASE_LOCK_LOOP", "UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK", "phase_lock_accuracy"),
    ]
    lines = ["# STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK Result", "", "## Latest Run", "", "```text", f"phase_classes={args.phase_classes}", f"seeds={args.seeds}", f"train_examples={args.train_examples}", f"eval_examples={args.eval_examples}", f"epochs={args.epochs}", f"width={args.width}", f"jobs={jobs}", f"device={args.device}", f"completed_jobs={completed}", "```", "", "## Verdict", "", "```json", json.dumps(labels, indent=2), "```", "", "| Arm | Acc | Long | Pair | Wall |", "|---|---:|---:|---:|---:|"]
    for row in sorted(agg.values(), key=lambda r: str(r["arm"])):
        lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(row["arm"], m(row, "phase_lock_accuracy"), m(row, "long_path_accuracy"), m(row, "same_target_neighborhood_pair_accuracy"), m(row, "post_mask_wall_leak")))
    lines.append("\n## Paired Deltas\n")
    for delta in deltas:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}`: mean `{delta['mean_delta']:.4f}`, std `{delta['std_delta']:.4f}`, min `{delta['min_delta']:.4f}`, max `{delta['max_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    lines.extend(["", "## Claim Boundary", "", "This proves or disproves only the local phase-lock primitive in this toy task. It does not prove consciousness, full VRAXION, language grounding, or general reasoning."])
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, Any]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(c, s) for c in configs for s in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "queue.json", [{**asdict(c), "seed": s} for c, s in queue])
    _, sample = build_dataset(2026, min(args.train_examples, 64), min(args.eval_examples, 128), args.phase_classes)
    write_jsonl(args.out / "examples_sample.jsonl", [asdict(r) for r in sample[:64]])
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
