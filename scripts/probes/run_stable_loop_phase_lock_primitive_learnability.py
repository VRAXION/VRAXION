from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
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
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY_RESULT.md"

PHASE_CLASSES = 4
COMPOSITION_LENGTHS = (1, 2, 4, 8, 16, 32)
ANGLE_REPRS = ("gate_as_angle_theta", "gate_as_cos_sin", "gate_as_complex_pair")
GATE_NORMS = ("normalized_gate", "unnormalized_gate")
NORM_POLICIES = ("no_renorm_between_steps", "renorm_to_unit_magnitude_between_steps", "learned_or_raw_magnitude_preserved")
READOUT_THRESHOLD = 0.2

ALL_ARMS = (
    "FIXED_COMPLEX_MULTIPLY_TEACHER",
    "CURRENT_FACTOR_CELL_SINGLE_STEP",
    "CURRENT_FACTOR_CELL_MULTI_STEP",
    "RICH_PHASE_CELL_SINGLE_STEP",
    "RICH_PHASE_CELL_MULTI_STEP",
    "LOCAL_BILINEAR_SINGLE_STEP",
    "LOCAL_BILINEAR_MULTI_STEP",
    "TINY_MLP_BASELINE",
    "COMPLEX_MULTIPLY_GNN",
    "RANDOM_INIT_PHASE_CELL_TRANSFER",
    "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER",
    "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER",
)
TEACHER_ARMS = {"FIXED_COMPLEX_MULTIPLY_TEACHER", "COMPLEX_MULTIPLY_GNN"}
TRANSFER_ARMS = {
    "RANDOM_INIT_PHASE_CELL_TRANSFER",
    "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER",
    "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER",
}


@dataclass(frozen=True)
class Config:
    arm: str
    gate_repr: str
    gate_norm: str
    norm_policy: str


@dataclass
class JobResult:
    config: dict[str, str]
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
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--jobs", default="6")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument("--transfer-grid", type=int, default=16)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def complex_mul(a_re: torch.Tensor, a_im: torch.Tensor, b_re: torch.Tensor, b_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


def renorm_like(re: torch.Tensor, im: torch.Tensor, ref_re: torch.Tensor, ref_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_mag = torch.sqrt(ref_re.square() + ref_im.square() + 1e-8)
    mag = torch.sqrt(re.square() + im.square() + 1e-8)
    scale = target_mag / mag.clamp_min(1e-6)
    return re * scale, im * scale


def wrapped_angle_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(pred - true), torch.cos(pred - true)).abs()


def phase_bucket_from_vec(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    mag = torch.sqrt(re.square() + im.square() + 1e-8)
    theta = torch.atan2(im, re)
    theta = torch.remainder(theta + 2.0 * math.pi, 2.0 * math.pi)
    bucket = torch.round(theta / (2.0 * math.pi / PHASE_CLASSES)).long() % PHASE_CLASSES
    return torch.where(mag < READOUT_THRESHOLD, torch.zeros_like(bucket), bucket + 1)


def make_primitive_data(n: int, max_steps: int, seed: int, gate_norm: str, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mag = 0.5 + torch.rand(n, generator=gen)
    theta_z = (torch.rand(n, generator=gen) * 2.0 - 1.0) * math.pi
    z_re = mag * torch.cos(theta_z)
    z_im = mag * torch.sin(theta_z)
    gate_theta = (torch.rand(n, max_steps, generator=gen) * 2.0 - 1.0) * math.pi
    gate_mag = torch.ones(n, max_steps)
    if gate_norm == "unnormalized_gate":
        gate_mag = 0.5 + torch.rand(n, max_steps, generator=gen)
    gate_re = gate_mag * torch.cos(gate_theta)
    gate_im = gate_mag * torch.sin(gate_theta)
    unit_re = torch.cos(gate_theta)
    unit_im = torch.sin(gate_theta)
    y_re = z_re.clone()
    y_im = z_im.clone()
    ys_re = []
    ys_im = []
    for step in range(max_steps):
        y_re, y_im = complex_mul(y_re, y_im, unit_re[:, step], unit_im[:, step])
        ys_re.append(y_re)
        ys_im.append(y_im)
    return {
        "z_re": z_re.to(device),
        "z_im": z_im.to(device),
        "gate_theta": gate_theta.to(device),
        "gate_re": gate_re.to(device),
        "gate_im": gate_im.to(device),
        "unit_re": unit_re.to(device),
        "unit_im": unit_im.to(device),
        "target_re": torch.stack(ys_re, dim=1).to(device),
        "target_im": torch.stack(ys_im, dim=1).to(device),
    }


def gate_features(data: dict[str, torch.Tensor], gate_repr: str, step: int) -> torch.Tensor:
    if gate_repr == "gate_as_angle_theta":
        theta = data["gate_theta"][:, step]
        return theta.unsqueeze(1)
    if gate_repr == "gate_as_cos_sin":
        theta = data["gate_theta"][:, step]
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return torch.stack([data["gate_re"][:, step], data["gate_im"][:, step]], dim=1)


def gate_unit(data: dict[str, torch.Tensor], step: int) -> tuple[torch.Tensor, torch.Tensor]:
    return data["unit_re"][:, step], data["unit_im"][:, step]


def apply_norm_policy(re: torch.Tensor, im: torch.Tensor, ref_re: torch.Tensor, ref_im: torch.Tensor, policy: str) -> tuple[torch.Tensor, torch.Tensor]:
    if policy == "renorm_to_unit_magnitude_between_steps":
        return renorm_like(re, im, ref_re, ref_im)
    return re, im


class BaseCell(nn.Module):
    gate_repr: str

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CurrentFactorCell(BaseCell):
    def __init__(self, gate_repr: str):
        super().__init__()
        self.gate_repr = gate_repr
        self.scale = nn.Parameter(torch.tensor([0.9, 0.9], dtype=torch.float32))
        self.mix = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def _gate_pair(self, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_repr == "gate_as_angle_theta":
            return torch.cos(gate[:, 0]), torch.sin(gate[:, 0])
        return gate[:, 0], gate[:, 1]

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g_re, g_im = self._gate_pair(gate)
        cm_re, cm_im = complex_mul(z_re, z_im, g_re, g_im)
        cj_re, cj_im = complex_mul(z_re, z_im, g_re, -g_im)
        return self.scale[0] * cm_re + self.mix[0] * cj_re + self.bias[0], self.scale[1] * cm_im + self.mix[1] * cj_im + self.bias[1]


class RichPhaseCell(BaseCell):
    def __init__(self, gate_repr: str, hidden: int = 32):
        super().__init__()
        self.gate_repr = gate_repr
        gate_dim = 1 if gate_repr == "gate_as_angle_theta" else 2
        self.net = nn.Sequential(nn.Linear(2 + gate_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z_re.unsqueeze(1), z_im.unsqueeze(1), gate], dim=1))
        return out[:, 0], out[:, 1]


class LocalBilinearCell(BaseCell):
    def __init__(self, gate_repr: str):
        super().__init__()
        self.gate_repr = gate_repr
        self.lin = nn.Linear(8 if gate_repr != "gate_as_angle_theta" else 6, 2)

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_repr == "gate_as_angle_theta":
            g_re = torch.cos(gate[:, 0])
            g_im = torch.sin(gate[:, 0])
        else:
            g_re, g_im = gate[:, 0], gate[:, 1]
        feats = [z_re * g_re, z_re * g_im, z_im * g_re, z_im * g_im, z_re, z_im]
        if self.gate_repr != "gate_as_angle_theta":
            feats.extend([g_re, g_im])
        out = self.lin(torch.stack(feats, dim=1))
        return out[:, 0], out[:, 1]


class TinyMLPCell(BaseCell):
    def __init__(self, gate_repr: str):
        super().__init__()
        self.gate_repr = gate_repr
        gate_dim = 1 if gate_repr == "gate_as_angle_theta" else 2
        self.net = nn.Sequential(nn.Linear(2 + gate_dim, 24), nn.ReLU(), nn.Linear(24, 2))

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z_re.unsqueeze(1), z_im.unsqueeze(1), gate], dim=1))
        return out[:, 0], out[:, 1]


class FixedComplexCell(BaseCell):
    def __init__(self, gate_repr: str):
        super().__init__()
        self.gate_repr = gate_repr

    def step(self, z_re: torch.Tensor, z_im: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_repr == "gate_as_angle_theta":
            g_re, g_im = torch.cos(gate[:, 0]), torch.sin(gate[:, 0])
        else:
            mag = torch.sqrt(gate[:, 0].square() + gate[:, 1].square() + 1e-8)
            g_re, g_im = gate[:, 0] / mag, gate[:, 1] / mag
        return complex_mul(z_re, z_im, g_re, g_im)


def make_cell(arm: str, gate_repr: str) -> BaseCell:
    if arm in {"FIXED_COMPLEX_MULTIPLY_TEACHER", "COMPLEX_MULTIPLY_GNN"}:
        return FixedComplexCell(gate_repr)
    if arm.startswith("CURRENT_FACTOR_CELL") or arm.startswith("RANDOM_INIT") or arm.startswith("PRETRAINED"):
        return CurrentFactorCell(gate_repr)
    if arm.startswith("RICH_PHASE_CELL"):
        return RichPhaseCell(gate_repr)
    if arm.startswith("LOCAL_BILINEAR"):
        return LocalBilinearCell(gate_repr)
    return TinyMLPCell(gate_repr)


def roll_cell(cell: BaseCell, data: dict[str, torch.Tensor], steps: int, gate_repr: str, norm_policy: str, teacher_forced: bool) -> tuple[torch.Tensor, torch.Tensor]:
    z_re = data["z_re"]
    z_im = data["z_im"]
    init_re, init_im = z_re, z_im
    for step in range(steps):
        gate = gate_features(data, gate_repr, step)
        if teacher_forced and step > 0:
            z_re = data["target_re"][:, step - 1]
            z_im = data["target_im"][:, step - 1]
        z_re, z_im = cell.step(z_re, z_im, gate)
        z_re, z_im = apply_norm_policy(z_re, z_im, init_re, init_im, norm_policy)
    return z_re, z_im


def primitive_loss(cell: BaseCell, data: dict[str, torch.Tensor], steps: int, gate_repr: str, norm_policy: str, teacher_forced: bool) -> torch.Tensor:
    pred_re, pred_im = roll_cell(cell, data, steps, gate_repr, norm_policy, teacher_forced)
    true_re = data["target_re"][:, steps - 1]
    true_im = data["target_im"][:, steps - 1]
    return F.mse_loss(pred_re, true_re) + F.mse_loss(pred_im, true_im)


def train_cell(cell: BaseCell, arm: str, data: dict[str, torch.Tensor], args: argparse.Namespace, gate_repr: str, norm_policy: str, progress: Path, seed: int) -> dict[str, float]:
    if arm in TEACHER_ARMS or arm == "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER":
        return {"train_steps": 0.0, "final_train_loss": 0.0, "grad_norm_mean": 0.0, "grad_norm_max": 0.0, "activation_norm_mean": 0.0, "activation_norm_max": 0.0}
    opt = torch.optim.AdamW(cell.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = random.Random(seed + 171)
    n = int(data["z_re"].shape[0])
    grad_norms: list[float] = []
    act_norms: list[float] = []
    losses: list[float] = []
    step_count = 0
    max_train_step = 1 if "SINGLE_STEP" in arm else max(COMPOSITION_LENGTHS)
    for epoch in range(1, args.epochs + 1):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, args.batch_size):
            idx = torch.tensor(order[start : start + args.batch_size], dtype=torch.long, device=data["z_re"].device)
            batch = {k: v[idx] for k, v in data.items()}
            steps = 1 if "SINGLE_STEP" in arm else rng.choice(COMPOSITION_LENGTHS)
            steps = min(steps, max_train_step)
            teacher_forced = bool(rng.getrandbits(1))
            opt.zero_grad(set_to_none=True)
            pred_re, pred_im = roll_cell(cell, batch, steps, gate_repr, norm_policy, teacher_forced)
            loss = F.mse_loss(pred_re, batch["target_re"][:, steps - 1]) + F.mse_loss(pred_im, batch["target_im"][:, steps - 1])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {arm} seed={seed}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(cell.parameters(), 5.0)
            opt.step()
            step_count += 1
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad.detach().cpu()))
            act_norms.append(float((pred_re.abs() + pred_im.abs()).mean().detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            append_jsonl(progress, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses[-max(1, n // args.batch_size):])), "arm": arm, "seed": seed})
    return {
        "train_steps": float(step_count),
        "final_train_loss": float(np.mean(losses[-max(1, n // args.batch_size):])) if losses else 0.0,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "activation_norm_mean": float(np.mean(act_norms)) if act_norms else 0.0,
        "activation_norm_max": float(np.max(act_norms)) if act_norms else 0.0,
    }


def eval_cell(cell: BaseCell, data: dict[str, torch.Tensor], gate_repr: str, norm_policy: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with torch.no_grad():
        single_re, single_im = roll_cell(cell, data, 1, gate_repr, norm_policy, teacher_forced=True)
        true_re = data["target_re"][:, 0]
        true_im = data["target_im"][:, 0]
        metrics.update(vector_metrics(single_re, single_im, true_re, true_im, "single_step"))
        tf_accs = []
        fr_accs = []
        phase_drifts = []
        for steps in COMPOSITION_LENGTHS:
            tf_re, tf_im = roll_cell(cell, data, steps, gate_repr, norm_policy, teacher_forced=True)
            fr_re, fr_im = roll_cell(cell, data, steps, gate_repr, norm_policy, teacher_forced=False)
            true_re = data["target_re"][:, steps - 1]
            true_im = data["target_im"][:, steps - 1]
            tf = vector_metrics(tf_re, tf_im, true_re, true_im, f"teacher_forced_N{steps}")
            fr = vector_metrics(fr_re, fr_im, true_re, true_im, f"free_run_N{steps}")
            metrics.update(tf)
            metrics.update(fr)
            metrics[f"composition_accuracy_N{steps}"] = fr[f"free_run_N{steps}_phase_class_accuracy"]
            tf_accs.append(tf[f"teacher_forced_N{steps}_phase_class_accuracy"])
            fr_accs.append(fr[f"free_run_N{steps}_phase_class_accuracy"])
            phase_drifts.append(fr[f"free_run_N{steps}_phase_angle_mae"])
        metrics["teacher_forced_composition_accuracy"] = float(np.mean(tf_accs))
        metrics["free_run_composition_accuracy"] = float(np.mean(fr_accs))
        metrics["phase_drift_per_step"] = float(np.mean([phase_drifts[i] / COMPOSITION_LENGTHS[i] for i in range(len(COMPOSITION_LENGTHS))]))
        metrics["composition_accuracy_by_N"] = metrics["free_run_composition_accuracy"]
    return metrics


def vector_metrics(pred_re: torch.Tensor, pred_im: torch.Tensor, true_re: torch.Tensor, true_im: torch.Tensor, prefix: str) -> dict[str, float]:
    complex_mse = F.mse_loss(pred_re, true_re) + F.mse_loss(pred_im, true_im)
    pred_theta = torch.atan2(pred_im, pred_re)
    true_theta = torch.atan2(true_im, true_re)
    angle = wrapped_angle_error(pred_theta, true_theta)
    pred_mag = torch.sqrt(pred_re.square() + pred_im.square() + 1e-8)
    true_mag = torch.sqrt(true_re.square() + true_im.square() + 1e-8)
    pred_bucket = phase_bucket_from_vec(pred_re, pred_im)
    true_bucket = phase_bucket_from_vec(true_re, true_im)
    return {
        f"{prefix}_complex_mse": float(complex_mse.detach().cpu()),
        f"{prefix}_phase_angle_mae": float(angle.mean().detach().cpu()),
        f"{prefix}_phase_class_accuracy": float((pred_bucket == true_bucket).float().mean().detach().cpu()),
        f"{prefix}_magnitude_error": float((pred_mag - true_mag).abs().mean().detach().cpu()),
        f"{prefix}_magnitude_drift": float((pred_mag - pred_mag.mean()).abs().mean().detach().cpu()),
    }


def shuffle_controls(cell: BaseCell, data: dict[str, torch.Tensor], gate_repr: str, norm_policy: str) -> dict[str, float]:
    n = int(data["z_re"].shape[0])
    device = data["z_re"].device
    idx = torch.randperm(n, device=device)
    shuffled = {k: v.clone() for k, v in data.items()}
    shuffled["gate_theta"] = shuffled["gate_theta"][idx]
    shuffled["gate_re"] = shuffled["gate_re"][idx]
    shuffled["gate_im"] = shuffled["gate_im"][idx]
    shuffled["unit_re"] = shuffled["unit_re"][idx]
    shuffled["unit_im"] = shuffled["unit_im"][idx]
    pred_re, pred_im = roll_cell(cell, data, 1, gate_repr, norm_policy, teacher_forced=True)
    true_re = data["target_re"][:, 0]
    true_im = data["target_im"][:, 0]
    label_idx = torch.randperm(n, device=device)
    label_shuffle = float((phase_bucket_from_vec(pred_re, pred_im) == phase_bucket_from_vec(true_re[label_idx], true_im[label_idx])).float().mean().detach().cpu())
    gate_re, gate_im = roll_cell(cell, shuffled, 1, gate_repr, norm_policy, teacher_forced=True)
    gate_shuffle = float((phase_bucket_from_vec(gate_re, gate_im) == phase_bucket_from_vec(true_re, true_im)).float().mean().detach().cpu())
    mag = torch.sqrt(pred_re.square() + pred_im.square() + 1e-8)
    true_mag = torch.sqrt(true_re.square() + true_im.square() + 1e-8)
    magnitude_only = float((mag - true_mag).abs().mean().detach().cpu())
    angle_only = float(wrapped_angle_error(torch.atan2(pred_im, pred_re), torch.atan2(true_im, true_re)).mean().detach().cpu())
    return {
        "label_shuffle_control": label_shuffle,
        "gate_shuffle_control": gate_shuffle,
        "theta_shuffle_control": gate_shuffle,
        "target_shuffle_control": label_shuffle,
        "magnitude_only_control": magnitude_only,
        "angle_only_control": angle_only,
    }


def transfer_eval(cell: BaseCell, seed: int, args: argparse.Namespace, gate_repr: str, norm_policy: str, device: torch.device, finetune: bool, progress: Path) -> tuple[dict[str, float], dict[str, float]]:
    train = make_transfer_data(args.train_examples, args.transfer_grid, seed, device)
    eval_data = make_transfer_data(args.eval_examples, args.transfer_grid, seed + 1, device)
    audit: dict[str, float] = {"transfer_train_steps": 0.0}
    if finetune and any(p.requires_grad for p in cell.parameters()):
        opt = torch.optim.AdamW(cell.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
        grad_norms = []
        losses = []
        for epoch in range(max(1, args.epochs // 2)):
            order = torch.randperm(train["src_re"].shape[0], device=device)
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                opt.zero_grad(set_to_none=True)
                out_re, out_im = transfer_roll(cell, train, idx, gate_repr, norm_policy)
                loss = F.mse_loss(out_re, train["target_re"][idx]) + F.mse_loss(out_im, train["target_im"][idx])
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(cell.parameters(), 5.0)
                opt.step()
                grad_norms.append(float(grad.detach().cpu()))
                losses.append(float(loss.detach().cpu()))
        audit["transfer_train_steps"] = float(len(losses))
        audit["transfer_final_train_loss"] = float(np.mean(losses[-max(1, args.train_examples // args.batch_size):])) if losses else 0.0
        audit["transfer_grad_norm_mean"] = float(np.mean(grad_norms)) if grad_norms else 0.0
    with torch.no_grad():
        idx = torch.arange(eval_data["src_re"].shape[0], device=device)
        pred_re, pred_im = transfer_roll(cell, eval_data, idx, gate_repr, norm_policy)
        metrics = vector_metrics(pred_re, pred_im, eval_data["target_re"], eval_data["target_im"], "transfer")
        metrics["transfer_accuracy"] = metrics["transfer_phase_class_accuracy"]
    return metrics, audit


def make_transfer_data(n: int, grid: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 7100)
    path_len = min(12, max(4, grid - 3))
    src_phase = torch.randint(0, PHASE_CLASSES, (n,), generator=gen).float()
    src_theta = 2.0 * math.pi * src_phase / PHASE_CLASSES
    src_re = torch.cos(src_theta)
    src_im = torch.sin(src_theta)
    gates = torch.randint(0, PHASE_CLASSES, (n, path_len), generator=gen).float()
    theta = 2.0 * math.pi * gates / PHASE_CLASSES
    gate_re = torch.cos(theta)
    gate_im = torch.sin(theta)
    y_re = src_re.clone()
    y_im = src_im.clone()
    for step in range(path_len):
        y_re, y_im = complex_mul(y_re, y_im, gate_re[:, step], gate_im[:, step])
    return {
        "src_re": src_re.to(device),
        "src_im": src_im.to(device),
        "gate_re": gate_re.to(device),
        "gate_im": gate_im.to(device),
        "gate_theta": theta.to(device),
        "target_re": y_re.to(device),
        "target_im": y_im.to(device),
        "path_len": torch.full((n,), path_len, dtype=torch.long, device=device),
    }


def transfer_roll(cell: BaseCell, data: dict[str, torch.Tensor], idx: torch.Tensor, gate_repr: str, norm_policy: str) -> tuple[torch.Tensor, torch.Tensor]:
    z_re = data["src_re"][idx]
    z_im = data["src_im"][idx]
    init_re, init_im = z_re, z_im
    path_len = int(data["gate_re"].shape[1])
    for step in range(path_len):
        if gate_repr == "gate_as_angle_theta":
            gate = data["gate_theta"][idx, step].unsqueeze(1)
        elif gate_repr == "gate_as_cos_sin":
            gate = torch.stack([torch.cos(data["gate_theta"][idx, step]), torch.sin(data["gate_theta"][idx, step])], dim=1)
        else:
            gate = torch.stack([data["gate_re"][idx, step], data["gate_im"][idx, step]], dim=1)
        z_re, z_im = cell.step(z_re, z_im, gate)
        z_re, z_im = apply_norm_policy(z_re, z_im, init_re, init_im, norm_policy)
    return z_re, z_im


def audit_model(cell: BaseCell, train_audit: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    params = float(sum(p.numel() for p in cell.parameters()))
    trainable = float(sum(p.numel() for p in cell.parameters() if p.requires_grad))
    return {
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "optimizer": "AdamW" if trainable else "none",
        "lr": args.lr if trainable else 0.0,
        **train_audit,
    }


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path) -> JobResult:
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    progress = progress_root / f"{config.arm}_{config.gate_repr}_{config.gate_norm}_{config.norm_policy}_seed{seed}.jsonl"
    append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
    train = make_primitive_data(args.train_examples, max(COMPOSITION_LENGTHS), seed, config.gate_norm, device)
    eval_data = make_primitive_data(args.eval_examples, max(COMPOSITION_LENGTHS), seed + 997, config.gate_norm, device)
    cell = make_cell(config.arm, config.gate_repr).to(device)
    train_audit = train_cell(cell, config.arm, train, args, config.gate_repr, config.norm_policy, progress, seed)
    metrics = eval_cell(cell, eval_data, config.gate_repr, config.norm_policy)
    metrics.update(shuffle_controls(cell, eval_data, config.gate_repr, config.norm_policy))
    if config.arm == "RANDOM_INIT_PHASE_CELL_TRANSFER":
        random_cell = CurrentFactorCell(config.gate_repr).to(device)
        transfer_metrics, transfer_audit = transfer_eval(random_cell, seed, args, config.gate_repr, config.norm_policy, device, finetune=True, progress=progress)
        metrics.update({f"random_init_{k}": v for k, v in transfer_metrics.items()})
        metrics["random_init_transfer_accuracy"] = transfer_metrics["transfer_accuracy"]
        train_audit.update(transfer_audit)
    elif config.arm == "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER":
        transfer_metrics, _ = transfer_eval(cell, seed, args, config.gate_repr, config.norm_policy, device, finetune=False, progress=progress)
        metrics.update({f"pretrained_frozen_{k}": v for k, v in transfer_metrics.items()})
        metrics["pretrained_frozen_transfer_accuracy"] = transfer_metrics["transfer_accuracy"]
    elif config.arm == "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER":
        transfer_metrics, transfer_audit = transfer_eval(cell, seed, args, config.gate_repr, config.norm_policy, device, finetune=True, progress=progress)
        metrics.update({f"pretrained_finetuned_{k}": v for k, v in transfer_metrics.items()})
        metrics["pretrained_finetuned_transfer_accuracy"] = transfer_metrics["transfer_accuracy"]
        train_audit.update(transfer_audit)
    metrics["complex_mse"] = metrics["single_step_complex_mse"]
    metrics["phase_angle_mae"] = metrics["single_step_phase_angle_mae"]
    metrics["phase_class_accuracy"] = metrics["single_step_phase_class_accuracy"]
    metrics["magnitude_error"] = metrics["single_step_magnitude_error"]
    metrics["magnitude_drift"] = metrics["single_step_magnitude_drift"]
    audit = audit_model(cell, train_audit, args)
    for key, val in audit.items():
        if isinstance(val, (int, float)):
            metrics[key] = float(val)
    append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "metrics": metrics, "audit": audit})
    return JobResult(asdict(config), seed, metrics, audit)


def build_queue(args: argparse.Namespace) -> list[Config]:
    if args.arms:
        arms = parse_csv(args.arms)
        return [Config(arm, "gate_as_complex_pair", "normalized_gate", "no_renorm_between_steps") for arm in arms]
    queue = []
    primary = ("gate_as_complex_pair", "normalized_gate", "no_renorm_between_steps")
    for arm in ALL_ARMS:
        queue.append(Config(arm, *primary))
    for gate_repr in ANGLE_REPRS:
        for gate_norm in GATE_NORMS:
            for arm in ("CURRENT_FACTOR_CELL_SINGLE_STEP", "RICH_PHASE_CELL_SINGLE_STEP", "LOCAL_BILINEAR_SINGLE_STEP", "TINY_MLP_BASELINE"):
                cfg = Config(arm, gate_repr, gate_norm, "no_renorm_between_steps")
                if cfg not in queue:
                    queue.append(cfg)
    for norm_policy in NORM_POLICIES:
        for arm in ("CURRENT_FACTOR_CELL_MULTI_STEP", "RICH_PHASE_CELL_MULTI_STEP", "LOCAL_BILINEAR_MULTI_STEP"):
            cfg = Config(arm, "gate_as_complex_pair", "normalized_gate", norm_policy)
            if cfg not in queue:
                queue.append(cfg)
    return queue


def aggregate(results: list[JobResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        cfg = result.config
        key = f"{cfg['arm']}|{cfg['gate_repr']}|{cfg['gate_norm']}|{cfg['norm_policy']}"
        groups[key].append(result)
    agg: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        metric_keys = sorted({k for row in rows for k in row.metrics})
        means = {}
        stds = {}
        for metric in metric_keys:
            vals = [float(row.metrics[metric]) for row in rows if metric in row.metrics and not math.isnan(float(row.metrics[metric]))]
            means[metric] = float(np.mean(vals)) if vals else math.nan
            stds[metric] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg[key] = {"config": rows[0].config, "seeds": [r.seed for r in rows], "metric_mean": means, "metric_std": stds, "audit": rows[0].audit}
    return agg


def m(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    return float(row["metric_mean"].get(key, default)) if row else default


def find_primary(agg: dict[str, dict[str, Any]], arm: str) -> dict[str, Any] | None:
    for row in agg.values():
        cfg = row["config"]
        if cfg["arm"] == arm and cfg["gate_repr"] == "gate_as_complex_pair" and cfg["gate_norm"] == "normalized_gate" and cfg["norm_policy"] == "no_renorm_between_steps":
            return row
    return None


def paired(results: list[JobResult], left: str, right: str, key: str) -> dict[str, Any]:
    return paired_keys(results, left, key, right, key)


def paired_keys(results: list[JobResult], left: str, left_key: str, right: str, right_key: str) -> dict[str, Any]:
    def is_primary(r: JobResult, arm: str) -> bool:
        cfg = r.config
        return cfg["arm"] == arm and cfg["gate_repr"] == "gate_as_complex_pair" and cfg["gate_norm"] == "normalized_gate" and cfg["norm_policy"] == "no_renorm_between_steps"
    lmap = {r.seed: r.metrics[left_key] for r in results if is_primary(r, left) and left_key in r.metrics}
    rmap = {r.seed: r.metrics[right_key] for r in results if is_primary(r, right) and right_key in r.metrics}
    seeds = sorted(set(lmap) & set(rmap))
    nums = [float(lmap[seed] - rmap[seed]) for seed in seeds]
    mean = float(np.mean(nums)) if nums else math.nan
    std = float(np.std(nums)) if len(nums) > 1 else 0.0
    lower95 = mean - 1.96 * std / math.sqrt(len(nums)) if nums else math.nan
    return {
        "left": left,
        "right": right,
        "metric": f"{left_key} - {right_key}" if left_key != right_key else left_key,
        "deltas": [{"seed": seed, "delta": float(lmap[seed] - rmap[seed])} for seed in seeds],
        "mean_delta": mean,
        "lower95_delta": lower95,
        "std_delta": std,
        "min_delta": float(np.min(nums)) if nums else math.nan,
        "max_delta": float(np.max(nums)) if nums else math.nan,
        "positive_seed_count": int(sum(v > 0 for v in nums)),
        "paired_seed_count": len(nums),
    }


def verdicts(agg: dict[str, dict[str, Any]], results: list[JobResult]) -> list[str]:
    labels: list[str] = []
    teacher = find_primary(agg, "FIXED_COMPLEX_MULTIPLY_TEACHER")
    current_single = find_primary(agg, "CURRENT_FACTOR_CELL_SINGLE_STEP")
    current_multi = find_primary(agg, "CURRENT_FACTOR_CELL_MULTI_STEP")
    rich_single = find_primary(agg, "RICH_PHASE_CELL_SINGLE_STEP")
    rich_multi = find_primary(agg, "RICH_PHASE_CELL_MULTI_STEP")
    bilinear = find_primary(agg, "LOCAL_BILINEAR_SINGLE_STEP")
    mlp = find_primary(agg, "TINY_MLP_BASELINE")
    random_transfer = find_primary(agg, "RANDOM_INIT_PHASE_CELL_TRANSFER")
    frozen = find_primary(agg, "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER")
    finetuned = find_primary(agg, "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER")
    if teacher and m(teacher, "phase_class_accuracy", 0.0) >= 0.99 and m(teacher, "label_shuffle_control", 1.0) < 0.40:
        labels.append("TASK_VALID")
    learnable = any(row and m(row, "phase_class_accuracy", 0.0) >= 0.90 for row in [current_single, rich_single, bilinear, mlp])
    if learnable:
        labels.append("PRIMITIVE_LEARNABLE")
    if current_single and bilinear and mlp and m(current_single, "phase_class_accuracy", 0.0) < 0.70 and max(m(bilinear, "phase_class_accuracy", 0.0), m(mlp, "phase_class_accuracy", 0.0)) >= 0.90:
        labels.append("PRIMITIVE_NOT_LEARNABLE_AS_IMPLEMENTED")
    for single, multi in [(current_single, current_multi), (rich_single, rich_multi)]:
        if single and multi and m(single, "phase_class_accuracy", 0.0) >= 0.90 and m(multi, "free_run_composition_accuracy", 0.0) < m(multi, "teacher_forced_composition_accuracy", 0.0) - 0.20:
            labels.append("COMPOSITION_STABILITY_FAILURE")
    if frozen and random_transfer and m(frozen, "pretrained_frozen_transfer_accuracy", 0.0) > m(random_transfer, "random_init_transfer_accuracy", 0.0) + 0.10:
        labels.append("PRETRAINING_RESCUES_TRANSFER_STRONGLY")
    if finetuned and random_transfer and m(finetuned, "pretrained_finetuned_transfer_accuracy", 0.0) > m(random_transfer, "random_init_transfer_accuracy", 0.0) + 0.10:
        labels.append("PRETRAINING_RESCUES_INITIALIZATION")
    if teacher and frozen and finetuned and max(m(frozen, "pretrained_frozen_transfer_accuracy", 0.0), m(finetuned, "pretrained_finetuned_transfer_accuracy", 0.0)) < m(teacher, "phase_class_accuracy", 0.0) - 0.10:
        labels.append("FIXED_COMPLEX_OPERATOR_STILL_REQUIRED")
    if teacher and m(teacher, "label_shuffle_control", 0.0) > 0.50:
        labels.append("TASK_OR_CONTROL_INVALID")
    return sorted(set(labels or ["PRIMITIVE_LEARNABILITY_PARTIAL"]))


def write_outputs(out_dir: Path, args: argparse.Namespace, results: list[JobResult], status: str, jobs: int, sample_rows: list[dict[str, float]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    agg = aggregate(results)
    labels = verdicts(agg, results) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    deltas = [
        paired(results, "CURRENT_FACTOR_CELL_SINGLE_STEP", "LOCAL_BILINEAR_SINGLE_STEP", "phase_class_accuracy"),
        paired(results, "RICH_PHASE_CELL_SINGLE_STEP", "LOCAL_BILINEAR_SINGLE_STEP", "phase_class_accuracy"),
        paired(results, "RICH_PHASE_CELL_SINGLE_STEP", "TINY_MLP_BASELINE", "phase_class_accuracy"),
        paired_keys(results, "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER", "pretrained_frozen_transfer_accuracy", "RANDOM_INIT_PHASE_CELL_TRANSFER", "random_init_transfer_accuracy"),
        paired_keys(results, "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER", "pretrained_finetuned_transfer_accuracy", "RANDOM_INIT_PHASE_CELL_TRANSFER", "random_init_transfer_accuracy"),
    ]
    write_jsonl(out_dir / "paired_seed_deltas.jsonl", deltas)
    write_jsonl(out_dir / "primitive_cases.jsonl", sample_rows[:128])
    write_jsonl(out_dir / "composition_metrics.jsonl", [{"key": key, "config": row["config"], "teacher_forced": m(row, "teacher_forced_composition_accuracy"), "free_run": m(row, "free_run_composition_accuracy")} for key, row in agg.items()])
    write_jsonl(out_dir / "gate_encoding_metrics.jsonl", [{"key": key, "config": row["config"], "phase_acc": m(row, "phase_class_accuracy"), "angle_mae": m(row, "phase_angle_mae")} for key, row in agg.items()])
    write_jsonl(out_dir / "transfer_metrics.jsonl", [{"key": key, "config": row["config"], "random": m(row, "random_init_transfer_accuracy"), "frozen": m(row, "pretrained_frozen_transfer_accuracy"), "finetuned": m(row, "pretrained_finetuned_transfer_accuracy")} for key, row in agg.items()])
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {"status": status, "verdict": labels, "completed_jobs": len(results), "config": config, "aggregate": agg, "paired_deltas": deltas}
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        "",
        "| Arm | Gate | Norm | Policy | Single Acc | TF Comp | Free Comp | Angle MAE | MSE | Random Transfer | Frozen Transfer | Finetuned Transfer | Params |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (r["config"]["arm"], r["config"]["gate_repr"], r["config"]["gate_norm"], r["config"]["norm_policy"])):
        cfg = row["config"]
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.5f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(
                cfg["arm"],
                cfg["gate_repr"],
                cfg["gate_norm"],
                cfg["norm_policy"],
                m(row, "phase_class_accuracy"),
                m(row, "teacher_forced_composition_accuracy"),
                m(row, "free_run_composition_accuracy"),
                m(row, "phase_angle_mae"),
                m(row, "complex_mse"),
                m(row, "random_init_transfer_accuracy"),
                m(row, "pretrained_frozen_transfer_accuracy"),
                m(row, "pretrained_finetuned_transfer_accuracy"),
                m(row, "parameter_count"),
            )
        )
    lines.append("\n## Paired Deltas\n")
    for delta in deltas:
        lines.append(f"- `{delta['left']}` minus `{delta['right']}` on `{delta['metric']}`: mean `{delta['mean_delta']:.4f}`, lower95 `{delta['lower95_delta']:.4f}`, positive `{delta['positive_seed_count']}/{delta['paired_seed_count']}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc(agg: dict[str, dict[str, Any]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int, results: list[JobResult]) -> None:
    def row(arm: str) -> dict[str, Any] | None:
        return find_primary(agg, arm)

    lines = [
        "# STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY Result",
        "",
        "## Latest Run",
        "",
        "```text",
        f"out={args.out}",
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
        "## Primary Results",
        "",
        "| Arm | Single Acc | Teacher-Forced | Free-Run | Angle MAE | MSE | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "FIXED_COMPLEX_MULTIPLY_TEACHER",
        "CURRENT_FACTOR_CELL_SINGLE_STEP",
        "CURRENT_FACTOR_CELL_MULTI_STEP",
        "RICH_PHASE_CELL_SINGLE_STEP",
        "RICH_PHASE_CELL_MULTI_STEP",
        "LOCAL_BILINEAR_SINGLE_STEP",
        "LOCAL_BILINEAR_MULTI_STEP",
        "TINY_MLP_BASELINE",
        "COMPLEX_MULTIPLY_GNN",
        "RANDOM_INIT_PHASE_CELL_TRANSFER",
        "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER",
        "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER",
    ]:
        r = row(arm)
        if r:
            lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.4f}` | `{:.5f}` | `{:.0f}` |".format(arm, m(r, "phase_class_accuracy"), m(r, "teacher_forced_composition_accuracy"), m(r, "free_run_composition_accuracy"), m(r, "phase_angle_mae"), m(r, "complex_mse"), m(r, "parameter_count")))
    lines.extend(["", "## Transfer", "", "| Arm | Random | Frozen | Finetuned |", "|---|---:|---:|---:|"])
    for arm in ["RANDOM_INIT_PHASE_CELL_TRANSFER", "PRETRAINED_FROZEN_PHASE_CELL_TRANSFER", "PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER"]:
        r = row(arm)
        if r:
            lines.append("| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(arm, m(r, "random_init_transfer_accuracy"), m(r, "pretrained_frozen_transfer_accuracy"), m(r, "pretrained_finetuned_transfer_accuracy")))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe tests primitive learnability only. It should not be read as a full VRAXION, consciousness, language, or general reasoning claim.",
            "",
            "## Claim Boundary",
            "",
            "The result answers whether a learned cell can acquire and transfer the local complex phase-transport primitive under the tested gate encodings and normalization policies.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sample_rows(seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(128):
        theta_z = rng.uniform(-math.pi, math.pi)
        theta_g = rng.uniform(-math.pi, math.pi)
        rows.append({"theta_z": theta_z, "theta_gate": theta_g, "theta_target": math.atan2(math.sin(theta_z + theta_g), math.cos(theta_z + theta_g))})
    return rows


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, Any]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(config, seed) for config in configs for seed in seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "queue.json", [{**asdict(config), "seed": seed} for config, seed in queue])
    sample_rows = build_sample_rows(2026)
    write_jsonl(args.out / "examples_sample.jsonl", sample_rows)
    progress = args.out / "progress.jsonl"
    metrics = args.out / "metrics.jsonl"
    job_progress = args.out / "job_progress"
    append_jsonl(progress, {"time": now_iso(), "event": "run_start", "total_jobs": len(queue), "jobs": jobs})
    results: list[JobResult] = []
    write_outputs(args.out, args, results, "partial", jobs, sample_rows)
    if jobs <= 1:
        for config, seed in queue:
            append_jsonl(progress, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
            result = run_job(config, seed, args, job_progress)
            results.append(result)
            append_jsonl(metrics, asdict(result))
            append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results)})
            write_outputs(args.out, args, results, "partial", jobs, sample_rows)
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
                    write_outputs(args.out, args, results, "partial", jobs, sample_rows)
                    continue
                for fut in done:
                    config, seed = meta[fut]
                    result = fut.result()
                    results.append(result)
                    append_jsonl(metrics, asdict(result))
                    append_jsonl(progress, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out, args, results, "partial", jobs, sample_rows)
    agg, labels = write_outputs(args.out, args, results, "complete", jobs, sample_rows)
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
