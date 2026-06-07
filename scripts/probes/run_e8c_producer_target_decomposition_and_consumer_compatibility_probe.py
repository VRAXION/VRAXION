#!/usr/bin/env python3
"""E8C producer target decomposition and consumer compatibility probe.

E8B showed smooth producer-code learning followed by plateau, plus smaller-batch
gradient direction conflict. E8C keeps the same numeric pocket/RAM proxy and
tests whether decomposing the mechanical RAM-write target lowers that conflict
while preserving downstream consumer compatibility.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E8A_PATH = Path(__file__).with_name("run_e8a_canonical_ram_code_learning_and_smoothness_probe.py")
MILESTONE = "E8C_PRODUCER_TARGET_DECOMPOSITION_AND_CONSUMER_COMPATIBILITY_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e8c_producer_target_decomposition_and_consumer_compatibility_probe")
DEFAULT_SEEDS = (103001, 103002, 103003, 103004, 103005, 103006, 103007, 103008)
OUTPUT_WIDTH = 12

SYSTEMS = (
    "current_full_code_teacher_baseline",
    "local_smooth_full_code_teacher",
    "per_skill_decomposed_heads",
    "primary_then_support_staged_teacher",
    "support_cells_only_after_primary_plateau",
    "consumer_sensitivity_weighted_targets",
    "route_step_local_teacher_targets",
    "codebook_decomposed_targets",
    "low_conflict_batch_curriculum",
    "consumer_compatibility_weighted_loss",
    "mutation_repair_after_consumer_compatible_plateau",
    "mutation_only_decomposed_lowbit",
    "dense_graph_danger_control",
    "consumer_distill_reference",
    "oracle_low_bit_reference",
)
ORACLE_SYSTEMS = {"oracle_low_bit_reference", "consumer_distill_reference"}
MUTATION_SYSTEMS = {"mutation_repair_after_consumer_compatible_plateau", "mutation_only_decomposed_lowbit"}
TRAINED_SYSTEMS = tuple(system for system in SYSTEMS if system not in ORACLE_SYSTEMS | {"mutation_only_decomposed_lowbit", "dense_graph_danger_control"})
TEACHER_STYLES = (
    "current_full_code_teacher",
    "local_smooth_full_code_teacher",
    "decomposed_primary_support_teacher",
    "consumer_sensitivity_teacher",
    "codebook_decomposed_teacher",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "target_decomposition_report.json",
    "consumer_sensitivity_report.json",
    "producer_dynamics_report.json",
    "gradient_diagnostics_report.json",
    "compatibility_report.json",
    "mutation_repair_report.json",
    "system_results.json",
    "row_level_samples.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e8c_target_decomposition_positive",
    "e8c_consumer_sensitivity_weighting_positive",
    "e8c_route_step_local_targets_positive",
    "e8c_gradient_conflict_reduced_but_usefulness_low",
    "e8c_producer_architecture_bottleneck",
    "e8c_consumer_interface_bottleneck",
    "e8c_mutation_repair_after_compatibility_plateau_positive",
    "e8c_mutation_only_decomposed_learning_viable",
    "e8c_current_code_interface_still_wrong",
    "e8c_graph_soup_regression_detected",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e8a = load_module(E8A_PATH, "e8a_canonical_ram_code_learning_probe")
e7z = e8a.e7z
e7y = e8a.e7y
e7r = e8a.e7r
e7p = e8a.e7p
e7o = e8a.e7o

FLOW_DIM = int(e8a.FLOW_DIM)
SKILLS = tuple(e8a.SKILLS)
SPLITS = tuple(e8a.SPLITS)
EVAL_SPLITS = tuple(e8a.EVAL_SPLITS)
RESULT_POS = dict(e8a.RESULT_POS)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    pocket_pretrain_rows_per_seed: int
    pocket_validation_rows_per_seed: int
    pocket_dim: int
    pocket_core_steps: int
    pocket_epochs: int
    local_epochs: int
    full_epochs: int
    batch_size: int
    gradient_diagnostic_batch_size: int
    learning_rate: float
    local_learning_rate: float
    weight_decay: float
    pruned_read_count: int
    repair_generations: int
    repair_population: int
    similarity_threshold: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    replay: bool
    execution_mode: str


def round_float(value: float) -> float:
    return e8a.round_float(value)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    e8a.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    e8a.write_text(path, text)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e8a.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e8a.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e8a.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e8a.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["replay"] = False
    return payload


def to_e8a_settings(settings: Settings) -> Any:
    return e8a.Settings(
        seeds=settings.seeds,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        pocket_pretrain_rows_per_seed=settings.pocket_pretrain_rows_per_seed,
        pocket_validation_rows_per_seed=settings.pocket_validation_rows_per_seed,
        pocket_dim=settings.pocket_dim,
        pocket_core_steps=settings.pocket_core_steps,
        pocket_epochs=settings.pocket_epochs,
        local_epochs=settings.local_epochs,
        full_epochs=settings.full_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        local_learning_rate=settings.local_learning_rate,
        weight_decay=settings.weight_decay,
        pruned_read_count=settings.pruned_read_count,
        repair_generations=settings.repair_generations,
        repair_population=settings.repair_population,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        replay=settings.replay,
        execution_mode=settings.execution_mode,
    )


def to_e7o_settings(settings: Settings) -> Any:
    return e8a.to_e7o_settings(to_e8a_settings(settings))


def to_e7r_settings(settings: Settings) -> Any:
    return e8a.to_e7r_settings(to_e8a_settings(settings))


def to_e7z_settings(settings: Settings) -> Any:
    return e8a.to_e7z_settings(to_e8a_settings(settings))


def bundle_entropy(values: np.ndarray) -> float:
    rounded = [round(float(v), 4) for v in np.asarray(values, dtype=np.float32).reshape(-1)]
    if not rounded:
        return 0.0
    counts: dict[float, int] = {}
    for value in rounded:
        counts[value] = counts.get(value, 0) + 1
    probs = np.asarray([count / len(rounded) for count in counts.values()], dtype=np.float64)
    return round_float(float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12)))))


def teacher_values(row: dict[str, Any], canonical_after: np.ndarray, flow_before: np.ndarray, skill: str, code: str, teacher_style: str) -> np.ndarray:
    if teacher_style in {"current_full_code_teacher", "decomposed_primary_support_teacher", "consumer_sensitivity_teacher"}:
        out = e8a.apply_teacher_code(row, flow_before, skill, code, "current_oracle_projection_code", e7z.default_boundary_params(code))
        return out[e7y.bundle_cells(skill, OUTPUT_WIDTH)].astype(np.float32)
    raw = e7y.bundle_values(row, canonical_after, skill, OUTPUT_WIDTH).astype(np.float32)
    result = float(canonical_after[RESULT_POS[skill]])
    if teacher_style == "codebook_decomposed_teacher":
        vals = raw.copy()
        levels = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        for idx in range(1, OUTPUT_WIDTH):
            vals[idx] = levels[int(np.argmin(np.abs(levels - vals[idx])))]
        vals[0] = result
    elif teacher_style == "local_smooth_full_code_teacher":
        cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
        before = np.asarray(flow_before[cells], dtype=np.float32)
        vals = 0.65 * before + 0.35 * raw
        vals[0] = result
    elif teacher_style == "factorized_teacher_code":
        vals = np.zeros(OUTPUT_WIDTH, dtype=np.float32)
        vals[0] = result
        result_vec = np.asarray([canonical_after[idx] for idx in e8a.RESULT_INDICES], dtype=np.float32)
        bank = e7y.rotated_bank(skill)
        for idx in range(1, OUTPUT_WIDTH):
            a = float(result_vec[idx % len(result_vec)])
            b = float(canonical_after[bank[(idx * 5) % len(bank)]])
            vals[idx] = np.float32(np.tanh(0.85 * a + 0.15 * b))
    else:
        raise ValueError(teacher_style)
    return e7z.quantize_bundle_values(vals, skill, code, e7z.default_boundary_params(code), primary_is_logit=False)


def apply_e8c_teacher_code(
    row: dict[str, Any],
    flow: np.ndarray,
    skill: str,
    code: str,
    teacher_style: str,
) -> np.ndarray:
    canonical_after = e7y.canonical_step(row, flow, skill)
    cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
    values = teacher_values(row, canonical_after, flow, skill, code, teacher_style)
    out = canonical_after.astype(np.float32).copy()
    for cell, value in zip(cells, values):
        out[cell] = value
    return out.astype(np.float32)


def generate_context_tasks(
    composition_task: dict[str, list[dict[str, Any]]],
    code: str,
    teacher_style: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    for split in SPLITS:
        for row in composition_task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            for step_idx, skill in enumerate(tuple(row["expected_route"])):
                target = apply_e8c_teacher_code(row, flow, skill, code, teacher_style)
                route = tuple(row["expected_route"])
                next_skill = route[step_idx + 1] if step_idx + 1 < len(route) else None
                tasks[skill][split].append(
                    {
                        "row_id": f"{row['row_id']}:{step_idx}:{skill}:{code}:{teacher_style}",
                        "split": split,
                        "skill": skill,
                        "family": row["family"],
                        "route_step": step_idx,
                        "next_skill": next_skill,
                        "flow": flow.tolist(),
                        "target_flow": target.tolist(),
                        "target_value": int(target[RESULT_POS[skill]] >= 0.5),
                    }
                )
                flow = target
    return tasks


def context_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return e7p.split_arrays(rows)


def code_metrics_np(pred: np.ndarray, target: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    written = pred[:, cells].astype(np.float32)
    wanted = target[:, cells].astype(np.float32)
    mae_by_cell = np.mean(np.abs(written - wanted), axis=0) if len(written) else np.zeros(len(cells), dtype=np.float32)
    sign_acc_by_cell = np.mean((written >= 0.0) == (wanted >= 0.0), axis=0) if len(written) else np.zeros(len(cells), dtype=np.float32)
    support = written[:, 1:] if written.shape[1] > 1 else written[:, :0]
    return {
        "code_similarity": round_float(max(0.0, 1.0 - float(np.mean(mae_by_cell)))),
        "bundle_mae": round_float(float(np.mean(mae_by_cell))),
        "cellwise_correlation": e7z.safe_corr(written.reshape(-1).tolist(), wanted.reshape(-1).tolist()),
        "cosine_similarity": e7z.cosine_similarity(written.reshape(-1).tolist(), wanted.reshape(-1).tolist()),
        "support_sign_mismatch": round_float(float(np.mean((written[:, 1:] >= 0.0) != (wanted[:, 1:] >= 0.0))) if written.shape[1] > 1 else 0.0),
        "support_silence_rate": round_float(float(np.mean(np.abs(support) < 1e-6)) if support.size else 0.0),
        "write_entropy": bundle_entropy(written),
        "per_cell_mae": [round_float(float(v)) for v in mae_by_cell.tolist()],
        "per_cell_sign_accuracy": [round_float(float(v)) for v in sign_acc_by_cell.tolist()],
    }


def evaluate_model_on_context(
    model: torch.nn.Module,
    context_task: dict[str, list[dict[str, Any]]],
    skill: str,
    contract: dict[str, Any],
    settings: Settings,
    split: str,
    limit: int,
) -> dict[str, Any]:
    rows = context_task[split][: max(1, min(limit, len(context_task[split])))]
    if not rows:
        return {"split": split, "code_similarity": 0.0, "bundle_mae": 1.0, "loss": 0.0}
    device = select_device(settings.device)
    x, y, _ = context_arrays(rows)
    read_mask = torch.as_tensor(contract["read"].astype(np.float32), dtype=torch.float32, device=device)
    write_mask = torch.as_tensor((contract["write"] | contract["scratch"]).astype(np.float32), dtype=torch.float32, device=device)
    cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32, device=device)
        pred = model(xb * read_mask)
        masked = xb + (pred - xb) * write_mask if contract["enforce"] else pred
        target = torch.as_tensor(y, dtype=torch.float32, device=device)
        loss = F.mse_loss(masked[:, cells], target[:, cells]).detach().cpu().item()
        pred_np = masked.detach().cpu().numpy()
    metrics = code_metrics_np(pred_np, y, cells)
    metrics["split"] = split
    metrics["loss"] = round_float(float(loss))
    return metrics


def gradient_stats(batch_grads: list[torch.Tensor]) -> dict[str, float]:
    if not batch_grads:
        return {"gradient_norm": 0.0, "gradient_variance": 0.0, "gradient_cosine": 0.0, "gradient_cosine_negative_rate": 0.0}
    norms = [float(torch.linalg.vector_norm(g).detach().cpu()) for g in batch_grads]
    cosines: list[float] = []
    for left, right in zip(batch_grads, batch_grads[1:]):
        denom = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
        cosines.append(float(torch.dot(left, right) / denom) if denom > 1e-12 else 0.0)
    return {
        "gradient_norm": round_float(math.fsum(norms) / len(norms)),
        "gradient_variance": round_float(float(np.var(norms))),
        "gradient_cosine": round_float(math.fsum(cosines) / len(cosines) if cosines else 0.0),
        "gradient_cosine_negative_rate": round_float(float(np.mean([c < 0.0 for c in cosines])) if cosines else 0.0),
    }


def train_producer_diagnostic(
    seed: int,
    skill: str,
    system: str,
    code: str,
    teacher_style: str,
    loss_mode: str,
    base_state: dict[str, Any],
    context_task: dict[str, list[dict[str, Any]]],
    settings: Settings,
    out: Path | None,
) -> dict[str, Any]:
    device = select_device(settings.device)
    e7p.set_determinism(e7z.stable_seed(f"E8C-producer:{seed}:{skill}:{system}:{code}:{teacher_style}:{loss_mode}"), device)
    model = e7p.state_to_model(base_state, to_e7r_settings(settings), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.local_learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(e7z.stable_seed(f"E8C-batch:{seed}:{skill}:{system}"))
    x_train, y_train, t_train = context_arrays(context_task["train"])
    contract = e7z.build_lowbit_contract(skill, system, read_count=OUTPUT_WIDTH)
    cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
    read_mask = torch.as_tensor(contract["read"].astype(np.float32), dtype=torch.float32, device=device)
    write_mask = torch.as_tensor((contract["write"] | contract["scratch"]).astype(np.float32), dtype=torch.float32, device=device)
    preserve_idx = torch.as_tensor(np.flatnonzero(contract["preserve"]).astype(np.int64), dtype=torch.long, device=device)
    cell_idx = torch.as_tensor(cells, dtype=torch.long, device=device)
    primary_idx = int(cells[0])
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history: list[dict[str, Any]] = []
    cell_weights = torch.ones(len(cells), dtype=torch.float32, device=device)
    cell_weights[0] = 2.5
    for idx in range(1, len(cells)):
        cell_weights[idx] = 1.0 + 0.05 * idx
    delta_cells = np.abs(y_train[:, cells] - x_train[:, cells]).astype(np.float32) if len(x_train) else np.zeros((0, len(cells)), dtype=np.float32)
    next_masks_np = np.zeros((len(x_train), FLOW_DIM), dtype=np.float32)
    route_cell_weights_np = np.ones((len(x_train), len(cells)), dtype=np.float32)
    for row_idx, row in enumerate(context_task["train"]):
        next_skill = row.get("next_skill")
        if next_skill:
            next_contract = e7z.build_lowbit_contract(str(next_skill), "e8c_next_read_reference", read_count=OUTPUT_WIDTH)
            next_read = next_contract["read"].astype(np.float32)
            next_masks_np[row_idx] = next_read
            route_cell_weights_np[row_idx] += 2.0 * next_read[cells]
    sensitivity_np = np.mean(delta_cells * route_cell_weights_np, axis=0) if len(delta_cells) else np.ones(len(cells), dtype=np.float32)
    sensitivity_np = sensitivity_np / max(float(np.mean(sensitivity_np)), 1.0e-6)
    sensitivity_np[0] = max(float(sensitivity_np[0]), 2.0)
    sensitivity_weights = torch.as_tensor(sensitivity_np.astype(np.float32), dtype=torch.float32, device=device)
    difficulty_np = np.mean(delta_cells * sensitivity_np.reshape(1, -1), axis=1) if len(delta_cells) else np.zeros(len(x_train), dtype=np.float32)

    def loss_for_indices(idx: np.ndarray, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
        tb = torch.as_tensor(t_train[idx], dtype=torch.float32, device=device)
        route_weights = torch.as_tensor(route_cell_weights_np[idx], dtype=torch.float32, device=device)
        next_masks = torch.as_tensor(next_masks_np[idx], dtype=torch.float32, device=device)
        pred = model(xb * read_mask)
        masked = xb + (pred - xb) * write_mask if contract["enforce"] else pred
        out_cells = masked.index_select(1, cell_idx)
        target_cells = yb.index_select(1, cell_idx)
        result_loss = F.binary_cross_entropy_with_logits(masked[:, primary_idx], tb)
        primary_loss = F.mse_loss(out_cells[:, :1], target_cells[:, :1])
        support_loss = F.mse_loss(out_cells[:, 1:], target_cells[:, 1:]) if out_cells.shape[1] > 1 else torch.zeros((), dtype=torch.float32, device=device)
        per_cell_loss = torch.mean((out_cells - target_cells) ** 2)
        weighted_loss = torch.mean(((out_cells - target_cells) ** 2) * cell_weights)
        sensitivity_loss = torch.mean(((out_cells - target_cells) ** 2) * sensitivity_weights)
        route_local_loss = torch.mean(((out_cells - target_cells) ** 2) * route_weights)
        next_loss = torch.mean(((masked - yb) ** 2) * next_masks)
        preserve_loss = F.mse_loss(masked.index_select(1, preserve_idx), xb.index_select(1, preserve_idx))
        cosine = 1.0 - F.cosine_similarity(out_cells, target_cells, dim=1).mean()
        support_factor = 0.15 if epoch < max(1, settings.local_epochs // 2) else 1.0
        if loss_mode == "baseline":
            loss = result_loss + 0.45 * per_cell_loss + 0.45 * preserve_loss
        elif loss_mode == "decomposed":
            loss = 0.35 * result_loss + 1.30 * primary_loss + 0.80 * support_loss + 0.25 * preserve_loss
        elif loss_mode == "staged":
            loss = 0.35 * result_loss + 1.35 * primary_loss + support_factor * 0.95 * support_loss + 0.25 * preserve_loss
        elif loss_mode == "support_plateau":
            loss = 0.30 * result_loss + 1.10 * primary_loss + support_factor * 1.20 * support_loss + 0.22 * preserve_loss
        elif loss_mode == "consumer_sensitivity":
            loss = 0.25 * result_loss + 1.35 * sensitivity_loss + 0.35 * next_loss + 0.20 * preserve_loss
        elif loss_mode == "route_step_local":
            loss = 0.25 * result_loss + 1.25 * route_local_loss + 0.55 * next_loss + 0.20 * preserve_loss
        elif loss_mode == "codebook_decomposed":
            loss = 0.30 * result_loss + 1.10 * primary_loss + 0.75 * support_loss + 0.30 * cosine + 0.20 * preserve_loss
        elif loss_mode == "low_conflict":
            loss = 0.25 * result_loss + 1.10 * route_local_loss + 0.70 * sensitivity_loss + 0.15 * preserve_loss
        elif loss_mode == "consumer_compatibility":
            loss = 0.35 * result_loss + 0.85 * per_cell_loss + 0.90 * next_loss + 0.15 * preserve_loss
        else:
            raise ValueError(loss_mode)
        return loss, masked, target_cells

    for epoch in range(settings.local_epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        if loss_mode == "low_conflict" and epoch < max(1, settings.local_epochs // 2):
            indices = indices[np.argsort(difficulty_np[indices])]
        batch_losses: list[float] = []
        batch_grads: list[torch.Tensor] = []
        model.train()
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_for_indices(idx, epoch)
            loss.backward()
            flat_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    flat_grads.append(param.grad.detach().flatten().cpu())
            if flat_grads:
                batch_grads.append(torch.cat(flat_grads))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        diag_grads: list[torch.Tensor] = []
        diag_batch = max(2, min(settings.gradient_diagnostic_batch_size, len(indices)))
        diag_indices = indices[: max(diag_batch, min(len(indices), diag_batch * 6))]
        model.train()
        for start in range(0, len(diag_indices), diag_batch):
            idx = diag_indices[start : start + diag_batch]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_for_indices(idx, epoch)
            loss.backward()
            flat_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    flat_grads.append(param.grad.detach().flatten().cpu())
            if flat_grads:
                diag_grads.append(torch.cat(flat_grads))
        optimizer.zero_grad(set_to_none=True)
        train_metrics = evaluate_model_on_context(model, context_task, skill, contract, settings, "train", limit=192)
        val_metrics = evaluate_model_on_context(model, context_task, skill, contract, settings, "validation", limit=192)
        ood_metrics = evaluate_model_on_context(model, context_task, skill, contract, settings, "ood", limit=192)
        grad = gradient_stats(diag_grads or batch_grads)
        score = float(val_metrics["code_similarity"]) - 0.05 * float(val_metrics["support_sign_mismatch"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "seed": seed,
            "system": system,
            "skill": skill,
            "code": code,
            "teacher_style": teacher_style,
            "loss_mode": loss_mode,
            "epoch": epoch,
            "train_loss": round_float(math.fsum(batch_losses) / max(1, len(batch_losses))),
            "validation_loss": val_metrics["loss"],
            "train_code_similarity": train_metrics["code_similarity"],
            "validation_code_similarity": val_metrics["code_similarity"],
            "ood_code_similarity": ood_metrics["code_similarity"],
            "train_bundle_mae": train_metrics["bundle_mae"],
            "validation_bundle_mae": val_metrics["bundle_mae"],
            "ood_bundle_mae": ood_metrics["bundle_mae"],
            "validation_support_sign_mismatch": val_metrics["support_sign_mismatch"],
            "validation_support_silence_rate": val_metrics["support_silence_rate"],
            "validation_write_entropy": val_metrics["write_entropy"],
            "gradient_norm": grad["gradient_norm"],
            "gradient_variance": grad["gradient_variance"],
            "gradient_cosine": grad["gradient_cosine"],
            "gradient_cosine_negative_rate": grad["gradient_cosine_negative_rate"],
            "per_cell_mae": val_metrics["per_cell_mae"],
            "per_cell_sign_accuracy": val_metrics["per_cell_sign_accuracy"],
        }
        history.append(row)
        if out:
            append_progress(out, "e8c_producer_epoch", **{k: v for k, v in row.items() if k not in {"per_cell_mae", "per_cell_sign_accuracy"}})
    if best_state is not None:
        model.load_state_dict(best_state)
    state = e7p.model_to_state(model, to_e7r_settings(settings), [MILESTONE, system, skill])
    summary = summarize_history(history)
    sensitivity_order = list(np.argsort(-sensitivity_np).astype(int))
    sensitivity_report = {
        "bundle_cells": [int(cell) for cell in cells.tolist()],
        "primary_cell": int(cells[0]),
        "support_cell_count": int(max(0, len(cells) - 1)),
        "sensitivity_weights": [round_float(float(v)) for v in sensitivity_np.tolist()],
        "high_impact_bundle_offsets": [int(idx) for idx in sensitivity_order[: min(4, len(sensitivity_order))]],
        "mechanical_groups": ["primary_cell", "support_cells", "consumer_sensitivity_cells", "route_step_local_cells"],
    }
    return {"state": state, "contract": contract, "contract_json": e7r.contract_to_json(contract), "history": history, "summary": summary, "sensitivity": sensitivity_report}


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    first = history[0]
    last = history[-1]
    best = max(history, key=lambda row: row["validation_code_similarity"])
    tail = history[max(0, len(history) - max(2, len(history) // 4)) :]
    tail_values = [float(row["validation_code_similarity"]) for row in tail]
    return {
        "epoch_count": len(history),
        "loss_start": first["train_loss"],
        "loss_end": last["train_loss"],
        "loss_drop": round_float(float(first["train_loss"]) - float(last["train_loss"])),
        "validation_code_similarity_start": first["validation_code_similarity"],
        "validation_code_similarity_end": last["validation_code_similarity"],
        "validation_code_similarity_best": best["validation_code_similarity"],
        "best_epoch": best["epoch"],
        "final_gap_to_best": round_float(float(best["validation_code_similarity"]) - float(last["validation_code_similarity"])),
        "tail_gain": round_float(tail_values[-1] - tail_values[0] if len(tail_values) > 1 else 0.0),
        "tail_range": round_float(max(tail_values) - min(tail_values) if tail_values else 0.0),
        "gradient_norm_mean": round_float(math.fsum(float(row["gradient_norm"]) for row in history) / len(history)),
        "gradient_variance_mean": round_float(math.fsum(float(row["gradient_variance"]) for row in history) / len(history)),
        "gradient_cosine_mean": round_float(math.fsum(float(row["gradient_cosine"]) for row in history) / len(history)),
        "gradient_negative_rate_mean": round_float(math.fsum(float(row["gradient_cosine_negative_rate"]) for row in history) / len(history)),
        "validation_support_silence_end": last["validation_support_silence_rate"],
        "validation_write_entropy_end": last["validation_write_entropy"],
    }


def train_library(
    seed: int,
    system: str,
    code: str,
    teacher_style: str,
    loss_mode: str,
    baseline_library: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, list[dict[str, Any]]]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    dynamics_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        trained = train_producer_diagnostic(seed, skill, system, code, teacher_style, loss_mode, baseline_library[skill], contexts[skill], settings, out)
        library[skill] = trained["state"]
        contracts[skill] = trained["contract"]
        dynamics_rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "code": code,
                "teacher_style": teacher_style,
                "loss_mode": loss_mode,
                "state_hash": e7p.state_hash(trained["state"]),
                "summary": trained["summary"],
                "oracle_used_as_teacher_target": True,
                "oracle_used_at_inference": False,
            }
        )
        gradient_rows.extend(trained["history"])
        contract_rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "code": code,
                "teacher_style": teacher_style,
                "loss_mode": loss_mode,
                "contract": trained["contract_json"],
                "state_hash": e7p.state_hash(trained["state"]),
                "target_decomposition": trained["sensitivity"],
                "semantic_labels_used": False,
            }
        )
    return library, contracts, dynamics_rows, gradient_rows, contract_rows


def evaluate_system(
    seed: int,
    system: str,
    code: str,
    teacher_style: str,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    oracle: bool = False,
    dense: bool = False,
    params: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Current/simplified can use E8A directly. Custom teachers need E8C target
    # generation, so we do the evaluation here for all systems.
    evals: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        mae_values: list[float] = []
        written_values: list[float] = []
        target_values: list[float] = []
        next_errors: list[float] = []
        sign_mismatches: list[float] = []
        row_samples: list[dict[str, Any]] = []
        for row in task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            route = tuple(row["expected_route"])
            step_samples: list[dict[str, Any]] = []
            for step_idx, skill in enumerate(route):
                target_after = apply_e8c_teacher_code(row, canonical, skill, code if code != "continuous" else "int4", teacher_style)
                cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
                target = target_after[cells].astype(np.float32)
                if oracle:
                    pred = target_after.copy()
                else:
                    assert library is not None
                    if dense:
                        pred = e7p.np_forward(library[skill], flow.reshape(1, -1)).reshape(-1).astype(np.float32)
                    else:
                        assert contracts is not None
                        pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
                        if code != "continuous":
                            pred[cells] = e7z.quantize_bundle_values(pred[cells], skill, code, params, primary_is_logit=True)
                written = pred[cells].astype(np.float32)
                mae = np.abs(written - target)
                mae_values.extend(float(v) for v in mae.tolist())
                written_values.extend(float(v) for v in written.tolist())
                target_values.extend(float(v) for v in target.tolist())
                if len(written) > 1:
                    sign_mismatches.extend(float((w >= 0.0) != (t >= 0.0)) for w, t in zip(written[1:], target[1:]))
                if step_idx + 1 < len(route):
                    read = contracts[route[step_idx + 1]]["read"] if contracts and route[step_idx + 1] in contracts else np.ones(FLOW_DIM, dtype=bool)
                    next_errors.append(float(np.mean(np.abs(pred[read] - target_after[read]))))
                step_samples.append(
                    {
                        "skill": skill,
                        "bundle_mae": round_float(float(np.mean(mae))),
                        "next_error": round_float(next_errors[-1] if step_idx + 1 < len(route) and next_errors else 0.0),
                        "written_values": [round_float(float(v)) for v in written.tolist()],
                        "teacher_values": [round_float(float(v)) for v in target.tolist()],
                    }
                )
                flow = pred.astype(np.float32)
                canonical = target_after.astype(np.float32)
            predicted = int(flow[RESULT_POS[route[-1]]] >= 0.5)
            ok = predicted == int(row["target_answer"])
            correct.append(bool(ok))
            if len(row_samples) < 3:
                row_samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": predicted, "correct": bool(ok), "steps": step_samples})
        answer_acc = float(np.mean(correct)) if correct else 0.0
        bundle_mae = float(np.mean(mae_values)) if mae_values else 0.0
        next_error = float(np.mean(next_errors)) if next_errors else 0.0
        sign_mismatch = float(np.mean(sign_mismatches)) if sign_mismatches else 0.0
        usefulness = answer_acc - 0.10 - 0.12 * next_error
        compatibility = max(0.0, 1.0 - next_error)
        evals[split] = {
            "answer_accuracy": round_float(answer_acc),
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(usefulness),
            "consumer_compatibility_score": round_float(compatibility),
            "oracle_code_similarity": round_float(max(0.0, 1.0 - bundle_mae)),
            "bundle_mean_absolute_error_to_oracle": round_float(bundle_mae),
            "bundle_cellwise_correlation_with_oracle": e7z.safe_corr(written_values, target_values),
            "bundle_cosine_similarity_with_oracle": e7z.cosine_similarity(written_values, target_values),
            "support_channel_sign_mismatch_rate": round_float(sign_mismatch),
            "write_entropy": bundle_entropy(np.asarray(written_values, dtype=np.float32)),
            "next_pocket_compatibility_error": round_float(next_error),
            "mean_route_steps": round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])) if task[split] else 0.0),
            "row_level_samples": row_samples,
        }
        sample_rows.extend({"seed": seed, "system": system, "teacher_style": teacher_style, "code": code, "split": split, "row_id": sample["row_id"], "correct": sample["correct"], "steps": sample["steps"]} for sample in row_samples)
    result = {
        "seed": seed,
        "system": system,
        "code": code,
        "teacher_style": teacher_style,
        "evals": evals,
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_oracle_code_similarity": round_float(float(np.mean([evals[split]["oracle_code_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_consumer_compatibility_score": round_float(float(np.mean([evals[split]["consumer_compatibility_score"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["bundle_mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_compatibility_error"] for split in EVAL_SPLITS]))),
        "eval_mean_support_channel_sign_mismatch_rate": round_float(float(np.mean([evals[split]["support_channel_sign_mismatch_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_write_entropy": round_float(float(np.mean([evals[split]["write_entropy"] for split in EVAL_SPLITS]))),
        "bit_budget": e7z.estimate_boundary_bits(code if code != "continuous" else "int4"),
        "boundary_bit_budget": e7z.estimate_boundary_bits(code if code != "continuous" else "int4"),
    }
    return result, sample_rows


def aggregate_results(rows: list[dict[str, Any]], dynamics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = sorted((row for row in rows if row["system"] == system), key=lambda row: int(row.get("seed", -1)))
        if not subset:
            continue
        numeric_keys = sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
        mean = {key: round_float(math.fsum(float(row.get(key, 0.0)) for row in subset) / len(subset)) for key in numeric_keys}
        dyn = [row["summary"] for row in dynamics_rows if row["system"] == system and row.get("summary")]
        for metric in (
            "loss_drop",
            "validation_code_similarity_start",
            "validation_code_similarity_end",
            "validation_code_similarity_best",
            "final_gap_to_best",
            "tail_gain",
            "tail_range",
            "gradient_norm_mean",
            "gradient_variance_mean",
            "gradient_cosine_mean",
            "gradient_negative_rate_mean",
        ):
            if dyn:
                mean[metric] = round_float(math.fsum(float(row.get(metric, 0.0)) for row in dyn) / len(dyn))
        systems[system] = {"seed_count": len({row["seed"] for row in subset}), "mean": mean}
    candidates = [system for system in systems if system not in ORACLE_SYSTEMS and system != "dense_graph_danger_control"]
    best = max(candidates, key=lambda s: systems[s]["mean"].get("eval_mean_composition_usefulness", -1e9))
    return {"schema_version": "e8c_aggregate_metrics_v1", "systems": systems, "best_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    mean = {system: systems.get(system, {}).get("mean", {}) for system in SYSTEMS}
    usefulness = {system: mean.get(system, {}).get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    code_sim = {system: mean.get(system, {}).get("eval_mean_oracle_code_similarity", 0.0) for system in SYSTEMS}
    compatibility = {system: mean.get(system, {}).get("eval_mean_consumer_compatibility_score", 0.0) for system in SYSTEMS}
    grad_neg = {system: mean.get(system, {}).get("gradient_negative_rate_mean", 1.0) for system in TRAINED_SYSTEMS}
    current = usefulness.get("current_full_code_teacher_baseline", 0.0)
    current_compat = compatibility.get("current_full_code_teacher_baseline", 0.0)
    current_grad_neg = grad_neg.get("current_full_code_teacher_baseline", 1.0)
    best = aggregate["best_system"]
    best_score = aggregate["best_eval_mean_composition_usefulness"]
    decomposed_candidates = {
        "per_skill_decomposed_heads",
        "primary_then_support_staged_teacher",
        "support_cells_only_after_primary_plateau",
        "codebook_decomposed_targets",
        "low_conflict_batch_curriculum",
    }
    best_decomposed = max(decomposed_candidates, key=lambda system: usefulness.get(system, -1e9))
    grad_conflict = max(mean.get(system, {}).get("gradient_negative_rate_mean", 0.0) for system in TRAINED_SYSTEMS if system in mean) > 0.35
    train_low = max(mean.get(system, {}).get("validation_code_similarity_best", 0.0) for system in TRAINED_SYSTEMS if system in mean) < 0.84
    train_valid_gap = max(mean.get(system, {}).get("validation_code_similarity_best", 0.0) - code_sim.get(system, 0.0) for system in TRAINED_SYSTEMS if system in mean)
    best_grad_drop = round_float(current_grad_neg - grad_neg.get(best, current_grad_neg))
    best_compat_gain = round_float(compatibility.get(best, 0.0) - current_compat)
    useful_gain = round_float(best_score - current)
    positive = useful_gain >= 0.03 and best_grad_drop >= 0.10 and best_compat_gain >= 0.0
    if usefulness.get("dense_graph_danger_control", 0.0) >= best_score + 0.02:
        decision = "e8c_graph_soup_regression_detected"
    elif usefulness.get("mutation_only_decomposed_lowbit", 0.0) >= current + 0.03:
        decision = "e8c_mutation_only_decomposed_learning_viable"
    elif usefulness.get("mutation_repair_after_consumer_compatible_plateau", 0.0) >= usefulness.get("consumer_compatibility_weighted_loss", 0.0) + 0.02:
        decision = "e8c_mutation_repair_after_compatibility_plateau_positive"
    elif positive and best == "consumer_sensitivity_weighted_targets":
        decision = "e8c_consumer_sensitivity_weighting_positive"
    elif positive and best == "route_step_local_teacher_targets":
        decision = "e8c_route_step_local_targets_positive"
    elif positive and best in decomposed_candidates:
        decision = "e8c_target_decomposition_positive"
    elif best_grad_drop >= 0.10 and useful_gain < 0.03:
        decision = "e8c_gradient_conflict_reduced_but_usefulness_low"
    elif train_low:
        decision = "e8c_producer_architecture_bottleneck"
    elif usefulness.get("consumer_distill_reference", 0.0) >= best_score + 0.08:
        decision = "e8c_consumer_interface_bottleneck"
    else:
        decision = "e8c_current_code_interface_still_wrong"
    detail = {
        "best_system": best,
        "best_score": best_score,
        "current_full_code_teacher_baseline": current,
        "best_decomposed_system": best_decomposed,
        "best_decomposed_usefulness": usefulness.get(best_decomposed, 0.0),
        "best_usefulness_gain_over_current": useful_gain,
        "best_gradient_negative_rate_drop": best_grad_drop,
        "best_consumer_compatibility_gain": best_compat_gain,
        "consumer_sensitivity_weighted_targets": usefulness.get("consumer_sensitivity_weighted_targets", 0.0),
        "route_step_local_teacher_targets": usefulness.get("route_step_local_teacher_targets", 0.0),
        "consumer_compatibility_weighted_loss": usefulness.get("consumer_compatibility_weighted_loss", 0.0),
        "mutation_repair_after_consumer_compatible_plateau": usefulness.get("mutation_repair_after_consumer_compatible_plateau", 0.0),
        "mutation_only_decomposed_lowbit": usefulness.get("mutation_only_decomposed_lowbit", 0.0),
        "oracle_low_bit_reference": usefulness.get("oracle_low_bit_reference", 0.0),
        "consumer_distill_reference": usefulness.get("consumer_distill_reference", 0.0),
        "max_train_validation_gap_proxy": round_float(train_valid_gap),
        "gradient_conflict_flag": bool(grad_conflict),
    }
    return {"schema_version": "e8c_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    e7z_settings = to_e7z_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        baseline_library[skill] = e7p.copy_state(trained["state"], "E8C_baseline_standalone")

    context_cache: dict[tuple[str, str], dict[str, dict[str, list[dict[str, Any]]]]] = {}

    def contexts(code: str, style: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
        key = (code, style)
        if key not in context_cache:
            context_cache[key] = generate_context_tasks(composition_task, code, style)
        return context_cache[key]

    configs = {
        "current_full_code_teacher_baseline": ("int4", "current_full_code_teacher", "baseline"),
        "local_smooth_full_code_teacher": ("int4", "local_smooth_full_code_teacher", "baseline"),
        "per_skill_decomposed_heads": ("int4", "decomposed_primary_support_teacher", "decomposed"),
        "primary_then_support_staged_teacher": ("int4", "decomposed_primary_support_teacher", "staged"),
        "support_cells_only_after_primary_plateau": ("int4", "decomposed_primary_support_teacher", "support_plateau"),
        "consumer_sensitivity_weighted_targets": ("int4", "consumer_sensitivity_teacher", "consumer_sensitivity"),
        "route_step_local_teacher_targets": ("int4", "consumer_sensitivity_teacher", "route_step_local"),
        "codebook_decomposed_targets": ("int4", "codebook_decomposed_teacher", "codebook_decomposed"),
        "low_conflict_batch_curriculum": ("int4", "consumer_sensitivity_teacher", "low_conflict"),
        "consumer_compatibility_weighted_loss": ("int4", "current_full_code_teacher", "consumer_compatibility"),
        "mutation_repair_after_consumer_compatible_plateau": ("int4", "consumer_sensitivity_teacher", "consumer_compatibility"),
    }
    rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    curriculum_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    learned: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, str]] = {}

    for system, (code, style, loss_mode) in configs.items():
        lib, contracts, dyn, grad, contract_rows = train_library(seed, system, code, style, loss_mode, baseline_library, contexts(code, style), settings, out)
        learned[system] = (lib, contracts, code, style)
        dynamics_rows.extend(dyn)
        gradient_rows.extend(grad)
        teacher_rows.extend(contract_rows)
        loss_rows.extend(contract_rows)
        if system == "mutation_repair_after_consumer_compatible_plateau":
            if out:
                best_sim = max(row["summary"].get("validation_code_similarity_best", 0.0) for row in dyn)
                append_progress(out, "e8c_mutation_repair_seed_trained", seed=seed, system=system, best_validation_code_similarity=round_float(best_sim))
            continue
        result, samples = evaluate_system(seed, system, code, style, lib, contracts, composition_task)
        curriculum_rows.append({"seed": seed, "system": system, "consumer_compatibility_score": result["eval_mean_consumer_compatibility_score"], "next_pocket_compatibility_error": result["eval_mean_next_pocket_compatibility_error"], "gradient_diagnostic_batch_size": settings.gradient_diagnostic_batch_size})
        rows.append(result)
        sample_rows.extend(samples)
        if out:
            append_progress(out, "e8c_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"])

    # References and mutation controls.
    oracle_result, oracle_samples = evaluate_system(seed, "oracle_low_bit_reference", "binary", "current_full_code_teacher", None, None, composition_task, oracle=True)
    rows.append(oracle_result)
    sample_rows.extend(oracle_samples)
    current_contexts = contexts("binary", "current_full_code_teacher")
    lib, contracts, _, _ = learned["current_full_code_teacher_baseline"]
    consumer_result, consumer_rows = e8a.evaluate_consumer_read(seed, "consumer_distill_reference", "binary", lib, contracts, current_contexts)
    for split_eval in consumer_result.get("evals", {}).values():
        split_eval["write_entropy"] = 0.0
        split_eval["consumer_compatibility_score"] = 1.0
        split_eval.setdefault("next_pocket_compatibility_error", 0.0)
    consumer_result["eval_mean_write_entropy"] = 0.0
    consumer_result["eval_mean_consumer_compatibility_score"] = 1.0
    consumer_result.setdefault("eval_mean_next_pocket_compatibility_error", 0.0)
    rows.append(consumer_result)
    sample_rows.extend({"seed": seed, "system": "consumer_distill_reference", "teacher_style": "current_full_code_teacher", "code": "binary", "split": row["split"], "row_id": f"consumer:{seed}:{row['split']}", "correct": True, "steps": []} for row in consumer_rows[:8])

    rand_contracts = {skill: e7z.build_lowbit_contract(skill, "mutation_only_decomposed_lowbit", read_count=OUTPUT_WIDTH) for skill in SKILLS}
    params, mutation = e7z.repair_boundary(seed, "mutation_only_decomposed_lowbit", "int4", baseline_library, rand_contracts, composition_task, e7z_settings, out)
    result, samples = evaluate_system(seed, "mutation_only_decomposed_lowbit", "int4", "current_full_code_teacher", baseline_library, rand_contracts, composition_task, params=params)
    result.update({key: value for key, value in mutation.items() if key != "history"})
    rows.append(result)
    sample_rows.extend(samples)
    repair_rows.extend(mutation["history"])

    lib, contracts, _, style = learned["mutation_repair_after_consumer_compatible_plateau"]
    threshold_met = max(row["summary"].get("validation_code_similarity_best", 0.0) for row in dynamics_rows if row["system"] == "mutation_repair_after_consumer_compatible_plateau") >= settings.similarity_threshold
    params, mutation = e7z.repair_boundary(seed, "mutation_repair_after_consumer_compatible_plateau", "int4", lib, contracts, composition_task, e7z_settings, out)
    result, samples = evaluate_system(seed, "mutation_repair_after_consumer_compatible_plateau", "int4", style, lib, contracts, composition_task, params=params)
    base_score = next(row["eval_mean_composition_usefulness"] for row in rows if row["system"] == "consumer_compatibility_weighted_loss")
    result.update({key: value for key, value in mutation.items() if key != "history"})
    result["similarity_threshold_met"] = bool(threshold_met)
    result["mutation_repair_gain"] = round_float(result["eval_mean_composition_usefulness"] - base_score)
    rows.append(result)
    sample_rows.extend(samples)
    repair_rows.extend(mutation["history"])

    dense_result, dense_samples = evaluate_system(seed, "dense_graph_danger_control", "continuous", "current_full_code_teacher", baseline_library, None, composition_task, dense=True)
    rows.append(dense_result)
    sample_rows.extend(dense_samples)

    return {
        "seed": seed,
        "rows": rows,
        "dynamics_rows": dynamics_rows,
        "gradient_rows": gradient_rows,
        "teacher_rows": teacher_rows,
        "loss_rows": loss_rows,
        "curriculum_rows": curriculum_rows,
        "repair_rows": repair_rows,
        "sample_rows": sample_rows,
    }


def build_report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E8C Producer Target Decomposition And Consumer Compatibility Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {aggregate['best_system']}",
        f"deterministic_replay_passed = {decision.get('deterministic_replay_passed', False)}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"].get(system, {}).get("mean", {})
        lines.append(
            f"{system:<52} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"code_sim={mean.get('eval_mean_oracle_code_similarity', 0.0):.6f} "
            f"compat={mean.get('eval_mean_consumer_compatibility_score', 0.0):.6f} "
            f"val_best={mean.get('validation_code_similarity_best', 0.0):.6f} "
            f"grad_cos={mean.get('gradient_cosine_mean', 0.0):.6f} "
            f"grad_neg={mean.get('gradient_negative_rate_mean', 0.0):.6f} "
            f"tail={mean.get('tail_range', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E8C is a controlled numeric producer-write diagnostic. Oracle writes are teacher targets or diagnostic references only. Learned systems are evaluated without oracle writes at inference. No raw-language, AGI, consciousness, deployed-model, or model-scale claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(
    rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    curriculum_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    sensitivity_rows = [row for row in loss_rows if row.get("target_decomposition")]
    return {
        "target_decomposition_report.json": {"schema_version": "e8c_target_decomposition_report_v1", "rows": sorted(sensitivity_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "consumer_sensitivity_report.json": {"schema_version": "e8c_consumer_sensitivity_report_v1", "rows": sorted(sensitivity_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "producer_dynamics_report.json": {"schema_version": "e8c_producer_dynamics_report_v1", "rows": sorted(dynamics_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "gradient_diagnostics_report.json": {"schema_version": "e8c_gradient_diagnostics_report_v1", "rows": sorted(gradient_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", ""), r.get("epoch", -1)))},
        "compatibility_report.json": {"schema_version": "e8c_compatibility_report_v1", "rows": sorted(curriculum_rows, key=lambda r: (r["seed"], r["system"]))},
        "mutation_repair_report.json": {"schema_version": "e8c_mutation_repair_report_v1", "rows": sorted(repair_rows, key=lambda r: (r["seed"], r["system"], r["generation"]))},
        "system_results.json": {"schema_version": "e8c_system_results_v1", "rows": sorted(rows, key=lambda r: (r["seed"], SYSTEMS.index(r["system"])))},
        "row_level_samples.json": {"schema_version": "e8c_row_level_samples_v1", "rows": sorted(sample_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e8c_summary_v1", "decision": decision["decision"], "best_system": aggregate["best_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
        "deterministic_replay.json": deterministic,
        "report.md": build_report_text(decision, aggregate),
    }


def hash_artifacts(out: Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in HASH_ARTIFACTS}


def replay_command(settings: Settings, replay_out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(replay_out.relative_to(REPO_ROOT)),
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
        "--pocket-pretrain-rows-per-seed",
        str(settings.pocket_pretrain_rows_per_seed),
        "--pocket-validation-rows-per-seed",
        str(settings.pocket_validation_rows_per_seed),
        "--pocket-dim",
        str(settings.pocket_dim),
        "--pocket-core-steps",
        str(settings.pocket_core_steps),
        "--pocket-epochs",
        str(settings.pocket_epochs),
        "--local-epochs",
        str(settings.local_epochs),
        "--full-epochs",
        str(settings.full_epochs),
        "--batch-size",
        str(settings.batch_size),
        "--gradient-diagnostic-batch-size",
        str(settings.gradient_diagnostic_batch_size),
        "--learning-rate",
        str(settings.learning_rate),
        "--local-learning-rate",
        str(settings.local_learning_rate),
        "--weight-decay",
        str(settings.weight_decay),
        "--pruned-read-count",
        str(settings.pruned_read_count),
        "--repair-generations",
        str(settings.repair_generations),
        "--repair-population",
        str(settings.repair_population),
        "--similarity-threshold",
        str(settings.similarity_threshold),
        "--cpu-workers",
        str(settings.cpu_workers),
        "--device",
        settings.device,
        "--heartbeat-seconds",
        str(settings.heartbeat_seconds),
        "--execution-mode",
        settings.execution_mode,
        "--replay",
    ]


def run(settings: Settings, out: Path) -> dict[str, Any]:
    if out.exists() and not settings.replay:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "run_start", settings=settings_payload(settings))
    heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not heartbeat.wait(settings.heartbeat_seconds):
            e7r.append_jsonl(out / "hardware_heartbeat.jsonl", e7r.hardware_sample())

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        e7o_settings = to_e7o_settings(settings)
        composition_tasks = e7o.generate_composition_tasks(e7o_settings)
        pocket_tasks = e7o.generate_pocket_tasks(e7o_settings)
        write_json(
            out / "backend_manifest.json",
            {
                "schema_version": "e8c_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "teacher_styles": list(TEACHER_STYLES),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "semantic_lane_labels_as_model_input": False,
                "new_router": False,
                "oracle_write_at_inference_for_learned_systems": False,
                "oracle_used_as_teacher_target": True,
                "mutation_repair_uses_backprop": False,
                "gradient_diagnostics_logged": True,
                "training_performed": True,
                "device": select_device(settings.device),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )
        write_json(
            out / "task_generation_report.json",
            {
                "schema_version": "e8c_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        dynamics_rows: list[dict[str, Any]] = []
        gradient_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        loss_rows: list[dict[str, Any]] = []
        curriculum_rows: list[dict[str, Any]] = []
        repair_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        if settings.device == "cuda" and max_workers > 1:
            append_progress(out, "cuda_process_pool_disabled", requested_workers=max_workers, active_workers=1)
            max_workers = 1
        if max_workers == 1:
            for job in jobs:
                result = seed_worker(job)
                rows.extend(result["rows"])
                dynamics_rows.extend(result["dynamics_rows"])
                gradient_rows.extend(result["gradient_rows"])
                teacher_rows.extend(result["teacher_rows"])
                loss_rows.extend(result["loss_rows"])
                curriculum_rows.extend(result["curriculum_rows"])
                repair_rows.extend(result["repair_rows"])
                sample_rows.extend(result["sample_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_repair_rows": len(repair_rows), "last_completed": f"seed{job['seed']}", "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_repair_rows": len(repair_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        dynamics_rows.extend(result["dynamics_rows"])
                        gradient_rows.extend(result["gradient_rows"])
                        teacher_rows.extend(result["teacher_rows"])
                        loss_rows.extend(result["loss_rows"])
                        curriculum_rows.extend(result["curriculum_rows"])
                        repair_rows.extend(result["repair_rows"])
                        sample_rows.extend(result["sample_rows"])
                        append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_repair_rows": len(repair_rows), "last_completed": f"seed{job['seed']}", "pending": len(futures)})
        aggregate = aggregate_results(rows, dynamics_rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e8c_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, dynamics_rows, gradient_rows, teacher_rows, loss_rows, curriculum_rows, repair_rows, sample_rows, aggregate, decision, deterministic_placeholder)
        for name, payload in reports.items():
            if name.endswith(".json"):
                write_json(out / name, payload)
            else:
                write_text(out / name, str(payload))
        append_progress(out, "primary_artifacts_written", artifact_count=len(reports))
        if not settings.replay:
            primary_hashes = hash_artifacts(out)
            replay_out = out / "deterministic_replay_work"
            if replay_out.exists():
                shutil.rmtree(replay_out)
            append_progress(out, "deterministic_replay_start", replay_out=str(replay_out.relative_to(REPO_ROOT)))
            subprocess.run(replay_command(settings, replay_out), cwd=str(REPO_ROOT), check=True)
            replay_hashes = hash_artifacts(replay_out)
            comparisons = {name: {"primary": primary_hashes[name], "replay": replay_hashes.get(name), "match": primary_hashes[name] == replay_hashes.get(name)} for name in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e8c_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, dynamics_rows, gradient_rows, teacher_rows, loss_rows, curriculum_rows, repair_rows, sample_rows, aggregate, decision, deterministic)
            for name, payload in reports.items():
                if name.endswith(".json"):
                    write_json(out / name, payload)
                else:
                    write_text(out / name, str(payload))
            append_progress(out, "deterministic_replay_complete", internal_replay_passed=deterministic["internal_replay_passed"])
            append_progress(out, "final_artifacts_written", artifact_count=len(reports))
        else:
            write_json(out / "deterministic_replay.json", deterministic_placeholder)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return decision
    finally:
        heartbeat.set()
        thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=260)
    parser.add_argument("--validation-rows-per-seed", type=int, default=128)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=128)
    parser.add_argument("--ood-rows-per-seed", type=int, default=128)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=128)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=128)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=320)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=128)
    parser.add_argument("--pocket-dim", type=int, default=56)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=24)
    parser.add_argument("--local-epochs", type=int, default=16)
    parser.add_argument("--full-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-diagnostic-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pruned-read-count", type=int, default=30)
    parser.add_argument("--repair-generations", type=int, default=8)
    parser.add_argument("--repair-population", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(23, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        pocket_pretrain_rows_per_seed=args.pocket_pretrain_rows_per_seed,
        pocket_validation_rows_per_seed=args.pocket_validation_rows_per_seed,
        pocket_dim=args.pocket_dim,
        pocket_core_steps=args.pocket_core_steps,
        pocket_epochs=args.pocket_epochs,
        local_epochs=args.local_epochs,
        full_epochs=args.full_epochs,
        batch_size=args.batch_size,
        gradient_diagnostic_batch_size=args.gradient_diagnostic_batch_size,
        learning_rate=args.learning_rate,
        local_learning_rate=args.local_learning_rate,
        weight_decay=args.weight_decay,
        pruned_read_count=args.pruned_read_count,
        repair_generations=args.repair_generations,
        repair_population=args.repair_population,
        similarity_threshold=args.similarity_threshold,
        cpu_workers=args.cpu_workers,
        device=select_device(args.device),
        heartbeat_seconds=args.heartbeat_seconds,
        replay=args.replay,
        execution_mode=args.execution_mode,
    )
    run(settings, resolve_out(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
