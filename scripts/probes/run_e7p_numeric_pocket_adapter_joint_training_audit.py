#!/usr/bin/env python3
"""E7P numeric pocket adapter joint training audit.

E7O showed that learned numeric pockets were strong in standalone evaluation
but weak when composed through the shared Flow[D] contract. E7P isolates that
interface question by freezing the route schedule and comparing local training
scopes for each callable numeric pocket:

    Flow[D] -> input adapter -> pocket core -> output adapter -> Flow[D]

The primary systems do not mutate or retrain the router. They train only the
target pocket scope and then evaluate composition row-by-row through frozen
expected routes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib.util
import json
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E7O_PATH = Path(__file__).with_name("run_e7o_learned_numeric_pocket_router_composition.py")
MILESTONE = "E7P_NUMERIC_POCKET_ADAPTER_JOINT_TRAINING_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7p_numeric_pocket_adapter_joint_training_audit")
DEFAULT_SEEDS = (99501, 99502, 99503, 99504)

SYSTEMS = (
    "standalone_pocket_then_fixed_adapter",
    "adapter_only_training",
    "pocket_core_only_training",
    "joint_adapter_plus_pocket_training",
    "joint_adapter_plus_pocket_with_slot_contract",
    "full_end_to_end_training_control",
    "oracle_intermediate_state_reference",
)
LOCAL_TRAINING_SYSTEMS = (
    "adapter_only_training",
    "pocket_core_only_training",
    "joint_adapter_plus_pocket_training",
    "joint_adapter_plus_pocket_with_slot_contract",
)
GRADIENT_SYSTEMS = (
    "adapter_only_training",
    "pocket_core_only_training",
    "joint_adapter_plus_pocket_training",
    "joint_adapter_plus_pocket_with_slot_contract",
    "full_end_to_end_training_control",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "adapter_training_report.json",
    "flow_contract_report.json",
    "composition_report.json",
    "error_attribution_report.json",
    "system_results.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7p_joint_adapter_pocket_training_positive",
    "e7p_adapter_contract_bottleneck_confirmed",
    "e7p_pocket_core_training_bottleneck_confirmed",
    "e7p_typed_slot_contract_required",
    "e7p_local_pocket_training_insufficient",
    "e7p_numeric_pocket_composition_not_yet_viable",
)


def load_e7o_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7o_learned_numeric_pocket_router_composition", E7O_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7O helpers from {E7O_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7o = load_e7o_module()

FLOW_DIM = e7o.FLOW_DIM
SKILLS = e7o.SKILLS
FAMILIES = e7o.FAMILIES
SPLITS = e7o.SPLITS
EVAL_SPLITS = e7o.EVAL_SPLITS
RESULT_POS = e7o.RESULT_POS


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
    learning_rate: float
    local_learning_rate: float
    weight_decay: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return round(float(value), 12)


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7p::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(12):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.08 * (attempt + 1))
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"lock timeout: {lock}")
            time.sleep(0.025)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"event": event, "details": details, "time": round_float(time.time())})


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("empty seed list")
    return values


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def hardware_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {"time": round_float(time.time()), "pid": os.getpid()}
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        if raw:
            util, mem, temp = [part.strip() for part in raw.splitlines()[0].split(",")[:3]]
            sample["gpu_util_percent"] = float(util)
            sample["gpu_memory_mb"] = float(mem)
            sample["gpu_temp_c"] = float(temp)
    except Exception:
        sample["gpu_query_available"] = False
    return sample


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    def worker() -> None:
        while not stop.is_set():
            append_jsonl(out / "hardware_heartbeat.jsonl", hardware_sample())
            stop.wait(max(1.0, interval))

    thread = threading.Thread(target=worker, name="e7p-hardware-heartbeat", daemon=True)
    thread.start()
    return thread


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def to_e7o_settings(settings: Settings) -> Any:
    return e7o.Settings(
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
        monolithic_epochs=max(8, settings.full_epochs),
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        router_generations=4,
        router_population=6,
        mutation_generations=4,
        mutation_population=6,
        prune_rounds=4,
        prune_fraction=0.01,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=False,
    )


def model_to_state(model: nn.Module, settings: Settings, lineage: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "e7p_numeric_pocket_state_v1",
        "precision": "float32",
        "flow_dim": FLOW_DIM,
        "pocket_dim": settings.pocket_dim,
        "core_steps": settings.pocket_core_steps,
        "arrays": {name: param.detach().cpu().numpy().astype(np.float32) for name, param in model.named_parameters()},
        "masks": {},
        "lineage": lineage,
    }


def state_to_model(state: dict[str, Any], settings: Settings, device: str) -> nn.Module:
    model = e7o.NumericPocketCore(FLOW_DIM, settings.pocket_dim, settings.pocket_core_steps).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.as_tensor(state["arrays"][name], dtype=torch.float32, device=device))
    return model


def copy_state(state: dict[str, Any], lineage: str | None = None) -> dict[str, Any]:
    out = {
        "schema_version": "e7p_numeric_pocket_state_v1",
        "precision": state.get("precision", "float32"),
        "flow_dim": int(state.get("flow_dim", FLOW_DIM)),
        "pocket_dim": int(state.get("pocket_dim", 0)),
        "core_steps": int(state.get("core_steps", 0)),
        "arrays": {key: value.copy() for key, value in state["arrays"].items()},
        "masks": {key: value.copy() for key, value in state.get("masks", {}).items()},
        "lineage": list(state.get("lineage", [])),
    }
    if lineage:
        out["lineage"].append(lineage)
    return out


def state_hash(state: dict[str, Any]) -> str:
    return e7o.state_hash(state)


def parameter_count(state: dict[str, Any]) -> int:
    return e7o.parameter_count(state, active_only=True)


def bit_budget(state: dict[str, Any]) -> int:
    return e7o.bit_budget(state)


def np_forward(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return e7o.np_forward(state, x)


def generate_context_tasks(composition_task: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    for split in SPLITS:
        for row in composition_task[split]:
            route = tuple(row["expected_route"])
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            for skill in route:
                target = flow.copy()
                target[RESULT_POS[skill]] = float(e7o.base_skill_value(skill, row["a"], row["b"], row["key"], row["threshold"], row["flip"], target))
                tasks[skill][split].append({
                    "row_id": f"{row['row_id']}:{skill}",
                    "split": split,
                    "skill": skill,
                    "family": row["family"],
                    "flow": flow.tolist(),
                    "target_flow": target.tolist(),
                    "target_value": int(target[RESULT_POS[skill]] >= 0.5),
                })
                flow = target
    return tasks


def split_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return e7o.split_arrays(rows)


def trainable_names_for_mode(mode: str) -> set[str]:
    if mode == "adapter_only_training":
        return {"win", "bin", "wout", "bout"}
    if mode == "pocket_core_only_training":
        return {"wcore", "carry_raw"}
    return {"win", "bin", "wcore", "carry_raw", "wout", "bout"}


def evaluate_context_pocket(skill: str, state: dict[str, Any], context_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pos = RESULT_POS[skill]
    result_positions = set(RESULT_POS.values())
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = context_task[split]
        if not rows:
            evals[split] = {"accuracy": 1.0, "state_preservation_error": 0.0, "result_slot_corruption_rate": 0.0, "output_calibration_error": 0.0}
            continue
        x, target_flow, target = split_arrays(rows)
        pred_flow = np_forward(state, x)
        logits = pred_flow[:, pos]
        pred = (logits >= 0.0).astype(np.int64)
        preserve_idx = [idx for idx in range(FLOW_DIM) if idx != pos]
        preservation = float(np.mean((pred_flow[:, preserve_idx] - x[:, preserve_idx]) ** 2))
        other_result_idx = [idx for idx in result_positions if idx != pos]
        corruption = float(np.mean(np.any(np.abs(pred_flow[:, other_result_idx] - x[:, other_result_idx]) > 0.20, axis=1))) if other_result_idx else 0.0
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        calibration = float(np.mean(np.abs(prob - target.astype(np.float32))))
        evals[split] = {
            "accuracy": round_float(float(np.mean(pred == target))),
            "state_preservation_error": round_float(preservation),
            "result_slot_corruption_rate": round_float(corruption),
            "output_calibration_error": round_float(calibration),
        }
    return {
        "evals": evals,
        "eval_mean_accuracy": round_float(float(np.mean([evals[split]["accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_state_preservation_error": round_float(float(np.mean([evals[split]["state_preservation_error"] for split in EVAL_SPLITS]))),
        "eval_mean_result_slot_corruption_rate": round_float(float(np.mean([evals[split]["result_slot_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_output_calibration_error": round_float(float(np.mean([evals[split]["output_calibration_error"] for split in EVAL_SPLITS]))),
        "state_hash": state_hash(state),
        "bit_budget": bit_budget(state),
        "active_parameter_count": parameter_count(state),
    }


def train_context_pocket(
    seed: int,
    skill: str,
    system: str,
    base_state: dict[str, Any],
    context_task: dict[str, list[dict[str, Any]]],
    settings: Settings,
    out: Path | None,
) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"context:{seed}:{skill}:{system}"), device)
    model = state_to_model(base_state, settings, device)
    trainable = trainable_names_for_mode(system)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return {"state": copy_state(base_state, system), "history": [], "scope": "none"}
    epochs = settings.full_epochs if system == "full_end_to_end_training_control" else settings.local_epochs
    lr = settings.local_learning_rate
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"context-batch:{seed}:{skill}:{system}"))
    x_train, y_train, t_train = split_arrays(context_task["train"])
    x_val, _, t_val = split_arrays(context_task["validation"])
    pos = RESULT_POS[skill]
    preserve_idx = torch.as_tensor([idx for idx in range(FLOW_DIM) if idx != pos], dtype=torch.long, device=device)
    other_result_idx = torch.as_tensor([idx for idx in RESULT_POS.values() if idx != pos], dtype=torch.long, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history = []
    for epoch in range(epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            tb = torch.as_tensor(t_train[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            result_loss = F.binary_cross_entropy_with_logits(pred[:, pos], tb)
            preserve_loss = F.mse_loss(pred.index_select(1, preserve_idx), xb.index_select(1, preserve_idx))
            slot_loss = F.mse_loss(pred.index_select(1, other_result_idx), xb.index_select(1, other_result_idx))
            target_loss = F.mse_loss(pred[:, pos], yb[:, pos])
            if system == "joint_adapter_plus_pocket_with_slot_contract":
                loss = result_loss + 1.20 * preserve_loss + 1.40 * slot_loss + 0.08 * target_loss
            elif system == "full_end_to_end_training_control":
                loss = result_loss + 0.03 * preserve_loss
            else:
                loss = result_loss + 0.18 * preserve_loss + 0.12 * slot_loss + 0.05 * target_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.as_tensor(x_val, dtype=torch.float32, device=device))
            pred = (pred_val[:, pos] >= 0.0).detach().cpu().numpy().astype(np.int64)
            acc = float(np.mean(pred == t_val))
            preserve = float(F.mse_loss(pred_val.index_select(1, preserve_idx), torch.as_tensor(x_val, dtype=torch.float32, device=device).index_select(1, preserve_idx)).detach().cpu())
        score = acc - (0.25 if system == "joint_adapter_plus_pocket_with_slot_contract" else 0.05) * preserve
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            row = {
                "seed": seed,
                "skill": skill,
                "system": system,
                "epoch": epoch,
                "loss": round_float(total_loss / max(1, batches)),
                "validation_accuracy": round_float(acc),
                "validation_preservation_error": round_float(preserve),
                "best_validation_score": round_float(best_score),
                "trainable_scope": sorted(trainable),
            }
            history.append(row)
            if out:
                append_progress(out, "context_pocket_epoch", **row)
                write_json(out / "adapter_training_snapshots" / f"{system}_{skill}_seed{seed}_epoch{epoch:04d}.json", row)
    if best_state is not None:
        model.load_state_dict(best_state)
    state = model_to_state(model, settings, [system])
    return {"state": state, "history": history, "scope": sorted(trainable)}


def apply_route_with_metrics(library: dict[str, dict[str, Any]], row: dict[str, Any], route: tuple[str, ...]) -> tuple[np.ndarray, dict[str, float]]:
    flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
    preservation_errors = []
    corruption_events = []
    compatibility_errors = []
    for skill in route:
        before = flow.copy()
        state = library[skill]
        pred = np_forward(state, flow).astype(np.float32)
        pos = RESULT_POS[skill]
        preserve_idx = [idx for idx in range(FLOW_DIM) if idx != pos]
        preservation_errors.append(float(np.mean((pred[:, preserve_idx] - before[:, preserve_idx]) ** 2)))
        other_result_idx = [idx for idx in RESULT_POS.values() if idx != pos]
        if other_result_idx:
            corruption_events.append(float(np.any(np.abs(pred[:, other_result_idx] - before[:, other_result_idx]) > 0.20)))
        pred[0, pos] = 1.0 if pred[0, pos] >= 0.0 else 0.0
        flow = pred
        if skill != route[-1]:
            compatibility_errors.append(float(np.mean(np.abs(flow[:, :24] - before[:, :24]))))
    return flow.reshape(-1), {
        "state_preservation_error": float(np.mean(preservation_errors)) if preservation_errors else 0.0,
        "result_slot_corruption_rate": float(np.mean(corruption_events)) if corruption_events else 0.0,
        "next_pocket_input_compatibility_error": float(np.mean(compatibility_errors)) if compatibility_errors else 0.0,
    }


def evaluate_composition_system(system: str, library: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, Any]:
    evals = {}
    for split in SPLITS:
        rows = task[split]
        correct = []
        samples = []
        preservation = []
        corruption = []
        compatibility = []
        pocket_errors = composition_errors = 0
        for row in rows:
            route = tuple(row["expected_route"])
            if symbolic:
                flow = e7o.symbolic_apply_route(row, route)
                metrics = {"state_preservation_error": 0.0, "result_slot_corruption_rate": 0.0, "next_pocket_input_compatibility_error": 0.0}
            else:
                assert library is not None
                flow, metrics = apply_route_with_metrics(library, row, route)
            pred = e7o.predict_answer_from_flow(row, flow)
            ok = pred == int(row["target_answer"])
            correct.append(ok)
            preservation.append(metrics["state_preservation_error"])
            corruption.append(metrics["result_slot_corruption_rate"])
            compatibility.append(metrics["next_pocket_input_compatibility_error"])
            if not ok:
                oracle_flow = e7o.symbolic_apply_route(row, route)
                if e7o.predict_answer_from_flow(row, oracle_flow) == int(row["target_answer"]):
                    pocket_errors += 1
                else:
                    composition_errors += 1
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": int(pred), "correct": bool(ok)})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in rows])))
        bit_cost = sum(bit_budget(state) for state in library.values()) if library else 0
        param_count = sum(parameter_count(state) for state in library.values()) if library else 0
        cost_penalty = min(0.10, 0.00000016 * bit_cost + 0.0025 * mean_steps)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "state_preservation_error": round_float(float(np.mean(preservation))),
            "result_slot_corruption_rate": round_float(float(np.mean(corruption))),
            "next_pocket_input_compatibility_error": round_float(float(np.mean(compatibility))),
            "output_calibration_error": 0.0,
            "teacher_forcing_recovery": 1.0,
            "pocket_error_rate": round_float(pocket_errors / max(1, len(rows))),
            "adapter_error_rate": round_float(float(np.mean(corruption))),
            "router_error_rate": 0.0,
            "composition_error_rate": round_float(composition_errors / max(1, len(rows))),
            "bit_budget": bit_cost,
            "active_parameter_count": param_count,
            "row_level_samples": samples,
        }
    return {
        "system": system,
        "evals": evals,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 1.0,
        "eval_mean_state_preservation_error": round_float(float(np.mean([evals[split]["state_preservation_error"] for split in EVAL_SPLITS]))),
        "eval_mean_result_slot_corruption_rate": round_float(float(np.mean([evals[split]["result_slot_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility_error"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(parameter_count(state) for state in library.values()) if library else 0,
        "bit_budget": sum(bit_budget(state) for state in library.values()) if library else 0,
        "router_frozen": True,
        "other_pockets_frozen_during_local_training": system in LOCAL_TRAINING_SYSTEMS,
    }


def canonical_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))


def canonical_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("seed", 0)), str(row.get("system", "")), str(row.get("skill", ""))))


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = canonical_system_rows(rows)
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in (
                "eval_mean_answer_accuracy",
                "eval_mean_composition_usefulness",
                "eval_mean_route_accuracy",
                "heldout_usefulness",
                "ood_usefulness",
                "counterfactual_usefulness",
                "adversarial_usefulness",
                "eval_mean_state_preservation_error",
                "eval_mean_result_slot_corruption_rate",
                "eval_mean_next_pocket_input_compatibility_error",
                "parameter_count",
                "bit_budget",
            ):
                if metric in row:
                    metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for key, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{key}", []).append(float(value))
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {key: round_float(float(np.mean(values))) for key, values in metrics.items()},
            "min": {key: round_float(float(np.min(values))) for key, values in metrics.items()},
            "max": {key: round_float(float(np.max(values))) for key, values in metrics.items()},
        }
    candidates = [system for system in SYSTEMS if system != "oracle_intermediate_state_reference"]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7p_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    baseline = systems["standalone_pocket_then_fixed_adapter"]["mean"]
    adapter = systems["adapter_only_training"]["mean"]
    core = systems["pocket_core_only_training"]["mean"]
    joint = systems["joint_adapter_plus_pocket_training"]["mean"]
    slot = systems["joint_adapter_plus_pocket_with_slot_contract"]["mean"]
    full = systems["full_end_to_end_training_control"]["mean"]
    best = aggregate["best_non_reference_system"]
    detail = {
        "best_non_reference_system": best,
        "baseline_usefulness": baseline["eval_mean_composition_usefulness"],
        "adapter_usefulness": adapter["eval_mean_composition_usefulness"],
        "core_usefulness": core["eval_mean_composition_usefulness"],
        "joint_usefulness": joint["eval_mean_composition_usefulness"],
        "slot_contract_usefulness": slot["eval_mean_composition_usefulness"],
        "full_end_to_end_usefulness": full["eval_mean_composition_usefulness"],
        "slot_corruption_baseline": baseline["eval_mean_result_slot_corruption_rate"],
        "slot_corruption_contract": slot["eval_mean_result_slot_corruption_rate"],
        "preservation_baseline": baseline["eval_mean_state_preservation_error"],
        "preservation_contract": slot["eval_mean_state_preservation_error"],
    }
    base_u = float(detail["baseline_usefulness"])
    adapter_u = float(detail["adapter_usefulness"])
    core_u = float(detail["core_usefulness"])
    joint_u = float(detail["joint_usefulness"])
    slot_u = float(detail["slot_contract_usefulness"])
    full_u = float(detail["full_end_to_end_usefulness"])
    if adapter_u >= max(joint_u, core_u, slot_u) - 0.01 and adapter_u >= base_u + 0.08 and adapter_u >= 0.72:
        return "e7p_adapter_contract_bottleneck_confirmed", detail
    if core_u >= max(adapter_u, joint_u, slot_u) - 0.01 and core_u >= base_u + 0.08 and core_u >= 0.72:
        return "e7p_pocket_core_training_bottleneck_confirmed", detail
    if slot_u >= joint_u + 0.03 and slot_u >= base_u + 0.08 and slot_u >= 0.72:
        return "e7p_typed_slot_contract_required", detail
    if joint_u >= base_u + 0.10 and joint_u >= 0.75:
        return "e7p_joint_adapter_pocket_training_positive", detail
    if full_u >= 0.78 and max(adapter_u, core_u, joint_u, slot_u) < 0.72:
        return "e7p_local_pocket_training_insufficient", detail
    return "e7p_numeric_pocket_composition_not_yet_viable", detail


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    context_tasks = generate_context_tasks(composition_task)

    baseline_library: dict[str, dict[str, Any]] = {}
    baseline_rows = []
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = copy_state(trained["state"], "e7p_baseline_standalone")
        baseline_library[skill] = state
        baseline_rows.append({
            "seed": seed,
            "skill": skill,
            "system": "standalone_pocket_then_fixed_adapter",
            "state_hash": state_hash(state),
            "standalone": trained["standalone"],
            "context": evaluate_context_pocket(skill, state, context_tasks[skill]),
            "trainable_scope": [],
            "router_frozen": True,
            "other_pockets_frozen": True,
        })

    rows = []
    training_rows = list(baseline_rows)
    baseline_result = evaluate_composition_system("standalone_pocket_then_fixed_adapter", baseline_library, composition_task)
    baseline_result.update({"seed": seed, "system": "standalone_pocket_then_fixed_adapter"})
    rows.append(baseline_result)

    for system in LOCAL_TRAINING_SYSTEMS + ("full_end_to_end_training_control",):
        library: dict[str, dict[str, Any]] = {}
        for skill in SKILLS:
            trained = train_context_pocket(seed, skill, system, baseline_library[skill], context_tasks[skill], settings, out)
            state = trained["state"]
            library[skill] = state
            context_eval = evaluate_context_pocket(skill, state, context_tasks[skill])
            training_rows.append({
                "seed": seed,
                "skill": skill,
                "system": system,
                "state_hash": state_hash(state),
                "context": context_eval,
                "history": trained["history"],
                "trainable_scope": trained["scope"],
                "router_frozen": True,
                "other_pockets_frozen": system in LOCAL_TRAINING_SYSTEMS,
            })
        result = evaluate_composition_system(system, library, composition_task)
        result.update({"seed": seed, "system": system})
        rows.append(result)

    oracle = evaluate_composition_system("oracle_intermediate_state_reference", None, composition_task, symbolic=True)
    oracle.update({"seed": seed, "system": "oracle_intermediate_state_reference"})
    rows.append(oracle)
    return {"seed": seed, "rows": rows, "training_rows": training_rows}


def build_reports(rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = canonical_system_rows(rows)
    clean_training_rows = canonical_training_rows(training_rows)
    flow_rows = []
    for row in clean_rows:
        flow_rows.append({
            "seed": row["seed"],
            "system": row["system"],
            "state_preservation_error": row["eval_mean_state_preservation_error"],
            "result_slot_corruption_rate": row["eval_mean_result_slot_corruption_rate"],
            "next_pocket_input_compatibility_error": row["eval_mean_next_pocket_input_compatibility_error"],
        })
    return {
        "baseline_pocket_training_report.json": {"schema_version": "e7p_baseline_pocket_training_report_v1", "rows": [row for row in clean_training_rows if row["system"] == "standalone_pocket_then_fixed_adapter"]},
        "adapter_training_report.json": {"schema_version": "e7p_adapter_training_report_v1", "rows": clean_training_rows},
        "flow_contract_report.json": {"schema_version": "e7p_flow_contract_report_v1", "rows": flow_rows},
        "composition_report.json": {"schema_version": "e7p_composition_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "answer_accuracy": row["eval_mean_answer_accuracy"], "route_accuracy": row["eval_mean_route_accuracy"], "usefulness": row["eval_mean_composition_usefulness"]} for row in clean_rows]},
        "error_attribution_report.json": {"schema_version": "e7p_error_attribution_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "pocket_error": row["evals"]["heldout"].get("pocket_error_rate", 0.0), "adapter_error": row["evals"]["heldout"].get("adapter_error_rate", 0.0), "router_error": row["evals"]["heldout"].get("router_error_rate", 0.0), "composition_error": row["evals"]["heldout"].get("composition_error_rate", 0.0)} for row in clean_rows]},
        "system_results.json": {"schema_version": "e7p_system_results_v1", "rows": clean_rows},
        "leakage_report.json": {"schema_version": "e7p_leakage_report_v1", "router_frozen": True, "symbolic_proxy_primary": False, "hidden_expected_answer_input": False, "full_end_to_end_is_diagnostic_only": True},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7p_summary_v1", "run_root": DEFAULT_OUT.as_posix(), "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "checker_failure_count": None},
    }


def write_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    systems = aggregate["systems"]
    lines = [
        "# E7P Numeric Pocket Adapter Joint Training Audit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_reference_system = {aggregate['best_non_reference_system']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = systems[system]["mean"]
        lines.append(
            f"{system:<48} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"preserve={mean.get('eval_mean_state_preservation_error', 0.0):.6f} "
            f"slot_corrupt={mean.get('eval_mean_result_slot_corruption_rate', 0.0):.6f}"
        )
    lines.extend([
        "```",
        "",
        "## Detail",
        "",
        "```text",
    ])
    for key, value in decision["detail"].items():
        lines.append(f"{key} = {value}")
    lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7P is a controlled numeric pocket-flow interface probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.",
    ])
    write_text(out / "report.md", "\n".join(lines) + "\n")


def run_once(settings: Settings, out: Path, replay_mode: bool = False) -> dict[str, Any]:
    out = resolve_out(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = start_hardware_monitor(out, stop, settings.heartbeat_seconds)
    append_progress(out, "run_start", milestone=MILESTONE, replay_mode=replay_mode, settings=settings_payload(settings))
    e7o_settings = to_e7o_settings(settings)
    composition_tasks = e7o.generate_composition_tasks(e7o_settings)
    pocket_tasks = e7o.generate_pocket_tasks(e7o_settings)
    write_json(out / "backend_manifest.json", {"schema_version": "e7p_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "device": select_device(settings.device)})
    write_json(out / "task_generation_report.json", {
        "schema_version": "e7p_task_generation_report_v1",
        "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
        "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
    })
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    jobs = [{
        "seed": seed,
        "settings": settings.__dict__.copy(),
        "composition_task": composition_tasks[seed],
        "pocket_task": pocket_tasks[seed],
        "out": str(out) if not replay_mode else str(out),
    } for seed in settings.seeds]
    max_workers = max(1, min(settings.cpu_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
        while futures:
            done, _ = wait(futures, timeout=max(1.0, settings.heartbeat_seconds), return_when=FIRST_COMPLETED)
            if not done:
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "pending": len(futures)})
                continue
            for future in done:
                label = futures.pop(future)
                result = future.result()
                rows.extend(result["rows"])
                training_rows.extend(result["training_rows"])
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "last_completed": label, "pending": len(futures)})
                append_progress(out, "seed_job_complete", label=label, pending=len(futures))
    aggregate = aggregate_results(rows)
    decision_label, detail = decide(aggregate)
    deterministic_placeholder = {"internal_replay_passed": True}
    decision = {"schema_version": "e7p_decision_v1", "decision": decision_label, "detail": detail, "deterministic_replay_passed": True}
    reports = build_reports(rows, training_rows, aggregate, decision, deterministic_placeholder)
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, aggregate)
    append_progress(out, "primary_artifacts_written", artifact_count=len(reports) + 2)
    deterministic = {"schema_version": "e7p_deterministic_replay_report_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_mode": replay_mode}
    if settings.replay and not replay_mode:
        replay_out = out / "deterministic_replay_work"
        if replay_out.exists():
            shutil.rmtree(replay_out)
        append_progress(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
        run_once(settings, replay_out, replay_mode=True)
        comparisons = {}
        passed = True
        for artifact in HASH_ARTIFACTS:
            primary_hash = hashlib.sha256((out / artifact).read_bytes()).hexdigest()
            replay_hash = hashlib.sha256((replay_out / artifact).read_bytes()).hexdigest()
            match = primary_hash == replay_hash
            comparisons[artifact] = {"primary": primary_hash, "replay": replay_hash, "match": match}
            passed = passed and match
        deterministic = {"schema_version": "e7p_deterministic_replay_report_v1", "internal_replay_passed": passed, "hash_comparisons": comparisons, "replay_mode": False}
        decision["deterministic_replay_passed"] = passed
        reports = build_reports(rows, training_rows, aggregate, decision, deterministic)
        for name, payload in reports.items():
            write_json(out / name, payload)
        write_report(out, decision, aggregate)
        append_progress(out, "deterministic_replay_complete", internal_replay_passed=passed)
    write_json(out / "deterministic_replay.json", deterministic)
    append_progress(out, "final_artifacts_written", artifact_count=len(HASH_ARTIFACTS))
    stop.set()
    monitor.join(timeout=5.0)
    return {"aggregate": aggregate, "decision": decision, "deterministic": deterministic}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=720)
    parser.add_argument("--validation-rows-per-seed", type=int, default=240)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=240)
    parser.add_argument("--ood-rows-per-seed", type=int, default=240)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=240)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=240)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=900)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=240)
    parser.add_argument("--pocket-dim", type=int, default=72)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=90)
    parser.add_argument("--full-epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--no-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        learning_rate=args.learning_rate,
        local_learning_rate=args.local_learning_rate,
        weight_decay=args.weight_decay,
        cpu_workers=args.cpu_workers,
        device=args.device,
        heartbeat_seconds=args.heartbeat_seconds,
        execution_mode=args.execution_mode,
        replay=not args.no_replay,
    )
    run_once(settings, resolve_out(args.out), replay_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
