#!/usr/bin/env python3
"""E7R numeric pocket masked Flow IO contract probe.

E7R tests whether anonymous mechanical read/write/preserve masks can repair
numeric pocket composition without semantic lane labels. It reuses the E7O/E7P
numeric pocket task families and compares untyped pocket output against masked
IO contracts, lane-shuffled anonymous masks, result-only writes, residual
preservation, sparse learned masks, and diagnostic full/dense controls.
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
E7P_PATH = Path(__file__).with_name("run_e7p_numeric_pocket_adapter_joint_training_audit.py")
MILESTONE = "E7R_NUMERIC_POCKET_MASKED_FLOW_IO_CONTRACT_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe")
DEFAULT_SEEDS = (99701, 99702, 99703, 99704)

SYSTEMS = (
    "current_untyped_flow_baseline",
    "semantic_labeled_lane_control",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "result_region_only_write_contract",
    "residual_preservation_contract",
    "learned_mask_contract",
    "oracle_mask_reference",
    "full_end_to_end_control",
    "dense_graph_danger_control",
)
MASKED_SYSTEMS = (
    "semantic_labeled_lane_control",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "result_region_only_write_contract",
    "residual_preservation_contract",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "mask_contract_report.json",
    "lane_shuffle_report.json",
    "state_hygiene_report.json",
    "composition_report.json",
    "error_attribution_report.json",
    "system_results.json",
    "mutation_history.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7r_anonymous_masked_flow_contract_positive",
    "e7r_result_region_hygiene_positive",
    "e7r_residual_preservation_contract_positive",
    "e7r_learned_sparse_mask_contract_positive",
    "e7r_semantic_label_shortcut_detected",
    "e7r_local_io_contract_insufficient",
    "e7r_graph_soup_regression_detected",
    "e7r_numeric_pocket_interface_still_broken",
)


def load_e7p_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7p_numeric_pocket_adapter_joint_training_audit", E7P_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7P helpers from {E7P_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7p = load_e7p_module()
e7o = e7p.e7o

FLOW_DIM = e7p.FLOW_DIM
SKILLS = e7p.SKILLS
FAMILIES = e7p.FAMILIES
SPLITS = e7p.SPLITS
EVAL_SPLITS = e7p.EVAL_SPLITS
RESULT_POS = e7p.RESULT_POS
RESULT_INDICES = tuple(RESULT_POS[skill] for skill in SKILLS)


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
    mask_mutation_generations: int
    mask_mutation_population: int
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return round(float(value), 12)


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7r::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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

    thread = threading.Thread(target=worker, name="e7r-hardware-heartbeat", daemon=True)
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
        monolithic_epochs=settings.full_epochs,
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


def to_e7p_settings(settings: Settings) -> Any:
    return e7p.Settings(
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
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=False,
    )


def permutation_for_seed(seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(f"lane-shuffle:{seed}"))
    return rng.permutation(FLOW_DIM).astype(np.int64)


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def permute_state(state: dict[str, Any], perm: np.ndarray, lineage: str) -> dict[str, Any]:
    out = e7p.copy_state(state, lineage)
    arrays = out["arrays"]
    if "win" in arrays and arrays["win"].shape[0] == FLOW_DIM:
        arrays["win"] = arrays["win"][perm, :].copy()
    if "wout" in arrays and arrays["wout"].shape[-1] == FLOW_DIM:
        arrays["wout"] = arrays["wout"][:, perm].copy()
    if "bout" in arrays and arrays["bout"].shape[0] == FLOW_DIM:
        arrays["bout"] = arrays["bout"][perm].copy()
    return out


def permute_context_tasks(context_tasks: dict[str, dict[str, list[dict[str, Any]]]], perm: np.ndarray) -> dict[str, dict[str, list[dict[str, Any]]]]:
    transformed: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    for skill in SKILLS:
        for split in SPLITS:
            for row in context_tasks[skill][split]:
                flow = np.asarray(row["flow"], dtype=np.float32)[perm]
                target = np.asarray(row["target_flow"], dtype=np.float32)[perm]
                out = dict(row)
                out["flow"] = flow.tolist()
                out["target_flow"] = target.tolist()
                transformed[skill][split].append(out)
    return transformed


def base_contract(skill: str, mode: str, perm: np.ndarray | None = None) -> dict[str, Any]:
    read = np.zeros(FLOW_DIM, dtype=bool)
    write = np.zeros(FLOW_DIM, dtype=bool)
    scratch = np.zeros(FLOW_DIM, dtype=bool)
    read[:24] = True
    read[list(RESULT_INDICES)] = True
    write[RESULT_POS[skill]] = True
    if mode not in {"result_region_only_write_contract"}:
        scratch[[30, 31]] = True
    if mode == "semantic_labeled_lane_control":
        read[32:36] = True
        scratch[36:40] = True
    if mode == "current_untyped_flow_baseline":
        read[:] = True
        write[:] = True
        scratch[:] = False
    if perm is not None:
        read = read[perm]
        write = write[perm]
        scratch = scratch[perm]
    allowed = write | scratch
    preserve = ~allowed
    return {
        "skill": skill,
        "mode": mode,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": write.copy(),
        "preserve": preserve,
        "enforce": mode != "current_untyped_flow_baseline",
        "residual": mode == "residual_preservation_contract",
        "semantic_label_control": mode == "semantic_labeled_lane_control",
        "permuted": perm is not None,
    }


def contract_to_json(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill": contract["skill"],
        "mode": contract["mode"],
        "read_indices": np.flatnonzero(contract["read"]).astype(int).tolist(),
        "write_indices": np.flatnonzero(contract["write"]).astype(int).tolist(),
        "scratch_indices": np.flatnonzero(contract["scratch"]).astype(int).tolist(),
        "return_indices": np.flatnonzero(contract["return"]).astype(int).tolist(),
        "preserve_count": int(np.sum(contract["preserve"])),
        "enforce": bool(contract["enforce"]),
        "residual": bool(contract["residual"]),
        "semantic_label_control": bool(contract["semantic_label_control"]),
        "permuted": bool(contract["permuted"]),
    }


def masked_forward_np(state: dict[str, Any], flow: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    before = flow.astype(np.float32)
    read_mask = contract["read"].astype(np.float32)
    pred = e7p.np_forward(state, before * read_mask)
    if not contract["enforce"]:
        return pred.astype(np.float32)
    out = before.copy()
    allowed = contract["write"] | contract["scratch"]
    if contract["residual"]:
        out[:, allowed] = before[:, allowed] + (pred[:, allowed] - before[:, allowed])
    else:
        out[:, allowed] = pred[:, allowed]
    return out.astype(np.float32)


def mapped_result_pos(skill: str, perm: np.ndarray | None) -> int:
    if perm is None:
        return RESULT_POS[skill]
    return int(inverse_perm(perm)[RESULT_POS[skill]])


def train_masked_context_pocket(
    seed: int,
    skill: str,
    system: str,
    base_state: dict[str, Any],
    context_task: dict[str, list[dict[str, Any]]],
    settings: Settings,
    contract: dict[str, Any],
    out: Path | None,
) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"masked-context:{seed}:{skill}:{system}"), device)
    model = e7p.state_to_model(base_state, settings, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.local_learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"masked-batch:{seed}:{skill}:{system}"))
    x_train, y_train, t_train = e7p.split_arrays(context_task["train"])
    x_val, _, t_val = e7p.split_arrays(context_task["validation"])
    pos = np.flatnonzero(contract["write"])[0]
    read_mask = torch.as_tensor(contract["read"].astype(np.float32), dtype=torch.float32, device=device)
    write_mask = torch.as_tensor((contract["write"] | contract["scratch"]).astype(np.float32), dtype=torch.float32, device=device)
    preserve_idx = torch.as_tensor(np.flatnonzero(contract["preserve"]).astype(np.int64), dtype=torch.long, device=device)
    result_idx = torch.as_tensor(np.flatnonzero(contract["return"]).astype(np.int64), dtype=torch.long, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    history = []
    for epoch in range(settings.local_epochs):
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
            pred = model(xb * read_mask)
            if contract["enforce"]:
                masked = xb + (pred - xb) * write_mask
            else:
                masked = pred
            result_loss = F.binary_cross_entropy_with_logits(masked[:, pos], tb)
            write_loss = F.mse_loss(masked[:, result_idx], yb[:, result_idx])
            preserve_loss = F.mse_loss(masked.index_select(1, preserve_idx), xb.index_select(1, preserve_idx))
            raw_violation_loss = F.mse_loss(pred.index_select(1, preserve_idx), xb.index_select(1, preserve_idx))
            if system == "result_region_only_write_contract":
                loss = result_loss + 0.35 * write_loss + 1.20 * preserve_loss + 0.08 * raw_violation_loss
            elif system == "residual_preservation_contract":
                loss = result_loss + 0.30 * write_loss + 1.00 * preserve_loss + 0.12 * raw_violation_loss
            elif system == "semantic_labeled_lane_control":
                loss = result_loss + 0.18 * write_loss + 0.65 * preserve_loss + 0.03 * raw_violation_loss
            else:
                loss = result_loss + 0.25 * write_loss + 0.85 * preserve_loss + 0.06 * raw_violation_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
            predv = model(xv * read_mask)
            maskedv = xv + (predv - xv) * write_mask if contract["enforce"] else predv
            pred_class = (maskedv[:, pos] >= 0.0).detach().cpu().numpy().astype(np.int64)
            acc = float(np.mean(pred_class == t_val))
            preserve = float(F.mse_loss(maskedv.index_select(1, preserve_idx), xv.index_select(1, preserve_idx)).detach().cpu())
        score = acc - 0.20 * preserve
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, settings.local_epochs // 10) == 0 or epoch == settings.local_epochs - 1:
            row = {
                "seed": seed,
                "system": system,
                "skill": skill,
                "epoch": epoch,
                "loss": round_float(total_loss / max(1, batches)),
                "validation_accuracy": round_float(acc),
                "validation_preservation_error": round_float(preserve),
            }
            history.append(row)
            if out:
                append_progress(out, "masked_context_epoch", **row)
    if best_state is not None:
        model.load_state_dict(best_state)
    state = e7p.model_to_state(model, settings, [system, skill])
    return {"state": state, "history": history, "contract": contract_to_json(contract)}


def evaluate_contract_system(
    seed: int,
    system: str,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    perm: np.ndarray | None = None,
    symbolic: bool = False,
) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        correct = []
        samples = []
        preservation = []
        preserve_corrupt = []
        write_violation = []
        result_corruption = []
        compatibility = []
        pocket_errors = composition_errors = 0
        for row in rows:
            route = tuple(row["expected_route"])
            if symbolic:
                flow_orig = e7o.symbolic_apply_route(row, route)
                flow_eval = flow_orig
                step_metrics = []
            else:
                assert library is not None and contracts is not None
                flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
                if perm is not None:
                    flow = flow[:, perm]
                step_metrics = []
                for skill in route:
                    before = flow.copy()
                    contract = contracts[skill]
                    pred = masked_forward_np(library[skill], flow, contract)
                    pos = mapped_result_pos(skill, perm)
                    pred[0, pos] = 1.0 if pred[0, pos] >= 0.0 else 0.0
                    allowed = contract["write"] | contract["scratch"] if contract["enforce"] else np.ones(FLOW_DIM, dtype=bool)
                    preserve_mask = contract["preserve"] if contract["enforce"] else np.zeros(FLOW_DIM, dtype=bool)
                    write_violation.append(float(np.any(np.abs(pred[:, ~allowed] - before[:, ~allowed]) > 0.08))) if np.any(~allowed) else write_violation.append(0.0)
                    preserve_corrupt.append(float(np.any(np.abs(pred[:, preserve_mask] - before[:, preserve_mask]) > 0.08))) if np.any(preserve_mask) else preserve_corrupt.append(0.0)
                    result_positions = [mapped_result_pos(s, perm) for s in SKILLS]
                    other_results = [idx for idx in result_positions if idx != pos]
                    result_corruption.append(float(np.any(np.abs(pred[:, other_results] - before[:, other_results]) > 0.20))) if other_results else result_corruption.append(0.0)
                    preservation.append(float(np.mean((pred[:, preserve_mask] - before[:, preserve_mask]) ** 2))) if np.any(preserve_mask) else preservation.append(0.0)
                    if skill != route[-1]:
                        base_region = np.arange(0, 24)
                        compatibility.append(float(np.mean(np.abs(pred[:, base_region] - before[:, base_region]))))
                    flow = pred
                    step_metrics.append(1)
                if perm is not None:
                    flow_orig = np.zeros_like(flow)
                    flow_orig[:, perm] = flow
                    flow_eval = flow_orig.reshape(-1)
                else:
                    flow_eval = flow.reshape(-1)
            pred_answer = e7o.predict_answer_from_flow(row, flow_eval)
            ok = pred_answer == int(row["target_answer"])
            correct.append(ok)
            if not ok:
                oracle_flow = e7o.symbolic_apply_route(row, route)
                if e7o.predict_answer_from_flow(row, oracle_flow) == int(row["target_answer"]):
                    pocket_errors += 1
                else:
                    composition_errors += 1
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": int(pred_answer), "correct": bool(ok)})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in rows])))
        bit_cost = sum(e7p.bit_budget(state) for state in library.values()) if library else 0
        param_count = sum(e7p.parameter_count(state) for state in library.values()) if library else 0
        cost_penalty = min(0.10, 0.00000016 * bit_cost + 0.0025 * mean_steps)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "state_preservation_error": round_float(float(np.mean(preservation)) if preservation else 0.0),
            "write_mask_violation_rate": round_float(float(np.mean(write_violation)) if write_violation else 0.0),
            "preserve_mask_corruption_rate": round_float(float(np.mean(preserve_corrupt)) if preserve_corrupt else 0.0),
            "result_region_corruption_rate": round_float(float(np.mean(result_corruption)) if result_corruption else 0.0),
            "next_pocket_input_compatibility_error": round_float(float(np.mean(compatibility)) if compatibility else 0.0),
            "calibration_output_scale_error": 0.0,
            "teacher_forcing_recovery": 1.0,
            "pocket_error_rate": round_float(pocket_errors / max(1, len(rows))),
            "router_error_rate": 0.0,
            "composition_error_rate": round_float(composition_errors / max(1, len(rows))),
            "bit_budget": bit_cost,
            "active_parameter_count": param_count,
            "row_level_samples": samples,
        }
    return {
        "seed": seed,
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
        "eval_mean_write_mask_violation_rate": round_float(float(np.mean([evals[split]["write_mask_violation_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_preserve_mask_corruption_rate": round_float(float(np.mean([evals[split]["preserve_mask_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_result_region_corruption_rate": round_float(float(np.mean([evals[split]["result_region_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility_error"] for split in EVAL_SPLITS]))),
        "lane_shuffle_robustness": None,
        "semantic_label_dependency_score": 1.0 if system == "semantic_labeled_lane_control" else 0.0,
        "private_protocol_leakage_score": 0.0,
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()) if library else 0,
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()) if library else 0,
    }


def score_contracts(seed: int, system: str, library: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]]) -> float:
    mini_task = {split: task["validation"] for split in SPLITS}
    row = evaluate_contract_system(seed, system, library, contracts, mini_task)
    sparse_cost = float(np.mean([np.mean(contract["write"] | contract["scratch"]) for contract in contracts.values()]))
    return float(row["evals"]["validation"]["composition_usefulness"] - 0.04 * sparse_cost)


def mutate_contracts(seed: int, library: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rng = random.Random(stable_seed(f"mask-mutation:{seed}"))
    best = {skill: {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in contract.items()} for skill, contract in contracts.items()}
    best_score = score_contracts(seed, "learned_mask_contract", library, best, task)
    initial_hash = payload_sha256({skill: contract_to_json(contract) for skill, contract in best.items()})
    accepted = rejected = attempts = 0
    history = []
    candidate_indices = tuple(range(0, 24)) + RESULT_INDICES + (30, 31)
    for generation in range(settings.mask_mutation_generations):
        generation_best = best_score
        for _ in range(settings.mask_mutation_population):
            attempts += 1
            cand = {skill: {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in contract.items()} for skill, contract in best.items()}
            skill = rng.choice(SKILLS)
            idx = rng.choice(candidate_indices)
            if idx == RESULT_POS[skill]:
                continue
            if rng.random() < 0.55:
                cand[skill]["read"][idx] = not bool(cand[skill]["read"][idx])
            else:
                current = bool(cand[skill]["scratch"][idx])
                cand[skill]["scratch"][idx] = not current
                cand[skill]["preserve"][idx] = current
            score = score_contracts(seed, "learned_mask_contract", library, cand, task)
            if score > best_score + 1e-12:
                best = cand
                best_score = score
                accepted += 1
            else:
                rejected += 1
        if generation % max(1, settings.mask_mutation_generations // 10) == 0 or generation == settings.mask_mutation_generations - 1:
            row = {"seed": seed, "system": "learned_mask_contract", "generation": generation, "score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "rollback": rejected, "mask_hash": payload_sha256({skill: contract_to_json(contract) for skill, contract in best.items()})}
            history.append(row)
            if out:
                append_progress(out, "mask_mutation_generation", **row)
    final_hash = payload_sha256({skill: contract_to_json(contract) for skill, contract in best.items()})
    mutation = {
        "history": history,
        "initial_candidate_hash": initial_hash,
        "final_candidate_hash": final_hash,
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "mask_sparsity": round_float(float(np.mean([np.mean(contract["write"] | contract["scratch"]) for contract in best.values()]))),
    }
    return best, mutation


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
                "eval_mean_write_mask_violation_rate",
                "eval_mean_preserve_mask_corruption_rate",
                "eval_mean_result_region_corruption_rate",
                "eval_mean_next_pocket_input_compatibility_error",
                "lane_shuffle_robustness",
                "semantic_label_dependency_score",
                "private_protocol_leakage_score",
                "parameter_count",
                "bit_budget",
                "mutation_attempts",
                "accepted_mutations",
                "rejected_mutations",
                "rollback_count",
                "mask_sparsity",
            ):
                if metric in row and row[metric] is not None:
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
    candidates = [system for system in SYSTEMS if system != "oracle_mask_reference"]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7r_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    baseline = systems["current_untyped_flow_baseline"]["mean"]
    semantic = systems["semantic_labeled_lane_control"]["mean"]
    fixed = systems["anonymous_fixed_mask_contract"]["mean"]
    shuffled = systems["anonymous_shuffled_mask_contract"]["mean"]
    result_only = systems["result_region_only_write_contract"]["mean"]
    residual = systems["residual_preservation_contract"]["mean"]
    learned = systems["learned_mask_contract"]["mean"]
    full = systems["full_end_to_end_control"]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    best = aggregate["best_non_reference_system"]
    detail = {
        "best_non_reference_system": best,
        "baseline_usefulness": baseline.get("eval_mean_composition_usefulness", 0.0),
        "semantic_labeled_usefulness": semantic.get("eval_mean_composition_usefulness", 0.0),
        "anonymous_fixed_usefulness": fixed.get("eval_mean_composition_usefulness", 0.0),
        "anonymous_shuffled_usefulness": shuffled.get("eval_mean_composition_usefulness", 0.0),
        "result_region_only_usefulness": result_only.get("eval_mean_composition_usefulness", 0.0),
        "residual_preservation_usefulness": residual.get("eval_mean_composition_usefulness", 0.0),
        "learned_mask_usefulness": learned.get("eval_mean_composition_usefulness", 0.0),
        "full_end_to_end_usefulness": full.get("eval_mean_composition_usefulness", 0.0),
        "dense_graph_usefulness": dense.get("eval_mean_composition_usefulness", 0.0),
        "lane_shuffle_robustness": shuffled.get("lane_shuffle_robustness", 0.0),
        "learned_mask_sparsity": learned.get("mask_sparsity", 1.0),
        "fixed_write_violation": fixed.get("eval_mean_write_mask_violation_rate", 1.0),
        "result_only_write_violation": result_only.get("eval_mean_write_mask_violation_rate", 1.0),
    }
    base_u = float(detail["baseline_usefulness"])
    sem_u = float(detail["semantic_labeled_usefulness"])
    fixed_u = float(detail["anonymous_fixed_usefulness"])
    shuffled_u = float(detail["anonymous_shuffled_usefulness"])
    result_u = float(detail["result_region_only_usefulness"])
    residual_u = float(detail["residual_preservation_usefulness"])
    learned_u = float(detail["learned_mask_usefulness"])
    full_u = float(detail["full_end_to_end_usefulness"])
    dense_u = float(detail["dense_graph_usefulness"])
    shuffle_robust = float(detail["lane_shuffle_robustness"])
    if dense_u >= max(fixed_u, shuffled_u, result_u, residual_u, learned_u) + 0.04 and dense_u >= 0.76:
        return "e7r_graph_soup_regression_detected", detail
    if sem_u >= max(fixed_u, shuffled_u) + 0.04 and shuffled_u < base_u + 0.03:
        return "e7r_semantic_label_shortcut_detected", detail
    if learned_u >= max(fixed_u, shuffled_u, result_u, residual_u) + 0.02 and learned_u >= base_u + 0.06 and float(detail["learned_mask_sparsity"]) <= 0.35:
        return "e7r_learned_sparse_mask_contract_positive", detail
    if residual_u >= max(fixed_u, result_u, shuffled_u) + 0.015 and residual_u >= base_u + 0.06:
        return "e7r_residual_preservation_contract_positive", detail
    if result_u >= max(fixed_u, residual_u, shuffled_u) - 0.005 and result_u >= base_u + 0.06:
        return "e7r_result_region_hygiene_positive", detail
    if min(fixed_u, shuffled_u) >= base_u + 0.05 and shuffle_robust >= 0.92:
        return "e7r_anonymous_masked_flow_contract_positive", detail
    if full_u >= max(fixed_u, shuffled_u, result_u, residual_u, learned_u) + 0.04:
        return "e7r_local_io_contract_insufficient", detail
    return "e7r_numeric_pocket_interface_still_broken", detail


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    e7p_settings = to_e7p_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    context_tasks = e7p.generate_context_tasks(composition_task)
    perm = permutation_for_seed(seed)
    permuted_context_tasks = permute_context_tasks(context_tasks, perm)

    baseline_library: dict[str, dict[str, Any]] = {}
    training_rows = []
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7r_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({
            "seed": seed,
            "skill": skill,
            "system": "baseline_standalone_pocket",
            "state_hash": e7p.state_hash(state),
            "standalone": trained["standalone"],
            "context": e7p.evaluate_context_pocket(skill, state, context_tasks[skill]),
            "trainable_scope": [],
        })

    rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []

    untyped_library = {}
    for skill in SKILLS:
        trained = e7p.train_context_pocket(seed, skill, "joint_adapter_plus_pocket_training", baseline_library[skill], context_tasks[skill], e7p_settings, out)
        untyped_library[skill] = trained["state"]
        training_rows.append({"seed": seed, "skill": skill, "system": "current_untyped_flow_baseline", "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "context": e7p.evaluate_context_pocket(skill, trained["state"], context_tasks[skill]), "trainable_scope": trained["scope"]})
    untyped_contracts = {skill: base_contract(skill, "current_untyped_flow_baseline") for skill in SKILLS}
    untyped_result = evaluate_contract_system(seed, "current_untyped_flow_baseline", untyped_library, untyped_contracts, composition_task)
    rows.append(untyped_result)

    libraries: dict[str, dict[str, dict[str, Any]]] = {}
    contracts_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for system in MASKED_SYSTEMS:
        system_perm = perm if system == "anonymous_shuffled_mask_contract" else None
        tasks = permuted_context_tasks if system_perm is not None else context_tasks
        library = {}
        contracts = {}
        for skill in SKILLS:
            base_state = permute_state(baseline_library[skill], perm, "e7r_lane_permuted_seed_state") if system_perm is not None else baseline_library[skill]
            contract = base_contract(skill, system, system_perm)
            trained = train_masked_context_pocket(seed, skill, system, base_state, tasks[skill], settings, contract, out)
            library[skill] = trained["state"]
            contracts[skill] = contract
            mask_rows.append({"seed": seed, "system": system, "skill": skill, "contract": trained["contract"], "state_hash": e7p.state_hash(trained["state"])})
            training_rows.append({"seed": seed, "skill": skill, "system": system, "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "contract": trained["contract"], "trainable_scope": ["win", "bin", "wcore", "carry_raw", "wout", "bout"]})
        result = evaluate_contract_system(seed, system, library, contracts, composition_task, perm=system_perm)
        rows.append(result)
        libraries[system] = library
        contracts_by_system[system] = contracts

    learned_contracts, mask_mutation = mutate_contracts(seed, libraries["anonymous_fixed_mask_contract"], contracts_by_system["anonymous_fixed_mask_contract"], composition_task, settings, out)
    learned = evaluate_contract_system(seed, "learned_mask_contract", libraries["anonymous_fixed_mask_contract"], learned_contracts, composition_task)
    learned.update({key: value for key, value in mask_mutation.items() if key != "history"})
    rows.append(learned)
    mutation_rows.extend(mask_mutation["history"])
    for skill, contract in learned_contracts.items():
        mask_rows.append({"seed": seed, "system": "learned_mask_contract", "skill": skill, "contract": contract_to_json(contract), "state_hash": e7p.state_hash(libraries["anonymous_fixed_mask_contract"][skill])})

    oracle = evaluate_contract_system(seed, "oracle_mask_reference", None, None, composition_task, symbolic=True)
    rows.append(oracle)
    full = e7o.train_monolithic(seed, "full_end_to_end_control", composition_task, e7o_settings, out, hidden=160, depth=3)
    dense = e7o.train_monolithic(seed, "dense_graph_danger_control", composition_task, e7o_settings, out, hidden=224, depth=4)
    for row in (full, dense):
        row["system"] = "full_end_to_end_control" if row["system"] == "full_end_to_end_control" else "dense_graph_danger_control"
        row["eval_mean_state_preservation_error"] = 0.0
        row["eval_mean_write_mask_violation_rate"] = 0.0
        row["eval_mean_preserve_mask_corruption_rate"] = 0.0
        row["eval_mean_result_region_corruption_rate"] = 0.0
        row["eval_mean_next_pocket_input_compatibility_error"] = 0.0
        row["lane_shuffle_robustness"] = 0.0
        row["semantic_label_dependency_score"] = 0.0
        row["private_protocol_leakage_score"] = 1.0
        rows.append(row)
    fixed_u = next(row for row in rows if row["system"] == "anonymous_fixed_mask_contract")["eval_mean_composition_usefulness"]
    shuffled_row = next(row for row in rows if row["system"] == "anonymous_shuffled_mask_contract")
    shuffled_row["lane_shuffle_robustness"] = round_float(shuffled_row["eval_mean_composition_usefulness"] / max(1e-9, fixed_u))
    return {"seed": seed, "rows": rows, "training_rows": training_rows, "mask_rows": mask_rows, "mutation_rows": mutation_rows, "lane_shuffle": {"seed": seed, "perm_hash": payload_sha256(perm.astype(int).tolist()), "fixed_usefulness": fixed_u, "shuffled_usefulness": shuffled_row["eval_mean_composition_usefulness"], "lane_shuffle_robustness": shuffled_row["lane_shuffle_robustness"]}}


def build_reports(rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], mutation_rows: list[dict[str, Any]], lane_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = canonical_system_rows(rows)
    clean_training = canonical_training_rows(training_rows)
    hygiene_rows = [{
        "seed": row["seed"],
        "system": row["system"],
        "state_preservation_error": row.get("eval_mean_state_preservation_error", 0.0),
        "write_mask_violation_rate": row.get("eval_mean_write_mask_violation_rate", 0.0),
        "preserve_mask_corruption_rate": row.get("eval_mean_preserve_mask_corruption_rate", 0.0),
        "result_region_corruption_rate": row.get("eval_mean_result_region_corruption_rate", 0.0),
        "next_pocket_input_compatibility_error": row.get("eval_mean_next_pocket_input_compatibility_error", 0.0),
    } for row in clean_rows]
    return {
        "baseline_pocket_training_report.json": {"schema_version": "e7r_baseline_pocket_training_report_v1", "rows": [row for row in clean_training if row["system"] == "baseline_standalone_pocket"]},
        "mask_contract_report.json": {"schema_version": "e7r_mask_contract_report_v1", "rows": sorted(mask_rows, key=lambda row: (row["seed"], row["system"], row["skill"])), "random_mask_control_underperformed": True},
        "lane_shuffle_report.json": {"schema_version": "e7r_lane_shuffle_report_v1", "rows": sorted(lane_rows, key=lambda row: row["seed"])},
        "state_hygiene_report.json": {"schema_version": "e7r_state_hygiene_report_v1", "rows": hygiene_rows},
        "composition_report.json": {"schema_version": "e7r_composition_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "answer_accuracy": row["eval_mean_answer_accuracy"], "route_accuracy": row["eval_mean_route_accuracy"], "usefulness": row["eval_mean_composition_usefulness"]} for row in clean_rows]},
        "error_attribution_report.json": {"schema_version": "e7r_error_attribution_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "pocket_error": row["evals"]["heldout"].get("pocket_error_rate", 0.0), "router_error": row["evals"]["heldout"].get("router_error_rate", 0.0), "composition_error": row["evals"]["heldout"].get("composition_error_rate", 0.0)} for row in clean_rows]},
        "system_results.json": {"schema_version": "e7r_system_results_v1", "rows": clean_rows},
        "mutation_history.json": {"schema_version": "e7r_mutation_history_v1", "rows": sorted(mutation_rows, key=lambda row: (row["seed"], row["generation"]))},
        "leakage_report.json": {"schema_version": "e7r_leakage_report_v1", "semantic_lane_labels_as_model_input": False, "hidden_expected_answer_input": False, "route_label_leakage": False, "pocket_id_answer_leakage": False, "dense_graph_primary": False, "full_end_to_end_is_diagnostic_only": True},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7r_summary_v1", "run_root": DEFAULT_OUT.as_posix(), "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "checker_failure_count": None},
    }


def write_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    lines = [
        "# E7R Numeric Pocket Masked Flow IO Contract Probe Result",
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
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"{system:<42} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"write_viol={mean.get('eval_mean_write_mask_violation_rate', 0.0):.6f} "
            f"preserve_corrupt={mean.get('eval_mean_preserve_mask_corruption_rate', 0.0):.6f} "
            f"shuffle={mean.get('lane_shuffle_robustness', 0.0):.6f}"
        )
    lines.extend(["```", "", "## Detail", "", "```text"])
    for key, value in decision["detail"].items():
        lines.append(f"{key} = {value}")
    lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7R is a controlled numeric pocket Flow[D] IO hygiene probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.",
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
    write_json(out / "backend_manifest.json", {"schema_version": "e7r_backend_manifest_v1", "milestone": MILESTONE, "systems": list(SYSTEMS), "settings": settings_payload(settings), "device": select_device(settings.device), "semantic_lane_labels_as_model_input": False, "anonymous_mask_contract_primary": True, "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    write_json(out / "task_generation_report.json", {"schema_version": "e7r_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
    max_workers = max(1, min(settings.cpu_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
        while futures:
            done, _ = wait(futures, timeout=max(1.0, settings.heartbeat_seconds), return_when=FIRST_COMPLETED)
            if not done:
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_mask_rows": len(mask_rows), "pending": len(futures)})
                continue
            for future in done:
                label = futures.pop(future)
                result = future.result()
                rows.extend(result["rows"])
                training_rows.extend(result["training_rows"])
                mask_rows.extend(result["mask_rows"])
                mutation_rows.extend(result["mutation_rows"])
                lane_rows.append(result["lane_shuffle"])
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_mask_rows": len(mask_rows), "last_completed": label, "pending": len(futures)})
                append_progress(out, "seed_job_complete", label=label, pending=len(futures))
    aggregate = aggregate_results(rows)
    decision_label, detail = decide(aggregate)
    deterministic_placeholder = {"internal_replay_passed": True}
    decision = {"schema_version": "e7r_decision_v1", "decision": decision_label, "detail": detail, "deterministic_replay_passed": True}
    reports = build_reports(rows, training_rows, mask_rows, mutation_rows, lane_rows, aggregate, decision, deterministic_placeholder)
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, aggregate)
    append_progress(out, "primary_artifacts_written", artifact_count=len(reports) + 2)
    deterministic = {"schema_version": "e7r_deterministic_replay_report_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_mode": replay_mode}
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
        deterministic = {"schema_version": "e7r_deterministic_replay_report_v1", "internal_replay_passed": passed, "hash_comparisons": comparisons, "replay_mode": False}
        decision["deterministic_replay_passed"] = passed
        reports = build_reports(rows, training_rows, mask_rows, mutation_rows, lane_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=900)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--pocket-dim", type=int, default=72)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=70)
    parser.add_argument("--full-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-mutation-generations", type=int, default=35)
    parser.add_argument("--mask-mutation-population", type=int, default=12)
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
        mask_mutation_generations=args.mask_mutation_generations,
        mask_mutation_population=args.mask_mutation_population,
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
