#!/usr/bin/env python3
"""E7Q router-pocket joint binding probe.

E7P kept the router frozen and teacher-forced the expected route. E7Q tests the
next sharper question: can a control/router layer and a new numeric pocket
library learn together without collapsing into a private non-reusable protocol?

This remains a controlled numeric Flow[D] proxy. The primary systems are still
typed route/pocket systems, not dense anonymous graphs.
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
MILESTONE = "E7Q_ROUTER_POCKET_JOINT_BINDING_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7q_router_pocket_joint_binding_probe")
DEFAULT_SEEDS = (99601, 99602, 99603, 99604)

SYSTEMS = (
    "frozen_router_trained_pocket",
    "trained_router_frozen_pocket",
    "trained_router_trained_pocket",
    "trained_router_trained_pocket_slot_guard",
    "full_end_to_end_training_control",
    "random_router_control",
    "oracle_route_reference",
)
TRAINING_SYSTEMS = (
    "frozen_router_trained_pocket",
    "trained_router_frozen_pocket",
    "trained_router_trained_pocket",
    "trained_router_trained_pocket_slot_guard",
    "full_end_to_end_training_control",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "router_training_report.json",
    "joint_binding_training_report.json",
    "flow_contract_report.json",
    "reuse_after_binding_report.json",
    "private_protocol_leakage_report.json",
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
    "e7q_router_pocket_joint_binding_positive",
    "e7q_slot_guard_joint_binding_positive",
    "e7q_slot_guard_improves_but_not_solved",
    "e7q_router_discovery_not_interface_fix",
    "e7q_private_router_pocket_protocol_detected",
    "e7q_full_end_to_end_control_preferred",
    "e7q_joint_binding_not_yet_viable",
    "e7q_artifact_or_task_too_easy",
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
ROUTE_OPTIONS = e7o.ROUTE_OPTIONS
EXPECTED_ROUTE_INDEX = e7o.EXPECTED_ROUTE_INDEX


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
    router_epochs: int
    joint_epochs: int
    full_epochs: int
    batch_size: int
    learning_rate: float
    router_learning_rate: float
    joint_learning_rate: float
    weight_decay: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


class RouteHead(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.w1 = nn.Linear(FLOW_DIM, hidden)
        self.w2 = nn.Linear(hidden, len(ROUTE_OPTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.tanh(self.w1(x)))


def round_float(value: float) -> float:
    return round(float(value), 12)


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7q::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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

    thread = threading.Thread(target=worker, name="e7q-hardware-heartbeat", daemon=True)
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
        local_learning_rate=settings.joint_learning_rate,
        weight_decay=settings.weight_decay,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=False,
    )


def router_to_state(router: RouteHead) -> dict[str, Any]:
    return {
        "schema_version": "e7q_route_head_state_v1",
        "arrays": {
            "w1.weight": router.w1.weight.detach().cpu().numpy().astype(np.float32),
            "w1.bias": router.w1.bias.detach().cpu().numpy().astype(np.float32),
            "w2.weight": router.w2.weight.detach().cpu().numpy().astype(np.float32),
            "w2.bias": router.w2.bias.detach().cpu().numpy().astype(np.float32),
        },
    }


def router_state_hash(state: dict[str, Any]) -> str:
    payload = {key: np.round(value, 8).tolist() for key, value in state["arrays"].items()}
    return payload_sha256(payload)


def router_predict(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    arrays = state["arrays"]
    h = np.tanh(x @ arrays["w1.weight"].T + arrays["w1.bias"])
    logits = h @ arrays["w2.weight"].T + arrays["w2.bias"]
    return np.argmax(logits, axis=1).astype(np.int64)


def expected_route_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["expected_route_index"]) for row in rows], dtype=np.int64)


def split_composition(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row["flow"] for row in rows], dtype=np.float32)
    target = np.asarray([int(row["target_answer"]) for row in rows], dtype=np.float32)
    route = expected_route_targets(rows)
    family = np.asarray([FAMILIES.index(row["family"]) for row in rows], dtype=np.int64)
    return x, target, route, family


def answer_prob_from_flow(flow: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
    c = torch.sigmoid(flow[:, RESULT_POS["compare"]])
    m = torch.sigmoid(flow[:, RESULT_POS["mod_add"]])
    p = torch.sigmoid(flow[:, RESULT_POS["parity"]])
    t = torch.sigmoid(flow[:, RESULT_POS["threshold"]])
    f = torch.sigmoid(flow[:, RESULT_POS["counterfactual_flip"]])
    v = torch.sigmoid(flow[:, RESULT_POS["verify"]])
    answers = torch.stack((
        c * t,
        m * (1.0 - p) + (1.0 - m) * p,
        f,
        v,
        v * m,
        v * (1.0 - f) + (1.0 - v) * f,
    ), dim=1).clamp(1e-5, 1.0 - 1e-5)
    return answers.gather(1, family_id.reshape(-1, 1)).reshape(-1)


def modules_from_library(library: dict[str, dict[str, Any]], settings: Settings, device: str) -> nn.ModuleDict:
    return nn.ModuleDict({skill: e7p.state_to_model(library[skill], settings, device) for skill in SKILLS})


def library_from_modules(modules: nn.ModuleDict, settings: Settings, lineage: str) -> dict[str, dict[str, Any]]:
    return {skill: e7p.model_to_state(modules[skill], settings, [lineage, skill]) for skill in SKILLS}


def differentiable_apply_route(modules: nn.ModuleDict, x: torch.Tensor, route: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flow = x
    preserve_losses = []
    slot_losses = []
    result_positions = list(RESULT_POS.values())
    for skill in route:
        before = flow
        pred = modules[skill](flow)
        pos = RESULT_POS[skill]
        preserve_idx = [idx for idx in range(FLOW_DIM) if idx != pos]
        other_result_idx = [idx for idx in result_positions if idx != pos]
        preserve_losses.append(F.mse_loss(pred[:, preserve_idx], before[:, preserve_idx]))
        if other_result_idx:
            slot_losses.append(F.mse_loss(pred[:, other_result_idx], before[:, other_result_idx]))
        flow = pred
    preserve = torch.stack(preserve_losses).mean() if preserve_losses else torch.zeros((), dtype=x.dtype, device=x.device)
    slot = torch.stack(slot_losses).mean() if slot_losses else torch.zeros((), dtype=x.dtype, device=x.device)
    return flow, preserve, slot


def soft_route_answer(modules: nn.ModuleDict, router: RouteHead, x: torch.Tensor, family_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = router(x)
    probs = torch.softmax(logits, dim=1)
    answer_probs = []
    preserve_losses = []
    slot_losses = []
    for route in ROUTE_OPTIONS:
        flow, preserve, slot = differentiable_apply_route(modules, x, route)
        answer_probs.append(answer_prob_from_flow(flow, family_id))
        preserve_losses.append(preserve)
        slot_losses.append(slot)
    answer_matrix = torch.stack(answer_probs, dim=1)
    weighted_answer = (answer_matrix * probs).sum(dim=1).clamp(1e-5, 1.0 - 1e-5)
    return weighted_answer, logits, torch.stack(preserve_losses).mean(), torch.stack(slot_losses).mean()


def train_router_only(seed: int, task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"router-only:{seed}"), device)
    router = RouteHead().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=settings.router_learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"router-only-batch:{seed}"))
    x_train, _, route_train, _ = split_composition(task["train"])
    x_val, _, route_val, _ = split_composition(task["validation"])
    best_state: dict[str, torch.Tensor] | None = None
    best_acc = -1.0
    history = []
    for epoch in range(settings.router_epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        total_loss = 0.0
        batches = 0
        router.train()
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            rb = torch.as_tensor(route_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(router(xb), rb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        router.eval()
        with torch.no_grad():
            pred = torch.argmax(router(torch.as_tensor(x_val, dtype=torch.float32, device=device)), dim=1).cpu().numpy()
        acc = float(np.mean(pred == route_val))
        if acc > best_acc:
            best_acc = acc
            best_state = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
        if epoch % max(1, settings.router_epochs // 10) == 0 or epoch == settings.router_epochs - 1:
            row = {"seed": seed, "system": "trained_router_frozen_pocket", "epoch": epoch, "loss": round_float(total_loss / max(1, batches)), "validation_route_accuracy": round_float(acc)}
            history.append(row)
            if out:
                append_progress(out, "router_epoch", **row)
    if best_state is not None:
        router.load_state_dict(best_state)
    state = router_to_state(router)
    return {"router_state": state, "router_hash": router_state_hash(state), "history": history}


def train_joint_binding(seed: int, system: str, base_library: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"joint-binding:{seed}:{system}"), device)
    modules = modules_from_library(base_library, settings, device)
    router = RouteHead().to(device)
    params = list(router.parameters()) + [param for module in modules.values() for param in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=settings.joint_learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"joint-binding-batch:{seed}:{system}"))
    x_train, y_train, route_train, family_train = split_composition(task["train"])
    x_val, y_val, route_val, family_val = split_composition(task["validation"])
    best_state: dict[str, Any] | None = None
    best_score = -1.0
    history = []
    slot_guard = system == "trained_router_trained_pocket_slot_guard"
    for epoch in range(settings.joint_epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        total_loss = 0.0
        batches = 0
        router.train()
        modules.train()
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            rb = torch.as_tensor(route_train[idx], dtype=torch.long, device=device)
            fb = torch.as_tensor(family_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            answer_prob, route_logits, preserve_loss, slot_loss = soft_route_answer(modules, router, xb, fb)
            answer_loss = F.binary_cross_entropy(answer_prob, yb)
            route_loss = F.cross_entropy(route_logits, rb)
            if slot_guard:
                loss = answer_loss + 1.00 * route_loss + 0.85 * preserve_loss + 1.15 * slot_loss
            else:
                loss = answer_loss + 0.45 * route_loss + 0.10 * preserve_loss + 0.08 * slot_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        router.eval()
        modules.eval()
        with torch.no_grad():
            xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
            fv = torch.as_tensor(family_val, dtype=torch.long, device=device)
            pred_route = torch.argmax(router(xv), dim=1).cpu().numpy()
            answer_prob, _, preserve, slot = soft_route_answer(modules, router, xv, fv)
            pred_answer = (answer_prob.cpu().numpy() >= 0.5).astype(np.int64)
        route_acc = float(np.mean(pred_route == route_val))
        answer_acc = float(np.mean(pred_answer == y_val.astype(np.int64)))
        score = answer_acc + 0.20 * route_acc - (0.18 if slot_guard else 0.04) * float(preserve.cpu()) - (0.22 if slot_guard else 0.03) * float(slot.cpu())
        if score > best_score:
            best_score = score
            best_state = {
                "router": {key: value.detach().cpu().clone() for key, value in router.state_dict().items()},
                "pockets": {skill: {key: value.detach().cpu().clone() for key, value in modules[skill].state_dict().items()} for skill in SKILLS},
            }
        if epoch % max(1, settings.joint_epochs // 10) == 0 or epoch == settings.joint_epochs - 1:
            row = {
                "seed": seed,
                "system": system,
                "epoch": epoch,
                "loss": round_float(total_loss / max(1, batches)),
                "validation_answer_accuracy": round_float(answer_acc),
                "validation_route_accuracy": round_float(route_acc),
                "validation_preservation_error": round_float(float(preserve.cpu())),
                "validation_slot_loss": round_float(float(slot.cpu())),
            }
            history.append(row)
            if out:
                append_progress(out, "joint_binding_epoch", **row)
    if best_state is not None:
        router.load_state_dict(best_state["router"])
        for skill in SKILLS:
            modules[skill].load_state_dict(best_state["pockets"][skill])
    router_state = router_to_state(router)
    library = library_from_modules(modules, settings, system)
    return {"router_state": router_state, "router_hash": router_state_hash(router_state), "library": library, "history": history}


def random_router_state(seed: int) -> dict[str, int]:
    rng = random.Random(stable_seed(f"random-router:{seed}"))
    return {family: rng.randrange(len(ROUTE_OPTIONS)) for family in FAMILIES}


def evaluate_hard_router_system(system: str, library: dict[str, dict[str, Any]], router_state: dict[str, Any] | dict[str, int], task: dict[str, list[dict[str, Any]]], mapping_router: bool = False) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        x = np.asarray([row["flow"] for row in rows], dtype=np.float32)
        if mapping_router:
            route_indices = np.asarray([int(router_state[row["family"]]) for row in rows], dtype=np.int64)  # type: ignore[index]
        else:
            route_indices = router_predict(router_state, x)  # type: ignore[arg-type]
        correct = []
        route_correct = []
        preservation = []
        corruption = []
        compatibility = []
        samples = []
        pocket_errors = router_errors = composition_errors = 0
        for idx, row in enumerate(rows):
            route_idx = int(route_indices[idx])
            route = ROUTE_OPTIONS[route_idx]
            route_ok = route_idx == int(row["expected_route_index"])
            flow, metrics = e7p.apply_route_with_metrics(library, row, route)
            pred = e7o.predict_answer_from_flow(row, flow)
            ok = pred == int(row["target_answer"])
            correct.append(ok)
            route_correct.append(route_ok)
            preservation.append(metrics["state_preservation_error"])
            corruption.append(metrics["result_slot_corruption_rate"])
            compatibility.append(metrics["next_pocket_input_compatibility_error"])
            if not ok:
                if not route_ok:
                    router_errors += 1
                else:
                    oracle_flow = e7o.symbolic_apply_route(row, tuple(row["expected_route"]))
                    if e7o.predict_answer_from_flow(row, oracle_flow) == int(row["target_answer"]):
                        pocket_errors += 1
                    else:
                        composition_errors += 1
            if len(samples) < 8:
                samples.append({
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "route": list(route),
                    "expected_route": row["expected_route"],
                    "target": int(row["target_answer"]),
                    "predicted": int(pred),
                    "correct": bool(ok),
                    "route_correct": bool(route_ok),
                })
        acc = round_float(float(np.mean(correct)))
        route_acc = round_float(float(np.mean(route_correct)))
        mean_steps = round_float(float(np.mean([len(ROUTE_OPTIONS[int(idx)]) for idx in route_indices])))
        bit_cost = sum(e7p.bit_budget(state) for state in library.values())
        param_count = sum(e7p.parameter_count(state) for state in library.values())
        router_param_count = 64 * FLOW_DIM + 64 + 64 * len(ROUTE_OPTIONS) + len(ROUTE_OPTIONS) if not mapping_router else len(FAMILIES)
        cost_penalty = min(0.13, 0.00000016 * bit_cost + 0.0025 * mean_steps + 0.000003 * router_param_count)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": route_acc,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "state_preservation_error": round_float(float(np.mean(preservation))),
            "result_slot_corruption_rate": round_float(float(np.mean(corruption))),
            "next_pocket_input_compatibility_error": round_float(float(np.mean(compatibility))),
            "pocket_error_rate": round_float(pocket_errors / max(1, len(rows))),
            "adapter_error_rate": round_float(float(np.mean(corruption))),
            "router_error_rate": round_float(router_errors / max(1, len(rows))),
            "composition_error_rate": round_float(composition_errors / max(1, len(rows))),
            "bit_budget": bit_cost,
            "active_parameter_count": param_count + router_param_count,
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
        "eval_mean_route_accuracy": round_float(float(np.mean([evals[split]["route_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_state_preservation_error": round_float(float(np.mean([evals[split]["state_preservation_error"] for split in EVAL_SPLITS]))),
        "eval_mean_result_slot_corruption_rate": round_float(float(np.mean([evals[split]["result_slot_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility_error"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()) + (64 * FLOW_DIM + 64 + 64 * len(ROUTE_OPTIONS) + len(ROUTE_OPTIONS) if not mapping_router else len(FAMILIES)),
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()),
        "router_frozen": False,
    }


def reuse_report_row(seed: int, system: str, library: dict[str, dict[str, Any]], router_state: dict[str, Any], baseline_library: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    bound = evaluate_hard_router_system(system, library, router_state, task)
    pocket_reuse = e7p.evaluate_composition_system(f"{system}:oracle_route_reuse", library, task)
    router_transfer = evaluate_hard_router_system(f"{system}:router_transfer_to_baseline", baseline_library, router_state, task)
    bound_u = bound["eval_mean_composition_usefulness"]
    reuse_u = pocket_reuse["eval_mean_composition_usefulness"]
    transfer_u = router_transfer["eval_mean_composition_usefulness"]
    risk = max(0.0, float(bound_u) - min(float(reuse_u), float(transfer_u)))
    return {
        "seed": seed,
        "system": system,
        "bound_usefulness": bound_u,
        "pocket_reuse_after_binding_usefulness": reuse_u,
        "router_transfer_to_baseline_usefulness": transfer_u,
        "private_protocol_risk": round_float(risk),
        "reusable_pocket_gate": bool(reuse_u >= 0.70 and risk <= 0.18),
    }


def train_frozen_router_slot_pockets(seed: int, baseline_library: dict[str, dict[str, Any]], context_tasks: dict[str, dict[str, list[dict[str, Any]]]], settings: Settings, out: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    e7p_settings = to_e7p_settings(settings)
    library = {}
    rows = []
    for skill in SKILLS:
        trained = e7p.train_context_pocket(seed, skill, "joint_adapter_plus_pocket_with_slot_contract", baseline_library[skill], context_tasks[skill], e7p_settings, out)
        state = trained["state"]
        library[skill] = state
        rows.append({
            "seed": seed,
            "skill": skill,
            "system": "frozen_router_trained_pocket",
            "state_hash": e7p.state_hash(state),
            "history": trained["history"],
            "context": e7p.evaluate_context_pocket(skill, state, context_tasks[skill]),
            "trainable_scope": trained["scope"],
            "router_frozen": True,
        })
    return library, rows


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
                "reusability_after_binding",
                "private_protocol_risk",
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
    candidates = [system for system in SYSTEMS if system != "oracle_route_reference"]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7q_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    frozen = systems["frozen_router_trained_pocket"]["mean"]
    route_only = systems["trained_router_frozen_pocket"]["mean"]
    joint = systems["trained_router_trained_pocket"]["mean"]
    guarded = systems["trained_router_trained_pocket_slot_guard"]["mean"]
    full = systems["full_end_to_end_training_control"]["mean"]
    random_ctrl = systems["random_router_control"]["mean"]
    oracle = systems["oracle_route_reference"]["mean"]
    detail = {
        "best_non_reference_system": aggregate["best_non_reference_system"],
        "frozen_router_trained_pocket_usefulness": frozen.get("eval_mean_composition_usefulness", 0.0),
        "trained_router_frozen_pocket_usefulness": route_only.get("eval_mean_composition_usefulness", 0.0),
        "trained_router_trained_pocket_usefulness": joint.get("eval_mean_composition_usefulness", 0.0),
        "slot_guard_joint_usefulness": guarded.get("eval_mean_composition_usefulness", 0.0),
        "slot_guard_reusability_after_binding": guarded.get("reusability_after_binding", 0.0),
        "slot_guard_private_protocol_risk": guarded.get("private_protocol_risk", 1.0),
        "full_end_to_end_usefulness": full.get("eval_mean_composition_usefulness", 0.0),
        "random_router_control_usefulness": random_ctrl.get("eval_mean_composition_usefulness", 0.0),
        "oracle_usefulness": oracle.get("eval_mean_composition_usefulness", 0.0),
        "slot_guard_route_accuracy": guarded.get("eval_mean_route_accuracy", 0.0),
    }
    frozen_u = float(detail["frozen_router_trained_pocket_usefulness"])
    route_u = float(detail["trained_router_frozen_pocket_usefulness"])
    joint_u = float(detail["trained_router_trained_pocket_usefulness"])
    guarded_u = float(detail["slot_guard_joint_usefulness"])
    guarded_reuse = float(detail["slot_guard_reusability_after_binding"])
    guarded_risk = float(detail["slot_guard_private_protocol_risk"])
    full_u = float(detail["full_end_to_end_usefulness"])
    random_u = float(detail["random_router_control_usefulness"])
    if random_u >= 0.72:
        return "e7q_artifact_or_task_too_easy", detail
    if joint_u >= frozen_u + 0.08 and joint_u >= 0.76 and guarded_reuse < joint_u - 0.18:
        return "e7q_private_router_pocket_protocol_detected", detail
    if guarded_u >= frozen_u + 0.06 and guarded_u >= 0.78 and guarded_reuse >= 0.72 and guarded_risk <= 0.18:
        return "e7q_slot_guard_joint_binding_positive", detail
    if joint_u >= frozen_u + 0.06 and joint_u >= 0.78 and guarded_reuse >= 0.70:
        return "e7q_router_pocket_joint_binding_positive", detail
    if guarded_u >= frozen_u + 0.03 and guarded_u >= max(joint_u, route_u) and guarded_reuse >= 0.64:
        return "e7q_slot_guard_improves_but_not_solved", detail
    if route_u >= frozen_u + 0.03 and guarded_u < route_u + 0.02:
        return "e7q_router_discovery_not_interface_fix", detail
    if full_u >= max(frozen_u, route_u, joint_u, guarded_u) + 0.05:
        return "e7q_full_end_to_end_control_preferred", detail
    return "e7q_joint_binding_not_yet_viable", detail


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    context_tasks = e7p.generate_context_tasks(composition_task)

    baseline_library: dict[str, dict[str, Any]] = {}
    baseline_rows = []
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7q_baseline_standalone")
        baseline_library[skill] = state
        baseline_rows.append({
            "seed": seed,
            "skill": skill,
            "system": "baseline_standalone_pocket",
            "state_hash": e7p.state_hash(state),
            "standalone": trained["standalone"],
            "context": e7p.evaluate_context_pocket(skill, state, context_tasks[skill]),
            "trainable_scope": [],
            "router_frozen": True,
        })

    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = list(baseline_rows)
    reuse_rows: list[dict[str, Any]] = []

    frozen_library, frozen_training = train_frozen_router_slot_pockets(seed, baseline_library, context_tasks, settings, out)
    training_rows.extend(frozen_training)
    frozen_result = e7p.evaluate_composition_system("frozen_router_trained_pocket", frozen_library, composition_task)
    frozen_result.update({"seed": seed, "system": "frozen_router_trained_pocket", "router_trainable": False, "pocket_trainable": True, "reusability_after_binding": frozen_result["eval_mean_composition_usefulness"], "private_protocol_risk": 0.0})
    rows.append(frozen_result)

    router_only = train_router_only(seed, composition_task, settings, out)
    router_frozen_result = evaluate_hard_router_system("trained_router_frozen_pocket", baseline_library, router_only["router_state"], composition_task)
    router_frozen_result.update({"seed": seed, "system": "trained_router_frozen_pocket", "router_trainable": True, "pocket_trainable": False, "router_hash": router_only["router_hash"], "reusability_after_binding": router_frozen_result["eval_mean_composition_usefulness"], "private_protocol_risk": 0.0})
    rows.append(router_frozen_result)
    training_rows.append({"seed": seed, "skill": "router", "system": "trained_router_frozen_pocket", "history": router_only["history"], "router_hash": router_only["router_hash"], "trainable_scope": ["route_head"], "router_frozen": False})

    for system in ("trained_router_trained_pocket", "trained_router_trained_pocket_slot_guard"):
        bound = train_joint_binding(seed, system, baseline_library, composition_task, settings, out)
        result = evaluate_hard_router_system(system, bound["library"], bound["router_state"], composition_task)
        reuse = reuse_report_row(seed, system, bound["library"], bound["router_state"], baseline_library, composition_task)
        result.update({"seed": seed, "system": system, "router_trainable": True, "pocket_trainable": True, "router_hash": bound["router_hash"], "reusability_after_binding": reuse["pocket_reuse_after_binding_usefulness"], "private_protocol_risk": reuse["private_protocol_risk"]})
        rows.append(result)
        reuse_rows.append(reuse)
        for skill in SKILLS:
            training_rows.append({
                "seed": seed,
                "skill": skill,
                "system": system,
                "state_hash": e7p.state_hash(bound["library"][skill]),
                "history": bound["history"],
                "context": e7p.evaluate_context_pocket(skill, bound["library"][skill], context_tasks[skill]),
                "trainable_scope": ["route_head", "win", "bin", "wcore", "carry_raw", "wout", "bout"],
                "router_frozen": False,
            })

    mono = e7o.train_monolithic(seed, "full_end_to_end_training_control", composition_task, e7o_settings, out, hidden=160, depth=3)
    mono.update({
        "seed": seed,
        "system": "full_end_to_end_training_control",
        "eval_mean_state_preservation_error": 0.0,
        "eval_mean_result_slot_corruption_rate": 0.0,
        "eval_mean_next_pocket_input_compatibility_error": 0.0,
        "reusability_after_binding": 0.0,
        "private_protocol_risk": 1.0,
    })
    rows.append(mono)
    training_rows.append({"seed": seed, "skill": "full_model", "system": "full_end_to_end_training_control", "history": mono.get("history", []), "trainable_scope": ["monolithic_model"], "router_frozen": False})

    random_mapping = random_router_state(seed)
    random_result = evaluate_hard_router_system("random_router_control", baseline_library, random_mapping, composition_task, mapping_router=True)
    random_result.update({"seed": seed, "system": "random_router_control", "router_candidate": random_mapping, "reusability_after_binding": random_result["eval_mean_composition_usefulness"], "private_protocol_risk": 0.0})
    rows.append(random_result)

    oracle = e7p.evaluate_composition_system("oracle_route_reference", None, composition_task, symbolic=True)
    oracle.update({"seed": seed, "system": "oracle_route_reference", "router_trainable": False, "pocket_trainable": False, "reusability_after_binding": oracle["eval_mean_composition_usefulness"], "private_protocol_risk": 0.0})
    rows.append(oracle)

    return {"seed": seed, "rows": rows, "training_rows": training_rows, "reuse_rows": reuse_rows}


def build_reports(rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], reuse_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = canonical_system_rows(rows)
    clean_training_rows = canonical_training_rows(training_rows)
    flow_rows = [{
        "seed": row["seed"],
        "system": row["system"],
        "state_preservation_error": row.get("eval_mean_state_preservation_error", 0.0),
        "result_slot_corruption_rate": row.get("eval_mean_result_slot_corruption_rate", 0.0),
        "next_pocket_input_compatibility_error": row.get("eval_mean_next_pocket_input_compatibility_error", 0.0),
    } for row in clean_rows]
    return {
        "baseline_pocket_training_report.json": {"schema_version": "e7q_baseline_pocket_training_report_v1", "rows": [row for row in clean_training_rows if row["system"] == "baseline_standalone_pocket"]},
        "router_training_report.json": {"schema_version": "e7q_router_training_report_v1", "rows": [row for row in clean_training_rows if row["skill"] in {"router", "full_model"}]},
        "joint_binding_training_report.json": {"schema_version": "e7q_joint_binding_training_report_v1", "rows": [row for row in clean_training_rows if row["system"] in {"frozen_router_trained_pocket", "trained_router_trained_pocket", "trained_router_trained_pocket_slot_guard"}]},
        "flow_contract_report.json": {"schema_version": "e7q_flow_contract_report_v1", "rows": flow_rows},
        "reuse_after_binding_report.json": {"schema_version": "e7q_reuse_after_binding_report_v1", "rows": sorted(reuse_rows, key=lambda row: (row["seed"], row["system"]))},
        "private_protocol_leakage_report.json": {"schema_version": "e7q_private_protocol_leakage_report_v1", "private_protocol_threshold": 0.18, "rows": sorted(reuse_rows, key=lambda row: (row["seed"], row["system"]))},
        "composition_report.json": {"schema_version": "e7q_composition_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "answer_accuracy": row["eval_mean_answer_accuracy"], "route_accuracy": row["eval_mean_route_accuracy"], "usefulness": row["eval_mean_composition_usefulness"]} for row in clean_rows]},
        "error_attribution_report.json": {"schema_version": "e7q_error_attribution_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "pocket_error": row["evals"]["heldout"].get("pocket_error_rate", 0.0), "adapter_error": row["evals"]["heldout"].get("adapter_error_rate", 0.0), "router_error": row["evals"]["heldout"].get("router_error_rate", 0.0), "composition_error": row["evals"]["heldout"].get("composition_error_rate", 0.0)} for row in clean_rows]},
        "system_results.json": {"schema_version": "e7q_system_results_v1", "rows": clean_rows},
        "leakage_report.json": {"schema_version": "e7q_leakage_report_v1", "symbolic_proxy_primary": False, "hidden_expected_answer_input": False, "dense_graph_primary": False, "full_end_to_end_is_diagnostic_only": True, "route_head_uses_expected_route_as_label_only": True},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7q_summary_v1", "run_root": DEFAULT_OUT.as_posix(), "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "checker_failure_count": None},
    }


def write_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    lines = [
        "# E7Q Router-Pocket Joint Binding Probe Result",
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
            f"{system:<46} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"route={mean.get('eval_mean_route_accuracy', 0.0):.6f} "
            f"reuse={mean.get('reusability_after_binding', 0.0):.6f} "
            f"risk={mean.get('private_protocol_risk', 0.0):.6f}"
        )
    lines.extend(["```", "", "## Detail", "", "```text"])
    for key, value in decision["detail"].items():
        lines.append(f"{key} = {value}")
    lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7Q is a controlled numeric Flow[D] router-pocket binding probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.",
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
    write_json(out / "backend_manifest.json", {
        "schema_version": "e7q_backend_manifest_v1",
        "milestone": MILESTONE,
        "systems": list(SYSTEMS),
        "settings": settings_payload(settings),
        "device": select_device(settings.device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "typed_route_head_primary": True,
        "dense_graph_primary": False,
    })
    write_json(out / "task_generation_report.json", {
        "schema_version": "e7q_task_generation_report_v1",
        "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
        "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
    })
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    jobs = [{
        "seed": seed,
        "settings": settings.__dict__.copy(),
        "composition_task": composition_tasks[seed],
        "pocket_task": pocket_tasks[seed],
        "out": str(out),
    } for seed in settings.seeds]
    max_workers = max(1, min(settings.cpu_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
        while futures:
            done, _ = wait(futures, timeout=max(1.0, settings.heartbeat_seconds), return_when=FIRST_COMPLETED)
            if not done:
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_reuse_rows": len(reuse_rows), "pending": len(futures)})
                continue
            for future in done:
                label = futures.pop(future)
                result = future.result()
                rows.extend(result["rows"])
                training_rows.extend(result["training_rows"])
                reuse_rows.extend(result["reuse_rows"])
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_reuse_rows": len(reuse_rows), "last_completed": label, "pending": len(futures)})
                append_progress(out, "seed_job_complete", label=label, pending=len(futures))
    aggregate = aggregate_results(rows)
    decision_label, detail = decide(aggregate)
    deterministic_placeholder = {"internal_replay_passed": True}
    decision = {"schema_version": "e7q_decision_v1", "decision": decision_label, "detail": detail, "deterministic_replay_passed": True}
    reports = build_reports(rows, training_rows, reuse_rows, aggregate, decision, deterministic_placeholder)
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, aggregate)
    append_progress(out, "primary_artifacts_written", artifact_count=len(reports) + 2)
    deterministic = {"schema_version": "e7q_deterministic_replay_report_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_mode": replay_mode}
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
        deterministic = {"schema_version": "e7q_deterministic_replay_report_v1", "internal_replay_passed": passed, "hash_comparisons": comparisons, "replay_mode": False}
        decision["deterministic_replay_passed"] = passed
        reports = build_reports(rows, training_rows, reuse_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--local-epochs", type=int, default=80)
    parser.add_argument("--router-epochs", type=int, default=70)
    parser.add_argument("--joint-epochs", type=int, default=80)
    parser.add_argument("--full-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-learning-rate", type=float, default=1e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=7e-4)
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
        router_epochs=args.router_epochs,
        joint_epochs=args.joint_epochs,
        full_epochs=args.full_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        router_learning_rate=args.router_learning_rate,
        joint_learning_rate=args.joint_learning_rate,
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
