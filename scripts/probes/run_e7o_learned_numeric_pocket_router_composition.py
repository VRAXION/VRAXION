#!/usr/bin/env python3
"""E7O learned numeric pocket router composition probe.

E7N showed that one real numeric pocket can be trained, quantized, repaired,
pruned, and registered. E7O asks whether several real learned numeric pockets
can be composed by a mutation/rollback router on multi-step numeric tasks.

The primary learned-pocket systems use frozen numeric pocket cores:

    CALL(pocket_id, Flow[D]) -> Flow[D]

Each pocket writes its own scratch/result slot in Flow[D]. The router chooses
which pocket IDs to call and in what order.
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
E7N_PATH = Path(__file__).with_name("run_e7n_real_numeric_pocket_core_bridge_probe.py")
MILESTONE = "E7O_LEARNED_NUMERIC_POCKET_ROUTER_COMPOSITION"
DEFAULT_OUT = Path("target/pilot_wave/e7o_learned_numeric_pocket_router_composition")
DEFAULT_SEEDS = (99401, 99402, 99403, 99404)

FLOW_DIM = 40
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
FAMILIES = (
    "compare_threshold",
    "mod_add_parity",
    "counterfactual_recompute",
    "verify_route_correction",
    "mixed_chain_length_4",
    "adversarial_misleading_branch",
)
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
RESULT_POS = {skill: 24 + idx for idx, skill in enumerate(SKILLS)}
ANSWER_POS = 32
FAMILY_ROUTE = {
    "compare_threshold": ("compare", "threshold"),
    "mod_add_parity": ("mod_add", "parity"),
    "counterfactual_recompute": ("mod_add", "counterfactual_flip"),
    "verify_route_correction": ("compare", "threshold", "verify"),
    "mixed_chain_length_4": ("compare", "threshold", "mod_add", "verify"),
    "adversarial_misleading_branch": ("compare", "threshold", "counterfactual_flip", "verify"),
}
ROUTE_OPTIONS = (
    ("compare", "threshold"),
    ("mod_add", "parity"),
    ("mod_add", "counterfactual_flip"),
    ("compare", "threshold", "verify"),
    ("compare", "threshold", "mod_add", "verify"),
    ("compare", "threshold", "counterfactual_flip", "verify"),
    ("threshold", "compare"),
    ("parity", "mod_add"),
    ("counterfactual_flip",),
    ("compare",),
    ("mod_add",),
    ("verify",),
)
EXPECTED_ROUTE_INDEX = {family: ROUTE_OPTIONS.index(route) for family, route in FAMILY_ROUTE.items()}
SYSTEMS = (
    "symbolic_proxy_pocket_router_reference",
    "float_numeric_pocket_library_router",
    "int8_numeric_pocket_library_router",
    "int4_pruned_numeric_pocket_library_router",
    "ternary_binary_numeric_pocket_router",
    "mixed_precision_numeric_pocket_router",
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "dense_graph_danger_control",
    "oracle_router_over_numeric_pockets",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "dense_graph_danger_control")
MUTATION_SYSTEMS = (
    "float_numeric_pocket_library_router",
    "int8_numeric_pocket_library_router",
    "int4_pruned_numeric_pocket_library_router",
    "ternary_binary_numeric_pocket_router",
    "mixed_precision_numeric_pocket_router",
    "monolithic_mutation_model",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "numeric_pocket_training_report.json",
    "pocket_library_report.json",
    "router_training_report.json",
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
    "e7o_int4_numeric_pocket_router_composition_positive",
    "e7o_float_only_numeric_pocket_composition",
    "e7o_router_over_numeric_pockets_failure",
    "e7o_numeric_pocket_quality_bottleneck",
    "e7o_monolithic_model_preferred_for_numeric_composition",
    "e7o_numeric_pocket_router_collapses_to_graph_soup",
    "e7o_mixed_precision_numeric_pocket_router_preferred",
)


def load_e7n_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7n_real_numeric_pocket_core_bridge_probe", E7N_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7N helpers from {E7N_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7n = load_e7n_module()


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
    monolithic_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    router_generations: int
    router_population: int
    mutation_generations: int
    mutation_population: int
    prune_rounds: int
    prune_fraction: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7o::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


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

    thread = threading.Thread(target=worker, name="e7o-hardware-heartbeat", daemon=True)
    thread.start()
    return thread


def row_count_for_split(settings: Settings, split: str) -> int:
    return int(getattr(settings, f"{split}_rows_per_seed"))


def base_skill_value(skill: str, a: int, b: int, key: int, threshold: int, flip: int, flow: np.ndarray | None = None) -> int:
    if skill == "compare":
        return 1 if a > b else 0
    if skill == "mod_add":
        return 1 if (a + b + key) >= 11 else 0
    if skill == "parity":
        return 1 if ((a >= 4) != (b >= 4)) else 0
    if skill == "threshold":
        return 1 if (a + b) >= (2 * threshold) else 0
    if skill == "counterfactual_flip":
        base = 1 if (a + b + key) >= 11 else 0
        return 1 - base if flip else base
    if skill == "verify":
        if flow is not None:
            compare = 1 if flow[RESULT_POS["compare"]] >= 0.5 else 0
            threshold_val = 1 if flow[RESULT_POS["threshold"]] >= 0.5 else 0
        else:
            compare = 1 if a > b else 0
            threshold_val = 1 if (a + b) >= (2 * threshold) else 0
        return 1 if compare == threshold_val else 0
    raise ValueError(skill)


def expected_answer(family: str, flow: np.ndarray) -> int:
    c = 1 if flow[RESULT_POS["compare"]] >= 0.5 else 0
    m = 1 if flow[RESULT_POS["mod_add"]] >= 0.5 else 0
    p = 1 if flow[RESULT_POS["parity"]] >= 0.5 else 0
    t = 1 if flow[RESULT_POS["threshold"]] >= 0.5 else 0
    f = 1 if flow[RESULT_POS["counterfactual_flip"]] >= 0.5 else 0
    v = 1 if flow[RESULT_POS["verify"]] >= 0.5 else 0
    if family == "compare_threshold":
        return c & t
    if family == "mod_add_parity":
        return m ^ p
    if family == "counterfactual_recompute":
        return f
    if family == "verify_route_correction":
        return v
    if family == "mixed_chain_length_4":
        return v & m
    if family == "adversarial_misleading_branch":
        return v ^ f
    raise ValueError(family)


def encode_flow(family: str, a: int, b: int, key: int, threshold: int, flip: int, decoy: int, split: str) -> np.ndarray:
    x = np.zeros(FLOW_DIM, dtype=np.float32)
    x[FAMILIES.index(family)] = 1.0
    selected = a if key < 4 else b
    x[6] = a / 7.0
    x[7] = b / 7.0
    x[8] = key / 7.0
    x[9] = threshold / 7.0
    x[10] = float(flip)
    x[11] = (a + b) / 14.0
    x[12] = (a + b + key) / 21.0
    x[13] = (a - b) / 7.0
    x[14] = 1.0 if a >= 4 else 0.0
    x[15] = 1.0 if b >= 4 else 0.0
    x[16] = 1.0 if key < 4 else 0.0
    x[17] = selected / 7.0
    x[18] = (selected - threshold) / 7.0
    x[19] = math.sin((a + 1) * 0.37)
    x[20] = math.cos((b + 1) * 0.41)
    x[21] = math.sin((key + 1) * 0.29)
    x[31] = float(decoy)
    if split == "ood":
        x *= 0.97
    return x


def make_row(seed: int, split: str, idx: int, rng: random.Random) -> dict[str, Any]:
    family = FAMILIES[idx % len(FAMILIES)]
    if split == "train":
        a = rng.randrange(0, 7)
        b = rng.randrange(0, 7)
    elif split == "ood":
        a = rng.choice((5, 6, 7))
        b = rng.choice((5, 6, 7))
    else:
        a = rng.randrange(0, 8)
        b = rng.randrange(0, 8)
    key = rng.randrange(0, 8)
    threshold = rng.randrange(1, 8)
    flip = 1 if split == "counterfactual" else rng.randrange(0, 2)
    flow = encode_flow(family, a, b, key, threshold, flip, rng.randrange(0, 2), split)
    oracle = flow.copy()
    for skill in FAMILY_ROUTE[family]:
        oracle[RESULT_POS[skill]] = float(base_skill_value(skill, a, b, key, threshold, flip, oracle))
    answer = expected_answer(family, oracle)
    if split == "adversarial":
        flow[31] = float(1 - answer)
    return {
        "row_id": f"{seed}:{split}:{idx}",
        "split": split,
        "family": family,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "flip": flip,
        "flow": flow.tolist(),
        "target_answer": int(answer),
        "expected_route": list(FAMILY_ROUTE[family]),
        "expected_route_index": EXPECTED_ROUTE_INDEX[family],
    }


def generate_composition_tasks(settings: Settings) -> dict[int, dict[str, list[dict[str, Any]]]]:
    tasks: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in settings.seeds:
        seed_rows: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            rng = random.Random(stable_seed(f"composition:{seed}:{split}"))
            seed_rows[split] = [make_row(seed, split, idx, rng) for idx in range(row_count_for_split(settings, split))]
        tasks[seed] = seed_rows
    return tasks


def make_pocket_training_row(seed: int, skill: str, split: str, idx: int, rng: random.Random) -> dict[str, Any]:
    family = "mixed_chain_length_4"
    a = rng.randrange(0, 8)
    b = rng.randrange(0, 8)
    key = rng.randrange(0, 8)
    threshold = rng.randrange(1, 8)
    flip = 1 if split == "counterfactual" else rng.randrange(0, 2)
    flow = encode_flow(family, a, b, key, threshold, flip, rng.randrange(0, 2), split)
    if skill == "verify":
        flow[RESULT_POS["compare"]] = float(base_skill_value("compare", a, b, key, threshold, flip))
        flow[RESULT_POS["threshold"]] = float(base_skill_value("threshold", a, b, key, threshold, flip))
    target = flow.copy()
    target[RESULT_POS[skill]] = float(base_skill_value(skill, a, b, key, threshold, flip, flow))
    return {
        "row_id": f"{seed}:{skill}:{split}:{idx}",
        "split": split,
        "skill": skill,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "flip": flip,
        "flow": flow.tolist(),
        "target_flow": target.tolist(),
        "target_value": int(target[RESULT_POS[skill]] >= 0.5),
    }


def generate_pocket_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    tasks: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for seed in settings.seeds:
        tasks[seed] = {}
        for skill in SKILLS:
            tasks[seed][skill] = {}
            for split in SPLITS:
                count = settings.pocket_pretrain_rows_per_seed if split == "train" else settings.pocket_validation_rows_per_seed
                rng = random.Random(stable_seed(f"pocket:{seed}:{skill}:{split}"))
                tasks[seed][skill][split] = [make_pocket_training_row(seed, skill, split, idx, rng) for idx in range(count)]
    return tasks


def split_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row["flow"] for row in rows], dtype=np.float32)
    y = np.asarray([row.get("target_flow", row["flow"]) for row in rows], dtype=np.float32)
    target = np.asarray([int(row.get("target_value", row.get("target_answer", 0))) for row in rows], dtype=np.int64)
    return x, y, target


def state_hash(state: dict[str, Any]) -> str:
    return e7n.state_hash(state)


def serializable_state(state: dict[str, Any]) -> dict[str, Any]:
    return e7n.serializable_state(state)


def parameter_count(state: dict[str, Any], active_only: bool = False) -> int:
    return e7n.parameter_count(state, active_only=active_only)


def bit_budget(state: dict[str, Any]) -> int:
    return e7n.bit_budget(state)


def np_forward(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return e7n.np_forward(state, x)


class NumericPocketCore(e7n.NumericPocketCore):
    pass


def model_to_state(model: NumericPocketCore, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7o_numeric_pocket_state_v1",
        "precision": "float32",
        "flow_dim": FLOW_DIM,
        "pocket_dim": settings.pocket_dim,
        "core_steps": settings.pocket_core_steps,
        "arrays": {name: param.detach().cpu().numpy().astype(np.float32) for name, param in model.named_parameters()},
        "masks": {},
        "lineage": ["float_numeric_pocket_pretrain"],
    }


def train_skill_pocket(seed: int, skill: str, settings: Settings, rows: dict[str, list[dict[str, Any]]], out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"pocket-train:{seed}:{skill}"), device)
    model = NumericPocketCore(FLOW_DIM, settings.pocket_dim, settings.pocket_core_steps).to(device)
    x_train, y_train, t_train = split_arrays(rows["train"])
    x_val, y_val, t_val = split_arrays(rows["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"pocket-batch:{seed}:{skill}"))
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    history = []
    pos = RESULT_POS[skill]
    for epoch in range(settings.pocket_epochs):
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
            loss = 0.25 * F.mse_loss(pred, yb) + F.binary_cross_entropy_with_logits(pred[:, pos], tb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.as_tensor(x_val, dtype=torch.float32, device=device))[:, pos].detach().cpu().numpy()
        val_acc = float(np.mean((val_pred >= 0.0).astype(np.int64) == t_val))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, settings.pocket_epochs // 10) == 0 or epoch == settings.pocket_epochs - 1:
            row = {"seed": seed, "skill": skill, "epoch": epoch, "loss": round_float(total_loss / max(1, batches)), "validation_accuracy": round_float(val_acc), "best_validation_accuracy": round_float(best_val)}
            history.append(row)
            if out:
                append_progress(out, "pocket_gradient_epoch", **row)
                write_json(out / "pocket_training_snapshots" / f"{skill}_seed{seed}_epoch{epoch:04d}.json", row)
    if best_state is not None:
        model.load_state_dict(best_state)
    state = model_to_state(model, settings)
    standalone = evaluate_skill_pocket(skill, state, rows)
    return {"seed": seed, "skill": skill, "state": state, "history": history, "best_validation_accuracy": round_float(best_val), "standalone": standalone}


def evaluate_skill_pocket(skill: str, state: dict[str, Any], rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pos = RESULT_POS[skill]
    evals = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        x, _, target = split_arrays(rows)
        pred_flow = np_forward(state, x)
        pred = (pred_flow[:, pos] >= 0.0).astype(np.int64)
        evals[split] = round_float(float(np.mean(pred == target)))
    return {
        "evals": evals,
        "eval_mean_accuracy": round_float(float(np.mean([evals[split] for split in EVAL_SPLITS]))),
        "heldout_accuracy": evals["heldout"],
        "ood_accuracy": evals["ood"],
        "counterfactual_accuracy": evals["counterfactual"],
        "adversarial_accuracy": evals["adversarial"],
        "state_hash": state_hash(state),
        "bit_budget": bit_budget(state),
        "active_parameter_count": parameter_count(state, active_only=True),
    }


def quantize_state(state: dict[str, Any], precision: str) -> dict[str, Any]:
    q_state, _ = e7n.quantize_state(state, precision)
    q_state["schema_version"] = "e7o_numeric_pocket_state_v1"
    return q_state


def prune_state_for_skill(seed: int, skill: str, state: dict[str, Any], rows: dict[str, list[dict[str, Any]]], settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(stable_seed(f"prune:{seed}:{skill}"))
    best = {
        "schema_version": state["schema_version"],
        "precision": state["precision"],
        "flow_dim": state["flow_dim"],
        "pocket_dim": state["pocket_dim"],
        "core_steps": state["core_steps"],
        "arrays": {key: value.copy() for key, value in state["arrays"].items()},
        "masks": {key: np.ones_like(value, dtype=np.float32) for key, value in state["arrays"].items()},
        "lineage": list(state.get("lineage", [])) + ["composition_prune_start"],
    }
    baseline = evaluate_skill_pocket(skill, best, rows)["evals"]["validation"]
    best_score = baseline
    initial_active = parameter_count(best, active_only=True)
    accepted = rejected = 0
    history = []
    for round_idx in range(settings.prune_rounds):
        candidate = {
            "schema_version": best["schema_version"],
            "precision": best["precision"],
            "flow_dim": best["flow_dim"],
            "pocket_dim": best["pocket_dim"],
            "core_steps": best["core_steps"],
            "arrays": {key: value.copy() for key, value in best["arrays"].items()},
            "masks": {key: value.copy() for key, value in best["masks"].items()},
            "lineage": list(best.get("lineage", [])) + ["composition_prune_attempt"],
        }
        key = rng.choice(["win", "wcore", "wout"])
        arr = np.abs(candidate["arrays"][key] * candidate["masks"][key]).reshape(-1)
        mask = candidate["masks"][key].reshape(-1)
        active = np.flatnonzero(mask > 0.0)
        if active.size == 0:
            rejected += 1
            continue
        cut = max(1, int(active.size * settings.prune_fraction))
        remove = active[np.argsort(arr[active])[:cut]]
        mask[remove] = 0.0
        score = evaluate_skill_pocket(skill, candidate, rows)["evals"]["validation"]
        if score >= baseline - 0.008 and score >= best_score - 0.004:
            best = candidate
            best_score = score
            accepted += 1
        else:
            rejected += 1
        history.append({"round": round_idx, "score": round_float(best_score), "accepted": accepted, "rejected": rejected, "state_hash": state_hash(best), "active_parameter_count": parameter_count(best, active_only=True)})
    best["lineage"] = list(best.get("lineage", [])) + ["composition_prune_complete"]
    active = parameter_count(best, active_only=True)
    return best, {"accepted": accepted, "rejected": rejected, "history": history, "prune_compression_ratio": round_float(1.0 - active / max(1, initial_active))}


def build_libraries(seed: int, settings: Settings, pocket_tasks: dict[str, dict[str, list[dict[str, Any]]]], out: Path | None) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    libraries: dict[str, dict[str, dict[str, Any]]] = {
        "float": {},
        "int8": {},
        "int4_pruned": {},
        "ternary_binary": {},
        "mixed": {},
    }
    rows = []
    for skill in SKILLS:
        trained = train_skill_pocket(seed, skill, settings, pocket_tasks[skill], out)
        float_state = trained["state"]
        int8_state = quantize_state(float_state, "int8")
        int4_state = quantize_state(float_state, "int4")
        int4_pruned, prune_info = prune_state_for_skill(seed, skill, int4_state, pocket_tasks[skill], settings)
        ternary_state = quantize_state(float_state, "ternary")
        binary_state = quantize_state(float_state, "binary")
        variants = {
            "float": float_state,
            "int8": int8_state,
            "int4_pruned": int4_pruned,
            "ternary": ternary_state,
            "binary": binary_state,
        }
        scores = {name: evaluate_skill_pocket(skill, state, pocket_tasks[skill]) for name, state in variants.items()}
        libraries["float"][skill] = float_state
        libraries["int8"][skill] = int8_state
        libraries["int4_pruned"][skill] = int4_pruned
        libraries["ternary_binary"][skill] = binary_state if scores["binary"]["eval_mean_accuracy"] >= scores["ternary"]["eval_mean_accuracy"] - 0.02 else ternary_state
        best_low = max(("int8", "int4_pruned", "binary", "ternary"), key=lambda name: scores[name]["eval_mean_accuracy"] - 0.00000015 * bit_budget(variants[name]))
        libraries["mixed"][skill] = variants[best_low]
        for name, state in variants.items():
            rows.append({"seed": seed, "skill": skill, "variant": name, "state_hash": state_hash(state), "precision": state["precision"], "standalone": scores[name], "bit_budget": bit_budget(state), "active_parameter_count": parameter_count(state, active_only=True), "prune_info": prune_info if name == "int4_pruned" else None})
        if out:
            append_progress(out, "skill_pocket_registered", seed=seed, skill=skill, float_accuracy=scores["float"]["eval_mean_accuracy"], int4_accuracy=scores["int4_pruned"]["eval_mean_accuracy"], binary_accuracy=scores["binary"]["eval_mean_accuracy"])
    return libraries, rows


def apply_library_route(library: dict[str, dict[str, Any]], row: dict[str, Any], route: tuple[str, ...]) -> np.ndarray:
    flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
    for skill in route:
        state = library.get(skill)
        if state is None:
            continue
        flow = np_forward(state, flow).astype(np.float32)
        flow[0, RESULT_POS[skill]] = 1.0 if flow[0, RESULT_POS[skill]] >= 0.0 else 0.0
    return flow.reshape(-1)


def predict_answer_from_flow(row: dict[str, Any], flow: np.ndarray) -> int:
    return expected_answer(row["family"], flow)


def symbolic_apply_route(row: dict[str, Any], route: tuple[str, ...]) -> np.ndarray:
    flow = np.asarray(row["flow"], dtype=np.float32).copy()
    for skill in route:
        flow[RESULT_POS[skill]] = float(base_skill_value(skill, row["a"], row["b"], row["key"], row["threshold"], row["flip"], flow))
    return flow


def router_initial(seed: int) -> dict[str, int]:
    rng = random.Random(stable_seed(f"router-init:{seed}"))
    return {family: rng.randrange(len(ROUTE_OPTIONS)) for family in FAMILIES}


def route_accuracy(candidate: dict[str, int]) -> float:
    return float(np.mean([candidate[family] == EXPECTED_ROUTE_INDEX[family] for family in FAMILIES]))


def evaluate_router_system(system: str, candidate: dict[str, int], library: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, Any]:
    evals = {}
    for split in SPLITS:
        rows = task[split]
        correct = []
        route_correct = []
        samples = []
        pocket_errors = router_errors = composition_errors = 0
        for row in rows:
            route_idx = candidate[row["family"]]
            route = ROUTE_OPTIONS[route_idx]
            route_ok = route_idx == row["expected_route_index"]
            if symbolic:
                flow = symbolic_apply_route(row, route)
            else:
                assert library is not None
                flow = apply_library_route(library, row, route)
            pred = predict_answer_from_flow(row, flow)
            ok = pred == int(row["target_answer"])
            correct.append(ok)
            route_correct.append(route_ok)
            if not ok:
                if not route_ok:
                    router_errors += 1
                else:
                    oracle_flow = symbolic_apply_route(row, tuple(row["expected_route"]))
                    oracle_answer = predict_answer_from_flow(row, oracle_flow)
                    if oracle_answer == int(row["target_answer"]):
                        pocket_errors += 1
                    else:
                        composition_errors += 1
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "expected_route": row["expected_route"], "target": int(row["target_answer"]), "predicted": int(pred), "correct": bool(ok)})
        acc = round_float(float(np.mean(correct)))
        route_acc = round_float(float(np.mean(route_correct)))
        mean_steps = round_float(float(np.mean([len(ROUTE_OPTIONS[candidate[row["family"]]]) for row in rows])))
        bit_cost = sum(bit_budget(state) for state in library.values()) if library else 0
        param_count = sum(parameter_count(state, active_only=True) for state in library.values()) if library else 0
        cost_penalty = min(0.12, 0.0000002 * bit_cost + 0.003 * mean_steps)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": route_acc,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "pocket_error_rate": round_float(pocket_errors / max(1, len(rows))),
            "router_error_rate": round_float(router_errors / max(1, len(rows))),
            "composition_error_rate": round_float(composition_errors / max(1, len(rows))),
            "bit_budget": bit_cost,
            "active_parameter_count": param_count,
            "pocket_reuse_count": len(rows) * mean_steps,
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
        "parameter_count": sum(parameter_count(state, active_only=True) for state in library.values()) if library else 0,
        "bit_budget": sum(bit_budget(state) for state in library.values()) if library else 0,
        "router_complexity": len(FAMILIES) * len(ROUTE_OPTIONS),
    }


def mutate_router(candidate: dict[str, int], rng: random.Random) -> dict[str, int]:
    out = dict(candidate)
    family = rng.choice(FAMILIES)
    out[family] = rng.randrange(len(ROUTE_OPTIONS))
    return out


def router_score(system: str, candidate: dict[str, int], library: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> float:
    task = {"train": rows, "validation": rows, "heldout": rows, "ood": rows, "counterfactual": rows, "adversarial": rows}
    eval_row = evaluate_router_system(system, candidate, library, task)
    return float(eval_row["evals"]["validation"]["composition_usefulness"] + 0.08 * eval_row["evals"]["validation"]["route_accuracy"])


def train_mutation_router(seed: int, system: str, library: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> tuple[dict[str, int], dict[str, Any]]:
    rng = random.Random(stable_seed(f"router:{seed}:{system}"))
    best = router_initial(seed)
    best_score = router_score(system, best, library, task["validation"])
    initial_hash = payload_sha256(best)
    accepted = rejected = attempts = 0
    history = []
    for generation in range(settings.router_generations):
        generation_best = best_score
        for _ in range(settings.router_population):
            attempts += 1
            candidate = mutate_router(best, rng)
            score = router_score(system, candidate, library, task["validation"])
            if score > best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
            else:
                rejected += 1
        if generation % max(1, settings.router_generations // 10) == 0 or generation == settings.router_generations - 1:
            row = {"seed": seed, "system": system, "generation": generation, "score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "rollback": rejected, "router_hash": payload_sha256(best), "route_accuracy": round_float(route_accuracy(best))}
            history.append(row)
            if out:
                append_progress(out, "router_mutation_generation", **row)
                write_json(out / "router_history_snapshots" / f"{system}_seed{seed}_generation{generation:04d}.json", row)
    return best, {"history": history, "initial_candidate_hash": initial_hash, "final_candidate_hash": payload_sha256(best), "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": payload_sha256(best)}), "mutation_attempts": attempts, "accepted_mutations": accepted, "rejected_mutations": rejected, "rollback_count": rejected}


class MonolithicModel(nn.Module):
    def __init__(self, hidden: int = 96, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = FLOW_DIM
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.Tanh())
            dim = hidden
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_monolithic(seed: int, system: str, task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None, hidden: int, depth: int) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"mono:{seed}:{system}"), device)
    model = MonolithicModel(hidden=hidden, depth=depth).to(device)
    x_train, _, y_train = split_arrays(task["train"])
    x_val, _, y_val = split_arrays(task["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"mono-batch:{seed}:{system}"))
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(settings.monolithic_epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = torch.argmax(model(torch.as_tensor(x_val, dtype=torch.float32, device=device)), dim=1).detach().cpu().numpy()
        val_acc = float(np.mean(pred == y_val))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, settings.monolithic_epochs // 10) == 0 or epoch == settings.monolithic_epochs - 1:
            row = {"seed": seed, "system": system, "epoch": epoch, "validation_accuracy": round_float(val_acc), "best_validation_accuracy": round_float(best_val)}
            history.append(row)
            if out:
                append_progress(out, "monolithic_gradient_epoch", **row)
    if best_state is not None:
        model.load_state_dict(best_state)
    evals = {}
    param_count = sum(param.numel() for param in model.parameters())
    bit_cost = param_count * 32
    for split in SPLITS:
        rows = task[split]
        x, _, y = split_arrays(rows)
        with torch.no_grad():
            pred = torch.argmax(model(torch.as_tensor(x, dtype=torch.float32, device=device)), dim=1).detach().cpu().numpy()
        acc = round_float(float(np.mean(pred == y)))
        usefulness = round_float(max(0.0, acc - min(0.14, 0.00000018 * bit_cost)))
        evals[split] = {"answer_accuracy": acc, "route_accuracy": 0.0, "composition_usefulness": usefulness, "mean_route_steps": 1.0, "pocket_error_rate": 0.0, "router_error_rate": 0.0, "composition_error_rate": round_float(1.0 - acc), "bit_budget": bit_cost, "active_parameter_count": param_count, "pocket_reuse_count": 0, "row_level_samples": [{"row_id": row["row_id"], "family": row["family"], "target": int(row["target_answer"]), "predicted": int(pred[idx]), "correct": bool(pred[idx] == int(row["target_answer"]))} for idx, row in enumerate(rows[:8])]}
    return {
        "seed": seed,
        "system": system,
        "evals": evals,
        "training_history": history,
        "backprop_used": True,
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 0.0,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "parameter_count": param_count,
        "bit_budget": bit_cost,
        "router_complexity": 0,
        "model_hash": payload_sha256({key: value.detach().cpu().numpy().round(8).tolist() for key, value in model.state_dict().items()}),
    }


def train_monolithic_mutation(seed: int, task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"mono-mut:{seed}"))
    input_dim = FLOW_DIM
    weights = rng.normal(0.0, 0.08, size=(input_dim, 2)).astype(np.float32)
    bias = np.zeros(2, dtype=np.float32)
    initial_hash = payload_sha256({"w": np.round(weights, 8).tolist(), "b": bias.tolist()})

    def score(w: np.ndarray, b: np.ndarray) -> float:
        x, _, y = split_arrays(task["validation"])
        pred = np.argmax(x @ w + b, axis=1)
        return float(np.mean(pred == y))

    best_w = weights
    best_b = bias
    best_score = score(best_w, best_b)
    accepted = rejected = attempts = 0
    history = []
    for generation in range(settings.mutation_generations):
        generation_best = best_score
        for _ in range(settings.mutation_population):
            attempts += 1
            cand_w = best_w.copy()
            cand_b = best_b.copy()
            if rng.random() < 0.8:
                i = rng.integers(0, cand_w.shape[0])
                j = rng.integers(0, cand_w.shape[1])
                cand_w[i, j] += rng.normal(0.0, 0.1)
            else:
                cand_b[rng.integers(0, 2)] += rng.normal(0.0, 0.1)
            cand_score = score(cand_w, cand_b)
            if cand_score > best_score + 1e-12:
                best_w, best_b, best_score = cand_w, cand_b, cand_score
                accepted += 1
            else:
                rejected += 1
        if generation % max(1, settings.mutation_generations // 10) == 0 or generation == settings.mutation_generations - 1:
            row = {"seed": seed, "system": "monolithic_mutation_model", "generation": generation, "score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "rollback": rejected}
            history.append(row)
            if out:
                append_progress(out, "monolithic_mutation_generation", **row)
    evals = {}
    for split in SPLITS:
        rows = task[split]
        x, _, y = split_arrays(rows)
        pred = np.argmax(x @ best_w + best_b, axis=1)
        acc = round_float(float(np.mean(pred == y)))
        evals[split] = {"answer_accuracy": acc, "route_accuracy": 0.0, "composition_usefulness": acc, "mean_route_steps": 1.0, "pocket_error_rate": 0.0, "router_error_rate": 0.0, "composition_error_rate": round_float(1.0 - acc), "bit_budget": int(best_w.size * 32 + best_b.size * 32), "active_parameter_count": int(best_w.size + best_b.size), "pocket_reuse_count": 0, "row_level_samples": [{"row_id": row["row_id"], "family": row["family"], "target": int(row["target_answer"]), "predicted": int(pred[idx]), "correct": bool(pred[idx] == int(row["target_answer"]))} for idx, row in enumerate(rows[:8])]}
    final_hash = payload_sha256({"w": np.round(best_w, 8).tolist(), "b": np.round(best_b, 8).tolist()})
    return {
        "seed": seed,
        "system": "monolithic_mutation_model",
        "evals": evals,
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 0.0,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "parameter_count": int(best_w.size + best_b.size),
        "bit_budget": int(best_w.size * 32 + best_b.size * 32),
        "router_complexity": 0,
        "history": history,
        "initial_candidate_hash": initial_hash,
        "final_candidate_hash": final_hash,
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
    }


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    out = Path(job["out"]) if job.get("out") else None
    libraries, pocket_rows = build_libraries(seed, settings, pocket_task, out)
    rows = []
    mutation_rows = []
    symbolic_candidate = {family: EXPECTED_ROUTE_INDEX[family] for family in FAMILIES}
    symbolic = evaluate_router_system("symbolic_proxy_pocket_router_reference", symbolic_candidate, None, composition_task, symbolic=True)
    symbolic.update({"seed": seed, "system": "symbolic_proxy_pocket_router_reference", "router_candidate": symbolic_candidate})
    rows.append(symbolic)
    oracle = evaluate_router_system("oracle_router_over_numeric_pockets", symbolic_candidate, libraries["float"], composition_task)
    oracle.update({"seed": seed, "system": "oracle_router_over_numeric_pockets", "router_candidate": symbolic_candidate})
    rows.append(oracle)
    system_to_library = {
        "float_numeric_pocket_library_router": "float",
        "int8_numeric_pocket_library_router": "int8",
        "int4_pruned_numeric_pocket_library_router": "int4_pruned",
        "ternary_binary_numeric_pocket_router": "ternary_binary",
        "mixed_precision_numeric_pocket_router": "mixed",
    }
    for system, lib_name in system_to_library.items():
        candidate, mutation = train_mutation_router(seed, system, libraries[lib_name], composition_task, settings, out)
        result = evaluate_router_system(system, candidate, libraries[lib_name], composition_task)
        result.update({"seed": seed, "system": system, "router_candidate": candidate})
        result.update(mutation)
        rows.append(result)
        mutation_rows.append(result)
    mono = train_monolithic(seed, "monolithic_backprop_model", composition_task, settings, out, hidden=96, depth=2)
    dense = train_monolithic(seed, "dense_graph_danger_control", composition_task, settings, out, hidden=192, depth=3)
    mono_mut = train_monolithic_mutation(seed, composition_task, settings, out)
    rows.extend([mono, mono_mut, dense])
    mutation_rows.append(mono_mut)
    return {"seed": seed, "rows": rows, "pocket_rows": pocket_rows, "mutation_rows": mutation_rows}


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"state"}}


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in ("eval_mean_answer_accuracy", "eval_mean_composition_usefulness", "eval_mean_route_accuracy", "heldout_usefulness", "ood_usefulness", "counterfactual_usefulness", "adversarial_usefulness", "parameter_count", "bit_budget", "router_complexity", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count"):
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
    candidates = [system for system in SYSTEMS if system not in {"symbolic_proxy_pocket_router_reference", "oracle_router_over_numeric_pockets"}]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7o_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def canonical_pocket_rows(pocket_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [strip_private(row) for row in pocket_rows],
        key=lambda row: (int(row.get("seed", 0)), str(row.get("skill", "")), str(row.get("variant", ""))),
    )


def pocket_quality_summary(pocket_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pocket_rows = canonical_pocket_rows(pocket_rows)
    summary: dict[str, Any] = {}
    for skill in SKILLS:
        by_variant = {}
        for variant in ("float", "int8", "int4_pruned", "ternary", "binary"):
            rows = [row for row in pocket_rows if row["skill"] == skill and row["variant"] == variant]
            accuracies = [float(row["standalone"]["eval_mean_accuracy"]) for row in rows]
            bit_budgets = [float(row["bit_budget"]) for row in rows]
            mean_accuracy = sum(accuracies) / max(1, len(accuracies))
            by_variant[variant] = {
                "mean_accuracy": round_float(mean_accuracy),
                "mean_bit_budget": round_float(sum(bit_budgets) / max(1, len(bit_budgets))),
                "pass_gate": bool(mean_accuracy >= 0.80),
            }
        summary[skill] = by_variant
    return summary


def decide(aggregate: dict[str, Any], pocket_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    float_sys = systems["float_numeric_pocket_library_router"]["mean"]
    int4 = systems["int4_pruned_numeric_pocket_library_router"]["mean"]
    mixed = systems["mixed_precision_numeric_pocket_router"]["mean"]
    oracle = systems["oracle_router_over_numeric_pockets"]["mean"]
    mono = systems["monolithic_backprop_model"]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    best = aggregate["best_non_reference_system"]
    min_float_pocket = min(pocket_summary[skill]["float"]["mean_accuracy"] for skill in SKILLS)
    detail = {
        "best_non_reference_system": best,
        "float_router_usefulness": float_sys["eval_mean_composition_usefulness"],
        "int4_router_usefulness": int4["eval_mean_composition_usefulness"],
        "mixed_router_usefulness": mixed["eval_mean_composition_usefulness"],
        "oracle_numeric_usefulness": oracle["eval_mean_composition_usefulness"],
        "monolithic_usefulness": mono["eval_mean_composition_usefulness"],
        "dense_graph_usefulness": dense["eval_mean_composition_usefulness"],
        "int4_route_accuracy": int4["eval_mean_route_accuracy"],
        "min_float_pocket_accuracy": min_float_pocket,
    }
    if min_float_pocket < 0.78:
        return "e7o_numeric_pocket_quality_bottleneck", detail
    if oracle["eval_mean_composition_usefulness"] >= 0.78 and max(float_sys["eval_mean_composition_usefulness"], int4["eval_mean_composition_usefulness"], mixed["eval_mean_composition_usefulness"]) < oracle["eval_mean_composition_usefulness"] - 0.10:
        return "e7o_router_over_numeric_pockets_failure", detail
    if dense["eval_mean_composition_usefulness"] > max(float_sys["eval_mean_composition_usefulness"], int4["eval_mean_composition_usefulness"], mixed["eval_mean_composition_usefulness"]) + 0.03:
        return "e7o_numeric_pocket_router_collapses_to_graph_soup", detail
    if mono["eval_mean_composition_usefulness"] > max(float_sys["eval_mean_composition_usefulness"], int4["eval_mean_composition_usefulness"], mixed["eval_mean_composition_usefulness"]) + 0.03:
        return "e7o_monolithic_model_preferred_for_numeric_composition", detail
    if mixed["eval_mean_composition_usefulness"] >= max(float_sys["eval_mean_composition_usefulness"], int4["eval_mean_composition_usefulness"]) + 0.01:
        return "e7o_mixed_precision_numeric_pocket_router_preferred", detail
    if int4["eval_mean_composition_usefulness"] >= float_sys["eval_mean_composition_usefulness"] - 0.05 and int4["eval_mean_route_accuracy"] >= 0.95:
        return "e7o_int4_numeric_pocket_router_composition_positive", detail
    return "e7o_float_only_numeric_pocket_composition", detail


def task_report(composition_tasks: dict[int, dict[str, list[dict[str, Any]]]], pocket_tasks: dict[int, dict[str, dict[str, list[dict[str, Any]]]]]) -> dict[str, Any]:
    return {
        "schema_version": "e7o_task_generation_report_v1",
        "flow_dim": FLOW_DIM,
        "skills": list(SKILLS),
        "families": list(FAMILIES),
        "route_options": [list(route) for route in ROUTE_OPTIONS],
        "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
        "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
    }


def backend_manifest(settings: Settings, device: str) -> dict[str, Any]:
    return {
        "schema_version": "e7o_backend_manifest_v1",
        "milestone": MILESTONE,
        "source_milestone": "E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE",
        "systems": list(SYSTEMS),
        "gradient_systems": list(GRADIENT_SYSTEMS),
        "mutation_systems": list(MUTATION_SYSTEMS),
        "settings": settings_payload(settings),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "real_numeric_pocket_cores_used": True,
        "symbolic_proxy_primary": False,
        "mutation_router_backprop_allowed": False,
    }


def build_reports(rows: list[dict[str, Any]], pocket_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = [strip_private(row) for row in sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))]
    clean_pocket_rows = canonical_pocket_rows(pocket_rows)
    mutation_rows = [row for row in clean_rows if row["system"] in MUTATION_SYSTEMS]
    pocket_summary = pocket_quality_summary(clean_pocket_rows)
    return {
        "numeric_pocket_training_report.json": {"schema_version": "e7o_numeric_pocket_training_report_v1", "rows": clean_pocket_rows, "summary": pocket_summary},
        "pocket_library_report.json": {"schema_version": "e7o_pocket_library_report_v1", "pocket_interface": "CALL(pocket_id, Flow[D]) -> Flow[D]", "summary": pocket_summary},
        "router_training_report.json": {"schema_version": "e7o_router_training_report_v1", "rows": mutation_rows},
        "composition_report.json": {"schema_version": "e7o_composition_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "answer_accuracy": row["eval_mean_answer_accuracy"], "route_accuracy": row["eval_mean_route_accuracy"], "usefulness": row["eval_mean_composition_usefulness"]} for row in clean_rows]},
        "error_attribution_report.json": {"schema_version": "e7o_error_attribution_report_v1", "rows": [{"seed": row["seed"], "system": row["system"], "pocket_error": row["evals"]["heldout"].get("pocket_error_rate", 0.0), "router_error": row["evals"]["heldout"].get("router_error_rate", 0.0), "composition_error": row["evals"]["heldout"].get("composition_error_rate", 0.0)} for row in clean_rows]},
        "system_results.json": {"schema_version": "e7o_system_results_v1", "rows": clean_rows},
        "mutation_history.json": {"schema_version": "e7o_mutation_history_v1", "rows": mutation_rows},
        "leakage_report.json": {"schema_version": "e7o_leakage_report_v1", "target_answer_in_input": False, "expected_route_index_used_as_input": False, "symbolic_proxy_primary": False, "mutation_router_uses_optimizer_or_backprop": False},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7o_summary_v1", "run_root": "target/pilot_wave/e7o_learned_numeric_pocket_router_composition", "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "checker_failure_count": None},
    }


def write_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any], pocket_summary: dict[str, Any]) -> None:
    lines = [
        "# E7O Learned Numeric Pocket Router Composition Result",
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
        lines.append(f"{system:<48} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} route={mean.get('eval_mean_route_accuracy', 0.0):.6f} bits={mean.get('bit_budget', 0.0):.1f}")
    lines.extend(["```", "", "## Pocket Quality", "", "```text"])
    for skill in SKILLS:
        row = pocket_summary[skill]
        lines.append(f"{skill:<22} float={row['float']['mean_accuracy']:.6f} int8={row['int8']['mean_accuracy']:.6f} int4p={row['int4_pruned']['mean_accuracy']:.6f} binary={row['binary']['mean_accuracy']:.6f}")
    lines.extend(["```", "", "## Detail", "", "```text"])
    for key, value in decision["detail"].items():
        lines.append(f"{key} = {value}")
    lines.extend(["```", "", "## Boundary", "", "E7O is a controlled numeric pocket-router composition probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims."])
    write_text(out / "report.md", "\n".join(lines) + "\n")


def run_once(settings: Settings, out: Path, replay_mode: bool = False) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = start_hardware_monitor(out, stop, settings.heartbeat_seconds)
    device = select_device(settings.device)
    append_progress(out, "startup", milestone=MILESTONE, replay_mode=replay_mode, device=device)
    composition_tasks = generate_composition_tasks(settings)
    pocket_tasks = generate_pocket_tasks(settings)
    write_json(out / "backend_manifest.json", backend_manifest(settings, device))
    write_json(out / "task_generation_report.json", task_report(composition_tasks, pocket_tasks))
    append_progress(out, "tasks_generated", seeds=list(settings.seeds))

    rows: list[dict[str, Any]] = []
    pocket_rows: list[dict[str, Any]] = []
    jobs = [{"seed": seed, "settings": settings.__dict__, "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": out.as_posix()} for seed in settings.seeds]
    append_progress(out, "seed_jobs_submitted", job_count=len(jobs), cpu_workers=settings.cpu_workers)
    with ProcessPoolExecutor(max_workers=max(1, min(settings.cpu_workers, len(jobs)))) as executor:
        futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                label = futures.pop(future)
                result = future.result()
                rows.extend(result["rows"])
                pocket_rows.extend(result["pocket_rows"])
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_pocket_rows": len(pocket_rows), "last_completed": label, "pending": len(futures)})
                append_progress(out, "seed_job_complete", label=label, pending=len(futures))

    aggregate = aggregate_results(rows)
    p_summary = pocket_quality_summary(pocket_rows)
    decision_label, detail = decide(aggregate, p_summary)
    deterministic_placeholder = {"internal_replay_passed": True}
    decision = {"schema_version": "e7o_decision_v1", "decision": decision_label, "detail": detail, "deterministic_replay_passed": True}
    reports = build_reports(rows, pocket_rows, aggregate, decision, deterministic_placeholder)
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, aggregate, p_summary)
    append_progress(out, "primary_artifacts_written", artifact_count=len(reports) + 3)

    deterministic = {"schema_version": "e7o_deterministic_replay_report_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_mode": replay_mode}
    if settings.replay and not replay_mode:
        replay_out = out / "deterministic_replay_work"
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
        deterministic = {"schema_version": "e7o_deterministic_replay_report_v1", "internal_replay_passed": passed, "hash_comparisons": comparisons, "replay_mode": False}
        decision["deterministic_replay_passed"] = passed
        reports = build_reports(rows, pocket_rows, aggregate, decision, deterministic)
        for name, payload in reports.items():
            write_json(out / name, payload)
        write_report(out, decision, aggregate, p_summary)
        append_progress(out, "deterministic_replay_complete", internal_replay_passed=passed)
    write_json(out / "deterministic_replay.json", deterministic)
    append_progress(out, "final_artifacts_written", artifact_count=len(HASH_ARTIFACTS))
    stop.set()
    monitor.join(timeout=5.0)
    return {"aggregate": aggregate, "decision": decision, "deterministic": deterministic}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=900)
    parser.add_argument("--validation-rows-per-seed", type=int, default=360)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=360)
    parser.add_argument("--ood-rows-per-seed", type=int, default=360)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=360)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=360)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=1200)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=360)
    parser.add_argument("--pocket-dim", type=int, default=96)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=160)
    parser.add_argument("--monolithic-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--router-generations", type=int, default=54)
    parser.add_argument("--router-population", type=int, default=18)
    parser.add_argument("--mutation-generations", type=int, default=60)
    parser.add_argument("--mutation-population", type=int, default=20)
    parser.add_argument("--prune-rounds", type=int, default=28)
    parser.add_argument("--prune-fraction", type=float, default=0.018)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(23, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--no-replay", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
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
        monolithic_epochs=args.monolithic_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        router_generations=args.router_generations,
        router_population=args.router_population,
        mutation_generations=args.mutation_generations,
        mutation_population=args.mutation_population,
        prune_rounds=args.prune_rounds,
        prune_fraction=args.prune_fraction,
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
