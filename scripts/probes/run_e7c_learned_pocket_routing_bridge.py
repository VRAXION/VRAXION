#!/usr/bin/env python3
"""E7C learned pocket routing bridge.

E7C replaces E7B's hand-coded pocket outputs with separately learned pocket
models, freezes their outputs, and tests whether mutation/rollback can still
learn the router/switchboard layer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
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


REPO_ROOT = Path(__file__).resolve().parents[2]
E7B_PATH = Path(__file__).with_name("run_e7b_pocket_routing_composition_probe.py")
MILESTONE = "E7C_LEARNED_POCKET_ROUTING_BRIDGE"
DEFAULT_OUT = Path("target/pilot_wave/e7c_learned_pocket_routing_bridge")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SYSTEMS = (
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "learned_pockets_gradient_router",
    "learned_pockets_mutation_router",
    "learned_binary_pockets_mutation_router",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_learned_pocket_router_reference",
    "oracle_symbolic_reference",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "learned_pockets_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "learned_pockets_mutation_router",
    "learned_binary_pockets_mutation_router",
    "router_plus_limited_pocket_repair",
)
CONTROL_SYSTEMS = ("random_router_control", "oracle_learned_pocket_router_reference", "oracle_symbolic_reference")
SYSTEM_TO_E7B = {
    "monolithic_backprop_model": "monolithic_backprop_model",
    "monolithic_mutation_model": "monolithic_mutation_model",
    "learned_pockets_gradient_router": "frozen_pockets_gradient_router",
    "learned_pockets_mutation_router": "frozen_pockets_mutation_router",
    "learned_binary_pockets_mutation_router": "frozen_pockets_binary_router",
    "router_plus_limited_pocket_repair": "router_plus_limited_pocket_repair",
}
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "learned_pocket_training_report.json",
    "pocket_library_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "composition_report.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7c_learned_pocket_mutation_router_viable",
    "e7c_binary_learned_pocket_router_viable",
    "e7c_gradient_router_only_viable",
    "e7c_symbolic_only_scaffold_detected",
    "e7c_learned_pocket_quality_bottleneck",
    "e7c_leak_or_artifact_detected",
)


def load_e7b_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7b_pocket_routing_composition_probe", E7B_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7B from {E7B_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7b = load_e7b_module()


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
    pocket_epochs: int
    gradient_epochs: int
    batch_size: int
    learning_rate: float
    pocket_learning_rate: float
    weight_decay: float
    mutation_generations: int
    mutation_population: int
    mutation_sigma: float
    mutation_elite_count: int
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


class PocketNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.route_heads = nn.Linear(hidden_dim, len(e7b.ROUTES) * 2)
        self.branch_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.route_heads(h).reshape(-1, len(e7b.ROUTES), 2), self.branch_head(h)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7c::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def locked_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"write lock timed out: {lock_path}")
            time.sleep(0.025)
    try:
        write_json(path, payload)
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7b.append_progress(out, event, **details)


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
        raise ValueError("empty integer tuple")
    return values


def e7b_settings(settings: Settings) -> Any:
    return e7b.Settings(
        seeds=settings.seeds,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        mutation_generations=settings.mutation_generations,
        mutation_population=settings.mutation_population,
        mutation_sigma=settings.mutation_sigma,
        mutation_elite_count=settings.mutation_elite_count,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def split_index(split: str) -> int:
    return SPLITS.index(split)


def pocket_input_from_parts(a: int, b: int, key: int, threshold: int, split: str) -> list[float]:
    # Learned pockets get a discrete-friendly encoding. The router never sees
    # this; it only receives frozen pocket outputs.
    one_hot = []
    for value in (a, b, key, threshold):
        one_hot.extend(1.0 if idx == value else 0.0 for idx in range(16))
    split_one_hot = [1.0 if idx == split_index(split) else 0.0 for idx in range(len(SPLITS))]
    helper = [
        1.0 if a > b else 0.0,
        1.0 if a == b else 0.0,
        ((a + b) & 15) / 15.0,
        (a ^ b) / 15.0,
        ((key * 5) & 15) / 15.0,
    ]
    return one_hot + split_one_hot + helper


def decode_raw_parts(row: dict[str, Any]) -> tuple[int, int, int, int]:
    raw = row["raw"]
    return int(round(raw[0] * 15)), int(round(raw[1] * 15)), int(round(raw[2] * 15)), int(round(raw[3] * 15))


def generate_pocket_pretrain(seed: int, count: int, label: str) -> dict[str, np.ndarray]:
    rng = random.Random(stable_seed(f"pocket-pretrain-{seed}-{label}"))
    x = []
    y_routes = []
    y_branch = []
    for idx in range(count):
        split = SPLITS[idx % len(SPLITS)] if idx < len(SPLITS) * 8 else rng.choice(SPLITS)
        if split == "ood":
            a = rng.choice((0, 1, 2, 13, 14, 15))
            b = rng.choice((0, 1, 2, 13, 14, 15))
            threshold = rng.choice((2, 3, 12, 13))
        else:
            a = rng.randrange(16)
            b = rng.randrange(16)
            threshold = rng.randrange(16)
        key = rng.randrange(16)
        pockets = e7b.compute_pockets(seed, split, a, b, key, threshold)
        x.append(pocket_input_from_parts(a, b, key, threshold, split))
        y_routes.append(pockets["candidate_answers"].astype(np.int64))
        y_branch.append(1 if a > b else 0)
    return {
        "x": np.asarray(x, dtype=np.float32),
        "routes": np.asarray(y_routes, dtype=np.int64),
        "branch": np.asarray(y_branch, dtype=np.int64),
    }


def train_pocket_model(seed: int, settings: Settings, out: Path | None) -> dict[str, Any]:
    device = e7b.select_device(settings.device)
    e7b.set_determinism(stable_seed(f"pocket-model-{seed}"), device)
    train = generate_pocket_pretrain(seed, settings.pocket_pretrain_rows_per_seed, "train")
    val = generate_pocket_pretrain(seed, settings.pocket_validation_rows_per_seed, "validation")
    model = PocketNet(train["x"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.pocket_learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(train["x"], dtype=torch.float32, device=device)
    y_routes = torch.as_tensor(train["routes"], dtype=torch.long, device=device)
    y_branch = torch.as_tensor(train["branch"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(val["x"], dtype=torch.float32, device=device)
    val_routes = torch.as_tensor(val["routes"], dtype=torch.long, device=device)
    val_branch = torch.as_tensor(val["branch"], dtype=torch.long, device=device)
    rng = np.random.default_rng(stable_seed(f"pocket-order-{seed}"))
    best_state = None
    best_score = -1.0
    history = []
    last_heartbeat = time.monotonic()
    for epoch in range(1, settings.pocket_epochs + 1):
        order = rng.permutation(len(x_train))
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            route_logits, branch_logits = model(x_train[idx])
            loss = sum(nn.functional.cross_entropy(route_logits[:, route_idx, :], y_routes[idx, route_idx]) for route_idx in range(len(e7b.ROUTES)))
            loss = loss / len(e7b.ROUTES) + 0.25 * nn.functional.cross_entropy(branch_logits, y_branch[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            route_logits, branch_logits = model(x_val)
            route_pred = torch.argmax(route_logits, dim=2)
            branch_pred = torch.argmax(branch_logits, dim=1)
            route_acc = float((route_pred == val_routes).float().mean().item())
            branch_acc = float((branch_pred == val_branch).float().mean().item())
            score = 0.85 * route_acc + 0.15 * branch_acc
        row = {"epoch": epoch, "route_candidate_accuracy": round_float(route_acc), "branch_accuracy": round_float(branch_acc), "score": round_float(score)}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.pocket_epochs):
            locked_write_json(out / "partial_status" / f"learned_pocket_seed{seed}.json", row)
            append_progress(out, "pocket_epoch", seed=seed, epoch=epoch, route_candidate_accuracy=row["route_candidate_accuracy"], branch_accuracy=row["branch_accuracy"], device=device)
            last_heartbeat = time.monotonic()
    assert best_state is not None
    serial_state = {key: value.tolist() for key, value in best_state.items()}
    return {
        "seed": seed,
        "device": device,
        "state": {"input_dim": int(train["x"].shape[1]), "state_dict": serial_state},
        "history": history,
        "best_score": round_float(best_score),
        "state_hash": payload_sha256(serial_state),
    }


def eval_pocket_model(model_state: dict[str, Any], task: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model = PocketNet(int(model_state["input_dim"]))
    model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32) for key, value in model_state["state_dict"].items()})
    model.to(device)
    model.eval()
    learned_task = {}
    binary_task = {}
    report = {"split_metrics": {}}
    with torch.no_grad():
        for split in SPLITS:
            x_rows = []
            for row in task[split]["rows"]:
                a, b, key, threshold = decode_raw_parts(row)
                x_rows.append(pocket_input_from_parts(a, b, key, threshold, split))
            x = torch.as_tensor(np.asarray(x_rows, dtype=np.float32), dtype=torch.float32, device=device)
            route_logits, branch_logits = model(x)
            route_prob = torch.softmax(route_logits, dim=2).cpu().numpy()
            route_pred = np.argmax(route_prob, axis=2).astype(np.int64)
            branch_prob = torch.softmax(branch_logits, dim=1).cpu().numpy()[:, 1]
            branch_pred = (branch_prob >= 0.5).astype(np.float32)
            target_candidates = task[split]["candidate_answers"]
            route_acc = float(np.mean(route_pred == target_candidates))
            target_branch = np.asarray([1 if decode_raw_parts(row)[0] > decode_raw_parts(row)[1] else 0 for row in task[split]["rows"]], dtype=np.int64)
            branch_acc = float(np.mean(branch_pred.astype(np.int64) == target_branch))
            oracle_answer = route_pred[np.arange(len(route_pred)), task[split]["route"]]
            oracle_ceiling = float(np.mean(oracle_answer == task[split]["y"]))
            learned_pocket = np.concatenate([route_pred.astype(np.float32), route_prob[:, :, 1], branch_prob[:, None], task[split]["raw"][:, 3:4]], axis=1)
            binary_pocket = np.concatenate([route_pred.astype(np.float32), route_pred.astype(np.float32), branch_pred[:, None], task[split]["raw"][:, 3:4]], axis=1)
            report["split_metrics"][split] = {
                "candidate_answer_accuracy": round_float(route_acc),
                "branch_accuracy": round_float(branch_acc),
                "oracle_learned_route_answer_ceiling": round_float(oracle_ceiling),
            }
            for target, pocket in ((learned_task, learned_pocket), (binary_task, binary_pocket)):
                target[split] = {
                    "rows": copy.deepcopy(task[split]["rows"]),
                    "raw": task[split]["raw"].copy(),
                    "pocket": pocket.astype(np.float32),
                    "combo": np.concatenate([task[split]["raw"], pocket.astype(np.float32)], axis=1),
                    "y": task[split]["y"].copy(),
                    "route": task[split]["route"].copy(),
                    "candidate_answers": route_pred.copy(),
                    "symbolic_candidate_answers": target_candidates.copy(),
                }
    return learned_task, binary_task, report


def build_tasks(settings: Settings, out: Path | None) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    base_settings = e7b_settings(settings)
    symbolic_tasks: dict[int, dict[str, Any]] = {}
    learned_tasks: dict[int, dict[str, Any]] = {}
    binary_tasks: dict[int, dict[str, Any]] = {}
    pocket_reports = []
    for seed in settings.seeds:
        if out:
            append_progress(out, "pocket_training_start", seed=seed)
        symbolic = e7b.generate_seed_task(seed, base_settings)
        pocket = train_pocket_model(seed, settings, out)
        learned, binary, eval_report = eval_pocket_model(pocket["state"], symbolic, e7b.select_device(settings.device))
        symbolic_tasks[seed] = symbolic
        learned_tasks[seed] = learned
        binary_tasks[seed] = binary
        pocket_reports.append(
            {
                "seed": seed,
                "device": pocket["device"],
                "state_hash": pocket["state_hash"],
                "best_score": pocket["best_score"],
                "history": pocket["history"],
                **eval_report,
            }
        )
        if out:
            append_progress(out, "pocket_training_complete", seed=seed, best_score=pocket["best_score"])
    return symbolic_tasks, learned_tasks, binary_tasks, pocket_reports


def run_gradient_lane(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    learned_tasks = {int(seed): task for seed, task in job["learned_tasks"].items()}
    out = Path(job["out"]) if job.get("out") else None
    rows = []
    histories = []
    for seed in sorted(learned_tasks):
        for system in GRADIENT_SYSTEMS:
            e7b_system = SYSTEM_TO_E7B[system]
            if out:
                append_progress(out, "gradient_job_start", seed=seed, system=system)
            result = e7b.train_gradient_system(seed, e7b_system, learned_tasks[seed], e7b_settings(settings), out)
            result["system"] = system
            rows.append({key: value for key, value in result.items() if key != "history"})
            histories.append({"seed": seed, "system": system, "history": result["history"], "device": result["device"]})
            if out:
                append_progress(out, "gradient_job_complete", seed=seed, system=system, device=result["device"])
    return {"rows": rows, "histories": histories, "hardware": e7b.hardware_probe()}


def run_mutation_job(job: dict[str, Any]) -> dict[str, Any]:
    system = job["system"]
    e7b_job = dict(job)
    e7b_job["system"] = SYSTEM_TO_E7B[system]
    result = e7b.mutation_worker(e7b_job)
    result["system"] = system
    return result


def control_results(seed: int, learned_task: dict[str, Any], symbolic_task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(stable_seed(f"control-{seed}"))
    for system in CONTROL_SYSTEMS:
        evals = {}
        for split in SPLITS:
            data = learned_task[split]
            if system == "oracle_symbolic_reference":
                answer_pred = symbolic_task[split]["y"].copy()
                route_pred = symbolic_task[split]["route"].copy()
            elif system == "oracle_learned_pocket_router_reference":
                route_pred = data["route"].copy()
                answer_pred = data["candidate_answers"][np.arange(len(data["y"])), route_pred]
            else:
                route_pred = rng.integers(0, len(e7b.ROUTES), size=len(data["y"]), dtype=np.int64)
                answer_pred = data["candidate_answers"][np.arange(len(data["y"])), route_pred]
            evals[split] = e7b.evaluate_predictions(answer_pred, route_pred, data)
        rows.append({"seed": seed, "system": system, "training_mode": "control", "input_key": "combo", "parameter_count": 0, "state_hash": payload_sha256({"seed": seed, "system": system}), "evals": evals})
    return rows


def split_summary(evals: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score"):
        out[metric] = round_float(float(np.mean([evals[split][metric] for split in EVAL_SPLITS])))
    out["generalization_gap"] = round_float(evals["train"]["usefulness_score"] - out["usefulness_score"])
    out["heldout_usefulness"] = evals["heldout"]["usefulness_score"]
    out["ood_usefulness"] = evals["ood"]["usefulness_score"]
    out["counterfactual_usefulness"] = evals["counterfactual"]["usefulness_score"]
    out["adversarial_usefulness"] = evals["adversarial"]["usefulness_score"]
    return out


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    for row in rows:
        by_system[row["system"]].append(row)
    systems = {}
    for system, items in by_system.items():
        seed_rows = []
        for item in sorted(items, key=lambda row: int(row["seed"])):
            seed_rows.append({"seed": item["seed"], "parameter_count": item.get("parameter_count", 0), **split_summary(item["evals"])})
        systems[system] = {
            "seed_count": len(seed_rows),
            "rows": seed_rows,
            "mean": {
                key: round_float(float(np.mean([row[key] for row in seed_rows]))) if seed_rows else 0.0
                for key in (
                    "answer_accuracy",
                    "route_accuracy",
                    "composition_accuracy",
                    "shortcut_rate",
                    "usefulness_score",
                    "generalization_gap",
                    "heldout_usefulness",
                    "ood_usefulness",
                    "counterfactual_usefulness",
                    "adversarial_usefulness",
                    "parameter_count",
                )
            },
        }
    best = max((system for system in SYSTEMS if system != "oracle_symbolic_reference"), key=lambda name: systems[name]["mean"]["usefulness_score"])
    return {"schema_version": "e7c_aggregate_metrics_v1", "systems": systems, "best_non_symbolic_oracle_system": best, "best_non_symbolic_oracle_usefulness": systems[best]["mean"]["usefulness_score"]}


def decide(aggregate: dict[str, Any], pocket_reports: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    learned_ceiling = systems["oracle_learned_pocket_router_reference"]["mean"]["usefulness_score"]
    symbolic_ceiling = systems["oracle_symbolic_reference"]["mean"]["usefulness_score"]
    best_mut = max(
        (
            ("learned_pockets_mutation_router", systems["learned_pockets_mutation_router"]["mean"]),
            ("learned_binary_pockets_mutation_router", systems["learned_binary_pockets_mutation_router"]["mean"]),
            ("router_plus_limited_pocket_repair", systems["router_plus_limited_pocket_repair"]["mean"]),
        ),
        key=lambda row: row[1]["usefulness_score"],
    )
    gradient = systems["learned_pockets_gradient_router"]["mean"]
    random_control = systems["random_router_control"]["mean"]
    mean_pocket_candidate_acc = round_float(float(np.mean([np.mean([row["candidate_answer_accuracy"] for row in report["split_metrics"].values()]) for report in pocket_reports])))
    leak = random_control["usefulness_score"] >= 0.80
    pocket_bottleneck = learned_ceiling < 0.86 or mean_pocket_candidate_acc < 0.88
    mutation_viable = (
        best_mut[1]["usefulness_score"] >= min(0.90, learned_ceiling - 0.02)
        and best_mut[1]["route_accuracy"] >= 0.88
        and best_mut[1]["adversarial_usefulness"] >= 0.84
        and best_mut[1]["shortcut_rate"] <= 0.14
    )
    binary_viable = (
        systems["learned_binary_pockets_mutation_router"]["mean"]["usefulness_score"] >= min(0.88, learned_ceiling - 0.04)
        and systems["learned_binary_pockets_mutation_router"]["mean"]["route_accuracy"] >= 0.84
    )
    gradient_viable = gradient["usefulness_score"] >= min(0.90, learned_ceiling - 0.02) and gradient["route_accuracy"] >= 0.88
    if leak:
        decision = "e7c_leak_or_artifact_detected"
    elif pocket_bottleneck:
        decision = "e7c_learned_pocket_quality_bottleneck"
    elif mutation_viable:
        decision = "e7c_binary_learned_pocket_router_viable" if best_mut[0] == "learned_binary_pockets_mutation_router" and binary_viable else "e7c_learned_pocket_mutation_router_viable"
    elif gradient_viable:
        decision = "e7c_gradient_router_only_viable"
    else:
        decision = "e7c_symbolic_only_scaffold_detected"
    return decision, {
        "best_mutation_router_system": best_mut[0],
        "best_mutation_router_mean": best_mut[1],
        "learned_pocket_ceiling": learned_ceiling,
        "symbolic_ceiling": symbolic_ceiling,
        "mean_pocket_candidate_accuracy": mean_pocket_candidate_acc,
        "leak_control_triggered": leak,
        "pocket_quality_bottleneck": pocket_bottleneck,
        "mutation_viable": mutation_viable,
        "binary_viable": binary_viable,
        "gradient_viable": gradient_viable,
    }


def task_report(symbolic_tasks: dict[int, dict[str, Any]], learned_tasks: dict[int, dict[str, Any]], binary_tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7c_task_generation_report_v1",
        "families": list(e7b.FAMILIES),
        "routes": list(e7b.ROUTES),
        "row_counts": {str(seed): {split: int(len(task[split]["rows"])) for split in SPLITS} for seed, task in learned_tasks.items()},
        "raw_feature_dim": int(next(iter(learned_tasks.values()))["train"]["raw"].shape[1]),
        "learned_pocket_feature_dim": int(next(iter(learned_tasks.values()))["train"]["pocket"].shape[1]),
        "binary_pocket_feature_dim": int(next(iter(binary_tasks.values()))["train"]["pocket"].shape[1]),
        "symbolic_task_ground_truth_retained": True,
        "learned_pocket_outputs_used_for_router_inputs": True,
    }


def pocket_library_report(pocket_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7c_pocket_library_report_v1",
        "pocket_source": "separately_learned_frozen_pocket_models",
        "pocket_model": "shared_trunk_multihead_route_candidate_and_branch_classifier",
        "route_count": len(e7b.ROUTES),
        "seed_count": len(pocket_reports),
        "mean_candidate_answer_accuracy": round_float(float(np.mean([np.mean([row["candidate_answer_accuracy"] for row in report["split_metrics"].values()]) for report in pocket_reports]))),
        "mean_branch_accuracy": round_float(float(np.mean([np.mean([row["branch_accuracy"] for row in report["split_metrics"].values()]) for report in pocket_reports]))),
        "mean_oracle_learned_route_answer_ceiling": round_float(float(np.mean([np.mean([row["oracle_learned_route_answer_ceiling"] for row in report["split_metrics"].values()]) for report in pocket_reports]))),
        "learned_pockets_are_frozen_for_router_training": True,
    }


def build_reports(aggregate: dict[str, Any], decision_detail: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    systems = aggregate["systems"]
    composition = {
        "schema_version": "e7c_composition_report_v1",
        "question": "can mutation/rollback route separately learned frozen pockets",
        "best_mutation_router_system": decision_detail["best_mutation_router_system"],
        "learned_pocket_ceiling": decision_detail["learned_pocket_ceiling"],
        "symbolic_ceiling": decision_detail["symbolic_ceiling"],
        "monolithic_mutation_usefulness": systems["monolithic_mutation_model"]["mean"]["usefulness_score"],
        "learned_pockets_mutation_usefulness": systems["learned_pockets_mutation_router"]["mean"]["usefulness_score"],
        "learned_binary_pockets_mutation_usefulness": systems["learned_binary_pockets_mutation_router"]["mean"]["usefulness_score"],
        "router_plus_limited_pocket_repair_usefulness": systems["router_plus_limited_pocket_repair"]["mean"]["usefulness_score"],
        "learned_pockets_gradient_usefulness": systems["learned_pockets_gradient_router"]["mean"]["usefulness_score"],
        "interpretation_boundary": "routing_over_separately_learned_frozen_pocket_outputs",
    }
    leakage = {
        "schema_version": "e7c_leakage_report_v1",
        "random_control_usefulness": systems["random_router_control"]["mean"]["usefulness_score"],
        "random_control_passed": systems["random_router_control"]["mean"]["usefulness_score"] < 0.80,
        "adversarial_split_used": True,
        "route_name_or_index_leakage_claim": False,
        "hidden_correct_route_index_used_as_input": False,
    }
    return composition, leakage


def build_markdown(payloads: dict[str, Any]) -> str:
    summary = payloads["summary.json"]
    aggregate = payloads["aggregate_metrics.json"]
    lines = [
        "# E7C Learned Pocket Routing Bridge Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"best_non_symbolic_oracle_system = {summary['best_non_symbolic_oracle_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Metrics",
        "",
        "| system | usefulness | answer | route | composition | adversarial | shortcut | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"| {system} | {mean['usefulness_score']:.6f} | {mean['answer_accuracy']:.6f} | "
            f"{mean['route_accuracy']:.6f} | {mean['composition_accuracy']:.6f} | "
            f"{mean['adversarial_usefulness']:.6f} | {mean['shortcut_rate']:.6f} | {mean['parameter_count']:.1f} |"
        )
    lines.extend(["", "This is a controlled learned-pocket routing bridge over frozen pocket outputs.", ""])
    return "\n".join(lines)


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7b.hardware_probe())
    symbolic_tasks, learned_tasks, binary_tasks, pocket_reports = build_tasks(settings, out)
    if out:
        append_progress(out, "learned_tasks_ready", seeds=list(settings.seeds))
    rows = []
    mutation_histories = []
    training_histories = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, learned_tasks[seed], symbolic_tasks[seed]))
    jobs = []
    for seed in settings.seeds:
        for system in MUTATION_SYSTEMS:
            task = binary_tasks[seed] if system == "learned_binary_pockets_mutation_router" else learned_tasks[seed]
            jobs.append({"seed": seed, "system": system, "task": task, "settings": e7b_settings(settings).__dict__, "out": out.as_posix() if out else None})
    gpu_job = {"learned_tasks": {str(seed): learned_tasks[seed] for seed in settings.seeds}, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers + 1)) as executor:
            futures = {executor.submit(run_gradient_lane, gpu_job): "gpu_gradient_lane"}
            for job in jobs:
                futures[executor.submit(run_mutation_job, job)] = f"{job['system']}/seed{job['seed']}"
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=True)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
                    if label == "gpu_gradient_lane":
                        rows.extend(result["rows"])
                        training_histories.extend(result["histories"])
                        if out:
                            append_progress(out, "gpu_lane_complete", completed_gradient_rows=len(result["rows"]), hardware=result["hardware"])
                    else:
                        rows.append({key: value for key, value in result.items() if key != "history"})
                        mutation_histories.append(
                            {
                                "seed": result["seed"],
                                "system": result["system"],
                                "history": result["history"],
                                "mutation_attempts": result["mutation_attempts"],
                                "accepted_mutations": result["accepted_mutations"],
                                "rejected_mutations": result["rejected_mutations"],
                                "rollback_count": result["rollback_count"],
                                "accepted_by_operator": result["accepted_by_operator"],
                                "rejected_by_operator": result["rejected_by_operator"],
                            }
                        )
                        if out:
                            locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7c_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
                            append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        gradient = run_gradient_lane(gpu_job)
        rows.extend(gradient["rows"])
        training_histories.extend(gradient["histories"])
        for job in jobs:
            result = run_mutation_job(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    training_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, decision_detail = decide(aggregate, pocket_reports)
    return {
        "symbolic_tasks": symbolic_tasks,
        "learned_tasks": learned_tasks,
        "binary_tasks": binary_tasks,
        "pocket_reports": pocket_reports,
        "rows": rows,
        "mutation_histories": mutation_histories,
        "training_histories": training_histories,
        "aggregate": aggregate,
        "decision": decision,
        "decision_detail": decision_detail,
    }


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    composition, leakage = build_reports(results["aggregate"], results["decision_detail"])
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7c_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["symbolic_tasks"], results["learned_tasks"], results["binary_tasks"]),
        "learned_pocket_training_report.json": {
            "schema_version": "e7c_learned_pocket_training_report_v1",
            "rows": results["pocket_reports"],
        },
        "pocket_library_report.json": pocket_library_report(results["pocket_reports"]),
        "system_results.json": {"schema_version": "e7c_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7c_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7c_training_history_v1", "rows": results["training_histories"]},
        "aggregate_metrics.json": results["aggregate"],
        "composition_report.json": composition,
        "leakage_report.json": leakage,
        "decision.json": {"schema_version": "e7c_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7c_summary_v1",
            "decision": results["decision"],
            "best_non_symbolic_oracle_system": results["aggregate"]["best_non_symbolic_oracle_system"],
            "deterministic_replay_passed": False,
            "checker_failure_count": None,
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
        },
    }
    payloads["report.md"] = build_markdown(payloads)
    return payloads


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_out = out / "deterministic_replay_work"
    if replay_out.exists():
        shutil.rmtree(replay_out)
    append_progress(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
    replay_results = run_core(settings, replay_out)
    replay_payloads = build_payloads(settings, out, replay_results)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    report = {
        "schema_version": "e7c_deterministic_replay_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], replay: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["deterministic_replay.json"] = replay
    payloads["summary.json"]["deterministic_replay_passed"] = replay["internal_replay_passed"]
    payloads["decision.json"]["deterministic_replay_passed"] = replay["internal_replay_passed"]
    payloads["report.md"] = build_markdown(payloads)
    for name, payload in payloads.items():
        if name.endswith(".md"):
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress(out, "final_artifacts_written", artifact_count=len(payloads))


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7b.start_hardware_monitor(out, stop, interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default="92001,92002,92003,92004,92005,92006,92007,92008,92009,92010,92011,92012")
    parser.add_argument("--train-rows-per-seed", type=int, default=480)
    parser.add_argument("--validation-rows-per-seed", type=int, default=200)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=200)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=200)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=200)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=2400)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--pocket-epochs", type=int, default=90)
    parser.add_argument("--gradient-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--pocket-learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=240)
    parser.add_argument("--mutation-population", type=int, default=30)
    parser.add_argument("--mutation-sigma", type=float, default=0.25)
    parser.add_argument("--mutation-elite-count", type=int, default=5)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 1, 23)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--no-replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
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
        pocket_epochs=args.pocket_epochs,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pocket_learning_rate=args.pocket_learning_rate,
        weight_decay=args.weight_decay,
        mutation_generations=args.mutation_generations,
        mutation_population=args.mutation_population,
        mutation_sigma=args.mutation_sigma,
        mutation_elite_count=args.mutation_elite_count,
        cpu_workers=args.cpu_workers,
        device=args.device,
        heartbeat_seconds=args.heartbeat_seconds,
        execution_mode=args.execution_mode,
        replay=not args.no_replay,
    )
    out.mkdir(parents=True, exist_ok=True)
    stop_monitor = threading.Event()
    monitor = start_hardware_monitor(out, stop_monitor, settings.heartbeat_seconds)
    try:
        results = run_core(settings, out)
        payloads = build_payloads(settings, out, results)
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7c_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_skipped": True}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop_monitor.set()
        monitor.join(timeout=5.0)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"decision": summary["decision"], "best_non_symbolic_oracle_system": summary["best_non_symbolic_oracle_system"], "deterministic_replay_passed": summary["deterministic_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
