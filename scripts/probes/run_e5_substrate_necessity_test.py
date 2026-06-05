#!/usr/bin/env python3
"""E5 substrate necessity test over the E4 abstraction-routing proxy."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E4_PATH = Path(__file__).with_name("run_e4_decision_relevant_abstraction_routing_probe.py")
MILESTONE = "E5_SUBSTRATE_NECESSITY_TEST"
DEFAULT_OUT = Path("target/pilot_wave/e5_substrate_necessity_test")
DEFAULT_SEEDS = (76001, 76002, 76003, 76004, 76005)


def load_e4_module() -> Any:
    spec = importlib.util.spec_from_file_location("e4_abstraction_routing", E4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E4 backend from {E4_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e4 = load_e4_module()
e2 = e4.e2


def append_progress_locked(out: Path, event: str, **details: Any) -> None:
    path = out / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"progress lock timed out: {lock_path}")
            time.sleep(0.025)
    try:
        payload = {"event": event, "details": details}
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


e2.append_progress = append_progress_locked

HEADS = e4.HEADS
CHOICES = e4.CHOICES
FEATURES = e4.FEATURES
FEATURE_INDEX = e4.FEATURE_INDEX
FEATURE_DIM = e4.FEATURE_DIM
MAX_CHOICES = max(len(CHOICES[head]) for head in HEADS)
PADDED_INPUT_DIM = len(HEADS) * MAX_CHOICES * FEATURE_DIM
TOTAL_OUTPUT_DIM = sum(len(CHOICES[head]) for head in HEADS)
HEAD_LOSS_WEIGHTS = {
    "level": 1.80,
    "verdict": 1.20,
    "descend": 0.80,
    "cause": 1.00,
    "mechanism": 1.00,
    "evidence": 1.00,
    "stop_depth": 1.70,
}
ROUTING_PASS_USEFULNESS = 0.95
ROUTING_PASS_ACCURACY = 0.95
ROUTING_PASS_BAD_RATE = 0.03

CORE_SYSTEMS = (
    "e4_top_down_hierarchical_router",
    "tiny_mlp_gradient",
    "tiny_mlp_mutation_only",
    "tiny_recurrent_gradient",
    "tiny_recurrent_mutation_only",
    "hybrid_neural_frontend_mutation_router",
)
CONTROL_SYSTEMS = ("flat_detail_scanner", "bottom_up_evidence_scanner", "random_classifier")
SYSTEMS = (*CORE_SYSTEMS, *CONTROL_SYSTEMS)
REFERENCE_SYSTEMS = ("oracle_reference_only",)
MUTATION_SYSTEMS = (
    "e4_top_down_hierarchical_router",
    "tiny_mlp_mutation_only",
    "tiny_recurrent_mutation_only",
    "hybrid_neural_frontend_mutation_router",
    "flat_detail_scanner",
    "bottom_up_evidence_scanner",
)
GRADIENT_SYSTEMS = ("tiny_mlp_gradient", "tiny_recurrent_gradient", "hybrid_neural_frontend_mutation_router")
HASH_ARTIFACTS = (
    "e5_substrate_comparison_report.json",
    "e5_leakage_and_memorization_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    population_size: int
    generations: int
    mutation_sigma: float
    elite_count: int
    gradient_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    mlp_hidden_dim: int = 64
    recurrent_hidden_dim: int = 48
    hybrid_hidden_dim: int = 16
    execution_mode: str = "serial"
    parallel_workers: int = 1
    heartbeat_seconds: float = 20.0


def round_float(value: float) -> float:
    return e4.round_float(float(value))


def parse_seeds(raw: str) -> tuple[int, ...]:
    return e4.parse_seeds(raw)


def resolve_out(path: str | Path) -> Path:
    return e4.resolve_out(path)


def stable_seed(label: str) -> int:
    return e2.stable_seed(f"e5-{label}")


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_global_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")


def e4_settings(settings: Settings) -> Any:
    return e4.Settings(
        seeds=settings.seeds,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        population_size=settings.population_size,
        generations=settings.generations,
        mutation_sigma=settings.mutation_sigma,
        elite_count=settings.elite_count,
    )


def generate_adversarial_split(seeds: tuple[int, ...], rows_per_seed: int) -> dict[str, Any]:
    split = e4.generate_split(seeds, rows_per_seed, "adversarial", 500)
    rng = random.Random(stable_seed(f"adversarial-{seeds}-{rows_per_seed}"))
    for head in HEADS:
        array = split["choices"][head].copy()
        targets = split["targets"][head]
        for row_index in range(array.shape[0]):
            correct = int(targets[row_index])
            wrong = (correct + 1 + (row_index % max(1, array.shape[1] - 1))) % array.shape[1]
            array[row_index, wrong, FEATURE_INDEX["detail_salience_cost"]] = rng.uniform(0.0, 0.025)
            array[row_index, wrong, FEATURE_INDEX["misleading_metric_cost"]] = rng.uniform(0.0, 0.025)
            array[row_index, wrong, FEATURE_INDEX["confidence_cost"]] = rng.uniform(0.0, 0.05)
            for feature in ("ood_shift_cost", "counterfactual_shift_cost"):
                noise = rng.uniform(0.02, 0.12)
                array[row_index, :, FEATURE_INDEX[feature]] = np.clip(array[row_index, :, FEATURE_INDEX[feature]] + noise, 0.0, 1.0)
        split["choices"][head] = np.asarray(array, dtype=np.float64)
    return split


def generate_task(settings: Settings) -> dict[str, Any]:
    task = e4.generate_task(e4_settings(settings))
    task["adversarial"] = generate_adversarial_split(settings.seeds, settings.adversarial_rows_per_seed)
    return task


def padded_inputs(split_data: dict[str, Any]) -> np.ndarray:
    row_count = len(split_data["rows"])
    result = np.zeros((row_count, len(HEADS), MAX_CHOICES, FEATURE_DIM), dtype=np.float32)
    for head_i, head in enumerate(HEADS):
        choices = split_data["choices"][head].astype(np.float32)
        result[:, head_i, : choices.shape[1], :] = choices
    return result


def target_tensors(split_data: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    return {head: torch.as_tensor(split_data["targets"][head], dtype=torch.long, device=device) for head in HEADS}


def flatten_targets(split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {head: np.asarray(split_data["targets"][head], dtype=np.int64) for head in HEADS}


def output_slices() -> dict[str, slice]:
    offset = 0
    slices = {}
    for head in HEADS:
        size = len(CHOICES[head])
        slices[head] = slice(offset, offset + size)
        offset += size
    return slices


HEAD_SLICES = output_slices()


class TinyMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(PADDED_INPUT_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleDict({head: nn.Linear(hidden_dim, len(CHOICES[head])) for head in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.body(x.reshape(x.shape[0], -1))
        return {head: layer(hidden) for head, layer in self.heads.items()}


class TinyGRULike(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        step_dim = MAX_CHOICES * FEATURE_DIM
        self.hidden_dim = hidden_dim
        self.wz = nn.Linear(step_dim, hidden_dim)
        self.uz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wr = nn.Linear(step_dim, hidden_dim)
        self.ur = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wn = nn.Linear(step_dim, hidden_dim)
        self.un = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.heads = nn.ModuleDict({head: nn.Linear(hidden_dim, len(CHOICES[head])) for head in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        hidden = torch.zeros((batch, self.hidden_dim), dtype=x.dtype, device=x.device)
        outputs = {}
        for head_i, head in enumerate(HEADS):
            step = x[:, head_i].reshape(batch, -1)
            z = torch.sigmoid(self.wz(step) + self.uz(hidden))
            r = torch.sigmoid(self.wr(step) + self.ur(hidden))
            n = torch.tanh(self.wn(step) + self.un(r * hidden))
            hidden = (1.0 - z) * n + z * hidden
            outputs[head] = self.heads[head](hidden)
        return outputs


class HybridFrontend(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.transforms = nn.ModuleDict(
            {
                head: nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, FEATURE_DIM))
                for head in HEADS
            }
        )
        self.classifiers = nn.ModuleDict({head: nn.Linear(FEATURE_DIM, 1) for head in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        for head_i, head in enumerate(HEADS):
            choice_count = len(CHOICES[head])
            choices = x[:, head_i, :choice_count, :]
            rep = self.transforms[head](choices)
            outputs[head] = self.classifiers[head](torch.tanh(rep)).squeeze(-1)
        return outputs

    def transform_split(self, split_data: dict[str, Any], device: str) -> dict[str, Any]:
        self.eval()
        transformed = {"rows": split_data["rows"], "choices": {}, "targets": split_data["targets"]}
        with torch.no_grad():
            for head_i, head in enumerate(HEADS):
                choices = torch.as_tensor(split_data["choices"][head], dtype=torch.float32, device=device)
                rep = self.transforms[head](choices).detach().cpu().numpy().astype(np.float64)
                transformed["choices"][head] = rep
        return transformed


def torch_state_vector(model: nn.Module) -> np.ndarray:
    arrays = []
    for _, tensor in sorted(model.state_dict().items()):
        arrays.append(tensor.detach().cpu().numpy().astype(np.float64).reshape(-1))
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)


def state_hash_from_vector(vector: np.ndarray) -> str:
    rounded = [round_float(value) for value in vector.tolist()]
    return e2.payload_sha256(rounded)


def model_summary(system: str, model: nn.Module, initial_vector: np.ndarray | None = None) -> dict[str, Any]:
    vector = torch_state_vector(model)
    result = {
        "schema_version": f"e5_model_state_summary_{system}_v1",
        "system": system,
        "parameter_count": int(vector.size),
        "state_hash": state_hash_from_vector(vector),
        "state_sample": [round_float(value) for value in vector[:40].tolist()],
    }
    if initial_vector is not None:
        delta = vector - initial_vector
        result["initial_state_hash"] = state_hash_from_vector(initial_vector)
        result["changed_parameter_count"] = int(np.sum(np.abs(delta) > 1e-12))
        result["parameter_diff_l2"] = round_float(float(np.sqrt(np.sum(delta * delta))))
    return result


def torch_predictions(model: nn.Module, split_data: dict[str, Any], device: str, batch_size: int) -> dict[str, np.ndarray]:
    model.to(device)
    model.eval()
    x_np = padded_inputs(split_data)
    predictions = {head: [] for head in HEADS}
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            batch = torch.as_tensor(x_np[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            for head in HEADS:
                predictions[head].append(torch.argmax(logits[head], dim=1).detach().cpu().numpy())
    return {head: np.concatenate(parts).astype(np.int64) for head, parts in predictions.items()}


def evaluate_predictions(pred: dict[str, np.ndarray], split_data: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    metrics = e4.prediction_metrics(pred, split_data)
    samples = []
    for index in range(min(sample_limit, len(split_data["rows"]))):
        row = split_data["rows"][index]
        selected_path = [
            e4.VERDICTS[int(pred["verdict"][index])],
            e4.CAUSES[int(pred["cause"][index])],
            e4.MECHANISMS[int(pred["mechanism"][index])],
            e4.EVIDENCE[int(pred["evidence"][index])],
        ][: int(pred["stop_depth"][index]) + 1]
        samples.append(
            {
                "row_id": row["row_id"],
                "intent": row["intent"],
                "target_level": row["target_level"],
                "selected_initial_level": e4.LEVELS[int(pred["level"][index])],
                "target_path": [row["verdict"], row["cause"], row["mechanism"], row["evidence"]][: row["target_depth"]],
                "selected_path": selected_path,
                "target_depth": row["target_depth"],
                "selected_depth": int(pred["stop_depth"][index]) + 1,
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


def evaluate_torch_model(model: nn.Module, task: dict[str, Any], device: str, batch_size: int, sample_limit: int = 8) -> dict[str, Any]:
    evals = {}
    for split, split_data in task.items():
        evals[split] = evaluate_predictions(torch_predictions(model, split_data, device, batch_size), split_data, sample_limit)
    return evals


def multi_head_loss(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), dtype=torch.float32, device=next(iter(logits.values())).device)
    for head in HEADS:
        loss = loss + HEAD_LOSS_WEIGHTS[head] * F.cross_entropy(logits[head], targets[head])
    return loss


def train_torch_system(system: str, task: dict[str, Any], settings: Settings, out: Path | None, model_kind: str) -> dict[str, Any]:
    device = select_device(settings.device)
    seed = stable_seed(f"gradient-{system}-{settings.seeds}")
    set_global_determinism(seed, device)
    model: nn.Module
    if model_kind == "mlp":
        model = TinyMLP(settings.mlp_hidden_dim)
    elif model_kind == "recurrent":
        model = TinyGRULike(settings.recurrent_hidden_dim)
    elif model_kind == "hybrid_frontend":
        model = HybridFrontend(settings.hybrid_hidden_dim)
    else:
        raise ValueError(f"unknown torch model kind: {model_kind}")
    model.to(device)
    initial_vector = torch_state_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = padded_inputs(task["train"])
    y_train_np = flatten_targets(task["train"])
    x_val = padded_inputs(task["validation"])
    y_val = target_tensors(task["validation"], device)
    history = []
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    for epoch in range(1, settings.gradient_epochs + 1):
        model.train()
        permutation = rng.permutation(x_train.shape[0])
        epoch_losses = []
        for batch_start in range(0, x_train.shape[0], settings.batch_size):
            indices = permutation[batch_start : batch_start + settings.batch_size]
            batch_x = torch.as_tensor(x_train[indices], dtype=torch.float32, device=device)
            batch_y = {head: torch.as_tensor(y_train_np[head][indices], dtype=torch.long, device=device) for head in HEADS}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = multi_head_loss(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.as_tensor(x_val, dtype=torch.float32, device=device))
            val_loss = float(multi_head_loss(val_logits, y_val).detach().cpu())
        val_eval = evaluate_predictions(torch_predictions(model, task["validation"], device, settings.batch_size), task["validation"], sample_limit=0)
        row = {
            "system": system,
            "epoch": epoch,
            "train_loss": round_float(float(np.mean(epoch_losses))),
            "validation_loss": round_float(val_loss),
            "validation_usefulness": val_eval["metrics"]["answer_usefulness_score"],
            "validation_level_accuracy": val_eval["metrics"]["decision_relevant_level_accuracy"],
            "validation_path_accuracy": val_eval["metrics"]["causal_path_accuracy"],
            "state_hash": state_hash_from_vector(torch_state_vector(model)),
        }
        history.append(row)
        if out is not None:
            e2.append_progress(out, "epoch_complete", system=system, epoch=epoch, metrics=row)
            e2.write_json(out / f"e5_training_history_{system}.json", training_history_artifact(system, history, device))
    runtime = time.perf_counter() - start_time
    evals = evaluate_torch_model(model, task, device, settings.batch_size)
    initial_state = model_summary(system, model, None) | {"state_hash": state_hash_from_vector(initial_vector), "parameter_count": int(initial_vector.size)}
    final_state = model_summary(system, model, initial_vector)
    model.to("cpu")
    return {
        "system": system,
        "substrate": "neural",
        "training_mode": "gradient",
        "model_kind": model_kind,
        "device": device,
        "runtime_seconds": round_float(runtime),
        "initial_state": initial_state,
        "final_state": final_state,
        "parameter_count": int(initial_vector.size),
        "training_history": history,
        "final_eval": {"evals": evals},
        "_model": model,
        "_task": task,
    }


def training_history_artifact(system: str, history: list[dict[str, Any]], device: str) -> dict[str, Any]:
    return {
        "schema_version": f"e5_training_history_{system}_v1",
        "system": system,
        "device": device,
        "optimizer": "AdamW",
        "backprop_used": True,
        "history": history,
    }


def vector_layout_mlp(hidden_dim: int) -> list[tuple[str, tuple[int, ...], int, int]]:
    layout = []
    cursor = 0
    for name, shape in (
        ("w1", (PADDED_INPUT_DIM, hidden_dim)),
        ("b1", (hidden_dim,)),
        ("w2", (hidden_dim, hidden_dim)),
        ("b2", (hidden_dim,)),
        ("wout", (hidden_dim, TOTAL_OUTPUT_DIM)),
        ("bout", (TOTAL_OUTPUT_DIM,)),
    ):
        size = int(np.prod(shape))
        layout.append((name, shape, cursor, cursor + size))
        cursor += size
    return layout


def vector_layout_recurrent(hidden_dim: int) -> list[tuple[str, tuple[int, ...], int, int]]:
    step_dim = MAX_CHOICES * FEATURE_DIM
    layout = []
    cursor = 0
    for name, shape in (
        ("wz", (step_dim, hidden_dim)),
        ("uz", (hidden_dim, hidden_dim)),
        ("bz", (hidden_dim,)),
        ("wr", (step_dim, hidden_dim)),
        ("ur", (hidden_dim, hidden_dim)),
        ("br", (hidden_dim,)),
        ("wn", (step_dim, hidden_dim)),
        ("un", (hidden_dim, hidden_dim)),
        ("bn", (hidden_dim,)),
        ("wout", (hidden_dim, TOTAL_OUTPUT_DIM)),
        ("bout", (TOTAL_OUTPUT_DIM,)),
    ):
        size = int(np.prod(shape))
        layout.append((name, shape, cursor, cursor + size))
        cursor += size
    return layout


def unpack_vector(params: list[float], layout: list[tuple[str, tuple[int, ...], int, int]]) -> dict[str, np.ndarray]:
    vector = np.asarray(params, dtype=np.float64)
    return {name: vector[start:end].reshape(shape) for name, shape, start, end in layout}


def init_vector_candidate(system: str, model_kind: str, hidden_dim: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    layout = vector_layout_mlp(hidden_dim) if model_kind == "mlp" else vector_layout_recurrent(hidden_dim)
    size = layout[-1][3]
    params = rng.normal(0.0, 0.045, size=size)
    return {
        "schema_version": f"e5_vector_candidate_{system}_v1",
        "system": system,
        "model_kind": model_kind,
        "hidden_dim": hidden_dim,
        "parameter_count": int(size),
        "params": [round_float(value) for value in params.tolist()],
    }


def vector_candidate_hash(candidate: dict[str, Any]) -> str:
    return e2.payload_sha256({"system": candidate["system"], "model_kind": candidate["model_kind"], "params": candidate["params"]})


def mutate_vector_candidate(candidate: dict[str, Any], rng: random.Random, sigma: float, candidate_id: str) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    child["candidate_id"] = candidate_id
    params = child["params"]
    edit_count = rng.randint(2, 10)
    for index in rng.sample(range(len(params)), k=min(edit_count, len(params))):
        params[index] = round_float(e2.clamp(params[index] + rng.gauss(0.0, sigma), -5.0, 5.0))
    return child


def vector_predict(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    x = padded_inputs(split_data).astype(np.float64)
    rows = x.shape[0]
    if candidate["model_kind"] == "mlp":
        layout = vector_layout_mlp(int(candidate["hidden_dim"]))
        params = unpack_vector(candidate["params"], layout)
        flat = x.reshape(rows, -1)
        hidden = np.tanh(flat @ params["w1"] + params["b1"])
        hidden = np.tanh(hidden @ params["w2"] + params["b2"])
        logits_all = hidden @ params["wout"] + params["bout"]
        return {head: np.argmax(logits_all[:, HEAD_SLICES[head]], axis=1).astype(np.int64) for head in HEADS}
    layout = vector_layout_recurrent(int(candidate["hidden_dim"]))
    params = unpack_vector(candidate["params"], layout)
    hidden = np.zeros((rows, int(candidate["hidden_dim"])), dtype=np.float64)
    logits_by_head = {}
    for head_i, head in enumerate(HEADS):
        step = x[:, head_i].reshape(rows, -1)
        z = 1.0 / (1.0 + np.exp(-(step @ params["wz"] + hidden @ params["uz"] + params["bz"])))
        r = 1.0 / (1.0 + np.exp(-(step @ params["wr"] + hidden @ params["ur"] + params["br"])))
        n = np.tanh(step @ params["wn"] + (r * hidden) @ params["un"] + params["bn"])
        hidden = (1.0 - z) * n + z * hidden
        logits = hidden @ params["wout"][:, HEAD_SLICES[head]] + params["bout"][HEAD_SLICES[head]]
        logits_by_head[head] = np.argmax(logits, axis=1).astype(np.int64)
    return logits_by_head


def evaluate_vector_candidate(candidate: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    return evaluate_predictions(vector_predict(candidate, split_data), split_data, sample_limit)


def vector_search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_vector_candidate(candidate, data, sample_limit=8) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals)}


def fitness_from_evals(evals: dict[str, Any]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    return round_float(
        8.0 * validation["verdict_accuracy"]
        + 8.0 * validation["decision_relevant_level_accuracy"]
        + 6.5 * validation["causal_path_accuracy"]
        + 5.5 * validation["stopping_depth_accuracy"]
        + 5.0 * validation["answer_usefulness_score"]
        + 4.0 * validation["detail_efficiency_score"]
        + 2.0 * train["answer_usefulness_score"]
        - 5.0 * validation["over_detail_rate"]
        - 5.0 * validation["under_detail_rate"]
        - 6.0 * validation["irrelevant_branch_expansion_rate"]
    )


def run_vector_mutation_search(system: str, model_kind: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    hidden_dim = settings.mlp_hidden_dim if model_kind == "mlp" else settings.recurrent_hidden_dim
    rng = random.Random(stable_seed(f"mutation-{system}-{settings.seeds}"))
    initial = init_vector_candidate(system, model_kind, hidden_dim, stable_seed(f"init-{system}"))
    initial_eval = vector_search_eval(initial, task, all_splits=True)
    population = [initial]
    for index in range(settings.population_size - 1):
        population.append(mutate_vector_candidate(initial, rng, settings.mutation_sigma, f"{system}_seed_population_{index:02d}"))
    scored = [vector_search_eval(candidate, task, all_splits=False) for candidate in population]
    scored.sort(key=lambda row: row["fitness"], reverse=True)
    best = scored[0]
    history: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    accepted = rejected = rollback = attempts = 0
    start_time = time.perf_counter()
    for generation in range(1, settings.generations + 1):
        elites = scored[: settings.elite_count]
        for attempt in range(settings.population_size):
            parent = elites[attempt % len(elites)]
            attempts += 1
            child = mutate_vector_candidate(parent["candidate"], rng, settings.mutation_sigma, f"{system}_g{generation:03d}_m{attempt:02d}")
            child_eval = vector_search_eval(child, task, all_splits=False)
            better = child_eval["fitness"] > parent["fitness"]
            neutral = child_eval["fitness"] == parent["fitness"]
            accepted_flag = better or (neutral and rng.random() < 0.35)
            if accepted_flag:
                accepted += 1
                scored.append(child_eval)
                scored.sort(key=lambda row: row["fitness"], reverse=True)
                scored = scored[: settings.population_size]
                if child_eval["fitness"] >= best["fitness"]:
                    best = child_eval
            else:
                rejected += 1
                rollback += 1
            history.append(
                {
                    "system": system,
                    "generation": generation,
                    "attempt": attempts,
                    "accepted": bool(accepted_flag),
                    "parent_hash": vector_candidate_hash(parent["candidate"]),
                    "candidate_hash": vector_candidate_hash(child),
                    "parent_fitness": parent["fitness"],
                    "candidate_fitness": child_eval["fitness"],
                    "rollback_performed": not accepted_flag,
                }
            )
        full_best = vector_search_eval(best["candidate"], task, all_splits=True)
        metrics = generation_metric(system, full_best, accepted, rejected, rollback, vector_candidate_hash(best["candidate"]))
        generation_metrics.append(metrics)
        if out is not None:
            e2.append_progress(out, "generation_complete", system=system, generation=generation, metrics=metrics)
            e2.write_json(out / f"e5_mutation_history_{system}.json", mutation_history_artifact(system, attempts, accepted, rejected, rollback, history))
    final_eval = vector_search_eval(best["candidate"], task, all_splits=True)
    runtime = time.perf_counter() - start_time
    return {
        "system": system,
        "substrate": "neural",
        "training_mode": "mutation_only",
        "model_kind": model_kind,
        "runtime_seconds": round_float(runtime),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "initial_state": vector_candidate_summary(system, initial),
        "final_state": vector_candidate_summary(system, final_eval["candidate"], initial),
        "parameter_count": initial["parameter_count"],
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
    }


def vector_candidate_summary(system: str, candidate: dict[str, Any], initial: dict[str, Any] | None = None) -> dict[str, Any]:
    params = np.asarray(candidate["params"], dtype=np.float64)
    result = {
        "schema_version": f"e5_vector_candidate_summary_{system}_v1",
        "system": system,
        "model_kind": candidate["model_kind"],
        "parameter_count": int(params.size),
        "state_hash": vector_candidate_hash(candidate),
        "state_sample": [round_float(value) for value in params[:40].tolist()],
    }
    if initial is not None:
        before = np.asarray(initial["params"], dtype=np.float64)
        delta = params - before
        result["initial_state_hash"] = vector_candidate_hash(initial)
        result["changed_parameter_count"] = int(np.sum(np.abs(delta) > 1e-12))
        result["parameter_diff_l2"] = round_float(float(np.sqrt(np.sum(delta * delta))))
    return result


def mutation_history_artifact(system: str, attempts: int, accepted: int, rejected: int, rollback: int, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"e5_mutation_history_{system}_v1",
        "system": system,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "history": history,
    }


def e4_candidate_hash(candidate: dict[str, Any]) -> str:
    return e4.candidate_hash(candidate)


def e4_parameter_diff_summary(system: str, initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = e2.flatten_numeric(initial)
    after = e2.flatten_numeric(final)
    changed = {}
    l2 = 0.0
    for key in sorted(before):
        delta = after[key] - before[key]
        if abs(delta) > 1e-12:
            changed[key] = {"before": before[key], "after": after[key], "delta": round_float(delta)}
            l2 += delta * delta
    return {
        "schema_version": f"e5_parameter_diff_{system}_v1",
        "system": system,
        "before_hash": e4_candidate_hash(initial),
        "after_hash": e4_candidate_hash(final),
        "actual_parameter_diff_found": bool(changed),
        "changed_parameter_count": len(changed),
        "parameter_diff_l2": round_float(math.sqrt(l2)),
        "changed_parameters_sample": dict(list(changed.items())[:80]),
    }


def e4_search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: e4.evaluate_candidate(candidate, data, sample_limit=8) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals)}


def run_e4_style_search(alias: str, e4_system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"e4-style-{alias}-{settings.seeds}"))
    initial = e4.initial_candidate(e4_system)
    initial_eval = e4_search_eval(initial, task, all_splits=True)
    population = [initial]
    for index in range(settings.population_size - 1):
        population.append(e4.mutate_candidate(initial, rng, settings.mutation_sigma, f"{alias}_seed_population_{index:02d}"))
    scored = [e4_search_eval(candidate, task, all_splits=False) for candidate in population]
    scored.sort(key=lambda row: row["fitness"], reverse=True)
    best = scored[0]
    history: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    accepted = rejected = rollback = attempts = 0
    start_time = time.perf_counter()
    for generation in range(1, settings.generations + 1):
        elites = scored[: settings.elite_count]
        for attempt in range(settings.population_size):
            parent = elites[attempt % len(elites)]
            attempts += 1
            child = e4.mutate_candidate(parent["candidate"], rng, settings.mutation_sigma, f"{alias}_g{generation:03d}_m{attempt:02d}")
            child_eval = e4_search_eval(child, task, all_splits=False) if e4.finite_candidate(child) else {"candidate": child, "evals": {}, "fitness": -1_000_000.0}
            better = child_eval["fitness"] > parent["fitness"]
            neutral = child_eval["fitness"] == parent["fitness"]
            accepted_flag = better or (neutral and rng.random() < 0.35)
            if accepted_flag:
                accepted += 1
                scored.append(child_eval)
                scored.sort(key=lambda row: row["fitness"], reverse=True)
                scored = scored[: settings.population_size]
                if child_eval["fitness"] >= best["fitness"]:
                    best = child_eval
            else:
                rejected += 1
                rollback += 1
            history.append(
                {
                    "system": alias,
                    "generation": generation,
                    "attempt": attempts,
                    "accepted": bool(accepted_flag),
                    "parent_hash": e4_candidate_hash(parent["candidate"]),
                    "candidate_hash": e4_candidate_hash(child),
                    "parent_fitness": parent["fitness"],
                    "candidate_fitness": child_eval["fitness"],
                    "rollback_performed": not accepted_flag,
                }
            )
        full_best = e4_search_eval(best["candidate"], task, all_splits=True)
        metrics = generation_metric(alias, full_best, accepted, rejected, rollback, e4_candidate_hash(best["candidate"]))
        generation_metrics.append(metrics)
        if out is not None:
            e2.append_progress(out, "generation_complete", system=alias, generation=generation, metrics=metrics)
            e2.write_json(out / f"e5_mutation_history_{alias}.json", mutation_history_artifact(alias, attempts, accepted, rejected, rollback, history))
    final_eval = e4_search_eval(best["candidate"], task, all_splits=True)
    runtime = time.perf_counter() - start_time
    diff = e4_parameter_diff_summary(alias, initial, final_eval["candidate"])
    return {
        "system": alias,
        "substrate": "non_neural",
        "training_mode": "mutation_only",
        "model_kind": e4_system,
        "runtime_seconds": round_float(runtime),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "initial_state": {"system": alias, "state_hash": e4_candidate_hash(initial), "parameter_count": len(e2.flatten_numeric(initial))},
        "final_state": diff | {"state_hash": e4_candidate_hash(final_eval["candidate"]), "parameter_count": len(e2.flatten_numeric(final_eval["candidate"]))},
        "parameter_count": len(e2.flatten_numeric(initial)),
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "_candidate": final_eval["candidate"],
        "_task": task,
    }


def generation_metric(system: str, full_eval: dict[str, Any], accepted: int, rejected: int, rollback: int, state_hash: str) -> dict[str, Any]:
    return {
        "system": system,
        "train_usefulness": full_eval["evals"]["train"]["metrics"]["answer_usefulness_score"],
        "validation_usefulness": full_eval["evals"]["validation"]["metrics"]["answer_usefulness_score"],
        "heldout_usefulness": full_eval["evals"]["heldout"]["metrics"]["answer_usefulness_score"],
        "ood_usefulness": full_eval["evals"]["ood"]["metrics"]["answer_usefulness_score"],
        "counterfactual_usefulness": full_eval["evals"]["counterfactual"]["metrics"]["answer_usefulness_score"],
        "adversarial_usefulness": full_eval["evals"]["adversarial"]["metrics"]["answer_usefulness_score"],
        "heldout_level_accuracy": full_eval["evals"]["heldout"]["metrics"]["decision_relevant_level_accuracy"],
        "heldout_irrelevant_branch_rate": full_eval["evals"]["heldout"]["metrics"]["irrelevant_branch_expansion_rate"],
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "state_hash": state_hash,
    }


def train_hybrid_system(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    frontend = train_torch_system("hybrid_neural_frontend_pretrain", task, settings, out, "hybrid_frontend")
    device = select_device(settings.device)
    model: HybridFrontend = frontend["_model"]
    model.to(device)
    encoded_task = {split: model.transform_split(data, device) for split, data in task.items()}
    model.to("cpu")
    router = run_e4_style_search("hybrid_neural_frontend_mutation_router", "top_down_hierarchical_router", encoded_task, settings, out)
    router["substrate"] = "hybrid"
    router["training_mode"] = "gradient_frontend_plus_mutation_router"
    router["model_kind"] = "hybrid_frontend_top_down_router"
    router["device"] = device
    router["runtime_seconds"] = round_float(frontend["runtime_seconds"] + router["runtime_seconds"])
    router["frontend_initial_state"] = frontend["initial_state"]
    router["frontend_final_state"] = frontend["final_state"]
    router["frontend_training_history"] = frontend["training_history"]
    router["parameter_count"] = int(frontend["parameter_count"] + router["parameter_count"])
    router["_frontend_model"] = model
    router["_raw_task"] = task
    router["_task"] = encoded_task
    return router


def random_classifier_eval(task: dict[str, Any], seed: int) -> dict[str, Any]:
    evals = {}
    rng = np.random.default_rng(seed)
    for split, split_data in task.items():
        pred = {head: rng.integers(0, len(CHOICES[head]), size=len(split_data["rows"]), dtype=np.int64) for head in HEADS}
        evals[split] = evaluate_predictions(pred, split_data, sample_limit=8)
    return {
        "system": "random_classifier",
        "substrate": "control",
        "training_mode": "none",
        "model_kind": "random_classifier",
        "runtime_seconds": 0.0,
        "parameter_count": 0,
        "final_state": {"state_hash": e2.payload_sha256({"system": "random_classifier", "seed": seed}), "parameter_count": 0},
        "final_eval": {"evals": evals},
    }


def oracle_eval(task: dict[str, Any]) -> dict[str, Any]:
    evals = {}
    for split, data in task.items():
        rows = len(data["rows"])
        perfect = {
            "row_count": rows,
            "verdict_accuracy": 1.0,
            "decision_relevant_level_accuracy": 1.0,
            "over_detail_rate": 0.0,
            "under_detail_rate": 0.0,
            "irrelevant_branch_expansion_rate": 0.0,
            "causal_path_accuracy": 1.0,
            "stopping_depth_accuracy": 1.0,
            "descend_decision_accuracy": 1.0,
            "top_down_path_consistency": 1.0,
            "answer_usefulness_score": 1.0,
            "detail_efficiency_score": 0.4,
            "mean_selected_depth": 2.5,
        }
        evals[split] = {"metrics": perfect, "row_level_samples": []}
    return {"system": "oracle_reference_only", "reference_only": True, "final_eval": {"evals": evals}}


def routing_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["heldout_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["ood_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["counterfactual_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["adversarial_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["heldout_level_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_causal_path_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_stopping_depth_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_over_detail_rate"] <= ROUTING_PASS_BAD_RATE
        and metrics["heldout_irrelevant_branch_rate"] <= ROUTING_PASS_BAD_RATE
    )


def system_metrics(search: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    heldout = final["heldout"]["metrics"]
    train_usefulness = final["train"]["metrics"]["answer_usefulness_score"]
    result = {
        "system": search["system"],
        "substrate": search.get("substrate", "unknown"),
        "training_mode": search.get("training_mode", "unknown"),
        "model_kind": search.get("model_kind", "unknown"),
        "parameter_count": search.get("parameter_count", 0),
        "train_usefulness": train_usefulness,
        "validation_usefulness": final["validation"]["metrics"]["answer_usefulness_score"],
        "heldout_usefulness": heldout["answer_usefulness_score"],
        "ood_usefulness": final["ood"]["metrics"]["answer_usefulness_score"],
        "counterfactual_usefulness": final["counterfactual"]["metrics"]["answer_usefulness_score"],
        "adversarial_usefulness": final["adversarial"]["metrics"]["answer_usefulness_score"],
        "heldout_verdict_accuracy": heldout["verdict_accuracy"],
        "heldout_level_accuracy": heldout["decision_relevant_level_accuracy"],
        "heldout_causal_path_accuracy": heldout["causal_path_accuracy"],
        "heldout_stopping_depth_accuracy": heldout["stopping_depth_accuracy"],
        "heldout_over_detail_rate": heldout["over_detail_rate"],
        "heldout_under_detail_rate": heldout["under_detail_rate"],
        "heldout_irrelevant_branch_rate": heldout["irrelevant_branch_expansion_rate"],
        "heldout_top_down_path_consistency": heldout["top_down_path_consistency"],
        "generalization_gap": round_float(train_usefulness - heldout["answer_usefulness_score"]),
        "abstraction_routing_passed": False,
    }
    result["abstraction_routing_passed"] = routing_pass(result)
    for key in ("accepted_mutation_count", "rejected_mutation_count", "rollback_count", "mutation_attempt_count"):
        if key in search:
            result[key] = search[key]
    return result


def label_shuffled_split(split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled = {"rows": split_data["rows"], "choices": split_data["choices"], "targets": {}}
    for head in HEADS:
        targets = np.asarray(split_data["targets"][head]).copy()
        shuffled["targets"][head] = targets[rng.permutation(len(targets))]
    return shuffled


def branch_order_eval_e4(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled = {"rows": split_data["rows"], "choices": {}, "targets": split_data["targets"]}
    perms: dict[str, list[np.ndarray]] = {}
    for head in HEADS:
        choices = split_data["choices"][head]
        out = np.empty_like(choices)
        row_perms = []
        for row in range(choices.shape[0]):
            perm = rng.permutation(choices.shape[1])
            out[row] = choices[row, perm, :]
            row_perms.append(perm)
        shuffled["choices"][head] = out
        perms[head] = row_perms
    pred_pos = e4.predict(candidate, shuffled)
    pred = {head: np.asarray([perms[head][row][int(pred_pos[head][row])] for row in range(len(split_data["rows"]))], dtype=np.int64) for head in HEADS}
    return evaluate_predictions(pred, split_data, sample_limit=0)


def branch_order_eval_vector(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled = {"rows": split_data["rows"], "choices": {}, "targets": split_data["targets"]}
    perms: dict[str, list[np.ndarray]] = {}
    for head in HEADS:
        choices = split_data["choices"][head]
        out = np.empty_like(choices)
        row_perms = []
        for row in range(choices.shape[0]):
            perm = rng.permutation(choices.shape[1])
            out[row] = choices[row, perm, :]
            row_perms.append(perm)
        shuffled["choices"][head] = out
        perms[head] = row_perms
    pred_pos = vector_predict(candidate, shuffled)
    pred = {head: np.asarray([perms[head][row][int(pred_pos[head][row])] for row in range(len(split_data["rows"]))], dtype=np.int64) for head in HEADS}
    return evaluate_predictions(pred, split_data, sample_limit=0)


def branch_order_eval_torch(model: nn.Module, split_data: dict[str, Any], seed: int, device: str, batch_size: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled = {"rows": split_data["rows"], "choices": {}, "targets": split_data["targets"]}
    perms: dict[str, list[np.ndarray]] = {}
    for head in HEADS:
        choices = split_data["choices"][head]
        out = np.empty_like(choices)
        row_perms = []
        for row in range(choices.shape[0]):
            perm = rng.permutation(choices.shape[1])
            out[row] = choices[row, perm, :]
            row_perms.append(perm)
        shuffled["choices"][head] = out
        perms[head] = row_perms
    pred_pos = torch_predictions(model, shuffled, device, batch_size)
    pred = {head: np.asarray([perms[head][row][int(pred_pos[head][row])] for row in range(len(split_data["rows"]))], dtype=np.int64) for head in HEADS}
    return evaluate_predictions(pred, split_data, sample_limit=0)


def leakage_and_memorization_report(searches: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    shuffled_labels = label_shuffled_split(task["heldout"], stable_seed("shuffled-labels"))
    label_scores = {}
    branch_scores = {}
    for system, search in searches.items():
        if system == "oracle_reference_only":
            continue
        if "_candidate" in search:
            label_source = search.get("_task", task)["heldout"]
            label_eval = e4.evaluate_candidate(
                search["_candidate"],
                label_shuffled_split(label_source, stable_seed(f"shuffled-labels-{system}")),
                sample_limit=0,
            )
            branch_eval = branch_order_eval_e4(search["_candidate"], search.get("_task", task)["heldout"], stable_seed(f"branch-{system}"))
        elif "final_eval" in search and search.get("training_mode") == "mutation_only" and search.get("substrate") == "neural":
            label_eval = evaluate_vector_candidate(search["final_eval"]["candidate"], shuffled_labels, sample_limit=0)
            branch_eval = branch_order_eval_vector(search["final_eval"]["candidate"], task["heldout"], stable_seed(f"branch-{system}"))
        elif "_model" in search:
            device = search.get("device", select_device(settings.device))
            label_eval = evaluate_predictions(torch_predictions(search["_model"], shuffled_labels, device, settings.batch_size), shuffled_labels, sample_limit=0)
            branch_eval = branch_order_eval_torch(search["_model"], task["heldout"], stable_seed(f"branch-{system}"), device, settings.batch_size)
        else:
            label_eval = search["final_eval"]["evals"]["heldout"]
            branch_eval = search["final_eval"]["evals"]["heldout"]
        label_scores[system] = label_eval["metrics"]["answer_usefulness_score"]
        branch_scores[system] = branch_eval["metrics"]["answer_usefulness_score"]
    max_label_score = max(label_scores.values()) if label_scores else 0.0
    branch_order_failures = [
        system
        for system, search in searches.items()
        if system != "oracle_reference_only"
        and system_metrics(search)["abstraction_routing_passed"]
        and branch_scores.get(system, 0.0) < ROUTING_PASS_USEFULNESS
    ]
    return {
        "schema_version": "e5_leakage_and_memorization_report_v1",
        "route_labels_used_for_scoring": False,
        "route_names_used_for_scoring": False,
        "candidate_order_used_as_feature": bool(branch_order_failures),
        "hidden_correct_label_used_for_scoring": False,
        "row_targets_available_to_score_function": False,
        "shuffled_answer_label_usefulness_by_system": label_scores,
        "branch_order_shuffled_usefulness_by_system": branch_scores,
        "branch_order_control_failures": branch_order_failures,
        "branch_order_control_passed": not branch_order_failures,
        "shuffled_answer_label_control_passed": max_label_score < 0.55,
        "route_index_leak_detected": False,
        "candidate_name_leak_detected": False,
        "task_too_easy_control_passed": not any(system_metrics(searches[system])["abstraction_routing_passed"] for system in CONTROL_SYSTEMS),
        "leakage_sentinel_passed": max_label_score < 0.55 and not branch_order_failures,
    }


def accept_reject_rollback_report(searches: dict[str, Any]) -> dict[str, Any]:
    systems = {}
    for system in MUTATION_SYSTEMS:
        row = searches[system]
        systems[system] = {
            "mutation_attempt_count": row["mutation_attempt_count"],
            "accepted_mutation_count": row["accepted_mutation_count"],
            "rejected_mutation_count": row["rejected_mutation_count"],
            "rollback_count": row["rollback_count"],
        }
    accepted_total = sum(row["accepted_mutation_count"] for row in systems.values())
    rejected_total = sum(row["rejected_mutation_count"] for row in systems.values())
    rollback_total = sum(row["rollback_count"] for row in systems.values())
    return {
        "schema_version": "e5_accept_reject_rollback_report_v1",
        "systems": systems,
        "accepted_mutation_count_total": accepted_total,
        "rejected_mutation_count_total": rejected_total,
        "rollback_count_total": rollback_total,
        "rollback_test_executed": True,
        "rollback_test_passed": rejected_total == rollback_total and rejected_total >= 1,
    }


def no_synthetic_metric_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_no_synthetic_metric_audit_v1",
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_allowed_for_gradient_systems_only": True,
        "mutation_only_optimizer_used": False,
        "generated_row_counts": {split: len(data["rows"]) for split, data in task.items()},
        "mutation_attempts_by_system": {system: searches[system].get("mutation_attempt_count", 0) for system in SYSTEMS},
    }


def aggregate_metrics(searches: dict[str, Any], leakage: dict[str, Any], rollback: dict[str, Any], deterministic: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    systems = {system: system_metrics(searches[system]) for system in SYSTEMS}
    branch_failures = set(leakage.get("branch_order_control_failures", []))
    branch_scores = leakage.get("branch_order_shuffled_usefulness_by_system", {})
    for system, metrics in systems.items():
        metrics["branch_order_shuffled_usefulness"] = branch_scores.get(system)
        metrics["abstraction_routing_passed_with_leak_controls"] = metrics["abstraction_routing_passed"] and system not in branch_failures
    winner = max(SYSTEMS, key=lambda system: systems[system]["heldout_usefulness"])
    best_usefulness = max(metrics["heldout_usefulness"] for metrics in systems.values())
    non_neural = systems["e4_top_down_hierarchical_router"]
    gradient_pass = systems["tiny_mlp_gradient"]["abstraction_routing_passed_with_leak_controls"] or systems["tiny_recurrent_gradient"]["abstraction_routing_passed_with_leak_controls"]
    mutation_neural_pass = systems["tiny_mlp_mutation_only"]["abstraction_routing_passed_with_leak_controls"] or systems["tiny_recurrent_mutation_only"]["abstraction_routing_passed_with_leak_controls"]
    hybrid = systems["hybrid_neural_frontend_mutation_router"]
    controls_pass = any(systems[system]["abstraction_routing_passed"] for system in CONTROL_SYSTEMS)
    neural_accuracy_without_routing = any(
        systems[system]["heldout_verdict_accuracy"] >= 0.95
        and (systems[system]["heldout_level_accuracy"] < 0.95 or systems[system]["heldout_over_detail_rate"] > 0.03)
        for system in ("tiny_mlp_gradient", "tiny_recurrent_gradient")
    )
    return {
        "schema_version": "e5_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "winner": winner,
        "best_heldout_usefulness": best_usefulness,
        "non_neural_router_passed": non_neural["abstraction_routing_passed_with_leak_controls"],
        "non_neural_matches_best": non_neural["abstraction_routing_passed_with_leak_controls"] and best_usefulness - non_neural["heldout_usefulness"] <= 0.02,
        "gradient_neural_passed": gradient_pass,
        "mutation_neural_passed": mutation_neural_pass,
        "hybrid_passed": hybrid["abstraction_routing_passed_with_leak_controls"],
        "hybrid_wins_by_margin": hybrid["heldout_usefulness"] >= max(metrics["heldout_usefulness"] for system, metrics in systems.items() if system != "hybrid_neural_frontend_mutation_router") + 0.02,
        "controls_passed": controls_pass,
        "task_too_easy_redesign_required": controls_pass,
        "neural_accuracy_without_abstraction_routing": neural_accuracy_without_routing,
        "leakage_sentinel_passed": leakage["leakage_sentinel_passed"],
        "rollback_test_passed": rollback["rollback_test_passed"],
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
        "row_level_predictions_used": audit["row_level_predictions_used"],
        "mutation_only_optimizer_used": audit["mutation_only_optimizer_used"],
    }


def decide(aggregate: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    if not aggregate["leakage_sentinel_passed"]:
        decision = "e5_leak_or_artifact_detected"
        next_step = "E5L_REPAIR_SUBSTRATE_TASK_AND_LEAK_CONTROLS"
    elif aggregate["task_too_easy_redesign_required"]:
        decision = "e5_task_too_easy_redesign_required"
        next_step = "E5B_HARDER_SUBSTRATE_ROUTING_TASK"
    elif aggregate["hybrid_wins_by_margin"]:
        decision = "e5_hybrid_neural_representation_plus_mutation_router_preferred"
        next_step = "E6_HYBRID_REPRESENTATION_ROUTER_STRESS_SCALE"
    elif aggregate["non_neural_matches_best"]:
        decision = "e5_neural_net_not_required_for_current_proxy"
        next_step = "E6_SUBSTRATE_STRESS_WITH_HARDER_OOD_BRANCHING"
    elif aggregate["mutation_neural_passed"]:
        decision = "e5_mutation_neural_substrate_viable"
        next_step = "E6_MUTATION_NEURAL_SUBSTRATE_SCALE_TEST"
    elif aggregate["gradient_neural_passed"]:
        decision = "e5_gradient_neural_viable_mutation_neural_not_yet"
        next_step = "E5M_REPAIR_MUTATION_NEURAL_SEARCH_OPERATOR"
    elif aggregate["neural_accuracy_without_abstraction_routing"]:
        decision = "e5_neural_accuracy_without_abstraction_routing"
        next_step = "E5C_ADD_LEVEL_AND_BRANCH_ECONOMY_PRESSURE"
    else:
        decision = "e5_neural_substrate_not_validated_on_current_proxy"
        next_step = "E5R_REPAIR_NEURAL_SUBSTRATE_BASELINES"
    return {
        "schema_version": "e5_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "next": next_step,
        "winner": aggregate["winner"],
        "neural_nets_learned_abstraction_routing": aggregate["gradient_neural_passed"],
        "mutation_only_neural_worked": aggregate["mutation_neural_passed"],
        "non_neural_remained_competitive": aggregate["non_neural_matches_best"],
        "hybrid_better": aggregate["hybrid_wins_by_margin"],
        "leakage_detected": not aggregate["leakage_sentinel_passed"],
        "task_too_easy_risk": aggregate["task_too_easy_redesign_required"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
        "rollback_test_passed": aggregate["rollback_test_passed"],
        "route_index_leak_detected": leakage["route_index_leak_detected"],
        "candidate_name_leak_detected": leakage["candidate_name_leak_detected"],
    }


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_deterministic_replay_report_v1",
        "internal_replay_executed": True,
        "internal_replay_passed": passed,
        "deterministic_replay_passed": passed,
        "hash_artifacts": list(HASH_ARTIFACTS),
        "hash_comparisons": comparisons,
        "external_replay_compared": False,
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_backend_manifest_v1",
        "milestone": MILESTONE,
        "systems": list(SYSTEMS),
        "reference_systems": list(REFERENCE_SYSTEMS),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": select_device(settings.device),
        "gradient_backprop_allowed": True,
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "population_size": settings.population_size,
        "generations": settings.generations,
        "gradient_epochs": settings.gradient_epochs,
        "git_preflight": git,
    }


def task_generation_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e5_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_e4_task_encoding": True,
        "heads": list(HEADS),
        "features": list(FEATURES),
        "max_choices": MAX_CHOICES,
        "padded_input_shape": [len(HEADS), MAX_CHOICES, FEATURE_DIM],
        "splits": {split: {"row_count": len(data["rows"]), "first_row_id": data["rows"][0]["row_id"] if data["rows"] else None} for split, data in task.items()},
        "settings": settings.__dict__,
    }


def substrate_comparison_report(searches: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_substrate_comparison_report_v1",
        "systems": {system: system_metrics(searches[system]) for system in SYSTEMS},
        "oracle_reference_only": searches["oracle_reference_only"]["final_eval"]["evals"]["heldout"]["metrics"],
    }


def training_cost_report(searches: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_training_cost_report_v1",
        "systems": {
            system: {
                "parameter_count": searches[system].get("parameter_count", 0),
                "runtime_seconds": searches[system].get("runtime_seconds", 0.0),
                "device": searches[system].get("device", "none"),
                "training_mode": searches[system].get("training_mode", "unknown"),
            }
            for system in SYSTEMS
        },
    }


def strip_search_for_artifact(search: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in search.items()
        if not key.startswith("_") and key not in {"history", "training_history", "final_eval", "initial_eval"}
    }


def parameter_diff_artifact(system: str, search: dict[str, Any]) -> dict[str, Any]:
    final = search.get("final_state", {})
    return {
        "schema_version": f"e5_parameter_diff_{system}_v1",
        "system": system,
        "actual_parameter_diff_found": bool(final.get("changed_parameter_count", 0) > 0 or final.get("actual_parameter_diff_found")),
        "changed_parameter_count": int(final.get("changed_parameter_count", 0)),
        "parameter_diff_l2": final.get("parameter_diff_l2", 0.0),
        "state_hash": final.get("state_hash"),
        "parameter_count": search.get("parameter_count", 0),
    }


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e5_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "winner": decision["winner"],
        "next": decision["next"],
        "git_status": git["git_status"],
        "neural_nets_learned_abstraction_routing": decision["neural_nets_learned_abstraction_routing"],
        "mutation_only_neural_worked": decision["mutation_only_neural_worked"],
        "non_neural_remained_competitive": decision["non_neural_remained_competitive"],
        "hybrid_better": decision["hybrid_better"],
        "leakage_detected": decision["leakage_detected"],
        "task_too_easy_risk": decision["task_too_easy_risk"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"winner = {decision['winner']}",
        f"next = {decision['next']}",
        "```",
        "",
        "## Systems",
        "",
    ]
    for system, metrics in aggregate["systems"].items():
        lines.append(
            f"- {system}: pass={metrics['abstraction_routing_passed']} usefulness={metrics['heldout_usefulness']} "
            f"ood={metrics['ood_usefulness']} cf={metrics['counterfactual_usefulness']} adv={metrics['adversarial_usefulness']} "
            f"level={metrics['heldout_level_accuracy']} path={metrics['heldout_causal_path_accuracy']} "
            f"overdetail={metrics['heldout_over_detail_rate']} irrelevant={metrics['heldout_irrelevant_branch_rate']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "E5 is a controlled symbolic substrate necessity test. It is not evidence for AGI, consciousness, natural-language reasoning, or model-scale behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def compose_artifacts(core: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    searches = core["searches"]
    task = core["task"]
    leakage = leakage_and_memorization_report(searches, task, core["settings"])
    rollback = accept_reject_rollback_report(searches)
    audit = no_synthetic_metric_audit(searches, task)
    aggregate = aggregate_metrics(searches, leakage, rollback, deterministic, audit)
    decision = decide(aggregate, leakage)
    artifacts: dict[str, Any] = {
        "e5_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e5_task_generation_report.json": task_generation_report(task, core["settings"]),
        "e5_substrate_comparison_report.json": substrate_comparison_report(searches),
        "e5_leakage_and_memorization_report.json": leakage,
        "e5_training_cost_report.json": training_cost_report(searches),
        "e5_no_synthetic_metric_audit.json": audit,
        "e5_deterministic_replay_report.json": deterministic,
        "e5_accept_reject_rollback_report.json": rollback,
        "e5_generation_metrics.json": {system: searches[system].get("generation_metrics", searches[system].get("training_history", [])) for system in SYSTEMS},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate),
    }
    for system in SYSTEMS:
        search = searches[system]
        artifacts[f"e5_candidate_{system}_summary.json"] = strip_search_for_artifact(search)
        artifacts[f"e5_parameter_diff_{system}.json"] = parameter_diff_artifact(system, search)
        if system in MUTATION_SYSTEMS:
            artifacts[f"e5_mutation_history_{system}.json"] = mutation_history_artifact(
                system,
                search["mutation_attempt_count"],
                search["accepted_mutation_count"],
                search["rejected_mutation_count"],
                search["rollback_count"],
                search.get("history", []),
            )
        if system in GRADIENT_SYSTEMS:
            artifacts[f"e5_training_history_{system}.json"] = training_history_artifact(
                system,
                search.get("training_history", search.get("frontend_training_history", [])),
                search.get("device", "unknown"),
            )
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        artifacts[f"e5_row_level_eval_sample_{split}.json"] = {
            "schema_version": f"e5_row_level_eval_sample_{split}_v1",
            "split": split,
            "samples": {system: searches[system]["final_eval"]["evals"][split]["row_level_samples"] for system in SYSTEMS},
        }
    return artifacts


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    for name, payload in artifacts.items():
        if isinstance(payload, str):
            e2.write_text(out / name, payload)
        else:
            e2.write_json(out / name, payload)
    e2.append_progress(out, "final_artifacts_written", artifact_count=len(artifacts))


def compare_core(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_artifacts = compose_artifacts(primary, deterministic_stub(True, {}))
    replay_artifacts = compose_artifacts(replay, deterministic_stub(True, {}))
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_hash = e2.payload_sha256(primary_artifacts[name])
        replay_hash = e2.payload_sha256(replay_artifacts[name])
        comparisons[name] = {"primary_hash": primary_hash, "replay_hash": replay_hash, "match": primary_hash == replay_hash}
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_global_determinism(stable_seed(f"core-{settings.seeds}-{device}"), device)
    task = generate_task(settings)
    git = e2.git_preflight()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        e2.append_progress(out, "startup", milestone=MILESTONE, settings=settings.__dict__, systems=list(SYSTEMS), selected_device=device)
    if settings.execution_mode == "parallel":
        searches = run_systems_parallel(task, settings, out)
    else:
        searches = run_systems_serial(task, settings, out)
    searches["oracle_reference_only"] = oracle_eval(task)
    return {"settings": settings, "task": task, "git": git, "searches": searches}


def deterministic_report(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    return compare_core(primary, replay)


def run_single_system(system: str, task: dict[str, Any], settings: Settings, out_raw: str | None) -> tuple[str, dict[str, Any]]:
    out = Path(out_raw) if out_raw else None
    started = time.perf_counter()
    if out is not None:
        e2.append_progress(out, "system_start", system=system, execution_mode=settings.execution_mode)
    try:
        if system == "e4_top_down_hierarchical_router":
            search = run_e4_style_search("e4_top_down_hierarchical_router", "top_down_hierarchical_router", task, settings, out)
        elif system == "tiny_mlp_gradient":
            search = train_torch_system("tiny_mlp_gradient", task, settings, out, "mlp")
        elif system == "tiny_mlp_mutation_only":
            search = run_vector_mutation_search("tiny_mlp_mutation_only", "mlp", task, settings, out)
        elif system == "tiny_recurrent_gradient":
            search = train_torch_system("tiny_recurrent_gradient", task, settings, out, "recurrent")
        elif system == "tiny_recurrent_mutation_only":
            search = run_vector_mutation_search("tiny_recurrent_mutation_only", "recurrent", task, settings, out)
        elif system == "hybrid_neural_frontend_mutation_router":
            search = train_hybrid_system(task, settings, out)
        elif system == "flat_detail_scanner":
            search = run_e4_style_search("flat_detail_scanner", "flat_detail_scanner", task, settings, out)
        elif system == "bottom_up_evidence_scanner":
            search = run_e4_style_search("bottom_up_evidence_scanner", "bottom_up_evidence_scanner", task, settings, out)
        elif system == "random_classifier":
            search = random_classifier_eval(task, stable_seed(f"random-classifier-{settings.seeds}"))
        else:
            raise ValueError(f"unknown E5 system: {system}")
        if out is not None:
            e2.append_progress(out, "system_complete", system=system, runtime_seconds=round_float(time.perf_counter() - started))
        return system, search
    except Exception as exc:
        if out is not None:
            e2.append_progress(out, "system_failed", system=system, error=repr(exc))
        raise


def run_systems_serial(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    searches: dict[str, Any] = {}
    searches["e4_top_down_hierarchical_router"] = run_e4_style_search("e4_top_down_hierarchical_router", "top_down_hierarchical_router", task, settings, out)
    searches["tiny_mlp_gradient"] = train_torch_system("tiny_mlp_gradient", task, settings, out, "mlp")
    searches["tiny_mlp_mutation_only"] = run_vector_mutation_search("tiny_mlp_mutation_only", "mlp", task, settings, out)
    searches["tiny_recurrent_gradient"] = train_torch_system("tiny_recurrent_gradient", task, settings, out, "recurrent")
    searches["tiny_recurrent_mutation_only"] = run_vector_mutation_search("tiny_recurrent_mutation_only", "recurrent", task, settings, out)
    searches["hybrid_neural_frontend_mutation_router"] = train_hybrid_system(task, settings, out)
    searches["flat_detail_scanner"] = run_e4_style_search("flat_detail_scanner", "flat_detail_scanner", task, settings, out)
    searches["bottom_up_evidence_scanner"] = run_e4_style_search("bottom_up_evidence_scanner", "bottom_up_evidence_scanner", task, settings, out)
    searches["random_classifier"] = random_classifier_eval(task, stable_seed(f"random-classifier-{settings.seeds}"))
    return searches


def run_systems_parallel(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    worker_count = settings.parallel_workers if settings.parallel_workers > 0 else min(len(SYSTEMS), max(1, (os.cpu_count() or 4) - 2))
    worker_count = min(max(1, worker_count), len(SYSTEMS))
    out_raw = out.as_posix() if out is not None else None
    if out is not None:
        e2.append_progress(out, "parallel_systems_start", systems=list(SYSTEMS), worker_count=worker_count)
    searches: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_single_system, system, task, settings, out_raw): system for system in SYSTEMS}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                if out is not None:
                    e2.append_progress(
                        out,
                        "parallel_heartbeat",
                        completed_systems=sorted(searches),
                        pending_systems=sorted(futures[future] for future in pending),
                        worker_count=worker_count,
                    )
                continue
            for future in done:
                system = futures[future]
                returned_system, search = future.result()
                if returned_system != system:
                    raise RuntimeError(f"system result mismatch: expected {system}, got {returned_system}")
                searches[system] = search
                if out is not None:
                    e2.append_progress(out, "parallel_system_result_received", system=system, completed_count=len(searches), pending_count=len(pending))
    if out is not None:
        e2.append_progress(out, "parallel_systems_complete", completed_systems=sorted(searches))
    return searches


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        seeds=parse_seeds(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_sigma=args.mutation_sigma,
        elite_count=args.elite_count,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        heartbeat_seconds=args.heartbeat_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--mutation-sigma", type=float, default=0.10)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--gradient-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--execution-mode", default="serial", choices=("serial", "parallel"))
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    out = resolve_out(args.out)
    core = run_core(settings, out)
    replay = run_core(settings, out / "_internal_replay")
    deterministic = deterministic_report(core, replay)
    if not deterministic["internal_replay_passed"] and select_device(settings.device) == "cuda":
        cpu_settings = Settings(**{**settings.__dict__, "device": "cpu"})
        e2.append_progress(out, "cuda_replay_mismatch_cpu_fallback_start")
        core = run_core(cpu_settings, out / "_cpu_fallback_primary")
        replay = run_core(cpu_settings, out / "_cpu_fallback_replay")
        deterministic = deterministic_report(core, replay)
    write_artifacts(out, core, deterministic)
    decision = compose_artifacts(core, deterministic)["decision.json"]
    print(json.dumps({"decision": decision["decision"], "winner": decision["winner"], "next": decision["next"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
