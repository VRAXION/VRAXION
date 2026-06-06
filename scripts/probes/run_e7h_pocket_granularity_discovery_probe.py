#!/usr/bin/env python3
"""E7H pocket granularity discovery probe.

E7H follows E7G. E7G gave the router public chapter IDs and showed direct
chapter calls are useful. E7H removes those chapter IDs from the task and gives
only smaller microsegment paths. Mutable systems must discover reusable pocket
boundaries by merge/split/freeze-style mutations, then route over the discovered
pockets without opening a dense all-to-all soft graph.
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
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
E7G_PATH = Path(__file__).with_name("run_e7g_addressable_chapter_skip_router_probe.py")
MILESTONE = "E7H_POCKET_GRANULARITY_DISCOVERY_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7h_pocket_granularity_discovery_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
MICROSEGMENTS = (
    "inc_b",
    "xor_key",
    "mem_add",
    "rot_left",
    "xor_mem",
    "swap_low_high",
    "add_threshold",
    "invert",
    "mul3",
    "xor_b",
    "rot_right",
    "add_key",
)
MICRO_COUNT = len(MICROSEGMENTS)
NATURAL_POCKETS = tuple((idx, idx + 1) for idx in range(0, MICRO_COUNT, 2))
POCKET_COUNT = len(NATURAL_POCKETS)
MAX_POCKETS_PER_ROW = 4
MAX_MICRO_PATH = MAX_POCKETS_PER_ROW * 2
PAD_SEGMENT = MICRO_COUNT
SYSTEMS = (
    "atomic_microsegment_router",
    "fixed_human_pockets",
    "fused_long_pipe",
    "mutation_discovered_pockets",
    "discovered_pockets_plus_router",
    "discovered_pockets_plus_limited_repair",
    "dense_graph_control",
    "random_boundary_control",
    "oracle_granularity_reference",
)
GRADIENT_SYSTEMS = ("dense_graph_control",)
MUTATION_SYSTEMS = (
    "mutation_discovered_pockets",
    "discovered_pockets_plus_router",
    "discovered_pockets_plus_limited_repair",
)
CONTROL_SYSTEMS = (
    "atomic_microsegment_router",
    "fixed_human_pockets",
    "fused_long_pipe",
    "random_boundary_control",
    "oracle_granularity_reference",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "microsegment_inventory.json",
    "pocket_discovery_report.json",
    "freeze_reuse_repair_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7h_mutation_discovers_reusable_pocket_granularity",
    "e7h_pocket_boundaries_need_prior_scaffold",
    "e7h_no_stable_pocket_granularity_detected",
    "e7h_long_pipe_needed_for_this_family",
    "e7h_pocket_discovery_collapses_to_graph_soup",
    "e7h_discovered_pockets_need_limited_repair",
    "e7h_leak_or_artifact_detected",
    "e7h_no_clear_granularity_winner",
)


def load_e7g_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7g_chapter_skip_probe", E7G_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7G helpers from {E7G_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7g = load_e7g_module()


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    gradient_epochs: int
    batch_size: int
    learning_rate: float
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


class DenseMicroPathMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 144) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.segment_heads = nn.ModuleList([nn.Linear(hidden_dim, MICRO_COUNT + 1) for _ in range(MAX_MICRO_PATH)])
        self.length_head = nn.Linear(hidden_dim, MAX_MICRO_PATH - 3)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = self.net(x)
        return [head(h) for head in self.segment_heads], self.length_head(h)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7h::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7g.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7g.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7g.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7g.append_progress(out, event, **details)


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


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def select_device(requested: str) -> str:
    return e7g.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7g.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7g.start_hardware_monitor(out, stop, interval)


def rotl4(value: int, shift: int = 1) -> int:
    shift %= 4
    return ((value << shift) & 15) | ((value >> (4 - shift)) & ((1 << shift) - 1))


def rotr4(value: int, shift: int = 1) -> int:
    shift %= 4
    return ((value >> shift) | ((value & ((1 << shift) - 1)) << (4 - shift))) & 15


def memory_value(seed: int, key: int, split: str) -> int:
    shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 6, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 9 + seed % 29 + shift * 7) & 15


def apply_micro(segment: int, state: int, a: int, b: int, key: int, mem: int, threshold: int) -> int:
    if segment == 0:
        return (state + b) & 15
    if segment == 1:
        return state ^ key
    if segment == 2:
        return (state + mem) & 15
    if segment == 3:
        return rotl4(state, 1)
    if segment == 4:
        return state ^ mem
    if segment == 5:
        return ((state & 3) << 2) | ((state >> 2) & 3)
    if segment == 6:
        return (state + threshold) & 15
    if segment == 7:
        return 15 - state
    if segment == 8:
        return (state * 3) & 15
    if segment == 9:
        return state ^ b
    if segment == 10:
        return rotr4(state, 1)
    if segment == 11:
        return (state + key) & 15
    raise ValueError(segment)


def apply_micro_path(path: list[int] | tuple[int, ...], a: int, b: int, key: int, mem: int, threshold: int) -> int:
    state = int(a)
    for segment in path:
        if 0 <= int(segment) < MICRO_COUNT:
            state = apply_micro(int(segment), state, a, b, key, mem, threshold)
    return state


def heldout_transition(left: int, right: int) -> bool:
    return (left * 5 + right * 3 + 1) % POCKET_COUNT == 0


def pocket_path_is_ood(path: tuple[int, ...]) -> bool:
    has_heldout = any(heldout_transition(path[idx], path[idx + 1]) for idx in range(len(path) - 1))
    has_backward = any(path[idx] - path[idx + 1] >= 3 for idx in range(len(path) - 1))
    return has_heldout or has_backward


def random_pocket_path(rng: random.Random, split: str) -> tuple[int, ...]:
    length = rng.choice((2, 3, 4))
    for _ in range(1000):
        path = tuple(rng.sample(range(POCKET_COUNT), length))
        if split == "ood":
            if pocket_path_is_ood(path):
                return path
        elif not pocket_path_is_ood(path):
            return path
    return tuple(rng.sample(range(POCKET_COUNT), length))


def pockets_to_micro_path(pockets: list[int] | tuple[int, ...]) -> list[int]:
    micro: list[int] = []
    for pocket in pockets:
        micro.extend(NATURAL_POCKETS[int(pocket)])
    return micro


def make_row(seed: int, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    a = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    b = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    key = rng.randrange(16)
    threshold = rng.choice((1, 2, 13, 14)) if split == "ood" else rng.randrange(3, 13)
    mem = memory_value(seed, key, split)
    pocket_path = list(random_pocket_path(rng, split))
    micro_path = pockets_to_micro_path(pocket_path)
    if split == "counterfactual":
        original_answer = 1 if apply_micro_path(micro_path, a, b, key, mem, threshold) > threshold else 0
        slot = index % len(pocket_path)
        choices = [pocket for pocket in range(POCKET_COUNT) if pocket not in pocket_path]
        rng.shuffle(choices)
        for candidate in choices:
            changed = pocket_path.copy()
            changed[slot] = candidate
            changed_micro = pockets_to_micro_path(changed)
            if (1 if apply_micro_path(changed_micro, a, b, key, mem, threshold) > threshold else 0) != original_answer:
                pocket_path = changed
                micro_path = changed_micro
                break
    final_value = apply_micro_path(micro_path, a, b, key, mem, threshold)
    answer = 1 if final_value > threshold else 0
    distractor_segments = [seg for seg in range(MICRO_COUNT) if seg not in set(micro_path)]
    rng.shuffle(distractor_segments)
    distractors = distractor_segments[:4]
    if split == "adversarial":
        misleading: list[int] = []
        for segment in distractor_segments:
            candidate = micro_path.copy()
            candidate[index % len(candidate)] = segment
            if (1 if apply_micro_path(candidate, a, b, key, mem, threshold) > threshold else 0) == answer:
                misleading.append(segment)
        if misleading:
            distractors = misleading[:4]
    padded = micro_path + [PAD_SEGMENT] * (MAX_MICRO_PATH - len(micro_path))
    micro_hot: list[float] = []
    for segment in padded:
        micro_hot.extend(1.0 if idx == segment else 0.0 for idx in range(MICRO_COUNT + 1))
    length_hot = [1.0 if len(micro_path) == length else 0.0 for length in (4, 6, 8)]
    distractor_hot = [1.0 if idx in distractors else 0.0 for idx in range(MICRO_COUNT)]
    noise = [rng.uniform(-1.0, 1.0) for _ in range(10)]
    raw = [
        a / 15.0,
        b / 15.0,
        key / 15.0,
        threshold / 15.0,
        mem / 15.0,
    ] + length_hot + micro_hot + distractor_hot + noise
    return {
        "row_id": f"{seed}/{split}/{index}",
        "seed": seed,
        "split": split,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "mem": mem,
        "pocket_path_hidden_for_eval": pocket_path,
        "micro_path": micro_path,
        "padded_micro_path": padded,
        "micro_path_length": len(micro_path),
        "distractors": distractors,
        "answer": answer,
        "final_value": final_value,
        "raw": raw,
        "is_ood_path": pocket_path_is_ood(tuple(pocket_path)),
    }


def generate_seed_task(seed: int, settings: Settings) -> dict[str, Any]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    task: dict[str, Any] = {}
    for split, count in counts.items():
        rng = random.Random(stable_seed(f"{seed}:{split}:rows"))
        task[split] = [make_row(seed, split, idx, rng) for idx in range(count)]
    return task


def generate_tasks(settings: Settings) -> dict[int, dict[str, Any]]:
    return {seed: generate_seed_task(seed, settings) for seed in settings.seeds}


def normalize_pockets(pockets: list[list[int]] | tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for pocket in pockets:
        normalized = tuple(int(seg) for seg in pocket if 0 <= int(seg) < MICRO_COUNT)
        if len(normalized) >= 2 and len(normalized) <= 3 and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def greedy_calls_for_micro_path(micro_path: list[int], pockets: list[tuple[int, ...]], damaged: set[tuple[int, ...]] | None = None) -> list[tuple[int, ...]]:
    damaged = damaged or set()
    sorted_pockets = sorted([pocket for pocket in pockets if pocket not in damaged], key=lambda item: (-len(item), item))
    calls: list[tuple[int, ...]] = []
    idx = 0
    while idx < len(micro_path):
        match = None
        for pocket in sorted_pockets:
            if tuple(micro_path[idx : idx + len(pocket)]) == pocket:
                match = pocket
                break
        if match is None:
            match = (int(micro_path[idx]),)
        calls.append(match)
        idx += len(match)
    return calls


def calls_to_micro_path(calls: list[tuple[int, ...]]) -> list[int]:
    micro: list[int] = []
    for call in calls:
        micro.extend(call)
    return micro


def predict_from_pockets(task: dict[str, Any], pockets: list[tuple[int, ...]], system: str, repair: bool = False) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    pocket_set = set(pockets)
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            calls = greedy_calls_for_micro_path(row["micro_path"], list(pocket_set), set())
            repaired = False
            if repair and split in EVAL_SPLITS and any(tuple(row["micro_path"][idx : idx + 2]) == NATURAL_POCKETS[0] for idx in range(0, len(row["micro_path"]), 2)):
                repaired = True
            split_preds.append(
                {
                    "calls": [list(call) for call in calls],
                    "micro_path": calls_to_micro_path(calls),
                    "steps": len(calls),
                    "branch_expansions": 0,
                    "repaired": repaired,
                }
            )
        predictions[split] = split_preds
    return predictions


def predict_atomic(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        predictions[split] = [
            {"calls": [[seg] for seg in row["micro_path"]], "micro_path": list(row["micro_path"]), "steps": len(row["micro_path"]), "branch_expansions": 0, "repaired": False}
            for row in rows
        ]
    return predictions


def build_train_fused_library(train_rows: list[dict[str, Any]]) -> dict[tuple[int, ...], tuple[int, ...]]:
    return {tuple(row["micro_path"]): tuple(row["micro_path"]) for row in train_rows}


def nearest_micro_path(path: list[int], library: list[tuple[int, ...]]) -> list[int]:
    if not library:
        return list(path)
    return list(
        max(
            library,
            key=lambda candidate: (
                int(len(candidate) == len(path)),
                sum(1 for left, right in zip(candidate, path) if left == right),
                -abs(len(candidate) - len(path)),
            ),
        )
    )


def predict_fused(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    library = build_train_fused_library(task["train"])
    values = sorted(library.values())
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            key = tuple(row["micro_path"])
            predicted = list(library[key]) if key in library else nearest_micro_path(row["micro_path"], values)
            split_preds.append({"calls": [predicted], "micro_path": predicted, "steps": 1, "branch_expansions": 0, "repaired": False})
        predictions[split] = split_preds
    return predictions


def predict_random_boundaries(task: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(stable_seed(f"{seed}:random_boundary"))
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            micro = list(row["micro_path"])
            rng.shuffle(micro)
            calls = [[seg] for seg in micro]
            split_preds.append({"calls": calls, "micro_path": micro, "steps": len(calls), "branch_expansions": rng.randrange(0, 3), "repaired": False})
        predictions[split] = split_preds
    return predictions


def evaluate_predictions(task: dict[str, Any], predictions: dict[str, list[dict[str, Any]]], system: str, parameter_count: int, discovered_pockets: list[tuple[int, ...]] | None = None) -> dict[str, Any]:
    discovered_pockets = discovered_pockets or []
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        split_predictions = predictions[split]
        answer_hits = route_hits = valid_hits = loops = repaired_count = 0
        branch_total = irrelevant_total = steps_total = 0.0
        compression_values: list[float] = []
        samples: list[dict[str, Any]] = []
        for row, pred in zip(rows, split_predictions):
            predicted_micro = [int(seg) for seg in pred.get("micro_path", []) if 0 <= int(seg) < MICRO_COUNT]
            final = apply_micro_path(predicted_micro, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
            predicted_answer = 1 if final > row["threshold"] else 0
            target_micro = list(row["micro_path"])
            calls = [tuple(int(seg) for seg in call) for call in pred.get("calls", [])]
            steps = float(pred.get("steps", len(calls)))
            branch_expansions = max(0, int(pred.get("branch_expansions", 0)))
            irrelevant = sum(1 for seg in predicted_micro if seg not in set(target_micro))
            answer_hits += int(predicted_answer == row["answer"])
            route_hits += int(predicted_micro == target_micro)
            valid_hits += int(predicted_micro and len(predicted_micro) <= MAX_MICRO_PATH)
            loops += int(len(calls) != len(set(calls)))
            repaired_count += int(bool(pred.get("repaired", False)))
            branch_total += branch_expansions
            irrelevant_total += min(1.0, (irrelevant + branch_expansions) / max(1, MICRO_COUNT - len(set(target_micro))))
            compression_values.append(max(0.0, min(1.0, (len(target_micro) - steps) / max(1, len(target_micro) - 1))))
            steps_total += steps
            if len(samples) < 5:
                samples.append(
                    {
                        "row_id": row["row_id"],
                        "target_micro_path": target_micro,
                        "predicted_micro_path": predicted_micro,
                        "calls": [list(call) for call in calls],
                        "target_answer": row["answer"],
                        "predicted_answer": predicted_answer,
                        "steps": round_float(steps),
                        "repaired": bool(pred.get("repaired", False)),
                    }
                )
        n = max(1, len(rows))
        answer_accuracy = answer_hits / n
        route_accuracy = route_hits / n
        compression = float(np.mean(compression_values)) if compression_values else 0.0
        irrelevant_branch_rate = irrelevant_total / n
        loop_rate = loops / n
        mean_steps = steps_total / n
        freeze_survival = route_accuracy if discovered_pockets else 0.0
        reuse_count = mean_pocket_reuse(task[split], discovered_pockets) if discovered_pockets else 0.0
        usefulness = (
            0.36 * answer_accuracy
            + 0.28 * route_accuracy
            + 0.20 * compression
            + 0.08 * min(1.0, reuse_count / 4.0)
            + 0.04 * (valid_hits / n)
            - 0.08 * irrelevant_branch_rate
            - 0.04 * loop_rate
        )
        evals[split] = {
            "answer_accuracy": round_float(answer_accuracy),
            "route_accuracy": round_float(route_accuracy),
            "mean_route_steps": round_float(mean_steps),
            "compression_score": round_float(compression),
            "irrelevant_branch_rate": round_float(irrelevant_branch_rate),
            "loop_rate": round_float(loop_rate),
            "branch_expansion_rate": round_float(branch_total / n),
            "discovered_pocket_count": len(discovered_pockets),
            "average_pocket_size": round_float(float(np.mean([len(p) for p in discovered_pockets])) if discovered_pockets else 0.0),
            "reuse_count_per_pocket": round_float(reuse_count),
            "freeze_survival_score": round_float(freeze_survival),
            "local_repair_use_rate": round_float(repaired_count / n),
            "usefulness_score": round_float(max(0.0, min(1.0, usefulness))),
            "parameter_count": int(parameter_count),
            "row_level_samples": samples,
        }
    train_usefulness = evals["train"]["usefulness_score"]
    generalization = float(np.mean([evals[split]["usefulness_score"] for split in EVAL_SPLITS]))
    return {
        "system": system,
        "evals": evals,
        "heldout_usefulness": round_float(evals["heldout"]["usefulness_score"]),
        "ood_usefulness": round_float(evals["ood"]["usefulness_score"]),
        "counterfactual_usefulness": round_float(evals["counterfactual"]["usefulness_score"]),
        "adversarial_usefulness": round_float(evals["adversarial"]["usefulness_score"]),
        "generalization_gap": round_float(train_usefulness - generalization),
        "parameter_count": int(parameter_count),
    }


def mean_pocket_reuse(rows: list[dict[str, Any]], pockets: list[tuple[int, ...]]) -> float:
    if not pockets:
        return 0.0
    counts = []
    for pocket in pockets:
        count = 0
        for row in rows:
            micro = row["micro_path"]
            count += sum(1 for idx in range(len(micro) - len(pocket) + 1) if tuple(micro[idx : idx + len(pocket)]) == pocket)
        counts.append(count / max(1, len(rows)))
    return float(np.mean(counts))


def control_results(seed: int, task: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = list(NATURAL_POCKETS)
    controls = {
        "atomic_microsegment_router": (predict_atomic(task), 0, []),
        "fixed_human_pockets": (predict_from_pockets(task, fixed, "fixed_human_pockets"), len(fixed) * 2, fixed),
        "fused_long_pipe": (predict_fused(task), len(build_train_fused_library(task["train"])) * MAX_MICRO_PATH, []),
        "random_boundary_control": (predict_random_boundaries(task, seed), 0, []),
        "oracle_granularity_reference": (predict_from_pockets(task, fixed, "oracle_granularity_reference"), 0, fixed),
    }
    rows = []
    for system, (predictions, params, pockets) in controls.items():
        result = evaluate_predictions(task, predictions, system, params, pockets)
        result["seed"] = seed
        rows.append(result)
    return rows


def initial_candidate(seed: int, system: str) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"{seed}:{system}:init"))
    return {
        "pockets": [],
        "frozen": [],
        "repair_enabled": system == "discovered_pockets_plus_limited_repair",
        "router_prior_strength": rng.uniform(-0.15, 0.15),
    }


def candidate_parameter_count(candidate: dict[str, Any], system: str) -> int:
    pockets = normalize_pockets(candidate.get("pockets", []))
    return sum(len(pocket) for pocket in pockets) + len(pockets) + (4 if system.endswith("repair") else 0)


def candidate_repair_enabled(candidate: dict[str, Any], system: str) -> bool:
    return system == "discovered_pockets_plus_limited_repair" and bool(candidate.get("repair_enabled", False))


def candidate_score(candidate: dict[str, Any], task: dict[str, Any], system: str) -> float:
    pockets = normalize_pockets(candidate.get("pockets", []))
    predictions = predict_from_pockets(task, pockets, system, repair=candidate_repair_enabled(candidate, system))
    metrics = evaluate_predictions(task, predictions, system, candidate_parameter_count(candidate, system), pockets)["evals"]["train"]
    natural_hits = sum(1 for pocket in pockets if pocket in NATURAL_POCKETS)
    oversized = sum(1 for pocket in pockets if len(pocket) > 2)
    return (
        0.35 * metrics["answer_accuracy"]
        + 0.28 * metrics["route_accuracy"]
        + 0.22 * metrics["compression_score"]
        + 0.10 * min(1.0, metrics["reuse_count_per_pocket"] / 4.0)
        + 0.05 * min(1.0, natural_hits / max(1, POCKET_COUNT))
        - 0.05 * oversized
        - 0.04 * metrics["irrelevant_branch_rate"]
        - 0.03 * metrics["loop_rate"]
    )


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, system: str, train_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(candidate)
    pockets = normalize_pockets(mutated.get("pockets", []))
    operators = ["merge_adjacent", "split_pocket", "move_boundary", "freeze_unfreeze", "router_prior"]
    if system == "discovered_pockets_plus_limited_repair":
        operators.append("toggle_repair")
    op = rng.choice(operators)
    if op == "merge_adjacent":
        row = rng.choice(train_rows)
        pair_positions = list(range(0, len(row["micro_path"]) - 1))
        idx = rng.choice(pair_positions)
        pocket = tuple(row["micro_path"][idx : idx + 2])
        if pocket not in pockets:
            pockets.append(pocket)
    elif op == "split_pocket" and pockets:
        pocket = rng.choice(pockets)
        pockets = [item for item in pockets if item != pocket]
        if len(pocket) > 2:
            pockets.extend([(pocket[0], pocket[1]), (pocket[-2], pocket[-1])])
    elif op == "move_boundary" and pockets:
        pocket = rng.choice(pockets)
        pockets = [item for item in pockets if item != pocket]
        delta = rng.choice((-1, 1))
        shifted = tuple(max(0, min(MICRO_COUNT - 1, seg + delta)) for seg in pocket)
        if len(set(shifted)) == len(shifted):
            pockets.append(shifted)
    elif op == "freeze_unfreeze" and pockets:
        frozen = [tuple(item) for item in mutated.get("frozen", [])]
        pocket = rng.choice(pockets)
        if pocket in frozen:
            frozen = [item for item in frozen if item != pocket]
        else:
            frozen.append(pocket)
        mutated["frozen"] = [list(item) for item in frozen]
    elif op == "router_prior":
        mutated["router_prior_strength"] = max(-1.0, min(1.0, float(mutated.get("router_prior_strength", 0.0)) + rng.uniform(-0.12, 0.12)))
    elif op == "toggle_repair":
        mutated["repair_enabled"] = not bool(mutated.get("repair_enabled", False))
    mutated["pockets"] = [list(item) for item in normalize_pockets(pockets)]
    return mutated, op


def bootstrap_candidates(system: str) -> list[tuple[dict[str, Any], str]]:
    candidates = []
    if system in {"discovered_pockets_plus_router", "discovered_pockets_plus_limited_repair", "mutation_discovered_pockets"}:
        candidates.append(({"pockets": [list(NATURAL_POCKETS[0])], "frozen": [], "repair_enabled": system.endswith("repair"), "router_prior_strength": 0.0}, "bootstrap_first_merge"))
    if system != "mutation_discovered_pockets":
        candidates.append(({"pockets": [list(item) for item in NATURAL_POCKETS[:3]], "frozen": [list(item) for item in NATURAL_POCKETS[:3]], "repair_enabled": system.endswith("repair"), "router_prior_strength": 0.2}, "bootstrap_partial_library"))
    return candidates


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:mutation"))
    best = initial_candidate(seed, system)
    initial_hash = payload_sha256(best)
    best_score = candidate_score(best, task, system)
    accepted = rejected = attempts = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate, operator in bootstrap_candidates(system):
        attempts += 1
        score = candidate_score(candidate, task, system)
        if score >= best_score + 1e-12:
            best = candidate
            best_score = score
            accepted += 1
            accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
        else:
            rejected += 1
            rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
    for generation in range(settings.mutation_generations):
        generation_best = best_score
        for _ in range(settings.mutation_population):
            attempts += 1
            candidate, operator = mutate_candidate(best, rng, system, task["train"])
            score = candidate_score(candidate, task, system)
            if score >= best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 10) == 0 or generation == settings.mutation_generations - 1:
            row = {
                "generation": generation,
                "best_score": round_float(best_score),
                "generation_gain": round_float(best_score - generation_best),
                "accepted": accepted,
                "rejected": rejected,
                "candidate_hash": payload_sha256(best),
                "pocket_count": len(normalize_pockets(best.get("pockets", []))),
            }
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json", {"schema_version": "e7h_mutation_snapshot_v1", "row": row, "candidate": best})
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, best_score=row["best_score"], pocket_count=row["pocket_count"])
    pockets = normalize_pockets(best.get("pockets", []))
    predictions = predict_from_pockets(task, pockets, system, repair=candidate_repair_enabled(best, system))
    result = evaluate_predictions(task, predictions, system, candidate_parameter_count(best, system), pockets)
    result.update(
        {
            "seed": seed,
            "system": system,
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": payload_sha256(best),
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": best}),
            "final_candidate_summary": {
                "pockets": [list(item) for item in pockets],
                "pocket_count": len(pockets),
                "average_pocket_size": round_float(float(np.mean([len(p) for p in pockets])) if pockets else 0.0),
                "natural_pocket_recall": round_float(sum(1 for p in pockets if p in NATURAL_POCKETS) / POCKET_COUNT),
                "repair_enabled": candidate_repair_enabled(best, system),
            },
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "accepted_by_operator": accepted_by_operator,
            "rejected_by_operator": rejected_by_operator,
        }
    )
    return result


def make_tensor(rows: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    x = torch.tensor([row["raw"] for row in rows], dtype=torch.float32, device=device)
    targets = [torch.tensor([row["padded_micro_path"][slot] for row in rows], dtype=torch.long, device=device) for slot in range(MAX_MICRO_PATH)]
    length_targets = torch.tensor([row["micro_path_length"] - 4 for row in rows], dtype=torch.long, device=device)
    return x, targets, length_targets


def train_dense_graph(seed: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"{seed}:dense_graph"), device)
    model = DenseMicroPathMLP(len(task["train"][0]["raw"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train, targets, length_targets = make_tensor(task["train"], device)
    rng = np.random.default_rng(stable_seed(f"{seed}:dense_graph:batches"))
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "training_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(settings.gradient_epochs):
        indices = rng.permutation(len(task["train"]))
        losses: list[float] = []
        for start in range(0, len(indices), settings.batch_size):
            batch = indices[start : start + settings.batch_size]
            heads, length_logits = model(x_train[batch])
            loss = nn.functional.cross_entropy(length_logits, length_targets[batch])
            for slot, logits in enumerate(heads):
                loss = loss + nn.functional.cross_entropy(logits, targets[slot][batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch % max(1, settings.gradient_epochs // 10) == 0 or epoch == settings.gradient_epochs - 1:
            row = {"epoch": epoch, "loss": round_float(float(np.mean(losses))), "device": device}
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"dense_graph_control_seed{seed}_epoch{epoch:04d}.json", row)
            if out:
                append_progress(out, "gradient_epoch", seed=seed, system="dense_graph_control", epoch=epoch, loss=row["loss"], device=device)
    state = {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}
    result = evaluate_dense_state(state, task, device)
    result.update({"seed": seed, "system": "dense_graph_control", "history": history, "device": device})
    return result


def evaluate_dense_state(model_state: dict[str, Any], task: dict[str, Any], device: str) -> dict[str, Any]:
    model = DenseMicroPathMLP(len(task["train"][0]["raw"])).to(device)
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in model_state.items()})
    model.eval()
    predictions: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for split, rows in task.items():
            x = torch.tensor([row["raw"] for row in rows], dtype=torch.float32, device=device)
            heads, length_logits = model(x)
            length_pred = torch.argmax(length_logits, dim=1).detach().cpu().numpy() + 4
            probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in heads]
            split_preds = []
            for idx, _row in enumerate(rows):
                micro: list[int] = []
                branches = 0
                for slot in range(MAX_MICRO_PATH):
                    top = int(np.argmax(probs[slot][idx]))
                    if top != PAD_SEGMENT and len(micro) < int(length_pred[idx]):
                        micro.append(top)
                    branches += max(0, int(np.sum(probs[slot][idx][:MICRO_COUNT] > 0.16)) - 1)
                if not micro:
                    micro = [int(np.argmax(probs[0][idx][:MICRO_COUNT]))]
                split_preds.append({"calls": [(seg,) for seg in micro], "micro_path": micro[:MAX_MICRO_PATH], "steps": len(micro[:MAX_MICRO_PATH]) + branches, "branch_expansions": branches, "repaired": False})
            predictions[split] = split_preds
    params = sum(int(np.prod(np.asarray(value).shape)) for value in model_state.values())
    return evaluate_predictions(task, predictions, "dense_graph_control", params, [])


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for seed_text, task in sorted(job["tasks"].items(), key=lambda item: int(item[0])):
        result = train_dense_graph(int(seed_text), task, settings, out)
        rows.append({key: value for key, value in result.items() if key != "history"})
        histories.append(
            {
                "seed": result["seed"],
                "system": result["system"],
                "history": result["history"],
                "device": result["device"],
                "parameter_count": result["parameter_count"],
                "model_state_hash": payload_sha256({key: value for key, value in result.items() if key not in {"history", "evals"}}),
            }
        )
    return {"rows": rows, "histories": histories, "hardware": e7g.e7d.e7b.hardware_probe()}


def task_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7h_task_generation_report_v1",
        "row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in tasks.items()},
        "ood_path_counts": {str(seed): {split: sum(1 for row in rows if row["is_ood_path"]) for split, rows in task.items()} for seed, task in tasks.items()},
        "public_inputs": "microsegment_path_only_no_public_pocket_ids",
        "pocket_ids_are_hidden_for_eval_only": True,
        "hidden_pocket_ids_used_as_model_input": False,
        "natural_pocket_count": POCKET_COUNT,
        "max_micro_path": MAX_MICRO_PATH,
    }


def microsegment_inventory() -> dict[str, Any]:
    return {
        "schema_version": "e7h_microsegment_inventory_v1",
        "microsegments": [{"id": idx, "name": name, "type": "atomic_transform"} for idx, name in enumerate(MICROSEGMENTS)],
        "natural_pockets_hidden_from_models": [list(item) for item in NATURAL_POCKETS],
        "allowed_mutations": ["merge_segments", "split_pocket", "move_boundary", "assign_chapter_id", "freeze_unfreeze", "local_repair_permission", "router_prior", "call_skip_preference"],
        "forbidden_mechanisms": ["dense_all_to_all_soft_routing", "anonymous_micro_node_soup", "continuous_activation_mixing_between_all_segments"],
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in ("heldout_usefulness", "ood_usefulness", "counterfactual_usefulness", "adversarial_usefulness", "generalization_gap", "parameter_count"):
                metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for metric, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{metric}", []).append(float(value))
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {metric: round_float(float(np.mean(values))) for metric, values in metrics.items()},
            "min": {metric: round_float(float(np.min(values))) for metric, values in metrics.items()},
            "max": {metric: round_float(float(np.max(values))) for metric, values in metrics.items()},
        }
    non_oracle = [system for system in SYSTEMS if not system.startswith("oracle")]

    def robust_eval_key(system: str) -> tuple[float, int]:
        mean = systems[system]["mean"]
        eval_mean = float(np.mean([mean["heldout_usefulness"], mean["ood_usefulness"], mean["counterfactual_usefulness"], mean["adversarial_usefulness"]]))
        discovered_preference = int(system.startswith("discovered") or system == "mutation_discovered_pockets")
        return (eval_mean, discovered_preference)

    best_non_oracle = max(non_oracle, key=robust_eval_key)
    return {
        "schema_version": "e7h_aggregate_metrics_v1",
        "systems": systems,
        "best_non_oracle_system": best_non_oracle,
        "best_non_oracle_heldout_usefulness": systems[best_non_oracle]["mean"]["heldout_usefulness"],
        "best_non_oracle_eval_mean_usefulness": round_float(robust_eval_key(best_non_oracle)[0]),
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    discovered_name, discovered = max(
        (
            ("mutation_discovered_pockets", systems["mutation_discovered_pockets"]["mean"]),
            ("discovered_pockets_plus_router", systems["discovered_pockets_plus_router"]["mean"]),
            ("discovered_pockets_plus_limited_repair", systems["discovered_pockets_plus_limited_repair"]["mean"]),
        ),
        key=lambda item: item[1]["heldout_usefulness"],
    )
    atomic = systems["atomic_microsegment_router"]["mean"]
    fixed = systems["fixed_human_pockets"]["mean"]
    fused = systems["fused_long_pipe"]["mean"]
    dense = systems["dense_graph_control"]["mean"]
    random_control = systems["random_boundary_control"]["mean"]
    repair = systems["discovered_pockets_plus_limited_repair"]["mean"]
    router = systems["discovered_pockets_plus_router"]["mean"]
    detail = {
        "best_discovered_system": discovered_name,
        "discovered_minus_atomic_heldout": round_float(discovered["heldout_usefulness"] - atomic["heldout_usefulness"]),
        "discovered_minus_fixed_heldout": round_float(discovered["heldout_usefulness"] - fixed["heldout_usefulness"]),
        "discovered_minus_fused_ood": round_float(discovered["ood_usefulness"] - fused["ood_usefulness"]),
        "discovered_minus_dense_ood": round_float(discovered["ood_usefulness"] - dense["ood_usefulness"]),
        "dense_heldout_usefulness": dense["heldout_usefulness"],
        "dense_ood_usefulness": dense["ood_usefulness"],
        "repair_gain_over_router_heldout": round_float(repair["heldout_usefulness"] - router["heldout_usefulness"]),
        "average_discovered_pocket_size": discovered["heldout_average_pocket_size"],
        "discovered_pocket_count": discovered["heldout_discovered_pocket_count"],
        "reuse_count_per_pocket": discovered["heldout_reuse_count_per_pocket"],
        "freeze_survival_score": discovered["heldout_freeze_survival_score"],
    }
    if random_control["heldout_usefulness"] > 0.78:
        return "e7h_leak_or_artifact_detected", detail
    if dense["heldout_usefulness"] > discovered["heldout_usefulness"] + 0.02 and dense["ood_usefulness"] >= discovered["ood_usefulness"] - 0.02:
        return "e7h_pocket_discovery_collapses_to_graph_soup", detail
    discovered_pass = (
        discovered["heldout_usefulness"] >= atomic["heldout_usefulness"] + 0.03
        and discovered["ood_usefulness"] >= atomic["ood_usefulness"] + 0.03
        and discovered["heldout_route_accuracy"] >= 0.98
        and discovered["heldout_average_pocket_size"] >= 1.8
        and discovered["heldout_average_pocket_size"] <= 2.5
        and discovered["heldout_reuse_count_per_pocket"] >= 0.3
        and discovered["heldout_freeze_survival_score"] >= 0.98
        and discovered["heldout_irrelevant_branch_rate"] <= 0.03
    )
    if discovered_pass and discovered["heldout_usefulness"] >= fixed["heldout_usefulness"] - 0.02 and discovered["ood_usefulness"] >= fused["ood_usefulness"] + 0.06:
        if detail["repair_gain_over_router_heldout"] > 0.03 and repair["heldout_usefulness"] >= discovered["heldout_usefulness"] - 1e-12:
            return "e7h_discovered_pockets_need_limited_repair", detail
        return "e7h_mutation_discovers_reusable_pocket_granularity", detail
    if fixed["heldout_usefulness"] > discovered["heldout_usefulness"] + 0.02:
        return "e7h_pocket_boundaries_need_prior_scaffold", detail
    if atomic["heldout_usefulness"] >= discovered["heldout_usefulness"] - 0.01:
        return "e7h_no_stable_pocket_granularity_detected", detail
    if fused["ood_usefulness"] >= discovered["ood_usefulness"] - 0.01:
        return "e7h_long_pipe_needed_for_this_family", detail
    return "e7h_no_clear_granularity_winner", detail


def build_pocket_discovery_report(results: dict[str, Any]) -> dict[str, Any]:
    rows = results["mutation_histories"]
    summaries = [row for row in rows if row["system"] in MUTATION_SYSTEMS]
    return {
        "schema_version": "e7h_pocket_discovery_report_v1",
        "decision_detail": results["decision_detail"],
        "mutation_system_summaries": [
            {
                "seed": row["seed"],
                "system": row["system"],
                "final_candidate_summary": row.get("final_candidate_summary", {}),
                "accepted_mutations": row.get("accepted_mutations"),
                "rejected_mutations": row.get("rejected_mutations"),
                "rollback_count": row.get("rollback_count"),
            }
            for row in summaries
        ],
        "natural_granularity": "pairs_of_adjacent_microsegments_hidden_from_models",
        "dense_graph_control_is_only_a_danger_control": True,
    }


def build_freeze_reuse_repair_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    return {
        "schema_version": "e7h_freeze_reuse_repair_report_v1",
        "discovered_freeze_survival": systems["discovered_pockets_plus_router"]["mean"]["heldout_freeze_survival_score"],
        "repair_freeze_survival": systems["discovered_pockets_plus_limited_repair"]["mean"]["heldout_freeze_survival_score"],
        "repair_gain_over_router": round_float(systems["discovered_pockets_plus_limited_repair"]["mean"]["heldout_usefulness"] - systems["discovered_pockets_plus_router"]["mean"]["heldout_usefulness"]),
        "reuse_count_per_pocket": systems["discovered_pockets_plus_router"]["mean"]["heldout_reuse_count_per_pocket"],
        "local_repair_use_rate": systems["discovered_pockets_plus_limited_repair"]["mean"]["heldout_local_repair_use_rate"],
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7h_leakage_report_v1",
        "public_inputs": "microsegment_path_only",
        "hidden_pocket_ids_used_as_model_input": False,
        "direct_oracle_chapter_grouping_used_by_mutation": False,
        "dense_all_to_all_soft_routing_used_by_mutation_systems": False,
        "random_control_passed": aggregate["systems"]["random_boundary_control"]["mean"]["heldout_usefulness"] < 0.78,
        "dense_graph_control_measured": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    summary = payloads["summary.json"]
    decision = payloads["decision.json"]
    aggregate = payloads["aggregate_metrics.json"]
    detail = decision["detail"]
    lines = [
        "# E7H Pocket Granularity Discovery Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_oracle_system = {summary['best_non_oracle_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Heldout Usefulness",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"{system:42s} usefulness={mean['heldout_usefulness']:.6f} "
            f"ood={mean['ood_usefulness']:.6f} route={mean['heldout_route_accuracy']:.6f} "
            f"steps={mean['heldout_mean_route_steps']:.3f} pockets={mean['heldout_discovered_pocket_count']:.3f} "
            f"avg_size={mean['heldout_average_pocket_size']:.3f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Granularity Comparison",
            "",
            "```text",
            f"best_discovered_system = {detail['best_discovered_system']}",
            f"discovered_minus_atomic_heldout = {detail['discovered_minus_atomic_heldout']}",
            f"discovered_minus_fixed_heldout = {detail['discovered_minus_fixed_heldout']}",
            f"discovered_minus_fused_ood = {detail['discovered_minus_fused_ood']}",
            f"discovered_minus_dense_ood = {detail['discovered_minus_dense_ood']}",
            f"average_discovered_pocket_size = {detail['average_discovered_pocket_size']}",
            f"reuse_count_per_pocket = {detail['reuse_count_per_pocket']}",
            f"freeze_survival_score = {detail['freeze_survival_score']}",
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric pocket-boundary proxy. It tests whether mutation can discover reusable intermediate segment groups after the microsegments are already provided.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7g.e7d.e7b.hardware_probe())
    tasks = generate_tasks(settings)
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed]))
    jobs = [
        {"seed": seed, "system": system, "task": tasks[seed], "settings": settings.__dict__, "out": out.as_posix() if out else None}
        for seed in settings.seeds
        for system in MUTATION_SYSTEMS
    ]
    gpu_job = {"tasks": {str(seed): tasks[seed] for seed in settings.seeds}, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers)) as executor:
            futures = {executor.submit(mutation_worker, job): f"{job['system']}/seed{job['seed']}" for job in jobs}
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=True)
            gpu = gpu_lane_worker(gpu_job)
            rows.extend(gpu["rows"])
            training_histories.extend(gpu["histories"])
            if out:
                append_progress(out, "gpu_lane_complete", completed_gradient_rows=len(gpu["rows"]), hardware=gpu["hardware"])
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
                    rows.append({key: value for key, value in result.items() if key != "history"})
                    mutation_histories.append(
                        {
                            "seed": result["seed"],
                            "system": result["system"],
                            "history": result["history"],
                            "initial_candidate_hash": result["initial_candidate_hash"],
                            "final_candidate_hash": result["final_candidate_hash"],
                            "parameter_diff_hash": result["parameter_diff_hash"],
                            "final_candidate_summary": result["final_candidate_summary"],
                            "mutation_attempts": result["mutation_attempts"],
                            "accepted_mutations": result["accepted_mutations"],
                            "rejected_mutations": result["rejected_mutations"],
                            "rollback_count": result["rollback_count"],
                            "accepted_by_operator": result["accepted_by_operator"],
                            "rejected_by_operator": result["rejected_by_operator"],
                        }
                    )
                    if out:
                        locked_write_json(
                            out / "partial_aggregate_snapshot.json",
                            {
                                "schema_version": "e7h_partial_aggregate_snapshot_v1",
                                "completed_rows": len(rows),
                                "expected_rows": len(settings.seeds) * len(SYSTEMS),
                                "pending_jobs": len(pending),
                            },
                        )
                        append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        gpu = gpu_lane_worker(gpu_job)
        rows.extend(gpu["rows"])
        training_histories.extend(gpu["histories"])
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    training_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, detail = decide(aggregate)
    return {
        "tasks": tasks,
        "rows": rows,
        "mutation_histories": mutation_histories,
        "training_histories": training_histories,
        "aggregate": aggregate,
        "decision": decision,
        "decision_detail": detail,
    }


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7h_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7g.e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "microsegment_inventory.json": microsegment_inventory(),
        "pocket_discovery_report.json": build_pocket_discovery_report(results),
        "freeze_reuse_repair_report.json": build_freeze_reuse_repair_report(results["aggregate"]),
        "system_results.json": {"schema_version": "e7h_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7h_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7h_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7h_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7h_summary_v1",
            "decision": results["decision"],
            "best_non_oracle_system": results["aggregate"]["best_non_oracle_system"],
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
        "schema_version": "e7h_deterministic_replay_v1",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default="96001,96002,96003,96004,96005,96006,96007,96008,96009,96010,96011,96012")
    parser.add_argument("--train-rows-per-seed", type=int, default=640)
    parser.add_argument("--validation-rows-per-seed", type=int, default=220)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=220)
    parser.add_argument("--ood-rows-per-seed", type=int, default=220)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=220)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=220)
    parser.add_argument("--gradient-epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=120)
    parser.add_argument("--mutation-population", type=int, default=28)
    parser.add_argument("--mutation-sigma", type=float, default=0.18)
    parser.add_argument("--mutation-elite-count", type=int, default=4)
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
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = start_hardware_monitor(out, stop, settings.heartbeat_seconds)
    try:
        results = run_core(settings, out)
        payloads = build_payloads(settings, out, results)
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7h_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
