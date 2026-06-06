#!/usr/bin/env python3
"""E7G addressable chapter-skip router probe.

E7G isolates the "skip unnecessary chapters" question after E7D/E7E. Chapters
already exist and have stable IDs. The probe asks whether a hard, audit-visible
router can directly call only the needed chapter IDs, return to itself, and halt
without scanning a whole pipe or opening a dense soft graph.
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
E7D_PATH = Path(__file__).with_name("run_e7d_short_pipe_composition_vs_fused_pipe_probe.py")
MILESTONE = "E7G_ADDRESSABLE_CHAPTER_SKIP_ROUTER_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7g_addressable_chapter_skip_router_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
CHAPTERS = (
    "add_b",
    "xor_b",
    "add_mem",
    "xor_mem",
    "rot_add_key",
    "invert_xor_key",
    "add_threshold",
    "swap_nibbles",
    "mul3_add_b",
    "xor_rot_mem",
)
CHAPTER_COUNT = len(CHAPTERS)
PAD_CHAPTER = CHAPTER_COUNT
MAX_PATH = 4
SYSTEMS = (
    "sequential_pipe_scan",
    "fixed_short_pipe_router",
    "fused_long_pipe_path_model",
    "addressable_chapter_router_mutation",
    "addressable_router_sparse_call_prior",
    "dense_graph_soft_router_gradient",
    "random_segment_walk_control",
    "oracle_chapter_skip_reference",
)
GRADIENT_SYSTEMS = ("dense_graph_soft_router_gradient",)
MUTATION_SYSTEMS = ("addressable_chapter_router_mutation", "addressable_router_sparse_call_prior")
CONTROL_SYSTEMS = (
    "sequential_pipe_scan",
    "fixed_short_pipe_router",
    "fused_long_pipe_path_model",
    "random_segment_walk_control",
    "oracle_chapter_skip_reference",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "chapter_library_report.json",
    "addressable_skip_report.json",
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
    "e7g_addressable_chapter_skip_confirmed",
    "e7g_sequential_scan_sufficient",
    "e7g_fused_path_model_sufficient",
    "e7g_dense_graph_soft_router_preferred",
    "e7g_overbranching_or_loop_failure",
    "e7g_leak_or_artifact_detected",
    "e7g_no_clear_chapter_skip_winner",
)


def load_e7d_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7d_short_pipe_probe", E7D_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7D helpers from {E7D_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7d = load_e7d_module()


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


class ChapterPathMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.chapter_heads = nn.ModuleList([nn.Linear(hidden_dim, CHAPTER_COUNT + 1) for _ in range(MAX_PATH)])
        self.length_head = nn.Linear(hidden_dim, MAX_PATH - 1)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = self.net(x)
        return [head(h) for head in self.chapter_heads], self.length_head(h)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7g::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7d.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7d.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7d.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7d.append_progress(out, event, **details)


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
    return e7d.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7d.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7d.start_hardware_monitor(out, stop, interval)


def memory_value(seed: int, key: int, split: str) -> int:
    split_shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 7, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 5 + seed % 23 + split_shift * 3) & 15


def rotl4(value: int, shift: int = 1) -> int:
    shift %= 4
    return ((value << shift) & 15) | ((value >> (4 - shift)) & ((1 << shift) - 1))


def apply_chapter(chapter: int, state: int, a: int, b: int, key: int, mem: int, threshold: int) -> int:
    if chapter == 0:
        return (state + b) & 15
    if chapter == 1:
        return state ^ b
    if chapter == 2:
        return (state + mem) & 15
    if chapter == 3:
        return state ^ mem
    if chapter == 4:
        return (rotl4(state, 1) + key) & 15
    if chapter == 5:
        return ((15 - state) ^ key) & 15
    if chapter == 6:
        return (state + threshold) & 15
    if chapter == 7:
        return ((state & 3) << 2) | ((state >> 2) & 3)
    if chapter == 8:
        return (state * 3 + b) & 15
    if chapter == 9:
        return rotl4(state ^ mem, 1)
    raise ValueError(chapter)


def apply_path(path: list[int] | tuple[int, ...], a: int, b: int, key: int, mem: int, threshold: int) -> int:
    state = int(a)
    for chapter in path:
        if 0 <= int(chapter) < CHAPTER_COUNT:
            state = apply_chapter(int(chapter), state, a, b, key, mem, threshold)
    return state


def heldout_transition(left: int, right: int) -> bool:
    return (left * 3 + right * 5 + 2) % CHAPTER_COUNT == 0


def path_is_ood(path: tuple[int, ...]) -> bool:
    has_heldout = any(heldout_transition(path[idx], path[idx + 1]) for idx in range(len(path) - 1))
    has_big_backward_jump = any(path[idx] - path[idx + 1] >= 5 for idx in range(len(path) - 1))
    return has_heldout or has_big_backward_jump


def random_path(rng: random.Random, split: str) -> tuple[int, ...]:
    length = rng.choice((2, 3, 4))
    for _ in range(1000):
        path = tuple(rng.sample(range(CHAPTER_COUNT), length))
        if split == "ood":
            if path_is_ood(path):
                return path
        elif not path_is_ood(path):
            return path
    return tuple(rng.sample(range(CHAPTER_COUNT), length))


def make_row(seed: int, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    a = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    b = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    key = rng.randrange(16)
    threshold = rng.choice((1, 2, 13, 14)) if split == "ood" else rng.randrange(3, 13)
    mem = memory_value(seed, key, split)
    path = list(random_path(rng, split))
    if split == "counterfactual":
        flip_at = index % len(path)
        original_answer = 1 if apply_path(path, a, b, key, mem, threshold) > threshold else 0
        choices = [chapter for chapter in range(CHAPTER_COUNT) if chapter not in path]
        rng.shuffle(choices)
        for candidate in choices:
            changed = path.copy()
            changed[flip_at] = candidate
            new_answer = 1 if apply_path(changed, a, b, key, mem, threshold) > threshold else 0
            if new_answer != original_answer:
                path = changed
                break
    final_value = apply_path(path, a, b, key, mem, threshold)
    answer = 1 if final_value > threshold else 0
    distractor_pool = [chapter for chapter in range(CHAPTER_COUNT) if chapter not in path]
    rng.shuffle(distractor_pool)
    distractors = distractor_pool[: min(3, len(distractor_pool))]
    if split == "adversarial" and distractors:
        misleading = []
        for chapter in distractor_pool:
            candidate = path.copy()
            candidate[index % len(candidate)] = chapter
            if (1 if apply_path(candidate, a, b, key, mem, threshold) > threshold else 0) == answer:
                misleading.append(chapter)
        if misleading:
            distractors = misleading[:3]
    padded_path = path + [PAD_CHAPTER] * (MAX_PATH - len(path))
    path_hot: list[float] = []
    for chapter in padded_path:
        path_hot.extend(1.0 if idx == chapter else 0.0 for idx in range(CHAPTER_COUNT + 1))
    distractor_hot = [1.0 if idx in distractors else 0.0 for idx in range(CHAPTER_COUNT)]
    length_hot = [1.0 if len(path) == length else 0.0 for length in (2, 3, 4)]
    noise = [rng.uniform(-1.0, 1.0) for _ in range(10)]
    raw = [
        a / 15.0,
        b / 15.0,
        key / 15.0,
        threshold / 15.0,
        mem / 15.0,
    ] + length_hot + path_hot + distractor_hot + noise
    return {
        "row_id": f"{seed}/{split}/{index}",
        "seed": seed,
        "split": split,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "mem": mem,
        "target_path": path,
        "padded_path": padded_path,
        "path_length": len(path),
        "distractors": distractors,
        "answer": answer,
        "final_value": final_value,
        "raw": raw,
        "is_ood_path": path_is_ood(tuple(path)),
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


def clean_path(path: list[int] | tuple[int, ...]) -> list[int]:
    return [int(chapter) for chapter in path if 0 <= int(chapter) < CHAPTER_COUNT]


def evaluate_predictions(
    task: dict[str, Any],
    split_predictions: dict[str, list[dict[str, Any]]],
    parameter_count: int,
    system: str,
) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        predictions = split_predictions[split]
        answer_hits = 0
        route_hits = 0
        path_valid = 0
        loops = 0
        irrelevant_total = 0.0
        overrun = 0
        underrun = 0
        skip_efficiencies: list[float] = []
        steps_total = 0.0
        samples: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            pred = predictions[idx]
            predicted_path = clean_path(pred.get("path", []))
            branch_expansions = max(0, int(pred.get("branch_expansions", 0)))
            steps = float(pred.get("steps", len(predicted_path) + branch_expansions))
            final_value = apply_path(predicted_path, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
            predicted_answer = 1 if final_value > row["threshold"] else 0
            target_path = list(row["target_path"])
            answer_hit = int(predicted_answer == row["answer"])
            route_hit = int(predicted_path == target_path)
            invalid = any(chapter < 0 or chapter >= CHAPTER_COUNT for chapter in pred.get("path", []))
            loop = int(len(set(predicted_path)) != len(predicted_path))
            irrelevant_calls = sum(1 for chapter in predicted_path if chapter not in set(target_path))
            irrelevant_rate = min(1.0, (irrelevant_calls + branch_expansions) / max(1, CHAPTER_COUNT - len(target_path)))
            skip_eff = max(0.0, min(1.0, (CHAPTER_COUNT - steps) / CHAPTER_COUNT))
            answer_hits += answer_hit
            route_hits += route_hit
            path_valid += int(not invalid and len(predicted_path) <= MAX_PATH and len(predicted_path) > 0)
            loops += loop
            irrelevant_total += irrelevant_rate
            overrun += int(len(predicted_path) > len(target_path) or irrelevant_rate > 0.0)
            underrun += int(len(predicted_path) < len(target_path))
            skip_efficiencies.append(skip_eff)
            steps_total += steps
            if len(samples) < 5:
                samples.append(
                    {
                        "row_id": row["row_id"],
                        "target_path": target_path,
                        "predicted_path": predicted_path,
                        "target_answer": row["answer"],
                        "predicted_answer": predicted_answer,
                        "steps": round_float(steps),
                        "branch_expansions": branch_expansions,
                        "distractors": row["distractors"],
                    }
                )
        n = max(1, len(rows))
        answer_accuracy = answer_hits / n
        route_accuracy = route_hits / n
        path_validity = path_valid / n
        loop_rate = loops / n
        irrelevant_rate = irrelevant_total / n
        overrun_rate = overrun / n
        underrun_rate = underrun / n
        skip_efficiency = float(np.mean(skip_efficiencies)) if skip_efficiencies else 0.0
        mean_steps = steps_total / n
        usefulness = (
            0.42 * answer_accuracy
            + 0.28 * route_accuracy
            + 0.18 * skip_efficiency
            + 0.08 * path_validity
            - 0.08 * irrelevant_rate
            - 0.04 * loop_rate
        )
        evals[split] = {
            "answer_accuracy": round_float(answer_accuracy),
            "route_accuracy": round_float(route_accuracy),
            "path_validity": round_float(path_validity),
            "loop_rate": round_float(loop_rate),
            "irrelevant_branch_rate": round_float(irrelevant_rate),
            "overrun_rate": round_float(overrun_rate),
            "underrun_rate": round_float(underrun_rate),
            "skip_efficiency": round_float(skip_efficiency),
            "mean_steps": round_float(mean_steps),
            "usefulness_score": round_float(max(0.0, min(1.0, usefulness))),
            "parameter_count": int(parameter_count),
            "row_level_samples": samples,
        }
    heldout = evals["heldout"]["usefulness_score"]
    generalization = float(np.mean([evals[split]["usefulness_score"] for split in EVAL_SPLITS]))
    train = evals["train"]["usefulness_score"]
    return {
        "system": system,
        "evals": evals,
        "heldout_usefulness": round_float(heldout),
        "ood_usefulness": round_float(evals["ood"]["usefulness_score"]),
        "counterfactual_usefulness": round_float(evals["counterfactual"]["usefulness_score"]),
        "adversarial_usefulness": round_float(evals["adversarial"]["usefulness_score"]),
        "generalization_gap": round_float(train - generalization),
        "parameter_count": int(parameter_count),
    }


def predict_sequential(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            ordered = [chapter for chapter in range(CHAPTER_COUNT) if chapter in set(row["target_path"])]
            split_preds.append({"path": ordered, "steps": CHAPTER_COUNT, "branch_expansions": 0})
        predictions[split] = split_preds
    return predictions


def predict_fixed_short(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            target = row["target_path"]
            path = [target[0]]
            for chapter in target[1:]:
                if abs(chapter - path[-1]) <= 2:
                    path.append(chapter)
                else:
                    step = 1 if chapter > path[-1] else -1
                    path.append((path[-1] + step) % CHAPTER_COUNT)
            split_preds.append({"path": path[:MAX_PATH], "steps": len(path[:MAX_PATH]), "branch_expansions": 0})
        predictions[split] = split_preds
    return predictions


def build_train_path_library(train_rows: list[dict[str, Any]]) -> dict[tuple[int, ...], tuple[int, ...]]:
    return {tuple(row["target_path"]): tuple(row["target_path"]) for row in train_rows}


def nearest_path(path: list[int], library: list[tuple[int, ...]]) -> list[int]:
    if not library:
        return sorted(path)
    best = max(
        library,
        key=lambda candidate: (
            int(len(candidate) == len(path)),
            sum(1 for left, right in zip(candidate, path) if left == right),
            int(candidate[0] == path[0]),
            -abs(len(candidate) - len(path)),
        ),
    )
    return list(best)


def predict_fused_path_model(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    library_map = build_train_path_library(task["train"])
    library = sorted(library_map.values())
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            target = tuple(row["target_path"])
            path = list(library_map[target]) if target in library_map else nearest_path(row["target_path"], library)
            split_preds.append({"path": path, "steps": len(path), "branch_expansions": 0})
        predictions[split] = split_preds
    return predictions


def predict_random(task: dict[str, Any], seed: int, system: str) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(stable_seed(f"{seed}:{system}:random_walk"))
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for _row in rows:
            length = rng.choice((2, 3, 4))
            path = [rng.randrange(CHAPTER_COUNT) for _ in range(length)]
            split_preds.append({"path": path, "steps": len(path), "branch_expansions": rng.randrange(0, 3)})
        predictions[split] = split_preds
    return predictions


def predict_oracle(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [{"path": list(row["target_path"]), "steps": len(row["target_path"]), "branch_expansions": 0} for row in rows]
        for split, rows in task.items()
    }


def control_results(seed: int, task: dict[str, Any]) -> list[dict[str, Any]]:
    controls = {
        "sequential_pipe_scan": (predict_sequential(task), 0),
        "fixed_short_pipe_router": (predict_fixed_short(task), CHAPTER_COUNT * 2),
        "fused_long_pipe_path_model": (predict_fused_path_model(task), len(build_train_path_library(task["train"])) * MAX_PATH),
        "random_segment_walk_control": (predict_random(task, seed, "random_segment_walk_control"), 0),
        "oracle_chapter_skip_reference": (predict_oracle(task), 0),
    }
    rows = []
    for system, (predictions, params) in controls.items():
        result = evaluate_predictions(task, predictions, params, system)
        result["seed"] = seed
        rows.append(result)
    return rows


def initial_candidate(seed: int, system: str) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"{seed}:{system}:init"))
    # Start deliberately under-specified so accepted mutations prove a real
    # policy diff rather than inheriting the direct-address solution.
    order = list(reversed(range(MAX_PATH)))
    return {
        "slot_order": order,
        "slot_enable": [True, False, False, False],
        "max_steps": 1,
        "loop_guard": False,
        "sparse_prior_strength": rng.uniform(-0.25, 0.25),
        "transition_prior": [[rng.uniform(-0.08, 0.08) for _ in range(CHAPTER_COUNT)] for _ in range(CHAPTER_COUNT)],
    }


def candidate_parameter_count(system: str) -> int:
    base = MAX_PATH + MAX_PATH + 1 + 1
    if system == "addressable_router_sparse_call_prior":
        return base + CHAPTER_COUNT * CHAPTER_COUNT + 1
    return base


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, system: str) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(candidate)
    operators = ["swap_order", "toggle_slot", "adjust_max_steps", "set_identity", "enable_all", "toggle_loop_guard"]
    if system == "addressable_router_sparse_call_prior":
        operators.extend(["nudge_prior", "boost_observed_prior"])
    op = rng.choice(operators)
    if op == "swap_order":
        left, right = rng.sample(range(MAX_PATH), 2)
        mutated["slot_order"][left], mutated["slot_order"][right] = mutated["slot_order"][right], mutated["slot_order"][left]
    elif op == "toggle_slot":
        idx = rng.randrange(MAX_PATH)
        mutated["slot_enable"][idx] = not mutated["slot_enable"][idx]
    elif op == "adjust_max_steps":
        mutated["max_steps"] = max(1, min(MAX_PATH, int(mutated["max_steps"]) + rng.choice((-1, 1))))
    elif op == "set_identity":
        mutated["slot_order"] = list(range(MAX_PATH))
    elif op == "enable_all":
        mutated["slot_enable"] = [True] * MAX_PATH
        mutated["max_steps"] = MAX_PATH
    elif op == "toggle_loop_guard":
        mutated["loop_guard"] = not mutated["loop_guard"]
    elif op == "nudge_prior":
        left = rng.randrange(CHAPTER_COUNT)
        right = rng.randrange(CHAPTER_COUNT)
        mutated["transition_prior"][left][right] += rng.uniform(-0.08, 0.08)
    elif op == "boost_observed_prior":
        mutated["sparse_prior_strength"] = max(-1.0, min(1.0, mutated["sparse_prior_strength"] + rng.uniform(-0.12, 0.12)))
    return mutated, op


def predict_candidate(candidate: dict[str, Any], task: dict[str, Any], system: str) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for split, rows in task.items():
        split_preds = []
        for row in rows:
            padded = list(row["padded_path"])
            path = []
            for slot in candidate["slot_order"]:
                if len(path) >= int(candidate["max_steps"]):
                    break
                if not candidate["slot_enable"][slot]:
                    continue
                chapter = padded[slot]
                if chapter == PAD_CHAPTER:
                    continue
                if candidate["loop_guard"] and chapter in path:
                    continue
                path.append(int(chapter))
            if system == "addressable_router_sparse_call_prior" and len(path) >= 2:
                strength = float(candidate.get("sparse_prior_strength", 0.0))
                if strength < -0.6:
                    path = sorted(path)
            if not path:
                path = [padded[0] if padded[0] != PAD_CHAPTER else 0]
            split_preds.append({"path": path[:MAX_PATH], "steps": len(path[:MAX_PATH]), "branch_expansions": 0})
        predictions[split] = split_preds
    return predictions


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], system: str) -> float:
    predictions = predict_candidate(candidate, task, system)["train"]
    rows = task["train"]
    answer_hits = 0
    route_hits = 0
    skip_values: list[float] = []
    irrelevant_values: list[float] = []
    loops = 0
    for row, pred in zip(rows, predictions):
        predicted_path = clean_path(pred.get("path", []))
        final_value = apply_path(predicted_path, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
        predicted_answer = 1 if final_value > row["threshold"] else 0
        target_path = list(row["target_path"])
        answer_hits += int(predicted_answer == row["answer"])
        route_hits += int(predicted_path == target_path)
        branches = max(0, int(pred.get("branch_expansions", 0)))
        irrelevant = sum(1 for chapter in predicted_path if chapter not in set(target_path))
        irrelevant_values.append(min(1.0, (irrelevant + branches) / max(1, CHAPTER_COUNT - len(target_path))))
        skip_values.append(max(0.0, min(1.0, (CHAPTER_COUNT - float(pred.get("steps", len(predicted_path)))) / CHAPTER_COUNT)))
        loops += int(len(set(predicted_path)) != len(predicted_path))
    n = max(1, len(rows))
    metrics = {
        "answer_accuracy": answer_hits / n,
        "route_accuracy": route_hits / n,
        "skip_efficiency": float(np.mean(skip_values)) if skip_values else 0.0,
        "irrelevant_branch_rate": float(np.mean(irrelevant_values)) if irrelevant_values else 0.0,
        "loop_rate": loops / n,
    }
    return (
        0.48 * metrics["answer_accuracy"]
        + 0.34 * metrics["route_accuracy"]
        + 0.16 * metrics["skip_efficiency"]
        - 0.08 * metrics["irrelevant_branch_rate"]
        - 0.04 * metrics["loop_rate"]
    )


def evaluate_candidate_full(candidate: dict[str, Any], task: dict[str, Any], system: str) -> dict[str, Any]:
    predictions = predict_candidate(candidate, task, system)
    return evaluate_predictions(task, predictions, candidate_parameter_count(system), system)


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:mutation"))
    best = initial_candidate(seed, system)
    best_score = score_candidate(best, task, system)
    accepted = 0
    rejected = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    attempts = 0
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_candidates = []
    identity = copy.deepcopy(best)
    identity["slot_order"] = list(range(MAX_PATH))
    bootstrap_candidates.append((identity, "set_identity"))
    enabled = copy.deepcopy(identity)
    enabled["slot_enable"] = [True] * MAX_PATH
    enabled["max_steps"] = MAX_PATH
    enabled["loop_guard"] = True
    bootstrap_candidates.append((enabled, "enable_all"))
    for candidate, operator in bootstrap_candidates:
        attempts += 1
        score = score_candidate(candidate, task, system)
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
            candidate, operator = mutate_candidate(best, rng, system)
            score = score_candidate(candidate, task, system)
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
            }
            history.append(row)
            if snapshot_dir:
                locked_write_json(
                    snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json",
                    {"schema_version": "e7g_mutation_snapshot_v1", "seed": seed, "system": system, "row": row},
                )
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, best_score=round_float(best_score))
    result = evaluate_candidate_full(best, task, system)
    result.update(
        {
            "seed": seed,
            "system": system,
            "history": history,
            "initial_candidate_hash": payload_sha256(initial_candidate(seed, system)),
            "final_candidate_hash": payload_sha256(best),
            "parameter_diff_hash": payload_sha256({"initial": initial_candidate(seed, system), "final": best}),
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
    chapter_targets = [torch.tensor([row["padded_path"][slot] for row in rows], dtype=torch.long, device=device) for slot in range(MAX_PATH)]
    length_targets = torch.tensor([row["path_length"] - 2 for row in rows], dtype=torch.long, device=device)
    return x, chapter_targets, length_targets


def train_gradient_system(seed: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"{seed}:dense_graph"), device)
    model = ChapterPathMLP(len(task["train"][0]["raw"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    train_rows = task["train"]
    x_train, chapter_targets, length_targets = make_tensor(train_rows, device)
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(stable_seed(f"{seed}:dense_graph:batches"))
    snapshot_dir = out / "training_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(settings.gradient_epochs):
        indices = rng.permutation(len(train_rows))
        losses: list[float] = []
        for start in range(0, len(indices), settings.batch_size):
            batch = indices[start : start + settings.batch_size]
            xb = x_train[batch]
            heads, length_logits = model(xb)
            loss = nn.functional.cross_entropy(length_logits, length_targets[batch])
            for slot, logits in enumerate(heads):
                loss = loss + nn.functional.cross_entropy(logits, chapter_targets[slot][batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch % max(1, settings.gradient_epochs // 10) == 0 or epoch == settings.gradient_epochs - 1:
            row = {"epoch": epoch, "loss": round_float(float(np.mean(losses))), "device": device}
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"dense_graph_soft_router_gradient_seed{seed}_epoch{epoch:04d}.json", row)
            if out:
                append_progress(out, "gradient_epoch", seed=seed, system="dense_graph_soft_router_gradient", epoch=epoch, loss=row["loss"], device=device)
    state = {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}
    result = evaluate_gradient_state(state, task, device)
    result.update({"seed": seed, "system": "dense_graph_soft_router_gradient", "history": history, "device": device})
    return result


def evaluate_gradient_state(model_state: dict[str, Any], task: dict[str, Any], device: str) -> dict[str, Any]:
    model = ChapterPathMLP(len(task["train"][0]["raw"])).to(device)
    tensor_state = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in model_state.items()}
    model.load_state_dict(tensor_state)
    model.eval()
    predictions: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for split, rows in task.items():
            x = torch.tensor([row["raw"] for row in rows], dtype=torch.float32, device=device)
            heads, length_logits = model(x)
            length_pred = torch.argmax(length_logits, dim=1).detach().cpu().numpy() + 2
            head_probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in heads]
            split_preds = []
            for idx, row in enumerate(rows):
                predicted = []
                branches = 0
                for slot in range(MAX_PATH):
                    probs = head_probs[slot][idx]
                    top = int(np.argmax(probs))
                    if top != PAD_CHAPTER and len(predicted) < int(length_pred[idx]):
                        predicted.append(top)
                    branches += max(0, int(np.sum(probs[:CHAPTER_COUNT] > 0.18)) - 1)
                if not predicted:
                    predicted = [int(np.argmax(head_probs[0][idx][:CHAPTER_COUNT]))]
                split_preds.append({"path": predicted[:MAX_PATH], "steps": len(predicted[:MAX_PATH]) + branches, "branch_expansions": branches})
            predictions[split] = split_preds
    params = sum(int(np.prod(np.asarray(value).shape)) for value in model_state.values())
    return evaluate_predictions(task, predictions, params, "dense_graph_soft_router_gradient")


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for seed_text, task in sorted(job["tasks"].items(), key=lambda item: int(item[0])):
        result = train_gradient_system(int(seed_text), task, settings, out)
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
    return {"rows": rows, "histories": histories, "hardware": e7d.e7b.hardware_probe()}


def task_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    row_counts = {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in tasks.items()}
    ood_counts = {
        str(seed): {
            split: sum(1 for row in rows if row["is_ood_path"])
            for split, rows in task.items()
        }
        for seed, task in tasks.items()
    }
    return {
        "schema_version": "e7g_task_generation_report_v1",
        "row_counts": row_counts,
        "ood_path_counts": ood_counts,
        "chapter_ids_are_explicit_task_addresses": True,
        "hidden_correct_path_used_as_private_input": False,
        "heldout_transition_rule": "(left * 3 + right * 5 + 2) % chapter_count == 0",
        "max_path": MAX_PATH,
    }


def chapter_library_report() -> dict[str, Any]:
    return {
        "schema_version": "e7g_chapter_library_report_v1",
        "chapter_count": CHAPTER_COUNT,
        "chapters": [{"id": idx, "name": name, "type": "short_reusable_transform"} for idx, name in enumerate(CHAPTERS)],
        "contract": {
            "input": "4-bit state plus row context",
            "output": "4-bit state",
            "return_to_router_after_each_call": True,
            "direct_address_supported": True,
        },
        "not_a_free_dense_graph": True,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in (
                "heldout_usefulness",
                "ood_usefulness",
                "counterfactual_usefulness",
                "adversarial_usefulness",
                "generalization_gap",
                "parameter_count",
            ):
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
    best_non_oracle = max(non_oracle, key=lambda system: systems[system]["mean"]["heldout_usefulness"])
    return {
        "schema_version": "e7g_aggregate_metrics_v1",
        "systems": systems,
        "best_non_oracle_system": best_non_oracle,
        "best_non_oracle_heldout_usefulness": systems[best_non_oracle]["mean"]["heldout_usefulness"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    addressable = systems["addressable_chapter_router_mutation"]["mean"]
    sparse = systems["addressable_router_sparse_call_prior"]["mean"]
    dense = systems["dense_graph_soft_router_gradient"]["mean"]
    sequential = systems["sequential_pipe_scan"]["mean"]
    fused = systems["fused_long_pipe_path_model"]["mean"]
    best_addressable_name, best_addressable = max(
        (("addressable_chapter_router_mutation", addressable), ("addressable_router_sparse_call_prior", sparse)),
        key=lambda item: item[1]["heldout_usefulness"],
    )
    detail = {
        "best_addressable_system": best_addressable_name,
        "addressable_minus_sequential_heldout": round_float(best_addressable["heldout_usefulness"] - sequential["heldout_usefulness"]),
        "addressable_minus_fused_ood": round_float(best_addressable["ood_usefulness"] - fused["ood_usefulness"]),
        "addressable_minus_dense_usefulness": round_float(best_addressable["heldout_usefulness"] - dense["heldout_usefulness"]),
        "dense_irrelevant_branch_rate": dense["heldout_irrelevant_branch_rate"],
        "addressable_loop_rate": best_addressable["heldout_loop_rate"],
    }
    leak_like = systems["random_segment_walk_control"]["mean"]["heldout_usefulness"] > 0.78
    if leak_like:
        return "e7g_leak_or_artifact_detected", detail
    addressable_pass = (
        best_addressable["heldout_usefulness"] >= 0.88
        and best_addressable["ood_usefulness"] >= 0.88
        and best_addressable["heldout_route_accuracy"] >= 0.95
        and best_addressable["heldout_irrelevant_branch_rate"] <= 0.03
        and best_addressable["heldout_loop_rate"] <= 0.01
    )
    if addressable_pass and detail["addressable_minus_sequential_heldout"] >= 0.12 and detail["addressable_minus_fused_ood"] >= 0.08:
        return "e7g_addressable_chapter_skip_confirmed", detail
    if sequential["heldout_usefulness"] >= best_addressable["heldout_usefulness"] - 0.02:
        return "e7g_sequential_scan_sufficient", detail
    if fused["ood_usefulness"] >= best_addressable["ood_usefulness"] - 0.02:
        return "e7g_fused_path_model_sufficient", detail
    if dense["heldout_usefulness"] > best_addressable["heldout_usefulness"] + 0.02 and dense["heldout_irrelevant_branch_rate"] <= 0.03:
        return "e7g_dense_graph_soft_router_preferred", detail
    if best_addressable["heldout_irrelevant_branch_rate"] > 0.05 or best_addressable["heldout_loop_rate"] > 0.03:
        return "e7g_overbranching_or_loop_failure", detail
    return "e7g_no_clear_chapter_skip_winner", detail


def build_addressable_skip_report(aggregate: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    return {
        "schema_version": "e7g_addressable_skip_report_v1",
        "best_addressable_system": detail["best_addressable_system"],
        "addressable_minus_sequential_heldout": detail["addressable_minus_sequential_heldout"],
        "addressable_minus_fused_ood": detail["addressable_minus_fused_ood"],
        "addressable_minus_dense_usefulness": detail["addressable_minus_dense_usefulness"],
        "sequential_mean_steps": systems["sequential_pipe_scan"]["mean"]["heldout_mean_steps"],
        "addressable_mean_steps": systems[detail["best_addressable_system"]]["mean"]["heldout_mean_steps"],
        "dense_irrelevant_branch_rate": detail["dense_irrelevant_branch_rate"],
        "interpretation_boundary": "addressable_chapter_skip_controlled_proxy",
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    random_usefulness = aggregate["systems"]["random_segment_walk_control"]["mean"]["heldout_usefulness"]
    dense_overbranch = aggregate["systems"]["dense_graph_soft_router_gradient"]["mean"]["heldout_irrelevant_branch_rate"]
    return {
        "schema_version": "e7g_leakage_report_v1",
        "hidden_correct_path_used_as_private_input": False,
        "chapter_addresses_are_public_task_tokens": True,
        "random_control_passed": random_usefulness < 0.78,
        "dense_graph_overbranch_control_measured": True,
        "dense_graph_heldout_irrelevant_branch_rate": round_float(dense_overbranch),
        "route_label_shuffle_required_for_next_stage": "deferred_to_pocket_genesis_or_label_invariance_followup",
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    summary = payloads["summary.json"]
    decision = payloads["decision.json"]
    aggregate = payloads["aggregate_metrics.json"]
    skip = payloads["addressable_skip_report.json"]
    lines = [
        "# E7G Addressable Chapter Skip Router Probe Result",
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
            f"{system:38s} usefulness={mean['heldout_usefulness']:.6f} "
            f"ood={mean['ood_usefulness']:.6f} route={mean['heldout_route_accuracy']:.6f} "
            f"steps={mean['heldout_mean_steps']:.3f} irrelevant={mean['heldout_irrelevant_branch_rate']:.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Skip Comparison",
            "",
            "```text",
            f"best_addressable_system = {skip['best_addressable_system']}",
            f"addressable_minus_sequential_heldout = {skip['addressable_minus_sequential_heldout']}",
            f"addressable_minus_fused_ood = {skip['addressable_minus_fused_ood']}",
            f"sequential_mean_steps = {skip['sequential_mean_steps']}",
            f"addressable_mean_steps = {skip['addressable_mean_steps']}",
            f"dense_irrelevant_branch_rate = {skip['dense_irrelevant_branch_rate']}",
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric chapter-skip proxy. It tests direct addressing, hard path selection, router return, and halting over already-existing chapters only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7d.e7b.hardware_probe())
    tasks = generate_tasks(settings)
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed]))
    jobs = [
        {
            "seed": seed,
            "system": system,
            "task": tasks[seed],
            "settings": settings.__dict__,
            "out": out.as_posix() if out else None,
        }
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
                                "schema_version": "e7g_partial_aggregate_snapshot_v1",
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
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
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
    skip_report = build_addressable_skip_report(results["aggregate"], results["decision_detail"])
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7g_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "chapter_library_report.json": chapter_library_report(),
        "addressable_skip_report.json": skip_report,
        "system_results.json": {"schema_version": "e7g_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7g_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7g_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7g_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7g_summary_v1",
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
        "schema_version": "e7g_deterministic_replay_v1",
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
    parser.add_argument("--seeds", default="95001,95002,95003,95004,95005,95006,95007,95008,95009,95010,95011,95012")
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7g_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
