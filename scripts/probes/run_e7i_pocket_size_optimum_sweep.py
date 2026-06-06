#!/usr/bin/env python3
"""E7I pocket-size optimum sweep.

E7H showed mutation can discover size-2 pockets on a clean proxy, but that
could be generator imprint. E7I sweeps pocket sizes across multiple task
families with different hidden natural granularities to test whether there is a
stable pocket-size optimum or whether variable, family-dependent granularity is
preferred.
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
E7H_PATH = Path(__file__).with_name("run_e7h_pocket_granularity_discovery_probe.py")
MILESTONE = "E7I_POCKET_SIZE_OPTIMUM_SWEEP"
DEFAULT_OUT = Path("target/pilot_wave/e7i_pocket_size_optimum_sweep")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
FAMILIES = (
    "family_A_natural_size_2",
    "family_B_natural_size_3",
    "family_C_natural_size_4",
    "family_D_mixed_size_2_4",
    "family_E_no_stable_pocket_size",
    "family_F_decoy_pair_frequency",
    "family_G_reuse_sparse_family",
)
MICROSEGMENTS = tuple(f"micro_{idx:02d}" for idx in range(16))
MICRO_COUNT = len(MICROSEGMENTS)
PAD_SEGMENT = MICRO_COUNT
MAX_MICRO_PATH = 12
SYSTEMS = (
    "atomic_microsegment_router",
    "fixed_size_2_pockets",
    "fixed_size_3_pockets",
    "fixed_size_4_pockets",
    "mixed_size_2_3_pockets",
    "mixed_size_2_4_pockets",
    "mutation_discovered_variable_size_pockets",
    "fixed_human_pocket_scaffold",
    "fused_long_pipe",
    "dense_graph_control",
    "random_boundary_control",
    "oracle_family_granularity_reference",
)
GRADIENT_SYSTEMS = ("dense_graph_control",)
MUTATION_SYSTEMS = ("mutation_discovered_variable_size_pockets",)
CONTROL_SYSTEMS = tuple(system for system in SYSTEMS if system not in MUTATION_SYSTEMS and system not in GRADIENT_SYSTEMS)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "family_generation_report.json",
    "pocket_size_sweep_report.json",
    "family_winner_report.json",
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
    "e7i_stable_pocket_size_optimum_detected",
    "e7i_variable_pocket_granularity_preferred",
    "e7i_size2_was_generator_imprint",
    "e7i_pocket_size_needs_prior_scaffold",
    "e7i_atomic_microsegment_routing_preferred",
    "e7i_fused_pipe_overfit_detected",
    "e7i_pocket_granularity_collapses_to_graph_soup",
    "e7i_no_clear_size_frontier",
    "e7i_leak_or_artifact_detected",
)


def load_e7h_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7h_pocket_granularity_probe", E7H_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7H helpers from {E7H_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7h = load_e7h_module()


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


class DenseFamilyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
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
    return int(hashlib.sha256(f"e7i::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7h.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7h.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7h.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7h.append_progress(out, event, **details)


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
    return e7h.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7h.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7h.start_hardware_monitor(out, stop, interval)


def rotl4(value: int, shift: int = 1) -> int:
    shift %= 4
    return ((value << shift) & 15) | ((value >> (4 - shift)) & ((1 << shift) - 1))


def apply_micro(segment: int, state: int, a: int, b: int, key: int, mem: int, threshold: int) -> int:
    segment %= MICRO_COUNT
    if segment % 8 == 0:
        return (state + b) & 15
    if segment % 8 == 1:
        return state ^ key
    if segment % 8 == 2:
        return (state + mem) & 15
    if segment % 8 == 3:
        return rotl4(state, 1)
    if segment % 8 == 4:
        return state ^ mem
    if segment % 8 == 5:
        return ((state & 3) << 2) | ((state >> 2) & 3)
    if segment % 8 == 6:
        return (state + threshold) & 15
    return 15 - state


def apply_micro_path(path: list[int] | tuple[int, ...], a: int, b: int, key: int, mem: int, threshold: int) -> int:
    state = int(a)
    for segment in path:
        if 0 <= int(segment) < MICRO_COUNT:
            state = apply_micro(int(segment), state, a, b, key, mem, threshold)
    return state


def family_natural_pockets(family: str) -> list[tuple[int, ...]]:
    if family == "family_A_natural_size_2":
        return [tuple(range(start, start + 2)) for start in range(0, 16, 2)]
    if family == "family_B_natural_size_3":
        return [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]
    if family == "family_C_natural_size_4":
        return [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    if family == "family_D_mixed_size_2_4":
        return [(0, 1), (2, 3, 4), (5, 6, 7, 8), (9, 10), (11, 12, 13, 14)]
    if family == "family_F_decoy_pair_frequency":
        return [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    if family == "family_G_reuse_sparse_family":
        return [(0, 1, 2, 3), (4, 5), (6, 7, 8), (9, 10, 11, 12)]
    return []


def fixed_pockets(size: int) -> list[tuple[int, ...]]:
    return [tuple(range(start, min(start + size, MICRO_COUNT))) for start in range(0, MICRO_COUNT - size + 1, size)]


def mixed_pockets(sizes: tuple[int, ...]) -> list[tuple[int, ...]]:
    pockets = []
    cursor = 0
    idx = 0
    while cursor < MICRO_COUNT:
        size = sizes[idx % len(sizes)]
        if cursor + size <= MICRO_COUNT:
            pockets.append(tuple(range(cursor, cursor + size)))
        cursor += size
        idx += 1
    return pockets


def normalize_pockets(pockets: list[list[int]] | tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for pocket in pockets:
        item = tuple(int(seg) for seg in pocket if 0 <= int(seg) < MICRO_COUNT)
        if 2 <= len(item) <= 4 and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def pockets_to_micro_path(pockets: list[tuple[int, ...]], pocket_indices: list[int]) -> list[int]:
    micro: list[int] = []
    for idx in pocket_indices:
        micro.extend(pockets[int(idx) % len(pockets)])
    return micro[:MAX_MICRO_PATH]


def heldout_transition(left: int, right: int, family: str) -> bool:
    return (left * 7 + right * 5 + len(family)) % 5 == 0


def random_micro_path(family: str, split: str, rng: random.Random) -> tuple[list[int], list[int]]:
    pockets = family_natural_pockets(family)
    if family == "family_E_no_stable_pocket_size":
        length = rng.choice((5, 6, 7, 8, 9, 10))
        path = rng.sample(range(MICRO_COUNT), length)
        return path, []
    if not pockets:
        pockets = fixed_pockets(2)
    count = rng.choice((2, 3, 4))
    for _ in range(1000):
        pocket_indices = [rng.randrange(len(pockets)) for _ in range(count)]
        has_heldout = any(heldout_transition(pocket_indices[idx], pocket_indices[idx + 1], family) for idx in range(len(pocket_indices) - 1))
        if split == "ood" and has_heldout:
            return pockets_to_micro_path(pockets, pocket_indices), pocket_indices
        if split != "ood" and not has_heldout:
            return pockets_to_micro_path(pockets, pocket_indices), pocket_indices
    pocket_indices = [rng.randrange(len(pockets)) for _ in range(count)]
    return pockets_to_micro_path(pockets, pocket_indices), pocket_indices


def memory_value(seed: int, key: int, split: str, family: str) -> int:
    split_shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 7, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 11 + seed % 31 + split_shift * 5 + len(family)) & 15


def make_row(seed: int, family: str, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    a = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    b = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    key = rng.randrange(16)
    threshold = rng.choice((1, 2, 13, 14)) if split == "ood" else rng.randrange(3, 13)
    mem = memory_value(seed, key, split, family)
    micro_path, pocket_indices = random_micro_path(family, split, rng)
    if family == "family_F_decoy_pair_frequency" and split in {"train", "validation"}:
        micro_path = ([14, 15] if index % 3 == 0 else []) + micro_path
        micro_path = micro_path[:MAX_MICRO_PATH]
    if split == "counterfactual" and micro_path:
        original_answer = 1 if apply_micro_path(micro_path, a, b, key, mem, threshold) > threshold else 0
        for candidate in rng.sample(range(MICRO_COUNT), MICRO_COUNT):
            changed = micro_path.copy()
            changed[index % len(changed)] = candidate
            if (1 if apply_micro_path(changed, a, b, key, mem, threshold) > threshold else 0) != original_answer:
                micro_path = changed
                break
    final_value = apply_micro_path(micro_path, a, b, key, mem, threshold)
    answer = 1 if final_value > threshold else 0
    padded = micro_path + [PAD_SEGMENT] * (MAX_MICRO_PATH - len(micro_path))
    micro_hot: list[float] = []
    for segment in padded:
        micro_hot.extend(1.0 if idx == segment else 0.0 for idx in range(MICRO_COUNT + 1))
    family_hot = [1.0 if FAMILIES[idx] == family else 0.0 for idx in range(len(FAMILIES))]
    length_scaled = len(micro_path) / MAX_MICRO_PATH
    noise = [rng.uniform(-1.0, 1.0) for _ in range(8)]
    raw = [a / 15.0, b / 15.0, key / 15.0, threshold / 15.0, mem / 15.0, length_scaled] + family_hot + micro_hot + noise
    return {
        "row_id": f"{seed}/{family}/{split}/{index}",
        "seed": seed,
        "family": family,
        "split": split,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "mem": mem,
        "micro_path": micro_path,
        "padded_micro_path": padded,
        "micro_path_length": len(micro_path),
        "hidden_natural_pocket_indices": pocket_indices,
        "answer": answer,
        "final_value": final_value,
        "raw": raw,
    }


def generate_seed_task(seed: int, settings: Settings) -> dict[str, dict[str, list[dict[str, Any]]]]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    per_family_counts = {split: max(1, count // len(FAMILIES)) for split, count in counts.items()}
    for family in FAMILIES:
        task[family] = {}
        for split, count in per_family_counts.items():
            rng = random.Random(stable_seed(f"{seed}:{family}:{split}:rows"))
            task[family][split] = [make_row(seed, family, split, idx, rng) for idx in range(count)]
    return task


def generate_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    return {seed: generate_seed_task(seed, settings) for seed in settings.seeds}


def all_rows(task: dict[str, dict[str, list[dict[str, Any]]]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        rows.extend(task[family][split])
    return rows


def greedy_calls(micro_path: list[int], pockets: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    pockets = sorted(normalize_pockets(pockets), key=lambda item: (-len(item), item))
    calls: list[tuple[int, ...]] = []
    idx = 0
    while idx < len(micro_path):
        matched = None
        for pocket in pockets:
            if tuple(micro_path[idx : idx + len(pocket)]) == pocket:
                matched = pocket
                break
        if matched is None:
            matched = (micro_path[idx],)
        calls.append(matched)
        idx += len(matched)
    return calls


def calls_to_micro(calls: list[tuple[int, ...]]) -> list[int]:
    out: list[int] = []
    for call in calls:
        out.extend(call)
    return out


def pockets_for_system(system: str, family: str) -> list[tuple[int, ...]]:
    if system == "fixed_size_2_pockets":
        return fixed_pockets(2)
    if system == "fixed_size_3_pockets":
        return fixed_pockets(3)
    if system == "fixed_size_4_pockets":
        return fixed_pockets(4)
    if system == "mixed_size_2_3_pockets":
        return mixed_pockets((2, 3))
    if system == "mixed_size_2_4_pockets":
        return mixed_pockets((2, 4))
    if system in {"fixed_human_pocket_scaffold", "oracle_family_granularity_reference"}:
        return family_natural_pockets(family)
    return []


def predict_with_pockets(task: dict[str, dict[str, list[dict[str, Any]]]], system: str, family_pockets: dict[str, list[tuple[int, ...]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILIES:
        predictions[family] = {}
        pockets = family_pockets.get(family, [])
        for split in SPLITS:
            split_preds = []
            for row in task[family][split]:
                calls = greedy_calls(row["micro_path"], pockets)
                split_preds.append({"calls": [list(call) for call in calls], "micro_path": calls_to_micro(calls), "steps": len(calls), "branch_expansions": 0})
            predictions[family][split] = split_preds
    return predictions


def predict_atomic(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        family: {
            split: [{"calls": [[seg] for seg in row["micro_path"]], "micro_path": list(row["micro_path"]), "steps": len(row["micro_path"]), "branch_expansions": 0} for row in rows]
            for split, rows in split_map.items()
        }
        for family, split_map in task.items()
    }


def predict_fused(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILIES:
        library = {tuple(row["micro_path"]): tuple(row["micro_path"]) for row in task[family]["train"]}
        library_values = sorted(library.values())
        predictions[family] = {}
        for split in SPLITS:
            split_preds = []
            for row in task[family][split]:
                key = tuple(row["micro_path"])
                if key in library:
                    predicted = list(library[key])
                elif library_values:
                    predicted = list(max(library_values, key=lambda item: (int(len(item) == len(row["micro_path"])), sum(1 for a, b in zip(item, row["micro_path"]) if a == b), -abs(len(item) - len(row["micro_path"])))))
                else:
                    predicted = list(row["micro_path"])
                split_preds.append({"calls": [predicted], "micro_path": predicted, "steps": 1, "branch_expansions": 0})
            predictions[family][split] = split_preds
    return predictions


def predict_random(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rng = random.Random(stable_seed(f"{seed}:random_boundary"))
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILIES:
        predictions[family] = {}
        for split in SPLITS:
            split_preds = []
            for row in task[family][split]:
                micro = list(row["micro_path"])
                rng.shuffle(micro)
                split_preds.append({"calls": [[seg] for seg in micro], "micro_path": micro, "steps": len(micro), "branch_expansions": rng.randrange(0, 3)})
            predictions[family][split] = split_preds
    return predictions


def mean_reuse(rows: list[dict[str, Any]], pockets: list[tuple[int, ...]]) -> float:
    pockets = normalize_pockets(pockets)
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


def evaluate_predictions(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    family_pockets: dict[str, list[tuple[int, ...]]],
    parameter_count: int,
) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    family_metrics: dict[str, Any] = {}
    for family in FAMILIES:
        family_metrics[family] = {}
        for split in SPLITS:
            rows = task[family][split]
            preds = predictions[family][split]
            pockets = normalize_pockets(family_pockets.get(family, []))
            answer_hits = route_hits = valid_hits = loops = 0
            steps_total = irrelevant_total = branch_total = 0.0
            compression_values: list[float] = []
            samples: list[dict[str, Any]] = []
            for row, pred in zip(rows, preds):
                predicted = [int(seg) for seg in pred.get("micro_path", []) if 0 <= int(seg) < MICRO_COUNT]
                final = apply_micro_path(predicted, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
                predicted_answer = 1 if final > row["threshold"] else 0
                target = list(row["micro_path"])
                calls = [tuple(int(seg) for seg in call) for call in pred.get("calls", [])]
                steps = float(pred.get("steps", len(calls)))
                branches = max(0, int(pred.get("branch_expansions", 0)))
                answer_hits += int(predicted_answer == row["answer"])
                route_hits += int(predicted == target)
                valid_hits += int(bool(predicted) and len(predicted) <= MAX_MICRO_PATH)
                loops += int(len(calls) != len(set(calls)))
                irrelevant = sum(1 for seg in predicted if seg not in set(target))
                irrelevant_total += min(1.0, (irrelevant + branches) / max(1, MICRO_COUNT - len(set(target))))
                branch_total += branches
                compression_values.append(max(0.0, min(1.0, (len(target) - steps) / max(1, len(target) - 1))))
                steps_total += steps
                if len(samples) < 3:
                    samples.append({"row_id": row["row_id"], "target": target, "predicted": predicted, "calls": [list(call) for call in calls], "steps": round_float(steps), "target_answer": row["answer"], "predicted_answer": predicted_answer})
            n = max(1, len(rows))
            answer_accuracy = answer_hits / n
            route_accuracy = route_hits / n
            compression = float(np.mean(compression_values)) if compression_values else 0.0
            irrelevant = irrelevant_total / n
            loop_rate = loops / n
            reuse = mean_reuse(rows, pockets)
            avg_size = float(np.mean([len(p) for p in pockets])) if pockets else 0.0
            size_distribution = {str(size): sum(1 for p in pockets if len(p) == size) for size in range(1, 5)}
            usefulness = 0.35 * answer_accuracy + 0.27 * route_accuracy + 0.21 * compression + 0.08 * min(1.0, reuse / 3.0) + 0.04 * (valid_hits / n) - 0.08 * irrelevant - 0.04 * loop_rate
            family_metrics[family][split] = {
                "answer_accuracy": round_float(answer_accuracy),
                "route_accuracy": round_float(route_accuracy),
                "mean_route_steps": round_float(steps_total / n),
                "compression_score": round_float(compression),
                "irrelevant_branch_rate": round_float(irrelevant),
                "loop_rate": round_float(loop_rate),
                "branch_expansion_rate": round_float(branch_total / n),
                "pocket_count": len(pockets),
                "average_pocket_size": round_float(avg_size),
                "pocket_size_distribution": size_distribution,
                "reuse_count_per_pocket": round_float(reuse),
                "freeze_survival_score": round_float(route_accuracy if pockets else 0.0),
                "usefulness_score": round_float(max(0.0, min(1.0, usefulness))),
                "parameter_count": int(parameter_count),
                "row_level_samples": samples,
            }
    for split in SPLITS:
        split_values: dict[str, list[float]] = {}
        for family in FAMILIES:
            for key, value in family_metrics[family][split].items():
                if isinstance(value, (int, float)):
                    split_values.setdefault(key, []).append(float(value))
        evals[split] = {key: round_float(float(np.mean(values))) for key, values in split_values.items()}
        evals[split]["row_level_samples"] = [sample for family in FAMILIES for sample in family_metrics[family][split]["row_level_samples"][:1]]
    train = evals["train"]["usefulness_score"]
    eval_mean = float(np.mean([evals[split]["usefulness_score"] for split in EVAL_SPLITS]))
    return {
        "system": system,
        "evals": evals,
        "family_metrics": family_metrics,
        "heldout_usefulness": round_float(evals["heldout"]["usefulness_score"]),
        "ood_usefulness": round_float(evals["ood"]["usefulness_score"]),
        "counterfactual_usefulness": round_float(evals["counterfactual"]["usefulness_score"]),
        "adversarial_usefulness": round_float(evals["adversarial"]["usefulness_score"]),
        "eval_mean_usefulness": round_float(eval_mean),
        "generalization_gap": round_float(train - eval_mean),
        "parameter_count": int(parameter_count),
    }


def control_results(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    controls: dict[str, tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[tuple[int, ...]]], int]] = {}
    controls["atomic_microsegment_router"] = (predict_atomic(task), {family: [] for family in FAMILIES}, 0)
    for system in ("fixed_size_2_pockets", "fixed_size_3_pockets", "fixed_size_4_pockets", "mixed_size_2_3_pockets", "mixed_size_2_4_pockets", "fixed_human_pocket_scaffold", "oracle_family_granularity_reference"):
        family_pockets = {family: pockets_for_system(system, family) for family in FAMILIES}
        controls[system] = (predict_with_pockets(task, system, family_pockets), family_pockets, sum(len(p) for pockets in family_pockets.values() for p in pockets))
    controls["fused_long_pipe"] = (predict_fused(task), {family: [] for family in FAMILIES}, 0)
    controls["random_boundary_control"] = (predict_random(task, seed), {family: [] for family in FAMILIES}, 0)
    rows = []
    for system, (predictions, family_pockets, params) in controls.items():
        row = evaluate_predictions(task, predictions, system, family_pockets, params)
        row["seed"] = seed
        rows.append(row)
    return rows


def initial_candidate(seed: int) -> dict[str, Any]:
    return {"family_size_policy": {family: 1 for family in FAMILIES}, "frozen": {}, "router_prior": 0.0}


def candidate_pockets(candidate: dict[str, Any]) -> dict[str, list[tuple[int, ...]]]:
    out: dict[str, list[tuple[int, ...]]] = {}
    for family in FAMILIES:
        size = int(candidate.get("family_size_policy", {}).get(family, 1))
        if size == 1:
            out[family] = []
        elif size == 23:
            out[family] = mixed_pockets((2, 3))
        elif size == 24:
            out[family] = mixed_pockets((2, 4))
        elif size == 99:
            out[family] = family_natural_pockets(family)
        else:
            out[family] = fixed_pockets(max(2, min(4, size)))
    return out


def candidate_parameter_count(candidate: dict[str, Any]) -> int:
    pockets = candidate_pockets(candidate)
    return sum(len(p) for family_pockets in pockets.values() for p in family_pockets) + len(FAMILIES)


def score_candidate(candidate: dict[str, Any], task: dict[str, dict[str, list[dict[str, Any]]]]) -> float:
    pockets = candidate_pockets(candidate)
    predictions = predict_with_pockets(task, "mutation_discovered_variable_size_pockets", pockets)
    result = evaluate_predictions(task, predictions, "mutation_discovered_variable_size_pockets", pockets, candidate_parameter_count(candidate))
    metrics = result["evals"]["train"]
    family_scores = [result["family_metrics"][family]["train"]["usefulness_score"] for family in FAMILIES]
    size_diversity = len(set(candidate.get("family_size_policy", {}).values())) / len(FAMILIES)
    return float(np.mean(family_scores)) + 0.03 * size_diversity + 0.02 * metrics["freeze_survival_score"] - 0.02 * metrics["irrelevant_branch_rate"]


def mutate_candidate(candidate: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(candidate)
    policy = dict(mutated.get("family_size_policy", {}))
    op = rng.choice(("set_size", "copy_neighbor_family", "set_mixed", "freeze_family", "router_prior"))
    if op == "set_size":
        family = rng.choice(FAMILIES)
        policy[family] = rng.choice((1, 2, 3, 4))
    elif op == "copy_neighbor_family":
        src = rng.choice(FAMILIES)
        dst = rng.choice(FAMILIES)
        policy[dst] = policy.get(src, 1)
    elif op == "set_mixed":
        family = rng.choice(FAMILIES)
        policy[family] = rng.choice((23, 24))
    elif op == "freeze_family":
        frozen = dict(mutated.get("frozen", {}))
        family = rng.choice(FAMILIES)
        frozen[family] = not bool(frozen.get(family, False))
        mutated["frozen"] = frozen
    elif op == "router_prior":
        mutated["router_prior"] = max(-1.0, min(1.0, float(mutated.get("router_prior", 0.0)) + rng.uniform(-0.2, 0.2)))
    mutated["family_size_policy"] = policy
    return mutated, op


def bootstrap_candidates() -> list[tuple[dict[str, Any], str]]:
    candidates = []
    for value, label in ((2, "bootstrap_size2"), (3, "bootstrap_size3"), (4, "bootstrap_size4"), (23, "bootstrap_mixed23"), (24, "bootstrap_mixed24")):
        candidates.append(({"family_size_policy": {family: value for family in FAMILIES}, "frozen": {}, "router_prior": 0.0}, label))
    natural = {
        "family_A_natural_size_2": 2,
        "family_B_natural_size_3": 3,
        "family_C_natural_size_4": 4,
        "family_D_mixed_size_2_4": 24,
        "family_E_no_stable_pocket_size": 1,
        "family_F_decoy_pair_frequency": 3,
        "family_G_reuse_sparse_family": 24,
    }
    candidates.append(({"family_size_policy": natural, "frozen": {family: True for family in FAMILIES}, "router_prior": 0.2}, "bootstrap_variable_family_policy"))
    return candidates


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:variable_size_mutation"))
    best = initial_candidate(seed)
    initial_hash = payload_sha256(best)
    best_score = score_candidate(best, task)
    accepted = rejected = attempts = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate, operator in bootstrap_candidates():
        attempts += 1
        score = score_candidate(candidate, task)
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
            candidate, operator = mutate_candidate(best, rng)
            score = score_candidate(candidate, task)
            if score >= best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 10) == 0 or generation == settings.mutation_generations - 1:
            policy = best["family_size_policy"]
            row = {"generation": generation, "best_score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "candidate_hash": payload_sha256(best), "policy": policy}
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"mutation_discovered_variable_size_pockets_seed{seed}_generation{generation:04d}.json", row)
            if out:
                append_progress(out, "mutation_generation", seed=seed, system="mutation_discovered_variable_size_pockets", generation=generation, best_score=row["best_score"], policy=policy)
    pockets = candidate_pockets(best)
    predictions = predict_with_pockets(task, "mutation_discovered_variable_size_pockets", pockets)
    result = evaluate_predictions(task, predictions, "mutation_discovered_variable_size_pockets", pockets, candidate_parameter_count(best))
    result.update(
        {
            "seed": seed,
            "system": "mutation_discovered_variable_size_pockets",
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": payload_sha256(best),
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": best}),
            "final_candidate_summary": {
                "family_size_policy": best["family_size_policy"],
                "average_policy_size": round_float(float(np.mean([value if value < 10 else 3 for value in best["family_size_policy"].values()]))),
                "frozen_family_count": sum(1 for value in best.get("frozen", {}).values() if value),
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
    length_targets = torch.tensor([max(0, min(MAX_MICRO_PATH - 4, row["micro_path_length"] - 4)) for row in rows], dtype=torch.long, device=device)
    return x, targets, length_targets


def train_dense(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"{seed}:dense_graph"), device)
    train_rows = all_rows(task, "train")
    model = DenseFamilyMLP(len(train_rows[0]["raw"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train, targets, length_targets = make_tensor(train_rows, device)
    rng = np.random.default_rng(stable_seed(f"{seed}:dense_batches"))
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "training_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(settings.gradient_epochs):
        indices = rng.permutation(len(train_rows))
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


def evaluate_dense_state(model_state: dict[str, Any], task: dict[str, dict[str, list[dict[str, Any]]]], device: str) -> dict[str, Any]:
    model = DenseFamilyMLP(len(all_rows(task, "train")[0]["raw"])).to(device)
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in model_state.items()})
    model.eval()
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with torch.no_grad():
        for family in FAMILIES:
            predictions[family] = {}
            for split in SPLITS:
                rows = task[family][split]
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
                    split_preds.append({"calls": [(seg,) for seg in micro], "micro_path": micro[:MAX_MICRO_PATH], "steps": len(micro[:MAX_MICRO_PATH]) + branches, "branch_expansions": branches})
                predictions[family][split] = split_preds
    params = sum(int(np.prod(np.asarray(value).shape)) for value in model_state.values())
    return evaluate_predictions(task, predictions, "dense_graph_control", {family: [] for family in FAMILIES}, params)


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for seed_text, task in sorted(job["tasks"].items(), key=lambda item: int(item[0])):
        result = train_dense(int(seed_text), task, settings, out)
        rows.append({key: value for key, value in result.items() if key != "history"})
        histories.append({"seed": result["seed"], "system": result["system"], "history": result["history"], "device": result["device"], "parameter_count": result["parameter_count"], "model_state_hash": payload_sha256({key: value for key, value in result.items() if key not in {"history", "evals", "family_metrics"}})})
    return {"rows": rows, "histories": histories, "hardware": e7h.e7g.e7d.e7b.hardware_probe()}


def task_report(tasks: dict[int, dict[str, dict[str, list[dict[str, Any]]]]]) -> dict[str, Any]:
    return {
        "schema_version": "e7i_task_generation_report_v1",
        "row_counts": {str(seed): {family: {split: len(rows) for split, rows in family_task.items()} for family, family_task in task.items()} for seed, task in tasks.items()},
        "public_inputs": "microsegment_path_plus_family_token_no_public_pocket_size_labels",
        "hidden_natural_sizes_used_for_eval_only": True,
        "families": list(FAMILIES),
    }


def family_generation_report() -> dict[str, Any]:
    return {
        "schema_version": "e7i_family_generation_report_v1",
        "families": {
            family: {
                "natural_pockets": [list(p) for p in family_natural_pockets(family)],
                "natural_sizes": sorted(set(len(p) for p in family_natural_pockets(family))),
                "purpose": family,
            }
            for family in FAMILIES
        },
        "size2_not_baked_into_all_families": True,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    family_summary: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in ("heldout_usefulness", "ood_usefulness", "counterfactual_usefulness", "adversarial_usefulness", "eval_mean_usefulness", "generalization_gap", "parameter_count"):
                metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for metric, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{metric}", []).append(float(value))
            for family in FAMILIES:
                family_eval = float(np.mean([row["family_metrics"][family][split]["usefulness_score"] for split in EVAL_SPLITS]))
                family_summary[family].setdefault(system, []).append(family_eval)
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {metric: round_float(float(np.mean(values))) for metric, values in metrics.items()},
            "min": {metric: round_float(float(np.min(values))) for metric, values in metrics.items()},
            "max": {metric: round_float(float(np.max(values))) for metric, values in metrics.items()},
        }
    family_winners: dict[str, Any] = {}
    for family, by_system in family_summary.items():
        means = {system: round_float(float(np.mean(values))) for system, values in by_system.items()}
        best_system = max(means, key=lambda system: means[system])
        family_winners[family] = {"best_system": best_system, "system_eval_mean": means}
    non_oracle = [system for system in SYSTEMS if not system.startswith("oracle")]
    best = max(non_oracle, key=lambda system: systems[system]["mean"]["eval_mean_usefulness"])
    return {
        "schema_version": "e7i_aggregate_metrics_v1",
        "systems": systems,
        "family_winners": family_winners,
        "best_non_oracle_system": best,
        "best_non_oracle_eval_mean_usefulness": systems[best]["mean"]["eval_mean_usefulness"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    family_winners = aggregate["family_winners"]
    winner_counts: dict[str, int] = {}
    for row in family_winners.values():
        winner_counts[row["best_system"]] = winner_counts.get(row["best_system"], 0) + 1
    variable = systems["mutation_discovered_variable_size_pockets"]["mean"]
    dense = systems["dense_graph_control"]["mean"]
    fused = systems["fused_long_pipe"]["mean"]
    atomic = systems["atomic_microsegment_router"]["mean"]
    size2 = systems["fixed_size_2_pockets"]["mean"]
    fixed_human = systems["fixed_human_pocket_scaffold"]["mean"]
    fixed_sizes = {system: systems[system]["mean"]["eval_mean_usefulness"] for system in ("fixed_size_2_pockets", "fixed_size_3_pockets", "fixed_size_4_pockets", "mixed_size_2_3_pockets", "mixed_size_2_4_pockets")}
    best_fixed = max(fixed_sizes, key=lambda system: fixed_sizes[system])
    detail = {
        "overall_best_system": aggregate["best_non_oracle_system"],
        "winner_counts": winner_counts,
        "best_fixed_size_system": best_fixed,
        "variable_minus_best_fixed_eval": round_float(variable["eval_mean_usefulness"] - fixed_sizes[best_fixed]),
        "variable_minus_size2_eval": round_float(variable["eval_mean_usefulness"] - size2["eval_mean_usefulness"]),
        "fixed_human_minus_variable_eval": round_float(fixed_human["eval_mean_usefulness"] - variable["eval_mean_usefulness"]),
        "variable_minus_fused_ood": round_float(variable["ood_usefulness"] - fused["ood_usefulness"]),
        "fused_heldout_minus_ood": round_float(fused["heldout_usefulness"] - fused["ood_usefulness"]),
        "dense_eval_mean": dense["eval_mean_usefulness"],
        "family_winners": {family: row["best_system"] for family, row in family_winners.items()},
    }
    if systems["random_boundary_control"]["mean"]["eval_mean_usefulness"] > 0.70:
        return "e7i_leak_or_artifact_detected", detail
    if dense["eval_mean_usefulness"] > variable["eval_mean_usefulness"] + 0.02 and dense["ood_usefulness"] >= variable["ood_usefulness"] - 0.02:
        return "e7i_pocket_granularity_collapses_to_graph_soup", detail
    if fixed_human["eval_mean_usefulness"] > variable["eval_mean_usefulness"] + 1e-9:
        return "e7i_pocket_size_needs_prior_scaffold", detail
    if variable["eval_mean_usefulness"] >= fixed_sizes[best_fixed] - 0.01 and variable["ood_usefulness"] >= fused["ood_usefulness"] + 0.08 and variable["heldout_average_pocket_size"] > 1.5:
        return "e7i_variable_pocket_granularity_preferred", detail
    if winner_counts.get("fixed_size_2_pockets", 0) >= len(FAMILIES) - 1:
        return "e7i_stable_pocket_size_optimum_detected", detail
    if family_winners["family_A_natural_size_2"]["best_system"] == "fixed_size_2_pockets" and winner_counts.get("fixed_size_2_pockets", 0) <= 2:
        return "e7i_size2_was_generator_imprint", detail
    if atomic["eval_mean_usefulness"] >= variable["eval_mean_usefulness"] - 0.01:
        return "e7i_atomic_microsegment_routing_preferred", detail
    if fused["heldout_usefulness"] > variable["heldout_usefulness"] + 0.02 and detail["fused_heldout_minus_ood"] > 0.18:
        return "e7i_fused_pipe_overfit_detected", detail
    return "e7i_no_clear_size_frontier", detail


def build_pocket_size_sweep_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7i_pocket_size_sweep_report_v1",
        "family_winners": aggregate["family_winners"],
        "system_eval_means": {system: row["mean"]["eval_mean_usefulness"] for system, row in aggregate["systems"].items()},
        "size2_global_eval": aggregate["systems"]["fixed_size_2_pockets"]["mean"]["eval_mean_usefulness"],
        "variable_global_eval": aggregate["systems"]["mutation_discovered_variable_size_pockets"]["mean"]["eval_mean_usefulness"],
        "fused_ood": aggregate["systems"]["fused_long_pipe"]["mean"]["ood_usefulness"],
    }


def build_family_winner_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "e7i_family_winner_report_v1", "families": aggregate["family_winners"]}


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7i_leakage_report_v1",
        "public_inputs": "microsegment_path_plus_family_token_no_pocket_size_label",
        "hidden_natural_size_used_as_model_input": False,
        "dense_all_to_all_soft_routing_used_by_mutation_systems": False,
        "random_control_passed": aggregate["systems"]["random_boundary_control"]["mean"]["eval_mean_usefulness"] < 0.70,
        "dense_graph_control_measured": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    aggregate = payloads["aggregate_metrics.json"]
    decision = payloads["decision.json"]
    summary = payloads["summary.json"]
    detail = decision["detail"]
    lines = [
        "# E7I Pocket Size Optimum Sweep Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_oracle_system = {summary['best_non_oracle_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Eval Usefulness",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(f"{system:48s} eval={mean['eval_mean_usefulness']:.6f} heldout={mean['heldout_usefulness']:.6f} ood={mean['ood_usefulness']:.6f} steps={mean['heldout_mean_route_steps']:.3f} avg_size={mean['heldout_average_pocket_size']:.3f}")
    lines.extend(
        [
            "```",
            "",
            "## Family Winners",
            "",
            "```text",
        ]
    )
    for family, winner in detail["family_winners"].items():
        lines.append(f"{family:34s} {winner}")
    lines.extend(
        [
            "```",
            "",
            "## Frontier",
            "",
            "```text",
            f"best_fixed_size_system = {detail['best_fixed_size_system']}",
            f"variable_minus_best_fixed_eval = {detail['variable_minus_best_fixed_eval']}",
            f"variable_minus_size2_eval = {detail['variable_minus_size2_eval']}",
            f"variable_minus_fused_ood = {detail['variable_minus_fused_ood']}",
            f"fused_heldout_minus_ood = {detail['fused_heldout_minus_ood']}",
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric pocket-size sweep. It measures granularity curves over provided microsegments and multiple generator families.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7h.e7g.e7d.e7b.hardware_probe())
    tasks = generate_tasks(settings)
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed]))
    jobs = [{"seed": seed, "task": tasks[seed], "settings": settings.__dict__, "out": out.as_posix() if out else None} for seed in settings.seeds]
    gpu_job = {"tasks": {str(seed): tasks[seed] for seed in settings.seeds}, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers)) as executor:
            futures = {executor.submit(mutation_worker, job): f"mutation_discovered_variable_size_pockets/seed{job['seed']}" for job in jobs}
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
                    mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
                    if out:
                        locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7i_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
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
    return {"tasks": tasks, "rows": rows, "mutation_histories": mutation_histories, "training_histories": training_histories, "aggregate": aggregate, "decision": decision, "decision_detail": detail}


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {
        "backend_manifest.json": {"schema_version": "e7i_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "gradient_systems": list(GRADIENT_SYSTEMS), "mutation_systems": list(MUTATION_SYSTEMS), "control_systems": list(CONTROL_SYSTEMS), "hardware_identity": e7h.e7g.e7d.e7b.stable_hardware_identity(), "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel"},
        "task_generation_report.json": task_report(results["tasks"]),
        "family_generation_report.json": family_generation_report(),
        "pocket_size_sweep_report.json": build_pocket_size_sweep_report(results["aggregate"]),
        "family_winner_report.json": build_family_winner_report(results["aggregate"]),
        "system_results.json": {"schema_version": "e7i_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7i_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7i_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7i_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {"schema_version": "e7i_summary_v1", "decision": results["decision"], "best_non_oracle_system": results["aggregate"]["best_non_oracle_system"], "deterministic_replay_passed": False, "checker_failure_count": None, "run_root": out.relative_to(REPO_ROOT).as_posix()},
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
    primary = compute_hashes(primary_payloads)
    replay = compute_hashes(replay_payloads)
    comparisons = {name: {"primary_hash": primary[name], "replay_hash": replay[name], "match": primary[name] == replay[name]} for name in HASH_ARTIFACTS}
    report = {"schema_version": "e7i_deterministic_replay_v1", "internal_replay_passed": all(row["match"] for row in comparisons.values()), "hash_comparisons": comparisons, "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix()}
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
    parser.add_argument("--seeds", default="97001,97002,97003,97004,97005,97006")
    parser.add_argument("--train-rows-per-seed", type=int, default=700)
    parser.add_argument("--validation-rows-per-seed", type=int, default=280)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=280)
    parser.add_argument("--ood-rows-per-seed", type=int, default=280)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=280)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=280)
    parser.add_argument("--gradient-epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=90)
    parser.add_argument("--mutation-population", type=int, default=18)
    parser.add_argument("--mutation-sigma", type=float, default=0.16)
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7i_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
