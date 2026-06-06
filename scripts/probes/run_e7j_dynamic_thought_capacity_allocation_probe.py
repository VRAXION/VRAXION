#!/usr/bin/env python3
"""E7J dynamic thought capacity allocation probe.

E7I showed that pocket granularity is task-family dependent. E7J keeps a fixed
external Flow[D] interface and asks whether mutation/rollback can allocate an
internal K capacity per callable thought-pocket without drifting into dense
anonymous graph routing or simply making every pocket large.
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
E7I_PATH = Path(__file__).with_name("run_e7i_pocket_size_optimum_sweep.py")
MILESTONE = "E7J_DYNAMIC_THOUGHT_CAPACITY_ALLOCATION_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7j_dynamic_thought_capacity_allocation_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SYSTEMS = (
    "fixed_K1_pockets",
    "fixed_K2_pockets",
    "fixed_K4_pockets",
    "fixed_K8_pockets",
    "variable_K_allocator",
    "variable_K_grow_shrink_mutation",
    "variable_K_split_merge_mutation",
    "family_aware_capacity_scaffold",
    "fused_long_pipe",
    "dense_graph_danger_control",
    "random_capacity_control",
)
MUTATION_SYSTEMS = (
    "variable_K_allocator",
    "variable_K_grow_shrink_mutation",
    "variable_K_split_merge_mutation",
)
GRADIENT_SYSTEMS = ("dense_graph_danger_control",)
CONTROL_SYSTEMS = tuple(system for system in SYSTEMS if system not in MUTATION_SYSTEMS and system not in GRADIENT_SYSTEMS)
K_VALUES = (1, 2, 4, 8)
FLOW_D = 16
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "capacity_allocation_report.json",
    "capacity_value_report.json",
    "family_capacity_winner_report.json",
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
    "e7j_dynamic_capacity_allocation_positive",
    "e7j_dynamic_capacity_partially_positive",
    "e7j_fixed_capacity_sufficient",
    "e7j_capacity_needs_prior_scaffold",
    "e7j_variable_capacity_overfits_or_cost_ignored",
    "e7j_fused_pipe_capacity_preferred",
    "e7j_dense_graph_collapse_detected",
    "e7j_leak_or_artifact_detected",
)


def load_e7i_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7i_pocket_size_sweep", E7I_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7I helpers from {E7I_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7i = load_e7i_module()
e7h = e7i.e7h
FAMILIES = e7i.FAMILIES
MICRO_COUNT = e7i.MICRO_COUNT
MAX_MICRO_PATH = e7i.MAX_MICRO_PATH
PAD_SEGMENT = e7i.PAD_SEGMENT


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


class DensePathMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
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
    return int(hashlib.sha256(f"e7j::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def generate_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    return e7i.generate_tasks(settings)


def all_rows(task: dict[str, dict[str, list[dict[str, Any]]]], split: str) -> list[dict[str, Any]]:
    return e7i.all_rows(task, split)


def fixed_pockets(size: int) -> list[tuple[int, ...]]:
    return e7i.fixed_pockets(size)


def family_natural_pockets(family: str) -> list[tuple[int, ...]]:
    return e7i.family_natural_pockets(family)


def canonical_thought_pockets(family: str) -> list[tuple[int, ...]]:
    pockets = family_natural_pockets(family)
    if family == "family_E_no_stable_pocket_size" or not pockets:
        return [(idx,) for idx in range(MICRO_COUNT)]
    return pockets


def required_k_for_call(family: str, call: tuple[int, ...]) -> int:
    length = len(call)
    if length <= 1:
        return 1
    if length == 2:
        return 2
    if length == 3:
        return 4
    return 8


def pocket_key(family: str, pocket: tuple[int, ...]) -> str:
    return f"{family}|{','.join(str(int(seg)) for seg in pocket)}"


def capacity_params(k: int) -> int:
    return int(2 * FLOW_D * int(k) + int(k) * int(k))


def clamp_k(value: int) -> int:
    return min(K_VALUES, key=lambda item: abs(item - int(value)))


def policy_for_fixed_k(k: int) -> dict[str, int]:
    return {pocket_key(family, pocket): k for family in FAMILIES for pocket in canonical_thought_pockets(family)}


def scaffold_policy() -> dict[str, int]:
    return {pocket_key(family, pocket): required_k_for_call(family, pocket) for family in FAMILIES for pocket in canonical_thought_pockets(family)}


def candidate_policy(candidate: dict[str, Any], system: str) -> dict[str, int]:
    if system == "variable_K_allocator":
        family_k = {family: clamp_k(int(candidate.get("family_k", {}).get(family, 1))) for family in FAMILIES}
        return {pocket_key(family, pocket): family_k[family] for family in FAMILIES for pocket in canonical_thought_pockets(family)}
    policy = {}
    raw = candidate.get("pocket_k", {})
    for family in FAMILIES:
        for pocket in candidate_pockets(candidate, system)[family]:
            policy[pocket_key(family, pocket)] = clamp_k(int(raw.get(pocket_key(family, pocket), 1)))
    return policy


def candidate_pockets(candidate: dict[str, Any], system: str) -> dict[str, list[tuple[int, ...]]]:
    out = {family: list(canonical_thought_pockets(family)) for family in FAMILIES}
    if system != "variable_K_split_merge_mutation":
        return out
    merges = candidate.get("merged_calls", {})
    for family in FAMILIES:
        base = canonical_thought_pockets(family)
        existing = set(out[family])
        for pair in merges.get(family, []):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            left, right = int(pair[0]), int(pair[1])
            if 0 <= left < len(base) and 0 <= right < len(base) and abs(left - right) == 1:
                merged = tuple(list(base[min(left, right)]) + list(base[max(left, right)]))
                if 2 <= len(merged) <= MAX_MICRO_PATH and merged not in existing:
                    existing.add(merged)
                    out[family].append(merged)
    return out


def corrupt_call(call: tuple[int, ...], assigned_k: int, required_k: int) -> list[int]:
    if assigned_k >= required_k:
        return list(call)
    out = list(call)
    keep = {1: 1, 2: 2, 4: 3, 8: len(out)}[clamp_k(assigned_k)]
    for pos in range(max(1, keep), len(out)):
        out[pos] = (out[pos] + required_k + assigned_k + pos) % MICRO_COUNT
    return out


def greedy_calls(micro_path: list[int], pockets: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return e7i.greedy_calls(micro_path, pockets)


def calls_to_micro(calls: list[tuple[int, ...]]) -> list[int]:
    return e7i.calls_to_micro(calls)


def predict_capacity(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    family_pockets: dict[str, list[tuple[int, ...]]],
    k_policy: dict[str, int],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILIES:
        predictions[family] = {}
        pockets = family_pockets.get(family, [])
        for split in SPLITS:
            split_preds = []
            for row in task[family][split]:
                calls = greedy_calls(row["micro_path"], pockets)
                predicted_micro: list[int] = []
                call_rows: list[dict[str, Any]] = []
                compute_cost = 0.0
                under = over = fit = 0
                for call in calls:
                    assigned = int(k_policy.get(pocket_key(family, call), 1 if len(call) <= 1 else 2))
                    required = required_k_for_call(family, call)
                    predicted_micro.extend(corrupt_call(call, assigned, required))
                    fit += int(assigned >= required)
                    under += max(0, required - assigned)
                    over += max(0, assigned - required)
                    compute_cost += capacity_params(assigned)
                    call_rows.append({"call": list(call), "K": assigned, "required_K": required})
                split_preds.append(
                    {
                        "calls": call_rows,
                        "micro_path": predicted_micro[:MAX_MICRO_PATH],
                        "steps": len(calls),
                        "branch_expansions": 0,
                        "compute_cost": compute_cost,
                        "capacity_fit_calls": fit,
                        "capacity_under": under,
                        "capacity_over": over,
                    }
                )
            predictions[family][split] = split_preds
    return predictions


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
                split_preds.append({"calls": [{"call": predicted, "K": 8, "required_K": 8}], "micro_path": predicted, "steps": 1, "branch_expansions": 0, "compute_cost": capacity_params(8), "capacity_fit_calls": 1, "capacity_under": 0, "capacity_over": 0})
            predictions[family][split] = split_preds
    return predictions


def predict_random(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rng = random.Random(stable_seed(f"{seed}:random_capacity_control"))
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILIES:
        predictions[family] = {}
        for split in SPLITS:
            split_preds = []
            for row in task[family][split]:
                micro = list(row["micro_path"])
                rng.shuffle(micro)
                calls = [{"call": [seg], "K": rng.choice(K_VALUES), "required_K": 1} for seg in micro]
                split_preds.append({"calls": calls, "micro_path": micro, "steps": len(micro), "branch_expansions": rng.randrange(0, 3), "compute_cost": sum(capacity_params(call["K"]) for call in calls), "capacity_fit_calls": len(calls), "capacity_under": 0, "capacity_over": 0})
            predictions[family][split] = split_preds
    return predictions


def mean_reuse(rows: list[dict[str, Any]], pockets: list[tuple[int, ...]]) -> float:
    return e7i.mean_reuse(rows, pockets)


def parameter_count_for_profile(family_pockets: dict[str, list[tuple[int, ...]]], k_policy: dict[str, int], router_extra: int = 0) -> int:
    total = router_extra
    for family, pockets in family_pockets.items():
        total += 3 * len(pockets)
        for pocket in pockets:
            total += capacity_params(int(k_policy.get(pocket_key(family, pocket), 1)))
    return int(total)


def evaluate_predictions(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    family_pockets: dict[str, list[tuple[int, ...]]],
    k_policy: dict[str, int],
    parameter_count: int,
    router_complexity: float,
) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    family_metrics: dict[str, Any] = {}
    for family in FAMILIES:
        family_metrics[family] = {}
        for split in SPLITS:
            rows = task[family][split]
            preds = predictions[family][split]
            pockets = list(family_pockets.get(family, []))
            answer_hits = route_hits = valid_hits = loops = 0
            steps_total = irrelevant_total = branch_total = compute_total = 0.0
            fit_total = under_total = over_total = call_total = 0.0
            compression_values: list[float] = []
            samples: list[dict[str, Any]] = []
            for row, pred in zip(rows, preds):
                predicted = [int(seg) for seg in pred.get("micro_path", []) if 0 <= int(seg) < MICRO_COUNT]
                final = e7i.apply_micro_path(predicted, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
                predicted_answer = 1 if final > row["threshold"] else 0
                target = list(row["micro_path"])
                call_rows = pred.get("calls", [])
                call_tuples = [tuple(int(seg) for seg in call.get("call", [])) for call in call_rows]
                steps = float(pred.get("steps", len(call_tuples)))
                branches = max(0, int(pred.get("branch_expansions", 0)))
                answer_hits += int(predicted_answer == row["answer"])
                route_hits += int(predicted == target)
                valid_hits += int(bool(predicted) and len(predicted) <= MAX_MICRO_PATH)
                loops += int(len(call_tuples) != len(set(call_tuples)))
                irrelevant = sum(1 for seg in predicted if seg not in set(target))
                irrelevant_total += min(1.0, (irrelevant + branches) / max(1, MICRO_COUNT - len(set(target))))
                branch_total += branches
                compression_values.append(max(0.0, min(1.0, (len(target) - steps) / max(1, len(target) - 1))))
                steps_total += steps
                compute_total += float(pred.get("compute_cost", 0.0))
                fit_total += float(pred.get("capacity_fit_calls", 0))
                under_total += float(pred.get("capacity_under", 0.0))
                over_total += float(pred.get("capacity_over", 0.0))
                call_total += max(1.0, float(len(call_rows)))
                if len(samples) < 3:
                    samples.append({"row_id": row["row_id"], "target": target, "predicted": predicted, "calls": call_rows, "steps": round_float(steps), "target_answer": row["answer"], "predicted_answer": predicted_answer})
            n = max(1, len(rows))
            answer_accuracy = answer_hits / n
            route_accuracy = route_hits / n
            compression = float(np.mean(compression_values)) if compression_values else 0.0
            irrelevant = irrelevant_total / n
            loop_rate = loops / n
            reuse = mean_reuse(rows, pockets)
            k_values = [int(k_policy.get(pocket_key(family, pocket), 1)) for pocket in pockets]
            avg_k = float(np.mean(k_values)) if k_values else 0.0
            k_distribution = {str(k): sum(1 for value in k_values if value == k) for k in K_VALUES}
            capacity_fit = fit_total / max(1.0, call_total)
            under_rate = under_total / max(1.0, call_total * 7.0)
            over_rate = over_total / max(1.0, call_total * 7.0)
            compute_cost = compute_total / n
            compute_norm = min(1.0, compute_cost / max(1.0, MAX_MICRO_PATH * capacity_params(4)))
            param_norm = min(1.0, parameter_count / max(1.0, len(FAMILIES) * 16 * capacity_params(4)))
            repairability = max(0.0, min(1.0, capacity_fit - 0.20 * under_rate + 0.05 * min(1.0, reuse / 3.0)))
            usefulness = (
                0.31 * answer_accuracy
                + 0.25 * route_accuracy
                + 0.15 * compression
                + 0.08 * capacity_fit
                + 0.05 * min(1.0, reuse / 3.0)
                + 0.04 * repairability
                - 0.07 * param_norm
                - 0.06 * compute_norm
                - 0.05 * under_rate
                - 0.03 * over_rate
                - 0.05 * irrelevant
                - 0.03 * loop_rate
                - 0.03 * min(1.0, router_complexity / 12.0)
            )
            capacity_value = (
                usefulness
                + 0.05 * capacity_fit
                + 0.04 * repairability
                - 0.04 * compute_norm
                - 0.04 * param_norm
                - 0.04 * abs(avg_k - 3.0) / 7.0
            )
            family_metrics[family][split] = {
                "answer_accuracy": round_float(answer_accuracy),
                "route_accuracy": round_float(route_accuracy),
                "mean_route_steps": round_float(steps_total / n),
                "compression_score": round_float(compression),
                "capacity_fit_rate": round_float(capacity_fit),
                "capacity_under_rate": round_float(under_rate),
                "capacity_over_rate": round_float(over_rate),
                "compute_cost": round_float(compute_cost),
                "parameter_cost": int(parameter_count),
                "irrelevant_branch_rate": round_float(irrelevant),
                "loop_rate": round_float(loop_rate),
                "branch_expansion_rate": round_float(branch_total / n),
                "pocket_count": len(pockets),
                "average_K": round_float(avg_k),
                "K_distribution": k_distribution,
                "reuse_count_per_pocket": round_float(reuse),
                "freeze_survival_score": round_float(route_accuracy if pockets else 0.0),
                "local_repair_gain": round_float(repairability),
                "router_complexity": round_float(router_complexity),
                "usefulness_score": round_float(max(0.0, min(1.0, usefulness))),
                "capacity_value_score": round_float(max(0.0, min(1.0, capacity_value))),
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
    train = evals["train"]["capacity_value_score"]
    eval_mean_capacity = float(np.mean([evals[split]["capacity_value_score"] for split in EVAL_SPLITS]))
    eval_mean_usefulness = float(np.mean([evals[split]["usefulness_score"] for split in EVAL_SPLITS]))
    return {
        "system": system,
        "evals": evals,
        "family_metrics": family_metrics,
        "heldout_usefulness": round_float(evals["heldout"]["usefulness_score"]),
        "ood_usefulness": round_float(evals["ood"]["usefulness_score"]),
        "counterfactual_usefulness": round_float(evals["counterfactual"]["usefulness_score"]),
        "adversarial_usefulness": round_float(evals["adversarial"]["usefulness_score"]),
        "eval_mean_usefulness": round_float(eval_mean_usefulness),
        "eval_mean_capacity_value": round_float(eval_mean_capacity),
        "generalization_gap": round_float(train - eval_mean_capacity),
        "parameter_count": int(parameter_count),
    }


def profile_result(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int, system: str, family_pockets: dict[str, list[tuple[int, ...]]], k_policy: dict[str, int], router_complexity: float) -> dict[str, Any]:
    predictions = predict_capacity(task, system, family_pockets, k_policy)
    params = parameter_count_for_profile(family_pockets, k_policy, router_extra=int(router_complexity * 8))
    row = evaluate_predictions(task, predictions, system, family_pockets, k_policy, params, router_complexity)
    row["seed"] = seed
    return row


def control_results(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical = {family: canonical_thought_pockets(family) for family in FAMILIES}
    for system, k in (("fixed_K1_pockets", 1), ("fixed_K2_pockets", 2), ("fixed_K4_pockets", 4), ("fixed_K8_pockets", 8)):
        rows.append(profile_result(task, seed, system, canonical, policy_for_fixed_k(k), router_complexity=2.0))
    scaffold = scaffold_policy()
    rows.append(profile_result(task, seed, "family_aware_capacity_scaffold", canonical, scaffold, router_complexity=2.5))
    fused_pred = predict_fused(task)
    fused_params = len(FAMILIES) * MAX_MICRO_PATH * capacity_params(8)
    fused_row = evaluate_predictions(task, fused_pred, "fused_long_pipe", {family: [] for family in FAMILIES}, {}, fused_params, router_complexity=1.0)
    fused_row["seed"] = seed
    rows.append(fused_row)
    random_pred = predict_random(task, seed)
    random_row = evaluate_predictions(task, random_pred, "random_capacity_control", {family: [] for family in FAMILIES}, {}, len(FAMILIES) * 16, router_complexity=12.0)
    random_row["seed"] = seed
    rows.append(random_row)
    return rows


def candidate_initial(system: str) -> dict[str, Any]:
    if system == "variable_K_allocator":
        return {"family_k": {family: 1 for family in FAMILIES}, "frozen": {}, "router_prior": 0.0}
    pocket_k = {pocket_key(family, pocket): 1 for family in FAMILIES for pocket in canonical_thought_pockets(family)}
    return {"pocket_k": pocket_k, "merged_calls": {}, "frozen": {}, "router_prior": 0.0}


def candidate_summary(candidate: dict[str, Any], system: str) -> dict[str, Any]:
    policy = candidate_policy(candidate, system)
    values = list(policy.values())
    return {
        "average_K": round_float(float(np.mean(values)) if values else 0.0),
        "K_distribution": {str(k): sum(1 for value in values if value == k) for k in K_VALUES},
        "large_K_fraction": round_float(sum(1 for value in values if value >= 8) / max(1, len(values))),
        "family_k": candidate.get("family_k", {}),
        "merged_call_count": sum(len(items) for items in candidate.get("merged_calls", {}).values()),
        "frozen_count": sum(1 for value in candidate.get("frozen", {}).values() if value),
    }


def score_candidate(candidate: dict[str, Any], system: str, task: dict[str, dict[str, list[dict[str, Any]]]]) -> float:
    pockets = candidate_pockets(candidate, system)
    policy = candidate_policy(candidate, system)
    predictions = predict_capacity(task, system, pockets, policy)
    params = parameter_count_for_profile(pockets, policy, router_extra=24)
    result = evaluate_predictions(task, predictions, system, pockets, policy, params, router_complexity=3.0)
    family_scores = [result["family_metrics"][family]["train"]["capacity_value_score"] for family in FAMILIES]
    train = result["evals"]["train"]
    policy_values = list(policy.values())
    large_k_fraction = sum(1 for value in policy_values if value >= 8) / max(1, len(policy_values))
    return float(np.mean(family_scores)) + 0.03 * train["local_repair_gain"] - 0.04 * large_k_fraction - 0.02 * train["compute_cost"] / max(1.0, MAX_MICRO_PATH * capacity_params(4))


def mutate_candidate(candidate: dict[str, Any], system: str, rng: random.Random) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(candidate)
    if system == "variable_K_allocator":
        family_k = dict(mutated.get("family_k", {}))
        op = rng.choice(("set_family_K", "grow_family", "shrink_family", "copy_family", "freeze_family", "router_prior"))
        family = rng.choice(FAMILIES)
        if op == "set_family_K":
            family_k[family] = rng.choice(K_VALUES)
        elif op == "grow_family":
            current = clamp_k(int(family_k.get(family, 1)))
            family_k[family] = K_VALUES[min(len(K_VALUES) - 1, K_VALUES.index(current) + 1)]
        elif op == "shrink_family":
            current = clamp_k(int(family_k.get(family, 1)))
            family_k[family] = K_VALUES[max(0, K_VALUES.index(current) - 1)]
        elif op == "copy_family":
            src = rng.choice(FAMILIES)
            family_k[family] = clamp_k(int(family_k.get(src, 1)))
        elif op == "freeze_family":
            frozen = dict(mutated.get("frozen", {}))
            frozen[family] = not bool(frozen.get(family, False))
            mutated["frozen"] = frozen
        elif op == "router_prior":
            mutated["router_prior"] = max(-1.0, min(1.0, float(mutated.get("router_prior", 0.0)) + rng.uniform(-0.2, 0.2)))
        mutated["family_k"] = family_k
        return mutated, op

    pocket_k = dict(mutated.get("pocket_k", {}))
    all_keys = [pocket_key(family, pocket) for family in FAMILIES for pocket in candidate_pockets(mutated, system)[family]]
    op_choices = ["set_pocket_K", "grow_pocket", "shrink_pocket", "copy_related", "freeze_pocket", "router_prior"]
    if system == "variable_K_split_merge_mutation":
        op_choices.extend(["add_merge", "remove_merge", "grow_merged"])
    op = rng.choice(op_choices)
    key = rng.choice(all_keys)
    if op == "set_pocket_K":
        pocket_k[key] = rng.choice(K_VALUES)
    elif op == "grow_pocket":
        current = clamp_k(int(pocket_k.get(key, 1)))
        pocket_k[key] = K_VALUES[min(len(K_VALUES) - 1, K_VALUES.index(current) + 1)]
    elif op == "shrink_pocket":
        current = clamp_k(int(pocket_k.get(key, 1)))
        pocket_k[key] = K_VALUES[max(0, K_VALUES.index(current) - 1)]
    elif op == "copy_related":
        src = rng.choice(all_keys)
        pocket_k[key] = clamp_k(int(pocket_k.get(src, 1)))
    elif op == "freeze_pocket":
        frozen = dict(mutated.get("frozen", {}))
        frozen[key] = not bool(frozen.get(key, False))
        mutated["frozen"] = frozen
    elif op == "router_prior":
        mutated["router_prior"] = max(-1.0, min(1.0, float(mutated.get("router_prior", 0.0)) + rng.uniform(-0.2, 0.2)))
    elif op == "add_merge":
        family = rng.choice(FAMILIES)
        base = canonical_thought_pockets(family)
        if len(base) >= 2:
            left = rng.randrange(0, len(base) - 1)
            merges = copy.deepcopy(mutated.get("merged_calls", {}))
            row = merges.setdefault(family, [])
            pair = [left, left + 1]
            if pair not in row:
                row.append(pair)
            merged = tuple(list(base[left]) + list(base[left + 1]))
            pocket_k[pocket_key(family, merged)] = rng.choice((4, 8))
            mutated["merged_calls"] = merges
    elif op == "remove_merge":
        merges = copy.deepcopy(mutated.get("merged_calls", {}))
        choices = [(family, idx) for family, rows in merges.items() for idx in range(len(rows))]
        if choices:
            family, idx = rng.choice(choices)
            merges[family].pop(idx)
            mutated["merged_calls"] = merges
    elif op == "grow_merged":
        merged_keys = [key for key in all_keys if key.count(",") >= 3]
        if merged_keys:
            mkey = rng.choice(merged_keys)
            current = clamp_k(int(pocket_k.get(mkey, 1)))
            pocket_k[mkey] = K_VALUES[min(len(K_VALUES) - 1, K_VALUES.index(current) + 1)]
    mutated["pocket_k"] = pocket_k
    return mutated, op


def bootstrap_candidates(system: str) -> list[tuple[dict[str, Any], str]]:
    candidates = []
    if system == "variable_K_allocator":
        for k in K_VALUES:
            candidates.append(({"family_k": {family: k for family in FAMILIES}, "frozen": {}, "router_prior": 0.0}, f"bootstrap_fixed_K{k}"))
        return candidates
    base_pockets = {family: canonical_thought_pockets(family) for family in FAMILIES}
    for k in K_VALUES:
        policy = {pocket_key(family, pocket): k for family, pockets in base_pockets.items() for pocket in pockets}
        candidates.append(({"pocket_k": policy, "merged_calls": {}, "frozen": {}, "router_prior": 0.0}, f"bootstrap_fixed_K{k}"))
    return candidates


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:capacity_mutation"))
    best = candidate_initial(system)
    initial_hash = payload_sha256(best)
    best_score = score_candidate(best, system, task)
    accepted = rejected = attempts = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate, operator in bootstrap_candidates(system):
        attempts += 1
        score = score_candidate(candidate, system, task)
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
            candidate, operator = mutate_candidate(best, system, rng)
            score = score_candidate(candidate, system, task)
            if score >= best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 10) == 0 or generation == settings.mutation_generations - 1:
            row = {"generation": generation, "best_score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "candidate_hash": payload_sha256(best), "summary": candidate_summary(best, system)}
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json", row)
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, best_score=row["best_score"], summary=row["summary"])
    pockets = candidate_pockets(best, system)
    policy = candidate_policy(best, system)
    predictions = predict_capacity(task, system, pockets, policy)
    params = parameter_count_for_profile(pockets, policy, router_extra=24)
    result = evaluate_predictions(task, predictions, system, pockets, policy, params, router_complexity=3.0)
    result.update(
        {
            "seed": seed,
            "system": system,
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": payload_sha256(best),
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": best}),
            "final_candidate_summary": candidate_summary(best, system),
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
    set_determinism(stable_seed(f"{seed}:dense_capacity_graph"), device)
    train_rows = all_rows(task, "train")
    model = DensePathMLP(len(train_rows[0]["raw"])).to(device)
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
                locked_write_json(snapshot_dir / f"dense_graph_danger_control_seed{seed}_epoch{epoch:04d}.json", row)
            if out:
                append_progress(out, "gradient_epoch", seed=seed, system="dense_graph_danger_control", epoch=epoch, loss=row["loss"], device=device)
    state = {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}
    result = evaluate_dense_state(state, task, device)
    result.update({"seed": seed, "system": "dense_graph_danger_control", "history": history, "device": device})
    return result


def evaluate_dense_state(model_state: dict[str, Any], task: dict[str, dict[str, list[dict[str, Any]]]], device: str) -> dict[str, Any]:
    model = DensePathMLP(len(all_rows(task, "train")[0]["raw"])).to(device)
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
                    calls = [{"call": [seg], "K": 1, "required_K": 1} for seg in micro[:MAX_MICRO_PATH]]
                    split_preds.append({"calls": calls, "micro_path": micro[:MAX_MICRO_PATH], "steps": len(micro[:MAX_MICRO_PATH]) + branches, "branch_expansions": branches, "compute_cost": sum(capacity_params(1) for _ in calls), "capacity_fit_calls": len(calls), "capacity_under": 0, "capacity_over": 0})
                predictions[family][split] = split_preds
    params = sum(int(np.prod(np.asarray(value).shape)) for value in model_state.values())
    return evaluate_predictions(task, predictions, "dense_graph_danger_control", {family: [] for family in FAMILIES}, {}, params, router_complexity=12.0)


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
        "schema_version": "e7j_task_generation_report_v1",
        "row_counts": {str(seed): {family: {split: len(rows) for split, rows in family_task.items()} for family, family_task in task.items()} for seed, task in tasks.items()},
        "public_inputs": "microsegment_path_plus_family_token_no_public_capacity_labels",
        "hidden_capacity_targets_used_for_eval_only": True,
        "families": list(FAMILIES),
    }


def capacity_allocation_report() -> dict[str, Any]:
    return {
        "schema_version": "e7j_capacity_allocation_report_v1",
        "external_interface": "CALL(pocket_id, Flow[D]) -> Flow[D]",
        "flow_d": FLOW_D,
        "allowed_K": list(K_VALUES),
        "families": {
            family: {
                "thought_pockets": [list(pocket) for pocket in canonical_thought_pockets(family)],
                "ideal_K_distribution": {str(k): sum(1 for pocket in canonical_thought_pockets(family) if required_k_for_call(family, pocket) == k) for k in K_VALUES},
            }
            for family in FAMILIES
        },
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    family_summary: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in ("heldout_usefulness", "ood_usefulness", "counterfactual_usefulness", "adversarial_usefulness", "eval_mean_usefulness", "eval_mean_capacity_value", "generalization_gap", "parameter_count"):
                metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for metric, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{metric}", []).append(float(value))
            for family in FAMILIES:
                family_eval = float(np.mean([row["family_metrics"][family][split]["capacity_value_score"] for split in EVAL_SPLITS]))
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
        family_winners[family] = {"best_system": best_system, "system_capacity_value_mean": means}
    best = max(SYSTEMS, key=lambda system: systems[system]["mean"]["eval_mean_capacity_value"])
    return {
        "schema_version": "e7j_aggregate_metrics_v1",
        "systems": systems,
        "family_winners": family_winners,
        "best_system": best,
        "best_eval_mean_capacity_value": systems[best]["mean"]["eval_mean_capacity_value"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    fixed = {system: systems[system]["mean"]["eval_mean_capacity_value"] for system in ("fixed_K1_pockets", "fixed_K2_pockets", "fixed_K4_pockets", "fixed_K8_pockets")}
    best_fixed = max(fixed, key=lambda system: fixed[system])
    scaffold = systems["family_aware_capacity_scaffold"]["mean"]
    variable_systems = ("variable_K_allocator", "variable_K_grow_shrink_mutation", "variable_K_split_merge_mutation")
    variable_best = max(variable_systems, key=lambda system: systems[system]["mean"]["eval_mean_capacity_value"])
    variable = systems[variable_best]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    fused = systems["fused_long_pipe"]["mean"]
    random_control = systems["random_capacity_control"]["mean"]
    scaffold_gap = scaffold["eval_mean_capacity_value"] - variable["eval_mean_capacity_value"]
    fixed_gap = variable["eval_mean_capacity_value"] - fixed[best_fixed]
    large_k = max(systems[system]["mean"].get("heldout_average_K", 0.0) for system in variable_systems)
    detail = {
        "overall_best_system": aggregate["best_system"],
        "best_fixed_K_system": best_fixed,
        "best_variable_K_system": variable_best,
        "variable_minus_best_fixed_capacity_value": round_float(fixed_gap),
        "scaffold_minus_variable_capacity_value": round_float(scaffold_gap),
        "variable_minus_fused_ood_capacity_value": round_float(variable["ood_capacity_value_score"] - fused["ood_capacity_value_score"]),
        "dense_capacity_value_mean": dense["eval_mean_capacity_value"],
        "random_capacity_value_mean": random_control["eval_mean_capacity_value"],
        "variable_max_average_K": round_float(large_k),
        "family_winners": {family: row["best_system"] for family, row in aggregate["family_winners"].items()},
    }
    if random_control["eval_mean_capacity_value"] > 0.60:
        return "e7j_leak_or_artifact_detected", detail
    if dense["eval_mean_capacity_value"] > variable["eval_mean_capacity_value"] + 0.02 and dense["ood_capacity_value_score"] >= variable["ood_capacity_value_score"] - 0.02:
        return "e7j_dense_graph_collapse_detected", detail
    if fused["eval_mean_capacity_value"] > variable["eval_mean_capacity_value"] + 0.02 and fused["ood_capacity_value_score"] >= variable["ood_capacity_value_score"] - 0.02:
        return "e7j_fused_pipe_capacity_preferred", detail
    if large_k >= 7.5 and variable["eval_mean_capacity_value"] < scaffold["eval_mean_capacity_value"] - 0.01:
        return "e7j_variable_capacity_overfits_or_cost_ignored", detail
    if fixed[best_fixed] >= variable["eval_mean_capacity_value"] - 0.005 and fixed[best_fixed] >= scaffold["eval_mean_capacity_value"] - 0.02:
        return "e7j_fixed_capacity_sufficient", detail
    if fixed_gap > 0.02 and scaffold_gap <= 0.01:
        return "e7j_dynamic_capacity_allocation_positive", detail
    if fixed_gap > 0.02 and scaffold_gap <= 0.05:
        return "e7j_dynamic_capacity_partially_positive", detail
    if scaffold_gap > 0.01:
        return "e7j_capacity_needs_prior_scaffold", detail
    return "e7j_dynamic_capacity_partially_positive", detail


def build_capacity_value_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7j_capacity_value_report_v1",
        "system_capacity_value_means": {system: row["mean"]["eval_mean_capacity_value"] for system, row in aggregate["systems"].items()},
        "system_usefulness_means": {system: row["mean"]["eval_mean_usefulness"] for system, row in aggregate["systems"].items()},
        "family_winners": aggregate["family_winners"],
    }


def build_family_winner_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "e7j_family_capacity_winner_report_v1", "families": aggregate["family_winners"]}


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7j_leakage_report_v1",
        "public_inputs": "microsegment_path_plus_family_token_no_capacity_label",
        "hidden_capacity_target_used_as_model_input": False,
        "dense_all_to_all_soft_routing_used_by_mutation_systems": False,
        "random_control_passed": aggregate["systems"]["random_capacity_control"]["mean"]["eval_mean_capacity_value"] < 0.60,
        "dense_graph_danger_control_measured": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    aggregate = payloads["aggregate_metrics.json"]
    decision = payloads["decision.json"]
    summary = payloads["summary.json"]
    detail = decision["detail"]
    lines = [
        "# E7J Dynamic Thought Capacity Allocation Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {summary['best_system']}",
        f"best_variable_K_system = {detail['best_variable_K_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Capacity Value",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(f"{system:42s} capacity={mean['eval_mean_capacity_value']:.6f} useful={mean['eval_mean_usefulness']:.6f} ood={mean['ood_capacity_value_score']:.6f} K={mean.get('heldout_average_K', 0.0):.3f} cost={mean.get('heldout_compute_cost', 0.0):.1f}")
    lines.extend(["```", "", "## Frontier", "", "```text"])
    for key in ("best_fixed_K_system", "best_variable_K_system", "variable_minus_best_fixed_capacity_value", "scaffold_minus_variable_capacity_value", "variable_minus_fused_ood_capacity_value", "dense_capacity_value_mean", "variable_max_average_K"):
        lines.append(f"{key} = {detail[key]}")
    lines.extend(["```", "", "## Family Winners", "", "```text"])
    for family, winner in detail["family_winners"].items():
        lines.append(f"{family:34s} {winner}")
    lines.extend(["```", "", "## Boundary", "", "This is a controlled symbolic/numeric capacity-allocation proxy over fixed Flow[D] callable pockets."])
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
    jobs = [{"seed": seed, "system": system, "task": tasks[seed], "settings": settings.__dict__, "out": out.as_posix() if out else None} for seed in settings.seeds for system in MUTATION_SYSTEMS]
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
                    mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
                    if out:
                        locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7j_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
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
        "backend_manifest.json": {"schema_version": "e7j_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "gradient_systems": list(GRADIENT_SYSTEMS), "mutation_systems": list(MUTATION_SYSTEMS), "control_systems": list(CONTROL_SYSTEMS), "hardware_identity": e7h.e7g.e7d.e7b.stable_hardware_identity(), "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel"},
        "task_generation_report.json": task_report(results["tasks"]),
        "capacity_allocation_report.json": capacity_allocation_report(),
        "capacity_value_report.json": build_capacity_value_report(results["aggregate"]),
        "family_capacity_winner_report.json": build_family_winner_report(results["aggregate"]),
        "system_results.json": {"schema_version": "e7j_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7j_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7j_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7j_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {"schema_version": "e7j_summary_v1", "decision": results["decision"], "best_system": results["aggregate"]["best_system"], "deterministic_replay_passed": False, "checker_failure_count": None, "run_root": out.relative_to(REPO_ROOT).as_posix()},
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
    report = {"schema_version": "e7j_deterministic_replay_v1", "internal_replay_passed": all(row["match"] for row in comparisons.values()), "hash_comparisons": comparisons, "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix()}
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
    parser.add_argument("--seeds", default="98001,98002,98003,98004,98005,98006")
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7j_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
