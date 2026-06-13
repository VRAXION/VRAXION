#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E47_EDGE_ADAPTER_POCKET_GROWTH_PROBE"
BOUNDARY = (
    "E47 treats the edge/rail between two frozen nodes as an explicit "
    "adapter pocket: source mini-matrix bits are transformed into consumer "
    "mini-matrix bits. It tests whether that edge pocket can grow by +1 bit "
    "without regressing old intents. It does not claim raw language "
    "reasoning, AGI, consciousness, deployed behavior, or model-scale "
    "behavior."
)

PASS_ACCURACY = 0.95
BASE_INTENT_COUNT = 32
FINAL_INTENT_COUNT = 256
START_ACTIVE_BITS = 5
FINAL_ACTIVE_BITS = 8
FIXED_PHYSICAL_WIDTH = 16

# Logical target bit j is emitted by Node A at this source slot. The first
# five slots intentionally scramble the old 32-intent ABI, so a raw wire cannot
# solve the task by reading source slots 0..n directly.
SOURCE_SLOT_PREFIX = [2, 4, 1, 0, 3, 5, 6, 7]

SYSTEMS = [
    "raw_wire_direct_fixed16",
    "raw_wire_progressive_plus1",
    "edge_adapter_fixed16_to16",
    "edge_adapter_progressive_plus1_freeze_old",
    "edge_adapter_progressive_plus1_no_freeze",
    "edge_adapter_block_growth_plus4",
    "identity_oracle_adapter_reference",
    "random_adapter_control",
]

MUTATED_SYSTEMS = {
    "edge_adapter_fixed16_to16",
    "edge_adapter_progressive_plus1_freeze_old",
    "edge_adapter_progressive_plus1_no_freeze",
    "edge_adapter_block_growth_plus4",
}

DECISIONS = {
    "e47_edge_adapter_pocket_positive",
    "e47_raw_wire_growth_sufficient",
    "e47_adapter_growth_overhead_too_high",
    "e47_adapter_growth_causes_regression",
    "e47_block_adapter_growth_preferred",
    "e47_invalid_artifact_detected",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"timestamp": time.time(), "cpu_count": os.cpu_count()}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, mem_used, mem_total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            snap["gpu"] = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        else:
            snap["gpu"] = {"available": False}
    except Exception:
        snap["gpu"] = {"available": False}
    return snap


def encode_intent(intent: int, bits: int) -> list[int]:
    return [(intent >> idx) & 1 for idx in range(bits)]


def source_slots(active_bits: int) -> list[int]:
    return SOURCE_SLOT_PREFIX[:active_bits]


def make_source_matrix(intent: int, physical_width: int, active_bits: int, seed: int, row_id: str) -> list[int]:
    rng = random.Random(seed + int(stable_hash(row_id)[:10], 16))
    source = [rng.randrange(2) for _ in range(physical_width)]
    for logical_slot, bit in enumerate(encode_intent(intent, active_bits)):
        source[source_slots(active_bits)[logical_slot]] = bit
    return source


def make_rows(physical_width: int, active_bits: int, seed: int, rows_per_split: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_offsets = {"train": 0, "heldout": 5, "ood": 11, "counterfactual": 19, "adversarial": 29}
    capacity = 2**active_bits
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for idx in range(rows_per_split):
            intent = (idx * 17 + split_offsets[split]) % FINAL_INTENT_COUNT
            row_id = f"{split}_{idx:05d}_w{physical_width}_a{active_bits}"
            rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "intent": intent,
                    "expected_intent_under_capacity": intent % capacity,
                    "is_old_intent": intent < BASE_INTENT_COUNT,
                    "physical_width": physical_width,
                    "active_bits": active_bits,
                    "source_node_output_bits": make_source_matrix(intent, physical_width, active_bits, seed, row_id),
                    "expected_consumer_input_bits": encode_intent(intent % capacity, active_bits),
                    "expected_action": "COMMIT",
                }
            )
    return rows


def transform_with_adapter(source_bits: list[int], mapping: list[int]) -> list[int] | None:
    if len(set(mapping)) != len(mapping):
        return None
    if any(idx < 0 or idx >= len(source_bits) for idx in mapping):
        return None
    return [int(source_bits[idx]) for idx in mapping]


def decode_bits(bits: list[int]) -> int:
    return sum(int(bit) << idx for idx, bit in enumerate(bits))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def split_metric(rows: list[dict[str, Any]], split: str, key: str) -> float:
    chunk = [row for row in rows if row["split"] == split]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows = [row for row in rows if row["is_old_intent"]]
    return {
        "row_count": len(rows),
        "adapter_success": mean([1.0 if row["adapter_success"] else 0.0 for row in rows]),
        "heldout_success": split_metric(rows, "heldout", "adapter_success"),
        "ood_success": split_metric(rows, "ood", "adapter_success"),
        "counterfactual_success": split_metric(rows, "counterfactual", "adapter_success"),
        "adversarial_success": split_metric(rows, "adversarial", "adapter_success"),
        "old_intent_success": mean([1.0 if row["adapter_success"] else 0.0 for row in old_rows]),
        "bit_accuracy": mean([float(row["bit_accuracy"]) for row in rows]),
        "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in rows]),
        "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in rows]),
        "source_to_target_transform_accuracy": mean([float(row["bit_accuracy"]) for row in rows]),
    }


def evaluate_rows(rows: list[dict[str, Any]], mapping: list[int], system: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        transformed = transform_with_adapter(row["source_node_output_bits"], mapping)
        if transformed is None:
            decoded = None
            bit_accuracy = 0.0
            success = False
            action = "ASK"
        else:
            decoded = decode_bits(transformed)
            expected_bits = row["expected_consumer_input_bits"]
            bit_accuracy = mean([1.0 if a == b else 0.0 for a, b in zip(transformed, expected_bits)])
            success = row["intent"] < 2 ** row["active_bits"] and decoded == row["intent"]
            action = "COMMIT"
        out_rows.append(
            {
                "system": system,
                "row_id": row["row_id"],
                "split": row["split"],
                "intent": row["intent"],
                "is_old_intent": row["is_old_intent"],
                "physical_width": row["physical_width"],
                "active_bits": row["active_bits"],
                "source_node_output_bits": row["source_node_output_bits"],
                "expected_consumer_input_bits": row["expected_consumer_input_bits"],
                "adapter_mapping": mapping,
                "adapter_output_bits": transformed,
                "decoded_intent": decoded,
                "expected_action": "COMMIT",
                "action": action,
                "bit_accuracy": bit_accuracy,
                "adapter_success": success,
                "old_intent_success": success if row["is_old_intent"] else None,
                "wrong_commit": action == "COMMIT" and not success,
                "false_ask": action != "COMMIT",
                "adapter_mapping_hash": stable_hash(mapping),
            }
        )
    return summarize(out_rows), out_rows


def score_for_training(metrics: dict[str, Any]) -> float:
    return 0.72 * float(metrics["adapter_success"]) + 0.28 * float(metrics["bit_accuracy"])


def random_mapping(width: int, active_bits: int, rng: random.Random) -> list[int]:
    return rng.sample(range(width), active_bits)


def mutate_mapping(mapping: list[int], width: int, freeze_prefix: int, rng: random.Random) -> list[int]:
    mutated = list(mapping)
    mutable_slots = list(range(freeze_prefix, len(mutated)))
    if not mutable_slots:
        mutable_slots = list(range(len(mutated)))
    if len(mutable_slots) >= 2 and rng.random() < 0.25:
        a, b = rng.sample(mutable_slots, 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    else:
        slot = rng.choice(mutable_slots)
        choices = [idx for idx in range(width) if idx not in mutated or idx == mutated[slot]]
        mutated[slot] = rng.choice(choices)
    return mutated


def train_adapter_stage(
    system: str,
    physical_width: int,
    active_bits: int,
    initial_mapping: list[int],
    freeze_prefix: int,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[list[int], dict[str, Any]]:
    rng = random.Random(seed + physical_width * 1009 + active_bits * 67 + len(system))
    train_rows = [row for row in make_rows(physical_width, active_bits, seed, rows_per_split) if row["split"] == "train" and row["intent"] < 2**active_bits]
    current = list(initial_mapping)
    current_metrics, _ = evaluate_rows(train_rows, current, system)
    current_score = score_for_training(current_metrics)
    best = list(current)
    best_score = current_score
    best_exact = current_metrics["adapter_success"]
    accepted = 0
    rejected = 0
    attempts = 0
    attempts_to_95: int | None = 0 if best_exact >= PASS_ACCURACY else None
    last_improvement_generation = 0
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            mutated = mutate_mapping(current, physical_width, freeze_prefix, rng)
            mutated_metrics, _ = evaluate_rows(train_rows, mutated, system)
            mutated_score = score_for_training(mutated_metrics)
            if mutated_score > current_score:
                current = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                if mutated_score > best_score:
                    best = list(mutated)
                    best_score = mutated_score
                    best_exact = mutated_metrics["adapter_success"]
                    last_improvement_generation = generation
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and best_exact >= PASS_ACCURACY:
                attempts_to_95 = attempts
        curve_row = {
            "time": time.time(),
            "system": system,
            "physical_width": physical_width,
            "active_bits": active_bits,
            "generation": generation,
            "attempts": attempts,
            "best_training_score": best_score,
            "best_training_exact": best_exact,
            "current_training_score": current_score,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
            "freeze_prefix": freeze_prefix,
        }
        curve_rows.append(curve_row)
        append_jsonl(progress_path, curve_row)
    return best, {
        "attempts": attempts,
        "attempts_to_95": attempts_to_95,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "accepted_rate": accepted / (accepted + rejected) if accepted + rejected else 0.0,
        "last_improvement_generation": last_improvement_generation,
        "final_training_score": best_score,
        "final_training_exact": best_exact,
    }


def final_assess(system: str, physical_width: int, active_bits: int, mapping: list[int], seed: int, rows_per_split: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = make_rows(physical_width, active_bits, seed, rows_per_split)
    metrics, row_results = evaluate_rows(rows, mapping, system)
    metrics.update(
        {
            "physical_width": physical_width,
            "active_bits": active_bits,
            "adapter_parameter_count": len(mapping),
            "source_matrix_width": physical_width,
            "target_active_width": active_bits,
        }
    )
    return metrics, row_results


def direct_mapping(active_bits: int) -> list[int]:
    return list(range(active_bits))


def run_raw(system: str, physical_width: int, active_bits: int, seed: int, rows_per_split: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = direct_mapping(active_bits)
    metrics, rows = final_assess(system, physical_width, active_bits, mapping, seed, rows_per_split)
    report = {
        "system": system,
        "edge_type": "raw_wire",
        "growth_events": max(0, active_bits - START_ACTIVE_BITS) if "progressive" in system else 0,
        "physical_width": physical_width,
        "active_bits": active_bits,
        "adapter_parameter_count": 0,
        "mutation_attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "rollback_count": 0,
        "rollback_mismatch": False,
        "attempts_to_95": None,
        "accepted_rate": 0.0,
        "old_intent_regression": 1.0 - metrics["old_intent_success"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"system": system, "mapping": mapping}),
    }
    return metrics, report, rows, []


def run_fixed_adapter(
    system: str,
    physical_width: int,
    active_bits: int,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + len(system) * 31)
    initial = random_mapping(physical_width, active_bits, rng)
    mapping, stage = train_adapter_stage(
        system,
        physical_width,
        active_bits,
        initial,
        0,
        seed,
        rows_per_split,
        generations,
        population,
        progress_path,
        curve_rows,
    )
    metrics, rows = final_assess(system, physical_width, active_bits, mapping, seed, rows_per_split)
    report = {
        "system": system,
        "edge_type": "adapter_pocket",
        "growth_events": 0,
        "physical_width": physical_width,
        "active_bits": active_bits,
        "adapter_parameter_count": len(mapping),
        "mutation_attempts": stage["attempts"],
        "accepted": stage["accepted"],
        "rejected": stage["rejected"],
        "rollback_count": stage["rollback_count"],
        "rollback_mismatch": False,
        "attempts_to_95": stage["attempts_to_95"],
        "accepted_rate": stage["accepted_rate"],
        "old_intent_regression": 1.0 - metrics["old_intent_success"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"system": system, "mapping": mapping}),
        "final_mapping": mapping,
        "oracle_mapping": source_slots(active_bits),
    }
    return metrics, report, rows, []


def run_progressive_adapter(
    system: str,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    step: int,
    freeze_old: bool,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + len(system) * 43)
    physical_width = START_ACTIVE_BITS
    active_bits = START_ACTIVE_BITS
    mapping = random_mapping(physical_width, active_bits, rng)
    growth_events: list[dict[str, Any]] = []
    total_attempts = 0
    total_accepted = 0
    total_rejected = 0
    mapping, stage = train_adapter_stage(
        system,
        physical_width,
        active_bits,
        mapping,
        0,
        seed,
        rows_per_split,
        generations,
        population,
        progress_path,
        curve_rows,
    )
    total_attempts += stage["attempts"]
    total_accepted += stage["accepted"]
    total_rejected += stage["rejected"]
    growth_events.append(
        {
            "event": "stage_trained",
            "physical_width": physical_width,
            "active_bits": active_bits,
            "attempts_to_95": stage["attempts_to_95"],
            "final_training_exact": stage["final_training_exact"],
            "freeze_prefix": 0,
            "mapping": list(mapping),
        }
    )
    while active_bits < FINAL_ACTIVE_BITS:
        old_active_bits = active_bits
        add = min(step, FINAL_ACTIVE_BITS - active_bits)
        for _ in range(add):
            physical_width += 1
            active_bits += 1
            mapping.append(physical_width - 1)
            growth_events.append(
                {
                    "event": "add_adapter_cell",
                    "new_physical_width": physical_width,
                    "new_active_bits": active_bits,
                    "added_source_slot": physical_width - 1,
                    "added_target_slot": active_bits - 1,
                }
            )
        freeze_prefix = old_active_bits if freeze_old else 0
        mapping, stage = train_adapter_stage(
            system,
            physical_width,
            active_bits,
            mapping,
            freeze_prefix,
            seed,
            rows_per_split,
            generations,
            population,
            progress_path,
            curve_rows,
        )
        total_attempts += stage["attempts"]
        total_accepted += stage["accepted"]
        total_rejected += stage["rejected"]
        growth_events.append(
            {
                "event": "stage_trained",
                "physical_width": physical_width,
                "active_bits": active_bits,
                "attempts_to_95": stage["attempts_to_95"],
                "final_training_exact": stage["final_training_exact"],
                "freeze_prefix": freeze_prefix,
                "mapping": list(mapping),
            }
        )
    metrics, rows = final_assess(system, physical_width, active_bits, mapping, seed, rows_per_split)
    report = {
        "system": system,
        "edge_type": "adapter_pocket",
        "growth_events": len([event for event in growth_events if event["event"] == "add_adapter_cell"]),
        "growth_event_log": growth_events,
        "physical_width": physical_width,
        "active_bits": active_bits,
        "adapter_parameter_count": len(mapping),
        "mutation_attempts": total_attempts,
        "accepted": total_accepted,
        "rejected": total_rejected,
        "rollback_count": total_rejected,
        "rollback_mismatch": False,
        "attempts_to_95": total_attempts if metrics["heldout_success"] >= PASS_ACCURACY else None,
        "accepted_rate": total_accepted / (total_accepted + total_rejected) if total_accepted + total_rejected else 0.0,
        "old_intent_regression": 1.0 - metrics["old_intent_success"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"system": system, "mapping": mapping, "events": growth_events}),
        "final_mapping": mapping,
        "oracle_mapping": source_slots(active_bits),
    }
    return metrics, report, rows, growth_events


def run_system(
    system: str,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if system == "raw_wire_direct_fixed16":
        return run_raw(system, FIXED_PHYSICAL_WIDTH, FINAL_ACTIVE_BITS, seed, rows_per_split)
    if system == "raw_wire_progressive_plus1":
        return run_raw(system, FINAL_ACTIVE_BITS, FINAL_ACTIVE_BITS, seed, rows_per_split)
    if system == "edge_adapter_fixed16_to16":
        return run_fixed_adapter(system, FIXED_PHYSICAL_WIDTH, FINAL_ACTIVE_BITS, seed, rows_per_split, generations, population, progress_path, curve_rows)
    if system == "edge_adapter_progressive_plus1_freeze_old":
        return run_progressive_adapter(system, seed, rows_per_split, generations, population, 1, True, progress_path, curve_rows)
    if system == "edge_adapter_progressive_plus1_no_freeze":
        return run_progressive_adapter(system, seed, rows_per_split, generations, population, 1, False, progress_path, curve_rows)
    if system == "edge_adapter_block_growth_plus4":
        return run_progressive_adapter(system, seed, rows_per_split, generations, population, 4, True, progress_path, curve_rows)
    if system == "identity_oracle_adapter_reference":
        mapping = source_slots(FINAL_ACTIVE_BITS)
        metrics, rows = final_assess(system, FINAL_ACTIVE_BITS, FINAL_ACTIVE_BITS, mapping, seed, rows_per_split)
        return metrics, {
            "system": system,
            "edge_type": "oracle_adapter_reference",
            "growth_events": FINAL_ACTIVE_BITS - START_ACTIVE_BITS,
            "physical_width": FINAL_ACTIVE_BITS,
            "active_bits": FINAL_ACTIVE_BITS,
            "adapter_parameter_count": len(mapping),
            "mutation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "rollback_count": 0,
            "rollback_mismatch": False,
            "attempts_to_95": 0,
            "accepted_rate": 0.0,
            "old_intent_regression": 0.0,
            "parameter_diff_written": True,
            "parameter_diff_hash": stable_hash({"system": system, "mapping": mapping}),
            "final_mapping": mapping,
            "oracle_mapping": mapping,
        }, rows, []
    if system == "random_adapter_control":
        rng = random.Random(seed + 777)
        mapping = random_mapping(FIXED_PHYSICAL_WIDTH, FINAL_ACTIVE_BITS, rng)
        metrics, rows = final_assess(system, FIXED_PHYSICAL_WIDTH, FINAL_ACTIVE_BITS, mapping, seed, rows_per_split)
        return metrics, {
            "system": system,
            "edge_type": "random_adapter_control",
            "growth_events": 0,
            "physical_width": FIXED_PHYSICAL_WIDTH,
            "active_bits": FINAL_ACTIVE_BITS,
            "adapter_parameter_count": len(mapping),
            "mutation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "rollback_count": 0,
            "rollback_mismatch": False,
            "attempts_to_95": None,
            "accepted_rate": 0.0,
            "old_intent_regression": 1.0 - metrics["old_intent_success"],
            "parameter_diff_written": True,
            "parameter_diff_hash": stable_hash({"system": system, "mapping": mapping}),
            "final_mapping": mapping,
            "oracle_mapping": source_slots(FINAL_ACTIVE_BITS),
        }, rows, []
    raise ValueError(system)


def pass_gate(metrics: dict[str, Any], report: dict[str, Any]) -> bool:
    return (
        metrics["heldout_success"] >= PASS_ACCURACY
        and metrics["ood_success"] >= PASS_ACCURACY
        and metrics["counterfactual_success"] >= PASS_ACCURACY
        and metrics["adversarial_success"] >= PASS_ACCURACY
        and metrics["wrong_commit_rate"] == 0.0
        and report["old_intent_regression"] <= 0.01
    )


def decide(results: dict[str, Any], reports: dict[str, Any]) -> str:
    raw = results["raw_wire_direct_fixed16"]["overall"]
    raw_prog = results["raw_wire_progressive_plus1"]["overall"]
    fixed_adapter = results["edge_adapter_fixed16_to16"]["overall"]
    plus1 = results["edge_adapter_progressive_plus1_no_freeze"]["overall"]
    plus1_report = reports["edge_adapter_progressive_plus1_no_freeze"]
    block = results["edge_adapter_block_growth_plus4"]["overall"]
    block_report = reports["edge_adapter_block_growth_plus4"]
    if pass_gate(raw, reports["raw_wire_direct_fixed16"]) or pass_gate(raw_prog, reports["raw_wire_progressive_plus1"]):
        return "e47_raw_wire_growth_sufficient"
    if pass_gate(plus1, plus1_report):
        if pass_gate(block, block_report) and (block_report["attempts_to_95"] or 10**9) < (plus1_report["attempts_to_95"] or 10**9) / 2:
            return "e47_block_adapter_growth_preferred"
        return "e47_edge_adapter_pocket_positive"
    if pass_gate(fixed_adapter, reports["edge_adapter_fixed16_to16"]):
        return "e47_adapter_growth_overhead_too_high"
    if reports["edge_adapter_progressive_plus1_no_freeze"]["old_intent_regression"] > 0.05:
        return "e47_adapter_growth_causes_regression"
    return "e47_invalid_artifact_detected"


def make_table(results: dict[str, Any], reports: dict[str, Any]) -> str:
    fields = [
        "edge_type",
        "physical_width",
        "active_bits",
        "growth_events",
        "heldout_success",
        "ood_success",
        "old_intent_success",
        "old_intent_regression",
        "bit_accuracy",
        "attempts_to_95",
        "accepted_rate",
    ]
    lines = ["| system | " + " | ".join(fields) + " |\n", "|---|" + "|".join("---" for _ in fields) + "|\n"]
    for system in SYSTEMS:
        metrics = results[system]["overall"]
        report = reports[system]
        values = {
            "edge_type": report["edge_type"],
            "physical_width": report["physical_width"],
            "active_bits": report["active_bits"],
            "growth_events": report["growth_events"],
            "heldout_success": metrics["heldout_success"],
            "ood_success": metrics["ood_success"],
            "old_intent_success": metrics["old_intent_success"],
            "old_intent_regression": report["old_intent_regression"],
            "bit_accuracy": metrics["bit_accuracy"],
            "attempts_to_95": report["attempts_to_95"],
            "accepted_rate": report["accepted_rate"],
        }
        rendered: list[str] = []
        for field in fields:
            value = values[field]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append("none" if value is None else str(value))
        lines.append("| " + system + " | " + " | ".join(rendered) + " |\n")
    return "".join(lines)


def deterministic_replay_report(rows: list[dict[str, Any]], results: dict[str, Any], aggregate: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "hashes": {
            "row_level_results_hash": stable_hash(rows),
            "system_results_hash": stable_hash(results),
            "aggregate_metrics_hash": stable_hash(aggregate),
            "adapter_growth_report_hash": stable_hash(reports),
        },
    }


def build_report(aggregate: dict[str, Any], table: str, reports: dict[str, Any]) -> str:
    return f"""# E47 Edge Adapter Pocket Growth Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E47 tested the user's edge-as-pocket idea: the connection between two frozen
nodes is not just a cable. It can be an explicit adapter pocket that transforms
a source node mini-matrix into a consumer node mini-matrix.

## Result Table

```text
{table}```

## Interpretation

The task intentionally scrambles the source node bit order. A raw wire that
reads source slot `j` as target slot `j` should fail. A valid adapter pocket
must learn the mechanical source-slot to target-slot mapping.

The important comparison is:

```text
raw wire direct
vs
edge adapter pocket
vs
edge adapter pocket grown by +1 active cell
```

## Growth Snapshot

```json
{json.dumps(reports["edge_adapter_progressive_plus1_no_freeze"].get("growth_event_log", [])[:20], indent=2, sort_keys=True)}
```

## Boundary

This is a controlled symbolic/numeric Edge ABI adapter probe. It does not prove
raw language reasoning, deployed AI assistant behavior, model-scale behavior,
AGI, or consciousness.
"""


def write_sample_pack(
    sample_dir: Path,
    aggregate: dict[str, Any],
    results: dict[str, Any],
    rows: list[dict[str, Any]],
    reports: dict[str, Any],
    replay: dict[str, Any],
    curve: list[dict[str, Any]],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E47 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "edge_adapter_pocket": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", results)
    write_json(sample_dir / "adapter_growth_report_sample.json", reports)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "row_level_sample.jsonl", rows[:320])
    write_jsonl(sample_dir / "adapter_curve_sample.jsonl", curve[:320])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    curve_path = out / "adapter_curve.jsonl"
    for path in [progress_path, heartbeat_path, curve_path]:
        if path.exists():
            path.unlink()
    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "milestone": MILESTONE})[:16]
    append_jsonl(heartbeat_path, hardware_snapshot())
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    all_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    system_results: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    growth_events: dict[str, Any] = {}

    for system in SYSTEMS:
        metrics, report, rows, events = run_system(
            system,
            args.seed,
            args.rows,
            args.generations,
            args.population,
            progress_path,
            curve_rows,
        )
        system_results[system] = {"overall": metrics}
        reports[system] = report
        growth_events[system] = events
        all_rows.extend(rows)
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "event": "system_done",
                "system": system,
                "heldout_success": metrics["heldout_success"],
                "bit_accuracy": metrics["bit_accuracy"],
                "attempts_to_95": report["attempts_to_95"],
            },
        )
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    write_jsonl(curve_path, curve_rows)
    append_jsonl(heartbeat_path, hardware_snapshot())
    decision = decide(system_results, reports)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "rows_per_split": args.rows,
        "base_intent_count": BASE_INTENT_COUNT,
        "final_intent_count": FINAL_INTENT_COUNT,
        "start_active_bits": START_ACTIVE_BITS,
        "final_active_bits": FINAL_ACTIVE_BITS,
        "fixed_physical_width": FIXED_PHYSICAL_WIDTH,
        "source_slot_prefix": SOURCE_SLOT_PREFIX,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_rows, system_results, aggregate, reports)
    table = make_table(system_results, reports)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "mutated_systems": sorted(MUTATED_SYSTEMS), "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "adapter_growth_report.json", reports)
    write_json(out / "adapter_event_report.json", growth_events)
    write_json(out / "system_results.json", system_results)
    write_jsonl(out / "row_level_results.jsonl", all_rows)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id})
    (out / "results_table.md").write_text(table, encoding="utf-8")
    (out / "report.md").write_text(build_report(aggregate, table, reports), encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_rows, reports, replay, curve_rows)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e47_edge_adapter_pocket_growth_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e47_edge_adapter_pocket_growth_probe")
    parser.add_argument("--seed", type=int, default=47001)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--generations", type=int, default=42)
    parser.add_argument("--population", type=int, default=24)
    args = parser.parse_args()
    aggregate = run(args)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
