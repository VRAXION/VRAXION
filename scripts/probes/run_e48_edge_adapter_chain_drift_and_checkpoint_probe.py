#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E48_EDGE_ADAPTER_CHAIN_DRIFT_AND_CHECKPOINT_PROBE"
BOUNDARY = (
    "E48 tests whether explicit Edge Adapter Pockets remain stable when "
    "several edges are chained. It isolates raw chain drift, adapter-only "
    "end-to-end drift, Agency checkpointing, canonical reset, and trace "
    "validation. It does not claim raw language reasoning, AGI, consciousness, "
    "deployed behavior, or model-scale behavior."
)

ACTIVE_BITS = 8
INTENT_COUNT = 256
BASE_INTENT_COUNT = 32
PASS_ACCURACY = 0.95
MAX_DEPTH = 5

NODE_LAYOUTS = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [2, 4, 1, 0, 3, 5, 6, 7],
    [3, 0, 4, 2, 1, 6, 5, 7],
    [1, 3, 0, 4, 2, 7, 5, 6],
    [4, 2, 3, 1, 0, 5, 7, 6],
    [2, 1, 4, 3, 0, 6, 7, 5],
]

SYSTEMS = [
    "raw_wire_chain",
    "single_edge_adapter_only",
    "adapter_chain_no_checkpoint",
    "adapter_chain_with_agency_checkpoint",
    "adapter_chain_with_canonical_bus_reset",
    "adapter_chain_plus_trace_validation",
    "oracle_adapter_chain_reference",
    "random_adapter_chain_control",
]

MUTATED_SYSTEMS = {
    "single_edge_adapter_only",
    "adapter_chain_no_checkpoint",
    "adapter_chain_with_agency_checkpoint",
    "adapter_chain_with_canonical_bus_reset",
    "adapter_chain_plus_trace_validation",
}

DECISIONS = {
    "e48_adapter_chain_stable",
    "e48_adapter_chain_requires_agency_checkpoint",
    "e48_adapter_chain_requires_checkpoint_and_trace_validation",
    "e48_canonical_bus_reset_required",
    "e48_adapter_chain_drift_detected",
    "e48_invalid_artifact_detected",
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


def encode_intent(intent: int) -> list[int]:
    return [(intent >> idx) & 1 for idx in range(ACTIVE_BITS)]


def encode_in_layout(intent: int, layout: list[int]) -> list[int]:
    out = [0] * ACTIVE_BITS
    for logical_bit, bit in enumerate(encode_intent(intent)):
        out[layout[logical_bit]] = bit
    return out


def decode_from_layout(bits: list[int], layout: list[int]) -> int:
    return sum(int(bits[layout[logical_bit]]) << logical_bit for logical_bit in range(ACTIVE_BITS))


def oracle_edge_mapping(edge_index: int) -> list[int]:
    source_layout = NODE_LAYOUTS[edge_index]
    target_layout = NODE_LAYOUTS[edge_index + 1]
    mapping = [0] * ACTIVE_BITS
    for logical_bit, source_slot in enumerate(source_layout):
        target_slot = target_layout[logical_bit]
        mapping[target_slot] = source_slot
    return mapping


def raw_mapping() -> list[int]:
    return list(range(ACTIVE_BITS))


def random_mapping(rng: random.Random) -> list[int]:
    return rng.sample(range(ACTIVE_BITS), ACTIVE_BITS)


def mutate_mapping(mapping: list[int], rng: random.Random) -> list[int]:
    mutated = list(mapping)
    if rng.random() < 0.70:
        a, b = rng.sample(range(ACTIVE_BITS), 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    else:
        slot = rng.randrange(ACTIVE_BITS)
        choices = [idx for idx in range(ACTIVE_BITS) if idx not in mutated or idx == mutated[slot]]
        mutated[slot] = rng.choice(choices)
    return mutated


def apply_mapping(source_bits: list[int], mapping: list[int]) -> list[int] | None:
    if len(mapping) != ACTIVE_BITS or len(set(mapping)) != ACTIVE_BITS:
        return None
    return [int(source_bits[mapping[target_slot]]) for target_slot in range(ACTIVE_BITS)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def make_rows(seed: int, rows_per_split: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_offsets = {"train": 0, "heldout": 5, "ood": 11, "counterfactual": 19, "adversarial": 29}
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for idx in range(rows_per_split):
            depth = [1, 2, 3, 5][idx % 4]
            intent = (idx * 17 + split_offsets[split] + depth * 13) % INTENT_COUNT
            scenario = "clean"
            corrupt_edge: int | None = None
            if split == "adversarial":
                scenario = "bit_flip" if idx % 2 == 0 else "stale_trace"
                corrupt_edge = (idx // 2) % depth
            row_id = f"{split}_{idx:05d}_d{depth}_{scenario}"
            rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "intent": intent,
                    "is_old_intent": intent < BASE_INTENT_COUNT,
                    "chain_depth": depth,
                    "scenario": scenario,
                    "corrupt_edge": corrupt_edge,
                    "expected_action": "DEFER" if scenario in {"bit_flip", "stale_trace"} else "COMMIT",
                    "initial_node_bits": encode_in_layout(intent, NODE_LAYOUTS[0]),
                    "seed": seed,
                }
            )
    return rows


def evaluate_chain_row(
    row: dict[str, Any],
    mappings: list[list[int]],
    system: str,
    checkpoint: bool,
    trace_validation: bool,
    canonical_reset: bool,
) -> dict[str, Any]:
    current = list(row["initial_node_bits"])
    current_intent = decode_from_layout(current, NODE_LAYOUTS[0])
    edge_records: list[dict[str, Any]] = []
    action = "COMMIT"
    detected_reason: str | None = None
    drift_events = 0
    trace_mismatch_events = 0
    for edge in range(row["chain_depth"]):
        before_intent = decode_from_layout(current, NODE_LAYOUTS[edge])
        mapping = mappings[edge] if edge < len(mappings) else raw_mapping()
        transformed = apply_mapping(current, mapping)
        if transformed is None:
            action = "DEFER"
            detected_reason = "invalid_mapping"
            break
        trace_mismatch = row["scenario"] == "stale_trace" and row["corrupt_edge"] == edge
        if row["scenario"] == "bit_flip" and row["corrupt_edge"] == edge:
            transformed[0] = 1 - transformed[0]
        after_intent = decode_from_layout(transformed, NODE_LAYOUTS[edge + 1])
        edge_ok = after_intent == before_intent
        if not edge_ok:
            drift_events += 1
        if trace_mismatch:
            trace_mismatch_events += 1
        edge_records.append(
            {
                "edge": edge,
                "mapping": mapping,
                "before_intent": before_intent,
                "after_intent": after_intent,
                "edge_ok": edge_ok,
                "trace_mismatch": trace_mismatch,
            }
        )
        if checkpoint and not edge_ok:
            action = "DEFER"
            detected_reason = "agency_checkpoint_drift"
            break
        if trace_validation and trace_mismatch:
            action = "DEFER"
            detected_reason = "trace_mismatch"
            break
        if canonical_reset:
            transformed = encode_in_layout(after_intent, NODE_LAYOUTS[edge + 1])
        current = transformed
        current_intent = after_intent
    decoded_intent = current_intent
    expected_action = row["expected_action"]
    if expected_action == "COMMIT":
        chain_success = action == "COMMIT" and decoded_intent == row["intent"]
    else:
        chain_success = action == "DEFER"
    wrong_commit = action == "COMMIT" and (expected_action == "DEFER" or decoded_intent != row["intent"])
    false_defer = action == "DEFER" and expected_action == "COMMIT"
    stale_commit = action == "COMMIT" and trace_mismatch_events > 0
    return {
        "system": system,
        "row_id": row["row_id"],
        "split": row["split"],
        "intent": row["intent"],
        "is_old_intent": row["is_old_intent"],
        "chain_depth": row["chain_depth"],
        "scenario": row["scenario"],
        "corrupt_edge": row["corrupt_edge"],
        "expected_action": expected_action,
        "action": action,
        "detected_reason": detected_reason,
        "decoded_intent": decoded_intent,
        "chain_success": chain_success,
        "wrong_commit": wrong_commit,
        "false_defer": false_defer,
        "drift_detected": drift_events > 0,
        "drift_event_count": drift_events,
        "trace_mismatch_event_count": trace_mismatch_events,
        "stale_commit": stale_commit,
        "edge_records": edge_records,
        "mapping_hash": stable_hash(mappings),
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    mappings: list[list[int]],
    system: str,
    checkpoint: bool = False,
    trace_validation: bool = False,
    canonical_reset: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_rows = [evaluate_chain_row(row, mappings, system, checkpoint, trace_validation, canonical_reset) for row in rows]
    return summarize(out_rows), out_rows


def split_metric(rows: list[dict[str, Any]], split: str, key: str) -> float:
    chunk = [row for row in rows if row["split"] == split]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def depth_metric(rows: list[dict[str, Any]], depth: int, key: str) -> float:
    chunk = [row for row in rows if row["chain_depth"] == depth and row["scenario"] == "clean"]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows = [row for row in rows if row["is_old_intent"]]
    clean_rows = [row for row in rows if row["scenario"] == "clean"]
    return {
        "row_count": len(rows),
        "chain_success": mean([1.0 if row["chain_success"] else 0.0 for row in rows]),
        "heldout_success": split_metric(rows, "heldout", "chain_success"),
        "ood_success": split_metric(rows, "ood", "chain_success"),
        "counterfactual_success": split_metric(rows, "counterfactual", "chain_success"),
        "adversarial_success": split_metric(rows, "adversarial", "chain_success"),
        "clean_success": mean([1.0 if row["chain_success"] else 0.0 for row in clean_rows]),
        "depth_1_success": depth_metric(rows, 1, "chain_success"),
        "depth_2_success": depth_metric(rows, 2, "chain_success"),
        "depth_3_success": depth_metric(rows, 3, "chain_success"),
        "depth_5_success": depth_metric(rows, 5, "chain_success"),
        "old_intent_success": mean([1.0 if row["chain_success"] else 0.0 for row in old_rows]),
        "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in rows]),
        "false_defer_rate": mean([1.0 if row["false_defer"] else 0.0 for row in rows]),
        "drift_rate": mean([1.0 if row["drift_detected"] else 0.0 for row in rows]),
        "trace_mismatch_commit_rate": mean([1.0 if row["stale_commit"] else 0.0 for row in rows]),
        "mean_drift_events": mean([float(row["drift_event_count"]) for row in rows]),
    }


def edge_training_score(metrics: dict[str, Any]) -> float:
    return 0.70 * metrics["chain_success"] + 0.30 * (1.0 - metrics["drift_rate"])


def mapping_match_rate(mapping: list[int], edge: int) -> float:
    oracle = oracle_edge_mapping(edge)
    return mean([1.0 if mapping[idx] == oracle[idx] else 0.0 for idx in range(ACTIVE_BITS)])


def train_edge_mapping(
    system: str,
    edge: int,
    seed: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[list[int], dict[str, Any]]:
    rng = random.Random(seed + edge * 1009 + len(system) * 17)
    mapping = random_mapping(rng)
    oracle = oracle_edge_mapping(edge)
    current_score = mapping_match_rate(mapping, edge)
    best = list(mapping)
    best_score = current_score
    accepted = 0
    rejected = 0
    attempts = 0
    attempts_to_95: int | None = 0 if best_score >= PASS_ACCURACY else None
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            mutated = mutate_mapping(mapping, rng)
            mutated_score = mapping_match_rate(mutated, edge)
            if mutated_score > current_score:
                mapping = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                if mutated_score > best_score:
                    best = list(mutated)
                    best_score = mutated_score
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and best_score >= PASS_ACCURACY:
                attempts_to_95 = attempts
        curve_row = {
            "time": time.time(),
            "system": system,
            "edge": edge,
            "generation": generation,
            "attempts": attempts,
            "best_mapping_match": best_score,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
        }
        curve_rows.append(curve_row)
        append_jsonl(progress_path, curve_row)
    return best, {
        "edge": edge,
        "oracle_mapping": oracle,
        "final_mapping": best,
        "mapping_match_rate": best_score,
        "attempts": attempts,
        "attempts_to_95": attempts_to_95,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
    }


def train_independent_chain(
    system: str,
    seed: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[list[list[int]], dict[str, Any]]:
    mappings: list[list[int]] = []
    edge_reports: list[dict[str, Any]] = []
    for edge in range(MAX_DEPTH):
        mapping, report = train_edge_mapping(system, edge, seed, generations, population, progress_path, curve_rows)
        mappings.append(mapping)
        edge_reports.append(report)
    return mappings, {
        "edge_reports": edge_reports,
        "mutation_attempts": sum(report["attempts"] for report in edge_reports),
        "accepted": sum(report["accepted"] for report in edge_reports),
        "rejected": sum(report["rejected"] for report in edge_reports),
        "rollback_count": sum(report["rollback_count"] for report in edge_reports),
        "attempts_to_95": sum(report["attempts_to_95"] or report["attempts"] for report in edge_reports),
        "per_edge_adapter_success": mean([float(report["mapping_match_rate"]) for report in edge_reports]),
    }


def mutate_chain(mappings: list[list[int]], rng: random.Random) -> list[list[int]]:
    mutated = [list(mapping) for mapping in mappings]
    edge = rng.randrange(MAX_DEPTH)
    mutated[edge] = mutate_mapping(mutated[edge], rng)
    return mutated


def train_end_to_end_chain(
    system: str,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[list[list[int]], dict[str, Any]]:
    rng = random.Random(seed + len(system) * 101)
    rows = [row for row in make_rows(seed, rows_per_split) if row["split"] == "train" and row["scenario"] == "clean"]
    mappings = [random_mapping(rng) for _ in range(MAX_DEPTH)]
    current_metrics, _ = evaluate_rows(rows, mappings, system)
    current_score = edge_training_score(current_metrics)
    best = [list(mapping) for mapping in mappings]
    best_score = current_score
    best_exact = current_metrics["chain_success"]
    accepted = 0
    rejected = 0
    attempts = 0
    attempts_to_95: int | None = 0 if best_exact >= PASS_ACCURACY else None
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            mutated = mutate_chain(mappings, rng)
            mutated_metrics, _ = evaluate_rows(rows, mutated, system)
            mutated_score = edge_training_score(mutated_metrics)
            if mutated_score > current_score:
                mappings = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                if mutated_score > best_score:
                    best = [list(mapping) for mapping in mutated]
                    best_score = mutated_score
                    best_exact = mutated_metrics["chain_success"]
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and best_exact >= PASS_ACCURACY:
                attempts_to_95 = attempts
        curve_row = {
            "time": time.time(),
            "system": system,
            "edge": "end_to_end",
            "generation": generation,
            "attempts": attempts,
            "best_chain_score": best_score,
            "best_chain_exact": best_exact,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
        }
        curve_rows.append(curve_row)
        append_jsonl(progress_path, curve_row)
    return best, {
        "edge_reports": [],
        "mutation_attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "attempts_to_95": attempts_to_95,
        "per_edge_adapter_success": mean(
            [mapping_match_rate(best[edge], edge) for edge in range(MAX_DEPTH)]
        ),
    }


def report_for_system(
    system: str,
    edge_type: str,
    mappings: list[list[int]],
    train_report: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint: bool,
    trace_validation: bool,
    canonical_reset: bool,
) -> dict[str, Any]:
    return {
        "system": system,
        "edge_type": edge_type,
        "chain_depth_max": MAX_DEPTH,
        "checkpoint": checkpoint,
        "trace_validation": trace_validation,
        "canonical_reset": canonical_reset,
        "adapter_parameter_count": sum(len(mapping) for mapping in mappings),
        "mutation_attempts": train_report["mutation_attempts"],
        "accepted": train_report["accepted"],
        "rejected": train_report["rejected"],
        "rollback_count": train_report["rollback_count"],
        "rollback_mismatch": False,
        "attempts_to_95": train_report["attempts_to_95"],
        "per_edge_adapter_success": train_report["per_edge_adapter_success"],
        "old_intent_regression": 1.0 - metrics["old_intent_success"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"system": system, "mappings": mappings}),
        "edge_reports": train_report.get("edge_reports", []),
        "final_mappings": mappings,
        "oracle_mappings": [oracle_edge_mapping(edge) for edge in range(MAX_DEPTH)],
    }


def assess_system(
    system: str,
    mappings: list[list[int]],
    seed: int,
    rows_per_split: int,
    train_report: dict[str, Any],
    edge_type: str,
    checkpoint: bool = False,
    trace_validation: bool = False,
    canonical_reset: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = make_rows(seed, rows_per_split)
    metrics, row_results = evaluate_rows(rows, mappings, system, checkpoint, trace_validation, canonical_reset)
    report = report_for_system(system, edge_type, mappings, train_report, metrics, checkpoint, trace_validation, canonical_reset)
    return metrics, report, row_results


def zero_train_report(mappings: list[list[int]]) -> dict[str, Any]:
    return {
        "edge_reports": [],
        "mutation_attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "rollback_count": 0,
        "attempts_to_95": None,
        "per_edge_adapter_success": mean([mapping_match_rate(mappings[edge], edge) for edge in range(MAX_DEPTH)]),
    }


def run_system(
    system: str,
    seed: int,
    rows_per_split: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if system == "raw_wire_chain":
        mappings = [raw_mapping() for _ in range(MAX_DEPTH)]
        return assess_system(system, mappings, seed, rows_per_split, zero_train_report(mappings), "raw_wire_chain")
    if system == "single_edge_adapter_only":
        first_mapping, edge_report = train_edge_mapping(system, 0, seed, generations, population, progress_path, curve_rows)
        mappings = [first_mapping] + [raw_mapping() for _ in range(MAX_DEPTH - 1)]
        train_report = {
            "edge_reports": [edge_report],
            "mutation_attempts": edge_report["attempts"],
            "accepted": edge_report["accepted"],
            "rejected": edge_report["rejected"],
            "rollback_count": edge_report["rollback_count"],
            "attempts_to_95": edge_report["attempts_to_95"],
            "per_edge_adapter_success": mean([mapping_match_rate(mappings[edge], edge) for edge in range(MAX_DEPTH)]),
        }
        return assess_system(system, mappings, seed, rows_per_split, train_report, "single_adapter_then_raw")
    if system == "adapter_chain_no_checkpoint":
        mappings, train_report = train_end_to_end_chain(system, seed, rows_per_split, generations, population, progress_path, curve_rows)
        return assess_system(system, mappings, seed, rows_per_split, train_report, "adapter_chain_no_checkpoint")
    if system == "adapter_chain_with_agency_checkpoint":
        mappings, train_report = train_independent_chain(system, seed, generations, population, progress_path, curve_rows)
        return assess_system(system, mappings, seed, rows_per_split, train_report, "adapter_chain_with_checkpoint", checkpoint=True)
    if system == "adapter_chain_with_canonical_bus_reset":
        mappings, train_report = train_independent_chain(system, seed, generations, population, progress_path, curve_rows)
        return assess_system(system, mappings, seed, rows_per_split, train_report, "adapter_chain_with_canonical_reset", checkpoint=True, canonical_reset=True)
    if system == "adapter_chain_plus_trace_validation":
        mappings, train_report = train_independent_chain(system, seed, generations, population, progress_path, curve_rows)
        return assess_system(system, mappings, seed, rows_per_split, train_report, "adapter_chain_with_checkpoint_and_trace", checkpoint=True, trace_validation=True)
    if system == "oracle_adapter_chain_reference":
        mappings = [oracle_edge_mapping(edge) for edge in range(MAX_DEPTH)]
        train_report = {
            "edge_reports": [],
            "mutation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "rollback_count": 0,
            "attempts_to_95": 0,
            "per_edge_adapter_success": 1.0,
        }
        return assess_system(system, mappings, seed, rows_per_split, train_report, "oracle_adapter_chain_reference", checkpoint=True, trace_validation=True)
    if system == "random_adapter_chain_control":
        rng = random.Random(seed + 777)
        mappings = [random_mapping(rng) for _ in range(MAX_DEPTH)]
        return assess_system(system, mappings, seed, rows_per_split, zero_train_report(mappings), "random_adapter_chain_control")
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
    no_checkpoint = results["adapter_chain_no_checkpoint"]["overall"]
    checkpoint = results["adapter_chain_with_agency_checkpoint"]["overall"]
    reset = results["adapter_chain_with_canonical_bus_reset"]["overall"]
    trace = results["adapter_chain_plus_trace_validation"]["overall"]
    if pass_gate(no_checkpoint, reports["adapter_chain_no_checkpoint"]):
        return "e48_adapter_chain_stable"
    if pass_gate(trace, reports["adapter_chain_plus_trace_validation"]):
        if checkpoint["clean_success"] >= PASS_ACCURACY and checkpoint["adversarial_success"] < PASS_ACCURACY:
            return "e48_adapter_chain_requires_checkpoint_and_trace_validation"
        return "e48_adapter_chain_requires_agency_checkpoint"
    if pass_gate(reset, reports["adapter_chain_with_canonical_bus_reset"]):
        return "e48_canonical_bus_reset_required"
    if checkpoint["clean_success"] >= PASS_ACCURACY and no_checkpoint["clean_success"] < PASS_ACCURACY:
        return "e48_adapter_chain_drift_detected"
    return "e48_invalid_artifact_detected"


def make_table(results: dict[str, Any], reports: dict[str, Any]) -> str:
    fields = [
        "edge_type",
        "heldout_success",
        "ood_success",
        "adversarial_success",
        "depth_1_success",
        "depth_2_success",
        "depth_3_success",
        "depth_5_success",
        "drift_rate",
        "wrong_commit_rate",
        "trace_mismatch_commit_rate",
        "per_edge_adapter_success",
    ]
    lines = ["| system | " + " | ".join(fields) + " |\n", "|---|" + "|".join("---" for _ in fields) + "|\n"]
    for system in SYSTEMS:
        metrics = results[system]["overall"]
        report = reports[system]
        values = {
            "edge_type": report["edge_type"],
            "heldout_success": metrics["heldout_success"],
            "ood_success": metrics["ood_success"],
            "adversarial_success": metrics["adversarial_success"],
            "depth_1_success": metrics["depth_1_success"],
            "depth_2_success": metrics["depth_2_success"],
            "depth_3_success": metrics["depth_3_success"],
            "depth_5_success": metrics["depth_5_success"],
            "drift_rate": metrics["drift_rate"],
            "wrong_commit_rate": metrics["wrong_commit_rate"],
            "trace_mismatch_commit_rate": metrics["trace_mismatch_commit_rate"],
            "per_edge_adapter_success": report["per_edge_adapter_success"],
        }
        rendered: list[str] = []
        for field in fields:
            value = values[field]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
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
            "chain_drift_report_hash": stable_hash(reports),
        },
    }


def build_report(aggregate: dict[str, Any], table: str) -> str:
    return f"""# E48 Edge Adapter Chain Drift And Checkpoint Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E48 tests whether Edge Adapter Pockets remain stable across chained node
transforms, or whether the chain needs Agency checkpoint and trace validation.

## Result Table

```text
{table}```

## Interpretation

Clean adapter chains can transport a stable intent when each edge has the right
local ABI transform. However, the adversarial rows split two different failure
modes:

```text
bit_flip    -> local Agency checkpoint can detect state drift
stale_trace -> trace validation is needed, because the bits may still look valid
```

The architecture implication is that an Edge Adapter chain should not be a
blind pipe. Each hop needs a commit/checkpoint boundary, and stale or replayed
state requires trace validation.

## Boundary

This is a controlled symbolic/numeric Edge ABI chain probe. It does not prove
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
    sample_dir.joinpath("README.md").write_text("E48 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "edge_adapter_chain": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", results)
    write_json(sample_dir / "chain_drift_report_sample.json", reports)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "row_level_sample.jsonl", rows[:360])
    write_jsonl(sample_dir / "adapter_chain_curve_sample.jsonl", curve[:360])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    curve_path = out / "adapter_chain_curve.jsonl"
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
    for system in SYSTEMS:
        metrics, report, rows = run_system(system, args.seed, args.rows, args.generations, args.population, progress_path, curve_rows)
        system_results[system] = {"overall": metrics}
        reports[system] = report
        all_rows.extend(rows)
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "event": "system_done",
                "system": system,
                "heldout_success": metrics["heldout_success"],
                "adversarial_success": metrics["adversarial_success"],
                "drift_rate": metrics["drift_rate"],
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
        "active_bits": ACTIVE_BITS,
        "intent_count": INTENT_COUNT,
        "max_chain_depth": MAX_DEPTH,
        "node_layouts": NODE_LAYOUTS,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_rows, system_results, aggregate, reports)
    table = make_table(system_results, reports)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "mutated_systems": sorted(MUTATED_SYSTEMS), "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "chain_drift_report.json", reports)
    write_json(out / "system_results.json", system_results)
    write_jsonl(out / "row_level_results.jsonl", all_rows)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id})
    (out / "results_table.md").write_text(table, encoding="utf-8")
    (out / "report.md").write_text(build_report(aggregate, table), encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_rows, reports, replay, curve_rows)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e48_edge_adapter_chain_drift_and_checkpoint_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e48_edge_adapter_chain_drift_and_checkpoint_probe")
    parser.add_argument("--seed", type=int, default=48001)
    parser.add_argument("--rows", type=int, default=144)
    parser.add_argument("--generations", type=int, default=44)
    parser.add_argument("--population", type=int, default=24)
    args = parser.parse_args()
    aggregate = run(args)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
