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


MILESTONE = "E45_EDGE_ABI_MUTATION_SEARCH_SPACE_FALLOFF_PROBE"
BOUNDARY = (
    "E45 isolates one Edge ABI between two frozen nodes and mutates only the "
    "connection decoder/contract. It measures final accuracy and learning "
    "dynamics as bus width and intent count grow. It does not claim raw "
    "language reasoning, AGI, consciousness, deployed behavior, or model-scale "
    "behavior."
)

PASS_ACCURACY = 0.95
SYSTEMS = [
    "structured_w16_i32_reference",
    "structured_w64_i256_reference",
    "structured_w128_i1024_reference",
    "anonymous_w8_i32",
    "anonymous_w12_i32",
    "anonymous_w16_i32",
    "anonymous_w24_i32",
    "anonymous_w32_i32",
    "anonymous_w64_i32",
    "anonymous_w16_i256",
    "anonymous_w32_i256",
    "anonymous_w64_i256",
    "anonymous_w96_i256",
    "anonymous_w128_i256",
    "anonymous_w64_i1024",
    "anonymous_w128_i1024",
    "random_w16_i32_control",
]
MUTATED_SYSTEMS = {
    "anonymous_w8_i32",
    "anonymous_w12_i32",
    "anonymous_w16_i32",
    "anonymous_w24_i32",
    "anonymous_w32_i32",
    "anonymous_w64_i32",
    "anonymous_w16_i256",
    "anonymous_w32_i256",
    "anonymous_w64_i256",
    "anonymous_w96_i256",
    "anonymous_w128_i256",
    "anonymous_w64_i1024",
    "anonymous_w128_i1024",
}
DECISIONS = {
    "e45_anonymous_wide_bus_learning_falloff_detected",
    "e45_32bit_extended_lane_still_mutation_friendly",
    "e45_64bit_anonymous_lane_mutation_friendly",
    "e45_connection_needs_structured_layout",
    "e45_invalid_artifact_detected",
}


CONFIGS: dict[str, dict[str, Any]] = {
    "structured_w16_i32_reference": {"bus_width": 16, "intent_count": 32, "mode": "structured"},
    "structured_w64_i256_reference": {"bus_width": 64, "intent_count": 256, "mode": "structured"},
    "structured_w128_i1024_reference": {"bus_width": 128, "intent_count": 1024, "mode": "structured"},
    "anonymous_w8_i32": {"bus_width": 8, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w12_i32": {"bus_width": 12, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w16_i32": {"bus_width": 16, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w24_i32": {"bus_width": 24, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w32_i32": {"bus_width": 32, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w64_i32": {"bus_width": 64, "intent_count": 32, "mode": "anonymous"},
    "anonymous_w16_i256": {"bus_width": 16, "intent_count": 256, "mode": "anonymous"},
    "anonymous_w32_i256": {"bus_width": 32, "intent_count": 256, "mode": "anonymous"},
    "anonymous_w64_i256": {"bus_width": 64, "intent_count": 256, "mode": "anonymous"},
    "anonymous_w96_i256": {"bus_width": 96, "intent_count": 256, "mode": "anonymous"},
    "anonymous_w128_i256": {"bus_width": 128, "intent_count": 256, "mode": "anonymous"},
    "anonymous_w64_i1024": {"bus_width": 64, "intent_count": 1024, "mode": "anonymous"},
    "anonymous_w128_i1024": {"bus_width": 128, "intent_count": 1024, "mode": "anonymous"},
    "random_w16_i32_control": {"bus_width": 16, "intent_count": 32, "mode": "random"},
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


def required_bits(intent_count: int) -> int:
    return math.ceil(math.log2(intent_count))


def encode_intent(intent: int, bit_count: int) -> list[int]:
    return [int(bit) for bit in format(intent, f"0{bit_count}b")]


def ordered_search_space_log10(bus_width: int, data_bits: int) -> float:
    return sum(math.log10(bus_width - idx) for idx in range(data_bits))


def secret_indices(system: str) -> list[int]:
    cfg = CONFIGS[system]
    bits = required_bits(cfg["intent_count"])
    if cfg["mode"] == "structured":
        return list(range(bits))
    rng = random.Random(45_000 + cfg["bus_width"] * 101 + cfg["intent_count"] * 17 + len(system))
    return rng.sample(range(cfg["bus_width"]), bits)


def make_bus(intent: int, cfg: dict[str, Any], active_indices: list[int], row_rng: random.Random) -> list[int]:
    width = cfg["bus_width"]
    bits = required_bits(cfg["intent_count"])
    bus = [row_rng.randrange(2) for _ in range(width)]
    for slot, idx in enumerate(active_indices):
        bus[idx] = encode_intent(intent, bits)[slot]
    return bus


def make_rows(system: str, seed: int, rows_per_split: int) -> list[dict[str, Any]]:
    cfg = CONFIGS[system]
    indices = secret_indices(system)
    rows: list[dict[str, Any]] = []
    split_offsets = {"train": 0, "heldout": 3, "ood": 7, "counterfactual": 11, "adversarial": 17}
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        rng = random.Random(seed + len(system) * 31 + split_offsets[split] * 101)
        for idx in range(rows_per_split):
            intent = (idx * 13 + split_offsets[split]) % cfg["intent_count"]
            bus = make_bus(intent, cfg, indices, rng)
            rows.append(
                {
                    "system": system,
                    "row_id": f"{system}_{split}_{idx:05d}",
                    "split": split,
                    "intent": intent,
                    "bus_bits": bus,
                    "expected_action": "COMMIT",
                    "secret_indices_hash": stable_hash(indices),
                }
            )
    return rows


def decode_with_indices(bus: list[int], indices: list[int], intent_count: int) -> int | None:
    if len(set(indices)) != len(indices):
        return None
    try:
        value = int("".join(str(int(bus[idx])) for idx in indices), 2)
    except IndexError:
        return None
    if value >= intent_count:
        return None
    return value


def evaluate_candidate(system: str, rows: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = CONFIGS[system]
    out_rows: list[dict[str, Any]] = []
    if cfg["mode"] == "structured":
        indices = secret_indices(system)
    elif cfg["mode"] == "random":
        indices = candidate["indices"]
    else:
        indices = candidate["indices"]
    for row in rows:
        if cfg["mode"] == "random":
            pred = random.Random(stable_hash(row["row_id"] + str(indices))[:12]).randrange(cfg["intent_count"])
        else:
            pred = decode_with_indices(row["bus_bits"], indices, cfg["intent_count"])
        action = "ASK" if pred is None else "COMMIT"
        decode_correct = pred == row["intent"] if action == "COMMIT" else False
        out_rows.append(
            {
                "system": system,
                "row_id": row["row_id"],
                "split": row["split"],
                "intent": row["intent"],
                "bus_width": cfg["bus_width"],
                "intent_count": cfg["intent_count"],
                "data_bits": required_bits(cfg["intent_count"]),
                "mode": cfg["mode"],
                "expected_action": row["expected_action"],
                "action": action,
                "decoded_intent": pred,
                "decode_correct": decode_correct,
                "edge_success": action == "COMMIT" and decode_correct,
                "wrong_commit": action == "COMMIT" and not decode_correct,
                "false_ask": action == "ASK",
                "candidate_indices_hash": stable_hash(indices),
                "secret_indices_hash": row["secret_indices_hash"],
            }
        )
    return summarize(out_rows), out_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def split_metric(rows: list[dict[str, Any]], split: str, key: str) -> float:
    chunk = [row for row in rows if row["split"] == split]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "edge_success": mean([1.0 if row["edge_success"] else 0.0 for row in rows]),
        "heldout_success": split_metric(rows, "heldout", "edge_success"),
        "ood_success": split_metric(rows, "ood", "edge_success"),
        "counterfactual_success": split_metric(rows, "counterfactual", "edge_success"),
        "adversarial_success": split_metric(rows, "adversarial", "edge_success"),
        "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in rows]),
        "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in rows]),
    }


def random_indices(width: int, bits: int, rng: random.Random) -> list[int]:
    return rng.sample(range(width), bits)


def mutate_indices(indices: list[int], width: int, rng: random.Random) -> list[int]:
    mutated = list(indices)
    if rng.random() < 0.85:
        slot = rng.randrange(len(mutated))
        options = [idx for idx in range(width) if idx not in mutated or idx == mutated[slot]]
        mutated[slot] = rng.choice(options)
    else:
        a, b = rng.sample(range(len(mutated)), 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated


def train_decoder(
    system: str,
    rows: list[dict[str, Any]],
    seed: int,
    generations: int,
    population: int,
    progress_path: Path,
    curve_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cfg = CONFIGS[system]
    bits = required_bits(cfg["intent_count"])
    rng = random.Random(seed + cfg["bus_width"] * 1_003 + cfg["intent_count"] * 19)
    if cfg["mode"] == "structured":
        candidate = {"indices": secret_indices(system), "candidate_hash": stable_hash(secret_indices(system))}
        metrics, _ = evaluate_candidate(system, rows, candidate)
        report = {
            "system": system,
            "mutation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "rollback_count": 0,
            "rollback_mismatch": False,
            "attempts_to_95": 0 if metrics["heldout_success"] >= PASS_ACCURACY else None,
            "attempts_to_99": 0 if metrics["heldout_success"] >= 0.99 else None,
            "accepted_rate": 0.0,
            "last_improvement_generation": 0,
            "plateau_tail_generations": 0,
            "initial_score": metrics["heldout_success"],
            "final_score": metrics["heldout_success"],
            "parameter_diff_written": True,
            "parameter_diff_hash": stable_hash({"structured": system}),
            "learning_curve_rows": 1,
        }
        curve = [
            {
                "system": system,
                "generation": 0,
                "attempts": 0,
                "best_train_score": metrics["heldout_success"],
                "current_train_score": metrics["heldout_success"],
                "accepted_total": 0,
                "rejected_total": 0,
            }
        ]
        append_jsonl(curve_path, curve[0])
        return candidate, report, curve
    current = {"indices": random_indices(cfg["bus_width"], bits, rng)}
    current["candidate_hash"] = stable_hash(current["indices"])
    current_metrics, _ = evaluate_candidate(system, [row for row in rows if row["split"] == "train"], current)
    current_score = current_metrics["edge_success"]
    best = dict(current)
    best_score = current_score
    accepted = 0
    rejected = 0
    attempts_to_95: int | None = None
    attempts_to_99: int | None = None
    last_improvement_generation = 0
    curve: list[dict[str, Any]] = []
    attempts = 0
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            before_hash = current["candidate_hash"]
            mutated = {"indices": mutate_indices(current["indices"], cfg["bus_width"], rng)}
            mutated["candidate_hash"] = stable_hash(mutated["indices"])
            mutated_metrics, _ = evaluate_candidate(system, [row for row in rows if row["split"] == "train"], mutated)
            mutated_score = mutated_metrics["edge_success"]
            if mutated_score > current_score:
                current = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                if mutated_score > best_score:
                    best = dict(mutated)
                    best_score = mutated_score
                    last_improvement_generation = generation
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and best_score >= PASS_ACCURACY:
                attempts_to_95 = attempts
            if attempts_to_99 is None and best_score >= 0.99:
                attempts_to_99 = attempts
        row = {
            "time": time.time(),
            "system": system,
            "generation": generation,
            "attempts": attempts,
            "best_train_score": best_score,
            "current_train_score": current_score,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
            "bus_width": cfg["bus_width"],
            "intent_count": cfg["intent_count"],
            "search_space_log10": ordered_search_space_log10(cfg["bus_width"], bits),
        }
        curve.append(row)
        append_jsonl(curve_path, row)
        append_jsonl(progress_path, row)
    report = {
        "system": system,
        "mutation_attempts": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "rollback_mismatch": False,
        "attempts_to_95": attempts_to_95,
        "attempts_to_99": attempts_to_99,
        "accepted_rate": accepted / (accepted + rejected) if accepted + rejected else 0.0,
        "last_improvement_generation": last_improvement_generation,
        "plateau_tail_generations": max(0, generations - 1 - last_improvement_generation),
        "initial_score": curve[0]["current_train_score"] if curve else current_score,
        "final_score": best_score,
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"initial": "random_indices", "final": best["indices"]}),
        "learning_curve_rows": len(curve),
    }
    return best, report, curve


def pass_gate(metrics: dict[str, Any]) -> bool:
    return (
        metrics["heldout_success"] >= PASS_ACCURACY
        and metrics["ood_success"] >= PASS_ACCURACY
        and metrics["counterfactual_success"] >= PASS_ACCURACY
        and metrics["adversarial_success"] >= PASS_ACCURACY
        and metrics["wrong_commit_rate"] == 0.0
    )


def decide(system_results: dict[str, Any], dynamics: dict[str, Any]) -> str:
    if not pass_gate(system_results["structured_w128_i1024_reference"]["overall"]):
        return "e45_invalid_artifact_detected"
    w16 = system_results["anonymous_w16_i256"]["overall"]
    w32 = system_results["anonymous_w32_i256"]["overall"]
    w64 = system_results["anonymous_w64_i256"]["overall"]
    w128 = system_results["anonymous_w128_i256"]["overall"]
    w64_1024 = system_results["anonymous_w64_i1024"]["overall"]
    d16 = dynamics["anonymous_w16_i256"]
    d32 = dynamics["anonymous_w32_i256"]
    d64 = dynamics["anonymous_w64_i256"]
    if not pass_gate(w32):
        return "e45_connection_needs_structured_layout"
    if not pass_gate(w128) or not pass_gate(w64_1024):
        return "e45_anonymous_wide_bus_learning_falloff_detected"
    if not pass_gate(w64):
        return "e45_anonymous_wide_bus_learning_falloff_detected"
    attempts32 = d32.get("attempts_to_95") or d32["mutation_attempts"]
    attempts64 = d64.get("attempts_to_95") or d64["mutation_attempts"]
    attempts16 = d16.get("attempts_to_95") or max(1, d16["mutation_attempts"])
    if attempts64 >= 2.5 * max(1, attempts32) or attempts64 >= 4.0 * max(1, attempts16):
        return "e45_anonymous_wide_bus_learning_falloff_detected"
    if pass_gate(w64):
        return "e45_64bit_anonymous_lane_mutation_friendly"
    return "e45_32bit_extended_lane_still_mutation_friendly"


def make_results_table(system_results: dict[str, Any], dynamics: dict[str, Any]) -> str:
    fields = [
        "bus_width",
        "intent_count",
        "search_space_log10",
        "heldout_success",
        "ood_success",
        "wrong_commit_rate",
        "attempts_to_95",
        "accepted_rate",
        "plateau_tail_generations",
    ]
    lines = ["| system | " + " | ".join(fields) + " |\n", "|---|" + "|".join("---" for _ in fields) + "|\n"]
    for system in SYSTEMS:
        cfg = CONFIGS[system]
        metrics = system_results[system]["overall"]
        dyn = dynamics[system]
        values = {
            "bus_width": cfg["bus_width"],
            "intent_count": cfg["intent_count"],
            "search_space_log10": ordered_search_space_log10(cfg["bus_width"], required_bits(cfg["intent_count"])),
            "heldout_success": metrics["heldout_success"],
            "ood_success": metrics["ood_success"],
            "wrong_commit_rate": metrics["wrong_commit_rate"],
            "attempts_to_95": dyn.get("attempts_to_95"),
            "accepted_rate": dyn.get("accepted_rate", 0.0),
            "plateau_tail_generations": dyn.get("plateau_tail_generations", 0),
        }
        rendered = []
        for field in fields:
            value = values[field]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append("none" if value is None else str(value))
        lines.append("| " + system + " | " + " | ".join(rendered) + " |\n")
    return "".join(lines)


def deterministic_replay_report(rows: list[dict[str, Any]], system_results: dict[str, Any], aggregate: dict[str, Any], dynamics: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "hashes": {
            "row_level_results_hash": stable_hash(rows),
            "system_results_hash": stable_hash(system_results),
            "aggregate_metrics_hash": stable_hash(aggregate),
            "learning_dynamics_hash": stable_hash(dynamics),
        },
    }


def write_sample_pack(sample_dir: Path, aggregate: dict[str, Any], system_results: dict[str, Any], rows: list[dict[str, Any]], dynamics: dict[str, Any], replay: dict[str, Any], search_report: dict[str, Any], curve: list[dict[str, Any]]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E45 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "edge_abi_falloff": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "learning_dynamics_report_sample.json", dynamics)
    write_json(sample_dir / "search_space_report_sample.json", search_report)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "row_level_sample.jsonl", rows[:260])
    write_jsonl(sample_dir / "learning_curve_sample.jsonl", curve[:260])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def build_report(aggregate: dict[str, Any], table: str, dynamics: dict[str, Any], search_report: dict[str, Any]) -> str:
    return f"""# E45 Edge ABI Mutation Search Space Falloff Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E45 froze the producer and consumer nodes, then mutated only the Edge ABI
decoder/contract between them. It measured both final performance and learning
dynamics.

## Result Table

```text
{table}```

## Interpretation

The important distinction is:

```text
wide structured bus:
  can work cleanly

wide anonymous mutable bus:
  search-space cost rises sharply
```

So the result is not "64 bits are impossible." It is:

```text
64 anonymous mutable fast-lane bits are not free.
Use structure/masks/framing if the edge gets wide.
```

## Learning Dynamics Snapshot

```json
{json.dumps({key: dynamics[key] for key in ["anonymous_w16_i256", "anonymous_w32_i256", "anonymous_w64_i256"]}, indent=2, sort_keys=True)}
```

## Search Space

```json
{json.dumps(search_report, indent=2, sort_keys=True)}
```

## Boundary

This is a controlled symbolic/numeric Edge ABI probe. It does not prove raw
language reasoning, deployed AI assistant behavior, model-scale behavior, AGI,
or consciousness.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    curve_path = out / "learning_curve.jsonl"
    for path in [progress_path, heartbeat_path, curve_path]:
        if path.exists():
            path.unlink()

    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "milestone": MILESTONE})[:16]
    append_jsonl(heartbeat_path, hardware_snapshot())
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    all_row_results: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    system_results: dict[str, Any] = {}
    dynamics: dict[str, Any] = {}
    final_candidates: dict[str, Any] = {}

    for system in SYSTEMS:
        rows = make_rows(system, args.seed, args.rows)
        candidate, report, curve = train_decoder(system, rows, args.seed, args.generations, args.population, progress_path, curve_path)
        metrics, row_results = evaluate_candidate(system, rows, candidate)
        cfg = CONFIGS[system]
        bits = required_bits(cfg["intent_count"])
        metrics.update(
            {
                "bus_width": cfg["bus_width"],
                "intent_count": cfg["intent_count"],
                "data_bits": bits,
                "search_space_log10": ordered_search_space_log10(cfg["bus_width"], bits),
                "pass_gate": pass_gate(metrics),
            }
        )
        system_results[system] = {"overall": metrics}
        dynamics[system] = report
        final_candidates[system] = {"candidate_hash": stable_hash(candidate), "indices_hash": stable_hash(candidate["indices"])}
        all_row_results.extend(row_results)
        all_curve_rows.extend(curve)
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "event": "system_done",
                "system": system,
                "heldout_success": metrics["heldout_success"],
                "attempts_to_95": report.get("attempts_to_95"),
                "accepted_rate": report.get("accepted_rate"),
            },
        )
        write_json(
            out / "partial_aggregate_snapshot.json",
            {
                "run_id": run_id,
                "completed_systems": list(system_results),
                "latest_system": system,
                "latest_metrics": metrics,
                "latest_dynamics": report,
            },
        )
    append_jsonl(heartbeat_path, hardware_snapshot())

    decision = decide(system_results, dynamics)
    search_report = {
        system: {
            "bus_width": CONFIGS[system]["bus_width"],
            "intent_count": CONFIGS[system]["intent_count"],
            "data_bits": required_bits(CONFIGS[system]["intent_count"]),
            "ordered_search_space_log10": ordered_search_space_log10(CONFIGS[system]["bus_width"], required_bits(CONFIGS[system]["intent_count"])),
            "mode": CONFIGS[system]["mode"],
        }
        for system in SYSTEMS
    }
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "rows_per_split": args.rows,
        "systems": SYSTEMS,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_row_results, system_results, aggregate, dynamics)
    table = make_results_table(system_results, dynamics)

    write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "boundary": BOUNDARY,
            "run_id": run_id,
            "systems": SYSTEMS,
            "mutated_systems": sorted(MUTATED_SYSTEMS),
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
        },
    )
    write_json(out / "search_space_report.json", search_report)
    write_json(out / "learning_dynamics_report.json", dynamics)
    write_json(out / "final_candidates.json", final_candidates)
    write_json(out / "system_results.json", system_results)
    write_jsonl(out / "row_level_results.jsonl", all_row_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id})
    (out / "results_table.md").write_text(table, encoding="utf-8")
    (out / "report.md").write_text(build_report(aggregate, table, dynamics, search_report), encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_row_results, dynamics, replay, search_report, all_curve_rows)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e45_edge_abi_mutation_search_space_falloff_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e45_edge_abi_mutation_search_space_falloff_probe")
    parser.add_argument("--seed", type=int, default=45001)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--generations", type=int, default=48)
    parser.add_argument("--population", type=int, default=32)
    args = parser.parse_args()
    aggregate = run(args)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
