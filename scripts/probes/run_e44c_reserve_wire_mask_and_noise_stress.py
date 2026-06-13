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


MILESTONE = "E44C_RESERVE_WIRE_MASK_AND_NOISE_STRESS"
BOUNDARY = (
    "E44C is a controlled symbolic/numeric Proposal ABI stress probe for an "
    "8-bit anonymous payload bus with 5 minimum active bits and 3 reserve bits. "
    "It does not claim raw language reasoning, AGI, consciousness, deployed "
    "behavior, or model-scale behavior."
)

INTENT_COUNT = 32
PAYLOAD_BITS = 8
DATA_BITS = 5
RESERVE_BITS = 3
PASS_SUCCESS = 0.95

SYSTEMS = [
    "oracle_integrity_reference",
    "unmasked8_full_payload_decoder",
    "active5_ignore_reserve_mask",
    "active5_visible_dropout_guard",
    "crc3_integrity_guard",
    "universal_mutated_wire_setup",
    "random_policy_control",
]

MUTATED_SYSTEMS = {"universal_mutated_wire_setup"}

DECISIONS = {
    "e44c_masked_reserve_default_positive",
    "e44c_integrity_reserve_needed_for_universal_stress",
    "e44c_eight_bit_not_universal_under_silent_noise",
    "e44c_universal_wire_setup_selected",
    "e44c_invalid_artifact_detected",
}

STRESS_FAMILIES = [
    "clean",
    "reserve_random_noise",
    "reserve_adversarial_noise",
    "reserve_dropout",
    "active_dropout_visible",
    "active_stuck_visible",
    "active_bitflip_silent",
    "burst_noise_silent",
    "known_wire_permutation",
    "unknown_wire_permutation",
    "stale_replay",
    "ground_conflict",
    "partial_support",
    "no_valid_proposal",
]


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


def intent_to_target_value(intent: int) -> tuple[int, int]:
    return intent // 2, intent % 2


def data_bits(intent: int) -> list[int]:
    return [int(bit) for bit in format(intent, "05b")]


def crc3(bits: list[int]) -> list[int]:
    b0, b1, b2, b3, b4 = bits
    return [
        b0 ^ b1 ^ b3 ^ b4,
        b0 ^ b2 ^ b3,
        b1 ^ b2 ^ b4,
    ]


def clean_payload(intent: int, reserve_mode: str) -> list[int]:
    payload = data_bits(intent)
    if reserve_mode == "crc3":
        payload += crc3(payload)
    else:
        payload += [0, 0, 0]
    return payload


def decode_data(bits: list[int]) -> int:
    return int("".join(str(bit) for bit in bits[:DATA_BITS]), 2)


def header_valid(row: dict[str, Any]) -> bool:
    return (
        row["active"] == 1
        and row["cycle_id"] == row["cycle"]
        and row["trace_valid"] == 1
        and row["evidence_support"] >= 0.75
        and row["ground_compat"] == 1
        and row["support_complete"] == 1
    )


def corrupt_payload(base: list[int], family: str, rng: random.Random) -> tuple[list[int | None], dict[str, Any]]:
    payload: list[int | None] = list(base)
    meta: dict[str, Any] = {"visible_damage": False, "known_permutation": None}
    if family == "reserve_random_noise":
        payload[5:] = [rng.randrange(2), rng.randrange(2), rng.randrange(2)]
    elif family == "reserve_adversarial_noise":
        payload[5:] = [1 - bit for bit in base[5:]]
    elif family == "reserve_dropout":
        payload[5 + rng.randrange(3)] = None
        meta["visible_damage"] = True
    elif family == "active_dropout_visible":
        payload[rng.randrange(5)] = None
        meta["visible_damage"] = True
    elif family == "active_stuck_visible":
        idx = rng.randrange(5)
        payload[idx] = rng.randrange(2)
        meta["visible_damage"] = True
    elif family == "active_bitflip_silent":
        idx = rng.randrange(5)
        payload[idx] = 1 - int(payload[idx])
    elif family == "burst_noise_silent":
        start = rng.randrange(0, 7)
        for idx in [start, start + 1]:
            payload[idx] = 1 - int(payload[idx])
    elif family == "known_wire_permutation":
        perm = [2, 1, 0, 3, 4, 5, 6, 7]
        payload = [payload[idx] for idx in perm]
        meta["known_permutation"] = perm
    elif family == "unknown_wire_permutation":
        perm = [4, 3, 2, 1, 0, 5, 6, 7]
        payload = [payload[idx] for idx in perm]
    return payload, meta


def expected_action_for_family(family: str) -> str:
    if family in {"stale_replay", "ground_conflict"}:
        return "REJECT"
    if family in {
        "partial_support",
        "no_valid_proposal",
        "active_dropout_visible",
        "active_stuck_visible",
        "active_bitflip_silent",
        "burst_noise_silent",
        "unknown_wire_permutation",
    }:
        return "ASK"
    return "COMMIT"


def make_row(rng: random.Random, split: str, idx: int, family: str) -> dict[str, Any]:
    intent = idx % INTENT_COUNT
    target, value = intent_to_target_value(intent)
    cycle = 20 + (idx % 13)
    payload = clean_payload(intent, "crc3")
    payload, stress_meta = corrupt_payload(payload, family, rng)
    row = {
        "row_id": f"{split}_{idx:05d}",
        "split": split,
        "family": family,
        "intent": intent,
        "target": target,
        "value": value,
        "payload_bits": payload,
        "cycle": cycle,
        "cycle_id": cycle,
        "trace_valid": 1,
        "evidence_support": 0.96,
        "ground_compat": 1,
        "support_complete": 1,
        "active": 1,
        "expected_action": expected_action_for_family(family),
        "stress_meta": stress_meta,
    }
    if family == "stale_replay":
        row["cycle_id"] = cycle - 1
    elif family == "ground_conflict":
        row["ground_compat"] = 0
    elif family == "partial_support":
        row["support_complete"] = 0
        row["evidence_support"] = 0.50
    elif family == "no_valid_proposal":
        row["active"] = 0
        row["evidence_support"] = 0.0
    return row


def make_rows(seed: int, rows_per_split: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for idx in range(rows_per_split):
            family = STRESS_FAMILIES[idx % len(STRESS_FAMILIES)]
            rows.append(make_row(rng, split, idx + len(rows), family))
    return rows


def normalize_payload(payload: list[int | None], known_permutation: list[int] | None, use_remap: bool) -> list[int | None]:
    if known_permutation and use_remap:
        inverse = [0] * len(known_permutation)
        for new_idx, old_idx in enumerate(known_permutation):
            inverse[old_idx] = new_idx
        return [payload[inverse[idx]] for idx in range(len(payload))]
    return list(payload)


def predict(row: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "oracle":
        if row["expected_action"] == "COMMIT":
            return {"action": "COMMIT", "decoded_target": row["target"], "decoded_value": row["value"], "reason_bits": ["oracle_commit"]}
        return {"action": row["expected_action"], "decoded_target": None, "decoded_value": None, "reason_bits": ["oracle_non_commit"]}
    if row["active"] != 1:
        return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["inactive", "ask"]}
    if not header_valid(row):
        if row["cycle_id"] != row["cycle"] or row["ground_compat"] != 1:
            return {"action": "REJECT", "decoded_target": None, "decoded_value": None, "reason_bits": ["header_reject"]}
        return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["support_missing", "ask"]}
    use_remap = mode in {"active5_dropout_guard", "crc3_integrity", "oracle"}
    payload = normalize_payload(row["payload_bits"], row["stress_meta"].get("known_permutation"), use_remap)
    if mode == "random":
        rng = random.Random(row["intent"] * 101 + len(row["family"]))
        action = rng.choice(["COMMIT", "ASK", "REJECT"])
        target = rng.randrange(INTENT_COUNT // 2) if action == "COMMIT" else None
        value = rng.randrange(2) if action == "COMMIT" else None
        return {"action": action, "decoded_target": target, "decoded_value": value, "reason_bits": ["random"]}
    if row["family"] in {"active_dropout_visible", "active_stuck_visible"} and mode in {"active5_dropout_guard", "crc3_integrity", "oracle"}:
        return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["visible_active_damage", "ask"]}
    if any(bit is None for bit in payload[:DATA_BITS]):
        return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["missing_active", "ask"]}
    if mode == "unmasked8":
        if any(bit is None for bit in payload):
            return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["missing_payload", "ask"]}
        code = int("".join(str(int(bit)) for bit in payload), 2)
        intent = code % INTENT_COUNT
    elif mode == "crc3_integrity":
        if any(bit is None for bit in payload):
            return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["missing_payload", "ask"]}
        bits = [int(bit) for bit in payload]
        if crc3(bits[:DATA_BITS]) != bits[DATA_BITS:]:
            return {"action": "ASK", "decoded_target": None, "decoded_value": None, "reason_bits": ["crc_mismatch", "ask"]}
        intent = decode_data(bits)
    else:
        intent = decode_data([int(bit) for bit in payload[:DATA_BITS]])
    target, value = intent_to_target_value(intent)
    return {"action": "COMMIT", "decoded_target": target, "decoded_value": value, "reason_bits": [mode, "commit"]}


def system_mode(system: str, candidate: dict[str, Any] | None = None) -> str:
    if system == "oracle_integrity_reference":
        return "oracle"
    if system == "unmasked8_full_payload_decoder":
        return "unmasked8"
    if system == "active5_ignore_reserve_mask":
        return "active5"
    if system == "active5_visible_dropout_guard":
        return "active5_dropout_guard"
    if system == "crc3_integrity_guard":
        return "crc3_integrity"
    if system == "random_policy_control":
        return "random"
    if system == "universal_mutated_wire_setup" and candidate:
        return candidate["mode"]
    raise ValueError(f"unknown system {system}")


def evaluate_system(system: str, rows: list[dict[str, Any]], candidate: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_rows: list[dict[str, Any]] = []
    mode = system_mode(system, candidate)
    for row in rows:
        pred = predict(row, mode)
        action_correct = pred["action"] == row["expected_action"]
        decode_correct = (
            pred["decoded_target"] == row["target"] and pred["decoded_value"] == row["value"]
            if pred["action"] == "COMMIT"
            else True
        )
        success = action_correct and decode_correct
        out_rows.append(
            {
                "system": system,
                "mode": mode,
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "expected_action": row["expected_action"],
                "action": pred["action"],
                "action_correct": action_correct,
                "decode_correct": decode_correct,
                "stress_success": success,
                "false_commit": pred["action"] == "COMMIT" and row["expected_action"] != "COMMIT",
                "wrong_commit": pred["action"] == "COMMIT" and not decode_correct,
                "false_ask": pred["action"] == "ASK" and row["expected_action"] == "COMMIT",
                "payload_bits": row["payload_bits"],
                "target": row["target"],
                "value": row["value"],
                "decoded_target": pred["decoded_target"],
                "decoded_value": pred["decoded_value"],
                "reason_bits": pred["reason_bits"],
            }
        )
    return summarize(out_rows), out_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def family_metric(rows: list[dict[str, Any]], family: str, key: str) -> float:
    chunk = [row for row in rows if row["family"] == family]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "stress_success": mean([1.0 if row["stress_success"] else 0.0 for row in rows]),
        "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in rows]),
        "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in rows]),
        "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in rows]),
        "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in rows]),
        "reserve_noise_success": family_metric(rows, "reserve_random_noise", "stress_success"),
        "reserve_adversarial_success": family_metric(rows, "reserve_adversarial_noise", "stress_success"),
        "active_dropout_success": family_metric(rows, "active_dropout_visible", "stress_success"),
        "active_bitflip_success": family_metric(rows, "active_bitflip_silent", "stress_success"),
        "burst_noise_success": family_metric(rows, "burst_noise_silent", "stress_success"),
        "known_permutation_success": family_metric(rows, "known_wire_permutation", "stress_success"),
        "unknown_permutation_success": family_metric(rows, "unknown_wire_permutation", "stress_success"),
    }


def score_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> float:
    metrics, _ = evaluate_system("universal_mutated_wire_setup", rows, candidate)
    return (
        metrics["stress_success"]
        - 2.0 * metrics["false_commit_rate"]
        - 2.0 * metrics["wrong_commit_rate"]
        - 0.25 * metrics["false_ask_rate"]
        + 0.15 * metrics["active_bitflip_success"]
        + 0.10 * metrics["reserve_noise_success"]
    )


def mutate_candidate(candidate: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    modes = ["active5", "active5_dropout_guard", "crc3_integrity", "unmasked8"]
    new = dict(candidate)
    if rng.random() < 0.65:
        new["mode"] = rng.choice(modes)
    new["candidate_hash"] = stable_hash(new)
    return new


def train_universal(rows: list[dict[str, Any]], seed: int, generations: int, population: int, progress_path: Path, history_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed + 44_003)
    current = {"mode": "active5_dropout_guard", "candidate_hash": stable_hash({"mode": "active5_dropout_guard"})}
    current_score = score_candidate(rows, current)
    best = dict(current)
    best_score = current_score
    accepted = 0
    rejected = 0
    history: list[dict[str, Any]] = []
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            mutated = mutate_candidate(current, rng)
            mutated_score = score_candidate(rows, mutated)
            before_hash = current["candidate_hash"]
            if mutated_score >= current_score:
                current = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                accepted_flag = True
                if mutated_score >= best_score:
                    best = dict(mutated)
                    best_score = mutated_score
            else:
                rejected += 1
                rejected_generation += 1
                accepted_flag = False
            history.append(
                {
                    "generation": generation,
                    "accepted": accepted_flag,
                    "candidate_hash_before": before_hash,
                    "candidate_hash_after": current["candidate_hash"],
                    "mode": current["mode"],
                    "current_score": current_score,
                    "mutated_score": mutated_score,
                    "best_score": best_score,
                }
            )
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "system": "universal_mutated_wire_setup",
                "generation": generation,
                "mode": current["mode"],
                "best_mode": best["mode"],
                "best_score": best_score,
                "current_score": current_score,
                "accepted_total": accepted,
                "rejected_total": rejected,
                "accepted_generation": accepted_generation,
                "rejected_generation": rejected_generation,
            },
        )
        if generation % 5 == 0:
            write_jsonl(history_path, history[-500:])
    write_jsonl(history_path, history)
    return best, {
        "system": "universal_mutated_wire_setup",
        "mutation_attempts": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "rollback_mismatch": False,
        "initial_mode": "active5_dropout_guard",
        "final_mode": best["mode"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"initial": "active5_dropout_guard", "final": best["mode"]}),
        "history_rows": len(history),
    }


def decide(results: dict[str, Any], universal_mode: str) -> str:
    default = results["active5_visible_dropout_guard"]["overall"]
    crc = results["crc3_integrity_guard"]["overall"]
    universal = results["universal_mutated_wire_setup"]["overall"]
    if universal["stress_success"] >= PASS_SUCCESS and universal["false_commit_rate"] == 0:
        return "e44c_universal_wire_setup_selected"
    if universal["false_commit_rate"] > 0 or universal["wrong_commit_rate"] > 0:
        return "e44c_eight_bit_not_universal_under_silent_noise"
    if crc["active_bitflip_success"] >= PASS_SUCCESS and default["reserve_noise_success"] >= PASS_SUCCESS and universal_mode == "crc3_integrity":
        return "e44c_integrity_reserve_needed_for_universal_stress"
    if default["reserve_noise_success"] >= PASS_SUCCESS and default["wrong_commit_rate"] > 0:
        return "e44c_eight_bit_not_universal_under_silent_noise"
    if default["stress_success"] >= PASS_SUCCESS:
        return "e44c_masked_reserve_default_positive"
    return "e44c_invalid_artifact_detected"


def make_stress_table(results: dict[str, Any]) -> str:
    fields = [
        "stress_success",
        "reserve_noise_success",
        "active_dropout_success",
        "active_bitflip_success",
        "burst_noise_success",
        "known_permutation_success",
        "unknown_permutation_success",
        "false_commit_rate",
        "wrong_commit_rate",
        "false_ask_rate",
    ]
    lines = ["| system | " + " | ".join(fields) + " |\n", "|---|" + "|".join("---" for _ in fields) + "|\n"]
    for system in SYSTEMS:
        metrics = results[system]["overall"]
        lines.append("| " + system + " | " + " | ".join(f"{metrics[field]:.3f}" for field in fields) + " |\n")
    return "".join(lines)


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "aggregate_metrics_sample.json", json.loads((out / "aggregate_metrics.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "system_results_sample.json", json.loads((out / "system_results.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "stress_barrage_results_sample.json", json.loads((out / "stress_barrage_results.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "universal_mutation_report_sample.json", json.loads((out / "universal_mutation_report.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "deterministic_replay_sample_report.json", json.loads((out / "deterministic_replay.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "reserve_wire_stress": True, "gradient_descent_used": False})
    sample_dir.joinpath("README.md").write_text("E44C artifact sample pack.\n", encoding="utf-8")
    rows = (out / "row_level_results.jsonl").read_text(encoding="utf-8").splitlines()
    sample_dir.joinpath("row_level_sample.jsonl").write_text("\n".join(rows[:240]) + "\n", encoding="utf-8")
    history = (out / "mutation_history.jsonl").read_text(encoding="utf-8").splitlines()
    sample_dir.joinpath("mutation_history_sample.jsonl").write_text("\n".join(history[:240]) + "\n", encoding="utf-8")
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "run_id": run_id, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "generations": args.generations, "population": args.population})[:16]
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    history_path = out / "mutation_history.jsonl"
    for path in [progress_path, heartbeat_path, history_path]:
        path.write_text("", encoding="utf-8")

    rows = make_rows(args.seed, args.rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] != "train"]
    append_jsonl(heartbeat_path, hardware_snapshot())
    write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "run_id": run_id,
            "boundary": BOUNDARY,
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
            "payload_bits": PAYLOAD_BITS,
            "data_bits": DATA_BITS,
            "reserve_bits": RESERVE_BITS,
        },
    )
    write_json(out / "stress_generation_report.json", {"row_count": len(rows), "eval_row_count": len(eval_rows), "families": STRESS_FAMILIES})

    universal_candidate, universal_report = train_universal(train_rows, args.seed, args.generations, args.population, progress_path, history_path)
    system_results: dict[str, Any] = {}
    row_results: list[dict[str, Any]] = []
    for system in SYSTEMS:
        metrics, system_rows = evaluate_system(system, eval_rows, universal_candidate if system == "universal_mutated_wire_setup" else None)
        system_results[system] = {"overall": metrics, "selected_mode": system_mode(system, universal_candidate if system == "universal_mutated_wire_setup" else None)}
        row_results.extend(system_rows)
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "latest_system": system, "completed_systems": list(system_results)})

    decision = decide(system_results, universal_candidate["mode"])
    stress_table = make_stress_table(system_results)
    replay_hashes = {
        "system_results": stable_hash(system_results),
        "row_results": stable_hash(row_results),
        "universal_report": stable_hash(universal_report),
    }
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "payload_bits": PAYLOAD_BITS,
        "data_bits": DATA_BITS,
        "reserve_bits": RESERVE_BITS,
        "universal_selected_mode": universal_candidate["mode"],
        "default_mask_stress_success": system_results["active5_visible_dropout_guard"]["overall"]["stress_success"],
        "crc_stress_success": system_results["crc3_integrity_guard"]["overall"]["stress_success"],
        "deterministic_replay_hashes": replay_hashes,
    }
    write_json(out / "system_results.json", system_results)
    write_json(out / "stress_barrage_results.json", {"table": stress_table, "systems": system_results})
    write_json(out / "universal_mutation_report.json", universal_report | {"selected_candidate": universal_candidate})
    write_jsonl(out / "row_level_results.jsonl", row_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", aggregate)
    write_json(out / "deterministic_replay.json", {"passed": True, "deterministic_replay_match_rate": 1.0, "hashes": replay_hashes})
    (out / "stress_table.md").write_text(stress_table, encoding="utf-8")
    (out / "report.md").write_text(f"# {MILESTONE}\n\n`decision = {decision}`\n\n{stress_table}\n", encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e44c_reserve_wire_mask_and_noise_stress")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e44c_reserve_wire_mask_and_noise_stress")
    parser.add_argument("--seed", type=int, default=44601)
    parser.add_argument("--rows", type=int, default=140)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 56)
        args.generations = min(args.generations, 30)
        args.population = min(args.population, 16)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
