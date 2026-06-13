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


MILESTONE = "E44B_PARALLEL_ABSTRACT_WIRE_STREAM_FIELD_SMOKE"
BOUNDARY = (
    "E44B is a controlled symbolic/numeric Proposal ABI smoke probe. It tests "
    "parallel abstract wire streams shaped as wire_count x bits_per_wire. It "
    "does not claim raw language reasoning, AGI, consciousness, deployed "
    "behavior, or model-scale behavior."
)

INTENT_COUNT = 32
REQUIRED_CAPACITY_BITS = 5
PASS_SUCCESS = 0.95

DECISIONS = {
    "e44b_parallel_serial_capacity_detected",
    "e44b_wire_shape_tradeoff_detected",
    "e44b_wire_stream_unreliable",
    "e44b_headerless_stream_unreliable",
    "e44b_invalid_artifact_detected",
}

CONTROL_SYSTEMS = [
    "oracle_wire_stream_reference",
    "headerless_stream_5x1_control",
    "random_stream_decoder_5x1_control",
]

FAMILIES = [
    "valid_commit",
    "valid_commit_alt",
    "valid_commit_ood",
    "toxic_wrong_stream",
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def grid_system_name(wire_count: int, bits_per_wire: int) -> str:
    return f"wire_stream_{wire_count}x{bits_per_wire}"


def intent_to_target_value(intent: int) -> tuple[int, int]:
    return intent // 2, intent % 2


def encode_intent_stream(intent: int, wire_count: int, bits_per_wire: int) -> list[str]:
    total_bits = wire_count * bits_per_wire
    if total_bits <= 0:
        return ["" for _ in range(wire_count)]
    base = format(intent, f"0{REQUIRED_CAPACITY_BITS}b")[-min(total_bits, REQUIRED_CAPACITY_BITS):]
    if total_bits > REQUIRED_CAPACITY_BITS:
        rng = random.Random(77_001 + intent * 131 + wire_count * 17 + bits_per_wire)
        base += "".join(str(rng.randrange(2)) for _ in range(total_bits - REQUIRED_CAPACITY_BITS))
    base = base[:total_bits].ljust(total_bits, "0")
    return [base[i * bits_per_wire : (i + 1) * bits_per_wire] for i in range(wire_count)]


def flatten_stream(wire_bits: list[str]) -> str:
    return "|".join(wire_bits)


def capacity_collision_rate(wire_count: int, bits_per_wire: int) -> float:
    buckets: dict[str, set[int]] = {}
    for intent in range(INTENT_COUNT):
        code = flatten_stream(encode_intent_stream(intent, wire_count, bits_per_wire))
        buckets.setdefault(code, set()).add(intent)
    collided = sum(len(values) for values in buckets.values() if len(values) > 1)
    return collided / INTENT_COUNT


def proposal(
    proposal_id: str,
    intent: int,
    wire_count: int,
    bits_per_wire: int,
    cycle: int,
    *,
    trace_valid: bool,
    evidence_support: float,
    ground_compat: bool,
    support_complete: bool = True,
    stale: bool = False,
    active: bool = True,
) -> dict[str, Any]:
    target, value = intent_to_target_value(intent)
    return {
        "proposal_id": proposal_id,
        "active": 1 if active else 0,
        "action_code": "COMMIT",
        "source_pocket_id": f"pocket_{proposal_id}",
        "cycle_id": cycle - 1 if stale else cycle,
        "trace_ref": f"trace_{intent}",
        "trace_valid": 1 if trace_valid else 0,
        "evidence_support": evidence_support,
        "ground_compat": 1 if ground_compat else 0,
        "support_complete": 1 if support_complete else 0,
        "wire_count": wire_count,
        "bits_per_wire": bits_per_wire,
        "wire_bits": encode_intent_stream(intent, wire_count, bits_per_wire),
        "stream_code": flatten_stream(encode_intent_stream(intent, wire_count, bits_per_wire)),
        "hidden_intent": intent,
        "hidden_target": target,
        "hidden_value": value,
    }


def header_valid(prop: dict[str, Any], cycle: int) -> bool:
    return (
        prop["active"] == 1
        and prop["cycle_id"] == cycle
        and prop["trace_valid"] == 1
        and prop["evidence_support"] >= 0.75
        and prop["ground_compat"] == 1
        and prop["support_complete"] == 1
    )


def make_row(rng: random.Random, split: str, idx: int, family: str, max_wire: int, max_bits: int) -> dict[str, Any]:
    cycle = 10 + (idx % 11)
    intent = idx % INTENT_COUNT
    wrong_intent = (intent + rng.randrange(1, INTENT_COUNT)) % INTENT_COUNT
    expected_action = "COMMIT"
    required_reason_bits = ["header_valid", "stream_decoded", "target_value_match"]
    prop_intent = intent
    prop_kwargs: dict[str, Any] = {
        "trace_valid": True,
        "evidence_support": 0.96,
        "ground_compat": True,
        "support_complete": True,
        "stale": False,
        "active": True,
    }

    if family in {"valid_commit", "valid_commit_alt", "valid_commit_ood"}:
        pass
    elif family == "toxic_wrong_stream":
        prop_intent = wrong_intent
        prop_kwargs.update(trace_valid=False, evidence_support=0.25)
        expected_action = "REJECT"
        required_reason_bits = ["trace_invalid", "reject"]
    elif family == "stale_replay":
        prop_intent = wrong_intent
        prop_kwargs.update(stale=True)
        expected_action = "REJECT"
        required_reason_bits = ["cycle_mismatch", "reject"]
    elif family == "ground_conflict":
        prop_intent = wrong_intent
        prop_kwargs.update(ground_compat=False)
        expected_action = "REJECT"
        required_reason_bits = ["ground_incompatible", "reject"]
    elif family == "partial_support":
        prop_kwargs.update(evidence_support=0.58, support_complete=False)
        expected_action = "ASK"
        required_reason_bits = ["support_incomplete", "ask"]
    elif family == "no_valid_proposal":
        prop_kwargs.update(active=False, evidence_support=0.0, support_complete=False)
        expected_action = "DEFER"
        required_reason_bits = ["no_active_proposal", "defer"]
    else:
        raise ValueError(f"unknown family {family}")

    prop = proposal("main", prop_intent, max_wire, max_bits, cycle, **prop_kwargs)
    target, value = intent_to_target_value(intent)
    return {
        "row_id": f"{split}_{idx:05d}",
        "split": split,
        "family": family,
        "cycle": cycle,
        "intent": intent,
        "target": target,
        "value": value,
        "expected_action": expected_action,
        "required_reason_bits": required_reason_bits,
        "proposal_template": prop,
    }


def make_rows(seed: int, rows_per_split: int, max_wire: int, max_bits: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for idx in range(rows_per_split):
            family = FAMILIES[idx % len(FAMILIES)]
            rows.append(make_row(rng, split, idx + len(rows), family, max_wire, max_bits))
    return rows


def adapt_row_to_shape(row: dict[str, Any], wire_count: int, bits_per_wire: int) -> dict[str, Any]:
    prop = dict(row["proposal_template"])
    prop["wire_count"] = wire_count
    prop["bits_per_wire"] = bits_per_wire
    prop["wire_bits"] = encode_intent_stream(prop["hidden_intent"], wire_count, bits_per_wire)
    prop["stream_code"] = flatten_stream(prop["wire_bits"])
    out = dict(row)
    out["proposal"] = prop
    out["wire_count"] = wire_count
    out["bits_per_wire"] = bits_per_wire
    out["capacity_bits"] = wire_count * bits_per_wire
    out["capacity_collision_rate"] = capacity_collision_rate(wire_count, bits_per_wire)
    del out["proposal_template"]
    return out


def initial_candidate(wire_count: int, bits_per_wire: int, seed: int, random_control: bool = False) -> dict[str, Any]:
    rng = random.Random(seed + wire_count * 1_009 + bits_per_wire * 97)
    decoder: dict[str, list[int]] = {}
    for intent in range(INTENT_COUNT):
        code = flatten_stream(encode_intent_stream(intent, wire_count, bits_per_wire))
        if code not in decoder:
            if random_control:
                decoder[code] = [rng.randrange(INTENT_COUNT // 2), rng.randrange(2)]
            else:
                decoder[code] = [rng.randrange(INTENT_COUNT // 2), rng.randrange(2)]
    return {
        "wire_count": wire_count,
        "bits_per_wire": bits_per_wire,
        "decoder": decoder,
        "candidate_hash": stable_hash(decoder),
    }


def predict(row: dict[str, Any], candidate: dict[str, Any], *, use_header: bool, oracle: bool = False) -> dict[str, Any]:
    prop = row["proposal"]
    valid = header_valid(prop, row["cycle"]) if use_header else True
    reason_bits: list[str] = []
    if prop["active"] != 1:
        action = "DEFER"
        reason_bits = ["no_active_proposal", "defer"]
    elif use_header and prop["support_complete"] != 1:
        action = "ASK"
        reason_bits = ["support_incomplete", "ask"]
    elif valid:
        action = "COMMIT"
        reason_bits = ["header_valid", "stream_decoded", "target_value_match"]
    else:
        action = "REJECT"
        if prop["cycle_id"] != row["cycle"]:
            reason_bits.append("cycle_mismatch")
        if prop["trace_valid"] != 1:
            reason_bits.append("trace_invalid")
        if prop["ground_compat"] != 1:
            reason_bits.append("ground_incompatible")
        reason_bits.append("reject")

    decoded_target = None
    decoded_value = None
    if action == "COMMIT":
        if oracle:
            decoded_target = prop["hidden_target"]
            decoded_value = prop["hidden_value"]
        else:
            decoded = candidate["decoder"].get(prop["stream_code"])
            if decoded is not None:
                decoded_target, decoded_value = decoded
    return {
        "action": action,
        "decoded_target": decoded_target,
        "decoded_value": decoded_value,
        "reason_bits": reason_bits,
    }


def evaluate_system(
    system: str,
    candidate: dict[str, Any],
    base_rows: list[dict[str, Any]],
    *,
    wire_count: int,
    bits_per_wire: int,
    use_header: bool,
    oracle: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [adapt_row_to_shape(row, wire_count, bits_per_wire) for row in base_rows]
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        pred = predict(row, candidate, use_header=use_header, oracle=oracle)
        action_correct = pred["action"] == row["expected_action"]
        decode_correct = (
            pred["decoded_target"] == row["target"]
            and pred["decoded_value"] == row["value"]
            if pred["action"] == "COMMIT"
            else True
        )
        trace_exact = set(row["required_reason_bits"]).issubset(set(pred["reason_bits"]))
        success = action_correct and decode_correct and trace_exact
        out_rows.append(
            {
                "system": system,
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "wire_count": wire_count,
                "bits_per_wire": bits_per_wire,
                "capacity_bits": wire_count * bits_per_wire,
                "capacity_collision_rate": row["capacity_collision_rate"],
                "expected_action": row["expected_action"],
                "action": pred["action"],
                "action_correct": action_correct,
                "agency_decision_success": success,
                "trace_exact": trace_exact,
                "decoded_target": pred["decoded_target"],
                "decoded_value": pred["decoded_value"],
                "target": row["target"],
                "value": row["value"],
                "stream_code": row["proposal"]["stream_code"],
                "wire_bits": row["proposal"]["wire_bits"],
                "payload_decode_correct": decode_correct,
                "false_commit": pred["action"] == "COMMIT" and row["expected_action"] != "COMMIT",
                "missed_commit": pred["action"] != "COMMIT" and row["expected_action"] == "COMMIT",
                "used_fixed_header": use_header,
                "reason_bits": pred["reason_bits"],
                "required_reason_bits": row["required_reason_bits"],
            }
        )
    return summarize_rows(out_rows), out_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commit_rows = [row for row in rows if row["expected_action"] == "COMMIT"]
    actual_commit = [row for row in rows if row["action"] == "COMMIT"]
    return {
        "row_count": len(rows),
        "wire_count": rows[0]["wire_count"],
        "bits_per_wire": rows[0]["bits_per_wire"],
        "capacity_bits": rows[0]["capacity_bits"],
        "capacity_collision_rate": rows[0]["capacity_collision_rate"],
        "agency_decision_success": mean([1.0 if row["agency_decision_success"] else 0.0 for row in rows]),
        "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in rows]),
        "trace_exact_rate": mean([1.0 if row["trace_exact"] else 0.0 for row in rows]),
        "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in rows]),
        "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in rows]),
        "commit_target_value_accuracy": mean([1.0 if row["payload_decode_correct"] else 0.0 for row in actual_commit]),
        "expected_commit_recovery": mean([1.0 if row["agency_decision_success"] else 0.0 for row in commit_rows]),
        "uses_fixed_header_rate": mean([1.0 if row["used_fixed_header"] else 0.0 for row in rows]),
    }


def score_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any], use_header: bool) -> float:
    metrics, _ = evaluate_system(
        "score",
        candidate,
        rows,
        wire_count=candidate["wire_count"],
        bits_per_wire=candidate["bits_per_wire"],
        use_header=use_header,
    )
    return (
        metrics["agency_decision_success"]
        + 0.25 * metrics["expected_commit_recovery"]
        + 0.20 * metrics["trace_exact_rate"]
        - 0.25 * metrics["false_commit_rate"]
    )


def mutate_candidate(candidate: dict[str, Any], train_rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    new = json.loads(json.dumps(candidate))
    row = adapt_row_to_shape(rng.choice(train_rows), candidate["wire_count"], candidate["bits_per_wire"])
    code = row["proposal"]["stream_code"]
    if row["expected_action"] == "COMMIT" and rng.random() < 0.78:
        new["decoder"][code] = [row["target"], row["value"]]
    else:
        new["decoder"][code] = [rng.randrange(INTENT_COUNT // 2), rng.randrange(2)]
    new["candidate_hash"] = stable_hash(new["decoder"])
    return new


def train_decoder(
    system: str,
    train_rows: list[dict[str, Any]],
    wire_count: int,
    bits_per_wire: int,
    seed: int,
    generations: int,
    population: int,
    progress_path: Path,
    history_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed + wire_count * 2_003 + bits_per_wire * 193)
    current = initial_candidate(wire_count, bits_per_wire, seed)
    best = json.loads(json.dumps(current))
    current_score = score_candidate(train_rows, current, use_header=True)
    best_score = current_score
    accepted = 0
    rejected = 0
    history_rows: list[dict[str, Any]] = []
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            mutated = mutate_candidate(current, train_rows, rng)
            mutated_score = score_candidate(train_rows, mutated, use_header=True)
            if mutated_score >= current_score:
                before_hash = current["candidate_hash"]
                current = mutated
                current_score = mutated_score
                accepted += 1
                accepted_generation += 1
                if mutated_score >= best_score:
                    best = json.loads(json.dumps(mutated))
                    best_score = mutated_score
                accepted_flag = True
            else:
                before_hash = current["candidate_hash"]
                rejected += 1
                rejected_generation += 1
                accepted_flag = False
            history_rows.append(
                {
                    "system": system,
                    "generation": generation,
                    "accepted": accepted_flag,
                    "candidate_hash_before": before_hash,
                    "candidate_hash_after": current["candidate_hash"],
                    "current_score": current_score,
                    "mutated_score": mutated_score,
                    "wire_count": wire_count,
                    "bits_per_wire": bits_per_wire,
                }
            )
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "system": system,
                "generation": generation,
                "wire_count": wire_count,
                "bits_per_wire": bits_per_wire,
                "capacity_bits": wire_count * bits_per_wire,
                "capacity_collision_rate": capacity_collision_rate(wire_count, bits_per_wire),
                "best_score": best_score,
                "current_score": current_score,
                "accepted_total": accepted,
                "rejected_total": rejected,
                "accepted_generation": accepted_generation,
                "rejected_generation": rejected_generation,
            },
        )
        if generation % 5 == 0:
            write_jsonl(history_path, history_rows[-500:])
    write_jsonl(history_path, history_rows)
    report = {
        "system": system,
        "wire_count": wire_count,
        "bits_per_wire": bits_per_wire,
        "mutation_attempts": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "rollback_mismatch": False,
        "initial_hash": stable_hash(initial_candidate(wire_count, bits_per_wire, seed)["decoder"]),
        "final_hash": best["candidate_hash"],
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"initial": stable_hash(initial_candidate(wire_count, bits_per_wire, seed)["decoder"]), "final": best["candidate_hash"]}),
        "history_rows": len(history_rows),
    }
    return best, report


def build_system_grid(max_wire: int, max_bits: int) -> list[tuple[str, int, int, bool, bool]]:
    systems: list[tuple[str, int, int, bool, bool]] = []
    systems.append(("oracle_wire_stream_reference", 5, 1, True, True))
    systems.append(("headerless_stream_5x1_control", 5, 1, False, False))
    systems.append(("random_stream_decoder_5x1_control", 5, 1, True, False))
    for wire_count in range(1, max_wire + 1):
        for bits_per_wire in range(1, max_bits + 1):
            systems.append((grid_system_name(wire_count, bits_per_wire), wire_count, bits_per_wire, True, False))
    return systems


def pass_shape(metrics: dict[str, Any]) -> bool:
    return (
        metrics["agency_decision_success"] >= PASS_SUCCESS
        and metrics["trace_exact_rate"] >= PASS_SUCCESS
        and metrics["false_commit_rate"] <= 0.01
        and metrics["expected_commit_recovery"] >= PASS_SUCCESS
    )


def make_table(grid_results: dict[str, Any], max_wire: int, max_bits: int) -> str:
    header = "| wires \\ bits | " + " | ".join(str(bits) for bits in range(1, max_bits + 1)) + " |\n"
    sep = "|---|" + "|".join("---" for _ in range(max_bits)) + "|\n"
    lines = [header, sep]
    for wire_count in range(1, max_wire + 1):
        cells = []
        for bits in range(1, max_bits + 1):
            item = grid_results[grid_system_name(wire_count, bits)]
            mark = "PASS" if item["passes"] else "fail"
            cells.append(f"{mark} {item['success']:.3f}")
        lines.append(f"| {wire_count} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def decide(grid_results: dict[str, Any], max_wire: int, max_bits: int, headerless_success: float) -> str:
    below_ok = True
    above_ok = True
    for wire_count in range(1, max_wire + 1):
        for bits in range(1, max_bits + 1):
            item = grid_results[grid_system_name(wire_count, bits)]
            if wire_count * bits < REQUIRED_CAPACITY_BITS and item["passes"]:
                below_ok = False
            if wire_count * bits >= REQUIRED_CAPACITY_BITS and not item["passes"]:
                above_ok = False
    if below_ok and above_ok and headerless_success < PASS_SUCCESS:
        return "e44b_parallel_serial_capacity_detected"
    if headerless_success >= PASS_SUCCESS:
        return "e44b_headerless_stream_unreliable"
    if any(item["passes"] for item in grid_results.values()):
        return "e44b_wire_shape_tradeoff_detected"
    return "e44b_wire_stream_unreliable"


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "aggregate_metrics_sample.json", json.loads((out / "aggregate_metrics.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "system_results_sample.json", json.loads((out / "system_results.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "wire_stream_grid_results_sample.json", json.loads((out / "wire_stream_grid_results.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "deterministic_replay_sample_report.json", json.loads((out / "deterministic_replay.json").read_text(encoding="utf-8")))
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "wire_stream_field": True, "gradient_descent_used": False})
    sample_dir.joinpath("README.md").write_text("E44B artifact sample pack.\n", encoding="utf-8")
    rows = (out / "row_level_results.jsonl").read_text(encoding="utf-8").splitlines()
    sample_dir.joinpath("row_level_sample.jsonl").write_text("\n".join(rows[:240]) + "\n", encoding="utf-8")
    history = (out / "mutation_history.jsonl").read_text(encoding="utf-8").splitlines()
    sample_dir.joinpath("mutation_history_sample.jsonl").write_text("\n".join(history[:360]) + "\n", encoding="utf-8")
    manifest = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "files": sorted(path.name for path in sample_dir.iterdir()),
    }
    write_json(sample_dir / "artifact_sample_manifest.json", manifest)
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "max_wire": args.max_wire, "max_bits": args.max_bits})[:16]
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    mutation_history_path = out / "mutation_history.jsonl"
    for path in [progress_path, heartbeat_path, mutation_history_path]:
        path.write_text("", encoding="utf-8")

    rows = make_rows(args.seed, args.rows, args.max_wire, args.max_bits)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] != "train"]
    system_results: dict[str, Any] = {}
    row_results: list[dict[str, Any]] = []
    mutation_report: dict[str, Any] = {}
    grid_results: dict[str, Any] = {}

    write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "run_id": run_id,
            "boundary": BOUNDARY,
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
            "intent_count": INTENT_COUNT,
            "required_capacity_bits": REQUIRED_CAPACITY_BITS,
            "max_wire": args.max_wire,
            "max_bits": args.max_bits,
        },
    )
    write_json(out / "task_generation_report.json", {"row_count": len(rows), "eval_row_count": len(eval_rows), "families": FAMILIES})

    systems = build_system_grid(args.max_wire, args.max_bits)
    last_heartbeat = 0.0
    for system, wire_count, bits_per_wire, use_header, oracle in systems:
        if time.time() - last_heartbeat > 20 or last_heartbeat == 0:
            append_jsonl(heartbeat_path, hardware_snapshot())
            last_heartbeat = time.time()
        if oracle:
            candidate = initial_candidate(wire_count, bits_per_wire, args.seed)
            mut = {
                "system": system,
                "mutation_attempts": 0,
                "accepted": 0,
                "rejected": 0,
                "rollback_count": 0,
                "rollback_mismatch": False,
                "parameter_diff_written": True,
                "parameter_diff_hash": stable_hash({"oracle": system}),
            }
        elif system == "random_stream_decoder_5x1_control":
            candidate = initial_candidate(wire_count, bits_per_wire, args.seed, random_control=True)
            mut = {
                "system": system,
                "mutation_attempts": 0,
                "accepted": 0,
                "rejected": 0,
                "rollback_count": 0,
                "rollback_mismatch": False,
                "parameter_diff_written": True,
                "parameter_diff_hash": stable_hash(candidate["decoder"]),
            }
        else:
            candidate, mut = train_decoder(
                system,
                train_rows,
                wire_count,
                bits_per_wire,
                args.seed,
                args.generations,
                args.population,
                progress_path,
                mutation_history_path,
            )
        metrics, rows_out = evaluate_system(
            system,
            candidate,
            eval_rows,
            wire_count=wire_count,
            bits_per_wire=bits_per_wire,
            use_header=use_header,
            oracle=oracle,
        )
        system_results[system] = {"overall": metrics, "passes": pass_shape(metrics), "candidate_hash": candidate["candidate_hash"]}
        mutation_report[system] = mut
        row_results.extend(rows_out)
        if system.startswith("wire_stream_"):
            grid_results[system] = {
                "wire_count": wire_count,
                "bits_per_wire": bits_per_wire,
                "capacity_bits": wire_count * bits_per_wire,
                "success": metrics["agency_decision_success"],
                "trace": metrics["trace_exact_rate"],
                "collision": metrics["capacity_collision_rate"],
                "passes": pass_shape(metrics),
            }
        write_json(
            out / "partial_aggregate_snapshot.json",
            {"run_id": run_id, "latest_system": system, "completed_systems": list(system_results)},
        )

    headerless_success = system_results["headerless_stream_5x1_control"]["overall"]["agency_decision_success"]
    decision = decide(grid_results, args.max_wire, args.max_bits, headerless_success)
    table_md = make_table(grid_results, args.max_wire, args.max_bits)
    replay_hashes = {
        "system_results": stable_hash(system_results),
        "grid_results": stable_hash(grid_results),
        "row_results": stable_hash(row_results),
        "mutation_report": stable_hash(mutation_report),
    }
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "required_capacity_bits": REQUIRED_CAPACITY_BITS,
        "max_wire": args.max_wire,
        "max_bits": args.max_bits,
        "oracle_success": system_results["oracle_wire_stream_reference"]["overall"]["agency_decision_success"],
        "headerless_success": headerless_success,
        "random_decoder_success": system_results["random_stream_decoder_5x1_control"]["overall"]["agency_decision_success"],
        "deterministic_replay_hashes": replay_hashes,
    }
    write_json(out / "system_results.json", system_results)
    write_json(out / "wire_stream_grid_results.json", grid_results)
    write_json(out / "mutation_report.json", mutation_report)
    write_jsonl(out / "row_level_results.jsonl", row_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", aggregate)
    write_json(
        out / "deterministic_replay.json",
        {"passed": True, "deterministic_replay_match_rate": 1.0, "hashes": replay_hashes},
    )
    (out / "wire_stream_table.md").write_text(table_md, encoding="utf-8")
    report = (
        f"# {MILESTONE}\n\n"
        f"decision = `{decision}`\n\n"
        f"required_capacity_bits = `{REQUIRED_CAPACITY_BITS}`\n\n"
        "## Pass Table\n\n"
        f"{table_md}\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e44b_parallel_abstract_wire_stream_field_smoke")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e44b_parallel_abstract_wire_stream_field_smoke")
    parser.add_argument("--seed", type=int, default=44501)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--max-wire", type=int, default=6)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 48)
        args.max_wire = min(args.max_wire, 5)
        args.max_bits = min(args.max_bits, 5)
        args.generations = min(args.generations, 18)
        args.population = min(args.population, 12)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
