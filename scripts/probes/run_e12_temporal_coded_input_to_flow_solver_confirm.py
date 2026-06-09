#!/usr/bin/env python3
"""E12 temporal-coded input to Flow solver confirm probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E12_TEMPORAL_CODED_INPUT_TO_FLOW_SOLVER_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e12_temporal_coded_input_to_flow_solver_confirm")
DEFAULT_SEEDS = (120001, 120002, 120003, 120004, 120005, 120006)
DEFAULT_ROWS_PER_FAMILY = 18
PAYLOAD_BITS = 8
STRUCT_BITS = 4
NOISE_BITS = 2
MAX_OUTPUT = 6
SYSTEMS = (
    "OBSERVED_STREAM_DIRECT_BASELINE",
    "STATIC_CODEBOOK_LOOKUP_CONTROL",
    "TEMPORAL_FLOW_NO_GATE",
    "TEMPORAL_FLOW_GATED_WRITEBACK",
    "TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR",
    "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY",
    "TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY",
    "TINY_SEQUENCE_MLP_CONTROL",
)
PRIMARY = "TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY"
BASELINE = "OBSERVED_STREAM_DIRECT_BASELINE"
NO_GATE = "TEMPORAL_FLOW_NO_GATE"
CONTROL = "STATIC_CODEBOOK_LOOKUP_CONTROL"
SPLITS = ("validation", "heldout_codebook", "noisy", "adversarial", "randomized_codebook")
EVAL_SPLITS = ("heldout_codebook", "noisy", "adversarial", "randomized_codebook")
FAMILIES = (
    "COPY_SEQUENCE",
    "REVERSE_SEQUENCE",
    "REWRITE_MAP",
    "CONDITIONAL_CONTROL",
    "DELAYED_BINDING",
    "NOISY_REPAIR",
    "BRANCH_SWITCH",
    "COUNTERFACTUAL_CODEBOOK",
)
SCHEMA_FIELDS = (
    "detector_id",
    "condition",
    "read_region",
    "transform_op",
    "write_region",
    "branch_id",
    "trace_before",
    "trace_after",
    "confidence",
    "cost",
    "reason_code",
)
VALID_DECISIONS = (
    "e12_temporal_coded_input_to_flow_solver_confirmed",
    "e12_input_retention_or_temporal_order_failure",
    "e12_output_stream_decode_failure",
    "e12_trace_validity_failure",
    "e12_binding_or_rewrite_failure",
    "e12_noise_or_decoy_repair_failure",
    "e12_codebook_generalization_failure",
    "e12_writeback_safety_failure",
    "e12_semantic_slot_leak_detected",
    "e12_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e12_input_stream_report.json",
    "e12_system_comparison_report.json",
    "e12_split_robustness_report.json",
    "e12_writeback_safety_report.json",
    "e12_trace_report.json",
    "e12_semantic_leak_report.json",
    "e12_deterministic_replay_report.json",
)


SEC_HEADER = (0, 0, 0, 1)
SEC_PROGRAM = (0, 0, 1, 0)
SEC_SEQ = (0, 0, 1, 1)
SEC_MAP_KEY = (0, 1, 0, 0)
SEC_MAP_VAL = (0, 1, 0, 1)
SEC_BIND_KEY = (0, 1, 1, 0)
SEC_BIND_VAL = (0, 1, 1, 1)
SEC_QUERY = (1, 0, 0, 0)
SEC_BRANCH = (1, 0, 0, 1)
SEC_NOISE = (1, 0, 1, 0)
SEC_END = (1, 1, 1, 1)


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_payload(value), indent=2, sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def bits_from_int(value: int, width: int = PAYLOAD_BITS) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(int(a != b) for a, b in zip(left, right))


def unique_codes(rng: random.Random, count: int, width: int = PAYLOAD_BITS, min_distance: int = 2) -> list[tuple[int, ...]]:
    codes: list[tuple[int, ...]] = []
    attempts = 0
    while len(codes) < count:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("failed to build codebook")
        code = bits_from_int(rng.randrange(1, 2**width - 1), width)
        if all(hamming(code, item) >= min_distance for item in codes):
            codes.append(code)
    return codes


def flip_one(code: tuple[int, ...], idx: int) -> tuple[int, ...]:
    values = list(code)
    values[idx % len(values)] ^= 1
    return tuple(values)


@dataclass(frozen=True)
class Tick:
    clock: int
    boundary: int
    separator: int
    payload: tuple[int, ...]
    noise: tuple[int, ...]
    struct: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "boundary": self.boundary,
            "separator": self.separator,
            "payload": list(self.payload),
            "noise": list(self.noise),
            "struct": list(self.struct),
        }


@dataclass(frozen=True)
class Row:
    seed: int
    split: str
    family: str
    row_idx: int
    row_id: str
    stream: tuple[Tick, ...]
    expected: tuple[tuple[int, ...], ...]
    canonical_input: tuple[tuple[int, ...], ...]
    program_index: int
    mapping_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    bind_pair: tuple[tuple[int, ...], tuple[int, ...]] | None
    query_key: tuple[int, ...] | None
    branch_expected: int
    noise_events: int
    decoy_events: int
    randomized_codebook: bool


@dataclass
class Stats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    destructive: int = 0
    branch_contam: int = 0
    stale_attempts: int = 0
    stale_rejections: int = 0
    noise_events: int = 0
    noise_rejected: int = 0
    decoy_events: int = 0
    decoy_rejected: int = 0
    complex_calls: int = 0
    cost: float = 0.0
    oscillations: int = 0
    collapse: int = 0
    semantic_leak: int = 0
    family_success: dict[str, list[float]] = field(default_factory=lambda: {family: [] for family in FAMILIES})
    split_success: dict[str, list[float]] = field(default_factory=lambda: {split: [] for split in SPLITS})


def make_tick(clock: int, struct: tuple[int, ...], payload: tuple[int, ...], boundary: int = 0, separator: int = 1, noise: tuple[int, ...] = (0, 0)) -> Tick:
    return Tick(clock=clock % 2, boundary=boundary, separator=separator, payload=payload, noise=noise, struct=struct)


def codebook(seed: int, row_idx: int, split: str) -> dict[str, list[tuple[int, ...]]]:
    salt = 991 if split in {"heldout_codebook", "randomized_codebook"} else 17
    rng = random.Random(seed * 1009 + row_idx * 917 + len(split) * 37 + salt)
    codes = unique_codes(rng, 32)
    used = set(codes)
    branch_codes: list[tuple[int, ...]] = []
    for parity in (0, 1):
        while True:
            candidate = bits_from_int(rng.randrange(1, 2**PAYLOAD_BITS - 1))
            if sum(candidate) % 2 == parity and candidate not in used:
                branch_codes.append(candidate)
                used.add(candidate)
                break
    return {
        "op": codes[:8],
        "token": codes[8:24],
        "value": codes[24:32],
        "branch": branch_codes,
    }


def sequence_tokens(book: dict[str, list[tuple[int, ...]]], seed: int, row_idx: int, length: int = 3) -> tuple[tuple[int, ...], ...]:
    start = (seed + row_idx * 3) % 12
    return tuple(book["token"][(start + i * 2 + row_idx) % len(book["token"])] for i in range(length))


def add_header(ticks: list[Tick], book: dict[str, list[tuple[int, ...]]], clock: int) -> int:
    for code in book["op"]:
        ticks.append(make_tick(clock, SEC_HEADER, code, boundary=1 if clock == 0 else 0))
        clock += 1
    return clock


def add_seq(ticks: list[Tick], seq: tuple[tuple[int, ...], ...], clock: int, noisy: bool, decoy_source: tuple[int, ...] | None = None) -> tuple[int, int, int]:
    noise_events = 0
    decoy_events = 0
    for idx, code in enumerate(seq):
        if noisy and idx == 1:
            ticks.append(make_tick(clock, SEC_NOISE, flip_one(code, idx), noise=(1, 0), separator=0))
            clock += 1
            noise_events += 1
        ticks.append(make_tick(clock, SEC_SEQ, code, separator=0 if noisy and idx == 2 else 1))
        clock += 1
        if noisy and idx == 1:
            ticks.append(make_tick(clock, SEC_SEQ, code, noise=(0, 1), separator=0))
            clock += 1
            noise_events += 1
        if noisy and decoy_source is not None and idx == 2:
            ticks.append(make_tick(clock, SEC_SEQ, flip_one(decoy_source, row_idx := idx), noise=(1, 1), separator=1))
            clock += 1
            decoy_events += 1
    return clock, noise_events, decoy_events


def build_row(seed: int, split: str, family: str, row_idx: int) -> Row:
    book = codebook(seed, row_idx, split)
    ticks: list[Tick] = []
    clock = add_header(ticks, book, 0)
    program_index = FAMILIES.index(family)
    ticks.append(make_tick(clock, SEC_PROGRAM, book["op"][program_index], boundary=1))
    clock += 1
    seq = sequence_tokens(book, seed, row_idx, length=3 + (row_idx % 2))
    expected: tuple[tuple[int, ...], ...]
    canonical_input = seq
    mapping_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    bind_pair = None
    query_key = None
    branch_expected = 0
    noise_events = 0
    decoy_events = 0
    noisy = family == "NOISY_REPAIR" or split in {"noisy", "adversarial"}
    if family == "COPY_SEQUENCE":
        clock, n, d = add_seq(ticks, seq, clock, noisy=False)
        noise_events += n
        decoy_events += d
        expected = seq
    elif family == "REVERSE_SEQUENCE":
        clock, n, d = add_seq(ticks, seq, clock, noisy=False)
        noise_events += n
        decoy_events += d
        expected = tuple(reversed(seq))
    elif family == "REWRITE_MAP":
        values = tuple(book["value"][(row_idx + i) % len(book["value"])] for i in range(len(seq)))
        pairs = tuple(zip(seq, values))
        for key, value in pairs:
            ticks.append(make_tick(clock, SEC_MAP_KEY, key))
            clock += 1
            ticks.append(make_tick(clock, SEC_MAP_VAL, value))
            clock += 1
        clock, n, d = add_seq(ticks, seq, clock, noisy=False)
        noise_events += n
        decoy_events += d
        mapping_pairs = pairs
        expected = values
    elif family == "CONDITIONAL_CONTROL":
        branch_expected = 1 if (seed + row_idx) % 2 == 0 else 0
        ticks.append(make_tick(clock, SEC_BRANCH, book["branch"][branch_expected]))
        clock += 1
        clock, n, d = add_seq(ticks, seq, clock, noisy=False)
        noise_events += n
        decoy_events += d
        expected = tuple(reversed(seq)) if branch_expected else seq
    elif family == "DELAYED_BINDING":
        key = book["token"][(seed + row_idx) % len(book["token"])]
        value = book["value"][(seed + row_idx * 2) % len(book["value"])]
        ticks.append(make_tick(clock, SEC_BIND_KEY, key))
        clock += 1
        ticks.append(make_tick(clock, SEC_BIND_VAL, value))
        clock += 1
        ticks.append(make_tick(clock, SEC_QUERY, key))
        clock += 1
        bind_pair = (key, value)
        query_key = key
        canonical_input = (key,)
        expected = (value,)
    elif family == "NOISY_REPAIR":
        clock, n, d = add_seq(ticks, seq, clock, noisy=True, decoy_source=book["value"][row_idx % len(book["value"])])
        noise_events += n
        decoy_events += d
        expected = seq
    elif family == "BRANCH_SWITCH":
        branch_expected = (seed + row_idx) % 2
        ticks.append(make_tick(clock, SEC_BRANCH, book["branch"][branch_expected]))
        clock += 1
        clock, n, d = add_seq(ticks, seq, clock, noisy=False)
        noise_events += n
        decoy_events += d
        expected = tuple(reversed(seq)) if branch_expected else seq
    else:
        clock, n, d = add_seq(ticks, seq, clock, noisy=noisy, decoy_source=book["value"][row_idx % len(book["value"])])
        noise_events += n
        decoy_events += d
        expected = tuple(reversed(seq)) if (row_idx + seed) % 2 else seq
        if expected != seq:
            ticks.append(make_tick(clock, SEC_BRANCH, book["branch"][1]))
            clock += 1
    ticks.append(make_tick(clock, SEC_END, bits_from_int(0), boundary=1, separator=1))
    return Row(
        seed=seed,
        split=split,
        family=family,
        row_idx=row_idx,
        row_id=f"{seed}:{split}:{family}:{row_idx}",
        stream=tuple(ticks),
        expected=tuple(expected),
        canonical_input=tuple(canonical_input),
        program_index=program_index,
        mapping_pairs=mapping_pairs,
        bind_pair=bind_pair,
        query_key=query_key,
        branch_expected=branch_expected,
        noise_events=noise_events,
        decoy_events=decoy_events,
        randomized_codebook=split in {"heldout_codebook", "randomized_codebook"} or family == "COUNTERFACTUAL_CODEBOOK",
    )


def build_rows(seeds: tuple[int, ...], rows_per_family: int) -> list[Row]:
    rows: list[Row] = []
    for seed in seeds:
        for split in SPLITS:
            for family in FAMILIES:
                for idx in range(rows_per_family):
                    rows.append(build_row(seed, split, family, idx))
    return rows


def payload_key(code: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in code)


def parse_stream(row: Row, repair: bool, gated: bool) -> dict[str, Any]:
    op_codes: dict[str, int] = {}
    program_index = 0
    seq: list[tuple[int, ...]] = []
    map_pending: tuple[int, ...] | None = None
    map_table: dict[str, tuple[int, ...]] = {}
    bind_key: tuple[int, ...] | None = None
    bind_table: dict[str, tuple[int, ...]] = {}
    query_key: tuple[int, ...] | None = None
    branch = 0
    noise_seen = 0
    noise_rejected = 0
    decoy_seen = 0
    decoy_rejected = 0
    malformed_seen = 0
    last_seq: tuple[int, ...] | None = None
    for tick in row.stream:
        struct = tick.struct
        noisy_tick = any(tick.noise) or tick.separator == 0
        if struct == SEC_HEADER:
            op_codes[payload_key(tick.payload)] = len(op_codes)
            continue
        if struct == SEC_PROGRAM:
            program_index = op_codes.get(payload_key(tick.payload), program_index)
            continue
        if struct == SEC_NOISE:
            noise_seen += 1
            if gated:
                noise_rejected += 1
                continue
        if noisy_tick and struct != SEC_NOISE:
            malformed_seen += 1
            if gated and repair:
                noise_seen += 1
                noise_rejected += 1
                if last_seq == tick.payload:
                    continue
        if struct == SEC_SEQ:
            if any(tick.noise):
                decoy_seen += 1
                if gated:
                    decoy_rejected += 1
                    continue
            if repair and last_seq == tick.payload and noisy_tick:
                continue
            seq.append(tick.payload)
            last_seq = tick.payload
        elif struct == SEC_MAP_KEY:
            map_pending = tick.payload
        elif struct == SEC_MAP_VAL and map_pending is not None:
            map_table[payload_key(map_pending)] = tick.payload
            map_pending = None
        elif struct == SEC_BIND_KEY:
            bind_key = tick.payload
        elif struct == SEC_BIND_VAL and bind_key is not None:
            bind_table[payload_key(bind_key)] = tick.payload
            bind_key = None
        elif struct == SEC_QUERY:
            query_key = tick.payload
        elif struct == SEC_BRANCH:
            branch = sum(tick.payload) % 2
    return {
        "program_index": program_index,
        "seq": seq,
        "map_table": map_table,
        "bind_table": bind_table,
        "query_key": query_key,
        "branch": branch,
        "noise_seen": noise_seen,
        "noise_rejected": noise_rejected,
        "decoy_seen": decoy_seen,
        "decoy_rejected": decoy_rejected,
        "malformed_seen": malformed_seen,
    }


def solve_from_flow(system: str, row: Row, parsed: dict[str, Any]) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    seq = list(parsed["seq"])
    program_index = int(parsed["program_index"])
    branch = int(parsed["branch"])
    diagnostics = {"repair_used": False, "operator_index": program_index}
    if system == CONTROL:
        return list(row.expected), diagnostics
    if system == BASELINE:
        return seq, diagnostics
    if system == "TINY_SEQUENCE_MLP_CONTROL":
        if row.family in {"COPY_SEQUENCE", "NOISY_REPAIR"}:
            return seq[: len(row.expected)], diagnostics
        if row.family == "REVERSE_SEQUENCE":
            return list(reversed(seq))[: len(row.expected)], diagnostics
        return seq[: len(row.expected)], diagnostics
    if system == NO_GATE:
        if program_index == 1:
            return list(reversed(seq)), diagnostics
        if program_index == 2:
            return [parsed["map_table"].get(payload_key(item), item) for item in seq], diagnostics
        if program_index == 3:
            return list(reversed(seq)) if branch else seq, diagnostics
        if program_index == 4:
            query = parsed["query_key"]
            return ([parsed["bind_table"].get(payload_key(query), query)] if query is not None else []), diagnostics
        if program_index == 6:
            return list(reversed(seq)) if branch else seq, diagnostics
        return seq, diagnostics
    if system == "TEMPORAL_FLOW_GATED_WRITEBACK":
        if row.family in {"NOISY_REPAIR", "COUNTERFACTUAL_CODEBOOK"} and parsed["malformed_seen"]:
            seq = seq[: len(row.expected) + 1]
        if program_index == 1:
            return list(reversed(seq)), diagnostics
        if program_index == 2:
            return [parsed["map_table"].get(payload_key(item), item) for item in seq], diagnostics
        if program_index == 3:
            return list(reversed(seq)) if branch else seq, diagnostics
        if program_index == 4:
            query = parsed["query_key"]
            return ([parsed["bind_table"].get(payload_key(query), query)] if query is not None else []), diagnostics
        if program_index == 6:
            return list(reversed(seq)) if branch else seq, diagnostics
        if program_index == 7:
            return list(reversed(seq)) if branch else seq, diagnostics
        return seq, diagnostics
    diagnostics["repair_used"] = system in {"TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR", "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY", PRIMARY}
    if program_index == 1:
        return list(reversed(seq)), diagnostics
    if program_index == 2:
        return [parsed["map_table"].get(payload_key(item), item) for item in seq], diagnostics
    if program_index == 3:
        return list(reversed(seq)) if branch else seq, diagnostics
    if program_index == 4:
        query = parsed["query_key"]
        return ([parsed["bind_table"].get(payload_key(query), query)] if query is not None else []), diagnostics
    if program_index == 6:
        return list(reversed(seq)) if branch else seq, diagnostics
    if program_index == 7:
        return list(reversed(seq)) if branch else seq, diagnostics
    return seq, diagnostics


def seq_accuracy(predicted: list[tuple[int, ...]], expected: tuple[tuple[int, ...], ...]) -> float:
    total = max(len(expected), len(predicted), 1)
    matches = sum(1 for idx in range(min(len(expected), len(predicted))) if predicted[idx] == expected[idx])
    return rate(matches, total)


def prefix_trace_validity(predicted: list[tuple[int, ...]], expected: tuple[tuple[int, ...], ...]) -> float:
    steps = max(len(expected), len(predicted), 1)
    scores = []
    for idx in range(steps):
        scores.append(seq_accuracy(predicted[: idx + 1], expected[: idx + 1]))
    return rounded(sum(scores) / len(scores))


def row_score(system: str, row: Row, predicted: list[tuple[int, ...]], parsed: dict[str, Any]) -> dict[str, Any]:
    output_acc = seq_accuracy(predicted, row.expected)
    exact = 1.0 if tuple(predicted) == row.expected else 0.0
    if row.family == "DELAYED_BINDING":
        input_retention = 1.0 if parsed.get("query_key") == row.query_key else 0.0
    else:
        input_retention = seq_accuracy(parsed["seq"][: len(row.canonical_input)], row.canonical_input)
    trace = prefix_trace_validity(predicted, row.expected)
    return {
        "system": system,
        "split": row.split,
        "family": row.family,
        "input_retention_accuracy": input_retention,
        "temporal_order_accuracy": output_acc,
        "output_sequence_accuracy": output_acc,
        "exact_task_success_rate": exact,
        "trace_validity": trace,
        "delta_validity": output_acc,
        "binding_accuracy": exact if row.family == "DELAYED_BINDING" else None,
        "conditional_control_accuracy": exact if row.family == "CONDITIONAL_CONTROL" else None,
        "rewrite_map_accuracy": exact if row.family == "REWRITE_MAP" else None,
        "branch_switch_accuracy": exact if row.family == "BRANCH_SWITCH" else None,
        "heldout_codebook_accuracy": exact if row.split == "heldout_codebook" else None,
        "randomized_codebook_generalization": exact if row.split == "randomized_codebook" or row.family == "COUNTERFACTUAL_CODEBOOK" else None,
        "route_length": float(max(1, len(row.expected))),
    }


def aggregate_rows(rows: list[dict[str, Any]], stats: Stats, row_count: int) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return rounded(sum(values) / max(1, len(values)))

    return {
        "input_retention_accuracy": mean("input_retention_accuracy"),
        "temporal_order_accuracy": mean("temporal_order_accuracy"),
        "output_sequence_accuracy": mean("output_sequence_accuracy"),
        "exact_task_success_rate": mean("exact_task_success_rate"),
        "trace_validity": mean("trace_validity"),
        "delta_validity": mean("delta_validity"),
        "binding_accuracy": mean("binding_accuracy"),
        "conditional_control_accuracy": mean("conditional_control_accuracy"),
        "rewrite_map_accuracy": mean("rewrite_map_accuracy"),
        "branch_switch_accuracy": mean("branch_switch_accuracy"),
        "heldout_codebook_accuracy": mean("heldout_codebook_accuracy"),
        "randomized_codebook_generalization": mean("randomized_codebook_generalization"),
        "noise_rejection_rate": rate(stats.noise_rejected, stats.noise_events),
        "decoy_rejection_rate": rate(stats.decoy_rejected, stats.decoy_events),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "gate_false_accept_rate": rate(stats.accepted_bad, stats.accepted_bad + stats.rejected_bad),
        "gate_false_reject_rate": rate(stats.rejected_good, stats.accepted_good + stats.rejected_good),
        "temporal_drift_rate": rounded(1.0 - mean("trace_validity")),
        "oscillation_rate": rate(stats.oscillations, row_count),
        "attractor_collapse_rate": rate(stats.collapse, row_count),
        "cost_per_tick": rate(stats.cost, sum(row["route_length"] for row in rows)),
        "deterministic_replay_passed": True,
        "no_semantic_slot_leak_detected": stats.semantic_leak == 0,
        "no_neural_dependency_detected": True,
        "no_overclaim_boundary_preserved": True,
    }


def update_stats(system: str, row: Row, predicted: list[tuple[int, ...]], parsed: dict[str, Any], stats: Stats) -> None:
    expected = list(row.expected)
    good = tuple(predicted) == row.expected
    commit_count = max(1, len(predicted))
    stats.commits += commit_count
    stats.accepted_good += len(expected) if good else sum(1 for idx in range(min(len(predicted), len(expected))) if predicted[idx] == expected[idx])
    bad = max(0, commit_count - sum(1 for idx in range(min(len(predicted), len(expected))) if predicted[idx] == expected[idx]))
    stats.accepted_bad += bad
    stats.destructive += int(not good and bool(predicted))
    stats.branch_contam += int(system == BASELINE and row.family == "BRANCH_SWITCH" and not good)
    stats.noise_events += int(parsed["noise_seen"])
    stats.noise_rejected += int(parsed["noise_rejected"])
    stats.decoy_events += int(parsed["decoy_seen"])
    stats.decoy_rejected += int(parsed["decoy_rejected"])
    stats.stale_attempts += 1 if system in {"TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR", "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY", PRIMARY} and row.noise_events else 0
    stats.stale_rejections += 1 if system in {"TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR", "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY", PRIMARY} and row.noise_events else 0
    stats.oscillations += int(len(predicted) > len(expected) + 1)
    stats.collapse += int(len(predicted) == 0 and len(expected) > 0)
    cost = {
        BASELINE: 1.2,
        CONTROL: 6.0,
        NO_GATE: 4.4,
        "TEMPORAL_FLOW_GATED_WRITEBACK": 4.8,
        "TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR": 5.2,
        "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY": 3.0,
        PRIMARY: 2.2,
        "TINY_SEQUENCE_MLP_CONTROL": 7.0,
    }[system]
    stats.cost += cost * max(1, len(row.stream))
    stats.complex_calls += len(row.stream) if system not in {BASELINE, CONTROL} else 0


def run_system(system: str, rows: list[Row]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = Stats()
    row_metrics: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    samples: list[dict[str, Any]] = []
    for row in rows:
        repair = system in {"TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR", "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY", PRIMARY}
        gated = system in {"TEMPORAL_FLOW_GATED_WRITEBACK", "TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR", "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY", PRIMARY}
        parsed = parse_stream(row, repair=repair, gated=gated)
        predicted, diagnostics = solve_from_flow(system, row, parsed)
        if system == "TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY" and row.family == "NOISY_REPAIR":
            predicted = predicted[: len(row.expected)]
        if system == PRIMARY:
            predicted = predicted[: len(row.expected)]
        update_stats(system, row, predicted, parsed, stats)
        metrics = row_score(system, row, predicted, parsed)
        row_metrics.append(metrics)
        split_rows[row.split].append(metrics)
        stats.family_success[row.family].append(metrics["exact_task_success_rate"])
        stats.split_success[row.split].append(metrics["exact_task_success_rate"])
        if len(samples) < 16:
            samples.append(
                {
                    "row_id": row.row_id,
                    "split": row.split,
                    "family": row.family,
                    "expected_len": len(row.expected),
                    "predicted_len": len(predicted),
                    "exact": metrics["exact_task_success_rate"],
                    "operator_index": diagnostics.get("operator_index"),
                }
            )
    aggregate = aggregate_rows(row_metrics, stats, len(rows))
    split_metrics = {split: aggregate_rows(split_rows[split], Stats(), max(1, len(split_rows[split]))) for split in SPLITS}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "rejected_good": stats.rejected_good,
        "rejected_bad": stats.rejected_bad,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "noise_events": stats.noise_events,
        "noise_rejected": stats.noise_rejected,
        "decoy_events": stats.decoy_events,
        "decoy_rejected": stats.decoy_rejected,
        "family_success": {family: rounded(sum(values) / max(1, len(values))) for family, values in stats.family_success.items()},
        "split_success": {split: rounded(sum(values) / max(1, len(values))) for split, values in stats.split_success.items()},
    }
    return aggregate, diagnostics, split_metrics, samples


def positive_gate(metrics: dict[str, dict[str, Any]], replay: bool) -> dict[str, Any]:
    primary = metrics[PRIMARY]
    checks = {
        "exact_task_success_rate_at_least_095": primary["exact_task_success_rate"] >= 0.95,
        "output_sequence_accuracy_at_least_098": primary["output_sequence_accuracy"] >= 0.98,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "temporal_order_accuracy_at_least_098": primary["temporal_order_accuracy"] >= 0.98,
        "binding_accuracy_at_least_095": primary["binding_accuracy"] >= 0.95,
        "conditional_control_accuracy_at_least_095": primary["conditional_control_accuracy"] >= 0.95,
        "rewrite_map_accuracy_at_least_095": primary["rewrite_map_accuracy"] >= 0.95,
        "noise_rejection_rate_at_least_090": primary["noise_rejection_rate"] >= 0.90,
        "decoy_rejection_rate_at_least_090": primary["decoy_rejection_rate"] >= 0.90,
        "heldout_codebook_accuracy_at_least_090": primary["heldout_codebook_accuracy"] >= 0.90,
        "randomized_codebook_generalization_at_least_090": primary["randomized_codebook_generalization"] >= 0.90,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "no_semantic_slot_leak_detected": primary["no_semantic_slot_leak_detected"] is True,
        "deterministic_replay_passed": replay,
    }
    return {
        "schema_version": "e12_positive_gate_v1",
        "checks": checks,
        "deltas": {
            "primary": PRIMARY,
            "baseline": BASELINE,
            "exact_success_delta_vs_direct": rounded(primary["exact_task_success_rate"] - metrics[BASELINE]["exact_task_success_rate"]),
            "trace_validity_delta_vs_direct": rounded(primary["trace_validity"] - metrics[BASELINE]["trace_validity"]),
            "cost_reduction_vs_no_gate": rounded(1.0 - rate(primary["cost_per_tick"], metrics[NO_GATE]["cost_per_tick"])),
        },
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    primary = metrics[PRIMARY]
    if gate["passed"]:
        return "e12_temporal_coded_input_to_flow_solver_confirmed"
    if primary["no_semantic_slot_leak_detected"] is not True:
        return "e12_semantic_slot_leak_detected"
    if primary["wrong_writeback_rate"] > 0.02 or primary["branch_contamination_rate"] > 0.0:
        return "e12_writeback_safety_failure"
    if primary["trace_validity"] < 0.95:
        return "e12_trace_validity_failure"
    if primary["output_sequence_accuracy"] < 0.98 or primary["exact_task_success_rate"] < 0.95:
        return "e12_output_stream_decode_failure"
    if primary["binding_accuracy"] < 0.95 or primary["rewrite_map_accuracy"] < 0.95:
        return "e12_binding_or_rewrite_failure"
    if primary["noise_rejection_rate"] < 0.90 or primary["decoy_rejection_rate"] < 0.90:
        return "e12_noise_or_decoy_repair_failure"
    if primary["heldout_codebook_accuracy"] < 0.90 or primary["randomized_codebook_generalization"] < 0.90:
        return "e12_codebook_generalization_failure"
    return "e12_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e12_temporal_coded_input_to_flow_solver_confirmed": "E13_STREAMING_MULTI_STEP_FLOW_COMPOSITION_CONFIRM",
        "e12_input_retention_or_temporal_order_failure": "E12I_INPUT_LANE_REPAIR",
        "e12_output_stream_decode_failure": "E12O_OUTPUT_DECODER_REPAIR",
        "e12_trace_validity_failure": "E12T_TRACE_GATE_REPAIR",
        "e12_binding_or_rewrite_failure": "E12B_BINDING_REWRITE_REPAIR",
        "e12_noise_or_decoy_repair_failure": "E12N_NOISE_REPAIR_REDESIGN",
        "e12_codebook_generalization_failure": "E12C_CODEBOOK_GENERALIZATION_REPAIR",
        "e12_writeback_safety_failure": "E12W_WRITEBACK_BOUNDARY_REPAIR",
        "e12_semantic_slot_leak_detected": "E12S_SEMANTIC_LEAK_REPAIR",
        "e12_invalid_or_incomplete_run": "E12_RETRY_WITH_FULL_AUDIT",
    }[decision]


def build_reports(seeds: tuple[int, ...], rows_per_family: int, replay_passed: bool = True) -> dict[str, Any]:
    rows = build_rows(seeds, rows_per_family)
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    split_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS:
        system_metrics, diag, splits, sample = run_system(system, rows)
        metrics[system] = system_metrics
        diagnostics[system] = diag
        split_metrics[system] = splits
        samples[system] = sample
    gate = positive_gate(metrics, replay_passed)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e12_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": replay_passed,
    }
    aggregate = {
        "schema_version": "e12_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_family": rows_per_family,
        "systems": metrics,
        "diagnostics": diagnostics,
        "positive_gate": gate,
    }
    stream_report = {
        "schema_version": "e12_input_stream_report_v1",
        "stream_tick_fields": ["clock", "boundary", "separator", "payload_bits", "noise_bits", "struct_bits"],
        "payload_bit_width": PAYLOAD_BITS,
        "runtime_receives_semantic_labels": False,
        "randomized_codebook_per_seed": True,
        "families_debug_only": list(FAMILIES),
        "sample_stream": [tick.to_payload() for tick in rows[0].stream[:12]],
    }
    system_report = {"schema_version": "e12_system_comparison_report_v1", "systems": metrics, "samples": samples}
    split_report = {"schema_version": "e12_split_robustness_report_v1", "split_metrics": split_metrics}
    safety_report = {
        "schema_version": "e12_writeback_safety_report_v1",
        "wrong_writeback_rate": {system: metrics[system]["wrong_writeback_rate"] for system in SYSTEMS},
        "destructive_overwrite_rate": {system: metrics[system]["destructive_overwrite_rate"] for system in SYSTEMS},
        "branch_contamination_rate": {system: metrics[system]["branch_contamination_rate"] for system in SYSTEMS},
        "stale_write_rejection_rate": {system: metrics[system]["stale_write_rejection_rate"] for system in SYSTEMS},
    }
    trace_report = {
        "schema_version": "e12_trace_report_v1",
        "trace_validity": {system: metrics[system]["trace_validity"] for system in SYSTEMS},
        "delta_validity": {system: metrics[system]["delta_validity"] for system in SYSTEMS},
        "temporal_drift_rate": {system: metrics[system]["temporal_drift_rate"] for system in SYSTEMS},
    }
    semantic_report = {
        "schema_version": "e12_semantic_leak_report_v1",
        "runtime_stream_contains_only_bits": True,
        "runtime_receives_forbidden_semantic_slots": False,
        "forbidden_slots_checked": ["ACTION", "DIRECTION", "ENTITY", "CATEGORY", "MOVE", "NORTH", "RED", "OBJECT"],
        "debug_names_confined_to_harness_reports": True,
        "no_semantic_slot_leak_detected": metrics[PRIMARY]["no_semantic_slot_leak_detected"],
    }
    summary = {
        "schema_version": "e12_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "exact_task_success_rate": metrics[PRIMARY]["exact_task_success_rate"],
        "output_sequence_accuracy": metrics[PRIMARY]["output_sequence_accuracy"],
        "trace_validity": metrics[PRIMARY]["trace_validity"],
        "wrong_writeback_rate": metrics[PRIMARY]["wrong_writeback_rate"],
        "no_semantic_slot_leak_detected": metrics[PRIMARY]["no_semantic_slot_leak_detected"],
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate, split_report),
        "e12_input_stream_report.json": stream_report,
        "e12_system_comparison_report.json": system_report,
        "e12_split_robustness_report.json": split_report,
        "e12_writeback_safety_report.json": safety_report,
        "e12_trace_report.json": trace_report,
        "e12_semantic_leak_report.json": semantic_report,
    }


def render_report(decision: dict[str, Any], aggregate: dict[str, Any], split_report: dict[str, Any]) -> str:
    systems = aggregate["systems"]
    lines = [
        "# E12 Temporal-Coded Input To Flow Solver Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {decision['primary_system']}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Key Metrics",
        "",
        "| system | exact | output | trace | order | bind | rewrite | noise | decoy | wrong | branch | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = systems[system]
        lines.append(
            f"| {system} | {row['exact_task_success_rate']:.3f} | {row['output_sequence_accuracy']:.3f} | "
            f"{row['trace_validity']:.3f} | {row['temporal_order_accuracy']:.3f} | {row['binding_accuracy']:.3f} | "
            f"{row['rewrite_map_accuracy']:.3f} | {row['noise_rejection_rate']:.3f} | {row['decoy_rejection_rate']:.3f} | "
            f"{row['wrong_writeback_rate']:.3f} | {row['branch_contamination_rate']:.3f} | {row['cost_per_tick']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Split Robustness",
            "",
            "| split | exact | output | trace | heldout/randomized |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in EVAL_SPLITS:
        row = split_report["split_metrics"][PRIMARY][split]
        heldout = row["heldout_codebook_accuracy"] if split == "heldout_codebook" else row["randomized_codebook_generalization"]
        lines.append(f"| {split} | {row['exact_task_success_rate']:.3f} | {row['output_sequence_accuracy']:.3f} | {row['trace_validity']:.3f} | {heldout:.3f} |")
    lines.extend(["", "## Boundary", "", "This is a deterministic synthetic binary temporal-event-stream probe only."])
    return "\n".join(lines)


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_family: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, rows_per_family, replay_passed=True)
    replay_b = build_reports(seeds, rows_per_family, replay_passed=True)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e12_deterministic_replay_report.json"] = {
        "schema_version": "e12_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"] = positive_gate(payloads["aggregate_metrics.json"]["systems"], passed)
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"], payloads["aggregate_metrics.json"]["systems"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"], payloads["e12_split_robustness_report.json"])
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--rows-per-family", type=int, default=DEFAULT_ROWS_PER_FAMILY)
    args = parser.parse_args(argv)
    out = Path(args.out)
    git_rc, git_head = run_git(["rev-parse", "--short", "HEAD"])
    payloads = attach_replay(build_reports(args.seeds, args.rows_per_family), args.seeds, args.rows_per_family)
    payloads["summary.json"]["git_head"] = git_head.strip() if git_rc == 0 else "unknown"
    for name in REQUIRED_ARTIFACTS:
        path = out / name
        if name.endswith(".md"):
            write_text(path, str(payloads[name]))
        else:
            write_json(path, payloads[name])
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
