#!/usr/bin/env python3
"""
D21e multi-slot/key-value A-block memory core.

Prototype goal:
    MARKER_A, PAYLOAD_A, MARKER_B, PAYLOAD_B, ..., QUERY_B -> PAYLOAD_B

The D21A byte lane and D21B context write lane are fixed. D21E evaluates a tiny
slot-addressed memory core that stores multiple payloads and emits only the
queried slot's context at query time.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from d21a_reciprocal_byte_ablock import all_visible_patterns, dedupe_entries, evaluate_ablock, robustness_metrics
from d21c_tiny_recurrent_ablock_core import byte_block, byte_logits, d21b_context_weights, entries_to_string, target_margins


ASCII_SHADE = " .:-=+*#%@"
DEFAULT_SEED = 20260502
VISIBLE_DIM = 8
CODE_DIM = 16
CONTEXT_DIM = 16
DEFAULT_MARKERS = (0xF1, 0xF2, 0xF3, 0xF4)
DEFAULT_QUERIES = (0x1F, 0x2F, 0x3F, 0x4F)


@dataclass(frozen=True)
class MemorySpec:
    candidate_id: int
    visible_dim: int
    code_dim: int
    state_dim: int
    context_dim: int
    slot_count: int
    family: str
    memory_edge_budget: int
    memory_entries: tuple[tuple[int, int, int, float], ...]


@dataclass(frozen=True)
class SequenceBatch:
    sequences: list[list[int]]
    payload_matrix: np.ndarray
    query_slots: np.ndarray
    target_payloads: np.ndarray
    slot_counts: np.ndarray
    distractor_lengths: np.ndarray
    patterns: np.ndarray
    markers: tuple[int, ...]
    queries: tuple[int, ...]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_hex_list(raw: str) -> tuple[int, ...]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part, 16) if part.lower().startswith("0x") else int(part))
    return tuple(values)


def dedupe_memory_entries(entries: Iterable[tuple[int, int, int, float]]) -> tuple[tuple[int, int, int, float], ...]:
    seen: dict[tuple[int, int, int], float] = {}
    for slot_idx, state_idx, visible_idx, value in entries:
        if abs(float(value)) > 1e-12:
            seen[(int(slot_idx), int(state_idx), int(visible_idx))] = float(value)
    return tuple((slot, state, visible, value) for (slot, state, visible), value in sorted(seen.items()))


def parse_entries(raw: str) -> tuple[tuple[int, int, int, float], ...]:
    entries = []
    for part in str(raw).split():
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) == 3:
            state_idx, visible_idx, value = pieces
            entries.append((0, int(state_idx), int(visible_idx), float(value)))
        else:
            slot_idx, state_idx, visible_idx, value = pieces
            entries.append((int(slot_idx), int(state_idx), int(visible_idx), float(value)))
    return dedupe_memory_entries(entries)


def entries_to_memory_string(entries: Sequence[tuple[int, int, int, float]]) -> str:
    return " ".join(f"{slot}:{state}:{visible}:{value:g}" for slot, state, visible, value in entries)


def memory_tensor(slot_count: int, state_dim: int, visible_dim: int, entries: Iterable[tuple[int, int, int, float]]) -> np.ndarray:
    tensor = np.zeros((slot_count, state_dim, visible_dim), dtype=np.float32)
    for slot_idx, state_idx, visible_idx, value in entries:
        if 0 <= slot_idx < slot_count and 0 <= state_idx < state_dim and 0 <= visible_idx < visible_dim:
            tensor[slot_idx, state_idx, visible_idx] = float(value)
    return tensor


def slot_start(slot_idx: int) -> int:
    return slot_idx * CONTEXT_DIM


def slot_query_context(states: np.ndarray, query_slots: np.ndarray, state_dim: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    context = np.zeros((states.shape[0], CONTEXT_DIM), dtype=np.float32)
    for row_idx, slot_idx in enumerate(query_slots.astype(np.int32)):
        start = slot_start(int(slot_idx))
        if start >= state_dim:
            continue
        limit = min(CONTEXT_DIM, state_dim - start)
        context[row_idx, :limit] = states[row_idx, start : start + limit]
    return context


def payload_pool(markers: Sequence[int], queries: Sequence[int]) -> np.ndarray:
    reserved = set(int(x) for x in markers) | set(int(x) for x in queries)
    return np.asarray([value for value in range(256) if value not in reserved], dtype=np.int32)


def make_sequence_batch(
    *,
    slot_counts: Sequence[int],
    distractor_lengths: Sequence[int],
    eval_sequences: int,
    markers: Sequence[int],
    queries: Sequence[int],
    strict_distractors: bool,
    seed: int,
) -> SequenceBatch:
    max_slots = max(slot_counts)
    if max_slots > len(markers) or max_slots > len(queries):
        raise ValueError("slot_count exceeds marker/query pair count")
    patterns = all_visible_patterns(VISIBLE_DIM)
    payload_values = payload_pool(markers, queries)
    rng = np.random.default_rng(seed)
    sequences: list[list[int]] = []
    payload_rows: list[list[int]] = []
    query_slots: list[int] = []
    targets: list[int] = []
    slots_out: list[int] = []
    lengths_out: list[int] = []
    combo_count = max(1, sum(int(slots) for slots in slot_counts) * max(1, len(distractor_lengths)))
    per_combo = max(1, eval_sequences // combo_count)
    reserved_base = set(int(x) for x in markers) | set(int(x) for x in queries)
    for slot_count in slot_counts:
        for length in distractor_lengths:
            for query_slot in range(int(slot_count)):
                for idx in range(per_combo):
                    start = (idx * int(slot_count) + query_slot) % len(payload_values)
                    payloads = [int(payload_values[(start + offset) % len(payload_values)]) for offset in range(int(slot_count))]
                    if len(set(payloads)) != len(payloads):
                        payloads = rng.choice(payload_values, size=int(slot_count), replace=False).astype(np.int32).tolist()
                    forbidden = reserved_base | set(payloads) if strict_distractors else reserved_base
                    distractor_pool = np.asarray([value for value in range(256) if value not in forbidden], dtype=np.int32)
                    distractors = rng.choice(distractor_pool, size=int(length), replace=True).astype(np.int32).tolist()
                    sequence: list[int] = []
                    for slot_idx, payload in enumerate(payloads):
                        sequence.extend([int(markers[slot_idx]), int(payload)])
                    sequence.extend(int(x) for x in distractors)
                    sequence.append(int(queries[query_slot]))
                    padded = payloads + [-1] * (max_slots - len(payloads))
                    sequences.append(sequence)
                    payload_rows.append(padded)
                    query_slots.append(int(query_slot))
                    targets.append(int(payloads[query_slot]))
                    slots_out.append(int(slot_count))
                    lengths_out.append(int(length))
    return SequenceBatch(
        sequences=sequences,
        payload_matrix=np.asarray(payload_rows, dtype=np.int32),
        query_slots=np.asarray(query_slots, dtype=np.int32),
        target_payloads=np.asarray(targets, dtype=np.int32),
        slot_counts=np.asarray(slots_out, dtype=np.int32),
        distractor_lengths=np.asarray(lengths_out, dtype=np.int32),
        patterns=patterns,
        markers=tuple(int(x) for x in markers),
        queries=tuple(int(x) for x in queries),
    )


def identity_entries(slot_count: int, state_dim: int, edge_budget: int | None = None) -> tuple[tuple[int, int, int, float], ...]:
    entries: list[tuple[int, int, int, float]] = []
    for slot_idx in range(slot_count):
        start = slot_start(slot_idx)
        for bit in range(VISIBLE_DIM):
            entries.append((slot_idx, start + bit, bit, 1.0))
    if edge_budget is not None:
        entries = entries[: int(edge_budget)]
    return dedupe_memory_entries(entries)


def oracle_entries(slot_count: int, state_dim: int, edge_budget: int | None = None) -> tuple[tuple[int, int, int, float], ...]:
    entries: list[tuple[int, int, int, float]] = []
    for slot_idx in range(slot_count):
        start = slot_start(slot_idx)
        for bit in range(VISIBLE_DIM):
            entries.append((slot_idx, start + bit, bit, 1.0))
        for bit in range(VISIBLE_DIM):
            entries.append((slot_idx, start + VISIBLE_DIM + bit, bit, -1.0))
    if edge_budget is not None:
        entries = entries[: int(edge_budget)]
    return dedupe_memory_entries(entries)


def random_entries(*, slot_count: int, state_dim: int, edge_budget: int, rng: random.Random) -> tuple[tuple[int, int, int, float], ...]:
    entries = []
    used: set[tuple[int, int, int]] = set()
    weights = (-1.0, -0.5, 0.5, 1.0)
    tries = 0
    while len(entries) < edge_budget and tries < edge_budget * 200:
        tries += 1
        slot_idx = rng.randrange(slot_count)
        state_idx = rng.randrange(state_dim)
        visible_idx = rng.randrange(VISIBLE_DIM)
        if (slot_idx, state_idx, visible_idx) in used:
            continue
        used.add((slot_idx, state_idx, visible_idx))
        entries.append((slot_idx, state_idx, visible_idx, rng.choice(weights)))
    return dedupe_memory_entries(entries)


def states_from_payloads(core: np.ndarray, batch: SequenceBatch, slot_count: int) -> np.ndarray:
    states = np.zeros((batch.target_payloads.shape[0], core.shape[1]), dtype=np.float32)
    for slot_idx in range(slot_count):
        payloads = batch.payload_matrix[:, slot_idx]
        active = payloads >= 0
        if not np.any(active):
            continue
        states[active] += batch.patterns[payloads[active]] @ core[slot_idx].T
    return states


def eval_query(
    *,
    base_decoded: np.ndarray,
    output_context: np.ndarray,
    targets: np.ndarray,
    patterns: np.ndarray,
) -> dict[str, float | np.ndarray]:
    decoded = base_decoded + output_context @ d21b_context_weights()
    logits = byte_logits(decoded, patterns)
    pred = np.argmax(logits, axis=1)
    pred_bits = np.where(decoded >= 0.0, 1.0, -1.0)
    target_patterns = patterns[targets]
    margins = target_margins(logits, targets)
    return {
        "exact_acc": float(np.mean(pred == targets)),
        "bit_acc": float(np.mean(pred_bits == target_patterns)),
        "margin_mean": float(np.mean(margins)),
        "margin_min": float(np.min(margins)),
        "pred": pred,
    }


def run_sequences(
    *,
    core: np.ndarray,
    spec: MemorySpec,
    batch: SequenceBatch,
    mode: str,
    seed: int,
) -> dict[str, float | np.ndarray]:
    block = byte_block()
    patterns = batch.patterns
    rng = np.random.default_rng(seed)
    states = states_from_payloads(core, batch, spec.slot_count)
    query_slots = batch.query_slots.copy()
    if mode == "reset" or mode == "marker_shuffle":
        output_context = np.zeros((batch.target_payloads.shape[0], CONTEXT_DIM), dtype=np.float32)
    elif mode == "time_shuffle":
        output_context = slot_query_context(np.roll(states, 17, axis=0), query_slots, spec.state_dim)
    elif mode == "random":
        random_state = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(batch.target_payloads.shape[0], spec.state_dim))
        output_context = slot_query_context(random_state, query_slots, spec.state_dim)
    elif mode == "query_shuffle":
        shifted_slots = (query_slots + 1) % batch.slot_counts
        output_context = slot_query_context(states, shifted_slots, spec.state_dim)
    else:
        output_context = slot_query_context(states, query_slots, spec.state_dim)

    query_tokens = np.asarray([batch.queries[int(slot)] for slot in query_slots], dtype=np.int32)
    base_decoded = block.encode_patterns(patterns[query_tokens]) @ block.encoder
    query = eval_query(base_decoded=base_decoded, output_context=output_context, targets=batch.target_payloads, patterns=patterns)

    non_query_targets = [byte for sequence in batch.sequences for byte in sequence[:-1]]
    non_query_targets_arr = np.asarray(non_query_targets, dtype=np.int32)
    non_query_decoded = block.encode_patterns(patterns[non_query_targets_arr]) @ block.encoder
    non_query_logits = byte_logits(non_query_decoded, patterns)
    non_query_pred = np.argmax(non_query_logits, axis=1)

    wrong_payloads = np.asarray(
        [batch.payload_matrix[row_idx, (int(slot) + 1) % int(slot_count)] for row_idx, (slot, slot_count) in enumerate(zip(batch.query_slots, batch.slot_counts))],
        dtype=np.int32,
    )
    pred = np.asarray(query["pred"], dtype=np.int32)
    context_rows = []
    collision_count = 0
    for slot_idx in range(spec.slot_count):
        active = batch.payload_matrix[:, slot_idx] >= 0
        if not np.any(active):
            continue
        unique_payloads, unique_indices = np.unique(batch.payload_matrix[active, slot_idx], return_index=True)
        active_indices = np.where(active)[0][unique_indices]
        slot_context = slot_query_context(states[active_indices], np.full(active_indices.shape[0], slot_idx, dtype=np.int32), spec.state_dim)
        rows = [tuple(np.round(row, 6).tolist()) for row in slot_context]
        collision_count += int(len(unique_payloads) - len(set(rows)))
        context_rows.extend(rows)

    return {
        "query_payload_exact_acc": float(query["exact_acc"]),
        "query_payload_bit_acc": float(query["bit_acc"]),
        "query_payload_margin_mean": float(query["margin_mean"]),
        "query_payload_margin_min": float(query["margin_min"]),
        "non_query_byte_reconstruction_acc": float(np.mean(non_query_pred == non_query_targets_arr)),
        "current_byte_cheat_rate": float(np.mean(pred == query_tokens)),
        "wrong_slot_recall_rate": float(np.mean(pred == wrong_payloads)),
        "slot_state_collision_count": int(collision_count),
        "query_pred": pred,
    }


def prev_byte_baseline(batch: SequenceBatch) -> float:
    correct = 0
    for sequence, target in zip(batch.sequences, batch.target_payloads):
        previous = sequence[-2] if len(sequence) >= 2 else -1
        if int(previous) == int(target):
            correct += 1
    return float(correct / max(1, len(batch.sequences)))


def query_removed_false_positive_rate(batch: SequenceBatch) -> float:
    block = byte_block()
    patterns = batch.patterns
    tokens = []
    targets = []
    for sequence, target, slot_count in zip(batch.sequences, batch.target_payloads, batch.slot_counts):
        distractor_start = 2 * int(slot_count)
        for byte_value in sequence[distractor_start:-1]:
            tokens.append(int(byte_value))
            targets.append(int(target))
    if not tokens:
        return 0.0
    decoded = block.encode_patterns(patterns[np.asarray(tokens, dtype=np.int32)]) @ block.encoder
    logits = byte_logits(decoded, patterns)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == np.asarray(targets, dtype=np.int32)))


def evaluate_memory_spec(
    spec: MemorySpec,
    *,
    batch: SequenceBatch,
    slot_counts: Sequence[int],
    distractor_lengths: Sequence[int],
    seed: int,
) -> dict[str, object]:
    core = memory_tensor(spec.slot_count, spec.state_dim, spec.visible_dim, spec.memory_entries)
    real = run_sequences(core=core, spec=spec, batch=batch, mode="real", seed=seed + spec.candidate_id)
    reset = run_sequences(core=core, spec=spec, batch=batch, mode="reset", seed=seed + 1000 + spec.candidate_id)
    time_shuffle = run_sequences(core=core, spec=spec, batch=batch, mode="time_shuffle", seed=seed + 2000 + spec.candidate_id)
    random_state = run_sequences(core=core, spec=spec, batch=batch, mode="random", seed=seed + 3000 + spec.candidate_id)
    marker_shuffle = run_sequences(core=core, spec=spec, batch=batch, mode="marker_shuffle", seed=seed + 4000 + spec.candidate_id)
    query_shuffle = run_sequences(core=core, spec=spec, batch=batch, mode="query_shuffle", seed=seed + 5000 + spec.candidate_id)

    pred = np.asarray(real["query_pred"], dtype=np.int32)
    slot_passes = []
    slot_metrics: dict[str, float] = {}
    for slot_count in slot_counts:
        mask = batch.slot_counts == int(slot_count)
        acc = float(np.mean(pred[mask] == batch.target_payloads[mask])) if np.any(mask) else 0.0
        slot_metrics[f"slot_count_{slot_count}_exact_acc"] = acc
        slot_passes.append(acc == 1.0)
    length_passes = []
    length_metrics: dict[str, float] = {}
    for length in distractor_lengths:
        mask = batch.distractor_lengths == int(length)
        acc = float(np.mean(pred[mask] == batch.target_payloads[mask])) if np.any(mask) else 0.0
        length_metrics[f"distractor_len_{length}_exact_acc"] = acc
        length_passes.append(acc == 1.0)

    block = byte_block()
    base = evaluate_ablock(block, batch.patterns)
    metrics: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "visible_dim": spec.visible_dim,
        "code_dim": spec.code_dim,
        "state_dim": spec.state_dim,
        "context_dim": spec.context_dim,
        "slot_count": spec.slot_count,
        "family": spec.family,
        "memory_edge_budget": spec.memory_edge_budget,
        "memory_edge_count": int(np.count_nonzero(core)),
        "memory_entries": entries_to_memory_string(spec.memory_entries),
        "query_payload_exact_acc": float(real["query_payload_exact_acc"]),
        "query_payload_bit_acc": float(real["query_payload_bit_acc"]),
        "query_payload_margin_mean": float(real["query_payload_margin_mean"]),
        "query_payload_margin_min": float(real["query_payload_margin_min"]),
        "all_slot_counts_pass": bool(all(slot_passes)),
        "all_distractor_lengths_pass": bool(all(length_passes)),
        "long_sequence_payload_acc": float(real["query_payload_exact_acc"]),
        "slot_state_collision_count": int(real["slot_state_collision_count"]),
        "non_query_byte_reconstruction_acc": float(real["non_query_byte_reconstruction_acc"]),
        "zero_context_byte_reconstruction_acc": float(base["exact_byte_acc"]),
        "reset_state_acc": float(reset["query_payload_exact_acc"]),
        "time_shuffle_state_acc": float(time_shuffle["query_payload_exact_acc"]),
        "random_state_acc": float(random_state["query_payload_exact_acc"]),
        "marker_shuffle_acc": float(marker_shuffle["query_payload_exact_acc"]),
        "query_shuffle_acc": float(query_shuffle["query_payload_exact_acc"]),
        "query_removed_false_positive_rate": query_removed_false_positive_rate(batch),
        "prev_byte_baseline_acc": prev_byte_baseline(batch),
        "current_byte_cheat_rate": float(real["current_byte_cheat_rate"]),
        "wrong_slot_recall_rate": float(real["wrong_slot_recall_rate"]),
    }
    metrics.update(slot_metrics)
    metrics.update(length_metrics)
    metrics["D21E_score"] = d21e_score(metrics)
    metrics["verdict"] = d21e_verdict(metrics)
    return metrics


def d21e_score(metrics: dict[str, object]) -> float:
    control_max = max(
        float(metrics["reset_state_acc"]),
        float(metrics["time_shuffle_state_acc"]),
        float(metrics["random_state_acc"]),
        float(metrics["marker_shuffle_acc"]),
        float(metrics["query_shuffle_acc"]),
        float(metrics["prev_byte_baseline_acc"]),
    )
    return (
        4.0 * float(metrics["query_payload_exact_acc"])
        + 1.0 * float(metrics["query_payload_bit_acc"])
        + 1.0 * float(metrics["long_sequence_payload_acc"])
        + 0.02 * float(metrics["query_payload_margin_min"])
        + 0.5 * float(metrics["non_query_byte_reconstruction_acc"])
        - 2.0 * control_max
        - 1.5 * float(metrics["wrong_slot_recall_rate"])
        - 1.0 * float(metrics["current_byte_cheat_rate"])
        - 0.5 * float(metrics["query_removed_false_positive_rate"])
        - 0.002 * float(metrics["slot_state_collision_count"])
        - 0.001 * float(metrics["memory_edge_count"])
    )


def d21e_verdict(metrics: dict[str, object]) -> str:
    controls_clean = (
        float(metrics["reset_state_acc"]) <= 0.01
        and float(metrics["time_shuffle_state_acc"]) <= 0.01
        and float(metrics["random_state_acc"]) <= 0.01
        and float(metrics["marker_shuffle_acc"]) <= 0.01
        and float(metrics["query_shuffle_acc"]) <= 0.01
        and float(metrics["query_removed_false_positive_rate"]) <= 0.01
        and float(metrics["prev_byte_baseline_acc"]) <= 0.01
        and float(metrics["current_byte_cheat_rate"]) <= 0.01
        and float(metrics["wrong_slot_recall_rate"]) <= 0.01
    )
    if float(metrics["zero_context_byte_reconstruction_acc"]) != 1.0 or float(metrics["non_query_byte_reconstruction_acc"]) != 1.0:
        return "D21E_BYTE_GATE_BROKEN"
    if int(metrics["slot_state_collision_count"]) > 0 and float(metrics["query_payload_exact_acc"]) >= 0.99:
        return "D21E_SLOT_COLLISION_FAIL"
    if "slot_count_4_exact_acc" in metrics and float(metrics.get("slot_count_2_exact_acc", 0.0)) == 1.0 and float(metrics.get("slot_count_4_exact_acc", 0.0)) < 1.0 and controls_clean:
        return "D21E_2SLOT_ONLY"
    if float(metrics["query_payload_exact_acc"]) == 1.0 and bool(metrics["all_slot_counts_pass"]) and bool(metrics["all_distractor_lengths_pass"]) and controls_clean:
        return "D21E_MULTISLOT_MEMORY_PASS"
    if max(float(metrics["reset_state_acc"]), float(metrics["time_shuffle_state_acc"]), float(metrics["random_state_acc"]), float(metrics["marker_shuffle_acc"]), float(metrics["query_shuffle_acc"])) > 0.01:
        return "D21E_STATE_ARTIFACT"
    if float(metrics["query_payload_exact_acc"]) > 0.99 and controls_clean:
        return "D21E_WEAK_PASS"
    return "D21E_NO_MULTISLOT_ROUTE"


def build_specs(*, state_dims: Sequence[int], slot_counts: Sequence[int], edge_budgets: Sequence[int], samples: int, seed: int) -> list[MemorySpec]:
    rng = random.Random(seed)
    specs: list[MemorySpec] = []

    def add_spec(state_dim: int, slot_count: int, edge_budget: int, family: str, entries: Iterable[tuple[int, int, int, float]]) -> None:
        cleaned = []
        for slot_idx, state_idx, visible_idx, value in entries:
            if 0 <= slot_idx < slot_count and 0 <= state_idx < state_dim and 0 <= visible_idx < VISIBLE_DIM and abs(float(value)) > 1e-12:
                cleaned.append((slot_idx, state_idx, visible_idx, value))
        specs.append(
            MemorySpec(
                candidate_id=len(specs),
                visible_dim=VISIBLE_DIM,
                code_dim=CODE_DIM,
                state_dim=state_dim,
                context_dim=CONTEXT_DIM,
                slot_count=slot_count,
                family=family,
                memory_edge_budget=edge_budget,
                memory_entries=dedupe_memory_entries(cleaned),
            )
        )

    for state_dim in state_dims:
        for slot_count in slot_counts:
            for edge_budget in edge_budgets:
                add_spec(state_dim, slot_count, edge_budget, "zero_memory", ())
                add_spec(state_dim, slot_count, edge_budget, "identity_multislot_memory", identity_entries(slot_count, state_dim, edge_budget))
                add_spec(state_dim, slot_count, edge_budget, "oracle_multislot_memory", oracle_entries(slot_count, state_dim, edge_budget))
                for _ in range(samples):
                    add_spec(
                        state_dim,
                        slot_count,
                        edge_budget,
                        "random_sparse_memory",
                        random_entries(slot_count=slot_count, state_dim=state_dim, edge_budget=edge_budget, rng=rng),
                    )
    return specs


def group_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["state_dim"]), int(row["slot_count"]), int(row["memory_edge_budget"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, state_dim, slot_count, edge_budget), items in sorted(groups.items()):
        best = max(items, key=lambda row: float(row["D21E_score"]))
        summary.append(
            {
                "family": family,
                "state_dim": state_dim,
                "slot_count": slot_count,
                "memory_edge_budget": edge_budget,
                "count": len(items),
                "pass_count": sum(1 for row in items if str(row["verdict"]) == "D21E_MULTISLOT_MEMORY_PASS"),
                "two_slot_count": sum(1 for row in items if str(row["verdict"]) == "D21E_2SLOT_ONLY"),
                "weak_count": sum(1 for row in items if str(row["verdict"]) == "D21E_WEAK_PASS"),
                "best_score": float(best["D21E_score"]),
                "best_verdict": str(best["verdict"]),
                "best_query_payload_exact_acc": float(best["query_payload_exact_acc"]),
                "best_control_max": max(
                    float(best["reset_state_acc"]),
                    float(best["time_shuffle_state_acc"]),
                    float(best["random_state_acc"]),
                    float(best["marker_shuffle_acc"]),
                    float(best["query_shuffle_acc"]),
                ),
                "best_memory_edge_count": int(best["memory_edge_count"]),
            }
        )
    return summary


def make_heatmap(summary: Sequence[dict[str, object]]) -> str:
    state_dims = sorted({int(row["state_dim"]) for row in summary})
    budgets = sorted({int(row["memory_edge_budget"]) for row in summary})
    values: dict[tuple[int, int], float] = {}
    verdicts: dict[tuple[int, int], str] = {}
    for row in summary:
        key = (int(row["state_dim"]), int(row["memory_edge_budget"]))
        value = float(row["best_query_payload_exact_acc"]) * (1.0 - min(1.0, float(row["best_control_max"])))
        if key not in values or value > values[key]:
            values[key] = value
            verdicts[key] = str(row["best_verdict"])
    all_values = list(values.values()) or [0.0]
    lo = min(all_values)
    hi = max(all_values)
    lines = ["D21E memory heatmap: brighter = query_exact * (1-control_max)"]
    lines.append("legend: PASS=P 2SLOT=2 WEAK=W ARTIFACT=F NONE=.")
    lines.append("state\\edge " + " ".join(f"{budget:>5}" for budget in budgets))
    for state_dim in state_dims:
        cells = []
        for budget in budgets:
            value = values.get((state_dim, budget), 0.0)
            scaled = 0 if hi <= lo else int(round((value - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
            scaled = max(0, min(len(ASCII_SHADE) - 1, scaled))
            verdict = verdicts.get((state_dim, budget), ".")
            marker = (
                "P"
                if verdict == "D21E_MULTISLOT_MEMORY_PASS"
                else "2"
                if verdict == "D21E_2SLOT_ONLY"
                else "W"
                if verdict == "D21E_WEAK_PASS"
                else "F"
                if verdict == "D21E_STATE_ARTIFACT"
                else "."
            )
            cells.append(f"{ASCII_SHADE[scaled]}{marker:>4}")
        lines.append(f"{state_dim:>10} " + " ".join(cells))
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    *,
    out_dir: Path,
    rows: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    heatmap: str,
    mode: str,
    config: dict[str, object],
    path_rows: Sequence[dict[str, object]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["D21E_score"]), reverse=True)
    write_csv(out_dir / "memory_candidates.csv", sorted_rows)
    write_csv(out_dir / "memory_family_summary.csv", summary)
    if path_rows is not None:
        write_csv(out_dir / "memory_paths.csv", path_rows)
    control_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "state_dim": row["state_dim"],
            "slot_count": row["slot_count"],
            "memory_edge_count": row["memory_edge_count"],
            "query_payload_exact_acc": row["query_payload_exact_acc"],
            "reset_state_acc": row["reset_state_acc"],
            "time_shuffle_state_acc": row["time_shuffle_state_acc"],
            "random_state_acc": row["random_state_acc"],
            "marker_shuffle_acc": row["marker_shuffle_acc"],
            "query_shuffle_acc": row["query_shuffle_acc"],
            "wrong_slot_recall_rate": row["wrong_slot_recall_rate"],
            "prev_byte_baseline_acc": row["prev_byte_baseline_acc"],
            "current_byte_cheat_rate": row["current_byte_cheat_rate"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows[: min(256, len(sorted_rows))]
    ]
    write_csv(out_dir / "memory_control_summary.csv", control_rows)
    (out_dir / "memory_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    pass_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21E_MULTISLOT_MEMORY_PASS"]
    two_slot_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21E_2SLOT_ONLY"]
    weak_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21E_WEAK_PASS"]
    verdict = (
        "D21E_MULTISLOT_MEMORY_PASS"
        if pass_rows
        else "D21E_2SLOT_ONLY"
        if two_slot_rows
        else "D21E_WEAK_PASS"
        if weak_rows
        else "D21E_NO_MULTISLOT_ROUTE"
    )
    payload = {
        "verdict": verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "pass_count": len(pass_rows),
        "two_slot_count": len(two_slot_rows),
        "weak_count": len(weak_rows),
        "best_candidate": sorted_rows[0] if sorted_rows else None,
        "best_pass_candidate": pass_rows[0] if pass_rows else None,
        "best_two_slot_candidate": two_slot_rows[0] if two_slot_rows else None,
        "best_weak_candidate": weak_rows[0] if weak_rows else None,
    }
    (out_dir / "memory_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# D21E Multi-Slot Memory A-Block Core Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{verdict}`",
        "",
        "## Best Candidate",
        "",
    ]
    if sorted_rows:
        best = sorted_rows[0]
        report.extend(
            [
                f"- family: `{best['family']}`",
                f"- state_dim: `{best['state_dim']}`",
                f"- slot_count: `{best['slot_count']}`",
                f"- memory_edges: `{best['memory_edge_count']}`",
                f"- query_payload_exact_acc: `{float(best['query_payload_exact_acc']):.6f}`",
                f"- query_payload_margin_min: `{float(best['query_payload_margin_min']):.6f}`",
                f"- non_query_byte_reconstruction_acc: `{float(best['non_query_byte_reconstruction_acc']):.6f}`",
                f"- reset_state_acc: `{float(best['reset_state_acc']):.6f}`",
                f"- time_shuffle_state_acc: `{float(best['time_shuffle_state_acc']):.6f}`",
                f"- random_state_acc: `{float(best['random_state_acc']):.6f}`",
                f"- marker_shuffle_acc: `{float(best['marker_shuffle_acc']):.6f}`",
                f"- query_shuffle_acc: `{float(best['query_shuffle_acc']):.6f}`",
                f"- wrong_slot_recall_rate: `{float(best['wrong_slot_recall_rate']):.6f}`",
                f"- verdict: `{best['verdict']}`",
            ]
        )
    report.extend(["", "## Heatmap", "", "```text", heatmap, "```", ""])
    (out_dir / "D21E_MULTISLOT_MEMORY_ABLOCK_CORE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_self_checks() -> None:
    patterns = all_visible_patterns(VISIBLE_DIM)
    block = byte_block()
    base = evaluate_ablock(block, patterns)
    robust = robustness_metrics(block, patterns)
    assert float(base["exact_byte_acc"]) == 1.0
    assert float(robust["single_edge_drop_mean_bit"]) == 1.0
    batch = make_sequence_batch(
        slot_counts=[2],
        distractor_lengths=[1, 2],
        eval_sequences=4096,
        markers=DEFAULT_MARKERS,
        queries=DEFAULT_QUERIES,
        strict_distractors=True,
        seed=DEFAULT_SEED,
    )
    spec = MemorySpec(0, VISIBLE_DIM, CODE_DIM, 32, CONTEXT_DIM, 2, "oracle_multislot_memory", 32, oracle_entries(2, 32, 32))
    row = evaluate_memory_spec(spec, batch=batch, slot_counts=[2], distractor_lengths=[1, 2], seed=DEFAULT_SEED)
    assert str(row["verdict"]) == "D21E_MULTISLOT_MEMORY_PASS", row


def evaluate_specs(args: argparse.Namespace, specs: Sequence[MemorySpec]) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    slot_counts = parse_int_list(args.slot_counts)
    distractor_lengths = parse_int_list(args.distractor_lengths)
    markers = parse_hex_list(args.markers)
    queries = parse_hex_list(args.queries)
    batch = make_sequence_batch(
        slot_counts=slot_counts,
        distractor_lengths=distractor_lengths,
        eval_sequences=int(args.eval_sequences),
        markers=markers,
        queries=queries,
        strict_distractors=not bool(args.loose_distractors),
        seed=int(args.seed),
    )
    rows = []
    for idx, spec in enumerate(specs, start=1):
        rows.append(evaluate_memory_spec(spec, batch=batch, slot_counts=slot_counts, distractor_lengths=distractor_lengths, seed=int(args.seed)))
        if idx % 500 == 0:
            best = max(rows, key=lambda row: float(row["D21E_score"]))
            print(
                f"[D21e] evaluated {idx}/{len(specs)} "
                f"best={best['family']} state={best['state_dim']} slots={best['slot_count']} "
                f"query={float(best['query_payload_exact_acc']):.4f} verdict={best['verdict']}",
                flush=True,
            )
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    return rows, summary, heatmap


def run_baseline_check(args: argparse.Namespace) -> int:
    run_self_checks()
    patterns = all_visible_patterns(VISIBLE_DIM)
    block = byte_block()
    base = {**evaluate_ablock(block, patterns), **robustness_metrics(block, patterns)}
    row = {
        "mode": "baseline-check",
        "zero_exact_byte_acc": float(base["exact_byte_acc"]),
        "zero_bit_acc": float(base["bit_acc"]),
        "single_edge_drop_mean_bit": float(base["single_edge_drop_mean_bit"]),
        "baseline_reproduced": True,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "baseline_check.csv", [row])
    (out_dir / "memory_top.json").write_text(json.dumps({"verdict": "D21E_BASELINE_REPRODUCED", "baseline": row}, indent=2), encoding="utf-8")
    (out_dir / "D21E_MULTISLOT_MEMORY_ABLOCK_CORE_REPORT.md").write_text("# D21E Baseline Check\n\nVerdict: `D21E_BASELINE_REPRODUCED`\n", encoding="utf-8")
    print("[D21e] baseline reproduced D21A/D21B fixed shell")
    return 0


def run_multislot_oracle(args: argparse.Namespace) -> int:
    run_self_checks()
    max_slots = max(parse_int_list(args.slot_counts))
    entries = oracle_entries(max_slots, int(args.state_dim), int(args.memory_edge_budget))
    spec = MemorySpec(0, VISIBLE_DIM, CODE_DIM, int(args.state_dim), CONTEXT_DIM, max_slots, "oracle_multislot_memory", int(args.memory_edge_budget), entries)
    rows, summary, heatmap = evaluate_specs(args, [spec])
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="multislot-oracle",
        config={
            "state_dim": int(args.state_dim),
            "slot_counts": parse_int_list(args.slot_counts),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    best = rows[0]
    print(
        "[D21e] oracle "
        f"query={float(best['query_payload_exact_acc']):.4f} "
        f"query_shuffle={float(best['query_shuffle_acc']):.4f} "
        f"wrong_slot={float(best['wrong_slot_recall_rate']):.4f} "
        f"verdict={best['verdict']}",
        flush=True,
    )
    return 0


def run_memory_atlas(args: argparse.Namespace) -> int:
    run_self_checks()
    specs = build_specs(
        state_dims=parse_int_list(args.state_dims),
        slot_counts=parse_int_list(args.slot_counts),
        edge_budgets=parse_int_list(args.memory_edge_budgets),
        samples=int(args.samples),
        seed=int(args.seed),
    )
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="memory-atlas",
        config={
            "state_dims": parse_int_list(args.state_dims),
            "slot_counts": parse_int_list(args.slot_counts),
            "memory_edge_budgets": parse_int_list(args.memory_edge_budgets),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "samples": int(args.samples),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    print(heatmap)
    best = max(rows, key=lambda row: float(row["D21E_score"]))
    print(
        "[D21e] best "
        f"family={best['family']} state={best['state_dim']} slots={best['slot_count']} E={best['memory_edge_count']} "
        f"query={float(best['query_payload_exact_acc']):.4f} verdict={best['verdict']}",
        flush=True,
    )
    return 0


def start_entries(start_family: str, slot_count: int, state_dim: int, edge_budget: int, seed: int) -> tuple[tuple[int, int, int, float], ...]:
    if start_family == "oracle_multislot_memory":
        return oracle_entries(slot_count, state_dim, edge_budget)
    if start_family == "identity_multislot_memory":
        return identity_entries(slot_count, state_dim, edge_budget)
    if start_family == "random_sparse_memory":
        return random_entries(slot_count=slot_count, state_dim=state_dim, edge_budget=edge_budget, rng=random.Random(seed))
    raise ValueError(f"unknown start family: {start_family}")


def run_crystallize_memory(args: argparse.Namespace) -> int:
    run_self_checks()
    slot_count = max(parse_int_list(args.slot_counts))
    state_dim = int(args.state_dim)
    edge_budget = int(args.memory_edge_budget)
    current_entries = start_entries(str(args.start_family), slot_count, state_dim, edge_budget, int(args.seed))

    def eval_one(candidate_id: int, family: str, entries: tuple[tuple[int, int, int, float]]) -> dict[str, object]:
        rows, _summary, _heat = evaluate_specs(
            args,
            [MemorySpec(candidate_id, VISIBLE_DIM, CODE_DIM, state_dim, CONTEXT_DIM, slot_count, family, edge_budget, entries)],
        )
        return rows[0]

    current = eval_one(0, f"start_{args.start_family}", current_entries)
    best_rows = [current]
    path_rows = [{**current, "step": 0, "accepted": True, "reason": "start"}]
    print(
        f"[D21e] crystallize start score={float(current['D21E_score']):.6f} "
        f"edges={current['memory_edge_count']} query={float(current['query_payload_exact_acc']):.4f} "
        f"verdict={current['verdict']}",
        flush=True,
    )

    for step in range(1, int(args.max_steps) + 1):
        proposals = []
        parsed = list(current_entries)
        for drop_idx in range(len(parsed)):
            proposals.append(eval_one(step * 1000 + len(proposals), "drop_edge", dedupe_memory_entries(parsed[:drop_idx] + parsed[drop_idx + 1 :])))
        for weight_idx, (slot_idx, state_idx, visible_idx, old_value) in enumerate(parsed):
            for new_value in (-1.0, -0.5, 0.5, 1.0):
                if abs(new_value - old_value) < 1e-12:
                    continue
                entries = parsed.copy()
                entries[weight_idx] = (slot_idx, state_idx, visible_idx, new_value)
                proposals.append(eval_one(step * 1000 + len(proposals), "reweight", dedupe_memory_entries(entries)))

        best_rows.extend(proposals)
        pass_props = [row for row in proposals if str(row["verdict"]) == "D21E_MULTISLOT_MEMORY_PASS"]
        accepted = False
        reason = "reject"
        if pass_props:
            best_prop = max(
                pass_props,
                key=lambda row: (
                    -int(row["memory_edge_count"]),
                    float(row["D21E_score"]),
                    float(row["query_payload_margin_min"]),
                ),
            )
            if int(best_prop["memory_edge_count"]) < int(current["memory_edge_count"]) or float(best_prop["D21E_score"]) > float(current["D21E_score"]) + 1e-9:
                current = best_prop
                current_entries = parse_entries(str(current["memory_entries"]))
                accepted = True
                reason = "gate_preserving_simplify"
        path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
        print(
            f"[D21e][crystallize {step}] score={float(current['D21E_score']):.6f} "
            f"edges={current['memory_edge_count']} query={float(current['query_payload_exact_acc']):.4f} "
            f"{current['verdict']} {reason}",
            flush=True,
        )
        if not accepted:
            break

    rows = sorted(best_rows, key=lambda row: float(row["D21E_score"]), reverse=True)
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="crystallize-memory",
        config={
            "state_dim": state_dim,
            "slot_count": slot_count,
            "start_family": str(args.start_family),
            "max_steps": int(args.max_steps),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
        path_rows=path_rows,
    )
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    run_self_checks()
    source = Path(args.candidates)
    if not source.exists():
        raise FileNotFoundError(source)
    rows_in = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_in.append(row)
    rows_in = sorted(rows_in, key=lambda row: float(row.get("D21E_score", row.get("D21D_score", 0.0))), reverse=True)[: int(args.top_k)]
    specs = []
    for row in rows_in:
        slot_count = int(row.get("slot_count", max(parse_int_list(args.slot_counts))))
        state_dim = int(row["state_dim"])
        entries = parse_entries(str(row["memory_entries"]))
        specs.append(
            MemorySpec(
                len(specs),
                VISIBLE_DIM,
                CODE_DIM,
                state_dim,
                CONTEXT_DIM,
                slot_count,
                f"confirm_{row['family']}",
                int(row.get("memory_edge_budget", row.get("memory_edge_count", 0))),
                entries,
            )
        )
    confirm_rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=confirm_rows,
        summary=summary,
        heatmap=heatmap,
        mode="confirm",
        config={
            "source": str(source),
            "top_k": int(args.top_k),
            "slot_counts": parse_int_list(args.slot_counts),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    best = max(confirm_rows, key=lambda row: float(row["D21E_score"])) if confirm_rows else None
    if best:
        print(
            "[D21e] confirm best "
            f"family={best['family']} state={best['state_dim']} slots={best['slot_count']} "
            f"edges={best['memory_edge_count']} query={float(best['query_payload_exact_acc']):.4f} "
            f"verdict={best['verdict']}",
            flush=True,
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline-check", "multislot-oracle", "memory-atlas", "crystallize-memory", "confirm"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--state-dims", default="32,64,128")
    parser.add_argument("--slot-counts", default="2,4")
    parser.add_argument("--memory-edge-budget", type=int, default=64)
    parser.add_argument("--memory-edge-budgets", default="16,32,64,96")
    parser.add_argument("--distractor-lengths", default="1,2,4,8,16")
    parser.add_argument("--eval-sequences", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--start-family", default="oracle_multislot_memory")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--markers", default="0xF1,0xF2,0xF3,0xF4")
    parser.add_argument("--queries", default="0x1F,0x2F,0x3F,0x4F")
    parser.add_argument("--loose-distractors", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode == "baseline-check":
        return run_baseline_check(args)
    if args.mode == "multislot-oracle":
        return run_multislot_oracle(args)
    if args.mode == "memory-atlas":
        return run_memory_atlas(args)
    if args.mode == "crystallize-memory":
        return run_crystallize_memory(args)
    if args.mode == "confirm":
        return run_confirm(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
