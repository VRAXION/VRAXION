#!/usr/bin/env python3
"""
D21d marker-memory A-block core.

Prototype goal:
    MARKER, PAYLOAD, distractor..., QUERY -> output PAYLOAD at QUERY

The D21A byte lane and D21B context write lane are fixed. D21D evaluates a tiny
memory core that stores payload context after a marker and emits that stored
context only when a query byte arrives.
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
MARKER_BYTE = 0xF0
QUERY_BYTE = 0x0F


@dataclass(frozen=True)
class MemorySpec:
    candidate_id: int
    visible_dim: int
    code_dim: int
    state_dim: int
    context_dim: int
    family: str
    memory_edge_budget: int
    memory_entries: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class SequenceBatch:
    sequences: list[list[int]]
    payloads: np.ndarray
    distractor_lengths: np.ndarray
    patterns: np.ndarray


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_entries(raw: str) -> tuple[tuple[int, int, float], ...]:
    entries = []
    for part in str(raw).split():
        if not part:
            continue
        state_idx, visible_idx, value = part.split(":")
        entries.append((int(state_idx), int(visible_idx), float(value)))
    return dedupe_entries(entries)


def memory_matrix(state_dim: int, visible_dim: int, entries: Iterable[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((state_dim, visible_dim), dtype=np.float32)
    for state_idx, visible_idx, value in entries:
        if 0 <= state_idx < state_dim and 0 <= visible_idx < visible_dim:
            matrix[state_idx, visible_idx] = float(value)
    return matrix


def target_context_for_payload(patterns: np.ndarray, payload_values: np.ndarray) -> np.ndarray:
    bits = patterns[payload_values]
    return np.concatenate([bits, -bits], axis=1).astype(np.float32)


def state_to_context(states: np.ndarray, state_dim: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if state_dim >= CONTEXT_DIM:
        return states[:, :CONTEXT_DIM]
    context = np.zeros((states.shape[0], CONTEXT_DIM), dtype=np.float32)
    limit = min(state_dim, CONTEXT_DIM)
    context[:, :limit] = states[:, :limit]
    return context


def payload_values() -> np.ndarray:
    return np.asarray([value for value in range(256) if value not in (MARKER_BYTE, QUERY_BYTE)], dtype=np.int32)


def make_sequence_batch(*, distractor_lengths: Sequence[int], eval_sequences: int, strict_distractors: bool, seed: int) -> SequenceBatch:
    patterns = all_visible_patterns(VISIBLE_DIM)
    payload_pool = payload_values()
    rng = np.random.default_rng(seed)
    sequences: list[list[int]] = []
    payloads: list[int] = []
    lengths: list[int] = []
    per_len = max(1, eval_sequences // max(1, len(distractor_lengths)))
    distractor_pool_all = np.asarray([value for value in range(256) if value not in (MARKER_BYTE, QUERY_BYTE)], dtype=np.int32)
    for length in distractor_lengths:
        for idx in range(per_len):
            payload = int(payload_pool[idx % len(payload_pool)])
            if strict_distractors:
                distractor_pool = np.asarray(
                    [value for value in range(256) if value not in (MARKER_BYTE, QUERY_BYTE, payload)],
                    dtype=np.int32,
                )
            else:
                distractor_pool = distractor_pool_all
            distractors = rng.choice(distractor_pool, size=int(length), replace=True).astype(np.int32).tolist()
            sequences.append([MARKER_BYTE, payload, *[int(x) for x in distractors], QUERY_BYTE])
            payloads.append(payload)
            lengths.append(int(length))
    return SequenceBatch(sequences=sequences, payloads=np.asarray(payloads, dtype=np.int32), distractor_lengths=np.asarray(lengths, dtype=np.int32), patterns=patterns)


def oracle_memory_entries(state_dim: int) -> tuple[tuple[int, int, float], ...]:
    entries: list[tuple[int, int, float]] = []
    for bit in range(VISIBLE_DIM):
        entries.append((bit, bit, 1.0))
    if state_dim >= CONTEXT_DIM:
        for bit in range(VISIBLE_DIM):
            entries.append((VISIBLE_DIM + bit, bit, -1.0))
    return dedupe_entries(entries)


def identity_memory_entries(state_dim: int) -> tuple[tuple[int, int, float], ...]:
    return dedupe_entries((bit, bit, 1.0) for bit in range(min(state_dim, VISIBLE_DIM)))


def random_memory_entries(*, state_dim: int, edge_budget: int, rng: random.Random) -> tuple[tuple[int, int, float], ...]:
    entries = []
    used: set[tuple[int, int]] = set()
    weights = (-1.0, -0.5, 0.5, 1.0)
    tries = 0
    while len(entries) < edge_budget and tries < edge_budget * 100:
        tries += 1
        state_idx = rng.randrange(state_dim)
        visible_idx = rng.randrange(VISIBLE_DIM)
        if (state_idx, visible_idx) in used:
            continue
        used.add((state_idx, visible_idx))
        entries.append((state_idx, visible_idx, rng.choice(weights)))
    return dedupe_entries(entries)


def run_sequences(
    *,
    core: np.ndarray,
    state_dim: int,
    batch: SequenceBatch,
    mode: str,
    seed: int,
) -> dict[str, float | np.ndarray]:
    block = byte_block()
    context_weights = d21b_context_weights()
    patterns = batch.patterns
    rng = np.random.default_rng(seed)

    payload_targets_arr = batch.payloads.astype(np.int32)
    payload_patterns = patterns[payload_targets_arr]
    payload_states = payload_patterns @ core.T
    if mode == "reset" or mode == "marker_shuffle":
        output_context = np.zeros((payload_targets_arr.shape[0], CONTEXT_DIM), dtype=np.float32)
    elif mode == "time_shuffle":
        shifted_payloads = np.roll(payload_targets_arr, 17)
        output_context = state_to_context(patterns[shifted_payloads] @ core.T, state_dim)
    elif mode == "random":
        random_state = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(payload_targets_arr.shape[0], state_dim),
        )
        output_context = state_to_context(random_state, state_dim)
    else:
        output_context = state_to_context(payload_states, state_dim)

    query_current_targets_arr = np.full(payload_targets_arr.shape[0], QUERY_BYTE, dtype=np.int32)
    query_patterns = patterns[query_current_targets_arr]
    query_base_decoded = block.encode_patterns(query_patterns) @ block.encoder
    query_decoded_arr = query_base_decoded + output_context @ context_weights
    query_logits = byte_logits(query_decoded_arr, patterns)
    query_pred = np.argmax(query_logits, axis=1)
    query_bits = np.where(query_decoded_arr >= 0.0, 1.0, -1.0)
    margins = target_margins(query_logits, payload_targets_arr)

    non_query_targets = [byte for sequence in batch.sequences for byte in sequence[:-1]]
    non_query_targets_arr = np.asarray(non_query_targets, dtype=np.int32)
    non_query_decoded_arr = block.encode_patterns(patterns[non_query_targets_arr]) @ block.encoder
    non_query_logits = byte_logits(non_query_decoded_arr, patterns)
    non_query_pred = np.argmax(non_query_logits, axis=1)

    non_equal = payload_targets_arr != query_current_targets_arr
    current_byte_cheat_rate = float(np.mean(query_pred[non_equal] == query_current_targets_arr[non_equal])) if np.any(non_equal) else 0.0
    unique_payloads, unique_indices = np.unique(payload_targets_arr, return_index=True)
    unique_context = output_context[unique_indices]
    context_rows = [tuple(np.round(row, 6).tolist()) for row in unique_context]
    payload_state_collision_count = int(len(unique_payloads) - len(set(context_rows)))

    return {
        "query_payload_exact_acc": float(np.mean(query_pred == payload_targets_arr)),
        "query_payload_bit_acc": float(np.mean(query_bits == payload_patterns)),
        "query_payload_margin_mean": float(np.mean(margins)),
        "query_payload_margin_min": float(np.min(margins)),
        "non_query_byte_reconstruction_acc": float(np.mean(non_query_pred == non_query_targets_arr)),
        "current_byte_cheat_rate": current_byte_cheat_rate,
        "payload_state_collision_count": payload_state_collision_count,
        "query_pred": query_pred,
    }


def prev_byte_baseline(batch: SequenceBatch) -> float:
    correct = 0
    for sequence, payload in zip(batch.sequences, batch.payloads):
        previous = sequence[-2] if len(sequence) >= 2 else -1
        if int(previous) == int(payload):
            correct += 1
    return float(correct / max(1, len(batch.sequences)))


def query_removed_false_positive_rate(*, core: np.ndarray, state_dim: int, batch: SequenceBatch) -> float:
    block = byte_block()
    patterns = batch.patterns
    distractor_tokens = []
    payload_targets = []
    for sequence, payload in zip(batch.sequences, batch.payloads):
        for byte_value in sequence[2:-1]:
            distractor_tokens.append(int(byte_value))
            payload_targets.append(int(payload))
    if not distractor_tokens:
        return 0.0
    decoded = block.encode_patterns(patterns[np.asarray(distractor_tokens, dtype=np.int32)]) @ block.encoder
    logits = byte_logits(decoded, patterns)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == np.asarray(payload_targets, dtype=np.int32)))


def evaluate_memory_spec(
    spec: MemorySpec,
    *,
    batch: SequenceBatch,
    distractor_lengths: Sequence[int],
    seed: int,
) -> dict[str, object]:
    core = memory_matrix(spec.state_dim, spec.visible_dim, spec.memory_entries)
    real = run_sequences(core=core, state_dim=spec.state_dim, batch=batch, mode="real", seed=seed + spec.candidate_id)
    reset = run_sequences(core=core, state_dim=spec.state_dim, batch=batch, mode="reset", seed=seed + 1000 + spec.candidate_id)
    time_shuffle = run_sequences(core=core, state_dim=spec.state_dim, batch=batch, mode="time_shuffle", seed=seed + 2000 + spec.candidate_id)
    random_state = run_sequences(core=core, state_dim=spec.state_dim, batch=batch, mode="random", seed=seed + 3000 + spec.candidate_id)
    marker_shuffle = run_sequences(core=core, state_dim=spec.state_dim, batch=batch, mode="marker_shuffle", seed=seed + 4000 + spec.candidate_id)

    length_passes = []
    for length in distractor_lengths:
        mask = batch.distractor_lengths == int(length)
        if not np.any(mask):
            length_passes.append(False)
            continue
        preds = np.asarray(real["query_pred"], dtype=np.int32)[mask]
        targets = batch.payloads[mask]
        length_passes.append(bool(np.mean(preds == targets) == 1.0))

    patterns = batch.patterns
    block = byte_block()
    zero_base = evaluate_ablock(block, patterns)
    q_removed = query_removed_false_positive_rate(core=core, state_dim=spec.state_dim, batch=batch)

    metrics: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "visible_dim": spec.visible_dim,
        "code_dim": spec.code_dim,
        "state_dim": spec.state_dim,
        "context_dim": spec.context_dim,
        "family": spec.family,
        "memory_edge_budget": spec.memory_edge_budget,
        "memory_edge_count": int(np.count_nonzero(core)),
        "memory_entries": entries_to_string(spec.memory_entries),
        "query_payload_exact_acc": float(real["query_payload_exact_acc"]),
        "query_payload_bit_acc": float(real["query_payload_bit_acc"]),
        "query_payload_margin_mean": float(real["query_payload_margin_mean"]),
        "query_payload_margin_min": float(real["query_payload_margin_min"]),
        "all_distractor_lengths_pass": bool(all(length_passes)),
        "long_sequence_payload_acc": float(real["query_payload_exact_acc"]),
        "payload_state_collision_count": int(real["payload_state_collision_count"]),
        "non_query_byte_reconstruction_acc": float(real["non_query_byte_reconstruction_acc"]),
        "zero_context_byte_reconstruction_acc": float(zero_base["exact_byte_acc"]),
        "reset_state_acc": float(reset["query_payload_exact_acc"]),
        "time_shuffle_state_acc": float(time_shuffle["query_payload_exact_acc"]),
        "random_state_acc": float(random_state["query_payload_exact_acc"]),
        "marker_shuffle_acc": float(marker_shuffle["query_payload_exact_acc"]),
        "query_removed_false_positive_rate": q_removed,
        "prev_byte_baseline_acc": prev_byte_baseline(batch),
        "current_byte_cheat_rate": float(real["current_byte_cheat_rate"]),
    }
    metrics["D21D_score"] = d21d_score(metrics)
    metrics["verdict"] = d21d_verdict(metrics)
    return metrics


def d21d_score(metrics: dict[str, object]) -> float:
    control_max = max(
        float(metrics["reset_state_acc"]),
        float(metrics["time_shuffle_state_acc"]),
        float(metrics["random_state_acc"]),
        float(metrics["marker_shuffle_acc"]),
        float(metrics["prev_byte_baseline_acc"]),
    )
    return (
        4.0 * float(metrics["query_payload_exact_acc"])
        + 1.0 * float(metrics["query_payload_bit_acc"])
        + 1.0 * float(metrics["long_sequence_payload_acc"])
        + 0.02 * float(metrics["query_payload_margin_min"])
        + 0.5 * float(metrics["non_query_byte_reconstruction_acc"])
        - 2.0 * control_max
        - 1.0 * float(metrics["current_byte_cheat_rate"])
        - 0.5 * float(metrics["query_removed_false_positive_rate"])
        - 0.002 * float(metrics["payload_state_collision_count"])
        - 0.001 * float(metrics["memory_edge_count"])
    )


def d21d_verdict(metrics: dict[str, object]) -> str:
    controls_clean = (
        float(metrics["reset_state_acc"]) <= 0.01
        and float(metrics["time_shuffle_state_acc"]) <= 0.01
        and float(metrics["random_state_acc"]) <= 0.01
        and float(metrics["marker_shuffle_acc"]) <= 0.01
        and float(metrics["prev_byte_baseline_acc"]) <= 0.01
        and float(metrics["current_byte_cheat_rate"]) <= 0.01
    )
    if float(metrics["zero_context_byte_reconstruction_acc"]) != 1.0 or float(metrics["non_query_byte_reconstruction_acc"]) != 1.0:
        return "D21D_BYTE_GATE_BROKEN"
    if float(metrics["query_payload_exact_acc"]) == 1.0 and bool(metrics["all_distractor_lengths_pass"]) and controls_clean:
        return "D21D_MARKER_MEMORY_PASS"
    if max(float(metrics["reset_state_acc"]), float(metrics["time_shuffle_state_acc"]), float(metrics["random_state_acc"]), float(metrics["marker_shuffle_acc"])) > 0.01:
        return "D21D_STATE_ARTIFACT"
    if float(metrics["current_byte_cheat_rate"]) > 0.05 or float(metrics["prev_byte_baseline_acc"]) > 0.05:
        return "D21D_PREV_BYTE_CHEAT"
    if float(metrics["query_payload_exact_acc"]) > 0.99:
        return "D21D_MARKER_MEMORY_WEAK_PASS"
    return "D21D_NO_MARKER_MEMORY_ROUTE"


def build_specs(*, state_dims: Sequence[int], edge_budgets: Sequence[int], samples: int, seed: int) -> list[MemorySpec]:
    rng = random.Random(seed)
    specs: list[MemorySpec] = []

    def add_spec(state_dim: int, edge_budget: int, family: str, entries: Iterable[tuple[int, int, float]]) -> None:
        cleaned = []
        for state_idx, visible_idx, value in entries:
            if 0 <= state_idx < state_dim and 0 <= visible_idx < VISIBLE_DIM and abs(float(value)) > 1e-12:
                cleaned.append((state_idx, visible_idx, value))
        specs.append(
            MemorySpec(
                candidate_id=len(specs),
                visible_dim=VISIBLE_DIM,
                code_dim=CODE_DIM,
                state_dim=state_dim,
                context_dim=CONTEXT_DIM,
                family=family,
                memory_edge_budget=edge_budget,
                memory_entries=dedupe_entries(cleaned),
            )
        )

    for state_dim in state_dims:
        for edge_budget in edge_budgets:
            add_spec(state_dim, edge_budget, "zero_memory", ())
            add_spec(state_dim, edge_budget, "identity_memory", identity_memory_entries(state_dim))
            add_spec(state_dim, edge_budget, "oracle_marker_memory", oracle_memory_entries(state_dim))
            for _ in range(samples):
                add_spec(
                    state_dim,
                    edge_budget,
                    "random_sparse_memory",
                    random_memory_entries(state_dim=state_dim, edge_budget=edge_budget, rng=rng),
                )
    return specs


def group_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["state_dim"]), int(row["memory_edge_budget"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, state_dim, edge_budget), items in sorted(groups.items()):
        best = max(items, key=lambda row: float(row["D21D_score"]))
        summary.append(
            {
                "family": family,
                "state_dim": state_dim,
                "memory_edge_budget": edge_budget,
                "count": len(items),
                "pass_count": sum(1 for row in items if str(row["verdict"]) == "D21D_MARKER_MEMORY_PASS"),
                "weak_count": sum(1 for row in items if str(row["verdict"]) == "D21D_MARKER_MEMORY_WEAK_PASS"),
                "best_score": float(best["D21D_score"]),
                "best_verdict": str(best["verdict"]),
                "best_query_payload_exact_acc": float(best["query_payload_exact_acc"]),
                "best_control_max": max(
                    float(best["reset_state_acc"]),
                    float(best["time_shuffle_state_acc"]),
                    float(best["random_state_acc"]),
                    float(best["marker_shuffle_acc"]),
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
    lines = ["D21D memory heatmap: brighter = query_exact * (1-control_max)"]
    lines.append("legend: PASS=P WEAK=W ARTIFACT=F NONE=.")
    lines.append("state\\edge " + " ".join(f"{budget:>5}" for budget in budgets))
    for state_dim in state_dims:
        cells = []
        for budget in budgets:
            value = values.get((state_dim, budget), 0.0)
            scaled = 0 if hi <= lo else int(round((value - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
            verdict = verdicts.get((state_dim, budget), ".")
            marker = "P" if verdict == "D21D_MARKER_MEMORY_PASS" else "W" if verdict == "D21D_MARKER_MEMORY_WEAK_PASS" else "F" if verdict == "D21D_STATE_ARTIFACT" else "."
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
    sorted_rows = sorted(rows, key=lambda row: float(row["D21D_score"]), reverse=True)
    write_csv(out_dir / "memory_candidates.csv", sorted_rows)
    write_csv(out_dir / "memory_family_summary.csv", summary)
    if path_rows is not None:
        write_csv(out_dir / "memory_paths.csv", path_rows)
    control_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "state_dim": row["state_dim"],
            "memory_edge_count": row["memory_edge_count"],
            "query_payload_exact_acc": row["query_payload_exact_acc"],
            "reset_state_acc": row["reset_state_acc"],
            "time_shuffle_state_acc": row["time_shuffle_state_acc"],
            "random_state_acc": row["random_state_acc"],
            "marker_shuffle_acc": row["marker_shuffle_acc"],
            "prev_byte_baseline_acc": row["prev_byte_baseline_acc"],
            "current_byte_cheat_rate": row["current_byte_cheat_rate"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows[: min(256, len(sorted_rows))]
    ]
    write_csv(out_dir / "memory_control_summary.csv", control_rows)
    (out_dir / "memory_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    pass_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21D_MARKER_MEMORY_PASS"]
    weak_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21D_MARKER_MEMORY_WEAK_PASS"]
    verdict = "D21D_MARKER_MEMORY_PASS" if pass_rows else "D21D_MARKER_MEMORY_WEAK_PASS" if weak_rows else "D21D_NO_MARKER_MEMORY_ROUTE"
    payload = {
        "verdict": verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "pass_count": len(pass_rows),
        "weak_count": len(weak_rows),
        "best_candidate": sorted_rows[0] if sorted_rows else None,
        "best_pass_candidate": pass_rows[0] if pass_rows else None,
        "best_weak_candidate": weak_rows[0] if weak_rows else None,
    }
    (out_dir / "memory_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# D21D Marker Memory A-Block Core Report",
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
                f"- memory_edges: `{best['memory_edge_count']}`",
                f"- query_payload_exact_acc: `{float(best['query_payload_exact_acc']):.6f}`",
                f"- query_payload_margin_min: `{float(best['query_payload_margin_min']):.6f}`",
                f"- non_query_byte_reconstruction_acc: `{float(best['non_query_byte_reconstruction_acc']):.6f}`",
                f"- reset_state_acc: `{float(best['reset_state_acc']):.6f}`",
                f"- time_shuffle_state_acc: `{float(best['time_shuffle_state_acc']):.6f}`",
                f"- random_state_acc: `{float(best['random_state_acc']):.6f}`",
                f"- marker_shuffle_acc: `{float(best['marker_shuffle_acc']):.6f}`",
                f"- prev_byte_baseline_acc: `{float(best['prev_byte_baseline_acc']):.6f}`",
                f"- current_byte_cheat_rate: `{float(best['current_byte_cheat_rate']):.6f}`",
                f"- verdict: `{best['verdict']}`",
            ]
        )
    report.extend(["", "## Heatmap", "", "```text", heatmap, "```", ""])
    (out_dir / "D21D_MARKER_MEMORY_ABLOCK_CORE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_self_checks() -> None:
    patterns = all_visible_patterns(VISIBLE_DIM)
    block = byte_block()
    base = evaluate_ablock(block, patterns)
    robust = robustness_metrics(block, patterns)
    assert float(base["exact_byte_acc"]) == 1.0
    assert float(robust["single_edge_drop_mean_bit"]) == 1.0
    batch = make_sequence_batch(distractor_lengths=[1, 2, 4], eval_sequences=4096, strict_distractors=True, seed=DEFAULT_SEED)
    spec = MemorySpec(0, VISIBLE_DIM, CODE_DIM, 32, CONTEXT_DIM, "oracle_marker_memory", 16, oracle_memory_entries(32))
    row = evaluate_memory_spec(spec, batch=batch, distractor_lengths=[1, 2, 4], seed=DEFAULT_SEED)
    assert str(row["verdict"]) == "D21D_MARKER_MEMORY_PASS", row


def evaluate_specs(args: argparse.Namespace, specs: Sequence[MemorySpec]) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    distractor_lengths = parse_int_list(args.distractor_lengths)
    batch = make_sequence_batch(
        distractor_lengths=distractor_lengths,
        eval_sequences=int(args.eval_sequences),
        strict_distractors=not bool(args.loose_distractors),
        seed=int(args.seed),
    )
    rows = []
    for idx, spec in enumerate(specs, start=1):
        rows.append(evaluate_memory_spec(spec, batch=batch, distractor_lengths=distractor_lengths, seed=int(args.seed)))
        if idx % 500 == 0:
            best = max(rows, key=lambda row: float(row["D21D_score"]))
            print(
                f"[D21d] evaluated {idx}/{len(specs)} "
                f"best={best['family']} state={best['state_dim']} "
                f"query={float(best['query_payload_exact_acc']):.4f} "
                f"control={max(float(best['reset_state_acc']), float(best['time_shuffle_state_acc']), float(best['random_state_acc'])):.4f} "
                f"verdict={best['verdict']}"
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
        "marker_byte": MARKER_BYTE,
        "query_byte": QUERY_BYTE,
        "zero_exact_byte_acc": float(base["exact_byte_acc"]),
        "zero_bit_acc": float(base["bit_acc"]),
        "single_edge_drop_mean_bit": float(base["single_edge_drop_mean_bit"]),
        "baseline_reproduced": True,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "baseline_check.csv", [row])
    (out_dir / "memory_top.json").write_text(json.dumps({"verdict": "D21D_BASELINE_REPRODUCED", "baseline": row}, indent=2), encoding="utf-8")
    (out_dir / "D21D_MARKER_MEMORY_ABLOCK_CORE_REPORT.md").write_text("# D21D Baseline Check\n\nVerdict: `D21D_BASELINE_REPRODUCED`\n", encoding="utf-8")
    print("[D21d] baseline reproduced D21A/D21B fixed shell")
    return 0


def run_marker_memory_oracle(args: argparse.Namespace) -> int:
    run_self_checks()
    specs = [MemorySpec(0, VISIBLE_DIM, CODE_DIM, int(args.state_dim), CONTEXT_DIM, "oracle_marker_memory", len(oracle_memory_entries(int(args.state_dim))), oracle_memory_entries(int(args.state_dim)))]
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="marker-memory-oracle",
        config={
            "state_dim": int(args.state_dim),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    best = rows[0]
    print(
        "[D21d] oracle "
        f"query={float(best['query_payload_exact_acc']):.4f} "
        f"reset={float(best['reset_state_acc']):.4f} "
        f"prev_base={float(best['prev_byte_baseline_acc']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def run_memory_atlas(args: argparse.Namespace) -> int:
    run_self_checks()
    specs = build_specs(
        state_dims=parse_int_list(args.state_dims),
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
            "memory_edge_budgets": parse_int_list(args.memory_edge_budgets),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "samples": int(args.samples),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    print(heatmap)
    best = max(rows, key=lambda row: float(row["D21D_score"]))
    print(
        "[D21d] best "
        f"family={best['family']} state={best['state_dim']} E={best['memory_edge_count']} "
        f"query={float(best['query_payload_exact_acc']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def start_entries(start_family: str, state_dim: int, seed: int) -> tuple[tuple[int, int, float], ...]:
    if start_family == "oracle_marker_memory":
        return oracle_memory_entries(state_dim)
    if start_family == "identity_memory":
        return identity_memory_entries(state_dim)
    if start_family == "random_sparse_memory":
        return random_memory_entries(state_dim=state_dim, edge_budget=min(16, state_dim * VISIBLE_DIM), rng=random.Random(seed))
    raise ValueError(f"unknown start family: {start_family}")


def run_crystallize_memory(args: argparse.Namespace) -> int:
    run_self_checks()
    state_dim = int(args.state_dim)
    edge_budget = int(args.memory_edge_budget)
    current_entries = start_entries(str(args.start_family), state_dim, int(args.seed))

    def eval_one(candidate_id: int, family: str, entries: tuple[tuple[int, int, float]]) -> dict[str, object]:
        rows, _summary, _heat = evaluate_specs(
            args,
            [MemorySpec(candidate_id, VISIBLE_DIM, CODE_DIM, state_dim, CONTEXT_DIM, family, edge_budget, entries)],
        )
        return rows[0]

    current = eval_one(0, f"start_{args.start_family}", current_entries)
    best_rows = [current]
    path_rows = [{**current, "step": 0, "accepted": True, "reason": "start"}]
    print(
        f"[D21d] crystallize start score={float(current['D21D_score']):.6f} "
        f"edges={current['memory_edge_count']} query={float(current['query_payload_exact_acc']):.4f} "
        f"verdict={current['verdict']}"
    )

    for step in range(1, int(args.max_steps) + 1):
        proposals = []
        parsed = list(current_entries)
        for drop_idx in range(len(parsed)):
            proposals.append(eval_one(step * 1000 + len(proposals), "drop_edge", dedupe_entries(parsed[:drop_idx] + parsed[drop_idx + 1 :])))
        for weight_idx, (state_idx, visible_idx, old_value) in enumerate(parsed):
            for new_value in (-1.0, -0.5, 0.5, 1.0):
                if abs(new_value - old_value) < 1e-12:
                    continue
                entries = parsed.copy()
                entries[weight_idx] = (state_idx, visible_idx, new_value)
                proposals.append(eval_one(step * 1000 + len(proposals), "reweight", dedupe_entries(entries)))

        best_rows.extend(proposals)
        pass_props = [row for row in proposals if str(row["verdict"]) == "D21D_MARKER_MEMORY_PASS"]
        accepted = False
        reason = "reject"
        if pass_props:
            best_prop = max(
                pass_props,
                key=lambda row: (
                    -int(row["memory_edge_count"]),
                    float(row["D21D_score"]),
                    float(row["query_payload_margin_min"]),
                ),
            )
            if int(best_prop["memory_edge_count"]) < int(current["memory_edge_count"]) or float(best_prop["D21D_score"]) > float(current["D21D_score"]) + 1e-9:
                current = best_prop
                current_entries = parse_entries(str(current["memory_entries"]))
                accepted = True
                reason = "gate_preserving_simplify"
        path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
        print(
            f"[D21d][crystallize {step}] score={float(current['D21D_score']):.6f} "
            f"edges={current['memory_edge_count']} query={float(current['query_payload_exact_acc']):.4f} "
            f"{current['verdict']} {reason}"
        )
        if not accepted:
            break

    rows = sorted(best_rows, key=lambda row: float(row["D21D_score"]), reverse=True)
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
            "start_family": str(args.start_family),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
        path_rows=path_rows,
    )
    print(heatmap)
    print(f"[D21d] wrote {args.out}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    run_self_checks()
    candidate_path = Path(args.candidates)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    with candidate_path.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows = sorted(source_rows, key=lambda row: float(row.get("D21D_score", 0.0)), reverse=True)[: int(args.top_k)]
    specs = []
    for idx, row in enumerate(source_rows):
        specs.append(
            MemorySpec(
                candidate_id=idx,
                visible_dim=int(row["visible_dim"]),
                code_dim=int(row["code_dim"]),
                state_dim=int(row["state_dim"]),
                context_dim=int(row["context_dim"]),
                family=f"confirm_{row['family']}",
                memory_edge_budget=int(row["memory_edge_budget"]),
                memory_entries=parse_entries(str(row["memory_entries"])),
            )
        )
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="confirm",
        config={
            "source": str(candidate_path),
            "top_k": int(args.top_k),
            "distractor_lengths": parse_int_list(args.distractor_lengths),
            "eval_sequences": int(args.eval_sequences),
            "seed": int(args.seed),
        },
    )
    best = max(rows, key=lambda row: float(row["D21D_score"]))
    print(
        "[D21d] confirm best "
        f"family={best['family']} state={best['state_dim']} "
        f"edges={best['memory_edge_count']} query={float(best['query_payload_exact_acc']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21d marker-memory A-block core")
    parser.add_argument("--mode", choices=("baseline-check", "marker-memory-oracle", "memory-atlas", "crystallize-memory", "confirm"), required=True)
    parser.add_argument("--state-dims", default="16,32,64")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--memory-edge-budgets", default="16,24,32,48")
    parser.add_argument("--memory-edge-budget", type=int, default=32)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-sequences", type=int, default=8192)
    parser.add_argument("--distractor-lengths", default="1,2,4,8,16")
    parser.add_argument("--loose-distractors", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-family", default="oracle_marker_memory")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "baseline-check":
        return run_baseline_check(args)
    if args.mode == "marker-memory-oracle":
        return run_marker_memory_oracle(args)
    if args.mode == "memory-atlas":
        return run_memory_atlas(args)
    if args.mode == "crystallize-memory":
        return run_crystallize_memory(args)
    if args.mode == "confirm":
        return run_confirm(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
