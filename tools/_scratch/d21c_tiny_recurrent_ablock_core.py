#!/usr/bin/env python3
"""
D21c tiny recurrent core behind the reciprocal A-block.

Prototype goal:
    tick t-1 byte -> tiny state/core -> context vector at tick t -> output previous byte

The D21A byte lane and D21B context write lane are fixed. D21C only searches a
tiny state update core that stores the current byte as the next tick's context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from d21a_reciprocal_byte_ablock import (
    ReciprocalABlock,
    all_visible_patterns,
    dedupe_entries,
    evaluate_ablock,
    redundant_copy_entries,
    robustness_metrics,
)
from d21b_context_ablock import context_matrix


ASCII_SHADE = " .:-=+*#%@"
DEFAULT_SEED = 20260502
VISIBLE_DIM = 8
CODE_DIM = 16
CONTEXT_DIM = 16


@dataclass(frozen=True)
class CoreSpec:
    candidate_id: int
    visible_dim: int
    code_dim: int
    state_dim: int
    context_dim: int
    family: str
    core_edge_budget: int
    core_entries: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class PairBatch:
    prev_indices: np.ndarray
    current_indices: np.ndarray
    prev_patterns: np.ndarray
    current_patterns: np.ndarray
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


def entries_to_string(entries: Sequence[tuple[int, int, float]]) -> str:
    return " ".join(f"{state}:{visible}:{value:g}" for state, visible, value in entries)


def core_matrix(state_dim: int, visible_dim: int, entries: Iterable[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((state_dim, visible_dim), dtype=np.float32)
    for state_idx, visible_idx, value in entries:
        if 0 <= state_idx < state_dim and 0 <= visible_idx < visible_dim:
            matrix[state_idx, visible_idx] = float(value)
    return matrix


def byte_block() -> ReciprocalABlock:
    return ReciprocalABlock.from_entries(VISIBLE_DIM, CODE_DIM, redundant_copy_entries(VISIBLE_DIM, CODE_DIM))


def d21b_context_entries() -> tuple[tuple[int, int, float], ...]:
    return tuple(
        [(idx, idx, 4.0) for idx in range(VISIBLE_DIM)]
        + [(VISIBLE_DIM + idx, idx, -4.0) for idx in range(VISIBLE_DIM)]
    )


def d21b_context_weights() -> np.ndarray:
    return context_matrix(CONTEXT_DIM, VISIBLE_DIM, d21b_context_entries())


def make_pair_batch(*, max_pairs: int, seed: int) -> PairBatch:
    patterns = all_visible_patterns(VISIBLE_DIM)
    pairs = [(prev, cur) for prev in range(1 << VISIBLE_DIM) for cur in range(1 << VISIBLE_DIM)]
    rng = np.random.default_rng(seed)
    if 0 < max_pairs < len(pairs):
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(idx)] for idx in keep]
    prev_indices = np.asarray([item[0] for item in pairs], dtype=np.int32)
    current_indices = np.asarray([item[1] for item in pairs], dtype=np.int32)
    return PairBatch(
        prev_indices=prev_indices,
        current_indices=current_indices,
        prev_patterns=patterns[prev_indices],
        current_patterns=patterns[current_indices],
        patterns=patterns,
    )


def state_to_context(states: np.ndarray, state_dim: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if state_dim >= CONTEXT_DIM:
        return states[:, :CONTEXT_DIM]
    if state_dim >= VISIBLE_DIM:
        first = states[:, :VISIBLE_DIM]
        return np.concatenate([first, -first], axis=1)
    context = np.zeros((states.shape[0], CONTEXT_DIM), dtype=np.float32)
    context[:, :state_dim] = states
    context[:, VISIBLE_DIM : VISIBLE_DIM + state_dim] = -states
    return context


def byte_logits(decoded: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    return decoded @ patterns.T


def target_margins(logits: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    rows = np.arange(target_indices.shape[0])
    target_logits = logits[rows, target_indices]
    masked = logits.copy()
    masked[rows, target_indices] = -np.inf
    return target_logits - np.max(masked, axis=1)


def eval_output(
    *,
    base_decoded: np.ndarray,
    output_context: np.ndarray,
    target_indices: np.ndarray,
    target_patterns: np.ndarray,
    patterns: np.ndarray,
    context_weights: np.ndarray,
) -> dict[str, float | np.ndarray]:
    decoded = base_decoded + output_context @ context_weights
    logits = byte_logits(decoded, patterns)
    pred = np.argmax(logits, axis=1)
    pred_bits = np.where(decoded >= 0.0, 1.0, -1.0)
    margins = target_margins(logits, target_indices)
    return {
        "exact_acc": float(np.mean(pred == target_indices)),
        "bit_acc": float(np.mean(pred_bits == target_patterns)),
        "margin_mean": float(np.mean(margins)),
        "margin_min": float(np.min(margins)),
        "pred": pred,
    }


def make_random_sequences(*, sequence_count: int, sequence_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << VISIBLE_DIM, size=(sequence_count, sequence_len), dtype=np.int32)


def evaluate_sequence_exact(
    *,
    core: np.ndarray,
    state_dim: int,
    block: ReciprocalABlock,
    context_weights: np.ndarray,
    sequence_count: int,
    sequence_len: int,
    seed: int,
) -> float:
    patterns = all_visible_patterns(VISIBLE_DIM)
    sequences = make_random_sequences(sequence_count=sequence_count, sequence_len=sequence_len, seed=seed)
    if sequence_len < 2 or sequence_count <= 0:
        return 0.0
    prev_indices = sequences[:, :-1].reshape(-1)
    current_indices = sequences[:, 1:].reshape(-1)
    prev_patterns = patterns[prev_indices]
    current_patterns = patterns[current_indices]
    prev_states = prev_patterns @ core.T
    output_context = state_to_context(prev_states, state_dim)
    base_decoded = block.encode_patterns(current_patterns) @ block.encoder
    logits = byte_logits(base_decoded + output_context @ context_weights, patterns)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == prev_indices))


def evaluate_core_spec(
    spec: CoreSpec,
    *,
    batch: PairBatch,
    block: ReciprocalABlock,
    context_weights: np.ndarray,
    sequence_len: int,
    sequence_count: int,
    seed: int,
) -> dict[str, object]:
    core = core_matrix(spec.state_dim, spec.visible_dim, spec.core_entries)
    prev_states = batch.prev_patterns @ core.T
    output_context = state_to_context(prev_states, spec.state_dim)
    base_decoded = block.encode_patterns(batch.current_patterns) @ block.encoder

    real = eval_output(
        base_decoded=base_decoded,
        output_context=output_context,
        target_indices=batch.prev_indices,
        target_patterns=batch.prev_patterns,
        patterns=batch.patterns,
        context_weights=context_weights,
    )

    zero_context = np.zeros_like(output_context, dtype=np.float32)
    reset = eval_output(
        base_decoded=base_decoded,
        output_context=zero_context,
        target_indices=batch.prev_indices,
        target_patterns=batch.prev_patterns,
        patterns=batch.patterns,
        context_weights=context_weights,
    )

    rng = np.random.default_rng(seed + spec.candidate_id + spec.state_dim)
    shuffled_context = output_context.copy()
    rng.shuffle(shuffled_context)
    time_shuffle = eval_output(
        base_decoded=base_decoded,
        output_context=shuffled_context,
        target_indices=batch.prev_indices,
        target_patterns=batch.prev_patterns,
        patterns=batch.patterns,
        context_weights=context_weights,
    )

    random_state = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=prev_states.shape)
    random_context = state_to_context(random_state.astype(np.float32), spec.state_dim)
    random_control = eval_output(
        base_decoded=base_decoded,
        output_context=random_context,
        target_indices=batch.prev_indices,
        target_patterns=batch.prev_patterns,
        patterns=batch.patterns,
        context_weights=context_weights,
    )

    current_rows = np.arange(batch.current_indices.shape[0])
    real_pred = np.asarray(real["pred"], dtype=np.int32)
    non_equal = batch.prev_indices != batch.current_indices
    current_byte_cheat_rate = float(np.mean(real_pred[non_equal] == batch.current_indices[non_equal])) if np.any(non_equal) else 0.0

    all_patterns = batch.patterns
    zero_decoded = block.encode_patterns(all_patterns) @ block.encoder
    zero_logits = byte_logits(zero_decoded, all_patterns)
    zero_pred = np.argmax(zero_logits, axis=1)
    zero_context_byte_reconstruction_acc = float(np.mean(zero_pred == np.arange(all_patterns.shape[0])))

    state_rows = [tuple(np.round(row, 6).tolist()) for row in all_patterns @ core.T]
    state_collision_count = int(len(state_rows) - len(set(state_rows)))
    long_sequence_exact_acc = evaluate_sequence_exact(
        core=core,
        state_dim=spec.state_dim,
        block=block,
        context_weights=context_weights,
        sequence_count=sequence_count,
        sequence_len=sequence_len,
        seed=seed + 99 + spec.candidate_id,
    )

    metrics: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "visible_dim": spec.visible_dim,
        "code_dim": spec.code_dim,
        "state_dim": spec.state_dim,
        "context_dim": spec.context_dim,
        "family": spec.family,
        "core_edge_budget": spec.core_edge_budget,
        "core_edge_count": int(np.count_nonzero(core)),
        "core_entries": entries_to_string(spec.core_entries),
        "prev_byte_exact_acc": float(real["exact_acc"]),
        "prev_byte_bit_acc": float(real["bit_acc"]),
        "prev_byte_margin_mean": float(real["margin_mean"]),
        "prev_byte_margin_min": float(real["margin_min"]),
        "long_sequence_exact_acc": long_sequence_exact_acc,
        "state_collision_count": state_collision_count,
        "reset_each_token_acc": float(reset["exact_acc"]),
        "time_shuffle_state_acc": float(time_shuffle["exact_acc"]),
        "random_state_acc": float(random_control["exact_acc"]),
        "current_byte_cheat_rate": current_byte_cheat_rate,
        "zero_context_byte_reconstruction_acc": zero_context_byte_reconstruction_acc,
    }
    metrics["D21C_score"] = d21c_score(metrics)
    metrics["verdict"] = d21c_verdict(metrics)
    return metrics


def d21c_score(metrics: dict[str, object]) -> float:
    control_max = max(
        float(metrics["reset_each_token_acc"]),
        float(metrics["time_shuffle_state_acc"]),
        float(metrics["random_state_acc"]),
    )
    return (
        3.0 * float(metrics["prev_byte_exact_acc"])
        + 1.0 * float(metrics["prev_byte_bit_acc"])
        + 1.0 * float(metrics["long_sequence_exact_acc"])
        + 0.02 * float(metrics["prev_byte_margin_min"])
        - 2.0 * control_max
        - 1.0 * float(metrics["current_byte_cheat_rate"])
        - 0.001 * float(metrics["core_edge_count"])
        - 0.002 * float(metrics["state_collision_count"])
    )


def d21c_verdict(metrics: dict[str, object]) -> str:
    controls_clean = (
        float(metrics["reset_each_token_acc"]) <= 0.01
        and float(metrics["time_shuffle_state_acc"]) <= 0.01
        and float(metrics["random_state_acc"]) <= 0.01
        and float(metrics["current_byte_cheat_rate"]) <= 0.01
    )
    zero_ok = float(metrics["zero_context_byte_reconstruction_acc"]) == 1.0
    if not zero_ok:
        return "D21C_BYTE_GATE_BROKEN"
    if (
        float(metrics["prev_byte_exact_acc"]) == 1.0
        and float(metrics["long_sequence_exact_acc"]) >= 0.999
        and controls_clean
    ):
        return "D21C_PREV_BYTE_CORE_PASS"
    if max(
        float(metrics["reset_each_token_acc"]),
        float(metrics["time_shuffle_state_acc"]),
        float(metrics["random_state_acc"]),
    ) > 0.01 or float(metrics["current_byte_cheat_rate"]) > 0.05:
        return "D21C_STATE_ARTIFACT"
    if float(metrics["prev_byte_exact_acc"]) == 1.0:
        return "D21C_CORE_WEAK_PASS"
    return "D21C_NO_STATE_ROUTE"


def oracle_prev_byte_entries(state_dim: int) -> tuple[tuple[int, int, float], ...]:
    entries: list[tuple[int, int, float]] = []
    for bit in range(VISIBLE_DIM):
        entries.append((bit, bit, 1.0))
    if state_dim >= CONTEXT_DIM:
        for bit in range(VISIBLE_DIM):
            entries.append((VISIBLE_DIM + bit, bit, -1.0))
    return dedupe_entries(entries)


def identity_state_entries(state_dim: int) -> tuple[tuple[int, int, float], ...]:
    return dedupe_entries((bit, bit, 1.0) for bit in range(min(state_dim, VISIBLE_DIM)))


def random_sparse_entries(*, state_dim: int, edge_budget: int, rng: random.Random) -> tuple[tuple[int, int, float], ...]:
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


def build_specs(*, state_dims: Sequence[int], edge_budgets: Sequence[int], samples: int, seed: int) -> list[CoreSpec]:
    rng = random.Random(seed)
    specs: list[CoreSpec] = []

    def add_spec(state_dim: int, edge_budget: int, family: str, entries: Iterable[tuple[int, int, float]]) -> None:
        cleaned = []
        for state_idx, visible_idx, value in entries:
            if 0 <= state_idx < state_dim and 0 <= visible_idx < VISIBLE_DIM and abs(float(value)) > 1e-12:
                cleaned.append((state_idx, visible_idx, value))
        specs.append(
            CoreSpec(
                candidate_id=len(specs),
                visible_dim=VISIBLE_DIM,
                code_dim=CODE_DIM,
                state_dim=state_dim,
                context_dim=CONTEXT_DIM,
                family=family,
                core_edge_budget=edge_budget,
                core_entries=dedupe_entries(cleaned),
            )
        )

    for state_dim in state_dims:
        for edge_budget in edge_budgets:
            add_spec(state_dim, edge_budget, "zero_state", ())
            add_spec(state_dim, edge_budget, "identity_state", identity_state_entries(state_dim))
            add_spec(state_dim, edge_budget, "oracle_prev_byte", oracle_prev_byte_entries(state_dim))
            for _ in range(samples):
                add_spec(
                    state_dim,
                    edge_budget,
                    "random_sparse_core",
                    random_sparse_entries(state_dim=state_dim, edge_budget=edge_budget, rng=rng),
                )
    return specs


def group_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["state_dim"]), int(row["core_edge_budget"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, state_dim, edge_budget), items in sorted(groups.items()):
        best = max(items, key=lambda row: float(row["D21C_score"]))
        summary.append(
            {
                "family": family,
                "state_dim": state_dim,
                "core_edge_budget": edge_budget,
                "count": len(items),
                "pass_count": sum(1 for row in items if str(row["verdict"]) == "D21C_PREV_BYTE_CORE_PASS"),
                "weak_count": sum(1 for row in items if str(row["verdict"]) == "D21C_CORE_WEAK_PASS"),
                "best_score": float(best["D21C_score"]),
                "best_verdict": str(best["verdict"]),
                "best_prev_byte_exact_acc": float(best["prev_byte_exact_acc"]),
                "best_long_sequence_exact_acc": float(best["long_sequence_exact_acc"]),
                "best_control_max": max(
                    float(best["reset_each_token_acc"]),
                    float(best["time_shuffle_state_acc"]),
                    float(best["random_state_acc"]),
                ),
                "best_core_edge_count": int(best["core_edge_count"]),
            }
        )
    return summary


def make_heatmap(summary: Sequence[dict[str, object]]) -> str:
    state_dims = sorted({int(row["state_dim"]) for row in summary})
    budgets = sorted({int(row["core_edge_budget"]) for row in summary})
    values: dict[tuple[int, int], float] = {}
    verdicts: dict[tuple[int, int], str] = {}
    for row in summary:
        key = (int(row["state_dim"]), int(row["core_edge_budget"]))
        value = float(row["best_prev_byte_exact_acc"]) * float(row["best_long_sequence_exact_acc"])
        if key not in values or value > values[key]:
            values[key] = value
            verdicts[key] = str(row["best_verdict"])
    all_values = list(values.values()) or [0.0]
    lo = min(all_values)
    hi = max(all_values)
    lines = ["D21C core heatmap: brighter = pair_exact * long_sequence_exact"]
    lines.append("legend: PASS=P WEAK=W ARTIFACT=F NONE=.")
    lines.append("state\\edge " + " ".join(f"{budget:>5}" for budget in budgets))
    for state_dim in state_dims:
        cells = []
        for budget in budgets:
            value = values.get((state_dim, budget), 0.0)
            scaled = 0 if hi <= lo else int(round((value - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
            verdict = verdicts.get((state_dim, budget), ".")
            marker = "P" if verdict == "D21C_PREV_BYTE_CORE_PASS" else "W" if verdict == "D21C_CORE_WEAK_PASS" else "F" if verdict == "D21C_STATE_ARTIFACT" else "."
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
    sorted_rows = sorted(rows, key=lambda row: float(row["D21C_score"]), reverse=True)
    write_csv(out_dir / "core_candidates.csv", sorted_rows)
    write_csv(out_dir / "core_family_summary.csv", summary)
    if path_rows is not None:
        write_csv(out_dir / "core_paths.csv", path_rows)
    control_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "state_dim": row["state_dim"],
            "core_edge_count": row["core_edge_count"],
            "prev_byte_exact_acc": row["prev_byte_exact_acc"],
            "reset_each_token_acc": row["reset_each_token_acc"],
            "time_shuffle_state_acc": row["time_shuffle_state_acc"],
            "random_state_acc": row["random_state_acc"],
            "current_byte_cheat_rate": row["current_byte_cheat_rate"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows[: min(256, len(sorted_rows))]
    ]
    write_csv(out_dir / "core_control_summary.csv", control_rows)
    (out_dir / "core_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    pass_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21C_PREV_BYTE_CORE_PASS"]
    weak_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21C_CORE_WEAK_PASS"]
    verdict = "D21C_PREV_BYTE_CORE_PASS" if pass_rows else "D21C_CORE_WEAK_PASS" if weak_rows else "D21C_NO_STATE_ROUTE"
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
    (out_dir / "core_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# D21C Tiny Recurrent A-Block Core Report",
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
                f"- core_edges: `{best['core_edge_count']}`",
                f"- prev_byte_exact_acc: `{float(best['prev_byte_exact_acc']):.6f}`",
                f"- long_sequence_exact_acc: `{float(best['long_sequence_exact_acc']):.6f}`",
                f"- reset_each_token_acc: `{float(best['reset_each_token_acc']):.6f}`",
                f"- time_shuffle_state_acc: `{float(best['time_shuffle_state_acc']):.6f}`",
                f"- random_state_acc: `{float(best['random_state_acc']):.6f}`",
                f"- current_byte_cheat_rate: `{float(best['current_byte_cheat_rate']):.6f}`",
                f"- prev_byte_margin_min: `{float(best['prev_byte_margin_min']):.6f}`",
                f"- verdict: `{best['verdict']}`",
            ]
        )
    report.extend(["", "## Heatmap", "", "```text", heatmap, "```", ""])
    (out_dir / "D21C_TINY_RECURRENT_ABLOCK_CORE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_self_checks() -> None:
    block = byte_block()
    patterns = all_visible_patterns(VISIBLE_DIM)
    base = evaluate_ablock(block, patterns)
    robust = robustness_metrics(block, patterns)
    assert float(base["exact_byte_acc"]) == 1.0
    assert float(base["bit_acc"]) == 1.0
    assert float(robust["single_edge_drop_mean_bit"]) == 1.0
    batch = make_pair_batch(max_pairs=65536, seed=DEFAULT_SEED)
    spec = CoreSpec(0, VISIBLE_DIM, CODE_DIM, 16, CONTEXT_DIM, "oracle_prev_byte", 16, oracle_prev_byte_entries(16))
    row = evaluate_core_spec(
        spec,
        batch=batch,
        block=block,
        context_weights=d21b_context_weights(),
        sequence_len=16,
        sequence_count=512,
        seed=DEFAULT_SEED,
    )
    assert str(row["verdict"]) == "D21C_PREV_BYTE_CORE_PASS", row
    assert float(row["prev_byte_exact_acc"]) == 1.0


def evaluate_specs(args: argparse.Namespace, specs: Sequence[CoreSpec]) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    block = byte_block()
    context_weights = d21b_context_weights()
    batch = make_pair_batch(max_pairs=int(args.eval_pairs), seed=int(args.seed))
    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(specs, start=1):
        rows.append(
            evaluate_core_spec(
                spec,
                batch=batch,
                block=block,
                context_weights=context_weights,
                sequence_len=int(args.sequence_len),
                sequence_count=int(args.sequence_count),
                seed=int(args.seed),
            )
        )
        if idx % 1000 == 0:
            best = max(rows, key=lambda row: float(row["D21C_score"]))
            print(
                f"[D21c] evaluated {idx}/{len(specs)} "
                f"best={best['family']} state={best['state_dim']} "
                f"pair={float(best['prev_byte_exact_acc']):.4f} "
                f"long={float(best['long_sequence_exact_acc']):.4f} "
                f"verdict={best['verdict']}"
            )
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    return rows, summary, heatmap


def run_baseline_check(args: argparse.Namespace) -> int:
    run_self_checks()
    block = byte_block()
    patterns = all_visible_patterns(VISIBLE_DIM)
    base = {**evaluate_ablock(block, patterns), **robustness_metrics(block, patterns)}
    row = {
        "mode": "baseline-check",
        "zero_exact_byte_acc": float(base["exact_byte_acc"]),
        "zero_bit_acc": float(base["bit_acc"]),
        "zero_byte_margin_min": float(base["byte_margin_min"]),
        "zero_reciprocity_error": float(base["reciprocity_error"]),
        "single_edge_drop_mean_bit": float(base["single_edge_drop_mean_bit"]),
        "d21b_context_entries": entries_to_string(d21b_context_entries()),
        "baseline_reproduced": True,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "baseline_check.csv", [row])
    (out_dir / "core_top.json").write_text(json.dumps({"verdict": "D21C_BASELINE_REPRODUCED", "baseline": row}, indent=2), encoding="utf-8")
    (out_dir / "D21C_TINY_RECURRENT_ABLOCK_CORE_REPORT.md").write_text(
        "# D21C Baseline Check\n\nVerdict: `D21C_BASELINE_REPRODUCED`\n",
        encoding="utf-8",
    )
    print("[D21c] baseline reproduced D21A byte lane and D21B context entries")
    return 0


def run_prev_byte_oracle(args: argparse.Namespace) -> int:
    run_self_checks()
    state_dim = int(args.state_dim)
    specs = [CoreSpec(0, VISIBLE_DIM, CODE_DIM, state_dim, CONTEXT_DIM, "oracle_prev_byte", len(oracle_prev_byte_entries(state_dim)), oracle_prev_byte_entries(state_dim))]
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="prev-byte-oracle",
        config={
            "state_dim": state_dim,
            "eval_pairs": int(args.eval_pairs),
            "sequence_len": int(args.sequence_len),
            "sequence_count": int(args.sequence_count),
        },
    )
    best = rows[0]
    print(
        "[D21c] oracle "
        f"pair={float(best['prev_byte_exact_acc']):.4f} "
        f"long={float(best['long_sequence_exact_acc']):.4f} "
        f"reset={float(best['reset_each_token_acc']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def run_core_atlas(args: argparse.Namespace) -> int:
    run_self_checks()
    specs = build_specs(
        state_dims=parse_int_list(args.state_dims),
        edge_budgets=parse_int_list(args.core_edge_budgets),
        samples=int(args.samples),
        seed=int(args.seed),
    )
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="core-atlas",
        config={
            "state_dims": parse_int_list(args.state_dims),
            "core_edge_budgets": parse_int_list(args.core_edge_budgets),
            "samples": int(args.samples),
            "eval_pairs": int(args.eval_pairs),
            "sequence_len": int(args.sequence_len),
            "sequence_count": int(args.sequence_count),
            "seed": int(args.seed),
        },
    )
    print(heatmap)
    best = max(rows, key=lambda row: float(row["D21C_score"]))
    print(
        "[D21c] best "
        f"family={best['family']} state={best['state_dim']} E={best['core_edge_count']} "
        f"pair={float(best['prev_byte_exact_acc']):.4f} "
        f"long={float(best['long_sequence_exact_acc']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def mutate_core_entries(
    entries: tuple[tuple[int, int, float], ...],
    *,
    state_dim: int,
    edge_budget: int,
    rng: random.Random,
    crystallize: bool,
) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    used = {(state, visible) for state, visible, _ in current}
    ops = ("remove", "reweight", "flip") if crystallize or len(current) >= edge_budget else ("add", "remove", "reweight", "flip")
    op = rng.choice(ops)
    if op == "add":
        for _ in range(100):
            state_idx = rng.randrange(state_dim)
            visible_idx = rng.randrange(VISIBLE_DIM)
            if (state_idx, visible_idx) not in used:
                current.append((state_idx, visible_idx, rng.choice((-1.0, 1.0))))
                return dedupe_entries(current)
    if op == "remove" and current:
        del current[rng.randrange(len(current))]
        return dedupe_entries(current)
    if op == "reweight" and current:
        idx = rng.randrange(len(current))
        state_idx, visible_idx, _ = current[idx]
        current[idx] = (state_idx, visible_idx, rng.choice((-1.0, -0.5, 0.5, 1.0)))
        return dedupe_entries(current)
    if op == "flip" and current:
        idx = rng.randrange(len(current))
        old_state, old_visible, value = current[idx]
        used.discard((old_state, old_visible))
        for _ in range(100):
            state_idx = rng.randrange(state_dim)
            visible_idx = rng.randrange(VISIBLE_DIM)
            if (state_idx, visible_idx) not in used:
                current[idx] = (state_idx, visible_idx, value)
                return dedupe_entries(current)
    return entries


def start_entries(start_family: str, state_dim: int, seed: int) -> tuple[tuple[int, int, float], ...]:
    if start_family == "oracle_prev_byte":
        return oracle_prev_byte_entries(state_dim)
    if start_family == "identity_state":
        return identity_state_entries(state_dim)
    if start_family == "random_sparse_core":
        return random_sparse_entries(state_dim=state_dim, edge_budget=min(16, state_dim * VISIBLE_DIM), rng=random.Random(seed))
    raise ValueError(f"unknown start family: {start_family}")


def run_crystallize_core(args: argparse.Namespace) -> int:
    run_self_checks()
    state_dim = int(args.state_dim)
    edge_budget = int(args.core_edge_budget)
    rng = random.Random(int(args.seed))
    current_entries = start_entries(str(args.start_family), state_dim, int(args.seed))

    def eval_one(candidate_id: int, family: str, entries: tuple[tuple[int, int, float], ...]) -> dict[str, object]:
        rows, _summary, _heat = evaluate_specs(
            args,
            [CoreSpec(candidate_id, VISIBLE_DIM, CODE_DIM, state_dim, CONTEXT_DIM, family, edge_budget, entries)],
        )
        return rows[0]

    current = eval_one(0, f"start_{args.start_family}", current_entries)
    best_rows = [current]
    path_rows = [{**current, "step": 0, "accepted": True, "reason": "start"}]
    print(
        f"[D21c] crystallize start score={float(current['D21C_score']):.6f} "
        f"edges={current['core_edge_count']} pair={float(current['prev_byte_exact_acc']):.4f} "
        f"verdict={current['verdict']}"
    )

    stale = 0
    for step in range(1, int(args.max_steps) + 1):
        proposals = []
        parsed = list(current_entries)
        if str(args.mode) == "crystallize-core":
            for drop_idx in range(len(parsed)):
                proposals.append(eval_one(step * 1000 + len(proposals), "drop_edge", dedupe_entries(parsed[:drop_idx] + parsed[drop_idx + 1 :])))
            for weight_idx, (state_idx, visible_idx, old_value) in enumerate(parsed):
                for new_value in (-1.0, -0.5, 0.5, 1.0):
                    if abs(new_value - old_value) < 1e-12:
                        continue
                    entries = parsed.copy()
                    entries[weight_idx] = (state_idx, visible_idx, new_value)
                    proposals.append(eval_one(step * 1000 + len(proposals), "reweight", dedupe_entries(entries)))
        if not proposals:
            for worker_idx in range(max(1, int(args.workers))):
                local_rng = random.Random(rng.randrange(1 << 30) + worker_idx)
                proposals.append(
                    eval_one(
                        step * 1000 + worker_idx,
                        "proposal",
                        mutate_core_entries(
                            current_entries,
                            state_dim=state_dim,
                            edge_budget=edge_budget,
                            rng=local_rng,
                            crystallize=True,
                        ),
                    )
                )
        best_rows.extend(proposals)
        pass_props = [row for row in proposals if str(row["verdict"]) == "D21C_PREV_BYTE_CORE_PASS"]
        accepted = False
        reason = "reject"
        if pass_props:
            best_prop = max(
                pass_props,
                key=lambda row: (
                    -int(row["core_edge_count"]),
                    float(row["D21C_score"]),
                    float(row["prev_byte_margin_min"]),
                ),
            )
            if int(best_prop["core_edge_count"]) < int(current["core_edge_count"]) or float(best_prop["D21C_score"]) > float(current["D21C_score"]) + 1e-9:
                current = best_prop
                current_entries = parse_entries(str(current["core_entries"]))
                accepted = True
                stale = 0
                reason = "gate_preserving_simplify"
            else:
                stale += len(proposals)
        else:
            stale += len(proposals)
        path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
        print(
            f"[D21c][crystallize {step}] score={float(current['D21C_score']):.6f} "
            f"edges={current['core_edge_count']} pair={float(current['prev_byte_exact_acc']):.4f} "
            f"long={float(current['long_sequence_exact_acc']):.4f} {current['verdict']} {reason}"
        )
        if not accepted or stale >= int(args.stop_after_stale):
            break

    rows = sorted(best_rows, key=lambda row: float(row["D21C_score"]), reverse=True)
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="crystallize-core",
        config={
            "state_dim": state_dim,
            "start_family": str(args.start_family),
            "max_steps": int(args.max_steps),
            "eval_pairs": int(args.eval_pairs),
            "sequence_len": int(args.sequence_len),
            "sequence_count": int(args.sequence_count),
            "seed": int(args.seed),
        },
        path_rows=path_rows,
    )
    print(heatmap)
    print(f"[D21c] wrote {args.out}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    run_self_checks()
    candidate_path = Path(args.candidates)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    with candidate_path.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows = sorted(source_rows, key=lambda row: float(row.get("D21C_score", 0.0)), reverse=True)[: int(args.top_k)]
    specs = []
    for idx, row in enumerate(source_rows):
        specs.append(
            CoreSpec(
                candidate_id=idx,
                visible_dim=int(row["visible_dim"]),
                code_dim=int(row["code_dim"]),
                state_dim=int(row["state_dim"]),
                context_dim=int(row["context_dim"]),
                family=f"confirm_{row['family']}",
                core_edge_budget=int(row["core_edge_budget"]),
                core_entries=parse_entries(str(row["core_entries"])),
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
            "eval_pairs": int(args.eval_pairs),
            "sequence_len": int(args.sequence_len),
            "sequence_count": int(args.sequence_count),
            "seed": int(args.seed),
        },
    )
    best = max(rows, key=lambda row: float(row["D21C_score"]))
    print(
        "[D21c] confirm best "
        f"family={best['family']} state={best['state_dim']} "
        f"edges={best['core_edge_count']} pair={float(best['prev_byte_exact_acc']):.4f} "
        f"long={float(best['long_sequence_exact_acc']):.4f} verdict={best['verdict']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21c tiny recurrent A-block core")
    parser.add_argument("--mode", choices=("baseline-check", "prev-byte-oracle", "core-atlas", "crystallize-core", "confirm"), required=True)
    parser.add_argument("--state-dims", default="8,16,32")
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--core-edge-budgets", default="8,16,24,32")
    parser.add_argument("--core-edge-budget", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--stop-after-stale", type=int, default=512)
    parser.add_argument("--eval-pairs", type=int, default=8192)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--sequence-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-family", default="oracle_prev_byte")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "baseline-check":
        return run_baseline_check(args)
    if args.mode == "prev-byte-oracle":
        return run_prev_byte_oracle(args)
    if args.mode == "core-atlas":
        return run_core_atlas(args)
    if args.mode == "crystallize-core":
        return run_crystallize_core(args)
    if args.mode == "confirm":
        return run_confirm(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
