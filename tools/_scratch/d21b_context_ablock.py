#!/usr/bin/env python3
"""
D21b context-extended reciprocal A-block.

Prototype goal:
    byte bits -> stable reciprocal byte lane -> byte logits
    target/context bits -> sparse context lane -> controllable byte logits

The D21a reciprocal byte lane remains fixed. Context is evaluated as a separate
write/steer lane: with zero context the block must reconstruct the input byte
exactly; with real context it should steer output toward a target byte; with
shuffled/random/fake context it should not score as well against that target.
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
    entries_to_string,
    evaluate_ablock,
    redundant_copy_entries,
    robustness_metrics,
)


ASCII_SHADE = " .:-=+*#%@"
DEFAULT_SEED = 20260502


@dataclass(frozen=True)
class ContextSpec:
    candidate_id: int
    visible_dim: int
    code_dim: int
    context_dim: int
    family: str
    context_edge_budget: int
    context_entries: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class EvalBatch:
    input_patterns: np.ndarray
    target_patterns: np.ndarray
    target_indices: np.ndarray
    context_vectors: np.ndarray
    shuffled_context_vectors: np.ndarray
    random_context_vectors: np.ndarray
    small_noise_context_vectors: np.ndarray
    patterns: np.ndarray


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_entries(raw: str) -> tuple[tuple[int, int, float], ...]:
    entries = []
    for part in str(raw).split():
        code_idx, visible_idx, value = part.split(":")
        entries.append((int(code_idx), int(visible_idx), float(value)))
    return dedupe_entries(entries)


def context_entries_to_string(entries: Sequence[tuple[int, int, float]]) -> str:
    return " ".join(f"{c}:{v}:{value:g}" for c, v, value in entries)


def context_matrix(context_dim: int, visible_dim: int, entries: Iterable[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((context_dim, visible_dim), dtype=np.float32)
    for ctx_idx, visible_idx, value in entries:
        if 0 <= ctx_idx < context_dim and 0 <= visible_idx < visible_dim:
            matrix[ctx_idx, visible_idx] = float(value)
    return matrix


def target_values_for_context_dim(context_dim: int, visible_dim: int) -> np.ndarray:
    if context_dim >= visible_dim:
        return np.arange(1 << visible_dim, dtype=np.int32)
    # Low-dimensional context cannot address all bytes. Use a deterministic
    # repeated-nibble style target set and report its capacity separately.
    count = 1 << context_dim
    values = []
    for value in range(count):
        expanded = 0
        for bit in range(visible_dim):
            if (value >> (bit % context_dim)) & 1:
                expanded |= 1 << bit
        values.append(expanded)
    return np.asarray(values, dtype=np.int32)


def context_vector_for_targets(target_values: np.ndarray, context_dim: int, visible_dim: int) -> np.ndarray:
    rows = []
    for value in target_values:
        bits = [1.0 if ((int(value) >> bit) & 1) else -1.0 for bit in range(visible_dim)]
        row = []
        for idx in range(context_dim):
            if idx < visible_dim:
                row.append(bits[idx])
            else:
                # Extra context dims repeat visible bits with a sign-stable
                # parity feature. It gives redundancy without leaking input.
                source = idx % visible_dim
                sign = 1.0 if ((idx // visible_dim) % 2 == 0) else -1.0
                row.append(sign * bits[source])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def make_eval_batch(
    *,
    visible_dim: int,
    context_dim: int,
    max_pairs: int,
    seed: int,
) -> EvalBatch:
    patterns = all_visible_patterns(visible_dim)
    target_values = target_values_for_context_dim(context_dim, visible_dim)
    context_targets = context_vector_for_targets(target_values, context_dim, visible_dim)

    pairs: list[tuple[int, int, int]] = []
    for input_value in range(1 << visible_dim):
        for target_slot, target_value in enumerate(target_values):
            if int(input_value) != int(target_value):
                pairs.append((input_value, int(target_value), target_slot))

    rng = np.random.default_rng(seed)
    if 0 < max_pairs < len(pairs):
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(idx)] for idx in keep]

    input_indices = np.asarray([item[0] for item in pairs], dtype=np.int32)
    target_indices = np.asarray([item[1] for item in pairs], dtype=np.int32)
    target_slots = np.asarray([item[2] for item in pairs], dtype=np.int32)

    context_vectors = context_targets[target_slots]
    shuffled_slots = target_slots.copy()
    rng.shuffle(shuffled_slots)
    random_context_vectors = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=context_vectors.shape)
    small_noise_context_vectors = rng.normal(loc=0.0, scale=0.15, size=context_vectors.shape).astype(np.float32)

    return EvalBatch(
        input_patterns=patterns[input_indices],
        target_patterns=patterns[target_indices],
        target_indices=target_indices,
        context_vectors=context_vectors,
        shuffled_context_vectors=context_targets[shuffled_slots],
        random_context_vectors=random_context_vectors.astype(np.float32),
        small_noise_context_vectors=small_noise_context_vectors,
        patterns=patterns,
    )


def base_entries_from_json(path: str | None, visible_dim: int, code_dim: int) -> tuple[tuple[int, int, float], ...]:
    if path:
        candidate_path = Path(path)
        if candidate_path.exists():
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate = payload.get("compact_gate_candidate") or payload.get("best_candidate") or {}
            entries = candidate.get("entries")
            if entries:
                return parse_entries(str(entries))
    return redundant_copy_entries(visible_dim, code_dim)


def make_byte_block(visible_dim: int, code_dim: int, entries: tuple[tuple[int, int, float], ...]) -> ReciprocalABlock:
    return ReciprocalABlock.from_entries(visible_dim, code_dim, entries)


def context_identity_entries(context_dim: int, visible_dim: int, weight: float = 4.0) -> tuple[tuple[int, int, float], ...]:
    return tuple((idx, idx, weight) for idx in range(min(context_dim, visible_dim)))


def context_redundant_entries(context_dim: int, visible_dim: int, weight: float = 2.0) -> tuple[tuple[int, int, float], ...]:
    entries: list[tuple[int, int, float]] = []
    for idx in range(context_dim):
        visible_idx = idx % visible_dim
        sign = 1.0 if ((idx // visible_dim) % 2 == 0) else -1.0
        entries.append((idx, visible_idx, sign * weight))
    return dedupe_entries(entries)


def context_random_entries(
    *,
    context_dim: int,
    visible_dim: int,
    edge_budget: int,
    rng: random.Random,
) -> tuple[tuple[int, int, float], ...]:
    used: set[tuple[int, int]] = set()
    entries = []
    weights = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
    tries = 0
    while len(entries) < edge_budget and tries < edge_budget * 100:
        tries += 1
        ctx_idx = rng.randrange(context_dim)
        visible_idx = rng.randrange(visible_dim)
        if (ctx_idx, visible_idx) in used:
            continue
        used.add((ctx_idx, visible_idx))
        entries.append((ctx_idx, visible_idx, rng.choice(weights)))
    return dedupe_entries(entries)


def build_specs(
    *,
    visible_dim: int,
    code_dim: int,
    context_dims: Sequence[int],
    edge_budgets: Sequence[int],
    samples: int,
    seed: int,
) -> list[ContextSpec]:
    specs: list[ContextSpec] = []
    rng = random.Random(seed)

    def add_spec(context_dim: int, edge_budget: int, family: str, entries: Iterable[tuple[int, int, float]]) -> None:
        cleaned = []
        for ctx_idx, visible_idx, value in entries:
            if 0 <= ctx_idx < context_dim and 0 <= visible_idx < visible_dim and abs(float(value)) > 1e-12:
                cleaned.append((ctx_idx, visible_idx, value))
        specs.append(
            ContextSpec(
                candidate_id=len(specs),
                visible_dim=visible_dim,
                code_dim=code_dim,
                context_dim=context_dim,
                family=family,
                context_edge_budget=edge_budget,
                context_entries=dedupe_entries(cleaned),
            )
        )

    for context_dim in context_dims:
        for edge_budget in edge_budgets:
            add_spec(context_dim, edge_budget, "no_context", ())
            add_spec(context_dim, edge_budget, "identity_context", context_identity_entries(context_dim, visible_dim, weight=4.0))
            add_spec(context_dim, edge_budget, "redundant_context", context_redundant_entries(context_dim, visible_dim, weight=2.0))
            add_spec(context_dim, edge_budget, "soft_identity_context", context_identity_entries(context_dim, visible_dim, weight=2.5))
            for _ in range(samples):
                add_spec(
                    context_dim,
                    edge_budget,
                    "random_sparse_context",
                    context_random_entries(
                        context_dim=context_dim,
                        visible_dim=visible_dim,
                        edge_budget=edge_budget,
                        rng=rng,
                    ),
                )
    return specs


def byte_logits_from_decoded(decoded: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    return decoded @ patterns.T


def target_margins(byte_logits: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    rows = np.arange(target_indices.shape[0])
    target_logits = byte_logits[rows, target_indices]
    masked = byte_logits.copy()
    masked[rows, target_indices] = -np.inf
    return target_logits - np.max(masked, axis=1)


def eval_context_control(
    *,
    base_decoded: np.ndarray,
    context_vectors: np.ndarray,
    context_weights: np.ndarray,
    batch: EvalBatch,
) -> dict[str, float]:
    context_bias = context_vectors @ context_weights
    decoded = base_decoded + context_bias
    byte_logits = byte_logits_from_decoded(decoded, batch.patterns)
    pred = np.argmax(byte_logits, axis=1)
    margins = target_margins(byte_logits, batch.target_indices)
    return {
        "success": float(np.mean(pred == batch.target_indices)),
        "margin_mean": float(np.mean(margins)),
        "margin_min": float(np.min(margins)),
    }


def evaluate_context_spec(
    spec: ContextSpec,
    *,
    byte_block: ReciprocalABlock,
    batch: EvalBatch,
    zero_metrics: dict[str, float | int],
    zero_robustness: dict[str, float | int],
) -> dict[str, object]:
    weights = context_matrix(spec.context_dim, spec.visible_dim, spec.context_entries)
    base_codes = byte_block.encode_patterns(batch.input_patterns)
    base_decoded = base_codes @ byte_block.encoder

    real = eval_context_control(
        base_decoded=base_decoded,
        context_vectors=batch.context_vectors,
        context_weights=weights,
        batch=batch,
    )
    shuffled = eval_context_control(
        base_decoded=base_decoded,
        context_vectors=batch.shuffled_context_vectors,
        context_weights=weights,
        batch=batch,
    )
    random_control = eval_context_control(
        base_decoded=base_decoded,
        context_vectors=batch.random_context_vectors,
        context_weights=weights,
        batch=batch,
    )
    no_context = eval_context_control(
        base_decoded=base_decoded,
        context_vectors=np.zeros_like(batch.context_vectors, dtype=np.float32),
        context_weights=weights,
        batch=batch,
    )

    # Small random context is a robustness/noise check, not a steering command.
    small_noise_decoded = base_decoded + (batch.small_noise_context_vectors @ weights)
    small_noise_bits = np.where(small_noise_decoded >= 0.0, 1.0, -1.0)
    small_noise_bit_acc = float(np.mean(small_noise_bits == batch.input_patterns))
    small_noise_exact_acc = float(np.mean(np.all(small_noise_bits == batch.input_patterns, axis=1)))

    fake_success = max(float(shuffled["success"]), float(random_control["success"]), float(no_context["success"]))
    fake_margin = max(float(shuffled["margin_mean"]), float(random_control["margin_mean"]), float(no_context["margin_mean"]))
    context_selectivity = float(real["success"]) - fake_success
    context_margin_selectivity = float(real["margin_mean"]) - fake_margin
    edge_count = int(np.count_nonzero(weights))
    target_count = int(len(np.unique(batch.target_indices)))
    context_capacity_bits = float(math.log2(max(1, target_count)))

    metrics: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "visible_dim": spec.visible_dim,
        "code_dim": spec.code_dim,
        "context_dim": spec.context_dim,
        "family": spec.family,
        "context_edge_budget": spec.context_edge_budget,
        "context_edge_count": edge_count,
        "context_entries": context_entries_to_string(spec.context_entries),
        "context_target_count": target_count,
        "context_capacity_bits": context_capacity_bits,
        "zero_exact_byte_acc": float(zero_metrics["exact_byte_acc"]),
        "zero_bit_acc": float(zero_metrics["bit_acc"]),
        "zero_byte_margin_min": float(zero_metrics["byte_margin_min"]),
        "zero_hidden_collisions": int(zero_metrics["hidden_collisions"]),
        "zero_reciprocity_error": float(zero_metrics["reciprocity_error"]),
        "single_edge_drop_mean_bit": float(zero_robustness["single_edge_drop_mean_bit"]),
        "real_context_success": float(real["success"]),
        "real_context_margin_mean": float(real["margin_mean"]),
        "real_context_margin_min": float(real["margin_min"]),
        "shuffle_context_success": float(shuffled["success"]),
        "shuffle_context_margin_mean": float(shuffled["margin_mean"]),
        "random_context_success": float(random_control["success"]),
        "random_context_margin_mean": float(random_control["margin_mean"]),
        "no_context_target_success": float(no_context["success"]),
        "no_context_target_margin_mean": float(no_context["margin_mean"]),
        "fake_context_success": fake_success,
        "fake_context_margin_mean": fake_margin,
        "context_selectivity": context_selectivity,
        "context_margin_selectivity": context_margin_selectivity,
        "small_noise_context_exact_acc": small_noise_exact_acc,
        "small_noise_context_bit_acc": small_noise_bit_acc,
    }
    metrics["D21B_score"] = d21b_score(metrics)
    metrics["verdict"] = d21b_verdict(metrics)
    return metrics


def d21b_score(metrics: dict[str, object]) -> float:
    safety_penalty = 0.0
    if float(metrics["zero_exact_byte_acc"]) < 1.0:
        safety_penalty += 5.0 * (1.0 - float(metrics["zero_exact_byte_acc"]))
    if float(metrics["small_noise_context_bit_acc"]) < 0.99:
        safety_penalty += 1.0 * (0.99 - float(metrics["small_noise_context_bit_acc"]))
    return (
        2.0 * float(metrics["zero_exact_byte_acc"])
        + 0.5 * float(metrics["zero_bit_acc"])
        + 0.1 * float(metrics["zero_byte_margin_min"])
        + 2.0 * float(metrics["real_context_success"])
        + 1.0 * float(metrics["context_selectivity"])
        + 0.05 * float(metrics["context_margin_selectivity"])
        + 0.05 * float(metrics["context_capacity_bits"])
        - 0.02 * float(metrics["fake_context_success"])
        - 0.001 * float(metrics["context_edge_count"])
        - safety_penalty
    )


def d21b_verdict(metrics: dict[str, object]) -> str:
    zero_ok = (
        float(metrics["zero_exact_byte_acc"]) == 1.0
        and float(metrics["zero_bit_acc"]) == 1.0
        and float(metrics["zero_byte_margin_min"]) > 0.0
        and int(metrics["zero_hidden_collisions"]) == 0
        and float(metrics["zero_reciprocity_error"]) == 0.0
    )
    if not zero_ok:
        return "D21B_CONTEXT_DESTRUCTIVE"
    if float(metrics["real_context_success"]) >= 0.99 and float(metrics["context_selectivity"]) >= 0.95 and float(metrics["real_context_margin_min"]) > 0.0:
        return "D21B_CONTEXT_PASS"
    if float(metrics["real_context_success"]) >= 0.90 and float(metrics["context_selectivity"]) >= 0.75:
        return "D21B_CONTEXT_WEAK_PASS"
    if float(metrics["fake_context_success"]) >= max(0.50, float(metrics["real_context_success"]) - 0.05):
        return "D21B_CONTEXT_FAKE"
    return "D21B_NO_CONTEXT_CAPACITY"


def group_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["context_dim"]), int(row["context_edge_budget"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (family, context_dim, edge_budget), items in sorted(groups.items()):
        best = max(items, key=lambda row: float(row["D21B_score"]))
        pass_count = sum(1 for row in items if str(row["verdict"]) == "D21B_CONTEXT_PASS")
        weak_count = sum(1 for row in items if str(row["verdict"]) == "D21B_CONTEXT_WEAK_PASS")
        summary.append(
            {
                "family": family,
                "context_dim": context_dim,
                "context_edge_budget": edge_budget,
                "count": len(items),
                "pass_count": pass_count,
                "weak_count": weak_count,
                "best_score": float(best["D21B_score"]),
                "best_verdict": str(best["verdict"]),
                "best_real_context_success": float(best["real_context_success"]),
                "best_context_selectivity": float(best["context_selectivity"]),
                "best_context_margin_min": float(best["real_context_margin_min"]),
                "best_context_edge_count": int(best["context_edge_count"]),
            }
        )
    return summary


def make_heatmap(summary: Sequence[dict[str, object]]) -> str:
    dims = sorted({int(row["context_dim"]) for row in summary})
    budgets = sorted({int(row["context_edge_budget"]) for row in summary})
    best_by_cell: dict[tuple[int, int], float] = {}
    verdict_by_cell: dict[tuple[int, int], str] = {}
    for row in summary:
        key = (int(row["context_dim"]), int(row["context_edge_budget"]))
        value = float(row["best_real_context_success"]) * max(0.0, float(row["best_context_selectivity"]))
        if key not in best_by_cell or value > best_by_cell[key]:
            best_by_cell[key] = value
            verdict_by_cell[key] = str(row["best_verdict"])

    values = list(best_by_cell.values()) or [0.0]
    lo = min(values)
    hi = max(values)

    lines = ["D21B context heatmap: brighter = real_context_success * selectivity"]
    lines.append("legend: PASS=P WEAK=W FAKE=F NONE=.")
    header = "ctx\\edge " + " ".join(f"{budget:>5}" for budget in budgets)
    lines.append(header)
    for dim in dims:
        cells = []
        for budget in budgets:
            value = best_by_cell.get((dim, budget), 0.0)
            scaled = 0 if hi <= lo else int(round((value - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
            verdict = verdict_by_cell.get((dim, budget), ".")
            marker = "P" if verdict == "D21B_CONTEXT_PASS" else "W" if verdict == "D21B_CONTEXT_WEAK_PASS" else "F" if verdict == "D21B_CONTEXT_FAKE" else "."
            cells.append(f"{ASCII_SHADE[scaled]}{marker:>4}")
        lines.append(f"{dim:>8} " + " ".join(cells))
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
    sorted_rows = sorted(rows, key=lambda row: float(row["D21B_score"]), reverse=True)
    write_csv(out_dir / "context_ablock_candidates.csv", sorted_rows)
    write_csv(out_dir / "context_family_summary.csv", summary)
    if path_rows is not None:
        write_csv(out_dir / "context_ablock_paths.csv", path_rows)
    control_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "context_dim": row["context_dim"],
            "context_edge_count": row["context_edge_count"],
            "real_context_success": row["real_context_success"],
            "shuffle_context_success": row["shuffle_context_success"],
            "random_context_success": row["random_context_success"],
            "no_context_target_success": row["no_context_target_success"],
            "fake_context_success": row["fake_context_success"],
            "context_selectivity": row["context_selectivity"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows[: min(256, len(sorted_rows))]
    ]
    write_csv(out_dir / "context_control_summary.csv", control_rows)
    (out_dir / "context_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")

    best = sorted_rows[0] if sorted_rows else None
    pass_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21B_CONTEXT_PASS"]
    weak_rows = [row for row in sorted_rows if str(row["verdict"]) == "D21B_CONTEXT_WEAK_PASS"]
    verdict = "D21B_CONTEXT_PASS" if pass_rows else "D21B_CONTEXT_WEAK_PASS" if weak_rows else "D21B_NO_CONTEXT_CAPACITY"
    payload = {
        "verdict": verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "pass_count": len(pass_rows),
        "weak_count": len(weak_rows),
        "best_candidate": best,
        "best_pass_candidate": pass_rows[0] if pass_rows else None,
        "best_weak_candidate": weak_rows[0] if weak_rows else None,
    }
    (out_dir / "context_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# D21B Context A-Block Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{verdict}`",
        "",
        "## Best Candidate",
        "",
    ]
    if best:
        report.extend(
            [
                f"- family: `{best['family']}`",
                f"- context_dim: `{best['context_dim']}`",
                f"- context_edges: `{best['context_edge_count']}`",
                f"- zero_exact_byte_acc: `{float(best['zero_exact_byte_acc']):.6f}`",
                f"- real_context_success: `{float(best['real_context_success']):.6f}`",
                f"- fake_context_success: `{float(best['fake_context_success']):.6f}`",
                f"- context_selectivity: `{float(best['context_selectivity']):.6f}`",
                f"- real_context_margin_min: `{float(best['real_context_margin_min']):.6f}`",
                f"- small_noise_context_bit_acc: `{float(best['small_noise_context_bit_acc']):.6f}`",
                f"- verdict: `{best['verdict']}`",
            ]
        )
    report.extend(["", "## Heatmap", "", "```text", heatmap, "```", ""])
    (out_dir / "D21B_CONTEXT_ABLOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_self_checks() -> None:
    visible_dim = 8
    code_dim = 16
    patterns = all_visible_patterns(visible_dim)
    block = make_byte_block(visible_dim, code_dim, redundant_copy_entries(visible_dim, code_dim))
    metrics = evaluate_ablock(block, patterns)
    assert float(metrics["exact_byte_acc"]) == 1.0
    assert float(metrics["bit_acc"]) == 1.0
    assert float(metrics["reciprocity_error"]) == 0.0

    batch = make_eval_batch(visible_dim=8, context_dim=8, max_pairs=4096, seed=DEFAULT_SEED)
    spec = ContextSpec(0, 8, 16, 8, "identity_context", 8, context_identity_entries(8, 8, weight=4.0))
    row = evaluate_context_spec(
        spec,
        byte_block=block,
        batch=batch,
        zero_metrics=metrics,
        zero_robustness=robustness_metrics(block, patterns),
    )
    assert str(row["verdict"]) == "D21B_CONTEXT_PASS", row
    assert float(row["real_context_success"]) == 1.0
    assert float(row["zero_exact_byte_acc"]) == 1.0


def run_baseline_check(args: argparse.Namespace) -> int:
    run_self_checks()
    visible_dim = int(args.visible)
    code_dim = int(args.code_dim)
    patterns = all_visible_patterns(visible_dim)
    entries = base_entries_from_json(args.d21a_json, visible_dim, code_dim)
    block = make_byte_block(visible_dim, code_dim, entries)
    metrics = {**evaluate_ablock(block, patterns), **robustness_metrics(block, patterns)}
    row = {
        "mode": "baseline-check",
        "visible_dim": visible_dim,
        "code_dim": code_dim,
        "entries": entries_to_string(entries),
        **metrics,
        "gate_reproduced": (
            float(metrics["exact_byte_acc"]) == 1.0
            and float(metrics["bit_acc"]) == 1.0
            and float(metrics["reciprocity_error"]) == 0.0
        ),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "baseline_check.csv", [row])
    (out_dir / "context_top.json").write_text(json.dumps({"verdict": "D21B_BASELINE_REPRODUCED", "baseline": row}, indent=2), encoding="utf-8")
    (out_dir / "D21B_CONTEXT_ABLOCK_REPORT.md").write_text(
        "# D21B Baseline Check\n\n"
        f"Verdict: `D21B_BASELINE_REPRODUCED`\n\n"
        f"exact_byte_acc: `{float(metrics['exact_byte_acc']):.6f}`\n\n"
        f"bit_acc: `{float(metrics['bit_acc']):.6f}`\n\n"
        f"reciprocity_error: `{float(metrics['reciprocity_error']):.6f}`\n",
        encoding="utf-8",
    )
    print("[D21b] baseline reproduced exact=1.0000 bit=1.0000 reciprocity=0")
    return 0


def evaluate_specs(args: argparse.Namespace, specs: Sequence[ContextSpec]) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    visible_dim = int(args.visible)
    code_dim = int(args.code_dim)
    patterns = all_visible_patterns(visible_dim)
    entries = base_entries_from_json(args.d21a_json, visible_dim, code_dim)
    block = make_byte_block(visible_dim, code_dim, entries)
    zero_metrics = evaluate_ablock(block, patterns)
    zero_robustness = robustness_metrics(block, patterns)

    rows: list[dict[str, object]] = []
    batches = {
        context_dim: make_eval_batch(
            visible_dim=visible_dim,
            context_dim=context_dim,
            max_pairs=int(args.eval_pairs),
            seed=int(args.seed) + context_dim,
        )
        for context_dim in sorted({spec.context_dim for spec in specs})
    }
    for idx, spec in enumerate(specs, start=1):
        rows.append(
            evaluate_context_spec(
                spec,
                byte_block=block,
                batch=batches[spec.context_dim],
                zero_metrics=zero_metrics,
                zero_robustness=zero_robustness,
            )
        )
        if idx % 1000 == 0:
            best = max(rows, key=lambda row: float(row["D21B_score"]))
            print(
                f"[D21b] evaluated {idx}/{len(specs)} "
                f"best={best['family']} ctx={best['context_dim']} "
                f"real={float(best['real_context_success']):.4f} "
                f"fake={float(best['fake_context_success']):.4f} "
                f"sel={float(best['context_selectivity']):.4f}"
            )
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    return rows, summary, heatmap


def run_context_atlas(args: argparse.Namespace) -> int:
    run_self_checks()
    specs = build_specs(
        visible_dim=int(args.visible),
        code_dim=int(args.code_dim),
        context_dims=parse_int_list(args.context_dims),
        edge_budgets=parse_int_list(args.context_edge_budgets),
        samples=int(args.samples),
        seed=int(args.seed),
    )
    rows, summary, heatmap = evaluate_specs(args, specs)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="context-atlas",
        config={
            "visible": int(args.visible),
            "code_dim": int(args.code_dim),
            "context_dims": parse_int_list(args.context_dims),
            "context_edge_budgets": parse_int_list(args.context_edge_budgets),
            "samples": int(args.samples),
            "eval_pairs": int(args.eval_pairs),
            "seed": int(args.seed),
        },
    )
    print(heatmap)
    best = max(rows, key=lambda row: float(row["D21B_score"]))
    print(
        "[D21b] best "
        f"family={best['family']} ctx={best['context_dim']} E={best['context_edge_count']} "
        f"real={float(best['real_context_success']):.4f} "
        f"fake={float(best['fake_context_success']):.4f} "
        f"selectivity={float(best['context_selectivity']):.4f} "
        f"verdict={best['verdict']}"
    )
    return 0


def mutate_context_entries(
    entries: tuple[tuple[int, int, float], ...],
    *,
    context_dim: int,
    visible_dim: int,
    edge_budget: int,
    rng: random.Random,
) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    used = {(c, v) for c, v, _ in current}
    ops = ("add", "remove", "flip", "reweight")
    if len(current) >= edge_budget:
        ops = ("remove", "flip", "reweight")
    op = rng.choice(ops)
    if op == "add":
        for _ in range(100):
            ctx_idx = rng.randrange(context_dim)
            visible_idx = rng.randrange(visible_dim)
            if (ctx_idx, visible_idx) not in used:
                current.append((ctx_idx, visible_idx, rng.choice((-4.0, -2.0, 2.0, 4.0))))
                return dedupe_entries(current)
    if op == "remove" and current:
        del current[rng.randrange(len(current))]
        return dedupe_entries(current)
    if op == "flip" and current:
        idx = rng.randrange(len(current))
        old_ctx, old_visible, old_value = current[idx]
        used.discard((old_ctx, old_visible))
        for _ in range(100):
            ctx_idx = rng.randrange(context_dim)
            visible_idx = rng.randrange(visible_dim)
            if (ctx_idx, visible_idx) not in used:
                current[idx] = (ctx_idx, visible_idx, old_value)
                return dedupe_entries(current)
    if op == "reweight" and current:
        idx = rng.randrange(len(current))
        ctx_idx, visible_idx, _old_value = current[idx]
        current[idx] = (ctx_idx, visible_idx, rng.choice((-4.0, -3.0, -2.0, 2.0, 3.0, 4.0)))
        return dedupe_entries(current)
    return entries


def start_context_entries(start_family: str, context_dim: int, visible_dim: int, edge_budget: int, seed: int) -> tuple[tuple[int, int, float], ...]:
    if start_family == "identity_context":
        return context_identity_entries(context_dim, visible_dim, weight=4.0)
    if start_family == "soft_identity_context":
        return context_identity_entries(context_dim, visible_dim, weight=2.5)
    if start_family == "redundant_context":
        return context_redundant_entries(context_dim, visible_dim, weight=2.0)
    if start_family == "random_sparse_context":
        return context_random_entries(
            context_dim=context_dim,
            visible_dim=visible_dim,
            edge_budget=edge_budget,
            rng=random.Random(seed),
        )
    raise ValueError(f"unknown start family: {start_family}")


def run_mutate_context(args: argparse.Namespace) -> int:
    run_self_checks()
    visible_dim = int(args.visible)
    code_dim = int(args.code_dim)
    context_dim = int(args.context_dim)
    edge_budget = int(args.context_edge_budget)
    rng = random.Random(int(args.seed))
    current_entries = start_context_entries(str(args.start_family), context_dim, visible_dim, edge_budget, int(args.seed))

    def eval_one(candidate_id: int, family: str, entries: tuple[tuple[int, int, float], ...]) -> dict[str, object]:
        specs = [ContextSpec(candidate_id, visible_dim, code_dim, context_dim, family, edge_budget, entries)]
        rows, _summary, _heatmap = evaluate_specs(args, specs)
        return rows[0]

    current = eval_one(0, f"start_{args.start_family}", current_entries)
    best_rows = [current]
    path_rows = [{**current, "step": 0, "accepted": True, "reason": "start"}]
    print(
        f"[D21b] mutate start score={float(current['D21B_score']):.6f} "
        f"real={float(current['real_context_success']):.4f} "
        f"fake={float(current['fake_context_success']):.4f} verdict={current['verdict']}"
    )

    stale = 0
    for step in range(1, int(args.max_steps) + 1):
        proposals = []
        for worker_idx in range(max(1, int(args.workers))):
            local_rng = random.Random(rng.randrange(1 << 30) + worker_idx)
            entries = mutate_context_entries(
                current_entries,
                context_dim=context_dim,
                visible_dim=visible_dim,
                edge_budget=edge_budget,
                rng=local_rng,
            )
            proposals.append(eval_one(step * 1000 + worker_idx, "proposal", entries))
        best_prop = max(proposals, key=lambda row: float(row["D21B_score"]))
        best_rows.extend(proposals)

        accepted = False
        reason = "reject"
        if (
            float(best_prop["zero_exact_byte_acc"]) == 1.0
            and float(best_prop["D21B_score"]) > float(current["D21B_score"]) + 1e-9
            and float(best_prop["context_selectivity"]) >= float(current["context_selectivity"]) - 1e-9
        ):
            current = best_prop
            current_entries = parse_entries(str(best_prop["context_entries"]))
            accepted = True
            stale = 0
            reason = "score_and_selectivity_improve"
        else:
            stale += len(proposals)
        path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
        if step % 100 == 0 or accepted:
            print(
                f"[D21b][{step}] score={float(current['D21B_score']):.6f} "
                f"edges={current['context_edge_count']} real={float(current['real_context_success']):.4f} "
                f"fake={float(current['fake_context_success']):.4f} "
                f"sel={float(current['context_selectivity']):.4f} {current['verdict']} {reason}"
            )
        if stale >= int(args.stop_after_stale):
            print(f"[D21b] stop stale={stale}")
            break

    rows = sorted(best_rows, key=lambda row: float(row["D21B_score"]), reverse=True)
    summary = group_summary(rows)
    heatmap = make_heatmap(summary)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heatmap,
        mode="mutate-context",
        config={
            "visible": visible_dim,
            "code_dim": code_dim,
            "context_dim": context_dim,
            "context_edge_budget": edge_budget,
            "start_family": str(args.start_family),
            "max_steps": int(args.max_steps),
            "eval_pairs": int(args.eval_pairs),
            "seed": int(args.seed),
        },
        path_rows=path_rows,
    )
    print(heatmap)
    print(f"[D21b] wrote {args.out}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    run_self_checks()
    candidate_path = Path(args.candidates)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    with candidate_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
    source_rows = sorted(source_rows, key=lambda row: float(row.get("D21B_score", 0.0)), reverse=True)[: int(args.top_k)]
    specs = []
    for idx, row in enumerate(source_rows):
        specs.append(
            ContextSpec(
                candidate_id=idx,
                visible_dim=int(row["visible_dim"]),
                code_dim=int(row["code_dim"]),
                context_dim=int(row["context_dim"]),
                family=f"confirm_{row['family']}",
                context_edge_budget=int(row["context_edge_budget"]),
                context_entries=parse_entries(str(row["context_entries"])),
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
            "seed": int(args.seed),
        },
    )
    best = max(rows, key=lambda row: float(row["D21B_score"]))
    print(
        "[D21b] confirm best "
        f"family={best['family']} ctx={best['context_dim']} "
        f"real={float(best['real_context_success']):.4f} "
        f"fake={float(best['fake_context_success']):.4f} verdict={best['verdict']}"
    )
    return 0


def run_compare_old(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_path = Path(args.old_log)
    row = {
        "old_log": str(old_path),
        "old_log_exists": old_path.exists(),
        "note": "D21b compare-old is a placeholder summary; use exact byte/bit metrics from D21a/D21b reports for fair comparison.",
    }
    write_csv(out_dir / "compare_old.csv", [row])
    (out_dir / "D21B_CONTEXT_ABLOCK_REPORT.md").write_text(
        "# D21B Compare Old\n\n"
        "This mode records the old artifact path only. The useful comparison is exact byte reconstruction, "
        "byte margin, context selectivity, and controls from the D21 reports.\n",
        encoding="utf-8",
    )
    print(f"[D21b] compare-old wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21b context-extended reciprocal A-block")
    parser.add_argument("--mode", choices=("baseline-check", "context-atlas", "mutate-context", "crystallize-context", "confirm", "compare-old"), required=True)
    parser.add_argument("--visible", type=int, default=8)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--context-dims", default="4,8,16")
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--context-edge-budgets", default="4,8,16,32")
    parser.add_argument("--context-edge-budget", type=int, default=16)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--stop-after-stale", type=int, default=2000)
    parser.add_argument("--eval-pairs", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-family", default="identity_context")
    parser.add_argument("--d21a-json", default="")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--old-log", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "baseline-check":
        return run_baseline_check(args)
    if args.mode == "context-atlas":
        return run_context_atlas(args)
    if args.mode in ("mutate-context", "crystallize-context"):
        return run_mutate_context(args)
    if args.mode == "confirm":
        return run_confirm(args)
    if args.mode == "compare-old":
        return run_compare_old(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
