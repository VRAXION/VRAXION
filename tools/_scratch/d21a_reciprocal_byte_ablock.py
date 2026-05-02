#!/usr/bin/env python3
"""
D21a reciprocal A-block byte encoder.

Prototype goal:
    byte -> 8 visible bits -> sparse reciprocal code -> 8 bit logits -> byte logits

The decoder is always encoder.T. There is no independent decoder weight matrix,
so every visible/code edge is bidirectional by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ASCII_SHADE = " .:-=+*#%@"
_PATTERNS: np.ndarray | None = None


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: int
    visible_dim: int
    code_dim: int
    family: str
    edge_budget: int
    entries: tuple[tuple[int, int, float], ...]


class ReciprocalABlock:
    """Sparse byte A-block with tied reciprocal decoder."""

    def __init__(self, visible_dim: int, code_dim: int, encoder: np.ndarray):
        self.visible_dim = int(visible_dim)
        self.code_dim = int(code_dim)
        self.encoder = np.asarray(encoder, dtype=np.float32).reshape(self.code_dim, self.visible_dim)

    @classmethod
    def from_entries(
        cls,
        visible_dim: int,
        code_dim: int,
        entries: Iterable[tuple[int, int, float]],
    ) -> "ReciprocalABlock":
        encoder = np.zeros((code_dim, visible_dim), dtype=np.float32)
        for code_idx, visible_idx, value in entries:
            if 0 <= code_idx < code_dim and 0 <= visible_idx < visible_dim:
                encoder[code_idx, visible_idx] = float(value)
        return cls(visible_dim, code_dim, encoder)

    @property
    def decoder(self) -> np.ndarray:
        return self.encoder.T

    def encode_patterns(self, patterns: np.ndarray) -> np.ndarray:
        return patterns @ self.encoder.T

    def encode_byte(self, byte_value: int) -> np.ndarray:
        patterns = all_visible_patterns(self.visible_dim)
        return self.encode_patterns(patterns[int(byte_value) : int(byte_value) + 1])[0]

    def decode(self, code: np.ndarray) -> np.ndarray:
        return np.asarray(code, dtype=np.float32) @ self.encoder

    def byte_logits(self, code: np.ndarray) -> np.ndarray:
        decoded = self.decode(code)
        patterns = all_visible_patterns(self.visible_dim)
        return patterns @ decoded

    def reciprocity_error(self) -> float:
        return float(np.max(np.abs(self.decoder - self.encoder.T)))


def all_visible_patterns(visible_dim: int) -> np.ndarray:
    rows = []
    for value in range(1 << visible_dim):
        rows.append([1.0 if ((value >> bit) & 1) else -1.0 for bit in range(visible_dim)])
    return np.asarray(rows, dtype=np.float32)


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def init_worker(patterns: np.ndarray) -> None:
    global _PATTERNS
    _PATTERNS = patterns


def dedupe_entries(entries: Iterable[tuple[int, int, float]]) -> tuple[tuple[int, int, float], ...]:
    dedup: dict[tuple[int, int], float] = {}
    for code_idx, visible_idx, value in entries:
        if abs(float(value)) > 1e-12:
            dedup[(int(code_idx), int(visible_idx))] = float(value)
    return tuple((c, v, value) for (c, v), value in sorted(dedup.items()))


def entries_to_string(entries: Sequence[tuple[int, int, float]]) -> str:
    return " ".join(f"{c}:{v}:{value:g}" for c, v, value in entries)


def identity_entries(visible_dim: int) -> tuple[tuple[int, int, float], ...]:
    return tuple((idx, idx, 1.0) for idx in range(visible_dim))


def redundant_copy_entries(visible_dim: int, code_dim: int) -> tuple[tuple[int, int, float], ...]:
    if code_dim < visible_dim * 2:
        return identity_entries(visible_dim)
    entries = []
    for visible_idx in range(visible_dim):
        entries.append((visible_idx, visible_idx, 1.0))
        entries.append((visible_dim + visible_idx, visible_idx, 1.0))
    return tuple(entries)


def random_sign(rng: random.Random) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def build_spec(
    specs: list[CandidateSpec],
    *,
    visible_dim: int,
    code_dim: int,
    family: str,
    edge_budget: int,
    entries: Iterable[tuple[int, int, float]],
) -> None:
    cleaned = []
    for code_idx, visible_idx, value in entries:
        if 0 <= code_idx < code_dim and 0 <= visible_idx < visible_dim and abs(float(value)) > 1e-12:
            cleaned.append((code_idx, visible_idx, value))
    specs.append(
        CandidateSpec(
            candidate_id=len(specs),
            visible_dim=visible_dim,
            code_dim=code_dim,
            family=family,
            edge_budget=edge_budget,
            entries=dedupe_entries(cleaned),
        )
    )


def evaluate_ablock(block: ReciprocalABlock, patterns: np.ndarray) -> dict[str, float | int]:
    codes = block.encode_patterns(patterns)
    decoded = codes @ block.encoder
    pred_bits = np.where(decoded >= 0.0, 1.0, -1.0)

    byte_logits = decoded @ patterns.T
    target_idx = np.arange(patterns.shape[0])
    pred_byte = np.argmax(byte_logits, axis=1)
    target_logits = byte_logits[target_idx, target_idx]
    masked = byte_logits.copy()
    masked[target_idx, target_idx] = -np.inf
    byte_margins = target_logits - np.max(masked, axis=1)

    code_rows = [tuple(float(x) for x in row) for row in codes]
    hidden_collisions = int(len(code_rows) - len(set(code_rows)))

    edge_count = int(np.count_nonzero(block.encoder))
    return {
        "edge_count": edge_count,
        "rank": int(np.linalg.matrix_rank(block.encoder)),
        "bit_acc": float(np.mean(pred_bits == patterns)),
        "exact_byte_acc": float(np.mean(np.all(pred_bits == patterns, axis=1))),
        "byte_argmax_acc": float(np.mean(pred_byte == target_idx)),
        "byte_margin_min": float(np.min(byte_margins)),
        "byte_margin_mean": float(np.mean(byte_margins)),
        "mse": float(np.mean((decoded - patterns) ** 2)),
        "max_abs_drift": float(np.max(np.abs(decoded - patterns))),
        "mean_abs_logit": float(np.mean(np.abs(decoded))),
        "unique_hidden_codes": int(len(set(code_rows))),
        "hidden_collisions": hidden_collisions,
        "reciprocity_error": block.reciprocity_error(),
    }


def robustness_metrics(block: ReciprocalABlock, patterns: np.ndarray) -> dict[str, float | int]:
    edges = list(zip(*np.nonzero(block.encoder)))
    if not edges:
        return {
            "single_edge_drop_min_exact": 0.0,
            "single_edge_drop_mean_exact": 0.0,
            "single_edge_drop_min_bit": 0.0,
            "single_edge_drop_mean_bit": 0.0,
            "single_edge_drop_min_byte_margin": 0.0,
        }

    exacts: list[float] = []
    bits: list[float] = []
    margins: list[float] = []
    for code_idx, visible_idx in edges:
        encoder = block.encoder.copy()
        encoder[code_idx, visible_idx] = 0.0
        metrics = evaluate_ablock(ReciprocalABlock(block.visible_dim, block.code_dim, encoder), patterns)
        exacts.append(float(metrics["exact_byte_acc"]))
        bits.append(float(metrics["bit_acc"]))
        margins.append(float(metrics["byte_margin_min"]))

    return {
        "single_edge_drop_min_exact": float(min(exacts)),
        "single_edge_drop_mean_exact": float(np.mean(exacts)),
        "single_edge_drop_min_bit": float(min(bits)),
        "single_edge_drop_mean_bit": float(np.mean(bits)),
        "single_edge_drop_min_byte_margin": float(min(margins)),
    }


def a_score(metrics: dict[str, float | int]) -> float:
    return (
        2.0 * float(metrics["exact_byte_acc"])
        + 0.5 * float(metrics["bit_acc"])
        + 0.25 * float(metrics["byte_margin_min"])
        + 0.5 * float(metrics["single_edge_drop_mean_bit"])
        - 0.002 * float(metrics["hidden_collisions"])
        - 0.0005 * float(metrics["mse"])
        - 0.0001 * float(metrics["edge_count"])
    )


def gate_pass(metrics: dict[str, float | int]) -> bool:
    return (
        float(metrics["exact_byte_acc"]) == 1.0
        and float(metrics["bit_acc"]) == 1.0
        and int(metrics["hidden_collisions"]) == 0
        and float(metrics["single_edge_drop_mean_bit"]) >= 0.99
        and float(metrics["byte_margin_min"]) > 0.0
        and float(metrics["reciprocity_error"]) == 0.0
    )


def eval_spec(spec: CandidateSpec) -> dict[str, object]:
    if _PATTERNS is None:
        raise RuntimeError("patterns not initialized")
    block = ReciprocalABlock.from_entries(spec.visible_dim, spec.code_dim, spec.entries)
    metrics = evaluate_ablock(block, _PATTERNS)
    metrics.update(robustness_metrics(block, _PATTERNS))
    row: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "visible_dim": spec.visible_dim,
        "code_dim": spec.code_dim,
        "family": spec.family,
        "edge_budget": spec.edge_budget,
        "entries": entries_to_string(spec.entries),
    }
    row.update(metrics)
    row["A_score"] = a_score(metrics)
    row["gate_pass"] = gate_pass(metrics)
    return row


def generate_family_specs(
    *,
    visible_dim: int,
    code_dims: Sequence[int],
    edge_budgets: Sequence[int],
    samples: int,
    seed: int,
) -> list[CandidateSpec]:
    rng = random.Random(seed)
    specs: list[CandidateSpec] = []

    for code_dim in code_dims:
        for edge_budget in edge_budgets:
            if code_dim >= visible_dim and edge_budget >= visible_dim:
                build_spec(
                    specs,
                    visible_dim=visible_dim,
                    code_dim=code_dim,
                    family="identity",
                    edge_budget=edge_budget,
                    entries=identity_entries(visible_dim),
                )

                for _ in range(max(1, samples // 4)):
                    rows = rng.sample(range(code_dim), visible_dim)
                    entries = [(rows[v], v, random_sign(rng)) for v in range(visible_dim)]
                    build_spec(
                        specs,
                        visible_dim=visible_dim,
                        code_dim=code_dim,
                        family="permutation",
                        edge_budget=edge_budget,
                        entries=entries,
                    )

            if code_dim >= visible_dim * 2 and edge_budget >= visible_dim * 2:
                build_spec(
                    specs,
                    visible_dim=visible_dim,
                    code_dim=code_dim,
                    family="redundant_copy_2x",
                    edge_budget=edge_budget,
                    entries=redundant_copy_entries(visible_dim, code_dim),
                )

            if code_dim >= visible_dim and edge_budget > visible_dim:
                for weight in (0.25, 0.5, 1.0):
                    for _ in range(max(1, samples // 4)):
                        entries = list(identity_entries(visible_dim))
                        used = {(c, v) for c, v, _ in entries}
                        attempts = 0
                        while len(entries) < edge_budget and attempts < edge_budget * 30:
                            attempts += 1
                            code_idx = rng.randrange(code_dim)
                            visible_idx = rng.randrange(visible_dim)
                            if (code_idx, visible_idx) in used:
                                continue
                            used.add((code_idx, visible_idx))
                            entries.append((code_idx, visible_idx, weight * random_sign(rng)))
                        if len(entries) == edge_budget:
                            build_spec(
                                specs,
                                visible_dim=visible_dim,
                                code_dim=code_dim,
                                family=f"near_identity_w{weight:g}",
                                edge_budget=edge_budget,
                                entries=entries,
                            )

            for _ in range(samples):
                positions = rng.sample(range(code_dim * visible_dim), min(edge_budget, code_dim * visible_dim))
                entries = [(p // visible_dim, p % visible_dim, random_sign(rng)) for p in positions]
                build_spec(
                    specs,
                    visible_dim=visible_dim,
                    code_dim=code_dim,
                    family="random_sparse",
                    edge_budget=edge_budget,
                    entries=entries,
                )

    return specs


def group_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["code_dim"]), int(row["edge_budget"]), str(row["family"]))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (code_dim, edge_budget, family), group in sorted(groups.items()):
        best = max(group, key=lambda item: float(item["A_score"]))
        pass_count = sum(1 for item in group if str(item["gate_pass"]).lower() == "true" or item["gate_pass"] is True)
        out.append(
            {
                "code_dim": code_dim,
                "edge_budget": edge_budget,
                "family": family,
                "count": len(group),
                "gate_pass_count": pass_count,
                "gate_pass_rate": pass_count / max(1, len(group)),
                "best_A_score": float(best["A_score"]),
                "best_exact_byte_acc": float(best["exact_byte_acc"]),
                "best_bit_acc": float(best["bit_acc"]),
                "best_byte_margin_min": float(best["byte_margin_min"]),
                "best_single_edge_drop_mean_bit": float(best["single_edge_drop_mean_bit"]),
                "best_hidden_collisions": int(best["hidden_collisions"]),
                "best_candidate_id": int(best["candidate_id"]),
            }
        )
    return out


def shade(value: float, lo: float, hi: float) -> str:
    if hi <= lo:
        return ASCII_SHADE[-1]
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return ASCII_SHADE[int(round(t * (len(ASCII_SHADE) - 1)))]


def make_heatmap(summary: Sequence[dict[str, object]], code_dims: Sequence[int], edge_budgets: Sequence[int]) -> str:
    best_by_cell: dict[tuple[int, int], float] = {}
    label_by_cell: dict[tuple[int, int], str] = {}
    for row in summary:
        key = (int(row["code_dim"]), int(row["edge_budget"]))
        value = float(row["best_A_score"])
        if key not in best_by_cell or value > best_by_cell[key]:
            best_by_cell[key] = value
            label_by_cell[key] = str(row["family"])

    vals = list(best_by_cell.values())
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    lines = [
        "ASCII heatmap: x=edge_budget, y=code_dim, brighter=best A_score",
        f"scale lo={lo:.6f} hi={hi:.6f}",
        "code\\edges " + " ".join(f"{edge:>4}" for edge in edge_budgets),
    ]
    for code_dim in code_dims:
        cells = []
        for edge_budget in edge_budgets:
            value = best_by_cell.get((code_dim, edge_budget))
            cells.append(" " if value is None else shade(value, lo, hi))
        lines.append(f"{code_dim:>10} " + "    ".join(cells))
    lines.append("")
    lines.append("Best family per cell:")
    for code_dim in code_dims:
        parts = [f"E{edge}:{label_by_cell.get((code_dim, edge), '-')}" for edge in edge_budgets]
        lines.append(f"C{code_dim}: " + ", ".join(parts))
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    heatmap: str,
    mode: str,
    config: dict[str, object],
    path_rows: list[dict[str, object]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda item: float(item["A_score"]), reverse=True)
    best = rows_sorted[0] if rows_sorted else None
    gate_rows = [row for row in rows_sorted if row.get("gate_pass") is True]
    compact_gate = (
        min(gate_rows, key=lambda item: (int(item["edge_count"]), -float(item["A_score"])))
        if gate_rows
        else None
    )
    payload = {
        "verdict": "D21A_RECIPROCAL_ABLOCK_PASS" if best and best.get("gate_pass") else "D21A_RECIPROCAL_ABLOCK_NO_PASS",
        "mode": mode,
        "config": config,
        "candidate_count": len(rows_sorted),
        "gate_pass_count": len(gate_rows),
        "best_candidate": best,
        "compact_gate_candidate": compact_gate,
        "final_path_candidate": path_rows[-1] if path_rows else None,
    }

    write_csv(out_dir / "ablock_candidates.csv", rows_sorted)
    write_csv(out_dir / "ablock_family_summary.csv", summary)
    if path_rows is not None:
        write_csv(out_dir / "ablock_paths.csv", path_rows)
    (out_dir / "ablock_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    (out_dir / "ablock_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# D21A Reciprocal A-Block Report",
        "",
        "Decoder is always encoder.T, so the block is reciprocal by construction.",
        "",
        "## Summary",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Heatmap",
        "```text",
        heatmap,
        "```",
    ]
    (out_dir / "D21A_RECIPROCAL_ABLOCK_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_self_checks() -> None:
    patterns8 = all_visible_patterns(8)
    identity = ReciprocalABlock.from_entries(8, 8, identity_entries(8))
    identity_metrics = evaluate_ablock(identity, patterns8)
    identity_metrics.update(robustness_metrics(identity, patterns8))
    if float(identity_metrics["exact_byte_acc"]) != 1.0:
        raise AssertionError("identity 8D reciprocal block failed exact reconstruction")
    if identity.reciprocity_error() != 0.0:
        raise AssertionError("identity reciprocity error is non-zero")

    redundant = ReciprocalABlock.from_entries(8, 16, redundant_copy_entries(8, 16))
    redundant_metrics = evaluate_ablock(redundant, patterns8)
    redundant_metrics.update(robustness_metrics(redundant, patterns8))
    if float(redundant_metrics["exact_byte_acc"]) != 1.0:
        raise AssertionError("16D redundant block failed exact reconstruction")
    if float(redundant_metrics["single_edge_drop_mean_bit"]) < 1.0:
        raise AssertionError("16D redundant block failed single-edge-drop robustness")
    if float(redundant_metrics["byte_margin_min"]) <= 0.0:
        raise AssertionError("16D redundant block failed positive byte margin")


def run_family_atlas(args: argparse.Namespace) -> int:
    run_self_checks()
    visible_dim = int(args.visible)
    code_dims = parse_int_list(args.code_dims)
    edge_budgets = parse_int_list(args.edge_budgets)
    patterns = all_visible_patterns(visible_dim)
    specs = generate_family_specs(
        visible_dim=visible_dim,
        code_dims=code_dims,
        edge_budgets=edge_budgets,
        samples=int(args.samples),
        seed=int(args.seed),
    )
    print(f"[D21a] family-atlas candidates={len(specs)} workers={args.workers}")

    if int(args.workers) <= 1:
        init_worker(patterns)
        rows = [eval_spec(spec) for spec in specs]
    else:
        with mp.Pool(int(args.workers), initializer=init_worker, initargs=(patterns,)) as pool:
            rows = list(pool.imap_unordered(eval_spec, specs, chunksize=64))

    summary = group_summary(rows)
    heat = make_heatmap(summary, code_dims, edge_budgets)
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heat,
        mode="family-atlas",
        config={
            "visible": visible_dim,
            "code_dims": code_dims,
            "edge_budgets": edge_budgets,
            "samples": int(args.samples),
            "seed": int(args.seed),
        },
    )
    print(heat)
    best = max(rows, key=lambda item: float(item["A_score"]))
    print(
        "[D21a] best "
        f"family={best['family']} C={best['code_dim']} E={best['edge_count']} "
        f"exact={float(best['exact_byte_acc']):.4f} bit={float(best['bit_acc']):.4f} "
        f"margin_min={float(best['byte_margin_min']):.4f} "
        f"drop_bit={float(best['single_edge_drop_mean_bit']):.4f} "
        f"score={float(best['A_score']):.6f}"
    )
    return 0


def mutate_entries(
    entries: tuple[tuple[int, int, float], ...],
    *,
    visible_dim: int,
    code_dim: int,
    rng: random.Random,
    crystallize: bool,
) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    used = {(c, v) for c, v, _ in current}
    op = rng.choice(("remove", "flip", "reweight") if crystallize else ("add", "remove", "flip", "reweight"))

    if op == "add":
        for _ in range(100):
            c = rng.randrange(code_dim)
            v = rng.randrange(visible_dim)
            if (c, v) not in used:
                current.append((c, v, random_sign(rng)))
                return dedupe_entries(current)
    if op == "remove" and current:
        idx = rng.randrange(len(current))
        del current[idx]
        return dedupe_entries(current)
    if op == "flip" and current:
        idx = rng.randrange(len(current))
        c_old, v_old, value = current[idx]
        used.discard((c_old, v_old))
        for _ in range(100):
            c = rng.randrange(code_dim)
            v = rng.randrange(visible_dim)
            if (c, v) not in used:
                current[idx] = (c, v, value)
                return dedupe_entries(current)
    if op == "reweight" and current:
        idx = rng.randrange(len(current))
        c, v, value = current[idx]
        choices = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
        current[idx] = (c, v, rng.choice(choices))
        return dedupe_entries(current)
    return entries


def start_entries(start_family: str, visible_dim: int, code_dim: int, seed: int) -> tuple[tuple[int, int, float], ...]:
    if start_family == "identity":
        return identity_entries(visible_dim)
    if start_family == "redundant_copy_2x":
        return redundant_copy_entries(visible_dim, code_dim)
    if start_family == "permutation":
        rng = random.Random(seed)
        rows = rng.sample(range(code_dim), visible_dim)
        return dedupe_entries((rows[v], v, random_sign(rng)) for v in range(visible_dim))
    raise ValueError(f"unknown start family: {start_family}")


def run_local_search(args: argparse.Namespace, *, crystallize: bool) -> int:
    run_self_checks()
    visible_dim = int(args.visible)
    code_dim = int(args.code_dim)
    patterns = all_visible_patterns(visible_dim)
    rng = random.Random(int(args.seed))
    current_entries = start_entries(str(args.start_family), visible_dim, code_dim, int(args.seed))

    def eval_entries(candidate_id: int, family: str, entries: tuple[tuple[int, int, float], ...]) -> dict[str, object]:
        spec = CandidateSpec(candidate_id, visible_dim, code_dim, family, len(entries), entries)
        init_worker(patterns)
        return eval_spec(spec)

    current = eval_entries(0, f"start_{args.start_family}", current_entries)
    best_rows = [current]
    path_rows = [{**current, "step": 0, "accepted": True, "reason": "start"}]
    print(
        f"[D21a] {'crystallize' if crystallize else 'mutate-climb'} start "
        f"score={float(current['A_score']):.6f} edges={current['edge_count']} gate={current['gate_pass']}"
    )

    if crystallize:
        # Crystallization is a prune/polish question, not a random walk. Try every
        # single-edge removal and small reweight around the current candidate,
        # then accept only gate-preserving simplifications/improvements.
        for step in range(1, int(args.max_steps) + 1):
            proposals: list[dict[str, object]] = []
            parsed = list(current_entries)
            for drop_idx in range(len(parsed)):
                entries = parsed[:drop_idx] + parsed[drop_idx + 1 :]
                proposals.append(eval_entries(step * 1000 + len(proposals), "drop_edge", dedupe_entries(entries)))
            for weight_idx, (code_idx, visible_idx, old_value) in enumerate(parsed):
                for new_value in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                    if abs(float(new_value) - float(old_value)) < 1e-12:
                        continue
                    entries = parsed.copy()
                    entries[weight_idx] = (code_idx, visible_idx, new_value)
                    proposals.append(eval_entries(step * 1000 + len(proposals), "reweight", dedupe_entries(entries)))

            best_rows.extend(proposals)
            gated = [row for row in proposals if row["gate_pass"] is True]
            accepted = False
            reason = "no_gate_preserving_simplification"
            if gated:
                best_prop = max(
                    gated,
                    key=lambda row: (
                        -int(row["edge_count"]),
                        float(row["A_score"]),
                        float(row["byte_margin_min"]),
                    ),
                )
                if int(best_prop["edge_count"]) < int(current["edge_count"]) or float(best_prop["A_score"]) > float(current["A_score"]) + 1e-9:
                    current = best_prop
                    current_entries = tuple(
                        (int(part.split(":")[0]), int(part.split(":")[1]), float(part.split(":")[2]))
                        for part in str(best_prop["entries"]).split()
                    )
                    accepted = True
                    reason = "greedy_gate_preserving_crystallize"

            path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
            print(
                f"[D21a][crystallize {step}] score={float(current['A_score']):.6f} "
                f"edges={current['edge_count']} exact={float(current['exact_byte_acc']):.4f} "
                f"drop_bit={float(current['single_edge_drop_mean_bit']):.4f} gate={current['gate_pass']} {reason}"
            )
            if not accepted:
                break

        rows = sorted(best_rows, key=lambda row: float(row["A_score"]), reverse=True)
        summary = group_summary(rows)
        heat = make_heatmap(summary, [code_dim], sorted({int(row["edge_budget"]) for row in rows}))
        write_outputs(
            out_dir=Path(args.out),
            rows=rows,
            summary=summary,
            heatmap=heat,
            mode="crystallize",
            config={
                "visible": visible_dim,
                "code_dim": code_dim,
                "start_family": str(args.start_family),
                "max_steps": int(args.max_steps),
                "seed": int(args.seed),
                "strategy": "greedy_drop_or_reweight",
            },
            path_rows=path_rows,
        )
        print(heat)
        print(f"[D21a] wrote {args.out}")
        return 0

    for step in range(1, int(args.max_steps) + 1):
        proposals = []
        for worker_idx in range(max(1, int(args.workers))):
            local_rng = random.Random(rng.randrange(1 << 30) + worker_idx)
            entries = mutate_entries(
                current_entries,
                visible_dim=visible_dim,
                code_dim=code_dim,
                rng=local_rng,
                crystallize=crystallize,
            )
            proposals.append(eval_entries(step * 1000 + worker_idx, "proposal", entries))

        best_prop = max(proposals, key=lambda row: float(row["A_score"]))
        accepted = False
        reason = "reject"
        if float(best_prop["A_score"]) > float(current["A_score"]) + 1e-9 and float(best_prop["exact_byte_acc"]) >= float(current["exact_byte_acc"]):
            current = best_prop
            current_entries = tuple(
                (int(part.split(":")[0]), int(part.split(":")[1]), float(part.split(":")[2]))
                for part in str(best_prop["entries"]).split()
            )
            accepted = True
            reason = "score_improve"

        best_rows.extend(proposals)
        path_rows.append({**current, "step": step, "accepted": accepted, "reason": reason})
        if step % 250 == 0 or accepted:
            print(
                f"[D21a][{step}] score={float(current['A_score']):.6f} "
                f"edges={current['edge_count']} exact={float(current['exact_byte_acc']):.4f} "
                f"drop_bit={float(current['single_edge_drop_mean_bit']):.4f} gate={current['gate_pass']} {reason}"
            )

    rows = sorted(best_rows, key=lambda row: float(row["A_score"]), reverse=True)
    summary = group_summary(rows)
    heat = make_heatmap(summary, [code_dim], sorted({int(row["edge_budget"]) for row in rows}))
    write_outputs(
        out_dir=Path(args.out),
        rows=rows,
        summary=summary,
        heatmap=heat,
        mode="mutate-climb",
        config={
            "visible": visible_dim,
            "code_dim": code_dim,
            "start_family": str(args.start_family),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
        },
        path_rows=path_rows,
    )
    print(heat)
    print(f"[D21a] wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21a reciprocal byte A-block")
    parser.add_argument("--mode", choices=("family-atlas", "mutate-climb", "crystallize"), required=True)
    parser.add_argument("--visible", type=int, default=8)
    parser.add_argument("--code-dims", default="8,16")
    parser.add_argument("--edge-budgets", default="8,16")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--start-family", default="redundant_copy_2x")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "family-atlas":
        return run_family_atlas(args)
    if args.mode == "mutate-climb":
        return run_local_search(args, crystallize=False)
    if args.mode == "crystallize":
        return run_local_search(args, crystallize=True)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
