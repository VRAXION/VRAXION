#!/usr/bin/env python3
"""
D21F A-block natural geometry search.

This phase checks whether the exact reciprocal byte A-block can be made less
copy-like while preserving byte round-trip correctness. It does not replace the
locked AB codec automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21a_reciprocal_byte_ablock import (  # noqa: E402
    ReciprocalABlock,
    all_visible_patterns,
    dedupe_entries,
    entries_to_string,
    evaluate_ablock,
    identity_entries,
    redundant_copy_entries,
    robustness_metrics,
)


WEIGHT_CHOICES = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
SMALL_WEIGHT_CHOICES = (-0.5, -0.25, -0.125, 0.125, 0.25, 0.5)
ASCII_PRINTABLE = list(range(32, 127))
UPPER = list(range(ord("A"), ord("Z") + 1))
LOWER = list(range(ord("a"), ord("z") + 1))
DIGITS = list(range(ord("0"), ord("9") + 1))
PUNCT = [ord(ch) for ch in ",.;:!?()[]{}'\"+-*/=_"]
SPACE = [ord(" "), ord("\n"), ord("\t")]
CLASS_GROUPS = {
    "upper": UPPER,
    "lower": LOWER,
    "digit": DIGITS,
    "punct": PUNCT,
    "space": SPACE,
}
ASCII_NEAR_PAIRS = [(ch, ch + 1) for ch in UPPER[:-1]] + [(ch, ch + 1) for ch in LOWER[:-1]]
ASCII_FAR_PAIRS = [(ch, ord("Z")) for ch in UPPER[:-1]] + [(ch, ord("z")) for ch in LOWER[:-1]]
CASE_NEAR_PAIRS = [(ord(chr(ch).upper()), ord(chr(ch).lower())) for ch in LOWER]
CASE_FAR_PAIRS = [(ord(chr(ch).upper()), ord("7")) for ch in LOWER]
DIGIT_NEAR_PAIRS = [(ch, ch + 1) for ch in DIGITS[:-1]]
DIGIT_FAR_PAIRS = [(ch, ord("Z")) for ch in DIGITS[:-1]]
CLASS_INTRA_PAIRS = (
    [(ord("A"), ord("B")), (ord("A"), ord("M")), (ord("M"), ord("Z"))]
    + [(ord("a"), ord("b")), (ord("a"), ord("m")), (ord("m"), ord("z"))]
    + [(ord("0"), ord("1")), (ord("0"), ord("5")), (ord("5"), ord("9"))]
    + [(ord(","), ord(".")), (ord("!"), ord("?")), (ord("("), ord(")"))]
)
CLASS_INTER_PAIRS = (
    [(ord("A"), ord("a")), (ord("A"), ord("0")), (ord("A"), ord(",")), (ord("A"), ord(" "))]
    + [(ord("a"), ord("0")), (ord("a"), ord(".")), (ord("0"), ord(",")), (ord("0"), ord(" "))]
)
PUNCT_ALNUM_PAIRS = [(ord(","), ord("A")), (ord("."), ord("z")), (ord("!"), ord("0")), (ord("?"), ord("m"))]
PUNCT_INTRA_PAIRS = [(ord(","), ord(".")), (ord("!"), ord("?")), (ord("("), ord(")"))]
RANDOM_FAR_PROBES = [(ord("A"), ord("Z")), (ord("a"), ord("z")), (ord("0"), ord("9")), (ord("A"), ord("7")), (ord("m"), ord("0"))]


@dataclass(frozen=True)
class SearchCandidate:
    arm: str
    candidate_id: int
    entries: tuple[tuple[int, int, float], ...]
    source: str
    step: int


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_entries(raw: str) -> tuple[tuple[int, int, float], ...]:
    out = []
    for part in raw.split():
        c, v, value = part.split(":")
        out.append((int(c), int(v), float(value)))
    return dedupe_entries(out)


def gate_pass(metrics: dict[str, object]) -> bool:
    return (
        float(metrics["exact_byte_acc"]) == 1.0
        and float(metrics["bit_acc"]) == 1.0
        and float(metrics["byte_argmax_acc"]) == 1.0
        and int(metrics["hidden_collisions"]) == 0
        and float(metrics["byte_margin_min"]) > 0.0
        and float(metrics["reciprocity_error"]) == 0.0
    )


def pairwise_distances(codes: np.ndarray) -> np.ndarray:
    diff = codes[:, None, :] - codes[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def code_distance(codes: np.ndarray, left: int, right: int) -> float:
    diff = codes[left] - codes[right]
    return float(np.sqrt(np.dot(diff, diff)))


def closer_rate(codes: np.ndarray, near_pairs: Sequence[tuple[int, int]], far_pairs: Sequence[tuple[int, int]]) -> float:
    if not near_pairs or not far_pairs:
        return 0.0
    wins = 0
    count = 0
    for idx, near in enumerate(near_pairs):
        far = far_pairs[idx % len(far_pairs)]
        if code_distance(codes, near[0], near[1]) < code_distance(codes, far[0], far[1]):
            wins += 1
        count += 1
    return float(wins / max(1, count))


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def class_cluster_score(codes: np.ndarray) -> float:
    intra_mean = safe_mean([code_distance(codes, left, right) for left, right in CLASS_INTRA_PAIRS])
    inter_mean = safe_mean([code_distance(codes, left, right) for left, right in CLASS_INTER_PAIRS])
    if inter_mean <= 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, (inter_mean - intra_mean) / inter_mean)))


def punct_separation_score(codes: np.ndarray) -> float:
    punct_to_alnum = [code_distance(codes, left, right) for left, right in PUNCT_ALNUM_PAIRS]
    punct_intra = [code_distance(codes, left, right) for left, right in PUNCT_INTRA_PAIRS]
    denom = safe_mean(punct_to_alnum)
    if denom <= 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, (safe_mean(punct_to_alnum) - safe_mean(punct_intra)) / denom)))


def random_far_margin(codes: np.ndarray) -> float:
    neighbor_pairs = ASCII_NEAR_PAIRS + DIGIT_NEAR_PAIRS
    neighbor_mean = safe_mean([code_distance(codes, a, b) for a, b in neighbor_pairs])
    far_mean = safe_mean([code_distance(codes, a, b) for a, b in RANDOM_FAR_PROBES])
    return float(far_mean - neighbor_mean)


def geometry_metrics(block: ReciprocalABlock, *, permuted_ascii: bool = False) -> dict[str, float]:
    patterns = all_visible_patterns(block.visible_dim)
    codes = block.encode_patterns(patterns)
    if permuted_ascii:
        rng = random.Random(20260503)
        order = ASCII_PRINTABLE.copy()
        rng.shuffle(order)
        remap = {src: dst for src, dst in zip(ASCII_PRINTABLE, order)}
        codes = codes.copy()
        for src, dst in remap.items():
            codes[src] = block.encode_byte(dst)
    return {
        "ascii_neighbor_closer_rate": closer_rate(codes, ASCII_NEAR_PAIRS, ASCII_FAR_PAIRS),
        "case_pair_closer_rate": closer_rate(codes, CASE_NEAR_PAIRS, CASE_FAR_PAIRS),
        "digit_neighbor_closer_rate": closer_rate(codes, DIGIT_NEAR_PAIRS, DIGIT_FAR_PAIRS),
        "class_cluster_score": class_cluster_score(codes),
        "punct_separation_score": punct_separation_score(codes),
        "random_far_margin": random_far_margin(codes),
    }


def identity_copy_penalty(entries: Sequence[tuple[int, int, float]], visible_dim: int, code_dim: int) -> float:
    if not entries:
        return 1.0
    exact_copy = 0
    one_hot_copy_rows = 0
    by_row: dict[int, list[tuple[int, float]]] = {}
    for code_idx, visible_idx, value in entries:
        by_row.setdefault(code_idx, []).append((visible_idx, value))
        if visible_idx == (code_idx % visible_dim) and abs(abs(value) - 1.0) < 1e-9:
            exact_copy += 1
    for code_idx, row_entries in by_row.items():
        if len(row_entries) == 1:
            visible_idx, value = row_entries[0]
            if visible_idx == (code_idx % visible_dim) and abs(abs(value) - 1.0) < 1e-9:
                one_hot_copy_rows += 1
    return float(0.65 * exact_copy / max(1, len(entries)) + 0.35 * one_hot_copy_rows / max(1, code_dim))


def duplicate_lane_penalty(block: ReciprocalABlock) -> float:
    rows = block.encoder
    if rows.shape[0] <= 1:
        return 0.0
    duplicate = 0
    total = 0
    for i in range(rows.shape[0]):
        for j in range(i + 1, rows.shape[0]):
            left = rows[i]
            right = rows[j]
            if np.allclose(left, right, atol=1e-9) or np.allclose(left, -right, atol=1e-9):
                duplicate += 1
            total += 1
    return float(duplicate / max(1, total))


def distributed_code_score(block: ReciprocalABlock) -> float:
    nonzeros = np.count_nonzero(block.encoder, axis=1)
    active = int(np.count_nonzero(nonzeros))
    multi = int(np.count_nonzero(nonzeros > 1))
    if active == 0:
        return 0.0
    return float(0.5 * active / block.code_dim + 0.5 * multi / active)


def ascii_geometry_score(metrics: dict[str, object]) -> float:
    return float(
        0.24 * float(metrics["ascii_neighbor_closer_rate"])
        + 0.24 * float(metrics["case_pair_closer_rate"])
        + 0.18 * float(metrics["digit_neighbor_closer_rate"])
        + 0.18 * max(-1.0, float(metrics["class_cluster_score"]))
        + 0.08 * max(-1.0, float(metrics["punct_separation_score"]))
        + 0.08 * math.tanh(float(metrics["random_far_margin"]) / 2.0)
    )


def natural_score(metrics: dict[str, object]) -> float:
    hard = gate_pass(metrics)
    base = -100.0 if not hard else 20.0
    robustness = float(metrics.get("single_edge_drop_mean_bit", 0.0))
    margin = max(0.0, float(metrics["byte_margin_min"]))
    return float(
        base
        + 0.60 * margin
        + 8.0 * float(metrics["ascii_class_geometry"])
        + 1.25 * robustness
        + 1.20 * float(metrics["distributed_code_score"])
        - 2.50 * float(metrics["identity_copy_penalty"])
        - 1.50 * float(metrics["duplicate_lane_penalty"])
        - 0.012 * float(metrics["edge_count"])
    )


def evaluate_entries(
    *,
    arm: str,
    candidate_id: int,
    visible_dim: int,
    code_dim: int,
    entries: tuple[tuple[int, int, float], ...],
    source: str,
    step: int,
    include_robustness: bool,
    permuted_ascii: bool = False,
) -> dict[str, object]:
    block = ReciprocalABlock.from_entries(visible_dim, code_dim, entries)
    patterns = all_visible_patterns(visible_dim)
    metrics: dict[str, object] = {
        "candidate_id": candidate_id,
        "arm": arm,
        "source": source,
        "step": step,
        "visible_dim": visible_dim,
        "code_dim": code_dim,
        "entries": entries_to_string(entries),
    }
    metrics.update(evaluate_ablock(block, patterns))
    if include_robustness:
        metrics.update(robustness_metrics(block, patterns))
    else:
        metrics.update(
            {
                "single_edge_drop_min_exact": 0.0,
                "single_edge_drop_mean_exact": 0.0,
                "single_edge_drop_min_bit": 0.0,
                "single_edge_drop_mean_bit": 0.0,
                "single_edge_drop_min_byte_margin": 0.0,
            }
        )
    metrics.update(geometry_metrics(block, permuted_ascii=permuted_ascii))
    metrics["identity_copy_penalty"] = identity_copy_penalty(entries, visible_dim, code_dim)
    metrics["duplicate_lane_penalty"] = duplicate_lane_penalty(block)
    metrics["distributed_code_score"] = distributed_code_score(block)
    metrics["ascii_class_geometry"] = ascii_geometry_score(metrics)
    metrics["gate_pass"] = gate_pass(metrics)
    metrics["A_natural_score"] = natural_score(metrics)
    return metrics


def overlay_initial_entries(visible_dim: int, code_dim: int) -> tuple[tuple[int, int, float], ...]:
    entries = list(identity_entries(visible_dim))
    # Small non-canonical rows are allowed to shape A-space geometry while the
    # canonical rows preserve the decode highway.
    geometry_rows = [
        (8, 5, 0.25),
        (8, 6, -0.25),
        (9, 4, 0.25),
        (9, 5, 0.25),
        (10, 6, 0.25),
        (10, 7, -0.25),
        (11, 0, 0.25),
        (11, 1, -0.25),
        (12, 1, 0.25),
        (12, 2, -0.25),
        (13, 2, 0.25),
        (13, 3, -0.25),
        (14, 3, 0.25),
        (14, 4, -0.25),
        (15, 0, 0.25),
        (15, 7, -0.25),
    ]
    entries.extend(item for item in geometry_rows if item[0] < code_dim)
    return dedupe_entries(entries)


def no_prefill_initial_entries(visible_dim: int, code_dim: int, rng: random.Random) -> tuple[tuple[int, int, float], ...]:
    entries = []
    for code_idx in rng.sample(range(code_dim), min(code_dim, visible_dim)):
        entries.append((code_idx, rng.randrange(visible_dim), rng.choice(WEIGHT_CHOICES)))
    return dedupe_entries(entries)


def random_sparse_entries(visible_dim: int, code_dim: int, edge_budget: int, rng: random.Random) -> tuple[tuple[int, int, float], ...]:
    entries = []
    used: set[tuple[int, int]] = set()
    for _ in range(edge_budget):
        for _attempt in range(100):
            code_idx = rng.randrange(code_dim)
            visible_idx = rng.randrange(visible_dim)
            if (code_idx, visible_idx) not in used:
                used.add((code_idx, visible_idx))
                entries.append((code_idx, visible_idx, rng.choice(WEIGHT_CHOICES)))
                break
    return dedupe_entries(entries)


def hamming_only_entries(visible_dim: int, code_dim: int) -> tuple[tuple[int, int, float], ...]:
    # Exact but intentionally ASCII-oblivious: one signed bit per lane plus a
    # permuted mirror, useful as a non-semantic Hamming control.
    entries = list(identity_entries(visible_dim))
    for visible_idx in range(visible_dim):
        code_idx = visible_dim + ((visible_idx * 5) % visible_dim)
        if code_idx < code_dim:
            entries.append((code_idx, visible_idx, 1.0))
    return dedupe_entries(entries)


def mutate_entries(
    entries: tuple[tuple[int, int, float], ...],
    *,
    visible_dim: int,
    code_dim: int,
    rng: random.Random,
    locked_identity: bool,
    no_prefill: bool,
) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    locked = {(idx, idx) for idx in range(visible_dim)} if locked_identity else set()
    used = {(c, v) for c, v, _ in current}
    ops = ("add", "remove", "flip", "reweight")
    op = rng.choice(ops)

    def can_touch(idx: int) -> bool:
        c, v, _value = current[idx]
        return (c, v) not in locked

    if op == "add":
        for _ in range(100):
            row_start = visible_dim if locked_identity and code_dim > visible_dim else 0
            code_idx = rng.randrange(row_start, code_dim)
            visible_idx = rng.randrange(visible_dim)
            if (code_idx, visible_idx) not in used:
                weights = SMALL_WEIGHT_CHOICES if locked_identity else WEIGHT_CHOICES
                current.append((code_idx, visible_idx, rng.choice(weights)))
                return dedupe_entries(current)
    if op == "remove" and current:
        candidates = [idx for idx in range(len(current)) if can_touch(idx)]
        if candidates:
            del current[rng.choice(candidates)]
            return dedupe_entries(current)
    if op == "flip" and current:
        candidates = [idx for idx in range(len(current)) if can_touch(idx)]
        if candidates:
            idx = rng.choice(candidates)
            _old_c, _old_v, value = current[idx]
            for _ in range(100):
                row_start = visible_dim if locked_identity and code_dim > visible_dim else 0
                code_idx = rng.randrange(row_start, code_dim)
                visible_idx = rng.randrange(visible_dim)
                if (code_idx, visible_idx) not in used or no_prefill:
                    current[idx] = (code_idx, visible_idx, value)
                    return dedupe_entries(current)
    if op == "reweight" and current:
        candidates = [idx for idx in range(len(current)) if can_touch(idx)]
        if candidates:
            idx = rng.choice(candidates)
            code_idx, visible_idx, _value = current[idx]
            weights = SMALL_WEIGHT_CHOICES if locked_identity else WEIGHT_CHOICES
            current[idx] = (code_idx, visible_idx, rng.choice(weights))
            return dedupe_entries(current)
    return entries


def compact_reference_entries() -> tuple[tuple[int, int, float], ...]:
    # A deterministic 14-edge compact reference. It keeps exact roundtrip, but
    # does not claim to be the D21A crystallize winner if local outputs are gone.
    # Missing mirror lanes for bit 6/7 reduce robustness versus redundant_copy.
    entries = list(redundant_copy_entries(8, 16))
    entries = [entry for entry in entries if entry[0] not in (14, 15)]
    return dedupe_entries(entries)


def run_search_arm(
    *,
    arm: str,
    visible_dim: int,
    code_dim: int,
    max_steps: int,
    workers: int,
    seed: int,
    candidate_id_start: int,
) -> tuple[list[dict[str, object]], int]:
    rng = random.Random(seed)
    candidate_id = candidate_id_start
    rows: list[dict[str, object]] = []
    locked = arm == "overlay_locked"
    no_prefill = arm == "no_prefill"
    if locked:
        current_entries = overlay_initial_entries(visible_dim, code_dim)
    elif no_prefill:
        current_entries = no_prefill_initial_entries(visible_dim, code_dim, rng)
    else:
        raise ValueError(arm)

    current = evaluate_entries(
        arm=arm,
        candidate_id=candidate_id,
        visible_dim=visible_dim,
        code_dim=code_dim,
        entries=current_entries,
        source="start",
        step=0,
        include_robustness=True,
    )
    candidate_id += 1
    rows.append(current)

    accepted_without_improve = 0
    for step in range(1, max_steps + 1):
        proposals = []
        for _worker in range(max(1, workers)):
            mutated = mutate_entries(
                current_entries,
                visible_dim=visible_dim,
                code_dim=code_dim,
                rng=rng,
                locked_identity=locked,
                no_prefill=no_prefill,
            )
            row = evaluate_entries(
                arm=arm,
                candidate_id=candidate_id,
                visible_dim=visible_dim,
                code_dim=code_dim,
                entries=mutated,
                source="proposal",
                step=step,
                include_robustness=False,
            )
            candidate_id += 1
            proposals.append(row)

        best_prop = max(proposals, key=lambda item: float(item["A_natural_score"]))
        rows.extend(proposals)
        if float(best_prop["A_natural_score"]) > float(current["A_natural_score"]) + 1e-9 and (
            bool(best_prop["gate_pass"]) or not bool(current["gate_pass"])
        ):
            current_entries = parse_entries(str(best_prop["entries"]))
            current = evaluate_entries(
                arm=arm,
                candidate_id=candidate_id,
                visible_dim=visible_dim,
                code_dim=code_dim,
                entries=current_entries,
                source="accepted",
                step=step,
                include_robustness=True,
            )
            candidate_id += 1
            rows.append(current)
            accepted_without_improve = 0
        else:
            accepted_without_improve += 1
        if accepted_without_improve >= 1200:
            break

    return rows, candidate_id


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distance_rows(top_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in top_rows:
        entries = parse_entries(str(row["entries"]))
        block = ReciprocalABlock.from_entries(int(row["visible_dim"]), int(row["code_dim"]), entries)
        codes = block.encode_patterns(all_visible_patterns(block.visible_dim))
        dist = pairwise_distances(codes)
        for left, right in [
            (ord("A"), ord("B")),
            (ord("A"), ord("Z")),
            (ord("A"), ord("a")),
            (ord("A"), ord("7")),
            (ord("1"), ord("2")),
            (ord("1"), ord("Z")),
            (ord(","), ord(".")),
            (ord(","), ord("A")),
            (ord(" "), ord("A")),
        ]:
            out.append(
                {
                    "candidate_id": row["candidate_id"],
                    "arm": row["arm"],
                    "left_byte": left,
                    "left_char": repr(chr(left)),
                    "right_byte": right,
                    "right_char": repr(chr(right)),
                    "distance": float(dist[left, right]),
                }
            )
    return out


def topology_rows(top_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in top_rows:
        entries = parse_entries(str(row["entries"]))
        for edge_idx, (code_idx, visible_idx, value) in enumerate(entries):
            out.append(
                {
                    "candidate_id": row["candidate_id"],
                    "arm": row["arm"],
                    "edge_idx": edge_idx,
                    "code_lane": code_idx,
                    "visible_bit": visible_idx,
                    "weight": value,
                    "is_identity_copy": visible_idx == (code_idx % int(row["visible_dim"])) and abs(abs(value) - 1.0) < 1e-9,
                }
            )
    return out


def ascii_map(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "D21F A-block natural geometry map",
        "",
        "Higher geometry is better; lower copy penalty is better.",
        "",
        "arm                         exact margin  geom   copy  dup   dist  score",
        "--------------------------  ----- ------ ------ ----- ----- ----- --------",
    ]
    for row in rows[:18]:
        lines.append(
            f"{str(row['arm'])[:26]:26} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):6.3f} "
            f"{float(row['identity_copy_penalty']):5.3f} "
            f"{float(row['duplicate_lane_penalty']):5.3f} "
            f"{float(row['distributed_code_score']):5.3f} "
            f"{float(row['A_natural_score']):8.3f}"
        )
    lines.extend(
        [
            "",
            "Key distance probes for best candidate:",
        ]
    )
    if rows:
        row = rows[0]
        entries = parse_entries(str(row["entries"]))
        block = ReciprocalABlock.from_entries(int(row["visible_dim"]), int(row["code_dim"]), entries)
        dist = pairwise_distances(block.encode_patterns(all_visible_patterns(block.visible_dim)))
        probes = [
            ("A-B", ord("A"), ord("B")),
            ("A-Z", ord("A"), ord("Z")),
            ("A-a", ord("A"), ord("a")),
            ("A-7", ord("A"), ord("7")),
            ("1-2", ord("1"), ord("2")),
            ("1-Z", ord("1"), ord("Z")),
        ]
        max_dist = max(float(dist[a, b]) for _name, a, b in probes) or 1.0
        for name, a, b in probes:
            value = float(dist[a, b])
            bar = "#" * max(1, int(28 * value / max_dist))
            lines.append(f"  {name:4} {value:7.3f} {bar}")
    return "\n".join(lines) + "\n"


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    baseline = next((row for row in rows if row["arm"] == "baseline16"), None)
    if baseline is None:
        return "D21F_INTERNAL_ERROR", "baseline16 row missing"
    controls = [row for row in rows if str(row["arm"]) in {"random_sparse", "permuted_ascii", "hamming_only"}]
    noncopy = [
        row
        for row in rows
        if bool(row["gate_pass"])
        and str(row["arm"]) in {"overlay_locked", "no_prefill", "crystallized14"}
        and float(row["identity_copy_penalty"]) < float(baseline["identity_copy_penalty"]) - 0.02
    ]
    if not noncopy:
        return "D21F_COPY_BASELINE_STILL_WINS", "no less-copy candidate preserved exact roundtrip with positive margin"

    best = max(
        noncopy,
        key=lambda row: (
            float(row["ascii_class_geometry"]) - float(baseline["ascii_class_geometry"]),
            float(baseline["identity_copy_penalty"]) - float(row["identity_copy_penalty"]),
            float(row["A_natural_score"]),
        ),
    )
    geom_gain = float(best["ascii_class_geometry"]) - float(baseline["ascii_class_geometry"])
    copy_gain = float(baseline["identity_copy_penalty"]) - float(best["identity_copy_penalty"])
    control_leak = any(
        bool(row["gate_pass"])
        and str(row["arm"]) == "random_sparse"
        and (
            float(row["A_natural_score"]) >= float(best["A_natural_score"]) - 0.05
            or float(row["ascii_class_geometry"]) >= float(best["ascii_class_geometry"]) - 0.01
        )
        for row in controls
    )
    if control_leak:
        return "D21F_COPY_BASELINE_STILL_WINS", "random sparse controls matched the best candidate too closely"
    strong_margin = float(best["byte_margin_min"]) >= max(2.0, 0.50 * float(baseline["byte_margin_min"]))
    strong_robustness = float(best.get("single_edge_drop_mean_bit", 0.0)) >= 0.99
    if geom_gain > 0.04 and copy_gain > 0.08 and strong_margin and strong_robustness:
        return "D21F_A_NATURAL_GEOMETRY_PASS", f"{best['arm']} improves geometry by {geom_gain:+.4f} and copy penalty by {copy_gain:+.4f}"
    if geom_gain > 0.0 and copy_gain > 0.03:
        return (
            "D21F_A_NATURAL_WEAK_PASS",
            f"{best['arm']} improves geometry by {geom_gain:+.4f}, but margin/robustness is below deploy baseline",
        )
    return "D21F_COPY_BASELINE_STILL_WINS", "non-copy candidates did not beat baseline geometry by a meaningful margin"


def write_report(out_dir: Path, verdict_name: str, verdict_reason: str, rows: Sequence[dict[str, object]], config: dict[str, object]) -> None:
    baseline = next((row for row in rows if row["arm"] == "baseline16"), rows[0])
    best = rows[0]
    text = f"""# D21F A-Block Natural Geometry Report

Date: 2026-05-03

## Verdict

```text
{verdict_name}
{verdict_reason}
```

## What Was Tested

D21F asked whether the exact reciprocal A-block can stop being only a
copy-style byte codec and become a more natural sparse ASCII geometry:

```text
byte -> A16 -> byte exact
+
ASCII-neighbor / case / digit / punctuation geometry
+
lower identity-copy penalty
```

The locked AB codec is not changed by this run. D21F only promotes a candidate
if it beats the current deploy A-block while preserving exact reconstruction.

## Best Rows

```text
arm                         exact margin  geom   copy  dup   dist  score
--------------------------  ----- ------ ------ ----- ----- ----- --------
"""
    for row in rows[:10]:
        text += (
            f"{str(row['arm'])[:26]:26} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):6.3f} "
            f"{float(row['identity_copy_penalty']):5.3f} "
            f"{float(row['duplicate_lane_penalty']):5.3f} "
            f"{float(row['distributed_code_score']):5.3f} "
            f"{float(row['A_natural_score']):8.3f}\n"
        )
    text += """```

## Baseline vs Best

```text
baseline16:
"""
    text += (
        f"  exact_byte_acc={float(baseline['exact_byte_acc']):.6f}\n"
        f"  byte_margin_min={float(baseline['byte_margin_min']):.6f}\n"
        f"  ascii_class_geometry={float(baseline['ascii_class_geometry']):.6f}\n"
        f"  identity_copy_penalty={float(baseline['identity_copy_penalty']):.6f}\n"
        f"  duplicate_lane_penalty={float(baseline['duplicate_lane_penalty']):.6f}\n"
        f"  A_natural_score={float(baseline['A_natural_score']):.6f}\n\n"
        f"best:\n"
        f"  arm={best['arm']}\n"
        f"  exact_byte_acc={float(best['exact_byte_acc']):.6f}\n"
        f"  byte_margin_min={float(best['byte_margin_min']):.6f}\n"
        f"  ascii_class_geometry={float(best['ascii_class_geometry']):.6f}\n"
        f"  identity_copy_penalty={float(best['identity_copy_penalty']):.6f}\n"
        f"  duplicate_lane_penalty={float(best['duplicate_lane_penalty']):.6f}\n"
        f"  A_natural_score={float(best['A_natural_score']):.6f}\n"
    )
    text += """```

## Interpretation

If the verdict is `D21F_A_NATURAL_GEOMETRY_PASS`, an A_v2 candidate exists for
later AB-codec experiments. If the verdict is `D21F_COPY_BASELINE_STILL_WINS`,
the honest conclusion is that the current A-block remains a robust codec but
not a natural character manifold.

## Config

```json
"""
    text += json.dumps(config, indent=2)
    text += """
```
"""
    (out_dir / "D21F_ABLOCK_NATURAL_GEOMETRY_REPORT.md").write_text(text, encoding="utf-8")


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    visible_dim = int(args.visible)
    code_dim = int(args.code_dim)
    arms = parse_list(str(args.arms))
    seed = int(args.seed)
    max_steps = int(args.max_steps)
    workers = int(args.workers)
    samples = int(args.samples)
    rows: list[dict[str, object]] = []
    candidate_id = 0

    fixed: list[tuple[str, tuple[tuple[int, int, float], ...], bool]] = []
    if "baseline16" in arms:
        fixed.append(("baseline16", redundant_copy_entries(visible_dim, code_dim), False))
    if "crystallized14" in arms:
        fixed.append(("crystallized14", compact_reference_entries(), False))
    if "hamming_only" in arms:
        fixed.append(("hamming_only", hamming_only_entries(visible_dim, code_dim), False))
    if "permuted_ascii" in arms:
        fixed.append(("permuted_ascii", redundant_copy_entries(visible_dim, code_dim), True))

    for arm, entries, permuted in fixed:
        rows.append(
            evaluate_entries(
                arm=arm,
                candidate_id=candidate_id,
                visible_dim=visible_dim,
                code_dim=code_dim,
                entries=entries,
                source="fixed",
                step=0,
                include_robustness=True,
                permuted_ascii=permuted,
            )
        )
        candidate_id += 1

    if "random_sparse" in arms:
        rng = random.Random(seed + 71)
        for idx in range(max(8, min(samples, 1024))):
            edge_budget = rng.choice((8, 12, 16, 20, 24))
            rows.append(
                evaluate_entries(
                    arm="random_sparse",
                    candidate_id=candidate_id,
                    visible_dim=visible_dim,
                    code_dim=code_dim,
                    entries=random_sparse_entries(visible_dim, code_dim, edge_budget, rng),
                    source="random_control",
                    step=idx,
                    include_robustness=False,
                )
            )
            candidate_id += 1

    for arm in ("overlay_locked", "no_prefill"):
        if arm in arms:
            arm_rows, candidate_id = run_search_arm(
                arm=arm,
                visible_dim=visible_dim,
                code_dim=code_dim,
                max_steps=max_steps,
                workers=workers,
                seed=seed + (101 if arm == "overlay_locked" else 303),
                candidate_id_start=candidate_id,
            )
            rows.extend(arm_rows)

    # Recompute robustness for top candidates so ranking/report does not use
    # missing robustness placeholders from proposal rows.
    provisional = sorted(rows, key=lambda row: float(row["A_natural_score"]), reverse=True)[:64]
    robust_by_id: dict[int, dict[str, object]] = {}
    for row in provisional:
        entries = parse_entries(str(row["entries"]))
        robust_by_id[int(row["candidate_id"])] = evaluate_entries(
            arm=str(row["arm"]),
            candidate_id=int(row["candidate_id"]),
            visible_dim=visible_dim,
            code_dim=code_dim,
            entries=entries,
            source=str(row["source"]),
            step=int(row["step"]),
            include_robustness=True,
            permuted_ascii=str(row["arm"]) == "permuted_ascii",
        )
    final_rows = []
    for row in rows:
        final_rows.append(robust_by_id.get(int(row["candidate_id"]), row))
    return sorted(final_rows, key=lambda item: float(item["A_natural_score"]), reverse=True)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    verdict_name, verdict_reason = verdict(rows)
    top_rows = rows[: max(1, int(getattr(args, "top_k", 16)))]

    write_csv(out_dir / "a_natural_candidates.csv", rows)
    write_csv(out_dir / "a_distance_matrix.csv", distance_rows(top_rows[:4]))
    write_csv(out_dir / "a_topology_edges.csv", topology_rows(top_rows[:4]))
    (out_dir / "a_geometry_ascii_map.txt").write_text(ascii_map(rows), encoding="utf-8")

    config = {
        "mode": args.mode,
        "visible": int(args.visible),
        "code_dim": int(args.code_dim),
        "arms": parse_list(str(args.arms)),
        "samples": int(args.samples),
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "seed": int(args.seed),
    }
    payload = {
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
        "config": config,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if bool(row["gate_pass"])),
        "best_candidate": rows[0],
        "baseline16": next((row for row in rows if row["arm"] == "baseline16"), None),
    }
    (out_dir / "a_natural_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, verdict_name, verdict_reason, rows, config)

    print(ascii_map(rows))
    print(f"[D21F] verdict={verdict_name} reason={verdict_reason}")
    print(f"[D21F] wrote {out_dir}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    rows = []
    with Path(args.candidates).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if idx >= int(args.top_k):
                break
            entries = parse_entries(str(row["entries"]))
            rows.append(
                evaluate_entries(
                    arm=str(row["arm"]),
                    candidate_id=idx,
                    visible_dim=int(row.get("visible_dim", args.visible)),
                    code_dim=int(row.get("code_dim", args.code_dim)),
                    entries=entries,
                    source="confirm",
                    step=0,
                    include_robustness=True,
                    permuted_ascii=str(row["arm"]) == "permuted_ascii",
                )
            )
    rows = sorted(rows, key=lambda item: float(item["A_natural_score"]), reverse=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_name, verdict_reason = verdict(rows) if any(row["arm"] == "baseline16" for row in rows) else ("D21F_CONFIRM_ONLY", "top-k candidates re-evaluated")
    write_csv(out_dir / "a_natural_candidates.csv", rows)
    write_csv(out_dir / "a_distance_matrix.csv", distance_rows(rows[:4]))
    write_csv(out_dir / "a_topology_edges.csv", topology_rows(rows[:4]))
    (out_dir / "a_geometry_ascii_map.txt").write_text(ascii_map(rows), encoding="utf-8")
    payload = {
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if bool(row["gate_pass"])),
        "best_candidate": rows[0] if rows else None,
    }
    (out_dir / "a_natural_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, verdict_name, verdict_reason, rows, {"mode": "confirm", "top_k": int(args.top_k)})
    print(ascii_map(rows))
    print(f"[D21F] confirm wrote {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21F natural geometry A-block search")
    parser.add_argument("--mode", choices=("smoke", "main", "confirm"), required=True)
    parser.add_argument("--visible", type=int, default=8)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument(
        "--arms",
        default="baseline16,crystallized14,overlay_locked,no_prefill,random_sparse,hamming_only,permuted_ascii",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "confirm":
        if not args.candidates:
            raise SystemExit("--candidates is required for confirm")
        return run_confirm(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
