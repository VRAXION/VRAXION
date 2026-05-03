#!/usr/bin/env python3
"""A-v2 native int8 multi-seed search.

Searches hidden-only reciprocal A-block candidates with native int8_q6 layer
weights. The mutator edits integer q values only; float matrices are used only
as an adapter into the existing A geometry evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.a_hidden_natural_int8_artifact import VERSION, verify_payload  # noqa: E402
from tools._scratch.d21a_reciprocal_byte_ablock import ReciprocalABlock, all_visible_patterns  # noqa: E402
from tools._scratch.d21i_ablock_hidden_sparse_sweep import (  # noqa: E402
    CODE_DIM,
    VISIBLE_DIM,
    HiddenAState,
    direct_redundant_matrix,
    evaluate_state,
    gate_pass,
    write_csv,
)


SCALE = 64
Q_CHOICES = (-64, -56, -48, -40, -32, -24, -16, -8, 8, 16, 24, 32, 40, 48, 56, 64)
CURRENT_ARTIFACT = Path("tools/a_hidden_natural_margin_int8_v1.json")
DEFAULT_OUT = Path("output/phase_a_v2_native_int8_seed_sweep_20260503")
EPS = 1e-9


@dataclass(frozen=True)
class Int8AState:
    arm: str
    seed: int
    hidden_dim: int
    hidden_in_q: np.ndarray
    hidden_out_q: np.ndarray

    def clone(self) -> "Int8AState":
        return Int8AState(
            self.arm,
            self.seed,
            self.hidden_dim,
            self.hidden_in_q.copy(),
            self.hidden_out_q.copy(),
        )


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def q_to_float(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64) / float(SCALE)


def q_entries(matrix: np.ndarray) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            q = int(matrix[row, col])
            if q != 0:
                rows.append([row, col, q])
    return rows


def q_entries_string(matrix: np.ndarray) -> str:
    return " ".join(f"{row}:{col}:{q}" for row, col, q in q_entries(matrix))


def parse_q_entries(raw: str, rows: int, cols: int) -> np.ndarray:
    matrix = np.zeros((rows, cols), dtype=np.int16)
    for part in str(raw).split():
        row, col, q = part.split(":")
        matrix[int(row), int(col)] = int(q)
    return matrix


def count_q(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(matrix))


def to_hidden_state(state: Int8AState) -> HiddenAState:
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    return HiddenAState(
        state.arm,
        state.hidden_dim,
        direct,
        q_to_float(state.hidden_in_q),
        q_to_float(state.hidden_out_q),
    )


def effective_rank_for_state(state: Int8AState) -> float:
    hidden_state = to_hidden_state(state)
    block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, hidden_state.effective_encoder())
    codes = block.encode_patterns(all_visible_patterns(VISIBLE_DIM))
    centered = codes - codes.mean(axis=0, keepdims=True)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    total = float(singular.sum())
    if total <= 1e-12:
        return 0.0
    probs = singular / total
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))


def score(row: dict[str, object]) -> float:
    if not bool(row.get("gate_pass", False)):
        return -1000.0 + 100.0 * float(row["exact_byte_acc"]) + 10.0 * float(row["bit_acc"])
    return float(
        50.0 * min(float(row["byte_margin_min"]), 4.25)
        + 12.0 * float(row["ascii_class_geometry"])
        + 2.0 * float(row["random_far_margin"])
        + 1.0 * float(row["effective_rank"])
        - 14.0 * float(row["effective_copy_penalty"])
        - 2.0 * float(row["duplicate_lane_penalty"])
        - 0.03 * float(row["structural_edge_count"])
    )


def acceptance_gate(row: dict[str, object]) -> bool:
    return (
        gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and float(row["byte_margin_min"]) > 0.0
        and float(row["ascii_class_geometry"]) >= 0.750
        and float(row["effective_copy_penalty"]) <= 0.15
    )


def strong_gate(row: dict[str, object]) -> bool:
    return (
        acceptance_gate(row)
        and float(row["byte_margin_min"]) >= 4.0
        and float(row["ascii_class_geometry"]) >= 0.760
        and float(row["effective_copy_penalty"]) <= 0.10
        and float(row["duplicate_lane_penalty"]) <= 0.05
        and float(row.get("single_edge_drop_mean_bit", 0.0)) >= 0.995
    )


def key_for_state(state: Int8AState) -> str:
    return q_entries_string(state.hidden_in_q) + "|" + q_entries_string(state.hidden_out_q)


def evaluate_int8_state(
    state: Int8AState,
    *,
    candidate_id: int,
    source: str,
    step: int,
    include_robustness: bool,
    mutation: str,
) -> dict[str, object]:
    row = evaluate_state(
        to_hidden_state(state),
        candidate_id=candidate_id,
        source=source,
        step=step,
        include_robustness=include_robustness,
    )
    row.update(
        {
            "seed": state.seed,
            "search_arm": state.arm,
            "native_storage": "int8_q6",
            "scale": SCALE,
            "hidden_in_q_entries": q_entries_string(state.hidden_in_q),
            "hidden_out_q_entries": q_entries_string(state.hidden_out_q),
            "hidden_in_q_edge_count": count_q(state.hidden_in_q),
            "hidden_out_q_edge_count": count_q(state.hidden_out_q),
            "mutation": mutation,
            "effective_rank": effective_rank_for_state(state),
        }
    )
    row["native_int8_score"] = score(row)
    row["acceptance_gate"] = acceptance_gate(row)
    row["strong_gate"] = strong_gate(row)
    return row


def load_current_artifact(path: Path, *, hidden_dim: int, seed: int) -> Int8AState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_hidden_dim = int(payload["hidden_dim"])
    if hidden_dim < source_hidden_dim:
        raise ValueError(f"cannot fit source hidden_dim={source_hidden_dim} into hidden_dim={hidden_dim}")
    hidden_in_q = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.int16)
    hidden_out_q = np.zeros((CODE_DIM, hidden_dim), dtype=np.int16)
    for row, col, q in payload["hidden_in_q"]:
        hidden_in_q[int(row), int(col)] = int(q)
    for row, col, q in payload["hidden_out_q"]:
        hidden_out_q[int(row), int(col)] = int(q)
    return Int8AState(f"current_int8_polish_h{hidden_dim}", seed, hidden_dim, hidden_in_q, hidden_out_q)


def hidden_bridge_state(*, hidden_dim: int, seed: int) -> Int8AState:
    hidden_in_q = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.int16)
    hidden_out_q = np.zeros((CODE_DIM, hidden_dim), dtype=np.int16)
    for bit in range(min(VISIBLE_DIM, hidden_dim)):
        hidden_in_q[bit, bit] = 64
        hidden_out_q[bit, bit] = 64
        hidden_out_q[VISIBLE_DIM + bit, bit] = 64
    return Int8AState(f"hidden_bridge_restart_h{hidden_dim}", seed, hidden_dim, hidden_in_q, hidden_out_q)


def random_hidden_state(*, hidden_dim: int, seed: int, rng: random.Random) -> Int8AState:
    hidden_in_q = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.int16)
    hidden_out_q = np.zeros((CODE_DIM, hidden_dim), dtype=np.int16)
    for bit in range(VISIBLE_DIM):
        hidden_idx = rng.randrange(hidden_dim)
        hidden_in_q[hidden_idx, bit] = rng.choice(Q_CHOICES)
        hidden_out_q[rng.randrange(CODE_DIM), hidden_idx] = rng.choice(Q_CHOICES)
    for _ in range(max(8, hidden_dim)):
        hidden_out_q[rng.randrange(CODE_DIM), rng.randrange(hidden_dim)] = rng.choice(Q_CHOICES)
    return Int8AState(f"random_hidden_restart_h{hidden_dim}", seed, hidden_dim, hidden_in_q, hidden_out_q)


def start_state(arm: str, *, hidden_dim: int, seed: int, rng: random.Random) -> Int8AState:
    if arm == "current_int8_polish":
        return load_current_artifact(CURRENT_ARTIFACT, hidden_dim=hidden_dim, seed=seed)
    if arm == "hidden_bridge_restart":
        return hidden_bridge_state(hidden_dim=hidden_dim, seed=seed)
    if arm == "random_hidden_restart":
        return random_hidden_state(hidden_dim=hidden_dim, seed=seed, rng=rng)
    raise ValueError(f"unknown arm: {arm}")


def choose_existing(rng: random.Random, matrix: np.ndarray) -> tuple[int, int] | None:
    coords = list(zip(*np.nonzero(matrix)))
    if not coords:
        return None
    row, col = rng.choice(coords)
    return int(row), int(col)


def mutate_state(state: Int8AState, rng: random.Random) -> tuple[Int8AState, str]:
    next_state = state.clone()
    ops = (
        "reweight_hidden_out",
        "reweight_hidden_out",
        "reweight_hidden_in",
        "add_hidden_out_edge",
        "add_hidden_in_edge",
        "add_path",
        "remove_hidden_out_edge",
        "sign_flip",
        "move_hidden_out_edge",
    )
    op = rng.choice(ops)

    if op == "reweight_hidden_out":
        coord = choose_existing(rng, next_state.hidden_out_q)
        if coord is not None:
            row, col = coord
            next_state.hidden_out_q[row, col] = rng.choice(Q_CHOICES)
        return next_state, op

    if op == "reweight_hidden_in":
        coord = choose_existing(rng, next_state.hidden_in_q)
        if coord is not None:
            row, col = coord
            next_state.hidden_in_q[row, col] = rng.choice(Q_CHOICES)
        return next_state, op

    if op == "add_hidden_out_edge":
        for _ in range(64):
            row = rng.randrange(CODE_DIM)
            col = rng.randrange(next_state.hidden_dim)
            if next_state.hidden_out_q[row, col] == 0:
                next_state.hidden_out_q[row, col] = rng.choice(Q_CHOICES)
                break
        return next_state, op

    if op == "add_hidden_in_edge":
        for _ in range(64):
            row = rng.randrange(next_state.hidden_dim)
            col = rng.randrange(VISIBLE_DIM)
            if next_state.hidden_in_q[row, col] == 0:
                next_state.hidden_in_q[row, col] = rng.choice(Q_CHOICES)
                break
        return next_state, op

    if op == "add_path":
        h = rng.randrange(next_state.hidden_dim)
        v = rng.randrange(VISIBLE_DIM)
        c = rng.randrange(CODE_DIM)
        next_state.hidden_in_q[h, v] = rng.choice(Q_CHOICES)
        next_state.hidden_out_q[c, h] = rng.choice(Q_CHOICES)
        return next_state, op

    if op == "remove_hidden_out_edge" and count_q(next_state.hidden_out_q) > VISIBLE_DIM:
        coord = choose_existing(rng, next_state.hidden_out_q)
        if coord is not None:
            row, col = coord
            next_state.hidden_out_q[row, col] = 0
        return next_state, op

    if op == "sign_flip":
        matrix = rng.choice([next_state.hidden_in_q, next_state.hidden_out_q])
        coord = choose_existing(rng, matrix)
        if coord is not None:
            row, col = coord
            matrix[row, col] = -int(matrix[row, col])
        return next_state, op

    if op == "move_hidden_out_edge":
        coord = choose_existing(rng, next_state.hidden_out_q)
        if coord is not None:
            row, col = coord
            q = int(next_state.hidden_out_q[row, col])
            next_state.hidden_out_q[row, col] = 0
            next_state.hidden_out_q[rng.randrange(CODE_DIM), rng.randrange(next_state.hidden_dim)] = q
        return next_state, op

    return next_state, "noop"


def unique_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key = str(row.get("hidden_in_q_entries", "")) + "|" + str(row.get("hidden_out_q_entries", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def row_rank(row: dict[str, object]) -> tuple[float, ...]:
    """Decision ranking: replacement-safe candidates before copy-like controls."""

    return (
        1.0 if strong_gate(row) else 0.0,
        1.0 if acceptance_gate(row) else 0.0,
        float(row["byte_margin_min"]),
        float(row["ascii_class_geometry"]),
        -float(row["effective_copy_penalty"]),
        float(row.get("single_edge_drop_mean_bit", 0.0)),
        float(row["native_int8_score"]),
    )


def row_to_int8_state(row: dict[str, object]) -> Int8AState:
    hidden_dim = int(row["hidden_dim"])
    seed = int(float(row.get("seed", 0) or 0))
    hidden_in_q = parse_q_entries(str(row["hidden_in_q_entries"]), hidden_dim, VISIBLE_DIM)
    hidden_out_q = parse_q_entries(str(row["hidden_out_q_entries"]), CODE_DIM, hidden_dim)
    return Int8AState(str(row.get("search_arm", row.get("arm", "candidate"))), seed, hidden_dim, hidden_in_q, hidden_out_q)


def run_one_combo(
    *,
    hidden_dim: int,
    seed: int,
    arm: str,
    args: argparse.Namespace,
    candidate_id_start: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    rng = random.Random(seed * 1009 + hidden_dim * 97 + sum(ord(ch) for ch in arm))
    start = start_state(arm, hidden_dim=hidden_dim, seed=seed, rng=rng)
    candidate_id = candidate_id_start
    start_row = evaluate_int8_state(start, candidate_id=candidate_id, source="start", step=0, include_robustness=True, mutation="start")
    candidate_id += 1

    beams: list[tuple[Int8AState, dict[str, object]]] = [(start, start_row)]
    path_rows: list[dict[str, object]] = [start_row]
    best_rows: list[dict[str, object]] = [start_row]
    best_score = float(start_row["native_int8_score"])

    for step in range(1, int(args.max_steps) + 1):
        proposals: list[tuple[Int8AState, dict[str, object]]] = []
        for _ in range(max(1, int(args.workers))):
            parent, _parent_row = rng.choice(beams)
            proposal, mutation = mutate_state(parent, rng)
            row = evaluate_int8_state(
                proposal,
                candidate_id=candidate_id,
                source="proposal",
                step=step,
                include_robustness=False,
                mutation=mutation,
            )
            candidate_id += 1
            if acceptance_gate(row) or float(row["native_int8_score"]) >= best_score - 4.0:
                proposals.append((proposal, row))

            if acceptance_gate(row) and float(row["native_int8_score"]) > best_score + EPS:
                robust = evaluate_int8_state(
                    proposal,
                    candidate_id=candidate_id,
                    source="accepted",
                    step=step,
                    include_robustness=True,
                    mutation=mutation,
                )
                candidate_id += 1
                if acceptance_gate(robust):
                    best_score = float(robust["native_int8_score"])
                    path_rows.append(robust)
                    best_rows.append(robust)
                    proposals.append((proposal, robust))

        combined = beams + proposals
        combined = sorted(combined, key=lambda item: float(item[1]["native_int8_score"]), reverse=True)
        next_beams: list[tuple[Int8AState, dict[str, object]]] = []
        seen: set[str] = set()
        for state, row in combined:
            key = key_for_state(state)
            if key in seen:
                continue
            seen.add(key)
            next_beams.append((state, row))
            if len(next_beams) >= int(args.beam_width):
                break
        beams = next_beams or beams

        if step % 250 == 0:
            top = beams[0][1]
            print(
                f"[A-v2-int8] h={hidden_dim} seed={seed} arm={arm} step={step} "
                f"score={float(top['native_int8_score']):.2f} margin={float(top['byte_margin_min']):.3f} "
                f"geom={float(top['ascii_class_geometry']):.3f}",
                flush=True,
            )

    final_rows: list[dict[str, object]] = []
    for state, row in beams[: max(1, int(args.beam_width))]:
        final_rows.append(
            evaluate_int8_state(
                state,
                candidate_id=int(row["candidate_id"]),
                source="final_beam",
                step=int(row["step"]),
                include_robustness=True,
                mutation=str(row["mutation"]),
            )
        )
    all_rows = unique_rows(path_rows + best_rows + final_rows)
    all_rows = sorted(all_rows, key=lambda row: float(row["native_int8_score"]), reverse=True)
    return path_rows, all_rows, candidate_id


def control_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    direct_state = HiddenAState(
        "A-StableCopy16-control",
        0,
        direct_redundant_matrix(),
        np.zeros((0, VISIBLE_DIM), dtype=np.float64),
        np.zeros((CODE_DIM, 0), dtype=np.float64),
    )
    stable = evaluate_state(direct_state, candidate_id=-1, source="control", step=0, include_robustness=True)
    stable["native_storage"] = "direct_float_reference"
    stable["native_int8_score"] = score({**stable, "effective_rank": 8.0})
    rows.append(stable)

    current = load_current_artifact(CURRENT_ARTIFACT, hidden_dim=8, seed=0)
    rows.append(evaluate_int8_state(current, candidate_id=-2, source="control", step=0, include_robustness=True, mutation="control"))
    return rows


def summarize_by_hidden_dim(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_dim: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_dim.setdefault(int(row["hidden_dim"]), []).append(row)
    for hidden_dim, dim_rows in sorted(by_dim.items()):
        accepted = [row for row in dim_rows if acceptance_gate(row)]
        strong = [row for row in dim_rows if strong_gate(row)]
        best_pool = accepted if accepted else dim_rows
        best = max(best_pool, key=row_rank)
        out.append(
            {
                "hidden_dim": hidden_dim,
                "row_count": len(dim_rows),
                "accepted_count": len(accepted),
                "strong_count": len(strong),
                "best_margin": float(best["byte_margin_min"]),
                "best_geometry": float(best["ascii_class_geometry"]),
                "best_copy_penalty": float(best["effective_copy_penalty"]),
                "best_score": float(best["native_int8_score"]),
                "best_seed": best.get("seed", ""),
                "best_arm": best.get("search_arm", best.get("arm", "")),
            }
        )
    return out


def artifact_from_row(row: dict[str, object], *, name: str) -> dict[str, object]:
    hidden_dim = int(row["hidden_dim"])
    payload: dict[str, object] = {
        "version": VERSION,
        "name": name,
        "storage": "int8_q6",
        "scale": SCALE,
        "value_formula": "weight = q / 64",
        "visible_dim": VISIBLE_DIM,
        "hidden_dim": hidden_dim,
        "code_dim": CODE_DIM,
        "decoder": "transpose_chain",
        "source": "a_v2_native_int8_seed_sweep",
        "source_verdict": "",
        "source_metrics": {
            "exact_byte_acc": row.get("exact_byte_acc"),
            "bit_acc": row.get("bit_acc"),
            "byte_margin_min": row.get("byte_margin_min"),
            "ascii_class_geometry": row.get("ascii_class_geometry"),
            "effective_copy_penalty": row.get("effective_copy_penalty"),
        },
        "hidden_in_q": q_entries(parse_q_entries(str(row["hidden_in_q_entries"]), hidden_dim, VISIBLE_DIM)),
        "hidden_out_q": q_entries(parse_q_entries(str(row["hidden_out_q_entries"]), CODE_DIM, hidden_dim)),
    }
    payload["verification"] = verify_payload(payload)
    return payload


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    accepted = [row for row in rows if acceptance_gate(row)]
    if not accepted:
        return "A_V2_INT8_SEARCH_FAIL", "no exact non-copy candidate survived the native int8 search"

    strong = [row for row in accepted if strong_gate(row)]
    if strong:
        best = max(strong, key=lambda row: (float(row["byte_margin_min"]), float(row["ascii_class_geometry"])))
        return (
            "A_V2_INT8_STRONG_REPLACEMENT_CANDIDATE",
            f"best margin {float(best['byte_margin_min']):.3f}, geometry {float(best['ascii_class_geometry']):.3f}, h={int(best['hidden_dim'])}, seed={best.get('seed')}",
        )

    high_margin = [
        row
        for row in accepted
        if float(row["byte_margin_min"]) >= 4.0
        and float(row["effective_copy_penalty"]) <= 0.10
    ]
    if high_margin:
        best = max(high_margin, key=lambda row: (float(row["ascii_class_geometry"]), float(row.get("single_edge_drop_mean_bit", 0.0))))
        return (
            "A_V2_INT8_MARGIN_STRONG_GEOMETRY_NEAR_MISS",
            f"margin reached {float(best['byte_margin_min']):.3f}, but geometry {float(best['ascii_class_geometry']):.3f} is below the 0.760 strong gate",
        )

    high_geometry = [row for row in accepted if float(row["ascii_class_geometry"]) >= 0.760]
    if high_geometry:
        best = max(high_geometry, key=lambda row: (float(row["byte_margin_min"]), float(row["ascii_class_geometry"])))
        return (
            "A_V2_INT8_GEOMETRY_STRONG_MARGIN_WEAK",
            f"geometry stays strong at {float(best['ascii_class_geometry']):.3f}, but margin is {float(best['byte_margin_min']):.3f}",
        )

    copy_like = [
        row
        for row in rows
        if gate_pass(row) and float(row["byte_margin_min"]) >= 4.0 and float(row["effective_copy_penalty"]) > 0.10
    ]
    if copy_like:
        return "A_V2_INT8_COPY_REGRESSION", "margin improved only in copy-like candidates"

    seeds = {int(row["seed"]) for row in accepted if "seed" in row}
    if len(seeds) <= 2:
        return "A_V2_INT8_NO_STABLE_SEED_SIGNAL", "accepted candidates were too seed-sparse to treat as stable"

    return "A_V2_INT8_GEOMETRY_STRONG_MARGIN_WEAK", "accepted candidates exist, but no strong replacement candidate passed"


def write_report(out: Path, rows: Sequence[dict[str, object]], summary_rows: Sequence[dict[str, object]], verdict_name: str, reason: str, meta: dict[str, object]) -> None:
    lines = [
        "# A-v2 Native Int8 Seed Sweep",
        "",
        "Date: 2026-05-03",
        "",
        "## Verdict",
        "",
        "```text",
        verdict_name,
        reason,
        "```",
        "",
        "## Top Candidates",
        "",
        "```text",
        "rank h seed arm                         exact margin geom  copy robust score",
        "---- - ---- --------------------------- ----- ------ ----- ---- ------ ------",
    ]
    for idx, row in enumerate(rows[:24], start=1):
        lines.append(
            f"{idx:4d} "
            f"{int(row['hidden_dim']):1d} "
            f"{int(float(row.get('seed', 0) or 0)):4d} "
            f"{str(row.get('search_arm', row.get('arm', '')))[:27]:27} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):5.3f} "
            f"{float(row['effective_copy_penalty']):4.2f} "
            f"{float(row.get('single_edge_drop_mean_bit', 0.0)):6.3f} "
            f"{float(row['native_int8_score']):6.2f}"
        )
    lines.extend(["```", "", "## Hidden-Dim Summary", "", "```text"])
    for row in summary_rows:
        lines.append(
            f"h={int(row['hidden_dim']):2d} rows={int(row['row_count']):4d} accepted={int(row['accepted_count']):3d} "
            f"strong={int(row['strong_count']):2d} best_margin={float(row['best_margin']):.3f} "
            f"best_geom={float(row['best_geometry']):.3f} seed={row['best_seed']} arm={row['best_arm']}"
        )
    lines.extend(["```", "", "## Meta", "", "```json", json.dumps(meta, indent=2), "```", ""])
    (out / "A_V2_NATIVE_INT8_SEED_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(out: Path, rows: Sequence[dict[str, object]], path_rows: Sequence[dict[str, object]], meta: dict[str, object]) -> tuple[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    candidates_dir = out / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    rows_sorted = sorted(unique_rows(rows), key=row_rank, reverse=True)
    top_rows = rows_sorted[: max(1, int(meta.get("top_k", 24)))]
    summary_rows = summarize_by_hidden_dim(rows_sorted)
    controls = control_rows()
    verdict_name, reason = verdict(rows_sorted)

    write_csv(out / "seed_sweep_results.csv", rows_sorted)
    write_csv(out / "seed_sweep_top_candidates.csv", top_rows)
    write_csv(out / "seed_sweep_by_hidden_dim.csv", summary_rows)
    write_csv(out / "seed_sweep_controls.csv", controls)

    exported = []
    for idx, row in enumerate(top_rows[:24], start=1):
        payload = artifact_from_row(row, name=f"A-v2-native-int8-top-{idx:02d}")
        artifact_path = candidates_dir / f"top_{idx:02d}.json"
        artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exported.append({"rank": idx, "path": str(artifact_path), "verification": payload["verification"]})

    payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "meta": meta,
        "candidate_count": len(rows_sorted),
        "acceptance_gate_count": sum(1 for row in rows_sorted if acceptance_gate(row)),
        "strong_gate_count": sum(1 for row in rows_sorted if strong_gate(row)),
        "best_candidate": rows_sorted[0] if rows_sorted else None,
        "exported_candidates": exported,
    }
    (out / "a_v2_native_int8_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, rows_sorted, summary_rows, verdict_name, reason, meta)
    return verdict_name, reason


def run_search(args: argparse.Namespace) -> int:
    t0 = time.time()
    deadline = t0 + float(args.time_budget_s) if float(args.time_budget_s) > 0 else None
    out = Path(args.out)
    hidden_dims = parse_csv_ints(args.hidden_dims)
    arms = parse_csv_strings(args.arms)
    if args.seeds:
        seeds = parse_csv_ints(args.seeds)
    else:
        seeds = [int(args.seed_start) + idx for idx in range(int(args.seed_count))]

    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    candidate_id = 0
    for hidden_dim in hidden_dims:
        for seed in seeds:
            for arm in arms:
                if deadline is not None and time.time() >= deadline:
                    print("[A-v2-int8] time budget reached before next combo", flush=True)
                    break
                print(f"[A-v2-int8] start h={hidden_dim} seed={seed} arm={arm}", flush=True)
                combo_path, combo_rows, candidate_id = run_one_combo(
                    hidden_dim=hidden_dim,
                    seed=seed,
                    arm=arm,
                    args=args,
                    candidate_id_start=candidate_id,
                )
                path_rows.extend(combo_path)
                rows.extend(combo_rows)
                out.mkdir(parents=True, exist_ok=True)
                (out / "seed_sweep_progress.json").write_text(
                    json.dumps(
                        {
                            "hidden_dim": hidden_dim,
                            "seed": seed,
                            "arm": arm,
                            "rows_so_far": len(rows),
                            "path_rows_so_far": len(path_rows),
                            "runtime_s": time.time() - t0,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if deadline is not None and time.time() >= deadline:
                break
        if deadline is not None and time.time() >= deadline:
            break

    meta = {
        "mode": args.mode,
        "hidden_dims": hidden_dims,
        "seeds": seeds,
        "arms": arms,
        "max_steps": int(args.max_steps),
        "beam_width": int(args.beam_width),
        "workers": int(args.workers),
        "top_k": int(args.top_k),
        "time_budget_s": float(args.time_budget_s),
        "runtime_s": time.time() - t0,
    }
    verdict_name, reason = write_outputs(out, rows, path_rows, meta)
    print(f"[A-v2-int8] verdict={verdict_name} reason={reason}")
    print(f"[A-v2-int8] wrote {out}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    path = Path(args.candidates)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows_in = list(csv.DictReader(handle))
    rows_in = sorted(rows_in, key=lambda row: (
        str(row.get("acceptance_gate", "")).lower() == "true",
        float(row.get("byte_margin_min", 0.0)),
        float(row.get("ascii_class_geometry", 0.0)),
        -float(row.get("effective_copy_penalty", 1.0)),
        float(row.get("native_int8_score", 0.0)),
    ), reverse=True)

    confirmed: list[dict[str, object]] = []
    for idx, row in enumerate(rows_in[: int(args.top_k)]):
        state = row_to_int8_state(row)
        confirmed.append(
            evaluate_int8_state(
                state,
                candidate_id=idx,
                source="confirm",
                step=int(float(row.get("step", 0) or 0)),
                include_robustness=True,
                mutation=str(row.get("mutation", "confirm")),
            )
        )

    meta = {
        "mode": "confirm",
        "candidates": str(path),
        "top_k": int(args.top_k),
        "input_count": len(rows_in),
    }
    verdict_name, reason = write_outputs(Path(args.out), confirmed, confirmed, meta)
    print(f"[A-v2-int8] verdict={verdict_name} reason={reason}")
    print(f"[A-v2-int8] wrote {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "main", "confirm"), required=True)
    parser.add_argument("--hidden-dims", default="8,12,16")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed-start", type=int, default=20260503)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--arms", default="current_int8_polish,hidden_bridge_restart,random_hidden_restart")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--time-budget-s", type=float, default=0.0)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    if args.mode == "confirm":
        if not args.candidates:
            raise SystemExit("--candidates is required for confirm mode")
        return run_confirm(args)
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
