#!/usr/bin/env python3
"""
D21I A-block hidden sparse sweep.

This phase tests whether fixed byte IO plus extra hidden link locations can beat
the direct reciprocal A-block. It keeps the byte task exhaustive over all 256
bytes, but compares three structural families:

- io_only: direct visible->A16 reciprocal surface
- hidden_only: visible->hidden->A16, no direct visible->A16 highway
- overlay_locked: D21A direct highway plus sparse hidden residual paths

The hidden model is linear/tied for this proof:

    effective_encoder = direct + code_from_hidden @ hidden_from_visible

and decoding uses the tied mirror of that effective encoder. This isolates the
question "do hidden link locations help the A geometry?" without introducing a
new activation/function confound.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    redundant_copy_entries,
    robustness_metrics,
)
from tools._scratch.d21f_ablock_natural_geometry import (  # noqa: E402
    ascii_geometry_score,
    duplicate_lane_penalty,
    geometry_metrics,
    identity_copy_penalty,
    pairwise_distances,
    parse_entries,
)


VISIBLE_DIM = 8
CODE_DIM = 16
WEIGHTS = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
SMALL_WEIGHTS = (-0.5, -0.25, -0.125, 0.125, 0.25, 0.5)


@dataclass(frozen=True)
class HiddenAState:
    arm: str
    hidden_dim: int
    direct: np.ndarray
    hidden_from_visible: np.ndarray
    code_from_hidden: np.ndarray

    def effective_encoder(self) -> np.ndarray:
        if self.hidden_dim <= 0:
            return self.direct.copy()
        return self.direct + self.code_from_hidden @ self.hidden_from_visible

    def clone(self) -> "HiddenAState":
        return HiddenAState(
            arm=self.arm,
            hidden_dim=self.hidden_dim,
            direct=self.direct.copy(),
            hidden_from_visible=self.hidden_from_visible.copy(),
            code_from_hidden=self.code_from_hidden.copy(),
        )


def matrix_to_entries(matrix: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    entries: list[tuple[int, int, float]] = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if abs(value) > 1e-12:
                entries.append((row, col, value))
    return dedupe_entries(entries)


def entries_to_matrix(entries: Iterable[tuple[int, int, float]], rows: int, cols: int) -> np.ndarray:
    matrix = np.zeros((rows, cols), dtype=np.float64)
    for row, col, value in entries:
        if 0 <= row < rows and 0 <= col < cols:
            matrix[row, col] = float(value)
    return matrix


def direct_redundant_matrix() -> np.ndarray:
    return entries_to_matrix(redundant_copy_entries(VISIBLE_DIM, CODE_DIM), CODE_DIM, VISIBLE_DIM)


def load_d21g_direct(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    candidate = blob.get("best_natural_candidate")
    if not candidate:
        return None
    return entries_to_matrix(parse_entries(str(candidate["entries"])), CODE_DIM, VISIBLE_DIM)


def hidden_bridge_state(arm: str, hidden_dim: int) -> HiddenAState:
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    hidden_from_visible = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.float64)
    code_from_hidden = np.zeros((CODE_DIM, hidden_dim), dtype=np.float64)
    for bit in range(min(VISIBLE_DIM, hidden_dim)):
        hidden_from_visible[bit, bit] = 1.0
        code_from_hidden[bit, bit] = 1.0
        code_from_hidden[VISIBLE_DIM + bit, bit] = 1.0
    return HiddenAState(arm, hidden_dim, direct, hidden_from_visible, code_from_hidden)


def overlay_state(arm: str, hidden_dim: int) -> HiddenAState:
    return HiddenAState(
        arm,
        hidden_dim,
        direct_redundant_matrix(),
        np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.float64),
        np.zeros((CODE_DIM, hidden_dim), dtype=np.float64),
    )


def io_state(arm: str, direct: np.ndarray) -> HiddenAState:
    return HiddenAState(
        arm,
        0,
        direct.copy(),
        np.zeros((0, VISIBLE_DIM), dtype=np.float64),
        np.zeros((CODE_DIM, 0), dtype=np.float64),
    )


def count_nonzero(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(matrix) > 1e-12))


def hidden_active_count(state: HiddenAState) -> int:
    if state.hidden_dim <= 0:
        return 0
    active = 0
    for idx in range(state.hidden_dim):
        if np.any(np.abs(state.hidden_from_visible[idx]) > 1e-12) and np.any(np.abs(state.code_from_hidden[:, idx]) > 1e-12):
            active += 1
    return active


def structural_direct_copy_penalty(state: HiddenAState) -> float:
    direct_entries = matrix_to_entries(state.direct)
    return identity_copy_penalty(direct_entries, VISIBLE_DIM, CODE_DIM)


def gate_pass(row: dict[str, object]) -> bool:
    return (
        float(row["exact_byte_acc"]) == 1.0
        and float(row["bit_acc"]) == 1.0
        and float(row["byte_argmax_acc"]) == 1.0
        and int(row["hidden_collisions"]) == 0
        and float(row["byte_margin_min"]) > 0.0
        and float(row["reciprocity_error"]) == 0.0
    )


def hidden_sweep_score(row: dict[str, object]) -> float:
    if not bool(row["gate_pass"]):
        return (
            -100.0
            + 40.0 * float(row["exact_byte_acc"])
            + 10.0 * float(row["bit_acc"])
            + 8.0 * float(row["byte_argmax_acc"])
        )
    no_direct_bonus = 2.25 if int(row["direct_edge_count"]) == 0 else 0.0
    return float(
        25.0
        + 1.20 * min(4.0, max(0.0, float(row["byte_margin_min"])))
        + 10.0 * float(row["ascii_class_geometry"])
        + 1.0 * float(row["single_edge_drop_mean_bit"])
        + no_direct_bonus
        - 1.2 * float(row["effective_copy_penalty"])
        - 0.020 * float(row["structural_edge_count"])
    )


def evaluate_state(
    state: HiddenAState,
    *,
    candidate_id: int,
    source: str,
    step: int,
    include_robustness: bool,
) -> dict[str, object]:
    effective = state.effective_encoder()
    effective_entries = matrix_to_entries(effective)
    block = ReciprocalABlock(CODE_DIM // 2, CODE_DIM, effective)
    patterns = all_visible_patterns(VISIBLE_DIM)
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "arm": state.arm,
        "source": source,
        "step": step,
        "visible_dim": VISIBLE_DIM,
        "code_dim": CODE_DIM,
        "hidden_dim": state.hidden_dim,
        "direct_edge_count": count_nonzero(state.direct),
        "hidden_in_edge_count": count_nonzero(state.hidden_from_visible),
        "hidden_out_edge_count": count_nonzero(state.code_from_hidden),
        "hidden_active_count": hidden_active_count(state),
        "structural_edge_count": count_nonzero(state.direct) + count_nonzero(state.hidden_from_visible) + count_nonzero(state.code_from_hidden),
        "effective_edge_count": len(effective_entries),
        "effective_entries": entries_to_string(effective_entries),
        "direct_entries": entries_to_string(matrix_to_entries(state.direct)),
        "hidden_in_entries": entries_to_string(matrix_to_entries(state.hidden_from_visible)),
        "hidden_out_entries": entries_to_string(matrix_to_entries(state.code_from_hidden)),
    }
    row.update(evaluate_ablock(block, patterns))
    if include_robustness:
        row.update(robustness_metrics(block, patterns))
    else:
        row.update(
            {
                "single_edge_drop_min_exact": 0.0,
                "single_edge_drop_mean_exact": 0.0,
                "single_edge_drop_min_bit": 0.0,
                "single_edge_drop_mean_bit": 0.0,
                "single_edge_drop_min_byte_margin": 0.0,
            }
        )
    row.update(geometry_metrics(block))
    row["ascii_class_geometry"] = ascii_geometry_score(row)
    row["effective_copy_penalty"] = identity_copy_penalty(effective_entries, VISIBLE_DIM, CODE_DIM)
    row["structural_direct_copy_penalty"] = structural_direct_copy_penalty(state)
    row["duplicate_lane_penalty"] = duplicate_lane_penalty(block)
    row["gate_pass"] = gate_pass(row)
    row["D21I_hidden_score"] = hidden_sweep_score(row)
    return row


def random_weight(rng: random.Random, *, small: bool) -> float:
    return rng.choice(SMALL_WEIGHTS if small else WEIGHTS)


def mutate_state(state: HiddenAState, rng: random.Random, *, lock_direct: bool) -> HiddenAState:
    next_state = state.clone()
    if next_state.hidden_dim <= 0:
        return next_state

    ops = ["add_path", "add_in", "add_out", "remove", "reweight", "sign", "move"]
    if not lock_direct:
        ops.extend(["direct_add", "direct_remove", "direct_reweight"])
    op = rng.choice(ops)
    small = lock_direct

    if op == "add_path":
        h = rng.randrange(next_state.hidden_dim)
        v = rng.randrange(VISIBLE_DIM)
        c = rng.randrange(CODE_DIM)
        next_state.hidden_from_visible[h, v] = random_weight(rng, small=small)
        next_state.code_from_hidden[c, h] = random_weight(rng, small=small)
        return next_state

    if op == "add_in":
        next_state.hidden_from_visible[rng.randrange(next_state.hidden_dim), rng.randrange(VISIBLE_DIM)] = random_weight(rng, small=small)
        return next_state

    if op == "add_out":
        next_state.code_from_hidden[rng.randrange(CODE_DIM), rng.randrange(next_state.hidden_dim)] = random_weight(rng, small=small)
        return next_state

    if op == "remove":
        matrices = [next_state.hidden_from_visible, next_state.code_from_hidden]
        if not lock_direct:
            matrices.append(next_state.direct)
        nonempty = [mat for mat in matrices if count_nonzero(mat) > 0]
        if nonempty:
            mat = rng.choice(nonempty)
            coords = list(zip(*np.nonzero(np.abs(mat) > 1e-12)))
            row, col = rng.choice(coords)
            mat[row, col] = 0.0
        return next_state

    if op in {"reweight", "sign", "move"}:
        matrices = [next_state.hidden_from_visible, next_state.code_from_hidden]
        if not lock_direct:
            matrices.append(next_state.direct)
        nonempty = [mat for mat in matrices if count_nonzero(mat) > 0]
        if not nonempty:
            return next_state
        mat = rng.choice(nonempty)
        coords = list(zip(*np.nonzero(np.abs(mat) > 1e-12)))
        row, col = rng.choice(coords)
        if op == "reweight":
            mat[row, col] = random_weight(rng, small=small)
        elif op == "sign":
            mat[row, col] = -float(mat[row, col])
        else:
            value = float(mat[row, col])
            mat[row, col] = 0.0
            mat[rng.randrange(mat.shape[0]), rng.randrange(mat.shape[1])] = value
        return next_state

    if op == "direct_add":
        next_state.direct[rng.randrange(CODE_DIM), rng.randrange(VISIBLE_DIM)] = random_weight(rng, small=False)
        return next_state

    if op == "direct_remove" and count_nonzero(next_state.direct) > 0:
        coords = list(zip(*np.nonzero(np.abs(next_state.direct) > 1e-12)))
        row, col = rng.choice(coords)
        next_state.direct[row, col] = 0.0
        return next_state

    if op == "direct_reweight" and count_nonzero(next_state.direct) > 0:
        coords = list(zip(*np.nonzero(np.abs(next_state.direct) > 1e-12)))
        row, col = rng.choice(coords)
        next_state.direct[row, col] = random_weight(rng, small=False)
        return next_state

    return next_state


def parse_hidden_dims(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def initial_states(hidden_dims: Sequence[int], d21g_path: Path) -> list[HiddenAState]:
    states = [io_state("io_only_d21a", direct_redundant_matrix())]
    d21g = load_d21g_direct(d21g_path)
    if d21g is not None:
        states.append(io_state("io_only_d21g_natural", d21g))
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            continue
        states.append(hidden_bridge_state(f"hidden_only_h{hidden_dim}", hidden_dim))
        states.append(overlay_state(f"overlay_locked_h{hidden_dim}", hidden_dim))
    return states


def run_search(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = random.Random(int(args.seed))
    rows: list[dict[str, object]] = []
    candidate_id = 0
    beams: list[tuple[HiddenAState, dict[str, object]]] = []
    hidden_dims = parse_hidden_dims(str(args.hidden_dims))

    for state in initial_states(hidden_dims, Path(args.d21g)):
        row = evaluate_state(state, candidate_id=candidate_id, source="start", step=0, include_robustness=True)
        candidate_id += 1
        rows.append(row)
        beams.append((state, row))

    for step in range(1, int(args.max_steps) + 1):
        next_beams: list[tuple[HiddenAState, dict[str, object]]] = []
        for state, current_row in beams:
            if state.hidden_dim <= 0:
                next_beams.append((state, current_row))
                continue
            lock_direct = str(state.arm).startswith("overlay_locked")
            best_state = state
            best_row = current_row
            for _ in range(max(1, int(args.workers))):
                proposal = mutate_state(state, rng, lock_direct=lock_direct)
                row = evaluate_state(proposal, candidate_id=candidate_id, source="proposal", step=step, include_robustness=False)
                candidate_id += 1
                rows.append(row)
                if float(row["D21I_hidden_score"]) > float(best_row["D21I_hidden_score"]) + 1e-9 and (
                    bool(row["gate_pass"]) or not bool(best_row["gate_pass"])
                ):
                    best_state = proposal
                    best_row = row
            if best_state is not state:
                robust = evaluate_state(best_state, candidate_id=candidate_id, source="accepted", step=step, include_robustness=True)
                candidate_id += 1
                rows.append(robust)
                next_beams.append((best_state, robust))
            else:
                next_beams.append((state, current_row))
        beams = next_beams

    top = sorted(rows, key=lambda row: float(row["D21I_hidden_score"]), reverse=True)[:64]
    robust_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in top:
        state = row_to_state(row)
        robust_by_key[(str(row["arm"]), str(row["effective_entries"]))] = evaluate_state(
            state,
            candidate_id=int(row["candidate_id"]),
            source=str(row["source"]),
            step=int(row["step"]),
            include_robustness=True,
        )
    final_rows = [robust_by_key.get((str(row["arm"]), str(row["effective_entries"])), row) for row in rows]
    return sorted(final_rows, key=lambda row: float(row["D21I_hidden_score"]), reverse=True)


def row_to_state(row: dict[str, object]) -> HiddenAState:
    hidden_dim = int(row["hidden_dim"])
    direct = entries_to_matrix(parse_entries(str(row["direct_entries"])), CODE_DIM, VISIBLE_DIM)
    hidden_in = entries_to_matrix(parse_entries(str(row["hidden_in_entries"])), hidden_dim, VISIBLE_DIM)
    hidden_out = entries_to_matrix(parse_entries(str(row["hidden_out_entries"])), CODE_DIM, hidden_dim)
    return HiddenAState(str(row["arm"]), hidden_dim, direct, hidden_in, hidden_out)


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    baseline = next(row for row in rows if row["arm"] == "io_only_d21a")
    hidden_exact = [
        row
        for row in rows
        if bool(row["gate_pass"])
        and int(row["hidden_dim"]) > 0
        and int(row["direct_edge_count"]) == 0
    ]
    overlay_exact = [
        row
        for row in rows
        if bool(row["gate_pass"])
        and str(row["arm"]).startswith("overlay_locked")
    ]
    best_hidden = max(hidden_exact, key=lambda row: float(row["D21I_hidden_score"]), default=None)
    best_overlay = max(overlay_exact, key=lambda row: float(row["D21I_hidden_score"]), default=None)

    if best_hidden is not None:
        geom_gain = float(best_hidden["ascii_class_geometry"]) - float(baseline["ascii_class_geometry"])
        margin_ok = float(best_hidden["byte_margin_min"]) >= float(baseline["byte_margin_min"]) - 1e-9
        if geom_gain > 0.03 and margin_ok:
            if float(best_hidden["effective_copy_penalty"]) > 0.75:
                return (
                    "D21I_HIDDEN_BIT_GAIN_PASS",
                    f"{best_hidden['arm']} is exact/no-direct and improves geometry by {geom_gain:+.4f}, but the effective map is still copy-like",
                )
            return (
                "D21I_HIDDEN_SPARSE_A_PASS",
                f"{best_hidden['arm']} is exact/no-direct and improves geometry by {geom_gain:+.4f}",
            )
        if margin_ok and abs(geom_gain) <= 0.03:
            return (
                "D21I_HIDDEN_REPLICATES_IO_ONLY",
                f"{best_hidden['arm']} exactly replicates the IO-only A behavior through hidden link locations, but does not improve geometry",
            )

    if best_overlay is not None:
        geom_gain = float(best_overlay["ascii_class_geometry"]) - float(baseline["ascii_class_geometry"])
        if geom_gain > 0.03:
            return (
                "D21I_OVERLAY_HIDDEN_GEOMETRY_LEAD",
                f"{best_overlay['arm']} improves geometry by {geom_gain:+.4f}, but still relies on the direct IO highway",
            )

    return "D21I_IO_ONLY_STILL_BEST", "extra hidden link locations did not beat the current direct D21A A-block"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def topology_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        for matrix_name, key in [
            ("direct", "direct_entries"),
            ("hidden_in", "hidden_in_entries"),
            ("hidden_out", "hidden_out_entries"),
            ("effective", "effective_entries"),
        ]:
            for edge_idx, (left, right, value) in enumerate(parse_entries(str(row[key]))):
                out.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "arm": row["arm"],
                        "matrix": matrix_name,
                        "edge_idx": edge_idx,
                        "left": left,
                        "right": right,
                        "weight": value,
                    }
                )
    return out


def distance_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    probes = [
        ("A-B", ord("A"), ord("B")),
        ("A-Z", ord("A"), ord("Z")),
        ("A-a", ord("A"), ord("a")),
        ("A-7", ord("A"), ord("7")),
        ("1-2", ord("1"), ord("2")),
        ("1-Z", ord("1"), ord("Z")),
        (",-.", ord(","), ord(".")),
        (",-A", ord(","), ord("A")),
    ]
    out: list[dict[str, object]] = []
    for row in rows:
        effective = entries_to_matrix(parse_entries(str(row["effective_entries"])), CODE_DIM, VISIBLE_DIM)
        block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, effective)
        dist = pairwise_distances(block.encode_patterns(all_visible_patterns(VISIBLE_DIM)))
        for label, left, right in probes:
            out.append(
                {
                    "candidate_id": row["candidate_id"],
                    "arm": row["arm"],
                    "probe": label,
                    "left_byte": left,
                    "right_byte": right,
                    "distance": float(dist[left, right]),
                }
            )
    return out


def ascii_map(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "D21I hidden sparse A-block sweep",
        "",
        "Legend: direct=visible->A16 edges, h_in=visible->hidden, h_out=hidden->A16.",
        "",
        "arm                    H exact margin  geom direct h_in h_out eff copy score",
        "---------------------- -- ----- ------ ------ ------ ---- ----- --- ---- ------",
    ]
    for row in rows[:18]:
        lines.append(
            f"{str(row['arm'])[:22]:22} "
            f"{int(row['hidden_dim']):2d} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):6.3f} "
            f"{int(row['direct_edge_count']):6d} "
            f"{int(row['hidden_in_edge_count']):4d} "
            f"{int(row['hidden_out_edge_count']):5d} "
            f"{int(row['effective_edge_count']):3d} "
            f"{float(row['effective_copy_penalty']):4.2f} "
            f"{float(row['D21I_hidden_score']):6.2f}"
        )
    lines.append("")
    if rows:
        best = rows[0]
        effective = entries_to_matrix(parse_entries(str(best["effective_entries"])), CODE_DIM, VISIBLE_DIM)
        block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, effective)
        dist = pairwise_distances(block.encode_patterns(all_visible_patterns(VISIBLE_DIM)))
        probes = [
            ("A-B", ord("A"), ord("B")),
            ("A-Z", ord("A"), ord("Z")),
            ("A-a", ord("A"), ord("a")),
            ("A-7", ord("A"), ord("7")),
            ("1-2", ord("1"), ord("2")),
            ("1-Z", ord("1"), ord("Z")),
        ]
        max_dist = max(float(dist[left, right]) for _label, left, right in probes) or 1.0
        lines.append(f"Best probe distances: {best['arm']}")
        for label, left, right in probes:
            value = float(dist[left, right])
            bar = "#" * max(1, int(28 * value / max_dist))
            lines.append(f"  {label:4} {value:7.3f} {bar}")
    return "\n".join(lines) + "\n"


def write_report(out: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, config: dict[str, object]) -> None:
    baseline = next(row for row in rows if row["arm"] == "io_only_d21a")
    best = rows[0]
    lines = [
        "# D21I A-Block Hidden Sparse Sweep",
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
        "## What Was Tested",
        "",
        "```text",
        "io_only:        visible bits -> A16",
        "hidden_only:    visible bits -> hidden -> A16, no direct highway",
        "overlay_locked: D21A highway + hidden residual paths",
        "```",
        "",
        "The hidden proof uses a tied linear effective encoder:",
        "",
        "```text",
        "effective_encoder = direct + code_from_hidden @ hidden_from_visible",
        "decode = effective_encoder.T",
        "```",
        "",
        "## Baseline vs Best",
        "",
        "```text",
        f"baseline io_only_d21a: exact={float(baseline['exact_byte_acc']):.3f} margin={float(baseline['byte_margin_min']):.3f} geometry={float(baseline['ascii_class_geometry']):.3f} structural_edges={int(baseline['structural_edge_count'])}",
        f"best {best['arm']}: exact={float(best['exact_byte_acc']):.3f} margin={float(best['byte_margin_min']):.3f} geometry={float(best['ascii_class_geometry']):.3f} structural_edges={int(best['structural_edge_count'])}",
        "```",
        "",
        "## Top Rows",
        "",
        "```text",
        ascii_map(rows).strip(),
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
    ]
    (out / "D21I_ABLOCK_HIDDEN_SPARSE_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = run_search(args)
    verdict_name, reason = verdict(rows)
    config = {
        "mode": args.mode,
        "hidden_dims": parse_hidden_dims(str(args.hidden_dims)),
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "d21g": str(args.d21g),
    }
    top_rows = rows[: max(1, int(args.top_k))]
    write_csv(out / "hidden_sweep_results.csv", rows)
    write_csv(out / "hidden_topology_edges.csv", topology_rows(top_rows[:6]))
    write_csv(out / "hidden_distance_matrix.csv", distance_rows(top_rows[:6]))
    (out / "hidden_ascii_map.txt").write_text(ascii_map(rows), encoding="utf-8")
    payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "config": config,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if bool(row["gate_pass"])),
        "best_candidate": rows[0] if rows else None,
        "baseline": next((row for row in rows if row["arm"] == "io_only_d21a"), None),
    }
    (out / "hidden_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, rows, verdict_name, reason, config)
    print(ascii_map(rows))
    print(f"[D21I] verdict={verdict_name} reason={reason}")
    print(f"[D21I] wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21I hidden sparse A-block sweep")
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--hidden-dims", default="4,8,16,32")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--d21g", default="output/phase_d21g_ablock_margin_aware_polish_20260503/main/margin_top.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
