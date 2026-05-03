#!/usr/bin/env python3
"""
D21J hidden natural A-block search.

D21I showed hidden-only A can be exact, but the best candidate mostly amplified
one copy-like bit. D21J changes the objective: preserve exact roundtrip, but
strongly prefer non-copy effective geometry.

Important: the hidden model is still a tied linear proof:

    effective_encoder = code_from_hidden @ hidden_from_visible
    decode = effective_encoder.T

No direct visible->A16 highway is allowed in D21J candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21f_ablock_natural_geometry import parse_entries  # noqa: E402
from tools._scratch.d21i_ablock_hidden_sparse_sweep import (  # noqa: E402
    CODE_DIM,
    VISIBLE_DIM,
    HiddenAState,
    ascii_map,
    distance_rows,
    entries_to_matrix,
    evaluate_state,
    load_d21g_direct,
    mutate_state,
    topology_rows,
    write_csv,
)


def gate_pass(row: dict[str, object]) -> bool:
    return (
        float(row["exact_byte_acc"]) == 1.0
        and float(row["bit_acc"]) == 1.0
        and float(row["byte_argmax_acc"]) == 1.0
        and int(row["hidden_collisions"]) == 0
        and float(row["byte_margin_min"]) > 0.0
        and float(row["reciprocity_error"]) == 0.0
    )


def hidden_factor_state(name: str, effective: np.ndarray, hidden_dim: int) -> HiddenAState:
    if hidden_dim < VISIBLE_DIM:
        raise ValueError("hidden_dim must be >= visible dim for identity-bit factor")
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    hidden_from_visible = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.float64)
    code_from_hidden = np.zeros((CODE_DIM, hidden_dim), dtype=np.float64)
    for bit in range(VISIBLE_DIM):
        hidden_from_visible[bit, bit] = 1.0
        code_from_hidden[:, bit] = effective[:, bit]
    return HiddenAState(name, hidden_dim, direct, hidden_from_visible, code_from_hidden)


def d21a_effective() -> np.ndarray:
    matrix = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    for bit in range(VISIBLE_DIM):
        matrix[bit, bit] = 1.0
        matrix[VISIBLE_DIM + bit, bit] = 1.0
    return matrix


def score(row: dict[str, object]) -> float:
    if not gate_pass(row):
        return (
            -100.0
            + 30.0 * float(row["exact_byte_acc"])
            + 8.0 * float(row["bit_acc"])
            + 8.0 * float(row["byte_argmax_acc"])
        )
    margin = max(0.0, float(row["byte_margin_min"]))
    geometry = float(row["ascii_class_geometry"])
    copy = float(row["effective_copy_penalty"])
    duplicate = float(row["duplicate_lane_penalty"])
    edges = float(row["structural_edge_count"])
    robust = float(row.get("single_edge_drop_mean_bit", 0.0))
    return float(
        30.0
        + 1.75 * min(4.0, margin)
        + 14.0 * geometry
        + 1.0 * robust
        - 10.0 * copy
        - 1.5 * duplicate
        - 0.018 * edges
    )


def eval_with_score(state: HiddenAState, candidate_id: int, source: str, step: int, robust: bool) -> dict[str, object]:
    row = evaluate_state(state, candidate_id=candidate_id, source=source, step=step, include_robustness=robust)
    row["D21J_noncopy_score"] = score(row)
    return row


def initial_states(hidden_dims: Sequence[int], d21g_path: Path) -> list[HiddenAState]:
    states: list[HiddenAState] = []
    for hidden_dim in hidden_dims:
        if hidden_dim >= VISIBLE_DIM:
            states.append(hidden_factor_state(f"factor_d21a_h{hidden_dim}", d21a_effective(), hidden_dim))
    d21g = load_d21g_direct(d21g_path)
    if d21g is not None:
        for hidden_dim in hidden_dims:
            if hidden_dim >= VISIBLE_DIM:
                states.append(hidden_factor_state(f"factor_d21g_h{hidden_dim}", d21g, hidden_dim))
    return states


def parse_hidden_dims(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_search(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = random.Random(int(args.seed))
    hidden_dims = parse_hidden_dims(str(args.hidden_dims))
    rows: list[dict[str, object]] = []
    beams: list[tuple[HiddenAState, dict[str, object]]] = []
    candidate_id = 0

    for state in initial_states(hidden_dims, Path(args.d21g)):
        row = eval_with_score(state, candidate_id, "start", 0, True)
        candidate_id += 1
        rows.append(row)
        beams.append((state, row))

    for step in range(1, int(args.max_steps) + 1):
        next_beams: list[tuple[HiddenAState, dict[str, object]]] = []
        for state, current in beams:
            best_state = state
            best_row = current
            for _ in range(max(1, int(args.workers))):
                proposal = mutate_state(state, rng, lock_direct=False)
                # D21J forbids direct edges even if the generic mutator tried to add one.
                proposal.direct[:, :] = 0.0
                row = eval_with_score(proposal, candidate_id, "proposal", step, False)
                candidate_id += 1
                rows.append(row)
                if float(row["D21J_noncopy_score"]) > float(best_row["D21J_noncopy_score"]) + 1e-9 and (
                    gate_pass(row) or not gate_pass(best_row)
                ):
                    best_state = proposal
                    best_row = row
            if best_state is not state:
                robust = eval_with_score(best_state, candidate_id, "accepted", step, True)
                candidate_id += 1
                rows.append(robust)
                next_beams.append((best_state, robust))
            else:
                next_beams.append((state, current))
        beams = next_beams

    top = sorted(rows, key=lambda row: float(row["D21J_noncopy_score"]), reverse=True)[:64]
    robust_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in top:
        state = row_to_state(row)
        robust_by_key[(str(row["arm"]), str(row["effective_entries"]))] = eval_with_score(
            state,
            int(row["candidate_id"]),
            str(row["source"]),
            int(row["step"]),
            True,
        )
    final_rows = [robust_by_key.get((str(row["arm"]), str(row["effective_entries"])), row) for row in rows]
    return sorted(final_rows, key=lambda row: float(row["D21J_noncopy_score"]), reverse=True)


def row_to_state(row: dict[str, object]) -> HiddenAState:
    hidden_dim = int(row["hidden_dim"])
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    hidden_in = entries_to_matrix(parse_entries(str(row["hidden_in_entries"])), hidden_dim, VISIBLE_DIM)
    hidden_out = entries_to_matrix(parse_entries(str(row["hidden_out_entries"])), CODE_DIM, hidden_dim)
    return HiddenAState(str(row["arm"]), hidden_dim, direct, hidden_in, hidden_out)


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    d21a = next((row for row in rows if str(row["arm"]).startswith("factor_d21a")), None)
    candidates = [
        row
        for row in rows
        if gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and float(row["effective_copy_penalty"]) <= 0.45
    ]
    if not candidates:
        return "D21J_NO_NONCOPY_HIDDEN_A", "no exact no-direct hidden candidate reached the non-copy gate"
    best = max(
        candidates,
        key=lambda row: (
            float(row["byte_margin_min"]),
            float(row["ascii_class_geometry"]),
            -float(row["structural_edge_count"]),
        ),
    )
    baseline_geom = float(d21a["ascii_class_geometry"]) if d21a is not None else 0.0
    geom_gain = float(best["ascii_class_geometry"]) - baseline_geom
    if float(best["byte_margin_min"]) >= 2.0 and float(best["ascii_class_geometry"]) >= 0.74:
        return (
            "D21J_HIDDEN_NATURAL_A_PASS",
            f"{best['arm']} is exact/no-direct/non-copy with margin {float(best['byte_margin_min']):.3f} and geometry gain {geom_gain:+.4f}",
        )
    return (
        "D21J_HIDDEN_NATURAL_WEAK_PASS",
        f"{best['arm']} is exact/no-direct/non-copy, but margin/geometry remains below full gate",
    )


def write_report(out: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, config: dict[str, object]) -> None:
    lines = [
        "# D21J Hidden Natural A-Block Search",
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
        "## Interpretation",
        "",
        "D21J forbids direct visible->A16 edges and strongly penalizes copy-like effective maps.",
        "It asks whether hidden link locations can carry a non-copy A manifold, not just amplify one bit.",
        "",
        "## Top Rows",
        "",
        "```text",
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
            f"{float(row['D21J_noncopy_score']):6.2f}"
        )
    lines.extend(["```", "", "## Config", "", "```json", json.dumps(config, indent=2), "```", ""])
    (out / "D21J_ABLOCK_HIDDEN_NATURAL_SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
    write_csv(out / "hidden_natural_results.csv", rows)
    write_csv(out / "hidden_natural_topology_edges.csv", topology_rows(rows[:6]))
    write_csv(out / "hidden_natural_distance_matrix.csv", distance_rows(rows[:6]))
    payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "config": config,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if gate_pass(row)),
        "noncopy_gate_pass_count": sum(
            1
            for row in rows
            if gate_pass(row) and int(row["direct_edge_count"]) == 0 and float(row["effective_copy_penalty"]) <= 0.45
        ),
        "best_candidate": rows[0] if rows else None,
    }
    (out / "hidden_natural_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, rows, verdict_name, reason, config)
    print(ascii_map(rows))
    print(f"[D21J] verdict={verdict_name} reason={reason}")
    print(f"[D21J] wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21J hidden natural A-block search")
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--hidden-dims", default="8,16,32")
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--d21g", default="output/phase_d21g_ablock_margin_aware_polish_20260503/main/margin_top.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
