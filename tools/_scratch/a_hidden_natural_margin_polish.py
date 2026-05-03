#!/usr/bin/env python3
"""A-HiddenNaturalMarginPolish.

Local, discrete margin polish around A-HiddenNatural16. The goal is to increase
the byte decode margin while preserving the non-copy hidden-only A geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21a_reciprocal_byte_ablock import ReciprocalABlock, all_visible_patterns  # noqa: E402
from tools._scratch.d21f_ablock_natural_geometry import parse_entries  # noqa: E402
from tools._scratch.d21i_ablock_hidden_sparse_sweep import (  # noqa: E402
    CODE_DIM,
    VISIBLE_DIM,
    HiddenAState,
    distance_rows,
    entries_to_matrix,
    evaluate_state,
    gate_pass,
    topology_rows,
    write_csv,
)

START_DEFAULT = Path("output/phase_d21j_ablock_hidden_natural_search_20260503/main/hidden_natural_top.json")
WEIGHTS = (-1.0, -0.75, -0.5, -0.375, -0.25, 0.25, 0.375, 0.5, 0.75, 1.0)
EPS = 1e-9


def count_nonzero(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(matrix) > 1e-12))


def load_start_state(path: Path) -> HiddenAState:
    blob = json.loads(path.read_text(encoding="utf-8"))
    candidate = blob.get("best_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"missing best_candidate in {path}")
    hidden_dim = int(candidate["hidden_dim"])
    if hidden_dim != 8:
        raise ValueError(f"expected hidden_dim=8 start candidate, got {hidden_dim}")
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    hidden_in = entries_to_matrix(parse_entries(str(candidate["hidden_in_entries"])), hidden_dim, VISIBLE_DIM)
    hidden_out = entries_to_matrix(parse_entries(str(candidate["hidden_out_entries"])), CODE_DIM, hidden_dim)
    return HiddenAState("A-HiddenNaturalMarginPolish", hidden_dim, direct, hidden_in, hidden_out)


def row_to_state(row: dict[str, object]) -> HiddenAState:
    hidden_dim = int(row["hidden_dim"])
    direct = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    hidden_in = entries_to_matrix(parse_entries(str(row["hidden_in_entries"])), hidden_dim, VISIBLE_DIM)
    hidden_out = entries_to_matrix(parse_entries(str(row["hidden_out_entries"])), CODE_DIM, hidden_dim)
    return HiddenAState(str(row.get("arm", "A-HiddenNaturalMarginPolish")), hidden_dim, direct, hidden_in, hidden_out)


def effective_rank_for_state(state: HiddenAState) -> float:
    block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, state.effective_encoder())
    codes = block.encode_patterns(all_visible_patterns(VISIBLE_DIM))
    centered = codes - codes.mean(axis=0, keepdims=True)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    if float(singular.sum()) <= 1e-12:
        return 0.0
    probs = singular / singular.sum()
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))


def polish_score(row: dict[str, object]) -> float:
    if not gate_pass(row):
        return -1000.0 + 100.0 * float(row["exact_byte_acc"]) + 10.0 * float(row["bit_acc"])
    return float(
        40.0 * min(float(row["byte_margin_min"]), 4.0)
        + 10.0 * float(row["ascii_class_geometry"])
        + 2.0 * float(row["random_far_margin"])
        + 1.0 * float(row["effective_rank"])
        - 12.0 * float(row["effective_copy_penalty"])
        - 2.0 * float(row["duplicate_lane_penalty"])
        - 0.04 * float(row["structural_edge_count"])
    )


def enrich_row(row: dict[str, object], state: HiddenAState, *, mutation: str, accepted: bool) -> dict[str, object]:
    row["effective_rank"] = effective_rank_for_state(state)
    row["polish_score"] = polish_score(row)
    row["mutation"] = mutation
    row["accepted_for_margin"] = accepted
    return row


def eval_polish_state(
    state: HiddenAState,
    *,
    candidate_id: int,
    source: str,
    step: int,
    mutation: str,
    robust: bool,
    accepted: bool = False,
) -> dict[str, object]:
    # Direct highway is forbidden for this phase even if a future helper mutates it.
    state.direct[:, :] = 0.0
    row = evaluate_state(state, candidate_id=candidate_id, source=source, step=step, include_robustness=robust)
    return enrich_row(row, state, mutation=mutation, accepted=accepted)


def structural_gate(row: dict[str, object]) -> bool:
    return (
        gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and int(row["hidden_dim"]) == 8
        and float(row["ascii_class_geometry"]) >= 0.760
        and float(row["effective_copy_penalty"]) <= 0.10
        and float(row["duplicate_lane_penalty"]) <= 0.05
    )


def exploration_gate(row: dict[str, object]) -> bool:
    return (
        gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and int(row["hidden_dim"]) == 8
        and float(row["ascii_class_geometry"]) >= 0.750
        and float(row["effective_copy_penalty"]) <= 0.18
        and float(row["duplicate_lane_penalty"]) <= 0.08
        and float(row["byte_margin_min"]) >= 2.0
    )


def choose_existing_coord(rng: random.Random, matrix: np.ndarray) -> tuple[int, int] | None:
    coords = list(zip(*np.nonzero(np.abs(matrix) > 1e-12)))
    if not coords:
        return None
    row, col = rng.choice(coords)
    return int(row), int(col)


def mutate_polish(state: HiddenAState, rng: random.Random) -> tuple[HiddenAState, str]:
    next_state = state.clone()
    next_state.direct[:, :] = 0.0
    op = rng.choice(
        (
            "reweight_hidden_out",
            "reweight_hidden_out",
            "reweight_hidden_in",
            "add_hidden_out_edge",
            "remove_bad_edge",
            "sign_flip",
        )
    )

    if op == "reweight_hidden_out":
        coord = choose_existing_coord(rng, next_state.code_from_hidden)
        if coord is not None:
            row, col = coord
            next_state.code_from_hidden[row, col] = rng.choice(WEIGHTS)
        return next_state, op

    if op == "reweight_hidden_in":
        coord = choose_existing_coord(rng, next_state.hidden_from_visible)
        if coord is not None:
            row, col = coord
            next_state.hidden_from_visible[row, col] = rng.choice(WEIGHTS)
        return next_state, op

    if op == "add_hidden_out_edge":
        used = set(zip(*np.nonzero(np.abs(next_state.code_from_hidden) > 1e-12)))
        for _ in range(64):
            code_idx = rng.randrange(CODE_DIM)
            hidden_idx = rng.randrange(next_state.hidden_dim)
            if (code_idx, hidden_idx) not in used:
                next_state.code_from_hidden[code_idx, hidden_idx] = rng.choice(WEIGHTS)
                break
        return next_state, op

    if op == "remove_bad_edge" and count_nonzero(next_state.code_from_hidden) > VISIBLE_DIM:
        coord = choose_existing_coord(rng, next_state.code_from_hidden)
        if coord is not None:
            row, col = coord
            next_state.code_from_hidden[row, col] = 0.0
        return next_state, op

    if op == "sign_flip":
        matrix = rng.choice([next_state.hidden_from_visible, next_state.code_from_hidden])
        coord = choose_existing_coord(rng, matrix)
        if coord is not None:
            row, col = coord
            matrix[row, col] = -float(matrix[row, col])
        return next_state, op

    return next_state, "noop"


def unique_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key = str(row["hidden_in_entries"]) + "|" + str(row["hidden_out_entries"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def run_search(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = random.Random(int(args.seed))
    start_state = load_start_state(Path(args.start))
    candidate_id = 0
    start_row = eval_polish_state(
        start_state,
        candidate_id=candidate_id,
        source="start",
        step=0,
        mutation="start",
        robust=True,
        accepted=False,
    )
    candidate_id += 1

    beams: list[tuple[HiddenAState, dict[str, object]]] = [(start_state, start_row)]
    path_rows: list[dict[str, object]] = [start_row]
    candidate_rows: list[dict[str, object]] = [start_row]
    global_best_margin = float(start_row["byte_margin_min"])
    evaluated = 1
    accepted_count = 0
    t0 = time.time()

    for step in range(1, int(args.max_steps) + 1):
        proposal_pool: list[tuple[HiddenAState, dict[str, object]]] = []
        reached_early_stop = False
        for state, _row in beams:
            for _worker in range(max(1, int(args.workers))):
                proposal, mutation = mutate_polish(state, rng)
                row = eval_polish_state(
                    proposal,
                    candidate_id=candidate_id,
                    source="proposal",
                    step=step,
                    mutation=mutation,
                    robust=False,
                    accepted=False,
                )
                candidate_id += 1
                evaluated += 1
                if structural_gate(row) and float(row["byte_margin_min"]) > global_best_margin + EPS:
                    robust = eval_polish_state(
                        proposal,
                        candidate_id=candidate_id,
                        source="accepted",
                        step=step,
                        mutation=mutation,
                        robust=True,
                        accepted=True,
                    )
                    candidate_id += 1
                    evaluated += 1
                    if structural_gate(robust) and float(robust["byte_margin_min"]) > global_best_margin + EPS:
                        global_best_margin = float(robust["byte_margin_min"])
                        accepted_count += 1
                        path_rows.append(robust)
                        candidate_rows.append(robust)
                        proposal_pool.append((proposal, robust))
                        if float(robust["byte_margin_min"]) >= float(args.early_stop_margin):
                            reached_early_stop = True
                elif exploration_gate(row):
                    proposal_pool.append((proposal, row))

        combined = beams + proposal_pool
        combined = sorted(combined, key=lambda item: float(item[1]["polish_score"]), reverse=True)
        beams = []
        seen: set[str] = set()
        for state, row in combined:
            key = str(row["hidden_in_entries"]) + "|" + str(row["hidden_out_entries"])
            if key in seen:
                continue
            seen.add(key)
            beams.append((state, row))
            if len(beams) >= int(args.beam_width):
                break

        if step % 100 == 0 or accepted_count:
            best = max([row for _state, row in beams] + candidate_rows, key=lambda row: float(row["byte_margin_min"]))
            print(
                f"[A-polish] step={step} eval={evaluated} accepted={accepted_count} "
                f"best_margin={float(best['byte_margin_min']):.3f} best_geom={float(best['ascii_class_geometry']):.3f} "
                f"elapsed={time.time() - t0:.1f}s"
            )
        if reached_early_stop:
            print(f"[A-polish] early stop: margin >= {float(args.early_stop_margin):.3f} at step {step}")
            break

    final_beam_rows: list[dict[str, object]] = []
    for state, row in beams[: max(1, int(args.beam_width))]:
        robust = eval_polish_state(
            state,
            candidate_id=int(row["candidate_id"]),
            source="final_beam",
            step=int(row["step"]),
            mutation=str(row["mutation"]),
            robust=True,
            accepted=bool(row.get("accepted_for_margin", False)),
        )
        final_beam_rows.append(robust)
    candidate_rows = unique_rows(candidate_rows + final_beam_rows)
    candidate_rows = sorted(candidate_rows, key=lambda row: (float(row["byte_margin_min"]), float(row["polish_score"])), reverse=True)
    meta = {
        "evaluated_count": evaluated,
        "accepted_count": accepted_count,
        "start_margin": float(start_row["byte_margin_min"]),
        "start_geometry": float(start_row["ascii_class_geometry"]),
        "runtime_s": time.time() - t0,
    }
    return path_rows, candidate_rows, final_beam_rows, meta


def confirm_candidates(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    path = Path(args.candidates)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        start_state = load_start_state(Path(args.start))
        start = eval_polish_state(start_state, candidate_id=0, source="start_fallback", step=0, mutation="start", robust=True)
        return [start], {"input_count": 0, "confirmed_count": 1}

    sorted_rows = sorted(rows, key=lambda row: (float(row.get("byte_margin_min", 0.0)), float(row.get("polish_score", 0.0))), reverse=True)
    confirmed: list[dict[str, object]] = []
    for idx, row in enumerate(sorted_rows[: int(args.top_k)]):
        state = row_to_state(row)
        confirmed.append(
            eval_polish_state(
                state,
                candidate_id=idx,
                source="confirm",
                step=int(float(row.get("step", 0) or 0)),
                mutation=str(row.get("mutation", "confirm")),
                robust=True,
                accepted=bool(str(row.get("accepted_for_margin", "")).lower() == "true"),
            )
        )
    confirmed = sorted(unique_rows(confirmed), key=lambda row: (float(row["byte_margin_min"]), float(row["polish_score"])), reverse=True)
    return confirmed, {"input_count": len(rows), "confirmed_count": len(confirmed)}


def verdict(rows: Sequence[dict[str, object]], start_margin: float) -> tuple[str, str]:
    improving = [
        row
        for row in rows
        if structural_gate(row) and float(row["byte_margin_min"]) > start_margin + EPS
    ]
    if not improving:
        copy_like = [
            row
            for row in rows
            if gate_pass(row)
            and float(row["byte_margin_min"]) > start_margin + EPS
            and float(row["effective_copy_penalty"]) > 0.10
        ]
        if copy_like:
            return "A_HIDDEN_NATURAL_COPY_REGRESSION", "margin improved only by becoming copy-like"
        tradeoff = [
            row
            for row in rows
            if gate_pass(row)
            and float(row["byte_margin_min"]) > start_margin + EPS
            and float(row["ascii_class_geometry"]) < 0.760
        ]
        if tradeoff:
            return "A_HIDDEN_NATURAL_GEOMETRY_TRADEOFF", "margin improved only with geometry dropping below 0.760"
        return "A_HIDDEN_NATURAL_NO_MARGIN_GAIN", "no candidate improved margin while preserving exact non-copy geometry"

    best = max(improving, key=lambda row: (float(row["byte_margin_min"]), float(row["ascii_class_geometry"])))
    if float(best["byte_margin_min"]) >= 3.5:
        return (
            "A_HIDDEN_NATURAL_MARGIN_STRONG_PASS",
            f"{best['arm']} reached margin {float(best['byte_margin_min']):.3f} with geometry {float(best['ascii_class_geometry']):.3f}",
        )
    return (
        "A_HIDDEN_NATURAL_MARGIN_WEAK_PASS",
        f"{best['arm']} improved margin to {float(best['byte_margin_min']):.3f} with geometry {float(best['ascii_class_geometry']):.3f}",
    )


def write_report(out: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, meta: dict[str, object]) -> None:
    lines = [
        "# A-HiddenNaturalMarginPolish",
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
        "arm                    exact margin  geom copy dup direct h_in h_out eff score",
        "---------------------- ----- ------ ------ ---- --- ------ ---- ----- --- ------",
    ]
    for row in rows[:18]:
        lines.append(
            f"{str(row['arm'])[:22]:22} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):6.3f} "
            f"{float(row['effective_copy_penalty']):4.2f} "
            f"{float(row['duplicate_lane_penalty']):3.2f} "
            f"{int(row['direct_edge_count']):6d} "
            f"{int(row['hidden_in_edge_count']):4d} "
            f"{int(row['hidden_out_edge_count']):5d} "
            f"{int(row['effective_edge_count']):3d} "
            f"{float(row['polish_score']):6.2f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Meta",
            "",
            "```json",
            json.dumps(meta, indent=2),
            "```",
            "",
        ]
    )
    (out / "A_HIDDEN_NATURAL_MARGIN_POLISH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_top_payload(out: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, meta: dict[str, object]) -> None:
    payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "meta": meta,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if gate_pass(row)),
        "structural_gate_pass_count": sum(1 for row in rows if structural_gate(row)),
        "best_candidate": rows[0] if rows else None,
    }
    (out / "margin_polish_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "hidden_natural_margin_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_outputs(out: Path, path_rows: Sequence[dict[str, object]], candidate_rows: Sequence[dict[str, object]], verdict_name: str, reason: str, meta: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sorted_candidates = sorted(candidate_rows, key=lambda row: (float(row["byte_margin_min"]), float(row["polish_score"])), reverse=True)
    write_csv(out / "margin_polish_paths.csv", path_rows)
    write_csv(out / "margin_polish_candidates.csv", sorted_candidates)
    write_csv(out / "margin_polish_topology_edges.csv", topology_rows(sorted_candidates[:6]))
    write_csv(out / "margin_polish_distance_matrix.csv", distance_rows(sorted_candidates[:6]))
    write_top_payload(out, sorted_candidates, verdict_name, reason, meta)
    write_report(out, sorted_candidates, verdict_name, reason, meta)


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.mode in {"smoke", "main"}:
        path_rows, candidate_rows, _beam_rows, meta = run_search(args)
        verdict_name, reason = verdict(candidate_rows, float(meta["start_margin"]))
        meta.update(
            {
                "mode": args.mode,
                "start": str(args.start),
                "max_steps": int(args.max_steps),
                "workers": int(args.workers),
                "beam_width": int(args.beam_width),
                "seed": int(args.seed),
            }
        )
        write_outputs(out, path_rows, candidate_rows, verdict_name, reason, meta)
        print(f"[A-polish] verdict={verdict_name} reason={reason}")
        print(f"[A-polish] wrote {out}")
        return 0

    confirmed, meta = confirm_candidates(args)
    start_margin = 2.5
    start_path = Path(args.start)
    if start_path.exists():
        start_state = load_start_state(start_path)
        start_row = eval_polish_state(start_state, candidate_id=-1, source="start_reference", step=0, mutation="start", robust=True)
        start_margin = float(start_row["byte_margin_min"])
    verdict_name, reason = verdict(confirmed, start_margin)
    meta.update({"mode": args.mode, "start": str(args.start), "top_k": int(args.top_k), "start_margin": start_margin})
    write_outputs(out, confirmed, confirmed, verdict_name, reason, meta)
    print(f"[A-polish] verdict={verdict_name} reason={reason}")
    print(f"[A-polish] wrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "main", "confirm"), required=True)
    parser.add_argument("--start", default=str(START_DEFAULT))
    parser.add_argument("--candidates", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--early-stop-margin", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mode == "confirm" and not args.candidates:
        raise SystemExit("--candidates is required for confirm mode")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
