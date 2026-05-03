#!/usr/bin/env python3
"""Targeted A-v2 H12 native-int8 geometry polish.

This is a narrow follow-up to the A-v2 native int8 seed sweep. It starts from
the H12 near-miss artifact and only accepts native int8_q6 edits that preserve
the non-copy exact roundtrip while trying to push ASCII geometry over 0.760.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.a_v2_native_int8_seed_sweep import (  # noqa: E402
    CODE_DIM,
    CURRENT_ARTIFACT,
    Q_CHOICES,
    SCALE,
    VISIBLE_DIM,
    Int8AState,
    acceptance_gate,
    artifact_from_row as base_artifact_from_row,
    count_q,
    evaluate_int8_state,
    parse_q_entries,
    q_entries_string,
    row_to_int8_state,
    strong_gate,
)
from tools._scratch.d21i_ablock_hidden_sparse_sweep import gate_pass, write_csv  # noqa: E402


DEFAULT_START = Path("tools/a_v2_hidden_natural_int8_candidate.json")
DEFAULT_OUT = Path("output/phase_a_v2_h12_int8_geometry_polish_20260503")
TARGET_GEOMETRY = 0.760
TARGET_MARGIN = 4.0
EPS = 1e-12


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def load_artifact(path: Path, *, seed: int) -> Int8AState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hidden_dim = int(payload["hidden_dim"])
    hidden_in_q = np.zeros((hidden_dim, VISIBLE_DIM), dtype=np.int16)
    hidden_out_q = np.zeros((CODE_DIM, hidden_dim), dtype=np.int16)
    for row, col, q in payload["hidden_in_q"]:
        hidden_in_q[int(row), int(col)] = int(q)
    for row, col, q in payload["hidden_out_q"]:
        hidden_out_q[int(row), int(col)] = int(q)
    return Int8AState("A-v2-H12-polish", seed, hidden_dim, hidden_in_q, hidden_out_q)


def choose_existing(rng: random.Random, matrix: np.ndarray, *, prefer_small: bool = False) -> tuple[int, int] | None:
    coords = [(int(row), int(col)) for row, col in zip(*np.nonzero(matrix))]
    if not coords:
        return None
    if prefer_small:
        coords = sorted(coords, key=lambda rc: abs(int(matrix[rc[0], rc[1]])))
        return rng.choice(coords[: max(1, min(8, len(coords)))])
    return rng.choice(coords)


def clamp_q(q: int) -> int:
    if q == 0:
        return 0
    q = max(-64, min(64, int(q)))
    if abs(q) < 8:
        return 0
    # Keep values on native q6 / 8-step grid.
    q = int(round(q / 8.0) * 8)
    if q == 0:
        return 0
    return int(max(-64, min(64, q)))


def state_key(state: Int8AState) -> str:
    return q_entries_string(state.hidden_in_q) + "|" + q_entries_string(state.hidden_out_q)


def exact_safe(row: dict[str, object]) -> bool:
    return (
        gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and int(row["hidden_dim"]) == 12
        and float(row["effective_copy_penalty"]) <= 0.10
        and float(row["duplicate_lane_penalty"]) <= 0.05
        and float(row["byte_margin_min"]) >= TARGET_MARGIN - EPS
    )


def beam_safe(row: dict[str, object]) -> bool:
    """Allow tiny valleys during search, but never copy-like/direct solutions."""

    return (
        gate_pass(row)
        and int(row["direct_edge_count"]) == 0
        and int(row["hidden_dim"]) == 12
        and float(row["byte_margin_min"]) >= 3.75
        and float(row["ascii_class_geometry"]) >= 0.745
        and float(row["effective_copy_penalty"]) <= 0.12
        and float(row["duplicate_lane_penalty"]) <= 0.08
    )


def polish_rank(row: dict[str, object]) -> tuple[float, ...]:
    return (
        1.0 if strong_gate(row) else 0.0,
        1.0 if exact_safe(row) else 0.0,
        float(row["ascii_class_geometry"]),
        float(row["byte_margin_min"]),
        -float(row["effective_copy_penalty"]),
        -float(row["duplicate_lane_penalty"]),
        float(row.get("single_edge_drop_mean_bit", 0.0)),
        float(row["native_int8_score"]),
    )


def mutate_micro(state: Int8AState, rng: random.Random) -> tuple[Int8AState, str]:
    next_state = state.clone()
    ops = (
        "nudge_hidden_out",
        "nudge_hidden_out",
        "nudge_hidden_out",
        "nudge_hidden_in",
        "add_small_hidden_out",
        "add_small_hidden_out",
        "add_small_hidden_in",
        "remove_small_hidden_out",
        "remove_small_hidden_in",
        "rewire_hidden_out",
        "paired_out_nudge",
    )
    op = rng.choice(ops)

    if op == "nudge_hidden_out":
        coord = choose_existing(rng, next_state.hidden_out_q)
        if coord is not None:
            row, col = coord
            next_state.hidden_out_q[row, col] = clamp_q(int(next_state.hidden_out_q[row, col]) + rng.choice((-16, -8, 8, 16)))
        return next_state, op

    if op == "nudge_hidden_in":
        coord = choose_existing(rng, next_state.hidden_in_q)
        if coord is not None:
            row, col = coord
            next_state.hidden_in_q[row, col] = clamp_q(int(next_state.hidden_in_q[row, col]) + rng.choice((-16, -8, 8, 16)))
        return next_state, op

    if op == "add_small_hidden_out":
        for _ in range(48):
            row = rng.randrange(CODE_DIM)
            col = rng.randrange(next_state.hidden_dim)
            if next_state.hidden_out_q[row, col] == 0:
                next_state.hidden_out_q[row, col] = rng.choice((-24, -16, -8, 8, 16, 24))
                break
        return next_state, op

    if op == "add_small_hidden_in":
        for _ in range(48):
            row = rng.randrange(next_state.hidden_dim)
            col = rng.randrange(VISIBLE_DIM)
            if next_state.hidden_in_q[row, col] == 0:
                next_state.hidden_in_q[row, col] = rng.choice((-24, -16, -8, 8, 16, 24))
                break
        return next_state, op

    if op == "remove_small_hidden_out" and count_q(next_state.hidden_out_q) > 24:
        coord = choose_existing(rng, next_state.hidden_out_q, prefer_small=True)
        if coord is not None:
            row, col = coord
            next_state.hidden_out_q[row, col] = 0
        return next_state, op

    if op == "remove_small_hidden_in" and count_q(next_state.hidden_in_q) > VISIBLE_DIM:
        coord = choose_existing(rng, next_state.hidden_in_q, prefer_small=True)
        if coord is not None:
            row, col = coord
            next_state.hidden_in_q[row, col] = 0
        return next_state, op

    if op == "rewire_hidden_out":
        coord = choose_existing(rng, next_state.hidden_out_q, prefer_small=True)
        if coord is not None:
            old_row, old_col = coord
            q = int(next_state.hidden_out_q[old_row, old_col])
            next_state.hidden_out_q[old_row, old_col] = 0
            for _ in range(48):
                row = rng.randrange(CODE_DIM)
                col = rng.randrange(next_state.hidden_dim)
                if next_state.hidden_out_q[row, col] == 0:
                    next_state.hidden_out_q[row, col] = q
                    break
        return next_state, op

    if op == "paired_out_nudge":
        hidden_idx = rng.randrange(next_state.hidden_dim)
        rows = [int(row) for row in np.nonzero(next_state.hidden_out_q[:, hidden_idx])[0]]
        rng.shuffle(rows)
        for row in rows[:2]:
            next_state.hidden_out_q[row, hidden_idx] = clamp_q(
                int(next_state.hidden_out_q[row, hidden_idx]) + rng.choice((-8, 8))
            )
        return next_state, op

    return next_state, "noop"


def evaluate(
    state: Int8AState,
    *,
    candidate_id: int,
    source: str,
    step: int,
    mutation: str,
    include_robustness: bool,
) -> dict[str, object]:
    row = evaluate_int8_state(
        state,
        candidate_id=candidate_id,
        source=source,
        step=step,
        include_robustness=include_robustness,
        mutation=mutation,
    )
    row["target_geometry"] = TARGET_GEOMETRY
    row["target_margin"] = TARGET_MARGIN
    row["geometry_gap_to_target"] = TARGET_GEOMETRY - float(row["ascii_class_geometry"])
    row["margin_gap_to_target"] = TARGET_MARGIN - float(row["byte_margin_min"])
    row["exact_safe_gate"] = exact_safe(row)
    row["beam_safe_gate"] = beam_safe(row)
    return row


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


def write_progress(out: Path, payload: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry_polish_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    exact = [row for row in rows if exact_safe(row)]
    if not exact:
        tradeoff = [row for row in rows if gate_pass(row) and float(row["ascii_class_geometry"]) >= TARGET_GEOMETRY]
        if tradeoff:
            best = max(tradeoff, key=lambda row: float(row["ascii_class_geometry"]))
            return (
                "A_V2_H12_INT8_MARGIN_TRADEOFF",
                f"geometry reached {float(best['ascii_class_geometry']):.6f}, but margin fell to {float(best['byte_margin_min']):.3f}",
            )
        return "A_V2_H12_INT8_NO_EXACT_SAFE_CANDIDATE", "no exact-safe non-copy H12 candidate survived"

    passing = [row for row in exact if float(row["ascii_class_geometry"]) >= TARGET_GEOMETRY]
    if passing:
        best = max(passing, key=polish_rank)
        return (
            "A_V2_H12_INT8_GEOMETRY_POLISH_PASS",
            f"margin {float(best['byte_margin_min']):.3f}, geometry {float(best['ascii_class_geometry']):.6f}",
        )

    best = max(exact, key=polish_rank)
    start_geom = float(rows[0]["ascii_class_geometry"]) if rows else 0.0
    if float(best["ascii_class_geometry"]) > start_geom + 1e-6:
        return (
            "A_V2_H12_INT8_GEOMETRY_NEAR_MISS",
            f"best exact-safe geometry {float(best['ascii_class_geometry']):.6f}, target gap {TARGET_GEOMETRY - float(best['ascii_class_geometry']):.6f}",
        )
    return (
        "A_V2_H12_INT8_NO_GEOMETRY_GAIN",
        f"best exact-safe geometry stayed {float(best['ascii_class_geometry']):.6f}",
    )


def artifact_from_row(row: dict[str, object], *, name: str) -> dict[str, object]:
    payload = base_artifact_from_row(row, name=name)
    payload["source"] = "a_v2_h12_int8_geometry_polish"
    payload["source_verdict"] = ""
    return payload


def write_report(
    out: Path,
    rows: Sequence[dict[str, object]],
    verdict_name: str,
    reason: str,
    meta: dict[str, object],
) -> None:
    exact = [row for row in rows if exact_safe(row)]
    lines = [
        "# A-v2 H12 Native Int8 Geometry Polish",
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
        "## Target",
        "",
        "```text",
        f"margin   >= {TARGET_MARGIN:.3f}",
        f"geometry >= {TARGET_GEOMETRY:.3f}",
        "copy     <= 0.10",
        "direct edges == 0",
        "native int8_q6 only",
        "```",
        "",
        "## Top Exact-Safe Candidates",
        "",
        "```text",
        "rank margin geometry gap       copy dup  robust edges score",
        "---- ------ -------- --------- ---- ---- ------ ----- ------",
    ]
    for idx, row in enumerate(sorted(exact, key=polish_rank, reverse=True)[:16], start=1):
        lines.append(
            f"{idx:4d} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):8.6f} "
            f"{TARGET_GEOMETRY - float(row['ascii_class_geometry']):9.6f} "
            f"{float(row['effective_copy_penalty']):4.2f} "
            f"{float(row['duplicate_lane_penalty']):4.2f} "
            f"{float(row.get('single_edge_drop_mean_bit', 0.0)):6.3f} "
            f"{int(row['structural_edge_count']):5d} "
            f"{float(row['native_int8_score']):6.2f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## ASCII Mini Bar",
            "",
            "```text",
        ]
    )
    if rows:
        best = max(rows, key=polish_rank)
        start_geom = float(meta.get("start_geometry", rows[-1].get("ascii_class_geometry", 0.0)))
        best_geom = float(best["ascii_class_geometry"])
        scale = 50
        start_bar = "#" * int(max(0.0, min(1.0, start_geom / TARGET_GEOMETRY)) * scale)
        best_bar = "#" * int(max(0.0, min(1.0, best_geom / TARGET_GEOMETRY)) * scale)
        lines.append(f"start geometry {start_geom:.6f} [{start_bar:<50}]")
        lines.append(f"best  geometry {best_geom:.6f} [{best_bar:<50}]")
        lines.append(f"target        {TARGET_GEOMETRY:.6f} [{'#' * scale}]")
    lines.extend(["```", "", "## Meta", "", "```json", json.dumps(meta, indent=2), "```", ""])
    (out / "A_V2_H12_INT8_GEOMETRY_POLISH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out: Path,
    rows: Sequence[dict[str, object]],
    path_rows: Sequence[dict[str, object]],
    meta: dict[str, object],
) -> tuple[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    candidates_dir = out / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(unique_rows(rows), key=polish_rank, reverse=True)
    path_sorted = sorted(unique_rows(path_rows), key=lambda row: int(row["step"]))
    verdict_name, reason = verdict(rows_sorted)

    write_csv(out / "geometry_polish_candidates.csv", rows_sorted)
    write_csv(out / "geometry_polish_paths.csv", path_sorted)

    exported: list[dict[str, object]] = []
    for idx, row in enumerate(rows_sorted[: int(meta["top_k"])], start=1):
        payload = artifact_from_row(row, name=f"A-v2-H12-int8-geometry-polish-top-{idx:02d}")
        payload["source_verdict"] = verdict_name
        artifact_path = candidates_dir / f"top_{idx:02d}.json"
        artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exported.append({"rank": idx, "path": str(artifact_path), "verification": payload["verification"]})

    top_payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "meta": meta,
        "candidate_count": len(rows_sorted),
        "exact_safe_count": sum(1 for row in rows_sorted if exact_safe(row)),
        "target_pass_count": sum(1 for row in rows_sorted if exact_safe(row) and float(row["ascii_class_geometry"]) >= TARGET_GEOMETRY),
        "best_candidate": rows_sorted[0] if rows_sorted else None,
        "exported_candidates": exported,
    }
    (out / "geometry_polish_top.json").write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
    write_report(out, rows_sorted, verdict_name, reason, meta)
    return verdict_name, reason


def run_search(args: argparse.Namespace) -> int:
    t0 = time.time()
    out = Path(args.out)
    rng = random.Random(int(args.seed))
    start = load_artifact(Path(args.start), seed=int(args.seed))
    candidate_id = 0
    start_row = evaluate(
        start,
        candidate_id=candidate_id,
        source="start",
        step=0,
        mutation="start",
        include_robustness=True,
    )
    candidate_id += 1

    beams: list[tuple[Int8AState, dict[str, object]]] = [(start, start_row)]
    rows: list[dict[str, object]] = [start_row]
    path_rows: list[dict[str, object]] = [start_row]
    best_exact = start_row if exact_safe(start_row) else None
    best_logged_geometry = float(start_row["ascii_class_geometry"])
    deadline = t0 + float(args.time_budget_s) if float(args.time_budget_s) > 0 else None
    stop_requested = False

    for step in range(1, int(args.max_steps) + 1):
        if stop_requested:
            break
        if deadline is not None and time.time() >= deadline:
            print("[A-v2-H12-polish] time budget reached", flush=True)
            break
        proposals: list[tuple[Int8AState, dict[str, object]]] = []
        for _ in range(max(1, int(args.workers))):
            parent, _parent_row = rng.choice(beams)
            proposal, mutation = mutate_micro(parent, rng)
            key = state_key(proposal)
            if any(key == state_key(state) for state, _row in beams):
                continue
            row = evaluate(
                proposal,
                candidate_id=candidate_id,
                source="proposal",
                step=step,
                mutation=mutation,
                include_robustness=False,
            )
            candidate_id += 1
            rows.append(row)
            if beam_safe(row):
                proposals.append((proposal, row))

            improved_exact = exact_safe(row) and (
                best_exact is None or polish_rank(row) > polish_rank(best_exact)
            )
            target_hit = exact_safe(row) and float(row["ascii_class_geometry"]) >= TARGET_GEOMETRY
            if improved_exact or target_hit:
                robust = evaluate(
                    proposal,
                    candidate_id=candidate_id,
                    source="accepted",
                    step=step,
                    mutation=mutation,
                    include_robustness=True,
                )
                candidate_id += 1
                rows.append(robust)
                if exact_safe(robust):
                    best_exact = robust if best_exact is None or polish_rank(robust) > polish_rank(best_exact) else best_exact
                    path_rows.append(robust)
                    proposals.append((proposal, robust))
                    robust_geometry = float(robust["ascii_class_geometry"])
                    if robust_geometry > best_logged_geometry + 1e-9:
                        best_logged_geometry = robust_geometry
                        print(
                            f"[A-v2-H12-polish] target hit step={step} margin={float(robust['byte_margin_min']):.3f} "
                            f"geometry={robust_geometry:.6f}",
                            flush=True,
                        )
                        if bool(args.stop_on_pass) and robust_geometry >= TARGET_GEOMETRY:
                            beams.extend(proposals)
                            stop_requested = True
                            break

        combined = beams + proposals
        combined = sorted(combined, key=lambda item: polish_rank(item[1]), reverse=True)
        next_beams: list[tuple[Int8AState, dict[str, object]]] = []
        seen: set[str] = set()
        for state, row in combined:
            key = state_key(state)
            if key in seen:
                continue
            seen.add(key)
            next_beams.append((state, row))
            if len(next_beams) >= int(args.beam_width):
                break
        beams = next_beams or beams

        if step % 100 == 0 or step == 1:
            top = max((row for _state, row in beams), key=polish_rank)
            print(
                f"[A-v2-H12-polish] step={step} margin={float(top['byte_margin_min']):.3f} "
                f"geom={float(top['ascii_class_geometry']):.6f} gap={TARGET_GEOMETRY - float(top['ascii_class_geometry']):.6f} "
                f"beam={len(beams)} rows={len(rows)}",
                flush=True,
            )
            write_progress(
                out,
                {
                    "step": step,
                    "rows": len(rows),
                    "path_rows": len(path_rows),
                    "best_margin": float(top["byte_margin_min"]),
                    "best_geometry": float(top["ascii_class_geometry"]),
                    "geometry_gap": TARGET_GEOMETRY - float(top["ascii_class_geometry"]),
                    "runtime_s": time.time() - t0,
                },
            )

    # Re-evaluate final beam with robustness to make final rankings deterministic.
    for state, row in beams[: int(args.beam_width)]:
        rows.append(
            evaluate(
                state,
                candidate_id=candidate_id,
                source="final_beam",
                step=int(row["step"]),
                mutation=str(row["mutation"]),
                include_robustness=True,
            )
        )
        candidate_id += 1

    meta = {
        "mode": args.mode,
        "start": str(args.start),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "beam_width": int(args.beam_width),
        "workers": int(args.workers),
        "top_k": int(args.top_k),
        "time_budget_s": float(args.time_budget_s),
        "runtime_s": time.time() - t0,
        "target_margin": TARGET_MARGIN,
        "target_geometry": TARGET_GEOMETRY,
        "start_margin": float(start_row["byte_margin_min"]),
        "start_geometry": float(start_row["ascii_class_geometry"]),
    }
    verdict_name, reason = write_outputs(out, rows, path_rows, meta)
    print(f"[A-v2-H12-polish] verdict={verdict_name} reason={reason}")
    print(f"[A-v2-H12-polish] wrote {out}")
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    with Path(args.candidates).open("r", encoding="utf-8", newline="") as handle:
        rows_in = list(csv.DictReader(handle))
    rows_in = sorted(rows_in, key=lambda row: (
        str(row.get("exact_safe_gate", "")).lower() == "true",
        float(row.get("ascii_class_geometry", 0.0)),
        float(row.get("byte_margin_min", 0.0)),
    ), reverse=True)
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows_in[: int(args.top_k)]):
        state = row_to_int8_state(row)
        rows.append(
            evaluate(
                state,
                candidate_id=idx,
                source="confirm",
                step=int(float(row.get("step", 0) or 0)),
                mutation=str(row.get("mutation", "confirm")),
                include_robustness=True,
            )
        )
    meta = {
        "mode": "confirm",
        "candidates": args.candidates,
        "top_k": int(args.top_k),
        "target_margin": TARGET_MARGIN,
        "target_geometry": TARGET_GEOMETRY,
    }
    verdict_name, reason = write_outputs(Path(args.out), rows, rows, meta)
    print(f"[A-v2-H12-polish] verdict={verdict_name} reason={reason}")
    print(f"[A-v2-H12-polish] wrote {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "main", "confirm"), required=True)
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--time-budget-s", type=float, default=0.0)
    parser.add_argument("--stop-on-pass", action=argparse.BooleanOptionalAction, default=True)
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
