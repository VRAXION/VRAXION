#!/usr/bin/env python3
"""
D21G margin-aware natural A-block polish.

Starting point: D21F no_prefill geometry lead. The goal is to raise byte
margin without collapsing back to the ordered redundant-copy baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21a_reciprocal_byte_ablock import dedupe_entries, entries_to_string  # noqa: E402
from tools._scratch.d21f_ablock_natural_geometry import (  # noqa: E402
    evaluate_entries,
    parse_entries,
    topology_rows,
    distance_rows,
    ascii_map,
    write_csv,
)


VISIBLE_DIM = 8
CODE_DIM = 16
NO_PREFILL_D21F = dedupe_entries(
    [
        (4, 6, 1.0),
        (5, 2, 0.5),
        (6, 4, -1.0),
        (6, 6, 0.75),
        (8, 5, -0.25),
        (9, 3, -0.75),
        (10, 1, 0.25),
        (11, 7, 0.25),
        (15, 0, 0.5),
    ]
)
BASELINE_ENTRIES = dedupe_entries(
    [(idx, idx, 1.0) for idx in range(8)] + [(idx + 8, idx, 1.0) for idx in range(8)]
)
WEIGHTS = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)


def gate_bool(row: dict[str, object]) -> bool:
    return bool(row["gate_pass"])


def column_energy(entries: Sequence[tuple[int, int, float]]) -> list[float]:
    energy = [0.0 for _ in range(VISIBLE_DIM)]
    for _code_idx, visible_idx, value in entries:
        energy[visible_idx] += float(value) ** 2
    return energy


def ordered_copy_penalty(entries: Sequence[tuple[int, int, float]]) -> float:
    if not entries:
        return 1.0
    hits = 0
    for code_idx, visible_idx, value in entries:
        if code_idx % VISIBLE_DIM == visible_idx and abs(abs(float(value)) - 1.0) < 1e-9:
            hits += 1
    return float(hits / max(1, len(entries)))


def one_hot_ratio(entries: Sequence[tuple[int, int, float]]) -> float:
    by_row: dict[int, int] = {}
    for code_idx, _visible_idx, _value in entries:
        by_row[code_idx] = by_row.get(code_idx, 0) + 1
    if not by_row:
        return 0.0
    return float(sum(1 for count in by_row.values() if count == 1) / len(by_row))


def margin_objective(row: dict[str, object]) -> float:
    if not gate_bool(row):
        return -1000.0 + float(row["exact_byte_acc"]) * 10.0
    margin = float(row["byte_margin_min"])
    geometry = float(row["ascii_class_geometry"])
    copy = float(row["identity_copy_penalty"])
    duplicate = float(row["duplicate_lane_penalty"])
    distributed = float(row["distributed_code_score"])
    edges = float(row["edge_count"])
    one_hot = float(row.get("one_hot_ratio", 0.0))
    # Full-pass target is margin first, but retain natural geometry and avoid
    # returning to ordered copy layout.
    return (
        28.0 * min(margin, 4.0)
        + 7.0 * geometry
        + 2.5 * distributed
        - 9.0 * copy
        - 2.0 * duplicate
        - 0.35 * one_hot
        - 0.045 * edges
    )


def eval_entries(candidate_id: int, arm: str, entries: tuple[tuple[int, int, float], ...], step: int, source: str, robust: bool) -> dict[str, object]:
    row = evaluate_entries(
        arm=arm,
        candidate_id=candidate_id,
        visible_dim=VISIBLE_DIM,
        code_dim=CODE_DIM,
        entries=entries,
        source=source,
        step=step,
        include_robustness=robust,
    )
    row["ordered_copy_penalty"] = ordered_copy_penalty(entries)
    row["one_hot_ratio"] = one_hot_ratio(entries)
    row["D21G_margin_score"] = margin_objective(row)
    return row


def balanced_seed(entries: tuple[tuple[int, int, float], ...], *, target_energy: float, rng: random.Random) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    used = {(c, v) for c, v, _value in current}
    energy = column_energy(current)
    for visible_idx in sorted(range(VISIBLE_DIM), key=lambda idx: energy[idx]):
        while energy[visible_idx] < target_energy:
            candidates = [
                code_idx
                for code_idx in range(CODE_DIM)
                if (code_idx, visible_idx) not in used and code_idx % VISIBLE_DIM != visible_idx
            ]
            if not candidates:
                break
            code_idx = rng.choice(candidates)
            value = rng.choice((-1.0, 1.0))
            current.append((code_idx, visible_idx, value))
            used.add((code_idx, visible_idx))
            energy[visible_idx] += value * value
    return dedupe_entries(current)


def mutate(entries: tuple[tuple[int, int, float], ...], rng: random.Random) -> tuple[tuple[int, int, float], ...]:
    current = list(entries)
    used = {(c, v) for c, v, _value in current}
    energy = column_energy(current)
    op = rng.choice(("add_weak_bit", "add", "remove", "flip", "reweight", "sign"))

    if op in {"add_weak_bit", "add"}:
        if op == "add_weak_bit":
            visible_idx = min(range(VISIBLE_DIM), key=lambda idx: energy[idx])
        else:
            visible_idx = rng.randrange(VISIBLE_DIM)
        for _ in range(100):
            code_idx = rng.randrange(CODE_DIM)
            if (code_idx, visible_idx) not in used and code_idx % VISIBLE_DIM != visible_idx:
                current.append((code_idx, visible_idx, rng.choice(WEIGHTS)))
                return dedupe_entries(current)
    if op == "remove" and len(current) > 1:
        del current[rng.randrange(len(current))]
        return dedupe_entries(current)
    if op == "flip" and current:
        idx = rng.randrange(len(current))
        _old_code, _old_visible, value = current[idx]
        for _ in range(100):
            code_idx = rng.randrange(CODE_DIM)
            visible_idx = rng.randrange(VISIBLE_DIM)
            if (code_idx, visible_idx) not in used and code_idx % VISIBLE_DIM != visible_idx:
                current[idx] = (code_idx, visible_idx, value)
                return dedupe_entries(current)
    if op == "reweight" and current:
        idx = rng.randrange(len(current))
        code_idx, visible_idx, _value = current[idx]
        current[idx] = (code_idx, visible_idx, rng.choice(WEIGHTS))
        return dedupe_entries(current)
    if op == "sign" and current:
        idx = rng.randrange(len(current))
        code_idx, visible_idx, value = current[idx]
        current[idx] = (code_idx, visible_idx, -float(value))
        return dedupe_entries(current)
    return entries


def run_search(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = random.Random(int(args.seed))
    candidate_id = 0
    rows: list[dict[str, object]] = []

    start_specs = [
        ("baseline16", BASELINE_ENTRIES),
        ("d21f_no_prefill", NO_PREFILL_D21F),
        ("balanced_energy_1", balanced_seed(NO_PREFILL_D21F, target_energy=1.0, rng=random.Random(int(args.seed) + 1))),
        ("balanced_energy_2", balanced_seed(NO_PREFILL_D21F, target_energy=2.0, rng=random.Random(int(args.seed) + 2))),
    ]

    beams: list[tuple[str, tuple[tuple[int, int, float], ...], dict[str, object]]] = []
    for arm, entries in start_specs:
        row = eval_entries(candidate_id, arm, entries, 0, "start", True)
        candidate_id += 1
        rows.append(row)
        beams.append((arm, entries, row))

    for step in range(1, int(args.max_steps) + 1):
        next_beams = []
        for arm, entries, current in beams:
            best_row = current
            best_entries = entries
            for _worker in range(max(1, int(args.workers))):
                proposal_entries = mutate(entries, rng)
                row = eval_entries(candidate_id, arm, proposal_entries, step, "proposal", False)
                candidate_id += 1
                rows.append(row)
                if margin_objective(row) > margin_objective(best_row) + 1e-9:
                    best_row = row
                    best_entries = proposal_entries
            if best_row is not current:
                robust = eval_entries(candidate_id, arm, best_entries, step, "accepted", True)
                candidate_id += 1
                rows.append(robust)
                next_beams.append((arm, best_entries, robust))
            else:
                next_beams.append((arm, entries, current))
        beams = next_beams
    # Re-evaluate top rows with robustness populated. Include both score-top
    # rows and margin-top natural rows; otherwise the copy baseline can hide the
    # candidate we actually care about.
    score_top = sorted(rows, key=lambda row: float(row["D21G_margin_score"]), reverse=True)[:48]
    margin_top = sorted(
        [row for row in rows if str(row["arm"]) != "baseline16" and gate_bool(row)],
        key=lambda row: (float(row["byte_margin_min"]), float(row["ascii_class_geometry"])),
        reverse=True,
    )[:48]
    top = score_top + margin_top
    robust_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in top:
        entries = parse_entries(str(row["entries"]))
        robust_by_key[(str(row["arm"]), str(row["entries"]))] = eval_entries(
            int(row["candidate_id"]),
            str(row["arm"]),
            entries,
            int(row["step"]),
            "robust_top",
            True,
        )
    final_rows = [robust_by_key.get((str(row["arm"]), str(row["entries"])), row) for row in rows]
    return sorted(final_rows, key=lambda row: float(row["D21G_margin_score"]), reverse=True)


def verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    start = next(row for row in rows if row["arm"] == "d21f_no_prefill" and row["source"] == "start")
    candidates = [
        row
        for row in rows
        if gate_bool(row)
        and str(row["arm"]) not in {"baseline16"}
        and float(row["byte_margin_min"]) > float(start["byte_margin_min"]) + 1e-9
        and float(row["identity_copy_penalty"]) <= 0.45
        and float(row["ascii_class_geometry"]) >= 0.66
    ]
    if not candidates:
        return "D21G_NO_MARGIN_GAIN", "no exact non-baseline candidate improved margin under natural constraints"
    best = max(candidates, key=lambda row: (float(row["byte_margin_min"]), float(row["ascii_class_geometry"]), -float(row["identity_copy_penalty"])))
    margin = float(best["byte_margin_min"])
    geom = float(best["ascii_class_geometry"])
    if margin >= 2.0 and geom >= 0.70 and float(best["single_edge_drop_mean_bit"]) >= 0.99:
        return "D21G_MARGIN_NATURAL_PASS", f"{best['arm']} reached margin {margin:.3f} with geometry {geom:.3f}"
    if margin >= 1.0:
        return "D21G_MARGIN_NATURAL_WEAK_PASS", f"{best['arm']} improved margin to {margin:.3f}, but not full deploy gate"
    return "D21G_MARGIN_GAIN_THIN", f"{best['arm']} improved margin only to {margin:.3f}"


def write_report(out_dir: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, config: dict[str, object]) -> None:
    start = next(row for row in rows if row["arm"] == "d21f_no_prefill" and row["source"] == "start")
    best = best_natural_candidate(rows) or rows[0]
    lines = [
        "# D21G A-Block Margin-Aware Natural Polish",
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
        "## Why",
        "",
        "D21F no_prefill was natural-looking and exact, but its byte margin was only +0.125.",
        "D21G searched for a nearby sparse/non-copy wiring with higher margin.",
        "",
        "## Start vs Best",
        "",
        "```text",
        f"start no_prefill: margin={float(start['byte_margin_min']):.6f} geometry={float(start['ascii_class_geometry']):.6f} copy={float(start['identity_copy_penalty']):.6f}",
        f"best:             arm={best['arm']} margin={float(best['byte_margin_min']):.6f} geometry={float(best['ascii_class_geometry']):.6f} copy={float(best['identity_copy_penalty']):.6f}",
        "```",
        "",
        "## Top Rows",
        "",
        "```text",
        "arm                  exact margin  geom   copy  edges onehot score",
        "-------------------- ----- ------ ------ ----- ----- ------ ------",
    ]
    for row in rows[:12]:
        lines.append(
            f"{str(row['arm'])[:20]:20} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):6.3f} "
            f"{float(row['identity_copy_penalty']):5.3f} "
            f"{int(row['edge_count']):5d} "
            f"{float(row.get('one_hot_ratio', 0.0)):6.3f} "
            f"{float(row['D21G_margin_score']):6.2f}"
        )
    lines.extend(["```", "", "## Config", "", "```json", json.dumps(config, indent=2), "```", ""])
    (out_dir / "D21G_ABLOCK_MARGIN_AWARE_POLISH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, config: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "margin_candidates.csv", rows)
    write_csv(out_dir / "margin_topology_edges.csv", topology_rows(rows[:4]))
    write_csv(out_dir / "margin_distance_matrix.csv", distance_rows(rows[:4]))
    (out_dir / "margin_ascii_map.txt").write_text(ascii_map(rows), encoding="utf-8")
    payload = {
        "verdict": verdict_name,
        "verdict_reason": reason,
        "config": config,
        "candidate_count": len(rows),
        "gate_pass_count": sum(1 for row in rows if gate_bool(row)),
        "best_candidate": rows[0] if rows else None,
        "best_natural_candidate": best_natural_candidate(rows),
        "d21f_start": next((row for row in rows if row["arm"] == "d21f_no_prefill" and row["source"] == "start"), None),
    }
    (out_dir / "margin_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, rows, verdict_name, reason, config)


def best_natural_candidate(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if gate_bool(row)
        and str(row["arm"]) != "baseline16"
        and float(row["identity_copy_penalty"]) <= 0.45
        and float(row["ascii_class_geometry"]) >= 0.66
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["byte_margin_min"]),
            float(row.get("single_edge_drop_mean_bit", 0.0)),
            float(row["ascii_class_geometry"]),
            -float(row["identity_copy_penalty"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="D21G margin-aware A-block polish")
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = run_search(args)
    verdict_name, reason = verdict(rows)
    config = {
        "mode": args.mode,
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "start": "D21F no_prefill",
    }
    write_outputs(Path(args.out), rows, verdict_name, reason, config)
    print((Path(args.out) / "margin_ascii_map.txt").read_text(encoding="utf-8"))
    print(f"[D21G] verdict={verdict_name} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
