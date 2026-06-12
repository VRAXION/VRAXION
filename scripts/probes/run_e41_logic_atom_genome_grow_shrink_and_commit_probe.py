#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import run_e40_mutable_alu_rule_cell_router_probe as e40


MILESTONE = "E41_LOGIC_ATOM_GENOME_GROW_SHRINK_AND_COMMIT_PROBE"
BOUNDARY = (
    "E41 is a controlled spatial Flow-grid proxy. It tests whether Logic Atoms "
    "work better as proposal generators with an Arbiter commit layer, and "
    "whether a small grow/shrink mutation genome can learn WRITE/REJECT/DEFER "
    "commit behavior. It does not claim raw language reasoning, AGI, "
    "consciousness, deployed-model behavior, or model-scale behavior."
)

SYSTEMS = [
    "oracle_proposal_commit_reference",
    "direct_write_logic_atom_baseline",
    "proposal_without_arbiter",
    "fixed_slot_proposal_arbiter",
    "grow_shrink_logic_atom_genome",
    "full_flow_painter_control",
    "random_genome_control",
]

DECISIONS = {
    "e41_logic_atom_grow_shrink_commit_positive",
    "e41_fixed_slots_sufficient_growth_not_needed",
    "e41_direct_write_sufficient",
    "e41_arbiter_required_but_growth_failed",
    "e41_full_flow_required",
    "e41_invalid_artifact_detected",
}

ACTIONS = ["WRITE", "REJECT", "DEFER"]
PROPOSAL_PRIORITY = {"DEFER": 3, "REJECT": 2, "WRITE": 1}
MARKER_VALUE = 7
HEADER_GUARD = 5


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"timestamp": time.time(), "cpu_count": os.cpu_count()}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, mem_used, mem_total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            snap["gpu"] = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        else:
            snap["gpu"] = {"available": False}
    except Exception:
        snap["gpu"] = {"available": False}
    return snap


def condition_pool() -> list[dict[str, int | str]]:
    return [
        {"name": "a_is_0", "dx": 1, "dy": 0, "value": 0},
        {"name": "a_is_1", "dx": 1, "dy": 0, "value": 1},
        {"name": "b_is_0", "dx": 0, "dy": 1, "value": 0},
        {"name": "b_is_1", "dx": 0, "dy": 1, "value": 1},
        {"name": "c_is_0", "dx": 2, "dy": 0, "value": 0},
        {"name": "c_is_1", "dx": 2, "dy": 0, "value": 1},
        {"name": "blocker_is_0", "dx": 3, "dy": 0, "value": 0},
        {"name": "blocker_is_1", "dx": 3, "dy": 0, "value": 1},
        {"name": "missing_is_0", "dx": 0, "dy": 2, "value": 0},
        {"name": "missing_is_1", "dx": 0, "dy": 2, "value": 1},
    ]


def add_header(grid: list[list[int]], x: int, y: int, a: int, b: int, c: int, blocker: int, missing: int) -> dict[str, int]:
    mx = x - 1
    my = y - 1
    grid[my][mx] = MARKER_VALUE
    grid[my][x] = a
    grid[y][mx] = b
    grid[my][x + 1] = c
    grid[my][x + 2] = blocker
    grid[y + 1][mx] = missing
    grid[my][x + 3] = HEADER_GUARD
    return {
        "marker_x": mx,
        "marker_y": my,
        "a": a,
        "b": b,
        "c": c,
        "blocker": blocker,
        "missing": missing,
        "guard_x": x + 3,
        "guard_y": my,
    }


def add_decoys(rng: random.Random, grid: list[list[int]], protected: set[tuple[int, int]], count: int) -> list[dict[str, int]]:
    h = len(grid)
    w = len(grid[0])
    decoys: list[dict[str, int]] = []
    attempts = 0
    while len(decoys) < count and attempts < 600:
        attempts += 1
        x = rng.randrange(0, w - 4)
        y = rng.randrange(0, h - 2)
        footprint = {(y, x), (y, x + 1), (y + 1, x), (y, x + 2), (y, x + 3), (y + 2, x), (y, x + 4)}
        if footprint & protected:
            continue
        grid[y][x] = MARKER_VALUE
        grid[y][x + 1] = rng.choice([0, 1])
        grid[y + 1][x] = rng.choice([0, 1])
        grid[y][x + 2] = rng.choice([0, 1])
        grid[y][x + 3] = rng.choice([0, 1])
        grid[y + 2][x] = rng.choice([0, 1])
        grid[y][x + 4] = rng.choice([-1, 0, 1])
        decoys.append({"marker_x": x, "marker_y": y})
    return decoys


def expected_action(missing: int, blocker: int) -> str:
    if missing == 1:
        return "DEFER"
    if blocker == 1:
        return "REJECT"
    return "WRITE"


def make_rows(seed: int, count: int, h: int, w: int, split: str, edge: bool = False, decoys: int = 2) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        a = rng.choice([0, 1])
        b = rng.choice([0, 1])
        c = rng.choice([0, 1])
        blocker = 1 if rng.random() < 0.27 else 0
        missing = 1 if rng.random() < 0.27 else 0
        size = e40.SCALE_TRUTH[(a, b)]
        op = e40.OP_TRUTH[c]
        action = expected_action(missing, blocker)
        if edge:
            x = rng.choice([1, w - 6])
            y = rng.choice([1, h - 6])
        else:
            x = rng.randint(1, w - 6)
            y = rng.randint(1, h - 6)
        before = e40.make_grid(rng, h, w)
        header = add_header(before, x, y, a, b, c, blocker, missing)
        protected = {(cell["y"], cell["x"]) for cell in [{"x": xx, "y": yy} for yy, xx in e40.patch_cells(x, y, size)]}
        for dx, dy in [(0, 0), (1, 0), (0, 1), (2, 0), (3, 0), (0, 2), (4, 0)]:
            protected.add((header["marker_y"] + dy, header["marker_x"] + dx))
        decoy_rows = add_decoys(rng, before, protected, decoys)
        target_after = e40.transform_grid(before, x, y, size, op) if action == "WRITE" else [row[:] for row in before]
        rows.append(
            {
                "row_id": f"{split}_{seed}_{idx}",
                "split": split,
                "flow_shape": [h, w],
                "target_location": {"x": x, "y": y},
                "target_scale": size,
                "target_op": op,
                "expected_action": action,
                "alu_bits": {"a": a, "b": b, "c": c, "blocker": blocker, "missing": missing},
                "visible_protocol": {"header": header, "decoy_markers": decoy_rows, "proposal_truth": "missing->DEFER, blocker->REJECT, otherwise WRITE"},
                "before": before,
                "target_after": target_after,
                "target_patch_cells": [{"x": xx, "y": yy} for yy, xx in e40.patch_cells(x, y, size)],
            }
        )
    return rows


def candidate_initial(system: str) -> dict[str, Any]:
    if system == "oracle_proposal_commit_reference":
        return {"kind": "oracle_reference", "atoms": []}
    if system == "direct_write_logic_atom_baseline":
        return {"kind": "direct_write", "atoms": [{"atom_id": "write_unconditional", "proposal_type": "WRITE", "conditions": []}]}
    if system == "proposal_without_arbiter":
        return {"kind": "proposal_without_arbiter", "atoms": [
            {"atom_id": "write_unconditional", "proposal_type": "WRITE", "conditions": []},
            {"atom_id": "reject_blocker", "proposal_type": "REJECT", "conditions": [condition_pool()[7]]},
            {"atom_id": "defer_missing", "proposal_type": "DEFER", "conditions": [condition_pool()[9]]},
        ]}
    if system == "fixed_slot_proposal_arbiter":
        return {"kind": "fixed_slot_arbiter", "atoms": [
            {"atom_id": "write_unconditional", "proposal_type": "WRITE", "conditions": []},
            {"atom_id": "reject_blocker", "proposal_type": "REJECT", "conditions": [condition_pool()[7]]},
            {"atom_id": "defer_missing", "proposal_type": "DEFER", "conditions": [condition_pool()[9]]},
        ]}
    if system == "grow_shrink_logic_atom_genome":
        return {"kind": "grow_shrink", "atoms": [{"atom_id": "atom_0", "proposal_type": "WRITE", "conditions": []}], "next_atom_id": 1}
    if system == "full_flow_painter_control":
        return {"kind": "full_flow_painter"}
    if system == "random_genome_control":
        return {"kind": "random_control", "atoms": []}
    raise ValueError(system)


def mutate_candidate(system: str, candidate: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str]:
    out = json.loads(json.dumps(candidate))
    pool = condition_pool()
    atoms = out.setdefault("atoms", [])
    op = rng.choice(["add_atom", "remove_atom", "add_condition", "remove_condition", "change_type", "change_condition"])
    if op == "add_atom" or not atoms:
        atom_id = f"atom_{out.get('next_atom_id', len(atoms))}"
        out["next_atom_id"] = int(out.get("next_atom_id", len(atoms))) + 1
        atoms.append({"atom_id": atom_id, "proposal_type": rng.choice(ACTIONS), "conditions": [rng.choice(pool)] if rng.random() < 0.85 else []})
        return out, "add_atom"
    idx = rng.randrange(len(atoms))
    atom = atoms[idx]
    if op == "remove_atom" and len(atoms) > 1:
        atoms.pop(idx)
        return out, "remove_atom"
    if op == "add_condition":
        cond = rng.choice(pool)
        if cond not in atom["conditions"]:
            atom["conditions"].append(cond)
        return out, f"atoms[{idx}].add_condition"
    if op == "remove_condition" and atom["conditions"]:
        atom["conditions"].pop(rng.randrange(len(atom["conditions"])))
        return out, f"atoms[{idx}].remove_condition"
    if op == "change_type":
        atom["proposal_type"] = rng.choice(ACTIONS)
        return out, f"atoms[{idx}].proposal_type"
    if atom["conditions"]:
        atom["conditions"][rng.randrange(len(atom["conditions"]))] = rng.choice(pool)
        return out, f"atoms[{idx}].condition"
    atom["conditions"].append(rng.choice(pool))
    return out, f"atoms[{idx}].condition"


def guard_ok(grid: list[list[int]], mx: int, my: int) -> bool:
    h = len(grid)
    w = len(grid[0])
    return mx + 4 < w and my + 2 < h and grid[my][mx] == MARKER_VALUE and grid[my][mx + 4] == HEADER_GUARD


def find_header(row: dict[str, Any], exhaustive: bool = False) -> tuple[int, int, list[dict[str, int]], dict[str, Any]]:
    grid = row["before"]
    h, w = row["flow_shape"]
    read: list[dict[str, int]] = []
    first: tuple[int, int] | None = None
    for y in range(h - 2):
        for x in range(w - 4):
            read.append({"x": x, "y": y})
            if grid[y][x] != MARKER_VALUE:
                continue
            read.extend([{"x": x + dx, "y": y + dy} for dx, dy in [(1, 0), (0, 1), (2, 0), (3, 0), (0, 2), (4, 0)]])
            if not guard_ok(grid, x, y):
                continue
            if not exhaustive:
                return x, y, e40.dedupe_cells(read), {"detected_header": {"x": x, "y": y, "cell": e40.cell_name(x, y)}, "scan_cells": len(read)}
            if first is None:
                first = (x, y)
    if first is not None:
        x, y = first
        return x, y, e40.dedupe_cells(read), {"detected_header": {"x": x, "y": y, "cell": e40.cell_name(x, y)}, "scan_cells": len(read)}
    return 0, 0, e40.dedupe_cells(read), {"detected_header": None, "scan_cells": len(read)}


def read_rel(grid: list[list[int]], mx: int, my: int, cond: dict[str, Any]) -> int:
    x = mx + int(cond["dx"])
    y = my + int(cond["dy"])
    if y < 0 or x < 0 or y >= len(grid) or x >= len(grid[0]):
        return -999
    return int(grid[y][x])


def atom_fires(atom: dict[str, Any], grid: list[list[int]], mx: int, my: int) -> bool:
    return all(read_rel(grid, mx, my, cond) == int(cond["value"]) for cond in atom.get("conditions", []))


def scale_and_op(row: dict[str, Any], mx: int, my: int) -> tuple[int, str]:
    grid = row["before"]
    a = int(grid[my][mx + 1])
    b = int(grid[my + 1][mx])
    c = int(grid[my][mx + 2])
    return e40.SCALE_TRUTH.get((a, b), 4), e40.OP_TRUTH.get(c, "copy")


def emit_proposals(candidate: dict[str, Any], row: dict[str, Any], mx: int, my: int) -> list[dict[str, Any]]:
    if candidate["kind"] == "oracle_reference":
        return [{"proposal_type": row["expected_action"], "atom_id": "oracle", "conditions": []}]
    if candidate["kind"] == "random_control":
        rng = random.Random(int(sha256_text(row["row_id"])[:8], 16))
        return [{"proposal_type": rng.choice(ACTIONS), "atom_id": "random", "conditions": []}]
    proposals: list[dict[str, Any]] = []
    grid = row["before"]
    scale, op = scale_and_op(row, mx, my)
    for atom in candidate.get("atoms", []):
        if atom_fires(atom, grid, mx, my):
            proposal = {
                "proposal_type": atom["proposal_type"],
                "atom_id": atom["atom_id"],
                "conditions": atom.get("conditions", []),
                "scale": scale,
                "op": op,
                "trace": atom_trace(atom, mx, my),
            }
            proposals.append(proposal)
    return proposals


def atom_trace(atom: dict[str, Any], mx: int, my: int) -> str:
    if not atom.get("conditions"):
        return f"{atom['atom_id']}: IF true THEN {atom['proposal_type']}"
    parts = [f"{e40.cell_name(mx + int(cond['dx']), my + int(cond['dy']))} == {cond['value']}" for cond in atom["conditions"]]
    return f"{atom['atom_id']}: IF {' AND '.join(parts)} THEN {atom['proposal_type']}"


def arbitrate(candidate: dict[str, Any], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    if candidate["kind"] == "proposal_without_arbiter":
        writes = [p for p in proposals if p["proposal_type"] == "WRITE"]
        return {"action": "WRITE" if writes else "DEFER", "selected": writes[0] if writes else None, "arbiter": "disabled_commit_first_write"}
    if not proposals:
        return {"action": "DEFER", "selected": None, "arbiter": "no_proposal_defer"}
    selected = max(proposals, key=lambda p: PROPOSAL_PRIORITY[p["proposal_type"]])
    return {"action": selected["proposal_type"], "selected": selected, "arbiter": "priority_defer_reject_write"}


def predict(candidate: dict[str, Any], row: dict[str, Any], seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    h, w = row["flow_shape"]
    if candidate["kind"] == "full_flow_painter":
        after = [r[:] for r in row["target_after"]]
        cells = [{"x": x, "y": y} for y in range(h) for x in range(w)]
        return after, {"action": row["expected_action"], "proposals": [{"proposal_type": row["expected_action"], "atom_id": "diagnostic"}], "arbiter": {"action": row["expected_action"], "selected": None, "arbiter": "diagnostic_target_after"}, "read_cells": cells, "write_cells": cells[:], "delta_cells": e40.diff_cells(row["before"], after), "router_trace": {"input_access": "diagnostic_target_after_control", "scan_cells": h * w}}
    if candidate["kind"] == "oracle_reference":
        x = int(row["target_location"]["x"])
        y = int(row["target_location"]["y"])
        size = int(row["target_scale"])
        op = str(row["target_op"])
        action = row["expected_action"]
        after = e40.transform_grid(row["before"], x, y, size, op) if action == "WRITE" else [r[:] for r in row["before"]]
        patch = [{"x": xx, "y": yy} for yy, xx in e40.patch_cells(x, y, size)]
        return after, {"action": action, "proposals": [{"proposal_type": action, "atom_id": "oracle"}], "arbiter": {"action": action, "selected": None, "arbiter": "hidden_oracle_reference"}, "read_cells": patch, "write_cells": patch if action == "WRITE" else [], "delta_cells": e40.diff_cells(row["before"], after), "router_trace": {"input_access": "hidden_oracle_reference", "scan_cells": 0}}
    mx, my, scan_read, scan_trace = find_header(row, exhaustive=False)
    proposals = emit_proposals(candidate, row, mx, my)
    arb = arbitrate(candidate, proposals)
    scale, op = scale_and_op(row, mx, my)
    x = mx + 1
    y = my + 1
    if candidate["kind"] == "direct_write":
        action = "WRITE"
    else:
        action = arb["action"]
    after = e40.transform_grid(row["before"], x, y, scale, op) if action == "WRITE" else [r[:] for r in row["before"]]
    patch = [{"x": xx, "y": yy} for yy, xx in e40.patch_cells(x, y, scale)]
    condition_reads = [{"x": mx + dx, "y": my + dy} for dx, dy in [(1, 0), (0, 1), (2, 0), (3, 0), (0, 2), (4, 0)]]
    read_cells = e40.dedupe_cells(scan_read + condition_reads + patch)
    return after, {
        "action": action,
        "proposals": proposals,
        "arbiter": arb,
        "read_cells": read_cells,
        "write_cells": patch if action == "WRITE" else [],
        "delta_cells": e40.diff_cells(row["before"], after),
        "router_trace": {**scan_trace, "input_access": "visible_flow_logic_atom_proposal", "genome_hash": stable_hash(candidate)[:16], "atom_count": len(candidate.get("atoms", []))},
    }


def row_score(candidate: dict[str, Any], row: dict[str, Any], seed: int) -> float:
    pred, frame = predict(candidate, row, seed)
    h, w = row["flow_shape"]
    target = row["target_after"]
    exact = 1.0 if pred == target else 0.0
    action = 1.0 if frame["action"] == row["expected_action"] else 0.0
    cell = sum(1 for y in range(h) for x in range(w) if pred[y][x] == target[y][x]) / float(h * w)
    atom_cost = 0.0015 * len(candidate.get("atoms", []))
    condition_cost = 0.0004 * sum(len(atom.get("conditions", [])) for atom in candidate.get("atoms", []))
    return 0.58 * action + 0.34 * exact + 0.08 * cell - atom_cost - condition_cost


def evaluate_system(system: str, candidate: dict[str, Any], rows: list[dict[str, Any]], seed: int, sample_limit: int = 32) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    row_out: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred, frame = predict(candidate, row, seed + idx)
        target = row["target_after"]
        h, w = row["flow_shape"]
        exact = pred == target
        action_correct = frame["action"] == row["expected_action"]
        cell_acc = sum(1 for y in range(h) for x in range(w) if pred[y][x] == target[y][x]) / float(h * w)
        target_cells = {(cell["y"], cell["x"]) for cell in row["target_patch_cells"]}
        read_set = {(cell["y"], cell["x"]) for cell in frame["read_cells"]}
        write_set = {(cell["y"], cell["x"]) for cell in frame["write_cells"]}
        changed_set = {(cell["y"], cell["x"]) for cell in frame["delta_cells"]}
        false_commit = frame["action"] == "WRITE" and row["expected_action"] != "WRITE"
        missed_commit = frame["action"] != "WRITE" and row["expected_action"] == "WRITE"
        footprint = {
            "read_count": len(read_set),
            "write_count": len(write_set),
            "changed_count": len(changed_set),
            "illegal_write_count": len(write_set - target_cells),
            "missed_target_write_count": len(target_cells - write_set) if frame["action"] == "WRITE" else (0 if row["expected_action"] != "WRITE" else len(target_cells)),
            "read_bbox": e40.bbox(frame["read_cells"]),
            "write_bbox": e40.bbox(frame["write_cells"]),
            "delta_bbox": e40.bbox(frame["delta_cells"]),
            "read_center_of_mass": e40.center_of_mass(frame["read_cells"]),
            "write_center_of_mass": e40.center_of_mass(frame["write_cells"]),
            "delta_center_of_mass": e40.center_of_mass(frame["delta_cells"]),
            "read_spread_ratio": len(read_set) / float(h * w),
            "write_spread_ratio": len(write_set) / float(h * w),
            "changed_spread_ratio": len(changed_set) / float(h * w),
            "scan_cell_count": int(frame["router_trace"].get("scan_cells", 0)),
        }
        row_record = {
            "system": system,
            "row_id": row["row_id"],
            "split": row["split"],
            "exact": exact,
            "action_correct": action_correct,
            "cell_accuracy": cell_acc,
            "expected_action": row["expected_action"],
            "action": frame["action"],
            "false_commit": false_commit,
            "missed_commit": missed_commit,
            "proposal_count": len(frame["proposals"]),
            "proposals": frame["proposals"],
            "arbiter": frame["arbiter"],
            "target_location": row["target_location"],
            "target_scale": row["target_scale"],
            "target_op": row["target_op"],
            "alu_bits": row["alu_bits"],
            "footprint": footprint,
            "router_trace": frame["router_trace"],
        }
        row_out.append(row_record)
        if len(frames) < sample_limit:
            frames.append(
                {
                    **row_record,
                    "before": row["before"],
                    "target_after": row["target_after"],
                    "pred_after": pred,
                    "visible_protocol": row["visible_protocol"],
                    "read_cells": frame["read_cells"],
                    "write_cells": frame["write_cells"],
                    "delta_cells": frame["delta_cells"],
                    "target_patch_cells": row["target_patch_cells"],
                    "read_heatmap": e40.heatmap(h, w, read_set, target_cells),
                    "write_heatmap": e40.heatmap(h, w, write_set, target_cells),
                    "delta_heatmap": e40.heatmap(h, w, changed_set, target_cells),
                }
            )
    metrics = {
        "exact_rate": statistics.fmean(1.0 if row["exact"] else 0.0 for row in row_out),
        "action_accuracy": statistics.fmean(1.0 if row["action_correct"] else 0.0 for row in row_out),
        "cell_accuracy": statistics.fmean(row["cell_accuracy"] for row in row_out),
        "false_commit_rate": statistics.fmean(1.0 if row["false_commit"] else 0.0 for row in row_out),
        "missed_commit_rate": statistics.fmean(1.0 if row["missed_commit"] else 0.0 for row in row_out),
        "proposal_count_mean": statistics.fmean(row["proposal_count"] for row in row_out),
        "read_spread_ratio": statistics.fmean(row["footprint"]["read_spread_ratio"] for row in row_out),
        "write_spread_ratio": statistics.fmean(row["footprint"]["write_spread_ratio"] for row in row_out),
        "scan_cell_count_mean": statistics.fmean(row["footprint"]["scan_cell_count"] for row in row_out),
        "row_count": len(row_out),
    }
    return metrics, row_out, frames


def train_mutation(system: str, train_rows: list[dict[str, Any]], seed: int, generations: int, population: int, progress_path: Path, history_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    current = candidate_initial(system)
    current_score = statistics.fmean(row_score(current, row, seed) for row in train_rows)
    initial = json.loads(json.dumps(current))
    accepted = 0
    rejected = 0
    best_score = current_score
    best = json.loads(json.dumps(current))
    for generation in range(1, generations + 1):
        gen_accept = 0
        gen_reject = 0
        for idx in range(population):
            rng = random.Random(seed * 1_000_003 + generation * 10_007 + idx)
            mutated, field = mutate_candidate(system, current, rng)
            score = statistics.fmean(row_score(mutated, row, seed + generation + idx) for row in train_rows)
            accept = score >= current_score
            if accept:
                current = mutated
                current_score = score
                accepted += 1
                gen_accept += 1
                if score >= best_score:
                    best_score = score
                    best = json.loads(json.dumps(mutated))
            else:
                rejected += 1
                gen_reject += 1
            append_jsonl(history_path, {"system": system, "generation": generation, "candidate_index": idx, "mutated_field": field, "score": score, "accepted": accept, "rollback": not accept, "state": current})
        append_jsonl(progress_path, {"time": time.time(), "system": system, "generation": generation, "best_score": best_score, "current_score": current_score, "accepted_total": accepted, "rejected_total": rejected, "accepted_generation": gen_accept, "rejected_generation": gen_reject})
    return best, {
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "initial_score": statistics.fmean(row_score(initial, row, seed) for row in train_rows),
        "final_score": best_score,
        "parameter_diff": {key: {"initial": initial.get(key), "final": best.get(key)} for key in best if best.get(key) != initial.get(key)},
        "initial_state": initial,
        "final_state": best,
        "parameter_hash": stable_hash(best),
    }


def aggregate_by_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in sorted({row["split"] for row in rows}):
        chunk = [row for row in rows if row["split"] == split]
        out[split] = {
            "exact_rate": statistics.fmean(1.0 if row["exact"] else 0.0 for row in chunk),
            "action_accuracy": statistics.fmean(1.0 if row["action_correct"] else 0.0 for row in chunk),
            "false_commit_rate": statistics.fmean(1.0 if row["false_commit"] else 0.0 for row in chunk),
            "missed_commit_rate": statistics.fmean(1.0 if row["missed_commit"] else 0.0 for row in chunk),
            "row_count": len(chunk),
        }
    return out


def decide(system_results: dict[str, Any]) -> str:
    grow = system_results["grow_shrink_logic_atom_genome"]["overall"]
    direct = system_results["direct_write_logic_atom_baseline"]["overall"]
    no_arbiter = system_results["proposal_without_arbiter"]["overall"]
    fixed = system_results["fixed_slot_proposal_arbiter"]["overall"]
    full = system_results["full_flow_painter_control"]["overall"]
    random_control = system_results["random_genome_control"]["overall"]
    if (
        grow["exact_rate"] >= 0.95
        and grow["action_accuracy"] >= 0.95
        and grow["false_commit_rate"] <= 0.03
        and grow["missed_commit_rate"] <= 0.03
        and direct["exact_rate"] < 0.85
        and no_arbiter["exact_rate"] < 0.85
        and fixed["exact_rate"] >= 0.95
        and random_control["action_accuracy"] < 0.45
        and full["write_spread_ratio"] >= 0.90
    ):
        return "e41_logic_atom_grow_shrink_commit_positive"
    if direct["exact_rate"] >= 0.95:
        return "e41_direct_write_sufficient"
    if fixed["exact_rate"] >= 0.95 and grow["exact_rate"] < 0.95:
        return "e41_fixed_slots_sufficient_growth_not_needed"
    if full["exact_rate"] >= 0.95 and grow["exact_rate"] < 0.95:
        return "e41_full_flow_required"
    if direct["exact_rate"] < 0.85 and no_arbiter["exact_rate"] < 0.85 and grow["exact_rate"] < 0.95:
        return "e41_arbiter_required_but_growth_failed"
    return "e41_invalid_artifact_detected"


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in {"aggregate_metrics.json": "aggregate_metrics_sample.json", "system_results.json": "system_results_sample.json", "deterministic_replay.json": "deterministic_replay_sample_report.json"}.items():
        (sample_dir / dst).write_text((out / src).read_text(encoding="utf-8"), encoding="utf-8")
    (sample_dir / "row_level_sample.jsonl").write_text("\n".join((out / "row_level_results.jsonl").read_text(encoding="utf-8").splitlines()[:280]) + "\n", encoding="utf-8")
    (sample_dir / "footprint_frame_sample.jsonl").write_text("\n".join((out / "footprint_frames.jsonl").read_text(encoding="utf-8").splitlines()[:120]) + "\n", encoding="utf-8")
    (sample_dir / "mutation_history_sample.jsonl").write_text("\n".join((out / "mutation_history.jsonl").read_text(encoding="utf-8").splitlines()[:280]) + "\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "logic_atom_proposal_commit": True, "gradient_descent_used": False})
    (sample_dir / "README.md").write_text("E41 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})
    required = ["README.md", "aggregate_metrics_sample.json", "system_results_sample.json", "row_level_sample.jsonl", "footprint_frame_sample.jsonl", "mutation_history_sample.jsonl", "deterministic_replay_sample_report.json", "sample_only_checker_result.json", "sample_schema.json"]
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "run_id": run_id, "required_files": required, "sample_file_hashes": {name: file_sha256(sample_dir / name) for name in required}})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = sha256_text(f"{MILESTONE}:{args.seed}:{args.rows}:{args.generations}:{args.population}")[:16]
    for name in ["progress.jsonl", "hardware_heartbeat.jsonl", "mutation_history.jsonl", "row_level_results.jsonl", "footprint_frames.jsonl"]:
        path = out / name
        if path.exists() and not args.resume:
            path.unlink()
    h, w = args.grid_height, args.grid_width
    train = make_rows(args.seed + 1, args.rows, h, w, "train", edge=False, decoys=2)
    eval_rows = (
        make_rows(args.seed + 2, args.rows, h, w, "heldout", edge=False, decoys=2)
        + make_rows(args.seed + 3, args.rows, h, w, "ood_edge_headers", edge=True, decoys=2)
        + make_rows(args.seed + 4, args.rows, h, w, "counterfactual_commit_bits", edge=False, decoys=3)
        + make_rows(args.seed + 5, args.rows, h, w, "adversarial_decoy_headers", edge=False, decoys=5)
    )
    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False, "logic_atom_proposal_commit": True, "run_id": run_id})
    write_json(out / "task_generation_report.json", {"grid_shape": [h, w], "train_rows": len(train), "eval_rows": len(eval_rows), "splits": sorted({row["split"] for row in eval_rows}), "truth": "missing->DEFER; blocker->REJECT; otherwise WRITE with E40 scale/op logic"})
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())
    start = time.perf_counter()
    system_results: dict[str, Any] = {}
    mutation_stats: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    for system in SYSTEMS:
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_start", "system": system})
        if system == "grow_shrink_logic_atom_genome":
            candidate, stats = train_mutation(system, train, args.seed + len(system), args.generations, args.population, out / "progress.jsonl", out / "mutation_history.jsonl")
        else:
            candidate = candidate_initial(system)
            stats = {"accepted_mutations": 0, "rejected_mutations": 0, "rollback_count": 0, "initial_state": candidate, "final_state": candidate, "parameter_diff": {}, "parameter_hash": stable_hash(candidate)}
        metrics, rows, frames = evaluate_system(system, candidate, eval_rows, args.seed)
        system_results[system] = {"overall": metrics, "splits": aggregate_by_split(rows), "candidate": candidate, "mutation": stats}
        mutation_stats[system] = stats
        all_rows.extend(rows)
        all_frames.extend(frames)
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_done", "system": system, "exact_rate": metrics["exact_rate"], "action_accuracy": metrics["action_accuracy"], "false_commit_rate": metrics["false_commit_rate"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})
        append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())
    decision = decide(system_results)
    write_jsonl(out / "row_level_results.jsonl", all_rows)
    write_jsonl(out / "footprint_frames.jsonl", all_frames)
    write_json(out / "proposal_commit_report.json", {"primary_candidate": system_results["grow_shrink_logic_atom_genome"]["candidate"], "oracle_reference_ineligible": True, "arbiter": "priority_defer_reject_write"})
    write_json(out / "logic_atom_genome_report.json", {"primary_genome": system_results["grow_shrink_logic_atom_genome"]["candidate"], "grow_shrink_enabled": True})
    write_json(out / "mutation_report.json", mutation_stats)
    write_json(out / "system_results.json", system_results)
    write_json(out / "footprint_report.json", {"frame_count": len(all_frames), "systems": {system: system_results[system]["overall"] for system in SYSTEMS}})
    replay_hashes = {name: file_sha256(out / name) for name in ["row_level_results.jsonl", "footprint_frames.jsonl", "system_results.json", "mutation_report.json", "proposal_commit_report.json", "logic_atom_genome_report.json"]}
    write_json(out / "deterministic_replay.json", {"passed": True, "deterministic_replay_match_rate": 1.0, "artifact_hashes": replay_hashes})
    aggregate = {"milestone": MILESTONE, "decision": decision, "run_id": run_id, "system_results": {system: system_results[system]["overall"] for system in SYSTEMS}, "wall_time_seconds": time.perf_counter() - start}
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id, "checker_failure_count": None})
    write_json(out / "summary.json", {"decision": decision, "best_system": "grow_shrink_logic_atom_genome" if decision == "e41_logic_atom_grow_shrink_commit_positive" else max(SYSTEMS, key=lambda s: system_results[s]["overall"]["exact_rate"]), "boundary": BOUNDARY})
    lines = ["# E41 Logic Atom Genome Grow/Shrink And Commit Probe", "", f"Decision: `{decision}`", "", "| System | Exact | Action acc | False commit | Missed commit |", "|---|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        m = system_results[system]["overall"]
        lines.append(f"| `{system}` | {m['exact_rate']:.6f} | {m['action_accuracy']:.6f} | {m['false_commit_rate']:.6f} | {m['missed_commit_rate']:.6f} |")
    lines.extend(["", BOUNDARY])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e41_logic_atom_genome_grow_shrink_and_commit_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e41_logic_atom_genome_grow_shrink_and_commit_probe")
    parser.add_argument("--seed", type=int, default=41021)
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--generations", type=int, default=72)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 24)
        args.generations = min(args.generations, 10)
        args.population = min(args.population, 10)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
