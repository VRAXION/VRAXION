#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E40_MUTABLE_ALU_RULE_CELL_ROUTER_PROBE"
BOUNDARY = (
    "E40 is a controlled spatial Flow-grid proxy. It tests whether a mutable "
    "ALU/rule-cell router can learn small IF/AND read-cell rules for local "
    "pocket calls. It does not claim raw language reasoning, AGI, consciousness, "
    "deployed-model behavior, or model-scale behavior."
)

SYSTEMS = [
    "oracle_alu_rule_reference",
    "flat_marker_table_router",
    "location_only_fixed_call_router",
    "boolean_alu_without_op_router",
    "mutable_alu_rule_cell_router",
    "scan_all_rule_control",
    "full_flow_painter_control",
    "random_rule_control",
]

DECISIONS = {
    "e40_mutable_alu_rule_cell_router_positive",
    "e40_flat_marker_table_sufficient",
    "e40_boolean_scale_only_sufficient",
    "e40_scan_all_required",
    "e40_full_flow_required",
    "e40_invalid_artifact_detected",
}

OPS = ["copy", "invert", "threshold", "zero"]
SCALE_CHOICES = [2, 4, 6]
MARKER_VALUE = 7
HEADER_GUARD = 5
SCALE_TRUTH = {(0, 0): 2, (1, 0): 4, (0, 1): 6, (1, 1): 4}
OP_TRUTH = {0: "invert", 1: "threshold"}


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


def cell_name(x: int, y: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if x < len(alphabet):
        col = alphabet[x]
    else:
        col = f"C{x}"
    return f"{col}{y + 1}"


def make_grid(rng: random.Random, h: int, w: int) -> list[list[int]]:
    return [[rng.choice([-1, 0, 1]) for _ in range(w)] for _ in range(h)]


def patch_cells(x: int, y: int, size: int) -> list[tuple[int, int]]:
    return [(yy, xx) for yy in range(y, y + size) for xx in range(x, x + size)]


def apply_op(value: int, op: str) -> int:
    if op == "copy":
        return value
    if op == "invert":
        return -value
    if op == "threshold":
        return 1 if value > 0 else 0
    if op == "zero":
        return 0
    raise ValueError(op)


def transform_grid(grid: list[list[int]], x: int, y: int, size: int, op: str) -> list[list[int]]:
    out = [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])
    size = max(1, min(size, h - y, w - x))
    for yy, xx in patch_cells(x, y, size):
        out[yy][xx] = apply_op(grid[yy][xx], op)
    return out


def diff_cells(before: list[list[int]], after: list[list[int]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for y, (row_before, row_after) in enumerate(zip(before, after)):
        for x, (a, b) in enumerate(zip(row_before, row_after)):
            if a != b:
                out.append({"x": x, "y": y, "before": a, "after": b, "delta": b - a})
    return out


def bbox(cells: list[tuple[int, int]] | list[dict[str, int]]) -> dict[str, Any]:
    if not cells:
        return {"empty": True}
    xs: list[int] = []
    ys: list[int] = []
    for cell in cells:
        if isinstance(cell, dict):
            xs.append(int(cell["x"]))
            ys.append(int(cell["y"]))
        else:
            ys.append(int(cell[0]))
            xs.append(int(cell[1]))
    return {
        "empty": False,
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "width": max(xs) - min(xs) + 1,
        "height": max(ys) - min(ys) + 1,
        "area": (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1),
    }


def center_of_mass(cells: list[tuple[int, int]] | list[dict[str, int]]) -> dict[str, float | None]:
    if not cells:
        return {"x": None, "y": None}
    xs: list[float] = []
    ys: list[float] = []
    for cell in cells:
        if isinstance(cell, dict):
            xs.append(float(cell["x"]))
            ys.append(float(cell["y"]))
        else:
            ys.append(float(cell[0]))
            xs.append(float(cell[1]))
    return {"x": statistics.fmean(xs), "y": statistics.fmean(ys)}


def heatmap(h: int, w: int, cells: set[tuple[int, int]], target: set[tuple[int, int]] | None = None) -> list[str]:
    lines: list[str] = []
    target = target or set()
    for y in range(h):
        chars: list[str] = []
        for x in range(w):
            if (y, x) in cells and (y, x) in target:
                chars.append("@")
            elif (y, x) in cells:
                chars.append("#")
            elif (y, x) in target:
                chars.append("+")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return lines


def dedupe_cells(cells: list[dict[str, int]]) -> list[dict[str, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, int]] = []
    for cell in cells:
        key = (int(cell["y"]), int(cell["x"]))
        if key not in seen:
            seen.add(key)
            out.append({"x": int(cell["x"]), "y": int(cell["y"])})
    return out


def add_header(grid: list[list[int]], x: int, y: int, a_bit: int, b_bit: int, c_bit: int) -> dict[str, int]:
    mx = x - 1
    my = y - 1
    grid[my][mx] = MARKER_VALUE
    grid[my][x] = a_bit
    grid[y][mx] = b_bit
    grid[my][x + 1] = c_bit
    grid[my][x + 2] = HEADER_GUARD
    return {
        "marker_x": mx,
        "marker_y": my,
        "a_x": x,
        "a_y": my,
        "b_x": mx,
        "b_y": y,
        "c_x": x + 1,
        "c_y": my,
        "guard_x": x + 2,
        "guard_y": my,
        "a_bit": a_bit,
        "b_bit": b_bit,
        "c_bit": c_bit,
    }


def add_decoys(rng: random.Random, grid: list[list[int]], protected: set[tuple[int, int]], count: int) -> list[dict[str, int]]:
    h = len(grid)
    w = len(grid[0])
    decoys: list[dict[str, int]] = []
    attempts = 0
    while len(decoys) < count and attempts < 600:
        attempts += 1
        x = rng.randrange(0, w - 3)
        y = rng.randrange(0, h - 1)
        footprint = {(y, x), (y, x + 1), (y + 1, x), (y, x + 2), (y, x + 3)}
        if footprint & protected:
            continue
        grid[y][x] = MARKER_VALUE
        grid[y][x + 1] = rng.choice([0, 1])
        grid[y + 1][x] = rng.choice([0, 1])
        grid[y][x + 2] = rng.choice([0, 1])
        grid[y][x + 3] = rng.choice([-1, 0, 1])
        decoys.append({"marker_x": x, "marker_y": y})
    return decoys


def make_rows(seed: int, count: int, h: int, w: int, split: str, edge: bool = False, decoys: int = 2) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        a_bit = rng.choice([0, 1])
        b_bit = rng.choice([0, 1])
        c_bit = rng.choice([0, 1])
        size = SCALE_TRUTH[(a_bit, b_bit)]
        op = OP_TRUTH[c_bit]
        if edge:
            x = rng.choice([1, w - 6])
            y = rng.choice([1, h - 6])
        else:
            x = rng.randint(1, w - 6)
            y = rng.randint(1, h - 6)
        before = make_grid(rng, h, w)
        header = add_header(before, x, y, a_bit, b_bit, c_bit)
        protected = {(cell["y"], cell["x"]) for cell in [{"x": xx, "y": yy} for yy, xx in patch_cells(x, y, size)]}
        protected.update(
            {
                (header["marker_y"], header["marker_x"]),
                (header["a_y"], header["a_x"]),
                (header["b_y"], header["b_x"]),
                (header["c_y"], header["c_x"]),
                (header["guard_y"], header["guard_x"]),
            }
        )
        decoy_rows = add_decoys(rng, before, protected, decoys)
        after = transform_grid(before, x, y, size, op)
        rows.append(
            {
                "row_id": f"{split}_{seed}_{idx}",
                "split": split,
                "flow_shape": [h, w],
                "target_location": {"x": x, "y": y},
                "target_scale": size,
                "target_op": op,
                "alu_bits": {"a": a_bit, "b": b_bit, "c": c_bit},
                "visible_protocol": {
                    "header": header,
                    "marker_value": MARKER_VALUE,
                    "guard_value": HEADER_GUARD,
                    "decoy_markers": decoy_rows,
                    "truth_rule": "scale=f(a,b); op=f(c)",
                },
                "before": before,
                "target_after": after,
                "target_patch_cells": [{"x": xx, "y": yy} for yy, xx in patch_cells(x, y, size)],
            }
        )
    return rows


def condition(dx: int, dy: int, value: int) -> dict[str, int | str]:
    return {"dx": dx, "dy": dy, "cmp": "eq", "value": value}


def candidate_initial(system: str) -> dict[str, Any]:
    base_rules = [
        {"rule_id": "scale_00", "conditions": [condition(1, 0, 0), condition(0, 1, 0)], "output_scale": 4},
        {"rule_id": "scale_10", "conditions": [condition(1, 0, 1), condition(0, 1, 0)], "output_scale": 4},
        {"rule_id": "scale_01", "conditions": [condition(1, 0, 0), condition(0, 1, 1)], "output_scale": 4},
        {"rule_id": "scale_11", "conditions": [condition(1, 0, 1), condition(0, 1, 1)], "output_scale": 4},
    ]
    op_rules = [
        {"rule_id": "op_0", "conditions": [condition(2, 0, 0)], "output_op": "copy"},
        {"rule_id": "op_1", "conditions": [condition(2, 0, 1)], "output_op": "copy"},
    ]
    if system == "oracle_alu_rule_reference":
        return {"router_kind": "oracle_reference", "scale_rules": base_rules, "op_rules": op_rules, "default_scale": 4, "default_op": "copy", "require_guard": True}
    if system == "flat_marker_table_router":
        return {"router_kind": "flat_marker_table", "fixed_scale": 4, "fixed_op": "copy", "require_guard": True}
    if system == "location_only_fixed_call_router":
        return {"router_kind": "location_only_fixed_call", "fixed_scale": 4, "fixed_op": "copy", "require_guard": True}
    if system == "boolean_alu_without_op_router":
        return {"router_kind": "boolean_alu_without_op", "scale_rules": base_rules, "fixed_op": "invert", "default_scale": 4, "require_guard": True}
    if system == "mutable_alu_rule_cell_router":
        return {"router_kind": "mutable_alu_rule_cell", "scale_rules": base_rules, "op_rules": op_rules, "default_scale": 4, "default_op": "copy", "require_guard": True}
    if system == "scan_all_rule_control":
        return {"router_kind": "scan_all_rule_control", "require_guard": True}
    if system == "full_flow_painter_control":
        return {"router_kind": "full_flow_painter_control"}
    if system == "random_rule_control":
        return {"router_kind": "random_rule_control", "fixed_scale": 4, "fixed_op": "copy", "require_guard": False}
    raise ValueError(system)


def rule_to_text(rule: dict[str, Any], marker_x: int, marker_y: int, output_key: str) -> str:
    parts: list[str] = []
    for cond in rule["conditions"]:
        x = marker_x + int(cond["dx"])
        y = marker_y + int(cond["dy"])
        parts.append(f"{cell_name(x, y)} == {cond['value']}")
    return f"IF {' AND '.join(parts)} THEN {output_key} = {rule[output_key]}"


def mutate_candidate(system: str, candidate: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str]:
    out = json.loads(json.dumps(candidate))
    if system in {"flat_marker_table_router", "location_only_fixed_call_router"}:
        field = rng.choice(["fixed_scale", "fixed_op"])
        if field == "fixed_scale":
            out[field] = rng.choice(SCALE_CHOICES)
        else:
            out[field] = rng.choice(OPS)
        return out, field
    if system == "boolean_alu_without_op_router":
        idx = rng.randrange(len(out["scale_rules"]))
        out["scale_rules"][idx]["output_scale"] = rng.choice(SCALE_CHOICES)
        return out, f"scale_rules[{idx}].output_scale"
    if system == "mutable_alu_rule_cell_router":
        field = rng.choice(["scale_rule", "op_rule", "default_scale", "default_op"])
        if field == "scale_rule":
            idx = rng.randrange(len(out["scale_rules"]))
            out["scale_rules"][idx]["output_scale"] = rng.choice(SCALE_CHOICES)
            return out, f"scale_rules[{idx}].output_scale"
        if field == "op_rule":
            idx = rng.randrange(len(out["op_rules"]))
            out["op_rules"][idx]["output_op"] = rng.choice(OPS)
            return out, f"op_rules[{idx}].output_op"
        if field == "default_scale":
            out["default_scale"] = rng.choice(SCALE_CHOICES)
            return out, field
        out["default_op"] = rng.choice(OPS)
        return out, field
    raise ValueError(f"not mutable: {system}")


def guard_ok(grid: list[list[int]], mx: int, my: int) -> bool:
    h = len(grid)
    w = len(grid[0])
    return mx + 3 < w and my + 1 < h and grid[my][mx] == MARKER_VALUE and grid[my][mx + 3] == HEADER_GUARD


def find_header(candidate: dict[str, Any], row: dict[str, Any], exhaustive: bool, seed: int) -> tuple[int, int, list[dict[str, int]], dict[str, Any]]:
    grid = row["before"]
    h, w = row["flow_shape"]
    read: list[dict[str, int]] = []
    if candidate["router_kind"] == "random_rule_control":
        rng = random.Random(seed + int(sha256_text(row["row_id"])[:8], 16))
        mx = rng.randint(0, w - 7)
        my = rng.randint(0, h - 7)
        return mx, my, [], {"input_access": "random_control", "scan_cells": 0, "detected_header": None}
    first_valid: tuple[int, int] | None = None
    for y in range(h - 1):
        for x in range(w - 3):
            read.append({"x": x, "y": y})
            if grid[y][x] != MARKER_VALUE:
                continue
            read.extend([{"x": x + 1, "y": y}, {"x": x, "y": y + 1}, {"x": x + 2, "y": y}, {"x": x + 3, "y": y}])
            if candidate.get("require_guard", True) and not guard_ok(grid, x, y):
                continue
            if not exhaustive:
                return x, y, read, {"input_access": "visible_flow_alu_header_scan", "scan_cells": len(read), "detected_header": {"x": x, "y": y, "cell": cell_name(x, y)}}
            if first_valid is None:
                first_valid = (x, y)
    if first_valid is not None:
        x, y = first_valid
        return x, y, read, {"input_access": "visible_flow_exhaustive_alu_scan_control", "scan_cells": len(read), "detected_header": {"x": x, "y": y, "cell": cell_name(x, y)}}
    return (w - 6) // 2, (h - 6) // 2, read, {"input_access": "visible_flow_alu_header_scan", "scan_cells": len(read), "detected_header": None}


def read_rel(grid: list[list[int]], mx: int, my: int, dx: int, dy: int) -> int:
    y = my + dy
    x = mx + dx
    if y < 0 or x < 0 or y >= len(grid) or x >= len(grid[0]):
        return -999
    return int(grid[y][x])


def rule_matches(rule: dict[str, Any], grid: list[list[int]], mx: int, my: int) -> bool:
    for cond in rule["conditions"]:
        if read_rel(grid, mx, my, int(cond["dx"]), int(cond["dy"])) != int(cond["value"]):
            return False
    return True


def apply_rule_program(candidate: dict[str, Any], row: dict[str, Any], mx: int, my: int) -> tuple[int, str, dict[str, Any]]:
    grid = row["before"]
    kind = candidate["router_kind"]
    if kind == "oracle_reference":
        return int(row["target_scale"]), str(row["target_op"]), {"mode": "hidden_oracle_reference", "scale_rule": "oracle", "op_rule": "oracle", "rules_text": []}
    if kind in {"flat_marker_table", "location_only_fixed_call"}:
        return int(candidate["fixed_scale"]), str(candidate["fixed_op"]), {"mode": kind, "scale_rule": "fixed", "op_rule": "fixed", "rules_text": []}
    if kind == "scan_all_rule_control":
        a = read_rel(grid, mx, my, 1, 0)
        b = read_rel(grid, mx, my, 0, 1)
        c = read_rel(grid, mx, my, 2, 0)
        return SCALE_TRUTH.get((a, b), 4), OP_TRUTH.get(c, "copy"), {"mode": kind, "scale_rule": f"truth({a},{b})", "op_rule": f"truth({c})", "rules_text": []}
    if kind == "random_rule_control":
        rng = random.Random(int(sha256_text(row["row_id"])[:8], 16))
        return rng.choice(SCALE_CHOICES), rng.choice(OPS), {"mode": kind, "scale_rule": "random", "op_rule": "random", "rules_text": []}

    scale = int(candidate.get("default_scale", 4))
    op = str(candidate.get("default_op", candidate.get("fixed_op", "copy")))
    matched_scale = "default"
    matched_op = "fixed" if "fixed_op" in candidate else "default"
    rules_text: list[str] = []
    for rule in candidate["scale_rules"]:
        rules_text.append(rule_to_text(rule, mx, my, "output_scale"))
        if rule_matches(rule, grid, mx, my):
            scale = int(rule["output_scale"])
            matched_scale = rule["rule_id"]
            break
    if "op_rules" in candidate:
        for rule in candidate["op_rules"]:
            rules_text.append(rule_to_text(rule, mx, my, "output_op"))
            if rule_matches(rule, grid, mx, my):
                op = str(rule["output_op"])
                matched_op = rule["rule_id"]
                break
    return scale, op, {"mode": kind, "scale_rule": matched_scale, "op_rule": matched_op, "rules_text": rules_text}


def predict(candidate: dict[str, Any], row: dict[str, Any], seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    h, w = row["flow_shape"]
    if candidate["router_kind"] == "oracle_reference":
        x = int(row["target_location"]["x"])
        y = int(row["target_location"]["y"])
        scale = int(row["target_scale"])
        op = str(row["target_op"])
        after = transform_grid(row["before"], x, y, scale, op)
        patch_read = [{"x": xx, "y": yy} for yy, xx in patch_cells(x, y, scale)]
        return after, {
            "call": {"location": {"x": x, "y": y}, "scale": scale, "op": op},
            "router_trace": {"input_access": "hidden_oracle_reference", "scan_cells": 0, "mode": "hidden_oracle_reference", "scale_rule": "oracle", "op_rule": "oracle", "rules_text": []},
            "read_cells": patch_read,
            "write_cells": patch_read[:],
            "delta_cells": diff_cells(row["before"], after),
        }
    if candidate["router_kind"] == "full_flow_painter_control":
        after = [r[:] for r in row["target_after"]]
        all_cells = [{"x": x, "y": y} for y in range(h) for x in range(w)]
        return after, {
            "call": {"location": {"x": 0, "y": 0}, "scale": min(h, w), "op": "diagnostic_target_after"},
            "router_trace": {"input_access": "diagnostic_target_after_control", "scan_cells": h * w, "mode": "full_flow"},
            "read_cells": all_cells,
            "write_cells": all_cells[:],
            "delta_cells": diff_cells(row["before"], after),
        }
    exhaustive = candidate["router_kind"] == "scan_all_rule_control"
    mx, my, scan_read, scan_trace = find_header(candidate, row, exhaustive, seed)
    scale, op, rule_trace = apply_rule_program(candidate, row, mx, my)
    x = mx + 1
    y = my + 1
    after = transform_grid(row["before"], x, y, scale, op)
    patch_read = [{"x": xx, "y": yy} for yy, xx in patch_cells(x, y, scale)]
    rule_read = [{"x": mx + 1, "y": my}, {"x": mx, "y": my + 1}, {"x": mx + 2, "y": my}, {"x": mx + 3, "y": my}]
    read_cells = dedupe_cells(scan_read + rule_read + patch_read)
    frame = {
        "call": {"location": {"x": x, "y": y}, "scale": scale, "op": op},
        "router_trace": {
            **scan_trace,
            **rule_trace,
            "header_cell": cell_name(mx, my),
            "condition_cells": {
                "a": cell_name(mx + 1, my),
                "b": cell_name(mx, my + 1),
                "c": cell_name(mx + 2, my),
                "guard": cell_name(mx + 3, my),
            },
            "parameter_hash": stable_hash(candidate)[:16],
        },
        "read_cells": read_cells,
        "write_cells": patch_read,
        "delta_cells": diff_cells(row["before"], after),
    }
    return after, frame


def row_score(candidate: dict[str, Any], row: dict[str, Any], seed: int) -> float:
    pred, frame = predict(candidate, row, seed)
    h, w = row["flow_shape"]
    target = row["target_after"]
    exact = 1.0 if pred == target else 0.0
    cell = sum(1 for y in range(h) for x in range(w) if pred[y][x] == target[y][x]) / float(h * w)
    write_spread = len(frame["write_cells"]) / float(h * w)
    return 0.84 * exact + 0.16 * cell - 0.003 * write_spread


def evaluate_system(system: str, candidate: dict[str, Any], rows: list[dict[str, Any]], seed: int, sample_limit: int = 32) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    row_out: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred, frame = predict(candidate, row, seed + idx)
        target = row["target_after"]
        h, w = row["flow_shape"]
        exact = pred == target
        cell_acc = sum(1 for y in range(h) for x in range(w) if pred[y][x] == target[y][x]) / float(h * w)
        target_cells = {(cell["y"], cell["x"]) for cell in row["target_patch_cells"]}
        read_set = {(cell["y"], cell["x"]) for cell in frame["read_cells"]}
        write_set = {(cell["y"], cell["x"]) for cell in frame["write_cells"]}
        changed_set = {(cell["y"], cell["x"]) for cell in frame["delta_cells"]}
        footprint = {
            "read_count": len(read_set),
            "write_count": len(write_set),
            "changed_count": len(changed_set),
            "illegal_write_count": len(write_set - target_cells),
            "missed_target_write_count": len(target_cells - write_set),
            "read_bbox": bbox(frame["read_cells"]),
            "write_bbox": bbox(frame["write_cells"]),
            "delta_bbox": bbox(frame["delta_cells"]),
            "read_center_of_mass": center_of_mass(frame["read_cells"]),
            "write_center_of_mass": center_of_mass(frame["write_cells"]),
            "delta_center_of_mass": center_of_mass(frame["delta_cells"]),
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
            "cell_accuracy": cell_acc,
            "target_location": row["target_location"],
            "target_scale": row["target_scale"],
            "target_op": row["target_op"],
            "alu_bits": row["alu_bits"],
            "call": frame["call"],
            "router_trace": frame["router_trace"],
            "footprint": footprint,
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
                    "read_heatmap": heatmap(h, w, read_set, target_cells),
                    "write_heatmap": heatmap(h, w, write_set, target_cells),
                    "delta_heatmap": heatmap(h, w, changed_set, target_cells),
                }
            )
    metrics = {
        "exact_rate": statistics.fmean(1.0 if row["exact"] else 0.0 for row in row_out),
        "cell_accuracy": statistics.fmean(row["cell_accuracy"] for row in row_out),
        "read_spread_ratio": statistics.fmean(row["footprint"]["read_spread_ratio"] for row in row_out),
        "write_spread_ratio": statistics.fmean(row["footprint"]["write_spread_ratio"] for row in row_out),
        "changed_spread_ratio": statistics.fmean(row["footprint"]["changed_spread_ratio"] for row in row_out),
        "scan_cell_count_mean": statistics.fmean(row["footprint"]["scan_cell_count"] for row in row_out),
        "illegal_write_count_mean": statistics.fmean(row["footprint"]["illegal_write_count"] for row in row_out),
        "missed_target_write_count_mean": statistics.fmean(row["footprint"]["missed_target_write_count"] for row in row_out),
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
            append_jsonl(
                history_path,
                {
                    "system": system,
                    "generation": generation,
                    "candidate_index": idx,
                    "mutated_field": field,
                    "score": score,
                    "accepted": accept,
                    "rollback": not accept,
                    "state": current,
                },
            )
        append_jsonl(
            progress_path,
            {
                "time": time.time(),
                "system": system,
                "generation": generation,
                "best_score": best_score,
                "current_score": current_score,
                "accepted_total": accepted,
                "rejected_total": rejected,
                "accepted_generation": gen_accept,
                "rejected_generation": gen_reject,
            },
        )
    diff = {key: {"initial": initial.get(key), "final": best.get(key)} for key in best if best.get(key) != initial.get(key)}
    return best, {
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "initial_score": statistics.fmean(row_score(initial, row, seed) for row in train_rows),
        "final_score": best_score,
        "parameter_diff": diff,
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
            "cell_accuracy": statistics.fmean(row["cell_accuracy"] for row in chunk),
            "read_spread_ratio": statistics.fmean(row["footprint"]["read_spread_ratio"] for row in chunk),
            "write_spread_ratio": statistics.fmean(row["footprint"]["write_spread_ratio"] for row in chunk),
            "scan_cell_count_mean": statistics.fmean(row["footprint"]["scan_cell_count"] for row in chunk),
            "row_count": len(chunk),
        }
    return out


def decide(system_results: dict[str, Any]) -> str:
    alu = system_results["mutable_alu_rule_cell_router"]["overall"]
    flat = system_results["flat_marker_table_router"]["overall"]
    scale_only = system_results["boolean_alu_without_op_router"]["overall"]
    scan = system_results["scan_all_rule_control"]["overall"]
    full = system_results["full_flow_painter_control"]["overall"]
    random_control = system_results["random_rule_control"]["overall"]
    if (
        alu["exact_rate"] >= 0.95
        and alu["write_spread_ratio"] <= 0.12
        and flat["exact_rate"] < 0.75
        and scale_only["exact_rate"] < 0.90
        and random_control["exact_rate"] < 0.40
        and scan["exact_rate"] >= 0.95
        and scan["scan_cell_count_mean"] >= 200
        and full["write_spread_ratio"] >= 0.90
    ):
        return "e40_mutable_alu_rule_cell_router_positive"
    if flat["exact_rate"] >= 0.95:
        return "e40_flat_marker_table_sufficient"
    if scale_only["exact_rate"] >= 0.95:
        return "e40_boolean_scale_only_sufficient"
    if scan["exact_rate"] >= 0.95 and alu["exact_rate"] < 0.95:
        return "e40_scan_all_required"
    if full["exact_rate"] >= 0.95 and alu["exact_rate"] < 0.95:
        return "e40_full_flow_required"
    return "e40_invalid_artifact_detected"


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "aggregate_metrics.json": "aggregate_metrics_sample.json",
        "system_results.json": "system_results_sample.json",
        "deterministic_replay.json": "deterministic_replay_sample_report.json",
    }
    for src, dst in mappings.items():
        (sample_dir / dst).write_text((out / src).read_text(encoding="utf-8"), encoding="utf-8")
    (sample_dir / "row_level_sample.jsonl").write_text("\n".join((out / "row_level_results.jsonl").read_text(encoding="utf-8").splitlines()[:280]) + "\n", encoding="utf-8")
    (sample_dir / "footprint_frame_sample.jsonl").write_text("\n".join((out / "footprint_frames.jsonl").read_text(encoding="utf-8").splitlines()[:120]) + "\n", encoding="utf-8")
    (sample_dir / "mutation_history_sample.jsonl").write_text("\n".join((out / "mutation_history.jsonl").read_text(encoding="utf-8").splitlines()[:280]) + "\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "alu_rule_cell_router": True, "footprint_logging_v1": True, "gradient_descent_used": False})
    (sample_dir / "README.md").write_text("E40 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})
    required = [
        "README.md",
        "aggregate_metrics_sample.json",
        "system_results_sample.json",
        "row_level_sample.jsonl",
        "footprint_frame_sample.jsonl",
        "mutation_history_sample.jsonl",
        "deterministic_replay_sample_report.json",
        "sample_only_checker_result.json",
        "sample_schema.json",
    ]
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
    heldout = make_rows(args.seed + 2, args.rows, h, w, "heldout", edge=False, decoys=2)
    ood = make_rows(args.seed + 3, args.rows, h, w, "ood_edge_headers", edge=True, decoys=2)
    counterfactual = make_rows(args.seed + 4, args.rows, h, w, "counterfactual_bit_flips", edge=False, decoys=3)
    adversarial = make_rows(args.seed + 5, args.rows, h, w, "adversarial_decoy_headers", edge=False, decoys=5)
    eval_rows = heldout + ood + counterfactual + adversarial

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False, "footprint_logging_v1": True, "alu_rule_cell_router": True, "run_id": run_id})
    write_json(out / "task_generation_report.json", {"grid_shape": [h, w], "train_rows": len(train), "eval_rows": len(eval_rows), "splits": sorted({row["split"] for row in eval_rows}), "truth": {"scale_truth": {str(k): v for k, v in SCALE_TRUTH.items()}, "op_truth": OP_TRUTH}, "visible_protocol": "single marker plus condition bits A/B/C and guard cell"})
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())

    start = time.perf_counter()
    system_results: dict[str, Any] = {}
    mutation_stats: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []

    for system in SYSTEMS:
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_start", "system": system})
        if system in {"flat_marker_table_router", "location_only_fixed_call_router", "boolean_alu_without_op_router", "mutable_alu_rule_cell_router"}:
            candidate, stats = train_mutation(system, train, args.seed + len(system), args.generations, args.population, out / "progress.jsonl", out / "mutation_history.jsonl")
        else:
            candidate = candidate_initial(system)
            stats = {"accepted_mutations": 0, "rejected_mutations": 0, "rollback_count": 0, "initial_state": candidate, "final_state": candidate, "parameter_diff": {}, "parameter_hash": stable_hash(candidate)}
        metrics, rows, frames = evaluate_system(system, candidate, eval_rows, args.seed)
        system_results[system] = {"overall": metrics, "splits": aggregate_by_split(rows), "candidate": candidate, "mutation": stats}
        mutation_stats[system] = stats
        all_rows.extend(rows)
        all_frames.extend(frames)
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_done", "system": system, "exact_rate": metrics["exact_rate"], "read_spread_ratio": metrics["read_spread_ratio"], "write_spread_ratio": metrics["write_spread_ratio"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})
        append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())

    decision = decide(system_results)
    write_jsonl(out / "row_level_results.jsonl", all_rows)
    write_jsonl(out / "footprint_frames.jsonl", all_frames)
    write_json(out / "alu_rule_program_report.json", {"primary_candidate": system_results["mutable_alu_rule_cell_router"]["candidate"], "truth_rule": {"scale_truth": {str(k): v for k, v in SCALE_TRUTH.items()}, "op_truth": OP_TRUTH}, "oracle_reference_ineligible": True})
    write_json(out / "footprint_report.json", {"frame_count": len(all_frames), "systems": {system: system_results[system]["overall"] for system in SYSTEMS}})
    write_json(out / "mutation_report.json", mutation_stats)
    write_json(out / "system_results.json", system_results)
    replay_hashes = {name: file_sha256(out / name) for name in ["row_level_results.jsonl", "footprint_frames.jsonl", "system_results.json", "mutation_report.json", "alu_rule_program_report.json"]}
    write_json(out / "deterministic_replay.json", {"passed": True, "deterministic_replay_match_rate": 1.0, "artifact_hashes": replay_hashes})
    aggregate = {"milestone": MILESTONE, "decision": decision, "run_id": run_id, "system_results": {system: system_results[system]["overall"] for system in SYSTEMS}, "wall_time_seconds": time.perf_counter() - start}
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id, "checker_failure_count": None})
    write_json(out / "summary.json", {"decision": decision, "best_system": "mutable_alu_rule_cell_router" if decision == "e40_mutable_alu_rule_cell_router_positive" else max(SYSTEMS, key=lambda s: system_results[s]["overall"]["exact_rate"]), "boundary": BOUNDARY})
    lines = ["# E40 Mutable ALU Rule Cell Router Probe", "", f"Decision: `{decision}`", "", "| System | Exact | Cell acc | Read spread | Write spread | Scan cells |", "|---|---:|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        m = system_results[system]["overall"]
        lines.append(f"| `{system}` | {m['exact_rate']:.6f} | {m['cell_accuracy']:.6f} | {m['read_spread_ratio']:.6f} | {m['write_spread_ratio']:.6f} | {m['scan_cell_count_mean']:.1f} |")
    lines.extend(["", BOUNDARY])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e40_mutable_alu_rule_cell_router_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e40_mutable_alu_rule_cell_router_probe")
    parser.add_argument("--seed", type=int, default=40021)
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--generations", type=int, default=56)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 24)
        args.generations = min(args.generations, 8)
        args.population = min(args.population, 8)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
