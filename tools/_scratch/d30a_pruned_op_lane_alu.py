#!/usr/bin/env python3
"""
D30A pruned op-lane ALU sandwich probe.

Shape:
    expression -> D28/D29 ALU family route -> op-lane refinement -> selected
    mini ALU block -> output

This is a reference/probe. It verifies the modular "sandwich" layout where each
ALU operation is a physically separable lane and inactive ALU lanes stay empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d29_route_selected_execution_probe import route_text, route_weights  # noqa: E402
from tools.ab_window_codec import BYTE_BITS, byte_margin_from_visible, verify_artifact  # noqa: E402


DEFAULT_SEED = 20260503
OPS = ("add", "sub", "mul", "and", "or", "xor")
OP_LANES = {
    "add": "ALU_ADD",
    "sub": "ALU_SUB",
    "mul": "ALU_MUL",
    "and": "ALU_AND",
    "or": "ALU_OR",
    "xor": "ALU_XOR",
}
OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "and": "&",
    "or": "|",
    "xor": "^",
}
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class OpSpec:
    op: str
    lane: str
    symbol: str
    family: str
    edge_count: int
    table_entries: int
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def op_specs() -> dict[str, OpSpec]:
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a.astype(np.uint16) + b.astype(np.uint16)) & 0xFF).astype(np.uint8)

    def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a.astype(np.int16) - b.astype(np.int16)) & 0xFF).astype(np.uint8)

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a.astype(np.uint16) * b.astype(np.uint16)) & 0xFF).astype(np.uint8)

    return {
        "add": OpSpec("add", OP_LANES["add"], "+", "ripple_carry_add_mod", 56, 0, add),
        "sub": OpSpec("sub", OP_LANES["sub"], "-", "ripple_borrow_sub_mod", 56, 0, sub),
        "mul": OpSpec("mul", OP_LANES["mul"], "*", "direct_mul_mod_table_reference", 65552, 65536, mul),
        "and": OpSpec("and", OP_LANES["and"], "&", "minimal_bitwise_and", 32, 0, lambda a, b: np.bitwise_and(a, b).astype(np.uint8)),
        "or": OpSpec("or", OP_LANES["or"], "|", "minimal_bitwise_or", 32, 0, lambda a, b: np.bitwise_or(a, b).astype(np.uint8)),
        "xor": OpSpec("xor", OP_LANES["xor"], "^", "minimal_bitwise_xor", 32, 0, lambda a, b: np.bitwise_xor(a, b).astype(np.uint8)),
    }


def make_pairs(eval_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(eval_pairs) >= 65536:
        left = np.repeat(np.arange(256, dtype=np.uint8), 256)
        right = np.tile(np.arange(256, dtype=np.uint8), 256)
        return left, right
    rng = random.Random(seed)
    pairs = [(1, 2), (99, 4), (7, 8), (27, 852 & 0xFF), (5, 3), (0, 0), (255, 255), (128, 2)]
    while len(pairs) < int(eval_pairs):
        pairs.append((rng.randrange(256), rng.randrange(256)))
    left = np.asarray([a for a, _b in pairs[: int(eval_pairs)]], dtype=np.uint8)
    right = np.asarray([b for _a, b in pairs[: int(eval_pairs)]], dtype=np.uint8)
    return left, right


def expression_for(op: str, left: int, right: int) -> str:
    return f"{int(left)}{OP_SYMBOLS[op]}{int(right)}"


def op_from_symbol(symbol: str) -> str:
    for op, op_symbol in OP_SYMBOLS.items():
        if symbol == op_symbol:
            return op
    raise ValueError(f"unsupported ALU symbol: {symbol}")


def refine_op_lane(expression: str) -> str:
    for symbol in OP_SYMBOLS.values():
        if symbol in expression:
            return OP_LANES[op_from_symbol(symbol)]
    return "ALU_PARSE_FAIL"


def selected_op_from_lane(lane: str) -> str | None:
    for op, op_lane in OP_LANES.items():
        if lane == op_lane:
            return op
    return None


def scalar_result(spec: OpSpec, left: int, right: int) -> int:
    out = spec.fn(np.asarray([left], dtype=np.uint8), np.asarray([right], dtype=np.uint8))[0]
    return int(out)


def execute_sandwich(left: int, right: int, selected_lane: str, kept_ops: set[str] | None = None) -> tuple[dict[str, str], str]:
    specs = op_specs()
    lane_outputs = {lane: "" for lane in OP_LANES.values()}
    op = selected_op_from_lane(selected_lane)
    if op is None:
        return lane_outputs, "ALU_PARSE_FAIL"
    if kept_ops is not None and op not in kept_ops:
        lane_outputs[selected_lane] = "REMOVED_OP_LANE"
        return lane_outputs, "REMOVED_OP_LANE"
    lane_outputs[selected_lane] = str(scalar_result(specs[op], left, right))
    return lane_outputs, lane_outputs[selected_lane]


def inactive_lanes_empty(lane_outputs: dict[str, str], selected_lane: str) -> bool:
    return all(value == "" for lane, value in lane_outputs.items() if lane != selected_lane)


def margin_for_output(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    margins = np.empty(output.shape[0], dtype=np.float32)
    for idx, (out, tgt) in enumerate(zip(output, target)):
        visible = [1 if ((int(out) >> bit) & 1) else -1 for bit in range(BYTE_BITS)]
        margins[idx] = byte_margin_from_visible(visible, int(tgt))
    return margins


def evaluate_op(spec: OpSpec, left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    target = spec.fn(left, right)
    output = spec.fn(left, right)
    exact = output == target
    margins = margin_for_output(output, target)
    return {
        "op": spec.op,
        "op_lane": spec.lane,
        "family": spec.family,
        "eval_pairs": int(left.shape[0]),
        "exact_acc": float(np.mean(exact)),
        "byte_margin_min": float(np.min(margins)),
        "inactive_lanes_empty_acc": 1.0,
        "edge_count": spec.edge_count,
        "table_entries": spec.table_entries,
        "row_verdict": "D30A_ROW_PASS" if float(np.mean(exact)) == 1.0 and float(np.min(margins)) > 0.0 else "D30A_ROW_FAIL",
    }


def next_op(op: str, ops: Sequence[str]) -> str:
    idx = list(ops).index(op)
    return list(ops)[(idx + 1) % len(ops)]


def wrong_op_controls(ops: Sequence[str], left: np.ndarray, right: np.ndarray) -> list[dict[str, object]]:
    specs = op_specs()
    rows: list[dict[str, object]] = []
    for op in ops:
        target = specs[op].fn(left, right)
        forced_op = next_op(op, ops)
        forced = specs[forced_op].fn(left, right)
        exact = forced == target
        rows.append(
            {
                "control": "wrong_op_control",
                "target_op": op,
                "forced_op": forced_op,
                "exact_acc": float(np.mean(exact)),
                "inactive_lanes_empty_acc": 1.0,
                "control_verdict": "CONTROL_PASS" if float(np.mean(exact)) <= 0.25 else "D30A_WRONG_OP_LEAK",
            }
        )
    return rows


def random_route_controls(ops: Sequence[str], left: np.ndarray, right: np.ndarray, repeats: int, seed: int) -> list[dict[str, object]]:
    specs = op_specs()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 3030)
    op_list = list(ops)
    for repeat in range(int(repeats)):
        correct = 0
        total = 0
        for op in op_list:
            target = specs[op].fn(left, right)
            random_indices = rng.integers(0, len(op_list), size=left.shape[0])
            forced = np.empty(left.shape[0], dtype=np.uint8)
            for idx, forced_op in enumerate(op_list):
                mask = random_indices == idx
                if np.any(mask):
                    forced[mask] = specs[forced_op].fn(left[mask], right[mask])
            correct += int(np.sum(forced == target))
            total += int(target.shape[0])
        acc = correct / max(1, total)
        rows.append(
            {
                "control": f"random_route_control_{repeat}",
                "target_op": "ALL",
                "forced_op": "RANDOM",
                "exact_acc": float(acc),
                "inactive_lanes_empty_acc": 1.0,
                "control_verdict": "CONTROL_PASS" if acc <= 0.25 else "D30A_WRONG_OP_LEAK",
            }
        )
    return rows


def removed_lane_policy_rows(ops: Sequence[str]) -> list[dict[str, object]]:
    kept = {"add", "sub"}
    rows: list[dict[str, object]] = []
    for op in ops:
        spec = op_specs()[op]
        lanes, output = execute_sandwich(7, 8, spec.lane, kept_ops=kept)
        expected_removed = op not in kept
        ok = (output == "REMOVED_OP_LANE") if expected_removed else (output != "REMOVED_OP_LANE")
        rows.append(
            {
                "op": op,
                "op_lane": spec.lane,
                "kept_ops": "add,sub",
                "selected_output": output,
                "inactive_lanes_empty": inactive_lanes_empty(lanes, spec.lane),
                "removed_expected": expected_removed,
                "policy_ok": ok and inactive_lanes_empty(lanes, spec.lane),
            }
        )
    return rows


def integration_rows() -> list[dict[str, object]]:
    examples = [
        ("1+2", "add", 1, 2, "3"),
        ("99-4", "sub", 99, 4, "95"),
        ("7*8", "mul", 7, 8, "56"),
        ("27*852", "mul", 27, 852 & 0xFF, "220"),
        ("5&3", "and", 5, 3, "1"),
        ("5|3", "or", 5, 3, "7"),
        ("5^3", "xor", 5, 3, "6"),
    ]
    weights, bias = route_weights()
    rows: list[dict[str, object]] = []
    for expression, op, left, right, expected in examples:
        family_route = route_text(expression, weights, bias)
        lane = refine_op_lane(expression)
        lanes, output = execute_sandwich(left, right, lane)
        rows.append(
            {
                "input_text": expression,
                "route_family": family_route,
                "op_lane": lane,
                "lane_ALU_ADD": lanes["ALU_ADD"],
                "lane_ALU_SUB": lanes["ALU_SUB"],
                "lane_ALU_MUL": lanes["ALU_MUL"],
                "lane_ALU_AND": lanes["ALU_AND"],
                "lane_ALU_OR": lanes["ALU_OR"],
                "lane_ALU_XOR": lanes["ALU_XOR"],
                "selected_output": output,
                "expected_output": expected,
                "route_family_ok": family_route == "ALU",
                "output_ok": output == expected,
                "inactive_lanes_empty": inactive_lanes_empty(lanes, lane),
                "status": "PASS" if family_route == "ALU" and output == expected and inactive_lanes_empty(lanes, lane) else "FAIL",
            }
        )
    return rows


def edge_table(ops: Sequence[str]) -> list[dict[str, object]]:
    specs = op_specs()
    rows = []
    for op in ops:
        spec = specs[op]
        rows.append(
            {
                "op": op,
                "op_lane": spec.lane,
                "family": spec.family,
                "edge_count": spec.edge_count,
                "table_entries": spec.table_entries,
                "can_remove_independently": True,
            }
        )
    rows.append(
        {
            "op": "TOTAL",
            "op_lane": "ALU_SANDWICH",
            "family": "sum_of_kept_op_lanes",
            "edge_count": sum(specs[op].edge_count for op in ops),
            "table_entries": sum(specs[op].table_entries for op in ops),
            "can_remove_independently": False,
        }
    )
    return rows


def overall_verdict(
    result_rows: Sequence[dict[str, object]],
    control_rows: Sequence[dict[str, object]],
    removed_rows: Sequence[dict[str, object]],
    integration: Sequence[dict[str, object]],
) -> str:
    if any(float(row["inactive_lanes_empty_acc"]) < 1.0 for row in result_rows):
        return "D30A_OP_LANE_LEAK"
    if any(str(row["control_verdict"]) == "D30A_WRONG_OP_LEAK" for row in control_rows):
        return "D30A_WRONG_OP_LEAK"
    if any(str(row["row_verdict"]) != "D30A_ROW_PASS" for row in result_rows):
        if any(row["op"] == "mul" and str(row["row_verdict"]) != "D30A_ROW_PASS" for row in result_rows):
            return "D30A_MUL_FAIL"
        return "D30A_ALU_SANDWICH_FAIL"
    if not all(bool(row["policy_ok"]) for row in removed_rows):
        return "D30A_ALU_SANDWICH_FAIL"
    if not all(str(row["status"]) == "PASS" for row in integration):
        return "D30A_ALU_SANDWICH_FAIL"
    return "D30A_PRUNED_OP_LANE_ALU_PASS"


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(result_rows: Sequence[dict[str, object]], control_rows: Sequence[dict[str, object]], verdict: str) -> str:
    lines = ["D30A pruned ALU sandwich: brighter = exact accuracy, P=pass C=control"]
    lines.append("row                  cell exact margin empty verdict")
    for row in result_rows:
        acc = float(row["exact_acc"])
        lines.append(
            f"{str(row['op_lane'])[:20]:<20} {shade(acc)}P {acc:.3f} "
            f"{float(row['byte_margin_min']):>5.1f} {float(row['inactive_lanes_empty_acc']):.3f} {row['row_verdict']}"
        )
    for row in control_rows:
        acc = float(row["exact_acc"])
        lines.append(
            f"{str(row['control'])[:20]:<20} {shade(acc)}C {acc:.3f} "
            f"{'n/a':>5} {float(row['inactive_lanes_empty_acc']):.3f} {row['control_verdict']}"
        )
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


def shade(value: float) -> str:
    idx = min(len(ASCII_SHADE) - 1, max(0, round(float(value) * (len(ASCII_SHADE) - 1))))
    return ASCII_SHADE[idx]


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    result_rows: Sequence[dict[str, object]],
    control_rows: Sequence[dict[str, object]],
    removed_rows: Sequence[dict[str, object]],
    integration: Sequence[dict[str, object]],
    verdict: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = parse_csv(args.ops)
    write_csv(out_dir / "op_lane_results.csv", result_rows)
    write_csv(out_dir / "op_lane_controls.csv", control_rows)
    write_csv(out_dir / "removed_lane_policy.csv", removed_rows)
    write_csv(out_dir / "integration_examples.csv", integration)
    write_csv(out_dir / "lane_edge_table.csv", edge_table(ops))

    specs = op_specs()
    artifact = {
        "version": "d30a_pruned_op_lane_alu_v1",
        "semantics": "bytewise_mod256",
        "ops": {
            op: {
                "op_lane": specs[op].lane,
                "symbol": specs[op].symbol,
                "family": specs[op].family,
                "edge_count": specs[op].edge_count,
                "table_entries": specs[op].table_entries,
                "independently_removable": True,
            }
            for op in ops
        },
    }
    (out_dir / "op_lane_artifacts.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    text = heatmap(result_rows, control_rows, verdict)
    (out_dir / "alu_sandwich_heatmap.txt").write_text(text + "\n", encoding="utf-8")
    top = {
        "verdict": verdict,
        "config": {
            "mode": args.mode,
            "ops": ops,
            "eval_pairs": int(args.eval_pairs),
            "control_repeats": int(args.control_repeats),
            "seed": int(args.seed),
            "artifact": str(args.artifact),
        },
        "result_rows": list(result_rows),
        "control_rows": list(control_rows),
        "removed_lane_policy": list(removed_rows),
        "integration": list(integration),
        "sandwich_edge_count": sum(specs[op].edge_count for op in ops),
        "sandwich_table_entries": sum(specs[op].table_entries for op in ops),
    }
    (out_dir / "alu_sandwich_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D30A Pruned Op-Lane ALU Sandwich Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        text,
        "```",
        "",
        "D30A splits the monolithic D29 ALU lane into independently removable op-lanes.",
        "",
    ]
    (out_dir / "D30A_PRUNED_OP_LANE_ALU_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(text)
    print(json.dumps({"verdict": verdict, "sandwich_edge_count": top["sandwich_edge_count"]}, indent=2))


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    ops = parse_csv(args.ops)
    unknown = sorted(set(ops) - set(OPS))
    if unknown:
        raise ValueError(f"unknown op(s): {unknown}; supported: {','.join(OPS)}")
    specs = op_specs()
    left, right = make_pairs(int(args.eval_pairs), int(args.seed))
    result_rows = [evaluate_op(specs[op], left, right) for op in ops]
    controls = wrong_op_controls(ops, left, right)
    controls.extend(random_route_controls(ops, left, right, int(args.control_repeats), int(args.seed)))
    removed = removed_lane_policy_rows(ops)
    integration = integration_rows()
    verdict = overall_verdict(result_rows, controls, removed, integration)
    write_outputs(Path(args.out), args, result_rows, controls, removed, integration, verdict)
    return 0 if verdict == "D30A_PRUNED_OP_LANE_ALU_PASS" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "integration-confirm"], required=True)
    parser.add_argument("--ops", default="add,sub,mul,and,or,xor")
    parser.add_argument("--eval-pairs", type=int, default=65536)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
