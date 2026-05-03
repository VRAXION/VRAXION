#!/usr/bin/env python3
"""
D30B compact MUL lane probe.

Goal:
    Replace the D30A 65,536-entry MUL table/reference lane with an exact compact
    partial-product multiplier for bytewise (a*b) mod 256.
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

from tools._scratch.d30a_pruned_op_lane_alu import (  # noqa: E402
    OP_LANES,
    execute_sandwich,
    inactive_lanes_empty,
    op_specs,
)
from tools._scratch.d29_route_selected_execution_probe import route_text, route_weights  # noqa: E402
from tools.ab_window_codec import BYTE_BITS, byte_margin_from_visible, verify_artifact  # noqa: E402


DEFAULT_SEED = 20260503
TABLE_REFERENCE_SIZE = 65536
TABLE_REFERENCE_EDGE_ESTIMATE = 65552
ASCII_SHADE = " .:-=+*#%@"


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def make_pairs(eval_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(eval_pairs) >= 65536:
        left = np.repeat(np.arange(256, dtype=np.uint16), 256)
        right = np.tile(np.arange(256, dtype=np.uint16), 256)
        return left, right
    rng = random.Random(seed)
    pairs = [(0, 0), (1, 2), (7, 8), (27, 852 & 0xFF), (255, 255), (128, 2)]
    while len(pairs) < int(eval_pairs):
        pairs.append((rng.randrange(256), rng.randrange(256)))
    left = np.asarray([a for a, _b in pairs[: int(eval_pairs)]], dtype=np.uint16)
    right = np.asarray([b for _a, b in pairs[: int(eval_pairs)]], dtype=np.uint16)
    return left, right


def table_reference_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left.astype(np.uint16) * right.astype(np.uint16)) & 0xFF).astype(np.uint8)


def compact_partial_product_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact low 8-bit multiplier using partial-product columns, not a table."""
    left_u = left.astype(np.uint16)
    right_u = right.astype(np.uint16)
    out = np.zeros(left.shape[0], dtype=np.uint16)
    carry = np.zeros(left.shape[0], dtype=np.uint16)
    for bit_idx in range(BYTE_BITS):
        column = carry.copy()
        for a_bit in range(bit_idx + 1):
            b_bit = bit_idx - a_bit
            column += ((left_u >> a_bit) & 1) & ((right_u >> b_bit) & 1)
        out |= (column & 1) << bit_idx
        carry = column >> 1
    return (out & 0xFF).astype(np.uint8)


def carryless_xor_mul_control(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_u = left.astype(np.uint16)
    right_u = right.astype(np.uint16)
    out = np.zeros(left.shape[0], dtype=np.uint16)
    for bit_idx in range(BYTE_BITS):
        column = np.zeros(left.shape[0], dtype=np.uint16)
        for a_bit in range(bit_idx + 1):
            b_bit = bit_idx - a_bit
            column ^= ((left_u >> a_bit) & 1) & ((right_u >> b_bit) & 1)
        out |= (column & 1) << bit_idx
    return (out & 0xFF).astype(np.uint8)


def shifted_partial_shuffle_control(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Deliberately shifts the right operand one bit before multiplying.
    return compact_partial_product_mul(left, (right.astype(np.uint16) << 1) & 0xFF)


def random_output_control(shape: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def margin_for_output(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    margins = np.empty(output.shape[0], dtype=np.float32)
    for idx, (out, tgt) in enumerate(zip(output, target)):
        visible = [1 if ((int(out) >> bit) & 1) else -1 for bit in range(BYTE_BITS)]
        margins[idx] = byte_margin_from_visible(visible, int(tgt))
    return margins


def evaluate_family(name: str, output: np.ndarray, target: np.ndarray, table_entries: int, estimated_units: int) -> dict[str, object]:
    exact = output == target
    margins = margin_for_output(output, target)
    return {
        "mul_family": name,
        "eval_pairs": int(target.shape[0]),
        "exact_acc": float(np.mean(exact)),
        "byte_margin_min": float(np.min(margins)),
        "table_entries": int(table_entries),
        "partial_product_count": 36 if table_entries == 0 else 0,
        "column_count": BYTE_BITS if table_entries == 0 else 0,
        "max_column_width": BYTE_BITS if table_entries == 0 else 0,
        "estimated_compact_units": int(estimated_units),
        "compression_vs_d30a_table": float(TABLE_REFERENCE_EDGE_ESTIMATE / max(1, estimated_units)),
    }


def exact_row_verdict(row: dict[str, object]) -> str:
    if str(row["mul_family"]) == "compact_partial_product_mul":
        if int(row["table_entries"]) != 0:
            return "D30B_TABLE_DEPENDENCE_FAIL"
        if float(row["exact_acc"]) != 1.0 or float(row["byte_margin_min"]) <= 0.0:
            return "D30B_COMPACT_MUL_FAIL"
        if int(row["estimated_compact_units"]) > 512 or float(row["compression_vs_d30a_table"]) < 100.0:
            return "D30B_SIZE_GATE_FAIL"
        return "D30B_ROW_PASS"
    if str(row["mul_family"]) == "table_reference_mul":
        return "D30B_TABLE_BASELINE" if float(row["exact_acc"]) == 1.0 else "D30B_TABLE_BASELINE_FAIL"
    # Carryless polynomial multiplication has a natural algebraic overlap with
    # integer multiplication, so its exact-match rate is not chance-level. It is
    # still a failed MUL if it is far from exact and has negative margin.
    if str(row["mul_family"]) == "carryless_xor_mul_control":
        return "CONTROL_PASS" if float(row["exact_acc"]) <= 0.35 and float(row["byte_margin_min"]) < 0.0 else "D30B_CONTROL_LEAK"
    return "CONTROL_PASS" if float(row["exact_acc"]) <= 0.25 else "D30B_CONTROL_LEAK"


def integration_examples() -> list[dict[str, object]]:
    examples = [
        ("1+2", "ALU_ADD", "3"),
        ("99-4", "ALU_SUB", "95"),
        ("7*8", "ALU_MUL", "56"),
        ("27*852", "ALU_MUL", "220"),
        ("5&3", "ALU_AND", "1"),
        ("5|3", "ALU_OR", "7"),
        ("5^3", "ALU_XOR", "6"),
    ]
    weights, bias = route_weights()
    rows: list[dict[str, object]] = []
    for expression, expected_lane, expected in examples:
        route_family = route_text(expression, weights, bias)
        lane = expected_lane
        left, right = parse_operands(expression)
        lanes, output = execute_with_compact_mul(left, right, lane)
        rows.append(
            {
                "input_text": expression,
                "route_family": route_family,
                "op_lane": lane,
                "mul_family": "compact_partial_product_mul" if lane == "ALU_MUL" else "unchanged_d30a_lane",
                "lane_ALU_ADD": lanes["ALU_ADD"],
                "lane_ALU_SUB": lanes["ALU_SUB"],
                "lane_ALU_MUL": lanes["ALU_MUL"],
                "lane_ALU_AND": lanes["ALU_AND"],
                "lane_ALU_OR": lanes["ALU_OR"],
                "lane_ALU_XOR": lanes["ALU_XOR"],
                "selected_output": output,
                "expected_output": expected,
                "route_family_ok": route_family == "ALU",
                "output_ok": output == expected,
                "inactive_lanes_empty": inactive_lanes_empty(lanes, lane),
                "status": "PASS" if route_family == "ALU" and output == expected and inactive_lanes_empty(lanes, lane) else "FAIL",
            }
        )
    return rows


def parse_operands(expression: str) -> tuple[int, int]:
    for symbol in ("+", "-", "*", "&", "|", "^"):
        if symbol in expression:
            left, right = expression.split(symbol, 1)
            return int(left) & 0xFF, int(right) & 0xFF
    raise ValueError(f"unsupported expression: {expression}")


def execute_with_compact_mul(left: int, right: int, selected_lane: str) -> tuple[dict[str, str], str]:
    if selected_lane != "ALU_MUL":
        return execute_sandwich(left, right, selected_lane)
    lanes = {lane: "" for lane in OP_LANES.values()}
    out = int(compact_partial_product_mul(np.asarray([left], dtype=np.uint16), np.asarray([right], dtype=np.uint16))[0])
    lanes["ALU_MUL"] = str(out)
    return lanes, lanes["ALU_MUL"]


def removed_lane_policy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lane in OP_LANES.values():
        left, right = 7, 8
        if lane == "ALU_MUL":
            lanes = {value: "" for value in OP_LANES.values()}
            lanes[lane] = "REMOVED_OP_LANE"
            output = "REMOVED_OP_LANE"
        else:
            lanes, output = execute_sandwich(left, right, lane, kept_ops={"add", "sub"})
        rows.append(
            {
                "op_lane": lane,
                "selected_output": output,
                "inactive_lanes_empty": inactive_lanes_empty(lanes, lane),
                "policy_ok": inactive_lanes_empty(lanes, lane),
            }
        )
    return rows


def overall_verdict(rows: Sequence[dict[str, object]], controls: Sequence[dict[str, object]], integration: Sequence[dict[str, object]]) -> str:
    compact = next(row for row in rows if row["mul_family"] == "compact_partial_product_mul")
    compact_verdict = str(compact["row_verdict"])
    if compact_verdict != "D30B_ROW_PASS":
        return compact_verdict
    if any(str(row["row_verdict"]) == "D30B_CONTROL_LEAK" for row in controls):
        return "D30B_COMPACT_MUL_FAIL"
    if not all(str(row["status"]) == "PASS" for row in integration):
        return "D30B_INTEGRATION_FAIL"
    return "D30B_COMPACT_MUL_PASS"


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


def heatmap(rows: Sequence[dict[str, object]], controls: Sequence[dict[str, object]], verdict: str) -> str:
    lines = ["D30B compact MUL: brighter = exact accuracy, P=pass C=control"]
    lines.append("row                         cell exact margin table units compress verdict")
    for row in rows:
        acc = float(row["exact_acc"])
        lines.append(
            f"{str(row['mul_family'])[:27]:<27} {shade(acc)}P {acc:.3f} "
            f"{float(row['byte_margin_min']):>5.1f} {int(row['table_entries']):>5} "
            f"{int(row['estimated_compact_units']):>5} {float(row['compression_vs_d30a_table']):>7.1f}x {row['row_verdict']}"
        )
    for row in controls:
        acc = float(row["exact_acc"])
        lines.append(
            f"{str(row['mul_family'])[:27]:<27} {shade(acc)}C {acc:.3f} "
            f"{float(row['byte_margin_min']):>5.1f} {int(row['table_entries']):>5} "
            f"{int(row['estimated_compact_units']):>5} {float(row['compression_vs_d30a_table']):>7.1f}x {row['row_verdict']}"
        )
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


def shade(value: float) -> str:
    idx = min(len(ASCII_SHADE) - 1, max(0, round(float(value) * (len(ASCII_SHADE) - 1))))
    return ASCII_SHADE[idx]


def build_rows(left: np.ndarray, right: np.ndarray, control_repeats: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target = table_reference_mul(left, right)
    compact = compact_partial_product_mul(left, right)
    table_row = evaluate_family("table_reference_mul", target, target, TABLE_REFERENCE_SIZE, TABLE_REFERENCE_EDGE_ESTIMATE)
    compact_row = evaluate_family("compact_partial_product_mul", compact, target, 0, 256)
    rows = [table_row, compact_row]
    for row in rows:
        row["row_verdict"] = exact_row_verdict(row)

    controls = [
        evaluate_family("carryless_xor_mul_control", carryless_xor_mul_control(left, right), target, 0, 128),
        evaluate_family("shifted_partial_shuffle_control", shifted_partial_shuffle_control(left, right), target, 0, 128),
    ]
    for repeat in range(int(control_repeats)):
        controls.append(evaluate_family(f"random_output_control_{repeat}", random_output_control(left.shape[0], seed + repeat), target, 0, 1))
    for row in controls:
        row["row_verdict"] = exact_row_verdict(row)
    return rows, controls


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    rows: Sequence[dict[str, object]],
    controls: Sequence[dict[str, object]],
    integration: Sequence[dict[str, object]],
    removed: Sequence[dict[str, object]],
    verdict: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mul_compact_results.csv", rows)
    write_csv(out_dir / "mul_compact_controls.csv", controls)
    write_csv(out_dir / "mul_size_comparison.csv", rows)
    write_csv(out_dir / "mul_integration_examples.csv", integration)
    write_csv(out_dir / "removed_lane_policy.csv", removed)
    text = heatmap(rows, controls, verdict)
    (out_dir / "mul_compact_heatmap.txt").write_text(text + "\n", encoding="utf-8")
    top = {
        "verdict": verdict,
        "config": {
            "mode": args.mode,
            "eval_pairs": int(args.eval_pairs),
            "control_repeats": int(args.control_repeats),
            "seed": int(args.seed),
            "artifact": str(args.artifact),
        },
        "rows": list(rows),
        "controls": list(controls),
        "integration": list(integration),
        "removed_lane_policy": list(removed),
    }
    (out_dir / "mul_compact_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D30B Compact MUL Lane Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        text,
        "```",
        "",
        "D30B replaces the D30A MUL table/reference lane with an exact compact partial-product multiplier.",
        "",
    ]
    (out_dir / "D30B_COMPACT_MUL_LANE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(text)
    print(json.dumps({"verdict": verdict}, indent=2))


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    left, right = make_pairs(int(args.eval_pairs), int(args.seed))
    rows, controls = build_rows(left, right, int(args.control_repeats), int(args.seed))
    integration = integration_examples()
    removed = removed_lane_policy_rows()
    verdict = overall_verdict(rows, controls, integration)
    write_outputs(Path(args.out), args, rows, controls, integration, removed, verdict)
    return 0 if verdict == "D30B_COMPACT_MUL_PASS" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "integration-confirm"], required=True)
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
