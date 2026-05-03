#!/usr/bin/env python3
"""
D27 B-latent ALU/router probe.

Purpose:
    Put the old ALU/router idea onto the current AB/B64 bus.

Shape:
    A-window bytes -> B64
    B-window bytes -> B64
    opcode one-hot -> router
    selected ALU worker -> B64 output -> bytes

This is an exact reference/probe, not a learned model. It verifies that B64 can
serve as a common bus for an opcode-selected ALU worker.
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

from tools.ab_window_codec import (  # noqa: E402
    B_DIMS_PER_BYTE,
    B_WINDOW_DIMS,
    BYTE_BITS,
    WINDOW_BYTES,
    byte_margin_from_visible,
    verify_artifact,
)


DEFAULT_SEED = 20260503
DEFAULT_OPS = ("copy_a", "not_a", "and", "or", "xor", "add_mod", "sub_mod", "gt_mask", "eq_mask")
ASCII_SHADE = " .:-=+*#%@"


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def make_eval_pairs(eval_windows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    a_rows: list[list[int]] = []
    b_rows: list[list[int]] = []
    zero = [0] * WINDOW_BYTES
    for pos in range(WINDOW_BYTES):
        for av in range(256):
            row_a = zero.copy()
            row_b = zero.copy()
            row_a[pos] = av
            row_b[pos] = 255 - av
            a_rows.append(row_a)
            b_rows.append(row_b)
    adversarial = [0x00, 0x01, 0x0F, 0x10, 0x7F, 0x80, 0xFE, 0xFF, 0x55, 0xAA]
    for av in adversarial:
        for bv in adversarial:
            a_rows.append([av] * WINDOW_BYTES)
            b_rows.append([bv] * WINDOW_BYTES)
    while len(a_rows) < eval_windows:
        a_rows.append([rng.randrange(256) for _ in range(WINDOW_BYTES)])
        b_rows.append([rng.randrange(256) for _ in range(WINDOW_BYTES)])
    return np.asarray(a_rows[:eval_windows], dtype=np.uint8), np.asarray(b_rows[:eval_windows], dtype=np.uint8)


def encode_b64(windows: np.ndarray) -> np.ndarray:
    bits = ((windows[:, :, None].astype(np.uint16) >> np.arange(BYTE_BITS, dtype=np.uint16)) & 1).astype(np.int8)
    return np.where(bits.reshape(windows.shape[0], B_WINDOW_DIMS) > 0, 1, -1).astype(np.int8)


def decode_b64(latents: np.ndarray) -> np.ndarray:
    bits = (latents.reshape(latents.shape[0], WINDOW_BYTES, BYTE_BITS) >= 0).astype(np.uint16)
    powers = (1 << np.arange(BYTE_BITS, dtype=np.uint16)).reshape(1, 1, BYTE_BITS)
    return np.sum(bits * powers, axis=2).astype(np.uint8)


def target_for_op(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == "copy_a":
        return a.copy()
    if op == "not_a":
        return np.bitwise_xor(a, 0xFF).astype(np.uint8)
    if op == "and":
        return np.bitwise_and(a, b).astype(np.uint8)
    if op == "or":
        return np.bitwise_or(a, b).astype(np.uint8)
    if op == "xor":
        return np.bitwise_xor(a, b).astype(np.uint8)
    if op == "add_mod":
        return ((a.astype(np.uint16) + b.astype(np.uint16)) & 0xFF).astype(np.uint8)
    if op == "sub_mod":
        return ((a.astype(np.int16) - b.astype(np.int16)) & 0xFF).astype(np.uint8)
    if op == "gt_mask":
        return np.where(a > b, 0xFF, 0x00).astype(np.uint8)
    if op == "eq_mask":
        return np.where(a == b, 0xFF, 0x00).astype(np.uint8)
    raise ValueError(f"unsupported op: {op}")


def margin_matrix(latents: np.ndarray, targets: np.ndarray) -> np.ndarray:
    visible = latents.reshape(latents.shape[0], WINDOW_BYTES, B_DIMS_PER_BYTE)
    target_bits = ((targets[:, :, None].astype(np.uint16) >> np.arange(BYTE_BITS, dtype=np.uint16)) & 1).astype(np.int8)
    target_signs = np.where(target_bits > 0, 1, -1).astype(np.int8)
    mismatches = np.sum(visible != target_signs, axis=2).astype(np.float32)
    return np.where(mismatches == 0, 2.0, -2.0 * mismatches).astype(np.float32)


def evaluate_output(op: str, family: str, out_latents: np.ndarray, target: np.ndarray, edge_count: int, state_dim: int) -> dict[str, object]:
    decoded = decode_b64(out_latents)
    target_latents = encode_b64(target)
    margins = margin_matrix(out_latents, target)
    exact = np.all(decoded == target, axis=1)
    byte_exact = decoded == target
    bit_exact = encode_b64(decoded) == target_latents
    row = {
        "op": op,
        "family": family,
        "eval_count": int(target.shape[0]),
        "window_exact_acc": float(np.mean(exact)),
        "byte_exact_acc": float(np.mean(byte_exact)),
        "bit_acc": float(np.mean(bit_exact)),
        "byte_margin_min": float(np.min(margins)),
        "edge_count": int(edge_count),
        "state_dim": int(state_dim),
        "b_output_collision_count": int(collision_count(out_latents)),
    }
    row["row_verdict"] = row_verdict(row)
    row["D27_score"] = score(row)
    return row


def collision_count(latents: np.ndarray) -> int:
    sample = latents[: min(latents.shape[0], 8192)]
    return int(sample.shape[0] - np.unique(sample, axis=0).shape[0])


def edge_estimate(op: str, op_count: int) -> int:
    # B64 A input + optional B input + opcode router/select lanes + worker estimate.
    if op in ("copy_a", "not_a"):
        worker = B_WINDOW_DIMS
        inputs = B_WINDOW_DIMS
    elif op in ("and", "or", "xor", "gt_mask", "eq_mask"):
        worker = 2 * B_WINDOW_DIMS
        inputs = 2 * B_WINDOW_DIMS
    elif op in ("add_mod", "sub_mod"):
        # Bytewise ripple-carry ALU estimate: per bit A/B/carry gates.
        worker = WINDOW_BYTES * BYTE_BITS * 5
        inputs = 2 * B_WINDOW_DIMS
    else:
        worker = 2 * B_WINDOW_DIMS
        inputs = 2 * B_WINDOW_DIMS
    return int(inputs + op_count + worker)


def row_verdict(row: dict[str, object]) -> str:
    exact = float(row["window_exact_acc"]) == 1.0 and float(row["byte_exact_acc"]) == 1.0 and float(row["bit_acc"]) == 1.0
    if str(row["family"]) == "router_selected_alu" and exact and float(row["byte_margin_min"]) > 0.0:
        return "D27_ROW_PASS"
    if str(row["family"]).startswith("random_output_control") and float(row["window_exact_acc"]) > 0.01:
        return "D27_CONTROL_LEAK"
    # Wrong opcodes can partially agree by truth-table overlap (for example OR
    # and XOR). Treat only near-complete wrong-op solving as a routing leak.
    if str(row["family"]) == "wrong_opcode_control" and float(row["window_exact_acc"]) > 0.95:
        return "D27_CONTROL_LEAK"
    return "D27_ROW_FAIL"


def score(row: dict[str, object]) -> float:
    return (
        4.0 * float(row["window_exact_acc"])
        + 1.0 * float(row["byte_exact_acc"])
        + 0.5 * float(row["bit_acc"])
        + 0.01 * float(row["byte_margin_min"])
        - 0.00001 * float(row["edge_count"])
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    ops = parse_csv(args.ops)
    unknown = sorted(set(ops) - set(DEFAULT_OPS))
    if unknown:
        raise ValueError(f"unknown op(s): {unknown}; supported: {','.join(DEFAULT_OPS)}")
    a, b = make_eval_pairs(int(args.eval_windows), int(args.seed))
    rng = np.random.default_rng(int(args.seed) + 7001)
    rows: list[dict[str, object]] = []
    for op_idx, op in enumerate(ops):
        target = target_for_op(op, a, b)
        out_latents = encode_b64(target)
        edge_count = edge_estimate(op, len(ops))
        rows.append(evaluate_output(op, "router_selected_alu", out_latents, target, edge_count, 0))

        wrong_op = ops[(op_idx + 1) % len(ops)]
        rows.append(
            evaluate_output(
                op,
                "wrong_opcode_control",
                encode_b64(target_for_op(wrong_op, a, b)),
                target,
                edge_count,
                0,
            )
        )
        rows.append(
            evaluate_output(
                op,
                "operand_swap_control",
                encode_b64(target_for_op(op, b, a)),
                target,
                edge_count,
                0,
            )
        )
        for repeat in range(int(args.control_repeats)):
            random_latents = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=out_latents.shape).astype(np.int8)
            rows.append(evaluate_output(op, f"random_output_control_{repeat}", random_latents, target, edge_count, 0))
    return rows


def overall_verdict(rows: Sequence[dict[str, object]], ops: Sequence[str]) -> str:
    for op in ops:
        primary = [row for row in rows if row["op"] == op and row["family"] == "router_selected_alu"]
        if not primary or str(primary[0]["row_verdict"]) != "D27_ROW_PASS":
            return "D27_B64_ALU_FAIL"
    hard = [
        row
        for row in rows
        if str(row["family"]).startswith(("wrong_opcode", "random_output")) and str(row["row_verdict"]) == "D27_CONTROL_LEAK"
    ]
    if hard:
        return "D27_ROUTER_CONTROL_LEAK"
    return "D27_B64_ALU_ROUTER_PASS"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(rows: Sequence[dict[str, object]]) -> str:
    lines = ["D27 B64 ALU/router heatmap: P=pass C=control F=fail"]
    lines.append("op        family                  cell exact byte bit margin verdict")
    values = [float(row["D27_score"]) for row in rows]
    lo = min(values)
    hi = max(values)
    for row in sorted(rows, key=lambda item: (str(item["op"]), str(item["family"]))):
        scaled = 0 if hi <= lo else round((float(row["D27_score"]) - lo) / (hi - lo) * (len(ASCII_SHADE) - 1))
        marker = "P" if row["row_verdict"] == "D27_ROW_PASS" else "C" if "control" in str(row["family"]) else "F"
        lines.append(
            f"{str(row['op'])[:9]:<9} {str(row['family'])[:23]:<23} {ASCII_SHADE[int(scaled)]}{marker} "
            f"{float(row['window_exact_acc']):.3f} {float(row['byte_exact_acc']):.3f} "
            f"{float(row['bit_acc']):.3f} {float(row['byte_margin_min']):>5.1f} {row['row_verdict']}"
        )
    return "\n".join(lines)


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], config: dict[str, object]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = parse_csv(str(config["ops"]))
    verdict = overall_verdict(rows, ops)
    write_csv(out_dir / "alu_results.csv", rows)
    write_csv(out_dir / "alu_control_summary.csv", [row for row in rows if "control" in str(row["family"])])
    text = heatmap(rows)
    (out_dir / "alu_heatmap.txt").write_text(text + "\n", encoding="utf-8")
    primary = [row for row in rows if row["family"] == "router_selected_alu"]
    top = {
        "verdict": verdict,
        "config": config,
        "op_count": len(ops),
        "primary_rows": primary,
    }
    (out_dir / "alu_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D27 B-Latent ALU Router Probe",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        text,
        "```",
        "",
        "Interpretation: this is an exact B64 ALU/router reference. It proves the B64 bus can feed an opcode-selected ALU worker, not that the router has been learned.",
        "",
    ]
    (out_dir / "D27_BLATENT_ALU_ROUTER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return verdict


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    rows = build_rows(args)
    config = {
        "mode": args.mode,
        "ops": args.ops,
        "eval_windows": int(args.eval_windows),
        "control_repeats": int(args.control_repeats),
        "seed": int(args.seed),
        "artifact": str(args.artifact),
    }
    verdict = write_outputs(Path(args.out), rows, config)
    print((Path(args.out) / "alu_heatmap.txt").read_text(encoding="utf-8"))
    print(json.dumps({"verdict": verdict}, indent=2))
    return 0 if verdict == "D27_B64_ALU_ROUTER_PASS" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main"], required=True)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--eval-windows", type=int, default=65536)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
