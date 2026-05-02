#!/usr/bin/env python3
"""
D22 byte-to-word embedder probe.

Prototype goal:
    8 bytes -> 8 parallel D21A A-block codes -> 128D window code -> 8 bytes

This is not a semantic word model. It is a fixed-size byte-window adapter that
tests whether the D21 A-block can be composed into a clean, packable
word-ish embedding surface.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from d21a_reciprocal_byte_ablock import ReciprocalABlock, all_visible_patterns, evaluate_ablock, redundant_copy_entries, robustness_metrics


DEFAULT_SEED = 20260502
VISIBLE_DIM = 8
BYTE_CODE_DIM = 16
DEFAULT_WINDOW_BYTES = 8
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class PackSpec:
    name: str
    word_width: int
    per_byte_dims: int


def byte_block() -> ReciprocalABlock:
    return ReciprocalABlock.from_entries(VISIBLE_DIM, BYTE_CODE_DIM, redundant_copy_entries(VISIBLE_DIM, BYTE_CODE_DIM))


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def make_window_batch(*, window_bytes: int, eval_windows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_count = max(0, int(eval_windows) - window_bytes * 256)
    windows = []
    base = np.zeros(window_bytes, dtype=np.int32)
    for pos in range(window_bytes):
        for value in range(256):
            row = base.copy()
            row[pos] = value
            windows.append(row)
    if random_count > 0:
        random_windows = rng.integers(0, 256, size=(random_count, window_bytes), dtype=np.int32)
        windows.extend(random_windows)
    return np.asarray(windows, dtype=np.int32)


def encode_windows(windows: np.ndarray, block: ReciprocalABlock, patterns: np.ndarray) -> np.ndarray:
    flat_patterns = patterns[windows.reshape(-1)]
    codes = block.encode_patterns(flat_patterns).reshape(windows.shape[0], windows.shape[1], BYTE_CODE_DIM)
    return codes.reshape(windows.shape[0], windows.shape[1] * BYTE_CODE_DIM)


def pack_code(full_code: np.ndarray, *, window_bytes: int, per_byte_dims: int) -> np.ndarray:
    code = full_code.reshape(full_code.shape[0], window_bytes, BYTE_CODE_DIM)
    return code[:, :, :per_byte_dims].reshape(full_code.shape[0], window_bytes * per_byte_dims)


def decode_pack(packed: np.ndarray, *, window_bytes: int, per_byte_dims: int) -> np.ndarray:
    packed_3d = packed.reshape(packed.shape[0], window_bytes, per_byte_dims)
    decoded = np.zeros((packed.shape[0], window_bytes, VISIBLE_DIM), dtype=np.float32)
    keep = min(per_byte_dims, VISIBLE_DIM)
    decoded[:, :, :keep] += packed_3d[:, :, :keep]
    if per_byte_dims > VISIBLE_DIM:
        extra = packed_3d[:, :, VISIBLE_DIM:per_byte_dims]
        for idx in range(extra.shape[2]):
            decoded[:, :, idx % VISIBLE_DIM] += extra[:, :, idx]
    return decoded


def byte_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    rows = np.arange(flat_targets.shape[0])
    target_logits = flat_logits[rows, flat_targets]
    masked = flat_logits.copy()
    masked[rows, flat_targets] = -np.inf
    return target_logits - np.max(masked, axis=1)


def evaluate_pack(spec: PackSpec, *, windows: np.ndarray, full_code: np.ndarray, patterns: np.ndarray) -> dict[str, object]:
    packed = pack_code(full_code, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    decoded = decode_pack(packed, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    logits = decoded @ patterns.T
    pred = np.argmax(logits, axis=2)
    pred_bits = np.where(decoded >= 0.0, 1.0, -1.0)
    target_bits = patterns[windows]
    margins = byte_margin(logits, windows)
    by_window: dict[tuple[int, ...], tuple[float, ...]] = {}
    collision_keys: set[tuple[float, ...]] = set()
    used_codes: dict[tuple[float, ...], tuple[int, ...]] = {}
    for window, code_row in zip(windows, packed):
        window_key = tuple(int(x) for x in window)
        code_key = tuple(np.round(code_row, 6).tolist())
        by_window.setdefault(window_key, code_key)
    for window_key, code_key in by_window.items():
        previous = used_codes.get(code_key)
        if previous is not None and previous != window_key:
            collision_keys.add(code_key)
        used_codes[code_key] = window_key
    collisions = len(collision_keys)
    return {
        "pack_name": spec.name,
        "window_bytes": windows.shape[1],
        "word_width": spec.word_width,
        "per_byte_dims": spec.per_byte_dims,
        "window_exact_acc": float(np.mean(np.all(pred == windows, axis=1))),
        "byte_exact_acc": float(np.mean(pred == windows)),
        "bit_acc": float(np.mean(pred_bits == target_bits)),
        "byte_margin_min": float(np.min(margins)),
        "byte_margin_mean": float(np.mean(margins)),
        "sample_hidden_collisions": collisions,
        "sample_unique_windows": int(len(by_window)),
        "sample_unique_codes": int(len(set(by_window.values()))),
        "int8_lut_bytes": int(256 * spec.per_byte_dims),
        "window_vector_bytes_int8": int(spec.word_width),
        "estimated_shared_decoder_bytes": int(spec.per_byte_dims * VISIBLE_DIM),
    }


def single_dim_drop_metrics(spec: PackSpec, *, windows: np.ndarray, full_code: np.ndarray, patterns: np.ndarray) -> dict[str, float]:
    packed = pack_code(full_code, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    if packed.shape[1] == 0:
        return {"single_dim_drop_mean_window_exact": 0.0, "single_dim_drop_mean_bit": 0.0, "single_dim_drop_min_byte_margin": 0.0}
    window_scores = []
    bit_scores = []
    margin_scores = []
    for dim in range(packed.shape[1]):
        dropped = packed.copy()
        dropped[:, dim] = 0.0
        decoded = decode_pack(dropped, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
        logits = decoded @ patterns.T
        pred = np.argmax(logits, axis=2)
        pred_bits = np.where(decoded >= 0.0, 1.0, -1.0)
        target_bits = patterns[windows]
        margins = byte_margin(logits, windows)
        window_scores.append(float(np.mean(np.all(pred == windows, axis=1))))
        bit_scores.append(float(np.mean(pred_bits == target_bits)))
        margin_scores.append(float(np.min(margins)))
    return {
        "single_dim_drop_mean_window_exact": float(np.mean(window_scores)),
        "single_dim_drop_mean_bit": float(np.mean(bit_scores)),
        "single_dim_drop_min_byte_margin": float(np.min(margin_scores)),
    }


def control_metrics(spec: PackSpec, *, windows: np.ndarray, full_code: np.ndarray, patterns: np.ndarray, seed: int) -> dict[str, float]:
    packed = pack_code(full_code, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    rng = np.random.default_rng(seed)
    shuffled = packed.reshape(packed.shape[0], windows.shape[1], spec.per_byte_dims).copy()
    shuffled = np.roll(shuffled, 1, axis=1).reshape(packed.shape)
    decoded_shuffle = decode_pack(shuffled, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    pred_shuffle = np.argmax(decoded_shuffle @ patterns.T, axis=2)

    random_pack = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=packed.shape)
    decoded_random = decode_pack(random_pack, window_bytes=windows.shape[1], per_byte_dims=spec.per_byte_dims)
    pred_random = np.argmax(decoded_random @ patterns.T, axis=2)
    return {
        "position_shuffle_window_exact_acc": float(np.mean(np.all(pred_shuffle == windows, axis=1))),
        "position_shuffle_byte_exact_acc": float(np.mean(pred_shuffle == windows)),
        "random_code_window_exact_acc": float(np.mean(np.all(pred_random == windows, axis=1))),
        "random_code_byte_exact_acc": float(np.mean(pred_random == windows)),
    }


def verdict(row: dict[str, object]) -> str:
    if float(row["window_exact_acc"]) == 1.0 and float(row["byte_exact_acc"]) == 1.0 and float(row["byte_margin_min"]) > 0.0:
        if int(row["word_width"]) <= 64:
            return "D22_COMPACT_WORD_EMBEDDER_PASS"
        return "D22_WORD_EMBEDDER_PASS"
    if float(row["window_exact_acc"]) == 1.0:
        return "D22_WORD_EMBEDDER_WEAK"
    return "D22_WORD_EMBEDDER_FAIL"


def score(row: dict[str, object]) -> float:
    return (
        4.0 * float(row["window_exact_acc"])
        + 1.0 * float(row["byte_exact_acc"])
        + 0.5 * float(row["bit_acc"])
        + 0.02 * float(row["byte_margin_min"])
        + 0.25 * float(row["single_dim_drop_mean_bit"])
        - 0.01 * float(row["position_shuffle_byte_exact_acc"])
        - 0.0001 * float(row["word_width"])
    )


def pack_specs(window_bytes: int, widths: Sequence[int]) -> list[PackSpec]:
    specs = []
    for width in widths:
        if int(width) % int(window_bytes) != 0:
            continue
        per_byte = int(width) // int(window_bytes)
        if per_byte <= 0 or per_byte > BYTE_CODE_DIM:
            continue
        if width == window_bytes * BYTE_CODE_DIM:
            name = "full_128_ablock" if window_bytes == 8 else f"full_{width}_ablock"
        elif per_byte == VISIBLE_DIM:
            name = "compact_64_visible_bits" if window_bytes == 8 else f"compact_{width}_visible_bits"
        elif per_byte < VISIBLE_DIM:
            name = f"undercomplete_{width}"
        else:
            name = f"redundant_{width}"
        specs.append(PackSpec(name, int(width), per_byte))
    return specs


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_heatmap(rows: Sequence[dict[str, object]]) -> str:
    values = [float(row["D22_score"]) for row in rows] or [0.0]
    lo = min(values)
    hi = max(values)
    lines = ["D22 pack heatmap: brighter = score, P=pass F=fail"]
    lines.append("width   cell  exact  margin  drop_bit  verdict")
    for row in sorted(rows, key=lambda item: int(item["word_width"])):
        scaled = 0 if hi <= lo else int(round((float(row["D22_score"]) - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
        scaled = max(0, min(len(ASCII_SHADE) - 1, scaled))
        marker = "P" if str(row["verdict"]).endswith("_PASS") else "F"
        lines.append(
            f"{int(row['word_width']):>5}   {ASCII_SHADE[scaled]}{marker}    "
            f"{float(row['window_exact_acc']):.3f}  {float(row['byte_margin_min']):>6.2f}  "
            f"{float(row['single_dim_drop_mean_bit']):.3f}   {row['verdict']}"
        )
    return "\n".join(lines)


def run_eval(args: argparse.Namespace, mode: str) -> int:
    patterns = all_visible_patterns(VISIBLE_DIM)
    block = byte_block()
    base = {**evaluate_ablock(block, patterns), **robustness_metrics(block, patterns)}
    if float(base["exact_byte_acc"]) != 1.0:
        raise RuntimeError("D21A byte block failed exact reconstruction")
    windows = make_window_batch(window_bytes=int(args.window_bytes), eval_windows=int(args.eval_windows), seed=int(args.seed))
    full_code = encode_windows(windows, block, patterns)
    rows = []
    for spec in pack_specs(int(args.window_bytes), parse_int_list(args.widths)):
        row = evaluate_pack(spec, windows=windows, full_code=full_code, patterns=patterns)
        row.update(single_dim_drop_metrics(spec, windows=windows, full_code=full_code, patterns=patterns))
        row.update(control_metrics(spec, windows=windows, full_code=full_code, patterns=patterns, seed=int(args.seed) + spec.word_width))
        row["verdict"] = verdict(row)
        row["D22_score"] = score(row)
        rows.append(row)
    rows = sorted(rows, key=lambda row: float(row["D22_score"]), reverse=True)
    heatmap = make_heatmap(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "d22_pack_candidates.csv", rows)
    (out_dir / "d22_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    pass_rows = [row for row in rows if str(row["verdict"]).endswith("_PASS")]
    compact_rows = [row for row in pass_rows if int(row["word_width"]) <= 64]
    top_payload = {
        "verdict": "D22_COMPACT_WORD_EMBEDDER_PASS" if compact_rows else "D22_WORD_EMBEDDER_PASS" if pass_rows else "D22_WORD_EMBEDDER_FAIL",
        "mode": mode,
        "config": {
            "window_bytes": int(args.window_bytes),
            "eval_windows": int(args.eval_windows),
            "widths": parse_int_list(args.widths),
            "seed": int(args.seed),
        },
        "base_ablock": base,
        "candidate_count": len(rows),
        "best_candidate": rows[0] if rows else None,
        "best_compact_candidate": compact_rows[0] if compact_rows else None,
    }
    (out_dir / "d22_top.json").write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
    report = [
        "# D22 Byte-to-Word Embedder Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{top_payload['verdict']}`",
        "",
        "## Heatmap",
        "",
        "```text",
        heatmap,
        "```",
        "",
    ]
    if rows:
        best = rows[0]
        report.extend(
            [
                "## Best Candidate",
                "",
                f"- pack: `{best['pack_name']}`",
                f"- width: `{best['word_width']}`",
                f"- window_exact_acc: `{float(best['window_exact_acc']):.6f}`",
                f"- byte_margin_min: `{float(best['byte_margin_min']):.6f}`",
                f"- single_dim_drop_mean_bit: `{float(best['single_dim_drop_mean_bit']):.6f}`",
                f"- int8_lut_bytes: `{best['int8_lut_bytes']}`",
                f"- verdict: `{best['verdict']}`",
                "",
            ]
        )
    (out_dir / "D22_BYTE_WORD_EMBEDDER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(heatmap)
    print(json.dumps({"verdict": top_payload["verdict"], "best": top_payload["best_candidate"]}, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline-check", "pack-probe", "confirm"], required=True)
    parser.add_argument("--window-bytes", type=int, default=DEFAULT_WINDOW_BYTES)
    parser.add_argument("--widths", default="32,64,96,128")
    parser.add_argument("--eval-windows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_eval(args, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
