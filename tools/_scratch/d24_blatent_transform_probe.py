#!/usr/bin/env python3
"""
D24 B-latent transformation probe.

Prototype goal:
    B64 input -> sparse transform core -> B64 output -> decoded 8-byte window

D24 verifies that the AB codec's 64D B-window latent can be used as a working
surface for exact stateless byte-window transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
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
    ABWindowCodec,
    byte_margin_from_visible,
    verify_artifact,
)


DEFAULT_SEED = 20260503
PRIMARY_TASKS = ("copy", "reverse", "rotate_left", "rotate_right")
ALL_TASKS = PRIMARY_TASKS + ("bit_not",)
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class TransformSpec:
    task: str
    family: str
    repeat_id: int
    entries: tuple[tuple[int, int, int], ...]


def parse_tasks(raw: str) -> list[str]:
    tasks = [part.strip() for part in str(raw).split(",") if part.strip()]
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; supported: {','.join(ALL_TASKS)}")
    return tasks


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def make_eval_windows(eval_windows: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    windows: list[bytes] = []
    zero = [0] * WINDOW_BYTES
    for pos in range(WINDOW_BYTES):
        for value in range(256):
            row = zero.copy()
            row[pos] = value
            windows.append(bytes(row))
    while len(windows) < eval_windows:
        windows.append(bytes(rng.randrange(256) for _ in range(WINDOW_BYTES)))
    return windows[:eval_windows]


def target_window(task: str, window: bytes) -> bytes:
    if task == "copy":
        return window
    if task == "reverse":
        return bytes(reversed(window))
    if task == "rotate_left":
        return window[1:] + window[:1]
    if task == "rotate_right":
        return window[-1:] + window[:-1]
    if task == "bit_not":
        return bytes(byte ^ 0xFF for byte in window)
    raise ValueError(f"unsupported task: {task}")


def lane(byte_idx: int, bit_idx: int) -> int:
    return byte_idx * B_DIMS_PER_BYTE + bit_idx


def source_byte_for_task(task: str, out_byte_idx: int) -> tuple[int, int]:
    """Return (source byte index, sign) for one output byte."""
    if task == "copy":
        return out_byte_idx, 1
    if task == "reverse":
        return WINDOW_BYTES - 1 - out_byte_idx, 1
    if task == "rotate_left":
        return (out_byte_idx + 1) % WINDOW_BYTES, 1
    if task == "rotate_right":
        return (out_byte_idx - 1) % WINDOW_BYTES, 1
    if task == "bit_not":
        return out_byte_idx, -1
    raise ValueError(f"unsupported task: {task}")


def oracle_entries(task: str) -> tuple[tuple[int, int, int], ...]:
    entries: list[tuple[int, int, int]] = []
    for out_byte in range(WINDOW_BYTES):
        in_byte, sign = source_byte_for_task(task, out_byte)
        for bit_idx in range(BYTE_BITS):
            entries.append((lane(out_byte, bit_idx), lane(in_byte, bit_idx), sign))
    return tuple(entries)


def random_sparse_entries(task: str, repeat_id: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    rng = random.Random(seed + 1009 * repeat_id + 17 * sum(ord(ch) for ch in task))
    entries: list[tuple[int, int, int]] = []
    outputs = list(range(B_WINDOW_DIMS))
    rng.shuffle(outputs)
    for out_idx in outputs[:B_WINDOW_DIMS]:
        entries.append((out_idx, rng.randrange(B_WINDOW_DIMS), 1 if rng.random() < 0.5 else -1))
    return tuple(sorted(entries))


def drop_entries(entries: Sequence[tuple[int, int, int]], drop_count: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    rng = random.Random(seed + 313 * drop_count)
    drop = set(rng.sample(range(len(entries)), min(drop_count, len(entries))))
    return tuple(entry for idx, entry in enumerate(entries) if idx not in drop)


def apply_transform(entries: Sequence[tuple[int, int, int]], latent: Sequence[int]) -> list[int]:
    out = [0] * B_WINDOW_DIMS
    for out_idx, in_idx, sign in entries:
        out[out_idx] += int(sign) * int(latent[in_idx])
    return [1 if value >= 0 else -1 for value in out]


def apply_transform_matrix(entries: Sequence[tuple[int, int, int]], latents: np.ndarray) -> np.ndarray:
    out = np.zeros_like(latents, dtype=np.int16)
    for out_idx, in_idx, sign in entries:
        out[:, out_idx] += int(sign) * latents[:, in_idx]
    return np.where(out >= 0, 1, -1).astype(np.int8)


def decode_b64(latent: Sequence[int]) -> bytes:
    values = []
    for byte_idx in range(WINDOW_BYTES):
        value = 0
        offset = byte_idx * B_DIMS_PER_BYTE
        for bit_idx in range(BYTE_BITS):
            if int(latent[offset + bit_idx]) >= 0:
                value |= 1 << bit_idx
        values.append(value)
    return bytes(values)


def b64_for_window(codec: ABWindowCodec, window: bytes) -> tuple[int, ...]:
    return tuple(int(value) for value in codec.encode_window_b64(window))


def position_shuffle(latent: Sequence[int]) -> tuple[int, ...]:
    chunks = [tuple(latent[idx : idx + B_DIMS_PER_BYTE]) for idx in range(0, B_WINDOW_DIMS, B_DIMS_PER_BYTE)]
    return tuple(value for chunk in (chunks[-1:] + chunks[:-1]) for value in chunk)


def margin_for_window(latent: Sequence[int], target: bytes) -> float:
    margin = float("inf")
    for byte_idx, target_byte in enumerate(target):
        offset = byte_idx * B_DIMS_PER_BYTE
        margin = min(margin, byte_margin_from_visible(latent[offset : offset + B_DIMS_PER_BYTE], target_byte))
    return float(margin)


def windows_to_matrix(windows: Sequence[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(windows), dtype=np.uint8).reshape(len(windows), WINDOW_BYTES)


def latents_to_byte_matrix(latents: np.ndarray) -> np.ndarray:
    bits = (latents.reshape(latents.shape[0], WINDOW_BYTES, BYTE_BITS) >= 0).astype(np.uint16)
    powers = (1 << np.arange(BYTE_BITS, dtype=np.uint16)).reshape(1, 1, BYTE_BITS)
    return np.sum(bits * powers, axis=2).astype(np.uint8)


def collision_count_for_outputs(out_bytes: np.ndarray, target_bytes: np.ndarray) -> int:
    used_outputs: dict[tuple[int, ...], bytes] = {}
    collision_count = 0
    for out_row, target_row in zip(out_bytes, target_bytes):
        out_key = tuple(int(value) for value in out_row)
        target_value = bytes(int(value) for value in target_row)
        previous = used_outputs.get(out_key)
        if previous is not None and previous != target_value:
            collision_count += 1
        used_outputs[out_key] = target_value
    return collision_count


def evaluate_transform(
    spec: TransformSpec,
    windows: Sequence[bytes],
    input_latents: np.ndarray,
    target_latents: np.ndarray,
) -> dict[str, object]:
    del windows
    out_latents = apply_transform_matrix(spec.entries, input_latents)
    bit_match = out_latents == target_latents
    byte_match = np.all(bit_match.reshape(bit_match.shape[0], WINDOW_BYTES, BYTE_BITS), axis=2)
    window_match = np.all(bit_match, axis=1)
    dot = np.sum(
        out_latents.reshape(out_latents.shape[0], WINDOW_BYTES, BYTE_BITS)
        * target_latents.reshape(target_latents.shape[0], WINDOW_BYTES, BYTE_BITS),
        axis=2,
    )
    margins = np.where(dot == BYTE_BITS, 2, dot - BYTE_BITS)
    shuffled = np.concatenate(
        [
            out_latents.reshape(out_latents.shape[0], WINDOW_BYTES, BYTE_BITS)[:, -1:, :],
            out_latents.reshape(out_latents.shape[0], WINDOW_BYTES, BYTE_BITS)[:, :-1, :],
        ],
        axis=1,
    ).reshape(out_latents.shape[0], B_WINDOW_DIMS)

    return {
        "task": spec.task,
        "family": spec.family,
        "repeat_id": spec.repeat_id,
        "window_exact_acc": float(np.mean(window_match)),
        "byte_exact_acc": float(np.mean(byte_match)),
        "bit_acc": float(np.mean(bit_match)),
        "byte_margin_min": float(np.min(margins)),
        "b_output_collision_count": collision_count_for_outputs(latents_to_byte_matrix(out_latents), latents_to_byte_matrix(target_latents)),
        "transform_edge_count": len(spec.entries),
        "input_output_hamming_consistency": float(np.mean(bit_match)),
        "position_shuffle_control_acc": float(np.mean(np.all(shuffled == target_latents, axis=1))),
    }


def single_edge_drop_metrics(
    task: str,
    entries: Sequence[tuple[int, int, int]],
    windows: Sequence[bytes],
    input_latents: np.ndarray,
    target_latents: np.ndarray,
) -> dict[str, float]:
    del task, windows, input_latents
    # Dropping one oracle permutation edge leaves exactly one output B lane at
    # default +1. The affected window remains exact only when the target lane
    # was already +1, so the robustness telemetry can be computed directly.
    n = len(target_latents)
    total_bits = n * B_WINDOW_DIMS
    exacts: list[float] = []
    bits: list[float] = []
    for out_idx, _in_idx, _sign in entries:
        correct_default = int(np.sum(target_latents[:, out_idx] >= 0))
        exacts.append(correct_default / n)
        bits.append((total_bits - n + correct_default) / total_bits)
    return {
        "single_edge_drop_mean_exact": sum(exacts) / len(exacts) if exacts else 0.0,
        "single_edge_drop_mean_bit": sum(bits) / len(bits) if bits else 0.0,
    }


def verdict_for_row(row: dict[str, object]) -> str:
    exact = (
        float(row["window_exact_acc"]) == 1.0
        and float(row["byte_exact_acc"]) == 1.0
        and float(row["bit_acc"]) == 1.0
        and float(row["byte_margin_min"]) > 0.0
        and int(row["b_output_collision_count"]) == 0
    )
    if str(row["family"]) == "oracle_permutation" and exact:
        return "D24_ORACLE_TRANSFORM_PASS"
    if str(row["family"]) == "random_sparse_control" and float(row["window_exact_acc"]) > 0.01:
        return "D24_RANDOM_CONTROL_LEAK"
    if exact:
        return "D24_TRANSFORM_EXACT"
    return "D24_TRANSFORM_FAIL"


def score_row(row: dict[str, object]) -> float:
    return (
        3.0 * float(row["window_exact_acc"])
        + 1.0 * float(row["byte_exact_acc"])
        + 0.5 * float(row["bit_acc"])
        + 0.25 * float(row["input_output_hamming_consistency"])
        - 0.001 * float(row["b_output_collision_count"])
        - 0.0001 * float(row["transform_edge_count"])
    )


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
    values = [float(row["D24_score"]) for row in rows] or [0.0]
    lo = min(values)
    hi = max(values)
    lines = ["D24 B-latent transform heatmap: brighter = D24_score, P=pass C=control F=fail"]
    lines.append("task         family                 cell exact bit  margin ctrl  drop_exact verdict")
    for row in sorted(rows, key=lambda item: (str(item["task"]), str(item["family"]), int(item["repeat_id"]))):
        scaled = 0 if hi <= lo else int(round((float(row["D24_score"]) - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
        scaled = max(0, min(len(ASCII_SHADE) - 1, scaled))
        marker = "P" if str(row["row_verdict"]).endswith("_PASS") else "C" if str(row["family"]) == "random_sparse_control" else "F"
        lines.append(
            f"{str(row['task'])[:12]:<12} {str(row['family'])[:22]:<22} {ASCII_SHADE[scaled]}{marker} "
            f"{float(row['window_exact_acc']):.3f} {float(row['bit_acc']):.3f} "
            f"{float(row['byte_margin_min']):>6.2f} {float(row['random_sparse_control_acc']):.3f} "
            f"{float(row['single_edge_drop_mean_exact']):.3f} {row['row_verdict']}"
        )
    return "\n".join(lines)


def overall_verdict(rows: Sequence[dict[str, object]], tasks: Sequence[str]) -> str:
    primary = [task for task in PRIMARY_TASKS if task in tasks]
    if not primary:
        primary = list(tasks)
    for task in primary:
        oracle = [row for row in rows if row["task"] == task and row["family"] == "oracle_permutation"]
        if not oracle:
            return "D24_TRANSFORM_FAIL"
        best = max(oracle, key=lambda row: float(row["D24_score"]))
        if (
            float(best["window_exact_acc"]) != 1.0
            or float(best["byte_exact_acc"]) != 1.0
            or float(best["bit_acc"]) != 1.0
            or float(best["byte_margin_min"]) <= 0.0
            or int(best["b_output_collision_count"]) != 0
        ):
            return "D24_TRANSFORM_FAIL"
        if float(best["random_sparse_control_acc"]) > 0.01:
            return "D24_RANDOM_CONTROL_LEAK"
    return "D24_BLATENT_TRANSFORM_PASS"


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[bytes]]:
    checked_artifact(Path(args.artifact))
    codec = ABWindowCodec()
    tasks = parse_tasks(args.tasks)
    windows = make_eval_windows(int(args.eval_windows), int(args.seed))
    input_latents = np.asarray([b64_for_window(codec, window) for window in windows], dtype=np.int8)
    target_latents_by_task = {
        task: np.asarray([b64_for_window(codec, target_window(task, window)) for window in windows], dtype=np.int8)
        for task in tasks
    }
    rows: list[dict[str, object]] = []

    for task in tasks:
        oracle = oracle_entries(task)
        specs: list[TransformSpec] = [TransformSpec(task, "oracle_permutation", 0, oracle)]
        specs.extend(
            TransformSpec(task, "random_sparse_control", repeat_id, random_sparse_entries(task, repeat_id, int(args.seed)))
            for repeat_id in range(int(args.control_repeats))
        )
        for drop_count in (1, 2, 4):
            specs.append(
                TransformSpec(
                    task,
                    f"noisy_drop_control_{drop_count}",
                    drop_count,
                    drop_entries(oracle, drop_count, int(args.seed) + 41 * sum(ord(ch) for ch in task)),
                )
            )

        oracle_drop = single_edge_drop_metrics(task, oracle, windows, input_latents, target_latents_by_task[task])
        task_rows: list[dict[str, object]] = []
        for spec in specs:
            row = evaluate_transform(spec, windows, input_latents, target_latents_by_task[task])
            row.update(oracle_drop if spec.family == "oracle_permutation" else {"single_edge_drop_mean_exact": 0.0, "single_edge_drop_mean_bit": 0.0})
            task_rows.append(row)

        random_max = max(
            (float(row["window_exact_acc"]) for row in task_rows if row["family"] == "random_sparse_control"),
            default=0.0,
        )
        for row in task_rows:
            row["random_sparse_control_acc"] = random_max
            row["row_verdict"] = verdict_for_row(row)
            row["D24_score"] = score_row(row)
            rows.append(row)

    return rows, windows


def write_transform_artifacts(out_dir: Path, tasks: Sequence[str]) -> None:
    artifact = {
        "version": "d24_blatent_transform_artifacts_v1",
        "b_width": B_WINDOW_DIMS,
        "tasks": {
            task: {
                "family": "oracle_permutation",
                "entries": [
                    {"out_idx": out_idx, "in_idx": in_idx, "sign": sign}
                    for out_idx, in_idx, sign in oracle_entries(task)
                ],
            }
            for task in tasks
        },
    }
    (out_dir / "transform_artifacts.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], mode: str, config: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["D24_score"]), reverse=True)
    write_csv(out_dir / "transform_candidates.csv", sorted_rows)

    control_rows = []
    for task in sorted(set(str(row["task"]) for row in rows)):
        task_rows = [row for row in rows if row["task"] == task]
        control_rows.append(
            {
                "task": task,
                "oracle_exact": max(float(row["window_exact_acc"]) for row in task_rows if row["family"] == "oracle_permutation"),
                "random_sparse_control_acc": max(float(row["window_exact_acc"]) for row in task_rows if row["family"] == "random_sparse_control"),
                "position_shuffle_control_acc": max(float(row["position_shuffle_control_acc"]) for row in task_rows if row["family"] == "oracle_permutation"),
                "single_edge_drop_mean_exact": max(float(row["single_edge_drop_mean_exact"]) for row in task_rows),
            }
        )
    write_csv(out_dir / "transform_control_summary.csv", control_rows)

    heatmap = make_heatmap(sorted_rows)
    (out_dir / "transform_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    top_verdict = overall_verdict(rows, parse_tasks(str(config["tasks"])))
    payload = {
        "verdict": top_verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "best_candidate": sorted_rows[0] if sorted_rows else None,
        "primary_task_rows": [
            row
            for row in sorted_rows
            if row["family"] == "oracle_permutation" and row["task"] in PRIMARY_TASKS
        ],
    }
    (out_dir / "transform_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# D24 B-Latent Transform Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{top_verdict}`",
        "",
        "## Heatmap",
        "",
        "```text",
        heatmap,
        "```",
        "",
        "## Interpretation",
        "",
        "- Oracle transforms are sparse signed/permutation transforms over B64.",
        "- Random sparse controls use the same 64-edge budget and must fail.",
        "- Single-edge drop is reported as robustness telemetry, not as the primary pass gate.",
        "",
    ]
    (out_dir / "D24_BLATENT_TRANSFORM_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    rows, _windows = build_rows(args)
    config = {
        "mode": args.mode,
        "tasks": args.tasks,
        "eval_windows": int(args.eval_windows),
        "control_repeats": int(args.control_repeats),
        "seed": int(args.seed),
        "artifact": str(args.artifact),
    }
    out_dir = Path(args.out)
    write_outputs(out_dir, rows, str(args.mode), config)
    if str(args.mode) == "crystallize":
        write_transform_artifacts(out_dir, parse_tasks(args.tasks))
    top = json.loads((out_dir / "transform_top.json").read_text(encoding="utf-8"))
    print((out_dir / "transform_heatmap.txt").read_text(encoding="utf-8"))
    print(json.dumps({"verdict": top["verdict"], "best": top["best_candidate"]}, indent=2))
    return 0 if top["verdict"] in ("D24_BLATENT_TRANSFORM_PASS", "D24_TRANSFORM_PARTIAL_PASS") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "crystallize"], required=True)
    parser.add_argument("--tasks", default="copy,reverse,rotate_left,rotate_right")
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
