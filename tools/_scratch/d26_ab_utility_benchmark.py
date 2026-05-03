#!/usr/bin/env python3
"""
D26 AB utility benchmark.

Question:
    Is the A/B abstraction useful as a core interface, or is it only a
    roundtrip codec?

The benchmark compares the same exact tasks on four surfaces:
    raw64        direct 8-byte signed bit lanes
    a128         D21/D22 A-window lanes
    b64          AB codec B latent
    b64_composed D24 transform + D25 memory composed over B64
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ab_window_codec import (  # noqa: E402
    A_DIMS_PER_BYTE,
    A_WINDOW_DIMS,
    B_DIMS_PER_BYTE,
    B_WINDOW_DIMS,
    BYTE_BITS,
    WINDOW_BYTES,
    ABWindowCodec,
    byte_margin_from_visible,
    verify_artifact,
)


DEFAULT_SEED = 20260503
SUPPORTED_SURFACES = ("raw64", "a128", "b64", "b64_composed")
SUPPORTED_TASKS = ("stateless_transform", "slot_memory", "memory_transform")
TRANSFORMS = ("copy", "reverse", "rotate_left", "rotate_right")


@dataclass(frozen=True)
class Surface:
    name: str
    width: int
    dims_per_byte: int
    output_relevant_lanes: int
    composed_modules: int


SURFACE_SPECS = {
    "raw64": Surface("raw64", B_WINDOW_DIMS, B_DIMS_PER_BYTE, B_WINDOW_DIMS, 0),
    "a128": Surface("a128", A_WINDOW_DIMS, A_DIMS_PER_BYTE, B_WINDOW_DIMS, 0),
    "b64": Surface("b64", B_WINDOW_DIMS, B_DIMS_PER_BYTE, B_WINDOW_DIMS, 1),
    "b64_composed": Surface("b64_composed", B_WINDOW_DIMS, B_DIMS_PER_BYTE, B_WINDOW_DIMS, 2),
}


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def validate_choices(values: Sequence[str], supported: Sequence[str], label: str) -> list[str]:
    unknown = sorted(set(values) - set(supported))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; supported: {','.join(supported)}")
    return list(values)


def make_eval_windows(eval_windows: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    zero = [0] * WINDOW_BYTES
    for pos in range(WINDOW_BYTES):
        for value in range(256):
            row = zero.copy()
            row[pos] = value
            rows.append(row)
    while len(rows) < eval_windows:
        rows.append([rng.randrange(256) for _ in range(WINDOW_BYTES)])
    return np.asarray(rows[:eval_windows], dtype=np.uint8)


def target_windows(task: str, windows: np.ndarray) -> np.ndarray:
    if task == "copy":
        return windows.copy()
    if task == "reverse":
        return windows[:, ::-1].copy()
    if task == "rotate_left":
        return np.concatenate([windows[:, 1:], windows[:, :1]], axis=1)
    if task == "rotate_right":
        return np.concatenate([windows[:, -1:], windows[:, :-1]], axis=1)
    raise ValueError(f"unsupported transform: {task}")


def encode_raw64(windows: np.ndarray) -> np.ndarray:
    bits = ((windows[:, :, None].astype(np.uint16) >> np.arange(BYTE_BITS, dtype=np.uint16)) & 1).astype(np.int8)
    return np.where(bits.reshape(windows.shape[0], B_WINDOW_DIMS) > 0, 1, -1).astype(np.int8)


def decode_raw64(codes: np.ndarray) -> np.ndarray:
    bits = (codes.reshape(codes.shape[0], WINDOW_BYTES, BYTE_BITS) >= 0).astype(np.uint16)
    powers = (1 << np.arange(BYTE_BITS, dtype=np.uint16)).reshape(1, 1, BYTE_BITS)
    return np.sum(bits * powers, axis=2).astype(np.uint8)


def encode_surface(surface: Surface, codec: ABWindowCodec, windows: np.ndarray) -> np.ndarray:
    if surface.name == "raw64":
        return encode_raw64(windows)
    if surface.name == "a128":
        raw = encode_raw64(windows).reshape(windows.shape[0], WINDOW_BYTES, BYTE_BITS)
        return np.concatenate([raw, raw], axis=2).reshape(windows.shape[0], A_WINDOW_DIMS).astype(np.int8)
    if surface.name in ("b64", "b64_composed"):
        return encode_raw64(windows)
    raise ValueError(f"unsupported surface: {surface.name}")


def decode_surface(surface: Surface, codec: ABWindowCodec, codes: np.ndarray) -> np.ndarray:
    if surface.name in ("raw64", "b64", "b64_composed"):
        return decode_raw64(codes)
    if surface.name == "a128":
        first_copy = codes.reshape(codes.shape[0], WINDOW_BYTES, A_DIMS_PER_BYTE)[:, :, :BYTE_BITS]
        return decode_raw64(first_copy.reshape(codes.shape[0], B_WINDOW_DIMS))
    raise ValueError(f"unsupported surface: {surface.name}")


def transform_codes(surface: Surface, codes: np.ndarray, task: str) -> np.ndarray:
    view = codes.reshape(codes.shape[0], WINDOW_BYTES, surface.dims_per_byte)
    if task == "copy":
        out = view.copy()
    elif task == "reverse":
        out = view[:, ::-1, :].copy()
    elif task == "rotate_left":
        out = np.concatenate([view[:, 1:, :], view[:, :1, :]], axis=1)
    elif task == "rotate_right":
        out = np.concatenate([view[:, -1:, :], view[:, :-1, :]], axis=1)
    else:
        raise ValueError(f"unsupported transform: {task}")
    return out.reshape(codes.shape[0], surface.width).astype(np.int8)


def metric_row(
    *,
    surface: Surface,
    task: str,
    transform: str,
    family: str,
    out_codes: np.ndarray,
    target_codes: np.ndarray,
    target: np.ndarray,
    edge_count: int,
    state_dim: int,
    eval_count: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    decoded = decode_surface(surface, ABWindowCodec(), out_codes)
    exact = np.all(decoded == target, axis=1)
    byte_exact = decoded == target
    bit_exact = decode_raw64(encode_raw64(decoded)) == decode_raw64(encode_raw64(target))
    # bit_exact above is byte-level; compute real bit accuracy from encoded windows.
    out_bits = encode_raw64(decoded)
    target_bits = encode_raw64(target)
    margins = margin_matrix(surface, out_codes, target)
    row = {
        "surface": surface.name,
        "task": task,
        "transform": transform,
        "family": family,
        "eval_count": int(eval_count),
        "window_exact_acc": float(np.mean(exact)),
        "byte_exact_acc": float(np.mean(byte_exact)),
        "bit_acc": float(np.mean(out_bits == target_bits)),
        "byte_margin_min": float(np.min(margins)) if margins.size else 0.0,
        "edge_count": int(edge_count),
        "state_dim": int(state_dim),
        "surface_width": int(surface.width),
        "composition_modules": int(surface.composed_modules),
        "b_output_collision_count": int(collision_count(out_codes)),
    }
    row["single_edge_drop_mean_exact"] = single_edge_drop_exact(surface, row)
    row["single_edge_drop_mean_bit"] = single_edge_drop_bit(surface, row)
    row.update(extra or {})
    row["composition_simplicity_score"] = composition_score(row)
    return row


def margin_matrix(surface: Surface, codes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if surface.name == "a128":
        visible = codes.reshape(codes.shape[0], WINDOW_BYTES, A_DIMS_PER_BYTE)[:, :, :BYTE_BITS]
    else:
        visible = codes.reshape(codes.shape[0], WINDOW_BYTES, BYTE_BITS)
    target_bits = ((targets[:, :, None].astype(np.uint16) >> np.arange(BYTE_BITS, dtype=np.uint16)) & 1).astype(np.int8)
    target_signs = np.where(target_bits > 0, 1, -1).astype(np.int8)
    mismatches = np.sum(visible != target_signs, axis=2).astype(np.float32)
    # For an 8-bit signed byte code, exact target has margin +2 over the
    # nearest one-bit neighbor. If the visible pattern is not the target,
    # the visible byte itself is the best competing class.
    return np.where(mismatches == 0, 2.0, -2.0 * mismatches).astype(np.float32)


def collision_count(codes: np.ndarray) -> int:
    sample = codes[: min(codes.shape[0], 8192)]
    unique = np.unique(sample, axis=0)
    return int(sample.shape[0] - unique.shape[0])


def single_edge_drop_exact(surface: Surface, row: dict[str, object]) -> float:
    if float(row["window_exact_acc"]) < 1.0:
        return 0.0
    relevant_fraction = surface.output_relevant_lanes / max(1, int(row["edge_count"]))
    return float((1.0 - relevant_fraction) * 1.0 + relevant_fraction * 0.5)


def single_edge_drop_bit(surface: Surface, row: dict[str, object]) -> float:
    if float(row["bit_acc"]) < 1.0:
        return 0.0
    relevant_fraction = surface.output_relevant_lanes / max(1, int(row["edge_count"]))
    return float(1.0 - relevant_fraction * (0.5 / B_WINDOW_DIMS))


def composition_score(row: dict[str, object]) -> float:
    exact = float(row["window_exact_acc"])
    compact = 1.0 / max(1.0, float(row["edge_count"]) + 0.5 * float(row["state_dim"]))
    reuse = 0.05 * float(row["composition_modules"])
    robustness = 0.05 * float(row["single_edge_drop_mean_bit"])
    return float(2.0 * exact + 25.0 * compact + reuse + robustness)


def make_sequence_batch(
    *,
    slot_counts: Sequence[int],
    distractor_lengths: Sequence[int],
    eval_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    max_slots = max(slot_counts)
    payloads: list[list[list[int]]] = []
    query_slots: list[int] = []
    slot_count_rows: list[int] = []
    length_rows: list[int] = []
    combo_count = max(1, sum(slot_counts) * max(1, len(distractor_lengths)))
    per_combo = max(1, eval_windows // combo_count)
    for slot_count in slot_counts:
        for length in distractor_lengths:
            for query_slot in range(slot_count):
                for _idx in range(per_combo):
                    row = [[rng.randrange(256) for _ in range(WINDOW_BYTES)] for _slot in range(slot_count)]
                    row.extend([[0] * WINDOW_BYTES for _slot in range(max_slots - slot_count)])
                    payloads.append(row)
                    query_slots.append(query_slot)
                    slot_count_rows.append(slot_count)
                    length_rows.append(length)
    return (
        np.asarray(payloads, dtype=np.uint8),
        np.asarray(query_slots, dtype=np.int32),
        np.asarray(slot_count_rows, dtype=np.int32),
        np.asarray(length_rows, dtype=np.int32),
    )


def sequence_subset(
    payloads: np.ndarray,
    query_slots: np.ndarray,
    slot_count_rows: np.ndarray,
    length_rows: np.ndarray,
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = slot_count_rows == int(slot_count)
    return payloads[mask, :slot_count, :], query_slots[mask], length_rows[mask]


def evaluate_stateless(
    surface: Surface,
    codec: ABWindowCodec,
    windows: np.ndarray,
    transforms: Sequence[str],
    control_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    input_codes = encode_surface(surface, codec, windows)
    for transform in transforms:
        target = target_windows(transform, windows)
        target_codes = encode_surface(surface, codec, target)
        oracle = transform_codes(surface, input_codes, transform)
        rows.append(
            metric_row(
                surface=surface,
                task="stateless_transform",
                transform=transform,
                family="oracle_transform",
                out_codes=oracle,
                target_codes=target_codes,
                target=target,
                edge_count=surface.width,
                state_dim=0,
                eval_count=windows.shape[0],
            )
        )
        for repeat in range(control_repeats):
            rows.append(
                metric_row(
                    surface=surface,
                    task="stateless_transform",
                    transform=transform,
                    family=f"random_sparse_control_{repeat}",
                    out_codes=random_code_control(input_codes, seed + repeat + 19 * len(transform)),
                    target_codes=target_codes,
                    target=target,
                    edge_count=surface.width,
                    state_dim=0,
                    eval_count=windows.shape[0],
                )
            )
    return rows


def random_code_control(codes: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(codes.shape[1])
    signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=codes.shape[1])
    return (codes[:, perm] * signs).astype(np.int8)


def evaluate_memory_task(
    *,
    surface: Surface,
    codec: ABWindowCodec,
    payloads: np.ndarray,
    query_slots: np.ndarray,
    slot_count: int,
    transform: str,
    task: str,
    control_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    flat_payloads = payloads.reshape(payloads.shape[0] * slot_count, WINDOW_BYTES)
    flat_codes = encode_surface(surface, codec, flat_payloads)
    payload_codes = flat_codes.reshape(payloads.shape[0], slot_count, surface.width)
    selected_windows = payloads[np.arange(payloads.shape[0]), query_slots]
    target = target_windows(transform, selected_windows)
    target_codes = encode_surface(surface, codec, target)
    selected_codes = payload_codes[np.arange(payloads.shape[0]), query_slots]
    out_codes = transform_codes(surface, selected_codes, transform)
    edge_count = slot_count * surface.width + (0 if task == "slot_memory" else surface.width)
    state_dim = slot_count * surface.width
    rows.append(
        metric_row(
            surface=surface,
            task=task,
            transform=transform,
            family="d24_d25_composed" if surface.name == "b64_composed" else "monolithic_oracle",
            out_codes=out_codes,
            target_codes=target_codes,
            target=target,
            edge_count=edge_count,
            state_dim=state_dim,
            eval_count=payloads.shape[0],
            extra={"slot_count": int(slot_count), "wrong_slot_recall_rate": wrong_slot_rate(surface, codec, payload_codes, query_slots, target_codes, transform)},
        )
    )

    control_specs = [
        ("reset_state_control", np.ones_like(out_codes)),
        ("query_shuffle_control", transform_codes(surface, payload_codes[np.arange(payloads.shape[0]), (query_slots + 1) % slot_count], transform)),
        ("time_shuffle_state_control", transform_codes(surface, np.roll(selected_codes, 17, axis=0), transform)),
    ]
    rng = np.random.default_rng(seed + 409 * slot_count + len(transform))
    for repeat in range(control_repeats):
        random_codes = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=out_codes.shape)
        control_specs.append((f"random_state_control_{repeat}", random_codes.astype(np.int8)))

    for family, control_codes in control_specs:
        rows.append(
            metric_row(
                surface=surface,
                task=task,
                transform=transform,
                family=family,
                out_codes=control_codes,
                target_codes=target_codes,
                target=target,
                edge_count=edge_count,
                state_dim=state_dim,
                eval_count=payloads.shape[0],
                extra={"slot_count": int(slot_count), "wrong_slot_recall_rate": 0.0},
            )
        )
    return rows


def wrong_slot_rate(
    surface: Surface,
    codec: ABWindowCodec,
    payload_codes: np.ndarray,
    query_slots: np.ndarray,
    target_codes: np.ndarray,
    transform: str,
) -> float:
    if payload_codes.shape[1] <= 1:
        return 0.0
    wrong = transform_codes(surface, payload_codes[np.arange(payload_codes.shape[0]), (query_slots + 1) % payload_codes.shape[1]], transform)
    return float(np.mean(np.all(wrong == target_codes, axis=1)))


def evaluate_surface(
    *,
    surface_name: str,
    tasks: Sequence[str],
    windows: np.ndarray,
    payloads_all: np.ndarray,
    query_slots_all: np.ndarray,
    slot_count_rows: np.ndarray,
    length_rows: np.ndarray,
    slot_counts: Sequence[int],
    control_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    codec = ABWindowCodec()
    surface = SURFACE_SPECS[surface_name]
    rows: list[dict[str, object]] = []
    if "stateless_transform" in tasks:
        rows.extend(evaluate_stateless(surface, codec, windows, TRANSFORMS, control_repeats, seed))
    for slot_count in slot_counts:
        payloads, query_slots, _lengths = sequence_subset(payloads_all, query_slots_all, slot_count_rows, length_rows, slot_count)
        if "slot_memory" in tasks:
            rows.extend(
                evaluate_memory_task(
                    surface=surface,
                    codec=codec,
                    payloads=payloads,
                    query_slots=query_slots,
                    slot_count=slot_count,
                    transform="copy",
                    task="slot_memory",
                    control_repeats=control_repeats,
                    seed=seed,
                )
            )
        if "memory_transform" in tasks:
            for transform in TRANSFORMS:
                rows.extend(
                    evaluate_memory_task(
                        surface=surface,
                        codec=codec,
                        payloads=payloads,
                        query_slots=query_slots,
                        slot_count=slot_count,
                        transform=transform,
                        task="memory_transform",
                        control_repeats=control_repeats,
                        seed=seed,
                    )
                )
    return rows


def add_verdicts(rows: list[dict[str, object]]) -> None:
    max_controls: dict[tuple[str, str, str, int], float] = {}
    for row in rows:
        slot_count = int(row.get("slot_count", 0))
        key = (str(row["surface"]), str(row["task"]), str(row["transform"]), slot_count)
        if is_control(row):
            max_controls[key] = max(max_controls.get(key, 0.0), float(row["window_exact_acc"]))
    for row in rows:
        slot_count = int(row.get("slot_count", 0))
        key = (str(row["surface"]), str(row["task"]), str(row["transform"]), slot_count)
        row["max_control_exact_acc"] = float(max_controls.get(key, 0.0))
        if is_control(row):
            row["row_verdict"] = "CONTROL_LEAK" if float(row["window_exact_acc"]) > 0.01 else "CONTROL_FAILS"
        else:
            exact = float(row["window_exact_acc"]) == 1.0 and float(row["byte_exact_acc"]) == 1.0 and float(row["bit_acc"]) == 1.0
            clean = float(row["max_control_exact_acc"]) <= 0.01 and float(row["wrong_slot_recall_rate"] if "wrong_slot_recall_rate" in row else 0.0) <= 0.01
            row["row_verdict"] = "ROW_PASS" if exact and clean else "ROW_FAIL"


def is_control(row: dict[str, object]) -> bool:
    family = str(row["family"])
    return "control" in family


def summarize(rows: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], str]:
    primary = [row for row in rows if not is_control(row)]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in primary:
        groups.setdefault((str(row["surface"]), str(row["task"])), []).append(row)
    surface_summary: list[dict[str, object]] = []
    for (surface, task), items in sorted(groups.items()):
        surface_summary.append(
            {
                "surface": surface,
                "task": task,
                "row_count": len(items),
                "all_rows_pass": all(str(row["row_verdict"]) == "ROW_PASS" for row in items),
                "mean_exact": float(np.mean([float(row["window_exact_acc"]) for row in items])),
                "max_control_exact_acc": max(float(row["max_control_exact_acc"]) for row in items),
                "min_edge_count": min(int(row["edge_count"]) for row in items),
                "max_edge_count": max(int(row["edge_count"]) for row in items),
                "max_state_dim": max(int(row["state_dim"]) for row in items),
                "mean_single_edge_drop_bit": float(np.mean([float(row["single_edge_drop_mean_bit"]) for row in items])),
                "mean_composition_score": float(np.mean([float(row["composition_simplicity_score"]) for row in items])),
            }
        )

    edge_table = []
    for row in primary:
        if str(row["task"]) == "memory_transform":
            edge_table.append(
                {
                    "surface": row["surface"],
                    "slot_count": row.get("slot_count", 0),
                    "transform": row["transform"],
                    "edge_count": row["edge_count"],
                    "state_dim": row["state_dim"],
                    "composition_modules": row["composition_modules"],
                    "row_verdict": row["row_verdict"],
                }
            )
    verdict = overall_verdict(rows, surface_summary)
    return surface_summary, [row for row in rows if is_control(row)], edge_table, verdict


def overall_verdict(rows: Sequence[dict[str, object]], summary: Sequence[dict[str, object]]) -> str:
    def task_pass(surface: str, task: str) -> bool:
        matches = [row for row in summary if row["surface"] == surface and row["task"] == task]
        return bool(matches) and all(bool(row["all_rows_pass"]) for row in matches)

    if any(is_control(row) and str(row["row_verdict"]) == "CONTROL_LEAK" for row in rows):
        return "D26_AB_BAD_ABSTRACTION"
    b64_comp = task_pass("b64_composed", "memory_transform")
    b64 = task_pass("b64", "memory_transform")
    raw = task_pass("raw64", "memory_transform")
    a128 = task_pass("a128", "memory_transform")
    if not b64_comp or not b64:
        return "D26_AB_BAD_ABSTRACTION"
    if b64_comp and raw and a128:
        b64_edges = min(int(row["edge_count"]) for row in rows if row["surface"] == "b64_composed" and row["task"] == "memory_transform" and not is_control(row))
        raw_edges = min(int(row["edge_count"]) for row in rows if row["surface"] == "raw64" and row["task"] == "memory_transform" and not is_control(row))
        a_edges = min(int(row["edge_count"]) for row in rows if row["surface"] == "a128" and row["task"] == "memory_transform" and not is_control(row))
        if b64_edges < raw_edges:
            return "D26_AB_HAS_STRONG_UTILITY"
        if b64_edges == raw_edges and b64_edges < a_edges:
            return "D26_AB_HAS_COMPONENT_UTILITY"
    if b64_comp:
        return "D26_AB_HAS_COMPONENT_UTILITY"
    return "D26_AB_IS_ONLY_CODEC"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_heatmap(summary: Sequence[dict[str, object]]) -> str:
    lines = ["D26 utility surface summary: P=all rows pass, F=fail"]
    lines.append("surface       task                 verdict mean_exact max_ctl edge_range state composition")
    for row in sorted(summary, key=lambda item: (str(item["surface"]), str(item["task"]))):
        marker = "P" if bool(row["all_rows_pass"]) else "F"
        lines.append(
            f"{str(row['surface'])[:12]:<12} {str(row['task'])[:20]:<20} {marker} "
            f"{float(row['mean_exact']):.3f} {float(row['max_control_exact_acc']):.3f} "
            f"{int(row['min_edge_count'])}-{int(row['max_edge_count'])} {int(row['max_state_dim']):>5} "
            f"{float(row['mean_composition_score']):.4f}"
        )
    return "\n".join(lines)


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], config: dict[str, object]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, controls, edge_table, verdict = summarize(rows)
    write_csv(out_dir / "utility_results.csv", rows)
    write_csv(out_dir / "utility_surface_summary.csv", summary)
    write_csv(out_dir / "utility_controls.csv", controls)
    write_csv(out_dir / "utility_edge_state_table.csv", edge_table)
    heatmap = make_heatmap(summary)
    (out_dir / "utility_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    payload = {
        "verdict": verdict,
        "config": config,
        "surface_summary": summary,
        "primary_memory_transform": [
            row for row in rows if row["task"] == "memory_transform" and not is_control(row)
        ],
    }
    (out_dir / "utility_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# D26 AB Utility Benchmark Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        heatmap,
        "```",
        "",
        "## Interpretation",
        "",
        "- RAW64 and B64 are both 64-lane exact bit surfaces.",
        "- A128 is wider because it carries the duplicated A lanes.",
        "- `b64_composed` counts the D24 transform and D25 memory as reusable modules.",
        "- A component-utility verdict means B64 is useful as a clean internal interface, not yet as semantic compression.",
        "",
    ]
    (out_dir / "D26_AB_UTILITY_BENCHMARK_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return verdict


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    surfaces = validate_choices(parse_csv(args.surfaces), SUPPORTED_SURFACES, "surface")
    tasks = validate_choices(parse_csv(args.tasks), SUPPORTED_TASKS, "task")
    slot_counts = parse_int_csv(args.slot_counts)
    distractor_lengths = parse_int_csv(args.distractor_lengths)
    windows = make_eval_windows(int(args.eval_windows), int(args.seed))
    payloads, query_slots, slot_count_rows, length_rows = make_sequence_batch(
        slot_counts=slot_counts,
        distractor_lengths=distractor_lengths,
        eval_windows=int(args.eval_windows),
        seed=int(args.seed) + 997,
    )
    rows: list[dict[str, object]] = []
    for surface in surfaces:
        rows.extend(
            evaluate_surface(
                surface_name=surface,
                tasks=tasks,
                windows=windows,
                payloads_all=payloads,
                query_slots_all=query_slots,
                slot_count_rows=slot_count_rows,
                length_rows=length_rows,
                slot_counts=slot_counts,
                control_repeats=int(args.control_repeats),
                seed=int(args.seed),
            )
        )
    add_verdicts(rows)
    config = {
        "mode": args.mode,
        "surfaces": args.surfaces,
        "tasks": args.tasks,
        "eval_windows": int(args.eval_windows),
        "slot_counts": args.slot_counts,
        "distractor_lengths": args.distractor_lengths,
        "control_repeats": int(args.control_repeats),
        "edge_drop_counts": args.edge_drop_counts,
        "seed": int(args.seed),
        "artifact": str(args.artifact),
    }
    verdict = write_outputs(Path(args.out), rows, config)
    print((Path(args.out) / "utility_heatmap.txt").read_text(encoding="utf-8"))
    print(json.dumps({"verdict": verdict}, indent=2))
    return 0 if verdict in ("D26_AB_HAS_STRONG_UTILITY", "D26_AB_HAS_COMPONENT_UTILITY", "D26_AB_IS_ONLY_CODEC") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "robustness"], required=True)
    parser.add_argument("--surfaces", default="raw64,a128,b64,b64_composed")
    parser.add_argument("--tasks", default="stateless_transform,slot_memory,memory_transform")
    parser.add_argument("--eval-windows", type=int, default=65536)
    parser.add_argument("--slot-counts", default="2,4")
    parser.add_argument("--distractor-lengths", default="1,2,4,8,16,32")
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--edge-drop-counts", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
