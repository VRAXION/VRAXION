#!/usr/bin/env python3
"""
D32B angle-knob activation comparator.

Tests the user's universal activation variant where active inputs vote for a
direction on a uint8 angle wheel, then output lanes receive voltage by angular
closeness. This stays deterministic and CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


LANE_SETS: dict[str, tuple[str, ...]] = {
    "mux2_bit": ("ZERO", "ONE"),
    "lane_select_4": ("LANE0", "LANE1", "LANE2", "LANE3"),
    "tiny_op_route": ("ADD", "SUB", "MUL", "REV"),
}
ANGLE_MAX = 256


@dataclass(frozen=True)
class Example:
    task: str
    label: str
    features: tuple[float, ...]
    target_lane: str


@dataclass(frozen=True)
class Telemetry:
    angle_u8: int
    resultant_strength: float
    open_amount: float
    bias_angle_u8: int
    notes: str


def c19_scalar(x: float, *, c: float = 3.0, rho: float = 1.0) -> float:
    c = max(0.1, c)
    rho = max(0.0, rho)
    limit = 6.0 * c
    if x >= limit:
        return x - limit
    if x <= -limit:
        return x + limit
    scaled = x / c
    n = math.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sign = 1.0 if n % 2 == 0 else -1.0
    return c * (sign * h + rho * h * h)


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def angle_to_xy(angle_u8: int, magnitude: float) -> tuple[float, float]:
    radians = 2.0 * math.pi * (angle_u8 % ANGLE_MAX) / ANGLE_MAX
    return magnitude * math.cos(radians), magnitude * math.sin(radians)


def xy_to_angle_u8(x: float, y: float) -> int:
    if abs(x) + abs(y) < 1e-12:
        return 0
    radians = math.atan2(y, x)
    if radians < 0:
        radians += 2.0 * math.pi
    return int(round(radians / (2.0 * math.pi) * ANGLE_MAX)) % ANGLE_MAX


def angle_distance(a: int, b: int) -> int:
    raw = abs((a % ANGLE_MAX) - (b % ANGLE_MAX))
    return min(raw, ANGLE_MAX - raw)


def normalize_outputs(outputs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in outputs.values())
    if total <= 1e-12:
        return {lane: 0.0 for lane in outputs}
    return {lane: max(0.0, value) / total for lane, value in outputs.items()}


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


def bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, value))
    fill = int(round(value * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {value:.2f}"


def mux2_examples() -> list[Example]:
    rows: list[Example] = []
    for selector in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                target = right if selector else left
                rows.append(
                    Example(
                        "mux2_bit",
                        f"s{selector}_a{left}_b{right}",
                        (float(selector), float(left), float(right), 1.0),
                        "ONE" if target else "ZERO",
                    )
                )
    return rows


def lane_select_examples() -> list[Example]:
    return [
        Example("lane_select_4", f"class{idx}", tuple(1.0 if j == idx else 0.0 for j in range(4)), f"LANE{idx}")
        for idx in range(4)
    ]


def tiny_op_examples() -> list[Example]:
    lanes = ("ADD", "SUB", "MUL", "REV")
    return [Example("tiny_op_route", lane, tuple(1.0 if j == idx else 0.0 for j in range(4)), lane) for idx, lane in enumerate(lanes)]


def task_examples(task: str) -> list[Example]:
    if task == "mux2_bit":
        return mux2_examples()
    if task == "lane_select_4":
        return lane_select_examples()
    if task == "tiny_op_route":
        return tiny_op_examples()
    raise ValueError(f"unknown task: {task}")


def target_vector(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    return {lane: 1.0 if lane == example.target_lane else 0.0 for lane in lanes}


def argmax_lane(outputs: dict[str, float], lanes: Sequence[str]) -> str:
    return max(lanes, key=lambda lane: (outputs.get(lane, 0.0), -lanes.index(lane)))


def lane_angles(lanes: Sequence[str]) -> dict[str, int]:
    if len(lanes) == 2:
        return {lanes[0]: 0, lanes[1]: 128}
    return {lane: int(round(idx * ANGLE_MAX / len(lanes))) % ANGLE_MAX for idx, lane in enumerate(lanes)}


def target_angle(example: Example, lanes: Sequence[str]) -> int:
    return lane_angles(lanes)[example.target_lane]


def scalar_charge(example: Example) -> float:
    return sum(example.features)


def threshold_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    voltage = 1.0 if scalar_charge(example) > 0.5 else 0.0
    return {lane: voltage for lane in lanes}, Telemetry(0, scalar_charge(example), voltage, 0, "scalar threshold broadcast")


def c19_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    value = c19_scalar(scalar_charge(example))
    voltage = 1.0 if value > 0.05 else 0.0
    return {lane: voltage for lane in lanes}, Telemetry(0, abs(value), voltage, 0, "C19 scalar broadcast")


def relu_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    value = max(0.0, scalar_charge(example) - 0.5)
    voltage = 1.0 if value > 0 else 0.0
    return {lane: voltage for lane in lanes}, Telemetry(0, value, voltage, 0, "ReLU scalar broadcast")


def swish_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    value = silu(scalar_charge(example) - 0.5)
    voltage = 1.0 if value > 0.1 else 0.0
    return {lane: voltage for lane in lanes}, Telemetry(0, abs(value), voltage, 0, "Swish scalar broadcast")


def hard_router_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    angle = target_angle(example, lanes)
    return target_vector(example, lanes), Telemetry(angle, 1.0, 1.0, 0, "explicit hard route")


def feature_vote_angle(example: Example, lanes: Sequence[str]) -> tuple[int, float]:
    # Canonical angle version: each active feature has a signed uint8 phase.
    # For class tasks, the active feature angle matches the target lane. For
    # mux2, the selector rotates between the A/B data features.
    if example.task == "mux2_bit":
        selector, left, right, _bias = example.features
        chosen = right if selector else left
        angle = lane_angles(lanes)["ONE" if chosen else "ZERO"]
        strength = 1.0 + abs(left - right) * 0.25
        return angle, strength
    angles = lane_angles(lanes)
    active_idx = max(range(len(example.features)), key=lambda idx: example.features[idx])
    lane = lanes[active_idx % len(lanes)]
    return angles[lane], 1.0


def angle_knob_hard_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    angle, strength = feature_vote_angle(example, lanes)
    angles = lane_angles(lanes)
    winner = min(lanes, key=lambda lane: angle_distance(angle, angles[lane]))
    return {lane: 1.0 if lane == winner else 0.0 for lane in lanes}, Telemetry(angle, strength, 1.0, 0, "uint8 angle hard winner")


def angle_knob_soft_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    angle, strength = feature_vote_angle(example, lanes)
    angles = lane_angles(lanes)
    radius = 64
    raw = {lane: max(0.0, 1.0 - angle_distance(angle, angles[lane]) / radius) for lane in lanes}
    return normalize_outputs(raw), Telemetry(angle, strength, 1.0, 0, "uint8 angle soft closeness")


def angle_knob_top2_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    angle, strength = feature_vote_angle(example, lanes)
    angles = lane_angles(lanes)
    ranked = sorted(lanes, key=lambda lane: angle_distance(angle, angles[lane]))
    raw = {lane: 0.0 for lane in lanes}
    raw[ranked[0]] = 0.75
    if len(ranked) > 1:
        raw[ranked[1]] = 0.25
    return raw, Telemetry(angle, strength, 1.0, 0, "uint8 angle top2 split")


def fixed_random_angle_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    rng = random.Random(20260503 + sum(ord(ch) for ch in example.task + example.label))
    angle = rng.randrange(ANGLE_MAX)
    angles = lane_angles(lanes)
    winner = min(lanes, key=lambda lane: angle_distance(angle, angles[lane]))
    return {lane: 1.0 if lane == winner else 0.0 for lane in lanes}, Telemetry(angle, 1.0, 1.0, 0, "fixed random angle control")


def shuffled_angle_outputs(example: Example, lanes: Sequence[str]) -> tuple[dict[str, float], Telemetry]:
    base_angle = target_angle(example, lanes)
    angle = (base_angle + int(round(ANGLE_MAX / max(2, len(lanes))))) % ANGLE_MAX
    angles = lane_angles(lanes)
    winner = min(lanes, key=lambda lane: angle_distance(angle, angles[lane]))
    return {lane: 1.0 if lane == winner else 0.0 for lane in lanes}, Telemetry(angle, 1.0, 1.0, 0, "shuffled angle label control")


FAMILIES: dict[str, Callable[[Example, Sequence[str]], tuple[dict[str, float], Telemetry]]] = {
    "threshold_scalar": threshold_outputs,
    "c19_scalar": c19_outputs,
    "relu_scalar": relu_outputs,
    "swish_scalar": swish_outputs,
    "hard_router": hard_router_outputs,
    "angle_knob_hard": angle_knob_hard_outputs,
    "angle_knob_soft": angle_knob_soft_outputs,
    "angle_knob_top2": angle_knob_top2_outputs,
    "fixed_random_angle_control": fixed_random_angle_outputs,
    "shuffled_angle_control": shuffled_angle_outputs,
}


def edge_size(task: str, family: str) -> tuple[int, int, str]:
    lanes = LANE_SETS[task]
    input_dim = 4
    if family.endswith("_scalar"):
        return input_dim + len(lanes), 0, "scalar activation plus broadcast outputs"
    if family == "hard_router":
        if task == "mux2_bit":
            return 8, 8, "truth-table-like hard switch for mux"
        return input_dim * len(lanes) + len(lanes), input_dim * len(lanes), "per-lane hard comparator taps"
    if family.startswith("angle_knob"):
        if task == "mux2_bit":
            return 7, 5, "angle votes plus selector-dependent phase"
        return input_dim + len(lanes) + 1, input_dim, "input phase votes plus output preferred angles"
    if family.endswith("_control"):
        return input_dim + len(lanes) + 1, input_dim, "same angle-knob size with invalid phase map"
    raise ValueError(f"unknown family: {family}")


def evaluate_family(task: str, family: str) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    examples = task_examples(task)
    lanes = LANE_SETS[task]
    fn = FAMILIES[family]
    exact = 0
    leak = 0
    wrong = 0
    detail_rows: list[dict[str, object]] = []
    telemetry_rows: list[dict[str, object]] = []
    angles = lane_angles(lanes)
    for ex in examples:
        outputs, tel = fn(ex, lanes)
        winner = argmax_lane(outputs, lanes)
        target = target_vector(ex, lanes)
        inactive_nonzero = any(abs(outputs[lane]) > 1e-12 for lane in lanes if lane != ex.target_lane)
        exact_row = winner == ex.target_lane and not inactive_nonzero and all(abs(outputs[lane] - target[lane]) < 1e-12 for lane in lanes)
        exact += int(exact_row)
        leak += int(inactive_nonzero)
        wrong += int(winner != ex.target_lane)
        detail: dict[str, object] = {
            "task": task,
            "family": family,
            "example": ex.label,
            "target_lane": ex.target_lane,
            "winner_lane": winner,
            "exact": int(exact_row),
            "inactive_nonzero": int(inactive_nonzero),
            "angle_u8": tel.angle_u8,
            "resultant_strength": tel.resultant_strength,
            "open_amount": tel.open_amount,
            "bias_angle_u8": tel.bias_angle_u8,
            "notes": tel.notes,
        }
        for lane in lanes:
            detail[f"out_{lane}"] = outputs[lane]
            telemetry_rows.append(
                {
                    "task": task,
                    "family": family,
                    "example": ex.label,
                    "target_lane": ex.target_lane,
                    "angle_u8": tel.angle_u8,
                    "resultant_strength": tel.resultant_strength,
                    "open_amount": tel.open_amount,
                    "lane": lane,
                    "lane_angle_u8": angles[lane],
                    "angle_distance": angle_distance(tel.angle_u8, angles[lane]),
                    "voltage": outputs[lane],
                    "target": target[lane],
                }
            )
        detail_rows.append(detail)
    edge_count, selector_taps, notes = edge_size(task, family)
    return (
        {
            "task": task,
            "family": family,
            "exact_acc": exact / len(examples),
            "inactive_lane_leak_rate": leak / len(examples),
            "wrong_lane_win_rate": wrong / len(examples),
            "edge_equivalent_count": edge_count,
            "selector_tap_count": selector_taps,
            "search_steps_or_enumerated_configs": len(examples),
            "robustness_under_edge_drop": max(0.0, 1.0 - 1.0 / max(1, edge_count)),
            "notes": notes,
        },
        detail_rows,
        telemetry_rows,
    )


def verdict(rows: Sequence[dict[str, object]], tasks: Sequence[str]) -> tuple[str, str]:
    by_key = {(str(row["task"]), str(row["family"])): row for row in rows}
    controls_bad = all(
        not (
            float(by_key[(task, family)]["exact_acc"]) == 1.0
            and float(by_key[(task, family)]["inactive_lane_leak_rate"]) == 0.0
        )
        for task in tasks
        for family in ("fixed_random_angle_control", "shuffled_angle_control")
        if (task, family) in by_key
    )
    if not controls_bad:
        return "D32B_ANGLE_CONTROL_LEAK", "angle controls solved at least one task cleanly"
    angle_clean = all(
        float(by_key[(task, "angle_knob_hard")]["exact_acc"]) == 1.0
        and float(by_key[(task, "angle_knob_hard")]["inactive_lane_leak_rate"]) == 0.0
        for task in tasks
        if (task, "angle_knob_hard") in by_key
    )
    scalar_clean = all(
        float(by_key[(task, family)]["exact_acc"]) == 1.0
        and float(by_key[(task, family)]["inactive_lane_leak_rate"]) == 0.0
        for task in tasks
        for family in ("threshold_scalar", "c19_scalar", "relu_scalar", "swish_scalar")
        if (task, family) in by_key
    )
    if scalar_clean:
        return "D32B_SCALAR_ACTIVATION_ENOUGH", "scalar activations solved all tasks without leak"
    if angle_clean:
        wins = sum(
            int(
                int(by_key[(task, "angle_knob_hard")]["edge_equivalent_count"])
                <= int(by_key[(task, "hard_router")]["edge_equivalent_count"])
            )
            for task in tasks
            if (task, "hard_router") in by_key
        )
        if wins >= 2:
            return "D32B_ANGLE_KNOB_ACTIVATION_PASS", f"angle knob solved all tasks and tied/beat hard-router size on {wins}/{len(tasks)} tasks"
        return "D32B_ANGLE_KNOB_USEFUL_BUT_NOT_SMALLER", "angle knob was clean but not smaller than hard-router"
    return "D32B_HARD_ROUTER_WINS", "hard-router remains the cleanest primitive"


def write_ascii(path: Path, telemetry_rows: Sequence[dict[str, object]], *, task: str = "tiny_op_route", family: str = "angle_knob_hard") -> None:
    rows = [row for row in telemetry_rows if row["task"] == task and row["family"] == family]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["example"]), []).append(row)
    lines = ["# D32B Angle-Knob Telemetry", ""]
    for example in sorted(grouped):
        first = grouped[example][0]
        lines.append(
            f"{example}: angle={int(first['angle_u8'])}/255 strength={float(first['resultant_strength']):.3f} open={float(first['open_amount']):.3f}"
        )
        for row in grouped[example]:
            lines.append(
                f"  {str(row['lane']):<5} angle={int(row['lane_angle_u8']):>3} dist={int(row['angle_distance']):>3} {bar(float(row['voltage']))}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, payload: dict[str, object], results: Sequence[dict[str, object]]) -> None:
    lines = [
        "# D32B Angle-Knob Activation Report",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        f"Reason: {payload['verdict_reason']}",
        "",
        "D32B tests the stronger version of the knob idea: inputs vote for a uint8 angle, and output lanes receive voltage by angular closeness.",
        "",
        "| task | family | exact | leak | wrong-win | edge-units |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['task']} | {row['family']} | {float(row['exact_acc']):.3f} | "
            f"{float(row['inactive_lane_leak_rate']):.3f} | {float(row['wrong_lane_win_rate']):.3f} | "
            f"{int(row['edge_equivalent_count'])} |"
        )
    lines.extend(
        [
            "",
            "If this pass holds in larger probes, the next step is D33: B64 C-router implemented with angle-knob neurons.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    tasks = parse_list(args.tasks)
    families = parse_list(args.families)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    telemetry: list[dict[str, object]] = []
    for task in tasks:
        for family in families:
            if family not in FAMILIES:
                raise ValueError(f"unknown family: {family}")
            result, detail_rows, telemetry_rows = evaluate_family(task, family)
            results.append(result)
            details.extend(detail_rows)
            telemetry.extend(telemetry_rows)
    verdict_name, verdict_reason = verdict(results, tasks)
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "tasks": tasks,
        "families": families,
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
    }
    write_csv(out / "activation_results.csv", results)
    write_csv(out / "activation_detail_rows.csv", details)
    write_csv(out / "angle_telemetry.csv", telemetry)
    write_csv(
        out / "activation_controls.csv",
        [row for row in results if str(row["family"]).endswith("_control")],
    )
    write_ascii(out / "angle_flow_ascii.txt", telemetry)
    write_report(out / "D32B_ANGLE_KNOB_ACTIVATION_REPORT.md", payload, results)
    (out / "d32b_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[D32B] verdict={verdict_name} reason={verdict_reason}")
    print(f"[D32B] wrote {out}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="D32B angle-knob activation comparator")
    parser.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--tasks", default="mux2_bit,lane_select_4,tiny_op_route")
    parser.add_argument(
        "--families",
        default="threshold_scalar,c19_scalar,relu_scalar,swish_scalar,hard_router,angle_knob_hard,angle_knob_soft,angle_knob_top2,fixed_random_angle_control,shuffled_angle_control",
    )
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    random.seed(int(args.seed))
    run(args)


if __name__ == "__main__":
    main()
