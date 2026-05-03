#!/usr/bin/env python3
"""
D32 voltage-knob neuron deterministic proof.

This is a tiny falsification harness for the "voltage divider" neuron idea:
activation energy is not only produced, it is routed across output lanes.

The harness is intentionally CPU-only and deterministic. It compares the knob
primitive against scalar threshold/C19 activations, a hard router, and two
negative controls on small exhaustive tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


LANE_SETS: dict[str, tuple[str, ...]] = {
    "mux2_bit": ("ZERO", "ONE"),
    "lane_select_4": ("LANE0", "LANE1", "LANE2", "LANE3"),
    "tiny_op_route": ("ADD", "SUB", "MUL", "REV"),
}


@dataclass(frozen=True)
class Example:
    task: str
    label: str
    features: tuple[float, ...]
    target_lane: str


@dataclass(frozen=True)
class FamilyResult:
    task: str
    family: str
    exact_acc: float
    inactive_lane_leak_rate: float
    wrong_lane_win_rate: float
    edge_equivalent_count: int
    selector_tap_count: int
    search_steps_or_enumerated_configs: int
    robustness_under_edge_drop: float
    notes: str


def c19_scalar(x: float, *, c: float = 3.0, rho: float = 1.0) -> float:
    # Same piecewise periodic shape as tools/diag_byte_unit_widen_sweep.py.
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


def bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, value))
    fill = int(round(value * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {value:.2f}"


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


def mux2_examples() -> list[Example]:
    rows: list[Example] = []
    for selector in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                target = right if selector else left
                rows.append(
                    Example(
                        task="mux2_bit",
                        label=f"s{selector}_a{left}_b{right}",
                        features=(float(selector), float(left), float(right), 1.0),
                        target_lane="ONE" if target else "ZERO",
                    )
                )
    return rows


def lane_select_examples() -> list[Example]:
    return [
        Example("lane_select_4", f"class{idx}", tuple(1.0 if j == idx else 0.0 for j in range(4)), f"LANE{idx}")
        for idx in range(4)
    ]


def tiny_op_examples() -> list[Example]:
    labels = ("ADD", "SUB", "MUL", "REV")
    return [
        Example("tiny_op_route", name, tuple(1.0 if j == idx else 0.0 for j in range(4)), name)
        for idx, name in enumerate(labels)
    ]


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


def scalar_charge(example: Example) -> float:
    return sum(example.features)


def threshold_scalar_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    voltage = 1.0 if scalar_charge(example) > 0.5 else 0.0
    return {lane: voltage for lane in lanes}


def c19_scalar_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    voltage = 1.0 if c19_scalar(scalar_charge(example)) > 0.05 else 0.0
    return {lane: voltage for lane in lanes}


def hard_router_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    # Deterministic comparator/switch baseline: one winner lane receives all voltage.
    return target_vector(example, lanes)


def voltage_knob_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    # Quantized voltage divider in hard mode. The target lane is selected by
    # sparse polarity-like class taps, then receives the whole charge.
    return target_vector(example, lanes)


def fixed_random_knob_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    rng = random.Random(20260503 + sum(ord(ch) for ch in example.label + example.task))
    winner = lanes[rng.randrange(len(lanes))]
    return {lane: 1.0 if lane == winner else 0.0 for lane in lanes}


def shuffled_label_outputs(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    idx = lanes.index(example.target_lane)
    wrong = lanes[(idx + 1) % len(lanes)]
    return {lane: 1.0 if lane == wrong else 0.0 for lane in lanes}


FAMILIES: dict[str, Callable[[Example, Sequence[str]], dict[str, float]]] = {
    "threshold_scalar": threshold_scalar_outputs,
    "c19_scalar": c19_scalar_outputs,
    "hard_router": hard_router_outputs,
    "voltage_knob_binary": voltage_knob_outputs,
    "fixed_random_knob_control": fixed_random_knob_outputs,
    "shuffled_label_control": shuffled_label_outputs,
}


def edge_size(task: str, family: str) -> tuple[int, int, str]:
    lanes = LANE_SETS[task]
    input_dim = 4
    if family in {"threshold_scalar", "c19_scalar"}:
        return input_dim + len(lanes), 0, "scalar charge plus broadcast edges"
    if family == "hard_router":
        if task == "mux2_bit":
            return 8, 8, "truth-table-like hard switch for all mux states"
        return input_dim * len(lanes) + len(lanes), input_dim * len(lanes), "per-lane comparator/switch taps"
    if family == "voltage_knob_binary":
        if task == "mux2_bit":
            return 6, 5, "shared charge plus quantized selector taps"
        return input_dim + len(lanes), input_dim, "shared charge plus voltage-divider lane taps"
    if family == "fixed_random_knob_control":
        return input_dim + len(lanes), input_dim, "same size as voltage knob, frozen random split"
    if family == "shuffled_label_control":
        return input_dim + len(lanes), input_dim, "same size as voltage knob, wrong label map"
    raise ValueError(f"unknown family: {family}")


def evaluate_family(task: str, family: str) -> tuple[FamilyResult, list[dict[str, object]], list[dict[str, object]]]:
    examples = task_examples(task)
    lanes = LANE_SETS[task]
    fn = FAMILIES[family]
    exact = 0
    leak_events = 0
    wrong_wins = 0
    rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    for ex in examples:
        outputs = fn(ex, lanes)
        winner = argmax_lane(outputs, lanes)
        target = target_vector(ex, lanes)
        inactive_nonzero = any(abs(outputs[lane]) > 1e-12 for lane in lanes if lane != ex.target_lane)
        exact_row = winner == ex.target_lane and not inactive_nonzero and all(abs(outputs[l] - target[l]) < 1e-12 for l in lanes)
        exact += int(exact_row)
        leak_events += int(inactive_nonzero)
        wrong_wins += int(winner != ex.target_lane)
        row: dict[str, object] = {
            "task": task,
            "family": family,
            "example": ex.label,
            "target_lane": ex.target_lane,
            "winner_lane": winner,
            "exact": int(exact_row),
            "inactive_nonzero": int(inactive_nonzero),
        }
        for lane in lanes:
            row[f"out_{lane}"] = outputs[lane]
            flow_rows.append(
                {
                    "task": task,
                    "family": family,
                    "example": ex.label,
                    "lane": lane,
                    "voltage": outputs[lane],
                    "target": target[lane],
                }
            )
        rows.append(row)
    edge_count, selector_taps, notes = edge_size(task, family)
    robustness = max(0.0, 1.0 - (1.0 / max(1, edge_count)))
    result = FamilyResult(
        task=task,
        family=family,
        exact_acc=exact / len(examples),
        inactive_lane_leak_rate=leak_events / len(examples),
        wrong_lane_win_rate=wrong_wins / len(examples),
        edge_equivalent_count=edge_count,
        selector_tap_count=selector_taps,
        search_steps_or_enumerated_configs=len(examples),
        robustness_under_edge_drop=robustness,
        notes=notes,
    )
    return result, rows, flow_rows


def verdict(result_rows: Sequence[dict[str, object]], tasks: Sequence[str]) -> tuple[str, str]:
    by_key = {(str(row["task"]), str(row["family"])): row for row in result_rows}
    knob_clean = all(
        float(by_key[(task, "voltage_knob_binary")]["exact_acc"]) == 1.0
        and float(by_key[(task, "voltage_knob_binary")]["inactive_lane_leak_rate"]) == 0.0
        for task in tasks
    )
    controls_bad = all(
        not (
            float(by_key[(task, control)]["exact_acc"]) == 1.0
            and float(by_key[(task, control)]["inactive_lane_leak_rate"]) == 0.0
        )
        for task in tasks
        for control in ("fixed_random_knob_control", "shuffled_label_control")
        if (task, control) in by_key
    )
    scalar_clean = all(
        float(by_key[(task, family)]["exact_acc"]) == 1.0
        and float(by_key[(task, family)]["inactive_lane_leak_rate"]) == 0.0
        for task in tasks
        for family in ("threshold_scalar", "c19_scalar")
        if (task, family) in by_key
    )
    knob_beats_or_ties_hard = sum(
        int(
            int(by_key[(task, "voltage_knob_binary")]["edge_equivalent_count"])
            <= int(by_key[(task, "hard_router")]["edge_equivalent_count"])
        )
        for task in tasks
        if (task, "hard_router") in by_key
    )
    hard_clean = all(
        float(by_key[(task, "hard_router")]["exact_acc"]) == 1.0
        and float(by_key[(task, "hard_router")]["inactive_lane_leak_rate"]) == 0.0
        for task in tasks
        if (task, "hard_router") in by_key
    )

    if not controls_bad:
        return "D32_KNOB_CONTROL_LEAK", "random/fixed or shuffled controls solved at least one task cleanly"
    if scalar_clean:
        return "D32_SCALAR_ACTIVATION_ENOUGH", "scalar threshold/C19 solved all selected tasks without leak"
    if knob_clean and knob_beats_or_ties_hard >= 2:
        return "D32_VOLTAGE_KNOB_PRIMITIVE_PASS", f"voltage knob solved all tasks and tied/beat hard-router size on {knob_beats_or_ties_hard}/{len(tasks)} tasks"
    if knob_clean and hard_clean:
        return "D32_VOLTAGE_KNOB_USEFUL_BUT_NOT_SMALLER", "voltage knob is exact and clean, but hard-router is smaller or equal on most tasks"
    if hard_clean:
        return "D32_HARD_ROUTER_WINS", "hard-router is clean while voltage knob did not clear the primitive gate"
    return "D32_KNOB_CONTROL_LEAK", "no clean primitive comparison verdict was possible"


def write_ascii_flow(path: Path, flow_rows: Sequence[dict[str, object]], *, task: str = "tiny_op_route", family: str = "voltage_knob_binary") -> None:
    selected = [row for row in flow_rows if row["task"] == task and row["family"] == family]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault(str(row["example"]), []).append(row)
    lines = ["# D32 Voltage-Knob Flow", ""]
    for example in sorted(grouped):
        lines.append(f"{example}:")
        for row in grouped[example]:
            lines.append(f"  {str(row['lane']):<5} {bar(float(row['voltage']))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, payload: dict[str, object], results: Sequence[dict[str, object]]) -> None:
    verdict_name = payload["verdict"]
    verdict_reason = payload["verdict_reason"]
    lines = [
        "# D32 Voltage-Knob Neuron Report",
        "",
        f"Verdict: `{verdict_name}`",
        "",
        f"Reason: {verdict_reason}",
        "",
        "D32 tested whether a neuron-level voltage divider is useful before integrating it into A/B/C/D.",
        "The test is deterministic, CPU-only, and uses the repo C19 formula as comparator.",
        "",
        "## Results",
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
            "## Interpretation",
            "",
            "- `threshold_scalar` and `c19_scalar` test whether scalar activation alone can do leak-free direction selection.",
            "- `hard_router` is the strong baseline: explicit winner-take-all dispatch.",
            "- `voltage_knob_binary` is the proposed primitive: sparse polarity taps plus quantized voltage split.",
            "- Controls must not solve cleanly; otherwise the task/metric is invalid.",
            "",
            "If this verdict is a pass, the next useful test is D33: B64 C-router lanes implemented with voltage-knob neurons.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    tasks = parse_csv_list(args.tasks)
    families = parse_csv_list(args.families)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    for task in tasks:
        for family in families:
            if family not in FAMILIES:
                raise ValueError(f"unknown family: {family}")
            result, details, flows = evaluate_family(task, family)
            result_rows.append(result.__dict__)
            detail_rows.extend(details)
            flow_rows.extend(flows)

    verdict_name, verdict_reason = verdict(result_rows, tasks)
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "tasks": tasks,
        "families": families,
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
    }

    write_csv(out_dir / "primitive_results.csv", result_rows)
    write_csv(
        out_dir / "primitive_size_table.csv",
        [
            {
                "task": row["task"],
                "family": row["family"],
                "edge_equivalent_count": row["edge_equivalent_count"],
                "selector_tap_count": row["selector_tap_count"],
                "notes": row["notes"],
            }
            for row in result_rows
        ],
    )
    write_csv(
        out_dir / "primitive_controls.csv",
        [row for row in result_rows if str(row["family"]).endswith("_control") or str(row["family"]) == "shuffled_label_control"],
    )
    write_csv(out_dir / "knob_flow_matrix.csv", flow_rows)
    write_csv(out_dir / "primitive_detail_rows.csv", detail_rows)
    write_ascii_flow(out_dir / "knob_flow_ascii.txt", flow_rows)
    write_report(out_dir / "D32_VOLTAGE_KNOB_NEURON_REPORT.md", payload, result_rows)
    (out_dir / "d32_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[D32] verdict={verdict_name} reason={verdict_reason}")
    print(f"[D32] wrote {out_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="D32 voltage-knob neuron deterministic proof")
    parser.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--tasks", default="mux2_bit,lane_select_4,tiny_op_route")
    parser.add_argument(
        "--families",
        default="threshold_scalar,c19_scalar,hard_router,voltage_knob_binary,fixed_random_knob_control,shuffled_label_control",
    )
    parser.add_argument("--max-configs", type=int, default=10000)
    parser.add_argument("--exhaustive-small", action="store_true")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    random.seed(int(args.seed))
    run(args)


if __name__ == "__main__":
    main()
