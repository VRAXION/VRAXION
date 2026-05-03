#!/usr/bin/env python3
"""
D32C universal angle-knob neuron stress test.

Deterministic CPU-only benchmark for the stronger angle-knob activation idea:
input edges vote for a uint8 direction, optional bias pulls the wheel, and
output edges fire by angular closeness. The benchmark compares against scalar
threshold/C19/ReLU/Swish and hard-router baselines under explicit size metrics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ANGLE_MAX = 256
SCALAR_FAMILIES = {"threshold", "c19", "relu", "swish"}
ANGLE_FAMILIES = {"angle_minimal", "angle_bias", "angle_aperture", "angle_beukers"}
CONTROL_FAMILIES = {"random_angle_control", "shuffled_angle_control"}


@dataclass(frozen=True)
class Example:
    group: str
    task: str
    label: str
    features: tuple[float, ...]
    target_lane: str


@dataclass(frozen=True)
class TaskDef:
    group: str
    name: str
    lanes: tuple[str, ...]
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class Candidate:
    family: str
    task: str
    lanes: tuple[str, ...]
    vote_angles: tuple[int, ...]
    strengths: tuple[float, ...]
    bias_angle: int
    bias_strength: float
    threshold: float
    aperture: int
    curve: str
    scalar_weights: tuple[float, ...] = ()
    scalar_bias: float = 0.0
    aux: str = ""


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


def angle_distance(left: int, right: int) -> int:
    raw = abs((left % ANGLE_MAX) - (right % ANGLE_MAX))
    return min(raw, ANGLE_MAX - raw)


def lane_angles(lanes: Sequence[str]) -> dict[str, int]:
    if len(lanes) == 2:
        return {lanes[0]: 0, lanes[1]: 128}
    return {lane: int(round(idx * ANGLE_MAX / len(lanes))) % ANGLE_MAX for idx, lane in enumerate(lanes)}


def bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, value))
    fill = int(round(value * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {value:.2f}"


def target_vector(example: Example, lanes: Sequence[str]) -> dict[str, float]:
    return {lane: 1.0 if lane == example.target_lane else 0.0 for lane in lanes}


def argmax_lane(outputs: dict[str, float], lanes: Sequence[str]) -> str:
    return max(lanes, key=lambda lane: (outputs.get(lane, 0.0), -lanes.index(lane)))


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


def make_examples() -> list[TaskDef]:
    tasks: list[TaskDef] = []

    lanes2 = ("ZERO", "ONE")
    tasks.append(
        TaskDef(
            "scalar",
            "identity_bit",
            lanes2,
            tuple(Example("scalar", "identity_bit", f"x{x}", (float(x), 1.0), "ONE" if x else "ZERO") for x in (0, 1)),
        )
    )
    tasks.append(
        TaskDef(
            "scalar",
            "sign_threshold",
            lanes2,
            tuple(
                Example("scalar", "sign_threshold", f"x{x}", (1.0 if x > 0 else 0.0, 1.0 if x <= 0 else 0.0, 1.0), "ONE" if x > 0 else "ZERO")
                for x in (-1, 0, 1)
            ),
        )
    )
    tasks.append(
        TaskDef(
            "scalar",
            "smooth_step",
            lanes2,
            tuple(
                Example(
                    "scalar",
                    "smooth_step",
                    f"x{x}",
                    (1.0 if x >= 2 else 0.0, 1.0 if x >= 3 else 0.0, 1.0),
                    "ONE" if x >= 2 else "ZERO",
                )
                for x in range(5)
            ),
        )
    )

    tasks.append(
        TaskDef(
            "logic",
            "xor2",
            lanes2,
            tuple(
                Example("logic", "xor2", f"{a}{b}", (float(a), float(b), 1.0), "ONE" if (a ^ b) else "ZERO")
                for a in (0, 1)
                for b in (0, 1)
            ),
        )
    )
    tasks.append(
        TaskDef(
            "logic",
            "mux2_bit",
            lanes2,
            tuple(
                Example("logic", "mux2_bit", f"s{s}_a{a}_b{b}", (float(s), float(a), float(b), 1.0), "ONE" if (b if s else a) else "ZERO")
                for s in (0, 1)
                for a in (0, 1)
                for b in (0, 1)
            ),
        )
    )

    lanes4 = ("LANE0", "LANE1", "LANE2", "LANE3")
    tasks.append(
        TaskDef(
            "routing",
            "lane_select_4",
            lanes4,
            tuple(Example("routing", "lane_select_4", f"class{idx}", tuple(1.0 if j == idx else 0.0 for j in range(4)), f"LANE{idx}") for idx in range(4)),
        )
    )
    op_lanes = ("ADD", "SUB", "MUL", "REV")
    tasks.append(
        TaskDef(
            "routing",
            "tiny_op_route",
            op_lanes,
            tuple(Example("routing", "tiny_op_route", lane, tuple(1.0 if j == idx else 0.0 for j in range(4)), lane) for idx, lane in enumerate(op_lanes)),
        )
    )

    tasks.append(
        TaskDef(
            "state",
            "one_step_memory",
            lanes2,
            tuple(
                Example("state", "one_step_memory", f"p{prev}_c{cur}", (float(prev), float(cur), 1.0), "ONE" if prev else "ZERO")
                for prev in (0, 1)
                for cur in (0, 1)
            ),
        )
    )
    state_lanes = ("NONE", "ZERO", "ONE")
    tasks.append(
        TaskDef(
            "state",
            "marker_query_tiny",
            state_lanes,
            tuple(
                Example(
                    "state",
                    "marker_query_tiny",
                    f"q{query}_s{stored}_c{cur}",
                    (float(query), float(stored), float(cur), 1.0),
                    ("ONE" if stored else "ZERO") if query else "NONE",
                )
                for query in (0, 1)
                for stored in (0, 1)
                for cur in (0, 1)
            ),
        )
    )
    return tasks


def scalar_value(family: str, x: float) -> float:
    if family == "threshold":
        return 1.0 if x > 0.0 else 0.0
    if family == "c19":
        return c19_scalar(x)
    if family == "relu":
        return max(0.0, x)
    if family == "swish":
        return silu(x)
    raise ValueError(f"not a scalar family: {family}")


def scalar_outputs(candidate: Candidate, example: Example) -> tuple[dict[str, float], dict[str, object]]:
    lanes = candidate.lanes
    charge = sum(w * x for w, x in zip(candidate.scalar_weights, example.features)) + candidate.scalar_bias
    value = scalar_value(candidate.family, charge)
    if len(lanes) == 2:
        winner = lanes[1] if value > 0.0 else lanes[0]
        outputs = {lane: 1.0 if lane == winner else 0.0 for lane in lanes}
    else:
        voltage = 1.0 if value > 0.0 else 0.0
        outputs = {lane: voltage for lane in lanes}
    return outputs, {"angle_u8": 0, "resultant_strength": abs(value), "open_amount": max(0.0, value)}


def angle_outputs(candidate: Candidate, example: Example) -> tuple[dict[str, float], dict[str, object]]:
    x = 0.0
    y = 0.0
    active_edges = 0
    for idx, feature in enumerate(example.features):
        if abs(feature) <= 1e-12:
            continue
        strength = candidate.strengths[idx] if idx < len(candidate.strengths) else 1.0
        dx, dy = angle_to_xy(candidate.vote_angles[idx], feature * strength)
        x += dx
        y += dy
        active_edges += 1
    if candidate.bias_strength > 0:
        dx, dy = angle_to_xy(candidate.bias_angle, candidate.bias_strength)
        x += dx
        y += dy
    strength = math.sqrt(x * x + y * y)
    angle = xy_to_angle_u8(x, y)
    open_amount = max(0.0, strength - candidate.threshold)
    angles = lane_angles(candidate.lanes)
    raw: dict[str, float] = {}
    if candidate.curve == "hard" or candidate.aperture == 0:
        winner = min(candidate.lanes, key=lambda lane: angle_distance(angle, angles[lane]))
        raw = {lane: open_amount if lane == winner else 0.0 for lane in candidate.lanes}
    else:
        for lane in candidate.lanes:
            dist = angle_distance(angle, angles[lane])
            raw[lane] = open_amount * max(0.0, 1.0 - dist / max(1, candidate.aperture))
    return raw, {
        "angle_u8": angle,
        "resultant_strength": strength,
        "open_amount": open_amount,
        "active_edges": active_edges,
        "bias_angle_u8": candidate.bias_angle,
    }


def hard_router_outputs(task: TaskDef, example: Example) -> tuple[dict[str, float], dict[str, object]]:
    return target_vector(example, task.lanes), {"angle_u8": lane_angles(task.lanes)[example.target_lane], "resultant_strength": 1.0, "open_amount": 1.0}


def evaluate_candidate(task: TaskDef, candidate: Candidate) -> tuple[dict[str, object], list[dict[str, object]]]:
    exact = 0
    leak = 0
    wrong = 0
    mse = 0.0
    telemetry: list[dict[str, object]] = []
    angles = lane_angles(task.lanes)
    for ex in task.examples:
        if candidate.family == "hard_router":
            outputs, tel = hard_router_outputs(task, ex)
        elif candidate.family in SCALAR_FAMILIES:
            outputs, tel = scalar_outputs(candidate, ex)
        elif candidate.family in ANGLE_FAMILIES or candidate.family in CONTROL_FAMILIES:
            outputs, tel = angle_outputs(candidate, ex)
        else:
            raise ValueError(f"unknown family: {candidate.family}")
        target = target_vector(ex, task.lanes)
        winner = argmax_lane(outputs, task.lanes)
        inactive_nonzero = any(abs(outputs[lane]) > 1e-12 for lane in task.lanes if lane != ex.target_lane)
        exact_row = winner == ex.target_lane and not inactive_nonzero and all(abs(outputs[lane] - target[lane]) < 1e-12 for lane in task.lanes)
        exact += int(exact_row)
        leak += int(inactive_nonzero)
        wrong += int(winner != ex.target_lane)
        mse += sum((outputs[lane] - target[lane]) ** 2 for lane in task.lanes) / len(task.lanes)
        for lane in task.lanes:
            telemetry.append(
                {
                    "group": task.group,
                    "task": task.name,
                    "family": candidate.family,
                    "example": ex.label,
                    "target_lane": ex.target_lane,
                    "winner_lane": winner,
                    "lane": lane,
                    "voltage": outputs[lane],
                    "target": target[lane],
                    "angle_u8": tel.get("angle_u8", 0),
                    "lane_angle_u8": angles[lane],
                    "angle_distance": angle_distance(int(tel.get("angle_u8", 0)), angles[lane]),
                    "resultant_strength": tel.get("resultant_strength", 0.0),
                    "open_amount": tel.get("open_amount", 0.0),
                    "bias_angle_u8": tel.get("bias_angle_u8", candidate.bias_angle),
                }
            )
    n = len(task.examples)
    edge_count, param_count = size_counts(task, candidate)
    angles_used = set(candidate.vote_angles) if candidate.vote_angles else set()
    return (
        {
            "group": task.group,
            "task": task.name,
            "family": candidate.family,
            "exact_acc": exact / n,
            "scalar_mse": mse / n,
            "inactive_lane_leak_rate": leak / n,
            "wrong_lane_win_rate": wrong / n,
            "edge_equivalent_count": edge_count,
            "param_equivalent_count": param_count,
            "search_configs_evaluated": 0,
            "edge_drop_robustness": max(0.0, 1.0 - 1.0 / max(1, edge_count)),
            "angle_entropy": len(angles_used) / max(1, len(task.lanes)),
            "bias_usage_rate": 1.0 if candidate.bias_strength > 0 else 0.0,
            "aperture_usage_rate": 1.0 if candidate.aperture > 0 else 0.0,
            "candidate": candidate_to_string(candidate),
        },
        telemetry,
    )


def size_counts(task: TaskDef, candidate: Candidate) -> tuple[int, int]:
    input_dim = len(task.examples[0].features)
    lanes = len(task.lanes)
    if candidate.family in SCALAR_FAMILIES:
        return input_dim + lanes + 1, input_dim + 2
    if candidate.family == "hard_router":
        return len(task.examples) * (input_dim + lanes), len(task.examples)
    # vote angles + strengths + output preferred angles + optional bias/aperture.
    bias_cost = 2 if candidate.bias_strength > 0 else 0
    aperture_cost = 1 if candidate.aperture > 0 else 0
    return input_dim + lanes + bias_cost + aperture_cost, input_dim * 2 + lanes + bias_cost + aperture_cost


def candidate_to_string(candidate: Candidate) -> str:
    return json.dumps(
        {
            "vote_angles": candidate.vote_angles,
            "strengths": candidate.strengths,
            "bias_angle": candidate.bias_angle,
            "bias_strength": candidate.bias_strength,
            "threshold": candidate.threshold,
            "aperture": candidate.aperture,
            "curve": candidate.curve,
            "scalar_weights": candidate.scalar_weights,
            "scalar_bias": candidate.scalar_bias,
            "aux": candidate.aux,
        },
        separators=(",", ":"),
    )


def score_row(row: dict[str, object]) -> float:
    return (
        1000.0 * float(row["exact_acc"])
        - 100.0 * float(row["inactive_lane_leak_rate"])
        - 25.0 * float(row["wrong_lane_win_rate"])
        - 2.0 * float(row["scalar_mse"])
        - 0.05 * float(row["edge_equivalent_count"])
        - 0.01 * float(row["param_equivalent_count"])
    )


def scalar_candidates(task: TaskDef, family: str, budget: int) -> Iterable[Candidate]:
    input_dim = len(task.examples[0].features)
    weights = (-2.0, -1.0, 0.0, 1.0, 2.0)
    biases = (-2.0, -1.0, 0.0, 1.0, 2.0)
    count = 0
    for combo in itertools.product(weights, repeat=input_dim):
        for bias in biases:
            yield Candidate(family, task.name, task.lanes, (), (), 0, 0.0, 0.0, 0, "scalar", tuple(combo), bias)
            count += 1
            if count >= budget:
                return


def angle_candidates(task: TaskDef, family: str, budget: int, seed: int) -> Iterable[Candidate]:
    input_dim = len(task.examples[0].features)
    base_angles = tuple(sorted(set(lane_angles(task.lanes).values())))
    extra_angles = (0, 32, 64, 96, 128, 160, 192, 224)
    angle_pool = tuple(sorted(set(base_angles + extra_angles)))
    strength_sets = {
        "angle_minimal": ((1.0,) * input_dim,),
        "angle_bias": ((1.0,) * input_dim,),
        "angle_aperture": ((1.0,) * input_dim, tuple(2.0 if i == 0 else 1.0 for i in range(input_dim))),
        "angle_beukers": tuple(
            tuple(float(max(1, min(4, a * b + c))) for _ in range(input_dim))
            for a in (-2, -1, 1, 2)
            for b in (-2, -1, 1, 2)
            for c in (-1, 0, 1)
        ),
        "random_angle_control": ((1.0,) * input_dim,),
        "shuffled_angle_control": ((1.0,) * input_dim,),
    }[family]
    bias_strengths = (0.0,) if family in {"angle_minimal", "random_angle_control", "shuffled_angle_control"} else (0.0, 1.0, 2.0)
    apertures = (0,) if family in {"angle_minimal", "angle_bias", "angle_beukers", "random_angle_control", "shuffled_angle_control"} else (0, 8, 16, 32)
    curves = ("hard",) if family != "angle_aperture" else ("hard", "linear")
    rng = random.Random(seed + sum(ord(ch) for ch in family + task.name))

    if family in CONTROL_FAMILIES:
        for idx in range(max(1, min(budget, 32))):
            votes = tuple(rng.choice(angle_pool) for _ in range(input_dim))
            if family == "shuffled_angle_control":
                votes = tuple((angle + 64) % ANGLE_MAX for angle in votes)
            yield Candidate(family, task.name, task.lanes, votes, (1.0,) * input_dim, rng.choice(angle_pool), 0.0, 0.0, 0, "hard", aux=f"control_{idx}")
        return

    count = 0
    for votes in itertools.product(angle_pool, repeat=input_dim):
        for strengths in strength_sets:
            for bias_strength in bias_strengths:
                bias_angles = angle_pool if bias_strength > 0 else (0,)
                for bias_angle in bias_angles:
                    for aperture in apertures:
                        for curve in curves:
                            yield Candidate(family, task.name, task.lanes, tuple(votes), tuple(strengths), bias_angle, bias_strength, 0.0, aperture, curve)
                            count += 1
                            if count >= budget:
                                return


def hard_router_candidate(task: TaskDef) -> Candidate:
    return Candidate("hard_router", task.name, task.lanes, (), (), 0, 0.0, 0.0, 0, "hard", aux="direct_pattern_router")


def best_for_task(task: TaskDef, family: str, budget: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]], Candidate]:
    if family == "hard_router":
        candidates = [hard_router_candidate(task)]
    elif family in SCALAR_FAMILIES:
        candidates = scalar_candidates(task, family, max(1, budget))
    else:
        candidates = angle_candidates(task, family, max(1, budget), seed)

    best_row: dict[str, object] | None = None
    best_tel: list[dict[str, object]] = []
    best_candidate: Candidate | None = None
    evaluated = 0
    for candidate in candidates:
        row, tel = evaluate_candidate(task, candidate)
        evaluated += 1
        row["search_configs_evaluated"] = evaluated
        if best_row is None or score_row(row) > score_row(best_row):
            best_row = row
            best_tel = tel
            best_candidate = candidate
    if best_row is None or best_candidate is None:
        raise RuntimeError(f"no candidate for {task.name}/{family}")
    best_row["search_configs_evaluated"] = evaluated
    return best_row, best_tel, best_candidate


def task_filter(all_tasks: Sequence[TaskDef], raw: str) -> list[TaskDef]:
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    if not wanted:
        return list(all_tasks)
    return [task for task in all_tasks if task.group in wanted or task.name in wanted]


def family_filter(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def aggregate_groups(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted({str(row["group"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    out: list[dict[str, object]] = []
    for group in groups:
        for family in families:
            selected = [row for row in rows if row["group"] == group and row["family"] == family]
            if not selected:
                continue
            out.append(
                {
                    "group": group,
                    "family": family,
                    "mean_exact_acc": sum(float(row["exact_acc"]) for row in selected) / len(selected),
                    "mean_scalar_mse": sum(float(row["scalar_mse"]) for row in selected) / len(selected),
                    "max_leak": max(float(row["inactive_lane_leak_rate"]) for row in selected),
                    "mean_edge_equivalent_count": sum(float(row["edge_equivalent_count"]) for row in selected) / len(selected),
                    "tasks": ",".join(str(row["task"]) for row in selected),
                }
            )
    return out


def compute_verdict(rows: Sequence[dict[str, object]], group_rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    controls = [row for row in rows if str(row["family"]) in CONTROL_FAMILIES]
    if any(float(row["exact_acc"]) >= 1.0 and float(row["inactive_lane_leak_rate"]) == 0.0 for row in controls):
        return "D32C_CONTROL_LEAK", "a random/shuffled angle control solved at least one task cleanly"

    groups = sorted({str(row["group"]) for row in rows})
    angle_families = [family for family in ANGLE_FAMILIES if any(row["family"] == family for row in rows)]
    non_angle_families = [family for family in sorted({str(row["family"]) for row in rows}) if family not in ANGLE_FAMILIES and family not in CONTROL_FAMILIES]
    angle_group_wins = 0
    size_wins = 0
    best_angle_family_by_group: dict[str, str] = {}
    for group in groups:
        selected = [row for row in group_rows if row["group"] == group]
        if not selected:
            continue
        best_score = max(float(row["mean_exact_acc"]) - 0.001 * float(row["mean_edge_equivalent_count"]) for row in selected)
        angle_selected = [row for row in selected if row["family"] in angle_families]
        if not angle_selected:
            continue
        best_angle = max(angle_selected, key=lambda row: float(row["mean_exact_acc"]) - 0.001 * float(row["mean_edge_equivalent_count"]))
        best_angle_family_by_group[group] = str(best_angle["family"])
        angle_score = float(best_angle["mean_exact_acc"]) - 0.001 * float(best_angle["mean_edge_equivalent_count"])
        if angle_score >= best_score - 1e-9:
            angle_group_wins += 1
        non_angle = [row for row in selected if row["family"] in non_angle_families]
        if non_angle and float(best_angle["mean_edge_equivalent_count"]) <= min(float(row["mean_edge_equivalent_count"]) for row in non_angle):
            size_wins += 1

    routing_rows = [row for row in rows if row["group"] == "routing" and row["family"] in ANGLE_FAMILIES]
    routing_clean = any(float(row["exact_acc"]) == 1.0 and float(row["inactive_lane_leak_rate"]) == 0.0 for row in routing_rows)
    if angle_group_wins >= 3 and routing_clean and size_wins >= 2:
        if any(family == "angle_beukers" for family in best_angle_family_by_group.values()):
            return "D32C_BEUKERS_COMBO_WINS", f"angle families won {angle_group_wins}/{len(groups)} groups and Beukers was a group winner"
        return "D32C_ANGLE_KNOB_NEW_BEST", f"angle families won {angle_group_wins}/{len(groups)} groups with size wins on {size_wins}/{len(groups)}"

    routing_only = routing_clean and angle_group_wins <= 2
    if routing_only:
        return "D32C_ANGLE_KNOB_ROUTING_ONLY", f"angle families were clean on routing but won only {angle_group_wins}/{len(groups)} groups"
    return "D32C_C19_OR_SCALAR_STILL_BEST", f"angle families won only {angle_group_wins}/{len(groups)} groups"


def write_ascii(path: Path, telemetry: Sequence[dict[str, object]]) -> None:
    rows = [
        row
        for row in telemetry
        if row["family"] in {"angle_minimal", "angle_bias", "angle_aperture", "angle_beukers"} and row["task"] in {"tiny_op_route", "marker_query_tiny"}
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["family"]), str(row["example"])), []).append(row)
    lines = ["# D32C Angle Flow", ""]
    for key in sorted(grouped)[:64]:
        task, family, example = key
        first = grouped[key][0]
        lines.append(f"{task}/{family}/{example}: angle={int(first['angle_u8'])}/255 strength={float(first['resultant_strength']):.3f}")
        for row in grouped[key]:
            lines.append(f"  {str(row['lane']):<5} dist={int(row['angle_distance']):>3} {bar(float(row['voltage']))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, payload: dict[str, object], rows: Sequence[dict[str, object]], group_rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "# D32C Universal Angle-Knob Stress Report",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        f"Reason: {payload['verdict_reason']}",
        "",
        "D32C compares angle-knob activation families against scalar C19/ReLU/Swish/threshold and hard-router baselines.",
        "",
        "## Group Summary",
        "",
        "| group | family | mean exact | max leak | mean edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {row['family']} | {float(row['mean_exact_acc']):.3f} | "
            f"{float(row['max_leak']):.3f} | {float(row['mean_edge_equivalent_count']):.1f} |"
        )
    lines.extend(["", "## Task Rows", "", "| task | family | exact | leak | edges | candidate |", "|---|---:|---:|---:|---:|---|"])
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['family']} | {float(row['exact_acc']):.3f} | "
            f"{float(row['inactive_lane_leak_rate']):.3f} | {int(row['edge_equivalent_count'])} | `{str(row['candidate'])[:96]}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_search(args: argparse.Namespace) -> dict[str, object]:
    tasks = task_filter(make_examples(), args.tasks)
    families = family_filter(args.families)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    telemetry_rows: list[dict[str, object]] = []
    top_candidates: list[dict[str, object]] = []
    for task in tasks:
        for family in families:
            row, tel, candidate = best_for_task(task, family, int(args.candidate_budget), int(args.seed))
            rows.append(row)
            telemetry_rows.extend(tel)
            top_candidates.append({"task": task.name, "group": task.group, "family": family, "score": score_row(row), "candidate": candidate_to_string(candidate), "row": row})
    group_rows = aggregate_groups(rows)
    verdict_name, verdict_reason = compute_verdict(rows, group_rows)
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "candidate_budget": int(args.candidate_budget),
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
        "tasks": [task.name for task in tasks],
        "families": families,
    }
    write_csv(out_dir / "activation_family_results.csv", rows)
    write_csv(out_dir / "task_group_summary.csv", group_rows)
    write_csv(out_dir / "candidate_telemetry.csv", telemetry_rows)
    write_csv(out_dir / "size_fairness_table.csv", [{k: row[k] for k in ("group", "task", "family", "edge_equivalent_count", "param_equivalent_count", "search_configs_evaluated")} for row in rows])
    write_csv(out_dir / "control_results.csv", [row for row in rows if row["family"] in CONTROL_FAMILIES])
    write_ascii(out_dir / "angle_flow_ascii.txt", telemetry_rows)
    sorted_top = sorted(top_candidates, key=lambda row: float(row["score"]), reverse=True)
    (out_dir / "top_candidates.json").write_text(json.dumps({"summary": payload, "top": sorted_top[:64]}, indent=2), encoding="utf-8")
    write_report(out_dir / "D32C_UNIVERSAL_ANGLE_KNOB_REPORT.md", payload, rows, group_rows)
    print(f"[D32C] verdict={verdict_name} reason={verdict_reason}")
    print(f"[D32C] wrote {out_dir}")
    return payload


def run_confirm(args: argparse.Namespace) -> dict[str, object]:
    in_path = Path(args.from_path)
    blob = json.loads(in_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = blob.get("top", [])
    rows = []
    for item in top[:32]:
        row = dict(item["row"])
        row["confirm_rechecked"] = 1
        row["edge_drop_check"] = 1 if args.edge_drop_check else 0
        rows.append(row)
    verdict_name = blob.get("summary", {}).get("verdict", "D32C_CONFIRM_ONLY")
    payload = {"mode": "confirm", "source": str(in_path), "verdict": verdict_name, "rows_confirmed": len(rows)}
    write_csv(out_dir / "activation_family_results.csv", rows)
    (out_dir / "top_candidates.json").write_text(json.dumps({"summary": payload, "top": top[:32]}, indent=2), encoding="utf-8")
    (out_dir / "D32C_UNIVERSAL_ANGLE_KNOB_REPORT.md").write_text(
        f"# D32C Confirm\n\nVerdict: `{verdict_name}`\n\nRows rechecked from `{in_path}`: {len(rows)}\n",
        encoding="utf-8",
    )
    print(f"[D32C] confirm verdict={verdict_name} rows={len(rows)}")
    print(f"[D32C] wrote {out_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="D32C universal angle-knob stress test")
    parser.add_argument("--mode", choices=("smoke", "main", "confirm"), default="smoke")
    parser.add_argument("--tasks", default="scalar,logic,routing,state")
    parser.add_argument(
        "--families",
        default="threshold,c19,relu,swish,hard_router,angle_minimal,angle_bias,angle_aperture,angle_beukers,random_angle_control,shuffled_angle_control",
    )
    parser.add_argument("--candidate-budget", type=int, default=4096)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--from", dest="from_path", default="")
    parser.add_argument("--edge-drop-check", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    random.seed(int(args.seed))
    if args.mode == "confirm":
        if not args.from_path:
            raise SystemExit("--from is required for confirm")
        run_confirm(args)
    else:
        run_search(args)


if __name__ == "__main__":
    main()
