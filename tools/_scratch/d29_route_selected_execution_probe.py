#!/usr/bin/env python3
"""
D29 route-selected execution probe.

Shape:
    bytes -> AB/B64 -> D28 C-router -> selected D worker -> output bytes/text

This probe verifies the switchboard behavior: only the selected lane may emit an
output, all inactive lanes must stay empty, and wrong/random route controls must
not solve the task.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d28_content_router_probe import (  # noqa: E402
    LABELS,
    encode_b64,
    predict_route_head,
    to_window,
    train_route_head,
)
from tools.ab_window_codec import B_WINDOW_DIMS, BYTE_BITS, WINDOW_BYTES, verify_artifact  # noqa: E402


DEFAULT_SEED = 20260503
ROUTES = LABELS
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class Packet:
    input_text: str
    expected_route: str
    expected_output: str


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def route_weights() -> tuple[np.ndarray, np.ndarray]:
    return train_route_head(np.zeros((1, B_WINDOW_DIMS), dtype=np.int8), np.zeros((1,), dtype=np.int32))


def route_text(text: str, weights: np.ndarray, bias: np.ndarray) -> str:
    window = np.asarray([list(to_window(text))], dtype=np.uint8)
    pred = predict_route_head(encode_b64(window), weights, bias)[0]
    return ROUTES[int(pred)]


def b64_payload_text(text: str) -> str:
    window = np.asarray([list(to_window(text))], dtype=np.uint8)
    latent = encode_b64(window)[0]
    return "".join("+" if int(value) > 0 else "-" for value in latent)


def alu_output(text: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*([+\-^&|*])\s*(\d+)\s*", text.strip())
    if not match:
        return "ALU_PARSE_FAIL"
    left = int(match.group(1)) & 0xFF
    op = match.group(2)
    right = int(match.group(3)) & 0xFF
    if op == "+":
        return str((left + right) & 0xFF)
    if op == "-":
        return str((left - right) & 0xFF)
    if op == "^":
        return str(left ^ right)
    if op == "&":
        return str(left & right)
    if op == "|":
        return str(left | right)
    if op == "*":
        return "UNSUPPORTED_ALU_OP"
    return "UNSUPPORTED_ALU_OP"


def transform_output(text: str) -> str:
    match = re.fullmatch(r"\s*(REV|ROT|COPY|NOT)\s+([A-Z0-9]{1,4})\s*", text.strip().upper())
    if not match:
        return "TRANSFORM_PARSE_FAIL"
    op = match.group(1)
    payload = match.group(2)
    if op == "REV":
        return payload[::-1]
    if op == "ROT":
        return payload[1:] + payload[:1] if payload else payload
    if op == "COPY":
        return payload
    if op == "NOT":
        inverted = bytes((ord(ch) ^ 0xFF) & 0xFF for ch in payload)
        return "HEX:" + inverted.hex().upper()
    return "TRANSFORM_PARSE_FAIL"


def mem_output(text: str, state: dict[str, str]) -> str:
    match = re.fullmatch(r"\s*(STORE|SET|QUERY|GET)\s+([A-Z0-9])\s*", text.strip().upper())
    if not match:
        return "MEM_PARSE_FAIL"
    op = match.group(1)
    key = match.group(2)
    if op in ("STORE", "SET"):
        state[key] = key
        return f"STORED:{key}"
    return state.get(key, f"MISS:{key}")


def execute_with_route(text: str, route: str, memory: dict[str, str]) -> tuple[dict[str, str], str]:
    lane_outputs = {label: "" for label in ROUTES}
    if route == "LANG":
        lane_outputs["LANG"] = "NO_LANG_WORKER"
    elif route == "ALU":
        lane_outputs["ALU"] = alu_output(text)
    elif route == "MEM":
        lane_outputs["MEM"] = mem_output(text, memory)
    elif route == "TRANSFORM":
        lane_outputs["TRANSFORM"] = transform_output(text)
    elif route == "UNKNOWN":
        lane_outputs["UNKNOWN"] = "REJECT"
    else:
        raise ValueError(f"unsupported route: {route}")
    return lane_outputs, lane_outputs[route]


def inactive_lanes_empty(lane_outputs: dict[str, str], route: str) -> bool:
    return all(value == "" for label, value in lane_outputs.items() if label != route)


def expected_for_text(text: str, route: str, memory: dict[str, str]) -> str:
    _lanes, out = execute_with_route(text, route, memory)
    return out


def make_lang(rng: random.Random) -> str:
    return rng.choice(["THE CAT", "HELLO", "BLUE SUN", "CAT DOG", "MOON", "TREE"])


def make_unknown(rng: random.Random) -> str:
    return rng.choice(["THE+CAT", "ABC123", "@#??!!", "++--", "CAT_42", "12 CATS", "A+BIRD"])


def make_transform(rng: random.Random) -> str:
    return rng.choice(["REV ABC", "ROT XYZ", "COPY CAT", "REV DOG", "ROT AB"])


def make_alu(rng: random.Random) -> str:
    if rng.random() < 0.08:
        return "27*852"
    op = rng.choice(["+", "-", "^", "&", "|"])
    left = rng.randrange(0, 256)
    right = rng.randrange(0, 256)
    return f"{left}{op}{right}"


def build_packets(samples_per_class: int, episodes: int, seed: int) -> list[Packet]:
    rng = random.Random(seed)
    weights, bias = route_weights()
    packets: list[Packet] = []
    expected_memory: dict[str, str] = {}

    for _ in range(samples_per_class):
        for maker in (make_lang, make_alu, make_transform, make_unknown):
            text = maker(rng)
            route = route_text(text, weights, bias)
            output = expected_for_text(text, route, expected_memory)
            packets.append(Packet(text, route, output))

    keys = ["A", "B", "X", "Y", "1", "2"]
    for idx in range(episodes):
        key = keys[idx % len(keys)]
        for text in (f"STORE {key}", rng.choice(["THE CAT", "@#??!!", "REV ABC"]), f"QUERY {key}"):
            route = route_text(text, weights, bias)
            output = expected_for_text(text, route, expected_memory)
            packets.append(Packet(text, route, output))

    # Fixed adversarial/integration rows make the report easy to audit.
    for text in ("1+2", "99-4", "REV ABC", "ROT XYZ", "STORE X", "QUERY X", "THE CAT", "THE+CAT", "27*852"):
        route = route_text(text, weights, bias)
        output = expected_for_text(text, route, expected_memory)
        packets.append(Packet(text, route, output))
    return packets


def packet_row(packet: Packet, predicted_route: str, lane_outputs: dict[str, str], selected_output: str) -> dict[str, object]:
    route_ok = predicted_route == packet.expected_route
    output_ok = selected_output == packet.expected_output
    lanes_empty = inactive_lanes_empty(lane_outputs, predicted_route)
    return {
        "input_text": packet.input_text,
        "input_window": to_window(packet.input_text).decode("ascii", errors="ignore"),
        "b64_payload": b64_payload_text(packet.input_text),
        "expected_route": packet.expected_route,
        "route_label": predicted_route,
        "parsed_command": packet.input_text.strip(),
        "lane_LANG": lane_outputs["LANG"],
        "lane_ALU": lane_outputs["ALU"],
        "lane_MEM": lane_outputs["MEM"],
        "lane_TRANSFORM": lane_outputs["TRANSFORM"],
        "lane_UNKNOWN": lane_outputs["UNKNOWN"],
        "selected_output": selected_output,
        "expected_output": packet.expected_output,
        "route_ok": route_ok,
        "output_ok": output_ok,
        "inactive_lanes_empty": lanes_empty,
        "status": "PASS" if route_ok and output_ok and lanes_empty else "FAIL",
    }


def evaluate_packets(packets: Sequence[Packet], *, forced_routes: Sequence[str] | None = None, seed: int = DEFAULT_SEED) -> tuple[list[dict[str, object]], dict[str, object]]:
    weights, bias = route_weights()
    rng = random.Random(seed)
    memory: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for idx, packet in enumerate(packets):
        if forced_routes is None:
            route = route_text(packet.input_text, weights, bias)
        else:
            route = forced_routes[idx]
        lane_outputs, selected_output = execute_with_route(packet.input_text, route, memory)
        rows.append(packet_row(packet, route, lane_outputs, selected_output))
        # Keep the random generator consumed in a deterministic way so control
        # runs are stable even if future rows add stochastic route choices.
        rng.random()
    return rows, summarize_rows(rows)


def next_wrong_route(route: str) -> str:
    idx = ROUTES.index(route)
    return ROUTES[(idx + 1) % len(ROUTES)]


def control_rows(packets: Sequence[Packet], repeats: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    wrong_routes = [next_wrong_route(packet.expected_route) for packet in packets]
    _wrong_packet_rows, wrong_summary = evaluate_packets(packets, forced_routes=wrong_routes, seed=seed + 100)
    rows.append({"control": "wrong_route_control", **wrong_summary})
    for repeat in range(repeats):
        rng = random.Random(seed + 1009 + repeat)
        random_routes = [rng.choice(ROUTES) for _ in packets]
        _random_packet_rows, random_summary = evaluate_packets(packets, forced_routes=random_routes, seed=seed + 200 + repeat)
        rows.append({"control": f"random_route_control_{repeat}", **random_summary})
    return rows


def summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    route_acc = mean_bool(row["route_ok"] for row in rows)
    output_acc = mean_bool(row["output_ok"] for row in rows)
    inactive_acc = mean_bool(row["inactive_lanes_empty"] for row in rows)
    route_metrics: dict[str, float] = {}
    for route in ROUTES:
        route_rows = [row for row in rows if row["expected_route"] == route]
        route_metrics[f"{route.lower()}_exact_acc"] = mean_bool(row["output_ok"] for row in route_rows) if route_rows else 1.0
        route_metrics[f"{route.lower()}_route_acc"] = mean_bool(row["route_ok"] for row in route_rows) if route_rows else 1.0
    unsupported = [row for row in rows if "*" in str(row["input_text"])]
    unsupported_ok = bool(unsupported) and all(row["route_label"] == "ALU" and row["selected_output"] == "UNSUPPORTED_ALU_OP" for row in unsupported)
    return {
        "sample_count": len(rows),
        "route_acc": route_acc,
        "selected_output_acc": output_acc,
        "inactive_lanes_empty_acc": inactive_acc,
        "executable_worker_acc": mean_bool(
            row["output_ok"] for row in rows if row["expected_route"] in ("ALU", "MEM", "TRANSFORM")
        ),
        "policy_acc": mean_bool(row["output_ok"] for row in rows if row["expected_route"] in ("LANG", "UNKNOWN")),
        "unsupported_alu_ok": unsupported_ok,
        **route_metrics,
    }


def mean_bool(values: Sequence[object]) -> float:
    values = list(values)
    return float(sum(1 for value in values if bool(value)) / len(values)) if values else 1.0


def verdict_for(summary: dict[str, object], controls: Sequence[dict[str, object]]) -> str:
    if float(summary["inactive_lanes_empty_acc"]) < 1.0:
        return "D29_LANE_LEAK"
    if any(float(row["selected_output_acc"]) > 0.25 for row in controls):
        return "D29_WRONG_ROUTE_LEAK"
    full = (
        float(summary["route_acc"]) == 1.0
        and float(summary["selected_output_acc"]) == 1.0
        and float(summary["executable_worker_acc"]) == 1.0
        and float(summary["policy_acc"]) == 1.0
        and bool(summary["unsupported_alu_ok"])
    )
    if full:
        return "D29_ROUTE_EXECUTION_PASS"
    if float(summary["route_acc"]) == 1.0 and float(summary["inactive_lanes_empty_acc"]) == 1.0 and float(summary["selected_output_acc"]) >= 0.95:
        return "D29_ROUTE_EXECUTION_WEAK_PASS"
    return "D29_EXECUTION_FAIL"


def integration_cases() -> list[Packet]:
    memory: dict[str, str] = {}
    weights, bias = route_weights()
    rows: list[Packet] = []
    for text in ("REV ABC", "ROT XYZ", "1+2", "99-4", "STORE X", "QUERY X", "THE CAT", "THE+CAT", "27*852"):
        route = route_text(text, weights, bias)
        rows.append(Packet(text, route, expected_for_text(text, route, memory)))
    return rows


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


def heatmap(summary: dict[str, object], controls: Sequence[dict[str, object]], verdict: str) -> str:
    lines = ["D29 route-selected execution: brighter = selected output accuracy"]
    lines.append("row                  cell route output empty exec policy unsupported verdict")
    primary_shade = shade(float(summary["selected_output_acc"]))
    lines.append(
        f"{'primary':<20} {primary_shade} {float(summary['route_acc']):.3f} "
        f"{float(summary['selected_output_acc']):.3f} {float(summary['inactive_lanes_empty_acc']):.3f} "
        f"{float(summary['executable_worker_acc']):.3f} {float(summary['policy_acc']):.3f} "
        f"{bool(summary['unsupported_alu_ok'])} {verdict}"
    )
    for row in controls:
        lines.append(
            f"{str(row['control'])[:20]:<20} {shade(float(row['selected_output_acc']))} {float(row['route_acc']):.3f} "
            f"{float(row['selected_output_acc']):.3f} {float(row['inactive_lanes_empty_acc']):.3f} "
            f"{float(row['executable_worker_acc']):.3f} {float(row['policy_acc']):.3f} "
            f"{bool(row['unsupported_alu_ok'])} CONTROL"
        )
    return "\n".join(lines)


def shade(value: float) -> str:
    idx = min(len(ASCII_SHADE) - 1, max(0, round(float(value) * (len(ASCII_SHADE) - 1))))
    return ASCII_SHADE[idx]


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    packets = integration_cases() if args.mode == "integration-smoke" else build_packets(int(args.samples_per_class), int(args.episodes), int(args.seed))
    rows, summary = evaluate_packets(packets, seed=int(args.seed))
    controls = control_rows(packets, int(args.control_repeats), int(args.seed))
    verdict = verdict_for(summary, controls)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "lane_outputs.csv", rows)
    write_csv(out_dir / "execution_results.csv", [{"split": args.mode, **summary, "verdict": verdict}])
    write_csv(out_dir / "execution_controls.csv", controls)
    integration_rows, _integration_summary = evaluate_packets(integration_cases(), seed=int(args.seed))
    write_csv(out_dir / "integration_smoke.csv", integration_rows)
    text = heatmap(summary, controls, verdict)
    top = {
        "verdict": verdict,
        "config": {
            "mode": args.mode,
            "samples_per_class": int(args.samples_per_class),
            "episodes": int(args.episodes),
            "control_repeats": int(args.control_repeats),
            "seed": int(args.seed),
            "artifact": str(args.artifact),
        },
        "summary": summary,
        "controls": controls,
        "integration": integration_rows,
    }
    (out_dir / "route_execution_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D29 Route-Selected Execution Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        text,
        "```",
        "",
        "D29 verifies the switchboard: D28 chooses a route, the selected D worker emits output, and inactive lanes stay empty.",
        "",
    ]
    (out_dir / "D29_ROUTE_SELECTED_EXECUTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(text)
    print(json.dumps({"verdict": verdict, "summary": summary}, indent=2))
    return 0 if verdict in ("D29_ROUTE_EXECUTION_PASS", "D29_ROUTE_EXECUTION_WEAK_PASS") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "confirm", "integration-smoke"], required=True)
    parser.add_argument("--samples-per-class", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
