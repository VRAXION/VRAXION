#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_authority_graph_developmental_search as dev
import run_authority_graph_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_GRAMMAR_V2_PILOT.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-grammar-v2-pilot"
SUMMARY_NAME = "authority_graph_grammar_v2_pilot_summary.json"
DEFAULT_ARMS = [
    "route_gate_hub_grammar",
    "grammar_v2_graph",
    "damaged_hand_seeded_50",
    "hand_seeded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick pilot for distilled Authority Graph Grammar v2.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--search-train-samples", type=int, default=96)
    parser.add_argument("--validation-samples", type=int, default=96)
    parser.add_argument("--final-test-samples", type=int, default=256)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--mutation-scale", type=float, default=0.18)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-runtime-hours", type=float, default=3.0)
    parser.add_argument("--decay", type=float, default=0.35)
    parser.add_argument("--fitness-mode", choices=("coarse", "authority_shaped"), default="authority_shaped")
    parser.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = sorted(set(args.arms) - set(dev.ALL_ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {', '.join(unknown)}")
    if args.smoke:
        args.seeds = 1
        args.steps = 3
        args.search_train_samples = 32
        args.validation_samples = 32
        args.final_test_samples = 64
        args.generations = 3
        args.population_size = 4
        args.checkpoint_every = 1
        args.max_runtime_hours = 0.25
    return args


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def mean_field(aggregate: dict[str, Any], arm: str, field: str) -> float:
    value = aggregate.get(args_fitness_key(aggregate), {}).get(arm, {}).get("final_test", {}).get(field, {}).get("mean")
    return float(value) if value is not None else 0.0


def args_fitness_key(aggregate: dict[str, Any]) -> str:
    return next(iter(aggregate.keys())) if aggregate else "authority_shaped"


def get_mean(aggregate: dict[str, Any], arm: str, field: str) -> float:
    mode = args_fitness_key(aggregate)
    value = aggregate.get(mode, {}).get(arm, {}).get("final_test", {}).get(field, {}).get("mean")
    return float(value) if value is not None else 0.0


def get_success(aggregate: dict[str, Any], arm: str) -> float:
    mode = args_fitness_key(aggregate)
    return float(aggregate.get(mode, {}).get(arm, {}).get("success_rate", 0.0))


def pilot_verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    v1 = "route_gate_hub_grammar"
    v2 = "grammar_v2_graph"
    v2_auth = get_mean(aggregate, v2, "authority_refraction_score")
    v1_auth = get_mean(aggregate, v1, "authority_refraction_score")
    v2_temporal = get_mean(aggregate, v2, "temporal_order_accuracy")
    v1_temporal = get_mean(aggregate, v1, "temporal_order_accuracy")
    v2_acc = get_mean(aggregate, v2, "overall_accuracy")
    v1_acc = get_mean(aggregate, v1, "overall_accuracy")
    return {
        "grammar_v2_beats_v1": (
            v2_auth > v1_auth + 0.05
            and v2_temporal >= v1_temporal
            and v2_acc >= v1_acc - 0.03
        ),
        "grammar_v2_improves_authority": v2_auth > v1_auth + 0.05,
        "grammar_v2_improves_temporal_order": v2_temporal > v1_temporal + 0.10,
        "grammar_v2_reaches_success": get_success(aggregate, v2) > 0.0,
        "grammar_v2_closes_gap_to_damaged_hand": (
            v2_auth >= 0.70 * get_mean(aggregate, "damaged_hand_seeded_50", "authority_refraction_score")
            and v2_acc >= 0.85 * get_mean(aggregate, "damaged_hand_seeded_50", "overall_accuracy")
        ),
        "hand_seeded_still_upper_bound": get_success(aggregate, "hand_seeded") > get_success(aggregate, v2),
        "readout_policy_explicitly_documented": True,
        "final_verdict_uses_final_test": True,
    }


def round_floats(value: Any) -> Any:
    return dev.round_floats(value)


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def write_report(summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    mode = summary["config"]["fitness_mode"]
    arms = summary["config"]["arms_completed"]
    lines = [
        "# Authority Graph Grammar v2 Pilot",
        "",
        "## Goal",
        "",
        "Test whether the wiring rules distilled from the hand-seeded and damaged-success graphs improve the failed Grammar v1 prior.",
        "",
        "This is a quick pilot, not a long developmental search. It uses the same toy tasks and does not add new semantic concepts, neural layers, or backprop.",
        "",
        "## Readout Policy",
        "",
        "The pilot makes the previous readout caveat explicit: static outputs use `route_state` readout. The route state is the formal authority readout port for static refraction tasks. `readout_positive` and `readout_negative` nodes remain present for future explicit-edge readout work, but this pilot does not silently rely on them.",
        "",
        "Temporal order tasks likewise read from `temporal_route` state.",
        "",
        "## Grammar v2 Scaffold",
        "",
        "- one route group per frame with recurrence",
        "- guaranteed group-level token-to-route candidate coverage",
        "- shared token->hub and hub->route bridge",
        "- frame gates applied early to routes",
        "- full route-level suppressor matrix",
        "- subject/verb/object temporal role channel",
        "- route-state authority readout policy",
        "- redundant weak candidate paths for mutation",
        "",
        "Grammar v2 does not wire exact task solutions like `dog+bite->danger` by name. It gives broad coverage paths and leaves signs/gains to mutation.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Final-Test Results",
        "",
        "| Arm | Success | Strong Success | Accuracy | Latent | Multi | Temporal | Authority | Wrong Frame | Recurrence Drop | Route Spec | Inactive | Edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        item = aggregate.get(mode, {}).get(arm)
        if not item:
            continue
        final = item["final_test"]
        lines.append(
            f"| `{arm}` | `{fmt(item['success_rate'])}` | `{fmt(item['strong_success_rate'])}` "
            f"| `{fmt(final['overall_accuracy']['mean'])}` "
            f"| `{fmt(final['latent_refraction_accuracy']['mean'])}` "
            f"| `{fmt(final['multi_aspect_accuracy']['mean'])}` "
            f"| `{fmt(final['temporal_order_accuracy']['mean'])}` "
            f"| `{fmt(final['authority_refraction_score']['mean'])}` "
            f"| `{fmt(final['wrong_frame_drop']['mean'])}` "
            f"| `{fmt(final['recurrence_drop']['mean'])}` "
            f"| `{fmt(final['route_specialization']['mean'])}` "
            f"| `{fmt(final['inactive_influence']['mean'])}` "
            f"| `{fmt(final['edge_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Fitness Generalization",
        "",
        "| Arm | Train Fitness | Validation Fitness | Final-Test Fitness | Generations |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in arms:
        item = aggregate.get(mode, {}).get(arm)
        if not item:
            continue
        lines.append(
            f"| `{arm}` | `{fmt(item['train_fitness']['mean'])}` "
            f"| `{fmt(item['validation_fitness']['mean'])}` "
            f"| `{fmt(item['final_test_fitness']['mean'])}` "
            f"| `{fmt(item['generations_completed']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Leakage Audit",
        "",
        "| Arm | Grammar Audit Pass | Direct Token->Readout Edges |",
        "|---|---:|---:|",
    ])
    for arm in arms:
        item = aggregate.get(mode, {}).get(arm)
        if not item:
            continue
        audit = item["leakage_audit"]
        lines.append(
            f"| `{arm}` | `{audit['all_grammar_runs_pass']}` "
            f"| `{fmt(audit['direct_token_to_readout_edges']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Runtime Notes",
        "",
        f"- runtime seconds: `{fmt(summary['runtime_seconds'])}`",
        f"- interrupted by wall clock: `{summary['interrupted_by_wall_clock']}`",
        f"- completed records: `{len(summary['records'])}`",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + args.max_runtime_hours * 3600.0
    hand_edge_count = pilot.build_hand_seeded_graph(args.decay).edge_count()
    records: list[dict[str, Any]] = []
    interrupted = False

    for seed in range(args.seeds):
        splits = dev.split_datasets(args, seed)
        for arm in args.arms:
            if time.time() >= deadline:
                interrupted = True
                break
            print(f"[grammar-v2-pilot] arm={arm} seed={seed}", flush=True)
            record = dev.evolve_arm(
                arm,
                splits,
                args=args,
                seed=seed,
                hand_edge_count=hand_edge_count,
                fitness_mode=args.fitness_mode,
                out_dir=args.out_dir,
                deadline=deadline,
            )
            records.append(record)
            aggregate = dev.aggregate_records(records)
            partial = {
                "config": {
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "search_train_samples": args.search_train_samples,
                    "validation_samples": args.validation_samples,
                    "final_test_samples": args.final_test_samples,
                    "generations": args.generations,
                    "population_size": args.population_size,
                    "mutation_scale": args.mutation_scale,
                    "checkpoint_every": args.checkpoint_every,
                    "max_runtime_hours": args.max_runtime_hours,
                    "decay": args.decay,
                    "fitness_mode": args.fitness_mode,
                    "readout_policy": "route_state",
                    "arms_requested": args.arms,
                    "arms_completed": sorted({item["arm"] for item in records}),
                    "smoke": args.smoke,
                    "completed": False,
                    "started_unix": start,
                },
                "records": records,
                "aggregate": aggregate,
                "verdict": pilot_verdict(aggregate),
                "runtime_seconds": time.time() - start,
                "interrupted_by_wall_clock": interrupted,
                "platform": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
            }
            (args.out_dir / SUMMARY_NAME).write_text(
                json.dumps(round_floats(partial), indent=2) + "\n",
                encoding="utf-8",
            )
        if interrupted:
            break

    aggregate = dev.aggregate_records(records)
    summary = {
        "config": {
            "seeds": args.seeds,
            "steps": args.steps,
            "search_train_samples": args.search_train_samples,
            "validation_samples": args.validation_samples,
            "final_test_samples": args.final_test_samples,
            "generations": args.generations,
            "population_size": args.population_size,
            "mutation_scale": args.mutation_scale,
            "checkpoint_every": args.checkpoint_every,
            "max_runtime_hours": args.max_runtime_hours,
            "decay": args.decay,
            "fitness_mode": args.fitness_mode,
            "readout_policy": "route_state",
            "arms_requested": args.arms,
            "arms_completed": sorted({item["arm"] for item in records}),
            "smoke": args.smoke,
            "completed": not interrupted and len(records) == args.seeds * len(args.arms),
            "started_unix": start,
        },
        "records": records,
        "aggregate": aggregate,
        "verdict": pilot_verdict(aggregate),
        "runtime_seconds": time.time() - start,
        "interrupted_by_wall_clock": interrupted,
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (args.out_dir / SUMMARY_NAME).write_text(json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps({
        "verdict": summary["verdict"],
        "json": str(args.out_dir / SUMMARY_NAME),
        "report": str(REPORT_PATH),
        "interrupted_by_wall_clock": interrupted,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
