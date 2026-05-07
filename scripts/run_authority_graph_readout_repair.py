#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import run_authority_graph_guided_pruning as guided
import run_authority_graph_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_EXPLICIT_READOUT_REPAIR.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-readout-repair"
SUMMARY_NAME = "authority_graph_readout_repair_summary.json"

ARMS = [
    "hand_seeded_route_state_reference",
    "hand_seeded_explicit_readout_untrained",
    "hand_seeded_train_readout_only",
    "hand_seeded_train_readout_plus_route_gains",
    "damaged_hand_50_train_readout_plus_route_gains",
    "random_graph_train_readout_plus_route_gains",
    "grammar_v2_train_readout_plus_route_gains",
]

SUCCESS = {
    "overall_accuracy": 0.90,
    "temporal_order_accuracy": 0.90,
    "output_influence_authority_score": 0.25,
    "wrong_frame_drop": 0.20,
}


@dataclass
class TrainMask:
    edge: torch.Tensor
    bias: torch.Tensor
    train_frame_gate: bool
    train_suppressor: bool
    train_decay: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair explicit readout edges for the authority graph toy.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--final-test-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-runtime-hours", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arms", type=lambda value: value.split(","), default=list(ARMS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.train_samples = 64
        args.validation_samples = 64
        args.final_test_samples = 128
        args.steps = 3
        args.epochs = 4
        args.checkpoint_every = 2
        args.max_runtime_hours = 0.25
        args.arms = [
            "hand_seeded_route_state_reference",
            "hand_seeded_explicit_readout_untrained",
            "hand_seeded_train_readout_only",
            "hand_seeded_train_readout_plus_route_gains",
            "random_graph_train_readout_plus_route_gains",
        ]
    return args


def split_datasets(samples: int, seed: int) -> dict[str, list[Any]]:
    return pilot.make_datasets(samples, seed)


def build_matched_random_spec(seed: int, hand_spec: guided.GraphSpec) -> guided.GraphSpec:
    rng = np.random.default_rng(seed)
    nodes, types = guided.base_nodes()
    readout_edges = sum(1 for edge in hand_spec.edges if edge.edge_type == "route->readout")
    internal_budget = len(hand_spec.edges) - readout_edges
    edges: list[guided.EdgeSpec] = []
    attempts = 0
    while len(guided.dedupe_edges(edges)) < internal_budget and attempts < internal_budget * 80:
        attempts += 1
        source = str(rng.choice(nodes))
        target = str(rng.choice(nodes))
        if source == target:
            continue
        if types.get(target) == "readout":
            continue
        if types.get(source) == "token_input" and types.get(target) == "readout":
            continue
        weight = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.75))
        guided.add_edge(edges, source, target, weight, guided.classify_edge(types, source, target))
        edges = guided.dedupe_edges(edges)
    guided.add_readout_edges(edges, scale=float(rng.uniform(0.65, 1.10)))
    return guided.GraphSpec(
        name="matched_random_explicit_readout",
        nodes=nodes,
        types=types,
        edges=guided.dedupe_edges(edges),
        bias={node: float(rng.normal(0.0, 0.08)) for node in nodes if types[node] in {"frame_route", "shared_hub"}},
        frame_gate={frame: float(rng.uniform(0.55, 1.15)) for frame in pilot.FRAMES},
        suppressor_strength=float(rng.uniform(0.65, 1.25)),
        decay=0.35,
    )


def labels_static(examples: list[pilot.StaticExample]) -> torch.Tensor:
    return torch.tensor([float(ex.label) for ex in examples], dtype=torch.float32)


def labels_temporal(examples: list[pilot.TemporalExample]) -> torch.Tensor:
    return torch.tensor([float(ex.label) for ex in examples], dtype=torch.float32)


def binary_task_loss(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], steps: int) -> torch.Tensor:
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    losses = [
        F.binary_cross_entropy_with_logits(model.forward_static(latent, steps=steps)["logit"], labels_static(latent)),
        F.binary_cross_entropy_with_logits(model.forward_static(multi, steps=steps)["logit"], labels_static(multi)),
        F.binary_cross_entropy_with_logits(model.forward_temporal(temporal, steps=steps)["logit"], labels_temporal(temporal)),
    ]
    return sum(losses) / len(losses)


def make_train_mask(model: guided.DifferentiableAuthorityGraph, mode: str) -> TrainMask:
    edge = torch.zeros_like(model.edge_weights, dtype=torch.float32)
    if mode == "readout_only":
        allowed = {"route->readout"}
        train_frame_gate = False
        train_suppressor = False
    elif mode == "readout_plus_route_gains":
        allowed = {"route->readout", "route recurrent"}
        train_frame_gate = True
        train_suppressor = True
    else:
        raise ValueError(f"unknown train mask mode: {mode}")
    for index, edge_type in enumerate(model.edge_types):
        if edge_type in allowed:
            edge[index] = 1.0
    bias = torch.zeros_like(model.bias, dtype=torch.float32)
    for node in ("readout_positive", "readout_negative"):
        bias[model.idx[node]] = 1.0
    return TrainMask(edge=edge, bias=bias, train_frame_gate=train_frame_gate, train_suppressor=train_suppressor)


def apply_train_mask(model: guided.DifferentiableAuthorityGraph, mask: TrainMask) -> None:
    if model.edge_weights.grad is not None:
        model.edge_weights.grad *= mask.edge
    if model.bias.grad is not None:
        model.bias.grad *= mask.bias
    if model.frame_gate.grad is not None and not mask.train_frame_gate:
        model.frame_gate.grad.zero_()
    if model.suppressor_strength.grad is not None and not mask.train_suppressor:
        model.suppressor_strength.grad.zero_()
    if model.decay_logit.grad is not None and not mask.train_decay:
        model.decay_logit.grad.zero_()


def repair_loss(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], steps: int, mask: TrainMask) -> torch.Tensor:
    task = binary_task_loss(model, datasets, steps)
    edge_reg = (model.edge_weights * mask.edge).pow(2).mean()
    bias_reg = (model.bias * mask.bias).pow(2).mean()
    return task + 0.0005 * edge_reg + 0.0002 * bias_reg


def train_repair(
    model: guided.DifferentiableAuthorityGraph,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    arm: str,
    mode: str,
    deadline: float,
) -> tuple[guided.DifferentiableAuthorityGraph, dict[str, Any]]:
    torch.manual_seed(seed)
    mask = make_train_mask(model, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    best = model.clone_frozen()
    best_val = evaluate_explicit(best, validation, steps=args.steps)
    best_score = selection_score(best_val)
    history = []
    interrupted = False
    for epoch in range(1, args.epochs + 1):
        if time.time() >= deadline:
            interrupted = True
            break
        opt.zero_grad(set_to_none=True)
        loss = repair_loss(model, train, args.steps, mask)
        loss.backward()
        apply_train_mask(model, mask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        with torch.no_grad():
            model.edge_weights.clamp_(-3.5, 3.5)
            model.bias.clamp_(-3.5, 3.5)
            model.frame_gate.clamp_(0.0, 2.5)
            model.suppressor_strength.clamp_(0.0, 2.5)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            val = evaluate_explicit(model, validation, steps=args.steps)
            score = selection_score(val)
            history.append({"epoch": epoch, "validation": val, "selection_score": score})
            print(
                "[readout-repair] "
                f"{arm} seed={seed} epoch={epoch}/{args.epochs} "
                f"val_score={score:.4f} val_acc={val['overall_accuracy']:.4f} "
                f"out_auth={val['output_influence_authority_score']:.4f}",
                flush=True,
            )
            if score > best_score:
                best = model.clone_frozen()
                best_val = val
                best_score = score
    return best, {
        "mode": mode,
        "loss_type": "binary_cross_entropy_with_logits",
        "loss_reason": "The current graph exposes one binary positive-vs-negative readout logit for all toy tasks.",
        "best_validation": best_val,
        "best_selection_score": best_score,
        "history": history,
        "interrupted": interrupted,
    }


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["overall_accuracy"]
        + 0.50 * metrics["output_influence_authority_score"]
        + 0.25 * metrics["wrong_frame_drop"]
        + 0.20 * metrics["route_specialization"]
    )


@torch.no_grad()
def route_state_authority_metrics(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
) -> dict[str, float]:
    rng = np.random.default_rng(2026)
    by_frame = {}
    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        active_group = pilot.ACTIVE_GROUP_BY_FRAME[frame]
        route_index = pilot.FRAMES.index(frame)
        base_routes = model.forward_static(frame_examples, steps=steps)["routes"][:, route_index].cpu().numpy()
        group_delta = {}
        for group in pilot.GROUP_FIELDS:
            swapped = []
            for ex in frame_examples:
                donor = frame_examples[int(rng.integers(0, len(frame_examples)))].obs
                swapped.append(
                    pilot.StaticExample(
                        obs=guided.swap_observation(ex.obs, donor, group),
                        frame=frame,
                        label=ex.label,
                    )
                )
            swapped_routes = model.forward_static(swapped, steps=steps)["routes"][:, route_index].cpu().numpy()
            group_delta[group] = float(np.mean(np.abs(base_routes - swapped_routes)))
        inactive = max(value for group, value in group_delta.items() if group != active_group)
        by_frame[frame] = {
            "active": group_delta[active_group],
            "inactive": inactive,
            "refraction": group_delta[active_group] - inactive,
        }
    return {
        "route_state_authority_score": float(np.mean([item["refraction"] for item in by_frame.values()])),
        "route_state_active_group_influence": float(np.mean([item["active"] for item in by_frame.values()])),
        "route_state_inactive_group_influence": float(np.mean([item["inactive"] for item in by_frame.values()])),
    }


@torch.no_grad()
def evaluate_explicit(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], *, steps: int) -> dict[str, Any]:
    base = guided.evaluate_model(model, datasets, steps=steps, controls=True)
    route_state = route_state_authority_metrics(
        model,
        datasets["latent_refraction_small"] + datasets["multi_aspect_small"],
        steps=steps,
    )
    metrics = dict(base)
    metrics["output_influence_authority_score"] = metrics.pop("authority_refraction_score")
    metrics["authority_refraction_score"] = metrics["output_influence_authority_score"]
    metrics.update(route_state)
    metrics["inactive_influence"] = metrics["inactive_group_influence"]
    metrics["route_node_count"] = len(pilot.route_nodes()) + 1
    metrics["readout_node_count"] = 2
    metrics["trainable_readout_parameter_count"] = trainable_readout_parameter_count(model)
    return metrics


def trainable_readout_parameter_count(model: guided.DifferentiableAuthorityGraph) -> int:
    route_readout_edges = sum(
        1 for edge, mask in zip(model.spec.edges, model.edge_mask.detach().cpu().numpy()) if edge.edge_type == "route->readout" and mask > 0.0
    )
    return int(route_readout_edges + 2)


def evaluate_route_state_reference(graph: pilot.AuthorityGraph, datasets: dict[str, list[Any]], *, steps: int, seed: int) -> dict[str, Any]:
    raw = pilot.evaluate_graph(graph, datasets, steps=steps, seed=seed)
    return {
        "overall_accuracy": raw["accuracy"],
        "latent_refraction_accuracy": raw["latent_refraction_small_accuracy"],
        "multi_aspect_accuracy": raw["multi_aspect_small_accuracy"],
        "temporal_order_accuracy": raw["temporal_order_contrast_small_accuracy"],
        "route_state_authority_score": raw["refraction_index_final"],
        "route_state_active_group_influence": float(np.mean([
            raw["latent_authority"]["active_group_influence"],
            raw["multi_aspect_authority"]["active_group_influence"],
        ])),
        "route_state_inactive_group_influence": float(np.mean([
            raw["latent_authority"]["inactive_group_influence"],
            raw["multi_aspect_authority"]["inactive_group_influence"],
        ])),
        "output_influence_authority_score": None,
        "authority_refraction_score": None,
        "active_group_influence": None,
        "inactive_group_influence": None,
        "inactive_influence": None,
        "wrong_frame_accuracy": raw["wrong_frame_accuracy"],
        "wrong_frame_drop": raw["wrong_frame_drop"],
        "recurrence_drop": raw["no_recurrence_drop"],
        "route_specialization": None,
        "edge_count": int(np.count_nonzero(graph.w)),
        "node_count": len(graph.nodes),
        "route_node_count": len(pilot.route_nodes()) + 1,
        "readout_node_count": 2,
        "trainable_readout_parameter_count": 0,
        "final_loss": None,
    }


def evaluate_record(
    arm: str,
    model_or_graph: Any,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    final_test: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    metadata: dict[str, Any] | None = None,
    route_state_reference: bool = False,
) -> dict[str, Any]:
    if route_state_reference:
        train_metrics = evaluate_route_state_reference(model_or_graph, train, steps=args.steps, seed=seed + 1)
        validation_metrics = evaluate_route_state_reference(model_or_graph, validation, steps=args.steps, seed=seed + 2)
        final_metrics = evaluate_route_state_reference(model_or_graph, final_test, steps=args.steps, seed=seed + 3)
    else:
        train_metrics = evaluate_explicit(model_or_graph, train, steps=args.steps)
        validation_metrics = evaluate_explicit(model_or_graph, validation, steps=args.steps)
        final_metrics = evaluate_explicit(model_or_graph, final_test, steps=args.steps)
    return {
        "arm": arm,
        "seed": seed,
        "train": train_metrics,
        "validation": validation_metrics,
        "final_test": final_metrics,
        "metadata": metadata or {},
        "success": is_success(final_metrics),
    }


def is_success(metrics: dict[str, Any]) -> bool:
    return (
        metrics["overall_accuracy"] >= SUCCESS["overall_accuracy"]
        and metrics["temporal_order_accuracy"] >= SUCCESS["temporal_order_accuracy"]
        and (metrics.get("output_influence_authority_score") or 0.0) >= SUCCESS["output_influence_authority_score"]
        and metrics["wrong_frame_drop"] >= SUCCESS["wrong_frame_drop"]
    )


def run_seed(args: argparse.Namespace, seed: int, deadline: float) -> list[dict[str, Any]]:
    train = split_datasets(args.train_samples, seed + 10_000)
    validation = split_datasets(args.validation_samples, seed + 20_000)
    final_test = split_datasets(args.final_test_samples, seed + 30_000)
    hand_graph = pilot.build_hand_seeded_graph(0.35)
    hand_spec = guided.build_hand_spec()
    damaged_spec = guided.damage_spec(hand_spec, 0.50, seed + 40_000)
    random_spec = build_matched_random_spec(seed + 50_000, hand_spec)
    grammar_spec = guided.build_overcomplete_spec(seed + 60_000, name="grammar_v2_explicit_readout")
    records: list[dict[str, Any]] = []

    def add_reference() -> None:
        if "hand_seeded_route_state_reference" in args.arms:
            print(f"[readout-repair] evaluating hand_seeded_route_state_reference seed={seed}", flush=True)
            records.append(
                evaluate_record(
                    "hand_seeded_route_state_reference",
                    hand_graph,
                    train,
                    validation,
                    final_test,
                    args=args,
                    seed=seed,
                    metadata={"readout_policy": "route_state_reference_only"},
                    route_state_reference=True,
                )
            )

    def add_explicit(arm: str, model: guided.DifferentiableAuthorityGraph, metadata: dict[str, Any] | None = None) -> None:
        if arm in args.arms:
            print(f"[readout-repair] evaluating {arm} seed={seed}", flush=True)
            records.append(evaluate_record(arm, model, train, validation, final_test, args=args, seed=seed, metadata=metadata))

    add_reference()
    add_explicit(
        "hand_seeded_explicit_readout_untrained",
        guided.DifferentiableAuthorityGraph(hand_spec, trainable=False),
        {"readout_policy": "explicit_edges_untrained"},
    )

    repair_specs = {
        "hand_seeded_train_readout_only": (hand_spec, "readout_only", seed + 70_000),
        "hand_seeded_train_readout_plus_route_gains": (hand_spec, "readout_plus_route_gains", seed + 80_000),
        "damaged_hand_50_train_readout_plus_route_gains": (damaged_spec, "readout_plus_route_gains", seed + 90_000),
        "random_graph_train_readout_plus_route_gains": (random_spec, "readout_plus_route_gains", seed + 100_000),
        "grammar_v2_train_readout_plus_route_gains": (grammar_spec, "readout_plus_route_gains", seed + 110_000),
    }
    for arm, (spec, mode, arm_seed) in repair_specs.items():
        if arm not in args.arms:
            continue
        model = guided.DifferentiableAuthorityGraph(spec, trainable=True)
        trained, info = train_repair(
            model,
            train,
            validation,
            args=args,
            seed=arm_seed,
            arm=arm,
            mode=mode,
            deadline=deadline,
        )
        add_explicit(arm, trained, {"training": info, "readout_policy": "explicit_edges"})
    return records


def numeric_summary(values: list[Any]) -> dict[str, float]:
    nums = [float(value) for value in values if isinstance(value, (int, float)) and value is not None]
    if not nums:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(nums)), "std": float(np.std(nums))}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in records}):
        rows = [row for row in records if row["arm"] == arm]
        arm_data: dict[str, Any] = {
            "runs": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
        }
        for split in ("train", "validation", "final_test"):
            keys = sorted({key for row in rows for key, value in row[split].items() if isinstance(value, (int, float)) or value is None})
            arm_data[split] = {key: numeric_summary([row[split].get(key) for row in rows]) for key in keys}
        out[arm] = arm_data
    return out


def metric_mean(aggregate_data: dict[str, Any], arm: str, key: str) -> float:
    value = aggregate_data.get(arm, {}).get("final_test", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def success_rate(aggregate_data: dict[str, Any], arm: str) -> float:
    return float(aggregate_data.get(arm, {}).get("success_rate", 0.0))


def verdict(aggregate_data: dict[str, Any]) -> dict[str, bool]:
    ref_acc = metric_mean(aggregate_data, "hand_seeded_route_state_reference", "overall_accuracy")
    ref_route_auth = metric_mean(aggregate_data, "hand_seeded_route_state_reference", "route_state_authority_score")
    explicit_acc = metric_mean(aggregate_data, "hand_seeded_explicit_readout_untrained", "overall_accuracy")
    explicit_auth = metric_mean(aggregate_data, "hand_seeded_explicit_readout_untrained", "output_influence_authority_score")
    readout_acc = metric_mean(aggregate_data, "hand_seeded_train_readout_only", "overall_accuracy")
    readout_auth = metric_mean(aggregate_data, "hand_seeded_train_readout_only", "output_influence_authority_score")
    plus_acc = metric_mean(aggregate_data, "hand_seeded_train_readout_plus_route_gains", "overall_accuracy")
    plus_auth = metric_mean(aggregate_data, "hand_seeded_train_readout_plus_route_gains", "output_influence_authority_score")
    damaged_acc = metric_mean(aggregate_data, "damaged_hand_50_train_readout_plus_route_gains", "overall_accuracy")
    damaged_auth = metric_mean(aggregate_data, "damaged_hand_50_train_readout_plus_route_gains", "output_influence_authority_score")
    random_acc = metric_mean(aggregate_data, "random_graph_train_readout_plus_route_gains", "overall_accuracy")
    random_auth = metric_mean(aggregate_data, "random_graph_train_readout_plus_route_gains", "output_influence_authority_score")
    plus_success = success_rate(aggregate_data, "hand_seeded_train_readout_plus_route_gains") > 0.0
    readout_success = success_rate(aggregate_data, "hand_seeded_train_readout_only") > 0.0
    random_success = success_rate(aggregate_data, "random_graph_train_readout_plus_route_gains") > 0.0
    return {
        "route_state_shortcut_confirmed": (ref_acc - explicit_acc) >= 0.20 or (ref_route_auth - explicit_auth) >= 0.20,
        "explicit_readout_repair_successful": plus_success or readout_success,
        "readout_only_sufficient": readout_success or (abs(readout_acc - plus_acc) <= 0.05 and abs(readout_auth - plus_auth) <= 0.05),
        "route_gain_training_required": plus_success and not readout_success,
        "hand_topology_still_useful_under_explicit_readout": (plus_acc - random_acc) >= 0.15 or (plus_auth - random_auth) >= 0.15,
        "damaged_hand_repair_under_explicit_readout": damaged_acc >= 0.80 and damaged_auth >= 0.15,
        "random_graph_control_fails": not random_success,
        "final_verdict_prioritizes_output_influence": True,
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def config_dict(args: argparse.Namespace, *, completed: bool, arms_completed: list[str]) -> dict[str, Any]:
    return {
        "seeds": args.seeds,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "final_test_samples": args.final_test_samples,
        "steps": args.steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint_every": args.checkpoint_every,
        "max_runtime_hours": args.max_runtime_hours,
        "torch_threads": args.torch_threads,
        "arms_requested": args.arms,
        "arms_completed": arms_completed,
        "loss_type": "binary_cross_entropy_with_logits",
        "loss_reason": "The current authority graph exposes one binary positive-vs-negative readout logit, not a multiclass output vector.",
        "final_test_used_for_selection": False,
        "completed": completed,
        "smoke": args.smoke,
    }


def write_report(summary: dict[str, Any]) -> None:
    aggregate_data = summary["aggregate"]
    arms = summary["config"]["arms_completed"]
    lines = [
        "# Authority Graph Explicit Readout Repair",
        "",
        "## Goal",
        "",
        "Test whether the hand-seeded authority graph can recover its route-state behavior through explicit route-to-readout edges.",
        "",
        "The route-state readout is included only as a diagnostic reference. Verdicts prioritize explicit output influence.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Final-Test Results",
        "",
        "| Arm | Success | Accuracy | Latent | Multi | Temporal | Output Authority | Route-State Authority | Wrong Frame | Recurrence | Route Spec | Edges | Readout Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        if arm not in aggregate_data:
            continue
        item = aggregate_data[arm]
        final = item["final_test"]
        lines.append(
            f"| `{arm}` | `{fmt(item['success_rate'])}` "
            f"| `{fmt(final['overall_accuracy']['mean'])}` "
            f"| `{fmt(final['latent_refraction_accuracy']['mean'])}` "
            f"| `{fmt(final['multi_aspect_accuracy']['mean'])}` "
            f"| `{fmt(final['temporal_order_accuracy']['mean'])}` "
            f"| `{fmt(final['output_influence_authority_score']['mean'])}` "
            f"| `{fmt(final['route_state_authority_score']['mean'])}` "
            f"| `{fmt(final['wrong_frame_drop']['mean'])}` "
            f"| `{fmt(final['recurrence_drop']['mean'])}` "
            f"| `{fmt(final['route_specialization']['mean'])}` "
            f"| `{fmt(final['edge_count']['mean'])}` "
            f"| `{fmt(final['trainable_readout_parameter_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Train / Validation / Final-Test Gap",
        "",
        "| Arm | Train Acc | Val Acc | Final Acc | Train Output Auth | Val Output Auth | Final Output Auth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in arms:
        if arm not in aggregate_data:
            continue
        item = aggregate_data[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item['train']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['validation']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['final_test']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['train']['output_influence_authority_score']['mean'])}` "
            f"| `{fmt(item['validation']['output_influence_authority_score']['mean'])}` "
            f"| `{fmt(item['final_test']['output_influence_authority_score']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation Notes",
        "",
        "- BCE is used because the current graph exposes one binary positive-vs-negative readout logit.",
        "- Route-state authority is diagnostic only; explicit output-influence authority is the mechanism metric for verdicts.",
        "- If readout-only fails but plus-route-gains works, interpret that as route-state calibration being required for explicit readout.",
        "- Random graph readout capacity is matched to hand-seeded readout capacity; only internal topology differs.",
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


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rounded = round_floats(summary)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(rounded, indent=2) + "\n", encoding="utf-8")
    write_report(rounded)


def platform_dict() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + args.max_runtime_hours * 3600.0
    records: list[dict[str, Any]] = []
    interrupted = False
    for seed in range(args.seeds):
        if time.time() >= deadline:
            interrupted = True
            break
        print(f"[readout-repair] seed={seed}", flush=True)
        records.extend(run_seed(args, seed, deadline))
        aggregate_data = aggregate(records)
        arms_completed = sorted({row["arm"] for row in records})
        partial = {
            "config": config_dict(args, completed=False, arms_completed=arms_completed),
            "records": records,
            "aggregate": aggregate_data,
            "verdict": verdict(aggregate_data),
            "runtime_seconds": time.time() - start,
            "interrupted_by_wall_clock": interrupted,
            "platform": platform_dict(),
        }
        write_summary(args.out_dir, partial)
    aggregate_data = aggregate(records)
    arms_completed = sorted({row["arm"] for row in records})
    summary = {
        "config": config_dict(args, completed=not interrupted, arms_completed=arms_completed),
        "records": records,
        "aggregate": aggregate_data,
        "verdict": verdict(aggregate_data),
        "runtime_seconds": time.time() - start,
        "interrupted_by_wall_clock": interrupted,
        "platform": platform_dict(),
    }
    write_summary(args.out_dir, summary)
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "json": str(args.out_dir / SUMMARY_NAME),
                "report": str(REPORT_PATH),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
