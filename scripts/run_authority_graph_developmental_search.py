#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import run_authority_graph_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_DEVELOPMENTAL_SEARCH.md"
QUICK_REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRADIENT_SEARCH_QUICK_TEST.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-developmental-search"
QUICK_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-gradient-search-quick"

DEFAULT_ARMS = [
    "random_graph",
    "route_grammar_graph",
    "route_gate_grammar_graph",
    "route_gate_recurrence_grammar",
    "route_gate_hub_grammar",
    "damaged_hand_seeded_50",
    "hand_seeded",
]
EVOLVED_ARMS = {arm for arm in DEFAULT_ARMS if arm != "hand_seeded"}
GRAMMAR_ARMS = {
    "route_grammar_graph",
    "route_gate_grammar_graph",
    "route_gate_recurrence_grammar",
    "route_gate_hub_grammar",
}
SUCCESS_THRESHOLDS = {
    "accuracy": 0.90,
    "temporal_order_accuracy": 0.90,
    "authority_refraction_score": 0.25,
    "wrong_frame_drop": 0.20,
}
STRONG_SUCCESS_THRESHOLDS = {
    "accuracy": 0.95,
    "task_accuracy": 0.90,
    "authority_refraction_score": 0.30,
}


@dataclass
class Individual:
    graph: pilot.AuthorityGraph
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    train_fitness: float
    validation_fitness: float
    generation: int
    mutation_type: str
    fitness_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Developmental search for frame-gated authority graphs.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--search-train-samples", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--final-test-samples", type=int, default=512)
    parser.add_argument("--generations", type=int, default=2_000)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--mutation-scale", type=float, default=0.18)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-runtime-hours", type=float, default=11.5)
    parser.add_argument("--decay", type=float, default=0.35)
    parser.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    parser.add_argument("--fitness-mode", choices=("coarse", "authority_shaped", "ab_compare"), default="coarse")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.fitness_mode == "ab_compare" and args.out_dir == DEFAULT_OUT:
        args.out_dir = QUICK_OUT
    args.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = sorted(set(args.arms) - set(DEFAULT_ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {', '.join(unknown)}")
    if args.smoke:
        args.seeds = 1
        args.search_train_samples = 32
        args.validation_samples = 32
        args.final_test_samples = 64
        args.generations = 3
        args.population_size = 4
        args.checkpoint_every = 1
        args.max_runtime_hours = 0.25
    return args


def split_datasets(args: argparse.Namespace, seed: int) -> dict[str, dict[str, list[Any]]]:
    return {
        "search_train": pilot.make_datasets(args.search_train_samples, seed + 10_000),
        "validation": pilot.make_datasets(args.validation_samples, seed + 20_000),
        "final_test": pilot.make_datasets(args.final_test_samples, seed + 30_000),
    }


def mean_inactive_influence(metrics: dict[str, Any]) -> float:
    values = []
    for key in ("latent_authority", "multi_aspect_authority"):
        value = metrics.get(key, {}).get("inactive_group_influence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def route_specialization_metric(graph: pilot.AuthorityGraph, datasets: dict[str, list[Any]], *, steps: int) -> float:
    examples = list(datasets["latent_refraction_small"]) + list(datasets["multi_aspect_small"])
    if not examples:
        return 0.0
    values = []
    for ex in examples:
        label_sign = 1.0 if ex.label else -1.0
        correct = label_sign * pilot.static_score(graph, ex.obs, ex.frame, steps=steps)
        wrong = max(
            label_sign * pilot.static_score(graph, ex.obs, other, steps=steps)
            for other in pilot.FRAMES
            if other != ex.frame
        )
        values.append(float(np.tanh(correct - wrong)))
    return float(np.mean(values))


def raw_metrics(
    metrics: dict[str, Any],
    hand_edge_count: int,
    *,
    route_specialization: float = 0.0,
) -> dict[str, Any]:
    return {
        "overall_accuracy": float(metrics["accuracy"]),
        "latent_refraction_accuracy": float(metrics["latent_refraction_small_accuracy"]),
        "multi_aspect_accuracy": float(metrics["multi_aspect_small_accuracy"]),
        "temporal_order_accuracy": float(metrics["temporal_order_contrast_small_accuracy"]),
        "authority_refraction_score": float(metrics["authority_switch_score"]),
        "refraction_index_final": float(metrics["refraction_index_final"]),
        "wrong_frame_drop": float(metrics["wrong_frame_drop"]),
        "no_frame_gate_drop": float(metrics["no_frame_gate_drop"]),
        "no_recurrence_drop": float(metrics["no_recurrence_drop"]),
        "recurrence_drop": float(metrics["no_recurrence_drop"]),
        "route_specialization": float(route_specialization),
        "no_suppressor_drop": float(metrics["no_suppressor_drop"]),
        "inactive_influence": mean_inactive_influence(metrics),
        "edge_count": int(metrics["edge_count"]),
        "node_count": int(metrics["node_count"]),
        "edge_count_penalty": float(metrics["edge_count"]) / max(1.0, float(hand_edge_count)),
    }


def fitness_from_raw(raw: dict[str, Any], fitness_mode: str) -> float:
    if fitness_mode == "coarse":
        return (
            1.0 * raw["overall_accuracy"]
            + 0.5 * raw["authority_refraction_score"]
            + 0.3 * raw["temporal_order_accuracy"]
            + 0.2 * max(0.0, raw["wrong_frame_drop"])
            - 0.05 * raw["inactive_influence"]
            - 0.02 * raw["edge_count_penalty"]
        )
    if fitness_mode == "authority_shaped":
        return (
            1.0 * raw["overall_accuracy"]
            + 0.5 * raw["authority_refraction_score"]
            + 0.3 * raw["wrong_frame_drop"]
            + 0.25 * raw["recurrence_drop"]
            + 0.25 * raw["route_specialization"]
            + 0.3 * raw["temporal_order_accuracy"]
            - 0.05 * raw["inactive_influence"]
            - 0.02 * raw["edge_count_penalty"]
        )
    raise ValueError(fitness_mode)


def developmental_fitness(metrics: dict[str, Any], hand_edge_count: int, fitness_mode: str) -> float:
    raw = raw_metrics(metrics, hand_edge_count)
    return fitness_from_raw(raw, fitness_mode)


def developmental_fitness_from_raw(raw: dict[str, Any], fitness_mode: str) -> float:
    return (
        fitness_from_raw(raw, fitness_mode)
    )


def evaluate_split(
    graph: pilot.AuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    steps: int,
    seed: int,
    hand_edge_count: int,
    fitness_mode: str,
) -> dict[str, Any]:
    metrics = pilot.evaluate_graph(graph, datasets, steps=steps, seed=seed)
    route_specialization = route_specialization_metric(graph, datasets, steps=steps)
    raw = raw_metrics(metrics, hand_edge_count, route_specialization=route_specialization)
    return {
        "raw": raw,
        "fitness": developmental_fitness_from_raw(raw, fitness_mode),
        "metrics": metrics,
    }


def success_flags(raw: dict[str, Any]) -> dict[str, bool]:
    success = (
        raw["overall_accuracy"] >= SUCCESS_THRESHOLDS["accuracy"]
        and raw["temporal_order_accuracy"] >= SUCCESS_THRESHOLDS["temporal_order_accuracy"]
        and raw["authority_refraction_score"] >= SUCCESS_THRESHOLDS["authority_refraction_score"]
        and raw["wrong_frame_drop"] >= SUCCESS_THRESHOLDS["wrong_frame_drop"]
    )
    strong = (
        raw["overall_accuracy"] >= STRONG_SUCCESS_THRESHOLDS["accuracy"]
        and raw["latent_refraction_accuracy"] >= STRONG_SUCCESS_THRESHOLDS["task_accuracy"]
        and raw["multi_aspect_accuracy"] >= STRONG_SUCCESS_THRESHOLDS["task_accuracy"]
        and raw["temporal_order_accuracy"] >= STRONG_SUCCESS_THRESHOLDS["task_accuracy"]
        and raw["authority_refraction_score"] >= STRONG_SUCCESS_THRESHOLDS["authority_refraction_score"]
    )
    return {"success": success, "strong_success": strong}


def set_route_bias(graph: pilot.AuthorityGraph, bias: float) -> None:
    graph.route_bias = float(bias)
    for route in pilot.route_nodes():
        graph.bias[graph.idx[route]] = graph.route_bias
    graph.bias[graph.idx["temporal_route"]] = max(-0.35, graph.route_bias * 0.20)


def zero_direct_token_readout_edges(graph: pilot.AuthorityGraph) -> None:
    for target_name, target in graph.idx.items():
        if graph.types.get(target_name) != "readout":
            continue
        for source_name, source in graph.idx.items():
            if graph.types.get(source_name) == "token_input":
                graph.w[target, source] = 0.0


def graph_setup(
    graph: pilot.AuthorityGraph,
    rng: np.random.Generator,
    *,
    frame_gates: bool,
    recurrence: bool,
) -> pilot.AuthorityGraph:
    graph.frame_gate = float(rng.uniform(0.35, 1.25)) if frame_gates else 0.0
    graph.suppressor_strength = float(rng.uniform(0.0, 1.0))
    graph.competition = float(rng.uniform(0.15, 0.80))
    graph.decay = float(rng.uniform(0.18, 0.55)) if recurrence else 0.0
    set_route_bias(graph, float(rng.uniform(-1.70, -0.45)))
    zero_direct_token_readout_edges(graph)
    return graph


def add_random_token_route_edges(graph: pilot.AuthorityGraph, rng: np.random.Generator, density: float) -> None:
    for route in pilot.route_nodes():
        for token in pilot.TOKENS:
            if rng.random() < density:
                pilot.add_edge(graph, route, f"tok_{token}", float(rng.normal(0.0, 0.35)))


def add_random_temporal_edges(graph: pilot.AuthorityGraph, rng: np.random.Generator, density: float) -> None:
    temporal_sources = [
        node
        for node in pilot.temporal_nodes()
        if node != "temporal_route" and node in graph.idx
    ]
    for source in temporal_sources:
        if rng.random() < density:
            pilot.add_edge(graph, "temporal_route", source, float(rng.normal(0.0, 0.45)))


def add_route_recurrence(graph: pilot.AuthorityGraph, rng: np.random.Generator) -> None:
    for route in pilot.route_nodes() + ["temporal_route"]:
        pilot.add_edge(graph, route, route, float(rng.uniform(0.04, 0.22)))
    for node in [item for item in pilot.temporal_nodes() if item != "temporal_route"]:
        if rng.random() < 0.40:
            pilot.add_edge(graph, node, node, float(rng.uniform(0.05, 0.30)))


def add_random_hub_edges(graph: pilot.AuthorityGraph, rng: np.random.Generator) -> None:
    hubs = ["shared_actor_hub", "shared_action_hub", "shared_context_hub"]
    for token in pilot.TOKENS:
        for hub in hubs:
            if rng.random() < 0.22:
                pilot.add_edge(graph, hub, f"tok_{token}", float(rng.normal(0.0, 0.28)))
    for route in pilot.route_nodes():
        for hub in hubs:
            if rng.random() < 0.70:
                pilot.add_edge(graph, route, hub, float(rng.normal(0.0, 0.30)))


def build_route_grammar_graph(decay: float, seed: int, *, frame_gates: bool, recurrence: bool, hubs: bool) -> pilot.AuthorityGraph:
    rng = np.random.default_rng(seed)
    graph = pilot.build_empty_graph(decay)
    graph_setup(graph, rng, frame_gates=frame_gates, recurrence=recurrence)
    add_random_token_route_edges(graph, rng, density=0.22)
    if recurrence:
        add_route_recurrence(graph, rng)
        add_random_temporal_edges(graph, rng, density=0.35)
    else:
        add_random_temporal_edges(graph, rng, density=0.18)
    if hubs:
        add_random_hub_edges(graph, rng)
    return graph


def build_seed_graph(arm: str, args: argparse.Namespace, seed: int, hand_edge_count: int) -> pilot.AuthorityGraph:
    if arm == "random_graph":
        return pilot.build_random_graph(args.decay, seed, hand_edge_count)
    if arm == "route_grammar_graph":
        return build_route_grammar_graph(args.decay, seed, frame_gates=False, recurrence=False, hubs=False)
    if arm == "route_gate_grammar_graph":
        return build_route_grammar_graph(args.decay, seed, frame_gates=True, recurrence=False, hubs=False)
    if arm == "route_gate_recurrence_grammar":
        return build_route_grammar_graph(args.decay, seed, frame_gates=True, recurrence=True, hubs=False)
    if arm == "route_gate_hub_grammar":
        return build_route_grammar_graph(args.decay, seed, frame_gates=True, recurrence=True, hubs=True)
    if arm == "damaged_hand_seeded_50":
        return pilot.damage_graph(pilot.build_hand_seeded_graph(args.decay), 0.50, seed)
    if arm == "hand_seeded":
        return pilot.build_hand_seeded_graph(args.decay)
    raise ValueError(arm)


def all_mutable_edges(graph: pilot.AuthorityGraph) -> list[tuple[int, int]]:
    forbidden_targets = {
        graph.idx[name]
        for name, typ in graph.types.items()
        if typ == "readout"
    }
    return [
        (target, source)
        for target in range(len(graph.nodes))
        for source in range(len(graph.nodes))
        if target != source and target not in forbidden_targets
    ]


def nonzero_edges(graph: pilot.AuthorityGraph) -> list[tuple[int, int]]:
    return [(int(t), int(s)) for t, s in np.argwhere(np.abs(graph.w) > 1.0e-9)]


def mutate_developmental(
    graph: pilot.AuthorityGraph,
    rng: np.random.Generator,
    scale: float,
) -> tuple[pilot.AuthorityGraph, str]:
    candidate = graph.clone()
    mutation_type = str(rng.choice([
        "edge_add",
        "edge_remove",
        "edge_sign_flip",
        "edge_gain_perturb",
        "recurrent_decay",
        "frame_gate_strength",
        "add_route_recurrent_edge",
        "add_or_remove_suppressor_edge",
        "reassign_edge_to_route",
        "add_hub_to_route_edge",
        "remove_direct_shortcut",
    ]))
    edges = nonzero_edges(candidate)
    if mutation_type == "edge_add":
        target, source = all_mutable_edges(candidate)[int(rng.integers(0, len(all_mutable_edges(candidate))))]
        candidate.w[target, source] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.85))
    elif mutation_type == "edge_remove" and edges:
        target, source = edges[int(rng.integers(0, len(edges)))]
        candidate.w[target, source] = 0.0
    elif mutation_type == "edge_sign_flip" and edges:
        target, source = edges[int(rng.integers(0, len(edges)))]
        candidate.w[target, source] *= -1.0
    elif mutation_type == "edge_gain_perturb" and edges:
        target, source = edges[int(rng.integers(0, len(edges)))]
        candidate.w[target, source] = float(np.clip(candidate.w[target, source] + rng.normal(0.0, scale), -1.8, 1.8))
    elif mutation_type == "recurrent_decay":
        candidate.decay = float(np.clip(candidate.decay + rng.normal(0.0, scale * 0.25), 0.0, 0.95))
    elif mutation_type == "frame_gate_strength":
        candidate.frame_gate = float(np.clip(candidate.frame_gate + rng.normal(0.0, scale), 0.0, 1.8))
    elif mutation_type == "add_route_recurrent_edge":
        route = str(rng.choice(pilot.route_nodes() + ["temporal_route"]))
        candidate.w[candidate.idx[route], candidate.idx[route]] = float(
            np.clip(candidate.w[candidate.idx[route], candidate.idx[route]] + rng.uniform(0.03, 0.22), -1.8, 1.8)
        )
    elif mutation_type == "add_or_remove_suppressor_edge":
        frame = str(rng.choice(pilot.FRAMES))
        other = str(rng.choice(pilot.FRAMES))
        target = candidate.idx[pilot.FRAME_BY_ROUTE[frame]]
        source = candidate.idx[f"suppress_{other}"]
        if abs(candidate.w[target, source]) > 1.0e-9 and rng.random() < 0.5:
            candidate.w[target, source] = 0.0
        else:
            sign = -1.0 if frame == other else float(rng.choice([-1.0, 1.0]))
            candidate.w[target, source] = sign * float(rng.uniform(0.05, 1.2))
    elif mutation_type == "reassign_edge_to_route":
        route_indices = {candidate.idx[route] for route in pilot.route_nodes()}
        route_edges = [(target, source) for target, source in edges if target in route_indices]
        if route_edges:
            target, source = route_edges[int(rng.integers(0, len(route_edges)))]
            new_route = str(rng.choice(pilot.route_nodes()))
            new_target = candidate.idx[new_route]
            if new_target != target:
                weight = candidate.w[target, source]
                candidate.w[target, source] = 0.0
                candidate.w[new_target, source] = weight
    elif mutation_type == "add_hub_to_route_edge":
        route = str(rng.choice(pilot.route_nodes()))
        hub = str(rng.choice(["shared_actor_hub", "shared_action_hub", "shared_context_hub"]))
        candidate.w[candidate.idx[route], candidate.idx[hub]] = float(rng.normal(0.0, 0.35))
    elif mutation_type == "remove_direct_shortcut":
        zero_direct_token_readout_edges(candidate)
    zero_direct_token_readout_edges(candidate)
    return candidate, mutation_type


def graph_to_json(graph: pilot.AuthorityGraph) -> dict[str, Any]:
    edges = []
    for target, source in nonzero_edges(graph):
        edges.append({
            "source": graph.nodes[source],
            "target": graph.nodes[target],
            "weight": round(float(graph.w[target, source]), 6),
        })
    return {
        "nodes": [{"name": name, "type": graph.types[name]} for name in graph.nodes],
        "edges": edges,
        "params": {
            "decay": round(float(graph.decay), 6),
            "frame_gate": round(float(graph.frame_gate), 6),
            "suppressor_strength": round(float(graph.suppressor_strength), 6),
            "competition": round(float(graph.competition), 6),
            "route_bias": round(float(graph.route_bias), 6),
        },
        "edge_count": graph.edge_count(),
        "node_count": len(graph.nodes),
    }


def leakage_audit(arm: str, graph: pilot.AuthorityGraph) -> dict[str, Any]:
    direct_token_readout_edges = 0
    token_route_edges = 0
    for target_name, target in graph.idx.items():
        for source_name, source in graph.idx.items():
            if abs(graph.w[target, source]) <= 1.0e-9:
                continue
            if graph.types.get(source_name) == "token_input" and graph.types.get(target_name) == "readout":
                direct_token_readout_edges += 1
            if graph.types.get(source_name) == "token_input" and graph.types.get(target_name) == "frame_route":
                token_route_edges += 1
    return {
        "arm": arm,
        "is_grammar_arm": arm in GRAMMAR_ARMS,
        "uses_task_label_rules": arm in {"hand_seeded", "damaged_hand_seeded_50"},
        "uses_named_correct_token_to_route_mapping": arm in {"hand_seeded", "damaged_hand_seeded_50"},
        "direct_token_to_readout_edges": direct_token_readout_edges,
        "token_route_edges": token_route_edges,
        "grammar_policy": (
            "rng_uniform_without_label_lookup"
            if arm in GRAMMAR_ARMS
            else "baseline_or_hand_seeded"
        ),
        "passes_grammar_leakage_audit": (
            arm not in GRAMMAR_ARMS
            or direct_token_readout_edges == 0
        ),
    }


def evaluate_individual(
    graph: pilot.AuthorityGraph,
    splits: dict[str, dict[str, list[Any]]],
    *,
    args: argparse.Namespace,
    seed: int,
    hand_edge_count: int,
    fitness_mode: str,
    generation: int,
    mutation_type: str,
    eval_validation: bool = True,
) -> Individual:
    train = evaluate_split(
        graph,
        splits["search_train"],
        steps=args.steps,
        seed=seed + 100_000 + generation,
        hand_edge_count=hand_edge_count,
        fitness_mode=fitness_mode,
    )
    if eval_validation:
        validation = evaluate_split(
            graph,
            splits["validation"],
            steps=args.steps,
            seed=seed + 200_000 + generation,
            hand_edge_count=hand_edge_count,
            fitness_mode=fitness_mode,
        )
    else:
        validation = {"fitness": float("-inf"), "raw": {}, "metrics": {}}
    return Individual(
        graph=graph,
        train_metrics=train,
        validation_metrics=validation,
        train_fitness=float(train["fitness"]),
        validation_fitness=float(validation["fitness"]),
        generation=generation,
        mutation_type=mutation_type,
        fitness_mode=fitness_mode,
    )


def compact_individual(individual: Individual) -> dict[str, Any]:
    return {
        "generation": individual.generation,
        "mutation_type": individual.mutation_type,
        "fitness_mode": individual.fitness_mode,
        "train_fitness": individual.train_fitness,
        "validation_fitness": individual.validation_fitness,
        "train": individual.train_metrics["raw"],
        "validation": individual.validation_metrics["raw"],
    }


def checkpoint_path(out_dir: Path, fitness_mode: str, arm: str, seed: int, generation: int) -> Path:
    return out_dir / "checkpoints" / fitness_mode / f"{arm}_seed{seed}_gen{generation}.json"


def write_checkpoint(
    out_dir: Path,
    fitness_mode: str,
    arm: str,
    seed: int,
    generation: int,
    best_validation: Individual,
    population: list[Individual],
    accepted_mutations: dict[str, int],
) -> None:
    path = checkpoint_path(out_dir, fitness_mode, arm, seed, generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "fitness_mode": fitness_mode,
        "seed": seed,
        "generation": generation,
        "best_validation": compact_individual(best_validation),
        "population_train_best": compact_individual(max(population, key=lambda item: item.train_fitness)),
        "population_validation_best": compact_individual(max(population, key=lambda item: item.validation_fitness)),
        "accepted_mutations": accepted_mutations,
    }
    path.write_text(json.dumps(round_floats(payload), indent=2) + "\n", encoding="utf-8")


def save_best_graph(out_dir: Path, fitness_mode: str, arm: str, seed: int, graph: pilot.AuthorityGraph) -> str:
    path = out_dir / "best_graphs" / fitness_mode / f"{arm}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_json(graph), indent=2) + "\n", encoding="utf-8")
    return str(path)


def evaluate_upper_bound(
    graph: pilot.AuthorityGraph,
    splits: dict[str, dict[str, list[Any]]],
    *,
    args: argparse.Namespace,
    seed: int,
    hand_edge_count: int,
    fitness_mode: str,
    out_dir: Path,
    arm: str,
) -> dict[str, Any]:
    train = evaluate_split(
        graph,
        splits["search_train"],
        steps=args.steps,
        seed=seed + 10,
        hand_edge_count=hand_edge_count,
        fitness_mode=fitness_mode,
    )
    validation = evaluate_split(
        graph,
        splits["validation"],
        steps=args.steps,
        seed=seed + 20,
        hand_edge_count=hand_edge_count,
        fitness_mode=fitness_mode,
    )
    final_test = evaluate_split(
        graph,
        splits["final_test"],
        steps=args.steps,
        seed=seed + 30,
        hand_edge_count=hand_edge_count,
        fitness_mode=fitness_mode,
    )
    graph_path = save_best_graph(out_dir, fitness_mode, arm, seed, graph)
    return {
        "arm": arm,
        "fitness_mode": fitness_mode,
        "seed": seed,
        "evolved": False,
        "generations_completed": 0,
        "generations_to_threshold": 0 if success_flags(validation["raw"])["success"] else None,
        "selected_by": "upper_bound",
        "train": train["raw"],
        "train_fitness": train["fitness"],
        "validation": validation["raw"],
        "validation_fitness": validation["fitness"],
        "final_test": final_test["raw"],
        "final_test_fitness": final_test["fitness"],
        "final_test_flags": success_flags(final_test["raw"]),
        "accepted_mutations": {},
        "leakage_audit": leakage_audit(arm, graph),
        "best_graph_path": graph_path,
    }


def evolve_arm(
    arm: str,
    splits: dict[str, dict[str, list[Any]]],
    *,
    args: argparse.Namespace,
    seed: int,
    hand_edge_count: int,
    fitness_mode: str,
    out_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 500_000)
    base = build_seed_graph(arm, args, seed + 700_000, hand_edge_count)
    if arm == "hand_seeded":
        return evaluate_upper_bound(
            base,
            splits,
            args=args,
            seed=seed,
            hand_edge_count=hand_edge_count,
            fitness_mode=fitness_mode,
            out_dir=out_dir,
            arm=arm,
        )

    population: list[Individual] = [
        evaluate_individual(
            base,
            splits,
            args=args,
            seed=seed,
            hand_edge_count=hand_edge_count,
            fitness_mode=fitness_mode,
            generation=0,
            mutation_type="seed",
        )
    ]
    for index in range(max(0, args.population_size - 1)):
        mutated, mutation_type = mutate_developmental(base, rng, args.mutation_scale)
        population.append(
            evaluate_individual(
                mutated,
                splits,
                args=args,
                seed=seed + index + 1,
                hand_edge_count=hand_edge_count,
                fitness_mode=fitness_mode,
                generation=0,
                mutation_type=mutation_type,
            )
        )
    population.sort(key=lambda item: item.train_fitness, reverse=True)
    population = population[: args.population_size]
    best_validation = max(population, key=lambda item: item.validation_fitness)
    generations_to_threshold = 0 if success_flags(best_validation.validation_metrics["raw"])["success"] else None
    accepted_mutations: dict[str, int] = {}
    interrupted_by_time = False
    last_generation = 0

    for generation in range(1, args.generations + 1):
        if time.time() >= deadline:
            interrupted_by_time = True
            break
        last_generation = generation
        parent_pool = population[: max(1, min(len(population), max(1, args.population_size // 2)))]
        parent = parent_pool[int(rng.integers(0, len(parent_pool)))]
        candidate_graph, mutation_type = mutate_developmental(parent.graph, rng, args.mutation_scale)
        train_eval = evaluate_split(
            candidate_graph,
            splits["search_train"],
            steps=args.steps,
            seed=seed + 800_000 + generation,
            hand_edge_count=hand_edge_count,
            fitness_mode=fitness_mode,
        )
        candidate_train_fitness = float(train_eval["fitness"])
        if candidate_train_fitness >= population[-1].train_fitness:
            validation_eval = evaluate_split(
                candidate_graph,
                splits["validation"],
                steps=args.steps,
                seed=seed + 900_000 + generation,
                hand_edge_count=hand_edge_count,
                fitness_mode=fitness_mode,
            )
            candidate = Individual(
                graph=candidate_graph,
                train_metrics=train_eval,
                validation_metrics=validation_eval,
                train_fitness=candidate_train_fitness,
                validation_fitness=float(validation_eval["fitness"]),
                generation=generation,
                mutation_type=mutation_type,
                fitness_mode=fitness_mode,
            )
            population.append(candidate)
            population.sort(key=lambda item: item.train_fitness, reverse=True)
            population = population[: args.population_size]
            accepted_mutations[mutation_type] = accepted_mutations.get(mutation_type, 0) + 1
            if candidate.validation_fitness > best_validation.validation_fitness:
                best_validation = candidate
                if generations_to_threshold is None and success_flags(candidate.validation_metrics["raw"])["success"]:
                    generations_to_threshold = generation
        if generation % args.checkpoint_every == 0:
            write_checkpoint(out_dir, fitness_mode, arm, seed, generation, best_validation, population, accepted_mutations)

    generations_completed = last_generation
    final_test = evaluate_split(
        best_validation.graph,
        splits["final_test"],
        steps=args.steps,
        seed=seed + 1_000_000,
        hand_edge_count=hand_edge_count,
        fitness_mode=fitness_mode,
    )
    graph_path = save_best_graph(out_dir, fitness_mode, arm, seed, best_validation.graph)
    write_checkpoint(out_dir, fitness_mode, arm, seed, generations_completed, best_validation, population, accepted_mutations)
    return {
        "arm": arm,
        "fitness_mode": fitness_mode,
        "seed": seed,
        "evolved": True,
        "interrupted_by_time": interrupted_by_time,
        "generations_completed": generations_completed,
        "generations_to_threshold": generations_to_threshold,
        "selected_by": "validation_fitness",
        "train": best_validation.train_metrics["raw"],
        "train_fitness": best_validation.train_fitness,
        "validation": best_validation.validation_metrics["raw"],
        "validation_fitness": best_validation.validation_fitness,
        "final_test": final_test["raw"],
        "final_test_fitness": final_test["fitness"],
        "final_test_flags": success_flags(final_test["raw"]),
        "accepted_mutations": accepted_mutations,
        "leakage_audit": leakage_audit(arm, best_validation.graph),
        "best_graph_path": graph_path,
    }


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


def aggregate_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "overall_accuracy",
        "latent_refraction_accuracy",
        "multi_aspect_accuracy",
        "temporal_order_accuracy",
        "authority_refraction_score",
        "wrong_frame_drop",
        "recurrence_drop",
        "route_specialization",
        "inactive_influence",
        "edge_count",
        "node_count",
    ]
    out: dict[str, Any] = {
        "runs": len(records),
        "success_rate": float(np.mean([record["final_test_flags"]["success"] for record in records])),
        "strong_success_rate": float(np.mean([record["final_test_flags"]["strong_success"] for record in records])),
        "generations_completed": numeric_summary([record["generations_completed"] for record in records]),
        "generations_to_threshold": numeric_summary([
            record["generations_to_threshold"]
            for record in records
            if record["generations_to_threshold"] is not None
        ]),
        "train_fitness": numeric_summary([record["train_fitness"] for record in records]),
        "validation_fitness": numeric_summary([record["validation_fitness"] for record in records]),
        "final_test_fitness": numeric_summary([record["final_test_fitness"] for record in records]),
    }
    for split in ("train", "validation", "final_test"):
        out[split] = {
            field: numeric_summary([record[split][field] for record in records])
            for field in fields
        }
    mutation_counts: dict[str, list[int]] = {}
    for record in records:
        for mutation_type, count in record["accepted_mutations"].items():
            mutation_counts.setdefault(mutation_type, []).append(int(count))
    out["accepted_mutations"] = {
        mutation_type: numeric_summary(counts)
        for mutation_type, counts in sorted(mutation_counts.items())
    }
    out["leakage_audit"] = {
        "all_grammar_runs_pass": all(
            record["leakage_audit"]["passes_grammar_leakage_audit"]
            for record in records
            if record["arm"] in GRAMMAR_ARMS
        ),
        "direct_token_to_readout_edges": numeric_summary([
            record["leakage_audit"]["direct_token_to_readout_edges"]
            for record in records
        ]),
    }
    return out


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        by_mode.setdefault(record["fitness_mode"], {}).setdefault(record["arm"], []).append(record)
    return {
        mode: {
            arm: aggregate_arm(items)
            for arm, items in sorted(arms.items())
        }
        for mode, arms in sorted(by_mode.items())
    }


def mode_aggregate(aggregate: dict[str, Any], mode: str) -> dict[str, Any]:
    return aggregate.get(mode, {})


def get_mean(aggregate: dict[str, Any], mode: str, arm: str, split: str, field: str) -> float:
    value = mode_aggregate(aggregate, mode).get(arm, {}).get(split, {}).get(field, {}).get("mean")
    return float(value) if value is not None else 0.0


def get_success(aggregate: dict[str, Any], mode: str, arm: str) -> float:
    return float(mode_aggregate(aggregate, mode).get(arm, {}).get("success_rate", 0.0))


def verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    if "coarse" in aggregate and "authority_shaped" in aggregate:
        grammar = [arm for arm in GRAMMAR_ARMS if arm in aggregate["authority_shaped"]]
        shaped_auth = np.mean([
            get_mean(aggregate, "authority_shaped", arm, "final_test", "authority_refraction_score")
            for arm in grammar
        ]) if grammar else 0.0
        coarse_auth = np.mean([
            get_mean(aggregate, "coarse", arm, "final_test", "authority_refraction_score")
            for arm in grammar
        ]) if grammar else 0.0
        shaped_temporal = np.mean([
            get_mean(aggregate, "authority_shaped", arm, "final_test", "temporal_order_accuracy")
            for arm in grammar
        ]) if grammar else 0.0
        coarse_temporal = np.mean([
            get_mean(aggregate, "coarse", arm, "final_test", "temporal_order_accuracy")
            for arm in grammar
        ]) if grammar else 0.0
        shaped_success = max([get_success(aggregate, "authority_shaped", arm) for arm in grammar] or [0.0])
        random_success = max(
            get_success(aggregate, "coarse", "random_graph"),
            get_success(aggregate, "authority_shaped", "random_graph"),
        )
        damaged_shaped = get_mean(aggregate, "authority_shaped", "damaged_hand_seeded_50", "final_test", "authority_refraction_score")
        damaged_coarse = get_mean(aggregate, "coarse", "damaged_hand_seeded_50", "final_test", "authority_refraction_score")
        shaped_improves = shaped_auth > coarse_auth + 0.03 or shaped_temporal > coarse_temporal + 0.10 or shaped_success > 0.0
        return {
            "shaped_fitness_improves_search": shaped_improves,
            "search_space_problem_supported": shaped_improves,
            "grammar_prior_still_too_weak": shaped_success == 0.0,
            "damaged_hand_repair_improves": damaged_shaped > damaged_coarse + 0.03,
            "random_search_still_insufficient": random_success == 0.0,
            "final_verdict_uses_final_test": True,
        }

    mode = next(iter(aggregate.keys())) if aggregate else "coarse"
    single = mode_aggregate(aggregate, mode)
    grammar_arms = [arm for arm in GRAMMAR_ARMS if arm in single]
    random_success = get_success(aggregate, mode, "random_graph")
    best_grammar_success = max([get_success(aggregate, mode, arm) for arm in grammar_arms] or [0.0])
    hand_success = get_success(aggregate, mode, "hand_seeded")
    damaged_success = get_success(aggregate, mode, "damaged_hand_seeded_50")
    route_acc = get_mean(aggregate, mode, "route_grammar_graph", "final_test", "overall_accuracy")
    gate_acc = get_mean(aggregate, mode, "route_gate_grammar_graph", "final_test", "overall_accuracy")
    recurrence_temporal = get_mean(aggregate, mode, "route_gate_recurrence_grammar", "final_test", "temporal_order_accuracy")
    gate_temporal = get_mean(aggregate, mode, "route_gate_grammar_graph", "final_test", "temporal_order_accuracy")
    hub_success = get_success(aggregate, mode, "route_gate_hub_grammar")
    recurrence_success = get_success(aggregate, mode, "route_gate_recurrence_grammar")
    return {
        "supports_developmental_prior_search": best_grammar_success > random_success + 0.20,
        "route_structure_required_for_evolution": route_acc > get_mean(aggregate, mode, "random_graph", "final_test", "overall_accuracy") + 0.05,
        "frame_gates_required_for_evolution": gate_acc > route_acc + 0.05,
        "recurrence_required_for_evolution": recurrence_temporal > gate_temporal + 0.15,
        "hubs_help_evolution": hub_success > recurrence_success + 0.10,
        "random_search_sufficient": random_success >= 0.40,
        "damaged_hand_recovery_supported": damaged_success >= 0.40,
        "manual_structure_still_dominates": hand_success > best_grammar_success + 0.20,
        "final_verdict_uses_final_test": True,
    }


def summary_json_name(summary: dict[str, Any]) -> str:
    if summary.get("config", {}).get("fitness_mode") == "ab_compare":
        return "authority_gradient_search_quick_summary.json"
    return "developmental_search_summary.json"


def report_path_for_args(args: argparse.Namespace) -> Path:
    return QUICK_REPORT_PATH if args.fitness_mode == "ab_compare" else REPORT_PATH


def write_partial_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / summary_json_name(summary)
    path.write_text(json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def write_report(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    arms = summary["config"]["arms_completed"]
    modes = summary["config"].get("fitness_modes_run", [summary["config"].get("fitness_mode", "coarse")])
    title = (
        "# Authority Gradient Search Quick Test"
        if summary["config"].get("fitness_mode") == "ab_compare"
        else "# Authority Graph Developmental Search"
    )
    lines = [
        title,
        "",
        "## Background",
        "",
        "The hand-seeded frame-gated authority graph can solve the small latent-refraction, multi-aspect, and temporal-order tasks without neural layers or backprop. The previous minimality run showed that random weak priors do not rediscover the mechanism, while damaged hand graphs can partially recover.",
        "",
        "## Why This Run Matters",
        "",
        "This run asks whether a compact developmental grammar can generate and evolve the route/gate/recurrent/readout structure, or whether the mechanism still depends on manual wiring.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Graph Grammar Arms",
        "",
        "- `random_graph`: pure random signed graph baseline.",
        "- `route_grammar_graph`: route groups and readout ports, but no frame gates or guaranteed recurrence.",
        "- `route_gate_grammar_graph`: route grammar plus frame gates.",
        "- `route_gate_recurrence_grammar`: route/gate grammar plus recurrent route and temporal memory edges.",
        "- `route_gate_hub_grammar`: recurrence grammar plus shared hubs.",
        "- `damaged_hand_seeded_50`: hand graph with 50% edges removed.",
        "- `hand_seeded`: upper bound, evaluated but not evolved.",
        "",
        "Grammar arms use random token-route wiring without task-label rule lookup. Exact task-solution wiring is allowed only in the hand-seeded upper bound and damaged-hand recovery baseline.",
        "",
        "## Fitness Definition",
        "",
        "```text",
        "coarse fitness =",
        "  1.0 * overall_accuracy",
        "  + 0.5 * authority_refraction_score",
        "  + 0.3 * temporal_order_accuracy",
        "  + 0.2 * max(wrong_frame_drop, 0)",
        "  - 0.05 * inactive_influence",
        "  - 0.02 * edge_count_penalty",
        "",
        "authority_shaped fitness =",
        "  1.0 * overall_accuracy",
        "  + 0.5 * authority_refraction_score",
        "  + 0.3 * wrong_frame_drop",
        "  + 0.25 * recurrence_drop",
        "  + 0.25 * route_specialization",
        "  + 0.3 * temporal_order_accuracy",
        "  - 0.05 * inactive_influence",
        "  - 0.02 * edge_count_penalty",
        "```",
        "",
        "Evolution uses search-train fitness, but best graph selection uses validation fitness. Final verdicts use final-test metrics.",
        "",
        "## Main Final-Test Results",
        "",
        "| Fitness | Arm | Success | Strong Success | Accuracy | Latent | Multi | Temporal | Authority | Wrong Frame | Recurrence Drop | Route Spec | Inactive | Edges |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        for arm in arms:
            if arm not in aggregate.get(mode, {}):
                continue
            item = aggregate[mode][arm]
            final = item["final_test"]
            lines.append(
                f"| `{mode}` | `{arm}` | `{fmt(item['success_rate'])}` | `{fmt(item['strong_success_rate'])}` "
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
        "## Train / Validation / Final-Test Fitness",
        "",
        "| Fitness | Arm | Train Fitness | Validation Fitness | Final-Test Fitness | Generations Completed | Generations To Threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for mode in modes:
        for arm in arms:
            if arm not in aggregate.get(mode, {}):
                continue
            item = aggregate[mode][arm]
            lines.append(
                f"| `{mode}` | `{arm}` | `{fmt(item['train_fitness']['mean'])}` "
                f"| `{fmt(item['validation_fitness']['mean'])}` "
                f"| `{fmt(item['final_test_fitness']['mean'])}` "
                f"| `{fmt(item['generations_completed']['mean'])}` "
                f"| `{fmt(item['generations_to_threshold']['mean'])}` |"
            )
    lines.extend([
        "",
        "## Leakage Audit",
        "",
        "| Fitness | Arm | Grammar Runs Pass | Direct Token->Readout Edges |",
        "|---|---|---:|---:|",
    ])
    for mode in modes:
        for arm in arms:
            if arm not in aggregate.get(mode, {}):
                continue
            audit = aggregate[mode][arm]["leakage_audit"]
            lines.append(
                f"| `{mode}` | `{arm}` | `{audit['all_grammar_runs_pass']}` "
                f"| `{fmt(audit['direct_token_to_readout_edges']['mean'])}` |"
            )
    lines.extend([
        "",
        "## Best Graph Examples",
        "",
    ])
    for record in summary["records"]:
        lines.append(
            f"- `{record['fitness_mode']}` / `{record['arm']}` seed `{record['seed']}`: "
            f"`{record['best_graph_path']}`"
        )
    lines.extend([
        "",
        "## Failure Cases",
        "",
        "- If grammar arms remain near random on final-test, the current mechanism still requires too much manual structure.",
        "- If train fitness improves but validation/final-test does not, the search is overfitting the small train split.",
        "- If accuracy improves without authority/refraction or wrong-frame drop, the graph is solving labels without the target authority mechanism.",
        "",
        "## Minimal Surviving Prior",
        "",
        "Read from final-test comparisons only: random vs route grammar, route vs route+gate, route+gate vs route+gate+recurrence, and recurrence vs recurrence+hub.",
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
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    started = time.time()
    deadline = started + max(1.0, args.max_runtime_hours * 3600.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hand_edge_count = pilot.build_hand_seeded_graph(args.decay).edge_count()
    records: list[dict[str, Any]] = []
    interrupted = False
    fitness_modes = fitness_modes_to_run(args)
    report_path = report_path_for_args(args)
    for seed in range(args.seeds):
        splits = split_datasets(args, seed)
        for fitness_mode in fitness_modes:
            for arm in args.arms:
                if time.time() >= deadline:
                    interrupted = True
                    break
                print(f"[developmental-search] fitness={fitness_mode} arm={arm} seed={seed}", flush=True)
                record = evolve_arm(
                    arm,
                    splits,
                    args=args,
                    seed=seed,
                    hand_edge_count=hand_edge_count,
                    fitness_mode=fitness_mode,
                    out_dir=args.out_dir,
                    deadline=deadline,
                )
                records.append(record)
                aggregate = aggregate_records(records)
                partial = {
                    "config": config_dict(args, started, completed=True),
                    "aggregate": aggregate,
                    "verdict": verdict(aggregate),
                    "records": records,
                    "runtime_seconds": time.time() - started,
                    "interrupted_by_wall_clock": interrupted,
                    "environment": environment_dict(),
                }
                write_partial_summary(args.out_dir, partial)
                write_report(round_floats(partial), report_path)
            if interrupted:
                break
        if interrupted:
            break
    aggregate = aggregate_records(records)
    summary = {
        "config": config_dict(args, started, completed=not interrupted),
        "aggregate": aggregate,
        "verdict": verdict(aggregate),
        "records": records,
        "runtime_seconds": time.time() - started,
        "interrupted_by_wall_clock": interrupted,
        "environment": environment_dict(),
    }
    summary = round_floats(summary)
    write_partial_summary(args.out_dir, summary)
    write_report(summary, report_path)
    print(json.dumps({
        "verdict": summary["verdict"],
        "aggregate": summary["aggregate"],
        "json": str(args.out_dir / summary_json_name(summary)),
        "report": str(report_path),
        "interrupted_by_wall_clock": interrupted,
    }, indent=2))
    return 0


def fitness_modes_to_run(args: argparse.Namespace) -> list[str]:
    if args.fitness_mode == "ab_compare":
        return ["coarse", "authority_shaped"]
    return [args.fitness_mode]


def config_dict(args: argparse.Namespace, started: float, *, completed: bool) -> dict[str, Any]:
    return {
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
        "fitness_modes_run": fitness_modes_to_run(args),
        "arms_requested": args.arms,
        "arms_completed": args.arms,
        "smoke": args.smoke,
        "completed": completed,
        "started_unix": started,
    }


def environment_dict() -> dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
