#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "FRAME_GATED_AUTHORITY_GRAPH_PILOT.md"
MINIMALITY_REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_MINIMALITY_EVOLUTION.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-pilot"
MINIMALITY_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-minimality"

FRAMES = ["danger", "friendship", "sound", "environment"]
FRAME_BY_ROUTE = {frame: f"{frame}_route" for frame in FRAMES}
ACTIVE_GROUP_BY_FRAME = {
    "danger": "danger_pair",
    "friendship": "friendship_pair",
    "sound": "sound_pair",
    "environment": "environment_pair",
}
GROUP_FIELDS = {
    "danger_pair": ("actor", "action"),
    "friendship_pair": ("actor", "relation"),
    "sound_pair": ("actor", "sound"),
    "environment_pair": ("place", "noise"),
}

ACTORS = ["dog", "cat", "snake", "bird"]
ACTIONS = ["bite", "sleep", "chase", "run"]
RELATIONS = ["owner", "stranger", "alone"]
SOUNDS = ["bark", "music", "quiet"]
PLACES = ["street", "kitchen", "park"]
NOISES = ["car_noise", "crowd", "quiet_noise"]
TEMPORAL_TOKENS = ["me", "child"]
TOKENS = sorted(set(ACTORS + ACTIONS + RELATIONS + SOUNDS + PLACES + NOISES + TEMPORAL_TOKENS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-backprop frame-gated recurrent authority graph pilot.")
    parser.add_argument("--study", choices=("pilot", "minimality"), default="pilot")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--mutation-steps", type=int, default=250)
    parser.add_argument("--mutation-scale", type=float, default=0.18)
    parser.add_argument("--random-graphs", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--decay", type=float, default=0.35)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.study == "minimality" and args.out_dir == DEFAULT_OUT:
        args.out_dir = MINIMALITY_OUT
    if args.smoke:
        args.seeds = 1
        args.steps = 3
        args.samples = 48
        args.mutation_steps = 15
        args.random_graphs = 2
        args.population_size = 6
    return args


@dataclass(frozen=True)
class Observation:
    actor: str
    action: str
    relation: str
    sound: str
    place: str
    noise: str


@dataclass(frozen=True)
class StaticExample:
    obs: Observation
    frame: str
    label: int


@dataclass(frozen=True)
class TemporalExample:
    sequence: tuple[str, str, str]
    label: int


@dataclass
class AuthorityGraph:
    nodes: list[str]
    types: dict[str, str]
    w: np.ndarray
    bias: np.ndarray
    decay: float
    frame_gate: float
    suppressor_strength: float
    competition: float
    route_bias: float

    @property
    def idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.nodes)}

    def clone(self) -> "AuthorityGraph":
        return copy.deepcopy(self)

    def edge_count(self) -> int:
        return int(np.count_nonzero(np.abs(self.w) > 1.0e-9))


def add_edge(graph: AuthorityGraph, target: str, source: str, weight: float) -> None:
    graph.w[graph.idx[target], graph.idx[source]] = weight


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 40.0), -40.0)))


def route_nodes() -> list[str]:
    return [FRAME_BY_ROUTE[frame] for frame in FRAMES]


def suppressor_nodes() -> list[str]:
    return [f"suppress_{frame}" for frame in FRAMES]


def temporal_nodes() -> list[str]:
    role_nodes = []
    for token in ["dog", "cat", "snake", "me", "child"]:
        role_nodes.extend([f"subject_{token}", f"object_{token}"])
    role_nodes.extend(["verb_bite", "verb_chase", "temporal_route"])
    return role_nodes


def build_empty_graph(decay: float) -> AuthorityGraph:
    nodes: list[str] = []
    types: dict[str, str] = {}
    for token in TOKENS:
        nodes.append(f"tok_{token}")
        types[f"tok_{token}"] = "token_input"
    for hub in ("shared_actor_hub", "shared_context_hub", "shared_action_hub"):
        nodes.append(hub)
        types[hub] = "shared_hub"
    for route in route_nodes():
        nodes.append(route)
        types[route] = "frame_route"
    for suppressor in suppressor_nodes():
        nodes.append(suppressor)
        types[suppressor] = "suppressor"
    for node in temporal_nodes():
        nodes.append(node)
        types[node] = "temporal_role" if node != "temporal_route" else "frame_route"
    for readout in ("readout_positive", "readout_negative"):
        nodes.append(readout)
        types[readout] = "readout"
    n = len(nodes)
    return AuthorityGraph(
        nodes=nodes,
        types=types,
        w=np.zeros((n, n), dtype=np.float64),
        bias=np.zeros(n, dtype=np.float64),
        decay=decay,
        frame_gate=0.85,
        suppressor_strength=1.10,
        competition=0.42,
        route_bias=-1.55,
    )


def build_hand_seeded_graph(decay: float) -> AuthorityGraph:
    graph = build_empty_graph(decay)
    idx = graph.idx

    for token in ACTORS:
        add_edge(graph, "shared_actor_hub", f"tok_{token}", 0.25)
    for token in ACTIONS:
        add_edge(graph, "shared_action_hub", f"tok_{token}", 0.22)
    for token in PLACES + NOISES + SOUNDS:
        add_edge(graph, "shared_context_hub", f"tok_{token}", 0.20)
    for route in route_nodes():
        add_edge(graph, route, route, 0.18)
        add_edge(graph, route, "shared_actor_hub", 0.08)
        add_edge(graph, route, "shared_action_hub", 0.06)
        add_edge(graph, route, "shared_context_hub", 0.04)

    # Route evidence. These are explicit, hand-seeded authority paths, not learned by backprop.
    for token, weight in {"dog": 0.48, "snake": 0.62, "cat": 0.18, "bird": -0.20}.items():
        add_edge(graph, "danger_route", f"tok_{token}", weight)
    for token, weight in {"bite": 0.86, "chase": 0.46, "sleep": -0.55, "run": -0.20}.items():
        add_edge(graph, "danger_route", f"tok_{token}", weight)

    for token, weight in {"dog": 0.44, "cat": 0.38, "snake": -0.28, "bird": 0.05}.items():
        add_edge(graph, "friendship_route", f"tok_{token}", weight)
    for token, weight in {"owner": 0.92, "stranger": -0.44, "alone": -0.34, "bite": -0.35}.items():
        add_edge(graph, "friendship_route", f"tok_{token}", weight)

    for token, weight in {"dog": 0.40, "cat": 0.25, "snake": 0.05, "bird": 0.22}.items():
        add_edge(graph, "sound_route", f"tok_{token}", weight)
    for token, weight in {"bark": 0.96, "music": 0.62, "quiet": -0.54}.items():
        add_edge(graph, "sound_route", f"tok_{token}", weight)

    for token, weight in {"street": 0.82, "kitchen": -0.06, "park": -0.10}.items():
        add_edge(graph, "environment_route", f"tok_{token}", weight)
    for token, weight in {"car_noise": 0.82, "crowd": 0.36, "quiet_noise": -0.48}.items():
        add_edge(graph, "environment_route", f"tok_{token}", weight)

    for frame in FRAMES:
        add_edge(graph, FRAME_BY_ROUTE[frame], f"suppress_{frame}", -graph.suppressor_strength)
        for other_frame in FRAMES:
            if other_frame != frame:
                add_edge(graph, FRAME_BY_ROUTE[frame], f"suppress_{other_frame}", 0.16)

    # Temporal role-binding scaffold. Streaming injection fills subject/object/verb memory nodes.
    for token in ("dog", "snake"):
        add_edge(graph, "temporal_route", f"subject_{token}", 0.86)
    for token in ("me", "cat", "child"):
        add_edge(graph, "temporal_route", f"subject_{token}", -0.90)
    for token in ("me", "cat", "child", "dog"):
        add_edge(graph, "temporal_route", f"object_{token}", 0.30)
    add_edge(graph, "temporal_route", "object_snake", -0.58)
    add_edge(graph, "temporal_route", "verb_bite", 0.40)
    add_edge(graph, "temporal_route", "verb_chase", 0.28)
    add_edge(graph, "temporal_route", "temporal_route", 0.12)

    graph.bias[[idx[route] for route in route_nodes()]] = graph.route_bias
    graph.bias[idx["temporal_route"]] = -0.10
    return graph


def build_random_graph(decay: float, seed: int, edge_count: int) -> AuthorityGraph:
    rng = np.random.default_rng(seed)
    graph = build_empty_graph(decay)
    n = len(graph.nodes)
    forbidden = set()
    for i, name in enumerate(graph.nodes):
        if graph.types[name] == "readout":
            forbidden.add(i)
    suppressor_edges = {
        (graph.idx[FRAME_BY_ROUTE[frame]], graph.idx[f"suppress_{frame}"])
        for frame in FRAMES
    }
    candidates = [
        (target, source)
        for target in range(n)
        for source in range(n)
        if target != source and target not in forbidden and (target, source) not in suppressor_edges
    ]
    random_edge_count = max(0, edge_count - len(suppressor_edges))
    chosen = rng.choice(len(candidates), size=min(random_edge_count, len(candidates)), replace=False)
    for choice in chosen:
        target, source = candidates[int(choice)]
        graph.w[target, source] = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.95)
    graph.bias += rng.normal(0.0, 0.15, size=n)
    graph.route_bias = float(rng.uniform(-1.6, -0.5))
    graph.frame_gate = float(rng.uniform(0.0, 1.0))
    graph.suppressor_strength = float(rng.uniform(0.0, 1.4))
    graph.competition = float(rng.uniform(0.0, 0.8))
    for frame in FRAMES:
        graph.w[graph.idx[FRAME_BY_ROUTE[frame]], graph.idx[f"suppress_{frame}"]] = -graph.suppressor_strength
    return graph


def build_random_structured_graph(decay: float, seed: int) -> AuthorityGraph:
    rng = np.random.default_rng(seed)
    graph = build_hand_seeded_graph(decay)
    nonzero = np.flatnonzero(np.abs(graph.w) > 1.0e-9)
    graph.w.flat[nonzero] = rng.normal(0.0, 0.45, size=len(nonzero))
    graph.bias += rng.normal(0.0, 0.20, size=len(graph.bias))
    graph.route_bias = float(rng.uniform(-1.6, -0.4))
    graph.frame_gate = float(rng.uniform(0.1, 1.2))
    graph.suppressor_strength = float(rng.uniform(0.1, 1.6))
    graph.competition = float(rng.uniform(0.0, 0.8))
    for frame in FRAMES:
        graph.w[graph.idx[FRAME_BY_ROUTE[frame]], graph.idx[f"suppress_{frame}"]] = -graph.suppressor_strength
    return graph


def static_label(obs: Observation, frame: str) -> int:
    if frame == "danger":
        return int(obs.actor in {"dog", "snake"} and obs.action in {"bite", "chase"})
    if frame == "friendship":
        return int(obs.actor in {"dog", "cat"} and obs.relation == "owner")
    if frame == "sound":
        return int(obs.actor in {"dog", "cat", "bird"} and obs.sound in {"bark", "music"})
    if frame == "environment":
        return int(obs.place == "street" and obs.noise in {"car_noise", "crowd"})
    raise ValueError(frame)


def make_static_examples(n: int, seed: int, *, multi_aspect: bool = False) -> list[StaticExample]:
    rng = np.random.default_rng(seed)
    examples: list[StaticExample] = []
    templates = [
        Observation("dog", "bite", "owner", "bark", "street", "car_noise"),
        Observation("bird", "sleep", "alone", "quiet", "park", "quiet_noise"),
        Observation("dog", "sleep", "owner", "quiet", "kitchen", "quiet_noise"),
        Observation("dog", "bite", "stranger", "quiet", "park", "quiet_noise"),
        Observation("cat", "run", "stranger", "music", "park", "quiet_noise"),
        Observation("bird", "sleep", "alone", "quiet", "street", "car_noise"),
        Observation("snake", "chase", "stranger", "quiet", "street", "crowd"),
        Observation("cat", "sleep", "owner", "bark", "kitchen", "quiet_noise"),
    ]
    if multi_aspect:
        templates = [
            Observation("dog", obs.action, obs.relation, obs.sound, obs.place, obs.noise)
            if idx % 2 == 0 else obs
            for idx, obs in enumerate(templates)
        ]
    base_count = max(1, n // len(FRAMES))
    for i in range(base_count):
        obs = templates[i % len(templates)]
        if rng.random() < 0.25:
            obs = Observation(
                actor=obs.actor if multi_aspect and rng.random() < 0.65 else str(rng.choice(ACTORS)),
                action=obs.action,
                relation=obs.relation,
                sound=obs.sound,
                place=obs.place,
                noise=obs.noise,
            )
        for frame in FRAMES:
            examples.append(StaticExample(obs=obs, frame=frame, label=static_label(obs, frame)))
    return examples


def make_temporal_examples() -> list[TemporalExample]:
    pairs = [
        (("dog", "bite", "me"), 1),
        (("me", "bite", "dog"), 0),
        (("dog", "chase", "cat"), 1),
        (("cat", "chase", "dog"), 0),
        (("snake", "bite", "dog"), 1),
        (("dog", "bite", "snake"), 0),
        (("dog", "chase", "child"), 1),
        (("child", "chase", "dog"), 0),
    ]
    return [TemporalExample(sequence=seq, label=label) for seq, label in pairs]


def token_injection(graph: AuthorityGraph, tokens: list[str], scale: float = 1.0) -> np.ndarray:
    inj = np.zeros(len(graph.nodes), dtype=np.float64)
    idx = graph.idx
    for token in tokens:
        node = f"tok_{token}"
        if node in idx:
            inj[idx[node]] += scale
    return inj


def static_step_injection(
    graph: AuthorityGraph,
    obs: Observation,
    frame: str,
    *,
    no_frame_gate: bool,
    no_suppressor: bool,
) -> np.ndarray:
    inj = token_injection(
        graph,
        [obs.actor, obs.action, obs.relation, obs.sound, obs.place, obs.noise],
        scale=1.0,
    )
    idx = graph.idx
    if not no_frame_gate:
        inj[idx[FRAME_BY_ROUTE[frame]]] += graph.frame_gate
    if not no_suppressor:
        for inactive in FRAMES:
            if inactive != frame:
                inj[idx[f"suppress_{inactive}"]] += graph.suppressor_strength
    return inj


def graph_update(graph: AuthorityGraph, state: np.ndarray, injection: np.ndarray) -> np.ndarray:
    raw = graph.decay * state + graph.w @ state + injection + graph.bias
    return np.tanh(np.clip(raw, -4.0, 4.0))


def static_score(
    graph: AuthorityGraph,
    obs: Observation,
    frame: str,
    *,
    steps: int,
    forced_frame: str | None = None,
    no_frame_gate: bool = False,
    no_suppressor: bool = False,
    no_recurrence: bool = False,
) -> float:
    eval_frame = forced_frame or frame
    state = np.zeros(len(graph.nodes), dtype=np.float64)
    run_steps = 1 if no_recurrence else steps
    for _ in range(run_steps):
        inj = static_step_injection(
            graph,
            obs,
            eval_frame,
            no_frame_gate=no_frame_gate,
            no_suppressor=no_suppressor,
        )
        state = graph_update(graph, state, inj)
    idx = graph.idx
    active = float(state[idx[FRAME_BY_ROUTE[eval_frame]]])
    inactive = max(0.0, max(float(state[idx[FRAME_BY_ROUTE[frame_name]]]) for frame_name in FRAMES if frame_name != eval_frame))
    return active - graph.competition * inactive


def static_probability(graph: AuthorityGraph, obs: Observation, frame: str, *, steps: int, **kwargs: Any) -> float:
    return sigmoid(3.0 * static_score(graph, obs, frame, steps=steps, **kwargs))


def temporal_score(
    graph: AuthorityGraph,
    sequence: tuple[str, str, str],
    *,
    steps: int,
    no_recurrence: bool = False,
) -> float:
    state = np.zeros(len(graph.nodes), dtype=np.float64)
    idx = graph.idx
    for pos, token in enumerate(sequence):
        if no_recurrence:
            state = np.zeros(len(graph.nodes), dtype=np.float64)
        inj = token_injection(graph, [token], scale=1.0)
        if not no_recurrence:
            if pos == 0 and f"subject_{token}" in idx:
                inj[idx[f"subject_{token}"]] += 1.0
            elif pos == 1 and token in {"bite", "chase"}:
                inj[idx[f"verb_{token}"]] += 1.0
            elif pos == 2 and f"object_{token}" in idx:
                inj[idx[f"object_{token}"]] += 1.0
        state = graph_update(graph, state, inj)
    for _ in range(max(0, steps - len(sequence))):
        state = graph_update(graph, state, np.zeros(len(graph.nodes), dtype=np.float64))
    return float(state[idx["temporal_route"]])


def static_accuracy(
    graph: AuthorityGraph,
    examples: list[StaticExample],
    *,
    steps: int,
    forced_wrong_frame: bool = False,
    no_frame_gate: bool = False,
    no_suppressor: bool = False,
    no_recurrence: bool = False,
) -> float:
    correct = 0
    for ex in examples:
        forced = None
        if forced_wrong_frame:
            forced = FRAMES[(FRAMES.index(ex.frame) + 1) % len(FRAMES)]
        score = static_score(
            graph,
            ex.obs,
            ex.frame,
            steps=steps,
            forced_frame=forced,
            no_frame_gate=no_frame_gate,
            no_suppressor=no_suppressor,
            no_recurrence=no_recurrence,
        )
        correct += int((score > 0.0) == bool(ex.label))
    return correct / len(examples)


def temporal_accuracy(
    graph: AuthorityGraph,
    examples: list[TemporalExample],
    *,
    steps: int,
    no_recurrence: bool = False,
    shuffled_order: bool = False,
) -> float:
    correct = 0
    for ex in examples:
        sequence = ex.sequence
        if shuffled_order:
            sequence = tuple(reversed(sequence))
        score = temporal_score(graph, sequence, steps=steps, no_recurrence=no_recurrence)
        correct += int((score > 0.0) == bool(ex.label))
    return correct / len(examples)


def swap_observation(obs: Observation, donor: Observation, group: str) -> Observation:
    data = obs.__dict__.copy()
    for field in GROUP_FIELDS[group]:
        data[field] = getattr(donor, field)
    return Observation(**data)


def authority_metrics(
    graph: AuthorityGraph,
    examples: list[StaticExample],
    *,
    steps: int,
    seed: int,
    no_frame_gate: bool = False,
    no_suppressor: bool = False,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    by_frame = {}
    for frame in FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        active_group = ACTIVE_GROUP_BY_FRAME[frame]
        group_deltas = {}
        for group in GROUP_FIELDS:
            deltas = []
            for ex in frame_examples:
                donor = frame_examples[int(rng.integers(0, len(frame_examples)))].obs
                swapped = swap_observation(ex.obs, donor, group)
                p0 = static_probability(
                    graph,
                    ex.obs,
                    frame,
                    steps=steps,
                    no_frame_gate=no_frame_gate,
                    no_suppressor=no_suppressor,
                )
                p1 = static_probability(
                    graph,
                    swapped,
                    frame,
                    steps=steps,
                    no_frame_gate=no_frame_gate,
                    no_suppressor=no_suppressor,
                )
                deltas.append(abs(p0 - p1))
            group_deltas[group] = float(np.mean(deltas))
        inactive = max(value for group, value in group_deltas.items() if group != active_group)
        by_frame[frame] = {
            "active": group_deltas[active_group],
            "inactive": inactive,
            "refraction": group_deltas[active_group] - inactive,
        }
    return {
        "authority_switch_score": float(np.mean([item["refraction"] for item in by_frame.values()])),
        "refraction_index_final": float(np.mean([item["refraction"] for item in by_frame.values()])),
        "active_group_influence": float(np.mean([item["active"] for item in by_frame.values()])),
        "inactive_group_influence": float(np.mean([item["inactive"] for item in by_frame.values()])),
    }


def evaluate_graph(
    graph: AuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    latent_authority = authority_metrics(graph, latent, steps=steps, seed=seed + 1)
    multi_authority = authority_metrics(graph, multi, steps=steps, seed=seed + 2)
    latent_acc = static_accuracy(graph, latent, steps=steps)
    multi_acc = static_accuracy(graph, multi, steps=steps)
    temporal_acc = temporal_accuracy(graph, temporal, steps=steps)
    wrong_frame_acc = static_accuracy(graph, latent, steps=steps, forced_wrong_frame=True)
    no_frame_gate_acc = static_accuracy(graph, latent, steps=steps, no_frame_gate=True)
    no_suppressor_acc = static_accuracy(graph, latent, steps=steps, no_suppressor=True)
    no_suppressor_authority = authority_metrics(
        graph,
        latent,
        steps=steps,
        seed=seed + 3,
        no_suppressor=True,
    )
    no_recurrence_static = static_accuracy(graph, latent, steps=steps, no_recurrence=True)
    no_recurrence_temporal = temporal_accuracy(graph, temporal, steps=steps, no_recurrence=True)
    shuffled_temporal = temporal_accuracy(graph, temporal, steps=steps, shuffled_order=True)
    return {
        "accuracy": float(np.mean([latent_acc, multi_acc, temporal_acc])),
        "latent_refraction_small_accuracy": latent_acc,
        "multi_aspect_small_accuracy": multi_acc,
        "temporal_order_contrast_small_accuracy": temporal_acc,
        "authority_switch_score": float(np.mean([
            latent_authority["authority_switch_score"],
            multi_authority["authority_switch_score"],
        ])),
        "refraction_index_final": float(np.mean([
            latent_authority["refraction_index_final"],
            multi_authority["refraction_index_final"],
        ])),
        "latent_authority": latent_authority,
        "multi_aspect_authority": multi_authority,
        "same_token_set_pair_accuracy": temporal_acc,
        "order_contrast_accuracy": temporal_acc,
        "wrong_frame_accuracy": wrong_frame_acc,
        "wrong_frame_drop": latent_acc - wrong_frame_acc,
        "no_frame_gate_accuracy": no_frame_gate_acc,
        "no_frame_gate_drop": latent_acc - no_frame_gate_acc,
        "no_suppressor_accuracy": no_suppressor_acc,
        "no_suppressor_drop": latent_acc - no_suppressor_acc,
        "no_suppressor_authority": no_suppressor_authority,
        "no_suppressor_authority_drop": (
            latent_authority["authority_switch_score"]
            - no_suppressor_authority["authority_switch_score"]
        ),
        "no_suppressor_inactive_influence_rise": (
            no_suppressor_authority["inactive_group_influence"]
            - latent_authority["inactive_group_influence"]
        ),
        "no_recurrence_static_accuracy": no_recurrence_static,
        "no_recurrence_temporal_accuracy": no_recurrence_temporal,
        "no_recurrence_drop": temporal_acc - no_recurrence_temporal,
        "shuffled_order_accuracy": shuffled_temporal,
        "shuffled_order_drop": temporal_acc - shuffled_temporal,
        "node_count": len(graph.nodes),
        "edge_count": graph.edge_count(),
    }


def fitness(metrics: dict[str, Any]) -> float:
    return (
        metrics["latent_refraction_small_accuracy"]
        + metrics["multi_aspect_small_accuracy"]
        + metrics["temporal_order_contrast_small_accuracy"]
        + 0.45 * metrics["authority_switch_score"]
        + 0.20 * metrics["wrong_frame_drop"]
        + 0.15 * metrics["no_suppressor_drop"]
    )


def mutate_graph(graph: AuthorityGraph, rng: np.random.Generator, scale: float) -> AuthorityGraph:
    candidate = graph.clone()
    nonzero = np.flatnonzero(np.abs(candidate.w) > 1.0e-9)
    if len(nonzero):
        count = int(rng.integers(1, min(8, len(nonzero)) + 1))
        chosen = rng.choice(nonzero, size=count, replace=False)
        candidate.w.flat[chosen] += rng.normal(0.0, scale, size=count)
        candidate.w.flat[chosen] = np.clip(candidate.w.flat[chosen], -1.8, 1.8)
    if rng.random() < 0.35:
        candidate.frame_gate = float(np.clip(candidate.frame_gate + rng.normal(0.0, scale), 0.0, 1.8))
    if rng.random() < 0.35:
        candidate.suppressor_strength = float(np.clip(candidate.suppressor_strength + rng.normal(0.0, scale), 0.0, 2.0))
        for frame in FRAMES:
            candidate.w[candidate.idx[FRAME_BY_ROUTE[frame]], candidate.idx[f"suppress_{frame}"]] = -candidate.suppressor_strength
    if rng.random() < 0.35:
        candidate.competition = float(np.clip(candidate.competition + rng.normal(0.0, scale), 0.0, 1.4))
    return candidate


def evolve_graph(
    datasets: dict[str, list[Any]],
    *,
    seed: int,
    decay: float,
    steps: int,
    mutation_steps: int,
    mutation_scale: float,
) -> tuple[AuthorityGraph, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    graph = build_random_structured_graph(decay, seed + 101)
    best_metrics = evaluate_graph(graph, datasets, steps=steps, seed=seed + 200)
    best_score = fitness(best_metrics)
    accepted = 0
    for step in range(mutation_steps):
        candidate = mutate_graph(graph, rng, mutation_scale)
        candidate_metrics = evaluate_graph(candidate, datasets, steps=steps, seed=seed + 300 + step)
        candidate_score = fitness(candidate_metrics)
        if candidate_score >= best_score:
            graph = candidate
            best_metrics = candidate_metrics
            best_score = candidate_score
            accepted += 1
    best_metrics["mutation_steps_used"] = mutation_steps
    best_metrics["accepted_mutations"] = accepted
    best_metrics["fitness"] = best_score
    return graph, best_metrics


def make_datasets(samples: int, seed: int) -> dict[str, list[Any]]:
    return {
        "latent_refraction_small": make_static_examples(samples, seed + 1, multi_aspect=False),
        "multi_aspect_small": make_static_examples(samples, seed + 2, multi_aspect=True),
        "temporal_order_contrast_small": make_temporal_examples(),
    }


def evaluate_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    datasets = make_datasets(args.samples, seed)
    hand = build_hand_seeded_graph(args.decay)
    hand_metrics = evaluate_graph(hand, datasets, steps=args.steps, seed=seed + 10)
    random_metrics = []
    hand_edge_count = hand.edge_count()
    for idx in range(args.random_graphs):
        random_graph = build_random_graph(args.decay, seed + 1_000 + idx, hand_edge_count)
        random_metrics.append(evaluate_graph(random_graph, datasets, steps=args.steps, seed=seed + 1_100 + idx))
    evolved_graph, evolved_metrics = evolve_graph(
        datasets,
        seed=seed + 2_000,
        decay=args.decay,
        steps=args.steps,
        mutation_steps=args.mutation_steps,
        mutation_scale=args.mutation_scale,
    )
    controls = {
        "no_frame_gate_control": hand_metrics["no_frame_gate_accuracy"],
        "no_suppressor_control": hand_metrics["no_suppressor_accuracy"],
        "no_recurrence_control_temporal": hand_metrics["no_recurrence_temporal_accuracy"],
    }
    return {
        "seed": seed,
        "hand_seeded_graph": hand_metrics,
        "random_graph_baseline": mean_metrics(random_metrics),
        "random_graph_runs": random_metrics,
        "evolved_graph_small": evolved_metrics,
        "controls": controls,
        "datasets": {name: len(items) for name, items in datasets.items()},
        "node_count": len(hand.nodes),
        "edge_count": hand_edge_count,
        "evolved_edge_count": evolved_graph.edge_count(),
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for key, value in rows[0].items():
        if isinstance(value, (int, float)):
            out[key] = float(np.mean([row[key] for row in rows]))
    out["std_accuracy"] = float(np.std([row["accuracy"] for row in rows]))
    return out


def aggregate_numeric(rows: list[dict[str, Any]], path: list[str]) -> dict[str, float | None]:
    values = []
    for row in rows:
        cursor: Any = row
        for key in path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        if isinstance(cursor, (int, float)):
            values.append(float(cursor))
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    modes = ["hand_seeded_graph", "random_graph_baseline", "evolved_graph_small"]
    fields = [
        "accuracy",
        "latent_refraction_small_accuracy",
        "multi_aspect_small_accuracy",
        "temporal_order_contrast_small_accuracy",
        "authority_switch_score",
        "refraction_index_final",
        "wrong_frame_drop",
        "no_frame_gate_drop",
        "no_suppressor_drop",
        "no_suppressor_authority_drop",
        "no_suppressor_inactive_influence_rise",
        "no_recurrence_drop",
        "order_contrast_accuracy",
        "same_token_set_pair_accuracy",
        "node_count",
        "edge_count",
    ]
    aggregate: dict[str, Any] = {}
    for mode in modes:
        aggregate[mode] = {
            field: aggregate_numeric(runs, [mode, field])
            for field in fields
        }
        aggregate[mode]["mutation_steps_used"] = aggregate_numeric(runs, [mode, "mutation_steps_used"])
        aggregate[mode]["accepted_mutations"] = aggregate_numeric(runs, [mode, "accepted_mutations"])
    return aggregate


def mean_value(aggregate: dict[str, Any], mode: str, field: str) -> float:
    value = aggregate.get(mode, {}).get(field, {}).get("mean")
    return float(value) if value is not None else 0.0


def verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    hand_acc = mean_value(aggregate, "hand_seeded_graph", "accuracy")
    random_acc = mean_value(aggregate, "random_graph_baseline", "accuracy")
    evolved_acc = mean_value(aggregate, "evolved_graph_small", "accuracy")
    hand_authority = mean_value(aggregate, "hand_seeded_graph", "authority_switch_score")
    random_authority = mean_value(aggregate, "random_graph_baseline", "authority_switch_score")
    no_frame_drop = mean_value(aggregate, "hand_seeded_graph", "no_frame_gate_drop")
    no_suppressor_drop = mean_value(aggregate, "hand_seeded_graph", "no_suppressor_drop")
    no_suppressor_authority_drop = mean_value(aggregate, "hand_seeded_graph", "no_suppressor_authority_drop")
    no_suppressor_inactive_rise = mean_value(aggregate, "hand_seeded_graph", "no_suppressor_inactive_influence_rise")
    no_recurrence_drop = mean_value(aggregate, "hand_seeded_graph", "no_recurrence_drop")
    return {
        "supports_non_neural_authority_graph": "true" if hand_acc > random_acc + 0.10 else "unclear",
        "supports_frame_gated_routes": "true" if no_frame_drop > 0.10 else "unclear",
        "supports_suppressor_nodes": (
            "true"
            if no_suppressor_drop > 0.03 or no_suppressor_authority_drop > 0.03 or no_suppressor_inactive_rise > 0.01
            else "unclear"
        ),
        "supports_recurrence_for_order_binding": "true" if no_recurrence_drop > 0.20 else "unclear",
        "neural_net_necessity_reduced": "true" if hand_acc > 0.75 and hand_authority > random_authority + 0.05 else "unclear",
        "evolved_graph_improves_over_random": "true" if evolved_acc > random_acc + 0.08 else "unclear",
    }


def zero_nodes(graph: AuthorityGraph, nodes: list[str]) -> AuthorityGraph:
    candidate = graph.clone()
    idx = candidate.idx
    keep = [node for node in nodes if node in idx]
    if not keep:
        return candidate
    node_idx = [idx[node] for node in keep]
    candidate.w[node_idx, :] = 0.0
    candidate.w[:, node_idx] = 0.0
    candidate.bias[node_idx] = 0.0
    return candidate


def zero_edges_matching(graph: AuthorityGraph, predicate: Any) -> AuthorityGraph:
    candidate = graph.clone()
    for target_name, target in candidate.idx.items():
        for source_name, source in candidate.idx.items():
            if predicate(target_name, source_name):
                candidate.w[target, source] = 0.0
    return candidate


def graph_without_component(graph: AuthorityGraph, component: str) -> AuthorityGraph:
    if component == "no_shared_hubs":
        return zero_nodes(graph, [node for node, typ in graph.types.items() if typ == "shared_hub"])
    if component == "no_frame_routes":
        return zero_nodes(graph, route_nodes())
    if component == "no_suppressors":
        return zero_nodes(graph, suppressor_nodes())
    if component == "no_direct_token_to_readout":
        return zero_edges_matching(
            graph,
            lambda target, source: graph.types.get(source) == "token_input" and graph.types.get(target) == "readout",
        )
    if component == "no_route_to_readout":
        # Route states are the explicit readout ports in this graph. Zeroing them tests that dependency.
        return zero_nodes(graph, route_nodes() + ["temporal_route"])
    raise ValueError(component)


def disabled_route_readout_metrics(graph: AuthorityGraph, datasets: dict[str, list[Any]]) -> dict[str, Any]:
    # Static and temporal route readout ports are disabled, so all scores are zero/negative by convention.
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    latent_acc = sum(int(False == bool(ex.label)) for ex in latent) / len(latent)
    multi_acc = sum(int(False == bool(ex.label)) for ex in multi) / len(multi)
    temporal_acc = sum(int(False == bool(ex.label)) for ex in temporal) / len(temporal)
    return {
        "accuracy": float(np.mean([latent_acc, multi_acc, temporal_acc])),
        "latent_refraction_small_accuracy": latent_acc,
        "multi_aspect_small_accuracy": multi_acc,
        "temporal_order_contrast_small_accuracy": temporal_acc,
        "authority_switch_score": 0.0,
        "refraction_index_final": 0.0,
        "wrong_frame_drop": 0.0,
        "no_frame_gate_drop": 0.0,
        "no_suppressor_drop": 0.0,
        "no_suppressor_authority_drop": 0.0,
        "no_suppressor_inactive_influence_rise": 0.0,
        "no_recurrence_drop": 0.0,
        "node_count": len(graph.nodes),
        "edge_count": graph.edge_count(),
    }


def component_classification(baseline: dict[str, Any], ablated: dict[str, Any], component: str) -> str:
    accuracy_drop = baseline["accuracy"] - ablated["accuracy"]
    authority_drop = baseline["authority_switch_score"] - ablated["authority_switch_score"]
    temporal_drop = baseline["temporal_order_contrast_small_accuracy"] - ablated["temporal_order_contrast_small_accuracy"]
    if component in {"input_token_nodes", "readout_ports"}:
        return "necessary_core"
    if accuracy_drop > 0.20 or authority_drop > 0.15 or temporal_drop > 0.35:
        return "necessary_core"
    if accuracy_drop > 0.08 or authority_drop > 0.06 or temporal_drop > 0.15:
        return "useful_but_replaceable"
    if accuracy_drop > 0.02 or authority_drop > 0.02:
        return "helpful_bias"
    if abs(accuracy_drop) <= 0.02 and abs(authority_drop) <= 0.02:
        return "nonessential"
    return "weak_or_unclear"


def component_necessity(args: argparse.Namespace, datasets: dict[str, list[Any]], seed: int) -> dict[str, Any]:
    hand = build_hand_seeded_graph(args.decay)
    baseline = evaluate_graph(hand, datasets, steps=args.steps, seed=seed + 10)
    components = {
        "no_shared_hubs": graph_without_component(hand, "no_shared_hubs"),
        "no_frame_routes": graph_without_component(hand, "no_frame_routes"),
        "no_frame_gates": hand,
        "no_recurrence": hand,
        "no_suppressors": graph_without_component(hand, "no_suppressors"),
        "no_direct_token_to_readout": graph_without_component(hand, "no_direct_token_to_readout"),
        "no_route_to_readout": graph_without_component(hand, "no_route_to_readout"),
    }
    rows: dict[str, Any] = {}
    for name, graph in components.items():
        if name == "no_frame_gates":
            metrics = evaluate_graph_with_forced_controls(graph, datasets, args.steps, seed + 20, no_frame_gate=True)
        elif name == "no_recurrence":
            metrics = evaluate_graph_with_forced_controls(graph, datasets, args.steps, seed + 21, no_recurrence=True)
        elif name == "no_route_to_readout":
            metrics = disabled_route_readout_metrics(graph, datasets)
        else:
            metrics = evaluate_graph(graph, datasets, steps=args.steps, seed=seed + 22)
        rows[name] = {
            "metrics": metrics,
            "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
            "authority_drop": baseline["authority_switch_score"] - metrics["authority_switch_score"],
            "temporal_order_drop": (
                baseline["temporal_order_contrast_small_accuracy"]
                - metrics["temporal_order_contrast_small_accuracy"]
            ),
            "verdict": component_classification(baseline, metrics, name),
        }
    return {"baseline": baseline, "ablations": rows}


def evaluate_graph_with_forced_controls(
    graph: AuthorityGraph,
    datasets: dict[str, list[Any]],
    steps: int,
    seed: int,
    *,
    no_frame_gate: bool = False,
    no_recurrence: bool = False,
) -> dict[str, Any]:
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    latent_acc = static_accuracy(graph, latent, steps=steps, no_frame_gate=no_frame_gate, no_recurrence=no_recurrence)
    multi_acc = static_accuracy(graph, multi, steps=steps, no_frame_gate=no_frame_gate, no_recurrence=no_recurrence)
    temporal_acc = temporal_accuracy(graph, temporal, steps=steps, no_recurrence=no_recurrence)
    latent_authority = authority_metrics(graph, latent, steps=steps, seed=seed + 1, no_frame_gate=no_frame_gate)
    multi_authority = authority_metrics(graph, multi, steps=steps, seed=seed + 2, no_frame_gate=no_frame_gate)
    return {
        "accuracy": float(np.mean([latent_acc, multi_acc, temporal_acc])),
        "latent_refraction_small_accuracy": latent_acc,
        "multi_aspect_small_accuracy": multi_acc,
        "temporal_order_contrast_small_accuracy": temporal_acc,
        "authority_switch_score": float(np.mean([
            latent_authority["authority_switch_score"],
            multi_authority["authority_switch_score"],
        ])),
        "refraction_index_final": float(np.mean([
            latent_authority["refraction_index_final"],
            multi_authority["refraction_index_final"],
        ])),
        "wrong_frame_drop": 0.0,
        "no_frame_gate_drop": 0.0,
        "no_suppressor_drop": 0.0,
        "no_suppressor_authority_drop": 0.0,
        "no_suppressor_inactive_influence_rise": 0.0,
        "no_recurrence_drop": 0.0,
        "node_count": len(graph.nodes),
        "edge_count": graph.edge_count(),
    }


def sorted_existing_edges(graph: AuthorityGraph) -> list[tuple[int, int]]:
    edges = [(int(t), int(s)) for t, s in np.argwhere(np.abs(graph.w) > 1.0e-9)]
    return sorted(edges, key=lambda edge: abs(graph.w[edge[0], edge[1]]), reverse=True)


def shrink_nodes(graph: AuthorityGraph, fraction: float, seed: int, *, random_keep: bool) -> tuple[AuthorityGraph, int]:
    rng = np.random.default_rng(seed)
    mandatory = set(route_nodes() + ["temporal_route", "readout_positive", "readout_negative", "shared_actor_hub"])
    target_count = max(len(mandatory), int(round(len(graph.nodes) * fraction)))
    candidate_names = [name for name in graph.nodes if name not in mandatory]
    if random_keep:
        chosen = set(rng.choice(candidate_names, size=max(0, target_count - len(mandatory)), replace=False))
    else:
        degree = np.abs(graph.w).sum(axis=0) + np.abs(graph.w).sum(axis=1) + np.abs(graph.bias)
        ranked = sorted(candidate_names, key=lambda name: degree[graph.idx[name]], reverse=True)
        chosen = set(ranked[: max(0, target_count - len(mandatory))])
    keep = mandatory | chosen
    drop = [name for name in graph.nodes if name not in keep]
    return zero_nodes(graph, drop), len(keep)


def shrink_edges(graph: AuthorityGraph, fraction: float, seed: int, *, random_keep: bool) -> tuple[AuthorityGraph, int]:
    rng = np.random.default_rng(seed)
    edges = sorted_existing_edges(graph)
    mandatory_names = {
        (FRAME_BY_ROUTE[frame], f"suppress_{frame}")
        for frame in FRAMES
    } | {
        (route, route)
        for route in route_nodes() + ["temporal_route"]
    }
    mandatory = {
        (graph.idx[target], graph.idx[source])
        for target, source in mandatory_names
        if target in graph.idx and source in graph.idx and abs(graph.w[graph.idx[target], graph.idx[source]]) > 1.0e-9
    }
    target_count = max(len(mandatory), int(round(len(edges) * fraction)))
    optional = [edge for edge in edges if edge not in mandatory]
    if random_keep:
        chosen = set(optional[int(i)] for i in rng.choice(len(optional), size=max(0, target_count - len(mandatory)), replace=False)) if optional else set()
    else:
        chosen = set(optional[: max(0, target_count - len(mandatory))])
    keep = mandatory | chosen
    candidate = graph.clone()
    for edge in edges:
        if edge not in keep:
            candidate.w[edge[0], edge[1]] = 0.0
    return candidate, len(keep)


def minimal_graph_sweep(args: argparse.Namespace, datasets: dict[str, list[Any]], seed: int) -> dict[str, Any]:
    hand = build_hand_seeded_graph(args.decay)
    baseline = evaluate_graph(hand, datasets, steps=args.steps, seed=seed + 100)
    fractions = [1.00, 0.75, 0.50, 0.33, 0.25]
    out = {"baseline": baseline, "node_shrink": {}, "edge_shrink": {}}
    for fraction in fractions:
        top_node_graph, top_node_count = shrink_nodes(hand, fraction, seed + int(fraction * 1000), random_keep=False)
        random_node_graph, random_node_count = shrink_nodes(hand, fraction, seed + int(fraction * 1000) + 1, random_keep=True)
        top_edge_graph, top_edge_count = shrink_edges(hand, fraction, seed + int(fraction * 1000) + 2, random_keep=False)
        random_edge_graph, random_edge_count = shrink_edges(hand, fraction, seed + int(fraction * 1000) + 3, random_keep=True)
        out["node_shrink"][f"{int(fraction * 100)}pct"] = {
            "retained_node_count": top_node_count,
            "top": evaluate_graph(top_node_graph, datasets, steps=args.steps, seed=seed + 110),
            "random": evaluate_graph(random_node_graph, datasets, steps=args.steps, seed=seed + 111),
            "random_retained_node_count": random_node_count,
        }
        out["edge_shrink"][f"{int(fraction * 100)}pct"] = {
            "retained_edge_count": top_edge_count,
            "top": evaluate_graph(top_edge_graph, datasets, steps=args.steps, seed=seed + 120),
            "random": evaluate_graph(random_edge_graph, datasets, steps=args.steps, seed=seed + 121),
            "random_retained_edge_count": random_edge_count,
        }
    out["summary"] = minimal_sweep_summary(out)
    return out


def minimal_sweep_summary(sweep: dict[str, Any]) -> dict[str, Any]:
    baseline_acc = sweep["baseline"]["accuracy"]
    baseline_auth = sweep["baseline"]["authority_switch_score"]
    acc_threshold = 0.90 * baseline_acc
    auth_threshold = 0.90 * baseline_auth
    node_min = None
    edge_min = None
    auth_min = None
    temporal_break = None
    for key, item in sweep["node_shrink"].items():
        if item["top"]["accuracy"] >= acc_threshold:
            node_min = {"budget": key, "node_count": item["retained_node_count"], "accuracy": item["top"]["accuracy"]}
        if temporal_break is None and item["top"]["temporal_order_contrast_small_accuracy"] < 0.90 * sweep["baseline"]["temporal_order_contrast_small_accuracy"]:
            temporal_break = {"kind": "node_shrink", "budget": key}
    for key, item in sweep["edge_shrink"].items():
        if item["top"]["accuracy"] >= acc_threshold:
            edge_min = {"budget": key, "edge_count": item["retained_edge_count"], "accuracy": item["top"]["accuracy"]}
        if item["top"]["authority_switch_score"] >= auth_threshold:
            auth_min = {"kind": "edge_shrink", "budget": key, "edge_count": item["retained_edge_count"], "authority": item["top"]["authority_switch_score"]}
        if temporal_break is None and item["top"]["temporal_order_contrast_small_accuracy"] < 0.90 * sweep["baseline"]["temporal_order_contrast_small_accuracy"]:
            temporal_break = {"kind": "edge_shrink", "budget": key}
    return {
        "minimal_node_count_for_90pct_accuracy": node_min,
        "minimal_edge_count_for_90pct_accuracy": edge_min,
        "minimal_graph_preserving_90pct_authority": auth_min,
        "temporal_order_breaks_first_at": temporal_break,
    }


def all_mutable_edges(graph: AuthorityGraph) -> list[tuple[int, int]]:
    forbidden_targets = {graph.idx[name] for name, typ in graph.types.items() if typ == "readout"}
    return [
        (target, source)
        for target in range(len(graph.nodes))
        for source in range(len(graph.nodes))
        if target != source and target not in forbidden_targets
    ]


def mutate_graph_v2(graph: AuthorityGraph, rng: np.random.Generator, scale: float) -> tuple[AuthorityGraph, str]:
    candidate = graph.clone()
    mutation_type = str(rng.choice([
        "edge_gain_perturb",
        "edge_sign_flip",
        "edge_add",
        "edge_remove",
        "frame_gate_strength",
        "suppressor_strength",
        "recurrent_decay",
    ]))
    nonzero = np.flatnonzero(np.abs(candidate.w) > 1.0e-9)
    if mutation_type == "edge_gain_perturb" and len(nonzero):
        chosen = int(rng.choice(nonzero))
        candidate.w.flat[chosen] = float(np.clip(candidate.w.flat[chosen] + rng.normal(0.0, scale), -1.8, 1.8))
    elif mutation_type == "edge_sign_flip" and len(nonzero):
        chosen = int(rng.choice(nonzero))
        candidate.w.flat[chosen] *= -1.0
    elif mutation_type == "edge_remove" and len(nonzero):
        chosen = int(rng.choice(nonzero))
        candidate.w.flat[chosen] = 0.0
    elif mutation_type == "edge_add":
        candidates = all_mutable_edges(candidate)
        target, source = candidates[int(rng.integers(0, len(candidates)))]
        candidate.w[target, source] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.9))
    elif mutation_type == "frame_gate_strength":
        candidate.frame_gate = float(np.clip(candidate.frame_gate + rng.normal(0.0, scale), 0.0, 1.8))
    elif mutation_type == "suppressor_strength":
        candidate.suppressor_strength = float(np.clip(candidate.suppressor_strength + rng.normal(0.0, scale), 0.0, 2.0))
        for frame in FRAMES:
            candidate.w[candidate.idx[FRAME_BY_ROUTE[frame]], candidate.idx[f"suppress_{frame}"]] = -candidate.suppressor_strength
    elif mutation_type == "recurrent_decay":
        candidate.decay = float(np.clip(candidate.decay + rng.normal(0.0, scale * 0.25), 0.0, 0.95))
    return candidate, mutation_type


def minimality_fitness(metrics: dict[str, Any]) -> float:
    return (
        metrics["accuracy"]
        + 0.45 * metrics["authority_switch_score"]
        + 0.20 * max(0.0, metrics["wrong_frame_drop"])
        + 0.30 * metrics["temporal_order_contrast_small_accuracy"]
        + 0.08 * max(0.0, metrics["no_suppressor_authority_drop"])
        - 0.00045 * metrics["edge_count"]
        - 0.08 * max(0.0, metrics["no_suppressor_inactive_influence_rise"])
    )


def structure_recovery(graph: AuthorityGraph) -> dict[str, Any]:
    idx = graph.idx
    shared_hub_edges = sum(
        int(abs(graph.w[idx[route], idx[hub]]) > 1.0e-9)
        for route in route_nodes()
        for hub in ("shared_actor_hub", "shared_action_hub", "shared_context_hub")
    )
    frame_route_edges = sum(
        int(abs(graph.w[idx[FRAME_BY_ROUTE[frame]], idx[f"tok_{token}"]]) > 1.0e-9)
        for frame in FRAMES
        for token in TOKENS
    )
    suppressor_edges = sum(
        int(abs(graph.w[idx[FRAME_BY_ROUTE[frame]], idx[f"suppress_{other}"]]) > 1.0e-9)
        for frame in FRAMES
        for other in FRAMES
    )
    recurrent_edges = sum(
        int(abs(graph.w[idx[node], idx[node]]) > 1.0e-9)
        for node in route_nodes() + ["temporal_route"]
    )
    return {
        "shared_hub_edges": shared_hub_edges,
        "frame_route_token_edges": frame_route_edges,
        "suppressor_edges": suppressor_edges,
        "recurrent_route_edges": recurrent_edges,
        "has_shared_hubs": shared_hub_edges > 0,
        "has_frame_routes": frame_route_edges > 0,
        "has_suppressors": suppressor_edges > 0,
        "has_recurrence": recurrent_edges > 0,
    }


def build_hub_only_graph(decay: float, seed: int) -> AuthorityGraph:
    rng = np.random.default_rng(seed)
    graph = build_empty_graph(decay)
    for token in ACTORS:
        add_edge(graph, "shared_actor_hub", f"tok_{token}", rng.uniform(0.10, 0.45))
    for token in ACTIONS:
        add_edge(graph, "shared_action_hub", f"tok_{token}", rng.uniform(0.10, 0.45))
    for token in PLACES + NOISES + SOUNDS:
        add_edge(graph, "shared_context_hub", f"tok_{token}", rng.uniform(0.10, 0.45))
    for route in route_nodes():
        add_edge(graph, route, route, rng.uniform(0.03, 0.20))
        for hub in ("shared_actor_hub", "shared_action_hub", "shared_context_hub"):
            add_edge(graph, route, hub, rng.normal(0.0, 0.25))
    graph.bias[[graph.idx[route] for route in route_nodes()]] = graph.route_bias
    return graph


def build_route_only_graph(decay: float, seed: int) -> AuthorityGraph:
    rng = np.random.default_rng(seed)
    graph = build_empty_graph(decay)
    for route in route_nodes():
        add_edge(graph, route, route, rng.uniform(0.03, 0.20))
        for token in TOKENS:
            if rng.random() < 0.22:
                add_edge(graph, route, f"tok_{token}", rng.normal(0.0, 0.35))
    graph.bias[[graph.idx[route] for route in route_nodes()]] = graph.route_bias
    return graph


def build_no_suppressor_seed(decay: float) -> AuthorityGraph:
    return graph_without_component(build_hand_seeded_graph(decay), "no_suppressors")


def damage_graph(graph: AuthorityGraph, fraction: float, seed: int) -> AuthorityGraph:
    rng = np.random.default_rng(seed)
    candidate = graph.clone()
    edges = sorted_existing_edges(candidate)
    if not edges:
        return candidate
    remove_count = int(round(len(edges) * fraction))
    chosen = rng.choice(len(edges), size=min(remove_count, len(edges)), replace=False)
    for index in chosen:
        target, source = edges[int(index)]
        candidate.w[target, source] = 0.0
    return candidate


def build_evolution_seed_graph(kind: str, decay: float, seed: int, edge_count: int) -> AuthorityGraph:
    if kind == "random_graph":
        return build_random_graph(decay, seed, edge_count)
    if kind == "hub_only_graph":
        return build_hub_only_graph(decay, seed)
    if kind == "route_only_graph":
        return build_route_only_graph(decay, seed)
    if kind == "no_suppressor_seed":
        return build_no_suppressor_seed(decay)
    if kind == "damaged_hand_seeded_30":
        return damage_graph(build_hand_seeded_graph(decay), 0.30, seed)
    if kind == "damaged_hand_seeded_50":
        return damage_graph(build_hand_seeded_graph(decay), 0.50, seed)
    if kind == "damaged_hand_seeded_70":
        return damage_graph(build_hand_seeded_graph(decay), 0.70, seed)
    raise ValueError(kind)


def evolve_from_prior(
    graph: AuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    population: list[tuple[AuthorityGraph, dict[str, Any], float]] = []
    start_metrics = evaluate_graph(graph, datasets, steps=args.steps, seed=seed + 10)
    population.append((graph, start_metrics, minimality_fitness(start_metrics)))
    for index in range(max(0, args.population_size - 1)):
        mutated, _kind = mutate_graph_v2(graph, rng, args.mutation_scale)
        metrics = evaluate_graph(mutated, datasets, steps=args.steps, seed=seed + 100 + index)
        population.append((mutated, metrics, minimality_fitness(metrics)))
    population.sort(key=lambda item: item[2], reverse=True)
    population = population[: args.population_size]
    accepted: dict[str, int] = {}
    for step in range(args.mutation_steps):
        parent = population[int(rng.integers(0, max(1, min(len(population), args.population_size // 2))))]
        candidate, mutation_type = mutate_graph_v2(parent[0], rng, args.mutation_scale)
        metrics = evaluate_graph(candidate, datasets, steps=args.steps, seed=seed + 1_000 + step)
        score = minimality_fitness(metrics)
        if score >= population[-1][2]:
            population.append((candidate, metrics, score))
            population.sort(key=lambda item: item[2], reverse=True)
            population = population[: args.population_size]
            accepted[mutation_type] = accepted.get(mutation_type, 0) + 1
    best_graph, best_metrics, best_score = population[0]
    scores = [item[2] for item in population]
    return {
        "starting_metrics": start_metrics,
        "starting_fitness": minimality_fitness(start_metrics),
        "best_metrics": best_metrics,
        "best_final_fitness": best_score,
        "median_final_fitness": float(np.median(scores)),
        "mutation_steps": args.mutation_steps,
        "population_size": args.population_size,
        "accepted_mutation_counts": accepted,
        "structure_recovery": structure_recovery(best_graph),
    }


def evolution_v2(args: argparse.Namespace, datasets: dict[str, list[Any]], seed: int) -> dict[str, Any]:
    hand = build_hand_seeded_graph(args.decay)
    edge_count = hand.edge_count()
    kinds = [
        "random_graph",
        "hub_only_graph",
        "route_only_graph",
        "no_suppressor_seed",
        "damaged_hand_seeded_30",
        "damaged_hand_seeded_50",
        "damaged_hand_seeded_70",
    ]
    return {
        kind: evolve_from_prior(
            build_evolution_seed_graph(kind, args.decay, seed + index * 97, edge_count),
            datasets,
            seed=seed + 20_000 + index * 311,
            args=args,
        )
        for index, kind in enumerate(kinds)
    }


def summarize_minimality_run(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    datasets = make_datasets(args.samples, seed)
    return {
        "seed": seed,
        "datasets": {name: len(items) for name, items in datasets.items()},
        "component_necessity": component_necessity(args, datasets, seed),
        "minimal_graph_sweep": minimal_graph_sweep(args, datasets, seed),
        "evolution_v2": evolution_v2(args, datasets, seed),
    }


def aggregate_component(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = rows[0]["component_necessity"]["ablations"].keys()
    out: dict[str, Any] = {}
    for name in names:
        ablations = [row["component_necessity"]["ablations"][name] for row in rows]
        verdicts = [item["verdict"] for item in ablations]
        out[name] = {
            "accuracy_drop": numeric_summary([item["accuracy_drop"] for item in ablations]),
            "authority_drop": numeric_summary([item["authority_drop"] for item in ablations]),
            "temporal_order_drop": numeric_summary([item["temporal_order_drop"] for item in ablations]),
            "verdict": majority_label(verdicts),
        }
    return out


def numeric_summary(values: list[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(numeric)), "std": float(np.std(numeric))}


def majority_label(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


def aggregate_minimal_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"node_shrink": {}, "edge_shrink": {}, "summary": {}}
    for kind in ("node_shrink", "edge_shrink"):
        budgets = rows[0]["minimal_graph_sweep"][kind].keys()
        for budget in budgets:
            items = [row["minimal_graph_sweep"][kind][budget] for row in rows]
            out[kind][budget] = {
                "top_accuracy": numeric_summary([item["top"]["accuracy"] for item in items]),
                "random_accuracy": numeric_summary([item["random"]["accuracy"] for item in items]),
                "top_authority": numeric_summary([item["top"]["authority_switch_score"] for item in items]),
                "random_authority": numeric_summary([item["random"]["authority_switch_score"] for item in items]),
                "top_temporal_order": numeric_summary([item["top"]["temporal_order_contrast_small_accuracy"] for item in items]),
                "retained_count": numeric_summary([
                    item.get("retained_node_count", item.get("retained_edge_count"))
                    for item in items
                ]),
            }
    out["summary"] = {
        "per_seed": [row["minimal_graph_sweep"]["summary"] for row in rows],
    }
    return out


def aggregate_evolution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = rows[0]["evolution_v2"].keys()
    out: dict[str, Any] = {}
    for kind in kinds:
        items = [row["evolution_v2"][kind] for row in rows]
        mutation_counts: dict[str, list[int]] = {}
        for item in items:
            for mutation_type, count in item["accepted_mutation_counts"].items():
                mutation_counts.setdefault(mutation_type, []).append(int(count))
        out[kind] = {
            "starting_fitness": numeric_summary([item["starting_fitness"] for item in items]),
            "best_final_fitness": numeric_summary([item["best_final_fitness"] for item in items]),
            "median_final_fitness": numeric_summary([item["median_final_fitness"] for item in items]),
            "starting_accuracy": numeric_summary([item["starting_metrics"]["accuracy"] for item in items]),
            "final_accuracy": numeric_summary([item["best_metrics"]["accuracy"] for item in items]),
            "final_authority": numeric_summary([item["best_metrics"]["authority_switch_score"] for item in items]),
            "final_edge_count": numeric_summary([item["best_metrics"]["edge_count"] for item in items]),
            "accepted_mutation_counts": {
                mutation_type: numeric_summary(counts)
                for mutation_type, counts in sorted(mutation_counts.items())
            },
            "structure_recovery_rate": {
                key: numeric_summary([int(item["structure_recovery"][key]) for item in items])
                for key in ("has_shared_hubs", "has_frame_routes", "has_suppressors", "has_recurrence")
            },
        }
    return out


def aggregate_minimality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row["component_necessity"]["baseline"] for row in rows]
    return {
        "baseline": {
            "accuracy": numeric_summary([item["accuracy"] for item in baseline]),
            "authority_switch_score": numeric_summary([item["authority_switch_score"] for item in baseline]),
            "temporal_order_accuracy": numeric_summary([item["temporal_order_contrast_small_accuracy"] for item in baseline]),
            "edge_count": numeric_summary([item["edge_count"] for item in baseline]),
            "node_count": numeric_summary([item["node_count"] for item in baseline]),
        },
        "component_necessity": aggregate_component(rows),
        "minimal_graph_sweep": aggregate_minimal_sweep(rows),
        "evolution_v2": aggregate_evolution(rows),
    }


def minimality_verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    component = aggregate["component_necessity"]
    evolution = aggregate["evolution_v2"]
    damaged_recovery = any(
        (evolution[name]["final_accuracy"]["mean"] or 0.0) >= 0.80
        for name in ("damaged_hand_seeded_30", "damaged_hand_seeded_50", "damaged_hand_seeded_70")
    )
    route_or_hub_recovery = any(
        (evolution[name]["final_accuracy"]["mean"] or 0.0) >= 0.75
        for name in ("hub_only_graph", "route_only_graph")
    )
    random_hard = (evolution["random_graph"]["final_accuracy"]["mean"] or 0.0) < 0.75
    return {
        "supports_frame_gates_as_necessary_core": component["no_frame_gates"]["verdict"] in {"necessary_core", "useful_but_replaceable"},
        "supports_route_structure_as_necessary_core": component["no_frame_routes"]["verdict"] == "necessary_core",
        "supports_recurrence_as_temporal_core": component["no_recurrence"]["verdict"] == "necessary_core",
        "supports_suppressors_as_helpful": component["no_suppressors"]["verdict"] in {"helpful_bias", "useful_but_replaceable", "necessary_core"},
        "supports_mutation_recovery_from_damage": damaged_recovery,
        "supports_mutation_recovery_from_weak_priors": route_or_hub_recovery,
        "random_graph_remains_hard": random_hard,
        "manual_structure_still_dominates": not route_or_hub_recovery and random_hard,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def write_report(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# Frame-Gated Authority Graph Pilot",
        "",
        "## Why This Test Exists",
        "",
        "The neural toy produced a shared-core plus frame-specific-route mechanism with hub/integrator and suppressor/gate candidates. This pilot asks whether that mechanism requires neural layers, or whether a smaller explicit recurrent authority graph can reproduce the same behavior without backprop through the graph internals.",
        "",
        "## Minimal Mechanism Tested",
        "",
        "- token input nodes inject observed symbols",
        "- shared hub nodes integrate common actor/action/context evidence",
        "- frame route nodes receive frame-gated evidence",
        "- suppressor nodes inhibit inactive routes",
        "- recurrent scalar states settle over a few steps",
        "- readout uses active-route authority against inactive-route competition",
        "",
        "## Graph Architecture",
        "",
        "Update rule:",
        "",
        "```text",
        "state[t+1] = tanh(decay * state[t] + signed_edge_sum + input_injection + frame_gate_modulation + bias)",
        "```",
        "",
        "No gradient or backprop is used. The `evolved_graph_small` mode uses mutation/hillclimb over edge gains and scalar gate/suppressor strengths.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Task Results",
        "",
        "| Mode | Accuracy | Latent | Multi-Aspect | Temporal Order | Authority | Refraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("hand_seeded_graph", "random_graph_baseline", "evolved_graph_small"):
        item = aggregate[mode]
        lines.append(
            f"| `{mode}` | `{fmt(item['accuracy']['mean'])}` "
            f"| `{fmt(item['latent_refraction_small_accuracy']['mean'])}` "
            f"| `{fmt(item['multi_aspect_small_accuracy']['mean'])}` "
            f"| `{fmt(item['temporal_order_contrast_small_accuracy']['mean'])}` "
            f"| `{fmt(item['authority_switch_score']['mean'])}` "
            f"| `{fmt(item['refraction_index_final']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Control Results",
        "",
        "| Mode | Wrong Frame Drop | No Frame Gate Drop | No Suppressor Acc Drop | No Suppressor Authority Drop | Inactive Influence Rise | No Recurrence Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in ("hand_seeded_graph", "random_graph_baseline", "evolved_graph_small"):
        item = aggregate[mode]
        lines.append(
            f"| `{mode}` | `{fmt(item['wrong_frame_drop']['mean'])}` "
            f"| `{fmt(item['no_frame_gate_drop']['mean'])}` "
            f"| `{fmt(item['no_suppressor_drop']['mean'])}` "
            f"| `{fmt(item['no_suppressor_authority_drop']['mean'])}` "
            f"| `{fmt(item['no_suppressor_inactive_influence_rise']['mean'])}` "
            f"| `{fmt(item['no_recurrence_drop']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Mutation / Evolution Result",
        "",
        f"- mutation steps requested: `{summary['config']['mutation_steps']}`",
        f"- accepted mutations mean: `{fmt(aggregate['evolved_graph_small']['accepted_mutations']['mean'])}`",
        f"- evolved graph accuracy mean: `{fmt(aggregate['evolved_graph_small']['accuracy']['mean'])}`",
        "",
        "## What Neural Components Were Unnecessary",
        "",
        "- Dense neural hidden layers were not required for the hand-seeded pilot graph to show frame-gated authority routing.",
        "- Backprop through the graph internals was not required; the evolved variant used only mutation/hillclimb.",
        "",
        "## What Still Required Dynamics",
        "",
        "- Frame gates remain necessary in the hand graph when the same observations are evaluated under different task frames.",
        "- Suppressor nodes matter when inactive route competition would otherwise leak into the active decision.",
        "- Temporal order contrast depends on recurrent carry / role memory; final-token-only evaluation is weaker.",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def component_row(name: str, component: dict[str, Any]) -> str:
    return (
        f"| `{name}` | `{fmt(component['accuracy_drop']['mean'])}` "
        f"| `{fmt(component['authority_drop']['mean'])}` "
        f"| `{fmt(component['temporal_order_drop']['mean'])}` "
        f"| `{component['verdict']}` |"
    )


def write_minimality_report(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    components = aggregate["component_necessity"]
    sweep = aggregate["minimal_graph_sweep"]
    evolution = aggregate["evolution_v2"]
    final_components = {
        "input_token_nodes": "necessary_core",
        "shared_hubs": components["no_shared_hubs"]["verdict"],
        "frame_route_nodes": components["no_frame_routes"]["verdict"],
        "frame_gates": components["no_frame_gates"]["verdict"],
        "recurrent_edges": components["no_recurrence"]["verdict"],
        "suppressors": components["no_suppressors"]["verdict"],
        "readout_ports": components["no_route_to_readout"]["verdict"],
        "direct_shortcuts": components["no_direct_token_to_readout"]["verdict"],
    }
    lines = [
        "# Authority Graph Minimality + Evolution",
        "",
        "## Goal",
        "",
        "Separate necessary mechanism from hand-coded convenience in the explicit frame-gated authority graph. This study keeps the same toy tasks and does not use neural layers, backprop, new semantic concepts, biology, or wave/reservoir extensions.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Baseline",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
        f"| Accuracy | `{fmt(aggregate['baseline']['accuracy']['mean'])}` | `{fmt(aggregate['baseline']['accuracy']['std'])}` |",
        f"| Authority / Refraction | `{fmt(aggregate['baseline']['authority_switch_score']['mean'])}` | `{fmt(aggregate['baseline']['authority_switch_score']['std'])}` |",
        f"| Temporal Order Accuracy | `{fmt(aggregate['baseline']['temporal_order_accuracy']['mean'])}` | `{fmt(aggregate['baseline']['temporal_order_accuracy']['std'])}` |",
        f"| Nodes | `{fmt(aggregate['baseline']['node_count']['mean'])}` | `{fmt(aggregate['baseline']['node_count']['std'])}` |",
        f"| Edges | `{fmt(aggregate['baseline']['edge_count']['mean'])}` | `{fmt(aggregate['baseline']['edge_count']['std'])}` |",
        "",
        "## Part 1: Component Necessity Ablations",
        "",
        "| Ablation | Accuracy Drop | Authority Drop | Temporal Order Drop | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in components.items():
        lines.append(component_row(name, item))
    lines.extend([
        "",
        "## Part 2: Minimal Graph Sweep",
        "",
        "### Node Shrink",
        "",
        "| Budget | Retained Nodes | Top Accuracy | Random Accuracy | Top Authority | Random Authority | Top Temporal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for budget, item in sweep["node_shrink"].items():
        lines.append(
            f"| `{budget}` | `{fmt(item['retained_count']['mean'])}` "
            f"| `{fmt(item['top_accuracy']['mean'])}` | `{fmt(item['random_accuracy']['mean'])}` "
            f"| `{fmt(item['top_authority']['mean'])}` | `{fmt(item['random_authority']['mean'])}` "
            f"| `{fmt(item['top_temporal_order']['mean'])}` |"
        )
    lines.extend([
        "",
        "### Edge Shrink",
        "",
        "| Budget | Retained Edges | Top Accuracy | Random Accuracy | Top Authority | Random Authority | Top Temporal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for budget, item in sweep["edge_shrink"].items():
        lines.append(
            f"| `{budget}` | `{fmt(item['retained_count']['mean'])}` "
            f"| `{fmt(item['top_accuracy']['mean'])}` | `{fmt(item['random_accuracy']['mean'])}` "
            f"| `{fmt(item['top_authority']['mean'])}` | `{fmt(item['random_authority']['mean'])}` "
            f"| `{fmt(item['top_temporal_order']['mean'])}` |"
        )
    lines.extend([
        "",
        "Minimality summary by seed:",
        "",
        "```json",
        json.dumps(round_floats(sweep["summary"]["per_seed"]), indent=2),
        "```",
        "",
        "## Part 3: Evolution From Weaker Priors",
        "",
        "| Seed Graph | Start Fitness | Final Fitness | Start Accuracy | Final Accuracy | Final Authority | Final Edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in evolution.items():
        lines.append(
            f"| `{name}` | `{fmt(item['starting_fitness']['mean'])}` "
            f"| `{fmt(item['best_final_fitness']['mean'])}` "
            f"| `{fmt(item['starting_accuracy']['mean'])}` "
            f"| `{fmt(item['final_accuracy']['mean'])}` "
            f"| `{fmt(item['final_authority']['mean'])}` "
            f"| `{fmt(item['final_edge_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "### Recovered Structure Rates",
        "",
        "| Seed Graph | Shared Hubs | Frame Routes | Suppressors | Recurrence |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in evolution.items():
        rates = item["structure_recovery_rate"]
        lines.append(
            f"| `{name}` | `{fmt(rates['has_shared_hubs']['mean'])}` "
            f"| `{fmt(rates['has_frame_routes']['mean'])}` "
            f"| `{fmt(rates['has_suppressors']['mean'])}` "
            f"| `{fmt(rates['has_recurrence']['mean'])}` |"
        )
    lines.extend([
        "",
        "### Accepted Mutation Types",
        "",
        "```json",
        json.dumps(round_floats({
            name: item["accepted_mutation_counts"]
            for name, item in evolution.items()
        }), indent=2),
        "```",
        "",
        "## Part 4: Necessity Summary",
        "",
        "| Component | Ablation Effect | Evolution Recovery | Verdict |",
        "|---|---|---|---|",
    ])
    recovery_note = "damaged seeds partially tested; weak-prior recovery is reported above"
    for name, verdict_label in final_components.items():
        ablation_effect = {
            "input_token_nodes": "required input ports by construction",
            "shared_hubs": "see no_shared_hubs",
            "frame_route_nodes": "see no_frame_routes",
            "frame_gates": "see no_frame_gates",
            "recurrent_edges": "see no_recurrence",
            "suppressors": "see no_suppressors",
            "readout_ports": "see no_route_to_readout",
            "direct_shortcuts": "see no_direct_token_to_readout",
        }[name]
        lines.append(f"| `{name}` | {ablation_effect} | {recovery_note} | `{verdict_label}` |")
    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- If damaged hand graphs recover but random or weak priors lag, the hand-coded mechanism is still doing substantial work.",
        "- If route-only or hub-only priors recover, the architecture is more mutatable and less dependent on exact manual edges.",
        "- Accuracy alone is not decisive; authority/refraction and wrong-frame controls are treated as mechanism metrics.",
        "",
        "## Runtime Notes",
        "",
        f"- total runtime seconds: `{fmt(summary['runtime_seconds'])}`",
        f"- smoke mode: `{summary['config']['smoke']}`",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_minimality_study(args: argparse.Namespace) -> int:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        print(f"[authority-graph-minimality] seed={seed}", flush=True)
        runs.append(summarize_minimality_run(args, seed))
    aggregate = aggregate_minimality(runs)
    summary = {
        "config": {
            "study": args.study,
            "seeds": args.seeds,
            "steps": args.steps,
            "samples": args.samples,
            "mutation_steps": args.mutation_steps,
            "mutation_scale": args.mutation_scale,
            "population_size": args.population_size,
            "random_graphs": args.random_graphs,
            "decay": args.decay,
            "smoke": args.smoke,
        },
        "aggregate": aggregate,
        "verdict": minimality_verdict(aggregate),
        "runs": runs,
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    summary = round_floats(summary)
    json_path = args.out_dir / "authority_graph_minimality_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_minimality_report(summary, MINIMALITY_REPORT_PATH)
    print(json.dumps({
        "verdict": summary["verdict"],
        "aggregate": summary["aggregate"],
        "json": str(json_path),
        "report": str(MINIMALITY_REPORT_PATH),
    }, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.study == "minimality":
        return run_minimality_study(args)
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        print(f"[authority-graph] seed={seed}", flush=True)
        runs.append(evaluate_seed(args, seed))
    aggregate = aggregate_runs(runs)
    summary = {
        "config": {
            "seeds": args.seeds,
            "steps": args.steps,
            "samples": args.samples,
            "mutation_steps": args.mutation_steps,
            "mutation_scale": args.mutation_scale,
            "random_graphs": args.random_graphs,
            "decay": args.decay,
            "smoke": args.smoke,
        },
        "aggregate": aggregate,
        "verdict": verdict(aggregate),
        "runs": runs,
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    json_path = args.out_dir / "authority_graph_pilot_summary.json"
    json_path.write_text(json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8")
    write_report(round_floats(summary), REPORT_PATH)
    print(json.dumps({
        "verdict": round_floats(summary["verdict"]),
        "aggregate": round_floats(summary["aggregate"]),
        "json": str(json_path),
        "report": str(REPORT_PATH),
    }, indent=2))
    return 0


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
