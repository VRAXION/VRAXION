#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_authority_graph_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_GUIDED_PRUNING_PILOT.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-guided-pruning"
SUMMARY_NAME = "authority_graph_guided_pruning_summary.json"

ARMS = [
    "hand_seeded",
    "damaged_hand_seeded_50",
    "random_graph_baseline",
    "grammar_v2_untrained",
    "grammar_v2_mutation_only",
    "overcomplete_guided_train",
    "overcomplete_guided_prune_30",
    "overcomplete_guided_prune_50",
    "overcomplete_guided_prune_70",
    "overcomplete_guided_prune_50_repair",
]
SUCCESS = {
    "overall_accuracy": 0.90,
    "temporal_order_accuracy": 0.90,
    "authority_refraction_score": 0.25,
    "wrong_frame_drop": 0.20,
}


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    weight: float
    edge_type: str


@dataclass
class GraphSpec:
    name: str
    nodes: list[str]
    types: dict[str, str]
    edges: list[EdgeSpec]
    bias: dict[str, float]
    frame_gate: dict[str, float]
    suppressor_strength: float
    decay: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differentiable guided pruning pilot for explicit authority graphs.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--final-test-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fine-tune-epochs", type=int, default=80)
    parser.add_argument("--mutation-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2.5e-3)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-runtime-hours", type=float, default=5.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {', '.join(unknown)}")
    if args.smoke:
        args.seeds = 1
        args.steps = 3
        args.train_samples = 48
        args.validation_samples = 48
        args.final_test_samples = 96
        args.epochs = 3
        args.fine_tune_epochs = 2
        args.mutation_steps = 4
        args.checkpoint_every = 1
        args.max_runtime_hours = 0.25
        args.arms = [
            "hand_seeded",
            "grammar_v2_untrained",
            "overcomplete_guided_train",
            "overcomplete_guided_prune_50",
        ]
    return args


def base_nodes() -> tuple[list[str], dict[str, str]]:
    graph = pilot.build_empty_graph(0.35)
    return list(graph.nodes), dict(graph.types)


def add_edge(edges: list[EdgeSpec], source: str, target: str, weight: float, edge_type: str) -> None:
    edges.append(EdgeSpec(source=source, target=target, weight=float(weight), edge_type=edge_type))


def graph_to_spec(graph: pilot.AuthorityGraph, name: str, *, add_explicit_readout: bool = True) -> GraphSpec:
    edges: list[EdgeSpec] = []
    for target, source in np.argwhere(np.abs(graph.w) > 1.0e-9):
        source_name = graph.nodes[int(source)]
        target_name = graph.nodes[int(target)]
        edge_type = classify_edge(graph.types, source_name, target_name)
        if graph.types.get(source_name) == "token_input" and graph.types.get(target_name) == "readout":
            continue
        edges.append(EdgeSpec(source_name, target_name, float(graph.w[int(target), int(source)]), edge_type))
    if add_explicit_readout:
        add_readout_edges(edges, scale=1.0)
    bias = {node: float(graph.bias[graph.idx[node]]) for node in graph.nodes if abs(float(graph.bias[graph.idx[node]])) > 1.0e-9}
    return GraphSpec(
        name=name,
        nodes=list(graph.nodes),
        types=dict(graph.types),
        edges=dedupe_edges(edges),
        bias=bias,
        frame_gate={frame: float(graph.frame_gate) for frame in pilot.FRAMES},
        suppressor_strength=float(graph.suppressor_strength),
        decay=float(graph.decay),
    )


def classify_edge(types: dict[str, str], source: str, target: str) -> str:
    source_type = types.get(source, "")
    target_type = types.get(target, "")
    if source_type == "token_input" and target_type == "frame_route":
        return "token->route"
    if source_type == "token_input" and target_type == "shared_hub":
        return "token->hub"
    if source_type == "shared_hub" and target_type == "frame_route":
        return "hub->route"
    if source_type == "suppressor" and target_type == "frame_route":
        return "suppressor->route"
    if source == target and target_type == "frame_route":
        return "route recurrent"
    if source_type == "temporal_role" and target == "temporal_route":
        return "temporal role"
    if target_type == "readout":
        return "route->readout"
    return f"{source_type}->{target_type}"


def dedupe_edges(edges: list[EdgeSpec]) -> list[EdgeSpec]:
    merged: dict[tuple[str, str], EdgeSpec] = {}
    for edge in edges:
        merged[(edge.source, edge.target)] = edge
    return list(merged.values())


def add_readout_edges(edges: list[EdgeSpec], *, scale: float) -> None:
    for route in pilot.route_nodes() + ["temporal_route"]:
        add_edge(edges, route, "readout_positive", scale, "route->readout")
        add_edge(edges, route, "readout_negative", -scale, "route->readout")


def build_hand_spec() -> GraphSpec:
    return graph_to_spec(pilot.build_hand_seeded_graph(0.35), "hand_seeded")


def damage_spec(spec: GraphSpec, fraction: float, seed: int) -> GraphSpec:
    rng = np.random.default_rng(seed)
    protected = {"route->readout"}
    candidates = [i for i, edge in enumerate(spec.edges) if edge.edge_type not in protected]
    drop_count = int(round(len(candidates) * fraction))
    drop = set(int(i) for i in rng.choice(candidates, size=min(drop_count, len(candidates)), replace=False))
    return GraphSpec(
        name=f"{spec.name}_damaged_{int(fraction * 100)}",
        nodes=spec.nodes,
        types=spec.types,
        edges=[edge for i, edge in enumerate(spec.edges) if i not in drop],
        bias=dict(spec.bias),
        frame_gate=dict(spec.frame_gate),
        suppressor_strength=spec.suppressor_strength,
        decay=spec.decay,
    )


def build_overcomplete_spec(seed: int, *, name: str = "overcomplete_scaffold") -> GraphSpec:
    rng = np.random.default_rng(seed)
    nodes, types = base_nodes()
    edges: list[EdgeSpec] = []
    bias: dict[str, float] = {}
    for route in pilot.route_nodes():
        bias[route] = float(rng.uniform(-1.35, -0.75))
        add_edge(edges, route, route, float(rng.uniform(0.08, 0.22)), "route recurrent")
    bias["temporal_route"] = float(rng.uniform(-0.20, 0.05))
    add_edge(edges, "temporal_route", "temporal_route", float(rng.uniform(0.08, 0.22)), "route recurrent")
    # broad typed token coverage, not exact task-solution wiring
    route_sources = {
        "danger_route": pilot.ACTORS + pilot.ACTIONS,
        "friendship_route": pilot.ACTORS + pilot.RELATIONS + ["bite"],
        "sound_route": pilot.ACTORS + pilot.SOUNDS,
        "environment_route": pilot.PLACES + pilot.NOISES + pilot.SOUNDS,
    }
    for route, tokens in route_sources.items():
        for token in tokens:
            add_edge(edges, f"tok_{token}", route, float(rng.normal(0.0, 0.22)), "token->route")
    for route in pilot.route_nodes():
        for token in pilot.TOKENS:
            if rng.random() < 0.16:
                add_edge(edges, f"tok_{token}", route, float(rng.normal(0.0, 0.10)), "token->route")
    hub_tokens = {
        "shared_actor_hub": pilot.ACTORS,
        "shared_action_hub": pilot.ACTIONS + pilot.RELATIONS + pilot.SOUNDS,
        "shared_context_hub": pilot.PLACES + pilot.NOISES + pilot.SOUNDS,
    }
    for hub, tokens in hub_tokens.items():
        bias[hub] = 0.0
        for token in tokens:
            add_edge(edges, f"tok_{token}", hub, float(rng.normal(0.12, 0.08)), "token->hub")
    for route in pilot.route_nodes():
        for hub in hub_tokens:
            add_edge(edges, hub, route, float(rng.normal(0.0, 0.20)), "hub->route")
    for target_frame in pilot.FRAMES:
        target = pilot.FRAME_BY_ROUTE[target_frame]
        for source_frame in pilot.FRAMES:
            source = f"suppress_{source_frame}"
            if target_frame == source_frame:
                add_edge(edges, source, target, float(rng.uniform(-1.20, -0.55)), "suppressor->route")
            else:
                add_edge(edges, source, target, float(rng.uniform(0.05, 0.28)), "suppressor->route")
    for node in [item for item in pilot.temporal_nodes() if item != "temporal_route"]:
        add_edge(edges, node, node, float(rng.uniform(0.04, 0.20)), "route recurrent")
        add_edge(edges, node, "temporal_route", float(rng.normal(0.0, 0.30)), "temporal role")
    add_readout_edges(edges, scale=float(rng.uniform(0.65, 1.10)))
    return GraphSpec(
        name=name,
        nodes=nodes,
        types=types,
        edges=dedupe_edges(edges),
        bias=bias,
        frame_gate={frame: float(rng.uniform(0.65, 1.20)) for frame in pilot.FRAMES},
        suppressor_strength=float(rng.uniform(0.70, 1.35)),
        decay=float(rng.uniform(0.25, 0.55)),
    )


def build_random_spec(seed: int, edge_budget: int) -> GraphSpec:
    rng = np.random.default_rng(seed)
    nodes, types = base_nodes()
    forbidden = {
        (source, target)
        for source in nodes
        for target in nodes
        if source == target or (types.get(source) == "token_input" and types.get(target) == "readout")
    }
    candidates = [(source, target) for source in nodes for target in nodes if (source, target) not in forbidden]
    rng.shuffle(candidates)
    edges: list[EdgeSpec] = []
    for source, target in candidates[: max(0, edge_budget - 10)]:
        add_edge(edges, source, target, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.75)), classify_edge(types, source, target))
    add_readout_edges(edges, scale=float(rng.uniform(0.4, 1.0)))
    return GraphSpec(
        name="random_graph_baseline",
        nodes=nodes,
        types=types,
        edges=dedupe_edges(edges),
        bias={node: float(rng.normal(0.0, 0.10)) for node in nodes if types[node] in {"frame_route", "shared_hub", "readout"}},
        frame_gate={frame: float(rng.uniform(0.0, 1.2)) for frame in pilot.FRAMES},
        suppressor_strength=float(rng.uniform(0.0, 1.2)),
        decay=float(rng.uniform(0.10, 0.60)),
    )


def edge_budget_flags(edge_count: int, hand_edges: int, damaged_edges: int) -> dict[str, bool]:
    return {
        "leq_hand_seeded_edge_count": edge_count <= hand_edges,
        "leq_damaged_hand_edge_count": edge_count <= damaged_edges,
        "still_bloated": edge_count > hand_edges,
    }


class DifferentiableAuthorityGraph(nn.Module):
    def __init__(self, spec: GraphSpec, *, trainable: bool = True, mask: torch.Tensor | None = None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.nodes = list(spec.nodes)
        self.types = dict(spec.types)
        self.idx = {name: i for i, name in enumerate(self.nodes)}
        self.edge_sources = torch.tensor([self.idx[edge.source] for edge in spec.edges], dtype=torch.long)
        self.edge_targets = torch.tensor([self.idx[edge.target] for edge in spec.edges], dtype=torch.long)
        self.edge_types = [edge.edge_type for edge in spec.edges]
        init_weights = torch.tensor([edge.weight for edge in spec.edges], dtype=torch.float32)
        self.edge_weights = nn.Parameter(init_weights, requires_grad=trainable)
        init_bias = torch.zeros(len(self.nodes), dtype=torch.float32)
        for node, value in spec.bias.items():
            if node in self.idx:
                init_bias[self.idx[node]] = float(value)
        self.bias = nn.Parameter(init_bias, requires_grad=trainable)
        self.frame_gate = nn.Parameter(
            torch.tensor([spec.frame_gate[frame] for frame in pilot.FRAMES], dtype=torch.float32),
            requires_grad=trainable,
        )
        self.suppressor_strength = nn.Parameter(torch.tensor(float(spec.suppressor_strength)), requires_grad=trainable)
        self.decay_logit = nn.Parameter(torch.tensor(logit_clamped(spec.decay)), requires_grad=trainable)
        if mask is None:
            mask = torch.ones(len(spec.edges), dtype=torch.float32)
        self.register_buffer("edge_mask", mask.float())
        self.register_buffer("last_source_activation", torch.zeros(len(spec.edges), dtype=torch.float32))

    def clone_frozen(self) -> "DifferentiableAuthorityGraph":
        clone = DifferentiableAuthorityGraph(self.spec, trainable=False, mask=self.edge_mask.detach().cpu().clone())
        clone.edge_weights.data.copy_(self.edge_weights.detach().cpu())
        clone.bias.data.copy_(self.bias.detach().cpu())
        clone.frame_gate.data.copy_(self.frame_gate.detach().cpu())
        clone.suppressor_strength.data.copy_(self.suppressor_strength.detach().cpu())
        clone.decay_logit.data.copy_(self.decay_logit.detach().cpu())
        return clone

    def active_edge_count(self) -> int:
        return int((self.edge_mask > 0.0).sum().item())

    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)

    def edge_values(self) -> torch.Tensor:
        return self.edge_weights * self.edge_mask

    def update(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        raw = self.decay() * state + injection + self.bias
        messages = state[:, self.edge_sources] * self.edge_values().view(1, -1)
        raw = raw.index_add(1, self.edge_targets.to(raw.device), messages)
        return torch.tanh(torch.clamp(raw, -5.0, 5.0))

    def static_injection(
        self,
        examples: list[pilot.StaticExample],
        *,
        forced_wrong_frame: bool = False,
        no_frame_gate: bool = False,
        no_suppressor: bool = False,
    ) -> torch.Tensor:
        inj = torch.zeros((len(examples), len(self.nodes)), dtype=torch.float32)
        for row, ex in enumerate(examples):
            tokens = [ex.obs.actor, ex.obs.action, ex.obs.relation, ex.obs.sound, ex.obs.place, ex.obs.noise]
            for token in tokens:
                node = f"tok_{token}"
                if node in self.idx:
                    inj[row, self.idx[node]] += 1.0
            frame = ex.frame
            if forced_wrong_frame:
                frame = pilot.FRAMES[(pilot.FRAMES.index(frame) + 1) % len(pilot.FRAMES)]
            if not no_frame_gate:
                inj[row, self.idx[pilot.FRAME_BY_ROUTE[frame]]] += self.frame_gate[pilot.FRAMES.index(frame)]
            if not no_suppressor:
                for inactive in pilot.FRAMES:
                    if inactive != frame:
                        inj[row, self.idx[f"suppress_{inactive}"]] += torch.relu(self.suppressor_strength)
        return inj

    def forward_static(
        self,
        examples: list[pilot.StaticExample],
        *,
        steps: int,
        forced_wrong_frame: bool = False,
        no_frame_gate: bool = False,
        no_suppressor: bool = False,
        no_recurrence: bool = False,
        record_usage: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = torch.zeros((len(examples), len(self.nodes)), dtype=torch.float32)
        run_steps = 1 if no_recurrence else steps
        for _ in range(run_steps):
            inj = self.static_injection(
                examples,
                forced_wrong_frame=forced_wrong_frame,
                no_frame_gate=no_frame_gate,
                no_suppressor=no_suppressor,
            )
            state = self.update(state, inj)
        if record_usage and len(examples):
            self.last_source_activation = state[:, self.edge_sources].detach().abs().mean(dim=0).cpu()
        return {
            "state": state,
            "logit": state[:, self.idx["readout_positive"]] - state[:, self.idx["readout_negative"]],
            "routes": state[:, [self.idx[pilot.FRAME_BY_ROUTE[frame]] for frame in pilot.FRAMES]],
        }

    def forward_temporal(
        self,
        examples: list[pilot.TemporalExample],
        *,
        steps: int,
        no_recurrence: bool = False,
        shuffled_order: bool = False,
        record_usage: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = torch.zeros((len(examples), len(self.nodes)), dtype=torch.float32)
        for pos in range(3):
            if no_recurrence:
                state = torch.zeros_like(state)
            inj = torch.zeros_like(state)
            for row, ex in enumerate(examples):
                sequence = tuple(reversed(ex.sequence)) if shuffled_order else ex.sequence
                token = sequence[pos]
                tok_node = f"tok_{token}"
                if tok_node in self.idx:
                    inj[row, self.idx[tok_node]] += 1.0
                if not no_recurrence:
                    if pos == 0 and f"subject_{token}" in self.idx:
                        inj[row, self.idx[f"subject_{token}"]] += 1.0
                    elif pos == 1 and f"verb_{token}" in self.idx:
                        inj[row, self.idx[f"verb_{token}"]] += 1.0
                    elif pos == 2 and f"object_{token}" in self.idx:
                        inj[row, self.idx[f"object_{token}"]] += 1.0
            state = self.update(state, inj)
        for _ in range(max(0, steps - 3)):
            state = self.update(state, torch.zeros_like(state))
        if record_usage and len(examples):
            self.last_source_activation = state[:, self.edge_sources].detach().abs().mean(dim=0).cpu()
        return {
            "state": state,
            "logit": state[:, self.idx["readout_positive"]] - state[:, self.idx["readout_negative"]],
            "routes": state[:, [self.idx[pilot.FRAME_BY_ROUTE[frame]] for frame in pilot.FRAMES]],
        }


def logit_clamped(value: float) -> float:
    value = float(np.clip(value, 1.0e-4, 1.0 - 1.0e-4))
    return float(np.log(value / (1.0 - value)))


def labels_static(examples: list[pilot.StaticExample]) -> torch.Tensor:
    return torch.tensor([float(ex.label) for ex in examples], dtype=torch.float32)


def labels_temporal(examples: list[pilot.TemporalExample]) -> torch.Tensor:
    return torch.tensor([float(ex.label) for ex in examples], dtype=torch.float32)


def split_datasets(samples: int, seed: int) -> dict[str, list[Any]]:
    return pilot.make_datasets(samples, seed)


def task_loss(model: DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], steps: int) -> dict[str, torch.Tensor]:
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    latent_out = model.forward_static(latent, steps=steps, record_usage=True)
    multi_out = model.forward_static(multi, steps=steps, record_usage=True)
    temporal_out = model.forward_temporal(temporal, steps=steps, record_usage=True)
    bce = (
        F.binary_cross_entropy_with_logits(latent_out["logit"], labels_static(latent))
        + F.binary_cross_entropy_with_logits(multi_out["logit"], labels_static(multi))
        + F.binary_cross_entropy_with_logits(temporal_out["logit"], labels_temporal(temporal))
    ) / 3.0
    static_examples = latent + multi
    static_routes = torch.cat([latent_out["routes"], multi_out["routes"]], dim=0)
    labels = torch.cat([labels_static(latent), labels_static(multi)])
    frame_indices = torch.tensor([pilot.FRAMES.index(ex.frame) for ex in static_examples], dtype=torch.long)
    active = static_routes.gather(1, frame_indices.view(-1, 1)).squeeze(1)
    inactive_mask = torch.ones_like(static_routes, dtype=torch.bool)
    inactive_mask[torch.arange(len(static_examples)), frame_indices] = False
    inactive = static_routes[inactive_mask].view(len(static_examples), -1)
    label_sign = labels * 2.0 - 1.0
    signed_active = active * label_sign
    signed_inactive = inactive * label_sign.view(-1, 1)
    inactive_penalty = F.relu(signed_inactive.max(dim=1).values - 0.10).mean()
    wrong_leakage = F.relu(inactive.abs().max(dim=1).values - active.abs() + 0.15).mean()
    route_specialization = F.softplus(0.35 - (signed_active - signed_inactive.max(dim=1).values)).mean()
    active_margin_guard = F.relu(0.25 - active.abs()).mean()
    logits = torch.cat([latent_out["logit"], multi_out["logit"], temporal_out["logit"]])
    stability = F.relu(logits.abs() - 8.0).pow(2).mean()
    edge_l1 = model.edge_values().abs().mean()
    total = (
        bce
        + 0.50 * inactive_penalty
        + 0.50 * wrong_leakage
        + 0.40 * route_specialization
        + 0.30 * F.binary_cross_entropy_with_logits(temporal_out["logit"], labels_temporal(temporal))
        + 0.10 * edge_l1
        + 0.10 * stability
        + 0.15 * active_margin_guard
    )
    return {
        "total": total,
        "task_bce": bce.detach(),
        "inactive_penalty": inactive_penalty.detach(),
        "wrong_leakage": wrong_leakage.detach(),
        "route_specialization_loss": route_specialization.detach(),
        "active_margin_guard": active_margin_guard.detach(),
        "edge_l1": edge_l1.detach(),
        "stability": stability.detach(),
    }


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(((logits > 0.0) == (labels > 0.5)).float().mean().item()) if len(labels) else 0.0


@torch.no_grad()
def evaluate_model(
    model: DifferentiableAuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    steps: int,
    controls: bool = False,
) -> dict[str, Any]:
    latent = datasets["latent_refraction_small"]
    multi = datasets["multi_aspect_small"]
    temporal = datasets["temporal_order_contrast_small"]
    latent_out = model.forward_static(latent, steps=steps, record_usage=True)
    multi_out = model.forward_static(multi, steps=steps, record_usage=True)
    temporal_out = model.forward_temporal(temporal, steps=steps, record_usage=True)
    latent_acc = binary_accuracy(latent_out["logit"], labels_static(latent))
    multi_acc = binary_accuracy(multi_out["logit"], labels_static(multi))
    temporal_acc = binary_accuracy(temporal_out["logit"], labels_temporal(temporal))
    authority = authority_metrics(model, latent + multi, steps=steps)
    wrong_out = model.forward_static(latent, steps=steps, forced_wrong_frame=True)
    wrong_acc = binary_accuracy(wrong_out["logit"], labels_static(latent))
    no_rec_temporal = model.forward_temporal(temporal, steps=steps, no_recurrence=True)
    no_rec_acc = binary_accuracy(no_rec_temporal["logit"], labels_temporal(temporal))
    route_spec = route_specialization_metric(model, latent + multi, steps=steps)
    logit_margin = float(torch.cat([latent_out["logit"], multi_out["logit"], temporal_out["logit"]]).abs().mean().item())
    raw_loss = task_loss(model, datasets, steps)["total"].item()
    out: dict[str, Any] = {
        "overall_accuracy": float(np.mean([latent_acc, multi_acc, temporal_acc])),
        "latent_refraction_accuracy": latent_acc,
        "multi_aspect_accuracy": multi_acc,
        "temporal_order_accuracy": temporal_acc,
        "authority_refraction_score": authority["authority_refraction_score"],
        "active_group_influence": authority["active_group_influence"],
        "inactive_group_influence": authority["inactive_group_influence"],
        "active_minus_inactive_margin": authority["active_group_influence"] - authority["inactive_group_influence"],
        "output_logit_margin": logit_margin,
        "wrong_frame_accuracy": wrong_acc,
        "wrong_frame_drop": latent_acc - wrong_acc,
        "recurrence_drop": temporal_acc - no_rec_acc,
        "route_specialization": route_spec,
        "edge_count": model.active_edge_count(),
        "node_count": len(model.nodes),
        "final_loss": float(raw_loss),
    }
    if controls:
        no_gate = model.forward_static(latent, steps=steps, no_frame_gate=True)
        no_supp = model.forward_static(latent, steps=steps, no_suppressor=True)
        out["controls"] = {
            "no_frame_gate_accuracy": binary_accuracy(no_gate["logit"], labels_static(latent)),
            "no_suppressor_accuracy": binary_accuracy(no_supp["logit"], labels_static(latent)),
            "no_recurrence_temporal_accuracy": no_rec_acc,
            "wrong_frame_accuracy": wrong_acc,
        }
    return out


@torch.no_grad()
def positive_probabilities(model: DifferentiableAuthorityGraph, examples: list[pilot.StaticExample], steps: int) -> np.ndarray:
    logits = model.forward_static(examples, steps=steps)["logit"]
    return torch.sigmoid(logits).cpu().numpy()


def swap_observation(obs: pilot.Observation, donor: pilot.Observation, group: str) -> pilot.Observation:
    data = obs.__dict__.copy()
    for field in pilot.GROUP_FIELDS[group]:
        data[field] = getattr(donor, field)
    return pilot.Observation(**data)


@torch.no_grad()
def authority_metrics(model: DifferentiableAuthorityGraph, examples: list[pilot.StaticExample], *, steps: int) -> dict[str, float]:
    rng = np.random.default_rng(1234)
    by_frame: dict[str, dict[str, float]] = {}
    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        base = positive_probabilities(model, frame_examples, steps)
        group_delta: dict[str, float] = {}
        active_group = pilot.ACTIVE_GROUP_BY_FRAME[frame]
        for group in pilot.GROUP_FIELDS:
            swapped: list[pilot.StaticExample] = []
            for ex in frame_examples:
                donor = frame_examples[int(rng.integers(0, len(frame_examples)))].obs
                swapped.append(
                    pilot.StaticExample(
                        obs=swap_observation(ex.obs, donor, group),
                        frame=frame,
                        label=ex.label,
                    )
                )
            prob = positive_probabilities(model, swapped, steps)
            group_delta[group] = float(np.mean(np.abs(base - prob)))
        inactive = max(value for key, value in group_delta.items() if key != active_group)
        by_frame[frame] = {
            "active": group_delta[active_group],
            "inactive": inactive,
            "refraction": group_delta[active_group] - inactive,
        }
    return {
        "authority_refraction_score": float(np.mean([item["refraction"] for item in by_frame.values()])),
        "active_group_influence": float(np.mean([item["active"] for item in by_frame.values()])),
        "inactive_group_influence": float(np.mean([item["inactive"] for item in by_frame.values()])),
    }


@torch.no_grad()
def route_specialization_metric(model: DifferentiableAuthorityGraph, examples: list[pilot.StaticExample], *, steps: int) -> float:
    values = []
    for ex in examples:
        label_sign = 1.0 if ex.label else -1.0
        correct = float(model.forward_static([ex], steps=steps)["logit"].item()) * label_sign
        wrong = []
        for other in pilot.FRAMES:
            if other == ex.frame:
                continue
            forced = pilot.StaticExample(obs=ex.obs, frame=other, label=ex.label)
            wrong.append(float(model.forward_static([forced], steps=steps)["logit"].item()) * label_sign)
        values.append(float(np.tanh(correct - max(wrong))))
    return float(np.mean(values)) if values else 0.0


def train_guided(
    model: DifferentiableAuthorityGraph,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    epochs: int,
    deadline: float,
    arm: str,
) -> tuple[DifferentiableAuthorityGraph, dict[str, Any]]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    best = model.clone_frozen()
    best_val = evaluate_model(best, validation, steps=args.steps)
    best_score = selection_score(best_val)
    history = []
    interrupted = False
    for epoch in range(1, epochs + 1):
        if time.time() >= deadline:
            interrupted = True
            break
        opt.zero_grad(set_to_none=True)
        loss = task_loss(model, train, args.steps)["total"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        with torch.no_grad():
            model.edge_weights.clamp_(-2.5, 2.5)
            model.bias.clamp_(-2.5, 2.5)
            model.frame_gate.clamp_(0.0, 2.0)
            model.suppressor_strength.clamp_(0.0, 2.0)
            model.decay_logit.clamp_(logit_clamped(0.0 + 1.0e-4), logit_clamped(0.95))
        if epoch % args.checkpoint_every == 0 or epoch == epochs:
            val = evaluate_model(model, validation, steps=args.steps)
            score = selection_score(val)
            history.append({"epoch": epoch, "validation": val, "selection_score": score})
            print(
                "[guided-pruning] "
                f"{arm} seed={seed} epoch={epoch}/{epochs} "
                f"val_score={score:.4f} "
                f"val_acc={val['overall_accuracy']:.4f} "
                f"val_authority={val['authority_refraction_score']:.4f}",
                flush=True,
            )
            if score > best_score:
                best = model.clone_frozen()
                best_val = val
                best_score = score
    return best, {
        "best_validation": best_val,
        "best_selection_score": best_score,
        "history": history,
        "interrupted": interrupted,
    }


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["overall_accuracy"]
        + 0.65 * metrics["authority_refraction_score"]
        + 0.25 * metrics["wrong_frame_drop"]
        + 0.25 * metrics["recurrence_drop"]
        + 0.20 * metrics["route_specialization"]
        - 0.10 * metrics["inactive_group_influence"]
        - 0.02 * max(0.0, metrics["edge_count"] - 91) / 91.0
    )


def saliency(model: DifferentiableAuthorityGraph) -> torch.Tensor:
    return model.edge_weights.detach().abs().cpu() * model.edge_mask.detach().cpu() * (model.last_source_activation.detach().cpu() + 0.05)


def prune_model(model: DifferentiableAuthorityGraph, fraction: float) -> tuple[DifferentiableAuthorityGraph, dict[str, Any]]:
    score = saliency(model)
    active = torch.where(model.edge_mask.detach().cpu() > 0.0)[0]
    protected = torch.tensor([edge_type == "route->readout" for edge_type in model.edge_types], dtype=torch.bool)
    candidates = active[~protected[active]]
    drop_count = int(round(len(candidates) * fraction))
    if drop_count <= 0:
        return model.clone_frozen(), {"pruned_edge_type_counts": {}, "pruned_edges": 0}
    ordered = candidates[torch.argsort(score[candidates])]
    drop = ordered[:drop_count]
    pruned = model.clone_frozen()
    new_mask = pruned.edge_mask.detach().cpu().clone()
    new_mask[drop] = 0.0
    pruned.edge_mask.copy_(new_mask)
    counts = Counter(model.edge_types[int(i)] for i in drop)
    return pruned, {
        "pruned_edge_type_counts": dict(counts),
        "pruned_edges": int(drop_count),
    }


def mutation_repair(
    model: DifferentiableAuthorityGraph,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    deadline: float,
) -> tuple[DifferentiableAuthorityGraph, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    best = model.clone_frozen()
    best_val = evaluate_model(best, validation, steps=args.steps)
    best_score = selection_score(best_val)
    accepted = 0
    for step in range(args.mutation_steps):
        if time.time() >= deadline:
            break
        candidate = best.clone_frozen()
        active = torch.where(candidate.edge_mask.detach().cpu() > 0.0)[0].numpy()
        if len(active) == 0:
            break
        chosen = rng.choice(active, size=min(6, len(active)), replace=False)
        with torch.no_grad():
            noise = torch.tensor(rng.normal(0.0, 0.12, size=len(chosen)), dtype=torch.float32)
            candidate.edge_weights[torch.tensor(chosen, dtype=torch.long)] += noise
        train_score = selection_score(evaluate_model(candidate, train, steps=args.steps))
        if train_score >= selection_score(evaluate_model(best, train, steps=args.steps)) - 0.02:
            val = evaluate_model(candidate, validation, steps=args.steps)
            score = selection_score(val)
            if score > best_score:
                best = candidate
                best_val = val
                best_score = score
                accepted += 1
    return best, {"accepted_mutations": accepted, "best_validation": best_val, "best_selection_score": best_score}


def evaluate_arm_record(
    arm: str,
    model: DifferentiableAuthorityGraph,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    final_test: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_metrics = evaluate_model(model, train, steps=args.steps)
    validation_metrics = evaluate_model(model, validation, steps=args.steps)
    final_metrics = evaluate_model(model, final_test, steps=args.steps, controls=True)
    metadata = metadata or {}
    return {
        "arm": arm,
        "seed": seed,
        "train": train_metrics,
        "validation": validation_metrics,
        "final_test": final_metrics,
        "selection_score_validation": selection_score(validation_metrics),
        "success": is_success(final_metrics),
        "strong_success": is_strong_success(final_metrics),
        "metadata": metadata,
    }


def is_success(metrics: dict[str, Any]) -> bool:
    return (
        metrics["overall_accuracy"] >= SUCCESS["overall_accuracy"]
        and metrics["temporal_order_accuracy"] >= SUCCESS["temporal_order_accuracy"]
        and metrics["authority_refraction_score"] >= SUCCESS["authority_refraction_score"]
        and metrics["wrong_frame_drop"] >= SUCCESS["wrong_frame_drop"]
    )


def is_strong_success(metrics: dict[str, Any]) -> bool:
    return (
        is_success(metrics)
        and metrics["overall_accuracy"] >= 0.95
        and metrics["latent_refraction_accuracy"] >= 0.90
        and metrics["multi_aspect_accuracy"] >= 0.90
        and metrics["edge_count"] <= 91
    )


def grouped_ablation_saliency(model: DifferentiableAuthorityGraph, validation: dict[str, list[Any]], steps: int) -> dict[str, Any]:
    baseline = evaluate_model(model, validation, steps=steps)
    out: dict[str, Any] = {}
    for edge_type in sorted(set(model.edge_types)):
        idx = torch.tensor([i for i, item in enumerate(model.edge_types) if item == edge_type and model.edge_mask[i] > 0.0], dtype=torch.long)
        if len(idx) == 0:
            continue
        candidate = model.clone_frozen()
        new_mask = candidate.edge_mask.detach().cpu().clone()
        new_mask[idx] = 0.0
        candidate.edge_mask.copy_(new_mask)
        metrics = evaluate_model(candidate, validation, steps=steps)
        out[edge_type] = {
            "edge_count": int(len(idx)),
            "accuracy_drop": baseline["overall_accuracy"] - metrics["overall_accuracy"],
            "authority_drop": baseline["authority_refraction_score"] - metrics["authority_refraction_score"],
            "temporal_drop": baseline["temporal_order_accuracy"] - metrics["temporal_order_accuracy"],
        }
    return out


def run_seed(args: argparse.Namespace, seed: int, deadline: float) -> list[dict[str, Any]]:
    train = split_datasets(args.train_samples, seed + 10_000)
    validation = split_datasets(args.validation_samples, seed + 20_000)
    final_test = split_datasets(args.final_test_samples, seed + 30_000)
    hand_spec = build_hand_spec()
    damaged_spec = damage_spec(hand_spec, 0.50, seed + 40_000)
    over_spec = build_overcomplete_spec(seed + 50_000)
    records: list[dict[str, Any]] = []

    def maybe_add(arm: str, model: DifferentiableAuthorityGraph, metadata: dict[str, Any] | None = None) -> None:
        if arm in args.arms:
            print(f"[guided-pruning] evaluating {arm} seed={seed}", flush=True)
            final_edge_count = model.active_edge_count()
            meta = dict(metadata or {})
            meta["edge_budget_flags"] = edge_budget_flags(final_edge_count, len(hand_spec.edges), len(damaged_spec.edges))
            records.append(evaluate_arm_record(arm, model, train, validation, final_test, args=args, seed=seed, metadata=meta))

    maybe_add("hand_seeded", DifferentiableAuthorityGraph(hand_spec, trainable=False))
    maybe_add("damaged_hand_seeded_50", DifferentiableAuthorityGraph(damaged_spec, trainable=False))
    if "random_graph_baseline" in args.arms:
        random_spec = build_random_spec(seed + 60_000, edge_budget=len(over_spec.edges))
        maybe_add("random_graph_baseline", DifferentiableAuthorityGraph(random_spec, trainable=False))
    maybe_add("grammar_v2_untrained", DifferentiableAuthorityGraph(over_spec, trainable=False))

    trained_model: DifferentiableAuthorityGraph | None = None
    trained_info: dict[str, Any] | None = None
    if any(arm.startswith("overcomplete_guided") for arm in args.arms):
        base_model = DifferentiableAuthorityGraph(over_spec, trainable=True)
        trained_model, trained_info = train_guided(
            base_model,
            train,
            validation,
            args=args,
            seed=seed + 70_000,
            epochs=args.epochs,
            deadline=deadline,
            arm="overcomplete_guided_train",
        )
        maybe_add("overcomplete_guided_train", trained_model, {"training": trained_info})

    if "grammar_v2_mutation_only" in args.arms:
        mut_model, mut_info = mutation_repair(
            DifferentiableAuthorityGraph(over_spec, trainable=False),
            train,
            validation,
            args=args,
            seed=seed + 75_000,
            deadline=deadline,
        )
        maybe_add("grammar_v2_mutation_only", mut_model, {"repair": mut_info})

    if trained_model is not None:
        best_prune50_record: tuple[DifferentiableAuthorityGraph, dict[str, Any]] | None = None
        for fraction, arm in [
            (0.30, "overcomplete_guided_prune_30"),
            (0.50, "overcomplete_guided_prune_50"),
            (0.70, "overcomplete_guided_prune_70"),
        ]:
            if arm not in args.arms and not (fraction == 0.50 and "overcomplete_guided_prune_50_repair" in args.arms):
                continue
            trained_model.forward_static(train["latent_refraction_small"] + train["multi_aspect_small"], steps=args.steps, record_usage=True)
            pruned, prune_info = prune_model(trained_model, fraction)
            fine_tuned, fine_info = train_guided(
                DifferentiableAuthorityGraph(model_to_spec(pruned), trainable=True),
                train,
                validation,
                args=args,
                seed=seed + int(80_000 + fraction * 100),
                epochs=args.fine_tune_epochs,
                deadline=deadline,
                arm=arm,
            )
            metadata = {"prune_percent": fraction, "pruning": prune_info, "fine_tune": fine_info}
            if fraction == 0.50:
                metadata["grouped_ablation_saliency_validation"] = grouped_ablation_saliency(
                    fine_tuned, validation, args.steps
                )
            if arm in args.arms:
                maybe_add(arm, fine_tuned, metadata)
            if fraction == 0.50:
                best_prune50_record = (fine_tuned, metadata)
        if "overcomplete_guided_prune_50_repair" in args.arms and best_prune50_record is not None:
            repaired, repair_info = mutation_repair(
                best_prune50_record[0],
                train,
                validation,
                args=args,
                seed=seed + 90_000,
                deadline=deadline,
            )
            meta = dict(best_prune50_record[1])
            meta["repair"] = repair_info
            maybe_add("overcomplete_guided_prune_50_repair", repaired, meta)
    return records


def model_to_spec(model: DifferentiableAuthorityGraph) -> GraphSpec:
    weights = model.edge_weights.detach().cpu().numpy()
    mask = model.edge_mask.detach().cpu().numpy()
    edges = [
        EdgeSpec(edge.source, edge.target, float(weights[i]), edge.edge_type)
        for i, edge in enumerate(model.spec.edges)
        if mask[i] > 0.0
    ]
    bias_values = model.bias.detach().cpu().numpy()
    bias = {node: float(bias_values[model.idx[node]]) for node in model.nodes if abs(float(bias_values[model.idx[node]])) > 1.0e-9}
    frame_gate_values = model.frame_gate.detach().cpu().numpy()
    return GraphSpec(
        name=model.spec.name,
        nodes=model.nodes,
        types=model.types,
        edges=edges,
        bias=bias,
        frame_gate={frame: float(frame_gate_values[i]) for i, frame in enumerate(pilot.FRAMES)},
        suppressor_strength=float(model.suppressor_strength.detach().cpu().item()),
        decay=float(torch.sigmoid(model.decay_logit.detach()).cpu().item()),
    )


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "overall_accuracy",
        "latent_refraction_accuracy",
        "multi_aspect_accuracy",
        "temporal_order_accuracy",
        "authority_refraction_score",
        "active_group_influence",
        "inactive_group_influence",
        "active_minus_inactive_margin",
        "output_logit_margin",
        "wrong_frame_drop",
        "recurrence_drop",
        "route_specialization",
        "edge_count",
        "node_count",
        "final_loss",
    ]
    out: dict[str, Any] = {}
    for arm in sorted({record["arm"] for record in records}):
        rows = [record for record in records if record["arm"] == arm]
        out[arm] = {
            "runs": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "strong_success_rate": float(np.mean([row["strong_success"] for row in rows])),
            "train": {field: numeric_summary([row["train"][field] for row in rows]) for field in fields},
            "validation": {field: numeric_summary([row["validation"][field] for row in rows]) for field in fields},
            "final_test": {field: numeric_summary([row["final_test"][field] for row in rows]) for field in fields},
            "selection_score_validation": numeric_summary([row["selection_score_validation"] for row in rows]),
            "metadata": summarize_metadata(rows),
        }
    return out


def summarize_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pruned_counts: Counter[str] = Counter()
    for row in rows:
        pruning = row.get("metadata", {}).get("pruning", {})
        pruned_counts.update(pruning.get("pruned_edge_type_counts", {}))
    return {
        "pruned_edge_type_counts": dict(pruned_counts),
        "edge_budget_flags": {
            "all_leq_hand": all(row["metadata"].get("edge_budget_flags", {}).get("leq_hand_seeded_edge_count", False) for row in rows),
            "all_leq_damaged": all(row["metadata"].get("edge_budget_flags", {}).get("leq_damaged_hand_edge_count", False) for row in rows),
            "any_still_bloated": any(row["metadata"].get("edge_budget_flags", {}).get("still_bloated", False) for row in rows),
        },
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


def mean(aggregate: dict[str, Any], arm: str, field: str) -> float:
    value = aggregate.get(arm, {}).get("final_test", {}).get(field, {}).get("mean")
    return float(value) if value is not None else 0.0


def verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    guided_auth = mean(aggregate, "overcomplete_guided_train", "authority_refraction_score")
    untrained_auth = mean(aggregate, "grammar_v2_untrained", "authority_refraction_score")
    mutation_auth = mean(aggregate, "grammar_v2_mutation_only", "authority_refraction_score")
    prune50_auth = mean(aggregate, "overcomplete_guided_prune_50", "authority_refraction_score")
    prune50_repair_auth = mean(aggregate, "overcomplete_guided_prune_50_repair", "authority_refraction_score")
    hand_auth = mean(aggregate, "hand_seeded", "authority_refraction_score")
    return {
        "guided_training_helps": guided_auth > max(untrained_auth, mutation_auth) + 0.05,
        "pruning_preserves_authority": prune50_auth >= 0.85 * guided_auth if guided_auth else False,
        "sparse_graph_emerges": (
            aggregate.get("overcomplete_guided_prune_50", {}).get("success_rate", 0.0) > 0.0
            and mean(aggregate, "overcomplete_guided_prune_50", "edge_count") <= mean(aggregate, "hand_seeded", "edge_count")
        ),
        "repair_after_pruning_helps": prune50_repair_auth > prune50_auth + 0.03,
        "overcomplete_scaffold_viable": guided_auth > 0.15 and mean(aggregate, "overcomplete_guided_train", "overall_accuracy") > 0.75,
        "hand_seeded_still_dominates": hand_auth > max(guided_auth, prune50_auth, prune50_repair_auth) + 0.05,
        "mutation_only_still_insufficient": (
            aggregate.get("grammar_v2_mutation_only", {}).get("success_rate", 0.0) == 0.0
            and mutation_auth < SUCCESS["authority_refraction_score"]
        ),
        "final_verdict_uses_final_test": True,
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def write_report(summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    arms = summary["config"]["arms_completed"]
    lines = [
        "# Authority Graph Guided Pruning Pilot",
        "",
        "## Goal",
        "",
        "Test whether an overcomplete explicit authority-graph scaffold can be trained through scalar gains and then pruned into a sparse authority-flow graph.",
        "",
        "All arms use the same explicit readout-node scoring. The old route-state shortcut is not used for comparisons in this report.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Final-Test Results",
        "",
        "| Arm | Success | Strong | Accuracy | Latent | Multi | Temporal | Authority | Active | Inactive | Margin | Wrong Frame | Recurrence | Route Spec | Edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        if arm not in aggregate:
            continue
        item = aggregate[arm]
        final = item["final_test"]
        lines.append(
            f"| `{arm}` | `{fmt(item['success_rate'])}` | `{fmt(item['strong_success_rate'])}` "
            f"| `{fmt(final['overall_accuracy']['mean'])}` "
            f"| `{fmt(final['latent_refraction_accuracy']['mean'])}` "
            f"| `{fmt(final['multi_aspect_accuracy']['mean'])}` "
            f"| `{fmt(final['temporal_order_accuracy']['mean'])}` "
            f"| `{fmt(final['authority_refraction_score']['mean'])}` "
            f"| `{fmt(final['active_group_influence']['mean'])}` "
            f"| `{fmt(final['inactive_group_influence']['mean'])}` "
            f"| `{fmt(final['active_minus_inactive_margin']['mean'])}` "
            f"| `{fmt(final['wrong_frame_drop']['mean'])}` "
            f"| `{fmt(final['recurrence_drop']['mean'])}` "
            f"| `{fmt(final['route_specialization']['mean'])}` "
            f"| `{fmt(final['edge_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Train / Validation / Final-Test Gap",
        "",
        "| Arm | Train Acc | Val Acc | Final Acc | Train Loss | Val Loss | Final Loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in arms:
        if arm not in aggregate:
            continue
        item = aggregate[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item['train']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['validation']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['final_test']['overall_accuracy']['mean'])}` "
            f"| `{fmt(item['train']['final_loss']['mean'])}` "
            f"| `{fmt(item['validation']['final_loss']['mean'])}` "
            f"| `{fmt(item['final_test']['final_loss']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Pruning Audit",
        "",
        "| Arm | Pruned edge type counts | <= hand edges | <= damaged edges | Still bloated |",
        "|---|---|---:|---:|---:|",
    ])
    for arm in arms:
        if arm not in aggregate:
            continue
        meta = aggregate[arm]["metadata"]
        flags = meta["edge_budget_flags"]
        lines.append(
            f"| `{arm}` | `{json.dumps(meta['pruned_edge_type_counts'], sort_keys=True)}` "
            f"| `{flags['all_leq_hand']}` | `{flags['all_leq_damaged']}` | `{flags['any_still_bloated']}` |"
        )
    lines.extend([
        "",
        "## Best Prune-50 Grouped Ablation",
        "",
        "```json",
        json.dumps(summary.get("best_prune50_grouped_ablation", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation Notes",
        "",
        "- Positive evidence requires authority/refraction and route specialization, not only raw accuracy.",
        "- If pruning preserves accuracy but hurts authority, this report treats that as a mechanism failure.",
        "- Suppressor pruning and inactive leakage are reported explicitly in the pruning audit and influence metrics.",
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


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8")
    write_report(round_floats(summary))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + args.max_runtime_hours * 3600.0
    all_records: list[dict[str, Any]] = []
    interrupted = False
    for seed in range(args.seeds):
        if time.time() >= deadline:
            interrupted = True
            break
        print(f"[guided-pruning] seed={seed}", flush=True)
        all_records.extend(run_seed(args, seed, deadline))
        aggregate_data = aggregate(all_records)
        partial = {
            "config": config_dict(args, completed=False),
            "records": all_records,
            "aggregate": aggregate_data,
            "verdict": verdict(aggregate_data),
            "runtime_seconds": time.time() - start,
            "interrupted_by_wall_clock": interrupted,
            "best_prune50_grouped_ablation": {},
            "platform": platform_dict(),
        }
        write_summary(args.out_dir, partial)
    aggregate_data = aggregate(all_records)
    best_ablation = compute_best_prune50_ablation(all_records, args)
    summary = {
        "config": config_dict(args, completed=not interrupted),
        "records": all_records,
        "aggregate": aggregate_data,
        "verdict": verdict(aggregate_data),
        "runtime_seconds": time.time() - start,
        "interrupted_by_wall_clock": interrupted,
        "best_prune50_grouped_ablation": best_ablation,
        "platform": platform_dict(),
    }
    write_summary(args.out_dir, summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(args.out_dir / SUMMARY_NAME), "report": str(REPORT_PATH)}, indent=2), flush=True)


def config_dict(args: argparse.Namespace, *, completed: bool) -> dict[str, Any]:
    return {
        "seeds": args.seeds,
        "steps": args.steps,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "final_test_samples": args.final_test_samples,
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "mutation_steps": args.mutation_steps,
        "learning_rate": args.learning_rate,
        "checkpoint_every": args.checkpoint_every,
        "max_runtime_hours": args.max_runtime_hours,
        "torch_threads": args.torch_threads,
        "arms_requested": args.arms,
        "arms_completed": sorted(set(args.arms)),
        "explicit_readout_policy": True,
        "completed": completed,
        "smoke": args.smoke,
    }


def platform_dict() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}


def compute_best_prune50_ablation(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    prune_rows = [row for row in records if row["arm"] == "overcomplete_guided_prune_50"]
    if not prune_rows:
        return {}
    best = max(prune_rows, key=lambda row: row["validation"]["authority_refraction_score"])
    return {
        "selected_seed": best["seed"],
        "validation_grouped_ablation": best.get("metadata", {}).get("grouped_ablation_saliency_validation", {}),
        "final_test_authority": best["final_test"]["authority_refraction_score"],
        "final_test_accuracy": best["final_test"]["overall_accuracy"],
    }


if __name__ == "__main__":
    main()
