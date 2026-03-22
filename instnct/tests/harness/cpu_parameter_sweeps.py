"""Shared helpers for active CPU parameter sweeps on the current graph API."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import random
from typing import Any, Callable

import numpy as np

from model.graph import SelfWiringGraph


@dataclass
class ParameterSweepConfig:
    vocab: int
    neurons: int
    density: float
    threshold: float = SelfWiringGraph.DEFAULT_THETA
    ticks: int = 8
    budget: int = 32000
    stale_limit: int = 0


@dataclass
class SweepOutcome:
    best_acc: float
    best_score: float
    steps: int
    connections: int
    context: dict[str, Any] = field(default_factory=dict)
    accepted_records: list[Any] = field(default_factory=list)


def build_sweep_net(config: ParameterSweepConfig, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(
        config.vocab,
        hidden=config.neurons,
        density=config.density,
        theta_init=config.threshold,
    )
    targets = np.random.permutation(config.vocab)
    return net, targets


def evaluate_parameterized_permutation(
    net: SelfWiringGraph,
    targets,
    *,
    ticks: int = 8,
    charge_rate: float = 1.0,
) -> dict[str, float]:
    """Permutation score using current graph semantics with optional charge-rate override."""
    vocab, hidden = net.V, net.H
    charges = np.zeros((vocab, hidden), dtype=np.float32)
    acts = np.zeros((vocab, hidden), dtype=np.float32)
    retention = net.retention_vec
    thresholds = net.theta
    projected = np.eye(vocab, dtype=np.float32) @ net.input_projection
    use_sparse = len(net.alive) < hidden * hidden * 0.1

    for tick in range(ticks):
        if tick == 0:
            acts += projected
        raw = net._sparse_mul_2d(acts) if use_sparse else acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * np.float32(charge_rate)
        charges *= retention
        acts = np.maximum(charges - thresholds, 0.0)
        charges = np.maximum(charges, 0.0)

    logits = charges @ net.output_projection
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = float((preds == targets[:vocab]).mean())
    target_prob = float(probs[np.arange(vocab), targets[:vocab]].mean())
    return {
        "acc": acc,
        "score": 0.5 * acc + 0.5 * target_prob,
        "target_prob": target_prob,
    }


def mutate_structure(net: SelfWiringGraph, n_changes: int = 1, op_weights: dict[str, float] | None = None) -> None:
    """Balanced structural mutation on the current forced-op API."""
    choices = op_weights or {
        "add": 1.0,
        "remove": 1.0,
        "rewire": 1.0,
        "flip": 1.0,
    }
    ops = list(choices)
    weights = [float(choices[op]) for op in ops]
    for _ in range(max(1, int(n_changes))):
        op = random.choices(ops, weights=weights, k=1)[0]
        net.mutate(forced_op=op)


def quantized_step(value: float, *, step: float, min_value: float, max_value: float) -> float:
    direction = random.choice([-1.0, 1.0])
    moved = value + direction * step
    quantized = round(moved / step) * step
    return float(np.clip(quantized, min_value, max_value))


def run_parameter_search(
    net: SelfWiringGraph,
    targets,
    config: ParameterSweepConfig,
    *,
    context: dict[str, Any] | None = None,
    propose_fn: Callable[[SelfWiringGraph, dict[str, Any]], None] | None = None,
    on_accept: Callable[[SelfWiringGraph, dict[str, Any], int, dict[str, float]], Any] | None = None,
) -> SweepOutcome:
    """Shared accept/reject loop for canon-aligned CPU sweeps."""
    context = {} if context is None else dict(context)
    current = evaluate_parameterized_permutation(
        net,
        targets,
        ticks=config.ticks,
        charge_rate=float(context.get("charge_rate", 1.0)),
    )
    best_acc = current["acc"]
    best_score = current["score"]
    stale = 0
    accepted_records = []

    for step in range(config.budget):
        saved_state = net.save_state()
        saved_context = copy.deepcopy(context)
        if propose_fn is not None:
            propose_fn(net, context)
        trial = evaluate_parameterized_permutation(
            net,
            targets,
            ticks=config.ticks,
            charge_rate=float(context.get("charge_rate", 1.0)),
        )

        if trial["score"] > current["score"]:
            current = trial
            best_acc = max(best_acc, trial["acc"])
            best_score = max(best_score, trial["score"])
            stale = 0
            if on_accept is not None:
                accepted_records.append(on_accept(net, context, step, trial))
        else:
            net.restore_state(saved_state)
            context.clear()
            context.update(saved_context)
            stale += 1
            if config.stale_limit and stale >= config.stale_limit:
                return SweepOutcome(
                    best_acc=best_acc,
                    best_score=best_score,
                    steps=step + 1,
                    connections=net.count_connections(),
                    context=context,
                    accepted_records=accepted_records,
                )

    return SweepOutcome(
        best_acc=best_acc,
        best_score=best_score,
        steps=config.budget,
        connections=net.count_connections(),
        context=context,
        accepted_records=accepted_records,
    )
