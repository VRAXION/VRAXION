from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import random

import numpy as np

from model.graph import SelfWiringGraph


@dataclass(frozen=True)
class PermutationHarnessConfig:
    vocab: int
    neurons: int
    density: float = 0.06
    threshold: float = 0.5
    ticks: int = 8
    budget: int = 16000
    stale_limit: int = 6000


@dataclass
class SearchOutcome:
    best_score: float
    best_acc: float
    best_target_probability: float
    steps: int
    stale: int
    connections: int
    curve: list[dict] = field(default_factory=list)
    policy_state: dict[str, object] = field(default_factory=dict)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def build_net_and_targets(config: PermutationHarnessConfig, seed: int):
    set_seeds(seed)
    net = SelfWiringGraph(
        config.neurons,
        config.vocab,
        density=config.density,
        threshold=config.threshold,
    )
    targets = np.random.permutation(config.vocab)
    return net, targets


def evaluate_permutation(net: SelfWiringGraph, targets, ticks: int = 8) -> dict[str, float]:
    logits = net.forward_batch(ticks=ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    vocab = min(net.V, len(targets))
    acc = (np.argmax(probs, axis=1)[:vocab] == targets[:vocab]).mean()
    target_probability = probs[np.arange(vocab), targets[:vocab]].mean()
    return {
        "acc": float(acc),
        "target_probability": float(target_probability),
        "score": float(0.5 * acc + 0.5 * target_probability),
    }


def _curve_row(attempt, outcome, stale, net, adapter, elapsed):
    row = {
        "attempt": int(attempt),
        "best_acc": float(outcome["acc"]),
        "best_score": float(outcome["score"]),
        "best_target_probability": float(outcome["target_probability"]),
        "connections": int(net.count_connections()),
        "stale": int(stale),
        "elapsed_sec": float(elapsed),
    }
    row.update(adapter.describe_state())
    return row


def run_budgeted_search(
    net: SelfWiringGraph,
    targets,
    config: PermutationHarnessConfig,
    adapter,
    checkpoints=None,
) -> SearchOutcome:
    current = evaluate_permutation(net, targets, ticks=config.ticks)
    best_score = current["score"]
    best_acc = current["acc"]
    best_target_probability = current["target_probability"]
    stale = 0
    curve = []
    checkpoint_idx = 0
    checkpoints = list(checkpoints or [])
    started = perf_counter()
    steps = 0

    for attempt in range(1, config.budget + 1):
        steps = attempt
        net_state = net.save_state()
        proposal = adapter.propose(net)
        trial = evaluate_permutation(net, targets, ticks=config.ticks)
        improved = trial["score"] > current["score"]

        if improved:
            current = trial
            best_score = current["score"]
            best_acc = max(best_acc, current["acc"])
            best_target_probability = max(best_target_probability, current["target_probability"])
            stale = 0
            adapter.on_accept(proposal)
        else:
            net.restore_state(net_state)
            stale += 1
            adapter.on_reject(proposal)

        adapter.after_step(improved, attempt)

        while checkpoint_idx < len(checkpoints) and attempt >= checkpoints[checkpoint_idx]:
            elapsed = perf_counter() - started
            curve.append(
                _curve_row(
                    checkpoints[checkpoint_idx],
                    {
                        "score": best_score,
                        "acc": best_acc,
                        "target_probability": best_target_probability,
                    },
                    stale,
                    net,
                    adapter,
                    elapsed,
                )
            )
            checkpoint_idx += 1

        if best_acc >= 1.0:
            break
        if config.stale_limit and stale >= config.stale_limit:
            break

    return SearchOutcome(
        best_score=best_score,
        best_acc=best_acc,
        best_target_probability=best_target_probability,
        steps=steps,
        stale=stale,
        connections=int(net.count_connections()),
        curve=curve,
        policy_state=adapter.describe_state(),
    )

