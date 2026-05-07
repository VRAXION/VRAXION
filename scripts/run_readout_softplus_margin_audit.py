#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic readout-authority loss audit.

Compares the current probability-delta authority loss against a label-signed
softplus margin-authority loss without running full training.
"""

import json
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import run_authority_graph_guided_pruning as guided
import run_authority_graph_pilot as pilot
import run_authority_graph_readout_alignment as align
import run_authority_graph_readout_repair as repair


def softplus_margin_authority_loss(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
    target_margin: float = 0.10,
    beta: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not examples:
        z = model.edge_weights.sum() * 0.0
        return z, {"active_margin_drop": z, "inactive_margin_effect": z, "authority_gap": z}

    rng = np.random.default_rng(917)
    losses: list[torch.Tensor] = []
    active_values: list[torch.Tensor] = []
    inactive_values: list[torch.Tensor] = []
    gap_values: list[torch.Tensor] = []

    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        labels = repair.labels_static(frame_examples)
        sign = labels * 2.0 - 1.0
        active_group = pilot.ACTIVE_GROUP_BY_FRAME[frame]
        base_margin = sign * model.forward_static(frame_examples, steps=steps)["logit"]
        group_effects: dict[str, torch.Tensor] = {}

        for group in pilot.GROUP_FIELDS:
            swapped: list[pilot.StaticExample] = []
            for ex in frame_examples:
                donor = frame_examples[int(rng.integers(0, len(frame_examples)))].obs
                swapped.append(
                    pilot.StaticExample(
                        obs=guided.swap_observation(ex.obs, donor, group),
                        frame=frame,
                        label=ex.label,
                    )
                )
            swapped_margin = sign * model.forward_static(swapped, steps=steps)["logit"]
            delta = base_margin - swapped_margin
            group_effects[group] = delta.mean() if group == active_group else delta.abs().mean()

        active_drop = group_effects[active_group]
        inactive_effect = torch.stack([v for k, v in group_effects.items() if k != active_group]).max()
        authority_gap = active_drop - inactive_effect
        losses.append(F.softplus(beta * (target_margin - authority_gap)) / beta)
        active_values.append(active_drop)
        inactive_values.append(inactive_effect)
        gap_values.append(authority_gap)

    if not losses:
        z = model.edge_weights.sum() * 0.0
        return z, {"active_margin_drop": z, "inactive_margin_effect": z, "authority_gap": z}

    return torch.stack(losses).mean(), {
        "active_margin_drop": torch.stack(active_values).mean().detach(),
        "inactive_margin_effect": torch.stack(inactive_values).mean().detach(),
        "authority_gap": torch.stack(gap_values).mean().detach(),
    }


def _readout_mask(model: guided.DifferentiableAuthorityGraph) -> torch.Tensor:
    return torch.tensor([t == "route->readout" for t in model.edge_types], dtype=torch.bool)


def _audit_row(
    name: str,
    model: guided.DifferentiableAuthorityGraph,
    make_loss: Callable[[], torch.Tensor],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    loss = make_loss()
    row: dict[str, Any] = {
        "loss": name,
        "value": float(loss.detach().item()),
        "requires_grad": bool(loss.requires_grad),
        "grad_fn": type(loss.grad_fn).__name__ if loss.grad_fn is not None else None,
    }
    try:
        loss.backward()
    except RuntimeError as exc:
        row["backward_error"] = str(exc)
        return row
    edge_grad = model.edge_weights.grad
    readout_mask = _readout_mask(model)
    row["all_edge_grad_norm"] = None if edge_grad is None else float(edge_grad.detach().norm().item())
    row["route_readout_grad_norm"] = (
        float(edge_grad[readout_mask].detach().norm().item())
        if edge_grad is not None and bool(readout_mask.any())
        else None
    )
    if model.bias.grad is not None:
        idx = [model.idx["readout_positive"], model.idx["readout_negative"]]
        row["readout_bias_grad_norm"] = float(model.bias.grad[idx].detach().norm().item())
    else:
        row["readout_bias_grad_norm"] = None
    return row


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    steps = 3
    datasets = align.split_datasets(64, 12345)
    examples = align.static_examples(datasets)
    hand_spec = guided.build_hand_spec()

    def fresh() -> guided.DifferentiableAuthorityGraph:
        return guided.DifferentiableAuthorityGraph(hand_spec, trainable=True)

    rows: list[dict[str, Any]] = []
    model = fresh()
    rows.append(_audit_row("current_probability_delta_authority", model, lambda: align.output_authority_loss(model, examples, steps=steps, margin=0.10)[0]))
    model = fresh()
    rows.append(_audit_row("softplus_margin_authority", model, lambda: softplus_margin_authority_loss(model, examples, steps=steps, target_margin=0.10, beta=5.0)[0]))
    model = fresh()
    rows.append(_audit_row("bce_loss", model, lambda: align.bce_loss(model, datasets, steps)))

    print(json.dumps({"softplus_margin_audit": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
