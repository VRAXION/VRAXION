#!/usr/bin/env python3
from __future__ import annotations

"""Detach-fixed entrypoint for authority graph readout alignment.

This wrapper leaves run_authority_graph_readout_alignment.py intact, patches only the
inactive-leakage detach bug at runtime, and adds --grad-audit for deterministic
pre-training validation.
"""

import json
import sys
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import run_authority_graph_guided_pruning as guided
import run_authority_graph_pilot as pilot
import run_authority_graph_readout_alignment as align

_ORIGINAL_INACTIVE_LEAKAGE_LOSS = align.inactive_leakage_loss


def _zero(model: guided.DifferentiableAuthorityGraph) -> torch.Tensor:
    return model.edge_weights.sum() * 0.0


def output_authority_components(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        z = _zero(model)
        return z, z, z

    rng = np.random.default_rng(911)
    active_values: list[torch.Tensor] = []
    inactive_values: list[torch.Tensor] = []

    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        active_group = pilot.ACTIVE_GROUP_BY_FRAME[frame]
        base = torch.sigmoid(model.forward_static(frame_examples, steps=steps)["logit"])
        group_deltas: dict[str, torch.Tensor] = {}
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
            prob = torch.sigmoid(model.forward_static(swapped, steps=steps)["logit"])
            group_deltas[group] = torch.mean(torch.abs(base - prob))
        active = group_deltas[active_group]
        inactive = torch.stack([v for k, v in group_deltas.items() if k != active_group]).max()
        active_values.append(active)
        inactive_values.append(inactive)

    if not active_values:
        z = _zero(model)
        return z, z, z
    active_mean = torch.stack(active_values).mean()
    inactive_mean = torch.stack(inactive_values).mean()
    return active_mean, inactive_mean, active_mean - inactive_mean


def output_authority_loss_fixed(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
    margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active, inactive, active_minus_inactive = output_authority_components(model, examples, steps=steps)
    loss = F.relu(margin - active_minus_inactive)
    return loss, {
        "active": active.detach(),
        "inactive": inactive.detach(),
        "active_minus_inactive": active_minus_inactive.detach(),
    }


def inactive_leakage_loss_fixed(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
    limit: float = 0.08,
) -> torch.Tensor:
    _, inactive, _ = output_authority_components(model, examples, steps=steps)
    return F.relu(inactive - limit)


def patch_alignment_losses() -> None:
    align.output_authority_loss = output_authority_loss_fixed
    align.inactive_leakage_loss = inactive_leakage_loss_fixed


def _readout_mask(model: guided.DifferentiableAuthorityGraph) -> torch.Tensor:
    return torch.tensor([t == "route->readout" for t in model.edge_types], dtype=torch.bool)


def _norm(tensor: torch.Tensor | None) -> float | None:
    return None if tensor is None else float(tensor.detach().norm().item())


def _audit_row(
    name: str,
    model: guided.DifferentiableAuthorityGraph,
    make_loss: Callable[[], torch.Tensor],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    loss = make_loss()
    row: dict[str, Any] = {
        "loss": name,
        "value": float(loss.detach().item()) if loss.numel() == 1 else None,
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
    row["all_edge_grad_norm"] = _norm(edge_grad)
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
    row["frame_gate_grad_norm"] = _norm(model.frame_gate.grad)
    row["suppressor_strength_grad_norm"] = _norm(model.suppressor_strength.grad)
    return row


def run_grad_audit() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    steps = 3
    datasets = align.split_datasets(64, 12345)
    examples = align.static_examples(datasets)
    hand_spec = guided.build_hand_spec()

    def fresh() -> guided.DifferentiableAuthorityGraph:
        return guided.DifferentiableAuthorityGraph(hand_spec, trainable=True)

    rows: list[dict[str, Any]] = []
    m = fresh()
    rows.append(_audit_row("original_inactive_leakage_loss", m, lambda: _ORIGINAL_INACTIVE_LEAKAGE_LOSS(m, examples, steps=steps, limit=0.08)))
    m = fresh()
    rows.append(_audit_row("fixed_inactive_leakage_loss", m, lambda: inactive_leakage_loss_fixed(m, examples, steps=steps, limit=0.08)))
    m = fresh()
    rows.append(_audit_row("fixed_output_authority_loss", m, lambda: output_authority_loss_fixed(m, examples, steps=steps, margin=0.10)[0]))
    m = fresh()
    rows.append(_audit_row("bce_loss", m, lambda: align.bce_loss(m, datasets, steps)))
    m = fresh()
    patch_alignment_losses()
    rows.append(_audit_row("fixed_combined_authority_readout_loss", m, lambda: align.alignment_loss(m, datasets, steps=steps, mode="combined_authority_readout_loss")[0]))

    print(json.dumps({"grad_audit": rows}, indent=2), flush=True)


def main() -> None:
    if "--grad-audit" in sys.argv:
        sys.argv.remove("--grad-audit")
        run_grad_audit()
        return
    patch_alignment_losses()
    align.main()


if __name__ == "__main__":
    main()
