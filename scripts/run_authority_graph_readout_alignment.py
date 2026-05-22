#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import run_authority_graph_guided_pruning as guided
import run_authority_graph_pilot as pilot
import run_authority_graph_readout_repair as repair


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_GRAPH_EXPLICIT_READOUT_ALIGNMENT.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-graph-readout-alignment"
SUMMARY_NAME = "authority_graph_readout_alignment_summary.json"

ARMS = [
    "route_state_reference",
    "explicit_untrained",
    "CE_only",
    "CE_plus_output_authority",
    "CE_plus_inactive_leakage_penalty",
    "CE_plus_wrong_frame_margin",
    "combined_authority_readout_loss",
    "matched_random_combined_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align explicit authority graph readout with output-causal authority metrics.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--final-test-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
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
        args.learning_rate = 0.01
        args.checkpoint_every = 2
        args.max_runtime_hours = 0.25
        args.arms = [
            "route_state_reference",
            "explicit_untrained",
            "CE_only",
            "combined_authority_readout_loss",
            "matched_random_combined_loss",
        ]
    return args


def split_datasets(samples: int, seed: int) -> dict[str, list[Any]]:
    return pilot.make_datasets(samples, seed)


def all_logits(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], steps: int) -> torch.Tensor:
    return torch.cat([
        model.forward_static(datasets["latent_refraction_small"], steps=steps)["logit"],
        model.forward_static(datasets["multi_aspect_small"], steps=steps)["logit"],
        model.forward_temporal(datasets["temporal_order_contrast_small"], steps=steps)["logit"],
    ])


def all_labels(datasets: dict[str, list[Any]]) -> torch.Tensor:
    return torch.cat([
        repair.labels_static(datasets["latent_refraction_small"]),
        repair.labels_static(datasets["multi_aspect_small"]),
        repair.labels_temporal(datasets["temporal_order_contrast_small"]),
    ])


def bce_loss(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], steps: int) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(all_logits(model, datasets, steps), all_labels(datasets))


def static_examples(datasets: dict[str, list[Any]]) -> list[pilot.StaticExample]:
    return datasets["latent_refraction_small"] + datasets["multi_aspect_small"]


def capped_static_examples(examples: list[pilot.StaticExample], per_frame: int = 24) -> list[pilot.StaticExample]:
    capped: list[pilot.StaticExample] = []
    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        capped.extend(frame_examples[:per_frame])
    return capped


def output_authority_loss(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
    margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not examples:
        zero = torch.tensor(0.0)
        return zero, {"active": zero, "inactive": zero, "active_minus_inactive": zero}
    rng = np.random.default_rng(911)
    losses = []
    active_values = []
    inactive_values = []
    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in examples if ex.frame == frame]
        if not frame_examples:
            continue
        active_group = pilot.ACTIVE_GROUP_BY_FRAME[frame]
        base = torch.sigmoid(model.forward_static(frame_examples, steps=steps)["logit"])
        group_deltas: dict[str, torch.Tensor] = {}
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
            prob = torch.sigmoid(model.forward_static(swapped, steps=steps)["logit"])
            group_deltas[group] = torch.mean(torch.abs(base - prob))
        active = group_deltas[active_group]
        inactive = torch.stack([value for key, value in group_deltas.items() if key != active_group]).max()
        losses.append(F.relu(margin - (active - inactive)))
        active_values.append(active)
        inactive_values.append(inactive)
    active_mean = torch.stack(active_values).mean()
    inactive_mean = torch.stack(inactive_values).mean()
    return torch.stack(losses).mean(), {
        "active": active_mean,
        "inactive": inactive_mean,
        "active_minus_inactive": active_mean - inactive_mean,
    }


def inactive_leakage_loss(
    model: guided.DifferentiableAuthorityGraph,
    examples: list[pilot.StaticExample],
    *,
    steps: int,
    limit: float = 0.08,
) -> torch.Tensor:
    _, stats = output_authority_loss(model, examples, steps=steps, margin=0.0)
    return F.relu(stats["inactive"] - limit)


def wrong_frame_margin_loss(
    model: guided.DifferentiableAuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    steps: int,
    margin: float = 0.20,
) -> torch.Tensor:
    examples = datasets["latent_refraction_small"]
    if not examples:
        return torch.tensor(0.0)
    labels = repair.labels_static(examples)
    sign = labels * 2.0 - 1.0
    correct = model.forward_static(examples, steps=steps)["logit"] * sign
    wrong = model.forward_static(examples, steps=steps, forced_wrong_frame=True)["logit"] * sign
    return F.relu(margin - (correct - wrong)).mean()


def alignment_loss(
    model: guided.DifferentiableAuthorityGraph,
    datasets: dict[str, list[Any]],
    *,
    steps: int,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    task = bce_loss(model, datasets, steps)
    weights = {
        "CE_only": (0.0, 0.0, 0.0),
        "CE_plus_output_authority": (0.70, 0.0, 0.0),
        "CE_plus_inactive_leakage_penalty": (0.0, 0.80, 0.0),
        "CE_plus_wrong_frame_margin": (0.0, 0.0, 0.35),
        "combined_authority_readout_loss": (0.65, 0.55, 0.30),
        "matched_random_combined_loss": (0.65, 0.55, 0.30),
    }
    authority_w, inactive_w, wrong_w = weights[mode]
    zero = task.new_tensor(0.0)
    authority = zero
    inactive = zero
    wrong = zero
    auth_stats = {"active": zero, "inactive": zero}
    if authority_w > 0.0 or inactive_w > 0.0:
        authority_examples = capped_static_examples(static_examples(datasets), per_frame=24)
        authority, auth_stats = output_authority_loss(model, authority_examples, steps=steps, margin=0.10)
        if inactive_w > 0.0:
            inactive = F.relu(auth_stats["inactive"] - 0.08)
    if wrong_w > 0.0:
        wrong = wrong_frame_margin_loss(model, datasets, steps=steps, margin=0.20)
    edge_l2 = model.edge_weights.pow(2).mean()
    bias_l2 = model.bias.pow(2).mean()
    total = task + authority_w * authority + inactive_w * inactive + wrong_w * wrong + 0.0005 * edge_l2 + 0.0002 * bias_l2
    return total, {
        "task_bce": float(task.detach().item()),
        "authority_margin_loss": float(authority.detach().item()),
        "inactive_leakage_loss": float(inactive.detach().item()),
        "wrong_frame_margin_loss": float(wrong.detach().item()),
        "train_active_influence": float(auth_stats["active"].detach().item()),
        "train_inactive_influence": float(auth_stats["inactive"].detach().item()),
    }


def make_readout_mask(model: guided.DifferentiableAuthorityGraph) -> repair.TrainMask:
    return repair.make_train_mask(model, "readout_only")


def checkpoint_score(metrics: dict[str, Any], mode: str) -> float:
    if mode == "CE_only":
        return metrics["overall_accuracy"] - 0.10 * metrics["final_loss"]
    return (
        metrics["overall_accuracy"]
        + 0.85 * metrics["output_influence_authority_score"]
        + 0.35 * metrics["wrong_frame_drop"]
        + 0.25 * metrics["active_minus_inactive_margin"]
        - 0.15 * max(0.0, metrics["fraction_near_zero_logits"] - 0.50)
    )


def train_alignment(
    model: guided.DifferentiableAuthorityGraph,
    train: dict[str, list[Any]],
    validation: dict[str, list[Any]],
    *,
    args: argparse.Namespace,
    seed: int,
    arm: str,
    mode: str,
    deadline: float,
) -> tuple[guided.DifferentiableAuthorityGraph, guided.DifferentiableAuthorityGraph, dict[str, Any]]:
    torch.manual_seed(seed)
    mask = make_readout_mask(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    best = model.clone_frozen()
    best_auth = model.clone_frozen()
    best_metrics = evaluate_explicit_with_guards(best, validation, steps=args.steps)
    best_auth_metrics = best_metrics
    best_score = checkpoint_score(best_metrics, mode)
    best_auth_score = best_metrics["output_influence_authority_score"]
    history = []
    interrupted = False
    for epoch in range(1, args.epochs + 1):
        if time.time() >= deadline:
            interrupted = True
            break
        opt.zero_grad(set_to_none=True)
        loss, loss_parts = alignment_loss(model, train, steps=args.steps, mode=mode)
        loss.backward()
        repair.apply_train_mask(model, mask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        with torch.no_grad():
            model.edge_weights.clamp_(-4.0, 4.0)
            model.bias.clamp_(-4.0, 4.0)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            val = evaluate_explicit_with_guards(model, validation, steps=args.steps)
            score = checkpoint_score(val, mode)
            auth_score = val["output_influence_authority_score"]
            history.append({"epoch": epoch, "validation": val, "selection_score": score, "loss_parts": loss_parts})
            print(
                "[readout-align] "
                f"{arm} seed={seed} epoch={epoch}/{args.epochs} "
                f"score={score:.4f} acc={val['overall_accuracy']:.4f} "
                f"out_auth={val['output_influence_authority_score']:.4f} "
                f"near_zero={val['fraction_near_zero_logits']:.3f}",
                flush=True,
            )
            if score > best_score:
                best = model.clone_frozen()
                best_metrics = val
                best_score = score
            if auth_score > best_auth_score:
                best_auth = model.clone_frozen()
                best_auth_metrics = val
                best_auth_score = auth_score
    return best, best_auth, {
        "loss_mode": mode,
        "loss_type": "binary_cross_entropy_with_logits",
        "best_validation": best_metrics,
        "best_validation_authority_checkpoint": best_auth_metrics,
        "best_validation_authority_differs": best_auth_score != best_metrics["output_influence_authority_score"],
        "best_selection_score": best_score,
        "history": history,
        "interrupted": interrupted,
    }


@torch.no_grad()
def confidence_metrics(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], *, steps: int) -> dict[str, float]:
    latent = datasets["latent_refraction_small"]
    labels = repair.labels_static(latent)
    sign = labels * 2.0 - 1.0
    correct_logits = model.forward_static(latent, steps=steps)["logit"] * sign
    wrong_logits = model.forward_static(latent, steps=steps, forced_wrong_frame=True)["logit"] * sign
    logits = all_logits(model, datasets, steps)
    correct_conf = torch.sigmoid(correct_logits).mean()
    wrong_conf = torch.sigmoid(wrong_logits).mean()
    return {
        "correct_frame_confidence": float(correct_conf.item()),
        "wrong_frame_confidence": float(wrong_conf.item()),
        "mean_abs_logit": float(logits.abs().mean().item()),
        "fraction_near_zero_logits": float((logits.abs() < 0.15).float().mean().item()),
    }


@torch.no_grad()
def evaluate_explicit_with_guards(model: guided.DifferentiableAuthorityGraph, datasets: dict[str, list[Any]], *, steps: int) -> dict[str, Any]:
    metrics = repair.evaluate_explicit(model, datasets, steps=steps)
    metrics.update(confidence_metrics(model, datasets, steps=steps))
    metrics["active_minus_inactive_margin"] = metrics["active_group_influence"] - metrics["inactive_group_influence"]
    metrics["collapse_detected"] = bool(
        metrics["fraction_near_zero_logits"] > 0.70
        or (metrics["correct_frame_confidence"] < 0.58 and metrics["output_influence_authority_score"] > 0.02)
    )
    return metrics


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
        train_metrics = repair.evaluate_route_state_reference(model_or_graph, train, steps=args.steps, seed=seed + 1)
        validation_metrics = repair.evaluate_route_state_reference(model_or_graph, validation, steps=args.steps, seed=seed + 2)
        final_metrics = repair.evaluate_route_state_reference(model_or_graph, final_test, steps=args.steps, seed=seed + 3)
    else:
        train_metrics = evaluate_explicit_with_guards(model_or_graph, train, steps=args.steps)
        validation_metrics = evaluate_explicit_with_guards(model_or_graph, validation, steps=args.steps)
        final_metrics = evaluate_explicit_with_guards(model_or_graph, final_test, steps=args.steps)
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
        metrics["overall_accuracy"] >= 0.90
        and metrics["temporal_order_accuracy"] >= 0.90
        and (metrics.get("output_influence_authority_score") or 0.0) >= 0.18
        and not metrics.get("collapse_detected", False)
    )


def run_seed(args: argparse.Namespace, seed: int, deadline: float) -> list[dict[str, Any]]:
    train = split_datasets(args.train_samples, seed + 10_000)
    validation = split_datasets(args.validation_samples, seed + 20_000)
    final_test = split_datasets(args.final_test_samples, seed + 30_000)
    hand_graph = pilot.build_hand_seeded_graph(0.35)
    hand_spec = guided.build_hand_spec()
    random_spec = repair.build_matched_random_spec(seed + 50_000, hand_spec)
    records: list[dict[str, Any]] = []

    if "route_state_reference" in args.arms:
        print(f"[readout-align] evaluating route_state_reference seed={seed}", flush=True)
        records.append(
            evaluate_record(
                "route_state_reference",
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
    if "explicit_untrained" in args.arms:
        print(f"[readout-align] evaluating explicit_untrained seed={seed}", flush=True)
        records.append(
            evaluate_record(
                "explicit_untrained",
                guided.DifferentiableAuthorityGraph(hand_spec, trainable=False),
                train,
                validation,
                final_test,
                args=args,
                seed=seed,
                metadata={"readout_policy": "explicit_edges_untrained"},
            )
        )

    loss_specs = {
        "CE_only": (hand_spec, "CE_only", seed + 70_000),
        "CE_plus_output_authority": (hand_spec, "CE_plus_output_authority", seed + 80_000),
        "CE_plus_inactive_leakage_penalty": (hand_spec, "CE_plus_inactive_leakage_penalty", seed + 90_000),
        "CE_plus_wrong_frame_margin": (hand_spec, "CE_plus_wrong_frame_margin", seed + 100_000),
        "combined_authority_readout_loss": (hand_spec, "combined_authority_readout_loss", seed + 110_000),
        "matched_random_combined_loss": (random_spec, "matched_random_combined_loss", seed + 120_000),
    }
    for arm, (spec, mode, arm_seed) in loss_specs.items():
        if arm not in args.arms:
            continue
        model = guided.DifferentiableAuthorityGraph(spec, trainable=True)
        selected, best_auth, info = train_alignment(
            model,
            train,
            validation,
            args=args,
            seed=arm_seed,
            arm=arm,
            mode=mode,
            deadline=deadline,
        )
        print(f"[readout-align] evaluating {arm} seed={seed}", flush=True)
        record = evaluate_record(
            arm,
            selected,
            train,
            validation,
            final_test,
            args=args,
            seed=seed,
            metadata={
                "training": info,
                "selection_policy": "validation_accuracy_bce" if mode == "CE_only" else "validation_combined_authority_objective",
                "best_authority_checkpoint_final_test": evaluate_explicit_with_guards(best_auth, final_test, steps=args.steps),
            },
        )
        records.append(record)
    return records


def numeric_summary(values: list[Any]) -> dict[str, float | None]:
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
            "collapse_rate": float(np.mean([bool(row["final_test"].get("collapse_detected", False)) for row in rows])),
        }
        for split in ("train", "validation", "final_test"):
            keys = sorted({key for row in rows for key, value in row[split].items() if isinstance(value, (int, float, bool)) or value is None})
            arm_data[split] = {key: numeric_summary([row[split].get(key) for row in rows]) for key in keys}
        out[arm] = arm_data
    return out


def metric_mean(aggregate_data: dict[str, Any], arm: str, key: str) -> float:
    value = aggregate_data.get(arm, {}).get("final_test", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(aggregate_data: dict[str, Any]) -> dict[str, bool]:
    ce_auth = metric_mean(aggregate_data, "CE_only", "output_influence_authority_score")
    ce_acc = metric_mean(aggregate_data, "CE_only", "overall_accuracy")
    ce_inactive = metric_mean(aggregate_data, "CE_only", "inactive_group_influence")
    ce_wrong = metric_mean(aggregate_data, "CE_only", "wrong_frame_drop")
    combined_auth = metric_mean(aggregate_data, "combined_authority_readout_loss", "output_influence_authority_score")
    combined_acc = metric_mean(aggregate_data, "combined_authority_readout_loss", "overall_accuracy")
    combined_inactive = metric_mean(aggregate_data, "combined_authority_readout_loss", "inactive_group_influence")
    combined_wrong = metric_mean(aggregate_data, "combined_authority_readout_loss", "wrong_frame_drop")
    random_acc = metric_mean(aggregate_data, "matched_random_combined_loss", "overall_accuracy")
    random_auth = metric_mean(aggregate_data, "matched_random_combined_loss", "output_influence_authority_score")
    collapse = aggregate_data.get("combined_authority_readout_loss", {}).get("collapse_rate", 0.0) > 0.0
    best_aligned_auth = max(
        metric_mean(aggregate_data, arm, "output_influence_authority_score")
        for arm in [
            "CE_plus_output_authority",
            "CE_plus_inactive_leakage_penalty",
            "CE_plus_wrong_frame_margin",
            "combined_authority_readout_loss",
        ]
    )
    return {
        "authority_alignment_improves_output_authority": best_aligned_auth > ce_auth + 0.05,
        "accuracy_preserved_under_alignment": combined_acc >= max(0.90, ce_acc - 0.05),
        "inactive_leakage_reduced": combined_inactive < ce_inactive - 0.02,
        "wrong_frame_sensitivity_preserved": combined_wrong >= max(0.20, ce_wrong - 0.05),
        "combined_loss_beats_ce_only": combined_auth > ce_auth + 0.05 and combined_acc >= ce_acc - 0.05,
        "collapse_detected": collapse,
        "random_control_still_weak": random_acc < combined_acc - 0.15 or random_auth < combined_auth - 0.10,
        "route_state_shortcut_still_diagnostic_only": True,
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


def summary_mean(split: dict[str, Any], key: str) -> Any:
    return split.get(key, {}).get("mean")


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
        "# Authority Graph Explicit Readout Alignment",
        "",
        "## Goal",
        "",
        "Test whether explicit readout training can align output-causal authority with the existing hand-seeded route-state mechanism.",
        "",
        "Route-state authority is diagnostic only. Verdicts prioritize explicit output influence and collapse guards.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Final-Test Results",
        "",
        "| Arm | Success | Collapse | Accuracy | Temporal | Output Authority | Route-State Authority | Active | Inactive | Margin | Wrong Frame | Correct Conf | Wrong Conf | Mean Abs Logit | Near-Zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        if arm not in aggregate_data:
            continue
        item = aggregate_data[arm]
        final = item["final_test"]
        lines.append(
            f"| `{arm}` | `{fmt(item['success_rate'])}` | `{fmt(item['collapse_rate'])}` "
            f"| `{fmt(summary_mean(final, 'overall_accuracy'))}` "
            f"| `{fmt(summary_mean(final, 'temporal_order_accuracy'))}` "
            f"| `{fmt(summary_mean(final, 'output_influence_authority_score'))}` "
            f"| `{fmt(summary_mean(final, 'route_state_authority_score'))}` "
            f"| `{fmt(summary_mean(final, 'active_group_influence'))}` "
            f"| `{fmt(summary_mean(final, 'inactive_group_influence'))}` "
            f"| `{fmt(summary_mean(final, 'active_minus_inactive_margin'))}` "
            f"| `{fmt(summary_mean(final, 'wrong_frame_drop'))}` "
            f"| `{fmt(summary_mean(final, 'correct_frame_confidence'))}` "
            f"| `{fmt(summary_mean(final, 'wrong_frame_confidence'))}` "
            f"| `{fmt(summary_mean(final, 'mean_abs_logit'))}` "
            f"| `{fmt(summary_mean(final, 'fraction_near_zero_logits'))}` |"
        )
    lines.extend([
        "",
        "## Train / Validation / Final-Test Gap",
        "",
        "| Arm | Train Acc | Val Acc | Final Acc | Train Auth | Val Auth | Final Auth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in arms:
        if arm not in aggregate_data:
            continue
        item = aggregate_data[arm]
        lines.append(
            f"| `{arm}` | `{fmt(summary_mean(item['train'], 'overall_accuracy'))}` "
            f"| `{fmt(summary_mean(item['validation'], 'overall_accuracy'))}` "
            f"| `{fmt(summary_mean(item['final_test'], 'overall_accuracy'))}` "
            f"| `{fmt(summary_mean(item['train'], 'output_influence_authority_score'))}` "
            f"| `{fmt(summary_mean(item['validation'], 'output_influence_authority_score'))}` "
            f"| `{fmt(summary_mean(item['final_test'], 'output_influence_authority_score'))}` |"
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
        "- BCE is used because the graph exposes one binary positive-vs-negative readout logit.",
        "- CE-only selects by validation accuracy/loss; authority-aligned arms select by validation combined authority objective.",
        "- Each aligned arm also records the best-validation-authority checkpoint in JSON metadata.",
        "- Collapse is flagged when authority appears to improve through low-confidence or near-zero logits.",
        "- Matched random control uses the same readout capacity and training budget; only internal topology differs.",
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
        print(f"[readout-align] seed={seed}", flush=True)
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
