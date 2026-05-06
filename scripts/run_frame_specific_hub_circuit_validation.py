#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_authority_circuit_dissection as dissect  # noqa: E402
import run_context_cancellation_probe as probe  # noqa: E402


DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "frame-specific-hub-circuit"
REPORT_PATH = ROOT / "docs" / "research" / "FRAME_SPECIFIC_HUB_CIRCUIT_VALIDATION.md"
FRAME_LABELS = {
    "danger_frame": "danger_specific",
    "environment_frame": "environment_specific",
    "visibility_frame": "visibility_specific",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame-specific hub/route diagnostic for latent_refraction authority switching."
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--test-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sparse-density", type=float, default=0.12)
    parser.add_argument("--update-rate", type=float, default=0.3)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--active-value", type=float, default=1.0)
    parser.add_argument("--embed-scale", type=float, default=0.80)
    parser.add_argument("--frame-scale", type=float, default=1.10)
    parser.add_argument("--opponent-strength", type=float, default=0.80)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--topology-modes",
        nargs="+",
        choices=("random_sparse", "hub_rich", "hub_degree_preserving_random"),
        default=["hub_degree_preserving_random", "hub_rich", "random_sparse"],
    )
    parser.add_argument(
        "--max-node-ablation",
        type=int,
        default=64,
        help="Use hidden size for exhaustive node ablation; lower this for quick partial runs.",
    )
    parser.add_argument("--edge-group-fraction", type=float, default=0.05)
    parser.add_argument("--random-edge-controls", type=int, default=3)
    parser.add_argument("--top-k-fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.hidden = 16
        args.steps = 2
        args.epochs = 2
        args.train_size = 96
        args.test_size = 48
        args.batch_size = 32
        args.topology_modes = ["random_sparse"]
        args.max_node_ablation = 4
        args.random_edge_controls = 1
        args.top_k_fractions = [0.10, 0.20]
    return args


def authority_args(args: argparse.Namespace, topology_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        seeds=args.seeds,
        latent_hidden=args.hidden,
        multi_hidden=args.hidden,
        steps=args.steps,
        latent_epochs=args.epochs,
        multi_epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        lr=args.lr,
        sparse_density=args.sparse_density,
        topology_modes=[topology_mode],
        include_hub_rich_multi=False,
        max_node_ablation=args.max_node_ablation,
        random_node_controls=1,
        edge_group_fraction=args.edge_group_fraction,
        minimal_fractions=args.top_k_fractions,
        out_dir=args.out_dir,
        smoke=args.smoke,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        active_value=args.active_value,
        embed_scale=args.embed_scale,
        frame_scale=args.frame_scale,
        opponent_strength=args.opponent_strength,
        holdout_fraction=args.holdout_fraction,
        ridge=args.ridge,
        device=args.device,
    )


def frame_metrics(
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    prediction = probe.refraction_prediction_summary(model=model, bundle=test, args=args)
    influence = probe.run_refraction_influence(model=model, test=test, args=args, seed=seed)
    by_frame: dict[str, dict[str, float | str]] = {}
    for frame_name in test.frame_names:
        active_group = probe.bundle_active_group(test, frame_name)
        active_curve = influence["active_core_influence_by_step"][frame_name]
        inactive_curve = influence["inactive_group_influence_by_step"][frame_name]
        refraction_curve = influence["refraction_index_by_step"][frame_name]
        by_frame[frame_name] = {
            "active_group": active_group,
            "accuracy": float(prediction["accuracy_by_frame"][frame_name]),
            "active_group_influence": float(active_curve[-1]),
            "inactive_group_influence": float(inactive_curve[-1]),
            "refraction_index": float(refraction_curve[-1]),
            "authority": float(refraction_curve[-1]),
        }
    mean_active = float(np.mean([item["active_group_influence"] for item in by_frame.values()]))
    mean_inactive = float(np.mean([item["inactive_group_influence"] for item in by_frame.values()]))
    mean_refraction = float(np.mean([item["refraction_index"] for item in by_frame.values()]))
    return {
        "accuracy": float(prediction["accuracy"]),
        "accuracy_by_frame": prediction["accuracy_by_frame"],
        "authority_switch_score": influence["authority_switch_score"],
        "refraction_index_final": mean_refraction,
        "active_group_influence": mean_active,
        "inactive_group_influence": mean_inactive,
        "by_frame": by_frame,
    }


def frame_drop(baseline: dict[str, Any], ablated: dict[str, Any], frame_name: str) -> dict[str, float]:
    base = baseline["by_frame"][frame_name]
    after = ablated["by_frame"][frame_name]
    return {
        "target_frame_accuracy_drop": float(base["accuracy"] - after["accuracy"]),
        "target_frame_active_group_influence_drop": float(
            base["active_group_influence"] - after["active_group_influence"]
        ),
        "target_frame_inactive_group_influence_rise": float(
            after["inactive_group_influence"] - base["inactive_group_influence"]
        ),
        "target_frame_refraction_drop": float(base["refraction_index"] - after["refraction_index"]),
        "target_frame_authority_drop": float(base["authority"] - after["authority"]),
    }


def overall_drop(baseline: dict[str, Any], ablated: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy_drop": float(baseline["accuracy"] - ablated["accuracy"]),
        "authority_switch_drop": float(
            (baseline.get("authority_switch_score") or 0.0) - (ablated.get("authority_switch_score") or 0.0)
        ),
        "refraction_drop": float(baseline["refraction_index_final"] - ablated["refraction_index_final"]),
        "active_group_influence_drop": float(
            baseline["active_group_influence"] - ablated["active_group_influence"]
        ),
        "inactive_group_influence_rise": float(
            ablated["inactive_group_influence"] - baseline["inactive_group_influence"]
        ),
    }


def classify_node(per_frame: dict[str, dict[str, float]], total_degree: float) -> dict[str, Any]:
    refraction = {
        frame: values["target_frame_refraction_drop"]
        for frame, values in per_frame.items()
    }
    inactive_rise = {
        frame: values["target_frame_inactive_group_influence_rise"]
        for frame, values in per_frame.items()
    }
    active_drop = {
        frame: values["target_frame_active_group_influence_drop"]
        for frame, values in per_frame.items()
    }
    best_frame = max(refraction, key=lambda frame: refraction[frame])
    best_drop = refraction[best_frame]
    other = [value for frame, value in refraction.items() if frame != best_frame]
    other_mean = float(np.mean(other)) if other else 0.0
    specificity = float(best_drop - other_mean)
    min_drop = min(refraction.values()) if refraction else 0.0
    max_inactive = max(inactive_rise.values()) if inactive_rise else 0.0
    max_active = max(active_drop.values()) if active_drop else 0.0

    suppressor = max_inactive > 0.04 and max_inactive >= max_active * 0.50 and best_drop > 0.04
    if min_drop > 0.07 and specificity < 0.06:
        label = "global_hub"
    elif best_drop > 0.08 and specificity > 0.06:
        label = FRAME_LABELS.get(best_frame, f"{best_frame}_specific")
    elif suppressor:
        label = "suppressor_candidate"
    else:
        label = "unclear"

    return {
        "classification": label,
        "best_frame": best_frame,
        "best_refraction_drop": float(best_drop),
        "mean_non_target_refraction_drop": other_mean,
        "frame_specificity_score": specificity,
        "max_inactive_group_influence_rise": float(max_inactive),
        "max_active_group_influence_drop": float(max_active),
        "suppressor_candidate": bool(suppressor),
        "total_degree": float(total_degree),
    }


def node_candidates(model: probe.RecurrentClassifier, max_nodes: int, seed: int) -> list[int]:
    hidden = model.recurrent.shape[0]
    if max_nodes >= hidden:
        return list(range(hidden))
    return dissect.candidate_nodes(model, max_nodes, seed)


def run_per_frame_node_saliency(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    seed: int,
    max_nodes: int,
) -> dict[str, Any]:
    total_degree = dissect.degree_stats(model)["total_degree"]
    rows = []
    for node in node_candidates(model, max_nodes, seed):
        ablated = frame_metrics(
            dissect.clone_ablate_nodes(model, [node]),
            test,
            args,
            seed=seed + 10_000 + node,
        )
        per_frame = {
            frame_name: frame_drop(baseline, ablated, frame_name)
            for frame_name in test.frame_names
        }
        row = {
            "node": int(node),
            "total_degree": float(total_degree[node]),
            "overall_drop": overall_drop(baseline, ablated),
            "per_frame": per_frame,
            **classify_node(per_frame, float(total_degree[node])),
        }
        rows.append(row)

    top_by_frame = {}
    for frame_name in test.frame_names:
        top_by_frame[frame_name] = sorted(
            rows,
            key=lambda item: item["per_frame"][frame_name]["target_frame_refraction_drop"],
            reverse=True,
        )[:10]
    classification_counts: dict[str, int] = {}
    for row in rows:
        classification_counts[row["classification"]] = classification_counts.get(row["classification"], 0) + 1
    suppressors = [
        row for row in rows
        if row["suppressor_candidate"]
    ]
    suppressors.sort(key=lambda item: item["max_inactive_group_influence_rise"], reverse=True)
    rows.sort(key=lambda item: item["overall_drop"]["authority_switch_drop"], reverse=True)
    return {
        "tested_node_count": len(rows),
        "classification_counts": classification_counts,
        "top10_overall": rows[:10],
        "top10_by_frame": top_by_frame,
        "suppressor_candidates_top10": suppressors[:10],
        "all_tested": rows,
    }


def mask_edges(model: probe.RecurrentClassifier) -> list[tuple[int, int]]:
    mask = model.mask.detach().cpu().numpy() > 0
    return [(int(target), int(source)) for target, source in np.argwhere(mask) if int(target) != int(source)]


def sorted_edges(model: probe.RecurrentClassifier, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    weights = np.abs(model.recurrent.detach().cpu().numpy())
    return sorted(edges, key=lambda edge: weights[edge[0], edge[1]], reverse=True)


def edge_group_drop(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    edges: list[tuple[int, int]],
    seed: int,
) -> dict[str, Any]:
    summary = frame_metrics(dissect.clone_zero_edges(model, edges), test, args, seed=seed)
    return {
        "edge_count": len(edges),
        "drop": overall_drop(baseline, summary),
        "per_frame_refraction_drop": {
            frame_name: frame_drop(baseline, summary, frame_name)["target_frame_refraction_drop"]
            for frame_name in test.frame_names
        },
        "per_frame_inactive_rise": {
            frame_name: frame_drop(baseline, summary, frame_name)["target_frame_inactive_group_influence_rise"]
            for frame_name in test.frame_names
        },
    }


def mean_drop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    scalar_keys = rows[0]["drop"].keys()
    out = {
        "edge_count": float(np.mean([row["edge_count"] for row in rows])),
        "drop": {
            key: float(np.mean([row["drop"][key] for row in rows]))
            for key in scalar_keys
        },
    }
    frame_keys = rows[0].get("per_frame_refraction_drop", {}).keys()
    out["per_frame_refraction_drop"] = {
        frame: float(np.mean([row["per_frame_refraction_drop"].get(frame, 0.0) for row in rows]))
        for frame in frame_keys
    }
    out["per_frame_inactive_rise"] = {
        frame: float(np.mean([row["per_frame_inactive_rise"].get(frame, 0.0) for row in rows]))
        for frame in frame_keys
    }
    return out


def run_hub_edge_routes(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    seed: int,
    fraction: float,
    random_controls: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    edges = mask_edges(model)
    if not edges:
        return {}
    total_degree = dissect.degree_stats(model)["total_degree"]
    hub_count = max(1, int(round(model.recurrent.shape[0] * 0.10)))
    hubs = set(int(idx) for idx in np.argsort(-total_degree)[:hub_count])
    budget = max(1, int(round(len(edges) * fraction)))
    groups = {
        "hub_outgoing_edges": [edge for edge in edges if edge[1] in hubs],
        "hub_incoming_edges": [edge for edge in edges if edge[0] in hubs],
        "hub_to_hub_edges": [edge for edge in edges if edge[0] in hubs and edge[1] in hubs],
        "hub_any_edges": [edge for edge in edges if edge[0] in hubs or edge[1] in hubs],
    }
    out: dict[str, Any] = {}
    for name, candidates in groups.items():
        chosen = sorted_edges(model, candidates)[: min(budget, len(candidates))]
        if not chosen:
            continue
        group_result = edge_group_drop(
            model=model,
            test=test,
            args=args,
            baseline=baseline,
            edges=chosen,
            seed=seed + len(chosen) + 20_000,
        )
        random_rows = []
        for control_idx in range(random_controls):
            random_edges = [edges[int(idx)] for idx in rng.choice(len(edges), size=len(chosen), replace=False)]
            random_rows.append(
                edge_group_drop(
                    model=model,
                    test=test,
                    args=args,
                    baseline=baseline,
                    edges=random_edges,
                    seed=seed + control_idx + len(chosen) + 21_000,
                )
            )
        out[name] = {
            **group_result,
            "random_same_count_mean": mean_drop(random_rows),
        }
    return out


def top_node_sets_by_frame(
    node_saliency: dict[str, Any],
    hidden: int,
    fractions: list[float],
) -> dict[str, Any]:
    rows = node_saliency["all_tested"]
    frames = list(node_saliency["top10_by_frame"].keys())
    out = {}
    for fraction in fractions:
        count = max(1, int(round(hidden * fraction)))
        frame_sets = {
            frame: {
                int(row["node"])
                for row in sorted(
                    rows,
                    key=lambda item: item["per_frame"][frame]["target_frame_refraction_drop"],
                    reverse=True,
                )[:count]
            }
            for frame in frames
        }
        out[f"{int(fraction * 100)}pct"] = overlap_report(frame_sets)
    return out


def frame_edge_proxy_sets(
    *,
    model: probe.RecurrentClassifier,
    node_saliency: dict[str, Any],
    fractions: list[float],
) -> dict[str, Any]:
    edges = mask_edges(model)
    if not edges:
        return {}
    weights = np.abs(model.recurrent.detach().cpu().numpy())
    node_scores: dict[str, dict[int, float]] = {}
    frames = list(node_saliency["top10_by_frame"].keys())
    for frame in frames:
        node_scores[frame] = {
            int(row["node"]): max(0.0, row["per_frame"][frame]["target_frame_refraction_drop"])
            for row in node_saliency["all_tested"]
        }
    out = {}
    for fraction in fractions:
        count = max(1, int(round(len(edges) * fraction)))
        frame_sets = {}
        for frame in frames:
            scores = node_scores[frame]
            ranked = sorted(
                edges,
                key=lambda edge: weights[edge[0], edge[1]] * (scores.get(edge[0], 0.0) + scores.get(edge[1], 0.0)),
                reverse=True,
            )
            frame_sets[frame] = set(ranked[:count])
        out[f"{int(fraction * 100)}pct"] = overlap_report(frame_sets)
    return out


def overlap_report(frame_sets: dict[str, set[Any]]) -> dict[str, Any]:
    frames = list(frame_sets)
    matrix = {}
    for left in frames:
        matrix[left] = {}
        for right in frames:
            union = frame_sets[left] | frame_sets[right]
            intersection = frame_sets[left] & frame_sets[right]
            matrix[left][right] = float(len(intersection) / len(union)) if union else 0.0
    shared = set.intersection(*frame_sets.values()) if frame_sets else set()
    unique = {
        frame: len(nodes - set.union(*(frame_sets[other] for other in frames if other != frame)))
        if len(frames) > 1 else len(nodes)
        for frame, nodes in frame_sets.items()
    }
    return {
        "jaccard_matrix": matrix,
        "shared_count": len(shared),
        "unique_count_by_frame": unique,
        "set_size_by_frame": {frame: len(nodes) for frame, nodes in frame_sets.items()},
    }


def run_config(*, seed: int, topology_mode: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    authority_ns = authority_args(args, topology_mode)
    artifact = dissect.train_refraction_model(
        experiment="latent_refraction",
        seed=seed,
        args=authority_ns,
        topology_mode=topology_mode,
    )
    model = artifact["model"]
    test = artifact["test"]
    probe_args = artifact["args"]
    baseline = frame_metrics(model, test, probe_args, seed=seed + 60_000)
    node_saliency = run_per_frame_node_saliency(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        seed=seed + 61_000,
        max_nodes=args.max_node_ablation,
    )
    hub_routes = run_hub_edge_routes(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        seed=seed + 62_000,
        fraction=args.edge_group_fraction,
        random_controls=args.random_edge_controls,
    )
    return {
        "seed": seed,
        "topology_mode": topology_mode,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "baseline": baseline,
        "topology": getattr(model, "topology_stats", {}),
        "node_saliency": node_saliency,
        "hub_edge_routes": hub_routes,
        "route_overlap": {
            "node_overlap": top_node_sets_by_frame(node_saliency, args.hidden, args.top_k_fractions),
            "edge_proxy_overlap": frame_edge_proxy_sets(
                model=model,
                node_saliency=node_saliency,
                fractions=args.top_k_fractions,
            ),
        },
        "runtime_seconds": time.time() - started,
    }


def aggregate_numeric(values: list[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(numeric)), "std": float(np.std(numeric))}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["topology_mode"], []).append(run)
    out: dict[str, Any] = {}
    for topology, rows in grouped.items():
        labels = sorted({
            label
            for row in rows
            for label in row["node_saliency"]["classification_counts"]
        })
        classification_counts = {
            label: aggregate_numeric([
                row["node_saliency"]["classification_counts"].get(label, 0)
                for row in rows
            ])
            for label in labels
        }
        edge_routes = {}
        route_names = sorted({
            name
            for row in rows
            for name in row["hub_edge_routes"]
        })
        for name in route_names:
            edge_routes[name] = {
                "authority_switch_drop": aggregate_numeric([
                    row["hub_edge_routes"].get(name, {}).get("drop", {}).get("authority_switch_drop")
                    for row in rows
                ]),
                "random_authority_switch_drop": aggregate_numeric([
                    row["hub_edge_routes"].get(name, {}).get("random_same_count_mean", {}).get("drop", {}).get("authority_switch_drop")
                    for row in rows
                ]),
                "refraction_drop": aggregate_numeric([
                    row["hub_edge_routes"].get(name, {}).get("drop", {}).get("refraction_drop")
                    for row in rows
                ]),
                "random_refraction_drop": aggregate_numeric([
                    row["hub_edge_routes"].get(name, {}).get("random_same_count_mean", {}).get("drop", {}).get("refraction_drop")
                    for row in rows
                ]),
                "inactive_group_influence_rise": aggregate_numeric([
                    row["hub_edge_routes"].get(name, {}).get("drop", {}).get("inactive_group_influence_rise")
                    for row in rows
                ]),
            }
        out[topology] = {
            "run_count": len(rows),
            "baseline_accuracy": aggregate_numeric([row["baseline"]["accuracy"] for row in rows]),
            "baseline_authority_switch_score": aggregate_numeric([
                row["baseline"]["authority_switch_score"] for row in rows
            ]),
            "baseline_refraction_index_final": aggregate_numeric([
                row["baseline"]["refraction_index_final"] for row in rows
            ]),
            "classification_counts": classification_counts,
            "suppressor_candidate_count": aggregate_numeric([
                len(row["node_saliency"]["suppressor_candidates_top10"])
                for row in rows
            ]),
            "hub_edge_routes": edge_routes,
            "node_overlap_10pct_shared": aggregate_numeric([
                row["route_overlap"]["node_overlap"].get("10pct", {}).get("shared_count")
                for row in rows
            ]),
        }
    return out


def verdict(agg: dict[str, Any]) -> dict[str, str]:
    global_count = sum(
        (item["classification_counts"].get("global_hub", {}).get("mean") or 0.0)
        for item in agg.values()
    )
    frame_specific_count = sum(
        (item["classification_counts"].get(label, {}).get("mean") or 0.0)
        for item in agg.values()
        for label in ("danger_specific", "environment_specific", "visibility_specific")
    )
    suppressor_count = sum(
        item["suppressor_candidate_count"]["mean"] or 0.0
        for item in agg.values()
    )
    route_positive = False
    for item in agg.values():
        for route in item["hub_edge_routes"].values():
            hub_drop = route["authority_switch_drop"]["mean"] or 0.0
            random_drop = route["random_authority_switch_drop"]["mean"] or 0.0
            ref_drop = route["refraction_drop"]["mean"] or 0.0
            random_ref = route["random_refraction_drop"]["mean"] or 0.0
            if hub_drop > random_drop + 0.04 and ref_drop > random_ref + 0.04:
                route_positive = True
    return {
        "supports_global_hub_bottlenecks": "true" if global_count >= 1 else "unclear",
        "supports_frame_specific_hubs": "true" if frame_specific_count >= 2 else "unclear",
        "supports_hub_edge_authority_routes": "true" if route_positive else "unclear",
        "supports_suppressor_hub_candidates": "true" if suppressor_count >= 1 else "unclear",
        "supports_shared_core_plus_frame_routes": (
            "true" if global_count >= 1 and frame_specific_count >= 2 else "unclear"
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def short_node_row(row: dict[str, Any], frame_name: str | None = None) -> str:
    if frame_name:
        drops = row["per_frame"][frame_name]
        return (
            f"| `{row['node']}` | `{fmt(row['total_degree'])}` | `{row['classification']}` "
            f"| `{fmt(drops['target_frame_accuracy_drop'])}` "
            f"| `{fmt(drops['target_frame_refraction_drop'])}` "
            f"| `{fmt(drops['target_frame_active_group_influence_drop'])}` "
            f"| `{fmt(drops['target_frame_inactive_group_influence_rise'])}` "
            f"| `{fmt(row['frame_specificity_score'])}` |"
        )
    return (
        f"| `{row['node']}` | `{fmt(row['total_degree'])}` | `{row['classification']}` "
        f"| `{row['best_frame']}` | `{fmt(row['overall_drop']['authority_switch_drop'])}` "
        f"| `{fmt(row['overall_drop']['refraction_drop'])}` "
        f"| `{fmt(row['max_inactive_group_influence_rise'])}` "
        f"| `{fmt(row['frame_specificity_score'])}` |"
    )


def write_report(summary: dict[str, Any], path: Path) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Frame-Specific Hub Circuit Validation",
        "",
        "## Goal",
        "",
        "Test whether load-bearing hubs act as global traffic bottlenecks, frame-specific authority routes, or suppressor/gate candidates in the existing `latent_refraction` toy probe.",
        "",
        "No new semantics, architecture, FlyWire biology, or wave/pointer mechanisms are added.",
        "",
        "## Completed Configs",
        "",
        "| Topology | Runs | Accuracy | Authority | Refraction | Global Hubs | Frame-Specific Nodes | Suppressor Candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topology, item in agg.items():
        frame_specific = sum(
            item["classification_counts"].get(label, {}).get("mean") or 0.0
            for label in ("danger_specific", "environment_specific", "visibility_specific")
        )
        lines.append(
            f"| `{topology}` | `{item['run_count']}` "
            f"| `{fmt(item['baseline_accuracy']['mean'])}` "
            f"| `{fmt(item['baseline_authority_switch_score']['mean'])}` "
            f"| `{fmt(item['baseline_refraction_index_final']['mean'])}` "
            f"| `{fmt(item['classification_counts'].get('global_hub', {}).get('mean'))}` "
            f"| `{fmt(frame_specific)}` "
            f"| `{fmt(item['suppressor_candidate_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
    ])
    if summary["verdict"]["supports_global_hub_bottlenecks"] == "true":
        lines.append("- Some ablated hubs hurt multiple frames, supporting global bottleneck/integrator candidates.")
    if summary["verdict"]["supports_frame_specific_hubs"] == "true":
        lines.append("- Some nodes are more frame-selective, supporting frame-specific authority route candidates.")
    if summary["verdict"]["supports_hub_edge_authority_routes"] == "true":
        lines.append("- Edge-count matched hub-route ablations beat random edge controls in at least one route group.")
    if summary["verdict"]["supports_suppressor_hub_candidates"] == "true":
        lines.append("- Some ablations raise inactive-group influence, marking suppressor/gate candidates rather than pure amplifiers.")
    lines.append("- Treat all positives as toy circuit candidates, not unique circuits or biological claims.")
    lines.extend([
        "",
        "## Hub Edge Route Ablation",
        "",
        "| Topology | Route | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop | Inactive Rise |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for topology, item in agg.items():
        for route, route_item in item["hub_edge_routes"].items():
            lines.append(
                f"| `{topology}` | `{route}` "
                f"| `{fmt(route_item['authority_switch_drop']['mean'])}` "
                f"| `{fmt(route_item['random_authority_switch_drop']['mean'])}` "
                f"| `{fmt(route_item['refraction_drop']['mean'])}` "
                f"| `{fmt(route_item['random_refraction_drop']['mean'])}` "
                f"| `{fmt(route_item['inactive_group_influence_rise']['mean'])}` |"
            )
    lines.extend([
        "",
        "## Per-Frame Node Saliency",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['topology_mode']}` seed `{run['seed']}`",
            "",
            "Top overall authority nodes:",
            "",
            "| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |",
            "|---:|---:|---|---|---:|---:|---:|---:|",
        ])
        for row in run["node_saliency"]["top10_overall"]:
            lines.append(short_node_row(row))
        lines.append("")
        for frame_name, rows in run["node_saliency"]["top10_by_frame"].items():
            lines.extend([
                f"Top nodes for `{frame_name}`:",
                "",
                "| Node | Degree | Class | Accuracy Drop | Refraction Drop | Active Drop | Inactive Rise | Specificity |",
                "|---:|---:|---|---:|---:|---:|---:|---:|",
            ])
            for row in rows[:5]:
                lines.append(short_node_row(row, frame_name))
            lines.append("")
    lines.extend([
        "## Suppression / Leakage Candidates",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['topology_mode']}` seed `{run['seed']}`",
            "",
            "| Node | Degree | Class | Best Frame | Authority Drop | Refraction Drop | Inactive Rise | Specificity |",
            "|---:|---:|---|---|---:|---:|---:|---:|",
        ])
        for row in run["node_saliency"]["suppressor_candidates_top10"]:
            lines.append(short_node_row(row))
        if not run["node_saliency"]["suppressor_candidates_top10"]:
            lines.append("| `none` | `null` | `unclear` | `null` | `null` | `null` | `null` | `null` |")
        lines.append("")
    lines.extend([
        "## Route Overlap Matrix",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['topology_mode']}` seed `{run['seed']}`",
            "",
            "Node top-K overlap:",
            "",
            "```json",
            json.dumps(probe.round_floats(run["route_overlap"]["node_overlap"]), indent=2),
            "```",
            "",
            "Edge proxy top-K overlap:",
            "",
            "```json",
            json.dumps(probe.round_floats(run["route_overlap"]["edge_proxy_overlap"]), indent=2),
            "```",
            "",
        ])
    lines.extend([
        "## Runtime Notes",
        "",
        f"- total runtime seconds: `{fmt(summary['runtime_seconds'])}`",
        f"- smoke mode: `{summary['config']['smoke']}`",
        f"- node ablation budget per model: `{summary['config']['max_node_ablation']}`",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, FlyWire validation, or production validation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested --device {args.device!r}, but CUDA is not available")
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for topology_mode in args.topology_modes:
        for seed in range(args.seeds):
            print(f"[frame-hub] topology={topology_mode} seed={seed}", flush=True)
            runs.append(run_config(seed=seed, topology_mode=topology_mode, args=args))
    agg = aggregate(runs)
    summary = {
        "config": {
            "experiment": "latent_refraction",
            "input_mode": "entangled",
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "topology_modes": args.topology_modes,
            "max_node_ablation": args.max_node_ablation,
            "edge_group_fraction": args.edge_group_fraction,
            "random_edge_controls": args.random_edge_controls,
            "top_k_fractions": args.top_k_fractions,
            "smoke": args.smoke,
        },
        "aggregate": agg,
        "verdict": verdict(agg),
        "runs": runs,
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    summary = probe.round_floats(summary)
    json_path = args.out_dir / "frame_specific_hub_circuit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, REPORT_PATH)
    print(json.dumps({
        "verdict": summary["verdict"],
        "aggregate": summary["aggregate"],
        "json": str(json_path),
        "report": str(REPORT_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
