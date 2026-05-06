#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

import run_context_cancellation_probe as probe  # noqa: E402


DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "authority-circuit-dissection"
REPORT_PATH = ROOT / "docs" / "research" / "AUTHORITY_CIRCUIT_DISSECTION.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budgeted node/edge/motif dissection for the existing authority-switch toy probes."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("latent_refraction", "multi_aspect_token_refraction"),
        default=["latent_refraction", "multi_aspect_token_refraction"],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--latent-hidden", type=int, default=64)
    parser.add_argument("--multi-hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--latent-epochs", type=int, default=200)
    parser.add_argument("--multi-epochs", type=int, default=220)
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
        choices=("random_sparse", "hub_degree_preserving_random", "hub_rich"),
        default=["random_sparse"],
        help="hub_rich is automatically skipped for multi_aspect unless --include-hub-rich-multi is set.",
    )
    parser.add_argument("--include-hub-rich-multi", action="store_true")
    parser.add_argument("--max-node-ablation", type=int, default=24)
    parser.add_argument("--random-node-controls", type=int, default=3)
    parser.add_argument("--edge-group-fraction", type=float, default=0.05)
    parser.add_argument("--minimal-fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="Tiny end-to-end run for validation.")
    args = parser.parse_args()

    if args.smoke:
        args.experiments = ["latent_refraction"]
        args.seeds = 1
        args.latent_hidden = 16
        args.multi_hidden = 16
        args.latent_epochs = 2
        args.multi_epochs = 2
        args.train_size = 96
        args.test_size = 48
        args.batch_size = 32
        args.topology_modes = ["random_sparse"]
        args.max_node_ablation = 4
        args.random_node_controls = 1
        args.minimal_fractions = [0.10, 0.40]
    return args


def make_probe_args(args: argparse.Namespace, *, hidden: int, epochs: int, topology_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        hidden=hidden,
        steps=args.steps,
        epochs=epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        lr=args.lr,
        sparse_density=args.sparse_density,
        topology_mode=topology_mode,
        flywire_graphml=probe.DEFAULT_FLYWIRE_GRAPHML,
        holdout_fraction=args.holdout_fraction,
        active_value=args.active_value,
        embed_scale=args.embed_scale,
        embedding_mode="learned",
        resonance_mode="none",
        frame_scale=args.frame_scale,
        nuisance_scale=1.05,
        opponent_strength=args.opponent_strength,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        ridge=args.ridge,
        random_label_control=False,
        device=args.device,
    )


def train_refraction_model(
    *,
    experiment: str,
    seed: int,
    args: argparse.Namespace,
    topology_mode: str,
) -> dict[str, Any]:
    if experiment == "latent_refraction":
        hidden = args.latent_hidden
        epochs = args.latent_epochs
        dataset_seed = 4_001
        frame_seed = 620_011
    elif experiment == "multi_aspect_token_refraction":
        hidden = args.multi_hidden
        epochs = args.multi_epochs
        dataset_seed = 4_001
        frame_seed = 620_011
    else:
        raise ValueError(f"unsupported experiment: {experiment}")

    probe_args = make_probe_args(args, hidden=hidden, epochs=epochs, topology_mode=topology_mode)
    schema = probe.build_schema(hidden)
    train_combos, heldout_combos = probe.split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = probe.build_embeddings(
        schema=schema,
        input_mode="entangled",
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode="learned",
        resonance_mode="none",
    )
    frame_embeddings = probe.build_named_frame_embeddings(
        probe.MULTI_ASPECT_FRAMES if experiment == "multi_aspect_token_refraction" else probe.TASK_FRAMES,
        hidden,
        seed + frame_seed,
        args.frame_scale,
    )

    if experiment == "latent_refraction":
        train = probe.make_refraction_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + dataset_seed,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
        )
        test = probe.make_refraction_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + dataset_seed + 1,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
        )
    else:
        train = probe.make_multi_aspect_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + dataset_seed,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
        )
        test = probe.make_multi_aspect_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + dataset_seed + 1,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
        )

    model = probe.train_model(train=train, hidden=hidden, args=probe_args, seed=seed + 51)
    return {
        "experiment": experiment,
        "seed": seed,
        "topology_mode": topology_mode,
        "hidden": hidden,
        "epochs": epochs,
        "schema": schema,
        "args": probe_args,
        "train": train,
        "test": test,
        "model": model,
    }


def final_mean(curves: dict[str, list[float]]) -> float | None:
    values = [float(curve[-1]) for curve in curves.values() if curve]
    return float(np.mean(values)) if values else None


def summarize_model(model: probe.RecurrentClassifier, test: probe.RefractionDataBundle, args: argparse.Namespace, *, seed: int) -> dict[str, Any]:
    pred = probe.refraction_prediction_summary(model=model, bundle=test, args=args)
    influence = probe.run_refraction_influence(model=model, test=test, args=args, seed=seed)
    refraction_curve = influence.get("mean_refraction_index_by_step", [])
    per_frame_refraction = {
        frame: (curve[-1] if curve else None)
        for frame, curve in influence.get("refraction_index_by_step", {}).items()
    }
    return {
        "accuracy": pred["accuracy"],
        "accuracy_by_frame": pred["accuracy_by_frame"],
        "authority_switch_score": influence.get("authority_switch_score"),
        "refraction_index_final": refraction_curve[-1] if refraction_curve else None,
        "active_group_influence": final_mean(influence.get("active_core_influence_by_step", {})),
        "inactive_group_influence": final_mean(influence.get("inactive_group_influence_by_step", {})),
        "per_frame_refraction_index_final": per_frame_refraction,
    }


def drop_metrics(baseline: dict[str, Any], ablated: dict[str, Any]) -> dict[str, Any]:
    out = {
        "accuracy_drop": baseline["accuracy"] - ablated["accuracy"],
        "authority_switch_drop": (baseline.get("authority_switch_score") or 0.0) - (ablated.get("authority_switch_score") or 0.0),
        "refraction_drop": (baseline.get("refraction_index_final") or 0.0) - (ablated.get("refraction_index_final") or 0.0),
        "active_influence_drop": (baseline.get("active_group_influence") or 0.0) - (ablated.get("active_group_influence") or 0.0),
        "inactive_influence_delta": (ablated.get("inactive_group_influence") or 0.0) - (baseline.get("inactive_group_influence") or 0.0),
    }
    per_frame = {}
    for frame, base_value in baseline.get("per_frame_refraction_index_final", {}).items():
        per_frame[frame] = (base_value or 0.0) - (ablated.get("per_frame_refraction_index_final", {}).get(frame) or 0.0)
    out["per_frame_refraction_drop"] = per_frame
    return out


def clone_ablate_nodes(model: probe.RecurrentClassifier, nodes: list[int]) -> probe.RecurrentClassifier:
    clone = copy.deepcopy(model)
    with torch.no_grad():
        node_t = torch.tensor(nodes, dtype=torch.long, device=clone.recurrent.device)
        clone.recurrent.index_fill_(0, node_t, 0.0)
        clone.recurrent.index_fill_(1, node_t, 0.0)
        clone.threshold.index_fill_(0, node_t, 0.0)
        clone.head.weight[:, node_t] = 0.0
    return clone


def clone_keep_nodes(model: probe.RecurrentClassifier, keep_nodes: list[int]) -> probe.RecurrentClassifier:
    hidden = model.recurrent.shape[0]
    drop = sorted(set(range(hidden)) - set(keep_nodes))
    return clone_ablate_nodes(model, drop)


def clone_zero_edges(model: probe.RecurrentClassifier, edges: list[tuple[int, int]]) -> probe.RecurrentClassifier:
    clone = copy.deepcopy(model)
    if not edges:
        return clone
    with torch.no_grad():
        targets = torch.tensor([edge[0] for edge in edges], dtype=torch.long, device=clone.recurrent.device)
        sources = torch.tensor([edge[1] for edge in edges], dtype=torch.long, device=clone.recurrent.device)
        clone.recurrent[targets, sources] = 0.0
    return clone


def clone_keep_edges(model: probe.RecurrentClassifier, keep_edges: list[tuple[int, int]]) -> probe.RecurrentClassifier:
    clone = copy.deepcopy(model)
    keep = set(keep_edges)
    mask = model.mask.detach().cpu().numpy() > 0
    all_edges = [(int(target), int(source)) for target, source in np.argwhere(mask) if int(target) != int(source)]
    drop = [edge for edge in all_edges if edge not in keep]
    return clone_zero_edges(clone, drop)


def node_proxy_scores(model: probe.RecurrentClassifier) -> np.ndarray:
    mask = model.mask.detach().cpu().numpy().astype(bool)
    recurrent = np.abs(model.recurrent.detach().cpu().numpy()) * mask
    head = np.abs(model.head.weight.detach().cpu().numpy()).sum(axis=0)
    return recurrent.sum(axis=0) + recurrent.sum(axis=1) + head


def degree_stats(model: probe.RecurrentClassifier) -> dict[str, np.ndarray]:
    mask = model.mask.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    row_degree = mask.sum(axis=1)
    col_degree = mask.sum(axis=0)
    return {
        "row_degree": row_degree,
        "col_degree": col_degree,
        "total_degree": row_degree + col_degree,
    }


def candidate_nodes(model: probe.RecurrentClassifier, max_count: int, seed: int) -> list[int]:
    hidden = model.recurrent.shape[0]
    if max_count >= hidden:
        return list(range(hidden))
    rng = np.random.default_rng(seed)
    proxy_order = list(np.argsort(-node_proxy_scores(model)))
    take_top = max_count // 2
    chosen = set(int(idx) for idx in proxy_order[:take_top])
    remaining = [idx for idx in range(hidden) if idx not in chosen]
    random_take = max_count - len(chosen)
    if random_take > 0 and remaining:
        chosen.update(int(idx) for idx in rng.choice(remaining, size=min(random_take, len(remaining)), replace=False))
    return sorted(chosen)


def edge_proxy_data(model: probe.RecurrentClassifier) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float]]:
    mask = model.mask.detach().cpu().numpy() > 0
    recurrent = np.abs(model.recurrent.detach().cpu().numpy())
    edges = [(int(target), int(source)) for target, source in np.argwhere(mask) if int(target) != int(source)]
    proxy = {edge: float(recurrent[edge[0], edge[1]]) for edge in edges}
    return edges, proxy


def random_nodes(hidden: int, count: int, rng: np.random.Generator) -> list[int]:
    return [int(idx) for idx in rng.choice(hidden, size=min(count, hidden), replace=False)]


def run_node_ablation(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    seed: int,
    max_nodes: int,
) -> dict[str, Any]:
    rows = []
    for node in candidate_nodes(model, max_nodes, seed):
        ablated = summarize_model(clone_ablate_nodes(model, [node]), test, args, seed=seed + 10_000 + node)
        rows.append({
            "node": int(node),
            **drop_metrics(baseline, ablated),
        })
    rows.sort(key=lambda item: item["authority_switch_drop"], reverse=True)
    return {
        "tested_node_count": len(rows),
        "top10": rows[:10],
        "all_tested": rows,
    }


def hub_vs_random_ablation(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    seed: int,
    random_controls: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    hidden = model.recurrent.shape[0]
    total_degree = degree_stats(model)["total_degree"]
    ranked = [int(idx) for idx in np.argsort(-total_degree)]
    out: dict[str, Any] = {}
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(round(hidden * fraction)))
        hubs = ranked[:count]
        hub_summary = summarize_model(clone_ablate_nodes(model, hubs), test, args, seed=seed + int(fraction * 1000))
        random_drops = []
        for control_idx in range(random_controls):
            nodes = random_nodes(hidden, count, rng)
            random_summary = summarize_model(
                clone_ablate_nodes(model, nodes),
                test,
                args,
                seed=seed + 20_000 + control_idx + count,
            )
            random_drops.append(drop_metrics(baseline, random_summary))
        out[f"top_{int(fraction * 100)}pct"] = {
            "node_count": count,
            "hub_nodes": hubs,
            "hub_drop": drop_metrics(baseline, hub_summary),
            "random_drop_mean": mean_drop_dict(random_drops),
        }
    return out


def mean_drop_dict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    out = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    frame_keys = rows[0].get("per_frame_refraction_drop", {}).keys()
    out["per_frame_refraction_drop"] = {
        frame: float(np.mean([row.get("per_frame_refraction_drop", {}).get(frame, 0.0) for row in rows]))
        for frame in frame_keys
    }
    return out


def edge_groups(model: probe.RecurrentClassifier, seed: int, fraction: float) -> dict[str, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    edges, proxy = edge_proxy_data(model)
    if not edges:
        return {}
    count = max(1, int(round(len(edges) * fraction)))
    total_degree = degree_stats(model)["total_degree"]
    hub_count = max(1, int(round(model.recurrent.shape[0] * 0.10)))
    hubs = set(int(idx) for idx in np.argsort(-total_degree)[:hub_count])
    top_edges = sorted(edges, key=lambda edge: proxy[edge], reverse=True)[:count]
    hub_incoming = [edge for edge in edges if edge[0] in hubs][:count]
    hub_outgoing = [edge for edge in edges if edge[1] in hubs][:count]
    reciprocal = [edge for edge in edges if (edge[1], edge[0]) in proxy][:count]
    random_same = [edges[int(idx)] for idx in rng.choice(len(edges), size=min(count, len(edges)), replace=False)]
    return {
        "top_proxy_edges": top_edges,
        "hub_incoming_edges": hub_incoming,
        "hub_outgoing_edges": hub_outgoing,
        "reciprocal_pair_edges": reciprocal,
        "random_same_count_edges": random_same,
    }


def run_edge_group_ablation(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    seed: int,
    fraction: float,
) -> dict[str, Any]:
    out = {}
    for name, edges in edge_groups(model, seed, fraction).items():
        summary = summarize_model(clone_zero_edges(model, edges), test, args, seed=seed + len(edges) + 30_000)
        out[name] = {
            "edge_count": len(edges),
            "drop": drop_metrics(baseline, summary),
        }
    return out


def saliency_rank(model: probe.RecurrentClassifier, node_ablation: dict[str, Any]) -> list[int]:
    hidden = model.recurrent.shape[0]
    scored = {int(row["node"]): float(row["authority_switch_drop"]) for row in node_ablation.get("all_tested", [])}
    proxy = node_proxy_scores(model)
    return sorted(range(hidden), key=lambda idx: (scored.get(idx, -1.0e9), proxy[idx]), reverse=True)


def run_minimal_circuit(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    node_ablation: dict[str, Any],
    seed: int,
    fractions: list[float],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    hidden = model.recurrent.shape[0]
    node_rank = saliency_rank(model, node_ablation)
    edges, proxy = edge_proxy_data(model)
    edge_rank = sorted(edges, key=lambda edge: proxy[edge], reverse=True)
    out: dict[str, Any] = {}
    for fraction in fractions:
        node_count = max(1, int(round(hidden * fraction)))
        edge_count = max(1, int(round(len(edges) * fraction))) if edges else 0
        top_nodes = node_rank[:node_count]
        random_keep_nodes = random_nodes(hidden, node_count, rng)
        top_node_summary = summarize_model(clone_keep_nodes(model, top_nodes), test, args, seed=seed + node_count + 40_000)
        random_node_summary = summarize_model(
            clone_keep_nodes(model, random_keep_nodes),
            test,
            args,
            seed=seed + node_count + 41_000,
        )
        top_edges = edge_rank[:edge_count]
        random_edges = [edges[int(idx)] for idx in rng.choice(len(edges), size=min(edge_count, len(edges)), replace=False)] if edges else []
        top_edge_summary = summarize_model(clone_keep_edges(model, top_edges), test, args, seed=seed + edge_count + 42_000)
        random_edge_summary = summarize_model(clone_keep_edges(model, random_edges), test, args, seed=seed + edge_count + 43_000)
        out[f"{int(fraction * 100)}pct"] = {
            "node_count": node_count,
            "edge_count": edge_count,
            "top_nodes_retained": retention_metrics(baseline, top_node_summary),
            "random_nodes_retained": retention_metrics(baseline, random_node_summary),
            "top_edges_retained": retention_metrics(baseline, top_edge_summary),
            "random_edges_retained": retention_metrics(baseline, random_edge_summary),
        }
    return out


def run_saliency_group_ablation(
    *,
    model: probe.RecurrentClassifier,
    test: probe.RefractionDataBundle,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    node_ablation: dict[str, Any],
    seed: int,
    random_controls: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    hidden = model.recurrent.shape[0]
    ranked = saliency_rank(model, node_ablation)
    out: dict[str, Any] = {}
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(round(hidden * fraction)))
        nodes = ranked[:count]
        saliency_summary = summarize_model(clone_ablate_nodes(model, nodes), test, args, seed=seed + count + 50_000)
        random_drops = []
        for control_idx in range(random_controls):
            control_nodes = random_nodes(hidden, count, rng)
            random_summary = summarize_model(
                clone_ablate_nodes(model, control_nodes),
                test,
                args,
                seed=seed + 51_000 + control_idx + count,
            )
            random_drops.append(drop_metrics(baseline, random_summary))
        out[f"top_{int(fraction * 100)}pct"] = {
            "node_count": count,
            "saliency_nodes": nodes,
            "saliency_drop": drop_metrics(baseline, saliency_summary),
            "random_drop_mean": mean_drop_dict(random_drops),
        }
    return out


def retention_metrics(baseline: dict[str, Any], retained: dict[str, Any]) -> dict[str, Any]:
    def positive_retention(value: float | None, base: float | None) -> float | None:
        if value is None or base is None or base <= 1.0e-9:
            return None
        return value / base

    return {
        "accuracy_retention": positive_retention(retained["accuracy"], baseline["accuracy"]),
        "authority_switch_retention": positive_retention(
            retained.get("authority_switch_score"),
            baseline.get("authority_switch_score"),
        ),
        "refraction_retention": positive_retention(
            retained.get("refraction_index_final"),
            baseline.get("refraction_index_final"),
        ),
        "accuracy": retained["accuracy"],
        "authority_switch_score": retained.get("authority_switch_score"),
        "refraction_index_final": retained.get("refraction_index_final"),
    }


def frame_specificity(node_ablation: dict[str, Any]) -> dict[str, Any]:
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in node_ablation.get("all_tested", []):
        for frame, drop in row.get("per_frame_refraction_drop", {}).items():
            by_frame.setdefault(frame, []).append({"node": row["node"], "target_drop": drop, "row": row})
    out: dict[str, Any] = {}
    scores = []
    for frame, rows in by_frame.items():
        rows.sort(key=lambda item: item["target_drop"], reverse=True)
        if not rows:
            continue
        best = rows[0]
        other_drops = [
            value
            for other_frame, value in best["row"].get("per_frame_refraction_drop", {}).items()
            if other_frame != frame
        ]
        specificity = float(best["target_drop"] - np.mean(other_drops)) if other_drops else float(best["target_drop"])
        scores.append(specificity)
        out[frame] = {
            "top_node": int(best["node"]),
            "target_refraction_drop": float(best["target_drop"]),
            "mean_non_target_refraction_drop": float(np.mean(other_drops)) if other_drops else None,
            "frame_specificity_score": specificity,
        }
    return {
        "by_frame": out,
        "mean_frame_specificity_score": float(np.mean(scores)) if scores else None,
    }


def run_dissection_config(
    *,
    experiment: str,
    seed: int,
    topology_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    artifact = train_refraction_model(experiment=experiment, seed=seed, args=args, topology_mode=topology_mode)
    model = artifact["model"]
    test = artifact["test"]
    probe_args = artifact["args"]
    baseline = summarize_model(model, test, probe_args, seed=seed + 60_000)
    node_ablation = run_node_ablation(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        seed=seed + 61_000,
        max_nodes=args.max_node_ablation,
    )
    hub_ablation = hub_vs_random_ablation(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        seed=seed + 62_000,
        random_controls=args.random_node_controls,
    )
    edge_ablation = run_edge_group_ablation(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        seed=seed + 63_000,
        fraction=args.edge_group_fraction,
    )
    minimal = run_minimal_circuit(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        node_ablation=node_ablation,
        seed=seed + 64_000,
        fractions=args.minimal_fractions,
    )
    saliency_groups = run_saliency_group_ablation(
        model=model,
        test=test,
        args=probe_args,
        baseline=baseline,
        node_ablation=node_ablation,
        seed=seed + 65_000,
        random_controls=args.random_node_controls,
    )
    frame_spec = frame_specificity(node_ablation)
    return {
        "experiment": experiment,
        "seed": seed,
        "topology_mode": topology_mode,
        "hidden": artifact["hidden"],
        "epochs": artifact["epochs"],
        "runtime_seconds": time.time() - started,
        "baseline": baseline,
        "topology": getattr(model, "topology_stats", {}),
        "node_ablation": node_ablation,
        "hub_ablation_vs_random": hub_ablation,
        "saliency_group_ablation_vs_random": saliency_groups,
        "edge_group_ablation": edge_ablation,
        "minimal_circuit_survival": minimal,
        "frame_specificity": frame_spec,
    }


def aggregate_numeric(values: list[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(numeric)), "std": float(np.std(numeric))}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = f"{run['experiment']}::{run['topology_mode']}"
        grouped.setdefault(key, []).append(run)

    summaries = {}
    for key, rows in grouped.items():
        top_node_drops = [
            row["node_ablation"]["top10"][0]["authority_switch_drop"]
            for row in rows
            if row["node_ablation"].get("top10")
        ]
        hub_scores = []
        random_scores = []
        saliency5_scores = []
        saliency5_random_scores = []
        saliency10_scores = []
        saliency10_random_scores = []
        for row in rows:
            top10 = row["hub_ablation_vs_random"].get("top_10pct", {})
            if top10:
                hub_scores.append(top10["hub_drop"]["authority_switch_drop"])
                random_scores.append(top10["random_drop_mean"].get("authority_switch_drop"))
            saliency5 = row.get("saliency_group_ablation_vs_random", {}).get("top_5pct", {})
            if saliency5:
                saliency5_scores.append(saliency5["saliency_drop"]["authority_switch_drop"])
                saliency5_random_scores.append(saliency5["random_drop_mean"].get("authority_switch_drop"))
            saliency10 = row.get("saliency_group_ablation_vs_random", {}).get("top_10pct", {})
            if saliency10:
                saliency10_scores.append(saliency10["saliency_drop"]["authority_switch_drop"])
                saliency10_random_scores.append(saliency10["random_drop_mean"].get("authority_switch_drop"))
        minimal_top = []
        minimal_random = []
        minimal_top_edge = []
        minimal_random_edge = []
        for row in rows:
            item = row["minimal_circuit_survival"].get("10pct")
            if item:
                minimal_top.append(item["top_nodes_retained"]["authority_switch_retention"])
                minimal_random.append(item["random_nodes_retained"]["authority_switch_retention"])
                minimal_top_edge.append(item["top_edges_retained"]["authority_switch_retention"])
                minimal_random_edge.append(item["random_edges_retained"]["authority_switch_retention"])
        reciprocal_scores = []
        reciprocal_random_scores = []
        for row in rows:
            reciprocal = row["edge_group_ablation"].get("reciprocal_pair_edges", {})
            random_edges = row["edge_group_ablation"].get("random_same_count_edges", {})
            if reciprocal and random_edges:
                reciprocal_scores.append(reciprocal["drop"]["authority_switch_drop"])
                reciprocal_random_scores.append(random_edges["drop"]["authority_switch_drop"])
        summaries[key] = {
            "run_count": len(rows),
            "baseline_accuracy": aggregate_numeric([row["baseline"]["accuracy"] for row in rows]),
            "baseline_authority_switch_score": aggregate_numeric([row["baseline"].get("authority_switch_score") for row in rows]),
            "baseline_refraction_index_final": aggregate_numeric([row["baseline"].get("refraction_index_final") for row in rows]),
            "top_node_authority_drop": aggregate_numeric(top_node_drops),
            "top5pct_saliency_authority_drop": aggregate_numeric(saliency5_scores),
            "random5pct_node_authority_drop": aggregate_numeric(saliency5_random_scores),
            "top10pct_saliency_authority_drop": aggregate_numeric(saliency10_scores),
            "random10pct_saliency_control_authority_drop": aggregate_numeric(saliency10_random_scores),
            "top10pct_hub_authority_drop": aggregate_numeric(hub_scores),
            "random10pct_node_authority_drop": aggregate_numeric(random_scores),
            "top10pct_node_authority_retention": aggregate_numeric(minimal_top),
            "random10pct_node_authority_retention": aggregate_numeric(minimal_random),
            "top10pct_edge_authority_retention": aggregate_numeric(minimal_top_edge),
            "random10pct_edge_authority_retention": aggregate_numeric(minimal_random_edge),
            "reciprocal_edge_authority_drop": aggregate_numeric(reciprocal_scores),
            "random_edge_authority_drop": aggregate_numeric(reciprocal_random_scores),
            "frame_specificity_score": aggregate_numeric([
                row["frame_specificity"].get("mean_frame_specificity_score")
                for row in rows
            ]),
        }
    return summaries


def topology_summary(aggregate: dict[str, Any], experiment: str, topology: str) -> dict[str, Any] | None:
    return aggregate.get(f"{experiment}::{topology}")


def metric_mean(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary:
        return None
    value = summary.get(key, {}).get("mean")
    return float(value) if value is not None else None


def verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    top_node_positive = any(
        (summary["top_node_authority_drop"]["mean"] or 0.0) > 0.10
        for summary in aggregate.values()
    )
    hub_positive = any(
        (summary["top10pct_hub_authority_drop"]["mean"] or 0.0)
        > (summary["random10pct_node_authority_drop"]["mean"] or 0.0) + 0.05
        for summary in aggregate.values()
    )
    reciprocal_positive = any(
        (summary["reciprocal_edge_authority_drop"]["mean"] or 0.0)
        > (summary["random_edge_authority_drop"]["mean"] or 0.0) + 0.05
        for summary in aggregate.values()
    )
    minimal_positive = any(
        max(
            summary["top10pct_node_authority_retention"]["mean"] or 0.0,
            summary["top10pct_edge_authority_retention"]["mean"] or 0.0,
        )
        > max(
            summary["random10pct_node_authority_retention"]["mean"] or 0.0,
            summary["random10pct_edge_authority_retention"]["mean"] or 0.0,
        ) + 0.20
        for summary in aggregate.values()
    )
    frame_specific = any(
        (summary["frame_specificity_score"]["mean"] or 0.0) > 0.05
        for summary in aggregate.values()
    )
    degree_prior = "unclear"
    specific_wiring = "unclear"
    for experiment in ("latent_refraction", "multi_aspect_token_refraction"):
        random_sparse = topology_summary(aggregate, experiment, "random_sparse")
        hub = topology_summary(aggregate, experiment, "hub_rich")
        degree_random = topology_summary(aggregate, experiment, "hub_degree_preserving_random")
        if not (random_sparse and hub and degree_random):
            continue
        random_authority = metric_mean(random_sparse, "baseline_authority_switch_score") or 0.0
        hub_authority = metric_mean(hub, "baseline_authority_switch_score") or 0.0
        degree_authority = metric_mean(degree_random, "baseline_authority_switch_score") or 0.0
        random_refraction = metric_mean(random_sparse, "baseline_refraction_index_final") or 0.0
        hub_refraction = metric_mean(hub, "baseline_refraction_index_final") or 0.0
        degree_refraction = metric_mean(degree_random, "baseline_refraction_index_final") or 0.0
        degree_beats_random = degree_authority > random_authority + 0.03 and degree_refraction > random_refraction + 0.03
        degree_matches_hub = abs(degree_authority - hub_authority) <= 0.05 and abs(degree_refraction - hub_refraction) <= 0.05
        hub_beats_degree = hub_authority > degree_authority + 0.05 and hub_refraction > degree_refraction + 0.05
        if degree_beats_random or degree_matches_hub:
            degree_prior = "true"
        if hub_beats_degree:
            specific_wiring = "true"
    return {
        "supports_localized_authority_circuit": "true" if top_node_positive or minimal_positive else "unclear",
        "supports_hub_load_bearing_authority": "true" if hub_positive else "unclear",
        "supports_degree_distribution_as_circuit_prior": degree_prior,
        "supports_specific_hub_wiring": specific_wiring,
        "supports_reciprocal_motif_importance": "true" if reciprocal_positive else "unclear",
        "supports_minimal_circuit_extractability": "true" if minimal_positive else "unclear",
        "supports_frame_specific_circuit_candidates": "true" if frame_specific else "unclear",
        "notes_on_task_specificity": (
            "This is a budgeted dissection. Treat positives as candidate circuit localization, "
            "not proof of a unique circuit, until rerun with larger node/edge budgets."
        ),
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# Authority Circuit Dissection",
        "",
        "## Goal",
        "",
        "Locate load-bearing recurrent nodes, edge groups, and motif candidates responsible for decision-authority switching in existing toy probes.",
        "",
        "This report uses budgeted ablations. It does not add semantics, architecture, FlyWire biology, or wave/pointer mechanisms.",
        "",
        "## Multi-Seed Validation Summary",
        "",
        "| Experiment | Topology | Runs | Baseline Acc | Authority | Refraction | Top Node Drop | Top 5% Saliency Drop | Random 5% Drop | Hub 10% Drop | Random 10% Drop | Top 10% Retention | Random 10% Retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in aggregate.items():
        experiment, topology = key.split("::", maxsplit=1)
        lines.append(
            f"| `{experiment}` | `{topology}` | `{item['run_count']}` "
            f"| `{fmt(item['baseline_accuracy']['mean'])}` "
            f"| `{fmt(item['baseline_authority_switch_score']['mean'])}` "
            f"| `{fmt(item['baseline_refraction_index_final']['mean'])}` "
            f"| `{fmt(item['top_node_authority_drop']['mean'])}` "
            f"| `{fmt(item['top5pct_saliency_authority_drop']['mean'])}` "
            f"| `{fmt(item['random5pct_node_authority_drop']['mean'])}` "
            f"| `{fmt(item['top10pct_hub_authority_drop']['mean'])}` "
            f"| `{fmt(item['random10pct_node_authority_drop']['mean'])}` "
            f"| `{fmt(item['top10pct_node_authority_retention']['mean'])}` "
            f"| `{fmt(item['random10pct_node_authority_retention']['mean'])}` |"
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
    if summary["verdict"].get("supports_localized_authority_circuit") == "true":
        lines.append(
            "- Node and saliency-group ablations indicate that authority switching is not fully homogeneous; "
            "small recurrent subsets can be disproportionately load-bearing."
        )
    if summary["verdict"].get("supports_hub_load_bearing_authority") == "true":
        lines.append(
            "- Hub ablations hurt authority/refraction more than same-count random node ablations in the validated configs."
        )
    if summary["verdict"].get("supports_degree_distribution_as_circuit_prior") == "true":
        lines.append(
            "- The degree-preserving hub-random control performs at least as well as hub-rich here, so the useful prior currently looks more like hub/degree concentration than a specific hand-built hub wiring pattern."
        )
    if summary["verdict"].get("supports_specific_hub_wiring") != "true":
        lines.append(
            "- Specific hub wiring remains unproven; current evidence does not require the original hub-rich edge pattern."
        )
    if summary["verdict"].get("supports_minimal_circuit_extractability") != "true":
        lines.append(
            "- Minimal circuit extraction remains unclear: keeping only top-K nodes/edges does not yet preserve authority/refraction cleanly enough to call this a compact extracted circuit."
        )
    if summary["verdict"].get("supports_reciprocal_motif_importance") != "true":
        lines.append(
            "- Reciprocal motifs are not yet load-bearing beyond random edge controls."
        )
    lines.extend([
        "",
        "## Hub-Rich vs Degree-Preserving Hub",
        "",
        "| Experiment | Random Authority | Hub-Rich Authority | Degree-Preserving Authority | Random Refraction | Hub-Rich Refraction | Degree-Preserving Refraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    experiments = sorted({key.split("::", maxsplit=1)[0] for key in aggregate})
    for experiment in experiments:
        random_sparse = topology_summary(aggregate, experiment, "random_sparse")
        hub = topology_summary(aggregate, experiment, "hub_rich")
        degree_random = topology_summary(aggregate, experiment, "hub_degree_preserving_random")
        lines.append(
            f"| `{experiment}` "
            f"| `{fmt(metric_mean(random_sparse, 'baseline_authority_switch_score'))}` "
            f"| `{fmt(metric_mean(hub, 'baseline_authority_switch_score'))}` "
            f"| `{fmt(metric_mean(degree_random, 'baseline_authority_switch_score'))}` "
            f"| `{fmt(metric_mean(random_sparse, 'baseline_refraction_index_final'))}` "
            f"| `{fmt(metric_mean(hub, 'baseline_refraction_index_final'))}` "
            f"| `{fmt(metric_mean(degree_random, 'baseline_refraction_index_final'))}` |"
        )
    lines.extend([
        "",
        "## Node Ablation Top 10",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "| Node | Accuracy Drop | Authority Drop | Refraction Drop | Active Influence Drop |",
            "|---:|---:|---:|---:|---:|",
        ])
        for row in run["node_ablation"]["top10"]:
            lines.append(
                f"| `{row['node']}` | `{fmt(row['accuracy_drop'])}` | `{fmt(row['authority_switch_drop'])}` "
                f"| `{fmt(row['refraction_drop'])}` | `{fmt(row['active_influence_drop'])}` |"
            )
        lines.append("")
    lines.extend([
        "## Saliency Group Ablation",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "| Node Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, item in run.get("saliency_group_ablation_vs_random", {}).items():
            saliency_drop = item["saliency_drop"]
            random_drop = item["random_drop_mean"]
            lines.append(
                f"| `{name}` | `{item['node_count']}` | `{fmt(saliency_drop['authority_switch_drop'])}` "
                f"| `{fmt(random_drop.get('authority_switch_drop'))}` | `{fmt(saliency_drop['refraction_drop'])}` "
                f"| `{fmt(random_drop.get('refraction_drop'))}` |"
            )
        lines.append("")
    lines.extend([
        "## Hub Ablation Vs Random Nodes",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "| Hub Group | Count | Authority Drop | Random Authority Drop | Refraction Drop | Random Refraction Drop |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, item in run["hub_ablation_vs_random"].items():
            hub_drop = item["hub_drop"]
            random_drop = item["random_drop_mean"]
            lines.append(
                f"| `{name}` | `{item['node_count']}` | `{fmt(hub_drop['authority_switch_drop'])}` "
                f"| `{fmt(random_drop.get('authority_switch_drop'))}` | `{fmt(hub_drop['refraction_drop'])}` "
                f"| `{fmt(random_drop.get('refraction_drop'))}` |"
            )
        lines.append("")
    lines.extend([
        "## Edge Group Ablation",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "| Edge Group | Count | Accuracy Drop | Authority Drop | Refraction Drop |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, item in run["edge_group_ablation"].items():
            drop = item["drop"]
            lines.append(
                f"| `{name}` | `{item['edge_count']}` | `{fmt(drop['accuracy_drop'])}` "
                f"| `{fmt(drop['authority_switch_drop'])}` | `{fmt(drop['refraction_drop'])}` |"
            )
        lines.append("")
    lines.extend([
        "## Minimal Circuit Survival",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "| Retained Budget | Top Nodes Authority Retention | Random Nodes Authority Retention | Top Edges Authority Retention | Random Edges Authority Retention |",
            "|---|---:|---:|---:|---:|",
        ])
        for budget, item in run["minimal_circuit_survival"].items():
            lines.append(
                f"| `{budget}` | `{fmt(item['top_nodes_retained']['authority_switch_retention'])}` "
                f"| `{fmt(item['random_nodes_retained']['authority_switch_retention'])}` "
                f"| `{fmt(item['top_edges_retained']['authority_switch_retention'])}` "
                f"| `{fmt(item['random_edges_retained']['authority_switch_retention'])}` |"
            )
        lines.append("")
    lines.extend([
        "## Frame Specificity",
        "",
    ])
    for run in summary["runs"]:
        lines.extend([
            f"### `{run['experiment']}` seed `{run['seed']}` topology `{run['topology_mode']}`",
            "",
            "```json",
            json.dumps(probe.round_floats(run["frame_specificity"]), indent=2),
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
        "Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> int:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested --device {args.device!r}, but CUDA is not available")

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for experiment in args.experiments:
        for topology_mode in args.topology_modes:
            if experiment == "multi_aspect_token_refraction" and topology_mode == "hub_rich" and not args.include_hub_rich_multi:
                continue
            for seed in range(args.seeds):
                print(f"[authority-circuit] {experiment} topology={topology_mode} seed={seed}", flush=True)
                runs.append(
                    run_dissection_config(
                        experiment=experiment,
                        seed=seed,
                        topology_mode=topology_mode,
                        args=args,
                    )
                )

    aggregate = aggregate_runs(runs)
    summary = {
        "config": {
            "experiments": args.experiments,
            "seeds": args.seeds,
            "latent_hidden": args.latent_hidden,
            "multi_hidden": args.multi_hidden,
            "steps": args.steps,
            "latent_epochs": args.latent_epochs,
            "multi_epochs": args.multi_epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "topology_modes": args.topology_modes,
            "max_node_ablation": args.max_node_ablation,
            "random_node_controls": args.random_node_controls,
            "edge_group_fraction": args.edge_group_fraction,
            "minimal_fractions": args.minimal_fractions,
            "smoke": args.smoke,
        },
        "aggregate": aggregate,
        "verdict": verdict(aggregate),
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
    json_path = args.out_dir / "authority_circuit_dissection_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, REPORT_PATH)

    printable = {
        "verdict": summary["verdict"],
        "aggregate": summary["aggregate"],
        "json": str(json_path),
        "report": str(REPORT_PATH),
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
