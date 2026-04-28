"""Phase D9.0: latent genome decoder toy audit.

Offline-only falsification test for a deterministic genome compiler:

    z -> D(z) -> toy genome -> graph/descriptor behavior

No Rust runtime, no SAF changes, no live search. The first goal is to
catch hash-like or identity-like decoders before they can be connected to
real VRAXION search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform


DEFAULT_OUT = Path("output/phase_d9_latent_genome_toy_20260428")
DEFAULT_REPORT = Path("docs/research/PHASE_D9_0_LATENT_DECODER_TOY_AUDIT.md")
TOOL_VERSION = "d9.0-pic-toy-1"
EPS = 1e-12
VERDICT_PASS = "D9_LATENT_DECODER_TOY_PASS"
VERDICT_VALIDITY = "D9_DECODER_VALIDITY_FAIL"
VERDICT_NO_LOCALITY = "D9_DECODER_NO_LOCALITY"
VERDICT_HASHLIKE = "D9_DECODER_HASHLIKE_BEHAVIOR"
VERDICT_SCAN = "D9_TILE_SCAN_NO_SIGNAL"
VERDICT_CONTROL = "D9_CONTROL_PARITY_FAIL"
VERDICT_TOO_HEAVY = "DNP_TOO_HEAVY"
VERDICT_KILLER_BAD = "KILLER_TEST_DEGENERATE"


@dataclass
class ToyGenome:
    decoder: str
    z: np.ndarray
    root_seed: int
    H: int
    edges: list[tuple[int, int]]
    polarity: list[int]
    thr: list[int]
    channel: list[int]
    rule_trace: list[dict[str, Any]]
    validity_flag: bool
    validity_reasons: list[str]
    descriptor: np.ndarray
    edge_vector: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--killer-n", type=int, default=200)
    parser.add_argument("--full-n", type=int, default=500)
    parser.add_argument("--killer-permutations", type=int, default=499)
    parser.add_argument("--full-permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--scan-budget", type=int, default=300)
    parser.add_argument("--time-limit-sec", type=float, default=30 * 60)
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def stable_hash_int(*parts: Any, bits: int = 64) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[: bits // 8], "little", signed=False)


def stable_unit(*parts: Any) -> float:
    return stable_hash_int(*parts, bits=64) / float(2**64 - 1)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def edge_index_map(H: int) -> dict[tuple[int, int], int]:
    idx: dict[tuple[int, int], int] = {}
    k = 0
    for i in range(H):
        for j in range(H):
            if i == j:
                continue
            idx[(i, j)] = k
            k += 1
    return idx


def edges_to_vector(edges: list[tuple[int, int]], index: dict[tuple[int, int], int]) -> np.ndarray:
    vec = np.zeros(len(index), dtype=np.uint8)
    for edge in edges:
        pos = index.get(edge)
        if pos is not None:
            vec[pos] = 1
    return vec


def entropy_from_counts(counts: np.ndarray) -> float:
    vals = np.asarray(counts, dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals[vals > 0] / total
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log2(p)).sum() / math.log2(len(vals)))


def graph_descriptor(H: int, edges: list[tuple[int, int]], polarity: list[int], thr: list[int], channel: list[int]) -> np.ndarray:
    edge_count = max(1, len(edges))
    possible = H * (H - 1)
    out_deg = np.zeros(H, dtype=float)
    in_deg = np.zeros(H, dtype=float)
    recurrent = 0
    chain_like = 0
    same_channel = 0
    mirror = 0
    modules = 4
    same_module = 0
    for i, j in edges:
        out_deg[i] += 1
        in_deg[j] += 1
        recurrent += int(j < i)
        chain_like += int(j == (i + 1) % H)
        same_channel += int(channel[i] == channel[j])
        mirror += int((i + j) == H - 1)
        same_module += int((i * modules) // H == (j * modules) // H)
    deg = in_deg + out_deg
    return np.asarray(
        [
            len(edges) / possible,
            float(np.mean(np.asarray(polarity) < 0)),
            float(np.mean(thr) / 255.0),
            float(np.std(thr) / 128.0),
            recurrent / edge_count,
            chain_like / edge_count,
            same_channel / edge_count,
            mirror / edge_count,
            same_module / edge_count,
            entropy_from_counts(deg + 1e-9),
            float(np.std(out_deg) / max(1.0, np.mean(out_deg) + EPS)),
            float(len(set(channel)) / max(1, H)),
        ],
        dtype=float,
    )


def validate_graph(H: int, edges: list[tuple[int, int]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not edges:
        reasons.append("empty_edge_set")
    invalid = [(i, j) for i, j in edges if i == j or i < 0 or j < 0 or i >= H or j >= H]
    if invalid:
        reasons.append("invalid_edge")
    deg = np.zeros(H, dtype=int)
    for i, j in edges:
        if 0 <= i < H and 0 <= j < H and i != j:
            deg[i] += 1
            deg[j] += 1
    if np.any(deg == 0):
        reasons.append("orphan_node")
    return len(reasons) == 0, reasons


def z_to_u(z: np.ndarray) -> np.ndarray:
    u = np.asarray(sigmoid(z), dtype=float)
    if len(u) < 12:
        raise ValueError("PIC requires 12 z dimensions")
    return u


def decode_pic(z: np.ndarray, root_seed: int, H: int, index: dict[tuple[int, int], int]) -> ToyGenome:
    u = z_to_u(z)
    min_density = 0.03 if H <= 64 else 0.01
    density = min_density + 0.17 * u[0]
    possible = H * (H - 1)
    target_edges = int(round(density * possible))
    target_edges = max(H, min(possible, target_edges))
    chain_budget = int(round((0.02 + 0.16 * u[3]) * H))
    coords = (np.arange(H, dtype=float) + 0.5) / H
    modules = 2 + int(round(6 * u[9]))
    phase = 2 * math.pi * u[11]
    irregular_amp = 0.10 * u[10]

    polarity_scores = np.asarray([0.85 * math.sin((i + 1) * (0.7 + u[2]) + phase) + 0.15 * stable_unit("pol", i) for i in range(H)])
    inhib_count = int(round((0.05 + 0.45 * u[1]) * H))
    inhib_nodes = set(np.argsort(polarity_scores)[:inhib_count].tolist())
    polarity = [-1 if i in inhib_nodes else 1 for i in range(H)]

    thr_center = 64 + int(round(128 * u[4]))
    thr_spread = 4 + 56 * u[5]
    thr = []
    for i in range(H):
        val = thr_center + thr_spread * math.sin((i + 1) * (0.3 + u[6]) + phase)
        thr.append(int(max(0, min(255, round(val)))))
    channels = 2 + int(round(6 * u[6]))
    channel = [int((i * channels) // H) for i in range(H)]

    base_edges: set[tuple[int, int]] = {(i, (i + 1) % H) for i in range(H)}
    for i in range(chain_budget):
        base_edges.add((i % H, (i + 2) % H))

    edges = set(base_edges)

    def ordered_candidates(rule_id: str, score_fn) -> list[tuple[int, int]]:
        rows: list[tuple[float, int, int]] = []
        for i in range(H):
            mi = (i * modules) // H
            for j in range(H):
                if i == j or (i, j) in base_edges:
                    continue
                mj = (j * modules) // H
                score = float(score_fn(i, j, mi, mj)) + 1e-6 * stable_unit("pic-order", rule_id, i, j)
                rows.append((-score, i, j))
        rows.sort()
        return [(i, j) for _, i, j in rows]

    group_specs = [
        ("local_chain", 0.10 + 1.20 * u[3], lambda i, j, mi, mj: -abs(j - i - 1)),
        ("recurrent", 0.10 + 1.20 * u[7], lambda i, j, mi, mj: 1.0 if j < i else 0.0),
        ("mirror", 0.10 + 1.10 * u[8], lambda i, j, mi, mj: -abs((i + j) - (H - 1))),
        ("module", 0.10 + 1.25 * u[9], lambda i, j, mi, mj: 1.0 if mi == mj else 0.0),
        ("channel", 0.10 + 0.90 * u[6], lambda i, j, mi, mj: 1.0 if channel[i] == channel[j] else 0.0),
        ("inhib_bridge", 0.10 + 0.80 * u[1], lambda i, j, mi, mj: 1.0 if polarity[i] < 0 and polarity[j] > 0 else 0.0),
        ("threshold_flow", 0.10 + 0.70 * u[4], lambda i, j, mi, mj: (thr[i] - thr[j]) / 255.0),
        ("spread", 0.10 + 0.70 * u[5], lambda i, j, mi, mj: abs(i - j) / max(1, H - 1)),
        ("irregular", 0.10 + 0.80 * u[10], lambda i, j, mi, mj: math.sin((i + 1) * (0.11 + 0.7 * u[10]) + (j + 1) * (0.17 + 0.5 * u[11]) + phase)),
        ("phase", 0.10 + 0.60 * u[11], lambda i, j, mi, mj: math.cos((i - j) * (0.13 + u[11]) + phase)),
    ]
    weights = np.asarray([spec[1] for spec in group_specs], dtype=float)
    weights = weights / max(EPS, float(weights.sum()))
    remaining_target = max(0, target_edges - len(edges))
    quotas = np.floor(weights * remaining_target).astype(int)
    shortfall = remaining_target - int(quotas.sum())
    for pos in np.argsort(-(weights * remaining_target - quotas))[:shortfall]:
        quotas[int(pos)] += 1

    for quota, (rule_id, _, score_fn) in zip(quotas, group_specs):
        if len(edges) >= target_edges:
            break
        added = 0
        for i, j in ordered_candidates(rule_id, score_fn):
            if len(edges) >= target_edges or added >= int(quota):
                break
            if (i, j) not in edges:
                edges.add((i, j))
                added += 1

    if len(edges) < target_edges:
        for i, j in ordered_candidates("fill", lambda i, j, mi, mj: stable_unit("fill", i, j)):
            if len(edges) >= target_edges:
                break
            edges.add((i, j))

    edge_list = sorted(edges)
    valid, reasons = validate_graph(H, edge_list)
    desc = graph_descriptor(H, edge_list, polarity, thr, channel)
    trace = [
        {"rule_id": "density", "site": [0], "score": float(density), "subseed": "closed-form"},
        {"rule_id": "polarity", "site": [inhib_count], "score": float(u[1]), "subseed": "closed-form"},
        {"rule_id": "chain", "site": [chain_budget], "score": float(u[3]), "subseed": "closed-form"},
        {"rule_id": "edge_groups", "site": [target_edges], "score": float(target_edges), "subseed": "quota-groups"},
        {"rule_id": "threshold", "site": [thr_center], "score": float(u[4]), "subseed": "closed-form"},
        {"rule_id": "channel", "site": [channels], "score": float(u[6]), "subseed": "closed-form"},
    ]
    return ToyGenome("PIC", z.copy(), root_seed, H, edge_list, polarity, thr, channel, trace, valid, reasons, desc, edges_to_vector(edge_list, index))


def rng_from_z(name: str, z: np.ndarray, root_seed: int) -> np.random.Generator:
    zb = np.asarray(z, dtype=np.float64).tobytes()
    h = hashlib.sha256(name.encode("utf-8") + root_seed.to_bytes(8, "little", signed=False) + zb).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)


def build_from_edge_scores(
    decoder: str,
    z: np.ndarray,
    root_seed: int,
    H: int,
    index: dict[tuple[int, int], int],
    scores: np.ndarray,
    target_edges: int,
    rule_ids: list[str],
) -> ToyGenome:
    base_edges = {(i, (i + 1) % H) for i in range(H)}
    pairs = [(i, j) for i in range(H) for j in range(H) if i != j and (i, j) not in base_edges]
    order = np.argsort(-scores[: len(pairs)])
    edges = set(base_edges)
    for pos in order:
        if len(edges) >= target_edges:
            break
        edges.add(pairs[int(pos)])
    edge_list = sorted(edges)
    rng = rng_from_z(decoder + "-attrs", z, root_seed)
    polarity = np.where(rng.random(H) < 0.25, -1, 1).astype(int).tolist()
    thr = rng.integers(32, 224, size=H).astype(int).tolist()
    channel = rng.integers(0, 6, size=H).astype(int).tolist()
    valid, reasons = validate_graph(H, edge_list)
    desc = graph_descriptor(H, edge_list, polarity, thr, channel)
    trace = [{"rule_id": rid, "site": [k], "score": float(k), "subseed": decoder} for k, rid in enumerate(rule_ids)]
    return ToyGenome(decoder, z.copy(), root_seed, H, edge_list, polarity, thr, channel, trace, valid, reasons, desc, edges_to_vector(edge_list, index))


def decode_hash(z: np.ndarray, root_seed: int, H: int, index: dict[tuple[int, int], int]) -> ToyGenome:
    rng = rng_from_z("NCI_RANDOM_HASH_DECODER", z, root_seed)
    possible = H * (H - 1)
    target_edges = int(round((0.03 + 0.17 * rng.random()) * possible))
    target_edges = max(H, min(possible, target_edges))
    scores = rng.random(possible)
    return build_from_edge_scores("NCI_RANDOM_HASH_DECODER", z, root_seed, H, index, scores, target_edges, ["hash_graph"])


def decode_nonlocal(z: np.ndarray, root_seed: int, H: int, index: dict[tuple[int, int], int]) -> ToyGenome:
    genome = decode_pic(np.sort(z), root_seed, H, index)
    genome.decoder = "NCI_NONLOCAL_DECODER"
    return genome


def decode_advr(z: np.ndarray, root_seed: int, H: int, index: dict[tuple[int, int], int]) -> ToyGenome:
    q = np.clip(np.floor((sigmoid(z) * 255.0)), 0, 255).astype(np.uint8)
    h = hashlib.sha256(q.tobytes() + root_seed.to_bytes(8, "little", signed=False)).digest()
    possible = H * (H - 1)
    scores = np.zeros(possible, dtype=float)
    for k in range(possible):
        qsig = q[k % len(q)] / 255.0
        noise = h[k % len(h)] / 255.0
        scores[k] = 0.72 * qsig + 0.28 * noise + 1e-9 * stable_unit("advr", k)
    target_edges = max(H, min(possible, int(round((0.04 + 0.12 * (q[0] / 255.0)) * possible))))
    return build_from_edge_scores("D_advr", z, root_seed, H, index, scores, target_edges, ["quantized_z_prefix", "sha256_noise"])


def decode_all(z_values: np.ndarray, decoder: str, root_seed: int, H: int, index: dict[tuple[int, int], int]) -> list[ToyGenome]:
    out: list[ToyGenome] = []
    for i, z in enumerate(z_values):
        seed = stable_hash_int(root_seed, decoder, i)
        if decoder == "PIC":
            out.append(decode_pic(z, seed, H, index))
        elif decoder == "NCI_RANDOM_HASH_DECODER":
            out.append(decode_hash(z, seed, H, index))
        elif decoder == "D_advr":
            out.append(decode_advr(z, seed, H, index))
        elif decoder == "NCI_NONLOCAL_DECODER":
            out.append(decode_nonlocal(z, seed, H, index))
        else:
            raise ValueError(f"unknown decoder: {decoder}")
    return out


def genome_matrix(genomes: list[ToyGenome]) -> np.ndarray:
    return np.vstack([g.edge_vector for g in genomes]).astype(np.uint8)


def descriptor_matrix(genomes: list[ToyGenome]) -> np.ndarray:
    return np.vstack([g.descriptor for g in genomes]).astype(float)


def pairwise_graph_dist(genomes: list[ToyGenome]) -> np.ndarray:
    return pdist(genome_matrix(genomes), metric="hamming")


def pairwise_desc_dist(genomes: list[ToyGenome]) -> np.ndarray:
    mat = descriptor_matrix(genomes)
    sd = np.std(mat, axis=0)
    sd[sd <= EPS] = 1.0
    return pdist((mat - np.mean(mat, axis=0)) / sd, metric="euclidean")


def pearson_fast(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return 0.0
    a = x[ok] - np.mean(x[ok])
    b = y[ok] - np.mean(y[ok])
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def mantel_spearman(x_dist: np.ndarray, y_dist: np.ndarray, permutations: int, rng: np.random.Generator) -> dict[str, float]:
    if len(x_dist) != len(y_dist):
        raise ValueError("distance vectors must have equal length")
    x_rank = stats.rankdata(x_dist)
    y_rank = stats.rankdata(y_dist)
    r_obs = pearson_fast(x_rank, y_rank)
    n = squareform(x_dist).shape[0]
    y_square = squareform(y_rank)
    more = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        yp = squareform(y_square[perm][:, perm], checks=False)
        if abs(pearson_fast(x_rank, yp)) >= abs(r_obs):
            more += 1
    p = (more + 1) / (permutations + 1)
    return {"r": float(r_obs), "p": float(p)}


def bootstrap_r(
    z_values: np.ndarray,
    decoder: str,
    root_seed: int,
    H: int,
    index: dict[tuple[int, int], int],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if n_boot <= 0:
        return math.nan, math.nan
    vals: list[float] = []
    n = len(z_values)
    sample_n = min(n, 80)
    for b in range(n_boot):
        ids = rng.integers(0, n, size=sample_n)
        zs = z_values[ids]
        genomes = decode_all(zs, decoder, stable_hash_int(root_seed, "boot", decoder, b), H, index)
        vals.append(pearson_fast(stats.rankdata(pdist(zs, metric="euclidean")), stats.rankdata(pairwise_graph_dist(genomes))))
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def nontriviality(genomes: list[ToyGenome], advr: list[ToyGenome] | None = None) -> dict[str, Any]:
    mat = genome_matrix(genomes).astype(float)
    occupancy = np.mean(mat, axis=0)
    edge_entropy = float(np.mean(-(occupancy * np.log2(occupancy + EPS) + (1 - occupancy) * np.log2(1 - occupancy + EPS))))
    edge_entropy = max(0.0, min(1.0, edge_entropy))
    deg_ent = []
    rule_counts = []
    for g in genomes:
        deg = np.zeros(g.H, dtype=float)
        for i, j in g.edges:
            deg[i] += 1
            deg[j] += 1
        deg_ent.append(entropy_from_counts(deg + 1e-9))
        rule_counts.append(len(set(str(r.get("rule_id")) for r in g.rule_trace)) / 6.0)
    result: dict[str, Any] = {
        "edge_structural_entropy": edge_entropy,
        "degree_distribution_entropy": float(np.mean(deg_ent)),
        "rule_trace_diversity": float(np.mean(rule_counts)),
    }
    if advr is not None:
        real = genome_matrix(genomes)
        adv = genome_matrix(advr)
        graph_sep = float(np.mean(np.mean(real != adv, axis=1)))
        desc_sep = float(np.mean(np.linalg.norm(descriptor_matrix(genomes) - descriptor_matrix(advr), axis=1)))
        result["advr_graph_separation"] = graph_sep
        result["advr_descriptor_separation"] = desc_sep
    gates = {
        "edge_structural_entropy": result["edge_structural_entropy"] >= 0.02,
        "degree_distribution_entropy": result["degree_distribution_entropy"] >= 0.25,
        "rule_trace_diversity": result["rule_trace_diversity"] >= 0.30,
        "advr_graph_separation": result.get("advr_graph_separation", 1.0) >= 0.03,
        "advr_descriptor_separation": result.get("advr_descriptor_separation", 1.0) >= 0.10,
    }
    result["gates"] = gates
    result["pass"] = bool(all(gates.values()))
    return result


def validity_rate(genomes: list[ToyGenome]) -> float:
    if not genomes:
        return 0.0
    return float(np.mean([g.validity_flag for g in genomes]))


def killer_microtest(args: argparse.Namespace, index: dict[tuple[int, int], int], start: float) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    z_values = rng.normal(0, 1, size=(args.killer_n, 12))
    decoders = ["PIC", "NCI_RANDOM_HASH_DECODER", "D_advr"]
    genomes = {d: decode_all(z_values, d, args.seed, args.H, index) for d in decoders}
    z_dist = pdist(z_values, metric="euclidean")
    metrics: dict[str, Any] = {}
    for d in decoders:
        gdist = pairwise_graph_dist(genomes[d])
        metrics[d] = mantel_spearman(z_dist, gdist, args.killer_permutations, np.random.default_rng(stable_hash_int(args.seed, "mantel", d)))
        lo, hi = bootstrap_r(z_values, d, args.seed, args.H, index, args.bootstrap, np.random.default_rng(stable_hash_int(args.seed, "boot", d)))
        metrics[d]["bootstrap_ci"] = [lo, hi]
        metrics[d]["valid_network_rate"] = validity_rate(genomes[d])
    nt = nontriviality(genomes["PIC"], genomes["D_advr"])

    verdict = None
    reasons: list[str] = []
    if time.time() - start > args.time_limit_sec:
        verdict = VERDICT_TOO_HEAVY
        reasons.append("time_limit_exceeded")
    elif metrics["PIC"]["valid_network_rate"] < 0.99 or not nt["pass"]:
        verdict = VERDICT_VALIDITY
        reasons.append("validity_or_nontriviality_failed")
    elif metrics["PIC"]["r"] < 0.30 or metrics["PIC"]["p"] >= 0.05:
        verdict = VERDICT_NO_LOCALITY
        reasons.append("pic_locality_below_absolute_gate")
    elif metrics["PIC"]["r"] < metrics["NCI_RANDOM_HASH_DECODER"]["r"] + 0.20:
        verdict = VERDICT_HASHLIKE
        reasons.append("pic_did_not_clear_hash_margin")
    else:
        advr_passes = (
            metrics["D_advr"]["valid_network_rate"] >= 0.99
            and metrics["D_advr"]["r"] >= 0.30
            and metrics["D_advr"]["p"] < 0.05
            and metrics["D_advr"]["r"] >= metrics["NCI_RANDOM_HASH_DECODER"]["r"] + 0.20
        )
        if advr_passes:
            verdict = VERDICT_KILLER_BAD
            reasons.append("adversarial_decoder_passed_killer")
        else:
            verdict = None
            reasons.append("killer_passed")
    return {
        "verdict_if_failed": verdict,
        "reasons": reasons,
        "z_values": z_values,
        "genomes": genomes,
        "metrics": metrics,
        "nontriviality": nt,
        "elapsed_sec": time.time() - start,
    }


def toy_fitness(landscape: str, z_values: np.ndarray, genomes: list[ToyGenome], seed: int) -> tuple[np.ndarray, np.ndarray]:
    desc = descriptor_matrix(genomes)
    zsig = sigmoid(z_values)
    if landscape == "smooth":
        target = np.asarray([0.11, 0.25, 0.55, 0.30, 0.45, 0.07, 0.60, 0.05, 0.55, 0.85, 0.55, 0.10])
        dist = np.linalg.norm((desc - target) / np.asarray([0.06, 0.2, 0.3, 0.4, 0.3, 0.1, 0.3, 0.1, 0.3, 0.2, 0.4, 0.2]), axis=1)
        fit = np.exp(-0.5 * dist)
        labels = (fit >= np.quantile(fit, 0.80)).astype(int)
    elif landscape == "deceptive":
        bits = (desc[:, [0, 1, 4, 6, 8, 9, 10, 11]] > np.median(desc[:, [0, 1, 4, 6, 8, 9, 10, 11]], axis=0)).astype(int)
        scores = []
        for row in bits:
            total = 0.0
            for k in range(0, len(row), 4):
                ones = int(row[k : k + 4].sum())
                total += 1.0 if ones == 0 else (0.80 * ones / 4.0)
            scores.append(total / 2.0)
        fit = np.asarray(scores)
        labels = (fit >= np.quantile(fit, 0.80)).astype(int)
    elif landscape == "multi_basin":
        centers = np.asarray(
            [
                [0.08, 0.20, 0.60, 0.20, 0.25, 0.10, 0.60, 0.03, 0.70, 0.80, 0.30, 0.12],
                [0.16, 0.40, 0.45, 0.35, 0.65, 0.05, 0.35, 0.05, 0.45, 0.70, 0.70, 0.10],
                [0.12, 0.30, 0.55, 0.25, 0.45, 0.15, 0.80, 0.08, 0.55, 0.90, 0.45, 0.16],
            ]
        )
        dists = np.stack([np.linalg.norm(desc - c, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        fit = np.exp(-2.0 * np.min(dists, axis=1))
    elif landscape == "needle":
        target = np.asarray([0.12, 0.33, 0.50, 0.22, 0.58, 0.08, 0.50, 0.04, 0.62, 0.86, 0.38, 0.12])
        dist = np.linalg.norm(desc - target, axis=1)
        fit = (dist <= np.quantile(dist, 0.02)).astype(float)
        labels = fit.astype(int)
    elif landscape == "random_control":
        rng = np.random.default_rng(stable_hash_int(seed, "random_control"))
        fit = rng.random(len(genomes))
        labels = (fit >= np.quantile(fit, 0.80)).astype(int)
    else:
        raise ValueError(landscape)
    return fit.astype(float), labels.astype(int)


def safe_corr_distance_to_fitness(z_values: np.ndarray, fitness: np.ndarray) -> float:
    best = int(np.argmax(fitness))
    dist = np.linalg.norm(z_values - z_values[best], axis=1)
    val = stats.spearmanr(dist, -fitness).correlation
    return float(0.0 if not np.isfinite(val) else val)


def progressive_scan_ratio(z_values: np.ndarray, fitness: np.ndarray, budget: int, seed: int) -> dict[str, float]:
    n = len(fitness)
    budget = min(budget, n)
    rng = np.random.default_rng(seed)
    random_scores = []
    prog_scores = []
    for trial in range(40):
        order = rng.permutation(n)
        random_scores.append(float(np.max(fitness[order[:budget]])))
        init = list(order[:20])
        seen = set(init)
        best_idx = init[int(np.argmax(fitness[init]))]
        while len(seen) < budget:
            remaining = np.asarray([i for i in range(n) if i not in seen], dtype=int)
            d = np.linalg.norm(z_values[remaining] - z_values[best_idx], axis=1)
            # Mostly local exploitation, with deterministic escape every fifth pick.
            if len(seen) % 5 == 0:
                pick = int(remaining[rng.integers(0, len(remaining))])
            else:
                pick = int(remaining[int(np.argmin(d))])
            seen.add(pick)
            if fitness[pick] > fitness[best_idx]:
                best_idx = pick
        prog_scores.append(float(np.max(fitness[list(seen)])))
    rand = float(np.mean(random_scores))
    prog = float(np.mean(prog_scores))
    ratio = prog / max(rand, EPS)
    return {"random_best_mean": rand, "progressive_best_mean": prog, "ratio": ratio}


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_a, labels_b))
    except Exception:
        return 0.0


def full_suite(args: argparse.Namespace, index: dict[tuple[int, int], int], start: float) -> dict[str, Any]:
    rng = np.random.default_rng(stable_hash_int(args.seed, "full"))
    z_values = rng.normal(0, 1, size=(args.full_n, 12))
    decoders = ["PIC", "NCI_RANDOM_HASH_DECODER", "NCI_NONLOCAL_DECODER", "D_advr"]
    genomes = {d: decode_all(z_values, d, args.seed, args.H, index) for d in decoders}
    z_dist = pdist(z_values, metric="euclidean")
    decoder_metrics: dict[str, Any] = {}
    for d in decoders:
        decoder_metrics[d] = {
            "valid_network_rate": validity_rate(genomes[d]),
            "z_genome": mantel_spearman(z_dist, pairwise_graph_dist(genomes[d]), args.full_permutations, np.random.default_rng(stable_hash_int(args.seed, "full-mantel-g", d))),
            "z_behavior": mantel_spearman(z_dist, pairwise_desc_dist(genomes[d]), args.full_permutations, np.random.default_rng(stable_hash_int(args.seed, "full-mantel-b", d))),
            "nontriviality": nontriviality(genomes[d], genomes["D_advr"] if d == "PIC" else None),
        }
    landscapes = ["smooth", "deceptive", "multi_basin", "needle", "random_control"]
    rows: list[dict[str, Any]] = []
    landscape_metrics: dict[str, Any] = {}
    for decoder in decoders:
        landscape_metrics[decoder] = {}
        for landscape in landscapes:
            fit, labels = toy_fitness(landscape, z_values, genomes[decoder], args.seed)
            scan = progressive_scan_ratio(z_values, fit, args.scan_budget, stable_hash_int(args.seed, "scan", decoder, landscape))
            fdc = safe_corr_distance_to_fitness(z_values, fit)
            pred_labels = pd.qcut(pd.Series(fit).rank(method="first"), q=min(5, len(np.unique(fit))), labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
            ari = adjusted_rand_index(labels, pred_labels)
            hit_rate = float(np.mean(fit >= np.quantile(fit, 0.80)))
            landscape_metrics[decoder][landscape] = {
                "fitness_mean": float(np.mean(fit)),
                "fitness_max": float(np.max(fit)),
                "fitness_distance_corr": fdc,
                "target_hit_rate_top20": hit_rate,
                "progressive_scan": scan,
                "ari_vs_oracle": ari,
            }
            for i, value in enumerate(fit):
                if decoder == "PIC":
                    rows.append(
                        {
                            "z_id": i,
                            "landscape": landscape,
                            "fitness": float(value),
                            "basin_label": int(labels[i]),
                            "edge_density": float(genomes[decoder][i].descriptor[0]),
                            "inhibitory_fraction": float(genomes[decoder][i].descriptor[1]),
                            "degree_entropy": float(genomes[decoder][i].descriptor[9]),
                        }
                    )
    verdict = VERDICT_PASS
    reasons: list[str] = []
    if time.time() - start > args.time_limit_sec:
        verdict = VERDICT_TOO_HEAVY
        reasons.append("time_limit_exceeded")
    elif decoder_metrics["PIC"]["valid_network_rate"] < 0.99:
        verdict = VERDICT_VALIDITY
        reasons.append("valid_network_rate_below_0_99")
    elif decoder_metrics["PIC"]["z_genome"]["r"] < max(0.30, decoder_metrics["NCI_RANDOM_HASH_DECODER"]["z_genome"]["r"] + 0.20):
        verdict = VERDICT_NO_LOCALITY
        reasons.append("z_genome_locality_gate_failed")
    elif decoder_metrics["PIC"]["z_behavior"]["r"] < decoder_metrics["NCI_RANDOM_HASH_DECODER"]["z_behavior"]["r"] + 0.10:
        verdict = VERDICT_HASHLIKE
        reasons.append("z_behavior_hashlike_gate_failed")
    elif landscape_metrics["PIC"]["smooth"]["progressive_scan"]["ratio"] < 1.5 and landscape_metrics["PIC"]["multi_basin"]["progressive_scan"]["ratio"] < 1.5:
        verdict = VERDICT_SCAN
        reasons.append("progressive_scan_no_gain")
    elif decoder_metrics["NCI_NONLOCAL_DECODER"]["z_genome"]["r"] >= decoder_metrics["PIC"]["z_genome"]["r"] - 0.02:
        verdict = VERDICT_CONTROL
        reasons.append("nonlocal_decoder_control_parity")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "z_values": z_values,
        "genomes": genomes,
        "decoder_metrics": decoder_metrics,
        "landscape_metrics": landscape_metrics,
        "landscape_rows": rows,
        "elapsed_sec": time.time() - start,
    }


def genome_to_json(g: ToyGenome, z_id: int) -> dict[str, Any]:
    return {
        "version": "d9.0-pic-1" if g.decoder == "PIC" else f"d9.0-control-{g.decoder}",
        "decoder": g.decoder,
        "z_id": z_id,
        "z_logged": [float(x) for x in g.z],
        "root_seed": f"{g.root_seed:016x}",
        "rule_trace": g.rule_trace,
        "genome": {
            "H": g.H,
            "edges": [[int(i), int(j)] for i, j in g.edges],
            "polarity": [int(x) for x in g.polarity],
            "thr": [int(x) for x in g.thr],
            "channel": [int(x) for x in g.channel],
        },
        "validity_flag": bool(g.validity_flag),
        "validity_reasons": g.validity_reasons,
        "descriptor": [float(x) for x in g.descriptor],
    }


def write_outputs(args: argparse.Namespace, killer: dict[str, Any], suite: dict[str, Any] | None, verdict: str, reasons: list[str], start: float) -> dict[str, Any]:
    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    provenance_path = args.out / "genome_provenance.jsonl"
    landscape_path = args.out / "landscape_results.csv"
    controls_path = args.out / "control_baselines.json"
    summary_path = analysis / "summary.json"

    provenance_genomes = suite["genomes"] if suite is not None else killer["genomes"]
    with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
        for decoder, genomes in provenance_genomes.items():
            limit = len(genomes) if decoder == "PIC" else min(len(genomes), 50)
            for i, genome in enumerate(genomes[:limit]):
                handle.write(json.dumps(json_ready(genome_to_json(genome, i)), sort_keys=True) + "\n")

    if suite is not None:
        pd.DataFrame(suite["landscape_rows"]).to_csv(landscape_path, index=False)
        controls = {
            "decoder_metrics": suite["decoder_metrics"],
            "landscape_metrics": suite["landscape_metrics"],
        }
    else:
        with landscape_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"])
            writer.writeheader()
            writer.writerow({"status": "skipped_after_killer"})
        controls = {"killer_metrics": killer["metrics"]}

    controls_path.write_text(json.dumps(json_ready(controls), indent=2, sort_keys=True), encoding="utf-8")
    numpy_version = np.__version__
    reproducibility = hashlib.sha256(f"{args.seed}|{TOOL_VERSION}|{numpy_version}".encode("utf-8")).hexdigest()[:16]
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "tool_version": TOOL_VERSION,
        "root_seed": args.seed,
        "H": args.H,
        "numpy_version": numpy_version,
        "reproducibility_hash": reproducibility,
        "elapsed_sec": time.time() - start,
        "killer": {
            "metrics": killer["metrics"],
            "nontriviality": killer["nontriviality"],
            "reasons": killer["reasons"],
            "elapsed_sec": killer["elapsed_sec"],
        },
        "full_suite": None
        if suite is None
        else {
            "decoder_metrics": suite["decoder_metrics"],
            "landscape_metrics": suite["landscape_metrics"],
            "reasons": suite["reasons"],
            "elapsed_sec": suite["elapsed_sec"],
        },
        "outputs": {
            "landscape_results": str(landscape_path),
            "genome_provenance": str(provenance_path),
            "control_baselines": str(controls_path),
            "summary": str(summary_path),
            "report": str(args.report),
        },
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    killer = summary["killer"]
    full = summary["full_suite"]
    lines: list[str] = [
        "# Phase D9.0 Latent Genome Decoder Toy Audit",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "This is an offline-only Python toy audit. It does not modify Rust, SAF, K(H), operator schedules, archive steering, or live search.",
        "",
        "## Configuration",
        "",
        f"- tool_version: `{summary['tool_version']}`",
        f"- root_seed: `{summary['root_seed']}`",
        f"- H: `{summary['H']}`",
        f"- elapsed_sec: `{summary['elapsed_sec']:.2f}`",
        f"- reproducibility_hash: `{summary['reproducibility_hash']}`",
        "",
        "## Killer Microtest",
        "",
        "| Decoder | valid_rate | z->genome r | p | bootstrap CI |",
        "|---|---:|---:|---:|---|",
    ]
    for decoder, metrics in killer["metrics"].items():
        ci = metrics.get("bootstrap_ci", [None, None])
        lines.append(
            f"| `{decoder}` | {metrics['valid_network_rate']:.3f} | {metrics['r']:.3f} | {metrics['p']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    lines += [
        "",
        "### Non-Triviality",
        "",
        "```json",
        json.dumps(json_ready(killer["nontriviality"]), indent=2, sort_keys=True),
        "```",
        "",
    ]
    lines += [
        "### Gate Outcome",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- reasons: `{', '.join(summary['reasons']) if summary['reasons'] else 'none'}`",
    ]
    if full is None:
        lines.append("- full_suite: skipped because the fail-fast killer gate did not pass")
    lines.append("")
    if full is not None:
        lines += [
            "## Full Suite Decoder Metrics",
            "",
            "| Decoder | valid_rate | z->genome r | z->behavior r |",
            "|---|---:|---:|---:|",
        ]
        for decoder, metrics in full["decoder_metrics"].items():
            lines.append(
                f"| `{decoder}` | {metrics['valid_network_rate']:.3f} | {metrics['z_genome']['r']:.3f} | {metrics['z_behavior']['r']:.3f} |"
            )
        lines += [
            "",
            "## Progressive Scan Ratios",
            "",
            "| Decoder | Landscape | Ratio | Progressive best | Random best |",
            "|---|---|---:|---:|---:|",
        ]
        for decoder, landscapes in full["landscape_metrics"].items():
            for name, metrics in landscapes.items():
                scan = metrics["progressive_scan"]
                lines.append(
                    f"| `{decoder}` | `{name}` | {scan['ratio']:.3f} | {scan['progressive_best_mean']:.3f} | {scan['random_best_mean']:.3f} |"
                )
    lines += [
        "",
        "## Output Files",
        "",
        f"- `{summary['outputs']['summary']}`",
        f"- `{summary['outputs']['landscape_results']}`",
        f"- `{summary['outputs']['genome_provenance']}`",
        f"- `{summary['outputs']['control_baselines']}`",
        "",
        "## Interpretation",
        "",
        "A pass means only that the toy compiler survived the configured negative controls. It is not a live VRAXION improvement claim.",
        "A fail verdict is expected to be useful: it identifies which D9 assumption died first.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start = time.time()
    index = edge_index_map(args.H)
    killer = killer_microtest(args, index, start)
    if killer["verdict_if_failed"] is not None:
        verdict = killer["verdict_if_failed"]
        reasons = list(killer["reasons"])
        suite = None
    elif args.skip_full:
        verdict = VERDICT_PASS
        reasons = ["killer_passed_full_suite_skipped"]
        suite = None
    else:
        suite = full_suite(args, index, start)
        verdict = suite["verdict"]
        reasons = list(suite["reasons"])
    summary = write_outputs(args, killer, suite, verdict, reasons, start)
    write_report(args.report, summary)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "summary": summary["outputs"]["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
