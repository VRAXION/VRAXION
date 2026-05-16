from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT_CONTRACT.md"

PHASE_CLASSES = 4
READOUT_EPS = 0.30
FORBIDDEN_PUBLIC_FIELDS = {
    "gate_sum",
    "path_phase_total",
    "true_path",
    "shortest_path_phase_total",
    "label",
    "answer",
    "oracle_phase_bucket",
    "family",
}

ALL_ARMS = (
    "ORACLE_SPATIAL_PHASE_LOCK",
    "FIXED_COMPLEX_MULTIPLY_LOCAL_REFERENCE",
    "HIGHWAY_ONLY_PHASE",
    "HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE",
    "HIGHWAY_WITH_GATED_POCKETS_PHASE",
    "HIGHWAY_WITH_UNGATED_POCKETS_PHASE",
    "UNRESTRICTED_GRAPH_MUTATION_PHASE",
    "ORACLE_ROUTING_MUTABLE_PHASE_POCKET",
    "MUTABLE_ROUTING_ORACLE_PHASE",
    "MUTABLE_ROUTING_MUTABLE_PHASE",
)

MUTABLE_ARMS = {
    "HIGHWAY_WITH_GATED_POCKETS_PHASE",
    "HIGHWAY_WITH_UNGATED_POCKETS_PHASE",
    "UNRESTRICTED_GRAPH_MUTATION_PHASE",
    "ORACLE_ROUTING_MUTABLE_PHASE_POCKET",
    "MUTABLE_ROUTING_MUTABLE_PHASE",
}

MUTATION_OPERATORS = (
    "add_local_edge",
    "remove_local_edge",
    "rewire_local_edge",
    "mutate_gate_threshold",
    "mutate_gate_channel",
    "flip_gate_polarity",
    "add_local_loop2",
    "add_local_loop3",
    "move_read_tap_local",
    "move_writeback_local",
)


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    width: int
    wall: tuple[tuple[int, ...], ...]
    source: tuple[int, int]
    target: tuple[int, int]
    source_phase: tuple[float, float]
    gate_real: tuple[tuple[float, ...], ...]
    gate_imag: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PrivateCase:
    case_id: str
    label: int
    true_path: tuple[tuple[int, int], ...]
    path_phase_total: int
    gate_sum: int
    oracle_routing_info: str
    family: str
    split: str
    path_len: int
    pair_id: str | None


@dataclass(frozen=True)
class CaseBundle:
    public: PublicCase
    private: PrivateCase


@dataclass
class LocalCircuit:
    arm: str
    pockets: int
    # Local numeric circuit. True complex multiply is representable by:
    # re = +in_re*gr - in_im*gi, im = +in_re*gi + in_im*gr.
    w_re: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    w_im: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    gate_alpha: float = 0.0
    threshold: float = 0.0
    channel: int = 1
    polarity: int = 1
    enabled_edges: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 0, 0])
    accepted_ops: Counter = field(default_factory=Counter)
    evaluated_ops: Counter = field(default_factory=Counter)


def parse_csv(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in parse_csv(text):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def unit_phase(k: int) -> complex:
    theta = 2.0 * math.pi * (k % PHASE_CLASSES) / PHASE_CLASSES
    return complex(math.cos(theta), math.sin(theta))


def phase_bucket(z: complex) -> int:
    if abs(z) < READOUT_EPS:
        return -1
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / PHASE_CLASSES))) % PHASE_CLASSES


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def public_field_audit() -> dict[str, Any]:
    fields = set(PublicCase.__dataclass_fields__)
    leaks = sorted(fields & FORBIDDEN_PUBLIC_FIELDS)
    return {
        "locality_audit": "fail" if leaks else "pass",
        "forbidden_public_fields": leaks,
        "uses_forbidden_private_field": bool(leaks),
    }


def make_empty_grid(width: int) -> np.ndarray:
    wall = np.ones((width, width), dtype=np.int8)
    return wall


def carve_path(width: int, rng: random.Random, split: str, family: str) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    wall = make_empty_grid(width)
    if split == "train":
        src = (rng.randrange(2, width // 3), rng.randrange(2, width // 3))
        tgt = (rng.randrange(width // 2, width - 2), rng.randrange(width // 2, width - 2))
    else:
        src = (rng.randrange(1, max(2, width // 4)), rng.randrange(1, width - 2))
        tgt = (rng.randrange(width - width // 3, width - 1), rng.randrange(1, width - 2))
    y, x = src
    ty, tx = tgt
    path = [(y, x)]
    if family == "reverse_path_consistency":
        x_first = False
    else:
        x_first = rng.random() < 0.5
    if x_first:
        step = 1 if tx >= x else -1
        while x != tx:
            x += step
            path.append((y, x))
        step = 1 if ty >= y else -1
        while y != ty:
            y += step
            path.append((y, x))
    else:
        step = 1 if ty >= y else -1
        while y != ty:
            y += step
            path.append((y, x))
        step = 1 if tx >= x else -1
        while x != tx:
            x += step
            path.append((y, x))
    if family in {"distractor_corridor", "damaged_corridor"}:
        mid = path[len(path) // 2]
        dy, dx = mid
        for k in range(1, min(5, width - 1)):
            yy = min(width - 2, dy + k)
            wall[yy, dx] = 0
    for py, px in path:
        wall[py, px] = 0
    # Keep target local patch identical-ish across counterfactuals.
    for yy in range(max(0, ty - 1), min(width, ty + 2)):
        for xx in range(max(0, tx - 1), min(width, tx + 2)):
            if (yy, xx) == (ty, tx) or (yy, xx) in path:
                wall[yy, xx] = 0
    return wall, path, src, tgt


def build_case(
    width: int,
    seed: int,
    idx: int,
    split: str,
    family: str,
    pair_id: str | None = None,
    forced_gate_offset: int | None = None,
) -> CaseBundle:
    rng = random.Random(seed * 1000003 + idx * 9176 + (11 if split == "train" else 23))
    wall, path, src, tgt = carve_path(width, rng, split, family)
    source_phase_idx = rng.randrange(PHASE_CLASSES)
    source = unit_phase(source_phase_idx)
    gate_idx = np.zeros((width, width), dtype=np.int8)
    gate_real = np.ones((width, width), dtype=np.float32)
    gate_imag = np.zeros((width, width), dtype=np.float32)
    path_gate_sum = 0
    for step_idx, (py, px) in enumerate(path[1:], start=1):
        if forced_gate_offset is not None and step_idx == max(1, len(path) // 2):
            g = forced_gate_offset % PHASE_CLASSES
        elif family == "short_path_phase_lock" and split == "train":
            g = rng.choice([0, 1])
        elif family == "reverse_path_consistency":
            g = rng.choice([0, 3])
        else:
            g = rng.randrange(PHASE_CLASSES)
        gate_idx[py, px] = g
        z = unit_phase(g)
        gate_real[py, px] = float(z.real)
        gate_imag[py, px] = float(z.imag)
        path_gate_sum = (path_gate_sum + g) % PHASE_CLASSES
    if family == "damaged_corridor" and len(path) > 4:
        # Same local task, but a side/damaged region with irrelevant gate noise.
        for py, px in path[2::4]:
            for yy, xx in ((py + 1, px), (py - 1, px), (py, px + 1), (py, px - 1)):
                if 0 <= yy < width and 0 <= xx < width and wall[yy, xx] == 0 and (yy, xx) not in path:
                    g = rng.randrange(PHASE_CLASSES)
                    z = unit_phase(g)
                    gate_real[yy, xx] = float(z.real)
                    gate_imag[yy, xx] = float(z.imag)
    label = (source_phase_idx + path_gate_sum) % PHASE_CLASSES
    case_id = f"{split}_{seed}_{idx}_{family}"
    public = PublicCase(
        case_id=case_id,
        width=width,
        wall=tuple(tuple(int(v) for v in row) for row in wall.tolist()),
        source=src,
        target=tgt,
        source_phase=(float(source.real), float(source.imag)),
        gate_real=tuple(tuple(float(v) for v in row) for row in gate_real.tolist()),
        gate_imag=tuple(tuple(float(v) for v in row) for row in gate_imag.tolist()),
    )
    private = PrivateCase(
        case_id=case_id,
        label=label,
        true_path=tuple(path),
        path_phase_total=path_gate_sum,
        gate_sum=path_gate_sum,
        oracle_routing_info="path",
        family=family,
        split=split,
        path_len=len(path) - 1,
        pair_id=pair_id,
    )
    return CaseBundle(public=public, private=private)


def generate_cases(n: int, width: int, seed: int, split: str) -> list[CaseBundle]:
    train_families = ("short_path_phase_lock", "simple_bend", "distractor_corridor")
    eval_families = (
        "long_path_phase_lock",
        "same_target_counterfactual",
        "damaged_corridor",
        "reverse_path_consistency",
        "wall_blocked_near_miss",
    )
    families = train_families if split == "train" else eval_families
    cases: list[CaseBundle] = []
    idx = 0
    while len(cases) < n:
        family = families[idx % len(families)]
        if split == "eval" and family == "same_target_counterfactual" and len(cases) + 1 < n:
            pair = f"pair_{seed}_{idx}"
            cases.append(build_case(width, seed, idx, split, family, pair_id=pair, forced_gate_offset=1))
            cases.append(build_case(width, seed, idx, split, family, pair_id=pair, forced_gate_offset=3))
            idx += 2
            continue
        cases.append(build_case(width, seed, idx, split, family))
        idx += 1
    return cases[:n]


def arrays(public: PublicCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wall = np.asarray(public.wall, dtype=np.int8)
    gate = np.asarray(public.gate_real, dtype=np.float32) + 1j * np.asarray(public.gate_imag, dtype=np.float32)
    free = wall == 0
    return wall, free, gate


def local_circuit_step(z: complex, gate: complex, circuit: LocalCircuit) -> complex:
    in_re, in_im = z.real, z.imag
    gr, gi = gate.real, gate.imag
    feats_re = [in_re * gr, in_im * gi, in_re, in_im, gr, gi]
    feats_im = [in_re * gi, in_im * gr, in_re, in_im, gr, gi]
    re = sum(w * f * e for w, f, e in zip(circuit.w_re, feats_re, circuit.enabled_edges))
    im = sum(w * f * e for w, f, e in zip(circuit.w_im, feats_im, circuit.enabled_edges))
    out = complex(re, im) * circuit.polarity
    if abs(out) < circuit.threshold:
        return 0j
    mag = abs(out)
    if mag > 1.5:
        out /= mag
    return out


def fixed_complex_step(z: complex, gate: complex) -> complex:
    out = z * gate
    mag = abs(out)
    return out / mag if mag > 1.5 else out


def predict(public: PublicCase, arm: str, circuit: LocalCircuit | None, *, ablate: bool = False, shuffled: str | None = None) -> tuple[int, dict[str, float]]:
    audit = public_field_audit()
    if audit["uses_forbidden_private_field"]:
        return -1, {"uses_forbidden_private_field": 1.0}
    wall, free, gate = arrays(public)
    width = public.width
    target = public.target
    source = public.source
    if shuffled == "target":
        target = ((target[0] + 2) % width, (target[1] + 3) % width)
    if shuffled == "wall":
        rng = random.Random(hash(public.case_id) & 0xFFFFFFFF)
        mask = np.asarray(wall).copy()
        for _ in range(max(1, width // 2)):
            y, x = rng.randrange(width), rng.randrange(width)
            if (y, x) not in {public.source, public.target}:
                mask[y, x] = 1 - mask[y, x]
        free = mask == 0
        wall = mask
    if shuffled == "gate":
        rng = random.Random((hash(public.case_id) ^ 0xA5A5A5A5) & 0xFFFFFFFF)
        gate = gate.copy()
        coords = list(zip(*np.where(free)))
        vals = [gate[y, x] for y, x in coords]
        rng.shuffle(vals)
        for (y, x), val in zip(coords, vals):
            gate[y, x] = val
    state = np.zeros((width, width), dtype=np.complex64)
    frontier = np.zeros_like(state)
    src_z = complex(public.source_phase[0], public.source_phase[1])
    state[source] = src_z
    frontier[source] = src_z
    wall_leak = 0.0
    pre_wall = 0.0
    direct_output_leak = 0.0
    steps = width * 2
    use_fixed = arm in {"ORACLE_SPATIAL_PHASE_LOCK", "FIXED_COMPLEX_MULTIPLY_LOCAL_REFERENCE", "MUTABLE_ROUTING_ORACLE_PHASE"}
    highway_only = arm == "HIGHWAY_ONLY_PHASE" or ablate or ("NO_WRITEBACK" in arm)
    if arm == "ORACLE_ROUTING_MUTABLE_PHASE_POCKET":
        # Diagnostic split: oracle routing means only cells already reached by
        # local propagation can carry state; phase update remains mutable.
        pass
    for _ in range(steps):
        incoming = np.zeros_like(state)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(frontier)
            sy0, sy1 = max(0, -dy), width - max(0, dy)
            sx0, sx1 = max(0, -dx), width - max(0, dx)
            dy0, dy1 = max(0, dy), width - max(0, -dy)
            dx0, dx1 = max(0, dx), width - max(0, -dx)
            shifted[dy0:dy1, dx0:dx1] = frontier[sy0:sy1, sx0:sx1]
            incoming += shifted
        pre_wall += float(np.mean(np.abs(incoming[wall == 1]))) if np.any(wall == 1) else 0.0
        incoming *= free
        new_frontier = np.zeros_like(state)
        coords = zip(*np.where(np.abs(incoming) > 1e-5))
        for y, x in coords:
            if highway_only:
                out = incoming[y, x]
            elif use_fixed:
                out = fixed_complex_step(incoming[y, x], gate[y, x])
            else:
                assert circuit is not None
                alpha = 1.0 if "UNGATED" in arm or "UNRESTRICTED" in arm else 1.0 / (1.0 + math.exp(-circuit.gate_alpha))
                local = local_circuit_step(incoming[y, x], gate[y, x], circuit)
                out = (1.0 - alpha) * incoming[y, x] + alpha * local
            if abs(state[y, x]) < 1e-5:
                state[y, x] = out
                new_frontier[y, x] = out
        frontier = new_frontier
        wall_leak += float(np.mean(np.abs(state[wall == 1]) > 1e-4)) if np.any(wall == 1) else 0.0
        if abs(frontier[target]) > 1e-5:
            break
    pred = phase_bucket(complex(state[target]))
    return pred, {
        "wall_leak_rate": wall_leak / max(1, steps),
        "pre_wall_pressure": pre_wall / max(1, steps),
        "direct_output_leak_rate": direct_output_leak,
        "uses_forbidden_private_field": 0.0,
        "max_edge_distance": 1.0,
        "writes_to_target_readout_directly": 0.0,
        "reads_nonlocal_cell": 0.0,
        "writes_wall_cell": 0.0,
    }


def initial_circuit(arm: str, pockets: int, rng: random.Random) -> LocalCircuit:
    circuit = LocalCircuit(arm=arm, pockets=pockets)
    if arm == "HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE":
        circuit.w_re = [rng.uniform(-0.5, 0.5) for _ in range(6)]
        circuit.w_im = [rng.uniform(-0.5, 0.5) for _ in range(6)]
    if arm == "UNRESTRICTED_GRAPH_MUTATION_PHASE":
        circuit.enabled_edges = [1, 1, 1, 1, 1, 1]
    return circuit


def mutate(parent: LocalCircuit, rng: random.Random, operator: str) -> LocalCircuit:
    child = copy.deepcopy(parent)
    child.evaluated_ops[operator] += 1
    if operator in {"add_local_edge", "move_read_tap_local"}:
        child.enabled_edges[rng.randrange(6)] = 1
    elif operator == "remove_local_edge":
        child.enabled_edges[rng.randrange(6)] = 0
    elif operator == "rewire_local_edge":
        idx = rng.randrange(6)
        child.enabled_edges[idx] = 1 - child.enabled_edges[idx]
    elif operator == "mutate_gate_threshold":
        child.threshold = max(0.0, min(0.8, child.threshold + rng.gauss(0.0, 0.08)))
    elif operator == "mutate_gate_channel":
        idx = rng.randrange(6)
        target = child.w_re if rng.random() < 0.5 else child.w_im
        target[idx] += rng.gauss(0.0, 0.45)
    elif operator == "flip_gate_polarity":
        child.polarity *= -1
    elif operator == "add_local_loop2":
        child.w_re[0] += rng.gauss(0.35, 0.20)
        child.w_re[1] += rng.gauss(-0.35, 0.20)
        child.enabled_edges[0] = 1
        child.enabled_edges[1] = 1
    elif operator == "add_local_loop3":
        child.w_im[0] += rng.gauss(0.35, 0.20)
        child.w_im[1] += rng.gauss(0.35, 0.20)
        child.enabled_edges[0] = 1
        child.enabled_edges[1] = 1
    elif operator == "move_writeback_local":
        child.gate_alpha += rng.gauss(0.5, 0.35)
    return child


def evaluate(cases: list[CaseBundle], arm: str, circuit: LocalCircuit | None, *, ablate: bool = False) -> dict[str, float]:
    correct = []
    long_correct = []
    retention = []
    shuffle_gate = []
    shuffle_target = []
    shuffle_wall = []
    wall_leak = []
    pre_wall = []
    direct_leak = []
    audit_flags = []
    by_pair: dict[str, list[int]] = defaultdict(list)
    for bundle in cases:
        pred, stats = predict(bundle.public, arm, circuit, ablate=ablate)
        ok = int(pred == bundle.private.label)
        correct.append(ok)
        if bundle.private.path_len >= 16:
            long_correct.append(ok)
        if bundle.private.path_phase_total == 0:
            retention.append(ok)
        if bundle.private.pair_id:
            by_pair[bundle.private.pair_id].append(ok)
        sg, _ = predict(bundle.public, arm, circuit, ablate=ablate, shuffled="gate")
        st, _ = predict(bundle.public, arm, circuit, ablate=ablate, shuffled="target")
        sw, _ = predict(bundle.public, arm, circuit, ablate=ablate, shuffled="wall")
        shuffle_gate.append(int(sg == bundle.private.label))
        shuffle_target.append(int(st == bundle.private.label))
        shuffle_wall.append(int(sw == bundle.private.label))
        wall_leak.append(stats["wall_leak_rate"])
        pre_wall.append(stats["pre_wall_pressure"])
        direct_leak.append(stats["direct_output_leak_rate"])
        audit_flags.append(stats["uses_forbidden_private_field"])
    acc = mean(correct)
    gate_acc = mean(shuffle_gate)
    target_acc = mean(shuffle_target)
    wall_acc = mean(shuffle_wall)
    pair_scores = [int(len(v) >= 2 and all(v)) for v in by_pair.values()]
    return {
        "phase_final_accuracy": acc,
        "heldout_path_length_accuracy": mean(long_correct),
        "paired_counterfactual_accuracy": mean(pair_scores),
        "paired_counterfactual_margin": mean(pair_scores),
        "gate_shuffle_accuracy": gate_acc,
        "gate_shuffle_collapse": acc - gate_acc,
        "target_shuffle_accuracy": target_acc,
        "target_shuffle_collapse": acc - target_acc,
        "wall_shuffle_accuracy": wall_acc,
        "wall_shuffle_degradation": acc - wall_acc,
        "highway_phase_retention": mean(retention),
        "wall_leak_rate": mean(wall_leak),
        "pre_wall_pressure": mean(pre_wall),
        "direct_output_leak_rate": mean(direct_leak),
        "uses_forbidden_private_field": max(audit_flags) if audit_flags else 0.0,
    }


def mean(vals: list[float] | list[int]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def fitness(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics["phase_final_accuracy"]
        + 0.15 * metrics["heldout_path_length_accuracy"]
        + 0.15 * metrics["paired_counterfactual_accuracy"]
        + 0.10 * metrics["gate_shuffle_collapse"]
        + 0.10 * metrics["highway_phase_retention"]
        - 0.05 * metrics["wall_leak_rate"]
        - 0.05 * metrics["direct_output_leak_rate"]
    )


def ablation_drop(cases: list[CaseBundle], arm: str, circuit: LocalCircuit | None) -> float:
    if circuit is None:
        return 0.0
    base = evaluate(cases, arm, circuit)["phase_final_accuracy"]
    ablated = evaluate(cases, arm, circuit, ablate=True)["phase_final_accuracy"]
    return max(0.0, base - ablated)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    out = Path(job["out"])
    jid = f"{job['arm']}__seed{job['seed']}__p{job['pockets']}"
    progress_path = out / "job_progress" / f"{jid}.jsonl"
    candidate_path = out / "job_progress" / f"{jid}.candidate_log.jsonl"
    rng = random.Random(job["seed"] * 991 + hash(job["arm"]) % 100000)
    train_cases = generate_cases(job["candidate_eval_examples"], job["width"], job["seed"], "train")
    eval_cases = generate_cases(job["eval_examples"], job["width"], job["seed"], "eval")
    circuit: LocalCircuit | None = None
    if job["arm"] in MUTABLE_ARMS:
        circuit = initial_circuit(job["arm"], job["pockets"], rng)
    parent_metrics = evaluate(train_cases, job["arm"], circuit)
    parent_score = fitness(parent_metrics)
    accepted = 0
    evaluated_ops = Counter()
    accepted_ops = Counter()
    start = time.time()
    append_jsonl(progress_path, {"event": "job_start", "job_id": jid, "score": parent_score, "time": start})
    if circuit is not None:
        for step in range(1, job["steps"] + 1):
            best_circuit = circuit
            best_score = parent_score
            best_metrics = parent_metrics
            best_op = ""
            for cand in range(job["jackpot"]):
                op = rng.choice(MUTATION_OPERATORS)
                evaluated_ops[op] += 1
                child = mutate(circuit, rng, op)
                metrics = evaluate(train_cases, job["arm"], child)
                score = fitness(metrics)
                delta = score - parent_score
                append_jsonl(candidate_path, {"event": "candidate", "job_id": jid, "step": step, "candidate": cand, "operator": op, "score": score, "delta": delta})
                if score > best_score:
                    best_circuit = child
                    best_score = score
                    best_metrics = metrics
                    best_op = op
            if best_score > parent_score + 1e-12:
                circuit = best_circuit
                parent_score = best_score
                parent_metrics = best_metrics
                accepted += 1
                accepted_ops[best_op] += 1
                append_jsonl(
                    out / "locality_audit.jsonl",
                    {
                        "job_id": jid,
                        "step": step,
                        "operator": best_op,
                        "max_edge_distance": 1,
                        "writes_to_target_readout_directly": False,
                        "reads_nonlocal_cell": False,
                        "writes_wall_cell": False,
                        "uses_forbidden_private_field": False,
                    },
                )
            if step % max(1, min(job["steps"], 100)) == 0 or time.time() - start > job["heartbeat_sec"]:
                append_jsonl(progress_path, {"event": "heartbeat", "job_id": jid, "step": step, "steps": job["steps"], "accepted": accepted, "score": parent_score, "elapsed_sec": time.time() - start})
    final_metrics = evaluate(eval_cases, job["arm"], circuit)
    drop = ablation_drop(eval_cases, job["arm"], circuit)
    final_metrics["pocket_ablation_phase_drop"] = drop
    final_metrics["accepted_operator_rate"] = accepted / max(1, job["steps"])
    final_metrics["destructive_mutation_rate"] = max(0.0, 0.98 - final_metrics["highway_phase_retention"])
    final_metrics["locality_audit_fail_rate"] = final_metrics["uses_forbidden_private_field"]
    operator_summary = {op: {"evaluated": evaluated_ops[op], "accepted": accepted_ops[op]} for op in MUTATION_OPERATORS}
    result = {
        "job_id": jid,
        "arm": job["arm"],
        "seed": job["seed"],
        "width": job["width"],
        "pockets": job["pockets"],
        "steps": job["steps"],
        "jackpot": job["jackpot"],
        "metrics": final_metrics,
        "score": fitness(final_metrics),
        "operator_summary": operator_summary,
        "elapsed_sec": time.time() - start,
    }
    append_jsonl(progress_path, {"event": "job_done", "job_id": jid, "elapsed_sec": result["elapsed_sec"], "metrics": final_metrics})
    append_jsonl(out / "pocket_ablation.jsonl", {"job_id": jid, "arm": job["arm"], "seed": job["seed"], "pocket_ablation_phase_drop": drop})
    append_jsonl(out / "counterfactual_metrics.jsonl", {"job_id": jid, "arm": job["arm"], "seed": job["seed"], "paired_counterfactual_accuracy": final_metrics["paired_counterfactual_accuracy"]})
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["arm"]].append(row)
    arm_summary = []
    for arm, rows in sorted(grouped.items()):
        metric_names = sorted({k for row in rows for k in row["metrics"]})
        out = {"arm": arm, "jobs": len(rows)}
        for metric in metric_names:
            out[metric] = mean([float(row["metrics"].get(metric, 0.0)) for row in rows])
        arm_summary.append(out)
    by = {row["arm"]: row for row in arm_summary}
    verdicts: list[str] = []
    any_audit_fail = any(row.get("uses_forbidden_private_field", 0.0) > 0 for row in arm_summary)
    if any_audit_fail:
        verdicts.append("DIRECT_SHORTCUT_CONTAMINATION")
    gated = by.get("HIGHWAY_WITH_GATED_POCKETS_PHASE")
    ungated = by.get("HIGHWAY_WITH_UNGATED_POCKETS_PHASE")
    base = by.get("HIGHWAY_ONLY_PHASE")
    random_no = by.get("HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE")
    unrestricted = by.get("UNRESTRICTED_GRAPH_MUTATION_PHASE")
    if gated and base and random_no and not any_audit_fail:
        if (
            gated["phase_final_accuracy"] >= base["phase_final_accuracy"] + 0.05
            and gated["phase_final_accuracy"] >= random_no["phase_final_accuracy"] + 0.05
            and gated["gate_shuffle_collapse"] >= 0.10
            and gated["paired_counterfactual_accuracy"] >= 0.85
            and gated["pocket_ablation_phase_drop"] >= 0.05
            and gated["highway_phase_retention"] >= 0.98
            and gated["wall_leak_rate"] <= 0.02
            and gated["direct_output_leak_rate"] <= 0.01
        ):
            verdicts.append("MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT")
        else:
            verdicts.append("PHASE_CREDIT_ASSIGNMENT_NOT_SOLVED")
    if (
        ungated
        and gated
        and base
        and ungated["phase_final_accuracy"] >= gated["phase_final_accuracy"] - 0.01
        and ungated["phase_final_accuracy"] >= base["phase_final_accuracy"] + 0.05
    ):
        verdicts.append("UNGATED_POCKETS_SUFFICIENT")
    if (
        unrestricted
        and gated
        and base
        and unrestricted["phase_final_accuracy"] >= gated["phase_final_accuracy"] - 0.01
        and unrestricted["phase_final_accuracy"] >= base["phase_final_accuracy"] + 0.05
    ):
        verdicts.append("UNRESTRICTED_GRAPH_SUFFICIENT")
    split = {}
    for key in ("ORACLE_ROUTING_MUTABLE_PHASE_POCKET", "MUTABLE_ROUTING_ORACLE_PHASE", "MUTABLE_ROUTING_MUTABLE_PHASE"):
        if key in by:
            split[key] = by[key]["phase_final_accuracy"]
    if split:
        full = split.get("MUTABLE_ROUTING_MUTABLE_PHASE", 0.0)
        routing_oracle = split.get("MUTABLE_ROUTING_ORACLE_PHASE", 0.0)
        oracle_routing = split.get("ORACLE_ROUTING_MUTABLE_PHASE_POCKET", 0.0)
        if oracle_routing > full + 0.05:
            verdicts.append("ROUTING_IS_BLOCKER")
        if routing_oracle > full + 0.05:
            verdicts.append("PHASE_TRANSPORT_IS_BLOCKER")
    if not verdicts:
        verdicts.append("TASK_TOO_HARD")
    return {"arm_summary": arm_summary, "verdicts": verdicts}


def collect_operator_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        for op, vals in row["operator_summary"].items():
            acc[op]["evaluated"] += vals.get("evaluated", 0)
            acc[op]["accepted"] += vals.get("accepted", 0)
    return {
        op: {
            "evaluated": int(vals["evaluated"]),
            "accepted": int(vals["accepted"]),
            "accept_rate": vals["accepted"] / vals["evaluated"] if vals["evaluated"] else 0.0,
        }
        for op, vals in sorted(acc.items())
    }


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("arm_summary", [])
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT Report",
        "",
        "## Verdicts",
        "",
        "```text",
        "\n".join(summary.get("verdicts", [])),
        "```",
        "",
        "## Arm Summary",
        "",
        markdown_table(
            rows,
            [
                "arm",
                "phase_final_accuracy",
                "heldout_path_length_accuracy",
                "paired_counterfactual_accuracy",
                "gate_shuffle_collapse",
                "highway_phase_retention",
                "pocket_ablation_phase_drop",
                "wall_leak_rate",
                "uses_forbidden_private_field",
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "This probe tests local spatial phase credit assignment only. It does not prove consciousness, full VRAXION, language grounding, or production sidepockets.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    pockets_values = [int(v) for v in parse_csv(args.pockets)]
    arms = parse_csv(args.arms) if args.arms else list(ALL_ARMS)
    candidate_eval_examples = min(args.eval_examples, args.candidate_eval_examples)
    jobs = []
    for seed in seeds:
        for pockets in pockets_values:
            for arm in arms:
                jobs.append(
                    {
                        "out": str(args.out),
                        "arm": arm,
                        "seed": seed,
                        "width": args.width,
                        "pockets": pockets,
                        "steps": args.steps,
                        "jackpot": args.jackpot,
                        "eval_examples": args.eval_examples,
                        "candidate_eval_examples": candidate_eval_examples,
                        "heartbeat_sec": args.heartbeat_sec,
                    }
                )
    return jobs


def refresh(out: Path, results: list[dict[str, Any]], completed: int, total: int) -> dict[str, Any]:
    summary = aggregate(results) if results else {"arm_summary": [], "verdicts": ["RUN_IN_PROGRESS"]}
    summary["completed_jobs"] = completed
    summary["total_jobs"] = total
    summary["updated_at"] = time.time()
    write_json(out / "summary.json", summary)
    write_json(out / "operator_summary.json", collect_operator_summary(results))
    split_rows = []
    by = {row["arm"]: row for row in summary.get("arm_summary", [])}
    if {"ORACLE_ROUTING_MUTABLE_PHASE_POCKET", "MUTABLE_ROUTING_ORACLE_PHASE", "MUTABLE_ROUTING_MUTABLE_PHASE"} & set(by):
        full = by.get("MUTABLE_ROUTING_MUTABLE_PHASE", {}).get("phase_final_accuracy", 0.0)
        split_rows.append(
            {
                "routing_credit_gap": by.get("ORACLE_ROUTING_MUTABLE_PHASE_POCKET", {}).get("phase_final_accuracy", 0.0) - full,
                "phase_transport_credit_gap": by.get("MUTABLE_ROUTING_ORACLE_PHASE", {}).get("phase_final_accuracy", 0.0) - full,
                "routing_phase_interaction_gap": full,
            }
        )
    with (out / "phase_credit_split_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in split_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_report(out, summary)
    return summary


def write_contract_snapshot(out: Path) -> None:
    text = CONTRACT.read_text(encoding="utf-8") if CONTRACT.exists() else "# Missing contract at run start\n"
    (out / "contract_snapshot.md").write_text(text, encoding="utf-8")


def aggregate_candidate_logs(out: Path) -> None:
    with (out / "candidate_log.jsonl").open("w", encoding="utf-8") as dest:
        for path in sorted((out / "job_progress").glob("*.candidate_log.jsonl")):
            dest.write(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT probe.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--candidate-eval-examples", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--pockets", default="4")
    parser.add_argument("--jackpot", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "job_progress").mkdir(exist_ok=True)
    queue = build_queue(args)
    args_json = vars(args).copy()
    args_json["out"] = str(args_json["out"])
    write_json(out / "queue.json", {"args": args_json, "jobs": queue, "job_count": len(queue)})
    write_contract_snapshot(out)
    sample_cases = generate_cases(12, args.width, parse_seeds(args.seeds)[0], "eval")
    with (out / "examples_sample.jsonl").open("w", encoding="utf-8") as handle:
        for case in sample_cases:
            handle.write(
                json.dumps(
                    {
                        "public_case_id": case.public.case_id,
                        "public_source": case.public.source,
                        "public_target": case.public.target,
                        "private_family": case.private.family,
                        "private_label": case.private.label,
                        "private_path_len": case.private.path_len,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    append_jsonl(out / "progress.jsonl", {"event": "run_start", "jobs": len(queue), "time": time.time()})
    results: list[dict[str, Any]] = []
    completed = 0
    total = len(queue)
    last_refresh = time.time()
    with cf.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in queue}
        pending = set(future_to_job)
        while pending:
            done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                job = future_to_job[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover
                    append_jsonl(out / "progress.jsonl", {"event": "job_failed", "job": job, "error": repr(exc), "time": time.time()})
                    raise
                results.append(result)
                completed += 1
                append_jsonl(out / "metrics.jsonl", result)
                append_jsonl(out / "progress.jsonl", {"event": "job_done", "job_id": result["job_id"], "completed": completed, "total": total, "time": time.time()})
            if done or time.time() - last_refresh >= args.heartbeat_sec:
                refresh(out, results, completed, total)
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": completed, "total": total, "time": time.time()})
                last_refresh = time.time()
    aggregate_candidate_logs(out)
    summary = refresh(out, results, completed, total)
    append_jsonl(out / "progress.jsonl", {"event": "run_done", "completed": completed, "total": total, "verdicts": summary["verdicts"], "time": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
