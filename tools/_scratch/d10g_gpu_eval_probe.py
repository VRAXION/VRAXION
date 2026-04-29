#!/usr/bin/env python3
"""D10g GPU evaluator feasibility probe.

This is intentionally a scratch/prototype tool:
- It does not change checkpoints.
- It does not share output paths with D10b.
- It uses PyTorch CUDA first, not Rust-CUDA/nvcc.

The probe answers two questions:
1. Does a Torch CPU/CUDA implementation preserve the current propagation
   semantics closely enough to be useful as an evaluator?
2. Does batching candidates make GPU evaluation fast enough to justify a
   deeper integration later?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


MAX_CHARGE = 15
MAX_CLASSES = 397
PHASE_BASE = torch.tensor([7, 8, 10, 12, 13, 12, 10, 8], dtype=torch.int16)


class BinReader:
    def __init__(self, data: bytes):
        self.data = data
        self.off = 0

    def read(self, n: int) -> bytes:
        if self.off + n > len(self.data):
            raise ValueError(f"read past end: off={self.off} n={n} len={len(self.data)}")
        out = self.data[self.off : self.off + n]
        self.off += n
        return out

    def u8(self) -> int:
        return self.read(1)[0]

    def i8_vec(self) -> np.ndarray:
        n = self.usize()
        return np.frombuffer(self.read(n), dtype=np.int8).copy()

    def u8_vec(self) -> bytes:
        n = self.usize()
        return self.read(n)

    def u32_vec(self) -> np.ndarray:
        n = self.usize()
        return np.frombuffer(self.read(n * 4), dtype="<u4").astype(np.int64)

    def i32_vec(self) -> np.ndarray:
        n = self.usize()
        return np.frombuffer(self.read(n * 4), dtype="<i4").astype(np.int16)

    def usize(self) -> int:
        return struct.unpack_from("<Q", self.read(8))[0]

    def usize_vec(self) -> np.ndarray:
        n = self.usize()
        return np.frombuffer(self.read(n * 8), dtype="<u8").astype(np.int64)

    def f64(self) -> float:
        return struct.unpack_from("<d", self.read(8))[0]

    def string(self) -> str:
        n = self.usize()
        return self.read(n).decode("utf-8")

    def done(self) -> bool:
        return self.off == len(self.data)


@dataclass
class NetworkArrays:
    h: int
    sources: np.ndarray
    targets: np.ndarray
    threshold: np.ndarray
    channel: np.ndarray
    polarity: np.ndarray


@dataclass
class ProjectionArrays:
    weights: np.ndarray
    input_dim: int
    output_classes: int


@dataclass
class CheckpointArrays:
    path: str
    network: NetworkArrays
    projection: ProjectionArrays
    meta: dict


@dataclass
class VcbpTablePy:
    hot_float: np.ndarray
    oov: np.ndarray
    row_map: np.ndarray
    chan_min: np.ndarray
    chan_max: np.ndarray
    e: int
    vocab_size: int
    n_hot: int

    def embed_id(self, pair_id: int) -> np.ndarray:
        row = self.row_map[pair_id]
        if row >= 0:
            return self.hot_float[row]
        return self.oov

    def is_hot(self, pair_id: int) -> bool:
        return self.row_map[pair_id] >= 0

    def quantize_to_input(self, pair_id: int) -> np.ndarray:
        emb = self.embed_id(pair_id)
        norm = (emb - self.chan_min) / (self.chan_max - self.chan_min)
        return np.rint(norm * MAX_CHARGE).clip(0, MAX_CHARGE).astype(np.int16)


def parse_network(bytes_: bytes) -> NetworkArrays:
    r = BinReader(bytes_)
    version = r.u8()
    if version != 1:
        raise ValueError(f"network version {version} != 1")
    h = r.usize()
    sources = r.usize_vec()
    targets = r.usize_vec()
    threshold = r.u32_vec().astype(np.int16)
    channel = np.frombuffer(r.u8_vec(), dtype=np.uint8).astype(np.int16)
    polarity = r.i32_vec().astype(np.int16)
    if not r.done():
        raise ValueError(f"trailing network bytes: {len(bytes_) - r.off}")
    if len(threshold) != h or len(channel) != h or len(polarity) != h:
        raise ValueError("network parameter length mismatch")
    return NetworkArrays(h, sources, targets, threshold, channel, polarity)


def parse_projection(bytes_: bytes) -> ProjectionArrays:
    r = BinReader(bytes_)
    weights = r.i8_vec().astype(np.int16)
    input_dim = r.usize()
    output_classes = r.usize()
    if not r.done():
        raise ValueError(f"trailing projection bytes: {len(bytes_) - r.off}")
    if weights.size != input_dim * output_classes:
        raise ValueError("projection weight length mismatch")
    return ProjectionArrays(weights.reshape(input_dim, output_classes), input_dim, output_classes)


def load_checkpoint(path: Path) -> CheckpointArrays:
    r = BinReader(path.read_bytes())
    version = r.u8()
    if version != 1:
        raise ValueError(f"checkpoint version {version} != 1")
    network_bytes = r.u8_vec()
    projection_bytes = r.u8_vec()
    meta = {
        "step": r.usize(),
        "accuracy": r.f64(),
        "label": r.string(),
    }
    if not r.done():
        raise ValueError(f"trailing checkpoint bytes: {path}")
    return CheckpointArrays(
        str(path),
        parse_network(network_bytes),
        parse_projection(projection_bytes),
        meta,
    )


def load_vcbp(path: Path) -> VcbpTablePy:
    data = path.read_bytes()
    if data[:4] != b"VCBP" or data[4] != 1:
        raise ValueError("unsupported VCBP packed file")
    vocab_size = struct.unpack_from("<I", data, 8)[0]
    e = struct.unpack_from("<I", data, 12)[0]
    n_hot = struct.unpack_from("<I", data, 16)[0]
    off = 32
    scales = np.frombuffer(data, dtype="<f2", count=e, offset=off).astype(np.float32)
    off += e * 2
    oov = np.frombuffer(data, dtype="<f2", count=e, offset=off).astype(np.float32)
    off += e * 2
    bitmap_bytes = (vocab_size + 7) // 8
    bitmap = data[off : off + bitmap_bytes]
    off += bitmap_bytes
    hot_mask = np.zeros(vocab_size, dtype=bool)
    for i in range(vocab_size):
        hot_mask[i] = ((bitmap[i // 8] >> (i % 8)) & 1) == 1
    if int(hot_mask.sum()) != n_hot:
        raise ValueError("VCBP hot bitmap popcount mismatch")
    row_bytes = e // 2
    hot_float = np.zeros((n_hot, e), dtype=np.float32)
    for row in range(n_hot):
        raw = data[off + row * row_bytes : off + (row + 1) * row_bytes]
        vals = []
        for byte in raw:
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            vals.append(low - 16 if low >= 8 else low)
            vals.append(high - 16 if high >= 8 else high)
        hot_float[row] = np.asarray(vals, dtype=np.float32) * scales
    off += n_hot * row_bytes
    if off != len(data):
        raise ValueError("VCBP trailing bytes")
    row_map = np.full(vocab_size, -1, dtype=np.int32)
    row_map[np.where(hot_mask)[0]] = np.arange(n_hot, dtype=np.int32)
    chan_min = hot_float.min(axis=0)
    chan_max = hot_float.max(axis=0)
    zero_range = np.abs(chan_max - chan_min) < 1e-8
    chan_max[zero_range] = chan_min[zero_range] + 1.0
    return VcbpTablePy(hot_float, oov, row_map, chan_min, chan_max, e, vocab_size, n_hot)


def build_corpus_pairs(corpus_path: Path, table: VcbpTablePy, max_classes: int = MAX_CLASSES):
    data = corpus_path.read_bytes()
    n_pairs = len(data) // 2
    pair_ids = np.empty(n_pairs, dtype=np.int32)
    freq = np.zeros(65536, dtype=np.int64)
    for i in range(n_pairs):
        pid = (data[i * 2] << 8) | data[i * 2 + 1]
        pair_ids[i] = pid
        freq[pid] += 1
    hot_ids = [i for i in range(65536) if table.is_hot(i) and freq[i] > 0]
    hot_ids.sort(key=lambda v: int(freq[v]), reverse=True)
    hot_ids = hot_ids[:max_classes]
    hot_to_idx = np.full(65536, -1, dtype=np.int32)
    for idx, pid in enumerate(hot_ids):
        hot_to_idx[pid] = idx
    return pair_ids, hot_to_idx, len(hot_ids)


def build_bigram_unigram(pair_ids: np.ndarray, hot_to_idx: np.ndarray, n_hot: int):
    counts = np.zeros((n_hot, n_hot), dtype=np.float32)
    unigram_counts = np.zeros(n_hot, dtype=np.float32)
    for pid in pair_ids:
        idx = hot_to_idx[int(pid)]
        if idx >= 0:
            unigram_counts[idx] += 1
    for cur, nxt in zip(pair_ids[:-1], pair_ids[1:]):
        cur_idx = hot_to_idx[int(cur)]
        nxt_idx = hot_to_idx[int(nxt)]
        if cur_idx >= 0 and nxt_idx >= 0:
            counts[cur_idx, nxt_idx] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    bigram = np.where(row_sum < 1.0, 1.0 / max(n_hot, 1), counts / np.maximum(row_sum, 1.0))
    total = unigram_counts.sum()
    unigram = unigram_counts / total if total >= 1.0 else np.full(n_hot, 1.0 / max(n_hot, 1))
    return bigram.astype(np.float32), unigram.astype(np.float32)


def phi_dim(h: int) -> int:
    return int(round(h / 1.618033988749895))


def deterministic_offset(seed: int, n_pairs: int, eval_len: int) -> int:
    # This is intentionally deterministic and cross-language simple.
    # It is not Rust StdRng-compatible; CPU/GPU equivalence is tested on the
    # exact same offset schedule.
    span = n_pairs - eval_len - 1
    if span <= 1:
        return 0
    return 1 + (seed % span)


def adjacency_dense(nets: list[NetworkArrays], device: torch.device) -> torch.Tensor:
    b = len(nets)
    h = nets[0].h
    adj = torch.zeros((b, h, h), dtype=torch.int16, device=device)
    for bi, net in enumerate(nets):
        if net.h != h:
            raise ValueError("all batched networks must have same H")
        adj[bi, torch.as_tensor(net.sources, device=device), torch.as_tensor(net.targets, device=device)] = 1
    return adj


def tensors_for_checkpoints(checkpoints: list[CheckpointArrays], device: torch.device):
    nets = [c.network for c in checkpoints]
    h = nets[0].h
    output_start = h - phi_dim(h)
    thresholds = torch.as_tensor(np.stack([n.threshold for n in nets]), dtype=torch.int16, device=device)
    channels = torch.as_tensor(np.stack([n.channel for n in nets]), dtype=torch.int16, device=device)
    polarity = torch.as_tensor(np.stack([n.polarity for n in nets]), dtype=torch.int16, device=device)
    weights = torch.as_tensor(np.stack([c.projection.weights for c in checkpoints]), dtype=torch.int16, device=device)
    adj = adjacency_dense(nets, device)
    return {
        "h": h,
        "output_start": output_start,
        "threshold": thresholds,
        "channel": channels,
        "polarity": polarity,
        "weights": weights,
        "adj": adj,
    }


def quantized_inputs(pair_ids: Iterable[int], table: VcbpTablePy, h: int, input_scatter: bool = False) -> np.ndarray:
    e = table.e
    input_end = phi_dim(h)
    rows = []
    for pid in pair_ids:
        base = table.quantize_to_input(int(pid))
        row = np.zeros(h, dtype=np.int16)
        if input_scatter:
            for idx in range(input_end):
                row[idx] = base[idx % e]
        else:
            row[:e] = base
        rows.append(row)
    return np.stack(rows)


def propagate_sequence_torch(
    state: dict,
    input_rows: torch.Tensor,
    ticks_per_token: int = 6,
    input_duration_ticks: int = 2,
    decay_interval_ticks: int = 6,
) -> torch.Tensor:
    b = state["threshold"].shape[0]
    h = state["h"]
    charge = torch.zeros((b, h), dtype=torch.int16, device=input_rows.device)
    activation = torch.zeros((b, h), dtype=torch.int16, device=input_rows.device)
    phase_base = PHASE_BASE.to(input_rows.device)
    output_charges = []
    for token_idx in range(input_rows.shape[0]):
        inp = input_rows[token_idx].unsqueeze(0).expand(b, -1)
        for tick in range(ticks_per_token):
            if decay_interval_ticks > 0 and tick % decay_interval_ticks == 0:
                charge = torch.clamp(charge - 1, min=0)
            if tick < input_duration_ticks:
                activation = torch.clamp(activation + inp, min=-128, max=127)
            # CUDA does not implement int16 bmm. Values are small integers, so
            # float32 bmm is exact for this range; round back to integer state.
            incoming = torch.bmm(
                activation.unsqueeze(1).to(torch.float32),
                state["adj"].to(torch.float32),
            ).squeeze(1).round().to(torch.int16)
            charge = torch.clamp(charge + incoming, min=0, max=MAX_CHARGE)
            activation = torch.zeros_like(activation)
            phase_tick = tick % 8
            channel_idx = state["channel"]
            phase_indices = (phase_tick + 9 - channel_idx) & 7
            phase_mult = phase_base[phase_indices.to(torch.long)]
            should_fire = (charge * 10) >= ((state["threshold"] + 1) * phase_mult)
            activation = torch.where(should_fire, state["polarity"], activation)
            charge = torch.where(should_fire, torch.zeros_like(charge), charge)
        output_charges.append(charge[:, state["output_start"] :])
    return torch.stack(output_charges, dim=1)


def projection_scores(output_charges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # output_charges: B x T x O, weights: B x O x C -> B x T x C
    return torch.bmm(output_charges.to(torch.float32), weights.to(torch.float32))


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return torch.where(den <= eps, torch.zeros_like(num), num / den)


def evaluate_metrics(
    checkpoints: list[CheckpointArrays],
    table: VcbpTablePy,
    pair_ids: np.ndarray,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    eval_len: int,
    eval_seeds: list[int],
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    t0 = time.perf_counter()
    state = tensors_for_checkpoints(checkpoints, device)
    b = len(checkpoints)
    acc_total = torch.zeros(b, dtype=torch.float64, device=device)
    echo_total = torch.zeros(b, dtype=torch.float64, device=device)
    smooth_total = torch.zeros(b, dtype=torch.float64, device=device)
    unigram_total = torch.zeros(b, dtype=torch.float64, device=device)
    for seed in eval_seeds:
        off = deterministic_offset(seed, len(pair_ids), eval_len)
        cur_ids = pair_ids[off : off + eval_len]
        next_ids = pair_ids[off + 1 : off + eval_len + 1]
        input_np = quantized_inputs(cur_ids, table, state["h"], input_scatter=False)
        inputs = torch.as_tensor(input_np, dtype=torch.int16, device=device)
        outputs = propagate_sequence_torch(state, inputs)
        scores = projection_scores(outputs, state["weights"])
        probs = torch.softmax(scores, dim=-1)
        target_idx = torch.as_tensor(hot_to_idx[next_ids], dtype=torch.long, device=device)
        cur_idx = torch.as_tensor(hot_to_idx[cur_ids], dtype=torch.long, device=device)
        preds = scores.argmax(dim=-1)
        valid_target = target_idx >= 0
        acc_total += ((preds == target_idx.unsqueeze(0)) & valid_target.unsqueeze(0)).sum(dim=1).to(torch.float64) / eval_len
        valid_cur = cur_idx >= 0
        if valid_cur.any():
            idxs = cur_idx[valid_cur]
            one_hot = torch.nn.functional.one_hot(idxs, num_classes=probs.shape[-1]).to(torch.float32).to(device)
            echo_total += cosine(probs[:, valid_cur, :], one_hot.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
            bigram_t = torch.as_tensor(bigram[idxs.cpu().numpy()], dtype=torch.float32, device=device)
            smooth_total += cosine(probs[:, valid_cur, :], bigram_t.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
        unigram_t = torch.as_tensor(unigram, dtype=torch.float32, device=device)
        unigram_total += cosine(probs, unigram_t.view(1, 1, -1)).sum(dim=1).to(torch.float64) / eval_len
    denom = float(len(eval_seeds))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    metrics = {
        "accuracy": (acc_total / denom).detach().cpu().tolist(),
        "echo": (echo_total / denom).detach().cpu().tolist(),
        "smooth": (smooth_total / denom).detach().cpu().tolist(),
        "unigram": (unigram_total / denom).detach().cpu().tolist(),
    }
    return metrics, elapsed


def toy_arrays() -> CheckpointArrays:
    net = NetworkArrays(
        h=8,
        sources=np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
        targets=np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        threshold=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16),
        channel=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16),
        polarity=np.array([1, 1, 1, -1, 1, -1, 1, 1], dtype=np.int16),
    )
    proj = ProjectionArrays(np.ones((3, 2), dtype=np.int16), 3, 2)
    return CheckpointArrays("toy", net, proj, {"step": 0, "accuracy": 0.0, "label": "toy"})


def propagate_sequence_python_toy(net: NetworkArrays, inputs: np.ndarray) -> np.ndarray:
    h = net.h
    charge = np.zeros(h, dtype=np.int16)
    activation = np.zeros(h, dtype=np.int16)
    outs = []
    for row in inputs:
        for tick in range(6):
            if tick % 6 == 0:
                charge = np.maximum(charge - 1, 0)
            if tick < 2:
                activation = np.clip(activation + row, -128, 127)
            incoming = np.zeros(h, dtype=np.int16)
            for s, t in zip(net.sources, net.targets):
                incoming[t] += activation[s]
            charge = np.clip(charge + incoming, 0, MAX_CHARGE)
            activation[:] = 0
            phase_tick = tick % 8
            for i in range(h):
                phase_mult = int(PHASE_BASE[(phase_tick + 9 - int(net.channel[i])) & 7])
                if int(charge[i]) * 10 >= (int(net.threshold[i]) + 1) * phase_mult:
                    activation[i] = net.polarity[i]
                    charge[i] = 0
        outs.append(charge.copy())
    return np.stack(outs)


def run_toy(device_name: str, out: Path | None) -> dict:
    device = torch.device(device_name)
    toy = toy_arrays()
    inputs = np.array(
        [
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 2],
        ],
        dtype=np.int16,
    )
    py = propagate_sequence_python_toy(toy.network, inputs)
    state = tensors_for_checkpoints([toy], device)
    torch_out = propagate_sequence_torch(state, torch.as_tensor(inputs, dtype=torch.int16, device=device))
    got = torch_out[0].detach().cpu().numpy()
    # toy output_start uses phi geometry; compare the full final state by running an H=8 full-output substitute.
    # For correctness gate here, compare the available output-zone tail and deterministic shape.
    py_tail = py[:, state["output_start"] :]
    passed = np.array_equal(py_tail, got)
    result = {
        "gate": "CPU_GPU_STATE_MATCH_TOY",
        "device": device_name,
        "passed": bool(passed),
        "max_abs_diff": int(np.max(np.abs(py_tail.astype(np.int32) - got.astype(np.int32)))),
        "first_8_token_output_tail": got.tolist(),
    }
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"toy_{device_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_real_inputs(args) -> tuple[list[CheckpointArrays], VcbpTablePy, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoints = [load_checkpoint(Path(args.baseline)), load_checkpoint(Path(args.target))]
    if args.negative:
        checkpoints.append(load_checkpoint(Path(args.negative)))
    table = load_vcbp(Path(args.packed))
    pair_ids, hot_to_idx, n_classes = build_corpus_pairs(Path(args.corpus), table, MAX_CLASSES)
    bigram, unigram = build_bigram_unigram(pair_ids, hot_to_idx, n_classes)
    for ckpt in checkpoints:
        if ckpt.projection.output_classes != n_classes:
            raise ValueError(
                f"projection classes {ckpt.projection.output_classes} != corpus classes {n_classes} for {ckpt.path}"
            )
    return checkpoints, table, pair_ids, hot_to_idx, bigram, unigram


def write_metric_csv(path: Path, names: list[str], metrics: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "smooth", "accuracy", "echo", "unigram"])
        for idx, name in enumerate(names):
            w.writerow([name, metrics["smooth"][idx], metrics["accuracy"][idx], metrics["echo"][idx], metrics["unigram"][idx]])


def run_real(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints, table, pair_ids, hot_to_idx, bigram, unigram = load_real_inputs(args)
    names = [Path(c.path).stem if c.path != "toy" else "toy" for c in checkpoints]
    device = torch.device(args.device)
    metrics, elapsed = evaluate_metrics(
        checkpoints,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )
    write_metric_csv(out / f"metrics_{args.device}.csv", names, metrics)
    deltas = {
        metric: [values[i] - values[0] for i in range(len(values))]
        for metric, values in metrics.items()
    }
    result = {
        "device": args.device,
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "elapsed_s": elapsed,
        "candidate_count": len(checkpoints),
        "metrics": metrics,
        "deltas_vs_baseline": deltas,
        "beta8_direction_match": {
            "smooth_positive": deltas["smooth"][1] > 0,
            "accuracy_positive": deltas["accuracy"][1] > 0,
            "unigram_non_negative": deltas["unigram"][1] >= 0,
            "echo_safe": abs(deltas["echo"][1]) <= 0.001,
        },
    }
    if args.device == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["vram_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    (out / f"result_{args.device}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_throughput(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints, table, pair_ids, hot_to_idx, bigram, unigram = load_real_inputs(args)
    sizes = parse_int_csv(args.batch_sizes)
    rows = []
    for device_name in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        device = torch.device(device_name)
        for size in sizes:
            reps = [checkpoints[i % len(checkpoints)] for i in range(size)]
            if device_name == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _, elapsed = evaluate_metrics(
                reps,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                unigram,
                args.eval_len,
                parse_int_csv(args.eval_seeds),
                device,
            )
            rows.append({
                "device": device_name,
                "batch_size": size,
                "elapsed_s": elapsed,
                "candidates_per_s": size / elapsed if elapsed > 0 else 0.0,
                "vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if device_name == "cuda" else 0.0,
            })
    with (out / "throughput.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["device", "batch_size", "elapsed_s", "candidates_per_s", "vram_mb"])
        w.writeheader()
        w.writerows(rows)
    result = {"rows": rows}
    (out / "throughput.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--throughput", action="store_true")
    p.add_argument("--eval-len", type=int, default=128)
    p.add_argument("--eval-seeds", default="990001,990002")
    p.add_argument("--batch-sizes", default="1,4,8,16,32,64")
    p.add_argument("--out", default="output/phase_d10g_gpu_eval_probe_20260429/smoke")
    p.add_argument("--baseline", default="output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--negative", default="")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    out = Path(args.out)
    if args.toy:
        result = run_toy(args.device, out)
    elif args.throughput:
        result = run_throughput(args)
    else:
        result = run_real(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
