"""Shared helpers for deterministic English GPU probes."""

from __future__ import annotations

import hashlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.graph import SelfWiringGraph


IO_DIM = 256
SEQ_LEN = 200
N_TRAIN_SEQS = 5
N_EVAL_SEQS = 3
DEFAULT_TICKS = 6
PROBE_BYTES = [32, 97, 101, 116]
ROOT_REPO = Path("S:/AI/work/VRAXION_DEV")
DEFAULT_DATA_PATH = ROOT_REPO / "Diamond Code" / "data" / "traindat" / "fineweb_edu.traindat"
DEFAULT_CHECKPOINT_DIR = ROOT_REPO / "v4.2" / "checkpoints"


@dataclass
class EnglishInit:
    mask: torch.Tensor
    w_in: torch.Tensor
    w_out: torch.Tensor
    bp: torch.Tensor
    bp_norm: torch.Tensor
    retention: float
    io_dim: int
    neurons: int
    mask_hash: str


def resolve_data_path(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"English data file missing: {path}")
    return path


def resolve_checkpoint_path(name: str, checkpoint_dir: str = "") -> Path:
    base = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    return path


def make_bp(io_dim: int = IO_DIM, seed: int = 12345) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def load_all_data(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def make_fixed_eval_sequences(all_data: np.ndarray, seed: int = 9999, n_eval: int = N_EVAL_SEQS, seq_len: int = SEQ_LEN) -> np.ndarray:
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_eval):
        off = rng.randint(0, len(all_data) - seq_len)
        rows.append(all_data[off : off + seq_len])
    return np.stack(rows, axis=0).astype(np.uint8, copy=False)


def make_fixed_train_report_sequences(
    all_data: np.ndarray,
    seed: int = 424242,
    n_train: int = N_TRAIN_SEQS,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_train):
        off = rng.randint(0, len(all_data) - seq_len)
        rows.append(all_data[off : off + seq_len])
    return np.stack(rows, axis=0).astype(np.uint8, copy=False)


def load_checkpoint_init(
    checkpoint_path: Path,
    *,
    device: torch.device,
    io_dim: int = IO_DIM,
    net_seed: int = 42,
) -> EnglishInit:
    random.seed(net_seed)
    np.random.seed(net_seed)
    net = SelfWiringGraph(io_dim)
    bp_np = make_bp(io_dim)

    d = np.load(checkpoint_path)
    mask_np = np.zeros((net.H, net.H), dtype=np.float32)
    mask_np[d["rows"], d["cols"]] = d["vals"].astype(np.float32, copy=False)
    loss_pct = int(d["loss_pct"]) if "loss_pct" in d else int(net.loss_pct)
    retention = (100 - loss_pct) * 0.01

    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
    w_in = torch.from_numpy(net.W_in.copy()).to(device=device, dtype=torch.float32)
    w_out = torch.from_numpy(net.W_out.copy()).to(device=device, dtype=torch.float32)
    bp = torch.from_numpy(bp_np.copy()).to(device=device, dtype=torch.float32)
    bp_norm = bp / (torch.linalg.norm(bp, dim=1, keepdim=True) + 1e-8)
    mask_hash = hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]

    return EnglishInit(
        mask=mask,
        w_in=w_in,
        w_out=w_out,
        bp=bp,
        bp_norm=bp_norm,
        retention=retention,
        io_dim=io_dim,
        neurons=net.H,
        mask_hash=mask_hash,
    )


def make_empty_init(*, device: torch.device, io_dim: int = IO_DIM, net_seed: int = 42) -> EnglishInit:
    random.seed(net_seed)
    np.random.seed(net_seed)
    net = SelfWiringGraph(io_dim)
    net.mask[:] = 0
    bp_np = make_bp(io_dim)

    mask = torch.from_numpy(net.mask.copy()).to(device=device, dtype=torch.float32)
    w_in = torch.from_numpy(net.W_in.copy()).to(device=device, dtype=torch.float32)
    w_out = torch.from_numpy(net.W_out.copy()).to(device=device, dtype=torch.float32)
    bp = torch.from_numpy(bp_np.copy()).to(device=device, dtype=torch.float32)
    bp_norm = bp / (torch.linalg.norm(bp, dim=1, keepdim=True) + 1e-8)
    mask_hash = hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]

    return EnglishInit(
        mask=mask,
        w_in=w_in,
        w_out=w_out,
        bp=bp,
        bp_norm=bp_norm,
        retention=float(net.retention),
        io_dim=io_dim,
        neurons=net.H,
        mask_hash=mask_hash,
    )


def gather_seq_batch(all_data: np.ndarray, offsets: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    steps = np.arange(seq_len, dtype=np.int64)[None, :]
    return all_data[offsets[:, None] + steps]


def seq_batch_to_device(seq_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(seq_batch.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)


@torch.inference_mode()
def eval_sequence_batch(
    *,
    mask: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    bp: torch.Tensor,
    bp_norm: torch.Tensor,
    seq_batch: torch.Tensor,
    retention: float,
    threshold: float,
    ticks: int,
) -> dict[str, float]:
    batch, seq_len = seq_batch.shape
    neurons = mask.shape[0]
    state = torch.zeros((batch, neurons), dtype=torch.float32, device=mask.device)
    charge = torch.zeros((batch, neurons), dtype=torch.float32, device=mask.device)
    correct = torch.zeros((batch,), dtype=torch.float32, device=mask.device)
    prob_sum = torch.zeros((batch,), dtype=torch.float32, device=mask.device)

    for i in range(seq_len - 1):
        act = state.clone()
        src = seq_batch[:, i]
        target = seq_batch[:, i + 1]
        injected = bp[src] @ w_in
        for t in range(ticks):
            if t == 0:
                act = act + injected
            raw = act @ mask
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            charge = charge + raw
            charge = charge * retention
            act = torch.clamp(charge - threshold, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)
        state = act.clone()
        out = charge @ w_out
        out_n = out / (torch.linalg.norm(out, dim=1, keepdim=True) + 1e-8)
        sims = out_n @ bp_norm.T
        probs = torch.softmax(sims, dim=1)
        pred = torch.argmax(probs, dim=1)
        correct = correct + (pred == target).to(torch.float32)
        prob_sum = prob_sum + probs[torch.arange(batch, device=mask.device), target]

    denom = float(seq_len - 1)
    acc = correct / denom
    avg_prob = prob_sum / denom
    score = 0.5 * acc.mean() + 0.5 * avg_prob.mean()
    return {
        "score": float(score.item()),
        "acc": float(acc.mean().item()),
        "avg_target_prob": float(avg_prob.mean().item()),
    }


@torch.inference_mode()
def probe_threshold_dynamics(
    *,
    mask: torch.Tensor,
    w_in: torch.Tensor,
    bp: torch.Tensor,
    probe_bytes: list[int],
    retention: float,
    threshold: float,
    ticks: int,
) -> dict[str, object]:
    charge_max_runs: list[list[float]] = []
    act_max_runs: list[list[float]] = []
    active_count_runs: list[list[int]] = []
    active_ratio_runs: list[list[float]] = []
    newly_active_runs: list[int] = []
    probe_details: list[dict[str, object]] = []
    neurons = mask.shape[0]

    for byte in probe_bytes:
        state = torch.zeros((neurons,), dtype=torch.float32, device=mask.device)
        charge = torch.zeros((neurons,), dtype=torch.float32, device=mask.device)
        charge_max_this: list[float] = []
        act_max_this: list[float] = []
        active_counts_this: list[int] = []
        active_ratios_this: list[float] = []
        active_sets: list[set[int]] = []
        injected = bp[byte] @ w_in
        act = state.clone()

        for t in range(ticks):
            if t == 0:
                act = act + injected
            raw = act @ mask
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            charge = charge + raw
            charge = charge * retention
            act = torch.clamp(charge - threshold, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)

            active_idx = torch.nonzero(act > 0.0, as_tuple=False).flatten().detach().cpu().tolist()
            active_set = set(int(x) for x in active_idx)
            active_sets.append(active_set)
            active_count = len(active_set)
            charge_max_this.append(float(torch.abs(charge).max().item()))
            act_max_this.append(float(act.max().item()))
            active_counts_this.append(active_count)
            active_ratios_this.append(float(active_count / neurons))

        tick0 = active_sets[0] if active_sets else set()
        later = set().union(*active_sets[1:]) if len(active_sets) > 1 else set()
        newly_active = len(later - tick0)
        newly_active_runs.append(newly_active)
        charge_max_runs.append(charge_max_this)
        act_max_runs.append(act_max_this)
        active_count_runs.append(active_counts_this)
        active_ratio_runs.append(active_ratios_this)
        probe_details.append(
            {
                "byte": int(byte),
                "charge_max_per_tick": charge_max_this,
                "act_max_per_tick": act_max_this,
                "active_neurons_per_tick": active_counts_this,
                "active_ratio_per_tick": active_ratios_this,
                "newly_active_after_tick0": newly_active,
            }
        )

    charge_max = np.mean(np.array(charge_max_runs, dtype=np.float64), axis=0).tolist()
    act_max = np.mean(np.array(act_max_runs, dtype=np.float64), axis=0).tolist()
    active_counts = np.mean(np.array(active_count_runs, dtype=np.float64), axis=0).tolist()
    active_ratios = np.mean(np.array(active_ratio_runs, dtype=np.float64), axis=0).tolist()

    return {
        "charge_max_per_tick": [float(x) for x in charge_max],
        "act_max_per_tick": [float(x) for x in act_max],
        "active_neurons_per_tick": [float(x) for x in active_counts],
        "active_ratio_per_tick": [float(x) for x in active_ratios],
        "newly_active_after_tick0": float(np.mean(newly_active_runs)),
        "probe_details": probe_details,
    }


def mask_hash(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def make_output_path(explicit: str, stem: str) -> Path:
    if explicit:
        return Path(explicit)
    logs_dir = ROOT_REPO / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / stem
