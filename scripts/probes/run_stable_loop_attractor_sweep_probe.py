#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "stable_loop_attractor_sweep_001"
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_ATTRACTOR_SWEEP_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STABLE_LOOP_ATTRACTOR_SWEEP_001_RESULT.md"

PAD = "<PAD>"
SYMBOLS = ("A", "B", "C", "D")
TOKENS = (
    PAD,
    "A",
    "B",
    "C",
    "D",
    "anti_A",
    "anti_B",
    "anti_C",
    "anti_D",
    "reset",
    "delay",
    "mention_A",
    "mention_B",
    "mention_C",
    "mention_D",
    "quote_anti_A",
    "quote_anti_B",
    "quote_anti_C",
    "quote_anti_D",
    "actually_A",
    "actually_B",
    "actually_C",
    "actually_D",
    "instead_A",
    "instead_B",
    "instead_C",
    "instead_D",
    "create_X",
    "remove_X",
    "restore_X",
    "query_count",
    "noise",
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
LABELS = ("NONE", "A", "B", "C", "D", "COUNT0", "COUNT1", "COUNT2", "COUNT3", "COUNT4")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ALL_ARMS = (
    "MLP_STATIC",
    "DEEP_MLP_MATCHED_COMPUTE",
    "SIMPLE_RNN",
    "GRU",
    "GRU_EXTRA_NOOP_STEPS",
    "LSTM",
    "SUMMARY_DIRECT_HEAD",
    "MAIN_LOOP_MLP_AUTONOMOUS",
    "MAIN_LOOP_MLP_CONDITIONED",
    "MAIN_LOOP_GRU_AUTONOMOUS",
    "MAIN_LOOP_GRU_CONDITIONED",
    "HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS",
    "HIGHWAY_FF_SIDEPOCKETS_CONDITIONED",
    "HIGHWAY_RECURRENT_SIDEPOCKETS",
    "HIGHWAY_SPARSE_SIDEPOCKETS",
    "HIGHWAY_DENSE_SIDEPOCKETS",
    "HIGHWAY_PRISMION_SIDEPOCKETS",
    "VRAXION_LITE_LOOP",
)
LOOP_ARMS = tuple(arm for arm in ALL_ARMS if arm.startswith("MAIN_LOOP") or arm.startswith("HIGHWAY") or arm == "VRAXION_LITE_LOOP")
AUTONOMOUS_ARMS = tuple(arm for arm in LOOP_ARMS if "AUTONOMOUS" in arm or arm in {"HIGHWAY_RECURRENT_SIDEPOCKETS", "HIGHWAY_SPARSE_SIDEPOCKETS", "HIGHWAY_DENSE_SIDEPOCKETS", "HIGHWAY_PRISMION_SIDEPOCKETS", "VRAXION_LITE_LOOP"})
CONDITIONED_ARMS = tuple(arm for arm in LOOP_ARMS if "CONDITIONED" in arm)


@dataclass(frozen=True)
class Example:
    case_id: str
    tokens: tuple[str, ...]
    label: int
    tags: tuple[str, ...]
    active_symbol: int
    blocked_symbol: int
    entity_count: int
    mutated: int
    scope_active: int
    split: str


@dataclass(frozen=True)
class Config:
    arm: str
    width: int
    side_count: int
    settling_steps: int
    gate: str
    decay: str


@dataclass(frozen=True)
class JobResult:
    arm: str
    width: int
    side_count: int
    settling_steps: int
    gate: str
    decay: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STABLE_LOOP_ATTRACTOR_SWEEP_001 stable-loop topology probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("smoke", "valid_slice", "full_survivors"), default="smoke")
    parser.add_argument("--from", dest="from_path", type=Path, default=None)
    parser.add_argument("--seeds", default="2026,2027")
    parser.add_argument("--arms", default="")
    parser.add_argument("--widths", default="")
    parser.add_argument("--side-counts", default="")
    parser.add_argument("--settling-steps", default="")
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", default="auto50")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in parse_csv(spec):
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def resolve_jobs(spec: str) -> int:
    cpu = os.cpu_count() or 1
    lowered = spec.lower()
    if lowered.startswith("auto"):
        suffix = lowered.removeprefix("auto")
        percent = int(suffix) if suffix else 50
        return max(1, min(cpu, math.floor(cpu * percent / 100.0)))
    return max(1, int(spec))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def set_worker(seed: int, device: str) -> torch.device:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    set_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def symbol_index(symbol: str | None) -> int:
    return 0 if symbol is None else SYMBOLS.index(symbol) + 1


def tick_scopes(scopes: dict[str, int]) -> dict[str, int]:
    return {sym: ttl - 1 for sym, ttl in scopes.items() if ttl - 1 > 0}


def interpret(tokens: tuple[str, ...]) -> tuple[int, int, int, int, int, int, tuple[str, ...]]:
    active: str | None = None
    scopes: dict[str, int] = {}
    count = 0
    removed = 0
    query = False
    mutated = 0
    tags: list[str] = []
    for token in tokens:
        if token == "delay":
            scopes = tick_scopes(scopes)
            tags.append("scope")
            continue
        if token == "reset":
            scopes.clear()
            tags.append("scope")
            continue
        if token in SYMBOLS:
            if scopes.get(token, 0) > 0:
                tags.append("scope")
            else:
                active = token
            scopes = tick_scopes(scopes)
            continue
        if token.startswith("anti_"):
            sym = token.removeprefix("anti_")
            if active == sym:
                active = None
                tags.append("cancellation")
            scopes[sym] = 2
            scopes = tick_scopes(scopes)
            continue
        if token.startswith("mention_"):
            tags.extend(["mention_noop", "no_mutation"])
            scopes = tick_scopes(scopes)
            continue
        if token.startswith("quote_anti_"):
            tags.extend(["mention_noop", "no_mutation"])
            scopes = tick_scopes(scopes)
            continue
        if token.startswith("actually_"):
            active = token.removeprefix("actually_")
            tags.append("refocus")
            scopes = tick_scopes(scopes)
            continue
        if token.startswith("instead_"):
            active = token.removeprefix("instead_")
            tags.append("refocus")
            scopes = tick_scopes(scopes)
            continue
        if token == "create_X":
            count = min(4, count + 1)
            active = None
            mutated = 1
            tags.append("entity_count")
            scopes = tick_scopes(scopes)
            continue
        if token == "remove_X":
            if count > 0:
                count -= 1
                removed = min(4, removed + 1)
                mutated = 1
            else:
                tags.append("no_mutation")
            active = None
            tags.append("entity_count")
            scopes = tick_scopes(scopes)
            continue
        if token == "restore_X":
            if removed > 0:
                removed -= 1
                count = min(4, count + 1)
                mutated = 1
            else:
                tags.extend(["no_mutation", "invalid_restore"])
            active = None
            tags.append("entity_count")
            scopes = tick_scopes(scopes)
            continue
        if token == "query_count":
            query = True
            tags.append("entity_count")
            scopes = tick_scopes(scopes)
            continue
        if token == "noise":
            tags.append("noise")
            scopes = tick_scopes(scopes)
            continue
        raise ValueError(f"unknown token: {token}")
    label = LABEL_TO_ID[f"COUNT{count}"] if query else LABEL_TO_ID[active or "NONE"]
    blocked = sorted(scopes)[0] if scopes else None
    return label, symbol_index(active), symbol_index(blocked), count, mutated, int(bool(scopes)), tuple(sorted(set(tags)))


def make_example(case_id: str, tokens: list[str], split: str, tags: tuple[str, ...] = ()) -> Example:
    label, active, blocked, count, mutated, scope_active, interp_tags = interpret(tuple(tokens))
    all_tags = tuple(sorted(set(tags + interp_tags)))
    return Example(case_id, tuple(tokens), label, all_tags, active, blocked, count, mutated, scope_active, split)


def base_patterns(split: str) -> list[Example]:
    rows: list[Example] = []
    symbols = ["A", "B"] if split == "train" else ["A", "B", "C", "D"]
    for idx, sym in enumerate(symbols):
        other = symbols[(idx + 1) % len(symbols)]
        third = symbols[(idx + 2) % len(symbols)]
        rows.extend(
            [
                make_example(f"{split}_single_{sym}", [sym], split, ("simple",)),
                make_example(f"{split}_cancel_{sym}", [sym, f"anti_{sym}"], split, ("cancellation", "train_composition")),
                make_example(f"{split}_cancel_refocus_{sym}", [sym, f"anti_{sym}", other], split, ("cancellation", "refocus", "heldout_composition")),
                make_example(f"{split}_double_cancel_{sym}", [sym, f"anti_{sym}", other, f"anti_{other}"], split, ("cancellation", "heldout_composition")),
                make_example(f"{split}_blocked_{sym}", [f"anti_{sym}", sym], split, ("scope", "no_mutation")),
                make_example(f"{split}_reset_{sym}", [f"anti_{sym}", "reset", sym], split, ("scope", "heldout_composition")),
                make_example(f"{split}_delay_cancel_{sym}", [sym, "delay", f"anti_{sym}"], split, ("scope", "cancellation")),
                make_example(f"{split}_reset_breaks_{sym}", [sym, "reset", f"anti_{sym}"], split, ("scope", "heldout_composition")),
                make_example(f"{split}_actually_{sym}", [other, f"actually_{sym}"], split, ("refocus",)),
                make_example(f"{split}_instead_{sym}", [other, f"instead_{sym}"], split, ("refocus",)),
                make_example(f"{split}_anti_actually_{sym}", [other, f"anti_{other}", f"actually_{sym}"], split, ("refocus", "cancellation", "heldout_composition")),
                make_example(f"{split}_two_refocus_{sym}", [sym, other, f"actually_{third}"], split, ("refocus", "heldout_composition")),
                make_example(f"{split}_mention_{sym}", [f"mention_{sym}"], split, ("mention_noop", "no_mutation")),
                make_example(f"{split}_mention_then_{sym}", [f"mention_{sym}", other], split, ("mention_noop", "heldout_composition")),
                make_example(f"{split}_quote_anti_{sym}", [f"quote_anti_{sym}", sym], split, ("mention_noop",)),
                make_example(f"{split}_mention_anti_{sym}", [f"mention_{sym}", f"anti_{sym}", other], split, ("mention_noop", "heldout_composition")),
            ]
        )
    rows.extend(
        [
            make_example(f"{split}_count_create", ["create_X", "query_count"], split, ("entity_count",)),
            make_example(f"{split}_count_remove", ["create_X", "remove_X", "query_count"], split, ("entity_count",)),
            make_example(f"{split}_count_restore", ["create_X", "remove_X", "restore_X", "query_count"], split, ("entity_count", "heldout_composition")),
            make_example(f"{split}_count_two_remove", ["create_X", "create_X", "remove_X", "query_count"], split, ("entity_count", "heldout_composition")),
            make_example(f"{split}_count_invalid_restore", ["create_X", "restore_X", "query_count"], split, ("entity_count", "invalid_restore", "no_mutation")),
        ]
    )
    return rows


def random_sequence(rng: np.random.Generator, split: str, idx: int) -> Example:
    symbols = ["A", "B"] if split == "train" else ["A", "B", "C", "D"]
    length = int(rng.integers(1, 4 if split == "train" else 8))
    if rng.random() < 0.22:
        ops = ["create_X", "remove_X", "restore_X", "noise", "delay"]
        tokens = [str(rng.choice(ops, p=[0.42, 0.22, 0.16, 0.12, 0.08])) for _ in range(max(1, length - 1))]
        tokens.append("query_count")
        tags = ["entity_count"]
        if len(tokens) >= 4:
            tags.append("length_generalization")
        return make_example(f"{split}_random_entity_{idx}", tokens, split, tuple(tags))
    families = ["sym", "anti", "reset", "delay", "mention", "quote", "actual", "instead", "noise"]
    probs = [0.23, 0.17, 0.07, 0.08, 0.11, 0.07, 0.11, 0.10, 0.06]
    tokens = []
    tags = []
    for _ in range(length):
        sym = str(rng.choice(symbols))
        family = str(rng.choice(families, p=probs))
        if family == "sym":
            tokens.append(sym)
        elif family == "anti":
            tokens.append(f"anti_{sym}")
            tags.append("cancellation")
        elif family == "reset":
            tokens.append("reset")
            tags.append("scope")
        elif family == "delay":
            tokens.append("delay")
            tags.append("scope")
        elif family == "mention":
            tokens.append(f"mention_{sym}")
            tags.append("mention_noop")
        elif family == "quote":
            tokens.append(f"quote_anti_{sym}")
            tags.append("mention_noop")
        elif family == "actual":
            tokens.append(f"actually_{sym}")
            tags.append("refocus")
        elif family == "instead":
            tokens.append(f"instead_{sym}")
            tags.append("refocus")
        else:
            tokens.append("noise")
    if split != "train" and (len(tokens) >= 4 or any(token.endswith("_C") or token.endswith("_D") or token in {"C", "D"} for token in tokens)):
        tags.append("heldout_composition")
    if len(tokens) >= 4:
        tags.append("length_generalization")
    return make_example(f"{split}_random_symbol_{idx}", tokens, split, tuple(tags))


def repeat_to_count(pool: list[Example], count: int, prefix: str) -> list[Example]:
    out = []
    for idx in range(count):
        src = pool[idx % len(pool)]
        out.append(Example(f"{prefix}_{idx}_{src.case_id}", src.tokens, src.label, src.tags, src.active_symbol, src.blocked_symbol, src.entity_count, src.mutated, src.scope_active, src.split))
    return out


def build_dataset(seed: int, train_examples: int, eval_examples: int, family_filter: str | None = None) -> tuple[list[Example], list[Example]]:
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1009)
    train_pool = base_patterns("train")
    eval_pool = base_patterns("eval")
    train_pool.extend(random_sequence(rng_train, "train", idx) for idx in range(max(128, train_examples // 2)))
    eval_pool.extend(random_sequence(rng_eval, "eval", idx) for idx in range(max(128, eval_examples // 2)))
    if family_filter is not None:
        train_pool = [row for row in train_pool if family_filter in row.tags]
        eval_pool = [row for row in eval_pool if family_filter in row.tags]
    rng_train.shuffle(train_pool)
    rng_eval.shuffle(eval_pool)
    return repeat_to_count(train_pool, train_examples, f"train_{family_filter or 'all'}"), repeat_to_count(eval_pool, eval_examples, f"eval_{family_filter or 'all'}")


def max_len(rows: list[Example]) -> int:
    return max(len(row.tokens) for row in rows)


def encode(rows: list[Example], length: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        ids = [TOKEN_TO_ID[token] for token in row.tokens[:length]]
        x[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return x


def labels(rows: list[Example], device: torch.device) -> torch.Tensor:
    return torch.tensor([row.label for row in rows], dtype=torch.long, device=device)


class SummaryEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.embed = nn.Embedding(len(TOKENS), width, padding_idx=TOKEN_TO_ID[PAD])
        self.mix = nn.Sequential(nn.Linear(width * 3, width), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        mask = (x != TOKEN_TO_ID[PAD]).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = summed / denom
        maxed = emb.masked_fill(mask == 0, -1e4).max(dim=1).values
        maxed = torch.where(torch.isfinite(maxed), maxed, torch.zeros_like(maxed))
        last_idx = mask.squeeze(-1).sum(dim=1).long().sub(1).clamp_min(0)
        last = emb[torch.arange(x.shape[0], device=x.device), last_idx]
        return self.mix(torch.cat([mean, maxed, last], dim=-1))


class SummaryDirectHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.encoder = SummaryEncoder(width)
        self.out = nn.Linear(width, len(LABELS))

    def represent(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.out(self.represent(x))


class StaticMLP(nn.Module):
    def __init__(self, width: int, deep: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(len(TOKENS), width * 2), nn.ReLU()]
        repeat = 4 if deep else 1
        dim = width * 2
        for _ in range(repeat):
            layers.extend([nn.Linear(dim, dim), nn.ReLU()])
        layers.append(nn.Linear(dim, width))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, len(LABELS))

    def bag(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(x, num_classes=len(TOKENS)).float()
        one_hot[:, :, TOKEN_TO_ID[PAD]] = 0.0
        return one_hot.sum(dim=1)

    def represent(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.net(self.bag(x))

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.out(self.represent(x))


class RNNClassifier(nn.Module):
    def __init__(self, kind: str, width: int, extra_noop_steps: int = 0):
        super().__init__()
        self.extra_noop_steps = extra_noop_steps
        self.embed = nn.Embedding(len(TOKENS), width, padding_idx=TOKEN_TO_ID[PAD])
        cls = {"SIMPLE_RNN": nn.RNN, "GRU": nn.GRU, "GRU_EXTRA_NOOP_STEPS": nn.GRU, "LSTM": nn.LSTM}[kind]
        self.rnn = cls(width, width, batch_first=True)
        self.out = nn.Linear(width, len(LABELS))

    def represent(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        emb = self.embed(x)
        if self.extra_noop_steps:
            noop = torch.zeros((x.shape[0], self.extra_noop_steps, emb.shape[-1]), dtype=emb.dtype, device=emb.device)
            emb = torch.cat([emb, noop], dim=1)
        _, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.out(self.represent(x))


class LoopModel(nn.Module):
    def __init__(self, arm: str, width: int, side_count: int, settling_steps: int, gate: str, decay: str):
        super().__init__()
        self.arm = arm
        self.width = width
        self.side_count = max(1, side_count)
        self.settling_steps = settling_steps
        self.gate = gate
        self.decay = decay
        self.encoder = SummaryEncoder(width)
        self.conditioned = "CONDITIONED" in arm
        self.autonomous = not self.conditioned
        self.use_gru = "LOOP_GRU" in arm
        self.use_prismion = "PRISMION" in arm
        self.use_side = arm.startswith("HIGHWAY") or arm == "VRAXION_LITE_LOOP"
        self.sparse = "SPARSE" in arm or arm == "VRAXION_LITE_LOOP"
        self.dense = "DENSE" in arm
        self.recurrent_side = "RECURRENT_SIDEPOCKETS" in arm or arm == "VRAXION_LITE_LOOP"
        in_dim = width if self.autonomous else width * 2
        if self.use_gru:
            self.cell = nn.GRUCell(width if self.autonomous else width * 2, width)
        else:
            self.cell = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.side_in = nn.ModuleList(nn.Linear(in_dim, width) for _ in range(self.side_count))
        self.side_delta = nn.ModuleList(nn.Linear(width, width) for _ in range(self.side_count))
        self.side_gate = nn.ModuleList(nn.Linear(width, 1) for _ in range(self.side_count))
        self.side_link = nn.ModuleList(nn.Linear(width, width) for _ in range(self.side_count))
        self.side_cell = nn.ModuleList(nn.GRUCell(width, width) for _ in range(self.side_count))
        self.decay_param = nn.Parameter(torch.tensor(0.0))
        if self.use_prismion:
            self.phase = nn.Parameter(torch.randn(self.side_count, width) * 0.5)
            self.prism_gain = nn.Parameter(torch.zeros(self.side_count, width))
        self.out = nn.Linear(width, len(LABELS))
        self.last_trace: list[torch.Tensor] = []
        self.last_gate_values: list[torch.Tensor] = []

    def step_input(self, h: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return h if self.autonomous else torch.cat([h, summary], dim=-1)

    def decay_value(self) -> torch.Tensor | float:
        if self.decay == "fixed_decay_0.9":
            return 0.9
        if self.decay == "learned_decay":
            return torch.sigmoid(self.decay_param)
        return 1.0

    def side_step(self, h: torch.Tensor, summary: torch.Tensor, side_state: list[torch.Tensor] | None, force_gates: str | None = None) -> tuple[torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor]]:
        shared = self.step_input(h, summary)
        feats = [torch.tanh(layer(shared)) for layer in self.side_in]
        if self.sparse:
            feats = [torch.tanh(feats[i] + self.side_link[i](feats[(i - 1) % self.side_count])) for i in range(self.side_count)]
        elif self.dense:
            mean_feat = torch.stack(feats, dim=0).mean(dim=0)
            feats = [torch.tanh(feats[i] + self.side_link[i](mean_feat)) for i in range(self.side_count)]
        if self.recurrent_side:
            if side_state is None:
                side_state = [torch.zeros_like(h) for _ in range(self.side_count)]
            side_state = [self.side_cell[i](feats[i], side_state[i]) for i in range(self.side_count)]
            feats = side_state
        gates = []
        updates = []
        for i, feat in enumerate(feats):
            gate = torch.sigmoid(self.side_gate[i](feat))
            if force_gates == "zero":
                gate = torch.zeros_like(gate)
            elif force_gates == "open":
                gate = torch.ones_like(gate)
            if self.use_prismion:
                gain = F.softplus(self.prism_gain[i]).unsqueeze(0)
                feat = torch.cos(feat + self.phase[i].unsqueeze(0)).pow(2) * gain
            updates.append(gate * torch.tanh(self.side_delta[i](feat)))
            gates.append(gate)
        delta = torch.stack(updates, dim=0).sum(dim=0) / math.sqrt(self.side_count)
        return delta, side_state, gates

    def settle(self, summary: torch.Tensor, steps: int | None = None, noise_std: float = 0.0, force_gates: str | None = None) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        steps = self.settling_steps if steps is None else steps
        h = summary
        if noise_std:
            h = h + torch.randn_like(h) * noise_std
        trace = [h]
        gate_values: list[torch.Tensor] = []
        side_state: list[torch.Tensor] | None = None
        for _ in range(steps):
            if self.use_side:
                delta, side_state, gates = self.side_step(h, summary, side_state, force_gates)
                new_h = torch.tanh(h * self.decay_value() + delta)
                gate_values.extend(gates)
            elif self.use_gru:
                new_h = self.cell(self.step_input(h, summary), h)
            else:
                new_h = torch.tanh(h * self.decay_value() + self.cell(self.step_input(h, summary)))
            h = new_h
            trace.append(h)
        return h, trace, gate_values

    def represent(self, x: torch.Tensor, steps: int | None = None, noise_std: float = 0.0, force_gates: str | None = None, collect_trace: bool = False) -> torch.Tensor:
        summary = self.encoder(x)
        h, trace, gates = self.settle(summary, steps, noise_std, force_gates)
        if collect_trace:
            self.last_trace = [item.detach() for item in trace]
            self.last_gate_values = [item.detach() for item in gates]
        return h

    def forward(self, x: torch.Tensor, steps: int | None = None, noise_std: float = 0.0, force_gates: str | None = None, collect_trace: bool = False) -> torch.Tensor:
        return self.out(self.represent(x, steps, noise_std, force_gates, collect_trace))


def make_model(config: Config, device: torch.device) -> nn.Module:
    arm = config.arm
    if arm == "MLP_STATIC":
        model: nn.Module = StaticMLP(config.width, deep=False)
    elif arm == "DEEP_MLP_MATCHED_COMPUTE":
        model = StaticMLP(config.width, deep=True)
    elif arm == "SUMMARY_DIRECT_HEAD":
        model = SummaryDirectHead(config.width)
    elif arm in {"SIMPLE_RNN", "GRU", "GRU_EXTRA_NOOP_STEPS", "LSTM"}:
        model = RNNClassifier(arm, config.width, extra_noop_steps=config.settling_steps if arm == "GRU_EXTRA_NOOP_STEPS" else 0)
    else:
        model = LoopModel(arm, config.width, config.side_count, config.settling_steps, config.gate, config.decay)
    return model.to(device)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def accuracy(preds: list[int], gold: list[int]) -> float:
    return sum(int(p == g) for p, g in zip(preds, gold)) / len(gold) if gold else math.nan


def tag_accuracy(rows: list[Example], preds: list[int], tag: str) -> float:
    pairs = [(pred, row.label) for pred, row in zip(preds, rows) if tag in row.tags]
    return accuracy([p for p, _ in pairs], [g for _, g in pairs])


def tag_error_rate(rows: list[Example], preds: list[int], tag: str) -> float:
    acc = tag_accuracy(rows, preds, tag)
    return math.nan if math.isnan(acc) else 1.0 - acc


def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 1024, **kwargs: object) -> list[int]:
    out: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits = model(x[start : start + batch_size], **kwargs)
            out.extend(logits.argmax(dim=1).cpu().tolist())
    return out


def train_model(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor, seed: int, args: argparse.Namespace, progress_path: Path | None) -> tuple[nn.Module, int | None]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 77)
    epochs_to_threshold: int | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in batch_indices(train_x.shape[0], args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            preds = predict(model, eval_x)
            eval_acc = accuracy(preds, eval_y.cpu().tolist())
            if epochs_to_threshold is None and eval_acc >= 0.90:
                epochs_to_threshold = epoch
            if progress_path is not None:
                append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch, "loss": float(np.mean(losses)), "eval_accuracy": eval_acc})
    return model, epochs_to_threshold


def represent(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    reps = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 1024):
            reps.append(model.represent(x[start : start + 1024]).detach())
    return torch.cat(reps, dim=0)


def train_linear_probe(train_h: torch.Tensor, train_y: torch.Tensor, eval_h: torch.Tensor, eval_y: torch.Tensor, classes: int, seed: int, hidden: bool = False) -> float:
    if classes <= 1:
        return math.nan
    set_seed(seed)
    if hidden:
        probe = nn.Sequential(nn.Linear(train_h.shape[1], train_h.shape[1]), nn.ReLU(), nn.Linear(train_h.shape[1], classes)).to(train_h.device)
    else:
        probe = nn.Linear(train_h.shape[1], classes).to(train_h.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=0.02, weight_decay=1e-4)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_h), train_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(eval_h).argmax(dim=1)
    return float((preds == eval_y).float().mean().cpu())


def probe_metrics(model: nn.Module, train_x: torch.Tensor, eval_x: torch.Tensor, train_rows: list[Example], eval_rows: list[Example], seed: int, device: torch.device) -> dict[str, float]:
    train_h = represent(model, train_x)
    eval_h = represent(model, eval_x)
    targets = {
        "active_symbol": ([row.active_symbol for row in train_rows], [row.active_symbol for row in eval_rows], 5),
        "blocked_symbol": ([row.blocked_symbol for row in train_rows], [row.blocked_symbol for row in eval_rows], 5),
        "current_focus": ([row.active_symbol for row in train_rows], [row.active_symbol for row in eval_rows], 5),
        "entity_count": ([row.entity_count for row in train_rows], [row.entity_count for row in eval_rows], 5),
        "mutation_no_mutation": ([row.mutated for row in train_rows], [row.mutated for row in eval_rows], 2),
        "scope_active": ([row.scope_active for row in train_rows], [row.scope_active for row in eval_rows], 2),
    }
    linear_scores = []
    mlp_scores = []
    out: dict[str, float] = {}
    for idx, (name, (tr, ev, classes)) in enumerate(targets.items()):
        tr_y = torch.tensor(tr, dtype=torch.long, device=device)
        ev_y = torch.tensor(ev, dtype=torch.long, device=device)
        lin = train_linear_probe(train_h, tr_y, eval_h, ev_y, classes, seed + 500 + idx, hidden=False)
        mlp = train_linear_probe(train_h, tr_y, eval_h, ev_y, classes, seed + 700 + idx, hidden=True)
        out[f"{name}_linear_probe_accuracy"] = lin
        out[f"{name}_mlp_probe_accuracy"] = mlp
        linear_scores.append(lin)
        mlp_scores.append(mlp)
    out["linear_probe_accuracy"] = float(np.nanmean(linear_scores))
    out["MLP_probe_accuracy_separate"] = float(np.nanmean(mlp_scores))
    return out


def loop_dynamics(model: nn.Module, config: Config, eval_x: torch.Tensor, eval_rows: list[Example], base_preds: list[int]) -> dict[str, float]:
    if not isinstance(model, LoopModel):
        return {
            "settling_gain": math.nan,
            "final_state_delta": math.nan,
            "convergence_rate": math.nan,
            "overrun_stability": math.nan,
            "noise_recovery_accuracy": math.nan,
        }
    gold = [row.label for row in eval_rows]
    s1 = accuracy(predict(model, eval_x, steps=1), gold)
    sN = accuracy(base_preds, gold)
    over = predict(model, eval_x, steps=min(16, config.settling_steps + 4))
    noisy = predict(model, eval_x, noise_std=0.10)
    with torch.no_grad():
        _ = model(eval_x[: min(512, eval_x.shape[0])], collect_trace=True)
        trace = model.last_trace
        deltas = [float(torch.norm(trace[i] - trace[i - 1], dim=1).mean().cpu()) for i in range(1, len(trace))]
    final_delta = float(np.mean(deltas[-3:])) if deltas else math.nan
    first_delta = deltas[0] if deltas else math.nan
    convergence = 1.0 - (final_delta / (first_delta + 1e-6)) if not math.isnan(final_delta) and not math.isnan(first_delta) else math.nan
    return {
        "settling_gain": sN - s1,
        "final_state_delta": final_delta,
        "convergence_rate": convergence,
        "overrun_stability": accuracy(over, base_preds),
        "noise_recovery_accuracy": accuracy(noisy, base_preds),
    }


def continuation_metrics(config: Config, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if config.arm not in LOOP_ARMS:
        return {"retention_after_new_training": math.nan, "catastrophic_interference_rate": math.nan, "stable_function_retention": math.nan}
    tiny_args = argparse.Namespace(**vars(args))
    tiny_args.epochs = max(8, args.epochs // 4)
    tiny_args.train_examples = min(args.train_examples, 1024)
    tiny_args.eval_examples = min(args.eval_examples, 1024)
    model = make_model(config, device)
    retained = []
    previous_scores = []
    families = ["cancellation", "refocus", "entity_count"]
    eval_sets: dict[str, tuple[list[Example], torch.Tensor, torch.Tensor]] = {}
    length = 10
    for idx, family in enumerate(families):
        train_rows, eval_rows = build_dataset(seed + 900 + idx, tiny_args.train_examples, tiny_args.eval_examples, family)
        length = max(length, max_len(train_rows), max_len(eval_rows))
        train_x = encode(train_rows, length, device)
        eval_x = encode(eval_rows, length, device)
        train_y = labels(train_rows, device)
        eval_y = labels(eval_rows, device)
        model, _ = train_model(model, train_x, train_y, eval_x, eval_y, seed + 3000 + idx, tiny_args, None)
        eval_sets[family] = (eval_rows, eval_x, eval_y)
        current_scores = []
        for seen_family in families[: idx + 1]:
            rows, x, y = eval_sets[seen_family]
            current_scores.append(accuracy(predict(model, x), y.cpu().tolist()))
        if idx > 0:
            retained.append(float(np.mean(current_scores[:-1])))
            previous_scores.extend(current_scores[:-1])
    retention = float(np.mean(retained)) if retained else math.nan
    catastrophic = 1.0 - retention if not math.isnan(retention) else math.nan
    return {
        "retention_after_new_training": retention,
        "catastrophic_interference_rate": catastrophic,
        "stable_function_retention": retention,
    }


def failed_cases(rows: list[Example], preds: list[int], limit: int = 12) -> list[dict[str, object]]:
    out = []
    for row, pred in zip(rows, preds):
        if pred != row.label:
            out.append({"case_id": row.case_id, "tokens": row.tokens, "gold": LABELS[row.label], "pred": LABELS[pred], "tags": row.tags})
            if len(out) >= limit:
                break
    return out


def run_job(config: Config, seed: int, args: argparse.Namespace, progress_root: Path | None) -> JobResult:
    device = set_worker(seed, args.device)
    progress_path = progress_root / f"{config.arm}__w{config.width}__k{config.side_count}__s{config.settling_steps}__{config.gate}__{config.decay}__seed{seed}.jsonl" if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", **asdict(config), "seed": seed, "device": str(device)})
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples)
    length = max(max_len(train_rows), max_len(eval_rows))
    train_x = encode(train_rows, length, device)
    eval_x = encode(eval_rows, length, device)
    train_y = labels(train_rows, device)
    eval_y = labels(eval_rows, device)
    model = make_model(config, device)
    params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    model, epochs_to_threshold = train_model(model, train_x, train_y, eval_x, eval_y, seed, args, progress_path)
    preds = predict(model, eval_x)
    gold = eval_y.cpu().tolist()
    metrics: dict[str, object] = {
        "final_answer_accuracy": accuracy(preds, gold),
        "heldout_composition_accuracy": tag_accuracy(eval_rows, preds, "heldout_composition"),
        "length_generalization_accuracy": tag_accuracy(eval_rows, preds, "length_generalization"),
        "false_mutation_rate": tag_error_rate(eval_rows, preds, "no_mutation"),
        "false_cancellation_rate": tag_error_rate(eval_rows, preds, "cancellation"),
        "false_refocus_rate": tag_error_rate(eval_rows, preds, "refocus"),
        "mention_noop_error_rate": tag_error_rate(eval_rows, preds, "mention_noop"),
        "scope_error_rate": tag_error_rate(eval_rows, preds, "scope"),
        "cancellation_accuracy": tag_accuracy(eval_rows, preds, "cancellation"),
        "refocus_accuracy": tag_accuracy(eval_rows, preds, "refocus"),
        "scope_accuracy": tag_accuracy(eval_rows, preds, "scope"),
        "entity_count_accuracy": tag_accuracy(eval_rows, preds, "entity_count"),
        "parameter_count": params,
        "epochs_to_threshold": epochs_to_threshold if epochs_to_threshold is not None else math.nan,
    }
    metrics.update(loop_dynamics(model, config, eval_x, eval_rows, preds))
    metrics.update(probe_metrics(model, train_x, eval_x, train_rows, eval_rows, seed, device))
    if args.stage == "full_survivors":
        metrics.update(continuation_metrics(config, seed, args, device))
    else:
        metrics.update({"retention_after_new_training": math.nan, "catastrophic_interference_rate": math.nan, "stable_function_retention": math.nan})
    return JobResult(config.arm, config.width, config.side_count, config.settling_steps, config.gate, config.decay, seed, metrics, failed_cases(eval_rows, preds))


def stage_defaults(stage: str) -> tuple[list[str], list[int], list[int], list[int]]:
    if stage == "smoke":
        return (
            ["GRU", "SUMMARY_DIRECT_HEAD", "MAIN_LOOP_MLP_AUTONOMOUS", "MAIN_LOOP_MLP_CONDITIONED", "HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS", "HIGHWAY_PRISMION_SIDEPOCKETS", "DEEP_MLP_MATCHED_COMPUTE"],
            [16],
            [0, 4],
            [1, 4],
        )
    if stage == "valid_slice":
        return (
            ["GRU", "LSTM", "SUMMARY_DIRECT_HEAD", "DEEP_MLP_MATCHED_COMPUTE", "GRU_EXTRA_NOOP_STEPS", "MAIN_LOOP_MLP_AUTONOMOUS", "MAIN_LOOP_MLP_CONDITIONED", "MAIN_LOOP_GRU_AUTONOMOUS", "MAIN_LOOP_GRU_CONDITIONED", "HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS", "HIGHWAY_RECURRENT_SIDEPOCKETS", "HIGHWAY_SPARSE_SIDEPOCKETS", "HIGHWAY_PRISMION_SIDEPOCKETS"],
            [16, 32],
            [0, 4],
            [1, 4, 8],
        )
    return ([], [], [], [])


def load_survivor_configs(path: Path) -> list[Config]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Config(str(row["arm"]), int(row["width"]), int(row["side_count"]), int(row["settling_steps"]), str(row.get("gate", "sigmoid_gate")), str(row.get("decay", "none"))) for row in rows]


def build_queue(args: argparse.Namespace) -> list[Config]:
    if args.stage == "full_survivors" and args.from_path:
        survivors = load_survivor_configs(args.from_path)
        configs = []
        for base in survivors:
            for width in (16, 32, 64):
                for steps in (2, 4, 8, 16):
                    for side_count in ((2, 4, 8) if base.arm in LOOP_ARMS and ("HIGHWAY" in base.arm or base.arm == "VRAXION_LITE_LOOP") else (0,)):
                        configs.append(Config(base.arm, width, side_count, steps, base.gate, base.decay))
        return sorted(set(configs), key=lambda c: (c.arm, c.width, c.side_count, c.settling_steps, c.gate, c.decay))
    arms, widths, side_counts, steps = stage_defaults(args.stage)
    if args.arms:
        arms = parse_csv(args.arms)
    if args.widths:
        widths = parse_int_csv(args.widths)
    if args.side_counts:
        side_counts = parse_int_csv(args.side_counts)
    if args.settling_steps:
        steps = parse_int_csv(args.settling_steps)
    configs = []
    for arm in arms:
        for width in widths:
            if arm in {"MLP_STATIC", "DEEP_MLP_MATCHED_COMPUTE", "SIMPLE_RNN", "GRU", "LSTM", "SUMMARY_DIRECT_HEAD"}:
                configs.append(Config(arm, width, 0, 1, "sigmoid_gate", "none"))
            elif arm == "GRU_EXTRA_NOOP_STEPS":
                for step in steps:
                    configs.append(Config(arm, width, 0, step, "sigmoid_gate", "none"))
            else:
                for side_count in side_counts:
                    if side_count == 0:
                        continue
                    for step in steps:
                        configs.append(Config(arm, width, side_count, step, "sigmoid_gate", "none"))
    return configs


def result_record(row: JobResult) -> dict[str, object]:
    return asdict(row)


def merge_metric_values(values: list[object]) -> object:
    if all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
        floats = [float(value) for value in values]
        return {"mean": round(float(np.mean(floats)), 6), "min": round(float(np.min(floats)), 6), "max": round(float(np.max(floats)), 6), "std": round(float(np.std(floats)), 6)}
    return values[0] if values else None


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_key: dict[tuple[str, int, int, int, str, str], list[JobResult]] = defaultdict(list)
    for row in results:
        by_key[(row.arm, row.width, row.side_count, row.settling_steps, row.gate, row.decay)].append(row)
    out: dict[str, dict[str, object]] = {}
    for key_tuple, rows in sorted(by_key.items()):
        arm, width, side_count, steps, gate, decay = key_tuple
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics = {name: merge_metric_values([row.metrics.get(name) for row in rows]) for name in metric_names}
        key = f"{arm}/w{width}/k{side_count}/s{steps}/{gate}/{decay}"
        out[key] = {"arm": arm, "width": width, "side_count": side_count, "settling_steps": steps, "gate": gate, "decay": decay, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(row: dict[str, object], name: str, default: float = math.nan) -> float:
    value = row["metrics"].get(name) if isinstance(row.get("metrics"), dict) else None
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    if isinstance(value, (int, float)):
        return float(value)
    return default


def score(row: dict[str, object]) -> float:
    return float(np.nanmean([metric_mean(row, "heldout_composition_accuracy"), metric_mean(row, "length_generalization_accuracy")]))


def best(agg: dict[str, dict[str, object]], arms: set[str]) -> dict[str, object] | None:
    rows = [row for row in agg.values() if str(row["arm"]) in arms]
    return max(rows, key=score, default=None)


def survivor_configs(agg: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = list(agg.values())
    if not rows:
        return []
    best_heldout = max(metric_mean(row, "heldout_composition_accuracy", -1.0) for row in rows)
    best_length = max(metric_mean(row, "length_generalization_accuracy", -1.0) for row in rows)
    survivors = []
    for row in rows:
        if metric_mean(row, "heldout_composition_accuracy", -1.0) >= best_heldout - 0.05 or metric_mean(row, "length_generalization_accuracy", -1.0) >= best_length - 0.05:
            survivors.append({k: row[k] for k in ("arm", "width", "side_count", "settling_steps", "gate", "decay")})
    return survivors


def control_gaps(agg: dict[str, dict[str, object]]) -> dict[str, float]:
    loop_rows = [row for row in agg.values() if str(row["arm"]) in LOOP_ARMS]
    auto_rows = [row for row in loop_rows if str(row["arm"]) in AUTONOMOUS_ARMS]
    cond_rows = [row for row in loop_rows if str(row["arm"]) in CONDITIONED_ARMS]
    best_loop = max(loop_rows, key=score, default=None)
    best_auto = max(auto_rows, key=score, default=None)
    best_cond = max(cond_rows, key=score, default=None)
    best_summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    best_compute = best(agg, {"DEEP_MLP_MATCHED_COMPUTE", "GRU_EXTRA_NOOP_STEPS"})
    return {
        "summary_direct_gap": round(score(best_loop) - score(best_summary), 6) if best_loop and best_summary else math.nan,
        "matched_compute_gap": round(score(best_loop) - score(best_compute), 6) if best_loop and best_compute else math.nan,
        "autonomous_vs_conditioned_gap": round(score(best_auto) - score(best_cond), 6) if best_auto and best_cond else math.nan,
    }


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    rnn = best(agg, {"GRU", "LSTM"})
    summary = best(agg, {"SUMMARY_DIRECT_HEAD"})
    matched = best(agg, {"DEEP_MLP_MATCHED_COMPUTE", "GRU_EXTRA_NOOP_STEPS"})
    loops = [row for row in agg.values() if str(row["arm"]) in LOOP_ARMS]
    auto = [row for row in loops if str(row["arm"]) in AUTONOMOUS_ARMS]
    cond = [row for row in loops if "CONDITIONED" in str(row["arm"])]
    loop_best = max(loops, key=score, default=None)
    auto_best = max(auto, key=score, default=None)
    cond_best = max(cond, key=score, default=None)
    static = best(agg, {"MLP_STATIC"})
    all_best = max(agg.values(), key=lambda row: metric_mean(row, "final_answer_accuracy", -1.0), default=None)
    if static is not None and score(static) >= 0.90:
        labels.append("TASK_TOO_EASY")
    if all_best is not None and metric_mean(all_best, "final_answer_accuracy", 0.0) < 0.45:
        labels.append("TASK_TOO_HARD")
    if loop_best is not None and summary is not None and score(summary) >= score(loop_best) - 0.03:
        labels.append("SUMMARY_SOLVES_TASK")
    if loop_best is not None and matched is not None and score(matched) >= score(loop_best) - 0.03:
        labels.append("COMPUTE_BUDGET_CONFONDED")
    if auto_best is not None and cond_best is not None and score(cond_best) > score(auto_best) + 0.05:
        labels.append("CONDITIONED_LOOP_ONLY")
    if auto_best is not None and rnn is not None:
        settling = metric_mean(auto_best, "settling_gain", 0.0)
        overrun = metric_mean(auto_best, "overrun_stability", 0.0)
        noise = metric_mean(auto_best, "noise_recovery_accuracy", 0.0)
        if score(auto_best) >= score(rnn) + 0.10 and settling > 0.0 and overrun >= 0.90 and noise >= 0.85:
            labels.append("STABLE_LOOP_POSITIVE")
    if rnn is not None and loop_best is not None and score(rnn) >= score(loop_best) - 0.03:
        labels.append("STANDARD_RNN_SUFFICIENT")
    if loop_best is not None and metric_mean(loop_best, "settling_gain", 0.0) < -0.03:
        labels.append("LOOP_UNSTABLE")
    if loop_best is not None and metric_mean(loop_best, "stable_function_retention", math.nan) >= 0.85:
        labels.append("STABLE_FUNCTION_RETENTION_POSITIVE")
    if loop_best is not None and metric_mean(loop_best, "catastrophic_interference_rate", 0.0) > 0.25:
        labels.append("CATASTROPHIC_INTERFERENCE_WARNING")
    if any(str(row["arm"]).startswith("HIGHWAY") and rnn is not None and score(row) >= score(rnn) + 0.10 for row in loops):
        labels.append("HIGHWAY_TOPOLOGY_POSITIVE")
    if any("SPARSE" in str(row["arm"]) and loop_best is row for row in loops):
        labels.append("SPARSE_COORDINATION_POSITIVE")
    if any("DENSE" in str(row["arm"]) and loop_best is row for row in loops):
        labels.append("DENSE_MONOLITH_WARNING")
    prism = best(agg, {"HIGHWAY_PRISMION_SIDEPOCKETS"})
    ff = best(agg, {"HIGHWAY_FF_SIDEPOCKETS_AUTONOMOUS", "HIGHWAY_RECURRENT_SIDEPOCKETS", "HIGHWAY_SPARSE_SIDEPOCKETS"})
    if prism is not None and ff is not None:
        prism_focus = float(np.nanmean([metric_mean(prism, "cancellation_accuracy"), metric_mean(prism, "refocus_accuracy"), metric_mean(prism, "scope_accuracy")]))
        ff_focus = float(np.nanmean([metric_mean(ff, "cancellation_accuracy"), metric_mean(ff, "refocus_accuracy"), metric_mean(ff, "scope_accuracy")]))
        if prism_focus >= ff_focus + 0.05 and score(prism) >= score(ff):
            labels.append("PRISMION_UPDATE_POSITIVE")
    if any(str(row["arm"]) == "VRAXION_LITE_LOOP" and rnn is not None and score(row) >= score(rnn) + 0.10 for row in loops):
        labels.append("VRAXION_LITE_POSITIVE")
    return sorted(set(labels or ["STABLE_LOOP_PARTIAL"]))


def write_curves(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    conv = []
    probes = []
    for key, row in sorted(agg.items()):
        base = {k: row[k] for k in ("arm", "width", "side_count", "settling_steps", "gate", "decay")}
        conv.append({**base, "key": key, "settling_gain": metric_mean(row, "settling_gain"), "final_state_delta": metric_mean(row, "final_state_delta"), "convergence_rate": metric_mean(row, "convergence_rate"), "overrun_stability": metric_mean(row, "overrun_stability"), "noise_recovery_accuracy": metric_mean(row, "noise_recovery_accuracy")})
        probes.append({**base, "key": key, "linear_probe_accuracy": metric_mean(row, "linear_probe_accuracy"), "MLP_probe_accuracy_separate": metric_mean(row, "MLP_probe_accuracy_separate")})
    write_jsonl(out_dir / "convergence_curves.jsonl", conv)
    write_jsonl(out_dir / "probe_results.jsonl", probes)
    write_jsonl(out_dir / "ablation_results.jsonl", [])
    write_jsonl(out_dir / "continuation_results.jsonl", [{**{k: row[k] for k in ("arm", "width", "side_count", "settling_steps")}, "retention_after_new_training": metric_mean(row, "retention_after_new_training"), "catastrophic_interference_rate": metric_mean(row, "catastrophic_interference_rate"), "stable_function_retention": metric_mean(row, "stable_function_retention")} for row in agg.values()])


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    survivors = survivor_configs(agg)
    gaps = control_gaps(agg)
    write_json(out_dir / "survivor_configs.json", survivors)
    write_curves(out_dir, agg)
    summary = {"status": status, "verdict": labels, "completed_jobs": len(results), "config": {"stage": args.stage, "seeds": args.seeds, "epochs": args.epochs, "jobs": jobs, "device": args.device, "os_cpu_count": os.cpu_count(), "torch_threads_per_worker": 1}, "control_gaps": gaps, "survivor_count": len(survivors), "aggregate": agg}
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# STABLE_LOOP_ATTRACTOR_SWEEP_001 Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Survivor configs: `{len(survivors)}`",
        "",
        "| Arm | W | K | S | Final | Heldout | Length | SettleGain | Delta | Overrun | Noise | LinearProbe | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["side_count"]), int(r["settling_steps"]))):
        lines.append("| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(row["arm"], row["width"], row["side_count"], row["settling_steps"], metric_mean(row, "final_answer_accuracy"), metric_mean(row, "heldout_composition_accuracy"), metric_mean(row, "length_generalization_accuracy"), metric_mean(row, "settling_gain"), metric_mean(row, "final_state_delta"), metric_mean(row, "overrun_stability"), metric_mean(row, "noise_recovery_accuracy"), metric_mean(row, "linear_probe_accuracy"), metric_mean(row, "parameter_count")))
    lines.extend([
        "",
        "## Control Gaps",
        "",
        f"- summary_direct_gap: `{gaps['summary_direct_gap']:.3f}`",
        f"- matched_compute_gap: `{gaps['matched_compute_gap']:.3f}`",
        f"- autonomous_vs_conditioned_gap: `{gaps['autonomous_vs_conditioned_gap']:.3f}`",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int, completed: int) -> None:
    gaps = control_gaps(agg)
    lines = [
        "# STABLE_LOOP_ATTRACTOR_SWEEP_001 Result",
        "",
        "## Run",
        "",
        "```text",
        f"stage={args.stage}",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        f"device={args.device}",
        f"completed_jobs={completed}",
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Summary Table",
        "",
        "| Arm | W | K | S | Final | Heldout | Length | SettleGain | Overrun | Noise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["side_count"]), int(r["settling_steps"]))):
        lines.append("| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(row["arm"], row["width"], row["side_count"], row["settling_steps"], metric_mean(row, "final_answer_accuracy"), metric_mean(row, "heldout_composition_accuracy"), metric_mean(row, "length_generalization_accuracy"), metric_mean(row, "settling_gain"), metric_mean(row, "overrun_stability"), metric_mean(row, "noise_recovery_accuracy")))
    lines.extend([
        "",
        "## Control Gaps",
        "",
        "```json",
        json.dumps(gaps, indent=2),
        "```",
    ])
    lines.extend(["", "## Claim Boundary", "", "This is an abstract symbolic stable-loop probe. It is not a parser, factuality system, language benchmark, consciousness claim, or full VRAXION architecture test."])
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(out_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    train, eval_rows = build_dataset(2026, min(args.train_examples, 24), min(args.eval_examples, 48))
    write_jsonl(out_dir / "examples_sample.jsonl", [{"case_id": row.case_id, "tokens": row.tokens, "label": LABELS[row.label], "tags": row.tags, "split": row.split} for row in train[:12] + eval_rows[:24]])


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, object]], list[str], list[JobResult]]:
    configs = build_queue(args)
    seeds = parse_seeds(args.seeds)
    queue = [(config, seed) for config in configs for seed in seeds]
    write_json(args.out_dir / "queue.json", [{**asdict(config), "seed": seed} for config, seed in queue])
    write_samples(args.out_dir, args, torch.device("cpu"))
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "jobs": jobs, "total_jobs": len(queue), "stage": args.stage, "device": args.device})
    results: list[JobResult] = []
    write_outputs(args.out_dir, results, args, "partial", jobs)
    if jobs <= 1:
        for config, seed in queue:
            result = run_job(config, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, "partial", jobs)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = set()
            future_meta = {}
            for config, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", **asdict(config), "seed": seed})
                future = pool.submit(run_job, config, seed, args, job_progress_root)
                pending.add(future)
                future_meta[future] = (config, seed)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    continue
                for future in done:
                    config, seed = future_meta[future]
                    result = future.result()
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", **asdict(config), "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
    agg, labels = write_outputs(args.out_dir, results, args, "complete", jobs)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "verdict": labels, "completed_jobs": len(results)})
    return agg, labels, results


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")
    agg, labels, results = run_all(args, jobs)
    write_doc_result(agg, labels, args, jobs, len(results))
    print(json.dumps({"verdict": labels, "out": str(args.out_dir), "completed_jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
