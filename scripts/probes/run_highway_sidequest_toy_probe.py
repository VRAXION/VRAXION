#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
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


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "highway_sidequest_toy_001"
CONTRACT = ROOT / "docs" / "research" / "HIGHWAY_SIDEQUEST_TOY_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "HIGHWAY_SIDEQUEST_TOY_001_RESULT.md"

PAD = "<PAD>"
TOKENS = (
    PAD,
    "A",
    "B",
    "C",
    "anti_A",
    "anti_B",
    "anti_C",
    "reset",
    "mention_A",
    "mention_B",
    "mention_C",
    "quote_anti_A",
    "quote_anti_B",
    "quote_anti_C",
    "actually_A",
    "actually_B",
    "actually_C",
    "instead_A",
    "instead_B",
    "instead_C",
    "create_X",
    "remove_X",
    "restore_X",
    "query_count",
    "noise",
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
SYMBOLS = ("A", "B", "C")
LABELS = ("NONE", "A", "B", "C", "COUNT0", "COUNT1", "COUNT2", "COUNT3", "COUNT4")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
COUNT_LABEL_OFFSET = LABEL_TO_ID["COUNT0"]
ALL_ARMS = (
    "MLP_STATIC",
    "SimpleRNN",
    "GRU",
    "LSTM",
    "HIGHWAY_ONLY_SIDEPROCESSORS",
    "HIGHWAY_SPARSE_SIDE_LINKS",
    "HIGHWAY_DENSE_SIDE_LINKS",
    "HIGHWAY_PRISMION_SIDEPROCESSORS",
)
HIGHWAY_ARMS = {
    "HIGHWAY_ONLY_SIDEPROCESSORS",
    "HIGHWAY_SPARSE_SIDE_LINKS",
    "HIGHWAY_DENSE_SIDE_LINKS",
    "HIGHWAY_PRISMION_SIDEPROCESSORS",
}


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
    split: str


@dataclass(frozen=True)
class JobResult:
    arm: str
    width: int
    side_count: int
    depth: int
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIGHWAY_SIDEQUEST_TOY_001 topology probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2030")
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--widths", default="8,16,32")
    parser.add_argument("--side-counts", default="2,4,8")
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", default="1")
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
        percent = int(suffix) if suffix else 85
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


def set_worker(seed: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    set_seed(seed)


def symbol_index(symbol: str | None) -> int:
    if symbol is None:
        return 0
    return SYMBOLS.index(symbol) + 1


def interpret(tokens: tuple[str, ...]) -> tuple[int, int, int, int, tuple[str, ...]]:
    active: str | None = None
    blocked: set[str] = set()
    count = 0
    removed = 0
    query = False
    mutated = 0
    tags: list[str] = []
    for token in tokens:
        if token in SYMBOLS:
            if token not in blocked:
                active = token
            else:
                tags.append("scope")
            continue
        if token.startswith("anti_"):
            sym = token.removeprefix("anti_")
            if active == sym:
                active = None
                tags.append("cancellation")
            blocked.add(sym)
            continue
        if token == "reset":
            blocked.clear()
            tags.append("scope")
            continue
        if token.startswith("mention_"):
            tags.extend(["mention_noop", "no_mutation"])
            continue
        if token.startswith("quote_anti_"):
            tags.extend(["mention_noop", "no_mutation"])
            continue
        if token.startswith("actually_"):
            active = token.removeprefix("actually_")
            tags.append("refocus")
            continue
        if token.startswith("instead_"):
            active = token.removeprefix("instead_")
            tags.append("refocus")
            continue
        if token == "create_X":
            count = min(4, count + 1)
            active = None
            mutated = 1
            tags.append("entity_count")
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
            continue
        if token == "query_count":
            query = True
            tags.append("entity_count")
            continue
        if token == "noise":
            tags.append("noise")
            continue
        raise ValueError(f"unknown token: {token}")
    if query:
        label = LABEL_TO_ID[f"COUNT{count}"]
    else:
        label = LABEL_TO_ID[active or "NONE"]
    blocked_idx = 0
    if blocked:
        blocked_idx = symbol_index(sorted(blocked)[0])
    return label, symbol_index(active), blocked_idx, count, mutated, tuple(sorted(set(tags)))


def make_example(case_id: str, tokens: list[str], split: str, tags: tuple[str, ...] = ()) -> Example:
    label, active, blocked, count, mutated, interp_tags = interpret(tuple(tokens))
    all_tags = tuple(sorted(set(tags + interp_tags)))
    return Example(case_id, tuple(tokens), label, all_tags, active, blocked, count, mutated, split)


def base_patterns(split: str) -> list[Example]:
    rows: list[Example] = []
    symbols = ["A", "B"] if split == "train" else ["A", "B", "C"]
    idx = 0
    for sym in symbols:
        other = "B" if sym == "A" else "A"
        if sym == "C":
            other = "A"
        rows.extend(
            [
                make_example(f"{split}_single_{sym}_{idx}", [sym], split, ("simple",)),
                make_example(f"{split}_cancel_{sym}_{idx}", [sym, f"anti_{sym}"], split, ("cancellation", "heldout_composition" if sym == "C" else "train_composition")),
                make_example(f"{split}_cancel_refocus_{sym}_{idx}", [sym, f"anti_{sym}", other], split, ("cancellation", "refocus", "heldout_composition")),
                make_example(f"{split}_blocked_{sym}_{idx}", [f"anti_{sym}", sym], split, ("scope", "no_mutation")),
                make_example(f"{split}_reset_{sym}_{idx}", [f"anti_{sym}", "reset", sym], split, ("scope", "heldout_composition")),
                make_example(f"{split}_actually_{sym}_{idx}", [other, f"actually_{sym}"], split, ("refocus",)),
                make_example(f"{split}_instead_{sym}_{idx}", [other, f"instead_{sym}"], split, ("refocus",)),
                make_example(f"{split}_mention_{sym}_{idx}", [f"mention_{sym}"], split, ("mention_noop", "no_mutation")),
                make_example(f"{split}_mention_then_{sym}_{idx}", [f"mention_{sym}", other], split, ("mention_noop", "heldout_composition")),
                make_example(f"{split}_quote_anti_{sym}_{idx}", [f"quote_anti_{sym}", sym], split, ("mention_noop",)),
            ]
        )
        idx += 1
    rows.extend(
        [
            make_example(f"{split}_count_create", ["create_X", "query_count"], split, ("entity_count",)),
            make_example(f"{split}_count_remove", ["create_X", "remove_X", "query_count"], split, ("entity_count",)),
            make_example(f"{split}_count_restore", ["create_X", "remove_X", "restore_X", "query_count"], split, ("entity_count", "heldout_composition")),
            make_example(f"{split}_count_invalid_restore", ["create_X", "restore_X", "query_count"], split, ("entity_count", "invalid_restore", "no_mutation")),
            make_example(f"{split}_count_two_remove", ["create_X", "create_X", "remove_X", "query_count"], split, ("entity_count", "heldout_composition")),
        ]
    )
    return rows


def random_sequence(rng: np.random.Generator, split: str, idx: int) -> Example:
    symbols = ["A", "B"] if split == "train" else ["A", "B", "C"]
    length = int(rng.integers(3, 7 if split == "train" else 12))
    tokens: list[str] = []
    tags: list[str] = []
    if rng.random() < 0.22:
        # Entity-count branch.
        count_ops = ["create_X", "remove_X", "restore_X", "noise"]
        for _ in range(length - 1):
            tokens.append(str(rng.choice(count_ops, p=[0.42, 0.24, 0.18, 0.16])))
        tokens.append("query_count")
        tags.append("entity_count")
        if length >= 8:
            tags.append("length_generalization")
        return make_example(f"{split}_random_entity_{idx}", tokens, split, tuple(tags))
    for _ in range(length):
        sym = str(rng.choice(symbols))
        family = rng.choice(["sym", "anti", "reset", "mention", "quote", "actual", "instead", "noise"], p=[0.25, 0.18, 0.08, 0.12, 0.08, 0.12, 0.10, 0.07])
        if family == "sym":
            tokens.append(sym)
        elif family == "anti":
            tokens.append(f"anti_{sym}")
            tags.append("cancellation")
        elif family == "reset":
            tokens.append("reset")
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
    if split != "train" and "C" in tokens or any(token.endswith("_C") for token in tokens):
        tags.append("heldout_composition")
    if length >= 8:
        tags.append("length_generalization")
    return make_example(f"{split}_random_symbol_{idx}", tokens, split, tuple(tags))


def repeat_to_count(pool: list[Example], count: int, prefix: str) -> list[Example]:
    out: list[Example] = []
    for idx in range(count):
        src = pool[idx % len(pool)]
        out.append(
            Example(
                f"{prefix}_{idx}_{src.case_id}",
                src.tokens,
                src.label,
                src.tags,
                src.active_symbol,
                src.blocked_symbol,
                src.entity_count,
                src.mutated,
                src.split,
            )
        )
    return out


def build_dataset(seed: int, train_examples: int, eval_examples: int) -> tuple[list[Example], list[Example]]:
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1009)
    train_pool = base_patterns("train")
    eval_pool = base_patterns("eval")
    train_pool.extend(random_sequence(rng_train, "train", idx) for idx in range(max(128, train_examples // 2)))
    eval_pool.extend(random_sequence(rng_eval, "eval", idx) for idx in range(max(128, eval_examples // 2)))
    rng_train.shuffle(train_pool)
    rng_eval.shuffle(eval_pool)
    return repeat_to_count(train_pool, train_examples, "train"), repeat_to_count(eval_pool, eval_examples, "eval")


def max_len(rows: list[Example]) -> int:
    return max(len(row.tokens) for row in rows)


def encode(rows: list[Example], length: int) -> torch.Tensor:
    x = torch.zeros((len(rows), length), dtype=torch.long)
    for row_idx, row in enumerate(rows):
        ids = [TOKEN_TO_ID[token] for token in row.tokens[:length]]
        x[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x


def labels(rows: list[Example]) -> torch.Tensor:
    return torch.tensor([row.label for row in rows], dtype=torch.long)


class StaticMLP(nn.Module):
    def __init__(self, vocab_size: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vocab_size, width * 2), nn.ReLU(), nn.Linear(width * 2, width), nn.ReLU())
        self.out = nn.Linear(width, len(LABELS))

    def bag(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(x, num_classes=len(TOKENS)).float()
        one_hot[:, :, TOKEN_TO_ID[PAD]] = 0.0
        return one_hot.sum(dim=1)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.bag(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.represent(x))


class RNNClassifier(nn.Module):
    def __init__(self, kind: str, width: int):
        super().__init__()
        self.embed = nn.Embedding(len(TOKENS), width, padding_idx=TOKEN_TO_ID[PAD])
        if kind == "SimpleRNN":
            self.rnn: nn.Module = nn.RNN(width, width, batch_first=True)
        elif kind == "GRU":
            self.rnn = nn.GRU(width, width, batch_first=True)
        elif kind == "LSTM":
            self.rnn = nn.LSTM(width, width, batch_first=True)
        else:
            raise ValueError(kind)
        self.out = nn.Linear(width, len(LABELS))

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        _, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.represent(x))


class HighwaySideProcessor(nn.Module):
    def __init__(self, width: int, side_count: int, depth: int, connectivity: str, prismion: bool = False):
        super().__init__()
        self.width = width
        self.side_count = side_count
        self.depth = depth
        self.connectivity = connectivity
        self.prismion = prismion
        self.embed = nn.Embedding(len(TOKENS), width, padding_idx=TOKEN_TO_ID[PAD])
        in_dim = width * 2
        self.side_in = nn.ModuleList(nn.Linear(in_dim, width) for _ in range(side_count))
        self.side_delta = nn.ModuleList(nn.Linear(width, width) for _ in range(side_count))
        self.side_gate = nn.ModuleList(nn.Linear(width, 1) for _ in range(side_count))
        self.link = nn.ModuleList(nn.Linear(width, width) for _ in range(side_count))
        if prismion:
            self.phase = nn.Parameter(torch.randn(side_count, width) * 0.5)
            self.prism_gain = nn.Parameter(torch.zeros(side_count, width))
        self.out = nn.Linear(width, len(LABELS))
        self.last_gate_mean = math.nan
        self.last_gate_std = math.nan

    def side_features(self, shared: torch.Tensor, randomize_links: bool) -> list[torch.Tensor]:
        feats = [torch.tanh(layer(shared)) for layer in self.side_in]
        if self.connectivity == "sparse":
            source = feats
            if randomize_links:
                perm = torch.randperm(self.side_count)
                source = [source[int(idx)] for idx in perm]
            feats = [torch.tanh(feats[i] + self.link[i](source[(i - 1) % self.side_count])) for i in range(self.side_count)]
        elif self.connectivity == "dense":
            source = feats
            if randomize_links:
                perm = torch.randperm(self.side_count)
                source = [source[int(idx)] for idx in perm]
            mean_feat = torch.stack(source, dim=0).mean(dim=0)
            feats = [torch.tanh(feats[i] + self.link[i](mean_feat)) for i in range(self.side_count)]
        return feats

    def update_state(
        self,
        h: torch.Tensor,
        emb: torch.Tensor,
        ablate_side: int | None = None,
        force_gates: str | None = None,
        randomize_links: bool = False,
        collect_gates: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        for _ in range(self.depth):
            shared = torch.cat([h, emb], dim=-1)
            feats = self.side_features(shared, randomize_links)
            updates = []
            gates = []
            for idx, feat in enumerate(feats):
                gate = torch.sigmoid(self.side_gate[idx](feat))
                if force_gates == "zero":
                    gate = torch.zeros_like(gate)
                elif force_gates == "open":
                    gate = torch.ones_like(gate)
                if ablate_side == idx:
                    gate = torch.zeros_like(gate)
                if self.prismion:
                    gain = F.softplus(self.prism_gain[idx]).unsqueeze(0)
                    phase = self.phase[idx].unsqueeze(0)
                    prism = torch.cos(feat + phase).pow(2) * gain
                    delta = torch.tanh(self.side_delta[idx](prism))
                else:
                    delta = torch.tanh(self.side_delta[idx](feat))
                updates.append(gate * delta)
                gates.append(gate)
            if collect_gates is not None:
                collect_gates.extend(gates)
            h = torch.tanh(h + torch.stack(updates, dim=0).sum(dim=0) / math.sqrt(max(1, self.side_count)))
        return h

    def represent(
        self,
        x: torch.Tensor,
        ablate_side: int | None = None,
        force_gates: str | None = None,
        randomize_links: bool = False,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        emb = self.embed(x)
        h = torch.zeros((x.shape[0], self.width), dtype=emb.dtype, device=emb.device)
        gates: list[torch.Tensor] = []
        for step in range(x.shape[1]):
            token = x[:, step]
            mask = (token != TOKEN_TO_ID[PAD]).float().unsqueeze(-1)
            new_h = self.update_state(h, emb[:, step, :], ablate_side, force_gates, randomize_links, gates if collect_stats else None)
            h = mask * new_h + (1.0 - mask) * h
        if collect_stats and gates:
            all_gates = torch.cat([gate.detach().flatten() for gate in gates])
            self.last_gate_mean = float(all_gates.mean().cpu())
            self.last_gate_std = float(all_gates.std(unbiased=False).cpu())
        return h

    def forward(
        self,
        x: torch.Tensor,
        ablate_side: int | None = None,
        force_gates: str | None = None,
        randomize_links: bool = False,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        return self.out(self.represent(x, ablate_side, force_gates, randomize_links, collect_stats))


def make_model(arm: str, width: int, side_count: int, depth: int) -> nn.Module:
    if arm == "MLP_STATIC":
        return StaticMLP(len(TOKENS), width)
    if arm in {"SimpleRNN", "GRU", "LSTM"}:
        return RNNClassifier(arm, width)
    if arm == "HIGHWAY_ONLY_SIDEPROCESSORS":
        return HighwaySideProcessor(width, side_count, depth, "none")
    if arm == "HIGHWAY_SPARSE_SIDE_LINKS":
        return HighwaySideProcessor(width, side_count, depth, "sparse")
    if arm == "HIGHWAY_DENSE_SIDE_LINKS":
        return HighwaySideProcessor(width, side_count, depth, "dense")
    if arm == "HIGHWAY_PRISMION_SIDEPROCESSORS":
        return HighwaySideProcessor(width, side_count, depth, "sparse", prismion=True)
    raise ValueError(f"unknown arm: {arm}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def accuracy(preds: list[int], gold: list[int]) -> float:
    if not gold:
        return math.nan
    return sum(int(p == g) for p, g in zip(preds, gold)) / len(gold)


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
            logits = model(x[start : start + batch_size], **kwargs) if kwargs else model(x[start : start + batch_size])
            out.extend(logits.argmax(dim=1).cpu().tolist())
    return out


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    progress_path: Path | None,
) -> tuple[nn.Module, int | None]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 77)
    epochs_to_threshold: int | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in batch_indices(train_x.shape[0], args.batch_size, rng):
            xb = train_x[idx]
            yb = train_y[idx]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
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


def train_probe(train_h: torch.Tensor, train_y: torch.Tensor, eval_h: torch.Tensor, eval_y: torch.Tensor, classes: int, seed: int) -> float:
    if classes <= 1:
        return math.nan
    set_seed(seed)
    probe = nn.Linear(train_h.shape[1], classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=0.02, weight_decay=1e-4)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_h), train_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(eval_h).argmax(dim=1)
    return float((preds == eval_y).float().mean().cpu())


def linear_probes(model: nn.Module, train_x: torch.Tensor, eval_x: torch.Tensor, train_rows: list[Example], eval_rows: list[Example], seed: int) -> dict[str, float]:
    train_h = represent(model, train_x)
    eval_h = represent(model, eval_x)
    targets = {
        "active_symbol_probe_accuracy": ([row.active_symbol for row in train_rows], [row.active_symbol for row in eval_rows], 4),
        "blocked_symbol_probe_accuracy": ([row.blocked_symbol for row in train_rows], [row.blocked_symbol for row in eval_rows], 4),
        "current_focus_probe_accuracy": ([row.active_symbol for row in train_rows], [row.active_symbol for row in eval_rows], 4),
        "entity_count_probe_accuracy": ([row.entity_count for row in train_rows], [row.entity_count for row in eval_rows], 5),
        "mutation_probe_accuracy": ([row.mutated for row in train_rows], [row.mutated for row in eval_rows], 2),
    }
    out: dict[str, float] = {}
    for idx, (name, (tr, ev, classes)) in enumerate(targets.items()):
        out[name] = train_probe(train_h, torch.tensor(tr, dtype=torch.long), eval_h, torch.tensor(ev, dtype=torch.long), classes, seed + idx + 500)
    out["probe_mean_accuracy"] = float(np.nanmean(list(out.values())))
    return out


def failed_cases(rows: list[Example], preds: list[int], limit: int = 12) -> list[dict[str, object]]:
    out = []
    for row, pred in zip(rows, preds):
        if pred != row.label:
            out.append({"case_id": row.case_id, "tokens": row.tokens, "gold": LABELS[row.label], "pred": LABELS[pred], "tags": row.tags})
            if len(out) >= limit:
                break
    return out


def ablation_metrics(model: nn.Module, arm: str, eval_x: torch.Tensor, eval_rows: list[Example], base_acc: float) -> dict[str, float]:
    if arm not in HIGHWAY_ARMS or not isinstance(model, HighwaySideProcessor):
        return {
            "gate_mean": math.nan,
            "gate_std": math.nan,
            "side_ablation_max_drop": math.nan,
            "side_ablation_mean_drop": math.nan,
            "zero_gate_accuracy": math.nan,
            "open_gate_accuracy": math.nan,
            "randomized_link_accuracy": math.nan,
            "specialization_score": math.nan,
        }
    gold = [row.label for row in eval_rows]
    _ = predict(model, eval_x, collect_stats=True)
    drops = []
    for side_idx in range(model.side_count):
        preds = predict(model, eval_x, ablate_side=side_idx)
        drops.append(max(0.0, base_acc - accuracy(preds, gold)))
    zero_acc = accuracy(predict(model, eval_x, force_gates="zero"), gold)
    open_acc = accuracy(predict(model, eval_x, force_gates="open"), gold)
    rand_acc = accuracy(predict(model, eval_x, randomize_links=True), gold)
    return {
        "gate_mean": model.last_gate_mean,
        "gate_std": model.last_gate_std,
        "side_ablation_max_drop": max(drops) if drops else math.nan,
        "side_ablation_mean_drop": float(np.mean(drops)) if drops else math.nan,
        "zero_gate_accuracy": zero_acc,
        "open_gate_accuracy": open_acc,
        "randomized_link_accuracy": rand_acc,
        "specialization_score": (max(drops) - min(drops)) if drops else math.nan,
    }


def run_job(arm: str, width: int, side_count: int, depth: int, seed: int, args: argparse.Namespace, progress_root: Path | None) -> JobResult:
    set_worker(seed)
    progress_path = progress_root / f"{arm}__w{width}__k{side_count}__d{depth}__seed{seed}.jsonl" if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "arm": arm, "width": width, "side_count": side_count, "depth": depth, "seed": seed})
    train_rows, eval_rows = build_dataset(seed, args.train_examples, args.eval_examples)
    length = max(max_len(train_rows), max_len(eval_rows))
    train_x = encode(train_rows, length)
    eval_x = encode(eval_rows, length)
    train_y = labels(train_rows)
    eval_y = labels(eval_rows)
    model = make_model(arm, width, side_count, depth)
    params = parameter_count(model)
    model, epochs_to_threshold = train_model(model, train_x, train_y, eval_x, eval_y, seed, args, progress_path)
    preds = predict(model, eval_x)
    final_acc = accuracy(preds, eval_y.tolist())
    metrics: dict[str, object] = {
        "final_answer_accuracy": final_acc,
        "heldout_composition_accuracy": tag_accuracy(eval_rows, preds, "heldout_composition"),
        "length_generalization_accuracy": tag_accuracy(eval_rows, preds, "length_generalization"),
        "false_mutation_rate": tag_error_rate(eval_rows, preds, "no_mutation"),
        "false_cancellation_rate": tag_error_rate(eval_rows, preds, "cancellation"),
        "cancellation_accuracy": tag_accuracy(eval_rows, preds, "cancellation"),
        "scope_accuracy": tag_accuracy(eval_rows, preds, "scope"),
        "refocus_accuracy": tag_accuracy(eval_rows, preds, "refocus"),
        "mention_noop_accuracy": tag_accuracy(eval_rows, preds, "mention_noop"),
        "entity_count_accuracy": tag_accuracy(eval_rows, preds, "entity_count"),
        "parameter_count": params,
        "epochs_to_threshold": epochs_to_threshold if epochs_to_threshold is not None else math.nan,
    }
    metrics.update(linear_probes(model, train_x, eval_x, train_rows, eval_rows, seed))
    metrics.update(ablation_metrics(model, arm, eval_x, eval_rows, final_acc))
    return JobResult(arm, width, side_count, depth, seed, metrics, failed_cases(eval_rows, preds))


def merge_metric_values(values: list[object]) -> object:
    if all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
        floats = [float(value) for value in values]
        return {"mean": round(float(np.mean(floats)), 6), "min": round(float(np.min(floats)), 6), "max": round(float(np.max(floats)), 6), "std": round(float(np.std(floats)), 6)}
    return values[0] if values else None


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_key: dict[tuple[str, int, int, int], list[JobResult]] = defaultdict(list)
    for row in results:
        by_key[(row.arm, row.width, row.side_count, row.depth)].append(row)
    out: dict[str, dict[str, object]] = {}
    for (arm, width, side_count, depth), rows in sorted(by_key.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics = {name: merge_metric_values([row.metrics.get(name) for row in rows]) for name in metric_names}
        metrics["seed_stability"] = metrics.get("final_answer_accuracy", {}).get("std") if isinstance(metrics.get("final_answer_accuracy"), dict) else math.nan
        key = f"{arm}/w{width}/k{side_count}/d{depth}"
        out[key] = {"arm": arm, "width": width, "side_count": side_count, "depth": depth, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(row: dict[str, object], name: str, default: float = math.nan) -> float:
    value = row["metrics"].get(name) if isinstance(row.get("metrics"), dict) else None
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    if isinstance(value, (int, float)):
        return float(value)
    return default


def best_by_arm(agg: dict[str, dict[str, object]], arms: set[str], score_metric: str = "final_answer_accuracy") -> dict[str, object] | None:
    candidates = [row for row in agg.values() if str(row["arm"]) in arms]
    if not candidates:
        return None
    return max(candidates, key=lambda row: metric_mean(row, score_metric, -1.0))


def score_for_topology(row: dict[str, object] | None) -> float:
    if row is None:
        return math.nan
    return float(np.nanmean([metric_mean(row, "heldout_composition_accuracy"), metric_mean(row, "length_generalization_accuracy")]))


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    mlp = best_by_arm(agg, {"MLP_STATIC"})
    rnn = best_by_arm(agg, {"GRU", "LSTM"})
    highway_only = best_by_arm(agg, {"HIGHWAY_ONLY_SIDEPROCESSORS"})
    sparse = best_by_arm(agg, {"HIGHWAY_SPARSE_SIDE_LINKS"})
    dense = best_by_arm(agg, {"HIGHWAY_DENSE_SIDE_LINKS"})
    prismion = best_by_arm(agg, {"HIGHWAY_PRISMION_SIDEPROCESSORS"})
    highway_best = max([row for row in (highway_only, sparse, dense) if row is not None], key=score_for_topology, default=None)
    all_best = best_by_arm(agg, set(ALL_ARMS))
    if mlp is not None and score_for_topology(mlp) >= 0.90:
        labels.append("TASK_TOO_EASY")
    if all_best is not None and metric_mean(all_best, "final_answer_accuracy", 0.0) < 0.45:
        labels.append("TASK_TOO_HARD")
    if highway_best is not None and rnn is not None and score_for_topology(highway_best) >= score_for_topology(rnn) + 0.10:
        labels.append("HIGHWAY_TOPOLOGY_POSITIVE")
    if rnn is not None and highway_best is not None and score_for_topology(rnn) >= score_for_topology(highway_best) - 0.03:
        labels.append("STANDARD_RNN_SUFFICIENT")
    if sparse is not None and highway_only is not None and score_for_topology(sparse) >= score_for_topology(highway_only) + 0.05 and (dense is None or score_for_topology(dense) <= score_for_topology(sparse) + 0.03):
        labels.append("SPARSE_COORDINATION_POSITIVE")
    if dense is not None:
        dense_special = metric_mean(dense, "specialization_score", 0.0)
        dense_probe = metric_mean(dense, "probe_mean_accuracy", 0.0)
        if highway_best is dense and (dense_special < 0.03 or dense_probe < 0.70):
            labels.append("DENSE_MONOLITH_WARNING")
    if prismion is not None and sparse is not None:
        prismion_focus = float(np.nanmean([metric_mean(prismion, "cancellation_accuracy"), metric_mean(prismion, "refocus_accuracy"), metric_mean(prismion, "scope_accuracy")]))
        sparse_focus = float(np.nanmean([metric_mean(sparse, "cancellation_accuracy"), metric_mean(sparse, "refocus_accuracy"), metric_mean(sparse, "scope_accuracy")]))
        if prismion_focus >= sparse_focus + 0.05 and score_for_topology(prismion) >= score_for_topology(sparse):
            labels.append("PRISMION_UPDATE_POSITIVE")
    if not labels:
        labels.append("HIGHWAY_SIDEQUEST_PARTIAL")
    return labels


def result_record(result: JobResult) -> dict[str, object]:
    return asdict(result)


def write_curves(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    topo_rows = []
    probe_rows = []
    ablation_rows = []
    for key, row in sorted(agg.items()):
        base = {"key": key, "arm": row["arm"], "width": row["width"], "side_count": row["side_count"], "depth": row["depth"]}
        topo_rows.append(
            {
                **base,
                "final_answer_accuracy": metric_mean(row, "final_answer_accuracy"),
                "heldout_composition_accuracy": metric_mean(row, "heldout_composition_accuracy"),
                "length_generalization_accuracy": metric_mean(row, "length_generalization_accuracy"),
                "parameter_count": metric_mean(row, "parameter_count"),
                "seed_stability": metric_mean(row, "seed_stability"),
            }
        )
        probe_rows.append(
            {
                **base,
                "active_symbol_probe_accuracy": metric_mean(row, "active_symbol_probe_accuracy"),
                "blocked_symbol_probe_accuracy": metric_mean(row, "blocked_symbol_probe_accuracy"),
                "entity_count_probe_accuracy": metric_mean(row, "entity_count_probe_accuracy"),
                "mutation_probe_accuracy": metric_mean(row, "mutation_probe_accuracy"),
                "probe_mean_accuracy": metric_mean(row, "probe_mean_accuracy"),
            }
        )
        ablation_rows.append(
            {
                **base,
                "gate_mean": metric_mean(row, "gate_mean"),
                "gate_std": metric_mean(row, "gate_std"),
                "side_ablation_max_drop": metric_mean(row, "side_ablation_max_drop"),
                "side_ablation_mean_drop": metric_mean(row, "side_ablation_mean_drop"),
                "zero_gate_accuracy": metric_mean(row, "zero_gate_accuracy"),
                "open_gate_accuracy": metric_mean(row, "open_gate_accuracy"),
                "randomized_link_accuracy": metric_mean(row, "randomized_link_accuracy"),
                "specialization_score": metric_mean(row, "specialization_score"),
            }
        )
    write_json(out_dir / "topology_curve.json", topo_rows)
    write_json(out_dir / "probe_curve.json", probe_rows)
    write_json(out_dir / "ablation_curve.json", ablation_rows)


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    summary = {
        "status": status,
        "verdict": labels,
        "completed_jobs": len(results),
        "config": {
            "seeds": args.seeds,
            "arms": args.arms,
            "widths": args.widths,
            "side_counts": args.side_counts,
            "depths": args.depths,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "jobs": jobs,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
        },
        "aggregate": agg,
    }
    write_json(out_dir / "summary.json", summary)
    write_curves(out_dir, agg)
    lines = [
        "# HIGHWAY_SIDEQUEST_TOY_001 Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Jobs: `{jobs}`",
        "",
        "| Arm | W | K | D | Final | Heldout | Length | Cancel | Scope | Refocus | Mention | Probe | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["side_count"]), int(r["depth"]))):
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(
                row["arm"],
                row["width"],
                row["side_count"],
                row["depth"],
                metric_mean(row, "final_answer_accuracy"),
                metric_mean(row, "heldout_composition_accuracy"),
                metric_mean(row, "length_generalization_accuracy"),
                metric_mean(row, "cancellation_accuracy"),
                metric_mean(row, "scope_accuracy"),
                metric_mean(row, "refocus_accuracy"),
                metric_mean(row, "mention_noop_accuracy"),
                metric_mean(row, "probe_mean_accuracy"),
                metric_mean(row, "parameter_count"),
            )
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int, completed_jobs: int) -> None:
    lines = [
        "# HIGHWAY_SIDEQUEST_TOY_001 Result",
        "",
        "## Run",
        "",
        "```text",
        f"seeds={args.seeds}",
        f"arms={args.arms}",
        f"widths={args.widths}",
        f"side_counts={args.side_counts}",
        f"depths={args.depths}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        f"completed_jobs={completed_jobs}",
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Arm Summary",
        "",
        "| Arm | W | K | D | Final | Heldout | Length | Probe | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(agg.values(), key=lambda r: (str(r["arm"]), int(r["width"]), int(r["side_count"]), int(r["depth"]))):
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.0f}` |".format(
                row["arm"],
                row["width"],
                row["side_count"],
                row["depth"],
                metric_mean(row, "final_answer_accuracy"),
                metric_mean(row, "heldout_composition_accuracy"),
                metric_mean(row, "length_generalization_accuracy"),
                metric_mean(row, "probe_mean_accuracy"),
                metric_mean(row, "parameter_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is an abstract symbolic topology probe. It is not an English parser, symbol grounding proof, consciousness claim, or full VRAXION architecture test.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(out_dir: Path, args: argparse.Namespace) -> None:
    train, eval_rows = build_dataset(2026, min(args.train_examples, 24), min(args.eval_examples, 48))
    write_jsonl(out_dir / "examples_sample.jsonl", [{"case_id": row.case_id, "tokens": row.tokens, "label": LABELS[row.label], "tags": row.tags, "split": row.split} for row in train[:12] + eval_rows[:24]])


def build_queue(args: argparse.Namespace) -> list[tuple[str, int, int, int, int]]:
    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    widths = parse_int_csv(args.widths)
    side_counts = parse_int_csv(args.side_counts)
    depths = parse_int_csv(args.depths)
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    queue = []
    for seed in seeds:
        for arm in arms:
            for width in widths:
                if arm in HIGHWAY_ARMS:
                    for side_count in side_counts:
                        for depth in depths:
                            queue.append((arm, width, side_count, depth, seed))
                else:
                    queue.append((arm, width, 0, 0, seed))
    return queue


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, object]], list[str], list[JobResult]]:
    queue = build_queue(args)
    write_json(args.out_dir / "queue.json", [{"arm": arm, "width": width, "side_count": side_count, "depth": depth, "seed": seed} for arm, width, side_count, depth, seed in queue])
    write_samples(args.out_dir, args)
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "jobs": jobs, "total_jobs": len(queue)})
    results: list[JobResult] = []
    write_outputs(args.out_dir, results, args, "partial", jobs)

    if jobs <= 1:
        for arm, width, side_count, depth, seed in queue:
            result = run_job(arm, width, side_count, depth, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, "partial", jobs)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": arm, "width": width, "side_count": side_count, "depth": depth, "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = set()
            future_meta = {}
            for arm, width, side_count, depth, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": arm, "width": width, "side_count": side_count, "depth": depth, "seed": seed})
                future = pool.submit(run_job, arm, width, side_count, depth, seed, args, job_progress_root)
                pending.add(future)
                future_meta[future] = (arm, width, side_count, depth, seed)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    continue
                for future in done:
                    arm, width, side_count, depth, seed = future_meta[future]
                    result = future.result()
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": arm, "width": width, "side_count": side_count, "depth": depth, "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})
    agg, labels = write_outputs(args.out_dir, results, args, "complete", jobs)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "verdict": labels, "completed_jobs": len(results)})
    return agg, labels, results


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")
    agg, labels, results = run_all(args, jobs)
    write_doc_result(agg, labels, args, jobs, len(results))
    print(json.dumps({"verdict": labels, "out": str(args.out_dir), "completed_jobs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
