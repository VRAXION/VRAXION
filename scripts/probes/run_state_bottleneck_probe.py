#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
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

from scripts.probes import run_token_state_update_vs_latent_probe as base


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "state_bottleneck_001"
CONTRACT = ROOT / "docs" / "research" / "STATE_BOTTLENECK_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "STATE_BOTTLENECK_001_RESULT.md"

ENTITY_TYPES = base.ENTITY_TYPES
ENTITY_TO_INDEX = base.ENTITY_TO_INDEX
COUNT_CLASSES = base.COUNT_CLASSES
STATE_DIM = len(ENTITY_TYPES) * COUNT_CLASSES + len(ENTITY_TYPES) + 3
ALL_ARMS = (
    "EXPLICIT_LEDGER_ORACLE",
    "ORACLE_STATE_VISIBLE",
    "BAG_OF_TOKENS_MLP",
    "STATIC_POSITION_MLP",
    "GRU_DIRECT_ANSWER",
    "GRU_STATE_BOTTLENECK",
    "NEURAL_SLOT_BOTTLENECK",
    "SHUFFLED_STATE_BOTTLENECK",
)
BOTTLENECK_ARMS = {"GRU_STATE_BOTTLENECK", "NEURAL_SLOT_BOTTLENECK", "SHUFFLED_STATE_BOTTLENECK"}
SUPERVISION_MODES = ("final_state_only", "per_event_state_supervision")


@dataclass
class ReplayEntity:
    kind: str
    ordinal: int
    present: bool = True
    removed: bool = False


@dataclass(frozen=True)
class StateLabel:
    answer: int
    query_type: int
    counts_by_type: tuple[int, ...]
    flags: tuple[int, int, int]
    boundary_indices: tuple[int, ...]
    boundary_counts: tuple[tuple[int, ...], ...]
    boundary_flags: tuple[tuple[int, int, int], ...]
    replay_answer_matches_row: bool
    replay_error: str | None


@dataclass(frozen=True)
class JobResult:
    arm: str
    supervision_mode: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STATE_BOTTLENECK_001 hard-audited state bottleneck probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--supervision-modes", default=",".join(SUPERVISION_MODES))
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--slot-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in parse_csv(spec):
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def set_deterministic(seed: int) -> None:
    base.set_deterministic(seed)


def singular(noun: str) -> str:
    noun = noun.lower().strip()
    if noun.endswith("s"):
        noun = noun[:-1]
    return base.canonical_kind(noun)


def split_story(text: str) -> tuple[list[str], str]:
    clean = text.strip()
    query_match = re.search(r"How many ([a-z0-9_]+)s\?$", clean)
    query = singular(query_match.group(1)) if query_match else ENTITY_TYPES[0]
    prefix = clean[: query_match.start()].strip() if query_match else clean
    sentences = [part.strip() for part in prefix.split(".") if part.strip()]
    return sentences, query


def counts_from_entities(entities: list[ReplayEntity]) -> tuple[int, ...]:
    counts = [0 for _ in ENTITY_TYPES]
    for entity in entities:
        if entity.present:
            counts[ENTITY_TO_INDEX[entity.kind]] += 1
    return tuple(min(COUNT_CLASSES - 1, value) for value in counts)


def resolve_ref(
    entities: list[ReplayEntity],
    kind: str,
    ref: str,
    last_ref: ReplayEntity | None,
    want_removed: bool | None,
) -> tuple[ReplayEntity | None, bool]:
    candidates = [entity for entity in entities if entity.kind == kind]
    if want_removed is True:
        candidates = [entity for entity in candidates if entity.removed and not entity.present]
    elif want_removed is False:
        candidates = [entity for entity in candidates if entity.present]

    if ref in ("", "the"):
        target = candidates[0] if len(candidates) == 1 else None
        return target, target is None
    if ref == "first":
        target = next((entity for entity in sorted(candidates, key=lambda x: x.ordinal) if entity.ordinal == 1), None)
        return target, target is None
    if ref == "second":
        target = next((entity for entity in sorted(candidates, key=lambda x: x.ordinal) if entity.ordinal == 2), None)
        return target, target is None
    if ref in ("previous", "it"):
        return (last_ref, False) if last_ref is not None and last_ref.kind == kind else (None, True)
    if ref == "other":
        if last_ref is None:
            return None, True
        target = next((entity for entity in candidates if entity is not last_ref), None)
        return target, target is None
    return None, True


def replay_state(row: base.StoryRow, vocab: dict[str, int]) -> StateLabel:
    entities: list[ReplayEntity] = []
    ordinals: Counter[str] = Counter()
    last_ref: ReplayEntity | None = None
    invalid_restore = False
    impossible_reference = False
    ambiguous_reference = False
    boundary_indices: list[int] = []
    boundary_counts: list[tuple[int, ...]] = []
    boundary_flags: list[tuple[int, int, int]] = []
    token_cursor = 0
    sentences, query_kind = split_story(row.text)

    def add_boundary(sentence: str) -> None:
        nonlocal token_cursor
        token_cursor += len(base.tokenise(sentence))
        boundary_indices.append(max(0, min(base.MAX_LEN - 1, token_cursor - 1)))
        boundary_counts.append(counts_from_entities(entities))
        boundary_flags.append((int(invalid_restore), int(impossible_reference), int(ambiguous_reference)))

    try:
        for sentence in sentences:
            lower = sentence.lower().strip()
            kind: str | None = None

            m = re.fullmatch(r"i see (?:a|an) ([a-z0-9_]+)", lower)
            if m:
                kind = singular(m.group(1))
                ordinals[kind] += 1
                entity = ReplayEntity(kind=kind, ordinal=ordinals[kind])
                entities.append(entity)
                last_ref = entity
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"i see one more ([a-z0-9_]+)", lower)
            if m:
                kind = singular(m.group(1))
                ordinals[kind] += 1
                entity = ReplayEntity(kind=kind, ordinal=ordinals[kind])
                entities.append(entity)
                last_ref = entity
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"a ([a-z0-9_]+) arrives", lower)
            if m:
                kind = singular(m.group(1))
                ordinals[kind] += 1
                entity = ReplayEntity(kind=kind, ordinal=ordinals[kind])
                entities.append(entity)
                last_ref = entity
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"the (?:(first|second|previous|other) )?([a-z0-9_]+) gets stolen", lower)
            if m:
                ref = m.group(1) or ""
                kind = singular(m.group(2))
                target, missing = resolve_ref(entities, kind, ref, last_ref, want_removed=False)
                if missing or target is None:
                    impossible_reference = True
                    ambiguous_reference = True
                else:
                    target.present = False
                    target.removed = True
                    last_ref = target
                add_boundary(sentence)
                continue

            if lower == "it gets stolen":
                if last_ref is None:
                    impossible_reference = True
                    ambiguous_reference = True
                else:
                    target, missing = resolve_ref(entities, last_ref.kind, "it", last_ref, want_removed=False)
                    if missing or target is None:
                        impossible_reference = True
                        ambiguous_reference = True
                    else:
                        target.present = False
                        target.removed = True
                        last_ref = target
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"they bring it back(?: as a ([a-z0-9_]+))?", lower)
            if m:
                kind = singular(m.group(1)) if m.group(1) else (last_ref.kind if last_ref is not None else query_kind)
                target, missing = resolve_ref(entities, kind, "it", last_ref, want_removed=None)
                if missing or target is None:
                    invalid_restore = True
                    impossible_reference = True
                    ambiguous_reference = True
                elif not target.removed or target.present:
                    invalid_restore = True
                else:
                    target.present = True
                    target.removed = False
                if target is not None:
                    last_ref = target
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"they bring the (first|second|previous|other) ([a-z0-9_]+) back", lower)
            if m:
                ref = m.group(1)
                kind = singular(m.group(2))
                target, missing = resolve_ref(entities, kind, ref, last_ref, want_removed=None)
                if missing or target is None:
                    invalid_restore = True
                    impossible_reference = True
                    ambiguous_reference = True
                elif not target.removed or target.present:
                    invalid_restore = True
                else:
                    target.present = True
                    target.removed = False
                if target is not None:
                    last_ref = target
                add_boundary(sentence)
                continue

            m = re.fullmatch(r"they return the (first|second|previous|other) ([a-z0-9_]+)", lower)
            if m:
                ref = m.group(1)
                kind = singular(m.group(2))
                target, missing = resolve_ref(entities, kind, ref, last_ref, want_removed=None)
                if missing or target is None:
                    invalid_restore = True
                    impossible_reference = True
                    ambiguous_reference = True
                elif not target.removed or target.present:
                    invalid_restore = True
                else:
                    target.present = True
                    target.removed = False
                if target is not None:
                    last_ref = target
                add_boundary(sentence)
                continue

            ambiguous_reference = True
            add_boundary(sentence)
    except Exception as exc:  # pragma: no cover - report path, not normal flow
        query_type = ENTITY_TO_INDEX.get(query_kind, 0)
        return StateLabel(
            answer=row.answer,
            query_type=query_type,
            counts_by_type=row.counts_by_type,
            flags=(row.invalid_restore, row.impossible_reference, row.ambiguous_reference),
            boundary_indices=tuple(boundary_indices),
            boundary_counts=tuple(boundary_counts),
            boundary_flags=tuple(boundary_flags),
            replay_answer_matches_row=False,
            replay_error=str(exc),
        )

    counts = counts_from_entities(entities)
    query_type = ENTITY_TO_INDEX[query_kind]
    answer = counts[query_type]
    if not boundary_indices:
        boundary_indices = (0,)
        boundary_counts = [counts]
        boundary_flags = [(int(invalid_restore), int(impossible_reference), int(ambiguous_reference))]
    return StateLabel(
        answer=answer,
        query_type=query_type,
        counts_by_type=counts,
        flags=(int(invalid_restore), int(impossible_reference), int(ambiguous_reference)),
        boundary_indices=tuple(boundary_indices),
        boundary_counts=tuple(boundary_counts),
        boundary_flags=tuple(boundary_flags),
        replay_answer_matches_row=answer == row.answer,
        replay_error=None,
    )


def labels_for(rows: list[base.StoryRow], vocab: dict[str, int]) -> list[StateLabel]:
    return [replay_state(row, vocab) for row in rows]


def answer_targets(labels: list[StateLabel]) -> torch.Tensor:
    return torch.tensor([label.answer for label in labels], dtype=torch.long)


def count_targets(labels: list[StateLabel]) -> torch.Tensor:
    return torch.tensor([label.counts_by_type for label in labels], dtype=torch.long)


def query_targets(labels: list[StateLabel]) -> torch.Tensor:
    return torch.tensor([label.query_type for label in labels], dtype=torch.long)


def flag_targets(labels: list[StateLabel]) -> torch.Tensor:
    return torch.tensor([label.flags for label in labels], dtype=torch.float32)


def true_state_vector(labels: list[StateLabel]) -> torch.Tensor:
    counts = F.one_hot(count_targets(labels), num_classes=COUNT_CLASSES).float().reshape(len(labels), -1)
    query = F.one_hot(query_targets(labels), num_classes=len(ENTITY_TYPES)).float()
    flags = flag_targets(labels)
    return torch.cat([counts, query, flags], dim=1)


def harden_state_vector(state: torch.Tensor) -> torch.Tensor:
    count_part = state[:, : len(ENTITY_TYPES) * COUNT_CLASSES].reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES)
    query_part = state[:, len(ENTITY_TYPES) * COUNT_CLASSES : len(ENTITY_TYPES) * COUNT_CLASSES + len(ENTITY_TYPES)]
    flag_part = state[:, -3:]
    hard_counts = F.one_hot(count_part.argmax(dim=2), num_classes=COUNT_CLASSES).float().reshape(state.shape[0], -1)
    hard_query = F.one_hot(query_part.argmax(dim=1), num_classes=len(ENTITY_TYPES)).float()
    hard_flags = (flag_part >= 0.5).float()
    return torch.cat([hard_counts, hard_query, hard_flags], dim=1)


def deterministic_decode(state: torch.Tensor) -> torch.Tensor:
    count_part = state[:, : len(ENTITY_TYPES) * COUNT_CLASSES].reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES)
    query_part = state[:, len(ENTITY_TYPES) * COUNT_CLASSES : len(ENTITY_TYPES) * COUNT_CLASSES + len(ENTITY_TYPES)]
    count_pred = count_part.argmax(dim=2)
    query_pred = query_part.argmax(dim=1)
    row_idx = torch.arange(state.shape[0], device=state.device)
    return count_pred[row_idx, query_pred]


def state_slot_metrics(state: torch.Tensor, labels: list[StateLabel]) -> dict[str, float]:
    count_pred = state[:, : len(ENTITY_TYPES) * COUNT_CLASSES].reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES).argmax(dim=2)
    query_pred = state[:, len(ENTITY_TYPES) * COUNT_CLASSES : len(ENTITY_TYPES) * COUNT_CLASSES + len(ENTITY_TYPES)].argmax(dim=1)
    flag_pred = (state[:, -3:] >= 0.5).int()
    counts = count_targets(labels)
    query = query_targets(labels)
    flags = flag_targets(labels).int()
    count_acc = float((count_pred == counts).float().mean().item())
    query_acc = float((query_pred == query).float().mean().item())
    flag_acc = float((flag_pred == flags).float().mean().item())
    return {
        "count_slot_accuracy": count_acc,
        "query_slot_accuracy": query_acc,
        "flag_accuracy": flag_acc,
        "state_slot_accuracy": float(np.mean([count_acc, query_acc, flag_acc])),
    }


def batches(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(n, start + batch_size))


class DirectGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.answer = nn.Linear(hidden, COUNT_CLASSES)

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(seq)
        output, _ = self.gru(embedded)
        lengths = (seq != 0).sum(dim=1).clamp(min=1)
        row_idx = torch.arange(seq.shape[0], device=seq.device)
        return output[row_idx, lengths - 1], output

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode(seq)
        return self.answer(hidden)


class StateBottleneckGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, slot_dim: int | None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.slot = nn.Sequential(nn.Linear(hidden, slot_dim), nn.Tanh()) if slot_dim is not None else None
        state_input = slot_dim if slot_dim is not None else hidden
        self.counts = nn.Linear(state_input, len(ENTITY_TYPES) * COUNT_CLASSES)
        self.query = nn.Linear(state_input, len(ENTITY_TYPES))
        self.flags = nn.Linear(state_input, 3)
        self.answer_head = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, COUNT_CLASSES))

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(seq)
        output, _ = self.gru(embedded)
        lengths = (seq != 0).sum(dim=1).clamp(min=1)
        row_idx = torch.arange(seq.shape[0], device=seq.device)
        return output[row_idx, lengths - 1], output

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.slot(hidden) if self.slot is not None else hidden

    def state_logits_from_hidden(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self.project(hidden)
        return {
            "counts": self.counts(source).reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES),
            "query": self.query(source),
            "flags": self.flags(source),
        }

    def soft_state_from_logits(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        counts = F.softmax(logits["counts"], dim=2).reshape(logits["counts"].shape[0], -1)
        query = F.softmax(logits["query"], dim=1)
        flags = torch.sigmoid(logits["flags"])
        return torch.cat([counts, query, flags], dim=1)

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, output = self.encode(seq)
        logits = self.state_logits_from_hidden(hidden)
        soft_state = self.soft_state_from_logits(logits)
        hard_state = harden_state_vector(soft_state)
        return {
            "hidden": hidden,
            "output": output,
            "logits": logits,
            "soft_state": soft_state,
            "hard_state": hard_state,
            "soft_answer": self.answer_head(soft_state),
            "hard_answer": self.answer_head(hard_state),
            "det_answer": deterministic_decode(hard_state),
        }


def state_loss(logits: dict[str, torch.Tensor], rows: slice, counts: torch.Tensor, query: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    count_logits = logits["counts"]
    count_loss = torch.stack(
        [F.cross_entropy(count_logits[:, idx, :], counts[rows, idx]) for idx in range(len(ENTITY_TYPES))]
    ).mean()
    query_loss = F.cross_entropy(logits["query"], query[rows])
    flag_loss = F.binary_cross_entropy_with_logits(logits["flags"], flags[rows])
    return count_loss + query_loss + flag_loss


def per_event_loss(
    model: StateBottleneckGRU,
    output: torch.Tensor,
    labels: list[StateLabel],
    row_indices: list[int],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for local_idx, global_idx in enumerate(row_indices):
        label = labels[global_idx]
        if not label.boundary_indices:
            continue
        hidden = output[local_idx, torch.tensor(label.boundary_indices, dtype=torch.long)]
        logits = model.state_logits_from_hidden(hidden)
        counts = torch.tensor(label.boundary_counts, dtype=torch.long)
        flags = torch.tensor(label.boundary_flags, dtype=torch.float32)
        count_loss = torch.stack(
            [F.cross_entropy(logits["counts"][:, idx, :], counts[:, idx]) for idx in range(len(ENTITY_TYPES))]
        ).mean()
        flag_loss = F.binary_cross_entropy_with_logits(logits["flags"], flags)
        losses.append(count_loss + flag_loss)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def train_direct_gru(
    seq: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    vocab_size: int,
    embed_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    progress_path: Path | None,
    heartbeat_seconds: int,
) -> DirectGRU:
    set_deterministic(seed)
    model = DirectGRU(vocab_size, embed_dim, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_progress = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        total_steps = math.ceil(seq.shape[0] / batch_size)
        for sl in batches(seq.shape[0], batch_size):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(seq[sl]), y[sl])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_write_job_progress(
                progress_path,
                last_progress,
                heartbeat_seconds,
                {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)},
            )
        if progress_path is not None:
            append_jsonl(
                progress_path,
                {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)},
            )
            last_progress = time.time()
    return model


def train_state_model(
    seq: torch.Tensor,
    y: torch.Tensor,
    labels: list[StateLabel],
    seed: int,
    vocab_size: int,
    embed_dim: int,
    hidden: int,
    slot_dim: int | None,
    epochs: int,
    lr: float,
    batch_size: int,
    supervision_mode: str,
    shuffled: bool,
    progress_path: Path | None,
    heartbeat_seconds: int,
) -> StateBottleneckGRU:
    set_deterministic(seed)
    counts = count_targets(labels)
    query = query_targets(labels)
    flags = flag_targets(labels)
    if shuffled:
        perm = torch.randperm(seq.shape[0], generator=torch.Generator().manual_seed(seed + 7001))
        counts = counts[perm]
        query = query[perm]
        flags = flags[perm]
    model = StateBottleneckGRU(vocab_size, embed_dim, hidden, slot_dim=slot_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    all_indices = list(range(seq.shape[0]))
    last_progress = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        total_steps = math.ceil(seq.shape[0] / batch_size)
        for sl in batches(seq.shape[0], batch_size):
            optimizer.zero_grad()
            out = model(seq[sl])
            loss = F.cross_entropy(out["soft_answer"], y[sl])
            loss = loss + state_loss(out["logits"], sl, counts, query, flags)
            if supervision_mode == "per_event_state_supervision":
                row_indices = all_indices[sl.start : sl.stop]
                loss = loss + 0.30 * per_event_loss(model, out["output"], labels, row_indices)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_write_job_progress(
                progress_path,
                last_progress,
                heartbeat_seconds,
                {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)},
            )
        if progress_path is not None:
            append_jsonl(
                progress_path,
                {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)},
            )
            last_progress = time.time()
    return model


def accuracy(preds: list[int], labels: list[int]) -> float:
    return base.accuracy(preds, labels)


def metrics_for_predictions(rows: list[base.StoryRow], labels: list[StateLabel], preds: list[int]) -> dict[str, float]:
    patched_rows: list[base.StoryRow] = []
    for row, label in zip(rows, labels):
        patched_rows.append(
            base.StoryRow(
                story_id=row.story_id,
                text=row.text,
                answer=label.answer,
                query_type=label.query_type,
                counts_by_type=label.counts_by_type,
                invalid_restore=label.flags[0],
                impossible_reference=label.flags[1],
                ambiguous_reference=label.flags[2],
                tags=row.tags,
                split=row.split,
            )
        )
    return base.metrics_for_predictions(patched_rows, preds)


def count_mae_from_labels(labels: list[StateLabel], preds: list[int]) -> float:
    return sum(abs(pred - label.answer) for pred, label in zip(preds, labels)) / len(labels)


def failed_cases(rows: list[base.StoryRow], labels: list[StateLabel], preds: list[int], limit: int = 10) -> list[dict[str, object]]:
    failed: list[dict[str, object]] = []
    for row, label, pred in zip(rows, labels, preds):
        if pred != label.answer:
            failed.append(
                {
                    "story_id": row.story_id,
                    "text": row.text,
                    "expected": label.answer,
                    "predicted": pred,
                    "tags": list(row.tags),
                }
            )
            if len(failed) >= limit:
                break
    return failed


def finish_job(result: JobResult, progress_path: Path | None) -> JobResult:
    if progress_path is not None:
        append_jsonl(
            progress_path,
            {
                "time": now_iso(),
                "event": "job_worker_done",
                "arm": result.arm,
                "supervision_mode": result.supervision_mode,
                "seed": result.seed,
            },
        )
    return result


def feature_leak_audit(rows: list[base.StoryRow]) -> str:
    return base.feature_leak_audit(rows)


def state_replay_audit(labels: list[StateLabel]) -> str:
    if all(label.replay_error is None and len(label.boundary_indices) == len(label.boundary_counts) for label in labels):
        return "pass"
    return "fail"


def row_answer_match_rate(labels: list[StateLabel]) -> float:
    return sum(int(label.replay_answer_matches_row) for label in labels) / len(labels)


def train_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    progress_path: Path | None,
    heartbeat_seconds: int,
) -> base.MLP:
    set_deterministic(seed)
    model = base.MLP(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_progress = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        total_steps = math.ceil(x.shape[0] / batch_size)
        for sl in batches(x.shape[0], batch_size):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[sl]), y[sl])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_write_job_progress(
                progress_path,
                last_progress,
                heartbeat_seconds,
                {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)},
            )
        if progress_path is not None:
            append_jsonl(
                progress_path,
                {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)},
            )
            last_progress = time.time()
    return model


def run_job(
    arm: str,
    supervision_mode: str,
    seed: int,
    train_examples: int,
    eval_examples: int,
    epochs: int,
    hidden: int,
    slot_dim: int,
    embed_dim: int,
    batch_size: int,
    lr: float,
    progress_root: Path | None,
    heartbeat_seconds: int,
) -> JobResult:
    set_deterministic(seed)
    progress_path = None
    if progress_root is not None:
        progress_path = progress_root / safe_job_name(arm, supervision_mode, seed)
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "arm": arm, "supervision_mode": supervision_mode, "seed": seed})
    train_rows, eval_rows = base.build_dataset(seed, train_examples, eval_examples)
    vocab = base.build_vocab(train_rows + eval_rows)
    train_labels = labels_for(train_rows, vocab)
    eval_labels = labels_for(eval_rows, vocab)
    feature_audit = feature_leak_audit(train_rows + eval_rows)
    replay_audit = state_replay_audit(train_labels + eval_labels)

    train_seq = base.encode_sequences(train_rows, vocab)
    eval_seq = base.encode_sequences(eval_rows, vocab)
    train_y = answer_targets(train_labels)
    eval_y = answer_targets(eval_labels)

    if arm == "EXPLICIT_LEDGER_ORACLE":
        preds = [label.answer for label in eval_labels]
        metrics = metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_learned_head_accuracy": math.nan,
                "hard_state_learned_head_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": 1.0,
                "count_slot_accuracy": 1.0,
                "query_slot_accuracy": 1.0,
                "flag_accuracy": 1.0,
                "state_slot_accuracy": 1.0,
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": 0.0,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if arm == "ORACLE_STATE_VISIBLE":
        oracle_state = true_state_vector(eval_labels)
        preds = deterministic_decode(oracle_state).tolist()
        metrics = metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_learned_head_accuracy": math.nan,
                "hard_state_learned_head_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": accuracy(preds, eval_y.tolist()),
                "count_slot_accuracy": 1.0,
                "query_slot_accuracy": 1.0,
                "flag_accuracy": 1.0,
                "state_slot_accuracy": 1.0,
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": 0.0,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if arm == "BAG_OF_TOKENS_MLP":
        train_x = base.bag_features(train_seq, len(vocab))
        eval_x = base.bag_features(eval_seq, len(vocab))
        model = train_mlp(train_x, train_y, seed, hidden, epochs, lr, batch_size, progress_path, heartbeat_seconds)
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=1).tolist()
        metrics = metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_learned_head_accuracy": math.nan,
                "hard_state_learned_head_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": math.nan,
                "count_slot_accuracy": math.nan,
                "query_slot_accuracy": math.nan,
                "flag_accuracy": math.nan,
                "state_slot_accuracy": math.nan,
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": math.nan,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if arm == "STATIC_POSITION_MLP":
        train_x = base.static_position_features(train_seq, len(vocab))
        eval_x = base.static_position_features(eval_seq, len(vocab))
        model = train_mlp(train_x, train_y, seed, hidden, epochs, lr, batch_size, progress_path, heartbeat_seconds)
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=1).tolist()
        metrics = metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_learned_head_accuracy": math.nan,
                "hard_state_learned_head_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": math.nan,
                "count_slot_accuracy": math.nan,
                "query_slot_accuracy": math.nan,
                "flag_accuracy": math.nan,
                "state_slot_accuracy": math.nan,
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": math.nan,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if arm == "GRU_DIRECT_ANSWER":
        model = train_direct_gru(train_seq, train_y, seed, len(vocab), embed_dim, hidden, epochs, lr, batch_size, progress_path, heartbeat_seconds)
        with torch.no_grad():
            preds = model(eval_seq).argmax(dim=1).tolist()
        metrics = metrics_for_predictions(eval_rows, eval_labels, preds)
        metrics.update(
            {
                "soft_state_learned_head_accuracy": math.nan,
                "hard_state_learned_head_accuracy": math.nan,
                "deterministic_state_decoder_accuracy": math.nan,
                "count_slot_accuracy": math.nan,
                "query_slot_accuracy": math.nan,
                "flag_accuracy": math.nan,
                "state_slot_accuracy": math.nan,
                "soft_vs_hard_gap": math.nan,
                "deterministic_decoder_gap": math.nan,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, preds)), progress_path)

    if arm in BOTTLENECK_ARMS:
        slot = slot_dim if arm == "NEURAL_SLOT_BOTTLENECK" else None
        model = train_state_model(
            train_seq,
            train_y,
            train_labels,
            seed,
            len(vocab),
            embed_dim,
            hidden,
            slot,
            epochs,
            lr,
            batch_size,
            supervision_mode,
            shuffled=arm == "SHUFFLED_STATE_BOTTLENECK",
            progress_path=progress_path,
            heartbeat_seconds=heartbeat_seconds,
        )
        with torch.no_grad():
            out = model(eval_seq)
            soft_preds = out["soft_answer"].argmax(dim=1).tolist()
            hard_preds = out["hard_answer"].argmax(dim=1).tolist()
            det_preds = out["det_answer"].tolist()
            hard_state = out["hard_state"]
        metrics = metrics_for_predictions(eval_rows, eval_labels, det_preds)
        slot_metrics = state_slot_metrics(hard_state, eval_labels)
        metrics.update(slot_metrics)
        soft_acc = accuracy(soft_preds, eval_y.tolist())
        hard_acc = accuracy(hard_preds, eval_y.tolist())
        det_acc = accuracy(det_preds, eval_y.tolist())
        metrics.update(
            {
                "soft_state_learned_head_accuracy": soft_acc,
                "hard_state_learned_head_accuracy": hard_acc,
                "deterministic_state_decoder_accuracy": det_acc,
                "soft_vs_hard_gap": soft_acc - hard_acc,
                "deterministic_decoder_gap": soft_acc - det_acc,
                "feature_leak_audit": feature_audit,
                "state_replay_audit": replay_audit,
                "row_answer_match_rate": row_answer_match_rate(eval_labels),
                "vocab_size": len(vocab),
            }
        )
        return finish_job(JobResult(arm, supervision_mode, seed, metrics, failed_cases(eval_rows, eval_labels, det_preds)), progress_path)

    raise ValueError(f"unknown arm: {arm}")


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_key: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        by_key[f"{result.arm}/{result.supervision_mode}"].append(result)
    out: dict[str, dict[str, object]] = {}
    for key, rows in sorted(by_key.items()):
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics: dict[str, object] = {}
        for name in metric_names:
            values = [row.metrics.get(name) for row in rows]
            if all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
                floats = [float(value) for value in values]
                metrics[name] = {
                    "mean": round(float(np.mean(floats)), 6),
                    "min": round(float(np.min(floats)), 6),
                    "max": round(float(np.max(floats)), 6),
                }
            elif name in {"feature_leak_audit", "state_replay_audit"}:
                metrics[name] = sorted(set(str(value) for value in values))
            else:
                metrics[name] = values[0] if values else None
        out[key] = {
            "arm": rows[0].arm,
            "supervision_mode": rows[0].supervision_mode,
            "seeds": [row.seed for row in rows],
            "metrics": metrics,
        }
    return out


def metric_mean(agg: dict[str, dict[str, object]], key: str, name: str, default: float = math.nan) -> float:
    try:
        value = agg[key]["metrics"][name]
        if isinstance(value, dict):
            return float(value["mean"])
    except KeyError:
        pass
    return default


def find_key(agg: dict[str, dict[str, object]], arm: str, mode: str = "baseline") -> str | None:
    key = f"{arm}/{mode}"
    return key if key in agg else None


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    oracle_key = find_key(agg, "EXPLICIT_LEDGER_ORACLE")
    oracle_state_key = find_key(agg, "ORACLE_STATE_VISIBLE")
    bag_key = find_key(agg, "BAG_OF_TOKENS_MLP")
    direct_key = find_key(agg, "GRU_DIRECT_ANSWER")
    valid = (
        oracle_key is not None
        and oracle_state_key is not None
        and bag_key is not None
        and metric_mean(agg, oracle_key, "answer_accuracy", 0.0) >= 0.98
        and metric_mean(agg, oracle_state_key, "deterministic_state_decoder_accuracy", 0.0) >= 0.98
        and metric_mean(agg, bag_key, "same_token_set_accuracy", 1.0) <= 0.65
        and metric_mean(agg, bag_key, "event_order_shuffle_accuracy", 1.0) <= 0.65
    )
    audit_values: list[str] = []
    for data in agg.values():
        for audit_name in ("feature_leak_audit", "state_replay_audit"):
            audit = data["metrics"].get(audit_name)
            if isinstance(audit, list):
                audit_values.extend(audit)
    if set(audit_values or ["pass"]) != {"pass"}:
        valid = False
    if not valid:
        labels.append("STATE_BOTTLENECK_INVALID")

    direct_acc = metric_mean(agg, direct_key, "answer_accuracy", 0.0) if direct_key else 0.0
    shuffled_best = max(
        (
            metric_mean(agg, key, "deterministic_state_decoder_accuracy", 0.0)
            for key, data in agg.items()
            if data["arm"] == "SHUFFLED_STATE_BOTTLENECK"
        ),
        default=0.0,
    )
    positive = False
    partial = False
    covert = False
    for key, data in agg.items():
        if data["arm"] not in {"GRU_STATE_BOTTLENECK", "NEURAL_SLOT_BOTTLENECK"}:
            continue
        det = metric_mean(agg, key, "deterministic_state_decoder_accuracy", 0.0)
        soft = metric_mean(agg, key, "soft_state_learned_head_accuracy", 0.0)
        hard = metric_mean(agg, key, "hard_state_learned_head_accuracy", 0.0)
        state = metric_mean(agg, key, "state_slot_accuracy", 0.0)
        count = metric_mean(agg, key, "count_slot_accuracy", 0.0)
        same = metric_mean(agg, key, "same_token_set_accuracy", 0.0)
        order = metric_mean(agg, key, "event_order_shuffle_accuracy", 0.0)
        direct_same = metric_mean(agg, direct_key, "same_token_set_accuracy", 0.0) if direct_key else 0.0
        direct_order = metric_mean(agg, direct_key, "event_order_shuffle_accuracy", 0.0) if direct_key else 0.0
        if soft >= 0.85 and (hard < soft - 0.10 or det < soft - 0.10):
            covert = True
        if (
            valid
            and det >= direct_acc + 0.10
            and abs(soft - hard) <= 0.05
            and state >= 0.90
            and count >= 0.90
            and same >= direct_same + 0.10
            and order >= direct_order + 0.10
            and det >= shuffled_best + 0.15
        ):
            positive = True
        elif valid and (state >= 0.80 or det >= direct_acc + 0.03):
            partial = True
    if covert:
        labels.append("SOFT_BOTTLENECK_COVERT_CHANNEL")
    if positive:
        labels.append("STATE_BOTTLENECK_POSITIVE")
    elif partial:
        labels.append("BOTTLENECK_PARTIAL")

    if valid and shuffled_best >= direct_acc + 0.10:
        labels.append("SHUFFLED_CONTROL_FAIL")
    if valid and not positive and not partial and not covert:
        labels.append("EXPLICIT_LEDGER_REQUIRED_FOR_NOW")
    return labels


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_job_name(arm: str, supervision_mode: str, seed: int) -> str:
    clean_mode = supervision_mode.replace("/", "_")
    return f"{arm}__{clean_mode}__seed_{seed}.jsonl"


def maybe_write_job_progress(
    progress_path: Path | None,
    last_write: float,
    heartbeat_seconds: int,
    row: dict[str, object],
) -> float:
    now = time.time()
    if progress_path is not None and now - last_write >= heartbeat_seconds:
        append_jsonl(progress_path, {"time": now_iso(), **row})
        return now
    return last_write


def result_record(result: JobResult) -> dict[str, object]:
    return {
        "arm": result.arm,
        "supervision_mode": result.supervision_mode,
        "seed": result.seed,
        **result.metrics,
        "failed_cases": result.failed_cases,
    }


def record_to_result(record: dict[str, object]) -> JobResult:
    metrics = {
        key: value
        for key, value in record.items()
        if key not in {"arm", "supervision_mode", "seed", "failed_cases"}
    }
    return JobResult(
        arm=str(record["arm"]),
        supervision_mode=str(record["supervision_mode"]),
        seed=int(record["seed"]),
        metrics=metrics,
        failed_cases=list(record.get("failed_cases", [])),
    )


def load_existing_results(path: Path) -> list[JobResult]:
    if not path.exists():
        return []
    results: list[JobResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            results.append(record_to_result(json.loads(line)))
    return results


def result_key(result: JobResult) -> tuple[str, str, int]:
    return result.arm, result.supervision_mode, result.seed


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["STATE_BOTTLENECK_NO_COMPLETED_JOBS"]
    summary = {
        "aggregate": agg,
        "verdict": labels,
        "run_status": status,
        "completed_jobs": len(results),
        "config": {
            "seeds": base.parse_seeds(args.seeds),
            "arms": parse_csv(args.arms),
            "supervision_modes": parse_csv(args.supervision_modes),
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "slot_dim": args.slot_dim,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir, agg, labels, args)
    return agg, labels


def summary_table(agg: dict[str, dict[str, object]]) -> list[str]:
    lines = [
        "| Arm/Mode | Det Answer | Soft | Hard | Same-token | Order | State Slot | Count Slot | Soft-Hard Gap | Det Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(agg):
        det = metric_mean(agg, key, "deterministic_state_decoder_accuracy", metric_mean(agg, key, "answer_accuracy", math.nan))
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                key,
                det,
                metric_mean(agg, key, "soft_state_learned_head_accuracy", math.nan),
                metric_mean(agg, key, "hard_state_learned_head_accuracy", math.nan),
                metric_mean(agg, key, "same_token_set_accuracy", math.nan),
                metric_mean(agg, key, "event_order_shuffle_accuracy", math.nan),
                metric_mean(agg, key, "state_slot_accuracy", math.nan),
                metric_mean(agg, key, "count_slot_accuracy", math.nan),
                metric_mean(agg, key, "soft_vs_hard_gap", math.nan),
                metric_mean(agg, key, "deterministic_decoder_gap", math.nan),
            )
        )
    return lines


def write_report(out_dir: Path, agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# STATE_BOTTLENECK_001 Report",
        "",
        "## Run",
        "",
        f"- Seeds: `{args.seeds}`",
        f"- Train examples: `{args.train_examples}`",
        f"- Eval examples: `{args.eval_examples}`",
        f"- Epochs: `{args.epochs}`",
        "",
        "## Arm Summary",
        "",
        *summary_table(agg),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Claim Boundary",
        "",
        "Controlled toy state-update domain only. No symbol grounding, consciousness, or PrismionCell claim.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# STATE_BOTTLENECK_001 Result",
        "",
        "## Goal",
        "",
        "Test whether a hard-audited predicted-state bottleneck improves story state tracking over direct answer paths.",
        "",
        "## Arm Summary",
        "",
        *summary_table(agg),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "Read bottleneck results through the hard and deterministic decoders. A soft-state answer win is not sufficient because soft probabilities can act as a covert channel.",
        "",
        "## Claim Boundary",
        "",
        "Controlled toy state-update domain only. This does not prove open-ended natural-language grounding or consciousness.",
        "",
    ]
    DOC_REPORT.write_text("\n".join(lines), encoding="utf-8")


def job_plan(arms: list[str], supervision_modes: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    jobs: list[tuple[str, str, int]] = []
    for seed in seeds:
        for arm in arms:
            modes = supervision_modes if arm in BOTTLENECK_ARMS else ["baseline"]
            for mode in modes:
                jobs.append((arm, mode, seed))
    return jobs


def main() -> None:
    args = parse_args()
    arms = parse_csv(args.arms)
    unknown = sorted(set(arms) - set(ALL_ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    supervision_modes = parse_csv(args.supervision_modes)
    unknown_modes = sorted(set(supervision_modes) - set(SUPERVISION_MODES))
    if unknown_modes:
        raise SystemExit(f"unknown supervision modes: {unknown_modes}")
    seeds = base.parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")

    sample_train, sample_eval = base.build_dataset(seeds[0], min(args.train_examples, 32), min(args.eval_examples, 64))
    sample_vocab = base.build_vocab(sample_train + sample_eval)
    sample_train_labels = labels_for(sample_train, sample_vocab)
    sample_eval_labels = labels_for(sample_eval, sample_vocab)
    sample_pairs = list(zip(sample_train, sample_train_labels))[:8] + list(zip(sample_eval, sample_eval_labels))[:16]
    write_jsonl(
        args.out_dir / "examples_sample.jsonl",
        [
            {
                **asdict(row),
                "replayed_answer": label.answer,
                "replayed_counts_by_type": label.counts_by_type,
                "replayed_flags": label.flags,
                "replay_answer_matches_row": label.replay_answer_matches_row,
            }
            for row, label in sample_pairs
        ],
    )

    jobs = job_plan(arms, supervision_modes, seeds)
    queue_path = args.out_dir / "queue.json"
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    queue_path.write_text(
        json.dumps(
            [
                {"arm": arm, "supervision_mode": mode, "seed": seed}
                for arm, mode, seed in jobs
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results = load_existing_results(metrics_path)
    completed = {result_key(result) for result in results}
    pending_jobs = [job for job in jobs if job not in completed]
    job_progress_root = args.out_dir / "job_progress"
    append_jsonl(
        progress_path,
        {
            "time": now_iso(),
            "event": "run_start_or_resume",
            "total_jobs": len(jobs),
            "completed_jobs": len(results),
            "pending_jobs": len(pending_jobs),
        },
    )
    write_outputs(args.out_dir, results, args, status="partial")
    if args.jobs <= 1:
        for arm, mode, seed in pending_jobs:
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": arm, "supervision_mode": mode, "seed": seed})
            result = run_job(
                arm,
                mode,
                seed,
                args.train_examples,
                args.eval_examples,
                args.epochs,
                args.hidden,
                args.slot_dim,
                args.embed_dim,
                args.batch_size,
                args.lr,
                job_progress_root,
                args.heartbeat_seconds,
            )
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, status="partial")
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": arm, "supervision_mode": mode, "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_meta = {}
            pending_futures = set()
            for arm, mode, seed in pending_jobs:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": arm, "supervision_mode": mode, "seed": seed})
                future = pool.submit(
                    run_job,
                    arm,
                    mode,
                    seed,
                    args.train_examples,
                    args.eval_examples,
                    args.epochs,
                    args.hidden,
                    args.slot_dim,
                    args.embed_dim,
                    args.batch_size,
                    args.lr,
                    job_progress_root,
                    args.heartbeat_seconds,
                )
                future_meta[future] = (arm, mode, seed)
                pending_futures.add(future)
            last_heartbeat = time.time()
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=args.heartbeat_seconds, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(
                        progress_path,
                        {
                            "time": now_iso(),
                            "event": "heartbeat",
                            "completed_jobs": len(results),
                            "pending_jobs": len(pending_futures),
                        },
                    )
                    write_outputs(args.out_dir, results, args, status="partial")
                    last_heartbeat = time.time()
                    continue
                for future in done:
                    arm, mode, seed = future_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        append_jsonl(
                            progress_path,
                            {
                                "time": now_iso(),
                                "event": "job_error",
                                "arm": arm,
                                "supervision_mode": mode,
                                "seed": seed,
                                "error": repr(exc),
                                "completed_jobs": len(results),
                            },
                        )
                        write_outputs(args.out_dir, results, args, status="partial_error")
                        raise
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, status="partial")
                    append_jsonl(
                        progress_path,
                        {
                            "time": now_iso(),
                            "event": "job_done",
                            "arm": result.arm,
                            "supervision_mode": result.supervision_mode,
                            "seed": result.seed,
                            "completed_jobs": len(results),
                            "pending_jobs": len(pending_futures),
                        },
                    )
                if time.time() - last_heartbeat >= 60:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                    last_heartbeat = time.time()

    results.sort(key=lambda item: (item.arm, item.supervision_mode, item.seed))
    agg, labels = write_outputs(args.out_dir, results, args, status="complete")
    write_doc_result(agg, labels, args)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "completed_jobs": len(results), "total_jobs": len(jobs)})
    print(json.dumps({"verdict": labels, "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
