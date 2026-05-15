#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "token_state_update_vs_latent"
CONTRACT = ROOT / "docs" / "research" / "TOKEN_TO_STATE_UPDATE_VS_LATENT_001_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "TOKEN_TO_STATE_UPDATE_VS_LATENT_001_RESULT.md"

ENTITY_TYPES = ("dog", "cat", "coin", "robot", "candle", "key")
ENTITY_TO_INDEX = {name: idx for idx, name in enumerate(ENTITY_TYPES)}
COUNT_CLASSES = 5
MAX_LEN = 96
PAD = "<PAD>"
UNK = "<UNK>"

ALL_ARMS = (
    "EXPLICIT_LEDGER_ORACLE",
    "BAG_OF_TOKENS_MLP",
    "STATIC_POSITION_MLP",
    "LATENT_GRU_ANSWER_ONLY",
    "LATENT_GRU_FROZEN_LINEAR_PROBES",
    "HYBRID_STATE_TEACHER",
    "SHUFFLED_STATE_TEACHER",
)

HELDOUT_NOUNS = {
    "hound": "dog",
    "kitten": "cat",
    "token": "coin",
    "android": "robot",
    "lantern": "candle",
    "fob": "key",
}
HELDOUT_VERBS = {
    "vanishes": "gets_stolen",
    "returns": "bring_back",
}


@dataclass(frozen=True)
class StoryRow:
    story_id: str
    text: str
    answer: int
    query_type: int
    counts_by_type: tuple[int, ...]
    invalid_restore: int
    impossible_reference: int
    ambiguous_reference: int
    tags: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class JobResult:
    arm: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


@dataclass
class Entity:
    kind: str
    ordinal: int
    present: bool = True
    removed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOKEN_TO_STATE_UPDATE_VS_LATENT_001 state-update probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", type=int, default=1)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def canonical_kind(noun: str) -> str:
    return HELDOUT_NOUNS.get(noun, noun)


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def resolve_ref(
    entities: list[Entity],
    kind: str,
    ref: str,
    last_ref: Entity | None,
    want_removed: bool | None,
) -> tuple[Entity | None, bool]:
    candidates = [entity for entity in entities if entity.kind == kind]
    if want_removed is True:
        candidates = [entity for entity in candidates if entity.removed and not entity.present]
    elif want_removed is False:
        candidates = [entity for entity in candidates if entity.present]

    if ref == "first":
        target = next((entity for entity in sorted(candidates, key=lambda x: x.ordinal) if entity.ordinal == 1), None)
        return target, target is None
    if ref == "second":
        target = next((entity for entity in sorted(candidates, key=lambda x: x.ordinal) if entity.ordinal == 2), None)
        return target, target is None
    if ref == "previous":
        return (last_ref, False) if last_ref is not None and last_ref.kind == kind else (None, True)
    if ref == "it":
        return (last_ref, False) if last_ref is not None and last_ref.kind == kind else (None, True)
    if ref == "other":
        if last_ref is None:
            return None, True
        target = next((entity for entity in candidates if entity is not last_ref), None)
        return target, target is None
    return None, True


def count_by_type(entities: list[Entity]) -> tuple[int, ...]:
    counts = [0 for _ in ENTITY_TYPES]
    for entity in entities:
        if entity.present:
            counts[ENTITY_TO_INDEX[entity.kind]] += 1
    return tuple(min(COUNT_CLASSES - 1, value) for value in counts)


def make_row(
    story_id: str,
    sentences: list[str],
    entities: list[Entity],
    query_kind: str,
    invalid_restore: bool,
    impossible_reference: bool,
    ambiguous_reference: bool,
    tags: Iterable[str],
    split: str,
) -> StoryRow:
    counts = count_by_type(entities)
    query_type = ENTITY_TO_INDEX[canonical_kind(query_kind)]
    answer = counts[query_type]
    text = " ".join(sentences + [f"How many {query_kind}s?"])
    return StoryRow(
        story_id=story_id,
        text=text,
        answer=answer,
        query_type=query_type,
        counts_by_type=counts,
        invalid_restore=int(invalid_restore),
        impossible_reference=int(impossible_reference),
        ambiguous_reference=int(ambiguous_reference),
        tags=tuple(sorted(set(tags))),
        split=split,
    )


def manual_adversarial_rows(split: str) -> list[StoryRow]:
    rows: list[StoryRow] = []

    def row(story_id: str, sentences: list[str], query: str, answer: int, tags: tuple[str, ...], invalid: int = 0) -> StoryRow:
        counts = [0 for _ in ENTITY_TYPES]
        counts[ENTITY_TO_INDEX[canonical_kind(query)]] = answer
        return StoryRow(
            story_id=story_id,
            text=" ".join(sentences + [f"How many {query}s?"]),
            answer=answer,
            query_type=ENTITY_TO_INDEX[canonical_kind(query)],
            counts_by_type=tuple(counts),
            invalid_restore=invalid,
            impossible_reference=0,
            ambiguous_reference=0,
            tags=tags,
            split=split,
        )

    rows.append(
        row(
            "contrast_restore_then_remove",
            [
                "I see a dog.",
                "I see one more dog.",
                "They bring the first dog back.",
                "The first dog gets stolen.",
            ],
            "dog",
            1,
            ("same_token_set", "event_order_shuffle", "invalid_restore", "coreference"),
            invalid=1,
        )
    )
    rows.append(
        row(
            "contrast_remove_then_restore",
            [
                "I see a dog.",
                "I see one more dog.",
                "The first dog gets stolen.",
                "They bring the first dog back.",
            ],
            "dog",
            2,
            ("same_token_set", "event_order_shuffle", "coreference"),
        )
    )
    rows.append(
        row(
            "previous_it_restore",
            [
                "I see a dog.",
                "I see one more dog.",
                "The previous dog gets stolen.",
                "They bring it back.",
            ],
            "dog",
            2,
            ("coreference",),
        )
    )
    rows.append(
        row(
            "previous_other_invalid",
            [
                "I see a dog.",
                "I see one more dog.",
                "The previous dog gets stolen.",
                "They bring the other dog back.",
            ],
            "dog",
            1,
            ("coreference", "invalid_restore"),
            invalid=1,
        )
    )
    rows.append(
        row(
            "distractor_query_cat",
            [
                "I see a dog.",
                "A cat arrives.",
                "The dog gets stolen.",
            ],
            "cat",
            1,
            ("distractor",),
        )
    )
    rows.append(
        row(
            "distractor_query_dog",
            [
                "I see a dog.",
                "A cat arrives.",
                "The dog gets stolen.",
            ],
            "dog",
            0,
            ("distractor",),
        )
    )
    rows.append(
        row(
            "heldout_noun_restore",
            [
                "I see a hound.",
                "I see one more hound.",
                "The first hound vanishes.",
                "They return the first hound.",
            ],
            "hound",
            2,
            ("heldout_noun", "heldout_verb", "coreference"),
        )
    )
    rows.append(
        row(
            "heldout_noun_invalid_restore",
            [
                "I see a fob.",
                "They return the first fob.",
            ],
            "fob",
            1,
            ("heldout_noun", "heldout_verb", "invalid_restore"),
            invalid=1,
        )
    )
    return rows


def random_story(rng: random.Random, split: str, idx: int) -> StoryRow:
    entities: list[Entity] = []
    ordinals: Counter[str] = Counter()
    sentences: list[str] = []
    last_ref: Entity | None = None
    invalid_restore = False
    impossible_reference = False
    ambiguous_reference = False
    tags: set[str] = set()

    query_kind = rng.choice(ENTITY_TYPES)
    steps = rng.randint(3, 7)

    for step in range(steps):
        present_by_kind = defaultdict(list)
        removed_by_kind = defaultdict(list)
        for entity in entities:
            if entity.present:
                present_by_kind[entity.kind].append(entity)
            elif entity.removed:
                removed_by_kind[entity.kind].append(entity)

        event = rng.choices(
            ("add", "remove", "restore", "distractor"),
            weights=(0.42, 0.24, 0.20, 0.14),
            k=1,
        )[0]
        kind = query_kind if rng.random() < 0.62 else rng.choice(ENTITY_TYPES)
        if event == "distractor":
            kind = rng.choice([name for name in ENTITY_TYPES if name != query_kind])
            tags.add("distractor")

        if event in ("add", "distractor") or not entities:
            if count_by_type(entities)[ENTITY_TO_INDEX[kind]] >= COUNT_CLASSES - 1:
                continue
            ordinals[kind] += 1
            entity = Entity(kind=kind, ordinal=ordinals[kind])
            entities.append(entity)
            last_ref = entity
            if ordinals[kind] == 1:
                sentence = f"I see {article(kind)} {kind}." if event == "add" else f"A {kind} arrives."
            else:
                sentence = f"I see one more {kind}."
            sentences.append(sentence)
            continue

        if event == "remove":
            ref = rng.choice(("first", "second", "previous", "it", "other"))
            target, missing = resolve_ref(entities, kind, ref, last_ref, want_removed=False)
            if missing or target is None:
                impossible_reference = True
                ambiguous_reference = True
                tags.add("ambiguous_reference")
                sentences.append(f"The {ref} {kind} gets stolen.")
                continue
            target.present = False
            target.removed = True
            last_ref = target
            tags.add("coreference")
            if ref == "it":
                sentences.append("It gets stolen.")
            else:
                sentences.append(f"The {ref} {kind} gets stolen.")
            continue

        if event == "restore":
            ref = rng.choice(("first", "second", "previous", "it", "other"))
            target, missing = resolve_ref(entities, kind, ref, last_ref, want_removed=None)
            tags.add("coreference")
            if missing or target is None:
                impossible_reference = True
                ambiguous_reference = True
                invalid_restore = True
                tags.add("invalid_restore")
                if ref == "it":
                    sentences.append(f"They bring it back as a {kind}.")
                else:
                    sentences.append(f"They bring the {ref} {kind} back.")
                continue
            if not target.removed or target.present:
                invalid_restore = True
                tags.add("invalid_restore")
            else:
                target.present = True
                target.removed = False
            last_ref = target
            if ref == "it":
                sentences.append("They bring it back.")
            else:
                sentences.append(f"They bring the {ref} {kind} back.")

    if not any(entity.kind == query_kind for entity in entities):
        ordinals[query_kind] += 1
        entity = Entity(kind=query_kind, ordinal=ordinals[query_kind])
        entities.append(entity)
        last_ref = entity
        sentences.insert(0, f"I see {article(query_kind)} {query_kind}.")

    return make_row(
        story_id=f"{split}_{idx:05d}",
        sentences=sentences,
        entities=entities,
        query_kind=query_kind,
        invalid_restore=invalid_restore,
        impossible_reference=impossible_reference,
        ambiguous_reference=ambiguous_reference,
        tags=tags or ("ordinary",),
        split=split,
    )


def build_dataset(seed: int, train_examples: int, eval_examples: int) -> tuple[list[StoryRow], list[StoryRow]]:
    train_rng = random.Random(seed * 17 + 3)
    eval_rng = random.Random(seed * 19 + 7)
    train = [random_story(train_rng, "train", idx) for idx in range(train_examples)]
    adversarial = manual_adversarial_rows("eval")
    eval_rows = adversarial[:]
    while len(eval_rows) < eval_examples:
        eval_rows.append(random_story(eval_rng, "eval", len(eval_rows)))
    return train, eval_rows[:eval_examples]


def build_vocab(rows: Iterable[StoryRow]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(tokenise(row.text))
    for noun in HELDOUT_NOUNS:
        tokens.add(noun)
    tokens.update(HELDOUT_VERBS)
    return {token: idx for idx, token in enumerate(sorted(tokens))}


def encode_sequences(rows: list[StoryRow], vocab: dict[str, int]) -> torch.Tensor:
    encoded: list[list[int]] = []
    unk = vocab[UNK]
    pad = vocab[PAD]
    for row in rows:
        ids = [vocab.get(token, unk) for token in tokenise(row.text)[:MAX_LEN]]
        ids.extend([pad] * (MAX_LEN - len(ids)))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def answers(rows: list[StoryRow]) -> torch.Tensor:
    return torch.tensor([row.answer for row in rows], dtype=torch.long)


def query_types(rows: list[StoryRow]) -> torch.Tensor:
    return torch.tensor([row.query_type for row in rows], dtype=torch.long)


def count_targets(rows: list[StoryRow]) -> torch.Tensor:
    return torch.tensor([row.counts_by_type for row in rows], dtype=torch.long)


def lifecycle_targets(rows: list[StoryRow]) -> torch.Tensor:
    return torch.tensor(
        [[row.invalid_restore, row.impossible_reference, row.ambiguous_reference] for row in rows],
        dtype=torch.float32,
    )


def bag_features(seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
    x = torch.zeros((seq.shape[0], vocab_size), dtype=torch.float32)
    for idx, row in enumerate(seq):
        counts = torch.bincount(row, minlength=vocab_size).float()
        counts[0] = 0.0
        x[idx] = counts / max(1.0, counts.sum().item())
    return x


def static_position_features(seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
    one_hot = F.one_hot(seq, num_classes=vocab_size).float()
    return one_hot.reshape(seq.shape[0], MAX_LEN * vocab_size)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int = COUNT_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StoryGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, aux: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.answer = nn.Linear(hidden, COUNT_CLASSES)
        self.aux = aux
        if aux:
            self.counts = nn.Linear(hidden, len(ENTITY_TYPES) * COUNT_CLASSES)
            self.query = nn.Linear(hidden, len(ENTITY_TYPES))
            self.lifecycle = nn.Linear(hidden, 3)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(seq)
        output, _ = self.gru(embedded)
        lengths = (seq != 0).sum(dim=1).clamp(min=1)
        row_idx = torch.arange(seq.shape[0], device=seq.device)
        return output[row_idx, lengths - 1]

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(seq)
        out = {"answer": self.answer(hidden), "hidden": hidden}
        if self.aux:
            out["counts"] = self.counts(hidden).reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES)
            out["query"] = self.query(hidden)
            out["lifecycle"] = self.lifecycle(hidden)
        return out


def batches(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(n, start + batch_size))


def train_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> MLP:
    set_deterministic(seed)
    model = MLP(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        for sl in batches(x.shape[0], batch_size):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[sl]), y[sl])
            loss.backward()
            optimizer.step()
    return model


def aux_loss(
    outputs: dict[str, torch.Tensor],
    row_slice: slice,
    count_y: torch.Tensor,
    query_y: torch.Tensor,
    lifecycle_y: torch.Tensor,
) -> torch.Tensor:
    count_logits = outputs["counts"]
    count_loss = torch.stack(
        [F.cross_entropy(count_logits[:, idx, :], count_y[row_slice, idx]) for idx in range(len(ENTITY_TYPES))]
    ).mean()
    query_loss = F.cross_entropy(outputs["query"], query_y[row_slice])
    life_loss = F.binary_cross_entropy_with_logits(outputs["lifecycle"], lifecycle_y[row_slice])
    return count_loss + query_loss + life_loss


def train_gru(
    seq: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    vocab_size: int,
    embed_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    aux: bool,
    count_y: torch.Tensor,
    query_y: torch.Tensor,
    lifecycle_y: torch.Tensor,
    shuffled_aux: bool,
) -> StoryGRU:
    set_deterministic(seed)
    if shuffled_aux:
        perm = torch.randperm(seq.shape[0], generator=torch.Generator().manual_seed(seed + 1009))
        count_y = count_y[perm]
        query_y = query_y[perm]
        lifecycle_y = lifecycle_y[perm]
    model = StoryGRU(vocab_size, embed_dim, hidden, aux=aux)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        for sl in batches(seq.shape[0], batch_size):
            optimizer.zero_grad()
            outputs = model(seq[sl])
            loss = F.cross_entropy(outputs["answer"], y[sl])
            if aux:
                loss = loss + 0.35 * aux_loss(outputs, sl, count_y, query_y, lifecycle_y)
            loss.backward()
            optimizer.step()
    return model


def train_linear_probes(
    train_hidden: torch.Tensor,
    eval_hidden: torch.Tensor,
    train_rows: list[StoryRow],
    eval_rows: list[StoryRow],
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    set_deterministic(seed)
    y_count = answers(train_rows)
    y_eval_count = answers(eval_rows)
    y_type_counts = count_targets(train_rows)
    y_eval_type_counts = count_targets(eval_rows)
    y_life = lifecycle_targets(train_rows)
    y_eval_life = lifecycle_targets(eval_rows)

    query_probe = nn.Linear(train_hidden.shape[1], COUNT_CLASSES)
    type_probe = nn.Linear(train_hidden.shape[1], len(ENTITY_TYPES) * COUNT_CLASSES)
    life_probe = nn.Linear(train_hidden.shape[1], 3)
    params = list(query_probe.parameters()) + list(type_probe.parameters()) + list(life_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        optimizer.zero_grad()
        query_loss = F.cross_entropy(query_probe(train_hidden), y_count)
        type_logits = type_probe(train_hidden).reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES)
        type_loss = torch.stack(
            [F.cross_entropy(type_logits[:, idx, :], y_type_counts[:, idx]) for idx in range(len(ENTITY_TYPES))]
        ).mean()
        life_loss = F.binary_cross_entropy_with_logits(life_probe(train_hidden), y_life)
        loss = query_loss + type_loss + life_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        query_pred = query_probe(eval_hidden).argmax(dim=1)
        type_pred = type_probe(eval_hidden).reshape(-1, len(ENTITY_TYPES), COUNT_CLASSES).argmax(dim=2)
        life_pred = (torch.sigmoid(life_probe(eval_hidden)) >= 0.5).int()
    return {
        "linear_probe_count_accuracy": accuracy(query_pred.tolist(), y_eval_count.tolist()),
        "entity_presence_probe_accuracy": float((type_pred == y_eval_type_counts).float().mean().item()),
        "lifecycle_probe_accuracy": float((life_pred == y_eval_life.int()).float().mean().item()),
    }


def accuracy(preds: list[int], labels: list[int]) -> float:
    if not labels:
        return math.nan
    return sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)


def tag_accuracy(rows: list[StoryRow], preds: list[int], tag: str) -> float:
    pairs = [(pred, row.answer) for row, pred in zip(rows, preds) if tag in row.tags]
    if not pairs:
        return math.nan
    return sum(int(pred == label) for pred, label in pairs) / len(pairs)


def count_mae(rows: list[StoryRow], preds: list[int]) -> float:
    if not rows:
        return math.nan
    return sum(abs(pred - row.answer) for row, pred in zip(rows, preds)) / len(rows)


def failed_cases(rows: list[StoryRow], preds: list[int], limit: int = 10) -> list[dict[str, object]]:
    failed: list[dict[str, object]] = []
    for row, pred in zip(rows, preds):
        if pred != row.answer:
            failed.append(
                {
                    "story_id": row.story_id,
                    "text": row.text,
                    "expected": row.answer,
                    "predicted": pred,
                    "tags": list(row.tags),
                }
            )
            if len(failed) >= limit:
                break
    return failed


def metrics_for_predictions(rows: list[StoryRow], preds: list[int]) -> dict[str, float]:
    return {
        "answer_accuracy": accuracy(preds, [row.answer for row in rows]),
        "heldout_composition_accuracy": float(np.nanmean([tag_accuracy(rows, preds, "heldout_noun"), tag_accuracy(rows, preds, "heldout_verb")])),
        "same_token_set_accuracy": tag_accuracy(rows, preds, "same_token_set"),
        "event_order_shuffle_accuracy": tag_accuracy(rows, preds, "event_order_shuffle"),
        "coreference_accuracy": tag_accuracy(rows, preds, "coreference"),
        "distractor_resistance": tag_accuracy(rows, preds, "distractor"),
        "invalid_restore_accuracy": tag_accuracy(rows, preds, "invalid_restore"),
        "ambiguous_reference_accuracy": tag_accuracy(rows, preds, "ambiguous_reference"),
        "count_mae": count_mae(rows, preds),
    }


def feature_leak_audit(rows: list[StoryRow]) -> str:
    bad_patterns = (" answer ", "=>", " count is ", " result ")
    for row in rows:
        lowered = f" {row.text.lower()} "
        if any(pattern in lowered for pattern in bad_patterns):
            return "fail"
    return "pass"


def run_job(
    arm: str,
    seed: int,
    train_examples: int,
    eval_examples: int,
    epochs: int,
    probe_epochs: int,
    hidden: int,
    embed_dim: int,
    batch_size: int,
    lr: float,
) -> JobResult:
    set_deterministic(seed)
    train_rows, eval_rows = build_dataset(seed, train_examples, eval_examples)
    vocab = build_vocab(train_rows + eval_rows)
    feature_audit = feature_leak_audit(train_rows + eval_rows)

    if arm == "EXPLICIT_LEDGER_ORACLE":
        preds = [row.answer for row in eval_rows]
        metrics = metrics_for_predictions(eval_rows, preds)
        metrics.update(
            {
                "linear_probe_count_accuracy": math.nan,
                "entity_presence_probe_accuracy": math.nan,
                "lifecycle_probe_accuracy": math.nan,
                "feature_leak_audit": feature_audit,
                "vocab_size": len(vocab),
            }
        )
        return JobResult(arm=arm, seed=seed, metrics=metrics, failed_cases=[])

    train_seq = encode_sequences(train_rows, vocab)
    eval_seq = encode_sequences(eval_rows, vocab)
    train_y = answers(train_rows)
    eval_y = answers(eval_rows)

    if arm == "BAG_OF_TOKENS_MLP":
        train_x = bag_features(train_seq, len(vocab))
        eval_x = bag_features(eval_seq, len(vocab))
        model = train_mlp(train_x, train_y, seed, hidden, epochs, lr, batch_size)
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=1).tolist()
        metrics = metrics_for_predictions(eval_rows, preds)
        metrics.update(
            {
                "linear_probe_count_accuracy": math.nan,
                "entity_presence_probe_accuracy": math.nan,
                "lifecycle_probe_accuracy": math.nan,
                "feature_leak_audit": feature_audit,
                "vocab_size": len(vocab),
            }
        )
        return JobResult(arm=arm, seed=seed, metrics=metrics, failed_cases=failed_cases(eval_rows, preds))

    if arm == "STATIC_POSITION_MLP":
        train_x = static_position_features(train_seq, len(vocab))
        eval_x = static_position_features(eval_seq, len(vocab))
        model = train_mlp(train_x, train_y, seed, hidden, epochs, lr, batch_size)
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=1).tolist()
        metrics = metrics_for_predictions(eval_rows, preds)
        metrics.update(
            {
                "linear_probe_count_accuracy": math.nan,
                "entity_presence_probe_accuracy": math.nan,
                "lifecycle_probe_accuracy": math.nan,
                "feature_leak_audit": feature_audit,
                "vocab_size": len(vocab),
            }
        )
        return JobResult(arm=arm, seed=seed, metrics=metrics, failed_cases=failed_cases(eval_rows, preds))

    aux = arm in ("HYBRID_STATE_TEACHER", "SHUFFLED_STATE_TEACHER")
    shuffled = arm == "SHUFFLED_STATE_TEACHER"
    model = train_gru(
        train_seq,
        train_y,
        seed,
        len(vocab),
        embed_dim,
        hidden,
        epochs,
        lr,
        batch_size,
        aux=aux,
        count_y=count_targets(train_rows),
        query_y=query_types(train_rows),
        lifecycle_y=lifecycle_targets(train_rows),
        shuffled_aux=shuffled,
    )
    with torch.no_grad():
        train_out = model(train_seq)
        eval_out = model(eval_seq)
        preds = eval_out["answer"].argmax(dim=1).tolist()
    metrics = metrics_for_predictions(eval_rows, preds)
    probe_metrics = train_linear_probes(
        train_out["hidden"].detach(),
        eval_out["hidden"].detach(),
        train_rows,
        eval_rows,
        seed + 404,
        probe_epochs,
        lr,
    )
    metrics.update(probe_metrics)
    metrics.update({"feature_leak_audit": feature_audit, "vocab_size": len(vocab)})
    if aux:
        with torch.no_grad():
            count_pred = eval_out["counts"].argmax(dim=2)
            life_pred = (torch.sigmoid(eval_out["lifecycle"]) >= 0.5).int()
        metrics["aux_entity_presence_accuracy"] = float((count_pred == count_targets(eval_rows)).float().mean().item())
        metrics["aux_lifecycle_accuracy"] = float((life_pred == lifecycle_targets(eval_rows).int()).float().mean().item())
    return JobResult(arm=arm, seed=seed, metrics=metrics, failed_cases=failed_cases(eval_rows, preds))


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_arm: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        by_arm[result.arm].append(result)
    out: dict[str, dict[str, object]] = {}
    for arm, arm_results in sorted(by_arm.items()):
        metric_names = sorted({key for result in arm_results for key in result.metrics})
        metrics: dict[str, object] = {}
        for name in metric_names:
            values = [result.metrics.get(name) for result in arm_results]
            if all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
                floats = [float(value) for value in values]
                metrics[name] = {
                    "mean": round(float(np.mean(floats)), 6),
                    "min": round(float(np.min(floats)), 6),
                    "max": round(float(np.max(floats)), 6),
                }
            elif name == "feature_leak_audit":
                metrics[name] = sorted(set(str(value) for value in values))
            else:
                metrics[name] = values[0] if values else None
        out[arm] = {"seeds": [result.seed for result in arm_results], "metrics": metrics}
    return out


def metric_mean(agg: dict[str, dict[str, object]], arm: str, name: str, default: float = math.nan) -> float:
    try:
        value = agg[arm]["metrics"][name]
        if isinstance(value, dict):
            return float(value["mean"])
    except KeyError:
        pass
    return default


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    oracle_ok = metric_mean(agg, "EXPLICIT_LEDGER_ORACLE", "answer_accuracy", 0.0) >= 0.98
    bag_same = metric_mean(agg, "BAG_OF_TOKENS_MLP", "same_token_set_accuracy", 1.0)
    leak_values: list[str] = []
    for arm_data in agg.values():
        leak = arm_data["metrics"].get("feature_leak_audit")
        if isinstance(leak, list):
            leak_values.extend(leak)
    valid = oracle_ok and bag_same <= 0.65 and set(leak_values or ["pass"]) == {"pass"}
    if not valid:
        labels.append("TOKEN_STATE_UPDATE_INVALID")

    latent = "LATENT_GRU_FROZEN_LINEAR_PROBES"
    latent_answer = metric_mean(agg, latent, "answer_accuracy")
    latent_count_probe = metric_mean(agg, latent, "linear_probe_count_accuracy")
    latent_lifecycle = metric_mean(agg, latent, "lifecycle_probe_accuracy")
    latent_same = metric_mean(agg, latent, "same_token_set_accuracy")
    latent_order = metric_mean(agg, latent, "event_order_shuffle_accuracy")

    if (
        valid
        and latent_answer >= 0.90
        and latent_count_probe >= 0.90
        and latent_lifecycle >= 0.85
        and latent_same >= 0.85
        and latent_order >= 0.85
    ):
        labels.append("LATENT_STATE_POSITIVE")
    elif valid and latent_answer >= 0.90 and (latent_count_probe < 0.90 or latent_lifecycle < 0.85):
        labels.append("SHORTCUT_OR_OPAQUE_SUCCESS")

    hybrid = "HYBRID_STATE_TEACHER"
    shuffled = "SHUFFLED_STATE_TEACHER"
    base_answer = metric_mean(agg, "LATENT_GRU_ANSWER_ONLY", "answer_accuracy", 0.0)
    hybrid_answer = metric_mean(agg, hybrid, "answer_accuracy", 0.0)
    shuffled_answer = metric_mean(agg, shuffled, "answer_accuracy", 0.0)
    hybrid_probe = metric_mean(agg, hybrid, "linear_probe_count_accuracy", 0.0)
    base_probe = metric_mean(agg, "LATENT_GRU_FROZEN_LINEAR_PROBES", "linear_probe_count_accuracy", 0.0)
    hybrid_counter = float(np.nanmean([metric_mean(agg, hybrid, "same_token_set_accuracy", 0.0), metric_mean(agg, hybrid, "event_order_shuffle_accuracy", 0.0)]))
    base_counter = float(np.nanmean([metric_mean(agg, "LATENT_GRU_ANSWER_ONLY", "same_token_set_accuracy", 0.0), metric_mean(agg, "LATENT_GRU_ANSWER_ONLY", "event_order_shuffle_accuracy", 0.0)]))
    if (
        valid
        and hybrid_answer >= base_answer + 0.10
        and hybrid_answer >= shuffled_answer + 0.15
        and hybrid_probe >= base_probe + 0.05
        and hybrid_counter >= base_counter + 0.10
    ):
        labels.append("HYBRID_STATE_SUPERVISION_POSITIVE")

    static_answer = metric_mean(agg, "STATIC_POSITION_MLP", "answer_accuracy", -1.0)
    if static_answer >= latent_answer - 0.03 and latent_answer >= 0.85 and (latent_count_probe < 0.90 or latent_lifecycle < 0.85):
        labels.append("STATIC_SHORTCUT_WARNING")

    if valid and not any(label in labels for label in ("LATENT_STATE_POSITIVE", "HYBRID_STATE_SUPERVISION_POSITIVE", "SHORTCUT_OR_OPAQUE_SUCCESS")):
        labels.append("EXPLICIT_LEDGER_REQUIRED_FOR_NOW")
    return labels


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(out_dir: Path, agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# TOKEN_TO_STATE_UPDATE_VS_LATENT_001 Report",
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
        "| Arm | Answer | Same-token | Order-shuffle | Coref | Invalid restore | Linear count probe | Lifecycle probe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "answer_accuracy", math.nan),
                metric_mean(agg, arm, "same_token_set_accuracy", math.nan),
                metric_mean(agg, arm, "event_order_shuffle_accuracy", math.nan),
                metric_mean(agg, arm, "coreference_accuracy", math.nan),
                metric_mean(agg, arm, "invalid_restore_accuracy", math.nan),
                metric_mean(agg, arm, "linear_probe_count_accuracy", math.nan),
                metric_mean(agg, arm, "lifecycle_probe_accuracy", math.nan),
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(labels, indent=2),
            "```",
            "",
            "## Claim Boundary",
            "",
            "Toy controlled story grammar only. No open natural-language grounding, consciousness, or PrismionCell claim.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# TOKEN_TO_STATE_UPDATE_VS_LATENT_001 Result",
        "",
        "## Goal",
        "",
        "Test explicit ledger, latent recurrent state, and hybrid state supervision on a controlled object-lifecycle counting suite.",
        "",
        "## Arm Summary",
        "",
        "| Arm | Answer | Same-token | Order-shuffle | Coref | Invalid restore | Linear count probe | Lifecycle probe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "answer_accuracy", math.nan),
                metric_mean(agg, arm, "same_token_set_accuracy", math.nan),
                metric_mean(agg, arm, "event_order_shuffle_accuracy", math.nan),
                metric_mean(agg, arm, "coreference_accuracy", math.nan),
                metric_mean(agg, arm, "invalid_restore_accuracy", math.nan),
                metric_mean(agg, arm, "linear_probe_count_accuracy", math.nan),
                metric_mean(agg, arm, "lifecycle_probe_accuracy", math.nan),
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(labels, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "This result should be read through the adversarial controls. High final answer accuracy without frozen linear state probes is marked as shortcut or opaque success, not as reliable latent state tracking.",
            "",
            "## Claim Boundary",
            "",
            "Controlled toy grammar only. The result does not prove open-ended natural-language grounding or consciousness.",
            "",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    arms = parse_csv(args.arms)
    unknown = sorted(set(arms) - set(ALL_ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    seeds = parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")

    sample_train, sample_eval = build_dataset(seeds[0], min(args.train_examples, 32), min(args.eval_examples, 64))
    write_jsonl(args.out_dir / "examples_sample.jsonl", [asdict(row) for row in sample_train[:8] + sample_eval[:16]])
    adversarial = [row for row in sample_eval if any(tag in row.tags for tag in ("same_token_set", "event_order_shuffle", "invalid_restore", "distractor", "heldout_noun"))]
    write_jsonl(args.out_dir / "counterfactual_cases.jsonl", [asdict(row) for row in adversarial])

    jobs = [(arm, seed) for seed in seeds for arm in arms]
    results: list[JobResult] = []
    if args.jobs <= 1:
        for arm, seed in jobs:
            results.append(
                run_job(
                    arm,
                    seed,
                    args.train_examples,
                    args.eval_examples,
                    args.epochs,
                    args.probe_epochs,
                    args.hidden,
                    args.embed_dim,
                    args.batch_size,
                    args.lr,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    run_job,
                    arm,
                    seed,
                    args.train_examples,
                    args.eval_examples,
                    args.epochs,
                    args.probe_epochs,
                    args.hidden,
                    args.embed_dim,
                    args.batch_size,
                    args.lr,
                )
                for arm, seed in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: (row.arm, row.seed))
    write_jsonl(
        args.out_dir / "metrics.jsonl",
        [
            {
                "arm": result.arm,
                "seed": result.seed,
                **result.metrics,
                "failed_cases": result.failed_cases,
            }
            for result in results
        ],
    )
    agg = aggregate(results)
    labels = verdict(agg)
    summary = {
        "aggregate": agg,
        "verdict": labels,
        "config": {
            "seeds": seeds,
            "arms": arms,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "probe_epochs": args.probe_epochs,
            "hidden": args.hidden,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out_dir, agg, labels, args)
    write_doc_result(agg, labels, args)
    print(json.dumps({"verdict": labels, "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
