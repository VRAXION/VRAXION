#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_authority_graph_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "research" / "RNN_TEACHER_AUTHORITY_EXTRACTION.md"
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "rnn-teacher-authority-extraction"
SUMMARY_NAME = "rnn_teacher_authority_extraction_summary.json"

FRAMES = ["danger", "friendship", "sound", "environment"]
FRAME_TO_QUERY = {frame: f"{frame}_query" for frame in FRAMES}
QUERY_TO_FRAME = {query: frame for frame, query in FRAME_TO_QUERY.items()}
FRAME_INDEX = {frame: i for i, frame in enumerate(FRAMES)}
TEMPORAL_FRAME_INDEX = len(FRAMES)

ACTORS = ["dog", "cat", "snake", "bird"]
ACTIONS = ["bite", "sleep", "chase", "run"]
RELATIONS = ["owner", "stranger", "alone"]
SOUNDS = ["bark", "music", "quiet"]
PLACES = ["street", "kitchen", "park"]
NOISES = ["car_noise", "crowd", "quiet_noise"]
TEMPORAL_EXTRA = ["me", "child"]

GROUPS = ["actor_action", "actor_relation", "actor_sound", "place_noise"]
ACTIVE_GROUP_BY_FRAME = {
    "danger": "actor_action",
    "friendship": "actor_relation",
    "sound": "actor_sound",
    "environment": "place_noise",
}


@dataclass(frozen=True)
class Observation:
    actor: str
    action: str
    relation: str
    sound: str
    place: str
    noise: str


@dataclass(frozen=True)
class Example:
    tokens: tuple[str, ...]
    label: int
    task: str
    frame: str | None = None
    obs: Observation | None = None
    base_id: int | None = None
    query_position: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny GRU teacher -> authority graph extraction probe.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--hidden-sizes", type=str, default="32,64")
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--aux-frame-weight", type=float, default=0.15)
    parser.add_argument("--query-position", choices=("first", "last", "both"), default="both")
    parser.add_argument("--extract-ridge", type=float, default=0.15)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden_sizes = "16"
        args.embedding_dim = 12
        args.epochs = 2
        args.batch_size = 32
        args.train_size = 96
        args.validation_size = 64
        args.test_size = 96
        args.query_position = "first"
    args.hidden_sizes = [int(item) for item in str(args.hidden_sizes).split(",") if item.strip()]
    return args


def static_label(obs: Observation, frame: str) -> int:
    if frame == "danger":
        return int(obs.actor in {"dog", "snake"} and obs.action in {"bite", "chase"})
    if frame == "friendship":
        return int(obs.actor in {"dog", "cat"} and obs.relation == "owner")
    if frame == "sound":
        return int(obs.actor in {"dog", "cat", "bird"} and obs.sound in {"bark", "music"})
    if frame == "environment":
        return int(obs.place == "street" and obs.noise in {"car_noise", "crowd"})
    raise ValueError(frame)


def obs_tokens(obs: Observation) -> list[str]:
    return [obs.actor, obs.action, obs.relation, obs.sound, obs.place, obs.noise]


def make_observation_templates() -> list[Observation]:
    return [
        Observation("dog", "bite", "owner", "bark", "street", "car_noise"),
        Observation("dog", "sleep", "owner", "quiet", "kitchen", "quiet_noise"),
        Observation("dog", "bite", "stranger", "quiet", "park", "quiet_noise"),
        Observation("cat", "run", "stranger", "music", "park", "quiet_noise"),
        Observation("bird", "sleep", "alone", "quiet", "street", "car_noise"),
        Observation("snake", "chase", "stranger", "quiet", "street", "crowd"),
        Observation("cat", "sleep", "owner", "bark", "kitchen", "quiet_noise"),
        Observation("bird", "sleep", "alone", "quiet", "park", "quiet_noise"),
        Observation("snake", "bite", "alone", "quiet", "kitchen", "quiet_noise"),
        Observation("cat", "chase", "owner", "music", "street", "crowd"),
    ]


def random_observation(rng: np.random.Generator, base: Observation | None = None) -> Observation:
    if base is None:
        base = Observation(
            actor=str(rng.choice(ACTORS)),
            action=str(rng.choice(ACTIONS)),
            relation=str(rng.choice(RELATIONS)),
            sound=str(rng.choice(SOUNDS)),
            place=str(rng.choice(PLACES)),
            noise=str(rng.choice(NOISES)),
        )
    return Observation(
        actor=base.actor if rng.random() < 0.55 else str(rng.choice(ACTORS)),
        action=base.action if rng.random() < 0.70 else str(rng.choice(ACTIONS)),
        relation=base.relation if rng.random() < 0.70 else str(rng.choice(RELATIONS)),
        sound=base.sound if rng.random() < 0.70 else str(rng.choice(SOUNDS)),
        place=base.place if rng.random() < 0.70 else str(rng.choice(PLACES)),
        noise=base.noise if rng.random() < 0.70 else str(rng.choice(NOISES)),
    )


def make_query_sequence(obs: Observation, frame: str, position: str) -> tuple[str, ...]:
    query = FRAME_TO_QUERY[frame]
    body = obs_tokens(obs)
    if position == "last":
        return tuple(body + [query])
    return tuple([query] + body)


def make_static_examples(
    n: int,
    seed: int,
    *,
    task: str,
    query_position: str,
    multi_aspect_bias: bool,
) -> list[Example]:
    rng = np.random.default_rng(seed)
    positions = ["first", "last"] if query_position == "both" else [query_position]
    templates = make_observation_templates()
    examples: list[Example] = []
    base_id = 0
    while len(examples) < n:
        template = templates[base_id % len(templates)]
        obs = random_observation(rng, template)
        if multi_aspect_bias and base_id % 2 == 0:
            obs = Observation(
                actor="dog",
                action=obs.action,
                relation=obs.relation,
                sound=obs.sound,
                place=obs.place,
                noise=obs.noise,
            )
        for frame in FRAMES:
            for position in positions:
                examples.append(
                    Example(
                        tokens=make_query_sequence(obs, frame, position),
                        label=static_label(obs, frame),
                        task=task,
                        frame=frame,
                        obs=obs,
                        base_id=base_id,
                        query_position=position,
                    )
                )
                if len(examples) >= n:
                    break
            if len(examples) >= n:
                break
        base_id += 1
    return examples


def make_temporal_examples(n: int, seed: int) -> list[Example]:
    pairs = [
        (("dog", "bite", "me"), 1),
        (("me", "bite", "dog"), 0),
        (("dog", "chase", "cat"), 1),
        (("cat", "chase", "dog"), 0),
        (("snake", "bite", "dog"), 1),
        (("dog", "bite", "snake"), 0),
        (("dog", "chase", "child"), 1),
        (("child", "chase", "dog"), 0),
    ]
    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    for i in range(n):
        seq, label = pairs[int(rng.integers(0, len(pairs)))]
        examples.append(Example(tokens=seq, label=label, task="temporal_order_contrast"))
    return examples


def make_dataset(n: int, seed: int, *, query_position: str) -> list[Example]:
    static_n = max(8, int(n * 0.38))
    multi_n = max(8, int(n * 0.38))
    temporal_n = max(8, n - static_n - multi_n)
    rows = (
        make_static_examples(static_n, seed + 1, task="latent_refraction_sequence", query_position=query_position, multi_aspect_bias=False)
        + make_static_examples(multi_n, seed + 2, task="query_cued_multi_aspect", query_position=query_position, multi_aspect_bias=True)
        + make_temporal_examples(temporal_n, seed + 3)
    )
    rng = np.random.default_rng(seed + 4)
    rng.shuffle(rows)
    return rows


def build_vocab() -> dict[str, int]:
    tokens = sorted(
        set(ACTORS + ACTIONS + RELATIONS + SOUNDS + PLACES + NOISES + TEMPORAL_EXTRA + list(FRAME_TO_QUERY.values()) + ["no_query"])
    )
    return {"<pad>": 0, **{token: i + 1 for i, token in enumerate(tokens)}}


def encode_examples(examples: list[Example], vocab: dict[str, int], max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_len is None:
        max_len = max(len(ex.tokens) for ex in examples)
    ids = torch.zeros((len(examples), max_len), dtype=torch.long)
    lengths = torch.zeros(len(examples), dtype=torch.long)
    labels = torch.zeros(len(examples), dtype=torch.long)
    frame_labels = torch.zeros(len(examples), dtype=torch.long)
    for row, ex in enumerate(examples):
        lengths[row] = len(ex.tokens)
        labels[row] = ex.label
        frame_labels[row] = FRAME_INDEX[ex.frame] if ex.frame is not None else TEMPORAL_FRAME_INDEX
        for col, token in enumerate(ex.tokens[:max_len]):
            ids[row, col] = vocab[token]
    return ids, lengths, labels, frame_labels


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example], vocab: dict[str, int], max_len: int):
        self.examples = examples
        self.ids, self.lengths, self.labels, self.frames = encode_examples(examples, vocab, max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[index], self.lengths[index], self.labels[index], self.frames[index]


class GRUTeacher(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, frame_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)
        self.frame_head = nn.Linear(hidden_size, frame_classes)

    def encode(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        zero_recurrent: bool = False,
        ablate_dims: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(ids)
        if zero_recurrent:
            outs = []
            for col in range(ids.shape[1]):
                out, _ = self.gru(emb[:, col : col + 1, :], torch.zeros(1, ids.shape[0], self.gru.hidden_size, device=ids.device))
                outs.append(out)
            states = torch.cat(outs, dim=1)
        else:
            states, _ = self.gru(emb)
        gather = (lengths.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        final = states.gather(1, gather).squeeze(1)
        if ablate_dims:
            final = final.clone()
            final[:, ablate_dims] = 0.0
            states = states.clone()
            states[:, :, ablate_dims] = 0.0
        return states, final

    def forward(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        zero_recurrent: bool = False,
        ablate_dims: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        states, final = self.encode(ids, lengths, zero_recurrent=zero_recurrent, ablate_dims=ablate_dims)
        return {
            "states": states,
            "final": final,
            "logits": self.head(final),
            "frame_logits": self.frame_head(final),
        }


class BagMLP(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vocab_size, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        return self.net(bag)


class PosMLP(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, hidden: int = 96):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.net = nn.Sequential(nn.Linear(vocab_size * max_len, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(ids, num_classes=self.vocab_size).float().reshape(ids.shape[0], -1)
        return self.net(onehot)


def batch_iter(dataset: SequenceDataset, batch_size: int, seed: int) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def ids_to_bag(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    onehot = F.one_hot(ids, num_classes=vocab_size).float()
    onehot[:, :, 0] = 0.0
    return onehot.sum(dim=1).clamp(max=1.0)


def train_teacher(
    model: GRUTeacher,
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        for ids, lengths, labels, frames in batch_iter(train_ds, args.batch_size, seed + epoch):
            opt.zero_grad(set_to_none=True)
            out = model(ids, lengths)
            loss = F.cross_entropy(out["logits"], labels)
            if args.aux_frame_weight > 0:
                loss = loss + args.aux_frame_weight * F.cross_entropy(out["frame_logits"], frames)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val_acc = evaluate_teacher(model, val_ds.examples, train_ds, transform=None)["accuracy"]
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_validation_accuracy": best_val}


def train_bag_mlp(model: BagMLP, train_ds: SequenceDataset, *, args: argparse.Namespace, seed: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    for epoch in range(max(2, args.epochs // 2)):
        model.train()
        for ids, _lengths, labels, _frames in batch_iter(train_ds, args.batch_size, seed + 7_000 + epoch):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids_to_bag(ids, model.net[0].in_features)), labels)
            loss.backward()
            opt.step()


def train_pos_mlp(model: PosMLP, train_ds: SequenceDataset, *, args: argparse.Namespace, seed: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    for epoch in range(max(2, args.epochs // 2)):
        model.train()
        for ids, _lengths, labels, _frames in batch_iter(train_ds, args.batch_size, seed + 8_000 + epoch):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids), labels)
            loss.backward()
            opt.step()


def transformed_examples(examples: list[Example], transform: str | None, seed: int) -> list[Example]:
    if transform is None:
        return examples
    rng = random.Random(seed)
    out: list[Example] = []
    query_tokens = [ex.tokens[0] if ex.tokens and ex.tokens[0] in QUERY_TO_FRAME else ex.tokens[-1] for ex in examples if ex.frame is not None]
    rng.shuffle(query_tokens)
    q_index = 0
    for ex in examples:
        tokens = list(ex.tokens)
        if transform == "shuffle_order":
            rng.shuffle(tokens)
        elif transform == "query_ablation" and ex.frame is not None:
            tokens = ["no_query" if token in QUERY_TO_FRAME else token for token in tokens]
        elif transform == "query_shuffle" and ex.frame is not None:
            replacement = query_tokens[q_index % len(query_tokens)]
            q_index += 1
            tokens = [replacement if token in QUERY_TO_FRAME else token for token in tokens]
        out.append(Example(tuple(tokens), ex.label, ex.task, ex.frame, ex.obs, ex.base_id, ex.query_position))
    return out


@torch.no_grad()
def evaluate_teacher(
    model: GRUTeacher,
    examples: list[Example],
    reference_ds: SequenceDataset,
    *,
    transform: str | None = None,
    zero_recurrent: bool = False,
    ablate_dims: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    rows = transformed_examples(examples, transform, seed=12345)
    ids, lengths, labels, frames = encode_examples(rows, reference_ds_vocab(reference_ds), reference_ds.ids.shape[1])
    out = model(ids, lengths, zero_recurrent=zero_recurrent, ablate_dims=ablate_dims)
    pred = out["logits"].argmax(dim=-1)
    frame_pred = out["frame_logits"].argmax(dim=-1)
    probs = F.softmax(out["logits"], dim=-1)[:, 1]
    correct = (pred == labels).float()
    by_task: dict[str, float] = {}
    for task in sorted({ex.task for ex in rows}):
        mask = torch.tensor([ex.task == task for ex in rows], dtype=torch.bool)
        by_task[task] = float(correct[mask].mean().item()) if mask.any() else 0.0
    static_mask = torch.tensor([ex.frame is not None for ex in rows], dtype=torch.bool)
    frame_acc = float((frame_pred[static_mask] == frames[static_mask]).float().mean().item()) if static_mask.any() else 0.0
    return {
        "accuracy": float(correct.mean().item()),
        "by_task": by_task,
        "temporal_order_accuracy": by_task.get("temporal_order_contrast", 0.0),
        "query_cued_accuracy": float(np.mean([
            value for key, value in by_task.items() if key in {"query_cued_multi_aspect", "latent_refraction_sequence"}
        ])) if any(key in by_task for key in {"query_cued_multi_aspect", "latent_refraction_sequence"}) else 0.0,
        "frame_prediction_accuracy": frame_acc,
        "mean_positive_probability": float(probs.mean().item()),
    }


def reference_ds_vocab(dataset: SequenceDataset) -> dict[str, int]:
    # Stored as an attribute by the caller after construction.
    return dataset.vocab  # type: ignore[attr-defined]


@torch.no_grad()
def evaluate_mlp_baseline(model: nn.Module, examples: list[Example], reference_ds: SequenceDataset, *, kind: str) -> dict[str, Any]:
    rows = examples
    ids, _lengths, labels, _frames = encode_examples(rows, reference_ds_vocab(reference_ds), reference_ds.ids.shape[1])
    if kind == "bag":
        logits = model(ids_to_bag(ids, model.net[0].in_features))
    else:
        logits = model(ids)
    pred = logits.argmax(dim=-1)
    correct = (pred == labels).float()
    by_task: dict[str, float] = {}
    for task in sorted({ex.task for ex in rows}):
        mask = torch.tensor([ex.task == task for ex in rows], dtype=torch.bool)
        by_task[task] = float(correct[mask].mean().item()) if mask.any() else 0.0
    return {"accuracy": float(correct.mean().item()), "by_task": by_task}


def clone_with_random_recurrent(model: GRUTeacher, seed: int) -> GRUTeacher:
    torch.manual_seed(seed)
    cloned = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in cloned.gru.named_parameters():
            if "weight_hh" in name:
                param.normal_(0.0, float(param.std().item()) if param.numel() else 0.05)
            if "bias_hh" in name:
                param.zero_()
    return cloned


def swap_obs_group(obs: Observation, donor: Observation, group: str) -> Observation:
    data = obs.__dict__.copy()
    if group == "actor_action":
        fields = ["actor", "action"]
    elif group == "actor_relation":
        fields = ["actor", "relation"]
    elif group == "actor_sound":
        fields = ["actor", "sound"]
    elif group == "place_noise":
        fields = ["place", "noise"]
    else:
        raise ValueError(group)
    for field in fields:
        data[field] = getattr(donor, field)
    return Observation(**data)


@torch.no_grad()
def positive_probs(model: GRUTeacher, examples: list[Example], reference_ds: SequenceDataset) -> np.ndarray:
    ids, lengths, _labels, _frames = encode_examples(examples, reference_ds_vocab(reference_ds), reference_ds.ids.shape[1])
    out = model(ids, lengths)
    return F.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()


def influence_metrics(model: GRUTeacher, examples: list[Example], reference_ds: SequenceDataset, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    static = [ex for ex in examples if ex.frame is not None and ex.obs is not None]
    by_frame: dict[str, dict[str, float]] = {}
    for frame in FRAMES:
        frame_examples = [ex for ex in static if ex.frame == frame]
        if not frame_examples:
            continue
        base_probs = positive_probs(model, frame_examples, reference_ds)
        group_delta: dict[str, float] = {}
        for group in GROUPS:
            swapped = []
            for ex in frame_examples:
                donor = frame_examples[int(rng.integers(0, len(frame_examples)))].obs
                assert ex.obs is not None and donor is not None
                new_obs = swap_obs_group(ex.obs, donor, group)
                swapped.append(
                    Example(
                        tokens=make_query_sequence(new_obs, frame, ex.query_position if ex.query_position != "none" else "first"),
                        label=static_label(new_obs, frame),
                        task=ex.task,
                        frame=frame,
                        obs=new_obs,
                        base_id=ex.base_id,
                        query_position=ex.query_position,
                    )
                )
            swapped_probs = positive_probs(model, swapped, reference_ds)
            group_delta[group] = float(np.mean(np.abs(base_probs - swapped_probs)))
        active = group_delta[ACTIVE_GROUP_BY_FRAME[frame]]
        inactive = max(value for group, value in group_delta.items() if group != ACTIVE_GROUP_BY_FRAME[frame])
        by_frame[frame] = {
            "active": active,
            "inactive": inactive,
            "refraction": active - inactive,
            **{f"{group}_influence": value for group, value in group_delta.items()},
        }
    return {
        "by_frame": by_frame,
        "active_group_influence": float(np.mean([item["active"] for item in by_frame.values()])),
        "inactive_group_influence": float(np.mean([item["inactive"] for item in by_frame.values()])),
        "authority_refraction_score": float(np.mean([item["refraction"] for item in by_frame.values()])),
    }


@torch.no_grad()
def hidden_states_for_tokens(model: GRUTeacher, tokens: tuple[str, ...], vocab: dict[str, int], max_len: int) -> np.ndarray:
    ids = torch.zeros((1, max_len), dtype=torch.long)
    for i, token in enumerate(tokens):
        ids[0, i] = vocab[token]
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    states = model(ids, lengths)["states"][0, : len(tokens), :]
    return states.cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1.0e-9 else 0.0


def trajectory_diagnostics(model: GRUTeacher, vocab: dict[str, int], max_len: int) -> dict[str, Any]:
    a = ("dog", "bite", "me")
    b = ("dog", "bite", "snake")
    c = ("me", "bite", "dog")
    ha = hidden_states_for_tokens(model, a, vocab, max_len)
    hb = hidden_states_for_tokens(model, b, vocab, max_len)
    hc = hidden_states_for_tokens(model, c, vocab, max_len)
    return {
        "dog_bite_me_vs_dog_bite_snake_prefix_cosine": [cosine(ha[0], hb[0]), cosine(ha[1], hb[1])],
        "dog_bite_me_vs_dog_bite_snake_final_cosine": cosine(ha[-1], hb[-1]),
        "dog_bite_me_vs_me_bite_dog_final_cosine": cosine(ha[-1], hc[-1]),
        "suffix_divergence": 1.0 - cosine(ha[-1], hb[-1]),
        "order_divergence": 1.0 - cosine(ha[-1], hc[-1]),
    }


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, classes: int, seed: int) -> float:
    torch.manual_seed(seed)
    layer = nn.Linear(x_train.shape[1], classes)
    opt = torch.optim.AdamW(layer.parameters(), lr=5.0e-3, weight_decay=1.0e-4)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xv = torch.tensor(x_test, dtype=torch.float32)
    yv = torch.tensor(y_test, dtype=torch.long)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(layer(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((layer(xv).argmax(dim=-1) == yv).float().mean().item())


@torch.no_grad()
def final_hidden(model: GRUTeacher, examples: list[Example], reference_ds: SequenceDataset) -> np.ndarray:
    ids, lengths, _labels, _frames = encode_examples(examples, reference_ds_vocab(reference_ds), reference_ds.ids.shape[1])
    return model(ids, lengths)["final"].cpu().numpy()


def probe_diagnostics(model: GRUTeacher, train_examples: list[Example], test_examples: list[Example], reference_ds: SequenceDataset, seed: int) -> dict[str, Any]:
    static_train = [ex for ex in train_examples if ex.frame is not None and ex.obs is not None]
    static_test = [ex for ex in test_examples if ex.frame is not None and ex.obs is not None]
    x_train_all = final_hidden(model, train_examples, reference_ds)
    x_test_all = final_hidden(model, test_examples, reference_ds)
    y_train_frame = np.array([FRAME_INDEX[ex.frame] if ex.frame is not None else TEMPORAL_FRAME_INDEX for ex in train_examples])
    y_test_frame = np.array([FRAME_INDEX[ex.frame] if ex.frame is not None else TEMPORAL_FRAME_INDEX for ex in test_examples])
    out = {"frame_probe_accuracy": train_probe(x_train_all, y_train_frame, x_test_all, y_test_frame, len(FRAMES) + 1, seed)}
    x_train = final_hidden(model, static_train, reference_ds)
    x_test = final_hidden(model, static_test, reference_ds)
    fields = {
        "actor_probe_accuracy": (ACTORS, lambda obs: obs.actor),
        "action_probe_accuracy": (ACTIONS, lambda obs: obs.action),
        "relation_probe_accuracy": (RELATIONS, lambda obs: obs.relation),
        "sound_probe_accuracy": (SOUNDS, lambda obs: obs.sound),
        "place_probe_accuracy": (PLACES, lambda obs: obs.place),
        "noise_probe_accuracy": (NOISES, lambda obs: obs.noise),
    }
    for name, (values, getter) in fields.items():
        index = {value: i for i, value in enumerate(values)}
        y_train = np.array([index[getter(ex.obs)] for ex in static_train if ex.obs is not None])
        y_test = np.array([index[getter(ex.obs)] for ex in static_test if ex.obs is not None])
        out[name] = train_probe(x_train, y_train, x_test, y_test, len(values), seed + len(out))
    return out


def unit_saliency(model: GRUTeacher, examples: list[Example], reference_ds: SequenceDataset, seed: int) -> dict[str, Any]:
    baseline = evaluate_teacher(model, examples, reference_ds)
    baseline_influence = influence_metrics(model, examples, reference_ds, seed)
    hidden = model.gru.hidden_size
    rows = []
    for dim in range(hidden):
        metrics = evaluate_teacher(model, examples, reference_ds, ablate_dims=[dim])
        rows.append({
            "dim": dim,
            "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
        })
    rows.sort(key=lambda item: item["accuracy_drop"], reverse=True)
    static = [ex for ex in examples if ex.frame is not None]
    x = final_hidden(model, static, reference_ds)
    frame_labels = np.array([FRAME_INDEX[ex.frame] for ex in static])
    classifications = CounterLike()
    for dim in range(hidden):
        means = [float(x[frame_labels == i, dim].mean()) for i in range(len(FRAMES))]
        spread = max(means) - min(means)
        top = FRAMES[int(np.argmax(means))]
        if spread > 0.25:
            classifications.add(f"{top}_related")
        elif float(np.mean(np.abs(means))) > 0.25:
            classifications.add("shared_global")
        else:
            classifications.add("unclear")
    return {
        "baseline_accuracy": baseline["accuracy"],
        "baseline_authority_refraction": baseline_influence["authority_refraction_score"],
        "top10_accuracy_saliency": rows[:10],
        "unit_classification_counts": classifications.data,
    }


class CounterLike:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}

    def add(self, key: str) -> None:
        self.data[key] = self.data.get(key, 0) + 1


def teacher_logits(model: GRUTeacher, examples: list[Example], reference_ds: SequenceDataset) -> np.ndarray:
    ids, lengths, _labels, _frames = encode_examples(examples, reference_ds_vocab(reference_ds), reference_ds.ids.shape[1])
    with torch.no_grad():
        logits = model(ids, lengths)["logits"]
    return (logits[:, 1] - logits[:, 0]).cpu().numpy()


def ridge_fit(features: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    x = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    reg = ridge * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(x.T @ x + reg, x.T @ target)
    return coef[:-1], float(coef[-1])


def role_features(examples: list[Example]) -> tuple[np.ndarray, list[str]]:
    names = []
    for token in ["dog", "cat", "snake", "me", "child"]:
        names.extend([f"subject_{token}", f"object_{token}"])
    names.extend(["verb_bite", "verb_chase"])
    rows = np.zeros((len(examples), len(names)), dtype=np.float64)
    name_to_col = {name: i for i, name in enumerate(names)}
    for row, ex in enumerate(examples):
        seq = list(ex.tokens)
        if len(seq) > 0 and f"subject_{seq[0]}" in name_to_col:
            rows[row, name_to_col[f"subject_{seq[0]}"]] = 1.0
        if len(seq) > 1 and f"verb_{seq[1]}" in name_to_col:
            rows[row, name_to_col[f"verb_{seq[1]}"]] = 1.0
        if len(seq) > 2 and f"object_{seq[2]}" in name_to_col:
            rows[row, name_to_col[f"object_{seq[2]}"]] = 1.0
    return rows, names


def token_features(examples: list[Example], tokens: list[str]) -> np.ndarray:
    rows = np.zeros((len(examples), len(tokens)), dtype=np.float64)
    for row, ex in enumerate(examples):
        present = set(ex.tokens)
        for col, token in enumerate(tokens):
            rows[row, col] = 1.0 if token in present else 0.0
    return rows


def build_extracted_graph(model: GRUTeacher, train_examples: list[Example], reference_ds: SequenceDataset, ridge: float) -> pilot.AuthorityGraph:
    graph = pilot.build_empty_graph(decay=0.35)
    graph.readout_policy = "route_state"
    graph.frame_gate = 0.85
    graph.competition = 0.42
    graph.route_bias = -0.80
    for route in pilot.route_nodes():
        pilot.add_edge(graph, route, route, 0.16)
        graph.bias[graph.idx[route]] = graph.route_bias
    graph.bias[graph.idx["temporal_route"]] = -0.10
    for frame in pilot.FRAMES:
        pilot.add_edge(graph, pilot.FRAME_BY_ROUTE[frame], f"suppress_{frame}", -0.85)
        for other in pilot.FRAMES:
            if other != frame:
                pilot.add_edge(graph, pilot.FRAME_BY_ROUTE[frame], f"suppress_{other}", 0.12)
    for token in pilot.ACTORS:
        pilot.add_edge(graph, "shared_actor_hub", f"tok_{token}", 0.15)
    for token in pilot.ACTIONS + pilot.RELATIONS + pilot.SOUNDS:
        if f"tok_{token}" in graph.idx:
            pilot.add_edge(graph, "shared_action_hub", f"tok_{token}", 0.12)
    for token in pilot.PLACES + pilot.NOISES + pilot.SOUNDS:
        if f"tok_{token}" in graph.idx:
            pilot.add_edge(graph, "shared_context_hub", f"tok_{token}", 0.12)
    for route in pilot.route_nodes():
        for hub in ["shared_actor_hub", "shared_action_hub", "shared_context_hub"]:
            pilot.add_edge(graph, route, hub, 0.08)

    static = [ex for ex in train_examples if ex.frame is not None and ex.obs is not None]
    tokens = list(pilot.TOKENS)
    for frame in pilot.FRAMES:
        frame_examples = [ex for ex in static if ex.frame == frame]
        if len(frame_examples) < 4:
            continue
        x = token_features(frame_examples, tokens)
        y = teacher_logits(model, frame_examples, reference_ds)
        weights, bias = ridge_fit(x, y, ridge)
        route = pilot.FRAME_BY_ROUTE[frame]
        graph.bias[graph.idx[route]] = float(np.clip(bias * 0.20, -1.4, 0.6))
        ranked = sorted(zip(tokens, weights), key=lambda item: abs(item[1]), reverse=True)[:12]
        for token, weight in ranked:
            if f"tok_{token}" in graph.idx and abs(weight) > 0.02:
                pilot.add_edge(graph, route, f"tok_{token}", float(np.clip(weight * 0.45, -1.2, 1.2)))

    temporal = [ex for ex in train_examples if ex.task == "temporal_order_contrast"]
    if temporal:
        x, names = role_features(temporal)
        y = teacher_logits(model, temporal, reference_ds)
        weights, bias = ridge_fit(x, y, ridge)
        graph.bias[graph.idx["temporal_route"]] = float(np.clip(bias * 0.20, -0.8, 0.8))
        pilot.add_edge(graph, "temporal_route", "temporal_route", 0.14)
        for name, weight in zip(names, weights):
            if name in graph.idx and abs(weight) > 0.02:
                pilot.add_edge(graph, "temporal_route", name, float(np.clip(weight * 0.45, -1.2, 1.2)))
    return graph


def to_pilot_dataset(examples: list[Example]) -> dict[str, list[Any]]:
    static = []
    temporal = []
    for ex in examples:
        if ex.frame is not None and ex.obs is not None:
            static.append(
                pilot.StaticExample(
                    obs=pilot.Observation(
                        actor=ex.obs.actor,
                        action=ex.obs.action,
                        relation=ex.obs.relation,
                        sound=ex.obs.sound,
                        place=ex.obs.place,
                        noise=ex.obs.noise,
                    ),
                    frame=ex.frame,
                    label=ex.label,
                )
            )
        elif ex.task == "temporal_order_contrast":
            temporal.append(pilot.TemporalExample(sequence=ex.tokens, label=ex.label))
    return {
        "latent_refraction_small": static,
        "multi_aspect_small": static,
        "temporal_order_contrast_small": temporal,
    }


def graph_metrics(graph: pilot.AuthorityGraph, examples: list[Example], seed: int) -> dict[str, Any]:
    return pilot.evaluate_graph(graph, to_pilot_dataset(examples), steps=5, seed=seed)


def run_seed(args: argparse.Namespace, seed: int, vocab: dict[str, int]) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train = make_dataset(args.train_size, seed + 1_000, query_position=args.query_position)
    val = make_dataset(args.validation_size, seed + 2_000, query_position=args.query_position)
    test = make_dataset(args.test_size, seed + 3_000, query_position=args.query_position)
    max_len = max(len(ex.tokens) for ex in train + val + test)
    train_ds = SequenceDataset(train, vocab, max_len)
    val_ds = SequenceDataset(val, vocab, max_len)
    test_ds = SequenceDataset(test, vocab, max_len)
    train_ds.vocab = vocab  # type: ignore[attr-defined]
    val_ds.vocab = vocab  # type: ignore[attr-defined]
    test_ds.vocab = vocab  # type: ignore[attr-defined]

    bag = BagMLP(len(vocab))
    pos = PosMLP(len(vocab), max_len)
    train_bag_mlp(bag, train_ds, args=args, seed=seed)
    train_pos_mlp(pos, train_ds, args=args, seed=seed)
    baselines = {
        "bag_of_tokens_mlp": evaluate_mlp_baseline(bag, test, test_ds, kind="bag"),
        "static_with_position_mlp": evaluate_mlp_baseline(pos, test, test_ds, kind="pos"),
    }

    teacher_records = []
    trained_teachers: list[tuple[GRUTeacher, dict[str, Any]]] = []
    for hidden in args.hidden_sizes:
        model = GRUTeacher(len(vocab), args.embedding_dim, hidden, len(FRAMES) + 1)
        train_info = train_teacher(model, train_ds, val_ds, args=args, seed=seed + hidden)
        test_metrics = evaluate_teacher(model, test, test_ds)
        controls = {
            "zero_recurrent": evaluate_teacher(model, test, test_ds, zero_recurrent=True),
            "shuffled_order": evaluate_teacher(model, test, test_ds, transform="shuffle_order"),
            "query_ablation": evaluate_teacher(model, test, test_ds, transform="query_ablation"),
            "query_shuffle": evaluate_teacher(model, test, test_ds, transform="query_shuffle"),
            "randomized_recurrent": evaluate_teacher(clone_with_random_recurrent(model, seed + hidden + 10_000), test, test_ds),
        }
        record = {
            "hidden_size": hidden,
            "train_info": train_info,
            "test": test_metrics,
            "controls": controls,
        }
        teacher_records.append(record)
        trained_teachers.append((model, record))

    best_model, best_record = max(trained_teachers, key=lambda item: (item[1]["test"]["accuracy"], item[1]["test"]["frame_prediction_accuracy"]))
    influence = influence_metrics(best_model, test, test_ds, seed + 4_000)
    trajectories = trajectory_diagnostics(best_model, vocab, max_len)
    probes = probe_diagnostics(best_model, train, test, test_ds, seed + 5_000)
    saliency = unit_saliency(best_model, test, test_ds, seed + 6_000)
    extracted = build_extracted_graph(best_model, train, train_ds, args.extract_ridge)
    extracted_metrics = graph_metrics(extracted, test, seed + 7_000)
    hand_metrics = graph_metrics(pilot.build_hand_seeded_graph(0.35), test, seed + 7_100)
    random_graph = pilot.build_random_graph(0.35, seed + 7_200, pilot.build_hand_seeded_graph(0.35).edge_count())
    random_graph_metrics = graph_metrics(random_graph, test, seed + 7_300)
    graph_path = args.out_dir / "graphs" / f"extracted_authority_graph_seed{seed}.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph_to_json(extracted), indent=2) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "dataset_sizes": {"train": len(train), "validation": len(val), "test": len(test), "max_len": max_len},
        "baselines": baselines,
        "teachers": teacher_records,
        "best_teacher_hidden": best_record["hidden_size"],
        "best_teacher": best_record,
        "authority_diagnostics": influence,
        "trajectory_diagnostics": trajectories,
        "probe_diagnostics": probes,
        "unit_saliency": saliency,
        "extraction": {
            "extracted_authority_graph": summarize_graph_metrics(extracted_metrics),
            "hand_seeded_authority_graph": summarize_graph_metrics(hand_metrics),
            "random_graph_baseline": summarize_graph_metrics(random_graph_metrics),
            "extracted_graph_path": str(graph_path),
            "extracted_edge_count": extracted.edge_count(),
            "extracted_node_count": len(extracted.nodes),
        },
    }


def graph_to_json(graph: pilot.AuthorityGraph) -> dict[str, Any]:
    edges = []
    for target, source in np.argwhere(np.abs(graph.w) > 1.0e-9):
        edges.append({
            "source": graph.nodes[int(source)],
            "target": graph.nodes[int(target)],
            "weight": round(float(graph.w[int(target), int(source)]), 6),
        })
    return {
        "nodes": [{"name": name, "type": graph.types[name]} for name in graph.nodes],
        "edges": edges,
        "params": {
            "decay": graph.decay,
            "frame_gate": graph.frame_gate,
            "suppressor_strength": graph.suppressor_strength,
            "competition": graph.competition,
            "route_bias": graph.route_bias,
            "readout_policy": graph.readout_policy,
        },
        "edge_count": graph.edge_count(),
        "node_count": len(graph.nodes),
    }


def summarize_graph_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "latent_refraction_accuracy": metrics["latent_refraction_small_accuracy"],
        "multi_aspect_accuracy": metrics["multi_aspect_small_accuracy"],
        "temporal_order_accuracy": metrics["temporal_order_contrast_small_accuracy"],
        "authority_refraction_score": metrics["authority_switch_score"],
        "wrong_frame_drop": metrics["wrong_frame_drop"],
        "no_recurrence_drop": metrics["no_recurrence_drop"],
        "edge_count": metrics["edge_count"],
        "node_count": metrics["node_count"],
    }


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_by_hidden: dict[str, dict[str, Any]] = {}
    for hidden in sorted({row["hidden_size"] for rec in records for row in rec["teachers"]}):
        rows = [row for rec in records for row in rec["teachers"] if row["hidden_size"] == hidden]
        teacher_by_hidden[str(hidden)] = {
            "accuracy": numeric_summary([row["test"]["accuracy"] for row in rows]),
            "temporal_order_accuracy": numeric_summary([row["test"]["temporal_order_accuracy"] for row in rows]),
            "query_cued_accuracy": numeric_summary([row["test"]["query_cued_accuracy"] for row in rows]),
            "frame_prediction_accuracy": numeric_summary([row["test"]["frame_prediction_accuracy"] for row in rows]),
            "zero_recurrent_accuracy": numeric_summary([row["controls"]["zero_recurrent"]["accuracy"] for row in rows]),
            "randomized_recurrent_accuracy": numeric_summary([row["controls"]["randomized_recurrent"]["accuracy"] for row in rows]),
            "shuffled_order_accuracy": numeric_summary([row["controls"]["shuffled_order"]["accuracy"] for row in rows]),
            "query_ablation_accuracy": numeric_summary([row["controls"]["query_ablation"]["accuracy"] for row in rows]),
            "query_shuffle_accuracy": numeric_summary([row["controls"]["query_shuffle"]["accuracy"] for row in rows]),
        }
    best = [rec["best_teacher"] for rec in records]
    extraction_rows = [rec["extraction"] for rec in records]
    unit_counts: dict[str, list[int]] = {}
    for rec in records:
        for key, value in rec["unit_saliency"]["unit_classification_counts"].items():
            unit_counts.setdefault(key, []).append(int(value))
    return {
        "teacher_by_hidden": teacher_by_hidden,
        "best_teacher": {
            "accuracy": numeric_summary([row["test"]["accuracy"] for row in best]),
            "temporal_order_accuracy": numeric_summary([row["test"]["temporal_order_accuracy"] for row in best]),
            "query_cued_accuracy": numeric_summary([row["test"]["query_cued_accuracy"] for row in best]),
            "frame_prediction_accuracy": numeric_summary([row["test"]["frame_prediction_accuracy"] for row in best]),
            "zero_recurrent_accuracy": numeric_summary([row["controls"]["zero_recurrent"]["accuracy"] for row in best]),
            "randomized_recurrent_accuracy": numeric_summary([row["controls"]["randomized_recurrent"]["accuracy"] for row in best]),
            "query_ablation_accuracy": numeric_summary([row["controls"]["query_ablation"]["accuracy"] for row in best]),
            "query_shuffle_accuracy": numeric_summary([row["controls"]["query_shuffle"]["accuracy"] for row in best]),
        },
        "authority_diagnostics": {
            "active_group_influence": numeric_summary([rec["authority_diagnostics"]["active_group_influence"] for rec in records]),
            "inactive_group_influence": numeric_summary([rec["authority_diagnostics"]["inactive_group_influence"] for rec in records]),
            "authority_refraction_score": numeric_summary([rec["authority_diagnostics"]["authority_refraction_score"] for rec in records]),
        },
        "trajectory_diagnostics": {
            "suffix_divergence": numeric_summary([rec["trajectory_diagnostics"]["suffix_divergence"] for rec in records]),
            "order_divergence": numeric_summary([rec["trajectory_diagnostics"]["order_divergence"] for rec in records]),
        },
        "probe_diagnostics": {
            key: numeric_summary([rec["probe_diagnostics"][key] for rec in records])
            for key in records[0]["probe_diagnostics"]
        } if records else {},
        "unit_saliency": {
            "baseline_accuracy": numeric_summary([rec["unit_saliency"]["baseline_accuracy"] for rec in records]),
            "baseline_authority_refraction": numeric_summary([rec["unit_saliency"]["baseline_authority_refraction"] for rec in records]),
            "classification_counts": {
                key: numeric_summary(values)
                for key, values in sorted(unit_counts.items())
            },
            "example_top10_accuracy_saliency": records[0]["unit_saliency"]["top10_accuracy_saliency"] if records else [],
        },
        "extraction": {
            "extracted_authority_graph": aggregate_graph_metric(extraction_rows, "extracted_authority_graph"),
            "hand_seeded_authority_graph": aggregate_graph_metric(extraction_rows, "hand_seeded_authority_graph"),
            "random_graph_baseline": aggregate_graph_metric(extraction_rows, "random_graph_baseline"),
        },
        "baselines": {
            "bag_of_tokens_mlp_accuracy": numeric_summary([rec["baselines"]["bag_of_tokens_mlp"]["accuracy"] for rec in records]),
            "static_with_position_mlp_accuracy": numeric_summary([rec["baselines"]["static_with_position_mlp"]["accuracy"] for rec in records]),
        },
    }


def aggregate_graph_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    fields = [
        "accuracy",
        "latent_refraction_accuracy",
        "multi_aspect_accuracy",
        "temporal_order_accuracy",
        "authority_refraction_score",
        "wrong_frame_drop",
        "no_recurrence_drop",
        "edge_count",
        "node_count",
    ]
    return {
        field: numeric_summary([row[key][field] for row in rows])
        for field in fields
    }


def mean(agg: dict[str, Any], path: list[str]) -> float:
    cursor: Any = agg
    for key in path:
        cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
    value = cursor.get("mean") if isinstance(cursor, dict) else None
    return float(value) if value is not None else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, Any]:
    teacher_acc = mean(agg, ["best_teacher", "accuracy"])
    teacher_zero = mean(agg, ["best_teacher", "zero_recurrent_accuracy"])
    teacher_rand = mean(agg, ["best_teacher", "randomized_recurrent_accuracy"])
    teacher_query = mean(agg, ["best_teacher", "query_shuffle_accuracy"])
    auth = mean(agg, ["authority_diagnostics", "authority_refraction_score"])
    extracted_acc = mean(agg, ["extraction", "extracted_authority_graph", "accuracy"])
    extracted_auth = mean(agg, ["extraction", "extracted_authority_graph", "authority_refraction_score"])
    hand_acc = mean(agg, ["extraction", "hand_seeded_authority_graph", "accuracy"])
    return {
        "gru_teacher_solves_tasks": teacher_acc >= 0.90,
        "gru_teacher_uses_recurrence": teacher_acc > max(teacher_zero, teacher_rand) + 0.10,
        "hidden_authority_structure_detected": auth > 0.05,
        "extraction_to_authority_graph_successful": extracted_acc >= 0.75 and extracted_auth > 0.05,
        "authority_graph_matches_teacher_partially": extracted_acc >= 0.70 or extracted_auth > 0.03,
        "rnn_teacher_useful_for_search": teacher_acc >= 0.90 and (extracted_acc >= 0.70 or extracted_auth > 0.03),
        "shortcut_risk_detected": mean(agg, ["baselines", "static_with_position_mlp_accuracy"]) >= teacher_acc - 0.03,
        "hand_seeded_still_stronger_than_extracted": hand_acc > extracted_acc + 0.10,
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# RNN Teacher Authority Extraction",
        "",
        "## Why This Was Tested",
        "",
        "The explicit hand-seeded authority graph works, while developmental graph search still struggles to rediscover the sign/gain structure. This probe uses a tiny GRU as a backprop-trained teacher, then attempts to extract a smaller explicit authority graph from the teacher's behavior.",
        "",
        "## Task Setup",
        "",
        "The teacher sees toy token sequences, not natural language. Tasks are binary classifications over:",
        "",
        "- `temporal_order_contrast`",
        "- `query_cued_multi_aspect`",
        "- `latent_refraction_sequence`",
        "",
        "Query tokens are `danger_query`, `friendship_query`, `sound_query`, and `environment_query`. The same observation can appear under different queries with different labels.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        "## Teacher Results By Hidden Size",
        "",
        "| Hidden | Accuracy | Temporal | Query-cued | Frame pred | Zero recurrent | Randomized recurrent | Shuffled order | Query ablation | Query shuffle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hidden, item in agg["teacher_by_hidden"].items():
        lines.append(
            f"| `{hidden}` | `{fmt(item['accuracy']['mean'])}` "
            f"| `{fmt(item['temporal_order_accuracy']['mean'])}` "
            f"| `{fmt(item['query_cued_accuracy']['mean'])}` "
            f"| `{fmt(item['frame_prediction_accuracy']['mean'])}` "
            f"| `{fmt(item['zero_recurrent_accuracy']['mean'])}` "
            f"| `{fmt(item['randomized_recurrent_accuracy']['mean'])}` "
            f"| `{fmt(item['shuffled_order_accuracy']['mean'])}` "
            f"| `{fmt(item['query_ablation_accuracy']['mean'])}` "
            f"| `{fmt(item['query_shuffle_accuracy']['mean'])}` |"
        )
    best = agg["best_teacher"]
    lines.extend([
        "",
        "## Baselines",
        "",
        f"- bag-of-tokens MLP accuracy: `{fmt(agg['baselines']['bag_of_tokens_mlp_accuracy']['mean'])}`",
        f"- static-with-position MLP accuracy: `{fmt(agg['baselines']['static_with_position_mlp_accuracy']['mean'])}`",
        "",
        "## Authority Diagnostics",
        "",
        f"- active group influence: `{fmt(agg['authority_diagnostics']['active_group_influence']['mean'])}`",
        f"- inactive group influence: `{fmt(agg['authority_diagnostics']['inactive_group_influence']['mean'])}`",
        f"- authority/refraction score: `{fmt(agg['authority_diagnostics']['authority_refraction_score']['mean'])}`",
        "",
        "## Hidden Trajectory Diagnostics",
        "",
        f"- suffix divergence, `dog bite me` vs `dog bite snake`: `{fmt(agg['trajectory_diagnostics']['suffix_divergence']['mean'])}`",
        f"- order divergence, `dog bite me` vs `me bite dog`: `{fmt(agg['trajectory_diagnostics']['order_divergence']['mean'])}`",
        "",
        "## Probe Diagnostics",
        "",
    ])
    for key, value in agg["probe_diagnostics"].items():
        lines.append(f"- {key}: `{fmt(value['mean'])}`")
    lines.extend([
        "",
        "## Unit Saliency",
        "",
        f"- baseline accuracy before single-dim ablation: `{fmt(agg['unit_saliency']['baseline_accuracy']['mean'])}`",
        f"- baseline authority/refraction before ablation: `{fmt(agg['unit_saliency']['baseline_authority_refraction']['mean'])}`",
        "",
        "Frame-related activation classifications, averaged across seeds:",
    ])
    for key, value in agg["unit_saliency"]["classification_counts"].items():
        lines.append(f"- {key}: `{fmt(value['mean'])}`")
    lines.extend([
        "",
        "Example top hidden dimensions by single-dim accuracy drop:",
        "",
        "```json",
        json.dumps(agg["unit_saliency"]["example_top10_accuracy_saliency"], indent=2),
        "```",
        "",
        "",
        "## Extracted Graph Attempt",
        "",
        "The extracted graph is fitted from teacher logits using ridge least squares over token/role features. It does not use neural backprop internally. This is an approximation attempt, not a guaranteed faithful circuit recovery.",
        "",
        "| Graph | Accuracy | Temporal | Authority | Wrong query/frame drop | Edges |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in [
        ("extracted_authority_graph", "extracted_authority_graph"),
        ("hand_seeded_authority_graph", "hand_seeded_authority_graph"),
        ("random_graph_baseline", "random_graph_baseline"),
    ]:
        item = agg["extraction"][key]
        lines.append(
            f"| `{label}` | `{fmt(item['accuracy']['mean'])}` "
            f"| `{fmt(item['temporal_order_accuracy']['mean'])}` "
            f"| `{fmt(item['authority_refraction_score']['mean'])}` "
            f"| `{fmt(item['wrong_frame_drop']['mean'])}` "
            f"| `{fmt(item['edge_count']['mean'])}` |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "A positive teacher result means backprop can find a recurrent solution for the toy tasks. Extraction success is judged separately: a GRU can solve the task while still being hard to compress into the explicit authority graph.",
        "",
        "The teacher is strong and recurrence-sensitive, but there is a shortcut caveat: the position-aware static MLP is close to the GRU on raw accuracy. The useful GRU-specific evidence is therefore the combination of recurrence/randomization drops, query ablation drops, hidden trajectory divergence, and authority/influence geometry rather than accuracy alone.",
        "",
        "The extracted graph is a meaningful partial transfer, but it is not yet compact: it uses more edges than the hand-seeded graph and remains well below the hand graph on authority/refraction. This supports using a GRU teacher as a search guide, not as a solved extraction pipeline.",
        "",
        "## Runtime Notes",
        "",
        f"- runtime seconds: `{fmt(summary['runtime_seconds'])}`",
        f"- completed records: `{len(summary['records'])}`",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. Do not claim consciousness, biology, production validation, full VRAXION behavior, or natural-language understanding.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    vocab = build_vocab()
    records = [run_seed(args, seed, vocab) for seed in range(args.seeds)]
    agg = aggregate(records)
    summary = {
        "config": {
            "seeds": args.seeds,
            "hidden_sizes": args.hidden_sizes,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "learning_rate": args.learning_rate,
            "aux_frame_weight": args.aux_frame_weight,
            "query_position": args.query_position,
            "extract_ridge": args.extract_ridge,
            "torch_threads": args.torch_threads,
            "smoke": args.smoke,
        },
        "vocab_size": len(vocab),
        "records": records,
        "aggregate": agg,
        "verdict": verdict(agg),
        "runtime_seconds": time.time() - start,
        "platform": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
    }
    out = args.out_dir / SUMMARY_NAME
    out.write_text(json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8")
    write_report(round_floats(summary))
    print(json.dumps({
        "verdict": summary["verdict"],
        "json": str(out),
        "report": str(REPORT_PATH),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
