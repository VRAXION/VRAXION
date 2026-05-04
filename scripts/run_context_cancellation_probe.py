#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        "This probe uses torch because VRAXION's local .venv already provides a CPU build. "
        "Activate it first, for example: source .venv/bin/activate"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "target" / "context-cancellation-probe"

ACTORS = ("dog", "cat", "bird", "child", "robot", "snake")
ACTIONS = ("bite", "sleep", "run", "help", "peck", "spark")
PLACES = ("street", "kitchen", "park", "yard", "lab")
LIGHTS = ("sunlight", "shadow", "neon", "dusk")
NOISES = ("quiet", "traffic", "music", "static")
OBJECTS = ("ball", "stick", "bag", "bench", "wire")

RELATION_POSITIVE_PAIRS = {
    ("dog", "bite"),
    ("cat", "bite"),
    ("snake", "bite"),
    ("bird", "peck"),
    ("robot", "spark"),
    ("child", "run"),
}
RISKY_PLACES = {"street", "lab"}
RISKY_NOISES = {"traffic", "static"}

INPUT_MODES = ("separable", "entangled", "opponent_entangled")


@dataclass(frozen=True)
class ProbeConfig:
    input_mode: str
    seeds: int
    hidden: int
    steps: int
    epochs: int
    train_size: int
    test_size: int
    batch_size: int
    lr: float
    sparse_density: float
    holdout_fraction: float
    active_value: float
    embed_scale: float
    nuisance_scale: float
    opponent_strength: float
    update_rate: float
    delta_scale: float
    ridge: float
    random_label_control: bool
    device: str
    out_dir: str
    telemetry_compromise: str
    cleanup_specificity_definition: str


@dataclass
class Schema:
    hidden: int
    input_feature_names: list[str]
    core_feature_keys: list[str]
    nuisance_feature_keys: list[str]


@dataclass
class DataBundle:
    x: np.ndarray
    y: np.ndarray
    core_y: np.ndarray
    nuisance_y: np.ndarray
    core_component: np.ndarray
    nuisance_component: np.ndarray


@dataclass
class FeatureEmbeddings:
    input_mode: str
    vectors: dict[tuple[str, str], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny recurrent core-recovery v4 probe. The main claim path is label-only; "
            "entangled inputs make core and nuisance share representation channels, and cleanup is "
            "measured post-hoc with probes and interventions."
        )
    )
    parser.add_argument(
        "--input-mode",
        default="all",
        help="One of separable, entangled, opponent_entangled, or all.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64, help="Use --hidden 16 for bottleneck pressure.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--test-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sparse-density", type=float, default=0.12)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--active-value", type=float, default=1.0)
    parser.add_argument("--embed-scale", type=float, default=0.80)
    parser.add_argument("--nuisance-scale", type=float, default=1.05)
    parser.add_argument("--opponent-strength", type=float, default=0.80)
    parser.add_argument(
        "--update-rate",
        type=float,
        default=0.20,
        help="Leaky recurrent update rate. Try 0.1 for stronger recurrent-depth pressure.",
    )
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--random-label-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.input_mode != "all" and args.input_mode not in INPUT_MODES:
        raise SystemExit(f"--input-mode must be one of all,{','.join(INPUT_MODES)}")
    if args.hidden < 8:
        raise SystemExit("--hidden must be at least 8")
    if not 0.0 < args.update_rate <= 1.0:
        raise SystemExit("--update-rate must be in (0, 1]")
    return args


def selected_input_modes(input_mode: str) -> list[str]:
    return list(INPUT_MODES) if input_mode == "all" else [input_mode]


def feature_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for group, values in (
        ("actor", ACTORS),
        ("action", ACTIONS),
        ("place", PLACES),
        ("light", LIGHTS),
        ("noise", NOISES),
        ("object", OBJECTS),
    ):
        keys.extend((group, value) for value in values)
    return keys


def build_schema(hidden: int) -> Schema:
    keys = feature_keys()
    core = [f"{group}:{value}" for group, value in keys if group in {"actor", "action"}]
    nuisance = [f"{group}:{value}" for group, value in keys if group not in {"actor", "action"}]
    return Schema(
        hidden=hidden,
        input_feature_names=[f"{group}:{value}" for group, value in keys],
        core_feature_keys=core,
        nuisance_feature_keys=nuisance,
    )


def unit_vector(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    vec /= max(float(np.linalg.norm(vec)), 1.0e-9)
    return vec * scale


def build_embeddings(
    *,
    schema: Schema,
    input_mode: str,
    seed: int,
    embed_scale: float,
    opponent_strength: float,
) -> FeatureEmbeddings:
    rng = np.random.default_rng(seed)
    hidden = schema.hidden
    core_dim = max(4, hidden // 2)
    nuisance_dim = hidden - core_dim
    core_keys = [(group, value) for group, value in feature_keys() if group in {"actor", "action"}]
    nuisance_keys = [(group, value) for group, value in feature_keys() if group not in {"actor", "action"}]
    vectors: dict[tuple[str, str], np.ndarray] = {}

    if input_mode == "separable":
        for key in core_keys:
            vec = np.zeros(hidden, dtype=np.float32)
            vec[:core_dim] = unit_vector(rng, core_dim, embed_scale)
            vectors[key] = vec
        for key in nuisance_keys:
            vec = np.zeros(hidden, dtype=np.float32)
            vec[core_dim:] = unit_vector(rng, nuisance_dim, embed_scale)
            vectors[key] = vec
    elif input_mode == "entangled":
        for key in core_keys + nuisance_keys:
            vectors[key] = unit_vector(rng, hidden, embed_scale)
    elif input_mode == "opponent_entangled":
        core_bank = [unit_vector(rng, hidden, embed_scale) for _ in core_keys]
        for key, vec in zip(core_keys, core_bank):
            vectors[key] = vec
        for idx, key in enumerate(nuisance_keys):
            opponent = -opponent_strength * core_bank[idx % len(core_bank)]
            noise = unit_vector(rng, hidden, embed_scale * (1.0 - min(opponent_strength, 0.95) * 0.45))
            vec = opponent + noise
            vec /= max(float(np.linalg.norm(vec)), 1.0e-9)
            vectors[key] = vec.astype(np.float32) * embed_scale
    else:
        raise ValueError(f"unknown input mode: {input_mode}")

    return FeatureEmbeddings(input_mode=input_mode, vectors=vectors)


def relation_label(actor: str, action: str) -> int:
    return int((actor, action) in RELATION_POSITIVE_PAIRS)


def nuisance_causal_label(place: str, noise: str) -> int:
    return int((place in RISKY_PLACES) ^ (noise in RISKY_NOISES))


def split_nuisance_combos(seed: int, holdout_fraction: float) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    combos = list(product(PLACES, LIGHTS, NOISES, OBJECTS))
    positives = [combo for combo in combos if nuisance_causal_label(combo[0], combo[2]) == 1]
    negatives = [combo for combo in combos if nuisance_causal_label(combo[0], combo[2]) == 0]
    rng = np.random.default_rng(seed + 10_007)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_group(group: list[tuple[str, str, str, str]]) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
        holdout_count = max(1, int(round(len(group) * holdout_fraction)))
        return group[holdout_count:], group[:holdout_count]

    train_pos, heldout_pos = split_group(positives)
    train_neg, heldout_neg = split_group(negatives)
    train = train_pos + train_neg
    heldout = heldout_pos + heldout_neg
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def choose_relation_pair(rng: np.random.Generator, positive: bool) -> tuple[str, str]:
    positive_pairs = sorted(RELATION_POSITIVE_PAIRS)
    negative_pairs = sorted(
        (actor, action)
        for actor in ACTORS
        for action in ACTIONS
        if (actor, action) not in RELATION_POSITIVE_PAIRS
    )
    pairs = positive_pairs if positive else negative_pairs
    return pairs[rng.integers(len(pairs))]


def choose_combo_for_label(
    rng: np.random.Generator,
    combos: list[tuple[str, str, str, str]],
    positive: bool,
) -> tuple[str, str, str, str]:
    candidates = [
        combo
        for combo in combos
        if nuisance_causal_label(combo[0], combo[2]) == int(positive)
    ]
    if not candidates:
        candidates = combos
    return candidates[rng.integers(len(candidates))]


def make_dataset(
    *,
    n: int,
    combos: list[tuple[str, str, str, str]],
    seed: int,
    embeddings: FeatureEmbeddings,
    active_value: float,
    nuisance_scale: float,
    task: str,
    random_labels: bool = False,
) -> DataBundle:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(next(iter(embeddings.vectors.values())))), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    core_y = np.zeros(n, dtype=np.int64)
    nuisance_y = np.zeros(n, dtype=np.int64)
    core_component = np.zeros_like(x)
    nuisance_component = np.zeros_like(x)

    for row in range(n):
        target_positive = bool(rng.integers(0, 2))
        if task == "nuisance_causal":
            actor = ACTORS[rng.integers(len(ACTORS))]
            action = ACTIONS[rng.integers(len(ACTIONS))]
            place, light, noise, obj = choose_combo_for_label(rng, combos, target_positive)
        elif task in {"relation", "relation_random_labels"}:
            actor, action = choose_relation_pair(rng, target_positive)
            place, light, noise, obj = combos[rng.integers(len(combos))]
        else:
            raise ValueError(f"unknown task: {task}")

        core = active_value * (embeddings.vectors[("actor", actor)] + embeddings.vectors[("action", action)])
        nuisance = active_value * nuisance_scale * (
            embeddings.vectors[("place", place)]
            + embeddings.vectors[("light", light)]
            + embeddings.vectors[("noise", noise)]
            + embeddings.vectors[("object", obj)]
        )
        core_component[row] = core
        nuisance_component[row] = nuisance
        x[row] = core + nuisance
        core_y[row] = relation_label(actor, action)
        nuisance_y[row] = nuisance_causal_label(place, noise)
        if random_labels:
            y[row] = rng.integers(0, 2)
        elif task == "nuisance_causal":
            y[row] = nuisance_y[row]
        else:
            y[row] = core_y[row]

    order = rng.permutation(n)
    return DataBundle(
        x=x[order],
        y=y[order],
        core_y=core_y[order],
        nuisance_y=nuisance_y[order],
        core_component=core_component[order],
        nuisance_component=nuisance_component[order],
    )


def make_sparse_mask(hidden: int, density: float, seed: int) -> np.ndarray:
    if density >= 1.0:
        return np.ones((hidden, hidden), dtype=np.float32)
    rng = np.random.default_rng(seed)
    mask = (rng.random((hidden, hidden)) < density).astype(np.float32)
    np.fill_diagonal(mask, 1.0)
    return mask


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        steps: int,
        mask: np.ndarray,
        update_rate: float,
        delta_scale: float,
    ):
        super().__init__()
        self.steps = steps
        self.update_rate = update_rate
        self.delta_scale = delta_scale
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.recurrent = nn.Parameter(torch.empty(hidden, hidden))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout(self, x: torch.Tensor, ablation: str | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h = torch.tanh(x)
        states = [h]
        masked_recurrent = self.recurrent * self.mask
        freeze_after = None
        if ablation and ablation.startswith("freeze_after_"):
            freeze_after = int(ablation.rsplit("_", maxsplit=1)[1])
            ablation = None
        for step in range(1, self.steps + 1):
            if freeze_after is not None and step > freeze_after:
                states.append(h)
                continue
            if ablation == "zero_recurrent_update":
                proposal = h
            else:
                if ablation == "zero_matrix_keep_threshold":
                    delta = self.threshold.unsqueeze(0).expand_as(h)
                elif ablation is None:
                    delta = h @ masked_recurrent.t() + self.threshold
                else:
                    raise ValueError(f"unknown ablation: {ablation}")
                proposal = torch.tanh(h + self.delta_scale * delta)
            h = (1.0 - self.update_rate) * h + self.update_rate * proposal
            states.append(h)

        logits = [self.head(state) for state in states]
        return states, logits


def to_tensor(x: np.ndarray, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def train_model(
    *,
    train: DataBundle,
    hidden: int,
    args: argparse.Namespace,
    seed: int,
) -> RecurrentClassifier:
    torch.manual_seed(seed)
    mask = make_sparse_mask(hidden, args.sparse_density, seed + 200_003)
    model = RecurrentClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x = to_tensor(train.x, args.device)
    y = to_tensor(train.y, args.device, torch.long)
    n = len(train.x)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_001)

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            xb = x.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            _states, logits = model.rollout(xb)
            loss = F.cross_entropy(logits[-1], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def component_energy(states: list[np.ndarray], component: np.ndarray) -> list[float]:
    denom = np.linalg.norm(component, axis=1, keepdims=True)
    unit = component / np.maximum(denom, 1.0e-9)
    return [float(np.mean(np.sum(state * unit, axis=1) ** 2)) for state in states]


def template_distance(states: list[np.ndarray], component: np.ndarray) -> list[float]:
    template = np.tanh(component)
    scale = float(np.sqrt(component.shape[1]))
    return [
        float(np.mean(np.linalg.norm(state - template, axis=1) / max(scale, 1.0e-9)))
        for state in states
    ]


def ridge_probe_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    ridge: float,
) -> float:
    train_aug = np.concatenate([train_x, np.ones((len(train_x), 1), dtype=np.float32)], axis=1)
    test_aug = np.concatenate([test_x, np.ones((len(test_x), 1), dtype=np.float32)], axis=1)
    target = np.zeros((len(train_y), 2), dtype=np.float32)
    target[np.arange(len(train_y)), train_y.astype(int)] = 1.0
    xtx = train_aug.T @ train_aug
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xtx + reg, train_aug.T @ target)
    logits = test_aug @ weights
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == test_y))


def rollout_arrays(
    model: RecurrentClassifier,
    x: np.ndarray,
    device: str,
    ablation: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    with torch.no_grad():
        tx = to_tensor(x, device)
        states_t, logits_t = model.rollout(tx, ablation=ablation)
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
    return states, logits


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def output_entropy_by_step(logits_by_step: list[np.ndarray]) -> list[float]:
    entropies: list[float] = []
    for logits in logits_by_step:
        probs = softmax_np(logits)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1.0e-9)), axis=1)
        entropies.append(float(np.mean(entropy)))
    return entropies


def logit_margin_by_step(logits_by_step: list[np.ndarray]) -> list[float]:
    return [
        float(np.mean(np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]))
        for logits in logits_by_step
    ]


def evaluate_model(
    *,
    model: RecurrentClassifier,
    train: DataBundle,
    test: DataBundle,
    args: argparse.Namespace,
    ablation: str | None = None,
) -> dict[str, Any]:
    train_states, _train_logits = rollout_arrays(model, train.x, args.device, ablation=ablation)
    test_states, test_logits = rollout_arrays(model, test.x, args.device, ablation=ablation)
    pred = np.argmax(test_logits[-1], axis=1)
    per_step_accuracy = [float(np.mean(np.argmax(logits, axis=1) == test.y)) for logits in test_logits]
    label_logit_confidence = [
        float(np.mean(softmax_np(logits)[np.arange(len(test.y)), test.y]))
        for logits in test_logits
    ]
    label_logit_margin = logit_margin_by_step(test_logits)
    entropy = output_entropy_by_step(test_logits)
    core_energy = component_energy(test_states, test.core_component)
    nuisance_energy = component_energy(test_states, test.nuisance_component)
    core_to_nuisance_ratio = [
        core / max(nuisance, 1.0e-9)
        for core, nuisance in zip(core_energy, nuisance_energy)
    ]
    core_preservation_by_step = [energy / max(core_energy[0], 1.0e-9) for energy in core_energy]
    nuisance_drop = nuisance_energy[0] - nuisance_energy[-1]
    core_distance = template_distance(test_states, test.core_component)
    nuisance_distance = template_distance(test_states, test.nuisance_component)
    core_probe = [
        ridge_probe_accuracy(train_states[step], train.core_y, test_states[step], test.core_y, args.ridge)
        for step in range(len(test_states))
    ]
    nuisance_probe = [
        ridge_probe_accuracy(train_states[step], train.nuisance_y, test_states[step], test.nuisance_y, args.ridge)
        for step in range(len(test_states))
    ]

    return {
        "accuracy": float(np.mean(pred == test.y)),
        "heldout_nuisance_accuracy": float(np.mean(pred == test.y)),
        "per_step_accuracy": per_step_accuracy,
        "core_probe_accuracy_by_step": core_probe,
        "nuisance_probe_accuracy_by_step": nuisance_probe,
        "core_probe_delta": core_probe[-1] - core_probe[0],
        "nuisance_probe_delta": nuisance_probe[-1] - nuisance_probe[0],
        "nuisance_probe_drop": nuisance_probe[0] - nuisance_probe[-1],
        "nuisance_energy_by_step": nuisance_energy,
        "core_energy_by_step": core_energy,
        "core_to_nuisance_ratio_by_step": core_to_nuisance_ratio,
        "hidden_state_distance_to_core_template_by_step": core_distance,
        "hidden_state_distance_to_nuisance_template_by_step": nuisance_distance,
        "core_preservation_by_step": core_preservation_by_step,
        "core_preservation": core_preservation_by_step[-1],
        "nuisance_energy_drop": float(nuisance_drop),
        "nuisance_energy_drop_ratio": float(nuisance_drop / max(nuisance_energy[0], 1.0e-9)),
        "output_entropy_by_step": entropy,
        "label_logit_confidence_by_step": label_logit_confidence,
        "label_logit_margin_by_step": label_logit_margin,
    }


def run_experiment_seed(
    *,
    name: str,
    input_mode: str,
    task: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
    random_labels: bool = False,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
    )
    train = make_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 1_001,
        embeddings=embeddings,
        active_value=args.active_value,
        nuisance_scale=args.nuisance_scale,
        task=task,
        random_labels=random_labels,
    )
    test = make_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 1_002,
        embeddings=embeddings,
        active_value=args.active_value,
        nuisance_scale=args.nuisance_scale,
        task=task,
        random_labels=random_labels,
    )
    model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 31)
    main = evaluate_model(model=model, train=train, test=test, args=args)
    zero = evaluate_model(model=model, train=train, test=test, args=args, ablation="zero_recurrent_update")
    threshold_only = evaluate_model(model=model, train=train, test=test, args=args, ablation="zero_matrix_keep_threshold")
    interventions = {}
    if name == "label_only_entangled":
        interventions = run_interventions(model=model, train=train, test=test, args=args, seed=seed + 900_001)
    return {
        "name": name,
        "input_mode": input_mode,
        "task": task,
        "seed": seed,
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "main": main,
        "controls": {
            "zero_recurrent_update": zero,
            "threshold_only": threshold_only,
        },
        "interventions": interventions,
    }


def prediction_summary(
    *,
    model: RecurrentClassifier,
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    ablation: str | None = None,
) -> dict[str, Any]:
    _states, logits = rollout_arrays(model, x, args.device, ablation=ablation)
    pred = np.argmax(logits[-1], axis=1)
    probs_by_step = [softmax_np(step_logits) for step_logits in logits]
    return {
        "accuracy": float(np.mean(pred == y)),
        "per_step_accuracy": [
            float(np.mean(np.argmax(step_logits, axis=1) == y))
            for step_logits in logits
        ],
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(probs[np.arange(len(y)), y]))
            for probs in probs_by_step
        ],
    }


def counterfactual_influence_summary(
    *,
    model: RecurrentClassifier,
    reference_x: np.ndarray,
    counterfactual_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _reference_states, reference_logits = rollout_arrays(model, reference_x, args.device)
    _counterfactual_states, counterfactual_logits = rollout_arrays(model, counterfactual_x, args.device)

    output_change_rate_by_step: list[float] = []
    target_accuracy_by_step: list[float] = []
    original_label_retention_by_step: list[float] = []
    mean_abs_label_probability_delta_by_step: list[float] = []
    mean_kl_divergence_by_step: list[float] = []

    for ref_logits, cf_logits in zip(reference_logits, counterfactual_logits):
        ref_probs = softmax_np(ref_logits)
        cf_probs = softmax_np(cf_logits)
        ref_pred = np.argmax(ref_probs, axis=1)
        cf_pred = np.argmax(cf_probs, axis=1)
        output_change_rate_by_step.append(float(np.mean(ref_pred != cf_pred)))
        target_accuracy_by_step.append(float(np.mean(cf_pred == counterfactual_y)))
        original_label_retention_by_step.append(float(np.mean(cf_pred == reference_y)))
        ref_label_prob = ref_probs[np.arange(len(reference_y)), reference_y]
        cf_label_prob = cf_probs[np.arange(len(reference_y)), reference_y]
        mean_abs_label_probability_delta_by_step.append(float(np.mean(np.abs(ref_label_prob - cf_label_prob))))
        kl = np.sum(ref_probs * (np.log(np.maximum(ref_probs, 1.0e-9)) - np.log(np.maximum(cf_probs, 1.0e-9))), axis=1)
        mean_kl_divergence_by_step.append(float(np.mean(kl)))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
    }


def recurrent_matrix_control(model: RecurrentClassifier, control: str, seed: int) -> RecurrentClassifier:
    clone = copy.deepcopy(model)
    with torch.no_grad():
        if control == "reverse":
            clone.recurrent.copy_(model.recurrent.t())
        elif control == "randomize":
            generator = torch.Generator(device=model.recurrent.device)
            generator.manual_seed(seed)
            std = float(model.recurrent.detach().std().item())
            randomized = torch.randn(
                model.recurrent.shape,
                generator=generator,
                device=model.recurrent.device,
                dtype=model.recurrent.dtype,
            ) * max(std, 1.0e-6)
            clone.recurrent.copy_(randomized)
        else:
            raise ValueError(f"unknown recurrent control: {control}")
    return clone


def run_interventions(
    *,
    model: RecurrentClassifier,
    train: DataBundle,
    test: DataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nuisance_perm = rng.permutation(len(test.x))
    core_perm = rng.permutation(len(test.x))
    interventions: dict[str, Any] = {
        "freeze_after_1": prediction_summary(model=model, x=test.x, y=test.y, args=args, ablation="freeze_after_1"),
        "freeze_after_2": prediction_summary(model=model, x=test.x, y=test.y, args=args, ablation="freeze_after_2"),
        "freeze_after_3": prediction_summary(model=model, x=test.x, y=test.y, args=args, ablation="freeze_after_3"),
        "shuffle_nuisance_keep_core": prediction_summary(
            model=model,
            x=test.core_component + test.nuisance_component[nuisance_perm],
            y=test.y,
            args=args,
        ),
        "shuffle_core_keep_nuisance": prediction_summary(
            model=model,
            x=test.core_component[core_perm] + test.nuisance_component,
            y=test.y[core_perm],
            args=args,
        ),
        "nuisance_influence_same_core_different_nuisance": counterfactual_influence_summary(
            model=model,
            reference_x=test.x,
            counterfactual_x=test.core_component + test.nuisance_component[nuisance_perm],
            reference_y=test.y,
            counterfactual_y=test.y,
            args=args,
        ),
        "core_influence_same_nuisance_different_core": counterfactual_influence_summary(
            model=model,
            reference_x=test.x,
            counterfactual_x=test.core_component[core_perm] + test.nuisance_component,
            reference_y=test.y,
            counterfactual_y=test.y[core_perm],
            args=args,
        ),
    }

    nuisance_scale: dict[str, Any] = {}
    for scale in (0.0, 0.5, 1.0, 2.0):
        x_scaled = test.core_component + scale * test.nuisance_component
        recurrent = prediction_summary(model=model, x=x_scaled, y=test.y, args=args)
        zero = prediction_summary(model=model, x=x_scaled, y=test.y, args=args, ablation="zero_recurrent_update")
        nuisance_scale[str(scale)] = {
            "recurrent": recurrent,
            "zero_recurrent": zero,
            "recurrence_gain": recurrent["accuracy"] - zero["accuracy"],
        }
    interventions["nuisance_amplitude_scale"] = nuisance_scale

    core_scale: dict[str, Any] = {}
    for scale in (0.5, 1.0, 2.0):
        x_scaled = scale * test.core_component + test.nuisance_component
        recurrent = prediction_summary(model=model, x=x_scaled, y=test.y, args=args)
        zero = prediction_summary(model=model, x=x_scaled, y=test.y, args=args, ablation="zero_recurrent_update")
        core_scale[str(scale)] = {
            "recurrent": recurrent,
            "zero_recurrent": zero,
            "recurrence_gain": recurrent["accuracy"] - zero["accuracy"],
        }
    interventions["core_amplitude_scale"] = core_scale

    reverse_model = recurrent_matrix_control(model, "reverse", seed + 1)
    random_model = recurrent_matrix_control(model, "randomize", seed + 2)
    interventions["reverse_recurrent_matrix"] = prediction_summary(
        model=reverse_model,
        x=test.x,
        y=test.y,
        args=args,
    )
    interventions["randomize_recurrent_matrix"] = prediction_summary(
        model=random_model,
        x=test.x,
        y=test.y,
        args=args,
    )
    return interventions


def mean_list(rows: list[list[float]]) -> list[float]:
    max_len = max(len(row) for row in rows)
    return [
        float(np.mean([row[col] for row in rows if col < len(row)]))
        for col in range(max_len)
    ]


def aggregate_metric_dict(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "accuracy",
        "heldout_nuisance_accuracy",
        "core_probe_delta",
        "nuisance_probe_delta",
        "nuisance_probe_drop",
        "core_preservation",
        "nuisance_energy_drop",
        "nuisance_energy_drop_ratio",
    ]
    list_keys = [
        "per_step_accuracy",
        "core_probe_accuracy_by_step",
        "nuisance_probe_accuracy_by_step",
        "nuisance_energy_by_step",
        "core_energy_by_step",
        "core_to_nuisance_ratio_by_step",
        "hidden_state_distance_to_core_template_by_step",
        "hidden_state_distance_to_nuisance_template_by_step",
        "core_preservation_by_step",
        "output_entropy_by_step",
        "label_logit_confidence_by_step",
        "label_logit_margin_by_step",
    ]
    out: dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(metric[key]) for metric in metrics]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    for key in list_keys:
        out[key] = mean_list([metric[key] for metric in metrics])
    return out


def aggregate_prediction_summaries(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = ["accuracy"]
    list_keys = [
        "per_step_accuracy",
        "output_entropy_by_step",
        "logit_margin_by_step",
        "label_logit_confidence_by_step",
    ]
    out: dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(metric[key]) for metric in metrics]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    for key in list_keys:
        out[key] = mean_list([metric[key] for metric in metrics])
    return out


def aggregate_counterfactual_summaries(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "label_change_rate",
        "output_change_rate",
        "target_accuracy",
        "original_label_retention",
        "mean_abs_label_probability_delta",
        "mean_kl_divergence",
    ]
    list_keys = [
        "output_change_rate_by_step",
        "target_accuracy_by_step",
        "original_label_retention_by_step",
        "mean_abs_label_probability_delta_by_step",
        "mean_kl_divergence_by_step",
    ]
    out: dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(metric[key]) for metric in metrics]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    for key in list_keys:
        out[key] = mean_list([metric[key] for metric in metrics])
    return out


def aggregate_interventions(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs or not runs[0].get("interventions"):
        return {}

    direct_keys = [
        "freeze_after_1",
        "freeze_after_2",
        "freeze_after_3",
        "shuffle_nuisance_keep_core",
        "shuffle_core_keep_nuisance",
        "reverse_recurrent_matrix",
        "randomize_recurrent_matrix",
    ]
    out: dict[str, Any] = {}
    for key in direct_keys:
        out[key] = aggregate_prediction_summaries([run["interventions"][key] for run in runs])

    counterfactual_keys = [
        "nuisance_influence_same_core_different_nuisance",
        "core_influence_same_nuisance_different_core",
    ]
    for key in counterfactual_keys:
        out[key] = aggregate_counterfactual_summaries([run["interventions"][key] for run in runs])

    for group_key in ("nuisance_amplitude_scale", "core_amplitude_scale"):
        out[group_key] = {}
        scales = sorted(runs[0]["interventions"][group_key], key=float)
        for scale in scales:
            recurrent = aggregate_prediction_summaries(
                [run["interventions"][group_key][scale]["recurrent"] for run in runs]
            )
            zero = aggregate_prediction_summaries(
                [run["interventions"][group_key][scale]["zero_recurrent"] for run in runs]
            )
            gains = [
                run["interventions"][group_key][scale]["recurrence_gain"]
                for run in runs
            ]
            out[group_key][scale] = {
                "recurrent": recurrent,
                "zero_recurrent": zero,
                "recurrence_gain": float(np.mean(gains)),
                "recurrence_gain_std": float(np.std(gains)),
            }
    return out


def aggregate_experiment_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    main = aggregate_metric_dict([run["main"] for run in runs])
    controls = {
        "zero_recurrent_update": aggregate_metric_dict(
            [run["controls"]["zero_recurrent_update"] for run in runs]
        ),
        "threshold_only": aggregate_metric_dict(
            [run["controls"]["threshold_only"] for run in runs]
        ),
    }
    recurrence_gain = main["accuracy"] - controls["zero_recurrent_update"]["accuracy"]
    interventions = aggregate_interventions(runs)
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task": runs[0]["task"],
        "test_accuracy": main["accuracy"],
        "heldout_nuisance_accuracy": main["heldout_nuisance_accuracy"],
        "core_probe_accuracy_by_step": main["core_probe_accuracy_by_step"],
        "nuisance_probe_accuracy_by_step": main["nuisance_probe_accuracy_by_step"],
        "core_probe_delta": main["core_probe_delta"],
        "nuisance_probe_delta": main["nuisance_probe_delta"],
        "nuisance_probe_drop": main["nuisance_probe_drop"],
        "nuisance_energy_by_step": main["nuisance_energy_by_step"],
        "core_energy_by_step": main["core_energy_by_step"],
        "core_to_nuisance_ratio_by_step": main["core_to_nuisance_ratio_by_step"],
        "hidden_state_distance_to_core_template_by_step": main["hidden_state_distance_to_core_template_by_step"],
        "hidden_state_distance_to_nuisance_template_by_step": main["hidden_state_distance_to_nuisance_template_by_step"],
        "core_preservation_by_step": main["core_preservation_by_step"],
        "core_preservation": main["core_preservation"],
        "output_entropy_by_step": main["output_entropy_by_step"],
        "logit_confidence_by_step": main["label_logit_confidence_by_step"],
        "label_logit_margin_by_step": main["label_logit_margin_by_step"],
        "recurrence_gain": recurrence_gain,
        "main": main,
        "controls": controls,
        "interventions": interventions,
        "runs": runs,
    }


def flatten_requested_report_keys(experiments: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, summary in experiments.items():
        out[name] = summary
        if name == "label_only_entangled":
            zero = dict(summary["controls"]["zero_recurrent_update"])
            zero.update(
                {
                    "name": "label_only_entangled_zero_recurrent",
                    "input_mode": "entangled",
                    "task": summary["task"],
                    "test_accuracy": zero["accuracy"],
                    "recurrence_gain": 0.0,
                }
            )
            threshold = dict(summary["controls"]["threshold_only"])
            threshold.update(
                {
                    "name": "label_only_entangled_threshold_only",
                    "input_mode": "entangled",
                    "task": summary["task"],
                    "test_accuracy": threshold["accuracy"],
                    "recurrence_gain": 0.0,
                }
            )
            out["label_only_entangled_zero_recurrent"] = zero
            out["label_only_entangled_threshold_only"] = threshold
    return out


def cleanup_specificity_score(experiments: dict[str, Any]) -> float | None:
    irrelevant = experiments.get("label_only_entangled")
    causal = experiments.get("nuisance_causal_entangled_control")
    if not irrelevant or not causal:
        return None
    return irrelevant["nuisance_probe_drop"] - causal["nuisance_probe_drop"]


def interpret_report(experiments: dict[str, Any], specificity: float | None, random_summary: dict[str, Any] | None) -> dict[str, str]:
    ent = experiments.get("label_only_entangled")
    causal = experiments.get("nuisance_causal_entangled_control")
    if not ent or not causal:
        return {
            "supports_context_cancellation": "unclear",
            "supports_recurrent_core_recovery": "unclear",
            "reason": "label_only_entangled and nuisance_causal_entangled_control are required for the v4 interpretation.",
        }

    zero = ent["controls"]["zero_recurrent_update"]
    recurrence_gain = ent["recurrence_gain"]
    nuisance_drop = ent["nuisance_probe_drop"]
    core_delta = ent["core_probe_delta"]
    core_final = ent["core_probe_accuracy_by_step"][-1]
    core_initial = ent["core_probe_accuracy_by_step"][0]
    ratio_gain = ent["core_to_nuisance_ratio_by_step"][-1] - ent["core_to_nuisance_ratio_by_step"][0]
    entropy_drop = ent["output_entropy_by_step"][0] - ent["output_entropy_by_step"][-1]
    margin_gain = ent["label_logit_margin_by_step"][-1] - ent["label_logit_margin_by_step"][0]
    causal_nuisance_drop = causal["nuisance_probe_drop"]
    causal_nuisance_final = causal["nuisance_probe_accuracy_by_step"][-1]
    interventions = ent.get("interventions", {})
    freeze_early_hurts = False
    randomize_destroys = False
    gain_increases_with_nuisance = False
    if interventions:
        freeze_early_hurts = interventions["freeze_after_1"]["accuracy"] <= ent["heldout_nuisance_accuracy"] - 0.05
        randomize_destroys = interventions["randomize_recurrent_matrix"]["accuracy"] <= ent["heldout_nuisance_accuracy"] - 0.20
        nuisance_scales = interventions["nuisance_amplitude_scale"]
        gain_increases_with_nuisance = (
            max(
                nuisance_scales["0.5"]["recurrence_gain"],
                nuisance_scales["1.0"]["recurrence_gain"],
                nuisance_scales["2.0"]["recurrence_gain"],
            )
            >= nuisance_scales["0.0"]["recurrence_gain"] + 0.05
        )
    random_ok = True
    if random_summary is not None:
        random_ok = random_summary["test_accuracy"] < 0.65

    core_recovery_supported = (
        ent["heldout_nuisance_accuracy"] >= 0.85
        and recurrence_gain >= 0.05
        and core_delta >= 0.10
        and core_final >= max(0.80, core_initial - 0.05)
        and ratio_gain > 0.0
        and entropy_drop > 0.05
        and margin_gain > 0.10
        and freeze_early_hurts
        and randomize_destroys
        and gain_increases_with_nuisance
        and random_ok
    )
    context_cancellation_supported = (
        core_recovery_supported
        and nuisance_drop >= 0.10
        and causal["heldout_nuisance_accuracy"] >= 0.80
        and causal_nuisance_drop <= max(0.05, nuisance_drop * 0.50)
        and causal_nuisance_final >= 0.80
    )

    if context_cancellation_supported:
        return {
            "supports_context_cancellation": "true",
            "supports_recurrent_core_recovery": "true",
            "reason": (
                "The entangled label-only recurrent model beats its zero-recurrent ablation, nuisance probe "
                "decodability drops while core probe decodability is preserved, the nuisance-causal control "
                "keeps nuisance decodable, and the random-label control fails."
            ),
        }

    if core_recovery_supported:
        return {
            "supports_context_cancellation": "false",
            "supports_recurrent_core_recovery": "true",
            "reason": (
                "The entangled label-only recurrent model shows recurrent core recovery: recurrence beats the "
                "zero-recurrent ablation, core probe decodability rises, core-to-nuisance ratio increases, output "
                "entropy falls, logit margin rises, early freezing hurts, randomizing recurrence destroys the gain, "
                "and the random-label control fails. Nuisance probe decodability does not fall enough to call this "
                "clean context cancellation."
            ),
        }

    if ent["heldout_nuisance_accuracy"] >= 0.85 and recurrence_gain < 0.03:
        return {
            "supports_context_cancellation": "false",
            "supports_recurrent_core_recovery": "false",
            "reason": (
                "The entangled label-only task learns, but recurrence does not materially beat the zero-recurrent "
                f"ablation (gain={recurrence_gain:.3f}, zero_acc={zero['accuracy']:.3f}). Cleanup is not necessary here."
            ),
        }

    if ent["heldout_nuisance_accuracy"] >= 0.85 and nuisance_drop < 0.05 and not core_recovery_supported:
        return {
            "supports_context_cancellation": "false",
            "supports_recurrent_core_recovery": "unclear",
            "reason": (
                "The entangled label-only task learns, but nuisance probe decodability does not drop meaningfully. "
                "That is not recurrent context cancellation."
            ),
        }

    return {
        "supports_context_cancellation": "unclear",
        "supports_recurrent_core_recovery": "unclear",
        "reason": (
            f"entangled_acc={ent['heldout_nuisance_accuracy']:.3f}, recurrence_gain={recurrence_gain:.3f}, "
            f"core_probe_delta={core_delta:.3f}, nuisance_probe_drop={nuisance_drop:.3f}, "
            f"causal_nuisance_drop={causal_nuisance_drop:.3f}, "
            f"specificity={specificity if specificity is not None else 'n/a'}. The v4 criteria are not cleanly met."
        ),
    }


def round_floats(obj: Any, digits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [round_floats(item, digits) for item in obj]
    if isinstance(obj, dict):
        return {key: round_floats(value, digits) for key, value in obj.items()}
    return obj


def compact_intervention_summary(interventions: dict[str, Any]) -> dict[str, Any]:
    if not interventions:
        return {}
    return {
        "freeze_accuracy": {
            key: interventions[key]["accuracy"]
            for key in ("freeze_after_1", "freeze_after_2", "freeze_after_3")
        },
        "shuffle_accuracy": {
            "shuffle_nuisance_keep_core": interventions["shuffle_nuisance_keep_core"]["accuracy"],
            "shuffle_core_keep_nuisance": interventions["shuffle_core_keep_nuisance"]["accuracy"],
        },
        "recurrent_matrix_control_accuracy": {
            "reverse": interventions["reverse_recurrent_matrix"]["accuracy"],
            "randomize": interventions["randomize_recurrent_matrix"]["accuracy"],
        },
        "counterfactual_influence": {
            "same_core_different_nuisance": {
                "label_change_rate": interventions["nuisance_influence_same_core_different_nuisance"]["label_change_rate"],
                "output_change_rate": interventions["nuisance_influence_same_core_different_nuisance"]["output_change_rate"],
                "target_accuracy": interventions["nuisance_influence_same_core_different_nuisance"]["target_accuracy"],
                "mean_abs_label_probability_delta": interventions["nuisance_influence_same_core_different_nuisance"][
                    "mean_abs_label_probability_delta"
                ],
                "mean_kl_divergence": interventions["nuisance_influence_same_core_different_nuisance"]["mean_kl_divergence"],
            },
            "same_nuisance_different_core": {
                "label_change_rate": interventions["core_influence_same_nuisance_different_core"]["label_change_rate"],
                "output_change_rate": interventions["core_influence_same_nuisance_different_core"]["output_change_rate"],
                "target_accuracy": interventions["core_influence_same_nuisance_different_core"]["target_accuracy"],
                "mean_abs_label_probability_delta": interventions["core_influence_same_nuisance_different_core"][
                    "mean_abs_label_probability_delta"
                ],
                "mean_kl_divergence": interventions["core_influence_same_nuisance_different_core"]["mean_kl_divergence"],
            },
        },
        "nuisance_scale_recurrence_gain": {
            scale: data["recurrence_gain"]
            for scale, data in interventions["nuisance_amplitude_scale"].items()
        },
        "core_scale_recurrence_gain": {
            scale: data["recurrence_gain"]
            for scale, data in interventions["core_amplitude_scale"].items()
        },
    }


def main() -> int:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested --device {args.device!r}, but CUDA is not available")

    out_root = args.out_dir or DEFAULT_OUT_ROOT
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    schema = build_schema(args.hidden)
    config = ProbeConfig(
        input_mode=args.input_mode,
        seeds=args.seeds,
        hidden=args.hidden,
        steps=args.steps,
        epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        lr=args.lr,
        sparse_density=args.sparse_density,
        holdout_fraction=args.holdout_fraction,
        active_value=args.active_value,
        embed_scale=args.embed_scale,
        nuisance_scale=args.nuisance_scale,
        opponent_strength=args.opponent_strength,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        ridge=args.ridge,
        random_label_control=args.random_label_control,
        device=args.device,
        out_dir=str(run_dir),
        telemetry_compromise=(
            "Core and nuisance component energies are measured by alignment to the known generated component "
            "vectors for each sample. Core/nuisance probe accuracies are post-hoc ridge probes trained on frozen "
            "states per recurrent step. The main v4 claim path uses label-only training; no hidden cleanup target "
            "is used for label_only or nuisance_causal controls."
        ),
        cleanup_specificity_definition=(
            "label_only_entangled nuisance_probe_drop minus nuisance_causal_entangled_control nuisance_probe_drop"
        ),
    )

    experiment_specs: list[tuple[str, str, str, bool]] = []
    for input_mode in selected_input_modes(args.input_mode):
        experiment_specs.append((f"label_only_{input_mode}", input_mode, "relation", False))
    if args.input_mode in {"all", "entangled"}:
        experiment_specs.append(("nuisance_causal_entangled_control", "entangled", "nuisance_causal", False))
    elif args.input_mode == "opponent_entangled":
        experiment_specs.append(("nuisance_causal_opponent_entangled_control", "opponent_entangled", "nuisance_causal", False))
    else:
        experiment_specs.append(("nuisance_causal_separable_control", "separable", "nuisance_causal", False))

    experiments: dict[str, Any] = {}
    for name, input_mode, task, random_labels in experiment_specs:
        runs = [
            run_experiment_seed(
                name=name,
                input_mode=input_mode,
                task=task,
                seed=seed,
                schema=schema,
                args=args,
                random_labels=random_labels,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_experiment_runs(name, runs)

    random_summary = None
    if args.random_label_control and args.input_mode in {"all", "entangled"}:
        random_runs = [
            run_experiment_seed(
                name="random_label_entangled_control",
                input_mode="entangled",
                task="relation_random_labels",
                seed=seed + 7_001,
                schema=schema,
                args=args,
                random_labels=True,
            )
            for seed in range(args.seeds)
        ]
        random_summary = aggregate_experiment_runs("random_label_entangled_control", random_runs)

    specificity = cleanup_specificity_score(experiments)
    interpretation = interpret_report(experiments, specificity, random_summary)
    requested_report = flatten_requested_report_keys(experiments)

    report = {
        "config": asdict(config),
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "recurrent_core_recovery_v4",
        "experiments": requested_report,
        "random_label_control": random_summary,
        "recurrence_gain": experiments.get("label_only_entangled", {}).get("recurrence_gain"),
        "cleanup_specificity_score": specificity,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "It does not prove the current VRAXION mainline already implements this behavior.",
            "The useful v4 signal is recurrence-dependent core recovery from entangled nuisance interference.",
            "A positive result requires post-hoc nuisance probe decodability to fall while core probe decodability is preserved.",
            "A separate recurrent-core-recovery interpretation can be positive even when clean context cancellation remains false.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "context_cancellation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    printable = {
        "mode": report["mode"],
        "recurrence_gain": report["recurrence_gain"],
        "cleanup_specificity_score": report["cleanup_specificity_score"],
        "interpretation": report["interpretation"],
        "experiments": {
            name: {
                key: summary[key]
                for key in (
                    "test_accuracy",
                    "heldout_nuisance_accuracy",
                    "core_probe_accuracy_by_step",
                    "nuisance_probe_accuracy_by_step",
                    "core_probe_delta",
                    "nuisance_probe_delta",
                    "nuisance_probe_drop",
                    "nuisance_energy_by_step",
                    "core_to_nuisance_ratio_by_step",
                    "hidden_state_distance_to_core_template_by_step",
                    "hidden_state_distance_to_nuisance_template_by_step",
                    "core_preservation_by_step",
                    "output_entropy_by_step",
                    "logit_confidence_by_step",
                    "label_logit_margin_by_step",
                    "recurrence_gain",
                )
                if key in summary
            }
            | ({"intervention_summary": compact_intervention_summary(summary.get("interventions", {}))}
               if name == "label_only_entangled" else {})
            for name, summary in report["experiments"].items()
            if name in {
                "label_only_separable",
                "label_only_entangled",
                "label_only_entangled_zero_recurrent",
                "label_only_entangled_threshold_only",
                "nuisance_causal_entangled_control",
            }
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
