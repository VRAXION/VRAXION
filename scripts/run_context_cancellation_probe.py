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
import xml.etree.ElementTree as ET

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
EMBEDDING_MODES = ("learned", "fixed_sincos", "trainable_phase", "multi_band_phase")
RESONANCE_MODES = (
    "none",
    "token_wave",
    "neuron_resonance",
    "pointer_resonance",
    "pointer_resonance_signed",
)
TOPOLOGY_MODES = (
    "random_sparse",
    "ring_sparse",
    "reciprocal_motif",
    "hub_rich",
    "hub_degree_preserving_random",
    "flywire_sampled",
    "flywire_class_sampled",
    "flywire_degree_preserving_random",
)
DEFAULT_FLYWIRE_GRAPHML = Path("/home/deck/work/flywire/mushroom_body.graphml")
_FLYWIRE_GRAPH_CACHE: dict[Path, dict[str, Any]] = {}
EXPERIMENTS = (
    "core_recovery",
    "latent_refraction",
    "multi_aspect_refraction",
    "multi_aspect_token_refraction",
    "frame_switch_diagnostics",
    "reframe_diagnostics",
    "inferred_frame_pointer",
    "query_cued_frame_pointer",
    "query_cued_pointer_bottleneck",
)

TASK_FRAMES = ("danger_frame", "environment_frame", "visibility_frame")
FEATURE_GROUPS = ("actor_action", "place_noise", "light", "object")
FRAME_ACTIVE_GROUP = {
    "danger_frame": "actor_action",
    "environment_frame": "place_noise",
    "visibility_frame": "light",
}
LOW_VISIBILITY_LIGHTS = {"shadow", "dusk"}
OBJECT_ALERT_ITEMS = {"stick", "wire"}

RELATIONS = ("owner", "stranger", "play", "ignore", "vet", "food")
SOUNDS = ("bark", "meow", "chirp", "laugh", "hiss", "beep", "silence")
FRIENDLY_RELATIONS_BY_ACTOR = {
    "dog": {"owner", "play"},
    "cat": {"owner", "play"},
    "bird": {"owner"},
    "child": {"owner", "play"},
    "robot": {"owner"},
    "snake": {"food"},
}
SOUND_BY_ACTOR = {
    "dog": "bark",
    "cat": "meow",
    "bird": "chirp",
    "child": "laugh",
    "robot": "beep",
    "snake": "hiss",
}

MULTI_ASPECT_FRAMES = ("danger_frame", "friendship_frame", "sound_frame", "environment_frame")
MULTI_ASPECT_FEATURE_GROUPS = (
    "actor",
    "danger_action",
    "friendship_relation",
    "sound",
    "place_noise",
    "object",
    "actor_action",
    "actor_relation",
    "actor_sound",
)
MULTI_ASPECT_FRAME_ACTIVE_GROUP = {
    "danger_frame": "actor_action",
    "friendship_frame": "actor_relation",
    "sound_frame": "actor_sound",
    "environment_frame": "place_noise",
}
INFERRED_FRAME_FEATURE_GROUPS = (
    "actor",
    "danger_action",
    "friendship_relation",
    "sound",
    "place_noise",
    "object",
    "light",
    "actor_action",
    "actor_relation",
    "actor_sound",
)
INFERRED_ACTIVE_SCALE = 1.35
INFERRED_INACTIVE_SCALE = 0.65
INFERRED_FRAME_LOSS_WEIGHT = 0.50
QUERY_CUES = ("danger_query", "friendship_query", "sound_query", "environment_query")
QUERY_TO_FRAME = {
    "danger_query": "danger_frame",
    "friendship_query": "friendship_frame",
    "sound_query": "sound_frame",
    "environment_query": "environment_frame",
}
QUERY_BOTTLENECK_SIZES = (2, 4, 8, 16)
TOKEN_FRAME_INVENTORY_SPECS = (
    ("dog", "actor", "dog"),
    ("cat", "actor", "cat"),
    ("snake", "actor", "snake"),
    ("bite", "action", "bite"),
    ("owner", "relation", "owner"),
    ("bark", "sound", "bark"),
    ("street", "place", "street"),
    ("car_noise", "noise", "traffic"),
    ("light", "light", "shadow"),
)
FRAME_PLACEMENTS = (
    "frame_in_recurrence_only",
    "frame_initial_only",
    "frame_at_output_only",
    "no_frame",
)
FRAME_SWITCH_PAIRS = (
    ("danger_frame", "environment_frame"),
    ("environment_frame", "danger_frame"),
    ("danger_frame", "sound_frame"),
    ("sound_frame", "danger_frame"),
    ("friendship_frame", "environment_frame"),
)
SOFT_FRAME_INTERPOLATION_PAIR = ("danger_frame", "environment_frame")
SOFT_FRAME_MIXES = (1.0, 0.75, 0.50, 0.25, 0.0)
REFRAME_RESET_SCALE = 1.0


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
    topology_mode: str
    flywire_graphml: str
    holdout_fraction: float
    active_value: float
    embed_scale: float
    embedding_mode: str
    resonance_mode: str
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
class RefractionDataBundle:
    x: np.ndarray
    y: np.ndarray
    frame: np.ndarray
    base_id: np.ndarray
    observation_component: np.ndarray
    frame_component: np.ndarray
    group_components: dict[str, np.ndarray]
    group_labels: dict[str, np.ndarray]
    frame_names: list[str]
    active_group_by_frame: dict[str, str] | None = None
    tokens: dict[str, np.ndarray] | None = None
    query_component: np.ndarray | None = None
    query: np.ndarray | None = None
    query_names: list[str] | None = None


@dataclass
class FeatureEmbeddings:
    input_mode: str
    embedding_mode: str
    resonance_mode: str
    vectors: dict[tuple[str, str], np.ndarray]
    token_phases: dict[tuple[str, str], np.ndarray] | None = None
    neuron_phases: np.ndarray | None = None
    pointer_phases: dict[str, np.ndarray] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny recurrent mechanism probes. The default path is the v4 core-recovery probe; "
            "--experiment latent_refraction runs the v6 task-frame/prism probe."
        )
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="core_recovery")
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
    parser.add_argument(
        "--topology-mode",
        choices=TOPOLOGY_MODES,
        default="random_sparse",
        help="Recurrent mask topology prior. All non-random modes are edge-budget matched to random_sparse.",
    )
    parser.add_argument(
        "--flywire-graphml",
        type=Path,
        default=DEFAULT_FLYWIRE_GRAPHML,
        help="Local GraphML sample used only by flywire_* topology modes.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--active-value", type=float, default=1.0)
    parser.add_argument("--embed-scale", type=float, default=0.80)
    parser.add_argument(
        "--embedding-mode",
        choices=EMBEDDING_MODES,
        default="learned",
        help=(
            "Token vector construction for small ablations. 'learned' is the existing random-vector baseline; "
            "phase modes are fixed phase-parameterized vectors in this precomputed-input probe."
        ),
    )
    parser.add_argument(
        "--resonance-mode",
        choices=RESONANCE_MODES,
        default="none",
        help="Small wave/resonance ablation layered onto existing refraction tasks.",
    )
    parser.add_argument("--nuisance-scale", type=float, default=1.05)
    parser.add_argument("--frame-scale", type=float, default=1.10)
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
    if args.topology_mode.startswith("flywire") and not args.flywire_graphml.is_file():
        raise SystemExit(f"--flywire-graphml not found: {args.flywire_graphml}")
    return args


def selected_input_modes(input_mode: str) -> list[str]:
    return list(INPUT_MODES) if input_mode == "all" else [input_mode]


def feature_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for group, values in (
        ("actor", ACTORS),
        ("action", ACTIONS),
        ("relation", RELATIONS),
        ("sound", SOUNDS),
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


def normalized(vec: np.ndarray, scale: float) -> np.ndarray:
    vec = vec.astype(np.float32)
    vec /= max(float(np.linalg.norm(vec)), 1.0e-9)
    return vec * scale


def token_index(key: tuple[str, str]) -> int:
    return feature_keys().index(key) + 1


def group_index(group: str) -> int:
    groups = ["actor", "action", "relation", "sound", "place", "light", "noise", "object"]
    return groups.index(group) + 1


def sincos_from_phase(phase: np.ndarray, hidden: int, scale: float) -> np.ndarray:
    half = hidden // 2
    vec = np.concatenate([np.sin(phase[:half]), np.cos(phase[:half])], axis=0)
    if hidden % 2:
        vec = np.concatenate([vec, np.array([np.sin(float(np.sum(phase)))], dtype=np.float32)], axis=0)
    return normalized(vec[:hidden], scale)


def phase_embedding_vector(
    *,
    key: tuple[str, str],
    hidden: int,
    scale: float,
    seed: int,
    embedding_mode: str,
) -> np.ndarray:
    if embedding_mode == "learned":
        rng = np.random.default_rng(seed + 17_003 * token_index(key))
        return unit_vector(rng, hidden, scale)

    token = token_index(key)
    group = group_index(key[0])
    half = max(1, hidden // 2)
    positions = np.arange(1, half + 1, dtype=np.float32)

    if embedding_mode == "fixed_sincos":
        base = 0.071 * token + 0.193 * group
        phase = base * positions + 0.37 * group
        return sincos_from_phase(phase, hidden, scale)

    if embedding_mode == "trainable_phase":
        rng = np.random.default_rng(seed + 29_011 * token + 311 * group)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=half).astype(np.float32)
        frequency = rng.uniform(0.35, 2.75, size=half).astype(np.float32)
        return sincos_from_phase(phase + frequency * positions, hidden, scale)

    if embedding_mode == "multi_band_phase":
        bands = 4
        chunks: list[np.ndarray] = []
        remaining = hidden
        for band in range(bands):
            width = remaining // (bands - band)
            remaining -= width
            band_half = max(1, width // 2)
            band_pos = np.arange(1, band_half + 1, dtype=np.float32)
            phase = (
                (0.047 * token * (band + 1) + 0.113 * group)
                * band_pos
                + 0.41 * band
                + 0.07 * token
            )
            chunk = np.concatenate([np.sin(phase), np.cos(phase)], axis=0)
            if width % 2:
                chunk = np.concatenate([chunk, np.array([np.cos(float(np.sum(phase)))], dtype=np.float32)], axis=0)
            chunks.append(chunk[:width])
        return normalized(np.concatenate(chunks, axis=0), scale)

    raise ValueError(f"unknown embedding mode: {embedding_mode}")


def build_wave_phase_tables(
    *,
    hidden: int,
    seed: int,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    token_phases: dict[tuple[str, str], np.ndarray] = {}
    positions = np.arange(1, hidden + 1, dtype=np.float32)
    for key in feature_keys():
        token = token_index(key)
        group = group_index(key[0])
        token_phases[key] = (
            0.031 * token * positions
            + 0.071 * group * np.sqrt(positions)
            + 0.17 * token
        ).astype(np.float32)

    neuron_positions = np.arange(1, hidden + 1, dtype=np.float32)
    neuron_phases = (0.097 * neuron_positions + 0.013 * neuron_positions**1.3).astype(np.float32)
    pointer_phases: dict[str, np.ndarray] = {}
    all_frames = tuple(dict.fromkeys((*TASK_FRAMES, *MULTI_ASPECT_FRAMES)))
    for frame_index, frame_name in enumerate(all_frames, start=1):
        pointer_phases[frame_name] = (
            0.043 * frame_index * neuron_positions
            + 0.29 * frame_index
            + 0.019 * np.sin(neuron_positions * frame_index)
        ).astype(np.float32)
    return token_phases, neuron_phases, pointer_phases


def resonance_token_vector(
    *,
    embeddings: FeatureEmbeddings,
    key: tuple[str, str],
    frame_name: str | None,
    neuron_phases_override: np.ndarray | None = None,
    pointer_frame_name: str | None = None,
) -> np.ndarray:
    if embeddings.resonance_mode in {"none", "token_wave"}:
        return embeddings.vectors[key]
    if embeddings.token_phases is None or embeddings.neuron_phases is None:
        return embeddings.vectors[key]

    neuron_phases = embeddings.neuron_phases if neuron_phases_override is None else neuron_phases_override
    pointer = np.zeros_like(neuron_phases)
    if embeddings.resonance_mode in {"pointer_resonance", "pointer_resonance_signed"}:
        pointer_key = pointer_frame_name if pointer_frame_name is not None else frame_name
        if pointer_key is not None and embeddings.pointer_phases is not None:
            pointer = embeddings.pointer_phases[pointer_key]

    phase = embeddings.token_phases[key] + pointer - neuron_phases
    vec = np.cos(phase)
    if embeddings.resonance_mode == "pointer_resonance_signed":
        carrier = np.sign(np.sin(neuron_phases * (group_index(key[0]) + 1) + 0.17 * token_index(key)))
        carrier[carrier == 0.0] = 1.0
        vec = vec * carrier
    return normalized(vec, 1.0)


def resonance_component(
    *,
    embeddings: FeatureEmbeddings,
    keys: list[tuple[str, str]],
    frame_name: str | None,
    scale: float,
    neuron_phases_override: np.ndarray | None = None,
    pointer_frame_name: str | None = None,
) -> np.ndarray:
    values = [
        resonance_token_vector(
            embeddings=embeddings,
            key=key,
            frame_name=frame_name,
            neuron_phases_override=neuron_phases_override,
            pointer_frame_name=pointer_frame_name,
        )
        for key in keys
    ]
    return scale * np.sum(values, axis=0).astype(np.float32)


def build_embeddings(
    *,
    schema: Schema,
    input_mode: str,
    seed: int,
    embed_scale: float,
    opponent_strength: float,
    embedding_mode: str = "learned",
    resonance_mode: str = "none",
) -> FeatureEmbeddings:
    rng = np.random.default_rng(seed)
    hidden = schema.hidden
    effective_embedding_mode = "fixed_sincos" if resonance_mode == "token_wave" else embedding_mode
    core_dim = max(4, hidden // 2)
    nuisance_dim = hidden - core_dim
    core_keys = [(group, value) for group, value in feature_keys() if group in {"actor", "action"}]
    nuisance_keys = [(group, value) for group, value in feature_keys() if group not in {"actor", "action"}]
    vectors: dict[tuple[str, str], np.ndarray] = {}

    if input_mode == "separable":
        for key in core_keys:
            vec = np.zeros(hidden, dtype=np.float32)
            if effective_embedding_mode == "learned":
                vec[:core_dim] = unit_vector(rng, core_dim, embed_scale)
            else:
                vec[:core_dim] = phase_embedding_vector(
                    key=key,
                    hidden=core_dim,
                    scale=embed_scale,
                    seed=seed,
                    embedding_mode=effective_embedding_mode,
                )
            vectors[key] = vec
        for key in nuisance_keys:
            vec = np.zeros(hidden, dtype=np.float32)
            if effective_embedding_mode == "learned":
                vec[core_dim:] = unit_vector(rng, nuisance_dim, embed_scale)
            else:
                vec[core_dim:] = phase_embedding_vector(
                    key=key,
                    hidden=nuisance_dim,
                    scale=embed_scale,
                    seed=seed,
                    embedding_mode=effective_embedding_mode,
                )
            vectors[key] = vec
    elif input_mode == "entangled":
        for key in core_keys + nuisance_keys:
            if effective_embedding_mode == "learned":
                vectors[key] = unit_vector(rng, hidden, embed_scale)
            else:
                vectors[key] = phase_embedding_vector(
                    key=key,
                    hidden=hidden,
                    scale=embed_scale,
                    seed=seed,
                    embedding_mode=effective_embedding_mode,
                )
    elif input_mode == "opponent_entangled":
        core_bank = [
            unit_vector(rng, hidden, embed_scale)
            if effective_embedding_mode == "learned"
            else phase_embedding_vector(
                key=key,
                hidden=hidden,
                scale=embed_scale,
                seed=seed,
                embedding_mode=effective_embedding_mode,
            )
            for key in core_keys
        ]
        for key, vec in zip(core_keys, core_bank):
            vectors[key] = vec
        for idx, key in enumerate(nuisance_keys):
            opponent = -opponent_strength * core_bank[idx % len(core_bank)]
            if effective_embedding_mode == "learned":
                noise = unit_vector(rng, hidden, embed_scale * (1.0 - min(opponent_strength, 0.95) * 0.45))
            else:
                noise = phase_embedding_vector(
                    key=key,
                    hidden=hidden,
                    scale=embed_scale * (1.0 - min(opponent_strength, 0.95) * 0.45),
                    seed=seed,
                    embedding_mode=effective_embedding_mode,
                )
            vec = opponent + noise
            vec /= max(float(np.linalg.norm(vec)), 1.0e-9)
            vectors[key] = vec.astype(np.float32) * embed_scale
    else:
        raise ValueError(f"unknown input mode: {input_mode}")

    token_phases = neuron_phases = pointer_phases = None
    if resonance_mode in {"neuron_resonance", "pointer_resonance", "pointer_resonance_signed"}:
        token_phases, neuron_phases, pointer_phases = build_wave_phase_tables(hidden=hidden, seed=seed)

    return FeatureEmbeddings(
        input_mode=input_mode,
        embedding_mode=effective_embedding_mode,
        resonance_mode=resonance_mode,
        vectors=vectors,
        token_phases=token_phases,
        neuron_phases=neuron_phases,
        pointer_phases=pointer_phases,
    )


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


def visibility_label(light: str) -> int:
    return int(light in LOW_VISIBILITY_LIGHTS)


def object_alert_label(obj: str) -> int:
    return int(obj in OBJECT_ALERT_ITEMS)


def frame_label(frame: str, *, actor: str, action: str, place: str, light: str, noise: str, obj: str) -> int:
    if frame == "danger_frame":
        return relation_label(actor, action)
    if frame == "environment_frame":
        return nuisance_causal_label(place, noise)
    if frame == "visibility_frame":
        return visibility_label(light)
    raise ValueError(f"unknown task frame: {frame}")


def friendship_label(actor: str, relation: str) -> int:
    return int(relation in FRIENDLY_RELATIONS_BY_ACTOR[actor])


def sound_label(actor: str, sound: str) -> int:
    return int(SOUND_BY_ACTOR[actor] == sound)


def multi_aspect_label(
    frame: str,
    *,
    actor: str,
    action: str,
    relation: str,
    sound: str,
    place: str,
    noise: str,
) -> int:
    if frame == "danger_frame":
        return relation_label(actor, action)
    if frame == "friendship_frame":
        return friendship_label(actor, relation)
    if frame == "sound_frame":
        return sound_label(actor, sound)
    if frame == "environment_frame":
        return nuisance_causal_label(place, noise)
    raise ValueError(f"unknown multi-aspect frame: {frame}")


def choose_value_for_binary_label(
    rng: np.random.Generator,
    values: tuple[str, ...],
    label_fn: Any,
    positive: bool,
) -> str:
    candidates = [value for value in values if label_fn(value) == int(positive)]
    if not candidates:
        candidates = list(values)
    return candidates[rng.integers(len(candidates))]


def choose_combo_for_refraction_targets(
    rng: np.random.Generator,
    combos: list[tuple[str, str, str, str]],
    *,
    environment_positive: bool,
    visibility_positive: bool,
) -> tuple[str, str, str, str]:
    candidates = [
        combo
        for combo in combos
        if nuisance_causal_label(combo[0], combo[2]) == int(environment_positive)
        and visibility_label(combo[1]) == int(visibility_positive)
    ]
    if not candidates:
        candidates = combos
    return candidates[rng.integers(len(candidates))]


def build_frame_embeddings(hidden: int, seed: int, frame_scale: float) -> dict[str, np.ndarray]:
    return build_named_frame_embeddings(TASK_FRAMES, hidden, seed, frame_scale)


def build_query_embeddings(hidden: int, seed: int, query_scale: float) -> dict[str, np.ndarray]:
    return build_named_frame_embeddings(QUERY_CUES, hidden, seed, query_scale)


def build_named_frame_embeddings(
    frame_names: tuple[str, ...],
    hidden: int,
    seed: int,
    frame_scale: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        frame: unit_vector(rng, hidden, frame_scale).astype(np.float32)
        for frame in frame_names
    }


def selected_refraction_input_modes(input_mode: str) -> list[str]:
    if input_mode == "all":
        return ["entangled", "separable"]
    return [input_mode]


def bundle_feature_groups(bundle: RefractionDataBundle) -> list[str]:
    return list(bundle.group_components.keys())


def bundle_active_group(bundle: RefractionDataBundle, frame_name: str) -> str:
    active = bundle.active_group_by_frame or FRAME_ACTIVE_GROUP
    return active[frame_name]


def refraction_x_from_components(
    *,
    observation_component: np.ndarray,
    frame_component: np.ndarray,
    no_frame_token: bool,
) -> np.ndarray:
    if no_frame_token:
        return observation_component.copy()
    return observation_component + frame_component


def make_refraction_dataset(
    *,
    n: int,
    combos: list[tuple[str, str, str, str]],
    seed: int,
    embeddings: FeatureEmbeddings,
    frame_embeddings: dict[str, np.ndarray],
    active_value: float,
    no_frame_token: bool = False,
    random_labels: bool = False,
) -> RefractionDataBundle:
    rng = np.random.default_rng(seed)
    hidden = len(next(iter(embeddings.vectors.values())))
    base_count = max(1, n // len(TASK_FRAMES))
    total = base_count * len(TASK_FRAMES)

    y = np.zeros(total, dtype=np.int64)
    frame = np.zeros(total, dtype=np.int64)
    base_id = np.zeros(total, dtype=np.int64)
    observation_component = np.zeros((total, hidden), dtype=np.float32)
    frame_component = np.zeros((total, hidden), dtype=np.float32)
    group_components = {
        group: np.zeros((total, hidden), dtype=np.float32)
        for group in FEATURE_GROUPS
    }
    group_labels = {
        group: np.zeros(total, dtype=np.int64)
        for group in FEATURE_GROUPS
    }

    row = 0
    for base in range(base_count):
        danger_positive = bool(rng.integers(0, 2))
        environment_positive = bool(rng.integers(0, 2))
        visibility_positive = bool(rng.integers(0, 2))
        actor, action = choose_relation_pair(rng, danger_positive)
        place, light, noise, obj = choose_combo_for_refraction_targets(
            rng,
            combos,
            environment_positive=environment_positive,
            visibility_positive=visibility_positive,
        )

        actor_action_component = active_value * (
            embeddings.vectors[("actor", actor)] + embeddings.vectors[("action", action)]
        )
        place_noise_component = active_value * (
            embeddings.vectors[("place", place)] + embeddings.vectors[("noise", noise)]
        )
        light_component = active_value * embeddings.vectors[("light", light)]
        object_component = active_value * embeddings.vectors[("object", obj)]
        observation = actor_action_component + place_noise_component + light_component + object_component
        labels = {
            "actor_action": relation_label(actor, action),
            "place_noise": nuisance_causal_label(place, noise),
            "light": visibility_label(light),
            "object": object_alert_label(obj),
        }

        for frame_index, frame_name in enumerate(TASK_FRAMES):
            frame[row] = frame_index
            base_id[row] = base
            observation_component[row] = observation
            frame_component[row] = active_value * frame_embeddings[frame_name]
            group_components["actor_action"][row] = actor_action_component
            group_components["place_noise"][row] = place_noise_component
            group_components["light"][row] = light_component
            group_components["object"][row] = object_component
            for group, value in labels.items():
                group_labels[group][row] = value
            if random_labels:
                y[row] = rng.integers(0, 2)
            else:
                y[row] = labels[FRAME_ACTIVE_GROUP[frame_name]]
            row += 1

    x = refraction_x_from_components(
        observation_component=observation_component,
        frame_component=frame_component,
        no_frame_token=no_frame_token,
    )
    order = rng.permutation(total)
    return RefractionDataBundle(
        x=x[order],
        y=y[order],
        frame=frame[order],
        base_id=base_id[order],
        observation_component=observation_component[order],
        frame_component=frame_component[order],
        group_components={group: values[order] for group, values in group_components.items()},
        group_labels={group: values[order] for group, values in group_labels.items()},
        frame_names=list(TASK_FRAMES),
        active_group_by_frame=dict(FRAME_ACTIVE_GROUP),
    )


def make_multi_aspect_dataset(
    *,
    n: int,
    combos: list[tuple[str, str, str, str]],
    seed: int,
    embeddings: FeatureEmbeddings,
    frame_embeddings: dict[str, np.ndarray],
    active_value: float,
    no_frame_token: bool = False,
    random_labels: bool = False,
) -> RefractionDataBundle:
    rng = np.random.default_rng(seed)
    hidden = len(next(iter(embeddings.vectors.values())))
    base_count = max(1, n // len(MULTI_ASPECT_FRAMES))
    total = base_count * len(MULTI_ASPECT_FRAMES)

    y = np.zeros(total, dtype=np.int64)
    frame = np.zeros(total, dtype=np.int64)
    base_id = np.zeros(total, dtype=np.int64)
    observation_component = np.zeros((total, hidden), dtype=np.float32)
    frame_component = np.zeros((total, hidden), dtype=np.float32)
    group_components = {
        group: np.zeros((total, hidden), dtype=np.float32)
        for group in MULTI_ASPECT_FEATURE_GROUPS
    }
    group_labels = {
        group: np.zeros(total, dtype=np.int64)
        for group in MULTI_ASPECT_FEATURE_GROUPS
    }
    tokens = {
        key: np.empty(total, dtype=object)
        for key in ("actor", "action", "relation", "sound", "place", "light", "noise", "object")
    }

    row = 0
    for base in range(base_count):
        actor = ACTORS[base % len(ACTORS)]
        danger_positive = bool(rng.integers(0, 2))
        friendship_positive = bool(rng.integers(0, 2))
        sound_positive = bool(rng.integers(0, 2))
        environment_positive = bool(rng.integers(0, 2))
        visibility_positive = bool(rng.integers(0, 2))
        action = choose_value_for_binary_label(
            rng,
            ACTIONS,
            lambda value, actor=actor: relation_label(actor, value),
            danger_positive,
        )
        relation = choose_value_for_binary_label(
            rng,
            RELATIONS,
            lambda value, actor=actor: friendship_label(actor, value),
            friendship_positive,
        )
        sound = choose_value_for_binary_label(
            rng,
            SOUNDS,
            lambda value, actor=actor: sound_label(actor, value),
            sound_positive,
        )
        place, light, noise, obj = choose_combo_for_refraction_targets(
            rng,
            combos,
            environment_positive=environment_positive,
            visibility_positive=visibility_positive,
        )

        labels = {
            "actor": int(actor == "dog"),
            "danger_action": relation_label(actor, action),
            "friendship_relation": friendship_label(actor, relation),
            "sound": sound_label(actor, sound),
            "place_noise": nuisance_causal_label(place, noise),
            "object": object_alert_label(obj),
            "actor_action": relation_label(actor, action),
            "actor_relation": friendship_label(actor, relation),
            "actor_sound": sound_label(actor, sound),
        }

        for frame_index, frame_name in enumerate(MULTI_ASPECT_FRAMES):
            resonance_frame = None if no_frame_token else frame_name
            actor_component = resonance_component(
                embeddings=embeddings,
                keys=[("actor", actor)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            action_component = resonance_component(
                embeddings=embeddings,
                keys=[("action", action)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            relation_component = resonance_component(
                embeddings=embeddings,
                keys=[("relation", relation)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            sound_component = resonance_component(
                embeddings=embeddings,
                keys=[("sound", sound)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            place_noise_component = resonance_component(
                embeddings=embeddings,
                keys=[("place", place), ("noise", noise)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            object_component = resonance_component(
                embeddings=embeddings,
                keys=[("object", obj)],
                frame_name=resonance_frame,
                scale=active_value,
            )
            observation = (
                actor_component
                + action_component
                + relation_component
                + sound_component
                + place_noise_component
                + object_component
            )
            frame[row] = frame_index
            base_id[row] = base
            observation_component[row] = observation
            frame_component[row] = active_value * frame_embeddings[frame_name]
            group_components["actor"][row] = actor_component
            group_components["danger_action"][row] = action_component
            group_components["friendship_relation"][row] = relation_component
            group_components["sound"][row] = sound_component
            group_components["place_noise"][row] = place_noise_component
            group_components["object"][row] = object_component
            group_components["actor_action"][row] = actor_component + action_component
            group_components["actor_relation"][row] = actor_component + relation_component
            group_components["actor_sound"][row] = actor_component + sound_component
            for group, value in labels.items():
                group_labels[group][row] = value
            for key, value in (
                ("actor", actor),
                ("action", action),
                ("relation", relation),
                ("sound", sound),
                ("place", place),
                ("light", light),
                ("noise", noise),
                ("object", obj),
            ):
                tokens[key][row] = value
            if random_labels:
                y[row] = rng.integers(0, 2)
            else:
                y[row] = labels[MULTI_ASPECT_FRAME_ACTIVE_GROUP[frame_name]]
            row += 1

    x = refraction_x_from_components(
        observation_component=observation_component,
        frame_component=frame_component,
        no_frame_token=no_frame_token,
    )
    order = rng.permutation(total)
    return RefractionDataBundle(
        x=x[order],
        y=y[order],
        frame=frame[order],
        base_id=base_id[order],
        observation_component=observation_component[order],
        frame_component=frame_component[order],
        group_components={group: values[order] for group, values in group_components.items()},
        group_labels={group: values[order] for group, values in group_labels.items()},
        frame_names=list(MULTI_ASPECT_FRAMES),
        active_group_by_frame=dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        tokens={key: values[order] for key, values in tokens.items()},
    )


def random_value(rng: np.random.Generator, values: tuple[str, ...]) -> str:
    return values[int(rng.integers(len(values)))]


def choose_place_noise_for_label(
    rng: np.random.Generator,
    combos: list[tuple[str, str, str, str]],
    positive: bool,
) -> tuple[str, str, str, str]:
    candidates = [combo for combo in combos if nuisance_causal_label(combo[0], combo[2]) == int(positive)]
    if not candidates:
        candidates = combos
    return candidates[int(rng.integers(len(candidates)))]


def make_inferred_frame_dataset(
    *,
    n: int,
    combos: list[tuple[str, str, str, str]],
    seed: int,
    embeddings: FeatureEmbeddings,
    frame_embeddings: dict[str, np.ndarray],
    active_value: float,
    random_labels: bool = False,
) -> RefractionDataBundle:
    rng = np.random.default_rng(seed)
    hidden = len(next(iter(embeddings.vectors.values())))
    total = n

    y = np.zeros(total, dtype=np.int64)
    frame = np.zeros(total, dtype=np.int64)
    base_id = np.arange(total, dtype=np.int64)
    observation_component = np.zeros((total, hidden), dtype=np.float32)
    frame_component = np.zeros((total, hidden), dtype=np.float32)
    group_components = {
        group: np.zeros((total, hidden), dtype=np.float32)
        for group in INFERRED_FRAME_FEATURE_GROUPS
    }
    group_labels = {
        group: np.zeros(total, dtype=np.int64)
        for group in INFERRED_FRAME_FEATURE_GROUPS
    }
    tokens = {
        key: np.empty(total, dtype=object)
        for key in ("actor", "action", "relation", "sound", "place", "light", "noise", "object")
    }

    for row in range(total):
        frame_index = row % len(MULTI_ASPECT_FRAMES)
        frame_name = MULTI_ASPECT_FRAMES[frame_index]
        target_positive = bool(rng.integers(0, 2))
        actor = ACTORS[(row // len(MULTI_ASPECT_FRAMES)) % len(ACTORS)]
        action = random_value(rng, ACTIONS)
        relation = random_value(rng, RELATIONS)
        sound = random_value(rng, SOUNDS)
        place, light, noise, obj = combos[int(rng.integers(len(combos)))]

        if frame_name == "danger_frame":
            action = choose_value_for_binary_label(
                rng,
                ACTIONS,
                lambda value, actor=actor: relation_label(actor, value),
                target_positive,
            )
        elif frame_name == "friendship_frame":
            relation = choose_value_for_binary_label(
                rng,
                RELATIONS,
                lambda value, actor=actor: friendship_label(actor, value),
                target_positive,
            )
        elif frame_name == "sound_frame":
            sound = choose_value_for_binary_label(
                rng,
                SOUNDS,
                lambda value, actor=actor: sound_label(actor, value),
                target_positive,
            )
        elif frame_name == "environment_frame":
            place, light, noise, obj = choose_place_noise_for_label(rng, combos, target_positive)
        else:
            raise ValueError(f"unknown inferred frame: {frame_name}")

        active_scale = active_value * INFERRED_ACTIVE_SCALE
        inactive_scale = active_value * INFERRED_INACTIVE_SCALE
        actor_scale = active_scale if frame_name in {"danger_frame", "friendship_frame", "sound_frame"} else inactive_scale
        action_scale = active_scale if frame_name == "danger_frame" else inactive_scale
        relation_scale = active_scale if frame_name == "friendship_frame" else inactive_scale
        sound_scale = active_scale if frame_name == "sound_frame" else inactive_scale
        place_noise_scale = active_scale if frame_name == "environment_frame" else inactive_scale

        actor_component = resonance_component(
            embeddings=embeddings,
            keys=[("actor", actor)],
            frame_name=None,
            scale=actor_scale,
        )
        action_component = resonance_component(
            embeddings=embeddings,
            keys=[("action", action)],
            frame_name=None,
            scale=action_scale,
        )
        relation_component = resonance_component(
            embeddings=embeddings,
            keys=[("relation", relation)],
            frame_name=None,
            scale=relation_scale,
        )
        sound_component = resonance_component(
            embeddings=embeddings,
            keys=[("sound", sound)],
            frame_name=None,
            scale=sound_scale,
        )
        place_noise_component = resonance_component(
            embeddings=embeddings,
            keys=[("place", place), ("noise", noise)],
            frame_name=None,
            scale=place_noise_scale,
        )
        object_component = resonance_component(
            embeddings=embeddings,
            keys=[("object", obj)],
            frame_name=None,
            scale=inactive_scale,
        )
        light_component = resonance_component(
            embeddings=embeddings,
            keys=[("light", light)],
            frame_name=None,
            scale=inactive_scale,
        )
        observation = (
            actor_component
            + action_component
            + relation_component
            + sound_component
            + place_noise_component
            + object_component
            + light_component
        )
        labels = {
            "actor": int(actor == "dog"),
            "danger_action": relation_label(actor, action),
            "friendship_relation": friendship_label(actor, relation),
            "sound": sound_label(actor, sound),
            "place_noise": nuisance_causal_label(place, noise),
            "object": object_alert_label(obj),
            "light": visibility_label(light),
            "actor_action": relation_label(actor, action),
            "actor_relation": friendship_label(actor, relation),
            "actor_sound": sound_label(actor, sound),
        }
        frame[row] = frame_index
        observation_component[row] = observation
        frame_component[row] = active_value * frame_embeddings[frame_name]
        group_components["actor"][row] = actor_component
        group_components["danger_action"][row] = action_component
        group_components["friendship_relation"][row] = relation_component
        group_components["sound"][row] = sound_component
        group_components["place_noise"][row] = place_noise_component
        group_components["object"][row] = object_component
        group_components["light"][row] = light_component
        group_components["actor_action"][row] = actor_component + action_component
        group_components["actor_relation"][row] = actor_component + relation_component
        group_components["actor_sound"][row] = actor_component + sound_component
        for group, value in labels.items():
            group_labels[group][row] = value
        for key, value in (
            ("actor", actor),
            ("action", action),
            ("relation", relation),
            ("sound", sound),
            ("place", place),
            ("light", light),
            ("noise", noise),
            ("object", obj),
        ):
            tokens[key][row] = value
        y[row] = int(rng.integers(0, 2)) if random_labels else labels[MULTI_ASPECT_FRAME_ACTIVE_GROUP[frame_name]]

    order = rng.permutation(total)
    return RefractionDataBundle(
        x=observation_component[order],
        y=y[order],
        frame=frame[order],
        base_id=base_id[order],
        observation_component=observation_component[order],
        frame_component=frame_component[order],
        group_components={group: values[order] for group, values in group_components.items()},
        group_labels={group: values[order] for group, values in group_labels.items()},
        frame_names=list(MULTI_ASPECT_FRAMES),
        active_group_by_frame=dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        tokens={key: values[order] for key, values in tokens.items()},
    )


def make_query_cued_frame_dataset(
    *,
    n: int,
    combos: list[tuple[str, str, str, str]],
    seed: int,
    embeddings: FeatureEmbeddings,
    frame_embeddings: dict[str, np.ndarray],
    query_embeddings: dict[str, np.ndarray],
    active_value: float,
    random_labels: bool = False,
) -> RefractionDataBundle:
    rng = np.random.default_rng(seed)
    hidden = len(next(iter(embeddings.vectors.values())))
    base_count = max(1, n // len(QUERY_CUES))
    total = base_count * len(QUERY_CUES)

    y = np.zeros(total, dtype=np.int64)
    frame = np.zeros(total, dtype=np.int64)
    query = np.zeros(total, dtype=np.int64)
    base_id = np.zeros(total, dtype=np.int64)
    full_input_component = np.zeros((total, hidden), dtype=np.float32)
    query_component = np.zeros((total, hidden), dtype=np.float32)
    frame_component = np.zeros((total, hidden), dtype=np.float32)
    group_components = {
        group: np.zeros((total, hidden), dtype=np.float32)
        for group in INFERRED_FRAME_FEATURE_GROUPS
    }
    group_labels = {
        group: np.zeros(total, dtype=np.int64)
        for group in INFERRED_FRAME_FEATURE_GROUPS
    }
    tokens = {
        key: np.empty(total, dtype=object)
        for key in ("actor", "action", "relation", "sound", "place", "light", "noise", "object")
    }

    row = 0
    for base in range(base_count):
        actor = ACTORS[base % len(ACTORS)]
        danger_positive = bool(rng.integers(0, 2))
        friendship_positive = bool(rng.integers(0, 2))
        sound_positive = bool(rng.integers(0, 2))
        environment_positive = bool(rng.integers(0, 2))
        visibility_positive = bool(rng.integers(0, 2))
        action = choose_value_for_binary_label(
            rng,
            ACTIONS,
            lambda value, actor=actor: relation_label(actor, value),
            danger_positive,
        )
        relation = choose_value_for_binary_label(
            rng,
            RELATIONS,
            lambda value, actor=actor: friendship_label(actor, value),
            friendship_positive,
        )
        sound = choose_value_for_binary_label(
            rng,
            SOUNDS,
            lambda value, actor=actor: sound_label(actor, value),
            sound_positive,
        )
        place, light, noise, obj = choose_combo_for_refraction_targets(
            rng,
            combos,
            environment_positive=environment_positive,
            visibility_positive=visibility_positive,
        )

        actor_component = resonance_component(
            embeddings=embeddings,
            keys=[("actor", actor)],
            frame_name=None,
            scale=active_value,
        )
        action_component = resonance_component(
            embeddings=embeddings,
            keys=[("action", action)],
            frame_name=None,
            scale=active_value,
        )
        relation_component = resonance_component(
            embeddings=embeddings,
            keys=[("relation", relation)],
            frame_name=None,
            scale=active_value,
        )
        sound_component = resonance_component(
            embeddings=embeddings,
            keys=[("sound", sound)],
            frame_name=None,
            scale=active_value,
        )
        place_noise_component = resonance_component(
            embeddings=embeddings,
            keys=[("place", place), ("noise", noise)],
            frame_name=None,
            scale=active_value,
        )
        object_component = resonance_component(
            embeddings=embeddings,
            keys=[("object", obj)],
            frame_name=None,
            scale=active_value,
        )
        light_component = resonance_component(
            embeddings=embeddings,
            keys=[("light", light)],
            frame_name=None,
            scale=active_value,
        )
        base_observation = (
            actor_component
            + action_component
            + relation_component
            + sound_component
            + place_noise_component
            + object_component
            + light_component
        )
        labels = {
            "actor": int(actor == "dog"),
            "danger_action": relation_label(actor, action),
            "friendship_relation": friendship_label(actor, relation),
            "sound": sound_label(actor, sound),
            "place_noise": nuisance_causal_label(place, noise),
            "object": object_alert_label(obj),
            "light": visibility_label(light),
            "actor_action": relation_label(actor, action),
            "actor_relation": friendship_label(actor, relation),
            "actor_sound": sound_label(actor, sound),
        }

        for query_index, query_name in enumerate(QUERY_CUES):
            frame_name = QUERY_TO_FRAME[query_name]
            frame_index = MULTI_ASPECT_FRAMES.index(frame_name)
            query_vec = active_value * query_embeddings[query_name]
            frame[row] = frame_index
            query[row] = query_index
            base_id[row] = base
            query_component[row] = query_vec
            full_input_component[row] = base_observation + query_vec
            frame_component[row] = active_value * frame_embeddings[frame_name]
            group_components["actor"][row] = actor_component
            group_components["danger_action"][row] = action_component
            group_components["friendship_relation"][row] = relation_component
            group_components["sound"][row] = sound_component
            group_components["place_noise"][row] = place_noise_component
            group_components["object"][row] = object_component
            group_components["light"][row] = light_component
            group_components["actor_action"][row] = actor_component + action_component
            group_components["actor_relation"][row] = actor_component + relation_component
            group_components["actor_sound"][row] = actor_component + sound_component
            for group, value in labels.items():
                group_labels[group][row] = value
            for key, value in (
                ("actor", actor),
                ("action", action),
                ("relation", relation),
                ("sound", sound),
                ("place", place),
                ("light", light),
                ("noise", noise),
                ("object", obj),
            ):
                tokens[key][row] = value
            y[row] = (
                int(rng.integers(0, 2))
                if random_labels
                else labels[MULTI_ASPECT_FRAME_ACTIVE_GROUP[frame_name]]
            )
            row += 1

    order = rng.permutation(total)
    return RefractionDataBundle(
        x=full_input_component[order],
        y=y[order],
        frame=frame[order],
        base_id=base_id[order],
        observation_component=full_input_component[order],
        frame_component=frame_component[order],
        group_components={group: values[order] for group, values in group_components.items()},
        group_labels={group: values[order] for group, values in group_labels.items()},
        frame_names=list(MULTI_ASPECT_FRAMES),
        active_group_by_frame=dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        tokens={key: values[order] for key, values in tokens.items()},
        query_component=query_component[order],
        query=query[order],
        query_names=list(QUERY_CUES),
    )


def make_sparse_mask(hidden: int, density: float, seed: int) -> np.ndarray:
    if density >= 1.0:
        return np.ones((hidden, hidden), dtype=np.float32)
    rng = np.random.default_rng(seed)
    mask = (rng.random((hidden, hidden)) < density).astype(np.float32)
    np.fill_diagonal(mask, 1.0)
    return mask


def mask_edge_stats(mask: np.ndarray, *, topology_mode: str, target_edge_count: int) -> dict[str, Any]:
    mask_i = (mask > 0).astype(np.int8)
    hidden = int(mask_i.shape[0])
    offdiag = mask_i.copy()
    np.fill_diagonal(offdiag, 0)
    actual_edge_count = int(mask_i.sum())
    self_loop_count = int(np.trace(mask_i))
    offdiag_edge_count = int(offdiag.sum())
    reciprocal_pair_count = int(np.triu(offdiag * offdiag.T, k=1).sum())
    in_degree = offdiag.sum(axis=0).astype(np.float32)
    out_degree = offdiag.sum(axis=1).astype(np.float32)
    total_degree = in_degree + out_degree
    hub_concentration: dict[str, float] = {}
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(round(hidden * fraction)))
        hubs = set(int(idx) for idx in np.argsort(-total_degree, kind="stable")[:count])
        incident_edges = 0
        for source, target in np.argwhere(offdiag > 0):
            if int(source) in hubs or int(target) in hubs:
                incident_edges += 1
        label = f"top_{int(fraction * 100)}pct_hub_incident_edge_fraction"
        hub_concentration[label] = float(incident_edges / max(offdiag_edge_count, 1))
    return {
        "topology_mode": topology_mode,
        "target_edge_count": int(target_edge_count),
        "actual_edge_count": actual_edge_count,
        "self_loop_count": self_loop_count,
        "offdiag_edge_count": offdiag_edge_count,
        "density": float(actual_edge_count / max(hidden * hidden, 1)),
        "offdiag_density": float(offdiag_edge_count / max(hidden * (hidden - 1), 1)),
        "reciprocal_pair_count": reciprocal_pair_count,
        "reciprocal_pair_fraction": float((2.0 * reciprocal_pair_count) / max(offdiag_edge_count, 1)),
        "in_degree_mean": float(np.mean(in_degree)),
        "in_degree_std": float(np.std(in_degree)),
        "in_degree_max": float(np.max(in_degree)) if len(in_degree) else 0.0,
        "out_degree_mean": float(np.mean(out_degree)),
        "out_degree_std": float(np.std(out_degree)),
        "out_degree_max": float(np.max(out_degree)) if len(out_degree) else 0.0,
        "total_degree_std": float(np.std(total_degree)),
        "total_degree_max": float(np.max(total_degree)) if len(total_degree) else 0.0,
        **hub_concentration,
    }


def matched_edge_budget(hidden: int, density: float, seed: int) -> int:
    return int(make_sparse_mask(hidden, density, seed).sum())


def add_edge_if_room(mask: np.ndarray, source: int, target: int, target_edge_count: int) -> bool:
    if source == target or mask[source, target] > 0 or int(mask.sum()) >= target_edge_count:
        return False
    mask[source, target] = 1.0
    return True


def fill_random_edges(mask: np.ndarray, target_edge_count: int, rng: np.random.Generator) -> None:
    hidden = int(mask.shape[0])
    if int(mask.sum()) >= target_edge_count:
        return
    candidates = [(i, j) for i in range(hidden) for j in range(hidden) if i != j and mask[i, j] == 0]
    rng.shuffle(candidates)
    for source, target in candidates:
        if int(mask.sum()) >= target_edge_count:
            break
        mask[source, target] = 1.0


def sample_weighted_edges(
    *,
    hidden: int,
    candidates: list[tuple[int, int, float]],
    target_offdiag_edges: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.eye(hidden, dtype=np.float32)
    unique: dict[tuple[int, int], float] = {}
    for source, target, weight in candidates:
        if source == target:
            continue
        key = (int(source), int(target))
        unique[key] = max(float(weight), unique.get(key, 0.0))
    edges = list(unique)
    if not edges:
        fill_random_edges(mask, hidden + target_offdiag_edges, rng)
        return mask

    weights = np.array([np.log1p(max(unique[edge], 0.0)) for edge in edges], dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(edges), dtype=np.float64)
    probs = weights / weights.sum()
    take = min(target_offdiag_edges, len(edges))
    chosen = rng.choice(len(edges), size=take, replace=False, p=probs)
    for idx in chosen:
        source, target = edges[int(idx)]
        mask[source, target] = 1.0
    fill_random_edges(mask, hidden + target_offdiag_edges, rng)
    return mask


def ring_sparse_mask(hidden: int, target_edge_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.eye(hidden, dtype=np.float32)
    offsets: list[int] = []
    for base in (1, -1, 2, -2, 4, -4, 8, -8, 16, -16):
        if abs(base) < hidden:
            offsets.append(base)
    start = int(rng.integers(0, hidden))
    for offset in offsets:
        for step in range(hidden):
            source = (start + step) % hidden
            target = (source + offset) % hidden
            add_edge_if_room(mask, source, target, target_edge_count)
            if int(mask.sum()) >= target_edge_count:
                return mask
    fill_random_edges(mask, target_edge_count, rng)
    return mask


def reciprocal_motif_mask(hidden: int, target_edge_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.eye(hidden, dtype=np.float32)
    pairs = [(i, j) for i in range(hidden) for j in range(i + 1, hidden)]
    rng.shuffle(pairs)
    for source, target in pairs:
        add_edge_if_room(mask, source, target, target_edge_count)
        add_edge_if_room(mask, target, source, target_edge_count)
        if int(mask.sum()) >= target_edge_count:
            return mask
    fill_random_edges(mask, target_edge_count, rng)
    return mask


def hub_rich_mask(hidden: int, target_edge_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.eye(hidden, dtype=np.float32)
    ranks = np.arange(1, hidden + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.35)
    probs /= probs.sum()
    permutation = rng.permutation(hidden)
    attempts = 0
    max_attempts = max(10_000, target_edge_count * 80)
    while int(mask.sum()) < target_edge_count and attempts < max_attempts:
        attempts += 1
        source = int(permutation[rng.choice(hidden, p=probs)])
        if rng.random() < 0.65:
            target = int(permutation[rng.choice(hidden, p=probs)])
        else:
            target = int(rng.integers(0, hidden))
        add_edge_if_room(mask, source, target, target_edge_count)
    fill_random_edges(mask, target_edge_count, rng)
    return mask


def load_flywire_graph(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    cached = _FLYWIRE_GRAPH_CACHE.get(path)
    if cached is not None:
        return cached

    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(path).getroot()
    key_names = {key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"]) for key in root.findall("g:key", ns)}
    graph = root.find("g:graph", ns)
    if graph is None:
        raise ValueError(f"GraphML has no graph element: {path}")

    nodes: list[dict[str, str]] = []
    node_index: dict[str, int] = {}
    for node in graph.findall("g:node", ns):
        data = {
            key_names.get(item.attrib.get("key", ""), item.attrib.get("key", "")): item.text or ""
            for item in node.findall("g:data", ns)
        }
        node_id = node.attrib["id"]
        node_index[node_id] = len(nodes)
        nodes.append(
            {
                "id": node_id,
                "class": data.get("Class", ""),
                "pair": data.get("Pair", ""),
                "hemisphere": data.get("Hemisphere", ""),
            }
        )

    raw_edges: list[tuple[int, int, float]] = []
    degree = np.zeros(len(nodes), dtype=np.float64)
    for edge in graph.findall("g:edge", ns):
        source_id = edge.attrib["source"]
        target_id = edge.attrib["target"]
        if source_id not in node_index or target_id not in node_index:
            continue
        data = {
            key_names.get(item.attrib.get("key", ""), item.attrib.get("key", "")): item.text or ""
            for item in edge.findall("g:data", ns)
        }
        weight = float(data.get("weight", "1") or 1.0)
        source = node_index[source_id]
        target = node_index[target_id]
        raw_edges.append((source, target, weight))
        if source != target:
            degree[source] += 1.0
            degree[target] += 1.0

    out = {
        "path": str(path),
        "nodes": nodes,
        "edges": raw_edges,
        "degree": degree,
    }
    _FLYWIRE_GRAPH_CACHE[path] = out
    return out


def weighted_node_sample(
    *,
    candidates: list[int],
    degree: np.ndarray,
    count: int,
    rng: np.random.Generator,
    already: set[int],
) -> list[int]:
    available = [node for node in candidates if node not in already]
    if count <= 0 or not available:
        return []
    take = min(count, len(available))
    weights = np.array([degree[node] + 1.0 for node in available], dtype=np.float64)
    weights /= weights.sum()
    chosen = rng.choice(len(available), size=take, replace=False, p=weights)
    return [available[int(idx)] for idx in chosen]


def flywire_nodes_for_sample(
    *,
    graph: dict[str, Any],
    hidden: int,
    seed: int,
    class_balanced: bool,
) -> list[int]:
    rng = np.random.default_rng(seed)
    nodes = graph["nodes"]
    degree = graph["degree"]
    all_nodes = list(range(len(nodes)))
    if hidden > len(all_nodes):
        raise ValueError(f"hidden={hidden} exceeds local FlyWire graph nodes={len(all_nodes)}")

    if not class_balanced:
        weights = degree + 1.0
        weights /= weights.sum()
        chosen = rng.choice(len(all_nodes), size=hidden, replace=False, p=weights)
        return [int(idx) for idx in chosen]

    buckets = {
        "KC": [idx for idx, node in enumerate(nodes) if "KC" in node["class"]],
        "MBON": [idx for idx, node in enumerate(nodes) if "MBON" in node["class"]],
        "MBIN": [idx for idx, node in enumerate(nodes) if "MBIN" in node["class"]],
        "ORN_PN": [
            idx
            for idx, node in enumerate(nodes)
            if "ORN" in node["class"] or "PN" in node["class"] or "Gust" in node["class"]
        ],
    }
    quotas = {
        "KC": int(round(hidden * 0.45)),
        "MBON": int(round(hidden * 0.20)),
        "MBIN": int(round(hidden * 0.15)),
    }
    quotas["ORN_PN"] = max(0, hidden - sum(quotas.values()))

    selected: list[int] = []
    selected_set: set[int] = set()
    for bucket_name in ("KC", "MBON", "MBIN", "ORN_PN"):
        picked = weighted_node_sample(
            candidates=buckets[bucket_name],
            degree=degree,
            count=quotas[bucket_name],
            rng=rng,
            already=selected_set,
        )
        selected.extend(picked)
        selected_set.update(picked)

    if len(selected) < hidden:
        picked = weighted_node_sample(
            candidates=all_nodes,
            degree=degree,
            count=hidden - len(selected),
            rng=rng,
            already=selected_set,
        )
        selected.extend(picked)
    rng.shuffle(selected)
    return selected[:hidden]


def flywire_sampled_mask(
    *,
    hidden: int,
    target_edge_count: int,
    seed: int,
    graphml: Path,
    class_balanced: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    graph = load_flywire_graph(graphml)
    sampled_nodes = flywire_nodes_for_sample(
        graph=graph,
        hidden=hidden,
        seed=seed,
        class_balanced=class_balanced,
    )
    local_index = {node: idx for idx, node in enumerate(sampled_nodes)}
    candidates: list[tuple[int, int, float]] = []
    for source, target, weight in graph["edges"]:
        if source in local_index and target in local_index and source != target:
            candidates.append((local_index[source], local_index[target], weight))
    target_offdiag_edges = max(0, target_edge_count - hidden)
    mask = sample_weighted_edges(
        hidden=hidden,
        candidates=candidates,
        target_offdiag_edges=target_offdiag_edges,
        seed=seed + 17_771,
    )

    class_counts: dict[str, int] = {}
    for node in sampled_nodes:
        node_class = graph["nodes"][node]["class"] or "unknown"
        class_counts[node_class] = class_counts.get(node_class, 0) + 1
    extra = {
        "flywire_graphml": graph["path"],
        "flywire_class_balanced": class_balanced,
        "flywire_sampled_node_count": len(sampled_nodes),
        "flywire_induced_edge_candidates": len(candidates),
        "flywire_sampled_class_counts": class_counts,
    }
    return mask, extra


def make_topology_mask(
    *,
    hidden: int,
    density: float,
    seed: int,
    topology_mode: str,
    flywire_graphml: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_edge_count = matched_edge_budget(hidden, density, seed)
    extra: dict[str, Any] = {}

    if topology_mode == "random_sparse":
        mask = make_sparse_mask(hidden, density, seed)
    elif topology_mode == "ring_sparse":
        mask = ring_sparse_mask(hidden, target_edge_count, seed + 1_001)
    elif topology_mode == "reciprocal_motif":
        mask = reciprocal_motif_mask(hidden, target_edge_count, seed + 2_001)
    elif topology_mode == "hub_rich":
        mask = hub_rich_mask(hidden, target_edge_count, seed + 3_001)
    elif topology_mode == "hub_degree_preserving_random":
        base_mask = hub_rich_mask(hidden, target_edge_count, seed + 3_001)
        shuffle_seed = seed + 33_001
        mask = degree_preserving_shuffle_mask(base_mask, shuffle_seed)
        extra = {
            "degree_preserving_source_topology": "hub_rich",
            "degree_preserving_shuffle_seed": shuffle_seed,
        }
    elif topology_mode == "flywire_sampled":
        mask, extra = flywire_sampled_mask(
            hidden=hidden,
            target_edge_count=target_edge_count,
            seed=seed + 4_001,
            graphml=flywire_graphml,
            class_balanced=False,
        )
    elif topology_mode == "flywire_class_sampled":
        mask, extra = flywire_sampled_mask(
            hidden=hidden,
            target_edge_count=target_edge_count,
            seed=seed + 5_001,
            graphml=flywire_graphml,
            class_balanced=True,
        )
    elif topology_mode == "flywire_degree_preserving_random":
        base_mask, base_extra = flywire_sampled_mask(
            hidden=hidden,
            target_edge_count=target_edge_count,
            seed=seed + 4_001,
            graphml=flywire_graphml,
            class_balanced=False,
        )
        shuffle_seed = seed + 44_001
        mask = degree_preserving_shuffle_mask(base_mask, shuffle_seed)
        extra = {
            **base_extra,
            "degree_preserving_source_topology": "flywire_sampled",
            "degree_preserving_shuffle_seed": shuffle_seed,
        }
    else:
        raise ValueError(f"unknown topology mode: {topology_mode}")

    stats = mask_edge_stats(mask, topology_mode=topology_mode, target_edge_count=target_edge_count)
    stats.update(extra)
    return mask.astype(np.float32), stats


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


class FramePlacementRecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        steps: int,
        mask: np.ndarray,
        update_rate: float,
        delta_scale: float,
        frame_placement: str,
    ):
        super().__init__()
        if frame_placement not in FRAME_PLACEMENTS:
            raise ValueError(f"unknown frame placement: {frame_placement}")
        self.steps = steps
        self.update_rate = update_rate
        self.delta_scale = delta_scale
        self.frame_placement = frame_placement
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.recurrent = nn.Parameter(torch.empty(hidden, hidden))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        head_width = hidden * 2 if frame_placement == "frame_at_output_only" else hidden
        self.head = nn.Linear(head_width, 2)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout_components(
        self,
        observation: torch.Tensor,
        frame: torch.Tensor,
        *,
        ablation: str | None = None,
        frame_schedule: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        def frame_at(step: int) -> torch.Tensor:
            if frame_schedule is None:
                return frame
            return frame_schedule[min(step, len(frame_schedule) - 1)]

        if self.frame_placement == "frame_initial_only":
            h = torch.tanh(observation + frame_at(0))
        else:
            h = torch.tanh(observation)

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
                if self.frame_placement == "frame_in_recurrence_only":
                    delta = delta + frame_at(step)
                proposal = torch.tanh(h + self.delta_scale * delta)
            h = (1.0 - self.update_rate) * h + self.update_rate * proposal
            states.append(h)

        logits: list[torch.Tensor] = []
        for step, state in enumerate(states):
            if self.frame_placement == "frame_at_output_only":
                logits.append(self.head(torch.cat([state, frame_at(step)], dim=1)))
            else:
                logits.append(self.head(state))
        return states, logits


class InferredFramePointerClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        steps: int,
        mask: np.ndarray,
        update_rate: float,
        delta_scale: float,
        frame_embeddings: dict[str, np.ndarray],
    ):
        super().__init__()
        self.steps = steps
        self.update_rate = update_rate
        self.delta_scale = delta_scale
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        frame_table = np.stack([frame_embeddings[frame] for frame in MULTI_ASPECT_FRAMES], axis=0)
        self.register_buffer("frame_table", torch.tensor(frame_table, dtype=torch.float32))
        self.recurrent = nn.Parameter(torch.empty(hidden, hidden))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.frame_head = nn.Linear(hidden, len(MULTI_ASPECT_FRAMES))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.zeros_(self.frame_head.bias)
        nn.init.normal_(self.frame_head.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout_observation(
        self,
        observation: torch.Tensor,
        *,
        frame_override: torch.Tensor | None = None,
        use_pointer: bool = True,
        hard_frame: bool = False,
        ablation: str | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        h = torch.tanh(observation)
        frame_logits = self.frame_head(h)
        if not use_pointer:
            pointer = torch.zeros_like(h)
        elif frame_override is not None:
            pointer = self.frame_table.index_select(dim=0, index=frame_override.long())
        elif hard_frame:
            pointer = self.frame_table.index_select(dim=0, index=torch.argmax(frame_logits, dim=1))
        else:
            pointer = F.softmax(frame_logits, dim=1) @ self.frame_table

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
                if use_pointer:
                    delta = delta + pointer
                proposal = torch.tanh(h + self.delta_scale * delta)
            h = (1.0 - self.update_rate) * h + self.update_rate * proposal
            states.append(h)

        logits = [self.head(state) for state in states]
        return states, logits, frame_logits


class QueryCuedPointerClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        steps: int,
        mask: np.ndarray,
        update_rate: float,
        delta_scale: float,
        frame_embeddings: dict[str, np.ndarray],
    ):
        super().__init__()
        self.steps = steps
        self.update_rate = update_rate
        self.delta_scale = delta_scale
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        frame_table = np.stack([frame_embeddings[frame] for frame in MULTI_ASPECT_FRAMES], axis=0)
        self.register_buffer("frame_table", torch.tensor(frame_table, dtype=torch.float32))
        self.recurrent = nn.Parameter(torch.empty(hidden, hidden))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.frame_head = nn.Linear(hidden, len(MULTI_ASPECT_FRAMES))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.zeros_(self.frame_head.bias)
        nn.init.normal_(self.frame_head.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout_inputs(
        self,
        frame_input: torch.Tensor,
        decision_input: torch.Tensor,
        *,
        frame_override: torch.Tensor | None = None,
        use_pointer: bool = True,
        hard_frame: bool = False,
        ablation: str | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        frame_logits = self.frame_head(torch.tanh(frame_input))
        if not use_pointer:
            pointer = torch.zeros_like(decision_input)
        elif frame_override is not None:
            pointer = self.frame_table.index_select(dim=0, index=frame_override.long())
        elif hard_frame:
            pointer = self.frame_table.index_select(dim=0, index=torch.argmax(frame_logits, dim=1))
        else:
            pointer = F.softmax(frame_logits, dim=1) @ self.frame_table

        h = torch.tanh(decision_input)
        states = [h]
        masked_recurrent = self.recurrent * self.mask
        for _step in range(1, self.steps + 1):
            if ablation == "zero_recurrent_update":
                proposal = h
            else:
                if ablation == "zero_matrix_keep_threshold":
                    delta = self.threshold.unsqueeze(0).expand_as(h)
                elif ablation is None:
                    delta = h @ masked_recurrent.t() + self.threshold
                else:
                    raise ValueError(f"unknown ablation: {ablation}")
                if use_pointer:
                    delta = delta + pointer
                proposal = torch.tanh(h + self.delta_scale * delta)
            h = (1.0 - self.update_rate) * h + self.update_rate * proposal
            states.append(h)

        logits = [self.head(state) for state in states]
        return states, logits, frame_logits


class QueryBottleneckDirectClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        bottleneck: int,
        steps: int,
        mask: np.ndarray,
        update_rate: float,
        delta_scale: float,
    ):
        super().__init__()
        self.steps = steps
        self.bottleneck = bottleneck
        self.update_rate = update_rate
        self.delta_scale = delta_scale
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.query_encoder = nn.Linear(hidden, bottleneck)
        self.query_decoder = nn.Linear(bottleneck, hidden)
        self.recurrent = nn.Parameter(torch.empty(hidden, hidden))
        self.threshold = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.query_encoder.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.query_encoder.bias)
        nn.init.normal_(self.query_decoder.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.query_decoder.bias)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout_inputs(
        self,
        observation: torch.Tensor,
        query: torch.Tensor,
        *,
        ablation: str | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        query_control = self.query_decoder(torch.tanh(self.query_encoder(query)))
        h = torch.tanh(observation + query_control)
        states = [h]
        masked_recurrent = self.recurrent * self.mask
        for _step in range(1, self.steps + 1):
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


class ReframeRecurrentClassifier(nn.Module):
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
        self.reset_vector = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.recurrent, mean=0.0, std=0.025)
        nn.init.normal_(self.reset_vector, mean=0.0, std=0.025)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.025)

    def rollout_components(
        self,
        observation: torch.Tensor,
        frame: torch.Tensor,
        *,
        ablation: str | None = None,
        frame_schedule: list[torch.Tensor] | None = None,
        reset_schedule: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        def frame_at(step: int) -> torch.Tensor:
            if frame_schedule is None:
                return frame
            return frame_schedule[min(step, len(frame_schedule) - 1)]

        def reset_at(step: int) -> torch.Tensor:
            if reset_schedule is None:
                return torch.zeros((observation.shape[0], 1), dtype=observation.dtype, device=observation.device)
            return reset_schedule[min(step, len(reset_schedule) - 1)]

        h = torch.tanh(observation)
        states = [h]
        masked_recurrent = self.recurrent * self.mask
        freeze_after = None
        disable_reset = False
        if ablation == "disable_reset_pulse":
            disable_reset = True
            ablation = None
        if ablation and ablation.startswith("freeze_after_"):
            freeze_after = int(ablation.rsplit("_", maxsplit=1)[1])
            ablation = None

        for step in range(1, self.steps + 1):
            if freeze_after is not None and step > freeze_after:
                states.append(h)
                continue
            frame_step = frame_at(step)
            reset_step = reset_at(step)
            if disable_reset:
                reset_step = torch.zeros_like(reset_step)
            if ablation == "zero_recurrent_update":
                proposal = h
            else:
                if ablation == "zero_matrix_keep_threshold":
                    delta = self.threshold.unsqueeze(0).expand_as(h)
                elif ablation is None:
                    reset_anchor = torch.tanh(observation + frame_step + self.reset_vector)
                    h_for_update = (1.0 - reset_step) * h + reset_step * reset_anchor
                    delta = h_for_update @ masked_recurrent.t() + self.threshold + frame_step
                    delta = delta + reset_step * self.reset_vector
                    proposal = torch.tanh(h_for_update + self.delta_scale * delta)
                else:
                    raise ValueError(f"unknown ablation: {ablation}")
                if ablation == "zero_matrix_keep_threshold":
                    proposal = torch.tanh(h + self.delta_scale * (delta + frame_step))
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
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = RecurrentClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
    ).to(args.device)
    model.topology_stats = topology_stats
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


def train_frame_placement_model(
    *,
    train: RefractionDataBundle,
    hidden: int,
    args: argparse.Namespace,
    seed: int,
    frame_placement: str,
) -> FramePlacementRecurrentClassifier:
    torch.manual_seed(seed)
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = FramePlacementRecurrentClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        frame_placement=frame_placement,
    ).to(args.device)
    model.topology_stats = topology_stats
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    observation = to_tensor(train.observation_component, args.device)
    frame = to_tensor(train.frame_component, args.device)
    y = to_tensor(train.y, args.device, torch.long)
    n = len(train.y)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_001)

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            xb = observation.index_select(dim=0, index=batch_idx)
            fb = frame.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            _states, logits = model.rollout_components(xb, fb)
            loss = F.cross_entropy(logits[-1], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def train_inferred_frame_pointer_model(
    *,
    train: RefractionDataBundle,
    frame_embeddings: dict[str, np.ndarray],
    hidden: int,
    args: argparse.Namespace,
    seed: int,
    use_pointer: bool,
) -> InferredFramePointerClassifier:
    torch.manual_seed(seed)
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = InferredFramePointerClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        frame_embeddings=frame_embeddings,
    ).to(args.device)
    model.topology_stats = topology_stats
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    observation = to_tensor(train.observation_component, args.device)
    y = to_tensor(train.y, args.device, torch.long)
    frame = to_tensor(train.frame, args.device, torch.long)
    n = len(train.y)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_001)

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            xb = observation.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            fb = frame.index_select(dim=0, index=batch_idx)
            _states, logits, frame_logits = model.rollout_observation(
                xb,
                use_pointer=use_pointer,
                hard_frame=False,
            )
            loss = F.cross_entropy(logits[-1], yb) + INFERRED_FRAME_LOSS_WEIGHT * F.cross_entropy(frame_logits, fb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def train_query_cued_pointer_model(
    *,
    train: RefractionDataBundle,
    frame_embeddings: dict[str, np.ndarray],
    hidden: int,
    args: argparse.Namespace,
    seed: int,
    use_pointer: bool,
) -> QueryCuedPointerClassifier:
    torch.manual_seed(seed)
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = QueryCuedPointerClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
        frame_embeddings=frame_embeddings,
    ).to(args.device)
    model.topology_stats = topology_stats
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    frame_input = to_tensor(train.observation_component, args.device)
    decision_input = to_tensor(query_removed_observation(train), args.device)
    y = to_tensor(train.y, args.device, torch.long)
    frame = to_tensor(train.frame, args.device, torch.long)
    n = len(train.y)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_001)

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            frame_xb = frame_input.index_select(dim=0, index=batch_idx)
            decision_xb = decision_input.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            fb = frame.index_select(dim=0, index=batch_idx)
            _states, logits, frame_logits = model.rollout_inputs(
                frame_xb,
                decision_xb,
                use_pointer=use_pointer,
                hard_frame=False,
            )
            loss = F.cross_entropy(logits[-1], yb) + INFERRED_FRAME_LOSS_WEIGHT * F.cross_entropy(frame_logits, fb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def train_query_bottleneck_direct_model(
    *,
    train: RefractionDataBundle,
    hidden: int,
    bottleneck: int,
    args: argparse.Namespace,
    seed: int,
) -> QueryBottleneckDirectClassifier:
    if train.query_component is None:
        raise ValueError("query bottleneck direct model requires query_component")
    torch.manual_seed(seed)
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = QueryBottleneckDirectClassifier(
        hidden=hidden,
        bottleneck=bottleneck,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
    ).to(args.device)
    model.topology_stats = topology_stats
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    observation = to_tensor(query_removed_observation(train), args.device)
    query = to_tensor(train.query_component, args.device)
    y = to_tensor(train.y, args.device, torch.long)
    n = len(train.y)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_001)

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            xb = observation.index_select(dim=0, index=batch_idx)
            qb = query.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            _states, logits = model.rollout_inputs(xb, qb)
            loss = F.cross_entropy(logits[-1], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def train_reframe_model(
    *,
    train: RefractionDataBundle,
    frame_embeddings: dict[str, np.ndarray],
    hidden: int,
    args: argparse.Namespace,
    seed: int,
    train_reframes: bool,
) -> ReframeRecurrentClassifier:
    torch.manual_seed(seed)
    mask, topology_stats = make_topology_mask(
        hidden=hidden,
        density=args.sparse_density,
        seed=seed + 200_003,
        topology_mode=args.topology_mode,
        flywire_graphml=args.flywire_graphml,
    )
    model = ReframeRecurrentClassifier(
        hidden=hidden,
        steps=args.steps,
        mask=mask,
        update_rate=args.update_rate,
        delta_scale=args.delta_scale,
    ).to(args.device)
    model.topology_stats = topology_stats
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    observation = to_tensor(train.observation_component, args.device)
    frame = to_tensor(train.frame_component, args.device)
    y = to_tensor(train.y, args.device, torch.long)
    frame_ids = to_tensor(train.frame, args.device, torch.long)
    frame_matrix = to_tensor(
        np.stack([frame_embeddings[name] for name in train.frame_names], axis=0),
        args.device,
    )
    n = len(train.y)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed + 300_001)
    rng = np.random.default_rng(seed + 310_001)
    switch_points = [point for point in (1, 2, 3, 4) if point < args.steps]

    for _epoch in range(args.epochs):
        order = torch.randperm(n, generator=order_generator)
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size].to(args.device)
            xb = observation.index_select(dim=0, index=batch_idx)
            fb = frame.index_select(dim=0, index=batch_idx)
            yb = y.index_select(dim=0, index=batch_idx)
            frame_id_b = frame_ids.index_select(dim=0, index=batch_idx)
            frame_schedule = None
            reset_schedule = None

            if train_reframes and switch_points:
                schedule_mode = rng.choice(("fixed", "wrong_no_reset", "wrong_with_reset"), p=(0.40, 0.30, 0.30))
                if schedule_mode != "fixed":
                    switch_after = int(rng.choice(switch_points))
                    wrong_offset = to_tensor(
                        rng.integers(1, len(train.frame_names), size=len(batch_idx)),
                        args.device,
                        torch.long,
                    )
                    wrong_ids = (frame_id_b + wrong_offset) % len(train.frame_names)
                    wrong_frame = frame_matrix.index_select(dim=0, index=wrong_ids)
                    frame_schedule = [
                        wrong_frame if step <= switch_after else fb
                        for step in range(args.steps + 1)
                    ]
                    reset_schedule = [
                        torch.zeros((len(batch_idx), 1), dtype=xb.dtype, device=xb.device)
                        for _step in range(args.steps + 1)
                    ]
                    if schedule_mode == "wrong_with_reset":
                        reset_step = min(switch_after + 1, args.steps)
                        reset_schedule[reset_step] = torch.full(
                            (len(batch_idx), 1),
                            REFRAME_RESET_SCALE,
                            dtype=xb.dtype,
                            device=xb.device,
                        )

            _states, logits = model.rollout_components(
                xb,
                fb,
                frame_schedule=frame_schedule,
                reset_schedule=reset_schedule,
            )
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


def frame_placement_rollout_arrays(
    model: FramePlacementRecurrentClassifier,
    observation: np.ndarray,
    frame: np.ndarray,
    device: str,
    *,
    ablation: str | None = None,
    frame_schedule: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    with torch.no_grad():
        observation_t = to_tensor(observation, device)
        frame_t = to_tensor(frame, device)
        schedule_t = None
        if frame_schedule is not None:
            schedule_t = [to_tensor(step_frame, device) for step_frame in frame_schedule]
        states_t, logits_t = model.rollout_components(
            observation_t,
            frame_t,
            ablation=ablation,
            frame_schedule=schedule_t,
        )
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
    return states, logits


def inferred_rollout_arrays(
    model: InferredFramePointerClassifier,
    observation: np.ndarray,
    device: str,
    *,
    frame_override: np.ndarray | None = None,
    use_pointer: bool = True,
    hard_frame: bool = True,
    ablation: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    model.eval()
    with torch.no_grad():
        observation_t = to_tensor(observation, device)
        frame_override_t = None
        if frame_override is not None:
            frame_override_t = to_tensor(frame_override, device, torch.long)
        states_t, logits_t, frame_logits_t = model.rollout_observation(
            observation_t,
            frame_override=frame_override_t,
            use_pointer=use_pointer,
            hard_frame=hard_frame,
            ablation=ablation,
        )
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
        frame_logits = frame_logits_t.detach().cpu().numpy()
    return states, logits, frame_logits


def query_pointer_rollout_arrays(
    model: QueryCuedPointerClassifier,
    frame_input: np.ndarray,
    decision_input: np.ndarray,
    device: str,
    *,
    frame_override: np.ndarray | None = None,
    use_pointer: bool = True,
    hard_frame: bool = True,
    ablation: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    model.eval()
    with torch.no_grad():
        frame_input_t = to_tensor(frame_input, device)
        decision_input_t = to_tensor(decision_input, device)
        frame_override_t = None
        if frame_override is not None:
            frame_override_t = to_tensor(frame_override, device, torch.long)
        states_t, logits_t, frame_logits_t = model.rollout_inputs(
            frame_input_t,
            decision_input_t,
            frame_override=frame_override_t,
            use_pointer=use_pointer,
            hard_frame=hard_frame,
            ablation=ablation,
        )
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
        frame_logits = frame_logits_t.detach().cpu().numpy()
    return states, logits, frame_logits


def query_bottleneck_rollout_arrays(
    model: QueryBottleneckDirectClassifier,
    observation: np.ndarray,
    query: np.ndarray,
    device: str,
    *,
    ablation: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    with torch.no_grad():
        observation_t = to_tensor(observation, device)
        query_t = to_tensor(query, device)
        states_t, logits_t = model.rollout_inputs(observation_t, query_t, ablation=ablation)
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
    return states, logits


def reframe_rollout_arrays(
    model: ReframeRecurrentClassifier,
    observation: np.ndarray,
    frame: np.ndarray,
    device: str,
    *,
    ablation: str | None = None,
    frame_schedule: list[np.ndarray] | None = None,
    reset_schedule: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    with torch.no_grad():
        observation_t = to_tensor(observation, device)
        frame_t = to_tensor(frame, device)
        frame_schedule_t = None
        reset_schedule_t = None
        if frame_schedule is not None:
            frame_schedule_t = [to_tensor(step_frame, device) for step_frame in frame_schedule]
        if reset_schedule is not None:
            reset_schedule_t = [to_tensor(step_reset, device) for step_reset in reset_schedule]
        states_t, logits_t = model.rollout_components(
            observation_t,
            frame_t,
            ablation=ablation,
            frame_schedule=frame_schedule_t,
            reset_schedule=reset_schedule_t,
        )
        states = [state.detach().cpu().numpy() for state in states_t]
        logits = [logit.detach().cpu().numpy() for logit in logits_t]
    return states, logits


def cosine_distance_mean(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0e-9)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1.0e-9)
    return float(1.0 - np.mean(np.sum(a_norm * b_norm, axis=1)))


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
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
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
    mean_abs_margin_delta_by_step: list[float] = []

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
        ref_margin = np.sort(ref_logits, axis=1)[:, -1] - np.sort(ref_logits, axis=1)[:, -2]
        cf_margin = np.sort(cf_logits, axis=1)[:, -1] - np.sort(cf_logits, axis=1)[:, -2]
        mean_abs_margin_delta_by_step.append(float(np.mean(np.abs(ref_margin - cf_margin))))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "mean_abs_margin_delta": mean_abs_margin_delta_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
        "mean_abs_margin_delta_by_step": mean_abs_margin_delta_by_step,
    }


def inferred_prediction_summary(
    *,
    model: InferredFramePointerClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    observation_override: np.ndarray | None = None,
    frame_override: np.ndarray | None = None,
    use_pointer: bool = True,
    hard_frame: bool = True,
    ablation: str | None = None,
) -> dict[str, Any]:
    observation = bundle.observation_component if observation_override is None else observation_override
    _states, logits, frame_logits = inferred_rollout_arrays(
        model,
        observation,
        args.device,
        frame_override=frame_override,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
        ablation=ablation,
    )
    pred = np.argmax(logits[-1], axis=1)
    frame_pred = np.argmax(frame_logits, axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    frame_prediction_accuracy_by_frame: dict[str, float] = {}
    confusion = np.zeros((len(bundle.frame_names), len(bundle.frame_names)), dtype=np.int64)
    for target, predicted_frame in zip(bundle.frame, frame_pred):
        confusion[int(target), int(predicted_frame)] += 1
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]
        frame_prediction_accuracy_by_frame[frame_name] = float(np.mean(frame_pred[idx] == bundle.frame[idx]))

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "frame_prediction_accuracy": float(np.mean(frame_pred == bundle.frame)),
        "frame_prediction_accuracy_by_frame": frame_prediction_accuracy_by_frame,
        "predicted_frame_confusion_matrix": {
            target_frame: {
                predicted_frame: int(confusion[target_index, predicted_index])
                for predicted_index, predicted_frame in enumerate(bundle.frame_names)
            }
            for target_index, target_frame in enumerate(bundle.frame_names)
        },
        "predicted_frame_distribution": {
            frame_name: float(np.mean(frame_pred == frame_index))
            for frame_index, frame_name in enumerate(bundle.frame_names)
        },
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def query_pointer_prediction_summary(
    *,
    model: QueryCuedPointerClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    frame_input_override: np.ndarray | None = None,
    decision_input_override: np.ndarray | None = None,
    frame_override: np.ndarray | None = None,
    use_pointer: bool = True,
    hard_frame: bool = True,
    ablation: str | None = None,
) -> dict[str, Any]:
    frame_input = bundle.observation_component if frame_input_override is None else frame_input_override
    decision_input = query_removed_observation(bundle) if decision_input_override is None else decision_input_override
    _states, logits, frame_logits = query_pointer_rollout_arrays(
        model,
        frame_input,
        decision_input,
        args.device,
        frame_override=frame_override,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
        ablation=ablation,
    )
    pred = np.argmax(logits[-1], axis=1)
    frame_pred = np.argmax(frame_logits, axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    frame_prediction_accuracy_by_frame: dict[str, float] = {}
    confusion = np.zeros((len(bundle.frame_names), len(bundle.frame_names)), dtype=np.int64)
    for target, predicted_frame in zip(bundle.frame, frame_pred):
        confusion[int(target), int(predicted_frame)] += 1
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]
        frame_prediction_accuracy_by_frame[frame_name] = float(np.mean(frame_pred[idx] == bundle.frame[idx]))

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "frame_prediction_accuracy": float(np.mean(frame_pred == bundle.frame)),
        "frame_prediction_accuracy_by_frame": frame_prediction_accuracy_by_frame,
        "predicted_frame_confusion_matrix": {
            target_frame: {
                predicted_frame: int(confusion[target_index, predicted_index])
                for predicted_index, predicted_frame in enumerate(bundle.frame_names)
            }
            for target_index, target_frame in enumerate(bundle.frame_names)
        },
        "predicted_frame_distribution": {
            frame_name: float(np.mean(frame_pred == frame_index))
            for frame_index, frame_name in enumerate(bundle.frame_names)
        },
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def query_bottleneck_prediction_summary(
    *,
    model: QueryBottleneckDirectClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    observation_override: np.ndarray | None = None,
    query_override: np.ndarray | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    if bundle.query_component is None:
        raise ValueError("query bottleneck prediction requires query_component")
    observation = query_removed_observation(bundle) if observation_override is None else observation_override
    query = bundle.query_component if query_override is None else query_override
    _states, logits = query_bottleneck_rollout_arrays(
        model,
        observation,
        query,
        args.device,
        ablation=ablation,
    )
    pred = np.argmax(logits[-1], axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def inferred_counterfactual_influence_summary(
    *,
    model: InferredFramePointerClassifier,
    reference_x: np.ndarray,
    counterfactual_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
    use_pointer: bool = True,
    hard_frame: bool = True,
) -> dict[str, Any]:
    _reference_states, reference_logits, _reference_frame_logits = inferred_rollout_arrays(
        model,
        reference_x,
        args.device,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
    )
    _counterfactual_states, counterfactual_logits, _counterfactual_frame_logits = inferred_rollout_arrays(
        model,
        counterfactual_x,
        args.device,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
    )

    output_change_rate_by_step: list[float] = []
    target_accuracy_by_step: list[float] = []
    original_label_retention_by_step: list[float] = []
    mean_abs_label_probability_delta_by_step: list[float] = []
    mean_kl_divergence_by_step: list[float] = []
    mean_abs_margin_delta_by_step: list[float] = []

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
        ref_margin = np.sort(ref_logits, axis=1)[:, -1] - np.sort(ref_logits, axis=1)[:, -2]
        cf_margin = np.sort(cf_logits, axis=1)[:, -1] - np.sort(cf_logits, axis=1)[:, -2]
        mean_abs_margin_delta_by_step.append(float(np.mean(np.abs(ref_margin - cf_margin))))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "mean_abs_margin_delta": mean_abs_margin_delta_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
        "mean_abs_margin_delta_by_step": mean_abs_margin_delta_by_step,
    }


def inferred_group_swap_x(
    *,
    bundle: RefractionDataBundle,
    group: str,
    row_idx: np.ndarray,
    permuted_row_idx: np.ndarray,
) -> np.ndarray:
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components[group][row_idx]
    observation += bundle.group_components[group][permuted_row_idx]
    return observation


def run_inferred_refraction_influence(
    *,
    model: InferredFramePointerClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
    use_pointer: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            permuted_idx = idx[rng.permutation(len(idx))]
            counterfactual_x = inferred_group_swap_x(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = inferred_counterfactual_influence_summary(
                model=model,
                reference_x=test.observation_component[idx],
                counterfactual_x=counterfactual_x,
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
                use_pointer=use_pointer,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    authority_by_group: dict[str, float | None] = {}
    for group in bundle_feature_groups(test):
        causal_frames = [
            frame_name
            for frame_name, active_group in (test.active_group_by_frame or MULTI_ASPECT_FRAME_ACTIVE_GROUP).items()
            if active_group == group
        ]
        if not causal_frames:
            authority_by_group[group] = None
            continue
        causal = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in causal_frames
        )
        nuisance = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in test.frame_names
            if frame_name not in causal_frames
        )
        authority_by_group[group] = float(causal - nuisance)

    numeric_authority = [value for value in authority_by_group.values() if value is not None]
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
        "authority_switch_score_by_group": authority_by_group,
        "authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def query_removed_observation(bundle: RefractionDataBundle) -> np.ndarray:
    if bundle.query_component is None:
        raise ValueError("query ablation requires query_component")
    return bundle.observation_component - bundle.query_component


def query_removed_bundle(bundle: RefractionDataBundle) -> RefractionDataBundle:
    stripped = copy.copy(bundle)
    stripped_observation = query_removed_observation(bundle)
    stripped.x = stripped_observation
    stripped.observation_component = stripped_observation
    return stripped


def query_shuffled_observation(bundle: RefractionDataBundle, seed: int) -> np.ndarray:
    if bundle.query_component is None:
        raise ValueError("query shuffle requires query_component")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bundle.y))
    return bundle.observation_component - bundle.query_component + bundle.query_component[perm]


def run_direct_refraction_influence(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            permuted_idx = idx[rng.permutation(len(idx))]
            counterfactual_x = inferred_group_swap_x(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = counterfactual_influence_summary(
                model=model,
                reference_x=test.observation_component[idx],
                counterfactual_x=counterfactual_x,
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    authority_by_group: dict[str, float | None] = {}
    for group in bundle_feature_groups(test):
        causal_frames = [
            frame_name
            for frame_name, active_group in (test.active_group_by_frame or MULTI_ASPECT_FRAME_ACTIVE_GROUP).items()
            if active_group == group
        ]
        if not causal_frames:
            authority_by_group[group] = None
            continue
        causal = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in causal_frames
        )
        nuisance = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in test.frame_names
            if frame_name not in causal_frames
        )
        authority_by_group[group] = float(causal - nuisance)

    numeric_authority = [value for value in authority_by_group.values() if value is not None]
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
        "authority_switch_score_by_group": authority_by_group,
        "authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def query_pointer_counterfactual_influence_summary(
    *,
    model: QueryCuedPointerClassifier,
    reference_frame_input: np.ndarray,
    reference_decision_input: np.ndarray,
    counterfactual_frame_input: np.ndarray,
    counterfactual_decision_input: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
    use_pointer: bool = True,
    hard_frame: bool = True,
) -> dict[str, Any]:
    _reference_states, reference_logits, _reference_frame_logits = query_pointer_rollout_arrays(
        model,
        reference_frame_input,
        reference_decision_input,
        args.device,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
    )
    _counterfactual_states, counterfactual_logits, _counterfactual_frame_logits = query_pointer_rollout_arrays(
        model,
        counterfactual_frame_input,
        counterfactual_decision_input,
        args.device,
        use_pointer=use_pointer,
        hard_frame=hard_frame,
    )
    return logits_counterfactual_summary(
        reference_logits=reference_logits,
        counterfactual_logits=counterfactual_logits,
        reference_y=reference_y,
        counterfactual_y=counterfactual_y,
    )


def query_bottleneck_counterfactual_influence_summary(
    *,
    model: QueryBottleneckDirectClassifier,
    reference_observation: np.ndarray,
    counterfactual_observation: np.ndarray,
    query: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _reference_states, reference_logits = query_bottleneck_rollout_arrays(
        model,
        reference_observation,
        query,
        args.device,
    )
    _counterfactual_states, counterfactual_logits = query_bottleneck_rollout_arrays(
        model,
        counterfactual_observation,
        query,
        args.device,
    )
    return logits_counterfactual_summary(
        reference_logits=reference_logits,
        counterfactual_logits=counterfactual_logits,
        reference_y=reference_y,
        counterfactual_y=counterfactual_y,
    )


def logits_counterfactual_summary(
    *,
    reference_logits: list[np.ndarray],
    counterfactual_logits: list[np.ndarray],
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
) -> dict[str, Any]:
    output_change_rate_by_step: list[float] = []
    target_accuracy_by_step: list[float] = []
    original_label_retention_by_step: list[float] = []
    mean_abs_label_probability_delta_by_step: list[float] = []
    mean_kl_divergence_by_step: list[float] = []
    mean_abs_margin_delta_by_step: list[float] = []

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
        ref_margin = np.sort(ref_logits, axis=1)[:, -1] - np.sort(ref_logits, axis=1)[:, -2]
        cf_margin = np.sort(cf_logits, axis=1)[:, -1] - np.sort(cf_logits, axis=1)[:, -2]
        mean_abs_margin_delta_by_step.append(float(np.mean(np.abs(ref_margin - cf_margin))))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "mean_abs_margin_delta": mean_abs_margin_delta_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
        "mean_abs_margin_delta_by_step": mean_abs_margin_delta_by_step,
    }


def run_query_pointer_refraction_influence(
    *,
    model: QueryCuedPointerClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
    use_pointer: bool = True,
) -> dict[str, Any]:
    if test.query_component is None:
        raise ValueError("query pointer influence requires query_component")
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            permuted_idx = idx[rng.permutation(len(idx))]
            counterfactual_frame_input = inferred_group_swap_x(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_decision_input = counterfactual_frame_input - test.query_component[idx]
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = query_pointer_counterfactual_influence_summary(
                model=model,
                reference_frame_input=test.observation_component[idx],
                reference_decision_input=test.observation_component[idx] - test.query_component[idx],
                counterfactual_frame_input=counterfactual_frame_input,
                counterfactual_decision_input=counterfactual_decision_input,
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
                use_pointer=use_pointer,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    authority_by_group: dict[str, float | None] = {}
    for group in bundle_feature_groups(test):
        causal_frames = [
            frame_name
            for frame_name, active_group in (test.active_group_by_frame or MULTI_ASPECT_FRAME_ACTIVE_GROUP).items()
            if active_group == group
        ]
        if not causal_frames:
            authority_by_group[group] = None
            continue
        causal = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in causal_frames
        )
        nuisance = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in test.frame_names
            if frame_name not in causal_frames
        )
        authority_by_group[group] = float(causal - nuisance)

    numeric_authority = [value for value in authority_by_group.values() if value is not None]
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
        "authority_switch_score_by_group": authority_by_group,
        "authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def run_query_bottleneck_refraction_influence(
    *,
    model: QueryBottleneckDirectClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if test.query_component is None:
        raise ValueError("query bottleneck influence requires query_component")
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            permuted_idx = idx[rng.permutation(len(idx))]
            counterfactual_full = inferred_group_swap_x(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = query_bottleneck_counterfactual_influence_summary(
                model=model,
                reference_observation=test.observation_component[idx] - test.query_component[idx],
                counterfactual_observation=counterfactual_full - test.query_component[idx],
                query=test.query_component[idx],
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    authority_by_group: dict[str, float | None] = {}
    for group in bundle_feature_groups(test):
        causal_frames = [
            frame_name
            for frame_name, active_group in (test.active_group_by_frame or MULTI_ASPECT_FRAME_ACTIVE_GROUP).items()
            if active_group == group
        ]
        if not causal_frames:
            authority_by_group[group] = None
            continue
        causal = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in causal_frames
        )
        nuisance = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in test.frame_names
            if frame_name not in causal_frames
        )
        authority_by_group[group] = float(causal - nuisance)

    numeric_authority = [value for value in authority_by_group.values() if value is not None]
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
        "authority_switch_score_by_group": authority_by_group,
        "authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
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


def clone_with_mask(model: RecurrentClassifier, mask: np.ndarray, topology_label: str) -> RecurrentClassifier:
    clone = copy.deepcopy(model)
    mask = mask.astype(np.float32)
    with torch.no_grad():
        clone.mask.copy_(torch.tensor(mask, dtype=clone.mask.dtype, device=clone.mask.device))
    base_stats = getattr(model, "topology_stats", {})
    target_edge_count = int(base_stats.get("target_edge_count", int(mask.sum())))
    clone.topology_stats = mask_edge_stats(mask, topology_mode=topology_label, target_edge_count=target_edge_count)
    return clone


def degree_preserving_shuffle_mask(mask: np.ndarray, seed: int, swap_multiplier: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = (mask > 0).astype(np.float32).copy()
    np.fill_diagonal(shuffled, 1.0)
    edges = [(int(source), int(target)) for source, target in np.argwhere(shuffled > 0.0) if source != target]
    edge_set = set(edges)
    if len(edges) < 2:
        return shuffled

    attempts = max(1_000, len(edges) * swap_multiplier)
    for _attempt in range(attempts):
        idx_a, idx_b = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[int(idx_a)]
        c, d = edges[int(idx_b)]
        if len({a, b, c, d}) < 4:
            continue
        new_1 = (a, d)
        new_2 = (c, b)
        if new_1[0] == new_1[1] or new_2[0] == new_2[1]:
            continue
        if new_1 in edge_set or new_2 in edge_set:
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(new_1)
        edge_set.add(new_2)
        edges[int(idx_a)] = new_1
        edges[int(idx_b)] = new_2

    shuffled = np.eye(mask.shape[0], dtype=np.float32)
    for source, target in edge_set:
        shuffled[source, target] = 1.0
    return shuffled


def hub_nodes_from_mask(mask: np.ndarray, fraction: float) -> list[int]:
    offdiag = (mask > 0).astype(np.float32).copy()
    np.fill_diagonal(offdiag, 0.0)
    degree = offdiag.sum(axis=0) + offdiag.sum(axis=1)
    count = max(1, int(round(len(degree) * fraction)))
    order = np.argsort(-degree, kind="stable")
    return [int(idx) for idx in order[:count]]


def random_nodes_for_ablation(mask: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(idx) for idx in rng.choice(mask.shape[0], size=count, replace=False)]


def ablate_nodes_in_mask(mask: np.ndarray, nodes: list[int], *, outgoing_only: bool = False) -> np.ndarray:
    ablated = (mask > 0).astype(np.float32).copy()
    for node in nodes:
        if outgoing_only:
            ablated[node, :] = 0.0
        else:
            ablated[node, :] = 0.0
            ablated[:, node] = 0.0
    return ablated


def refraction_authority_metrics(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    main = refraction_prediction_summary(model=model, bundle=test, args=args)
    zero = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="zero_recurrent_update")
    influence = run_refraction_influence(model=model, test=test, args=args, seed=seed)
    refraction_by_step = influence.get("mean_refraction_index_by_step", [])
    return {
        "accuracy": main["accuracy"],
        "zero_recurrent": zero["accuracy"],
        "recurrence_gain": main["accuracy"] - zero["accuracy"],
        "refraction_index_final": refraction_by_step[-1] if refraction_by_step else None,
        "authority_switch_score": influence.get("authority_switch_score"),
        "topology": getattr(model, "topology_stats", {}),
    }


def metric_drop(base: dict[str, Any], perturbed: dict[str, Any], key: str) -> float | None:
    if base.get(key) is None or perturbed.get(key) is None:
        return None
    return float(base[key] - perturbed[key])


def attach_drops(base: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    out = dict(perturbed)
    for key in ("accuracy", "recurrence_gain", "refraction_index_final", "authority_switch_score"):
        out[f"{key}_drop"] = metric_drop(base, perturbed, key)
    return out


def run_hub_topology_diagnostics(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    mask = model.mask.detach().cpu().numpy()
    diagnostics: dict[str, Any] = {
        "enabled": args.topology_mode == "hub_rich",
    }
    if args.topology_mode != "hub_rich":
        return diagnostics

    shuffled_mask = degree_preserving_shuffle_mask(mask, seed + 11)
    shuffled_model = clone_with_mask(model, shuffled_mask, "hub_rich_degree_preserving_shuffle")
    shuffled_metrics = refraction_authority_metrics(
        model=shuffled_model,
        test=test,
        args=args,
        seed=seed + 101,
    )
    diagnostics["degree_preserving_shuffle"] = attach_drops(base_metrics, shuffled_metrics)

    diagnostics["hub_node_ablation"] = {}
    diagnostics["random_node_ablation"] = {}
    diagnostics["hub_outgoing_edge_ablation"] = {}
    for percent in (0.05, 0.10, 0.20):
        label = f"top_{int(percent * 100)}pct"
        hub_nodes = hub_nodes_from_mask(mask, percent)
        random_nodes = random_nodes_for_ablation(mask, len(hub_nodes), seed + int(percent * 10_000) + 17)

        hub_model = clone_with_mask(
            model,
            ablate_nodes_in_mask(mask, hub_nodes),
            f"hub_rich_ablate_{label}",
        )
        random_model = clone_with_mask(
            model,
            ablate_nodes_in_mask(mask, random_nodes),
            f"hub_rich_random_ablate_{label}",
        )
        outgoing_model = clone_with_mask(
            model,
            ablate_nodes_in_mask(mask, hub_nodes, outgoing_only=True),
            f"hub_rich_outgoing_ablate_{label}",
        )

        hub_metrics = refraction_authority_metrics(model=hub_model, test=test, args=args, seed=seed + 201)
        random_metrics = refraction_authority_metrics(model=random_model, test=test, args=args, seed=seed + 301)
        outgoing_metrics = refraction_authority_metrics(model=outgoing_model, test=test, args=args, seed=seed + 401)

        hub_out = attach_drops(base_metrics, hub_metrics)
        random_out = attach_drops(base_metrics, random_metrics)
        outgoing_out = attach_drops(base_metrics, outgoing_metrics)
        hub_out["ablated_node_count"] = len(hub_nodes)
        hub_out["ablated_nodes"] = hub_nodes
        random_out["ablated_node_count"] = len(random_nodes)
        random_out["ablated_nodes"] = random_nodes
        outgoing_out["ablated_node_count"] = len(hub_nodes)
        outgoing_out["ablated_nodes"] = hub_nodes
        hub_out["accuracy_drop_minus_random"] = hub_out.get("accuracy_drop") - random_out.get("accuracy_drop")
        hub_out["authority_drop_minus_random"] = hub_out.get("authority_switch_score_drop") - random_out.get(
            "authority_switch_score_drop"
        )
        diagnostics["hub_node_ablation"][label] = hub_out
        diagnostics["random_node_ablation"][label] = random_out
        diagnostics["hub_outgoing_edge_ablation"][label] = outgoing_out

    return diagnostics


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


def frame_indices(bundle: RefractionDataBundle, frame_name: str) -> np.ndarray:
    frame_id = bundle.frame_names.index(frame_name)
    return np.flatnonzero(bundle.frame == frame_id)


def label_diversity_by_observation(bundle: RefractionDataBundle) -> float:
    diverse = 0
    total = 0
    for base in np.unique(bundle.base_id):
        labels = bundle.y[bundle.base_id == base]
        if len(labels) <= 1:
            continue
        total += 1
        diverse += int(len(set(labels.tolist())) > 1)
    return float(diverse / max(total, 1))


def refraction_prediction_summary(
    *,
    model: RecurrentClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    x_override: np.ndarray | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    x = bundle.x if x_override is None else x_override
    _states, logits = rollout_arrays(model, x, args.device, ablation=ablation)
    pred = np.argmax(logits[-1], axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def frame_placement_prediction_summary(
    *,
    model: FramePlacementRecurrentClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    frame_override: np.ndarray | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    frame = bundle.frame_component if frame_override is None else frame_override
    _states, logits = frame_placement_rollout_arrays(
        model,
        bundle.observation_component,
        frame,
        args.device,
        ablation=ablation,
    )
    pred = np.argmax(logits[-1], axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def reframe_prediction_summary(
    *,
    model: ReframeRecurrentClassifier,
    bundle: RefractionDataBundle,
    args: argparse.Namespace,
    frame_override: np.ndarray | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    frame = bundle.frame_component if frame_override is None else frame_override
    _states, logits = reframe_rollout_arrays(
        model,
        bundle.observation_component,
        frame,
        args.device,
        ablation=ablation,
    )
    pred = np.argmax(logits[-1], axis=1)
    per_step_accuracy = [
        float(np.mean(np.argmax(step_logits, axis=1) == bundle.y))
        for step_logits in logits
    ]
    accuracy_by_frame: dict[str, float] = {}
    accuracy_by_frame_by_step: dict[str, list[float]] = {}
    for frame_name in bundle.frame_names:
        idx = frame_indices(bundle, frame_name)
        accuracy_by_frame[frame_name] = float(np.mean(pred[idx] == bundle.y[idx]))
        accuracy_by_frame_by_step[frame_name] = [
            float(np.mean(np.argmax(step_logits[idx], axis=1) == bundle.y[idx]))
            for step_logits in logits
        ]

    return {
        "accuracy": float(np.mean(pred == bundle.y)),
        "accuracy_by_frame": accuracy_by_frame,
        "accuracy_by_frame_by_step": accuracy_by_frame_by_step,
        "per_step_accuracy": per_step_accuracy,
        "output_entropy_by_step": output_entropy_by_step(logits),
        "logit_margin_by_step": logit_margin_by_step(logits),
        "label_logit_confidence_by_step": [
            float(np.mean(softmax_np(step_logits)[np.arange(len(bundle.y)), bundle.y]))
            for step_logits in logits
        ],
    }


def frame_placement_counterfactual_influence_summary(
    *,
    model: FramePlacementRecurrentClassifier,
    reference_observation: np.ndarray,
    reference_frame: np.ndarray,
    counterfactual_observation: np.ndarray,
    counterfactual_frame: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
    reference_frame_schedule: list[np.ndarray] | None = None,
    counterfactual_frame_schedule: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    _reference_states, reference_logits = frame_placement_rollout_arrays(
        model,
        reference_observation,
        reference_frame,
        args.device,
        frame_schedule=reference_frame_schedule,
    )
    _counterfactual_states, counterfactual_logits = frame_placement_rollout_arrays(
        model,
        counterfactual_observation,
        counterfactual_frame,
        args.device,
        frame_schedule=counterfactual_frame_schedule,
    )

    output_change_rate_by_step: list[float] = []
    target_accuracy_by_step: list[float] = []
    original_label_retention_by_step: list[float] = []
    mean_abs_label_probability_delta_by_step: list[float] = []
    mean_kl_divergence_by_step: list[float] = []
    mean_abs_margin_delta_by_step: list[float] = []

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
        ref_margin = np.sort(ref_logits, axis=1)[:, -1] - np.sort(ref_logits, axis=1)[:, -2]
        cf_margin = np.sort(cf_logits, axis=1)[:, -1] - np.sort(cf_logits, axis=1)[:, -2]
        mean_abs_margin_delta_by_step.append(float(np.mean(np.abs(ref_margin - cf_margin))))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "mean_abs_margin_delta": mean_abs_margin_delta_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
        "mean_abs_margin_delta_by_step": mean_abs_margin_delta_by_step,
    }


def reframe_counterfactual_influence_summary(
    *,
    model: ReframeRecurrentClassifier,
    reference_observation: np.ndarray,
    reference_frame: np.ndarray,
    counterfactual_observation: np.ndarray,
    counterfactual_frame: np.ndarray,
    reference_y: np.ndarray,
    counterfactual_y: np.ndarray,
    args: argparse.Namespace,
    reference_frame_schedule: list[np.ndarray] | None = None,
    counterfactual_frame_schedule: list[np.ndarray] | None = None,
    reference_reset_schedule: list[np.ndarray] | None = None,
    counterfactual_reset_schedule: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    _reference_states, reference_logits = reframe_rollout_arrays(
        model,
        reference_observation,
        reference_frame,
        args.device,
        frame_schedule=reference_frame_schedule,
        reset_schedule=reference_reset_schedule,
    )
    _counterfactual_states, counterfactual_logits = reframe_rollout_arrays(
        model,
        counterfactual_observation,
        counterfactual_frame,
        args.device,
        frame_schedule=counterfactual_frame_schedule,
        reset_schedule=counterfactual_reset_schedule,
    )

    output_change_rate_by_step: list[float] = []
    target_accuracy_by_step: list[float] = []
    original_label_retention_by_step: list[float] = []
    mean_abs_label_probability_delta_by_step: list[float] = []
    mean_kl_divergence_by_step: list[float] = []
    mean_abs_margin_delta_by_step: list[float] = []

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
        ref_margin = np.sort(ref_logits, axis=1)[:, -1] - np.sort(ref_logits, axis=1)[:, -2]
        cf_margin = np.sort(cf_logits, axis=1)[:, -1] - np.sort(cf_logits, axis=1)[:, -2]
        mean_abs_margin_delta_by_step.append(float(np.mean(np.abs(ref_margin - cf_margin))))

    return {
        "label_change_rate": float(np.mean(reference_y != counterfactual_y)),
        "output_change_rate": output_change_rate_by_step[-1],
        "target_accuracy": target_accuracy_by_step[-1],
        "original_label_retention": original_label_retention_by_step[-1],
        "mean_abs_label_probability_delta": mean_abs_label_probability_delta_by_step[-1],
        "mean_kl_divergence": mean_kl_divergence_by_step[-1],
        "mean_abs_margin_delta": mean_abs_margin_delta_by_step[-1],
        "output_change_rate_by_step": output_change_rate_by_step,
        "target_accuracy_by_step": target_accuracy_by_step,
        "original_label_retention_by_step": original_label_retention_by_step,
        "mean_abs_label_probability_delta_by_step": mean_abs_label_probability_delta_by_step,
        "mean_kl_divergence_by_step": mean_kl_divergence_by_step,
        "mean_abs_margin_delta_by_step": mean_abs_margin_delta_by_step,
    }


def evaluate_refraction_model(
    *,
    model: RecurrentClassifier,
    train: RefractionDataBundle,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    ablation: str | None = None,
) -> dict[str, Any]:
    train_states, _train_logits = rollout_arrays(model, train.x, args.device, ablation=ablation)
    test_states, test_logits = rollout_arrays(model, test.x, args.device, ablation=ablation)
    base = refraction_prediction_summary(model=model, bundle=test, args=args, ablation=ablation)

    group_probe_accuracy_by_step: dict[str, list[float]] = {}
    group_probe_accuracy_by_frame_by_step: dict[str, dict[str, list[float]]] = {}
    for group in bundle_feature_groups(test):
        group_probe_accuracy_by_step[group] = [
            ridge_probe_accuracy(
                train_states[step],
                train.group_labels[group],
                test_states[step],
                test.group_labels[group],
                args.ridge,
            )
            for step in range(len(test_states))
        ]
        group_probe_accuracy_by_frame_by_step[group] = {}
        for frame_name in test.frame_names:
            train_idx = frame_indices(train, frame_name)
            test_idx = frame_indices(test, frame_name)
            group_probe_accuracy_by_frame_by_step[group][frame_name] = [
                ridge_probe_accuracy(
                    train_states[step][train_idx],
                    train.group_labels[group][train_idx],
                    test_states[step][test_idx],
                    test.group_labels[group][test_idx],
                    args.ridge,
                )
                for step in range(len(test_states))
            ]

    base.update(
        {
            "feature_group_probe_accuracy_by_step": group_probe_accuracy_by_step,
            "feature_group_probe_accuracy_by_frame_by_step": group_probe_accuracy_by_frame_by_step,
            "same_observation_label_diversity": label_diversity_by_observation(test),
        }
    )
    return base


def refraction_group_swap_x(
    *,
    bundle: RefractionDataBundle,
    group: str,
    row_idx: np.ndarray,
    permuted_row_idx: np.ndarray,
) -> np.ndarray:
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components[group][row_idx]
    observation += bundle.group_components[group][permuted_row_idx]
    return observation + bundle.frame_component[row_idx]


def run_refraction_influence(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            perm_local = rng.permutation(len(idx))
            permuted_idx = idx[perm_local]
            counterfactual_x = refraction_group_swap_x(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = counterfactual_influence_summary(
                model=model,
                reference_x=test.x[idx],
                counterfactual_x=counterfactual_x,
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    authority_by_group: dict[str, float | None] = {}
    for group in bundle_feature_groups(test):
        causal_frames = [
            frame_name
            for frame_name, active_group in (test.active_group_by_frame or FRAME_ACTIVE_GROUP).items()
            if active_group == group
        ]
        if not causal_frames:
            authority_by_group[group] = None
            continue
        causal = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in causal_frames
        )
        nuisance = max(
            influence_by_frame[frame_name][group]["output_change_rate"]
            for frame_name in test.frame_names
            if frame_name not in causal_frames
        )
        authority_by_group[group] = float(causal - nuisance)

    numeric_authority = [value for value in authority_by_group.values() if value is not None]
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
        "authority_switch_score_by_group": authority_by_group,
        "authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def multi_aspect_actor_swap_x(
    *,
    bundle: RefractionDataBundle,
    row_idx: np.ndarray,
    contrast_idx: np.ndarray,
) -> np.ndarray:
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components["actor"][row_idx]
    observation += bundle.group_components["actor"][contrast_idx]
    return observation + bundle.frame_component[row_idx]


def multi_aspect_actor_counterfactual_y(
    *,
    bundle: RefractionDataBundle,
    frame_name: str,
    row_idx: np.ndarray,
    contrast_idx: np.ndarray,
) -> np.ndarray:
    if bundle.tokens is None:
        raise ValueError("multi-aspect token influence requires token metadata")
    y = np.zeros(len(row_idx), dtype=np.int64)
    for out_idx, (row, contrast) in enumerate(zip(row_idx, contrast_idx)):
        y[out_idx] = multi_aspect_label(
            frame_name,
            actor=str(bundle.tokens["actor"][contrast]),
            action=str(bundle.tokens["action"][row]),
            relation=str(bundle.tokens["relation"][row]),
            sound=str(bundle.tokens["sound"][row]),
            place=str(bundle.tokens["place"][row]),
            noise=str(bundle.tokens["noise"][row]),
        )
    return y


def run_multi_aspect_token_influence(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if test.tokens is None:
        raise ValueError("multi-aspect token influence requires token metadata")
    rng = np.random.default_rng(seed)
    actor_influence_by_frame: dict[str, dict[str, Any]] = {}
    authority_switch_by_actor: dict[str, float | None] = {}

    for frame_name in test.frame_names:
        frame_idx = frame_indices(test, frame_name)
        actor_influence_by_frame[frame_name] = {}
        for actor in ACTORS:
            actor_idx = frame_idx[test.tokens["actor"][frame_idx] == actor]
            contrast_pool = frame_idx[test.tokens["actor"][frame_idx] != actor]
            if len(actor_idx) == 0 or len(contrast_pool) == 0:
                actor_influence_by_frame[frame_name][actor] = None
                continue
            contrast_idx = rng.choice(contrast_pool, size=len(actor_idx), replace=True)
            counterfactual_x = multi_aspect_actor_swap_x(
                bundle=test,
                row_idx=actor_idx,
                contrast_idx=contrast_idx,
            )
            counterfactual_y = multi_aspect_actor_counterfactual_y(
                bundle=test,
                frame_name=frame_name,
                row_idx=actor_idx,
                contrast_idx=contrast_idx,
            )
            actor_influence_by_frame[frame_name][actor] = counterfactual_influence_summary(
                model=model,
                reference_x=test.x[actor_idx],
                counterfactual_x=counterfactual_x,
                reference_y=test.y[actor_idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

    causal_frames = ("danger_frame", "friendship_frame", "sound_frame")
    for actor in ACTORS:
        causal_values = [
            actor_influence_by_frame[frame][actor]["output_change_rate"]
            for frame in causal_frames
            if actor_influence_by_frame.get(frame, {}).get(actor) is not None
        ]
        environment_value = actor_influence_by_frame.get("environment_frame", {}).get(actor)
        if not causal_values or environment_value is None:
            authority_switch_by_actor[actor] = None
            continue
        authority_switch_by_actor[actor] = float(np.mean(causal_values) - environment_value["output_change_rate"])

    dog_by_frame = {
        frame_name: actor_influence_by_frame[frame_name].get("dog")
        for frame_name in test.frame_names
    }
    numeric_authority = [value for value in authority_switch_by_actor.values() if value is not None]
    return {
        "actor_token_influence_by_frame": actor_influence_by_frame,
        "dog_influence_by_frame": dog_by_frame,
        "authority_switch_score_by_actor": authority_switch_by_actor,
        "mean_actor_authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def token_component_group(field: str) -> str:
    if field == "actor":
        return "actor"
    if field == "action":
        return "danger_action"
    if field == "relation":
        return "friendship_relation"
    if field == "sound":
        return "sound"
    if field in {"place", "noise"}:
        return "place_noise"
    if field == "light":
        return "light"
    if field == "object":
        return "object"
    raise ValueError(f"unknown token field: {field}")


def inferred_token_swap_x(
    *,
    bundle: RefractionDataBundle,
    field: str,
    row_idx: np.ndarray,
    contrast_idx: np.ndarray,
) -> np.ndarray:
    group = token_component_group(field)
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components[group][row_idx]
    observation += bundle.group_components[group][contrast_idx]
    return observation


def inferred_token_counterfactual_y(
    *,
    bundle: RefractionDataBundle,
    frame_name: str,
    field: str,
    row_idx: np.ndarray,
    contrast_idx: np.ndarray,
) -> np.ndarray:
    if bundle.tokens is None:
        raise ValueError("inferred token inventory requires token metadata")
    y = np.zeros(len(row_idx), dtype=np.int64)
    for out_idx, (row, contrast) in enumerate(zip(row_idx, contrast_idx)):
        actor = str(bundle.tokens["actor"][contrast if field == "actor" else row])
        action = str(bundle.tokens["action"][contrast if field == "action" else row])
        relation = str(bundle.tokens["relation"][contrast if field == "relation" else row])
        sound = str(bundle.tokens["sound"][contrast if field == "sound" else row])
        place = str(bundle.tokens["place"][contrast if field == "place" else row])
        noise = str(bundle.tokens["noise"][contrast if field == "noise" else row])
        y[out_idx] = multi_aspect_label(
            frame_name,
            actor=actor,
            action=action,
            relation=relation,
            sound=sound,
            place=place,
            noise=noise,
        )
    return y


def run_inferred_token_frame_inventory(
    *,
    model: InferredFramePointerClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if test.tokens is None:
        raise ValueError("inferred token inventory requires token metadata")
    rng = np.random.default_rng(seed)
    inventory: dict[str, Any] = {}
    for token_name, field, value in TOKEN_FRAME_INVENTORY_SPECS:
        by_frame: dict[str, Any] = {}
        for frame_name in test.frame_names:
            frame_idx = frame_indices(test, frame_name)
            token_idx = frame_idx[test.tokens[field][frame_idx] == value]
            contrast_pool = frame_idx[test.tokens[field][frame_idx] != value]
            if len(token_idx) == 0 or len(contrast_pool) == 0:
                by_frame[frame_name] = None
                continue
            contrast_idx = rng.choice(contrast_pool, size=len(token_idx), replace=True)
            counterfactual_x = inferred_token_swap_x(
                bundle=test,
                field=field,
                row_idx=token_idx,
                contrast_idx=contrast_idx,
            )
            counterfactual_y = inferred_token_counterfactual_y(
                bundle=test,
                frame_name=frame_name,
                field=field,
                row_idx=token_idx,
                contrast_idx=contrast_idx,
            )
            by_frame[frame_name] = inferred_counterfactual_influence_summary(
                model=model,
                reference_x=test.observation_component[token_idx],
                counterfactual_x=counterfactual_x,
                reference_y=test.y[token_idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )
        output_rates = {
            frame_name: (None if value_by_frame is None else value_by_frame["output_change_rate"])
            for frame_name, value_by_frame in by_frame.items()
        }
        numeric = [value for value in output_rates.values() if value is not None]
        inventory[token_name] = {
            "field": field,
            "value": value,
            "influence_by_frame": by_frame,
            "output_change_rate_by_frame": output_rates,
            "max_output_change_rate": float(max(numeric)) if numeric else None,
            "min_output_change_rate": float(min(numeric)) if numeric else None,
            "authority_span": float(max(numeric) - min(numeric)) if numeric else None,
        }
    return {
        "token_frame_inventory": inventory,
        "dog_influence_by_inferred_frame": inventory.get("dog", {}).get("influence_by_frame", {}),
    }


def run_query_pointer_token_frame_inventory(
    *,
    model: QueryCuedPointerClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if test.tokens is None or test.query_component is None:
        raise ValueError("query pointer token inventory requires token and query metadata")
    rng = np.random.default_rng(seed)
    inventory: dict[str, Any] = {}
    for token_name, field, value in TOKEN_FRAME_INVENTORY_SPECS:
        by_frame: dict[str, Any] = {}
        for frame_name in test.frame_names:
            frame_idx = frame_indices(test, frame_name)
            token_idx = frame_idx[test.tokens[field][frame_idx] == value]
            contrast_pool = frame_idx[test.tokens[field][frame_idx] != value]
            if len(token_idx) == 0 or len(contrast_pool) == 0:
                by_frame[frame_name] = None
                continue
            contrast_idx = rng.choice(contrast_pool, size=len(token_idx), replace=True)
            counterfactual_frame_input = inferred_token_swap_x(
                bundle=test,
                field=field,
                row_idx=token_idx,
                contrast_idx=contrast_idx,
            )
            counterfactual_y = inferred_token_counterfactual_y(
                bundle=test,
                frame_name=frame_name,
                field=field,
                row_idx=token_idx,
                contrast_idx=contrast_idx,
            )
            by_frame[frame_name] = query_pointer_counterfactual_influence_summary(
                model=model,
                reference_frame_input=test.observation_component[token_idx],
                reference_decision_input=test.observation_component[token_idx] - test.query_component[token_idx],
                counterfactual_frame_input=counterfactual_frame_input,
                counterfactual_decision_input=counterfactual_frame_input - test.query_component[token_idx],
                reference_y=test.y[token_idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )
        output_rates = {
            frame_name: (None if value_by_frame is None else value_by_frame["output_change_rate"])
            for frame_name, value_by_frame in by_frame.items()
        }
        numeric = [value for value in output_rates.values() if value is not None]
        inventory[token_name] = {
            "field": field,
            "value": value,
            "influence_by_frame": by_frame,
            "output_change_rate_by_frame": output_rates,
            "max_output_change_rate": float(max(numeric)) if numeric else None,
            "min_output_change_rate": float(min(numeric)) if numeric else None,
            "authority_span": float(max(numeric) - min(numeric)) if numeric else None,
        }
    return {
        "token_frame_inventory": inventory,
        "dog_influence_by_query_implied_frame": inventory.get("dog", {}).get("influence_by_frame", {}),
    }


def reconstruct_multi_aspect_x(
    *,
    bundle: RefractionDataBundle,
    embeddings: FeatureEmbeddings,
    active_value: float,
    pointer_map: dict[str, str | None] | None = None,
    neutral_pointer: bool = False,
    neuron_phases_override: np.ndarray | None = None,
) -> np.ndarray:
    if bundle.tokens is None:
        raise ValueError("reconstructing multi-aspect resonance inputs requires token metadata")
    observation = np.zeros_like(bundle.observation_component)
    for row in range(len(bundle.y)):
        frame_name = bundle.frame_names[int(bundle.frame[row])]
        if neutral_pointer:
            pointer_frame_name = None
        elif pointer_map is not None:
            pointer_frame_name = pointer_map.get(frame_name, frame_name)
        else:
            pointer_frame_name = frame_name
        keys = [
            ("actor", str(bundle.tokens["actor"][row])),
            ("action", str(bundle.tokens["action"][row])),
            ("relation", str(bundle.tokens["relation"][row])),
            ("sound", str(bundle.tokens["sound"][row])),
            ("place", str(bundle.tokens["place"][row])),
            ("noise", str(bundle.tokens["noise"][row])),
            ("object", str(bundle.tokens["object"][row])),
        ]
        observation[row] = resonance_component(
            embeddings=embeddings,
            keys=keys,
            frame_name=frame_name if not neutral_pointer else None,
            pointer_frame_name=pointer_frame_name,
            neuron_phases_override=neuron_phases_override,
            scale=active_value,
        )
    return observation + bundle.frame_component


def pointer_phase_summary(embeddings: FeatureEmbeddings) -> dict[str, Any]:
    if embeddings.pointer_phases is None:
        return {}
    frames = [frame for frame in MULTI_ASPECT_FRAMES if frame in embeddings.pointer_phases]
    distance: dict[str, float] = {}
    for left_index, left_frame in enumerate(frames):
        left = embeddings.pointer_phases[left_frame][None, :]
        for right_frame in frames[left_index + 1 :]:
            right = embeddings.pointer_phases[right_frame][None, :]
            distance[f"{left_frame}_to_{right_frame}"] = cosine_distance_mean(left, right)
    return {
        "pointer_phase_by_frame": {
            frame: [float(value) for value in embeddings.pointer_phases[frame][:8]]
            for frame in frames
        },
        "pointer_distance_between_frames": distance,
        "mean_pointer_distance_between_frames": float(np.mean(list(distance.values()))) if distance else None,
    }


def neuron_phase_specialization_score(embeddings: FeatureEmbeddings) -> float | None:
    if embeddings.neuron_phases is None or embeddings.pointer_phases is None:
        return None
    responses = np.stack(
        [
            np.cos(embeddings.neuron_phases - embeddings.pointer_phases[frame])
            for frame in MULTI_ASPECT_FRAMES
        ],
        axis=0,
    )
    return float(np.mean(np.std(responses, axis=0)))


def resonance_alignment_by_frame(
    *,
    model: RecurrentClassifier,
    bundle: RefractionDataBundle,
    embeddings: FeatureEmbeddings,
    args: argparse.Namespace,
) -> dict[str, float]:
    if embeddings.pointer_phases is None:
        return {}
    states, _logits = rollout_arrays(model, bundle.x, args.device)
    final_state = states[-1]
    out: dict[str, float] = {}
    for frame_name in MULTI_ASPECT_FRAMES:
        idx = frame_indices(bundle, frame_name)
        pointer = np.cos(embeddings.pointer_phases[frame_name] - embeddings.neuron_phases)[None, :]
        pointer = pointer / np.maximum(np.linalg.norm(pointer, axis=1, keepdims=True), 1.0e-9)
        state = final_state[idx] / np.maximum(np.linalg.norm(final_state[idx], axis=1, keepdims=True), 1.0e-9)
        out[frame_name] = float(np.mean(state @ pointer.T))
    return out


def run_pointer_resonance_diagnostics(
    *,
    model: RecurrentClassifier,
    test: RefractionDataBundle,
    embeddings: FeatureEmbeddings,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if embeddings.resonance_mode not in {"pointer_resonance", "pointer_resonance_signed"}:
        return {}
    rng = np.random.default_rng(seed)
    wrong_map = {
        "danger_frame": "environment_frame",
        "environment_frame": "danger_frame",
        "sound_frame": "danger_frame",
        "friendship_frame": "environment_frame",
    }
    shuffled_frames = list(MULTI_ASPECT_FRAMES)
    rng.shuffle(shuffled_frames)
    shuffled_map = dict(zip(MULTI_ASPECT_FRAMES, shuffled_frames))
    randomized_neuron_phases = rng.uniform(0.0, 2.0 * np.pi, size=test.x.shape[1]).astype(np.float32)

    main = refraction_prediction_summary(model=model, bundle=test, args=args)
    wrong_pointer = refraction_prediction_summary(
        model=model,
        bundle=test,
        args=args,
        x_override=reconstruct_multi_aspect_x(
            bundle=test,
            embeddings=embeddings,
            active_value=args.active_value,
            pointer_map=wrong_map,
        ),
    )
    frozen_neutral = refraction_prediction_summary(
        model=model,
        bundle=test,
        args=args,
        x_override=reconstruct_multi_aspect_x(
            bundle=test,
            embeddings=embeddings,
            active_value=args.active_value,
            neutral_pointer=True,
        ),
    )
    shuffled_pointer = refraction_prediction_summary(
        model=model,
        bundle=test,
        args=args,
        x_override=reconstruct_multi_aspect_x(
            bundle=test,
            embeddings=embeddings,
            active_value=args.active_value,
            pointer_map=shuffled_map,
        ),
    )
    randomized_neuron = refraction_prediction_summary(
        model=model,
        bundle=test,
        args=args,
        x_override=reconstruct_multi_aspect_x(
            bundle=test,
            embeddings=embeddings,
            active_value=args.active_value,
            neuron_phases_override=randomized_neuron_phases,
        ),
    )
    phase = pointer_phase_summary(embeddings)
    return {
        **phase,
        "neuron_phase_specialization_score": neuron_phase_specialization_score(embeddings),
        "resonance_alignment_by_frame": resonance_alignment_by_frame(
            model=model,
            bundle=test,
            embeddings=embeddings,
            args=args,
        ),
        "wrong_pointer": wrong_pointer,
        "wrong_pointer_inference_drop": main["accuracy"] - wrong_pointer["accuracy"],
        "frozen_neutral_pointer": frozen_neutral,
        "frozen_neutral_pointer_drop": main["accuracy"] - frozen_neutral["accuracy"],
        "shuffled_pointer_frame_mapping": shuffled_pointer,
        "shuffled_pointer_frame_mapping_drop": main["accuracy"] - shuffled_pointer["accuracy"],
        "randomized_neuron_phases": randomized_neuron,
        "randomized_neuron_phase_drop": main["accuracy"] - randomized_neuron["accuracy"],
        "pointer_map_used_for_wrong_pointer": wrong_map,
        "pointer_map_used_for_shuffle": shuffled_map,
    }


def shuffled_frame_component(bundle: RefractionDataBundle, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bundle.frame_component))
    return bundle.frame_component[perm]


def refraction_group_swap_observation(
    *,
    bundle: RefractionDataBundle,
    group: str,
    row_idx: np.ndarray,
    permuted_row_idx: np.ndarray,
) -> np.ndarray:
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components[group][row_idx]
    observation += bundle.group_components[group][permuted_row_idx]
    return observation


def multi_aspect_actor_swap_observation(
    *,
    bundle: RefractionDataBundle,
    row_idx: np.ndarray,
    contrast_idx: np.ndarray,
) -> np.ndarray:
    observation = bundle.observation_component[row_idx].copy()
    observation -= bundle.group_components["actor"][row_idx]
    observation += bundle.group_components["actor"][contrast_idx]
    return observation


def run_frame_placement_influence(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    influence_by_frame: dict[str, dict[str, Any]] = {}
    active_core_influence_by_step: dict[str, list[float]] = {}
    inactive_group_influence_by_step: dict[str, list[float]] = {}
    refraction_index_by_step: dict[str, list[float]] = {}

    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        active_group = bundle_active_group(test, frame_name)
        influence_by_frame[frame_name] = {}
        for group in bundle_feature_groups(test):
            permuted_idx = idx[rng.permutation(len(idx))]
            counterfactual_observation = refraction_group_swap_observation(
                bundle=test,
                group=group,
                row_idx=idx,
                permuted_row_idx=permuted_idx,
            )
            counterfactual_y = (
                test.group_labels[group][permuted_idx]
                if group == active_group
                else test.y[idx]
            )
            influence_by_frame[frame_name][group] = frame_placement_counterfactual_influence_summary(
                model=model,
                reference_observation=test.observation_component[idx],
                reference_frame=test.frame_component[idx],
                counterfactual_observation=counterfactual_observation,
                counterfactual_frame=test.frame_component[idx],
                reference_y=test.y[idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

        active_curve = influence_by_frame[frame_name][active_group]["output_change_rate_by_step"]
        inactive_curves = [
            influence_by_frame[frame_name][group]["output_change_rate_by_step"]
            for group in bundle_feature_groups(test)
            if group != active_group
        ]
        inactive_max = [
            float(max(curve[step] for curve in inactive_curves))
            for step in range(len(active_curve))
        ]
        active_core_influence_by_step[frame_name] = active_curve
        inactive_group_influence_by_step[frame_name] = inactive_max
        refraction_index_by_step[frame_name] = [
            float(active - inactive)
            for active, inactive in zip(active_curve, inactive_max)
        ]

    mean_refraction_index = mean_list(list(refraction_index_by_step.values()))
    return {
        "feature_group_influence_by_frame": influence_by_frame,
        "active_core_influence_by_step": active_core_influence_by_step,
        "inactive_group_influence_by_step": inactive_group_influence_by_step,
        "refraction_index_by_step": refraction_index_by_step,
        "mean_refraction_index_by_step": mean_refraction_index,
    }


def run_frame_placement_token_influence(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if test.tokens is None:
        raise ValueError("multi-aspect token influence requires token metadata")
    rng = np.random.default_rng(seed)
    dog_by_frame: dict[str, Any] = {}
    authority_switch_by_actor: dict[str, float | None] = {}
    actor_influence_by_frame: dict[str, dict[str, Any]] = {}

    for frame_name in test.frame_names:
        frame_idx = frame_indices(test, frame_name)
        actor_influence_by_frame[frame_name] = {}
        for actor in ACTORS:
            actor_idx = frame_idx[test.tokens["actor"][frame_idx] == actor]
            contrast_pool = frame_idx[test.tokens["actor"][frame_idx] != actor]
            if len(actor_idx) == 0 or len(contrast_pool) == 0:
                actor_influence_by_frame[frame_name][actor] = None
                continue
            contrast_idx = rng.choice(contrast_pool, size=len(actor_idx), replace=True)
            counterfactual_observation = multi_aspect_actor_swap_observation(
                bundle=test,
                row_idx=actor_idx,
                contrast_idx=contrast_idx,
            )
            counterfactual_y = multi_aspect_actor_counterfactual_y(
                bundle=test,
                frame_name=frame_name,
                row_idx=actor_idx,
                contrast_idx=contrast_idx,
            )
            actor_influence_by_frame[frame_name][actor] = frame_placement_counterfactual_influence_summary(
                model=model,
                reference_observation=test.observation_component[actor_idx],
                reference_frame=test.frame_component[actor_idx],
                counterfactual_observation=counterfactual_observation,
                counterfactual_frame=test.frame_component[actor_idx],
                reference_y=test.y[actor_idx],
                counterfactual_y=counterfactual_y,
                args=args,
            )

    for actor in ACTORS:
        causal_values = [
            actor_influence_by_frame[frame][actor]["output_change_rate"]
            for frame in ("danger_frame", "friendship_frame", "sound_frame")
            if actor_influence_by_frame.get(frame, {}).get(actor) is not None
        ]
        environment_value = actor_influence_by_frame.get("environment_frame", {}).get(actor)
        authority_switch_by_actor[actor] = (
            float(np.mean(causal_values) - environment_value["output_change_rate"])
            if causal_values and environment_value is not None
            else None
        )
    dog_by_frame = {
        frame_name: actor_influence_by_frame[frame_name].get("dog")
        for frame_name in test.frame_names
    }
    numeric_authority = [value for value in authority_switch_by_actor.values() if value is not None]
    return {
        "actor_token_influence_by_frame": actor_influence_by_frame,
        "dog_influence_by_frame": dog_by_frame,
        "authority_switch_score_by_actor": authority_switch_by_actor,
        "mean_actor_authority_switch_score": float(np.mean(numeric_authority)) if numeric_authority else None,
    }


def aligned_frame_rows(bundle: RefractionDataBundle, source_frame: str, target_frame: str) -> tuple[np.ndarray, np.ndarray]:
    source_idx = frame_indices(bundle, source_frame)
    target_idx = frame_indices(bundle, target_frame)
    source_by_base = {int(bundle.base_id[row]): int(row) for row in source_idx}
    target_by_base = {int(bundle.base_id[row]): int(row) for row in target_idx}
    bases = sorted(set(source_by_base) & set(target_by_base))
    return (
        np.array([source_by_base[base] for base in bases], dtype=np.int64),
        np.array([target_by_base[base] for base in bases], dtype=np.int64),
    )


def frame_switch_schedule(
    *,
    source_frame_component: np.ndarray,
    target_frame_component: np.ndarray,
    switch_after: int,
    steps: int,
) -> list[np.ndarray]:
    return [
        source_frame_component if step <= switch_after else target_frame_component
        for step in range(steps + 1)
    ]


def frame_switch_prediction_summary(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    source_frame: str,
    target_frame: str,
    switch_after: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_rows, target_rows = aligned_frame_rows(test, source_frame, target_frame)
    source_frame_component = test.frame_component[source_rows]
    target_frame_component = test.frame_component[target_rows]
    schedule = frame_switch_schedule(
        source_frame_component=source_frame_component,
        target_frame_component=target_frame_component,
        switch_after=switch_after,
        steps=args.steps,
    )

    source_states, source_logits = frame_placement_rollout_arrays(
        model,
        test.observation_component[source_rows],
        source_frame_component,
        args.device,
    )
    target_states, target_logits = frame_placement_rollout_arrays(
        model,
        test.observation_component[target_rows],
        target_frame_component,
        args.device,
    )
    switch_states, switch_logits = frame_placement_rollout_arrays(
        model,
        test.observation_component[source_rows],
        source_frame_component,
        args.device,
        frame_schedule=schedule,
    )

    current_y_by_step = [
        test.y[source_rows] if step <= switch_after else test.y[target_rows]
        for step in range(args.steps + 1)
    ]
    current_accuracy_by_step = [
        float(np.mean(np.argmax(logits, axis=1) == current_y_by_step[step]))
        for step, logits in enumerate(switch_logits)
    ]
    target_accuracy_by_step = [
        float(np.mean(np.argmax(logits, axis=1) == test.y[target_rows]))
        for logits in switch_logits
    ]
    source_distance_by_step = [
        cosine_distance_mean(switch_states[step], source_states[step])
        for step in range(args.steps + 1)
    ]
    target_distance_by_step = [
        cosine_distance_mean(switch_states[step], target_states[step])
        for step in range(args.steps + 1)
    ]

    source_active_group = bundle_active_group(test, source_frame)
    target_active_group = bundle_active_group(test, target_frame)
    influence_after_switch: dict[str, Any] = {}
    for group in sorted({source_active_group, target_active_group}):
        permuted_rows = source_rows[rng.permutation(len(source_rows))]
        counterfactual_observation = refraction_group_swap_observation(
            bundle=test,
            group=group,
            row_idx=source_rows,
            permuted_row_idx=permuted_rows,
        )
        counterfactual_y = (
            test.group_labels[group][permuted_rows]
            if group == target_active_group
            else test.y[target_rows]
        )
        influence_after_switch[group] = frame_placement_counterfactual_influence_summary(
            model=model,
            reference_observation=test.observation_component[source_rows],
            reference_frame=source_frame_component,
            counterfactual_observation=counterfactual_observation,
            counterfactual_frame=source_frame_component,
            reference_y=test.y[target_rows],
            counterfactual_y=counterfactual_y,
            args=args,
            reference_frame_schedule=schedule,
            counterfactual_frame_schedule=schedule,
        )

    old_influence = influence_after_switch[source_active_group]["output_change_rate"]
    new_influence = influence_after_switch[target_active_group]["output_change_rate"]
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "switch_after": switch_after,
        "current_frame_accuracy_by_step": current_accuracy_by_step,
        "target_frame_accuracy_by_step": target_accuracy_by_step,
        "final_target_accuracy": target_accuracy_by_step[-1],
        "output_entropy_by_step": output_entropy_by_step(switch_logits),
        "logit_margin_by_step": logit_margin_by_step(switch_logits),
        "hidden_cosine_distance_to_source_by_step": source_distance_by_step,
        "hidden_cosine_distance_to_target_by_step": target_distance_by_step,
        "target_frame_convergence_after_switch": float(target_distance_by_step[switch_after] - target_distance_by_step[-1]),
        "source_frame_departure_after_switch": float(source_distance_by_step[-1] - source_distance_by_step[switch_after]),
        "reorientation_score": float(source_distance_by_step[-1] - target_distance_by_step[-1]),
        "old_active_group": source_active_group,
        "new_active_group": target_active_group,
        "old_active_group_output_change": old_influence,
        "new_active_group_output_change": new_influence,
        "authority_switch_after_frame_switch": float(new_influence - old_influence),
        "influence_after_switch": influence_after_switch,
        "direct_source_final_accuracy": float(np.mean(np.argmax(source_logits[-1], axis=1) == test.y[source_rows])),
        "direct_target_final_accuracy": float(np.mean(np.argmax(target_logits[-1], axis=1) == test.y[target_rows])),
    }


def run_mid_run_switch_diagnostics(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    switch_points = [point for point in (1, 2, 3, 4) if point < args.steps]
    by_pair: dict[str, dict[str, Any]] = {}
    flat: list[dict[str, Any]] = []
    for pair_index, (source_frame, target_frame) in enumerate(FRAME_SWITCH_PAIRS):
        pair_key = f"{source_frame}_to_{target_frame}"
        by_pair[pair_key] = {}
        for switch_after in switch_points:
            summary = frame_switch_prediction_summary(
                model=model,
                test=test,
                source_frame=source_frame,
                target_frame=target_frame,
                switch_after=switch_after,
                args=args,
                seed=seed + 10_000 * pair_index + switch_after,
            )
            by_pair[pair_key][f"switch_after_{switch_after}"] = summary
            flat.append(summary)

    successes = [
        item["final_target_accuracy"] >= 0.75
        and item["hidden_cosine_distance_to_target_by_step"][-1]
        <= item["hidden_cosine_distance_to_source_by_step"][-1]
        for item in flat
    ]
    return {
        "by_pair": by_pair,
        "mid_run_switch_success_rate": float(np.mean(successes)) if successes else None,
        "reorientation_score": float(np.mean([item["reorientation_score"] for item in flat])) if flat else None,
        "target_frame_convergence_after_switch": float(
            np.mean([item["target_frame_convergence_after_switch"] for item in flat])
        ) if flat else None,
        "authority_switch_after_frame_switch": float(
            np.mean([item["authority_switch_after_frame_switch"] for item in flat])
        ) if flat else None,
    }


def run_hidden_trajectory_geometry(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    states_by_frame: dict[str, list[np.ndarray]] = {}
    for frame_name in test.frame_names:
        idx = frame_indices(test, frame_name)
        states, _logits = frame_placement_rollout_arrays(
            model,
            test.observation_component[idx],
            test.frame_component[idx],
            args.device,
        )
        states_by_frame[frame_name] = states

    centroid_by_step: list[dict[str, np.ndarray]] = []
    trajectory_divergence_by_step: list[float] = []
    frame_clustering_accuracy_by_step: list[float] = []
    for step in range(args.steps + 1):
        centroids = {
            frame_name: states_by_frame[frame_name][step].mean(axis=0)
            for frame_name in test.frame_names
        }
        centroid_by_step.append(centroids)
        distances = []
        for left_index, left_frame in enumerate(test.frame_names):
            for right_frame in test.frame_names[left_index + 1 :]:
                distances.append(
                    cosine_distance_mean(
                        centroids[left_frame][None, :],
                        centroids[right_frame][None, :],
                    )
                )
        trajectory_divergence_by_step.append(float(np.mean(distances)))

        rows = []
        labels = []
        for frame_id, frame_name in enumerate(test.frame_names):
            rows.append(states_by_frame[frame_name][step])
            labels.extend([frame_id] * len(states_by_frame[frame_name][step]))
        data = np.concatenate(rows, axis=0)
        labels_arr = np.array(labels, dtype=np.int64)
        centroid_matrix = np.stack([centroids[frame_name] for frame_name in test.frame_names], axis=0)
        data_norm = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1.0e-9)
        centroid_norm = centroid_matrix / np.maximum(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1.0e-9)
        pred = np.argmax(data_norm @ centroid_norm.T, axis=1)
        frame_clustering_accuracy_by_step.append(float(np.mean(pred == labels_arr)))

    return {
        "trajectory_divergence_by_step": trajectory_divergence_by_step,
        "final_frame_clustering_accuracy": frame_clustering_accuracy_by_step[-1],
        "frame_clustering_accuracy_by_step": frame_clustering_accuracy_by_step,
    }


def run_soft_frame_interpolation(
    *,
    model: FramePlacementRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    danger_frame, environment_frame = SOFT_FRAME_INTERPOLATION_PAIR
    danger_rows, environment_rows = aligned_frame_rows(test, danger_frame, environment_frame)
    danger_frame_component = test.frame_component[danger_rows]
    environment_frame_component = test.frame_component[environment_rows]
    danger_states, _danger_logits = frame_placement_rollout_arrays(
        model,
        test.observation_component[danger_rows],
        danger_frame_component,
        args.device,
    )
    environment_states, _environment_logits = frame_placement_rollout_arrays(
        model,
        test.observation_component[environment_rows],
        environment_frame_component,
        args.device,
    )

    mixes: dict[str, Any] = {}
    balances = []
    env_weights = []
    for danger_weight in SOFT_FRAME_MIXES:
        environment_weight = 1.0 - danger_weight
        mixed_frame = danger_weight * danger_frame_component + environment_weight * environment_frame_component
        states, logits = frame_placement_rollout_arrays(
            model,
            test.observation_component[danger_rows],
            mixed_frame,
            args.device,
        )
        probs = [softmax_np(step_logits) for step_logits in logits]
        group_influence: dict[str, Any] = {}
        for group in ("actor_action", "place_noise"):
            permuted_rows = danger_rows[rng.permutation(len(danger_rows))]
            counterfactual_observation = refraction_group_swap_observation(
                bundle=test,
                group=group,
                row_idx=danger_rows,
                permuted_row_idx=permuted_rows,
            )
            group_influence[group] = frame_placement_counterfactual_influence_summary(
                model=model,
                reference_observation=test.observation_component[danger_rows],
                reference_frame=mixed_frame,
                counterfactual_observation=counterfactual_observation,
                counterfactual_frame=mixed_frame,
                reference_y=test.y[danger_rows],
                counterfactual_y=test.y[danger_rows],
                args=args,
            )
        danger_distance = cosine_distance_mean(states[-1], danger_states[-1])
        environment_distance = cosine_distance_mean(states[-1], environment_states[-1])
        env_position = danger_distance / max(danger_distance + environment_distance, 1.0e-9)
        balance = (
            group_influence["place_noise"]["output_change_rate"]
            - group_influence["actor_action"]["output_change_rate"]
        )
        balances.append(balance)
        env_weights.append(environment_weight)
        mixes[f"danger_{danger_weight:.2f}_environment_{environment_weight:.2f}"] = {
            "danger_weight": danger_weight,
            "environment_weight": environment_weight,
            "mean_class_1_probability_by_step": [float(np.mean(step_probs[:, 1])) for step_probs in probs],
            "actor_action_output_change": group_influence["actor_action"]["output_change_rate"],
            "place_noise_output_change": group_influence["place_noise"]["output_change_rate"],
            "influence_balance_place_noise_minus_actor_action": balance,
            "hidden_position_between_danger_and_environment": env_position,
            "distance_to_danger_direct_final": danger_distance,
            "distance_to_environment_direct_final": environment_distance,
        }

    if len(set(round(value, 8) for value in balances)) <= 1:
        smoothness = 0.0
    else:
        smoothness = float(np.corrcoef(np.array(env_weights), np.array(balances))[0, 1])
    return {
        "pair": {
            "source": danger_frame,
            "target": environment_frame,
        },
        "mixes": mixes,
        "interpolation_smoothness_score": smoothness,
    }


def reframe_reset_schedule(
    *,
    rows: int,
    switch_after: int,
    steps: int,
    enabled: bool,
) -> list[np.ndarray]:
    schedule = [
        np.zeros((rows, 1), dtype=np.float32)
        for _step in range(steps + 1)
    ]
    if enabled:
        schedule[min(switch_after + 1, steps)] = np.full((rows, 1), REFRAME_RESET_SCALE, dtype=np.float32)
    return schedule


def reframe_switch_prediction_summary(
    *,
    model: ReframeRecurrentClassifier,
    test: RefractionDataBundle,
    source_frame: str,
    target_frame: str,
    switch_after: int,
    use_reset: bool,
    args: argparse.Namespace,
    seed: int,
    ablation: str | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_rows, target_rows = aligned_frame_rows(test, source_frame, target_frame)
    source_frame_component = test.frame_component[source_rows]
    target_frame_component = test.frame_component[target_rows]
    frame_schedule = frame_switch_schedule(
        source_frame_component=source_frame_component,
        target_frame_component=target_frame_component,
        switch_after=switch_after,
        steps=args.steps,
    )
    reset_schedule = reframe_reset_schedule(
        rows=len(source_rows),
        switch_after=switch_after,
        steps=args.steps,
        enabled=use_reset,
    )

    source_states, source_logits = reframe_rollout_arrays(
        model,
        test.observation_component[source_rows],
        source_frame_component,
        args.device,
        ablation=ablation,
    )
    target_states, target_logits = reframe_rollout_arrays(
        model,
        test.observation_component[target_rows],
        target_frame_component,
        args.device,
        ablation=ablation,
    )
    switch_states, switch_logits = reframe_rollout_arrays(
        model,
        test.observation_component[source_rows],
        source_frame_component,
        args.device,
        ablation=ablation,
        frame_schedule=frame_schedule,
        reset_schedule=reset_schedule,
    )

    current_y_by_step = [
        test.y[source_rows] if step <= switch_after else test.y[target_rows]
        for step in range(args.steps + 1)
    ]
    current_accuracy_by_step = [
        float(np.mean(np.argmax(logits, axis=1) == current_y_by_step[step]))
        for step, logits in enumerate(switch_logits)
    ]
    target_accuracy_by_step = [
        float(np.mean(np.argmax(logits, axis=1) == test.y[target_rows]))
        for logits in switch_logits
    ]
    source_distance_by_step = [
        cosine_distance_mean(switch_states[step], source_states[step])
        for step in range(args.steps + 1)
    ]
    target_distance_by_step = [
        cosine_distance_mean(switch_states[step], target_states[step])
        for step in range(args.steps + 1)
    ]

    source_active_group = bundle_active_group(test, source_frame)
    target_active_group = bundle_active_group(test, target_frame)
    influence_after_switch: dict[str, Any] = {}
    for group in sorted({source_active_group, target_active_group}):
        permuted_rows = source_rows[rng.permutation(len(source_rows))]
        counterfactual_observation = refraction_group_swap_observation(
            bundle=test,
            group=group,
            row_idx=source_rows,
            permuted_row_idx=permuted_rows,
        )
        counterfactual_y = (
            test.group_labels[group][permuted_rows]
            if group == target_active_group
            else test.y[target_rows]
        )
        influence_after_switch[group] = reframe_counterfactual_influence_summary(
            model=model,
            reference_observation=test.observation_component[source_rows],
            reference_frame=source_frame_component,
            counterfactual_observation=counterfactual_observation,
            counterfactual_frame=source_frame_component,
            reference_y=test.y[target_rows],
            counterfactual_y=counterfactual_y,
            args=args,
            reference_frame_schedule=frame_schedule,
            counterfactual_frame_schedule=frame_schedule,
            reference_reset_schedule=reset_schedule,
            counterfactual_reset_schedule=reset_schedule,
        )

    old_curve = influence_after_switch[source_active_group]["output_change_rate_by_step"]
    new_curve = influence_after_switch[target_active_group]["output_change_rate_by_step"]
    reset_step = min(switch_after + 1, args.steps)
    entropy = output_entropy_by_step(switch_logits)
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "switch_after": switch_after,
        "use_reset": use_reset,
        "current_frame_accuracy_by_step": current_accuracy_by_step,
        "target_frame_accuracy_by_step": target_accuracy_by_step,
        "final_target_accuracy": target_accuracy_by_step[-1],
        "output_entropy_by_step": entropy,
        "logit_margin_by_step": logit_margin_by_step(switch_logits),
        "reset_entropy_spike": float(entropy[reset_step] - entropy[max(reset_step - 1, 0)]) if use_reset else None,
        "hidden_cosine_distance_to_source_by_step": source_distance_by_step,
        "hidden_cosine_distance_to_target_by_step": target_distance_by_step,
        "target_frame_convergence_after_switch": float(target_distance_by_step[switch_after] - target_distance_by_step[-1]),
        "source_frame_departure_after_switch": float(source_distance_by_step[-1] - source_distance_by_step[switch_after]),
        "reorientation_score": float(source_distance_by_step[-1] - target_distance_by_step[-1]),
        "old_active_group": source_active_group,
        "new_active_group": target_active_group,
        "old_active_group_output_change": old_curve[-1],
        "new_active_group_output_change": new_curve[-1],
        "old_frame_authority_decay": float(old_curve[switch_after] - old_curve[-1]),
        "new_frame_authority_rise": float(new_curve[-1] - new_curve[switch_after]),
        "authority_switch_after_frame_switch": float(new_curve[-1] - old_curve[-1]),
        "old_active_group_output_change_by_step": old_curve,
        "new_active_group_output_change_by_step": new_curve,
        "influence_after_switch": influence_after_switch,
        "direct_source_final_accuracy": float(np.mean(np.argmax(source_logits[-1], axis=1) == test.y[source_rows])),
        "direct_target_final_accuracy": float(np.mean(np.argmax(target_logits[-1], axis=1) == test.y[target_rows])),
    }


def run_reframe_recovery_diagnostics(
    *,
    model: ReframeRecurrentClassifier,
    test: RefractionDataBundle,
    args: argparse.Namespace,
    seed: int,
    use_reset: bool,
    ablation: str | None = None,
) -> dict[str, Any]:
    switch_points = [point for point in (1, 2, 3, 4) if point < args.steps]
    by_pair: dict[str, dict[str, Any]] = {}
    flat: list[dict[str, Any]] = []
    for pair_index, (source_frame, target_frame) in enumerate(FRAME_SWITCH_PAIRS):
        pair_key = f"{source_frame}_to_{target_frame}"
        by_pair[pair_key] = {}
        for switch_after in switch_points:
            summary = reframe_switch_prediction_summary(
                model=model,
                test=test,
                source_frame=source_frame,
                target_frame=target_frame,
                switch_after=switch_after,
                use_reset=use_reset,
                args=args,
                seed=seed + 10_000 * pair_index + switch_after,
                ablation=ablation,
            )
            by_pair[pair_key][f"switch_after_{switch_after}"] = summary
            flat.append(summary)

    recovery_by_switch_step: dict[str, float] = {}
    for switch_after in switch_points:
        matching = [item["final_target_accuracy"] for item in flat if item["switch_after"] == switch_after]
        recovery_by_switch_step[f"switch_after_{switch_after}"] = float(np.mean(matching)) if matching else None

    successes = [
        item["final_target_accuracy"] >= 0.75
        and item["authority_switch_after_frame_switch"] > 0.0
        and item["hidden_cosine_distance_to_target_by_step"][-1] <= item["hidden_cosine_distance_to_source_by_step"][-1]
        for item in flat
    ]
    return {
        "use_reset": use_reset,
        "by_pair": by_pair,
        "final_accuracy_after_reframe": float(np.mean([item["final_target_accuracy"] for item in flat])) if flat else None,
        "reframe_success_rate": float(np.mean(successes)) if successes else None,
        "recovery_by_switch_step": recovery_by_switch_step,
        "old_frame_authority_decay": float(np.mean([item["old_frame_authority_decay"] for item in flat])) if flat else None,
        "new_frame_authority_rise": float(np.mean([item["new_frame_authority_rise"] for item in flat])) if flat else None,
        "authority_switch_after_frame_switch": float(
            np.mean([item["authority_switch_after_frame_switch"] for item in flat])
        ) if flat else None,
        "reorientation_score": float(np.mean([item["reorientation_score"] for item in flat])) if flat else None,
        "target_frame_convergence_after_switch": float(
            np.mean([item["target_frame_convergence_after_switch"] for item in flat])
        ) if flat else None,
        "reset_entropy_spike": float(
            np.mean([item["reset_entropy_spike"] for item in flat if item["reset_entropy_spike"] is not None])
        ) if use_reset and flat else None,
    }


def shuffled_frame_x(bundle: RefractionDataBundle, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bundle.x))
    return bundle.observation_component + bundle.frame_component[perm]


def run_refraction_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
    full_controls: bool,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_frame_embeddings(schema.hidden, seed + 600_007, args.frame_scale)
    train = make_refraction_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 2_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    test = make_refraction_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 2_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 41)
    main = evaluate_refraction_model(model=model, train=train, test=test, args=args)
    zero = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="zero_recurrent_update")
    threshold_only = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="zero_matrix_keep_threshold")
    controls: dict[str, Any] = {
        "zero_recurrent_update": zero,
        "threshold_only": threshold_only,
    }
    influence: dict[str, Any] = {}
    hub_diagnostics: dict[str, Any] = {"enabled": False}

    if full_controls:
        influence = run_refraction_influence(model=model, test=test, args=args, seed=seed + 910_001)
        base_metrics = {
            "accuracy": main["accuracy"],
            "zero_recurrent": zero["accuracy"],
            "recurrence_gain": main["accuracy"] - zero["accuracy"],
            "refraction_index_final": (
                influence.get("mean_refraction_index_by_step", [None])[-1]
                if influence.get("mean_refraction_index_by_step")
                else None
            ),
            "authority_switch_score": influence.get("authority_switch_score"),
            "topology": getattr(model, "topology_stats", {}),
        }
        hub_diagnostics = run_hub_topology_diagnostics(
            model=model,
            test=test,
            args=args,
            seed=seed + 915_001,
            base_metrics=base_metrics,
        )
        controls["freeze_after_1"] = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_1")
        controls["freeze_after_2"] = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_2")
        controls["freeze_after_3"] = refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_3")
        controls["shuffled_task_frame_token"] = refraction_prediction_summary(
            model=model,
            bundle=test,
            args=args,
            x_override=shuffled_frame_x(test, seed + 920_001),
        )

        no_frame_train = make_refraction_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 2_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            no_frame_token=True,
        )
        no_frame_test = make_refraction_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 2_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            no_frame_token=True,
        )
        no_frame_model = train_model(train=no_frame_train, hidden=schema.hidden, args=args, seed=seed + 42)
        controls["no_task_frame_token"] = refraction_prediction_summary(
            model=no_frame_model,
            bundle=no_frame_test,
            args=args,
        )

        random_model = recurrent_matrix_control(model, "randomize", seed + 930_001)
        controls["randomize_recurrent_matrix"] = refraction_prediction_summary(
            model=random_model,
            bundle=test,
            args=args,
        )

        if args.random_label_control:
            random_train = make_refraction_dataset(
                n=args.train_size,
                combos=train_combos,
                seed=seed + 3_001,
                embeddings=embeddings,
                frame_embeddings=frame_embeddings,
                active_value=args.active_value,
                random_labels=True,
            )
            random_test = make_refraction_dataset(
                n=args.test_size,
                combos=heldout_combos,
                seed=seed + 3_002,
                embeddings=embeddings,
                frame_embeddings=frame_embeddings,
                active_value=args.active_value,
                random_labels=True,
            )
            random_label_model = train_model(train=random_train, hidden=schema.hidden, args=args, seed=seed + 43)
            controls["random_label_control"] = refraction_prediction_summary(
                model=random_label_model,
                bundle=random_test,
                args=args,
            )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(TASK_FRAMES),
        "feature_groups": list(FEATURE_GROUPS),
        "frame_active_group": dict(FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.x)),
        "test_rows": int(len(test.x)),
        "topology": getattr(model, "topology_stats", {}),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "main": main,
        "controls": controls,
        "influence": influence,
        "hub_diagnostics": hub_diagnostics,
    }


def run_multi_aspect_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    train = make_multi_aspect_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 4_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    test = make_multi_aspect_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 4_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 51)
    main = evaluate_refraction_model(model=model, train=train, test=test, args=args)
    controls: dict[str, Any] = {
        "zero_recurrent_update": refraction_prediction_summary(
            model=model,
            bundle=test,
            args=args,
            ablation="zero_recurrent_update",
        ),
        "threshold_only": refraction_prediction_summary(
            model=model,
            bundle=test,
            args=args,
            ablation="zero_matrix_keep_threshold",
        ),
        "freeze_after_1": refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_1"),
        "freeze_after_2": refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_2"),
        "freeze_after_3": refraction_prediction_summary(model=model, bundle=test, args=args, ablation="freeze_after_3"),
        "shuffled_task_frame_token": refraction_prediction_summary(
            model=model,
            bundle=test,
            args=args,
            x_override=shuffled_frame_x(test, seed + 940_001),
        ),
    }

    no_frame_train = make_multi_aspect_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 4_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
        no_frame_token=True,
    )
    no_frame_test = make_multi_aspect_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 4_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
        no_frame_token=True,
    )
    no_frame_model = train_model(train=no_frame_train, hidden=schema.hidden, args=args, seed=seed + 52)
    controls["no_task_frame_token"] = refraction_prediction_summary(
        model=no_frame_model,
        bundle=no_frame_test,
        args=args,
    )

    random_model = recurrent_matrix_control(model, "randomize", seed + 950_001)
    controls["randomize_recurrent_matrix"] = refraction_prediction_summary(
        model=random_model,
        bundle=test,
        args=args,
    )

    if args.random_label_control:
        random_train = make_multi_aspect_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 5_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_multi_aspect_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 5_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_label_model = train_model(train=random_train, hidden=schema.hidden, args=args, seed=seed + 53)
        controls["random_label_control"] = refraction_prediction_summary(
            model=random_label_model,
            bundle=random_test,
            args=args,
        )

    influence = run_refraction_influence(model=model, test=test, args=args, seed=seed + 960_001)
    base_metrics = {
        "accuracy": main["accuracy"],
        "zero_recurrent": controls["zero_recurrent_update"]["accuracy"],
        "recurrence_gain": main["accuracy"] - controls["zero_recurrent_update"]["accuracy"],
        "refraction_index_final": (
            influence.get("mean_refraction_index_by_step", [None])[-1]
            if influence.get("mean_refraction_index_by_step")
            else None
        ),
        "authority_switch_score": influence.get("authority_switch_score"),
        "topology": getattr(model, "topology_stats", {}),
    }
    hub_diagnostics = run_hub_topology_diagnostics(
        model=model,
        test=test,
        args=args,
        seed=seed + 965_001,
        base_metrics=base_metrics,
    )
    token_influence = run_multi_aspect_token_influence(model=model, test=test, args=args, seed=seed + 970_001)
    pointer_diagnostics = run_pointer_resonance_diagnostics(
        model=model,
        test=test,
        embeddings=embeddings,
        args=args,
        seed=seed + 975_001,
    )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "feature_groups": list(MULTI_ASPECT_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.x)),
        "test_rows": int(len(test.x)),
        "topology": getattr(model, "topology_stats", {}),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "main": main,
        "controls": controls,
        "influence": influence,
        "token_influence": token_influence,
        "pointer_diagnostics": pointer_diagnostics,
        "hub_diagnostics": hub_diagnostics,
    }


def run_inferred_frame_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    train = make_inferred_frame_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 6_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    test = make_inferred_frame_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 6_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )

    oracle_model = train_frame_placement_model(
        train=train,
        hidden=schema.hidden,
        args=args,
        seed=seed + 91,
        frame_placement="frame_in_recurrence_only",
    )
    predicted_model = train_inferred_frame_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 92,
        use_pointer=True,
    )
    frame_head_only_model = train_inferred_frame_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 93,
        use_pointer=False,
    )
    no_frame_model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 94)

    oracle = frame_placement_prediction_summary(
        model=oracle_model,
        bundle=test,
        args=args,
    )
    predicted = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )
    predicted_soft = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=False,
    )
    frame_head_only = inferred_prediction_summary(
        model=frame_head_only_model,
        bundle=test,
        args=args,
        use_pointer=False,
    )
    no_frame = prediction_summary(
        model=no_frame_model,
        x=test.observation_component,
        y=test.y,
        args=args,
    )
    wrong_frame = ((test.frame + 1) % len(MULTI_ASPECT_FRAMES)).astype(np.int64)
    wrong_forced = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        frame_override=wrong_frame,
        use_pointer=True,
        hard_frame=True,
    )
    zero_recurrent = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
        ablation="zero_recurrent_update",
    )
    randomized_model = recurrent_matrix_control(predicted_model, "randomize", seed + 955_001)
    randomized_recurrent = inferred_prediction_summary(
        model=randomized_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )
    influence = run_inferred_refraction_influence(
        model=predicted_model,
        test=test,
        args=args,
        seed=seed + 960_001,
        use_pointer=True,
    )
    token_inventory = run_inferred_token_frame_inventory(
        model=predicted_model,
        test=test,
        args=args,
        seed=seed + 970_001,
    )

    random_label_control: dict[str, Any] | None = None
    if args.random_label_control:
        random_train = make_inferred_frame_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 7_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_inferred_frame_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 7_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_model = train_inferred_frame_pointer_model(
            train=random_train,
            frame_embeddings=frame_embeddings,
            hidden=schema.hidden,
            args=args,
            seed=seed + 95,
            use_pointer=True,
        )
        random_label_control = inferred_prediction_summary(
            model=random_model,
            bundle=random_test,
            args=args,
            use_pointer=True,
            hard_frame=True,
        )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "feature_groups": list(INFERRED_FRAME_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "topology": getattr(predicted_model, "topology_stats", {}),
        "cue_scale": {
            "active_scale": INFERRED_ACTIVE_SCALE,
            "inactive_scale": INFERRED_INACTIVE_SCALE,
            "frame_loss_weight": INFERRED_FRAME_LOSS_WEIGHT,
        },
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "oracle_frame": oracle,
        "predicted_frame_pointer": predicted,
        "predicted_frame_pointer_soft": predicted_soft,
        "frame_head_only": frame_head_only,
        "no_frame_baseline": no_frame,
        "wrong_forced_frame": wrong_forced,
        "zero_recurrent": zero_recurrent,
        "randomized_recurrent": randomized_recurrent,
        "influence": influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label_control,
    }


def run_query_cued_frame_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    query_embeddings = build_query_embeddings(schema.hidden, seed + 625_019, args.frame_scale)
    train = make_query_cued_frame_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 8_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        query_embeddings=query_embeddings,
        active_value=args.active_value,
    )
    test = make_query_cued_frame_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 8_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        query_embeddings=query_embeddings,
        active_value=args.active_value,
    )

    oracle_model = train_frame_placement_model(
        train=train,
        hidden=schema.hidden,
        args=args,
        seed=seed + 101,
        frame_placement="frame_in_recurrence_only",
    )
    predicted_model = train_inferred_frame_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 102,
        use_pointer=True,
    )
    query_head_only_model = train_inferred_frame_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 103,
        use_pointer=False,
    )
    no_pointer_query_model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 104)

    no_query_train = copy.copy(train)
    no_query_test = copy.copy(test)
    no_query_train.x = query_removed_observation(train)
    no_query_test.x = query_removed_observation(test)
    no_query_model = train_model(train=no_query_train, hidden=schema.hidden, args=args, seed=seed + 105)

    oracle = frame_placement_prediction_summary(
        model=oracle_model,
        bundle=test,
        args=args,
    )
    predicted = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )
    predicted_soft = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=False,
    )
    query_head_only = inferred_prediction_summary(
        model=query_head_only_model,
        bundle=test,
        args=args,
        use_pointer=False,
    )
    no_pointer_query = prediction_summary(
        model=no_pointer_query_model,
        x=test.x,
        y=test.y,
        args=args,
    )
    no_query = prediction_summary(
        model=no_query_model,
        x=no_query_test.x,
        y=test.y,
        args=args,
    )
    wrong_frame = ((test.frame + 1) % len(MULTI_ASPECT_FRAMES)).astype(np.int64)
    wrong_forced = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        frame_override=wrong_frame,
        use_pointer=True,
        hard_frame=True,
    )
    query_ablation = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        observation_override=query_removed_observation(test),
        use_pointer=True,
        hard_frame=True,
    )
    query_shuffle = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        observation_override=query_shuffled_observation(test, seed + 951_001),
        use_pointer=True,
        hard_frame=True,
    )
    zero_recurrent = inferred_prediction_summary(
        model=predicted_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
        ablation="zero_recurrent_update",
    )
    randomized_model = recurrent_matrix_control(predicted_model, "randomize", seed + 955_001)
    randomized_recurrent = inferred_prediction_summary(
        model=randomized_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )
    pointer_influence = run_inferred_refraction_influence(
        model=predicted_model,
        test=test,
        args=args,
        seed=seed + 960_001,
        use_pointer=True,
    )
    no_pointer_influence = run_direct_refraction_influence(
        model=no_pointer_query_model,
        test=test,
        args=args,
        seed=seed + 962_001,
    )
    token_inventory = run_inferred_token_frame_inventory(
        model=predicted_model,
        test=test,
        args=args,
        seed=seed + 970_001,
    )

    random_label_control: dict[str, Any] | None = None
    if args.random_label_control:
        random_train = make_query_cued_frame_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 9_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            query_embeddings=query_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_query_cued_frame_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 9_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            query_embeddings=query_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_model = train_inferred_frame_pointer_model(
            train=random_train,
            frame_embeddings=frame_embeddings,
            hidden=schema.hidden,
            args=args,
            seed=seed + 106,
            use_pointer=True,
        )
        random_label_control = inferred_prediction_summary(
            model=random_model,
            bundle=random_test,
            args=args,
            use_pointer=True,
            hard_frame=True,
        )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "query_cues": list(QUERY_CUES),
        "query_to_frame": dict(QUERY_TO_FRAME),
        "feature_groups": list(INFERRED_FRAME_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "topology": getattr(predicted_model, "topology_stats", {}),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "same_observation_label_diversity": label_diversity_by_observation(test),
        "oracle_frame": oracle,
        "query_predicted_frame_pointer": predicted,
        "query_predicted_frame_pointer_soft": predicted_soft,
        "query_head_only": query_head_only,
        "no_pointer_query_baseline": no_pointer_query,
        "no_query_baseline": no_query,
        "wrong_forced_frame": wrong_forced,
        "query_ablation": query_ablation,
        "query_shuffle": query_shuffle,
        "zero_recurrent": zero_recurrent,
        "randomized_recurrent": randomized_recurrent,
        "pointer_influence": pointer_influence,
        "no_pointer_influence": no_pointer_influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label_control,
    }


def run_query_cued_pointer_bottleneck_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    query_embeddings = build_query_embeddings(schema.hidden, seed + 625_019, args.frame_scale)
    train = make_query_cued_frame_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 8_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        query_embeddings=query_embeddings,
        active_value=args.active_value,
    )
    test = make_query_cued_frame_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 8_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        query_embeddings=query_embeddings,
        active_value=args.active_value,
    )
    queryless_train = query_removed_bundle(train)
    queryless_test = query_removed_bundle(test)

    oracle_model = train_frame_placement_model(
        train=queryless_train,
        hidden=schema.hidden,
        args=args,
        seed=seed + 111,
        frame_placement="frame_in_recurrence_only",
    )
    pointer_model = train_query_cued_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 112,
        use_pointer=True,
    )
    frame_head_only_model = train_inferred_frame_pointer_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 113,
        use_pointer=False,
    )
    full_query_direct_model = train_model(train=train, hidden=schema.hidden, args=args, seed=seed + 114)
    no_query_model = train_model(train=queryless_train, hidden=schema.hidden, args=args, seed=seed + 115)

    bottleneck_models: dict[int, QueryBottleneckDirectClassifier] = {}
    for size in QUERY_BOTTLENECK_SIZES:
        bottleneck_models[size] = train_query_bottleneck_direct_model(
            train=train,
            hidden=schema.hidden,
            bottleneck=size,
            args=args,
            seed=seed + 120 + size,
        )

    oracle = frame_placement_prediction_summary(
        model=oracle_model,
        bundle=queryless_test,
        args=args,
    )
    predicted = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )
    predicted_soft = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=False,
    )
    hard_discrete = predicted
    frame_head_only = inferred_prediction_summary(
        model=frame_head_only_model,
        bundle=test,
        args=args,
        use_pointer=False,
    )
    full_query_direct = prediction_summary(
        model=full_query_direct_model,
        x=test.x,
        y=test.y,
        args=args,
    )
    no_query = prediction_summary(
        model=no_query_model,
        x=queryless_test.x,
        y=test.y,
        args=args,
    )
    wrong_frame = ((test.frame + 1) % len(MULTI_ASPECT_FRAMES)).astype(np.int64)
    wrong_forced = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        frame_override=wrong_frame,
        use_pointer=True,
        hard_frame=True,
    )
    query_ablation = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        frame_input_override=query_removed_observation(test),
        use_pointer=True,
        hard_frame=True,
    )
    query_shuffle = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        frame_input_override=query_shuffled_observation(test, seed + 951_001),
        use_pointer=True,
        hard_frame=True,
    )
    zero_recurrent = query_pointer_prediction_summary(
        model=pointer_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
        ablation="zero_recurrent_update",
    )
    randomized_model = recurrent_matrix_control(pointer_model, "randomize", seed + 955_001)
    randomized_recurrent = query_pointer_prediction_summary(
        model=randomized_model,
        bundle=test,
        args=args,
        use_pointer=True,
        hard_frame=True,
    )

    bottleneck_query_direct: dict[str, Any] = {}
    bottleneck_influence: dict[str, Any] = {}
    for size, model in bottleneck_models.items():
        key = str(size)
        bottleneck_query_direct[key] = query_bottleneck_prediction_summary(
            model=model,
            bundle=test,
            args=args,
        )
        bottleneck_influence[key] = run_query_bottleneck_refraction_influence(
            model=model,
            test=test,
            args=args,
            seed=seed + 965_000 + size,
        )

    pointer_influence = run_query_pointer_refraction_influence(
        model=pointer_model,
        test=test,
        args=args,
        seed=seed + 960_001,
        use_pointer=True,
    )
    full_query_direct_influence = run_direct_refraction_influence(
        model=full_query_direct_model,
        test=test,
        args=args,
        seed=seed + 962_001,
    )
    token_inventory = run_query_pointer_token_frame_inventory(
        model=pointer_model,
        test=test,
        args=args,
        seed=seed + 970_001,
    )

    random_label_control: dict[str, Any] | None = None
    if args.random_label_control:
        random_train = make_query_cued_frame_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 9_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            query_embeddings=query_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_query_cued_frame_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 9_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            query_embeddings=query_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_model = train_query_cued_pointer_model(
            train=random_train,
            frame_embeddings=frame_embeddings,
            hidden=schema.hidden,
            args=args,
            seed=seed + 116,
            use_pointer=True,
        )
        random_label_control = query_pointer_prediction_summary(
            model=random_model,
            bundle=random_test,
            args=args,
            use_pointer=True,
            hard_frame=True,
        )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "query_cues": list(QUERY_CUES),
        "query_to_frame": dict(QUERY_TO_FRAME),
        "query_bottleneck_sizes": list(QUERY_BOTTLENECK_SIZES),
        "feature_groups": list(INFERRED_FRAME_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "topology": getattr(pointer_model, "topology_stats", {}),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "same_observation_label_diversity": label_diversity_by_observation(test),
        "oracle_frame_pointer": oracle,
        "predicted_frame_pointer": predicted,
        "predicted_soft_frame_pointer": predicted_soft,
        "hard_discrete_predicted_pointer": hard_discrete,
        "full_query_direct": full_query_direct,
        "bottleneck_query_direct": bottleneck_query_direct,
        "frame_head_only": frame_head_only,
        "no_query_baseline": no_query,
        "wrong_forced_frame": wrong_forced,
        "query_ablation": query_ablation,
        "query_shuffle": query_shuffle,
        "zero_recurrent": zero_recurrent,
        "randomized_recurrent": randomized_recurrent,
        "pointer_influence": pointer_influence,
        "full_query_direct_influence": full_query_direct_influence,
        "bottleneck_query_direct_influence": bottleneck_influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label_control,
    }


def run_frame_switch_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    train = make_multi_aspect_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 4_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    test = make_multi_aspect_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 4_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )

    placement_results: dict[str, Any] = {}
    placement_models: dict[str, FramePlacementRecurrentClassifier] = {}
    for placement_index, placement in enumerate(FRAME_PLACEMENTS):
        model = train_frame_placement_model(
            train=train,
            hidden=schema.hidden,
            args=args,
            seed=seed + 71 + placement_index,
            frame_placement=placement,
        )
        placement_models[placement] = model
        placement_results[placement] = {
            "main": frame_placement_prediction_summary(
                model=model,
                bundle=test,
                args=args,
            )
        }

    main_model = placement_models["frame_in_recurrence_only"]
    controls: dict[str, Any] = {
        "zero_recurrent_update": frame_placement_prediction_summary(
            model=main_model,
            bundle=test,
            args=args,
            ablation="zero_recurrent_update",
        ),
        "threshold_only": frame_placement_prediction_summary(
            model=main_model,
            bundle=test,
            args=args,
            ablation="zero_matrix_keep_threshold",
        ),
        "shuffled_task_frame_token": frame_placement_prediction_summary(
            model=main_model,
            bundle=test,
            args=args,
            frame_override=shuffled_frame_component(test, seed + 940_001),
        ),
    }
    random_model = recurrent_matrix_control(main_model, "randomize", seed + 950_001)
    controls["randomize_recurrent_matrix"] = frame_placement_prediction_summary(
        model=random_model,
        bundle=test,
        args=args,
    )

    if args.random_label_control:
        random_train = make_multi_aspect_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 5_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_multi_aspect_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 5_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_label_model = train_frame_placement_model(
            train=random_train,
            hidden=schema.hidden,
            args=args,
            seed=seed + 73,
            frame_placement="frame_in_recurrence_only",
        )
        controls["random_label_control"] = frame_placement_prediction_summary(
            model=random_label_model,
            bundle=random_test,
            args=args,
        )

    direct_influence = run_frame_placement_influence(
        model=main_model,
        test=test,
        args=args,
        seed=seed + 960_001,
    )
    token_influence = run_frame_placement_token_influence(
        model=main_model,
        test=test,
        args=args,
        seed=seed + 970_001,
    )
    mid_run_switch = run_mid_run_switch_diagnostics(
        model=main_model,
        test=test,
        args=args,
        seed=seed + 980_001,
    )
    trajectory_geometry = run_hidden_trajectory_geometry(
        model=main_model,
        test=test,
        args=args,
    )
    soft_interpolation = run_soft_frame_interpolation(
        model=main_model,
        test=test,
        args=args,
        seed=seed + 990_001,
    )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "feature_groups": list(MULTI_ASPECT_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "frame_placements": list(FRAME_PLACEMENTS),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "placement_results": placement_results,
        "controls": controls,
        "direct_influence": direct_influence,
        "token_influence": token_influence,
        "mid_run_switch": mid_run_switch,
        "trajectory_geometry": trajectory_geometry,
        "soft_frame_interpolation": soft_interpolation,
    }


def run_reframe_seed(
    *,
    name: str,
    input_mode: str,
    seed: int,
    schema: Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_combos, heldout_combos = split_nuisance_combos(seed, args.holdout_fraction)
    embeddings = build_embeddings(
        schema=schema,
        input_mode=input_mode,
        seed=seed + 500_003,
        embed_scale=args.embed_scale,
        opponent_strength=args.opponent_strength,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
    )
    frame_embeddings = build_named_frame_embeddings(
        MULTI_ASPECT_FRAMES,
        schema.hidden,
        seed + 620_011,
        args.frame_scale,
    )
    train = make_multi_aspect_dataset(
        n=args.train_size,
        combos=train_combos,
        seed=seed + 4_001,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )
    test = make_multi_aspect_dataset(
        n=args.test_size,
        combos=heldout_combos,
        seed=seed + 4_002,
        embeddings=embeddings,
        frame_embeddings=frame_embeddings,
        active_value=args.active_value,
    )

    baseline_model = train_reframe_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 81,
        train_reframes=False,
    )
    trained_reframe_model = train_reframe_model(
        train=train,
        frame_embeddings=frame_embeddings,
        hidden=schema.hidden,
        args=args,
        seed=seed + 82,
        train_reframes=True,
    )

    no_reframe_baseline = {
        "fixed_frame": reframe_prediction_summary(
            model=baseline_model,
            bundle=test,
            args=args,
        ),
        "wrong_initial_frame_no_reset": run_reframe_recovery_diagnostics(
            model=baseline_model,
            test=test,
            args=args,
            seed=seed + 980_001,
            use_reset=False,
        ),
        "wrong_initial_frame_with_reset": run_reframe_recovery_diagnostics(
            model=baseline_model,
            test=test,
            args=args,
            seed=seed + 981_001,
            use_reset=True,
        ),
    }
    trained_reframe = {
        "fixed_frame": reframe_prediction_summary(
            model=trained_reframe_model,
            bundle=test,
            args=args,
        ),
        "wrong_initial_frame_no_reset": run_reframe_recovery_diagnostics(
            model=trained_reframe_model,
            test=test,
            args=args,
            seed=seed + 982_001,
            use_reset=False,
        ),
        "wrong_initial_frame_with_reset": run_reframe_recovery_diagnostics(
            model=trained_reframe_model,
            test=test,
            args=args,
            seed=seed + 983_001,
            use_reset=True,
        ),
    }

    controls: dict[str, Any] = {
        "zero_recurrent_update": reframe_prediction_summary(
            model=trained_reframe_model,
            bundle=test,
            args=args,
            ablation="zero_recurrent_update",
        ),
        "threshold_only": reframe_prediction_summary(
            model=trained_reframe_model,
            bundle=test,
            args=args,
            ablation="zero_matrix_keep_threshold",
        ),
        "shuffled_task_frame_token": reframe_prediction_summary(
            model=trained_reframe_model,
            bundle=test,
            args=args,
            frame_override=shuffled_frame_component(test, seed + 940_001),
        ),
    }
    randomized_model = recurrent_matrix_control(trained_reframe_model, "randomize", seed + 950_001)
    controls["randomize_recurrent_matrix"] = reframe_prediction_summary(
        model=randomized_model,
        bundle=test,
        args=args,
    )

    if args.random_label_control:
        random_train = make_multi_aspect_dataset(
            n=args.train_size,
            combos=train_combos,
            seed=seed + 5_001,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_test = make_multi_aspect_dataset(
            n=args.test_size,
            combos=heldout_combos,
            seed=seed + 5_002,
            embeddings=embeddings,
            frame_embeddings=frame_embeddings,
            active_value=args.active_value,
            random_labels=True,
        )
        random_label_model = train_reframe_model(
            train=random_train,
            frame_embeddings=frame_embeddings,
            hidden=schema.hidden,
            args=args,
            seed=seed + 83,
            train_reframes=True,
        )
        controls["random_label_control"] = reframe_prediction_summary(
            model=random_label_model,
            bundle=random_test,
            args=args,
        )

    return {
        "name": name,
        "input_mode": input_mode,
        "seed": seed,
        "task_frames": list(MULTI_ASPECT_FRAMES),
        "feature_groups": list(MULTI_ASPECT_FEATURE_GROUPS),
        "frame_active_group": dict(MULTI_ASPECT_FRAME_ACTIVE_GROUP),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "nuisance_split": {
            "train_combo_count": len(train_combos),
            "heldout_combo_count": len(heldout_combos),
        },
        "no_reframe_baseline": no_reframe_baseline,
        "trained_reframe": trained_reframe,
        "controls": controls,
    }


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
        "mean_abs_margin_delta",
    ]
    list_keys = [
        "output_change_rate_by_step",
        "target_accuracy_by_step",
        "original_label_retention_by_step",
        "mean_abs_label_probability_delta_by_step",
        "mean_kl_divergence_by_step",
        "mean_abs_margin_delta_by_step",
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


def aggregate_nested(values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if first is None:
        return None
    if isinstance(first, (float, int, np.floating, np.integer)) and not isinstance(first, bool):
        return float(np.mean([float(value) for value in values]))
    if isinstance(first, list):
        if not first:
            return []
        if all(isinstance(item, (float, int, np.floating, np.integer)) for row in values for item in row):
            return mean_list([[float(item) for item in row] for row in values])
        return first
    if isinstance(first, dict):
        out: dict[str, Any] = {}
        for key in first:
            if all(isinstance(value, dict) and key in value for value in values):
                out[key] = aggregate_nested([value[key] for value in values])
        return out
    return first


def aggregate_refraction_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    main = aggregate_nested([run["main"] for run in runs])
    controls = aggregate_nested([run["controls"] for run in runs])
    influence = aggregate_nested([run["influence"] for run in runs])
    token_influence = aggregate_nested([run.get("token_influence", {}) for run in runs])
    pointer_diagnostics = aggregate_nested([run.get("pointer_diagnostics", {}) for run in runs])
    hub_diagnostics = aggregate_nested([run.get("hub_diagnostics", {}) for run in runs])
    topology = aggregate_nested([run.get("topology", {}) for run in runs])
    recurrence_gain = main["accuracy"] - controls["zero_recurrent_update"]["accuracy"]
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "topology": topology,
        "accuracy": main["accuracy"],
        "accuracy_by_frame": main["accuracy_by_frame"],
        "per_step_accuracy": main["per_step_accuracy"],
        "feature_group_probe_accuracy_by_step": main["feature_group_probe_accuracy_by_step"],
        "feature_group_probe_accuracy_by_frame_by_step": main["feature_group_probe_accuracy_by_frame_by_step"],
        "same_observation_label_diversity": main["same_observation_label_diversity"],
        "output_entropy_by_step": main["output_entropy_by_step"],
        "logit_margin_by_step": main["logit_margin_by_step"],
        "label_logit_confidence_by_step": main["label_logit_confidence_by_step"],
        "recurrence_gain": recurrence_gain,
        "main": main,
        "controls": controls,
        "influence": influence,
        "token_influence": token_influence,
        "pointer_diagnostics": pointer_diagnostics,
        "hub_diagnostics": hub_diagnostics,
        "runs": runs,
    }


def aggregate_inferred_frame_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = aggregate_nested([run["oracle_frame"] for run in runs])
    predicted = aggregate_nested([run["predicted_frame_pointer"] for run in runs])
    predicted_soft = aggregate_nested([run["predicted_frame_pointer_soft"] for run in runs])
    frame_head_only = aggregate_nested([run["frame_head_only"] for run in runs])
    no_frame = aggregate_nested([run["no_frame_baseline"] for run in runs])
    wrong = aggregate_nested([run["wrong_forced_frame"] for run in runs])
    zero = aggregate_nested([run["zero_recurrent"] for run in runs])
    randomized = aggregate_nested([run["randomized_recurrent"] for run in runs])
    influence = aggregate_nested([run["influence"] for run in runs])
    token_inventory = aggregate_nested([run["token_frame_inventory"] for run in runs])
    random_label_runs = [run["random_label_control"] for run in runs if run.get("random_label_control") is not None]
    random_label = aggregate_nested(random_label_runs) if random_label_runs else None
    topology = aggregate_nested([run.get("topology", {}) for run in runs])
    refraction_curve = influence.get("mean_refraction_index_by_step", []) if isinstance(influence, dict) else []
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "topology": topology,
        "cue_scale": runs[0]["cue_scale"],
        "frame_prediction_accuracy": predicted["frame_prediction_accuracy"],
        "task_accuracy_oracle_frame": oracle["accuracy"],
        "task_accuracy_predicted_frame": predicted["accuracy"],
        "task_accuracy_predicted_frame_soft": predicted_soft["accuracy"],
        "task_accuracy_frame_head_only": frame_head_only["accuracy"],
        "task_accuracy_no_frame": no_frame["accuracy"],
        "task_accuracy_wrong_forced_frame": wrong["accuracy"],
        "task_accuracy_zero_recurrent": zero["accuracy"],
        "task_accuracy_randomized_recurrent": randomized["accuracy"],
        "authority_switch_score_predicted_frame": influence.get("authority_switch_score"),
        "refraction_index_predicted_frame": refraction_curve[-1] if refraction_curve else None,
        "oracle_frame": oracle,
        "predicted_frame_pointer": predicted,
        "predicted_frame_pointer_soft": predicted_soft,
        "frame_head_only": frame_head_only,
        "no_frame_baseline": no_frame,
        "wrong_forced_frame": wrong,
        "zero_recurrent": zero,
        "randomized_recurrent": randomized,
        "influence": influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label,
        "runs": runs,
    }


def aggregate_query_cued_frame_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = aggregate_nested([run["oracle_frame"] for run in runs])
    predicted = aggregate_nested([run["query_predicted_frame_pointer"] for run in runs])
    predicted_soft = aggregate_nested([run["query_predicted_frame_pointer_soft"] for run in runs])
    query_head_only = aggregate_nested([run["query_head_only"] for run in runs])
    no_pointer_query = aggregate_nested([run["no_pointer_query_baseline"] for run in runs])
    no_query = aggregate_nested([run["no_query_baseline"] for run in runs])
    wrong = aggregate_nested([run["wrong_forced_frame"] for run in runs])
    query_ablation = aggregate_nested([run["query_ablation"] for run in runs])
    query_shuffle = aggregate_nested([run["query_shuffle"] for run in runs])
    zero = aggregate_nested([run["zero_recurrent"] for run in runs])
    randomized = aggregate_nested([run["randomized_recurrent"] for run in runs])
    pointer_influence = aggregate_nested([run["pointer_influence"] for run in runs])
    no_pointer_influence = aggregate_nested([run["no_pointer_influence"] for run in runs])
    token_inventory = aggregate_nested([run["token_frame_inventory"] for run in runs])
    random_label_runs = [run["random_label_control"] for run in runs if run.get("random_label_control") is not None]
    random_label = aggregate_nested(random_label_runs) if random_label_runs else None
    topology = aggregate_nested([run.get("topology", {}) for run in runs])
    pointer_curve = pointer_influence.get("mean_refraction_index_by_step", [])
    no_pointer_curve = no_pointer_influence.get("mean_refraction_index_by_step", [])
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "query_cues": runs[0]["query_cues"],
        "query_to_frame": runs[0]["query_to_frame"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "topology": topology,
        "same_observation_label_diversity": float(np.mean([run["same_observation_label_diversity"] for run in runs])),
        "frame_prediction_accuracy": predicted["frame_prediction_accuracy"],
        "oracle_frame_accuracy": oracle["accuracy"],
        "predicted_frame_pointer_accuracy": predicted["accuracy"],
        "predicted_frame_pointer_soft_accuracy": predicted_soft["accuracy"],
        "query_head_only_accuracy": query_head_only["accuracy"],
        "no_pointer_query_baseline_accuracy": no_pointer_query["accuracy"],
        "no_query_baseline_accuracy": no_query["accuracy"],
        "wrong_forced_frame_accuracy": wrong["accuracy"],
        "query_ablation_accuracy": query_ablation["accuracy"],
        "query_shuffle_accuracy": query_shuffle["accuracy"],
        "zero_recurrent_accuracy": zero["accuracy"],
        "randomized_recurrent_accuracy": randomized["accuracy"],
        "random_label_accuracy": random_label.get("accuracy") if isinstance(random_label, dict) else None,
        "authority_switch_score": pointer_influence.get("authority_switch_score"),
        "refraction_index_final": pointer_curve[-1] if pointer_curve else None,
        "no_pointer_authority_switch_score": no_pointer_influence.get("authority_switch_score"),
        "no_pointer_refraction_index_final": no_pointer_curve[-1] if no_pointer_curve else None,
        "oracle_frame": oracle,
        "query_predicted_frame_pointer": predicted,
        "query_predicted_frame_pointer_soft": predicted_soft,
        "query_head_only": query_head_only,
        "no_pointer_query_baseline": no_pointer_query,
        "no_query_baseline": no_query,
        "wrong_forced_frame": wrong,
        "query_ablation": query_ablation,
        "query_shuffle": query_shuffle,
        "zero_recurrent": zero,
        "randomized_recurrent": randomized,
        "pointer_influence": pointer_influence,
        "no_pointer_influence": no_pointer_influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label,
        "runs": runs,
    }


def final_mean_influence(influence: dict[str, Any], key: str) -> float | None:
    curves = influence.get(key, {}) if isinstance(influence, dict) else {}
    if not isinstance(curves, dict):
        return None
    values = [float(curve[-1]) for curve in curves.values() if isinstance(curve, list) and curve]
    return float(np.mean(values)) if values else None


def final_refraction_index(influence: dict[str, Any]) -> float | None:
    curve = influence.get("mean_refraction_index_by_step", []) if isinstance(influence, dict) else []
    return float(curve[-1]) if curve else None


def aggregate_query_cued_pointer_bottleneck_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = aggregate_nested([run["oracle_frame_pointer"] for run in runs])
    predicted = aggregate_nested([run["predicted_frame_pointer"] for run in runs])
    predicted_soft = aggregate_nested([run["predicted_soft_frame_pointer"] for run in runs])
    hard_discrete = aggregate_nested([run["hard_discrete_predicted_pointer"] for run in runs])
    full_query_direct = aggregate_nested([run["full_query_direct"] for run in runs])
    bottleneck_query_direct = aggregate_nested([run["bottleneck_query_direct"] for run in runs])
    frame_head_only = aggregate_nested([run["frame_head_only"] for run in runs])
    no_query = aggregate_nested([run["no_query_baseline"] for run in runs])
    wrong = aggregate_nested([run["wrong_forced_frame"] for run in runs])
    query_ablation = aggregate_nested([run["query_ablation"] for run in runs])
    query_shuffle = aggregate_nested([run["query_shuffle"] for run in runs])
    zero = aggregate_nested([run["zero_recurrent"] for run in runs])
    randomized = aggregate_nested([run["randomized_recurrent"] for run in runs])
    pointer_influence = aggregate_nested([run["pointer_influence"] for run in runs])
    full_query_direct_influence = aggregate_nested([run["full_query_direct_influence"] for run in runs])
    bottleneck_influence = aggregate_nested([run["bottleneck_query_direct_influence"] for run in runs])
    token_inventory = aggregate_nested([run["token_frame_inventory"] for run in runs])
    random_label_runs = [run["random_label_control"] for run in runs if run.get("random_label_control") is not None]
    random_label = aggregate_nested(random_label_runs) if random_label_runs else None
    topology = aggregate_nested([run.get("topology", {}) for run in runs])

    accuracy_vs_bottleneck = {
        size: value["accuracy"]
        for size, value in bottleneck_query_direct.items()
    }
    authority_vs_bottleneck = {
        size: value.get("authority_switch_score")
        for size, value in bottleneck_influence.items()
    }
    refraction_vs_bottleneck = {
        size: final_refraction_index(value)
        for size, value in bottleneck_influence.items()
    }
    active_vs_bottleneck = {
        size: final_mean_influence(value, "active_core_influence_by_step")
        for size, value in bottleneck_influence.items()
    }
    inactive_vs_bottleneck = {
        size: final_mean_influence(value, "inactive_group_influence_by_step")
        for size, value in bottleneck_influence.items()
    }

    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "query_cues": runs[0]["query_cues"],
        "query_to_frame": runs[0]["query_to_frame"],
        "query_bottleneck_sizes": runs[0]["query_bottleneck_sizes"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "topology": topology,
        "same_observation_label_diversity": float(np.mean([run["same_observation_label_diversity"] for run in runs])),
        "frame_prediction_accuracy": predicted["frame_prediction_accuracy"],
        "oracle_frame_pointer_accuracy": oracle["accuracy"],
        "predicted_frame_pointer_accuracy": predicted["accuracy"],
        "predicted_soft_frame_pointer_accuracy": predicted_soft["accuracy"],
        "hard_discrete_predicted_pointer_accuracy": hard_discrete["accuracy"],
        "full_query_direct_accuracy": full_query_direct["accuracy"],
        "bottleneck_query_direct_accuracy_by_size": accuracy_vs_bottleneck,
        "frame_head_only_accuracy": frame_head_only["accuracy"],
        "no_query_baseline_accuracy": no_query["accuracy"],
        "wrong_forced_frame_accuracy": wrong["accuracy"],
        "query_ablation_accuracy": query_ablation["accuracy"],
        "query_shuffle_accuracy": query_shuffle["accuracy"],
        "zero_recurrent_accuracy": zero["accuracy"],
        "randomized_recurrent_accuracy": randomized["accuracy"],
        "random_label_accuracy": random_label.get("accuracy") if isinstance(random_label, dict) else None,
        "authority_switch_score": pointer_influence.get("authority_switch_score"),
        "refraction_index_final": final_refraction_index(pointer_influence),
        "active_group_influence": final_mean_influence(pointer_influence, "active_core_influence_by_step"),
        "inactive_group_influence": final_mean_influence(pointer_influence, "inactive_group_influence_by_step"),
        "full_query_direct_authority_switch_score": full_query_direct_influence.get("authority_switch_score"),
        "full_query_direct_refraction_index_final": final_refraction_index(full_query_direct_influence),
        "full_query_direct_active_group_influence": final_mean_influence(
            full_query_direct_influence,
            "active_core_influence_by_step",
        ),
        "full_query_direct_inactive_group_influence": final_mean_influence(
            full_query_direct_influence,
            "inactive_group_influence_by_step",
        ),
        "authority_vs_bottleneck_size": authority_vs_bottleneck,
        "refraction_vs_bottleneck_size": refraction_vs_bottleneck,
        "active_influence_vs_bottleneck_size": active_vs_bottleneck,
        "inactive_influence_vs_bottleneck_size": inactive_vs_bottleneck,
        "oracle_frame_pointer": oracle,
        "predicted_frame_pointer": predicted,
        "predicted_soft_frame_pointer": predicted_soft,
        "hard_discrete_predicted_pointer": hard_discrete,
        "full_query_direct": full_query_direct,
        "bottleneck_query_direct": bottleneck_query_direct,
        "frame_head_only": frame_head_only,
        "no_query_baseline": no_query,
        "wrong_forced_frame": wrong,
        "query_ablation": query_ablation,
        "query_shuffle": query_shuffle,
        "zero_recurrent": zero,
        "randomized_recurrent": randomized,
        "pointer_influence": pointer_influence,
        "full_query_direct_influence": full_query_direct_influence,
        "bottleneck_query_direct_influence": bottleneck_influence,
        "token_frame_inventory": token_inventory,
        "random_label_control": random_label,
        "runs": runs,
    }


def aggregate_frame_switch_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    placement_results = aggregate_nested([run["placement_results"] for run in runs])
    controls = aggregate_nested([run["controls"] for run in runs])
    direct_influence = aggregate_nested([run["direct_influence"] for run in runs])
    token_influence = aggregate_nested([run["token_influence"] for run in runs])
    mid_run_switch = aggregate_nested([run["mid_run_switch"] for run in runs])
    trajectory_geometry = aggregate_nested([run["trajectory_geometry"] for run in runs])
    soft_frame_interpolation = aggregate_nested([run["soft_frame_interpolation"] for run in runs])
    frame_in_accuracy = placement_results["frame_in_recurrence_only"]["main"]["accuracy"]
    output_only_accuracy = placement_results["frame_at_output_only"]["main"]["accuracy"]
    zero_accuracy = controls["zero_recurrent_update"]["accuracy"]
    randomized_accuracy = controls["randomize_recurrent_matrix"]["accuracy"]
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "frame_placements": runs[0]["frame_placements"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "frame_in_recurrence_only_accuracy": frame_in_accuracy,
        "frame_at_output_only_accuracy": output_only_accuracy,
        "frame_initial_only_accuracy": placement_results["frame_initial_only"]["main"]["accuracy"],
        "no_frame_accuracy": placement_results["no_frame"]["main"]["accuracy"],
        "recurrent_vs_output_only_gain": frame_in_accuracy - output_only_accuracy,
        "recurrent_vs_zero_gain": frame_in_accuracy - zero_accuracy,
        "randomized_recurrent_accuracy": randomized_accuracy,
        "placement_results": placement_results,
        "controls": controls,
        "direct_influence": direct_influence,
        "token_influence": token_influence,
        "mid_run_switch": mid_run_switch,
        "trajectory_geometry": trajectory_geometry,
        "soft_frame_interpolation": soft_frame_interpolation,
        "runs": runs,
    }


def aggregate_reframe_runs(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    no_reframe_baseline = aggregate_nested([run["no_reframe_baseline"] for run in runs])
    trained_reframe = aggregate_nested([run["trained_reframe"] for run in runs])
    controls = aggregate_nested([run["controls"] for run in runs])
    baseline_fixed = no_reframe_baseline["fixed_frame"]["accuracy"]
    trained_fixed = trained_reframe["fixed_frame"]["accuracy"]
    trained_no_reset = trained_reframe["wrong_initial_frame_no_reset"]
    trained_with_reset = trained_reframe["wrong_initial_frame_with_reset"]
    reset_accuracy_gain = (
        trained_with_reset["final_accuracy_after_reframe"]
        - trained_no_reset["final_accuracy_after_reframe"]
    )
    reset_success_gain = (
        trained_with_reset["reframe_success_rate"]
        - trained_no_reset["reframe_success_rate"]
    )
    authority_transfer_gain = (
        trained_with_reset["authority_switch_after_frame_switch"]
        - trained_no_reset["authority_switch_after_frame_switch"]
    )
    return {
        "name": name,
        "input_mode": runs[0]["input_mode"],
        "task_frames": runs[0]["task_frames"],
        "feature_groups": runs[0]["feature_groups"],
        "frame_active_group": runs[0]["frame_active_group"],
        "train_rows": int(np.mean([run["train_rows"] for run in runs])),
        "test_rows": int(np.mean([run["test_rows"] for run in runs])),
        "no_reframe_baseline_accuracy": baseline_fixed,
        "trained_reframe_accuracy": trained_fixed,
        "wrong_initial_no_reset_accuracy": trained_no_reset["final_accuracy_after_reframe"],
        "wrong_initial_with_reset_accuracy": trained_with_reset["final_accuracy_after_reframe"],
        "reset_accuracy_gain": reset_accuracy_gain,
        "reset_success_gain": reset_success_gain,
        "authority_transfer_gain": authority_transfer_gain,
        "no_reframe_baseline": no_reframe_baseline,
        "trained_reframe": trained_reframe,
        "controls": controls,
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


def interpret_refraction_report(summary: dict[str, Any]) -> dict[str, str]:
    main = summary["main"]
    controls = summary["controls"]
    influence = summary.get("influence", {})
    accuracy = summary["accuracy"]
    recurrence_gain = summary["recurrence_gain"]
    zero_accuracy = controls["zero_recurrent_update"]["accuracy"]
    no_frame_accuracy = controls.get("no_task_frame_token", {}).get("accuracy")
    shuffled_accuracy = controls.get("shuffled_task_frame_token", {}).get("accuracy")
    randomized_accuracy = controls.get("randomize_recurrent_matrix", {}).get("accuracy")
    random_label_accuracy = controls.get("random_label_control", {}).get("accuracy")
    mean_refraction_index = influence.get("mean_refraction_index_by_step", [])
    authority_switch = influence.get("authority_switch_score")
    frame_accuracies = main["accuracy_by_frame"].values()

    high_accuracy = accuracy >= 0.85 and min(frame_accuracies) >= 0.80
    recurrence_matters = recurrence_gain >= 0.05
    frame_token_matters = no_frame_accuracy is not None and no_frame_accuracy <= accuracy - 0.10
    shuffled_hurts = shuffled_accuracy is not None and shuffled_accuracy <= accuracy - 0.10
    randomized_hurts = randomized_accuracy is not None and randomized_accuracy <= accuracy - 0.20
    random_fails = random_label_accuracy is None or random_label_accuracy < 0.65
    refraction_rises = (
        len(mean_refraction_index) >= 2
        and mean_refraction_index[-1] >= 0.15
        and mean_refraction_index[-1] >= mean_refraction_index[0] + 0.05
    )
    authority_switches = authority_switch is not None and authority_switch >= 0.15

    supports = (
        high_accuracy
        and recurrence_matters
        and frame_token_matters
        and shuffled_hurts
        and randomized_hurts
        and random_fails
        and refraction_rises
        and authority_switches
    )

    if supports:
        return {
            "supports_recurrent_latent_refraction": "true",
            "supports_task_frame_conditional_core_dominance": "true",
            "reason": (
                "The same entangled observations are solved under multiple task frames, recurrence beats the "
                "zero-recurrent baseline, removing or shuffling the frame token hurts, randomizing recurrence "
                "destroys the gain, active-group influence separates from inactive-group influence over steps, "
                "and feature groups show higher authority when causal than when nuisance."
            ),
        }

    return {
        "supports_recurrent_latent_refraction": "unclear",
        "supports_task_frame_conditional_core_dominance": "unclear",
        "reason": (
            f"accuracy={accuracy:.3f}, zero_accuracy={zero_accuracy:.3f}, recurrence_gain={recurrence_gain:.3f}, "
            f"no_frame_accuracy={no_frame_accuracy if no_frame_accuracy is not None else 'n/a'}, "
            f"shuffled_frame_accuracy={shuffled_accuracy if shuffled_accuracy is not None else 'n/a'}, "
            f"randomized_recurrent_accuracy={randomized_accuracy if randomized_accuracy is not None else 'n/a'}, "
            f"authority_switch_score={authority_switch if authority_switch is not None else 'n/a'}, "
            f"mean_refraction_index_final={mean_refraction_index[-1] if mean_refraction_index else 'n/a'}."
        ),
    }


def interpret_multi_aspect_report(summary: dict[str, Any]) -> dict[str, str]:
    controls = summary["controls"]
    accuracy = summary["accuracy"]
    frame_accuracies = summary["accuracy_by_frame"].values()
    recurrence_gain = summary["recurrence_gain"]
    no_frame_accuracy = controls.get("no_task_frame_token", {}).get("accuracy")
    shuffled_accuracy = controls.get("shuffled_task_frame_token", {}).get("accuracy")
    randomized_accuracy = controls.get("randomize_recurrent_matrix", {}).get("accuracy")
    random_label_accuracy = controls.get("random_label_control", {}).get("accuracy")
    token_influence = summary.get("token_influence", {})
    dog = token_influence.get("dog_influence_by_frame", {})
    dog_environment = dog.get("environment_frame") or {}
    dog_causal = [
        (dog.get(frame_name) or {}).get("output_change_rate")
        for frame_name in ("danger_frame", "friendship_frame", "sound_frame")
    ]
    dog_causal = [value for value in dog_causal if value is not None]
    dog_environment_rate = dog_environment.get("output_change_rate")
    dog_switch = None
    if dog_causal and dog_environment_rate is not None:
        dog_switch = float(np.mean(dog_causal) - dog_environment_rate)
    dog_probe = summary["feature_group_probe_accuracy_by_step"].get("actor", [])
    mean_actor_switch = token_influence.get("mean_actor_authority_switch_score")

    high_accuracy = accuracy >= 0.85 and min(frame_accuracies) >= 0.80
    recurrence_matters = recurrence_gain >= 0.10
    frame_token_matters = no_frame_accuracy is not None and no_frame_accuracy <= accuracy - 0.10
    shuffled_hurts = shuffled_accuracy is not None and shuffled_accuracy <= accuracy - 0.10
    randomized_hurts = randomized_accuracy is not None and randomized_accuracy <= accuracy - 0.20
    random_fails = random_label_accuracy is None or random_label_accuracy < 0.65
    dog_switches = dog_switch is not None and dog_switch >= 0.15
    actors_switch = mean_actor_switch is not None and mean_actor_switch >= 0.15
    actor_decodable = bool(dog_probe) and dog_probe[-1] >= 0.80

    supports_tokens = (
        high_accuracy
        and recurrence_matters
        and frame_token_matters
        and shuffled_hurts
        and randomized_hurts
        and random_fails
        and dog_switches
        and actors_switch
        and actor_decodable
    )
    if supports_tokens:
        return {
            "supports_multi_aspect_token_refraction": "true",
            "supports_same_token_different_frame_authority": "true",
            "reason": (
                "The model solves the same actor token under danger, friendship, sound, and environment frames. "
                "The actor token remains decodable, dog has higher decision influence in causal actor frames than "
                "in the environment frame, frame-token controls hurt, and randomized recurrence destroys the gain."
            ),
        }

    return {
        "supports_multi_aspect_token_refraction": "unclear",
        "supports_same_token_different_frame_authority": "unclear",
        "reason": (
            f"accuracy={accuracy:.3f}, recurrence_gain={recurrence_gain:.3f}, "
            f"no_frame_accuracy={no_frame_accuracy if no_frame_accuracy is not None else 'n/a'}, "
            f"shuffled_frame_accuracy={shuffled_accuracy if shuffled_accuracy is not None else 'n/a'}, "
            f"randomized_recurrent_accuracy={randomized_accuracy if randomized_accuracy is not None else 'n/a'}, "
            f"dog_authority_switch={dog_switch if dog_switch is not None else 'n/a'}, "
            f"mean_actor_authority_switch={mean_actor_switch if mean_actor_switch is not None else 'n/a'}, "
            f"actor_probe_final={dog_probe[-1] if dog_probe else 'n/a'}."
        ),
    }


def interpret_frame_switch_report(summary: dict[str, Any]) -> dict[str, str]:
    frame_in_accuracy = summary["frame_in_recurrence_only_accuracy"]
    output_only_accuracy = summary["frame_at_output_only_accuracy"]
    initial_only_accuracy = summary["frame_initial_only_accuracy"]
    randomized_accuracy = summary["randomized_recurrent_accuracy"]
    controls = summary["controls"]
    shuffled_accuracy = controls.get("shuffled_task_frame_token", {}).get("accuracy")
    random_label_accuracy = controls.get("random_label_control", {}).get("accuracy")
    switch = summary["mid_run_switch"]
    geometry = summary["trajectory_geometry"]
    interpolation = summary["soft_frame_interpolation"]

    switch_success = switch.get("mid_run_switch_success_rate")
    reorientation = switch.get("reorientation_score")
    convergence = switch.get("target_frame_convergence_after_switch")
    authority_transfer = switch.get("authority_switch_after_frame_switch")
    divergence = geometry.get("trajectory_divergence_by_step", [])
    smoothness = interpolation.get("interpolation_smoothness_score")

    recurrent_beats_output = frame_in_accuracy >= output_only_accuracy + 0.05
    randomized_collapses = randomized_accuracy <= frame_in_accuracy - 0.20
    shuffled_hurts = shuffled_accuracy is not None and shuffled_accuracy <= frame_in_accuracy - 0.10
    random_fails = random_label_accuracy is None or random_label_accuracy < 0.65
    switch_works = switch_success is not None and switch_success >= 0.60
    hidden_reorients = (
        reorientation is not None
        and reorientation > 0.0
        and convergence is not None
        and convergence > 0.0
    )
    geometry_separates = (
        len(divergence) >= 2
        and divergence[-1] >= divergence[0] + 0.02
    )
    interpolation_is_graded = smoothness is not None and smoothness >= 0.50
    authority_switches = authority_transfer is not None and authority_transfer > 0.0

    real_reorientation = (
        recurrent_beats_output
        and randomized_collapses
        and shuffled_hurts
        and random_fails
        and switch_works
        and hidden_reorients
        and geometry_separates
        and interpolation_is_graded
        and authority_switches
    )
    static_routing = output_only_accuracy >= frame_in_accuracy - 0.03
    output_routing_disfavored = output_only_accuracy <= frame_in_accuracy - 0.05
    early_frame_commitment = (
        initial_only_accuracy >= frame_in_accuracy - 0.03
        and (switch_success is not None and switch_success < 0.50)
    )

    if real_reorientation:
        return {
            "real_recurrent_reorientation": "true",
            "static_frame_routing": "false" if not static_routing else "unclear",
            "reason": (
                "Frame-in-recurrence beats the output-only routing baseline, randomized recurrence collapses, "
                "mid-run frame switches move hidden states toward the target-frame trajectory, authority transfers "
                "toward the new active group, and soft frame interpolation gives graded influence."
            ),
        }

    return {
        "real_recurrent_reorientation": "unclear",
        "static_frame_routing": "true" if static_routing else ("false" if output_routing_disfavored else "unclear"),
        "early_frame_commitment": "true" if early_frame_commitment else "unclear",
        "reason": (
            f"frame_in_recurrence_only_accuracy={frame_in_accuracy:.3f}, "
            f"frame_at_output_only_accuracy={output_only_accuracy:.3f}, "
            f"frame_initial_only_accuracy={initial_only_accuracy:.3f}, "
            f"randomized_recurrent_accuracy={randomized_accuracy:.3f}, "
            f"shuffled_frame_accuracy={shuffled_accuracy if shuffled_accuracy is not None else 'n/a'}, "
            f"mid_run_switch_success_rate={switch_success if switch_success is not None else 'n/a'}, "
            f"reorientation_score={reorientation if reorientation is not None else 'n/a'}, "
            f"target_frame_convergence_after_switch={convergence if convergence is not None else 'n/a'}, "
            f"interpolation_smoothness_score={smoothness if smoothness is not None else 'n/a'}, "
            f"authority_switch_after_frame_switch={authority_transfer if authority_transfer is not None else 'n/a'}."
        ),
    }


def interpret_reframe_report(summary: dict[str, Any]) -> dict[str, str]:
    baseline = summary["no_reframe_baseline"]
    trained = summary["trained_reframe"]
    controls = summary["controls"]
    baseline_fixed = baseline["fixed_frame"]["accuracy"]
    baseline_no_reset = baseline["wrong_initial_frame_no_reset"]["final_accuracy_after_reframe"]
    trained_fixed = trained["fixed_frame"]["accuracy"]
    trained_no_reset = trained["wrong_initial_frame_no_reset"]
    trained_with_reset = trained["wrong_initial_frame_with_reset"]
    no_reset_accuracy = trained_no_reset["final_accuracy_after_reframe"]
    with_reset_accuracy = trained_with_reset["final_accuracy_after_reframe"]
    reset_accuracy_gain = with_reset_accuracy - no_reset_accuracy
    reset_success_gain = trained_with_reset["reframe_success_rate"] - trained_no_reset["reframe_success_rate"]
    reset_authority_gain = (
        trained_with_reset["authority_switch_after_frame_switch"]
        - trained_no_reset["authority_switch_after_frame_switch"]
    )
    randomized_accuracy = controls.get("randomize_recurrent_matrix", {}).get("accuracy")
    shuffled_accuracy = controls.get("shuffled_task_frame_token", {}).get("accuracy")
    random_label_accuracy = controls.get("random_label_control", {}).get("accuracy")

    supports_early_commitment = (
        baseline_fixed >= 0.80
        and baseline_no_reset <= baseline_fixed - 0.10
    )
    supports_reset = (
        trained_fixed >= 0.80
        and with_reset_accuracy >= no_reset_accuracy + 0.05
        and trained_with_reset["reframe_success_rate"] >= trained_no_reset["reframe_success_rate"] + 0.15
        and reset_authority_gain > 0.05
        and randomized_accuracy is not None
        and randomized_accuracy <= trained_fixed - 0.20
        and shuffled_accuracy is not None
        and shuffled_accuracy <= trained_fixed - 0.10
        and (random_label_accuracy is None or random_label_accuracy < 0.65)
    )
    free_rotation = no_reset_accuracy >= trained_fixed - 0.05 and trained_no_reset["reframe_success_rate"] >= 0.60

    return {
        "supports_early_frame_commitment": "true" if supports_early_commitment else "unclear",
        "supports_online_reframe_with_reset": "true" if supports_reset else "unclear",
        "supports_free_midrun_rotation_without_reset": "true" if free_rotation else "false",
        "reason": (
            f"baseline_fixed_accuracy={baseline_fixed:.3f}, baseline_wrong_no_reset_accuracy={baseline_no_reset:.3f}, "
            f"trained_fixed_accuracy={trained_fixed:.3f}, no_reset_reframe_accuracy={no_reset_accuracy:.3f}, "
            f"with_reset_reframe_accuracy={with_reset_accuracy:.3f}, reset_accuracy_gain={reset_accuracy_gain:.3f}, "
            f"reset_success_gain={reset_success_gain:.3f}, reset_authority_transfer_gain={reset_authority_gain:.3f}, "
            f"randomized_recurrent_accuracy={randomized_accuracy if randomized_accuracy is not None else 'n/a'}, "
            f"shuffled_frame_accuracy={shuffled_accuracy if shuffled_accuracy is not None else 'n/a'}, "
            f"random_label_accuracy={random_label_accuracy if random_label_accuracy is not None else 'n/a'}."
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


def final_step(value: dict[str, Any], key: str) -> float:
    by_step = value.get(f"{key}_by_step")
    if by_step:
        return float(by_step[-1])
    return float(value.get(key, 0.0))


def markdown_refraction_table(summary: dict[str, Any]) -> str:
    influence = summary.get("influence", {}).get("feature_group_influence_by_frame", {})
    lines = [
        "| Frame | Active group | actor_action | place_noise | light | object |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for frame_name in TASK_FRAMES:
        row = [frame_name, FRAME_ACTIVE_GROUP[frame_name]]
        frame_influence = influence.get(frame_name, {})
        for group in FEATURE_GROUPS:
            item = frame_influence.get(group, {})
            row.append(f"{item.get('output_change_rate', 0.0):.4f}")
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
    return "\n".join(lines)


def markdown_multi_aspect_table(summary: dict[str, Any]) -> str:
    influence = summary.get("influence", {}).get("feature_group_influence_by_frame", {})
    columns = ["actor_action", "actor_relation", "actor_sound", "place_noise", "actor"]
    lines = [
        "| Frame | Active group | actor_action | actor_relation | actor_sound | place_noise | actor token |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for frame_name in MULTI_ASPECT_FRAMES:
        row = [frame_name, MULTI_ASPECT_FRAME_ACTIVE_GROUP[frame_name]]
        frame_influence = influence.get(frame_name, {})
        for group in columns:
            item = frame_influence.get(group, {})
            row.append(f"{item.get('output_change_rate', 0.0):.4f}")
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |"
        )
    return "\n".join(lines)


def markdown_dog_influence_table(summary: dict[str, Any]) -> str:
    dog = summary.get("token_influence", {}).get("dog_influence_by_frame", {})
    lines = [
        "| Frame | output_change_rate | label_change_rate | target_accuracy | mean_abs_label_probability_delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for frame_name in MULTI_ASPECT_FRAMES:
        item = dog.get(frame_name) or {}
        lines.append(
            f"| {frame_name} | {item.get('output_change_rate', 0.0):.4f} | "
            f"{item.get('label_change_rate', 0.0):.4f} | {item.get('target_accuracy', 0.0):.4f} | "
            f"{item.get('mean_abs_label_probability_delta', 0.0):.4f} |"
        )
    return "\n".join(lines)


def write_latent_refraction_finding(
    *,
    report: dict[str, Any],
    out_root: Path,
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("latent_refraction_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    controls = main_summary["controls"]
    influence = main_summary.get("influence", {})
    refraction_curve = influence.get("mean_refraction_index_by_step", [])
    authority = influence.get("authority_switch_score")
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    finding = f"""# Latent Refraction Finding

Source run:

- `{display_report_path}`

## Old Finding Summary

The previous toy finding supported **Recurrent Core Recovery under Entangled Interference**:

- recurrence can recover a task-causal core from entangled core+nuisance input,
- nuisance can remain decodable,
- decision authority can still shift toward the recovered core.

That result did not fully test the prism idea, because each feature group kept the same role across the task.

## New Hypothesis

**Recurrent Latent Refraction / Task-Frame Conditional Core Dominance**

The same observed feature bundle should be reinterpreted depending on a task-frame token. A feature group that is nuisance in one frame should become causal core in another frame. The recurrent loop should reorient the hidden state so the active task-core gains decision authority while inactive groups remain decodable but output-inert.

## Frame Task Setup

Every base observation is evaluated under three frames:

- `danger_frame`: label depends on `actor_action`.
- `environment_frame`: label depends on `place_noise`.
- `visibility_frame`: label depends on `light`.

Object features are included as an always-inactive distractor in this v6 version. This keeps the first prism test small and falsifiable.

## Accuracy Results

- input mode: `{main_summary["input_mode"]}`
- overall accuracy: `{main_summary["accuracy"]:.6f}`
- zero-recurrent accuracy: `{controls["zero_recurrent_update"]["accuracy"]:.6f}`
- recurrence gain: `{main_summary["recurrence_gain"]:.6f}`
- no-frame-token accuracy: `{controls.get("no_task_frame_token", {}).get("accuracy", "n/a")}`
- shuffled-frame-token accuracy: `{controls.get("shuffled_task_frame_token", {}).get("accuracy", "n/a")}`
- randomized-recurrent accuracy: `{controls.get("randomize_recurrent_matrix", {}).get("accuracy", "n/a")}`
- random-label accuracy: `{controls.get("random_label_control", {}).get("accuracy", "n/a")}`
- same-observation label diversity: `{main_summary["same_observation_label_diversity"]:.6f}`

Accuracy by frame:

```json
{json.dumps(round_floats(main_summary["accuracy_by_frame"]), indent=2)}
```

## Influence Table

Final-step output-change rate when swapping each feature group while holding the rest fixed:

{markdown_refraction_table(main_summary)}

## Refraction Index

Definition:

```text
refraction_index_by_step = active_core_output_change_rate - max(inactive_group_output_change_rate)
```

Mean refraction index by step:

```json
{json.dumps(round_floats(refraction_curve), indent=2)}
```

Authority switch score:

```json
{json.dumps(round_floats(influence.get("authority_switch_score_by_group", {})), indent=2)}
```

Mean authority switch score: `{authority if authority is not None else "n/a"}`

## Controls

- `zero_recurrent_update`: tests whether recurrence is carrying the frame-conditioned computation.
- `no_task_frame_token`: tests whether identical observations with different frame labels are unsolvable without the frame.
- `shuffled_task_frame_token`: tests whether the trained model actually uses the frame token.
- `freeze_after_1/2/3`: tests whether recurrent depth matters.
- `randomize_recurrent_matrix`: tests whether the learned recurrent matrix carries the useful dynamic.
- `random_label_control`: sanity check against memorizing arbitrary frame labels.

## Interpretation

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, full VRAXION behavior, production architecture validation, clean nuisance erasure, or biological equivalence.

Safe claim if positive:

> In a controlled toy setting, the recurrent loop can reorient an entangled representation according to a task frame, giving decision authority to different feature groups without necessarily erasing the others.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    finding_path = out_root / "LATENT_REFRACTION_FINDING.md"
    run_finding_path = run_dir / "LATENT_REFRACTION_FINDING.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_latent_refraction(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    experiments: dict[str, Any] = {}
    modes = selected_refraction_input_modes(args.input_mode)
    for input_mode in modes:
        name = f"latent_refraction_{input_mode}"
        full_controls = input_mode == "entangled"
        runs = [
            run_refraction_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
                full_controls=full_controls,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_refraction_runs(name, runs)

    main_summary = experiments.get("latent_refraction_entangled") or next(iter(experiments.values()))
    interpretation = interpret_refraction_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "recurrent_latent_refraction_v6",
        "hypothesis": "Recurrent Latent Refraction / Task-Frame Conditional Core Dominance",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "The exact same observation is duplicated under multiple task-frame tokens with different labels.",
            "The key metric is decision authority, not feature deletion.",
            "Inactive feature groups may remain decodable; positive evidence requires low inactive influence.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "latent_refraction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_latent_refraction_finding(
        report=report,
        out_root=out_root,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "accuracy": round_floats(main_summary["accuracy"]),
            "accuracy_by_frame": round_floats(main_summary["accuracy_by_frame"]),
            "recurrence_gain": round_floats(main_summary["recurrence_gain"]),
            "mean_refraction_index_by_step": round_floats(
                main_summary.get("influence", {}).get("mean_refraction_index_by_step", [])
            ),
            "authority_switch_score": round_floats(
                main_summary.get("influence", {}).get("authority_switch_score")
            ),
            "topology": round_floats(main_summary.get("topology", {})),
            "hub_diagnostics": round_floats(main_summary.get("hub_diagnostics", {})),
            "controls": {
                key: round_floats(value["accuracy"])
                for key, value in main_summary["controls"].items()
                if isinstance(value, dict) and "accuracy" in value
            },
            "pointer_diagnostics": round_floats(
                {
                    key: value
                    for key, value in main_summary.get("pointer_diagnostics", {}).items()
                    if key.endswith("_drop")
                    or key
                    in {
                        "mean_pointer_distance_between_frames",
                        "neuron_phase_specialization_score",
                    }
                }
            ),
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def write_multi_aspect_finding(
    *,
    report: dict[str, Any],
    out_root: Path,
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("multi_aspect_refraction_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    controls = main_summary["controls"]
    influence = main_summary.get("influence", {})
    token_influence = main_summary.get("token_influence", {})
    refraction_curve = influence.get("mean_refraction_index_by_step", [])
    dog = token_influence.get("dog_influence_by_frame", {})
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    finding = f"""# Multi-Aspect Token Refraction Finding

Source run:

- `{display_report_path}`

## Question

This probe asks whether a token such as `dog` behaves as a fixed label or as a multi-aspect token whose decision role changes under a task frame.

The intended non-contradictory mappings are:

- `dog + bite + danger_frame -> danger`
- `dog + owner/play + friendship_frame -> friend`
- `dog + bark + sound_frame -> sound_source`
- `dog + street/noise + environment_frame -> actor is distractor`

## Setup

Every base observation contains the same actor token plus action, relation, sound, place/noise, and object features. The same observation is evaluated under four frames:

- `danger_frame`: `actor_action` is causal.
- `friendship_frame`: `actor_relation` is causal.
- `sound_frame`: `actor_sound` is causal.
- `environment_frame`: `place_noise` is causal and actor is distractor.

## Main Result

- input mode: `{main_summary["input_mode"]}`
- accuracy: `{main_summary["accuracy"]:.6f}`
- zero-recurrent accuracy: `{controls["zero_recurrent_update"]["accuracy"]:.6f}`
- recurrence gain: `{main_summary["recurrence_gain"]:.6f}`
- no-frame-token accuracy: `{controls.get("no_task_frame_token", {}).get("accuracy", "n/a")}`
- shuffled-frame-token accuracy: `{controls.get("shuffled_task_frame_token", {}).get("accuracy", "n/a")}`
- randomized-recurrent accuracy: `{controls.get("randomize_recurrent_matrix", {}).get("accuracy", "n/a")}`
- random-label accuracy: `{controls.get("random_label_control", {}).get("accuracy", "n/a")}`
- same-observation label diversity: `{main_summary["same_observation_label_diversity"]:.6f}`

Accuracy by frame:

```json
{json.dumps(round_floats(main_summary["accuracy_by_frame"]), indent=2)}
```

## Feature Group Influence

Final-step output-change rate when swapping feature groups while holding the rest fixed:

{markdown_multi_aspect_table(main_summary)}

## Dog Token Influence

Actor-token intervention: keep the rest of the observation fixed and replace `dog` with another actor token.

{markdown_dog_influence_table(main_summary)}

Actor-token authority switch scores:

```json
{json.dumps(round_floats(token_influence.get("authority_switch_score_by_actor", {})), indent=2)}
```

Mean actor authority switch score: `{token_influence.get("mean_actor_authority_switch_score")}`

Actor/dog decodability by step:

```json
{json.dumps(round_floats(main_summary["feature_group_probe_accuracy_by_step"].get("actor", [])), indent=2)}
```

## Refraction Index

Mean refraction index by step:

```json
{json.dumps(round_floats(refraction_curve), indent=2)}
```

Authority switch score by active feature group:

```json
{json.dumps(round_floats(influence.get("authority_switch_score_by_group", {})), indent=2)}
```

## Interpretation

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, full VRAXION behavior, biological equivalence, production validation, or that the full VRAXION architecture already implements this.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    finding_path = out_root / "MULTI_ASPECT_REFRACTION_FINDING.md"
    run_finding_path = run_dir / "MULTI_ASPECT_REFRACTION_FINDING.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def interpret_inferred_frame_report(main_summary: dict[str, Any]) -> dict[str, Any]:
    frame_acc = float(main_summary["frame_prediction_accuracy"])
    oracle = float(main_summary["task_accuracy_oracle_frame"])
    predicted = float(main_summary["task_accuracy_predicted_frame"])
    frame_head = float(main_summary["task_accuracy_frame_head_only"])
    no_frame = float(main_summary["task_accuracy_no_frame"])
    wrong = float(main_summary["task_accuracy_wrong_forced_frame"])
    refraction = main_summary.get("refraction_index_predicted_frame")
    authority = main_summary.get("authority_switch_score_predicted_frame")
    close_to_oracle = oracle - predicted <= 0.08
    pointer_beats_head = predicted - frame_head >= 0.04
    pointer_beats_no_frame = predicted - no_frame >= 0.04
    wrong_hurts = predicted - wrong >= 0.10
    positive = frame_acc >= 0.85 and close_to_oracle and pointer_beats_no_frame and wrong_hurts
    return {
        "supports_inferred_frame_pointer": bool(positive),
        "supports_frame_prediction": bool(frame_acc >= 0.85),
        "supports_pointer_used_for_authority": bool(pointer_beats_head and wrong_hurts),
        "reason": (
            f"frame_acc={frame_acc:.3f}, oracle={oracle:.3f}, predicted={predicted:.3f}, "
            f"frame_head_only={frame_head:.3f}, no_frame={no_frame:.3f}, wrong_forced={wrong:.3f}, "
            f"refraction={refraction}, authority={authority}."
        ),
    }


def write_inferred_frame_finding(
    *,
    report: dict[str, Any],
    out_root: Path,
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("inferred_frame_pointer_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    token_inventory = main_summary.get("token_frame_inventory", {}).get("token_frame_inventory", {})
    dog = token_inventory.get("dog", {}).get("output_change_rate_by_frame", {})
    finding = f"""# Inferred Frame Pointer Finding

Source run:

- `{display_report_path}`

## Question

This probe moves from explicit frame-token control to inferred frame selection:

```text
input bundle -> predicted frame pointer -> recurrent decision pass
```

The task reuses the existing multi-aspect setup:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`

No new semantic concepts are added. The compromise is that the intended frame is made inferable by feature-group salience inside the input bundle rather than by an explicit frame token.

## Main Results

- frame prediction accuracy: `{main_summary["frame_prediction_accuracy"]:.6f}`
- oracle-frame task accuracy: `{main_summary["task_accuracy_oracle_frame"]:.6f}`
- predicted-frame task accuracy: `{main_summary["task_accuracy_predicted_frame"]:.6f}`
- predicted-frame soft-pointer task accuracy: `{main_summary["task_accuracy_predicted_frame_soft"]:.6f}`
- frame-head-only task accuracy: `{main_summary["task_accuracy_frame_head_only"]:.6f}`
- no-frame baseline task accuracy: `{main_summary["task_accuracy_no_frame"]:.6f}`
- wrong-forced-frame task accuracy: `{main_summary["task_accuracy_wrong_forced_frame"]:.6f}`
- zero-recurrent task accuracy: `{main_summary["task_accuracy_zero_recurrent"]:.6f}`
- randomized-recurrent task accuracy: `{main_summary["task_accuracy_randomized_recurrent"]:.6f}`
- authority-switch score, predicted frame: `{main_summary["authority_switch_score_predicted_frame"]}`
- refraction index final, predicted frame: `{main_summary["refraction_index_predicted_frame"]}`

## Accuracy By Frame

Predicted-frame pointer:

```json
{json.dumps(round_floats(main_summary["predicted_frame_pointer"]["accuracy_by_frame"]), indent=2)}
```

Frame prediction by frame:

```json
{json.dumps(round_floats(main_summary["predicted_frame_pointer"]["frame_prediction_accuracy_by_frame"]), indent=2)}
```

## Token-Frame Inventory

Selected token output-change rates by inferred/target frame:

```json
{json.dumps(round_floats({key: value.get("output_change_rate_by_frame", {}) for key, value in token_inventory.items()}), indent=2)}
```

Dog influence by frame:

```json
{json.dumps(round_floats(dog), indent=2)}
```

## Interpretation

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or that the main VRAXION architecture already performs inferred frame selection.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    finding_path = out_root / "INFERRED_FRAME_POINTER_FINDING.md"
    run_finding_path = run_dir / "INFERRED_FRAME_POINTER_FINDING.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_inferred_frame_pointer(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"inferred_frame_pointer_{input_mode}"
        runs = [
            run_inferred_frame_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_inferred_frame_runs(name, runs)

    main_summary = experiments.get("inferred_frame_pointer_entangled") or next(iter(experiments.values()))
    interpretation = interpret_inferred_frame_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "inferred_frame_pointer_v1",
        "hypothesis": "Input-inferred frame pointer for frame-conditioned authority switching",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "The intended frame is inferable from feature-group salience, not from an explicit frame token.",
            "The predicted hard frame is used as the recurrent pointer at evaluation; a soft-pointer score is also reported.",
            "The key controls are frame_head_only, no_frame_baseline, and wrong_forced_frame.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "inferred_frame_pointer_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_inferred_frame_finding(
        report=report,
        out_root=out_root,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "frame_prediction_accuracy": round_floats(main_summary["frame_prediction_accuracy"]),
            "task_accuracy_oracle_frame": round_floats(main_summary["task_accuracy_oracle_frame"]),
            "task_accuracy_predicted_frame": round_floats(main_summary["task_accuracy_predicted_frame"]),
            "task_accuracy_frame_head_only": round_floats(main_summary["task_accuracy_frame_head_only"]),
            "task_accuracy_no_frame": round_floats(main_summary["task_accuracy_no_frame"]),
            "task_accuracy_wrong_forced_frame": round_floats(main_summary["task_accuracy_wrong_forced_frame"]),
            "authority_switch_score_predicted_frame": round_floats(
                main_summary["authority_switch_score_predicted_frame"]
            ),
            "refraction_index_predicted_frame": round_floats(main_summary["refraction_index_predicted_frame"]),
            "dog_influence_by_inferred_frame": round_floats(
                main_summary.get("token_frame_inventory", {}).get("dog_influence_by_inferred_frame", {})
            ),
            "controls": {
                "zero_recurrent": round_floats(main_summary["task_accuracy_zero_recurrent"]),
                "randomized_recurrent": round_floats(main_summary["task_accuracy_randomized_recurrent"]),
                "random_label": round_floats(
                    (main_summary.get("random_label_control") or {}).get("accuracy")
                ),
            },
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def interpret_query_cued_frame_report(main_summary: dict[str, Any]) -> dict[str, Any]:
    frame_acc = float(main_summary["frame_prediction_accuracy"])
    oracle = float(main_summary["oracle_frame_accuracy"])
    predicted = float(main_summary["predicted_frame_pointer_accuracy"])
    query_head = float(main_summary["query_head_only_accuracy"])
    no_pointer = float(main_summary["no_pointer_query_baseline_accuracy"])
    no_query = float(main_summary["no_query_baseline_accuracy"])
    wrong = float(main_summary["wrong_forced_frame_accuracy"])
    query_ablation = float(main_summary["query_ablation_accuracy"])
    query_shuffle = float(main_summary["query_shuffle_accuracy"])
    randomized = float(main_summary["randomized_recurrent_accuracy"])
    random_label = main_summary.get("random_label_accuracy")
    pointer_authority = main_summary.get("authority_switch_score")
    direct_authority = main_summary.get("no_pointer_authority_switch_score")
    pointer_refraction = main_summary.get("refraction_index_final")
    direct_refraction = main_summary.get("no_pointer_refraction_index_final")
    close_to_oracle = oracle - predicted <= 0.08
    no_query_low = no_query <= predicted - 0.15
    wrong_hurts = wrong <= predicted - 0.10
    query_matters = query_ablation <= predicted - 0.10 and query_shuffle <= predicted - 0.10
    random_fails = random_label is None or random_label < 0.65
    recurrence_matters = randomized <= predicted - 0.20
    pointer_beats_head = predicted >= query_head + 0.04
    pointer_matches_direct = predicted >= no_pointer - 0.02
    cleaner_authority = (
        pointer_authority is not None
        and direct_authority is not None
        and pointer_authority >= direct_authority + 0.03
    )
    cleaner_refraction = (
        pointer_refraction is not None
        and direct_refraction is not None
        and pointer_refraction >= direct_refraction + 0.03
    )
    supports = (
        frame_acc >= 0.85
        and close_to_oracle
        and no_query_low
        and wrong_hurts
        and query_matters
        and random_fails
        and recurrence_matters
        and pointer_matches_direct
        and (pointer_beats_head or cleaner_authority or cleaner_refraction)
    )
    if supports:
        pointer_specific = "true" if cleaner_authority or cleaner_refraction or pointer_beats_head else "unclear"
    elif pointer_matches_direct and not cleaner_authority and not cleaner_refraction:
        pointer_specific = "false"
    else:
        pointer_specific = "unclear"
    if pointer_matches_direct and (cleaner_authority or cleaner_refraction):
        geometry_read = "query cue can solve the label, but explicit internal frame pointer produces cleaner decision-authority geometry"
    elif pointer_matches_direct:
        geometry_read = "pointer-specific necessity is not supported in this toy; query conditioning alone is sufficient"
    else:
        geometry_read = "query-conditioned direct baseline and pointer path are not matched cleanly enough for a geometry verdict"
    return {
        "supports_query_cued_frame_pointer": bool(supports),
        "supports_query_frame_prediction": bool(frame_acc >= 0.85),
        "supports_pointer_specific_authority": pointer_specific,
        "geometry_read": geometry_read,
        "reason": (
            f"frame_acc={frame_acc:.3f}, oracle={oracle:.3f}, predicted={predicted:.3f}, "
            f"query_head_only={query_head:.3f}, no_pointer_query={no_pointer:.3f}, no_query={no_query:.3f}, "
            f"wrong_forced={wrong:.3f}, query_ablation={query_ablation:.3f}, query_shuffle={query_shuffle:.3f}, "
            f"randomized={randomized:.3f}, random_label={random_label if random_label is not None else 'n/a'}, "
            f"pointer_authority={pointer_authority}, direct_authority={direct_authority}, "
            f"pointer_refraction={pointer_refraction}, direct_refraction={direct_refraction}."
        ),
    }


def query_authority_table(summary: dict[str, Any]) -> str:
    pointer = summary.get("pointer_influence", {})
    direct = summary.get("no_pointer_influence", {})
    rows = [
        "| Path | Refraction Final | Authority Switch |",
        "|---|---:|---:|",
        (
            f"| predicted pointer | `{summary.get('refraction_index_final')}` "
            f"| `{summary.get('authority_switch_score')}` |"
        ),
        (
            f"| no-pointer query baseline | `{summary.get('no_pointer_refraction_index_final')}` "
            f"| `{summary.get('no_pointer_authority_switch_score')}` |"
        ),
    ]
    active_pointer = pointer.get("active_core_influence_by_step", {})
    active_direct = direct.get("active_core_influence_by_step", {})
    if active_pointer and active_direct:
        rows.extend([
            "",
            "Final active-group influence by frame:",
            "",
            "| Frame | Predicted Pointer | No-Pointer Query |",
            "|---|---:|---:|",
        ])
        for frame_name in summary["task_frames"]:
            pointer_curve = active_pointer.get(frame_name, [])
            direct_curve = active_direct.get(frame_name, [])
            rows.append(
                f"| `{frame_name}` | `{pointer_curve[-1] if pointer_curve else None}` "
                f"| `{direct_curve[-1] if direct_curve else None}` |"
            )
    return "\n".join(rows)


def write_query_cued_frame_finding(
    *,
    report: dict[str, Any],
    out_root: Path,
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("query_cued_frame_pointer_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    token_inventory = main_summary.get("token_frame_inventory", {}).get("token_frame_inventory", {})
    token_rates = {key: value.get("output_change_rate_by_frame", {}) for key, value in token_inventory.items()}
    finding = f"""# Query-Cued Frame Pointer Finding

Source run:

- `{display_report_path}`

## Question

This probe tests whether a query-like toy goal cue can select an internal frame pointer:

```text
same observation + query cue -> predicted internal frame -> recurrent decision pass
```

The query cues are toy vectors, not natural language. They are separate from frame embeddings and imply a frame through supervision.

## Setup

Query cues:

```json
{json.dumps(main_summary["query_to_frame"], indent=2)}
```

Each base observation is duplicated under all four query cues, so the same observed feature bundle can require different labels.

Same-observation label diversity: `{main_summary["same_observation_label_diversity"]:.6f}`

## Accuracy And Controls

| Path | Accuracy |
|---|---:|
| oracle frame | `{main_summary["oracle_frame_accuracy"]:.6f}` |
| predicted frame pointer | `{main_summary["predicted_frame_pointer_accuracy"]:.6f}` |
| predicted soft pointer | `{main_summary["predicted_frame_pointer_soft_accuracy"]:.6f}` |
| query head only | `{main_summary["query_head_only_accuracy"]:.6f}` |
| no-pointer query baseline | `{main_summary["no_pointer_query_baseline_accuracy"]:.6f}` |
| no-query baseline | `{main_summary["no_query_baseline_accuracy"]:.6f}` |
| wrong forced frame | `{main_summary["wrong_forced_frame_accuracy"]:.6f}` |
| query ablation | `{main_summary["query_ablation_accuracy"]:.6f}` |
| query shuffle | `{main_summary["query_shuffle_accuracy"]:.6f}` |
| zero recurrent | `{main_summary["zero_recurrent_accuracy"]:.6f}` |
| randomized recurrent | `{main_summary["randomized_recurrent_accuracy"]:.6f}` |
| random label | `{main_summary.get("random_label_accuracy")}` |

Frame prediction accuracy: `{main_summary["frame_prediction_accuracy"]:.6f}`

Predicted-frame confusion matrix:

```json
{json.dumps(round_floats(main_summary["query_predicted_frame_pointer"]["predicted_frame_confusion_matrix"]), indent=2)}
```

## Pointer-Specific Authority Test

{query_authority_table(main_summary)}

## Token-Frame Inventory

Selected token output-change rates by query-implied frame:

```json
{json.dumps(round_floats(token_rates), indent=2)}
```

## Interpretation

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    finding_path = out_root / "QUERY_CUED_FRAME_POINTER_FINDING.md"
    run_finding_path = run_dir / "QUERY_CUED_FRAME_POINTER_FINDING.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_query_cued_frame_pointer(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"query_cued_frame_pointer_{input_mode}"
        runs = [
            run_query_cued_frame_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_query_cued_frame_runs(name, runs)

    main_summary = experiments.get("query_cued_frame_pointer_entangled") or next(iter(experiments.values()))
    interpretation = interpret_query_cued_frame_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "query_cued_frame_pointer_v1",
        "hypothesis": "Query-cued internal frame pointer for frame-conditioned authority switching",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "Query cues are toy goal vectors, not natural language understanding.",
            "The same base observation is duplicated under multiple query cues with different labels.",
            "Accuracy is not decisive; pointer-specific authority/refraction geometry is the main diagnostic.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "query_cued_frame_pointer_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_query_cued_frame_finding(
        report=report,
        out_root=out_root,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "frame_prediction_accuracy": round_floats(main_summary["frame_prediction_accuracy"]),
            "oracle_frame_accuracy": round_floats(main_summary["oracle_frame_accuracy"]),
            "predicted_frame_pointer_accuracy": round_floats(main_summary["predicted_frame_pointer_accuracy"]),
            "query_head_only_accuracy": round_floats(main_summary["query_head_only_accuracy"]),
            "no_pointer_query_baseline_accuracy": round_floats(main_summary["no_pointer_query_baseline_accuracy"]),
            "no_query_baseline_accuracy": round_floats(main_summary["no_query_baseline_accuracy"]),
            "wrong_forced_frame_accuracy": round_floats(main_summary["wrong_forced_frame_accuracy"]),
            "query_ablation_accuracy": round_floats(main_summary["query_ablation_accuracy"]),
            "query_shuffle_accuracy": round_floats(main_summary["query_shuffle_accuracy"]),
            "authority_switch_score": round_floats(main_summary["authority_switch_score"]),
            "refraction_index_final": round_floats(main_summary["refraction_index_final"]),
            "no_pointer_authority_switch_score": round_floats(main_summary["no_pointer_authority_switch_score"]),
            "no_pointer_refraction_index_final": round_floats(main_summary["no_pointer_refraction_index_final"]),
            "same_observation_label_diversity": round_floats(main_summary["same_observation_label_diversity"]),
            "controls": {
                "zero_recurrent": round_floats(main_summary["zero_recurrent_accuracy"]),
                "randomized_recurrent": round_floats(main_summary["randomized_recurrent_accuracy"]),
                "random_label": round_floats(main_summary.get("random_label_accuracy")),
            },
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def run_multi_aspect_refraction(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"multi_aspect_refraction_{input_mode}"
        runs = [
            run_multi_aspect_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_refraction_runs(name, runs)

    main_summary = experiments.get("multi_aspect_refraction_entangled") or next(iter(experiments.values()))
    interpretation = interpret_multi_aspect_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "multi_aspect_token_refraction_v1",
        "hypothesis": "Same token, different frame, different decision authority",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "Actor tokens are intentionally reused across frames instead of being assigned direct contradictory labels.",
            "The dog token is measured as decodable while its decision influence changes by frame.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "multi_aspect_refraction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_multi_aspect_finding(
        report=report,
        out_root=out_root,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "accuracy": round_floats(main_summary["accuracy"]),
            "accuracy_by_frame": round_floats(main_summary["accuracy_by_frame"]),
            "recurrence_gain": round_floats(main_summary["recurrence_gain"]),
            "mean_refraction_index_by_step": round_floats(
                main_summary.get("influence", {}).get("mean_refraction_index_by_step", [])
            ),
            "authority_switch_score": round_floats(
                main_summary.get("influence", {}).get("authority_switch_score")
            ),
            "topology": round_floats(main_summary.get("topology", {})),
            "hub_diagnostics": round_floats(main_summary.get("hub_diagnostics", {})),
            "dog_influence_by_frame": round_floats(
                main_summary.get("token_influence", {}).get("dog_influence_by_frame", {})
            ),
            "actor_authority_switch": round_floats(
                main_summary.get("token_influence", {}).get("authority_switch_score_by_actor", {})
            ),
            "controls": {
                key: round_floats(value["accuracy"])
                for key, value in main_summary["controls"].items()
                if isinstance(value, dict) and "accuracy" in value
            },
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def write_frame_switch_diagnostic(
    *,
    report: dict[str, Any],
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("frame_switch_diagnostics_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    controls = main_summary["controls"]
    switch = main_summary["mid_run_switch"]
    geometry = main_summary["trajectory_geometry"]
    interpolation = main_summary["soft_frame_interpolation"]
    dog = main_summary["token_influence"].get("dog_influence_by_frame", {})
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    placement_rows = []
    for placement in FRAME_PLACEMENTS:
        accuracy = main_summary["placement_results"][placement]["main"]["accuracy"]
        placement_rows.append(f"| `{placement}` | {accuracy:.6f} |")

    finding = f"""# Frame Switch Refraction Diagnostic

Source run:

- `{display_report_path}`

## Question

This diagnostic tests whether **Multi-Aspect Token Refraction** is real recurrent reorientation or mostly static frame-token routing.

The setup reuses the existing multi-aspect task:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`
- same actor tokens, especially `dog`
- entangled input

No new semantic concepts are added.

## Architecture Diagnostic

Frame placement comparison:

| Frame placement mode | accuracy |
|---|---:|
{chr(10).join(placement_rows)}

Main comparison:

- frame-in-recurrence-only accuracy: `{main_summary["frame_in_recurrence_only_accuracy"]:.6f}`
- frame-at-output-only accuracy: `{main_summary["frame_at_output_only_accuracy"]:.6f}`
- recurrent-vs-output-only gain: `{main_summary["recurrent_vs_output_only_gain"]:.6f}`
- no-frame accuracy: `{main_summary["no_frame_accuracy"]:.6f}`
- zero-recurrent accuracy: `{controls["zero_recurrent_update"]["accuracy"]:.6f}`
- randomized-recurrent accuracy: `{main_summary["randomized_recurrent_accuracy"]:.6f}`
- shuffled-frame-token accuracy: `{controls["shuffled_task_frame_token"]["accuracy"]:.6f}`
- random-label accuracy: `{controls.get("random_label_control", {}).get("accuracy", "n/a")}`

## Mid-Run Frame Switch

Switch pairs:

```json
{json.dumps([f"{left}->{right}" for left, right in FRAME_SWITCH_PAIRS], indent=2)}
```

Aggregate switch metrics:

- mid-run switch success rate: `{switch.get("mid_run_switch_success_rate")}`
- reorientation score: `{switch.get("reorientation_score")}`
- target-frame convergence after switch: `{switch.get("target_frame_convergence_after_switch")}`
- authority switch after frame switch: `{switch.get("authority_switch_after_frame_switch")}`

Definition notes:

- `reorientation_score = final_distance_to_old_source_trajectory - final_distance_to_new_target_trajectory`.
- Positive means the switched hidden state ends closer to the direct target-frame trajectory than to the old source-frame trajectory.
- `authority_switch_after_frame_switch` compares new-active-group influence against old-active-group influence after the switch.

## Hidden Trajectory Geometry

Trajectory divergence by step:

```json
{json.dumps(round_floats(geometry.get("trajectory_divergence_by_step", [])), indent=2)}
```

Frame clustering accuracy by step:

```json
{json.dumps(round_floats(geometry.get("frame_clustering_accuracy_by_step", [])), indent=2)}
```

Final frame clustering accuracy: `{geometry.get("final_frame_clustering_accuracy")}`

## Soft Frame Interpolation

Interpolation pair:

```json
{json.dumps(interpolation.get("pair", {}), indent=2)}
```

Interpolation smoothness score: `{interpolation.get("interpolation_smoothness_score")}`

Mix summary:

```json
{json.dumps(round_floats(interpolation.get("mixes", {})), indent=2)}
```

## Dog Influence By Direct Frame

```json
{json.dumps(round_floats(dog), indent=2)}
```

## Verdict

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy diagnostic only. Do not claim consciousness, biology, full VRAXION behavior, or production architecture validation.
"""
    docs_dir = ROOT / "docs" / "research"
    docs_dir.mkdir(parents=True, exist_ok=True)
    finding_path = docs_dir / "FRAME_SWITCH_REFRACTION_DIAGNOSTIC.md"
    run_finding_path = run_dir / "FRAME_SWITCH_REFRACTION_DIAGNOSTIC.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_frame_switch_diagnostics(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"frame_switch_diagnostics_{input_mode}"
        runs = [
            run_frame_switch_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_frame_switch_runs(name, runs)

    main_summary = experiments.get("frame_switch_diagnostics_entangled") or next(iter(experiments.values()))
    interpretation = interpret_frame_switch_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "frame_switch_refraction_diagnostic_v1",
        "hypothesis": "Frame-conditioned token authority should require recurrent hidden-state reorientation, not only output routing.",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This diagnostic reuses the multi-aspect setup and adds no new semantic concepts.",
            "The output-only frame placement is the static routing baseline.",
            "Mid-run frame switches test whether hidden trajectories move toward the new target frame.",
            "This is toy evidence only. It does not prove consciousness.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "frame_switch_refraction_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_frame_switch_diagnostic(
        report=report,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "frame_in_recurrence_only_accuracy": round_floats(main_summary["frame_in_recurrence_only_accuracy"]),
            "frame_at_output_only_accuracy": round_floats(main_summary["frame_at_output_only_accuracy"]),
            "frame_initial_only_accuracy": round_floats(main_summary["frame_initial_only_accuracy"]),
            "no_frame_accuracy": round_floats(main_summary["no_frame_accuracy"]),
            "recurrent_vs_output_only_gain": round_floats(main_summary["recurrent_vs_output_only_gain"]),
            "recurrent_vs_zero_gain": round_floats(main_summary["recurrent_vs_zero_gain"]),
            "randomized_recurrent_accuracy": round_floats(main_summary["randomized_recurrent_accuracy"]),
            "mid_run_switch_success_rate": round_floats(
                main_summary["mid_run_switch"].get("mid_run_switch_success_rate")
            ),
            "reorientation_score": round_floats(main_summary["mid_run_switch"].get("reorientation_score")),
            "target_frame_convergence_after_switch": round_floats(
                main_summary["mid_run_switch"].get("target_frame_convergence_after_switch")
            ),
            "authority_switch_after_frame_switch": round_floats(
                main_summary["mid_run_switch"].get("authority_switch_after_frame_switch")
            ),
            "trajectory_divergence_by_step": round_floats(
                main_summary["trajectory_geometry"].get("trajectory_divergence_by_step", [])
            ),
            "interpolation_smoothness_score": round_floats(
                main_summary["soft_frame_interpolation"].get("interpolation_smoothness_score")
            ),
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def write_reframe_diagnostic(
    *,
    report: dict[str, Any],
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("reframe_diagnostics_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    baseline = main_summary["no_reframe_baseline"]
    trained = main_summary["trained_reframe"]
    controls = main_summary["controls"]
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)

    finding = f"""# Reframe Trigger Diagnostic

Source run:

- `{display_report_path}`

## Question

This diagnostic tests the updated hypothesis:

> The toy system may commit to a frame early, then run recurrent simulation inside that frame. If the initial frame is wrong, recovery may require an explicit reframe/reset signal.

The setup reuses the existing multi-aspect token task:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`
- same actor tokens, especially `dog`
- entangled input

No new semantic concepts are added. The reset pulse is a control signal, not a new world feature.

## Main Metrics

- no-reframe fixed accuracy: `{baseline["fixed_frame"]["accuracy"]:.6f}`
- no-reframe wrong-initial/no-reset accuracy: `{baseline["wrong_initial_frame_no_reset"]["final_accuracy_after_reframe"]:.6f}`
- no-reframe wrong-initial/with-reset accuracy: `{baseline["wrong_initial_frame_with_reset"]["final_accuracy_after_reframe"]:.6f}`
- trained-reframe fixed accuracy: `{trained["fixed_frame"]["accuracy"]:.6f}`
- trained-reframe wrong-initial/no-reset accuracy: `{trained["wrong_initial_frame_no_reset"]["final_accuracy_after_reframe"]:.6f}`
- trained-reframe wrong-initial/with-reset accuracy: `{trained["wrong_initial_frame_with_reset"]["final_accuracy_after_reframe"]:.6f}`
- reset accuracy gain: `{main_summary["reset_accuracy_gain"]:.6f}`
- reset success gain: `{main_summary["reset_success_gain"]:.6f}`
- authority transfer gain: `{main_summary["authority_transfer_gain"]:.6f}`

## Recovery By Switch Step

No reset:

```json
{json.dumps(round_floats(trained["wrong_initial_frame_no_reset"].get("recovery_by_switch_step", {})), indent=2)}
```

With reset:

```json
{json.dumps(round_floats(trained["wrong_initial_frame_with_reset"].get("recovery_by_switch_step", {})), indent=2)}
```

## Authority Transfer

No reset:

- old-frame authority decay: `{trained["wrong_initial_frame_no_reset"].get("old_frame_authority_decay")}`
- new-frame authority rise: `{trained["wrong_initial_frame_no_reset"].get("new_frame_authority_rise")}`
- authority switch after frame switch: `{trained["wrong_initial_frame_no_reset"].get("authority_switch_after_frame_switch")}`
- target-frame convergence after switch: `{trained["wrong_initial_frame_no_reset"].get("target_frame_convergence_after_switch")}`

With reset:

- old-frame authority decay: `{trained["wrong_initial_frame_with_reset"].get("old_frame_authority_decay")}`
- new-frame authority rise: `{trained["wrong_initial_frame_with_reset"].get("new_frame_authority_rise")}`
- authority switch after frame switch: `{trained["wrong_initial_frame_with_reset"].get("authority_switch_after_frame_switch")}`
- target-frame convergence after switch: `{trained["wrong_initial_frame_with_reset"].get("target_frame_convergence_after_switch")}`
- reset entropy spike: `{trained["wrong_initial_frame_with_reset"].get("reset_entropy_spike")}`

## Controls

- zero-recurrent accuracy: `{controls["zero_recurrent_update"]["accuracy"]:.6f}`
- threshold-only accuracy: `{controls["threshold_only"]["accuracy"]:.6f}`
- shuffled-frame-token accuracy: `{controls["shuffled_task_frame_token"]["accuracy"]:.6f}`
- randomized-recurrent accuracy: `{controls["randomize_recurrent_matrix"]["accuracy"]:.6f}`
- random-label accuracy: `{controls.get("random_label_control", {}).get("accuracy", "n/a")}`

## Verdict

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy diagnostic only. Do not claim consciousness, biology, full VRAXION behavior, or production architecture validation.
"""
    docs_dir = ROOT / "docs" / "research"
    docs_dir.mkdir(parents=True, exist_ok=True)
    finding_path = docs_dir / "REFRAME_TRIGGER_DIAGNOSTIC.md"
    run_finding_path = run_dir / "REFRAME_TRIGGER_DIAGNOSTIC.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_reframe_diagnostics(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"reframe_diagnostics_{input_mode}"
        runs = [
            run_reframe_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_reframe_runs(name, runs)

    main_summary = experiments.get("reframe_diagnostics_entangled") or next(iter(experiments.values()))
    interpretation = interpret_reframe_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "reframe_trigger_diagnostic_v1",
        "hypothesis": "Early frame commitment may need an explicit reset/reframe signal to recover from an initial wrong frame.",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This diagnostic reuses the multi-aspect setup and adds no new semantic concepts.",
            "The reset pulse is a control signal that can reopen the frame decision.",
            "This is toy evidence only. It does not prove consciousness.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "reframe_trigger_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_reframe_diagnostic(
        report=report,
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "no_reframe_baseline_accuracy": round_floats(main_summary["no_reframe_baseline_accuracy"]),
            "trained_reframe_accuracy": round_floats(main_summary["trained_reframe_accuracy"]),
            "wrong_initial_no_reset_accuracy": round_floats(main_summary["wrong_initial_no_reset_accuracy"]),
            "wrong_initial_with_reset_accuracy": round_floats(main_summary["wrong_initial_with_reset_accuracy"]),
            "reset_accuracy_gain": round_floats(main_summary["reset_accuracy_gain"]),
            "reset_success_gain": round_floats(main_summary["reset_success_gain"]),
            "authority_transfer_gain": round_floats(main_summary["authority_transfer_gain"]),
            "controls": {
                key: round_floats(value["accuracy"])
                for key, value in main_summary["controls"].items()
                if isinstance(value, dict) and "accuracy" in value
            },
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def interpret_query_cued_pointer_bottleneck_report(main_summary: dict[str, Any]) -> dict[str, Any]:
    frame_acc = float(main_summary["frame_prediction_accuracy"])
    oracle = float(main_summary["oracle_frame_pointer_accuracy"])
    predicted = float(main_summary["predicted_frame_pointer_accuracy"])
    full_direct = float(main_summary["full_query_direct_accuracy"])
    no_query = float(main_summary["no_query_baseline_accuracy"])
    wrong = float(main_summary["wrong_forced_frame_accuracy"])
    query_ablation = float(main_summary["query_ablation_accuracy"])
    query_shuffle = float(main_summary["query_shuffle_accuracy"])
    randomized = float(main_summary["randomized_recurrent_accuracy"])
    random_label = main_summary.get("random_label_accuracy")
    bottleneck_acc = {
        int(size): float(value)
        for size, value in main_summary["bottleneck_query_direct_accuracy_by_size"].items()
    }
    bottleneck_authority = {
        int(size): None if value is None else float(value)
        for size, value in main_summary["authority_vs_bottleneck_size"].items()
    }
    bottleneck_refraction = {
        int(size): None if value is None else float(value)
        for size, value in main_summary["refraction_vs_bottleneck_size"].items()
    }
    small_sizes = [size for size in (2, 4) if size in bottleneck_acc]
    small_best_acc = max((bottleneck_acc[size] for size in small_sizes), default=max(bottleneck_acc.values()))
    small_best_authority = max(
        (bottleneck_authority[size] for size in small_sizes if bottleneck_authority[size] is not None),
        default=None,
    )
    small_best_refraction = max(
        (bottleneck_refraction[size] for size in small_sizes if bottleneck_refraction[size] is not None),
        default=None,
    )
    best_bottleneck_acc = max(bottleneck_acc.values())
    pointer_authority = main_summary.get("authority_switch_score")
    pointer_refraction = main_summary.get("refraction_index_final")
    close_to_oracle = oracle - predicted <= 0.08
    direct_shortcut_present = full_direct >= predicted - 0.03
    no_query_weak = no_query <= predicted - 0.10
    wrong_hurts = wrong <= predicted - 0.10
    query_matters = query_ablation <= predicted - 0.10 and query_shuffle <= predicted - 0.10
    random_fails = random_label is None or random_label < 0.65
    recurrence_matters = randomized <= predicted - 0.15
    pointer_beats_small_bottleneck_accuracy = predicted >= small_best_acc + 0.03
    pointer_beats_small_bottleneck_authority = (
        pointer_authority is not None
        and small_best_authority is not None
        and pointer_authority >= small_best_authority + 0.03
    )
    pointer_beats_small_bottleneck_refraction = (
        pointer_refraction is not None
        and small_best_refraction is not None
        and pointer_refraction >= small_best_refraction + 0.03
    )
    best_bottleneck_matches_pointer = best_bottleneck_acc >= predicted - 0.02
    compact_control = (
        frame_acc >= 0.85
        and close_to_oracle
        and recurrence_matters
        and random_fails
        and (pointer_beats_small_bottleneck_accuracy or pointer_beats_small_bottleneck_authority or pointer_beats_small_bottleneck_refraction)
    )
    strict_positive = (
        compact_control
        and no_query_weak
        and wrong_hurts
        and query_matters
        and predicted >= best_bottleneck_acc - 0.02
    )
    if strict_positive:
        pointer_necessity = "true"
    elif best_bottleneck_matches_pointer and not (
        pointer_beats_small_bottleneck_authority or pointer_beats_small_bottleneck_refraction
    ):
        pointer_necessity = "false"
    else:
        pointer_necessity = "unclear"
    if compact_control and direct_shortcut_present:
        read = "full direct query conditioning can shortcut the task, but the pointer acts as a more compact control channel than small direct-query bottlenecks"
    elif compact_control:
        read = "query pointer looks useful as a compact control channel under bottleneck"
    elif best_bottleneck_matches_pointer:
        read = "direct query conditioning remains sufficient under this bottleneck setting"
    else:
        read = "bottleneck comparison is inconclusive"
    return {
        "supports_query_frame_prediction": bool(frame_acc >= 0.85),
        "supports_pointer_as_compact_control": bool(compact_control),
        "supports_pointer_specific_necessity": pointer_necessity,
        "direct_query_shortcut_present": bool(direct_shortcut_present),
        "strict_positive": bool(strict_positive),
        "geometry_read": read,
        "reason": (
            f"frame_acc={frame_acc:.3f}, oracle={oracle:.3f}, predicted={predicted:.3f}, "
            f"full_direct={full_direct:.3f}, no_query={no_query:.3f}, wrong_forced={wrong:.3f}, "
            f"query_ablation={query_ablation:.3f}, query_shuffle={query_shuffle:.3f}, "
            f"randomized={randomized:.3f}, random_label={random_label if random_label is not None else 'n/a'}, "
            f"small_best_bottleneck={small_best_acc:.3f}, best_bottleneck={best_bottleneck_acc:.3f}, "
            f"pointer_authority={pointer_authority}, small_best_authority={small_best_authority}, "
            f"pointer_refraction={pointer_refraction}, small_best_refraction={small_best_refraction}."
        ),
    }


def bottleneck_sweep_table(main_summary: dict[str, Any]) -> str:
    rows = [
        "| Bottleneck | Accuracy | Authority Switch | Refraction Final | Active Influence | Inactive Influence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in main_summary["query_bottleneck_sizes"]:
        key = str(size)
        rows.append(
            f"| `{size}` | `{main_summary['bottleneck_query_direct_accuracy_by_size'].get(key)}` "
            f"| `{main_summary['authority_vs_bottleneck_size'].get(key)}` "
            f"| `{main_summary['refraction_vs_bottleneck_size'].get(key)}` "
            f"| `{main_summary['active_influence_vs_bottleneck_size'].get(key)}` "
            f"| `{main_summary['inactive_influence_vs_bottleneck_size'].get(key)}` |"
        )
    return "\n".join(rows)


def pointer_bottleneck_authority_table(main_summary: dict[str, Any]) -> str:
    rows = [
        "| Path | Accuracy | Authority Switch | Refraction Final | Active Influence | Inactive Influence |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| predicted frame pointer | `{main_summary['predicted_frame_pointer_accuracy']}` "
            f"| `{main_summary['authority_switch_score']}` "
            f"| `{main_summary['refraction_index_final']}` "
            f"| `{main_summary['active_group_influence']}` "
            f"| `{main_summary['inactive_group_influence']}` |"
        ),
        (
            f"| full query direct | `{main_summary['full_query_direct_accuracy']}` "
            f"| `{main_summary['full_query_direct_authority_switch_score']}` "
            f"| `{main_summary['full_query_direct_refraction_index_final']}` "
            f"| `{main_summary['full_query_direct_active_group_influence']}` "
            f"| `{main_summary['full_query_direct_inactive_group_influence']}` |"
        ),
    ]
    return "\n".join(rows)


def write_query_cued_pointer_bottleneck_finding(
    *,
    report: dict[str, Any],
    out_root: Path,
    run_dir: Path,
    report_path: Path,
) -> Path:
    main_summary = report["experiments"].get("query_cued_pointer_bottleneck_entangled")
    if main_summary is None:
        main_summary = next(iter(report["experiments"].values()))
    try:
        display_report_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        display_report_path = str(report_path)
    token_inventory = main_summary.get("token_frame_inventory", {}).get("token_frame_inventory", {})
    token_rates = {key: value.get("output_change_rate_by_frame", {}) for key, value in token_inventory.items()}
    finding = f"""# Query-Cued Pointer Bottleneck Finding

Source run:

- `{display_report_path}`

## Question

The previous query-cued probe showed perfect frame prediction and real query dependence, but pointer-specific necessity stayed unclear because direct query-conditioned baselines were close.

This bottleneck probe asks whether the query cue can be compressed into a small internal frame pointer that controls decision authority more efficiently than a direct query path under the same kind of pressure.

## Setup

The same base observation is duplicated under all four toy query cues:

```json
{json.dumps(main_summary["query_to_frame"], indent=2)}
```

Query cues are separate toy vectors, not natural language and not reused frame embeddings.

Same-observation label diversity: `{main_summary["same_observation_label_diversity"]:.6f}`

## Accuracy And Controls

| Path | Accuracy |
|---|---:|
| oracle frame pointer | `{main_summary["oracle_frame_pointer_accuracy"]:.6f}` |
| predicted frame pointer | `{main_summary["predicted_frame_pointer_accuracy"]:.6f}` |
| predicted soft frame pointer | `{main_summary["predicted_soft_frame_pointer_accuracy"]:.6f}` |
| hard discrete predicted pointer | `{main_summary["hard_discrete_predicted_pointer_accuracy"]:.6f}` |
| full query direct | `{main_summary["full_query_direct_accuracy"]:.6f}` |
| frame head only | `{main_summary["frame_head_only_accuracy"]:.6f}` |
| no query baseline | `{main_summary["no_query_baseline_accuracy"]:.6f}` |
| wrong forced frame | `{main_summary["wrong_forced_frame_accuracy"]:.6f}` |
| query ablation | `{main_summary["query_ablation_accuracy"]:.6f}` |
| query shuffle | `{main_summary["query_shuffle_accuracy"]:.6f}` |
| zero recurrent | `{main_summary["zero_recurrent_accuracy"]:.6f}` |
| randomized recurrent | `{main_summary["randomized_recurrent_accuracy"]:.6f}` |
| random label | `{main_summary.get("random_label_accuracy")}` |

Frame prediction accuracy: `{main_summary["frame_prediction_accuracy"]:.6f}`

Predicted-frame confusion matrix:

```json
{json.dumps(round_floats(main_summary["predicted_frame_pointer"]["predicted_frame_confusion_matrix"]), indent=2)}
```

## Bottleneck Sweep

{bottleneck_sweep_table(main_summary)}

## Authority And Refraction

{pointer_bottleneck_authority_table(main_summary)}

## Token-Frame Inventory

Selected token output-change rates by query-implied frame:

```json
{json.dumps(round_floats(token_rates), indent=2)}
```

## Verdict

```json
{json.dumps(report["interpretation"], indent=2)}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    finding_path = out_root / "QUERY_CUED_POINTER_BOTTLENECK_FINDING.md"
    run_finding_path = run_dir / "QUERY_CUED_POINTER_BOTTLENECK_FINDING.md"
    finding_path.write_text(finding, encoding="utf-8")
    run_finding_path.write_text(finding, encoding="utf-8")
    return finding_path


def run_query_cued_pointer_bottleneck(args: argparse.Namespace, run_dir: Path, out_root: Path) -> int:
    schema = build_schema(args.hidden)
    modes = ["entangled"] if args.input_mode == "all" else selected_refraction_input_modes(args.input_mode)
    experiments: dict[str, Any] = {}
    for input_mode in modes:
        name = f"query_cued_pointer_bottleneck_{input_mode}"
        runs = [
            run_query_cued_pointer_bottleneck_seed(
                name=name,
                input_mode=input_mode,
                seed=seed,
                schema=schema,
                args=args,
            )
            for seed in range(args.seeds)
        ]
        experiments[name] = aggregate_query_cued_pointer_bottleneck_runs(name, runs)

    main_summary = experiments.get("query_cued_pointer_bottleneck_entangled") or next(iter(experiments.values()))
    interpretation = interpret_query_cued_pointer_bottleneck_report(main_summary)
    report = {
        "config": {
            "experiment": args.experiment,
            "input_mode": args.input_mode,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "steps": args.steps,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sparse_density": args.sparse_density,
            "topology_mode": args.topology_mode,
            "flywire_graphml": str(args.flywire_graphml),
            "holdout_fraction": args.holdout_fraction,
            "active_value": args.active_value,
            "embed_scale": args.embed_scale,
            "embedding_mode": args.embedding_mode,
            "resonance_mode": args.resonance_mode,
            "frame_scale": args.frame_scale,
            "opponent_strength": args.opponent_strength,
            "update_rate": args.update_rate,
            "delta_scale": args.delta_scale,
            "ridge": args.ridge,
            "random_label_control": args.random_label_control,
            "device": args.device,
            "out_dir": str(run_dir),
        },
        "schema": asdict(schema),
        "seed": 0 if args.seeds == 1 else None,
        "seeds": list(range(args.seeds)),
        "mode": "query_cued_pointer_bottleneck_v1",
        "hypothesis": "Query-like cue compressed into a compact internal frame pointer under direct-query bottleneck pressure",
        "experiments": experiments,
        "interpretation": interpretation,
        "notes": [
            "This is a toy mechanism probe only. It does not prove consciousness.",
            "Query cues are toy goal vectors, not natural language understanding.",
            "Pointer modes route raw query only through the frame head; the recurrent decision path receives queryless observation plus frame pointer.",
            "Accuracy alone is not decisive; compactness and authority/refraction geometry are the main diagnostics.",
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report = round_floats(report)
    report_path = run_dir / "query_cued_pointer_bottleneck_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finding_path = write_query_cued_pointer_bottleneck_finding(
        report=report,
        out_root=ROOT / "docs" / "research",
        run_dir=run_dir,
        report_path=report_path,
    )

    printable = {
        "mode": report["mode"],
        "interpretation": report["interpretation"],
        "main": {
            "frame_prediction_accuracy": round_floats(main_summary["frame_prediction_accuracy"]),
            "oracle_frame_pointer_accuracy": round_floats(main_summary["oracle_frame_pointer_accuracy"]),
            "predicted_frame_pointer_accuracy": round_floats(main_summary["predicted_frame_pointer_accuracy"]),
            "predicted_soft_frame_pointer_accuracy": round_floats(main_summary["predicted_soft_frame_pointer_accuracy"]),
            "hard_discrete_predicted_pointer_accuracy": round_floats(
                main_summary["hard_discrete_predicted_pointer_accuracy"]
            ),
            "full_query_direct_accuracy": round_floats(main_summary["full_query_direct_accuracy"]),
            "bottleneck_query_direct_accuracy_by_size": round_floats(
                main_summary["bottleneck_query_direct_accuracy_by_size"]
            ),
            "frame_head_only_accuracy": round_floats(main_summary["frame_head_only_accuracy"]),
            "no_query_baseline_accuracy": round_floats(main_summary["no_query_baseline_accuracy"]),
            "wrong_forced_frame_accuracy": round_floats(main_summary["wrong_forced_frame_accuracy"]),
            "query_ablation_accuracy": round_floats(main_summary["query_ablation_accuracy"]),
            "query_shuffle_accuracy": round_floats(main_summary["query_shuffle_accuracy"]),
            "authority_switch_score": round_floats(main_summary["authority_switch_score"]),
            "refraction_index_final": round_floats(main_summary["refraction_index_final"]),
            "authority_vs_bottleneck_size": round_floats(main_summary["authority_vs_bottleneck_size"]),
            "refraction_vs_bottleneck_size": round_floats(main_summary["refraction_vs_bottleneck_size"]),
            "controls": {
                "zero_recurrent": round_floats(main_summary["zero_recurrent_accuracy"]),
                "randomized_recurrent": round_floats(main_summary["randomized_recurrent_accuracy"]),
                "random_label": round_floats(main_summary.get("random_label_accuracy")),
            },
        },
    }
    print(json.dumps(printable, indent=2))
    print(f"\nWrote JSON report: {report_path}")
    print(f"Wrote finding: {finding_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested --device {args.device!r}, but CUDA is not available")

    out_root = args.out_dir or DEFAULT_OUT_ROOT
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "latent_refraction":
        return run_latent_refraction(args, run_dir, out_root)
    if args.experiment in {"multi_aspect_refraction", "multi_aspect_token_refraction"}:
        return run_multi_aspect_refraction(args, run_dir, out_root)
    if args.experiment == "frame_switch_diagnostics":
        return run_frame_switch_diagnostics(args, run_dir, out_root)
    if args.experiment == "reframe_diagnostics":
        return run_reframe_diagnostics(args, run_dir, out_root)
    if args.experiment == "inferred_frame_pointer":
        return run_inferred_frame_pointer(args, run_dir, out_root)
    if args.experiment == "query_cued_frame_pointer":
        return run_query_cued_frame_pointer(args, run_dir, out_root)
    if args.experiment == "query_cued_pointer_bottleneck":
        return run_query_cued_pointer_bottleneck(args, run_dir, out_root)

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
        topology_mode=args.topology_mode,
        flywire_graphml=str(args.flywire_graphml),
        holdout_fraction=args.holdout_fraction,
        active_value=args.active_value,
        embed_scale=args.embed_scale,
        embedding_mode=args.embedding_mode,
        resonance_mode=args.resonance_mode,
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
