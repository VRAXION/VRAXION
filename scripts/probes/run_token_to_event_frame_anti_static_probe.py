#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probes import run_state_bottleneck_probe as sb
from scripts.probes import run_token_state_update_vs_latent_probe as base
from scripts.probes import run_token_to_event_frame_hard_contrasts_probe as v2


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "token_to_event_frame_003_anti_static"
CONTRACT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC_RESULT.md"

EVENT_TYPES = v2.EVENT_TYPES
ENTITY_TYPES = v2.ENTITY_TYPES
REF_TYPES = v2.REF_TYPES
QTY_CLASSES = v2.QTY_CLASSES
NONE_ENTITY = v2.NONE_ENTITY
MAX_CLAUSE_LEN = v2.MAX_CLAUSE_LEN

ALL_ARMS = (
    "EVENT_FRAME_ORACLE",
    "DIRECT_STORY_GRU_ANSWER",
    "SEGMENTED_EVENT_FRAME_CLASSIFIER",
    "STORY_TO_EVENT_FRAMES_GRU",
    "BAG_OF_TOKENS_EVENT_FRAME",
    "STATIC_POSITION_EVENT_FRAME",
    "POSITION_ONLY_EVENT_FRAME",
    "SHUFFLED_EVENT_FRAME_TEACHER",
)

HARD_METRICS = (
    "same_token_role_swap_pair_accuracy",
    "target_position_invariance_accuracy",
    "affected_entity_accuracy",
    "passive_active_contrast_accuracy",
    "negation_noop_accuracy",
    "near_miss_noop_accuracy",
    "modal_noop_accuracy",
    "mention_trap_accuracy",
    "null_action_accuracy",
    "noop_no_mutation_accuracy",
    "ghost_reference_accuracy",
    "invalid_restore_accuracy",
    "identity_restore_accuracy",
    "previous_vs_other_accuracy",
    "event_order_contrast_accuracy",
    "same_token_story_pair_accuracy",
    "query_target_accuracy",
    "distractor_resistance",
)


@dataclass(frozen=True)
class TemplateSpec:
    template_family: str
    surface_template: str
    event_type: str
    target_role: str
    observer_role: str | None
    target_position_tag: str
    train_allowed: bool
    heldout_eval: bool
    hardening_round_added: int


@dataclass(frozen=True)
class ClauseMeta:
    template_family: str
    target_position_tag: str
    train_allowed: bool
    heldout_eval: bool
    contrast_group_id: str | None
    hardening_round_added: int
    target_token_index: int | None
    observer_token_index: int | None
    trigger_token_index: int | None


@dataclass(frozen=True)
class ClauseItem:
    text: str
    frame: v2.EventFrame
    meta: ClauseMeta
    tags: tuple[str, ...]


@dataclass(frozen=True)
class StoryExample:
    story_id: str
    row: base.StoryRow
    clauses: tuple[str, ...]
    frames: tuple[v2.EventFrame, ...]
    answer: int
    tags: tuple[str, ...]
    flags: dict[str, int]
    clause_meta: tuple[ClauseMeta, ...]
    contrast_group_id: str | None


@dataclass(frozen=True)
class ClauseExample:
    story_index: int
    clause_index: int
    text: str
    frame: v2.EventFrame
    tags: tuple[str, ...]
    meta: ClauseMeta


@dataclass(frozen=True)
class JobResult:
    round_id: int
    arm: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOKEN_TO_EVENT_FRAME_003 anti-static syntactic shuffle probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--jobs", default="1")
    parser.add_argument("--heartbeat-sec", "--heartbeat-seconds", dest="heartbeat_sec", type=int, default=30)
    parser.add_argument("--adaptive-hardening", action="store_true")
    parser.add_argument("--max-hardening-rounds", type=int, default=4)
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


def set_worker(seed: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    base.set_deterministic(seed)


def safe_job_name(round_id: int, arm: str, seed: int) -> str:
    return f"round_{round_id}__{arm}__seed_{seed}.jsonl"


def tokenise(text: str) -> list[str]:
    return base.tokenise(text)


def bucket_pos(idx: int | None, n: int) -> str:
    if idx is None or idx < 0:
        return "ABSENT"
    q = idx / max(1, n - 1)
    if q <= 0.15:
        return "BEGIN"
    if q <= 0.35:
        return "EARLY"
    if q <= 0.65:
        return "MID"
    if q <= 0.85:
        return "LATE"
    return "END"


def find_token_index(text: str, entity: str | None) -> int | None:
    if entity is None or entity == NONE_ENTITY:
        return None
    tokens = tokenise(text)
    for idx, token in enumerate(tokens):
        if v2.singular(token) == entity:
            return idx
    return None


def trigger_index(text: str) -> int | None:
    triggers = {"stolen", "steal", "stole", "carried", "carries", "away", "back", "returned", "arrives", "see", "sits", "shines", "sleeps", "lies", "count", "many"}
    for idx, token in enumerate(tokenise(text)):
        if token in triggers:
            return idx
    return None


def meta_for(
    text: str,
    spec: TemplateSpec,
    target: str | None,
    observer: str | None,
    contrast_group_id: str | None,
) -> ClauseMeta:
    tokens = tokenise(text)
    return ClauseMeta(
        template_family=spec.template_family,
        target_position_tag=bucket_pos(find_token_index(text, target), len(tokens)),
        train_allowed=spec.train_allowed,
        heldout_eval=spec.heldout_eval,
        contrast_group_id=contrast_group_id,
        hardening_round_added=spec.hardening_round_added,
        target_token_index=find_token_index(text, target),
        observer_token_index=find_token_index(text, observer),
        trigger_token_index=trigger_index(text),
    )


def spec(family: str, template: str, event: str, target_position: str, train: bool, heldout: bool, round_added: int, observer_role: str | None = "observer") -> TemplateSpec:
    return TemplateSpec(family, template, event, "target", observer_role, target_position, train, heldout, round_added)


REMOVE_SPECS = (
    spec("active_simple", "The {observer} watches the {target} get stolen.", "REMOVE", "LATE", True, False, 0),
    spec("basic_passive", "The {target} was stolen while the {observer} watched.", "REMOVE", "BEGIN", True, False, 0),
    spec("fronted_subordinate", "While the {observer} watched, the {target} was stolen.", "REMOVE", "MID", False, True, 1),
    spec("relative_clause", "The {observer} watched as the {target} was stolen.", "REMOVE", "MID", False, True, 1),
    spec("cleft_focus", "It was the {target} that the {observer} watched get stolen.", "REMOVE", "EARLY", False, True, 2),
    spec("target_late", "The {observer} watched as someone stole the {target}.", "REMOVE", "END", False, True, 2),
    spec("distractor_before_target", "Near the {observer}, the {target} was stolen.", "REMOVE", "MID", False, True, 3),
    spec("target_before_distractor", "The {target}, near the {observer}, was stolen.", "REMOVE", "BEGIN", False, True, 3),
    spec("active_carry", "The {observer} carries the {target} away.", "REMOVE", "MID", True, False, 0),
    spec("passive_carry", "The {target} is carried away by the {observer}.", "REMOVE", "BEGIN", False, True, 2),
)

CREATE_SPECS = (
    spec("simple_create", "I see a {target}.", "CREATE", "END", True, False, 0, None),
    spec("simple_create", "I see one more {target}.", "CREATE", "END", True, False, 0, None),
    spec("arrival_create", "A {target} arrives.", "CREATE", "EARLY", True, False, 0, None),
    spec("fronted_create", "Near the {observer}, a {target} arrives.", "CREATE", "LATE", False, True, 1),
    spec("target_before_distractor_create", "A {target} appears near the {observer}.", "CREATE", "EARLY", False, True, 2),
)

NOOP_SPECS = (
    spec("negation_noop", "The {target} was not stolen.", "NOOP_OR_INVALID", "EARLY", True, False, 0, None),
    spec("near_miss_noop", "The {target} was almost stolen.", "NOOP_OR_INVALID", "EARLY", False, True, 3, None),
    spec("modal_noop", "They tried to steal the {target}.", "NOOP_OR_INVALID", "END", False, True, 3, None),
    spec("modal_noop", "They planned to steal the {target}.", "NOOP_OR_INVALID", "END", False, True, 3, None),
    spec("modal_noop", "They failed to steal the {target}.", "NOOP_OR_INVALID", "END", False, True, 3, None),
    spec("mention_trap", "The sign says the {target} was stolen.", "NOOP_OR_INVALID", "MID", False, True, 4, None),
    spec("mention_trap", "The word stolen appears next to the {target}.", "NOOP_OR_INVALID", "END", False, True, 4, None),
    spec("mention_trap", "Someone said the {target} was stolen.", "NOOP_OR_INVALID", "MID", False, True, 4, None),
    spec("null_action", "The {target} sits.", "NOOP_OR_INVALID", "EARLY", True, False, 0, None),
    spec("null_action", "The {target} shines.", "NOOP_OR_INVALID", "EARLY", True, False, 0, None),
    spec("null_action", "The {target} sleeps.", "NOOP_OR_INVALID", "EARLY", True, False, 0, None),
    spec("null_action", "The {target} lies on the table.", "NOOP_OR_INVALID", "EARLY", False, True, 2, None),
)

QUERY_SPECS = (
    spec("simple_query", "How many {target}s?", "QUERY_COUNT", "MID", True, False, 0, None),
    spec("count_query", "Count the {target}s.", "QUERY_COUNT", "END", False, True, 2, None),
    spec("number_query", "What is the number of {target}s?", "QUERY_COUNT", "END", False, True, 3, None),
)


def frame_for(event_type: str, target: str, ref_type: str = "NONE", quantity: int = 0) -> v2.EventFrame:
    if event_type == "CREATE":
        return v2.EventFrame("CREATE", target, "NEW", max(1, quantity))
    if event_type == "QUERY_COUNT":
        return v2.EventFrame("QUERY_COUNT", target, "NONE", 0)
    if event_type == "NOOP_OR_INVALID":
        return v2.EventFrame("NOOP_OR_INVALID", target, "NONE", 0)
    return v2.EventFrame(event_type, target, ref_type, 0)


def clause_from_spec(spec_obj: TemplateSpec, target: str, observer: str | None, tags: tuple[str, ...], group: str | None, ref_type: str = "NONE", quantity: int = 0) -> ClauseItem:
    text = spec_obj.surface_template.format(target=target, observer=observer or target)
    frame = frame_for(spec_obj.event_type, target, ref_type=ref_type, quantity=quantity)
    return ClauseItem(text, frame, meta_for(text, spec_obj, target, observer, group), tags)


def create_clause(entity: str, group: str | None = None) -> ClauseItem:
    spec_obj = CREATE_SPECS[0]
    return clause_from_spec(spec_obj, entity, None, ("simple_clause",), group, quantity=1)


def query_clause(entity: str, heldout: bool, group: str | None = None) -> ClauseItem:
    spec_obj = QUERY_SPECS[1] if heldout else QUERY_SPECS[0]
    return clause_from_spec(spec_obj, entity, None, ("simple_clause",), group)


def make_story(story_id: str, clauses: list[ClauseItem], tags: tuple[str, ...], split: str, group: str | None) -> StoryExample:
    answer, ledger = v2.run_ledger([clause.frame for clause in clauses])
    query_type = base.ENTITY_TO_INDEX.get(ledger.query_target_type or base.ENTITY_TYPES[0], 0)
    counts = []
    for kind in base.ENTITY_TYPES:
        counts.append(sum(1 for entity in ledger.entities if entity.entity_type == kind and entity.present))
    row = base.StoryRow(
        story_id=story_id,
        text=" ".join(clause.text for clause in clauses),
        answer=answer,
        query_type=query_type,
        counts_by_type=tuple(min(count, base.COUNT_CLASSES - 1) for count in counts),
        invalid_restore=ledger.invalid_restore,
        impossible_reference=ledger.impossible_reference,
        ambiguous_reference=ledger.ambiguous_reference,
        tags=tuple(sorted(set(tags))),
        split=split,
    )
    return StoryExample(
        story_id=story_id,
        row=row,
        clauses=tuple(clause.text for clause in clauses),
        frames=tuple(clause.frame for clause in clauses),
        answer=answer,
        tags=tuple(sorted(set(tags))),
        flags={
            "invalid_restore": ledger.invalid_restore,
            "impossible_reference": ledger.impossible_reference,
            "ambiguous_reference": ledger.ambiguous_reference,
            "illegal_frame": ledger.illegal_frame,
        },
        clause_meta=tuple(clause.meta for clause in clauses),
        contrast_group_id=group,
    )


def available_specs(specs: tuple[TemplateSpec, ...], split: str, round_id: int) -> list[TemplateSpec]:
    if split == "train" or split == "id_eval":
        return [item for item in specs if item.train_allowed]
    return [item for item in specs if item.heldout_eval and item.hardening_round_added <= round_id]


def generated_story_pool(seed: int, split: str, round_id: int) -> list[StoryExample]:
    rng = np.random.default_rng(seed + 1000 * (round_id + 1) + (0 if split == "train" else 99))
    entities = list(base.ENTITY_TYPES)
    pool: list[StoryExample] = []
    eval_tag = "in_distribution_hard" if split == "id_eval" else "heldout_template"
    hard_tag = eval_tag if split != "train" else "train"

    remove_specs = available_specs(REMOVE_SPECS, split, round_id)
    noop_specs = available_specs(NOOP_SPECS, split, round_id)

    idx = 0
    for target in entities:
        for observer in entities:
            if target == observer:
                continue
            for spec_obj in remove_specs:
                group = f"{split}_remove_{spec_obj.template_family}_{target}_{observer}_{idx}"
                tags = (hard_tag, "affected_entity", "target_position_invariance")
                if spec_obj.template_family in {"active_simple", "basic_passive", "fronted_subordinate", "relative_clause", "cleft_focus", "target_late", "distractor_before_target", "target_before_distractor"}:
                    tags += ("same_token_role_swap",)
                if "carry" in spec_obj.template_family:
                    tags += ("passive_active_contrast",)
                story = make_story(
                    f"{split}_remove_{idx}",
                    [create_clause(observer, group), create_clause(target, group), clause_from_spec(spec_obj, target, observer, tags, group), query_clause(target, split == "heldout_eval", group)],
                    tags,
                    split,
                    group,
                )
                pool.append(story)
                idx += 1

    for target in entities:
        for spec_obj in noop_specs:
            group = f"{split}_noop_{spec_obj.template_family}_{target}_{idx}"
            tags = (hard_tag, "no_mutation_on_noop")
            if spec_obj.template_family == "negation_noop":
                tags += ("negation_noop",)
            elif spec_obj.template_family == "near_miss_noop":
                tags += ("near_miss_noop",)
            elif spec_obj.template_family == "modal_noop":
                tags += ("modal_noop",)
            elif spec_obj.template_family == "mention_trap":
                tags += ("mention_trap",)
            elif spec_obj.template_family == "null_action":
                tags += ("null_action",)
            pool.append(make_story(f"{split}_noop_{idx}", [create_clause(target, group), clause_from_spec(spec_obj, target, None, tags, group), query_clause(target, split == "heldout_eval", group)], tags, split, group))
            idx += 1

    for target in entities:
        group = f"{split}_ghost_{target}"
        pool.append(
            make_story(
                f"{split}_ghost_{target}",
                [
                    create_clause(target, group),
                    ClauseItem(
                        f"The second {target} is stolen.",
                        v2.EventFrame("REMOVE", target, "SECOND", 0),
                        meta_for(f"The second {target} is stolen.", spec("ghost_reference", "The second {target} is stolen.", "REMOVE", "MID", split != "heldout_eval", split == "heldout_eval", 2, None), target, None, group),
                        ("ghost_reference",),
                    ),
                    query_clause(target, split == "heldout_eval", group),
                ],
                (hard_tag, "ghost_reference"),
                split,
                group,
            )
        )

    for target in entities:
        other = entities[(entities.index(target) + 1) % len(entities)]
        group = f"{split}_distractor_{target}_{other}"
        pool.append(make_story(f"{split}_distractor_{target}_{other}", [create_clause(target, group), create_clause(other, group), clause_from_spec(REMOVE_SPECS[0], target, other, ("query_target", "distractor_resistance"), group), query_clause(other, split == "heldout_eval", group)], (hard_tag, "query_target", "distractor_resistance"), split, group))

    # Hand-built lifecycle contrasts.
    for target in entities:
        group = f"{split}_restore_{target}"
        pool.append(
            make_story(
                f"{split}_restore_invalid_{target}",
                [
                    create_clause(target, group),
                    ClauseItem(
                        "They bring it back.",
                        v2.EventFrame("RESTORE", NONE_ENTITY, "IT", 0),
                        meta_for("They bring it back.", spec("simple_restore", "They bring it back.", "RESTORE", "ABSENT", True, False, 0, None), None, None, group),
                        ("invalid_restore", "identity_restore"),
                    ),
                    query_clause(target, split == "heldout_eval", group),
                ],
                (hard_tag, "invalid_restore", "identity_restore"),
                split,
                group,
            )
        )
        pool.append(
            make_story(
                f"{split}_event_order_{target}_a",
                [
                    create_clause(target, group),
                    ClauseItem(
                        "It is stolen.",
                        v2.EventFrame("REMOVE", NONE_ENTITY, "IT", 0),
                        meta_for("It is stolen.", spec("it_remove", "It is stolen.", "REMOVE", "ABSENT", True, False, 0, None), None, None, group),
                        ("event_order_contrast", "same_token_story_pair", "identity_restore"),
                    ),
                    ClauseItem(
                        "They bring it back.",
                        v2.EventFrame("RESTORE", NONE_ENTITY, "IT", 0),
                        meta_for("They bring it back.", spec("simple_restore", "They bring it back.", "RESTORE", "ABSENT", True, False, 0, None), None, None, group),
                        ("event_order_contrast", "same_token_story_pair", "identity_restore"),
                    ),
                    query_clause(target, split == "heldout_eval", group),
                ],
                (hard_tag, "event_order_contrast", "same_token_story_pair", "identity_restore"),
                split,
                group,
            )
        )
        pool.append(
            make_story(
                f"{split}_event_order_{target}_b",
                [
                    create_clause(target, group),
                    ClauseItem(
                        "They bring it back.",
                        v2.EventFrame("RESTORE", NONE_ENTITY, "IT", 0),
                        meta_for("They bring it back.", spec("simple_restore", "They bring it back.", "RESTORE", "ABSENT", True, False, 0, None), None, None, group),
                        ("event_order_contrast", "same_token_story_pair", "identity_restore"),
                    ),
                    ClauseItem(
                        "It is stolen.",
                        v2.EventFrame("REMOVE", NONE_ENTITY, "IT", 0),
                        meta_for("It is stolen.", spec("it_remove", "It is stolen.", "REMOVE", "ABSENT", True, False, 0, None), None, None, group),
                        ("event_order_contrast", "same_token_story_pair", "identity_restore"),
                    ),
                    query_clause(target, split == "heldout_eval", group),
                ],
                (hard_tag, "event_order_contrast", "same_token_story_pair", "identity_restore"),
                split,
                group,
            )
        )

    rng.shuffle(pool)
    return pool


def repeat_to_count(pool: list[StoryExample], count: int, prefix: str) -> list[StoryExample]:
    if not pool:
        return []
    out: list[StoryExample] = []
    for idx in range(count):
        src = pool[idx % len(pool)]
        row = src.row
        new_row = base.StoryRow(
            story_id=f"{prefix}_{idx}_{src.story_id}",
            text=row.text,
            answer=row.answer,
            query_type=row.query_type,
            counts_by_type=row.counts_by_type,
            invalid_restore=row.invalid_restore,
            impossible_reference=row.impossible_reference,
            ambiguous_reference=row.ambiguous_reference,
            tags=row.tags,
            split=row.split,
        )
        out.append(StoryExample(new_row.story_id, new_row, src.clauses, src.frames, src.answer, src.tags, src.flags, src.clause_meta, src.contrast_group_id))
    return out


def build_probe_dataset(seed: int, train_examples: int, eval_examples: int, round_id: int) -> tuple[list[StoryExample], list[StoryExample]]:
    train_pool = generated_story_pool(seed, "train", round_id)
    id_pool = generated_story_pool(seed + 17, "id_eval", round_id)
    heldout_pool = generated_story_pool(seed + 31, "heldout_eval", round_id)
    train = repeat_to_count(train_pool, train_examples, f"train_r{round_id}")
    id_count = eval_examples // 2
    heldout_count = eval_examples - id_count
    eval_set = repeat_to_count(id_pool, id_count, f"id_eval_r{round_id}") + repeat_to_count(heldout_pool or id_pool, heldout_count, f"heldout_eval_r{round_id}")
    return train, eval_set


def flatten_clauses(stories: list[StoryExample]) -> list[ClauseExample]:
    out: list[ClauseExample] = []
    for story_index, story in enumerate(stories):
        for clause_index, (text, frame, meta) in enumerate(zip(story.clauses, story.frames, story.clause_meta)):
            out.append(ClauseExample(story_index, clause_index, text, frame, story.tags, meta))
    return out


def build_vocab_from_stories(train: list[StoryExample], eval_set: list[StoryExample]) -> dict[str, int]:
    return v2.build_vocab_from_stories(train, eval_set)


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    return v2.encode_texts(texts, vocab, max_len)


def frame_targets(examples: list[ClauseExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2.frame_targets(examples)  # structural typing: examples have .frame


def position_text(example: ClauseExample) -> str:
    meta = example.meta
    tokens = tokenise(example.text)
    return " ".join(
        [
            f"len_{min(len(tokens), 12)}",
            f"target_{bucket_pos(meta.target_token_index, len(tokens))}",
            f"observer_{bucket_pos(meta.observer_token_index, len(tokens))}",
            f"trigger_{bucket_pos(meta.trigger_token_index, len(tokens))}",
            f"clause_{min(example.clause_index, 5)}",
        ]
    )


def group_flat_frames(clause_examples: list[ClauseExample], frames: list[v2.EventFrame], story_count: int) -> list[list[v2.EventFrame]]:
    grouped: list[list[v2.EventFrame]] = [[] for _ in range(story_count)]
    for example, frame in zip(clause_examples, frames):
        grouped[example.story_index].append(frame)
    return grouped


def run_predicted_ledger(stories: list[StoryExample], grouped_frames: list[list[v2.EventFrame]]) -> tuple[list[int], list[v2.LedgerState]]:
    preds: list[int] = []
    ledgers: list[v2.LedgerState] = []
    for frames in grouped_frames:
        answer, ledger = v2.run_ledger(frames)
        preds.append(answer)
        ledgers.append(ledger)
    return preds, ledgers


def accuracy(preds: list[object], gold: list[object]) -> float:
    if not gold:
        return math.nan
    return sum(1 for pred, truth in zip(preds, gold) if pred == truth) / len(gold)


def tag_accuracy(stories: list[StoryExample], preds: list[int], tag: str) -> float:
    values = [pred == story.answer for story, pred in zip(stories, preds) if tag in story.tags]
    return sum(values) / len(values) if values else math.nan


def group_accuracy(stories: list[StoryExample], preds: list[int], required_tag: str | None = None) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for story, pred in zip(stories, preds):
        if story.contrast_group_id is None:
            continue
        if required_tag is not None and required_tag not in story.tags:
            continue
        groups[story.contrast_group_id].append(pred == story.answer)
    if not groups:
        return math.nan
    return sum(1.0 if all(values) else 0.0 for values in groups.values()) / len(groups)


def clause_tag_frame_accuracy(examples: list[ClauseExample], pred_frames: list[v2.EventFrame], tag: str) -> float:
    values = [pred == example.frame for example, pred in zip(examples, pred_frames) if tag in example.tags]
    return sum(values) / len(values) if values else math.nan


def story_metrics(stories: list[StoryExample], preds: list[int], ledgers: list[v2.LedgerState] | None, frame_exact: float, clause_examples: list[ClauseExample] | None = None, pred_frames: list[v2.EventFrame] | None = None) -> dict[str, float]:
    ledger_acc = accuracy(preds, [story.answer for story in stories])
    metrics = {
        "ledger_answer_accuracy": ledger_acc,
        "in_distribution_hard_accuracy": tag_accuracy(stories, preds, "in_distribution_hard"),
        "heldout_template_accuracy": tag_accuracy(stories, preds, "heldout_template"),
        "target_position_invariance_accuracy": group_accuracy(stories, preds, "target_position_invariance"),
        "same_token_role_swap_pair_accuracy": group_accuracy(stories, preds, "same_token_role_swap"),
        "pair_accuracy": group_accuracy(stories, preds, None),
        "per_member_accuracy": ledger_acc,
        "affected_entity_accuracy": tag_accuracy(stories, preds, "affected_entity"),
        "passive_active_contrast_accuracy": tag_accuracy(stories, preds, "passive_active_contrast"),
        "negation_noop_accuracy": tag_accuracy(stories, preds, "negation_noop"),
        "near_miss_noop_accuracy": tag_accuracy(stories, preds, "near_miss_noop"),
        "modal_noop_accuracy": tag_accuracy(stories, preds, "modal_noop"),
        "mention_trap_accuracy": tag_accuracy(stories, preds, "mention_trap"),
        "null_action_accuracy": tag_accuracy(stories, preds, "null_action"),
        "noop_no_mutation_accuracy": tag_accuracy(stories, preds, "no_mutation_on_noop"),
        "ghost_reference_accuracy": tag_accuracy(stories, preds, "ghost_reference"),
        "invalid_restore_accuracy": tag_accuracy(stories, preds, "invalid_restore"),
        "identity_restore_accuracy": tag_accuracy(stories, preds, "identity_restore"),
        "previous_vs_other_accuracy": tag_accuracy(stories, preds, "previous_vs_other"),
        "event_order_contrast_accuracy": tag_accuracy(stories, preds, "event_order_contrast"),
        "same_token_story_pair_accuracy": group_accuracy(stories, preds, "same_token_story_pair"),
        "query_target_accuracy": tag_accuracy(stories, preds, "query_target"),
        "distractor_resistance": tag_accuracy(stories, preds, "distractor_resistance"),
        "coreference_accuracy": tag_accuracy(stories, preds, "coreference"),
        "static_simple_clause_score": tag_accuracy(stories, preds, "simple_clause"),
        "frame_to_answer_gap": frame_exact - ledger_acc if not math.isnan(frame_exact) else math.nan,
    }
    if clause_examples is not None and pred_frames is not None:
        metrics["noop_frame_accuracy"] = np.nanmean(
            [
                clause_tag_frame_accuracy(clause_examples, pred_frames, "negation_noop"),
                clause_tag_frame_accuracy(clause_examples, pred_frames, "near_miss_noop"),
                clause_tag_frame_accuracy(clause_examples, pred_frames, "modal_noop"),
                clause_tag_frame_accuracy(clause_examples, pred_frames, "mention_trap"),
                clause_tag_frame_accuracy(clause_examples, pred_frames, "null_action"),
            ]
        ).item()
    else:
        metrics["noop_frame_accuracy"] = math.nan
    metrics["hard_contrast_score"] = hard_score(metrics)
    return metrics


def hard_score(metrics: dict[str, float]) -> float:
    vals = [float(metrics[name]) for name in HARD_METRICS if name in metrics and not math.isnan(float(metrics[name]))]
    return float(np.mean(vals)) if vals else math.nan


def frame_metrics(gold_frames: list[v2.EventFrame], pred_frames: list[v2.EventFrame]) -> dict[str, object]:
    return v2.frame_metrics(gold_frames, pred_frames)


def failed_cases(stories: list[StoryExample], preds: list[int], limit: int = 10) -> list[dict[str, object]]:
    out = []
    for story, pred in zip(stories, preds):
        if pred != story.answer:
            out.append({"story_id": story.story_id, "text": story.row.text, "gold": story.answer, "pred": pred, "tags": story.tags})
            if len(out) >= limit:
                break
    return out


def feature_leak_audit(stories: list[StoryExample]) -> str:
    forbidden = {"create", "remove", "restore", "query_count", "noop_or_invalid", "distractor_create", "event_type", "entity_type", "ref_type", "target_count", "answer"}
    for story in stories:
        leaked = sorted(set(tokenise(" ".join(story.clauses))) & forbidden)
        if leaked:
            return f"fail:{leaked}"
    return "pass"


def train_direct_answer(train_stories: list[StoryExample], eval_stories: list[StoryExample], seed: int, args: argparse.Namespace, progress_path: Path | None) -> tuple[list[int], int]:
    train_rows = [story.row for story in train_stories]
    eval_rows = [story.row for story in eval_stories]
    vocab = base.build_vocab(train_rows + eval_rows)
    train_seq = base.encode_sequences(train_rows, vocab)
    eval_seq = base.encode_sequences(eval_rows, vocab)
    train_y = torch.tensor([story.answer for story in train_stories], dtype=torch.long)
    model = sb.train_direct_gru(train_seq, train_y, seed, len(vocab), args.embed_dim, args.hidden, args.epochs, args.lr, args.batch_size, progress_path, args.heartbeat_sec)
    with torch.no_grad():
        preds = model(eval_seq).argmax(dim=1).tolist()
    return preds, len(vocab)


def train_clause_arm(arm: str, train_clauses: list[ClauseExample], eval_clauses: list[ClauseExample], vocab: dict[str, int], seed: int, args: argparse.Namespace, progress_path: Path | None) -> list[v2.EventFrame]:
    if arm in {"SEGMENTED_EVENT_FRAME_CLASSIFIER", "SHUFFLED_EVENT_FRAME_TEACHER"}:
        train_texts = [example.text for example in train_clauses]
        eval_texts = [example.text for example in eval_clauses]
        model: nn.Module = v2.ClauseGRUFrameModel(len(vocab), args.embed_dim, args.hidden)
        train_x = encode_texts(train_texts, vocab, MAX_CLAUSE_LEN)
        eval_x = encode_texts(eval_texts, vocab, MAX_CLAUSE_LEN)
    elif arm == "BAG_OF_TOKENS_EVENT_FRAME":
        train_seq = encode_texts([example.text for example in train_clauses], vocab, MAX_CLAUSE_LEN)
        eval_seq = encode_texts([example.text for example in eval_clauses], vocab, MAX_CLAUSE_LEN)
        train_x = base.bag_features(train_seq, len(vocab))
        eval_x = base.bag_features(eval_seq, len(vocab))
        model = v2.FrameMLP(train_x.shape[1], args.hidden)
    elif arm == "STATIC_POSITION_EVENT_FRAME":
        train_seq = encode_texts([example.text for example in train_clauses], vocab, MAX_CLAUSE_LEN)
        eval_seq = encode_texts([example.text for example in eval_clauses], vocab, MAX_CLAUSE_LEN)
        train_x = base.static_position_features(train_seq, len(vocab))
        eval_x = base.static_position_features(eval_seq, len(vocab))
        model = v2.FrameMLP(train_x.shape[1], args.hidden)
    elif arm == "POSITION_ONLY_EVENT_FRAME":
        pos_vocab = {base.PAD: 0, base.UNK: 1}
        train_texts = [position_text(example) for example in train_clauses]
        eval_texts = [position_text(example) for example in eval_clauses]
        for text in train_texts + eval_texts:
            for token in tokenise(text):
                pos_vocab.setdefault(token, len(pos_vocab))
        train_seq = encode_texts(train_texts, pos_vocab, 16)
        eval_seq = encode_texts(eval_texts, pos_vocab, 16)
        train_x = base.bag_features(train_seq, len(pos_vocab))
        eval_x = base.bag_features(eval_seq, len(pos_vocab))
        model = v2.FrameMLP(train_x.shape[1], args.hidden)
    else:
        raise ValueError(f"unknown clause arm: {arm}")
    model = v2.train_frame_model(model, train_x, frame_targets(train_clauses), seed, args.epochs, args.lr, args.batch_size, progress_path, args.heartbeat_sec)
    with torch.no_grad():
        return v2.harden_logits(model(eval_x))


def run_job(round_id: int, arm: str, seed: int, args: argparse.Namespace, progress_root: Path | None) -> JobResult:
    set_worker(seed)
    progress_path = progress_root / safe_job_name(round_id, arm, seed) if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "round": round_id, "arm": arm, "seed": seed})
    train_stories, eval_stories = build_probe_dataset(seed, args.train_examples, args.eval_examples, round_id)
    vocab = build_vocab_from_stories(train_stories, eval_stories)
    audit = feature_leak_audit(train_stories + eval_stories)
    eval_clauses = flatten_clauses(eval_stories)
    gold_flat = [example.frame for example in eval_clauses]

    if arm == "EVENT_FRAME_ORACLE":
        grouped = [list(story.frames) for story in eval_stories]
        preds, ledgers = run_predicted_ledger(eval_stories, grouped)
        fm = frame_metrics(gold_flat, gold_flat)
        metrics = {**fm, **story_metrics(eval_stories, preds, ledgers, fm["event_frame_exact_accuracy"], eval_clauses, gold_flat), "feature_leak_audit": audit, "vocab_size": len(vocab)}
        return JobResult(round_id, arm, seed, metrics, failed_cases(eval_stories, preds))

    if arm == "DIRECT_STORY_GRU_ANSWER":
        preds, vocab_size = train_direct_answer(train_stories, eval_stories, seed, args, progress_path)
        metrics = {
            "event_frame_exact_accuracy": math.nan,
            "event_type_accuracy": math.nan,
            "entity_type_accuracy": math.nan,
            "ref_type_accuracy": math.nan,
            "quantity_accuracy": math.nan,
            "illegal_frame_rate": math.nan,
            **story_metrics(eval_stories, preds, None, math.nan),
            "feature_leak_audit": audit,
            "vocab_size": vocab_size,
        }
        return JobResult(round_id, arm, seed, metrics, failed_cases(eval_stories, preds))

    train_clauses = flatten_clauses(train_stories)
    if arm == "SHUFFLED_EVENT_FRAME_TEACHER":
        perm = torch.randperm(len(train_clauses), generator=torch.Generator().manual_seed(seed + 9101 + round_id)).tolist()
        shuffled = [train_clauses[idx].frame for idx in perm]
        train_clauses = [ClauseExample(ex.story_index, ex.clause_index, ex.text, frame, ex.tags, ex.meta) for ex, frame in zip(train_clauses, shuffled)]

    if arm == "STORY_TO_EVENT_FRAMES_GRU":
        train_seq, train_boundaries = v2.encode_story_sequences(train_stories, vocab)
        eval_seq, eval_boundaries = v2.encode_story_sequences(eval_stories, vocab)
        max_events = train_boundaries.shape[1]
        targets = v2.story_frame_targets(train_stories, max_events)
        model = v2.StoryFrameGRU(len(vocab), args.embed_dim, args.hidden)
        model = v2.train_story_frame_model(model, train_seq, train_boundaries, targets, seed, args.epochs, args.lr, args.batch_size, progress_path, args.heartbeat_sec)
        with torch.no_grad():
            grouped = v2.frames_from_story_logits(model(eval_seq, eval_boundaries), eval_boundaries >= 0)
        pred_flat = [frame for frames in grouped for frame in frames]
    else:
        pred_flat = train_clause_arm(arm, train_clauses, eval_clauses, vocab, seed, args, progress_path)
        grouped = group_flat_frames(eval_clauses, pred_flat, len(eval_stories))

    fm = frame_metrics(gold_flat, pred_flat)
    preds, ledgers = run_predicted_ledger(eval_stories, grouped)
    metrics = {**fm, **story_metrics(eval_stories, preds, ledgers, fm["event_frame_exact_accuracy"], eval_clauses, pred_flat), "feature_leak_audit": audit, "vocab_size": len(vocab)}
    return JobResult(round_id, arm, seed, metrics, failed_cases(eval_stories, preds))


def result_record(result: JobResult) -> dict[str, object]:
    return asdict(result)


def merge_confusions(values: list[object]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for value in values:
        if isinstance(value, dict):
            total.update({str(key): int(count) for key, count in value.items()})
    return dict(sorted(total.items()))


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_arm: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        by_arm[result.arm].append(result)
    out: dict[str, dict[str, object]] = {}
    for arm, rows in by_arm.items():
        metric_names = sorted({name for row in rows for name in row.metrics})
        metrics: dict[str, object] = {}
        for name in metric_names:
            values = [row.metrics.get(name) for row in rows]
            if name.endswith("_confusion"):
                metrics[name] = merge_confusions(values)
            elif all(isinstance(value, (int, float)) and not math.isnan(float(value)) for value in values):
                floats = [float(value) for value in values]
                metrics[name] = {"mean": round(float(np.mean(floats)), 6), "min": round(float(np.min(floats)), 6), "max": round(float(np.max(floats)), 6)}
            elif name == "feature_leak_audit":
                metrics[name] = sorted(set(str(value) for value in values))
            else:
                metrics[name] = values[0] if values else None
        out[arm] = {"arm": arm, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(agg: dict[str, dict[str, object]], arm: str, name: str, default: float = math.nan) -> float:
    try:
        value = agg[arm]["metrics"][name]
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
    except KeyError:
        pass
    return default


def safe_mean(vals: list[float]) -> float:
    clean = [val for val in vals if not math.isnan(val)]
    return float(np.mean(clean)) if clean else math.nan


def arm_hard_score(agg: dict[str, dict[str, object]], arm: str) -> float:
    return safe_mean([metric_mean(agg, arm, name) for name in HARD_METRICS])


def verdict(agg: dict[str, dict[str, object]], round_id: int, max_rounds: int) -> list[str]:
    labels: list[str] = []
    oracle = metric_mean(agg, "EVENT_FRAME_ORACLE", "ledger_answer_accuracy", 0.0)
    if oracle < 0.98:
        return ["LEDGER_UPDATE_BOTTLENECK"]
    audits: list[str] = []
    for data in agg.values():
        audit = data["metrics"].get("feature_leak_audit")
        if isinstance(audit, list):
            audits.extend(audit)
    if set(audits or ["pass"]) != {"pass"}:
        labels.append("TOKEN_TO_EVENT_FRAME_INVALID_AUDIT")

    seg_frame = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "event_frame_exact_accuracy", 0.0)
    seg_ledger = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "ledger_answer_accuracy", 0.0)
    seg_hard = arm_hard_score(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER")
    bag_hard = arm_hard_score(agg, "BAG_OF_TOKENS_EVENT_FRAME")
    static_hard = arm_hard_score(agg, "STATIC_POSITION_EVENT_FRAME")
    pos_hard = arm_hard_score(agg, "POSITION_ONLY_EVENT_FRAME")
    shuffled = metric_mean(agg, "SHUFFLED_EVENT_FRAME_TEACHER", "ledger_answer_accuracy", 1.0)
    heldout = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "heldout_template_accuracy", 0.0)
    in_dist = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "in_distribution_hard_accuracy", 0.0)
    pair = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "pair_accuracy", 0.0)
    ref = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "ref_type_accuracy", 0.0)
    noop = safe_mean([metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", name) for name in ("noop_frame_accuracy", "noop_no_mutation_accuracy", "negation_noop_accuracy", "near_miss_noop_accuracy", "modal_noop_accuracy", "mention_trap_accuracy")])

    if (
        seg_frame >= 0.90
        and seg_ledger >= 0.90
        and in_dist >= 0.85
        and heldout >= 0.85
        and pair >= 0.85
        and ref >= 0.85
        and noop >= 0.85
        and static_hard <= 0.70
        and bag_hard <= 0.70
        and pos_hard <= 0.70
        and shuffled <= 0.50
    ):
        labels.append("EVENT_FRAME_ANTI_STATIC_POSITIVE")
    if static_hard > 0.85 and round_id >= max_rounds:
        labels.append("STILL_STATIC_SOLVABLE")
    if pos_hard > 0.70 or static_hard > 0.85:
        labels.append("POSITION_LEAK_WARNING")
    if in_dist >= 0.85 and heldout < 0.85:
        labels.append("TEMPLATE_GENERALIZATION_WEAK")
    if seg_hard < 0.70:
        labels.append("PARSER_WEAK_UNDER_SHUFFLE")
    if metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "affected_entity_accuracy", 1.0) < 0.85:
        labels.append("ROLE_BINDING_BOTTLENECK")
    if ref < 0.85:
        labels.append("REFERENCE_BINDING_BOTTLENECK")
    if noop < 0.85:
        labels.append("NEGATION_MODAL_BOTTLENECK")
    story = metric_mean(agg, "STORY_TO_EVENT_FRAMES_GRU", "ledger_answer_accuracy", seg_ledger)
    if seg_ledger >= 0.90 and story < seg_ledger - 0.10:
        labels.append("EVENT_FRAME_SEGMENTATION_BOTTLENECK")
    if not labels:
        labels.append("ANTI_STATIC_PARTIAL")
    return labels


def should_continue_hardening(agg: dict[str, dict[str, object]], round_id: int, max_rounds: int) -> bool:
    seg_hard = arm_hard_score(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER")
    static_hard = arm_hard_score(agg, "STATIC_POSITION_EVENT_FRAME")
    if static_hard <= 0.70 and seg_hard >= 0.85:
        return False
    if seg_hard < 0.70:
        return False
    return round_id < max_rounds


def position_leak_audit(stories: list[StoryExample]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for story in stories:
        for frame, meta, clause in zip(story.frames, story.clause_meta, story.clauses):
            tokens = tokenise(clause)
            target_bucket = bucket_pos(meta.target_token_index, len(tokens))
            observer_bucket = bucket_pos(meta.observer_token_index, len(tokens))
            trigger_bucket = bucket_pos(meta.trigger_token_index, len(tokens))
            stats[f"event_type={frame.event_type}"][target_bucket] += 1
            stats[f"entity_type={frame.entity_type}"][target_bucket] += 1
            stats[f"ref_type={frame.ref_type}"][target_bucket] += 1
            stats[f"template_family={meta.template_family}"][target_bucket] += 1
            stats[f"observer={frame.event_type}"][observer_bucket] += 1
            stats[f"trigger={frame.event_type}"][trigger_bucket] += 1
    return {key: dict(value) for key, value in sorted(stats.items())}


def write_template_snapshot(out_dir: Path, round_id: int) -> None:
    specs = list(REMOVE_SPECS + CREATE_SPECS + NOOP_SPECS + QUERY_SPECS)
    write_json(
        out_dir / f"template_pool_snapshot_round_{round_id}.json",
        [
            {**asdict(item), "active_this_round": item.train_allowed or (item.heldout_eval and item.hardening_round_added <= round_id)}
            for item in specs
        ],
    )


def write_frame_curve(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    rows = []
    for arm in sorted(agg):
        rows.append(
            {
                "arm": arm,
                "event_frame_exact_accuracy": metric_mean(agg, arm, "event_frame_exact_accuracy"),
                "ledger_answer_accuracy": metric_mean(agg, arm, "ledger_answer_accuracy"),
                "hard_contrast_score": arm_hard_score(agg, arm),
                "in_distribution_hard_accuracy": metric_mean(agg, arm, "in_distribution_hard_accuracy"),
                "heldout_template_accuracy": metric_mean(agg, arm, "heldout_template_accuracy"),
            }
        )
    write_json(out_dir / "frame_curve.json", rows)


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int, round_id: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg, round_id, args.max_hardening_rounds) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    summary = {
        "status": status,
        "round": round_id,
        "verdict": labels,
        "completed_jobs": len(results),
        "config": {
            "seeds": args.seeds,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "epochs": args.epochs,
            "jobs": jobs,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
            "heartbeat_sec": args.heartbeat_sec,
            "adaptive_hardening": args.adaptive_hardening,
            "max_hardening_rounds": args.max_hardening_rounds,
        },
        "aggregate": agg,
    }
    write_json(out_dir / "summary.json", summary)
    write_frame_curve(out_dir, agg)
    lines = [
        "# TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC Report",
        "",
        f"- Status: `{status}`",
        f"- Round: `{round_id}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Jobs: `{jobs}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Frame | Ledger | Hard | In-Dist | Heldout | Pair | NoopFrame | NoopNoMut | Ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "event_frame_exact_accuracy"),
                metric_mean(agg, arm, "ledger_answer_accuracy"),
                arm_hard_score(agg, arm),
                metric_mean(agg, arm, "in_distribution_hard_accuracy"),
                metric_mean(agg, arm, "heldout_template_accuracy"),
                metric_mean(agg, arm, "pair_accuracy"),
                metric_mean(agg, arm, "noop_frame_accuracy"),
                metric_mean(agg, arm, "noop_no_mutation_accuracy"),
                metric_mean(agg, arm, "ref_type_accuracy"),
            )
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int, completed_jobs: int, round_id: int) -> None:
    lines = [
        "# TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC Result",
        "",
        "## Run",
        "",
        "```text",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        f"completed_jobs={completed_jobs}",
        f"final_round={round_id}",
        f"adaptive_hardening={args.adaptive_hardening}",
        "```",
        "",
        "## Arm Summary",
        "",
        "| Arm | Frame | Ledger | Hard | In-Dist | Heldout | Pair | Ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "event_frame_exact_accuracy"),
                metric_mean(agg, arm, "ledger_answer_accuracy"),
                arm_hard_score(agg, arm),
                metric_mean(agg, arm, "in_distribution_hard_accuracy"),
                metric_mean(agg, arm, "heldout_template_accuracy"),
                metric_mean(agg, arm, "pair_accuracy"),
                metric_mean(agg, arm, "ref_type_accuracy"),
            )
        )
    lines.extend(["", "## Verdict", "", "```json", json.dumps(labels, indent=2), "```", "", "## Claim Boundary", "", "This is a controlled anti-static toy parser probe. It is not full natural-language segmentation, symbol grounding, or a PrismionCell test."])
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_round(args: argparse.Namespace, jobs: int, round_id: int) -> tuple[dict[str, dict[str, object]], list[str], list[JobResult]]:
    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    write_template_snapshot(args.out_dir, round_id)
    sample_train, sample_eval = build_probe_dataset(seeds[0], min(args.train_examples, 16), min(args.eval_examples, 16), round_id)
    write_jsonl(
        args.out_dir / "examples_sample.jsonl",
        [
            {"story_id": story.story_id, "text": story.row.text, "frames": [asdict(frame) for frame in story.frames], "tags": story.tags, "flags": story.flags}
            for story in (sample_train[:6] + sample_eval[:12])
        ],
    )
    write_json(args.out_dir / "position_leak_audit.json", position_leak_audit(sample_train + sample_eval))
    write_jsonl(
        args.out_dir / "hard_contrast_cases.jsonl",
        [
            {"story_id": story.story_id, "text": story.row.text, "frames": [asdict(frame) for frame in story.frames], "tags": story.tags, "group": story.contrast_group_id}
            for story in sample_eval
            if any(tag in story.tags for tag in ("heldout_template", "same_token_role_swap", "target_position_invariance", "event_order_contrast", "no_mutation_on_noop"))
        ],
    )
    queue = [(round_id, arm, seed) for seed in seeds for arm in arms]
    write_json(args.out_dir / "queue.json", [{"round": r, "arm": arm, "seed": seed} for r, arm, seed in queue])
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    append_jsonl(progress_path, {"time": now_iso(), "event": "round_start", "round": round_id, "jobs": jobs, "total_jobs": len(queue)})
    results: list[JobResult] = []
    write_outputs(args.out_dir, results, args, "partial", jobs, round_id)

    if jobs <= 1:
        for _, arm, seed in queue:
            result = run_job(round_id, arm, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, "partial", jobs, round_id)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "round": round_id, "arm": arm, "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_meta = {}
            pending = set()
            for _, arm, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "round": round_id, "arm": arm, "seed": seed})
                future = pool.submit(run_job, round_id, arm, seed, args, job_progress_root)
                future_meta[future] = (arm, seed)
                pending.add(future)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "round": round_id, "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out_dir, results, args, "partial", jobs, round_id)
                    continue
                for future in done:
                    arm, seed = future_meta[future]
                    result = future.result()
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, "partial", jobs, round_id)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "round": round_id, "arm": arm, "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})

    agg, labels = write_outputs(args.out_dir, results, args, "complete", jobs, round_id)
    append_jsonl(progress_path, {"time": now_iso(), "event": "round_complete", "round": round_id, "verdict": labels})
    append_jsonl(
        args.out_dir / "hardening_rounds.jsonl",
        {
            "time": now_iso(),
            "round": round_id,
            "verdict": labels,
            "segmented_hard": arm_hard_score(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER"),
            "static_hard": arm_hard_score(agg, "STATIC_POSITION_EVENT_FRAME"),
            "bag_hard": arm_hard_score(agg, "BAG_OF_TOKENS_EVENT_FRAME"),
            "position_only_hard": arm_hard_score(agg, "POSITION_ONLY_EVENT_FRAME"),
        },
    )
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

    all_round_summaries = []
    final_agg: dict[str, dict[str, object]] = {}
    final_labels: list[str] = []
    final_results: list[JobResult] = []
    max_round = args.max_hardening_rounds if args.adaptive_hardening else 0
    for round_id in range(max_round + 1):
        agg, labels, results = run_round(args, jobs, round_id)
        final_agg, final_labels, final_results = agg, labels, results
        all_round_summaries.append({"round": round_id, "verdict": labels, "segmented_hard": arm_hard_score(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER"), "static_hard": arm_hard_score(agg, "STATIC_POSITION_EVENT_FRAME"), "bag_hard": arm_hard_score(agg, "BAG_OF_TOKENS_EVENT_FRAME"), "position_only_hard": arm_hard_score(agg, "POSITION_ONLY_EVENT_FRAME")})
        write_json(args.out_dir / "per_round_summary.json", all_round_summaries)
        if not args.adaptive_hardening or not should_continue_hardening(agg, round_id, args.max_hardening_rounds):
            break
    write_doc_result(final_agg, final_labels, args, jobs, len(final_results), all_round_summaries[-1]["round"] if all_round_summaries else 0)
    print(json.dumps({"verdict": final_labels, "out": str(args.out_dir), "rounds": all_round_summaries}, indent=2))


if __name__ == "__main__":
    main()
