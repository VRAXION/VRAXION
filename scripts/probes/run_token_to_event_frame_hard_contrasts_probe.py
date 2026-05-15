#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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

from scripts.probes import run_state_bottleneck_probe as sb
from scripts.probes import run_token_state_update_vs_latent_probe as base


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "token_to_event_frame_002_hard_contrasts"
CONTRACT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS_RESULT.md"

EVENT_TYPES = ("CREATE", "REMOVE", "RESTORE", "QUERY_COUNT", "NOOP_OR_INVALID")
ENTITY_TYPES = tuple(base.ENTITY_TYPES) + ("NONE",)
REF_TYPES = ("NEW", "FIRST", "SECOND", "PREVIOUS", "OTHER", "IT", "NONE")
QTY_CLASSES = 5
MAX_CLAUSE_LEN = base.MAX_LEN
MAX_STORY_LEN = 128
CLAUSE_SEP = "clause_sep"
NONE_ENTITY = "NONE"

EVENT_TO_IDX = {name: idx for idx, name in enumerate(EVENT_TYPES)}
ENTITY_TO_IDX = {name: idx for idx, name in enumerate(ENTITY_TYPES)}
REF_TO_IDX = {name: idx for idx, name in enumerate(REF_TYPES)}

ALL_ARMS = (
    "EVENT_FRAME_ORACLE",
    "DIRECT_STORY_GRU_ANSWER",
    "SEGMENTED_EVENT_FRAME_CLASSIFIER",
    "STORY_TO_EVENT_FRAMES_GRU",
    "BAG_OF_TOKENS_EVENT_FRAME",
    "STATIC_POSITION_EVENT_FRAME",
    "SHUFFLED_EVENT_FRAME_TEACHER",
)


@dataclass(frozen=True)
class EventFrame:
    event_type: str
    entity_type: str
    ref_type: str
    quantity: int


@dataclass
class EntityRecord:
    eid: int
    entity_type: str
    present: bool = True
    removed: bool = False
    created_step: int = 0
    last_mentioned_step: int = 0
    removed_step: int | None = None


@dataclass
class LedgerState:
    entities: list[EntityRecord]
    next_eid: int
    step: int
    last_mentioned_eid: int | None
    last_created_eid: int | None
    previous_created_eid_by_type: dict[str, int]
    last_created_eid_by_type: dict[str, int]
    last_removed_eid_by_type: dict[str, int]
    query_target_type: str | None
    impossible_reference: int = 0
    ambiguous_reference: int = 0
    invalid_restore: int = 0
    illegal_frame: int = 0


@dataclass(frozen=True)
class StoryExample:
    story_id: str
    row: base.StoryRow
    clauses: tuple[str, ...]
    frames: tuple[EventFrame, ...]
    answer: int
    tags: tuple[str, ...]
    flags: dict[str, int]


@dataclass(frozen=True)
class ClauseExample:
    story_index: int
    clause_index: int
    text: str
    frame: EventFrame
    tags: tuple[str, ...]


@dataclass(frozen=True)
class JobResult:
    arm: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


class AmbiguousRef:
    pass


AMBIGUOUS = AmbiguousRef()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOKEN_TO_EVENT_FRAME_002 hard contrast raw clause to event-frame probe.")
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


def maybe_progress(path: Path | None, last_write: float, heartbeat_sec: int, row: dict[str, object]) -> float:
    current = time.time()
    if path is not None and current - last_write >= heartbeat_sec:
        append_jsonl(path, {"time": now_iso(), **row})
        return current
    return last_write


def safe_job_name(arm: str, seed: int) -> str:
    return f"{arm}__seed_{seed}.jsonl"


def singular(noun: str) -> str:
    return sb.singular(noun)


def entity_or_none(noun: str | None) -> str:
    if not noun:
        return NONE_ENTITY
    kind = singular(noun)
    return kind if kind in base.ENTITY_TO_INDEX else NONE_ENTITY


def ref_name(ref: str | None) -> str:
    if not ref:
        return "NONE"
    return ref.upper()


def split_story(row: base.StoryRow) -> tuple[list[str], str]:
    return sb.split_story(row.text)


def query_clause(query_kind: str) -> str:
    return f"How many {query_kind}s?"


def parse_clause_to_frame(clause: str) -> EventFrame:
    lower = clause.lower().strip().rstrip(".?")

    m = re.fullmatch(r"how many ([a-z0-9_]+)s", lower)
    if m:
        return EventFrame("QUERY_COUNT", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"i see (?:a|an) ([a-z0-9_]+)", lower)
    if m:
        return EventFrame("CREATE", entity_or_none(m.group(1)), "NEW", 1)

    m = re.fullmatch(r"i see one more ([a-z0-9_]+)", lower)
    if m:
        return EventFrame("CREATE", entity_or_none(m.group(1)), "NEW", 1)

    m = re.fullmatch(r"i see two ([a-z0-9_]+)s", lower)
    if m:
        return EventFrame("CREATE", entity_or_none(m.group(1)), "NEW", 2)

    m = re.fullmatch(r"(?:a|an) ([a-z0-9_]+) arrives", lower)
    if m:
        return EventFrame("CREATE", entity_or_none(m.group(1)), "NEW", 1)

    m = re.fullmatch(r"the ([a-z0-9_]+) (?:sits|shines|sleeps|lies on the table)", lower)
    if m:
        return EventFrame("NOOP_OR_INVALID", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"the (?:sign says|word stolen appears next to|word appears next to) the ([a-z0-9_]+)(?: was stolen)?", lower)
    if m:
        return EventFrame("NOOP_OR_INVALID", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"someone said the ([a-z0-9_]+) was stolen", lower)
    if m:
        return EventFrame("NOOP_OR_INVALID", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"the ([a-z0-9_]+) was (?:not|almost) stolen", lower)
    if m:
        return EventFrame("NOOP_OR_INVALID", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"they (?:tried|planned|failed) to steal the ([a-z0-9_]+)", lower)
    if m:
        return EventFrame("NOOP_OR_INVALID", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"the ([a-z0-9_]+) watches the ([a-z0-9_]+) get stolen", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(2)), "NONE", 0)

    m = re.fullmatch(r"the ([a-z0-9_]+) is near the ([a-z0-9_]+) that disappears", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(2)), "NONE", 0)

    m = re.fullmatch(r"the ([a-z0-9_]+) carries the ([a-z0-9_]+) away", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(2)), "NONE", 0)

    m = re.fullmatch(r"the ([a-z0-9_]+) is carried away by the ([a-z0-9_]+)", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(1)), "NONE", 0)

    m = re.fullmatch(r"the (?:(first|second|previous|other) )?([a-z0-9_]+) (?:gets|is|was) stolen", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(2)), ref_name(m.group(1)), 0)

    m = re.fullmatch(r"the (?:(first|second|previous|other) )?([a-z0-9_]+) vanishes", lower)
    if m:
        return EventFrame("REMOVE", entity_or_none(m.group(2)), ref_name(m.group(1)), 0)

    if lower in {"it gets stolen", "it is stolen"}:
        return EventFrame("REMOVE", NONE_ENTITY, "IT", 0)

    m = re.fullmatch(r"they bring it back(?: as a ([a-z0-9_]+))?", lower)
    if m:
        return EventFrame("RESTORE", entity_or_none(m.group(1)), "IT", 0)

    m = re.fullmatch(r"they bring the (first|second|previous|other) ([a-z0-9_]+) back", lower)
    if m:
        return EventFrame("RESTORE", entity_or_none(m.group(2)), ref_name(m.group(1)), 0)

    m = re.fullmatch(r"they return the (first|second|previous|other) ([a-z0-9_]+)", lower)
    if m:
        return EventFrame("RESTORE", entity_or_none(m.group(2)), ref_name(m.group(1)), 0)

    m = re.fullmatch(r"they found the (first|second|previous|other) ([a-z0-9_]+) again", lower)
    if m:
        return EventFrame("RESTORE", entity_or_none(m.group(2)), ref_name(m.group(1)), 0)

    return EventFrame("NOOP_OR_INVALID", NONE_ENTITY, "NONE", 0)


def empty_ledger() -> LedgerState:
    return LedgerState(
        entities=[],
        next_eid=0,
        step=0,
        last_mentioned_eid=None,
        last_created_eid=None,
        previous_created_eid_by_type={},
        last_created_eid_by_type={},
        last_removed_eid_by_type={},
        query_target_type=None,
    )


def get_entity(ledger: LedgerState, eid: int | None) -> EntityRecord | None:
    if eid is None:
        return None
    return next((entity for entity in ledger.entities if entity.eid == eid), None)


def validate_frame(frame: EventFrame) -> tuple[EventFrame, str | None]:
    if frame.event_type not in EVENT_TO_IDX:
        return EventFrame("NOOP_OR_INVALID", NONE_ENTITY, "NONE", 0), "unknown_event"
    if frame.entity_type not in ENTITY_TO_IDX:
        return EventFrame("NOOP_OR_INVALID", NONE_ENTITY, "NONE", 0), "unknown_entity"
    if frame.ref_type not in REF_TO_IDX:
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, "NONE", 0), "unknown_ref"
    if frame.quantity < 0 or frame.quantity >= QTY_CLASSES:
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, frame.ref_type, 0), "bad_quantity"
    if frame.event_type in {"CREATE", "QUERY_COUNT"} and frame.entity_type == NONE_ENTITY:
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, frame.ref_type, 0), f"{frame.event_type}_missing_entity"
    if frame.event_type in {"REMOVE", "RESTORE"} and frame.ref_type == "NEW":
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, frame.ref_type, 0), f"{frame.event_type}_bad_new_ref"
    if frame.event_type == "CREATE" and frame.ref_type not in {"NEW", "NONE"}:
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, frame.ref_type, 0), "create_bad_ref"
    if frame.event_type != "CREATE" and frame.quantity > 1:
        return EventFrame("NOOP_OR_INVALID", frame.entity_type, frame.ref_type, 0), "quantity_bad_for_event"
    return frame, None


def resolve_ref(ledger: LedgerState, ref_type: str, entity_type: str, event_type: str) -> EntityRecord | AmbiguousRef | None:
    if ref_type == "IT":
        last = get_entity(ledger, ledger.last_mentioned_eid)
        if last is not None and (entity_type == NONE_ENTITY or last.entity_type == entity_type):
            return last
        if event_type == "RESTORE" and entity_type != NONE_ENTITY:
            return get_entity(ledger, ledger.last_removed_eid_by_type.get(entity_type))
        return None

    if entity_type == NONE_ENTITY:
        return None

    candidates = [entity for entity in ledger.entities if entity.entity_type == entity_type]
    candidates_sorted = sorted(candidates, key=lambda item: item.created_step)

    if ref_type == "FIRST":
        return candidates_sorted[0] if candidates_sorted else None
    if ref_type == "SECOND":
        return candidates_sorted[1] if len(candidates_sorted) >= 2 else None
    if ref_type == "PREVIOUS":
        last = get_entity(ledger, ledger.last_mentioned_eid)
        if last is not None and last.entity_type == entity_type:
            return last
        return get_entity(ledger, ledger.previous_created_eid_by_type.get(entity_type))
    if ref_type == "OTHER":
        last = get_entity(ledger, ledger.last_mentioned_eid)
        others = [entity for entity in candidates_sorted if last is None or entity.eid != last.eid]
        if len(others) == 1:
            return others[0]
        return AMBIGUOUS if others else None
    if ref_type == "NONE":
        if len(candidates_sorted) == 1:
            return candidates_sorted[0]
        return AMBIGUOUS if candidates_sorted else None
    return None


def apply_frame(ledger: LedgerState, frame: EventFrame) -> None:
    ledger.step += 1
    frame, invalid_reason = validate_frame(frame)
    if invalid_reason is not None:
        ledger.illegal_frame += 1
        return

    if frame.event_type == "NOOP_OR_INVALID":
        target = resolve_ref(ledger, "NONE", frame.entity_type, frame.event_type) if frame.entity_type != NONE_ENTITY else None
        if isinstance(target, EntityRecord):
            ledger.last_mentioned_eid = target.eid
            target.last_mentioned_step = ledger.step
        return

    if frame.event_type == "CREATE":
        quantity = max(1, frame.quantity)
        for _ in range(quantity):
            if frame.entity_type in ledger.last_created_eid_by_type:
                ledger.previous_created_eid_by_type[frame.entity_type] = ledger.last_created_eid_by_type[frame.entity_type]
            entity = EntityRecord(
                eid=ledger.next_eid,
                entity_type=frame.entity_type,
                created_step=ledger.step,
                last_mentioned_step=ledger.step,
            )
            ledger.next_eid += 1
            ledger.entities.append(entity)
            ledger.last_created_eid = entity.eid
            ledger.last_created_eid_by_type[frame.entity_type] = entity.eid
            ledger.last_mentioned_eid = entity.eid
        return

    if frame.event_type == "QUERY_COUNT":
        ledger.query_target_type = frame.entity_type
        return

    target = resolve_ref(ledger, frame.ref_type, frame.entity_type, frame.event_type)
    if target is AMBIGUOUS:
        ledger.ambiguous_reference += 1
        return
    if target is None:
        ledger.impossible_reference += 1
        return

    if frame.event_type == "REMOVE":
        if not target.present or target.removed:
            ledger.impossible_reference += 1
            return
        target.present = False
        target.removed = True
        target.removed_step = ledger.step
        target.last_mentioned_step = ledger.step
        ledger.last_mentioned_eid = target.eid
        ledger.last_removed_eid_by_type[target.entity_type] = target.eid
        return

    if frame.event_type == "RESTORE":
        if not target.removed or target.present:
            ledger.invalid_restore += 1
            target.last_mentioned_step = ledger.step
            ledger.last_mentioned_eid = target.eid
            return
        target.present = True
        target.removed = False
        target.last_mentioned_step = ledger.step
        ledger.last_mentioned_eid = target.eid


def run_ledger(frames: Iterable[EventFrame]) -> tuple[int, LedgerState]:
    ledger = empty_ledger()
    for frame in frames:
        apply_frame(ledger, frame)
    target = ledger.query_target_type or base.ENTITY_TYPES[0]
    answer = sum(1 for entity in ledger.entities if entity.entity_type == target and entity.present)
    return min(answer, base.COUNT_CLASSES - 1), ledger


def add_manual_stress(split: str) -> list[base.StoryRow]:
    def row(story_id: str, sentences: list[str], query: str, tags: tuple[str, ...]) -> base.StoryRow:
        return base.StoryRow(
            story_id=f"{split}_{story_id}",
            text=" ".join(sentences + [f"How many {query}s?"]),
            answer=0,
            query_type=base.ENTITY_TO_INDEX.get(base.canonical_kind(query), 0),
            counts_by_type=tuple(0 for _ in base.ENTITY_TYPES),
            invalid_restore=0,
            impossible_reference=0,
            ambiguous_reference=0,
            tags=tuple(sorted(set(tags))),
            split=split,
        )

    rows = [
        row("null_dog_sits", ["I see a dog.", "The dog sits."], "dog", ("null_action", "no_mutation_on_noop")),
        row("null_robot_shines", ["I see a robot.", "The robot shines."], "robot", ("null_action", "no_mutation_on_noop")),
        row("null_cat_sleeps", ["I see a cat.", "The cat sleeps."], "cat", ("null_action", "no_mutation_on_noop")),
        row("null_key_lies", ["I see a key.", "The key lies on the table."], "key", ("null_action", "no_mutation_on_noop")),
        row("ghost_second_dog", ["I see a dog.", "The second dog is stolen."], "dog", ("ghost_reference", "impossible_reference", "coreference")),
        row("ghost_second_cat", ["I see a cat.", "The second cat is stolen."], "cat", ("ghost_reference", "impossible_reference", "coreference")),
        row("ghost_other_robot", ["I see a robot.", "The other robot is stolen."], "robot", ("ghost_reference", "impossible_reference", "coreference")),
        row("nested_robot_target", ["I see a dog.", "I see a robot.", "The dog watches the robot get stolen."], "robot", ("nested_target", "coreference")),
        row("same_token_role_robot_stolen", ["I see a dog.", "I see a robot.", "The dog watches the robot get stolen."], "dog", ("same_token_role_swap", "same_token_story_pair", "affected_entity", "invisible_entity_target", "coreference")),
        row("same_token_role_dog_stolen", ["I see a dog.", "I see a robot.", "The robot watches the dog get stolen."], "dog", ("same_token_role_swap", "same_token_story_pair", "affected_entity", "invisible_entity_target", "coreference")),
        row("near_dog_disappears", ["I see a coin.", "I see a dog.", "The coin is near the dog that disappears."], "dog", ("invisible_entity_target", "affected_entity", "coreference")),
        row("active_carries_key", ["I see a robot.", "I see a key.", "The robot carries the key away."], "key", ("passive_active_contrast", "affected_entity", "coreference")),
        row("passive_key_carried", ["I see a robot.", "I see a key.", "The key is carried away by the robot."], "key", ("passive_active_contrast", "affected_entity", "coreference")),
        row("stolen_positive", ["I see a dog.", "The dog was stolen."], "dog", ("negation_pair", "affected_entity", "coreference")),
        row("not_stolen_noop", ["I see a dog.", "The dog was not stolen."], "dog", ("negation_noop", "no_mutation_on_noop")),
        row("almost_stolen_noop", ["I see a dog.", "The dog was almost stolen."], "dog", ("near_miss_noop", "no_mutation_on_noop")),
        row("tried_steal_noop", ["I see a dog.", "They tried to steal the dog."], "dog", ("modal_noop", "no_mutation_on_noop")),
        row("planned_steal_noop", ["I see a dog.", "They planned to steal the dog."], "dog", ("modal_noop", "no_mutation_on_noop")),
        row("failed_steal_noop", ["I see a dog.", "They failed to steal the dog."], "dog", ("modal_noop", "no_mutation_on_noop")),
        row("sign_says_stolen", ["I see a dog.", "The sign says the dog was stolen."], "dog", ("mention_trap", "no_mutation_on_noop")),
        row("word_stolen_appears", ["I see a dog.", "The word stolen appears next to the dog."], "dog", ("mention_trap", "no_mutation_on_noop")),
        row("someone_said_stolen", ["I see a dog.", "Someone said the dog was stolen."], "dog", ("mention_trap", "no_mutation_on_noop")),
        row("restore_without_remove", ["I see a dog.", "They bring it back."], "dog", ("invalid_restore", "identity_restore", "no_mutation_on_noop")),
        row("restore_after_remove", ["I see a dog.", "It is stolen.", "They bring it back."], "dog", ("identity_restore", "coreference")),
        row("previous_restore_valid", ["I see a dog.", "I see one more dog.", "The previous dog is stolen.", "They bring it back."], "dog", ("identity_restore", "previous_vs_other", "coreference")),
        row("previous_other_restore_invalid", ["I see a dog.", "I see one more dog.", "The previous dog is stolen.", "They bring the other dog back."], "dog", ("invalid_restore", "identity_restore", "previous_vs_other", "coreference")),
        row("query_cat_distractor", ["I see a dog.", "I see a cat.", "The dog is stolen."], "cat", ("query_target", "distractor")),
        row("query_dog_removed", ["I see a dog.", "I see a cat.", "The dog is stolen."], "dog", ("query_target", "distractor")),
        row("query_robot_distractor", ["I see two coins.", "A robot arrives.", "The first coin is stolen."], "robot", ("query_target", "distractor", "coreference")),
        row("order_remove_restore", ["I see a dog.", "It is stolen.", "They bring it back."], "dog", ("event_order_contrast", "same_token_story_pair", "identity_restore")),
        row("order_restore_remove", ["I see a dog.", "They bring it back.", "It is stolen."], "dog", ("event_order_contrast", "same_token_story_pair", "invalid_restore")),
        row("first_dog_stolen", ["I see a dog.", "I see one more dog.", "The first dog is stolen."], "dog", ("first_second", "coreference")),
        row("second_dog_stolen", ["I see a dog.", "I see one more dog.", "The second dog is stolen."], "dog", ("first_second", "coreference")),
        row("heldout_found_again", ["I see a key.", "The first key gets stolen.", "They found the first key again."], "key", ("heldout_verb", "coreference")),
    ]
    return rows


def label_story(row: base.StoryRow) -> StoryExample:
    sentences, query = split_story(row)
    clauses = sentences + [query_clause(query)]
    frames = tuple(parse_clause_to_frame(clause) for clause in clauses)
    answer, ledger = run_ledger(frames)
    tags = set(row.tags)
    if any(frame.ref_type not in {"NEW", "NONE"} for frame in frames):
        tags.add("heldout_reference")
    if ledger.impossible_reference:
        tags.add("impossible_reference")
    if ledger.ambiguous_reference:
        tags.add("ambiguous_reference")
    if ledger.invalid_restore:
        tags.add("invalid_restore")
    flags = {
        "impossible_reference": ledger.impossible_reference,
        "ambiguous_reference": ledger.ambiguous_reference,
        "invalid_restore": ledger.invalid_restore,
        "illegal_frame": ledger.illegal_frame,
    }
    return StoryExample(row.story_id, row, tuple(clauses), frames, answer, tuple(sorted(tags)), flags)


def build_probe_dataset(seed: int, train_examples: int, eval_examples: int) -> tuple[list[StoryExample], list[StoryExample]]:
    train_rows, eval_rows = base.build_dataset(seed, train_examples, eval_examples)
    train_rows = train_rows + add_manual_stress("train")
    eval_rows = add_manual_stress("eval") + eval_rows
    train = [label_story(row) for row in train_rows]
    eval_set = [label_story(row) for row in eval_rows[:eval_examples]]
    return train, eval_set


def flatten_clauses(stories: list[StoryExample]) -> list[ClauseExample]:
    out: list[ClauseExample] = []
    for story_idx, story in enumerate(stories):
        for clause_idx, (clause, frame) in enumerate(zip(story.clauses, story.frames)):
            out.append(ClauseExample(story_idx, clause_idx, clause, frame, story.tags))
    return out


def build_vocab_from_stories(train: list[StoryExample], eval_set: list[StoryExample]) -> dict[str, int]:
    tokens = {base.PAD, base.UNK, CLAUSE_SEP}
    for story in train + eval_set:
        for clause in story.clauses:
            tokens.update(base.tokenise(clause))
    for noun in base.HELDOUT_NOUNS:
        tokens.add(noun)
    tokens.update(base.HELDOUT_VERBS)
    return {token: idx for idx, token in enumerate(sorted(tokens))}


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    unk = vocab[base.UNK]
    pad = vocab[base.PAD]
    rows: list[list[int]] = []
    for text in texts:
        ids = [vocab.get(token, unk) for token in base.tokenise(text)[:max_len]]
        ids.extend([pad] * (max_len - len(ids)))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def encode_story_sequences(stories: list[StoryExample], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    unk = vocab[base.UNK]
    pad = vocab[base.PAD]
    sep = vocab[CLAUSE_SEP]
    seq_rows: list[list[int]] = []
    boundary_rows: list[list[int]] = []
    max_events = max(len(story.clauses) for story in stories)
    for story in stories:
        ids: list[int] = []
        boundaries: list[int] = []
        for clause in story.clauses:
            clause_ids = [vocab.get(token, unk) for token in base.tokenise(clause)]
            if not clause_ids:
                clause_ids = [unk]
            for token_id in clause_ids:
                if len(ids) < MAX_STORY_LEN:
                    ids.append(token_id)
            boundaries.append(min(len(ids) - 1, MAX_STORY_LEN - 1))
            if len(ids) < MAX_STORY_LEN:
                ids.append(sep)
        ids = ids[:MAX_STORY_LEN]
        ids.extend([pad] * (MAX_STORY_LEN - len(ids)))
        boundaries.extend([-1] * (max_events - len(boundaries)))
        seq_rows.append(ids)
        boundary_rows.append(boundaries[:max_events])
    return torch.tensor(seq_rows, dtype=torch.long), torch.tensor(boundary_rows, dtype=torch.long)


def frame_targets(examples: list[ClauseExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([EVENT_TO_IDX[item.frame.event_type] for item in examples], dtype=torch.long),
        torch.tensor([ENTITY_TO_IDX[item.frame.entity_type] for item in examples], dtype=torch.long),
        torch.tensor([REF_TO_IDX[item.frame.ref_type] for item in examples], dtype=torch.long),
        torch.tensor([item.frame.quantity for item in examples], dtype=torch.long),
    )


def story_frame_targets(stories: list[StoryExample], max_events: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    event = torch.full((len(stories), max_events), -100, dtype=torch.long)
    entity = torch.full((len(stories), max_events), -100, dtype=torch.long)
    ref = torch.full((len(stories), max_events), -100, dtype=torch.long)
    qty = torch.full((len(stories), max_events), -100, dtype=torch.long)
    mask = torch.zeros((len(stories), max_events), dtype=torch.bool)
    for row_idx, story in enumerate(stories):
        for col_idx, frame in enumerate(story.frames):
            event[row_idx, col_idx] = EVENT_TO_IDX[frame.event_type]
            entity[row_idx, col_idx] = ENTITY_TO_IDX[frame.entity_type]
            ref[row_idx, col_idx] = REF_TO_IDX[frame.ref_type]
            qty[row_idx, col_idx] = frame.quantity
            mask[row_idx, col_idx] = True
    return event, entity, ref, qty, mask


class FrameHeads(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.event = nn.Linear(in_dim, len(EVENT_TYPES))
        self.entity = nn.Linear(in_dim, len(ENTITY_TYPES))
        self.ref = nn.Linear(in_dim, len(REF_TYPES))
        self.qty = nn.Linear(in_dim, QTY_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"event": self.event(x), "entity": self.entity(x), "ref": self.ref(x), "qty": self.qty(x)}


class FrameMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = FrameHeads(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.body(x))


class ClauseGRUFrameModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.heads = FrameHeads(hidden)

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(seq)
        output, _ = self.gru(embedded)
        lengths = (seq != 0).sum(dim=1).clamp(min=1)
        row_idx = torch.arange(seq.shape[0], device=seq.device)
        return self.heads(output[row_idx, lengths - 1])


class StoryFrameGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.heads = FrameHeads(hidden)

    def forward(self, seq: torch.Tensor, boundaries: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(seq)
        output, _ = self.gru(embedded)
        safe_boundaries = boundaries.clamp(min=0)
        batch_idx = torch.arange(seq.shape[0], device=seq.device).unsqueeze(1).expand_as(safe_boundaries)
        gathered = output[batch_idx, safe_boundaries]
        logits = self.heads(gathered)
        return logits


def frame_loss(logits: dict[str, torch.Tensor], targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    event_y, entity_y, ref_y, qty_y = targets
    return (
        1.0 * F.cross_entropy(logits["event"], event_y)
        + 1.0 * F.cross_entropy(logits["entity"], entity_y)
        + 1.5 * F.cross_entropy(logits["ref"], ref_y)
        + 0.5 * F.cross_entropy(logits["qty"], qty_y)
    )


def story_frame_loss(logits: dict[str, torch.Tensor], targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    event_y, entity_y, ref_y, qty_y, mask = targets
    flat_mask = mask.reshape(-1)
    return (
        1.0 * F.cross_entropy(logits["event"].reshape(-1, len(EVENT_TYPES))[flat_mask], event_y.reshape(-1)[flat_mask])
        + 1.0 * F.cross_entropy(logits["entity"].reshape(-1, len(ENTITY_TYPES))[flat_mask], entity_y.reshape(-1)[flat_mask])
        + 1.5 * F.cross_entropy(logits["ref"].reshape(-1, len(REF_TYPES))[flat_mask], ref_y.reshape(-1)[flat_mask])
        + 0.5 * F.cross_entropy(logits["qty"].reshape(-1, QTY_CLASSES)[flat_mask], qty_y.reshape(-1)[flat_mask])
    )


def batches(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(n, start + batch_size))


def train_frame_model(
    model: nn.Module,
    x: torch.Tensor,
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    progress_path: Path | None,
    heartbeat_sec: int,
) -> nn.Module:
    set_worker(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_progress = 0.0
    total_steps = math.ceil(x.shape[0] / batch_size)
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for sl in batches(x.shape[0], batch_size):
            optimizer.zero_grad()
            loss = frame_loss(model(x[sl]), tuple(target[sl] for target in targets))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_progress(progress_path, last_progress, heartbeat_sec, {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)})
        if progress_path is not None:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)})
            last_progress = time.time()
    return model


def train_story_frame_model(
    model: StoryFrameGRU,
    seq: torch.Tensor,
    boundaries: torch.Tensor,
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    progress_path: Path | None,
    heartbeat_sec: int,
) -> StoryFrameGRU:
    set_worker(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_progress = 0.0
    total_steps = math.ceil(seq.shape[0] / batch_size)
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for sl in batches(seq.shape[0], batch_size):
            optimizer.zero_grad()
            loss = story_frame_loss(model(seq[sl], boundaries[sl]), tuple(target[sl] for target in targets))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
            last_progress = maybe_progress(progress_path, last_progress, heartbeat_sec, {"event": "batch", "epoch": epoch + 1, "epochs": epochs, "batch": steps, "batches": total_steps, "loss": epoch_loss / max(1, steps)})
        if progress_path is not None:
            append_jsonl(progress_path, {"time": now_iso(), "event": "epoch", "epoch": epoch + 1, "epochs": epochs, "loss": epoch_loss / max(1, steps)})
            last_progress = time.time()
    return model


def harden_logits(logits: dict[str, torch.Tensor]) -> list[EventFrame]:
    events = logits["event"].argmax(dim=-1).reshape(-1).tolist()
    entities = logits["entity"].argmax(dim=-1).reshape(-1).tolist()
    refs = logits["ref"].argmax(dim=-1).reshape(-1).tolist()
    qtys = logits["qty"].argmax(dim=-1).reshape(-1).tolist()
    return [EventFrame(EVENT_TYPES[event], ENTITY_TYPES[entity], REF_TYPES[ref], int(qty)) for event, entity, ref, qty in zip(events, entities, refs, qtys)]


def frames_from_story_logits(logits: dict[str, torch.Tensor], masks: torch.Tensor) -> list[list[EventFrame]]:
    events = logits["event"].argmax(dim=-1)
    entities = logits["entity"].argmax(dim=-1)
    refs = logits["ref"].argmax(dim=-1)
    qtys = logits["qty"].argmax(dim=-1)
    out: list[list[EventFrame]] = []
    for row_idx in range(events.shape[0]):
        frames: list[EventFrame] = []
        for col_idx in range(events.shape[1]):
            if bool(masks[row_idx, col_idx].item()):
                frames.append(EventFrame(EVENT_TYPES[int(events[row_idx, col_idx])], ENTITY_TYPES[int(entities[row_idx, col_idx])], REF_TYPES[int(refs[row_idx, col_idx])], int(qtys[row_idx, col_idx])))
        out.append(frames)
    return out


def group_flat_frames(clause_examples: list[ClauseExample], frames: list[EventFrame], story_count: int) -> list[list[EventFrame]]:
    grouped: list[list[EventFrame]] = [[] for _ in range(story_count)]
    for example, frame in zip(clause_examples, frames):
        grouped[example.story_index].append(frame)
    return grouped


def field_values(frames: list[EventFrame], field: str) -> list[object]:
    return [getattr(frame, field) for frame in frames]


def accuracy(preds: list[object], gold: list[object]) -> float:
    return sum(int(pred == truth) for pred, truth in zip(preds, gold)) / len(gold) if gold else math.nan


def confusion(gold: list[object], pred: list[object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for truth, guess in zip(gold, pred):
        counts[f"{truth}->{guess}"] += 1
    return dict(sorted(counts.items()))


def frame_metrics(gold_frames: list[EventFrame], pred_frames: list[EventFrame]) -> dict[str, object]:
    exact = [pred == gold for pred, gold in zip(pred_frames, gold_frames)]
    event_gold, event_pred = field_values(gold_frames, "event_type"), field_values(pred_frames, "event_type")
    entity_gold, entity_pred = field_values(gold_frames, "entity_type"), field_values(pred_frames, "entity_type")
    ref_gold, ref_pred = field_values(gold_frames, "ref_type"), field_values(pred_frames, "ref_type")
    qty_gold, qty_pred = field_values(gold_frames, "quantity"), field_values(pred_frames, "quantity")
    illegal = sum(int(validate_frame(frame)[1] is not None) for frame in pred_frames)
    return {
        "event_frame_exact_accuracy": sum(exact) / len(exact) if exact else math.nan,
        "event_type_accuracy": accuracy(event_pred, event_gold),
        "entity_type_accuracy": accuracy(entity_pred, entity_gold),
        "ref_type_accuracy": accuracy(ref_pred, ref_gold),
        "quantity_accuracy": accuracy(qty_pred, qty_gold),
        "illegal_frame_rate": illegal / len(pred_frames) if pred_frames else math.nan,
        "event_type_confusion": confusion(event_gold, event_pred),
        "entity_type_confusion": confusion(entity_gold, entity_pred),
        "ref_type_confusion": confusion(ref_gold, ref_pred),
        "quantity_confusion": confusion(qty_gold, qty_pred),
    }


def tag_accuracy(stories: list[StoryExample], preds: list[int], tag: str) -> float:
    rows = [(story, pred) for story, pred in zip(stories, preds) if tag in story.tags]
    if not rows:
        return math.nan
    return sum(int(pred == story.answer) for story, pred in rows) / len(rows)


def story_metrics(stories: list[StoryExample], preds: list[int], frame_exact: float) -> dict[str, float]:
    ledger_acc = accuracy(preds, [story.answer for story in stories])
    return {
        "ledger_answer_accuracy": ledger_acc,
        "same_token_set_accuracy": tag_accuracy(stories, preds, "same_token_set"),
        "event_order_contrast_accuracy": tag_accuracy(stories, preds, "event_order_shuffle"),
        "coreference_accuracy": tag_accuracy(stories, preds, "coreference"),
        "distractor_resistance": tag_accuracy(stories, preds, "distractor"),
        "same_token_role_swap_pair_accuracy": tag_accuracy(stories, preds, "same_token_role_swap"),
        "same_token_story_pair_accuracy": tag_accuracy(stories, preds, "same_token_story_pair"),
        "affected_entity_accuracy": tag_accuracy(stories, preds, "affected_entity"),
        "invisible_entity_target_accuracy": tag_accuracy(stories, preds, "invisible_entity_target"),
        "passive_active_contrast_accuracy": tag_accuracy(stories, preds, "passive_active_contrast"),
        "negation_noop_accuracy": tag_accuracy(stories, preds, "negation_noop"),
        "near_miss_noop_accuracy": tag_accuracy(stories, preds, "near_miss_noop"),
        "modal_noop_accuracy": tag_accuracy(stories, preds, "modal_noop"),
        "mention_trap_accuracy": tag_accuracy(stories, preds, "mention_trap"),
        "ghost_reference_accuracy": tag_accuracy(stories, preds, "ghost_reference"),
        "identity_restore_accuracy": tag_accuracy(stories, preds, "identity_restore"),
        "previous_vs_other_accuracy": tag_accuracy(stories, preds, "previous_vs_other"),
        "query_target_accuracy": tag_accuracy(stories, preds, "query_target"),
        "first_second_accuracy": tag_accuracy(stories, preds, "first_second"),
        "heldout_verb_accuracy": tag_accuracy(stories, preds, "heldout_verb"),
        "heldout_reference_accuracy": tag_accuracy(stories, preds, "heldout_reference"),
        "heldout_noun_accuracy_diagnostic": tag_accuracy(stories, preds, "heldout_noun"),
        "invalid_restore_accuracy": tag_accuracy(stories, preds, "invalid_restore"),
        "impossible_reference_accuracy": tag_accuracy(stories, preds, "impossible_reference"),
        "ambiguous_reference_accuracy": tag_accuracy(stories, preds, "ambiguous_reference"),
        "null_action_accuracy": tag_accuracy(stories, preds, "null_action"),
        "ghost_reference_accuracy": tag_accuracy(stories, preds, "ghost_reference"),
        "nested_target_accuracy": tag_accuracy(stories, preds, "nested_target"),
        "no_mutation_on_noop_accuracy": tag_accuracy(stories, preds, "no_mutation_on_noop"),
        "frame_to_answer_gap": frame_exact - ledger_acc if not math.isnan(frame_exact) else math.nan,
        "direct_answer_vs_frame_ledger_gap": math.nan,
    }


def run_predicted_ledger(stories: list[StoryExample], grouped_frames: list[list[EventFrame]]) -> list[int]:
    preds: list[int] = []
    for frames in grouped_frames:
        answer, _ledger = run_ledger(frames)
        preds.append(answer)
    return preds


def failed_cases(stories: list[StoryExample], preds: list[int], limit: int = 10) -> list[dict[str, object]]:
    failed = []
    for story, pred in zip(stories, preds):
        if pred != story.answer:
            failed.append({"story_id": story.story_id, "text": story.row.text, "expected": story.answer, "predicted": pred, "tags": list(story.tags)})
            if len(failed) >= limit:
                break
    return failed


def feature_leak_audit(stories: list[StoryExample]) -> str:
    forbidden = {"create", "remove", "restore", "query_count", "noop_or_invalid", "event_type", "entity_type", "ref_type", "distractor_create"}
    for story in stories:
        tokens = set(base.tokenise(" ".join(story.clauses)))
        if tokens & forbidden:
            return "fail"
    return "pass"


def finish_job(result: JobResult, progress_path: Path | None) -> JobResult:
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_done", "arm": result.arm, "seed": result.seed})
    return result


def run_job(
    arm: str,
    seed: int,
    train_examples: int,
    eval_examples: int,
    epochs: int,
    hidden: int,
    embed_dim: int,
    batch_size: int,
    lr: float,
    progress_root: Path | None,
    heartbeat_sec: int,
) -> JobResult:
    set_worker(seed)
    progress_path = progress_root / safe_job_name(arm, seed) if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "arm": arm, "seed": seed})

    train_stories, eval_stories = build_probe_dataset(seed, train_examples, eval_examples)
    vocab = build_vocab_from_stories(train_stories, eval_stories)
    feature_audit = feature_leak_audit(train_stories + eval_stories)
    eval_clause_examples = flatten_clauses(eval_stories)
    gold_flat = [example.frame for example in eval_clause_examples]

    if arm == "EVENT_FRAME_ORACLE":
        grouped = [list(story.frames) for story in eval_stories]
        preds = run_predicted_ledger(eval_stories, grouped)
        fm = frame_metrics(gold_flat, gold_flat)
        metrics = {**fm, **story_metrics(eval_stories, preds, fm["event_frame_exact_accuracy"]), "feature_leak_audit": feature_audit, "vocab_size": len(vocab)}
        return finish_job(JobResult(arm, seed, metrics, failed_cases(eval_stories, preds)), progress_path)

    if arm == "DIRECT_STORY_GRU_ANSWER":
        train_rows = [story.row for story in train_stories]
        eval_rows = [story.row for story in eval_stories]
        story_vocab = base.build_vocab(train_rows + eval_rows)
        train_seq = base.encode_sequences(train_rows, story_vocab)
        eval_seq = base.encode_sequences(eval_rows, story_vocab)
        train_y = torch.tensor([story.answer for story in train_stories], dtype=torch.long)
        eval_gold = [story.answer for story in eval_stories]
        model = sb.train_direct_gru(train_seq, train_y, seed, len(story_vocab), embed_dim, hidden, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            preds = model(eval_seq).argmax(dim=1).tolist()
        metrics = {
            "event_frame_exact_accuracy": math.nan,
            "event_type_accuracy": math.nan,
            "entity_type_accuracy": math.nan,
            "ref_type_accuracy": math.nan,
            "quantity_accuracy": math.nan,
            "illegal_frame_rate": math.nan,
            **story_metrics(eval_stories, preds, math.nan),
            "ledger_answer_accuracy": accuracy(preds, eval_gold),
            "feature_leak_audit": feature_audit,
            "vocab_size": len(story_vocab),
        }
        return finish_job(JobResult(arm, seed, metrics, failed_cases(eval_stories, preds)), progress_path)

    train_clause_examples = flatten_clauses(train_stories)
    if arm == "SHUFFLED_EVENT_FRAME_TEACHER":
        perm = torch.randperm(len(train_clause_examples), generator=torch.Generator().manual_seed(seed + 9101)).tolist()
        shuffled_frames = [train_clause_examples[idx].frame for idx in perm]
        train_clause_examples = [
            ClauseExample(example.story_index, example.clause_index, example.text, frame, example.tags)
            for example, frame in zip(train_clause_examples, shuffled_frames)
        ]

    if arm in {"SEGMENTED_EVENT_FRAME_CLASSIFIER", "SHUFFLED_EVENT_FRAME_TEACHER"}:
        train_seq = encode_texts([example.text for example in train_clause_examples], vocab, MAX_CLAUSE_LEN)
        eval_seq = encode_texts([example.text for example in eval_clause_examples], vocab, MAX_CLAUSE_LEN)
        model = ClauseGRUFrameModel(len(vocab), embed_dim, hidden)
        model = train_frame_model(model, train_seq, frame_targets(train_clause_examples), seed, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            pred_flat = harden_logits(model(eval_seq))
    elif arm == "BAG_OF_TOKENS_EVENT_FRAME":
        train_seq = encode_texts([example.text for example in train_clause_examples], vocab, MAX_CLAUSE_LEN)
        eval_seq = encode_texts([example.text for example in eval_clause_examples], vocab, MAX_CLAUSE_LEN)
        train_x = base.bag_features(train_seq, len(vocab))
        eval_x = base.bag_features(eval_seq, len(vocab))
        model = FrameMLP(train_x.shape[1], hidden)
        model = train_frame_model(model, train_x, frame_targets(train_clause_examples), seed, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            pred_flat = harden_logits(model(eval_x))
    elif arm == "STATIC_POSITION_EVENT_FRAME":
        train_seq = encode_texts([example.text for example in train_clause_examples], vocab, MAX_CLAUSE_LEN)
        eval_seq = encode_texts([example.text for example in eval_clause_examples], vocab, MAX_CLAUSE_LEN)
        train_x = base.static_position_features(train_seq, len(vocab))
        eval_x = base.static_position_features(eval_seq, len(vocab))
        model = FrameMLP(train_x.shape[1], hidden)
        model = train_frame_model(model, train_x, frame_targets(train_clause_examples), seed, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            pred_flat = harden_logits(model(eval_x))
    elif arm == "STORY_TO_EVENT_FRAMES_GRU":
        train_seq, train_boundaries = encode_story_sequences(train_stories, vocab)
        eval_seq, eval_boundaries = encode_story_sequences(eval_stories, vocab)
        max_events = train_boundaries.shape[1]
        train_targets = story_frame_targets(train_stories, max_events)
        model = StoryFrameGRU(len(vocab), embed_dim, hidden)
        model = train_story_frame_model(model, train_seq, train_boundaries, train_targets, seed, epochs, lr, batch_size, progress_path, heartbeat_sec)
        with torch.no_grad():
            eval_masks = eval_boundaries >= 0
            grouped = frames_from_story_logits(model(eval_seq, eval_boundaries), eval_masks)
        pred_flat = [frame for frames in grouped for frame in frames]
        fm = frame_metrics(gold_flat, pred_flat)
        preds = run_predicted_ledger(eval_stories, grouped)
        metrics = {**fm, **story_metrics(eval_stories, preds, fm["event_frame_exact_accuracy"]), "feature_leak_audit": feature_audit, "vocab_size": len(vocab)}
        return finish_job(JobResult(arm, seed, metrics, failed_cases(eval_stories, preds)), progress_path)
    else:
        raise ValueError(f"unknown arm: {arm}")

    grouped = group_flat_frames(eval_clause_examples, pred_flat, len(eval_stories))
    fm = frame_metrics(gold_flat, pred_flat)
    preds = run_predicted_ledger(eval_stories, grouped)
    metrics = {**fm, **story_metrics(eval_stories, preds, fm["event_frame_exact_accuracy"]), "feature_leak_audit": feature_audit, "vocab_size": len(vocab)}
    return finish_job(JobResult(arm, seed, metrics, failed_cases(eval_stories, preds)), progress_path)


def result_record(result: JobResult) -> dict[str, object]:
    return {"arm": result.arm, "seed": result.seed, **result.metrics, "failed_cases": result.failed_cases}


def record_to_result(record: dict[str, object]) -> JobResult:
    metrics = {key: value for key, value in record.items() if key not in {"arm", "seed", "failed_cases"}}
    return JobResult(str(record["arm"]), int(record["seed"]), metrics, list(record.get("failed_cases", [])))


def load_existing_results(path: Path) -> list[JobResult]:
    if not path.exists():
        return []
    rows: list[JobResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(record_to_result(json.loads(line)))
    return rows


def merge_confusions(values: list[object]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for value in values:
        if isinstance(value, dict):
            out.update({str(k): int(v) for k, v in value.items()})
    return dict(sorted(out.items()))


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_arm: dict[str, list[JobResult]] = defaultdict(list)
    for result in results:
        by_arm[result.arm].append(result)
    out: dict[str, dict[str, object]] = {}
    for arm, rows in sorted(by_arm.items()):
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


def safe_nanmean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.mean(clean)) if clean else math.nan


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    oracle = metric_mean(agg, "EVENT_FRAME_ORACLE", "ledger_answer_accuracy", 0.0)
    if oracle < 0.98:
        labels.append("LEDGER_UPDATE_BOTTLENECK")
        return labels
    audits: list[str] = []
    for data in agg.values():
        audit = data["metrics"].get("feature_leak_audit")
        if isinstance(audit, list):
            audits.extend(audit)
    if set(audits or ["pass"]) != {"pass"}:
        labels.append("TOKEN_TO_EVENT_FRAME_INVALID_AUDIT")

    seg_frame = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "event_frame_exact_accuracy", 0.0)
    seg_ledger = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "ledger_answer_accuracy", 0.0)
    story_ledger = metric_mean(agg, "STORY_TO_EVENT_FRAMES_GRU", "ledger_answer_accuracy", 0.0)
    shuffled = metric_mean(agg, "SHUFFLED_EVENT_FRAME_TEACHER", "ledger_answer_accuracy", 0.0)
    bag_story = safe_nanmean(
        [
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "same_token_set_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "event_order_contrast_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "coreference_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "invalid_restore_accuracy"),
        ]
    )
    static_story = safe_nanmean(
        [
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "same_token_set_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "event_order_contrast_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "coreference_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "invalid_restore_accuracy"),
        ]
    )

    role = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "same_token_role_swap_pair_accuracy", 0.0)
    affected = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "affected_entity_accuracy", 0.0)
    ref_acc = metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "ref_type_accuracy", 0.0)
    noop = safe_nanmean(
        [
            metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "negation_noop_accuracy"),
            metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "near_miss_noop_accuracy"),
            metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "modal_noop_accuracy"),
            metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "mention_trap_accuracy"),
        ]
    )
    bag_hard = safe_nanmean(
        [
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "same_token_role_swap_pair_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "affected_entity_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "negation_noop_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "mention_trap_accuracy"),
            metric_mean(agg, "BAG_OF_TOKENS_EVENT_FRAME", "previous_vs_other_accuracy"),
        ]
    )
    static_hard = safe_nanmean(
        [
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "same_token_role_swap_pair_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "affected_entity_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "negation_noop_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "mention_trap_accuracy"),
            metric_mean(agg, "STATIC_POSITION_EVENT_FRAME", "previous_vs_other_accuracy"),
        ]
    )

    if (
        seg_frame >= 0.90
        and seg_ledger >= 0.90
        and role >= 0.85
        and affected >= 0.85
        and ref_acc >= 0.85
        and noop >= 0.85
        and seg_ledger >= shuffled + 0.15
        and seg_ledger >= max(bag_hard, static_hard) + 0.05
    ):
        labels.append("EVENT_FRAME_HARD_POSITIVE")
        if story_ledger < seg_ledger - 0.10:
            labels.append("EVENT_FRAME_SEGMENTATION_BOTTLENECK")
    elif seg_frame < 0.75 or seg_ledger < 0.75:
        labels.append("EVENT_FRAME_WEAK")
        labels.append("EXPLICIT_EVENT_PARSER_REQUIRED_FOR_NOW")
    else:
        labels.append("EXPLICIT_EVENT_PARSER_REQUIRED_FOR_NOW")

    if (bag_hard >= 0.85 or static_hard >= 0.85 or bag_story >= 0.85 or static_story >= 0.85) and seg_ledger >= 0.85:
        labels.append("EVENT_FRAME_LEXICAL_SHORTCUT")
    if metric_mean(agg, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "event_type_accuracy", 0.0) >= 0.90 and affected < 0.85:
        labels.append("ROLE_BINDING_BOTTLENECK")
    if ref_acc < 0.85:
        labels.append("REFERENCE_BINDING_BOTTLENECK")
    if noop < 0.85:
        labels.append("NEGATION_MODAL_BOTTLENECK")
    return labels


def write_frame_curve(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    rows = []
    for arm in sorted(agg):
        rows.append(
            {
                "arm": arm,
                "event_frame_exact_accuracy": metric_mean(agg, arm, "event_frame_exact_accuracy"),
                "ledger_answer_accuracy": metric_mean(agg, arm, "ledger_answer_accuracy"),
                "event_type_accuracy": metric_mean(agg, arm, "event_type_accuracy"),
                "entity_type_accuracy": metric_mean(agg, arm, "entity_type_accuracy"),
                "ref_type_accuracy": metric_mean(agg, arm, "ref_type_accuracy"),
                "quantity_accuracy": metric_mean(agg, arm, "quantity_accuracy"),
                "illegal_frame_rate": metric_mean(agg, arm, "illegal_frame_rate"),
            }
        )
    write_json(out_dir / "frame_curve.json", rows)


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    direct = metric_mean(agg, "DIRECT_STORY_GRU_ANSWER", "ledger_answer_accuracy")
    for data in agg.values():
        metric = data["metrics"].get("direct_answer_vs_frame_ledger_gap")
        if isinstance(metric, dict) and not math.isnan(direct):
            ledger = metric_mean(agg, data["arm"], "ledger_answer_accuracy")
            data["metrics"]["direct_answer_vs_frame_ledger_gap"] = {"mean": round(direct - ledger, 6), "min": round(direct - ledger, 6), "max": round(direct - ledger, 6)}
    summary = {
        "status": status,
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
        },
        "aggregate": agg,
    }
    write_json(out_dir / "summary.json", summary)
    write_frame_curve(out_dir, agg)
    lines = [
        "# TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Jobs: `{jobs}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Frame Exact | Ledger | Event | Entity | Ref | Qty | Role Swap | Affected | Noop/Neg | Mention | Prev/Other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "event_frame_exact_accuracy"),
                metric_mean(agg, arm, "ledger_answer_accuracy"),
                metric_mean(agg, arm, "event_type_accuracy"),
                metric_mean(agg, arm, "entity_type_accuracy"),
                metric_mean(agg, arm, "ref_type_accuracy"),
                metric_mean(agg, arm, "quantity_accuracy"),
                metric_mean(agg, arm, "same_token_role_swap_pair_accuracy"),
                metric_mean(agg, arm, "affected_entity_accuracy"),
                safe_nanmean([metric_mean(agg, arm, "negation_noop_accuracy"), metric_mean(agg, arm, "near_miss_noop_accuracy"), metric_mean(agg, arm, "modal_noop_accuracy")]),
                metric_mean(agg, arm, "mention_trap_accuracy"),
                metric_mean(agg, arm, "previous_vs_other_accuracy"),
            )
        )
    lines.extend(["", "## Confusion Matrices", ""])
    for arm in sorted(agg):
        if arm in {"SEGMENTED_EVENT_FRAME_CLASSIFIER", "STORY_TO_EVENT_FRAMES_GRU", "BAG_OF_TOKENS_EVENT_FRAME", "STATIC_POSITION_EVENT_FRAME"}:
            lines.append(f"### {arm}")
            for name in ("event_type_confusion", "entity_type_confusion", "ref_type_confusion", "quantity_confusion"):
                matrix = agg[arm]["metrics"].get(name, {})
                lines.append("")
                lines.append(f"`{name}`")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(matrix, indent=2, sort_keys=True))
                lines.append("```")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int) -> None:
    lines = [
        "# TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS Result",
        "",
        "## Goal",
        "",
        "Test whether raw event clauses can be converted into hard event frames under contrast cases that should break bag/static shortcuts.",
        "",
        "## Run",
        "",
        "```text",
        f"seeds={args.seeds}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        f"completed_jobs={len(results)}",
        f"heartbeat_sec={args.heartbeat_sec}",
        "```",
        "",
        "## Arm Summary",
        "",
        "| Arm | Frame Exact | Ledger | Role Swap | Affected | Noop/Neg | Mention | Ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(agg):
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                metric_mean(agg, arm, "event_frame_exact_accuracy"),
                metric_mean(agg, arm, "ledger_answer_accuracy"),
                metric_mean(agg, arm, "same_token_role_swap_pair_accuracy"),
                metric_mean(agg, arm, "affected_entity_accuracy"),
                safe_nanmean([metric_mean(agg, arm, "negation_noop_accuracy"), metric_mean(agg, arm, "near_miss_noop_accuracy"), metric_mean(agg, arm, "modal_noop_accuracy")]),
                metric_mean(agg, arm, "mention_trap_accuracy"),
                metric_mean(agg, arm, "ref_type_accuracy"),
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
            "The deterministic identity ledger and hard-frame route are valid if `EVENT_FRAME_ORACLE` saturates and the shuffled teacher collapses.",
            "The run is not a grounding-positive result when `BAG_OF_TOKENS_EVENT_FRAME` or `STATIC_POSITION_EVENT_FRAME` match the segmented parser on hard contrast metrics.",
            "In that case, the dataset is still too template-solvable and the next step is to harden the contrast distribution rather than building PrismionCell.",
            "",
            "## Claim Boundary",
            "",
            "V2 still assumes gold clause segmentation for the segmented arm. This is not full natural-language segmentation, symbol grounding, or a PrismionCell test.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")

    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")

    sample_train, sample_eval = build_probe_dataset(seeds[0], min(args.train_examples, 16), min(args.eval_examples, 16))
    write_jsonl(
        args.out_dir / "examples_sample.jsonl",
        [
            {
                "story_id": story.story_id,
                "text": story.row.text,
                "clauses": story.clauses,
                "frames": [asdict(frame) for frame in story.frames],
                "ledger_answer": story.answer,
                "tags": story.tags,
                "flags": story.flags,
            }
            for story in (sample_train[:6] + sample_eval[:10])
        ],
    )
    hard_tags = {
        "same_token_role_swap",
        "affected_entity",
        "invisible_entity_target",
        "passive_active_contrast",
        "negation_noop",
        "near_miss_noop",
        "modal_noop",
        "mention_trap",
        "ghost_reference",
        "identity_restore",
        "previous_vs_other",
        "event_order_contrast",
        "same_token_story_pair",
        "query_target",
    }
    write_jsonl(
        args.out_dir / "hard_contrast_cases.jsonl",
        [
            {
                "story_id": story.story_id,
                "text": story.row.text,
                "clauses": story.clauses,
                "frames": [asdict(frame) for frame in story.frames],
                "ledger_answer": story.answer,
                "tags": story.tags,
                "flags": story.flags,
            }
            for story in sample_train + sample_eval
            if hard_tags.intersection(story.tags)
        ],
    )

    queue = [(arm, seed) for seed in seeds for arm in arms]
    write_json(args.out_dir / "queue.json", [{"arm": arm, "seed": seed} for arm, seed in queue])
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    results = load_existing_results(metrics_path)
    completed = {(result.arm, result.seed) for result in results}
    pending = [(arm, seed) for arm, seed in queue if (arm, seed) not in completed]
    append_jsonl(
        progress_path,
        {
            "time": now_iso(),
            "event": "run_start_or_resume",
            "total_jobs": len(queue),
            "completed_jobs": len(results),
            "pending_jobs": len(pending),
            "jobs": jobs,
            "os_cpu_count": os.cpu_count(),
            "torch_threads_per_worker": 1,
        },
    )
    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)

    if jobs <= 1:
        for arm, seed in pending:
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": arm, "seed": seed})
            result = run_job(arm, seed, args.train_examples, args.eval_examples, args.epochs, args.hidden, args.embed_dim, args.batch_size, args.lr, job_progress_root, args.heartbeat_sec)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": result.arm, "seed": result.seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_meta = {}
            pending_futures = set()
            for arm, seed in pending:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "arm": arm, "seed": seed})
                future = pool.submit(run_job, arm, seed, args.train_examples, args.eval_examples, args.epochs, args.hidden, args.embed_dim, args.batch_size, args.lr, job_progress_root, args.heartbeat_sec)
                future_meta[future] = (arm, seed)
                pending_futures.add(future)
            last_heartbeat = time.time()
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
                    last_heartbeat = time.time()
                    continue
                for future in done:
                    arm, seed = future_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        append_jsonl(progress_path, {"time": now_iso(), "event": "job_error", "arm": arm, "seed": seed, "error": repr(exc), "completed_jobs": len(results)})
                        write_outputs(args.out_dir, results, args, status="partial_error", jobs=jobs)
                        raise
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, status="partial", jobs=jobs)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "arm": result.arm, "seed": result.seed, "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                if time.time() - last_heartbeat >= args.heartbeat_sec:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending_futures)})
                    last_heartbeat = time.time()

    results.sort(key=lambda item: (item.arm, item.seed))
    agg, labels = write_outputs(args.out_dir, results, args, status="complete", jobs=jobs)
    write_doc_result(agg, labels, args, jobs)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "completed_jobs": len(results), "total_jobs": len(queue)})
    print(json.dumps({"verdict": labels, "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
