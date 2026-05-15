#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import warnings
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

from scripts.probes import run_token_state_update_vs_latent_probe as base
from scripts.probes import run_token_to_event_frame_anti_static_probe as v3
from scripts.probes import run_token_to_event_frame_hard_contrasts_probe as v2


DEFAULT_OUT = ROOT / "target" / "pilot_wave" / "token_to_event_frame_004_curriculum_factors"
CONTRACT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS_CONTRACT.md"
DOC_REPORT = ROOT / "docs" / "research" / "TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS_RESULT.md"

ALL_ARMS = v3.ALL_ARMS
ALL_SUITES = ("role_binding", "negation_modal", "template_curriculum", "combined_recheck")
CURRICULUM_ROUNDS = range(7)
HARD_METRICS = (
    "role_binding_accuracy",
    "affected_entity_accuracy",
    "pair_accuracy",
    "passive_active_contrast_accuracy",
    "noop_frame_accuracy",
    "noop_no_mutation_accuracy",
    "negation_noop_accuracy",
    "near_miss_noop_accuracy",
    "modal_noop_accuracy",
    "mention_trap_accuracy",
    "heldout_template_accuracy",
)


@dataclass(frozen=True)
class JobResult:
    suite: str
    curriculum_round: int
    arm: str
    seed: int
    metrics: dict[str, object]
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOKEN_TO_EVENT_FRAME_004 curriculum factor isolation probe.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--suite", default="all", choices=("all",) + ALL_SUITES)
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
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    base.set_deterministic(seed)


def safe_job_name(suite: str, curriculum_round: int, arm: str, seed: int) -> str:
    return f"{suite}__round_{curriculum_round}__{arm}__seed_{seed}.jsonl"


def role_binding_specs() -> tuple[v3.TemplateSpec, ...]:
    return (
        v3.spec("active_simple", "The {observer} watches the {target} get stolen.", "REMOVE", "LATE", True, False, 0),
        v3.spec("basic_passive", "The {target} was stolen while the {observer} watched.", "REMOVE", "BEGIN", True, False, 0),
        v3.spec("active_carry", "The {observer} carries the {target} away.", "REMOVE", "MID", True, False, 0),
        v3.spec("passive_carry", "The {target} is carried away by the {observer}.", "REMOVE", "BEGIN", False, True, 1),
    )


def simple_stolen_clause(entity: str, tags: tuple[str, ...], group: str | None) -> v3.ClauseItem:
    spec_obj = v3.spec("plain_stolen", "The {target} was stolen.", "REMOVE", "EARLY", True, False, 0, None)
    return v3.clause_from_spec(spec_obj, entity, None, tags, group)


def role_binding_pool(seed: int, split: str) -> list[v3.StoryExample]:
    rng = np.random.default_rng(seed + (0 if split == "train" else 101))
    entities = list(base.ENTITY_TYPES)
    specs = role_binding_specs()
    if split == "train":
        active_specs = [item for item in specs if item.template_family in {"active_simple", "basic_passive", "active_carry"}]
        split_tag = "train"
    elif split == "id_eval":
        active_specs = [item for item in specs if item.template_family in {"active_simple", "basic_passive", "active_carry"}]
        split_tag = "in_distribution_hard"
    else:
        active_specs = list(specs)
        split_tag = "heldout_template"

    pool: list[v3.StoryExample] = []
    idx = 0
    for left_i, left in enumerate(entities):
        for right in entities[left_i + 1 :]:
            group = f"{split}_role_swap_{left}_{right}_{idx}"
            pair_specs = [active_specs[idx % len(active_specs)]]
            if split != "train" and any(item.template_family == "passive_carry" for item in active_specs):
                pair_specs.append(next(item for item in active_specs if item.template_family == "passive_carry"))
            for spec_obj in pair_specs:
                for target, observer in ((right, left), (left, right)):
                    tags = (
                        split_tag,
                        "role_binding",
                        "affected_entity",
                        "same_token_role_swap",
                        "target_position_invariance",
                    )
                    if "carry" in spec_obj.template_family or "passive" in spec_obj.template_family:
                        tags += ("passive_active_contrast",)
                    story = v3.make_story(
                        f"{split}_role_{idx}_{spec_obj.template_family}_{observer}_{target}",
                        [
                            v3.create_clause(observer, group),
                            v3.create_clause(target, group),
                            v3.clause_from_spec(spec_obj, target, observer, tags, group),
                            v3.query_clause(target, split == "heldout_eval", group),
                        ],
                        tags,
                        split,
                        group,
                    )
                    pool.append(story)
                    idx += 1
    rng.shuffle(pool)
    return pool


def negation_modal_pool(seed: int, split: str) -> list[v3.StoryExample]:
    rng = np.random.default_rng(seed + (0 if split == "train" else 211))
    entities = list(base.ENTITY_TYPES)
    if split == "train":
        specs = [item for item in v3.NOOP_SPECS if item.template_family in {"negation_noop", "null_action"}]
        split_tag = "train"
    elif split == "id_eval":
        specs = [item for item in v3.NOOP_SPECS if item.template_family in {"negation_noop", "null_action"}]
        split_tag = "in_distribution_hard"
    else:
        specs = [item for item in v3.NOOP_SPECS if item.template_family in {"near_miss_noop", "modal_noop", "mention_trap", "null_action"}]
        split_tag = "heldout_template"

    pool: list[v3.StoryExample] = []
    idx = 0
    for target in entities:
        group = f"{split}_neg_pos_{target}"
        tags = (split_tag, "negation_modal", "affected_entity")
        pool.append(
            v3.make_story(
                f"{split}_plain_stolen_{target}",
                [v3.create_clause(target, group), simple_stolen_clause(target, tags, group), v3.query_clause(target, split == "heldout_eval", group)],
                tags,
                split,
                group,
            )
        )
        for spec_obj in specs:
            group = f"{split}_neg_{spec_obj.template_family}_{target}_{idx}"
            tags = (split_tag, "negation_modal", "no_mutation_on_noop")
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
            pool.append(
                v3.make_story(
                    f"{split}_neg_{idx}_{spec_obj.template_family}_{target}",
                    [v3.create_clause(target, group), v3.clause_from_spec(spec_obj, target, None, tags, group), v3.query_clause(target, split == "heldout_eval", group)],
                    tags,
                    split,
                    group,
                )
            )
            idx += 1
    rng.shuffle(pool)
    return pool


CURRICULUM_REMOVE_ROUND = {
    "active_simple": 0,
    "basic_passive": 0,
    "active_carry": 0,
    "fronted_subordinate": 1,
    "relative_clause": 2,
    "cleft_focus": 3,
    "target_late": 4,
    "distractor_before_target": 5,
    "target_before_distractor": 5,
    "passive_carry": 5,
}

CURRICULUM_NOOP_ROUND = {
    "negation_noop": 0,
    "null_action": 0,
    "near_miss_noop": 6,
    "modal_noop": 6,
    "mention_trap": 6,
}


def curriculum_remove_specs(split: str, round_id: int) -> list[v3.TemplateSpec]:
    if split == "heldout_eval":
        target_round = min(round_id + 1, max(CURRICULUM_REMOVE_ROUND.values()))
        return [item for item in v3.REMOVE_SPECS if CURRICULUM_REMOVE_ROUND.get(item.template_family) == target_round]
    return [item for item in v3.REMOVE_SPECS if CURRICULUM_REMOVE_ROUND.get(item.template_family, 99) <= round_id]


def curriculum_noop_specs(split: str, round_id: int) -> list[v3.TemplateSpec]:
    if split == "heldout_eval":
        target_round = 6 if round_id >= 5 else min(round_id + 1, 6)
        return [item for item in v3.NOOP_SPECS if CURRICULUM_NOOP_ROUND.get(item.template_family) == target_round]
    return [item for item in v3.NOOP_SPECS if CURRICULUM_NOOP_ROUND.get(item.template_family, 99) <= round_id]


def template_curriculum_pool(seed: int, split: str, round_id: int) -> list[v3.StoryExample]:
    rng = np.random.default_rng(seed + 1000 * (round_id + 1) + (0 if split == "train" else 313))
    entities = list(base.ENTITY_TYPES)
    split_tag = "train" if split == "train" else ("in_distribution_hard" if split == "id_eval" else "heldout_template")
    pool: list[v3.StoryExample] = []
    idx = 0
    remove_specs = curriculum_remove_specs(split, round_id)
    noop_specs = curriculum_noop_specs(split, round_id)
    if not remove_specs:
        remove_specs = curriculum_remove_specs("id_eval", round_id)

    for target in entities:
        for observer in entities:
            if target == observer:
                continue
            for spec_obj in remove_specs:
                group = f"{split}_curr_r{round_id}_{spec_obj.template_family}_{observer}_{target}_{idx}"
                tags = (
                    split_tag,
                    "template_curriculum",
                    "role_binding",
                    "affected_entity",
                    "target_position_invariance",
                    "same_token_role_swap",
                )
                if "carry" in spec_obj.template_family or "passive" in spec_obj.template_family:
                    tags += ("passive_active_contrast",)
                story = v3.make_story(
                    f"{split}_curr_remove_r{round_id}_{idx}",
                    [
                        v3.create_clause(observer, group),
                        v3.create_clause(target, group),
                        v3.clause_from_spec(spec_obj, target, observer, tags, group),
                        v3.query_clause(target, split == "heldout_eval", group),
                    ],
                    tags,
                    split,
                    group,
                )
                pool.append(story)
                idx += 1

    for target in entities:
        for spec_obj in noop_specs:
            group = f"{split}_curr_noop_r{round_id}_{spec_obj.template_family}_{target}_{idx}"
            tags = (split_tag, "template_curriculum", "negation_modal", "no_mutation_on_noop")
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
            pool.append(
                v3.make_story(
                    f"{split}_curr_noop_r{round_id}_{idx}",
                    [v3.create_clause(target, group), v3.clause_from_spec(spec_obj, target, None, tags, group), v3.query_clause(target, split == "heldout_eval", group)],
                    tags,
                    split,
                    group,
                )
            )
            idx += 1

    rng.shuffle(pool)
    return pool


def combined_recheck_pool(seed: int, split: str) -> list[v3.StoryExample]:
    pool = []
    pool.extend(role_binding_pool(seed + 5, split))
    pool.extend(negation_modal_pool(seed + 7, split))
    pool.extend(template_curriculum_pool(seed + 11, split, 2))
    rng = np.random.default_rng(seed + (0 if split == "train" else 419))
    rng.shuffle(pool)
    return pool


def build_suite_dataset(seed: int, train_examples: int, eval_examples: int, suite: str, curriculum_round: int) -> tuple[list[v3.StoryExample], list[v3.StoryExample]]:
    if suite == "role_binding":
        train_pool = role_binding_pool(seed, "train")
        id_pool = role_binding_pool(seed + 17, "id_eval")
        heldout_pool = role_binding_pool(seed + 31, "heldout_eval")
    elif suite == "negation_modal":
        train_pool = negation_modal_pool(seed, "train")
        id_pool = negation_modal_pool(seed + 17, "id_eval")
        heldout_pool = negation_modal_pool(seed + 31, "heldout_eval")
    elif suite == "template_curriculum":
        train_pool = template_curriculum_pool(seed, "train", curriculum_round)
        id_pool = template_curriculum_pool(seed + 17, "id_eval", curriculum_round)
        heldout_pool = template_curriculum_pool(seed + 31, "heldout_eval", curriculum_round)
    elif suite == "combined_recheck":
        train_pool = combined_recheck_pool(seed, "train")
        id_pool = combined_recheck_pool(seed + 17, "id_eval")
        heldout_pool = combined_recheck_pool(seed + 31, "heldout_eval")
    else:
        raise ValueError(f"unknown suite: {suite}")

    train = v3.repeat_to_count(train_pool, train_examples, f"train_{suite}_r{curriculum_round}")
    id_count = eval_examples // 2
    heldout_count = eval_examples - id_count
    eval_set = v3.repeat_to_count(id_pool, id_count, f"id_{suite}_r{curriculum_round}") + v3.repeat_to_count(heldout_pool or id_pool, heldout_count, f"heldout_{suite}_r{curriculum_round}")
    return train, eval_set


def selected_suite_rounds(suite_arg: str) -> list[tuple[str, int]]:
    if suite_arg == "all":
        out = [("role_binding", 0), ("negation_modal", 0)]
        out.extend(("template_curriculum", round_id) for round_id in CURRICULUM_ROUNDS)
        out.append(("combined_recheck", 0))
        return out
    if suite_arg == "template_curriculum":
        return [("template_curriculum", round_id) for round_id in CURRICULUM_ROUNDS]
    return [(suite_arg, 0)]


def suite_metrics(base_metrics: dict[str, float], stories: list[v3.StoryExample], preds: list[int], arm: str) -> dict[str, float]:
    out = dict(base_metrics)
    out["role_binding_accuracy"] = v3.tag_accuracy(stories, preds, "role_binding")
    out["hard_contrast_score"] = safe_mean([float(out[name]) for name in HARD_METRICS if name in out and not math.isnan(float(out[name]))])
    out["bag_hard_contrast_score"] = out["hard_contrast_score"] if arm == "BAG_OF_TOKENS_EVENT_FRAME" else math.nan
    out["static_hard_contrast_score"] = out["hard_contrast_score"] if arm == "STATIC_POSITION_EVENT_FRAME" else math.nan
    out["position_only_hard_contrast_score"] = out["hard_contrast_score"] if arm == "POSITION_ONLY_EVENT_FRAME" else math.nan
    out["first_collapse_round"] = math.nan
    return out


def frame_metrics(gold_frames: list[v2.EventFrame], pred_frames: list[v2.EventFrame]) -> dict[str, object]:
    return v3.frame_metrics(gold_frames, pred_frames)


def failed_cases(stories: list[v3.StoryExample], preds: list[int], limit: int = 10) -> list[dict[str, object]]:
    return v3.failed_cases(stories, preds, limit)


def safe_mean(vals: Iterable[float]) -> float:
    clean = [float(val) for val in vals if not math.isnan(float(val))]
    return float(np.mean(clean)) if clean else math.nan


def run_job(suite: str, curriculum_round: int, arm: str, seed: int, args: argparse.Namespace, progress_root: Path | None) -> JobResult:
    set_worker(seed)
    progress_path = progress_root / safe_job_name(suite, curriculum_round, arm, seed) if progress_root is not None else None
    if progress_path is not None:
        append_jsonl(progress_path, {"time": now_iso(), "event": "job_worker_start", "suite": suite, "round": curriculum_round, "arm": arm, "seed": seed})

    train_stories, eval_stories = build_suite_dataset(seed, args.train_examples, args.eval_examples, suite, curriculum_round)
    vocab = v3.build_vocab_from_stories(train_stories, eval_stories)
    audit = v3.feature_leak_audit(train_stories + eval_stories)
    eval_clauses = v3.flatten_clauses(eval_stories)
    gold_flat = [example.frame for example in eval_clauses]

    if arm == "EVENT_FRAME_ORACLE":
        grouped = [list(story.frames) for story in eval_stories]
        preds, ledgers = v3.run_predicted_ledger(eval_stories, grouped)
        fm = frame_metrics(gold_flat, gold_flat)
        sm = suite_metrics(v3.story_metrics(eval_stories, preds, ledgers, fm["event_frame_exact_accuracy"], eval_clauses, gold_flat), eval_stories, preds, arm)
        metrics = {**fm, **sm, "feature_leak_audit": audit, "vocab_size": len(vocab)}
        return JobResult(suite, curriculum_round, arm, seed, metrics, failed_cases(eval_stories, preds))

    if arm == "DIRECT_STORY_GRU_ANSWER":
        preds, vocab_size = v3.train_direct_answer(train_stories, eval_stories, seed, args, progress_path)
        sm = suite_metrics(v3.story_metrics(eval_stories, preds, None, math.nan), eval_stories, preds, arm)
        metrics = {
            "event_frame_exact_accuracy": math.nan,
            "event_type_accuracy": math.nan,
            "entity_type_accuracy": math.nan,
            "ref_type_accuracy": math.nan,
            "quantity_accuracy": math.nan,
            "illegal_frame_rate": math.nan,
            **sm,
            "feature_leak_audit": audit,
            "vocab_size": vocab_size,
        }
        return JobResult(suite, curriculum_round, arm, seed, metrics, failed_cases(eval_stories, preds))

    train_clauses = v3.flatten_clauses(train_stories)
    if arm == "SHUFFLED_EVENT_FRAME_TEACHER":
        perm = torch.randperm(len(train_clauses), generator=torch.Generator().manual_seed(seed + 9101 + curriculum_round)).tolist()
        shuffled = [train_clauses[idx].frame for idx in perm]
        train_clauses = [v3.ClauseExample(ex.story_index, ex.clause_index, ex.text, frame, ex.tags, ex.meta) for ex, frame in zip(train_clauses, shuffled)]

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
        pred_flat = v3.train_clause_arm(arm, train_clauses, eval_clauses, vocab, seed, args, progress_path)
        grouped = v3.group_flat_frames(eval_clauses, pred_flat, len(eval_stories))

    fm = frame_metrics(gold_flat, pred_flat)
    preds, ledgers = v3.run_predicted_ledger(eval_stories, grouped)
    sm = suite_metrics(v3.story_metrics(eval_stories, preds, ledgers, fm["event_frame_exact_accuracy"], eval_clauses, pred_flat), eval_stories, preds, arm)
    metrics = {**fm, **sm, "feature_leak_audit": audit, "vocab_size": len(vocab)}
    return JobResult(suite, curriculum_round, arm, seed, metrics, failed_cases(eval_stories, preds))


def result_record(result: JobResult) -> dict[str, object]:
    return asdict(result)


def merge_confusions(values: list[object]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for value in values:
        if isinstance(value, dict):
            total.update({str(key): int(count) for key, count in value.items()})
    return dict(sorted(total.items()))


def aggregate(results: list[JobResult]) -> dict[str, dict[str, object]]:
    by_key: dict[tuple[str, int, str], list[JobResult]] = defaultdict(list)
    for result in results:
        by_key[(result.suite, result.curriculum_round, result.arm)].append(result)

    out: dict[str, dict[str, object]] = {}
    for (suite, curriculum_round, arm), rows in sorted(by_key.items()):
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
        key = f"{suite}/round_{curriculum_round}/{arm}"
        out[key] = {"suite": suite, "curriculum_round": curriculum_round, "arm": arm, "seeds": [row.seed for row in rows], "metrics": metrics}
    return out


def metric_mean(agg: dict[str, dict[str, object]], suite: str, round_id: int, arm: str, name: str, default: float = math.nan) -> float:
    key = f"{suite}/round_{round_id}/{arm}"
    try:
        value = agg[key]["metrics"][name]
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
    except KeyError:
        pass
    return default


def suite_round_hard(agg: dict[str, dict[str, object]], suite: str, round_id: int, arm: str) -> float:
    return metric_mean(agg, suite, round_id, arm, "hard_contrast_score")


def available_suite_rounds(agg: dict[str, dict[str, object]]) -> list[tuple[str, int]]:
    return sorted({(str(row["suite"]), int(row["curriculum_round"])) for row in agg.values()})


def first_collapse_round(agg: dict[str, dict[str, object]]) -> int | None:
    rounds = sorted({round_id for suite, round_id in available_suite_rounds(agg) if suite == "template_curriculum"})
    for round_id in rounds:
        seg_hard = suite_round_hard(agg, "template_curriculum", round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER")
        heldout = metric_mean(agg, "template_curriculum", round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "heldout_template_accuracy")
        if seg_hard < 0.85 or heldout < 0.85:
            return round_id
    return None


def verdict(agg: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    oracle_values = [
        metric_mean(agg, suite, round_id, "EVENT_FRAME_ORACLE", "ledger_answer_accuracy", 0.0)
        for suite, round_id in available_suite_rounds(agg)
    ]
    if oracle_values and min(oracle_values) < 0.98:
        return ["LEDGER_UPDATE_BOTTLENECK"]

    audits: list[str] = []
    for data in agg.values():
        audit = data["metrics"].get("feature_leak_audit")
        if isinstance(audit, list):
            audits.extend(audit)
    if set(audits or ["pass"]) != {"pass"}:
        labels.append("TOKEN_TO_EVENT_FRAME_INVALID_AUDIT")

    for suite, round_id in available_suite_rounds(agg):
        seg_frame = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "event_frame_exact_accuracy", 0.0)
        seg_ledger = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "ledger_answer_accuracy", 0.0)
        seg_hard = suite_round_hard(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER")
        static_hard = suite_round_hard(agg, suite, round_id, "STATIC_POSITION_EVENT_FRAME")
        bag_hard = suite_round_hard(agg, suite, round_id, "BAG_OF_TOKENS_EVENT_FRAME")
        pos_hard = suite_round_hard(agg, suite, round_id, "POSITION_ONLY_EVENT_FRAME")
        shuffled = metric_mean(agg, suite, round_id, "SHUFFLED_EVENT_FRAME_TEACHER", "ledger_answer_accuracy", 1.0)

        if static_hard >= seg_hard - 0.03 and static_hard > 0.75:
            labels.append("STATIC_SHORTCUT_RECOVERED")
        if bag_hard >= seg_hard - 0.03 and bag_hard > 0.75:
            labels.append("BAG_SHORTCUT_RECOVERED")
        if seg_frame < 0.70 or seg_hard < 0.65:
            labels.append("PARSER_WEAK_UNDER_CURRICULUM")

        controls_lower = static_hard <= seg_hard - 0.10 and bag_hard <= seg_hard - 0.10 and pos_hard <= seg_hard - 0.10 and shuffled <= 0.55
        if suite == "role_binding":
            role = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "role_binding_accuracy", 0.0)
            affected = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "affected_entity_accuracy", 0.0)
            passive = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "passive_active_contrast_accuracy", 0.0)
            pair = metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", "pair_accuracy", 0.0)
            if seg_frame >= 0.90 and seg_ledger >= 0.90 and min(role, affected, passive, pair) >= 0.85 and controls_lower:
                labels.append("ROLE_BINDING_POSITIVE")
            elif min(role, affected, pair) < 0.85:
                labels.append("ROLE_BINDING_BOTTLENECK")
        elif suite == "negation_modal":
            noop = safe_mean(
                [
                    metric_mean(agg, suite, round_id, "SEGMENTED_EVENT_FRAME_CLASSIFIER", name)
                    for name in ("noop_frame_accuracy", "noop_no_mutation_accuracy", "negation_noop_accuracy", "near_miss_noop_accuracy", "modal_noop_accuracy", "mention_trap_accuracy")
                ]
            )
            if seg_frame >= 0.90 and seg_ledger >= 0.90 and noop >= 0.85 and controls_lower:
                labels.append("NEGATION_MODAL_POSITIVE")
            elif noop < 0.85:
                labels.append("NEGATION_MODAL_BOTTLENECK")
        elif suite == "combined_recheck":
            if seg_frame >= 0.90 and seg_ledger >= 0.90 and seg_hard >= 0.85 and controls_lower:
                labels.append("COMBINED_RECHECK_POSITIVE")

    collapse = first_collapse_round(agg)
    if collapse is None and any(suite == "template_curriculum" for suite, _ in available_suite_rounds(agg)):
        labels.append("TEMPLATE_CURRICULUM_POSITIVE")
    elif collapse is not None:
        labels.append(f"TEMPLATE_COLLAPSE_AT_ROUND_{collapse}")

    if not labels:
        labels.append("CURRICULUM_FACTORS_PARTIAL")
    return sorted(set(labels))


def write_suite_curve(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    rows = []
    for key, data in sorted(agg.items()):
        suite = str(data["suite"])
        round_id = int(data["curriculum_round"])
        arm = str(data["arm"])
        rows.append(
            {
                "suite": suite,
                "curriculum_round": round_id,
                "arm": arm,
                "event_frame_exact_accuracy": metric_mean(agg, suite, round_id, arm, "event_frame_exact_accuracy"),
                "ledger_answer_accuracy": metric_mean(agg, suite, round_id, arm, "ledger_answer_accuracy"),
                "hard_contrast_score": metric_mean(agg, suite, round_id, arm, "hard_contrast_score"),
                "heldout_template_accuracy": metric_mean(agg, suite, round_id, arm, "heldout_template_accuracy"),
            }
        )
    write_json(out_dir / "suite_curve.json", rows)


def write_curriculum_curve(out_dir: Path, agg: dict[str, dict[str, object]]) -> None:
    rows = []
    for round_id in sorted({round_id for suite, round_id in available_suite_rounds(agg) if suite == "template_curriculum"}):
        for arm in ALL_ARMS:
            rows.append(
                {
                    "curriculum_round": round_id,
                    "arm": arm,
                    "event_frame_exact_accuracy": metric_mean(agg, "template_curriculum", round_id, arm, "event_frame_exact_accuracy"),
                    "ledger_answer_accuracy": metric_mean(agg, "template_curriculum", round_id, arm, "ledger_answer_accuracy"),
                    "hard_contrast_score": suite_round_hard(agg, "template_curriculum", round_id, arm),
                    "heldout_template_accuracy": metric_mean(agg, "template_curriculum", round_id, arm, "heldout_template_accuracy"),
                }
            )
    write_json(out_dir / "curriculum_curve.json", rows)


def write_outputs(out_dir: Path, results: list[JobResult], args: argparse.Namespace, status: str, jobs: int) -> tuple[dict[str, dict[str, object]], list[str]]:
    agg = aggregate(results)
    labels = verdict(agg) if results else ["PARTIAL_NO_COMPLETED_JOBS"]
    collapse_round = first_collapse_round(agg)
    summary = {
        "status": status,
        "verdict": labels,
        "completed_jobs": len(results),
        "first_collapse_round": collapse_round,
        "config": {
            "seeds": args.seeds,
            "suite": args.suite,
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
    write_suite_curve(out_dir, agg)
    write_curriculum_curve(out_dir, agg)

    lines = [
        "# TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS Report",
        "",
        f"- Status: `{status}`",
        f"- Verdict: `{', '.join(labels)}`",
        f"- Completed jobs: `{len(results)}`",
        f"- Jobs: `{jobs}`",
        f"- First collapse round: `{collapse_round}`",
        "",
        "## Suite Summary",
        "",
        "| Suite | Round | Arm | Frame | Ledger | Hard | Role | NoopFrame | NoopNoMut | Heldout |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, round_id in available_suite_rounds(agg):
        for arm in sorted({str(data["arm"]) for data in agg.values() if data["suite"] == suite and int(data["curriculum_round"]) == round_id}):
            lines.append(
                "| `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                    suite,
                    round_id,
                    arm,
                    metric_mean(agg, suite, round_id, arm, "event_frame_exact_accuracy"),
                    metric_mean(agg, suite, round_id, arm, "ledger_answer_accuracy"),
                    suite_round_hard(agg, suite, round_id, arm),
                    metric_mean(agg, suite, round_id, arm, "role_binding_accuracy"),
                    metric_mean(agg, suite, round_id, arm, "noop_frame_accuracy"),
                    metric_mean(agg, suite, round_id, arm, "noop_no_mutation_accuracy"),
                    metric_mean(agg, suite, round_id, arm, "heldout_template_accuracy"),
                )
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg, labels


def write_doc_result(agg: dict[str, dict[str, object]], labels: list[str], args: argparse.Namespace, jobs: int, completed_jobs: int) -> None:
    lines = [
        "# TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS Result",
        "",
        "## Run",
        "",
        "```text",
        f"seeds={args.seeds}",
        f"suite={args.suite}",
        f"train_examples={args.train_examples}",
        f"eval_examples={args.eval_examples}",
        f"epochs={args.epochs}",
        f"jobs={jobs}",
        f"completed_jobs={completed_jobs}",
        f"first_collapse_round={first_collapse_round(agg)}",
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Suite Summary",
        "",
        "| Suite | Round | Arm | Frame | Ledger | Hard | Heldout |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for suite, round_id in available_suite_rounds(agg):
        for arm in sorted({str(data["arm"]) for data in agg.values() if data["suite"] == suite and int(data["curriculum_round"]) == round_id}):
            lines.append(
                "| `{}` | `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                    suite,
                    round_id,
                    arm,
                    metric_mean(agg, suite, round_id, arm, "event_frame_exact_accuracy"),
                    metric_mean(agg, suite, round_id, arm, "ledger_answer_accuracy"),
                    suite_round_hard(agg, suite, round_id, arm),
                    metric_mean(agg, suite, round_id, arm, "heldout_template_accuracy"),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a controlled factor-isolation toy parser probe. It is not full natural-language segmentation, symbol grounding, or a PrismionCell test.",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(args: argparse.Namespace, suite_rounds: list[tuple[str, int]]) -> None:
    samples: list[v3.StoryExample] = []
    for suite, round_id in suite_rounds[:4]:
        train, eval_set = build_suite_dataset(2026, min(args.train_examples, 12), min(args.eval_examples, 12), suite, round_id)
        samples.extend(train[:3] + eval_set[:6])
    write_jsonl(
        args.out_dir / "examples_sample.jsonl",
        [
            {"story_id": story.story_id, "text": story.row.text, "frames": [asdict(frame) for frame in story.frames], "tags": story.tags, "flags": story.flags, "suite_group": story.contrast_group_id}
            for story in samples[:48]
        ],
    )
    write_jsonl(
        args.out_dir / "hard_contrast_cases.jsonl",
        [
            {"story_id": story.story_id, "text": story.row.text, "frames": [asdict(frame) for frame in story.frames], "tags": story.tags, "group": story.contrast_group_id}
            for story in samples
            if any(tag in story.tags for tag in ("role_binding", "negation_modal", "template_curriculum", "heldout_template", "same_token_role_swap"))
        ],
    )


def run_all(args: argparse.Namespace, jobs: int) -> tuple[dict[str, dict[str, object]], list[str], list[JobResult]]:
    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    suite_rounds = selected_suite_rounds(args.suite)
    write_json(args.out_dir / "queue.json", [{"suite": suite, "curriculum_round": round_id, "arm": arm, "seed": seed} for suite, round_id in suite_rounds for seed in seeds for arm in arms])
    write_samples(args, suite_rounds)
    progress_path = args.out_dir / "progress.jsonl"
    metrics_path = args.out_dir / "metrics.jsonl"
    job_progress_root = args.out_dir / "job_progress"
    queue = [(suite, round_id, arm, seed) for suite, round_id in suite_rounds for seed in seeds for arm in arms]
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_start", "jobs": jobs, "total_jobs": len(queue), "suite": args.suite})
    results: list[JobResult] = []
    write_outputs(args.out_dir, results, args, "partial", jobs)

    if jobs <= 1:
        for suite, round_id, arm, seed in queue:
            result = run_job(suite, round_id, arm, seed, args, job_progress_root)
            results.append(result)
            append_jsonl(metrics_path, result_record(result))
            write_outputs(args.out_dir, results, args, "partial", jobs)
            append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "suite": suite, "round": round_id, "arm": arm, "seed": seed, "completed_jobs": len(results)})
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_meta = {}
            pending = set()
            for suite, round_id, arm, seed in queue:
                append_jsonl(progress_path, {"time": now_iso(), "event": "job_start", "suite": suite, "round": round_id, "arm": arm, "seed": seed})
                future = pool.submit(run_job, suite, round_id, arm, seed, args, job_progress_root)
                future_meta[future] = (suite, round_id, arm, seed)
                pending.add(future)
            while pending:
                done, pending = wait(pending, timeout=args.heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    append_jsonl(progress_path, {"time": now_iso(), "event": "heartbeat", "completed_jobs": len(results), "pending_jobs": len(pending)})
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    continue
                for future in done:
                    suite, round_id, arm, seed = future_meta[future]
                    result = future.result()
                    results.append(result)
                    append_jsonl(metrics_path, result_record(result))
                    write_outputs(args.out_dir, results, args, "partial", jobs)
                    append_jsonl(progress_path, {"time": now_iso(), "event": "job_done", "suite": suite, "round": round_id, "arm": arm, "seed": seed, "completed_jobs": len(results), "pending_jobs": len(pending)})

    agg, labels = write_outputs(args.out_dir, results, args, "complete", jobs)
    append_jsonl(progress_path, {"time": now_iso(), "event": "run_complete", "verdict": labels, "completed_jobs": len(results)})
    return agg, labels, results


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(str(args.jobs))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if CONTRACT.exists():
        shutil.copyfile(CONTRACT, args.out_dir / "contract_snapshot.md")
    agg, labels, results = run_all(args, jobs)
    write_doc_result(agg, labels, args, jobs, len(results))
    print(json.dumps({"verdict": labels, "out": str(args.out_dir), "completed_jobs": len(results), "first_collapse_round": first_collapse_round(agg)}, indent=2))


if __name__ == "__main__":
    main()
