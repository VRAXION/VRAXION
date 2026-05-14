#!/usr/bin/env python3
"""Parallel ANCHOR-MINI-010 held-out serialization robustness sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_010_SERIALIZATION_ROBUSTNESS_CONTRACT.md")
CARRIERS = [
    "BYTE_DIRECT_ANSWER",
    "BYTE_AUX_PLAN_DIRECT_ANSWER",
    "BYTE_PLAN_FIRST",
    "BYTE_PLAN_FIRST_HYBRID",
    "BYTE_SHUFFLED_TEACHER",
    "BYTE_SHORTCUT_TEACHER",
    "BYTE_ORACLE_PLAN_VISIBLE",
]
CANDIDATE_COUNT = 4
CATEGORY_COUNT = 4
GOAL_COUNT = 16
EFFECT_COUNT = 24
SLOT_NAMES = ["A", "B", "C", "D"]
TRAIN_FORMAT = "canonical_fixed"
EVAL_FORMATS = [
    "canonical_fixed",
    "field_order_swap",
    "slot_order_perm",
    "alias_long",
    "noise_fields",
]
DECODER_MODES = ["schema_aware", "fixed_template_control"]

JOB_COMPLETE = "ANCHOR_MINI_010_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_010_JOB_INVALID_STRESS"

STATUS_STRONG = "ANCHOR_MINI_010_SERIALIZATION_STRONG_POSITIVE"
STATUS_WEAK = "ANCHOR_MINI_010_SERIALIZATION_WEAK_POSITIVE"
STATUS_TEMPLATE_ONLY = "ANCHOR_MINI_010_TEMPLATE_ONLY"
STATUS_ORACLE_ONLY = "ANCHOR_MINI_010_ORACLE_ONLY"
STATUS_NEGATIVE = "ANCHOR_MINI_010_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_010_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_010_RESOURCE_BLOCKED"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return sorted(dict.fromkeys(seeds))


def parse_csv(raw: str, cast: Any = str) -> list[Any]:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-010 serialization robustness sweep.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2125")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--carriers", default=",".join(CARRIERS))
    parser.add_argument("--decoder-modes", default=",".join(DECODER_MODES))
    parser.add_argument("--eval-formats", default=",".join(EVAL_FORMATS))
    parser.add_argument("--train-examples", default="1024")
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--surface", default="0.90/0.90")
    parser.add_argument("--max-steps", default="1800")
    parser.add_argument("--proposals", type=int, default=9)
    parser.add_argument("--edge-cap", type=int, default=24)
    parser.add_argument("--aux-weight", type=float, default=4.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None, help="debug cap after queue construction")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed(metrics_path: Path) -> set[str]:
    completed: set[str] = set()
    if not metrics_path.exists():
        return completed
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(row["job_id"])
    return completed


def ensure_binary(root: Path, skip_build: bool) -> Path:
    exe = root / "target" / "release" / "examples" / (
        "evolve_anchor_mini010.exe" if os.name == "nt" else "evolve_anchor_mini010"
    )
    if skip_build:
        if not exe.exists():
            raise FileNotFoundError(f"missing built example: {exe}")
        return exe
    subprocess.run(
        ["cargo", "build", "--release", "-p", "instnct-core", "--example", "evolve_anchor_mini010"],
        cwd=root,
        check=True,
    )
    if not exe.exists():
        raise FileNotFoundError(f"cargo build succeeded but binary is missing: {exe}")
    return exe


def effect_category(effect_id: int) -> int:
    return effect_id % CATEGORY_COUNT


def choose_surface_shortcut(
    *,
    split: str,
    answer_slot: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> int:
    wrong_slots = [slot for slot in range(CANDIDATE_COUNT) if slot != answer_slot]
    if split == "train":
        if rng.random() < train_surface_gold_prob:
            return answer_slot
        return rng.choice(wrong_slots)
    if split == "eval":
        if rng.random() < eval_surface_wrong_prob:
            return rng.choice(wrong_slots)
        return answer_slot
    raise RuntimeError(f"unknown split: {split}")


def make_surface_priors(shortcut_slot: int, rng: random.Random) -> list[float]:
    surface = [0.03 + 0.24 * rng.random() for _ in range(CANDIDATE_COUNT)]
    surface[shortcut_slot] = 0.82 + 0.17 * rng.random()
    return surface


def surface_buckets(surface_priors: list[float], shortcut_slot: int) -> list[int]:
    buckets: list[int] = []
    for slot, value in enumerate(surface_priors):
        if slot == shortcut_slot:
            buckets.append(9)
        else:
            buckets.append(max(0, min(2, int(value * 10))))
    return buckets


def serialize_task(
    goal_category: int,
    effect_categories: list[int],
    buckets: list[int],
    *,
    serialization_format: str,
) -> str:
    if serialization_format == "canonical_fixed":
        chunks = [f"G={goal_category}"]
        for slot, letter in enumerate(SLOT_NAMES):
            chunks.append(f"{letter}=E{effect_categories[slot]}:S{buckets[slot]}")
        return ";".join(chunks)
    if serialization_format == "field_order_swap":
        chunks = [f"G={goal_category}"]
        for slot, letter in enumerate(SLOT_NAMES):
            chunks.append(f"{letter}=S{buckets[slot]}:E{effect_categories[slot]}")
        return ";".join(chunks)
    if serialization_format == "slot_order_perm":
        order = [2, 0, 3, 1]
        chunks = [f"G={goal_category}"]
        for slot in order:
            letter = SLOT_NAMES[slot]
            chunks.append(f"{letter}=E{effect_categories[slot]}:S{buckets[slot]}")
        return ";".join(chunks)
    if serialization_format == "alias_long":
        chunks = [f"GOAL={goal_category}"]
        for slot, letter in enumerate(SLOT_NAMES):
            chunks.append(f"{letter}(EFFECT={effect_categories[slot]},SURFACE={buckets[slot]})")
        return "|".join(chunks)
    if serialization_format == "noise_fields":
        chunks = [f"N=7", f"G={goal_category}"]
        for slot, letter in enumerate(SLOT_NAMES):
            noise = (goal_category + effect_categories[slot] + buckets[slot] + slot) % 10
            chunks.append(f"{letter}=E{effect_categories[slot]}:S{buckets[slot]}:X{noise}")
        return ";".join(chunks)
    raise ValueError(f"unknown serialization format: {serialization_format}")


def make_balanced_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
    serialization_format: str,
) -> list[dict[str, Any]]:
    effects_by_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) == category]
        for category in range(CATEGORY_COUNT)
    }
    rows: list[dict[str, Any]] = []
    for index in range(count):
        goal_id = rng.randrange(GOAL_COUNT)
        goal_category = goal_id % CATEGORY_COUNT
        answer_slot = rng.randrange(CANDIDATE_COUNT)
        non_goal_categories = [category for category in range(CATEGORY_COUNT) if category != goal_category]
        rng.shuffle(non_goal_categories)
        effect_ids: list[int] = []
        distractor_index = 0
        for slot in range(CANDIDATE_COUNT):
            if slot == answer_slot:
                effect_ids.append(rng.choice(effects_by_category[goal_category]))
            else:
                effect_ids.append(rng.choice(effects_by_category[non_goal_categories[distractor_index]]))
                distractor_index += 1
        effect_categories = [effect_category(effect) for effect in effect_ids]
        match_bits = [int(category == goal_category) for category in effect_categories]
        shortcut = choose_surface_shortcut(
            split=split,
            answer_slot=answer_slot,
            rng=rng,
            train_surface_gold_prob=train_surface_gold_prob,
            eval_surface_wrong_prob=eval_surface_wrong_prob,
        )
        surface_priors = make_surface_priors(shortcut, rng)
        buckets = surface_buckets(surface_priors, shortcut)
        shifted_bits = [int(category == (goal_category + 1) % CATEGORY_COUNT) for category in effect_categories]
        shortcut_bits = [int(slot == shortcut) for slot in range(CANDIDATE_COUNT)]
        rows.append(
            {
                "example_id": f"{split}_{index:05d}",
                "split": split,
                "serialization_format": serialization_format,
                "task_bytes": serialize_task(
                    goal_category,
                    effect_categories,
                    buckets,
                    serialization_format=serialization_format,
                ),
                "goal_id": goal_id,
                "goal_category": goal_category,
                "effect_ids": effect_ids,
                "effect_categories": effect_categories,
                "surface_priors": surface_priors,
                "surface_buckets": buckets,
                "answer_label": answer_slot,
                "match_bits": match_bits,
                "surface_shortcut_label": shortcut,
                "surface_shortcut_is_gold": shortcut == answer_slot,
                "observed_shortcut_slot": shortcut,
                "shortcut_valid": shortcut == answer_slot,
                "shortcut_reject": None if shortcut == answer_slot else shortcut,
                "shuffled_goal_category": (goal_category + 1) % CATEGORY_COUNT,
                "shuffled_match_bits": shifted_bits,
                "shortcut_teacher_bits": shortcut_bits,
            }
        )
    return rows


def make_dataset(
    *,
    dataset_path: Path,
    seed: int,
    train_examples: int,
    eval_examples: int,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
    eval_format: str,
) -> None:
    if dataset_path.exists() and dataset_path.stat().st_size > 0:
        return
    rng = random.Random(seed)
    train_rows = make_balanced_examples(
        split="train",
        count=train_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
        serialization_format=TRAIN_FORMAT,
    )
    eval_rows = make_balanced_examples(
        split="eval",
        count=eval_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
        serialization_format=eval_format,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_path,
        {
            "metadata": {
                "source": "run_anchor_mini010_serialization_robustness.py",
                "mini": "ANCHOR-MINI-010",
                "seed": seed,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "train_surface_gold_prob": train_surface_gold_prob,
                "eval_surface_wrong_prob": eval_surface_wrong_prob,
                "train_serialization_format": TRAIN_FORMAT,
                "eval_serialization_format": eval_format,
                "serialization": "held_out_ascii_byte_record",
            },
            "train_examples": train_rows,
            "eval_examples": eval_rows,
        },
    )


def write_examples_sample(
    path: Path,
    *,
    seed: int,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
    eval_formats: list[str],
) -> None:
    rng = random.Random(seed)
    rows = make_balanced_examples(
        split="train",
        count=3,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
        serialization_format=TRAIN_FORMAT,
    )
    for eval_format in eval_formats:
        rows.extend(
            make_balanced_examples(
                split="eval",
                count=2,
                rng=rng,
                train_surface_gold_prob=train_surface_gold_prob,
                eval_surface_wrong_prob=eval_surface_wrong_prob,
                serialization_format=eval_format,
            )
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def surface_id(raw: str) -> str:
    return raw.replace("/", "_").replace(".", "p")


def job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['carrier']}_seed{job['seed']}_n{job['train_examples']}"
        f"_s{surface_id(job['surface'])}_{job['decoder_mode']}_{job['eval_format']}"
        f"_steps{job['max_steps']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    carriers = parse_csv(args.carriers)
    unknown = sorted(set(carriers) - set(CARRIERS))
    if unknown:
        raise ValueError(f"unknown carriers: {unknown}")
    decoder_modes = parse_csv(args.decoder_modes)
    unknown_decoders = sorted(set(decoder_modes) - set(DECODER_MODES))
    if unknown_decoders:
        raise ValueError(f"unknown decoder modes: {unknown_decoders}")
    eval_formats = parse_csv(args.eval_formats)
    unknown_formats = sorted(set(eval_formats) - set(EVAL_FORMATS))
    if unknown_formats:
        raise ValueError(f"unknown eval formats: {unknown_formats}")
    train_sizes = parse_csv(args.train_examples, int)
    max_steps_values = parse_csv(args.max_steps, int)
    surfaces = parse_csv(args.surface)
    queue: list[dict[str, Any]] = []
    for seed in seeds:
        for carrier in carriers:
            for decoder_mode in decoder_modes:
                for eval_format in eval_formats:
                    for train_examples in train_sizes:
                        for max_steps in max_steps_values:
                            for surface in surfaces:
                                train_prob_raw, eval_prob_raw = surface.split("/", 1)
                                job = {
                                    "carrier": carrier,
                                    "decoder_mode": decoder_mode,
                                    "eval_format": eval_format,
                                    "seed": seed,
                                    "train_examples": train_examples,
                                    "eval_examples": args.eval_examples,
                                    "train_surface_gold_prob": float(train_prob_raw),
                                    "eval_surface_wrong_prob": float(eval_prob_raw),
                                    "surface": surface,
                                    "max_steps": max_steps,
                                    "proposals": args.proposals,
                                    "edge_cap": args.edge_cap,
                                    "aux_weight": args.aux_weight,
                                }
                                job["job_id"] = job_id(job)
                                queue.append(job)
    if args.max_jobs is not None:
        queue = queue[: args.max_jobs]
    return queue


def run_job(
    *,
    root: Path,
    exe: Path,
    out: Path,
    job: dict[str, Any],
    progress_path: Path,
    metrics_path: Path,
    lock: threading.Lock,
) -> dict[str, Any]:
    jid = job["job_id"]
    job_dir = out / "jobs" / jid
    dataset_path = out / "datasets" / (
        f"seed{job['seed']}_n{job['train_examples']}_eval{job['eval_examples']}"
        f"_s{surface_id(job['surface'])}_{job['eval_format']}.json"
    )
    with lock:
        make_dataset(
            dataset_path=dataset_path,
            seed=job["seed"],
            train_examples=job["train_examples"],
            eval_examples=job["eval_examples"],
            train_surface_gold_prob=job["train_surface_gold_prob"],
            eval_surface_wrong_prob=job["eval_surface_wrong_prob"],
            eval_format=job["eval_format"],
        )
    job_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(progress_path, {"event": "job_start", "job_id": jid, "job": job, "time": time.time()}, lock)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    started = time.time()
    proc = subprocess.run(
        [
            str(exe),
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(job_dir),
            "--carrier",
            job["carrier"],
            "--decoder-mode",
            job["decoder_mode"],
            "--seed",
            str(job["seed"]),
            "--max-steps",
            str(job["max_steps"]),
            "--proposals",
            str(job["proposals"]),
            "--edge-cap",
            str(job["edge_cap"]),
            "--aux-weight",
            str(job["aux_weight"]),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - started
    report_path = job_dir / "report.json"
    if proc.returncode != 0 or not report_path.exists():
        row = {
            "event": "job_complete",
            "job_id": jid,
            "job": job,
            "status": STATUS_BLOCKED,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 3),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-2000:],
            "time": time.time(),
        }
    else:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "event": "job_complete",
            "job_id": jid,
            "job": job,
            "status": payload["status"],
            "elapsed_s": round(elapsed, 3),
            "stress": payload["stress"],
            "metrics": payload["metrics"],
            "feature_visibility": payload["feature_visibility"],
            "edge_count": payload["metrics"]["edge_count"],
            "decoder_mode": job["decoder_mode"],
            "eval_format": job["eval_format"],
            "report_path": str(report_path),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "time": time.time(),
        }
    append_jsonl(metrics_path, row, lock)
    append_jsonl(progress_path, {"event": "job_done", "job_id": jid, "status": row["status"], "time": time.time()}, lock)
    return row


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") in {JOB_COMPLETE, JOB_INVALID}]
    blocked = [row for row in rows if row.get("status") == STATUS_BLOCKED]
    valid = [row for row in completed if not row.get("stress", {}).get("invalid_stress")]

    def summarize(subset: list[dict[str, Any]], *, carrier: str | None = None) -> dict[str, Any]:
        if not subset:
            return {}
        return {
            "carrier": carrier,
            "jobs": len(subset),
            "valid": len([row for row in subset if not row.get("stress", {}).get("invalid_stress")]),
            "valid_seed_count": len({row["job"]["seed"] for row in subset if not row.get("stress", {}).get("invalid_stress")}),
            "answer_eval_accuracy": mean([row["metrics"]["answer_eval_accuracy"] for row in subset]),
            "shortcut_trap_rate": mean([row["metrics"]["shortcut_trap_rate"] for row in subset]),
            "observed_shortcut_accuracy": mean([row["metrics"]["observed_shortcut_accuracy"] for row in subset]),
            "shortcut_validity_accuracy": mean([row["metrics"]["shortcut_validity_accuracy"] for row in subset]),
            "invalid_shortcut_rejection_accuracy": mean(
                [row["metrics"]["invalid_shortcut_rejection_accuracy"] for row in subset]
            ),
            "policy_bit_accuracy": mean([row["metrics"]["policy_bit_accuracy"] for row in subset]),
            "plan_exact_row_accuracy": mean([row["metrics"]["plan_exact_row_accuracy"] for row in subset]),
            "answer_from_plan_consistency": mean(
                [row["metrics"]["answer_from_plan_consistency"] for row in subset]
            ),
            "byte_input_integrity": mean([row["metrics"]["byte_input_integrity"] for row in subset]),
            "decode_integrity": mean([row["metrics"]["decode_integrity"] for row in subset]),
            "edge_count": mean([row["metrics"]["edge_count"] for row in subset]),
        }

    by_carrier: dict[str, dict[str, Any]] = {}
    schema_valid = [
        row for row in valid if row["job"]["decoder_mode"] == "schema_aware"
    ]
    for carrier in CARRIERS:
        subset = [row for row in schema_valid if row["job"]["carrier"] == carrier]
        if not subset:
            continue
        by_carrier[carrier] = summarize(subset, carrier=carrier)

    by_format: dict[str, dict[str, Any]] = {}
    format_conditions: dict[str, dict[str, bool]] = {}
    for eval_format in EVAL_FORMATS:
        by_format[eval_format] = {}
        carrier_rows: dict[str, dict[str, Any]] = {}
        for carrier in CARRIERS:
            subset = [
                row
                for row in schema_valid
                if row["job"]["eval_format"] == eval_format and row["job"]["carrier"] == carrier
            ]
            if subset:
                carrier_rows[carrier] = summarize(subset, carrier=carrier)
        by_format[eval_format] = carrier_rows
        direct = carrier_rows.get("BYTE_DIRECT_ANSWER")
        plan = carrier_rows.get("BYTE_PLAN_FIRST")
        hybrid = carrier_rows.get("BYTE_PLAN_FIRST_HYBRID")
        shuffled = carrier_rows.get("BYTE_SHUFFLED_TEACHER")
        shortcut = carrier_rows.get("BYTE_SHORTCUT_TEACHER")
        oracle = carrier_rows.get("BYTE_ORACLE_PLAN_VISIBLE")
        format_conditions[eval_format] = {
            "valid_seeds_at_least_80": bool(plan and plan["valid_seed_count"] >= 80),
            "byte_input_integrity_all": bool(plan and plan["byte_input_integrity"] >= 1.0),
            "decode_integrity_all": bool(plan and plan["decode_integrity"] >= 1.0),
            "direct_shortcut_trap_gte_0p45": bool(direct and direct["shortcut_trap_rate"] >= 0.45),
            "plan_beats_direct_by_0p25": bool(
                plan and direct and plan["answer_eval_accuracy"] >= direct["answer_eval_accuracy"] + 0.25
            ),
            "plan_beats_shuffled_by_0p25": bool(
                plan and shuffled and plan["answer_eval_accuracy"] >= shuffled["answer_eval_accuracy"] + 0.25
            ),
            "plan_beats_shortcut_teacher_by_0p25": bool(
                plan and shortcut and plan["answer_eval_accuracy"] >= shortcut["answer_eval_accuracy"] + 0.25
            ),
            "plan_trap_rate_lte_0p25": bool(plan and plan["shortcut_trap_rate"] <= 0.25),
            "plan_policy_bit_gte_0p90": bool(plan and plan["policy_bit_accuracy"] >= 0.90),
            "plan_exact_row_gte_0p85": bool(plan and plan["plan_exact_row_accuracy"] >= 0.85),
            "plan_answer_consistency_gte_0p95": bool(plan and plan["answer_from_plan_consistency"] >= 0.95),
            "hybrid_directionally_positive": bool(
                hybrid
                and direct
                and hybrid["answer_eval_accuracy"] >= direct["answer_eval_accuracy"] + 0.15
                and hybrid["shortcut_trap_rate"] < direct["shortcut_trap_rate"]
            ),
            "oracle_upper_bound_good": bool(
                oracle and oracle["answer_eval_accuracy"] >= 0.90 and oracle["shortcut_trap_rate"] <= 0.10
            ),
            "learned_matches_oracle_or_close": bool(
                plan and oracle and plan["answer_eval_accuracy"] >= oracle["answer_eval_accuracy"] - 0.05
            ),
        }

    valid_seed_count = len({row["job"]["seed"] for row in schema_valid})
    direct = by_carrier.get("BYTE_DIRECT_ANSWER")
    aux_direct = by_carrier.get("BYTE_AUX_PLAN_DIRECT_ANSWER")
    plan = by_carrier.get("BYTE_PLAN_FIRST")
    oracle = by_carrier.get("BYTE_ORACLE_PLAN_VISIBLE")

    schema_all_formats_pass = bool(format_conditions) and all(
        all(values.values()) for values in format_conditions.values()
    )
    fixed_completed = [
        row for row in completed if row["job"]["decoder_mode"] == "fixed_template_control"
    ]
    fixed_canonical = [
        row
        for row in fixed_completed
        if row["job"]["eval_format"] == "canonical_fixed" and not row.get("stress", {}).get("invalid_stress")
    ]
    fixed_heldout = [
        row for row in fixed_completed if row["job"]["eval_format"] != "canonical_fixed"
    ]
    fixed_canonical_passes = bool(fixed_canonical) and len({row["job"]["seed"] for row in fixed_canonical}) >= 80
    fixed_heldout_decode_fails = bool(fixed_heldout) and all(
        row.get("stress", {}).get("decode_integrity") is False for row in fixed_heldout
    )
    conditions = {
        "valid_seeds_at_least_80": valid_seed_count >= 80,
        "schema_all_formats_pass": schema_all_formats_pass,
        "fixed_template_control_passes_canonical": fixed_canonical_passes,
        "fixed_template_control_loses_decode_on_heldout": fixed_heldout_decode_fails,
        "direct_shortcut_trap_gte_0p45": bool(direct and direct["shortcut_trap_rate"] >= 0.45),
        "plan_beats_direct_by_0p25": bool(
            plan and direct and plan["answer_eval_accuracy"] >= direct["answer_eval_accuracy"] + 0.25
        ),
        "plan_trap_rate_lte_0p25": bool(plan and plan["shortcut_trap_rate"] <= 0.25),
        "plan_policy_bit_gte_0p90": bool(plan and plan["policy_bit_accuracy"] >= 0.90),
        "plan_exact_row_gte_0p85": bool(plan and plan["plan_exact_row_accuracy"] >= 0.85),
        "plan_answer_consistency_gte_0p95": bool(plan and plan["answer_from_plan_consistency"] >= 0.95),
        "oracle_upper_bound_good": bool(
            oracle and oracle["answer_eval_accuracy"] >= 0.90 and oracle["shortcut_trap_rate"] <= 0.10
        ),
        "learned_matches_oracle_or_close": bool(
            plan and oracle and plan["answer_eval_accuracy"] >= oracle["answer_eval_accuracy"] - 0.05
        ),
        "aux_direct_not_enough": bool(
            aux_direct and plan and aux_direct["answer_eval_accuracy"] <= plan["answer_eval_accuracy"] - 0.20
        ),
        "no_blocked_jobs": not blocked,
    }
    strong_keys = [
        "valid_seeds_at_least_80",
        "schema_all_formats_pass",
        "fixed_template_control_passes_canonical",
        "fixed_template_control_loses_decode_on_heldout",
        "oracle_upper_bound_good",
        "learned_matches_oracle_or_close",
        "no_blocked_jobs",
    ]
    if blocked and not completed:
        status = STATUS_BLOCKED
    elif valid_seed_count < 80 and len({row["job"]["seed"] for row in completed}) >= 80:
        status = STATUS_INVALID
    elif all(conditions[key] for key in strong_keys):
        status = STATUS_STRONG
    elif fixed_canonical_passes and not schema_all_formats_pass:
        status = STATUS_TEMPLATE_ONLY
    elif conditions["oracle_upper_bound_good"] and not (
        conditions["plan_beats_direct_by_0p25"] and conditions["plan_policy_bit_gte_0p90"]
    ):
        status = STATUS_ORACLE_ONLY
    elif plan and direct and plan["answer_eval_accuracy"] > direct["answer_eval_accuracy"] + 0.10:
        status = STATUS_WEAK
    else:
        status = STATUS_NEGATIVE
    return {
        "status": status,
        "completed_jobs": len(completed),
        "valid_jobs": len(valid),
        "valid_seed_count": valid_seed_count,
        "blocked_jobs": len(blocked),
        "by_carrier": by_carrier,
        "by_format": by_format,
        "format_conditions": format_conditions,
        "conditions": conditions,
    }


def write_report(path: Path, summary: dict[str, Any], *, elapsed: float, queue_len: int) -> None:
    lines = [
        "# ANCHOR-MINI-010 Serialization Robustness Sweep",
        "",
        "## Verdict",
        "",
        "```text",
        summary["status"],
        "```",
        "",
        "## Summary",
        "",
        "```text",
        f"completed_jobs: {summary['completed_jobs']} / {queue_len}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"valid_seed_count: {summary['valid_seed_count']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        f"elapsed_seconds: {elapsed:.2f}",
        "```",
        "",
        "## Schema-Aware Aggregate Metrics",
        "",
        "| carrier | jobs | answer_eval | trap_rate | shortcut | validity | invalid_reject | policy_bit | plan_exact | consistency | byte_ok | decode_ok | edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for carrier in CARRIERS:
        row = summary["by_carrier"].get(carrier)
        if not row:
            continue
        lines.append(
            f"| `{carrier}` | {row['jobs']} | {row['answer_eval_accuracy']:.3f} | "
            f"{row['shortcut_trap_rate']:.3f} | {row['observed_shortcut_accuracy']:.3f} | "
            f"{row['shortcut_validity_accuracy']:.3f} | "
            f"{row['invalid_shortcut_rejection_accuracy']:.3f} | "
            f"{row['policy_bit_accuracy']:.3f} | {row['plan_exact_row_accuracy']:.3f} | "
            f"{row['answer_from_plan_consistency']:.3f} | {row['byte_input_integrity']:.3f} | "
            f"{row['decode_integrity']:.3f} | {row['edge_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Format Curve",
            "",
            "| eval_format | plan_acc | plan_trap | plan_policy | plan_exact | direct_trap | decode_ok | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for eval_format in EVAL_FORMATS:
        rows = summary["by_format"].get(eval_format, {})
        plan = rows.get("BYTE_PLAN_FIRST", {})
        direct = rows.get("BYTE_DIRECT_ANSWER", {})
        passed = all(summary["format_conditions"].get(eval_format, {}).values())
        lines.append(
            f"| `{eval_format}` | {plan.get('answer_eval_accuracy', 0.0):.3f} | "
            f"{plan.get('shortcut_trap_rate', 0.0):.3f} | "
            f"{plan.get('policy_bit_accuracy', 0.0):.3f} | "
            f"{plan.get('plan_exact_row_accuracy', 0.0):.3f} | "
            f"{direct.get('shortcut_trap_rate', 0.0):.3f} | "
            f"{plan.get('decode_integrity', 0.0):.3f} | {passed} |"
        )
    lines.extend(["", "## Conditions", "", "```text"])
    for key, value in summary["conditions"].items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "```",
            "",
            "## Claim Boundary",
            "",
            "This is a schema-aware byte-serialization robustness result. It does not",
            "prove learned parsing, natural-language AnchorCells, Qwen behavior, full",
            "INSTNCT recurrent behavior, or symbol grounding at scale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_result_doc(root: Path, summary: dict[str, Any], command: str, elapsed: float) -> None:
    path = root / "docs" / "research" / "ANCHOR_MINI_010_RESULT.md"
    lines = [
        "# ANCHOR-MINI-010 Result",
        "",
        "## Verdict",
        "",
        "```text",
        summary["status"],
        "```",
        "",
        "ANCHOR-MINI-010 tested whether process-first sparse routing survives",
        "held-out byte serialization variants under a schema-aware decoder.",
        "",
        "## Run",
        "",
        "```bash",
        command,
        "```",
        "",
        "Runtime:",
        "",
        "```text",
        f"{elapsed:.2f} seconds",
        "```",
        "",
        "## Summary",
        "",
        "```text",
        f"completed_jobs: {summary['completed_jobs']}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"valid_seed_count: {summary['valid_seed_count']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        "```",
        "",
        "| carrier | answer_eval | trap_rate | invalid_reject | policy_bit | plan_exact | consistency | byte_ok | decode_ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for carrier in CARRIERS:
        row = summary["by_carrier"].get(carrier)
        if not row:
            continue
        lines.append(
            f"| `{carrier}` | {row['answer_eval_accuracy']:.3f} | "
            f"{row['shortcut_trap_rate']:.3f} | "
            f"{row['invalid_shortcut_rejection_accuracy']:.3f} | "
            f"{row['policy_bit_accuracy']:.3f} | {row['plan_exact_row_accuracy']:.3f} | "
            f"{row['answer_from_plan_consistency']:.3f} | {row['byte_input_integrity']:.3f} | "
            f"{row['decode_integrity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Format Curve",
            "",
            "| eval_format | BYTE_PLAN_FIRST acc | trap | policy_bit | plan_exact | decode_ok | pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for eval_format in EVAL_FORMATS:
        rows = summary["by_format"].get(eval_format, {})
        plan = rows.get("BYTE_PLAN_FIRST", {})
        passed = all(summary["format_conditions"].get(eval_format, {}).values())
        lines.append(
            f"| `{eval_format}` | {plan.get('answer_eval_accuracy', 0.0):.3f} | "
            f"{plan.get('shortcut_trap_rate', 0.0):.3f} | "
            f"{plan.get('policy_bit_accuracy', 0.0):.3f} | "
            f"{plan.get('plan_exact_row_accuracy', 0.0):.3f} | "
            f"{plan.get('decode_integrity', 0.0):.3f} | {passed} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive result means the schema-aware byte carrier closes the surface",
            "shortcut through the PLAN route across held-out serialization variants.",
            "The fixed-template control guards against claiming that MINI-009 only",
            "worked because one byte layout was hardcoded.",
            "",
            "## Claim Boundary",
            "",
            "This is still a toy schema-aware byte-carrier result. It does not prove",
            "learned parsing, natural-language AnchorCells, or symbol grounding at scale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    exe = ensure_binary(root, args.skip_build)
    if (root / CONTRACT_FILE).exists():
        shutil.copyfile(root / CONTRACT_FILE, out / "contract_snapshot.md")
    queue = build_queue(args)
    write_json(out / "queue.json", queue)
    first_seed = parse_seed_spec(args.seeds)[0]
    first_surface = parse_csv(args.surface)[0]
    first_train_prob, first_eval_prob = first_surface.split("/", 1)
    eval_formats = parse_csv(args.eval_formats)
    write_examples_sample(
        out / "examples_sample.jsonl",
        seed=first_seed,
        train_surface_gold_prob=float(first_train_prob),
        eval_surface_wrong_prob=float(first_eval_prob),
        eval_formats=eval_formats,
    )
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    completed = load_completed(metrics_path)
    pending = [job for job in queue if job["job_id"] not in completed]
    budget_seconds = None
    if args.budget_hours is not None:
        budget_seconds = args.budget_hours * 3600
    if args.budget_minutes is not None:
        budget_seconds = args.budget_minutes * 60
    started = time.time()
    lock = threading.Lock()
    append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "time": started,
            "pending_jobs": len(pending),
            "completed_existing": len(completed),
            "args": vars(args) | {"out": str(args.out)},
        },
        lock,
    )
    completed_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = []
        for job in pending:
            if budget_seconds is not None and time.time() - started >= budget_seconds:
                break
            futures.append(
                executor.submit(
                    run_job,
                    root=root,
                    exe=exe,
                    out=out,
                    job=job,
                    progress_path=progress_path,
                    metrics_path=metrics_path,
                    lock=lock,
                )
            )
        for future in as_completed(futures):
            completed_rows.append(future.result())
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elapsed = time.time() - started
    summary = aggregate(rows)
    summary["elapsed_seconds"] = elapsed
    summary["queue_jobs"] = len(queue)
    write_json(out / "summary.json", summary)
    write_json(out / "format_curve.json", summary["by_format"])
    write_report(out / "report.md", summary, elapsed=elapsed, queue_len=len(queue))
    append_jsonl(progress_path, {"event": "run_done", "status": summary["status"], "time": time.time()}, lock)
    if summary["status"] in {STATUS_STRONG, STATUS_WEAK}:
        command = "python " + " ".join([Path(__file__).as_posix(), *argv])
        write_result_doc(root, summary, command, elapsed)
    print(summary["status"])
    print(f"report: {out / 'report.md'}")
    return 0 if summary["status"] != STATUS_BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
