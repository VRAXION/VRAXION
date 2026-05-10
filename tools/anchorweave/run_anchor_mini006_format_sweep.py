#!/usr/bin/env python3
"""Parallel ANCHOR-MINI-006 symbolic process-format sweep."""

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


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_006_FORMAT_SWEEP_CONTRACT.md")
FORMAT_ARMS = [
    "ANSWER_ONLY",
    "ORACLE_MATCH_BITS",
    "RAW_SYMBOLIC_PROCESS",
    "PROSE_PROCESS",
    "INNER_MONOLOGUE_PROCESS",
    "STRICT_JSON_PROCESS",
    "FLAT_KEY_VALUE_PROCESS",
    "RELATIONAL_TRIPLES_PROCESS",
    "ACTION_OUTCOME_TABLE_PROCESS",
    "RELATION_PLUS_ACTION_PROCESS",
    "COMPACT_HYBRID_PROCESS",
    "SHUFFLED_COMPACT_HYBRID",
]
PRACTICAL_FORMATS = [
    "PROSE_PROCESS",
    "INNER_MONOLOGUE_PROCESS",
    "STRICT_JSON_PROCESS",
    "FLAT_KEY_VALUE_PROCESS",
    "RELATIONAL_TRIPLES_PROCESS",
    "ACTION_OUTCOME_TABLE_PROCESS",
    "RELATION_PLUS_ACTION_PROCESS",
    "COMPACT_HYBRID_PROCESS",
]
CANDIDATE_COUNT = 4
CATEGORY_COUNT = 4
GOAL_COUNT = 16
EFFECT_COUNT = 24

JOB_COMPLETE = "ANCHOR_MINI_006_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_006_JOB_INVALID_STRESS"

STATUS_STRONG = "ANCHOR_MINI_006_FORMAT_STRONG_SIGNAL"
STATUS_WEAK = "ANCHOR_MINI_006_FORMAT_WEAK_SIGNAL"
STATUS_NO_WINNER = "ANCHOR_MINI_006_FORMAT_NO_CLEAR_WINNER"
STATUS_INVALID = "ANCHOR_MINI_006_FORMAT_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_006_FORMAT_RESOURCE_BLOCKED"


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
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-006 format sweep.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2125")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--format-arms", default=",".join(FORMAT_ARMS))
    parser.add_argument("--train-examples", default="1024")
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--surface", default="0.90/0.90")
    parser.add_argument("--max-steps", default="1600")
    parser.add_argument("--proposals", type=int, default=9)
    parser.add_argument("--edge-cap", type=int, default=20)
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
        "evolve_anchor_mini006.exe" if os.name == "nt" else "evolve_anchor_mini006"
    )
    if skip_build:
        if not exe.exists():
            raise FileNotFoundError(f"missing built example: {exe}")
        return exe
    subprocess.run(
        ["cargo", "build", "--release", "-p", "instnct-core", "--example", "evolve_anchor_mini006"],
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


def make_balanced_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
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
        shifted_bits = [int(category == (goal_category + 1) % CATEGORY_COUNT) for category in effect_categories]
        rows.append(
            {
                "example_id": f"{split}_{index:05d}",
                "split": split,
                "goal_id": goal_id,
                "goal_category": goal_category,
                "effect_ids": effect_ids,
                "effect_categories": effect_categories,
                "surface_priors": make_surface_priors(shortcut, rng),
                "answer_label": answer_slot,
                "match_bits": match_bits,
                "surface_shortcut_label": shortcut,
                "surface_shortcut_is_gold": shortcut == answer_slot,
                "shuffled_goal_category": (goal_category + 1) % CATEGORY_COUNT,
                "shuffled_effect_categories": effect_categories,
                "shuffled_match_bits": shifted_bits,
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
    )
    eval_rows = make_balanced_examples(
        split="eval",
        count=eval_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_path,
        {
            "metadata": {
                "source": "run_anchor_mini006_format_sweep.py",
                "mini": "ANCHOR-MINI-006",
                "seed": seed,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "train_surface_gold_prob": train_surface_gold_prob,
                "eval_surface_wrong_prob": eval_surface_wrong_prob,
                "category_distractors": "balanced_one_per_non_goal_category",
                "wrong_process_control": "goal_category_plus_one_mod_4",
            },
            "train_examples": train_rows,
            "eval_examples": eval_rows,
        },
    )


def format_process_text(format_arm: str, example: dict[str, Any]) -> str:
    goal = example["goal_category"]
    effects = example["effect_categories"]
    surface = [round(value, 3) for value in example["surface_priors"]]
    matches = [int(category == goal) for category in effects]
    shifted = [int(category == (goal + 1) % CATEGORY_COUNT) for category in effects]
    if format_arm == "ANSWER_ONLY":
        return ""
    if format_arm == "ORACLE_MATCH_BITS":
        return f"match_bits={matches}"
    if format_arm == "RAW_SYMBOLIC_PROCESS":
        return f"goal={goal}; effects={effects}; match={matches}"
    if format_arm == "PROSE_PROCESS":
        return (
            f"The goal asks for category {goal}. The candidates have effect categories {effects}. "
            "The useful candidate is the one whose effect category supports the goal, while surface salience should not decide."
        )
    if format_arm == "INNER_MONOLOGUE_PROCESS":
        return (
            f"I see goal category {goal}, and the candidates feel tempting because their surface priors are {surface}. "
            f"I keep checking the effect categories {effects}, then I remind myself not to follow the shiny surface cue. "
            "The right move is whatever actually fits the goal, even if the shortcut is loud."
        )
    if format_arm == "STRICT_JSON_PROCESS":
        return json.dumps(
            {
                "implicit_job": "select_candidate_by_goal_effect_fit",
                "goal": {"category": goal},
                "candidates": [
                    {"slot": idx, "effect_category": category, "matches_goal": bool(matches[idx])}
                    for idx, category in enumerate(effects)
                ],
                "decision_rule": "choose_matches_goal_true",
            },
            sort_keys=True,
        )
    if format_arm == "FLAT_KEY_VALUE_PROCESS":
        return f"job=select_fit\ngoal_category={goal}\neffect_categories={effects}\nsurface_priors={surface}"
    if format_arm == "RELATIONAL_TRIPLES_PROCESS":
        triples = [f"goal -> requires -> category_{goal}"]
        for idx, category in enumerate(effects):
            triples.append(f"candidate_{idx} -> has_effect -> category_{category}")
            triples.append(f"candidate_{idx} -> matches_goal -> {bool(matches[idx])}")
        return "\n".join(triples)
    if format_arm == "ACTION_OUTCOME_TABLE_PROCESS":
        rows = ["candidate | effect_category | surface_prior | expected_policy"]
        for idx, category in enumerate(effects):
            policy = "choose" if matches[idx] else "reject"
            rows.append(f"{idx} | {category} | {surface[idx]} | {policy}")
        return "\n".join(rows)
    if format_arm == "RELATION_PLUS_ACTION_PROCESS":
        return (
            format_process_text("RELATIONAL_TRIPLES_PROCESS", example)
            + "\n"
            + format_process_text("ACTION_OUTCOME_TABLE_PROCESS", example)
        )
    if format_arm == "COMPACT_HYBRID_PROCESS":
        return (
            f"ImplicitJob: select candidate by goal/effect fit.\n"
            f"Salience: goal_category={goal}; ignore surface shortcut.\n"
            f"Relations: effects={effects}; match={matches}.\n"
            "ActionOutcome: matching effect supports the goal; nonmatching effect is a trap.\n"
            "DecisionRule: choose the matching candidate."
        )
    if format_arm == "SHUFFLED_COMPACT_HYBRID":
        return (
            f"ImplicitJob: select candidate by shifted wrong goal/effect fit.\n"
            f"Salience: shifted_goal_category={(goal + 1) % CATEGORY_COUNT}; ignore true match.\n"
            f"Relations: effects={effects}; shifted_match={shifted}.\n"
            "ActionOutcome: shifted matching effect is treated as useful.\n"
            "DecisionRule: choose the shifted matching candidate."
        )
    raise ValueError(f"unknown format arm: {format_arm}")


def token_count(text: str) -> int:
    if not text:
        return 0
    normalized = text
    for char in "{}[]():,;|\"'`=\n":
        normalized = normalized.replace(char, " ")
    return len([part for part in normalized.split(" ") if part])


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def format_token_stats(dataset_path: Path, format_arm: str) -> dict[str, float]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload["eval_examples"][:200]
    counts = [token_count(format_process_text(format_arm, row)) for row in rows]
    return {
        "format_token_count_mean": sum(counts) / len(counts) if counts else 0.0,
        "format_token_count_p95": percentile(counts, 0.95),
    }


def surface_id(raw: str) -> str:
    return raw.replace("/", "_").replace(".", "p")


def job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['format_arm']}_seed{job['seed']}_n{job['train_examples']}"
        f"_s{surface_id(job['surface'])}_steps{job['max_steps']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    arms = parse_csv(args.format_arms)
    unknown = sorted(set(arms) - set(FORMAT_ARMS))
    if unknown:
        raise ValueError(f"unknown format arms: {unknown}")
    train_sizes = parse_csv(args.train_examples, int)
    max_steps_values = parse_csv(args.max_steps, int)
    surfaces = parse_csv(args.surface)
    queue: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            for train_examples in train_sizes:
                for max_steps in max_steps_values:
                    for surface in surfaces:
                        train_prob_raw, eval_prob_raw = surface.split("/", 1)
                        job = {
                            "format_arm": arm,
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
        f"seed{job['seed']}_n{job['train_examples']}_eval{job['eval_examples']}_s{surface_id(job['surface'])}.json"
    )
    with lock:
        make_dataset(
            dataset_path=dataset_path,
            seed=job["seed"],
            train_examples=job["train_examples"],
            eval_examples=job["eval_examples"],
            train_surface_gold_prob=job["train_surface_gold_prob"],
            eval_surface_wrong_prob=job["eval_surface_wrong_prob"],
        )
        token_stats = format_token_stats(dataset_path, job["format_arm"])
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
            "--format-arm",
            job["format_arm"],
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
            "format_tokens": token_stats,
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
            "format_tokens": token_stats,
            "feature_visibility": payload["feature_visibility"],
            "edge_count": payload["metrics"]["edge_count"],
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


def is_practical_format(arm: str) -> bool:
    return arm in PRACTICAL_FORMATS


def useful(row: dict[str, Any], answer: dict[str, Any], shuffled: dict[str, Any]) -> bool:
    return (
        row["ood_accuracy"] >= answer["ood_accuracy"] + 0.25
        and row["ood_accuracy"] >= shuffled["ood_accuracy"] + 0.25
        and row["shortcut_trap_rate"] <= 0.25
        and row["true_process_bit_accuracy"] >= 0.90
        and not row["oracle_match_visible"]
    )


def choose_best_practical(by_arm: dict[str, dict[str, Any]], useful_arms: list[str]) -> str | None:
    candidates = [arm for arm in useful_arms if is_practical_format(arm)]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda arm: (
            by_arm[arm]["ood_accuracy"],
            -by_arm[arm]["shortcut_trap_rate"],
            by_arm[arm]["format_efficiency"],
            -by_arm[arm]["format_token_count_p95"],
        ),
        reverse=True,
    )[0]


def summarize(rows: list[dict[str, Any]], queue_size: int, budget_reached: bool) -> dict[str, Any]:
    completed = [row for row in rows if row.get("event") == "job_complete"]
    valid = [row for row in completed if row["status"] == JOB_COMPLETE]
    blocked = [row for row in completed if row["status"] == STATUS_BLOCKED]
    invalid = [row for row in completed if row["status"] == JOB_INVALID]
    by_arm: dict[str, dict[str, Any]] = {}
    present_arms = [arm for arm in FORMAT_ARMS if any(row["job"]["format_arm"] == arm for row in completed)]
    for arm in present_arms:
        subset = [row for row in completed if row["job"]["format_arm"] == arm]
        valid_subset = [row for row in subset if row["status"] == JOB_COMPLETE]
        token_mean = mean([row["format_tokens"]["format_token_count_mean"] for row in valid_subset])
        ood = mean([row["metrics"]["answer_eval_ood_accuracy"] for row in valid_subset])
        by_arm[arm] = {
            "jobs": len(subset),
            "valid_jobs": len(valid_subset),
            "blocked_jobs": len([row for row in subset if row["status"] == STATUS_BLOCKED]),
            "invalid_jobs": len([row for row in subset if row["status"] == JOB_INVALID]),
            "ood_accuracy": ood,
            "shortcut_trap_rate": mean([row["metrics"]["shortcut_trap_rate"] for row in valid_subset]),
            "process_bit_accuracy": mean([row["metrics"]["process_bit_accuracy"] for row in valid_subset]),
            "process_exact_row_accuracy": mean([row["metrics"]["process_exact_row_accuracy"] for row in valid_subset]),
            "true_process_bit_accuracy": mean([row["metrics"]["true_process_bit_accuracy"] for row in valid_subset]),
            "true_process_exact_row_accuracy": mean([row["metrics"]["true_process_exact_row_accuracy"] for row in valid_subset]),
            "format_token_count_mean": token_mean,
            "format_token_count_p95": mean([row["format_tokens"]["format_token_count_p95"] for row in valid_subset]),
            "format_efficiency": ood / max(token_mean, 1.0),
            "oracle_match_visible": arm == "ORACLE_MATCH_BITS",
            "edge_count": mean([row["metrics"]["edge_count"] for row in valid_subset]),
        }
    answer = by_arm.get("ANSWER_ONLY", {"ood_accuracy": 0.0})
    oracle = by_arm.get("ORACLE_MATCH_BITS", {"ood_accuracy": 0.0, "shortcut_trap_rate": 1.0})
    shuffled = by_arm.get("SHUFFLED_COMPACT_HYBRID", {"ood_accuracy": 0.0})
    useful_arms = [
        arm for arm, arm_row in by_arm.items()
        if arm not in {"ANSWER_ONLY", "ORACLE_MATCH_BITS", "SHUFFLED_COMPACT_HYBRID"}
        and useful(arm_row, answer, shuffled)
    ]
    best_practical = choose_best_practical(by_arm, useful_arms)
    valid_seed_count = len({row["job"]["seed"] for row in valid})
    compact = by_arm.get("COMPACT_HYBRID_PROCESS", {"ood_accuracy": 0.0})
    relation_action = by_arm.get("RELATION_PLUS_ACTION_PROCESS", {"ood_accuracy": 0.0})
    structured_best = max(compact["ood_accuracy"], relation_action["ood_accuracy"])
    prose_or_inner_best = best_practical in {"PROSE_PROCESS", "INNER_MONOLOGUE_PROCESS"}
    prose_inner_allowed = not prose_or_inner_best or by_arm[best_practical]["ood_accuracy"] > structured_best
    conditions = {
        "valid_seeds_at_least_80": valid_seed_count >= 80,
        "oracle_upper_bound_good": oracle["ood_accuracy"] >= 0.90 and oracle["shortcut_trap_rate"] <= 0.10,
        "has_useful_practical_format": best_practical is not None,
        "best_beats_answer_by_0p25": (
            best_practical is not None and by_arm[best_practical]["ood_accuracy"] >= answer["ood_accuracy"] + 0.25
        ),
        "best_beats_shuffled_by_0p25": (
            best_practical is not None and by_arm[best_practical]["ood_accuracy"] >= shuffled["ood_accuracy"] + 0.25
        ),
        "prose_inner_not_unfairly_best": prose_inner_allowed,
        "no_blocked_jobs": not blocked,
    }
    if not completed:
        final_status = STATUS_BLOCKED
    elif not valid and invalid:
        final_status = STATUS_INVALID
    elif all(conditions.values()):
        final_status = STATUS_STRONG
    elif conditions["has_useful_practical_format"]:
        final_status = STATUS_WEAK
    elif valid:
        final_status = STATUS_NO_WINNER
    else:
        final_status = STATUS_INVALID
    gaps = {
        "oracle_gap": oracle["ood_accuracy"] - (by_arm[best_practical]["ood_accuracy"] if best_practical else 0.0),
        "shuffled_control_gap": (
            by_arm[best_practical]["ood_accuracy"] - shuffled["ood_accuracy"] if best_practical else 0.0
        ),
    }
    return {
        "status": final_status,
        "completed_jobs": len(completed),
        "queue_size": queue_size,
        "budget_reached": budget_reached,
        "valid_jobs": len(valid),
        "invalid_jobs": len(invalid),
        "blocked_jobs": len(blocked),
        "valid_seed_count": valid_seed_count,
        "best_practical_format": best_practical,
        "useful_arms": useful_arms,
        "by_format_arm": by_arm,
        "gaps": gaps,
        "conditions": conditions,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-006 Process Format Sweep",
        "",
        f"Status: `{summary['status']}`",
        f"Best practical format: `{summary['best_practical_format']}`",
        f"Completed jobs: `{summary['completed_jobs']}` / `{summary['queue_size']}`",
        f"Valid jobs: `{summary['valid_jobs']}`",
        f"Valid seed count: `{summary['valid_seed_count']}`",
        f"Budget reached: `{summary['budget_reached']}`",
        "",
        "## By Format Arm",
        "",
        "| arm | jobs | valid | OOD acc | trap rate | true process bit | token mean | efficiency | edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in summary["by_format_arm"].items():
        lines.append(
            f"| `{arm}` | {row['jobs']} | {row['valid_jobs']} | "
            f"{row['ood_accuracy']:.3f} | {row['shortcut_trap_rate']:.3f} | "
            f"{row['true_process_bit_accuracy']:.3f} | {row['format_token_count_mean']:.1f} | "
            f"{row['format_efficiency']:.4f} | {row['edge_count']:.1f} |"
        )
    lines.extend(["", "## Useful Arms", ""])
    for arm in summary["useful_arms"]:
        lines.append(f"- `{arm}`")
    lines.extend(["", "## Gaps", ""])
    for key, value in summary["gaps"].items():
        lines.append(f"- `{key}`: `{value:.3f}`")
    lines.extend(["", "## Conditions", ""])
    for key, value in summary["conditions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a symbolic process-format control sweep. It does not prove literal prose, JSON, triples, or table serialization performance in an LLM.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> int:
    root = repo_root()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    contract = root / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, out / "contract_snapshot.md")
    queue = build_queue(args)
    write_json(out / "queue.json", queue)
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    completed = load_completed(metrics_path)
    remaining = [job for job in queue if job["job_id"] not in completed]
    exe = ensure_binary(root, args.skip_build)
    lock = threading.Lock()
    budget_seconds = None
    if args.budget_hours is not None:
        budget_seconds = args.budget_hours * 3600.0
    if args.budget_minutes is not None:
        budget_seconds = args.budget_minutes * 60.0
    started = time.time()
    budget_reached = False
    append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "time": started,
            "jobs": args.jobs,
            "queue_size": len(queue),
            "remaining": len(remaining),
        },
        lock,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        in_flight = {}
        iterator = iter(remaining)
        while True:
            while len(in_flight) < args.jobs:
                if budget_seconds is not None and time.time() - started >= budget_seconds:
                    budget_reached = True
                    break
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                future = executor.submit(
                    run_job,
                    root=root,
                    exe=exe,
                    out=out,
                    job=job,
                    progress_path=progress_path,
                    metrics_path=metrics_path,
                    lock=lock,
                )
                in_flight[future] = job["job_id"]
            if not in_flight:
                break
            for future in as_completed(list(in_flight), timeout=None):
                in_flight.pop(future)
                _ = future.result()
                break
            if budget_reached and not in_flight:
                break
    rows = read_metrics(metrics_path)
    summary = summarize(rows, len(queue), budget_reached)
    write_json(out / "summary.json", summary)
    write_json(out / "format_curve.json", summary["by_format_arm"])
    write_report(out / "report.md", summary)
    append_jsonl(progress_path, {"event": "run_done", "summary": summary, "time": time.time()}, lock)
    print(f"wrote {out}")
    print(f"status: {summary['status']}")
    print(f"best_practical_format: {summary['best_practical_format']}")
    return 0 if summary["status"] != STATUS_BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
