#!/usr/bin/env python3
"""Generate AWFT-001 forced-choice natural-language artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import generate_awft001


FORBIDDEN_CANDIDATE_TEXT = {"correct", "wrong", "trap"}
VIEWS = {
    "plain_qa_nl": "train_plain_qa_nl.jsonl",
    "rich_prose_nl": "train_rich_prose_nl.jsonl",
    "anchorweave_nl": "train_anchorweave_nl.jsonl",
    "shuffled_anchorweave_nl": "train_shuffled_anchorweave_nl.jsonl",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AWFT-001-FC artifacts.")
    parser.add_argument("--out", type=Path, default=Path("target/anchorweave/awft001_fc"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


def beads_text(scenario: dict[str, Any]) -> str:
    return "; ".join(scenario["perceptual_beads"])


def actions_text(scenario: dict[str, Any]) -> str:
    return "; ".join(
        f"{action['action_id']} means {action['description']}"
        for action in scenario["available_actions"]
    )


def plain_prompt_nl(scenario: dict[str, Any]) -> str:
    return (
        "You are deciding the first move at a route-memory corner.\n\n"
        f"Source memory: {scenario['source_memory']}\n\n"
        f"Current scene: {scenario['current_scene']}\n\n"
        f"Observed cues: {beads_text(scenario)}.\n\n"
        f"Available actions: {actions_text(scenario)}.\n\n"
        "Answer with one natural-language sentence that names the best first move "
        "and gives the reason."
    )


def rich_prompt_nl(scenario: dict[str, Any]) -> str:
    return (
        "A route memory is being checked against the current corner, not turned "
        "into a simple rule. "
        f"The remembered scene says: {scenario['source_memory']} "
        f"The current scene says: {scenario['current_scene']} "
        f"The visible cue bundle is: {beads_text(scenario)}. "
        "Some cues may be vivid without being decisive. "
        "Choose the first move in one natural sentence and explain why."
    )


def anchorweave_prompt_nl(scenario: dict[str, Any]) -> str:
    payload = {
        "episode": {
            "domain": "navigation_route_memory_disambiguation",
            "source_memory": scenario["source_memory"],
            "current_scene": scenario["current_scene"],
        },
        "relational_graph": {
            "nodes": [
                "source_memory",
                "current_scene",
                "perceptual_beads",
                "route_geometry",
                "first_move",
            ],
            "candidate_edges": [
                {"source": "source_memory", "relation": "contains", "target": "landmark_group"},
                {"source": "current_scene", "relation": "offers", "target": "cue_bundle"},
                {"source": "cue_bundle", "relation": "tests", "target": "route_memory"},
                {"source": "route_memory", "relation": "constrains", "target": "first_move"},
            ],
        },
        "salience_candidates": scenario["perceptual_beads"],
        "available_actions": scenario["available_actions"],
        "task": (
            "Choose the best first move as one natural-language sentence. "
            "Preserve the boundary between a useful memory cue and a sufficient route rule."
        ),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def prompt_for_view(scenario: dict[str, Any], view: str) -> str:
    if view == "plain_qa_nl":
        return plain_prompt_nl(scenario)
    if view == "rich_prose_nl":
        return rich_prompt_nl(scenario)
    if view in {"anchorweave_nl", "shuffled_anchorweave_nl"}:
        return anchorweave_prompt_nl(scenario)
    raise ValueError(f"unknown view: {view}")


def train_completion(scenario: dict[str, Any]) -> str:
    family = scenario["family"]
    if family == "exact_match":
        return (
            "The best first move is to take the remembered turn after checking the "
            "route-defining cues, because the landmark group, route geometry, and "
            "street marker line up together."
        )
    if family == "viewpoint_shift":
        return (
            "The best first move is to slow down and verify another route-defining "
            "cue, because a changed viewpoint can hide a cue without proving the "
            "corner is different."
        )
    if family == "near_miss":
        return (
            "The best first move is to check another route-defining cue before "
            "moving, because the partial landmark match conflicts with the remembered "
            "route geometry."
        )
    return (
        "The best first move is to verify the landmark group before turning, because "
        "one familiar object alone is only a cue and not enough evidence for the route."
    )


def shuffled_train_completion(scenario: dict[str, Any]) -> str:
    family = scenario["family"]
    if family == "exact_match":
        return (
            "The best first move is to leave the corner and backtrack, because any "
            "small uncertainty should override the matching route cues."
        )
    if family == "viewpoint_shift":
        return (
            "The best first move is to reject this place immediately, because a hidden "
            "or missing cue should be treated as proof that the route is different."
        )
    return (
        "The best first move is to follow the remembered turn immediately, because "
        "one familiar landmark and the recalled scene are enough to reuse the route."
    )


def eval_correct_candidate(scenario: dict[str, Any]) -> tuple[str, str]:
    family = scenario["family"]
    if family == "exact_match":
        return (
            "The best first move is to continue through the remembered turn, because "
            "several route-defining cues align together rather than just one cue.",
            "confirm",
        )
    if family == "viewpoint_shift":
        return (
            "The best first move is to pause and check one more route-defining cue, "
            "because the missing detail may be explained by the changed viewpoint.",
            "defer",
        )
    if family == "near_miss":
        return (
            "The best first move is to verify the route geometry before turning, "
            "because the partial landmark match conflicts with other route cues.",
            "defer",
        )
    return (
        "The best first move is to check another route-defining landmark before "
        "turning, because one familiar object is not enough to identify the corner.",
        "defer",
    )


def eval_candidates(scenario: dict[str, Any]) -> list[dict[str, str]]:
    correct_text, correct_commitment = eval_correct_candidate(scenario)
    candidates = [
        {
            "candidate_id": "correct_grounded_action",
            "trap_type": "grounded",
            "commitment_kind": correct_commitment,
            "text": correct_text,
        },
        {
            "candidate_id": "single_landmark_shortcut",
            "trap_type": "shortcut",
            "commitment_kind": "confirm",
            "text": (
                "The best first move is to follow the remembered turn immediately, "
                "because one familiar landmark is enough to reuse the route."
            ),
        },
        {
            "candidate_id": "premature_rejection",
            "trap_type": "overreject",
            "commitment_kind": "reject",
            "text": (
                "The best first move is to leave the corner and backtrack now, "
                "because one missing or unclear cue proves this is a different place."
            ),
        },
        {
            "candidate_id": "incidental_context_rule",
            "trap_type": "incidental",
            "commitment_kind": "confirm",
            "text": (
                "The best first move is to trust the recalled scene mood and continue, "
                "because the incidental memory cue brings the route back clearly."
            ),
        },
    ]
    return candidates


def train_row(scenario: dict[str, Any], view: str, corrupted: bool = False) -> dict[str, Any]:
    prompt = prompt_for_view(scenario, view)
    completion = shuffled_train_completion(scenario) if corrupted else train_completion(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "view": view,
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "split": scenario["split"],
            "family": scenario["family"],
            "prompt_char_count": len(prompt),
            "completion_char_count": len(completion),
        },
    }


def eval_row(scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    candidates = eval_candidates(scenario)
    rng = random.Random(f"{seed}:{scenario['scenario_id']}:candidate_order")
    rng.shuffle(candidates)
    text_blob = "\n".join(candidate["text"].lower() for candidate in candidates)
    forbidden = sorted(term for term in FORBIDDEN_CANDIDATE_TEXT if term in text_blob)
    if forbidden:
        raise ValueError(f"{scenario['scenario_id']}: forbidden candidate text terms: {forbidden}")
    gold_text, gold_commitment = eval_correct_candidate(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "family": scenario["family"],
        "prompt": plain_prompt_nl(scenario),
        "candidates": candidates,
        "gold_candidate_id": "correct_grounded_action",
        "gold_commitment_kind": gold_commitment,
        "overcommit_candidate_ids": ["single_landmark_shortcut"],
        "metadata": {
            "split": scenario["split"],
            "candidate_order_seed": seed,
            "gold_text_char_count": len(gold_text),
        },
    }


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "family": row["family"],
        "gold_candidate_id": row["gold_candidate_id"],
        "gold_commitment_kind": row["gold_commitment_kind"],
        "overcommit_candidate_ids": row["overcommit_candidate_ids"],
        "metadata": row["metadata"],
    }


def build_artifacts(seed: int) -> dict[str, list[dict[str, Any]]]:
    train = generate_awft001.build_scenarios("train", seed)
    dev = generate_awft001.build_scenarios("dev", seed)
    test = generate_awft001.build_scenarios("test", seed)
    eval_rows = [eval_row(scenario, seed) for scenario in dev + test]
    return {
        "train_plain_qa_nl": [train_row(scenario, "plain_qa_nl") for scenario in train],
        "train_rich_prose_nl": [train_row(scenario, "rich_prose_nl") for scenario in train],
        "train_anchorweave_nl": [train_row(scenario, "anchorweave_nl") for scenario in train],
        "train_shuffled_anchorweave_nl": [
            train_row(scenario, "shuffled_anchorweave_nl", corrupted=True)
            for scenario in train
        ],
        "eval_candidates": eval_rows,
        "eval_labels": [label_row(row) for row in eval_rows],
    }


def write_artifacts(out_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train_plain_qa_nl": "train_plain_qa_nl.jsonl",
        "train_rich_prose_nl": "train_rich_prose_nl.jsonl",
        "train_anchorweave_nl": "train_anchorweave_nl.jsonl",
        "train_shuffled_anchorweave_nl": "train_shuffled_anchorweave_nl.jsonl",
        "eval_candidates": "eval_candidates.jsonl",
        "eval_labels": "eval_labels.jsonl",
    }
    for key, filename in filenames.items():
        write_jsonl(out_dir / filename, artifacts[key])


def print_summary(out_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    print(f"wrote AWFT-001-FC artifacts to {out_dir}")
    print(
        "counts: "
        f"train={len(artifacts['train_plain_qa_nl'])} "
        f"eval={len(artifacts['eval_candidates'])}"
    )
    for key in VIEWS.values():
        artifact_key = key.removesuffix(".jsonl")
        rows = artifacts[artifact_key]
        avg_prompt = sum(row["metadata"]["prompt_char_count"] for row in rows) / len(rows)
        avg_completion = sum(row["metadata"]["completion_char_count"] for row in rows) / len(rows)
        print(
            f"{artifact_key}: rows={len(rows)} "
            f"avg_prompt_chars={avg_prompt:.1f} "
            f"avg_completion_chars={avg_completion:.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        artifacts = build_artifacts(args.seed)
        write_artifacts(args.out, artifacts)
        print_summary(args.out, artifacts)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
