#!/usr/bin/env python3
"""Generate AWFT-001 synthetic A/B test artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any


FAMILIES = [
    "exact_match",
    "near_miss",
    "viewpoint_shift",
    "misleading_single_landmark",
]

COUNTERFACTUAL_LABELS = [
    "strengthens_match",
    "weakens_match",
    "neutral",
    "requires_disambiguation",
    "rejects_match",
    "confirms_match",
]

COMMITMENT_LEVELS = [
    "confirmed_same_place",
    "rejected_same_place",
    "defer_and_disambiguate",
    "premature_commit",
]

ACTIONS = [
    {
        "action_id": "run_low_cost_disambiguation",
        "description": "Check street name, route geometry, and one additional landmark before choosing the route.",
    },
    {
        "action_id": "proceed_to_remembered_turn",
        "description": "Take the remembered turn now.",
    },
    {
        "action_id": "reject_location_and_backtrack",
        "description": "Treat the location as different and backtrack immediately.",
    },
    {
        "action_id": "pause_and_reconstruct_route",
        "description": "Pause and reconstruct the route before moving.",
    },
]

TRAIN_SURFACES = {
    "landmarks": [
        "yellow awning bakery",
        "blue bus sign",
        "red mailbox",
        "narrow alley entrance",
        "corner flower stand",
        "painted loading door",
    ],
    "secondary": [
        "cracked tile step",
        "brass street plaque",
        "small delivery ramp",
        "bent bicycle rail",
        "black drain grate",
        "white window bars",
    ],
    "streets": [
        "Mason Lane",
        "Harbor Street",
        "Willow Row",
        "North Arcade",
        "Market Bend",
        "Foundry Walk",
    ],
    "details": [
        "rain on the curb",
        "warm bread smell",
        "a passing tram bell",
        "wet newspaper stack",
        "faded chalk marks",
        "a delivery cart nearby",
    ],
}

TEST_SURFACES = {
    "landmarks": [
        "green awning pharmacy",
        "cracked stone fountain",
        "iron stairwell",
        "clock kiosk",
        "glass repair booth",
        "silver street cabinet",
    ],
    "secondary": [
        "mosaic doorway",
        "orange utility hatch",
        "arched side passage",
        "triangular warning sign",
        "rusted water pipe",
        "blue ceramic marker",
    ],
    "streets": [
        "Cedar Court",
        "Lantern Road",
        "Elm Passage",
        "Station Cut",
        "Bridge Close",
        "Anchor Yard",
    ],
    "details": [
        "fog near the curb",
        "coffee smell",
        "a bicycle bell",
        "fallen leaf pile",
        "fresh poster glue",
        "a parked service van",
    ],
}

DEV_SURFACES = TRAIN_SURFACES


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AWFT-001 synthetic artifacts.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/anchorweave/awft001"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic generation seed.")
    return parser.parse_args(argv)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_dumps(row))
            handle.write("\n")


def pick(pool: list[str], index: int, offset: int = 0) -> str:
    return pool[(index + offset) % len(pool)]


def action_ids() -> list[str]:
    return [action["action_id"] for action in ACTIONS]


def scenario_counts(split: str) -> dict[str, int]:
    if split in {"train", "test"}:
        return {family: 10 for family in FAMILIES}
    return {
        "exact_match": 3,
        "near_miss": 3,
        "viewpoint_shift": 2,
        "misleading_single_landmark": 2,
    }


def family_truth_and_target(
    family: str,
    landmark: str,
    secondary: str,
    street: str,
    detail: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if family == "exact_match":
        hidden_truth = {
            "same_general_area": True,
            "same_exact_corner": True,
            "landmark_replaced": False,
            "missing_landmark_explained_by_viewpoint": False,
            "correct_action": "proceed_to_remembered_turn",
            "wrong_shortcut": "reject_based_on_incidental_detail",
        }
        target = {
            "first_action": "proceed_to_remembered_turn",
            "high_salience": [
                "landmark_constellation",
                "route_geometry",
                "street_name_match",
            ],
            "low_salience": [
                "weather_or_mood",
                "single_vivid_detail",
                "passing_object",
            ],
            "symbol_attach": [
                "landmark_constellation_anchor",
                "spatial_memory_disambiguation",
            ],
            "symbol_reject": [
                "rain_means_turn_right",
                "single_landmark_means_same_place",
                "familiar_feeling_means_same_place",
            ],
            "counterfactual_answers": {
                "cf_missing_secondary_from_viewpoint": "requires_disambiguation",
                "cf_single_landmark_overlap": "requires_disambiguation",
                "cf_street_name_matches": "strengthens_match",
            },
            "commitment_level": "confirmed_same_place",
            "claim_boundary": (
                "Confirm the place only because multiple route-defining cues align; "
                "do not generalize from one landmark or vivid detail."
            ),
        }
    elif family == "near_miss":
        hidden_truth = {
            "same_general_area": True,
            "same_exact_corner": False,
            "landmark_replaced": True,
            "missing_landmark_explained_by_viewpoint": False,
            "correct_action": "run_low_cost_disambiguation",
            "wrong_shortcut": "turn_based_on_single_landmark",
        }
        target = {
            "first_action": "run_low_cost_disambiguation",
            "high_salience": [
                "landmark_constellation_mismatch",
                "route_geometry_conflict",
                "street_name_uncertain",
            ],
            "low_salience": [
                "weather_or_mood",
                "single_familiar_object",
                "vivid_color_memory",
            ],
            "symbol_attach": [
                "near_miss_anchor",
                "spatial_memory_disambiguation",
            ],
            "symbol_reject": [
                "single_landmark_means_same_place",
                "familiar_feeling_means_same_place",
                "memory_hook_is_causal_rule",
            ],
            "counterfactual_answers": {
                "cf_missing_secondary_from_viewpoint": "weakens_match",
                "cf_single_landmark_overlap": "requires_disambiguation",
                "cf_street_name_matches": "strengthens_match",
            },
            "commitment_level": "defer_and_disambiguate",
            "claim_boundary": (
                "Do not commit to the remembered route from a partial landmark overlap; "
                "verify route geometry first."
            ),
        }
    elif family == "viewpoint_shift":
        hidden_truth = {
            "same_general_area": True,
            "same_exact_corner": True,
            "landmark_replaced": False,
            "missing_landmark_explained_by_viewpoint": True,
            "correct_action": "run_low_cost_disambiguation",
            "wrong_shortcut": "reject_location_from_missing_landmark",
        }
        target = {
            "first_action": "run_low_cost_disambiguation",
            "high_salience": [
                "viewpoint_shift_explains_missing_cue",
                "route_geometry_partial_match",
                "diagnostic_landmark_needed",
            ],
            "low_salience": [
                "missing_sign_means_different_place",
                "weather_or_mood",
                "vivid_memory_fragment",
            ],
            "symbol_attach": [
                "viewpoint_sensitive_memory_anchor",
                "spatial_memory_disambiguation",
            ],
            "symbol_reject": [
                "missing_landmark_means_wrong_place",
                "single_landmark_means_same_place",
                "familiar_feeling_means_same_place",
            ],
            "counterfactual_answers": {
                "cf_missing_secondary_from_viewpoint": "requires_disambiguation",
                "cf_single_landmark_overlap": "requires_disambiguation",
                "cf_street_name_matches": "strengthens_match",
            },
            "commitment_level": "defer_and_disambiguate",
            "claim_boundary": (
                "A missing cue under a changed viewpoint should trigger disambiguation, "
                "not automatic rejection or confirmation."
            ),
        }
    elif family == "misleading_single_landmark":
        hidden_truth = {
            "same_general_area": True,
            "same_exact_corner": False,
            "landmark_replaced": False,
            "missing_landmark_explained_by_viewpoint": False,
            "correct_action": "run_low_cost_disambiguation",
            "wrong_shortcut": "turn_based_on_single_landmark",
        }
        target = {
            "first_action": "run_low_cost_disambiguation",
            "high_salience": [
                "single_landmark_overlap_insufficient",
                "landmark_constellation_absent",
                "route_geometry_conflict",
            ],
            "low_salience": [
                "single_vivid_detail",
                "weather_or_mood",
                "familiar_feeling",
            ],
            "symbol_attach": [
                "shortcut_risk_anchor",
                "spatial_memory_disambiguation",
            ],
            "symbol_reject": [
                "single_landmark_means_same_place",
                "rain_means_turn_right",
                "familiar_feeling_means_same_place",
            ],
            "counterfactual_answers": {
                "cf_missing_secondary_from_viewpoint": "weakens_match",
                "cf_single_landmark_overlap": "requires_disambiguation",
                "cf_street_name_matches": "strengthens_match",
            },
            "commitment_level": "defer_and_disambiguate",
            "claim_boundary": (
                "A single familiar landmark is a retrieval cue, not sufficient evidence "
                "for the route action."
            ),
        }
    else:
        raise ValueError(f"unknown family: {family}")

    hidden_truth["diagnostic_landmark"] = secondary
    hidden_truth["landmark_under_test"] = landmark
    hidden_truth["street_under_test"] = street
    hidden_truth["incidental_detail"] = detail
    return hidden_truth, target


def build_observation(
    family: str,
    landmark: str,
    secondary: str,
    street: str,
    detail: str,
    index: int,
) -> tuple[str, str, list[str]]:
    source_memory = (
        f"The remembered route had a {landmark}, then {secondary}, then the turn "
        f"onto {street}. The memory hook includes {detail}, but that detail was "
        "not the reason the route worked."
    )

    if family == "exact_match":
        current_scene = (
            f"The current corner shows the {landmark}, the {secondary}, and a clear "
            f"{street} marker. The route geometry matches the remembered turn."
        )
        beads = [
            f"{landmark} present",
            f"{secondary} present",
            f"{street} marker present",
            "route geometry matches",
            f"{detail} present as incidental context",
        ]
    elif family == "near_miss":
        alternate = f"replacement facade {index % 7}"
        current_scene = (
            f"The current corner has something like the {landmark}, but the {secondary} "
            f"is replaced by a {alternate}; the {street} marker is unclear and the "
            "turn angle feels shifted."
        )
        beads = [
            f"{landmark} partly resembles memory",
            f"{secondary} not found",
            f"{street} marker unclear",
            "route geometry conflicts",
            f"{detail} present as incidental context",
        ]
    elif family == "viewpoint_shift":
        current_scene = (
            f"The current view is from the opposite side of the corner. The {landmark} "
            f"is visible, the {secondary} may be hidden by viewpoint, and the {street} "
            "marker is partly visible."
        )
        beads = [
            f"{landmark} present",
            f"{secondary} missing from current viewpoint",
            f"{street} marker partly visible",
            "route geometry partially matches",
            f"{detail} present as incidental context",
        ]
    else:
        current_scene = (
            f"The current corner has the {landmark}, but the {secondary} and {street} "
            "marker are absent. Another turn nearby creates a tempting but shallow match."
        )
        beads = [
            f"{landmark} present",
            f"{secondary} absent",
            f"{street} marker absent",
            "route geometry conflicts",
            f"{detail} present as incidental context",
        ]

    return source_memory, current_scene, beads


def counterfactual_questions(secondary: str, street: str) -> dict[str, dict[str, Any]]:
    return {
        "cf_missing_secondary_from_viewpoint": {
            "question": (
                f"If the {secondary} is missing but the viewpoint changed, what should "
                "that do to the match judgment?"
            ),
            "allowed_labels": COUNTERFACTUAL_LABELS,
        },
        "cf_single_landmark_overlap": {
            "question": (
                "If only one familiar landmark overlaps and the route geometry is uncertain, "
                "what should the agent do?"
            ),
            "allowed_labels": COUNTERFACTUAL_LABELS,
        },
        "cf_street_name_matches": {
            "question": (
                f"If the {street} marker is confirmed along with the landmark constellation, "
                "how does that affect the match?"
            ),
            "allowed_labels": COUNTERFACTUAL_LABELS,
        },
    }


def build_scenario(
    split: str,
    family: str,
    family_index: int,
    global_index: int,
    surfaces: dict[str, list[str]],
) -> dict[str, Any]:
    landmark = pick(surfaces["landmarks"], global_index, 0)
    secondary = pick(surfaces["secondary"], global_index, 2)
    street = pick(surfaces["streets"], global_index, 4)
    detail = pick(surfaces["details"], global_index, 1)
    source_memory, current_scene, beads = build_observation(
        family, landmark, secondary, street, detail, global_index
    )
    hidden_truth, target = family_truth_and_target(family, landmark, secondary, street, detail)

    scenario_id = f"awft001_{split}_{family}_{family_index:02d}"
    return {
        "scenario_id": scenario_id,
        "split": split,
        "family": family,
        "hidden_truth": hidden_truth,
        "source_memory": source_memory,
        "current_scene": current_scene,
        "perceptual_beads": beads,
        "available_actions": ACTIONS,
        "correct_first_action": target["first_action"],
        "salience_labels": {
            "high": target["high_salience"],
            "low": target["low_salience"],
        },
        "symbol_attach_labels": target["symbol_attach"],
        "symbol_reject_labels": target["symbol_reject"],
        "counterfactual_questions": counterfactual_questions(secondary, street),
        "expected_claim_boundary": target["claim_boundary"],
        "commitment_level": target["commitment_level"],
        "target": target,
    }


def build_scenarios(split: str, seed: int) -> list[dict[str, Any]]:
    surfaces = TEST_SURFACES if split == "test" else DEV_SURFACES if split == "dev" else TRAIN_SURFACES
    rng = random.Random(seed + {"train": 11, "dev": 23, "test": 37}[split])
    scenarios: list[dict[str, Any]] = []
    global_index = 0
    for family, count in scenario_counts(split).items():
        for family_index in range(count):
            scenarios.append(
                build_scenario(split, family, family_index, global_index, surfaces)
            )
            global_index += 1
    rng.shuffle(scenarios)
    return scenarios


def output_request() -> str:
    return (
        "Return only JSON with these fields: first_action, high_salience, "
        "low_salience, symbol_attach, symbol_reject, counterfactual_answers, "
        "commitment_level, claim_boundary. Use counterfactual enum labels and one "
        "commitment_level from the allowed set."
    )


def plain_prompt(scenario: dict[str, Any]) -> str:
    cf_lines = "\n".join(
        f"- {cf_id}: {cf['question']} Allowed: {', '.join(cf['allowed_labels'])}"
        for cf_id, cf in scenario["counterfactual_questions"].items()
    )
    actions = "\n".join(
        f"- {action['action_id']}: {action['description']}"
        for action in scenario["available_actions"]
    )
    beads = "\n".join(f"- {bead}" for bead in scenario["perceptual_beads"])
    return (
        "You are deciding what to do at a route-memory corner.\n\n"
        f"Source memory:\n{scenario['source_memory']}\n\n"
        f"Current observation:\n{scenario['current_scene']}\n\n"
        f"Perceptual beads:\n{beads}\n\n"
        f"Available actions:\n{actions}\n\n"
        f"Counterfactual questions:\n{cf_lines}\n\n"
        f"Allowed commitment levels: {', '.join(COMMITMENT_LEVELS)}.\n"
        f"{output_request()}"
    )


def rich_prompt(scenario: dict[str, Any]) -> str:
    beads = "; ".join(scenario["perceptual_beads"])
    actions = "; ".join(
        f"{action['action_id']} means {action['description']}"
        for action in scenario["available_actions"]
    )
    cf_lines = " ".join(
        f"{cf_id}: {cf['question']} Choose one of {', '.join(cf['allowed_labels'])}."
        for cf_id, cf in scenario["counterfactual_questions"].items()
    )
    return (
        "The memory arrives as a small route scene rather than a rule. "
        f"{scenario['source_memory']} Now the agent stands at the observed corner: "
        f"{scenario['current_scene']} The perceptual beads are: {beads}. "
        "Some details are vivid, but vividness alone should not decide the route. "
        f"The available actions are: {actions}. "
        f"Counterfactual checks: {cf_lines} "
        f"Allowed commitment levels: {', '.join(COMMITMENT_LEVELS)}. "
        f"{output_request()}"
    )


def anchorweave_prompt(scenario: dict[str, Any]) -> str:
    graph_edges = [
        {"source": "source_memory", "relation": "contains", "target": "landmark_constellation"},
        {"source": "current_scene", "relation": "offers", "target": "perceptual_beads"},
        {"source": "perceptual_beads", "relation": "inform", "target": "first_action"},
        {"source": "counterfactuals", "relation": "test", "target": "symbol_binding"},
    ]
    prompt_payload = {
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
                "first_action",
                "symbol_binding",
            ],
            "candidate_edges": graph_edges,
        },
        "salience_candidates": scenario["perceptual_beads"],
        "available_actions": scenario["available_actions"],
        "counterfactual_questions": scenario["counterfactual_questions"],
        "allowed_commitment_levels": COMMITMENT_LEVELS,
        "task": output_request(),
    }
    return json.dumps(prompt_payload, ensure_ascii=True, indent=2, sort_keys=True)


def completion_target_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    target = dict(scenario["target"])
    return {
        "first_action": target["first_action"],
        "high_salience": list(target["high_salience"]),
        "low_salience": list(target["low_salience"]),
        "symbol_attach": list(target["symbol_attach"]),
        "symbol_reject": list(target["symbol_reject"]),
        "counterfactual_answers": dict(target["counterfactual_answers"]),
        "commitment_level": target["commitment_level"],
        "claim_boundary": target["claim_boundary"],
    }


def label_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    target = completion_target_from_scenario(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        **target,
    }


def shuffled_target(scenario: dict[str, Any]) -> dict[str, Any]:
    gold = label_from_scenario(scenario)
    if gold["commitment_level"] == "confirmed_same_place":
        first_action = "reject_location_and_backtrack"
        commitment = "rejected_same_place"
    else:
        first_action = "proceed_to_remembered_turn"
        commitment = "premature_commit"

    answers = {}
    for cf_id, answer in gold["counterfactual_answers"].items():
        if answer == "confirms_match":
            answers[cf_id] = "rejects_match"
        elif answer == "rejects_match":
            answers[cf_id] = "confirms_match"
        elif answer == "requires_disambiguation":
            answers[cf_id] = "confirms_match"
        elif answer == "strengthens_match":
            answers[cf_id] = "neutral"
        else:
            answers[cf_id] = "strengthens_match"

    return {
        "first_action": first_action,
        "high_salience": list(gold["low_salience"]),
        "low_salience": list(gold["high_salience"]),
        "symbol_attach": [
            "single_landmark_means_same_place",
            "rain_means_turn_right",
            "familiar_feeling_means_same_place",
        ],
        "symbol_reject": list(gold["symbol_attach"]),
        "counterfactual_answers": answers,
        "commitment_level": commitment,
        "claim_boundary": "Treat the most vivid cue as sufficient for the route decision.",
    }


def train_row(
    scenario: dict[str, Any],
    view: str,
    prompt: str,
    completion: dict[str, Any],
) -> dict[str, Any]:
    completion_chars = len(json_dumps(completion))
    return {
        "scenario_id": scenario["scenario_id"],
        "view": view,
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "split": scenario["split"],
            "family": scenario["family"],
            "prompt_char_count": len(prompt),
            "completion_char_count": completion_chars,
        },
    }


def eval_prompt_row(scenario: dict[str, Any]) -> dict[str, Any]:
    prompt = plain_prompt(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "prompt": prompt,
        "metadata": {
            "split": scenario["split"],
            "family": scenario["family"],
            "prompt_char_count": len(prompt),
        },
    }


def assert_no_prompt_leakage(rows: list[dict[str, Any]], label: str) -> None:
    forbidden = [
        "hidden_truth",
        "correct_first_action",
        "salience_labels",
        "symbol_attach_labels",
        "symbol_reject_labels",
        "expected_claim_boundary",
    ]
    for row in rows:
        prompt = row.get("prompt", "")
        lowered = prompt.lower()
        for term in forbidden:
            if term.lower() in lowered:
                raise RuntimeError(
                    f"prompt leakage in {label} for {row.get('scenario_id')}: {term}"
                )


def assert_blind_shuffled(rows: list[dict[str, Any]]) -> None:
    forbidden = ["corrupted", "wrong", "control"]
    for row in rows:
        joined = row.get("prompt", "") + "\n" + json_dumps(row.get("completion", {}))
        lowered = joined.lower()
        for term in forbidden:
            if term in lowered:
                raise RuntimeError(
                    f"shuffled prompt/completion discloses control status for {row.get('scenario_id')}: {term}"
                )


def print_view_stats(view_rows: dict[str, list[dict[str, Any]]]) -> None:
    print("AWFT-001 prompt/completion character count summary")
    for view, rows in sorted(view_rows.items()):
        prompt_avg = sum(row["metadata"]["prompt_char_count"] for row in rows) / len(rows)
        completion_avg = sum(row["metadata"]["completion_char_count"] for row in rows) / len(rows)
        print(
            f"{view}: rows={len(rows)} avg_prompt_chars={prompt_avg:.1f} "
            f"avg_completion_chars={completion_avg:.1f}"
        )


def generate(seed: int) -> dict[str, list[dict[str, Any]]]:
    train = build_scenarios("train", seed)
    dev = build_scenarios("dev", seed)
    test = build_scenarios("test", seed)

    labels = [label_from_scenario(scenario) for scenario in dev + test]
    eval_prompts = [eval_prompt_row(scenario) for scenario in dev + test]

    plain = [
        train_row(scenario, "plain_qa", plain_prompt(scenario), completion_target_from_scenario(scenario))
        for scenario in train
    ]
    rich = [
        train_row(scenario, "rich_prose", rich_prompt(scenario), completion_target_from_scenario(scenario))
        for scenario in train
    ]
    anchor = [
        train_row(scenario, "anchorweave_sft", anchorweave_prompt(scenario), completion_target_from_scenario(scenario))
        for scenario in train
    ]
    shuffled = [
        train_row(
            scenario,
            "shuffled_anchorweave_control",
            anchorweave_prompt(scenario),
            shuffled_target(scenario),
        )
        for scenario in train
    ]

    assert_no_prompt_leakage(plain, "plain_qa")
    assert_no_prompt_leakage(rich, "rich_prose")
    assert_no_prompt_leakage(anchor, "anchorweave_sft")
    assert_no_prompt_leakage(shuffled, "shuffled_anchorweave_control")
    assert_no_prompt_leakage(eval_prompts, "eval_prompts")
    assert_blind_shuffled(shuffled)

    return {
        "scenarios_train": train,
        "scenarios_dev": dev,
        "scenarios_test": test,
        "train_plain_qa": plain,
        "train_rich_prose": rich,
        "train_anchorweave_sft": anchor,
        "train_shuffled_anchorweave": shuffled,
        "eval_prompts": eval_prompts,
        "eval_labels": labels,
    }


def validate_counts(artifacts: dict[str, list[dict[str, Any]]]) -> None:
    expected = {
        "scenarios_train": 40,
        "scenarios_dev": 10,
        "scenarios_test": 40,
        "train_plain_qa": 40,
        "train_rich_prose": 40,
        "train_anchorweave_sft": 40,
        "train_shuffled_anchorweave": 40,
        "eval_prompts": 50,
        "eval_labels": 50,
    }
    for name, count in expected.items():
        actual = len(artifacts[name])
        if actual != count:
            raise RuntimeError(f"{name} expected {count} rows, found {actual}")

    for split_name in ["scenarios_train", "scenarios_test"]:
        counts: dict[str, int] = defaultdict(int)
        for scenario in artifacts[split_name]:
            counts[scenario["family"]] += 1
        for family in FAMILIES:
            if counts[family] != 10:
                raise RuntimeError(f"{split_name} {family} expected 10, found {counts[family]}")


def write_artifacts(out_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    paths = {
        "scenarios_train": "scenarios_train.jsonl",
        "scenarios_dev": "scenarios_dev.jsonl",
        "scenarios_test": "scenarios_test.jsonl",
        "train_plain_qa": "train_plain_qa.jsonl",
        "train_rich_prose": "train_rich_prose.jsonl",
        "train_anchorweave_sft": "train_anchorweave_sft.jsonl",
        "train_shuffled_anchorweave": "train_shuffled_anchorweave.jsonl",
        "eval_prompts": "eval_prompts.jsonl",
        "eval_labels": "eval_labels.jsonl",
    }
    for key, filename in paths.items():
        write_jsonl(out_dir / filename, artifacts[key])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    artifacts = generate(args.seed)
    validate_counts(artifacts)
    write_artifacts(args.out, artifacts)
    print(f"wrote AWFT-001 artifacts to {args.out}")
    print("split counts: train=40 dev=10 test=40 eval=50")
    print_view_stats(
        {
            "plain_qa": artifacts["train_plain_qa"],
            "rich_prose": artifacts["train_rich_prose"],
            "anchorweave_sft": artifacts["train_anchorweave_sft"],
            "shuffled_anchorweave_control": artifacts["train_shuffled_anchorweave"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
