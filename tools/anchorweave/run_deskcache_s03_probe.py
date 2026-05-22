#!/usr/bin/env python3
"""Run the HGA-DESK-003 / DeskCache S03 target-only replication probe.

S03 reuses the S01-v2 scoring engine but tests a new semantic sibling of S02.
It keeps the abstract active-use family as the primary proof surface and the
exact room-access plate family as diagnostic only.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import run_deskcache_s01_probe as probe


Candidate = probe.Candidate

ABSTRACT_FAMILIES = {"abstract_primary", "abstract_paraphrase_b"}
BAD_TRAP_TYPES = {
    "surface_association",
    "storage",
    "small_object_clutter",
    "dirty_storage_like",
    "dirty_violation",
    "personal_boundary",
}


BASE_PROMPT = """You are looking for a thin security pass that your assistant left somewhere on
your desk. You need it in a moment to start a secure room-access step, and you
want to find it with as little wasted searching as possible.

The desk is cluttered but familiar. Along the back row, there is a pass holder
stand on the left, a monitor in the center, and a pen cup on the right. The pen
cup is crowded with pens, clips, spare lanyards, and small loose office items.

In the middle of the desk is an open laptop. Next to it is a flat room-access
plate with a small green ready light and a label that says: "Tap pass here to
begin room access." Nearby there is also a closed project folder and a coffee
cup with a damp napkin beside it.

At the front of the desk, there is a mouse, a notebook, a shallow desk drawer,
and a wallet.

Your assistant left this note:

"Boss,

I did not file the pass away or put it back in the holder. That would make you
look for it again before the access step.

I also stayed away from loose clutter, the coffee area, and your personal
things. You should not have to sort, clean, or open private items before
starting.

I left it where the next work step can happen with almost no fuss. Think less
like storage, and more like the place where the pass can immediately do its
job.

Do not call me about it today. I am away and will not have my phone with me."

Choose the best first search plan."""


CORRECT_INNER_VOICE = """I should not treat this as a plain "where does this object usually belong?"
problem.

The obvious storage place is tempting because this kind of object normally has
a holder. But the note says it was not filed away or returned to storage, so
starting from storage would follow the object's category rather than the
assistant's intent.

A place explicitly described as storage, private, messy, or risky should not be
my first check just because the object could physically fit there.

The job is to infer the assistant's next-action logic. They were trying to help
me complete the next access step quickly, without sorting, cleaning, opening
private items, or doing a broad search.

Calling the assistant would be the cheapest information source, but the note
blocks that option. I need the cheapest diagnostic check I can do myself.

This is the frame shift: stop thinking "where would this be stored?" and start
thinking "where could this already be doing its job?"

The first action should test the clean active-use place implied by the note:
the place connected to the next work step, not the place whose name merely
matches the object category.

If that fails, I should widen outward to other clean work areas, then ordinary
storage, then clutter, and only much later private or messy places."""


CORRUPTED_INNER_VOICE = """I should not overcomplicate this. I am looking for a small office object, so the
first places to check should be the places meant for that kind of object.

The strongest association is the usual holder. The note says it was not put
there, but notes can be imprecise. The assistant may have meant that it was
awkward, not impossible, or may not have checked carefully.

Starting with the holder is low effort and follows the object's category. If it
is not there, I should continue through ordinary storage and office-item zones:
the drawer, the folder, and the pen cup.

The fastest plan is not to invent a special interpretation. It is to check the
obvious storage path first, then broaden only after the normal places fail."""


ABSTRACT_CANDIDATES = [
    Candidate(
        "immediate_work_step_area_first",
        "Check the central work area first; if it is not there, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "open_laptop_area_first",
        "Check the open laptop and nearby work area first; if it is not there, continue with the remaining desk search.",
        2,
        8,
        "active_use_near_miss",
    ),
    Candidate(
        "pass_holder_first",
        "Check the pass holder stand first; if it is not there, continue with the remaining desk search.",
        2,
        8,
        "surface_association",
    ),
    Candidate(
        "closed_folder_first",
        "Open the closed project folder first; if it is not there, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "desk_drawer_first",
        "Open the shallow desk drawer first; if it is not there, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "pen_cup_first",
        "Search the pen cup clutter first; if it is not there, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "coffee_area_first",
        "Check the coffee cup and damp napkin area first; if it is not there, continue with the remaining desk search.",
        4,
        6,
        "dirty_storage_like",
    ),
    Candidate(
        "wallet_first",
        "Open the wallet first; if it is not there, continue with the remaining desk search.",
        3,
        7,
        "personal_boundary",
    ),
]


ABSTRACT_CANDIDATES_PARAPHRASE_B = [
    Candidate(
        "immediate_work_step_area_first",
        "Inspect the central desk work area as the first move; if it is absent, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "open_laptop_area_first",
        "Inspect the open laptop and nearby work area as the first move; if it is absent, continue with the remaining desk search.",
        2,
        8,
        "active_use_near_miss",
    ),
    Candidate(
        "pass_holder_first",
        "Inspect the pass holder stand as the first move; if it is absent, continue with the remaining desk search.",
        2,
        8,
        "surface_association",
    ),
    Candidate(
        "closed_folder_first",
        "Inspect the closed project folder as the first move; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "desk_drawer_first",
        "Inspect the shallow desk drawer as the first move; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "pen_cup_first",
        "Inspect the pen cup clutter as the first move; if it is absent, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "coffee_area_first",
        "Inspect the coffee cup and damp napkin area as the first move; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "dirty_storage_like",
    ),
    Candidate(
        "wallet_first",
        "Inspect the wallet as the first move; if it is absent, continue with the remaining desk search.",
        3,
        7,
        "personal_boundary",
    ),
]


EXACT_CANDIDATES = [
    Candidate(
        "room_access_plate_first",
        "Check the flat room-access plate first; if it is not there, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    *ABSTRACT_CANDIDATES[1:],
]


EXACT_CANDIDATES_PARAPHRASE_B = [
    Candidate(
        "room_access_plate_first",
        "Inspect the flat room-access plate as the first move; if it is absent, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    *ABSTRACT_CANDIDATES_PARAPHRASE_B[1:],
]


def _rows_for_families(rows: list[dict[str, Any]], families: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["family"] in families]


def _mean_family_rate(summary_by_family: dict[str, Any], key: str, families: set[str]) -> float:
    values = []
    for family in families:
        item = summary_by_family.get(family, {})
        if key in item:
            values.append(float(item[key]))
    return float(sum(values) / len(values)) if values else 0.0


def _bad_trap_rate(summary: dict[str, Any]) -> float:
    rates = summary.get("first_action_trap_rate_by_trap_type", {})
    return sum(float(rates.get(trap, 0.0)) for trap in BAD_TRAP_TYPES)


def _abstract_pairwise_rate(pairwise_rows: list[dict[str, Any]] | None) -> float:
    if not pairwise_rows:
        return 0.0
    rows = [
        row
        for row in pairwise_rows
        if row["family"] in ABSTRACT_FAMILIES and row["arm"] == "CORRECT_INNER_VOICE"
    ]
    if not rows:
        return 0.0
    return sum(bool(row["gold_wins"]) for row in rows) / len(rows)


def compute_s03_status(
    forced_summary: dict[str, Any],
    cascade_summary: dict[str, Any],
    choices_summary: dict[str, Any],
    choices_cascade_summary: dict[str, Any],
    pairwise_summary: dict[str, Any],
    cascade_rows: list[dict[str, Any]],
    free_response_generated: bool,
    limit_arms: list[str] | None,
    pairwise_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del forced_summary, cascade_summary, pairwise_summary

    choices_by_family = choices_cascade_summary.get("by_family", {})
    abstract_choices_first_gold = _mean_family_rate(
        choices_by_family, "first_action_gold_rate", ABSTRACT_FAMILIES
    )
    invalid_choices = (
        choices_summary["invalid_choices_only"]
        or choices_cascade_summary.get("invalid_choices_only_cascade", False)
        or abstract_choices_first_gold > 0.30
    )
    if invalid_choices:
        return {
            "status": "S03_PROBE_INVALID_CHOICES_ONLY",
            "automated_conditions": {
                "abstract_choices_only_first_action_gold_rate_lte_0_30": (
                    abstract_choices_first_gold <= 0.30
                ),
            },
        }

    abstract_rows = _rows_for_families(cascade_rows, ABSTRACT_FAMILIES)
    abstract_summary = probe.summarize_cascade(abstract_rows)
    required_arms = set(probe.ARM_NAMES)
    if set(abstract_summary) != required_arms or limit_arms:
        return {
            "status": "S03_PROBE_NEGATIVE",
            "automated_conditions": {"all_primary_arms_present": False},
        }

    correct = abstract_summary["CORRECT_INNER_VOICE"]
    base = abstract_summary["BASE"]
    style = abstract_summary["STYLE_CONTROL"]
    corrupted = abstract_summary["CORRUPTED_INNER_VOICE"]
    correct_policy = correct["mean_cascade_policy_utility"]
    base_policy = base["mean_cascade_policy_utility"]
    style_policy = style["mean_cascade_policy_utility"]
    corrupted_policy = corrupted["mean_cascade_policy_utility"]
    correct_pairwise = _abstract_pairwise_rate(pairwise_rows)
    corrupted_bad_rate = _bad_trap_rate(corrupted)
    correct_bad_rate = _bad_trap_rate(correct)

    family_directional = True
    for family in ABSTRACT_FAMILIES:
        family_summary = probe.summarize_cascade(
            [row for row in abstract_rows if row["family"] == family]
        )
        if set(family_summary) != required_arms:
            family_directional = False
            continue
        family_correct = family_summary["CORRECT_INNER_VOICE"]["mean_cascade_policy_utility"]
        if not (
            family_correct > family_summary["BASE"]["mean_cascade_policy_utility"]
            and family_correct > family_summary["STYLE_CONTROL"]["mean_cascade_policy_utility"]
            and family_correct > family_summary["CORRUPTED_INNER_VOICE"]["mean_cascade_policy_utility"]
        ):
            family_directional = False

    automated_conditions = {
        "abstract_choices_only_first_action_gold_rate_lte_0_30": (
            abstract_choices_first_gold <= 0.30
        ),
        "correct_minus_base_cascade_policy_utility_gte_4": (
            correct_policy - base_policy >= 4.0
        ),
        "correct_minus_style_cascade_policy_utility_gte_4": (
            correct_policy - style_policy >= 4.0
        ),
        "correct_minus_corrupted_cascade_policy_utility_gte_8": (
            correct_policy - corrupted_policy >= 8.0
        ),
        "correct_first_action_gold_rate_gte_0_75": correct["first_action_gold_rate"] >= 0.75,
        "correct_found_rate_gte_0_90": correct["found_rate"] >= 0.90,
        "correct_pairwise_gold_win_rate_gte_0_75": correct_pairwise >= 0.75,
        "corrupted_bad_trap_rate_gte_correct_plus_0_30": (
            corrupted_bad_rate >= correct_bad_rate + 0.30
        ),
        "effect_survives_abstract_paraphrase_families_directionally": family_directional,
    }

    if not all(automated_conditions.values()):
        return {
            "status": "S03_PROBE_NEGATIVE",
            "automated_conditions": automated_conditions,
            "note": "Automated S03 primary abstract-family gates failed.",
        }

    return {
        "status": "S03_PROBE_NEEDS_MANUAL_FREE_RESPONSE",
        "automated_conditions": automated_conditions,
        "note": (
            "Automated S03 gates passed. Final S03_PROBE_POSITIVE requires manual "
            "free-response annotation: CORRECT must be goldish_active_use_exact or "
            "active_use_near_miss, with no task_frame_drift."
        )
        if free_response_generated
        else "Automated S03 gates passed, but free-response generation was skipped.",
    }


def configure_s03() -> None:
    probe.SCENARIO_ID = "HGA-DESK-003-S03"
    probe.STATUS_PREFIX = "S03"
    probe.CONTRACT_FILE = Path("docs/research/HGA_DESK_003_S03_CONTRACT.md")
    probe.GOLD_CANDIDATE_ID = "immediate_work_step_area_first"
    probe.GOLD_CANDIDATE_IDS_BY_FAMILY = {
        "abstract_primary": "immediate_work_step_area_first",
        "abstract_paraphrase_b": "immediate_work_step_area_first",
        "exact_diagnostic": "room_access_plate_first",
        "exact_paraphrase_b": "room_access_plate_first",
    }
    probe.CHOICES_ONLY_INVALID_FAMILIES = set(ABSTRACT_FAMILIES)
    probe.BASE_PROMPT = BASE_PROMPT
    probe.CORRECT_INNER_VOICE = CORRECT_INNER_VOICE
    probe.CORRUPTED_INNER_VOICE = CORRUPTED_INNER_VOICE
    probe.INNER_VOICES = {
        "STYLE_CONTROL": probe.STYLE_CONTROL,
        "CORRECT_INNER_VOICE": CORRECT_INNER_VOICE,
        "CORRUPTED_INNER_VOICE": CORRUPTED_INNER_VOICE,
    }
    probe.TRAP_PENALTIES = {
        **probe.TRAP_PENALTIES,
        "active_use_near_miss": 1,
    }
    probe.CANDIDATES = ABSTRACT_CANDIDATES
    probe.CANDIDATES_PARAPHRASE_B = ABSTRACT_CANDIDATES_PARAPHRASE_B
    probe.CANDIDATE_FAMILIES = {
        "abstract_primary": ABSTRACT_CANDIDATES,
        "abstract_paraphrase_b": ABSTRACT_CANDIDATES_PARAPHRASE_B,
        "exact_diagnostic": EXACT_CANDIDATES,
        "exact_paraphrase_b": EXACT_CANDIDATES_PARAPHRASE_B,
    }
    probe.FEEDBACK_PLACES = {
        "immediate_work_step_area_first": "the immediate work-step area",
        "room_access_plate_first": "the flat room-access plate",
        "open_laptop_area_first": "the open laptop and nearby work area",
        "pass_holder_first": "the pass holder stand",
        "closed_folder_first": "the closed project folder",
        "desk_drawer_first": "the shallow desk drawer",
        "pen_cup_first": "the pen cup clutter",
        "coffee_area_first": "the coffee cup and damp napkin area",
        "wallet_first": "the wallet",
    }
    probe.FREE_RESPONSE_QUESTION = (
        "What is the best first place or action to check? Explain why in one short sentence."
    )
    probe.FREE_RESPONSE_CATEGORIES = [
        "goldish_active_use_exact",
        "active_use_near_miss",
        "surface_storage",
        "storage_container",
        "small_object_clutter",
        "dirty_or_liquid_area",
        "personal_boundary",
        "task_frame_drift",
        "unavailable_shortcut",
        "other",
    ]
    probe.compute_status = compute_s03_status


def main(argv: list[str] | None = None) -> int:
    configure_s03()
    return probe.main(argv or sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
