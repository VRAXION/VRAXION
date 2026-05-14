#!/usr/bin/env python3
"""Run the HGA-DESK-002 / DeskCache S02 target-only probe.

S02 reuses the S01-v2 scoring engine but swaps in an explicit active-use
check-in-pad scenario. It does not train, does not export AnchorWeave-v1.0
cells, and does not write under data/anchorweave/cells.
"""

from __future__ import annotations

from pathlib import Path
import sys

import run_deskcache_s01_probe as probe


Candidate = probe.Candidate


BASE_PROMPT = """You are looking for a thin access badge that your assistant left somewhere on
your desk. You need it to start a meeting check-in soon, and you want to find it
with as little wasted searching as possible.

The desk is cluttered but familiar. Along the back row, there is a badge holder
stand on the left, a monitor in the center, and a pen cup on the right. The pen
cup is crowded with pens, clips, spare lanyards, and small loose office items.

In the middle of the desk is an open laptop. Next to it is a flat check-in pad
with a small green ready light and a label that says: "Place badge here to start
meeting check-in." Nearby there is also a messy intake tray, a closed project
folder, and a coffee cup with a damp napkin beside it.

At the front of the desk, there is a mouse, a notebook, a wallet, and a shallow
desk drawer.

Your assistant left this note:

"Boss,

I did not file the badge away or put it back in the holder. That would make you
look for it again before the check-in step.

I also avoided the messy tray, loose office clutter, coffee area, and your
personal things. You should not have to sort, clean, or open private items
before the meeting.

I left it where the next work step can happen with almost no searching. Think
less like storage, and more like the place where the badge can immediately do
its job.

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
me start the meeting step quickly, without sorting, cleaning, opening private
items, or doing a broad search.

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
the drawer, folder, pen cup, and intake tray.

The fastest plan is not to invent a special interpretation. It is to check the
obvious storage path first, then broaden only after the normal places fail."""


CANDIDATES = [
    Candidate(
        "checkin_pad_first",
        "Check the flat check-in pad first; if it is not there, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "badge_holder_first",
        "Check the badge holder stand first; if it is not there, continue with the remaining desk search.",
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
        "messy_tray_first",
        "Search the messy intake tray first; if it is not there, continue with the remaining desk search.",
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


CANDIDATES_PARAPHRASE_B = [
    Candidate(
        "checkin_pad_first",
        "Inspect the flat check-in pad as the first step; if it is absent, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "badge_holder_first",
        "Inspect the badge holder stand as the first step; if it is absent, continue with the remaining desk search.",
        2,
        8,
        "surface_association",
    ),
    Candidate(
        "closed_folder_first",
        "Inspect the closed project folder as the first step; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "desk_drawer_first",
        "Inspect the shallow desk drawer as the first step; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "pen_cup_first",
        "Inspect the pen cup clutter as the first step; if it is absent, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "messy_tray_first",
        "Inspect the messy intake tray as the first step; if it is absent, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "coffee_area_first",
        "Inspect the coffee cup and damp napkin area as the first step; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "dirty_storage_like",
    ),
    Candidate(
        "wallet_first",
        "Inspect the wallet as the first step; if it is absent, continue with the remaining desk search.",
        3,
        7,
        "personal_boundary",
    ),
]


def configure_s02() -> None:
    probe.SCENARIO_ID = "HGA-DESK-002-S02"
    probe.STATUS_PREFIX = "S02_V1"
    probe.CONTRACT_FILE = Path("docs/research/HGA_DESK_002_S02_CONTRACT.md")
    probe.GOLD_CANDIDATE_ID = "checkin_pad_first"
    probe.BASE_PROMPT = BASE_PROMPT
    probe.CORRECT_INNER_VOICE = CORRECT_INNER_VOICE
    probe.CORRUPTED_INNER_VOICE = CORRUPTED_INNER_VOICE
    probe.INNER_VOICES = {
        "STYLE_CONTROL": probe.STYLE_CONTROL,
        "CORRECT_INNER_VOICE": CORRECT_INNER_VOICE,
        "CORRUPTED_INNER_VOICE": CORRUPTED_INNER_VOICE,
    }
    probe.CANDIDATES = CANDIDATES
    probe.CANDIDATES_PARAPHRASE_B = CANDIDATES_PARAPHRASE_B
    probe.FEEDBACK_PLACES = {
        "checkin_pad_first": "the flat check-in pad",
        "badge_holder_first": "the badge holder stand",
        "closed_folder_first": "the closed project folder",
        "desk_drawer_first": "the shallow desk drawer",
        "pen_cup_first": "the pen cup clutter",
        "messy_tray_first": "the messy intake tray",
        "coffee_area_first": "the coffee cup and damp napkin area",
        "wallet_first": "the wallet",
    }
    probe.FREE_RESPONSE_QUESTION = (
        "What is the best first place or action to check? Explain why in one short sentence."
    )
    probe.FREE_RESPONSE_CATEGORIES = [
        "correct_affordance_reason",
        "surface_association_reason",
        "storage_reason",
        "clutter_reason",
        "boundary_reason",
        "task_frame_drift",
        "post_hoc_confabulation",
        "other",
    ]


def main(argv: list[str] | None = None) -> int:
    configure_s02()
    return probe.main(argv or sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
