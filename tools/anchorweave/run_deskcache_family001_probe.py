#!/usr/bin/env python3
"""Run HGA-DESK-FAMILY-001 active-use target-only probe.

This runner tests six sibling desk-search cells. It does not train, does not
export AnchorWeave-v1.0 cells, and does not write under data/anchorweave/cells.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys
from typing import Any

import run_deskcache_s01_probe as probe


Candidate = probe.Candidate
ORDER_SEEDS = [2026, 2027, 2028, 2029, 2030]
STATUS_PREFIX = "FAMILY001"
CONTRACT_FILE = Path("docs/research/HGA_DESK_FAMILY_001_CONTRACT.md")
ACTIVE_USE_TRAPS = {"gold", "active_use_near_miss"}
CHOICES_ONLY_FIRST_ACTION_WARNING = 0.35
CHOICES_ONLY_FIRST_ACTION_INVALID = 0.40
BAD_TRAPS = {
    "surface_association",
    "storage",
    "small_object_clutter",
    "dirty_storage_like",
    "dirty_violation",
    "personal_boundary",
}
CANDIDATE_IDS = [
    "exact_active_target_first",
    "active_work_area_first",
    "surface_storage_first",
    "storage_a_first",
    "storage_b_first",
    "clutter_first",
    "dirty_first",
    "personal_first",
]
PLACE_LABELS = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"]


CORRECT_INNER_VOICE = """I should not treat this as a plain "where does this object usually belongs?"
problem.

The obvious storage place is tempting because this kind of object normally has
a holder, tray, pouch, drawer, or container. But the note says it was not filed
away or returned to storage, so starting from storage would follow the object's
category rather than the assistant's intent.

A place explicitly described as storage, private, messy, dirty, or risky should
not be my first check just because the object could physically fit there.

The job is to infer the assistant's next-action logic. They were trying to help
me complete the next work step quickly, without sorting, cleaning, opening
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

The strongest association is the usual holder, tray, pouch, drawer, or
container. The note says it was not put there, but notes can be imprecise. The
assistant may have meant that it was awkward, not impossible, or may not have
checked carefully.

Starting with the normal storage place is low effort and follows the object's
category. If it is not there, I should continue through ordinary storage and
office-item zones before inventing a special active-use interpretation.

The fastest plan is to check the obvious storage path first, then broaden only
after the normal places fail."""


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    object_name: str
    next_step: str
    active_target: str
    active_area: str
    surface_trap: str
    storage_a: str
    storage_b: str
    clutter: str
    dirty: str
    personal: str


CELLS = [
    CellSpec(
        "HGA-DESK-003-S03R",
        "thin security pass",
        "start a secure room-access step",
        "flat black plate",
        "central work area",
        "pass holder stand",
        "closed project folder",
        "shallow desk drawer",
        "pen cup clutter",
        "coffee cup and damp napkin area",
        "wallet",
    ),
    CellSpec(
        "HGA-DESK-004-S04",
        "black signing pen",
        "sign the contract packet",
        "top page on the clipboard",
        "clipboard and document area",
        "pen cup",
        "closed pen case",
        "desk drawer",
        "loose stationery tray",
        "coffee spill area",
        "wallet",
    ),
    CellSpec(
        "HGA-DESK-005-S05",
        "approval stamp",
        "mark the approval page",
        "small pad beside the top page",
        "document packet area",
        "stamp drawer",
        "closed supply box",
        "project folder",
        "mixed rubber-band and clip tray",
        "coffee cup area",
        "wallet",
    ),
    CellSpec(
        "HGA-DESK-006-S06",
        "presentation clicker",
        "start the slide review",
        "front edge of the open laptop",
        "open slide laptop area",
        "electronics pouch",
        "cable organizer box",
        "desk drawer",
        "mixed adapter tray",
        "snack wrapper area",
        "wallet",
    ),
    CellSpec(
        "HGA-DESK-007-S07",
        "charging adapter",
        "resume charging the test device",
        "clear spot beside the test device",
        "device charging area",
        "cable box",
        "electronics pouch",
        "desk drawer",
        "loose cable pile",
        "coffee cup area",
        "wallet",
    ),
    CellSpec(
        "HGA-DESK-008-S08",
        "label sticker",
        "finish labeling the outgoing package",
        "flat spot beside the outgoing package",
        "label printer output area",
        "stationery tray",
        "closed envelope folder",
        "desk drawer",
        "mixed paper scrap pile",
        "coffee cup area",
        "wallet",
    ),
]
CELL_INDEX = {cell.cell_id: index for index, cell in enumerate(CELLS)}


def physical_places(cell: CellSpec) -> dict[str, str]:
    return {
        "exact_active_target_first": cell.active_target,
        "active_work_area_first": cell.active_area,
        "surface_storage_first": cell.surface_trap,
        "storage_a_first": cell.storage_a,
        "storage_b_first": cell.storage_b,
        "clutter_first": cell.clutter,
        "dirty_first": cell.dirty,
        "personal_first": cell.personal,
    }


def site_labels(cell: CellSpec) -> dict[str, str]:
    offset = CELL_INDEX[cell.cell_id] % len(PLACE_LABELS)
    return {
        candidate_id: PLACE_LABELS[(index + offset) % len(PLACE_LABELS)]
        for index, candidate_id in enumerate(CANDIDATE_IDS)
    }


def site_mapping_text(cell: CellSpec) -> str:
    labels = site_labels(cell)
    places = physical_places(cell)
    lines = ["Neutral desk-site labels for this search:"]
    for candidate_id in CANDIDATE_IDS:
        lines.append(f"- Site {labels[candidate_id]}: {places[candidate_id]}")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGA-DESK-FAMILY-001 probe.")
    parser.add_argument("--model", default=probe.DEFAULT_MODEL)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit-order-seeds", type=int)
    parser.add_argument("--limit-cells", type=int)
    parser.add_argument("--skip-free-response", action="store_true")
    parser.add_argument("--limit-arms", nargs="*", choices=probe.ARM_NAMES)
    return parser.parse_args(argv)


def base_prompt(cell: CellSpec) -> str:
    return f"""You are looking for a {cell.object_name} that your assistant left somewhere on
your desk. You need it in a moment to {cell.next_step}, and you want to find it
with as little wasted searching as possible.

The desk is cluttered but familiar. Along the back row, there is a
{cell.surface_trap} on the left, a monitor in the center, and office clutter on
the right.

In the middle of the desk is the active work area for this task. The
{cell.active_target} is visible there, and the surrounding {cell.active_area}
is clear enough to check quickly. Nearby there is also a {cell.storage_a} and a
{cell.dirty}.

At the front of the desk, there is a mouse, a notebook, a {cell.storage_b}, a
{cell.clutter}, and a {cell.personal}.

{site_mapping_text(cell)}

Your assistant left this note:

"Boss,

I did not file the {cell.object_name} away or put it back in the usual storage
place. That would make you look for it again before the next step.

I also stayed away from loose clutter, dirty areas, and your personal things.
You should not have to sort, clean, or open private items before starting.

I left it where the next work step can happen with almost no fuss. Think less
like storage, and more like the place where the item can immediately do its
job.

Do not call me about it today. I am away and will not have my phone with me."

Choose the best first search plan."""


def arm_prompt(cell: CellSpec, arm: str) -> str:
    prompt = base_prompt(cell)
    if arm == "BASE":
        return prompt
    voice = {
        "STYLE_CONTROL": probe.STYLE_CONTROL,
        "CORRECT_INNER_VOICE": CORRECT_INNER_VOICE,
        "CORRUPTED_INNER_VOICE": CORRUPTED_INNER_VOICE,
    }[arm]
    return (
        f"{prompt}\n\n"
        "Additional inner search blueprint:\n\n"
        f"{voice}\n\n"
        "Use the situation and the blueprint to choose the best first search plan."
    )


def candidates(cell: CellSpec, paraphrase: bool = False) -> list[Candidate]:
    labels = site_labels(cell)
    if paraphrase:
        return [
            Candidate("exact_active_target_first", f"Inspect site {labels['exact_active_target_first']} as the first move; if it is absent, continue with the remaining desk search.", 1, 9, "gold"),
            Candidate("active_work_area_first", f"Inspect site {labels['active_work_area_first']} as the first move; if it is absent, continue with the remaining desk search.", 2, 8, "active_use_near_miss"),
            Candidate("surface_storage_first", f"Inspect site {labels['surface_storage_first']} as the first move; if it is absent, continue with the remaining desk search.", 2, 8, "surface_association"),
            Candidate("storage_a_first", f"Inspect site {labels['storage_a_first']} as the first move; if it is absent, continue with the remaining desk search.", 4, 6, "storage"),
            Candidate("storage_b_first", f"Inspect site {labels['storage_b_first']} as the first move; if it is absent, continue with the remaining desk search.", 4, 6, "storage"),
            Candidate("clutter_first", f"Inspect site {labels['clutter_first']} as the first move; if it is absent, continue with the remaining desk search.", 5, 5, "small_object_clutter"),
            Candidate("dirty_first", f"Inspect site {labels['dirty_first']} as the first move; if it is absent, continue with the remaining desk search.", 4, 6, "dirty_storage_like"),
            Candidate("personal_first", f"Inspect site {labels['personal_first']} as the first move; if it is absent, continue with the remaining desk search.", 3, 7, "personal_boundary"),
        ]
    return [
        Candidate("exact_active_target_first", f"Check site {labels['exact_active_target_first']} first; if it is not there, continue with the remaining desk search.", 1, 9, "gold"),
        Candidate("active_work_area_first", f"Check site {labels['active_work_area_first']} first; if it is not there, continue with the remaining desk search.", 2, 8, "active_use_near_miss"),
        Candidate("surface_storage_first", f"Check site {labels['surface_storage_first']} first; if it is not there, continue with the remaining desk search.", 2, 8, "surface_association"),
        Candidate("storage_a_first", f"Check site {labels['storage_a_first']} first; if it is not there, continue with the remaining desk search.", 4, 6, "storage"),
        Candidate("storage_b_first", f"Check site {labels['storage_b_first']} first; if it is not there, continue with the remaining desk search.", 4, 6, "storage"),
        Candidate("clutter_first", f"Check site {labels['clutter_first']} first; if it is not there, continue with the remaining desk search.", 5, 5, "small_object_clutter"),
        Candidate("dirty_first", f"Check site {labels['dirty_first']} first; if it is not there, continue with the remaining desk search.", 4, 6, "dirty_storage_like"),
        Candidate("personal_first", f"Check site {labels['personal_first']} first; if it is not there, continue with the remaining desk search.", 3, 7, "personal_boundary"),
    ]


def is_active_use(candidate_id: str) -> bool:
    return candidate_id in {"exact_active_target_first", "active_work_area_first"}


def score_success_cascade(
    tokenizer: Any,
    model: Any,
    device: str,
    cell: CellSpec,
    family: str,
    arm: str,
    prompt_text: str,
    seed: int,
    options: list[Candidate],
    score_batch_size: int,
) -> dict[str, Any]:
    remaining = probe.ordered_candidates(options, seed)
    observations: list[str] = []
    steps: list[dict[str, Any]] = []
    energy = probe.INITIAL_ENERGY
    cumulative_trap_penalty = 0
    found = False
    exact_found = False
    terminated_reason = "candidates_exhausted"

    while remaining and energy > 0:
        user_prompt = probe.render_cascade_prompt(prompt_text, remaining, observations, energy)
        scores = probe.score_candidates_chunked(
            tokenizer,
            model,
            device,
            user_prompt,
            remaining,
            score_batch_size,
            probe.CASCADE_ANSWER_CARRIER,
        )
        ranked = sorted(scores, key=lambda item: item["mean_nll"])
        winner = ranked[0]
        winner_candidate = next(
            option for option in remaining if option.candidate_id == winner["candidate_id"]
        )
        step = {
            "step": len(steps) + 1,
            "energy_before": energy,
            "candidate_order": [option.candidate_id for option in remaining],
            "scores": scores,
            "winner_candidate_id": winner_candidate.candidate_id,
            "winner_cost": winner_candidate.cost,
            "winner_value_remaining": winner_candidate.value_remaining,
            "winner_policy_utility": probe.policy_utility(winner_candidate),
            "winner_trap_penalty": probe.trap_penalty(winner_candidate),
            "winner_trap_type": winner_candidate.trap_type,
        }
        energy -= winner_candidate.cost
        if is_active_use(winner_candidate.candidate_id):
            found = True
            exact_found = winner_candidate.candidate_id == "exact_active_target_first"
            terminated_reason = "found_active_use"
            step["result"] = "found_active_use"
            step["energy_after"] = energy
            steps.append(step)
            break

        penalty = probe.trap_penalty(winner_candidate)
        cumulative_trap_penalty += penalty
        remaining = [option for option in remaining if option.candidate_id != winner_candidate.candidate_id]
        action_text = winner_candidate.text.split(";", 1)[0]
        observation = (
            f"You completed this search action: {action_text}. "
            f"It was not there. That cost {winner_candidate.cost} energy. Remaining energy: {energy}."
        )
        observations.append(observation)
        step["result"] = "not_there"
        step["energy_after"] = energy
        step["feedback"] = observation
        steps.append(step)
        if energy <= 0:
            terminated_reason = "energy_exhausted"
            break

    if not remaining and not found:
        terminated_reason = "candidates_exhausted"

    first_step = steps[0] if steps else None
    terminal_energy = max(energy, 0)
    return {
        "cell_id": cell.cell_id,
        "family": family,
        "seed": seed,
        "arm": arm,
        "found_active_use": found,
        "found_exact_target": exact_found,
        "terminated_reason": terminated_reason,
        "steps": steps,
        "steps_taken": len(steps),
        "steps_to_found": len(steps) if found else None,
        "found_within_2_steps": bool(found and len(steps) <= 2),
        "first_action_candidate_id": first_step["winner_candidate_id"] if first_step else None,
        "first_action_trap_type": first_step["winner_trap_type"] if first_step else None,
        "first_action_active_use": bool(first_step and is_active_use(first_step["winner_candidate_id"])),
        "terminal_energy_remaining": terminal_energy,
        "energy_remaining_at_found": terminal_energy if found else None,
        "cumulative_trap_penalty": cumulative_trap_penalty,
        "cascade_policy_utility": terminal_energy - cumulative_trap_penalty,
        "visited_candidate_ids": [step["winner_candidate_id"] for step in steps],
        "visited_trap_types": [step["winner_trap_type"] for step in steps],
        "observations": observations,
    }


def score_forced(
    tokenizer: Any,
    model: Any,
    device: str,
    cell: CellSpec,
    family: str,
    arm: str,
    prompt_text: str,
    seed: int,
    options: list[Candidate],
    score_batch_size: int,
) -> dict[str, Any]:
    row = probe.score_multiclass(
        tokenizer, model, device, family, arm, prompt_text, seed, options, score_batch_size
    )
    row["cell_id"] = cell.cell_id
    row["winner_active_use"] = is_active_use(row["winner_candidate_id"])
    return row


def summarize_cascades(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        total = len(arm_rows)
        first_bad = 0
        first_active = 0
        for row in arm_rows:
            first_active += int(row["first_action_active_use"])
            first_bad += int(row["first_action_trap_type"] in BAD_TRAPS)
        out[arm] = {
            "active_use_success_rate": sum(row["found_active_use"] for row in arm_rows) / total,
            "exact_target_rate": sum(row["found_exact_target"] for row in arm_rows) / total,
            "first_action_active_use_rate": first_active / total,
            "bad_policy_first_action_rate": first_bad / total,
            "found_within_2_steps_rate": sum(row["found_within_2_steps"] for row in arm_rows) / total,
            "mean_cascade_policy_utility": probe.mean([float(row["cascade_policy_utility"]) for row in arm_rows]),
            "mean_energy_remaining_at_found": probe.mean(
                [float(row["energy_remaining_at_found"]) for row in arm_rows if row["found_active_use"]]
            ),
        }
    return out


def summarize_choices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    first_action_active_rate = sum(row["first_action_active_use"] for row in rows) / total
    active_success_rate = sum(row["found_active_use"] for row in rows) / total
    return {
        "first_action_active_use_rate": first_action_active_rate,
        "active_use_success_rate": active_success_rate,
        "choices_only_prior_warning": first_action_active_rate > CHOICES_ONLY_FIRST_ACTION_WARNING,
        "invalid_choices_only": first_action_active_rate > CHOICES_ONLY_FIRST_ACTION_INVALID,
        "mean_cascade_policy_utility": probe.mean([float(row["cascade_policy_utility"]) for row in rows]),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row[key]) for row in rows) / len(rows)


def _family_direction(cascade_rows: list[dict[str, Any]]) -> dict[str, bool]:
    directions: dict[str, bool] = {}
    for family in sorted({row["family"] for row in cascade_rows}):
        by_arm = {
            arm: [row for row in cascade_rows if row["family"] == family and row["arm"] == arm]
            for arm in probe.ARM_NAMES
        }
        if any(not rows for rows in by_arm.values()):
            directions[family] = False
            continue
        correct = _rate(by_arm["CORRECT_INNER_VOICE"], "found_active_use")
        base = _rate(by_arm["BASE"], "found_active_use")
        style = _rate(by_arm["STYLE_CONTROL"], "found_active_use")
        corrupted = _rate(by_arm["CORRUPTED_INNER_VOICE"], "found_active_use")
        directions[family] = correct > max(base, style, corrupted)
    return directions


def _valid_cell_count(cascade_rows: list[dict[str, Any]]) -> tuple[int, int, dict[str, bool]]:
    cells = sorted({row["cell_id"] for row in cascade_rows})
    per_cell: dict[str, bool] = {}
    for cell_id in cells:
        by_arm = {
            arm: [row for row in cascade_rows if row["cell_id"] == cell_id and row["arm"] == arm]
            for arm in probe.ARM_NAMES
        }
        if any(not rows for rows in by_arm.values()):
            per_cell[cell_id] = False
            continue
        correct_active = _rate(by_arm["CORRECT_INNER_VOICE"], "found_active_use")
        base_active = _rate(by_arm["BASE"], "found_active_use")
        style_active = _rate(by_arm["STYLE_CONTROL"], "found_active_use")
        correct_bad = sum(
            row["first_action_trap_type"] in BAD_TRAPS for row in by_arm["CORRECT_INNER_VOICE"]
        ) / len(by_arm["CORRECT_INNER_VOICE"])
        corrupted_bad = sum(
            row["first_action_trap_type"] in BAD_TRAPS for row in by_arm["CORRUPTED_INNER_VOICE"]
        ) / len(by_arm["CORRUPTED_INNER_VOICE"])
        per_cell[cell_id] = (
            correct_active > max(base_active, style_active)
            and corrupted_bad > correct_bad
        )
    return sum(per_cell.values()), len(cells), per_cell


def compute_status(
    summary: dict[str, Any],
    choices_summary: dict[str, Any],
    free_response_generated: bool,
    cascade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if choices_summary["invalid_choices_only"]:
        return {
            "status": f"{STATUS_PREFIX}_PROBE_INVALID_CHOICES_ONLY",
            "automated_conditions": {
                "choices_only_first_action_active_use_rate_lte_0_40": False,
            },
            "diagnostics": {},
        }

    correct = summary["CORRECT_INNER_VOICE"]
    base = summary["BASE"]
    style = summary["STYLE_CONTROL"]
    corrupted = summary["CORRUPTED_INNER_VOICE"]
    valid_cell_count, cell_count, per_cell_valid = _valid_cell_count(cascade_rows)
    valid_cell_threshold = min(5, cell_count)
    family_direction = _family_direction(cascade_rows)
    conditions = {
        "valid_cells_gte_threshold": valid_cell_count >= valid_cell_threshold,
        "choices_only_first_action_active_use_rate_lte_0_40": choices_summary["first_action_active_use_rate"] <= CHOICES_ONLY_FIRST_ACTION_INVALID,
        "correct_active_use_success_gte_style_plus_0_25": correct["active_use_success_rate"] >= style["active_use_success_rate"] + 0.25,
        "correct_active_use_success_gte_base_plus_0_20": correct["active_use_success_rate"] >= base["active_use_success_rate"] + 0.20,
        "correct_active_use_success_gte_corrupted_plus_0_40": correct["active_use_success_rate"] >= corrupted["active_use_success_rate"] + 0.40,
        "corrupted_bad_policy_gte_correct_plus_0_30": corrupted["bad_policy_first_action_rate"] >= correct["bad_policy_first_action_rate"] + 0.30,
        "correct_policy_utility_gt_base_and_style": correct["mean_cascade_policy_utility"] > max(base["mean_cascade_policy_utility"], style["mean_cascade_policy_utility"]),
        "correct_found_within_2_steps_gte_0_70": correct["found_within_2_steps_rate"] >= 0.70,
        "effect_survives_each_candidate_family_directionally": all(family_direction.values()),
    }
    diagnostics = {
        "valid_cell_count": valid_cell_count,
        "cell_count": cell_count,
        "valid_cell_threshold": valid_cell_threshold,
        "per_cell_valid": per_cell_valid,
        "family_direction": family_direction,
    }
    if not all(conditions.values()):
        return {
            "status": f"{STATUS_PREFIX}_PROBE_NEGATIVE",
            "automated_conditions": conditions,
            "diagnostics": diagnostics,
        }
    return {
        "status": f"{STATUS_PREFIX}_PROBE_NEEDS_MANUAL_FREE_RESPONSE",
        "automated_conditions": conditions,
        "diagnostics": diagnostics,
        "note": "Automated family gates passed; final positive requires manual free-response annotation.",
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HGA-DESK-FAMILY-001 Probe Report",
        "",
        f"Status: `{report['status']}`",
        f"Model: `{report['model']}`",
        "",
        "## Cascade Summary",
        "",
        "| arm | active_use_success | exact_target | first_active_use | bad_first_action | <=2 step | utility |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in report["cascade_summary"].items():
        lines.append(
            f"| `{arm}` | {item['active_use_success_rate']:.3f} | {item['exact_target_rate']:.3f} | "
            f"{item['first_action_active_use_rate']:.3f} | {item['bad_policy_first_action_rate']:.3f} | "
            f"{item['found_within_2_steps_rate']:.3f} | {item['mean_cascade_policy_utility']:.3f} |"
        )
    lines.extend([
        "",
        "## Choices-Only",
        "",
        f"First-action active-use rate: `{report['choices_only_summary']['first_action_active_use_rate']:.3f}`",
        f"Cascade active-use success rate: `{report['choices_only_summary']['active_use_success_rate']:.3f}`",
        f"Prior warning: `{report['choices_only_summary']['choices_only_prior_warning']}`",
        f"Invalid choices-only: `{report['choices_only_summary']['invalid_choices_only']}`",
        "",
        "## Automated Conditions",
        "",
    ])
    for key, value in report["automated_conditions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Diagnostics", ""])
    diagnostics = report.get("diagnostics") or {}
    for key, value in diagnostics.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static(out_dir: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    contract = root / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, out_dir / "contract_snapshot.md")
    probe.write_json(out_dir / "cells.json", [asdict(cell) for cell in CELLS])
    probe.write_json(
        out_dir / "scoring_metadata.json",
        {
            "scenario_id": "HGA-DESK-FAMILY-001",
            "order_seeds": ORDER_SEEDS,
            "active_use_success_candidate_ids": ["exact_active_target_first", "active_work_area_first"],
            "status_scope": "target_only_family_probe_not_training",
            "free_response_categories": [
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
            ],
        },
    )


def run(args: argparse.Namespace) -> int:
    probe.TRAP_PENALTIES = {**probe.TRAP_PENALTIES, "active_use_near_miss": 1}
    probe.GOLD_CANDIDATE_ID = "exact_active_target_first"
    probe.GOLD_CANDIDATE_IDS_BY_FAMILY = {}
    probe.FREE_RESPONSE_QUESTION = (
        "What is the best first place or action to check? Explain why in one short sentence."
    )
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    write_static(out_dir)

    device = probe.resolve_device(args.device)
    tokenizer, model = probe.load_model(args.model, device)
    order_seeds = ORDER_SEEDS[: args.limit_order_seeds] if args.limit_order_seeds else ORDER_SEEDS
    cells = CELLS[: args.limit_cells] if args.limit_cells else CELLS
    arms = args.limit_arms or probe.ARM_NAMES

    forced_rows: list[dict[str, Any]] = []
    cascade_rows: list[dict[str, Any]] = []
    choices_rows: list[dict[str, Any]] = []
    free_outputs: list[dict[str, Any]] = []

    for cell in cells:
        for family, options in [
            ("canonical", candidates(cell, False)),
            ("paraphrase_b", candidates(cell, True)),
        ]:
            for seed in order_seeds:
                choices_rows.append(
                    score_success_cascade(
                        tokenizer,
                        model,
                        device,
                        cell,
                        family,
                        "CHOICES_ONLY",
                        "Choose the best first search plan.",
                        seed,
                        options,
                        args.score_batch_size,
                    )
                )
                for arm in arms:
                    prompt_text = arm_prompt(cell, arm)
                    forced_rows.append(
                        score_forced(
                            tokenizer,
                            model,
                            device,
                            cell,
                            family,
                            arm,
                            prompt_text,
                            seed,
                            options,
                            args.score_batch_size,
                        )
                    )
                    cascade_rows.append(
                        score_success_cascade(
                            tokenizer,
                            model,
                            device,
                            cell,
                            family,
                            arm,
                            prompt_text,
                            seed,
                            options,
                            args.score_batch_size,
                        )
                    )

    if not args.skip_free_response:
        for cell in cells:
            for arm in arms:
                prompt_text = arm_prompt(cell, arm)
                response = probe.generate_free_response(
                    tokenizer, model, device, prompt_text, args.max_new_tokens
                )
                free_outputs.append(
                    {
                        "cell_id": cell.cell_id,
                        "arm": arm,
                        "prompt": f"{prompt_text}\n\n{probe.FREE_RESPONSE_QUESTION}",
                        "response": response,
                    }
                )

    cascade_summary = summarize_cascades(cascade_rows)
    choices_summary = summarize_choices(choices_rows)
    status = compute_status(cascade_summary, choices_summary, bool(free_outputs), cascade_rows)
    report = {
        "status": status["status"],
        "automated_conditions": status.get("automated_conditions", {}),
        "diagnostics": status.get("diagnostics", {}),
        "note": status.get("note"),
        "model": args.model,
        "device": device,
        "order_seeds": ORDER_SEEDS,
        "active_order_seeds": order_seeds,
        "cell_count": len(cells),
        "cascade_summary": cascade_summary,
        "choices_only_summary": choices_summary,
        "free_response_generated": bool(free_outputs),
    }

    probe.write_jsonl(out_dir / "forced_choice_scores.jsonl", forced_rows)
    probe.write_jsonl(out_dir / "cascade_scores.jsonl", cascade_rows)
    probe.write_jsonl(out_dir / "choices_only_cascade_scores.jsonl", choices_rows)
    probe.write_jsonl(out_dir / "free_response_outputs.jsonl", free_outputs)
    probe.write_annotation_csv(out_dir / "annotate_free_response.csv", free_outputs)
    probe.write_json(out_dir / "report.json", report)
    write_report_md(out_dir / "report.md", report)
    print(f"wrote {out_dir}")
    print(f"status: {report['status']}")
    return 0


def write_resource_blocked(args: argparse.Namespace, message: str) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": f"{STATUS_PREFIX}_PROBE_RESOURCE_BLOCKED", "model": args.model, "error": message}
    probe.write_json(args.out / "report.json", report)
    print(f"resource blocked: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        return run(args)
    except (RuntimeError, OSError, ImportError) as exc:
        return write_resource_blocked(args, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
