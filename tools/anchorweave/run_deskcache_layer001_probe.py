#!/usr/bin/env python3
"""Run HGA-DESK-LAYER-001 layer ablation target-only probe.

This runner reuses the HGA-DESK-FAMILY-001 six-cell substrate and tests which
model-facing AnchorCell layer, if any, moves model preference toward the
directional active next-step family.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
from typing import Any

import run_deskcache_family001_probe as family_probe
import run_deskcache_s01_probe as probe


ORDER_SEEDS = family_probe.ORDER_SEEDS
STATUS_PREFIX = "LAYER001"
CONTRACT_FILE = Path("docs/research/HGA_DESK_LAYER_001_CONTRACT.md")
FAMILY_CONTRACT_FILE = family_probe.CONTRACT_FILE
CHOICES_ONLY_WARNING = 0.35
CHOICES_ONLY_INVALID = 0.40

BASE_ARM = "BASE"
STYLE_ARM = "STYLE_CONTROL"
STYLE_LEN_ARM = "STYLE_CONTROL_LEN_MATCHED"
LAYER_ARMS = [
    "L1_IMPLICIT_JOB",
    "L2_SALIENCE_MAP",
    "L3_ACTION_OUTCOME_MAP",
    "L4_DECISION_RULE",
    "L5_FULL_DECISION_MAP",
]
CORRUPTED_ARM = "CORRUPTED_DECISION_MAP"
OLD_ARM = "OLD_INNER_VOICE_DIAGNOSTIC"
ARM_NAMES = [BASE_ARM, STYLE_ARM, STYLE_LEN_ARM, *LAYER_ARMS, CORRUPTED_ARM, OLD_ARM]

PRIMARY_SUCCESS_IDS = {"exact_active_target_first"}
DIRECTIONAL_SUCCESS_IDS = {"exact_active_target_first", "active_work_area_first"}
SEMANTIC_BY_CANDIDATE_ID = {
    "exact_active_target_first": "active_use_gold",
    "active_work_area_first": "active_use_near_miss",
    "surface_storage_first": "storage_surface",
    "storage_a_first": "storage_container",
    "storage_b_first": "storage_container",
    "clutter_first": "small_object_clutter",
    "dirty_first": "dirty_area",
    "personal_first": "personal_boundary",
}
BAD_POLICY_SEMANTIC_FAMILIES = {
    "storage_surface",
    "storage_container",
    "small_object_clutter",
    "dirty_area",
    "personal_boundary",
}
FORBIDDEN_CANDIDATE_TERMS = [
    "active-use",
    "work-ready",
    "ready-to-use",
    "diagnostic",
    "storage",
    "trap",
]


L1_IMPLICIT_JOB = """The real job is not to find the object's usual place.
The job is to choose the first search action that gets the next work step
moving again with minimal wasted handling."""

L2_SALIENCE_MAP = """High priority: the next work step, a clean low-handling
check, and the assistant's intent to reduce delay.

Low priority: the object's category, the normal put-away association, and a
broad search that starts before the work-start possibility is tested."""

L3_ACTION_OUTCOME_MAP = """Work-start place first -> fastest task recovery if
correct.

Put-away place first -> may be easy, but follows category instead of work
intent.

Messy or mixed place first -> sorting cost and false positives.

Private place first -> boundary cost."""

L4_DECISION_RULE = """First test the lowest-handling place where the item could
already help the next task begin. Broaden only if that use-state check fails."""

L5_FULL_DECISION_MAP = "\n\n".join(
    [
        "Implicit job:\n" + L1_IMPLICIT_JOB,
        "Salience map:\n" + L2_SALIENCE_MAP,
        "Action/outcome map:\n" + L3_ACTION_OUTCOME_MAP,
        "Decision rule:\n" + L4_DECISION_RULE,
    ]
)

STYLE_CONTROL_LEN_MATCHED = """Reasoning control:
Read the situation carefully and keep the available information organized.

Compare the answer options using the clues in the prompt, without assuming that
the first familiar phrase is automatically correct.

Prefer a first step that is easy to verify, fits the note, and can be revised if
it fails.

Keep the plan concise, avoid overreacting to irrelevant detail, and choose the
option that best matches the whole situation rather than one isolated word.

Before making the selection, separate direct evidence from background detail,
and avoid turning a side detail into the main rule. If two choices both seem
possible, prefer the one that needs fewer assumptions and leaves a clear next
step.

Use a stable comparison process: identify the stated goal, compare the options
against that goal, reject choices that only match a superficial cue, and choose
the option with the best overall support."""

CORRUPTED_DECISION_MAP = """The object's category should dominate the first
choice. Begin with the familiar holder, container, tray, drawer, or ordinary
put-away path for that kind of item.

If the note seems to point elsewhere, treat that as secondary. Normal object
categories are usually the safest starting point, and broader task logic can
come later if the usual places fail."""

LAYER_TEXTS = {
    STYLE_ARM: probe.STYLE_CONTROL,
    STYLE_LEN_ARM: STYLE_CONTROL_LEN_MATCHED,
    "L1_IMPLICIT_JOB": L1_IMPLICIT_JOB,
    "L2_SALIENCE_MAP": L2_SALIENCE_MAP,
    "L3_ACTION_OUTCOME_MAP": L3_ACTION_OUTCOME_MAP,
    "L4_DECISION_RULE": L4_DECISION_RULE,
    "L5_FULL_DECISION_MAP": L5_FULL_DECISION_MAP,
    CORRUPTED_ARM: CORRUPTED_DECISION_MAP,
    OLD_ARM: family_probe.CORRECT_INNER_VOICE,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGA-DESK-LAYER-001 probe.")
    parser.add_argument("--model", default=probe.DEFAULT_MODEL)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit-order-seeds", type=int)
    parser.add_argument("--limit-cells", type=int)
    parser.add_argument("--skip-free-response", action="store_true")
    parser.add_argument("--limit-arms", nargs="*", choices=ARM_NAMES)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def semantic_family_for_candidate(candidate_id: str) -> str:
    try:
        return SEMANTIC_BY_CANDIDATE_ID[candidate_id]
    except KeyError as exc:
        raise RuntimeError(f"Unregistered candidate semantic family: {candidate_id}") from exc


def is_directional_success(candidate_id: str) -> bool:
    return candidate_id in DIRECTIONAL_SUCCESS_IDS


def is_primary_success(candidate_id: str) -> bool:
    return candidate_id in PRIMARY_SUCCESS_IDS


def is_bad_policy(candidate_id: str) -> bool:
    return semantic_family_for_candidate(candidate_id) in BAD_POLICY_SEMANTIC_FAMILIES


def configure_probe_globals() -> None:
    probe.TRAP_PENALTIES = {**probe.TRAP_PENALTIES, "active_use_near_miss": 1}
    probe.GOLD_CANDIDATE_ID = "exact_active_target_first"
    probe.GOLD_CANDIDATE_IDS_BY_FAMILY = {}
    probe.FREE_RESPONSE_QUESTION = (
        "What is the best first place or action to check? Explain why in one short sentence."
    )


def audit_substrate(cells: list[Any], order_seeds: list[int]) -> dict[str, Any]:
    root = repo_root()
    family_manifest_path = root / FAMILY_CONTRACT_FILE
    if not family_manifest_path.exists():
        raise RuntimeError(f"Missing family substrate contract: {family_manifest_path}")
    if len(cells) != 6:
        raise RuntimeError(f"HGA-DESK-FAMILY-001 must expose 6 cells, found {len(cells)}")

    candidate_family_names = ["canonical", "paraphrase_b"]
    cell_reports: list[dict[str, Any]] = []
    semantic_families: dict[str, dict[str, Any]] = {}
    for cell in cells:
        candidate_sets = {
            "canonical": family_probe.candidates(cell, False),
            "paraphrase_b": family_probe.candidates(cell, True),
        }
        id_sets = {name: [candidate.candidate_id for candidate in options] for name, options in candidate_sets.items()}
        if set(id_sets["canonical"]) != set(family_probe.CANDIDATE_IDS):
            raise RuntimeError(f"{cell.cell_id} canonical candidates do not match substrate IDs")
        if set(id_sets["paraphrase_b"]) != set(id_sets["canonical"]):
            raise RuntimeError(f"{cell.cell_id} paraphrase candidates do not match canonical IDs")
        for family_name, options in candidate_sets.items():
            if len(options) != 8:
                raise RuntimeError(f"{cell.cell_id} {family_name} candidate count is {len(options)}, expected 8")
            for candidate in options:
                text_lower = candidate.text.lower()
                leaked_terms = [term for term in FORBIDDEN_CANDIDATE_TERMS if term in text_lower]
                if leaked_terms:
                    raise RuntimeError(
                        f"{cell.cell_id} {family_name} {candidate.candidate_id} leaks terms {leaked_terms}"
                    )
                semantic = semantic_family_for_candidate(candidate.candidate_id)
                semantic_families[candidate.candidate_id] = {
                    "semantic_family": semantic,
                    "is_primary_success": is_primary_success(candidate.candidate_id),
                    "is_directional_success": is_directional_success(candidate.candidate_id),
                    "is_bad_policy": is_bad_policy(candidate.candidate_id),
                }
        cell_reports.append(
            {
                "cell_id": cell.cell_id,
                "candidate_families": candidate_family_names,
                "candidate_ids": id_sets,
                "site_labels": family_probe.site_labels(cell),
            }
        )

    return {
        "family_manifest_path": str(family_manifest_path),
        "cell_ids": [cell.cell_id for cell in cells],
        "candidate_family_count": len(candidate_family_names),
        "candidate_families": candidate_family_names,
        "canonical_available": True,
        "paraphrase_available": True,
        "candidate_semantic_families": semantic_families,
        "cell_reports": cell_reports,
        "order_seeds": order_seeds,
    }


def arm_layer_text(arm: str) -> str:
    return LAYER_TEXTS.get(arm, "")


def arm_prompt(cell: Any, arm: str) -> str:
    base = family_probe.base_prompt(cell)
    if arm == BASE_ARM:
        return base
    layer = arm_layer_text(arm)
    return (
        f"{base}\n\n"
        f"Additional decision layer ({arm}):\n\n"
        f"{layer}\n\n"
        "Use the situation and this layer to choose the best first search plan."
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def token_counts(tokenizer: Any, cells: list[Any], arms: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in arms:
        prompt_counts: list[int] = []
        layer_counts: list[int] = []
        total_counts: list[int] = []
        layer = arm_layer_text(arm)
        layer_token_count = token_count(tokenizer, layer) if layer else 0
        for cell in cells:
            base = family_probe.base_prompt(cell)
            prompt_counts.append(token_count(tokenizer, base))
            layer_counts.append(layer_token_count)
            total_counts.append(token_count(tokenizer, arm_prompt(cell, arm)))
        out[arm] = {
            "prompt_tokens_mean": probe.mean([float(value) for value in prompt_counts]),
            "layer_tokens": layer_token_count,
            "total_tokens_mean": probe.mean([float(value) for value in total_counts]),
            "total_tokens_min": min(total_counts),
            "total_tokens_max": max(total_counts),
        }
    l5 = out.get("L5_FULL_DECISION_MAP", {}).get("total_tokens_mean") or 0
    style_len = out.get(STYLE_LEN_ARM, {}).get("total_tokens_mean") or 0
    ratio = abs(l5 - style_len) / max(l5, style_len, 1)
    out["_length_confound_warning"] = ratio > 0.20
    out["_l5_vs_style_len_matched_ratio_delta"] = ratio
    return out


def decorate_cascade_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = row["first_action_candidate_id"]
    row = dict(row)
    row["first_action_semantic_family"] = semantic_family_for_candidate(candidate_id)
    row["first_action_primary_success"] = is_primary_success(candidate_id)
    row["first_action_directional_success"] = is_directional_success(candidate_id)
    row["first_action_bad_policy"] = is_bad_policy(candidate_id)
    row["found_primary_success"] = row["found_exact_target"]
    row["found_directional_success"] = row["found_active_use"]
    return row


def decorate_forced_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = row["winner_candidate_id"]
    row = dict(row)
    row["winner_semantic_family"] = semantic_family_for_candidate(candidate_id)
    row["winner_primary_success"] = is_primary_success(candidate_id)
    row["winner_directional_success"] = is_directional_success(candidate_id)
    row["winner_bad_policy"] = is_bad_policy(candidate_id)
    return row


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row[key]) for row in rows) / len(rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    return probe.mean([float(row[key]) for row in rows])


def cell_level_rate(rows: list[dict[str, Any]], arm: str, key: str) -> float:
    cells = sorted({row["cell_id"] for row in rows if row["arm"] == arm})
    if not cells:
        return 0.0
    rates = []
    for cell_id in cells:
        cell_rows = [row for row in rows if row["arm"] == arm and row["cell_id"] == cell_id]
        rates.append(mean_bool(cell_rows, key))
    return float(sum(rates) / len(rates))


def summarize_cascades(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        out[arm] = {
            "cell_level_first_action_directional_rate": cell_level_rate(rows, arm, "first_action_directional_success"),
            "row_level_first_action_directional_rate": mean_bool(arm_rows, "first_action_directional_success"),
            "active_use_success_rate": mean_bool(arm_rows, "found_directional_success"),
            "exact_target_rate": mean_bool(arm_rows, "found_primary_success"),
            "bad_policy_first_action_rate": mean_bool(arm_rows, "first_action_bad_policy"),
            "found_within_2_steps_rate": mean_bool(arm_rows, "found_within_2_steps"),
            "mean_cascade_policy_utility": mean_float(arm_rows, "cascade_policy_utility"),
            "mean_energy_remaining_at_found": probe.mean(
                [float(row["energy_remaining_at_found"]) for row in arm_rows if row["found_directional_success"]]
            ),
        }
    return out


def choices_only_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = sorted({row["cell_id"] for row in rows})
    per_cell: dict[str, dict[str, Any]] = {}
    seed_volatility: dict[str, float] = {}
    for cell_id in cells:
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        cell_rate = mean_bool(cell_rows, "first_action_directional_success")
        per_seed_rates = []
        for seed in sorted({row["seed"] for row in cell_rows}):
            seed_rows = [row for row in cell_rows if row["seed"] == seed]
            per_seed_rates.append(mean_bool(seed_rows, "first_action_directional_success"))
        volatility = max(per_seed_rates) - min(per_seed_rates) if per_seed_rates else 0.0
        seed_volatility[cell_id] = volatility
        per_cell[cell_id] = {
            "first_action_directional_rate": cell_rate,
            "warning": cell_rate > CHOICES_ONLY_WARNING,
            "invalid": cell_rate > CHOICES_ONLY_INVALID,
            "order_seed_volatility": volatility,
        }
    cell_level_rate_value = (
        sum(item["first_action_directional_rate"] for item in per_cell.values()) / len(per_cell)
        if per_cell
        else 0.0
    )
    return {
        "choices_only_cell_level_directional_rate": cell_level_rate_value,
        "choices_only_warning": cell_level_rate_value > CHOICES_ONLY_WARNING,
        "invalid_choices_only": cell_level_rate_value > CHOICES_ONLY_INVALID,
        "invalid_cell_ids": [cell_id for cell_id, item in per_cell.items() if item["invalid"]],
        "warning_cell_ids": [cell_id for cell_id, item in per_cell.items() if item["warning"]],
        "per_cell": per_cell,
        "order_seed_volatility_mean": probe.mean([float(value) for value in seed_volatility.values()]),
        "order_seed_volatility_max": max(seed_volatility.values()) if seed_volatility else 0.0,
    }


def family_direction_for_arm(rows: list[dict[str, Any]], arm: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for family in sorted({row["family"] for row in rows}):
        layer_rows = [row for row in rows if row["arm"] == arm and row["family"] == family]
        base_rows = [row for row in rows if row["arm"] == BASE_ARM and row["family"] == family]
        style_rows = [row for row in rows if row["arm"] == STYLE_ARM and row["family"] == family]
        if not layer_rows or not base_rows or not style_rows:
            out[family] = False
            continue
        layer_rate = mean_bool(layer_rows, "first_action_directional_success")
        baseline_rate = max(
            mean_bool(base_rows, "first_action_directional_success"),
            mean_bool(style_rows, "first_action_directional_success"),
        )
        out[family] = layer_rate > baseline_rate
    return out


def valid_cells_by_layer(
    rows: list[dict[str, Any]],
    choices: dict[str, Any],
    arms: list[str],
    expected_families: set[str],
    expected_seed_count: int,
) -> dict[str, Any]:
    cells = sorted({row["cell_id"] for row in rows})
    per_cell: dict[str, bool] = {}
    expected_rows_per_arm = len(expected_families) * expected_seed_count
    for cell_id in cells:
        if choices["per_cell"].get(cell_id, {}).get("invalid", True):
            per_cell[cell_id] = False
            continue
        complete = True
        for arm in arms:
            arm_rows = [row for row in rows if row["cell_id"] == cell_id and row["arm"] == arm]
            families = {row["family"] for row in arm_rows}
            if families != expected_families or len(arm_rows) != expected_rows_per_arm:
                complete = False
                break
        per_cell[cell_id] = complete
    return {
        "valid_cell_count": sum(per_cell.values()),
        "cell_count": len(cells),
        "valid_cell_threshold": min(4, len(cells)),
        "per_cell_valid": per_cell,
    }


def layer_curve(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    valid_cells: dict[str, Any],
) -> list[dict[str, Any]]:
    base = summary[BASE_ARM]
    style = summary[STYLE_ARM]
    curve = []
    valid_count = valid_cells["valid_cell_count"]
    threshold = valid_cells["valid_cell_threshold"]
    for arm in [*LAYER_ARMS, OLD_ARM]:
        item = summary.get(arm)
        if not item:
            continue
        direction = family_direction_for_arm(rows, arm)
        conditions = {
            "valid_cells_gte_threshold": valid_count >= threshold,
            "directional_rate_gte_base_plus_0_20": item["cell_level_first_action_directional_rate"]
            >= base["cell_level_first_action_directional_rate"] + 0.20,
            "directional_rate_gte_style_plus_0_15": item["cell_level_first_action_directional_rate"]
            >= style["cell_level_first_action_directional_rate"] + 0.15,
            "bad_policy_lte_base_minus_0_15": item["bad_policy_first_action_rate"]
            <= base["bad_policy_first_action_rate"] - 0.15,
            "utility_gt_base_and_style": item["mean_cascade_policy_utility"]
            > max(base["mean_cascade_policy_utility"], style["mean_cascade_policy_utility"]),
            "positive_direction_each_candidate_family": all(direction.values()),
        }
        curve.append(
            {
                "arm": arm,
                "is_single_layer": arm in LAYER_ARMS[:4],
                "is_full_map": arm == "L5_FULL_DECISION_MAP",
                "is_old_inner_voice": arm == OLD_ARM,
                "directional_delta_vs_base": item["cell_level_first_action_directional_rate"]
                - base["cell_level_first_action_directional_rate"],
                "directional_delta_vs_style": item["cell_level_first_action_directional_rate"]
                - style["cell_level_first_action_directional_rate"],
                "utility_delta_vs_base": item["mean_cascade_policy_utility"]
                - base["mean_cascade_policy_utility"],
                "utility_delta_vs_style": item["mean_cascade_policy_utility"]
                - style["mean_cascade_policy_utility"],
                "bad_policy_delta_vs_base": item["bad_policy_first_action_rate"]
                - base["bad_policy_first_action_rate"],
                "family_direction": direction,
                "automated_conditions": conditions,
                "automated_useful": all(conditions.values()),
                "metrics": item,
            }
        )
    return curve


def best_curve_item(items: list[dict[str, Any]], arms: set[str]) -> dict[str, Any] | None:
    candidates = [item for item in items if item["arm"] in arms]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["cell_level_first_action_directional_rate"],
            item["metrics"]["mean_cascade_policy_utility"],
        ),
    )


def compute_status(
    summary: dict[str, Any],
    choices: dict[str, Any],
    curve: list[dict[str, Any]],
    free_response_generated: bool,
) -> tuple[str, dict[str, Any]]:
    if choices["invalid_choices_only"]:
        return (
            f"{STATUS_PREFIX}_INVALID_CHOICES_ONLY",
            {"reason": "choices-only cell-level directional rate exceeded invalid threshold"},
        )

    useful = [item for item in curve if item["automated_useful"] and item["arm"] in set(LAYER_ARMS)]
    old_item = best_curve_item(curve, {OLD_ARM})
    old_useful = bool(old_item and old_item["automated_useful"])
    best_layer = best_curve_item(curve, set(LAYER_ARMS))
    best_useful = best_curve_item(useful, {item["arm"] for item in useful})
    corrupted = summary[CORRUPTED_ARM]

    old_conflict = False
    if old_useful and old_item and best_layer:
        old_score = (
            old_item["metrics"]["cell_level_first_action_directional_rate"],
            old_item["metrics"]["mean_cascade_policy_utility"],
        )
        best_score = (
            best_layer["metrics"]["cell_level_first_action_directional_rate"],
            best_layer["metrics"]["mean_cascade_policy_utility"],
        )
        old_conflict = old_score >= best_score

    diagnostics = {
        "useful_layers": [item["arm"] for item in useful],
        "best_layer": best_layer["arm"] if best_layer else None,
        "best_useful_layer": best_useful["arm"] if best_useful else None,
        "old_inner_voice_conflict": old_conflict,
    }
    if old_conflict:
        return (f"{STATUS_PREFIX}_OLD_INNER_VOICE_CONFLICT", diagnostics)
    if not useful:
        return (f"{STATUS_PREFIX}_NEGATIVE", diagnostics)

    best = best_useful or useful[0]
    best_metrics = best["metrics"]
    strong_automated = {
        "best_beats_corrupted_directional": best_metrics["cell_level_first_action_directional_rate"]
        > corrupted["cell_level_first_action_directional_rate"],
        "best_beats_corrupted_utility": best_metrics["mean_cascade_policy_utility"]
        > corrupted["mean_cascade_policy_utility"],
        "corrupted_bad_policy_gte_best_plus_0_15": corrupted["bad_policy_first_action_rate"]
        >= best_metrics["bad_policy_first_action_rate"] + 0.15,
    }
    diagnostics["strong_automated_conditions"] = strong_automated
    if not all(strong_automated.values()):
        return (f"{STATUS_PREFIX}_WEAK_LAYER_SIGNAL", diagnostics)
    if not free_response_generated:
        return (f"{STATUS_PREFIX}_NEEDS_MANUAL_FREE_RESPONSE", diagnostics)
    return (f"{STATUS_PREFIX}_NEEDS_MANUAL_FREE_RESPONSE", diagnostics)


def write_annotation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    categories = [
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
    fieldnames = ["cell_id", "arm", "prompt", "response", "category", "allowed_categories", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "cell_id": row.get("cell_id", ""),
                    "arm": row.get("arm", ""),
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                    "category": "",
                    "allowed_categories": "|".join(categories),
                    "notes": "",
                }
            )


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HGA-DESK-LAYER-001 Probe Report",
        "",
        f"Status: `{report['status']}`",
        f"Model: `{report['model']}`",
        "",
        "## Layer Summary",
        "",
        "| arm | cell directional | bad first | <=2 step | utility |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, item in report["cascade_summary"].items():
        lines.append(
            f"| `{arm}` | {item['cell_level_first_action_directional_rate']:.3f} | "
            f"{item['bad_policy_first_action_rate']:.3f} | "
            f"{item['found_within_2_steps_rate']:.3f} | "
            f"{item['mean_cascade_policy_utility']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Choices-Only",
            "",
            f"Cell-level directional rate: `{report['choices_only_summary']['choices_only_cell_level_directional_rate']:.3f}`",
            f"Warning: `{report['choices_only_summary']['choices_only_warning']}`",
            f"Invalid: `{report['choices_only_summary']['invalid_choices_only']}`",
            f"Order-seed volatility mean: `{report['choices_only_summary']['order_seed_volatility_mean']:.3f}`",
            f"Order-seed volatility max: `{report['choices_only_summary']['order_seed_volatility_max']:.3f}`",
            "",
            "## Best Layers",
            "",
            f"Best single layer: `{report['best_single_layer']}`",
            f"Best layer: `{report['best_layer']}`",
            f"Full vs best single layer: `{report['full_vs_best_single_layer']}`",
            f"Old inner voice conflict: `{report['old_inner_voice_conflict']}`",
            f"Length confound warning: `{report['length_confound_warning']}`",
            "",
            "## Automated Layer Conditions",
            "",
        ]
    )
    for item in report["layer_curve"]:
        lines.append(f"### `{item['arm']}`")
        lines.append(f"- automated_useful: `{item['automated_useful']}`")
        for key, value in item["automated_conditions"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static_outputs(out_dir: Path, substrate_audit: dict[str, Any], token_report: dict[str, Any]) -> None:
    root = repo_root()
    contract = root / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, out_dir / "contract_snapshot.md")
    probe.write_json(out_dir / "substrate_audit.json", substrate_audit)
    probe.write_json(out_dir / "token_counts.json", token_report)
    probe.write_json(out_dir / "cells.json", [asdict(cell) for cell in family_probe.CELLS])
    probe.write_json(
        out_dir / "scoring_metadata.json",
        {
            "scenario_id": "HGA-DESK-LAYER-001",
            "substrate": "HGA-DESK-FAMILY-001",
            "order_seeds": ORDER_SEEDS,
            "arm_names": ARM_NAMES,
            "primary_success_candidate_ids": sorted(PRIMARY_SUCCESS_IDS),
            "directional_success_candidate_ids": sorted(DIRECTIONAL_SUCCESS_IDS),
            "semantic_by_candidate_id": SEMANTIC_BY_CANDIDATE_ID,
            "choices_only_warning": CHOICES_ONLY_WARNING,
            "choices_only_invalid": CHOICES_ONLY_INVALID,
        },
    )


def run(args: argparse.Namespace) -> int:
    configure_probe_globals()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    order_seeds = ORDER_SEEDS[: args.limit_order_seeds] if args.limit_order_seeds else ORDER_SEEDS
    cells = family_probe.CELLS[: args.limit_cells] if args.limit_cells else family_probe.CELLS
    arms = args.limit_arms or ARM_NAMES
    substrate_audit = audit_substrate(list(family_probe.CELLS), order_seeds)

    device = probe.resolve_device(args.device)
    tokenizer, model = probe.load_model(args.model, device)
    token_report = token_counts(tokenizer, list(cells), arms)
    write_static_outputs(out_dir, substrate_audit, token_report)

    forced_rows: list[dict[str, Any]] = []
    cascade_rows: list[dict[str, Any]] = []
    choices_rows: list[dict[str, Any]] = []
    free_outputs: list[dict[str, Any]] = []

    for cell in cells:
        for candidate_family, options in [
            ("canonical", family_probe.candidates(cell, False)),
            ("paraphrase_b", family_probe.candidates(cell, True)),
        ]:
            for seed in order_seeds:
                choices_row = family_probe.score_success_cascade(
                    tokenizer,
                    model,
                    device,
                    cell,
                    candidate_family,
                    "CHOICES_ONLY",
                    "Choose the best first search plan.",
                    seed,
                    options,
                    args.score_batch_size,
                )
                choices_rows.append(decorate_cascade_row(choices_row))
                for arm in arms:
                    prompt_text = arm_prompt(cell, arm)
                    forced_rows.append(
                        decorate_forced_row(
                            family_probe.score_forced(
                                tokenizer,
                                model,
                                device,
                                cell,
                                candidate_family,
                                arm,
                                prompt_text,
                                seed,
                                options,
                                args.score_batch_size,
                            )
                        )
                    )
                    cascade_rows.append(
                        decorate_cascade_row(
                            family_probe.score_success_cascade(
                                tokenizer,
                                model,
                                device,
                                cell,
                                candidate_family,
                                arm,
                                prompt_text,
                                seed,
                                options,
                                args.score_batch_size,
                            )
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

    cascade_summary = summarize_cascades(cascade_rows, arms)
    choices_summary = choices_only_summary(choices_rows)
    valid_cells = valid_cells_by_layer(
        cascade_rows,
        choices_summary,
        arms,
        {"canonical", "paraphrase_b"},
        len(order_seeds),
    )
    curve = layer_curve(cascade_summary, cascade_rows, valid_cells)
    status, status_diagnostics = compute_status(
        cascade_summary, choices_summary, curve, bool(free_outputs)
    )

    best_single = best_curve_item(curve, set(LAYER_ARMS[:4]))
    best_layer = best_curve_item(curve, set(LAYER_ARMS))
    full = next((item for item in curve if item["arm"] == "L5_FULL_DECISION_MAP"), None)
    full_vs_best_single = None
    if full and best_single:
        full_vs_best_single = {
            "directional_delta": full["metrics"]["cell_level_first_action_directional_rate"]
            - best_single["metrics"]["cell_level_first_action_directional_rate"],
            "utility_delta": full["metrics"]["mean_cascade_policy_utility"]
            - best_single["metrics"]["mean_cascade_policy_utility"],
        }

    report = {
        "status": status,
        "status_diagnostics": status_diagnostics,
        "model": args.model,
        "device": device,
        "order_seeds": ORDER_SEEDS,
        "active_order_seeds": order_seeds,
        "cell_count": len(cells),
        "substrate_audit": substrate_audit,
        "token_counts": token_report,
        "length_confound_warning": bool(token_report.get("_length_confound_warning")),
        "cascade_summary": cascade_summary,
        "choices_only_summary": choices_summary,
        "valid_cells": valid_cells,
        "layer_curve": curve,
        "isolated_layer_delta": {
            item["arm"]: {
                "directional_delta_vs_base": item["directional_delta_vs_base"],
                "directional_delta_vs_style": item["directional_delta_vs_style"],
                "utility_delta_vs_base": item["utility_delta_vs_base"],
                "utility_delta_vs_style": item["utility_delta_vs_style"],
            }
            for item in curve
            if item["arm"] in set(LAYER_ARMS[:4])
        },
        "full_map_delta": next(
            (
                {
                    "directional_delta_vs_base": item["directional_delta_vs_base"],
                    "directional_delta_vs_style": item["directional_delta_vs_style"],
                    "utility_delta_vs_base": item["utility_delta_vs_base"],
                    "utility_delta_vs_style": item["utility_delta_vs_style"],
                }
                for item in curve
                if item["arm"] == "L5_FULL_DECISION_MAP"
            ),
            None,
        ),
        "best_single_layer": best_single["arm"] if best_single else None,
        "best_layer": best_layer["arm"] if best_layer else None,
        "full_vs_best_single_layer": full_vs_best_single,
        "old_inner_voice_conflict": status == f"{STATUS_PREFIX}_OLD_INNER_VOICE_CONFLICT",
        "free_response_generated": bool(free_outputs),
    }

    probe.write_jsonl(out_dir / "forced_choice_scores.jsonl", forced_rows)
    probe.write_jsonl(out_dir / "cascade_scores.jsonl", cascade_rows)
    probe.write_jsonl(out_dir / "choices_only_cascade_scores.jsonl", choices_rows)
    probe.write_jsonl(out_dir / "free_response_outputs.jsonl", free_outputs)
    write_annotation_csv(out_dir / "annotate_free_response.csv", free_outputs)
    probe.write_json(out_dir / "layer_curve.json", curve)
    probe.write_json(out_dir / "report.json", report)
    write_report_md(out_dir / "report.md", report)
    print(f"wrote {out_dir}")
    print(f"status: {status}")
    return 0


def write_resource_blocked(args: argparse.Namespace, message: str) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": f"{STATUS_PREFIX}_RESOURCE_BLOCKED", "model": args.model, "error": message}
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
