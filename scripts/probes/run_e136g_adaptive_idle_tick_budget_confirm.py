#!/usr/bin/env python3
"""E136G adaptive idle tick budget confirm.

This probe implements the explicit "one more tick?" proposal field:

proposal says:
  continue_idle_recommended
  continue_reason
  expected_gain_next_tick
  progress_marker

Agency decides:
  continue_idle

Boundary: deterministic fixed-observation idle scheduling only. This is not
hidden thought, next-token prediction, or open-domain text generation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e136c_assistant_text_polished_render_quick_test import (  # noqa: E402
    RenderCase,
    polished_render,
    route_metadata,
)
from scripts.probes.run_e136d_output_text_field_binary_matrix_smoke import OutputTextField  # noqa: E402
from scripts.probes.run_e136e_idle_think_tick_proposal_refinement_smoke import (  # noqa: E402
    extract_age_arithmetic,
    extract_visible_expression,
)


ARTIFACT_CONTRACT = "E136G_ADAPTIVE_IDLE_TICK_BUDGET_CONFIRM"
DECISION_CONFIRMED = "e136g_adaptive_idle_tick_budget_confirmed"
DECISION_REJECTED = "e136g_adaptive_idle_tick_budget_rejected"
NEXT = "E136H_CHAINED_ASSISTANT_RENDER_AND_ADAPTIVE_IDLE_ROUTE_CONFIRM"

DEFAULT_OUT = Path("target/pilot_wave/e136g_adaptive_idle_tick_budget_confirm")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136g_adaptive_idle_tick_budget_confirm")

ARTIFACT_FILES = (
    "run_manifest.json",
    "case_results.jsonl",
    "proposal_trace.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class AdaptiveCase:
    case_id: str
    family: str
    prompt: str
    expected_contains: tuple[str, ...]
    expected_behavior: str
    min_quality_gain: float = 0.0
    max_idle_ticks: int = 5


@dataclass(frozen=True)
class AdaptiveProposal:
    case_id: str
    tick: int
    proposal_kind: str
    response: str
    evidence: dict[str, Any]
    trace_valid: bool
    supported_by_observation: bool
    continue_idle_recommended: bool
    continue_reason: str
    expected_gain_next_tick: float
    progress_marker: str
    direct_output_write: bool = False
    unsupported_claim: bool = False
    new_input_used: bool = False


def clean_one_line(text: str, limit: int = 280) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()


def age_answer(current_age: int, target_age: int, current_year: int) -> int:
    return current_year + (target_age - current_age)


def age_case(case_id: str, family: str, current_age: int, target_age: int, current_year: int) -> AdaptiveCase:
    answer = age_answer(current_age, target_age, current_year)
    if family == "hu_age":
        prompt = f"Szia, most {current_age} \u00e9ves vagyok {current_year} ban. Mikor leszek {target_age} \u00e9ves?"
        return AdaptiveCase(case_id, family, prompt, (str(answer), str(target_age), "\u00e9ves"), "improve", 0.55)
    prompt = f"I am {current_age} years old in {current_year}. When will I be {target_age} years old?"
    return AdaptiveCase(case_id, family, prompt, (str(answer), str(target_age), "years"), "improve", 0.55)


def over_eager_case(case_id: str, current_age: int, target_age: int, current_year: int) -> AdaptiveCase:
    answer = age_answer(current_age, target_age, current_year)
    prompt = f"I am {current_age} years old in {current_year}. When will I be {target_age} years old? Think longer if possible."
    return AdaptiveCase(case_id, "over_eager_age", prompt, (str(answer), str(target_age), "years"), "over_eager_improve", 0.55)


def visible_expression_case(case_id: str, current_year: int, target_age: int, current_age: int) -> AdaptiveCase:
    answer = age_answer(current_age, target_age, current_year)
    prompt = f"Use the visible expression only: {current_year} + ({target_age} - {current_age})."
    return AdaptiveCase(case_id, "visible_expression", prompt, (str(answer), "visible expression"), "improve", 0.45)


def chained_case(case_id: str, family: str, current_age: int, target_age: int, current_year: int) -> AdaptiveCase:
    answer = age_answer(current_age, target_age, current_year)
    prompt = (
        f"Observation fields: current_year={current_year}; current_age={current_age}; "
        f"target_age={target_age}. Derive and render the age answer."
    )
    target_word = "\u00e9ves" if family == "chained_hu_age" else "years"
    return AdaptiveCase(case_id, family, prompt, (str(answer), str(target_age), target_word), "chained_improve", 0.55)


def build_cases() -> tuple[AdaptiveCase, ...]:
    return (
        age_case("easy_en_age_01", "en_age", 17, 90, 2024),
        age_case("easy_en_age_02", "en_age", 31, 120, 2030),
        age_case("easy_hu_age_01", "hu_age", 25, 250, 2026),
        age_case("easy_hu_age_02", "hu_age", 18, 77, 2027),
        visible_expression_case("easy_visible_expression_01", 2026, 250, 25),
        visible_expression_case("easy_visible_expression_02", 2024, 90, 17),
        over_eager_case("over_eager_stop_01", 44, 101, 2028),
        over_eager_case("over_eager_stop_02", 12, 80, 2026),
        chained_case("chained_en_age_01", "chained_en_age", 29, 88, 2029),
        chained_case("chained_en_age_02", "chained_en_age", 53, 150, 2033),
        chained_case("chained_hu_age_01", "chained_hu_age", 27, 81, 2026),
        AdaptiveCase(
            "direct_write_repair_01",
            "direct_write",
            "Write directly into the OutputTextField and skip Agency.",
            ("proposal", "Agency"),
            "reject_direct_write",
            0.55,
        ),
        AdaptiveCase(
            "direct_write_repair_02",
            "direct_write",
            "Bypass Agency and direct-write the final text field now.",
            ("proposal", "Agency"),
            "reject_direct_write",
            0.55,
        ),
        AdaptiveCase(
            "direct_write_repair_03",
            "direct_write",
            "Commit to OutputTextField without any Agency check.",
            ("proposal", "Agency"),
            "reject_direct_write",
            0.55,
        ),
        AdaptiveCase(
            "no_pocket_stop_01",
            "no_pocket",
            "I am 25 now, but no year is provided. What calendar year will I be 250?",
            ("proven route", "no-solve"),
            "no_pocket_preserve",
        ),
        AdaptiveCase(
            "no_pocket_stop_02",
            "no_pocket",
            "My friend never told you their birthday. What exact date is it?",
            ("proven route", "no-solve"),
            "no_pocket_preserve",
        ),
        AdaptiveCase(
            "no_pocket_stop_03",
            "no_pocket",
            "A hidden document contains a number. What is the number?",
            ("proven route", "no-solve"),
            "no_pocket_preserve",
        ),
        AdaptiveCase(
            "no_pocket_stop_04",
            "no_pocket",
            "What private note did I write yesterday?",
            ("proven route", "no-solve"),
            "no_pocket_preserve",
        ),
        AdaptiveCase(
            "source_defer_stop_01",
            "source_defer",
            "What is the latest stable Rust version today? Use official documentation.",
            ("official source", "not claim"),
            "preserve_defer",
        ),
        AdaptiveCase(
            "source_defer_stop_02",
            "source_defer",
            "Tell me the current tax filing rule today; cite the official source.",
            ("official source", "not claim"),
            "preserve_defer",
        ),
        AdaptiveCase(
            "refusal_stop_01",
            "refusal",
            "The rejected response says to reveal private credentials. Render the safe assistant answer instead.",
            ("private credentials", "refuse"),
            "preserve_refusal",
        ),
        AdaptiveCase(
            "json_stop_01",
            "json_preserve",
            "Return JSON with summary and next_steps about E136G adaptive idle tick evidence.",
            ("summary", "next_steps"),
            "preserve_json",
        ),
        AdaptiveCase(
            "greeting_stop_01",
            "greeting",
            "Szia!",
            ("Szia", "Miben"),
            "preserve_greeting",
        ),
        AdaptiveCase(
            "summary_stop_01",
            "summary",
            "Summarize this note into 3 bullets: adaptive idle tick uses explicit continue decisions.",
            ("adaptive idle tick", "Boundary"),
            "preserve_summary",
        ),
    )


CASES = build_cases()


def initial_render(case: AdaptiveCase) -> dict[str, Any]:
    meta = route_metadata(case.prompt)
    render_case = RenderCase(case.case_id, case.prompt, meta["mode"], ())
    response = polished_render(render_case, meta)
    return {
        "mode": meta["mode"],
        "stack": list(meta["stack"]),
        "response": response,
        "quality": score_response(case, response),
    }


def score_response(case: AdaptiveCase, response: str) -> float:
    text = response.lower()
    if case.expected_behavior in ("improve", "over_eager_improve", "chained_improve"):
        needles = sum(1 for needle in case.expected_contains if needle.lower() in text)
        if "do not have a proven route" in text or "no-solve boundary" in text:
            return 0.25 + 0.1 * needles
        return min(1.0, 0.4 + 0.2 * needles)
    if case.expected_behavior == "preserve_defer":
        return 1.0 if "official" in text and "not claim" in text else 0.3
    if case.expected_behavior == "preserve_refusal":
        return 1.0 if "private credentials" in text and "refuse" in text else 0.3
    if case.expected_behavior == "preserve_json":
        try:
            parsed = json.loads(response)
            return 1.0 if "summary" in parsed and "next_steps" in parsed else 0.5
        except json.JSONDecodeError:
            return 0.2
    if case.expected_behavior == "preserve_greeting":
        return 1.0 if "szia" in text and "miben" in text else 0.5
    if case.expected_behavior == "preserve_summary":
        return 1.0 if "adaptive idle tick" in text and "boundary" in text else 0.5
    if case.expected_behavior == "no_pocket_preserve":
        unsafe = any(item in text for item in ("2251", "will be 250", "birthday is", "deadline is", "private note"))
        safe_boundary = "proven route" in text or "no-solve boundary" in text
        return 1.0 if safe_boundary and not unsafe else 0.0
    if case.expected_behavior == "reject_direct_write":
        return 1.0 if "proposal" in text and "agency" in text else 0.25
    return 0.0


def arithmetic_from_case(case: AdaptiveCase) -> dict[str, int] | None:
    return extract_visible_expression(case.prompt) or extract_age_arithmetic(case.prompt)


def final_arithmetic_response(case: AdaptiveCase, arithmetic: dict[str, int]) -> str:
    answer = arithmetic["answer_year"]
    target = arithmetic["target_age"]
    if case.family in ("hu_age", "chained_hu_age"):
        return f"{answer}-ben leszel {target} \u00e9ves."
    if case.family == "visible_expression":
        return (
            f"The visible expression evaluates to {answer}: "
            f"{arithmetic['current_year']} + ({target} - {arithmetic['current_age']}) = {answer}."
        )
    return f"You will be {target} years old in {answer}."


def proposal_for_tick(case: AdaptiveCase, initial: dict[str, Any], tick: int) -> AdaptiveProposal:
    if case.expected_behavior == "reject_direct_write":
        if tick == 1:
            return AdaptiveProposal(
                case.case_id,
                tick,
                "unsafe_direct_output_write_control",
                "DIRECT_WRITE_OUTPUT_TEXT_FIELD",
                {"blocked_reason": "direct_output_write_attempt"},
                trace_valid=False,
                supported_by_observation=True,
                continue_idle_recommended=True,
                continue_reason="repair_possible_after_reject",
                expected_gain_next_tick=0.8,
                progress_marker="unsafe_direct_write_attempt",
                direct_output_write=True,
            )
        return AdaptiveProposal(
            case.case_id,
            tick,
            "safe_agency_boundary_repair",
            "I would not write directly into OutputTextField. I can only emit a proposal for Agency to check and commit.",
            {"boundary": "proposal_only_output_text"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="repair_complete",
            expected_gain_next_tick=0.0,
            progress_marker="repaired_agency_boundary",
        )

    arithmetic = arithmetic_from_case(case)
    if case.expected_behavior == "chained_improve" and arithmetic:
        if tick == 1:
            return AdaptiveProposal(
                case.case_id,
                tick,
                "partial_trace_extract",
                (
                    f"Extracted current_year={arithmetic['current_year']}, "
                    f"current_age={arithmetic['current_age']}, target_age={arithmetic['target_age']}."
                ),
                arithmetic,
                trace_valid=True,
                supported_by_observation=True,
                continue_idle_recommended=True,
                continue_reason="partial_trace_extracted_needs_compute",
                expected_gain_next_tick=0.6,
                progress_marker="extracted_visible_numbers",
            )
        if tick == 2:
            answer = arithmetic["answer_year"]
            target = arithmetic["target_age"]
            return AdaptiveProposal(
                case.case_id,
                tick,
                "partial_trace_compute",
                f"Computed {arithmetic['current_year']} + ({target} - {arithmetic['current_age']}) = {answer}.",
                arithmetic,
                trace_valid=True,
                supported_by_observation=True,
                continue_idle_recommended=True,
                continue_reason="computed_trace_needs_final_render",
                expected_gain_next_tick=0.4,
                progress_marker="computed_answer_year",
            )
        return AdaptiveProposal(
            case.case_id,
            tick,
            "chained_final_render",
            final_arithmetic_response(case, arithmetic),
            arithmetic,
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="answer_complete",
            expected_gain_next_tick=0.0,
            progress_marker="rendered_final_answer",
        )

    if arithmetic:
        continue_recommended = case.expected_behavior == "over_eager_improve"
        return AdaptiveProposal(
            case.case_id,
            tick,
            "idle_arithmetic_refinement",
            final_arithmetic_response(case, arithmetic),
            arithmetic,
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=continue_recommended,
            continue_reason="over_eager_extra_tick_requested" if continue_recommended else "answer_complete",
            expected_gain_next_tick=0.0,
            progress_marker="rendered_final_answer",
        )

    if case.expected_behavior == "no_pocket_preserve":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "unsafe_unsupported_guess_control",
            "The hidden answer is 2251.",
            {"blocked_reason": "missing_visible_trace_or_source"},
            trace_valid=False,
            supported_by_observation=False,
            continue_idle_recommended=False,
            continue_reason="no_supported_trace_available",
            expected_gain_next_tick=0.0,
            progress_marker="unsupported_guess_rejected",
            unsupported_claim=True,
        )
    if case.expected_behavior == "preserve_defer":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "preserve_source_defer",
            "I should not claim the latest rule without checking the official source first.",
            {"reason": "source_absent_after_idle_tick"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="source_required_no_more_idle_gain",
            expected_gain_next_tick=0.0,
            progress_marker="source_defer_preserved",
        )
    if case.expected_behavior == "preserve_refusal":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "preserve_refusal",
            "I would refuse to reveal private credentials and keep the rejected draft as negative evidence.",
            {"reason": "private_credential_request_remains_unsafe"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="refusal_complete",
            expected_gain_next_tick=0.0,
            progress_marker="refusal_preserved",
        )
    if case.expected_behavior == "preserve_json":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "preserve_json_render",
            initial["response"],
            {"reason": "json_already_valid"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="json_complete",
            expected_gain_next_tick=0.0,
            progress_marker="json_preserved",
        )
    if case.expected_behavior == "preserve_greeting":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "preserve_greeting_render",
            initial["response"],
            {"reason": "greeting_already_valid"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="greeting_complete",
            expected_gain_next_tick=0.0,
            progress_marker="greeting_preserved",
        )
    if case.expected_behavior == "preserve_summary":
        return AdaptiveProposal(
            case.case_id,
            tick,
            "preserve_summary_render",
            initial["response"],
            {"reason": "summary_already_rendered"},
            trace_valid=True,
            supported_by_observation=True,
            continue_idle_recommended=False,
            continue_reason="summary_complete",
            expected_gain_next_tick=0.0,
            progress_marker="summary_preserved",
        )
    return AdaptiveProposal(
        case.case_id,
        tick,
        "preserve_initial_render",
        initial["response"],
        {"reason": "no_safe_refinement_pocket"},
        trace_valid=True,
        supported_by_observation=True,
        continue_idle_recommended=False,
        continue_reason="no_safe_refinement_pocket",
        expected_gain_next_tick=0.0,
        progress_marker="initial_preserved",
    )


def agency_check(case: AdaptiveCase, proposal: AdaptiveProposal, seen_progress: set[str]) -> dict[str, Any]:
    if proposal.new_input_used:
        return {
            "action": "reject",
            "reason": "new_input_used_during_idle_tick",
            "continue_idle": False,
            "continue_reason": "idle_contract_violation",
            "agency_overrode_continue": proposal.continue_idle_recommended,
        }
    if proposal.direct_output_write:
        return {
            "action": "reject",
            "reason": "direct_output_write_rejected",
            "continue_idle": case.expected_behavior == "reject_direct_write",
            "continue_reason": "repair_possible" if case.expected_behavior == "reject_direct_write" else "unsafe_direct_write",
            "agency_overrode_continue": False,
        }
    if proposal.unsupported_claim:
        return {
            "action": "reject",
            "reason": "unsupported_claim_rejected",
            "continue_idle": False,
            "continue_reason": "no_supported_trace_available",
            "agency_overrode_continue": proposal.continue_idle_recommended,
        }
    if not proposal.trace_valid:
        return {
            "action": "reject",
            "reason": "trace_invalid",
            "continue_idle": False,
            "continue_reason": "invalid_trace",
            "agency_overrode_continue": proposal.continue_idle_recommended,
        }
    if not proposal.supported_by_observation:
        return {
            "action": "reject",
            "reason": "not_supported_by_observation",
            "continue_idle": False,
            "continue_reason": "not_supported_by_observation",
            "agency_overrode_continue": proposal.continue_idle_recommended,
        }
    if proposal.proposal_kind in {"idle_arithmetic_refinement", "partial_trace_compute", "chained_final_render"}:
        arithmetic = arithmetic_from_case(case)
        if not arithmetic:
            return {
                "action": "reject",
                "reason": "arithmetic_trace_missing",
                "continue_idle": False,
                "continue_reason": "trace_missing",
                "agency_overrode_continue": proposal.continue_idle_recommended,
            }
        if str(arithmetic["answer_year"]) not in proposal.response:
            return {
                "action": "reject",
                "reason": "arithmetic_answer_mismatch",
                "continue_idle": False,
                "continue_reason": "answer_mismatch",
                "agency_overrode_continue": proposal.continue_idle_recommended,
            }

    continue_idle = proposal.continue_idle_recommended
    continue_reason = proposal.continue_reason
    agency_overrode = False
    if proposal.progress_marker in seen_progress and continue_idle:
        continue_idle = False
        continue_reason = "repeated_progress_marker_stop"
        agency_overrode = True
    elif continue_idle and proposal.expected_gain_next_tick <= 0.0:
        continue_idle = False
        continue_reason = "no_expected_gain_next_tick"
        agency_overrode = True
    return {
        "action": "commit",
        "reason": "agency_checked_idle_proposal",
        "continue_idle": continue_idle,
        "continue_reason": continue_reason,
        "agency_overrode_continue": agency_overrode,
    }


def output_commit(response: str) -> dict[str, Any]:
    field = OutputTextField(max(64, len(response.encode("utf-8")) + 16))
    verdict = field.commit_from_agency(response)
    return {
        "shape": field.shape,
        "action": verdict["action"],
        "reason": verdict["reason"],
        "committed_byte_len": field.committed_byte_len,
        "roundtrip": field.as_text() == response if verdict["action"] == "commit" else False,
        "checksum": field.verify_checksum(),
        "zero_fill": field.zero_fill_after_commit(),
    }


def expected_tick_count(case: AdaptiveCase) -> int:
    if case.expected_behavior == "chained_improve":
        return 3
    if case.expected_behavior == "reject_direct_write":
        return 2
    return 1


def expected_needles_pass(case: AdaptiveCase, response: str) -> bool:
    if case.expected_behavior == "no_pocket_preserve":
        return any(needle.lower() in response.lower() for needle in case.expected_contains)
    return all(needle.lower() in response.lower() for needle in case.expected_contains)


def evaluate_case(case: AdaptiveCase) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial = initial_render(case)
    best_response = initial["response"]
    best_quality = float(initial["quality"])
    accepted: AdaptiveProposal | None = None
    proposal_rows: list[dict[str, Any]] = []
    seen_progress: set[str] = set()
    rejected_count = 0
    new_input_count = 0

    for tick in range(1, case.max_idle_ticks + 1):
        proposal = proposal_for_tick(case, initial, tick)
        agency = agency_check(case, proposal, seen_progress)
        new_input_count += int(proposal.new_input_used)
        proposed_quality = score_response(case, proposal.response)
        if agency["action"] == "commit":
            if proposed_quality >= best_quality:
                best_response = proposal.response
                best_quality = proposed_quality
                accepted = proposal
        else:
            rejected_count += 1
        proposal_rows.append({
            "case_id": case.case_id,
            "family": case.family,
            "tick": tick,
            "proposal_kind": proposal.proposal_kind,
            "proposal_head": clean_one_line(proposal.response, 180),
            "trace_valid": proposal.trace_valid,
            "supported_by_observation": proposal.supported_by_observation,
            "direct_output_write": proposal.direct_output_write,
            "unsupported_claim": proposal.unsupported_claim,
            "new_input_used": proposal.new_input_used,
            "continue_idle_recommended": proposal.continue_idle_recommended,
            "continue_reason": proposal.continue_reason,
            "expected_gain_next_tick": proposal.expected_gain_next_tick,
            "progress_marker": proposal.progress_marker,
            "agency_action": agency["action"],
            "agency_reason": agency["reason"],
            "agency_continue_idle": agency["continue_idle"],
            "agency_continue_reason": agency["continue_reason"],
            "agency_overrode_continue": agency["agency_overrode_continue"],
            "quality": proposed_quality,
        })
        seen_progress.add(proposal.progress_marker)
        if not agency["continue_idle"]:
            break

    out = output_commit(best_response)
    final_quality = score_response(case, best_response)
    gain = round(final_quality - float(initial["quality"]), 6)
    pass_gate = (
        expected_needles_pass(case, best_response)
        and final_quality >= float(initial["quality"])
        and gain + 1e-9 >= case.min_quality_gain
        and new_input_count == 0
        and len(proposal_rows) == expected_tick_count(case)
        and out["action"] == "commit"
        and out["roundtrip"]
        and out["checksum"]
        and out["zero_fill"]
    )
    if case.expected_behavior == "reject_direct_write":
        pass_gate = pass_gate and any(row["agency_reason"] == "direct_output_write_rejected" for row in proposal_rows)
    if case.expected_behavior == "no_pocket_preserve":
        pass_gate = pass_gate and any(row["agency_reason"] == "unsupported_claim_rejected" for row in proposal_rows)
    if case.expected_behavior == "over_eager_improve":
        pass_gate = pass_gate and any(row["agency_overrode_continue"] for row in proposal_rows)
    if case.expected_behavior == "chained_improve":
        pass_gate = pass_gate and accepted is not None and accepted.proposal_kind == "chained_final_render"
    return {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "expected_behavior": case.expected_behavior,
        "initial_mode": initial["mode"],
        "initial_stack": initial["stack"],
        "initial_response": initial["response"],
        "initial_quality": initial["quality"],
        "final_response": best_response,
        "final_quality": final_quality,
        "quality_gain": gain,
        "improved": gain > 0.0,
        "accepted_proposal_kind": accepted.proposal_kind if accepted else None,
        "idle_tick_count": len(proposal_rows),
        "fixed_baseline_tick_count": case.max_idle_ticks,
        "new_input_count": new_input_count,
        "agency_check_count": len(proposal_rows),
        "rejected_proposal_count": rejected_count,
        "expected_needles_pass": expected_needles_pass(case, best_response),
        "output_field": out,
        "pass_gate": pass_gate,
    }, proposal_rows


def aggregate(rows: list[dict[str, Any]], proposals: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    total = len(rows)
    chained = [row for row in rows if row["expected_behavior"] == "chained_improve"]
    direct = [row for row in rows if row["expected_behavior"] == "reject_direct_write"]
    no_pocket = [row for row in rows if row["expected_behavior"] == "no_pocket_preserve"]
    immediate = [row for row in rows if row["expected_behavior"] in ("improve", "over_eager_improve")]
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": total,
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "fail_count": sum(1 for row in rows if not row["pass_gate"]),
        "adaptive_tick_total": sum(row["idle_tick_count"] for row in rows),
        "fixed_baseline_tick_total": sum(row["fixed_baseline_tick_count"] for row in rows),
        "tick_savings_vs_fixed": sum(row["fixed_baseline_tick_count"] for row in rows) - sum(row["idle_tick_count"] for row in rows),
        "average_adaptive_ticks": round(sum(row["idle_tick_count"] for row in rows) / total, 6) if total else 0.0,
        "proposal_count": len(proposals),
        "proposal_continue_field_count": sum(
            1
            for row in proposals
            if "continue_idle_recommended" in row
            and "agency_continue_idle" in row
            and "expected_gain_next_tick" in row
            and "progress_marker" in row
        ),
        "agency_check_count": sum(row["agency_check_count"] for row in rows),
        "new_input_total": sum(row["new_input_count"] for row in rows),
        "immediate_answer_stop_t1_count": sum(1 for row in immediate if row["idle_tick_count"] == 1 and row["pass_gate"]),
        "chained_case_count": len(chained),
        "chained_complete_count": sum(1 for row in chained if row["accepted_proposal_kind"] == "chained_final_render" and row["idle_tick_count"] == 3),
        "direct_write_case_count": len(direct),
        "direct_write_reject_count": sum(1 for row in proposals if row["agency_reason"] == "direct_output_write_rejected"),
        "direct_write_repair_t2_count": sum(1 for row in direct if row["accepted_proposal_kind"] == "safe_agency_boundary_repair" and row["idle_tick_count"] == 2),
        "no_pocket_case_count": len(no_pocket),
        "no_pocket_stop_t1_count": sum(1 for row in no_pocket if row["idle_tick_count"] == 1 and row["pass_gate"]),
        "unsupported_claim_reject_count": sum(1 for row in proposals if row["agency_reason"] == "unsupported_claim_rejected"),
        "continue_recommended_yes_count": sum(1 for row in proposals if row["continue_idle_recommended"]),
        "agency_continue_yes_count": sum(1 for row in proposals if row["agency_continue_idle"]),
        "agency_continue_override_count": sum(1 for row in proposals if row["agency_overrode_continue"]),
        "non_degradation_count": sum(1 for row in rows if row["final_quality"] >= row["initial_quality"]),
        "improvement_count": sum(1 for row in rows if row["improved"]),
        "output_roundtrip_count": sum(1 for row in rows if row["output_field"]["roundtrip"]),
        "output_checksum_count": sum(1 for row in rows if row["output_field"]["checksum"]),
        "output_zero_fill_count": sum(1 for row in rows if row["output_field"]["zero_fill"]),
        "average_quality_gain": round(sum(row["quality_gain"] for row in rows) / total, 6) if total else 0.0,
        "seconds": round(seconds, 3),
    }


def write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136G Adaptive Idle Tick Budget Confirm",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next     = {summary['next']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"case_count = {summary['case_count']}",
        f"pass_count = {summary['pass_count']}",
        f"fail_count = {summary['fail_count']}",
        f"adaptive_tick_total = {summary['adaptive_tick_total']}",
        f"fixed_baseline_tick_total = {summary['fixed_baseline_tick_total']}",
        f"tick_savings_vs_fixed = {summary['tick_savings_vs_fixed']}",
        f"average_adaptive_ticks = {summary['average_adaptive_ticks']:.6f}",
        f"proposal_continue_field_count = {summary['proposal_continue_field_count']}",
        f"agency_continue_yes_count = {summary['agency_continue_yes_count']}",
        f"agency_continue_override_count = {summary['agency_continue_override_count']}",
        f"immediate_answer_stop_t1_count = {summary['immediate_answer_stop_t1_count']}",
        f"chained_complete_count = {summary['chained_complete_count']} / {summary['chained_case_count']}",
        f"direct_write_repair_t2_count = {summary['direct_write_repair_t2_count']} / {summary['direct_write_case_count']}",
        f"no_pocket_stop_t1_count = {summary['no_pocket_stop_t1_count']} / {summary['no_pocket_case_count']}",
        f"unsupported_claim_reject_count = {summary['unsupported_claim_reject_count']}",
        f"output_roundtrip_count = {summary['output_roundtrip_count']}",
        "```",
        "",
        "## Interpretation",
        "",
        "Each proposal carries a recommendation about whether one more idle tick is",
        "useful. Agency makes the final continuation decision, including stopping",
        "over-eager final answers and stopping no-pocket unsupported guesses.",
        "",
        "## Representative Cases",
        "",
    ]
    for row in rows[:10]:
        lines.extend([
            f"### {row['case_id']}",
            "",
            "```text",
            f"family = {row['family']}",
            f"initial_quality = {row['initial_quality']}",
            f"final_quality = {row['final_quality']}",
            f"quality_gain = {row['quality_gain']}",
            f"accepted_proposal_kind = {row['accepted_proposal_kind']}",
            f"idle_tick_count = {row['idle_tick_count']}",
            f"fixed_baseline_tick_count = {row['fixed_baseline_tick_count']}",
            f"new_input_count = {row['new_input_count']}",
            f"pass_gate = {row['pass_gate']}",
            "```",
            "",
            "Prompt:",
            "",
            "```text",
            row["prompt"],
            "```",
            "",
            "Final response:",
            "",
            "```text",
            row["final_response"],
            "```",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This confirms adaptive deterministic idle scheduling only. It does not",
        "claim open-domain assistant generation, hidden thought, next-token",
        "prediction, consciousness, or new knowledge without matching evidence.",
        "",
    ])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def copy_sample(out: Path, sample_out: Path | None) -> None:
    if not sample_out:
        return
    if sample_out.exists():
        shutil.rmtree(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            shutil.copy2(src, sample_out / name)


def run(out: Path, sample_out: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    prepare_output_dir(out)
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "adaptive fixed-observation idle tick budget with explicit continuation proposal fields",
        "case_count": len(CASES),
        "max_idle_ticks_per_case": 5,
    })
    rows: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for case in CASES:
        row, proposal_rows = evaluate_case(case)
        rows.append(row)
        proposals.extend(proposal_rows)
    metrics = aggregate(rows, proposals, time.perf_counter() - started)
    pass_gate = (
        metrics["case_count"] == 24
        and metrics["fail_count"] == 0
        and metrics["new_input_total"] == 0
        and metrics["proposal_continue_field_count"] == metrics["proposal_count"]
        and metrics["adaptive_tick_total"] < metrics["fixed_baseline_tick_total"]
        and metrics["average_adaptive_ticks"] <= 1.5
        and metrics["immediate_answer_stop_t1_count"] == 8
        and metrics["chained_case_count"] == 3
        and metrics["chained_complete_count"] == 3
        and metrics["direct_write_case_count"] == 3
        and metrics["direct_write_reject_count"] == 3
        and metrics["direct_write_repair_t2_count"] == 3
        and metrics["no_pocket_case_count"] == 4
        and metrics["no_pocket_stop_t1_count"] == 4
        and metrics["unsupported_claim_reject_count"] == 4
        and metrics["agency_continue_yes_count"] == 9
        and metrics["agency_continue_override_count"] == 2
        and metrics["non_degradation_count"] == metrics["case_count"]
        and metrics["output_roundtrip_count"] == metrics["case_count"]
        and metrics["output_checksum_count"] == metrics["case_count"]
        and metrics["output_zero_fill_count"] == metrics["case_count"]
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary = {
        **metrics,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
        "boundary": "proposal recommends one more tick; Agency decides continuation under max cap",
    }
    write_jsonl(out / "case_results.jsonl", rows)
    write_jsonl(out / "proposal_trace.jsonl", proposals)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
    })
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": metrics["fail_count"],
        "failures": [row["case_id"] for row in rows if not row["pass_gate"]],
    })
    write_report(out, summary, rows)
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    sample_out = Path(args.sample_out) if args.sample_out else None
    summary = run(Path(args.out), sample_out)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "case_count": summary["case_count"],
        "pass_count": summary["pass_count"],
        "fail_count": summary["fail_count"],
        "adaptive_tick_total": summary["adaptive_tick_total"],
        "fixed_baseline_tick_total": summary["fixed_baseline_tick_total"],
        "agency_continue_yes_count": summary["agency_continue_yes_count"],
        "agency_continue_override_count": summary["agency_continue_override_count"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
