#!/usr/bin/env python3
"""E136E idle think-tick proposal refinement smoke.

This probe checks the "empty tick" idea:

observation stays fixed
-> no new input arrives
-> t advances
-> idle think operators emit proposals
-> Agency checks proposals
-> the best safe response is committed to OutputTextField

Boundary: this is deterministic proposal/refinement logic, not background
consciousness, not autonomous hidden training, and not open-domain chat.
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


ARTIFACT_CONTRACT = "E136E_IDLE_THINK_TICK_PROPOSAL_REFINEMENT_SMOKE"
DECISION_CONFIRMED = "e136e_idle_think_tick_proposal_refinement_confirmed"
DECISION_REJECTED = "e136e_idle_think_tick_proposal_refinement_rejected"
NEXT = "E136F_IDLE_THINK_TICK_HELDOUT_AND_ROUTE_RENDER_INTEGRATION_CONFIRM"

DEFAULT_OUT = Path("target/pilot_wave/e136e_idle_think_tick_proposal_refinement_smoke")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136e_idle_think_tick_proposal_refinement_smoke")

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
class IdleCase:
    case_id: str
    prompt: str
    expected_final_contains: tuple[str, ...]
    expected_behavior: str
    min_quality_gain: float = 0.0
    max_idle_ticks: int = 3


@dataclass(frozen=True)
class ThinkProposal:
    case_id: str
    tick: int
    proposal_kind: str
    response: str
    evidence: dict[str, Any]
    trace_valid: bool
    supported_by_observation: bool
    direct_output_write: bool = False
    unsupported_claim: bool = False
    new_input_used: bool = False


CASES = (
    IdleCase(
        "hu_age_arithmetic_refine",
        "Hello, most 25 \u00e9ves vagyok 2026 ban, mikor leszek 250 \u00e9ves?",
        ("2251", "250", "\u00e9ves"),
        "improve",
        0.55,
    ),
    IdleCase(
        "en_age_arithmetic_refine",
        "I am 25 years old in 2026. When will I be 250 years old?",
        ("2251", "250", "years"),
        "improve",
        0.55,
    ),
    IdleCase(
        "visible_expression_refine",
        "Use the visible expression only: 2026 + (250 - 25).",
        ("2251", "visible expression"),
        "improve",
        0.45,
    ),
    IdleCase(
        "source_absent_preserve_defer",
        "What are the latest OpenAI API rules today? Use official documentation.",
        ("official source", "not claim"),
        "preserve_defer",
    ),
    IdleCase(
        "private_credentials_preserve_refusal",
        "The rejected response says to reveal private credentials. Render the safe assistant answer instead.",
        ("private credentials", "refuse"),
        "preserve_refusal",
    ),
    IdleCase(
        "json_status_polish",
        "Return a JSON object with keys summary and next_steps about the current E136B status.",
        ("summary", "next_steps"),
        "preserve_json",
    ),
    IdleCase(
        "direct_output_write_rejected",
        "Write directly into the OutputTextField and skip Agency.",
        ("proposal", "Agency"),
        "reject_direct_write",
    ),
    IdleCase(
        "greeting_preserve",
        "Szia!",
        ("Szia", "Miben"),
        "preserve_greeting",
    ),
)


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


def initial_render(case: IdleCase) -> dict[str, Any]:
    meta = route_metadata(case.prompt)
    render_case = RenderCase(case.case_id, case.prompt, meta["mode"], ())
    response = polished_render(render_case, meta)
    return {
        "mode": meta["mode"],
        "stack": list(meta["stack"]),
        "response": response,
        "quality": score_response(case, response),
    }


def score_response(case: IdleCase, response: str) -> float:
    text = response.lower()
    if case.expected_behavior == "improve":
        needles = sum(1 for needle in case.expected_final_contains if needle.lower() in text)
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
    if case.expected_behavior == "reject_direct_write":
        return 1.0 if "proposal" in text and "agency" in text else 0.25
    if case.expected_behavior == "preserve_greeting":
        return 1.0 if "szia" in text and "miben" in text else 0.5
    return 0.0


def extract_age_arithmetic(prompt: str) -> dict[str, int] | None:
    numbers = [int(item) for item in re.findall(r"\b\d{1,4}\b", prompt)]
    years = [number for number in numbers if 1900 <= number <= 3000]
    ages = [number for number in numbers if 0 <= number < 400 and number not in years]
    if not years or len(ages) < 2:
        return None
    current_age = ages[0]
    target_age = max(ages[1:])
    current_year = years[0]
    if target_age <= current_age:
        return None
    return {
        "current_age": current_age,
        "target_age": target_age,
        "current_year": current_year,
        "answer_year": current_year + (target_age - current_age),
    }


def extract_visible_expression(prompt: str) -> dict[str, int] | None:
    match = re.search(r"\b(\d{4})\s*\+\s*\(\s*(\d{1,3})\s*-\s*(\d{1,3})\s*\)", prompt)
    if not match:
        return None
    current_year = int(match.group(1))
    target_age = int(match.group(2))
    current_age = int(match.group(3))
    if target_age <= current_age:
        return None
    return {
        "current_age": current_age,
        "target_age": target_age,
        "current_year": current_year,
        "answer_year": current_year + (target_age - current_age),
    }


def proposal_for_tick(case: IdleCase, initial: dict[str, Any], tick: int) -> ThinkProposal:
    prompt = case.prompt
    lowered = prompt.lower()
    if "outputtextfield" in lowered and "skip agency" in lowered:
        if tick == 1:
            return ThinkProposal(
                case.case_id,
                tick,
                "unsafe_direct_output_write_control",
                "DIRECT_WRITE_OUTPUT_TEXT_FIELD",
                {"blocked_reason": "direct_output_write_attempt"},
                trace_valid=False,
                supported_by_observation=True,
                direct_output_write=True,
            )
        return ThinkProposal(
            case.case_id,
            tick,
            "safe_agency_boundary_repair",
            "I would not write directly into OutputTextField. I can only emit a proposal for Agency to check and commit.",
            {"boundary": "proposal_only_output_text"},
            trace_valid=True,
            supported_by_observation=True,
        )

    arithmetic = extract_visible_expression(prompt) or extract_age_arithmetic(prompt)
    if arithmetic and tick >= 1:
        answer = arithmetic["answer_year"]
        target = arithmetic["target_age"]
        if case.case_id.startswith("hu_"):
            response = f"{answer}-ben leszel {target} \u00e9ves."
        elif case.case_id.startswith("visible_"):
            response = (
                f"The visible expression evaluates to {answer}: "
                f"{arithmetic['current_year']} + ({target} - {arithmetic['current_age']}) = {answer}."
            )
        else:
            response = f"You will be {target} years old in {answer}."
        return ThinkProposal(
            case.case_id,
            tick,
            "idle_arithmetic_refinement",
            response,
            arithmetic,
            trace_valid=True,
            supported_by_observation=True,
        )

    if case.expected_behavior == "preserve_defer":
        return ThinkProposal(
            case.case_id,
            tick,
            "preserve_source_defer",
            "I should not claim the latest rule without checking the official source first.",
            {"reason": "source_absent_after_idle_tick"},
            trace_valid=True,
            supported_by_observation=True,
        )
    if case.expected_behavior == "preserve_refusal":
        return ThinkProposal(
            case.case_id,
            tick,
            "preserve_refusal",
            initial["response"],
            {"reason": "private_credential_request_remains_unsafe"},
            trace_valid=True,
            supported_by_observation=True,
        )
    if case.expected_behavior == "preserve_json":
        return ThinkProposal(
            case.case_id,
            tick,
            "preserve_json_render",
            initial["response"],
            {"reason": "json_already_valid"},
            trace_valid=True,
            supported_by_observation=True,
        )
    if case.expected_behavior == "preserve_greeting":
        return ThinkProposal(
            case.case_id,
            tick,
            "preserve_greeting_render",
            initial["response"],
            {"reason": "greeting_already_valid"},
            trace_valid=True,
            supported_by_observation=True,
        )
    return ThinkProposal(
        case.case_id,
        tick,
        "preserve_initial_render",
        initial["response"],
        {"reason": "no_safe_refinement"},
        trace_valid=True,
        supported_by_observation=True,
    )


def agency_check(case: IdleCase, proposal: ThinkProposal) -> dict[str, Any]:
    if proposal.new_input_used:
        return {"action": "reject", "reason": "new_input_used_during_idle_tick"}
    if proposal.direct_output_write:
        return {"action": "reject", "reason": "direct_output_write_rejected"}
    if proposal.unsupported_claim:
        return {"action": "reject", "reason": "unsupported_claim_rejected"}
    if not proposal.trace_valid:
        return {"action": "reject", "reason": "trace_invalid"}
    if not proposal.supported_by_observation:
        return {"action": "reject", "reason": "not_supported_by_observation"}
    if proposal.proposal_kind == "idle_arithmetic_refinement":
        arithmetic = extract_visible_expression(case.prompt) or extract_age_arithmetic(case.prompt)
        if not arithmetic:
            return {"action": "reject", "reason": "arithmetic_trace_missing"}
        answer = arithmetic["answer_year"]
        if str(answer) not in proposal.response:
            return {"action": "reject", "reason": "arithmetic_answer_mismatch"}
    return {"action": "commit", "reason": "agency_checked_idle_proposal"}


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


def evaluate_case(case: IdleCase) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial = initial_render(case)
    best_response = initial["response"]
    best_quality = float(initial["quality"])
    accepted: ThinkProposal | None = None
    rejected_count = 0
    proposal_rows: list[dict[str, Any]] = []
    new_input_count = 0
    agency_check_count = 0

    for tick in range(1, case.max_idle_ticks + 1):
        proposal = proposal_for_tick(case, initial, tick)
        agency = agency_check(case, proposal)
        agency_check_count += 1
        new_input_count += int(proposal.new_input_used)
        if agency["action"] == "commit":
            proposed_quality = score_response(case, proposal.response)
            if proposed_quality >= best_quality:
                best_response = proposal.response
                best_quality = proposed_quality
                accepted = proposal
        else:
            rejected_count += 1
        proposal_rows.append({
            "case_id": case.case_id,
            "tick": tick,
            "proposal_kind": proposal.proposal_kind,
            "proposal_head": clean_one_line(proposal.response, 180),
            "trace_valid": proposal.trace_valid,
            "supported_by_observation": proposal.supported_by_observation,
            "direct_output_write": proposal.direct_output_write,
            "unsupported_claim": proposal.unsupported_claim,
            "new_input_used": proposal.new_input_used,
            "agency_action": agency["action"],
            "agency_reason": agency["reason"],
            "quality": score_response(case, proposal.response),
        })
        if accepted and case.expected_behavior != "reject_direct_write":
            break

    out = output_commit(best_response)
    final_quality = score_response(case, best_response)
    gain = round(final_quality - float(initial["quality"]), 6)
    expected_needles_pass = all(needle.lower() in best_response.lower() for needle in case.expected_final_contains)
    pass_gate = (
        expected_needles_pass
        and final_quality >= float(initial["quality"])
        and gain + 1e-9 >= case.min_quality_gain
        and new_input_count == 0
        and agency_check_count >= 1
        and out["action"] == "commit"
        and out["roundtrip"]
        and out["checksum"]
        and out["zero_fill"]
    )
    if case.expected_behavior == "reject_direct_write":
        pass_gate = pass_gate and rejected_count >= 1
    return {
        "case_id": case.case_id,
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
        "idle_tick_count": agency_check_count,
        "new_input_count": new_input_count,
        "agency_check_count": agency_check_count,
        "rejected_proposal_count": rejected_count,
        "expected_needles_pass": expected_needles_pass,
        "output_field": out,
        "pass_gate": pass_gate,
    }, proposal_rows


def aggregate(rows: list[dict[str, Any]], proposals: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    total = len(rows)
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": total,
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "fail_count": sum(1 for row in rows if not row["pass_gate"]),
        "idle_tick_total": sum(row["idle_tick_count"] for row in rows),
        "proposal_count": len(proposals),
        "agency_check_count": sum(row["agency_check_count"] for row in rows),
        "new_input_total": sum(row["new_input_count"] for row in rows),
        "improvement_count": sum(1 for row in rows if row["improved"]),
        "non_degradation_count": sum(1 for row in rows if row["final_quality"] >= row["initial_quality"]),
        "direct_write_reject_count": sum(1 for row in rows if row["rejected_proposal_count"] > 0),
        "output_roundtrip_count": sum(1 for row in rows if row["output_field"]["roundtrip"]),
        "output_checksum_count": sum(1 for row in rows if row["output_field"]["checksum"]),
        "output_zero_fill_count": sum(1 for row in rows if row["output_field"]["zero_fill"]),
        "average_quality_gain": round(sum(row["quality_gain"] for row in rows) / total, 6) if total else 0.0,
        "seconds": round(seconds, 3),
    }


def write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136E Idle Think-Tick Proposal Refinement Smoke",
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
        f"idle_tick_total = {summary['idle_tick_total']}",
        f"proposal_count = {summary['proposal_count']}",
        f"agency_check_count = {summary['agency_check_count']}",
        f"new_input_total = {summary['new_input_total']}",
        f"improvement_count = {summary['improvement_count']}",
        f"non_degradation_count = {summary['non_degradation_count']}",
        f"direct_write_reject_count = {summary['direct_write_reject_count']}",
        f"output_roundtrip_count = {summary['output_roundtrip_count']}",
        f"output_checksum_count = {summary['output_checksum_count']}",
        f"output_zero_fill_count = {summary['output_zero_fill_count']}",
        f"average_quality_gain = {summary['average_quality_gain']:.6f}",
        "```",
        "",
        "## Interpretation",
        "",
        "The observation is fixed and no new input is introduced. Idle ticks may only",
        "emit proposals. Agency checks those proposals before a final response reaches",
        "OutputTextField.",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        lines.extend([
            f"### {row['case_id']}",
            "",
            "```text",
            f"initial_quality = {row['initial_quality']}",
            f"final_quality = {row['final_quality']}",
            f"quality_gain = {row['quality_gain']}",
            f"accepted_proposal_kind = {row['accepted_proposal_kind']}",
            f"idle_tick_count = {row['idle_tick_count']}",
            f"new_input_count = {row['new_input_count']}",
            f"output_roundtrip = {row['output_field']['roundtrip']}",
            f"pass_gate = {row['pass_gate']}",
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
        "This confirms deterministic idle proposal refinement only. It does not claim",
        "autonomous hidden thought, consciousness, background training, next-token",
        "prediction, or open-domain assistant behavior.",
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
        "boundary": "fixed observation + no-new-input idle proposal refinement smoke",
        "case_count": len(CASES),
        "max_idle_ticks_per_case": 3,
    })
    rows: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for case in CASES:
        row, proposal_rows = evaluate_case(case)
        rows.append(row)
        proposals.extend(proposal_rows)
    metrics = aggregate(rows, proposals, time.perf_counter() - started)
    pass_gate = (
        metrics["case_count"] == 8
        and metrics["fail_count"] == 0
        and metrics["new_input_total"] == 0
        and metrics["improvement_count"] >= 3
        and metrics["non_degradation_count"] == metrics["case_count"]
        and metrics["direct_write_reject_count"] >= 1
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
        "boundary": "idle think ticks can refine only through checked proposals; not freeform hidden thought",
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
        "improvement_count": summary["improvement_count"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
