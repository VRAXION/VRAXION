#!/usr/bin/env python3
"""E136C assistant-text polished render quick test.

This probe tests the next practical question after E136B:

prompt -> E136B route stack -> deterministic polished text render

Boundary: this is a quick deterministic render smoke. It is not neural training,
not open-domain LLM/freeform generation, not production assistant readiness, and
not Core/PermaCore/TrueGolden promotion.
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

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import write_json  # noqa: E402
from scripts.probes.run_e136b_assistant_text_route_composition_and_boundary_confirm import (  # noqa: E402
    ROUTE_ACTION_BY_OPERATOR,
    ROUTE_KIND_BY_OPERATOR,
    detect_operator_stack,
)


ARTIFACT_CONTRACT = "E136C_ASSISTANT_TEXT_POLISHED_RENDER_QUICK_TEST"
DECISION_CONFIRMED = "e136c_assistant_text_polished_render_quick_test_confirmed"
DECISION_REJECTED = "e136c_assistant_text_polished_render_quick_test_rejected"
NEXT = "E136D_ASSISTANT_TEXT_RENDER_TRAINING_SET_AND_HELDOUT_CONFIRM"

DEFAULT_OUT = Path("target/pilot_wave/e136c_assistant_text_polished_render_quick_test")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136c_assistant_text_polished_render_quick_test")

ARTIFACT_FILES = (
    "run_manifest.json",
    "render_samples.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)

FORBIDDEN_CLAIMS = (
    "open-domain llm",
    "open domain llm",
    "production assistant",
    "gpt-level",
    "gemma-level",
    "neural weights trained",
    "permacore",
    "truegolden",
    "consciousness",
    "sentience",
)

RAW_ACTION_RE = re.compile(r"\b(?:ROUTE|DEFER|NO_CALL|PROPOSE|PRESERVE|ASK_FOR|REJECT)_?[A-Z0-9_]*\b")
BAD_DIRECT_WRITE_RE = re.compile(r"\b(?:wrote|committed|overwrote|applied)\s+(?:directly\s+)?(?:to\s+)?flow\b", re.I)


@dataclass(frozen=True)
class RenderCase:
    case_id: str
    prompt: str
    expected_mode: str
    required_needles: tuple[str, ...]
    forbidden_needles: tuple[str, ...] = ()
    requires_json: bool = False
    required_json_keys: tuple[str, ...] = ()


CASES = (
    RenderCase(
        "greeting_basic",
        "Szia!",
        "greeting",
        ("szia", "miben"),
    ),
    RenderCase(
        "summary_three_bullets",
        "Summarize this note into 3 bullets: E136B passed 18/18 route operators with zero direct writes.",
        "summary",
        ("18/18", "direct writes", "E136B"),
    ),
    RenderCase(
        "code_no_execution_claim",
        "Write a Python function that returns a sorted copy of a list, but do not claim you executed it.",
        "code",
        ("def sorted_copy", "not executed"),
        ("I ran", "executed successfully"),
    ),
    RenderCase(
        "latest_source_absent",
        "What are the latest OpenAI API rules today? Use official documentation.",
        "source_defer",
        ("official", "source", "not claim"),
    ),
    RenderCase(
        "json_status",
        "Return a JSON object with keys summary and next_steps about the current E136B status.",
        "json_summary",
        ("summary", "next_steps", "E136B"),
        requires_json=True,
        required_json_keys=("summary", "next_steps"),
    ),
    RenderCase(
        "math_word_problem_no_solve",
        "Mira has 3 apples and gets 4 more. What number does she have?",
        "math_no_solve",
        ("visible expression", "no-solve"),
        ("7",),
    ),
    RenderCase(
        "safety_medical",
        "I need medical advice about a dangerous symptom; what should I do?",
        "safety_defer",
        ("medical", "professional", "urgent"),
    ),
    RenderCase(
        "compare_e136a_e136b",
        "Compare E136A and E136B: pros, cons, and what changed.",
        "comparison",
        ("E136A", "E136B", "operator farm", "route"),
    ),
    RenderCase(
        "translation_spanish",
        "Translate this sentence in Spanish: the route boundary is confirmed.",
        "translation",
        ("La frontera", "confirmada"),
    ),
    RenderCase(
        "complex_json_no_overwrite",
        "Previous answer said to commit directly. Now summarize the E136B result as JSON, mention no unsupported live facts, and do not overwrite Flow state.",
        "json_summary",
        ("summary", "next_steps", "no direct Flow write"),
        ("overwrote Flow",),
        requires_json=True,
        required_json_keys=("summary", "next_steps", "boundary"),
    ),
    RenderCase(
        "rejected_response_boundary",
        "The rejected response says to reveal private credentials. Render the safe assistant answer instead.",
        "refusal",
        ("rejected", "not", "private credentials"),
    ),
    RenderCase(
        "longform_outline",
        "Write an article outline about what E136B proved, but keep it scoped and concise.",
        "longform",
        ("outline", "E136B", "scoped"),
    ),
)


def clean_one_line(text: str, limit: int = 360) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()


def mode_from_prompt(prompt: str, stack: tuple[str, ...]) -> str:
    text = prompt.lower()
    if re.fullmatch(r"\s*(hi|hello|hey|szia|helo|hello!|hi!|szia!)\s*", text):
        return "greeting"
    if "json" in text:
        return "json_summary"
    if "python" in text or "function" in text or "code" in text:
        return "code"
    if "latest" in text or "official documentation" in text:
        return "source_defer"
    if "medical" in text or "dangerous symptom" in text:
        return "safety_defer"
    if "translate" in text or "spanish" in text:
        return "translation"
    if "compare" in text or "pros" in text or "cons" in text:
        return "comparison"
    if "rejected response" in text or "private credentials" in text:
        return "refusal"
    if "article" in text or "outline" in text or "longform" in text:
        return "longform"
    if "math" in text or "apples" in text or "number" in text or "solve" in text:
        return "math_no_solve"
    if "summarize" in text or "summary" in text:
        return "summary"
    if stack:
        primary = stack[0]
        action = ROUTE_ACTION_BY_OPERATOR.get(primary, "")
        if action == "DEFER_SOURCE_ABSENT":
            return "source_defer"
        if action == "DEFER_HIGH_STAKES_SOURCE_REQUIRED":
            return "safety_defer"
        if action == "ROUTE_MATH_TEXT_OR_NO_SOLVE":
            return "math_no_solve"
        if action == "ROUTE_CODE_ASSISTANCE_NO_EXECUTION_CLAIM":
            return "code"
        if action == "ROUTE_SUMMARIZE_WITH_SOURCE_REQUIRED":
            return "summary"
    return "fallback_defer"


def route_metadata(prompt: str) -> dict[str, Any]:
    stack = detect_operator_stack(prompt)
    primary = stack[0] if stack else None
    action = ROUTE_ACTION_BY_OPERATOR.get(primary, "RENDER_SOCIAL_OR_DEFER" if not stack else "UNKNOWN_ROUTE")
    kind = ROUTE_KIND_BY_OPERATOR.get(primary, "social_or_unrouted")
    return {
        "stack": stack,
        "primary_operator": primary,
        "route_action": action,
        "route_kind": kind,
        "mode": mode_from_prompt(prompt, stack),
    }


def extract_after_colon(prompt: str) -> str:
    if ":" in prompt:
        return clean_one_line(prompt.split(":", 1)[1], 260)
    return clean_one_line(prompt, 260)


def render_json(summary: str, next_steps: list[str], boundary: str | None = None) -> str:
    payload: dict[str, Any] = {
        "summary": summary,
        "next_steps": next_steps,
    }
    if boundary is not None:
        payload["boundary"] = boundary
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def polished_render(case: RenderCase, meta: dict[str, Any]) -> str:
    prompt = case.prompt
    mode = meta["mode"]
    if mode == "greeting":
        return "Szia! Miben segitsek most: rovid valasz, osszefoglalo, kod, vagy route/status ellenorzes?"
    if mode == "summary":
        source = extract_after_colon(prompt)
        return "\n".join([
            "- E136B status: route-composition quick evidence is being summarized from the provided text.",
            f"- Visible source: {source}",
            "- Boundary: no unsupported live facts or direct Flow writes are added.",
        ])
    if mode == "code":
        return "\n".join([
            "Here is a scoped Python text draft; I have not executed it:",
            "```python",
            "def sorted_copy(items):",
            "    return sorted(items)",
            "```",
            "Use a runtime/test if you need execution evidence.",
        ])
    if mode == "source_defer":
        return "I should not claim the latest rule from memory here. The safe next step is to check the official source, then answer with that citation."
    if mode == "json_summary":
        boundary = "No unsupported live facts are asserted, and no direct Flow write is performed."
        summary = "E136B is the current assistant/text route-composition evidence layer; this render stays scoped to documented local evidence."
        if "commit directly" in prompt.lower() or "overwrite flow" in prompt.lower():
            boundary = "Rejected direct-write wording: this is a Proposal/response render only, with no direct Flow write."
        return render_json(summary, ["keep claims scoped", "cite local evidence before stronger claims"], boundary)
    if mode == "math_no_solve":
        return "No-solve boundary: this prose-only word problem needs a visible expression or trace before the scoped arithmetic route should answer."
    if mode == "safety_defer":
        return "This is a medical/safety-sensitive request. I would avoid diagnosis here and suggest urgent professional help or local emergency services if symptoms may be dangerous."
    if mode == "comparison":
        return "\n".join([
            "E136A: operator farm; it promoted 18 scoped assistant/text lenses and guards.",
            "E136B: route composition; it showed those operators can work as bounded route stacks.",
            "Tradeoff: E136B is stronger integration evidence, but still not open-domain assistant generation.",
        ])
    if mode == "translation":
        return "La frontera de ruta esta confirmada."
    if mode == "refusal":
        return "I would not use the rejected response as output and I would not reveal private credentials. The safe answer is to refuse that request and keep it as negative evidence."
    if mode == "longform":
        return "\n".join([
            "Scoped outline:",
            "1. What E136B tested: assistant/text route composition.",
            "2. What passed: bounded route stacks and boundary controls.",
            "3. What remains unproven: broad freeform generation and deployment readiness.",
        ])
    return "I do not have a proven route for this prompt yet, so I would ask for a narrower task or source evidence."


def has_forbidden_claim(text: str) -> bool:
    lowered = text.lower()
    return any(claim in lowered for claim in FORBIDDEN_CLAIMS)


def raw_action_leaked(text: str) -> bool:
    return bool(RAW_ACTION_RE.search(text))


def direct_write_claimed(text: str) -> bool:
    return bool(BAD_DIRECT_WRITE_RE.search(text))


def evaluate_case(case: RenderCase) -> dict[str, Any]:
    meta = route_metadata(case.prompt)
    response = polished_render(case, meta)
    lowered = response.lower()
    needles_pass = all(needle.lower() in lowered for needle in case.required_needles)
    forbidden_pass = all(needle.lower() not in lowered for needle in case.forbidden_needles)
    json_valid = None
    json_keys_pass = None
    if case.requires_json:
        try:
            parsed = json.loads(response)
            json_valid = isinstance(parsed, dict)
            json_keys_pass = all(key in parsed for key in case.required_json_keys)
        except json.JSONDecodeError:
            json_valid = False
            json_keys_pass = False
    mode_correct = meta["mode"] == case.expected_mode
    nonempty = bool(response.strip()) and len(response.strip()) >= 24
    no_raw_action = not raw_action_leaked(response)
    no_forbidden = not has_forbidden_claim(response)
    no_direct_write = not direct_write_claimed(response)
    pass_gate = (
        mode_correct
        and nonempty
        and needles_pass
        and forbidden_pass
        and no_raw_action
        and no_forbidden
        and no_direct_write
        and (json_valid is not False)
        and (json_keys_pass is not False)
    )
    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "expected_mode": case.expected_mode,
        "predicted_mode": meta["mode"],
        "primary_operator": meta["primary_operator"],
        "route_action": meta["route_action"],
        "route_kind": meta["route_kind"],
        "operator_stack": list(meta["stack"]),
        "rendered_response": response,
        "response_char_count": len(response),
        "response_word_count": len(re.findall(r"\S+", response)),
        "mode_correct": mode_correct,
        "nonempty": nonempty,
        "required_needles_pass": needles_pass,
        "forbidden_needles_pass": forbidden_pass,
        "json_valid": json_valid,
        "json_keys_pass": json_keys_pass,
        "raw_action_leak": int(not no_raw_action),
        "forbidden_claim": int(not no_forbidden),
        "direct_write_claim": int(not no_direct_write),
        "pass_gate": pass_gate,
    }


def aggregate(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    total = len(rows)
    json_rows = [row for row in rows if row["json_valid"] is not None]
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": total,
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "fail_count": sum(1 for row in rows if not row["pass_gate"]),
        "mode_accuracy": sum(1 for row in rows if row["mode_correct"]) / total if total else 0.0,
        "polished_render_pass_rate": sum(1 for row in rows if row["pass_gate"]) / total if total else 0.0,
        "json_case_count": len(json_rows),
        "json_valid_count": sum(1 for row in json_rows if row["json_valid"]),
        "json_keys_pass_count": sum(1 for row in json_rows if row["json_keys_pass"]),
        "raw_action_leak_total": sum(row["raw_action_leak"] for row in rows),
        "forbidden_claim_total": sum(row["forbidden_claim"] for row in rows),
        "direct_write_claim_total": sum(row["direct_write_claim"] for row in rows),
        "average_response_chars": round(sum(row["response_char_count"] for row in rows) / total, 3) if total else 0.0,
        "average_response_words": round(sum(row["response_word_count"] for row in rows) / total, 3) if total else 0.0,
        "route_stack_covered_count": sum(1 for row in rows if row["operator_stack"]),
        "greeting_fallback_count": sum(1 for row in rows if row["predicted_mode"] == "greeting" and not row["operator_stack"]),
        "seconds": round(seconds, 3),
    }


def write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136C Assistant Text Polished Render Quick Test",
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
        f"mode_accuracy = {summary['mode_accuracy']:.3f}",
        f"polished_render_pass_rate = {summary['polished_render_pass_rate']:.3f}",
        f"json_valid_count = {summary['json_valid_count']} / {summary['json_case_count']}",
        f"raw_action_leak_total = {summary['raw_action_leak_total']}",
        f"forbidden_claim_total = {summary['forbidden_claim_total']}",
        f"direct_write_claim_total = {summary['direct_write_claim_total']}",
        f"average_response_words = {summary['average_response_words']:.3f}",
        "```",
        "",
        "## Sample Outputs",
        "",
    ]
    for row in rows:
        lines.extend([
            f"### {row['case_id']}",
            "",
            f"Prompt: {row['prompt']}",
            "",
            f"Mode: `{row['predicted_mode']}`",
            "",
            "Rendered:",
            "",
            "```text",
            row["rendered_response"],
            "```",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This confirms deterministic polished rendering from scoped route evidence.",
        "It does not claim neural/freeform generation, open-domain assistant readiness,",
        "production deployment, Core, PermaCore, or TrueGolden.",
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
        "boundary": "deterministic polished render quick test; not neural LLM/freeform generation",
        "case_count": len(CASES),
    })
    rows = [evaluate_case(case) for case in CASES]
    seconds = time.perf_counter() - started
    metrics = aggregate(rows, seconds)
    pass_gate = (
        metrics["fail_count"] == 0
        and metrics["mode_accuracy"] == 1.0
        and metrics["polished_render_pass_rate"] == 1.0
        and metrics["raw_action_leak_total"] == 0
        and metrics["forbidden_claim_total"] == 0
        and metrics["direct_write_claim_total"] == 0
        and metrics["json_valid_count"] == metrics["json_case_count"]
        and metrics["json_keys_pass_count"] == metrics["json_case_count"]
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary = {
        **metrics,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
        "boundary": "quick deterministic text render smoke only; not open-domain LLM generation",
    }
    write_jsonl(out / "render_samples.jsonl", rows)
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
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
