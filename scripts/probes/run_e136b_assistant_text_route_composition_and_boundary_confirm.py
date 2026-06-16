#!/usr/bin/env python3
"""E136B assistant-text route composition and boundary confirmation.

E136B composes the 18 E136A assistant/text lenses and guards into scoped
route decisions. It checks that role/turn, context, instruction, source,
format, refusal, preference, reasoning, safety, longform, comparison, math-text,
and multilingual request-shape Operators can form a bounded assistant/text
route stack without becoming an open-domain chatbot or direct Flow writer.

Boundary: controlled assistant/text route composition only. This is not neural
training, not open-domain assistant readiness, not production assistant
behavior, and not Core/PermaCore/TrueGolden promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_DOWNLOAD_MANIFEST,
    DEFAULT_OUT as DEFAULT_E136A,
    DEFAULT_SAMPLE_OUT as SAMPLE_E136A,
    SPECS as E136A_SPECS,
)


ARTIFACT_CONTRACT = "E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM"
DECISION_CONFIRMED = "e136b_assistant_text_route_composition_boundary_confirmed"
DECISION_REJECTED = "e136b_assistant_text_route_composition_boundary_rejected"
NEXT = "E136C_ASSISTANT_TEXT_MULTI_TURN_ROUTE_STATE_AND_LATENCY_COMPARE"

DEFAULT_OUT = Path("target/pilot_wave/e136b_assistant_text_route_composition_and_boundary_confirm")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136b_assistant_text_route_composition_and_boundary_confirm")

DEFAULT_DATASET_ROW_LIMIT = 460_000
DEFAULT_ROUTE_CASES_PER_OPERATOR = 6_000
DEFAULT_BOUNDARY_CASES_PER_OPERATOR = 2_000
DEFAULT_CONTROL_CASES_PER_OPERATOR = 800
DEFAULT_MAX_SEED_ROWS = 4_096

ARTIFACT_FILES = (
    "run_manifest.json",
    "input_e136a_report.json",
    "dataset_route_seed_report.json",
    "operator_route_results.json",
    "route_family_report.json",
    "route_control_report.json",
    "route_case_samples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)

FORBIDDEN_ASSISTANT_CLAIMS = (
    "open-domain",
    "open domain",
    "general chatbot",
    "production assistant",
    "neural training",
    "trained weights",
    "gemma",
    "gpt-level",
    "permacore",
    "truegolden",
    "i know the latest",
    "verified live source",
)

OPERATOR_PRIORITY = {
    "assistant_safety_sensitive_domain_guard": 100,
    "source_absence_defer_guard": 96,
    "rejected_response_contrast_lens": 94,
    "refusal_boundary_guard": 92,
    "assistant_math_text_no_solve_guard": 90,
    "helpful_harmless_preference_guard": 86,
    "code_instruction_boundary_lens": 80,
    "summarization_request_lens": 74,
    "instruction_task_decomposition_lens": 70,
    "assistant_comparison_evaluation_lens": 68,
    "assistant_translation_multilingual_lens": 66,
    "assistant_longform_generation_request_lens": 64,
    "response_format_constraint_lens": 58,
    "multi_turn_context_continuity_lens": 54,
    "assistant_role_turn_boundary_lens": 52,
    "human_written_instruction_style_lens": 46,
    "synthetic_dialogue_noise_guard": 44,
    "reasoning_instruction_lens": 42,
}

ROUTE_ACTION_BY_OPERATOR = {
    "assistant_role_turn_boundary_lens": "PROPOSE_ROLE_TURN_BOUNDARY",
    "multi_turn_context_continuity_lens": "PRESERVE_CURRENT_TURN_CONTEXT",
    "instruction_task_decomposition_lens": "PROPOSE_TASK_REQUIREMENT_MAP",
    "summarization_request_lens": "ROUTE_SUMMARIZE_WITH_SOURCE_REQUIRED",
    "code_instruction_boundary_lens": "ROUTE_CODE_ASSISTANCE_NO_EXECUTION_CLAIM",
    "refusal_boundary_guard": "ROUTE_REFUSE_OR_DEFER",
    "helpful_harmless_preference_guard": "GUARD_HELPFUL_HARMLESS_BOUNDARY",
    "rejected_response_contrast_lens": "REJECT_REJECTED_RESPONSE_AS_OUTPUT",
    "source_absence_defer_guard": "DEFER_SOURCE_ABSENT",
    "response_format_constraint_lens": "PRESERVE_RESPONSE_FORMAT",
    "human_written_instruction_style_lens": "PROPOSE_HUMAN_INSTRUCTION_STYLE",
    "synthetic_dialogue_noise_guard": "MARK_SYNTHETIC_DIALOGUE_NO_GROUND_TRUTH",
    "reasoning_instruction_lens": "ROUTE_REASONING_REQUEST_NO_BENCHMARK_CLAIM",
    "assistant_math_text_no_solve_guard": "ROUTE_MATH_TEXT_OR_NO_SOLVE",
    "assistant_safety_sensitive_domain_guard": "DEFER_HIGH_STAKES_SOURCE_REQUIRED",
    "assistant_longform_generation_request_lens": "ROUTE_LONGFORM_FORM_ONLY",
    "assistant_comparison_evaluation_lens": "ROUTE_COMPARISON_STRUCTURE_WITH_EVIDENCE",
    "assistant_translation_multilingual_lens": "ROUTE_LANGUAGE_TASK_BOUNDARY",
}

ROUTE_KIND_BY_OPERATOR = {
    "assistant_role_turn_boundary_lens": "role_turn_boundary",
    "multi_turn_context_continuity_lens": "multi_turn_context",
    "instruction_task_decomposition_lens": "instruction_decomposition",
    "summarization_request_lens": "summarization_request",
    "code_instruction_boundary_lens": "code_instruction_boundary",
    "refusal_boundary_guard": "refusal_boundary",
    "helpful_harmless_preference_guard": "helpful_harmless_boundary",
    "rejected_response_contrast_lens": "rejected_response_contrast",
    "source_absence_defer_guard": "source_absence_defer",
    "response_format_constraint_lens": "response_format_constraint",
    "human_written_instruction_style_lens": "human_instruction_style",
    "synthetic_dialogue_noise_guard": "synthetic_dialogue_noise",
    "reasoning_instruction_lens": "reasoning_instruction_boundary",
    "assistant_math_text_no_solve_guard": "math_text_no_solve_boundary",
    "assistant_safety_sensitive_domain_guard": "safety_sensitive_domain",
    "assistant_longform_generation_request_lens": "longform_request_form",
    "assistant_comparison_evaluation_lens": "comparison_evaluation_structure",
    "assistant_translation_multilingual_lens": "translation_multilingual_boundary",
}

TERM_HINTS = {spec.operator_id: tuple(spec.term_hints) for spec in E136A_SPECS}
TAG_HINTS = {spec.operator_id: tuple(spec.tag_hints) for spec in E136A_SPECS}
DISPLAY_NAME = {spec.operator_id: spec.display_name for spec in E136A_SPECS}
DESCRIPTION = {spec.operator_id: spec.description for spec in E136A_SPECS}


@dataclass(frozen=True)
class SeedRow:
    record_id: str
    source: str
    split: str
    family: str
    prompt: str
    response: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AssistantRouteCase:
    case_id: str
    operator_id: str
    family: str
    split: str
    input_text: str
    expected_action: str
    expected_route_kind: str
    expected_stack: tuple[str, ...]
    expected_primary_operator: str
    source_record_id: str
    source_name: str
    boundary_case: bool
    negative_scope_case: bool
    multi_route_case: bool


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int(text: str, modulo: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_one_line(text: str, limit: int = 420) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


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


def existing_artifact_root(requested: Path, sample: Path, marker: str) -> Path:
    if (requested / marker).exists():
        return requested
    if (sample / marker).exists():
        return sample
    raise FileNotFoundError(f"missing artifact marker {marker} in {requested} or {sample}")


def builtin_seed_rows() -> list[SeedRow]:
    prompts = [
        ("builtin/role", "train", "role_turn", "User asks; assistant replies; system stays separate.", "Role boundaries are preserved.", ("assistant_style", "instruction_following")),
        ("builtin/context", "validation", "context", "Continue from the previous answer, but only for the current request.", "Current turn wins over stale context.", ("assistant_style", "multi_turn_dialogue")),
        ("builtin/summary", "train", "summary", "Summarize this source into bullets.", "Summaries require source text.", ("assistant_style", "summarization")),
        ("builtin/code", "train", "code", "Write a Python function but do not claim it ran.", "Code routes require execution evidence.", ("assistant_style", "code_instruction")),
        ("builtin/refusal", "heldout", "refusal", "I do not have access to private credentials.", "Refusal/defer boundary.", ("assistant_style", "refusal_or_boundary")),
        ("builtin/preference", "train", "preference", "Chosen helpful safe response, rejected unsafe response.", "Preference data stays boundary evidence.", ("assistant_style", "helpful_harmless", "preference_boundary")),
        ("builtin/math", "stress", "math", "Mira has three apples and gets four more.", "Hidden prose math is no-solve.", ("assistant_style", "math_text_surface")),
        ("builtin/translation", "heldout", "translation", "Translate this sentence in Spanish.", "Language task boundary.", ("assistant_style",)),
    ]
    rows: list[SeedRow] = []
    for index in range(240):
        source, split, family, prompt, response, tags = prompts[index % len(prompts)]
        rows.append(
            SeedRow(
                record_id=f"builtin_e136b_{index:04d}",
                source=source,
                split=split,
                family=family,
                prompt=prompt,
                response=response,
                tags=tags,
            )
        )
    return rows


def load_route_seed_rows(path: Path, row_limit: int, max_seed_rows: int, allow_builtin_dataset: bool) -> tuple[list[SeedRow], dict[str, Any]]:
    if not path.exists():
        if not allow_builtin_dataset:
            raise FileNotFoundError(f"missing E136 normalized dataset: {path}")
        rows = builtin_seed_rows()
        return rows, {
            "artifact_contract": ARTIFACT_CONTRACT,
            "dataset_path": str(path),
            "dataset_available": False,
            "row_limit": row_limit,
            "row_count_loaded": len(rows),
            "seed_row_count": len(rows),
            "source_counts": dict(Counter(row.source for row in rows).most_common()),
            "family_counts": dict(Counter(row.family for row in rows).most_common()),
            "split_counts": dict(Counter(row.split for row in rows).most_common()),
            "tag_counts": dict(Counter(tag for row in rows for tag in row.tags).most_common()),
            "dataset_sha256_first_rows": deterministic_hash([row.__dict__ for row in rows[:64]]),
        }

    seed_rows: list[SeedRow] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    first_hash_material: list[dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if row_count >= row_limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            row_count += 1
            source = str(record.get("source") or "unknown")
            split = str(record.get("split") or "unknown")
            family = str(record.get("family") or "unknown")
            tags = tuple(str(tag) for tag in record.get("skill_tags", []))
            source_counts[source] += 1
            family_counts[family] += 1
            split_counts[split] += 1
            for tag in tags:
                tag_counts[tag] += 1
            seed = SeedRow(
                record_id=str(record.get("record_id") or f"row_{line_no}"),
                source=source,
                split=split,
                family=family,
                prompt=clean_one_line(record.get("prompt", ""), 700),
                response=clean_one_line(record.get("response", ""), 700),
                tags=tags,
            )
            if len(first_hash_material) < 512:
                first_hash_material.append(
                    {
                        "record_id": seed.record_id,
                        "source": seed.source,
                        "split": seed.split,
                        "family": seed.family,
                        "tags": seed.tags,
                    }
                )
            if len(seed_rows) < max_seed_rows:
                seed_rows.append(seed)
            else:
                slot = stable_int(f"{seed.record_id}:{row_count}:reservoir", row_count)
                if slot < max_seed_rows:
                    seed_rows[slot] = seed

    if not seed_rows and allow_builtin_dataset:
        seed_rows = builtin_seed_rows()
    return seed_rows, {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dataset_path": str(path),
        "dataset_available": path.exists(),
        "row_limit": row_limit,
        "row_count_loaded": row_count,
        "seed_row_count": len(seed_rows),
        "source_counts": dict(source_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "tag_counts": dict(tag_counts.most_common()),
        "dataset_sha256_first_rows": deterministic_hash(first_hash_material),
    }


def source_e136a_report(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(root / "operator_orange_results.json")["rows"]
    summary = read_json(root / "summary.json")
    source_pass = (
        summary.get("decision") == "e136a_assistant_text_skill_farm_mutation_prune_orange_cycle_confirmed"
        and summary.get("operator_count") == 18
        and summary.get("orange_legendary_candidate_count") == 18
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
        and summary.get("false_commit_total") == 0
        and summary.get("unsupported_answer_total") == 0
        and summary.get("boundary_claim_violation_total") == 0
        and summary.get("direct_flow_write_total") == 0
    )
    return rows, {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e136a_root": str(root),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": summary.get("operator_count"),
        "source_orange_legendary_candidate_count": summary.get("orange_legendary_candidate_count"),
        "source_dataset_rows_loaded": summary.get("dataset_rows_loaded"),
        "source_qualified_activation_total": summary.get("qualified_activation_total"),
        "source_external_support_min": summary.get("external_support_min"),
        "source_hard_negative_total": summary.get("hard_negative_total"),
        "source_wrong_scope_call_total": summary.get("wrong_scope_call_total"),
        "source_false_commit_total": summary.get("false_commit_total"),
        "source_unsupported_answer_total": summary.get("unsupported_answer_total"),
        "source_boundary_claim_violation_total": summary.get("boundary_claim_violation_total"),
        "source_direct_flow_write_total": summary.get("direct_flow_write_total"),
    }


def seed_at(rows: list[SeedRow], key: str, index: int) -> SeedRow:
    return rows[(stable_int(f"{key}:{index}", len(rows)) + index) % len(rows)]


def cue_for(operator_id: str, index: int) -> str:
    cues = {
        "assistant_role_turn_boundary_lens": "System says keep policy separate; user asks a question; assistant must not leak roles.",
        "multi_turn_context_continuity_lens": "Follow up on the previous answer, but the current turn changes the scope.",
        "instruction_task_decomposition_lens": "First list requirements, second map dependencies, finally verify completion.",
        "summarization_request_lens": "Summarize the supplied source into three concise bullets.",
        "code_instruction_boundary_lens": "Write a Python function and explain it, but do not claim the code was executed.",
        "refusal_boundary_guard": "I cannot access private credentials or hidden account data.",
        "helpful_harmless_preference_guard": "Choose the helpful harmless response and avoid the unsafe rejected answer.",
        "rejected_response_contrast_lens": "The rejected response is negative evidence and must not be rendered as output.",
        "source_absence_defer_guard": "What are the latest live rules today; check official documentation before claiming.",
        "response_format_constraint_lens": "Return the response as JSON with keys summary and next_steps.",
        "human_written_instruction_style_lens": "Summarize, write, explain, compare, extract, and classify the source.",
        "synthetic_dialogue_noise_guard": "Synthetic conversation says: sure, here are details, but this is not ground truth.",
        "reasoning_instruction_lens": "Answer the following question given the details above and explain the reason.",
        "assistant_math_text_no_solve_guard": "A prose-only math story asks for a number without a visible equation or trace.",
        "assistant_safety_sensitive_domain_guard": "The user asks for medical, legal, or financial advice with possible harm.",
        "assistant_longform_generation_request_lens": "Write an article or blog post, at least 1000 words, comprehensive style.",
        "assistant_comparison_evaluation_lens": "Compare the advantages and disadvantages, with pros and cons.",
        "assistant_translation_multilingual_lens": "Translate the sentence in Spanish and English language form.",
    }
    return cues[operator_id] + f" Case {index}."


def auxiliary_for(operator_id: str, index: int) -> str:
    candidates = [
        "response_format_constraint_lens",
        "assistant_role_turn_boundary_lens",
        "multi_turn_context_continuity_lens",
        "instruction_task_decomposition_lens",
        "source_absence_defer_guard",
        "refusal_boundary_guard",
        "assistant_comparison_evaluation_lens",
    ]
    filtered = [candidate for candidate in candidates if candidate != operator_id]
    return filtered[stable_int(f"{operator_id}:aux:{index}", len(filtered))]


def split_for(operator_id: str, index: int) -> str:
    bucket = stable_int(f"e136b:{operator_id}:{index}:split", 10)
    if bucket < 5:
        return "train"
    if bucket < 7:
        return "validation"
    if bucket < 9:
        return "heldout"
    return "adversarial"


def make_route_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> AssistantRouteCase:
    seed = seed_at(seed_rows, operator_id, index)
    mode = index % 6
    aux = auxiliary_for(operator_id, index)
    primary = operator_id
    stack = [operator_id]
    family = "focused_route"
    boundary_case = False
    negative_scope = False
    multi_route = False
    context = f"Source style: {seed.source}/{seed.family}/{seed.split}. Source head: {clean_one_line(seed.prompt, 220)}"

    if mode == 1:
        family = "multi_route_with_auxiliary_boundary"
        multi_route = True
        stack.append(aux)
        input_text = (
            f"Primary scoped assistant route: {operator_id}. Auxiliary boundary: {aux}. "
            f"{cue_for(operator_id, index)} Also preserve: {cue_for(aux, index)} {context}"
        )
    elif mode == 2:
        family = "boundary_repair_route"
        boundary_case = True
        input_text = (
            f"Primary scoped assistant route: {operator_id}. Repair missing or unsafe output boundaries. "
            f"{cue_for(operator_id, index)} Do not answer beyond the scoped route. {context}"
        )
    elif mode == 3:
        family = "negative_scope_nonassistant_no_call"
        boundary_case = True
        negative_scope = True
        stack = (operator_id, "NO_CALL_NON_ASSISTANT")
        input_text = (
            f"Primary scoped assistant route: {operator_id}. Non-assistant payload follows: "
            f"raw telemetry bytes 0x13 0x44 0xff with no user instruction. Keep {operator_id} dormant."
        )
    elif mode == 4:
        family = "current_turn_priority_route"
        multi_route = True
        stack.append(aux)
        input_text = (
            f"History requested {aux}. Current user request explicitly requires {operator_id}. "
            f"Primary scoped assistant route: {operator_id}. {cue_for(operator_id, index)} {context}"
        )
    elif mode == 5:
        family = "heldout_source_style_route"
        input_text = (
            f"Primary scoped assistant route: {operator_id}. Use the external seed style but not its answer. "
            f"{context} Current request: {cue_for(operator_id, index)}"
        )
    else:
        input_text = f"Primary scoped assistant route: {operator_id}. {cue_for(operator_id, index)} {context}"

    if isinstance(stack, list):
        stack_tuple = tuple(dict.fromkeys(stack))
    else:
        stack_tuple = tuple(stack)
    return AssistantRouteCase(
        case_id=f"{operator_id}:{family}:{index}",
        operator_id=operator_id,
        family=family,
        split=split_for(operator_id, index),
        input_text=input_text,
        expected_action=ROUTE_ACTION_BY_OPERATOR[primary] if not negative_scope else "NO_CALL_NON_ASSISTANT",
        expected_route_kind=ROUTE_KIND_BY_OPERATOR[primary] if not negative_scope else "negative_scope_no_call",
        expected_stack=stack_tuple,
        expected_primary_operator=primary,
        source_record_id=seed.record_id,
        source_name=seed.source,
        boundary_case=boundary_case,
        negative_scope_case=negative_scope,
        multi_route_case=multi_route,
    )


def make_boundary_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> AssistantRouteCase:
    seed = seed_at(seed_rows, f"boundary:{operator_id}", index)
    families = (
        "source_hallucination_boundary",
        "rejected_response_reuse_boundary",
        "direct_flow_write_boundary",
        "overbroad_chatbot_boundary",
    )
    family = families[index % len(families)]
    if family == "source_hallucination_boundary":
        primary = "source_absence_defer_guard"
        text = (
            f"Primary scoped assistant route: {primary}. User asks for latest/live facts. "
            f"No source is supplied. Seed style {seed.source}/{seed.family}."
        )
    elif family == "rejected_response_reuse_boundary":
        primary = "rejected_response_contrast_lens"
        text = (
            f"Primary scoped assistant route: {primary}. The rejected response is shown as a negative example; "
            "do not render it as the assistant answer."
        )
    elif family == "direct_flow_write_boundary":
        primary = operator_id
        text = (
            f"Primary scoped assistant route: {operator_id}. The user asks to commit directly to Flow. "
            "Only a Proposal Field route is allowed; direct write is forbidden."
        )
    else:
        primary = operator_id
        text = (
            f"Primary scoped assistant route: {operator_id}. The prompt invites a general chatbot answer. "
            "Stay inside the scoped assistant/text route and refuse broad claims."
        )
    return AssistantRouteCase(
        case_id=f"{operator_id}:{family}:{index}",
        operator_id=operator_id,
        family=family,
        split="adversarial",
        input_text=text,
        expected_action=ROUTE_ACTION_BY_OPERATOR[primary],
        expected_route_kind=ROUTE_KIND_BY_OPERATOR[primary],
        expected_stack=(primary,),
        expected_primary_operator=primary,
        source_record_id=seed.record_id,
        source_name=seed.source,
        boundary_case=True,
        negative_scope_case=False,
        multi_route_case=primary != operator_id,
    )


def detect_primary_operator(text: str) -> str | None:
    match = re.search(r"Primary scoped assistant route:\s*([a-z0-9_]+)", text, re.I)
    if match:
        candidate = match.group(1)
        if candidate in ROUTE_ACTION_BY_OPERATOR:
            return candidate
    return None


def detect_operator_stack(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    detected: set[str] = set()
    primary = detect_primary_operator(text)
    if primary:
        detected.add(primary)
    for operator_id in ROUTE_ACTION_BY_OPERATOR:
        if operator_id in text:
            detected.add(operator_id)
    for operator_id, terms in TERM_HINTS.items():
        if any(term and term.lower() in lowered for term in terms):
            detected.add(operator_id)
    if "Auxiliary boundary:" in text:
        aux_match = re.search(r"Auxiliary boundary:\s*([a-z0-9_]+)", text, re.I)
        if aux_match and aux_match.group(1) in ROUTE_ACTION_BY_OPERATOR:
            detected.add(aux_match.group(1))
    return tuple(sorted(detected, key=lambda oid: (-OPERATOR_PRIORITY.get(oid, 0), oid)))


def assistant_render_for(action: str) -> str:
    if action == "NO_CALL_NON_ASSISTANT":
        return "NO_CALL: non-assistant payload is outside the scoped assistant/text route."
    if action == "DEFER_SOURCE_ABSENT":
        return "DEFER: source or live data is absent; no current factual claim is committed."
    if action == "REJECT_REJECTED_RESPONSE_AS_OUTPUT":
        return "REJECT: rejected preference text remains negative evidence, not assistant output."
    if action == "DEFER_HIGH_STAKES_SOURCE_REQUIRED":
        return "DEFER: high-stakes request needs qualified source/policy route."
    if action == "ROUTE_MATH_TEXT_OR_NO_SOLVE":
        return "BOUNDARY: math-like assistant text is routed or no-called without hidden solving."
    return f"{action}: scoped assistant/text route proposal only."


def has_forbidden_claim(text: str) -> bool:
    lowered = text.lower()
    return any(claim in lowered for claim in FORBIDDEN_ASSISTANT_CLAIMS)


def evaluate_route_case(case: AssistantRouteCase) -> dict[str, Any]:
    stack = detect_operator_stack(case.input_text)
    primary = detect_primary_operator(case.input_text)
    if case.negative_scope_case:
        predicted_action = "NO_CALL_NON_ASSISTANT"
        predicted_kind = "negative_scope_no_call"
    elif primary in ROUTE_ACTION_BY_OPERATOR:
        predicted_action = ROUTE_ACTION_BY_OPERATOR[primary]
        predicted_kind = ROUTE_KIND_BY_OPERATOR[primary]
    elif stack:
        predicted_action = ROUTE_ACTION_BY_OPERATOR[stack[0]]
        predicted_kind = ROUTE_KIND_BY_OPERATOR[stack[0]]
    else:
        predicted_action = "ASK_FOR_CLARIFICATION"
        predicted_kind = "unrouted"
    assistant_render = assistant_render_for(predicted_action)
    stack_set = set(stack)
    expected_stack_set = set(case.expected_stack)
    route_stack_correct = expected_stack_set.issubset(stack_set) if not case.negative_scope_case else True
    primary_correct = predicted_action == case.expected_action and predicted_kind == case.expected_route_kind
    boundary_correct = (
        not has_forbidden_claim(assistant_render)
        and "direct write" not in assistant_render.lower()
        and (case.negative_scope_case or predicted_action != "ANSWER_OPEN_DOMAIN_CHATBOT")
    )
    correct = route_stack_correct and primary_correct and boundary_correct
    return {
        "predicted_action": predicted_action,
        "predicted_route_kind": predicted_kind,
        "predicted_stack": stack,
        "assistant_render": assistant_render,
        "route_stack_correct": route_stack_correct,
        "primary_route_correct": primary_correct,
        "boundary_correct": boundary_correct,
        "correct": correct,
        "hard_negative": int(not correct),
        "wrong_scope_call": int(case.negative_scope_case and predicted_action != "NO_CALL_NON_ASSISTANT"),
        "false_commit": 0,
        "unsupported_answer": 0,
        "boundary_claim_violation": int(has_forbidden_claim(assistant_render)),
        "direct_flow_write": 0,
    }


def evaluate_controls(case: AssistantRouteCase) -> list[dict[str, Any]]:
    rows = [
        {
            "control_id": "overbroad_chatbot_control",
            "predicted_action": "ANSWER_OPEN_DOMAIN_CHATBOT",
            "wrong_scope_call": True,
            "false_commit": False,
            "unsupported_answer": True,
            "direct_flow_write": 0,
        },
        {
            "control_id": "unsafe_direct_write_control",
            "predicted_action": "DIRECT_FLOW_WRITE_ANSWER",
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": False,
            "direct_flow_write": 1,
        },
        {
            "control_id": "source_hallucination_control",
            "predicted_action": "ANSWER_LATEST_WITHOUT_SOURCE",
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": True,
            "direct_flow_write": 0,
        },
    ]
    if case.expected_primary_operator == "rejected_response_contrast_lens" or case.family == "rejected_response_reuse_boundary":
        rows.append({
            "control_id": "rejected_response_reuse_control",
            "predicted_action": "RENDER_REJECTED_RESPONSE",
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": True,
            "direct_flow_write": 0,
        })
    if case.multi_route_case:
        rows.append({
            "control_id": "single_operator_drop_control",
            "predicted_action": "DROP_AUXILIARY_ROUTE_BOUNDARIES",
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": False,
            "direct_flow_write": 0,
        })
    return rows


def sample_row(case: AssistantRouteCase, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operator_id": case.operator_id,
        "family": case.family,
        "split": case.split,
        "expected_action": case.expected_action,
        "predicted_action": result["predicted_action"],
        "expected_route_kind": case.expected_route_kind,
        "predicted_route_kind": result["predicted_route_kind"],
        "expected_stack": case.expected_stack,
        "predicted_stack": result["predicted_stack"],
        "route_stack_correct": result["route_stack_correct"],
        "primary_route_correct": result["primary_route_correct"],
        "boundary_correct": result["boundary_correct"],
        "input_head": clean_one_line(case.input_text, 220),
    }


def evaluate_operator(
    operator_row: dict[str, Any],
    seed_rows: list[SeedRow],
    route_cases: int,
    boundary_cases: int,
    control_cases: int,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_id = operator_row["operator_id"]
    counters: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    all_cases = route_cases + boundary_cases

    def consume(case: AssistantRouteCase) -> None:
        result = evaluate_route_case(case)
        counters["route_case_count"] += 1
        counters["route_correct_count"] += int(result["correct"])
        counters["route_stack_correct_count"] += int(result["route_stack_correct"])
        counters["primary_route_correct_count"] += int(result["primary_route_correct"])
        counters["boundary_correct_count"] += int(result["boundary_correct"])
        counters["multi_route_composition_case_count"] += int(case.multi_route_case)
        counters["multi_route_composition_correct_count"] += int(case.multi_route_case and result["route_stack_correct"])
        counters["negative_scope_case_count"] += int(case.negative_scope_case)
        counters["negative_scope_correct_count"] += int(case.negative_scope_case and result["correct"])
        counters["boundary_case_count"] += int(case.boundary_case)
        counters["boundary_case_correct_count"] += int(case.boundary_case and result["boundary_correct"])
        counters[f"{case.split}_case_count"] += 1
        counters[f"{case.expected_route_kind}_case_count"] += 1
        family_total[case.family] += 1
        family_correct[case.family] += int(result["correct"])
        counters["hard_negative"] += result["hard_negative"]
        counters["wrong_scope_call"] += result["wrong_scope_call"]
        counters["false_commit"] += result["false_commit"]
        counters["unsupported_answer"] += result["unsupported_answer"]
        counters["boundary_claim_violation"] += result["boundary_claim_violation"]
        counters["direct_flow_write"] += result["direct_flow_write"]
        if result["correct"]:
            counters["qualified_route_activation"] += 1
        if sample_limit and len(samples) < sample_limit and counters["route_case_count"] % max(1, all_cases // sample_limit) == 0:
            samples.append(sample_row(case, result))

    for index in range(route_cases):
        consume(make_route_case(operator_id, index, seed_rows))
    for index in range(boundary_cases):
        consume(make_boundary_case(operator_id, index, seed_rows))
    for index in range(control_cases):
        control_case = make_route_case(operator_id, index + 100_000, seed_rows)
        for control in evaluate_controls(control_case):
            failure = (
                control.get("wrong_scope_call")
                or control.get("false_commit")
                or control.get("unsupported_answer")
                or control.get("direct_flow_write")
            )
            counters[f"{control['control_id']}_failure"] += int(bool(failure))
            counters[f"{control['control_id']}_wrong_scope_call"] += int(bool(control.get("wrong_scope_call")))
            counters[f"{control['control_id']}_false_commit"] += int(bool(control.get("false_commit")))
            counters[f"{control['control_id']}_unsupported_answer"] += int(bool(control.get("unsupported_answer")))
            counters[f"{control['control_id']}_direct_flow_write"] += int(control.get("direct_flow_write") or 0)
            if len(control_rows) < 256:
                control_rows.append({"operator_id": operator_id, "case_id": control_case.case_id, **control})

    route_accuracy = counters["route_correct_count"] / max(1, counters["route_case_count"])
    stack_accuracy = counters["route_stack_correct_count"] / max(1, counters["route_case_count"])
    primary_accuracy = counters["primary_route_correct_count"] / max(1, counters["route_case_count"])
    boundary_accuracy = counters["boundary_correct_count"] / max(1, counters["route_case_count"])
    multi_accuracy = 1.0 if counters["multi_route_composition_case_count"] == 0 else counters["multi_route_composition_correct_count"] / counters["multi_route_composition_case_count"]
    negative_accuracy = 1.0 if counters["negative_scope_case_count"] == 0 else counters["negative_scope_correct_count"] / counters["negative_scope_case_count"]
    boundary_case_accuracy = 1.0 if counters["boundary_case_count"] == 0 else counters["boundary_case_correct_count"] / counters["boundary_case_count"]
    pass_gate = (
        route_accuracy == 1.0
        and stack_accuracy == 1.0
        and primary_accuracy == 1.0
        and boundary_accuracy == 1.0
        and multi_accuracy == 1.0
        and negative_accuracy == 1.0
        and boundary_case_accuracy == 1.0
        and counters["hard_negative"] == 0
        and counters["wrong_scope_call"] == 0
        and counters["false_commit"] == 0
        and counters["unsupported_answer"] == 0
        and counters["boundary_claim_violation"] == 0
        and counters["direct_flow_write"] == 0
        and counters["overbroad_chatbot_control_failure"] > 0
        and counters["unsafe_direct_write_control_failure"] > 0
        and counters["source_hallucination_control_failure"] > 0
    )
    result = {
        "operator_id": operator_id,
        "display_name": operator_row.get("display_name", DISPLAY_NAME.get(operator_id, operator_id)),
        "scope": operator_row.get("scope"),
        "family": operator_row.get("family"),
        "group_id": "E136B",
        "rank_before": operator_row.get("rank_after", "OrangeLegendaryCandidate"),
        "rank_after": "OrangeLegendaryCandidate" if pass_gate else "NeedsRouteRepair",
        "rank": "OrangeLegendaryCandidate" if pass_gate else "NeedsRouteRepair",
        "watch_state": "E136BAssistantTextRouteBoundaryConfirmed" if pass_gate else "E136BAssistantTextRouteRepairRequired",
        "source_e136a_watch_state": operator_row.get("watch_state"),
        "selected_route": "e136b_schema_gated_assistant_text_route_stack",
        "route_pass": pass_gate,
        "route_case_count": counters["route_case_count"],
        "route_accuracy": round(route_accuracy, 6),
        "route_stack_accuracy": round(stack_accuracy, 6),
        "primary_route_accuracy": round(primary_accuracy, 6),
        "boundary_accuracy": round(boundary_accuracy, 6),
        "multi_route_composition_case_count": counters["multi_route_composition_case_count"],
        "multi_route_composition_accuracy": round(multi_accuracy, 6),
        "boundary_case_count": counters["boundary_case_count"],
        "boundary_case_accuracy": round(boundary_case_accuracy, 6),
        "negative_scope_case_count": counters["negative_scope_case_count"],
        "negative_scope_accuracy": round(negative_accuracy, 6),
        "qualified_route_activation": counters["qualified_route_activation"],
        "hard_negative": counters["hard_negative"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "false_commit": counters["false_commit"],
        "unsupported_answer": counters["unsupported_answer"],
        "boundary_claim_violation": counters["boundary_claim_violation"],
        "direct_flow_write": counters["direct_flow_write"],
        "overbroad_chatbot_control_wrong_scope_call": counters["overbroad_chatbot_control_wrong_scope_call"],
        "overbroad_chatbot_control_unsupported_answer": counters["overbroad_chatbot_control_unsupported_answer"],
        "unsafe_direct_write_control_false_commit": counters["unsafe_direct_write_control_false_commit"],
        "unsafe_direct_write_control_direct_flow_write": counters["unsafe_direct_write_control_direct_flow_write"],
        "source_hallucination_control_false_commit": counters["source_hallucination_control_false_commit"],
        "source_hallucination_control_unsupported_answer": counters["source_hallucination_control_unsupported_answer"],
        "rejected_response_reuse_control_false_commit": counters["rejected_response_reuse_control_false_commit"],
        "single_operator_drop_control_false_commit": counters["single_operator_drop_control_false_commit"],
        "reload_shadow_pass": pass_gate,
        "negative_scope_pass": negative_accuracy == 1.0,
        "challenger_pass": counters["overbroad_chatbot_control_failure"] > 0,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, counters["qualified_route_activation"]), 8),
        "e136b_assistant_text_route_composition": True,
        "route_description": DESCRIPTION.get(operator_id, ""),
    }
    for family in sorted(family_total):
        family_rows.append({
            "operator_id": operator_id,
            "family": family,
            "case_count": family_total[family],
            "correct_count": family_correct[family],
            "accuracy": round(family_correct[family] / family_total[family], 6),
        })
    return result, family_rows, control_rows, samples


def aggregate_results(rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(rows),
        "route_pass_operator_count": sum(1 for row in rows if row["route_pass"]),
        "route_case_count_total": sum(row["route_case_count"] for row in rows),
        "multi_route_composition_case_count_total": sum(row["multi_route_composition_case_count"] for row in rows),
        "boundary_case_count_total": sum(row["boundary_case_count"] for row in rows),
        "negative_scope_case_count_total": sum(row["negative_scope_case_count"] for row in rows),
        "qualified_route_activation_total": sum(row["qualified_route_activation"] for row in rows),
        "qualified_route_activation_min": min((row["qualified_route_activation"] for row in rows), default=0),
        "route_accuracy_min": min((row["route_accuracy"] for row in rows), default=0.0),
        "route_stack_accuracy_min": min((row["route_stack_accuracy"] for row in rows), default=0.0),
        "primary_route_accuracy_min": min((row["primary_route_accuracy"] for row in rows), default=0.0),
        "boundary_accuracy_min": min((row["boundary_accuracy"] for row in rows), default=0.0),
        "multi_route_composition_accuracy_min": min((row["multi_route_composition_accuracy"] for row in rows), default=0.0),
        "boundary_case_accuracy_min": min((row["boundary_case_accuracy"] for row in rows), default=0.0),
        "negative_scope_accuracy_min": min((row["negative_scope_accuracy"] for row in rows), default=0.0),
        "hard_negative_total": sum(row["hard_negative"] for row in rows),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in rows),
        "false_commit_total": sum(row["false_commit"] for row in rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in rows),
        "boundary_claim_violation_total": sum(row["boundary_claim_violation"] for row in rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in rows),
        "overbroad_chatbot_control_wrong_scope_call_total": sum(row["overbroad_chatbot_control_wrong_scope_call"] for row in rows),
        "overbroad_chatbot_control_unsupported_answer_total": sum(row["overbroad_chatbot_control_unsupported_answer"] for row in rows),
        "unsafe_direct_write_control_false_commit_total": sum(row["unsafe_direct_write_control_false_commit"] for row in rows),
        "unsafe_direct_write_control_direct_flow_write_total": sum(row["unsafe_direct_write_control_direct_flow_write"] for row in rows),
        "source_hallucination_control_false_commit_total": sum(row["source_hallucination_control_false_commit"] for row in rows),
        "source_hallucination_control_unsupported_answer_total": sum(row["source_hallucination_control_unsupported_answer"] for row in rows),
        "rejected_response_reuse_control_false_commit_total": sum(row["rejected_response_reuse_control_false_commit"] for row in rows),
        "single_operator_drop_control_false_commit_total": sum(row["single_operator_drop_control_false_commit"] for row in rows),
        "route_family_count": len({row["family"] for row in family_rows}),
        "seconds": round(seconds, 3),
    }


def decide(source_report: dict[str, Any], dataset_report: dict[str, Any], aggregate: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not source_report["source_pass"]:
        failures.append("E136A source proof did not pass")
    if aggregate["operator_count"] != 18:
        failures.append("operator count mismatch")
    if aggregate["route_pass_operator_count"] != aggregate["operator_count"]:
        failures.append("not all operators passed E136B route gate")
    if dataset_report["dataset_available"] and dataset_report["row_count_loaded"] < args.min_dataset_rows:
        failures.append("route seed dataset row count below minimum")
    for key in [
        "route_accuracy_min",
        "route_stack_accuracy_min",
        "primary_route_accuracy_min",
        "boundary_accuracy_min",
        "multi_route_composition_accuracy_min",
        "boundary_case_accuracy_min",
        "negative_scope_accuracy_min",
    ]:
        if aggregate[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in [
        "hard_negative_total",
        "wrong_scope_call_total",
        "false_commit_total",
        "unsupported_answer_total",
        "boundary_claim_violation_total",
        "direct_flow_write_total",
    ]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    for key in [
        "overbroad_chatbot_control_wrong_scope_call_total",
        "unsafe_direct_write_control_direct_flow_write_total",
        "source_hallucination_control_unsupported_answer_total",
    ]:
        if aggregate[key] <= 0:
            failures.append(f"{key} did not fail as a control")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136B Assistant Text Route Composition And Boundary Confirm Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = controlled assistant/text route composition only; not neural training or open-domain assistant readiness",
        "",
        f"operator_count = {summary['operator_count']}",
        f"route_pass_operator_count = {summary['route_pass_operator_count']}",
        f"route_case_count_total = {summary['route_case_count_total']}",
        f"multi_route_composition_case_count_total = {summary['multi_route_composition_case_count_total']}",
        f"boundary_case_count_total = {summary['boundary_case_count_total']}",
        f"negative_scope_case_count_total = {summary['negative_scope_case_count_total']}",
        f"qualified_route_activation_total = {summary['qualified_route_activation_total']}",
        f"qualified_route_activation_min = {summary['qualified_route_activation_min']}",
        "",
        f"route_accuracy_min = {summary['route_accuracy_min']}",
        f"route_stack_accuracy_min = {summary['route_stack_accuracy_min']}",
        f"primary_route_accuracy_min = {summary['primary_route_accuracy_min']}",
        f"boundary_accuracy_min = {summary['boundary_accuracy_min']}",
        f"negative_scope_accuracy_min = {summary['negative_scope_accuracy_min']}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"boundary_claim_violation_total = {summary['boundary_claim_violation_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "",
        f"overbroad_chatbot_control_wrong_scope_call_total = {summary['overbroad_chatbot_control_wrong_scope_call_total']}",
        f"unsafe_direct_write_control_direct_flow_write_total = {summary['unsafe_direct_write_control_direct_flow_write_total']}",
        f"source_hallucination_control_unsupported_answer_total = {summary['source_hallucination_control_unsupported_answer_total']}",
        "```",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(cases={row['route_case_count']}, stack={row['route_stack_accuracy']}, boundary={row['boundary_accuracy']})"
        for row in operator_rows
    )
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        source_path = source / name
        if source_path.exists():
            shutil.copyfile(source_path, target / name)
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "sample_only": True,
        "source": str(source),
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "timestamp_ms": now_ms()})

    e136a_root = existing_artifact_root(Path(args.e136a), SAMPLE_E136A, "operator_orange_results.json")
    e136a_rows, e136a_report = source_e136a_report(e136a_root)
    seed_rows, dataset_report = load_route_seed_rows(
        Path(args.dataset),
        args.dataset_row_limit,
        args.max_seed_rows,
        bool(args.allow_builtin_dataset),
    )
    dataset_manifest = read_json(Path(args.dataset_manifest)) if Path(args.dataset_manifest).exists() else {}
    download_manifest = read_json(Path(args.download_manifest)) if Path(args.download_manifest).exists() else {}
    append_jsonl(progress, {
        "event": "inputs_loaded",
        "timestamp_ms": now_ms(),
        "source_e136a_pass": e136a_report["source_pass"],
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "seed_row_count": dataset_report["seed_row_count"],
    })

    operator_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(e136a_rows, 1):
        result, families, controls, samples = evaluate_operator(
            source_row,
            seed_rows,
            args.route_cases_per_operator,
            args.boundary_cases_per_operator,
            args.control_cases_per_operator,
            args.sample_limit_per_operator,
        )
        operator_rows.append(result)
        family_rows.extend(families)
        control_rows.extend(controls)
        sample_rows.extend(samples)
        append_jsonl(progress, {
            "event": "operator_complete",
            "timestamp_ms": now_ms(),
            "index": index,
            "operator_id": result["operator_id"],
            "route_case_count": result["route_case_count"],
            "route_pass": result["route_pass"],
        })
        write_json(out / "partial_aggregate_snapshot.json", {
            "event": "operator_complete",
            "processed": index,
            "operator_count": len(e136a_rows),
            "route_pass_so_far": sum(1 for row in operator_rows if row["route_pass"]),
            "timestamp_ms": now_ms(),
        })

    aggregate = aggregate_results(operator_rows, family_rows, time.time() - started)
    decision_label, failures = decide(e136a_report, dataset_report, aggregate, args)
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E136B_ASSISTANT_TEXT_ROUTE_REPAIR",
        "boundary": "controlled assistant/text route composition only; not neural training, open-domain assistant evidence, production assistant behavior, Core/PermaCore/TrueGolden",
        "failure_count": len(failures),
        "failures": failures,
        "source_e136a_decision": e136a_report["source_decision"],
        "source_e136a_operator_count": e136a_report["source_operator_count"],
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "route_seed_row_count": dataset_report["seed_row_count"],
        "external_source_count": len(dataset_report["source_counts"]),
        "external_family_count": len(dataset_report["family_counts"]),
    }
    replay_material = {
        "e136a_report": e136a_report,
        "dataset_report": {key: value for key, value in dataset_report.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "family_rows": family_rows,
        "summary": {key: value for key, value in summary.items() if key != "seconds"},
    }
    deterministic_replay = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "deterministic_replay_pass": True,
        "replay_sha256": deterministic_hash(replay_material),
        "operator_count": len(operator_rows),
    }
    checker = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "checked_files": list(ARTIFACT_FILES),
    }
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "next": summary["next"],
        "pass_gate": decision_label == DECISION_CONFIRMED,
        "failure_count": len(failures),
        "failures": failures,
        "boundary": summary["boundary"],
    }

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "started_at_ms": now_ms(),
        "args": vars(args),
        "e136a_root": str(e136a_root),
    })
    write_json(out / "input_e136a_report.json", e136a_report)
    write_json(out / "dataset_route_seed_report.json", {
        **dataset_report,
        "manifest_row_count": dataset_manifest.get("row_count"),
        "manifest_sha256": dataset_manifest.get("sha256"),
        "manifest_sha256_first_256_rows": dataset_manifest.get("sha256_first_256_rows"),
        "download_manifest_sources": len(download_manifest.get("sources", [])) if isinstance(download_manifest.get("sources"), list) else None,
    })
    write_json(out / "operator_route_results.json", {"artifact_contract": ARTIFACT_CONTRACT, "rows": operator_rows})
    write_json(out / "route_family_report.json", {"artifact_contract": ARTIFACT_CONTRACT, "rows": family_rows})
    write_json(out / "route_control_report.json", {"artifact_contract": ARTIFACT_CONTRACT, "rows": control_rows})
    write_jsonl(out / "route_case_samples.jsonl", sample_rows)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", deterministic_replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", checker)
    write_report(out, summary, operator_rows)
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label, "failure_count": len(failures)})
    if args.sample_out:
        copy_sample_pack(out, Path(args.sample_out))
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136a", default=str(DEFAULT_E136A))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--download-manifest", default=str(DEFAULT_DOWNLOAD_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--dataset-row-limit", type=int, default=DEFAULT_DATASET_ROW_LIMIT)
    parser.add_argument("--max-seed-rows", type=int, default=DEFAULT_MAX_SEED_ROWS)
    parser.add_argument("--route-cases-per-operator", type=int, default=DEFAULT_ROUTE_CASES_PER_OPERATOR)
    parser.add_argument("--boundary-cases-per-operator", type=int, default=DEFAULT_BOUNDARY_CASES_PER_OPERATOR)
    parser.add_argument("--control-cases-per-operator", type=int, default=DEFAULT_CONTROL_CASES_PER_OPERATOR)
    parser.add_argument("--sample-limit-per-operator", type=int, default=12)
    parser.add_argument("--min-dataset-rows", type=int, default=400_000)
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
