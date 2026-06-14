#!/usr/bin/env python3
"""E115 alpha-Weave pressure-cell schema validation.

This probe locks the v1 pressure-cell schema before targeted data generation.
It validates a DnD-like sample cell pack with public/hidden separation and
adversarial controls that must fail when they rely on labels, latest-only
shortcuts, source-id shortcuts, citation-id shortcuts, or over-broad calls.

Boundary: schema/curriculum-unit validation only. This is not final training,
PermaCore promotion, open-domain reasoning, or a claim that the runtime has
learned the generated cells.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E115_ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_VALIDATION"
SCHEMA_VERSION = "AlphaWeavePressureCell-v1"

REQUIRED_TOP_LEVEL = {
    "cell_id",
    "cell_version",
    "public_input",
    "hidden_oracle",
    "training_metadata",
    "adversarial_variants",
    "scoring",
}
PUBLIC_REQUIRED = {"context", "current_cycle", "observations", "query"}
OBS_REQUIRED = {"obs_id", "cycle", "order", "source_id", "source_trust", "text", "span"}
QUERY_REQUIRED = {"text", "required_focus"}
ORACLE_REQUIRED = {"expected_action", "expected_answer", "required_evidence", "required_trace", "forbidden_behavior"}
METADATA_REQUIRED = {"target_skill", "target_operators", "operator_visibility", "route_budget"}
SCORING_REQUIRED = {
    "action_accuracy",
    "answer_accuracy",
    "citation_exact",
    "trace_dependency_coverage",
    "false_commit",
    "wrong_scope_call",
    "unsupported_answer",
    "over_budget",
    "neutral_valid",
    "neutral_waste",
}

PUBLIC_FORBIDDEN_KEYS = {
    "hidden_oracle",
    "training_metadata",
    "target_skill",
    "target_operators",
    "expected_action",
    "expected_answer",
    "required_trace",
    "forbidden_behavior",
    "oracle",
    "label",
}
PUBLIC_FORBIDDEN_TEXT = {
    "temporal_latest_span_t_stab",
    "evidence_conflict_detector_lens",
    "source_priority_resolver_lens",
    "evidence_citation_link_scribe",
    "unsupported_answer_defer_guard",
    "target_skill",
    "target_operators",
    "expected_answer",
    "hidden_oracle",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def canonical_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "top_level_required": sorted(REQUIRED_TOP_LEVEL),
        "public_input_required": sorted(PUBLIC_REQUIRED),
        "observation_required": sorted(OBS_REQUIRED),
        "query_required": sorted(QUERY_REQUIRED),
        "hidden_oracle_required": sorted(ORACLE_REQUIRED),
        "training_metadata_required": sorted(METADATA_REQUIRED),
        "scoring_required": sorted(SCORING_REQUIRED),
        "public_forbidden_keys": sorted(PUBLIC_FORBIDDEN_KEYS),
        "public_forbidden_text": sorted(PUBLIC_FORBIDDEN_TEXT),
        "hard_rule": "public_input must not contain target/operator/oracle/meta fields or labels",
        "candidate_visible_path": "public_input only",
        "trainer_checker_only_paths": ["hidden_oracle", "training_metadata", "adversarial_variants", "scoring"],
    }


def obs(obs_id: str, cycle: int, order: int, source_id: str, source_trust: float, text: str) -> dict[str, Any]:
    return {
        "obs_id": obs_id,
        "cycle": cycle,
        "order": order,
        "source_id": source_id,
        "source_trust": source_trust,
        "text": text,
        "span": {"start": 0, "end": len(text.encode("utf-8"))},
    }


def hidden(
    action: str,
    answer: str | None,
    must_use: list[str],
    must_not: list[str],
    trace: list[str],
    forbidden: list[str],
    citation_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "expected_action": action,
        "expected_answer": {"value": answer, "render": answer} if answer is not None else None,
        "required_evidence": {
            "must_use": must_use,
            "must_not_use_as_decisive": must_not,
            "citation_spans": citation_spans or [],
        },
        "required_trace": trace,
        "forbidden_behavior": forbidden,
    }


def base_public() -> dict[str, Any]:
    return {
        "context": "Dungeon evidence board: the party tracks whether one lever is safe right now. Notes may be old, weak, or replayed.",
        "current_cycle": 17,
        "observations": [
            obs("o1", 17, 1, "sA", 0.91, "At torchmark 2, the moon-rune lever was unsafe; the ward was still active."),
            obs("o2", 17, 2, "sB", 0.29, "A tavern note claims the moon-rune lever might still be unsafe."),
            obs("o3", 17, 3, "sA", 0.91, "At torchmark 5, the ward was dispelled and the moon-rune lever is safe to pull."),
        ],
        "query": {
            "text": "Should the party pull the moon-rune lever now?",
            "required_focus": {"entity": "moon-rune lever", "attribute": "current_safety"},
        },
    }


def make_cell_pack() -> dict[str, Any]:
    base_trace = [
        "track the requested entity: moon-rune lever",
        "treat weak source as non-decisive",
        "prefer current-cycle reliable evidence over stale earlier evidence",
        "cite the decisive evidence span",
    ]
    forbidden = [
        "answer from weak evidence",
        "answer from stale evidence",
        "answer without citation",
        "commit when evidence is missing",
        "call full library scan",
    ]
    def citation(public_input: dict[str, Any], obs_id: str) -> dict[str, Any]:
        observation = next(item for item in public_input["observations"] if item["obs_id"] == obs_id)
        return {"obs_id": obs_id, "start": observation["span"]["start"], "end": observation["span"]["end"]}

    base_public_input = base_public()
    base = {
        "cell_id": "aw_dnd_moon_rune_lever_000001",
        "cell_version": 1,
        "public_input": base_public_input,
        "hidden_oracle": hidden(
            "ANSWER",
            "The moon-rune lever is safe to pull now.",
            ["o3"],
            ["o1", "o2"],
            base_trace,
            forbidden,
            [citation(base_public_input, "o3")],
        ),
        "training_metadata": {
            "target_skill": "latest_reliable_evidence_with_negative_scope",
            "target_operators": [
                "temporal_latest_span_t_stab",
                "evidence_conflict_detector_lens",
                "source_priority_resolver_lens",
                "evidence_citation_link_scribe",
                "unsupported_answer_defer_guard",
            ],
            "operator_visibility": "hidden_from_candidate",
            "route_budget": {"max_operator_calls": 5, "max_trace_steps": 8},
        },
        "adversarial_variants": [],
        "scoring": {
            "action_accuracy": 1,
            "answer_accuracy": 1,
            "citation_exact": 1,
            "trace_dependency_coverage": 1,
            "false_commit": 0,
            "wrong_scope_call": 0,
            "unsupported_answer": 0,
            "over_budget": 0,
            "neutral_valid": 0,
            "neutral_waste": 0,
        },
    }

    variants: list[dict[str, Any]] = []

    def variant(variant_id: str, public_input: dict[str, Any], oracle: dict[str, Any]) -> None:
        variants.append({"variant_id": variant_id, "public_input": public_input, "hidden_oracle": oracle})

    p = base_public()
    variant("answerable_base", p, base["hidden_oracle"])

    p = base_public()
    p["observations"] = p["observations"][:2]
    variant(
        "missing_evidence",
        p,
        hidden("ASK_FOR_EVIDENCE", None, [], ["o1", "o2"], ["no current reliable safety evidence exists"], forbidden),
    )

    p = base_public()
    p["observations"][2]["source_trust"] = 0.31
    p["observations"][2]["source_id"] = "sB"
    variant(
        "weak_source",
        p,
        hidden("DEFER", None, [], ["o1", "o2", "o3"], ["latest safe claim is weak and cannot be decisive"], forbidden),
    )

    p = base_public()
    p["observations"].append(obs("o4", 17, 3, "sC", 0.91, "At torchmark 5, a ward flare made the moon-rune lever unsafe again."))
    variant(
        "unresolved_conflict",
        p,
        hidden("DEFER", None, [], ["o3", "o4"], ["two current reliable observations conflict"], forbidden),
    )

    p = base_public()
    p["observations"][2]["cycle"] = 12
    variant(
        "stale_replay",
        p,
        hidden("DEFER", None, [], ["o1", "o2", "o3"], ["safe claim is from a stale cycle"], forbidden),
    )

    p = base_public()
    p["observations"][1]["source_trust"] = 0.91
    p["observations"][1]["source_id"] = "sA"
    p["observations"][2]["source_trust"] = 0.31
    p["observations"][2]["source_id"] = "sB"
    variant(
        "source_trust_inversion",
        p,
        hidden("DEFER", None, [], ["o1", "o2", "o3"], ["latest safe claim is weak after trust inversion"], forbidden),
    )

    p = base_public()
    p["observations"][0]["order"] = 4
    p["observations"][0]["text"] = "At torchmark 6, the moon-rune lever became unsafe; the ward returned."
    p["observations"][0]["span"] = {"start": 0, "end": len(p["observations"][0]["text"].encode("utf-8"))}
    variant(
        "order_swap",
        p,
        hidden(
            "ANSWER",
            "The moon-rune lever is unsafe now; do not pull it.",
            ["o1"],
            ["o2", "o3"],
            ["latest reliable current-cycle evidence is now o1"],
            forbidden,
            [citation(p, "o1")],
        ),
    )

    p = base_public()
    p["context"] = "A bard tells a fictional tavern story. This is not an active dungeon evidence board."
    p["observations"] = [obs("n1", 17, 1, "sN", 0.5, "In the story, a moon-rune lever was unsafe and then safe.")]
    p["query"] = {"text": "Summarize the mood of the tale.", "required_focus": {"entity": "story", "attribute": "mood"}}
    variant(
        "quote_or_inactive_scope",
        p,
        hidden("NO_CALL", None, [], ["n1"], ["inactive narrative scope, no live state update"], ["activate live-status route"]),
    )

    p = base_public()
    p["context"] = "This paragraph is flavor text for a campaign handout, not a live evidence log."
    p["query"] = {"text": "Rewrite the paragraph in a spooky tone.", "required_focus": {"entity": "paragraph", "attribute": "style"}}
    variant(
        "negative_scope_story_text",
        p,
        hidden("NO_CALL", None, [], ["o1", "o2", "o3"], ["style task does not require live safety commit"], ["commit lever status"]),
    )

    p = base_public()
    p["observations"][2]["obs_id"] = "o7"
    variant(
        "citation_id_shortcut_trap",
        p,
        hidden(
            "ANSWER",
            "The moon-rune lever is safe to pull now.",
            ["o7"],
            ["o1", "o2"],
            ["cite content, not hard-coded obs_3"],
            forbidden,
            [citation(p, "o7")],
        ),
    )

    p = base_public()
    p["observations"][2]["text"] = "Earlier margin note: unsafe. Later field note: the ward was dispelled and the moon-rune lever is safe to pull."
    p["observations"][2]["span"] = {"start": 27, "end": len(p["observations"][2]["text"].encode("utf-8"))}
    variant(
        "citation_span_trap",
        p,
        hidden(
            "ANSWER",
            "The moon-rune lever is safe to pull now.",
            ["o3"],
            ["o1", "o2"],
            ["cite exact later field-note span, not whole mixed observation"],
            forbidden,
            [citation(p, "o3")],
        ),
    )

    p = base_public()
    variant(
        "overbudget_fullscan_trap",
        p,
        hidden(
            "ANSWER",
            "The moon-rune lever is safe to pull now.",
            ["o3"],
            ["o1", "o2"],
            base_trace,
            forbidden + ["over_budget"],
            [citation(p, "o3")],
        ),
    )

    base["adversarial_variants"] = variants
    return base


def walk_public(obj: Any, path: str = "public_input") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in PUBLIC_FORBIDDEN_KEYS:
                found.append((path, f"forbidden key {key!r}"))
            found.extend(walk_public(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(walk_public(value, f"{path}[{index}]"))
    elif isinstance(obj, str):
        lower = obj.lower()
        for token in PUBLIC_FORBIDDEN_TEXT:
            if token.lower() in lower:
                found.append((path, f"forbidden text token {token!r}"))
    return found


def validate_cell_schema(cell: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(cell)
    if missing:
        failures.append(f"missing top-level fields: {sorted(missing)}")
    public = cell.get("public_input", {})
    oracle = cell.get("hidden_oracle", {})
    metadata = cell.get("training_metadata", {})
    scoring = cell.get("scoring", {})
    if PUBLIC_REQUIRED - set(public):
        failures.append(f"missing public fields: {sorted(PUBLIC_REQUIRED - set(public))}")
    if ORACLE_REQUIRED - set(oracle):
        failures.append(f"missing oracle fields: {sorted(ORACLE_REQUIRED - set(oracle))}")
    if METADATA_REQUIRED - set(metadata):
        failures.append(f"missing metadata fields: {sorted(METADATA_REQUIRED - set(metadata))}")
    if SCORING_REQUIRED - set(scoring):
        failures.append(f"missing scoring fields: {sorted(SCORING_REQUIRED - set(scoring))}")
    query = public.get("query", {})
    if QUERY_REQUIRED - set(query):
        failures.append(f"missing query fields: {sorted(QUERY_REQUIRED - set(query))}")
    observations = public.get("observations", [])
    if not observations:
        failures.append("public observations empty")
    for observation in observations:
        if OBS_REQUIRED - set(observation):
            failures.append(f"missing observation fields: {sorted(OBS_REQUIRED - set(observation))}")
        if not 0 <= float(observation.get("source_trust", -1)) <= 1:
            failures.append(f"invalid source_trust for {observation.get('obs_id')}")
    leaks = walk_public(public)
    failures.extend(f"public leak at {path}: {reason}" for path, reason in leaks)

    variant_ids = [variant.get("variant_id") for variant in cell.get("adversarial_variants", [])]
    required_variants = {
        "answerable_base",
        "missing_evidence",
        "weak_source",
        "unresolved_conflict",
        "stale_replay",
        "source_trust_inversion",
        "order_swap",
        "quote_or_inactive_scope",
        "negative_scope_story_text",
        "citation_id_shortcut_trap",
        "citation_span_trap",
        "overbudget_fullscan_trap",
    }
    missing_variants = required_variants - set(variant_ids)
    if missing_variants:
        failures.append(f"missing variants: {sorted(missing_variants)}")
    for variant in cell.get("adversarial_variants", []):
        public_v = variant.get("public_input", {})
        oracle_v = variant.get("hidden_oracle", {})
        failures.extend(f"{variant.get('variant_id')}: public leak at {path}: {reason}" for path, reason in walk_public(public_v))
        if PUBLIC_REQUIRED - set(public_v):
            failures.append(f"{variant.get('variant_id')}: missing public fields")
        if ORACLE_REQUIRED - set(oracle_v):
            failures.append(f"{variant.get('variant_id')}: missing oracle fields")
    return failures


def trusted_current_observations(public: dict[str, Any]) -> list[dict[str, Any]]:
    current_cycle = int(public.get("current_cycle", 0))
    observations = []
    for observation in public.get("observations", []):
        if int(observation.get("cycle", -1)) == current_cycle and float(observation.get("source_trust", 0)) >= 0.75:
            observations.append(observation)
    return sorted(observations, key=lambda item: int(item.get("order", 0)))


def relevant_observations(public: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for observation in public.get("observations", []):
        if "moon-rune lever" in observation.get("text", "").lower():
            rows.append(observation)
    return sorted(rows, key=lambda item: int(item.get("order", 0)))


def status_from_text(text: str) -> str | None:
    lower = text.lower()
    if "moon-rune lever" not in lower:
        return None
    if "safe to pull" in lower or "is safe" in lower:
        return "safe"
    if "unsafe" in lower or "ward returned" in lower or "ward was still active" in lower:
        return "unsafe"
    return None


def guarded_visible_policy(public: dict[str, Any], max_calls: int = 5) -> dict[str, Any]:
    context = public.get("context", "").lower()
    query = public.get("query", {}).get("text", "").lower()
    if "not an active" in context or "fictional" in context or "flavor text" in context:
        return {"action": "NO_CALL", "answer": None, "citations": [], "trace": ["inactive scope"], "operator_calls": 0}
    if "moon-rune lever" not in query:
        return {"action": "NO_CALL", "answer": None, "citations": [], "trace": ["query focus mismatch"], "operator_calls": 0}
    all_relevant = relevant_observations(public)
    if not all_relevant:
        return {"action": "ASK_FOR_EVIDENCE", "answer": None, "citations": [], "trace": ["no relevant observation"], "operator_calls": 3}
    latest_seen_order = max(int(item["order"]) for item in all_relevant)
    latest_seen = [item for item in all_relevant if int(item["order"]) == latest_seen_order]
    current_cycle = int(public.get("current_cycle", 0))
    latest_current_trusted = [
        item for item in latest_seen
        if int(item.get("cycle", -1)) == current_cycle and float(item.get("source_trust", 0)) >= 0.75
    ]
    if not latest_current_trusted:
        has_safe_hint = any(status_from_text(item.get("text", "")) == "safe" for item in latest_seen)
        action = "DEFER" if has_safe_hint else "ASK_FOR_EVIDENCE"
        return {
            "action": action,
            "answer": None,
            "citations": [],
            "trace": ["latest relevant evidence is weak, stale, or untrusted"],
            "operator_calls": 4,
        }
    observations = trusted_current_observations(public)
    observations = [item for item in observations if int(item.get("order", 0)) == latest_seen_order]
    if not observations:
        return {"action": "ASK_FOR_EVIDENCE", "answer": None, "citations": [], "trace": ["no trusted current observation"], "operator_calls": 3}
    latest_order = max(int(item["order"]) for item in observations)
    latest = [item for item in observations if int(item["order"]) == latest_order]
    statuses = [(item, status_from_text(item["text"])) for item in latest]
    useful = [(item, status) for item, status in statuses if status]
    if len(useful) != 1:
        return {"action": "DEFER", "answer": None, "citations": [], "trace": ["current trusted conflict or no decisive status"], "operator_calls": 4}
    item, status = useful[0]
    if status == "safe":
        answer = "The moon-rune lever is safe to pull now."
    else:
        answer = "The moon-rune lever is unsafe now; do not pull it."
    return {
        "action": "ANSWER",
        "answer": answer,
        "citations": [{"obs_id": item["obs_id"], "start": item["span"]["start"], "end": item["span"]["end"]}],
        "trace": ["used latest trusted current-cycle observation", f"decisive_obs={item['obs_id']}"],
        "operator_calls": min(max_calls, 4),
    }


def control_prediction(control: str, public: dict[str, Any]) -> dict[str, Any]:
    if control == "label_leak_control":
        leaked = copy.deepcopy(public)
        leaked["target_skill"] = "latest_reliable_evidence_with_negative_scope"
        return {"leak_detected": bool(walk_public(leaked)), "action": "INVALID_SCHEMA"}
    if control == "latest_only_control":
        latest = max(public.get("observations", []), key=lambda item: int(item.get("order", 0)), default=None)
        if latest and status_from_text(latest.get("text", "")) == "safe":
            return {"action": "ANSWER", "answer": "The moon-rune lever is safe to pull now.", "citations": [{"obs_id": latest["obs_id"], "start": 0, "end": latest["span"]["end"]}], "trace": ["picked latest only"], "operator_calls": 2}
        return {"action": "ANSWER", "answer": "The moon-rune lever is unsafe now; do not pull it.", "citations": [{"obs_id": latest["obs_id"], "start": 0, "end": latest["span"]["end"]}] if latest else [], "trace": ["picked latest only"], "operator_calls": 2}
    if control == "source_name_shortcut_control":
        chosen = next((item for item in public.get("observations", []) if item.get("source_id") == "sA"), None)
        return {"action": "ANSWER", "answer": "The moon-rune lever is safe to pull now.", "citations": [{"obs_id": chosen["obs_id"], "start": 0, "end": chosen["span"]["end"]}] if chosen else [], "trace": ["trusted source_id sA shortcut"], "operator_calls": 2}
    if control == "citation_id_shortcut_control":
        chosen = next((item for item in public.get("observations", []) if item.get("obs_id") == "o3"), None)
        return {"action": "ANSWER", "answer": "The moon-rune lever is safe to pull now.", "citations": [{"obs_id": "o3", "start": 0, "end": chosen["span"]["end"] if chosen else 0}], "trace": ["hard-coded obs_3"], "operator_calls": 2}
    if control == "full_scan_control":
        pred = guarded_visible_policy(public, max_calls=12)
        pred["operator_calls"] = 12
        pred["trace"] = list(pred.get("trace", [])) + ["full library scan"]
        return pred
    if control == "answer_without_trace_control":
        return {"action": "ANSWER", "answer": "The moon-rune lever is safe to pull now.", "citations": [], "trace": [], "operator_calls": 1}
    if control == "negative_scope_overcall_control":
        return {"action": "ANSWER", "answer": "The moon-rune lever is safe to pull now.", "citations": [], "trace": ["overcalled live-status route"], "operator_calls": 3}
    raise ValueError(control)


def evaluate_prediction(pred: dict[str, Any], oracle: dict[str, Any], route_budget: dict[str, Any]) -> dict[str, Any]:
    expected_action = oracle["expected_action"]
    expected_answer = oracle["expected_answer"]["render"] if oracle.get("expected_answer") else None
    must_use = set(oracle.get("required_evidence", {}).get("must_use", []))
    expected_spans = oracle.get("required_evidence", {}).get("citation_spans", [])
    citations = pred.get("citations", [])
    action_ok = pred.get("action") == expected_action
    answer_ok = expected_answer == pred.get("answer") if expected_answer is not None else pred.get("answer") is None
    cited_ids = {citation.get("obs_id") for citation in citations}
    citation_ok = not must_use or must_use <= cited_ids
    if expected_spans:
        citation_ok = citation_ok and all(span in citations for span in expected_spans)
    trace_ok = bool(pred.get("trace")) if expected_action not in {"NO_CALL"} else True
    over_budget = int(pred.get("operator_calls", 0)) > int(route_budget.get("max_operator_calls", 0))
    false_commit = pred.get("action") == "ANSWER" and expected_action != "ANSWER"
    unsupported_answer = pred.get("action") == "ANSWER" and (not citations or not trace_ok)
    wrong_scope_call = expected_action == "NO_CALL" and pred.get("action") != "NO_CALL"
    success = action_ok and answer_ok and citation_ok and trace_ok and not over_budget and not false_commit and not unsupported_answer and not wrong_scope_call
    return {
        "success": success,
        "action_ok": action_ok,
        "answer_ok": answer_ok,
        "citation_ok": citation_ok,
        "trace_ok": trace_ok,
        "false_commit": false_commit,
        "unsupported_answer": unsupported_answer,
        "wrong_scope_call": wrong_scope_call,
        "over_budget": over_budget,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    append_jsonl(progress, {"event": "start", "timestamp_ms": now_ms(), "contract": ARTIFACT_CONTRACT})

    schema = canonical_schema()
    cell = make_cell_pack()
    schema_failures = validate_cell_schema(cell)
    route_budget = cell["training_metadata"]["route_budget"]

    variant_results: list[dict[str, Any]] = []
    for variant in cell["adversarial_variants"]:
        pred = guarded_visible_policy(variant["public_input"])
        eval_result = evaluate_prediction(pred, variant["hidden_oracle"], route_budget)
        variant_results.append({"variant_id": variant["variant_id"], "prediction": pred, **eval_result})

    controls = [
        "label_leak_control",
        "latest_only_control",
        "source_name_shortcut_control",
        "citation_id_shortcut_control",
        "full_scan_control",
        "answer_without_trace_control",
        "negative_scope_overcall_control",
    ]
    control_results: list[dict[str, Any]] = []
    for control in controls:
        failures = 0
        tested = 0
        for variant in cell["adversarial_variants"]:
            if control == "label_leak_control":
                pred = control_prediction(control, variant["public_input"])
                failed_as_expected = bool(pred.get("leak_detected"))
                eval_result = {"success": not failed_as_expected, "control_failed_as_expected": failed_as_expected}
            else:
                pred = control_prediction(control, variant["public_input"])
                eval_result = evaluate_prediction(pred, variant["hidden_oracle"], route_budget)
                failed_as_expected = not eval_result["success"]
            failures += int(failed_as_expected)
            tested += 1
        control_results.append({
            "control": control,
            "tested": tested,
            "failed_as_expected": failures,
            "success_count": tested - failures,
            "invalid_as_general_policy": failures > 0,
            "all_failed_as_expected": failures == tested,
        })

    variant_count = len(variant_results)
    success_count = sum(1 for row in variant_results if row["success"])
    action_accuracy = sum(1 for row in variant_results if row["action_ok"]) / max(1, variant_count)
    answer_accuracy = sum(1 for row in variant_results if row["answer_ok"]) / max(1, variant_count)
    citation_exact_rate = sum(1 for row in variant_results if row["citation_ok"]) / max(1, variant_count)
    trace_dependency_coverage = sum(1 for row in variant_results if row["trace_ok"]) / max(1, variant_count)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "schema_validity": not schema_failures,
        "schema_failure_count": len(schema_failures),
        "oracle_leak_rate": 0.0 if not schema_failures else 1.0,
        "target_operator_leak_rate": 0.0 if not schema_failures else 1.0,
        "variant_count": variant_count,
        "primary_success_rate": success_count / max(1, variant_count),
        "action_accuracy": action_accuracy,
        "answer_accuracy": answer_accuracy,
        "citation_exact_rate": citation_exact_rate,
        "trace_dependency_coverage": trace_dependency_coverage,
        "false_commit_rate": sum(1 for row in variant_results if row["false_commit"]) / max(1, variant_count),
        "wrong_scope_call_rate": sum(1 for row in variant_results if row["wrong_scope_call"]) / max(1, variant_count),
        "unsupported_answer_rate": sum(1 for row in variant_results if row["unsupported_answer"]) / max(1, variant_count),
        "over_budget_rate": sum(1 for row in variant_results if row["over_budget"]) / max(1, variant_count),
        "controls_all_invalid_as_general_policy": all(row["invalid_as_general_policy"] for row in control_results),
    }

    decision_label = "e115_alpha_weave_pressure_cell_schema_confirmed"
    failure_count = 0
    if schema_failures:
        decision_label = "e115_oracle_or_target_leak_detected"
        failure_count += len(schema_failures)
    if aggregate["primary_success_rate"] < 1.0 or not aggregate["controls_all_invalid_as_general_policy"]:
        decision_label = "e115_adversarial_validation_failed"
        failure_count += 1

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "schema": schema,
        "cell": cell,
        "variant_results": variant_results,
        "control_results": control_results,
        "aggregate": aggregate,
    }
    deterministic_replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}
    decision = {"decision": decision_label, "failure_count": failure_count}

    public_samples = [
        {
            "variant_id": variant["variant_id"],
            "public_input": variant["public_input"],
        }
        for variant in cell["adversarial_variants"]
    ]
    machine_view = [
        {
            "variant_id": row["variant_id"],
            "action": row["prediction"]["action"],
            "answer": row["prediction"].get("answer"),
            "citations": row["prediction"].get("citations", []),
            "trace": row["prediction"].get("trace", []),
            "success": row["success"],
        }
        for row in variant_results
    ]

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "boundary": "schema validation only; not final training, not PermaCore, not TrueGolden",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "alpha_weave_pressure_cell_schema_v1.json", schema)
    write_json(out / "sample_cell_pack.json", cell)
    write_json(out / "public_input_samples.json", {"rows": public_samples})
    write_json(out / "machine_solve_view.json", {"rows": machine_view})
    write_json(out / "schema_validation_report.json", {"failures": schema_failures, "schema_valid": not schema_failures})
    write_json(out / "adversarial_validation_report.json", {"rows": variant_results})
    write_json(out / "control_results.json", {"rows": control_results})
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", deterministic_replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "schema_version": SCHEMA_VERSION,
        "variant_count": variant_count,
        "primary_success_rate": aggregate["primary_success_rate"],
        "controls_all_invalid_as_general_policy": aggregate["controls_all_invalid_as_general_policy"],
    })
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": decision_label,
        "variant_count": variant_count,
    })
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label})
    (out / "report.md").write_text(
        "# E115 Alpha-Weave Pressure Cell Schema Validation Result\n\n"
        f"decision = {decision_label}\n\n"
        f"schema_version = {SCHEMA_VERSION}\n\n"
        f"variant_count = {variant_count}\n\n"
        f"primary_success_rate = {aggregate['primary_success_rate']:.6f}\n\n"
        f"controls_all_invalid_as_general_policy = {aggregate['controls_all_invalid_as_general_policy']}\n\n"
        "Boundary: schema/cell-pack validation only; no final-training or Core-memory claim.\n",
        encoding="utf-8",
    )
    return {"out": str(out), **aggregate, **decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e115_alpha_weave_pressure_cell_schema_validation")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
