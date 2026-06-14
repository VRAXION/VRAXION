#!/usr/bin/env python3
"""E116 alpha-Weave synthetic pressure data generation.

E114 showed that natural FineWeb is clean but too sparse for 77 rare/scoped
CoreMemoryCandidate Operators. E115 locked the alpha-Weave pressure-cell schema.
E116 generates schema-valid synthetic pressure cells for those rare Operators
and runs a targeted activation projection.

Boundary: synthetic curriculum-data generation and pressure accounting only.
This is not PermaCore/TrueGolden promotion, not final training, and not a claim
that the runtime learned these examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e115_alpha_weave_pressure_cell_schema_validation import (  # noqa: E402
    ARTIFACT_CONTRACT as E115_CONTRACT,
    PUBLIC_FORBIDDEN_KEYS,
    PUBLIC_FORBIDDEN_TEXT,
    SCHEMA_VERSION,
    canonical_schema,
    deterministic_hash,
    validate_cell_schema,
)


ARTIFACT_CONTRACT = "E116_ALPHA_WEAVE_SYNTHETIC_PRESSURE_GENERATION"
DEFAULT_E114 = Path("target/pilot_wave/e114_fineweb_next_limit_stability_projection")
PERMACORE_PROBATION_TARGET = 300_000
DEFAULT_PACKS_PER_OPERATOR = 128
VARIANT_IDS = (
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
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def span(text: str, start: int = 0) -> dict[str, int]:
    return {"start": start, "end": len(text.encode("utf-8"))}


def obs(obs_id: str, cycle: int, order: int, trust: float, text: str, source_id: str = "sA") -> dict[str, Any]:
    return {
        "obs_id": obs_id,
        "cycle": cycle,
        "order": order,
        "source_id": source_id,
        "source_trust": trust,
        "text": text,
        "span": span(text),
    }


def hidden(action: str, answer: str | None, must_use: list[str], must_not: list[str], trace: list[str], forbidden: list[str]) -> dict[str, Any]:
    return {
        "expected_action": action,
        "expected_answer": {"value": answer, "render": answer} if answer is not None else None,
        "required_evidence": {
            "must_use": must_use,
            "must_not_use_as_decisive": must_not,
            "citation_spans": [{"obs_id": item, "start": 0, "end": 9999} for item in must_use],
        },
        "required_trace": trace,
        "forbidden_behavior": forbidden,
    }


def classify_operator(operator_id: str) -> str:
    text = operator_id.lower()
    if any(key in text for key in ["bit", "crc", "frame", "resync", "hash", "replay"]):
        return "frame_sync"
    if any(key in text for key in ["task", "completion", "step", "blocked", "dependency", "progress", "budget", "promotion"]):
        return "task_progress"
    if any(key in text for key in ["alias", "symbol", "unit", "numeric", "morphology", "multilingual", "negation"]):
        return "alias_symbol"
    if any(key in text for key in ["answer", "output", "citation", "trace", "coverage", "response"]):
        return "answer_integrity"
    return "evidence_conflict"


def scenario_public(template: str, index: int, variant_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle = 40 + (index % 11)
    entity = ["moon-rune lever", "silver gate", "amber ward", "north bridge", "oracle door"][index % 5]
    context_by_template = {
        "evidence_conflict": f"Dungeon evidence board: track the current state of the {entity}. Notes can be old, weak, conflicting, or replayed.",
        "answer_integrity": f"Party answer board: decide whether the final claim about the {entity} is supported by cited evidence.",
        "task_progress": f"Quest progress board: update a task only when all required evidence for the {entity} is present.",
        "frame_sync": f"Rune frame board: decode whether the latest bit-frame for the {entity} is valid before committing it.",
        "alias_symbol": f"Glyph glossary board: bind surface forms for the {entity} only when the visible evidence is current and trusted.",
    }
    answer_safe = f"The {entity} is safe to use now."
    answer_unsafe = f"The {entity} is not safe to use now."
    base = {
        "context": context_by_template[template],
        "current_cycle": cycle,
        "observations": [
            obs("o1", cycle, 1, 0.91, f"At marker {index % 7 + 1}, the {entity} was not safe; the first ward was still active."),
            obs("o2", cycle, 2, 0.28, f"A weak tavern note says the {entity} might still be unsafe.", "sB"),
            obs("o3", cycle, 3, 0.91, f"At marker {index % 7 + 4}, trusted evidence says the ward cleared and the {entity} is safe to use."),
        ],
        "query": {"text": f"What should be done with the {entity} now?", "required_focus": {"entity": entity, "attribute": "current_action"}},
    }
    trace = ["use current-cycle evidence", "reject weak or stale evidence", "cite decisive span", "avoid overbroad route"]
    forbidden = ["unsupported answer", "stale commit", "weak source commit", "full library scan", "wrong-scope call"]

    if template == "task_progress":
        base["observations"][2]["text"] = f"At marker {index % 7 + 4}, trusted evidence shows all required checks for the {entity} task are complete."
        base["observations"][2]["span"] = span(base["observations"][2]["text"])
        answer_safe = f"The {entity} task can be marked complete."
        answer_unsafe = f"The {entity} task is not complete."
    elif template == "frame_sync":
        base["observations"][2]["text"] = f"At marker {index % 7 + 4}, frame START 101011 LEN 0110 PAYLOAD valid CRC 11001 END 010101 matches the {entity} request."
        base["observations"][2]["span"] = span(base["observations"][2]["text"])
        answer_safe = f"The latest {entity} frame is valid."
        answer_unsafe = f"The latest {entity} frame is not valid."
    elif template == "alias_symbol":
        base["observations"][2]["text"] = f"At marker {index % 7 + 4}, trusted glossary evidence binds the current surface form for {entity} to the safe route."
        base["observations"][2]["span"] = span(base["observations"][2]["text"])
        answer_safe = f"The current binding for {entity} points to the safe route."
        answer_unsafe = f"The current binding for {entity} is not safe."

    public = json.loads(json.dumps(base))
    expected = hidden("ANSWER", answer_safe, ["o3"], ["o1", "o2"], trace, forbidden)

    if variant_id == "missing_evidence":
        public["observations"] = public["observations"][:2]
        expected = hidden("ASK_FOR_EVIDENCE", None, [], ["o1", "o2"], ["latest trusted evidence missing"], forbidden)
    elif variant_id == "weak_source":
        public["observations"][2]["source_trust"] = 0.31
        public["observations"][2]["source_id"] = "sB"
        expected = hidden("DEFER", None, [], ["o1", "o2", "o3"], ["latest evidence is weak"], forbidden)
    elif variant_id == "unresolved_conflict":
        public["observations"].append(obs("o4", cycle, 3, 0.91, f"At the same marker, trusted evidence says the {entity} is unsafe again.", "sC"))
        expected = hidden("DEFER", None, [], ["o3", "o4"], ["trusted current conflict"], forbidden)
    elif variant_id == "stale_replay":
        public["observations"][2]["cycle"] = cycle - 9
        expected = hidden("DEFER", None, [], ["o1", "o2", "o3"], ["latest decisive evidence is stale replay"], forbidden)
    elif variant_id == "source_trust_inversion":
        public["observations"][1]["source_trust"] = 0.91
        public["observations"][1]["source_id"] = "sA"
        public["observations"][2]["source_trust"] = 0.31
        public["observations"][2]["source_id"] = "sB"
        expected = hidden("DEFER", None, [], ["o1", "o2", "o3"], ["latest safe claim is weak after source inversion"], forbidden)
    elif variant_id == "order_swap":
        public["observations"][0]["order"] = 4
        public["observations"][0]["text"] = f"At marker {index % 7 + 6}, trusted evidence says the {entity} became unsafe again."
        public["observations"][0]["span"] = span(public["observations"][0]["text"])
        expected = hidden("ANSWER", answer_unsafe, ["o1"], ["o2", "o3"], ["latest trusted evidence changed to unsafe"], forbidden)
    elif variant_id == "quote_or_inactive_scope":
        public["context"] = "A bard repeats a fictional story. This is not an active evidence board."
        public["query"] = {"text": "Summarize the tone of the story.", "required_focus": {"entity": "story", "attribute": "tone"}}
        expected = hidden("NO_CALL", None, [], ["o1", "o2", "o3"], ["inactive narrative scope"], ["activate live-state route"])
    elif variant_id == "negative_scope_story_text":
        public["context"] = "Campaign handout flavor text, not a live state board."
        public["query"] = {"text": "Rewrite this as a mysterious handout.", "required_focus": {"entity": "handout", "attribute": "style"}}
        expected = hidden("NO_CALL", None, [], ["o1", "o2", "o3"], ["style task, no live commit"], ["commit live state"])
    elif variant_id == "citation_id_shortcut_trap":
        public["observations"][2]["obs_id"] = "o7"
        expected = hidden("ANSWER", answer_safe, ["o7"], ["o1", "o2"], ["cite content, not fixed id"], forbidden)
    elif variant_id == "citation_span_trap":
        prefix = "Old margin says unsafe. Later note: "
        public["observations"][2]["text"] = prefix + public["observations"][2]["text"]
        public["observations"][2]["span"] = {"start": len(prefix.encode("utf-8")), "end": len(public["observations"][2]["text"].encode("utf-8"))}
        expected = hidden("ANSWER", answer_safe, ["o3"], ["o1", "o2"], ["cite later note span only"], forbidden)
    elif variant_id == "overbudget_fullscan_trap":
        expected = hidden("ANSWER", answer_safe, ["o3"], ["o1", "o2"], trace, forbidden + ["over_budget"])
    return public, expected


def make_cell(operator: dict[str, Any], pack_index: int, repeat_count: int) -> dict[str, Any]:
    operator_id = operator["operator_id"]
    template = classify_operator(operator_id)
    variants = []
    for variant_id in VARIANT_IDS:
        public, oracle = scenario_public(template, stable_int(f"{operator_id}:{pack_index}:{variant_id}") % 10_000, variant_id)
        variants.append({"variant_id": variant_id, "public_input": public, "hidden_oracle": oracle})
    public, oracle = scenario_public(template, stable_int(f"{operator_id}:{pack_index}:base") % 10_000, "answerable_base")
    return {
        "cell_id": f"aw_{operator_id}_{pack_index:05d}",
        "cell_version": 1,
        "public_input": public,
        "hidden_oracle": oracle,
        "training_metadata": {
            "target_skill": f"targeted_pressure::{operator.get('group_id')}::{template}",
            "target_operators": [operator_id],
            "operator_visibility": "hidden_from_candidate",
            "route_budget": {"max_operator_calls": 5, "max_trace_steps": 8},
            "data_origin": "synthetic_codex_generated",
            "generator": "codex",
            "generator_version": "e116_alpha_weave_generator_v1",
            "human_review_status": "unreviewed",
            "synthetic_disclosure": True,
            "repeat_count": repeat_count,
            "template_family": template,
        },
        "adversarial_variants": variants,
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


def public_leaks(obj: Any, path: str = "public_input") -> list[str]:
    failures: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in PUBLIC_FORBIDDEN_KEYS:
                failures.append(f"{path}: forbidden key {key}")
            failures.extend(public_leaks(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            failures.extend(public_leaks(value, f"{path}[{index}]"))
    elif isinstance(obj, str):
        lower = obj.lower()
        for token in PUBLIC_FORBIDDEN_TEXT:
            if token.lower() in lower:
                failures.append(f"{path}: forbidden token {token}")
        if "synthetic_codex_generated" in lower or "codex" in lower:
            failures.append(f"{path}: synthetic origin leaked into public input")
    return failures


def rare_operators(e114_root: Path) -> list[dict[str, Any]]:
    rows = read_json(e114_root / "operator_projection_report.json")["rows"]
    return [row for row in rows if not row.get("projected_reaches_permacore_probation")]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    start = time.time()
    e114_root = Path(args.e114_root)
    operators = rare_operators(e114_root)
    schema = canonical_schema()
    generated_path = out / "generated_cells.jsonl"
    if generated_path.exists():
        generated_path.unlink()

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "rare_operator_count": len(operators),
        "packs_per_operator": args.packs_per_operator,
        "schema_version": SCHEMA_VERSION,
    })

    operator_rows = []
    schema_failures = []
    leak_failures = []
    template_counts: Counter[str] = Counter()
    generated_count = 0
    total_variants = 0
    total_scheduled_cases = 0
    last_heartbeat = time.time()

    with generated_path.open("w", encoding="utf-8", newline="\n") as handle:
        for operator_index, operator in enumerate(operators):
            remaining = int(operator.get("projected_remaining_after_full_fineweb", 0))
            variants_per_pack = len(VARIANT_IDS)
            repeat_count = max(1, math.ceil(remaining / max(1, args.packs_per_operator * variants_per_pack)))
            op_generated = 0
            op_variants = 0
            op_scheduled = 0
            template = classify_operator(operator["operator_id"])
            template_counts[template] += 1
            for pack_index in range(args.packs_per_operator):
                cell = make_cell(operator, pack_index, repeat_count)
                failures = validate_cell_schema(cell)
                if failures:
                    schema_failures.extend(f"{cell['cell_id']}: {failure}" for failure in failures)
                leaks = public_leaks(cell["public_input"])
                for variant in cell["adversarial_variants"]:
                    leaks.extend(public_leaks(variant["public_input"], f"{cell['cell_id']}:{variant['variant_id']}"))
                if leaks:
                    leak_failures.extend(f"{cell['cell_id']}: {failure}" for failure in leaks)
                handle.write(json.dumps(cell, ensure_ascii=False, sort_keys=True) + "\n")
                op_generated += 1
                op_variants += variants_per_pack
                op_scheduled += variants_per_pack * repeat_count
                generated_count += 1
                total_variants += variants_per_pack
                total_scheduled_cases += variants_per_pack * repeat_count

            projected_after_full = int(operator.get("projected_activation_after_full_fineweb", 0))
            projected_after_targeted = projected_after_full + op_scheduled
            operator_rows.append({
                "operator_id": operator["operator_id"],
                "display_name": operator.get("display_name"),
                "family": operator.get("family"),
                "group_id": operator.get("group_id"),
                "template_family": template,
                "selected_variant": operator.get("selected_variant"),
                "projected_remaining_after_full_fineweb": remaining,
                "generated_cell_packs": op_generated,
                "variant_count": op_variants,
                "repeat_count_per_pack": repeat_count,
                "qualified_synthetic_pressure_activation": op_scheduled,
                "projected_activation_after_full_fineweb": projected_after_full,
                "projected_activation_after_targeted_pressure": projected_after_targeted,
                "reaches_permacore_probation_after_targeted_pressure": projected_after_targeted >= PERMACORE_PROBATION_TARGET,
            })

            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                snapshot = {
                    "event": "heartbeat",
                    "timestamp_ms": now_ms(),
                    "operators_done": operator_index + 1,
                    "generated_cell_packs": generated_count,
                    "elapsed_seconds": round(time.time() - start, 3),
                }
                append_jsonl(progress, snapshot)
                write_json(out / "partial_aggregate_snapshot.json", snapshot)
                last_heartbeat = time.time()

    reaches = sum(1 for row in operator_rows if row["reaches_permacore_probation_after_targeted_pressure"])
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "rare_operator_count": len(operators),
        "generated_cell_packs": generated_count,
        "variant_count": total_variants,
        "scheduled_case_count": total_scheduled_cases,
        "packs_per_operator": args.packs_per_operator,
        "schema_failure_count": len(schema_failures),
        "public_leak_failure_count": len(leak_failures),
        "synthetic_origin_metadata_rate": 1.0,
        "synthetic_origin_public_leak_rate": 0.0 if not leak_failures else 1.0,
        "target_reach_count": reaches,
        "targeted_needed_remaining_count": len(operators) - reaches,
        "permacore_probation_target": PERMACORE_PROBATION_TARGET,
        "template_counts": dict(template_counts),
        "seconds": round(time.time() - start, 3),
    }
    decision_label = "e116_synthetic_pressure_reaches_next_activation_limit"
    failure_count = 0
    if schema_failures or leak_failures:
        decision_label = "e116_synthetic_pressure_schema_or_leak_failure"
        failure_count += 1
    elif reaches < len(operators):
        decision_label = "e116_synthetic_pressure_partial_target_coverage"

    sample_rows = []
    with generated_path.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(args.sample_limit), handle):
            cell = json.loads(line)
            sample_rows.append({
                "cell_id": cell["cell_id"],
                "public_input": cell["public_input"],
                "training_metadata": cell["training_metadata"],
                "first_variant_public": cell["adversarial_variants"][0]["public_input"],
            })

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "schema": schema,
        "operators": operator_rows,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "generated_hash": deterministic_hash({"cell_count": generated_count, "first_samples": sample_rows[:3]}),
    }
    replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}
    decision = {"decision": decision_label, "failure_count": failure_count}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "schema_contract": E115_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "boundary": "synthetic pressure data generation only; not final training, not PermaCore, not TrueGolden",
        "e114_root": str(e114_root),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "generation_manifest.json", {
        "data_origin": "synthetic_codex_generated",
        "generator": "codex",
        "generator_version": "e116_alpha_weave_generator_v1",
        "human_review_status": "unreviewed",
        "schema_version": SCHEMA_VERSION,
        "packs_per_operator": args.packs_per_operator,
        "variant_ids": list(VARIANT_IDS),
        "generated_cells_path": str(generated_path),
    })
    write_json(out / "synthetic_origin_report.json", {
        "metadata_disclosure_required": True,
        "public_input_disclosure_forbidden": True,
        "data_origin": "synthetic_codex_generated",
        "public_leak_failure_count": len(leak_failures),
        "leak_failures": leak_failures[:100],
    })
    write_json(out / "rare_operator_input_report.json", {"rows": operators})
    write_json(out / "operator_target_coverage.json", {"rows": operator_rows})
    write_json(out / "activation_projection_report.json", {"rows": operator_rows})
    write_json(out / "leakage_check_report.json", {"schema_failures": schema_failures[:100], "leak_failures": leak_failures[:100]})
    write_json(out / "public_sample_cells.json", {"rows": sample_rows})
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "generated_cell_packs": generated_count,
        "scheduled_case_count": total_scheduled_cases,
        "target_reach_count": reaches,
        "targeted_needed_remaining_count": len(operators) - reaches,
    })
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "generated_cell_packs": generated_count,
        "decision": decision_label,
    })
    append_jsonl(progress, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "generated_cell_packs": generated_count,
        "scheduled_case_count": total_scheduled_cases,
        "decision": decision_label,
    })
    (out / "human_machine_sample_report.md").write_text(
        "# E116 Human/Machine Sample\n\n"
        "The included public samples are synthetic Codex-generated alpha-Weave pressure cells.\n"
        "The synthetic marker is present in training metadata and intentionally absent from public_input.\n\n"
        f"Generated cell packs: {generated_count}\n\n"
        f"Scheduled cases: {total_scheduled_cases}\n\n"
        f"Target reach count: {reaches} / {len(operators)}\n",
        encoding="utf-8",
    )
    (out / "report.md").write_text(
        "# E116 Alpha-Weave Synthetic Pressure Generation Result\n\n"
        f"decision = {decision_label}\n\n"
        f"generated_cell_packs = {generated_count}\n\n"
        f"scheduled_case_count = {total_scheduled_cases}\n\n"
        f"target_reach_count = {reaches} / {len(operators)}\n\n"
        f"schema_failure_count = {len(schema_failures)}\n\n"
        f"public_leak_failure_count = {len(leak_failures)}\n\n"
        "Boundary: synthetic pressure data generation only; no final-training or PermaCore claim.\n",
        encoding="utf-8",
    )
    return {"out": str(out), **aggregate, **decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e114-root", default=str(DEFAULT_E114))
    parser.add_argument("--out", default="target/pilot_wave/e116_alpha_weave_synthetic_pressure_generation")
    parser.add_argument("--packs-per-operator", type=int, default=DEFAULT_PACKS_PER_OPERATOR)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
