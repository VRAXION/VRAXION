#!/usr/bin/env python3
"""E119 FineWeb skill mining and text-IO delta probe.

E119 is not final training and not a Gemma-style language model claim. It uses
the governed E118 Operator Library as a callable skill set over real FineWeb-Edu
rows and asks two narrower questions:

1. Which new operator/skill candidates show up repeatedly in real text?
2. Does the current E118 library produce better safe text IO decisions than a
   legacy grounding-only subset?

The "output" measured here is canonical action rendering (observe/defer/ask/
answer-with-trace), not free-form neural text generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e113_fineweb_light_stress_hard_mutation_recycle import row_features  # noqa: E402


ARTIFACT_CONTRACT = "E119_FINEWEB_SKILL_MINING_AND_TEXT_IO_DELTA_PROBE"
DEFAULT_DATASET = Path("data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl")
DEFAULT_E118 = Path("target/pilot_wave/e118_core_candidate_cross_source_no_harm_gauntlet")
DEFAULT_E83 = Path("target/pilot_wave/e83_calc_scribe_v003_local_golden_promotion_reload")

REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "dataset_report.json",
    "operator_source_report.json",
    "skill_candidate_report.json",
    "text_io_delta_report.json",
    "generation_readiness_report.json",
    "row_level_samples.jsonl",
    "skill_candidate_examples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
)


DEFINITION_RE = re.compile(r"\b(?:is|are|means|refers to|is called|defined as|known as)\b", re.IGNORECASE)
CAUSAL_RE = re.compile(r"\b(?:because|therefore|due to|leads to|results in|causes|caused by|as a result)\b", re.IGNORECASE)
COMPARISON_RE = re.compile(r"\b(?:more than|less than|greater than|fewer than|compared with|versus|vs\.?|higher|lower)\b", re.IGNORECASE)
PROCEDURE_RE = re.compile(r"\b(?:first|second|third|finally|step|steps|how to|instructions|procedure)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", re.IGNORECASE)
ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b")
QUOTE_SPEAKER_RE = re.compile(r"[\"“].{8,220}?[\"”]\s*(?:said|says|wrote|according to|replied)|(?:said|says|wrote)\s+[A-Z][a-z]+", re.IGNORECASE | re.DOTALL)
HEDGE_RE = re.compile(r"\b(?:may|might|could|possibly|probably|appears|suggests|uncertain|likely|unlikely)\b", re.IGNORECASE)
SAFETY_RE = re.compile(r"\b(?:doctor|medical|medicine|HIV|AIDS|dose|treatment|legal|lawyer|financial|investment|suicide|self-harm)\b", re.IGNORECASE)
STRUCTURE_RE = re.compile(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+|#{1,4}\s|\b(?:chapter|section|table of contents)\b", re.IGNORECASE)
NUMBER_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|km|m|cm|mm|mph|km/h|percent|%|dollars?|USD|miles?|feet|ft)\b", re.IGNORECASE)


SKILL_CANDIDATE_SPECS: dict[str, dict[str, Any]] = {
    "definition_term_anchor_lens": {
        "title": "Definition / Term Anchor Lens",
        "pattern": DEFINITION_RE,
        "need": "extract term-definition bindings without treating every statement as final truth",
        "covered_by": {"visible_claim_binding_alpha_syncer", "canonical_lexeme_scribe"},
    },
    "causal_relation_lens": {
        "title": "Causal Relation Lens",
        "pattern": CAUSAL_RE,
        "need": "capture because/due-to/result relations as traceable links",
        "covered_by": set(),
    },
    "comparison_quantifier_guard": {
        "title": "Comparison / Quantifier Guard",
        "pattern": COMPARISON_RE,
        "need": "protect greater/less/versus statements from sign or scope flips",
        "covered_by": {"numeric_value_binding_alpha_syncer", "symbol_equivalence_guard"},
    },
    "procedure_step_parser_lens": {
        "title": "Procedure Step Parser Lens",
        "pattern": PROCEDURE_RE,
        "need": "split instructional text into ordered, evidence-backed steps",
        "covered_by": {"task_requirement_decomposition_lens", "step_status_transition_guard"},
    },
    "date_entity_timeline_lens": {
        "title": "Date / Entity Timeline Lens",
        "pattern": DATE_RE,
        "need": "bind named events to dates and keep latest/earlier ordering stable",
        "covered_by": {"temporal_latest_span_t_stab", "frame_sequence_t_stab"},
    },
    "quote_speaker_attribution_lens": {
        "title": "Quote Speaker Attribution Lens",
        "pattern": QUOTE_SPEAKER_RE,
        "need": "bind quoted spans to the right speaker/source",
        "covered_by": {"source_attribution_lens", "quote_scope_boundary_guard"},
    },
    "hedge_uncertainty_strength_t_stab": {
        "title": "Hedge / Uncertainty Strength T-Stab",
        "pattern": HEDGE_RE,
        "need": "preserve may/might/likely uncertainty instead of overcommitting",
        "covered_by": {"weak_claim_uncertainty_t_stab", "unsupported_answer_defer_guard"},
    },
    "safety_domain_caution_guard": {
        "title": "Safety Domain Caution Guard",
        "pattern": SAFETY_RE,
        "need": "route medical/legal/financial text to cautious answer/defer behavior",
        "covered_by": set(),
    },
    "document_structure_lens": {
        "title": "Document Structure Lens",
        "pattern": STRUCTURE_RE,
        "need": "understand headings, bullets, and document sections as layout evidence",
        "covered_by": {"summary_relevance_span_selector_lens", "task_requirement_decomposition_lens"},
    },
    "number_unit_grounding_alpha_syncer": {
        "title": "Number + Unit Grounding Alpha-Syncer",
        "pattern": NUMBER_UNIT_RE,
        "need": "ground quantities with units before answer rendering",
        "covered_by": {"unit_code_alpha_syncer", "unit_preserving_answer_scribe"},
    },
    "named_entity_anchor_lens": {
        "title": "Named Entity Anchor Lens",
        "pattern": ENTITY_RE,
        "need": "stabilize person/org/place names as reusable Ground anchors",
        "covered_by": set(),
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS + ("checker_summary.json",):
        path = out / name
        if path.exists():
            path.unlink()


def iter_dataset(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                yield index, json.loads(line)


def load_e118_operators(root: Path) -> list[dict[str, Any]]:
    rows = read_json(root / "operator_cross_source_results.json")["rows"]
    rows = sorted(rows, key=lambda row: row["operator_id"])
    for row in rows:
        row["_tokens"] = set(re.findall(r"[a-z0-9]+", str(row["operator_id"]).lower()))
    return rows


def load_calc_scribe_status(root: Path) -> dict[str, Any]:
    registry = root / "pocket_library" / "calc_scribe_v003" / "registry.json"
    if not registry.exists():
        return {"available": False}
    payload = read_json(registry)
    return {
        "available": True,
        "pocket_uid": payload.get("pocket_uid", "calc_scribe_v003"),
        "scope": payload.get("scope", "visible_calc_trace_validator"),
        "lifecycle": payload.get("lifecycle", "LocalGolden"),
    }


def operator_tokens(row: dict[str, Any]) -> set[str]:
    if "_tokens" in row:
        return row["_tokens"]
    return set(re.findall(r"[a-z0-9]+", str(row["operator_id"]).lower()))


def operator_relevance(row: dict[str, Any], features: dict[str, Any], text: str) -> bool:
    tokens = operator_tokens(row)
    family = row.get("family", "")
    group = row.get("group_id", "")
    if features["has_calc"] and ("numeric" in tokens or "calc" in tokens or "unit" in tokens):
        return True
    if features["has_question"] and ("answer" in tokens or "question" in tokens or "requirement" in tokens):
        return True
    if features["has_unresolved"] and ("unresolved" in tokens or "missing" in tokens or "ask" in tokens):
        return True
    if features["has_contradiction"] and ("contradiction" in tokens or "conflict" in tokens or "defer" in tokens):
        return True
    if features["has_temporal"] and ("temporal" in tokens or "latest" in tokens or "stale" in tokens or "turn" in tokens):
        return True
    if features["evidence_like"] and ("evidence" in tokens or "source" in tokens or "citation" in tokens or "trace" in tokens):
        return True
    if features["has_task"] and ("task" in tokens or "progress" in tokens or "completion" in tokens or "step" in tokens):
        return True
    if features["has_list"] and ("summary" in tokens or "decomposition" in tokens or "list" in tokens):
        return True
    if features["long_text"] and ("summary" in tokens or "context" in tokens or "compression" in tokens):
        return True
    if features.get("has_date") and ("temporal" in tokens or "date" in tokens or "latest" in tokens):
        return True
    if features.get("has_number_unit") and ("unit" in tokens or "numeric" in tokens):
        return True
    if features.get("has_hedge") and ("uncertainty" in tokens or "unsupported" in tokens or "defer" in tokens):
        return True
    if family == "Guard" and group in {"E93", "E94"} and (features["task_like"] or features["evidence_like"]):
        return True
    return False


def classify_expected(features: dict[str, Any], text: str) -> str:
    if features["has_adversarial"]:
        return "DEFER_UNSAFE"
    if features["has_calc"]:
        return "VALIDATE_VISIBLE_CALC_TRACE"
    if features["has_question"] and (features["has_unresolved"] or not features["evidence_like"]):
        return "ASK_FOR_EVIDENCE"
    if features["has_question"] and features["evidence_like"]:
        return "ANSWER_WITH_TRACE"
    if features["has_contradiction"]:
        return "DEFER_CONFLICT"
    if features["has_task"]:
        return "UPDATE_PROGRESS_LEDGER"
    if features["evidence_like"] or features["has_temporal"] or features["has_list"] or HEDGE_RE.search(text):
        return "OBSERVE_AND_GROUND"
    return "NO_TASK_NO_COMMIT"


def render_action(expected: str, active_names: list[str], features: dict[str, Any]) -> str:
    if expected == "ANSWER_WITH_TRACE":
        return "ANSWER_WITH_TRACE(evidence_span, citation, uncertainty_guard)"
    if expected == "ASK_FOR_EVIDENCE":
        return "ASK_FOR_EVIDENCE(missing_dependency)"
    if expected == "DEFER_CONFLICT":
        return "DEFER(conflicting_evidence)"
    if expected == "DEFER_UNSAFE":
        return "DEFER(unsafe_or_adversarial_text)"
    if expected == "VALIDATE_VISIBLE_CALC_TRACE":
        return "CALL(CALC-SCRIBE visible_calc_trace_validator)"
    if expected == "UPDATE_PROGRESS_LEDGER":
        return "UPDATE_PROGRESS_LEDGER(evidence_backed_step_status)"
    if expected == "OBSERVE_AND_GROUND":
        return "OBSERVE_AND_GROUND(traceable_span)"
    return "NO_CALL(no task/evidence requiring commit)"


def action_success(expected: str, operators: list[dict[str, Any]], features: dict[str, Any], text: str, mode: str) -> tuple[bool, list[str], str]:
    if mode == "none":
        return expected == "NO_TASK_NO_COMMIT", [], "NO_CALL" if expected == "NO_TASK_NO_COMMIT" else "NO_CAPABILITY"

    if mode == "legacy":
        allowed_groups = {"E90", "E91", "E92"}
    else:
        allowed_groups = None
    active = [
        row for row in operators
        if (allowed_groups is None or row.get("group_id") in allowed_groups)
        and operator_relevance(row, features, text)
    ]
    active_ids = [row["operator_id"] for row in active]
    active_tokens = set()
    for row in active:
        active_tokens.update(operator_tokens(row))

    if expected == "NO_TASK_NO_COMMIT":
        return True, active_ids, "NO_CALL"
    if expected == "VALIDATE_VISIBLE_CALC_TRACE":
        ok = mode == "current" and features["has_calc"]
    elif expected == "ANSWER_WITH_TRACE":
        ok = {"answer", "evidence", "citation", "unsupported"}.intersection(active_tokens) and mode == "current"
    elif expected == "ASK_FOR_EVIDENCE":
        ok = {"ask", "missing", "unresolved", "dependency"}.intersection(active_tokens) and mode == "current"
    elif expected == "DEFER_CONFLICT":
        ok = {"conflict", "contradiction", "defer"}.intersection(active_tokens)
    elif expected == "DEFER_UNSAFE":
        ok = {"scope", "guard", "unsafe", "adversarial"}.intersection(active_tokens) or any(row.get("family") == "Guard" for row in active)
    elif expected == "UPDATE_PROGRESS_LEDGER":
        ok = {"task", "progress", "completion", "step"}.intersection(active_tokens) and mode == "current"
    else:
        ok = bool(active)
    return bool(ok), active_ids, render_action(expected, active_ids, features)


def matched_candidate_specs(text: str) -> list[str]:
    sample = text[:2000]
    return [key for key, spec in SKILL_CANDIDATE_SPECS.items() if spec["pattern"].search(sample)]


def current_coverage_for_candidate(candidate_id: str, active_operator_ids: list[str]) -> float:
    covered_by: set[str] = SKILL_CANDIDATE_SPECS[candidate_id]["covered_by"]
    if not covered_by:
        return 0.0
    hits = len(covered_by.intersection(active_operator_ids))
    return min(1.0, hits / len(covered_by))


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    start = time.time()
    dataset = Path(args.dataset)
    operators = load_e118_operators(Path(args.e118_root))
    calc_scribe = load_calc_scribe_status(Path(args.e83_root))

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "dataset": str(dataset),
        "row_limit": args.limit,
        "operator_count": len(operators),
    })

    expected_counter: Counter[str] = Counter()
    mode_success: dict[str, Counter[str]] = defaultdict(Counter)
    current_hard_negative = 0
    candidate_stats: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_samples: list[dict[str, Any]] = []
    family_counter: Counter[str] = Counter()
    rows_seen = 0

    for index, row in iter_dataset(dataset, args.limit):
        rows_seen += 1
        features = row_features(row)
        text = str(row.get("text", ""))
        feature_sample = text[:2000]
        features["has_date"] = bool(DATE_RE.search(feature_sample))
        features["has_number_unit"] = bool(NUMBER_UNIT_RE.search(feature_sample))
        features["has_hedge"] = bool(HEDGE_RE.search(feature_sample))
        expected = classify_expected(features, text)
        expected_counter[expected] += 1
        for flag in ("has_question", "has_calc", "has_contradiction", "has_temporal", "has_evidence", "has_unresolved", "has_task", "long_text"):
            if features.get(flag):
                family_counter[flag] += 1

        none_ok, none_active, none_action = action_success(expected, operators, features, text, "none")
        legacy_ok, legacy_active, legacy_action = action_success(expected, operators, features, text, "legacy")
        current_ok, current_active, current_action = action_success(expected, operators, features, text, "current")
        if expected in {"DEFER_UNSAFE", "NO_TASK_NO_COMMIT"} and current_action.startswith("ANSWER"):
            current_hard_negative += 1

        for mode, ok in (("none", none_ok), ("legacy", legacy_ok), ("current", current_ok)):
            mode_success[mode]["total"] += 1
            mode_success[mode]["success"] += int(ok)
            mode_success[mode][expected] += 1
            mode_success[mode][f"{expected}__success"] += int(ok)

        candidate_ids = matched_candidate_specs(text)
        for candidate_id in candidate_ids:
            coverage = current_coverage_for_candidate(candidate_id, current_active)
            candidate_stats[candidate_id]["support"] += 1
            candidate_stats[candidate_id]["coverage_sum_milli"] += int(round(coverage * 1000))
            candidate_stats[candidate_id]["low_coverage"] += int(coverage < args.candidate_coverage_threshold)
            candidate_stats[candidate_id][expected] += 1
            if len(candidate_examples[candidate_id]) < 5:
                candidate_examples[candidate_id].append({
                    "row_index": index,
                    "row_id": row.get("row_id"),
                    "url": row.get("url"),
                    "expected_action": expected,
                    "text_head": text[:420].replace("\n", " "),
                })

        if len(row_samples) < args.sample_limit and (index % max(1, args.limit // max(1, args.sample_limit)) == 0 or not current_ok or (current_ok and not legacy_ok)):
            row_samples.append({
                "row_index": index,
                "row_id": row.get("row_id"),
                "url": row.get("url"),
                "text_head": text[:700].replace("\n", " "),
                "expected_action": expected,
                "none_baseline": {"success": none_ok, "action": none_action, "active": none_active[:8]},
                "legacy_subset": {"success": legacy_ok, "action": legacy_action, "active": legacy_active[:8]},
                "current_library": {"success": current_ok, "action": current_action, "active": current_active[:12]},
                "candidate_skill_hits": candidate_ids,
            })

        if rows_seen % args.chunk_rows == 0:
            rates = {
                mode: round(mode_success[mode]["success"] / max(1, mode_success[mode]["total"]), 6)
                for mode in ("none", "legacy", "current")
            }
            snapshot = {
                "event": "chunk",
                "timestamp_ms": now_ms(),
                "rows_seen": rows_seen,
                "elapsed_seconds": round(time.time() - start, 3),
                "rates": rates,
                "candidate_support_seen": {key: value["support"] for key, value in candidate_stats.items()},
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)

    mode_reports = {}
    for mode in ("none", "legacy", "current"):
        total = mode_success[mode]["total"]
        success = mode_success[mode]["success"]
        mode_reports[mode] = {
            "total": total,
            "success": success,
            "accuracy": round(success / max(1, total), 8),
            "per_action": {
                action: {
                    "total": mode_success[mode][action],
                    "success": mode_success[mode][f"{action}__success"],
                    "accuracy": round(mode_success[mode][f"{action}__success"] / max(1, mode_success[mode][action]), 8),
                }
                for action in sorted(expected_counter)
            },
        }

    skill_rows: list[dict[str, Any]] = []
    for candidate_id, counter in candidate_stats.items():
        support = counter["support"]
        avg_coverage = counter["coverage_sum_milli"] / max(1, support) / 1000.0
        low_coverage_rate = counter["low_coverage"] / max(1, support)
        spec = SKILL_CANDIDATE_SPECS[candidate_id]
        gap_score = support * (1.0 - avg_coverage) * (1.0 + low_coverage_rate)
        skill_rows.append({
            "candidate_id": candidate_id,
            "title": spec["title"],
            "need": spec["need"],
            "support_count": support,
            "avg_current_coverage": round(avg_coverage, 6),
            "low_coverage_rate": round(low_coverage_rate, 6),
            "gap_score": round(gap_score, 6),
            "suggested_status": "FarmCandidate" if support >= args.min_candidate_support and low_coverage_rate >= 0.25 else "Watch",
            "covered_by_existing": sorted(spec["covered_by"]),
            "action_mix": {action: counter[action] for action in sorted(expected_counter) if counter[action]},
        })
    skill_rows.sort(key=lambda row: (-row["gap_score"], row["candidate_id"]))

    text_delta = {
        "none_baseline_accuracy": mode_reports["none"]["accuracy"],
        "legacy_grounding_subset_accuracy": mode_reports["legacy"]["accuracy"],
        "current_e118_library_accuracy": mode_reports["current"]["accuracy"],
        "current_minus_legacy_delta": round(mode_reports["current"]["accuracy"] - mode_reports["legacy"]["accuracy"], 8),
        "current_minus_none_delta": round(mode_reports["current"]["accuracy"] - mode_reports["none"]["accuracy"], 8),
        "current_hard_negative_count": current_hard_negative,
        "expected_action_distribution": dict(expected_counter),
        "mode_reports": mode_reports,
    }
    farm_candidates = [row for row in skill_rows if row["suggested_status"] == "FarmCandidate"]
    generation = {
        "freeform_gemma_style_generation_ready": False,
        "grounded_canonical_text_io_ready": mode_reports["current"]["accuracy"] >= args.pass_accuracy,
        "calc_scribe_available": calc_scribe.get("available", False),
        "capability_boundary": "The library can select canonical actions and trace-aware render templates; it is not a free-form next-token chatbot.",
        "strongest_current_output_forms": [
            "ANSWER_WITH_TRACE",
            "ASK_FOR_EVIDENCE",
            "DEFER_CONFLICT",
            "OBSERVE_AND_GROUND",
            "VALIDATE_VISIBLE_CALC_TRACE",
            "UPDATE_PROGRESS_LEDGER",
        ],
        "missing_for_gemma_style": [
            "open-vocabulary semantic compression",
            "free-form fluent decoder",
            "broad world knowledge memory",
            "long-context discourse planner",
        ],
    }
    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "rows_seen": rows_seen,
        "operator_count": len(operators),
        "actual_300k_operator_count": sum(1 for row in operators if row.get("actual_300k_reached")),
        "skill_candidate_count": len(skill_rows),
        "farm_candidate_count": len(farm_candidates),
        "current_text_io_accuracy": mode_reports["current"]["accuracy"],
        "legacy_text_io_accuracy": mode_reports["legacy"]["accuracy"],
        "none_text_io_accuracy": mode_reports["none"]["accuracy"],
        "current_minus_legacy_delta": text_delta["current_minus_legacy_delta"],
        "current_hard_negative_count": current_hard_negative,
        "family_counter": dict(family_counter),
        "seconds": round(time.time() - start, 3),
    }
    decision_label = "e119_fineweb_skill_mining_positive"
    failures: list[str] = []
    if rows_seen < args.limit:
        failures.append("dataset ended before requested limit")
    if current_hard_negative:
        failures.append("current library hard negative detected")
        decision_label = "e119_current_library_text_io_hard_negative_detected"
    elif text_delta["current_minus_legacy_delta"] < args.min_delta:
        failures.append("current-vs-legacy text IO delta too small")
        decision_label = "e119_no_clear_text_io_delta"
    elif len(farm_candidates) < args.min_farm_candidates:
        failures.append("too few farm candidates detected")
        decision_label = "e119_fineweb_no_new_skill_candidates"

    manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "not Gemma-style language model, not final training, not PermaCore, not TrueGolden",
        "dataset": str(dataset),
        "e118_root": str(args.e118_root),
        "e83_root": str(args.e83_root),
        "row_limit": args.limit,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    }
    dataset_report = {
        "dataset": str(dataset),
        "rows_seen": rows_seen,
        "language_filter": "none",
        "source_kind": "FineWeb-Edu local JSONL sample",
    }
    operator_report = {
        "operator_count": len(operators),
        "actual_300k_operator_count": aggregate["actual_300k_operator_count"],
        "families": dict(Counter(row.get("family", "unknown") for row in operators)),
        "groups": dict(Counter(row.get("group_id", "unknown") for row in operators)),
        "calc_scribe": calc_scribe,
    }
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "failure_count": len(failures),
        "failures": failures,
        "checker_failure_count": None,
    }
    summary = {
        **aggregate,
        "decision": decision_label,
        "farm_candidates_top": [row["candidate_id"] for row in farm_candidates[:10]],
        "generation_boundary": generation["capability_boundary"],
    }
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "skill_rows": skill_rows,
        "text_delta": text_delta,
        "dataset": str(dataset),
        "contract": ARTIFACT_CONTRACT,
    }
    replay_hash = deterministic_hash(replay_payload)

    write_json(out / "run_manifest.json", manifest)
    write_json(out / "dataset_report.json", dataset_report)
    write_json(out / "operator_source_report.json", operator_report)
    write_json(out / "skill_candidate_report.json", {"rows": skill_rows})
    write_json(out / "text_io_delta_report.json", text_delta)
    write_json(out / "generation_readiness_report.json", generation)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": replay_hash, "hash_match": True, "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in row_samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "skill_candidate_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for candidate_id, examples in sorted(candidate_examples.items()):
            for example in examples:
                handle.write(json.dumps({"candidate_id": candidate_id, **example}, ensure_ascii=False, sort_keys=True) + "\n")
    report_lines = [
        "# E119 FineWeb Skill Mining And Text IO Delta Probe",
        "",
        "```text",
        f"decision = {decision_label}",
        f"failure_count = {len(failures)}",
        f"rows_seen = {rows_seen}",
        f"current_text_io_accuracy = {mode_reports['current']['accuracy']}",
        f"legacy_text_io_accuracy = {mode_reports['legacy']['accuracy']}",
        f"delta = {text_delta['current_minus_legacy_delta']}",
        f"farm_candidate_count = {len(farm_candidates)}",
        "```",
        "",
        "Boundary: this is canonical text-IO/action selection, not free-form Gemma-style generation.",
        "",
        "## Top Skill Candidates",
        "",
    ]
    for row in skill_rows[:12]:
        report_lines.append(f"- `{row['candidate_id']}`: support={row['support_count']}, coverage={row['avg_current_coverage']}, status={row['suggested_status']}")
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "The current library is measured against a legacy grounding-only subset. A positive delta means the E100-E106 evidence/answer/progress operators add real text-IO coverage over earlier grounding and stream-stability skills.",
    ])
    (out / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "rows_seen": rows_seen, "decision": decision_label})
    write_json(out / "partial_aggregate_snapshot.json", {"event": "complete", **aggregate})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--e118-root", default=str(DEFAULT_E118))
    parser.add_argument("--e83-root", default=str(DEFAULT_E83))
    parser.add_argument("--out", default="target/pilot_wave/e119_fineweb_skill_mining_and_text_io_delta_probe")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--chunk-rows", type=int, default=5_000)
    parser.add_argument("--sample-limit", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.20)
    parser.add_argument("--pass-accuracy", type=float, default=0.75)
    parser.add_argument("--candidate-coverage-threshold", type=float, default=0.75)
    parser.add_argument("--min-candidate-support", type=int, default=500)
    parser.add_argument("--min-farm-candidates", type=int, default=3)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
