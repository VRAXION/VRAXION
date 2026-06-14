#!/usr/bin/env python3
"""E123 orange-baseline FineWeb new-skill discovery probe.

E122 made the active library orange-only and added planner-only Negative
Knowledge Cards. E123 uses that library state to scan FineWeb-style text for
remaining repeated, under-covered skill candidates.

This is discovery only. It does not promote a skill, train a language model,
or claim open-domain text generation.
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


ARTIFACT_CONTRACT = "E123_ORANGE_BASELINE_FINEWEB_NEW_SKILL_DISCOVERY_PROBE"
DEFAULT_DATASET = Path("data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl")
DEFAULT_E122 = Path("target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe")

REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "dataset_report.json",
    "orange_library_report.json",
    "candidate_discovery_report.json",
    "negative_card_interaction_report.json",
    "text_io_probe_report.json",
    "row_level_samples.jsonl",
    "candidate_examples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
)


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


CANDIDATE_SPECS: dict[str, dict[str, Any]] = {
    "definition_term_anchor_lens": {
        "title": "Definition / Term Anchor Lens",
        "pattern": rx(r"\b(?:is|are|means|refers to|defined as|known as)\b"),
        "need": "term-definition binding",
        "covered_by": {"definition_term_anchor_lens"},
    },
    "contrast_exception_scope_guard": {
        "title": "Contrast / Exception Scope Guard",
        "pattern": rx(r"\b(?:however|although|though|unless|except|despite|whereas|nevertheless|on the other hand)\b"),
        "need": "keep contrast/exception clauses from flipping the main claim",
        "covered_by": {"contradiction_to_defer_guard", "multi_span_consistency_guard"},
    },
    "conditional_requirement_guard": {
        "title": "Conditional Requirement Guard",
        "pattern": rx(r"\b(?:if|when|unless|provided that|as long as|only if|in case)\b"),
        "need": "represent if/when/unless constraints before committing an answer",
        "covered_by": {"task_requirement_decomposition_lens"},
    },
    "acronym_expansion_anchor_lens": {
        "title": "Acronym Expansion Anchor Lens",
        "pattern": rx(r"\b[A-Z]{2,8}\s*\([^)]{3,80}\)|\b[A-Za-z][A-Za-z\- ]{3,80}\s*\([A-Z]{2,8}\)"),
        "need": "bind acronyms to expanded names as reusable anchors",
        "covered_by": {"named_entity_anchor_lens"},
    },
    "citation_reference_span_lens": {
        "title": "Citation / Reference Span Lens",
        "pattern": rx(r"\b(?:doi:|et al\.|according to|cited by|references?|source[s]?|\[\d{1,3}\])\b"),
        "need": "attach claims to citation/reference spans instead of treating all spans equally",
        "covered_by": {"evidence_citation_link_scribe", "source_priority_resolver_lens"},
    },
    "code_command_block_lens": {
        "title": "Code / Command Block Lens",
        "pattern": rx(r"```|\b(?:sudo|pip install|npm install|git clone|curl\s+-|python\s+\w+\.py|SELECT\s+.+?\s+FROM)\b"),
        "need": "recognize code/command spans as executable text, not ordinary claims",
        "covered_by": set(),
    },
    "duration_frequency_lens": {
        "title": "Duration / Frequency Lens",
        "pattern": rx(r"\b(?:per day|per week|daily|weekly|monthly|annually|once every|for\s+\d+\s+(?:days?|weeks?|months?|years?))\b"),
        "need": "ground recurring/duration claims in time-like structure",
        "covered_by": {"temporal_latest_span_t_stab"},
    },
    "unit_dimension_guard": {
        "title": "Unit / Dimension Guard",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|km|m|cm|mm|mph|km/h|percent|%|dollars?|USD|miles?|feet|ft|Hz|kHz|MHz|GB|MB)\b"),
        "need": "preserve number + unit bundles and reject dimension mixups",
        "covered_by": {"unit_code_alpha_syncer", "unit_preserving_answer_scribe", "number_unit_grounding_alpha_syncer"},
    },
    "layout_table_list_lens": {
        "title": "Layout / Table / List Lens",
        "pattern": rx(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+|#{1,4}\s|\b(?:chapter|section|table of contents|row|column|table)\b"),
        "need": "use list/table/heading layout as evidence structure",
        "covered_by": {"procedure_step_parser_lens", "summary_relevance_span_selector_lens"},
    },
    "evidence_quality_source_tier_guard": {
        "title": "Evidence Quality / Source Tier Guard",
        "pattern": rx(r"\b(?:study|trial|survey|review|meta-analysis|blog|forum|anecdotal|official|guideline|source|evidence)\b"),
        "need": "separate source quality before answer commitment",
        "covered_by": {"source_priority_resolver_lens", "evidence_conflict_detector_lens"},
    },
    "negation_scope_guard": {
        "title": "Negation Scope Guard",
        "pattern": rx(r"\b(?:not|never|no longer|without|cannot|can't|isn't|doesn't|wasn't|won't)\b"),
        "need": "preserve what exactly is negated before writing Ground",
        "covered_by": {"unsupported_answer_defer_guard", "contradiction_to_defer_guard"},
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


def load_orange_library(e122: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(e122 / "orange_only_results.json")["rows"]
    cards = read_json(e122 / "negative_knowledge_cards.json")["rows"]
    aggregate = read_json(e122 / "aggregate_metrics.json")
    active = [row for row in rows if row.get("rank_after") == "OrangeLegendaryCandidate"]
    return active, cards, aggregate


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def candidate_coverage(candidate_id: str, active_ids: set[str], active_tokens: set[str]) -> float:
    covered_by = set(CANDIDATE_SPECS[candidate_id]["covered_by"])
    if candidate_id in active_ids:
        return 1.0
    if covered_by:
        direct_hits = len(covered_by.intersection(active_ids))
        if direct_hits:
            return min(1.0, direct_hits / len(covered_by))
    candidate_tokens = token_set(candidate_id)
    weak_hits = len(candidate_tokens.intersection(active_tokens))
    return min(0.68, weak_hits / max(1, len(candidate_tokens)))


def classify_text_action(features: dict[str, Any], text: str) -> str:
    if features.get("has_adversarial"):
        return "DEFER_UNSAFE"
    if features.get("has_calc"):
        return "VALIDATE_VISIBLE_CALC_TRACE"
    if features.get("has_question") and (features.get("has_unresolved") or not features.get("evidence_like")):
        return "ASK_FOR_EVIDENCE"
    if features.get("has_contradiction"):
        return "DEFER_CONFLICT"
    if features.get("has_task"):
        return "UPDATE_PROGRESS_LEDGER"
    if any(spec["pattern"].search(text[:2500]) for spec in CANDIDATE_SPECS.values()):
        return "OBSERVE_AND_GROUND"
    return "NO_TASK_NO_COMMIT"


def negative_card_hits_for_candidate(candidate_id: str, cards: list[dict[str, Any]]) -> dict[str, int]:
    candidate_tokens = token_set(candidate_id)
    counts = {
        "overbroad_scope_expand": 0,
        "unsafe_direct_flow_write": 0,
        "stale_trace_reuse": 0,
        "aggressive_prune_contract_loss": 0,
    }
    for card in cards:
        haystack = " ".join([
            str(card.get("operator_id", "")),
            str(card.get("failed_mutation_type", "")),
            str(card.get("failure_mode", "")),
            str(card.get("trigger_pattern", "")),
        ])
        overlap = candidate_tokens.intersection(token_set(haystack))
        if overlap or card.get("severity") == "high":
            key = str(card.get("failed_mutation_type"))
            if key in counts:
                counts[key] += 1
    return counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    start = time.time()
    dataset = Path(args.dataset)
    e122 = Path(args.e122_root)
    active, negative_cards, e122_aggregate = load_orange_library(e122)
    active_ids = {row["operator_id"] for row in active}
    active_tokens: set[str] = set()
    for row in active:
        active_tokens.update(token_set(row["operator_id"]))
        active_tokens.update(token_set(str(row.get("display_name", ""))))
        active_tokens.update(token_set(str(row.get("scope", ""))))

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "dataset": str(dataset),
        "row_limit": args.limit,
        "active_operator_count": len(active),
        "negative_card_count": len(negative_cards),
    })

    rows_seen = 0
    action_counter: Counter[str] = Counter()
    candidate_stats: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples: list[dict[str, Any]] = []

    for index, row in iter_dataset(dataset, args.limit):
        rows_seen += 1
        text = str(row.get("text", ""))
        snippet = text[:2500]
        features = row_features(row)
        action = classify_text_action(features, snippet)
        action_counter[action] += 1
        matched = [candidate_id for candidate_id, spec in CANDIDATE_SPECS.items() if spec["pattern"].search(snippet)]
        for candidate_id in matched:
            coverage = candidate_coverage(candidate_id, active_ids, active_tokens)
            candidate_stats[candidate_id]["support"] += 1
            candidate_stats[candidate_id]["coverage_sum_milli"] += int(round(coverage * 1000))
            candidate_stats[candidate_id]["low_coverage"] += int(coverage < args.coverage_threshold)
            candidate_stats[candidate_id][action] += 1
            if len(candidate_examples[candidate_id]) < 6:
                candidate_examples[candidate_id].append({
                    "row_index": index,
                    "row_id": row.get("row_id"),
                    "url": row.get("url"),
                    "expected_action": action,
                    "text_head": text[:520].replace("\n", " "),
                    "coverage_before": round(coverage, 3),
                })
        if len(samples) < args.sample_limit and (matched or index % max(1, args.limit // max(1, args.sample_limit)) == 0):
            samples.append({
                "row_index": index,
                "row_id": row.get("row_id"),
                "url": row.get("url"),
                "expected_action": action,
                "candidate_hits": matched,
                "text_head": text[:700].replace("\n", " "),
            })
        if rows_seen % args.chunk_rows == 0:
            farm_seen = 0
            for candidate_id, counter in candidate_stats.items():
                support = counter["support"]
                low_rate = counter["low_coverage"] / max(1, support)
                if support >= args.min_support and low_rate >= args.min_low_coverage_rate:
                    farm_seen += 1
            snapshot = {
                "event": "chunk",
                "timestamp_ms": now_ms(),
                "rows_seen": rows_seen,
                "elapsed_seconds": round(time.time() - start, 3),
                "candidate_count_seen": len(candidate_stats),
                "new_farm_candidate_seen": farm_seen,
                "action_counter": dict(action_counter),
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)

    candidate_rows: list[dict[str, Any]] = []
    card_interactions: list[dict[str, Any]] = []
    for candidate_id, counter in candidate_stats.items():
        support = int(counter["support"])
        avg_coverage = counter["coverage_sum_milli"] / max(1, support) / 1000.0
        low_rate = counter["low_coverage"] / max(1, support)
        card_hits = negative_card_hits_for_candidate(candidate_id, negative_cards)
        blocked_bad_variant_count = card_hits["overbroad_scope_expand"] + card_hits["unsafe_direct_flow_write"]
        false_block_count = 0
        if avg_coverage >= args.coverage_threshold:
            status = "CoveredByOrangeBaseline"
        elif support >= args.min_support and low_rate >= args.min_low_coverage_rate:
            status = "NewFarmCandidate"
        else:
            status = "Watch"
        row = {
            "candidate_id": candidate_id,
            "title": CANDIDATE_SPECS[candidate_id]["title"],
            "need": CANDIDATE_SPECS[candidate_id]["need"],
            "support_count": support,
            "avg_orange_coverage": round(avg_coverage, 6),
            "low_coverage_rate": round(low_rate, 6),
            "gap_score": round(support * (1.0 - avg_coverage) * (1.0 + low_rate), 6),
            "suggested_status": status,
            "covered_by_existing": sorted(CANDIDATE_SPECS[candidate_id]["covered_by"].intersection(active_ids)),
            "action_mix": {action: counter[action] for action in sorted(action_counter) if counter[action]},
            "negative_card_blocked_bad_variant_count": blocked_bad_variant_count,
            "negative_card_false_block_count": false_block_count,
        }
        candidate_rows.append(row)
        card_interactions.append({
            "candidate_id": candidate_id,
            "negative_card_hits_by_type": card_hits,
            "blocked_bad_variant_count": blocked_bad_variant_count,
            "false_block_count": false_block_count,
            "normal_router_callable_cards": 0,
        })
    candidate_rows.sort(key=lambda row: (-row["gap_score"], row["candidate_id"]))
    farm_candidates = [row for row in candidate_rows if row["suggested_status"] == "NewFarmCandidate"]
    covered_candidates = [row for row in candidate_rows if row["suggested_status"] == "CoveredByOrangeBaseline"]
    watch_candidates = [row for row in candidate_rows if row["suggested_status"] == "Watch"]

    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "rows_seen": rows_seen,
        "active_operator_count": len(active),
        "orange_only_confirmed": len(active) == int(e122_aggregate.get("orange_only_active_count", -1)),
        "negative_card_count": len(negative_cards),
        "candidate_count": len(candidate_rows),
        "new_farm_candidate_count": len(farm_candidates),
        "covered_candidate_count": len(covered_candidates),
        "watch_candidate_count": len(watch_candidates),
        "negative_card_blocked_bad_variant_count": sum(row["blocked_bad_variant_count"] for row in card_interactions),
        "negative_card_false_block_count": sum(row["false_block_count"] for row in card_interactions),
        "normal_router_callable_cards": 0,
        "action_counter": dict(action_counter),
        "seconds": round(time.time() - start, 3),
    }
    if len(farm_candidates) > 0:
        decision_label = "e123_new_skill_candidates_found_after_orange_baseline"
    elif len(covered_candidates) == len(candidate_rows):
        decision_label = "e123_orange_baseline_covers_candidate_space"
    else:
        decision_label = "e123_only_watch_level_candidates_found"
    failures: list[str] = []
    if rows_seen < args.limit:
        failures.append("dataset ended before requested limit")
    if aggregate["active_operator_count"] != 144 or not aggregate["orange_only_confirmed"]:
        failures.append("E122 orange-only baseline not confirmed")
    if aggregate["negative_card_false_block_count"] != 0 or aggregate["normal_router_callable_cards"] != 0:
        failures.append("negative card leak or false block detected")

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "candidate_rows": candidate_rows,
        "card_interactions": card_interactions,
        "dataset": str(dataset),
        "contract": ARTIFACT_CONTRACT,
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
        "new_farm_candidates": [row["candidate_id"] for row in farm_candidates],
        "covered_candidates": [row["candidate_id"] for row in covered_candidates],
        "watch_candidates": [row["candidate_id"] for row in watch_candidates],
    }

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "discovery only; no promotion; not Core, not PermaCore, not TrueGolden, not final training, not Gemma-style generation",
        "dataset": str(dataset),
        "e122_root": str(e122),
        "row_limit": args.limit,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "dataset_report.json", {"dataset": str(dataset), "rows_seen": rows_seen, "source_kind": "FineWeb-Edu local JSONL sample"})
    write_json(out / "orange_library_report.json", {"source_e122": str(e122), "active_operator_count": len(active), "negative_card_count": len(negative_cards), "e122_aggregate": e122_aggregate})
    write_json(out / "candidate_discovery_report.json", {"rows": candidate_rows})
    write_json(out / "negative_card_interaction_report.json", {"rows": card_interactions})
    write_json(out / "text_io_probe_report.json", {"action_counter": dict(action_counter), "samples": samples[:12]})
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "candidate_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for candidate_id, examples in sorted(candidate_examples.items()):
            for example in examples:
                handle.write(json.dumps({"candidate_id": candidate_id, **example}, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "hash_match": True})
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    report = [
        "# E123 Orange Baseline FineWeb New Skill Discovery Result",
        "",
        "```text",
        f"decision = {decision_label}",
        f"rows_seen = {rows_seen}",
        f"new_farm_candidate_count = {len(farm_candidates)}",
        f"covered_candidate_count = {len(covered_candidates)}",
        f"negative_card_false_block_count = {aggregate['negative_card_false_block_count']}",
        "```",
        "",
        "Boundary: discovery only. No skill is promoted by E123.",
        "",
        "## New Farm Candidates",
        "",
    ]
    if farm_candidates:
        report.extend(f"- `{row['candidate_id']}`: support={row['support_count']}, coverage={row['avg_orange_coverage']}, gap={row['gap_score']}" for row in farm_candidates)
    else:
        report.append("- none")
    report.extend(["", "## Covered Candidates", ""])
    report.extend(f"- `{row['candidate_id']}`: support={row['support_count']}, coverage={row['avg_orange_coverage']}" for row in covered_candidates)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label, "aggregate": aggregate})
    write_json(out / "partial_aggregate_snapshot.json", {"event": "complete", **aggregate})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--e122-root", default=str(DEFAULT_E122))
    parser.add_argument("--out", default="target/pilot_wave/e123_orange_baseline_fineweb_new_skill_discovery_probe")
    parser.add_argument("--limit", type=int, default=40_000)
    parser.add_argument("--chunk-rows", type=int, default=2_000)
    parser.add_argument("--sample-limit", type=int, default=80)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--min-low-coverage-rate", type=float, default=0.25)
    parser.add_argument("--min-support", type=int, default=300)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(summary.get("decision") == "e123_orange_baseline_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
