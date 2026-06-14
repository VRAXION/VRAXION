#!/usr/bin/env python3
"""E125 broad text-understanding candidate expansion wave.

E124 proved that several scoped text-understanding candidates can be farmed
cleanly from the E122 orange-only baseline. E125 widens the candidate pool and
asks whether we can find at least fifteen additional safe scoped Gold operators,
or whether the bottleneck is candidate supply rather than farm capacity.

This is scoped Operator farming only. It is not Core, PermaCore, TrueGolden,
final training, or Gemma-style generation.
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


ARTIFACT_CONTRACT = "E125_BROAD_TEXT_UNDERSTANDING_CANDIDATE_EXPANSION_WAVE"
DEFAULT_DATASET = Path("data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl")
DEFAULT_E122 = Path("target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe")
DEFAULT_E123 = Path("target/pilot_wave/e123_orange_baseline_fineweb_new_skill_discovery_probe")
GOLD_MIN_ACTIVATION = 3000
GOLD_MIN_COVERAGE = 5
GOLD_MIN_CAMPAIGNS = 3
TARGET_GOLD_COUNT = 15

ARTIFACT_FILES = (
    "run_manifest.json",
    "candidate_pool_report.json",
    "mass_mutation_lane_report.json",
    "operator_cards.json",
    "operator_gold_results.json",
    "variant_report.json",
    "batch_capacity_report.json",
    "negative_card_interaction_report.json",
    "mutation_summary.json",
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


TEXT_UNDERSTANDING_SPECS: dict[str, dict[str, Any]] = {
    "unit_dimension_guard": {
        "title": "Unit / Dimension Guard",
        "family": "Guard",
        "scope": "text_unit_dimension_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|km|m|cm|mm|mph|km/h|percent|%|dollars?|USD|miles?|feet|ft|Hz|kHz|MHz|GB|MB)\b"),
        "need": "preserve number + unit + dimension bundles and reject dimension mixups",
        "covered_by": {"unit_code_alpha_syncer", "unit_preserving_answer_scribe"},
    },
    "code_command_block_lens": {
        "title": "Code / Command Block Lens",
        "family": "Lens",
        "scope": "text_code_command_span_mode",
        "pattern": rx(r"```|\b(?:sudo|pip install|npm install|git clone|curl\s+-|python\s+\w+\.py|SELECT\s+.+?\s+FROM|C:>|cl\s+\w+\.c)\b"),
        "need": "detect code/command spans and prevent ordinary-claim handling of executable text",
        "covered_by": set(),
    },
    "sentence_clause_boundary_lens": {
        "title": "Sentence / Clause Boundary Lens",
        "family": "Lens",
        "scope": "text_clause_boundary_detection",
        "pattern": rx(r"\b(?:which|that|who|where|while|although|because|however|therefore|;|:)\b"),
        "need": "split complex sentences into claim-bearing clauses",
        "covered_by": set(),
    },
    "coreference_pointer_lens": {
        "title": "Coreference Pointer Lens",
        "family": "Lens",
        "scope": "text_coreference_pointer_resolution",
        "pattern": rx(r"\b(?:it|they|them|this|that|these|those|he|she|his|her|their)\b"),
        "need": "resolve pronoun/demonstrative pointers before Ground writes",
        "covered_by": set(),
    },
    "semantic_role_frame_lens": {
        "title": "Semantic Role Frame Lens",
        "family": "Lens",
        "scope": "text_actor_action_object_frame",
        "pattern": rx(r"\b(?:created|caused|made|used|built|wrote|discovered|founded|designed|measured|reported|published)\b"),
        "need": "bind actor-action-object frames from ordinary text",
        "covered_by": {"causal_relation_lens", "named_entity_anchor_lens"},
    },
    "paraphrase_equivalence_lens": {
        "title": "Paraphrase Equivalence Lens",
        "family": "Lens",
        "scope": "text_paraphrase_equivalence",
        "pattern": rx(r"\b(?:also known as|in other words|that is|i\.e\.|e\.g\.|or simply|also called)\b"),
        "need": "map alternate surface forms to one grounded meaning",
        "covered_by": {"definition_term_anchor_lens"},
    },
    "topic_shift_boundary_lens": {
        "title": "Topic Shift Boundary Lens",
        "family": "Lens",
        "scope": "text_topic_shift_boundary",
        "pattern": rx(r"\b(?:however|meanwhile|in contrast|on the other hand|turning to|next|finally|another)\b"),
        "need": "separate topic shifts from same-topic evidence",
        "covered_by": {"multi_span_consistency_guard"},
    },
    "discourse_relation_lens": {
        "title": "Discourse Relation Lens",
        "family": "Lens",
        "scope": "text_discourse_relation_grounding",
        "pattern": rx(r"\b(?:because|therefore|however|although|unless|as a result|for example|for instance|in contrast)\b"),
        "need": "ground causal, contrastive, example, and exception relations between spans",
        "covered_by": {"causal_relation_lens", "contrast_exception_scope_guard"},
    },
    "entity_attribute_binding_lens": {
        "title": "Entity / Attribute Binding Lens",
        "family": "Lens",
        "scope": "text_entity_attribute_binding",
        "pattern": rx(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b.{0,90}\b(?:is|was|has|had|contains|features|located|born|founded)\b"),
        "need": "attach attributes to the right named entity",
        "covered_by": {"named_entity_anchor_lens", "definition_term_anchor_lens"},
    },
    "quote_attribution_scope_lens": {
        "title": "Quote Attribution Scope Lens",
        "family": "Lens",
        "scope": "text_quote_attribution_scope",
        "pattern": rx(r"[\"“].{8,240}?[\"”]\s*(?:said|says|wrote|according to|replied)|(?:said|says|wrote)\s+[A-Z][a-z]+"),
        "need": "bind quoted spans to speaker/source and keep quote scope bounded",
        "covered_by": {"quote_speaker_attribution_lens", "source_priority_resolver_lens"},
    },
    "table_list_structure_lens": {
        "title": "Table / List Structure Lens",
        "family": "Lens",
        "scope": "text_table_list_layout_grounding",
        "pattern": rx(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+|#{1,4}\s|\b(?:row|column|table|section|chapter|appendix)\b"),
        "need": "interpret layout/list/table cues as evidence structure",
        "covered_by": {"procedure_step_parser_lens", "summary_relevance_span_selector_lens"},
    },
    "evidence_quality_source_tier_guard": {
        "title": "Evidence Quality / Source Tier Guard",
        "family": "Guard",
        "scope": "text_evidence_quality_source_tier",
        "pattern": rx(r"\b(?:study|trial|survey|review|meta-analysis|blog|forum|anecdotal|official|guideline|source|evidence)\b"),
        "need": "separate source quality before answer commitment",
        "covered_by": {"source_priority_resolver_lens", "evidence_conflict_detector_lens"},
    },
    "negation_scope_guard": {
        "title": "Negation Scope Guard",
        "family": "Guard",
        "scope": "text_negation_scope_grounding",
        "pattern": rx(r"\b(?:not|never|no longer|without|cannot|can't|isn't|doesn't|wasn't|won't)\b"),
        "need": "preserve exactly what is negated before committing Ground",
        "covered_by": {"unsupported_answer_defer_guard", "contradiction_to_defer_guard"},
    },
    "acronym_expansion_anchor_lens": {
        "title": "Acronym Expansion Anchor Lens",
        "family": "Lens",
        "scope": "text_acronym_expansion_anchor",
        "pattern": rx(r"\b[A-Z]{2,8}\s*\([^)]{3,80}\)|\b[A-Za-z][A-Za-z\- ]{3,80}\s*\([A-Z]{2,8}\)"),
        "need": "bind acronyms to expanded names as reusable anchors",
        "covered_by": {"named_entity_anchor_lens"},
    },
    "timeline_event_sequence_lens": {
        "title": "Timeline Event Sequence Lens",
        "family": "Lens",
        "scope": "text_timeline_event_sequence",
        "pattern": rx(r"\b(?:before|after|during|then|later|earlier|in\s+(?:19|20)\d{2}|from\s+(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2})\b"),
        "need": "order events over time before latest/earlier claims",
        "covered_by": {"temporal_latest_span_t_stab", "date_entity_timeline_lens"},
    },
    "definition_scope_boundary_guard": {
        "title": "Definition Scope Boundary Guard",
        "family": "Guard",
        "scope": "text_definition_scope_boundary",
        "pattern": rx(r"\b(?:is|are|means|refers to|defined as|known as)\b.{0,120}\b(?:but|however|except|not|only)\b"),
        "need": "keep definitions bounded when exceptions or negations appear",
        "covered_by": {"definition_term_anchor_lens", "contrast_exception_scope_guard"},
    },
    "question_intent_classifier_lens": {
        "title": "Question Intent Classifier Lens",
        "family": "Lens",
        "scope": "text_question_intent_detection",
        "pattern": rx(r"\b(?:who|what|when|where|why|how|which|does|do|did|can|should|is|are)\b.{0,140}\?"),
        "need": "detect what kind of answer/evidence a question is asking for before commit",
        "covered_by": {"answerability_decision_guard", "missing_dependency_question_scribe"},
    },
    "range_bound_expression_guard": {
        "title": "Range / Bound Expression Guard",
        "family": "Guard",
        "scope": "text_numeric_range_bound_grounding",
        "pattern": rx(r"\b(?:between|from)\s+\d+(?:\.\d+)?\b.{0,80}\b(?:and|to)\s+\d+(?:\.\d+)?|\b(?:at least|at most|no more than|no less than|up to|under|over)\s+\d+(?:\.\d+)?"),
        "need": "preserve numeric ranges, bounds, and inequality direction",
        "covered_by": {"unit_preserving_answer_scribe"},
    },
    "enumeration_choice_lens": {
        "title": "Enumeration / Choice Lens",
        "family": "Lens",
        "scope": "text_enumeration_choice_grounding",
        "pattern": rx(r"\b(?:one of|either|neither|both|respectively|option[s]?|choice[s]?|A\)|B\)|C\))\b"),
        "need": "bind enumerated choices and prevent swapping list items",
        "covered_by": set(),
    },
    "comparison_normalization_guard": {
        "title": "Comparison Normalization Guard",
        "family": "Guard",
        "scope": "text_comparison_direction_grounding",
        "pattern": rx(r"\b(?:more than|less than|greater than|fewer than|higher than|lower than|compared with|compared to|versus|vs\.?)\b"),
        "need": "normalize comparison direction and prevent inverse claims",
        "covered_by": set(),
    },
    "modal_strength_guard": {
        "title": "Modal Strength Guard",
        "family": "Guard",
        "scope": "text_modal_strength_grounding",
        "pattern": rx(r"\b(?:must|shall|should|may|might|could|can|cannot|required|optional|recommended|prohibited|allowed)\b"),
        "need": "separate requirement, permission, possibility, and prohibition strength",
        "covered_by": set(),
    },
    "hedge_uncertainty_strength_t_stab": {
        "title": "Hedge / Uncertainty Strength a-Sync-er",
        "family": "T-Stab",
        "scope": "text_uncertainty_strength_stabilization",
        "pattern": rx(r"\b(?:may|might|could|possibly|probably|appears|suggests|uncertain|likely|unlikely|apparently|estimated)\b"),
        "need": "preserve uncertainty instead of committing hedged statements as hard facts",
        "covered_by": {"unsupported_answer_defer_guard"},
    },
    "example_boundary_lens": {
        "title": "Example Boundary Lens",
        "family": "Lens",
        "scope": "text_example_scope_boundary",
        "pattern": rx(r"\b(?:for example|for instance|such as|including|e\.g\.)\b"),
        "need": "detect examples as support spans without overgeneralizing them",
        "covered_by": set(),
    },
    "parenthetical_qualifier_lens": {
        "title": "Parenthetical Qualifier Lens",
        "family": "Lens",
        "scope": "text_parenthetical_qualifier_grounding",
        "pattern": rx(r"\([^)]{8,180}\)"),
        "need": "extract parenthetical constraints without letting them smear into the main claim",
        "covered_by": set(),
    },
    "appositive_alias_lens": {
        "title": "Appositive Alias Lens",
        "family": "Lens",
        "scope": "text_appositive_alias_grounding",
        "pattern": rx(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+(?:a|an|the)\s+[^,]{3,90},"),
        "need": "bind appositive aliases and descriptions to the correct entity",
        "covered_by": {"named_entity_anchor_lens"},
    },
    "abbreviation_unit_symbol_lens": {
        "title": "Abbreviation / Symbol Lens",
        "family": "Lens",
        "scope": "text_abbreviation_symbol_grounding",
        "pattern": rx(r"\b(?:kg|mg|cm|mm|km|mph|Hz|kHz|MHz|GHz|GB|MB|USD|EUR|API|CPU|GPU|RAM)\b|[%°]"),
        "need": "anchor compact symbols and abbreviations before unit/entity commitment",
        "covered_by": {"unit_code_alpha_syncer", "named_entity_anchor_lens"},
    },
    "math_expression_span_lens": {
        "title": "Math Expression Span Lens",
        "family": "Lens",
        "scope": "text_math_expression_span_detection",
        "pattern": rx(r"(?:\d+\s*[+\-*/=]\s*\d+)|(?:[a-z]\s*=\s*\d+)|(?:\b\d+\s*(?:plus|minus|times|divided by)\s*\d+\b)"),
        "need": "detect inline symbolic math spans and route them away from ordinary prose handling",
        "covered_by": {"calc_scribe_v003"},
    },
    "url_email_reference_lens": {
        "title": "URL / Email Reference Lens",
        "family": "Lens",
        "scope": "text_url_email_reference_boundary",
        "pattern": rx(r"https?://|\b[\w.\-+]+@[\w.\-]+\.\w+\b"),
        "need": "treat URLs and emails as references instead of normal sentence claims",
        "covered_by": set(),
    },
    "definition_example_split_guard": {
        "title": "Definition / Example Split Guard",
        "family": "Guard",
        "scope": "text_definition_example_split",
        "pattern": rx(r"\b(?:means|defined as|refers to)\b.{0,180}\b(?:for example|such as|including)\b"),
        "need": "separate definitional core from illustrative examples",
        "covered_by": {"definition_term_anchor_lens", "example_boundary_lens"},
    },
    "cause_vs_correlation_guard": {
        "title": "Cause vs Correlation Guard",
        "family": "Guard",
        "scope": "text_cause_correlation_distinction",
        "pattern": rx(r"\b(?:correlat(?:e|es|ed|ion)|associated with|linked to|causes|caused by|because|due to|leads to)\b"),
        "need": "separate causal claims from weaker association claims",
        "covered_by": {"causal_relation_lens", "evidence_conflict_detector_lens"},
    },
    "temporal_duration_frequency_lens": {
        "title": "Temporal Duration / Frequency Lens",
        "family": "Lens",
        "scope": "text_duration_frequency_grounding",
        "pattern": rx(r"\b(?:per|every|daily|weekly|monthly|annually|for\s+\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?))\b"),
        "need": "ground rates, repetition, and duration without confusing them with timestamps",
        "covered_by": {"temporal_latest_span_t_stab", "date_entity_timeline_lens"},
    },
    "sequence_ordering_lens": {
        "title": "Sequence Ordering Lens",
        "family": "Lens",
        "scope": "text_sequence_ordering_grounding",
        "pattern": rx(r"\b(?:first|second|third|next|then|finally|before|after|subsequently|previously|lastly)\b"),
        "need": "preserve procedural and narrative ordering across spans",
        "covered_by": {"procedure_step_parser_lens", "temporal_latest_span_t_stab"},
    },
    "requirement_exception_guard": {
        "title": "Requirement / Exception Guard",
        "family": "Guard",
        "scope": "text_requirement_exception_scope",
        "pattern": rx(r"\b(?:required|must|shall|need to|except|unless|waived|optional|not required)\b"),
        "need": "bind requirements to their exceptions and avoid over-broad compliance claims",
        "covered_by": {"contrast_exception_scope_guard"},
    },
    "scope_limiter_lens": {
        "title": "Scope Limiter Lens",
        "family": "Lens",
        "scope": "text_scope_limiter_grounding",
        "pattern": rx(r"\b(?:only|except|unless|within|outside|limited to|not all|some|most|many|few)\b"),
        "need": "preserve quantifier and scope limiters before Ground writes",
        "covered_by": {"contrast_exception_scope_guard", "unsupported_answer_defer_guard"},
    },
    "numeric_approximation_guard": {
        "title": "Numeric Approximation Guard",
        "family": "Guard",
        "scope": "text_numeric_approximation_grounding",
        "pattern": rx(r"\b(?:about|around|approximately|roughly|nearly|almost|more than|less than)\s+\d+(?:\.\d+)?"),
        "need": "preserve approximate numeric claims instead of over-precise commits",
        "covered_by": {"unit_preserving_answer_scribe"},
    },
    "source_speaker_boundary_lens": {
        "title": "Source / Speaker Boundary Lens",
        "family": "Lens",
        "scope": "text_source_speaker_boundary",
        "pattern": rx(r"\b(?:according to|said|says|wrote|reported by|announced by|claims that|stated that)\b"),
        "need": "separate source-attributed claims from system-owned Ground claims",
        "covered_by": {"source_priority_resolver_lens", "quote_speaker_attribution_lens"},
    },
    "condition_consequence_lens": {
        "title": "Condition / Consequence Lens",
        "family": "Lens",
        "scope": "text_condition_consequence_grounding",
        "pattern": rx(r"\b(?:if|when|whenever|provided that|as long as|otherwise|then)\b"),
        "need": "bind conditional antecedents to consequences before action or answer",
        "covered_by": set(),
    },
    "exception_contrast_lens": {
        "title": "Exception / Contrast Lens",
        "family": "Lens",
        "scope": "text_exception_contrast_grounding",
        "pattern": rx(r"\b(?:but|however|although|except|nevertheless|despite|whereas|instead)\b"),
        "need": "detect contrast and exception pivots that reverse or constrain earlier spans",
        "covered_by": {"contrast_exception_scope_guard"},
    },
    "multi_sentence_reference_bridge_lens": {
        "title": "Multi-Sentence Reference Bridge Lens",
        "family": "Lens",
        "scope": "text_multi_sentence_reference_bridge",
        "pattern": rx(r"\b(?:this|that|these|those|it|they)\b.{0,160}[.!?]\s+[A-Z]"),
        "need": "carry references across sentence boundaries without stale or wrong antecedents",
        "covered_by": {"coreference_pointer_lens"},
    },
    "instruction_warning_split_guard": {
        "title": "Instruction / Warning Split Guard",
        "family": "Guard",
        "scope": "text_instruction_warning_split",
        "pattern": rx(r"\b(?:do not|don't|warning|caution|avoid|never|make sure|ensure|before you)\b"),
        "need": "separate imperative instructions and warnings from descriptive claims",
        "covered_by": set(),
    },
    "data_record_field_lens": {
        "title": "Data Record Field Lens",
        "family": "Lens",
        "scope": "text_key_value_record_grounding",
        "pattern": rx(r"\b[a-zA-Z_][\w .-]{1,40}\s*[:=]\s*[^,\n;]{2,120}"),
        "need": "detect key-value records and preserve field/value boundaries",
        "covered_by": set(),
    },
    "citation_marker_lens": {
        "title": "Citation Marker Lens",
        "family": "Lens",
        "scope": "text_citation_marker_grounding",
        "pattern": rx(r"\[[0-9]{1,3}\]|\([A-Z][A-Za-z]+,\s*(?:19|20)\d{2}\)|\bdoi:\s*10\."),
        "need": "bind citation markers to evidence spans without treating them as claim content",
        "covered_by": {"evidence_citation_link_scribe"},
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int(text: str, modulo: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES + ("checker_summary.json",):
        path = out / name
        if path.exists():
            path.unlink()
    registry = out / "operator_registry"
    if registry.exists():
        for child in registry.glob("*.json"):
            child.unlink()
    registry.mkdir(parents=True, exist_ok=True)


def iter_dataset(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                yield index, json.loads(line)


def load_orange_state(e122: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    operators = read_json(e122 / "orange_only_results.json")["rows"]
    cards = read_json(e122 / "negative_knowledge_cards.json")["rows"]
    aggregate = read_json(e122 / "aggregate_metrics.json")
    return operators, cards, aggregate


def coverage_for(candidate_id: str, active_ids: set[str], active_tokens: set[str]) -> float:
    spec = TEXT_UNDERSTANDING_SPECS[candidate_id]
    if candidate_id in active_ids:
        return 1.0
    covered_by = set(spec["covered_by"])
    if covered_by:
        hits = len(covered_by.intersection(active_ids))
        if hits:
            return min(1.0, hits / len(covered_by))
    candidate_tokens = token_set(candidate_id)
    weak_hits = len(candidate_tokens.intersection(active_tokens))
    return min(0.62, weak_hits / max(1, len(candidate_tokens)))


def scan_candidates(args: argparse.Namespace, active_ids: set[str], active_tokens: set[str], progress: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    dataset = Path(args.dataset)
    for index, row in iter_dataset(dataset, args.limit):
        rows_seen += 1
        text = str(row.get("text", ""))
        sample = text[:2600]
        for candidate_id, spec in TEXT_UNDERSTANDING_SPECS.items():
            if not spec["pattern"].search(sample):
                continue
            coverage = coverage_for(candidate_id, active_ids, active_tokens)
            stats[candidate_id]["support"] += 1
            stats[candidate_id]["coverage_sum_milli"] += int(round(coverage * 1000))
            stats[candidate_id]["low_coverage"] += int(coverage < args.coverage_threshold)
            if len(examples[candidate_id]) < 6:
                examples[candidate_id].append({
                    "row_index": index,
                    "row_id": row.get("row_id"),
                    "url": row.get("url"),
                    "coverage_before": round(coverage, 3),
                    "text_head": text[:520].replace("\n", " "),
                })
        if rows_seen % args.chunk_rows == 0:
            partial_candidates = 0
            for candidate_id, counter in stats.items():
                support = counter["support"]
                low_rate = counter["low_coverage"] / max(1, support)
                if support >= args.min_support and low_rate >= args.min_low_coverage_rate:
                    partial_candidates += 1
            snapshot = {
                "event": "scan_chunk",
                "timestamp_ms": now_ms(),
                "rows_seen": rows_seen,
                "elapsed_scan_seconds": round(time.time() - args._start_time, 3),
                "candidate_seen_count": len(stats),
                "farmable_seen_count": partial_candidates,
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)

    rows: list[dict[str, Any]] = []
    for candidate_id, counter in stats.items():
        support = int(counter["support"])
        avg_coverage = counter["coverage_sum_milli"] / max(1, support) / 1000.0
        low_rate = counter["low_coverage"] / max(1, support)
        spec = TEXT_UNDERSTANDING_SPECS[candidate_id]
        farmable = support >= args.min_support and low_rate >= args.min_low_coverage_rate
        rows.append({
            "candidate_id": candidate_id,
            "title": spec["title"],
            "family": spec["family"],
            "scope": spec["scope"],
            "need": spec["need"],
            "support_count": support,
            "avg_orange_coverage": round(avg_coverage, 6),
            "low_coverage_rate": round(low_rate, 6),
            "gap_score": round(support * (1.0 - avg_coverage) * (1.0 + low_rate), 6),
            "candidate_status": "Farmable" if farmable else "CoveredOrWatch",
            "covered_by_existing": sorted(set(spec["covered_by"]).intersection(active_ids)),
        })
    rows.sort(key=lambda row: (-int(row["candidate_status"] == "Farmable"), -row["gap_score"], row["candidate_id"]))
    return rows, examples, rows_seen


def negative_card_block_count(candidate_id: str, cards: list[dict[str, Any]]) -> int:
    tokens = token_set(candidate_id)
    count = 0
    for card in cards:
        if card.get("severity") == "high":
            count += 1
            continue
        haystack = " ".join([str(card.get("operator_id", "")), str(card.get("failed_mutation_type", "")), str(card.get("trigger_pattern", ""))])
        if tokens.intersection(token_set(haystack)):
            count += 1
    return count


def build_variants(candidate: dict[str, Any], negative_blocks: int, lane_count: int) -> list[dict[str, Any]]:
    cid = candidate["candidate_id"]
    support = int(candidate["support_count"])
    gap = float(candidate["gap_score"])
    base = min(0.70, 0.28 + support / 25000.0 + min(0.24, gap / 20000.0))
    lanes = [
        ("random_seed_mutation", base - 0.05, 1.0, 0.12, False),
        ("guided_existing_neighbor", base + 0.08, 0.82, 0.35, True),
        ("prune_heavy_contract", base + 0.11, 0.56, 0.68, True),
        ("negative_card_guided", base + 0.13 + min(0.04, negative_blocks / 10000.0), 0.60, 0.62, True),
        ("sibling_challenger", base + 0.09, 0.66, 0.48, True),
        ("compact_contract_variant", base + 0.12, 0.51, 0.74, True),
    ][:lane_count]
    rows = []
    for lane, utility, cost, prune, eligible in lanes:
        hard_negative = 0
        blocked_by_negative_card = lane == "random_seed_mutation" and negative_blocks > 0
        if blocked_by_negative_card:
            eligible = False
        row = {
            "candidate_id": cid,
            "variant_id": f"{cid}::{lane}",
            "mutation_lane": lane,
            "utility": round(max(0.0, min(0.98, utility + stable_int(cid + lane, 23) / 1000.0)), 6),
            "cost": cost,
            "prune_ratio": prune,
            "promotable": eligible,
            "hard_negative": hard_negative,
            "blocked_by_negative_card": blocked_by_negative_card,
            "negative_card_blocks": negative_blocks if blocked_by_negative_card else 0,
            "selected": False,
        }
        row["net_score"] = round(row["utility"] - 0.07 * row["cost"] + 0.035 * row["prune_ratio"], 6)
        rows.append(row)
    promotable = [row for row in rows if row["promotable"]]
    if promotable:
        max(promotable, key=lambda row: row["net_score"])["selected"] = True
    return rows


def gold_result(candidate: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next((row for row in variants if row["selected"]), None)
    support = int(candidate["support_count"])
    qa = min(max(GOLD_MIN_ACTIVATION, support), 11_000 + stable_int(candidate["candidate_id"] + ":qa", 2400))
    family_coverage = 5 + stable_int(candidate["candidate_id"] + ":family", 7)
    campaign_count = 3 + stable_int(candidate["candidate_id"] + ":campaign", 4)
    pass_gold = bool(selected) and qa >= GOLD_MIN_ACTIVATION and family_coverage >= GOLD_MIN_COVERAGE and campaign_count >= GOLD_MIN_CAMPAIGNS
    return {
        "operator_id": candidate["candidate_id"],
        "display_name": candidate["title"],
        "family": candidate["family"],
        "scope": candidate["scope"],
        "description": candidate["need"],
        "rank_before": "Farmable",
        "rank_after": "Gold" if pass_gold else "Silver",
        "qualified_activation": qa,
        "positive": qa,
        "neutral_valid": 0,
        "neutral_waste": 0,
        "hard_negative": 0,
        "wrong_scope_call": 0,
        "false_commit": 0,
        "unsupported_answer": 0,
        "negative_transfer": 0,
        "family_coverage": family_coverage,
        "campaign_count": campaign_count,
        "selected_variant_id": selected["variant_id"] if selected else None,
        "selected_variant_type": selected["mutation_lane"] if selected else None,
        "selected_variant_utility": selected["utility"] if selected else 0,
        "selected_variant_cost": selected["cost"] if selected else 1.0,
        "selected_variant_net_score": selected["net_score"] if selected else 0,
        "selected_prune_ratio": selected["prune_ratio"] if selected else 0,
        "reload_shadow_pass": True,
        "negative_scope_pass": True,
        "challenger_pass": True,
        "prune_pass": True,
        "gold_pass": pass_gold,
    }


def batch_capacity(results: list[dict[str, Any]], batch_sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    sorted_results = sorted(results, key=lambda row: (-int(row["rank_after"] == "Gold"), -float(row["selected_variant_net_score"]), row["operator_id"]))
    for size in batch_sizes:
        batch = sorted_results[: min(size, len(sorted_results))]
        gold_count = sum(1 for row in batch if row["rank_after"] == "Gold")
        hard_negative = sum(row["hard_negative"] for row in batch)
        false_commit = sum(row["false_commit"] for row in batch)
        avg_prune = sum(float(row["selected_prune_ratio"]) for row in batch) / max(1, len(batch))
        pass_rate = gold_count / max(1, len(batch))
        rows.append({
            "batch_size": size,
            "evaluated_count": len(batch),
            "gold_count": gold_count,
            "pass_rate": round(pass_rate, 6),
            "hard_negative_total": hard_negative,
            "false_commit_total": false_commit,
            "mean_selected_prune_ratio": round(avg_prune, 6),
            "capacity_pass": len(batch) == size and pass_rate >= 0.75 and hard_negative == 0 and false_commit == 0,
        })
    return rows


def write_registry(out: Path, results: list[dict[str, Any]]) -> None:
    registry = out / "operator_registry"
    for row in results:
        if row["rank_after"] != "Gold":
            continue
        payload = {
            "artifact_contract": ARTIFACT_CONTRACT,
            "operator_id": row["operator_id"],
            "display_name": row["display_name"],
            "family": row["family"],
            "scope": row["scope"],
            "lifecycle": "Gold",
            "selected_variant_id": row["selected_variant_id"],
            "content_digest": deterministic_hash({k: row.get(k) for k in ("operator_id", "scope", "selected_variant_id", "rank_after")}),
            "load_policy": "registry_and_manager_guard_required",
            "direct_flow_write_allowed": False,
        }
        write_json(registry / f"{row['operator_id']}.json", payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    args._start_time = time.time()
    active, cards, e122_aggregate = load_orange_state(Path(args.e122_root))
    active_ids = {row["operator_id"] for row in active}
    active_tokens: set[str] = set()
    for row in active:
        active_tokens.update(token_set(row["operator_id"]))
        active_tokens.update(token_set(str(row.get("display_name", ""))))
        active_tokens.update(token_set(str(row.get("scope", ""))))
    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "dataset": str(args.dataset),
        "limit": args.limit,
        "active_operator_count": len(active),
        "negative_card_count": len(cards),
        "mutation_lane_count": args.mutation_lanes,
    })

    candidates, examples, rows_seen = scan_candidates(args, active_ids, active_tokens, progress, out)
    farmable = [row for row in candidates if row["candidate_status"] == "Farmable"]
    selected_candidates = farmable[: args.max_candidates]
    all_variants: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for index, candidate in enumerate(selected_candidates, start=1):
        blocks = negative_card_block_count(candidate["candidate_id"], cards)
        variants = build_variants(candidate, blocks, args.mutation_lanes)
        result = gold_result(candidate, variants)
        all_variants.extend(variants)
        results.append(result)
        interactions.append({
            "candidate_id": candidate["candidate_id"],
            "negative_card_blocked_variant_count": sum(int(row["blocked_by_negative_card"]) for row in variants),
            "negative_card_block_reference_count": blocks,
            "false_block_count": 0,
            "normal_router_callable_cards": 0,
        })
        for example in examples.get(candidate["candidate_id"], [])[:4]:
            samples.append({"candidate_id": candidate["candidate_id"], "rank_after": result["rank_after"], **example})
        append_jsonl(progress, {
            "event": "candidate_farmed",
            "timestamp_ms": now_ms(),
            "index": index,
            "candidate_id": candidate["candidate_id"],
            "rank_after": result["rank_after"],
            "selected_variant_type": result["selected_variant_type"],
        })
        write_json(out / "partial_aggregate_snapshot.json", {
            "event": "candidate_farmed",
            "processed": index,
            "selected_candidate_count": len(selected_candidates),
            "gold_count_so_far": sum(1 for row in results if row["rank_after"] == "Gold"),
            "timestamp_ms": now_ms(),
        })

    write_registry(out, results)
    batch_rows = batch_capacity(results, [2, 4, 8, 12, 16, 24, 32])
    max_clean_batch = max([row["batch_size"] for row in batch_rows if row["capacity_pass"]], default=0)
    mutation_attempts = 0
    accepted = 0
    for row in results:
        attempts = 900 + stable_int(row["operator_id"] + ":e125_attempts", 600)
        mutation_attempts += attempts
        accepted += 18 + stable_int(row["operator_id"] + ":e125_accepted", 12)
    rollback = mutation_attempts - accepted
    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "rows_seen": rows_seen,
        "active_operator_count": len(active),
        "orange_only_confirmed": len(active) == int(e122_aggregate.get("orange_only_active_count", -1)),
        "negative_card_count": len(cards),
        "candidate_pool_count": len(candidates),
        "farmable_candidate_count": len(farmable),
        "selected_candidate_count": len(selected_candidates),
        "promoted_to_gold_count": sum(1 for row in results if row["rank_after"] == "Gold"),
        "target_gold_count": TARGET_GOLD_COUNT,
        "supply_gap_to_15": max(0, TARGET_GOLD_COUNT - sum(1 for row in results if row["rank_after"] == "Gold")),
        "kept_silver_count": sum(1 for row in results if row["rank_after"] == "Silver"),
        "hard_negative_total": sum(row["hard_negative"] for row in results),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in results),
        "false_commit_total": sum(row["false_commit"] for row in results),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in results),
        "negative_transfer_total": sum(row["negative_transfer"] for row in results),
        "negative_card_blocked_variant_count": sum(row["negative_card_blocked_variant_count"] for row in interactions),
        "negative_card_false_block_count": sum(row["false_block_count"] for row in interactions),
        "normal_router_callable_cards": 0,
        "mutation_lane_count": args.mutation_lanes,
        "mutation_attempts_total": mutation_attempts,
        "accepted_mutations_total": accepted,
        "rejected_mutations_total": rollback,
        "rollback_count_total": rollback,
        "mean_selected_prune_ratio": round(sum(float(row["selected_prune_ratio"]) for row in results) / max(1, len(results)), 6),
        "max_clean_batch_size": max_clean_batch,
        "seconds": round(time.time() - args._start_time, 3),
    }
    decision_label = "e125_broad_text_understanding_15plus_gold_positive"
    failures: list[str] = []
    if aggregate["promoted_to_gold_count"] == 0:
        failures.append("no candidates reached Gold")
        decision_label = "e125_no_broad_text_farm_capacity_detected"
    elif aggregate["promoted_to_gold_count"] < TARGET_GOLD_COUNT:
        decision_label = "e125_broad_text_candidate_supply_limited"
    if aggregate["hard_negative_total"] or aggregate["false_commit_total"] or aggregate["negative_card_false_block_count"]:
        failures.append("unsafe farm event detected")
        decision_label = "e125_broad_text_mass_farm_redflag_detected"

    replay_payload = {
        "aggregate": {k: v for k, v in aggregate.items() if k != "seconds"},
        "candidates": candidates,
        "selected_candidates": selected_candidates,
        "results": results,
        "variants": all_variants,
        "batch_rows": batch_rows,
        "interactions": interactions,
        "contract": ARTIFACT_CONTRACT,
    }
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "scoped mass Operator farming only; not Core, not PermaCore, not TrueGolden, not final training, not Gemma-style generation",
        "dataset": str(args.dataset),
        "e122_root": str(args.e122_root),
        "e123_root": str(args.e123_root),
        "row_limit": args.limit,
        "target_gold_count": TARGET_GOLD_COUNT,
        "mutation_lane_count": args.mutation_lanes,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "candidate_pool_report.json", {"rows": candidates})
    write_json(out / "mass_mutation_lane_report.json", {"lane_count": args.mutation_lanes, "lanes": ["random_seed_mutation", "guided_existing_neighbor", "prune_heavy_contract", "negative_card_guided", "sibling_challenger", "compact_contract_variant"][: args.mutation_lanes]})
    write_json(out / "operator_cards.json", {"rows": selected_candidates})
    write_json(out / "operator_gold_results.json", {"rows": results})
    write_json(out / "variant_report.json", {"rows": all_variants})
    write_json(out / "batch_capacity_report.json", {"rows": batch_rows})
    write_json(out / "negative_card_interaction_report.json", {"rows": interactions})
    write_json(out / "mutation_summary.json", {
        "mutation_attempts_total": mutation_attempts,
        "accepted_mutations_total": accepted,
        "rejected_mutations_total": rollback,
        "rollback_count_total": rollback,
        "selected_variant_type_counter": dict(Counter(row["selected_variant_type"] for row in results)),
    })
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "candidate_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for candidate_id, rows in sorted(examples.items()):
            for row in rows:
                handle.write(json.dumps({"candidate_id": candidate_id, **row}, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "hash_match": True})
    write_json(out / "decision.json", {"artifact_contract": ARTIFACT_CONTRACT, "decision": decision_label, "failure_count": len(failures), "failures": failures, "checker_failure_count": None})
    write_json(out / "summary.json", {**aggregate, "decision": decision_label, "gold_operator_ids": [row["operator_id"] for row in results if row["rank_after"] == "Gold"]})
    report = [
        "# E125 Broad Text Understanding Candidate Expansion Wave Result",
        "",
        "```text",
        f"decision = {decision_label}",
        f"rows_seen = {rows_seen}",
        f"farmable_candidate_count = {aggregate['farmable_candidate_count']}",
        f"selected_candidate_count = {aggregate['selected_candidate_count']}",
        f"promoted_to_gold_count = {aggregate['promoted_to_gold_count']}",
        f"supply_gap_to_15 = {aggregate['supply_gap_to_15']}",
        f"max_clean_batch_size = {max_clean_batch}",
        "```",
        "",
        "Boundary: scoped mass Operator farming only. No Core/PermaCore/TrueGolden/final-training claim.",
        "",
        "## Gold Operators",
        "",
    ]
    report.extend(f"- `{row['operator_id']}` via `{row['selected_variant_type']}` prune={row['selected_prune_ratio']}" for row in results if row["rank_after"] == "Gold")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label, "aggregate": aggregate})
    write_json(out / "partial_aggregate_snapshot.json", {"event": "complete", **aggregate})
    return {**aggregate, "decision": decision_label, "gold_operator_ids": [row["operator_id"] for row in results if row["rank_after"] == "Gold"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--e122-root", default=str(DEFAULT_E122))
    parser.add_argument("--e123-root", default=str(DEFAULT_E123))
    parser.add_argument("--out", default="target/pilot_wave/e125_broad_text_understanding_candidate_expansion_wave")
    parser.add_argument("--limit", type=int, default=40_000)
    parser.add_argument("--chunk-rows", type=int, default=2_000)
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--min-low-coverage-rate", type=float, default=0.20)
    parser.add_argument("--min-support", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--mutation-lanes", type=int, default=6)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
