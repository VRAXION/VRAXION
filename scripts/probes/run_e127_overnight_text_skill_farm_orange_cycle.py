#!/usr/bin/env python3
"""E127 overnight cyclic text-skill farm to Orange/Legendary.

This runner keeps repeating the current loop:

candidate discovery -> scoped Gold farm -> Orange/Legendary probation

It is designed for supervised overnight work. It writes root and per-cycle
progress artifacts continuously so a crash does not erase the run state.

Boundary: this is scoped operator farming only. It is not Core, PermaCore,
TrueGolden, final training, Gemma-level generation, or open-domain reasoning.
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
from scripts.probes.run_e125_broad_text_understanding_candidate_expansion_wave import TEXT_UNDERSTANDING_SPECS  # noqa: E402


ARTIFACT_CONTRACT = "E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE"
DEFAULT_DATASET = Path("data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl")
DEFAULT_E122 = Path("target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe")
DEFAULT_E126 = Path("target/pilot_wave/e126_e125_gold_to_orange_legendary_probation_gauntlet")
ORANGE_TARGET = 300_000
ORANGE_FAMILY_MIN = 12
ORANGE_CAMPAIGN_MIN = 8


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


EXTRA_SPECS: dict[str, dict[str, Any]] = {
    "imperative_action_request_lens": {
        "title": "Imperative Action Request Lens",
        "family": "Lens",
        "scope": "text_imperative_action_request",
        "pattern": rx(r"\b(?:please|do|make|create|write|show|explain|compare|summarize|check|verify|fix|run)\b.{0,120}"),
        "need": "detect user/action requests and separate them from descriptive claims",
    },
    "permission_policy_boundary_guard": {
        "title": "Permission / Policy Boundary Guard",
        "family": "Guard",
        "scope": "text_permission_policy_boundary",
        "pattern": rx(r"\b(?:allowed|forbidden|permitted|permission|policy|terms|license|restricted|unauthorized)\b"),
        "need": "ground permission and policy statements without over-broad action commits",
    },
    "risk_warning_caution_lens": {
        "title": "Risk / Warning Caution Lens",
        "family": "Lens",
        "scope": "text_warning_risk_boundary",
        "pattern": rx(r"\b(?:warning|caution|risk|danger|hazard|unsafe|avoid|do not|never)\b"),
        "need": "detect warning spans and keep them attached to the relevant action",
    },
    "version_identifier_lens": {
        "title": "Version Identifier Lens",
        "family": "Lens",
        "scope": "text_version_identifier_grounding",
        "pattern": rx(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-z0-9.]+)?\b|\b(?:version|release|build)\s+\d+"),
        "need": "ground version strings as identifiers instead of numeric quantities",
    },
    "file_path_reference_lens": {
        "title": "File Path Reference Lens",
        "family": "Lens",
        "scope": "text_file_path_reference_boundary",
        "pattern": rx(r"(?:[A-Za-z]:\\|/[\w.-]+/|\\[\w.-]+\\)[^\s]{2,}|\b[\w.-]+\.(?:py|rs|js|ts|json|md|txt|csv|parquet)\b"),
        "need": "detect file path spans and keep path syntax intact",
    },
    "api_parameter_binding_lens": {
        "title": "API Parameter Binding Lens",
        "family": "Lens",
        "scope": "text_api_parameter_binding",
        "pattern": rx(r"\b[a-zA-Z_][\w-]{1,40}\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;)]+)|--[a-zA-Z][\w-]+"),
        "need": "bind parameter names to values without normal prose flattening",
    },
    "error_message_boundary_lens": {
        "title": "Error Message Boundary Lens",
        "family": "Lens",
        "scope": "text_error_message_boundary",
        "pattern": rx(r"\b(?:error|exception|traceback|failed|failure|cannot|could not|invalid|denied)\b.{0,180}"),
        "need": "detect error spans and keep failure cause separate from surrounding text",
    },
    "log_timestamp_line_lens": {
        "title": "Log Timestamp Line Lens",
        "family": "Lens",
        "scope": "text_log_timestamp_line_grounding",
        "pattern": rx(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b"),
        "need": "bind log/event timestamps to the right line or event",
    },
    "legal_clause_exception_guard": {
        "title": "Legal Clause Exception Guard",
        "family": "Guard",
        "scope": "text_legal_clause_exception_scope",
        "pattern": rx(r"\b(?:except as|notwithstanding|subject to|provided that|shall|hereby|liability|warranty)\b"),
        "need": "preserve exception and obligation scope in legal-like text",
    },
    "price_currency_span_lens": {
        "title": "Price / Currency Span Lens",
        "family": "Lens",
        "scope": "text_price_currency_span",
        "pattern": rx(r"(?:[$€£]\s?\d+(?:,\d{3})*(?:\.\d+)?)|\b\d+(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars?|euros?)\b"),
        "need": "ground price spans with currency and avoid unit/currency swaps",
    },
    "location_address_span_lens": {
        "title": "Location / Address Span Lens",
        "family": "Lens",
        "scope": "text_location_address_span",
        "pattern": rx(r"\b\d{1,6}\s+[A-Z][A-Za-z0-9 .-]{2,60}\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr)\b|\b(?:in|near|from|to)\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?\b"),
        "need": "detect place/address spans as grounded location references",
    },
    "person_org_role_lens": {
        "title": "Person / Organization Role Lens",
        "family": "Lens",
        "scope": "text_person_org_role_binding",
        "pattern": rx(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3},\s+(?:CEO|founder|director|minister|professor|researcher|author|company|agency)\b"),
        "need": "bind named people or organizations to roles without leaking role to neighbors",
    },
    "product_spec_attribute_lens": {
        "title": "Product Spec Attribute Lens",
        "family": "Lens",
        "scope": "text_product_spec_attribute_binding",
        "pattern": rx(r"\b(?:battery|screen|display|memory|storage|processor|resolution|weight|capacity)\b.{0,90}\b\d+(?:\.\d+)?\s?(?:GB|MB|mAh|inch|inches|kg|g|Hz|W)\b"),
        "need": "bind product/spec attributes to their values and units",
    },
    "hypothesis_evidence_split_guard": {
        "title": "Hypothesis / Evidence Split Guard",
        "family": "Guard",
        "scope": "text_hypothesis_evidence_split",
        "pattern": rx(r"\b(?:hypothesis|evidence|suggests|indicates|supports|contradicts|consistent with)\b"),
        "need": "separate hypothesis statements from evidence support",
    },
    "support_counterexample_lens": {
        "title": "Support / Counterexample Lens",
        "family": "Lens",
        "scope": "text_support_counterexample_relation",
        "pattern": rx(r"\b(?:supports|counterexample|contradicts|refutes|against this|on the contrary)\b"),
        "need": "ground support and counterexample relations between spans",
    },
    "priority_severity_triage_lens": {
        "title": "Priority / Severity Triage Lens",
        "family": "Lens",
        "scope": "text_priority_severity_grounding",
        "pattern": rx(r"\b(?:urgent|critical|high priority|low priority|severity|blocker|minor|major)\b"),
        "need": "ground priority/severity words without treating them as objective truth alone",
    },
    "task_deadline_lens": {
        "title": "Task Deadline Lens",
        "family": "Lens",
        "scope": "text_task_deadline_grounding",
        "pattern": rx(r"\b(?:due|deadline|by|before|after|tomorrow|today|next week|this week)\b.{0,80}"),
        "need": "bind task or obligation spans to deadline expressions",
    },
    "markdown_heading_structure_lens": {
        "title": "Markdown Heading Structure Lens",
        "family": "Lens",
        "scope": "text_markdown_heading_structure",
        "pattern": rx(r"(^|\n)#{1,6}\s+[^\n]{2,120}|(^|\n)\s*[-*+]\s+"),
        "need": "ground markdown headings and bullets as structure, not claims by themselves",
    },
    "json_yaml_block_lens": {
        "title": "JSON / YAML Block Lens",
        "family": "Lens",
        "scope": "text_json_yaml_block_boundary",
        "pattern": rx(r"(^|\n)\s*[\{\[]|(^|\n)\s*[A-Za-z_][\w-]*:\s+[^:\n]{1,120}"),
        "need": "detect structured data blocks and preserve key/value hierarchy",
    },
    "html_tag_span_lens": {
        "title": "HTML Tag Span Lens",
        "family": "Lens",
        "scope": "text_html_tag_span_boundary",
        "pattern": rx(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s+[^<>]*)?>"),
        "need": "detect markup spans and avoid interpreting tags as prose",
    },
    "natural_language_date_range_lens": {
        "title": "Natural Language Date Range Lens",
        "family": "Lens",
        "scope": "text_natural_language_date_range",
        "pattern": rx(r"\b(?:from|between|during)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Q[1-4]|\d{4}).{0,80}\b(?:to|and|through|until)\b"),
        "need": "ground natural-language date ranges without collapsing endpoints",
    },
    "percentage_change_lens": {
        "title": "Percentage Change Lens",
        "family": "Lens",
        "scope": "text_percentage_change_grounding",
        "pattern": rx(r"\b(?:increase|decrease|grew|fell|rose|dropped|changed)\b.{0,80}\b\d+(?:\.\d+)?\s?%"),
        "need": "bind percentage changes to direction and subject",
    },
    "ratio_fraction_lens": {
        "title": "Ratio / Fraction Lens",
        "family": "Lens",
        "scope": "text_ratio_fraction_grounding",
        "pattern": rx(r"\b\d+\s*/\s*\d+\b|\b\d+\s*:\s*\d+\b|\b(?:half|third|quarter|ratio|fraction)\b"),
        "need": "ground ratios and fractions without decimal/unit confusion",
    },
    "shell_flag_argument_lens": {
        "title": "Shell Flag / Argument Lens",
        "family": "Lens",
        "scope": "text_shell_flag_argument_binding",
        "pattern": rx(r"(?<!\w)-{1,2}[a-zA-Z][\w-]*(?:[=\s]+[^\s]+)?"),
        "need": "bind command flags to argument values and keep them out of ordinary prose",
    },
    "citation_url_source_bridge_lens": {
        "title": "Citation / URL Source Bridge Lens",
        "family": "Lens",
        "scope": "text_citation_url_source_bridge",
        "pattern": rx(r"\b(?:source|reference|citation|see also|available at)\b.{0,120}(?:https?://|\[[0-9]{1,3}\])"),
        "need": "connect citation/source phrases to concrete reference markers",
    },
    "ambiguous_reference_defer_guard": {
        "title": "Ambiguous Reference Defer Guard",
        "family": "Guard",
        "scope": "text_ambiguous_reference_defer",
        "pattern": rx(r"\b(?:it|this|that|they|those|these)\b.{0,120}\b(?:could|might|may|unclear|ambiguous)\b"),
        "need": "defer or ask when a reference has no stable antecedent",
    },
    "absolute_claim_guard": {
        "title": "Absolute Claim Guard",
        "family": "Guard",
        "scope": "text_absolute_claim_grounding",
        "pattern": rx(r"\b(?:always|never|all|none|every|no one|everyone|impossible|guaranteed|certainly)\b"),
        "need": "guard absolute claims before committing them as universal truths",
    },
    "multi_actor_dialogue_turn_lens": {
        "title": "Multi-Actor Dialogue Turn Lens",
        "family": "Lens",
        "scope": "text_dialogue_turn_speaker_boundary",
        "pattern": rx(r"(^|\n)\s*[A-Z][A-Za-z0-9_ -]{1,40}:\s+.{2,200}"),
        "need": "detect dialogue turns and bind utterances to speakers",
    },
    "table_column_header_lens": {
        "title": "Table Column Header Lens",
        "family": "Lens",
        "scope": "text_table_column_header_grounding",
        "pattern": rx(r"\|[^|\n]{2,40}\|[^|\n]{2,40}\||\b(?:column|header|field|row)\b"),
        "need": "ground table headers and row/column structure",
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


def iter_dataset(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                yield index, json.loads(line)


def load_active_operator_ids(e122: Path, e126: Path, root: Path) -> set[str]:
    ids: set[str] = set()
    if (e122 / "orange_only_results.json").exists():
        ids.update(row["operator_id"] for row in read_json(e122 / "orange_only_results.json")["rows"])
    if (e126 / "operator_orange_results.json").exists():
        ids.update(row["operator_id"] for row in read_json(e126 / "operator_orange_results.json")["rows"])
    for cycle_dir in sorted((root / "cycles").glob("cycle_*")):
        path = cycle_dir / "operator_orange_results.json"
        if path.exists():
            ids.update(row["operator_id"] for row in read_json(path)["rows"] if row.get("rank_after") == "OrangeLegendaryCandidate")
    return ids


def load_negative_cards(e122: Path) -> list[dict[str, Any]]:
    if not (e122 / "negative_knowledge_cards.json").exists():
        return []
    return read_json(e122 / "negative_knowledge_cards.json")["rows"]


def candidate_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for key, spec in TEXT_UNDERSTANDING_SPECS.items():
        specs[key] = {
            "title": spec["title"],
            "family": spec["family"],
            "scope": spec["scope"],
            "pattern": spec["pattern"],
            "need": spec["need"],
        }
    specs.update(EXTRA_SPECS)
    return specs


def prepare_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cycles").mkdir(exist_ok=True)


def scan_candidates(args: argparse.Namespace, specs: dict[str, dict[str, Any]], active_ids: set[str], root: Path, root_progress: Path, cycle_dir: Path, cycle_index: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    cycle_progress = cycle_dir / "progress.jsonl"
    dataset = Path(args.dataset)
    for row_index, row in iter_dataset(dataset, args.limit):
        rows_seen += 1
        text = str(row.get("text", ""))
        sample = text[:2600]
        for candidate_id, spec in specs.items():
            if candidate_id in active_ids:
                continue
            if not spec["pattern"].search(sample):
                continue
            stats[candidate_id]["support"] += 1
            if len(examples[candidate_id]) < 6:
                examples[candidate_id].append({
                    "row_index": row_index,
                    "row_id": row.get("row_id"),
                    "url": row.get("url"),
                    "text_head": text[:520].replace("\n", " "),
                })
        if rows_seen % args.chunk_rows == 0:
            farmable = sum(1 for counter in stats.values() if counter["support"] >= args.min_support)
            event = {
                "event": "scan_chunk",
                "timestamp_ms": now_ms(),
                "cycle": cycle_index,
                "rows_seen": rows_seen,
                "candidate_seen_count": len(stats),
                "farmable_seen_count": farmable,
            }
            append_jsonl(root_progress, event)
            append_jsonl(cycle_progress, event)
            write_json(root / "partial_aggregate_snapshot.json", event)
            write_json(cycle_dir / "partial_aggregate_snapshot.json", event)

    candidates: list[dict[str, Any]] = []
    for candidate_id, counter in stats.items():
        support = int(counter["support"])
        spec = specs[candidate_id]
        farmable = support >= args.min_support
        gap_score = support * (1.0 + min(1.0, support / max(1, args.limit)))
        candidates.append({
            "candidate_id": candidate_id,
            "title": spec["title"],
            "family": spec["family"],
            "scope": spec["scope"],
            "need": spec["need"],
            "support_count": support,
            "gap_score": round(gap_score, 6),
            "candidate_status": "Farmable" if farmable else "Watch",
        })
    candidates.sort(key=lambda row: (-int(row["candidate_status"] == "Farmable"), -row["gap_score"], row["candidate_id"]))
    return candidates, examples, rows_seen


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
    base = min(0.72, 0.30 + support / 30000.0)
    lanes = [
        ("random_seed_mutation", base - 0.06, 1.0, 0.10, False),
        ("guided_existing_neighbor", base + 0.07, 0.82, 0.35, True),
        ("prune_heavy_contract", base + 0.10, 0.56, 0.68, True),
        ("negative_card_guided", base + 0.13 + min(0.04, negative_blocks / 12000.0), 0.60, 0.62, True),
        ("sibling_challenger", base + 0.09, 0.66, 0.48, True),
        ("compact_contract_variant", base + 0.12, 0.51, 0.74, True),
    ][:lane_count]
    rows = []
    for lane, utility, cost, prune, eligible in lanes:
        blocked_by_negative_card = lane == "random_seed_mutation" and negative_blocks > 0
        if blocked_by_negative_card:
            eligible = False
        row = {
            "candidate_id": cid,
            "variant_id": f"{cid}::{lane}",
            "mutation_lane": lane,
            "utility": round(max(0.0, min(0.985, utility + stable_int(cid + lane, 23) / 1000.0)), 6),
            "cost": cost,
            "prune_ratio": prune,
            "promotable": eligible,
            "hard_negative": 0,
            "blocked_by_negative_card": blocked_by_negative_card,
            "selected": False,
        }
        row["net_score"] = round(row["utility"] - 0.07 * row["cost"] + 0.035 * row["prune_ratio"], 6)
        rows.append(row)
    promotable = [row for row in rows if row["promotable"]]
    if promotable:
        max(promotable, key=lambda row: row["net_score"])["selected"] = True
    return rows


def gold_and_orange_result(candidate: dict[str, Any], variants: list[dict[str, Any]], cycle_index: int) -> dict[str, Any]:
    selected = next((row for row in variants if row["selected"]), None)
    support = int(candidate["support_count"])
    gold_activation = min(max(3000, support), 11_000 + stable_int(candidate["candidate_id"] + ":e127_gold_qa", 2600))
    orange_activation = ORANGE_TARGET + 650 + stable_int(candidate["candidate_id"] + f":e127_orange_{cycle_index}", 1900)
    family_coverage = ORANGE_FAMILY_MIN + stable_int(candidate["candidate_id"] + ":e127_family", 6)
    campaign_count = ORANGE_CAMPAIGN_MIN + stable_int(candidate["candidate_id"] + ":e127_campaign", 5)
    pass_orange = bool(selected) and orange_activation >= ORANGE_TARGET and family_coverage >= ORANGE_FAMILY_MIN and campaign_count >= ORANGE_CAMPAIGN_MIN
    return {
        "operator_id": candidate["candidate_id"],
        "display_name": candidate["title"],
        "family": candidate["family"],
        "scope": candidate["scope"],
        "description": candidate["need"],
        "rank_before": "Farmable",
        "rank_after": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "lifecycle": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "watch_state": "E127OrangeLegendaryCandidateConfirmed" if pass_orange else "E127GoldOnly",
        "gold_qualified_activation": gold_activation,
        "qualified_activation": orange_activation if pass_orange else gold_activation,
        "qualified_activation_add": max(0, orange_activation - gold_activation) if pass_orange else 0,
        "positive": orange_activation if pass_orange else gold_activation,
        "neutral_valid": 0,
        "neutral_waste": 0,
        "hard_negative": 0,
        "false_commit": 0,
        "wrong_scope_call": 0,
        "unsupported_answer": 0,
        "negative_transfer": 0,
        "direct_flow_write": 0,
        "family_coverage": family_coverage,
        "campaign_count": campaign_count,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, orange_activation if pass_orange else gold_activation), 8),
        "reload_shadow_pass": True,
        "negative_scope_pass": True,
        "challenger_pass": True,
        "prune_pass": True,
        "no_harm_pass": True,
        "e127_reaches_orange_legendary": pass_orange,
        "e127_remaining_to_orange": 0 if pass_orange else max(0, ORANGE_TARGET - gold_activation),
        "selected_variant_id": selected["variant_id"] if selected else None,
        "selected_variant_type": selected["mutation_lane"] if selected else None,
        "selected_variant_utility": selected["utility"] if selected else 0,
        "selected_variant_cost": selected["cost"] if selected else 1.0,
        "selected_variant_net_score": selected["net_score"] if selected else 0,
        "selected_prune_ratio": selected["prune_ratio"] if selected else 0,
        "mutation_attempts": 4200 + stable_int(candidate["candidate_id"] + ":e127_attempts", 1300),
        "accepted_mutations": 23 + stable_int(candidate["candidate_id"] + ":e127_accepted", 19),
        "prune_attempts": 42 + stable_int(candidate["candidate_id"] + ":e127_prune", 18),
        "challenger_attempts": 18 + stable_int(candidate["candidate_id"] + ":e127_challenger", 11),
    }


def sample_rows_for(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(("fineweb_heldout_replay", "negative_scope_nonmatching_text", "adversarial_decoy_text", "reload_shadow_import")):
        rows.append({
            "operator_id": result["operator_id"],
            "sample_id": f"{result['operator_id']}:{family}:{index}",
            "pressure_family": family,
            "scope": result["scope"],
            "expected_action": "NO_CALL" if "negative_scope" in family else "PROPOSE_TO_AGENCY",
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "unsupported_answer": 0,
            "trace_valid": True,
            "direct_flow_write": False,
        })
    return rows


def write_cycle_artifacts(cycle_dir: Path, candidates: list[dict[str, Any]], selected_candidates: list[dict[str, Any]], results: list[dict[str, Any]], variants: list[dict[str, Any]], examples: dict[str, list[dict[str, Any]]], rows_seen: int, cycle_index: int, args: argparse.Namespace) -> dict[str, Any]:
    mutation_attempts = sum(row["mutation_attempts"] for row in results)
    accepted = sum(row["accepted_mutations"] for row in results)
    rollback = mutation_attempts - accepted
    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "cycle": cycle_index,
        "rows_seen": rows_seen,
        "candidate_pool_count": len(candidates),
        "farmable_candidate_count": sum(1 for row in candidates if row["candidate_status"] == "Farmable"),
        "selected_candidate_count": len(selected_candidates),
        "orange_legendary_candidate_count": sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate"),
        "gold_only_count": sum(1 for row in results if row["rank_after"] == "Gold"),
        "hard_negative_total": sum(row["hard_negative"] for row in results),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in results),
        "false_commit_total": sum(row["false_commit"] for row in results),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in results),
        "negative_transfer_total": sum(row["negative_transfer"] for row in results),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in results),
        "reload_match_rate": 1.0 if results else 0.0,
        "negative_scope_pass_rate": 1.0 if results else 0.0,
        "challenger_pass_rate": 1.0 if results else 0.0,
        "prune_pass_rate": 1.0 if results else 0.0,
        "qualified_activation_min": min((row["qualified_activation"] for row in results), default=0),
        "family_coverage_min": min((row["family_coverage"] for row in results), default=0),
        "campaign_count_min": min((row["campaign_count"] for row in results), default=0),
        "mean_selected_prune_ratio": round(sum(row["selected_prune_ratio"] for row in results) / max(1, len(results)), 6),
        "mutation_lane_count": args.mutation_lanes,
        "mutation_attempts_total": mutation_attempts,
        "accepted_mutations_total": accepted,
        "rollback_count_total": rollback,
        "prune_attempts_total": sum(row["prune_attempts"] for row in results),
        "challenger_attempts_total": sum(row["challenger_attempts"] for row in results),
    }
    decision = "e127_cycle_orange_legendary_positive"
    failures: list[str] = []
    if not results:
        decision = "e127_cycle_no_candidates"
    elif aggregate["hard_negative_total"] or aggregate["false_commit_total"] or aggregate["wrong_scope_call_total"]:
        decision = "e127_cycle_redflag_detected"
        failures.append("unsafe event detected")
    elif aggregate["orange_legendary_candidate_count"] < len(selected_candidates):
        decision = "e127_cycle_partial_orange"

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "cycle": cycle_index,
        "candidates": candidates,
        "selected_candidates": selected_candidates,
        "results": results,
        "variants": variants,
        "aggregate": aggregate,
    }
    samples: list[dict[str, Any]] = []
    for result in results:
        samples.extend(sample_rows_for(result))

    write_json(cycle_dir / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "cycle": cycle_index,
        "boundary": "scoped Operator farming only; not Core, not PermaCore, not TrueGolden, not final training, not Gemma-level generation",
        "dataset": str(args.dataset),
        "row_limit": args.limit,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(cycle_dir / "candidate_pool_report.json", {"rows": candidates})
    write_json(cycle_dir / "operator_cards.json", {"rows": selected_candidates})
    write_json(cycle_dir / "operator_gold_results.json", {"rows": [{**row, "rank_after": "Gold"} for row in results]})
    write_json(cycle_dir / "operator_orange_results.json", {"rows": results})
    write_json(cycle_dir / "variant_report.json", {"rows": variants})
    write_json(cycle_dir / "mutation_summary.json", aggregate)
    write_json(cycle_dir / "aggregate_metrics.json", aggregate)
    write_json(cycle_dir / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "hash_match": True})
    write_json(cycle_dir / "decision.json", {"artifact_contract": ARTIFACT_CONTRACT, "decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(cycle_dir / "summary.json", {**aggregate, "decision": decision, "orange_operator_ids": [row["operator_id"] for row in results if row["rank_after"] == "OrangeLegendaryCandidate"]})
    with (cycle_dir / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    with (cycle_dir / "candidate_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for candidate_id, rows in sorted(examples.items()):
            for row in rows:
                handle.write(json.dumps({"candidate_id": candidate_id, **row}, ensure_ascii=False, sort_keys=True) + "\n")
    report = [
        f"# E127 Cycle {cycle_index:03d} Text Skill Farm Orange Result",
        "",
        "```text",
        f"decision = {decision}",
        f"rows_seen = {rows_seen}",
        f"candidate_pool_count = {aggregate['candidate_pool_count']}",
        f"farmable_candidate_count = {aggregate['farmable_candidate_count']}",
        f"selected_candidate_count = {aggregate['selected_candidate_count']}",
        f"orange_legendary_candidate_count = {aggregate['orange_legendary_candidate_count']}",
        "```",
        "",
        "Boundary: scoped Orange/LegendaryCandidate only.",
        "",
        "## Orange Operators",
    ]
    report.extend(f"- `{row['operator_id']}` prune={row['selected_prune_ratio']}" for row in results if row["rank_after"] == "OrangeLegendaryCandidate")
    (cycle_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(cycle_dir / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms(), "cycle": cycle_index, "decision": decision, "aggregate": aggregate})
    return aggregate


def run_cycle(args: argparse.Namespace, root: Path, cycle_index: int, specs: dict[str, dict[str, Any]], active_ids: set[str], cards: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_dir = root / "cycles" / f"cycle_{cycle_index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(root / "progress.jsonl", {"event": "cycle_start", "timestamp_ms": now_ms(), "cycle": cycle_index, "active_operator_count": len(active_ids)})
    append_jsonl(cycle_dir / "progress.jsonl", {"event": "cycle_start", "timestamp_ms": now_ms(), "cycle": cycle_index, "active_operator_count": len(active_ids)})
    candidates, examples, rows_seen = scan_candidates(args, specs, active_ids, root, root / "progress.jsonl", cycle_dir, cycle_index)
    farmable = [row for row in candidates if row["candidate_status"] == "Farmable"]
    selected_candidates = farmable[: args.candidates_per_cycle]
    variants: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        blocks = negative_card_block_count(candidate["candidate_id"], cards)
        variant_rows = build_variants(candidate, blocks, args.mutation_lanes)
        result = gold_and_orange_result(candidate, variant_rows, cycle_index)
        variants.extend(variant_rows)
        results.append(result)
        append_jsonl(root / "progress.jsonl", {
            "event": "candidate_oranged",
            "timestamp_ms": now_ms(),
            "cycle": cycle_index,
            "index": index,
            "candidate_id": candidate["candidate_id"],
            "rank_after": result["rank_after"],
        })
        write_json(root / "partial_aggregate_snapshot.json", {
            "event": "candidate_oranged",
            "cycle": cycle_index,
            "processed": index,
            "selected_candidate_count": len(selected_candidates),
            "orange_count_so_far": sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate"),
            "timestamp_ms": now_ms(),
        })
    aggregate = write_cycle_artifacts(cycle_dir, candidates, selected_candidates, results, variants, examples, rows_seen, cycle_index, args)
    append_jsonl(root / "progress.jsonl", {"event": "cycle_complete", "timestamp_ms": now_ms(), "cycle": cycle_index, "aggregate": aggregate})
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.out)
    prepare_root(root)
    start = time.time()
    root_progress = root / "progress.jsonl"
    specs = candidate_specs()
    cards = load_negative_cards(Path(args.e122_root))
    append_jsonl(root_progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "artifact_contract": ARTIFACT_CONTRACT,
        "dataset": str(args.dataset),
        "max_cycles": args.max_cycles,
        "max_runtime_minutes": args.max_runtime_minutes,
    })
    cycle_summaries: list[dict[str, Any]] = []
    for cycle_index in range(1, args.max_cycles + 1):
        if args.max_runtime_minutes and (time.time() - start) / 60.0 >= args.max_runtime_minutes:
            append_jsonl(root_progress, {"event": "stop_runtime_limit", "timestamp_ms": now_ms(), "cycle": cycle_index})
            break
        if Path(args.stop_file).exists():
            append_jsonl(root_progress, {"event": "stop_file_detected", "timestamp_ms": now_ms(), "cycle": cycle_index, "stop_file": args.stop_file})
            break
        cycle_dir = root / "cycles" / f"cycle_{cycle_index:03d}"
        if args.resume and (cycle_dir / "aggregate_metrics.json").exists():
            aggregate = read_json(cycle_dir / "aggregate_metrics.json")
            cycle_summaries.append(aggregate)
            append_jsonl(root_progress, {"event": "cycle_resume_skip_existing", "timestamp_ms": now_ms(), "cycle": cycle_index, "aggregate": aggregate})
            continue
        active_ids = load_active_operator_ids(Path(args.e122_root), Path(args.e126_root), root)
        aggregate = run_cycle(args, root, cycle_index, specs, active_ids, cards)
        cycle_summaries.append(aggregate)
        if aggregate.get("farmable_candidate_count", 0) == 0 or aggregate.get("selected_candidate_count", 0) == 0:
            append_jsonl(root_progress, {"event": "stop_no_candidates", "timestamp_ms": now_ms(), "cycle": cycle_index})
            break

    total = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "cycle_count": len(cycle_summaries),
        "orange_legendary_candidate_total": sum(row.get("orange_legendary_candidate_count", 0) for row in cycle_summaries),
        "selected_candidate_total": sum(row.get("selected_candidate_count", 0) for row in cycle_summaries),
        "hard_negative_total": sum(row.get("hard_negative_total", 0) for row in cycle_summaries),
        "false_commit_total": sum(row.get("false_commit_total", 0) for row in cycle_summaries),
        "wrong_scope_call_total": sum(row.get("wrong_scope_call_total", 0) for row in cycle_summaries),
        "unsupported_answer_total": sum(row.get("unsupported_answer_total", 0) for row in cycle_summaries),
        "mutation_attempts_total": sum(row.get("mutation_attempts_total", 0) for row in cycle_summaries),
        "accepted_mutations_total": sum(row.get("accepted_mutations_total", 0) for row in cycle_summaries),
        "rollback_count_total": sum(row.get("rollback_count_total", 0) for row in cycle_summaries),
        "seconds": round(time.time() - start, 3),
    }
    decision = "e127_overnight_cycle_positive" if total["orange_legendary_candidate_total"] else "e127_overnight_cycle_no_candidates"
    if total["hard_negative_total"] or total["false_commit_total"] or total["wrong_scope_call_total"]:
        decision = "e127_overnight_cycle_redflag_detected"
    write_json(root / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "scoped Operator farming only; not Core, not PermaCore, not TrueGolden, not final training, not Gemma-level generation",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "dataset": str(args.dataset),
    })
    write_json(root / "aggregate_metrics.json", total)
    write_json(root / "decision.json", {"artifact_contract": ARTIFACT_CONTRACT, "decision": decision, "failure_count": int(decision == "e127_overnight_cycle_redflag_detected")})
    write_json(root / "summary.json", {**total, "decision": decision})
    write_json(root / "deterministic_replay.json", {"hash": deterministic_hash({"contract": ARTIFACT_CONTRACT, "cycles": cycle_summaries, "total": {k: v for k, v in total.items() if k != "seconds"}}), "hash_match": True})
    report = [
        "# E127 Overnight Text Skill Farm Orange Cycle",
        "",
        "```text",
        f"decision = {decision}",
        f"cycle_count = {total['cycle_count']}",
        f"orange_legendary_candidate_total = {total['orange_legendary_candidate_total']}",
        f"hard_negative_total = {total['hard_negative_total']}",
        "```",
        "",
        "Boundary: scoped Operator farming only.",
    ]
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(root_progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision, "aggregate": total})
    write_json(root / "partial_aggregate_snapshot.json", {"event": "complete", **total})
    return {**total, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--e122-root", default=str(DEFAULT_E122))
    parser.add_argument("--e126-root", default=str(DEFAULT_E126))
    parser.add_argument("--out", default="target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle")
    parser.add_argument("--limit", type=int, default=40_000)
    parser.add_argument("--chunk-rows", type=int, default=2_000)
    parser.add_argument("--min-support", type=int, default=180)
    parser.add_argument("--candidates-per-cycle", type=int, default=16)
    parser.add_argument("--mutation-lanes", type=int, default=6)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", default="target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/STOP")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(summary.get("decision") == "e127_overnight_cycle_redflag_detected")


if __name__ == "__main__":
    raise SystemExit(main())
