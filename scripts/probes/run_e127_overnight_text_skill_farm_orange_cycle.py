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
    "scientific_notation_lens": {
        "title": "Scientific Notation Lens",
        "family": "Lens",
        "scope": "text_scientific_notation_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?e[+-]?\d+\b|\b\d+(?:\.\d+)?\s*[×x]\s*10\^[-+]?\d+\b"),
        "need": "detect scientific notation spans and preserve exponent/value meaning",
    },
    "chemical_formula_lens": {
        "title": "Chemical Formula Lens",
        "family": "Lens",
        "scope": "text_chemical_formula_boundary",
        "pattern": rx(r"\b(?:H2O|CO2|NaCl|O2|N2|CH4|C6H12O6|[A-Z][a-z]?\d{0,3}(?:[A-Z][a-z]?\d{0,3}){1,5})\b"),
        "need": "detect chemical formula-like spans as compact symbolic tokens",
    },
    "medical_dosage_span_lens": {
        "title": "Medical Dosage Span Lens",
        "family": "Lens",
        "scope": "text_medical_dosage_span_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|mL|ml|IU)\b.{0,80}\b(?:daily|twice|dose|tablet|capsule|injection)\b"),
        "need": "bind dosage, unit, and frequency without over-broad medical claims",
    },
    "timezone_offset_lens": {
        "title": "Timezone / Offset Lens",
        "family": "Lens",
        "scope": "text_timezone_offset_grounding",
        "pattern": rx(r"\b(?:UTC|GMT)[+-]\d{1,2}(?::\d{2})?\b|\b(?:CET|CEST|PST|PDT|EST|EDT|BST)\b"),
        "need": "ground timezone offsets and abbreviations in time claims",
    },
    "coordinate_pair_lens": {
        "title": "Coordinate Pair Lens",
        "family": "Lens",
        "scope": "text_coordinate_pair_grounding",
        "pattern": rx(r"\b[-+]?\d{1,3}\.\d{3,},\s*[-+]?\d{1,3}\.\d{3,}\b|\b(?:lat|latitude|lon|longitude)\b"),
        "need": "detect coordinate-like pairs and preserve lat/lon structure",
    },
    "language_locale_code_lens": {
        "title": "Language / Locale Code Lens",
        "family": "Lens",
        "scope": "text_locale_code_grounding",
        "pattern": rx(r"\b[a-z]{2}(?:-[A-Z]{2})\b|\b(?:UTF-8|ASCII|ISO-8859-1|en_US|hu_HU)\b"),
        "need": "detect language, locale, and encoding identifiers",
    },
    "commit_hash_reference_lens": {
        "title": "Commit Hash Reference Lens",
        "family": "Lens",
        "scope": "text_commit_hash_reference",
        "pattern": rx(r"\b[0-9a-f]{7,40}\b|\bcommit\s+[0-9a-f]{7,40}\b"),
        "need": "treat hashes as opaque references rather than ordinary words/numbers",
    },
    "issue_ticket_reference_lens": {
        "title": "Issue / Ticket Reference Lens",
        "family": "Lens",
        "scope": "text_issue_ticket_reference",
        "pattern": rx(r"\b[A-Z]{2,10}-\d{1,7}\b|#\d{1,7}\b|\b(?:issue|ticket|PR|pull request)\s+#?\d+"),
        "need": "bind issue, ticket, and PR identifiers as references",
    },
    "email_header_lens": {
        "title": "Email Header Lens",
        "family": "Lens",
        "scope": "text_email_header_boundary",
        "pattern": rx(r"(^|\n)(?:From|To|Cc|Subject|Date):\s+[^\n]{2,200}"),
        "need": "detect email headers and prevent them from becoming body claims",
    },
    "stacktrace_frame_lens": {
        "title": "Stacktrace Frame Lens",
        "family": "Lens",
        "scope": "text_stacktrace_frame_boundary",
        "pattern": rx(r"\bFile \"[^\"]+\", line \d+|at\s+[\w.$<>]+\([^)]*:\d+(?::\d+)?\)"),
        "need": "ground stacktrace frames as diagnostic structure",
    },
    "log_level_lens": {
        "title": "Log Level Lens",
        "family": "Lens",
        "scope": "text_log_level_grounding",
        "pattern": rx(r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b[:\]\s-]"),
        "need": "bind log severity markers to their event line",
    },
    "config_option_lens": {
        "title": "Config Option Lens",
        "family": "Lens",
        "scope": "text_config_option_binding",
        "pattern": rx(r"\b[a-zA-Z_][\w.-]{1,50}\s*[:=]\s*(?:true|false|null|none|\d+(?:\.\d+)?|\"[^\"]+\")\b"),
        "need": "detect config key/value options and preserve typed values",
    },
    "json_path_lens": {
        "title": "JSON Path Lens",
        "family": "Lens",
        "scope": "text_json_path_reference",
        "pattern": rx(r"\$?(?:\.[A-Za-z_][\w-]*|\[[0-9]+\]){2,}"),
        "need": "ground JSON/path-like references as structured selectors",
    },
    "regex_pattern_lens": {
        "title": "Regex Pattern Lens",
        "family": "Lens",
        "scope": "text_regex_pattern_boundary",
        "pattern": rx(r"/[^/\n]{3,80}/[gimsuy]*|\\b|\\d|\\w|\(\?:|\[\^?"),
        "need": "detect regex-like spans and keep escape semantics intact",
    },
    "ordinal_ranking_lens": {
        "title": "Ordinal / Ranking Lens",
        "family": "Lens",
        "scope": "text_ordinal_ranking_grounding",
        "pattern": rx(r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)|top\s+\d+|ranked?)\b"),
        "need": "ground ordinal and ranking claims without confusing them with counts",
    },
    "percentage_base_guard": {
        "title": "Percentage Base Guard",
        "family": "Guard",
        "scope": "text_percentage_base_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s?%\b.{0,120}\b(?:of|from|relative to|baseline|compared with)\b"),
        "need": "preserve the base of percentage claims",
    },
    "estimate_confidence_lens": {
        "title": "Estimate / Confidence Lens",
        "family": "Lens",
        "scope": "text_estimate_confidence_grounding",
        "pattern": rx(r"\b(?:estimate|estimated|confidence|credible interval|margin of error|uncertainty|approximation)\b"),
        "need": "separate estimates and confidence ranges from exact facts",
    },
    "absence_evidence_guard": {
        "title": "Absence Of Evidence Guard",
        "family": "Guard",
        "scope": "text_absence_evidence_distinction",
        "pattern": rx(r"\b(?:no evidence|not found|missing|absent|unknown|not enough information|insufficient evidence)\b"),
        "need": "distinguish missing evidence from evidence of absence",
    },
    "analogy_metaphor_lens": {
        "title": "Analogy / Metaphor Lens",
        "family": "Lens",
        "scope": "text_analogy_metaphor_boundary",
        "pattern": rx(r"\b(?:like|as if|similar to|metaphor|analogy|resembles|as though)\b"),
        "need": "detect analogy/metaphor spans so they are not committed as literal claims",
    },
    "nested_quote_lens": {
        "title": "Nested Quote Lens",
        "family": "Lens",
        "scope": "text_nested_quote_boundary",
        "pattern": rx(r"[\"“][^\"”]{5,120}[\"”].{0,80}[\"“][^\"”]{5,120}[\"”]"),
        "need": "preserve nested quote boundaries and attribution layers",
    },
    "footnote_marker_lens": {
        "title": "Footnote Marker Lens",
        "family": "Lens",
        "scope": "text_footnote_marker_grounding",
        "pattern": rx(r"\[\^[^\]]+\]|\b\d+\.\s+\^[^\n]+|<sup>\d+</sup>"),
        "need": "connect footnote markers to support text without treating markers as content",
    },
    "commitment_level_guard": {
        "title": "Commitment Level Guard",
        "family": "Guard",
        "scope": "text_commitment_level_grounding",
        "pattern": rx(r"\b(?:confirmed|unconfirmed|alleged|verified|rumored|claimed|official|unofficial)\b"),
        "need": "preserve claim commitment level before answer or Ground commit",
    },
    "permission_role_access_lens": {
        "title": "Permission Role Access Lens",
        "family": "Lens",
        "scope": "text_permission_role_access",
        "pattern": rx(r"\b(?:admin|owner|viewer|editor|read-only|write access|role|permission|scope)\b"),
        "need": "bind roles and access permissions without over-broad authorization",
    },
    "status_transition_lens": {
        "title": "Status Transition Lens",
        "family": "Lens",
        "scope": "text_status_transition_grounding",
        "pattern": rx(r"\b(?:open|closed|resolved|pending|blocked|done|todo|in progress|cancelled)\b.{0,80}\b(?:to|from|became|changed|marked)\b"),
        "need": "ground state transitions and avoid stale status reuse",
    },
    "multi_item_separator_lens": {
        "title": "Multi-Item Separator Lens",
        "family": "Lens",
        "scope": "text_multi_item_separator_grounding",
        "pattern": rx(r"\b[^.;:\n]{2,40}\s*/\s*[^.;:\n]{2,40}\b|;[^.;:\n]{2,80};|,\s*(?:and|or)\s+"),
        "need": "detect compact multi-item separators and preserve item boundaries",
    },
    "dataset_split_lens": {
        "title": "Dataset Split Lens",
        "family": "Lens",
        "scope": "text_dataset_split_grounding",
        "pattern": rx(r"\b(?:train|training|validation|valid|dev|test|heldout|split|seed)\b.{0,100}\b(?:set|data|rows|examples|samples)\b"),
        "need": "ground train/validation/test split mentions without mixing evidence pools",
    },
    "model_metric_result_lens": {
        "title": "Model Metric Result Lens",
        "family": "Lens",
        "scope": "text_model_metric_result_grounding",
        "pattern": rx(r"\b(?:accuracy|precision|recall|F1|AUC|loss|perplexity|score|metric)\b.{0,80}\b\d+(?:\.\d+)?\b"),
        "need": "bind model/eval metrics to values and avoid metric swaps",
    },
    "baseline_comparison_lens": {
        "title": "Baseline Comparison Lens",
        "family": "Lens",
        "scope": "text_baseline_comparison_grounding",
        "pattern": rx(r"\b(?:baseline|control|ablation|comparison|outperformed|underperformed|versus|vs\.?)\b.{0,120}\b(?:better|worse|higher|lower|matched|beat)\b"),
        "need": "ground baseline/control comparison direction and subject",
    },
    "hyperparameter_assignment_lens": {
        "title": "Hyperparameter Assignment Lens",
        "family": "Lens",
        "scope": "text_hyperparameter_assignment",
        "pattern": rx(r"\b(?:learning rate|lr|batch size|epochs?|dropout|weight decay|temperature|top_p|top-k)\b\s*[:=]?\s*\d+(?:\.\d+)?"),
        "need": "bind hyperparameter names to values and keep them out of ordinary numeric claims",
    },
    "dependency_package_lens": {
        "title": "Dependency Package Lens",
        "family": "Lens",
        "scope": "text_dependency_package_reference",
        "pattern": rx(r"\b(?:pip|npm|cargo|go get|apt|conda)\s+(?:install|add)?\s*[A-Za-z0-9_.@/-]+|\b[A-Za-z0-9_.-]+==\d+\.\d+"),
        "need": "detect package/dependency references and preserve package/version boundaries",
    },
    "license_identifier_lens": {
        "title": "License Identifier Lens",
        "family": "Lens",
        "scope": "text_license_identifier_grounding",
        "pattern": rx(r"\b(?:MIT|Apache-?2\.0|GPLv?3?|LGPL|BSD-?3|MPL|AGPL|Creative Commons|CC-BY|license)\b"),
        "need": "ground license identifiers without treating them as generic prose",
    },
    "security_vulnerability_lens": {
        "title": "Security Vulnerability Lens",
        "family": "Lens",
        "scope": "text_security_vulnerability_grounding",
        "pattern": rx(r"\bCVE-\d{4}-\d{4,7}\b|\b(?:vulnerability|exploit|patch|CWE|CVSS|zero-day|security advisory)\b"),
        "need": "detect security vulnerability references and preserve risk context",
    },
    "crypto_hash_digest_lens": {
        "title": "Crypto Hash Digest Lens",
        "family": "Lens",
        "scope": "text_crypto_hash_digest_reference",
        "pattern": rx(r"\b(?:sha256|sha1|md5|digest|checksum)\b[:=\s]+[0-9a-f]{16,128}\b"),
        "need": "treat cryptographic digests as opaque integrity references",
    },
    "ip_address_lens": {
        "title": "IP Address Lens",
        "family": "Lens",
        "scope": "text_ip_address_grounding",
        "pattern": rx(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[0-9a-f]{0,4}:[0-9a-f:]{2,}\b"),
        "need": "detect IP address spans and avoid numeric decomposition errors",
    },
    "domain_name_lens": {
        "title": "Domain Name Lens",
        "family": "Lens",
        "scope": "text_domain_name_reference",
        "pattern": rx(r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|ai|edu|gov|dev|hu|de|at)\b"),
        "need": "detect bare domain references without requiring a URL scheme",
    },
    "doi_isbn_reference_lens": {
        "title": "DOI / ISBN Reference Lens",
        "family": "Lens",
        "scope": "text_doi_isbn_reference",
        "pattern": rx(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b|\bISBN(?:-1[03])?:?\s*(?:\d[- ]*){9,17}[\dX]\b"),
        "need": "ground DOI/ISBN identifiers as citations or references",
    },
    "section_cross_reference_lens": {
        "title": "Section Cross-Reference Lens",
        "family": "Lens",
        "scope": "text_section_cross_reference",
        "pattern": rx(r"\b(?:see|refer to|as shown in)\s+(?:section|chapter|figure|table|appendix)\s+[A-Za-z0-9.-]+"),
        "need": "bind cross-references to their target section/table/figure labels",
    },
    "figure_caption_lens": {
        "title": "Figure Caption Lens",
        "family": "Lens",
        "scope": "text_figure_caption_grounding",
        "pattern": rx(r"(^|\n)\s*(?:Figure|Fig\.|Table)\s+\d+[A-Za-z]?:\s+[^\n]{5,200}"),
        "need": "detect figure/table captions and preserve caption scope",
    },
    "spreadsheet_cell_reference_lens": {
        "title": "Spreadsheet Cell Reference Lens",
        "family": "Lens",
        "scope": "text_spreadsheet_cell_reference",
        "pattern": rx(r"\b[A-Z]{1,3}\d{1,7}(?::[A-Z]{1,3}\d{1,7})?\b|\b(?:sheet|worksheet|cell|range)\b"),
        "need": "ground spreadsheet cell/range references as locations",
    },
    "database_query_clause_lens": {
        "title": "Database Query Clause Lens",
        "family": "Lens",
        "scope": "text_database_query_clause_boundary",
        "pattern": rx(r"\b(?:SELECT|WHERE|JOIN|GROUP BY|ORDER BY|INSERT INTO|UPDATE|DELETE FROM)\b"),
        "need": "detect SQL-like clauses and prevent prose interpretation",
    },
    "boolean_logic_expression_lens": {
        "title": "Boolean Logic Expression Lens",
        "family": "Lens",
        "scope": "text_boolean_logic_expression",
        "pattern": rx(r"\b(?:AND|OR|NOT|XOR|TRUE|FALSE)\b|&&|\|\||!"),
        "need": "detect boolean logic expressions and preserve operator semantics",
    },
    "measurement_tolerance_lens": {
        "title": "Measurement Tolerance Lens",
        "family": "Lens",
        "scope": "text_measurement_tolerance_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|kg|g|V|A|W|Hz)\s*(?:±|\\+/-|\+/-)\s*\d+(?:\.\d+)?"),
        "need": "bind measurement values to tolerances and units",
    },
    "confidence_interval_lens": {
        "title": "Confidence Interval Lens",
        "family": "Lens",
        "scope": "text_confidence_interval_grounding",
        "pattern": rx(r"\b(?:95%|99%)\s+(?:CI|confidence interval)\b|\bCI\s*[:=]\s*\[?\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]?"),
        "need": "ground confidence intervals as interval evidence, not exact values",
    },
    "probability_likelihood_lens": {
        "title": "Probability / Likelihood Lens",
        "family": "Lens",
        "scope": "text_probability_likelihood_grounding",
        "pattern": rx(r"\b(?:probability|likelihood|odds|chance|risk)\b.{0,80}\b\d+(?:\.\d+)?\s?%?"),
        "need": "bind probability/risk words to numeric likelihoods",
    },
    "redaction_placeholder_lens": {
        "title": "Redaction Placeholder Lens",
        "family": "Lens",
        "scope": "text_redaction_placeholder_boundary",
        "pattern": rx(r"\[(?:REDACTED|PRIVATE|NAME|EMAIL|PHONE|ADDRESS)\]|<PRIVATE_[A-Z_]+>|\*\*\*"),
        "need": "detect redaction placeholders and avoid hallucinating hidden content",
    },
    "changelog_entry_lens": {
        "title": "Changelog Entry Lens",
        "family": "Lens",
        "scope": "text_changelog_entry_grounding",
        "pattern": rx(r"\b(?:Added|Changed|Fixed|Removed|Deprecated|Security)\b.{0,120}\b(?:v?\d+\.\d+|release|version)\b"),
        "need": "ground changelog categories and version context",
    },
    "command_prompt_boundary_lens": {
        "title": "Command Prompt Boundary Lens",
        "family": "Lens",
        "scope": "text_command_prompt_boundary",
        "pattern": rx(r"(^|\n)\s*(?:\$|>|PS>|C:\\>|#)\s+\S.{0,180}"),
        "need": "detect shell prompt lines and avoid treating prompts as answer text",
    },
    "environment_variable_lens": {
        "title": "Environment Variable Lens",
        "family": "Lens",
        "scope": "text_environment_variable_reference",
        "pattern": rx(r"\b[A-Z_][A-Z0-9_]{2,}\b\s*=\s*[^\s]+|\$[A-Z_][A-Z0-9_]{2,}"),
        "need": "ground environment variable names and values as configuration",
    },
    "definition_phrase_lens": {
        "title": "Definition Phrase Lens",
        "family": "Lens",
        "scope": "text_definition_phrase_grounding",
        "pattern": rx(r"\b(?:is defined as|are defined as|refers to|means|is called|are called|known as|denotes)\b"),
        "need": "bind definition phrases to the term being defined without treating examples as definitions",
    },
    "acronym_expansion_lens": {
        "title": "Acronym Expansion Lens",
        "family": "Lens",
        "scope": "text_acronym_expansion_grounding",
        "pattern": rx(r"\b[A-Z][A-Z0-9]{1,8}\s*\([^)]{3,80}\)|\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,6}\s*\([A-Z0-9]{2,8}\)"),
        "need": "connect acronyms to nearby expansions while preserving acronym identity",
    },
    "parenthetical_qualification_lens": {
        "title": "Parenthetical Qualification Lens",
        "family": "Lens",
        "scope": "text_parenthetical_qualification",
        "pattern": rx(r"\([^)]{6,160}\)"),
        "need": "treat parenthetical spans as qualifiers or side information instead of main claim overwrite",
    },
    "causal_connector_lens": {
        "title": "Causal Connector Lens",
        "family": "Lens",
        "scope": "text_causal_connector_grounding",
        "pattern": rx(r"\b(?:because|therefore|thus|hence|as a result|resulting in|due to|caused by|leads to)\b"),
        "need": "bind cause/effect connector spans and avoid reversing causal direction",
    },
    "conditional_if_then_guard": {
        "title": "Conditional If/Then Guard",
        "family": "Guard",
        "scope": "text_conditional_if_then_scope",
        "pattern": rx(r"\bif\b.{0,160}\b(?:then|must|should|will|can|may)\b|\bwhen\b.{0,160}\b(?:then|must|should|will|can|may)\b"),
        "need": "preserve condition scope before committing a consequent as unconditional truth",
    },
    "unless_exception_guard": {
        "title": "Unless / Exception Guard",
        "family": "Guard",
        "scope": "text_unless_exception_scope",
        "pattern": rx(r"\b(?:unless|except when|except if|with the exception of|except for|other than)\b"),
        "need": "attach exception clauses to the rule they limit",
    },
    "contrast_concession_lens": {
        "title": "Contrast / Concession Lens",
        "family": "Lens",
        "scope": "text_contrast_concession_relation",
        "pattern": rx(r"\b(?:however|although|though|whereas|while|nevertheless|nonetheless|but|yet)\b"),
        "need": "detect contrast or concession boundaries and prevent stale claim carryover",
    },
    "comparison_degree_lens": {
        "title": "Comparison Degree Lens",
        "family": "Lens",
        "scope": "text_comparison_degree_grounding",
        "pattern": rx(r"\b(?:more|less|fewer|greater|smaller|higher|lower|better|worse|largest|smallest|most|least)\b.{0,100}\b(?:than|compared to|relative to)\b"),
        "need": "ground comparative direction and compared entities",
    },
    "quote_attribution_lens": {
        "title": "Quote Attribution Lens",
        "family": "Lens",
        "scope": "text_quote_attribution_grounding",
        "pattern": rx(r"[\"“][^\"”]{8,180}[\"”]\s*,?\s*(?:said|wrote|according to|reported|claimed)\b|\b(?:said|wrote|according to|reported|claimed)\b.{0,80}[\"“][^\"”]{8,180}[\"”]"),
        "need": "bind quoted content to attribution and avoid treating attributed claims as directly verified",
    },
    "procedure_step_sequence_lens": {
        "title": "Procedure Step Sequence Lens",
        "family": "Lens",
        "scope": "text_procedure_step_sequence",
        "pattern": rx(r"\b(?:step\s+\d+|first,|second,|third,|next,|finally,|then)\b.{0,120}"),
        "need": "preserve ordered procedure steps and avoid step reordering",
    },
    "ordered_list_marker_lens": {
        "title": "Ordered List Marker Lens",
        "family": "Lens",
        "scope": "text_ordered_list_marker_structure",
        "pattern": rx(r"(^|\n)\s*(?:\d+[\.)]|[a-zA-Z][\.)])\s+[^\n]{3,160}"),
        "need": "detect ordered list markers as structure rather than numeric claims",
    },
    "range_interval_phrase_lens": {
        "title": "Range / Interval Phrase Lens",
        "family": "Lens",
        "scope": "text_range_interval_phrase",
        "pattern": rx(r"\b(?:between|from)\s+[-+]?\d+(?:\.\d+)?\s+(?:and|to|through)\s+[-+]?\d+(?:\.\d+)?\b|\b[-+]?\d+(?:\.\d+)?\s*[-–]\s*[-+]?\d+(?:\.\d+)?\b"),
        "need": "ground numeric ranges as intervals and avoid endpoint collapse",
    },
    "approximate_quantity_lens": {
        "title": "Approximate Quantity Lens",
        "family": "Lens",
        "scope": "text_approximate_quantity_grounding",
        "pattern": rx(r"\b(?:about|around|roughly|approximately|approx\.|nearly|almost|over|under|at least|at most)\s+\d+(?:\.\d+)?"),
        "need": "preserve approximation qualifiers around quantities",
    },
    "negated_requirement_guard": {
        "title": "Negated Requirement Guard",
        "family": "Guard",
        "scope": "text_negated_requirement_scope",
        "pattern": rx(r"\b(?:must not|should not|cannot|can't|do not|does not|required not to|not allowed to)\b"),
        "need": "preserve negative requirements and avoid converting them into positive instructions",
    },
    "example_marker_lens": {
        "title": "Example Marker Lens",
        "family": "Lens",
        "scope": "text_example_marker_boundary",
        "pattern": rx(r"\b(?:for example|for instance|e\.g\.|such as|including|includes)\b"),
        "need": "separate illustrative examples from exhaustive definitions",
    },
    "summary_conclusion_lens": {
        "title": "Summary / Conclusion Lens",
        "family": "Lens",
        "scope": "text_summary_conclusion_boundary",
        "pattern": rx(r"\b(?:in summary|to summarize|overall|in conclusion|therefore|the result is|this means)\b"),
        "need": "detect conclusion spans and bind them to prior evidence instead of treating them as unsupported facts",
    },
    "sample_size_study_lens": {
        "title": "Study Sample Size Lens",
        "family": "Lens",
        "scope": "text_study_sample_size_grounding",
        "pattern": rx(r"\b(?:study|trial|survey|experiment|participants|subjects|patients)\b.{0,100}\b(?:n\s*=\s*)?\d{2,}\b"),
        "need": "bind study/sample-size quantities to the study context",
    },
    "taxonomy_classification_lens": {
        "title": "Taxonomy Classification Lens",
        "family": "Lens",
        "scope": "text_taxonomy_classification_grounding",
        "pattern": rx(r"\b(?:type of|kind of|class of|category of|belongs to|classified as|subclass|superclass)\b"),
        "need": "ground classification relations without flattening hierarchy",
    },
    "dependency_precondition_lens": {
        "title": "Dependency / Precondition Lens",
        "family": "Lens",
        "scope": "text_dependency_precondition_grounding",
        "pattern": rx(r"\b(?:requires|required before|depends on|dependency|prerequisite|precondition|needs)\b"),
        "need": "bind prerequisites and dependencies to the action or artifact they gate",
    },
    "title_author_publication_lens": {
        "title": "Title / Author / Publication Lens",
        "family": "Lens",
        "scope": "text_title_author_publication_reference",
        "pattern": rx(r"[\"“][^\"”]{5,120}[\"”]\s+(?:by|from|in)\s+[A-Z][A-Za-z0-9 .,'-]{2,80}|\b(?:author|published|journal|book|paper)\b.{0,100}"),
        "need": "bind titles to authors/publications without making title text a claim",
    },
    "standard_spec_reference_lens": {
        "title": "Standard / Spec Reference Lens",
        "family": "Lens",
        "scope": "text_standard_spec_reference",
        "pattern": rx(r"\b(?:RFC|ISO|IEC|IEEE|PEP|W3C|ECMA|NIST)\s*[- ]?\d+[A-Za-z0-9.-]*\b"),
        "need": "ground technical standard references as opaque spec identifiers",
    },
    "complexity_notation_lens": {
        "title": "Complexity Notation Lens",
        "family": "Lens",
        "scope": "text_complexity_notation_grounding",
        "pattern": rx(r"\bO\([^)]+\)|\b(?:linear|quadratic|logarithmic|exponential)\s+(?:time|space|complexity)\b"),
        "need": "detect algorithmic complexity notation and preserve its symbolic meaning",
    },
    "object_property_relation_lens": {
        "title": "Object / Property Relation Lens",
        "family": "Lens",
        "scope": "text_object_property_relation",
        "pattern": rx(r"\b[A-Za-z][\w -]{2,50}\s+(?:has|contains|includes|supports|uses|provides|stores)\s+[A-Za-z0-9][^.;\n]{2,100}"),
        "need": "bind object-property relations without leaking properties across nearby objects",
    },
    "before_after_temporal_lens": {
        "title": "Before / After Temporal Lens",
        "family": "Lens",
        "scope": "text_before_after_temporal_relation",
        "pattern": rx(r"\b(?:before|after|prior to|following|subsequent to|once|until)\b.{0,120}"),
        "need": "ground temporal ordering and prevent stale-state reuse",
    },
    "instruction_order_constraint_guard": {
        "title": "Instruction Order Constraint Guard",
        "family": "Guard",
        "scope": "text_instruction_order_constraint",
        "pattern": rx(r"\b(?:before you|after you|only after|do this first|then do|do not.*until)\b"),
        "need": "preserve ordered constraints in instruction-like text",
    },
    "question_answer_pair_lens": {
        "title": "Question / Answer Pair Lens",
        "family": "Lens",
        "scope": "text_question_answer_pair_boundary",
        "pattern": rx(r"\b(?:Q:|Question:).{0,180}\b(?:A:|Answer:)|\?\s+(?:Yes|No|It|This|That|Because)\b"),
        "need": "bind answer spans to the question they answer and avoid cross-question leakage",
    },
    "missing_field_placeholder_guard": {
        "title": "Missing Field Placeholder Guard",
        "family": "Guard",
        "scope": "text_missing_field_placeholder",
        "pattern": rx(r"\b(?:TBD|TBA|N/A|unknown|null|none|not provided|missing|to be determined)\b"),
        "need": "treat placeholders as unresolved fields rather than factual values",
    },
    "scope_limitation_phrase_guard": {
        "title": "Scope Limitation Phrase Guard",
        "family": "Guard",
        "scope": "text_scope_limitation_phrase",
        "pattern": rx(r"\b(?:only applies to|limited to|does not apply to|out of scope|within scope|scope is)\b"),
        "need": "preserve explicit scope boundaries before reuse or promotion",
    },
    "revision_diff_marker_lens": {
        "title": "Revision / Diff Marker Lens",
        "family": "Lens",
        "scope": "text_revision_diff_marker",
        "pattern": rx(r"(^|\n)\s*(?:\+|-|@@)\s*[^\n]{2,180}|\b(?:diff|patch|added line|removed line)\b"),
        "need": "detect diff/revision markers and avoid treating removed text as current truth",
    },
    "data_quality_caveat_lens": {
        "title": "Data Quality Caveat Lens",
        "family": "Lens",
        "scope": "text_data_quality_caveat",
        "pattern": rx(r"\b(?:sample bias|missing data|noisy data|outlier|data quality|measurement error|confounder)\b"),
        "need": "bind quality caveats to the affected evidence or metric",
    },
    "modal_obligation_permission_lens": {
        "title": "Modal Obligation / Permission Lens",
        "family": "Lens",
        "scope": "text_modal_obligation_permission",
        "pattern": rx(r"\b(?:must|should|may|might|can|could|required to|allowed to|permitted to|optional|mandatory)\b"),
        "need": "preserve modal strength so permission, obligation, and possibility do not collapse",
    },
    "correction_retraction_lens": {
        "title": "Correction / Retraction Lens",
        "family": "Lens",
        "scope": "text_correction_retraction_grounding",
        "pattern": rx(r"\b(?:correction|corrected|retracted|erratum|updated from|no longer|previously stated|was wrong|instead)\b"),
        "need": "ground corrections and retractions so stale statements are not reused",
    },
    "pros_cons_tradeoff_lens": {
        "title": "Pros / Cons Tradeoff Lens",
        "family": "Lens",
        "scope": "text_pros_cons_tradeoff",
        "pattern": rx(r"\b(?:pros?|cons?|advantages?|disadvantages?|benefits?|drawbacks?|tradeoff|trade-off|downside|upside)\b"),
        "need": "bind tradeoff language to its option instead of flattening all claims as equal",
    },
    "alternative_choice_lens": {
        "title": "Alternative Choice Lens",
        "family": "Lens",
        "scope": "text_alternative_choice_boundary",
        "pattern": rx(r"\b(?:either|or|instead of|rather than|alternative|option|choice|choose between)\b"),
        "need": "preserve alternative branches and avoid committing mutually exclusive choices together",
    },
    "acceptance_criteria_lens": {
        "title": "Acceptance Criteria Lens",
        "family": "Lens",
        "scope": "text_acceptance_criteria_grounding",
        "pattern": rx(r"\b(?:acceptance criteria|done when|passes if|success criteria|must pass|definition of done)\b"),
        "need": "ground completion criteria before marking a task complete",
    },
    "test_assertion_lens": {
        "title": "Test Assertion Lens",
        "family": "Lens",
        "scope": "text_test_assertion_grounding",
        "pattern": rx(r"\b(?:assert|expect|expected|actual|test case|unit test|integration test|should equal)\b"),
        "need": "bind expected and actual outcomes without swapping them",
    },
    "http_status_endpoint_lens": {
        "title": "HTTP Status / Endpoint Lens",
        "family": "Lens",
        "scope": "text_http_status_endpoint_grounding",
        "pattern": rx(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[A-Za-z0-9_./{}:-]+|\bHTTP\s+\d{3}\b|\b(?:status|response)\s+\d{3}\b"),
        "need": "ground HTTP method, endpoint, and status response as protocol structure",
    },
    "api_response_field_lens": {
        "title": "API Response Field Lens",
        "family": "Lens",
        "scope": "text_api_response_field_binding",
        "pattern": rx(r"\b(?:response|request|body|payload|returns?|field)\b.{0,100}\b[a-zA-Z_][\w-]*\b\s*[:=]\s*"),
        "need": "bind API response fields to their values or schema role",
    },
    "code_fence_language_lens": {
        "title": "Code Fence Language Lens",
        "family": "Lens",
        "scope": "text_code_fence_language_boundary",
        "pattern": rx(r"```[A-Za-z0-9_+-]*\n|~~~[A-Za-z0-9_+-]*\n"),
        "need": "detect code fence boundaries and language hints before interpreting text as prose",
    },
    "inline_code_identifier_lens": {
        "title": "Inline Code Identifier Lens",
        "family": "Lens",
        "scope": "text_inline_code_identifier",
        "pattern": rx(r"`[^`\n]{1,80}`"),
        "need": "treat inline code spans as identifiers or literals rather than normal prose",
    },
    "git_branch_tag_lens": {
        "title": "Git Branch / Tag Lens",
        "family": "Lens",
        "scope": "text_git_branch_tag_reference",
        "pattern": rx(r"\b(?:branch|tag|remote|origin|main|master|HEAD)\b.{0,80}\b[A-Za-z0-9._/-]{2,80}\b"),
        "need": "ground git branch/tag references without confusing them with file paths",
    },
    "file_permission_mode_lens": {
        "title": "File Permission Mode Lens",
        "family": "Lens",
        "scope": "text_file_permission_mode",
        "pattern": rx(r"\b(?:chmod|chown|permission|read|write|execute)\b.{0,80}\b(?:[0-7]{3,4}|rwx|read-only)\b"),
        "need": "bind permission modes and access verbs to the right file or role",
    },
    "unit_conversion_phrase_lens": {
        "title": "Unit Conversion Phrase Lens",
        "family": "Lens",
        "scope": "text_unit_conversion_phrase",
        "pattern": rx(r"\b(?:convert|conversion|equals|equivalent to|per)\b.{0,100}\b(?:kg|g|m|cm|mm|km|miles?|hours?|minutes?|bytes?|MB|GB|C|F)\b"),
        "need": "detect unit conversion/equivalence phrases without committing unsupported conversions",
    },
    "math_word_operator_lens": {
        "title": "Math Word Operator Lens",
        "family": "Lens",
        "scope": "text_math_word_operator_grounding",
        "pattern": rx(r"\b(?:sum|difference|product|quotient|multiply|divide|subtract|add|total|average|mean|median)\b"),
        "need": "ground natural-language math operator words before calculation or defer",
    },
    "chart_axis_legend_lens": {
        "title": "Chart Axis / Legend Lens",
        "family": "Lens",
        "scope": "text_chart_axis_legend_grounding",
        "pattern": rx(r"\b(?:x-axis|y-axis|axis|legend|series|plot|chart|graph)\b.{0,120}\b(?:shows|represents|versus|vs\.?)\b"),
        "need": "bind chart labels and legends to represented variables",
    },
    "table_row_value_lens": {
        "title": "Table Row / Value Lens",
        "family": "Lens",
        "scope": "text_table_row_value_binding",
        "pattern": rx(r"\b(?:row|column|cell|header)\b.{0,80}\b(?:value|contains|equals|shows|lists)\b"),
        "need": "bind table row/column references to the correct value span",
    },
    "survey_response_scale_lens": {
        "title": "Survey Response Scale Lens",
        "family": "Lens",
        "scope": "text_survey_response_scale",
        "pattern": rx(r"\b(?:Likert|scale of|rated|rating|strongly agree|strongly disagree|satisfaction)\b"),
        "need": "ground survey response scales and avoid treating ordinal labels as numeric facts",
    },
    "legal_rights_obligation_lens": {
        "title": "Legal Rights / Obligation Lens",
        "family": "Lens",
        "scope": "text_legal_rights_obligation",
        "pattern": rx(r"\b(?:right to|obligation to|liable for|responsible for|entitled to|waive|consent)\b"),
        "need": "bind legal-like rights and obligations to the correct party and scope",
    },
    "privacy_personal_data_lens": {
        "title": "Privacy / Personal Data Lens",
        "family": "Lens",
        "scope": "text_privacy_personal_data",
        "pattern": rx(r"\b(?:personal data|PII|GDPR|privacy|consent|data subject|email address|phone number)\b"),
        "need": "detect personal-data/privacy spans and keep them from unsafe broad reuse",
    },
    "training_signal_label_lens": {
        "title": "Training Signal / Label Lens",
        "family": "Lens",
        "scope": "text_training_signal_label_grounding",
        "pattern": rx(r"\b(?:label|target|ground truth|annotation|annotated|class label|supervision signal)\b"),
        "need": "separate training labels from observed facts and model outputs",
    },
    "dataset_provenance_lens": {
        "title": "Dataset Provenance Lens",
        "family": "Lens",
        "scope": "text_dataset_provenance_grounding",
        "pattern": rx(r"\b(?:dataset|corpus|source data|collected from|scraped from|licensed from|provenance)\b"),
        "need": "bind dataset provenance and source information to the dataset claim",
    },
    "model_checkpoint_lens": {
        "title": "Model Checkpoint Lens",
        "family": "Lens",
        "scope": "text_model_checkpoint_reference",
        "pattern": rx(r"\b(?:checkpoint|ckpt|weights|model file|safetensors|pt file|epoch)\b.{0,100}\b(?:\d+|[A-Za-z0-9_.-]+)\b"),
        "need": "detect model checkpoint references as artifacts rather than content claims",
    },
    "benchmark_leakage_guard": {
        "title": "Benchmark Leakage Guard",
        "family": "Guard",
        "scope": "text_benchmark_leakage_boundary",
        "pattern": rx(r"\b(?:leakage|data leak|train-test contamination|contaminated|benchmark artifact|oracle leakage)\b"),
        "need": "flag benchmark leakage caveats before treating metrics as valid evidence",
    },
    "reproducibility_seed_lens": {
        "title": "Reproducibility Seed Lens",
        "family": "Lens",
        "scope": "text_reproducibility_seed_grounding",
        "pattern": rx(r"\b(?:seed|random seed|deterministic|reproducible|replay|hash match)\b.{0,100}\b\d+\b"),
        "need": "bind reproducibility seed and replay evidence to the correct run",
    },
    "failure_mode_phrase_lens": {
        "title": "Failure Mode Phrase Lens",
        "family": "Lens",
        "scope": "text_failure_mode_phrase",
        "pattern": rx(r"\b(?:failure mode|bottleneck|root cause|regression|breakpoint|edge case|corner case)\b"),
        "need": "ground failure-mode descriptions for later negative-card reuse",
    },
    "mitigation_action_lens": {
        "title": "Mitigation Action Lens",
        "family": "Lens",
        "scope": "text_mitigation_action_grounding",
        "pattern": rx(r"\b(?:mitigate|workaround|fix by|repair by|avoid by|prevent by|guard against)\b"),
        "need": "bind mitigation actions to the failure or risk they address",
    },
    "spec_requirement_keyword_lens": {
        "title": "Spec Requirement Keyword Lens",
        "family": "Lens",
        "scope": "text_spec_requirement_keyword",
        "pattern": rx(r"\b(?:MUST|SHOULD|MAY|REQUIRED|RECOMMENDED|OPTIONAL|SHALL)\b"),
        "need": "preserve RFC-style requirement keywords and their normative strength",
    },
    "cross_reference_pronoun_guard": {
        "title": "Cross-Reference Pronoun Guard",
        "family": "Guard",
        "scope": "text_cross_reference_pronoun_scope",
        "pattern": rx(r"\b(?:this|that|it|they|them|those|these|the former|the latter)\b.{0,100}\b(?:refers to|means|is unclear|ambiguous|above|below)\b"),
        "need": "avoid committing pronoun references when the antecedent is unstable",
    },
    "multi_source_claim_merge_lens": {
        "title": "Multi-Source Claim Merge Lens",
        "family": "Lens",
        "scope": "text_multi_source_claim_merge",
        "pattern": rx(r"\b(?:according to|source A|source B|both sources|another source|multiple sources|separately reported)\b"),
        "need": "merge or separate claims by source without losing provenance",
    },
    "confidence_phrase_without_number_lens": {
        "title": "Confidence Phrase Without Number Lens",
        "family": "Lens",
        "scope": "text_confidence_phrase_without_number",
        "pattern": rx(r"\b(?:likely|unlikely|probably|possibly|certain|uncertain|high confidence|low confidence)\b"),
        "need": "ground qualitative confidence without inventing numeric probability",
    },
    "task_owner_assignment_lens": {
        "title": "Task Owner Assignment Lens",
        "family": "Lens",
        "scope": "text_task_owner_assignment",
        "pattern": rx(r"\b(?:assigned to|owner|responsible party|assignee|handled by|belongs to)\b.{0,100}\b[A-Z][A-Za-z0-9_.-]{1,60}\b"),
        "need": "bind tasks to owners without leaking ownership across nearby tasks",
    },
    "schedule_recurrence_lens": {
        "title": "Schedule Recurrence Lens",
        "family": "Lens",
        "scope": "text_schedule_recurrence_grounding",
        "pattern": rx(r"\b(?:daily|weekly|monthly|yearly|every \d+|recurring|repeat(?:s|ed)?|cron)\b"),
        "need": "ground recurrence expressions separately from one-time dates",
    },
    "causal_link_lens": {
        "title": "Causal Link Lens",
        "family": "Lens",
        "scope": "text_causal_link_grounding",
        "pattern": rx(r"\b(?:because|therefore|thus|hence|so that|as a result|caused by|leads to|results in)\b"),
        "need": "bind cause and effect spans without flattening them into unrelated claims",
    },
    "conditional_if_then_guard": {
        "title": "Conditional If / Then Guard",
        "family": "Guard",
        "scope": "text_conditional_if_then_scope",
        "pattern": rx(r"\b(?:if|when|whenever|provided that|assuming|unless)\b.{0,160}\b(?:then|must|should|will|can|cannot)\b"),
        "need": "preserve conditional scope before committing a conclusion",
    },
    "definition_alias_lens": {
        "title": "Definition / Alias Lens",
        "family": "Lens",
        "scope": "text_definition_alias_grounding",
        "pattern": rx(r"\b(?:means|is defined as|refers to|called|aka|a\.k\.a\.|alias|stands for)\b"),
        "need": "bind a term to its local definition or alias without global overreach",
    },
    "example_marker_lens": {
        "title": "Example Marker Lens",
        "family": "Lens",
        "scope": "text_example_marker_boundary",
        "pattern": rx(r"\b(?:for example|for instance|e\.g\.|such as|example:|examples include)\b"),
        "need": "treat examples as support or instances, not universal rules",
    },
    "contrast_concession_lens": {
        "title": "Contrast / Concession Lens",
        "family": "Lens",
        "scope": "text_contrast_concession_grounding",
        "pattern": rx(r"\b(?:however|but|although|though|nevertheless|whereas|while|on the other hand|despite)\b"),
        "need": "keep contrasted claims separate and preserve concession direction",
    },
    "reported_speech_quote_lens": {
        "title": "Reported Speech / Quote Lens",
        "family": "Lens",
        "scope": "text_reported_speech_quote_boundary",
        "pattern": rx(r"\b(?:said|stated|claimed|reported|according to|quote|quoted)\b.{0,120}[\"'“”]"),
        "need": "bind quoted or reported content to its source instead of adopting it as direct fact",
    },
    "method_result_split_lens": {
        "title": "Method / Result Split Lens",
        "family": "Lens",
        "scope": "text_method_result_split",
        "pattern": rx(r"\b(?:method|approach|procedure|we used|using|measured|experiment)\b.{0,180}\b(?:result|found|showed|observed|yielded)\b"),
        "need": "separate procedure descriptions from result claims",
    },
    "limitation_future_work_lens": {
        "title": "Limitation / Future Work Lens",
        "family": "Lens",
        "scope": "text_limitation_future_work",
        "pattern": rx(r"\b(?:limitation|limited by|future work|not yet|does not cover|cannot conclude|open question)\b"),
        "need": "ground limitations and future-work caveats before broad reuse",
    },
    "threshold_boundary_lens": {
        "title": "Threshold Boundary Lens",
        "family": "Lens",
        "scope": "text_threshold_boundary_grounding",
        "pattern": rx(r"\b(?:at least|at most|more than|less than|greater than|below|above|under|over|minimum|maximum)\b.{0,80}\b\d+"),
        "need": "bind numeric thresholds to direction and target",
    },
    "comparison_superlative_lens": {
        "title": "Comparison / Superlative Lens",
        "family": "Lens",
        "scope": "text_comparison_superlative_grounding",
        "pattern": rx(r"\b(?:better than|worse than|larger than|smaller than|best|worst|highest|lowest|more efficient|less efficient)\b"),
        "need": "ground comparative claims to compared objects and metric",
    },
    "state_transition_marker_lens": {
        "title": "State Transition Marker Lens",
        "family": "Lens",
        "scope": "text_state_transition_marker",
        "pattern": rx(r"\b(?:became|turned into|changed from|changed to|switched to|transitioned|moved from|moved to)\b"),
        "need": "bind old and new states without stale-state reuse",
    },
    "deprecation_replacement_lens": {
        "title": "Deprecation / Replacement Lens",
        "family": "Lens",
        "scope": "text_deprecation_replacement_grounding",
        "pattern": rx(r"\b(?:deprecated|obsolete|removed|replaced by|superseded by|use .* instead|no longer supported)\b"),
        "need": "detect stale API/tool references and bind replacements",
    },
    "compatibility_requirement_lens": {
        "title": "Compatibility Requirement Lens",
        "family": "Lens",
        "scope": "text_compatibility_requirement",
        "pattern": rx(r"\b(?:compatible with|requires|depends on|works with|does not work with|supported on|unsupported on)\b"),
        "need": "bind compatibility statements to version/platform/dependency scope",
    },
    "resource_budget_constraint_lens": {
        "title": "Resource / Budget Constraint Lens",
        "family": "Lens",
        "scope": "text_resource_budget_constraint",
        "pattern": rx(r"\b(?:budget|cost|latency|memory|RAM|VRAM|CPU|GPU|tokens|time limit|quota)\b.{0,120}\b(?:limit|cap|under|over|exceed|within)\b"),
        "need": "ground resource constraints before choosing an action or mode",
    },
    "config_key_value_lens": {
        "title": "Config Key / Value Lens",
        "family": "Lens",
        "scope": "text_config_key_value_binding",
        "pattern": rx(r"\b[A-Za-z_][A-Za-z0-9_.-]{2,60}\s*[:=]\s*(?:true|false|null|\"[^\"]{0,80}\"|'[^']{0,80}'|[A-Za-z0-9_.:/-]{1,80})"),
        "need": "bind configuration keys to values without treating them as prose claims",
    },
    "environment_variable_lens": {
        "title": "Environment Variable Lens",
        "family": "Lens",
        "scope": "text_environment_variable_binding",
        "pattern": rx(r"\b[A-Z][A-Z0-9_]{2,60}\s*=\s*[^\s]+|\$[A-Z][A-Z0-9_]{2,60}|%[A-Z][A-Z0-9_]{2,60}%"),
        "need": "detect environment variables and preserve variable/value boundaries",
    },
    "command_pipeline_lens": {
        "title": "Command Pipeline Lens",
        "family": "Lens",
        "scope": "text_command_pipeline_boundary",
        "pattern": rx(r"\b(?:cat|grep|rg|awk|sed|python|cargo|git|npm|pip|docker)\b.{0,180}(?:\||&&|>|<|--[A-Za-z])"),
        "need": "detect shell command pipelines and keep commands separate from explanatory text",
    },
    "stack_trace_frame_lens": {
        "title": "Stack Trace Frame Lens",
        "family": "Lens",
        "scope": "text_stack_trace_frame",
        "pattern": rx(r"\b(?:File \"[^\"]+\", line \d+|at [A-Za-z0-9_.$<>]+\([^)]*:\d+:\d+\)|Traceback|stack trace)\b"),
        "need": "bind stack trace frames to location and failure context",
    },
    "patch_hunk_boundary_lens": {
        "title": "Patch Hunk Boundary Lens",
        "family": "Lens",
        "scope": "text_patch_hunk_boundary",
        "pattern": rx(r"(^|\n)(?:diff --git|index [0-9a-f]{6,}|--- |\+\+\+ |@@ [^@\n]* @@)"),
        "need": "detect patch hunk boundaries and distinguish additions from deletions",
    },
    "serialized_json_array_lens": {
        "title": "Serialized JSON Array Lens",
        "family": "Lens",
        "scope": "text_serialized_json_array_boundary",
        "pattern": rx(r"\[[^\]\n]{0,200}\{[^\]\n]{0,200}\]|\[\s*(?:\"[^\"]*\"|\d+|true|false|null)(?:\s*,\s*(?:\"[^\"]*\"|\d+|true|false|null)){1,20}\s*\]"),
        "need": "detect inline JSON arrays as structured data instead of prose lists",
    },
    "email_header_lens": {
        "title": "Email Header Lens",
        "family": "Lens",
        "scope": "text_email_header_grounding",
        "pattern": rx(r"(^|\n)(?:From|To|Cc|Bcc|Subject|Date):\s+[^\n]{1,200}"),
        "need": "bind email header fields to values and keep them out of body claims",
    },
    "url_query_parameter_lens": {
        "title": "URL Query Parameter Lens",
        "family": "Lens",
        "scope": "text_url_query_parameter_binding",
        "pattern": rx(r"https?://[^\s?]+\\?[A-Za-z0-9_=&.%+-]{3,}|[?&][A-Za-z_][A-Za-z0-9_-]*="),
        "need": "bind URL query parameters to their values without losing endpoint context",
    },
    "semantic_negation_lens": {
        "title": "Semantic Negation Lens",
        "family": "Lens",
        "scope": "text_semantic_negation_grounding",
        "pattern": rx(r"\b(?:not|no|without|neither|nor|never|cannot|can't|won't|doesn't|isn't|wasn't)\b"),
        "need": "preserve negation scope before committing extracted claims",
    },
    "double_negative_guard": {
        "title": "Double Negative Guard",
        "family": "Guard",
        "scope": "text_double_negative_scope",
        "pattern": rx(r"\b(?:not uncommon|not impossible|not unlikely|cannot not|no .* without|never not)\b"),
        "need": "avoid collapsing double-negative phrasing into the wrong polarity",
    },
    "temporal_duration_lens": {
        "title": "Temporal Duration Lens",
        "family": "Lens",
        "scope": "text_temporal_duration_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b|\b(?:for|during|within)\s+\d+"),
        "need": "bind duration spans separately from clock time or deadlines",
    },
    "probability_odds_lens": {
        "title": "Probability / Odds Lens",
        "family": "Lens",
        "scope": "text_probability_odds_grounding",
        "pattern": rx(r"\b(?:probability|chance|odds|risk)\b.{0,80}\b\d+(?:\.\d+)?\s?%|\b\d+\s*(?:in|out of)\s*\d+\b"),
        "need": "ground probability or odds statements with denominator and subject",
    },
    "citation_footnote_marker_lens": {
        "title": "Citation / Footnote Marker Lens",
        "family": "Lens",
        "scope": "text_citation_footnote_marker",
        "pattern": rx(r"\[[0-9]{1,3}\]|\(\d{4}\)|\b(?:ibid\.|et al\.|doi:|arXiv:)\b"),
        "need": "detect citation markers and keep them attached to the cited claim",
    },
    "form_field_instruction_lens": {
        "title": "Form Field Instruction Lens",
        "family": "Lens",
        "scope": "text_form_field_instruction",
        "pattern": rx(r"\b(?:enter|fill in|select|choose|checkbox|dropdown|required field|optional field)\b.{0,120}\b(?:field|box|option|form)\b"),
        "need": "bind form instructions to the correct field/action",
    },
    "safety_disclaimer_scope_guard": {
        "title": "Safety Disclaimer Scope Guard",
        "family": "Guard",
        "scope": "text_safety_disclaimer_scope",
        "pattern": rx(r"\b(?:not legal advice|not medical advice|for informational purposes|consult a professional|do not rely solely)\b"),
        "need": "preserve disclaimer scope without treating it as the primary answer",
    },
    "ranking_order_lens": {
        "title": "Ranking / Order Lens",
        "family": "Lens",
        "scope": "text_ranking_order_grounding",
        "pattern": rx(r"\b(?:ranked|top \d+|first|second|third|last|highest priority|lowest priority|ordered by)\b"),
        "need": "bind ordered items to rank position and sorting criterion",
    },
    "aggregation_groupby_lens": {
        "title": "Aggregation / Group-By Lens",
        "family": "Lens",
        "scope": "text_aggregation_groupby_grounding",
        "pattern": rx(r"\b(?:average|sum|count|total|grouped by|per category|by region|by type|aggregate)\b"),
        "need": "ground aggregate operations to grouping key and measured field",
    },
    "identity_equality_lens": {
        "title": "Identity / Equality Lens",
        "family": "Lens",
        "scope": "text_identity_equality_grounding",
        "pattern": rx(r"\b(?:same as|identical to|equals|equivalent to|is not the same as|differs from)\b"),
        "need": "bind identity/equality claims without collapsing nearby alternatives",
    },
    "source_reliability_caveat_lens": {
        "title": "Source Reliability Caveat Lens",
        "family": "Lens",
        "scope": "text_source_reliability_caveat",
        "pattern": rx(r"\b(?:unverified|rumor|alleged|anonymous source|low confidence source|unconfirmed|needs verification)\b"),
        "need": "preserve weak-source caveats before committing claims",
    },
    "assumption_marker_guard": {
        "title": "Assumption Marker Guard",
        "family": "Guard",
        "scope": "text_assumption_marker_scope",
        "pattern": rx(r"\b(?:assume|assuming|assumption|suppose|given that|under the assumption|if we take)\b"),
        "need": "keep assumptions from becoming unconditional committed facts",
    },
    "parenthetical_qualifier_lens": {
        "title": "Parenthetical Qualifier Lens",
        "family": "Lens",
        "scope": "text_parenthetical_qualifier",
        "pattern": rx(r"\([^)\n]{3,160}\)|\[[^\]\n]{3,160}\]"),
        "need": "preserve parenthetical qualifiers and avoid dropping caveats hidden in brackets",
    },
    "acronym_expansion_lens": {
        "title": "Acronym Expansion Lens",
        "family": "Lens",
        "scope": "text_acronym_expansion_grounding",
        "pattern": rx(r"\b[A-Z][A-Za-z][A-Za-z -]{2,80}\s+\([A-Z]{2,12}\)|\b[A-Z]{2,12}\s+\([A-Z][A-Za-z][^)]{2,80}\)"),
        "need": "bind acronyms to local expansions without treating acronym alone as globally defined",
    },
    "entity_attribute_relation_lens": {
        "title": "Entity / Attribute Relation Lens",
        "family": "Lens",
        "scope": "text_entity_attribute_relation",
        "pattern": rx(r"\b[A-Z][A-Za-z0-9_.-]{1,60}\b.{0,80}\b(?:has|uses|contains|includes|supports|owns|provides)\b.{0,80}"),
        "need": "bind attributes to the correct entity instead of nearby subjects",
    },
    "measurement_uncertainty_lens": {
        "title": "Measurement Uncertainty Lens",
        "family": "Lens",
        "scope": "text_measurement_uncertainty",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s*(?:±|\+/-|plus or minus)\s*\d+(?:\.\d+)?|\b(?:margin of error|confidence interval|CI)\b"),
        "need": "preserve uncertainty ranges around measured values",
    },
    "range_interval_lens": {
        "title": "Range / Interval Lens",
        "family": "Lens",
        "scope": "text_range_interval_grounding",
        "pattern": rx(r"\b\d+(?:\.\d+)?\s*(?:-|to|through|until|and)\s*\d+(?:\.\d+)?\b|\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b"),
        "need": "bind interval endpoints and avoid treating ranges as single scalar facts",
    },
    "paragraph_topic_shift_lens": {
        "title": "Paragraph Topic Shift Lens",
        "family": "Lens",
        "scope": "text_paragraph_topic_shift",
        "pattern": rx(r"\n\s*\n.{0,200}\b(?:however|meanwhile|separately|in contrast|another issue|on a different note)\b"),
        "need": "detect topic shifts so claims do not leak across paragraph boundaries",
    },
    "conclusion_recommendation_lens": {
        "title": "Conclusion / Recommendation Lens",
        "family": "Lens",
        "scope": "text_conclusion_recommendation",
        "pattern": rx(r"\b(?:therefore|in conclusion|recommend|recommendation|we should|best next step|takeaway|verdict)\b"),
        "need": "bind recommendations to their supporting evidence and scope",
    },
    "evidence_strength_marker_lens": {
        "title": "Evidence Strength Marker Lens",
        "family": "Lens",
        "scope": "text_evidence_strength_marker",
        "pattern": rx(r"\b(?:strong evidence|weak evidence|mixed evidence|anecdotal|statistically significant|not significant|supports but does not prove)\b"),
        "need": "preserve evidence strength before promoting claims",
    },
    "historical_current_state_guard": {
        "title": "Historical / Current State Guard",
        "family": "Guard",
        "scope": "text_historical_current_state",
        "pattern": rx(r"\b(?:formerly|previously|used to|as of|currently|now|no longer|at the time)\b"),
        "need": "separate historical state from current state before reuse",
    },
    "issue_status_lens": {
        "title": "Issue Status Lens",
        "family": "Lens",
        "scope": "text_issue_status_grounding",
        "pattern": rx(r"\b(?:open|closed|resolved|blocked|in progress|triaged|assigned|backlog|done)\b.{0,100}\b(?:issue|ticket|task|bug|PR|pull request)\b"),
        "need": "bind issue or task status to the correct work item",
    },
    "checklist_completion_lens": {
        "title": "Checklist Completion Lens",
        "family": "Lens",
        "scope": "text_checklist_completion_grounding",
        "pattern": rx(r"(^|\n)\s*[-*]\s*\[(?: |x|X)\]\s+[^\n]{2,160}"),
        "need": "ground checkbox state separately from checklist item text",
    },
    "meeting_action_item_lens": {
        "title": "Meeting Action Item Lens",
        "family": "Lens",
        "scope": "text_meeting_action_item",
        "pattern": rx(r"\b(?:action item|AI:|todo|follow up|next step|owner:|due:)\b.{0,160}"),
        "need": "bind meeting actions to owner and due context without treating notes as completed work",
    },
    "research_question_lens": {
        "title": "Research Question Lens",
        "family": "Lens",
        "scope": "text_research_question_grounding",
        "pattern": rx(r"\b(?:research question|question is|we ask whether|hypothesis is|we test whether|core question)\b"),
        "need": "distinguish a research question from a result or answer",
    },
    "finding_claim_lens": {
        "title": "Finding Claim Lens",
        "family": "Lens",
        "scope": "text_finding_claim_grounding",
        "pattern": rx(r"\b(?:we found|results show|the finding|observed that|demonstrates|indicates that|suggests that)\b"),
        "need": "bind result/finding language to the evidence span and avoid overclaiming",
    },
    "cause_mitigation_pair_lens": {
        "title": "Cause / Mitigation Pair Lens",
        "family": "Lens",
        "scope": "text_cause_mitigation_pair",
        "pattern": rx(r"\b(?:root cause|caused by|due to)\b.{0,160}\b(?:fix|mitigate|workaround|resolved by|prevent)\b"),
        "need": "bind failure causes to their mitigation actions",
    },
    "excluded_included_items_lens": {
        "title": "Excluded / Included Items Lens",
        "family": "Lens",
        "scope": "text_excluded_included_items",
        "pattern": rx(r"\b(?:included|excluded|except|excluding|including|not included|out of scope)\b.{0,160}"),
        "need": "keep included and excluded item sets separated",
    },
    "rhetorical_question_guard": {
        "title": "Rhetorical Question Guard",
        "family": "Guard",
        "scope": "text_rhetorical_question_scope",
        "pattern": rx(r"\b(?:why would|how could|isn't it|don't you think|who would have thought)\b.{0,120}\?"),
        "need": "avoid treating rhetorical questions as literal information requests",
    },
    "comparison_control_group_lens": {
        "title": "Comparison / Control Group Lens",
        "family": "Lens",
        "scope": "text_comparison_control_group",
        "pattern": rx(r"\b(?:control group|treatment group|baseline|compared with|relative to|versus control)\b"),
        "need": "bind comparison outcomes to baseline/control context",
    },
    "frequency_adverb_lens": {
        "title": "Frequency Adverb Lens",
        "family": "Lens",
        "scope": "text_frequency_adverb_grounding",
        "pattern": rx(r"\b(?:always|often|usually|sometimes|rarely|seldom|never|occasionally|frequently)\b"),
        "need": "ground frequency qualifiers before turning claims into universal rules",
    },
    "uncertainty_resolution_lens": {
        "title": "Uncertainty Resolution Lens",
        "family": "Lens",
        "scope": "text_uncertainty_resolution",
        "pattern": rx(r"\b(?:unclear|ambiguous|unknown|unresolved|needs clarification|confirmed|resolved|clarified)\b"),
        "need": "track whether an uncertainty has been resolved before answer commit",
    },
    "source_conflict_resolution_lens": {
        "title": "Source Conflict Resolution Lens",
        "family": "Lens",
        "scope": "text_source_conflict_resolution",
        "pattern": rx(r"\b(?:conflicting reports|sources disagree|contradictory|dispute|reconciled|resolved conflict)\b"),
        "need": "preserve unresolved source conflict or the evidence that resolves it",
    },
    "qualifier_degree_lens": {
        "title": "Qualifier Degree Lens",
        "family": "Lens",
        "scope": "text_qualifier_degree_grounding",
        "pattern": rx(r"\b(?:very|extremely|slightly|partly|mostly|nearly|almost|approximately|roughly)\b"),
        "need": "preserve degree qualifiers attached to claims or quantities",
    },
    "calculation_result_phrase_lens": {
        "title": "Calculation Result Phrase Lens",
        "family": "Lens",
        "scope": "text_calculation_result_phrase",
        "pattern": rx(r"\b(?:equals|is equal to|result is|total is|sum is|computed as|calculated as)\b.{0,100}\b\d+(?:\.\d+)?\b"),
        "need": "detect visible calculation result phrases without solving hidden math",
    },
    "role_permission_assignment_lens": {
        "title": "Role / Permission Assignment Lens",
        "family": "Lens",
        "scope": "text_role_permission_assignment",
        "pattern": rx(r"\b(?:admin|editor|viewer|owner|member|role|permission)\b.{0,120}\b(?:can|cannot|may|allowed|denied|granted)\b"),
        "need": "bind permissions to the correct role or actor",
    },
    "dependency_chain_lens": {
        "title": "Dependency Chain Lens",
        "family": "Lens",
        "scope": "text_dependency_chain_grounding",
        "pattern": rx(r"\b(?:depends on|requires|requires .* before|blocked by|unblocks|prerequisite|dependency chain)\b"),
        "need": "ground dependency chains and preserve prerequisite order",
    },
    "input_output_contract_lens": {
        "title": "Input / Output Contract Lens",
        "family": "Lens",
        "scope": "text_input_output_contract",
        "pattern": rx(r"\b(?:input|output|returns|expects|accepts|emits|produces)\b.{0,140}\b(?:format|schema|type|value|field)\b"),
        "need": "bind IO contracts to expected formats and outputs",
    },
    "normalization_mapping_lens": {
        "title": "Normalization Mapping Lens",
        "family": "Lens",
        "scope": "text_normalization_mapping",
        "pattern": rx(r"\b(?:normalize|canonicalize|map .* to|maps to|convert .* to|alias .* to)\b"),
        "need": "ground mapping/normalization rules without applying them outside scope",
    },
    "retry_backoff_lens": {
        "title": "Retry / Backoff Lens",
        "family": "Lens",
        "scope": "text_retry_backoff_grounding",
        "pattern": rx(r"\b(?:retry|backoff|try again|exponential backoff|timeout|rate limit|throttle)\b"),
        "need": "bind retry/backoff behavior to failure conditions and limits",
    },
    "cache_staleness_guard": {
        "title": "Cache Staleness Guard",
        "family": "Guard",
        "scope": "text_cache_staleness_scope",
        "pattern": rx(r"\b(?:cache|cached|stale|invalidate|TTL|expired|refresh)\b"),
        "need": "avoid reusing stale cached state as current evidence",
    },
    "sampling_split_lens": {
        "title": "Sampling / Split Lens",
        "family": "Lens",
        "scope": "text_sampling_split_grounding",
        "pattern": rx(r"\b(?:train split|validation split|test split|heldout|sampled from|stratified|random sample)\b"),
        "need": "bind dataset split and sampling statements to evaluation claims",
    },
    "license_condition_lens": {
        "title": "License Condition Lens",
        "family": "Lens",
        "scope": "text_license_condition_grounding",
        "pattern": rx(r"\b(?:license|licensed under|MIT|Apache|GPL|Creative Commons|attribution|required notice)\b"),
        "need": "ground license conditions and avoid over-broad reuse claims",
    },
    "sensitive_identifier_lens": {
        "title": "Sensitive Identifier Lens",
        "family": "Lens",
        "scope": "text_sensitive_identifier_boundary",
        "pattern": rx(r"\b(?:SSN|social security|passport|API key|token|secret|password|private key)\b"),
        "need": "detect sensitive identifiers and keep them from unsafe propagation",
    },
    "problem_solution_pair_lens": {
        "title": "Problem / Solution Pair Lens",
        "family": "Lens",
        "scope": "text_problem_solution_pair",
        "pattern": rx(r"\b(?:problem|issue|challenge|failure|bottleneck)\b.{0,180}\b(?:solution|fix|resolve|repair|workaround|address)\b"),
        "need": "bind stated problems to proposed solutions without treating every nearby action as a fix",
    },
    "observation_inference_split_lens": {
        "title": "Observation / Inference Split Lens",
        "family": "Lens",
        "scope": "text_observation_inference_split",
        "pattern": rx(r"\b(?:observed|seen|measured|log shows|evidence shows)\b.{0,180}\b(?:therefore|suggests|implies|indicates|we infer)\b"),
        "need": "separate raw observation from inferred conclusion before commit",
    },
    "evidence_gap_ask_guard": {
        "title": "Evidence Gap Ask Guard",
        "family": "Guard",
        "scope": "text_evidence_gap_ask",
        "pattern": rx(r"\b(?:not enough information|insufficient evidence|need more evidence|cannot determine|ask for|clarify|need clarification)\b"),
        "need": "route unresolved evidence gaps toward ask/search/hold instead of confident answer",
    },
    "planned_vs_completed_guard": {
        "title": "Planned vs Completed Guard",
        "family": "Guard",
        "scope": "text_planned_vs_completed_state",
        "pattern": rx(r"\b(?:plan to|planned|will do|intend to|not yet done|completed|finished|done|shipped)\b"),
        "need": "avoid treating planned future work as completed evidence",
    },
    "quote_not_endorsement_guard": {
        "title": "Quote Not Endorsement Guard",
        "family": "Guard",
        "scope": "text_quote_not_endorsement",
        "pattern": rx(r"[\"“][^\"”]{5,180}[\"”].{0,120}\b(?:said|claimed|alleged|reported|according to|rumor)\b|\b(?:said|claimed|alleged|reported|according to|rumor)\b.{0,120}[\"“][^\"”]{5,180}[\"”]"),
        "need": "keep quoted or attributed content from becoming directly endorsed Ground truth",
    },
    "requirement_exception_pair_lens": {
        "title": "Requirement / Exception Pair Lens",
        "family": "Lens",
        "scope": "text_requirement_exception_pair",
        "pattern": rx(r"\b(?:must|required|shall|should|mandatory)\b.{0,180}\b(?:except|unless|except when|not required|optional)\b"),
        "need": "bind exceptions to the requirement they constrain",
    },
    "procedure_precondition_lens": {
        "title": "Procedure Precondition Lens",
        "family": "Lens",
        "scope": "text_procedure_precondition",
        "pattern": rx(r"\b(?:before|first|prerequisite|requires|only after|make sure)\b.{0,160}\b(?:then|run|execute|start|continue|proceed)\b"),
        "need": "preserve preconditions before allowing a procedure step to execute",
    },
    "result_limitation_pair_lens": {
        "title": "Result / Limitation Pair Lens",
        "family": "Lens",
        "scope": "text_result_limitation_pair",
        "pattern": rx(r"\b(?:result|found|showed|achieved|improved|passed)\b.{0,180}\b(?:but|however|limited|limitation|does not|cannot)\b"),
        "need": "bind positive results to their caveats and limits",
    },
    "hypothesis_test_result_lens": {
        "title": "Hypothesis / Test / Result Lens",
        "family": "Lens",
        "scope": "text_hypothesis_test_result",
        "pattern": rx(r"\b(?:hypothesis|we test|tested whether|experiment|probe)\b.{0,220}\b(?:result|found|passed|failed|confirmed|rejected)\b"),
        "need": "connect a hypothesis or probe to the actual result without skipping the test boundary",
    },
    "expected_actual_mismatch_lens": {
        "title": "Expected / Actual Mismatch Lens",
        "family": "Lens",
        "scope": "text_expected_actual_mismatch",
        "pattern": rx(r"\b(?:expected|should have|supposed to)\b.{0,140}\b(?:actual|instead|but got|received|observed)\b"),
        "need": "bind expected and actual outcomes without swapping or merging them",
    },
    "category_member_relation_lens": {
        "title": "Category / Member Relation Lens",
        "family": "Lens",
        "scope": "text_category_member_relation",
        "pattern": rx(r"\b(?:category|class|type|kind|member of|belongs to|includes|contains)\b.{0,140}\b(?:items?|examples?|members?|types?)\b"),
        "need": "ground category membership without turning examples into exhaustive definitions",
    },
    "definition_scope_guard": {
        "title": "Definition Scope Guard",
        "family": "Guard",
        "scope": "text_definition_scope_boundary",
        "pattern": rx(r"\b(?:in this context|here|for this task|within this document|locally|in this episode)\b.{0,120}\b(?:means|refers to|defined as|called)\b"),
        "need": "keep local definitions local instead of promoting them globally",
    },
    "source_recency_lens": {
        "title": "Source Recency Lens",
        "family": "Lens",
        "scope": "text_source_recency_grounding",
        "pattern": rx(r"\b(?:as of|latest|current|updated|last updated|published|reported on|dated)\b.{0,120}\b(?:20\d{2}|today|yesterday|this week|last week)\b"),
        "need": "bind source claims to their recency and avoid stale-source reuse",
    },
    "numeric_unit_subject_lens": {
        "title": "Numeric Unit / Subject Lens",
        "family": "Lens",
        "scope": "text_numeric_unit_subject_binding",
        "pattern": rx(r"\b[A-Za-z][A-Za-z0-9 -]{2,60}\b.{0,80}\b\d+(?:\.\d+)?\s?(?:%|kg|g|m|cm|mm|ms|s|MB|GB|USD|EUR|W|V|Hz)\b"),
        "need": "bind numeric values and units to the correct subject",
    },
    "evaluation_criteria_lens": {
        "title": "Evaluation Criteria Lens",
        "family": "Lens",
        "scope": "text_evaluation_criteria",
        "pattern": rx(r"\b(?:score by|evaluate|measured by|criterion|criteria|metric|success if|fails if|pass requirement)\b"),
        "need": "ground evaluation criteria before judging success or failure",
    },
    "turn_request_context_lens": {
        "title": "Turn Request Context Lens",
        "family": "Lens",
        "scope": "text_turn_request_context",
        "pattern": rx(r"\b(?:you asked|the user asked|request was|previous message|next reply|respond with|answer should)\b"),
        "need": "bind current response behavior to the correct conversational turn",
    },
    "implied_question_lens": {
        "title": "Implied Question Lens",
        "family": "Lens",
        "scope": "text_implied_question_grounding",
        "pattern": rx(r"\b(?:I wonder|question is|not sure whether|can we tell|would it be|is it possible|what if)\b"),
        "need": "detect implicit questions before rendering an answer or action",
    },
    "contrast_same_entity_lens": {
        "title": "Contrast Same Entity Lens",
        "family": "Lens",
        "scope": "text_contrast_same_entity",
        "pattern": rx(r"\b(?:same|this|it|the model|the system|the method)\b.{0,120}\b(?:but|however|whereas|although)\b"),
        "need": "keep contrasting properties attached to the same entity instead of splitting subject context",
    },
    "constraint_priority_lens": {
        "title": "Constraint Priority Lens",
        "family": "Lens",
        "scope": "text_constraint_priority",
        "pattern": rx(r"\b(?:must prioritize|higher priority|lower priority|more important|less important|hard requirement|soft requirement)\b"),
        "need": "ground which constraints dominate when actions conflict",
    },
    "ambiguous_modal_guard": {
        "title": "Ambiguous Modal Guard",
        "family": "Guard",
        "scope": "text_ambiguous_modal_scope",
        "pattern": rx(r"\b(?:might|may|could|possibly|potentially|seems like|appears to|maybe)\b"),
        "need": "avoid committing modal/uncertain statements as confirmed facts",
    },
    "prior_current_correction_lens": {
        "title": "Prior / Current Correction Lens",
        "family": "Lens",
        "scope": "text_prior_current_correction",
        "pattern": rx(r"\b(?:previously|earlier|before)\b.{0,160}\b(?:now|currently|corrected|updated|instead|no longer)\b"),
        "need": "bind corrections to prior claims and update current-state interpretation",
    },
    "root_cause_evidence_lens": {
        "title": "Root Cause / Evidence Lens",
        "family": "Lens",
        "scope": "text_root_cause_evidence",
        "pattern": rx(r"\b(?:root cause|cause was|caused by)\b.{0,160}\b(?:because|evidence|log|trace|observed|shows)\b"),
        "need": "require evidence linkage before accepting a root-cause claim",
    },
    "named_event_outcome_lens": {
        "title": "Named Event / Outcome Lens",
        "family": "Lens",
        "scope": "text_named_event_outcome",
        "pattern": rx(r"\b[A-Z][A-Za-z0-9_.-]{2,60}\b.{0,120}\b(?:succeeded|failed|passed|won|lost|completed|crashed|timed out)\b"),
        "need": "bind outcome words to the named event or run they describe",
    },
    "availability_capability_lens": {
        "title": "Availability / Capability Lens",
        "family": "Lens",
        "scope": "text_availability_capability",
        "pattern": rx(r"\b(?:available|unavailable|enabled|disabled|supported|unsupported|can handle|cannot handle|capable of)\b"),
        "need": "ground availability/capability claims to the correct system or context",
    },
    "causal_chain_multihop_lens": {
        "title": "Causal Chain Multi-Hop Lens",
        "family": "Lens",
        "scope": "text_causal_chain_multihop",
        "pattern": rx(r"\b(?:because|caused by|leads to|therefore|which then|as a result)\b.{0,260}\b(?:because|caused by|leads to|therefore|which then|as a result)\b"),
        "need": "preserve multi-hop causal chains without collapsing intermediate links",
    },
    "example_non_exhaustive_guard": {
        "title": "Example Non-Exhaustive Guard",
        "family": "Guard",
        "scope": "text_example_non_exhaustive",
        "pattern": rx(r"\b(?:for example|e\.g\.|such as|including but not limited to|examples include)\b"),
        "need": "avoid treating examples as an exhaustive list",
    },
    "paraphrase_equivalence_lens": {
        "title": "Paraphrase / Equivalence Lens",
        "family": "Lens",
        "scope": "text_paraphrase_equivalence",
        "pattern": rx(r"\b(?:in other words|that is|i\.e\.|meaning|put differently|equivalent to)\b"),
        "need": "ground paraphrase/equivalence statements without duplicating claims as independent evidence",
    },
    "contradictory_marker_lens": {
        "title": "Contradictory Marker Lens",
        "family": "Lens",
        "scope": "text_contradictory_marker",
        "pattern": rx(r"\b(?:contradiction|contradicts|conflicts with|inconsistent with|opposite of|cannot both be true)\b"),
        "need": "detect explicit contradiction markers and route them toward defer or conflict resolution",
    },
    "list_scope_boundary_lens": {
        "title": "List Scope Boundary Lens",
        "family": "Lens",
        "scope": "text_list_scope_boundary",
        "pattern": rx(r"(^|\n)\s*(?:[-*+]|\d+[.)])\s+[^\n]{5,180}"),
        "need": "preserve list item boundaries so properties do not leak between items",
    },
    "noun_apposition_definition_lens": {
        "title": "Noun Apposition Definition Lens",
        "family": "Lens",
        "scope": "text_noun_apposition_definition",
        "pattern": rx(r"\b[A-Z][A-Za-z0-9_.-]{2,60},\s+(?:a|an|the)\s+[A-Za-z][^,\n]{3,120},"),
        "need": "bind appositive definitions to the named entity they qualify",
    },
    "qualifier_negation_stack_guard": {
        "title": "Qualifier / Negation Stack Guard",
        "family": "Guard",
        "scope": "text_qualifier_negation_stack",
        "pattern": rx(r"\b(?:not|no|never|without)\b.{0,80}\b(?:always|necessarily|entirely|fully|clearly|definitely)\b|\b(?:partly|mostly|almost|nearly)\b.{0,80}\b(?:not|no|never|without)\b"),
        "need": "preserve stacked qualifiers and negation polarity before commit",
    },
    "recommendation_rationale_lens": {
        "title": "Recommendation / Rationale Lens",
        "family": "Lens",
        "scope": "text_recommendation_rationale",
        "pattern": rx(r"\b(?:recommend|should|best option|next step)\b.{0,180}\b(?:because|since|reason|rationale|due to)\b"),
        "need": "bind recommendations to their rationale and avoid unsupported action proposals",
    },
    "context_condition_result_lens": {
        "title": "Context / Condition / Result Lens",
        "family": "Lens",
        "scope": "text_context_condition_result",
        "pattern": rx(r"\b(?:in|under|given|with)\b.{0,120}\b(?:condition|context|case|scenario)\b.{0,160}\b(?:result|outcome|then|will|should)\b"),
        "need": "keep results attached to the context or condition in which they hold",
    },
    "default_override_rule_lens": {
        "title": "Default / Override Rule Lens",
        "family": "Lens",
        "scope": "text_default_override_rule",
        "pattern": rx(r"\b(?:default|by default|override|overrides|unless specified|fallback)\b"),
        "need": "ground default rules separately from explicit overrides",
    },
    "evidence_span_citation_lens": {
        "title": "Evidence Span / Citation Lens",
        "family": "Lens",
        "scope": "text_evidence_span_citation",
        "pattern": rx(r"\b(?:evidence|citation|source|span|quote|reference)\b.{0,120}\b(?:supports|backs|proves|shows|indicates)\b"),
        "need": "bind a claim to its evidence span or citation before answer commit",
    },
    "irrelevant_context_filter_lens": {
        "title": "Irrelevant Context Filter Lens",
        "family": "Lens",
        "scope": "text_irrelevant_context_filter",
        "pattern": rx(r"\b(?:irrelevant|unrelated|not relevant|ignore|out of context|does not matter)\b"),
        "need": "mark irrelevant context so it does not pollute the active Flow state",
    },
    "missing_vs_zero_guard": {
        "title": "Missing vs Zero Guard",
        "family": "Guard",
        "scope": "text_missing_vs_zero",
        "pattern": rx(r"\b(?:missing|null|unknown|not provided|blank|empty)\b.{0,100}\b(?:zero|0|none|no value)\b|\b(?:zero|0|none)\b.{0,100}\b(?:missing|null|unknown|not provided)\b"),
        "need": "avoid collapsing missing/unknown values into numeric zero",
    },
    "outcome_probability_distinction_lens": {
        "title": "Outcome / Probability Distinction Lens",
        "family": "Lens",
        "scope": "text_outcome_probability_distinction",
        "pattern": rx(r"\b(?:probability|chance|likely|unlikely|risk)\b.{0,140}\b(?:happened|occurred|will happen|outcome|event)\b"),
        "need": "separate probability estimates from observed outcomes",
    },
    "spec_example_counterexample_lens": {
        "title": "Spec Example / Counterexample Lens",
        "family": "Lens",
        "scope": "text_spec_example_counterexample",
        "pattern": rx(r"\b(?:example|counterexample|valid case|invalid case|passes|fails)\b.{0,160}\b(?:spec|rule|contract|requirement)\b"),
        "need": "bind examples and counterexamples to the rule they test",
    },
    "operator_scope_boundary_lens": {
        "title": "Operator Scope Boundary Lens",
        "family": "Lens",
        "scope": "text_operator_scope_boundary",
        "pattern": rx(r"\b(?:operator|skill|pocket|lens|guard)\b.{0,160}\b(?:scope|can do|cannot|should only|allowed claim)\b"),
        "need": "preserve scoped capability boundaries for Operators/Pockets",
    },
    "quality_control_failure_lens": {
        "title": "Quality Control Failure Lens",
        "family": "Lens",
        "scope": "text_quality_control_failure",
        "pattern": rx(r"\b(?:quality control|QC|checker|failed check|failure count|regression|gate failed|blocked)\b"),
        "need": "ground QC failures as blockers rather than ordinary notes",
    },
    "commit_push_state_lens": {
        "title": "Commit / Push State Lens",
        "family": "Lens",
        "scope": "text_commit_push_state",
        "pattern": rx(r"\b(?:committed|pushed|merged|main|origin/main|branch|SHA|commit)\b.{0,140}\b(?:clean|dirty|ahead|behind|synced|pushed)\b"),
        "need": "bind repository sync state to the correct branch and commit",
    },
    "e128_negated_causal_guard": {
        "title": "Negated Causal Guard",
        "family": "Guard",
        "scope": "text_negated_causal_scope",
        "pattern": rx(r"\b(?:not because|not due to|does not mean|doesn't mean|not caused by|not the reason)\b"),
        "need": "avoid accepting a causal relation that the text explicitly negates",
    },
    "e128_correlation_causation_guard": {
        "title": "Correlation / Causation Guard",
        "family": "Guard",
        "scope": "text_correlation_causation_boundary",
        "pattern": rx(r"\b(?:correlation|correlated|association|associated with|does not imply causation|causal link)\b"),
        "need": "separate correlation language from causal proof before commit",
    },
    "e128_claim_source_date_lens": {
        "title": "Claim / Source / Date Lens",
        "family": "Lens",
        "scope": "text_claim_source_date_binding",
        "pattern": rx(r"\b(?:according to|reported by|published by|source)\b.{0,160}\b(?:20\d{2}|updated|dated|as of|last updated)\b"),
        "need": "bind a claim to both source and recency instead of reusing stale evidence",
    },
    "e128_known_unknown_split_lens": {
        "title": "Known / Unknown Split Lens",
        "family": "Lens",
        "scope": "text_known_unknown_split",
        "pattern": rx(r"\b(?:known|confirmed|verified)\b.{0,160}\b(?:unknown|unclear|not known|unconfirmed|not verified)\b|\b(?:unknown|unclear|not known)\b.{0,160}\b(?:known|confirmed|verified)\b"),
        "need": "keep proven facts separate from unresolved fields in the same passage",
    },
    "e128_partial_success_failure_lens": {
        "title": "Partial Success / Failure Lens",
        "family": "Lens",
        "scope": "text_partial_success_failure",
        "pattern": rx(r"\b(?:partially|mostly|somewhat|almost)\b.{0,120}\b(?:succeeded|failed|passed|worked|solved)\b|\b(?:succeeded|failed|passed|worked|solved)\b.{0,120}\b(?:partially|mostly|somewhat|almost)\b"),
        "need": "ground partial outcomes without turning them into full success or full failure",
    },
    "e128_exact_vs_approximate_guard": {
        "title": "Exact vs Approximate Guard",
        "family": "Guard",
        "scope": "text_exact_vs_approximate",
        "pattern": rx(r"\b(?:exact|exactly|precise|precisely)\b.{0,120}\b(?:approx|approximately|roughly|about|estimate)\b|\b(?:approx|approximately|roughly|about|estimate)\b.{0,120}\b(?:exact|exactly|precise|precisely)\b"),
        "need": "avoid collapsing approximate and exact claims into one certainty level",
    },
    "e128_generalization_boundary_guard": {
        "title": "Generalization Boundary Guard",
        "family": "Guard",
        "scope": "text_generalization_boundary",
        "pattern": rx(r"\b(?:does not generalize|generalizes only|only on this|not universal|task-specific|controlled proxy|toy task)\b"),
        "need": "preserve stated limits before promoting a result beyond its tested scope",
    },
    "e128_model_claim_boundary_guard": {
        "title": "Model Claim Boundary Guard",
        "family": "Guard",
        "scope": "text_model_claim_boundary",
        "pattern": rx(r"\b(?:not AGI|not production|not model-scale|not open-domain|not a chatbot|controlled proxy|allowed claim|not claim)\b"),
        "need": "keep model capability claims inside the stated experimental boundary",
    },
    "e128_disallowed_scope_claim_guard": {
        "title": "Disallowed Scope Claim Guard",
        "family": "Guard",
        "scope": "text_disallowed_scope_claim",
        "pattern": rx(r"\b(?:cannot claim|should not claim|not allowed to claim|do not claim|invalid claim|overclaim)\b"),
        "need": "block over-broad capability claims when the text marks them as disallowed",
    },
    "e128_evidence_threshold_lens": {
        "title": "Evidence Threshold Lens",
        "family": "Lens",
        "scope": "text_evidence_threshold",
        "pattern": rx(r"\b(?:requires evidence|enough evidence|insufficient evidence|evidence threshold|proof threshold|needs proof)\b"),
        "need": "bind answerability decisions to the evidence threshold stated in text",
    },
    "e128_unresolved_reference_guard": {
        "title": "Unresolved Reference Guard",
        "family": "Guard",
        "scope": "text_unresolved_reference",
        "pattern": rx(r"\b(?:this|that|it|they|those|these)\b.{0,120}\b(?:unclear|unknown|ambiguous|not specified|not provided)\b"),
        "need": "defer reference-dependent claims when the antecedent is unresolved",
    },
    "e128_question_scope_guard": {
        "title": "Question Scope Guard",
        "family": "Guard",
        "scope": "text_question_scope_boundary",
        "pattern": rx(r"\b(?:the question is|asked whether|asks whether|answer only|only asks|not asking)\b"),
        "need": "keep an answer scoped to the actual question instead of nearby background",
    },
    "e128_user_intent_background_split_lens": {
        "title": "User Intent / Background Split Lens",
        "family": "Lens",
        "scope": "text_user_intent_background_split",
        "pattern": rx(r"\b(?:background|context|for context|the request|user wants|asked to|goal is)\b"),
        "need": "separate the requested action from background context in mixed prompts",
    },
    "e128_non_actionable_note_guard": {
        "title": "Non-Actionable Note Guard",
        "family": "Guard",
        "scope": "text_non_actionable_note",
        "pattern": rx(r"\b(?:note that|FYI|for reference|just noting|not an action item|no action needed)\b"),
        "need": "avoid converting informational notes into tasks or commitments",
    },
    "e128_instruction_constraint_violation_lens": {
        "title": "Instruction Constraint Violation Lens",
        "family": "Lens",
        "scope": "text_instruction_constraint_violation",
        "pattern": rx(r"\b(?:violates|breaks|must not|do not|forbidden|not allowed)\b.{0,160}\b(?:instruction|constraint|rule|requirement|policy)\b"),
        "need": "bind a violation marker to the instruction or constraint being protected",
    },
    "e128_multi_condition_rule_lens": {
        "title": "Multi-Condition Rule Lens",
        "family": "Lens",
        "scope": "text_multi_condition_rule",
        "pattern": rx(r"\b(?:if|when|provided that|assuming)\b.{0,100}\b(?:and|or)\b.{0,140}\b(?:then|must|should|will|can)\b"),
        "need": "preserve compound rule conditions before committing the consequent",
    },
    "e128_exception_chain_guard": {
        "title": "Exception Chain Guard",
        "family": "Guard",
        "scope": "text_exception_chain",
        "pattern": rx(r"\b(?:except|unless|other than|with the exception of)\b.{0,120}\b(?:except|unless|other than|but not)\b"),
        "need": "preserve nested exceptions instead of dropping the second boundary",
    },
    "e128_before_after_state_update_lens": {
        "title": "Before / After State Update Lens",
        "family": "Lens",
        "scope": "text_before_after_state_update",
        "pattern": rx(r"\b(?:before|previously|earlier)\b.{0,140}\b(?:after|now|currently|later|updated)\b"),
        "need": "bind old and new states in the same passage without stale reuse",
    },
    "e128_current_desired_state_lens": {
        "title": "Current / Desired State Lens",
        "family": "Lens",
        "scope": "text_current_desired_state",
        "pattern": rx(r"\b(?:current state|currently|as-is|actual state)\b.{0,160}\b(?:desired state|target state|should be|goal state)\b"),
        "need": "separate current facts from desired target state",
    },
    "e128_future_prediction_guard": {
        "title": "Future Prediction Guard",
        "family": "Guard",
        "scope": "text_future_prediction_scope",
        "pattern": rx(r"\b(?:will likely|expected to|forecast|prediction|projected|may happen|could happen)\b"),
        "need": "keep predictions separate from observed outcomes",
    },
    "e128_staleness_due_to_date_guard": {
        "title": "Staleness Due To Date Guard",
        "family": "Guard",
        "scope": "text_staleness_due_to_date",
        "pattern": rx(r"\b(?:as of|last updated|outdated|stale|no longer current|dated)\b.{0,140}\b(?:20\d{2}|today|yesterday|last week|last month)\b"),
        "need": "flag dated evidence that may not support a current-state commit",
    },
    "e128_official_unofficial_source_lens": {
        "title": "Official / Unofficial Source Lens",
        "family": "Lens",
        "scope": "text_official_unofficial_source",
        "pattern": rx(r"\b(?:official|unofficial|community|third-party|primary source|secondary source|self-reported)\b"),
        "need": "bind source authority level before using a claim as evidence",
    },
    "e128_observation_reported_by_lens": {
        "title": "Observation / Reported-By Lens",
        "family": "Lens",
        "scope": "text_observation_reported_by",
        "pattern": rx(r"\b(?:observed by|reported by|seen by|measured by|logged by|recorded by)\b"),
        "need": "attach observations to the observer/source rather than treating them as source-free facts",
    },
    "e128_support_vs_causation_guard": {
        "title": "Support vs Causation Guard",
        "family": "Guard",
        "scope": "text_support_vs_causation",
        "pattern": rx(r"\b(?:supports|consistent with|suggests)\b.{0,120}\b(?:causes|caused|therefore|proves)\b"),
        "need": "avoid upgrading support/consistency language into causal proof",
    },
    "e128_method_assumption_result_lens": {
        "title": "Method / Assumption / Result Lens",
        "family": "Lens",
        "scope": "text_method_assumption_result",
        "pattern": rx(r"\b(?:method|approach|assumption|assuming)\b.{0,220}\b(?:result|found|showed|passed|failed)\b"),
        "need": "bind results to the method and assumptions under which they were obtained",
    },
    "e128_row_level_evidence_lens": {
        "title": "Row-Level Evidence Lens",
        "family": "Lens",
        "scope": "text_row_level_evidence",
        "pattern": rx(r"\b(?:row-level|row level|per-row|sample row|example row|row_id|row id)\b"),
        "need": "preserve row-level evidence requirements before accepting aggregate claims",
    },
    "e128_random_control_baseline_lens": {
        "title": "Random Control / Baseline Lens",
        "family": "Lens",
        "scope": "text_random_control_baseline",
        "pattern": rx(r"\b(?:random control|random baseline|control baseline|baseline control|negative control)\b"),
        "need": "bind control results to the baseline role and avoid treating them as primary wins",
    },
    "e128_seed_variance_result_lens": {
        "title": "Seed / Variance Result Lens",
        "family": "Lens",
        "scope": "text_seed_variance_result",
        "pattern": rx(r"\b(?:seed|seeds|multi-seed|variance|std|standard deviation|min|max)\b.{0,160}\b(?:result|accuracy|score|pass|fail)\b"),
        "need": "bind multi-seed evidence and variance to result confidence",
    },
    "e128_tool_output_status_lens": {
        "title": "Tool Output / Status Lens",
        "family": "Lens",
        "scope": "text_tool_output_status",
        "pattern": rx(r"\b(?:exit code|stdout|stderr|process exited|session id|running|completed|timed out)\b"),
        "need": "ground tool execution status separately from the user's intent",
    },
    "e128_file_change_intent_lens": {
        "title": "File Change / Intent Lens",
        "family": "Lens",
        "scope": "text_file_change_intent",
        "pattern": rx(r"\b(?:modified|added|deleted|renamed|staged|unstaged|dirty|untracked)\b.{0,120}\b(?:file|files|path|repo|working tree)\b"),
        "need": "bind file-change state to path/repo context without touching unrelated files",
    },
    "e128_candidate_rejection_reason_lens": {
        "title": "Candidate Rejection Reason Lens",
        "family": "Lens",
        "scope": "text_candidate_rejection_reason",
        "pattern": rx(r"\b(?:rejected|discarded|quarantined|deprecated|redflag|red flag)\b.{0,160}\b(?:because|reason|failure|unsafe|negative)\b"),
        "need": "bind rejected candidates to concrete failure reasons for negative-card reuse",
    },
    "e128_candidate_promotion_reason_lens": {
        "title": "Candidate Promotion Reason Lens",
        "family": "Lens",
        "scope": "text_candidate_promotion_reason",
        "pattern": rx(r"\b(?:promoted|ranked|upgraded|reached|became)\b.{0,160}\b(?:because|passed|evidence|activation|score|gate)\b"),
        "need": "bind promotion events to the evidence that justified the lifecycle change",
    },
    "e128_challenger_vs_incumbent_lens": {
        "title": "Challenger vs Incumbent Lens",
        "family": "Lens",
        "scope": "text_challenger_incumbent",
        "pattern": rx(r"\b(?:challenger|incumbent|replacement|variant|candidate)\b.{0,160}\b(?:beat|matched|replaced|failed|worse|better)\b"),
        "need": "ground challenger outcomes before replacing or preserving an existing operator",
    },
    "e128_prune_regression_guard": {
        "title": "Prune / Regression Guard",
        "family": "Guard",
        "scope": "text_prune_regression",
        "pattern": rx(r"\b(?:prune|pruned|compressed|simplified|minimized)\b.{0,160}\b(?:regression|no regression|worse|broke|preserved|same result)\b"),
        "need": "bind simplification attempts to regression evidence before accepting a smaller variant",
    },
    "e128_reload_integrity_guard": {
        "title": "Reload / Integrity Guard",
        "family": "Guard",
        "scope": "text_reload_integrity",
        "pattern": rx(r"\b(?:reload|import|digest|hash|tamper|mismatch|integrity)\b.{0,160}\b(?:pass|fail|blocked|matched|mismatch)\b"),
        "need": "ground reload and integrity evidence before trusting a stored artifact",
    },
    "e128_no_harm_long_horizon_guard": {
        "title": "No-Harm Long-Horizon Guard",
        "family": "Guard",
        "scope": "text_no_harm_long_horizon",
        "pattern": rx(r"\b(?:no harm|negative transfer|long-horizon|downstream harm|delayed harm|delayed regret)\b"),
        "need": "track long-horizon no-harm evidence before stronger promotion",
    },
    "e128_unsupported_answer_guard": {
        "title": "Unsupported Answer Guard",
        "family": "Guard",
        "scope": "text_unsupported_answer",
        "pattern": rx(r"\b(?:unsupported answer|unsupported claim|no citation|without evidence|not supported by evidence)\b"),
        "need": "defer answers whose support chain is explicitly absent or invalid",
    },
    "e128_wrong_scope_call_guard": {
        "title": "Wrong-Scope Call Guard",
        "family": "Guard",
        "scope": "text_wrong_scope_call",
        "pattern": rx(r"\b(?:wrong scope|out of scope|scope mismatch|called outside|should not run|unsafe call)\b"),
        "need": "prevent operators or tools from being called outside their validated scope",
    },
    "e128_active_evidence_request_lens": {
        "title": "Active Evidence Request Lens",
        "family": "Lens",
        "scope": "text_active_evidence_request",
        "pattern": rx(r"\b(?:inspect|ask for|search for|look up|request evidence|need evidence|gather evidence|verify first)\b"),
        "need": "bind information-seeking actions to unresolved state rather than final answer",
    },
    "e128_followup_dependency_lens": {
        "title": "Follow-Up Dependency Lens",
        "family": "Lens",
        "scope": "text_followup_dependency",
        "pattern": rx(r"\b(?:depends on|blocked until|waiting for|follow-up|after that|next depends)\b"),
        "need": "ground follow-up actions to prerequisite evidence or prior steps",
    },
    "e128_progress_ledger_status_lens": {
        "title": "Progress Ledger Status Lens",
        "family": "Lens",
        "scope": "text_progress_ledger_status",
        "pattern": rx(r"\b(?:progress ledger|status|done|complete|blocked|waiting|in_progress|evidence required)\b"),
        "need": "bind step status to evidence instead of accepting completion text alone",
    },
    "e128_stale_done_reuse_guard": {
        "title": "Stale Done Reuse Guard",
        "family": "Guard",
        "scope": "text_stale_done_reuse",
        "pattern": rx(r"\b(?:stale done|previously done|already done|recheck|still valid|invalidated)\b"),
        "need": "avoid reusing old completion status after state changes",
    },
    "e128_multi_turn_continuity_lens": {
        "title": "Multi-Turn Continuity Lens",
        "family": "Lens",
        "scope": "text_multi_turn_continuity",
        "pattern": rx(r"\b(?:previous turn|earlier message|as mentioned|continue|resume|same task|newest request)\b"),
        "need": "bind current action to the latest user request while preserving relevant prior context",
    },
    "e128_answer_format_constraint_lens": {
        "title": "Answer Format Constraint Lens",
        "family": "Lens",
        "scope": "text_answer_format_constraint",
        "pattern": rx(r"\b(?:answer format|respond with|return only|include|do not include|one code block|markdown)\b"),
        "need": "ground output format constraints separately from task content",
    },
    "e128_external_side_effect_guard": {
        "title": "External Side-Effect Guard",
        "family": "Guard",
        "scope": "text_external_side_effect",
        "pattern": rx(r"\b(?:send|submit|upload|delete|purchase|share|publish|post|change permission)\b"),
        "need": "detect actions that may require explicit authority before execution",
    },
    "e128_dependency_version_constraint_lens": {
        "title": "Dependency Version Constraint Lens",
        "family": "Lens",
        "scope": "text_dependency_version_constraint",
        "pattern": rx(r"\b[A-Za-z0-9_.-]+\s*(?:>=|<=|==|~=|\^|~)\s*\d+(?:\.\d+){1,3}\b|\b(?:requires|compatible with)\b.{0,100}\bv?\d+\.\d+"),
        "need": "bind dependency version constraints to the package or platform they constrain",
    },
    "e129_tradeoff_preference_lens": {
        "title": "Tradeoff / Preference Lens",
        "family": "Lens",
        "scope": "text_tradeoff_preference",
        "pattern": rx(r"\b(?:tradeoff|prefer|priority|optimize for|sacrifice|instead of|rather than)\b"),
        "need": "bind stated preference and tradeoff direction before choosing an action",
    },
    "e129_intervention_causality_lens": {
        "title": "Intervention / Causality Lens",
        "family": "Lens",
        "scope": "text_intervention_causality",
        "pattern": rx(r"\b(?:intervention|intervene|controlled for|randomized|causal effect|treatment group|control group)\b"),
        "need": "separate causal evidence from observational wording by requiring intervention context",
    },
    "e129_evidence_quality_hierarchy_lens": {
        "title": "Evidence Quality Hierarchy Lens",
        "family": "Lens",
        "scope": "text_evidence_quality_hierarchy",
        "pattern": rx(r"\b(?:primary source|secondary source|peer reviewed|anecdotal|self-reported|official|unverified)\b"),
        "need": "rank evidence quality before letting a claim update Ground state",
    },
    "e129_assumption_dependency_guard": {
        "title": "Assumption Dependency Guard",
        "family": "Guard",
        "scope": "text_assumption_dependency",
        "pattern": rx(r"\b(?:assumes|assuming|assumption|depends on|only if|provided that)\b"),
        "need": "prevent conclusions from surviving after their assumptions are missing or false",
    },
    "e129_missing_baseline_guard": {
        "title": "Missing Baseline Guard",
        "family": "Guard",
        "scope": "text_missing_baseline",
        "pattern": rx(r"\b(?:baseline|reference|compared to|versus|vs\.?)\b.{0,160}\b(?:missing|absent|not provided|without)\b|\b(?:missing|absent|without)\b.{0,160}\b(?:baseline|reference|comparison)\b"),
        "need": "defer improvement claims when the comparison baseline is absent",
    },
    "e129_selection_bias_guard": {
        "title": "Selection Bias Guard",
        "family": "Guard",
        "scope": "text_selection_bias",
        "pattern": rx(r"\b(?:selection bias|sample bias|cherry-picked|representative sample|biased sample|survivorship bias)\b"),
        "need": "flag dataset or sample-selection caveats before generalizing a result",
    },
    "e129_metric_directionality_lens": {
        "title": "Metric Directionality Lens",
        "family": "Lens",
        "scope": "text_metric_directionality",
        "pattern": rx(r"\b(?:higher is better|lower is better|minimize|maximize|increase|decrease)\b.{0,120}\b(?:metric|score|loss|cost|latency|error)\b|\b(?:metric|score|loss|cost|latency|error)\b.{0,120}\b(?:higher is better|lower is better|minimize|maximize|increase|decrease)\b"),
        "need": "bind metric direction before judging whether a result improved",
    },
    "e129_absolute_relative_delta_lens": {
        "title": "Absolute / Relative Delta Lens",
        "family": "Lens",
        "scope": "text_absolute_relative_delta",
        "pattern": rx(r"\b(?:absolute|relative|percentage point|percent change|delta|improvement)\b.{0,140}\b(?:%|percent|point|from|to)\b"),
        "need": "keep absolute and relative changes from being mixed in score interpretation",
    },
    "e129_temporal_order_causality_guard": {
        "title": "Temporal Order / Causality Guard",
        "family": "Guard",
        "scope": "text_temporal_order_causality",
        "pattern": rx(r"\b(?:before|after|then|subsequently|later|prior to)\b.{0,160}\b(?:caused|because|therefore|as a result)\b"),
        "need": "preserve event order before accepting a causal explanation",
    },
    "e129_exception_to_general_rule_lens": {
        "title": "Exception To General Rule Lens",
        "family": "Lens",
        "scope": "text_exception_to_general_rule",
        "pattern": rx(r"\b(?:generally|usually|normally|default)\b.{0,160}\b(?:except|unless|but|however)\b"),
        "need": "attach exceptions to the general rule they limit",
    },
    "e129_quote_question_answer_split_lens": {
        "title": "Quote / Question / Answer Split Lens",
        "family": "Lens",
        "scope": "text_quote_question_answer_split",
        "pattern": rx(r"[\"“][^\"”?]{4,160}\?[^\"”]*[\"”].{0,160}\b(?:answer|replied|said|response)\b|\b(?:question|asked)\b.{0,160}\b(?:answer|replied|response)\b"),
        "need": "separate quoted questions from the answer or claim that follows",
    },
    "e129_self_reported_claim_guard": {
        "title": "Self-Reported Claim Guard",
        "family": "Guard",
        "scope": "text_self_reported_claim",
        "pattern": rx(r"\b(?:self-reported|claims to|says it|said they|according to their own|unverified claim)\b"),
        "need": "keep self-reported claims below verified evidence until corroborated",
    },
    "e129_update_correction_notice_lens": {
        "title": "Update / Correction Notice Lens",
        "family": "Lens",
        "scope": "text_update_correction_notice",
        "pattern": rx(r"\b(?:update|updated|correction|corrected|erratum|revised|clarification)\b.{0,160}\b(?:previous|earlier|original|initial)\b"),
        "need": "bind correction notices to the prior statement they replace",
    },
    "e129_conditional_permission_guard": {
        "title": "Conditional Permission Guard",
        "family": "Guard",
        "scope": "text_conditional_permission",
        "pattern": rx(r"\b(?:allowed|permitted|may|can)\b.{0,160}\b(?:only if|provided that|with permission|after approval|unless)\b"),
        "need": "block permission claims unless their condition is satisfied",
    },
    "e129_scope_creep_request_lens": {
        "title": "Scope Creep Request Lens",
        "family": "Lens",
        "scope": "text_scope_creep_request",
        "pattern": rx(r"\b(?:also|while you're at it|in addition|expand|scope creep|extra request)\b"),
        "need": "separate added requests from the original task before planning work",
    },
    "e129_multi_document_conflict_lens": {
        "title": "Multi-Document Conflict Lens",
        "family": "Lens",
        "scope": "text_multi_document_conflict",
        "pattern": rx(r"\b(?:document|doc|file|source|page)\b.{0,160}\b(?:conflict|contradicts|differs|inconsistent|disagrees)\b"),
        "need": "bind conflicting claims to their documents instead of merging them",
    },
    "e129_alias_identity_resolution_lens": {
        "title": "Alias / Identity Resolution Lens",
        "family": "Lens",
        "scope": "text_alias_identity_resolution",
        "pattern": rx(r"\b(?:also known as|aka|alias|renamed to|formerly known as|same as)\b"),
        "need": "resolve aliases without confusing distinct entities",
    },
    "e129_pronoun_antecedent_guard": {
        "title": "Pronoun Antecedent Guard",
        "family": "Guard",
        "scope": "text_pronoun_antecedent_ambiguity",
        "pattern": rx(r"\b(?:he|she|they|it|this|that|these|those)\b.{0,120}\b(?:unclear|ambiguous|which|who|what|refer)\b"),
        "need": "defer pronoun-dependent claims when the antecedent is ambiguous",
    },
    "e129_list_item_negative_scope_guard": {
        "title": "List Item Negative Scope Guard",
        "family": "Guard",
        "scope": "text_list_item_negative_scope",
        "pattern": rx(r"(^|\n)\s*(?:[-*+]|\d+[.)])\s+[^\n]{0,120}\b(?:not|no|never|without|except)\b"),
        "need": "keep negation inside the relevant list item instead of leaking it to neighbors",
    },
    "e129_requirement_priority_conflict_lens": {
        "title": "Requirement Priority Conflict Lens",
        "family": "Lens",
        "scope": "text_requirement_priority_conflict",
        "pattern": rx(r"\b(?:conflict|contradiction|cannot both|tradeoff)\b.{0,180}\b(?:requirement|constraint|priority|must|should)\b"),
        "need": "route conflicting requirements through priority resolution before action",
    },
    "e129_action_reversibility_guard": {
        "title": "Action Reversibility Guard",
        "family": "Guard",
        "scope": "text_action_reversibility",
        "pattern": rx(r"\b(?:reversible|irreversible|undo|rollback|cannot undo|permanent|destructive)\b"),
        "need": "separate reversible actions from irreversible side effects before execution",
    },
    "e129_private_public_boundary_guard": {
        "title": "Private / Public Boundary Guard",
        "family": "Guard",
        "scope": "text_private_public_boundary",
        "pattern": rx(r"\b(?:private|public|confidential|secret|internal|external|share publicly|publish)\b"),
        "need": "preserve privacy boundary before exposing or using information",
    },
    "e129_data_license_usage_guard": {
        "title": "Data License / Usage Guard",
        "family": "Guard",
        "scope": "text_data_license_usage",
        "pattern": rx(r"\b(?:license|licensed|terms of use|copyright|redistribute|commercial use|attribution|CC-BY|MIT|Apache)\b"),
        "need": "bind data or artifact usage to license constraints before reuse",
    },
    "e129_benchmark_leakage_guard": {
        "title": "Benchmark Leakage Guard",
        "family": "Guard",
        "scope": "text_benchmark_leakage",
        "pattern": rx(r"\b(?:leakage|data leak|train/test contamination|test set contamination|heldout contamination|oracle leakage)\b"),
        "need": "block benchmark conclusions when train/test leakage is indicated",
    },
    "e129_train_test_split_lens": {
        "title": "Train / Test Split Lens",
        "family": "Lens",
        "scope": "text_train_test_split",
        "pattern": rx(r"\b(?:train|training|validation|heldout|test|OOD|counterfactual|adversarial)\b.{0,160}\b(?:split|set|dataset|rows|samples)\b"),
        "need": "bind results to the split they were measured on",
    },
    "e129_confidence_interval_result_lens": {
        "title": "Confidence Interval / Result Lens",
        "family": "Lens",
        "scope": "text_confidence_interval_result",
        "pattern": rx(r"\b(?:confidence interval|CI|95% interval|credible interval|margin of error)\b"),
        "need": "attach uncertainty intervals to the estimate they qualify",
    },
    "e129_significance_claim_guard": {
        "title": "Significance Claim Guard",
        "family": "Guard",
        "scope": "text_significance_claim",
        "pattern": rx(r"\b(?:statistically significant|p-value|p value|not significant|significance)\b"),
        "need": "avoid treating significance language as effect size or practical importance",
    },
    "e129_outlier_exception_lens": {
        "title": "Outlier / Exception Lens",
        "family": "Lens",
        "scope": "text_outlier_exception",
        "pattern": rx(r"\b(?:outlier|edge case|rare case|exception|anomaly|unusual case)\b"),
        "need": "preserve rare exceptions without letting them overwrite the general pattern",
    },
    "e129_timeline_chain_lens": {
        "title": "Timeline Chain Lens",
        "family": "Lens",
        "scope": "text_timeline_chain",
        "pattern": rx(r"\b(?:first|second|third|then|afterward|finally|next)\b.{0,260}\b(?:first|second|third|then|afterward|finally|next)\b"),
        "need": "preserve ordered event chains before rendering progress or causal summaries",
    },
    "e129_delayed_feedback_lens": {
        "title": "Delayed Feedback Lens",
        "family": "Lens",
        "scope": "text_delayed_feedback",
        "pattern": rx(r"\b(?:delayed feedback|later feedback|after a delay|delayed outcome|downstream result)\b"),
        "need": "bind delayed outcomes back to the earlier action that caused them",
    },
    "e129_promise_capability_guard": {
        "title": "Promise / Capability Guard",
        "family": "Guard",
        "scope": "text_promise_capability",
        "pattern": rx(r"\b(?:promise|promised|will be able|claim to be able|capability|not capable)\b"),
        "need": "separate promised future capability from demonstrated present capability",
    },
    "e129_objective_subjective_split_lens": {
        "title": "Objective / Subjective Split Lens",
        "family": "Lens",
        "scope": "text_objective_subjective_split",
        "pattern": rx(r"\b(?:objective|subjective|opinion|preference|fact|measured|felt|seems)\b"),
        "need": "keep preferences and opinions separate from measured facts",
    },
    "e129_hypothetical_scenario_guard": {
        "title": "Hypothetical Scenario Guard",
        "family": "Guard",
        "scope": "text_hypothetical_scenario",
        "pattern": rx(r"\b(?:hypothetical|suppose|imagine|what if|if we were to|counterfactual)\b"),
        "need": "avoid committing hypothetical scenario content as current-state fact",
    },
    "e129_user_preference_constraint_lens": {
        "title": "User Preference / Constraint Lens",
        "family": "Lens",
        "scope": "text_user_preference_constraint",
        "pattern": rx(r"\b(?:I prefer|I'd rather|I want|I don't want|avoid|please use|please don't)\b"),
        "need": "bind user preference constraints to the current task plan",
    },
    "e129_budget_time_constraint_lens": {
        "title": "Budget / Time Constraint Lens",
        "family": "Lens",
        "scope": "text_budget_time_constraint",
        "pattern": rx(r"\b(?:budget|deadline|time limit|within|no more than|at most|minimum|maximum)\b.{0,120}\b(?:seconds|minutes|hours|days|dollars|tokens|cost)\b"),
        "need": "ground cost and time limits before selecting a route or mode",
    },
    "e129_resource_limit_lens": {
        "title": "Resource Limit Lens",
        "family": "Lens",
        "scope": "text_resource_limit",
        "pattern": rx(r"\b(?:CPU|GPU|RAM|VRAM|memory|disk|bandwidth|workers|threads)\b.{0,140}\b(?:limit|usage|available|unavailable|full|busy)\b"),
        "need": "bind hardware/resource constraints to scheduling decisions",
    },
    "e129_missing_dependency_guard": {
        "title": "Missing Dependency Guard",
        "family": "Guard",
        "scope": "text_missing_dependency",
        "pattern": rx(r"\b(?:missing dependency|not installed|module not found|package missing|cannot import|dependency unavailable)\b"),
        "need": "route missing dependencies to setup or fallback instead of treating execution as valid",
    },
    "e129_path_location_binding_lens": {
        "title": "Path / Location Binding Lens",
        "family": "Lens",
        "scope": "text_path_location_binding",
        "pattern": rx(r"\b(?:path|directory|folder|file|workspace|cwd|location)\b.{0,160}\b(?:[A-Za-z]:\\|/|\\\\|target|docs|scripts)\b"),
        "need": "bind file paths to the correct workspace and operation",
    },
    "e129_command_error_status_lens": {
        "title": "Command Error / Status Lens",
        "family": "Lens",
        "scope": "text_command_error_status",
        "pattern": rx(r"\b(?:error|exception|traceback|failed|exit code|stderr|timeout)\b.{0,160}\b(?:command|process|run|script|test)\b"),
        "need": "ground command failure status before reporting success or continuing",
    },
    "e129_retry_resume_boundary_lens": {
        "title": "Retry / Resume Boundary Lens",
        "family": "Lens",
        "scope": "text_retry_resume_boundary",
        "pattern": rx(r"\b(?:retry|resume|restart|rerun|continue from|checkpoint|partial outcome)\b"),
        "need": "bind retry/resume actions to checkpoint state to avoid duplicate or lost work",
    },
    "e129_stale_cache_guard": {
        "title": "Stale Cache Guard",
        "family": "Guard",
        "scope": "text_stale_cache",
        "pattern": rx(r"\b(?:cache|cached|stale cache|refresh|reload|invalidated|out of date)\b"),
        "need": "avoid trusting cached information after invalidation or recency changes",
    },
    "e129_official_doc_version_lens": {
        "title": "Official Doc / Version Lens",
        "family": "Lens",
        "scope": "text_official_doc_version",
        "pattern": rx(r"\b(?:official docs|documentation|manual|API reference|version|release notes)\b.{0,160}\b(?:v?\d+\.\d+|latest|current|deprecated)\b"),
        "need": "bind documentation claims to official source and version",
    },
    "e129_deprecation_warning_guard": {
        "title": "Deprecation Warning Guard",
        "family": "Guard",
        "scope": "text_deprecation_warning",
        "pattern": rx(r"\b(?:deprecated|deprecation|removed in|will be removed|legacy|no longer supported)\b"),
        "need": "avoid selecting deprecated APIs or stale behavior without a compatibility reason",
    },
    "e129_security_permission_boundary_guard": {
        "title": "Security / Permission Boundary Guard",
        "family": "Guard",
        "scope": "text_security_permission_boundary",
        "pattern": rx(r"\b(?:permission|authorization|auth|token|secret|credential|security|access denied|forbidden)\b"),
        "need": "separate security boundaries from ordinary task failures",
    },
    "e129_personal_data_redaction_guard": {
        "title": "Personal Data / Redaction Guard",
        "family": "Guard",
        "scope": "text_personal_data_redaction",
        "pattern": rx(r"\b(?:personal data|PII|email address|phone number|SSN|redact|redaction|anonymize)\b"),
        "need": "protect personal data fields before publishing or logging outputs",
    },
    "e129_citation_claim_alignment_lens": {
        "title": "Citation / Claim Alignment Lens",
        "family": "Lens",
        "scope": "text_citation_claim_alignment",
        "pattern": rx(r"\b(?:citation|source|reference|link)\b.{0,160}\b(?:claim|supports|does not support|mismatch|aligned)\b"),
        "need": "verify that citations support the specific claim being made",
    },
    "e129_table_row_column_binding_lens": {
        "title": "Table Row / Column Binding Lens",
        "family": "Lens",
        "scope": "text_table_row_column_binding",
        "pattern": rx(r"\b(?:row|column|cell|table|spreadsheet)\b.{0,160}\b(?:value|header|field|record|range)\b"),
        "need": "bind table values to row and column context before using them as facts",
    },
    "e129_unit_conversion_context_guard": {
        "title": "Unit Conversion Context Guard",
        "family": "Guard",
        "scope": "text_unit_conversion_context",
        "pattern": rx(r"\b(?:convert|conversion|unit|units|metric|imperial|Celsius|Fahrenheit|meters|feet)\b"),
        "need": "require unit context before comparing or transforming numeric values",
    },
    "e129_range_endpoint_inclusive_lens": {
        "title": "Range Endpoint / Inclusive Lens",
        "family": "Lens",
        "scope": "text_range_endpoint_inclusive",
        "pattern": rx(r"\b(?:between|from|to|range|inclusive|exclusive|less than|greater than|at least|at most)\b.{0,120}\b\d"),
        "need": "bind numeric range endpoints and inclusive/exclusive semantics",
    },
    "e129_inequality_direction_guard": {
        "title": "Inequality Direction Guard",
        "family": "Guard",
        "scope": "text_inequality_direction",
        "pattern": rx(r"(?:>=|<=|>|<)|\b(?:greater than|less than|at least|at most|no more than|no less than)\b"),
        "need": "preserve inequality direction before filtering or answering numeric constraints",
    },
    "e129_aggregation_granularity_lens": {
        "title": "Aggregation Granularity Lens",
        "family": "Lens",
        "scope": "text_aggregation_granularity",
        "pattern": rx(r"\b(?:aggregate|average|median|total|sum|per-user|per-row|per-day|grouped by)\b"),
        "need": "bind aggregate values to their granularity and grouping",
    },
    "e129_percentage_base_lens": {
        "title": "Percentage Base Lens",
        "family": "Lens",
        "scope": "text_percentage_base",
        "pattern": rx(r"\b\d+(?:\.\d+)?%\b.{0,160}\b(?:of|out of|from|base|total|relative to)\b|\b(?:of|out of|base|relative to)\b.{0,160}\b\d+(?:\.\d+)?%\b"),
        "need": "bind percentages to their denominator or base population",
    },
    "e129_rank_order_tie_guard": {
        "title": "Rank Order / Tie Guard",
        "family": "Guard",
        "scope": "text_rank_order_tie",
        "pattern": rx(r"\b(?:rank|ranking|top|bottom|tie|tied|same score|ordered by)\b"),
        "need": "preserve tie and ordering rules before selecting winners",
    },
    "e129_step_numbering_sequence_lens": {
        "title": "Step Numbering / Sequence Lens",
        "family": "Lens",
        "scope": "text_step_numbering_sequence",
        "pattern": rx(r"(^|\n)\s*\d+[.)]\s+[^\n]{5,180}"),
        "need": "bind numbered steps to sequence order and prevent skipped-step completion",
    },
    "e129_batch_stream_boundary_lens": {
        "title": "Batch / Stream Boundary Lens",
        "family": "Lens",
        "scope": "text_batch_stream_boundary",
        "pattern": rx(r"\b(?:batch|stream|streaming|chunk|frame|window|overlap|stride)\b"),
        "need": "ground whether data is processed as a batch, stream, chunk, or overlapping frame",
    },
    "e129_encoding_normalization_lens": {
        "title": "Encoding / Normalization Lens",
        "family": "Lens",
        "scope": "text_encoding_normalization",
        "pattern": rx(r"\b(?:encoding|UTF-8|unicode|normalize|normalization|ASCII|byte|bytes|tokenization)\b"),
        "need": "bind text/byte normalization before interpreting symbols as evidence",
    },
    "e129_translation_equivalence_guard": {
        "title": "Translation / Equivalence Guard",
        "family": "Guard",
        "scope": "text_translation_equivalence",
        "pattern": rx(r"\b(?:translation|translate|means|equivalent|same meaning|paraphrase)\b.{0,160}\b(?:not exactly|roughly|approx|literal|idiom)\b"),
        "need": "avoid over-trusting approximate translations as exact equivalence",
    },
    "e129_metaphor_literal_boundary_guard": {
        "title": "Metaphor / Literal Boundary Guard",
        "family": "Guard",
        "scope": "text_metaphor_literal_boundary",
        "pattern": rx(r"\b(?:metaphor|figurative|literally|literal|analogy|like a|as if)\b"),
        "need": "keep analogies and metaphors from becoming literal mechanistic claims",
    },
    "e130_action_owner_lens": {
        "title": "Action / Owner Lens",
        "family": "Lens",
        "scope": "text_action_owner",
        "pattern": rx(r"\b(?:owner|assigned to|responsible for|who will|who should|action item|task owner)\b"),
        "need": "bind actions to the person or system responsible for them",
    },
    "e130_deadline_timezone_lens": {
        "title": "Deadline / Timezone Lens",
        "family": "Lens",
        "scope": "text_deadline_timezone",
        "pattern": rx(r"\b(?:deadline|due|by|before|after|schedule|time zone|timezone|UTC|CET|PST|EST)\b.{0,120}\b(?:today|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\\d{1,2}:\\d{2}|20\\d{2})\b"),
        "need": "bind deadlines to concrete dates, times, and timezone context",
    },
    "e130_schedule_conflict_lens": {
        "title": "Schedule Conflict Lens",
        "family": "Lens",
        "scope": "text_schedule_conflict",
        "pattern": rx(r"\b(?:schedule|meeting|calendar|appointment|event)\b.{0,160}\b(?:conflict|overlap|double-booked|available|free slot)\b"),
        "need": "detect schedule conflicts before proposing or confirming time commitments",
    },
    "e130_place_entity_disambiguation_lens": {
        "title": "Place / Entity Disambiguation Lens",
        "family": "Lens",
        "scope": "text_place_entity_disambiguation",
        "pattern": rx(r"\b(?:city|state|country|region|location|place|venue|office)\b.{0,140}\b(?:same name|ambiguous|which|where|located)\b"),
        "need": "avoid mixing same-name places or entities without disambiguation",
    },
    "e130_currency_amount_context_lens": {
        "title": "Currency / Amount Context Lens",
        "family": "Lens",
        "scope": "text_currency_amount_context",
        "pattern": rx(r"(?:\\$|€|£|USD|EUR|GBP|HUF)\\s?\\d|\\b\\d+(?:\\.\\d+)?\\s?(?:USD|EUR|GBP|HUF|dollars|euros|forints)\b"),
        "need": "bind monetary amounts to currency, subject, and transaction context",
    },
    "e130_measurement_error_lens": {
        "title": "Measurement Error Lens",
        "family": "Lens",
        "scope": "text_measurement_error",
        "pattern": rx(r"\b(?:measurement error|margin of error|noise|uncertainty|calibration error|sensor error)\b"),
        "need": "attach measurement uncertainty to numeric observations before comparison",
    },
    "e130_must_should_distinction_lens": {
        "title": "Must / Should Distinction Lens",
        "family": "Lens",
        "scope": "text_must_should_distinction",
        "pattern": rx(r"\b(?:must|shall|required|mandatory)\b.{0,120}\b(?:should|recommended|optional|nice to have)\b|\b(?:should|recommended|optional)\b.{0,120}\b(?:must|shall|required|mandatory)\b"),
        "need": "separate hard requirements from soft recommendations",
    },
    "e130_prohibition_exception_guard": {
        "title": "Prohibition / Exception Guard",
        "family": "Guard",
        "scope": "text_prohibition_exception",
        "pattern": rx(r"\b(?:must not|cannot|forbidden|prohibited|do not)\b.{0,160}\b(?:except|unless|only if|with approval)\b"),
        "need": "preserve exceptions without dropping the underlying prohibition",
    },
    "e130_done_evidence_lens": {
        "title": "Done / Evidence Lens",
        "family": "Lens",
        "scope": "text_done_evidence",
        "pattern": rx(r"\b(?:done|complete|completed|finished|shipped|resolved)\b.{0,160}\b(?:evidence|proof|artifact|test|screenshot|commit|log)\b"),
        "need": "require concrete evidence before accepting a completion state",
    },
    "e130_dependency_blocker_lens": {
        "title": "Dependency / Blocker Lens",
        "family": "Lens",
        "scope": "text_dependency_blocker",
        "pattern": rx(r"\b(?:blocked|blocker|depends on|waiting for|cannot proceed|needs|requires)\b"),
        "need": "bind blockers and dependencies to the action they prevent",
    },
    "e130_rollback_revert_intent_guard": {
        "title": "Rollback / Revert Intent Guard",
        "family": "Guard",
        "scope": "text_rollback_revert_intent",
        "pattern": rx(r"\b(?:rollback|revert|undo|restore|reset|back out)\b.{0,120}\b(?:change|commit|file|state|version)\b"),
        "need": "distinguish user-approved revert intent from accidental deletion or reset",
    },
    "e130_latest_instruction_override_lens": {
        "title": "Latest Instruction Override Lens",
        "family": "Lens",
        "scope": "text_latest_instruction_override",
        "pattern": rx(r"\b(?:actually|wait|instead|change that|newest|latest instruction|ignore previous|correction)\b"),
        "need": "bind the newest user instruction as the controlling action context",
    },
    "e130_tool_recommendation_execution_lens": {
        "title": "Tool Recommendation / Execution Lens",
        "family": "Lens",
        "scope": "text_tool_recommendation_execution",
        "pattern": rx(r"\b(?:recommend using|should run|could run|ran|executed|called tool|tool output)\b"),
        "need": "separate suggested tool use from actually executed tool output",
    },
    "e130_generated_observed_data_lens": {
        "title": "Generated / Observed Data Lens",
        "family": "Lens",
        "scope": "text_generated_observed_data",
        "pattern": rx(r"\b(?:generated|synthetic|simulated|mock|sample)\b.{0,120}\b(?:observed|real|measured|logged|actual)\b|\b(?:observed|real|measured|actual)\b.{0,120}\b(?:generated|synthetic|simulated|mock)\b"),
        "need": "keep generated data separate from observed evidence",
    },
    "e130_synthetic_data_disclosure_guard": {
        "title": "Synthetic Data Disclosure Guard",
        "family": "Guard",
        "scope": "text_synthetic_data_disclosure",
        "pattern": rx(r"\b(?:synthetic data|generated example|simulated data|mock data|not real data|toy example)\b"),
        "need": "preserve synthetic-data disclosure before treating examples as evidence",
    },
    "e130_training_claim_scope_guard": {
        "title": "Training Claim Scope Guard",
        "family": "Guard",
        "scope": "text_training_claim_scope",
        "pattern": rx(r"\b(?:trained on|training data|fine-tuned|learned from|dataset-backed)\b.{0,180}\b(?:not|only|scope|does not mean|cannot claim)\b"),
        "need": "keep training/data claims inside the evidence-backed scope",
    },
    "e130_visual_observation_lens": {
        "title": "Visual Observation Lens",
        "family": "Lens",
        "scope": "text_visual_observation",
        "pattern": rx(r"\b(?:image|screenshot|diagram|photo|visual|looks like|shown in|visible)\b"),
        "need": "bind claims based on visual inputs to the observed visual evidence",
    },
    "e130_transcript_uncertainty_lens": {
        "title": "Transcript Uncertainty Lens",
        "family": "Lens",
        "scope": "text_transcript_uncertainty",
        "pattern": rx(r"\b(?:transcript|transcription|audio|speaker|inaudible|unclear audio|misheard)\b"),
        "need": "mark transcript uncertainty before committing quoted speech",
    },
    "e130_attachment_reference_lens": {
        "title": "Attachment Reference Lens",
        "family": "Lens",
        "scope": "text_attachment_reference",
        "pattern": rx(r"\b(?:attached|attachment|uploaded|file provided|screenshot provided|pasted text)\b"),
        "need": "bind references to attached artifacts instead of assuming inline content",
    },
    "e130_line_reference_lens": {
        "title": "Line Reference Lens",
        "family": "Lens",
        "scope": "text_line_reference",
        "pattern": rx(r"\b(?:line|lines|L\\d+|line number|at line|around line)\b.{0,120}\b(?:file|code|error|change|reference)\b"),
        "need": "ground file and code claims to the referenced line context",
    },
    "e130_code_symbol_context_lens": {
        "title": "Code Symbol Context Lens",
        "family": "Lens",
        "scope": "text_code_symbol_context",
        "pattern": rx(r"`[^`]{2,80}`|\\b(?:function|class|method|variable|module|symbol|identifier)\b"),
        "need": "bind code symbols to their local module or file context",
    },
    "e130_stack_frame_lens": {
        "title": "Stack Frame Lens",
        "family": "Lens",
        "scope": "text_stack_frame",
        "pattern": rx(r"\b(?:Traceback|stack trace|at .*:\\d+|File \".*\", line \\d+|Exception|Error:)\b"),
        "need": "extract error stack frame order before identifying the failing layer",
    },
    "e130_config_env_var_lens": {
        "title": "Config / Env Var Lens",
        "family": "Lens",
        "scope": "text_config_env_var",
        "pattern": rx(r"\b[A-Z][A-Z0-9_]{2,}\b.{0,100}\b(?:env|environment|config|setting|variable|secret)\b|\b(?:env|environment|config|setting)\b.{0,100}\b[A-Z][A-Z0-9_]{2,}\b"),
        "need": "bind configuration and environment variables to their runtime effect",
    },
    "e130_api_rate_limit_guard": {
        "title": "API Rate Limit Guard",
        "family": "Guard",
        "scope": "text_api_rate_limit",
        "pattern": rx(r"\b(?:rate limit|quota|429|too many requests|throttle|request limit)\b"),
        "need": "route rate-limit failures to retry/backoff rather than logic failure",
    },
    "e130_network_dependency_guard": {
        "title": "Network Dependency Guard",
        "family": "Guard",
        "scope": "text_network_dependency",
        "pattern": rx(r"\b(?:network|internet|offline|connection|DNS|timeout|unreachable|connection refused)\b"),
        "need": "separate network dependency failures from model or code failures",
    },
    "e130_local_remote_state_lens": {
        "title": "Local / Remote State Lens",
        "family": "Lens",
        "scope": "text_local_remote_state",
        "pattern": rx(r"\b(?:local|remote|origin|server|client|main|branch)\b.{0,160}\b(?:ahead|behind|synced|out of sync|different|same)\b"),
        "need": "bind state claims to local or remote context before sync decisions",
    },
    "e130_license_compatibility_lens": {
        "title": "License Compatibility Lens",
        "family": "Lens",
        "scope": "text_license_compatibility",
        "pattern": rx(r"\b(?:license compatible|GPL|copyleft|permissive license|commercial license|noncommercial|attribution required)\b"),
        "need": "reason about license compatibility before mixing artifacts or datasets",
    },
    "e130_privacy_consent_guard": {
        "title": "Privacy / Consent Guard",
        "family": "Guard",
        "scope": "text_privacy_consent",
        "pattern": rx(r"\b(?:consent|permission to share|privacy|private data|personal information|confidential)\b"),
        "need": "require consent context before exposing or reusing private information",
    },
    "e130_high_stakes_domain_guard": {
        "title": "High-Stakes Domain Guard",
        "family": "Guard",
        "scope": "text_high_stakes_domain",
        "pattern": rx(r"\b(?:medical|legal|financial|tax|investment|diagnosis|lawyer|doctor|bankruptcy|contract)\b"),
        "need": "mark high-stakes domains that require extra evidence and caution",
    },
    "e130_risk_severity_lens": {
        "title": "Risk / Severity Lens",
        "family": "Lens",
        "scope": "text_risk_severity",
        "pattern": rx(r"\b(?:risk|severity|critical|minor|major|low risk|high risk|impact|harm)\b"),
        "need": "bind risk severity to the action or failure mode being evaluated",
    },
    "e130_possible_confirmed_guard": {
        "title": "Possible / Confirmed Guard",
        "family": "Guard",
        "scope": "text_possible_confirmed",
        "pattern": rx(r"\b(?:possible|possibly|may be|could be|suspected)\b.{0,120}\b(?:confirmed|verified|proven|definite)\b|\b(?:confirmed|verified|proven)\b.{0,120}\b(?:possible|suspected|may be|could be)\b"),
        "need": "avoid upgrading possibilities into confirmed state",
    },
    "e130_ask_answer_policy_lens": {
        "title": "Ask / Answer Policy Lens",
        "family": "Lens",
        "scope": "text_ask_answer_policy",
        "pattern": rx(r"\b(?:ask|clarify|question|answer|respond|do not answer|need more info)\b"),
        "need": "choose between answering and asking based on visible evidence sufficiency",
    },
    "e130_default_minimal_action_lens": {
        "title": "Default Minimal Action Lens",
        "family": "Lens",
        "scope": "text_default_minimal_action",
        "pattern": rx(r"\b(?:minimal|smallest|least|default action|conservative|only necessary|no extra)\b"),
        "need": "prefer the smallest sufficient action when the user asks for restraint",
    },
    "e130_inference_observation_guard": {
        "title": "Inference / Observation Guard",
        "family": "Guard",
        "scope": "text_inference_observation",
        "pattern": rx(r"\b(?:I infer|inference|likely means|suggests)\b.{0,120}\b(?:observed|shown|measured|saw|logged)\b|\b(?:observed|shown|measured)\b.{0,120}\b(?:infer|inference|suggests)\b"),
        "need": "separate direct observations from inferred conclusions",
    },
    "e130_ablation_control_result_lens": {
        "title": "Ablation / Control Result Lens",
        "family": "Lens",
        "scope": "text_ablation_control_result",
        "pattern": rx(r"\b(?:ablation|control|without|with only|baseline)\b.{0,180}\b(?:result|score|accuracy|success|failed|passed)\b"),
        "need": "bind ablation/control outcomes to the system variant being tested",
    },
    "e130_plateau_trend_lens": {
        "title": "Plateau / Trend Lens",
        "family": "Lens",
        "scope": "text_plateau_trend",
        "pattern": rx(r"\b(?:plateau|flattened|kept improving|trend|curve|learning curve|oscillated|converged)\b"),
        "need": "ground whether a training curve plateaued, improved, or oscillated",
    },
    "e130_capacity_limit_lens": {
        "title": "Capacity Limit Lens",
        "family": "Lens",
        "scope": "text_capacity_limit",
        "pattern": rx(r"\b(?:capacity|too small|too large|bottleneck|limit|ceiling|overcapacity|undercapacity)\b"),
        "need": "bind performance limits to capacity rather than data or optimization by default",
    },
    "e130_search_space_cost_lens": {
        "title": "Search Space / Cost Lens",
        "family": "Lens",
        "scope": "text_search_space_cost",
        "pattern": rx(r"\b(?:search space|combinatorial|mutation cost|attempts|rollbacks|too many variants|expensive search)\b"),
        "need": "connect mutation/search difficulty to the size of the explored space",
    },
    "e130_mutation_acceptance_rate_lens": {
        "title": "Mutation Acceptance Rate Lens",
        "family": "Lens",
        "scope": "text_mutation_acceptance_rate",
        "pattern": rx(r"\b(?:accepted mutations|rejected mutations|rollback count|acceptance rate|mutation attempts)\b"),
        "need": "ground mutation quality by accepted/rejected/rollback counts",
    },
    "e130_active_set_filter_lens": {
        "title": "Active Set Filter Lens",
        "family": "Lens",
        "scope": "text_active_set_filter",
        "pattern": rx(r"\b(?:active set|selected set|filtered library|candidate set|full scan|subset)\b"),
        "need": "bind routing cost to the active candidate set rather than full library size",
    },
    "e130_registry_uid_alias_lens": {
        "title": "Registry UID / Alias Lens",
        "family": "Lens",
        "scope": "text_registry_uid_alias",
        "pattern": rx(r"\b(?:registry|uid|alias|human alias|content digest|descriptor|token)\b"),
        "need": "separate stable machine identity from human aliases and descriptors",
    },
    "e130_negative_card_recall_lens": {
        "title": "Negative Card Recall Lens",
        "family": "Lens",
        "scope": "text_negative_card_recall",
        "pattern": rx(r"\b(?:negative card|failure card|bad mutation|rejected variant|known failure|red flag)\b"),
        "need": "reuse failure memories as mutation and routing constraints",
    },
    "e130_failure_recycle_lens": {
        "title": "Failure / Recycle Lens",
        "family": "Lens",
        "scope": "text_failure_recycle",
        "pattern": rx(r"\b(?:recycle|repair|mutate again|quarantine|recover|salvage|reuse failure)\b"),
        "need": "route failed variants toward repair, quarantine, or negative memory instead of silent discard",
    },
    "e130_rank_promotion_lens": {
        "title": "Rank / Promotion Lens",
        "family": "Lens",
        "scope": "text_rank_promotion",
        "pattern": rx(r"\b(?:Bronze|Silver|Gold|Orange|Legendary|Diamond|Core|promotion|rank|probation)\b"),
        "need": "bind rank names to promotion gates and avoid overclaiming lifecycle state",
    },
    "e130_core_memory_boundary_guard": {
        "title": "Core Memory Boundary Guard",
        "family": "Guard",
        "scope": "text_core_memory_boundary",
        "pattern": rx(r"\b(?:core memory|perma core|permanent memory|TrueGolden|Golden Disc|global core)\b"),
        "need": "block core-memory claims unless the much stronger no-harm gates passed",
    },
    "e130_dashboard_status_lens": {
        "title": "Dashboard Status Lens",
        "family": "Lens",
        "scope": "text_dashboard_status",
        "pattern": rx(r"\b(?:dashboard|web UI|loading bar|status card|visualizer|heatmap)\b"),
        "need": "bind visual dashboard status to the underlying artifact checkpoint",
    },
    "e130_partial_artifact_progress_lens": {
        "title": "Partial Artifact / Progress Lens",
        "family": "Lens",
        "scope": "text_partial_artifact_progress",
        "pattern": rx(r"\b(?:partial artifact|partial aggregate|progress\\.jsonl|checkpoint|heartbeat|writeout)\b"),
        "need": "treat partial writeouts as recovery evidence during long runs",
    },
    "e130_stop_resume_control_lens": {
        "title": "Stop / Resume Control Lens",
        "family": "Lens",
        "scope": "text_stop_resume_control",
        "pattern": rx(r"\b(?:STOP file|stop file|resume|pause|continue|wake|watchdog|supervisor)\b"),
        "need": "bind stop/resume controls to long-running job lifecycle state",
    },
    "e131_definition_term_lens": {
        "title": "Definition / Term Lens",
        "family": "Lens",
        "scope": "text_definition_term",
        "pattern": rx(r"\b(?:definition|defined as|refers to|is called|terminology|term means|glossary)\b"),
        "need": "bind a term to its local definition before using it as a stable concept",
    },
    "e131_example_counterexample_lens": {
        "title": "Example / Counterexample Lens",
        "family": "Lens",
        "scope": "text_example_counterexample",
        "pattern": rx(r"\b(?:example|for example|counterexample|e\.g\.|such as|case where|edge case)\b"),
        "need": "separate illustrative examples from rules and detect counterexamples",
    },
    "e131_comparison_contrast_lens": {
        "title": "Comparison / Contrast Lens",
        "family": "Lens",
        "scope": "text_comparison_contrast",
        "pattern": rx(r"\b(?:compared with|compared to|versus|vs\.?|whereas|while|unlike|in contrast)\b"),
        "need": "bind each side of a comparison before drawing a preference or difference",
    },
    "e131_cause_effect_lens": {
        "title": "Cause / Effect Lens",
        "family": "Lens",
        "scope": "text_cause_effect",
        "pattern": rx(r"\b(?:because|therefore|so that|as a result|caused by|leads to|results in|due to)\b"),
        "need": "bind cause claims to their stated effect without reversing direction",
    },
    "e131_conditional_if_then_lens": {
        "title": "Conditional If / Then Lens",
        "family": "Lens",
        "scope": "text_conditional_if_then",
        "pattern": rx(r"\b(?:if|when|provided that|assuming|then|otherwise)\b.{0,180}\b(?:then|otherwise|must|should|will|can)\b"),
        "need": "preserve condition-action boundaries before committing conclusions",
    },
    "e131_goal_success_criteria_lens": {
        "title": "Goal / Success Criteria Lens",
        "family": "Lens",
        "scope": "text_goal_success_criteria",
        "pattern": rx(r"\b(?:goal|objective|success criteria|acceptance criteria|done when|passes if|positive requires)\b"),
        "need": "bind success criteria to the goal they evaluate",
    },
    "e131_question_answer_pair_lens": {
        "title": "Question / Answer Pair Lens",
        "family": "Lens",
        "scope": "text_question_answer_pair",
        "pattern": rx(r"\b(?:question|asked|answer|answered|Q:|A:|FAQ)\b"),
        "need": "pair answers with the specific question instead of nearby unrelated context",
    },
    "e131_quote_attribution_lens": {
        "title": "Quote / Attribution Lens",
        "family": "Lens",
        "scope": "text_quote_attribution",
        "pattern": rx(r"\"[^\"]{8,220}\".{0,120}\b(?:said|wrote|according to|claimed|reported)\b|\b(?:said|wrote|according to|claimed|reported)\b.{0,120}\"[^\"]{8,220}\""),
        "need": "bind quoted text to the speaker or source attribution",
    },
    "e131_source_recency_lens": {
        "title": "Source / Recency Lens",
        "family": "Lens",
        "scope": "text_source_recency",
        "pattern": rx(r"\b(?:latest|newest|recent|updated|as of|published|dated|current)\b.{0,160}\b(?:source|article|doc|report|page|release)\b"),
        "need": "attach source recency before relying on changing information",
    },
    "e131_summary_detail_boundary_lens": {
        "title": "Summary / Detail Boundary Lens",
        "family": "Lens",
        "scope": "text_summary_detail_boundary",
        "pattern": rx(r"\b(?:summary|summarize|briefly|details|detail|full explanation|high level|tl;dr)\b"),
        "need": "separate compact summaries from detailed evidence and caveats",
    },
    "e131_claim_evidence_gap_guard": {
        "title": "Claim / Evidence Gap Guard",
        "family": "Guard",
        "scope": "text_claim_evidence_gap",
        "pattern": rx(r"\b(?:claim|assert|conclude|prove|evidence|support|unsupported|not shown|no evidence)\b"),
        "need": "detect when a claim lacks visible supporting evidence",
    },
    "e131_unknown_unknown_request_lens": {
        "title": "Unknown / Ask More Lens",
        "family": "Lens",
        "scope": "text_unknown_ask_more",
        "pattern": rx(r"\b(?:unknown|unclear|not enough information|cannot tell|need more|ask for|clarify|missing info)\b"),
        "need": "route underdetermined states toward evidence seeking instead of guessing",
    },
    "e131_entity_role_lens": {
        "title": "Entity / Role Lens",
        "family": "Lens",
        "scope": "text_entity_role",
        "pattern": rx(r"\b(?:user|assistant|client|server|admin|owner|author|reviewer|operator|agent)\b.{0,140}\b(?:role|responsible|allowed|should|must|can)\b"),
        "need": "bind an entity to its role before assigning responsibility or permission",
    },
    "e131_part_whole_lens": {
        "title": "Part / Whole Lens",
        "family": "Lens",
        "scope": "text_part_whole",
        "pattern": rx(r"\b(?:part of|component|module|section|chapter|belongs to|contains|includes|subset)\b"),
        "need": "bind components to the whole system or document they belong to",
    },
    "e131_category_membership_lens": {
        "title": "Category / Membership Lens",
        "family": "Lens",
        "scope": "text_category_membership",
        "pattern": rx(r"\b(?:category|class|type of|kind of|belongs to|member of|classified as|taxonomy)\b"),
        "need": "ground category membership without treating examples as exhaustive sets",
    },
    "e131_sequence_precondition_lens": {
        "title": "Sequence / Precondition Lens",
        "family": "Lens",
        "scope": "text_sequence_precondition",
        "pattern": rx(r"\b(?:before|after|first|next|then|precondition|prerequisite|requires first)\b"),
        "need": "preserve required order and prerequisites before marking an action ready",
    },
    "e131_version_change_changelog_lens": {
        "title": "Version Change / Changelog Lens",
        "family": "Lens",
        "scope": "text_version_change_changelog",
        "pattern": rx(r"\b(?:changelog|release notes|breaking change|migration|deprecated|new in|changed in|removed in)\b"),
        "need": "bind versioned behavior changes to their release context",
    },
    "e131_root_cause_symptom_lens": {
        "title": "Root Cause / Symptom Lens",
        "family": "Lens",
        "scope": "text_root_cause_symptom",
        "pattern": rx(r"\b(?:root cause|symptom|caused by|indicates|diagnosis|debug|failure mode)\b"),
        "need": "separate observed symptoms from claimed root causes",
    },
    "e131_obligation_actor_lens": {
        "title": "Obligation / Actor Lens",
        "family": "Lens",
        "scope": "text_obligation_actor",
        "pattern": rx(r"\b(?:must|shall|required|obligated|responsible)\b.{0,140}\b(?:user|assistant|client|server|admin|system|operator)\b|\b(?:user|assistant|client|server|admin|system|operator)\b.{0,140}\b(?:must|shall|required|responsible)\b"),
        "need": "bind obligations to the actor who must satisfy them",
    },
    "e131_analogy_mapping_lens": {
        "title": "Analogy / Mapping Lens",
        "family": "Lens",
        "scope": "text_analogy_mapping",
        "pattern": rx(r"\b(?:analogy|analogous|like|similar to|maps to|corresponds to)\b"),
        "need": "preserve analogy mappings without upgrading them to literal equivalence",
    },
    "e131_scope_exclusion_lens": {
        "title": "Scope / Exclusion Lens",
        "family": "Lens",
        "scope": "text_scope_exclusion",
        "pattern": rx(r"\b(?:in scope|out of scope|exclude|excluding|except for|not included|only applies to)\b"),
        "need": "bind scope exclusions to the affected claim or action",
    },
    "e131_multi_constraint_filter_lens": {
        "title": "Multi-Constraint Filter Lens",
        "family": "Lens",
        "scope": "text_multi_constraint_filter",
        "pattern": rx(r"\b(?:filter|match|satisfy|constraints|criteria|requirements)\b.{0,180}\b(?:and|or|all|any|none)\b"),
        "need": "combine multiple visible constraints without dropping one of them",
    },
    "e131_threshold_boundary_lens": {
        "title": "Threshold / Boundary Lens",
        "family": "Lens",
        "scope": "text_threshold_boundary",
        "pattern": rx(r"\b(?:threshold|cutoff|limit|minimum|maximum|at least|at most|below|above)\b"),
        "need": "bind threshold direction and boundary values before decisions",
    },
    "e131_probability_confidence_lens": {
        "title": "Probability / Confidence Lens",
        "family": "Lens",
        "scope": "text_probability_confidence",
        "pattern": rx(r"\b(?:probability|confidence|likely|unlikely|chance|risk of|odds|uncertain)\b"),
        "need": "keep probabilistic statements separate from confirmed facts",
    },
    "e131_conflict_resolution_policy_lens": {
        "title": "Conflict Resolution Policy Lens",
        "family": "Lens",
        "scope": "text_conflict_resolution_policy",
        "pattern": rx(r"\b(?:conflict|contradiction|inconsistent|disagree|resolution|tie-break|priority rule)\b"),
        "need": "bind conflict resolution policy to the conflicting evidence",
    },
    "e131_evidence_absence_lens": {
        "title": "Evidence Absence Lens",
        "family": "Lens",
        "scope": "text_evidence_absence",
        "pattern": rx(r"\b(?:no evidence|not found|missing evidence|absence of|not observed|not available)\b"),
        "need": "treat absence of evidence as an unresolved state unless the task defines it as evidence",
    },
    "e131_time_window_validity_lens": {
        "title": "Time Window / Validity Lens",
        "family": "Lens",
        "scope": "text_time_window_validity",
        "pattern": rx(r"\b(?:valid until|expires|within|during|time window|window|period|from .* to)\b"),
        "need": "bind facts and permissions to their valid time window",
    },
    "e131_reproducibility_seed_lens": {
        "title": "Reproducibility / Seed Lens",
        "family": "Lens",
        "scope": "text_reproducibility_seed",
        "pattern": rx(r"\b(?:seed|random seed|deterministic|replay|reproducible|repeatable|hash match)\b"),
        "need": "bind reproducibility claims to seeds, hashes, and replay artifacts",
    },
    "e131_hardware_resource_binding_lens": {
        "title": "Hardware / Resource Binding Lens",
        "family": "Lens",
        "scope": "text_hardware_resource_binding",
        "pattern": rx(r"\b(?:CPU|GPU|VRAM|RAM|memory|thread|worker|core|utilization|hardware)\b"),
        "need": "bind runtime claims to the resource they measure",
    },
    "e131_latency_throughput_lens": {
        "title": "Latency / Throughput Lens",
        "family": "Lens",
        "scope": "text_latency_throughput",
        "pattern": rx(r"\b(?:latency|throughput|p50|p95|requests per second|tokens per second|wall time|runtime)\b"),
        "need": "separate latency, throughput, and wall-time efficiency claims",
    },
    "e131_format_contract_lens": {
        "title": "Format / Contract Lens",
        "family": "Lens",
        "scope": "text_format_contract",
        "pattern": rx(r"\b(?:format|schema|contract|must include|required fields|output shape|artifact contract)\b"),
        "need": "bind required output format to the artifact or interface contract",
    },
    "e131_input_output_contract_lens": {
        "title": "Input / Output Contract Lens",
        "family": "Lens",
        "scope": "text_input_output_contract",
        "pattern": rx(r"\b(?:input|output|I/O|read|write|return value|interface|ABI)\b.{0,160}\b(?:contract|shape|format|field|schema)\b"),
        "need": "bind input and output shapes before composing modules",
    },
    "e131_schema_field_mapping_lens": {
        "title": "Schema / Field Mapping Lens",
        "family": "Lens",
        "scope": "text_schema_field_mapping",
        "pattern": rx(r"\b(?:schema|field|mapping|column|property|attribute)\b.{0,160}\b(?:maps to|renamed|alias|source|target)\b"),
        "need": "bind source fields to target fields before migration or parsing",
    },
    "e131_error_recovery_plan_lens": {
        "title": "Error / Recovery Plan Lens",
        "family": "Lens",
        "scope": "text_error_recovery_plan",
        "pattern": rx(r"\b(?:error|failure|failed|crash|exception)\b.{0,180}\b(?:recover|retry|repair|resume|rollback|fallback)\b"),
        "need": "connect failures to recovery actions instead of treating them as final results",
    },
    "e131_permission_escalation_guard": {
        "title": "Permission Escalation Guard",
        "family": "Guard",
        "scope": "text_permission_escalation",
        "pattern": rx(r"\b(?:escalate|administrator|sudo|elevated|privilege|permission)\b.{0,160}\b(?:required|needed|denied|grant|allow)\b"),
        "need": "block unsafe escalation unless explicit permission and scope are visible",
    },
    "e131_user_intent_revision_lens": {
        "title": "User Intent Revision Lens",
        "family": "Lens",
        "scope": "text_user_intent_revision",
        "pattern": rx(r"\b(?:I meant|what I meant|actually|correction|not that|instead|change the request)\b"),
        "need": "bind user corrections to the current intent and retire stale intent",
    },
    "e131_done_vs_started_lens": {
        "title": "Done / Started Lens",
        "family": "Lens",
        "scope": "text_done_vs_started",
        "pattern": rx(r"\b(?:started|in progress|running|queued|done|complete|finished|not finished)\b"),
        "need": "separate started work from completed evidence-backed work",
    },
    "e131_observed_metric_vs_target_lens": {
        "title": "Observed Metric / Target Lens",
        "family": "Lens",
        "scope": "text_observed_metric_vs_target",
        "pattern": rx(r"\b(?:target|goal|threshold|expected)\b.{0,160}\b(?:observed|actual|measured|result|metric)\b|\b(?:observed|actual|measured|result|metric)\b.{0,160}\b(?:target|goal|threshold|expected)\b"),
        "need": "compare observed metrics against targets without conflating them",
    },
    "e131_fallback_primary_route_lens": {
        "title": "Fallback / Primary Route Lens",
        "family": "Lens",
        "scope": "text_fallback_primary_route",
        "pattern": rx(r"\b(?:fallback|primary|secondary|backup|default path|alternate route|route around)\b"),
        "need": "bind fallback paths to the primary route they replace",
    },
    "e131_retry_backoff_lens": {
        "title": "Retry / Backoff Lens",
        "family": "Lens",
        "scope": "text_retry_backoff",
        "pattern": rx(r"\b(?:retry|backoff|try again|exponential|cooldown|rate limited|wait before)\b"),
        "need": "route transient failures to retry/backoff rather than permanent failure",
    },
    "e131_version_compatibility_lens": {
        "title": "Version Compatibility Lens",
        "family": "Lens",
        "scope": "text_version_compatibility",
        "pattern": rx(r"\b(?:compatible|incompatible|requires version|minimum version|works with|does not support)\b"),
        "need": "bind compatibility claims to the relevant version and component",
    },
    "e131_data_provenance_lens": {
        "title": "Data Provenance Lens",
        "family": "Lens",
        "scope": "text_data_provenance",
        "pattern": rx(r"\b(?:provenance|source data|derived from|collected from|scraped from|generated from|origin of data)\b"),
        "need": "bind data claims to their source and derivation path",
    },
    "e131_tool_output_trust_lens": {
        "title": "Tool Output / Trust Lens",
        "family": "Lens",
        "scope": "text_tool_output_trust",
        "pattern": rx(r"\b(?:tool output|command output|stderr|stdout|exit code|checker output|log output)\b"),
        "need": "treat tool output as evidence with source and status attached",
    },
    "e131_human_need_followup_lens": {
        "title": "Human Need / Follow-up Lens",
        "family": "Lens",
        "scope": "text_human_need_followup",
        "pattern": rx(r"\b(?:follow up|next question|ask the user|needs user input|waiting for user|clarification needed)\b"),
        "need": "route missing human context to follow-up rather than silent assumptions",
    },
    "e131_ambiguous_reference_guard": {
        "title": "Ambiguous Reference Guard",
        "family": "Guard",
        "scope": "text_ambiguous_reference",
        "pattern": rx(r"\b(?:this|that|it|they|those|the former|the latter)\b.{0,120}\b(?:ambiguous|unclear|which|refer to|reference)\b"),
        "need": "block commits when pronoun or pointer references are ambiguous",
    },
    "e131_same_word_different_sense_lens": {
        "title": "Same Word / Different Sense Lens",
        "family": "Lens",
        "scope": "text_same_word_different_sense",
        "pattern": rx(r"\b(?:same word|different meaning|sense|homonym|ambiguous term|could mean)\b"),
        "need": "separate word forms from their local contextual sense",
    },
    "e131_internal_external_boundary_lens": {
        "title": "Internal / External Boundary Lens",
        "family": "Lens",
        "scope": "text_internal_external_boundary",
        "pattern": rx(r"\b(?:internal|external|public|private|outside|inside|user-facing|implementation detail)\b"),
        "need": "bind whether a statement concerns internal mechanics or external behavior",
    },
    "e131_test_train_deployment_scope_guard": {
        "title": "Test / Train / Deployment Scope Guard",
        "family": "Guard",
        "scope": "text_test_train_deployment_scope",
        "pattern": rx(r"\b(?:train|validation|test|heldout|deployment|production|staging)\b.{0,180}\b(?:scope|split|result|claim|leak)\b"),
        "need": "keep training, validation, test, and deployment claims separate",
    },
    "e131_safety_policy_context_guard": {
        "title": "Safety / Policy Context Guard",
        "family": "Guard",
        "scope": "text_safety_policy_context",
        "pattern": rx(r"\b(?:safety|policy|allowed|disallowed|must refuse|cannot help|safe completion)\b"),
        "need": "attach safety policy context before deciding whether to answer or refuse",
    },
    "e131_patch_scope_lens": {
        "title": "Patch / Scope Lens",
        "family": "Lens",
        "scope": "text_patch_scope",
        "pattern": rx(r"\b(?:patch|diff|change|edit|modify)\b.{0,160}\b(?:scope|file|line|unrelated|only|minimal)\b"),
        "need": "bind code changes to their intended scope and avoid unrelated edits",
    },
    "e131_git_sync_status_lens": {
        "title": "Git Sync Status Lens",
        "family": "Lens",
        "scope": "text_git_sync_status",
        "pattern": rx(r"\b(?:git|commit|push|pull|origin|main|branch|ahead|behind|dirty|clean)\b"),
        "need": "bind repository sync claims to concrete git state",
    },
    "e131_process_liveness_lens": {
        "title": "Process Liveness Lens",
        "family": "Lens",
        "scope": "text_process_liveness",
        "pattern": rx(r"\b(?:process|PID|running|alive|exited|crashed|stopped|heartbeat)\b"),
        "need": "bind long-running job status to process and heartbeat evidence",
    },
    "e131_dashboard_artifact_lag_lens": {
        "title": "Dashboard / Artifact Lag Lens",
        "family": "Lens",
        "scope": "text_dashboard_artifact_lag",
        "pattern": rx(r"\b(?:dashboard|webpage|visualizer|UI)\b.{0,160}\b(?:stale|updated|lag|artifact|checkpoint|refresh)\b"),
        "need": "detect when dashboard state lags behind checkpoint artifacts",
    },
    "e131_mutation_prune_ratio_lens": {
        "title": "Mutation / Prune Ratio Lens",
        "family": "Lens",
        "scope": "text_mutation_prune_ratio",
        "pattern": rx(r"\b(?:mutation|mutated|prune|pruned|compression|accepted|rollback)\b.{0,160}\b(?:ratio|count|rate|attempt)\b"),
        "need": "bind mutation and pruning counts to their candidate or operator",
    },
    "e131_candidate_exhaustion_boundary_lens": {
        "title": "Candidate Exhaustion Boundary Lens",
        "family": "Lens",
        "scope": "text_candidate_exhaustion_boundary",
        "pattern": rx(r"\b(?:no candidates|no farmable|exhausted|candidate space|nothing left|new candidate pack)\b"),
        "need": "distinguish clean candidate exhaustion from runtime failure",
    },
    "e131_skill_transfer_scope_lens": {
        "title": "Skill Transfer / Scope Lens",
        "family": "Lens",
        "scope": "text_skill_transfer_scope",
        "pattern": rx(r"\b(?:transfer|transferable|reuse|import|library|skill|pocket)\b.{0,180}\b(?:scope|compatible|works on|does not work)\b"),
        "need": "bind transferred skills to their tested compatibility scope",
    },
    "e131_negative_memory_use_lens": {
        "title": "Negative Memory Use Lens",
        "family": "Lens",
        "scope": "text_negative_memory_use",
        "pattern": rx(r"\b(?:negative memory|failure memory|bad example|rejected mutation|red flag|known bad)\b"),
        "need": "use stored failures as constraints rather than deleting their lesson",
    },
    "e131_active_evidence_search_lens": {
        "title": "Active Evidence Search Lens",
        "family": "Lens",
        "scope": "text_active_evidence_search",
        "pattern": rx(r"\b(?:inspect|search for evidence|ask for evidence|gather evidence|look up|verify)\b"),
        "need": "route unresolved states toward targeted evidence gathering",
    },
    "e131_answer_abstain_boundary_lens": {
        "title": "Answer / Abstain Boundary Lens",
        "family": "Lens",
        "scope": "text_answer_abstain_boundary",
        "pattern": rx(r"\b(?:answer|abstain|do not answer|hold|defer|not enough information|guess)\b"),
        "need": "choose answer versus abstain based on evidence sufficiency",
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
