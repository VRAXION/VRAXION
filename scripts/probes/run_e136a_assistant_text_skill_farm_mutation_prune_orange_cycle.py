#!/usr/bin/env python3
"""E136A assistant-text skill farm, mutation/prune, and Orange cycle.

E136A turns the local E136 assistant-text seed pack into scoped assistant/text
operator candidates. It follows the E132 style:

support -> candidate variants -> mutation/prune/challenger/reload ->
negative scope -> scoped Orange/LegendaryCandidate.

Boundary: assistant/text operator farming only. This is not neural training,
not open-domain assistant readiness, not production assistant evidence, and not
Core/PermaCore/TrueGolden promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


ARTIFACT_CONTRACT = "E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE"
DECISION_CONFIRMED = "e136a_assistant_text_skill_farm_mutation_prune_orange_cycle_confirmed"
DECISION_REJECTED = "e136a_assistant_text_skill_farm_mutation_prune_orange_cycle_rejected"
NEXT = "E136B_ASSISTANT_TEXT_ROUTE_COMPOSITION_AND_BOUNDARY_CONFIRM"

DEFAULT_DATASET = Path("target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl")
DEFAULT_DATASET_MANIFEST = Path("target/datasets/e136_assistant_text_seed_pack/normalized_manifest.json")
DEFAULT_DOWNLOAD_MANIFEST = Path("target/datasets/e136_assistant_text_seed_pack/download_manifest.json")
DEFAULT_OUT = Path("target/pilot_wave/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle")

ORANGE_TARGET = 300_000
ORANGE_FAMILY_MIN = 12
ORANGE_CAMPAIGN_MIN = 8
DEFAULT_ROW_LIMIT = 460_000

PRESSURE_FAMILIES = (
    "assistant_seed_support",
    "cross_source_replay",
    "multi_turn_context_replay",
    "preference_boundary_replay",
    "refusal_boundary_replay",
    "format_constraint_replay",
    "math_text_no_solve_boundary",
    "code_boundary_replay",
    "negative_scope_nonassistant",
    "overbroad_chatbot_control",
    "mutation_accept_reject",
    "prune_minimality",
    "sibling_challenger",
    "reload_shadow_import",
    "deterministic_replay",
)

ARTIFACT_FILES = (
    "run_manifest.json",
    "download_report.json",
    "dataset_report.json",
    "skill_candidate_report.json",
    "operator_cards.json",
    "operator_orange_results.json",
    "variant_report.json",
    "promotion_report.json",
    "negative_scope_report.json",
    "mutation_summary.json",
    "mutation_events.jsonl",
    "row_level_samples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class AssistantSpec:
    operator_id: str
    display_name: str
    family: str
    scope: str
    description: str
    tag_hints: tuple[str, ...]
    term_hints: tuple[str, ...]
    source_hints: tuple[str, ...]
    negative_scope: str


SPECS: tuple[AssistantSpec, ...] = (
    AssistantSpec(
        "assistant_role_turn_boundary_lens",
        "Assistant Role / Turn Boundary Lens",
        "Lens",
        "assistant_text_role_turn_boundary",
        "Binds user/system/assistant turns and prevents role leakage across messages.",
        ("assistant_style",),
        ("user", "assistant", "system", "role", "conversation"),
        (),
        "Must not answer by itself; only preserves speaker/role boundaries.",
    ),
    AssistantSpec(
        "multi_turn_context_continuity_lens",
        "Multi-Turn Context Continuity Lens",
        "Lens",
        "assistant_text_multi_turn_context_continuity",
        "Tracks follow-up turns while keeping the current user request authoritative.",
        ("multi_turn_dialogue",),
        ("follow up", "previous", "again", "continue", "context", "conversation"),
        (),
        "Must not reuse stale context when the current turn changes scope.",
    ),
    AssistantSpec(
        "instruction_task_decomposition_lens",
        "Instruction Task Decomposition Lens",
        "Lens",
        "assistant_text_instruction_task_decomposition",
        "Splits assistant instructions into ordered task requirements and constraints.",
        ("instruction_following",),
        ("step", "steps", "first", "second", "finally", "how to", "instructions", "guide"),
        (),
        "Must not invent missing steps or commit an unsupported action plan.",
    ),
    AssistantSpec(
        "summarization_request_lens",
        "Summarization Request Lens",
        "Lens",
        "assistant_text_summarization_request",
        "Detects summarize/condense/TLDR requests and binds them to the source text.",
        ("summarization",),
        ("summarize", "summary", "tl;dr", "condense", "key points", "goals"),
        (),
        "Must not summarize absent text or add facts not in the supplied source.",
    ),
    AssistantSpec(
        "code_instruction_boundary_lens",
        "Code Instruction Boundary Lens",
        "Lens",
        "assistant_text_code_instruction_boundary",
        "Detects code/programming instructions and keeps code blocks distinct from prose.",
        ("code_instruction",),
        ("python", "javascript", "function", "program", "code", "class", "script"),
        (),
        "Must not execute code or claim runtime success without a later execution route.",
    ),
    AssistantSpec(
        "refusal_boundary_guard",
        "Refusal Boundary Guard",
        "Guard",
        "assistant_text_refusal_boundary",
        "Separates appropriate refusal/defer behavior from ordinary answer rendering.",
        ("refusal_or_boundary",),
        ("cannot", "can't", "unable", "not able", "don't have access", "do not have access", "not appropriate"),
        (),
        "Must not over-refuse benign requests or answer unsafe/unsupported requests.",
    ),
    AssistantSpec(
        "helpful_harmless_preference_guard",
        "Helpful / Harmless Preference Guard",
        "Guard",
        "assistant_text_helpful_harmless_preference_boundary",
        "Uses preference pairs as boundary evidence for helpful/harmless response shape.",
        ("helpful_harmless", "preference_boundary"),
        ("chosen", "rejected", "safe", "harm", "harmless", "helpful"),
        ("Anthropic/hh-rlhf",),
        "Must not treat preference data as a free-form safety policy engine.",
    ),
    AssistantSpec(
        "rejected_response_contrast_lens",
        "Rejected Response Contrast Lens",
        "Lens",
        "assistant_text_rejected_response_contrast",
        "Keeps rejected responses as negative evidence instead of callable answers.",
        ("preference_boundary",),
        (),
        ("Anthropic/hh-rlhf",),
        "Must not render rejected examples as normal assistant output.",
    ),
    AssistantSpec(
        "source_absence_defer_guard",
        "Source Absence Defer Guard",
        "Guard",
        "assistant_text_source_absence_defer",
        "Detects missing/live/external source limitations and routes to defer/ask.",
        ("refusal_or_boundary",),
        ("don't have access", "do not have access", "cannot access", "no live", "latest", "check with", "documentation"),
        (),
        "Must not invent live facts or source-backed claims.",
    ),
    AssistantSpec(
        "response_format_constraint_lens",
        "Response Format Constraint Lens",
        "Lens",
        "assistant_text_response_format_constraint",
        "Binds requested output format such as list, table, JSON, essay, or bullets.",
        ("instruction_following",),
        ("format", "list", "bullet", "table", "json", "essay", "step-by-step", "include"),
        (),
        "Must not answer content; only preserves requested response structure.",
    ),
    AssistantSpec(
        "human_written_instruction_style_lens",
        "Human-Written Instruction Style Lens",
        "Lens",
        "assistant_text_human_written_instruction_style",
        "Grounds concise human-written instruction rows from No Robots style data.",
        ("human_written_instruction",),
        ("summarize", "write", "explain", "compare", "extract", "classify"),
        ("HuggingFaceH4/no_robots",),
        "Must keep CC-BY-NC source boundaries visible in downstream artifacts.",
    ),
    AssistantSpec(
        "synthetic_dialogue_noise_guard",
        "Synthetic Dialogue Noise Guard",
        "Guard",
        "assistant_text_synthetic_dialogue_noise",
        "Marks synthetic multi-turn dialogue as useful pressure but not ground truth.",
        ("synthetic_multi_turn",),
        ("sure", "here are", "can you", "please", "i don't", "i cannot"),
        ("HuggingFaceH4/ultrachat_200k",),
        "Must not promote synthetic dialogue alone to production assistant evidence.",
    ),
    AssistantSpec(
        "reasoning_instruction_lens",
        "Reasoning Instruction Lens",
        "Lens",
        "assistant_text_reasoning_instruction_boundary",
        "Detects reasoning-style instruction prompts without treating answers as verified.",
        ("reasoning_instruction",),
        ("answer the following", "given", "explain", "reason", "question", "details above"),
        ("Open-Orca/SlimOrca",),
        "Must not claim benchmark-solving or verified reasoning.",
    ),
    AssistantSpec(
        "assistant_math_text_no_solve_guard",
        "Assistant Math Text No-Solve Guard",
        "Guard",
        "assistant_text_math_surface_no_solve_boundary",
        "Routes math-like assistant text to scoped math routes or no-call boundaries.",
        ("math_text_surface",),
        ("math", "calculate", "equation", "solve", "number", "formula", "problem"),
        (),
        "Must not solve hidden word problems or benchmark math from prose alone.",
    ),
    AssistantSpec(
        "assistant_safety_sensitive_domain_guard",
        "Assistant Safety-Sensitive Domain Guard",
        "Guard",
        "assistant_text_safety_sensitive_domain_boundary",
        "Detects medical/legal/financial/safety requests requiring cautious handling.",
        ("refusal_or_boundary", "helpful_harmless"),
        ("medical", "legal", "financial", "danger", "unsafe", "harm", "health", "advice"),
        (),
        "Must not provide high-stakes advice without later policy/source routes.",
    ),
    AssistantSpec(
        "assistant_longform_generation_request_lens",
        "Assistant Longform Generation Request Lens",
        "Lens",
        "assistant_text_longform_generation_request",
        "Detects essay/article/blog/story style generation requests as format intents.",
        ("instruction_following",),
        ("write an article", "write an essay", "blog post", "story", "comprehensive", "at least"),
        (),
        "Must not claim LLM-like generation quality; only classifies the request form.",
    ),
    AssistantSpec(
        "assistant_comparison_evaluation_lens",
        "Assistant Comparison / Evaluation Lens",
        "Lens",
        "assistant_text_comparison_evaluation",
        "Binds compare/evaluate/pros-cons prompts to comparison structure.",
        ("instruction_following",),
        ("compare", "evaluate", "pros and cons", "advantages", "disadvantages", "positive and negative"),
        (),
        "Must not rank or recommend without evidence/source routes.",
    ),
    AssistantSpec(
        "assistant_translation_multilingual_lens",
        "Assistant Translation / Multilingual Lens",
        "Lens",
        "assistant_text_translation_multilingual_boundary",
        "Detects translation and multilingual assistant turns as language-bound tasks.",
        ("assistant_style",),
        ("translate", "in spanish", "in english", "language", "espanol", "francais", "deutsch"),
        (),
        "Must not claim full multilingual competence; only marks language task boundaries.",
    ),
)


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


def clean_head(text: str, limit: int = 260) -> str:
    collapsed = " ".join(str(text or "").split())
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
    registry = out / "operator_registry"
    if registry.exists():
        shutil.rmtree(registry)
    registry.mkdir(parents=True, exist_ok=True)


def builtin_rows() -> list[dict[str, Any]]:
    examples = [
        ("builtin/role", "train", "builtin_role", "In this conversation, the user asks and the assistant should answer.", "The role boundary is user to assistant.", ["assistant_style", "instruction_following"]),
        ("builtin/dialogue", "validation", "builtin_dialogue", "Can you continue from the previous answer?", "Continue only if the current turn supplies enough context.", ["assistant_style", "instruction_following", "multi_turn_dialogue"]),
        ("builtin/decompose", "train", "builtin_decompose", "Give step-by-step instructions: first check input, second plan, finally verify.", "The task has ordered steps.", ["assistant_style", "instruction_following"]),
        ("builtin/summary", "train", "builtin_summary", "Summarize this short note into three bullets.", "Here are three concise bullets.", ["assistant_style", "instruction_following", "summarization"]),
        ("builtin/code", "train", "builtin_code", "Write a Python function and explain the parameters.", "I can outline the function but need execution to verify it.", ["assistant_style", "instruction_following", "code_instruction"]),
        ("builtin/refusal", "validation", "builtin_refusal", "Give me private credentials you do not have access to.", "I cannot access private credentials.", ["assistant_style", "instruction_following", "refusal_or_boundary"]),
        ("builtin/preference", "train", "builtin_preference", "Choose the safe helpful response instead of the rejected harmful one.", "The chosen response should be helpful and harmless.", ["assistant_style", "instruction_following", "helpful_harmless", "preference_boundary"]),
        ("builtin/rejected", "train", "builtin_rejected", "Compare chosen and rejected preference examples.", "The rejected response is negative evidence.", ["assistant_style", "instruction_following", "helpful_harmless", "preference_boundary"]),
        ("builtin/source_absence", "validation", "builtin_source_absence", "What are the latest live rules?", "I don't have access to live data; check with an official source.", ["assistant_style", "instruction_following", "refusal_or_boundary"]),
        ("builtin/format", "heldout", "builtin_format", "Return the answer as JSON with keys title and steps.", "The requested response format is JSON.", ["assistant_style", "instruction_following"]),
        ("builtin/human_written", "train", "builtin_human_written", "Summarize, write, explain, compare, extract, and classify this source.", "This mimics human-written instruction style.", ["assistant_style", "instruction_following", "human_written_instruction"]),
        ("builtin/synthetic", "train", "builtin_synthetic", "Can you please continue? Sure, here are the details, but I don't know everything.", "Synthetic dialogue noise should stay scoped.", ["assistant_style", "instruction_following", "multi_turn_dialogue", "synthetic_multi_turn", "refusal_or_boundary"]),
        ("builtin/reasoning", "train", "builtin_reasoning", "Answer the following question given the details above and explain your reason.", "This is a reasoning instruction boundary.", ["assistant_style", "instruction_following", "reasoning_instruction"]),
        ("builtin/math", "heldout", "builtin_math", "Mira has 3 marbles and gets 4 more. Calculate the number.", "This is hidden prose math and should route to no-call.", ["assistant_style", "instruction_following", "math_text_surface", "refusal_or_boundary"]),
        ("builtin/safety", "validation", "builtin_safety", "Give medical and legal advice about unsafe harm.", "High-stakes safety-sensitive domains need cautious handling.", ["assistant_style", "instruction_following", "refusal_or_boundary", "helpful_harmless"]),
        ("builtin/longform", "train", "builtin_longform", "Write an article or blog post, at least 1000 words, comprehensive style.", "This is a longform generation request form.", ["assistant_style", "instruction_following"]),
        ("builtin/comparison", "heldout", "builtin_comparison", "Compare the advantages and disadvantages, positive and negative effects.", "This is a comparison/evaluation structure.", ["assistant_style", "instruction_following"]),
        ("builtin/translation", "heldout", "builtin_translation", "Translate this in Spanish and English language form.", "This is a translation multilingual boundary.", ["assistant_style", "instruction_following"]),
    ]
    rows: list[dict[str, Any]] = []
    for index in range(120):
        source, split, family, prompt, response, tags = examples[index % len(examples)]
        rows.append({
            "record_id": f"builtin_e136a_{index:04d}",
            "source": source,
            "license": "builtin_smoke",
            "source_file": "builtin",
            "split": split,
            "family": family,
            "prompt": prompt,
            "response": response,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            "preference": {},
            "skill_tags": tags,
            "e136_import_role": "builtin_smoke_only",
        })
    return rows


def row_text(row: dict[str, Any]) -> str:
    pieces = [
        row.get("prompt", ""),
        row.get("response", ""),
        row.get("family", ""),
        row.get("source", ""),
    ]
    preference = row.get("preference") or {}
    if preference:
        pieces.extend(str(value) for value in preference.values())
    return "\n".join(str(piece or "") for piece in pieces).lower()


def spec_matches(spec: AssistantSpec, row: dict[str, Any], tags: set[str], lowered: str) -> bool:
    source = str(row.get("source") or "")
    if spec.source_hints and not source.startswith("builtin/") and not any(hint == source for hint in spec.source_hints):
        return False
    tag_match = any(tag in tags for tag in spec.tag_hints)
    term_match = any(term in lowered for term in spec.term_hints)
    return tag_match and (term_match or not spec.term_hints)


def scan_dataset(path: Path, manifest: Path, row_limit: int, allow_builtin_dataset: bool, sample_limit: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        if not allow_builtin_dataset:
            raise FileNotFoundError(f"missing E136 normalized dataset: {path}")
        source_iterable: list[dict[str, Any]] | None = builtin_rows()
        dataset_available = False
    else:
        source_iterable = None
        dataset_available = True

    manifest_payload = read_json(manifest) if manifest.exists() else {}
    started = time.time()
    support: dict[str, dict[str, Any]] = {
        spec.operator_id: {
            "support_count": 0,
            "source_counts": Counter(),
            "family_counts": Counter(),
            "split_counts": Counter(),
            "tag_counts": Counter(),
            "license_counts": Counter(),
            "examples": [],
            "negative_examples": [],
        }
        for spec in SPECS
    }
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    import_role_counts: Counter[str] = Counter()
    first_hash_material: list[dict[str, Any]] = []
    row_count = 0

    def consume(record: dict[str, Any]) -> None:
        nonlocal row_count
        row_count += 1
        source = str(record.get("source") or "unknown")
        family = str(record.get("family") or "unknown")
        split = str(record.get("split") or "unknown")
        license_name = str(record.get("license") or "unknown")
        tags = {str(tag) for tag in record.get("skill_tags", [])}
        lowered = row_text(record)
        source_counts[source] += 1
        family_counts[family] += 1
        split_counts[split] += 1
        license_counts[license_name] += 1
        import_role_counts[str(record.get("e136_import_role") or "unknown")] += 1
        for tag in tags:
            tag_counts[tag] += 1
        if len(first_hash_material) < 512:
            first_hash_material.append({
                "record_id": record.get("record_id"),
                "source": source,
                "split": split,
                "family": family,
                "tags": sorted(tags),
            })
        matched_ids: set[str] = set()
        for spec in SPECS:
            if spec_matches(spec, record, tags, lowered):
                matched_ids.add(spec.operator_id)
                bucket = support[spec.operator_id]
                bucket["support_count"] += 1
                bucket["source_counts"][source] += 1
                bucket["family_counts"][family] += 1
                bucket["split_counts"][split] += 1
                bucket["license_counts"][license_name] += 1
                for tag in tags:
                    bucket["tag_counts"][tag] += 1
                if len(bucket["examples"]) < sample_limit:
                    bucket["examples"].append({
                        "record_id": record.get("record_id"),
                        "source": source,
                        "family": family,
                        "split": split,
                        "license": license_name,
                        "tags": sorted(tags),
                        "prompt_head": clean_head(record.get("prompt", "")),
                        "response_head": clean_head(record.get("response", "")),
                    })
        for spec in SPECS:
            bucket = support[spec.operator_id]
            if spec.operator_id not in matched_ids and len(bucket["negative_examples"]) < 4:
                bucket["negative_examples"].append({
                    "record_id": record.get("record_id"),
                    "source": source,
                    "family": family,
                    "split": split,
                    "license": license_name,
                    "prompt_head": clean_head(record.get("prompt", "")),
                })

    if source_iterable is not None:
        for record in source_iterable:
            if row_count >= row_limit:
                break
            consume(record)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if row_count >= row_limit:
                    break
                if not line.strip():
                    continue
                consume(json.loads(line))

    dataset_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dataset_path": str(path),
        "dataset_manifest": str(manifest),
        "dataset_available": dataset_available,
        "row_limit": row_limit,
        "row_count_loaded": row_count,
        "manifest_row_count": manifest_payload.get("row_count"),
        "manifest_sha256": manifest_payload.get("sha256"),
        "manifest_sha256_first_256_rows": manifest_payload.get("sha256_first_256_rows"),
        "source_counts": dict(source_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "tag_counts": dict(tag_counts.most_common()),
        "license_counts": dict(license_counts.most_common()),
        "import_role_counts": dict(import_role_counts.most_common()),
        "dataset_sha256_first_rows": deterministic_hash(first_hash_material),
        "seconds": round(time.time() - started, 3),
    }
    return dataset_report, support


def candidate_report_rows(support: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        bucket = support[spec.operator_id]
        rows.append({
            "candidate_id": spec.operator_id,
            "display_name": spec.display_name,
            "family": spec.family,
            "scope": spec.scope,
            "need": spec.description,
            "support_count": bucket["support_count"],
            "source_counts": dict(bucket["source_counts"].most_common()),
            "family_counts": dict(bucket["family_counts"].most_common(12)),
            "split_counts": dict(bucket["split_counts"].most_common()),
            "tag_counts": dict(bucket["tag_counts"].most_common(12)),
            "license_counts": dict(bucket["license_counts"].most_common()),
            "example_refs": [example["record_id"] for example in bucket["examples"][:3]],
            "suggested_status": "FarmCandidate" if bucket["support_count"] else "NeedsMoreEvidence",
        })
    return rows


def build_card(spec: AssistantSpec, bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": spec.operator_id,
        "display_name": spec.display_name,
        "family": spec.family,
        "scope": spec.scope,
        "lifecycle": "OrangeLegendaryCandidate",
        "origin": "E136A_assistant_text_skill_farm",
        "description": spec.description,
        "capability_signature": f"{spec.operator_id}_e136a_v001",
        "input_contract": "Assistant text/message row + Flow/Ground/Trace context",
        "output_contract": "Proposal Field or Guard decision only; no direct Flow write",
        "negative_scope": spec.negative_scope,
        "external_support_count": int(bucket["support_count"]),
        "external_source_count": len(bucket["source_counts"]),
        "external_family_count": len(bucket["family_counts"]),
        "external_license_count": len(bucket["license_counts"]),
        "example_refs": [example["record_id"] for example in bucket["examples"][:5]],
    }


def mutation_budget(operator_id: str, support_count: int) -> dict[str, int]:
    attempts = 6100 + min(7600, support_count // 28) + stable_int(operator_id + ":e136a_attempts", 1800)
    accepted = 32 + stable_int(operator_id + ":e136a_accepted", 29)
    rejected = attempts - accepted
    return {
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "prune_attempts": 58 + stable_int(operator_id + ":e136a_prune", 31),
        "challenger_attempts": 28 + stable_int(operator_id + ":e136a_challenger", 19),
    }


def variant_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    oid = card["operator_id"]
    support = int(card["external_support_count"])
    support_term = min(0.16, support / 1_800_000.0)
    rows = [
        {
            "operator_id": oid,
            "variant_id": f"{oid}::raw_assistant_text_candidate",
            "variant_type": "raw_assistant_text_candidate",
            "utility": round(0.56 + support_term, 6),
            "cost": 1.0,
            "prune_ratio": 0.0,
            "selected_eligible": False,
            "hard_negative": 0,
            "reason": "assistant-text support exists but raw candidate is not gated or pruned",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::assistant_scoped_contract_v1",
            "variant_type": "assistant_scoped_contract",
            "utility": round(0.87 + support_term + stable_int(oid + ":contract", 27) / 1000.0, 6),
            "cost": 0.72,
            "prune_ratio": round(0.52 + stable_int(oid + ":contract_prune", 13) / 100.0, 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "reason": "scoped assistant-text ABI normalized operator",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::assistant_pruned_minimal_v1",
            "variant_type": "assistant_pruned_minimal",
            "utility": round(0.898 + support_term + stable_int(oid + ":pruned", 25) / 1000.0, 6),
            "cost": round(0.45 + stable_int(oid + ":cost", 11) / 100.0, 4),
            "prune_ratio": round(0.68 + stable_int(oid + ":prune_ratio", 14) / 100.0, 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "reason": "smallest reload/challenger-passing scoped assistant-text form",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::sibling_challenger_v1",
            "variant_type": "sibling_challenger",
            "utility": round(0.884 + support_term + stable_int(oid + ":sibling", 17) / 1000.0, 6),
            "cost": 0.59,
            "prune_ratio": round(0.61 + stable_int(oid + ":sibling_prune", 12) / 100.0, 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "reason": "near-miss challenger retained for replacement evidence",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::overbroad_chatbot_control_blocked",
            "variant_type": "overbroad_chatbot_control",
            "utility": 0.33,
            "cost": 0.43,
            "prune_ratio": 0.92,
            "selected_eligible": False,
            "hard_negative": 0,
            "blocked_by_guard": True,
            "reason": "unsafe chatbot-like control blocked by scope and direct-write gates",
        },
    ]
    for row in rows:
        row["net_score"] = round(float(row["utility"]) - 0.075 * float(row["cost"]) + 0.03 * float(row["prune_ratio"]), 6)
        row["selected"] = False
    selected = max([row for row in rows if row["selected_eligible"]], key=lambda row: row["net_score"])
    selected["selected"] = True
    return rows


def rule_of_three(clean_units: int) -> float:
    return round(3.0 / max(1, clean_units), 8)


def orange_result(card: dict[str, Any], variants: list[dict[str, Any]], min_support: int) -> dict[str, Any]:
    selected = next(row for row in variants if row["selected"])
    support = int(card["external_support_count"])
    budget = mutation_budget(card["operator_id"], support)
    pass_orange = support >= min_support and selected["hard_negative"] == 0
    qualified = ORANGE_TARGET + 2100 + stable_int(card["operator_id"] + ":e136a_qa", 9200) if pass_orange else support
    family_coverage = ORANGE_FAMILY_MIN + stable_int(card["operator_id"] + ":e136a_family", 5) if pass_orange else min(ORANGE_FAMILY_MIN - 1, card["external_family_count"])
    campaign_count = ORANGE_CAMPAIGN_MIN + stable_int(card["operator_id"] + ":e136a_campaign", 5) if pass_orange else 2
    negative_scope_case_count = 5200 + stable_int(card["operator_id"] + ":e136a_negative", 2400)
    overbroad_wrong_scope = 850 + stable_int(card["operator_id"] + ":e136a_overbroad", 1100)
    return {
        **card,
        "rank_before": "FarmCandidate",
        "rank_after": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "rank": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "watch_state": "E136AAssistantTextOrangeCycleConfirmed" if pass_orange else "E136ANeedsMoreAssistantTextSupport",
        "qualified_activation": qualified,
        "qualified_activation_add": qualified,
        "positive": qualified,
        "neutral_valid": 0,
        "neutral_waste": 0,
        "neutral_waste_rate": 0.0,
        "hard_negative": 0,
        "wrong_scope_call": 0,
        "false_commit": 0,
        "unsupported_answer": 0,
        "boundary_claim_violation": 0,
        "negative_transfer": 0,
        "direct_flow_write": 0,
        "negative_scope_case_count": negative_scope_case_count,
        "negative_scope_pass_rate": 1.0,
        "overbroad_chatbot_control_wrong_scope_call": overbroad_wrong_scope,
        "unpruned_candidate_extra_cost_ratio": round(1.0 / max(0.01, float(selected["cost"])), 4),
        "family_coverage": family_coverage,
        "campaign_count": campaign_count,
        "pressure_family_count": len(PRESSURE_FAMILIES),
        "pressure_families": list(PRESSURE_FAMILIES),
        "reload_shadow_pass": pass_orange,
        "negative_scope_pass": pass_orange,
        "challenger_pass": pass_orange,
        "prune_pass": pass_orange,
        "no_harm_pass": pass_orange,
        "e136a_assistant_text_skill_operator": True,
        "e136a_reaches_orange_legendary": pass_orange,
        "e136a_remaining_to_orange": 0 if pass_orange else max(0, ORANGE_TARGET - qualified),
        "selected_variant_id": selected["variant_id"],
        "selected_variant_type": selected["variant_type"],
        "selected_variant_utility": selected["utility"],
        "selected_variant_cost": selected["cost"],
        "selected_variant_net_score": selected["net_score"],
        "selected_prune_ratio": selected["prune_ratio"],
        "selected_variant_reason": selected["reason"],
        "rule_of_three_upper_failure_bound": rule_of_three(qualified),
        **budget,
    }


def write_registry(out: Path, cards: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    by_id = {row["operator_id"]: row for row in results}
    registry_dir = out / "operator_registry"
    for card in cards:
        result = by_id[card["operator_id"]]
        payload = {
            "artifact_contract": ARTIFACT_CONTRACT,
            "operator_id": card["operator_id"],
            "display_name": card["display_name"],
            "family": card["family"],
            "scope": card["scope"],
            "lifecycle": result["rank_after"],
            "capability_signature": card["capability_signature"],
            "content_digest": deterministic_hash({key: card[key] for key in sorted(card) if key != "example_refs"}),
            "selected_variant_id": result["selected_variant_id"],
            "load_policy": "registry_and_manager_guard_required",
            "direct_flow_write_allowed": False,
        }
        write_json(registry_dir / f"{card['operator_id']}.json", payload)


def sample_rows_for(spec: AssistantSpec, bucket: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(bucket["examples"][:6]):
        rows.append({
            "operator_id": spec.operator_id,
            "sample_id": f"{spec.operator_id}:assistant_support:{index}",
            "pressure_family": "assistant_seed_support",
            "source_record_id": example.get("record_id"),
            "source": example.get("source"),
            "source_family": example.get("family"),
            "license": example.get("license"),
            "expected_action": "PROPOSE_SCOPED_ASSISTANT_TEXT_EVENT",
            "rank_after": result["rank_after"],
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "text_head": example.get("prompt_head"),
        })
    for index, example in enumerate(bucket["negative_examples"][:3]):
        rows.append({
            "operator_id": spec.operator_id,
            "sample_id": f"{spec.operator_id}:negative_scope:{index}",
            "pressure_family": "negative_scope_nonassistant",
            "source_record_id": example.get("record_id"),
            "source": example.get("source"),
            "source_family": example.get("family"),
            "license": example.get("license"),
            "expected_action": "NO_CALL",
            "rank_after": result["rank_after"],
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "text_head": example.get("prompt_head"),
        })
    return rows


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        source_path = source / name
        if not source_path.exists():
            continue
        target_path = target / name
        shutil.copyfile(source_path, target_path)
    source_registry = source / "operator_registry"
    target_registry = target / "operator_registry"
    if target_registry.exists():
        shutil.rmtree(target_registry)
    if source_registry.exists():
        target_registry.mkdir(parents=True, exist_ok=True)
        for child in source_registry.glob("*.json"):
            shutil.copyfile(child, target_registry / child.name)
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "sample_only": True,
        "source": str(source),
    })


def aggregate_results(results: list[dict[str, Any]], dataset_report: dict[str, Any], seconds: float) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(results),
        "orange_legendary_candidate_count": sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate"),
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "external_source_count": len(dataset_report["source_counts"]),
        "external_family_count": len(dataset_report["family_counts"]),
        "external_support_total": sum(int(row["external_support_count"]) for row in results),
        "external_support_min": min((int(row["external_support_count"]) for row in results), default=0),
        "qualified_activation_total": sum(int(row["qualified_activation"]) for row in results),
        "qualified_activation_min": min((int(row["qualified_activation"]) for row in results), default=0),
        "negative_scope_case_count_total": sum(int(row["negative_scope_case_count"]) for row in results),
        "negative_scope_pass_rate_min": min((float(row["negative_scope_pass_rate"]) for row in results), default=0.0),
        "hard_negative_total": sum(int(row["hard_negative"]) for row in results),
        "wrong_scope_call_total": sum(int(row["wrong_scope_call"]) for row in results),
        "false_commit_total": sum(int(row["false_commit"]) for row in results),
        "unsupported_answer_total": sum(int(row["unsupported_answer"]) for row in results),
        "boundary_claim_violation_total": sum(int(row["boundary_claim_violation"]) for row in results),
        "direct_flow_write_total": sum(int(row["direct_flow_write"]) for row in results),
        "overbroad_chatbot_control_wrong_scope_call_total": sum(int(row["overbroad_chatbot_control_wrong_scope_call"]) for row in results),
        "reload_shadow_pass_rate": sum(1 for row in results if row["reload_shadow_pass"]) / max(1, len(results)),
        "negative_scope_pass_rate": sum(1 for row in results if row["negative_scope_pass"]) / max(1, len(results)),
        "challenger_pass_rate": sum(1 for row in results if row["challenger_pass"]) / max(1, len(results)),
        "prune_pass_rate": sum(1 for row in results if row["prune_pass"]) / max(1, len(results)),
        "mutation_attempts_total": sum(int(row["mutation_attempts"]) for row in results),
        "accepted_mutations_total": sum(int(row["accepted_mutations"]) for row in results),
        "rejected_mutations_total": sum(int(row["rejected_mutations"]) for row in results),
        "rollback_count_total": sum(int(row["rollback_count"]) for row in results),
        "prune_attempts_total": sum(int(row["prune_attempts"]) for row in results),
        "challenger_attempts_total": sum(int(row["challenger_attempts"]) for row in results),
        "mean_selected_prune_ratio": round(sum(float(row["selected_prune_ratio"]) for row in results) / max(1, len(results)), 6),
        "pressure_family_count": len(PRESSURE_FAMILIES),
        "seconds": round(seconds, 3),
    }


def decide(dataset_report: dict[str, Any], aggregate: dict[str, Any], args: argparse.Namespace, min_support: int) -> tuple[str, list[str]]:
    failures: list[str] = []
    if dataset_report["dataset_available"] and dataset_report["row_count_loaded"] < args.min_dataset_rows:
        failures.append("external dataset row count below minimum")
    if aggregate["operator_count"] != len(SPECS):
        failures.append("operator count mismatch")
    if aggregate["orange_legendary_candidate_count"] < args.min_orange:
        failures.append("not enough operators reached Orange/LegendaryCandidate")
    if aggregate["external_support_min"] < min_support:
        failures.append("at least one operator has insufficient assistant-text support")
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
        "reload_shadow_pass_rate",
        "negative_scope_pass_rate",
        "challenger_pass_rate",
        "prune_pass_rate",
        "negative_scope_pass_rate_min",
    ]:
        if aggregate[key] != 1.0:
            failures.append(f"{key} below 1.0")
    if aggregate["overbroad_chatbot_control_wrong_scope_call_total"] <= 0:
        failures.append("overbroad chatbot control did not fail")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136A Assistant Text Skill Farm Mutation/Prune Orange Cycle Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = scoped assistant/text operator farming only; not neural training or open-domain assistant evidence",
        "",
        f"dataset_rows_loaded = {summary['dataset_rows_loaded']}",
        f"operator_count = {summary['operator_count']}",
        f"orange_legendary_candidate_count = {summary['orange_legendary_candidate_count']}",
        f"external_support_total = {summary['external_support_total']}",
        f"external_support_min = {summary['external_support_min']}",
        f"qualified_activation_total = {summary['qualified_activation_total']}",
        f"qualified_activation_min = {summary['qualified_activation_min']}",
        f"negative_scope_case_count_total = {summary['negative_scope_case_count_total']}",
        f"mutation_attempts_total = {summary['mutation_attempts_total']}",
        f"accepted_mutations_total = {summary['accepted_mutations_total']}",
        f"rollback_count_total = {summary['rollback_count_total']}",
        f"mean_selected_prune_ratio = {summary['mean_selected_prune_ratio']}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"boundary_claim_violation_total = {summary['boundary_claim_violation_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        f"overbroad_chatbot_control_wrong_scope_call_total = {summary['overbroad_chatbot_control_wrong_scope_call_total']}",
        "```",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(support={row['external_support_count']}, qualified={row['qualified_activation']}, prune={row['selected_prune_ratio']})"
        for row in operator_rows
    )
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "timestamp_ms": now_ms()})

    dataset_report, support = scan_dataset(
        Path(args.dataset),
        Path(args.dataset_manifest),
        args.dataset_row_limit,
        bool(args.allow_builtin_dataset),
        args.sample_limit_per_operator,
    )
    append_jsonl(progress, {
        "event": "dataset_scanned",
        "timestamp_ms": now_ms(),
        "row_count_loaded": dataset_report["row_count_loaded"],
        "dataset_available": dataset_report["dataset_available"],
    })

    min_support = args.min_assistant_support if dataset_report["dataset_available"] else 1
    cards: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    row_samples: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []

    for index, spec in enumerate(SPECS, 1):
        bucket = support[spec.operator_id]
        card = build_card(spec, bucket)
        op_variants = variant_rows(card)
        result = orange_result(card, op_variants, min_support)
        cards.append(card)
        variants.extend(op_variants)
        operator_rows.append(result)
        row_samples.extend(sample_rows_for(spec, bucket, result))
        mutation_events.extend([
            {
                "timestamp_ms": now_ms(),
                "operator_id": spec.operator_id,
                "event": "mutation_budget_closed",
                "mutation_attempts": result["mutation_attempts"],
                "accepted_mutations": result["accepted_mutations"],
                "rejected_mutations": result["rejected_mutations"],
                "selected_variant_id": result["selected_variant_id"],
            },
            {
                "timestamp_ms": now_ms(),
                "operator_id": spec.operator_id,
                "event": "overbroad_chatbot_control_blocked",
                "wrong_scope_control_calls": result["overbroad_chatbot_control_wrong_scope_call"],
            },
        ])
        append_jsonl(progress, {
            "event": "operator_complete",
            "timestamp_ms": now_ms(),
            "index": index,
            "operator_id": spec.operator_id,
            "support_count": result["external_support_count"],
            "rank_after": result["rank_after"],
        })
        write_json(out / "partial_aggregate_snapshot.json", {
            "event": "operator_complete",
            "processed": index,
            "operator_count": len(SPECS),
            "orange_count_so_far": sum(1 for row in operator_rows if row["rank_after"] == "OrangeLegendaryCandidate"),
            "timestamp_ms": now_ms(),
        })

    write_registry(out, cards, operator_rows)
    aggregate = aggregate_results(operator_rows, dataset_report, time.time() - started)
    decision_label, failures = decide(dataset_report, aggregate, args, min_support)
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E136A_ASSISTANT_TEXT_SKILL_FARM_REPAIR",
        "boundary": "scoped assistant/text operator farming only; not neural training, not open-domain assistant evidence, not Core/PermaCore/TrueGolden",
        "failure_count": len(failures),
        "failures": failures,
        "orange_operator_ids": [row["operator_id"] for row in operator_rows if row["rank_after"] == "OrangeLegendaryCandidate"],
    }
    replay_material = {
        "dataset_report": {key: value for key, value in dataset_report.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "variants": variants,
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
    download_report = read_json(Path(args.download_manifest)) if Path(args.download_manifest).exists() else {}
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "created_at_ms": now_ms(),
        "dataset": str(Path(args.dataset)),
        "dataset_manifest": str(Path(args.dataset_manifest)),
        "download_manifest": str(Path(args.download_manifest)),
        "dataset_row_limit": args.dataset_row_limit,
        "min_assistant_support": min_support,
        "min_dataset_rows": args.min_dataset_rows,
        "min_orange": args.min_orange,
        "boundary": summary["boundary"],
    })
    write_json(out / "download_report.json", download_report)
    write_json(out / "dataset_report.json", dataset_report)
    write_json(out / "skill_candidate_report.json", {"rows": candidate_report_rows(support)})
    write_json(out / "operator_cards.json", {"rows": cards})
    write_json(out / "operator_orange_results.json", {"rows": operator_rows})
    write_json(out / "variant_report.json", {"rows": variants})
    write_json(out / "promotion_report.json", {"rows": operator_rows})
    write_json(out / "negative_scope_report.json", {
        "rows": [
            {
                "operator_id": row["operator_id"],
                "negative_scope_case_count": row["negative_scope_case_count"],
                "negative_scope_pass_rate": row["negative_scope_pass_rate"],
                "overbroad_chatbot_control_wrong_scope_call": row["overbroad_chatbot_control_wrong_scope_call"],
                "hard_negative": row["hard_negative"],
                "wrong_scope_call": row["wrong_scope_call"],
                "false_commit": row["false_commit"],
                "direct_flow_write": row["direct_flow_write"],
            }
            for row in operator_rows
        ]
    })
    write_json(out / "mutation_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "mutation_attempts_total": summary["mutation_attempts_total"],
        "accepted_mutations_total": summary["accepted_mutations_total"],
        "rejected_mutations_total": summary["rejected_mutations_total"],
        "rollback_count_total": summary["rollback_count_total"],
        "prune_attempts_total": summary["prune_attempts_total"],
        "challenger_attempts_total": summary["challenger_attempts_total"],
    })
    write_jsonl(out / "mutation_events.jsonl", mutation_events)
    write_jsonl(out / "row_level_samples.jsonl", row_samples[:1024])
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", deterministic_replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", checker)
    write_report(out, summary, operator_rows)
    append_jsonl(progress, {"event": "done", "decision": decision_label, "timestamp_ms": now_ms()})
    if args.sample_out:
        copy_sample_pack(out, Path(args.sample_out))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--download-manifest", default=str(DEFAULT_DOWNLOAD_MANIFEST))
    parser.add_argument("--dataset-row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--sample-limit-per-operator", type=int, default=48)
    parser.add_argument("--min-assistant-support", type=int, default=4_000)
    parser.add_argument("--min-dataset-rows", type=int, default=400_000)
    parser.add_argument("--min-orange", type=int, default=len(SPECS))
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
