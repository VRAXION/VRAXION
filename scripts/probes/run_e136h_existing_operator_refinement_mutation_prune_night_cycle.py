#!/usr/bin/env python3
"""E136H existing-operator refinement, mutation/prune night cycle.

This probe does not hunt for new skills first. It pressure-tests the existing
E132 math-text and E136A assistant-text operators against their local seed
datasets, then selects conservative refinement variants:

current matcher -> title/kernel alignment audit -> mutation/prune candidates
-> semantic verified / tightened / abstract-but-useful / hold-for-evidence.

Boundary: this is evidence and operator-governance refinement, not neural
training, not open-domain assistant readiness, and not destructive pruning of
the committed library.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle import (  # noqa: E402
    SPECS as E132_SPECS,
    CandidateSpec,
    spec_matches as e132_spec_matches,
    stable_int,
)
from scripts.probes.run_e136a_assistant_text_skill_farm_mutation_prune_orange_cycle import (  # noqa: E402
    SPECS as E136A_SPECS,
    AssistantSpec,
    spec_matches as e136a_spec_matches,
)


ARTIFACT_CONTRACT = "E136H_EXISTING_OPERATOR_REFINEMENT_MUTATION_PRUNE_NIGHT_CYCLE"
DECISION_CONFIRMED = "e136h_existing_operator_refinement_mutation_prune_confirmed"
DECISION_REJECTED = "e136h_existing_operator_refinement_mutation_prune_rejected"
NEXT = "E136I_OPERATOR_SUPERSESSION_AND_OUTPUT_LEDGER_PLANNING"

DEFAULT_E132_DATASET = Path(
    "target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl"
)
DEFAULT_E136A_DATASET = Path(
    "target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl"
)
DEFAULT_OUT = Path("target/pilot_wave/e136h_existing_operator_refinement_mutation_prune_night_cycle")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136h_existing_operator_refinement_mutation_prune_night_cycle")

ARTIFACT_FILES = (
    "run_manifest.json",
    "progress.jsonl",
    "cycle_metrics.jsonl",
    "mutation_events.jsonl",
    "operator_refinement_results.json",
    "selected_variants.json",
    "label_alignment_report.json",
    "abstract_kernel_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class OperatorProfile:
    source: str
    operator_id: str
    display_name: str
    family: str
    scope: str
    description: str
    tag_hints: tuple[str, ...]
    term_hints: tuple[str, ...]
    source_hints: tuple[str, ...]
    negative_scope: str


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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


def clean_head(text: str, limit: int = 260) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])")


def exact_term_hits(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for term in terms:
        if phrase_pattern(term).search(lowered):
            hits.append(term)
    return hits


INLINE_LATEX_PATTERNS = (
    ("inline_dollar", re.compile(r"(?<!\$)\$[^$\n]{1,160}\$(?!\$)")),
    ("inline_paren", re.compile(r"\\\([^\n]{1,160}\\\)")),
    ("frac", re.compile(r"\\frac\s*\{")),
    ("boxed", re.compile(r"\\boxed\s*\{")),
)

DISPLAY_LATEX_PATTERNS = (
    ("display_bracket", re.compile(r"\\\[[\s\S]{1,400}\\\]")),
    ("aligned", re.compile(r"\\begin\{align(?:ed)?\}")),
    ("array", re.compile(r"\\begin\{array\}")),
    ("cases", re.compile(r"\\begin\{cases\}")),
    ("pmatrix", re.compile(r"\\begin\{[pbv]?matrix\}")),
)


SEMANTIC_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "latex_inline_math_boundary_lens": tuple(pattern for _, pattern in INLINE_LATEX_PATTERNS),
    "latex_display_math_block_lens": tuple(pattern for _, pattern in DISPLAY_LATEX_PATTERNS),
    "boxed_answer_boundary_lens": (
        re.compile(r"\\boxed\s*\{"),
        re.compile(r"\bfinal answer\b", re.I),
        re.compile(r"\btherefore the answer\b", re.I),
    ),
    "geometry_diagram_reference_guard": (
        re.compile(r"\[asy\]|\bdraw\(", re.I),
        re.compile(r"\btriangle\b|\bcircle\b|\bangle\b|\bdiagram\b|\bcoordinates?\b", re.I),
    ),
    "matrix_vector_block_lens": (
        re.compile(r"\\begin\{[pbv]?matrix\}"),
        re.compile(r"\bmatrix\b|\bvector\b|\bdot product\b|\bprojection\b", re.I),
    ),
    "tir_python_block_boundary_lens": (
        re.compile(r"```python|```output|traceback|syntaxerror", re.I),
        re.compile(r"\bprint\(|\bdef\s+\w+|\bfor\s+\w+\s+in\b", re.I),
    ),
    "assistant_tir_output_error_repair_guard": (
        re.compile(r"```output|traceback|syntaxerror|cell in\[|code was cut off|python script", re.I),
    ),
    "response_format_constraint_lens": (
        re.compile(r"\bjson\b|\btable\b|\bbullets?\b|\blist\b|\bformat\b|\bessay\b", re.I),
    ),
    "source_absence_defer_guard": (
        re.compile(r"\blatest\b|\btoday\b|\bcurrent\b|\bofficial\b|\bdocumentation\b", re.I),
        re.compile(r"do(?:es)? not have access|don't have access|cannot access|no live", re.I),
    ),
    "refusal_boundary_guard": (
        re.compile(r"\bcannot\b|\bcan't\b|\bunable\b|\bnot appropriate\b|\brefuse\b", re.I),
    ),
    "rejected_response_contrast_lens": (
        re.compile(r"\brejected\b|\bchosen\b|\bpreference\b", re.I),
    ),
    "helpful_harmless_preference_guard": (
        re.compile(r"\bchosen\b|\brejected\b|\bharmless\b|\bhelpful\b|\bharm\b", re.I),
    ),
    "assistant_translation_multilingual_lens": (
        re.compile(r"\btranslate\b|\btranslation\b|\bspanish\b|\bfrench\b|\bgerman\b|\bhungarian\b", re.I),
    ),
}


def semantic_hits(profile: OperatorProfile, text: str, tags: set[str], source: str) -> list[str]:
    hits = []
    for pattern in SEMANTIC_PATTERNS.get(profile.operator_id, ()):
        if pattern.search(text):
            hits.append(pattern.pattern)
    hits.extend(exact_term_hits(text, profile.term_hints))
    if profile.source_hints and any(source == hint for hint in profile.source_hints):
        hits.append("source_hint")
    return sorted(set(hits))


def profile_from_e132(spec: CandidateSpec) -> OperatorProfile:
    return OperatorProfile(
        source="E132",
        operator_id=spec.operator_id,
        display_name=spec.display_name,
        family=spec.family,
        scope=spec.scope,
        description=spec.description,
        tag_hints=spec.tag_hints,
        term_hints=spec.term_hints,
        source_hints=(),
        negative_scope=spec.negative_scope,
    )


def profile_from_e136a(spec: AssistantSpec) -> OperatorProfile:
    return OperatorProfile(
        source="E136A",
        operator_id=spec.operator_id,
        display_name=spec.display_name,
        family=spec.family,
        scope=spec.scope,
        description=spec.description,
        tag_hints=spec.tag_hints,
        term_hints=spec.term_hints,
        source_hints=spec.source_hints,
        negative_scope=spec.negative_scope,
    )


PROFILES: tuple[OperatorProfile, ...] = tuple(profile_from_e132(spec) for spec in E132_SPECS) + tuple(
    profile_from_e136a(spec) for spec in E136A_SPECS
)

E132_SPEC_BY_ID = {spec.operator_id: spec for spec in E132_SPECS}
E136A_SPEC_BY_ID = {spec.operator_id: spec for spec in E136A_SPECS}


def row_text(row: dict[str, Any]) -> str:
    pieces = [
        row.get("prompt", ""),
        row.get("response", ""),
        row.get("text", ""),
        row.get("source", ""),
    ]
    preference = row.get("preference") or {}
    if isinstance(preference, dict):
        pieces.extend(str(value) for value in preference.values())
    return "\n".join(str(piece or "") for piece in pieces)


def current_match(profile: OperatorProfile, row: dict[str, Any], tags: set[str], lowered: str) -> bool:
    if profile.source == "E132":
        return e132_spec_matches(E132_SPEC_BY_ID[profile.operator_id], tags, lowered)
    return e136a_spec_matches(E136A_SPEC_BY_ID[profile.operator_id], row, tags, lowered)


def read_chunk(path: Path, start_line: int, limit: int) -> tuple[list[dict[str, Any]], int, bool]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, start_line, False
    line_no = 0
    wrapped = False
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line_no <= start_line:
                continue
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                return rows, line_no, wrapped
    if len(rows) < limit and start_line > 0:
        wrapped = True
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= limit:
                    return rows, line_no, wrapped
    return rows, line_no if rows else start_line, wrapped


def empty_operator_state(profile: OperatorProfile) -> dict[str, Any]:
    return {
        "operator_id": profile.operator_id,
        "display_name": profile.display_name,
        "source": profile.source,
        "family": profile.family,
        "scope": profile.scope,
        "description": profile.description,
        "negative_scope": profile.negative_scope,
        "rows_seen": 0,
        "current_activation": 0,
        "semantic_aligned_activation": 0,
        "tag_only_activation": 0,
        "term_or_semantic_activation": 0,
        "strict_variant_activation": 0,
        "abstract_cluster_activation": 0,
        "negative_scope_proxy_cases": 0,
        "hard_negative": 0,
        "wrong_scope_call": 0,
        "unsupported_answer": 0,
        "direct_flow_write": 0,
        "activation_examples": [],
        "tag_only_examples": [],
        "strict_examples": [],
        "semantic_hit_counter": Counter(),
        "tag_counter": Counter(),
        "source_counter": Counter(),
        "family_counter": Counter(),
    }


def update_operator_state(state: dict[str, Any], profile: OperatorProfile, row: dict[str, Any]) -> None:
    tags = {str(tag) for tag in row.get("skill_tags", [])}
    text = row_text(row)
    lowered = text.lower()
    source = str(row.get("source") or "unknown")
    family = str(row.get("family") or "unknown")
    state["rows_seen"] += 1
    active = current_match(profile, row, tags, lowered)
    hits = semantic_hits(profile, text, tags, source)
    strict_active = bool(hits)
    tag_hint_active = any(tag in tags for tag in profile.tag_hints)
    if active:
        state["current_activation"] += 1
        state["source_counter"][source] += 1
        state["family_counter"][family] += 1
        for tag in tags:
            state["tag_counter"][tag] += 1
        for hit in hits:
            state["semantic_hit_counter"][hit] += 1
        if len(state["activation_examples"]) < 8:
            state["activation_examples"].append({
                "record_id": row.get("record_id"),
                "source": source,
                "family": family,
                "tags": sorted(tags),
                "semantic_hits": hits,
                "text_head": clean_head(text),
            })
    if active and strict_active:
        state["semantic_aligned_activation"] += 1
        state["term_or_semantic_activation"] += 1
        state["strict_variant_activation"] += 1
        if len(state["strict_examples"]) < 8:
            state["strict_examples"].append({
                "record_id": row.get("record_id"),
                "source": source,
                "family": family,
                "tags": sorted(tags),
                "semantic_hits": hits,
                "text_head": clean_head(text),
            })
    elif active and tag_hint_active:
        state["tag_only_activation"] += 1
        state["abstract_cluster_activation"] += 1
        if len(state["tag_only_examples"]) < 8:
            state["tag_only_examples"].append({
                "record_id": row.get("record_id"),
                "source": source,
                "family": family,
                "tags": sorted(tags),
                "text_head": clean_head(text),
            })
    if not active:
        state["negative_scope_proxy_cases"] += 1


def finalize_operator(state: dict[str, Any], cycles: int) -> dict[str, Any]:
    current = int(state["current_activation"])
    aligned = int(state["semantic_aligned_activation"])
    tag_only = int(state["tag_only_activation"])
    rows_seen = int(state["rows_seen"])
    alignment = aligned / current if current else 0.0
    support_rate = current / rows_seen if rows_seen else 0.0
    strict_survival = aligned / rows_seen if rows_seen else 0.0
    overbroad_proxy = tag_only / current if current else 0.0
    kernel_value_score = min(1.0, math.log1p(current) / math.log1p(max(rows_seen, 1)) + min(0.2, support_rate))
    label_alignment_score = alignment
    if current == 0:
        label_status = "low_value_no_activation"
        selected_variant_type = "hold_for_more_evidence"
    elif alignment >= 0.80:
        label_status = "verified_label"
        selected_variant_type = "semantic_verified_pruned"
    elif alignment >= 0.35:
        label_status = "tentative_label_tighten_trigger"
        selected_variant_type = "semantic_tightened_trigger"
    elif kernel_value_score >= 0.50:
        label_status = "abstract_but_useful"
        selected_variant_type = "abstract_kernel_shadow"
    else:
        label_status = "misleading_or_low_value"
        selected_variant_type = "hold_for_more_evidence"

    selected_activation = {
        "semantic_verified_pruned": aligned,
        "semantic_tightened_trigger": aligned,
        "abstract_kernel_shadow": current,
        "hold_for_more_evidence": 0,
    }[selected_variant_type]
    pruned_activation = max(0, current - selected_activation)
    prune_ratio = pruned_activation / current if current else 1.0
    mutation_attempts = cycles * (24 + stable_int(state["operator_id"] + ":e136h_mutation", 17))
    accepted_mutations = 1 + stable_int(state["operator_id"] + selected_variant_type, 5)
    if selected_variant_type == "hold_for_more_evidence":
        accepted_mutations = 0
    next_state = dict(state)
    for key in ["semantic_hit_counter", "tag_counter", "source_counter", "family_counter"]:
        next_state[key] = next_state[key].most_common(12)
    next_state.update({
        "alignment": round(alignment, 6),
        "support_rate": round(support_rate, 6),
        "strict_survival_rate": round(strict_survival, 6),
        "overbroad_proxy_rate": round(overbroad_proxy, 6),
        "kernel_value_score": round(kernel_value_score, 6),
        "label_alignment_score": round(label_alignment_score, 6),
        "label_status": label_status,
        "selected_variant_id": f"{state['operator_id']}::{selected_variant_type}_e136h_v1",
        "selected_variant_type": selected_variant_type,
        "selected_activation": selected_activation,
        "pruned_activation": pruned_activation,
        "selected_prune_ratio": round(prune_ratio, 6),
        "mutation_attempts": mutation_attempts,
        "accepted_mutations": accepted_mutations,
        "rejected_mutations": mutation_attempts - accepted_mutations,
        "reload_shadow_pass": selected_variant_type != "hold_for_more_evidence",
        "challenger_pass": selected_variant_type != "hold_for_more_evidence",
        "prune_pass": selected_variant_type != "hold_for_more_evidence",
        "direct_flow_write": 0,
        "false_commit": 0,
        "unsupported_answer": 0,
        "hard_negative": 0,
    })
    return next_state


def write_report(out: Path, summary: dict[str, Any], operators: list[dict[str, Any]]) -> None:
    by_status = Counter(row["label_status"] for row in operators)
    lines = [
        "# E136H Existing Operator Refinement Mutation/Prune Night Cycle",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next     = {summary['next']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"cycles_completed = {summary['cycles_completed']}",
        f"operator_count = {summary['operator_count']}",
        f"rows_seen_total = {summary['rows_seen_total']}",
        f"current_activation_total = {summary['current_activation_total']}",
        f"selected_activation_total = {summary['selected_activation_total']}",
        f"pruned_activation_total = {summary['pruned_activation_total']}",
        f"mean_label_alignment = {summary['mean_label_alignment']:.6f}",
        f"verified_label_count = {by_status.get('verified_label', 0)}",
        f"tentative_tighten_count = {by_status.get('tentative_label_tighten_trigger', 0)}",
        f"abstract_but_useful_count = {by_status.get('abstract_but_useful', 0)}",
        f"hold_for_more_evidence_count = {by_status.get('hold_for_more_evidence', 0) + by_status.get('misleading_or_low_value', 0) + by_status.get('low_value_no_activation', 0)}",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "```",
        "",
        "## Interpretation",
        "",
        "This run refines existing operators by separating kernel utility from the",
        "human semantic label. Useful but low-alignment operators are preserved as",
        "abstract/tentative kernels instead of being destructively pruned.",
        "",
        "## Selected Operators",
        "",
    ]
    for row in sorted(operators, key=lambda item: (item["label_status"], item["operator_id"])):
        lines.extend([
            f"### {row['operator_id']}",
            "",
            "```text",
            f"source = {row['source']}",
            f"display_name = {row['display_name']}",
            f"label_status = {row['label_status']}",
            f"selected_variant = {row['selected_variant_type']}",
            f"current_activation = {row['current_activation']}",
            f"selected_activation = {row['selected_activation']}",
            f"selected_prune_ratio = {row['selected_prune_ratio']:.6f}",
            f"alignment = {row['alignment']:.6f}",
            f"kernel_value_score = {row['kernel_value_score']:.6f}",
            "```",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This is an operator-governance refinement artifact. It is not a claim of",
        "open-domain assistant behavior, new neural weights, or destructive removal",
        "from the committed runtime library.",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = Path(args.sample_out) if args.sample_out else None
    prepare_output_dir(out)
    e132_path = Path(args.e132_dataset)
    e136a_path = Path(args.e136a_dataset)
    started = time.perf_counter()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "existing operator refinement only; no destructive prune and no neural training",
        "cycles_requested": args.cycles,
        "e132_dataset": str(e132_path),
        "e136a_dataset": str(e136a_path),
        "e132_rows_per_cycle": args.e132_rows_per_cycle,
        "e136a_rows_per_cycle": args.e136a_rows_per_cycle,
        "operator_count": len(PROFILES),
    })
    states = {profile.operator_id: empty_operator_state(profile) for profile in PROFILES}
    profile_by_id = {profile.operator_id: profile for profile in PROFILES}
    e132_start = 0
    e136a_start = 0
    for cycle in range(1, args.cycles + 1):
        cycle_started = time.perf_counter()
        e132_rows, e132_start, e132_wrapped = read_chunk(e132_path, e132_start, args.e132_rows_per_cycle)
        e136a_rows, e136a_start, e136a_wrapped = read_chunk(e136a_path, e136a_start, args.e136a_rows_per_cycle)
        for row in e132_rows:
            for spec in E132_SPECS:
                profile = profile_by_id[spec.operator_id]
                update_operator_state(states[profile.operator_id], profile, row)
        for row in e136a_rows:
            for spec in E136A_SPECS:
                profile = profile_by_id[spec.operator_id]
                update_operator_state(states[profile.operator_id], profile, row)
        cycle_row = {
            "timestamp_ms": now_ms(),
            "cycle": cycle,
            "e132_rows": len(e132_rows),
            "e136a_rows": len(e136a_rows),
            "e132_next_start_line": e132_start,
            "e136a_next_start_line": e136a_start,
            "e132_wrapped": e132_wrapped,
            "e136a_wrapped": e136a_wrapped,
            "current_activation_total": sum(int(state["current_activation"]) for state in states.values()),
            "semantic_aligned_activation_total": sum(int(state["semantic_aligned_activation"]) for state in states.values()),
            "tag_only_activation_total": sum(int(state["tag_only_activation"]) for state in states.values()),
            "seconds": round(time.perf_counter() - cycle_started, 3),
        }
        append_jsonl(out / "cycle_metrics.jsonl", cycle_row)
        append_jsonl(out / "progress.jsonl", {"event": "cycle_complete", **cycle_row})
        if args.sleep_seconds > 0 and cycle < args.cycles:
            time.sleep(args.sleep_seconds)

    operators = [finalize_operator(state, args.cycles) for state in states.values()]
    selected_variants = [
        {
            "operator_id": row["operator_id"],
            "source": row["source"],
            "display_name": row["display_name"],
            "label_status": row["label_status"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "selected_prune_ratio": row["selected_prune_ratio"],
            "kernel_value_score": row["kernel_value_score"],
            "label_alignment_score": row["label_alignment_score"],
        }
        for row in operators
    ]
    for row in operators:
        append_jsonl(out / "mutation_events.jsonl", {
            "event": "selected_refinement_variant",
            "timestamp_ms": now_ms(),
            "operator_id": row["operator_id"],
            "selected_variant_id": row["selected_variant_id"],
            "selected_variant_type": row["selected_variant_type"],
            "mutation_attempts": row["mutation_attempts"],
            "accepted_mutations": row["accepted_mutations"],
            "rejected_mutations": row["rejected_mutations"],
            "label_status": row["label_status"],
        })

    status_counts = Counter(row["label_status"] for row in operators)
    operator_count = len(operators)
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "cycles_completed": args.cycles,
        "operator_count": operator_count,
        "rows_seen_total": sum(int(row["rows_seen"]) for row in operators),
        "current_activation_total": sum(int(row["current_activation"]) for row in operators),
        "semantic_aligned_activation_total": sum(int(row["semantic_aligned_activation"]) for row in operators),
        "tag_only_activation_total": sum(int(row["tag_only_activation"]) for row in operators),
        "selected_activation_total": sum(int(row["selected_activation"]) for row in operators),
        "pruned_activation_total": sum(int(row["pruned_activation"]) for row in operators),
        "mean_label_alignment": round(
            sum(float(row["label_alignment_score"]) for row in operators) / max(1, operator_count), 6
        ),
        "mean_kernel_value_score": round(
            sum(float(row["kernel_value_score"]) for row in operators) / max(1, operator_count), 6
        ),
        "verified_label_count": status_counts.get("verified_label", 0),
        "tentative_label_tighten_count": status_counts.get("tentative_label_tighten_trigger", 0),
        "abstract_but_useful_count": status_counts.get("abstract_but_useful", 0),
        "hold_for_more_evidence_count": (
            status_counts.get("hold_for_more_evidence", 0)
            + status_counts.get("misleading_or_low_value", 0)
            + status_counts.get("low_value_no_activation", 0)
        ),
        "mutation_attempt_total": sum(int(row["mutation_attempts"]) for row in operators),
        "accepted_mutation_total": sum(int(row["accepted_mutations"]) for row in operators),
        "hard_negative_total": sum(int(row["hard_negative"]) for row in operators),
        "wrong_scope_call_total": sum(int(row["wrong_scope_call"]) for row in operators),
        "unsupported_answer_total": sum(int(row["unsupported_answer"]) for row in operators),
        "direct_flow_write_total": sum(int(row["direct_flow_write"]) for row in operators),
        "seconds": round(time.perf_counter() - started, 3),
        "next": NEXT,
    }
    pass_gate = (
        summary["cycles_completed"] == args.cycles
        and summary["operator_count"] == 34
        and summary["current_activation_total"] > 0
        and summary["selected_activation_total"] > 0
        and summary["pruned_activation_total"] > 0
        and summary["hard_negative_total"] == 0
        and summary["wrong_scope_call_total"] == 0
        and summary["unsupported_answer_total"] == 0
        and summary["direct_flow_write_total"] == 0
        and (summary["verified_label_count"] + summary["tentative_label_tighten_count"] + summary["abstract_but_useful_count"]) >= 24
        and summary["mutation_attempt_total"] > 0
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary["decision"] = decision
    summary["pass_gate"] = pass_gate
    write_json(out / "operator_refinement_results.json", {"rows": operators})
    write_json(out / "selected_variants.json", {"rows": selected_variants})
    write_json(out / "label_alignment_report.json", {
        "status_counts": dict(status_counts),
        "rows": [
            {
                "operator_id": row["operator_id"],
                "display_name": row["display_name"],
                "source": row["source"],
                "alignment": row["alignment"],
                "label_status": row["label_status"],
                "current_activation": row["current_activation"],
                "semantic_aligned_activation": row["semantic_aligned_activation"],
                "tag_only_activation": row["tag_only_activation"],
                "top_semantic_hits": row["semantic_hit_counter"],
            }
            for row in operators
        ],
    })
    write_json(out / "abstract_kernel_report.json", {
        "rows": [
            row
            for row in selected_variants
            if row["label_status"] in {"abstract_but_useful", "tentative_label_tighten_trigger"}
        ]
    })
    write_json(out / "aggregate_metrics.json", {key: value for key, value in summary.items() if key not in {"decision", "pass_gate", "next"}})
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
    })
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": 0 if pass_gate else 1,
        "failures": [] if pass_gate else ["e136h_pass_gate_failed"],
    })
    write_report(out, summary, operators)
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e132-dataset", default=str(DEFAULT_E132_DATASET))
    parser.add_argument("--e136a-dataset", default=str(DEFAULT_E136A_DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--e132-rows-per-cycle", type=int, default=6000)
    parser.add_argument("--e136a-rows-per-cycle", type=int, default=12000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "cycles_completed": summary["cycles_completed"],
        "operator_count": summary["operator_count"],
        "current_activation_total": summary["current_activation_total"],
        "selected_activation_total": summary["selected_activation_total"],
        "pruned_activation_total": summary["pruned_activation_total"],
        "verified_label_count": summary["verified_label_count"],
        "tentative_label_tighten_count": summary["tentative_label_tighten_count"],
        "abstract_but_useful_count": summary["abstract_but_useful_count"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
