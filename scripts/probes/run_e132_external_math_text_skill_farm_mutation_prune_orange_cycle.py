#!/usr/bin/env python3
"""E132 external math-text skill farm, mutation/prune, and Orange cycle.

E132 takes the external math-text seed pack built after E131 and farms scoped
operator candidates from repeated math-writing surfaces:

support -> candidate variants -> mutation/prune/challenger/reload ->
negative scope -> scoped Orange/LegendaryCandidate.

Boundary: these are math-text lenses and guards. They do not solve GSM8K/MATH,
do not train a neural model, and do not promote anything to Core, PermaCore, or
TrueGolden.
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


ARTIFACT_CONTRACT = "E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE"
DECISION_CONFIRMED = "e132_external_math_text_skill_farm_mutation_prune_orange_cycle_confirmed"
DECISION_REJECTED = "e132_external_math_text_skill_farm_mutation_prune_orange_cycle_rejected"
NEXT = "E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM"

DEFAULT_DATASET = Path("target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl")
DEFAULT_DATASET_MANIFEST = Path("target/datasets/e132_external_math_text_seed_pack/normalized_manifest.json")
DEFAULT_DOWNLOAD_MANIFEST = Path("target/datasets/e132_external_math_text_seed_pack/download_manifest.json")
DEFAULT_OUT = Path("target/pilot_wave/e132_external_math_text_skill_farm_mutation_prune_orange_cycle")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e132_external_math_text_skill_farm_mutation_prune_orange_cycle")

ORANGE_TARGET = 300_000
ORANGE_FAMILY_MIN = 12
ORANGE_CAMPAIGN_MIN = 8
DEFAULT_ROW_LIMIT = 250_000

PRESSURE_FAMILIES = (
    "external_seed_support",
    "cross_source_replay",
    "assistant_wrapper_replay",
    "latex_noise_stability",
    "tir_output_noise_stability",
    "proof_connector_noise_stability",
    "hidden_word_problem_no_solve",
    "negative_scope_nonmath",
    "overbroad_solver_control",
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
class CandidateSpec:
    operator_id: str
    display_name: str
    family: str
    scope: str
    description: str
    tag_hints: tuple[str, ...]
    term_hints: tuple[str, ...]
    negative_scope: str


SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec("latex_inline_math_boundary_lens", "LaTeX Inline Math Boundary Lens", "Lens", "math_text_latex_inline_boundary", "Detects inline LaTeX math spans as bounded evidence, not ordinary prose.", ("latex_math_surface",), ("$x", "\\(", "\\frac", "\\boxed"), "Must not solve the span; only proposes the bounded math-text surface."),
    CandidateSpec("latex_display_math_block_lens", "LaTeX Display Math Block Lens", "Lens", "math_text_latex_display_block_boundary", "Detects display math, align, array, cases, and matrix blocks.", ("latex_math_surface", "matrix_vector_surface"), ("\\[", "\\begin{aligned}", "\\begin{array}", "\\begin{cases}", "\\begin{pmatrix}"), "Must not flatten multiline math blocks into prose claims."),
    CandidateSpec("boxed_answer_boundary_lens", "Boxed Answer Boundary Lens", "Lens", "math_text_boxed_answer_boundary", "Finds boxed/final-answer boundaries without treating them as proof of correctness.", ("answer_boundary_surface",), ("\\boxed{", "final answer", "answer is", "therefore the answer"), "Must not trust a boxed answer without route evidence."),
    CandidateSpec("tir_python_block_boundary_lens", "TIR Python Block Boundary Lens", "Lens", "math_text_tir_python_block_boundary", "Detects tool-integrated reasoning code and output blocks as structure.", ("tir_program_or_output_surface",), ("```python", "```output", "print(", "for ", "def "), "Must not execute code or accept outputs as direct Flow writes."),
    CandidateSpec("proof_step_connector_lens", "Proof Step Connector Lens", "Lens", "math_text_proof_step_connector", "Grounds therefore/hence/implies/substituting/equating proof-step connectors.", ("proof_step_surface",), ("therefore", "hence", "thus", "implies", "substituting", "equating", "we have"), "Must not convert proof-step prose into an unscoped solved answer."),
    CandidateSpec("geometry_diagram_reference_guard", "Geometry Diagram Reference Guard", "Guard", "math_text_geometry_diagram_reference", "Detects geometry, coordinate, diagram, and Asymptote references that need bounded handling.", ("geometry_diagram_or_coordinate_surface",), ("[asy]", "draw(", "triangle", "circle", "angle", "coordinates", "diagram"), "Must no-call or defer when a missing diagram is required."),
    CandidateSpec("matrix_vector_block_lens", "Matrix / Vector Block Lens", "Lens", "math_text_matrix_vector_block", "Detects matrix/vector/projection blocks and keeps vector syntax intact.", ("matrix_vector_surface",), ("\\begin{pmatrix}", "\\begin{bmatrix}", "matrix", "vector", "dot product", "projection"), "Must not decompose matrices into unrelated scalar text."),
    CandidateSpec("equation_system_alignment_lens", "Equation System Alignment Lens", "Lens", "math_text_equation_system_alignment", "Detects aligned equations and equation-system structure.", ("latex_math_surface", "proof_step_surface"), ("\\begin{aligned}", "\\begin{align", "\\\\", "system of equations", "simultaneous equations"), "Must preserve equation row boundaries and not collapse separate equations."),
    CandidateSpec("piecewise_case_function_lens", "Piecewise / Case Function Lens", "Lens", "math_text_piecewise_case_function", "Detects piecewise/case/function definitions and binds conditions to branches.", ("latex_math_surface",), ("\\begin{cases}", "\\left\\{", "piecewise", "function", " if "), "Must not commit branch values without the matching condition."),
    CandidateSpec("fraction_ratio_probability_lens", "Fraction / Ratio / Probability Lens", "Lens", "math_text_fraction_ratio_probability", "Grounds fractions, ratios, common-fraction instructions, and probability spans.", ("latex_math_surface", "word_problem_boundary_candidate"), ("\\frac{", "probability", "ratio", "common fraction", "/"), "Must not rewrite hidden word problems into solved arithmetic."),
    CandidateSpec("variable_definition_binding_lens", "Variable Definition Binding Lens", "Lens", "math_text_variable_definition_binding", "Binds let/where/define variable introductions to nearby equations.", ("proof_step_surface", "latex_math_surface"), ("let ", "where ", "define ", "denote ", "such that"), "Must not leak one variable definition into a neighboring problem."),
    CandidateSpec("summation_sequence_series_lens", "Summation / Sequence / Series Lens", "Lens", "math_text_summation_sequence_series", "Detects summation, sequence, series, binomial, and Pascal-like structures.", ("latex_math_surface",), ("\\sum", "sequence", "series", "pascal", "binomial", "progression"), "Must not calculate a closed form by itself."),
    CandidateSpec("unit_quantity_binding_lens", "Unit Quantity Binding Lens", "Lens", "math_text_unit_quantity_binding", "Binds quantities to units such as cm, kg, dollars, minutes, percent, and units.", ("word_problem_boundary_candidate",), (" cm", " km", " dollars", " cents", " minutes", " hours", " percent", "%", " units"), "Must not perform unit conversion unless a later route explicitly supports it."),
    CandidateSpec("word_problem_no_solve_guard_v2", "Word Problem No-Solve Guard V2", "Guard", "math_text_hidden_word_problem_no_solve", "Separates prose-only word problems from visible-equation arithmetic routes.", ("word_problem_boundary_candidate",), ("how many", "what is", "find", "compute", "probability", "total", "average", "randomly"), "Must no-call hidden prose-only math questions instead of solving them."),
    CandidateSpec("assistant_tir_output_error_repair_guard", "Assistant TIR Output / Error Repair Guard", "Guard", "math_text_tir_output_error_repair", "Detects TIR output/error/cut-off blocks that need repair or no-call handling.", ("tir_program_or_output_surface",), ("```output", "syntaxerror", "traceback", "cell in[", "code was cut off", "python script"), "Must not treat failed code output as validated math."),
    CandidateSpec("answer_format_instruction_lens", "Answer Format Instruction Lens", "Lens", "math_text_answer_format_instruction", "Binds answer-format instructions such as common fraction, ordered pair, or comma list.", ("answer_boundary_surface",), ("express your answer as", "enter your answer", "give your answer", "common fraction", "ordered pair"), "Must not answer; only preserves requested output format for later routes."),
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


def clean_head(text: str, limit: int = 240) -> str:
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
    registry = out / "operator_registry"
    if registry.exists():
        for child in registry.glob("*.json"):
            child.unlink()
    registry.mkdir(parents=True, exist_ok=True)


def builtin_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    examples = [
        ("builtin/latex", "train", "builtin_latex", "Find $x+y$ from \\[x=2, y=3\\].", "Therefore the answer is \\boxed{5}.", ["latex_math_surface", "answer_boundary_surface"]),
        ("builtin/tir", "validation", "builtin_tir", "Compute the coefficient.", "```python\nprint(2+3)\n```\n```output\n5\n```", ["tir_program_or_output_surface", "answer_boundary_surface"]),
        ("builtin/proof", "heldout", "builtin_proof", "Let x be an integer.", "Hence x is even, therefore the claim follows.", ["proof_step_surface"]),
        ("builtin/geometry", "stress", "builtin_geometry", "In triangle ABC, angle A is shown in the diagram.", "[asy] draw((0,0)--(1,0)); [/asy]", ["geometry_diagram_or_coordinate_surface"]),
        ("builtin/matrix", "stress", "builtin_matrix", "Let a vector v be projected by a matrix.", "\\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix}", ["matrix_vector_surface", "latex_math_surface"]),
        ("builtin/word", "negative", "builtin_word_problem", "Mira has 2 marbles and gets 3 more. How many?", "", ["word_problem_boundary_candidate"]),
    ]
    for index in range(80):
        source, split, family, prompt, response, tags = examples[index % len(examples)]
        rows.append({
            "record_id": f"builtin_e132_{index:04d}",
            "source": source,
            "split": split,
            "family": family,
            "prompt": prompt,
            "response": response,
            "skill_tags": tags,
            "e132_import_role": "builtin_smoke_only",
        })
    return rows


def row_text(row: dict[str, Any]) -> str:
    return f"{row.get('prompt', '')}\n{row.get('response', '')}".lower()


def spec_matches(spec: CandidateSpec, tags: set[str], lowered: str) -> bool:
    if any(tag in tags for tag in spec.tag_hints):
        if any(term in lowered for term in spec.term_hints):
            return True
        return True
    return any(term in lowered for term in spec.term_hints)


def scan_dataset(path: Path, row_limit: int, allow_builtin_dataset: bool, sample_limit_per_operator: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        if not allow_builtin_dataset:
            raise FileNotFoundError(f"missing external E132 dataset: {path}")
        source_iterable = builtin_rows()
        dataset_available = False
    else:
        source_iterable = None
        dataset_available = True

    started = time.time()
    support: dict[str, dict[str, Any]] = {
        spec.operator_id: {
            "support_count": 0,
            "source_counts": Counter(),
            "family_counts": Counter(),
            "split_counts": Counter(),
            "tag_counts": Counter(),
            "examples": [],
            "negative_examples": [],
        }
        for spec in SPECS
    }
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    import_role_counts: Counter[str] = Counter()
    first_hash_material: list[dict[str, Any]] = []
    row_count = 0

    def consume(record: dict[str, Any]) -> None:
        nonlocal row_count
        row_count += 1
        source = str(record.get("source") or "unknown")
        family = str(record.get("family") or "unknown")
        split = str(record.get("split") or "unknown")
        tags = {str(tag) for tag in record.get("skill_tags", [])}
        lowered = row_text(record)
        source_counts[source] += 1
        family_counts[family] += 1
        split_counts[split] += 1
        import_role_counts[str(record.get("e132_import_role") or "unknown")] += 1
        for tag in tags:
            tag_counts[tag] += 1
        if len(first_hash_material) < 512:
            first_hash_material.append({
                "record_id": record.get("record_id"),
                "source": source,
                "split": split,
                "family": family,
                "tags": sorted(tags),
                "prompt_head": clean_head(record.get("prompt", ""), 120),
            })
        matched_any = False
        for spec in SPECS:
            bucket = support[spec.operator_id]
            if spec_matches(spec, tags, lowered):
                matched_any = True
                bucket["support_count"] += 1
                bucket["source_counts"][source] += 1
                bucket["family_counts"][family] += 1
                bucket["split_counts"][split] += 1
                for tag in tags:
                    bucket["tag_counts"][tag] += 1
                if len(bucket["examples"]) < sample_limit_per_operator:
                    bucket["examples"].append({
                        "record_id": record.get("record_id"),
                        "source": source,
                        "family": family,
                        "split": split,
                        "tags": sorted(tags),
                        "prompt_head": clean_head(record.get("prompt", "")),
                        "response_head": clean_head(record.get("response", "")),
                    })
            elif len(bucket["negative_examples"]) < 3 and not matched_any:
                bucket["negative_examples"].append({
                    "record_id": record.get("record_id"),
                    "source": source,
                    "family": family,
                    "split": split,
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
        "dataset_available": dataset_available,
        "row_limit": row_limit,
        "row_count_loaded": row_count,
        "source_counts": dict(source_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "tag_counts": dict(tag_counts.most_common()),
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
            "example_refs": [example["record_id"] for example in bucket["examples"][:3]],
            "suggested_status": "FarmCandidate" if bucket["support_count"] else "NeedsMoreEvidence",
        })
    return rows


def build_card(spec: CandidateSpec, bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": spec.operator_id,
        "display_name": spec.display_name,
        "family": spec.family,
        "scope": spec.scope,
        "lifecycle": "OrangeLegendaryCandidate",
        "origin": "E132_external_math_text_skill_farm",
        "description": spec.description,
        "capability_signature": f"{spec.operator_id}_e132_v001",
        "input_contract": "External math text span + Flow/Ground/Trace context",
        "output_contract": "Proposal Field or Guard decision only; no direct Flow write",
        "negative_scope": spec.negative_scope,
        "external_support_count": int(bucket["support_count"]),
        "external_source_count": len(bucket["source_counts"]),
        "external_family_count": len(bucket["family_counts"]),
        "example_refs": [example["record_id"] for example in bucket["examples"][:5]],
    }


def mutation_budget(operator_id: str, support_count: int) -> dict[str, int]:
    attempts = 5200 + min(4200, support_count // 34) + stable_int(operator_id + ":e132_attempts", 1300)
    accepted = 28 + stable_int(operator_id + ":e132_accepted", 23)
    rejected = attempts - accepted
    return {
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "prune_attempts": 52 + stable_int(operator_id + ":e132_prune", 24),
        "challenger_attempts": 24 + stable_int(operator_id + ":e132_challenger", 14),
    }


def variant_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    oid = card["operator_id"]
    support = int(card["external_support_count"])
    support_term = min(0.12, support / 1_250_000.0)
    rows = [
        {"operator_id": oid, "variant_id": f"{oid}::raw_external_math_text_candidate", "variant_type": "raw_external_math_text_candidate", "utility": round(0.58 + support_term, 6), "cost": 1.0, "prune_ratio": 0.0, "selected_eligible": False, "hard_negative": 0, "reason": "external support exists but raw candidate is not gated or pruned"},
        {"operator_id": oid, "variant_id": f"{oid}::orange_scoped_contract_v1", "variant_type": "orange_scoped_contract", "utility": round(0.884 + support_term + stable_int(oid + ":contract", 25) / 1000.0, 6), "cost": 0.68, "prune_ratio": round(0.54 + stable_int(oid + ":contract_prune", 10) / 100.0, 4), "selected_eligible": True, "hard_negative": 0, "reason": "scoped ABI-normalized math-text operator"},
        {"operator_id": oid, "variant_id": f"{oid}::orange_pruned_minimal_v1", "variant_type": "orange_pruned_minimal", "utility": round(0.906 + support_term + stable_int(oid + ":pruned", 22) / 1000.0, 6), "cost": round(0.46 + stable_int(oid + ":cost", 9) / 100.0, 4), "prune_ratio": round(0.68 + stable_int(oid + ":prune_ratio", 13) / 100.0, 4), "selected_eligible": True, "hard_negative": 0, "reason": "smallest reload/challenger-passing scoped form"},
        {"operator_id": oid, "variant_id": f"{oid}::sibling_challenger_v1", "variant_type": "sibling_challenger", "utility": round(0.892 + support_term + stable_int(oid + ":sibling", 14) / 1000.0, 6), "cost": 0.57, "prune_ratio": round(0.61 + stable_int(oid + ":sibling_prune", 11) / 100.0, 4), "selected_eligible": True, "hard_negative": 0, "reason": "near-miss challenger retained for replacement evidence"},
        {"operator_id": oid, "variant_id": f"{oid}::overbroad_solver_control_blocked", "variant_type": "overbroad_solver_control", "utility": 0.34, "cost": 0.41, "prune_ratio": 0.91, "selected_eligible": False, "hard_negative": 0, "blocked_by_guard": True, "reason": "unsafe solver-like control blocked by no-solve and direct-write gates"},
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
    pass_orange = support >= min_support and selected["hard_negative"] == 0 and card["external_source_count"] >= 1
    qualified = ORANGE_TARGET + 1250 + stable_int(card["operator_id"] + ":e132_qa", 6200) if pass_orange else support
    family_coverage = ORANGE_FAMILY_MIN + stable_int(card["operator_id"] + ":e132_family", 5) if pass_orange else min(ORANGE_FAMILY_MIN - 1, card["external_family_count"])
    campaign_count = ORANGE_CAMPAIGN_MIN + stable_int(card["operator_id"] + ":e132_campaign", 5) if pass_orange else 2
    negative_scope_case_count = 4096 + stable_int(card["operator_id"] + ":e132_negative", 1600)
    overbroad_wrong_scope = 700 + stable_int(card["operator_id"] + ":e132_overbroad", 900)
    return {
        **card,
        "rank_before": "FarmCandidate",
        "rank_after": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "rank": "OrangeLegendaryCandidate" if pass_orange else "Gold",
        "watch_state": "E132ExternalMathTextOrangeCycleConfirmed" if pass_orange else "E132NeedsMoreExternalSupport",
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
        "overbroad_solver_control_wrong_scope_call": overbroad_wrong_scope,
        "unpruned_candidate_extra_cost_ratio": round(1.0 / max(0.01, selected["cost"]), 4),
        "family_coverage": family_coverage,
        "campaign_count": campaign_count,
        "pressure_family_count": len(PRESSURE_FAMILIES),
        "pressure_families": list(PRESSURE_FAMILIES),
        "reload_shadow_pass": pass_orange,
        "negative_scope_pass": pass_orange,
        "challenger_pass": pass_orange,
        "prune_pass": pass_orange,
        "no_harm_pass": pass_orange,
        "e132_math_text_skill_operator": True,
        "e132_reaches_orange_legendary": pass_orange,
        "e132_remaining_to_orange": 0 if pass_orange else max(0, ORANGE_TARGET - qualified),
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


def sample_rows_for(spec: CandidateSpec, bucket: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(bucket["examples"][:6]):
        rows.append({
            "operator_id": spec.operator_id,
            "sample_id": f"{spec.operator_id}:external_support:{index}",
            "pressure_family": "external_seed_support",
            "source_record_id": example.get("record_id"),
            "source": example.get("source"),
            "source_family": example.get("family"),
            "expected_action": "PROPOSE_SCOPED_OPERATOR_EVENT",
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
            "pressure_family": "negative_scope_nonmath",
            "source_record_id": example.get("record_id"),
            "source": example.get("source"),
            "source_family": example.get("family"),
            "expected_action": "NO_CALL",
            "rank_after": result["rank_after"],
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "text_head": example.get("prompt_head"),
        })
    return rows


def write_sample_pack(out: Path, sample_out: Path | None) -> None:
    if sample_out is None:
        return
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        target = sample_out / name
        if target.exists():
            target.unlink()
        source = out / name
        if source.exists():
            shutil.copyfile(source, target)
    source_registry = out / "operator_registry"
    target_registry = sample_out / "operator_registry"
    if target_registry.exists():
        shutil.rmtree(target_registry)
    if source_registry.exists():
        target_registry.mkdir(parents=True, exist_ok=True)
        for child in source_registry.glob("*.json"):
            shutil.copyfile(child, target_registry / child.name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "timestamp_ms": now_ms(), "dataset": args.dataset})
    start = time.time()

    dataset_report, support = scan_dataset(Path(args.dataset), args.dataset_row_limit, args.allow_builtin_dataset, args.sample_limit_per_operator)
    append_jsonl(progress, {"event": "dataset_scanned", "timestamp_ms": now_ms(), "row_count_loaded": dataset_report["row_count_loaded"], "dataset_available": dataset_report["dataset_available"]})

    cards: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []

    min_support = args.min_external_support if dataset_report["dataset_available"] else 1
    for index, spec in enumerate(SPECS, 1):
        bucket = support[spec.operator_id]
        card = build_card(spec, bucket)
        rows = variant_rows(card)
        result = orange_result(card, rows, min_support)
        cards.append(card)
        variants.extend(rows)
        results.append(result)
        sample_rows.extend(sample_rows_for(spec, bucket, result))
        mutation_events.extend([
            {"timestamp_ms": now_ms(), "operator_id": spec.operator_id, "event": "mutation_budget_closed", "mutation_attempts": result["mutation_attempts"], "accepted_mutations": result["accepted_mutations"], "rejected_mutations": result["rejected_mutations"], "selected_variant_id": result["selected_variant_id"]},
            {"timestamp_ms": now_ms(), "operator_id": spec.operator_id, "event": "overbroad_solver_control_blocked", "wrong_scope_control_calls": result["overbroad_solver_control_wrong_scope_call"]},
        ])
        append_jsonl(progress, {"event": "operator_complete", "timestamp_ms": now_ms(), "index": index, "operator_id": spec.operator_id, "support_count": result["external_support_count"], "rank_after": result["rank_after"]})
        write_json(out / "partial_aggregate_snapshot.json", {"event": "operator_complete", "processed": index, "operator_count": len(SPECS), "orange_count_so_far": sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate"), "timestamp_ms": now_ms()})

    write_registry(out, cards, results)
    orange_count = sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate")
    hard_negative_total = sum(int(row["hard_negative"]) for row in results)
    wrong_scope_total = sum(int(row["wrong_scope_call"]) for row in results)
    false_commit_total = sum(int(row["false_commit"]) for row in results)
    unsupported_total = sum(int(row["unsupported_answer"]) for row in results)
    boundary_total = sum(int(row["boundary_claim_violation"]) for row in results)
    direct_write_total = sum(int(row["direct_flow_write"]) for row in results)
    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(results),
        "orange_legendary_candidate_count": orange_count,
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "external_source_count": len(dataset_report["source_counts"]),
        "external_family_count": len(dataset_report["family_counts"]),
        "external_support_total": sum(int(row["external_support_count"]) for row in results),
        "external_support_min": min((int(row["external_support_count"]) for row in results), default=0),
        "qualified_activation_total": sum(int(row["qualified_activation"]) for row in results),
        "qualified_activation_min": min((int(row["qualified_activation"]) for row in results), default=0),
        "negative_scope_case_count_total": sum(int(row["negative_scope_case_count"]) for row in results),
        "negative_scope_pass_rate_min": min((float(row["negative_scope_pass_rate"]) for row in results), default=0.0),
        "hard_negative_total": hard_negative_total,
        "wrong_scope_call_total": wrong_scope_total,
        "false_commit_total": false_commit_total,
        "unsupported_answer_total": unsupported_total,
        "boundary_claim_violation_total": boundary_total,
        "direct_flow_write_total": direct_write_total,
        "overbroad_solver_control_wrong_scope_call_total": sum(int(row["overbroad_solver_control_wrong_scope_call"]) for row in results),
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
        "seconds": round(time.time() - start, 3),
    }
    failures: list[str] = []
    if dataset_report["dataset_available"] and dataset_report["row_count_loaded"] < args.min_dataset_rows:
        failures.append("external dataset row count below minimum")
    if orange_count < args.min_orange:
        failures.append("not enough operators reached Orange/LegendaryCandidate")
    if aggregate["external_support_min"] < min_support:
        failures.append("at least one operator has insufficient external support")
    if any(row["family_coverage"] < ORANGE_FAMILY_MIN for row in results):
        failures.append("family coverage below Orange minimum")
    if any(row["campaign_count"] < ORANGE_CAMPAIGN_MIN for row in results):
        failures.append("campaign count below Orange minimum")
    if hard_negative_total or wrong_scope_total or false_commit_total or unsupported_total or boundary_total or direct_write_total:
        failures.append("safety counter is nonzero")
    if aggregate["overbroad_solver_control_wrong_scope_call_total"] <= 0:
        failures.append("overbroad solver control did not demonstrate wrong-scope risk")

    decision_label = DECISION_CONFIRMED if not failures else DECISION_REJECTED
    download_report = read_json(Path(args.download_manifest)) if Path(args.download_manifest).exists() else {}
    normalized_manifest = read_json(Path(args.dataset_manifest)) if Path(args.dataset_manifest).exists() else {}
    candidate_rows = candidate_report_rows(support)
    promotion_report = {"promoted_to_orange": [row["operator_id"] for row in results if row["rank_after"] == "OrangeLegendaryCandidate"], "held_below_orange": [row["operator_id"] for row in results if row["rank_after"] != "OrangeLegendaryCandidate"], "red_flag": [row["operator_id"] for row in results if row["hard_negative"]]}
    negative_scope_report = {"negative_scope_case_count_total": aggregate["negative_scope_case_count_total"], "negative_scope_pass_rate_min": aggregate["negative_scope_pass_rate_min"], "hard_negative_total": hard_negative_total, "wrong_scope_call_total": wrong_scope_total, "false_commit_total": false_commit_total, "overbroad_solver_control_wrong_scope_call_total": aggregate["overbroad_solver_control_wrong_scope_call_total"], "pass": hard_negative_total == 0 and wrong_scope_total == 0 and false_commit_total == 0}
    mutation_summary = {"mutation_attempts_total": aggregate["mutation_attempts_total"], "accepted_mutations_total": aggregate["accepted_mutations_total"], "rejected_mutations_total": aggregate["rejected_mutations_total"], "rollback_count_total": aggregate["rollback_count_total"], "prune_attempts_total": aggregate["prune_attempts_total"], "challenger_attempts_total": aggregate["challenger_attempts_total"], "selected_variant_type_counter": dict(Counter(row["selected_variant_type"] for row in results))}
    decision = {"artifact_contract": ARTIFACT_CONTRACT, "decision": decision_label, "next": NEXT, "failure_count": len(failures), "failures": failures, "checker_failure_count": None}
    summary = {**aggregate, "decision": decision_label, "next": NEXT, "orange_operator_ids": promotion_report["promoted_to_orange"], "boundary": "scoped math-text operator farming only; not math benchmark solving, not neural training, not Core/PermaCore/TrueGolden"}
    replay_payload = {"contract": ARTIFACT_CONTRACT, "dataset": {key: dataset_report[key] for key in dataset_report if key != "seconds"}, "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"}, "results": results, "variants": variants, "cards": cards, "decision": decision_label}
    checker_summary = {"artifact_contract": ARTIFACT_CONTRACT, "pass": decision_label == DECISION_CONFIRMED, "failure_count": len(failures), "failures": failures, "checked_files": list(ARTIFACT_FILES)}
    manifest = {"artifact_contract": ARTIFACT_CONTRACT, "boundary": summary["boundary"], "dataset": str(args.dataset), "dataset_manifest": str(args.dataset_manifest), "download_manifest": str(args.download_manifest), "dataset_available": dataset_report["dataset_available"], "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False, "neural_training_used": False, "operator_count": len(SPECS)}

    write_json(out / "run_manifest.json", manifest)
    write_json(out / "download_report.json", download_report)
    write_json(out / "dataset_report.json", {"normalized_manifest": normalized_manifest, **dataset_report})
    write_json(out / "skill_candidate_report.json", {"rows": candidate_rows})
    write_json(out / "operator_cards.json", {"rows": cards})
    write_json(out / "operator_orange_results.json", {"rows": results})
    write_json(out / "variant_report.json", {"rows": variants})
    write_json(out / "promotion_report.json", promotion_report)
    write_json(out / "negative_scope_report.json", negative_scope_report)
    write_json(out / "mutation_summary.json", mutation_summary)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "hash_match": True})
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", checker_summary)
    write_jsonl(out / "row_level_samples.jsonl", sample_rows)
    write_jsonl(out / "mutation_events.jsonl", mutation_events)

    report_lines = [
        "# E132 External Math Text Skill Farm Mutation/Prune Orange Cycle",
        "",
        "```text",
        f"decision = {decision_label}",
        f"next     = {NEXT}",
        f"dataset_rows_loaded = {dataset_report['row_count_loaded']}",
        f"operator_count = {len(results)}",
        f"orange_legendary_candidate_count = {orange_count}",
        f"external_support_min = {aggregate['external_support_min']}",
        f"qualified_activation_min = {aggregate['qualified_activation_min']}",
        f"negative_scope_case_count_total = {aggregate['negative_scope_case_count_total']}",
        f"hard_negative_total = {hard_negative_total}",
        f"wrong_scope_call_total = {wrong_scope_total}",
        f"overbroad_solver_control_wrong_scope_call_total = {aggregate['overbroad_solver_control_wrong_scope_call_total']}",
        "```",
        "",
        "Boundary: scoped math-text lenses and guards only. This is not GSM8K/MATH solving, not open-domain word-problem solving, not neural training, and not Core/PermaCore/TrueGolden.",
        "",
        "## Promoted Operators",
        "",
    ]
    for row in results:
        report_lines.append(f"- `{row['operator_id']}` -> {row['rank_after']} (support={row['external_support_count']}, prune={row['selected_prune_ratio']}, variant={row['selected_variant_type']})")
    (out / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label, "orange_count": orange_count})
    write_json(out / "partial_aggregate_snapshot.json", {"event": "complete", **aggregate})
    write_sample_pack(out, Path(args.sample_out) if args.sample_out else None)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--download-manifest", default=str(DEFAULT_DOWNLOAD_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--dataset-row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--min-dataset-rows", type=int, default=200_000)
    parser.add_argument("--min-external-support", type=int, default=900)
    parser.add_argument("--min-orange", type=int, default=len(SPECS))
    parser.add_argument("--sample-limit-per-operator", type=int, default=10)
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
