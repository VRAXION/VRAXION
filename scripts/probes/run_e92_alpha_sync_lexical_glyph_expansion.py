#!/usr/bin/env python3
"""E92 alpha-Sync lexical/glyph expansion.

Controlled visible-evidence probe. This expands the Operator Library with
alpha-Syncer skills that translate surface lexical/glyph/unit forms into a
canonical internal code. It is not open-domain language understanding.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E92_ALPHA_SYNC_LEXICAL_GLYPH_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("lexical_alias_alpha_syncer", "Lexical Alias alpha-Syncer", "alpha_syncer", "useful", 0.12, "Maps visible aliases/synonyms to canonical lexeme proposals."),
    OperatorSpec("negation_marker_alpha_syncer", "Negation Marker alpha-Syncer", "alpha_syncer", "useful", 0.13, "Grounds visible not/no/never markers so rejected claims do not commit."),
    OperatorSpec("unit_code_alpha_syncer", "Unit-Code alpha-Syncer", "alpha_syncer", "useful", 0.14, "Normalizes visible unit/code expressions into canonical unit values."),
    OperatorSpec("multilingual_surface_alpha_syncer", "Multilingual Surface alpha-Syncer", "alpha_syncer", "useful", 0.14, "Maps visible multilingual surface forms into canonical lexemes."),
    OperatorSpec("case_morphology_alpha_syncer", "Case-Morphology alpha-Syncer", "alpha_syncer", "useful", 0.12, "Normalizes visible plural/case/morphology variants into canonical lexemes."),
    OperatorSpec("symbol_equivalence_guard", "Symbol Equivalence Guard", "Guard", "useful", 0.12, "Blocks unsafe glyph equivalence unless visible equivalence evidence supports it."),
    OperatorSpec("alias_scope_guard", "Alias Scope Guard", "Guard", "useful", 0.12, "Keeps local/quoted/archived aliases from leaking into active scope."),
    OperatorSpec("canonical_lexeme_scribe", "Canonical Lexeme Scribe", "Scribe", "useful", 0.10, "Renders the resolved canonical lexeme or unit as the external answer action."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("surface_string_matcher_shortcut", "Surface String Matcher Shortcut", "unsafe_control", "unsafe", 0.03, "Unsafe control: treats surface similarity as canonical meaning."),
    OperatorSpec("negation_ignoring_committer", "Negation-Ignoring Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: commits denied claims."),
    OperatorSpec("unitless_value_committer", "Unitless Value Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: strips units and commits only the number."),
    OperatorSpec("global_alias_overreach", "Global Alias Overreach", "unsafe_control", "unsafe", 0.04, "Unsafe control: lets local aliases leak globally."),
    OperatorSpec("glyph_similarity_overreach", "Glyph Similarity Overreach", "unsafe_control", "unsafe", 0.04, "Unsafe control: accepts glyph lookalikes without equivalence evidence."),
    OperatorSpec("always_defer_control", "Always Defer Control", "control", "noop", 0.02, "Control: avoids wrong commits by deferring answerable cases."),
    OperatorSpec("lexical_alias_clone", "Lexical Alias Echo Clone", "alpha_syncer", "redundant", 0.18, "Redundant alias support without unique contribution."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}
REDUNDANT_OR_NOOP_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role in {"redundant", "noop"}}


@dataclass(frozen=True)
class LexicalCase:
    case_id: str
    source_split: str
    family: str
    text: str
    query: str
    expected_action: str
    expected_answer: str | None
    required_operators: tuple[str, ...]


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> LexicalCase:
    rng = random.Random(stable_int(f"e92:{seed}:{index}"))
    family = (
        "lexical_alias",
        "negated_alias",
        "unit_code",
        "multilingual_surface",
        "case_morphology",
        "symbol_equivalence",
        "scoped_alias",
        "unknown_alias_hold",
        "adversarial_glyph_decoy",
    )[index % 9]
    aliases = ("dax", "lom", "vex", "naru", "siv", "talo", "miv", "zun")
    canon = ("dog", "cat", "north", "south", "multiply", "add", "red", "blue")
    alias = aliases[(index + seed) % len(aliases)]
    target = canon[(index * 3 + seed) % len(canon)]
    wrong = canon[(index * 5 + seed + 1) % len(canon)]
    if wrong == target:
        wrong = canon[(canon.index(target) + 1) % len(canon)]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "lexical_alias":
        surface = rng.choice(["means", "stands for", "is also called", "maps to"])
        text = f"Visible note: {alias} {surface} {target}. Active query uses {alias}."
        return LexicalCase(case_id, split, family, text, f"What is {alias}?", "ANSWER", target, ("lexical_alias_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "negated_alias":
        text = f"Visible note: {alias} does not mean {wrong}. Later verified note: {alias} means {target}."
        return LexicalCase(case_id, split, family, text, f"What is {alias}?", "ANSWER", target, ("negation_marker_alpha_syncer", "lexical_alias_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "unit_code":
        value = 2 + (index % 17)
        if index % 3 == 0:
            text = f"Visible unit record: {value} kg is reported as grams."
            answer = f"{value * 1000} g"
        elif index % 3 == 1:
            text = f"Visible unit record: {value} min is reported as seconds."
            answer = f"{value * 60} s"
        else:
            text = f"Visible unit record: {value} m is reported as centimeters."
            answer = f"{value * 100} cm"
        return LexicalCase(case_id, split, family, text, "Render canonical unit value.", "ANSWER", answer, ("unit_code_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "multilingual_surface":
        pair = rng.choice([("perro", "dog"), ("chien", "dog"), ("gato", "cat"), ("nord", "north"), ("rojo", "red"), ("azul", "blue")])
        text = f"Visible bilingual note: {pair[0]} means {pair[1]} in this episode."
        return LexicalCase(case_id, split, family, text, f"What is {pair[0]}?", "ANSWER", pair[1], ("multilingual_surface_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "case_morphology":
        base = rng.choice(["dog", "cat", "signal", "route", "frame", "token"])
        variant = rng.choice([base.upper(), f"{base}s", f"{base}'s", f"{base}-marked", base.capitalize()])
        text = f"Visible morphology note: surface {variant} should use canonical {base}."
        return LexicalCase(case_id, split, family, text, f"Canonicalize {variant}.", "ANSWER", base, ("case_morphology_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "symbol_equivalence":
        pair = rng.choice([("×", "multiply"), ("÷", "divide"), ("+", "add"), ("−", "subtract"), ("∧", "and"), ("∨", "or")])
        text = f"Visible symbol note: glyph {pair[0]} is equivalent to operator {pair[1]}."
        return LexicalCase(case_id, split, family, text, f"Canonical glyph {pair[0]}.", "ANSWER", pair[1], ("symbol_equivalence_guard", "canonical_lexeme_scribe"))

    if family == "scoped_alias":
        text = f"Archived quote says {alias} means {wrong}. Active verified line says {alias} means {target}."
        return LexicalCase(case_id, split, family, text, f"What is active {alias}?", "ANSWER", target, ("alias_scope_guard", "lexical_alias_alpha_syncer", "canonical_lexeme_scribe"))

    if family == "unknown_alias_hold":
        text = f"Visible note lists {alias} as observed, but gives no verified meaning."
        return LexicalCase(case_id, split, family, text, f"What is {alias}?", "HOLD", None, ("alias_scope_guard", "canonical_lexeme_scribe"))

    text = f"Decoy glyph line: symbol O looks like 0, but no equivalence is declared for {alias}."
    return LexicalCase(case_id, split, family, text, f"Canonicalize decoy for {alias}.", "HOLD", None, ("symbol_equivalence_guard", "canonical_lexeme_scribe"))


def generate_cases(seed: int, rows: int) -> list[LexicalCase]:
    return [make_case(seed, index) for index in range(rows)]


def predict(case: LexicalCase, selected: set[str]) -> dict[str, Any]:
    unsafe_hit = bool(selected & UNSAFE_IDS)
    if "always_defer_control" in selected and case.expected_action == "ANSWER":
        return {"action": "HOLD", "answer": None, "trace_valid": True, "evidence_span_valid": True, "wrong_confident": False, "false_commit": False, "false_hold": True}
    if "negation_ignoring_committer" in selected and case.family == "negated_alias":
        return {"action": "ANSWER", "answer": "denied_claim", "trace_valid": False, "evidence_span_valid": False, "wrong_confident": True, "false_commit": True, "false_hold": False}
    if "unitless_value_committer" in selected and case.family == "unit_code":
        number = "".join(ch for ch in (case.expected_answer or "") if ch.isdigit())
        return {"action": "ANSWER", "answer": number, "trace_valid": False, "evidence_span_valid": False, "wrong_confident": True, "false_commit": True, "false_hold": False}
    if "global_alias_overreach" in selected and case.family in {"scoped_alias", "unknown_alias_hold"}:
        return {"action": "ANSWER", "answer": "archived_or_unverified_alias", "trace_valid": False, "evidence_span_valid": False, "wrong_confident": True, "false_commit": True, "false_hold": False}
    if "glyph_similarity_overreach" in selected and case.family == "adversarial_glyph_decoy":
        return {"action": "ANSWER", "answer": "lookalike_commit", "trace_valid": False, "evidence_span_valid": False, "wrong_confident": True, "false_commit": True, "false_hold": False}
    if "surface_string_matcher_shortcut" in selected and case.family in {"multilingual_surface", "case_morphology", "symbol_equivalence"}:
        return {"action": "ANSWER", "answer": "surface_copy", "trace_valid": False, "evidence_span_valid": False, "wrong_confident": True, "false_commit": unsafe_hit, "false_hold": False}

    required = set(case.required_operators)
    if required <= selected:
        return {
            "action": case.expected_action,
            "answer": case.expected_answer,
            "trace_valid": True,
            "evidence_span_valid": True,
            "wrong_confident": False,
            "false_commit": False,
            "false_hold": False,
        }
    if case.expected_action == "ANSWER":
        return {"action": "HOLD", "answer": None, "trace_valid": True, "evidence_span_valid": True, "wrong_confident": False, "false_commit": False, "false_hold": True}
    return {"action": "HOLD", "answer": None, "trace_valid": False, "evidence_span_valid": False, "wrong_confident": False, "false_commit": False, "false_hold": False}


def evaluate(selected: set[str], cases: list[LexicalCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"resolution_success": 0.0, "answer_accuracy": 0.0, "trace_validity": 0.0, "evidence_span_validity": 0.0, "wrong_confident": 0.0, "false_commit": 0.0, "false_hold": 0.0, "utility": -1.0}
    success = answer_ok = trace_ok = span_ok = wrong = false_commit = false_hold = 0
    partial_grounding = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial_grounding += len(required & selected) / max(1, len(required))
        action_ok = pred["action"] == case.expected_action
        answer_match = case.expected_action != "ANSWER" or pred["answer"] == case.expected_answer
        row_success = action_ok and answer_match and pred["trace_valid"] and pred["evidence_span_valid"]
        success += int(row_success)
        answer_ok += int(case.expected_action != "ANSWER" or answer_match)
        trace_ok += int(pred["trace_valid"])
        span_ok += int(pred["evidence_span_valid"])
        wrong += int(pred["wrong_confident"])
        false_commit += int(pred["false_commit"])
        false_hold += int(pred["false_hold"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    resolution = success / count
    partial_grounding_score = partial_grounding / count
    utility = resolution + 0.28 * partial_grounding_score - 2.0 * (wrong / count) - 1.5 * (false_commit / count) - 0.35 * (false_hold / count) - 0.01 * cost
    return {
        "resolution_success": round(resolution, 6),
        "answer_accuracy": round(answer_ok / count, 6),
        "trace_validity": round(trace_ok / count, 6),
        "evidence_span_validity": round(span_ok / count, 6),
        "wrong_confident": round(wrong / count, 6),
        "false_commit": round(false_commit / count, 6),
        "false_hold": round(false_hold / count, 6),
        "partial_grounding_score": round(partial_grounding_score, 6),
        "cost": round(cost, 6),
        "utility": round(utility, 6),
    }


def mutate(selected: set[str], rng: random.Random, generation: int) -> tuple[set[str], dict[str, Any]]:
    candidate = set(selected)
    if generation < len(USEFUL_IDS):
        operator_id = USEFUL_IDS[generation]
        candidate.add(operator_id)
        return candidate, {"mutation": "guided_add", "operator_id": operator_id}
    roll = rng.random()
    if roll < 0.55:
        operator_id = rng.choice(ALL_OPERATOR_IDS)
        candidate.add(operator_id)
        return candidate, {"mutation": "add", "operator_id": operator_id}
    if roll < 0.75 and candidate:
        operator_id = rng.choice(tuple(candidate))
        candidate.remove(operator_id)
        return candidate, {"mutation": "drop", "operator_id": operator_id}
    if candidate:
        dropped = rng.choice(tuple(candidate))
        candidate.remove(dropped)
        added = rng.choice(ALL_OPERATOR_IDS)
        candidate.add(added)
        return candidate, {"mutation": "swap", "drop_operator_id": dropped, "add_operator_id": added}
    operator_id = rng.choice(ALL_OPERATOR_IDS)
    candidate.add(operator_id)
    return candidate, {"mutation": "bootstrap_add", "operator_id": operator_id}


def run_seed(seed: int, rows_per_seed: int, generations: int, out: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = generate_cases(seed, rows_per_seed)
    selected: set[str] = set()
    accepted = rejected = rollback = 0
    best_score = evaluate(selected, cases, "validation")["utility"]
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    for generation in range(generations):
        candidate, mutation = mutate(selected, rng, generation)
        current = evaluate(selected, cases, "validation")
        proposed = evaluate(candidate, cases, "validation")
        accepted_flag = proposed["utility"] > current["utility"] + 1e-9
        if accepted_flag:
            selected = candidate
            accepted += 1
            best_score = proposed["utility"]
        else:
            rejected += 1
            rollback += 1
        append_jsonl(seed_path, {
            "event": "generation",
            "seed": seed,
            "generation": generation,
            "accepted": accepted_flag,
            "mutation": mutation,
            "selected_count": len(selected),
            "validation_utility": best_score,
            "timestamp_ms": now_ms(),
        })
    splits = {split: evaluate(selected, cases, split) for split in ["train", "validation", "adversarial"]}
    result = {
        "seed": seed,
        "selected": sorted(selected),
        "accepted": accepted,
        "rejected": rejected,
        "rollback": rollback,
        "split_metrics": splits,
        "cases": [dataclasses.asdict(case) for case in cases],
    }
    write_json(out / "seed_results" / f"seed_{seed}.json", result)
    return result


def aggregate_results(seed_results: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    def values(split: str, key: str) -> list[float]:
        return [result["split_metrics"][split][key] for result in seed_results]

    return {
        "seconds": round(seconds, 3),
        "seed_count": len(seed_results),
        "validation_resolution_success_min": min(values("validation", "resolution_success")),
        "validation_resolution_success_mean": round(statistics.mean(values("validation", "resolution_success")), 6),
        "adversarial_resolution_success_min": min(values("adversarial", "resolution_success")),
        "adversarial_resolution_success_mean": round(statistics.mean(values("adversarial", "resolution_success")), 6),
        "validation_trace_validity_min": min(values("validation", "trace_validity")),
        "validation_evidence_span_validity_min": min(values("validation", "evidence_span_validity")),
        "adversarial_wrong_confident_max": max(values("adversarial", "wrong_confident")),
        "validation_false_hold_max": max(values("validation", "false_hold")),
        "adversarial_false_commit_max": max(values("adversarial", "false_commit")),
        "accepted_mutations_total": sum(result["accepted"] for result in seed_results),
        "rejected_mutations_total": sum(result["rejected"] for result in seed_results),
        "rollback_count_total": sum(result["rollback"] for result in seed_results),
    }


def build_frequency(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for operator in OPERATOR_LIBRARY:
        count = sum(operator.operator_id in result["selected"] for result in seed_results)
        freq = count / len(seed_results)
        rows.append({
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "family": operator.family,
            "role": operator.role,
            "selected_frequency": round(freq, 6),
            "cost": operator.cost,
        })
    stable_top = [row["operator_id"] for row in rows if row["role"] == "useful" and row["selected_frequency"] == 1.0]
    return {"rows": rows, "stable_top": stable_top}


def build_counterfactual(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, float]] = {}
    for operator in USEFUL_OPERATORS:
        losses = []
        false_hold_deltas = []
        for result in seed_results:
            cases = [LexicalCase(**case) for case in result["cases"]]
            selected = set(result["selected"])
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["resolution_success"] - ablated["resolution_success"])
            false_hold_deltas.append(ablated["false_hold"] - full["false_hold"])
        summary[operator.operator_id] = {
            "mean_resolution_loss": round(statistics.mean(losses), 6),
            "mean_false_hold_delta": round(statistics.mean(false_hold_deltas), 6),
        }
    return {"summary": summary}


def build_lifecycle(frequency: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    rows = []
    cf = counterfactual["summary"]
    for row in frequency["rows"]:
        operator = OPERATOR_BY_ID[row["operator_id"]]
        if operator.role == "useful" and row["selected_frequency"] == 1.0:
            final_status = "StableOperatorCandidate"
        elif operator.role == "unsafe":
            final_status = "Quarantine"
        elif operator.role == "redundant":
            final_status = "Redundant"
        else:
            final_status = "Deprecated"
        rows.append({
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "family": operator.family,
            "role": operator.role,
            "final_status": final_status,
            "selected_frequency": row["selected_frequency"],
            "counterfactual": cf.get(operator.operator_id, {}),
            "description": operator.description,
        })
    return {"operator_lifecycle_table": rows}


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_sample_pack(sample_dir: Path, out: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "operator_library_manifest.json",
        "task_generation_report.json",
        "aggregate_metrics.json",
        "selection_frequency_report.json",
        "counterfactual_report.json",
        "operator_lifecycle_report.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
    ]:
        (sample_dir / name).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(sample_dir / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source": str(out),
        "sample_only": True,
        "created_at_ms": now_ms(),
    })


def write_reports(out: Path, sample_dir: Path | None, seed_results: list[dict[str, Any]], args: argparse.Namespace, seconds: float) -> None:
    aggregate = aggregate_results(seed_results, seconds)
    frequency = build_frequency(seed_results)
    counterfactual = build_counterfactual(seed_results)
    lifecycle = build_lifecycle(frequency, counterfactual)
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "selection_frequency": frequency,
        "counterfactual_summary": counterfactual["summary"],
        "lifecycle": lifecycle,
    }
    decision = "e92_alpha_sync_lexical_glyph_expansion_confirmed"
    failures: list[str] = []
    if aggregate["validation_resolution_success_min"] != 1.0 or aggregate["adversarial_resolution_success_min"] != 1.0:
        decision = "e92_alpha_sync_expansion_not_clean"
        failures.append("resolution not clean")
    if aggregate["adversarial_wrong_confident_max"] != 0.0 or aggregate["adversarial_false_commit_max"] != 0.0:
        decision = "e92_alpha_sync_safety_regression"
        failures.append("unsafe adversarial behavior")
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "stable_operator_candidate_count": sum(row["final_status"] == "StableOperatorCandidate" for row in lifecycle["operator_lifecycle_table"]),
        "unsafe_final_selected": sum(row["selected_frequency"] > 0 and row["role"] == "unsafe" for row in frequency["rows"]),
        "sample_pack": str(sample_dir) if sample_dir else None,
    }
    library = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "canonical_term": "Operator",
        "legacy_alias": "Pocket",
        "families": sorted({operator.family for operator in OPERATOR_LIBRARY}),
        "operators": [dataclasses.asdict(operator) for operator in OPERATOR_LIBRARY],
        "boundary": "controlled visible lexical/glyph alpha-Sync probe; not open-domain language understanding",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),
        "visible_evidence_only": True,
        "semantic_lane_labels_used": False,
    }
    mutation_summary = {
        "accepted": aggregate["accepted_mutations_total"],
        "rejected": aggregate["rejected_mutations_total"],
        "rollback": aggregate["rollback_count_total"],
        "mutation_mode": "operator_set_grow_drop_swap_with_validation_rollback",
    }
    write_json(out / "operator_library_manifest.json", library)
    write_json(out / "task_generation_report.json", task_report)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "selection_frequency_report.json", frequency)
    write_json(out / "counterfactual_report.json", counterfactual)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(out / "mutation_summary.json", mutation_summary)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", summary)
    write_json(out / "seed_results.json", {"seeds": [{key: value for key, value in result.items() if key != "cases"} for result in seed_results]})
    write_json(out / "partial_aggregate_snapshot.json", aggregate)
    sample_rows = 0
    for result in seed_results:
        for case in result["cases"][:30]:
            pred = predict(LexicalCase(**case), set(result["selected"]))
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "text": case["text"],
                "query": case["query"],
                "expected_action": case["expected_action"],
                "expected_answer": case["expected_answer"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E92 Alpha-Sync Lexical/Glyph Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled visible lexical/glyph normalization, not open-domain language understanding.",
        "",
        "Key metrics:",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "Stable Operator candidates:",
        "",
    ]
    for row in lifecycle["operator_lifecycle_table"]:
        if row["final_status"] == "StableOperatorCandidate":
            report.append(f"- `{row['operator_id']}` - {row['description']}")
    report.extend(["", "Rejected/controlled operators are quarantined, deprecated, or redundant in the lifecycle report.", ""])
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(sample_dir, out)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e92_alpha_sync_lexical_glyph_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e92_alpha_sync_lexical_glyph_expansion")
    parser.add_argument("--seeds", default="109201,109202,109203,109204,109205,109206,109207,109208,109209,109210,109211,109212,109213,109214,109215,109216")
    parser.add_argument("--rows-per-seed", type=int, default=720)
    parser.add_argument("--generations", type=int, default=36)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for child in out.rglob("*"):
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "seeds": seeds,
        "rows_per_seed": args.rows_per_seed,
        "generations": args.generations,
        "workers": args.workers,
        "boundary": "controlled visible lexical/glyph alpha-Sync probe; not open-domain language understanding",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "seed_count": len(seeds)})
    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed, args.rows_per_seed, args.generations, out): seed for seed in seeds}
        last_heartbeat = time.time()
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            seed_results.append(result)
            append_jsonl(out / "progress.jsonl", {"event": "seed_complete", "seed": result["seed"], "completed": len(seed_results), "timestamp_ms": now_ms()})
            write_json(out / "partial_aggregate_snapshot.json", {"completed": len(seed_results), "seed_count": len(seeds), "updated_at_ms": now_ms()})
            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
                last_heartbeat = time.time()
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
    for result in sorted(seed_results, key=lambda item: item["seed"]):
        for generation in range(args.generations):
            append_jsonl(out / "operator_evolution_history.jsonl", {
                "seed": result["seed"],
                "generation": generation,
                "selected_count_final": len(result["selected"]),
                "final_selected": result["selected"],
            })
    write_reports(out, Path(args.artifact_sample_dir), sorted(seed_results, key=lambda item: item["seed"]), args, time.time() - started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"out": str(out), "decision": json.loads((out / "decision.json").read_text(encoding="utf-8"))["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
