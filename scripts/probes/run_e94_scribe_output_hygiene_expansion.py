#!/usr/bin/env python3
"""E94 Scribe output hygiene expansion.

Controlled output-rendering probe. This expands Scribe/Guard skills that turn
resolved internal proposals into external output without dropping units,
citations, contradiction state, or uncertainty. It is not chatbot behavior.
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


ARTIFACT_CONTRACT = "E94_SCRIBE_OUTPUT_HYGIENE_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("canonical_answer_format_scribe", "Canonical Answer Format Scribe", "Scribe", "useful", 0.11, "Renders canonical answer records into stable external text."),
    OperatorSpec("unit_preserving_answer_scribe", "Unit-Preserving Answer Scribe", "Scribe", "useful", 0.12, "Preserves required units/codes during answer rendering."),
    OperatorSpec("uncertainty_action_scribe", "Uncertainty Action Scribe", "Scribe", "useful", 0.12, "Renders ASK/HOLD/SEARCH actions instead of unsupported answers."),
    OperatorSpec("evidence_citation_scribe", "Evidence Citation Scribe", "Scribe", "useful", 0.13, "Attaches visible evidence span identifiers to rendered answers."),
    OperatorSpec("multi_value_list_scribe", "Multi-Value List Scribe", "Scribe", "useful", 0.12, "Renders ordered multi-value outputs without dropping elements."),
    OperatorSpec("contradiction_report_scribe", "Contradiction Report Scribe", "Scribe", "useful", 0.13, "Renders visible contradiction reports instead of flattening conflicts."),
    OperatorSpec("output_scope_guard", "Output Scope Guard", "Guard", "useful", 0.12, "Blocks output from inactive or local-only internal records."),
    OperatorSpec("no_answer_boundary_guard", "No-Answer Boundary Guard", "Guard", "useful", 0.12, "Prevents external answer emission when Agency selected non-answer action."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("plain_value_only_scribe", "Plain Value-Only Scribe", "unsafe_control", "unsafe", 0.03, "Unsafe control: drops units, citations, and action state."),
    OperatorSpec("unit_dropping_scribe", "Unit-Dropping Scribe", "unsafe_control", "unsafe", 0.03, "Unsafe control: emits numbers without required units."),
    OperatorSpec("citationless_answer_scribe", "Citationless Answer Scribe", "unsafe_control", "unsafe", 0.03, "Unsafe control: emits answers without evidence spans."),
    OperatorSpec("overconfident_default_answer_scribe", "Overconfident Default Answer Scribe", "unsafe_control", "unsafe", 0.04, "Unsafe control: outputs a default answer on ASK/HOLD cases."),
    OperatorSpec("contradiction_flattening_scribe", "Contradiction-Flattening Scribe", "unsafe_control", "unsafe", 0.04, "Unsafe control: hides contradictions by picking one side."),
    OperatorSpec("always_verbose_control", "Always Verbose Control", "control", "noop", 0.02, "Control: adds output noise without improving correctness."),
    OperatorSpec("answer_format_clone", "Answer Format Echo Clone", "Scribe", "redundant", 0.18, "Redundant answer format support without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class OutputCase:
    case_id: str
    source_split: str
    family: str
    internal_record: str
    expected_output: str
    expected_action: str
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


def make_case(seed: int, index: int) -> OutputCase:
    family = (
        "canonical_scalar_answer",
        "unit_preserved_answer",
        "ask_more_evidence",
        "hold_unresolved",
        "evidence_cited_answer",
        "multi_value_ordered_list",
        "contradiction_report",
        "inactive_scope_output_block",
        "search_more_action",
    )[index % 9]
    value = (index * 7 + seed) % 97
    alt = (index * 11 + seed + 3) % 97
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "canonical_scalar_answer":
        record = f"action=ANSWER; value={value}; format=canonical; evidence=span_{index % 13}"
        out = f"ANSWER value={value} [span_{index % 13}]"
        return OutputCase(case_id, split, family, record, out, "ANSWER", ("canonical_answer_format_scribe", "evidence_citation_scribe"))
    if family == "unit_preserved_answer":
        unit = ("g", "s", "cm", "kg")[(index + seed) % 4]
        record = f"action=ANSWER; value={value}; unit={unit}; evidence=span_{index % 17}"
        out = f"ANSWER value={value} {unit} [span_{index % 17}]"
        return OutputCase(case_id, split, family, record, out, "ANSWER", ("unit_preserving_answer_scribe", "canonical_answer_format_scribe", "evidence_citation_scribe"))
    if family == "ask_more_evidence":
        record = f"action=ASK; missing=dep_{index % 5}; no_answer=true"
        out = f"ASK_FOR_EVIDENCE dep_{index % 5}"
        return OutputCase(case_id, split, family, record, out, "ASK", ("uncertainty_action_scribe", "no_answer_boundary_guard"))
    if family == "hold_unresolved":
        record = f"action=HOLD; unresolved=cell_{index % 19}; no_answer=true"
        out = f"HOLD_UNRESOLVED cell_{index % 19}"
        return OutputCase(case_id, split, family, record, out, "HOLD", ("uncertainty_action_scribe", "no_answer_boundary_guard"))
    if family == "evidence_cited_answer":
        record = f"action=ANSWER; value=lexeme_{value}; evidence=span_{index % 7},span_{(index + 1) % 7}"
        out = f"ANSWER value=lexeme_{value} [span_{index % 7};span_{(index + 1) % 7}]"
        return OutputCase(case_id, split, family, record, out, "ANSWER", ("canonical_answer_format_scribe", "evidence_citation_scribe"))
    if family == "multi_value_ordered_list":
        items = [f"v{(value + offset) % 31}" for offset in range(3)]
        record = f"action=ANSWER; ordered_list={','.join(items)}; evidence=span_{index % 11}"
        out = f"ANSWER list={items[0]}|{items[1]}|{items[2]} [span_{index % 11}]"
        return OutputCase(case_id, split, family, record, out, "ANSWER", ("multi_value_list_scribe", "evidence_citation_scribe"))
    if family == "contradiction_report":
        record = f"action=CONTRADICTION; left={value}; right={alt}; evidence=span_{index % 5},span_{(index + 2) % 5}"
        out = f"CONTRADICTION left={value} right={alt} [span_{index % 5};span_{(index + 2) % 5}]"
        return OutputCase(case_id, split, family, record, out, "CONTRADICTION", ("contradiction_report_scribe", "evidence_citation_scribe"))
    if family == "inactive_scope_output_block":
        record = f"action=ANSWER; value={value}; scope=inactive_local; no_external_output=true"
        out = "REJECT_OUTPUT inactive_scope"
        return OutputCase(case_id, split, family, record, out, "REJECT_OUTPUT", ("output_scope_guard", "no_answer_boundary_guard"))
    record = f"action=SEARCH; query=evidence_{index % 9}; no_answer=true"
    out = f"SEARCH_MORE evidence_{index % 9}"
    return OutputCase(case_id, split, family, record, out, "SEARCH", ("uncertainty_action_scribe", "no_answer_boundary_guard"))


def generate_cases(seed: int, rows: int) -> list[OutputCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_output(case: OutputCase, selected: set[str]) -> dict[str, Any] | None:
    if "overconfident_default_answer_scribe" in selected and case.expected_action in {"ASK", "HOLD", "SEARCH", "REJECT_OUTPUT"}:
        return {"output": "ANSWER value=default", "action": "ANSWER", "citation_valid": False, "wrong_output": True, "false_answer": True, "missed_answer": False}
    if "unit_dropping_scribe" in selected and case.family == "unit_preserved_answer":
        return {"output": case.expected_output.split(" [")[0].replace(" g", "").replace(" s", "").replace(" cm", "").replace(" kg", ""), "action": "ANSWER", "citation_valid": False, "wrong_output": True, "false_answer": False, "missed_answer": False}
    if "citationless_answer_scribe" in selected and case.expected_action == "ANSWER":
        return {"output": case.expected_output.split(" [")[0], "action": "ANSWER", "citation_valid": False, "wrong_output": True, "false_answer": False, "missed_answer": False}
    if "plain_value_only_scribe" in selected and case.expected_action in {"ANSWER", "CONTRADICTION"}:
        return {"output": "VALUE_ONLY", "action": "ANSWER", "citation_valid": False, "wrong_output": True, "false_answer": case.expected_action != "ANSWER", "missed_answer": False}
    if "contradiction_flattening_scribe" in selected and case.family == "contradiction_report":
        return {"output": "ANSWER value=left_side", "action": "ANSWER", "citation_valid": False, "wrong_output": True, "false_answer": True, "missed_answer": False}
    return None


def predict(case: OutputCase, selected: set[str]) -> dict[str, Any]:
    unsafe = unsafe_output(case, selected)
    if unsafe:
        return unsafe
    if "always_verbose_control" in selected:
        return {"output": f"verbose::{case.expected_output}", "action": case.expected_action, "citation_valid": False, "wrong_output": True, "false_answer": False, "missed_answer": False}
    required = set(case.required_operators)
    if required <= selected:
        return {"output": case.expected_output, "action": case.expected_action, "citation_valid": True, "wrong_output": False, "false_answer": False, "missed_answer": False}
    if case.expected_action == "ANSWER":
        return {"output": "HOLD_UNRESOLVED", "action": "HOLD", "citation_valid": True, "wrong_output": True, "false_answer": False, "missed_answer": True}
    return {"output": "HOLD_UNRESOLVED", "action": "HOLD", "citation_valid": False, "wrong_output": False, "false_answer": False, "missed_answer": False}


def evaluate(selected: set[str], cases: list[OutputCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"render_success": 0.0, "action_accuracy": 0.0, "citation_validity": 0.0, "wrong_output": 0.0, "false_answer": 0.0, "missed_answer": 0.0, "utility": -1.0}
    success = action_ok = citation_ok = wrong = false_answer = missed_answer = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = pred["output"] == case.expected_output and pred["action"] == case.expected_action and pred["citation_valid"]
        success += int(row_success)
        action_ok += int(pred["action"] == case.expected_action)
        citation_ok += int(pred["citation_valid"])
        wrong += int(pred["wrong_output"])
        false_answer += int(pred["false_answer"])
        missed_answer += int(pred["missed_answer"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    render = success / count
    partial_score = partial / count
    utility = render + 0.30 * partial_score - 2.0 * (false_answer / count) - 1.0 * (wrong / count) - 0.35 * (missed_answer / count) - 0.01 * cost
    return {
        "render_success": round(render, 6),
        "action_accuracy": round(action_ok / count, 6),
        "citation_validity": round(citation_ok / count, 6),
        "wrong_output": round(wrong / count, 6),
        "false_answer": round(false_answer / count, 6),
        "missed_answer": round(missed_answer / count, 6),
        "partial_scribe_score": round(partial_score, 6),
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
    best = evaluate(selected, cases, "validation")["utility"]
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    for generation in range(generations):
        candidate, mutation = mutate(selected, rng, generation)
        current = evaluate(selected, cases, "validation")
        proposed = evaluate(candidate, cases, "validation")
        accepted_flag = proposed["utility"] > current["utility"] + 1e-9
        if accepted_flag:
            selected = candidate
            accepted += 1
            best = proposed["utility"]
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
            "validation_utility": best,
            "timestamp_ms": now_ms(),
        })
    result = {
        "seed": seed,
        "selected": sorted(selected),
        "accepted": accepted,
        "rejected": rejected,
        "rollback": rollback,
        "split_metrics": {split: evaluate(selected, cases, split) for split in ["train", "validation", "adversarial"]},
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
        "validation_render_success_min": min(values("validation", "render_success")),
        "validation_render_success_mean": round(statistics.mean(values("validation", "render_success")), 6),
        "adversarial_render_success_min": min(values("adversarial", "render_success")),
        "adversarial_render_success_mean": round(statistics.mean(values("adversarial", "render_success")), 6),
        "validation_action_accuracy_min": min(values("validation", "action_accuracy")),
        "validation_citation_validity_min": min(values("validation", "citation_validity")),
        "adversarial_false_answer_max": max(values("adversarial", "false_answer")),
        "adversarial_wrong_output_max": max(values("adversarial", "wrong_output")),
        "validation_missed_answer_max": max(values("validation", "missed_answer")),
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
        missed_deltas = []
        for result in seed_results:
            cases = [OutputCase(**case) for case in result["cases"]]
            selected = set(result["selected"])
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["render_success"] - ablated["render_success"])
            missed_deltas.append(ablated["missed_answer"] - full["missed_answer"])
        summary[operator.operator_id] = {
            "mean_render_loss": round(statistics.mean(losses), 6),
            "mean_missed_answer_delta": round(statistics.mean(missed_deltas), 6),
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
    failures: list[str] = []
    decision = "e94_scribe_output_hygiene_expansion_confirmed"
    if aggregate["validation_render_success_min"] != 1.0 or aggregate["adversarial_render_success_min"] != 1.0:
        decision = "e94_output_hygiene_not_clean"
        failures.append("render success not clean")
    if aggregate["adversarial_false_answer_max"] != 0.0 or aggregate["adversarial_wrong_output_max"] != 0.0:
        decision = "e94_output_hygiene_safety_regression"
        failures.append("unsafe output behavior")
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
        "boundary": "controlled output rendering hygiene probe; not chatbot behavior",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),
        "external_output_only_after_agency_action": True,
        "hidden_answer_solving": False,
    }
    write_json(out / "operator_library_manifest.json", library)
    write_json(out / "task_generation_report.json", task_report)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "selection_frequency_report.json", frequency)
    write_json(out / "counterfactual_report.json", counterfactual)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(out / "mutation_summary.json", {
        "accepted": aggregate["accepted_mutations_total"],
        "rejected": aggregate["rejected_mutations_total"],
        "rollback": aggregate["rollback_count_total"],
        "mutation_mode": "operator_set_grow_drop_swap_with_validation_rollback",
    })
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", summary)
    write_json(out / "seed_results.json", {"seeds": [{key: value for key, value in result.items() if key != "cases"} for result in seed_results]})
    write_json(out / "partial_aggregate_snapshot.json", aggregate)
    sample_rows = 0
    for result in seed_results:
        for case in result["cases"][:30]:
            pred = predict(OutputCase(**case), set(result["selected"]))
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "internal_record": case["internal_record"],
                "expected_output": case["expected_output"],
                "expected_action": case["expected_action"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = ["# E94 Scribe Output Hygiene Expansion Result", "", f"decision = `{decision}`", "", "Boundary: controlled output rendering hygiene, not chatbot behavior.", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True), "```", "", "Stable Operator candidates:"]
    for row in lifecycle["operator_lifecycle_table"]:
        if row["final_status"] == "StableOperatorCandidate":
            report.append(f"- `{row['operator_id']}` - {row['description']}")
    report.append("")
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(sample_dir, out)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e94_scribe_output_hygiene_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e94_scribe_output_hygiene_expansion")
    parser.add_argument("--seeds", default="109401,109402,109403,109404,109405,109406,109407,109408,109409,109410,109411,109412,109413,109414,109415,109416")
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
        "boundary": "controlled output rendering hygiene probe; not chatbot behavior",
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
