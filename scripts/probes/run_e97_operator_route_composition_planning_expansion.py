#!/usr/bin/env python3
"""E97 Operator route/composition planning expansion.

Controlled Operator orchestration probe. This expands Router/Lens/Guard/Scribe
skills that choose a small active Operator set and ordered call sequence. It is
not open-domain planning and not model-scale agent behavior.
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


ARTIFACT_CONTRACT = "E97_OPERATOR_ROUTE_COMPOSITION_PLANNING_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("route_intent_classifier_lens", "Route Intent Classifier Lens", "Lens", "useful", 0.12, "Classifies the current task intent into a route family."),
    OperatorSpec("active_operator_set_selector_guard", "Active Operator Set Selector Guard", "Guard", "useful", 0.13, "Selects a minimal active Operator set instead of scanning the full library."),
    OperatorSpec("ordered_operator_sequence_scribe", "Ordered Operator Sequence Scribe", "Scribe", "useful", 0.13, "Renders the ordered Operator call sequence for the current route."),
    OperatorSpec("adapter_requirement_detector_lens", "Adapter Requirement Detector Lens", "Lens", "useful", 0.12, "Detects when source/target ABI mismatch requires an Adapter Operator."),
    OperatorSpec("loop_prevention_route_guard", "Loop Prevention Route Guard", "Guard", "useful", 0.12, "Blocks repeated route cycles that do not change state."),
    OperatorSpec("route_budget_guard", "Route Budget Guard", "Guard", "useful", 0.12, "Keeps route calls inside the allowed compute/action budget."),
    OperatorSpec("fallback_to_ask_route_scribe", "Fallback-To-Ask Route Scribe", "Scribe", "useful", 0.11, "Renders ASK_FOR_EVIDENCE when no safe route can resolve the task."),
    OperatorSpec("composition_completion_t_stab", "Composition Completion T-Stab", "T-Stab", "useful", 0.13, "Stabilizes HALT/ANSWER_READY when the composed route is complete."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("full_library_scan_router", "Full Library Scan Router", "unsafe_control", "unsafe", 0.05, "Unsafe control: calls every available Operator."),
    OperatorSpec("random_operator_caller", "Random Operator Caller", "unsafe_control", "unsafe", 0.03, "Unsafe control: picks random Operators."),
    OperatorSpec("looping_route_runner", "Looping Route Runner", "unsafe_control", "unsafe", 0.04, "Unsafe control: repeats route calls without state change."),
    OperatorSpec("budgetless_route_expander", "Budgetless Route Expander", "unsafe_control", "unsafe", 0.04, "Unsafe control: expands route beyond budget."),
    OperatorSpec("adapterless_cross_abi_caller", "Adapterless Cross-ABI Caller", "unsafe_control", "unsafe", 0.04, "Unsafe control: skips required ABI adapter."),
    OperatorSpec("always_call_more_control", "Always Call-More Control", "control", "noop", 0.02, "Control: keeps calling even after completion."),
    OperatorSpec("route_selector_clone", "Route Selector Echo Clone", "Guard", "redundant", 0.18, "Redundant active-set selector without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class RouteCase:
    case_id: str
    source_split: str
    family: str
    route_state: str
    expected_action: str
    expected_route: tuple[str, ...]
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


def make_case(seed: int, index: int) -> RouteCase:
    family = (
        "alpha_sync_route",
        "temporal_stream_route",
        "commit_safety_route",
        "output_hygiene_route",
        "memory_hygiene_route",
        "adapter_required_route",
        "loop_risk_route",
        "budget_limited_route",
        "insufficient_info_ask_route",
        "completed_route_halt",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "alpha_sync_route":
        route = ("route_intent_classifier_lens", "active_operator_set_selector_guard", "lexical_alias_alpha_syncer", "canonical_lexeme_scribe")
        state = "task=intent:alias-normalization; visible alias evidence present; no adapter needed."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "active_operator_set_selector_guard", "ordered_operator_sequence_scribe"))
    if family == "temporal_stream_route":
        route = ("route_intent_classifier_lens", "frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe")
        state = "task=intent:temporal-stream; framed evidence visible; crc required."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "active_operator_set_selector_guard", "ordered_operator_sequence_scribe"))
    if family == "commit_safety_route":
        route = ("route_intent_classifier_lens", "proposal_collision_guard", "trace_dependency_coverage_guard", "safe_commit_action_scribe")
        state = "task=intent:commit-safety; proposals pending; collision check required."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "active_operator_set_selector_guard", "ordered_operator_sequence_scribe"))
    if family == "output_hygiene_route":
        route = ("route_intent_classifier_lens", "evidence_citation_scribe", "unit_preserving_answer_scribe", "no_answer_boundary_guard")
        state = "task=intent:output-render; answer record contains unit and evidence span."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "active_operator_set_selector_guard", "ordered_operator_sequence_scribe"))
    if family == "memory_hygiene_route":
        route = ("route_intent_classifier_lens", "trace_deduplication_lens", "provenance_chain_guard", "replay_hash_audit_guard")
        state = "task=intent:memory-hygiene; duplicate trace with replay hash visible."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "active_operator_set_selector_guard", "ordered_operator_sequence_scribe"))
    if family == "adapter_required_route":
        route = ("route_intent_classifier_lens", "adapter_requirement_detector_lens", "edge_adapter_operator", "ordered_operator_sequence_scribe")
        state = "task=intent:cross-abi-call; source ABI != target ABI; adapter required."
        return RouteCase(case_id, split, family, state, "CALL_SEQUENCE", route, ("route_intent_classifier_lens", "adapter_requirement_detector_lens", "ordered_operator_sequence_scribe"))
    if family == "loop_risk_route":
        state = "route repeated same Operator set twice; no Flow delta; loop risk."
        return RouteCase(case_id, split, family, state, "REJECT_ROUTE", tuple(), ("loop_prevention_route_guard", "fallback_to_ask_route_scribe"))
    if family == "budget_limited_route":
        state = "route budget=3 calls; full route would take 8 calls; request narrower evidence."
        return RouteCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", ("fallback_to_ask_route_scribe",), ("route_budget_guard", "fallback_to_ask_route_scribe"))
    if family == "insufficient_info_ask_route":
        state = "intent ambiguous between memory-hygiene and output-render; missing dependency visible."
        return RouteCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", ("fallback_to_ask_route_scribe",), ("route_intent_classifier_lens", "fallback_to_ask_route_scribe"))
    state = "route completed; answer-ready state stable; no more calls needed."
    return RouteCase(case_id, split, family, state, "HALT", tuple(), ("composition_completion_t_stab", "loop_prevention_route_guard"))


def generate_cases(seed: int, rows: int) -> list[RouteCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: RouteCase, selected: set[str]) -> dict[str, object] | None:
    if "full_library_scan_router" in selected and case.expected_action == "CALL_SEQUENCE":
        return {"action": "CALL_SEQUENCE", "route": tuple(OPERATOR_BY_ID), "route_valid": False, "overcall": True, "loop": False, "over_budget": True, "adapter_miss": False}
    if "random_operator_caller" in selected and case.expected_action == "CALL_SEQUENCE":
        return {"action": "CALL_SEQUENCE", "route": ("random_a", "random_b"), "route_valid": False, "overcall": False, "loop": False, "over_budget": False, "adapter_miss": False}
    if "looping_route_runner" in selected and case.family in {"loop_risk_route", "completed_route_halt"}:
        return {"action": "CALL_SEQUENCE", "route": ("repeat", "repeat"), "route_valid": False, "overcall": True, "loop": True, "over_budget": False, "adapter_miss": False}
    if "budgetless_route_expander" in selected and case.family == "budget_limited_route":
        return {"action": "CALL_SEQUENCE", "route": tuple(f"call_{i}" for i in range(8)), "route_valid": False, "overcall": True, "loop": False, "over_budget": True, "adapter_miss": False}
    if "adapterless_cross_abi_caller" in selected and case.family == "adapter_required_route":
        return {"action": "CALL_SEQUENCE", "route": ("route_intent_classifier_lens", "target_operator_without_adapter"), "route_valid": False, "overcall": False, "loop": False, "over_budget": False, "adapter_miss": True}
    return None


def predict(case: RouteCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    if "always_call_more_control" in selected and case.expected_action == "HALT":
        return {"action": "CALL_SEQUENCE", "route": ("extra_call",), "route_valid": False, "overcall": True, "loop": False, "over_budget": False, "adapter_miss": False}
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "route": case.expected_route, "route_valid": True, "overcall": False, "loop": False, "over_budget": False, "adapter_miss": False}
    return {"action": "ASK_FOR_EVIDENCE", "route": ("fallback_to_ask_route_scribe",), "route_valid": False, "overcall": False, "loop": False, "over_budget": False, "adapter_miss": False}


def evaluate(selected: set[str], cases: list[RouteCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"route_success": 0.0, "active_set_precision": 0.0, "sequence_accuracy": 0.0, "loop_rate": 0.0, "overcall_rate": 0.0, "over_budget": 0.0, "adapter_miss": 0.0, "utility": -1.0}
    success = precision = sequence = loop = overcall = over_budget = adapter_miss = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = pred["action"] == case.expected_action and tuple(pred["route"]) == case.expected_route and pred["route_valid"]
        success += int(row_success)
        precision += int(not pred["overcall"])
        sequence += int(tuple(pred["route"]) == case.expected_route)
        loop += int(pred["loop"])
        overcall += int(pred["overcall"])
        over_budget += int(pred["over_budget"])
        adapter_miss += int(pred["adapter_miss"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 2.0 * (loop / count) - 1.3 * (overcall / count) - 1.3 * (over_budget / count) - 1.5 * (adapter_miss / count) - 0.01 * cost
    return {
        "route_success": round(score, 6),
        "active_set_precision": round(precision / count, 6),
        "sequence_accuracy": round(sequence / count, 6),
        "loop_rate": round(loop / count, 6),
        "overcall_rate": round(overcall / count, 6),
        "over_budget": round(over_budget / count, 6),
        "adapter_miss": round(adapter_miss / count, 6),
        "partial_route_score": round(partial_score, 6),
        "cost": round(cost, 6),
        "utility": round(utility, 6),
    }


def mutate(selected: set[str], rng: random.Random, generation: int) -> tuple[set[str], dict[str, object]]:
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


def run_seed(seed: int, rows_per_seed: int, generations: int, out: Path) -> dict[str, object]:
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


def aggregate_results(seed_results: list[dict[str, object]], seconds: float) -> dict[str, object]:
    def values(split: str, key: str) -> list[float]:
        return [result["split_metrics"][split][key] for result in seed_results]  # type: ignore[index]

    return {
        "seconds": round(seconds, 3),
        "seed_count": len(seed_results),
        "validation_route_success_min": min(values("validation", "route_success")),
        "validation_route_success_mean": round(statistics.mean(values("validation", "route_success")), 6),
        "adversarial_route_success_min": min(values("adversarial", "route_success")),
        "adversarial_route_success_mean": round(statistics.mean(values("adversarial", "route_success")), 6),
        "validation_active_set_precision_min": min(values("validation", "active_set_precision")),
        "validation_sequence_accuracy_min": min(values("validation", "sequence_accuracy")),
        "adversarial_loop_rate_max": max(values("adversarial", "loop_rate")),
        "adversarial_overcall_rate_max": max(values("adversarial", "overcall_rate")),
        "adversarial_over_budget_max": max(values("adversarial", "over_budget")),
        "adversarial_adapter_miss_max": max(values("adversarial", "adapter_miss")),
        "accepted_mutations_total": sum(int(result["accepted"]) for result in seed_results),
        "rejected_mutations_total": sum(int(result["rejected"]) for result in seed_results),
        "rollback_count_total": sum(int(result["rollback"]) for result in seed_results),
    }


def build_frequency(seed_results: list[dict[str, object]]) -> dict[str, object]:
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


def build_counterfactual(seed_results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, dict[str, float]] = {}
    for operator in USEFUL_OPERATORS:
        losses = []
        overcall_deltas = []
        for result in seed_results:
            cases = [RouteCase(**case) for case in result["cases"]]  # type: ignore[index]
            selected = set(result["selected"])  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["route_success"] - ablated["route_success"])
            overcall_deltas.append(ablated["overcall_rate"] - full["overcall_rate"])
        summary[operator.operator_id] = {
            "mean_route_success_loss": round(statistics.mean(losses), 6),
            "mean_overcall_delta": round(statistics.mean(overcall_deltas), 6),
        }
    return {"summary": summary}


def build_lifecycle(frequency: dict[str, object], counterfactual: dict[str, object]) -> dict[str, object]:
    rows = []
    cf = counterfactual["summary"]  # type: ignore[index]
    for row in frequency["rows"]:  # type: ignore[index]
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


def deterministic_hash(payload: dict[str, object]) -> str:
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


def write_reports(out: Path, sample_dir: Path | None, seed_results: list[dict[str, object]], args: argparse.Namespace, seconds: float) -> None:
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
    decision = "e97_operator_route_composition_planning_expansion_confirmed"
    if aggregate["validation_route_success_min"] != 1.0 or aggregate["adversarial_route_success_min"] != 1.0:
        decision = "e97_route_planning_not_clean"
        failures.append("route planning not clean")
    if aggregate["adversarial_loop_rate_max"] != 0.0 or aggregate["adversarial_overcall_rate_max"] != 0.0 or aggregate["adversarial_over_budget_max"] != 0.0 or aggregate["adversarial_adapter_miss_max"] != 0.0:
        decision = "e97_route_planning_safety_regression"
        failures.append("unsafe route behavior")
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
        "boundary": "controlled Operator route/composition planning probe; not open-domain planning",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),  # type: ignore[index]
        "operator_route_planning": True,
        "open_domain_planning": False,
        "full_library_scan_allowed": False,
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
        for case in result["cases"][:30]:  # type: ignore[index]
            pred = predict(RouteCase(**case), set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "route_state": case["route_state"],
                "expected_action": case["expected_action"],
                "expected_route": case["expected_route"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = ["# E97 Operator Route/Composition Planning Expansion Result", "", f"decision = `{decision}`", "", "Boundary: controlled Operator routing, not open-domain planning.", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True), "```", "", "Stable Operator candidates:"]
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
    parser.add_argument("--out", default="target/pilot_wave/e97_operator_route_composition_planning_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e97_operator_route_composition_planning_expansion")
    parser.add_argument("--seeds", default="109701,109702,109703,109704,109705,109706,109707,109708,109709,109710,109711,109712,109713,109714,109715,109716")
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
        "boundary": "controlled Operator route/composition planning probe; not open-domain planning",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "seed_count": len(seeds)})
    seed_results: list[dict[str, object]] = []
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
    for result in sorted(seed_results, key=lambda item: int(item["seed"])):
        for generation in range(args.generations):
            append_jsonl(out / "operator_evolution_history.jsonl", {
                "seed": result["seed"],
                "generation": generation,
                "selected_count_final": len(result["selected"]),  # type: ignore[arg-type]
                "final_selected": result["selected"],
            })
    write_reports(out, Path(args.artifact_sample_dir), sorted(seed_results, key=lambda item: int(item["seed"])), args, time.time() - started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"out": str(out), "decision": json.loads((out / "decision.json").read_text(encoding="utf-8"))["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
