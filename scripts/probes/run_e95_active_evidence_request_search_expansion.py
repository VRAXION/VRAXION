#!/usr/bin/env python3
"""E95 active evidence request/search expansion.

Controlled active-evidence probe. This expands Lens/Guard/Scribe Operators that
choose a targeted evidence request/search action when the Flow state is
unresolved. It is not open-domain retrieval or chatbot behavior.
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


ARTIFACT_CONTRACT = "E95_ACTIVE_EVIDENCE_REQUEST_SEARCH_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("missing_dependency_locator_lens", "Missing Dependency Locator Lens", "Lens", "useful", 0.13, "Finds which unresolved dependency blocks the current answer."),
    OperatorSpec("targeted_evidence_request_scribe", "Targeted Evidence Request Scribe", "Scribe", "useful", 0.12, "Renders the minimal ASK_FOR_EVIDENCE request for the missing dependency."),
    OperatorSpec("source_reliability_rank_guard", "Source Reliability Rank Guard", "Guard", "useful", 0.13, "Ranks verified sources above rumor/decoy sources."),
    OperatorSpec("redundant_request_guard", "Redundant Request Guard", "Guard", "useful", 0.11, "Prevents asking for evidence already present in Trace/Ground."),
    OperatorSpec("search_budget_guard", "Search Budget Guard", "Guard", "useful", 0.12, "Keeps evidence search inside the allowed action budget."),
    OperatorSpec("adversarial_decoy_source_guard", "Adversarial Decoy Source Guard", "Guard", "useful", 0.14, "Rejects decoy evidence sources that match surface form but not requested dependency."),
    OperatorSpec("retrieved_evidence_integrator_t_stab", "Retrieved Evidence Integrator T-Stab", "T-Stab", "useful", 0.14, "Stabilizes the transition from retrieved evidence to answer-ready state."),
    OperatorSpec("answer_ready_after_evidence_scribe", "Answer-Ready After Evidence Scribe", "Scribe", "useful", 0.10, "Renders ANSWER_READY only after required evidence has been integrated."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("broad_search_spammer", "Broad Search Spammer", "unsafe_control", "unsafe", 0.04, "Unsafe control: searches every source instead of the missing dependency."),
    OperatorSpec("random_evidence_picker", "Random Evidence Picker", "unsafe_control", "unsafe", 0.03, "Unsafe control: requests random evidence."),
    OperatorSpec("rumor_source_committer", "Rumor Source Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: trusts rumor evidence over verified evidence."),
    OperatorSpec("decoy_surface_match_committer", "Decoy Surface Match Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: accepts surface-matching decoy evidence."),
    OperatorSpec("budgetless_search_runner", "Budgetless Search Runner", "unsafe_control", "unsafe", 0.05, "Unsafe control: exceeds the search/action budget."),
    OperatorSpec("always_ask_all_control", "Always Ask-All Control", "control", "noop", 0.02, "Control: succeeds by asking too much when targeted evidence is enough."),
    OperatorSpec("request_locator_clone", "Request Locator Echo Clone", "Lens", "redundant", 0.18, "Redundant dependency locator without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    source_split: str
    family: str
    flow_state: str
    expected_action: str
    expected_request: str | None
    expected_answer_state: str | None
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


def make_case(seed: int, index: int) -> EvidenceCase:
    family = (
        "missing_single_dependency",
        "missing_two_choose_blocker",
        "verified_vs_rumor_source",
        "redundant_request_present",
        "budget_limited_search",
        "decoy_surface_source",
        "retrieved_evidence_transition",
        "already_answer_ready",
        "unanswerable_no_source",
    )[index % 9]
    dep = f"dep_{(index * 5 + seed) % 23}"
    alt = f"dep_{(index * 7 + seed + 1) % 23}"
    if alt == dep:
        alt = f"dep_{(index * 11 + seed + 2) % 23}"
    value = f"value_{(index * 13 + seed) % 37}"
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "missing_single_dependency":
        state = f"query needs {dep}; Trace missing {dep}; verified source available."
        return EvidenceCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", dep, None, ("missing_dependency_locator_lens", "targeted_evidence_request_scribe"))
    if family == "missing_two_choose_blocker":
        state = f"query needs {dep} and {alt}; {alt} already present; {dep} missing."
        return EvidenceCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", dep, None, ("missing_dependency_locator_lens", "targeted_evidence_request_scribe", "redundant_request_guard"))
    if family == "verified_vs_rumor_source":
        state = f"{dep} available from rumor and verified source; request must prefer verified source."
        return EvidenceCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", f"{dep}@verified", None, ("source_reliability_rank_guard", "targeted_evidence_request_scribe"))
    if family == "redundant_request_present":
        state = f"query needs {dep}; Trace already has {dep}={value}; no request needed."
        return EvidenceCase(case_id, split, family, state, "ANSWER_READY", None, value, ("redundant_request_guard", "answer_ready_after_evidence_scribe"))
    if family == "budget_limited_search":
        state = f"query needs {dep}; budget=1; broad search would exceed budget."
        return EvidenceCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", dep, None, ("search_budget_guard", "missing_dependency_locator_lens", "targeted_evidence_request_scribe"))
    if family == "decoy_surface_source":
        state = f"query needs {dep}; decoy source mentions {dep} surface but payload is {alt}; verified source exists."
        return EvidenceCase(case_id, split, family, state, "ASK_FOR_EVIDENCE", f"{dep}@verified", None, ("adversarial_decoy_source_guard", "source_reliability_rank_guard", "targeted_evidence_request_scribe"))
    if family == "retrieved_evidence_transition":
        state = f"retrieved evidence says {dep}={value}; integrate then answer-ready."
        return EvidenceCase(case_id, split, family, state, "ANSWER_READY", None, value, ("retrieved_evidence_integrator_t_stab", "answer_ready_after_evidence_scribe"))
    if family == "already_answer_ready":
        state = f"all dependencies present; answer state={value}; no search allowed."
        return EvidenceCase(case_id, split, family, state, "ANSWER_READY", None, value, ("redundant_request_guard", "answer_ready_after_evidence_scribe"))
    state = f"query needs {dep}; no visible source can provide it."
    return EvidenceCase(case_id, split, family, state, "HOLD_UNRESOLVED", None, None, ("missing_dependency_locator_lens", "targeted_evidence_request_scribe", "search_budget_guard"))


def generate_cases(seed: int, rows: int) -> list[EvidenceCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: EvidenceCase, selected: set[str]) -> dict[str, object] | None:
    if "broad_search_spammer" in selected and case.expected_action == "ASK_FOR_EVIDENCE":
        return {"action": "SEARCH_ALL", "request": "all_sources", "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": True, "over_budget": True}
    if "random_evidence_picker" in selected and case.expected_action == "ASK_FOR_EVIDENCE":
        return {"action": "ASK_FOR_EVIDENCE", "request": "random_dep", "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": False, "over_budget": False}
    if "rumor_source_committer" in selected and case.family == "verified_vs_rumor_source":
        return {"action": "ANSWER_READY", "request": None, "answer_state": "rumor_value", "targeted": False, "wrong_confident": True, "false_search": False, "over_budget": False}
    if "decoy_surface_match_committer" in selected and case.family == "decoy_surface_source":
        return {"action": "ANSWER_READY", "request": None, "answer_state": "decoy_payload", "targeted": False, "wrong_confident": True, "false_search": False, "over_budget": False}
    if "budgetless_search_runner" in selected and case.family == "budget_limited_search":
        return {"action": "SEARCH_ALL", "request": "all_sources", "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": True, "over_budget": True}
    return None


def predict(case: EvidenceCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    if "always_ask_all_control" in selected and case.expected_action in {"ASK_FOR_EVIDENCE", "ANSWER_READY"}:
        return {"action": "SEARCH_ALL", "request": "all_sources", "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": case.expected_action == "ANSWER_READY", "over_budget": True}
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "request": case.expected_request, "answer_state": case.expected_answer_state, "targeted": True, "wrong_confident": False, "false_search": False, "over_budget": False}
    if case.expected_action == "ASK_FOR_EVIDENCE":
        return {"action": "HOLD_UNRESOLVED", "request": None, "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": False, "over_budget": False}
    if case.expected_action == "ANSWER_READY":
        return {"action": "HOLD_UNRESOLVED", "request": None, "answer_state": None, "targeted": False, "wrong_confident": False, "false_search": False, "over_budget": False}
    return {"action": "HOLD_UNRESOLVED", "request": None, "answer_state": None, "targeted": True, "wrong_confident": False, "false_search": False, "over_budget": False}


def evaluate(selected: set[str], cases: list[EvidenceCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"evidence_action_success": 0.0, "targeted_request_accuracy": 0.0, "answer_ready_accuracy": 0.0, "wrong_confident": 0.0, "false_search": 0.0, "over_budget": 0.0, "utility": -1.0}
    success = targeted = answer_ready = wrong = false_search = over_budget = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = pred["action"] == case.expected_action and pred["request"] == case.expected_request and pred["answer_state"] == case.expected_answer_state
        success += int(row_success)
        targeted += int(case.expected_action != "ASK_FOR_EVIDENCE" or pred["request"] == case.expected_request)
        answer_ready += int(case.expected_action != "ANSWER_READY" or pred["answer_state"] == case.expected_answer_state)
        wrong += int(pred["wrong_confident"])
        false_search += int(pred["false_search"])
        over_budget += int(pred["over_budget"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 2.0 * (wrong / count) - 1.2 * (false_search / count) - 1.0 * (over_budget / count) - 0.01 * cost
    return {
        "evidence_action_success": round(score, 6),
        "targeted_request_accuracy": round(targeted / count, 6),
        "answer_ready_accuracy": round(answer_ready / count, 6),
        "wrong_confident": round(wrong / count, 6),
        "false_search": round(false_search / count, 6),
        "over_budget": round(over_budget / count, 6),
        "partial_evidence_policy_score": round(partial_score, 6),
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
        "validation_evidence_action_success_min": min(values("validation", "evidence_action_success")),
        "validation_evidence_action_success_mean": round(statistics.mean(values("validation", "evidence_action_success")), 6),
        "adversarial_evidence_action_success_min": min(values("adversarial", "evidence_action_success")),
        "adversarial_evidence_action_success_mean": round(statistics.mean(values("adversarial", "evidence_action_success")), 6),
        "validation_targeted_request_accuracy_min": min(values("validation", "targeted_request_accuracy")),
        "validation_answer_ready_accuracy_min": min(values("validation", "answer_ready_accuracy")),
        "adversarial_wrong_confident_max": max(values("adversarial", "wrong_confident")),
        "adversarial_false_search_max": max(values("adversarial", "false_search")),
        "adversarial_over_budget_max": max(values("adversarial", "over_budget")),
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
        false_search_deltas = []
        for result in seed_results:
            cases = [EvidenceCase(**case) for case in result["cases"]]  # type: ignore[index]
            selected = set(result["selected"])  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["evidence_action_success"] - ablated["evidence_action_success"])
            false_search_deltas.append(ablated["false_search"] - full["false_search"])
        summary[operator.operator_id] = {
            "mean_evidence_action_loss": round(statistics.mean(losses), 6),
            "mean_false_search_delta": round(statistics.mean(false_search_deltas), 6),
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
    decision = "e95_active_evidence_request_search_expansion_confirmed"
    if aggregate["validation_evidence_action_success_min"] != 1.0 or aggregate["adversarial_evidence_action_success_min"] != 1.0:
        decision = "e95_active_evidence_policy_not_clean"
        failures.append("active evidence policy not clean")
    if aggregate["adversarial_wrong_confident_max"] != 0.0 or aggregate["adversarial_false_search_max"] != 0.0 or aggregate["adversarial_over_budget_max"] != 0.0:
        decision = "e95_active_evidence_safety_regression"
        failures.append("unsafe search/evidence behavior")
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
        "boundary": "controlled active evidence request/search probe; not open-domain retrieval",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),  # type: ignore[index]
        "active_evidence_selection": True,
        "open_domain_retrieval": False,
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
            pred = predict(EvidenceCase(**case), set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "flow_state": case["flow_state"],
                "expected_action": case["expected_action"],
                "expected_request": case["expected_request"],
                "expected_answer_state": case["expected_answer_state"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = ["# E95 Active Evidence Request/Search Expansion Result", "", f"decision = `{decision}`", "", "Boundary: controlled active evidence selection, not open-domain retrieval.", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True), "```", "", "Stable Operator candidates:"]
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
    parser.add_argument("--out", default="target/pilot_wave/e95_active_evidence_request_search_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e95_active_evidence_request_search_expansion")
    parser.add_argument("--seeds", default="109501,109502,109503,109504,109505,109506,109507,109508,109509,109510,109511,109512,109513,109514,109515,109516")
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
        "boundary": "controlled active evidence request/search probe; not open-domain retrieval",
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
