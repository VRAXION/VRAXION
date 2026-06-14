#!/usr/bin/env python3
"""E96 Trace/Ground memory hygiene expansion.

Controlled Trace Ledger / Ground Field probe. This expands Lens/Guard/T-Stab/
Scribe Operators that keep evidence history deduplicated, provenance-valid,
replay-auditable, and safe for scoped Ground promotion.
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


ARTIFACT_CONTRACT = "E96_TRACE_GROUND_MEMORY_HYGIENE_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("trace_deduplication_lens", "Trace Deduplication Lens", "Lens", "useful", 0.12, "Merges duplicate evidence events without losing distinct provenance."),
    OperatorSpec("provenance_chain_guard", "Provenance Chain Guard", "Guard", "useful", 0.13, "Requires every memory update to preserve source/cycle/proposal lineage."),
    OperatorSpec("delayed_feedback_integrator_t_stab", "Delayed Feedback Integrator T-Stab", "T-Stab", "useful", 0.14, "Applies delayed positive/negative feedback to prior trace events."),
    OperatorSpec("contradiction_memory_index_lens", "Contradiction Memory Index Lens", "Lens", "useful", 0.13, "Indexes contradictory evidence pairs for later Agency checks."),
    OperatorSpec("ground_promotion_candidate_scribe", "Ground Promotion Candidate Scribe", "Scribe", "useful", 0.12, "Renders scoped Ground-promotion candidates from stable evidence chains."),
    OperatorSpec("stale_trace_pruner_guard", "Stale Trace Pruner Guard", "Guard", "useful", 0.12, "Prunes expired stale traces while preserving active dependencies."),
    OperatorSpec("scope_lifetime_t_stab", "Scope Lifetime T-Stab", "T-Stab", "useful", 0.13, "Stabilizes trace lifetime across local/global scope transitions."),
    OperatorSpec("replay_hash_audit_guard", "Replay Hash Audit Guard", "Guard", "useful", 0.12, "Verifies replay hashes before memory consolidation is accepted."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("duplicate_trace_accumulator", "Duplicate Trace Accumulator", "unsafe_control", "unsafe", 0.03, "Unsafe control: stores duplicate traces as separate support."),
    OperatorSpec("provenance_dropping_committer", "Provenance-Dropping Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: writes memory without lineage."),
    OperatorSpec("delayed_feedback_ignorer", "Delayed Feedback Ignorer", "unsafe_control", "unsafe", 0.03, "Unsafe control: ignores later negative feedback."),
    OperatorSpec("contradiction_forgetting_committer", "Contradiction-Forgetting Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: forgets contradiction pairs during consolidation."),
    OperatorSpec("always_promote_to_ground", "Always Promote To Ground", "unsafe_control", "unsafe", 0.05, "Unsafe control: promotes every stable-looking event into Ground."),
    OperatorSpec("stale_trace_keeper", "Stale Trace Keeper", "unsafe_control", "unsafe", 0.04, "Unsafe control: keeps expired local traces as active."),
    OperatorSpec("always_prune_control", "Always Prune Control", "control", "noop", 0.02, "Control: prunes useful active evidence."),
    OperatorSpec("trace_dedup_clone", "Trace Dedup Echo Clone", "Lens", "redundant", 0.18, "Redundant deduplication support without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class MemoryCase:
    case_id: str
    source_split: str
    family: str
    trace_state: str
    expected_action: str
    expected_memory_state: str
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


def make_case(seed: int, index: int) -> MemoryCase:
    family = (
        "deduplicate_same_evidence",
        "preserve_provenance_chain",
        "delayed_negative_feedback",
        "contradiction_index_update",
        "ground_promotion_candidate",
        "stale_trace_prune",
        "scope_lifetime_expire",
        "replay_hash_audit",
        "memory_budget_compaction",
    )[index % 9]
    event = f"ev_{(index * 7 + seed) % 97}"
    source = f"src_{(index * 5 + seed) % 17}"
    cycle = f"cy_{(index * 3 + seed) % 19}"
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "deduplicate_same_evidence":
        state = f"{event} appears twice from {source}/{cycle}; same payload and hash."
        return MemoryCase(case_id, split, family, state, "MERGE_TRACE", f"{event}:deduped:provenance_preserved", ("trace_deduplication_lens", "provenance_chain_guard"))
    if family == "preserve_provenance_chain":
        state = f"memory update for {event}; source={source}; cycle={cycle}; proposal=pr_{index % 11}."
        return MemoryCase(case_id, split, family, state, "KEEP_TRACE", f"{event}:lineage_complete", ("provenance_chain_guard", "replay_hash_audit_guard"))
    if family == "delayed_negative_feedback":
        state = f"prior trace {event} was committed; delayed feedback says invalid."
        return MemoryCase(case_id, split, family, state, "DOWNRANK_TRACE", f"{event}:negative_feedback_integrated", ("delayed_feedback_integrator_t_stab", "provenance_chain_guard"))
    if family == "contradiction_index_update":
        state = f"{event} says A=1; ev_alt_{index % 13} says A=0; both active."
        return MemoryCase(case_id, split, family, state, "INDEX_CONTRADICTION", f"{event}:contradiction_indexed", ("contradiction_memory_index_lens", "provenance_chain_guard"))
    if family == "ground_promotion_candidate":
        state = f"{event} has repeated verified support, no conflict, stable scope."
        return MemoryCase(case_id, split, family, state, "PROPOSE_GROUND_PROMOTION", f"{event}:promotion_candidate_scoped", ("ground_promotion_candidate_scribe", "provenance_chain_guard", "replay_hash_audit_guard"))
    if family == "stale_trace_prune":
        state = f"{event} expired; no active dependency; safe to prune."
        return MemoryCase(case_id, split, family, state, "PRUNE_STALE_TRACE", f"{event}:stale_pruned", ("stale_trace_pruner_guard", "scope_lifetime_t_stab"))
    if family == "scope_lifetime_expire":
        state = f"local scope ended; {event} was local-only and must not stay active."
        return MemoryCase(case_id, split, family, state, "EXPIRE_LOCAL_TRACE", f"{event}:scope_expired", ("scope_lifetime_t_stab", "stale_trace_pruner_guard"))
    if family == "replay_hash_audit":
        state = f"{event} replay hash matches deterministic ledger; consolidation requested."
        return MemoryCase(case_id, split, family, state, "ACCEPT_REPLAY_AUDIT", f"{event}:replay_verified", ("replay_hash_audit_guard", "provenance_chain_guard"))
    state = f"memory budget high; {event} stale duplicate, active dep_{index % 5} must stay."
    return MemoryCase(case_id, split, family, state, "COMPACT_TRACE_LEDGER", f"{event}:compacted_without_active_loss", ("trace_deduplication_lens", "stale_trace_pruner_guard", "scope_lifetime_t_stab"))


def generate_cases(seed: int, rows: int) -> list[MemoryCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: MemoryCase, selected: set[str]) -> dict[str, object] | None:
    if "duplicate_trace_accumulator" in selected and case.family == "deduplicate_same_evidence":
        return {"action": "KEEP_DUPLICATES", "memory_state": "duplicate_support_inflated", "provenance_valid": False, "bad_promotion": False, "stale_pollution": False, "false_prune": False}
    if "provenance_dropping_committer" in selected and case.family in {"preserve_provenance_chain", "ground_promotion_candidate", "replay_hash_audit"}:
        return {"action": case.expected_action, "memory_state": "lineage_missing", "provenance_valid": False, "bad_promotion": case.family == "ground_promotion_candidate", "stale_pollution": False, "false_prune": False}
    if "delayed_feedback_ignorer" in selected and case.family == "delayed_negative_feedback":
        return {"action": "KEEP_TRACE", "memory_state": "negative_feedback_ignored", "provenance_valid": True, "bad_promotion": False, "stale_pollution": True, "false_prune": False}
    if "contradiction_forgetting_committer" in selected and case.family == "contradiction_index_update":
        return {"action": "KEEP_TRACE", "memory_state": "contradiction_forgotten", "provenance_valid": True, "bad_promotion": False, "stale_pollution": True, "false_prune": False}
    if "always_promote_to_ground" in selected and case.family not in {"ground_promotion_candidate", "replay_hash_audit"}:
        return {"action": "PROMOTE_TO_GROUND", "memory_state": "unsafe_global_promotion", "provenance_valid": False, "bad_promotion": True, "stale_pollution": True, "false_prune": False}
    if "stale_trace_keeper" in selected and case.family in {"stale_trace_prune", "scope_lifetime_expire", "memory_budget_compaction"}:
        return {"action": "KEEP_TRACE", "memory_state": "stale_active", "provenance_valid": True, "bad_promotion": False, "stale_pollution": True, "false_prune": False}
    return None


def predict(case: MemoryCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    if "always_prune_control" in selected:
        return {"action": "PRUNE_TRACE", "memory_state": "active_trace_pruned", "provenance_valid": False, "bad_promotion": False, "stale_pollution": False, "false_prune": True}
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "memory_state": case.expected_memory_state, "provenance_valid": True, "bad_promotion": False, "stale_pollution": False, "false_prune": False}
    return {"action": "HOLD_MEMORY_UPDATE", "memory_state": "incomplete_hygiene", "provenance_valid": False, "bad_promotion": False, "stale_pollution": False, "false_prune": False}


def evaluate(selected: set[str], cases: list[MemoryCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"memory_hygiene_success": 0.0, "provenance_validity": 0.0, "replay_safe": 0.0, "bad_ground_promotion": 0.0, "stale_pollution": 0.0, "false_prune": 0.0, "utility": -1.0}
    success = provenance = replay_safe = bad_promotion = stale = false_prune = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = pred["action"] == case.expected_action and pred["memory_state"] == case.expected_memory_state and pred["provenance_valid"]
        success += int(row_success)
        provenance += int(pred["provenance_valid"])
        replay_safe += int(not pred["bad_promotion"] and not pred["stale_pollution"] and not pred["false_prune"])
        bad_promotion += int(pred["bad_promotion"])
        stale += int(pred["stale_pollution"])
        false_prune += int(pred["false_prune"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 2.0 * (bad_promotion / count) - 1.5 * (stale / count) - 1.3 * (false_prune / count) - 0.01 * cost
    return {
        "memory_hygiene_success": round(score, 6),
        "provenance_validity": round(provenance / count, 6),
        "replay_safe": round(replay_safe / count, 6),
        "bad_ground_promotion": round(bad_promotion / count, 6),
        "stale_pollution": round(stale / count, 6),
        "false_prune": round(false_prune / count, 6),
        "partial_memory_hygiene_score": round(partial_score, 6),
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
        "validation_memory_hygiene_success_min": min(values("validation", "memory_hygiene_success")),
        "validation_memory_hygiene_success_mean": round(statistics.mean(values("validation", "memory_hygiene_success")), 6),
        "adversarial_memory_hygiene_success_min": min(values("adversarial", "memory_hygiene_success")),
        "adversarial_memory_hygiene_success_mean": round(statistics.mean(values("adversarial", "memory_hygiene_success")), 6),
        "validation_provenance_validity_min": min(values("validation", "provenance_validity")),
        "validation_replay_safe_min": min(values("validation", "replay_safe")),
        "adversarial_bad_ground_promotion_max": max(values("adversarial", "bad_ground_promotion")),
        "adversarial_stale_pollution_max": max(values("adversarial", "stale_pollution")),
        "adversarial_false_prune_max": max(values("adversarial", "false_prune")),
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
        stale_deltas = []
        for result in seed_results:
            cases = [MemoryCase(**case) for case in result["cases"]]  # type: ignore[index]
            selected = set(result["selected"])  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["memory_hygiene_success"] - ablated["memory_hygiene_success"])
            stale_deltas.append(ablated["stale_pollution"] - full["stale_pollution"])
        summary[operator.operator_id] = {
            "mean_memory_hygiene_loss": round(statistics.mean(losses), 6),
            "mean_stale_pollution_delta": round(statistics.mean(stale_deltas), 6),
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
    decision = "e96_trace_ground_memory_hygiene_expansion_confirmed"
    if aggregate["validation_memory_hygiene_success_min"] != 1.0 or aggregate["adversarial_memory_hygiene_success_min"] != 1.0:
        decision = "e96_memory_hygiene_not_clean"
        failures.append("memory hygiene not clean")
    if aggregate["adversarial_bad_ground_promotion_max"] != 0.0 or aggregate["adversarial_stale_pollution_max"] != 0.0 or aggregate["adversarial_false_prune_max"] != 0.0:
        decision = "e96_memory_hygiene_safety_regression"
        failures.append("unsafe memory behavior")
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
        "boundary": "controlled Trace/Ground memory hygiene probe; not persistent user memory",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),  # type: ignore[index]
        "trace_ground_memory_hygiene": True,
        "persistent_user_memory": False,
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
            pred = predict(MemoryCase(**case), set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "trace_state": case["trace_state"],
                "expected_action": case["expected_action"],
                "expected_memory_state": case["expected_memory_state"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = ["# E96 Trace/Ground Memory Hygiene Expansion Result", "", f"decision = `{decision}`", "", "Boundary: controlled Trace/Ground memory hygiene, not persistent user memory.", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True), "```", "", "Stable Operator candidates:"]
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
    parser.add_argument("--out", default="target/pilot_wave/e96_trace_ground_memory_hygiene_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e96_trace_ground_memory_hygiene_expansion")
    parser.add_argument("--seeds", default="109601,109602,109603,109604,109605,109606,109607,109608,109609,109610,109611,109612,109613,109614,109615,109616")
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
        "boundary": "controlled Trace/Ground memory hygiene probe; not persistent user memory",
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
