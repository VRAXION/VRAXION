#!/usr/bin/env python3
"""E93 Agency/Guard commit-safety expansion.

Controlled proposal/commit probe. This teaches Operator Library guard skills
for deciding when a proposal can become a Flow/Ground commit. It is not a new
architecture and not open-domain model behavior.
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


ARTIFACT_CONTRACT = "E93_AGENCY_GUARD_COMMIT_SAFETY_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("proposal_collision_guard", "Proposal Collision Guard", "Guard", "useful", 0.13, "Detects conflicting same-target proposals before commit."),
    OperatorSpec("ground_conflict_guard", "Ground Conflict Guard", "Guard", "useful", 0.14, "Blocks proposals that contradict stable Ground without revocation evidence."),
    OperatorSpec("evidence_recency_guard", "Evidence Recency Guard", "Guard", "useful", 0.12, "Prefers newer verified evidence over older active evidence."),
    OperatorSpec("trace_dependency_coverage_guard", "Trace Dependency Coverage Guard", "Guard", "useful", 0.14, "Requires every proposal dependency to have visible trace support."),
    OperatorSpec("cycle_freshness_guard", "Cycle Freshness Guard", "Guard", "useful", 0.12, "Rejects stale-cycle proposals replayed into the current decision cycle."),
    OperatorSpec("local_scope_exit_t_stab", "Local Scope Exit T-Stab", "T-Stab", "useful", 0.13, "Stabilizes cleanup of local bindings after their scope ends."),
    OperatorSpec("agency_commit_quorum_guard", "Agency Commit Quorum Guard", "Guard", "useful", 0.13, "Requires compatible evidence quorum for high-risk commits."),
    OperatorSpec("safe_commit_action_scribe", "Safe Commit Action Scribe", "Scribe", "useful", 0.10, "Renders COMMIT/REJECT/DEFER/ASK actions after guard checks."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("first_proposal_committer", "First Proposal Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: commits whichever proposal appears first."),
    OperatorSpec("majority_without_trace_committer", "Majority Without Trace Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: commits majority proposals without trace coverage."),
    OperatorSpec("stale_cycle_committer", "Stale Cycle Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: accepts stale-cycle proposals."),
    OperatorSpec("ground_overwrite_committer", "Ground Overwrite Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: overwrites stable Ground without revocation evidence."),
    OperatorSpec("local_scope_leak_committer", "Local Scope Leak Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: leaks local bindings after scope exit."),
    OperatorSpec("always_reject_control", "Always Reject Control", "control", "noop", 0.02, "Control: rejects even safe commits."),
    OperatorSpec("quorum_guard_clone", "Quorum Guard Echo Clone", "Guard", "redundant", 0.18, "Redundant quorum guard support without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class CommitCase:
    case_id: str
    source_split: str
    family: str
    proposal_text: str
    expected_action: str
    expected_value: str | None
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


def make_case(seed: int, index: int) -> CommitCase:
    family = (
        "clean_commit",
        "proposal_collision",
        "ground_conflict",
        "recency_update",
        "missing_trace_dependency",
        "stale_cycle_replay",
        "local_scope_exit",
        "quorum_commit",
        "ask_for_more_evidence",
    )[index % 9]
    target = f"cell_{(index * 7 + seed) % 41}"
    value = f"value_{(index * 5 + seed) % 17}"
    alt = f"value_{(index * 11 + seed + 1) % 17}"
    if alt == value:
        alt = f"value_{(index * 13 + seed + 2) % 17}"
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    if family == "clean_commit":
        text = f"Proposal A writes {target}={value}; trace ok; ground compatible; cycle current."
        return CommitCase(case_id, split, family, text, "COMMIT", value, ("trace_dependency_coverage_guard", "cycle_freshness_guard", "safe_commit_action_scribe"))
    if family == "proposal_collision":
        text = f"Proposal A writes {target}={value}; proposal B writes {target}={alt}; both visible."
        return CommitCase(case_id, split, family, text, "DEFER", None, ("proposal_collision_guard", "safe_commit_action_scribe"))
    if family == "ground_conflict":
        text = f"Ground has {target}={alt}; proposal writes {target}={value}; no revocation evidence."
        return CommitCase(case_id, split, family, text, "REJECT", None, ("ground_conflict_guard", "safe_commit_action_scribe"))
    if family == "recency_update":
        text = f"Older verified evidence says {target}={alt}; newer verified evidence says {target}={value}."
        return CommitCase(case_id, split, family, text, "COMMIT", value, ("evidence_recency_guard", "trace_dependency_coverage_guard", "safe_commit_action_scribe"))
    if family == "missing_trace_dependency":
        text = f"Proposal writes {target}={value}; dependency dep_{index % 5} has no visible trace."
        return CommitCase(case_id, split, family, text, "ASK", None, ("trace_dependency_coverage_guard", "safe_commit_action_scribe"))
    if family == "stale_cycle_replay":
        text = f"Old cycle proposal replays {target}={value}; current cycle id does not match."
        return CommitCase(case_id, split, family, text, "REJECT", None, ("cycle_freshness_guard", "safe_commit_action_scribe"))
    if family == "local_scope_exit":
        text = f"Local block ended; proposal tries to reuse local {target}={value} outside its scope."
        return CommitCase(case_id, split, family, text, "REJECT", None, ("local_scope_exit_t_stab", "safe_commit_action_scribe"))
    if family == "quorum_commit":
        text = f"Two independent traces support {target}={value}; no conflict; high-risk commit requires quorum."
        return CommitCase(case_id, split, family, text, "COMMIT", value, ("agency_commit_quorum_guard", "trace_dependency_coverage_guard", "safe_commit_action_scribe"))
    text = f"Only one weak proposal writes {target}={value}; high-risk target; no quorum yet."
    return CommitCase(case_id, split, family, text, "ASK", None, ("agency_commit_quorum_guard", "safe_commit_action_scribe"))


def generate_cases(seed: int, rows: int) -> list[CommitCase]:
    return [make_case(seed, index) for index in range(rows)]


def predict(case: CommitCase, selected: set[str]) -> dict[str, Any]:
    if "always_reject_control" in selected and case.expected_action == "COMMIT":
        return {"action": "REJECT", "value": None, "trace_valid": True, "wrong_commit": False, "false_commit": False, "missed_commit": True}
    if "first_proposal_committer" in selected and case.family in {"proposal_collision", "missing_trace_dependency", "ask_for_more_evidence"}:
        return {"action": "COMMIT", "value": "first_visible", "trace_valid": False, "wrong_commit": True, "false_commit": True, "missed_commit": False}
    if "majority_without_trace_committer" in selected and case.family in {"missing_trace_dependency", "ask_for_more_evidence"}:
        return {"action": "COMMIT", "value": "unsupported_majority", "trace_valid": False, "wrong_commit": True, "false_commit": True, "missed_commit": False}
    if "stale_cycle_committer" in selected and case.family == "stale_cycle_replay":
        return {"action": "COMMIT", "value": "stale_value", "trace_valid": False, "wrong_commit": True, "false_commit": True, "missed_commit": False}
    if "ground_overwrite_committer" in selected and case.family == "ground_conflict":
        return {"action": "COMMIT", "value": "overwrite_value", "trace_valid": False, "wrong_commit": True, "false_commit": True, "missed_commit": False}
    if "local_scope_leak_committer" in selected and case.family == "local_scope_exit":
        return {"action": "COMMIT", "value": "leaked_local", "trace_valid": False, "wrong_commit": True, "false_commit": True, "missed_commit": False}

    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "value": case.expected_value, "trace_valid": True, "wrong_commit": False, "false_commit": False, "missed_commit": False}
    if case.expected_action == "COMMIT":
        return {"action": "DEFER", "value": None, "trace_valid": True, "wrong_commit": False, "false_commit": False, "missed_commit": True}
    return {"action": "DEFER", "value": None, "trace_valid": False, "wrong_commit": False, "false_commit": False, "missed_commit": False}


def evaluate(selected: set[str], cases: list[CommitCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"commit_safety_success": 0.0, "action_accuracy": 0.0, "trace_validity": 0.0, "wrong_commit": 0.0, "false_commit": 0.0, "missed_commit": 0.0, "utility": -1.0}
    success = action_ok = trace_ok = wrong = false_commit = missed_commit = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_action_ok = pred["action"] == case.expected_action
        value_ok = case.expected_action != "COMMIT" or pred["value"] == case.expected_value
        row_success = row_action_ok and value_ok and pred["trace_valid"]
        success += int(row_success)
        action_ok += int(row_action_ok)
        trace_ok += int(pred["trace_valid"])
        wrong += int(pred["wrong_commit"])
        false_commit += int(pred["false_commit"])
        missed_commit += int(pred["missed_commit"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    safety = success / count
    partial_score = partial / count
    utility = safety + 0.30 * partial_score - 2.0 * (wrong / count) - 1.7 * (false_commit / count) - 0.4 * (missed_commit / count) - 0.01 * cost
    return {
        "commit_safety_success": round(safety, 6),
        "action_accuracy": round(action_ok / count, 6),
        "trace_validity": round(trace_ok / count, 6),
        "wrong_commit": round(wrong / count, 6),
        "false_commit": round(false_commit / count, 6),
        "missed_commit": round(missed_commit / count, 6),
        "partial_guard_score": round(partial_score, 6),
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
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    best = evaluate(selected, cases, "validation")["utility"]
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
        "validation_commit_safety_success_min": min(values("validation", "commit_safety_success")),
        "validation_commit_safety_success_mean": round(statistics.mean(values("validation", "commit_safety_success")), 6),
        "adversarial_commit_safety_success_min": min(values("adversarial", "commit_safety_success")),
        "adversarial_commit_safety_success_mean": round(statistics.mean(values("adversarial", "commit_safety_success")), 6),
        "validation_trace_validity_min": min(values("validation", "trace_validity")),
        "adversarial_wrong_commit_max": max(values("adversarial", "wrong_commit")),
        "adversarial_false_commit_max": max(values("adversarial", "false_commit")),
        "validation_missed_commit_max": max(values("validation", "missed_commit")),
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
            cases = [CommitCase(**case) for case in result["cases"]]
            selected = set(result["selected"])
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator.operator_id}, cases, "validation")
            losses.append(full["commit_safety_success"] - ablated["commit_safety_success"])
            missed_deltas.append(ablated["missed_commit"] - full["missed_commit"])
        summary[operator.operator_id] = {
            "mean_commit_safety_loss": round(statistics.mean(losses), 6),
            "mean_missed_commit_delta": round(statistics.mean(missed_deltas), 6),
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
    decision = "e93_agency_guard_commit_safety_expansion_confirmed"
    if aggregate["validation_commit_safety_success_min"] != 1.0 or aggregate["adversarial_commit_safety_success_min"] != 1.0:
        decision = "e93_commit_safety_expansion_not_clean"
        failures.append("commit safety not clean")
    if aggregate["adversarial_wrong_commit_max"] != 0.0 or aggregate["adversarial_false_commit_max"] != 0.0:
        decision = "e93_commit_safety_regression"
        failures.append("unsafe commit behavior")
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
        "boundary": "controlled Proposal Field commit-safety probe; not open-domain model behavior",
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),
        "proposal_field_commit_boundary": True,
        "direct_flow_write_allowed": False,
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
            pred = predict(CommitCase(**case), set(result["selected"]))
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": case["case_id"],
                "family": case["family"],
                "proposal_text": case["proposal_text"],
                "expected_action": case["expected_action"],
                "expected_value": case["expected_value"],
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = ["# E93 Agency/Guard Commit-Safety Expansion Result", "", f"decision = `{decision}`", "", "Boundary: controlled Proposal Field commit safety, not open-domain model behavior.", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True), "```", "", "Stable Operator candidates:"]
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
    parser.add_argument("--out", default="target/pilot_wave/e93_agency_guard_commit_safety_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e93_agency_guard_commit_safety_expansion")
    parser.add_argument("--seeds", default="109301,109302,109303,109304,109305,109306,109307,109308,109309,109310,109311,109312,109313,109314,109315,109316")
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
        "boundary": "controlled Proposal Field commit-safety probe; not open-domain model behavior",
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
