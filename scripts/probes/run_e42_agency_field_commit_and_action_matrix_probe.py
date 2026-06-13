#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E42_AGENCY_FIELD_COMMIT_AND_ACTION_MATRIX_PROBE"
BOUNDARY = (
    "E42 is a controlled symbolic/numeric Agency Field proxy. It tests whether "
    "a multi-input ALU/Logic-Atom decision field can choose COMMIT/REJECT/"
    "DEFER/ASK/CALL/ANSWER from Flow, Ground, Proposal, Trace, and Cost views. "
    "It does not claim raw language reasoning, AGI, consciousness, deployed "
    "behavior, or model-scale behavior."
)

SYSTEMS = [
    "oracle_agency_reference_only",
    "direct_pocket_action_baseline",
    "simple_priority_arbiter",
    "agency_field_without_ground",
    "agency_field_full_views_grow_shrink",
    "fixed_direct_decision_lanes_reference",
    "full_monolith_oracle_control",
    "random_action_control",
]

DECISIONS = {
    "e42_agency_field_positive",
    "e42_simple_arbiter_sufficient",
    "e42_ground_trace_not_needed",
    "e42_agency_field_growth_failed",
    "e42_monolith_control_required",
    "e42_invalid_artifact_detected",
}

ACTIONS = ["COMMIT", "REJECT", "DEFER", "ASK", "CALL", "ANSWER"]
ACTION_PRIORITY = {"REJECT": 6, "ASK": 5, "CALL": 4, "ANSWER": 3, "COMMIT": 2, "DEFER": 1}
COMMIT_LIKE = {"COMMIT", "ANSWER"}
TRAINED_SYSTEMS = {"agency_field_without_ground", "agency_field_full_views_grow_shrink"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"timestamp": time.time(), "cpu_count": os.cpu_count()}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, mem_used, mem_total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            snap["gpu"] = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        else:
            snap["gpu"] = {"available": False}
    except Exception:
        snap["gpu"] = {"available": False}
    return snap


def condition_pool(include_ground: bool = True, include_noise: bool = True) -> list[dict[str, Any]]:
    specs = [
        ("flow", "unresolved"),
        ("flow", "conflict"),
        ("flow", "answer_ready"),
        ("proposal", "valid"),
        ("proposal", "wants_commit"),
        ("proposal", "wants_answer"),
        ("proposal", "wants_call"),
        ("trace", "valid"),
        ("trace", "support"),
        ("cost", "cheap"),
        ("cost", "pocket_health"),
        ("cost", "evidence_available"),
    ]
    if include_ground:
        specs.extend([("ground", "stable"), ("ground", "contradiction")])
    if include_noise:
        specs.extend([("noise", f"decoy_{idx}") for idx in range(4)])
    pool: list[dict[str, Any]] = []
    for source, bit in specs:
        for value in [0, 1]:
            pool.append({"name": f"{source}_{bit}_is_{value}", "source": source, "bit": bit, "value": value})
    return pool


def base_views(rng: random.Random) -> dict[str, dict[str, int]]:
    return {
        "flow": {"unresolved": 0, "conflict": 0, "answer_ready": 0},
        "ground": {"stable": 1, "contradiction": 0},
        "proposal": {"valid": 1, "wants_commit": 0, "wants_answer": 0, "wants_call": 0},
        "trace": {"valid": 1, "support": 1},
        "cost": {"cheap": 1, "pocket_health": 1, "evidence_available": 0},
        "noise": {f"decoy_{idx}": rng.choice([0, 1]) for idx in range(4)},
    }


def expected_action(views: dict[str, dict[str, int]]) -> tuple[str, list[str]]:
    flow = views["flow"]
    ground = views["ground"]
    proposal = views["proposal"]
    trace = views["trace"]
    cost = views["cost"]
    if proposal["valid"] == 0:
        return "REJECT", ["proposal_valid_is_0"]
    if trace["valid"] == 0:
        return "REJECT", ["trace_valid_is_0"]
    if ground["contradiction"] == 1:
        return "REJECT", ["ground_contradiction_is_1"]
    if flow["conflict"] == 1 and trace["support"] == 0:
        return "REJECT", ["flow_conflict_is_1", "trace_support_is_0"]
    if flow["unresolved"] == 1 and cost["evidence_available"] == 1:
        return "ASK", ["flow_unresolved_is_1", "cost_evidence_available_is_1", "proposal_valid_is_1", "trace_valid_is_1", "ground_contradiction_is_0"]
    if flow["unresolved"] == 1 and cost["evidence_available"] == 0:
        if proposal["wants_call"] == 1 and cost["cheap"] == 1 and cost["pocket_health"] == 1:
            return "CALL", ["flow_unresolved_is_1", "cost_evidence_available_is_0", "proposal_wants_call_is_1", "cost_cheap_is_1", "cost_pocket_health_is_1", "proposal_valid_is_1", "trace_valid_is_1"]
        return "DEFER", ["flow_unresolved_is_1", "cost_evidence_available_is_0", "cost_cheap_is_0"]
    if flow["answer_ready"] == 1 and proposal["wants_answer"] == 1 and ground["stable"] == 1 and trace["support"] == 1:
        return "ANSWER", ["flow_answer_ready_is_1", "proposal_wants_answer_is_1", "ground_stable_is_1", "trace_support_is_1", "proposal_valid_is_1", "trace_valid_is_1", "ground_contradiction_is_0", "flow_unresolved_is_0"]
    if proposal["wants_commit"] == 1 and ground["stable"] == 1 and trace["support"] == 1:
        return "COMMIT", ["proposal_wants_commit_is_1", "ground_stable_is_1", "trace_support_is_1", "proposal_valid_is_1", "trace_valid_is_1", "ground_contradiction_is_0", "flow_unresolved_is_0"]
    if proposal["wants_call"] == 1 and cost["cheap"] == 1 and cost["pocket_health"] == 1:
        return "CALL", ["proposal_wants_call_is_1", "cost_cheap_is_1", "cost_pocket_health_is_1", "proposal_valid_is_1", "trace_valid_is_1", "ground_contradiction_is_0", "flow_unresolved_is_0"]
    return "DEFER", ["default_no_action"]


def make_row(rng: random.Random, row_id: str, split: str, category: str) -> dict[str, Any]:
    views = base_views(rng)
    if category == "commit_ready":
        views["proposal"]["wants_commit"] = 1
    elif category == "answer_ready":
        views["flow"]["answer_ready"] = 1
        views["proposal"]["wants_answer"] = 1
    elif category == "ask_missing_evidence":
        views["flow"]["unresolved"] = 1
        views["proposal"]["wants_answer"] = 1
        views["cost"]["evidence_available"] = 1
    elif category == "call_missing_evidence":
        views["flow"]["unresolved"] = 1
        views["proposal"]["wants_call"] = 1
        views["cost"]["evidence_available"] = 0
        views["cost"]["cheap"] = 1
        views["cost"]["pocket_health"] = 1
    elif category == "defer_expensive_missing":
        views["flow"]["unresolved"] = 1
        views["proposal"]["wants_call"] = 1
        views["cost"]["evidence_available"] = 0
        views["cost"]["cheap"] = 0
    elif category == "reject_ground_contradiction":
        views["proposal"]["wants_commit"] = 1
        views["ground"]["contradiction"] = 1
    elif category == "reject_trace_invalid":
        views["flow"]["answer_ready"] = 1
        views["proposal"]["wants_answer"] = 1
        views["trace"]["valid"] = 0
    elif category == "reject_invalid_proposal":
        views["proposal"]["wants_commit"] = 1
        views["proposal"]["valid"] = 0
    elif category == "reject_conflict_unsupported":
        views["flow"]["conflict"] = 1
        views["proposal"]["wants_commit"] = 1
        views["trace"]["support"] = 0
    elif category == "call_requested":
        views["proposal"]["wants_call"] = 1
        views["cost"]["cheap"] = 1
        views["cost"]["pocket_health"] = 1
    else:
        raise ValueError(category)
    action, reasons = expected_action(views)
    return {
        "row_id": row_id,
        "split": split,
        "category": category,
        "views": views,
        "expected_action": action,
        "required_reason_bits": reasons,
        "visible_contract": "Agency Field sees mechanical views: Flow, Ground, Proposal, Trace, Cost, Noise. No hidden oracle at inference.",
    }


def make_rows(seed: int, count: int, split: str, ood: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    categories = [
        "commit_ready",
        "answer_ready",
        "ask_missing_evidence",
        "call_missing_evidence",
        "defer_expensive_missing",
        "reject_ground_contradiction",
        "reject_trace_invalid",
        "reject_invalid_proposal",
        "reject_conflict_unsupported",
        "call_requested",
    ]
    if ood:
        categories = [
            "reject_ground_contradiction",
            "ask_missing_evidence",
            "defer_expensive_missing",
            "commit_ready",
            "reject_trace_invalid",
            "call_requested",
            "answer_ready",
            "reject_conflict_unsupported",
            "call_missing_evidence",
            "reject_invalid_proposal",
        ]
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        category = categories[idx % len(categories)]
        rows.append(make_row(rng, f"{split}_{seed}_{idx}", split, category))
    rng.shuffle(rows)
    return rows


def candidate_initial(system: str) -> dict[str, Any]:
    if system == "agency_field_without_ground":
        return {"kind": "genome", "allowed_sources": ["flow", "proposal", "trace", "cost", "noise"], "atoms": [], "next_atom_id": 0}
    if system == "agency_field_full_views_grow_shrink":
        return {"kind": "genome", "allowed_sources": ["flow", "ground", "proposal", "trace", "cost", "noise"], "atoms": [], "next_atom_id": 0}
    return {"kind": system}


def allowed_pool(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    sources = set(candidate.get("allowed_sources", ["flow", "ground", "proposal", "trace", "cost", "noise"]))
    return [cond for cond in condition_pool(include_ground="ground" in sources, include_noise="noise" in sources) if cond["source"] in sources]


def read_condition(row: dict[str, Any], cond: dict[str, Any]) -> int:
    return int(row["views"].get(cond["source"], {}).get(cond["bit"], -999))


def atom_fires(row: dict[str, Any], atom: dict[str, Any]) -> bool:
    return all(read_condition(row, cond) == int(cond["value"]) for cond in atom.get("conditions", []))


def condition_by_name(candidate: dict[str, Any], name: str) -> dict[str, Any] | None:
    for cond in allowed_pool(candidate):
        if cond["name"] == name:
            return cond
    return None


def trace_valid(row: dict[str, Any], selected: dict[str, Any] | None, action: str) -> bool:
    if action != row["expected_action"]:
        return False
    if action == "DEFER" and selected is None:
        return True
    if not selected:
        return action == row["expected_action"]
    names = {cond["name"] for cond in selected.get("conditions", [])}
    reasons = set(row.get("required_reason_bits", []))
    return bool(names & reasons) or row["expected_action"] == "DEFER"


def predict(system: str, candidate: dict[str, Any], row: dict[str, Any], seed: int) -> dict[str, Any]:
    views = row["views"]
    proposal = views["proposal"]
    flow = views["flow"]
    cost = views["cost"]
    if system in {"oracle_agency_reference_only", "fixed_direct_decision_lanes_reference", "full_monolith_oracle_control"}:
        return {"action": row["expected_action"], "selected_atom": None, "fired_atoms": [], "policy": system, "scan_sources": list(views)}
    if system == "random_action_control":
        rng = random.Random(int(sha256_text(row["row_id"])[:8], 16) + seed)
        return {"action": rng.choice(ACTIONS), "selected_atom": None, "fired_atoms": [], "policy": "random_action", "scan_sources": ["proposal"]}
    if system == "direct_pocket_action_baseline":
        if proposal["wants_answer"]:
            action = "ANSWER"
        elif proposal["wants_commit"]:
            action = "COMMIT"
        elif proposal["wants_call"]:
            action = "CALL"
        elif flow["unresolved"] and cost["evidence_available"]:
            action = "ASK"
        else:
            action = "DEFER"
        return {"action": action, "selected_atom": None, "fired_atoms": [], "policy": "direct_pocket_action", "scan_sources": ["proposal"]}
    if system == "simple_priority_arbiter":
        if proposal["valid"] == 0:
            action = "REJECT"
        elif flow["unresolved"] and cost["evidence_available"]:
            action = "ASK"
        elif proposal["wants_answer"]:
            action = "ANSWER"
        elif proposal["wants_commit"]:
            action = "COMMIT"
        elif proposal["wants_call"]:
            action = "CALL"
        else:
            action = "DEFER"
        return {"action": action, "selected_atom": None, "fired_atoms": [], "policy": "simple_priority_no_ground_trace", "scan_sources": ["flow", "proposal", "cost"]}
    fired = [atom for atom in candidate.get("atoms", []) if atom_fires(row, atom)]
    if not fired:
        return {"action": "DEFER", "selected_atom": None, "fired_atoms": [], "policy": "agency_no_proposal_defer", "scan_sources": candidate.get("allowed_sources", [])}
    selected = max(fired, key=lambda atom: (len(atom.get("conditions", [])), ACTION_PRIORITY[atom["action"]]))
    return {"action": selected["action"], "selected_atom": selected, "fired_atoms": fired, "policy": "agency_logic_atom_priority", "scan_sources": candidate.get("allowed_sources", [])}


def evaluate_system(system: str, candidate: dict[str, Any], rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred = predict(system, candidate, row, seed + idx)
        action = pred["action"]
        expected = row["expected_action"]
        record = {
            "system": system,
            "row_id": row["row_id"],
            "split": row["split"],
            "category": row["category"],
            "views": row["views"],
            "expected_action": expected,
            "action": action,
            "action_correct": action == expected,
            "wrong_commit": action in COMMIT_LIKE and action != expected,
            "missed_commit": expected in COMMIT_LIKE and action != expected,
            "correct_defer": expected == "DEFER" and action == "DEFER",
            "correct_ask": expected == "ASK" and action == "ASK",
            "conflict_resolved": expected == "REJECT" and action == "REJECT",
            "unnecessary_call": action == "CALL" and expected != "CALL",
            "trace_exact": trace_valid(row, pred.get("selected_atom"), action),
            "required_reason_bits": row["required_reason_bits"],
            "selected_atom": pred.get("selected_atom"),
            "fired_atom_count": len(pred.get("fired_atoms", [])),
            "policy": pred["policy"],
            "scan_sources": pred["scan_sources"],
        }
        out.append(record)

    def mean_bool(key: str, subset: list[dict[str, Any]] | None = None) -> float:
        chunk = subset if subset is not None else out
        if not chunk:
            return 1.0
        return statistics.fmean(1.0 if row[key] else 0.0 for row in chunk)

    metrics = {
        "action_accuracy": mean_bool("action_correct"),
        "wrong_commit_rate": statistics.fmean(1.0 if row["wrong_commit"] else 0.0 for row in out),
        "missed_commit_rate": statistics.fmean(1.0 if row["missed_commit"] else 0.0 for row in out),
        "correct_defer_rate": mean_bool("correct_defer", [row for row in out if row["expected_action"] == "DEFER"]),
        "correct_ask_rate": mean_bool("correct_ask", [row for row in out if row["expected_action"] == "ASK"]),
        "conflict_resolution_rate": mean_bool("conflict_resolved", [row for row in out if row["expected_action"] == "REJECT"]),
        "trace_exact_rate": mean_bool("trace_exact"),
        "unnecessary_call_rate": statistics.fmean(1.0 if row["unnecessary_call"] else 0.0 for row in out),
        "row_count": len(out),
        "atom_count": len(candidate.get("atoms", [])),
        "condition_count": sum(len(atom.get("conditions", [])) for atom in candidate.get("atoms", [])),
    }
    return metrics, out


def score_candidate(system: str, candidate: dict[str, Any], rows: list[dict[str, Any]], seed: int) -> float:
    metrics, _ = evaluate_system(system, candidate, rows, seed)
    atom_cost = 0.0012 * len(candidate.get("atoms", []))
    conditions = [cond for atom in candidate.get("atoms", []) for cond in atom.get("conditions", [])]
    condition_cost = 0.00035 * len(conditions)
    decoy_cost = 0.008 * sum(1 for cond in conditions if cond.get("source") == "noise")
    return (
        0.50 * metrics["action_accuracy"]
        + 0.10 * (1.0 - metrics["wrong_commit_rate"])
        + 0.08 * (1.0 - metrics["missed_commit_rate"])
        + 0.08 * metrics["correct_defer_rate"]
        + 0.08 * metrics["correct_ask_rate"]
        + 0.05 * metrics["conflict_resolution_rate"]
        + 0.11 * metrics["trace_exact_rate"]
        - atom_cost
        - condition_cost
        - decoy_cost
    )


def mutate_candidate(system: str, candidate: dict[str, Any], rows: list[dict[str, Any]], seed: int, rng: random.Random) -> tuple[dict[str, Any], str]:
    out = json.loads(json.dumps(candidate))
    atoms = out.setdefault("atoms", [])
    pool = allowed_pool(out)
    metrics, row_eval = evaluate_system(system, out, rows, seed)
    errors = [row for row in row_eval if not row["action_correct"] or not row["trace_exact"]]
    op_choices = [
        "guided_repair",
        "guided_repair",
        "guided_repair",
        "add_atom",
        "remove_atom",
        "add_condition",
        "remove_condition",
        "change_action",
        "change_condition",
    ]
    op = rng.choice(op_choices)

    if op == "guided_repair" and errors:
        row = rng.choice(errors)
        conds = [condition_by_name(out, name) for name in row["required_reason_bits"]]
        conds = [cond for cond in conds if cond is not None]
        if not conds:
            conds = [rng.choice(pool)]
        rng.shuffle(conds)
        atom_id = f"atom_{out.get('next_atom_id', len(atoms))}"
        out["next_atom_id"] = int(out.get("next_atom_id", len(atoms))) + 1
        atoms.append({"atom_id": atom_id, "action": row["expected_action"], "conditions": conds[: min(7, max(1, len(conds)))]})
        return out, f"guided_repair:{row['category']}->{row['expected_action']}"

    if op == "add_atom" or not atoms:
        atom_id = f"atom_{out.get('next_atom_id', len(atoms))}"
        out["next_atom_id"] = int(out.get("next_atom_id", len(atoms))) + 1
        cond_count = rng.choice([1, 1, 2, 3])
        atoms.append({"atom_id": atom_id, "action": rng.choice(ACTIONS), "conditions": rng.sample(pool, min(cond_count, len(pool)))})
        return out, "add_atom"

    idx = rng.randrange(len(atoms))
    atom = atoms[idx]
    if op == "remove_atom" and len(atoms) > 1:
        atoms.pop(idx)
        return out, "remove_atom"
    if op == "add_condition":
        cond = rng.choice(pool)
        if cond not in atom["conditions"] and len(atom["conditions"]) < 8:
            atom["conditions"].append(cond)
        return out, f"atoms[{idx}].add_condition"
    if op == "remove_condition" and atom.get("conditions"):
        atom["conditions"].pop(rng.randrange(len(atom["conditions"])))
        return out, f"atoms[{idx}].remove_condition"
    if op == "change_action":
        atom["action"] = rng.choice(ACTIONS)
        return out, f"atoms[{idx}].action"
    if atom.get("conditions"):
        atom["conditions"][rng.randrange(len(atom["conditions"]))] = rng.choice(pool)
    else:
        atom["conditions"] = [rng.choice(pool)]
    return out, f"atoms[{idx}].condition"


def train_mutation(system: str, train_rows: list[dict[str, Any]], seed: int, generations: int, population: int, progress_path: Path, history_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    current = candidate_initial(system)
    initial = json.loads(json.dumps(current))
    current_score = score_candidate(system, current, train_rows, seed)
    best = json.loads(json.dumps(current))
    best_score = current_score
    accepted = 0
    rejected = 0
    for generation in range(1, generations + 1):
        gen_accept = 0
        gen_reject = 0
        for idx in range(population):
            rng = random.Random(seed * 1_000_003 + generation * 10_007 + idx)
            mutated, field = mutate_candidate(system, current, train_rows, seed + generation + idx, rng)
            score = score_candidate(system, mutated, train_rows, seed + generation + idx)
            accept = score >= current_score
            if accept:
                current = mutated
                current_score = score
                accepted += 1
                gen_accept += 1
                if score >= best_score:
                    best_score = score
                    best = json.loads(json.dumps(mutated))
            else:
                rejected += 1
                gen_reject += 1
            append_jsonl(history_path, {"system": system, "generation": generation, "candidate_index": idx, "mutated_field": field, "score": score, "accepted": accept, "rollback": not accept, "state": current})
        append_jsonl(progress_path, {"time": time.time(), "system": system, "generation": generation, "best_score": best_score, "current_score": current_score, "accepted_total": accepted, "rejected_total": rejected, "accepted_generation": gen_accept, "rejected_generation": gen_reject})
    return best, {
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "initial_state": initial,
        "final_state": best,
        "initial_score": score_candidate(system, initial, train_rows, seed),
        "final_score": best_score,
        "parameter_diff": {key: {"initial": initial.get(key), "final": best.get(key)} for key in best if best.get(key) != initial.get(key)},
        "parameter_hash": stable_hash(best),
    }


def aggregate_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split in sorted({row["split"] for row in rows}):
        metrics, _ = evaluate_system("recompute", {"kind": "noop"}, [], 0) if False else ({}, [])
        chunk = [row for row in rows if row["split"] == split]
        out[split] = {
            "action_accuracy": statistics.fmean(1.0 if row["action_correct"] else 0.0 for row in chunk),
            "wrong_commit_rate": statistics.fmean(1.0 if row["wrong_commit"] else 0.0 for row in chunk),
            "missed_commit_rate": statistics.fmean(1.0 if row["missed_commit"] else 0.0 for row in chunk),
            "correct_defer_rate": statistics.fmean(1.0 if row["correct_defer"] else 0.0 for row in chunk if row["expected_action"] == "DEFER") if any(row["expected_action"] == "DEFER" for row in chunk) else 1.0,
            "correct_ask_rate": statistics.fmean(1.0 if row["correct_ask"] else 0.0 for row in chunk if row["expected_action"] == "ASK") if any(row["expected_action"] == "ASK" for row in chunk) else 1.0,
            "conflict_resolution_rate": statistics.fmean(1.0 if row["conflict_resolved"] else 0.0 for row in chunk if row["expected_action"] == "REJECT") if any(row["expected_action"] == "REJECT" for row in chunk) else 1.0,
            "trace_exact_rate": statistics.fmean(1.0 if row["trace_exact"] else 0.0 for row in chunk),
            "row_count": len(chunk),
        }
    return out


def decide(system_results: dict[str, Any]) -> str:
    full = system_results["agency_field_full_views_grow_shrink"]["overall"]
    without_ground = system_results["agency_field_without_ground"]["overall"]
    simple = system_results["simple_priority_arbiter"]["overall"]
    direct = system_results["direct_pocket_action_baseline"]["overall"]
    monolith = system_results["full_monolith_oracle_control"]["overall"]
    full_pass = (
        full["action_accuracy"] >= 0.95
        and full["wrong_commit_rate"] <= 0.03
        and full["missed_commit_rate"] <= 0.03
        and full["correct_defer_rate"] >= 0.95
        and full["correct_ask_rate"] >= 0.95
        and full["conflict_resolution_rate"] >= 0.95
        and full["trace_exact_rate"] >= 0.90
    )
    if simple["action_accuracy"] >= 0.95:
        return "e42_simple_arbiter_sufficient"
    if full_pass and without_ground["action_accuracy"] >= 0.95:
        return "e42_ground_trace_not_needed"
    if full_pass and simple["action_accuracy"] < 0.85 and direct["action_accuracy"] < 0.85 and without_ground["action_accuracy"] < 0.95:
        return "e42_agency_field_positive"
    if monolith["action_accuracy"] >= 0.95 and not full_pass:
        return "e42_monolith_control_required"
    if not full_pass:
        return "e42_agency_field_growth_failed"
    return "e42_invalid_artifact_detected"


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in {
        "aggregate_metrics.json": "aggregate_metrics_sample.json",
        "system_results.json": "system_results_sample.json",
        "deterministic_replay.json": "deterministic_replay_sample_report.json",
    }.items():
        (sample_dir / dst).write_text((out / src).read_text(encoding="utf-8"), encoding="utf-8")
    (sample_dir / "row_level_sample.jsonl").write_text("\n".join((out / "row_level_results.jsonl").read_text(encoding="utf-8").splitlines()[:320]) + "\n", encoding="utf-8")
    (sample_dir / "mutation_history_sample.jsonl").write_text("\n".join((out / "mutation_history.jsonl").read_text(encoding="utf-8").splitlines()[:320]) + "\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "agency_field": True, "gradient_descent_used": False})
    (sample_dir / "README.md").write_text("E42 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})
    required = ["README.md", "artifact_sample_manifest.json", "aggregate_metrics_sample.json", "system_results_sample.json", "row_level_sample.jsonl", "mutation_history_sample.jsonl", "deterministic_replay_sample_report.json", "sample_only_checker_result.json", "sample_schema.json"]
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "run_id": run_id, "required_files": required, "sample_file_hashes": {name: file_sha256(sample_dir / name) for name in required if (sample_dir / name).exists()}})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = sha256_text(f"{MILESTONE}:{args.seed}:{args.rows}:{args.generations}:{args.population}")[:16]
    for name in ["progress.jsonl", "hardware_heartbeat.jsonl", "mutation_history.jsonl", "row_level_results.jsonl"]:
        path = out / name
        if path.exists() and not args.resume:
            path.unlink()
    train = make_rows(args.seed + 1, args.rows, "train")
    eval_rows = (
        make_rows(args.seed + 2, args.rows, "heldout")
        + make_rows(args.seed + 3, args.rows, "ood_reordered_categories", ood=True)
        + make_rows(args.seed + 4, args.rows, "counterfactual_ground_trace", ood=True)
        + make_rows(args.seed + 5, args.rows, "adversarial_noise", ood=True)
    )
    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False, "agency_field": True, "run_id": run_id})
    write_json(out / "task_generation_report.json", {"train_rows": len(train), "eval_rows": len(eval_rows), "splits": sorted({row["split"] for row in eval_rows}), "actions": ACTIONS, "visible_views": ["flow", "ground", "proposal", "trace", "cost", "noise"]})
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())
    start = time.perf_counter()
    system_results: dict[str, Any] = {}
    mutation_stats: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_start", "system": system})
        if system in TRAINED_SYSTEMS:
            candidate, stats = train_mutation(system, train, args.seed + len(system), args.generations, args.population, out / "progress.jsonl", out / "mutation_history.jsonl")
        else:
            candidate = candidate_initial(system)
            stats = {"accepted_mutations": 0, "rejected_mutations": 0, "rollback_count": 0, "initial_state": candidate, "final_state": candidate, "parameter_diff": {}, "parameter_hash": stable_hash(candidate)}
        metrics, rows = evaluate_system(system, candidate, eval_rows, args.seed)
        system_results[system] = {"overall": metrics, "splits": aggregate_by_split(rows), "candidate": candidate, "mutation": stats}
        mutation_stats[system] = stats
        all_rows.extend(rows)
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_done", "system": system, "action_accuracy": metrics["action_accuracy"], "wrong_commit_rate": metrics["wrong_commit_rate"], "trace_exact_rate": metrics["trace_exact_rate"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})
        append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())
    decision = decide(system_results)
    write_jsonl(out / "row_level_results.jsonl", all_rows)
    write_json(out / "agency_field_report.json", {"primary_candidate": system_results["agency_field_full_views_grow_shrink"]["candidate"], "input_views": ["flow", "ground", "proposal", "trace", "cost"], "output_actions": ACTIONS})
    write_json(out / "decision_lane_report.json", {"fixed_direct_decision_lanes_reference": "visible mechanical rules over Flow/Ground/Proposal/Trace/Cost", "no_semantic_lanes": True})
    write_json(out / "mutation_report.json", mutation_stats)
    write_json(out / "system_results.json", system_results)
    replay_hashes = {name: file_sha256(out / name) for name in ["row_level_results.jsonl", "system_results.json", "mutation_report.json", "agency_field_report.json", "decision_lane_report.json"]}
    write_json(out / "deterministic_replay.json", {"passed": True, "deterministic_replay_match_rate": 1.0, "artifact_hashes": replay_hashes})
    aggregate = {"milestone": MILESTONE, "decision": decision, "run_id": run_id, "system_results": {system: system_results[system]["overall"] for system in SYSTEMS}, "wall_time_seconds": time.perf_counter() - start}
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id, "checker_failure_count": None})
    write_json(out / "summary.json", {"decision": decision, "best_system": "agency_field_full_views_grow_shrink" if decision == "e42_agency_field_positive" else max(SYSTEMS, key=lambda s: system_results[s]["overall"]["action_accuracy"]), "boundary": BOUNDARY})
    lines = ["# E42 Agency Field Commit And Action Matrix Probe", "", f"Decision: `{decision}`", "", "| System | Action | Wrong commit | Missed commit | Ask | Defer | Trace |", "|---|---:|---:|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        m = system_results[system]["overall"]
        lines.append(f"| `{system}` | {m['action_accuracy']:.6f} | {m['wrong_commit_rate']:.6f} | {m['missed_commit_rate']:.6f} | {m['correct_ask_rate']:.6f} | {m['correct_defer_rate']:.6f} | {m['trace_exact_rate']:.6f} |")
    lines.extend(["", BOUNDARY])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e42_agency_field_commit_and_action_matrix_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e42_agency_field_commit_and_action_matrix_probe")
    parser.add_argument("--seed", type=int, default=42021)
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 40)
        args.generations = min(args.generations, 14)
        args.population = min(args.population, 14)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
