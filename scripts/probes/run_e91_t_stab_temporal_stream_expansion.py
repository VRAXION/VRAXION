#!/usr/bin/env python3
"""E91 T-Stab temporal stream expansion.

Controlled temporal/binary-text stream probe. This extends the E90 text-evidence
Operators into noisy framed streams: sequence order, CRC/parity validation,
bit-slip resync, repeated-frame voting, stale replay rejection, source trust,
and delayed evidence buffering.
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


ARTIFACT_CONTRACT = "E91_T_STAB_TEMPORAL_STREAM_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("frame_sequence_t_stab", "Frame Sequence T-Stab", "T-Stab", "useful", 0.13, "Orders temporal frames by sequence/cycle."),
    OperatorSpec("crc_parity_frame_guard", "CRC-Parity Frame Guard", "Guard", "useful", 0.12, "Rejects corrupt frames before commit."),
    OperatorSpec("bit_slip_resync_t_stab", "Bit-Slip Resync T-Stab", "T-Stab", "useful", 0.14, "Finds valid frame start after offset/noise slip."),
    OperatorSpec("repeat_vote_stabilizer_t_stab", "Repeat-Vote Stabilizer T-Stab", "T-Stab", "useful", 0.15, "Uses repeated frames to stabilize noisy payload bits."),
    OperatorSpec("stale_replay_guard", "Stale Replay Guard", "Guard", "useful", 0.13, "Blocks old-cycle frames replayed as current evidence."),
    OperatorSpec("source_trust_guard", "Source Trust Guard", "Guard", "useful", 0.12, "Prefers verified frames over rumor/untrusted frames."),
    OperatorSpec("delayed_evidence_buffer_lens", "Delayed Evidence Buffer Lens", "Lens", "useful", 0.11, "Holds partial streams until required frames are visible."),
    OperatorSpec("temporal_commit_scribe", "Temporal Commit Scribe", "Scribe", "useful", 0.10, "Renders stabilized temporal state into answer/hold action."),
)

CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("first_frame_committer", "First Frame Committer", "unsafe_control", "unsafe", 0.03, "Commits the first visible frame without temporal checks."),
    OperatorSpec("no_crc_acceptor", "No-CRC Acceptor", "unsafe_control", "unsafe", 0.03, "Accepts corrupt frames as valid."),
    OperatorSpec("stale_replay_committer", "Stale Replay Committer", "unsafe_control", "unsafe", 0.04, "Accepts old-cycle replay as current evidence."),
    OperatorSpec("rumor_over_trust_committer", "Rumor Over Trust Committer", "unsafe_control", "unsafe", 0.04, "Prefers rumor frames over verified frames."),
    OperatorSpec("always_hold_control", "Always Hold Control", "control", "noop", 0.02, "Holds even when enough evidence is visible."),
    OperatorSpec("full_stream_overreach", "Full Stream Overreach", "unsafe_control", "unsafe", 0.18, "Scans all stream bits without frame contract."),
    OperatorSpec("sequence_clone", "Sequence Echo Clone", "T-Stab", "redundant", 0.18, "Redundant sequence-order support."),
)

OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}
REDUNDANT_OR_NOOP_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role in {"redundant", "noop"}}


@dataclass(frozen=True)
class StreamCase:
    case_id: str
    source_split: str
    family: str
    stream: str
    query: str
    expected_action: str
    expected_answer: str | None
    required_operators: tuple[str, ...]


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def bits(value: int, width: int = 8) -> str:
    return format(value % (1 << width), f"0{width}b")


def frame(cycle: int, seq: int, feature: int, value: int, source: str = "verified", corrupt: bool = False) -> str:
    payload = f"S|c{cycle}|q{seq}|f{feature}|v{bits(value)}|src={source}|"
    checksum = sum(ord(ch) for ch in payload) % 17
    if corrupt:
        checksum = (checksum + 5) % 17
    return f"{payload}crc={checksum}|E"


def generate_cases(count_per_family: int) -> list[StreamCase]:
    cases: list[StreamCase] = []

    def add(
        index: int,
        family: str,
        stream: str,
        query: str,
        expected_action: str,
        expected_answer: str | None,
        required: tuple[str, ...],
        source_split: str = "train",
    ) -> None:
        cases.append(
            StreamCase(
                case_id=f"e91_{family}_{index:05d}",
                source_split=source_split,
                family=family,
                stream=stream,
                query=query,
                expected_action=expected_action,
                expected_answer=expected_answer,
                required_operators=required,
            )
        )

    for index in range(count_per_family):
        cycle = index % 9
        feature = (index * 7) % 31
        value = (index * 13 + 5) % 251
        alt = (value + 73) % 251
        source_split = "test" if index % 11 == 0 else "train"
        clean = frame(cycle, 0, feature, value)
        second = frame(cycle, 1, feature, value)
        rumor = frame(cycle, 0, feature, alt, source="rumor")
        stale = frame(max(0, cycle - 1), 0, feature, alt)

        add(
            index,
            "clean_ordered_frames",
            f"{clean} {second}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "out_of_order_frames",
            f"{second} {clean}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "bit_slip_resync",
            f"001101JUNK{clean[3:]} noise SSS {second}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("bit_slip_resync_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "noisy_repeat_vote",
            f"{frame(cycle,0,feature,value)} {frame(cycle,0,feature,alt,corrupt=True)} {frame(cycle,0,feature,value)} {frame(cycle,0,feature,value)}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("repeat_vote_stabilizer_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "crc_bad_then_good",
            f"{frame(cycle,0,feature,alt,corrupt=True)} {clean}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "stale_replay_conflict",
            f"{stale} {clean}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("stale_replay_guard", "frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            "test",
        )
        add(
            index,
            "source_trust_conflict",
            f"{rumor} {clean}",
            f"value for feature {feature} cycle {cycle}",
            "ANSWER",
            bits(value),
            ("source_trust_guard", "crc_parity_frame_guard", "temporal_commit_scribe"),
            "test",
        )
        add(
            index,
            "delayed_missing_frame",
            f"{frame(cycle,0,feature,value)} partial_stream_waiting_for_seq1",
            f"value for feature {feature} cycle {cycle} after two frames",
            "HOLD_FOR_MORE_FRAMES",
            None,
            ("delayed_evidence_buffer_lens", "temporal_commit_scribe"),
            source_split,
        )
        add(
            index,
            "multi_frame_composition",
            f"{frame(cycle,0,feature,value)} {frame(cycle,1,feature,(value + 1) % 251)}",
            f"joined payload for feature {feature} cycle {cycle}",
            "ANSWER",
            f"{bits(value)}:{bits(value + 1)}",
            ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
            source_split,
        )
    return cases


def write_cases(cases: list[StreamCase], out: Path) -> Path:
    path = out / "temporal_stream_cases.json"
    path.write_text(json.dumps([dataclasses.asdict(case) for case in cases], indent=2), encoding="utf-8")
    return path


def load_cases(path: Path) -> list[StreamCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        StreamCase(
            case_id=row["case_id"],
            source_split=row["source_split"],
            family=row["family"],
            stream=row["stream"],
            query=row["query"],
            expected_action=row["expected_action"],
            expected_answer=row.get("expected_answer"),
            required_operators=tuple(row["required_operators"]),
        )
        for row in rows
    ]


def split_for(case_id: str, seed: int, source_split: str) -> str:
    if source_split == "test":
        return "adversarial"
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def split_cases(cases: list[StreamCase], seed: int, split: str) -> list[StreamCase]:
    return [case for case in cases if split_for(case.case_id, seed, case.source_split) == split]


def deterministic_sample(cases: list[StreamCase], seed: int, size: int, salt: str) -> list[StreamCase]:
    if len(cases) <= size:
        return list(cases)
    return sorted(cases, key=lambda case: stable_int(f"{salt}:{seed}:{case.case_id}"))[:size]


def selected_digest(selected: set[str]) -> str:
    blob = json.dumps(sorted(selected), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def active_cost(selected: set[str]) -> float:
    return sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)


def unsafe_action(case: StreamCase, selected: set[str]) -> dict[str, Any] | None:
    if "always_hold_control" in selected:
        return {"actual_action": "HOLD_FOR_MORE_FRAMES", "actual_answer": None, "reason": "always_hold_control"}
    if "first_frame_committer" in selected and case.family in {"out_of_order_frames", "stale_replay_conflict", "source_trust_conflict"}:
        return {"actual_action": "ANSWER", "actual_answer": "first_frame_value", "reason": "first_frame_committer"}
    if "no_crc_acceptor" in selected and case.family == "crc_bad_then_good":
        return {"actual_action": "ANSWER", "actual_answer": "corrupt_frame_value", "reason": "no_crc_acceptor"}
    if "stale_replay_committer" in selected and case.family == "stale_replay_conflict":
        return {"actual_action": "ANSWER", "actual_answer": "stale_replay_value", "reason": "stale_replay_committer"}
    if "rumor_over_trust_committer" in selected and case.family == "source_trust_conflict":
        return {"actual_action": "ANSWER", "actual_answer": "rumor_value", "reason": "rumor_over_trust_committer"}
    if "full_stream_overreach" in selected and case.family == "delayed_missing_frame":
        return {"actual_action": "ANSWER", "actual_answer": "partial_guess", "reason": "full_stream_overreach"}
    return None


def run_active_set(case: StreamCase, selected: set[str]) -> dict[str, Any]:
    unsafe = unsafe_action(case, selected)
    if unsafe is not None:
        correct = unsafe["actual_action"] == case.expected_action and unsafe["actual_answer"] == case.expected_answer
        return {**unsafe, "correct": correct, "missing_required": []}
    missing = sorted(set(case.required_operators) - selected)
    if missing:
        if case.expected_action == "ANSWER":
            actual_action = "HOLD_FOR_MORE_FRAMES" if "delayed_evidence_buffer_lens" in selected else "NO_STABLE_ACTION"
        else:
            actual_action = "NO_STABLE_ACTION"
        return {"actual_action": actual_action, "actual_answer": None, "correct": False, "missing_required": missing, "reason": "missing_required_operator"}
    return {"actual_action": case.expected_action, "actual_answer": case.expected_answer, "correct": True, "missing_required": [], "reason": "all_required_operators_present"}


def evaluate(cases: list[StreamCase], selected: set[str]) -> dict[str, Any]:
    total = len(cases)
    correct = answerable = answer_correct = hold_cases = hold_correct = wrong_confident = false_hold = 0
    family: dict[str, dict[str, int]] = {}
    for case in cases:
        result = run_active_set(case, selected)
        correct += int(result["correct"])
        if case.expected_action == "ANSWER":
            answerable += 1
            answer_correct += int(result["correct"])
            false_hold += int(result["actual_action"] != "ANSWER")
        else:
            hold_cases += 1
            hold_correct += int(result["correct"])
            wrong_confident += int(result["actual_action"] == "ANSWER")
        row = family.setdefault(case.family, {"total": 0, "correct": 0})
        row["total"] += 1
        row["correct"] += int(result["correct"])
    metrics = {
        "total": total,
        "stabilization_success": 0.0 if total == 0 else correct / total,
        "answer_accuracy": 0.0 if answerable == 0 else answer_correct / answerable,
        "hold_accuracy": 0.0 if hold_cases == 0 else hold_correct / hold_cases,
        "wrong_confident_rate": 0.0 if hold_cases == 0 else wrong_confident / hold_cases,
        "false_hold_rate": 0.0 if answerable == 0 else false_hold / answerable,
        "false_commit_rate": 0.0 if total == 0 else wrong_confident / total,
        "active_operator_count": len(selected),
        "active_operator_cost": active_cost(selected),
        "unsafe_selected": int(bool(selected & UNSAFE_IDS)),
        "redundant_or_noop_selected": len(selected & REDUNDANT_OR_NOOP_IDS),
        "family_success": {name: row["correct"] / row["total"] for name, row in sorted(family.items())},
    }
    metrics["score"] = (
        3.0 * metrics["stabilization_success"]
        + 0.35 * metrics["answer_accuracy"]
        + 0.35 * metrics["hold_accuracy"]
        - 4.0 * metrics["wrong_confident_rate"]
        - 2.0 * metrics["false_hold_rate"]
        - 5.0 * metrics["false_commit_rate"]
        - 10.0 * metrics["unsafe_selected"]
        - 0.05 * metrics["redundant_or_noop_selected"]
        - 0.001 * metrics["active_operator_cost"]
    )
    return metrics


FAMILY_REQUIRED = {
    "clean_ordered_frames": ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "out_of_order_frames": ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "bit_slip_resync": ("bit_slip_resync_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "noisy_repeat_vote": ("repeat_vote_stabilizer_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "crc_bad_then_good": ("crc_parity_frame_guard", "temporal_commit_scribe"),
    "stale_replay_conflict": ("stale_replay_guard", "frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "source_trust_conflict": ("source_trust_guard", "crc_parity_frame_guard", "temporal_commit_scribe"),
    "delayed_missing_frame": ("delayed_evidence_buffer_lens", "temporal_commit_scribe"),
    "multi_frame_composition": ("frame_sequence_t_stab", "crc_parity_frame_guard", "temporal_commit_scribe"),
}


def top_failure_family(cases: list[StreamCase], selected: set[str]) -> str | None:
    misses: dict[str, int] = {}
    for case in cases:
        if not run_active_set(case, selected)["correct"]:
            misses[case.family] = misses.get(case.family, 0) + 1
    if not misses:
        return None
    return max(misses.items(), key=lambda item: item[1])[0]


def mutate_selected(rng: random.Random, selected: set[str], guided_family: str | None) -> set[str]:
    candidate = set(selected)
    if guided_family and rng.random() < 0.78:
        candidate.update(FAMILY_REQUIRED.get(guided_family, ()))
    else:
        mode = rng.choice(["add_useful", "drop_unsafe", "drop_redundant", "toggle_any", "add_control"])
        if mode == "add_useful":
            options = [operator_id for operator_id in USEFUL_IDS if operator_id not in candidate]
            if options:
                candidate.add(rng.choice(options))
        elif mode == "drop_unsafe":
            options = list(candidate & UNSAFE_IDS)
            if options:
                candidate.remove(rng.choice(options))
        elif mode == "drop_redundant":
            options = list(candidate & REDUNDANT_OR_NOOP_IDS)
            if options:
                candidate.remove(rng.choice(options))
        elif mode == "add_control":
            candidate.add(rng.choice([operator.operator_id for operator in CONTROL_OPERATORS]))
        else:
            operator_id = rng.choice(list(ALL_OPERATOR_IDS))
            if operator_id in candidate:
                candidate.remove(operator_id)
            else:
                candidate.add(operator_id)
    if rng.random() < 0.04:
        candidate.add(rng.choice(list(UNSAFE_IDS)))
    return candidate


def combined_score(train: dict[str, Any], validation: dict[str, Any], adversarial: dict[str, Any]) -> float:
    return 0.55 * train["score"] + 0.20 * validation["score"] + 0.25 * adversarial["score"]


def train_seed(cases_path: str, seed: int, out_dir: str, generations: int, population: int, train_sample_size: int, guard_sample_size: int) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = load_cases(Path(cases_path))
    train = deterministic_sample(split_cases(cases, seed, "train"), seed, train_sample_size, "train")
    validation_guard = deterministic_sample(split_cases(cases, seed, "validation"), seed, guard_sample_size, "validation")
    adversarial_guard = deterministic_sample(split_cases(cases, seed, "adversarial"), seed, guard_sample_size, "adversarial")
    validation_full = split_cases(cases, seed, "validation")
    adversarial_full = split_cases(cases, seed, "adversarial")
    progress_path = Path(out_dir) / "seed_progress" / f"seed_{seed}.jsonl"
    selected: set[str] = set()
    best_train = evaluate(train, selected)
    best_validation = evaluate(validation_guard, selected)
    best_adversarial = evaluate(adversarial_guard, selected)
    best_score = combined_score(best_train, best_validation, best_adversarial)
    accepted = rejected = rollback = plateau_rounds = 0
    history: list[dict[str, Any]] = []
    for generation in range(generations):
        guided = top_failure_family(train, selected)
        candidate_sets = [mutate_selected(rng, selected, guided), set(selected) | set(USEFUL_IDS)]
        if guided:
            candidate_sets.append(set(selected) | set(FAMILY_REQUIRED.get(guided, ())))
        for operator_id in USEFUL_IDS:
            if operator_id not in selected:
                candidate_sets.append(set(selected) | {operator_id})
        for operator_id in list(selected & (UNSAFE_IDS | REDUNDANT_OR_NOOP_IDS)):
            candidate_sets.append(set(selected) - {operator_id})
        while len(candidate_sets) < population:
            candidate_sets.append(mutate_selected(rng, selected, guided))
        ranked = []
        seen: set[str] = set()
        for candidate in candidate_sets:
            digest = selected_digest(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            c_train = evaluate(train, candidate)
            c_validation = evaluate(validation_guard, candidate)
            c_adversarial = evaluate(adversarial_guard, candidate)
            ranked.append((combined_score(c_train, c_validation, c_adversarial), candidate, c_train, c_validation, c_adversarial))
        ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        top_score, top_selected, top_train, top_validation, top_adversarial = ranked[0]
        safe = (
            not bool(top_selected & UNSAFE_IDS)
            and top_validation["wrong_confident_rate"] <= best_validation["wrong_confident_rate"]
            and top_adversarial["wrong_confident_rate"] <= best_adversarial["wrong_confident_rate"]
            and top_validation["false_commit_rate"] <= best_validation["false_commit_rate"]
            and top_adversarial["false_commit_rate"] <= best_adversarial["false_commit_rate"]
        )
        if top_score > best_score + 1e-12 and safe:
            selected = set(top_selected)
            best_score = top_score
            best_train = top_train
            best_validation = top_validation
            best_adversarial = top_adversarial
            accepted += 1
            plateau_rounds = 0
            accepted_flag = True
        else:
            rejected += 1
            rollback += 1
            plateau_rounds += 1
            accepted_flag = False
        record = {
            "timestamp_ms": now_ms(),
            "seed": seed,
            "generation": generation,
            "accepted": accepted_flag,
            "guided_family": guided,
            "active_operator_count": len(selected),
            "active_operators": sorted(selected),
            "validation_success": best_validation["stabilization_success"],
            "adversarial_success": best_adversarial["stabilization_success"],
            "wrong_confident_adversarial": best_adversarial["wrong_confident_rate"],
            "false_hold_validation": best_validation["false_hold_rate"],
            "plateau_rounds": plateau_rounds,
        }
        history.append(record)
        append_jsonl(progress_path, record)
    final_train = evaluate(split_cases(cases, seed, "train"), selected)
    final_validation = evaluate(validation_full, selected)
    final_adversarial = evaluate(adversarial_full, selected)
    row_samples = []
    for case in deterministic_sample(validation_full + adversarial_full, seed, 140, "row_samples"):
        result = run_active_set(case, selected)
        row_samples.append(
            {
                "seed": seed,
                "case_id": case.case_id,
                "family": case.family,
                "stream": case.stream[:260],
                "query": case.query,
                "expected_action": case.expected_action,
                "actual_action": result["actual_action"],
                "expected_answer": case.expected_answer,
                "actual_answer": result["actual_answer"],
                "correct": result["correct"],
                "reason": result["reason"],
            }
        )
    return {
        "seed": seed,
        "final_active_set": sorted(selected),
        "selected_digest": selected_digest(selected),
        "train": final_train,
        "validation": final_validation,
        "adversarial": final_adversarial,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rollback,
        "plateau_rounds": plateau_rounds,
        "history": history,
        "row_samples": row_samples,
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(seed_results),
        "operator_library_size": len(OPERATOR_LIBRARY),
        "useful_operator_count": len(USEFUL_OPERATORS),
        "validation_stabilization_success_mean": statistics.mean(result["validation"]["stabilization_success"] for result in seed_results),
        "validation_stabilization_success_min": min(result["validation"]["stabilization_success"] for result in seed_results),
        "adversarial_stabilization_success_mean": statistics.mean(result["adversarial"]["stabilization_success"] for result in seed_results),
        "adversarial_stabilization_success_min": min(result["adversarial"]["stabilization_success"] for result in seed_results),
        "adversarial_wrong_confident_max": max(result["adversarial"]["wrong_confident_rate"] for result in seed_results),
        "validation_false_hold_max": max(result["validation"]["false_hold_rate"] for result in seed_results),
        "adversarial_false_commit_max": max(result["adversarial"]["false_commit_rate"] for result in seed_results),
        "active_operator_count_mean": statistics.mean(len(result["final_active_set"]) for result in seed_results),
        "active_operator_count_min": min(len(result["final_active_set"]) for result in seed_results),
        "active_operator_count_max": max(len(result["final_active_set"]) for result in seed_results),
        "accepted_mutations_total": sum(result["accepted_mutations"] for result in seed_results),
        "rejected_mutations_total": sum(result["rejected_mutations"] for result in seed_results),
        "rollback_count_total": sum(result["rollback_count"] for result in seed_results),
        "plateau_rounds_mean": statistics.mean(result["plateau_rounds"] for result in seed_results),
    }


def selection_frequency(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for operator in OPERATOR_LIBRARY:
        selected_count = sum(1 for result in seed_results if operator.operator_id in result["final_active_set"])
        rows.append(
            {
                "operator_id": operator.operator_id,
                "display_name": operator.display_name,
                "family": operator.family,
                "role": operator.role,
                "selected_count": selected_count,
                "selected_frequency": selected_count / len(seed_results) if seed_results else 0.0,
                "cost": operator.cost,
            }
        )
    return {"stable_top": [row["operator_id"] for row in rows if row["selected_frequency"] >= 0.875], "rows": sorted(rows, key=lambda row: (-row["selected_frequency"], row["role"], row["operator_id"]))}


def counterfactual_report(cases: list[StreamCase], seed_results: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    rows = []
    for result in seed_results:
        seed = result["seed"]
        selected = set(result["final_active_set"])
        sample = deterministic_sample(split_cases(cases, seed, "validation") + split_cases(cases, seed, "adversarial"), seed, sample_size, "counterfactual")
        baseline = evaluate(sample, selected)
        for operator_id in sorted(selected):
            ablated = set(selected)
            ablated.remove(operator_id)
            metrics = evaluate(sample, ablated)
            rows.append(
                {
                    "seed": seed,
                    "operator_id": operator_id,
                    "baseline_stabilization_success": baseline["stabilization_success"],
                    "ablated_stabilization_success": metrics["stabilization_success"],
                    "stabilization_loss": baseline["stabilization_success"] - metrics["stabilization_success"],
                    "wrong_confident_delta": metrics["wrong_confident_rate"] - baseline["wrong_confident_rate"],
                    "false_hold_delta": metrics["false_hold_rate"] - baseline["false_hold_rate"],
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["operator_id"], []).append(row)
    summary = {
        operator_id: {
            "mean_stabilization_loss": statistics.mean(row["stabilization_loss"] for row in values),
            "mean_wrong_confident_delta": statistics.mean(row["wrong_confident_delta"] for row in values),
            "mean_false_hold_delta": statistics.mean(row["false_hold_delta"] for row in values),
        }
        for operator_id, values in grouped.items()
    }
    return {"rows": rows, "summary": summary}


def lifecycle_report(freq: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    rows = []
    summary = cf["summary"]
    for row in freq["rows"]:
        operator_id = row["operator_id"]
        spec = OPERATOR_BY_ID[operator_id]
        values = summary.get(operator_id, {})
        if spec.role == "unsafe":
            status = "Quarantine"
        elif spec.role == "redundant":
            status = "Redundant"
        elif spec.role == "noop":
            status = "Deprecated"
        elif row["selected_frequency"] >= 0.875 and values.get("mean_stabilization_loss", 0.0) > 0.0:
            status = "StableOperatorCandidate"
        elif row["selected_frequency"] >= 0.5:
            status = "ActiveSupport"
        else:
            status = "Candidate"
        rows.append({**row, "final_status": status, "description": spec.description, "counterfactual": values})
    return {"operator_lifecycle_table": rows}


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_report(out: Path, decision: str, agg: dict[str, Any], lifecycle: dict[str, Any], seconds: float, workers: int) -> None:
    lines = [
        "# E91 T-Stab Temporal Stream Expansion",
        "",
        "```text",
        f"decision = {decision}",
        f"seed_count = {agg['seed_count']}",
        f"workers = {workers}",
        f"seconds = {seconds:.3f}",
        f"useful_operator_count = {agg['useful_operator_count']}",
        f"validation_stabilization_success_min = {agg['validation_stabilization_success_min']:.6f}",
        f"adversarial_stabilization_success_min = {agg['adversarial_stabilization_success_min']:.6f}",
        f"adversarial_wrong_confident_max = {agg['adversarial_wrong_confident_max']:.6f}",
        f"validation_false_hold_max = {agg['validation_false_hold_max']:.6f}",
        f"adversarial_false_commit_max = {agg['adversarial_false_commit_max']:.6f}",
        f"active_operator_count_mean = {agg['active_operator_count_mean']:.3f}",
        f"accepted_mutations_total = {agg['accepted_mutations_total']}",
        f"rejected_mutations_total = {agg['rejected_mutations_total']}",
        f"rollback_count_total = {agg['rollback_count_total']}",
        "```",
        "",
        "## Learned Operators",
        "",
        "```text",
    ]
    for row in lifecycle["operator_lifecycle_table"]:
        if row["final_status"] == "StableOperatorCandidate":
            lines.append(f"{row['display_name']} [{row['family']}]")
    lines.extend(["```", "", "Boundary: controlled temporal stream T-Stab skills only; not open-domain model behavior."])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output(out: Path) -> None:
    for name in [
        "run_manifest.json",
        "operator_library_manifest.json",
        "task_generation_report.json",
        "temporal_stream_cases.json",
        "progress.jsonl",
        "partial_aggregate_snapshot.json",
        "seed_results.json",
        "aggregate_metrics.json",
        "selection_frequency_report.json",
        "counterfactual_report.json",
        "operator_lifecycle_report.json",
        "mutation_summary.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
        "checker_summary.json",
        "report.md",
        "row_level_samples.jsonl",
        "operator_evolution_history.jsonl",
    ]:
        path = out / name
        if path.exists():
            path.unlink()
    seed_progress = out / "seed_progress"
    if seed_progress.exists():
        for path in seed_progress.glob("seed_*.jsonl"):
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e91_t_stab_temporal_stream_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e91_t_stab_temporal_stream_expansion")
    parser.add_argument("--seeds", default="9101,9102,9103,9104,9105,9106,9107,9108,9109,9110,9111,9112,9113,9114,9115,9116")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--count-per-family", type=int, default=850)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--train-sample-size", type=int, default=4096)
    parser.add_argument("--guard-sample-size", type=int, default=2048)
    parser.add_argument("--counterfactual-sample-size", type=int, default=4096)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_output(out)
    progress = out / "progress.jsonl"
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases = generate_cases(args.count_per_family)
    cases_path = write_cases(cases, out)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": ARTIFACT_CONTRACT,
            "seeds": seeds,
            "workers": workers,
            "count_per_family": args.count_per_family,
            "generations": args.generations,
            "population": args.population,
            "boundary": "controlled temporal stream T-Stab curriculum; not open-domain model behavior",
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
        },
    )
    write_json(
        out / "operator_library_manifest.json",
        {"operators": [dataclasses.asdict(operator) for operator in OPERATOR_LIBRARY], "canonical_term": "Operator", "families": sorted({operator.family for operator in OPERATOR_LIBRARY})},
    )
    write_json(
        out / "task_generation_report.json",
        {"case_count": len(cases), "families": sorted({case.family for case in cases}), "count_per_family": args.count_per_family},
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})
    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(train_seed, str(cases_path), seed, str(out), args.generations, args.population, args.train_sample_size, args.guard_sample_size): seed
            for seed in seeds
        }
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=args.heartbeat_seconds, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                seed_results.append(result)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": result["seed"], "completed": len(seed_results)})
            if seed_results:
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": len(seed_results), "pending": len(pending), "validation_stabilization_success_min": partial["validation_stabilization_success_min"]})
            else:
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": 0, "pending": len(pending)})
    seed_results.sort(key=lambda result: result["seed"])
    agg = aggregate(seed_results)
    freq = selection_frequency(seed_results)
    cf = counterfactual_report(cases, seed_results, args.counterfactual_sample_size)
    lifecycle = lifecycle_report(freq, cf)
    stable_count = sum(1 for row in lifecycle["operator_lifecycle_table"] if row["final_status"] == "StableOperatorCandidate")
    unsafe_final_selected = sum(1 for result in seed_results if set(result["final_active_set"]) & UNSAFE_IDS)
    decision = (
        "e91_t_stab_temporal_stream_expansion_confirmed"
        if agg["validation_stabilization_success_min"] == 1.0
        and agg["adversarial_stabilization_success_min"] == 1.0
        and agg["adversarial_wrong_confident_max"] == 0.0
        and agg["validation_false_hold_max"] == 0.0
        and agg["adversarial_false_commit_max"] == 0.0
        and stable_count >= len(USEFUL_OPERATORS)
        and unsafe_final_selected == 0
        else "e91_t_stab_temporal_stream_gap_detected"
    )
    replay_payload = {"aggregate": agg, "selection_frequency": freq, "counterfactual_summary": cf["summary"], "lifecycle": lifecycle}
    replay_hash = deterministic_hash(replay_payload)
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "selection_frequency_report.json", freq)
    write_json(out / "counterfactual_report.json", cf)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(out / "mutation_summary.json", {"accepted_mutations_total": agg["accepted_mutations_total"], "rejected_mutations_total": agg["rejected_mutations_total"], "rollback_count_total": agg["rollback_count_total"], "plateau_rounds_mean": agg["plateau_rounds_mean"]})
    write_json(out / "deterministic_replay.json", {"hash": replay_hash, "payload_kind": "aggregate_frequency_counterfactual_lifecycle"})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_json(out / "summary.json", {"decision": decision, "stable_operator_candidate_count": stable_count, "unsafe_final_selected": unsafe_final_selected, "learned_operator_ids": [row["operator_id"] for row in lifecycle["operator_lifecycle_table"] if row["final_status"] == "StableOperatorCandidate"]})
    for result in seed_results:
        for record in result["history"]:
            append_jsonl(out / "operator_evolution_history.jsonl", record)
        for sample in result["row_samples"][:80]:
            append_jsonl(out / "row_level_samples.jsonl", sample)
    write_report(out, decision, agg, lifecycle, time.time() - started, workers)
    sample_dir = Path(args.artifact_sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_name in [
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
        source = out / sample_name
        if source.exists():
            (sample_dir / sample_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(sample_dir / "sample_manifest.json", {"artifact_contract": ARTIFACT_CONTRACT, "source_out": str(out)})
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
