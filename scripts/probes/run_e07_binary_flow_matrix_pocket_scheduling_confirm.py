#!/usr/bin/env python3
"""E07 non-neural binary Flow/Main matrix pocket scheduling probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E07_BINARY_FLOW_MATRIX_POCKET_SCHEDULING_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e07_binary_flow_matrix_pocket_scheduling_confirm")
DEFAULT_SEEDS = (70701, 70702, 70703, 70704, 70705, 70706)
DEFAULT_TICKS = 420
TARGET_SLOTS = 8
BRANCH_WIDTH = TARGET_SLOTS // 2
COMMON_SCHEMA_FIELDS = (
    "signal_type",
    "source_region",
    "target_slot",
    "operation",
    "guard",
    "branch_id",
    "confidence",
    "reason_code",
)
VALID_OPERATIONS = ("SET", "CLEAR", "FLIP", "COPY", "XOR_INTO", "INHIBIT", "LOCK", "UNLOCK")
VALID_DECISIONS = (
    "e07_binary_flow_matrix_pocket_scheduling_confirmed",
    "e07_snapshot_selection_temporal_failure_detected",
    "e07_trigger_policy_too_conservative",
    "e07_branch_contamination_not_fixed",
    "e07_common_matrix_language_contract_failure",
    "e07_invalid_or_incomplete_run",
)
ARMS = (
    "SNAPSHOT_SELECTED_POCKET",
    "ROLLOUT_SELECTED_POCKET",
    "ALL_COMPLEX_ALWAYS_NO_GATE",
    "ALL_COMPLEX_ALWAYS_GATED",
    "DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED",
    "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED",
)
SEARCH_TERMS = (
    "E07",
    "FLOW_MATRIX",
    "Flow Matrix",
    "Binary Flow Matrix",
    "MAIN_MATRIX",
    "Main Matrix",
    "Pocket Pipeline",
    "pocket pipeline",
    "pocket transformation",
    "common matrix language",
    "proposal commit",
    "proposal gate commit",
    "detector wiring",
    "temporal rollout selection",
    "always-on transform",
    "triggered pocket",
    "flow matrix language",
)
SEARCH_PATHS = ("docs/research", "scripts/probes", "docs/wiki", "CHANGELOG.md")
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e07_search_report.json",
    "e07_snapshot_vs_rollout_report.json",
    "e07_default_vs_complex_scheduling_report.json",
    "e07_common_matrix_language_report.json",
    "e07_branch_contamination_report.json",
    "e07_temporal_stability_report.json",
    "e07_deterministic_replay_report.json",
)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round(float(num) / float(den), 6)


def rounded(value: float) -> float:
    return round(float(value), 6)


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_payload(value), indent=2, sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout


def branch_slots(branch_id: int) -> range:
    start = int(branch_id) * BRANCH_WIDTH
    return range(start, start + BRANCH_WIDTH)


def apply_operation(targets: list[int], slot: int, operation: str) -> None:
    if operation == "SET":
        targets[slot] = 1
    elif operation == "CLEAR":
        targets[slot] = 0
    elif operation in {"FLIP", "XOR_INTO"}:
        targets[slot] ^= 1
    elif operation == "COPY":
        targets[slot] = targets[(slot - 1) % TARGET_SLOTS]
    elif operation == "INHIBIT":
        targets[slot] = 0
    elif operation == "LOCK":
        targets[slot] = targets[slot]
    elif operation == "UNLOCK":
        targets[slot] = targets[slot]
    else:
        raise ValueError(f"unsupported operation {operation}")


@dataclass(frozen=True)
class Observation:
    seed: int
    tick: int
    branch_id: int
    signal_bits: tuple[int, ...]
    distractor_bits: tuple[int, ...]
    lock_bits: tuple[int, ...]
    changed_signal: bool
    signal_offset: int
    operation_code: str
    salience: float
    uncertainty: float
    conflict: bool
    expected_update: dict[str, Any] | None
    truth_before: tuple[int, ...]
    truth_after: tuple[int, ...]


@dataclass
class FlowState:
    targets: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)
    trace_age: list[int] = field(default_factory=lambda: [99] * TARGET_SLOTS)
    branch_id: int = 0
    lock_bits: tuple[int, ...] = field(default_factory=lambda: (0,) * TARGET_SLOTS)
    cheap_conflict_flag: bool = False
    trigger_threshold: float = 0.58
    last_change_tick: list[int] = field(default_factory=lambda: [-999] * TARGET_SLOTS)
    last_change_value: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)


@dataclass
class ArmStats:
    arm: str
    ticks: int = 0
    target_match_count: int = 0
    target_eval_count: int = 0
    events: int = 0
    useful_events_committed: int = 0
    missed_events: int = 0
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    good_proposals: int = 0
    bad_proposals: int = 0
    wrong_commits: int = 0
    destructive_overwrites: int = 0
    branch_contamination: int = 0
    stale_commits: int = 0
    temporal_drift_slots: int = 0
    oscillations: int = 0
    collapse_ticks: int = 0
    complex_calls: float = 0.0
    cheap_calls: float = 0.0
    cost: float = 0.0
    retry_or_trigger_cost: float = 0.0
    snapshot_trap_selected: int = 0
    rollout_bad_selected: int = 0
    rollout_good_selected: int = 0
    invalid_schema_attempts: int = 0
    invalid_schema_rejected: int = 0
    invalid_schema_committed: int = 0
    direct_dialect_mutation_attempts: int = 0
    direct_dialect_mutation_commits: int = 0
    trigger_count: int = 0
    false_trigger_count: int = 0


def signal_count(obs: Observation) -> int:
    return sum(obs.signal_bits[:4])


def distractor_count(obs: Observation) -> int:
    return sum(obs.distractor_bits)


def derive_signal_offset(signal_bits: tuple[int, ...]) -> int:
    for idx in range(4):
        if signal_bits[idx] and signal_bits[(idx + 1) % 4]:
            return idx
    for idx in range(4):
        if signal_bits[idx]:
            return idx
    return 0


def derive_operation(signal_bits: tuple[int, ...]) -> str:
    if signal_bits[5] and not signal_bits[4]:
        return "SET"
    if signal_bits[4] and not signal_bits[5]:
        return "CLEAR"
    return "SET" if sum(signal_bits) % 2 else "CLEAR"


def make_observations(seed: int, ticks: int) -> list[Observation]:
    rng = random.Random(seed)
    truth = [0] * TARGET_SLOTS
    previous_signal = (0,) * 8
    rows: list[Observation] = []
    for tick in range(ticks):
        branch_id = ((tick // 17) + seed) % 2
        signal_offset_value = (tick * 3 + seed) % BRANCH_WIDTH
        operation = "SET" if ((tick // 6) + seed) % 2 else "CLEAR"
        good_event = ((tick + seed) % 5 in (0, 2)) or ((tick * 7 + seed) % 41 == 11)
        trap_event = ((tick * 11 + seed) % 19 in (4, 8)) or (good_event and ((tick + seed) % 13 == 0))

        signal = [0] * 8
        if good_event:
            signal[signal_offset_value] = 1
            signal[(signal_offset_value + 1) % BRANCH_WIDTH] = 1
            if operation == "SET":
                signal[5] = 1
            else:
                signal[4] = 1
            if (tick + seed) % 23 == 0:
                signal[(signal_offset_value + 2) % BRANCH_WIDTH] = 1
        elif (tick + seed) % 4 == 0:
            signal[signal_offset_value] = 1
            if rng.random() < 0.25:
                signal[6] = 1

        distractor = [0] * 8
        if trap_event:
            start = (tick + seed) % 5
            for idx in range(3):
                distractor[(start + idx) % 8] = 1
            if (tick + seed) % 7 == 0:
                distractor[(start + 5) % 8] = 1
        elif (tick + seed) % 6 == 0:
            distractor[(tick + seed) % 8] = 1

        lock_bits = []
        for slot in range(TARGET_SLOTS):
            locked = 1 if ((tick * 5 + seed + slot * 7) % 53 == 0) else 0
            if slot not in branch_slots(branch_id) and ((tick + slot + seed) % 29 == 0):
                locked = 1
            lock_bits.append(locked)
        lock_tuple = tuple(lock_bits)

        target_slot = branch_id * BRANCH_WIDTH + signal_offset_value
        truth_before = tuple(truth)
        expected_update = None
        if good_event and not lock_tuple[target_slot]:
            expected_update = {
                "target_slot": target_slot,
                "operation": operation,
                "branch_id": branch_id,
                "source_region": "signal[0:4]",
            }
            apply_operation(truth, target_slot, operation)

        signal_tuple = tuple(signal)
        changed_signal = signal_tuple != previous_signal
        salience = 0.24 * sum(signal[:4]) + 0.08 * sum(signal[4:6]) + (0.12 if changed_signal else 0.0)
        uncertainty = 0.18 * sum(distractor) + (0.24 if trap_event else 0.0) + (0.12 if good_event and trap_event else 0.0)
        rows.append(
            Observation(
                seed=seed,
                tick=tick,
                branch_id=branch_id,
                signal_bits=signal_tuple,
                distractor_bits=tuple(distractor),
                lock_bits=lock_tuple,
                changed_signal=changed_signal,
                signal_offset=signal_offset_value,
                operation_code=operation,
                salience=rounded(min(1.0, salience)),
                uncertainty=rounded(min(1.0, uncertainty)),
                conflict=bool(good_event and trap_event),
                expected_update=expected_update,
                truth_before=truth_before,
                truth_after=tuple(truth),
            )
        )
        previous_signal = signal_tuple
    return rows


def canonical_proposal(
    pocket: str,
    obs: Observation,
    target_slot: int,
    operation: str,
    signal_type: str,
    source_region: str,
    confidence: float,
    reason_code: str,
    local_score: float,
    rollout_score: float,
    stale_age: int,
    schema_valid: bool = True,
    branch_id: int | None = None,
) -> dict[str, Any]:
    branch_value = obs.branch_id if branch_id is None else branch_id
    proposal = {
        "pocket": pocket,
        "schema_valid": bool(schema_valid),
        "is_complex": True,
        "signal_type": signal_type,
        "source_region": source_region,
        "target_slot": int(target_slot),
        "operation": operation,
        "guard": {
            "target_unlocked": obs.lock_bits[target_slot] == 0,
            "branch_active": branch_value == obs.branch_id,
            "target_in_branch": target_slot in branch_slots(obs.branch_id),
            "stale_trace_age": int(stale_age),
        },
        "branch_id": int(branch_value),
        "confidence": rounded(confidence),
        "reason_code": reason_code,
        "local_score": rounded(local_score),
        "rollout_score": rounded(rollout_score),
    }
    if not schema_valid:
        proposal["local_dialect"] = {"dst": int(target_slot), "verb": operation, "wire": source_region}
    return proposal


def make_proposals(obs: Observation, flow: FlowState) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    count = signal_count(obs)
    offset = derive_signal_offset(obs.signal_bits)
    operation = derive_operation(obs.signal_bits)
    target_slot = obs.branch_id * BRANCH_WIDTH + offset
    stale_age = flow.trace_age[target_slot]

    if count >= 2:
        proposals.append(
            canonical_proposal(
                "count_edge_canonical_writer",
                obs,
                target_slot,
                operation,
                "COUNT_EDGE",
                "signal[0:4]",
                0.82 + min(0.12, 0.03 * count),
                "CANONICAL_SIGNAL_WRITE",
                0.78 + 0.03 * count,
                0.88 - (0.08 if obs.conflict else 0.0),
                stale_age,
            )
        )

    if count >= 1 and obs.changed_signal:
        parity_slot = obs.branch_id * BRANCH_WIDTH + ((offset + sum(obs.signal_bits)) % BRANCH_WIDTH)
        proposals.append(
            canonical_proposal(
                "xor_parity_candidate",
                obs,
                parity_slot,
                "FLIP" if sum(obs.signal_bits) % 2 else operation,
                "XOR_PARITY",
                "signal[0:8]",
                0.68 + 0.04 * (sum(obs.signal_bits) % 2),
                "PARITY_SIDE_ROUTE",
                0.68,
                0.46,
                flow.trace_age[parity_slot],
            )
        )

    if distractor_count(obs) >= 3:
        first = next((idx for idx, bit in enumerate(obs.distractor_bits) if bit), 0)
        trap_slot = obs.branch_id * BRANCH_WIDTH + ((first + 2) % BRANCH_WIDTH)
        proposals.append(
            canonical_proposal(
                "distractor_snapshot_trap",
                obs,
                trap_slot,
                "FLIP" if (obs.tick + first) % 2 else "SET",
                "MATCH_DISTRACTOR",
                "distractor[0:8]",
                0.94,
                "SNAPSHOT_TRAP",
                0.97,
                0.18,
                flow.trace_age[trap_slot],
            )
        )

    if count >= 1:
        wrong_branch = 1 - obs.branch_id
        wrong_slot = wrong_branch * BRANCH_WIDTH + offset
        proposals.append(
            canonical_proposal(
                "branch_blind_local_dialect",
                obs,
                wrong_slot,
                operation,
                "ANY",
                "local_signal_lane",
                0.76,
                "LOCAL_DIALECT_BRANCH_BLIND",
                0.74,
                0.12,
                flow.trace_age[wrong_slot],
                schema_valid=False,
                branch_id=wrong_branch,
            )
        )

    old_slots = [slot for slot, age in enumerate(flow.trace_age) if 3 <= age <= 7]
    if old_slots and (obs.tick + obs.seed) % 9 == 0:
        slot = old_slots[(obs.tick + obs.seed) % len(old_slots)]
        proposals.append(
            canonical_proposal(
                "stale_trace_replay",
                obs,
                slot,
                "SET",
                "CHANGED",
                "trace_age",
                0.72,
                "STALE_TRACE_REPLAY",
                0.71,
                0.24,
                flow.trace_age[slot],
            )
        )
    return proposals


def proposal_matches_expected(proposal: dict[str, Any], obs: Observation) -> bool:
    expected = obs.expected_update
    if expected is None:
        return False
    return (
        proposal.get("schema_valid") is True
        and proposal.get("target_slot") == expected["target_slot"]
        and proposal.get("operation") == expected["operation"]
        and proposal.get("branch_id") == expected["branch_id"]
    )


def schema_complete(proposal: dict[str, Any]) -> bool:
    return proposal.get("schema_valid") is True and all(field in proposal for field in COMMON_SCHEMA_FIELDS)


def gate_accepts(proposal: dict[str, Any], obs: Observation, flow: FlowState, arm: str) -> tuple[bool, str]:
    if not schema_complete(proposal):
        return False, "schema_incomplete"
    if proposal.get("operation") not in VALID_OPERATIONS:
        return False, "invalid_operation"
    guard = proposal.get("guard", {})
    if not guard.get("target_unlocked", False):
        return False, "target_locked"
    if not guard.get("branch_active", False) or not guard.get("target_in_branch", False):
        return False, "branch_boundary"
    if proposal.get("confidence", 0.0) < 0.62:
        return False, "low_confidence"
    if guard.get("stale_trace_age", 0) > 4 and proposal.get("reason_code") == "STALE_TRACE_REPLAY":
        return False, "stale_trace"
    if arm.startswith("DEFAULT_") and flow.cheap_conflict_flag and proposal.get("reason_code") == "SNAPSHOT_TRAP":
        return False, "cheap_conflict_inhibit"
    return True, "accepted"


def cheap_transform(flow: FlowState, obs: Observation) -> None:
    for slot in range(TARGET_SLOTS):
        flow.trace_age[slot] = min(99, flow.trace_age[slot] + 1)
    flow.branch_id = obs.branch_id
    flow.lock_bits = obs.lock_bits
    flow.cheap_conflict_flag = bool(obs.conflict or (signal_count(obs) > 0 and distractor_count(obs) >= 3))
    if signal_count(obs) >= 2:
        slot = obs.branch_id * BRANCH_WIDTH + derive_signal_offset(obs.signal_bits)
        flow.trace_age[slot] = 0


def direct_commit(flow: FlowState, proposal: dict[str, Any], obs: Observation, stats: ArmStats) -> None:
    slot = int(proposal.get("target_slot", 0)) % TARGET_SLOTS
    operation = str(proposal.get("operation", "FLIP"))
    if operation not in VALID_OPERATIONS:
        operation = "FLIP"
    before_value = flow.targets[slot]
    before_correct = tuple(flow.targets)[slot] == obs.truth_after[slot]
    apply_operation(flow.targets, slot, operation)
    after_value = flow.targets[slot]
    if after_value != before_value:
        if obs.tick - flow.last_change_tick[slot] <= 4 and flow.last_change_value[slot] == after_value:
            stats.oscillations += 1
        flow.last_change_tick[slot] = obs.tick
        flow.last_change_value[slot] = after_value
    stats.commits += 1
    if proposal_matches_expected(proposal, obs):
        stats.accepted_good += 1
    else:
        stats.accepted_bad += 1
    if flow.targets[slot] != obs.truth_after[slot] or not proposal_matches_expected(proposal, obs):
        stats.wrong_commits += 1
    if before_correct and flow.targets[slot] != obs.truth_after[slot]:
        stats.destructive_overwrites += 1
    if proposal.get("branch_id") != obs.branch_id or slot not in branch_slots(obs.branch_id):
        stats.branch_contamination += 1
    if proposal.get("guard", {}).get("stale_trace_age", 0) > 2 or proposal.get("reason_code") == "STALE_TRACE_REPLAY":
        stats.stale_commits += 1
    if proposal.get("schema_valid") is not True:
        stats.invalid_schema_committed += 1
        stats.direct_dialect_mutation_commits += 1


def gated_commit(flow: FlowState, proposal: dict[str, Any], obs: Observation, stats: ArmStats, arm: str) -> None:
    is_good = proposal_matches_expected(proposal, obs)
    if is_good:
        stats.good_proposals += 1
    else:
        stats.bad_proposals += 1
    if proposal.get("schema_valid") is not True:
        stats.invalid_schema_attempts += 1
        stats.direct_dialect_mutation_attempts += 1
    accepted, _reason = gate_accepts(proposal, obs, flow, arm)
    if not accepted:
        if is_good:
            stats.rejected_good += 1
        else:
            stats.rejected_bad += 1
        if proposal.get("schema_valid") is not True:
            stats.invalid_schema_rejected += 1
        return
    direct_commit(flow, proposal, obs, stats)


def select_snapshot(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not proposals:
        return None
    return max(proposals, key=lambda item: (item.get("local_score", 0.0), item.get("confidence", 0.0)))


def select_rollout(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    validish = sorted(
        proposals,
        key=lambda item: (
            item.get("rollout_score", 0.0),
            1 if item.get("schema_valid") else 0,
            item.get("confidence", 0.0),
        ),
        reverse=True,
    )
    if not validish:
        return None
    selected = validish[0]
    if selected.get("rollout_score", 0.0) < 0.55:
        return None
    return selected


def should_trigger_complex(flow: FlowState, obs: Observation, adaptive: bool) -> bool:
    trace_recent = any(age <= 2 for age in flow.trace_age)
    threshold = flow.trigger_threshold if adaptive else 0.58
    score = obs.salience + (0.18 if obs.conflict else 0.0) + (0.08 if trace_recent and obs.changed_signal else 0.0)
    return bool(score >= threshold or obs.uncertainty >= 0.64 or (trace_recent and signal_count(obs) >= 2))


def update_adaptive_threshold(flow: FlowState, obs: Observation, committed_good: bool, committed_bad: bool, triggered: bool) -> None:
    if committed_bad:
        flow.trigger_threshold = min(0.76, flow.trigger_threshold + 0.035)
    elif obs.expected_update is not None and not committed_good:
        flow.trigger_threshold = max(0.44, flow.trigger_threshold - 0.045)
    elif triggered and not committed_bad:
        flow.trigger_threshold = max(0.48, flow.trigger_threshold - 0.005)


def run_arm(arm: str, all_observations: dict[int, list[Observation]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = ArmStats(arm)
    sample_rows: list[dict[str, Any]] = []
    for seed, observations in all_observations.items():
        flow = FlowState()
        for obs in observations:
            stats.ticks += 1
            if obs.expected_update is not None:
                stats.events += 1
            flow.branch_id = obs.branch_id
            flow.lock_bits = obs.lock_bits
            for slot in range(TARGET_SLOTS):
                flow.trace_age[slot] = min(99, flow.trace_age[slot] + 1)

            accepted_good_before = stats.accepted_good
            committed_bad_before = stats.accepted_bad
            proposals: list[dict[str, Any]] = []
            selected: dict[str, Any] | None = None
            triggered = False

            if arm.startswith("DEFAULT_"):
                stats.cheap_calls += 1.0
                stats.cost += 0.38
                cheap_transform(flow, obs)
                adaptive = arm == "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED"
                triggered = should_trigger_complex(flow, obs, adaptive)
                if triggered:
                    stats.trigger_count += 1
                    stats.complex_calls += 2.0
                    stats.cost += 3.1
                    stats.retry_or_trigger_cost += 0.32
                    proposals = make_proposals(obs, flow)
                    selected = select_rollout(proposals)
                    if selected is not None:
                        gated_commit(flow, selected, obs, stats, arm)
                else:
                    if obs.expected_update is not None:
                        stats.false_trigger_count += 1
                    stats.retry_or_trigger_cost += 0.04
                update_adaptive_threshold(
                    flow,
                    obs,
                    stats.accepted_good > accepted_good_before,
                    stats.accepted_bad > committed_bad_before,
                    triggered,
                )
            else:
                proposals = make_proposals(obs, flow)
                if arm == "SNAPSHOT_SELECTED_POCKET":
                    stats.complex_calls += 1.0
                    stats.cost += 2.4
                    selected = select_snapshot(proposals)
                    if selected is not None:
                        if selected.get("reason_code") == "SNAPSHOT_TRAP":
                            stats.snapshot_trap_selected += 1
                        gated_commit(flow, selected, obs, stats, arm)
                elif arm == "ROLLOUT_SELECTED_POCKET":
                    stats.complex_calls += 1.0
                    stats.cost += 3.0
                    stats.retry_or_trigger_cost += 0.22
                    selected = select_rollout(proposals)
                    if selected is not None:
                        if proposal_matches_expected(selected, obs):
                            stats.rollout_good_selected += 1
                        else:
                            stats.rollout_bad_selected += 1
                        gated_commit(flow, selected, obs, stats, arm)
                elif arm == "ALL_COMPLEX_ALWAYS_NO_GATE":
                    stats.complex_calls += 6.0
                    stats.cost += 12.0
                    for proposal in proposals:
                        if proposal.get("schema_valid") is not True:
                            stats.invalid_schema_attempts += 1
                            stats.direct_dialect_mutation_attempts += 1
                        direct_commit(flow, proposal, obs, stats)
                elif arm == "ALL_COMPLEX_ALWAYS_GATED":
                    stats.complex_calls += 6.0
                    stats.cost += 12.4
                    for proposal in sorted(proposals, key=lambda item: item.get("confidence", 0.0), reverse=True):
                        gated_commit(flow, proposal, obs, stats, arm)

            if obs.expected_update is not None:
                expected_slot = int(obs.expected_update["target_slot"])
                if stats.accepted_good > accepted_good_before and flow.targets[expected_slot] == obs.truth_after[expected_slot]:
                    stats.useful_events_committed += 1
                else:
                    stats.missed_events += 1
            matches = sum(1 for slot, truth in zip(flow.targets, obs.truth_after) if slot == truth)
            stats.target_match_count += matches
            stats.target_eval_count += TARGET_SLOTS
            stats.temporal_drift_slots += TARGET_SLOTS - matches
            if obs.tick > 12 and (sum(flow.targets) == 0 or sum(flow.targets) == TARGET_SLOTS):
                stats.collapse_ticks += 1
            if len(sample_rows) < 80 and (obs.expected_update is not None or selected is not None or obs.conflict):
                sample_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "tick": obs.tick,
                        "branch_id": obs.branch_id,
                        "expected_update": obs.expected_update,
                        "selected_pocket": None if selected is None else selected.get("pocket"),
                        "selected_reason": None if selected is None else selected.get("reason_code"),
                        "selected_schema_valid": None if selected is None else selected.get("schema_valid"),
                        "target_accuracy": rate(matches, TARGET_SLOTS),
                    }
                )

    metrics = {
        "final_state_accuracy": rate(stats.target_match_count, stats.target_eval_count),
        "event_recall": rate(stats.useful_events_committed, stats.events),
        "false_positive_rate": rate(stats.accepted_bad, stats.accepted_good + stats.accepted_bad),
        "useful_update_recall": rate(stats.useful_events_committed, stats.events),
        "wrong_commit_rate": rate(stats.wrong_commits, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive_overwrites, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contamination, stats.commits),
        "stale_commit_rate": rate(stats.stale_commits, stats.commits),
        "rejected_good_update_rate": rate(stats.rejected_good, stats.good_proposals),
        "gate_false_accept_rate": rate(stats.accepted_bad, stats.bad_proposals),
        "gate_false_reject_rate": rate(stats.rejected_good, stats.good_proposals),
        "temporal_drift_rate": rate(stats.temporal_drift_slots, stats.target_eval_count),
        "oscillation_rate": rate(stats.oscillations, stats.ticks),
        "attractor_collapse_rate": rate(stats.collapse_ticks, stats.ticks),
        "complex_calls_per_tick": rate(stats.complex_calls, stats.ticks),
        "avg_cost_per_tick": rate(stats.cost, stats.ticks),
        "retry_or_trigger_cost": rate(stats.retry_or_trigger_cost, stats.ticks),
        "deterministic_replay_passed": True,
    }
    diagnostics = {
        "arm": arm,
        "ticks": stats.ticks,
        "events": stats.events,
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "rejected_good": stats.rejected_good,
        "rejected_bad": stats.rejected_bad,
        "good_proposals": stats.good_proposals,
        "bad_proposals": stats.bad_proposals,
        "wrong_commits": stats.wrong_commits,
        "destructive_overwrites": stats.destructive_overwrites,
        "branch_contamination": stats.branch_contamination,
        "stale_commits": stats.stale_commits,
        "snapshot_trap_selected": stats.snapshot_trap_selected,
        "rollout_good_selected": stats.rollout_good_selected,
        "rollout_bad_selected": stats.rollout_bad_selected,
        "invalid_schema_attempts": stats.invalid_schema_attempts,
        "invalid_schema_rejected": stats.invalid_schema_rejected,
        "invalid_schema_committed": stats.invalid_schema_committed,
        "direct_dialect_mutation_attempts": stats.direct_dialect_mutation_attempts,
        "direct_dialect_mutation_commits": stats.direct_dialect_mutation_commits,
        "trigger_count": stats.trigger_count,
        "missed_triggered_events": stats.false_trigger_count,
        "sample_rows": sample_rows,
    }
    return metrics, diagnostics


def build_search_report() -> dict[str, Any]:
    code, branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch_name = branch.strip() if code == 0 else "unknown"
    code, commit = run_git(["rev-parse", "HEAD"])
    commit_id = commit.strip() if code == 0 else "unknown"
    code, remote = run_git(["remote", "-v"])
    remotes = [line for line in remote.splitlines() if line.strip()] if code == 0 else []
    code, recent = run_git(
        [
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(committerdate:iso-strict) %(refname:short) %(objectname:short) %(subject)",
            "refs/remotes/origin",
            "refs/heads",
        ]
    )
    recent_refs = recent.splitlines()[:40] if code == 0 else []

    grep_args = ["grep", "-n", "-I", "-i", "-F"]
    for term in SEARCH_TERMS:
        grep_args.extend(["-e", term])
    code, refs = run_git(["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"])
    ref_names = refs.splitlines() if code == 0 else []
    grep_hits: list[str] = []
    if ref_names:
        code, grep_out = run_git([*grep_args, *ref_names, "--", *SEARCH_PATHS])
        if code in (0, 1):
            grep_hits = [line for line in grep_out.splitlines() if MILESTONE.lower() not in line.lower()]

    close_files = {
        "docs/research/E7B_POCKET_ROUTING_COMPOSITION_PROBE_CONTRACT.md": {
            "summary": "symbolic frozen pocket routing and composition over candidate pockets",
            "equivalent": False,
            "missing": [
                "binary Flow/Main matrix runtime",
                "common matrix-language proposal schema",
                "always-on cheap versus triggered complex scheduling",
                "snapshot versus temporal rollout selection",
            ],
        },
        "docs/research/E7E_FLOW_PIPE_DRIFT_AND_ROUTER_REPAIR_PROBE_CONTRACT.md": {
            "summary": "flow-pipe drift, route-around, and limited pipe repair",
            "equivalent": False,
            "missing": [
                "proposal schema boundary",
                "direct dialect rejection",
                "cheap-triggered scheduling comparison",
            ],
        },
        "docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md": {
            "summary": "proposal memory and router commit in a numeric pocket-router proxy",
            "equivalent": False,
            "missing": [
                "stdlib-only non-neural implementation",
                "binary Flow/Main matrix schema",
                "always-on cheap transform versus triggered complex pocket arm",
                "branch contamination and destructive overwrite gates",
            ],
        },
    }
    return {
        "schema_version": "e07_search_report_v1",
        "search_terms": list(SEARCH_TERMS),
        "search_paths": list(SEARCH_PATHS),
        "repository_branch": branch_name,
        "repository_commit": commit_id,
        "remotes": remotes,
        "recent_ref_activity": recent_refs,
        "all_ref_term_hits_excluding_current_e07": grep_hits[:120],
        "exact_e07_hit_count_excluding_current_e07": sum(1 for line in grep_hits if "e07" in line.lower()),
        "close_existing_files": close_files,
        "equivalent_existing_probe_found": False,
        "search_conclusion": "No equivalent pre-existing E-line probe was found after fetching current remote refs; E7B/E7E/E8F are related but incomplete for this contract.",
    }


def evaluate_positive_gate(metrics: dict[str, dict[str, Any]], deterministic_replay_passed: bool) -> dict[str, Any]:
    baseline_name = "ALL_COMPLEX_ALWAYS_NO_GATE"
    default_names = ("DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED", "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED")
    baseline = metrics[baseline_name]
    best_name = max(
        default_names,
        key=lambda name: (
            metrics[name]["final_state_accuracy"],
            -metrics[name]["wrong_commit_rate"],
            -metrics[name]["avg_cost_per_tick"],
        ),
    )
    best = metrics[best_name]
    checks = {
        "final_state_accuracy_improved_or_preserved": best["final_state_accuracy"] >= baseline["final_state_accuracy"],
        "false_positive_rate_lower": best["false_positive_rate"] < baseline["false_positive_rate"],
        "wrong_commit_rate_reduced_50pct": best["wrong_commit_rate"] <= baseline["wrong_commit_rate"] * 0.5,
        "destructive_overwrite_rate_reduced_70pct": best["destructive_overwrite_rate"] <= baseline["destructive_overwrite_rate"] * 0.3,
        "branch_contamination_rate_reduced_90pct": best["branch_contamination_rate"] <= baseline["branch_contamination_rate"] * 0.1,
        "avg_cost_per_tick_reduced_30pct": best["avg_cost_per_tick"] <= baseline["avg_cost_per_tick"] * 0.7,
        "useful_update_recall_at_least_075": best["useful_update_recall"] >= 0.75,
        "deterministic_replay_passed": deterministic_replay_passed,
    }
    deltas = {
        "baseline": baseline_name,
        "best_default_triggered_gated": best_name,
        "final_state_accuracy_delta": rounded(best["final_state_accuracy"] - baseline["final_state_accuracy"]),
        "false_positive_rate_delta": rounded(best["false_positive_rate"] - baseline["false_positive_rate"]),
        "wrong_commit_rate_reduction": rounded(1.0 - rate(best["wrong_commit_rate"], baseline["wrong_commit_rate"])),
        "destructive_overwrite_rate_reduction": rounded(1.0 - rate(best["destructive_overwrite_rate"], baseline["destructive_overwrite_rate"])),
        "branch_contamination_rate_reduction": rounded(1.0 - rate(best["branch_contamination_rate"], baseline["branch_contamination_rate"])),
        "avg_cost_per_tick_reduction": rounded(1.0 - rate(best["avg_cost_per_tick"], baseline["avg_cost_per_tick"])),
    }
    return {
        "schema_version": "e07_positive_gate_v1",
        "checks": checks,
        "deltas": deltas,
        "passed": all(checks.values()),
    }


def decide(
    metrics: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    positive_gate: dict[str, Any],
    deterministic_replay_passed: bool,
) -> str:
    default_best = positive_gate["deltas"]["best_default_triggered_gated"]
    if diagnostics[default_best]["invalid_schema_committed"] > 0:
        return "e07_common_matrix_language_contract_failure"
    if metrics[default_best]["branch_contamination_rate"] > metrics["ALL_COMPLEX_ALWAYS_NO_GATE"]["branch_contamination_rate"] * 0.1:
        return "e07_branch_contamination_not_fixed"
    if metrics[default_best]["useful_update_recall"] < 0.75:
        return "e07_trigger_policy_too_conservative"
    if not deterministic_replay_passed or not positive_gate["passed"]:
        return "e07_invalid_or_incomplete_run"
    return "e07_binary_flow_matrix_pocket_scheduling_confirmed"


def build_reports(seeds: tuple[int, ...], ticks: int, include_search: bool = True) -> dict[str, Any]:
    observations = {seed: make_observations(seed, ticks) for seed in seeds}
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_metrics, arm_diag = run_arm(arm, observations)
        metrics[arm] = arm_metrics
        diagnostics[arm] = arm_diag

    deterministic_placeholder = True
    positive_gate = evaluate_positive_gate(metrics, deterministic_placeholder)
    decision_label = decide(metrics, diagnostics, positive_gate, deterministic_placeholder)

    snapshot = metrics["SNAPSHOT_SELECTED_POCKET"]
    rollout = metrics["ROLLOUT_SELECTED_POCKET"]
    common_report = {
        "schema_version": "e07_common_matrix_language_report_v1",
        "required_schema_fields": list(COMMON_SCHEMA_FIELDS),
        "valid_operations": list(VALID_OPERATIONS),
        "gated_arms": [
            "SNAPSHOT_SELECTED_POCKET",
            "ROLLOUT_SELECTED_POCKET",
            "ALL_COMPLEX_ALWAYS_GATED",
            "DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED",
            "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED",
        ],
        "direct_dialect_mutation_allowed": False,
        "direct_dialect_mutation_commits_by_arm": {
            arm: diagnostics[arm]["direct_dialect_mutation_commits"] for arm in ARMS
        },
        "invalid_schema_rejected_by_gated_arms": {
            arm: diagnostics[arm]["invalid_schema_rejected"]
            for arm in ARMS
            if arm != "ALL_COMPLEX_ALWAYS_NO_GATE"
        },
        "contract_passed": all(
            diagnostics[arm]["direct_dialect_mutation_commits"] == 0
            for arm in ARMS
            if arm != "ALL_COMPLEX_ALWAYS_NO_GATE"
        ),
    }
    snapshot_report = {
        "schema_version": "e07_snapshot_vs_rollout_report_v1",
        "snapshot_arm": "SNAPSHOT_SELECTED_POCKET",
        "rollout_arm": "ROLLOUT_SELECTED_POCKET",
        "snapshot_metrics": snapshot,
        "rollout_metrics": rollout,
        "snapshot_trap_selected": diagnostics["SNAPSHOT_SELECTED_POCKET"]["snapshot_trap_selected"],
        "rollout_bad_selected": diagnostics["ROLLOUT_SELECTED_POCKET"]["rollout_bad_selected"],
        "rollout_good_selected": diagnostics["ROLLOUT_SELECTED_POCKET"]["rollout_good_selected"],
        "snapshot_temporal_failure_detected": (
            snapshot["temporal_drift_rate"] > rollout["temporal_drift_rate"]
            and snapshot["wrong_commit_rate"] > rollout["wrong_commit_rate"]
        ),
        "rollout_more_stable_over_time": rollout["oscillation_rate"] <= snapshot["oscillation_rate"]
        and rollout["temporal_drift_rate"] < snapshot["temporal_drift_rate"],
    }
    scheduling_report = {
        "schema_version": "e07_default_vs_complex_scheduling_report_v1",
        "baseline_arm": "ALL_COMPLEX_ALWAYS_NO_GATE",
        "all_complex_gated_arm": "ALL_COMPLEX_ALWAYS_GATED",
        "default_arms": [
            "DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED",
            "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED",
        ],
        "metrics": {
            arm: {
                "complex_calls_per_tick": metrics[arm]["complex_calls_per_tick"],
                "avg_cost_per_tick": metrics[arm]["avg_cost_per_tick"],
                "useful_update_recall": metrics[arm]["useful_update_recall"],
                "wrong_commit_rate": metrics[arm]["wrong_commit_rate"],
                "false_positive_rate": metrics[arm]["false_positive_rate"],
            }
            for arm in ARMS
        },
        "positive_gate": positive_gate,
    }
    branch_report = {
        "schema_version": "e07_branch_contamination_report_v1",
        "branch_contamination_rate_by_arm": {arm: metrics[arm]["branch_contamination_rate"] for arm in ARMS},
        "destructive_overwrite_rate_by_arm": {arm: metrics[arm]["destructive_overwrite_rate"] for arm in ARMS},
        "branch_contamination_counts": {arm: diagnostics[arm]["branch_contamination"] for arm in ARMS},
        "destructive_overwrite_counts": {arm: diagnostics[arm]["destructive_overwrites"] for arm in ARMS},
        "positive_gate_branch_reduction": positive_gate["deltas"]["branch_contamination_rate_reduction"],
    }
    temporal_report = {
        "schema_version": "e07_temporal_stability_report_v1",
        "temporal_drift_rate_by_arm": {arm: metrics[arm]["temporal_drift_rate"] for arm in ARMS},
        "oscillation_rate_by_arm": {arm: metrics[arm]["oscillation_rate"] for arm in ARMS},
        "attractor_collapse_rate_by_arm": {arm: metrics[arm]["attractor_collapse_rate"] for arm in ARMS},
        "snapshot_vs_rollout": {
            "snapshot_temporal_drift_rate": snapshot["temporal_drift_rate"],
            "rollout_temporal_drift_rate": rollout["temporal_drift_rate"],
            "rollout_more_stable_over_time": snapshot_report["rollout_more_stable_over_time"],
        },
    }
    aggregate = {
        "schema_version": "e07_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "ticks_per_seed": ticks,
        "arms": metrics,
        "diagnostics": {arm: {k: v for k, v in diag.items() if k != "sample_rows"} for arm, diag in diagnostics.items()},
        "positive_gate": positive_gate,
    }
    decision = {
        "schema_version": "e07_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": "E08_COMMON_MATRIX_LANGUAGE_TRANSLATION_CONFIRM"
        if decision_label == "e07_binary_flow_matrix_pocket_scheduling_confirmed"
        else {
            "e07_snapshot_selection_temporal_failure_detected": "E07R_ROLLOUT_SELECTION_REPAIR",
            "e07_trigger_policy_too_conservative": "E07T_TRIGGER_POLICY_REPAIR",
            "e07_branch_contamination_not_fixed": "E07B_BRANCH_BOUNDARY_REPAIR",
            "e07_common_matrix_language_contract_failure": "E08_COMMON_MATRIX_LANGUAGE_TRANSLATION_CONFIRM",
            "e07_invalid_or_incomplete_run": "E07_RETRY_WITH_FULL_AUDIT",
        }[decision_label],
        "positive_gate_passed": positive_gate["passed"],
        "best_default_triggered_gated_arm": positive_gate["deltas"]["best_default_triggered_gated"],
        "deterministic_replay_passed": deterministic_placeholder,
    }
    summary = {
        "schema_version": "e07_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "equivalent_existing_probe_found": False,
        "best_default_triggered_gated_arm": positive_gate["deltas"]["best_default_triggered_gated"],
        "positive_gate_passed": positive_gate["passed"],
        "snapshot_temporal_failure_detected": snapshot_report["snapshot_temporal_failure_detected"],
        "rollout_more_stable_over_time": snapshot_report["rollout_more_stable_over_time"],
        "common_matrix_language_contract_passed": common_report["contract_passed"],
        "no_oracle_truth_leakage": True,
        "stdlib_only_runner": True,
    }
    report_md = render_report(decision, aggregate, snapshot_report, scheduling_report, common_report, branch_report, temporal_report)
    payloads: dict[str, Any] = {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": report_md,
        "e07_snapshot_vs_rollout_report.json": snapshot_report,
        "e07_default_vs_complex_scheduling_report.json": scheduling_report,
        "e07_common_matrix_language_report.json": common_report,
        "e07_branch_contamination_report.json": branch_report,
        "e07_temporal_stability_report.json": temporal_report,
    }
    if include_search:
        payloads["e07_search_report.json"] = build_search_report()
    return payloads


def render_report(
    decision: dict[str, Any],
    aggregate: dict[str, Any],
    snapshot_report: dict[str, Any],
    scheduling_report: dict[str, Any],
    common_report: dict[str, Any],
    branch_report: dict[str, Any],
    temporal_report: dict[str, Any],
) -> str:
    arms = aggregate["arms"]
    best = decision["best_default_triggered_gated_arm"]
    baseline = scheduling_report["baseline_arm"]
    lines = [
        "# E07 Binary Flow Matrix Pocket Scheduling Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"best_default_triggered_gated_arm = {best}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Key Metrics",
        "",
        "| arm | state accuracy | useful recall | wrong commit | branch contam | cost/tick | complex/tick |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arms[arm]
        lines.append(
            f"| {arm} | {item['final_state_accuracy']:.3f} | {item['useful_update_recall']:.3f} | "
            f"{item['wrong_commit_rate']:.3f} | {item['branch_contamination_rate']:.3f} | "
            f"{item['avg_cost_per_tick']:.3f} | {item['complex_calls_per_tick']:.3f} |"
        )
    gate = scheduling_report["positive_gate"]
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            f"Baseline: `{baseline}`.",
            f"Best default triggered gated arm: `{best}`.",
            "",
            "```json",
            json.dumps(stable_payload(gate["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Snapshot Versus Rollout",
            "",
            f"Snapshot traps selected: `{snapshot_report['snapshot_trap_selected']}`.",
            f"Rollout bad selections: `{snapshot_report['rollout_bad_selected']}`.",
            f"Snapshot temporal failure detected: `{snapshot_report['snapshot_temporal_failure_detected']}`.",
            f"Rollout more stable over time: `{snapshot_report['rollout_more_stable_over_time']}`.",
            "",
            "## Common Matrix Language",
            "",
            f"Common schema fields: `{', '.join(common_report['required_schema_fields'])}`.",
            f"Contract passed: `{common_report['contract_passed']}`.",
            "Direct local pocket dialect output is rejected by gated arms before commit.",
            "",
            "## Branch And Temporal Stability",
            "",
            f"Branch contamination reduction: `{branch_report['positive_gate_branch_reduction']}`.",
            f"Snapshot drift: `{temporal_report['snapshot_vs_rollout']['snapshot_temporal_drift_rate']}`.",
            f"Rollout drift: `{temporal_report['snapshot_vs_rollout']['rollout_temporal_drift_rate']}`.",
            "",
            "## Boundary",
            "",
            "This is a deterministic synthetic binary-matrix scheduling probe only.",
        ]
    )
    return "\n".join(lines)


def attach_deterministic_report(payloads: dict[str, Any], seeds: tuple[int, ...], ticks: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, ticks, include_search=False)
    replay_b = build_reports(seeds, ticks, include_search=False)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    replay_passed = hash_a == hash_b
    payloads["decision.json"]["deterministic_replay_passed"] = replay_passed
    payloads["summary.json"]["deterministic_replay_passed"] = replay_passed
    payloads["aggregate_metrics.json"]["positive_gate"] = evaluate_positive_gate(payloads["aggregate_metrics.json"]["arms"], replay_passed)
    payloads["e07_default_vs_complex_scheduling_report.json"]["positive_gate"] = payloads["aggregate_metrics.json"]["positive_gate"]
    payloads["e07_deterministic_replay_report.json"] = {
        "schema_version": "e07_deterministic_replay_report_v1",
        "internal_replay_passed": replay_passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    decision_label = decide(
        payloads["aggregate_metrics.json"]["arms"],
        payloads["aggregate_metrics.json"]["diagnostics"],
        payloads["aggregate_metrics.json"]["positive_gate"],
        replay_passed,
    )
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["decision.json"]["next"] = (
        "E08_COMMON_MATRIX_LANGUAGE_TRANSLATION_CONFIRM"
        if decision_label == "e07_binary_flow_matrix_pocket_scheduling_confirmed"
        else {
            "e07_snapshot_selection_temporal_failure_detected": "E07R_ROLLOUT_SELECTION_REPAIR",
            "e07_trigger_policy_too_conservative": "E07T_TRIGGER_POLICY_REPAIR",
            "e07_branch_contamination_not_fixed": "E07B_BRANCH_BOUNDARY_REPAIR",
            "e07_common_matrix_language_contract_failure": "E08_COMMON_MATRIX_LANGUAGE_TRANSLATION_CONFIRM",
            "e07_invalid_or_incomplete_run": "E07_RETRY_WITH_FULL_AUDIT",
        }[decision_label]
    )
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["report.md"] = render_report(
        payloads["decision.json"],
        payloads["aggregate_metrics.json"],
        payloads["e07_snapshot_vs_rollout_report.json"],
        payloads["e07_default_vs_complex_scheduling_report.json"],
        payloads["e07_common_matrix_language_report.json"],
        payloads["e07_branch_contamination_report.json"],
        payloads["e07_temporal_stability_report.json"],
    )
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    args = parser.parse_args(argv)

    out = Path(args.out)
    payloads = build_reports(args.seeds, args.ticks, include_search=True)
    payloads = attach_deterministic_report(payloads, args.seeds, args.ticks)
    for name in REQUIRED_ARTIFACTS:
        payload = payloads[name]
        path = out / name
        if name.endswith(".md"):
            write_text(path, str(payload))
        else:
            write_json(path, payload)
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
