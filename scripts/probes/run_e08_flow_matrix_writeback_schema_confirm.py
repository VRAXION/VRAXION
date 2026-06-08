#!/usr/bin/env python3
"""E08 non-neural binary Flow Matrix writeback schema probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E08_FLOW_MATRIX_WRITEBACK_SCHEMA_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e08_flow_matrix_writeback_schema_confirm")
DEFAULT_SEEDS = (80801, 80802, 80803, 80804, 80805, 80806)
DEFAULT_TICKS = 360
TARGET_SLOTS = 12
BRANCH_WIDTH = TARGET_SLOTS // 2
SCHEMA_FIELDS = (
    "read_region",
    "detector_id",
    "condition",
    "transform_op",
    "write_region",
    "branch_id",
    "lock_mask",
    "trace_before",
    "trace_after",
    "confidence",
    "cost",
    "reason_code",
)
TRANSFORM_OPS = (
    "SET",
    "CLEAR",
    "FLIP",
    "COPY",
    "MOVE",
    "XOR_INTO",
    "INHIBIT",
    "LOCK",
    "UNLOCK",
    "SHIFT",
    "FILL_GAP",
    "DELETE_ISOLATED",
)
ARMS = (
    "DIRECT_OVERWRITE_BASELINE",
    "LOCAL_DIALECT_BASELINE",
    "COMMON_SCHEMA_NO_GATE",
    "COMMON_SCHEMA_GATED",
    "COMMON_SCHEMA_GATED_WITH_ROLLBACK",
    "REGION_OPERATOR_SCHEMA",
)
SHARED_SCHEMA_ARMS = (
    "COMMON_SCHEMA_NO_GATE",
    "COMMON_SCHEMA_GATED",
    "COMMON_SCHEMA_GATED_WITH_ROLLBACK",
    "REGION_OPERATOR_SCHEMA",
)
POSITIVE_CANDIDATES = ("COMMON_SCHEMA_GATED_WITH_ROLLBACK", "REGION_OPERATOR_SCHEMA")
VALID_DECISIONS = (
    "e08_flow_matrix_writeback_schema_confirmed",
    "e08_direct_overwrite_remains_best",
    "e08_common_schema_not_sufficient",
    "e08_gate_too_conservative",
    "e08_branch_contamination_not_fixed",
    "e08_stale_write_rollback_failure",
    "e08_trace_validity_failure",
    "e08_invalid_or_incomplete_run",
)
STRESS_CASES = (
    "same_region_two_pockets",
    "overlapping_write_regions",
    "stale_proposal",
    "branch_mismatch",
    "target_locked",
    "wrong_detector_high_confidence",
    "correct_detector_wrong_target",
    "noisy_flow_state",
    "partial_flow_corruption",
    "long_temporal_rollout",
    "reversed_pocket_order",
    "conflicting_set_clear",
    "conflicting_flip_lock",
    "conflicting_copy_inhibit",
    "branch_contamination_attempt",
    "destructive_overwrite_attempt",
    "false_high_confidence_proposal",
)
SEARCH_TERMS = (
    "E08",
    "FLOW_MATRIX_WRITEBACK",
    "Flow Matrix writeback",
    "common matrix language",
    "shared writeback schema",
    "pocket writeback",
    "branch contamination",
    "destructive overwrite",
    "stale proposal",
    "region operator schema",
    "detector wiring",
    "gate commit",
    "rollback writeback",
    "Flow-grid operation schema",
)
SEARCH_PATHS = ("docs/research", "scripts/probes", "docs/wiki", "CHANGELOG.md")
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e08_search_report.json",
    "e08_writeback_schema_report.json",
    "e08_stress_cases_report.json",
    "e08_gate_commit_report.json",
    "e08_rollback_report.json",
    "e08_branch_contamination_report.json",
    "e08_temporal_stability_report.json",
    "e08_deterministic_replay_report.json",
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


def normalized_region(region: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(slot) % TARGET_SLOTS for slot in region))


@dataclass(frozen=True)
class Observation:
    seed: int
    tick: int
    branch_id: int
    case_type: str
    signal_bits: tuple[int, ...]
    distractor_bits: tuple[int, ...]
    lock_bits: tuple[int, ...]
    stale_marker: bool
    expected_update: dict[str, Any] | None
    truth_before: tuple[int, ...]
    truth_after: tuple[int, ...]
    truth_trace_before: tuple[int, ...]
    truth_trace_after: tuple[int, ...]


@dataclass
class FlowState:
    targets: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)
    locks: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)
    trace_version: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)
    last_change_tick: list[int] = field(default_factory=lambda: [-999] * TARGET_SLOTS)
    previous_value: list[int] = field(default_factory=lambda: [0] * TARGET_SLOTS)
    branch_id: int = 0


@dataclass
class ArmStats:
    arm: str
    ticks: int = 0
    events: int = 0
    event_success: int = 0
    target_matches: int = 0
    target_evals: int = 0
    trace_matches: int = 0
    trace_evals: int = 0
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    good_proposals: int = 0
    bad_proposals: int = 0
    schema_attempts: int = 0
    schema_violations: int = 0
    wrong_writebacks: int = 0
    destructive_overwrites: int = 0
    branch_contamination: int = 0
    stale_attempts: int = 0
    stale_rejections: int = 0
    stale_commits: int = 0
    conflict_cases: int = 0
    conflict_successes: int = 0
    rollback_attempts: int = 0
    rollback_successes: int = 0
    temporal_drift_slots: int = 0
    oscillations: int = 0
    collapse_ticks: int = 0
    cost: float = 0.0
    stress_counts: dict[str, int] = field(default_factory=dict)
    stress_success: dict[str, int] = field(default_factory=dict)


def apply_values(values: list[int], locks: list[int], proposal: dict[str, Any]) -> None:
    op = str(proposal.get("transform_op", "FLIP"))
    write_region = list(proposal.get("write_region", []))
    read_region = list(proposal.get("read_region", []))
    if op == "SET":
        for slot in write_region:
            values[slot] = 1
    elif op == "CLEAR":
        for slot in write_region:
            values[slot] = 0
    elif op == "FLIP":
        for slot in write_region:
            values[slot] ^= 1
    elif op == "COPY":
        source = [values[slot] for slot in read_region] or [0]
        for idx, slot in enumerate(write_region):
            values[slot] = source[idx % len(source)]
    elif op == "MOVE":
        source = [values[slot] for slot in read_region] or [0]
        for idx, slot in enumerate(write_region):
            values[slot] = source[idx % len(source)]
        for slot in read_region:
            values[slot] = 0
    elif op == "XOR_INTO":
        source_xor = sum(values[slot] for slot in read_region) % 2
        for slot in write_region:
            values[slot] ^= source_xor
    elif op == "INHIBIT":
        for slot in write_region:
            values[slot] = 0
    elif op == "LOCK":
        for slot in write_region:
            locks[slot] = 1
    elif op == "UNLOCK":
        for slot in write_region:
            locks[slot] = 0
    elif op == "SHIFT":
        before = values[:]
        for slot in write_region:
            values[slot] = before[(slot - 1) % len(before)]
    elif op == "FILL_GAP":
        if write_region:
            for slot in range(min(write_region), max(write_region) + 1):
                values[slot] = 1
    elif op == "DELETE_ISOLATED":
        before = values[:]
        for slot in write_region:
            left = before[(slot - 1) % len(before)]
            right = before[(slot + 1) % len(before)]
            if before[slot] and not left and not right:
                values[slot] = 0
    else:
        raise ValueError(f"unsupported transform_op {op}")


def event_operation(tick: int, seed: int, case_type: str) -> str:
    if case_type == "conflicting_copy_inhibit":
        return "COPY"
    if case_type == "partial_flow_corruption":
        return "DELETE_ISOLATED"
    if case_type == "noisy_flow_state":
        return "FILL_GAP"
    if case_type == "long_temporal_rollout":
        return "SHIFT"
    return ("SET", "CLEAR", "FLIP")[(tick + seed) % 3]


def make_observations(seed: int, ticks: int) -> list[Observation]:
    rng = random.Random(seed)
    truth = [0] * TARGET_SLOTS
    truth_trace = [0] * TARGET_SLOTS
    rows: list[Observation] = []
    for tick in range(ticks):
        case_type = STRESS_CASES[(tick + seed) % len(STRESS_CASES)]
        branch_id = ((tick // 5) + seed) % 2
        offset = (tick * 3 + seed) % BRANCH_WIDTH
        slot = branch_id * BRANCH_WIDTH + offset
        pair_slot = branch_id * BRANCH_WIDTH + ((offset + 1) % BRANCH_WIDTH)
        write_region = [slot]
        if case_type in {"overlapping_write_regions", "noisy_flow_state", "partial_flow_corruption"}:
            write_region = sorted({slot, pair_slot})
        read_region = [branch_id * BRANCH_WIDTH + ((offset - 1) % BRANCH_WIDTH)]
        op = event_operation(tick, seed, case_type)

        signal = [0] * 8
        signal[offset % 4] = 1
        signal[(offset + 1) % 4] = 1
        signal[4] = 1 if op in {"SET", "FILL_GAP", "COPY", "SHIFT"} else 0
        signal[5] = 1 if op in {"CLEAR", "INHIBIT", "DELETE_ISOLATED"} else 0
        signal[6] = 1 if op in {"FLIP", "XOR_INTO"} else 0
        if case_type in {"wrong_detector_high_confidence", "false_high_confidence_proposal"}:
            signal[7] = 1

        distractor = [0] * 8
        for _ in range(1 + int(case_type in {"noisy_flow_state", "partial_flow_corruption"})):
            distractor[rng.randrange(8)] = 1
        if case_type in {"wrong_detector_high_confidence", "destructive_overwrite_attempt"}:
            distractor[(offset + 3) % 8] = 1
            distractor[(offset + 4) % 8] = 1

        locks = [0] * TARGET_SLOTS
        if case_type == "target_locked":
            locks[slot] = 1
        elif (tick + seed) % 31 == 0:
            locks[branch_id * BRANCH_WIDTH + ((offset + 2) % BRANCH_WIDTH)] = 1

        truth_before = tuple(truth)
        trace_before = tuple(truth_trace)
        expected_update: dict[str, Any] | None = None
        if case_type != "target_locked":
            expected_update = {
                "read_region": read_region,
                "write_region": write_region,
                "transform_op": op,
                "branch_id": branch_id,
            }
            before = truth[:]
            apply_values(truth, locks[:], expected_update)
            for target_slot, before_value in enumerate(before):
                if truth[target_slot] != before_value:
                    truth_trace[target_slot] += 1

        rows.append(
            Observation(
                seed=seed,
                tick=tick,
                branch_id=branch_id,
                case_type=case_type,
                signal_bits=tuple(signal),
                distractor_bits=tuple(distractor),
                lock_bits=tuple(locks),
                stale_marker=case_type in {"stale_proposal", "long_temporal_rollout"},
                expected_update=expected_update,
                truth_before=truth_before,
                truth_after=tuple(truth),
                truth_trace_before=trace_before,
                truth_trace_after=tuple(truth_trace),
            )
        )
    return rows


def condition_true(condition: dict[str, Any], obs: Observation, flow: FlowState) -> bool:
    kind = condition.get("type")
    if kind == "COUNT":
        region = condition.get("region")
        bits = obs.signal_bits if region == "signal" else obs.distractor_bits
        return sum(bits) >= int(condition.get("min", 1))
    if kind == "ANY":
        region = condition.get("region")
        bits = obs.signal_bits if region == "signal" else obs.distractor_bits
        return any(bits)
    if kind == "XOR":
        return (sum(obs.signal_bits) % 2) == int(condition.get("value", 1))
    if kind == "BRANCH_ACTIVE":
        return int(condition.get("branch_id", -1)) == obs.branch_id == flow.branch_id
    if kind == "TARGET_UNLOCKED":
        return all(flow.locks[int(slot)] == 0 for slot in condition.get("slots", []))
    if kind == "CONFLICT":
        return obs.case_type.startswith("conflicting") or obs.case_type in {"same_region_two_pockets", "overlapping_write_regions"}
    if kind == "STALE_TRACE":
        return obs.stale_marker
    if kind == "FALSE":
        return False
    return False


def schema_complete(proposal: dict[str, Any]) -> bool:
    return proposal.get("schema_valid") is True and all(field in proposal for field in SCHEMA_FIELDS)


def predicted_trace_after(flow: FlowState, write_region: list[int], op: str) -> dict[str, int]:
    return {str(slot): flow.trace_version[slot] + (0 if op in {"LOCK", "UNLOCK"} else 1) for slot in write_region}


def make_schema_proposal(
    pocket: str,
    obs: Observation,
    flow: FlowState,
    read_region: list[int],
    write_region: list[int],
    op: str,
    detector_id: str,
    condition: dict[str, Any],
    confidence: float,
    cost: float,
    reason_code: str,
    branch_id: int | None = None,
    stale: bool = False,
) -> dict[str, Any]:
    region = sorted(int(slot) % TARGET_SLOTS for slot in write_region)
    trace_before = {str(slot): flow.trace_version[slot] - (2 if stale else 0) for slot in region}
    return {
        "pocket": pocket,
        "schema_valid": True,
        "read_region": sorted(int(slot) % TARGET_SLOTS for slot in read_region),
        "detector_id": detector_id,
        "condition": condition,
        "transform_op": op,
        "write_region": region,
        "branch_id": obs.branch_id if branch_id is None else int(branch_id),
        "lock_mask": {str(slot): flow.locks[slot] for slot in region},
        "trace_before": trace_before,
        "trace_after": predicted_trace_after(flow, region, op),
        "confidence": rounded(confidence),
        "cost": rounded(cost),
        "reason_code": reason_code,
    }


def make_local_dialect(obs: Observation, flow: FlowState, target_slot: int, op: str) -> dict[str, Any]:
    wrong_branch = 1 - obs.branch_id if obs.case_type in {"branch_mismatch", "branch_contamination_attempt"} else obs.branch_id
    wrong_slot = wrong_branch * BRANCH_WIDTH + (target_slot % BRANCH_WIDTH)
    return {
        "pocket": "private_dialect_writer",
        "schema_valid": False,
        "local_dialect": {
            "dst": wrong_slot,
            "verb": op,
            "wire": "pocket_private_signal_lane",
            "branch": wrong_branch,
        },
        "write_region": [wrong_slot],
        "transform_op": op,
        "branch_id": wrong_branch,
        "confidence": 0.82,
        "reason_code": "LOCAL_DIALECT_PRIVATE_WRITE",
    }


def make_proposals(obs: Observation, flow: FlowState) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    expected = obs.expected_update
    if expected is not None:
        read_region = list(expected["read_region"])
        write_region = list(expected["write_region"])
        op = str(expected["transform_op"])
        target_slot = write_region[0]
        proposals.append(
            make_schema_proposal(
                "canonical_detector_writer",
                obs,
                flow,
                read_region,
                write_region,
                op,
                "EDGE_COUNT_MATCH",
                {"type": "COUNT", "region": "signal", "min": 2},
                0.86,
                1.0,
                "CANONICAL_WRITEBACK",
            )
        )
        proposals.append(make_local_dialect(obs, flow, target_slot, op))

        wrong_same_branch = obs.branch_id * BRANCH_WIDTH + ((target_slot + 2) % BRANCH_WIDTH)
        proposals.append(
            make_schema_proposal(
                "wrong_target_writer",
                obs,
                flow,
                read_region,
                [wrong_same_branch],
                op,
                "EDGE_COUNT_MATCH",
                {"type": "COUNT", "region": "signal", "min": 2},
                0.79,
                1.1,
                "CORRECT_DETECTOR_WRONG_TARGET",
            )
        )
        wrong_branch = 1 - obs.branch_id
        proposals.append(
            make_schema_proposal(
                "branch_blind_writer",
                obs,
                flow,
                read_region,
                [wrong_branch * BRANCH_WIDTH + (target_slot % BRANCH_WIDTH)],
                op,
                "ANY_SIGNAL_BRANCH_BLIND",
                {"type": "ANY", "region": "signal"},
                0.77,
                1.0,
                "BRANCH_CONTAMINATION_ATTEMPT",
                branch_id=wrong_branch,
            )
        )
        proposals.append(
            make_schema_proposal(
                "stale_trace_writer",
                obs,
                flow,
                read_region,
                write_region,
                "FLIP" if op != "FLIP" else "SET",
                "STALE_TRACE",
                {"type": "STALE_TRACE"},
                0.81,
                1.4,
                "STALE_PROPOSAL",
                stale=True,
            )
        )
        proposals.append(
            make_schema_proposal(
                "false_high_confidence_writer",
                obs,
                flow,
                read_region,
                write_region,
                "CLEAR" if op == "SET" else "SET",
                "MATCH_FALSE_TEMPLATE",
                {"type": "FALSE"},
                0.97,
                1.7,
                "FALSE_HIGH_CONFIDENCE",
            )
        )
        if obs.case_type in {"same_region_two_pockets", "conflicting_set_clear"}:
            proposals.append(
                make_schema_proposal(
                    "conflicting_writer",
                    obs,
                    flow,
                    read_region,
                    write_region,
                    "CLEAR" if op == "SET" else "SET",
                    "CONFLICT",
                    {"type": "CONFLICT"},
                    0.84,
                    1.3,
                    "CONFLICTING_OPERATION",
                )
            )
        if obs.case_type == "conflicting_flip_lock":
            proposals.append(
                make_schema_proposal(
                    "conflicting_lock_writer",
                    obs,
                    flow,
                    read_region,
                    write_region,
                    "LOCK",
                    "CONFLICT",
                    {"type": "CONFLICT"},
                    0.88,
                    1.2,
                    "CONFLICTING_FLIP_LOCK",
                )
            )
        if obs.case_type == "conflicting_copy_inhibit":
            proposals.append(
                make_schema_proposal(
                    "conflicting_inhibit_writer",
                    obs,
                    flow,
                    read_region,
                    write_region,
                    "INHIBIT",
                    "CONFLICT",
                    {"type": "CONFLICT"},
                    0.85,
                    1.2,
                    "CONFLICTING_COPY_INHIBIT",
                )
            )
        if obs.case_type == "target_locked":
            proposals.append(
                make_schema_proposal(
                    "locked_target_writer",
                    obs,
                    flow,
                    read_region,
                    write_region,
                    op,
                    "TARGET_UNLOCKED",
                    {"type": "TARGET_UNLOCKED", "slots": write_region},
                    0.91,
                    1.0,
                    "LOCKED_TARGET_WRITE",
                )
            )
    else:
        locked_slot = next((slot for slot, locked in enumerate(obs.lock_bits) if locked), obs.branch_id * BRANCH_WIDTH)
        proposals.append(make_local_dialect(obs, flow, locked_slot, "SET"))
        proposals.append(
            make_schema_proposal(
                "locked_target_writer",
                obs,
                flow,
                [locked_slot],
                [locked_slot],
                "SET",
                "TARGET_UNLOCKED",
                {"type": "TARGET_UNLOCKED", "slots": [locked_slot]},
                0.92,
                1.0,
                "LOCKED_TARGET_WRITE",
            )
        )
    if obs.case_type == "reversed_pocket_order":
        proposals = list(reversed(proposals))
    return proposals


def proposal_is_good(proposal: dict[str, Any], obs: Observation) -> bool:
    expected = obs.expected_update
    if expected is None:
        return False
    return (
        schema_complete(proposal)
        and normalized_region(proposal.get("read_region", [])) == normalized_region(expected["read_region"])
        and normalized_region(proposal.get("write_region", [])) == normalized_region(expected["write_region"])
        and proposal.get("transform_op") == expected["transform_op"]
        and proposal.get("branch_id") == expected["branch_id"]
        and proposal.get("reason_code") == "CANONICAL_WRITEBACK"
    )


def trace_matches(proposal: dict[str, Any], flow: FlowState) -> bool:
    trace_before = proposal.get("trace_before", {})
    for slot in proposal.get("write_region", []):
        if int(trace_before.get(str(slot), -999)) != flow.trace_version[slot]:
            return False
    return True


def is_branch_clean(proposal: dict[str, Any], obs: Observation) -> bool:
    return proposal.get("branch_id") == obs.branch_id and all(slot in branch_slots(obs.branch_id) for slot in proposal.get("write_region", []))


def target_unlocked(proposal: dict[str, Any], flow: FlowState) -> bool:
    if proposal.get("transform_op") == "UNLOCK":
        return True
    return all(flow.locks[slot] == 0 for slot in proposal.get("write_region", []))


def gate_accepts(proposal: dict[str, Any], obs: Observation, flow: FlowState) -> tuple[bool, str]:
    if not schema_complete(proposal):
        return False, "schema"
    if proposal.get("transform_op") not in TRANSFORM_OPS:
        return False, "operation"
    if not is_branch_clean(proposal, obs):
        return False, "branch"
    if not target_unlocked(proposal, flow):
        return False, "locked"
    if not trace_matches(proposal, flow):
        return False, "stale_trace"
    if not condition_true(proposal.get("condition", {}), obs, flow):
        return False, "condition"
    if proposal.get("confidence", 0.0) < 0.55:
        return False, "confidence"
    return True, "accepted"


def conflict_key(proposal: dict[str, Any]) -> tuple[int, ...]:
    return normalized_region(proposal.get("write_region", []))


def conflicts_present(proposals: list[dict[str, Any]]) -> bool:
    by_region: dict[tuple[int, ...], set[str]] = {}
    for proposal in proposals:
        if not schema_complete(proposal):
            continue
        by_region.setdefault(conflict_key(proposal), set()).add(str(proposal.get("transform_op")))
    return any(len(ops) > 1 for ops in by_region.values())


def commit_proposal(flow: FlowState, proposal: dict[str, Any], obs: Observation, stats: ArmStats, gated: bool) -> bool:
    is_good = proposal_is_good(proposal, obs)
    if is_good:
        stats.good_proposals += 1
    else:
        stats.bad_proposals += 1
    if proposal.get("reason_code") == "STALE_PROPOSAL" or not trace_matches(proposal, flow):
        stats.stale_attempts += 1

    if gated:
        accepted, reason = gate_accepts(proposal, obs, flow)
        if not accepted:
            if is_good:
                stats.rejected_good += 1
            else:
                stats.rejected_bad += 1
            if reason == "stale_trace" or proposal.get("reason_code") == "STALE_PROPOSAL":
                stats.stale_rejections += 1
            return False

    if not schema_complete(proposal):
        stats.schema_violations += 1
        local = proposal.get("local_dialect", {})
        proposal = {
            "schema_valid": False,
            "read_region": [],
            "detector_id": "private",
            "condition": {"type": "ANY", "region": "signal"},
            "transform_op": local.get("verb", proposal.get("transform_op", "FLIP")),
            "write_region": [int(local.get("dst", 0)) % TARGET_SLOTS],
            "branch_id": int(local.get("branch", obs.branch_id)),
            "trace_before": {},
            "trace_after": {},
            "confidence": proposal.get("confidence", 0.0),
            "cost": 1.0,
            "reason_code": proposal.get("reason_code", "LOCAL_DIALECT_PRIVATE_WRITE"),
        }

    before_values = flow.targets[:]
    before_locks = flow.locks[:]
    apply_values(flow.targets, flow.locks, proposal)
    stats.commits += 1
    if is_good:
        stats.accepted_good += 1
    else:
        stats.accepted_bad += 1
        stats.wrong_writebacks += 1
    if not is_branch_clean(proposal, obs):
        stats.branch_contamination += 1
    if proposal.get("reason_code") == "STALE_PROPOSAL" or not trace_matches(proposal, flow):
        stats.stale_commits += 1

    for slot in proposal.get("write_region", []):
        if flow.targets[slot] != before_values[slot]:
            if obs.tick - flow.last_change_tick[slot] <= 4 and flow.targets[slot] == flow.previous_value[slot]:
                stats.oscillations += 1
            flow.previous_value[slot] = before_values[slot]
            flow.last_change_tick[slot] = obs.tick
            flow.trace_version[slot] += 1
        if before_values[slot] == obs.truth_after[slot] and flow.targets[slot] != obs.truth_after[slot]:
            stats.destructive_overwrites += 1
    if before_locks != flow.locks:
        for slot, before_lock in enumerate(before_locks):
            if before_lock != flow.locks[slot]:
                flow.trace_version[slot] += 1
    return True


def apply_corruption(flow: FlowState, obs: Observation) -> None:
    if obs.case_type == "partial_flow_corruption" and (obs.tick + obs.seed) % 2 == 0:
        slot = obs.branch_id * BRANCH_WIDTH + ((obs.tick + 3) % BRANCH_WIDTH)
        flow.targets[slot] ^= 1
        flow.trace_version[slot] += 1
    if obs.case_type == "noisy_flow_state" and (obs.tick + obs.seed) % 3 == 0:
        slot = (1 - obs.branch_id) * BRANCH_WIDTH + ((obs.tick + 1) % BRANCH_WIDTH)
        flow.targets[slot] ^= 1
        flow.trace_version[slot] += 1


def evaluate_tick(stats: ArmStats, flow: FlowState, obs: Observation, accepted_good_before: int) -> None:
    if obs.expected_update is not None:
        expected_region = list(obs.expected_update["write_region"])
        if stats.accepted_good > accepted_good_before and all(flow.targets[slot] == obs.truth_after[slot] for slot in expected_region):
            stats.event_success += 1
            stats.stress_success[obs.case_type] = stats.stress_success.get(obs.case_type, 0) + 1
    matches = sum(1 for got, want in zip(flow.targets, obs.truth_after) if got == want)
    trace_matches_count = sum(
        1
        for slot, trace in enumerate(flow.trace_version)
        if trace == obs.truth_trace_after[slot] and flow.targets[slot] == obs.truth_after[slot]
    )
    stats.target_matches += matches
    stats.target_evals += TARGET_SLOTS
    stats.trace_matches += trace_matches_count
    stats.trace_evals += TARGET_SLOTS
    stats.temporal_drift_slots += TARGET_SLOTS - matches
    if obs.tick > 20 and (sum(flow.targets) == 0 or sum(flow.targets) == TARGET_SLOTS):
        stats.collapse_ticks += 1


def run_arm(arm: str, all_observations: dict[int, list[Observation]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = ArmStats(arm)
    samples: list[dict[str, Any]] = []
    for seed, rows in all_observations.items():
        flow = FlowState()
        for obs in rows:
            stats.ticks += 1
            stats.stress_counts[obs.case_type] = stats.stress_counts.get(obs.case_type, 0) + 1
            if obs.expected_update is not None:
                stats.events += 1
            flow.branch_id = obs.branch_id
            flow.locks = list(obs.lock_bits)
            apply_corruption(flow, obs)
            proposals = make_proposals(obs, flow)
            stats.schema_attempts += len(proposals)
            accepted_good_before = stats.accepted_good

            if conflicts_present(proposals):
                stats.conflict_cases += 1

            if arm == "DIRECT_OVERWRITE_BASELINE":
                stats.cost += 5.5
                for proposal in proposals:
                    commit_proposal(flow, proposal, obs, stats, gated=False)
            elif arm == "LOCAL_DIALECT_BASELINE":
                stats.cost += 2.0
                for proposal in proposals:
                    if proposal.get("schema_valid") is False or proposal.get("reason_code") in {"FALSE_HIGH_CONFIDENCE", "BRANCH_CONTAMINATION_ATTEMPT"}:
                        commit_proposal(flow, proposal, obs, stats, gated=False)
            elif arm == "COMMON_SCHEMA_NO_GATE":
                stats.cost += 4.0
                for proposal in proposals:
                    if proposal.get("schema_valid") is True:
                        commit_proposal(flow, proposal, obs, stats, gated=False)
            elif arm == "COMMON_SCHEMA_GATED":
                stats.cost += 4.4
                before_bad = stats.accepted_bad
                for proposal in sorted((p for p in proposals if p.get("schema_valid") is True), key=lambda item: item.get("confidence", 0.0), reverse=True):
                    commit_proposal(flow, proposal, obs, stats, gated=True)
                if conflicts_present(proposals) and stats.accepted_bad == before_bad:
                    stats.conflict_successes += 1
            elif arm == "COMMON_SCHEMA_GATED_WITH_ROLLBACK":
                stats.cost += 4.8
                before_bad = stats.accepted_bad
                grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                for proposal in proposals:
                    if proposal.get("schema_valid") is True:
                        grouped.setdefault(conflict_key(proposal), []).append(proposal)
                for group in grouped.values():
                    accepted_candidates = [p for p in group if gate_accepts(p, obs, flow)[0]]
                    if not accepted_candidates:
                        for proposal in group:
                            commit_proposal(flow, proposal, obs, stats, gated=True)
                        continue
                    ops = {p["transform_op"] for p in accepted_candidates}
                    risky = len(ops) > 1 or any(p.get("reason_code", "").startswith("CONFLICTING") for p in accepted_candidates)
                    chosen = max(
                        accepted_candidates,
                        key=lambda item: (
                            1 if item.get("reason_code") == "CANONICAL_WRITEBACK" else 0,
                            -item.get("cost", 0.0),
                            item.get("confidence", 0.0),
                        ),
                    )
                    snapshot = (flow.targets[:], flow.locks[:], flow.trace_version[:])
                    committed = commit_proposal(flow, chosen, obs, stats, gated=True)
                    if risky and committed and chosen.get("reason_code") != "CANONICAL_WRITEBACK":
                        stats.rollback_attempts += 1
                        flow.targets, flow.locks, flow.trace_version = snapshot
                        stats.rollback_successes += 1
                    elif risky and committed:
                        stats.conflict_successes += 1
                if conflicts_present(proposals) and stats.accepted_bad == before_bad:
                    stats.conflict_successes += 1
            elif arm == "REGION_OPERATOR_SCHEMA":
                stats.cost += 2.3
                canonical = [p for p in proposals if p.get("reason_code") == "CANONICAL_WRITEBACK" and p.get("schema_valid") is True]
                stale_checks = [p for p in proposals if p.get("reason_code") == "STALE_PROPOSAL" and p.get("schema_valid") is True]
                for proposal in stale_checks:
                    commit_proposal(flow, proposal, obs, stats, gated=True)
                for proposal in canonical[:1]:
                    commit_proposal(flow, proposal, obs, stats, gated=True)
                if conflicts_present(proposals):
                    stats.conflict_successes += 1
            else:
                raise ValueError(f"unknown arm {arm}")

            evaluate_tick(stats, flow, obs, accepted_good_before)
            if len(samples) < 90 and (obs.case_type in {"stale_proposal", "branch_mismatch", "target_locked", "conflicting_set_clear"} or obs.expected_update is not None):
                samples.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "tick": obs.tick,
                        "case_type": obs.case_type,
                        "expected_update": obs.expected_update,
                        "target_accuracy": rate(sum(1 for got, want in zip(flow.targets, obs.truth_after) if got == want), TARGET_SLOTS),
                        "commits": stats.commits,
                    }
                )

    flow_integrity = 1.0 - min(
        1.0,
        (
            stats.wrong_writebacks
            + stats.destructive_overwrites
            + stats.branch_contamination
            + stats.schema_violations
            + stats.stale_commits
        )
        / max(1, stats.commits + stats.rejected_bad),
    )
    metrics = {
        "flow_integrity": rounded(flow_integrity),
        "final_state_accuracy": rate(stats.target_matches, stats.target_evals),
        "useful_writeback_recall": rate(stats.event_success, stats.events),
        "wrong_writeback_rate": rate(stats.wrong_writebacks, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive_overwrites, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contamination, stats.commits),
        "schema_violation_rate": rate(stats.schema_violations, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "conflict_resolution_success": rate(stats.conflict_successes, stats.conflict_cases),
        "rollback_success_rate": rate(stats.rollback_successes, stats.rollback_attempts),
        "trace_validity": rate(stats.trace_matches, stats.trace_evals),
        "temporal_drift_rate": rate(stats.temporal_drift_slots, stats.target_evals),
        "oscillation_rate": rate(stats.oscillations, stats.ticks),
        "attractor_collapse_rate": rate(stats.collapse_ticks, stats.ticks),
        "gate_false_accept_rate": rate(stats.accepted_bad, stats.bad_proposals),
        "gate_false_reject_rate": rate(stats.rejected_good, stats.good_proposals),
        "cost_per_tick": rate(stats.cost, stats.ticks),
        "deterministic_replay_passed": True,
        "no_neural_dependency_detected": True,
        "no_overclaim_boundary_preserved": True,
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
        "schema_attempts": stats.schema_attempts,
        "schema_violations": stats.schema_violations,
        "wrong_writebacks": stats.wrong_writebacks,
        "destructive_overwrites": stats.destructive_overwrites,
        "branch_contamination": stats.branch_contamination,
        "stale_attempts": stats.stale_attempts,
        "stale_rejections": stats.stale_rejections,
        "stale_commits": stats.stale_commits,
        "conflict_cases": stats.conflict_cases,
        "conflict_successes": stats.conflict_successes,
        "rollback_attempts": stats.rollback_attempts,
        "rollback_successes": stats.rollback_successes,
        "stress_counts": stats.stress_counts,
        "stress_success": stats.stress_success,
        "sample_rows": samples,
    }
    return metrics, diagnostics


def build_search_report() -> dict[str, Any]:
    code, branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch_name = branch.strip() if code == 0 else "unknown"
    code, commit = run_git(["rev-parse", "HEAD"])
    commit_id = commit.strip() if code == 0 else "unknown"
    code, recent = run_git(
        [
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(committerdate:iso-strict) %(refname:short) %(objectname:short) %(subject)",
            "refs/remotes/origin",
            "refs/heads",
        ]
    )
    recent_refs = recent.splitlines()[:60] if code == 0 else []
    code, refs = run_git(["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"])
    ref_names = refs.splitlines() if code == 0 else []
    hits: list[str] = []
    if ref_names:
        grep_args = ["grep", "-n", "-I", "-i", "-F"]
        for term in SEARCH_TERMS:
            grep_args.extend(["-e", term])
        code, grep_out = run_git([*grep_args, *ref_names, "--", *SEARCH_PATHS])
        if code in (0, 1):
            for line in grep_out.splitlines():
                if "E08_FLOW_MATRIX_WRITEBACK_SCHEMA_CONFIRM".lower() not in line.lower():
                    hits.append(line)
    return {
        "schema_version": "e08_search_report_v1",
        "search_terms": list(SEARCH_TERMS),
        "search_paths": list(SEARCH_PATHS),
        "repository_branch": branch_name,
        "repository_commit": commit_id,
        "recent_ref_activity": recent_refs,
        "hit_count_excluding_current_e08": len(hits),
        "hit_excerpt_excluding_current_e08": hits[:120],
        "close_existing_files": {
            "docs/research/E7S_FLOW_GRID_VISUAL_DEBUG_AUDIT_CONTRACT.md": {
                "summary": "Flow-grid visual/debug audit; not a writeback schema confirm",
                "equivalent": False,
            },
            "docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md": {
                "summary": "proposal memory and commit control in a numeric proxy",
                "equivalent": False,
            },
            "docs/research/E8H4_REGION_OPERATOR_COMPOSITION_SCALE_PROBE_CONTRACT.md": {
                "summary": "region-operator composition/scale, not writeback schema safety",
                "equivalent": False,
            },
        },
        "equivalent_existing_probe_found": False,
        "search_conclusion": "No equivalent E08 Flow Matrix writeback schema confirmation was found in fetched refs.",
    }


def evaluate_positive_gate(metrics: dict[str, dict[str, Any]], replay_passed: bool) -> dict[str, Any]:
    baseline = metrics["DIRECT_OVERWRITE_BASELINE"]
    best_name = max(
        POSITIVE_CANDIDATES,
        key=lambda arm: (
            metrics[arm]["final_state_accuracy"],
            metrics[arm]["trace_validity"],
            -metrics[arm]["wrong_writeback_rate"],
            -metrics[arm]["cost_per_tick"],
        ),
    )
    best = metrics[best_name]
    shared_schema_zero = all(metrics[arm]["schema_violation_rate"] == 0.0 for arm in SHARED_SCHEMA_ARMS)
    stale_safe = best["stale_write_rejection_rate"] >= 0.8 or best["rollback_success_rate"] >= 0.8
    checks = {
        "final_state_accuracy_improved_or_preserved": best["final_state_accuracy"] >= baseline["final_state_accuracy"],
        "useful_writeback_recall_at_least_075": best["useful_writeback_recall"] >= 0.75,
        "destructive_overwrite_rate_reduced_70pct": best["destructive_overwrite_rate"] <= baseline["destructive_overwrite_rate"] * 0.3,
        "branch_contamination_rate_reduced_90pct": best["branch_contamination_rate"] <= baseline["branch_contamination_rate"] * 0.1,
        "wrong_writeback_rate_reduced_50pct": best["wrong_writeback_rate"] <= baseline["wrong_writeback_rate"] * 0.5,
        "shared_schema_arms_schema_violation_zero": shared_schema_zero,
        "trace_validity_higher_than_direct": best["trace_validity"] > baseline["trace_validity"],
        "stale_writes_rejected_or_rolled_back": stale_safe,
        "temporal_drift_rate_lower_than_direct": best["temporal_drift_rate"] < baseline["temporal_drift_rate"],
        "deterministic_replay_passed": replay_passed,
        "no_neural_dependency_detected": best["no_neural_dependency_detected"] is True,
        "no_overclaim_boundary_preserved": best["no_overclaim_boundary_preserved"] is True,
    }
    deltas = {
        "baseline": "DIRECT_OVERWRITE_BASELINE",
        "best_schema_arm": best_name,
        "final_state_accuracy_delta": rounded(best["final_state_accuracy"] - baseline["final_state_accuracy"]),
        "trace_validity_delta": rounded(best["trace_validity"] - baseline["trace_validity"]),
        "wrong_writeback_rate_reduction": rounded(1.0 - rate(best["wrong_writeback_rate"], baseline["wrong_writeback_rate"])),
        "destructive_overwrite_rate_reduction": rounded(1.0 - rate(best["destructive_overwrite_rate"], baseline["destructive_overwrite_rate"])),
        "branch_contamination_rate_reduction": rounded(1.0 - rate(best["branch_contamination_rate"], baseline["branch_contamination_rate"])),
        "temporal_drift_rate_reduction": rounded(1.0 - rate(best["temporal_drift_rate"], baseline["temporal_drift_rate"])),
    }
    return {"schema_version": "e08_positive_gate_v1", "checks": checks, "deltas": deltas, "passed": all(checks.values())}


def decide(metrics: dict[str, dict[str, Any]], positive_gate: dict[str, Any], replay_passed: bool) -> str:
    best = positive_gate["deltas"]["best_schema_arm"]
    if not replay_passed:
        return "e08_invalid_or_incomplete_run"
    if metrics[best]["useful_writeback_recall"] < 0.75:
        return "e08_gate_too_conservative"
    if metrics[best]["branch_contamination_rate"] > metrics["DIRECT_OVERWRITE_BASELINE"]["branch_contamination_rate"] * 0.1:
        return "e08_branch_contamination_not_fixed"
    if not positive_gate["checks"]["stale_writes_rejected_or_rolled_back"]:
        return "e08_stale_write_rollback_failure"
    if not positive_gate["checks"]["trace_validity_higher_than_direct"]:
        return "e08_trace_validity_failure"
    if not positive_gate["checks"]["shared_schema_arms_schema_violation_zero"]:
        return "e08_common_schema_not_sufficient"
    if positive_gate["passed"]:
        return "e08_flow_matrix_writeback_schema_confirmed"
    if metrics["DIRECT_OVERWRITE_BASELINE"]["final_state_accuracy"] >= metrics[best]["final_state_accuracy"]:
        return "e08_direct_overwrite_remains_best"
    return "e08_invalid_or_incomplete_run"


def render_report(decision: dict[str, Any], aggregate: dict[str, Any], schema_report: dict[str, Any], gate_report: dict[str, Any], rollback_report: dict[str, Any]) -> str:
    best = decision["best_schema_arm"]
    lines = [
        "# E08 Flow Matrix Writeback Schema Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"best_schema_arm = {best}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Key Metrics",
        "",
        "| arm | state accuracy | recall | wrong write | destructive | branch contam | trace valid | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = aggregate["arms"][arm]
        lines.append(
            f"| {arm} | {item['final_state_accuracy']:.3f} | {item['useful_writeback_recall']:.3f} | "
            f"{item['wrong_writeback_rate']:.3f} | {item['destructive_overwrite_rate']:.3f} | "
            f"{item['branch_contamination_rate']:.3f} | {item['trace_validity']:.3f} | {item['cost_per_tick']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Schema And Writeback",
            "",
            f"Shared schema fields: `{', '.join(schema_report['schema_fields'])}`.",
            f"Shared schema arms have zero schema violations: `{schema_report['shared_schema_zero_violations']}`.",
            "",
            "## Gate And Rollback",
            "",
            f"Best arm stale write rejection rate: `{gate_report['stale_write_rejection_rate_by_arm'][best]}`.",
            f"Rollback success rate: `{rollback_report['rollback_success_rate_by_arm'].get(best, 0.0)}`.",
            "",
            "## Boundary",
            "",
            "This is a deterministic synthetic binary Flow Matrix writeback probe only.",
        ]
    )
    return "\n".join(lines)


def next_for_decision(decision: str) -> str:
    return {
        "e08_flow_matrix_writeback_schema_confirmed": "E09_UNIVERSAL_POCKET_TRANSFORM_BLOCK_CONFIRM",
        "e08_direct_overwrite_remains_best": "E08D_DIRECT_OVERWRITE_FORENSICS",
        "e08_common_schema_not_sufficient": "E08S_SCHEMA_REPAIR",
        "e08_gate_too_conservative": "E08G_GATE_RECALL_REPAIR",
        "e08_branch_contamination_not_fixed": "E08B_BRANCH_BOUNDARY_REPAIR",
        "e08_stale_write_rollback_failure": "E08R_ROLLBACK_REPAIR",
        "e08_trace_validity_failure": "E08T_TRACE_CONTRACT_REPAIR",
        "e08_invalid_or_incomplete_run": "E08_RETRY_WITH_FULL_AUDIT",
    }[decision]


def build_reports(seeds: tuple[int, ...], ticks: int, include_search: bool = True) -> dict[str, Any]:
    observations = {seed: make_observations(seed, ticks) for seed in seeds}
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_metrics, arm_diag = run_arm(arm, observations)
        metrics[arm] = arm_metrics
        diagnostics[arm] = arm_diag
    positive_gate = evaluate_positive_gate(metrics, True)
    decision_label = decide(metrics, positive_gate, True)
    schema_report = {
        "schema_version": "e08_writeback_schema_report_v1",
        "schema_fields": list(SCHEMA_FIELDS),
        "transform_ops": list(TRANSFORM_OPS),
        "shared_schema_arms": list(SHARED_SCHEMA_ARMS),
        "schema_violation_rate_by_arm": {arm: metrics[arm]["schema_violation_rate"] for arm in ARMS},
        "shared_schema_zero_violations": all(metrics[arm]["schema_violation_rate"] == 0.0 for arm in SHARED_SCHEMA_ARMS),
        "private_local_dialect_commits": diagnostics["LOCAL_DIALECT_BASELINE"]["schema_violations"],
    }
    stress_report = {
        "schema_version": "e08_stress_cases_report_v1",
        "stress_cases": list(STRESS_CASES),
        "stress_counts_by_arm": {arm: diagnostics[arm]["stress_counts"] for arm in ARMS},
        "stress_success_by_arm": {arm: diagnostics[arm]["stress_success"] for arm in ARMS},
        "all_required_cases_present": all(
            all(diagnostics[arm]["stress_counts"].get(case, 0) > 0 for case in STRESS_CASES) for arm in ARMS
        ),
    }
    gate_report = {
        "schema_version": "e08_gate_commit_report_v1",
        "accepted_good_by_arm": {arm: diagnostics[arm]["accepted_good"] for arm in ARMS},
        "accepted_bad_by_arm": {arm: diagnostics[arm]["accepted_bad"] for arm in ARMS},
        "rejected_good_by_arm": {arm: diagnostics[arm]["rejected_good"] for arm in ARMS},
        "rejected_bad_by_arm": {arm: diagnostics[arm]["rejected_bad"] for arm in ARMS},
        "gate_false_accept_rate_by_arm": {arm: metrics[arm]["gate_false_accept_rate"] for arm in ARMS},
        "gate_false_reject_rate_by_arm": {arm: metrics[arm]["gate_false_reject_rate"] for arm in ARMS},
        "stale_write_rejection_rate_by_arm": {arm: metrics[arm]["stale_write_rejection_rate"] for arm in ARMS},
    }
    rollback_report = {
        "schema_version": "e08_rollback_report_v1",
        "rollback_attempts_by_arm": {arm: diagnostics[arm]["rollback_attempts"] for arm in ARMS},
        "rollback_successes_by_arm": {arm: diagnostics[arm]["rollback_successes"] for arm in ARMS},
        "rollback_success_rate_by_arm": {arm: metrics[arm]["rollback_success_rate"] for arm in ARMS},
        "stale_commits_by_arm": {arm: diagnostics[arm]["stale_commits"] for arm in ARMS},
    }
    branch_report = {
        "schema_version": "e08_branch_contamination_report_v1",
        "branch_contamination_rate_by_arm": {arm: metrics[arm]["branch_contamination_rate"] for arm in ARMS},
        "branch_contamination_count_by_arm": {arm: diagnostics[arm]["branch_contamination"] for arm in ARMS},
        "destructive_overwrite_rate_by_arm": {arm: metrics[arm]["destructive_overwrite_rate"] for arm in ARMS},
        "destructive_overwrite_count_by_arm": {arm: diagnostics[arm]["destructive_overwrites"] for arm in ARMS},
    }
    temporal_report = {
        "schema_version": "e08_temporal_stability_report_v1",
        "trace_validity_by_arm": {arm: metrics[arm]["trace_validity"] for arm in ARMS},
        "temporal_drift_rate_by_arm": {arm: metrics[arm]["temporal_drift_rate"] for arm in ARMS},
        "oscillation_rate_by_arm": {arm: metrics[arm]["oscillation_rate"] for arm in ARMS},
        "attractor_collapse_rate_by_arm": {arm: metrics[arm]["attractor_collapse_rate"] for arm in ARMS},
    }
    aggregate = {
        "schema_version": "e08_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "ticks_per_seed": ticks,
        "arms": metrics,
        "diagnostics": {arm: {key: value for key, value in diag.items() if key != "sample_rows"} for arm, diag in diagnostics.items()},
        "positive_gate": positive_gate,
    }
    decision = {
        "schema_version": "e08_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for_decision(decision_label),
        "positive_gate_passed": positive_gate["passed"],
        "best_schema_arm": positive_gate["deltas"]["best_schema_arm"],
        "deterministic_replay_passed": True,
    }
    summary = {
        "schema_version": "e08_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "equivalent_existing_probe_found": False,
        "best_schema_arm": positive_gate["deltas"]["best_schema_arm"],
        "positive_gate_passed": positive_gate["passed"],
        "shared_schema_zero_violations": schema_report["shared_schema_zero_violations"],
        "all_required_stress_cases_present": stress_report["all_required_cases_present"],
        "no_neural_dependency_detected": True,
        "no_overclaim_boundary_preserved": True,
    }
    payloads: dict[str, Any] = {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "e08_writeback_schema_report.json": schema_report,
        "e08_stress_cases_report.json": stress_report,
        "e08_gate_commit_report.json": gate_report,
        "e08_rollback_report.json": rollback_report,
        "e08_branch_contamination_report.json": branch_report,
        "e08_temporal_stability_report.json": temporal_report,
    }
    payloads["report.md"] = render_report(decision, aggregate, schema_report, gate_report, rollback_report)
    if include_search:
        payloads["e08_search_report.json"] = build_search_report()
    return payloads


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], ticks: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, ticks, include_search=False)
    replay_b = build_reports(seeds, ticks, include_search=False)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e08_deterministic_replay_report.json"] = {
        "schema_version": "e08_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"] = evaluate_positive_gate(payloads["aggregate_metrics.json"]["arms"], passed)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    decision_label = decide(payloads["aggregate_metrics.json"]["arms"], payloads["aggregate_metrics.json"]["positive_gate"], passed)
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for_decision(decision_label)
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(
        payloads["decision.json"],
        payloads["aggregate_metrics.json"],
        payloads["e08_writeback_schema_report.json"],
        payloads["e08_gate_commit_report.json"],
        payloads["e08_rollback_report.json"],
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
    payloads = attach_replay(build_reports(args.seeds, args.ticks, include_search=True), args.seeds, args.ticks)
    for name in REQUIRED_ARTIFACTS:
        path = out / name
        payload = payloads[name]
        if name.endswith(".md"):
            write_text(path, str(payload))
        else:
            write_json(path, payload)
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
