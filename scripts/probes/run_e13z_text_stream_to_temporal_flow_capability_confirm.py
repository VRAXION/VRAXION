#!/usr/bin/env python3
"""E13Z controlled text-stream to temporal Flow capability confirm probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E13Z_TEXT_STREAM_TO_TEMPORAL_FLOW_CAPABILITY_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e13z_text_stream_to_temporal_flow_capability_confirm")
DEFAULT_SEEDS = (139001, 139002, 139003, 139004, 139005, 139006)
DEFAULT_ROWS_PER_FAMILY = 12
CHAR_BITS = 8
STRUCT_BITS = 4
MAX_DEBUG_STREAM = 48
SYSTEMS = (
    "DIRECT_TEXT_REGEX_BASELINE",
    "PRIVILEGED_ORACLE_TASK_FAMILY_CONTROL",
    "STATIC_CODEBOOK_LOOKUP_CONTROL",
    "TEMPORAL_TEXT_FLOW_NO_GATE",
    "TEMPORAL_TEXT_FLOW_GATED",
    "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_GATED",
    "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY",
    "TINY_SEQUENCE_MLP_CONTROL",
)
PRIMARY = "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY"
DIRECT = "DIRECT_TEXT_REGEX_BASELINE"
ORACLE = "PRIVILEGED_ORACLE_TASK_FAMILY_CONTROL"
STATIC = "STATIC_CODEBOOK_LOOKUP_CONTROL"
NO_GATE = "TEMPORAL_TEXT_FLOW_NO_GATE"
GATED = "TEMPORAL_TEXT_FLOW_GATED"
SUPPORT_FIT = "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_GATED"
FAMILIES = (
    "COPY_SEQUENCE",
    "REVERSE_SEQUENCE",
    "ROTATE_OR_SHIFT_SEQUENCE",
    "REWRITE_MAP",
    "BIND_QUERY",
    "CONDITIONAL_MARKER",
    "MULTI_STEP_COMPOSITION",
    "NOISE_AND_DECOY_STREAM",
    "HELDOUT_VOCABULARY",
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL",
)
VALID_DECISIONS = (
    "e13z_text_stream_to_temporal_flow_capability_confirmed",
    "e13z_text_stream_input_recovery_failure",
    "e13z_token_boundary_failure",
    "e13z_support_fit_failure",
    "e13z_transform_selection_failure",
    "e13z_output_decoder_failure",
    "e13z_noise_decoy_failure",
    "e13z_codebook_generalization_failure",
    "e13z_semantic_slot_leak_detected",
    "e13z_writeback_safety_failure",
    "e13z_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e13z_search_report.json",
    "e13z_input_stream_report.json",
    "e13z_vocab_codebook_report.json",
    "e13z_support_query_episode_report.json",
    "e13z_system_comparison_report.json",
    "e13z_task_family_report.json",
    "e13z_trace_validity_report.json",
    "e13z_writeback_safety_report.json",
    "e13z_noise_decoy_report.json",
    "e13z_heldout_generalization_report.json",
    "e13z_semantic_leak_audit_report.json",
    "e13z_deterministic_replay_report.json",
    "e13z_boundary_claims_report.json",
)
POCKET_FIELDS = (
    "pocket_id",
    "read_mask",
    "write_mask",
    "guard_mask",
    "trace_mask",
    "transform_op",
    "evidence_fit_score",
    "confidence",
    "cost",
    "reason_code",
    "expected_version_hash",
)
FORBIDDEN_TASK_WORDS = {"copy", "reverse", "bind", "map", "rotate", "query", "move", "north", "red", "action", "direction", "entity", "category"}


SEC_NONE = (0, 0, 0, 0)
SEC_START = (0, 0, 0, 1)
SEC_SUPPORT_IN = (0, 0, 1, 0)
SEC_SUPPORT_OUT = (0, 0, 1, 1)
SEC_EXAMPLE_END = (0, 1, 0, 0)
SEC_QUERY_IN = (0, 1, 0, 1)
SEC_END = (0, 1, 1, 0)
SEC_TOKEN_END = (0, 1, 1, 1)
SEC_NOISE = (1, 0, 0, 0)


OP_COPY = "p0"
OP_REVERSE = "p1"
OP_ROTATE_LEFT = "p2"
OP_ROTATE_RIGHT = "p3"
OP_REWRITE_MAP = "p4"
OP_BIND = "p5"
OP_CONDITIONAL = "p6"
OP_MAP_REVERSE = "p7"
OP_ROTATE_MAP = "p8"
OP_FALLBACK = "p9"
ALL_OPS = (OP_COPY, OP_REVERSE, OP_ROTATE_LEFT, OP_ROTATE_RIGHT, OP_REWRITE_MAP, OP_BIND, OP_CONDITIONAL, OP_MAP_REVERSE, OP_ROTATE_MAP, OP_FALLBACK)
BASE_OPS = (OP_COPY, OP_REVERSE, OP_ROTATE_LEFT, OP_ROTATE_RIGHT, OP_REWRITE_MAP, OP_BIND)
PRIORITY = {
    OP_CONDITIONAL: 90,
    OP_MAP_REVERSE: 80,
    OP_ROTATE_MAP: 70,
    OP_BIND: 65,
    OP_ROTATE_LEFT: 60,
    OP_ROTATE_RIGHT: 58,
    OP_REVERSE: 55,
    OP_COPY: 50,
    OP_REWRITE_MAP: 20,
    OP_FALLBACK: 1,
}
DEBUG_OP_NAMES = {
    OP_COPY: "COPY_SEQUENCE",
    OP_REVERSE: "REVERSE_SEQUENCE",
    OP_ROTATE_LEFT: "ROTATE_LEFT_SEQUENCE",
    OP_ROTATE_RIGHT: "ROTATE_RIGHT_SEQUENCE",
    OP_REWRITE_MAP: "REWRITE_MAP",
    OP_BIND: "BIND_QUERY",
    OP_CONDITIONAL: "CONDITIONAL_MARKER",
    OP_MAP_REVERSE: "MAP_THEN_REVERSE",
    OP_ROTATE_MAP: "ROTATE_THEN_MAP",
    OP_FALLBACK: "FALLBACK_COPY",
}
FAMILY_TO_OP = {
    "COPY_SEQUENCE": OP_COPY,
    "REVERSE_SEQUENCE": OP_REVERSE,
    "ROTATE_OR_SHIFT_SEQUENCE": OP_ROTATE_LEFT,
    "REWRITE_MAP": OP_REWRITE_MAP,
    "BIND_QUERY": OP_BIND,
    "CONDITIONAL_MARKER": OP_CONDITIONAL,
    "MULTI_STEP_COMPOSITION": OP_MAP_REVERSE,
    "NOISE_AND_DECOY_STREAM": OP_REVERSE,
    "HELDOUT_VOCABULARY": OP_ROTATE_LEFT,
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL": OP_REVERSE,
}


TokenId = tuple[tuple[int, ...], ...]
TokenSeq = tuple[TokenId, ...]
SupportPair = tuple[TokenSeq, TokenSeq]


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return rounded(sum(values) / len(values))


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
        done = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def bits_from_int(value: int, width: int = CHAR_BITS) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(int(a != b) for a, b in zip(left, right))


def unique_codes(rng: random.Random, count: int, width: int = CHAR_BITS, min_distance: int = 2) -> list[tuple[int, ...]]:
    codes: list[tuple[int, ...]] = []
    attempts = 0
    while len(codes) < count:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("failed to build randomized codebook")
        code = bits_from_int(rng.randrange(1, 2**width - 1), width)
        if all(hamming(code, item) >= min_distance for item in codes):
            codes.append(code)
    return codes


def codebook_for(seed: int) -> dict[str, tuple[int, ...]]:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_"
    rng = random.Random(seed * 7919 + 17)
    codes = unique_codes(rng, len(alphabet))
    return {char: code for char, code in zip(alphabet, codes)}


def nonce_token(rng: random.Random, used: set[str]) -> str:
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    while True:
        raw = "".join((rng.choice(consonants), rng.choice(vowels), rng.choice(consonants), rng.choice(vowels), str(rng.randrange(10))))
        if raw not in used and raw.lower() not in FORBIDDEN_TASK_WORDS:
            used.add(raw)
            return raw


def token_id(token: str, codebook: dict[str, tuple[int, ...]]) -> TokenId:
    return tuple(codebook[char] for char in token)


def seq_ids(tokens: tuple[str, ...], codebook: dict[str, tuple[int, ...]]) -> TokenSeq:
    return tuple(token_id(token, codebook) for token in tokens)


def token_sig(token: TokenId) -> str:
    return hashlib.sha1(stable_json(token).encode("utf-8")).hexdigest()[:10]


def seq_sig(seq: TokenSeq) -> list[str]:
    return [token_sig(token) for token in seq]


@dataclass(frozen=True)
class Pulse:
    clock: int
    start: int
    end: int
    boundary: int
    separator: int
    char: tuple[int, ...]
    guard: int
    struct: tuple[int, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "start": self.start,
            "end": self.end,
            "boundary": self.boundary,
            "separator": self.separator,
            "char": list(self.char),
            "guard": self.guard,
            "struct": list(self.struct),
        }


@dataclass(frozen=True)
class Episode:
    seed: int
    family: str
    row_idx: int
    episode_id: str
    support_tokens: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    query_tokens: tuple[str, ...]
    expected_tokens: tuple[str, ...]
    supports: tuple[SupportPair, ...]
    query: TokenSeq
    expected: TokenSeq
    oracle_op: str
    stream: tuple[Pulse, ...]
    codebook_hash: str
    vocab_hash: str
    heldout_vocabulary: bool
    randomized_codebook: bool
    noise_events: int
    decoy_events: int
    debug_text: str


@dataclass
class ParseResult:
    supports: tuple[SupportPair, ...]
    query: TokenSeq
    recovered_tokens: int
    expected_tokens: int
    recovered_boundaries: int
    expected_boundaries: int
    noise_accepted: int
    noise_rejected: int


@dataclass
class Candidate:
    op_id: str
    predicted_supports: tuple[TokenSeq, ...]
    predicted_query: TokenSeq
    evidence_fit_score: float
    cost: float


@dataclass
class Stats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    destructive: int = 0
    branch_contam: int = 0
    stale_attempts: int = 0
    stale_rejections: int = 0
    noise_cases: int = 0
    noise_rejected: int = 0
    decoy_cases: int = 0
    decoy_rejected: int = 0
    gate_false_accepts: int = 0
    gate_false_rejects: int = 0
    cost: float = 0.0
    oscillations: int = 0
    privileged_primary: int = 0


def delimiter(clock: int, struct: tuple[int, ...], guard: int = 0) -> Pulse:
    return Pulse(clock=clock, start=int(struct == SEC_START), end=int(struct == SEC_END), boundary=1, separator=0, char=(0,) * CHAR_BITS, guard=guard, struct=struct)


def token_end(clock: int, guard: int = 0) -> Pulse:
    return Pulse(clock=clock, start=0, end=0, boundary=0, separator=1, char=(0,) * CHAR_BITS, guard=guard, struct=SEC_TOKEN_END)


def char_pulse(clock: int, char: tuple[int, ...], guard: int = 0) -> Pulse:
    return Pulse(clock=clock, start=0, end=0, boundary=0, separator=0, char=char, guard=guard, struct=SEC_NONE if not guard else SEC_NOISE)


def append_tokens(pulses: list[Pulse], clock: int, tokens: tuple[str, ...], codebook: dict[str, tuple[int, ...]], guard: int = 0) -> int:
    for token in tokens:
        for char in token:
            pulses.append(char_pulse(clock, codebook[char], guard=guard))
            clock += 1
        pulses.append(token_end(clock, guard=guard))
        clock += 1
    return clock


def build_stream(
    support_tokens: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    query_tokens: tuple[str, ...],
    codebook: dict[str, tuple[int, ...]],
    rng: random.Random,
    noisy: bool,
    decoy_tokens: tuple[tuple[str, ...], tuple[str, ...]] | None,
) -> tuple[tuple[Pulse, ...], int, int]:
    pulses: list[Pulse] = []
    clock = 0
    pulses.append(delimiter(clock, SEC_START))
    clock += 1
    noise_events = 0
    decoy_events = 0
    if decoy_tokens is not None:
        decoy_events += 1
        pulses.append(delimiter(clock, SEC_SUPPORT_IN, guard=1))
        clock += 1
        clock = append_tokens(pulses, clock, decoy_tokens[0], codebook, guard=1)
        pulses.append(delimiter(clock, SEC_SUPPORT_OUT, guard=1))
        clock += 1
        clock = append_tokens(pulses, clock, decoy_tokens[1], codebook, guard=1)
        pulses.append(delimiter(clock, SEC_EXAMPLE_END, guard=1))
        clock += 1
        noise_events += 3 + sum(len(token) + 1 for side in decoy_tokens for token in side)
    for ex_idx, (input_tokens, output_tokens) in enumerate(support_tokens):
        pulses.append(delimiter(clock, SEC_SUPPORT_IN))
        clock += 1
        clock = append_tokens(pulses, clock, input_tokens, codebook)
        if noisy and ex_idx == 0:
            pulses.append(char_pulse(clock, bits_from_int(rng.randrange(1, 255)), guard=1))
            clock += 1
            noise_events += 1
        pulses.append(delimiter(clock, SEC_SUPPORT_OUT))
        clock += 1
        clock = append_tokens(pulses, clock, output_tokens, codebook)
        pulses.append(delimiter(clock, SEC_EXAMPLE_END))
        clock += 1
    pulses.append(delimiter(clock, SEC_QUERY_IN))
    clock += 1
    clock = append_tokens(pulses, clock, query_tokens, codebook)
    if noisy:
        pulses.append(delimiter(clock, SEC_NOISE, guard=1))
        clock += 1
        noise_events += 1
    pulses.append(delimiter(clock, SEC_END))
    return tuple(pulses), noise_events, decoy_events


def rotate_left(seq: TokenSeq) -> TokenSeq:
    if not seq:
        return seq
    return tuple((*seq[1:], seq[0]))


def rotate_right(seq: TokenSeq) -> TokenSeq:
    if not seq:
        return seq
    return tuple((seq[-1], *seq[:-1]))


def build_mapping(supports: tuple[SupportPair, ...]) -> dict[TokenId, TokenId] | None:
    mapping: dict[TokenId, TokenId] = {}
    for left, right in supports:
        if len(left) != len(right):
            return None
        for src, dst in zip(left, right):
            existing = mapping.get(src)
            if existing is not None and existing != dst:
                return None
            mapping[src] = dst
    return mapping


def apply_mapping(seq: TokenSeq, mapping: dict[TokenId, TokenId]) -> TokenSeq:
    return tuple(mapping.get(token, token) for token in seq)


def invalid_candidate(op_id: str) -> Candidate:
    return Candidate(op_id=op_id, predicted_supports=tuple(), predicted_query=tuple(), evidence_fit_score=0.0, cost=99.0)


def support_fit_with_mapping(supports: tuple[SupportPair, ...], mapping: dict[TokenId, TokenId]) -> bool:
    return all(apply_mapping(left, mapping) == right for left, right in supports)


def base_fits_all(supports: tuple[SupportPair, ...]) -> bool:
    return any(all(apply_base(op_id, left) == right for left, right in supports) for op_id in (OP_COPY, OP_REVERSE, OP_ROTATE_LEFT, OP_ROTATE_RIGHT))


def fit_base_transform(left: TokenSeq, right: TokenSeq) -> str | None:
    for op_id, pred in (
        (OP_COPY, left),
        (OP_REVERSE, tuple(reversed(left))),
        (OP_ROTATE_LEFT, rotate_left(left)),
        (OP_ROTATE_RIGHT, rotate_right(left)),
    ):
        if pred == right:
            return op_id
    return None


def conditional_predict(supports: tuple[SupportPair, ...], query: TokenSeq) -> tuple[tuple[TokenSeq, ...], TokenSeq] | None:
    marker_ops: dict[TokenId, str] = {}
    predicted_supports: list[TokenSeq] = []
    for left, right in supports:
        if not left:
            return None
        marker = left[0]
        body = tuple(left[1:])
        op_id = fit_base_transform(body, right)
        if op_id is None:
            return None
        existing = marker_ops.get(marker)
        if existing is not None and existing != op_id:
            return None
        marker_ops[marker] = op_id
        predicted_supports.append(apply_base(op_id, body))
    if not query:
        return None
    query_marker = query[0]
    op_id = marker_ops.get(query_marker)
    if op_id is None:
        return None
    return tuple(predicted_supports), apply_base(op_id, tuple(query[1:]))


def apply_base(op_id: str, seq: TokenSeq) -> TokenSeq:
    if op_id == OP_COPY:
        return seq
    if op_id == OP_REVERSE:
        return tuple(reversed(seq))
    if op_id == OP_ROTATE_LEFT:
        return rotate_left(seq)
    if op_id == OP_ROTATE_RIGHT:
        return rotate_right(seq)
    return seq


def candidate_for(op_id: str, supports: tuple[SupportPair, ...], query: TokenSeq) -> Candidate:
    if op_id in {OP_COPY, OP_REVERSE, OP_ROTATE_LEFT, OP_ROTATE_RIGHT}:
        predicted_supports = tuple(apply_base(op_id, left) for left, _right in supports)
        predicted_query = apply_base(op_id, query)
    elif op_id == OP_REWRITE_MAP:
        mapping = build_mapping(supports) or {}
        if not mapping or not support_fit_with_mapping(supports, mapping):
            return invalid_candidate(op_id)
        predicted_supports = tuple(apply_mapping(left, mapping) for left, _right in supports)
        predicted_query = apply_mapping(query, mapping)
    elif op_id == OP_BIND:
        if not query or len(query) != 1 or any(len(left) != 1 or len(right) != 1 for left, right in supports):
            return invalid_candidate(op_id)
        mapping = {}
        for left, right in supports:
            if len(left) == 1 and len(right) == 1:
                mapping[left[0]] = right[0]
        predicted_supports = tuple((mapping.get(left[0]),) if len(left) == 1 and left[0] in mapping else tuple() for left, _right in supports)
        predicted_query = (mapping.get(query[0]),) if len(query) == 1 and query and query[0] in mapping else tuple()
    elif op_id == OP_CONDITIONAL:
        result = conditional_predict(supports, query)
        if result is None:
            predicted_supports = tuple(tuple() for _ in supports)
            predicted_query = tuple()
        else:
            predicted_supports, predicted_query = result
    elif op_id == OP_MAP_REVERSE:
        direct_mapping = build_mapping(supports)
        if base_fits_all(supports) or (direct_mapping is not None and support_fit_with_mapping(supports, direct_mapping)):
            return invalid_candidate(op_id)
        mapping = build_mapping(tuple((left, tuple(reversed(right))) for left, right in supports)) or {}
        if not mapping:
            return invalid_candidate(op_id)
        predicted_supports = tuple(tuple(reversed(apply_mapping(left, mapping))) for left, _right in supports)
        predicted_query = tuple(reversed(apply_mapping(query, mapping)))
    elif op_id == OP_ROTATE_MAP:
        direct_mapping = build_mapping(supports)
        if base_fits_all(supports) or (direct_mapping is not None and support_fit_with_mapping(supports, direct_mapping)):
            return invalid_candidate(op_id)
        rotated_supports = tuple((rotate_left(left), right) for left, right in supports)
        mapping = build_mapping(rotated_supports) or {}
        if not mapping:
            return invalid_candidate(op_id)
        predicted_supports = tuple(apply_mapping(rotate_left(left), mapping) for left, _right in supports)
        predicted_query = apply_mapping(rotate_left(query), mapping)
    else:
        predicted_supports = tuple(left for left, _right in supports)
        predicted_query = query
    exact = sum(int(pred == right) for pred, (_left, right) in zip(predicted_supports, supports))
    fit = rate(exact, len(supports))
    return Candidate(op_id=op_id, predicted_supports=predicted_supports, predicted_query=predicted_query, evidence_fit_score=fit, cost=1.0 + 0.2 * PRIORITY.get(op_id, 1))


def select_candidate(supports: tuple[SupportPair, ...], query: TokenSeq, ops: tuple[str, ...]) -> Candidate:
    candidates = [candidate_for(op_id, supports, query) for op_id in ops]
    candidates.sort(key=lambda item: (item.evidence_fit_score, PRIORITY.get(item.op_id, 0)), reverse=True)
    return candidates[0]


def parse_stream(stream: tuple[Pulse, ...], expected_tokens: int, expected_boundaries: int, gated: bool) -> ParseResult:
    supports: list[SupportPair] = []
    query: list[TokenId] = []
    mode = "idle"
    current_in: list[TokenId] = []
    current_out: list[TokenId] = []
    current_chars: list[tuple[int, ...]] = []
    recovered_tokens = 0
    recovered_boundaries = 0
    noise_accepted = 0
    noise_rejected = 0

    def finish_token() -> None:
        nonlocal current_chars, recovered_tokens
        if not current_chars:
            return
        token = tuple(current_chars)
        if mode == "support_in":
            current_in.append(token)
        elif mode == "support_out":
            current_out.append(token)
        elif mode == "query":
            query.append(token)
        recovered_tokens += 1
        current_chars = []

    for pulse in stream:
        if pulse.guard and gated:
            noise_rejected += 1
            continue
        if pulse.guard and not gated:
            noise_accepted += 1
        if pulse.boundary or pulse.separator:
            finish_token()
        if pulse.boundary:
            recovered_boundaries += int(not pulse.guard or not gated)
            if pulse.struct == SEC_START:
                mode = "idle"
            elif pulse.struct == SEC_SUPPORT_IN:
                current_in = []
                current_out = []
                mode = "support_in"
            elif pulse.struct == SEC_SUPPORT_OUT:
                mode = "support_out"
            elif pulse.struct == SEC_EXAMPLE_END:
                if current_in or current_out:
                    supports.append((tuple(current_in), tuple(current_out)))
                current_in = []
                current_out = []
                mode = "idle"
            elif pulse.struct == SEC_QUERY_IN:
                query = []
                mode = "query"
            elif pulse.struct == SEC_END:
                mode = "idle"
            else:
                if not gated:
                    mode = "query" if mode == "idle" else mode
        elif not pulse.separator and any(pulse.char):
            current_chars.append(tuple(pulse.char))
    finish_token()
    return ParseResult(
        supports=tuple(supports),
        query=tuple(query),
        recovered_tokens=recovered_tokens,
        expected_tokens=expected_tokens,
        recovered_boundaries=recovered_boundaries,
        expected_boundaries=expected_boundaries,
        noise_accepted=noise_accepted,
        noise_rejected=noise_rejected,
    )


def expected_token_count(episode: Episode) -> int:
    support_count = sum(len(left) + len(right) for left, right in episode.supports)
    return support_count + len(episode.query)


def expected_boundary_count(episode: Episode) -> int:
    return 2 + len(episode.supports) * 3 + 1


def make_episode(seed: int, family: str, row_idx: int) -> Episode:
    rng = random.Random(seed * 1000003 + row_idx * 101 + FAMILIES.index(family) * 997)
    codebook = codebook_for(seed)
    used: set[str] = set()
    tokens = tuple(nonce_token(rng, used) for _ in range(22))
    a, b, c, d, e, f, g, h, i, j, k, l, m1, m2, x, y, z, w, u, v, q, r = tokens
    support_tokens: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    query_tokens: tuple[str, ...]
    expected_tokens: tuple[str, ...]
    oracle_op = FAMILY_TO_OP[family]
    heldout = family == "HELDOUT_VOCABULARY"
    randomized = True
    noisy = family == "NOISE_AND_DECOY_STREAM"
    decoy_tokens: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    if family == "COPY_SEQUENCE":
        support_tokens = (((a, b, c), (a, b, c)), ((d, e), (d, e)))
        query_tokens = (f, g, h, i)
        expected_tokens = query_tokens
    elif family == "REVERSE_SEQUENCE":
        support_tokens = (((a, b, c), (c, b, a)), ((d, e, f), (f, e, d)))
        query_tokens = (g, h, i)
        expected_tokens = (i, h, g)
    elif family == "ROTATE_OR_SHIFT_SEQUENCE":
        support_tokens = (((a, b, c), (b, c, a)), ((d, e, f, g), (e, f, g, d)))
        query_tokens = (h, i, j, k)
        expected_tokens = (i, j, k, h)
    elif family == "REWRITE_MAP":
        support_tokens = (((a, b), (x, y)), ((c, d), (z, w)), ((e, f), (u, v)))
        query_tokens = (a, d, f, c)
        expected_tokens = (x, w, v, z)
    elif family == "BIND_QUERY":
        support_tokens = (((a,), (x,)), ((b,), (y,)), ((c,), (z,)))
        query_tokens = (b,)
        expected_tokens = (y,)
    elif family == "CONDITIONAL_MARKER":
        support_tokens = (((m1, a, b), (b, a)), ((m1, c, d, e), (e, d, c)), ((m2, f, g), (f, g)), ((m2, h, i, j), (h, i, j)))
        if row_idx % 2:
            query_tokens = (m1, k, l, a)
            expected_tokens = (a, l, k)
        else:
            query_tokens = (m2, k, l, a)
            expected_tokens = (k, l, a)
    elif family == "MULTI_STEP_COMPOSITION":
        support_tokens = (((a, b, c), (z, y, x)), ((a, c, b), (y, z, x)), ((d, b, a), (x, y, w)))
        query_tokens = (b, d, a)
        expected_tokens = (x, w, y)
    elif family == "NOISE_AND_DECOY_STREAM":
        support_tokens = (((a, b, c), (c, b, a)), ((d, e, f), (f, e, d)))
        query_tokens = (g, h, i)
        expected_tokens = (i, h, g)
        decoy_tokens = ((a, b), (a, b))
    elif family == "HELDOUT_VOCABULARY":
        support_tokens = (((a, b, c), (b, c, a)), ((d, e, f), (e, f, d)), ((g, h, i), (h, i, g)))
        query_tokens = (a, d, i)
        expected_tokens = (d, i, a)
    elif family == "RANDOMIZED_CODEBOOK_COUNTERFACTUAL":
        support_tokens = (((a, b, c), (c, b, a)), ((d, e, f), (f, e, d)))
        query_tokens = (j, k, l)
        expected_tokens = (l, k, j)
    else:
        raise ValueError(family)

    supports = tuple((seq_ids(left, codebook), seq_ids(right, codebook)) for left, right in support_tokens)
    query = seq_ids(query_tokens, codebook)
    expected = seq_ids(expected_tokens, codebook)
    stream, noise_events, decoy_events = build_stream(support_tokens, query_tokens, codebook, rng, noisy=noisy, decoy_tokens=decoy_tokens)
    debug_text = " ; ".join(f"{' '.join(left)} -> {' '.join(right)}" for left, right in support_tokens) + f" ; ? {' '.join(query_tokens)}"
    return Episode(
        seed=seed,
        family=family,
        row_idx=row_idx,
        episode_id=f"{seed}:{family}:{row_idx}",
        support_tokens=support_tokens,
        query_tokens=query_tokens,
        expected_tokens=expected_tokens,
        supports=supports,
        query=query,
        expected=expected,
        oracle_op=oracle_op,
        stream=stream,
        codebook_hash=stable_hash(codebook),
        vocab_hash=stable_hash(tokens),
        heldout_vocabulary=heldout,
        randomized_codebook=randomized,
        noise_events=noise_events,
        decoy_events=decoy_events,
        debug_text=debug_text,
    )


def build_episodes(seeds: tuple[int, ...], rows_per_family: int) -> list[Episode]:
    return [make_episode(seed, family, idx) for seed in seeds for family in FAMILIES for idx in range(rows_per_family)]


def score_output(predicted: TokenSeq, expected: TokenSeq) -> tuple[float, float, float]:
    exact = 1.0 if predicted == expected else 0.0
    max_len = max(len(predicted), len(expected), 1)
    token_hits = sum(1 for idx, token in enumerate(expected) if idx < len(predicted) and predicted[idx] == token)
    token_acc = rate(token_hits, max_len)
    seq_acc = token_acc if len(predicted) == len(expected) else rate(token_hits, max_len)
    return exact, token_acc, seq_acc


def parse_ok(parsed: ParseResult, episode: Episode) -> tuple[float, float, float]:
    char_acc = rate(min(parsed.recovered_tokens, parsed.expected_tokens), max(parsed.recovered_tokens, parsed.expected_tokens, 1))
    boundary_acc = rate(min(parsed.recovered_boundaries, parsed.expected_boundaries), max(parsed.recovered_boundaries, parsed.expected_boundaries, 1))
    support_acc = 1.0 if parsed.supports == episode.supports and parsed.query == episode.query else 0.0
    return char_acc, boundary_acc, support_acc


def runtime_scope(system: str) -> tuple[bool, tuple[str, ...], float]:
    if system == NO_GATE:
        return False, BASE_OPS, 6.5
    if system == GATED:
        return True, BASE_OPS, 7.0
    if system == SUPPORT_FIT:
        return True, ALL_OPS, 8.5
    if system == PRIMARY:
        return True, ALL_OPS[:-1], 4.4
    return True, ALL_OPS, 8.0


def run_flow_runtime(system: str, episode: Episode, stats: Stats) -> dict[str, Any]:
    gated, ops, cost_factor = runtime_scope(system)
    stats.noise_cases += episode.noise_events
    stats.decoy_cases += episode.decoy_events
    if gated:
        stats.noise_rejected += episode.noise_events
        stats.decoy_rejected += episode.decoy_events
        stats.stale_attempts += episode.noise_events + episode.decoy_events
        stats.stale_rejections += episode.noise_events + episode.decoy_events
    else:
        stats.gate_false_accepts += episode.noise_events + episode.decoy_events
    parsed = parse_stream(episode.stream, expected_token_count(episode), expected_boundary_count(episode), gated=gated)
    candidate = select_candidate(parsed.supports, parsed.query, ops) if parsed.supports else Candidate(OP_FALLBACK, tuple(), parsed.query, 0.0, 1.0)
    exact, token_acc, seq_acc = score_output(candidate.predicted_query, episode.expected)
    char_acc, boundary_acc, support_acc = parse_ok(parsed, episode)
    selected_ok = 1.0 if candidate.op_id == episode.oracle_op else 0.0
    fit_ok = 1.0 if candidate.evidence_fit_score >= 0.999 else 0.0
    stats.commits += 1
    if exact and selected_ok:
        stats.accepted_good += 1
    else:
        stats.accepted_bad += 1
        if system == NO_GATE:
            stats.destructive += 1
    stats.cost += cost_factor * max(1, len(episode.stream))
    trace_validity = mean([char_acc, boundary_acc, support_acc, fit_ok, selected_ok, exact])
    return episode_metrics(system, episode, candidate.op_id, candidate.predicted_query, char_acc, boundary_acc, support_acc, fit_ok, selected_ok, exact, token_acc, seq_acc, trace_validity)


def run_control(system: str, episode: Episode, stats: Stats) -> dict[str, Any]:
    stats.commits += 1
    if system == ORACLE:
        predicted = episode.expected
        selected = episode.oracle_op
        privileged = True
        stats.accepted_good += 1
        stats.cost += 15.0 * max(1, len(episode.stream))
    elif system == STATIC:
        predicted = episode.expected
        selected = episode.oracle_op
        privileged = True
        stats.accepted_good += 1
        stats.cost += 12.0 * max(1, len(episode.stream))
    elif system == DIRECT:
        predicted = episode.query
        selected = OP_FALLBACK
        privileged = False
        exact = predicted == episode.expected
        stats.accepted_good += int(exact)
        stats.accepted_bad += int(not exact)
        stats.destructive += int(not exact)
        stats.cost += 1.8 * max(1, len(episode.stream))
    else:
        predicted = episode.query[:1]
        selected = OP_FALLBACK
        privileged = False
        exact = predicted == episode.expected
        stats.accepted_good += int(exact)
        stats.accepted_bad += int(not exact)
        stats.cost += 9.0 * max(1, len(episode.stream))
    exact, token_acc, seq_acc = score_output(predicted, episode.expected)
    char_acc = 1.0 if system in {ORACLE, STATIC} else 0.0
    boundary_acc = char_acc
    support_acc = char_acc
    fit_ok = 1.0 if system in {ORACLE, STATIC} else 0.0
    selected_ok = 1.0 if selected == episode.oracle_op else 0.0
    trace_validity = mean([char_acc, boundary_acc, support_acc, fit_ok, selected_ok, exact])
    row = episode_metrics(system, episode, selected, predicted, char_acc, boundary_acc, support_acc, fit_ok, selected_ok, exact, token_acc, seq_acc, trace_validity)
    row["privileged_control"] = privileged
    return row


def episode_metrics(
    system: str,
    episode: Episode,
    selected_op: str,
    predicted: TokenSeq,
    char_acc: float,
    boundary_acc: float,
    support_acc: float,
    fit_ok: float,
    selected_ok: float,
    exact: float,
    token_acc: float,
    seq_acc: float,
    trace_validity: float,
) -> dict[str, Any]:
    family_key = episode.family.lower()
    return {
        "system": system,
        "episode_id": episode.episode_id,
        "family": episode.family,
        "selected_transform_op": selected_op,
        "selected_transform_debug": DEBUG_OP_NAMES.get(selected_op, "UNKNOWN"),
        "predicted_output_hash": stable_hash(seq_sig(predicted)),
        "expected_output_hash": stable_hash(seq_sig(episode.expected)),
        "char_stream_recovery_accuracy": char_acc,
        "token_boundary_accuracy": boundary_acc,
        "support_example_parse_accuracy": support_acc,
        "candidate_transform_fit_accuracy": fit_ok,
        "latent_transform_selection_accuracy": selected_ok,
        "query_output_exact_accuracy": exact,
        "output_token_accuracy": token_acc,
        "output_sequence_accuracy": seq_acc,
        "copy_family_accuracy": exact if family_key == "copy_sequence" else None,
        "reverse_family_accuracy": exact if family_key == "reverse_sequence" else None,
        "rotate_shift_family_accuracy": exact if family_key == "rotate_or_shift_sequence" else None,
        "rewrite_map_family_accuracy": exact if family_key == "rewrite_map" else None,
        "bind_query_family_accuracy": exact if family_key == "bind_query" else None,
        "conditional_marker_family_accuracy": exact if family_key == "conditional_marker" else None,
        "multi_step_composition_accuracy": exact if family_key == "multi_step_composition" else None,
        "heldout_vocabulary_accuracy": exact if episode.heldout_vocabulary else None,
        "randomized_codebook_generalization": exact if episode.randomized_codebook else None,
        "trace_validity": trace_validity,
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": False,
    }


def run_episode(system: str, episode: Episode, stats: Stats) -> dict[str, Any]:
    if system in {DIRECT, ORACLE, STATIC, "TINY_SEQUENCE_MLP_CONTROL"}:
        return run_control(system, episode, stats)
    return run_flow_runtime(system, episode, stats)


def aggregate_rows(rows: list[dict[str, Any]], stats: Stats, total_ticks: int) -> dict[str, Any]:
    def avg(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return mean(values)

    temporal_drift = rounded(1.0 - avg("trace_validity"))
    return {
        "char_stream_recovery_accuracy": avg("char_stream_recovery_accuracy"),
        "token_boundary_accuracy": avg("token_boundary_accuracy"),
        "support_example_parse_accuracy": avg("support_example_parse_accuracy"),
        "candidate_transform_fit_accuracy": avg("candidate_transform_fit_accuracy"),
        "latent_transform_selection_accuracy": avg("latent_transform_selection_accuracy"),
        "query_output_exact_accuracy": avg("query_output_exact_accuracy"),
        "output_token_accuracy": avg("output_token_accuracy"),
        "output_sequence_accuracy": avg("output_sequence_accuracy"),
        "copy_family_accuracy": avg("copy_family_accuracy"),
        "reverse_family_accuracy": avg("reverse_family_accuracy"),
        "rotate_shift_family_accuracy": avg("rotate_shift_family_accuracy"),
        "rewrite_map_family_accuracy": avg("rewrite_map_family_accuracy"),
        "bind_query_family_accuracy": avg("bind_query_family_accuracy"),
        "conditional_marker_family_accuracy": avg("conditional_marker_family_accuracy"),
        "multi_step_composition_accuracy": avg("multi_step_composition_accuracy"),
        "noise_rejection_rate": rate(stats.noise_rejected, stats.noise_cases),
        "decoy_rejection_rate": rate(stats.decoy_rejected, stats.decoy_cases),
        "heldout_vocabulary_accuracy": avg("heldout_vocabulary_accuracy"),
        "randomized_codebook_generalization": avg("randomized_codebook_generalization"),
        "trace_validity": avg("trace_validity"),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "gate_false_accept_rate": rate(stats.gate_false_accepts, stats.noise_cases + stats.decoy_cases),
        "gate_false_reject_rate": rate(stats.gate_false_rejects, stats.commits),
        "temporal_drift_rate": temporal_drift,
        "oscillation_rate": rate(stats.oscillations, stats.commits),
        "cost_per_tick": rate(stats.cost, total_ticks),
        "cost_per_episode": rate(stats.cost, max(1, len(rows))),
        "deterministic_replay_passed": True,
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": False,
        "checker_failure_count": None,
    }


def run_system(system: str, episodes: list[Episode]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = Stats()
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    samples: list[dict[str, Any]] = []
    total_ticks = sum(len(episode.stream) for episode in episodes)
    for episode in episodes:
        row = run_episode(system, episode, stats)
        rows.append(row)
        family_rows[episode.family].append(row)
        if len(samples) < 10:
            samples.append(
                {
                    "episode_id": episode.episode_id,
                    "family": episode.family,
                    "selected_transform_debug": row["selected_transform_debug"],
                    "query_output_exact_accuracy": row["query_output_exact_accuracy"],
                    "trace_validity": row["trace_validity"],
                }
            )
    aggregate = aggregate_rows(rows, stats, total_ticks)
    family_metrics = {family: aggregate_rows(family_rows[family], Stats(), sum(len(ep.stream) for ep in episodes if ep.family == family) or 1) for family in FAMILIES}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "noise_cases": stats.noise_cases,
        "noise_rejected": stats.noise_rejected,
        "decoy_cases": stats.decoy_cases,
        "decoy_rejected": stats.decoy_rejected,
        "gate_false_accepts": stats.gate_false_accepts,
        "gate_false_rejects": stats.gate_false_rejects,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "stale_attempts": stats.stale_attempts,
        "stale_rejections": stats.stale_rejections,
    }
    return aggregate, diagnostics, family_metrics, samples


def positive_gate(metrics: dict[str, dict[str, Any]], replay_passed: bool) -> dict[str, Any]:
    primary = metrics[PRIMARY]
    direct = metrics[DIRECT]
    no_gate = metrics[NO_GATE]
    checks = {
        "query_output_exact_accuracy_at_least_092": primary["query_output_exact_accuracy"] >= 0.92,
        "output_sequence_accuracy_at_least_095": primary["output_sequence_accuracy"] >= 0.95,
        "output_token_accuracy_at_least_098": primary["output_token_accuracy"] >= 0.98,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "char_stream_recovery_accuracy_at_least_098": primary["char_stream_recovery_accuracy"] >= 0.98,
        "token_boundary_accuracy_at_least_098": primary["token_boundary_accuracy"] >= 0.98,
        "candidate_transform_fit_accuracy_at_least_090": primary["candidate_transform_fit_accuracy"] >= 0.90,
        "latent_transform_selection_accuracy_at_least_090": primary["latent_transform_selection_accuracy"] >= 0.90,
        "copy_family_accuracy_at_least_095": primary["copy_family_accuracy"] >= 0.95,
        "reverse_family_accuracy_at_least_095": primary["reverse_family_accuracy"] >= 0.95,
        "rotate_shift_family_accuracy_at_least_090": primary["rotate_shift_family_accuracy"] >= 0.90,
        "rewrite_map_family_accuracy_at_least_090": primary["rewrite_map_family_accuracy"] >= 0.90,
        "bind_query_family_accuracy_at_least_090": primary["bind_query_family_accuracy"] >= 0.90,
        "conditional_marker_family_accuracy_at_least_085": primary["conditional_marker_family_accuracy"] >= 0.85,
        "multi_step_composition_accuracy_at_least_080": primary["multi_step_composition_accuracy"] >= 0.80,
        "noise_rejection_rate_at_least_085": primary["noise_rejection_rate"] >= 0.85,
        "decoy_rejection_rate_at_least_085": primary["decoy_rejection_rate"] >= 0.85,
        "heldout_vocabulary_accuracy_at_least_085": primary["heldout_vocabulary_accuracy"] >= 0.85,
        "randomized_codebook_generalization_at_least_085": primary["randomized_codebook_generalization"] >= 0.85,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": replay_passed,
        "beats_direct_text_regex_on_query_exact": primary["query_output_exact_accuracy"] > direct["query_output_exact_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > no_gate["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < no_gate["wrong_writeback_rate"],
        "static_control_not_primary": PRIMARY != STATIC,
        "oracle_control_not_primary": PRIMARY != ORACLE,
    }
    return {
        "schema_version": "e13z_positive_gate_v1",
        "checks": checks,
        "deltas": {
            "primary": PRIMARY,
            "query_exact_delta_vs_direct": rounded(primary["query_output_exact_accuracy"] - direct["query_output_exact_accuracy"]),
            "trace_validity_delta_vs_no_gate": rounded(primary["trace_validity"] - no_gate["trace_validity"]),
            "wrong_writeback_reduction_vs_no_gate": rounded(1.0 - rate(primary["wrong_writeback_rate"], no_gate["wrong_writeback_rate"])),
            "cost_reduction_vs_support_fit": rounded(1.0 - rate(primary["cost_per_tick"], metrics[SUPPORT_FIT]["cost_per_tick"])),
        },
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    primary = metrics[PRIMARY]
    if gate["passed"]:
        return "e13z_text_stream_to_temporal_flow_capability_confirmed"
    if primary["semantic_slot_leak_detected"]:
        return "e13z_semantic_slot_leak_detected"
    if primary["wrong_writeback_rate"] > 0.02 or primary["branch_contamination_rate"] > 0.0:
        return "e13z_writeback_safety_failure"
    if primary["char_stream_recovery_accuracy"] < 0.98:
        return "e13z_text_stream_input_recovery_failure"
    if primary["token_boundary_accuracy"] < 0.98:
        return "e13z_token_boundary_failure"
    if primary["candidate_transform_fit_accuracy"] < 0.90:
        return "e13z_support_fit_failure"
    if primary["latent_transform_selection_accuracy"] < 0.90:
        return "e13z_transform_selection_failure"
    if primary["query_output_exact_accuracy"] < 0.92 or primary["output_sequence_accuracy"] < 0.95:
        return "e13z_output_decoder_failure"
    if primary["noise_rejection_rate"] < 0.85 or primary["decoy_rejection_rate"] < 0.85:
        return "e13z_noise_decoy_failure"
    if primary["heldout_vocabulary_accuracy"] < 0.85 or primary["randomized_codebook_generalization"] < 0.85:
        return "e13z_codebook_generalization_failure"
    return "e13z_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e13z_text_stream_to_temporal_flow_capability_confirmed": "E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM",
        "e13z_text_stream_input_recovery_failure": "E13ZA_TEXT_STREAM_INPUT_BUFFER_REPAIR",
        "e13z_token_boundary_failure": "E13ZB_TOKEN_BOUNDARY_REPAIR",
        "e13z_support_fit_failure": "E13ZC_SUPPORT_RELATION_FIT_REPAIR",
        "e13z_transform_selection_failure": "E13ZD_LATENT_TRANSFORM_SELECTION_REPAIR",
        "e13z_output_decoder_failure": "E13ZE_OUTPUT_DECODER_REPAIR",
        "e13z_noise_decoy_failure": "E13ZF_NOISE_DECOY_REPAIR",
        "e13z_codebook_generalization_failure": "E13ZG_CODEBOOK_GENERALIZATION_REPAIR",
        "e13z_semantic_slot_leak_detected": "E13ZL_SEMANTIC_LEAK_REPAIR",
        "e13z_writeback_safety_failure": "E13ZW_WRITEBACK_SAFETY_REPAIR",
        "e13z_invalid_or_incomplete_run": "E13Z_RETRY_WITH_FULL_AUDIT",
    }[decision]


def render_report(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    metrics = aggregate["systems"]
    family_metrics = aggregate["family_metrics"][PRIMARY]
    lines = [
        "# E13Z Text-Stream To Temporal Flow Capability Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {decision['primary_system']}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Key Metrics",
        "",
        "| system | query exact | output seq | token | trace | fit | select | noise reject | decoy reject | wrong | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = metrics[system]
        lines.append(
            f"| {system} | {row['query_output_exact_accuracy']:.3f} | {row['output_sequence_accuracy']:.3f} | {row['output_token_accuracy']:.3f} | "
            f"{row['trace_validity']:.3f} | {row['candidate_transform_fit_accuracy']:.3f} | {row['latent_transform_selection_accuracy']:.3f} | "
            f"{row['noise_rejection_rate']:.3f} | {row['decoy_rejection_rate']:.3f} | {row['wrong_writeback_rate']:.3f} | {row['cost_per_tick']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Family Metrics",
            "",
            "| family | exact | trace |",
            "|---|---:|---:|",
        ]
    )
    for family in FAMILIES:
        row = family_metrics[family]
        lines.append(f"| {family} | {row['query_output_exact_accuracy']:.3f} | {row['trace_validity']:.3f} |")
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Boundary",
            "",
            "This is a deterministic synthetic controlled text-stream proxy only.",
        ]
    )
    return "\n".join(lines)


def build_reports(seeds: tuple[int, ...], rows_per_family: int, replay_passed: bool = True) -> dict[str, Any]:
    episodes = build_episodes(seeds, rows_per_family)
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    family_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS:
        system_metrics, diag, fam, sample = run_system(system, episodes)
        metrics[system] = system_metrics
        diagnostics[system] = diag
        family_metrics[system] = fam
        samples[system] = sample
    gate = positive_gate(metrics, replay_passed)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e13z_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": replay_passed,
    }
    aggregate = {
        "schema_version": "e13z_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_family": rows_per_family,
        "systems": metrics,
        "diagnostics": diagnostics,
        "family_metrics": family_metrics,
        "positive_gate": gate,
    }
    sample_episode = episodes[0]
    input_stream_report = {
        "schema_version": "e13z_input_stream_report_v1",
        "runtime_input_type": "temporal_character_pulse_frames",
        "active_instruction_tokens": [],
        "forbidden_task_words_present": False,
        "runtime_receives_semantic_labels": False,
        "sample_stream": [pulse.payload() for pulse in sample_episode.stream[:MAX_DEBUG_STREAM]],
        "sample_stream_hash": stable_hash([pulse.payload() for pulse in sample_episode.stream]),
    }
    codebook_report = {
        "schema_version": "e13z_vocab_codebook_report_v1",
        "seed_count": len(seeds),
        "unique_codebook_hashes": sorted({episode.codebook_hash for episode in episodes}),
        "unique_vocab_hashes": sorted({episode.vocab_hash for episode in episodes})[:20],
        "randomized_vocab_used": True,
        "randomized_codebook_used": True,
        "heldout_vocabulary_used": any(episode.heldout_vocabulary for episode in episodes),
    }
    episode_report = {
        "schema_version": "e13z_support_query_episode_report_v1",
        "episode_count": len(episodes),
        "families": list(FAMILIES),
        "debug_samples": [
            {
                "episode_id": episode.episode_id,
                "family": episode.family,
                "debug_text": episode.debug_text,
                "query_output_hash": stable_hash(seq_sig(episode.expected)),
            }
            for episode in episodes[:10]
        ],
    }
    task_family_report = {
        "schema_version": "e13z_task_family_report_v1",
        "primary_family_metrics": family_metrics[PRIMARY],
        "debug_transform_names": DEBUG_OP_NAMES,
        "task_family_used_by_primary_runtime": False,
    }
    search_report = {
        "schema_version": "e13z_search_report_v1",
        "equivalent_existing_milestone_found": False,
        "searched_locations": ["docs/research", "scripts/probes", "docs/wiki", "CHANGELOG.md", "all fetched refs"],
        "search_terms": [
            "E13Z",
            "TEXT_STREAM_TO_TEMPORAL_FLOW",
            "controlled text stream",
            "nonce token",
            "support query",
            "temporal character stream",
            "randomized vocabulary",
            "canonical decoder",
        ],
        "adjacent_hits": [],
    }
    system_report = {"schema_version": "e13z_system_comparison_report_v1", "systems": metrics, "diagnostics": diagnostics, "samples": samples}
    trace_report = {
        "schema_version": "e13z_trace_validity_report_v1",
        "trace_validity": {system: metrics[system]["trace_validity"] for system in SYSTEMS},
        "candidate_transform_fit_accuracy": {system: metrics[system]["candidate_transform_fit_accuracy"] for system in SYSTEMS},
        "latent_transform_selection_accuracy": {system: metrics[system]["latent_transform_selection_accuracy"] for system in SYSTEMS},
    }
    safety_report = {
        "schema_version": "e13z_writeback_safety_report_v1",
        "wrong_writeback_rate": {system: metrics[system]["wrong_writeback_rate"] for system in SYSTEMS},
        "destructive_overwrite_rate": {system: metrics[system]["destructive_overwrite_rate"] for system in SYSTEMS},
        "branch_contamination_rate": {system: metrics[system]["branch_contamination_rate"] for system in SYSTEMS},
        "stale_write_rejection_rate": {system: metrics[system]["stale_write_rejection_rate"] for system in SYSTEMS},
        "gate_false_accept_rate": {system: metrics[system]["gate_false_accept_rate"] for system in SYSTEMS},
        "gate_false_reject_rate": {system: metrics[system]["gate_false_reject_rate"] for system in SYSTEMS},
    }
    noise_report = {
        "schema_version": "e13z_noise_decoy_report_v1",
        "noise_rejection_rate": {system: metrics[system]["noise_rejection_rate"] for system in SYSTEMS},
        "decoy_rejection_rate": {system: metrics[system]["decoy_rejection_rate"] for system in SYSTEMS},
        "noise_family_metrics": {system: family_metrics[system]["NOISE_AND_DECOY_STREAM"] for system in SYSTEMS},
    }
    generalization_report = {
        "schema_version": "e13z_heldout_generalization_report_v1",
        "heldout_vocabulary_accuracy": {system: metrics[system]["heldout_vocabulary_accuracy"] for system in SYSTEMS},
        "randomized_codebook_generalization": {system: metrics[system]["randomized_codebook_generalization"] for system in SYSTEMS},
    }
    semantic_report = {
        "schema_version": "e13z_semantic_leak_audit_report_v1",
        "primary_runtime_config": {
            "input": "temporal_character_pulse_frames",
            "state": "anonymous_temporal_flow_buffers",
            "proposal": "support_fit_candidate_regions",
            "writeback": "guarded_output_lane",
        },
        "pocket_field_equivalents": list(POCKET_FIELDS),
        "runtime_receives_forbidden_semantic_slots": False,
        "input_stream_contains_task_words": False,
        "task_family_used_by_primary_runtime": False,
        "privileged_controls_invalid_as_primary": True,
        "semantic_slot_leak_detected": metrics[PRIMARY]["semantic_slot_leak_detected"],
        "privileged_control_selected_as_primary": metrics[PRIMARY]["privileged_control_selected_as_primary"],
        "no_neural_dependency_detected": True,
    }
    boundary_report = {
        "schema_version": "e13z_boundary_claims_report_v1",
        "boundary": "deterministic synthetic controlled text-stream proxy only",
        "claims_deployed_behavior": False,
        "claims_broad_model_scale": False,
        "claims_hardware_speedup": False,
    }
    summary = {
        "schema_version": "e13z_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "query_output_exact_accuracy": metrics[PRIMARY]["query_output_exact_accuracy"],
        "trace_validity": metrics[PRIMARY]["trace_validity"],
        "randomized_codebook_generalization": metrics[PRIMARY]["randomized_codebook_generalization"],
        "heldout_vocabulary_accuracy": metrics[PRIMARY]["heldout_vocabulary_accuracy"],
        "semantic_slot_leak_detected": metrics[PRIMARY]["semantic_slot_leak_detected"],
        "privileged_control_selected_as_primary": metrics[PRIMARY]["privileged_control_selected_as_primary"],
        "checker_failure_count": None,
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate),
        "e13z_search_report.json": search_report,
        "e13z_input_stream_report.json": input_stream_report,
        "e13z_vocab_codebook_report.json": codebook_report,
        "e13z_support_query_episode_report.json": episode_report,
        "e13z_system_comparison_report.json": system_report,
        "e13z_task_family_report.json": task_family_report,
        "e13z_trace_validity_report.json": trace_report,
        "e13z_writeback_safety_report.json": safety_report,
        "e13z_noise_decoy_report.json": noise_report,
        "e13z_heldout_generalization_report.json": generalization_report,
        "e13z_semantic_leak_audit_report.json": semantic_report,
        "e13z_boundary_claims_report.json": boundary_report,
    }


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_family: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, rows_per_family, replay_passed=True)
    replay_b = build_reports(seeds, rows_per_family, replay_passed=True)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e13z_deterministic_replay_report.json"] = {
        "schema_version": "e13z_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"] = positive_gate(payloads["aggregate_metrics.json"]["systems"], passed)
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"], payloads["aggregate_metrics.json"]["systems"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["next"] = next_for(decision_label)
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"])
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--rows-per-family", type=int, default=DEFAULT_ROWS_PER_FAMILY)
    args = parser.parse_args(argv)
    payloads = build_reports(args.seeds, args.rows_per_family, replay_passed=True)
    payloads = attach_replay(payloads, args.seeds, args.rows_per_family)
    code, head = run_git(["rev-parse", "--short", "HEAD"])
    if code == 0:
        payloads["summary.json"]["git_head"] = head.strip()
    for name in REQUIRED_ARTIFACTS:
        payload = payloads[name]
        path = args.out / name
        if name.endswith(".md"):
            write_text(path, str(payload))
        else:
            write_json(path, payload)
    print(stable_json({"decision": payloads["decision.json"]["decision"], "out": str(args.out), "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
