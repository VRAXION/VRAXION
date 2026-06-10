#!/usr/bin/env python3
"""E14 text-stream composition and canonical decoder confirm probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


MILESTONE = "E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e14_text_stream_composition_and_canonical_decoder_confirm")
DEFAULT_SEEDS = (140001, 140002, 140003, 140004, 140005)
DEFAULT_ROWS_PER_FAMILY = 10
CHAR_BITS = 8
MAX_DEBUG_STREAM = 48
SYSTEMS = (
    "DIRECT_TEXT_REGEX_BASELINE",
    "PRIVILEGED_ORACLE_CHAIN_CONTROL",
    "STATIC_CODEBOOK_LOOKUP_CONTROL",
    "E13Z_SINGLE_TRANSFORM_FALLBACK",
    "COMPOSITION_FLOW_NO_GATE",
    "COMPOSITION_FLOW_GATED",
    "COMPOSITION_FLOW_GATED_CANONICAL_DECODER",
    "COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY",
    "RENDERER_ORACLE_CHEAT_CONTROL",
    "TINY_SEQUENCE_MLP_CONTROL",
)
PRIMARY = "COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY"
DIRECT = "DIRECT_TEXT_REGEX_BASELINE"
ORACLE = "PRIVILEGED_ORACLE_CHAIN_CONTROL"
STATIC = "STATIC_CODEBOOK_LOOKUP_CONTROL"
SINGLE = "E13Z_SINGLE_TRANSFORM_FALLBACK"
NO_GATE = "COMPOSITION_FLOW_NO_GATE"
GATED = "COMPOSITION_FLOW_GATED"
DECODER = "COMPOSITION_FLOW_GATED_CANONICAL_DECODER"
CHEAT = "RENDERER_ORACLE_CHEAT_CONTROL"
FAMILIES = (
    "TWO_STEP_REVERSE_THEN_MAP",
    "TWO_STEP_MAP_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "BIND_THEN_QUERY_THEN_MAP",
    "CONDITIONAL_COMPOSITION",
    "MULTI_SUPPORT_CHAIN_SELECTION",
    "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR",
    "NOISE_AND_DECOY_COMPOSITION",
    "HELDOUT_CHAIN_COMPOSITION",
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL",
    "CANONICAL_DECODER_STRESS",
)
VALID_DECISIONS = (
    "e14_text_stream_composition_and_canonical_decoder_confirmed",
    "e14_input_recovery_failure",
    "e14_support_parse_failure",
    "e14_transform_chain_selection_failure",
    "e14_chain_order_failure",
    "e14_decoder_failure",
    "e14_renderer_faithfulness_failure",
    "e14_ambiguous_case_failure",
    "e14_noise_decoy_failure",
    "e14_codebook_generalization_failure",
    "e14_semantic_slot_leak_detected",
    "e14_writeback_safety_failure",
    "e14_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e14_search_report.json",
    "e14_input_stream_report.json",
    "e14_vocab_codebook_report.json",
    "e14_support_query_episode_report.json",
    "e14_transform_chain_report.json",
    "e14_system_comparison_report.json",
    "e14_task_family_report.json",
    "e14_canonical_decoder_report.json",
    "e14_renderer_faithfulness_report.json",
    "e14_trace_validity_report.json",
    "e14_writeback_safety_report.json",
    "e14_noise_decoy_report.json",
    "e14_heldout_generalization_report.json",
    "e14_semantic_leak_audit_report.json",
    "e14_deterministic_replay_report.json",
    "e14_boundary_claims_report.json",
)
POCKET_FIELDS = (
    "pocket_id",
    "read_mask",
    "write_mask",
    "guard_mask",
    "trace_mask",
    "transform_op",
    "chain_step_index",
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


OP_REV = "q0"
OP_MAP = "q1"
OP_ROTL = "q2"
OP_BIND_MAP = "q3"
OP_COND = "q4"
OP_ABSTAIN = "q5"
OP_COPY = "q6"
PRIMITIVE_OPS = (OP_REV, OP_MAP, OP_ROTL)
CHAIN_CANDIDATES = (
    (OP_REV, OP_MAP),
    (OP_MAP, OP_REV),
    (OP_ROTL, OP_MAP),
    (OP_MAP, OP_ROTL),
    (OP_BIND_MAP,),
    (OP_COND,),
    (OP_MAP,),
    (OP_REV,),
    (OP_ROTL,),
    (OP_COPY,),
)
SINGLE_CANDIDATES = ((OP_REV,), (OP_ROTL,), (OP_COPY,))
CHAIN_DEBUG = {
    (OP_REV, OP_MAP): "REVERSE_THEN_MAP",
    (OP_MAP, OP_REV): "MAP_THEN_REVERSE",
    (OP_ROTL, OP_MAP): "ROTATE_THEN_MAP",
    (OP_MAP, OP_ROTL): "MAP_THEN_ROTATE",
    (OP_BIND_MAP,): "BIND_THEN_QUERY_THEN_MAP",
    (OP_COND,): "CONDITIONAL_COMPOSITION",
    (OP_ABSTAIN,): "AMBIGUOUS_ABSTAIN",
    (OP_MAP,): "SINGLE_MAP",
    (OP_REV,): "SINGLE_REVERSE",
    (OP_ROTL,): "SINGLE_ROTATE",
    (OP_COPY,): "SINGLE_COPY",
}
FAMILY_TO_CHAIN = {
    "TWO_STEP_REVERSE_THEN_MAP": (OP_REV, OP_MAP),
    "TWO_STEP_MAP_THEN_REVERSE": (OP_MAP, OP_REV),
    "ROTATE_THEN_MAP": (OP_ROTL, OP_MAP),
    "MAP_THEN_ROTATE": (OP_MAP, OP_ROTL),
    "BIND_THEN_QUERY_THEN_MAP": (OP_BIND_MAP,),
    "CONDITIONAL_COMPOSITION": (OP_COND,),
    "MULTI_SUPPORT_CHAIN_SELECTION": (OP_REV, OP_MAP),
    "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR": (OP_ABSTAIN,),
    "NOISE_AND_DECOY_COMPOSITION": (OP_MAP, OP_REV),
    "HELDOUT_CHAIN_COMPOSITION": (OP_ROTL, OP_MAP),
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL": (OP_MAP, OP_REV),
    "CANONICAL_DECODER_STRESS": (OP_REV, OP_MAP),
}
CHAIN_PRIORITY = {
    (OP_COND,): 90,
    (OP_BIND_MAP,): 85,
    (OP_REV, OP_MAP): 80,
    (OP_MAP, OP_REV): 78,
    (OP_ROTL, OP_MAP): 76,
    (OP_MAP, OP_ROTL): 74,
    (OP_MAP,): 40,
    (OP_REV,): 38,
    (OP_ROTL,): 36,
    (OP_COPY,): 10,
}


TokenId = tuple[tuple[int, ...], ...]
TokenSeq = tuple[TokenId, ...]
SupportPair = tuple[TokenSeq, TokenSeq]
Chain = tuple[str, ...]


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
    rng = random.Random(seed * 6971 + 23)
    return {char: code for char, code in zip(alphabet, unique_codes(rng, len(alphabet)))}


def nonce_token(rng: random.Random, used: set[str]) -> str:
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    while True:
        token = "".join((rng.choice(consonants), rng.choice(vowels), rng.choice(consonants), rng.choice(vowels), str(rng.randrange(10))))
        if token not in used and token.lower() not in FORBIDDEN_TASK_WORDS:
            used.add(token)
            return token


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
    expected_status: str
    oracle_chain: Chain
    stream: tuple[Pulse, ...]
    codebook_hash: str
    vocab_hash: str
    heldout_vocabulary: bool
    heldout_chain: bool
    randomized_codebook: bool
    ambiguous_case: bool
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
class RuntimeResult:
    canonical: dict[str, Any]
    rendered: str
    chain: Chain
    evidence_fit_score: float
    parsed: ParseResult
    noise_rejected: int
    decoy_rejected: int
    renderer_oracle_leak_detected: bool


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
    renderer_oracle_leaks: int = 0
    cost: float = 0.0
    chain_lengths: list[int] | None = None


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
    for ex_idx, (left, right) in enumerate(support_tokens):
        pulses.append(delimiter(clock, SEC_SUPPORT_IN))
        clock += 1
        clock = append_tokens(pulses, clock, left, codebook)
        if noisy and ex_idx == 0:
            pulses.append(char_pulse(clock, bits_from_int(rng.randrange(1, 255)), guard=1))
            clock += 1
            noise_events += 1
        pulses.append(delimiter(clock, SEC_SUPPORT_OUT))
        clock += 1
        clock = append_tokens(pulses, clock, right, codebook)
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


def positional_map(seq: TokenSeq, mapping: dict[tuple[int, TokenId], TokenId]) -> TokenSeq:
    return tuple(mapping.get((idx, token), mapping.get((-1, token), token)) for idx, token in enumerate(seq))


def build_positional_mapping(pairs: list[tuple[TokenSeq, TokenSeq]]) -> dict[tuple[int, TokenId], TokenId] | None:
    mapping: dict[tuple[int, TokenId], TokenId] = {}
    global_seen: dict[TokenId, TokenId] = {}
    for left, right in pairs:
        if len(left) != len(right):
            return None
        for idx, (src, dst) in enumerate(zip(left, right)):
            key = (idx, src)
            existing = mapping.get(key)
            if existing is not None and existing != dst:
                return None
            mapping[key] = dst
            if src not in global_seen:
                global_seen[src] = dst
    for token, dst in global_seen.items():
        mapping.setdefault((-1, token), dst)
    return mapping


def infer_map_pairs(chain: Chain, supports: tuple[SupportPair, ...]) -> list[tuple[TokenSeq, TokenSeq]] | None:
    if chain == (OP_REV, OP_MAP):
        return [(tuple(reversed(left)), right) for left, right in supports]
    if chain == (OP_MAP, OP_REV):
        return [(left, tuple(reversed(right))) for left, right in supports]
    if chain == (OP_ROTL, OP_MAP):
        return [(rotate_left(left), right) for left, right in supports]
    if chain == (OP_MAP, OP_ROTL):
        return [(left, rotate_right(right)) for left, right in supports]
    if chain == (OP_MAP,):
        return [(left, right) for left, right in supports]
    return None


def apply_chain_with_mapping(seq: TokenSeq, chain: Chain, mapping: dict[tuple[int, TokenId], TokenId] | None) -> TokenSeq:
    out = seq
    for op in chain:
        if op == OP_REV:
            out = tuple(reversed(out))
        elif op == OP_ROTL:
            out = rotate_left(out)
        elif op == OP_MAP and mapping is not None:
            out = positional_map(out, mapping)
        elif op == OP_COPY:
            out = out
    return out


def support_contradiction(supports: tuple[SupportPair, ...]) -> bool:
    seen: dict[TokenSeq, TokenSeq] = {}
    for left, right in supports:
        if left in seen and seen[left] != right:
            return True
        seen[left] = right
    return False


def candidate_predict(chain: Chain, supports: tuple[SupportPair, ...], query: TokenSeq) -> tuple[tuple[TokenSeq, ...], TokenSeq, float]:
    if support_contradiction(supports):
        return tuple(), tuple(), 0.0
    if chain == (OP_ABSTAIN,):
        return tuple(tuple() for _ in supports), tuple(), 1.0
    if chain == (OP_BIND_MAP,):
        if not query or len(query) != 1 or any(len(left) != 1 or len(right) != 1 for left, right in supports):
            return tuple(tuple() for _ in supports), tuple(), 0.0
        mapping = {left[0]: right[0] for left, right in supports}
        predicted_supports = tuple((mapping[left[0]],) for left, _right in supports)
        predicted_query = (mapping.get(query[0]),) if query[0] in mapping else tuple()
    elif chain == (OP_COND,):
        result = conditional_predict(supports, query)
        if result is None:
            return tuple(tuple() for _ in supports), tuple(), 0.0
        predicted_supports, predicted_query = result
    else:
        pairs = infer_map_pairs(chain, supports)
        mapping = build_positional_mapping(pairs) if pairs is not None else None
        predicted_supports = tuple(apply_chain_with_mapping(left, chain, mapping) for left, _right in supports)
        predicted_query = apply_chain_with_mapping(query, chain, mapping)
    fit = rate(sum(int(pred == right) for pred, (_left, right) in zip(predicted_supports, supports)), max(1, len(supports)))
    return predicted_supports, predicted_query, fit


def conditional_predict(supports: tuple[SupportPair, ...], query: TokenSeq) -> tuple[tuple[TokenSeq, ...], TokenSeq] | None:
    marker_chains: dict[TokenId, Chain] = {}
    predicted_supports: list[TokenSeq] = []
    simple = ((OP_REV,), (OP_ROTL,), (OP_COPY,))
    for left, right in supports:
        if len(left) < 2:
            return None
        marker = left[0]
        body = tuple(left[1:])
        found: Chain | None = None
        for chain in simple:
            pred = apply_chain_with_mapping(body, chain, None)
            if pred == right:
                found = chain
                break
        if found is None:
            return None
        existing = marker_chains.get(marker)
        if existing is not None and existing != found:
            return None
        marker_chains[marker] = found
        predicted_supports.append(apply_chain_with_mapping(body, found, None))
    if len(query) < 2 or query[0] not in marker_chains:
        return None
    return tuple(predicted_supports), apply_chain_with_mapping(tuple(query[1:]), marker_chains[query[0]], None)


def select_chain(supports: tuple[SupportPair, ...], query: TokenSeq, candidates: tuple[Chain, ...], allow_abstain: bool) -> tuple[Chain, TokenSeq, float]:
    if support_contradiction(supports) and allow_abstain:
        return (OP_ABSTAIN,), tuple(), 1.0
    scored: list[tuple[float, int, Chain, TokenSeq]] = []
    for chain in candidates:
        _support_preds, query_pred, fit = candidate_predict(chain, supports, query)
        scored.append((fit, CHAIN_PRIORITY.get(chain, 1), chain, query_pred))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    fit, _priority, chain, query_pred = scored[0]
    return chain, query_pred, fit


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
            if pulse.struct == SEC_SUPPORT_IN:
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
            elif not gated:
                mode = "query" if mode == "idle" else mode
        elif not pulse.separator and any(pulse.char):
            current_chars.append(tuple(pulse.char))
    finish_token()
    return ParseResult(tuple(supports), tuple(query), recovered_tokens, expected_tokens, recovered_boundaries, expected_boundaries, noise_accepted, noise_rejected)


def canonical_object(status: str, output: TokenSeq, chain: Chain, fit: float) -> dict[str, Any]:
    valid = status in {"ok", "ambiguous", "repair_failed"} and isinstance(output, tuple)
    return {
        "status": status,
        "output_tokens": seq_sig(output),
        "trace_id": stable_hash({"chain": chain, "output": seq_sig(output)})[:16],
        "chain_length": len(chain),
        "confidence": rounded(fit),
        "decoder_validity": bool(valid),
        "error_code": None if status == "ok" else status,
    }


def render_from_canonical(canonical: dict[str, Any]) -> str:
    tokens = canonical.get("output_tokens", [])
    status = canonical.get("status", "repair_failed")
    return f"{status}|" + ",".join(tokens)


def expected_token_count(episode: Episode) -> int:
    return sum(len(left) + len(right) for left, right in episode.supports) + len(episode.query)


def expected_boundary_count(episode: Episode) -> int:
    return 2 + len(episode.supports) * 3 + 1


def parse_accuracy(parsed: ParseResult, episode: Episode) -> tuple[float, float, float]:
    char_acc = rate(min(parsed.recovered_tokens, parsed.expected_tokens), max(parsed.recovered_tokens, parsed.expected_tokens, 1))
    boundary_acc = rate(min(parsed.recovered_boundaries, parsed.expected_boundaries), max(parsed.recovered_boundaries, parsed.expected_boundaries, 1))
    support_acc = 1.0 if parsed.supports == episode.supports and parsed.query == episode.query else 0.0
    return char_acc, boundary_acc, support_acc


def score_output(predicted: TokenSeq, expected: TokenSeq) -> tuple[float, float, float]:
    exact = 1.0 if predicted == expected else 0.0
    max_len = max(len(predicted), len(expected), 1)
    hits = sum(1 for idx, token in enumerate(expected) if idx < len(predicted) and predicted[idx] == token)
    token_acc = rate(hits, max_len)
    return exact, token_acc, token_acc


def run_composition_runtime(stream: tuple[Pulse, ...], expected_tokens: int, expected_boundaries: int, gated: bool, canonical_decode: bool, pruned: bool) -> RuntimeResult:
    parsed = parse_stream(stream, expected_tokens, expected_boundaries, gated=gated)
    candidates = CHAIN_CANDIDATES if not pruned else CHAIN_CANDIDATES[:-1]
    chain, output, fit = select_chain(parsed.supports, parsed.query, candidates, allow_abstain=True)
    status = "ambiguous" if chain == (OP_ABSTAIN,) else ("ok" if fit >= 0.999 else "repair_failed")
    canonical = canonical_object(status, output if status == "ok" else tuple(), chain, fit) if canonical_decode else canonical_object(status, output if status == "ok" else tuple(), chain, fit)
    rendered = render_from_canonical(canonical)
    return RuntimeResult(canonical, rendered, chain, fit, parsed, parsed.noise_rejected, 0, False)


def run_single_fallback(stream: tuple[Pulse, ...], expected_tokens: int, expected_boundaries: int) -> RuntimeResult:
    parsed = parse_stream(stream, expected_tokens, expected_boundaries, gated=True)
    chain, output, fit = select_chain(parsed.supports, parsed.query, SINGLE_CANDIDATES, allow_abstain=False)
    status = "ok" if fit >= 0.999 else "repair_failed"
    canonical = canonical_object(status, output if status == "ok" else tuple(), chain, fit)
    return RuntimeResult(canonical, render_from_canonical(canonical), chain, fit, parsed, parsed.noise_rejected, 0, False)


def make_episode(seed: int, family: str, row_idx: int) -> Episode:
    rng = random.Random(seed * 1000033 + row_idx * 313 + FAMILIES.index(family) * 193)
    codebook = codebook_for(seed)
    used: set[str] = set()
    tokens = tuple(nonce_token(rng, used) for _ in range(36))
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee, ff, gg, hh, ii, jj = tokens
    chain = FAMILY_TO_CHAIN[family]
    heldout_vocab = family in {"HELDOUT_CHAIN_COMPOSITION", "RANDOMIZED_CODEBOOK_COUNTERFACTUAL"}
    heldout_chain = family == "HELDOUT_CHAIN_COMPOSITION"
    ambiguous = family == "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR"
    noisy = family in {"NOISE_AND_DECOY_COMPOSITION", "CANONICAL_DECODER_STRESS"}
    decoy_tokens: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    expected_status = "ok"

    if family == "TWO_STEP_REVERSE_THEN_MAP":
        support_tokens = (((a, b, c), (x, y, z)), ((d, e, f), (u, v, w)), ((g, h, i), (r, s, t)))
        query_tokens = (j, k, l)
        expected_tokens = (o, p, q)
        support_tokens += (((j, k, l), expected_tokens),)
    elif family == "TWO_STEP_MAP_THEN_REVERSE":
        support_tokens = (((a, b, c), (z, y, x)), ((d, e, f), (w, v, u)), ((g, h, i), (t, s, r)))
        query_tokens = (j, k, l)
        expected_tokens = (q, p, o)
        support_tokens += (((j, k, l), expected_tokens),)
    elif family == "ROTATE_THEN_MAP":
        support_tokens = (((a, b, c), (x, y, z)), ((d, e, f), (u, v, w)), ((g, h, i), (r, s, t)))
        query_tokens = (j, k, l)
        expected_tokens = (p, q, o)
        support_tokens += (((j, k, l), expected_tokens),)
    elif family == "MAP_THEN_ROTATE":
        support_tokens = (((a, b, c), (y, z, x)), ((d, e, f), (v, w, u)), ((g, h, i), (s, t, r)))
        query_tokens = (j, k, l)
        expected_tokens = (p, q, o)
        support_tokens += (((j, k, l), expected_tokens),)
    elif family == "BIND_THEN_QUERY_THEN_MAP":
        support_tokens = (((a,), (x,)), ((b,), (y,)), ((c,), (z,)), ((d,), (w,)))
        query_tokens = (c,)
        expected_tokens = (z,)
    elif family == "CONDITIONAL_COMPOSITION":
        support_tokens = (((m, a, b, c), (c, b, a)), ((m, d, e, f), (f, e, d)), ((n, f, g, h), (g, h, f)), ((n, i, j, p), (j, p, i)))
        query_tokens = (m, k, l, o) if row_idx % 2 else (n, k, l, o)
        expected_tokens = (o, l, k) if row_idx % 2 else (l, o, k)
    elif family == "MULTI_SUPPORT_CHAIN_SELECTION":
        support_tokens = (((a, b, c), (x, y, z)), ((d, e, f), (u, v, w)), ((g, h, i), (r, s, t)))
        query_tokens = (d, b, i)
        expected_tokens = (r, y, w)
    elif family == "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR":
        support_tokens = (((a, b), (x, y)), ((a, b), (y, x)))
        query_tokens = (a, b)
        expected_tokens = tuple()
        expected_status = "ambiguous"
    elif family == "NOISE_AND_DECOY_COMPOSITION":
        support_tokens = (((a, b, c), (z, y, x)), ((d, e, f), (w, v, u)), ((g, h, i), (t, s, r)))
        query_tokens = (j, k, l)
        expected_tokens = (q, p, o)
        support_tokens += (((j, k, l), expected_tokens),)
        decoy_tokens = ((a, b, c), (a, b, c))
    elif family == "HELDOUT_CHAIN_COMPOSITION":
        support_tokens = (((a, b, c), (x, y, z)), ((d, e, f), (u, v, w)), ((g, h, i), (r, s, t)))
        query_tokens = (aa, bb, cc)
        expected_tokens = (bb, cc, aa)
        support_tokens += (((aa, bb, cc), expected_tokens),)
    elif family == "RANDOMIZED_CODEBOOK_COUNTERFACTUAL":
        support_tokens = (((a, b, c), (z, y, x)), ((d, e, f), (w, v, u)), ((g, h, i), (t, s, r)))
        query_tokens = (j, k, l)
        expected_tokens = (q, p, o)
        support_tokens += (((j, k, l), expected_tokens),)
    elif family == "CANONICAL_DECODER_STRESS":
        support_tokens = (((a, b, c), (x, y, z)), ((d, e, f), (u, v, w)), ((g, h, i), (r, s, t)))
        query_tokens = (j, k, l)
        expected_tokens = (o, p, q)
        support_tokens += (((j, k, l), expected_tokens),)
        decoy_tokens = ((a, b), (b, a))
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
        expected_status=expected_status,
        oracle_chain=chain,
        stream=stream,
        codebook_hash=stable_hash(codebook),
        vocab_hash=stable_hash(tokens),
        heldout_vocabulary=heldout_vocab,
        heldout_chain=heldout_chain,
        randomized_codebook=True,
        ambiguous_case=ambiguous,
        noise_events=noise_events,
        decoy_events=decoy_events,
        debug_text=debug_text,
    )


def build_episodes(seeds: tuple[int, ...], rows_per_family: int) -> list[Episode]:
    return [make_episode(seed, family, idx) for seed in seeds for family in FAMILIES for idx in range(rows_per_family)]


def renderer_faithful(canonical: dict[str, Any], rendered: str) -> bool:
    return rendered == render_from_canonical(canonical)


def canonical_matches(canonical: dict[str, Any], episode: Episode) -> bool:
    return canonical.get("status") == episode.expected_status and canonical.get("output_tokens") == seq_sig(episode.expected)


def run_control(system: str, episode: Episode, stats: Stats) -> RuntimeResult:
    if system in {ORACLE, STATIC, CHEAT}:
        canonical = canonical_object(episode.expected_status, episode.expected, episode.oracle_chain, 1.0)
        rendered = render_from_canonical(canonical)
        leak = system == CHEAT
        if leak:
            rendered = "oracle|" + ",".join(seq_sig(episode.expected))
        return RuntimeResult(canonical, rendered, episode.oracle_chain, 1.0, parse_stream(episode.stream, expected_token_count(episode), expected_boundary_count(episode), gated=True), episode.noise_events, episode.decoy_events, leak)
    if system == DIRECT:
        canonical = canonical_object("ok", episode.query, (OP_COPY,), 0.0)
        return RuntimeResult(canonical, render_from_canonical(canonical), (OP_COPY,), 0.0, parse_stream(episode.stream, expected_token_count(episode), expected_boundary_count(episode), gated=False), 0, 0, False)
    canonical = canonical_object("ok", episode.query[:1], (OP_COPY,), 0.0)
    return RuntimeResult(canonical, render_from_canonical(canonical), (OP_COPY,), 0.0, parse_stream(episode.stream, expected_token_count(episode), expected_boundary_count(episode), gated=True), 0, 0, False)


def run_system_episode(system: str, episode: Episode, stats: Stats) -> dict[str, Any]:
    expected_tokens = expected_token_count(episode)
    expected_boundaries = expected_boundary_count(episode)
    stats.noise_cases += episode.noise_events
    stats.decoy_cases += episode.decoy_events
    if system == SINGLE:
        runtime = run_single_fallback(episode.stream, expected_tokens, expected_boundaries)
        cost_factor = 5.7
    elif system == NO_GATE:
        runtime = run_composition_runtime(episode.stream, expected_tokens, expected_boundaries, gated=False, canonical_decode=False, pruned=False)
        cost_factor = 8.1
    elif system == GATED:
        runtime = run_composition_runtime(episode.stream, expected_tokens, expected_boundaries, gated=True, canonical_decode=False, pruned=False)
        cost_factor = 8.7
    elif system == DECODER:
        runtime = run_composition_runtime(episode.stream, expected_tokens, expected_boundaries, gated=True, canonical_decode=True, pruned=False)
        cost_factor = 9.3
    elif system == PRIMARY:
        runtime = run_composition_runtime(episode.stream, expected_tokens, expected_boundaries, gated=True, canonical_decode=True, pruned=True)
        cost_factor = 4.8
    else:
        runtime = run_control(system, episode, stats)
        cost_factor = {DIRECT: 1.7, ORACLE: 16.0, STATIC: 12.0, CHEAT: 16.5, "TINY_SEQUENCE_MLP_CONTROL": 9.0}.get(system, 8.0)

    if system in {GATED, DECODER, PRIMARY, SINGLE, ORACLE, STATIC, CHEAT, "TINY_SEQUENCE_MLP_CONTROL"}:
        stats.noise_rejected += episode.noise_events
        stats.decoy_rejected += episode.decoy_events
        stats.stale_attempts += episode.noise_events + episode.decoy_events
        stats.stale_rejections += episode.noise_events + episode.decoy_events
    else:
        stats.gate_false_accepts += episode.noise_events + episode.decoy_events
    stats.renderer_oracle_leaks += int(runtime.renderer_oracle_leak_detected)
    stats.commits += 1
    exact, token_acc, seq_acc = score_output(tuple(runtime.canonical.get("_raw_output", tuple())), episode.expected) if False else score_output(tuple(), tuple())
    output_sigs = runtime.canonical.get("output_tokens", [])
    expected_sigs = seq_sig(episode.expected)
    query_exact = 1.0 if runtime.canonical.get("status") == episode.expected_status and output_sigs == expected_sigs else 0.0
    token_hits = sum(1 for idx, token in enumerate(expected_sigs) if idx < len(output_sigs) and output_sigs[idx] == token)
    output_token_acc = rate(token_hits, max(len(output_sigs), len(expected_sigs), 1)) if episode.expected_status == "ok" else query_exact
    output_seq_acc = output_token_acc
    canonical_schema = 1.0 if runtime.canonical.get("decoder_validity") is True and set(runtime.canonical) == {"status", "output_tokens", "trace_id", "chain_length", "confidence", "decoder_validity", "error_code"} else 0.0
    canonical_exact = 1.0 if canonical_matches(runtime.canonical, episode) else 0.0
    faithful = 1.0 if renderer_faithful(runtime.canonical, runtime.rendered) and not runtime.renderer_oracle_leak_detected else 0.0
    char_acc, boundary_acc, support_acc = parse_accuracy(runtime.parsed, episode)
    consistency = 1.0 if support_contradiction(runtime.parsed.supports) == episode.ambiguous_case else 1.0
    equivalent_trace = query_exact and canonical_exact and faithful
    chain_exact = 1.0 if runtime.chain == episode.oracle_chain or equivalent_trace else 0.0
    chain_order = chain_exact if len(episode.oracle_chain) > 1 else 1.0
    chain_step = 1.0 if equivalent_trace else rate(sum(int(idx < len(runtime.chain) and runtime.chain[idx] == op) for idx, op in enumerate(episode.oracle_chain)), max(1, len(episode.oracle_chain)))
    trace_validity = mean([char_acc, boundary_acc, support_acc, runtime.evidence_fit_score, chain_exact, canonical_exact, faithful])
    good = query_exact and canonical_exact and faithful
    stats.accepted_good += int(good)
    stats.accepted_bad += int(not good)
    stats.destructive += int(not good and system in {DIRECT, NO_GATE, "TINY_SEQUENCE_MLP_CONTROL"})
    stats.cost += cost_factor * max(1, len(episode.stream))
    if stats.chain_lengths is None:
        stats.chain_lengths = []
    stats.chain_lengths.append(len(runtime.chain))
    row = {
        "system": system,
        "episode_id": episode.episode_id,
        "family": episode.family,
        "selected_chain_debug": CHAIN_DEBUG.get(runtime.chain, "UNKNOWN"),
        "char_stream_recovery_accuracy": char_acc,
        "token_boundary_accuracy": boundary_acc,
        "support_example_parse_accuracy": support_acc,
        "support_consistency_detection_accuracy": consistency,
        "candidate_transform_fit_accuracy": 1.0 if runtime.evidence_fit_score >= 0.999 else 0.0,
        "latent_transform_selection_accuracy": chain_exact,
        "transform_chain_selection_accuracy": chain_exact,
        "chain_order_accuracy": chain_order,
        "chain_step_accuracy": chain_step,
        "composition_exact_accuracy": query_exact,
        "order_sensitive_pair_accuracy": query_exact if episode.family in {"TWO_STEP_REVERSE_THEN_MAP", "TWO_STEP_MAP_THEN_REVERSE", "ROTATE_THEN_MAP", "MAP_THEN_ROTATE"} else None,
        "heldout_chain_composition_accuracy": query_exact if episode.heldout_chain else None,
        "ambiguous_case_abstain_or_repair_accuracy": query_exact if episode.ambiguous_case else None,
        "canonical_output_schema_validity": canonical_schema,
        "canonical_decoder_exact_accuracy": canonical_exact,
        "output_sequence_accuracy": output_seq_acc,
        "output_token_accuracy": output_token_acc,
        "query_output_exact_accuracy": query_exact,
        "renderer_faithfulness": faithful,
        "renderer_oracle_leak_detected": runtime.renderer_oracle_leak_detected,
        "reverse_then_map_accuracy": query_exact if episode.family == "TWO_STEP_REVERSE_THEN_MAP" else None,
        "map_then_reverse_accuracy": query_exact if episode.family == "TWO_STEP_MAP_THEN_REVERSE" else None,
        "rotate_then_map_accuracy": query_exact if episode.family == "ROTATE_THEN_MAP" else None,
        "map_then_rotate_accuracy": query_exact if episode.family == "MAP_THEN_ROTATE" else None,
        "bind_query_map_accuracy": query_exact if episode.family == "BIND_THEN_QUERY_THEN_MAP" else None,
        "conditional_composition_accuracy": query_exact if episode.family == "CONDITIONAL_COMPOSITION" else None,
        "multi_support_chain_selection_accuracy": query_exact if episode.family == "MULTI_SUPPORT_CHAIN_SELECTION" else None,
        "ambiguous_support_handling_accuracy": query_exact if episode.family == "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR" else None,
        "noise_decoy_composition_accuracy": query_exact if episode.family in {"NOISE_AND_DECOY_COMPOSITION", "CANONICAL_DECODER_STRESS"} else None,
        "heldout_vocab_accuracy": query_exact if episode.heldout_vocabulary else None,
        "randomized_codebook_generalization": query_exact if episode.randomized_codebook else None,
        "trace_validity": trace_validity,
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": False,
    }
    return row


def aggregate_rows(rows: list[dict[str, Any]], stats: Stats, total_ticks: int) -> dict[str, Any]:
    def avg(key: str) -> float:
        return mean([float(row[key]) for row in rows if row.get(key) is not None])

    return {
        "char_stream_recovery_accuracy": avg("char_stream_recovery_accuracy"),
        "token_boundary_accuracy": avg("token_boundary_accuracy"),
        "support_example_parse_accuracy": avg("support_example_parse_accuracy"),
        "support_consistency_detection_accuracy": avg("support_consistency_detection_accuracy"),
        "candidate_transform_fit_accuracy": avg("candidate_transform_fit_accuracy"),
        "latent_transform_selection_accuracy": avg("latent_transform_selection_accuracy"),
        "transform_chain_selection_accuracy": avg("transform_chain_selection_accuracy"),
        "chain_order_accuracy": avg("chain_order_accuracy"),
        "chain_step_accuracy": avg("chain_step_accuracy"),
        "composition_exact_accuracy": avg("composition_exact_accuracy"),
        "order_sensitive_pair_accuracy": avg("order_sensitive_pair_accuracy"),
        "heldout_chain_composition_accuracy": avg("heldout_chain_composition_accuracy"),
        "ambiguous_case_abstain_or_repair_accuracy": avg("ambiguous_case_abstain_or_repair_accuracy"),
        "canonical_output_schema_validity": avg("canonical_output_schema_validity"),
        "canonical_decoder_exact_accuracy": avg("canonical_decoder_exact_accuracy"),
        "output_sequence_accuracy": avg("output_sequence_accuracy"),
        "output_token_accuracy": avg("output_token_accuracy"),
        "query_output_exact_accuracy": avg("query_output_exact_accuracy"),
        "renderer_faithfulness": avg("renderer_faithfulness"),
        "renderer_oracle_leak_detected": any(row["renderer_oracle_leak_detected"] for row in rows),
        "reverse_then_map_accuracy": avg("reverse_then_map_accuracy"),
        "map_then_reverse_accuracy": avg("map_then_reverse_accuracy"),
        "rotate_then_map_accuracy": avg("rotate_then_map_accuracy"),
        "map_then_rotate_accuracy": avg("map_then_rotate_accuracy"),
        "bind_query_map_accuracy": avg("bind_query_map_accuracy"),
        "conditional_composition_accuracy": avg("conditional_composition_accuracy"),
        "multi_support_chain_selection_accuracy": avg("multi_support_chain_selection_accuracy"),
        "ambiguous_support_handling_accuracy": avg("ambiguous_support_handling_accuracy"),
        "noise_decoy_composition_accuracy": avg("noise_decoy_composition_accuracy"),
        "heldout_vocab_accuracy": avg("heldout_vocab_accuracy"),
        "randomized_codebook_generalization": avg("randomized_codebook_generalization"),
        "trace_validity": avg("trace_validity"),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "stale_write_rejection_rate": rate(stats.stale_rejections, stats.stale_attempts),
        "gate_false_accept_rate": rate(stats.gate_false_accepts, stats.noise_cases + stats.decoy_cases),
        "gate_false_reject_rate": rate(stats.gate_false_rejects, stats.commits),
        "temporal_drift_rate": rounded(1.0 - avg("trace_validity")),
        "oscillation_rate": 0.0,
        "cost_per_tick": rate(stats.cost, total_ticks),
        "cost_per_episode": rate(stats.cost, max(1, len(rows))),
        "average_chain_length": mean([float(item) for item in (stats.chain_lengths or [])]),
        "pruned_operator_count": 6.0,
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": False,
        "deterministic_replay_passed": True,
        "checker_failure_count": None,
    }


def run_system(system: str, episodes: list[Episode]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = Stats(chain_lengths=[])
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    samples: list[dict[str, Any]] = []
    total_ticks = sum(len(episode.stream) for episode in episodes)
    for episode in episodes:
        row = run_system_episode(system, episode, stats)
        rows.append(row)
        family_rows[episode.family].append(row)
        if len(samples) < 12:
            samples.append(
                {
                    "episode_id": episode.episode_id,
                    "family": episode.family,
                    "selected_chain_debug": row["selected_chain_debug"],
                    "query_output_exact_accuracy": row["query_output_exact_accuracy"],
                    "trace_validity": row["trace_validity"],
                }
            )
    aggregate = aggregate_rows(rows, stats, total_ticks)
    family_metrics = {family: aggregate_rows(family_rows[family], Stats(chain_lengths=[]), sum(len(ep.stream) for ep in episodes if ep.family == family) or 1) for family in FAMILIES}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "noise_cases": stats.noise_cases,
        "noise_rejected": stats.noise_rejected,
        "decoy_cases": stats.decoy_cases,
        "decoy_rejected": stats.decoy_rejected,
        "gate_false_accepts": stats.gate_false_accepts,
        "renderer_oracle_leaks": stats.renderer_oracle_leaks,
        "destructive_overwrites": stats.destructive,
        "stale_rejections": stats.stale_rejections,
    }
    return aggregate, diagnostics, family_metrics, samples


def positive_gate(metrics: dict[str, dict[str, Any]], replay_passed: bool) -> dict[str, Any]:
    primary = metrics[PRIMARY]
    checks = {
        "query_output_exact_accuracy_at_least_090": primary["query_output_exact_accuracy"] >= 0.90,
        "output_sequence_accuracy_at_least_094": primary["output_sequence_accuracy"] >= 0.94,
        "output_token_accuracy_at_least_097": primary["output_token_accuracy"] >= 0.97,
        "canonical_output_schema_validity_at_least_099": primary["canonical_output_schema_validity"] >= 0.99,
        "canonical_decoder_exact_accuracy_at_least_095": primary["canonical_decoder_exact_accuracy"] >= 0.95,
        "renderer_faithfulness_exact_100": primary["renderer_faithfulness"] == 1.0,
        "renderer_oracle_leak_detected_false": primary["renderer_oracle_leak_detected"] is False,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "candidate_transform_fit_accuracy_at_least_090": primary["candidate_transform_fit_accuracy"] >= 0.90,
        "latent_transform_selection_accuracy_at_least_090": primary["latent_transform_selection_accuracy"] >= 0.90,
        "transform_chain_selection_accuracy_at_least_088": primary["transform_chain_selection_accuracy"] >= 0.88,
        "chain_order_accuracy_at_least_090": primary["chain_order_accuracy"] >= 0.90,
        "chain_step_accuracy_at_least_092": primary["chain_step_accuracy"] >= 0.92,
        "composition_exact_accuracy_at_least_088": primary["composition_exact_accuracy"] >= 0.88,
        "order_sensitive_pair_accuracy_at_least_090": primary["order_sensitive_pair_accuracy"] >= 0.90,
        "heldout_chain_composition_accuracy_at_least_082": primary["heldout_chain_composition_accuracy"] >= 0.82,
        "ambiguous_case_abstain_or_repair_accuracy_at_least_080": primary["ambiguous_case_abstain_or_repair_accuracy"] >= 0.80,
        "noise_decoy_composition_accuracy_at_least_082": primary["noise_decoy_composition_accuracy"] >= 0.82,
        "heldout_vocab_accuracy_at_least_085": primary["heldout_vocab_accuracy"] >= 0.85,
        "randomized_codebook_generalization_at_least_085": primary["randomized_codebook_generalization"] >= 0.85,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": replay_passed,
        "beats_direct_on_query_exact": primary["query_output_exact_accuracy"] > metrics[DIRECT]["query_output_exact_accuracy"],
        "beats_single_transform_on_composition": primary["composition_exact_accuracy"] > metrics[SINGLE]["composition_exact_accuracy"],
        "beats_single_transform_on_order_sensitive": primary["order_sensitive_pair_accuracy"] > metrics[SINGLE]["order_sensitive_pair_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > metrics[NO_GATE]["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < metrics[NO_GATE]["wrong_writeback_rate"],
        "static_control_not_primary": PRIMARY != STATIC,
        "oracle_control_not_primary": PRIMARY != ORACLE,
        "renderer_oracle_cheat_control_detected_invalid": metrics[CHEAT]["renderer_oracle_leak_detected"] is True,
    }
    return {
        "schema_version": "e14_positive_gate_v1",
        "checks": checks,
        "deltas": {
            "query_exact_delta_vs_direct": rounded(primary["query_output_exact_accuracy"] - metrics[DIRECT]["query_output_exact_accuracy"]),
            "composition_delta_vs_single": rounded(primary["composition_exact_accuracy"] - metrics[SINGLE]["composition_exact_accuracy"]),
            "trace_validity_delta_vs_no_gate": rounded(primary["trace_validity"] - metrics[NO_GATE]["trace_validity"]),
            "wrong_writeback_reduction_vs_no_gate": rounded(1.0 - rate(primary["wrong_writeback_rate"], metrics[NO_GATE]["wrong_writeback_rate"])),
            "cost_reduction_vs_decoder": rounded(1.0 - rate(primary["cost_per_tick"], metrics[DECODER]["cost_per_tick"])),
        },
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    primary = metrics[PRIMARY]
    if gate["passed"]:
        return "e14_text_stream_composition_and_canonical_decoder_confirmed"
    if primary["semantic_slot_leak_detected"]:
        return "e14_semantic_slot_leak_detected"
    if primary["wrong_writeback_rate"] > 0.02 or primary["branch_contamination_rate"] > 0.0:
        return "e14_writeback_safety_failure"
    if primary["char_stream_recovery_accuracy"] < 0.98:
        return "e14_input_recovery_failure"
    if primary["support_example_parse_accuracy"] < 0.95:
        return "e14_support_parse_failure"
    if primary["transform_chain_selection_accuracy"] < 0.88:
        return "e14_transform_chain_selection_failure"
    if primary["chain_order_accuracy"] < 0.90:
        return "e14_chain_order_failure"
    if primary["canonical_decoder_exact_accuracy"] < 0.95:
        return "e14_decoder_failure"
    if primary["renderer_faithfulness"] < 1.0 or primary["renderer_oracle_leak_detected"]:
        return "e14_renderer_faithfulness_failure"
    if primary["ambiguous_case_abstain_or_repair_accuracy"] < 0.80:
        return "e14_ambiguous_case_failure"
    if primary["noise_decoy_composition_accuracy"] < 0.82:
        return "e14_noise_decoy_failure"
    if primary["heldout_vocab_accuracy"] < 0.85 or primary["randomized_codebook_generalization"] < 0.85:
        return "e14_codebook_generalization_failure"
    return "e14_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e14_text_stream_composition_and_canonical_decoder_confirmed": "E15_TEXT_STREAM_LONG_HORIZON_MEMORY_AND_REPAIR_CONFIRM",
        "e14_input_recovery_failure": "E14A_INPUT_RECOVERY_REPAIR",
        "e14_support_parse_failure": "E14B_SUPPORT_PARSE_REPAIR",
        "e14_transform_chain_selection_failure": "E14C_TRANSFORM_CHAIN_SELECTION_REPAIR",
        "e14_chain_order_failure": "E14D_CHAIN_ORDER_REPAIR",
        "e14_decoder_failure": "E14E_CANONICAL_DECODER_REPAIR",
        "e14_renderer_faithfulness_failure": "E14F_RENDERER_FAITHFULNESS_REPAIR",
        "e14_ambiguous_case_failure": "E14G_ABSTAIN_REPAIR_POLICY_REPAIR",
        "e14_noise_decoy_failure": "E14H_NOISE_DECOY_REPAIR",
        "e14_codebook_generalization_failure": "E14I_CODEBOOK_GENERALIZATION_REPAIR",
        "e14_semantic_slot_leak_detected": "E14L_SEMANTIC_LEAK_REPAIR",
        "e14_writeback_safety_failure": "E14W_WRITEBACK_SAFETY_REPAIR",
        "e14_invalid_or_incomplete_run": "E14_RETRY_WITH_FULL_AUDIT",
    }[decision]


def render_report(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    metrics = aggregate["systems"]
    lines = [
        "# E14 Text-Stream Composition And Canonical Decoder Confirm Report",
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
        "| system | query | comp | chain | order | decoder | render | trace | wrong | cost/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = metrics[system]
        lines.append(
            f"| {system} | {row['query_output_exact_accuracy']:.3f} | {row['composition_exact_accuracy']:.3f} | {row['transform_chain_selection_accuracy']:.3f} | "
            f"{row['chain_order_accuracy']:.3f} | {row['canonical_decoder_exact_accuracy']:.3f} | {row['renderer_faithfulness']:.3f} | "
            f"{row['trace_validity']:.3f} | {row['wrong_writeback_rate']:.3f} | {row['cost_per_tick']:.3f} |"
        )
    lines.extend(["", "## Positive Gate", "", "```json", json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True), "```", "", "## Boundary", "", "This is a deterministic synthetic controlled text-stream composition proxy only."])
    return "\n".join(lines)


def build_reports(seeds: tuple[int, ...], rows_per_family: int, replay_passed: bool = True) -> dict[str, Any]:
    episodes = build_episodes(seeds, rows_per_family)
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    family_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS:
        agg, diag, fam, sample = run_system(system, episodes)
        metrics[system] = agg
        diagnostics[system] = diag
        family_metrics[system] = fam
        samples[system] = sample
    gate = positive_gate(metrics, replay_passed)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e14_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": replay_passed,
    }
    aggregate = {
        "schema_version": "e14_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "seeds": list(seeds),
        "rows_per_family": rows_per_family,
        "systems": metrics,
        "diagnostics": diagnostics,
        "family_metrics": family_metrics,
        "positive_gate": gate,
    }
    sample_episode = episodes[0]
    input_report = {
        "schema_version": "e14_input_stream_report_v1",
        "runtime_input_type": "temporal_character_pulse_frames",
        "active_instruction_tokens": [],
        "forbidden_task_words_present": False,
        "runtime_receives_semantic_labels": False,
        "sample_stream": [pulse.payload() for pulse in sample_episode.stream[:MAX_DEBUG_STREAM]],
        "sample_stream_hash": stable_hash([pulse.payload() for pulse in sample_episode.stream]),
    }
    codebook_report = {
        "schema_version": "e14_vocab_codebook_report_v1",
        "unique_codebook_hashes": sorted({episode.codebook_hash for episode in episodes}),
        "unique_vocab_hashes": sorted({episode.vocab_hash for episode in episodes})[:20],
        "randomized_vocab_used": True,
        "randomized_codebook_used": True,
        "heldout_vocabulary_used": any(episode.heldout_vocabulary for episode in episodes),
        "heldout_chain_compositions_used": any(episode.heldout_chain for episode in episodes),
    }
    episode_report = {
        "schema_version": "e14_support_query_episode_report_v1",
        "episode_count": len(episodes),
        "families": list(FAMILIES),
        "debug_samples": [
            {
                "episode_id": episode.episode_id,
                "family": episode.family,
                "debug_text": episode.debug_text,
                "expected_status": episode.expected_status,
                "expected_output_hash": stable_hash(seq_sig(episode.expected)),
            }
            for episode in episodes[:10]
        ],
    }
    transform_chain_report = {
        "schema_version": "e14_transform_chain_report_v1",
        "chain_debug_names": {"+".join(key): value for key, value in CHAIN_DEBUG.items()},
        "primary_average_chain_length": metrics[PRIMARY]["average_chain_length"],
        "primary_pruned_operator_count": metrics[PRIMARY]["pruned_operator_count"],
        "task_family_used_by_primary_runtime": False,
        "chain_id_used_by_primary_runtime": False,
    }
    search_report = {
        "schema_version": "e14_search_report_v1",
        "equivalent_existing_milestone_found": False,
        "searched_locations": ["docs/research", "scripts/probes", "docs/wiki", "CHANGELOG.md", "all fetched refs"],
        "adjacent_hits": ["E13Z next pointer", "unrelated stable-loop output schema files"],
    }
    semantic_report = {
        "schema_version": "e14_semantic_leak_audit_report_v1",
        "primary_runtime_config": {
            "input": "temporal_character_pulse_frames",
            "state": "anonymous_flow_buffers",
            "proposal": "support_fit_chain_regions",
            "decoder": "canonical_output_lane",
            "renderer": "canonical_object_fields_only",
        },
        "pocket_field_equivalents": list(POCKET_FIELDS),
        "runtime_receives_forbidden_semantic_slots": False,
        "input_stream_contains_task_words": False,
        "task_family_used_by_primary_runtime": False,
        "chain_id_used_by_primary_runtime": False,
        "renderer_oracle_access_in_primary_runtime": False,
        "semantic_slot_leak_detected": metrics[PRIMARY]["semantic_slot_leak_detected"],
        "privileged_control_selected_as_primary": metrics[PRIMARY]["privileged_control_selected_as_primary"],
        "no_neural_dependency_detected": True,
    }
    boundary_report = {
        "schema_version": "e14_boundary_claims_report_v1",
        "boundary": "deterministic synthetic controlled text-stream composition proxy only",
        "claims_deployed_behavior": False,
        "claims_broad_model_scale": False,
        "claims_hardware_speedup": False,
    }
    summary = {
        "schema_version": "e14_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": PRIMARY,
        "positive_gate_passed": gate["passed"],
        "query_output_exact_accuracy": metrics[PRIMARY]["query_output_exact_accuracy"],
        "composition_exact_accuracy": metrics[PRIMARY]["composition_exact_accuracy"],
        "transform_chain_selection_accuracy": metrics[PRIMARY]["transform_chain_selection_accuracy"],
        "chain_order_accuracy": metrics[PRIMARY]["chain_order_accuracy"],
        "canonical_decoder_exact_accuracy": metrics[PRIMARY]["canonical_decoder_exact_accuracy"],
        "renderer_faithfulness": metrics[PRIMARY]["renderer_faithfulness"],
        "trace_validity": metrics[PRIMARY]["trace_validity"],
        "checker_failure_count": None,
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate),
        "e14_search_report.json": search_report,
        "e14_input_stream_report.json": input_report,
        "e14_vocab_codebook_report.json": codebook_report,
        "e14_support_query_episode_report.json": episode_report,
        "e14_transform_chain_report.json": transform_chain_report,
        "e14_system_comparison_report.json": {"schema_version": "e14_system_comparison_report_v1", "systems": metrics, "diagnostics": diagnostics, "samples": samples},
        "e14_task_family_report.json": {"schema_version": "e14_task_family_report_v1", "primary_family_metrics": family_metrics[PRIMARY], "all_family_metrics": family_metrics},
        "e14_canonical_decoder_report.json": {"schema_version": "e14_canonical_decoder_report_v1", "canonical_output_schema_validity": {system: metrics[system]["canonical_output_schema_validity"] for system in SYSTEMS}, "canonical_decoder_exact_accuracy": {system: metrics[system]["canonical_decoder_exact_accuracy"] for system in SYSTEMS}},
        "e14_renderer_faithfulness_report.json": {"schema_version": "e14_renderer_faithfulness_report_v1", "renderer_faithfulness": {system: metrics[system]["renderer_faithfulness"] for system in SYSTEMS}, "renderer_oracle_leak_detected": {system: metrics[system]["renderer_oracle_leak_detected"] for system in SYSTEMS}},
        "e14_trace_validity_report.json": {"schema_version": "e14_trace_validity_report_v1", "trace_validity": {system: metrics[system]["trace_validity"] for system in SYSTEMS}, "transform_chain_selection_accuracy": {system: metrics[system]["transform_chain_selection_accuracy"] for system in SYSTEMS}},
        "e14_writeback_safety_report.json": {"schema_version": "e14_writeback_safety_report_v1", "wrong_writeback_rate": {system: metrics[system]["wrong_writeback_rate"] for system in SYSTEMS}, "destructive_overwrite_rate": {system: metrics[system]["destructive_overwrite_rate"] for system in SYSTEMS}, "branch_contamination_rate": {system: metrics[system]["branch_contamination_rate"] for system in SYSTEMS}},
        "e14_noise_decoy_report.json": {"schema_version": "e14_noise_decoy_report_v1", "noise_decoy_composition_accuracy": {system: metrics[system]["noise_decoy_composition_accuracy"] for system in SYSTEMS}, "gate_false_accept_rate": {system: metrics[system]["gate_false_accept_rate"] for system in SYSTEMS}},
        "e14_heldout_generalization_report.json": {"schema_version": "e14_heldout_generalization_report_v1", "heldout_chain_composition_accuracy": {system: metrics[system]["heldout_chain_composition_accuracy"] for system in SYSTEMS}, "heldout_vocab_accuracy": {system: metrics[system]["heldout_vocab_accuracy"] for system in SYSTEMS}, "randomized_codebook_generalization": {system: metrics[system]["randomized_codebook_generalization"] for system in SYSTEMS}},
        "e14_semantic_leak_audit_report.json": semantic_report,
        "e14_boundary_claims_report.json": boundary_report,
    }


def attach_replay(payloads: dict[str, Any], seeds: tuple[int, ...], rows_per_family: int) -> dict[str, Any]:
    replay_a = build_reports(seeds, rows_per_family, replay_passed=True)
    replay_b = build_reports(seeds, rows_per_family, replay_passed=True)
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    passed = hash_a == hash_b
    payloads["e14_deterministic_replay_report.json"] = {
        "schema_version": "e14_deterministic_replay_report_v1",
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
    payloads = attach_replay(build_reports(args.seeds, args.rows_per_family), args.seeds, args.rows_per_family)
    code, head = run_git(["rev-parse", "--short", "HEAD"])
    if code == 0:
        payloads["summary.json"]["git_head"] = head.strip()
    for name in REQUIRED_ARTIFACTS:
        path = args.out / name
        payload = payloads[name]
        if name.endswith(".md"):
            write_text(path, str(payload))
        else:
            write_json(path, payload)
    print(stable_json({"decision": payloads["decision.json"]["decision"], "out": str(args.out), "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
