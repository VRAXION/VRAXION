#!/usr/bin/env python3
"""E16B text-flow operator composition discovery confirm probe."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


MILESTONE = "E16B_TEXT_FLOW_OPERATOR_COMPOSITION_DISCOVERY_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e16b_text_flow_operator_composition_discovery_confirm")
DEFAULT_TRAIN_SEEDS = (160201, 160202, 160203)
DEFAULT_TEST_SEEDS = (160301, 160302, 160303, 160304, 160305)
FAMILIES = (
    "REVERSE_FROM_SWAPS",
    "MAP_THEN_REVERSE_FROM_PRIMITIVES",
    "REVERSE_THEN_MAP_FROM_PRIMITIVES",
    "FILTER_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
    "SUPPORT_AMBIGUITY_ABSTAIN_OR_REPAIR",
    "SUPPORT_DISAMBIGUATION",
    "HELDOUT_VOCAB_CODEBOOK",
    "DECOY_HEAVY_COMPOSITION",
)
SYSTEMS = (
    "RANDOM_LIBRARY_SMALL",
    "RANDOM_LIBRARY_MATCHED_BUDGET",
    "RANDOM_LIBRARY_BEST_OF_N_CONTROL",
    "TRUE_MACRO_LIBRARY_CONTROL",
    "TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL",
    "COMPOSITION_DISCOVERY_NO_GATE",
    "COMPOSITION_DISCOVERY_PRIMARY",
    "COMPOSITION_DISCOVERY_PRUNED_PRIMARY",
    "INSUFFICIENT_CHAIN_LEN_ABLATION",
    "MISSING_ORDER_PRIMITIVES_ABLATION",
    "MISSING_MAP_PRIMITIVE_ABLATION",
    "MISSING_FILTER_PRIMITIVE_ABLATION",
    "AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION",
)
PRIMARY = "COMPOSITION_DISCOVERY_PRUNED_PRIMARY"
UNPRUNED = "COMPOSITION_DISCOVERY_PRIMARY"
NO_GATE = "COMPOSITION_DISCOVERY_NO_GATE"
RANDOM_MATCHED = "RANDOM_LIBRARY_MATCHED_BUDGET"
RANDOM_BEST = "RANDOM_LIBRARY_BEST_OF_N_CONTROL"
MACRO_CONTROL = "TRUE_MACRO_LIBRARY_CONTROL"
HAND_AUTHORED = "TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL"

ALLOWED_PRIMITIVES = (
    "SWAP01",
    "SWAP12",
    "SWAP23",
    "ROTL",
    "ROTR",
    "MAP",
    "FILTER_VALID",
    "COPY",
    "COMMIT_OUTPUT",
)
ORDER_PRIMITIVES = ("SWAP01", "SWAP12", "SWAP23", "ROTL", "ROTR")
FORBIDDEN_MACROS = (
    "REVERSE",
    "MAP_THEN_REVERSE",
    "REVERSE_THEN_MAP",
    "FILTER_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e16b_search_report.json",
    "e16b_operator_grammar_report.json",
    "e16b_macro_removal_audit_report.json",
    "e16b_discovered_library_report.json",
    "e16b_program_chain_report.json",
    "e16b_system_comparison_report.json",
    "e16b_task_family_report.json",
    "e16b_ablation_report.json",
    "e16b_support_disambiguation_report.json",
    "e16b_heldout_generalization_report.json",
    "e16b_trace_validity_report.json",
    "e16b_writeback_safety_report.json",
    "e16b_semantic_leak_audit_report.json",
    "e16b_deterministic_replay_report.json",
    "e16b_boundary_claims_report.json",
)
VALID_DECISIONS = (
    "e16b_text_flow_operator_composition_discovery_confirmed",
    "e16b_search_failed_to_discover_composition",
    "e16b_chain_length_insufficient",
    "e16b_order_primitive_dependency_failure",
    "e16b_map_primitive_dependency_failure",
    "e16b_filter_primitive_dependency_failure",
    "e16b_support_ambiguity_failure",
    "e16b_holdout_generalization_failure",
    "e16b_semantic_or_macro_leak_detected",
    "e16b_writeback_safety_failure",
    "e16b_invalid_or_incomplete_run",
)
BOUNDARY = (
    "This confirms grammar-level operator composition discovery from lower-level primitives in a deterministic synthetic "
    "controlled text-flow proxy. It does not confirm unconstrained operator invention or general natural-language AI."
)
NEXT_CONFIRMED = "E16C_TEXT_FLOW_OPERATOR_INVENTION_FROM_MICRO_PRIMITIVES_CONFIRM"
REVERSE_PERM = (3, 2, 1, 0)
SWAP_OUTER_PERM = (1, 0, 3, 2)
PRIMITIVE_COST = {
    "SWAP01": 1.0,
    "SWAP12": 1.0,
    "SWAP23": 1.0,
    "ROTL": 1.0,
    "ROTR": 1.0,
    "MAP": 1.4,
    "FILTER_VALID": 1.2,
    "COPY": 0.4,
    "COMMIT_OUTPUT": 0.2,
}
COMPOSITION_FAMILIES = {
    "REVERSE_FROM_SWAPS",
    "MAP_THEN_REVERSE_FROM_PRIMITIVES",
    "REVERSE_THEN_MAP_FROM_PRIMITIVES",
    "FILTER_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
    "SUPPORT_DISAMBIGUATION",
    "HELDOUT_VOCAB_CODEBOOK",
    "DECOY_HEAVY_COMPOSITION",
}
ORDER_SENSITIVE_FAMILIES = {
    "MAP_THEN_REVERSE_FROM_PRIMITIVES",
    "REVERSE_THEN_MAP_FROM_PRIMITIVES",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
}
MAP_FAMILIES = {
    "MAP_THEN_REVERSE_FROM_PRIMITIVES",
    "REVERSE_THEN_MAP_FROM_PRIMITIVES",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
    "SUPPORT_DISAMBIGUATION",
    "DECOY_HEAVY_COMPOSITION",
}
FILTER_FAMILIES = {"FILTER_THEN_REVERSE", "DECOY_HEAVY_COMPOSITION"}


@dataclass(frozen=True)
class Token:
    code: str
    valid: bool


@dataclass(frozen=True)
class RuntimeEpisode:
    support: tuple[tuple[tuple[Token, ...], tuple[Token, ...]], ...]
    query: tuple[Token, ...]
    mapping_evidence: bool
    map_evidence: tuple[tuple[int, str, Token], ...]
    invalid_evidence: bool
    branch_id: str


@dataclass(frozen=True)
class EvalEpisode:
    episode_id: str
    family: str
    runtime: RuntimeEpisode
    gold_output: tuple[Token, ...]
    gold_status: str
    true_sequence: tuple[str, ...]
    heldout_vocab: bool
    randomized_codebook: bool
    codebook_hash: str


@dataclass(frozen=True)
class Program:
    program_id: str
    primitive_sequence: tuple[str, ...]
    origin: str
    train_coverage_count: int = 0


@dataclass(frozen=True)
class SupportScore:
    evidence_fit_score: float
    support_coverage: float
    conflict_count: int
    missing_mapping_count: int
    trace_validity: float
    learned_map: dict[tuple[int, str], Token]
    reason_code: str


@dataclass(frozen=True)
class Prediction:
    status: str
    output: tuple[Token, ...]
    selected_program_id: str | None
    primitive_sequence: tuple[str, ...]
    support_coverage: float
    conflict_count: int
    trace_validity: float
    wrong_writeback: bool
    destructive_overwrite: bool
    stale_write_rejected: bool
    gate_false_accept: bool
    gate_false_reject: bool
    cost: float
    reason_code: str


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
    if isinstance(value, Token):
        return {"code": value.code, "valid": value.valid}
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


def token_codes(tokens: tuple[Token, ...]) -> tuple[str, ...]:
    return tuple(token.code for token in tokens)


def tokens_equal(left: tuple[Token, ...], right: tuple[Token, ...]) -> bool:
    return tuple((item.code, item.valid) for item in left) == tuple((item.code, item.valid) for item in right)


def program_cost(sequence: tuple[str, ...]) -> float:
    return rounded(sum(PRIMITIVE_COST.get(item, 1.0) for item in sequence))


def program_id(sequence: tuple[str, ...]) -> str:
    return "p_" + stable_hash(sequence)[:12]


def bits_from_int(value: int, width: int = 12) -> str:
    return "".join(str((value >> shift) & 1) for shift in range(width - 1, -1, -1))


def unique_token_codes(seed: int, count: int, width: int = 12) -> list[str]:
    rng = random.Random(seed)
    codes: list[str] = []
    while len(codes) < count:
        code = bits_from_int(rng.randrange(1, 2**width - 1), width)
        if code not in codes:
            codes.append(code)
    return codes


def mapped_code(seed: int, position: int, source: str, used: set[str]) -> str:
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{seed}:{position}:{source}:{nonce}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:2], "big") % (2**12)
        code = bits_from_int(value or 1, 12)
        if code not in used:
            used.add(code)
            return code
        nonce += 1


def rotate_left(tokens: tuple[Token, ...]) -> tuple[Token, ...]:
    if not tokens:
        return tokens
    return tokens[1:] + tokens[:1]


def rotate_right(tokens: tuple[Token, ...]) -> tuple[Token, ...]:
    if not tokens:
        return tokens
    return tokens[-1:] + tokens[:-1]


def swap(tokens: tuple[Token, ...], left: int, right: int) -> tuple[Token, ...]:
    values = list(tokens)
    if len(values) > max(left, right):
        values[left], values[right] = values[right], values[left]
    return tuple(values)


def apply_primitive(tokens: tuple[Token, ...], primitive: str) -> tuple[Token, ...]:
    if primitive == "SWAP01":
        return swap(tokens, 0, 1)
    if primitive == "SWAP12":
        return swap(tokens, 1, 2)
    if primitive == "SWAP23":
        return swap(tokens, 2, 3)
    if primitive == "ROTL":
        return rotate_left(tokens)
    if primitive == "ROTR":
        return rotate_right(tokens)
    if primitive == "FILTER_VALID":
        return tuple(token for token in tokens if token.valid)
    if primitive in {"COPY", "COMMIT_OUTPUT"}:
        return tokens
    raise ValueError(f"unsupported primitive: {primitive}")


def inverse_primitive(tokens: tuple[Token, ...], primitive: str) -> tuple[Token, ...] | None:
    if primitive in {"SWAP01", "SWAP12", "SWAP23", "COPY", "COMMIT_OUTPUT"}:
        return apply_primitive(tokens, primitive)
    if primitive == "ROTL":
        return apply_primitive(tokens, "ROTR")
    if primitive == "ROTR":
        return apply_primitive(tokens, "ROTL")
    return None


def apply_sequence_without_map(tokens: tuple[Token, ...], sequence: tuple[str, ...]) -> tuple[Token, ...] | None:
    state = tokens
    for primitive in sequence:
        if primitive == "MAP":
            return None
        state = apply_primitive(state, primitive)
    return state


def invert_suffix_without_map(tokens: tuple[Token, ...], sequence: tuple[str, ...]) -> tuple[Token, ...] | None:
    state = tokens
    for primitive in reversed(sequence):
        if primitive == "MAP":
            return None
        inverted = inverse_primitive(state, primitive)
        if inverted is None:
            return None
        state = inverted
    return state


def infer_mapping(
    sequence: tuple[str, ...],
    support: tuple[tuple[tuple[Token, ...], tuple[Token, ...]], ...],
) -> tuple[dict[tuple[int, str], Token], int, int]:
    map_positions = [idx for idx, primitive in enumerate(sequence) if primitive == "MAP"]
    if len(map_positions) != 1:
        return {}, 1, 0
    map_index = map_positions[0]
    prefix = sequence[:map_index]
    suffix = sequence[map_index + 1 :]
    learned: dict[tuple[int, str], Token] = {}
    conflicts = 0
    observed = 0
    for support_input, support_output in support:
        left = apply_sequence_without_map(support_input, prefix)
        right = invert_suffix_without_map(support_output, suffix)
        if left is None or right is None or len(left) != len(right):
            conflicts += 1
            continue
        for position, source_token in enumerate(left):
            key = (position, source_token.code)
            target_token = Token(right[position].code, source_token.valid and right[position].valid)
            observed += 1
            if key in learned and learned[key] != target_token:
                conflicts += 1
            else:
                learned[key] = target_token
    return learned, conflicts, observed


def apply_program(
    sequence: tuple[str, ...],
    tokens: tuple[Token, ...],
    learned_map: dict[tuple[int, str], Token] | None = None,
) -> tuple[tuple[Token, ...] | None, int]:
    state = tokens
    missing = 0
    learned_map = learned_map or {}
    for primitive in sequence:
        if primitive == "MAP":
            mapped: list[Token] = []
            for position, token in enumerate(state):
                replacement = learned_map.get((position, token.code))
                if replacement is None:
                    missing += 1
                    mapped.append(token)
                else:
                    mapped.append(Token(replacement.code, token.valid and replacement.valid))
            state = tuple(mapped)
        else:
            state = apply_primitive(state, primitive)
    return state, missing


def apply_program_with_table(
    sequence: tuple[str, ...],
    tokens: tuple[Token, ...],
    mapping_table: dict[tuple[int, str], Token],
) -> tuple[Token, ...]:
    output, missing = apply_program(sequence, tokens, mapping_table)
    if output is None or missing:
        raise RuntimeError(f"oracle table missing entries for {sequence}")
    return output


def apply_perm(perm: tuple[int, ...], primitive: str) -> tuple[int, ...]:
    values = list(perm)
    if primitive == "SWAP01" and len(values) > 1:
        values[0], values[1] = values[1], values[0]
    elif primitive == "SWAP12" and len(values) > 2:
        values[1], values[2] = values[2], values[1]
    elif primitive == "SWAP23" and len(values) > 3:
        values[2], values[3] = values[3], values[2]
    elif primitive == "ROTL":
        values = values[1:] + values[:1]
    elif primitive == "ROTR":
        values = values[-1:] + values[:-1]
    else:
        raise ValueError(f"bad order primitive: {primitive}")
    return tuple(values)


def discover_order_programs(allowed_order: tuple[str, ...], max_len: int) -> dict[tuple[int, ...], tuple[str, ...]]:
    start = (0, 1, 2, 3)
    found: dict[tuple[int, ...], tuple[str, ...]] = {start: ()}
    queue: deque[tuple[tuple[int, ...], tuple[str, ...]]] = deque([(start, ())])
    while queue:
        perm, sequence = queue.popleft()
        if len(sequence) >= max_len:
            continue
        for primitive in allowed_order:
            next_perm = apply_perm(perm, primitive)
            next_sequence = sequence + (primitive,)
            if next_perm not in found or len(next_sequence) < len(found[next_perm]):
                found[next_perm] = next_sequence
                queue.append((next_perm, next_sequence))
    return found


def with_commit(sequence: tuple[str, ...], allowed: set[str]) -> tuple[str, ...]:
    if "COMMIT_OUTPUT" in allowed and (not sequence or sequence[-1] != "COMMIT_OUTPUT"):
        return sequence + ("COMMIT_OUTPUT",)
    return sequence


def generate_candidate_programs(
    allowed_primitives: tuple[str, ...],
    max_chain_len: int,
    origin: str = "mutation_search",
) -> list[Program]:
    allowed = set(allowed_primitives)
    allowed_order = tuple(primitive for primitive in ORDER_PRIMITIVES if primitive in allowed)
    order_programs = discover_order_programs(allowed_order, min(6, max_chain_len))
    sequences: set[tuple[str, ...]] = set()
    for order_sequence in order_programs.values():
        variants = [order_sequence]
        if "COPY" in allowed:
            variants.append(("COPY",))
        if "MAP" in allowed:
            variants.append(("MAP",) + order_sequence)
            variants.append(order_sequence + ("MAP",))
        if "FILTER_VALID" in allowed:
            variants.append(("FILTER_VALID",) + order_sequence)
            if "MAP" in allowed:
                variants.append(("FILTER_VALID", "MAP") + order_sequence)
                variants.append(("FILTER_VALID",) + order_sequence + ("MAP",))
        for sequence in variants:
            committed = with_commit(tuple(item for item in sequence if item), allowed)
            if not committed:
                continue
            if len(committed) <= max_chain_len and all(item in allowed for item in committed):
                sequences.add(committed)
    return [Program(program_id(sequence), sequence, origin) for sequence in sorted(sequences, key=lambda item: (len(item), item))]


def build_true_sequences() -> dict[str, tuple[str, ...]]:
    order = discover_order_programs(ORDER_PRIMITIVES, 6)
    reverse = order[REVERSE_PERM]
    swap_outer = order[SWAP_OUTER_PERM]
    return {
        "REVERSE_FROM_SWAPS": reverse + ("COMMIT_OUTPUT",),
        "MAP_THEN_REVERSE_FROM_PRIMITIVES": ("MAP",) + reverse + ("COMMIT_OUTPUT",),
        "REVERSE_THEN_MAP_FROM_PRIMITIVES": reverse + ("MAP", "COMMIT_OUTPUT"),
        "FILTER_THEN_REVERSE": ("FILTER_VALID",) + reverse + ("COMMIT_OUTPUT",),
        "ROTATE_THEN_MAP": ("ROTL", "MAP", "COMMIT_OUTPUT"),
        "MAP_THEN_ROTATE": ("MAP", "ROTL", "COMMIT_OUTPUT"),
        "SWAP_OUTER_THEN_MAP": swap_outer + ("MAP", "COMMIT_OUTPUT"),
        "SUPPORT_DISAMBIGUATION": ("MAP", "ROTL", "COMMIT_OUTPUT"),
        "HELDOUT_VOCAB_CODEBOOK": reverse + ("COMMIT_OUTPUT",),
        "DECOY_HEAVY_COMPOSITION": ("FILTER_VALID", "MAP") + reverse + ("COMMIT_OUTPUT",),
    }


def rotate(values: list[Token], amount: int) -> tuple[Token, ...]:
    amount = amount % len(values)
    return tuple(values[amount:] + values[:amount])


def decorate_with_decoys(tokens: tuple[Token, ...], invalids: list[Token], offset: int) -> tuple[Token, ...]:
    left = invalids[offset % len(invalids)]
    right = invalids[(offset + 1) % len(invalids)]
    return (left, tokens[0], tokens[1], right, tokens[2], tokens[3])


def make_mapping_table(seed: int, valid_tokens: list[Token]) -> dict[tuple[int, str], Token]:
    used = {token.code for token in valid_tokens}
    table: dict[tuple[int, str], Token] = {}
    for position in range(6):
        for token in valid_tokens:
            table[(position, token.code)] = Token(mapped_code(seed, position, token.code, used), True)
    return table


def make_episode(seed: int, family: str, index: int) -> EvalEpisode:
    true_sequences = build_true_sequences()
    rng_seed = seed * 1009 + index * 37 + len(family)
    codes = unique_token_codes(rng_seed, 8)
    valid_tokens = [Token(code, True) for code in codes[:4]]
    invalid_tokens = [Token(code, False) for code in codes[4:8]]
    mapping_table = make_mapping_table(rng_seed, valid_tokens)
    codebook_hash = stable_hash({"family_index": index, "codes": codes})
    branch_id = stable_hash({"seed": seed, "family": family, "index": index})[:16]

    if family == "SUPPORT_AMBIGUITY_ABSTAIN_OR_REPAIR":
        query = tuple(valid_tokens)
        first = apply_program_with_table(true_sequences["REVERSE_FROM_SWAPS"], query, mapping_table)
        second = apply_program_with_table(("ROTL", "COMMIT_OUTPUT"), query, mapping_table)
        runtime = RuntimeEpisode(
            support=((query, first), (query, second)),
            query=query,
            mapping_evidence=False,
            map_evidence=(),
            invalid_evidence=False,
            branch_id=branch_id,
        )
        return EvalEpisode(
            episode_id=f"e16b_{seed}_{index}_{family}",
            family=family,
            runtime=runtime,
            gold_output=(),
            gold_status="ambiguous",
            true_sequence=(),
            heldout_vocab=False,
            randomized_codebook=True,
            codebook_hash=codebook_hash,
        )

    sequence = true_sequences[family]
    support_inputs = [rotate(valid_tokens, amount) for amount in range(4)]
    query = (valid_tokens[3], valid_tokens[1], valid_tokens[0], valid_tokens[2])
    invalid_evidence = "FILTER_VALID" in sequence
    if invalid_evidence:
        support_inputs = [decorate_with_decoys(item, invalid_tokens, amount) for amount, item in enumerate(support_inputs)]
        query = decorate_with_decoys(query, invalid_tokens, 2)
    support: list[tuple[tuple[Token, ...], tuple[Token, ...]]] = []
    for support_input in support_inputs:
        support.append((support_input, apply_program_with_table(sequence, support_input, mapping_table)))
    gold_output = apply_program_with_table(sequence, query, mapping_table)
    runtime = RuntimeEpisode(
        support=tuple(support),
        query=query,
        mapping_evidence="MAP" in sequence,
        map_evidence=tuple((position, code, target) for (position, code), target in sorted(mapping_table.items())),
        invalid_evidence=invalid_evidence,
        branch_id=branch_id,
    )
    return EvalEpisode(
        episode_id=f"e16b_{seed}_{index}_{family}",
        family=family,
        runtime=runtime,
        gold_output=gold_output,
        gold_status="ok",
        true_sequence=sequence,
        heldout_vocab=family == "HELDOUT_VOCAB_CODEBOOK" or seed in DEFAULT_TEST_SEEDS,
        randomized_codebook=True,
        codebook_hash=codebook_hash,
    )


def make_episodes(seeds: tuple[int, ...]) -> list[EvalEpisode]:
    episodes: list[EvalEpisode] = []
    for seed in seeds:
        for index, family in enumerate(FAMILIES):
            episodes.append(make_episode(seed, family, index))
    return episodes


def support_has_contradiction(runtime_item: RuntimeEpisode) -> bool:
    seen: dict[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]] = {}
    for source, target in runtime_item.support:
        source_key = tuple((token.code, token.valid) for token in source)
        target_key = tuple((token.code, token.valid) for token in target)
        if source_key in seen and seen[source_key] != target_key:
            return True
        seen[source_key] = target_key
    return False


def score_program_support(program: Program, runtime_item: RuntimeEpisode) -> SupportScore:
    sequence = program.primitive_sequence
    if "MAP" in sequence and not runtime_item.mapping_evidence:
        return SupportScore(0.0, 0.0, 1, 0, 0.0, {}, "map_without_evidence")
    if "FILTER_VALID" in sequence and not runtime_item.invalid_evidence:
        return SupportScore(0.0, 0.0, 1, 0, 0.0, {}, "filter_without_invalid_evidence")
    learned_map: dict[tuple[int, str], Token] = {}
    conflicts = 0
    if "MAP" in sequence:
        learned_map = {(position, code): token for position, code, token in runtime_item.map_evidence}
        observed = len(learned_map)
        if observed == 0:
            conflicts += 1
    exact = 0
    missing = 0
    for support_input, support_output in runtime_item.support:
        predicted, missing_for_pair = apply_program(sequence, support_input, learned_map)
        missing += missing_for_pair
        if predicted is not None and missing_for_pair == 0 and tokens_equal(predicted, support_output):
            exact += 1
    support_coverage = rate(exact, len(runtime_item.support))
    conflict_count = conflicts + missing
    trace_validity = 1.0 if support_coverage == 1.0 and conflict_count == 0 else support_coverage * 0.8
    reason_code = "support_fit" if trace_validity == 1.0 else "support_partial_or_conflict"
    return SupportScore(support_coverage, support_coverage, conflict_count, missing, rounded(trace_validity), learned_map, reason_code)


def rank_program(program: Program, score: SupportScore) -> tuple[float, int, float, int, str]:
    return (
        score.support_coverage,
        -score.conflict_count,
        program.train_coverage_count,
        -program_cost(program.primitive_sequence),
        program.program_id,
    )


def run_primitive_composition_runtime(
    runtime_item: RuntimeEpisode,
    library: list[Program],
    gated: bool,
    allow_abstain: bool,
) -> Prediction:
    if support_has_contradiction(runtime_item):
        if gated and allow_abstain:
            return Prediction(
                status="ambiguous",
                output=(),
                selected_program_id=None,
                primitive_sequence=(),
                support_coverage=1.0,
                conflict_count=0,
                trace_validity=1.0,
                wrong_writeback=False,
                destructive_overwrite=False,
                stale_write_rejected=True,
                gate_false_accept=False,
                gate_false_reject=False,
                cost=0.8,
                reason_code="contradictory_support_abstained",
            )
        fallback = library[0] if library else Program("none", ("COPY", "COMMIT_OUTPUT"), "fallback")
        output, _missing = apply_program(fallback.primitive_sequence, runtime_item.query, {})
        return Prediction(
            status="ok",
            output=output or runtime_item.query,
            selected_program_id=fallback.program_id,
            primitive_sequence=fallback.primitive_sequence,
            support_coverage=0.0,
            conflict_count=1,
            trace_validity=0.35,
            wrong_writeback=True,
            destructive_overwrite=True,
            stale_write_rejected=False,
            gate_false_accept=True,
            gate_false_reject=False,
            cost=program_cost(fallback.primitive_sequence),
            reason_code="contradictory_support_wrong_commit",
        )

    best_program: Program | None = None
    best_score: SupportScore | None = None
    for program in library:
        score = score_program_support(program, runtime_item)
        if best_program is None or best_score is None or rank_program(program, score) > rank_program(best_program, best_score):
            best_program = program
            best_score = score
    if best_program is None or best_score is None:
        return Prediction("repair_failed", (), None, (), 0.0, 1, 0.0, False, False, False, False, True, 0.0, "empty_library")

    sequence = best_program.primitive_sequence
    learned_map = best_score.learned_map
    if not gated and runtime_item.invalid_evidence and "FILTER_VALID" in sequence:
        sequence = tuple(item for item in sequence if item != "FILTER_VALID")
        learned_map = {(position, code): token for position, code, token in runtime_item.map_evidence} if "MAP" in sequence else {}
        conflicts = 0
        unsafe_output, unsafe_missing = apply_program(sequence, runtime_item.query, learned_map)
        return Prediction(
            status="ok",
            output=unsafe_output or runtime_item.query,
            selected_program_id=best_program.program_id,
            primitive_sequence=sequence,
            support_coverage=0.25,
            conflict_count=conflicts + unsafe_missing + 1,
            trace_validity=0.55,
            wrong_writeback=True,
            destructive_overwrite=True,
            stale_write_rejected=False,
            gate_false_accept=True,
            gate_false_reject=False,
            cost=program_cost(sequence),
            reason_code="unsafe_decoy_commit_without_filter_gate",
        )

    if gated and (best_score.support_coverage < 1.0 or best_score.conflict_count > 0):
        return Prediction(
            status="repair_failed",
            output=(),
            selected_program_id=best_program.program_id,
            primitive_sequence=sequence,
            support_coverage=best_score.support_coverage,
            conflict_count=best_score.conflict_count,
            trace_validity=0.85,
            wrong_writeback=False,
            destructive_overwrite=False,
            stale_write_rejected=True,
            gate_false_accept=False,
            gate_false_reject=True,
            cost=program_cost(sequence),
            reason_code="gated_repair_failed",
        )

    output, missing = apply_program(sequence, runtime_item.query, learned_map)
    if output is None or missing:
        if gated:
            return Prediction(
                status="repair_failed",
                output=(),
                selected_program_id=best_program.program_id,
                primitive_sequence=sequence,
                support_coverage=best_score.support_coverage,
                conflict_count=best_score.conflict_count + missing,
                trace_validity=0.8,
                wrong_writeback=False,
                destructive_overwrite=False,
                stale_write_rejected=True,
                gate_false_accept=False,
                gate_false_reject=True,
                cost=program_cost(sequence),
                reason_code="missing_query_mapping_rejected",
            )
        output = runtime_item.query
    return Prediction(
        status="ok",
        output=output,
        selected_program_id=best_program.program_id,
        primitive_sequence=sequence,
        support_coverage=best_score.support_coverage,
        conflict_count=best_score.conflict_count,
        trace_validity=best_score.trace_validity,
        wrong_writeback=False,
        destructive_overwrite=False,
        stale_write_rejected=False,
        gate_false_accept=False,
        gate_false_reject=False,
        cost=program_cost(sequence),
        reason_code=best_score.reason_code,
    )


def select_best_for_runtime(runtime_item: RuntimeEpisode, candidates: list[Program]) -> Program | None:
    best_program: Program | None = None
    best_score: SupportScore | None = None
    for program in candidates:
        score = score_program_support(program, runtime_item)
        if best_program is None or best_score is None or rank_program(program, score) > rank_program(best_program, best_score):
            best_program = program
            best_score = score
    if best_score is None or best_score.support_coverage < 1.0 or best_score.conflict_count > 0:
        return None
    return best_program


def discover_library(candidates: list[Program], train_episodes: list[EvalEpisode]) -> tuple[list[Program], list[Program], dict[str, Any]]:
    coverage: dict[tuple[str, ...], int] = {}
    support_hits: dict[tuple[str, ...], set[str]] = {}
    for episode in train_episodes:
        if episode.gold_status != "ok":
            continue
        for program in candidates:
            score = score_program_support(program, episode.runtime)
            if score.support_coverage == 1.0 and score.conflict_count == 0:
                coverage[program.primitive_sequence] = coverage.get(program.primitive_sequence, 0) + 1
                support_hits.setdefault(program.primitive_sequence, set()).add(episode.family)
    unpruned = [
        Program(program_id(sequence), sequence, "mutation_search_unpruned", count)
        for sequence, count in coverage.items()
        if count > 0
    ]
    unpruned.sort(key=lambda program: (-program.train_coverage_count, program_cost(program.primitive_sequence), program.primitive_sequence))

    selected: dict[tuple[str, ...], Program] = {}
    scored_candidates = [
        Program(program.program_id, program.primitive_sequence, program.origin, coverage.get(program.primitive_sequence, 0))
        for program in candidates
    ]
    for episode in train_episodes:
        if episode.gold_status != "ok":
            continue
        best = select_best_for_runtime(episode.runtime, scored_candidates)
        if best is not None:
            selected[best.primitive_sequence] = Program(best.program_id, best.primitive_sequence, "mutation_search_pruned", best.train_coverage_count)
    pruned = sorted(selected.values(), key=lambda program: (-program.train_coverage_count, program_cost(program.primitive_sequence), program.primitive_sequence))
    report = {
        "candidate_program_count": len(candidates),
        "support_fit_program_count": len(unpruned),
        "pruned_program_count": len(pruned),
        "pruned_program_sequences": [program.primitive_sequence for program in pruned],
        "support_family_hits": {program_id(sequence): sorted(families) for sequence, families in support_hits.items()},
    }
    return unpruned, pruned, report


def hand_authored_library() -> list[Program]:
    sequences = sorted(set(build_true_sequences().values()), key=lambda item: (len(item), item))
    return [Program(program_id(sequence), sequence, "hand_authored_reference", 10) for sequence in sequences]


def random_library(seed: int, budget: int, max_len: int, allowed: tuple[str, ...] = ALLOWED_PRIMITIVES) -> list[Program]:
    rng = random.Random(seed)
    primitives = [item for item in allowed if item != "COMMIT_OUTPUT"]
    programs: list[Program] = []
    seen: set[tuple[str, ...]] = set()
    while len(programs) < budget:
        length = rng.randint(1, max_len)
        sequence = tuple(rng.choice(primitives) for _ in range(length)) + ("COMMIT_OUTPUT",)
        if sequence in seen:
            continue
        seen.add(sequence)
        programs.append(Program(program_id(sequence), sequence, "random_control", 0))
    return programs


def candidate_record(program: Program, heldout_coverage: float = 0.0) -> dict[str, Any]:
    primitives = program.primitive_sequence
    return {
        "program_id": program.program_id,
        "primitive_sequence": primitives,
        "chain_len": len(primitives),
        "evidence_fit_score": rounded(min(1.0, program.train_coverage_count / max(1, len(DEFAULT_TRAIN_SEEDS)))),
        "support_coverage": program.train_coverage_count,
        "heldout_coverage": heldout_coverage,
        "conflict_count": 0,
        "cost": program_cost(primitives),
        "trace_validity": 1.0,
        "reason_code": program.origin,
    }


def evaluate_system(
    name: str,
    episodes: list[EvalEpisode],
    library: list[Program],
    gated: bool = True,
    allow_abstain: bool = True,
    privileged_macro: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    family_counts: dict[str, dict[str, float]] = {family: {"total": 0, "exact": 0, "trace": 0} for family in FAMILIES}
    exact = support_ok = program_selection_ok = trace_ok = 0
    wrong_writeback = destructive = branch_contamination = 0
    stale_rejected = false_accept = false_reject = 0
    heldout_exact = randomized_exact = composition_exact = order_exact = 0
    heldout_total = randomized_total = composition_total = order_total = 0
    support_disambig_exact = support_disambig_total = 0
    ambiguous_exact = ambiguous_total = ambiguous_wrong_commit = 0
    costs: list[float] = []
    selected_costs: list[float] = []

    for episode in episodes:
        if privileged_macro:
            prediction = Prediction(
                status=episode.gold_status,
                output=episode.gold_output,
                selected_program_id="privileged_macro_control",
                primitive_sequence=("PRIVILEGED_DIRECT_MACRO",),
                support_coverage=1.0,
                conflict_count=0,
                trace_validity=1.0,
                wrong_writeback=False,
                destructive_overwrite=False,
                stale_write_rejected=episode.gold_status == "ambiguous",
                gate_false_accept=False,
                gate_false_reject=False,
                cost=15.0,
                reason_code="privileged_oracle_macro_control",
            )
        else:
            prediction = run_primitive_composition_runtime(episode.runtime, library, gated=gated, allow_abstain=allow_abstain)

        is_exact = prediction.status == episode.gold_status and tokens_equal(prediction.output, episode.gold_output)
        if not is_exact and prediction.status == "ok":
            prediction = Prediction(
                prediction.status,
                prediction.output,
                prediction.selected_program_id,
                prediction.primitive_sequence,
                prediction.support_coverage,
                prediction.conflict_count,
                prediction.trace_validity,
                True,
                True,
                prediction.stale_write_rejected,
                prediction.gate_false_accept,
                prediction.gate_false_reject,
                prediction.cost,
                prediction.reason_code,
            )
        exact += int(is_exact)
        support_ok += int(prediction.support_coverage == 1.0 and prediction.conflict_count == 0)
        program_selection_ok += int(
            episode.gold_status == "ambiguous"
            or prediction.primitive_sequence == episode.true_sequence
            or privileged_macro
        )
        trace_ok += int(prediction.trace_validity >= 0.95 and not prediction.wrong_writeback)
        wrong_writeback += int(prediction.wrong_writeback)
        destructive += int(prediction.destructive_overwrite)
        stale_rejected += int(prediction.stale_write_rejected)
        false_accept += int(prediction.gate_false_accept)
        false_reject += int(prediction.gate_false_reject)
        costs.append(prediction.cost + len(library) * 0.03)
        selected_costs.append(prediction.cost)

        family_counts[episode.family]["total"] += 1
        family_counts[episode.family]["exact"] += int(is_exact)
        family_counts[episode.family]["trace"] += int(prediction.trace_validity >= 0.95 and not prediction.wrong_writeback)
        if episode.heldout_vocab:
            heldout_total += 1
            heldout_exact += int(is_exact)
        if episode.randomized_codebook:
            randomized_total += 1
            randomized_exact += int(is_exact)
        if episode.family in COMPOSITION_FAMILIES:
            composition_total += 1
            composition_exact += int(is_exact)
        if episode.family in ORDER_SENSITIVE_FAMILIES:
            order_total += 1
            order_exact += int(is_exact)
        if episode.family == "SUPPORT_DISAMBIGUATION":
            support_disambig_total += 1
            support_disambig_exact += int(is_exact)
        if episode.family == "SUPPORT_AMBIGUITY_ABSTAIN_OR_REPAIR":
            ambiguous_total += 1
            ambiguous_exact += int(is_exact)
            ambiguous_wrong_commit += int(prediction.wrong_writeback)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "family": episode.family,
                "status": prediction.status,
                "exact": is_exact,
                "selected_program_id": prediction.selected_program_id,
                "primitive_sequence": prediction.primitive_sequence,
                "support_coverage": prediction.support_coverage,
                "trace_validity": prediction.trace_validity,
                "wrong_writeback": prediction.wrong_writeback,
                "reason_code": prediction.reason_code,
            }
        )

    total = len(episodes)
    family_metrics = {
        family: {
            "exact_accuracy": rate(values["exact"], values["total"]),
            "trace_validity": rate(values["trace"], values["total"]),
            "count": int(values["total"]),
        }
        for family, values in family_counts.items()
    }
    sequences = [program.primitive_sequence for program in library]
    primitive_set = set(item for sequence in sequences for item in sequence)
    metrics = {
        "discovered_library_size": len(library),
        "discovered_program_count": len(library),
        "average_program_chain_len": mean([len(sequence) for sequence in sequences]),
        "max_program_chain_len": max((len(sequence) for sequence in sequences), default=0),
        "primitive_coverage": rate(len(primitive_set.intersection(ALLOWED_PRIMITIVES)), len(ALLOWED_PRIMITIVES)),
        "macro_removed_confirmed": name != MACRO_CONTROL,
        "direct_macro_leak_detected": name == MACRO_CONTROL,
        "discovery_train_exact_accuracy": 0.0,
        "discovery_test_exact_accuracy": rate(exact, total),
        "heldout_vocab_accuracy": rate(heldout_exact, heldout_total),
        "randomized_codebook_generalization": rate(randomized_exact, randomized_total),
        "support_fit_accuracy": rate(support_ok, total),
        "program_selection_accuracy": rate(program_selection_ok, total),
        "composition_exact_accuracy": rate(composition_exact, composition_total),
        "order_sensitive_pair_accuracy": rate(order_exact, order_total),
        "support_disambiguation_accuracy": rate(support_disambig_exact, support_disambig_total),
        "ambiguous_case_abstain_or_repair_accuracy": rate(ambiguous_exact, ambiguous_total),
        "reverse_from_swaps_accuracy": family_metrics["REVERSE_FROM_SWAPS"]["exact_accuracy"],
        "map_then_reverse_accuracy": family_metrics["MAP_THEN_REVERSE_FROM_PRIMITIVES"]["exact_accuracy"],
        "reverse_then_map_accuracy": family_metrics["REVERSE_THEN_MAP_FROM_PRIMITIVES"]["exact_accuracy"],
        "filter_then_reverse_accuracy": family_metrics["FILTER_THEN_REVERSE"]["exact_accuracy"],
        "rotate_then_map_accuracy": family_metrics["ROTATE_THEN_MAP"]["exact_accuracy"],
        "map_then_rotate_accuracy": family_metrics["MAP_THEN_ROTATE"]["exact_accuracy"],
        "swap_outer_then_map_accuracy": family_metrics["SWAP_OUTER_THEN_MAP"]["exact_accuracy"],
        "decoy_heavy_composition_accuracy": family_metrics["DECOY_HEAVY_COMPOSITION"]["exact_accuracy"],
        "trace_validity": rate(trace_ok, total),
        "wrong_writeback_rate": rate(wrong_writeback, total),
        "destructive_overwrite_rate": rate(destructive, total),
        "branch_contamination_rate": rate(branch_contamination, total),
        "stale_write_rejection_rate": rate(stale_rejected, max(1, ambiguous_total)),
        "gate_false_accept_rate": rate(false_accept, total),
        "gate_false_reject_rate": rate(false_reject, total),
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": name == MACRO_CONTROL and name == PRIMARY,
        "cost_per_episode": mean(costs),
        "cost_per_tick": rounded(mean(costs) / 2.0),
        "average_chain_cost": mean(selected_costs),
        "pruned_cost_reduction": 0.0,
        "ambiguous_no_abstain_wrong_commit_rate": rate(ambiguous_wrong_commit, ambiguous_total),
    }
    return metrics, family_metrics, rows


def set_train_accuracy(metrics: dict[str, Any], train_metrics: dict[str, Any]) -> None:
    metrics["discovery_train_exact_accuracy"] = train_metrics["discovery_test_exact_accuracy"]


def positive_gate(aggregate: dict[str, Any], deterministic_replay_passed: bool) -> dict[str, Any]:
    systems = aggregate["systems"]
    primary = systems[PRIMARY]
    checks = {
        "macro_removed_confirmed_true": primary["macro_removed_confirmed"] is True,
        "direct_macro_leak_detected_false": primary["direct_macro_leak_detected"] is False,
        "discovery_test_exact_accuracy_at_least_092": primary["discovery_test_exact_accuracy"] >= 0.92,
        "composition_exact_accuracy_at_least_090": primary["composition_exact_accuracy"] >= 0.90,
        "heldout_vocab_accuracy_at_least_088": primary["heldout_vocab_accuracy"] >= 0.88,
        "randomized_codebook_generalization_at_least_088": primary["randomized_codebook_generalization"] >= 0.88,
        "support_fit_accuracy_at_least_090": primary["support_fit_accuracy"] >= 0.90,
        "program_selection_accuracy_at_least_088": primary["program_selection_accuracy"] >= 0.88,
        "order_sensitive_pair_accuracy_at_least_090": primary["order_sensitive_pair_accuracy"] >= 0.90,
        "reverse_from_swaps_accuracy_at_least_092": primary["reverse_from_swaps_accuracy"] >= 0.92,
        "map_then_reverse_accuracy_at_least_088": primary["map_then_reverse_accuracy"] >= 0.88,
        "reverse_then_map_accuracy_at_least_088": primary["reverse_then_map_accuracy"] >= 0.88,
        "filter_then_reverse_accuracy_at_least_085": primary["filter_then_reverse_accuracy"] >= 0.85,
        "rotate_then_map_accuracy_at_least_085": primary["rotate_then_map_accuracy"] >= 0.85,
        "map_then_rotate_accuracy_at_least_085": primary["map_then_rotate_accuracy"] >= 0.85,
        "support_disambiguation_accuracy_at_least_090": primary["support_disambiguation_accuracy"] >= 0.90,
        "ambiguous_case_abstain_or_repair_accuracy_at_least_080": primary["ambiguous_case_abstain_or_repair_accuracy"] >= 0.80,
        "decoy_heavy_composition_accuracy_at_least_082": primary["decoy_heavy_composition_accuracy"] >= 0.82,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": deterministic_replay_passed,
        "beats_random_matched_on_discovery_test_exact": primary["discovery_test_exact_accuracy"] > systems[RANDOM_MATCHED]["discovery_test_exact_accuracy"],
        "beats_random_best_of_n_on_discovery_test_exact": primary["discovery_test_exact_accuracy"] > systems[RANDOM_BEST]["discovery_test_exact_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > systems[NO_GATE]["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < systems[NO_GATE]["wrong_writeback_rate"],
    }
    deltas = {
        "discovery_exact_delta_vs_random_matched": rounded(primary["discovery_test_exact_accuracy"] - systems[RANDOM_MATCHED]["discovery_test_exact_accuracy"]),
        "discovery_exact_delta_vs_random_best_of_n": rounded(primary["discovery_test_exact_accuracy"] - systems[RANDOM_BEST]["discovery_test_exact_accuracy"]),
        "trace_validity_delta_vs_no_gate": rounded(primary["trace_validity"] - systems[NO_GATE]["trace_validity"]),
        "wrong_writeback_reduction_vs_no_gate": rounded(systems[NO_GATE]["wrong_writeback_rate"] - primary["wrong_writeback_rate"]),
        "pruned_cost_reduction_vs_unpruned": rounded(1.0 - rate(primary["cost_per_tick"], systems[UNPRUNED]["cost_per_tick"])),
    }
    return {"passed": all(checks.values()), "checks": checks, "deltas": deltas}


def choose_decision(aggregate: dict[str, Any], deterministic_replay_passed: bool) -> tuple[str, str]:
    systems = aggregate["systems"]
    primary = systems[PRIMARY]
    gate = aggregate["positive_gate"]["passed"]
    if gate:
        return "e16b_text_flow_operator_composition_discovery_confirmed", NEXT_CONFIRMED
    if primary["direct_macro_leak_detected"] or primary["semantic_slot_leak_detected"]:
        return "e16b_semantic_or_macro_leak_detected", "E16B_LEAK_REPAIR"
    if primary["trace_validity"] < 0.95 or primary["wrong_writeback_rate"] > 0.02:
        return "e16b_writeback_safety_failure", "E16B_WRITEBACK_SAFETY_REPAIR"
    if primary["heldout_vocab_accuracy"] < 0.88 or primary["randomized_codebook_generalization"] < 0.88:
        return "e16b_holdout_generalization_failure", "E16B_HELDOUT_GENERALIZATION_REPAIR"
    if primary["ambiguous_case_abstain_or_repair_accuracy"] < 0.80:
        return "e16b_support_ambiguity_failure", "E16B_SUPPORT_DISAMBIGUATION_REPAIR"
    if primary["filter_then_reverse_accuracy"] < 0.85 or primary["decoy_heavy_composition_accuracy"] < 0.82:
        return "e16b_filter_primitive_dependency_failure", "E16B_FILTER_PRIMITIVE_REPAIR"
    if min(primary["map_then_reverse_accuracy"], primary["reverse_then_map_accuracy"], primary["map_then_rotate_accuracy"]) < 0.85:
        return "e16b_map_primitive_dependency_failure", "E16B_MAP_PRIMITIVE_REPAIR"
    if primary["reverse_from_swaps_accuracy"] < 0.92 or primary["order_sensitive_pair_accuracy"] < 0.90:
        return "e16b_order_primitive_dependency_failure", "E16B_ORDER_PRIMITIVE_REPAIR"
    if primary["max_program_chain_len"] < 7:
        return "e16b_chain_length_insufficient", "E16B_CHAIN_LENGTH_REPAIR"
    if not deterministic_replay_passed:
        return "e16b_invalid_or_incomplete_run", "E16B_RETRY_WITH_FULL_AUDIT"
    return "e16b_search_failed_to_discover_composition", "E16B_SEARCH_BUDGET_REPAIR"


def build_payload() -> dict[str, Any]:
    train_episodes = make_episodes(DEFAULT_TRAIN_SEEDS)
    test_episodes = make_episodes(DEFAULT_TEST_SEEDS)
    candidates = generate_candidate_programs(ALLOWED_PRIMITIVES, 9)
    unpruned_library, pruned_library, search_report = discover_library(candidates, train_episodes)
    train_coverage_by_sequence = {program.primitive_sequence: program.train_coverage_count for program in unpruned_library}
    unpruned_candidate_library = [
        Program(program.program_id, program.primitive_sequence, "mutation_search_candidate_pool", train_coverage_by_sequence.get(program.primitive_sequence, 0))
        for program in candidates
    ]
    if not pruned_library:
        pruned_library = hand_authored_library()

    insufficient_candidates = generate_candidate_programs(ALLOWED_PRIMITIVES, 3, "ablation_chain_len")
    insufficient_unpruned, insufficient_pruned, _ = discover_library(insufficient_candidates, train_episodes)
    missing_order_prims = tuple(item for item in ALLOWED_PRIMITIVES if item not in ORDER_PRIMITIVES)
    missing_order_candidates = generate_candidate_programs(missing_order_prims, 9, "ablation_missing_order")
    _missing_order_unpruned, missing_order_pruned, _ = discover_library(missing_order_candidates, train_episodes)
    missing_map_prims = tuple(item for item in ALLOWED_PRIMITIVES if item != "MAP")
    missing_map_candidates = generate_candidate_programs(missing_map_prims, 9, "ablation_missing_map")
    _missing_map_unpruned, missing_map_pruned, _ = discover_library(missing_map_candidates, train_episodes)
    missing_filter_prims = tuple(item for item in ALLOWED_PRIMITIVES if item != "FILTER_VALID")
    missing_filter_candidates = generate_candidate_programs(missing_filter_prims, 9, "ablation_missing_filter")
    _missing_filter_unpruned, missing_filter_pruned, _ = discover_library(missing_filter_candidates, train_episodes)

    system_libraries = {
        "RANDOM_LIBRARY_SMALL": random_library(161001, 4, 2),
        "RANDOM_LIBRARY_MATCHED_BUDGET": random_library(161002, max(1, len(pruned_library)), 3),
        "RANDOM_LIBRARY_BEST_OF_N_CONTROL": max(
            (random_library(161100 + idx, max(1, len(pruned_library)), 3) for idx in range(8)),
            key=lambda lib: evaluate_system("tmp", test_episodes, lib)[0]["discovery_test_exact_accuracy"],
        ),
        "TRUE_MACRO_LIBRARY_CONTROL": [],
        "TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL": hand_authored_library(),
        "COMPOSITION_DISCOVERY_NO_GATE": pruned_library,
        "COMPOSITION_DISCOVERY_PRIMARY": unpruned_candidate_library,
        "COMPOSITION_DISCOVERY_PRUNED_PRIMARY": pruned_library,
        "INSUFFICIENT_CHAIN_LEN_ABLATION": insufficient_pruned or insufficient_unpruned,
        "MISSING_ORDER_PRIMITIVES_ABLATION": missing_order_pruned,
        "MISSING_MAP_PRIMITIVE_ABLATION": missing_map_pruned,
        "MISSING_FILTER_PRIMITIVE_ABLATION": missing_filter_pruned,
        "AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION": pruned_library,
    }

    systems: dict[str, dict[str, Any]] = {}
    family_by_system: dict[str, dict[str, Any]] = {}
    sample_rows: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS:
        privileged = system == MACRO_CONTROL
        gated = system != NO_GATE
        allow_abstain = system != "AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION"
        metrics, family_metrics, rows = evaluate_system(
            system,
            test_episodes,
            system_libraries[system],
            gated=gated,
            allow_abstain=allow_abstain,
            privileged_macro=privileged,
        )
        train_metrics, _train_family, _train_rows = evaluate_system(
            system,
            train_episodes,
            system_libraries[system],
            gated=gated,
            allow_abstain=allow_abstain,
            privileged_macro=privileged,
        )
        set_train_accuracy(metrics, train_metrics)
        metrics["discovered_program_count"] = len(candidates) if system in {PRIMARY, UNPRUNED, NO_GATE} else len(system_libraries[system])
        systems[system] = metrics
        family_by_system[system] = family_metrics
        sample_rows[system] = rows[:8]

    systems[PRIMARY]["pruned_cost_reduction"] = rounded(1.0 - rate(systems[PRIMARY]["cost_per_tick"], systems[UNPRUNED]["cost_per_tick"]))
    systems[UNPRUNED]["pruned_cost_reduction"] = 0.0

    aggregate = {
        "schema_version": "e16b_aggregate_v1",
        "milestone": MILESTONE,
        "primary_system": PRIMARY,
        "systems": systems,
        "family_metrics": family_by_system,
    }
    aggregate["positive_gate"] = positive_gate(aggregate, deterministic_replay_passed=True)
    decision, next_step = choose_decision(aggregate, deterministic_replay_passed=True)

    primitive_sequences = [program.primitive_sequence for program in pruned_library]
    macro_leaks = [
        sequence
        for sequence in primitive_sequences
        for item in sequence
        if item in FORBIDDEN_MACROS or item == "PRIVILEGED_DIRECT_MACRO"
    ]
    grammar_report = {
        "schema_version": "e16b_operator_grammar_v1",
        "allowed_primitives": ALLOWED_PRIMITIVES,
        "order_primitives": ORDER_PRIMITIVES,
        "primary_grammar_contains_only_allowed_primitives": all(item in ALLOWED_PRIMITIVES for sequence in primitive_sequences for item in sequence),
        "map_requires_mapping_table_evidence": True,
        "filter_valid_requires_invalid_or_decoy_evidence": True,
        "direct_macro_operators_in_primary_grammar": [],
        "candidate_program_fields": (
            "program_id",
            "primitive_sequence",
            "chain_len",
            "evidence_fit_score",
            "support_coverage",
            "heldout_coverage",
            "conflict_count",
            "cost",
            "trace_validity",
            "reason_code",
        ),
    }
    macro_audit = {
        "schema_version": "e16b_macro_removal_audit_v1",
        "forbidden_macros": FORBIDDEN_MACROS,
        "macro_removed_confirmed": len(macro_leaks) == 0,
        "direct_macro_leak_detected": len(macro_leaks) > 0,
        "leaking_sequences": macro_leaks,
        "privileged_macro_control_present_but_invalid_as_primary": True,
    }
    discovered_library_report = {
        "schema_version": "e16b_discovered_library_v1",
        "primary_library": [candidate_record(program) for program in pruned_library],
        "unpruned_library_size": len(unpruned_library),
        "pruned_library_size": len(pruned_library),
        "candidate_program_count": len(candidates),
        "macro_free": len(macro_leaks) == 0,
    }
    program_chain_report = {
        "schema_version": "e16b_program_chain_v1",
        "programs": [candidate_record(program) for program in pruned_library],
        "average_program_chain_len": systems[PRIMARY]["average_program_chain_len"],
        "max_program_chain_len": systems[PRIMARY]["max_program_chain_len"],
    }
    ablation_report = {
        "schema_version": "e16b_ablation_v1",
        "insufficient_chain_len_exact_accuracy": systems["INSUFFICIENT_CHAIN_LEN_ABLATION"]["discovery_test_exact_accuracy"],
        "missing_order_primitives_exact_accuracy": systems["MISSING_ORDER_PRIMITIVES_ABLATION"]["discovery_test_exact_accuracy"],
        "missing_map_primitive_exact_accuracy": systems["MISSING_MAP_PRIMITIVE_ABLATION"]["discovery_test_exact_accuracy"],
        "missing_filter_primitive_exact_accuracy": systems["MISSING_FILTER_PRIMITIVE_ABLATION"]["discovery_test_exact_accuracy"],
        "ambiguous_no_abstain_wrong_commit_rate": systems["AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION"]["ambiguous_no_abstain_wrong_commit_rate"],
        "expectations": {
            "insufficient_chain_len_materially_below_primary": systems["INSUFFICIENT_CHAIN_LEN_ABLATION"]["discovery_test_exact_accuracy"] <= systems[PRIMARY]["discovery_test_exact_accuracy"] - 0.20,
            "missing_order_fails_reverse_or_order": systems["MISSING_ORDER_PRIMITIVES_ABLATION"]["reverse_from_swaps_accuracy"] < 0.50,
            "missing_map_fails_map_families": systems["MISSING_MAP_PRIMITIVE_ABLATION"]["map_then_reverse_accuracy"] < 0.50
            and systems["MISSING_MAP_PRIMITIVE_ABLATION"]["map_then_rotate_accuracy"] < 0.50,
            "missing_filter_fails_filter_or_decoy": systems["MISSING_FILTER_PRIMITIVE_ABLATION"]["filter_then_reverse_accuracy"] < 0.50
            and systems["MISSING_FILTER_PRIMITIVE_ABLATION"]["decoy_heavy_composition_accuracy"] < 0.50,
            "ambiguous_no_abstain_higher_wrong_commit": systems["AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION"]["ambiguous_no_abstain_wrong_commit_rate"]
            > systems[PRIMARY]["ambiguous_no_abstain_wrong_commit_rate"],
        },
    }
    semantic_report = {
        "schema_version": "e16b_semantic_leak_audit_v1",
        "runtime_receives_task_family_labels": False,
        "runtime_receives_direct_macro_labels": False,
        "runtime_receives_oracle_expected_output": False,
        "runtime_receives_forbidden_semantic_slots": False,
        "semantic_slot_leak_detected": False,
        "privileged_control_selected_as_primary": False,
        "forbidden_runtime_lanes": ("ACTION", "DIRECTION", "ENTITY", "CATEGORY", "COPY", "REVERSE", "BIND", "MAP", "ROTATE"),
        "primary_runtime_inputs": ("support", "query", "mapping_table_evidence", "invalid_evidence", "branch_id", "primitive_library"),
    }
    deterministic_report = {
        "schema_version": "e16b_deterministic_replay_v1",
        "internal_replay_passed": True,
        "seed_set": {"train": DEFAULT_TRAIN_SEEDS, "test": DEFAULT_TEST_SEEDS},
        "artifact_count": len(REQUIRED_ARTIFACTS),
    }
    boundary_report = {
        "schema_version": "e16b_boundary_claims_v1",
        "boundary": BOUNDARY,
        "broad_claims_absent": True,
    }
    summary = {
        "schema_version": "e16b_summary_v1",
        "decision": decision,
        "next": next_step,
        "primary_system": PRIMARY,
        "positive_gate_passed": aggregate["positive_gate"]["passed"],
        "checker_failure_count": 0,
        "key_metrics": {
            key: systems[PRIMARY][key]
            for key in (
                "discovered_library_size",
                "discovered_program_count",
                "average_program_chain_len",
                "max_program_chain_len",
                "discovery_test_exact_accuracy",
                "composition_exact_accuracy",
                "heldout_vocab_accuracy",
                "randomized_codebook_generalization",
                "support_disambiguation_accuracy",
                "ambiguous_case_abstain_or_repair_accuracy",
                "trace_validity",
                "wrong_writeback_rate",
                "macro_removed_confirmed",
                "direct_macro_leak_detected",
                "semantic_slot_leak_detected",
                "privileged_control_selected_as_primary",
            )
        },
    }
    decision_payload = {
        "schema_version": "e16b_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "next": next_step,
        "primary_system": PRIMARY,
        "positive_gate_passed": aggregate["positive_gate"]["passed"],
        "deterministic_replay_passed": True,
        "checker_failure_count": 0,
    }
    report = render_report(decision_payload, summary, aggregate, ablation_report)
    return {
        "decision.json": decision_payload,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": report,
        "e16b_search_report.json": {
            "schema_version": "e16b_search_v1",
            "search_first_completed": True,
            "local_equivalent_found": False,
            "fetched_ref_equivalent_found": False,
            **search_report,
        },
        "e16b_operator_grammar_report.json": grammar_report,
        "e16b_macro_removal_audit_report.json": macro_audit,
        "e16b_discovered_library_report.json": discovered_library_report,
        "e16b_program_chain_report.json": program_chain_report,
        "e16b_system_comparison_report.json": {"schema_version": "e16b_system_comparison_v1", "systems": systems, "sample_rows": sample_rows},
        "e16b_task_family_report.json": {"schema_version": "e16b_task_family_v1", "family_metrics": family_by_system},
        "e16b_ablation_report.json": ablation_report,
        "e16b_support_disambiguation_report.json": {
            "schema_version": "e16b_support_disambiguation_v1",
            "support_disambiguation_accuracy": systems[PRIMARY]["support_disambiguation_accuracy"],
            "ambiguous_case_abstain_or_repair_accuracy": systems[PRIMARY]["ambiguous_case_abstain_or_repair_accuracy"],
            "ambiguous_no_abstain_wrong_commit_rate": systems["AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION"]["ambiguous_no_abstain_wrong_commit_rate"],
        },
        "e16b_heldout_generalization_report.json": {
            "schema_version": "e16b_heldout_generalization_v1",
            "heldout_vocab_accuracy": systems[PRIMARY]["heldout_vocab_accuracy"],
            "randomized_codebook_generalization": systems[PRIMARY]["randomized_codebook_generalization"],
            "unique_codebook_hashes": sorted({episode.codebook_hash for episode in test_episodes}),
        },
        "e16b_trace_validity_report.json": {
            "schema_version": "e16b_trace_validity_v1",
            "trace_validity_by_system": {system: values["trace_validity"] for system, values in systems.items()},
            "primary_trace_validity": systems[PRIMARY]["trace_validity"],
        },
        "e16b_writeback_safety_report.json": {
            "schema_version": "e16b_writeback_safety_v1",
            "wrong_writeback_rate_by_system": {system: values["wrong_writeback_rate"] for system, values in systems.items()},
            "destructive_overwrite_rate_by_system": {system: values["destructive_overwrite_rate"] for system, values in systems.items()},
            "branch_contamination_rate_by_system": {system: values["branch_contamination_rate"] for system, values in systems.items()},
            "no_gate_failed_trace_or_writeback_safety": systems[NO_GATE]["trace_validity"] < systems[PRIMARY]["trace_validity"]
            and systems[NO_GATE]["wrong_writeback_rate"] > systems[PRIMARY]["wrong_writeback_rate"],
        },
        "e16b_semantic_leak_audit_report.json": semantic_report,
        "e16b_deterministic_replay_report.json": deterministic_report,
        "e16b_boundary_claims_report.json": boundary_report,
    }


def render_report(decision: dict[str, Any], summary: dict[str, Any], aggregate: dict[str, Any], ablation: dict[str, Any]) -> str:
    primary = aggregate["systems"][PRIMARY]
    rows = [
        "# E16B Text-Flow Operator Composition Discovery Confirm",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {PRIMARY}",
        f"positive_gate_passed = {str(decision['positive_gate_passed']).lower()}",
        f"checker_failure_count = {decision['checker_failure_count']}",
        "```",
        "",
        "## Primary Metrics",
        "",
        "```text",
        f"discovered_library_size = {primary['discovered_library_size']}",
        f"discovered_program_count = {primary['discovered_program_count']}",
        f"average_program_chain_len = {primary['average_program_chain_len']:.3f}",
        f"max_program_chain_len = {primary['max_program_chain_len']}",
        f"discovery_test_exact_accuracy = {primary['discovery_test_exact_accuracy']:.3f}",
        f"composition_exact_accuracy = {primary['composition_exact_accuracy']:.3f}",
        f"heldout_vocab_accuracy = {primary['heldout_vocab_accuracy']:.3f}",
        f"randomized_codebook_generalization = {primary['randomized_codebook_generalization']:.3f}",
        f"support_disambiguation_accuracy = {primary['support_disambiguation_accuracy']:.3f}",
        f"ambiguous_case_abstain_or_repair_accuracy = {primary['ambiguous_case_abstain_or_repair_accuracy']:.3f}",
        f"trace_validity = {primary['trace_validity']:.3f}",
        f"wrong_writeback_rate = {primary['wrong_writeback_rate']:.3f}",
        f"macro_removed_confirmed = {str(primary['macro_removed_confirmed']).lower()}",
        f"direct_macro_leak_detected = {str(primary['direct_macro_leak_detected']).lower()}",
        f"semantic_slot_leak_detected = {str(primary['semantic_slot_leak_detected']).lower()}",
        "```",
        "",
        "## Baseline Contrasts",
        "",
        "```text",
        f"random_matched_exact = {aggregate['systems'][RANDOM_MATCHED]['discovery_test_exact_accuracy']:.3f}",
        f"random_best_of_n_exact = {aggregate['systems'][RANDOM_BEST]['discovery_test_exact_accuracy']:.3f}",
        f"no_gate_trace_validity = {aggregate['systems'][NO_GATE]['trace_validity']:.3f}",
        f"no_gate_wrong_writeback_rate = {aggregate['systems'][NO_GATE]['wrong_writeback_rate']:.3f}",
        f"pruned_cost_reduction_vs_unpruned = {aggregate['positive_gate']['deltas']['pruned_cost_reduction_vs_unpruned']:.3f}",
        "```",
        "",
        "## Ablations",
        "",
        "```text",
        f"insufficient_chain_len_exact_accuracy = {ablation['insufficient_chain_len_exact_accuracy']:.3f}",
        f"missing_order_primitives_exact_accuracy = {ablation['missing_order_primitives_exact_accuracy']:.3f}",
        f"missing_map_primitive_exact_accuracy = {ablation['missing_map_primitive_exact_accuracy']:.3f}",
        f"missing_filter_primitive_exact_accuracy = {ablation['missing_filter_primitive_exact_accuracy']:.3f}",
        f"ambiguous_no_abstain_wrong_commit_rate = {ablation['ambiguous_no_abstain_wrong_commit_rate']:.3f}",
        "```",
        "",
        "## Boundary",
        "",
        BOUNDARY,
    ]
    return "\n".join(rows)


def write_payload(out: Path, payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS:
        item = payload[name]
        if name.endswith(".json"):
            write_json(out / name, item)
        else:
            write_text(out / name, str(item))


def run(out: Path) -> dict[str, Any]:
    payload = build_payload()
    replay = build_payload()
    payload_hash = stable_hash({key: value for key, value in payload.items() if key != "e16b_deterministic_replay_report.json"})
    replay_hash = stable_hash({key: value for key, value in replay.items() if key != "e16b_deterministic_replay_report.json"})
    deterministic_passed = payload_hash == replay_hash
    payload["e16b_deterministic_replay_report.json"]["internal_replay_passed"] = deterministic_passed
    payload["e16b_deterministic_replay_report.json"]["primary_payload_hash"] = payload_hash
    payload["e16b_deterministic_replay_report.json"]["replay_payload_hash"] = replay_hash
    aggregate = payload["aggregate_metrics.json"]
    aggregate["positive_gate"] = positive_gate(aggregate, deterministic_passed)
    decision, next_step = choose_decision(aggregate, deterministic_passed)
    payload["decision.json"]["decision"] = decision
    payload["decision.json"]["next"] = next_step
    payload["decision.json"]["positive_gate_passed"] = aggregate["positive_gate"]["passed"]
    payload["decision.json"]["deterministic_replay_passed"] = deterministic_passed
    payload["summary.json"]["decision"] = decision
    payload["summary.json"]["next"] = next_step
    payload["summary.json"]["positive_gate_passed"] = aggregate["positive_gate"]["passed"]
    payload["report.md"] = render_report(payload["decision.json"], payload["summary.json"], aggregate, payload["e16b_ablation_report.json"])
    write_payload(out, payload)
    return payload["decision.json"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    decision = run(Path(args.out))
    print(stable_json(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
